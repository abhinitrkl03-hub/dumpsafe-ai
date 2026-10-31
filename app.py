"""
DumpSafe AI - sim-to-real factor-of-safety platform for OB dump slopes
Run:  streamlit run app.py
"""
from __future__ import annotations
import io, os, datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as px_colors
import streamlit as st

from core.lem import DumpGeometry, bishop_fos, slip_arc, bench_analysis, bench_points
from core.slide2 import (Slide2Settings, search as slide2_search, bench_by_bench, bench_limits,
                         default_limits, Slide2Model, METHOD_NAMES)
from core.models import FOSModel, FEATURES, LABELS, METHOD_LABELS, method_class, engineer, metrics
from core.moisture import moisture_state, saturation_moisture
from core.montecarlo import sample_inputs, run_mc, reliability, exact_check, rank_sensitivity
from core.monitoring import (process_series, inverse_velocity_forecast, classify_velocity,
                             demo_series, demo_storm, VELOCITY_TARP, LEVEL_ORDER, add_moving_averages,
                             onset_of_acceleration, iv_window, calibrate_ru)
from core.tarp import (fos_trigger, rain_trigger, accel_trigger, visual_trigger, VISUAL_SIGNS, VEL_MAP,
                       ACTION_TABLE, rolling_sum, antecedent_index, ru_from_piezometer, level_history,
                       first_time, evaluate as tarp_evaluate, COL as TCOL)
TRIG_AUTO = ["T2 Live FOS", "T3 Rainfall", "T4 Velocity", "T5 Acceleration"]
from core.design import design_envelope, back_analyse
from core.dgms import check as dgms_check, min_berm, MAX_DECK_ANGLE, MAX_OVERALL

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

st.set_page_config(page_title="DumpSafe AI - dump slope FOS", page_icon="⛰️", layout="wide")

# full-width argument that works on old and new Streamlit versions
_v = tuple(int(x) for x in getattr(st, "__version__", "1.40.0").split(".")[:2])
FULL = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

# ------------------------------------------------------------------ styling
OB = "#A8977A"        # overburden fill
OB_LINE = "#6E5E45"
SLIP = "#B03A2E"
INK = "#1C2B2D"
TEAL = "#2F6B6A"
TARP_COL = {"Green": "#2e7d4f", "Yellow": "#d8a31a", "Orange": "#d9731f", "Red": "#b03a2e"}
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText, p, li, label {font-family: 'IBM Plex Sans', system-ui, sans-serif;}
  h1 {font-family:'IBM Plex Sans'; letter-spacing:-0.6px; font-weight:700; color:#1C2B2D;
      border-left:6px solid #2F6B6A; padding-left:14px; margin-bottom:0.2rem;}
  h2, h3 {font-family:'IBM Plex Sans'; letter-spacing:-0.2px; color:#1C2B2D;}
  div[data-testid="stMetric"] {background:#ffffff; border:1px solid #d9e0de; border-radius:10px;
      padding:12px 16px; box-shadow:0 1px 2px rgba(28,43,45,0.05);}
  div[data-testid="stMetricValue"] {font-family:'IBM Plex Mono', monospace; font-variant-numeric:tabular-nums;}
  div[data-testid="stMetricLabel"] p {font-size:0.82rem; color:#4a5a5c;}
  .tarp {border-radius:10px; padding:14px 18px; color:white; margin:6px 0 12px 0; line-height:1.45;}
  .tarp b {font-size:1.25rem; letter-spacing:0.3px;}
  .src {font-size:0.82rem; color:#4a5a5c;}
  .badge {display:inline-block; border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600;
          margin-right:6px;}
  .ok {background:#dcefe4; color:#1f5c3a;} .warn {background:#fbeccc; color:#7a5510;}
  .bad {background:#f6d9d5; color:#7b1d1d;} .info {background:#dde9f3; color:#23496b;}
  section[data-testid="stSidebar"] {background:#e7ecea;}
  .lede {color:#3c4b4d; font-size:1.02rem; margin-top:-0.3rem;}
</style>""", unsafe_allow_html=True)


def badge(text, kind="info"):
    return f"<span class='badge {kind}'>{text}</span>"


def validated(text, kind="ok"):
    """Small line under a result saying how it was checked (see the Validation page)."""
    st.markdown(badge("validated" if kind == "ok" else "assumption" if kind == "warn" else "note", kind)
                + f"<span class='src'>{text}</span>", unsafe_allow_html=True)


PLOT = dict(template="simple_white", margin=dict(l=10, r=10, t=48, b=10),
            font=dict(color=INK, family="IBM Plex Sans, sans-serif", size=13),
            title_font=dict(size=15), height=390)


# ------------------------------------------------------------------ data
@st.cache_data
def load_default_real():
    return pd.read_csv(os.path.join(DATA, "real_cases.csv"))


@st.cache_data
def load_physics_all():
    return pd.read_csv(os.path.join(DATA, "physics_dataset.csv"))


def load_physics():
    ph = load_physics_all()
    return ph if st.session_state.get("use_noncompliant") else ph[ph.dgms_compliant]


REQ = ["c_kPa", "phi_deg", "gamma_kNm3", "H_m", "FOS"]


def training_rows(real: pd.DataFrame) -> pd.DataFrame:
    r = real.copy()
    if "use_for_training" in r:
        r = r[r["use_for_training"].astype(str).str.lower().isin(["true", "1", "yes"])]
    # derive overall angle from deck geometry where missing
    need = r["beta_overall_deg"].isna() & r["deck_angle_deg"].notna() & r["n_decks"].notna()
    for i in r[need].index:
        bw = r.at[i, "berm_width_m"]
        g = DumpGeometry(int(r.at[i, "n_decks"]), float(r.at[i, "deck_height_m"]),
                         float(r.at[i, "deck_angle_deg"]), 0.0 if pd.isna(bw) else float(bw))
        r.at[i, "beta_overall_deg"] = g.overall_angle
    return r.dropna(subset=REQ + ["beta_overall_deg"])


@st.cache_resource(show_spinner="Training the sim-to-real model ...")
def train_model(real_csv: str, alpha: float, noncompliant: bool):
    real = pd.read_csv(io.StringIO(real_csv))
    ph = load_physics_all()
    tr = training_rows(real)
    if not noncompliant:
        ph = ph[ph.dgms_compliant]
        if "dgms_compliant" in tr:
            tr = tr[~tr["dgms_compliant"].astype(str).str.startswith("no")]
    return FOSModel().fit(ph, tr, alpha=alpha)


if "real" not in st.session_state:
    st.session_state.real = load_default_real()
if "last" not in st.session_state:
    st.session_state.last = {}

# ------------------------------------------------------------------ sidebar
st.sidebar.title("DumpSafe AI")
st.sidebar.caption("Sim-to-real FOS prediction for overburden dumps")
PAGE = st.sidebar.radio("Go to", [
    "Overview", "Data and sources", "Methods and equations", "Validation", "Model lab",
    "Predict and TARP", "Monte Carlo reliability", "Design envelope", "Multi-parameter TARP"])

# Minimum FoS: DGMS circular on the scientific study under Reg. 106(2) of CMR 2017, para 3.2:
# not less than 1.50 for permanent or long-standing slopes and 1.30 for all other cases.
# DGMS gives no PoF limit; the PoF values are those of Read & Stacey (2009), Table 9.9.
DGMS_TABLE = {"Permanent / long-standing dump": (1.50, 0.05), "Other (working / temporary) dump": (1.30, 0.10)}
with st.sidebar.expander("Acceptance criteria (TARP)", expanded=False):
    cons = st.selectbox("Type of dump slope", list(DGMS_TABLE),
                        help="DGMS circular for the scientific study under Reg. 106(2) of CMR 2017, para 3.2: "
                             "minimum FoS 1.50 for permanent or long-standing slopes and 1.30 for all other "
                             "cases. Maximum PoF (not set by DGMS) from Read & Stacey (2009), Table 9.9.")
    fs_def, pf_def = DGMS_TABLE[cons]
    FOS_MIN = st.number_input("Minimum acceptable FOS", 1.0, 2.0, fs_def, 0.05)
    POF_MAX = st.number_input("Maximum acceptable PoF (%)", 0.5, 50.0, pf_def * 100, 0.5) / 100
    st.caption("Use the values in your mine's approved scientific study / DGMS permission if they differ.")
alpha = 0.90   # coverage of the prediction interval (fixed; split-conformal, see Methods page)
st.session_state.use_noncompliant = st.sidebar.toggle(
    "DGMS permission for steeper / higher dumps", value=False,
    help="Reg. 106 of CMR 2017 lets a spoil bank exceed 37.5 deg, 30 m benches or 1V:1.5H only when a "
         "scientific study recommends it and the Regional Inspector permits it by order. Switch on if your "
         "mine holds such an order (or to study failed non-compliant dumps). The model then also "
         "trains on 2,850 non-compliant designs and the 14 WCL cases, and designs beyond Reg. 106 are shown "
         "as 'needs DGMS order' instead of 'not allowed'.")
model: FOSModel = train_model(st.session_state.real.to_csv(index=False), 1 - alpha,
                              st.session_state.use_noncompliant)
st.sidebar.caption(f"Model trained on {len(load_physics()):,} physics rows + "
                   f"{model.n_real} real rows" + ("" if st.session_state.use_noncompliant
                                                  else " (DGMS Reg. 106 designs only)"))

# Minimum FoS from DGMS (para 3.2); stable / marginal / unstable classes as in Read & Stacey (2009)
# ch. 9, plus a failure class (FOS < 1).
ACTIONS = {
    "Green": "Stable: meets the acceptance criteria. Normal operations and routine monitoring.",
    "Yellow": "Marginal: fails one criterion (or the uncertainty band crosses it). Minor geometry changes, "
              "closer monitoring, check drainage and cracks.",
    "Orange": "Unstable design: fails the acceptance criteria. Major geometry changes (flatten, lower, wider "
              "benches), dewatering, restrict access below the toe.",
    "Red": "Failure predicted (FOS < 1). Stop dumping, evacuate the influence zone, unload or redesign.",
}
ORDER = ["Green", "Yellow", "Orange", "Red"]


def det_level(fos, fos_lo=None):
    """Deterministic class from FOS and the lower end of its prediction interval."""
    if fos < 1.0: return "Red"
    if fos < FOS_MIN: return "Orange"
    if fos_lo is not None and fos_lo < FOS_MIN: return "Yellow"
    return "Green"


def prob_level(mean_fos, pof):
    """Probabilistic class (Read & Stacey): both criteria met = stable, one = marginal, none = unstable."""
    if mean_fos < 1.0: return "Red"
    met = int(bool(mean_fos >= FOS_MIN)) + int(bool(pof <= POF_MAX))
    return {2: "Green", 1: "Yellow", 0: "Orange"}[int(met)]


def fos_level(f):          # single value (e.g. one Monte Carlo realisation)
    return det_level(f)


def vel_to_tarp(level):
    return {"Normal": "Green", "Watch": "Green", "Caution": "Yellow", "Alert": "Orange",
            "Alarm": "Red", "Stop work": "Red"}[level]


def worst(*levels):
    return max(levels, key=ORDER.index)


def tarp_box(level, title, detail=""):
    st.markdown(f"<div class='tarp' style='background:{TARP_COL[level]}'><b>{level}</b> "
                f"&nbsp;{title}<br>{ACTIONS[level]}{('<br><i>' + detail + '</i>') if detail else ''}"
                f"</div>", unsafe_allow_html=True)


def _arc_xy(xc, yc, R, geom, x0=None, x1=None, n=160):
    """Lower arc of a circle between its exit and entry on the slope (for drawing)."""
    mdl = Slide2Model(geom)
    if x0 is None:
        e, en, _ = mdl.intersections(np.array([xc]), np.array([yc]), np.array([R]))
        x0, x1 = e[0], en[0]
    X = np.linspace(x0, x1, n)
    return X, yc - np.sqrt(np.clip(R ** 2 - (X - xc) ** 2, 0, None))


TIE_TOL = 0.001   # benches within 0.1 % of the minimum FOS are equally critical


def profile_fig(geom: DumpGeometry, res=None, title="", fos=None, material=None, benches=None,
                surfaces=0, limits=None, show_ties=False):
    """Section drawn like a Slide2 model: external boundary from the toe (0, 0), base y = 0, 1:1 scale.
    Nothing is written on the dump. As in Slide2: the global minimum surface is bold, with its centre,
    the two radius lines and the FOS label above the slope; optionally the lowest `surfaces` valid
    surfaces are drawn thin and colour-coded by FOS ("display all surfaces"); slope limits are marked
    on the slope surface. If `benches` (one result per bench) is given, every bench's critical circle is
    drawn dotted and the most critical bench is the global minimum."""
    xs, ys = geom.surface()
    L, H = geom.horizontal_extent, geom.height
    mdl = Slide2Model(geom)
    xr = mdl.x_right
    px = np.r_[mdl.sx, xr, 0.0]
    py = np.r_[mdl.sy, 0.0, 0.0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px, y=py, fill="toself", fillcolor=OB, line=dict(color=OB_LINE, width=2),
                             name="Dump", hoverinfo="skip"))
    tied = []
    if benches:
        res = min(benches, key=lambda r_: r_.fos)
        tied = [r_ for r_ in benches if np.isfinite(r_.fos) and r_.fos <= res.fos * (1 + TIE_TOL)]
        if not show_ties:          # Slide2 style: one global minimum, other benches dotted
            tied = [res]
    if surfaces and res is not None and hasattr(res, "lowest") and res.all_fos.size:
        XC, YC, RR, FF = res.lowest(int(surfaces))
        lo_f, hi_f = float(FF.min()), float(max(FF.max(), FF.min() + 1e-6))
        for j in np.argsort(FF)[::-1]:
            ax, ay = _arc_xy(XC[j], YC[j], RR[j], geom)
            t = (FF[j] - lo_f) / (hi_f - lo_f)
            col = px_colors.sample_colorscale("Turbo", [0.15 + 0.8 * t])[0]
            fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=col, width=0.8),
                                     showlegend=False, hovertemplate=f"FOS {FF[j]:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Lowest {len(FF)} surfaces "
                                 f"(FOS {lo_f:.3f} - {hi_f:.3f}, blue = lowest)",
                                 marker=dict(color=FF, colorscale="Turbo", cmin=lo_f, cmax=hi_f, size=6)))
    if benches:
        for k, r_ in enumerate(benches):
            if r_ in tied or not np.isfinite(r_.fos):
                continue
            ax, ay = slip_arc(r_, geom)
            fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color="#3d3426", width=1.5, dash="dot"),
                                     name=f"Bench {k + 1} circle (FOS {r_.fos:.3f})", hoverinfo="name"))
        # benches equally critical with the global minimum: all drawn like the global minimum
        for r_ in tied:
            if r_ is res:
                continue
            ax, ay = slip_arc(r_, geom)
            fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=SLIP, width=3.5), showlegend=False,
                                     hovertemplate=f"Bench {benches.index(r_) + 1}: FOS {r_.fos:.3f}<extra></extra>"))
            if len(ax):
                fig.add_trace(go.Scatter(x=[ax[0], r_.xc, ax[-1]], y=[ay[0], r_.yc, ay[-1]], mode="lines",
                                         line=dict(color=SLIP, width=1), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[r_.xc], y=[r_.yc], mode="markers+text", text=[f"{r_.fos:.3f}"],
                                     textposition="top center", marker=dict(color=SLIP, size=7, symbol="x"),
                                     showlegend=False, textfont=dict(color=SLIP, size=13), hoverinfo="skip"))
    for lim in ([limits] if limits else []):
        pts = sorted(set([lim[0][0], lim[0][1], lim[1][0], lim[1][1]]))
        fig.add_trace(go.Scatter(x=pts, y=mdl.yg(np.array(pts)), mode="markers", name="Slope limits",
                                 marker=dict(symbol="triangle-down", size=11, color="#1f4e79"),
                                 hovertemplate="limit x = %{x:.1f} m<extra></extra>"))
    y_top = H
    if res is not None and np.isfinite(res.fos):
        ax, ay = slip_arc(res, geom)
        lab = "Global minimum slip circle"
        if benches and len(tied) > 1:
            lab = "Equally critical benches " + ", ".join(str(benches.index(r_) + 1) for r_ in tied)
        elif benches:
            lab += f" - bench {benches.index(res) + 1}"
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=SLIP, width=3.5),
                                 name=f"{lab} (FOS {res.fos:.3f})"))
        if len(ax):
            fig.add_trace(go.Scatter(x=[ax[0], res.xc, ax[-1]], y=[ay[0], res.yc, ay[-1]], mode="lines",
                                     line=dict(color=SLIP, width=1), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[res.xc], y=[res.yc], mode="markers+text", text=[f"{res.fos:.3f}"],
                                 textposition="top center", marker=dict(color=SLIP, size=7, symbol="x"),
                                 name="Slip centre", textfont=dict(color=SLIP, size=13)))
        y_top = max(H, res.yc + 0.08 * H)
    pad = 0.08 * max(xr, y_top)
    fig.update_layout(**PLOT, title=title, showlegend=True, legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(title="Distance from toe (m)", range=[-pad, xr + pad], constrain="domain")
    fig.update_yaxes(title="Elevation above dump base (m)", range=[-pad, y_top + pad],
                     scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


def bench_table(geom: DumpGeometry, benches):
    """Bench-by-bench check: critical FOS of each bench, ranked, with circle coordinates
    (so the same circle can be entered in Slide2 as a single surface)."""
    rows = []
    worst = min(r_.fos for r_ in benches)
    for k, ((xt, yt, xcr, ycr, h), r_) in enumerate(zip(bench_points(geom), benches)):
        ax, ay = slip_arc(r_, geom)
        rows.append({"Bench": f"{k + 1}" + (" (bottom)" if k == 0 else " (top)" if k == len(benches) - 1 else ""),
                     "Elevation (m)": f"{yt:.0f} - {ycr:.0f}",
                     "Critical FOS": round(r_.fos, 3),
                     "Status": ("most critical" if abs(r_.fos - worst) < 1e-9 else
                                "equally critical (within 0.1 %)" if r_.fos <= worst * (1 + TIE_TOL) else
                                f"+{(r_.fos - worst) / worst * 100:.1f} % above most critical"),
                     "Meets minimum": "yes" if r_.fos >= FOS_MIN else "NO",
                     "Centre x, y (m)": f"{r_.xc:.1f}, {r_.yc:.1f}",
                     "Radius (m)": round(r_.radius, 1),
                     "Exit x, y (m)": f"{ax[0]:.1f}, {ay[0]:.1f}" if len(ax) else "-",
                     "Entry x, y (m)": f"{ax[-1]:.1f}, {ay[-1]:.1f}" if len(ax) else "-",
                     "Surfaces (valid / generated)": f"{getattr(r_, 'n_valid', 0):,} / {r_.n_circles:,}"})
    return pd.DataFrame(rows)


def slide2_settings_table(stg: Slide2Settings, scope: str, n_gen: int, n_valid: int):
    """Project-settings summary in the order a Slide2 user sets them up."""
    meth = METHOD_NAMES[stg.method]
    detail = {"auto_refine": f"{stg.divisions} divisions along slope, {stg.circles_per_division} circles per "
                             f"division, {stg.iterations} iterations, {int(stg.keep_fraction * 100)} % of divisions "
                             "kept for the next iteration",
              "grid": f"{stg.grid_intervals[0]} x {stg.grid_intervals[1]} grid intervals, radius increment "
                      f"{stg.radius_increment}",
              "slope": f"{stg.n_surfaces:,} random surfaces"}[stg.method]
    return pd.DataFrame([
        ["1 Model", "External boundary: dump section from the toe (0, 0), base y = 0, right boundary behind the crest"],
        ["2 Materials", "Mohr-Coulomb (c, phi, unit weight); water by r_u"],
        ["3 Analysis method", f"Bishop simplified, {stg.n_slices} slices, tolerance {stg.tolerance}, "
                              f"max {stg.max_iterations} iterations"],
        ["4 Surface type / search", f"Circular, {meth}: {detail}"],
        ["5 Slope limits", scope],
        ["6 Surface checks", "rejected if < 2 slope intersections, below the dump base, outside the slope limits, "
                             "or not converged"],
        ["7 Result", f"{n_valid:,} valid of {n_gen:,} surfaces generated; global minimum shown in red"],
    ], columns=["Step (as in Slide2)", "Setting used"])


def show_bench_check(geom, benches):
    st.markdown("**Bench-by-bench check** (lowest-FOS circle exiting on each bench; circles may run back "
                "through the benches above)")
    st.dataframe(bench_table(geom, benches), hide_index=True, **FULL)
    worst = min(r_.fos for r_ in benches)
    tied = [k + 1 for k, r_ in enumerate(benches) if r_.fos <= worst * (1 + TIE_TOL)]
    if len(tied) == len(benches) and len(tied) > 1:
        st.caption(f"The red circle is the global minimum (bench {benches.index(min(benches, key=lambda r_: r_.fos)) + 1}), "
                   f"as Slide2 shows it. Note that all {len(tied)} benches have the same critical FOS ({worst:.3f}, "
                   "within 0.1 %): the benches are identical and wide enough to act as separate slopes, so which one "
                   "is the 'minimum' is decided by the third or fourth decimal place. Apply the same precautions to "
                   "every bench.")
    elif len(tied) > 1:
        st.caption(f"The red circle is the global minimum, as in Slide2. Benches {', '.join(map(str, tied))} are "
                   f"equally critical (FOS {worst:.3f}, within 0.1 %): treat them alike when planning strengthening "
                   "or monitoring.")
    else:
        st.caption(f"Bench {tied[0]} is the most critical: prioritise it for strengthening, drainage and "
                   "monitoring prisms.")


def search_inputs(key, geom):
    """Slip-surface settings, set up in the same order as in Slide2."""
    mdl = Slide2Model(geom)
    c1, c2, c3 = st.columns(3)
    meth = c1.selectbox("Search method", list(METHOD_NAMES), format_func=METHOD_NAMES.get, key=f"{key}sm",
                        help="Same three circular search methods as Slide2. Auto refine search is the "
                             "Slide2 default and the most reliable for benched dumps.")
    scope = c2.selectbox("Slope limits", ["Bench by bench", "Whole slope", "Custom limits"], key=f"{key}sl",
                         help="Bench by bench = one search per bench with the slope limits moved onto that "
                              "bench (exit on that bench). Whole slope = Slide2 default limits.")
    shown = c3.number_input("Lowest surfaces to display", 0, 300, 0, 10, key=f"{key}ns",
                            help="0 = global minimum only. Otherwise the lowest surfaces are drawn colour-coded "
                                 "by FOS, like 'display all surfaces' in Slide2.")
    stg = Slide2Settings(method=meth)
    st.session_state.show_ties = st.toggle("Also highlight benches that are equally critical (within 0.1 %)",
                                           value=False, key=f"{key}tie",
                                           help="Off = one global minimum circle, as Slide2 displays it.")
    lim = None
    if scope == "Custom limits":
        a1, a2, a3, a4 = st.columns(4)
        e0 = a1.number_input("Exit from x (m)", 0.0, float(mdl.x_right), 0.0, key=f"{key}e0")
        e1 = a2.number_input("Exit to x (m)", 0.0, float(mdl.x_right), float(geom.horizontal_extent), key=f"{key}e1")
        n0 = a3.number_input("Entry from x (m)", 0.0, float(mdl.x_right), 0.0, key=f"{key}n0")
        n1 = a4.number_input("Entry to x (m)", 0.0, float(mdl.x_right), float(mdl.x_right), key=f"{key}n1")
        lim = ((min(e0, e1), max(e0, e1)), (min(n0, n1), max(n0, n1)))
    with st.expander("Analysis and search settings (Slide2 defaults)"):
        b1, b2, b3 = st.columns(3)
        stg.n_slices = int(b1.number_input("Number of slices", 10, 200, 50, key=f"{key}sn"))
        stg.tolerance = float(b2.number_input("Tolerance", 0.0001, 0.05, 0.005, 0.0005, format="%.4f", key=f"{key}tol"))
        stg.max_iterations = int(b3.number_input("Maximum iterations", 10, 500, 75, key=f"{key}it"))
        if meth == "auto_refine":
            d1, d2, d3, d4 = st.columns(4)
            stg.divisions = int(d1.number_input("Divisions along slope", 5, 60, 20, key=f"{key}dv"))
            stg.circles_per_division = int(d2.number_input("Circles per division", 2, 40, 10, key=f"{key}cd"))
            stg.iterations = int(d3.number_input("Number of iterations", 1, 30, 10, key=f"{key}ni"))
            stg.keep_fraction = d4.number_input("Divisions kept for next iteration (%)", 10, 90, 50, key=f"{key}kf") / 100
        elif meth == "grid":
            d1, d2, d3 = st.columns(3)
            stg.grid_intervals = (int(d1.number_input("Grid intervals in x", 5, 60, 20, key=f"{key}gx")),
                                  int(d2.number_input("Grid intervals in y", 5, 60, 20, key=f"{key}gy")))
            stg.radius_increment = int(d3.number_input("Radius increment", 2, 50, 10, key=f"{key}ri"))
        else:
            stg.n_surfaces = int(st.number_input("Number of surfaces", 500, 50000, 5000, 500, key=f"{key}nsf"))
        stg.min_depth = float(st.number_input("Minimum slip depth filter (m, 0 = off)", 0.0, 30.0, 0.0, key=f"{key}md"))
    return stg, scope, lim, int(shown)


def run_search(geom, c, phi, gamma, ru, stg, scope, lim):
    """Returns (global-minimum result, per-bench results or None, limits drawn, scope text)."""
    if scope == "Bench by bench":
        benches = bench_by_bench(geom, c, phi, gamma, ru, stg)
        res = min(benches, key=lambda r_: r_.fos)
        txt = ("Bench by bench: for each bench the exit must lie on that bench (its face or the half bench width "
               "in front of its toe); the entry may lie anywhere above, so multi-bench circles are included")
        return res, benches, res.limits, txt
    lim = lim or default_limits(geom)
    res = slide2_search(geom, c, phi, gamma, ru, stg, lim)
    txt = (f"exit between x = {lim[0][0]:.1f} and {lim[0][1]:.1f} m, entry between x = {lim[1][0]:.1f} and "
           f"{lim[1][1]:.1f} m" + (" (Slide2 default: whole slope)" if scope == "Whole slope" else ""))
    return res, None, lim, txt


def show_search_result(geom, c, phi, gamma, ru, stg, scope, lim, shown, title_prefix=""):
    with st.spinner("Generating and checking slip surfaces (Slide2 procedure) ..."):
        res, benches, lim_d, txt = run_search(geom, c, phi, gamma, ru, stg, scope, lim)
    if not np.isfinite(res.fos):
        st.error("No valid slip surface was found inside these slope limits. Widen the limits.")
        return None
    allr = benches or [res]
    n_gen = sum(r_.n_circles for r_ in allr); n_val = sum(r_.n_valid for r_ in allr)
    ties = st.session_state.get("show_ties", False)
    st.plotly_chart(profile_fig(geom, res, title=f"{title_prefix}Bishop simplified, {METHOD_NAMES[stg.method]}: "
                                f"global minimum FOS {res.fos:.3f}", benches=benches, surfaces=shown,
                                limits=None if benches else lim_d, show_ties=ties), **FULL)
    if getattr(res, "edge_warning", False):
        st.warning("The minimum lies on the edge of the search grid (Slide2 gives the same warning). The grid was "
                   "extended once; try Auto refine search to confirm.")
    if benches:
        show_bench_check(geom, benches)
    st.dataframe(slide2_settings_table(stg, txt, n_gen, n_val), hide_index=True, **FULL)
    return res, benches


def geometry_inputs(key, default=(3, 30.0, 32.0, 30.0)):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    nd = c1.number_input("Number of decks", 1, 8, default[0], key=f"{key}nd")
    dh = c2.number_input("Deck height (m)", 5.0, 150.0, default[1], key=f"{key}dh",
                         help="Reg. 106: at most 30 m when the dump is higher than 30 m. Larger values are "
                              "allowed here only to study unbenched or permitted dumps.")
    top = c3.number_input("Top deck height (m)", 0.0, 150.0, 0.0, key=f"{key}top", disabled=nd == 1,
                          help="0 = same as the other decks. Mines usually build full 30 m decks "
                               "and put the remainder on top (e.g. 30 + 30 + 20 m).")
    da = c4.number_input("Deck angle (deg)", 10.0, 60.0, default[2], key=f"{key}da")
    bw = c5.number_input("Bench width (m)", 0.0, 150.0, default[3], key=f"{key}bw",
                         disabled=nd == 1)
    cl = c6.number_input("Crest length behind top (m)", 0.0, 300.0, 0.0, key=f"{key}cl",
                         help="Flat ground drawn behind the top crest, up to the model's right boundary (as in "
                              "the Slide2 model). 0 = same as the bench width (30 m for a single bench).")
    g = DumpGeometry(int(nd), dh, da, bw if nd > 1 else 0.0,
                     top if (nd > 1 and top > 0) else None, cl if cl > 0 else None)
    ok, viol = dgms_check(g.n_decks, g.deck_height, g.deck_angle, g.berm_width, g.overall_angle,
                          H=g.height)
    if ok:
        st.success(f"DGMS / CMR 2017 Reg. 106 compliant: benches {g.deck_height:.0f} m, deck "
                   f"{g.deck_angle:.1f} deg, overall {g.overall_angle:.1f} deg (limit 33.7).")
    else:
        need = min_berm(g.n_decks, g.deck_height, g.deck_angle, H=g.height)
        tip = f" A bench width of at least {need:.0f} m would meet the overall-slope limit." if g.n_decks > 1 else ""
        if st.session_state.get("use_noncompliant"):
            st.info("Beyond CMR 2017 Reg. 106 limits (" + "; ".join(viol) + "). Acceptable only under a DGMS "
                    "order based on a scientific study - check that the order covers this geometry." + tip)
        else:
            st.warning("Not DGMS compliant: " + "; ".join(viol) + "." + tip +
                       " If the mine holds a DGMS permission order, switch it on in the sidebar.")
    return g


def material_inputs(key, default=(44.0, 30.0, 18.63)):
    real = training_rows(st.session_state.real)
    opts = ["Custom"] + [f"{r.mine} (c={r.c_kPa}, phi={r.phi_deg}, g={r.gamma_kNm3})"
                         for r in real.itertuples()]
    pick = st.selectbox("Load material from a real case", opts, key=f"{key}pick")
    if pick != "Custom":
        r = real.iloc[opts.index(pick) - 1]
        default = (float(r.c_kPa), float(r.phi_deg), float(r.gamma_kNm3))
    c1, c2, c3 = st.columns(3)
    c = c1.number_input("Cohesion c (kPa)", 0.0, 500.0, default[0], key=f"{key}c{pick}")
    phi = c2.number_input("Friction angle phi (deg)", 1.0, 60.0, default[1], key=f"{key}p{pick}")
    g = c3.number_input("Unit weight (kN/m3)", 10.0, 35.0, default[2], key=f"{key}g{pick}",
                        help="Bulk unit weight. With the moisture channel ON this becomes the "
                             "DRY unit weight and bulk weight is computed from moisture.")
    return c, phi, g


def moisture_inputs(key):
    """Simple moisture / water inputs (no charts)."""
    st.markdown("Enter the water condition of the dump. Leave moisture off for a dry analysis.")
    on = st.toggle("Include moisture content", value=False, key=f"{key}mon",
                   help="Moisture makes the dump heavier and, once the voids are nearly full of water, "
                        "creates pore pressure. Standard phase relations; see the Methods page.")
    cfg = None; w = None
    c1, c2 = st.columns(2)
    if on:
        w = c1.number_input("Moisture content w (%)", 0.0, 40.0, 10.0, 0.5, key=f"{key}w")
        cfg = dict(Gs=c2.number_input("Specific gravity of solids Gs", 2.2, 2.9, 2.60, 0.01, key=f"{key}gs"),
                   S_crit=0.80, max_phreatic_ratio=1.0, w_ref=8.0, kc=0.0, kphi=0.0)
        with st.expander("Advanced moisture settings (assumptions)"):
            a1, a2 = st.columns(2)
            cfg["S_crit"] = a1.number_input("Saturation at which pore pressure starts", 0.5, 1.0, 0.80, 0.05,
                                            key=f"{key}sc")
            cfg["w_ref"] = a2.number_input("Reference moisture / OMC (%)", 0.0, 30.0, 8.0, key=f"{key}wr")
            a3, a4 = st.columns(2)
            cfg["kc"] = a3.number_input("Cohesion loss per 1 % above reference (fraction)", 0.0, 0.2, 0.0, 0.005,
                                        key=f"{key}kc", help="0 = off. Set from your own direct-shear tests.")
            cfg["kphi"] = a4.number_input("Friction loss per 1 % above reference (deg)", 0.0, 2.0, 0.0, 0.05,
                                          key=f"{key}kp", help="0 = off. Set from your own direct-shear tests.")
    ru_ext = st.number_input("Pore-pressure ratio r_u from piezometers / water table (0 = dry)", 0.0, 0.6, 0.0,
                             0.01, key=f"{key}ru")
    return on, w, cfg, ru_ext


def effective(c, phi, g, on, w, cfg, ru_ext):
    if not on:
        return c, phi, g, ru_ext, None
    ce, pe, ge, ru, S = moisture_state(c, phi, g, w, **cfg, ru_external=ru_ext)
    return float(ce), float(pe), float(ge), float(ru), float(S)


def row(c, phi, g, ru, geom):
    return pd.DataFrame([dict(c_kPa=c, phi_deg=phi, gamma_kNm3=g, r_u=ru, **geom.cols())])


# =================================================================== pages
if PAGE == "Overview":
    st.title("How safe is this dump, and how sure are we?")
    st.markdown("<p class='lede'>Factor of safety of coal-mine overburden dumps from a validated physics engine, "
                "calibrated on real studies, with uncertainty, DGMS checks and real-time alerts.</p>",
                unsafe_allow_html=True)
    geom = DumpGeometry(3, 30, 32, 30)
    ov_benches = bench_by_bench(geom, 44, 30, 18.63)
    res = min(ov_benches, key=lambda r_: r_.fos)
    pred = model.predict(row(44, 30, 18.63, 0, geom)).iloc[0]
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(profile_fig(geom, title="Gevra OCP dump section (SECL): 3 benches x 30 m",
                                    benches=ov_benches), **FULL)
        show_bench_check(geom, ov_benches)
        st.caption("Slip circles found with the Slide2 procedure: Bishop simplified, 50 slices, tolerance 0.005, "
                   "auto refine search (20 divisions, 10 circles per division, 10 iterations), slope limits moved "
                   "onto each bench in turn. Slide2 on the same section: 1.883 (bench 2).")
        st.markdown("**Input parameters of this section**")
        st.dataframe(pd.DataFrame([
            ["Unit weight, gamma", "18.63 kN/m3", "NIT Rourkela SECL study (2025)"],
            ["Cohesion, c", "44 kPa", "NIT Rourkela SECL study (2025)"],
            ["Friction angle, phi", "30 deg", "NIT Rourkela SECL study (2025)"],
            ["Benches", "3 x 30 m (total 90 m)", "NIT Rourkela SECL study (2025)"],
            ["Bench face angle", "32 deg", "NIT Rourkela SECL study (2025)"],
            ["Bench width", "30 m", "NIT Rourkela SECL study (2025) - 'width = 30 m'"],
            ["Overall slope angle", f"{geom.overall_angle:.1f} deg (report: 24 deg)", "computed from the above"],
            ["Water", "dry (r_u = 0)", "assumed - not stated in the notes"],
            ["Foundation", "competent; slip may not pass below the dump base", "modelling assumption (Slide2 default)"],
        ], columns=["Parameter", "Value", "Source"]), hide_index=True, **FULL)
    with right:
        st.metric("Reported FOS (RS2, NIT Rourkela study)", "1.940")
        st.metric("In-house Bishop solver", f"{res.fos:.3f}")
        st.metric("Sim-to-real ML prediction", f"{pred.FOS:.3f}",
                  help=f"{int(alpha*100)}% interval {pred.FOS_lo:.2f} - {pred.FOS_hi:.2f}")
        st.caption(f"{int(alpha*100)}% prediction interval: {pred.FOS_lo:.2f} to {pred.FOS_hi:.2f}")
    st.subheader("What this platform does differently")
    st.markdown("""
- **Coal mines only, DGMS first.** Default simulations follow CMR 2017 Reg. 106 (benches up to 30 m, decks up
  to 37.5 deg, overall slope within 1V:1.5H). 20 Reg. 106-compliant real coal cases (8 SECL in RS2, 12
  Jagannathpur in FLAC) calibrate the model. Mines holding a DGMS permission order for steeper or higher
  dumps can switch it on in the sidebar; the WCL failed dump is then included. 29 more cited coal cases are
  waiting for missing values.
- **Every result is checked.** The Validation page compares each part of the model with an independent
  reference (published results, closed-form solutions, the exact solver, a real failure); the Methods page
  gives every equation and source.
- **Acceptance criteria from DGMS.** The TARP minimum FOS follows para 3.2 of the DGMS circular for the
  scientific study under Reg. 106(2) of CMR 2017: 1.50 for permanent or long-standing dumps, 1.30 for all others.
- **Own physics engine, matched to Slide2.** Every synthetic label comes from a Bishop simplified solver
  written for this project, checked against the 8 SECL study values, 2,250 published Slide cases and the
  author's own Slide2 runs, not from FOS-scaling formulas.
- **Method-aware.** Each real FOS is tagged with the software that produced it (RS2 for SECL, FLAC for
  Jagannathpur, FEM for WCL). The model learns a separate correction for each and tells you which
  method its answer corresponds to.
- **Sim-to-real learning.** A neural-network ensemble on dimensionless groups learns slope mechanics
  from simulations; a Gaussian process then learns how *real* sites deviate from it, using only the
  real cases. Validated by leave-one-site-out.
- **Honest uncertainty.** Every prediction carries a 90 % conformal prediction interval.
- **Moisture acts through mechanisms** (unit weight, saturation, pore pressure, softening) on the
  Predict page, instead of being a column the model can ignore.
- **Monte Carlo with an exact cross-check**, correlated c-phi, reliability index and PoF.
- **Unified TARP** combining deterministic FOS, probability of failure and measured slope velocity,
  with inverse-velocity time-of-failure forecasting.
- **Design envelope** for planning DGMS-compliant bench height and angle.""")
    ph = load_physics()
    c1, c2, c3 = st.columns(3)
    c1.metric("Physics simulations", f"{len(ph):,}")
    c2.metric("Real cases used for calibration", model.n_real)
    c3.metric("Real rows on file (all statuses)", len(st.session_state.real))

# ------------------------------------------------------------------
elif PAGE == "Data and sources":
    st.title("Data and sources")
    st.markdown("Every real row carries a citation. Rows marked **partial** are placeholders: "
                "open the cited paper, fill the missing columns, set `use_for_training` to True.")
    real = st.session_state.real
    status = st.multiselect("Show status", sorted(real.status.dropna().unique()),
                            default=sorted(real.status.dropna().unique()))
    st.dataframe(real[real.status.isin(status)], **FULL, height=380)
    st.download_button("Download template / current table (CSV)", real.to_csv(index=False),
                       "real_cases.csv", "text/csv")
    st.subheader("Load your own real dataset")
    up = st.file_uploader("CSV with the same columns (your 50-60 cases)", type="csv")
    mode = st.radio("How to use it", ["Append to current table", "Replace current table"],
                    horizontal=True)
    if up is not None and st.button("Load file"):
        new = pd.read_csv(up)
        miss = [c for c in REQ if c not in new]
        if miss:
            st.error(f"Missing columns: {miss}. Download the template above for the format.")
        else:
            for col in real.columns:
                if col not in new: new[col] = np.nan
            new["use_for_training"] = new["use_for_training"].fillna(True)
            new["status"] = new["status"].fillna("complete")
            new["case_id"] = new["case_id"].fillna(pd.Series([f"U{i+1:03d}" for i in range(len(new))]))
            st.session_state.real = (pd.concat([real, new[real.columns]], ignore_index=True)
                                     if mode.startswith("Append") else new[real.columns])
            st.success(f"Loaded {len(new)} rows. The model retrains automatically.")
            st.rerun()
    st.subheader("Sahoo et al. (2025) Mendeley dataset - included")
    st.markdown(
        "The 2,250 published Slide (Janbu simplified) cases are built in, used two ways:\n"
        "- **Solver check:** each case re-solved with the in-house Bishop solver on the same "
        "geometry (a uniform slope). The published FOS matches a *uniform, unbenched* slope, so the "
        "paper did not bench its dumps.\n"
        "- **DGMS version (block A):** the 1,200 cases with overall angle 25 or 30 deg benched the "
        "way mines do (30 m decks, remainder on top, deck angle up to 37.5 deg, bench widths keeping the "
        "published overall angle) and re-solved. The 35 and 40 deg cases break the 1V:1.5H rule "
        "and stay in the non-DGMS set with the 2,250 unbenched originals (block C2).")
    chk_path = os.path.join(DATA, "mendeley_solver_check.csv")
    if os.path.exists(chk_path):
        chk = pd.read_csv(chk_path)
        k = st.columns(3)
        k[0].metric("Cases compared", f"{len(chk):,}")
        k[1].metric("Bishop minus published (mean)", f"{chk.diff_pct.mean():+.2f}%")
        k[2].metric("Spread of the difference (SD)", f"{chk.diff_pct.std():.2f}%")
        st.caption("A steady +3-4 % offset is expected: Janbu's simplified method without its "
                   "correction factor gives lower FOS than Bishop's method for circular slips.")
# ------------------------------------------------------------------
elif PAGE == "Methods and equations":
    from pages_extra import methods_page
    methods_page()

# ------------------------------------------------------------------
elif PAGE == "Validation":
    from pages_extra import validation_page
    validation_page(model, load_physics(), training_rows(st.session_state.real), DATA, PLOT, FULL, badge)

# ------------------------------------------------------------------
elif PAGE == "Model lab":
    st.title("Model lab")
    rep = model.report
    c1 = st.container()
    with c1:
        st.subheader("Physics hold-out (20%)")
        st.dataframe(pd.DataFrame({"Neural ensemble on dimensionless groups (used)": rep["surrogate_holdout"],
                                   "Gradient boosting, monotonic, raw inputs": rep["mono_holdout"],
                                   "Gradient boosting, raw inputs": rep["free_holdout"]}).T
                     .style.format("{:.4f}"))
        st.caption("Accuracy on simulations only shows the surrogate copies the solver. "
                   "The real test is the held-out sites below.")
        viol = pd.Series(rep["monotonic_violations"]) * 100
        st.caption("Physics check - share of sweeps where FOS moved the wrong way: " +
                   ", ".join(f"{k} {v:.1f}%" for k, v in viol.items()))
    if "loo" in rep:
        p = rep["loo"]["pred"]
        fig = go.Figure()
        lim = [p[["FOS_real", "hybrid", "physics"]].min().min() * 0.9,
               p[["FOS_real", "hybrid", "physics"]].max().max() * 1.1]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#999", dash="dot"),
                                 name="Perfect"))
        for col, clr, nm in [("physics", "#8c8c8c", "Physics only"), ("real_only", "#d9731f", "Real-data only"),
                             ("hybrid", TEAL, "Sim-to-real hybrid")]:
            fig.add_trace(go.Scatter(x=p.FOS_real, y=p[col], mode="markers", name=nm, text=p.mine,
                                     marker=dict(color=clr, size=10)))
        fig.update_layout(**PLOT, title="Predicted vs reported FOS (held-out mines)")
        fig.update_xaxes(title="Reported FOS"); fig.update_yaxes(title="Predicted FOS")
        st.plotly_chart(fig, **FULL)
    st.subheader("Physics consistency check")
    st.caption("Sweep one input with everything else fixed. A physically valid model never shows FOS "
               "falling when cohesion rises, and should be smooth. Tree models give steps.")
    feat = st.selectbox("Sweep", ["c_kPa", "phi_deg", "H_m", "r_u"], format_func=lambda f: LABELS[f])
    rng_ = {"c_kPa": (1, 120), "phi_deg": (8, 42), "H_m": (20, 150), "r_u": (0, 0.5)}[feat]
    vals = np.linspace(*rng_, 60)
    base = dict(c_kPa=40, phi_deg=28, gamma_kNm3=18.5, r_u=0.0, **DumpGeometry(3, 30, 32, 30).cols())
    df = pd.DataFrame([{**base, feat: v} for v in vals])
    if feat == "H_m":
        df["deck_height_m"] = df["H_m"] / df["n_decks"]
    fc = go.Figure([go.Scatter(x=vals, y=model.predict(df)["FOS"], name="Neural ensemble + real correction",
                               line=dict(color=TEAL, width=3)),
                    go.Scatter(x=vals, y=model.predict_free(df), name="Gradient boosting on raw inputs",
                               line=dict(color="#d9731f", width=2, dash="dot"))])
    fc.update_layout(**PLOT); fc.update_xaxes(title=LABELS[feat]); fc.update_yaxes(title="FOS")
    st.plotly_chart(fc, **FULL)
    st.subheader("Which inputs matter (permutation importance)")
    imp = model.permutation_importance(load_physics())
    st.plotly_chart(px.bar(imp, orientation="h", labels={"value": "RMSE increase", "index": ""})
                    .update_layout(**PLOT, showlegend=False), **FULL)

# ------------------------------------------------------------------
elif PAGE == "Predict and TARP":
    st.title("Predict FOS and TARP level")
    st.subheader("Material"); c, phi, g = material_inputs("p")
    st.subheader("Geometry"); geom = geometry_inputs("p")
    st.caption(f"Total height {geom.height:.0f} m, overall slope angle {geom.overall_angle:.1f} deg")
    st.subheader("Moisture and water"); on, w, cfg, ru_ext = moisture_inputs("p")
    ce, pe, ge, ru, S = effective(c, phi, g, on, w, cfg, ru_ext)
    if on and w > float(saturation_moisture(g, cfg["Gs"])):
        st.warning(f"Moisture {w:.1f} % is above the saturation limit "
                   f"({float(saturation_moisture(g, cfg['Gs'])):.1f} %) for this dry unit weight; it is "
                   "capped at saturation.")
    seen = [model.default_method] + [m for m in METHOD_LABELS
                                     if m in model.methods_seen and m != model.default_method]
    meth = st.selectbox("Calibrate to", seen, format_func=lambda m: METHOD_LABELS.get(m, m),
                        help="Each real study used its own software and assumptions. The model learns a "
                             "separate correction for each study; the default is the one covering the "
                             "most dump sites. 'Physics surrogate (Bishop)' is shown alongside.")
    pr = model.predict(row(ce, pe, ge, ru, geom), method=meth).iloc[0]
    lvl = det_level(pr.FOS, pr.FOS_lo)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted FOS", f"{pr.FOS:.3f}")
    c2.metric(f"{int(alpha*100)}% interval", f"{pr.FOS_lo:.2f} - {pr.FOS_hi:.2f}")
    c3.metric("Physics-only surrogate (Bishop)", f"{pr.FOS_physics:.3f}")
    if on:
        dry = model.predict(row(c, phi, g, ru_ext, geom), method=meth).iloc[0]
        st.markdown("**Effect of moisture**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bulk unit weight", f"{ge:.2f} kN/m3", f"dry {g:.2f}", delta_color="off")
        m2.metric("Degree of saturation", f"{S:.2f}",
                  f"saturated at w = {float(saturation_moisture(g, cfg['Gs'])):.1f} %", delta_color="off")
        m3.metric("Pore-pressure ratio r_u", f"{ru:.3f}")
        m4.metric("FOS dry -> with moisture", f"{dry.FOS:.3f} -> {pr.FOS:.3f}",
                  f"{(pr.FOS - dry.FOS) / dry.FOS * 100:+.1f} %", delta_color="off")
        if cfg["kc"] > 0 or cfg["kphi"] > 0:
            st.caption(f"Strength after softening: c = {ce:.1f} kPa, phi = {pe:.1f} deg (assumed coefficients).")
    tarp_box(lvl, f"FOS {pr.FOS:.2f} against minimum {FOS_MIN:.2f} ({cons.lower()}, DGMS para 3.2)",
             "The lower end of the prediction interval falls below the minimum." if lvl == "Yellow" else "")
    extrap = []
    ph = load_physics()
    for f, v in dict(c_kPa=ce, phi_deg=pe, gamma_kNm3=ge, H_m=geom.height).items():
        lo, hi = ph[f].quantile(0.005), ph[f].quantile(0.995)
        if not lo <= v <= hi: extrap.append(f"{LABELS[f]} = {v:.1f} (trained {lo:.1f}-{hi:.1f})")
    if extrap:
        st.warning("Outside the training range - verify with the exact solver: " + "; ".join(extrap))
    st.subheader("Slip surface (Slide2 procedure)")
    stg, scope, lim, shown = search_inputs("p", geom)
    if st.button("Run slip surface search and verify with exact Bishop solver", type="primary"):
        out = show_search_result(geom, ce, pe, ge, ru, stg, scope, lim, shown)
        st.caption(f"Material: unit weight {ge:.2f} kN/m3, cohesion {ce:.1f} kPa, friction angle {pe:.1f} deg, "
                   f"r_u {ru:.2f}.")
        if out:
            st.session_state.last["bishop"] = out[0].fos
    else:
        st.plotly_chart(profile_fig(geom, None, "Dump section"), **FULL)
    st.session_state.last.update(dict(time=str(dt.datetime.now())[:16], c=ce, phi=pe, gamma=ge, ru=ru,
                                      H=geom.height, beta=geom.overall_angle, deck=geom.deck_angle,
                                      FOS=pr.FOS, lo=pr.FOS_lo, hi=pr.FOS_hi, level=lvl))

# ------------------------------------------------------------------
elif PAGE == "Monte Carlo reliability":
    st.title("Monte Carlo reliability")
    st.subheader("Material means"); c, phi, g = material_inputs("m")
    st.subheader("Variability")
    c1, c2, c3, c4 = st.columns(4)
    cc = c1.slider("COV of cohesion", 0.05, 0.6, 0.30, 0.05)
    pc = c2.slider("COV of friction angle", 0.02, 0.25, 0.10, 0.01)
    gc = c3.slider("COV of unit weight", 0.01, 0.12, 0.05, 0.01)
    rho = c4.slider("c-phi correlation", -0.8, 0.8, 0.0, 0.1)
    st.caption("Typical COVs: unit weight 3-7%, friction 2-13%, cohesion 10-50% (Kulhawy 1992).")
    st.subheader("Geometry"); geom = geometry_inputs("m")
    c1, c2, c3 = st.columns(3)
    ru_lo, ru_hi = c1.slider("r_u range", 0.0, 0.6, (0.0, 0.0), 0.01)
    n = c2.select_slider("Realisations", [1000, 5000, 10000, 20000, 50000], value=10000)
    seed = c3.number_input("Random seed", 0, 9999, 42)
    df = run_mc(model, geom, sample_inputs(n, c, cc, phi, pc, g, gc, rho, ru_range=(ru_lo, ru_hi),
                                           seed=int(seed)))
    r = reliability(df.FOS)
    k = st.columns(5)
    k[0].metric("Mean FOS", f"{r['mean']:.3f}")
    k[1].metric("5th percentile", f"{r['p05']:.3f}")
    k[2].metric("PoF (FOS < 1)", f"{r['pof']*100:.2f}%", help=f"95% CI +/- {r['pof_ci']*100:.2f}%")
    k[3].metric("Reliability index (lognormal)", f"{r['beta_lognormal']:.2f}")
    k[4].metric("COV of FOS", f"{r['cov']*100:.1f}%")
    lvl = prob_level(r["mean"], r["pof"])
    tarp_box(lvl, f"mean FOS {r['mean']:.2f} (min {FOS_MIN:.2f}), PoF {r['pof']*100:.2f}% (max {POF_MAX*100:.0f}%)",
             "Stable = both criteria met, marginal = one, unstable = none (Read & Stacey 2009, ch. 9).")
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x="FOS", nbins=80, color_discrete_sequence=[TEAL])
    for thr, col in [(1.0, SLIP), (FOS_MIN, TARP_COL["Green"])]:
        fig.add_vline(x=thr, line_dash="dash", line_color=col)
    c1.plotly_chart(fig.update_layout(**PLOT, title="FOS distribution"), **FULL)
    lv = df.FOS.map(fos_level).value_counts().reindex(["Green", "Orange", "Red"]).fillna(0) / n * 100
    c2.plotly_chart(px.bar(x=lv.index, y=lv.values, color=lv.index, color_discrete_map=TARP_COL,
                           labels={"x": "", "y": "% of realisations"})
                    .update_layout(**PLOT, showlegend=False, title="TARP level distribution"),
                    **FULL)
    sens = rank_sensitivity(df, ["c_kPa", "phi_deg", "gamma_kNm3", "r_u"])
    st.plotly_chart(px.bar(sens.rename(index=LABELS), orientation="h",
                           labels={"value": "Spearman rank correlation with FOS", "index": ""})
                    .update_layout(**PLOT, showlegend=False, title="What drives the uncertainty"),
                    **FULL)
    if st.button("Cross-check 80 realisations with the exact Bishop solver"):
        with st.spinner("Running exact LEM ..."):
            chk = exact_check(geom, df)
        m = metrics(chk.exact_bishop, chk.surrogate)
        st.write(f"Physics surrogate vs exact Bishop: R2 {m['R2']:.3f}, MAE {m['MAE']:.3f}, MAPE {m['MAPE']:.2f}%")
        st.plotly_chart(px.scatter(chk, x="exact_bishop", y="surrogate").update_layout(**PLOT),
                        **FULL)
        from scipy.stats import spearmanr
        sub = df.loc[df.sample(min(80, len(df)), random_state=1).index]
        cmp = pd.DataFrame({
            "Exact Bishop (80 runs)": {LABELS[c]: spearmanr(sub[c], chk.exact_bishop).statistic
                                       for c in ["c_kPa", "phi_deg", "gamma_kNm3"]},
            f"Model ({n:,} runs)": {LABELS[c]: sens.get(c, np.nan) for c in ["c_kPa", "phi_deg", "gamma_kNm3"]}})
        st.write("Sensitivity check - rank correlation with FOS:")
        st.dataframe(cmp.style.format("{:+.2f}"))
        st.caption("Signs and ranking should agree. Small differences come from the real-data correction "
                   "and from the smaller exact sample (about +/-0.1 with 80 runs).")
    st.session_state.last["mc"] = dict(n=n, **{k_: float(v) for k_, v in r.items()}, level=lvl)
    st.download_button("Download realisations (CSV)", df.to_csv(index=False), "monte_carlo.csv")

# ------------------------------------------------------------------
elif PAGE == "Design envelope":
    st.title("Reliability-based design envelope")
    c, phi, g = material_inputs("d")
    st.caption("Every design is benched per CMR 2017 Reg. 106: benches of at most 30 m, and bench widths at "
               "least wide enough to keep the overall slope within 1V:1.5H.")
    c2, c3, c4 = st.columns(3)
    bw = c2.number_input("Preferred bench width (m)", 10.0, 60.0, 30.0)
    ru = c3.slider("Design r_u", 0.0, 0.5, 0.0, 0.01)
    target = c4.number_input("Target FOS", 1.0, 2.0, float(FOS_MIN), 0.05,
                             help="Defaults to the sidebar acceptance criterion (DGMS circular, para 3.2).")
    pof_t = st.slider("Maximum acceptable PoF (%)", 0.5, 50.0, float(POF_MAX * 100), 0.5) / 100
    heights = np.arange(30, 121, 10); angles = np.arange(24, 41, 1)
    with st.spinner("Evaluating designs ..."):
        env = design_envelope(model, c, phi, g, ru, heights, angles, berm=bw)
    allowed = env.dgms_ok | bool(st.session_state.get("use_noncompliant"))
    env["ok"] = (env.FOS_mean >= target) & (env.PoF <= pof_t) & allowed
    Z = env.pivot(index="H_m", columns="deck_angle_deg", values="FOS_mean")
    fig = go.Figure(go.Heatmap(z=Z.values, x=Z.columns, y=Z.index, colorscale="RdYlGn",
                               zmin=0.8, zmax=2.0, colorbar=dict(title="FOS")))
    OK = env.pivot(index="H_m", columns="deck_angle_deg", values="ok").astype(float)
    fig.add_trace(go.Contour(z=OK.values, x=OK.columns, y=OK.index, showscale=False,
                             contours=dict(start=0.5, end=0.5, coloring="lines"),
                             line=dict(color="black", width=3), name="Acceptable limit"))
    NC = env.pivot(index="H_m", columns="deck_angle_deg", values="dgms_ok")
    ncx, ncy = np.meshgrid(NC.columns, NC.index)
    fig.add_trace(go.Scatter(x=ncx[~NC.values], y=ncy[~NC.values], mode="markers",
                             marker=dict(symbol="x", color="#333", size=7),
                             name=("Needs DGMS order (beyond Reg. 106)" if st.session_state.get("use_noncompliant")
                                   else "Not allowed by Reg. 106")))
    fig.update_layout(**PLOT, title="Mean FOS; black line = limit meeting FOS, PoF and DGMS")
    fig.update_xaxes(title="Deck angle (deg)"); fig.update_yaxes(title="Total dump height (m)")
    st.plotly_chart(fig, **FULL)
# ------------------------------------------------------------------
elif PAGE == "Multi-parameter TARP":
    st.title("Multi-parameter TARP for the dump")
    st.markdown("<p class='lede'>Six triggers, each with its own Green / Yellow / Orange / Red level: the design "
                "FOS, the live FOS with today's water, rainfall, slope velocity, acceleration (inverse velocity) "
                "and shift-inspection signs. The worst trigger sets the TARP level, and the level is raised one "
                "step when people or haul roads are in the run-out zone.</p>", unsafe_allow_html=True)
    src = st.radio("Data source", ["Demo A: creep to failure (synthetic)", "Demo B: monsoon storm (synthetic)",
                                   "Upload CSV", "Live CSV URL"], horizontal=True)
    st.caption("Required columns: timestamp, displacement_mm (prism, total station or radar). Optional: "
               "rainfall_mm (rain gauge, IMD or GPM), piezo_head_m (piezometer head above the tip) or r_u.")
    url, t_true = None, None
    if src == "Upload CSV":
        up = st.file_uploader("Monitoring CSV", type="csv")
        raw = pd.read_csv(up) if up else None
    elif src == "Live CSV URL":
        url = st.text_input("Published CSV link (e.g. Google Sheets > Publish to web > CSV)")
        raw = None
    elif src.startswith("Demo A"):
        raw, t_true = demo_series()
        st.info("Synthetic data: steady creep, then accelerating (tertiary) creep that ends in failure on day "
                f"{t_true:.0f}. Used only to show how the triggers behave; replace with real readings.")
    else:
        raw, t_true = demo_storm()
        st.info("Synthetic data: steady slow creep and a two-day storm of about 150 mm/day from day 20, with a "
                "piezometer that rises and drains. Movement stays Normal; watch the rainfall and live-FOS "
                "triggers.")

    with st.expander("Site: material, geometry and what lies below the dump", expanded=False):
        c, phi, g = material_inputs("rt")
        geom = geometry_inputs("rt")
        k1, k2 = st.columns(2)
        ru_design = k1.number_input("Design r_u (used for T1)", 0.0, 0.6, 0.0, 0.01,
                                    help="Pore-pressure ratio assumed in the design. 0 = drained (dry) design.")
        people = k2.toggle("People, haul road or structures in the run-out zone", value=False,
                           help="Raises Yellow to Orange and Orange to Red (consequence of failure). Design FOS and "
                                "rainfall alone are raised to Orange at most; Red needs a movement, live-FOS or "
                                "visual trigger.")
    with st.expander("Trigger thresholds and processing (calibrate for your mine)", expanded=False):
        st.markdown("**T3 Rainfall in 24 h (mm).** Default: IMD classes heavy / very heavy / extremely heavy.")
        r1, r2, r3 = st.columns(3)
        rain_thr = (r1.number_input("Yellow from", 0.0, 500.0, 64.5, 0.5),
                    r2.number_input("Orange from", 0.0, 500.0, 115.6, 0.5),
                    r3.number_input("Red from", 0.0, 800.0, 204.5, 0.5))
        st.markdown("**T4 Velocity.** Bands of the NIT Rourkela scientific-study notes (table on the right "
                    "tab). Velocity = slope of a line through the readings in a trailing window.")
        p1, p2 = st.columns(2)
        window = p1.number_input("Velocity window (days)", 0.25, 5.0, 1.0, 0.25)
        floor = p2.number_input("Noise floor (mm/day)", 0.0, 2.0, 0.2, 0.05,
                                help="Velocities below this are treated as no measurable movement.")
        st.markdown("**T5 Acceleration** (Carla et al. 2017): onset when the short moving average of velocity "
                    "rises above the long one and stays there; inverse-velocity window from the raw, short and "
                    "long series.")
        a1, a2, a3, a4 = st.columns(4)
        sma_d = a1.number_input("Short average (days)", 0.25, 5.0, 1.0, 0.25)
        lma_d = a2.number_input("Long average (days)", 1.0, 30.0, 5.0, 0.5)
        ratio = a3.number_input("Short / long ratio", 1.0, 3.0, 1.3, 0.05)
        persist = a4.number_input("Held for (days)", 0.0, 10.0, 2.0, 0.25)
        a5, a6, a7, a8 = st.columns(4)
        fit_d = a5.number_input("1/v fit length (days)", 2.0, 30.0, 5.0, 0.5)
        orange_d = a6.number_input("Orange if failure within (days)", 1.0, 60.0, 7.0, 0.5,
                                   help="Also needs the velocity above the Normal band (2 mm/day).")
        red_d = a7.number_input("Red if failure within (days)", 0.5, 30.0, 3.0, 0.5)
        red_v = a8.number_input("and velocity at least (mm/day)", 0.0, 200.0, 10.0, 1.0,
                                help="Red from inverse velocity also needs the slope to be in the Alert band, "
                                     "so a noisy early fit cannot evacuate the pit on its own.")
        st.markdown("**T2 Water link.** Measured r_u from a piezometer is used when available; otherwise r_u is "
                    "estimated from the antecedent rainfall index (ARI).")
        w1, w2, w3, w4 = st.columns(4)
        decay = w1.slider("ARI decay per day", 0.5, 0.99, 0.90, 0.01)
        ru_per_mm = w2.number_input("r_u per mm of ARI", 0.0, 0.01, 0.0012, 0.0001, format="%.4f",
                                    help="Replaced by the fitted value when rain and piezometer data are both "
                                         "present.")
        ru_max = w3.slider("Maximum r_u", 0.0, 0.6, 0.40, 0.01)
        z_tip = w4.number_input("Piezometer tip depth (m)", 1.0, 100.0, 10.0, 0.5)
    st.markdown("**T6 Shift inspection.** Tick what the shift overman / geotech engineer saw today.")
    seen = st.multiselect("Visual signs observed", [s_ for s_, _ in VISUAL_SIGNS], default=[],
                          label_visibility="collapsed", placeholder="No signs observed")

    def card(col, name, level, value, note=""):
        col.markdown(f"<div style='border-radius:10px;padding:10px 12px;color:white;min-height:118px;"
                     f"background:{TCOL[level]}'><div style='font-size:0.78rem;opacity:0.9'>{name}</div>"
                     f"<div style='font-size:1.15rem;font-weight:700'>{level}</div>"
                     f"<div style='font-size:0.86rem'>{value}</div>"
                     f"<div style='font-size:0.74rem;opacity:0.9'>{note}</div></div>", unsafe_allow_html=True)

    def render(raw):
        d = process_series(raw, window_days=window, noise_floor=floor)
        if len(d) > 12:
            t_end = float(d.t_days.iloc[-1])
            upto = st.slider("Replay: show the TARP as it stood on day", float(d.t_days.iloc[12]), t_end, t_end,
                             0.25, help="Every calculation uses only the readings up to this day, as the mine "
                                        "would have seen it then.")
            d = d[d.t_days <= upto].reset_index(drop=True)
        d = add_moving_averages(d, sma_d, lma_d)
        has_rain = "rainfall_mm" in d
        d["rain_24h"] = rolling_sum(d.t_days, d.rainfall_mm) if has_rain else np.nan
        d["ARI_mm"] = antecedent_index(d.t_days, d.rainfall_mm, decay) if has_rain else 0.0
        ru_src, k_fit = "none (dry)", None
        if "piezo_head_m" in d:
            d["r_u_piezo"] = ru_from_piezometer(d.piezo_head_m, z_tip, g)
        elif "r_u" in d:
            d["r_u_piezo"] = d["r_u"].astype(float)
        if "r_u_piezo" in d:
            d["r_u_est"] = d["r_u_piezo"].clip(0, ru_max)
            ru_src = "piezometer"
            if has_rain:
                k_fit, r2_fit = calibrate_ru(d.ARI_mm, d.r_u_piezo)
        elif has_rain:
            d["r_u_est"] = np.minimum(ru_max, ru_per_mm * d.ARI_mm)
            ru_src = "rainfall (ARI x coefficient)"
        else:
            d["r_u_est"] = ru_design
        X = pd.DataFrame(dict(c_kPa=c, phi_deg=phi, gamma_kNm3=g, r_u=d.r_u_est, **geom.cols()))
        d["FOS"] = model.predict(X, with_interval=False)["FOS"].values
        des = model.predict(row(c, phi, g, ru_design, geom)).iloc[0]
        cur = d.iloc[-1]
        onset = onset_of_acceleration(d, ratio, persist, floor, lma_d)
        win = iv_window(d, onset, floor, fit_days=fit_d) if onset is not None else None
        v_info = classify_velocity(cur.velocity_mm_day)
        trig = {
            "T1 Design FOS": fos_trigger(des.FOS, FOS_MIN, des.FOS_lo),
            "T2 Live FOS": fos_trigger(cur.FOS, FOS_MIN),
            "T3 Rainfall": rain_trigger(cur.rain_24h, rain_thr) if has_rain else "No data",
            "T4 Velocity": VEL_MAP[v_info["level"]],
            "T5 Acceleration": accel_trigger(onset, win, cur.velocity_mm_day, red_d, orange_d, red_v, 2.0),
            "T6 Visual signs": visual_trigger(seen),
        }
        final, gov, note, missing = tarp_evaluate(trig, people)
        act = next(a for a in ACTION_TABLE if a["Level"] == final)
        st.markdown(f"<div class='tarp' style='background:{TCOL[final]}'><b>TARP level: {final}</b>"
                    f"&nbsp; {act['Meaning']}<br>"
                    f"Governing trigger: {', '.join(gov) if gov else 'none'}"
                    f"{(' &middot; ' + note) if note else ''}<br>"
                    f"<b style='font-size:1rem'>Action:</b> {act['Action']}<br>"
                    f"<b style='font-size:1rem'>Responsible:</b> {act['Responsible']} &middot; "
                    f"<b style='font-size:1rem'>Response:</b> {act['Response_time']}</div>",
                    unsafe_allow_html=True)
        cols = st.columns(6)
        card(cols[0], "T1 Design FOS", trig["T1 Design FOS"], f"{des.FOS:.2f} ({des.FOS_lo:.2f}-{des.FOS_hi:.2f})",
             f"min {FOS_MIN:.2f}, r_u {ru_design:.2f}")
        card(cols[1], "T2 Live FOS", trig["T2 Live FOS"], f"{cur.FOS:.2f}", f"r_u {cur.r_u_est:.2f} from {ru_src}")
        card(cols[2], "T3 Rainfall (24 h)", trig["T3 Rainfall"],
             f"{cur.rain_24h:.0f} mm" if has_rain else "no gauge", f"ARI {cur.ARI_mm:.0f} mm" if has_rain else "")
        card(cols[3], "T4 Velocity", trig["T4 Velocity"], f"{cur.velocity_mm_day:.1f} mm/day",
             f"{v_info['level']}; survey {v_info['frequency'].lower()}")
        if win:
            t5v, t5n = (f"failure day {win['t_early']:.1f}-{win['t_late']:.1f}",
                        f"{max(win['days_early'], 0):.1f}-{max(win['days_late'], 0):.1f} days left")
        elif onset is not None:
            t5v, t5n = f"accelerating since day {onset:.1f}", "no reliable 1/v fit yet"
        else:
            t5v, t5n = "not accelerating", f"short avg {cur.v_sma:.1f}, long {cur.v_lma:.1f} mm/day"
        card(cols[4], "T5 Acceleration", trig["T5 Acceleration"], t5v, t5n)
        card(cols[5], "T6 Visual signs", trig["T6 Visual signs"], f"{len(seen)} sign(s) ticked",
             "from shift inspection")
        if missing:
            st.caption("No data for: " + ", ".join(missing) + " (not counted; install the instrument).")

        hist = level_history(d, FOS_MIN, rain_thr, ratio, persist, floor, red_d, orange_d, red_v, lma_d, fit_d)
        hist["T1 Design FOS"] = trig["T1 Design FOS"]
        hist["Overall"] = [tarp_evaluate({k: r[k] for k in TRIG_AUTO + ["T1 Design FOS"]}, people)[0]
                           for _, r in hist.iterrows()]
        tabs = st.tabs(["TARP history", "Movement", "Inverse velocity", "Water and FOS", "Action plan"])
        with tabs[0]:
            rows_ = ["T1 Design FOS"] + TRIG_AUTO + ["Overall"]
            z = [[ORDER.index(x) if x in ORDER else np.nan for x in hist[r_]] for r_ in rows_]
            cs = [[0.00, TCOL["Green"]], [0.25, TCOL["Green"]], [0.25, TCOL["Yellow"]], [0.50, TCOL["Yellow"]],
                  [0.50, TCOL["Orange"]], [0.75, TCOL["Orange"]], [0.75, TCOL["Red"]], [1.00, TCOL["Red"]]]
            hm = go.Figure(go.Heatmap(z=z, x=d.timestamp, y=rows_, zmin=-0.5, zmax=3.5, colorscale=cs,
                                      showscale=False, hovertemplate="%{y}<br>%{x}<extra></extra>"))
            hm.update_layout(**PLOT, title="Level of each trigger through the record (T6 is entered by hand)")
            hm.update_layout(height=330)
            hm.update_yaxes(autorange="reversed")
            st.plotly_chart(hm, **FULL)
            first = {L: first_time(hist, "Overall", L) for L in ["Yellow", "Orange", "Red"]}
            vel_only = {L: first_time(hist, "T4 Velocity", L) for L in ["Orange", "Red"]}
            m = st.columns(4)
            for j, L in enumerate(["Yellow", "Orange", "Red"]):
                m[j].metric(f"First {L}", "never" if first[L] is None else f"day {first[L]:.1f}",
                            None if (first[L] is None or t_true is None) else
                            f"{t_true - first[L]:.1f} days before failure", delta_color="off")
            m[3].metric("Velocity bands alone: first Orange",
                        "never" if vel_only["Orange"] is None else f"day {vel_only['Orange']:.1f}",
                        "Red: never" if vel_only["Red"] is None else f"Red: day {vel_only['Red']:.1f}",
                        delta_color="off")
            st.caption("Each automatic trigger is re-evaluated at every reading using only the data up to that "
                       "time, so the lead times are what the mine would have seen in real time.")
        with tabs[1]:
            top = max(150.0, float(d.velocity_mm_day.max()) * 1.2)
            bands = [0.05] + [b[0] for b in VELOCITY_TARP[:-1]] + [top]
            fig = go.Figure()
            for j, b in enumerate(VELOCITY_TARP):
                fig.add_hrect(y0=bands[j], y1=bands[j + 1], fillcolor=b[2], opacity=0.16, line_width=0,
                              annotation_text=f"{b[1]} ({b[4].lower()})", annotation_position="right",
                              annotation_font_size=11)
            fig.add_trace(go.Scatter(x=d.timestamp, y=d.velocity_mm_day.clip(lower=0.05), name="Velocity",
                                     line=dict(color="#9aa5a6", width=1)))
            fig.add_trace(go.Scatter(x=d.timestamp, y=d.v_sma.clip(lower=0.05), name=f"Short avg ({sma_d:g} d)",
                                     line=dict(color=INK, width=2)))
            fig.add_trace(go.Scatter(x=d.timestamp, y=d.v_lma.clip(lower=0.05), name=f"Long avg ({lma_d:g} d)",
                                     line=dict(color=TEAL, width=2, dash="dash")))
            if onset is not None:
                fig.add_vline(x=d.timestamp.iloc[int(np.argmin(abs(d.t_days - onset)))], line_color=SLIP,
                              line_dash="dot", annotation_text="onset of acceleration")
            fig.update_layout(**PLOT, title="Slope velocity against the movement bands")
            fig.update_layout(margin=dict(l=10, r=150, t=48, b=10), legend=dict(orientation="h", y=-0.15))
            fig.update_yaxes(title="mm/day (log scale)", type="log", range=[np.log10(0.05), np.log10(top)])
            st.plotly_chart(fig, **FULL)
            st.dataframe(pd.DataFrame(VELOCITY_TARP, columns=["Upper limit (mm/day)", "Level", "Colour",
                                                              "Method", "Frequency", "Response"])
                         .drop(columns="Colour").assign(TARP=lambda x: x.Level.map(VEL_MAP)), **FULL)
            st.caption("Velocity bands and survey frequencies: slope-movement table of the NIT Rourkela "
                       "scientific-study notes supplied with this project.")
        with tabs[2]:
            if onset is None:
                st.info("No onset of acceleration: the short average of velocity is not held above the long "
                        "average, so an inverse-velocity forecast is not meaningful yet.")
            dd = d[(d.velocity_mm_day >= floor) & (d.t_days >= (onset if onset is not None else 0))]
            fi = go.Figure(go.Scatter(x=dd.t_days, y=1 / dd.velocity_mm_day, mode="markers",
                                      marker=dict(color="#9aa5a6", size=4), name="1/v raw"))
            if win:
                tt0 = max(onset, float(d.t_days.iloc[-1]) - fit_d)
                colr = {"raw": "#6b7778", "SMA": INK, "LMA": TEAL}
                for nm, ft in win["fits"].items():
                    tt = np.linspace(tt0, ft["t_fail"], 40)
                    fi.add_trace(go.Scatter(x=tt, y=ft["slope"] * tt + ft["intercept"],
                                            name=f"{nm} fit: day {ft['t_fail']:.1f} (R2 {ft['r2']:.2f})",
                                            line=dict(color=colr[nm], dash="dash")))
                fi.add_vrect(x0=win["t_early"], x1=max(win["t_late"], win["t_early"] + 0.1),
                             fillcolor=SLIP, opacity=0.18, line_width=0,
                             annotation_text="forecast window", annotation_position="top left")
            if t_true is not None:
                fi.add_vline(x=t_true, line_color="#333", line_dash="dash", annotation_text=f"true day {t_true:.0f}",
                             annotation_position="bottom right")
            ymax = float(np.nanpercentile(1 / dd.velocity_mm_day, 98)) * 1.1 if len(dd) else 1
            fi.update_yaxes(title="1/v (day/mm)", range=[0, ymax])
            st.plotly_chart(fi.update_layout(**PLOT, title="Inverse velocity after the onset of acceleration")
                            .update_xaxes(title="Days since start"), **FULL)
            if win and t_true is not None:
                validated(f"Synthetic record: forecast window day {win['t_early']:.1f}-{win['t_late']:.1f} "
                          f"against the true failure on day {t_true:.0f}. Early forecasts are on the safe side.")
        with tabs[3]:
            ff = go.Figure(go.Scatter(x=d.timestamp, y=d.FOS, line=dict(color=TEAL, width=3), name="Live FOS"))
            ff.add_hline(y=FOS_MIN, line_dash="dot", line_color=TARP_COL["Green"],
                         annotation_text=f"DGMS minimum {FOS_MIN:.2f}")
            ff.add_hline(y=1.0, line_dash="dot", line_color=SLIP, annotation_text="FOS 1.0")
            if has_rain:
                ff.add_trace(go.Bar(x=d.timestamp, y=d.rainfall_mm, name="Rain (mm)", yaxis="y2",
                                    marker_color="#7fa7c9", opacity=0.6))
                ff.update_layout(yaxis2=dict(overlaying="y", side="right", title="Rain (mm)", showgrid=False))
            ff.update_layout(**PLOT, title="Live FOS from the AI model with today's pore pressure")
            ff.update_layout(legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(ff.update_yaxes(title="FOS"), **FULL)
            rr = go.Figure(go.Scatter(x=d.timestamp, y=d.r_u_est, name="r_u used", line=dict(color=INK, width=2)))
            if "r_u_piezo" in d and has_rain and k_fit:
                rr.add_trace(go.Scatter(x=d.timestamp, y=np.minimum(ru_max, k_fit * d.ARI_mm),
                                        name="r_u from rain (fitted)", line=dict(color="#7fa7c9", dash="dash")))
            rr.update_layout(**PLOT, title="Pore-pressure ratio r_u")
            rr.update_layout(height=300, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(rr, **FULL)
            if k_fit:
                validated(f"Rain-to-r_u link learned from this record: r_u = {k_fit:.5f} x ARI "
                          f"(R2 {r2_fit:.2f}). Use it for dumps that have a rain gauge but no piezometer.")
            else:
                validated("r_u from rainfall uses an assumed coefficient. Fit it to piezometer readings "
                          "after storms before relying on the live FOS.", "warn")
        with tabs[4]:
            st.dataframe(pd.DataFrame(ACTION_TABLE).rename(columns={"Response_time": "Response time"}),
                         hide_index=True, **FULL)
            st.dataframe(pd.DataFrame([
                ["T1 Design FOS", "AI model (Bishop surrogate + field correction)", "near minimum or 90 % range "
                 "below it", "below DGMS minimum", "below 1.0"],
                ["T2 Live FOS", "AI model with today's r_u", "within 0.05 of minimum", "below DGMS minimum",
                 "below 1.0"],
                ["T3 Rainfall (24 h)", "Rain gauge / IMD / GPM", f">= {rain_thr[0]:g} mm", f">= {rain_thr[1]:g} mm",
                 f">= {rain_thr[2]:g} mm"],
                ["T4 Velocity", "Prism / total station / radar", "5-10 mm/day", "10-50 mm/day", "> 50 mm/day"],
                ["T5 Acceleration", "Moving averages + inverse velocity", "onset of acceleration",
                 f"failure within {orange_d:g} days and v >= 2 mm/day",
                 f"within {red_d:g} days and v >= {red_v:g} mm/day"],
                ["T6 Visual signs", "Shift inspection", "new cracks, settlement, seepage",
                 "widening cracks, toe bulging, muddy outflow", "scarp, visible mass movement"],
            ], columns=["Trigger", "Source", "Yellow", "Orange", "Red"]), hide_index=True, **FULL)
            st.caption("Rule: the worst trigger governs; Yellow and Orange are raised one level when people, haul "
                       "roads or structures are in the run-out zone (design FOS and rainfall alone stop at Orange). DGMS minimum FOS from the Reg. 106(2) "
                       "scientific-study circular, para 3.2; para 4.1 asks the study to set TARP trigger points. "
                       "Responsibilities and response times are proposed for the mine's Safety Management Plan.")
            st.download_button("Download today's TARP record (CSV)",
                               pd.DataFrame([dict(time=str(cur.timestamp), level=final,
                                                  governing="; ".join(gov), **trig)]).to_csv(index=False),
                               "tarp_record.csv", "text/csv")
        st.session_state.last["monitoring"] = dict(velocity=float(cur.velocity_mm_day),
                                                   fos=float(cur.FOS), level=final)

    if src == "Live CSV URL" and url:
        every = st.number_input("Refresh every (seconds)", 30, 3600, 300)
        if hasattr(st, "fragment"):
            @st.fragment(run_every=int(every))
            def live():
                try:
                    render(pd.read_csv(url))
                    st.caption(f"Last refreshed {dt.datetime.now():%H:%M:%S}")
                except Exception as e:
                    st.error(f"Could not read the feed: {e}")
            live()
        else:
            render(pd.read_csv(url))
    elif raw is not None:
        render(raw)
