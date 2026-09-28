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
import streamlit as st

from core.lem import DumpGeometry, bishop_fos, slip_arc
from core.models import FOSModel, FEATURES, LABELS, METHOD_LABELS, method_class, engineer, metrics
from core.moisture import moisture_state, saturation_moisture
from core.montecarlo import sample_inputs, run_mc, reliability, exact_check, rank_sensitivity
from core.monitoring import (process_series, inverse_velocity_forecast, classify_velocity,
                             demo_series, VELOCITY_TARP, LEVEL_ORDER)
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
    "Predict and TARP", "Monte Carlo reliability", "Moisture and rainfall", "Design envelope",
    "Back-analysis of failures", "Real-time monitoring", "Report"])

# Acceptance criteria: Read & Stacey (2009) Guidelines for Open Pit Slope Design, Table 9.9
# (Wesseloo & Read), overall slope scale. Minimum static FoS / maximum PoF.
RS_TABLE = {"Low": (1.2, 0.20), "Medium": (1.3, 0.10), "High": (1.3, 0.05)}
with st.sidebar.expander("Acceptance criteria (TARP)", expanded=False):
    cons = st.selectbox("Consequence of failure", ["Medium", "Low", "High"],
                        help="Overall-slope criteria of Read & Stacey (2009), Table 9.9: Low FoS 1.2-1.3 / "
                             "PoF 15-20 %; Medium FoS 1.3 / PoF 5-10 %; High FoS 1.3-1.5 / PoF 5 %. A dump "
                             "next to haul roads, workings or villages is at least Medium.")
    fs_def, pf_def = RS_TABLE[cons]
    FOS_MIN = st.number_input("Minimum acceptable FOS", 1.0, 2.0, fs_def, 0.05)
    POF_MAX = st.number_input("Maximum acceptable PoF (%)", 0.5, 50.0, pf_def * 100, 0.5) / 100
    st.caption("Use the values in your mine's approved scientific study / DGMS permission if they differ.")
alpha = st.sidebar.select_slider("Prediction-interval coverage", [0.80, 0.90, 0.95], value=0.90)
st.session_state.use_noncompliant = st.sidebar.toggle(
    "DGMS permission for steeper / higher dumps", value=False,
    help="Reg. 106 of CMR 2017 lets a spoil bank exceed 37.5 deg, 30 m benches or 1V:1.5H only when a "
         "scientific study recommends it and the Regional Inspector permits it by order. Switch on if your "
         "mine holds such an order (or to back-analyse failed non-compliant dumps). The model then also "
         "trains on 2,850 non-compliant designs and the 14 WCL cases, and designs beyond Reg. 106 are shown "
         "as 'needs DGMS order' instead of 'not allowed'.")
model: FOSModel = train_model(st.session_state.real.to_csv(index=False), 1 - alpha,
                              st.session_state.use_noncompliant)
st.sidebar.caption(f"Model trained on {len(load_physics()):,} physics rows + "
                   f"{model.n_real} real rows" + ("" if st.session_state.use_noncompliant
                                                  else " (DGMS Reg. 106 designs only)"))

# Actions follow the stability classes of Read & Stacey (2009) ch. 9 (stable / marginal /
# unstable) plus a failure class (FOS < 1).
ACTIONS = {
    "Green": "Stable: meets the acceptance criteria. Normal operations and routine monitoring.",
    "Yellow": "Marginal: fails one criterion (or the uncertainty band crosses it). Minor geometry changes, "
              "closer monitoring, check drainage and cracks.",
    "Orange": "Unstable design: fails the acceptance criteria. Major geometry changes (flatten, lower, wider "
              "berms), dewatering, restrict access below the toe.",
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


def profile_fig(geom: DumpGeometry, res=None, title="", fos=None):
    xs, ys = geom.surface()
    L, H = geom.horizontal_extent, geom.height
    x0, x1 = -0.35 * L - 10, L + 0.45 * L + 20
    X = np.linspace(x0, x1, 400); Y = np.interp(X, xs, ys)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[X, X[::-1]], y=np.r_[Y, np.full_like(X, -0.08 * H)],
                             fill="toself", fillcolor=OB, line=dict(color=OB_LINE, width=2),
                             name="Dump", hoverinfo="skip"))
    fig.add_hrect(y0=-0.18 * H, y1=-0.08 * H, fillcolor="#6f6a60", opacity=0.6, line_width=0)
    if res is not None and np.isfinite(res.fos):
        ax, ay = slip_arc(res, geom)
        fig.add_trace(go.Scatter(x=ax, y=ay, mode="lines", line=dict(color=SLIP, width=3, dash="dash"),
                                 name=f"Critical slip (Bishop FOS {res.fos:.2f})"))
        fig.add_trace(go.Scatter(x=[res.xc], y=[res.yc], mode="markers",
                                 marker=dict(color=SLIP, size=7, symbol="x"), name="Slip centre"))
    fig.update_layout(**PLOT, title=title, showlegend=True,
                      legend=dict(orientation="h", y=-0.15))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="Elevation (m)")
    fig.update_xaxes(title="Distance (m)", range=[x0, x1])
    return fig


def geometry_inputs(key, default=(3, 30.0, 32.0, 30.0)):
    c1, c2, c3, c4, c5 = st.columns(5)
    nd = c1.number_input("Number of decks", 1, 8, default[0], key=f"{key}nd")
    dh = c2.number_input("Deck height (m)", 5.0, 150.0, default[1], key=f"{key}dh",
                         help="Reg. 106: at most 30 m when the dump is higher than 30 m. Larger values are "
                              "allowed here only to study unbenched or permitted dumps.")
    top = c3.number_input("Top deck height (m)", 0.0, 150.0, 0.0, key=f"{key}top", disabled=nd == 1,
                          help="0 = same as the other decks. Mines usually build full 30 m decks "
                               "and put the remainder on top (e.g. 30 + 30 + 20 m).")
    da = c4.number_input("Deck angle (deg)", 10.0, 60.0, default[2], key=f"{key}da")
    bw = c5.number_input("Berm width (m)", 0.0, 150.0, default[3], key=f"{key}bw",
                         disabled=nd == 1)
    g = DumpGeometry(int(nd), dh, da, bw if nd > 1 else 0.0,
                     top if (nd > 1 and top > 0) else None)
    ok, viol = dgms_check(g.n_decks, g.deck_height, g.deck_angle, g.berm_width, g.overall_angle,
                          H=g.height)
    if ok:
        st.success(f"DGMS / CMR 2017 Reg. 106 compliant: benches {g.deck_height:.0f} m, deck "
                   f"{g.deck_angle:.1f} deg, overall {g.overall_angle:.1f} deg (limit 33.7).")
    else:
        need = min_berm(g.n_decks, g.deck_height, g.deck_angle, H=g.height)
        tip = f" A berm of at least {need:.0f} m would meet the overall-slope limit." if g.n_decks > 1 else ""
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
    on = st.toggle("Moisture channel (physics-based)", value=False, key=f"{key}mon",
                   help="Moisture changes unit weight, saturation and pore pressure, "
                        "and optionally softens strength beyond a reference moisture.")
    cfg = None; w = None; ru_ext = 0.0
    if on:
        c1, c2, c3 = st.columns(3)
        w = c1.slider("Moisture content w (%)", 0.0, 35.0, 10.0, 0.5, key=f"{key}w")
        cfg = dict(Gs=c2.number_input("Specific gravity Gs", 2.2, 2.9, 2.60, 0.01, key=f"{key}gs"),
                   S_crit=c3.slider("Saturation where pore pressure starts", 0.5, 1.0, 0.80, 0.05,
                                    key=f"{key}sc"),
                   max_phreatic_ratio=1.0)
        c4, c5, c6 = st.columns(3)
        cfg["w_ref"] = c4.number_input("Reference moisture / OMC (%)", 0.0, 30.0, 8.0, key=f"{key}wr")
        cfg["kc"] = c5.number_input("Cohesion loss per 1% above ref (fraction)", 0.0, 0.2, 0.0, 0.005,
                                    key=f"{key}kc", help="0 = off. Calibrate from your shear tests.")
        cfg["kphi"] = c6.number_input("Friction loss per 1% above ref (deg)", 0.0, 2.0, 0.0, 0.05,
                                      key=f"{key}kp", help="0 = off. Calibrate from your shear tests.")
    ru_ext = st.slider("Pore-pressure ratio r_u from piezometers / water table", 0.0, 0.6, 0.0, 0.01,
                       key=f"{key}ru")
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
    res = bishop_fos(geom, 44, 30, 18.63)
    pred = model.predict(row(44, 30, 18.63, 0, geom)).iloc[0]
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(profile_fig(geom, res, "Gevra OCP dump section (SECL): 3 decks x 30 m"),
                        **FULL)
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
- **Acceptance criteria from practice.** TARP classes follow the overall-slope FoS / PoF criteria of Read &
  Stacey (2009) for the chosen consequence of failure.
- **Own physics engine.** Every synthetic label comes from a Bishop simplified solver written for this
  project and checked against the 8 SECL study values, not from FOS-scaling formulas.
- **Method-aware.** Each real FOS is tagged with the software that produced it (RS2 for SECL, FLAC for
  Jagannathpur, FEM for WCL). The model learns a separate correction for each and tells you which
  method its answer corresponds to.
- **Sim-to-real learning.** A monotonic (physics-constrained) surrogate learns slope mechanics from
  simulations; a Gaussian process then learns how *real* sites deviate from it, using only the
  real cases. Validated by leave-one-site-out.
- **Honest uncertainty.** Every prediction carries a conformal prediction interval.
- **Moisture acts through mechanisms** (unit weight, saturation, pore pressure, softening) instead
  of being a column the model can ignore.
- **Monte Carlo with an exact cross-check**, correlated c-phi, reliability index and PoF.
- **Unified TARP** combining deterministic FOS, probability of failure and measured slope velocity,
  with inverse-velocity time-of-failure forecasting.
- **Design envelope and back-analysis** for planning new decks and learning from failures.""")
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
        "way mines do (30 m decks, remainder on top, deck angle up to 37.5 deg, berms keeping the "
        "published overall angle) and re-solved. The 35 and 40 deg cases break the 1V:1.5H rule "
        "and stay in the non-DGMS set with the 2,250 unbenched originals (block C2).")
    chk_path = os.path.join(DATA, "mendeley_solver_check.csv")
    if os.path.exists(chk_path):
        chk = pd.read_csv(chk_path)
        k = st.columns(3)
        k[0].metric("Cases compared", f"{len(chk):,}")
        k[1].metric("Bishop minus published (mean)", f"{chk.diff_pct.mean():+.2f}%")
        k[2].metric("Spread of the difference (SD)", f"{chk.diff_pct.std():.2f}%")
        fch = px.scatter(chk, x="FOS_paper", y="FOS", color=chk.beta_overall_deg.astype(int).astype(str),
                         labels={"FOS_paper": "Published FOS (Slide, Janbu simplified)",
                                 "FOS": "In-house Bishop FOS", "color": "Overall angle"}, opacity=0.6)
        lim = [chk[["FOS", "FOS_paper"]].min().min(), chk[["FOS", "FOS_paper"]].max().max()]
        fch.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#999", dash="dot"),
                                 name="1:1"))
        st.plotly_chart(fch.update_layout(**PLOT, title="Solver check on 2,250 published cases"), **FULL)
        st.caption("A steady +3-4 % offset is expected: Janbu's simplified method without its "
                   "correction factor gives lower FOS than Bishop's method for circular slips.")
    st.subheader("Physics dataset")
    ph = load_physics()
    st.write(ph.groupby("block").agg(rows=("FOS", "size"), FOS_min=("FOS", "min"),
                                     FOS_median=("FOS", "median"), FOS_max=("FOS", "max")))
    st.plotly_chart(px.histogram(ph, x="FOS", color="block", nbins=60, barmode="overlay",
                                 opacity=0.65).update_layout(**PLOT), **FULL)
    st.markdown("<p class='src'>Block A: the published inputs of Sahoo et al. (2025), Sci Rep 15:40985 "
                "(Mendeley data), benched to CMR 2017 Reg. 106. Block B samples around each real "
                "material with COVs from Kulhawy (1992) and literature ranges of Kumar et al. (2023), "
                "Geotech Geol Eng 41:4707. Labels from the in-house Bishop solver (core/lem.py).</p>",
                unsafe_allow_html=True)

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
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Physics hold-out (20%)")
        st.dataframe(pd.DataFrame({"Neural ensemble on dimensionless groups (used)": rep["surrogate_holdout"],
                                   "Gradient boosting, monotonic, raw inputs": rep["mono_holdout"],
                                   "Gradient boosting, raw inputs": rep["free_holdout"]}).T
                     .style.format("{:.4f}"))
        st.caption("Accuracy on simulations only shows the surrogate copies the solver. "
                   "The real test is on the right.")
        viol = pd.Series(rep["monotonic_violations"]) * 100
        st.caption("Physics check - share of sweeps where FOS moved the wrong way: " +
                   ", ".join(f"{k} {v:.1f}%" for k, v in viol.items()))
    with c2:
        st.subheader("Leave-one-site-out on real cases")
        if "loo" in rep:
            st.dataframe(rep["loo"]["table"].style.format("{:.4f}"))
            st.caption("Each dump site is predicted by a model that never saw any of its rows.")
        else:
            st.warning("Need at least 3 complete real rows with use_for_training=True.")
    if "loo" in rep:
        p = rep["loo"]["pred"]
        st.subheader("Error by site and study (MAPE %)")
        by = (p.assign(physics=lambda d: abs(d.physics - d.FOS_real) / d.FOS_real * 100,
                       hybrid=lambda d: abs(d.hybrid - d.FOS_real) / d.FOS_real * 100,
                       real_only=lambda d: abs(d.real_only - d.FOS_real) / d.FOS_real * 100)
              .groupby(["mine", "method"]).agg(rows=("FOS_real", "size"), physics=("physics", "mean"),
                                               hybrid=("hybrid", "mean"), real_only=("real_only", "mean")))
        st.dataframe(by.style.format({"physics": "{:.1f}", "hybrid": "{:.1f}", "real_only": "{:.1f}"}), **FULL)
        st.caption("A study's correction can only be learned when another site from the same study is in "
                   "the training set. Single-site studies fall back to pure physics when held out.")
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
    st.subheader("Transfer test: new geometry and water, same real materials")
    st.caption("The SECL rows share one geometry, so a model trained only on them cannot answer "
               "'what if the dump is higher, steeper or wet?'. Here both models are asked exactly that, "
               "for four DGMS-compliant designs, and compared with the exact Bishop solver.")
    if st.button("Run transfer test (about 10 s)"):
        tr = training_rows(st.session_state.real)
        from core.models import group_of
        tr = tr[group_of(tr) == model.default_method]
        scen = {"1 deck, 30 m, 33 deg": (DumpGeometry(1, 30, 33, 0), 0.0),
                "4 decks x 30 m, 34 deg, 20 m berms": (DumpGeometry(4, 30, 34, 20), 0.0),
                "3 decks, wet r_u 0.25": (DumpGeometry(3, 30, 32, 30), 0.25),
                "2 decks x 20 m, 30 deg": (DumpGeometry(2, 20, 30, 15), 0.0)}
        with st.spinner("Solving exact Bishop for each case ..."):
            out, tab = model.transfer_test(tr, lambda g_, c_, p_, y_, r_: bishop_fos(g_, c_, p_, y_, r_).fos,
                                           scen)
        st.dataframe(tab.style.format("{:.4f}"))
        st.dataframe(out.round(3), **FULL, height=260)
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
    st.subheader("Water"); on, w, cfg, ru_ext = moisture_inputs("p")
    ce, pe, ge, ru, S = effective(c, phi, g, on, w, cfg, ru_ext)
    if on and w > float(saturation_moisture(g, cfg["Gs"])):
        st.warning(f"Moisture {w:.1f} % is above the saturation limit "
                   f"({float(saturation_moisture(g, cfg['Gs'])):.1f} %) for this dry unit weight; it is "
                   "capped at saturation.")
    if on:
        st.caption(f"Effective values: c={ce:.1f} kPa, phi={pe:.1f} deg, bulk unit weight={ge:.2f} kN/m3, "
                   f"saturation={S:.2f}, r_u={ru:.3f}")
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
    tarp_box(lvl, f"FOS {pr.FOS:.2f} against minimum {FOS_MIN:.2f} ({cons} consequence, Read & Stacey 2009)",
             "The lower end of the prediction interval falls below the minimum." if lvl == "Yellow" else "")
    extrap = []
    ph = load_physics()
    for f, v in dict(c_kPa=ce, phi_deg=pe, gamma_kNm3=ge, H_m=geom.height).items():
        lo, hi = ph[f].quantile(0.005), ph[f].quantile(0.995)
        if not lo <= v <= hi: extrap.append(f"{LABELS[f]} = {v:.1f} (trained {lo:.1f}-{hi:.1f})")
    if extrap:
        st.warning("Outside the training range - verify with the exact solver: " + "; ".join(extrap))
    if st.button("Verify with exact Bishop solver and draw the slip surface", type="primary"):
        with st.spinner("Searching slip circles ..."):
            res = bishop_fos(geom, ce, pe, ge, ru)
        st.plotly_chart(profile_fig(geom, res, f"Bishop FOS {res.fos:.3f} ({res.n_circles:,} circles searched)"),
                        **FULL)
        st.session_state.last["bishop"] = res.fos
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
elif PAGE == "Moisture and rainfall":
    st.title("Moisture and rainfall")
    st.markdown("How wetting changes stability through four mechanisms you can switch on and off.")
    c, phi, gd = material_inputs("w", default=(44.0, 30.0, 17.0))
    geom = geometry_inputs("w")
    c1, c2, c3 = st.columns(3)
    Gs = c1.number_input("Gs", 2.2, 2.9, 2.60)
    Sc = c2.slider("Saturation where pore pressure starts", 0.5, 1.0, 0.80, 0.05)
    wref = c3.number_input("Reference moisture / OMC (%)", 0.0, 30.0, 8.0)
    c4, c5 = st.columns(2)
    kc = c4.number_input("Cohesion loss per 1% above ref", 0.0, 0.2, 0.02, 0.005)
    kp = c5.number_input("Friction loss per 1% above ref (deg)", 0.0, 2.0, 0.2, 0.05)
    st.caption("Softening coefficients here are illustrative. Replace them with values fitted to your "
               "own direct-shear tests at several moisture contents. The pore-pressure line assumes the "
               "water table rises through the whole dump once saturation passes the threshold - a "
               "worst case. Real dumps usually saturate from the surface or the base first.")
    w_sat = float(saturation_moisture(gd, Gs))
    ws = np.linspace(2, w_sat, 60)
    st.info(f"With dry unit weight {gd:.1f} kN/m3 and Gs {Gs:.2f}, the voids are completely full of water at "
            f"**{w_sat:.1f} % moisture**. Higher moisture is physically impossible at this density, so the "
            f"curves stop there.")
    curves = {}
    for name, cfg in {"Unit weight only": dict(S_crit=1.01, kc=0, kphi=0),
                      "+ pore pressure": dict(S_crit=Sc, kc=0, kphi=0),
                      "+ strength softening": dict(S_crit=Sc, kc=kc, kphi=kp)}.items():
        ce, pe, ge, ru, S = moisture_state(c, phi, gd, ws, Gs=Gs, w_ref=wref, **cfg)
        d = pd.DataFrame(dict(c_kPa=ce, phi_deg=pe, gamma_kNm3=ge, r_u=ru, **geom.cols()))
        curves[name] = model.predict(d, with_interval=False)["FOS"].values
    fig = go.Figure()
    for (name, y), col in zip(curves.items(), ["#8c8c8c", "#3b7ea1", SLIP]):
        fig.add_trace(go.Scatter(x=ws, y=y, name=name, line=dict(color=col, width=3)))
    fig.add_hline(y=1.0, line_dash="dot", line_color=SLIP)
    fig.add_vline(x=w_sat, line_dash="dash", line_color="#3b7ea1",
                  annotation_text="fully saturated", annotation_position="top left")
    fig.update_layout(**PLOT, title="FOS vs moisture content")
    fig.update_xaxes(title="Moisture content (%)"); fig.update_yaxes(title="FOS")
    st.plotly_chart(fig, **FULL)
    _, _, ge, ru, S = moisture_state(c, phi, gd, ws, Gs=Gs, S_crit=Sc)
    fs = go.Figure([go.Scatter(x=ws, y=S, name="Degree of saturation", line=dict(color="#3b7ea1")),
                    go.Scatter(x=ws, y=ru, name="r_u", line=dict(color=SLIP))])
    fs.update_layout(**PLOT, title="Degree of saturation and r_u")
    fs.update_xaxes(title="Moisture content (%)")
    st.plotly_chart(fs, **FULL)

# ------------------------------------------------------------------
elif PAGE == "Design envelope":
    st.title("Reliability-based design envelope")
    c, phi, g = material_inputs("d")
    st.caption("Every design is benched per CMR 2017 Reg. 106: benches of at most 30 m, and berms at "
               "least wide enough to keep the overall slope within 1V:1.5H.")
    c2, c3, c4 = st.columns(3)
    bw = c2.number_input("Preferred berm width (m)", 10.0, 60.0, 30.0)
    ru = c3.slider("Design r_u", 0.0, 0.5, 0.0, 0.01)
    target = c4.number_input("Target FOS", 1.0, 2.0, float(FOS_MIN), 0.05,
                             help="Defaults to the sidebar acceptance criterion (Read & Stacey 2009).")
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
    best = env[env.ok].sort_values(["H_m", "deck_angle_deg"], ascending=False).groupby("H_m").head(1)
    st.write("Steepest acceptable deck angle for each height:")
    st.dataframe(best[["H_m", "n_decks", "berm_m", "deck_angle_deg", "overall_angle", "FOS_mean", "PoF"]]
                 .sort_values("H_m").style.format({"overall_angle": "{:.1f}", "berm_m": "{:.0f}", "FOS_mean": "{:.2f}",
                                                   "PoF": "{:.2%}"}))

# ------------------------------------------------------------------
elif PAGE == "Back-analysis of failures":
    st.title("Back-analysis of a failed dump")
    st.markdown("Assume FOS = 1 at failure and find every (c, phi) pair consistent with the failure. "
                "Use it to judge whether lab strength values are realistic for the site.")
    geom = geometry_inputs("b", default=(1, 75.0, 43.0, 0.0))
    c1, c2 = st.columns(2)
    g = c1.number_input("Unit weight (kN/m3)", 12.0, 30.0, 20.0)
    ru = c2.slider("r_u at failure", 0.0, 0.6, 0.0, 0.01)
    if st.button("Run back-analysis", type="primary"):
        with st.spinner("Solving FOS = 1 with the exact Bishop solver ..."):
            ba = back_analyse(geom, g, ru, np.arange(10, 42, 2))
        fig = px.line(ba, x="phi_deg", y="c_kPa", markers=True)
        fig.update_traces(line=dict(color=SLIP, width=3))
        lab = training_rows(st.session_state.real)
        fig.add_trace(go.Scatter(x=lab.phi_deg, y=lab.c_kPa, mode="markers", text=lab.mine,
                                 marker=dict(color=TEAL, size=9), name="Real lab strengths"))
        fig.update_layout(**PLOT, title="Strength pairs giving FOS = 1 (above the line = stable)")
        fig.update_xaxes(title="Friction angle (deg)"); fig.update_yaxes(title="Cohesion (kPa)")
        st.plotly_chart(fig, **FULL)
        st.caption("Example: the WCL dump in Kainthola et al. (2011) slipped at 75 m height and 43 deg, an "
                   "unbenched slope that Reg. 106 does not allow. Switch on 'Also train on non-DGMS "
                   "geometries' in the sidebar when studying failures like this.")

# ------------------------------------------------------------------
elif PAGE == "Real-time monitoring":
    st.title("Real-time monitoring and unified TARP")
    st.markdown("<p class='lede'>Slope movement from prisms / radar, the rainfall-updated FOS and an "
                "inverse-velocity failure forecast, combined into one alert level.</p>", unsafe_allow_html=True)
    src = st.radio("Data source", ["Demo feed (synthetic)", "Upload CSV", "Live CSV URL"], horizontal=True)
    st.caption("Required columns: timestamp, displacement_mm. Optional: rainfall_mm, r_u (piezometer). "
               "Total-station, prism or radar exports work once renamed.")
    url, t_true = None, None
    if src == "Upload CSV":
        up = st.file_uploader("Monitoring CSV", type="csv")
        raw = pd.read_csv(up) if up else None
    elif src == "Live CSV URL":
        url = st.text_input("Published CSV link (e.g. Google Sheets > Publish to web > CSV)")
        raw = None
    else:
        raw, t_true = demo_series()
        st.info("Synthetic demonstration data: steady creep, then accelerating (tertiary) creep that ends in "
                f"failure on day {t_true:.0f}. Replace with real prism or radar readings.")
    with st.expander("Processing settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        window = c1.number_input("Velocity window (days)", 0.25, 5.0, 1.0, 0.25,
                                 help="Velocity = slope of a straight line fitted to the displacements in this "
                                      "trailing window. Longer windows suppress survey noise.")
        floor = c2.number_input("Noise floor (mm/day)", 0.0, 2.0, 0.2, 0.05,
                                help="Velocities below this are treated as no measurable movement and left "
                                     "out of the 1/v plot (a daily total-station survey resolves roughly "
                                     "0.1-0.5 mm/day).")
        last_n = c3.number_input("Points for inverse-velocity fit", 5, 100, 20)
        c4, c5, c6 = st.columns(3)
        decay = c4.slider("Antecedent-rain decay per day", 0.5, 0.99, 0.90, 0.01)
        ru_per_mm = c5.number_input("r_u per mm of antecedent rain", 0.0, 0.01, 0.002, 0.0005, format="%.4f")
        ru_max = c6.slider("Maximum r_u", 0.0, 0.6, 0.35, 0.01)
    with st.expander("Material and geometry of the monitored dump"):
        c, phi, g = material_inputs("rt")
        geom = geometry_inputs("rt")

    def render(raw):
        d = process_series(raw, window_days=window, noise_floor=floor)
        if "rainfall_mm" in d:
            dtd = np.r_[0, np.diff(d.t_days)]
            ari = np.zeros(len(d))
            for i in range(1, len(d)):
                ari[i] = ari[i - 1] * decay ** dtd[i] + d.rainfall_mm.iloc[i]
            d["ARI_mm"] = ari
            d["r_u_est"] = np.minimum(ru_max, ru_per_mm * ari)
        if "r_u" in d:
            d["r_u_est"] = np.maximum(d.get("r_u_est", 0), d["r_u"])
        if "r_u_est" not in d:
            d["r_u_est"] = 0.0
        X = pd.DataFrame(dict(c_kPa=c, phi_deg=phi, gamma_kNm3=g, r_u=d.r_u_est, **geom.cols()))
        d["FOS"] = model.predict(X, with_interval=False)["FOS"].values
        cur = d.iloc[-1]
        v_info = classify_velocity(cur.velocity_mm_day)
        f = inverse_velocity_forecast(d, int(last_n))
        overall = worst(vel_to_tarp(v_info["level"]), det_level(cur.FOS))
        if f and f["trend"] == "accelerating" and f["days_left"] < 3:
            overall = "Red"
        k = st.columns(4)
        k[0].metric("Velocity now", f"{cur.velocity_mm_day:.1f} mm/day", v_info["level"], delta_color="off")
        k[1].metric("Monitoring", v_info["frequency"], v_info["method"], delta_color="off")
        k[2].metric("FOS now (with rain)", f"{cur.FOS:.2f}", f"r_u {cur.r_u_est:.2f}", delta_color="off")
        if f and f["trend"] == "accelerating":
            k[3].metric("Failure forecast (1/v)", f"{f['days_left']:.1f} days", f"fit R2 {f['r2']:.2f}",
                        delta_color="off")
        else:
            k[3].metric("Inverse-velocity trend", "Not accelerating")
        tarp_box(overall, "unified TARP (worse of movement class and FOS class; Red if failure is "
                          "forecast within 3 days)", f"Movement: {v_info['response']}")
        # velocity against bands
        top = max(150.0, float(d.velocity_mm_day.max()) * 1.2)
        bands = [0.05] + [b[0] for b in VELOCITY_TARP[:-1]] + [top]
        fig = go.Figure()
        for i, b in enumerate(VELOCITY_TARP):
            fig.add_hrect(y0=bands[i], y1=bands[i + 1], fillcolor=b[2], opacity=0.16, line_width=0,
                          annotation_text=f"{b[1]} ({b[4].lower()})", annotation_position="right",
                          annotation_font_size=11)
        fig.add_trace(go.Scatter(x=d.timestamp, y=d.velocity_mm_day.clip(lower=0.05), name="Velocity",
                                 line=dict(color=INK, width=2)))
        fig.update_layout(**PLOT, title="Slope velocity against the movement TARP bands", showlegend=False)
        fig.update_layout(margin=dict(l=10, r=150, t=48, b=10))
        fig.update_yaxes(title="mm/day (log scale)", type="log", range=[np.log10(0.05), np.log10(top)])
        st.plotly_chart(fig, **FULL)
        c1, c2 = st.columns(2)
        dd = d.dropna(subset=["inv_velocity"])
        fi = go.Figure(go.Scatter(x=dd.t_days, y=dd.inv_velocity, mode="markers",
                                  marker=dict(color=TEAL, size=5), name="1/v"))
        if f and f["trend"] == "accelerating":
            t0 = float(dd.t_days.iloc[-min(int(last_n), len(dd))])
            tt = np.linspace(t0, f["t_fail"], 50)
            fi.add_trace(go.Scatter(x=tt, y=f["slope"] * tt + f["intercept"], name="Linear trend",
                                    line=dict(color=SLIP, dash="dash")))
            fi.add_vline(x=f["t_fail"], line_color=SLIP, line_dash="dot",
                         annotation_text=f"forecast day {f['t_fail']:.1f}")
        if t_true is not None:
            fi.add_vline(x=t_true, line_color="#333", line_dash="dash", annotation_text=f"true day {t_true:.0f}",
                         annotation_position="bottom right")
        fi.update_yaxes(title="1/v (day/mm)", range=[0, float(np.nanpercentile(dd.inv_velocity, 98)) * 1.1])
        c1.plotly_chart(fi.update_layout(**PLOT, title="Inverse velocity (Fukuzono 1985)")
                        .update_xaxes(title="Days since start"), **FULL)
        ff = go.Figure(go.Scatter(x=d.timestamp, y=d.FOS, line=dict(color=TEAL, width=3), name="FOS"))
        ff.add_hline(y=FOS_MIN, line_dash="dot", line_color=TARP_COL["Green"],
                     annotation_text=f"minimum {FOS_MIN:.2f}")
        if "rainfall_mm" in d:
            ff.add_trace(go.Bar(x=d.timestamp, y=d.rainfall_mm, name="Rain (mm)", yaxis="y2",
                                marker_color="#7fa7c9", opacity=0.6))
            ff.update_layout(yaxis2=dict(overlaying="y", side="right", title="Rain (mm)", showgrid=False))
        c2.plotly_chart(ff.update_layout(**PLOT, title="FOS updated with rainfall")
                        .update_yaxes(title="FOS"), **FULL)
        if f and f["trend"] == "accelerating" and t_true is not None:
            validated(f"On this synthetic tertiary-creep record the inverse-velocity forecast is day "
                      f"{f['t_fail']:.1f} against the true failure on day {t_true:.0f} "
                      f"(error {abs(f['t_fail'] - t_true):.1f} days).")
        validated("Rain-to-r_u link (antecedent rainfall index x coefficient) is an assumption: fit the "
                  "coefficient to piezometer readings after storms before relying on the FOS line.", "warn")
        st.dataframe(pd.DataFrame(VELOCITY_TARP, columns=["Upper limit (mm/day)", "Level", "Colour",
                                                          "Method", "Frequency", "Response"])
                     .drop(columns="Colour"), **FULL)
        st.caption("Velocity bands and monitoring frequencies: slope-movement table of the NIT Rourkela "
                   "scientific study notes supplied with this project.")
        st.session_state.last["monitoring"] = dict(velocity=float(cur.velocity_mm_day),
                                                   fos=float(cur.FOS), level=overall)

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

# ------------------------------------------------------------------
elif PAGE == "Report":
    st.title("Report")
    L = st.session_state.last
    if not L:
        st.info("Run a prediction, Monte Carlo or monitoring page first; results collect here.")
    else:
        lines = [f"# DumpSafe AI report - {dt.date.today()}", ""]
        if "FOS" in L:
            lines += ["## Deterministic prediction",
                      f"- Inputs: c={L['c']:.1f} kPa, phi={L['phi']:.1f} deg, gamma={L['gamma']:.2f} kN/m3, "
                      f"r_u={L['ru']:.2f}, H={L['H']:.0f} m, overall angle={L['beta']:.1f} deg, "
                      f"deck angle={L['deck']:.1f} deg",
                      f"- Predicted FOS {L['FOS']:.3f} ({int(alpha*100)}% interval {L['lo']:.2f}-{L['hi']:.2f})",
                      f"- TARP level: {L['level']}"]
            if "bishop" in L: lines.append(f"- Exact Bishop check: {L['bishop']:.3f}")
        if "mc" in L:
            m = L["mc"]
            lines += ["", "## Monte Carlo", f"- {m['n']} realisations, mean FOS {m['mean']:.3f}, "
                      f"PoF {m['pof']*100:.2f}%, reliability index {m['beta_lognormal']:.2f}, "
                      f"TARP {m['level']}"]
        if "monitoring" in L:
            m = L["monitoring"]
            lines += ["", "## Monitoring", f"- Current velocity {m['velocity']:.1f} mm/day, "
                      f"FOS {m['fos']:.2f}, unified TARP {m['level']}"]
        if "loo" in model.report:
            lines += ["", "## Model validation (leave-one-site-out)",
                      model.report["loo"]["table"].round(4).to_string()]
        md = "\n".join(lines)
        st.markdown(md)
        st.download_button("Download report (Markdown)", md, "dumpsafe_report.md")
