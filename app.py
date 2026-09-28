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
from core.moisture import moisture_state
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
  h1 {letter-spacing:-0.5px; font-weight:700;}
  h2, h3 {letter-spacing:-0.2px;}
  .tarp {border-radius:6px; padding:14px 18px; color:white; margin:6px 0 10px 0;}
  .tarp b {font-size:1.25rem;}
  .src {font-size:0.82rem; color:#4a5a5c;}
  div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>""", unsafe_allow_html=True)
PLOT = dict(template="simple_white", margin=dict(l=10, r=10, t=40, b=10),
            font=dict(color=INK), height=380)


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
    "Overview", "Data and sources", "Model lab", "Predict and TARP", "Monte Carlo reliability",
    "Moisture and rainfall", "Design envelope", "Back-analysis of failures",
    "Real-time monitoring", "Report"])

with st.sidebar.expander("TARP thresholds", expanded=False):
    preset = st.selectbox("Preset", ["4-level (default)", "3-level (ICSSMT-26 draft)"])
    if preset.startswith("4"):
        g_thr = st.number_input("Green if FOS >=", value=1.50, step=0.05)
        y_thr = st.number_input("Yellow if FOS >=", value=1.30, step=0.05)
        o_thr = st.number_input("Orange if FOS >=", value=1.00, step=0.05)
    else:
        g_thr = st.number_input("Green if FOS >=", value=1.80, step=0.05)
        y_thr = st.number_input("Yellow if FOS >=", value=1.50, step=0.05)
        o_thr = y_thr
    p_g = st.number_input("Green if PoF < (%)", value=1.0) / 100
    p_y = st.number_input("Yellow if PoF < (%)", value=5.0) / 100
    p_o = st.number_input("Orange if PoF < (%)", value=10.0) / 100
    st.caption("Defaults are editable starting points. Set them to your mine's "
               "approved design criteria / DGMS permission conditions.")
alpha = st.sidebar.select_slider("Prediction-interval coverage", [0.80, 0.90, 0.95], value=0.90)
st.session_state.use_noncompliant = st.sidebar.toggle(
    "Also train on non-DGMS geometries", value=False,
    help="Off: simulations follow CMR 2017 Reg. 106 only (benches <= 30 m, deck <= 37.5 deg, overall "
         "<= 1V:1.5H). On: adds 600 non-compliant dumps so the model also learns the failure zone "
         "(useful for back-analysing failed dumps such as WCL or Jayant).")
model: FOSModel = train_model(st.session_state.real.to_csv(index=False), 1 - alpha,
                              st.session_state.use_noncompliant)
st.sidebar.caption(f"Model trained on {len(load_physics()):,} physics rows + "
                   f"{model.n_real} real rows" + ("" if st.session_state.use_noncompliant
                                                  else " (DGMS-compliant only)"))

ACTIONS = {
    "Green": "Normal operations. Routine monitoring as per plan.",
    "Yellow": "Increase survey frequency, inspect for cracks and seepage, check drainage.",
    "Orange": "Restrict access below the toe, stop dumping on the affected deck, "
              "install extra prisms/piezometers, review geometry with the geotechnical engineer.",
    "Red": "Stop dumping, evacuate the toe and haul road in the influence zone, "
           "flatten or unload the dump, dewater, and redesign.",
}
ORDER = ["Green", "Yellow", "Orange", "Red"]


def fos_level(f):
    return "Green" if f >= g_thr else "Yellow" if f >= y_thr else "Orange" if f >= o_thr else "Red"


def pof_level(p):
    return "Green" if p < p_g else "Yellow" if p < p_y else "Orange" if p < p_o else "Red"


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
    c1, c2, c3, c4 = st.columns(4)
    nd = c1.number_input("Number of decks", 1, 6, default[0], key=f"{key}nd")
    dh = c2.number_input("Deck height (m)", 5.0, 60.0, default[1], key=f"{key}dh")
    da = c3.number_input("Deck angle (deg)", 15.0, 45.0, default[2], key=f"{key}da")
    bw = c4.number_input("Berm width (m)", 0.0, 60.0, default[3], key=f"{key}bw",
                         disabled=nd == 1)
    g = DumpGeometry(int(nd), dh, da, bw if nd > 1 else 0.0)
    ok, viol = dgms_check(g.n_decks, g.deck_height, g.deck_angle, g.berm_width, g.overall_angle)
    if ok:
        st.success(f"DGMS / CMR 2017 Reg. 106 compliant: benches {g.deck_height:.0f} m, deck "
                   f"{g.deck_angle:.1f} deg, overall {g.overall_angle:.1f} deg (limit 33.7).")
    else:
        need = min_berm(g.n_decks, g.deck_height, g.deck_angle)
        tip = f" A berm of at least {need:.0f} m would meet the overall-slope limit." if g.n_decks > 1 else ""
        st.warning("Not DGMS compliant: " + "; ".join(viol) + "." + tip)
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
    c = c1.number_input("Cohesion c (kPa)", 0.0, 300.0, default[0], key=f"{key}c{pick}")
    phi = c2.number_input("Friction angle phi (deg)", 1.0, 50.0, default[1], key=f"{key}p{pick}")
    g = c3.number_input("Unit weight (kN/m3)", 10.0, 32.0, default[2], key=f"{key}g{pick}",
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
- **Coal mines only, DGMS first.** All simulations follow CMR 2017 Reg. 106 (benches up to 30 m, decks up
  to 37.5 deg, overall slope within 1V:1.5H). 20 DGMS-compliant real SECL cases calibrate the model by
  default; the 14-row WCL failed dump (unbenched, non-compliant) is added when you switch on
  non-DGMS geometries. 29 more cited coal cases are waiting for missing values.
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
    st.subheader("Optional: add the Sahoo et al. (2025) Mendeley dataset")
    st.caption("Download it from the paper's data-availability link (data.mendeley.com/datasets/459cbkwwdr/1) "
               "and upload the CSV/XLSX. Columns are mapped automatically; rows are tagged as simulated.")
    md = st.file_uploader("Mendeley file", type=["csv", "xlsx"], key="mend")
    if md is not None:
        m = pd.read_csv(md) if md.name.endswith("csv") else pd.read_excel(md)
        mp = {"Cohesion (kN/m2)": "c_kPa", "Phi (deg)": "phi_deg", "Unit Weight (kN/m3)": "gamma_kNm3",
              "Overall Bench Height": "H_m", "Overall Slope angle": "beta_overall_deg",
              "Natural Moisture content": "moisture_pct", "FOS": "FOS"}
        m = m.rename(columns={k: v for k, v in mp.items() if k in m})
        st.write(f"{len(m)} rows read. First rows:"); st.dataframe(m.head())
        st.info("These are SLIDE-simulated rows (not field cases). Use them as extra physics training "
                "data by appending them to data/physics_dataset.csv with block='C_mendeley'.")
    st.subheader("Physics dataset")
    ph = load_physics()
    st.write(ph.groupby("block").agg(rows=("FOS", "size"), FOS_min=("FOS", "min"),
                                     FOS_median=("FOS", "median"), FOS_max=("FOS", "max")))
    st.plotly_chart(px.histogram(ph, x="FOS", color="block", nbins=60, barmode="overlay",
                                 opacity=0.65).update_layout(**PLOT), **FULL)
    st.markdown("<p class='src'>Block A replicates the parametric design of Sahoo et al. (2025), "
                "Sci Rep 15:40985 (Table 11 geometry, Table 2 ranges). Block B samples around each real "
                "material with COVs from Kulhawy (1992) and literature ranges of Kumar et al. (2023), "
                "Geotech Geol Eng 41:4707. Labels from the in-house Bishop solver (core/lem.py).</p>",
                unsafe_allow_html=True)

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
        st.subheader("Leave-one-mine-out on real cases")
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
    lvl = fos_level(pr.FOS)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted FOS", f"{pr.FOS:.3f}")
    c2.metric(f"{int(alpha*100)}% interval", f"{pr.FOS_lo:.2f} - {pr.FOS_hi:.2f}")
    c3.metric("Physics-only surrogate (Bishop)", f"{pr.FOS_physics:.3f}")
    tarp_box(lvl, f"FOS {pr.FOS:.2f}",
             "Lower bound of the interval is in a worse band - treat with caution."
             if fos_level(pr.FOS_lo) != lvl else "")
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
    lvl = worst(fos_level(r["mean"]), pof_level(r["pof"]))
    tarp_box(lvl, f"mean FOS {r['mean']:.2f}, PoF {r['pof']*100:.2f}%",
             "Level = worse of the FOS band and the PoF band.")
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x="FOS", nbins=80, color_discrete_sequence=[TEAL])
    for thr, col in [(1.0, SLIP), (y_thr, TARP_COL["Yellow"]), (g_thr, TARP_COL["Green"])]:
        fig.add_vline(x=thr, line_dash="dash", line_color=col)
    c1.plotly_chart(fig.update_layout(**PLOT, title="FOS distribution"), **FULL)
    lv = df.FOS.map(fos_level).value_counts().reindex(ORDER).fillna(0) / n * 100
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
               "own direct-shear tests at several moisture contents.")
    ws = np.linspace(2, 30, 57)
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
    target = c4.number_input("Target FOS", 1.0, 2.0, 1.3, 0.05)
    pof_t = st.slider("Maximum acceptable PoF (%)", 0.5, 20.0, 5.0, 0.5) / 100
    heights = np.arange(30, 121, 10); angles = np.arange(24, 41, 1)
    with st.spinner("Evaluating designs ..."):
        env = design_envelope(model, c, phi, g, ru, heights, angles, berm=bw)
    env["ok"] = (env.FOS_mean >= target) & (env.PoF <= pof_t) & env.dgms_ok
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
                             marker=dict(symbol="x", color="#333", size=7), name="Not allowed by Reg. 106"))
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
    src = st.radio("Data source", ["Demo feed (synthetic)", "Upload CSV", "Live CSV URL"],
                   horizontal=True)
    st.caption("Required columns: timestamp, displacement_mm. Optional: rainfall_mm, r_u (piezometer). "
               "Total-station, prism or radar exports work once renamed.")
    url = None
    if src == "Upload CSV":
        up = st.file_uploader("Monitoring CSV", type="csv")
        raw = pd.read_csv(up) if up else None
    elif src == "Live CSV URL":
        url = st.text_input("Published CSV link (e.g. Google Sheets > Publish to web > CSV)")
        raw = None
    else:
        raw = demo_series()
        st.info("Synthetic demonstration data. Replace with real prism/radar readings.")
    c1, c2, c3 = st.columns(3)
    smooth = c1.number_input("Velocity smoothing window (days)", 0.25, 5.0, 1.0, 0.25)
    last_n = c2.number_input("Points for inverse-velocity fit", 5, 100, 20)
    decay = c3.slider("Antecedent-rain decay per day", 0.5, 0.99, 0.90, 0.01)
    c4, c5 = st.columns(2)
    ru_per_mm = c4.number_input("r_u per mm of antecedent rain", 0.0, 0.01, 0.002, 0.0005, format="%.4f",
                                help="Site calibration: fit to piezometer readings after storms.")
    ru_max = c5.slider("Maximum r_u", 0.0, 0.6, 0.35, 0.01)
    with st.expander("Material and geometry of the monitored dump"):
        c, phi, g = material_inputs("rt")
        geom = geometry_inputs("rt")

    def render(raw):
        d = process_series(raw, smooth_days=smooth)
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
        overall = worst(vel_to_tarp(v_info["level"]), fos_level(cur.FOS))
        if f and f["trend"] == "accelerating" and f["days_left"] < 3:
            overall = "Red"
        k = st.columns(4)
        k[0].metric("Velocity now", f"{cur.velocity_mm_day:.1f} mm/day", v_info["level"])
        k[1].metric("Monitoring", v_info["frequency"], v_info["method"])
        k[2].metric("FOS now (with rain)", f"{cur.FOS:.2f}", f"r_u {cur.r_u_est:.2f}")
        if f and f["trend"] == "accelerating":
            k[3].metric("Inverse-velocity failure forecast", f"{f['days_left']:.1f} days",
                        f"fit R2 {f['r2']:.2f}")
        else:
            k[3].metric("Inverse-velocity trend", "Not accelerating")
        tarp_box(overall, "unified TARP", f"Movement: {v_info['response']}")
        fig = go.Figure()
        bands = [0] + [b[0] for b in VELOCITY_TARP[:-1]] + [max(150, d.velocity_mm_day.max() * 1.1)]
        for i, b in enumerate(VELOCITY_TARP):
            fig.add_hrect(y0=bands[i], y1=bands[i + 1], fillcolor=b[2], opacity=0.12, line_width=0)
        fig.add_trace(go.Scatter(x=d.timestamp, y=d.velocity_mm_day, name="Velocity",
                                 line=dict(color=INK, width=2)))
        fig.update_layout(**PLOT, title="Slope velocity against movement TARP bands")
        fig.update_yaxes(title="mm/day", type="log", range=[-1, np.log10(bands[-1])])
        st.plotly_chart(fig, **FULL)
        c1, c2 = st.columns(2)
        fi = go.Figure(go.Scatter(x=d.t_days, y=d.inv_velocity, mode="markers",
                                  marker=dict(color=TEAL, size=5), name="1/v"))
        if f and f["trend"] == "accelerating":
            tt = np.linspace(d.t_days.iloc[-int(last_n)], f["t_fail"], 50)
            fi.add_trace(go.Scatter(x=tt, y=f["slope"] * tt + f["intercept"], name="Linear trend",
                                    line=dict(color=SLIP, dash="dash")))
        c1.plotly_chart(fi.update_layout(**PLOT, title="Inverse velocity (Fukuzono)")
                        .update_xaxes(title="Days").update_yaxes(title="day/mm"),
                        **FULL)
        ff = go.Figure(go.Scatter(x=d.timestamp, y=d.FOS, line=dict(color=TEAL, width=3), name="FOS"))
        if "rainfall_mm" in d:
            ff.add_trace(go.Bar(x=d.timestamp, y=d.rainfall_mm, name="Rain (mm)", yaxis="y2",
                                marker_color="#7fa7c9", opacity=0.6))
            ff.update_layout(yaxis2=dict(overlaying="y", side="right", title="Rain (mm)"))
        c2.plotly_chart(ff.update_layout(**PLOT, title="FOS updated with rainfall"),
                        **FULL)
        st.dataframe(pd.DataFrame(VELOCITY_TARP, columns=["Upper limit (mm/day)", "Level", "Colour",
                                                          "Method", "Frequency", "Response"])
                     .drop(columns="Colour"), **FULL)
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
