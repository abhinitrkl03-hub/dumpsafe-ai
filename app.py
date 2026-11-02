"""
DumpSafe AI - Trigger Action Response Plan (TARP) dashboard for coal-mine OB dumps.
Run:  streamlit run app.py
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.lem import DumpGeometry
from core.dgms import check as dgms_check
from core.training import load as load_saved, fit_model
from core.monitoring import (process_series, add_moving_averages, onset_of_acceleration, iv_window,
                             classify_velocity, demo_series, demo_storm, VELOCITY_TARP)
from core import tarp as T
from core.backanalysis import velocity_record, movement_summary
import tarp_pages as TP

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
st.set_page_config(page_title="DumpSafe AI - TARP for dump slopes", page_icon="⛰️", layout="wide")
_v = tuple(int(x) for x in getattr(st, "__version__", "1.40.0").split(".")[:2])
FULL = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
COL = T.COL
INK, TEAL = "#1C2B2D", "#2F6B6A"
PLOT = dict(template="simple_white", margin=dict(l=10, r=10, t=44, b=10), height=330,
            font=dict(color=INK, family="IBM Plex Sans, sans-serif", size=13), title_font=dict(size=15))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label {font-family:'IBM Plex Sans', system-ui, sans-serif;}
.block-container {padding-top:1.6rem;}
h1 {font-weight:700; letter-spacing:-0.5px; color:#1C2B2D; margin-bottom:0;}
.sub {color:#4a5a5c; margin:0 0 0.8rem 0;}
.banner {border-radius:14px; padding:16px 20px; color:#fff; margin:4px 0 10px 0;}
.banner .lv {font-size:1.7rem; font-weight:700; letter-spacing:0.3px;}
.banner .ln {font-size:0.95rem; margin-top:4px; line-height:1.45;}
.card {background:#fff; border:1px solid #dfe5e3; border-radius:12px; padding:10px 12px; height:132px;
       box-shadow:0 1px 2px rgba(28,43,45,0.05);}
.card .t {font-size:0.76rem; color:#4a5a5c; text-transform:uppercase; letter-spacing:0.4px;}
.card .v {font-size:1.05rem; font-weight:600; color:#1C2B2D; margin:4px 0 2px 0;}
.card .n {font-size:0.78rem; color:#5b6a6c;}
.pill {display:inline-block; border-radius:999px; padding:1px 10px; color:#fff; font-size:0.78rem; font-weight:600;}
.big {font-size:2.6rem; font-weight:700; color:#1C2B2D; line-height:1.1;}
.muted {color:#5b6a6c; font-size:0.85rem;}
.box {background:#f4f7f6; border-radius:12px; padding:12px 16px; margin:6px 0;}
div[data-testid="stMetricValue"] {font-variant-numeric:tabular-nums;}
section[data-testid="stSidebar"] {background:#eef2f0;}
</style>""", unsafe_allow_html=True)


def pill(level):
    return f"<span class='pill' style='background:{COL.get(level, '#8a9597')}'>{level}</span>"


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner="Loading the AI FOS engine ...")
def engine():
    m, _ = load_saved(False, 0.10)
    if m is None:       # saved engine missing or built with another scikit-learn version
        m = fit_model(pd.read_csv(os.path.join(DATA, "real_cases.csv")),
                      pd.read_csv(os.path.join(DATA, "physics_dataset.csv")), False, 0.10)
    return m


@st.cache_data
def presets():
    r = pd.read_csv(os.path.join(DATA, "real_cases.csv"))
    out = {}
    for x in r[r.status == "complete"].itertuples():
        nm = x.mine.split(" (")[0]
        if nm in out:
            continue
        out[nm] = dict(c=float(x.c_kPa), phi=float(x.phi_deg), g=float(x.gamma_kNm3),
                       fos=float(x.FOS), method=str(x.fos_method).split(" (")[0],
                       nd=int(x.n_decks), dh=float(x.deck_height_m), da=float(x.deck_angle_deg),
                       bw=float(0 if pd.isna(x.berm_width_m) else x.berm_width_m), src=str(x.source)[:60])
    return out


@st.cache_data
def csv(name):
    return pd.read_csv(os.path.join(DATA, "tarp", name))


@st.cache_data
def movement_record(src, upload_bytes, window, floor):
    if src.startswith("Demo A"):
        raw, tf = demo_series()
    elif src.startswith("Demo B"):
        raw, tf = demo_storm()
    else:
        import io
        raw, tf = pd.read_csv(io.BytesIO(upload_bytes)), None
    d = add_moving_averages(process_series(raw, window_days=window, noise_floor=floor), 1.0, 5.0)
    return d, tf


@st.cache_data
def movement_history(d: pd.DataFrame, fos_min, vel_thr):
    d = d.copy()
    d["FOS"] = 9.9          # FOS and rain are judged on their own tabs
    h = T.level_history(d, fos_min, vel_thr=vel_thr)
    h["Movement"] = [T.worst(a, b) for a, b in zip(h["T4 Velocity"], h["T5 Acceleration"])]
    return h


def X(c, phi, g, ru, geom):
    return pd.DataFrame([dict(c_kPa=c, phi_deg=phi, gamma_kNm3=g, r_u=ru, **geom.cols())])


model = engine()
if "thr" not in st.session_state:
    st.session_state.thr = TP.default_thresholds()
THR = st.session_state.thr

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## ⛰️ DumpSafe AI")
st.sidebar.caption("Trigger Action Response Plan for overburden dumps of Indian coal mines")
P = presets()
site = st.sidebar.selectbox("Dump / mine", list(P) + ["Custom dump"],
                            help="Material and geometry of the mines with published studies; edit them on tab 1.")
pre = P.get(site)
dtype = st.sidebar.radio("Type of dump", ["Permanent / long-standing", "Working / temporary"],
                         help="DGMS circular for the scientific study under Reg. 106(2) of CMR 2017, para 3.2: "
                              "minimum FOS 1.50 for permanent and 1.30 for other slopes.")
FOS_MIN = 1.50 if dtype.startswith("Perm") else 1.30
people = st.sidebar.toggle("People / haul road / buildings below the toe", value=True,
                           help="Consequence step: Yellow is treated as Orange, Orange as Red.")
have = st.sidebar.multiselect("Instruments at the mine", list(TP.SENSORS),
                              default=["Total station + prisms", "Rain gauge (or IMD station)"])
st.sidebar.markdown("---")
st.sidebar.caption("Levels: " + " ".join(pill(x) for x in T.ORDER), unsafe_allow_html=True)

# ------------------------------------------------------------------ header + status placeholder
st.markdown("<h1>Trigger Action Response Plan</h1><p class='sub'>One place for every warning sign of an OB dump: "
            "design FOS and live FOS from the AI engine, rain and pore pressure, movement and acceleration, and "
            "field signs. The worst one sets the level and the action.</p>", unsafe_allow_html=True)
status = st.container()

tabs = st.tabs(["① Design  ·  T1", "② Water & rain  ·  T2 T3", "③ Movement  ·  T4 T5", "④ Field signs  ·  T6",
                "⑤ Response plan", "⑥ Why this TARP"])

# ================================================================== 1 Design
with tabs[0]:
    k = site.replace(" ", "")
    L, R = st.columns([1, 1.25], gap="large")
    with L:
        st.markdown("**Material**")
        a, b, c_ = st.columns(3)
        c = a.number_input("Cohesion c (kPa)", 0.0, 300.0, pre["c"] if pre else 40.0, 1.0, key=f"c{k}")
        phi = b.number_input("Friction φ (deg)", 5.0, 45.0, pre["phi"] if pre else 28.0, 0.5, key=f"p{k}")
        g = c_.number_input("Unit weight γ (kN/m³)", 14.0, 28.0, pre["g"] if pre else 19.0, 0.1, key=f"g{k}")
        st.markdown("**Geometry**")
        a, b = st.columns(2)
        nd = a.slider("Number of decks", 1, 4, pre["nd"] if pre and pre["nd"] <= 4 else 3, key=f"n{k}")
        dh = b.slider("Deck height (m)", 10.0, 30.0, min(pre["dh"], 30.0) if pre else 30.0, 1.0, key=f"h{k}")
        a, b = st.columns(2)
        da = a.slider("Deck angle (deg)", 20.0, 37.5, min(pre["da"], 37.5) if pre else 32.0, 0.5, key=f"a{k}")
        bw = b.slider("Bench width (m)", 0.0, 40.0, pre["bw"] if pre else 30.0, 1.0, key=f"b{k}", disabled=nd == 1)
        ru_d = st.slider("Design pore-pressure ratio r_u", 0.0, 0.45, 0.0, 0.01,
                         help="0 = drained dump. Today's water is handled on tab 2.")
    geom = DumpGeometry(int(nd), dh, da, bw if nd > 1 else 0.0)
    t0 = time.perf_counter()
    des = model.predict(X(c, phi, g, ru_d, geom)).iloc[0]
    ms = (time.perf_counter() - t0) * 1000
    t1 = T.fos_trigger(des.FOS, FOS_MIN, des.FOS_lo)
    ok, viol = dgms_check(geom.n_decks, geom.deck_height, geom.deck_angle, geom.berm_width, geom.overall_angle,
                          H=geom.height)
    with R:
        st.markdown(f"<div class='box'><span class='muted'>AI design FOS ({ms:.0f} ms)</span><br>"
                    f"<span class='big'>{des.FOS:.2f}</span> &nbsp;{pill(t1)}<br>"
                    f"<span class='muted'>90 % range {des.FOS_lo:.2f} - {des.FOS_hi:.2f} &nbsp;·&nbsp; DGMS minimum "
                    f"{FOS_MIN:.2f} &nbsp;·&nbsp; height {geom.height:.0f} m, overall {geom.overall_angle:.1f}°"
                    f"</span></div>", unsafe_allow_html=True)
        if pre and abs(pre["dh"] * pre["nd"] - geom.height) < 1 and abs(pre["da"] - da) < 0.1 and pre["bw"] == bw \
                and abs(pre["c"] - c) < 1e-6 and abs(pre["phi"] - phi) < 1e-6 and abs(pre["g"] - g) < 1e-6 and ru_d == 0:
            st.markdown(f"<span class='muted'>Same section in the published study ({pre['method']}): FOS "
                        f"<b>{pre['fos']:.2f}</b></span>", unsafe_allow_html=True)
        st.markdown(("✅ Geometry within CMR 2017 Reg. 106" if ok else "⚠️ " + "; ".join(viol)))
        ns = list(range(1, 5))
        ph = model.predict(pd.concat([X(c, phi, g, ru_d, DumpGeometry(n_, dh, da, bw if n_ > 1 else 0.0))
                                      for n_ in ns], ignore_index=True))
        fig = go.Figure(go.Bar(x=[f"{n_ * dh:.0f} m" for n_ in ns], y=ph.FOS,
                               marker_color=[COL[T.fos_trigger(f_, FOS_MIN, l_)] for f_, l_ in zip(ph.FOS, ph.FOS_lo)],
                               text=[f"{f_:.2f}" for f_ in ph.FOS], textposition="outside"))
        fig.add_hline(y=FOS_MIN, line_dash="dot", line_color=COL["Orange"], annotation_text=f"minimum {FOS_MIN:.2f}")
        fig.update_layout(**PLOT, title="How high can this dump be raised?", showlegend=False)
        fig.update_yaxes(title="FOS", range=[0, max(2.2, float(ph.FOS.max()) * 1.15)])
        st.plotly_chart(fig, **FULL)
    with st.expander("Where does this FOS come from?"):
        st.markdown(
            "- **Physics first.** 3,995 dump sections that obey CMR 2017 Reg. 106 were solved with the Bishop "
            "simplified method (the method used in Slide2 and most Indian scientific studies). Our solver matches "
            "Slide2 and the published SECL results. These solutions teach the AI how a dump behaves.\n"
            "- **Corrected to real studies.** 34 FOS values from field studies (SECL mines with RS2, Jagannathpur "
            "with FLAC, WCL with FEM) teach the AI how real dumps differ from the physics.\n"
            "- **Checked on unseen sites.** Each mine was left out in turn and predicted: average error about 10 %. "
            "The 90 % range is calibrated on those errors, so Yellow already allows for the AI's own error.\n"
            "- **Why not train on real failures only?** Fewer than 50 analysed dump sections are published for "
            "Indian coal mines: too few for an AI alone. Physics plus real correction is the standard way round "
            "this.\n"
            "- **Why AI and not just Slide2?** A Slide2 run takes minutes per section; the AI answers in "
            "milliseconds, so the TARP can be re-evaluated at every reading and hundreds of what-if cases (tab 2) "
            "can set the site's trigger values.")

# ================================================================== 2 Water & rain
with tabs[1]:
    L, R = st.columns([1, 1.25], gap="large")
    with L:
        st.markdown("**Today's readings**")
        rain24 = st.number_input("Rain in the last 24 h (mm)", 0.0, 600.0, 0.0, 5.0)
        rain7 = st.number_input("Rain in the last 7 days (mm)", 0.0, 2000.0, 0.0, 10.0,
                                help="Antecedent rain: water that has soaked into the dump.")
        use_pz = st.toggle("Piezometer reading available", value="Vibrating-wire piezometers" in have)
        a, b = st.columns(2)
        z_tip = a.number_input("Piezometer tip depth (m)", 1.0, 100.0, 10.0, 0.5)
        head_now = b.number_input("Water head above the tip (m)", 0.0, 100.0, 0.0, 0.1, disabled=not use_pz)
        with st.expander("Rain-to-pore-pressure link"):
            kappa = st.number_input("r_u per mm of 7-day rain", 0.0001, 0.01, 0.0012, 0.0001, format="%.4f",
                                    help="Learned from paired rain-gauge and piezometer records once available; "
                                         "this default is an assumption.")
    ru_now = min(0.45, head_now * 9.81 / (g * z_tip)) if use_pz else min(0.45, kappa * rain7)
    rus = np.round(np.arange(0.0, 0.451, 0.01), 2)
    sw = model.predict(pd.concat([X(c, phi, g, r_, geom) for r_ in rus], ignore_index=True))
    live = model.predict(X(c, phi, g, max(ru_now, ru_d), geom)).iloc[0]
    ru_y, ru_o, ru_r = (TP._cross(rus, sw.FOS_lo.values, FOS_MIN), TP._cross(rus, sw.FOS.values, FOS_MIN),
                        TP._cross(rus, sw.FOS.values, 1.0))
    t2 = T.fos_trigger(live.FOS, FOS_MIN)
    rain_lim = tuple((r_ / kappa) if r_ is not None else 1e9 for r_ in (ru_y, ru_o, ru_r))
    t3 = T.worst(T.rain_trigger(rain24, THR["rain"]), T.ari_trigger(rain7, rain_lim))
    head = lambda r_: "-" if r_ is None else f"{r_ * g * z_tip / 9.81:.1f} m"
    with R:
        a, b, c3 = st.columns(3)
        a.markdown(f"<div class='box'><span class='muted'>Live FOS (T2)</span><br><span class='big'>"
                   f"{live.FOS:.2f}</span><br>{pill(t2)}</div>", unsafe_allow_html=True)
        b.markdown(f"<div class='box'><span class='muted'>r_u now</span><br><span class='big'>{ru_now:.2f}</span><br>"
                   f"<span class='muted'>{'piezometer' if use_pz else 'from 7-day rain'}</span></div>",
                   unsafe_allow_html=True)
        c3.markdown(f"<div class='box'><span class='muted'>Rainfall (T3)</span><br><span class='big'>{rain24:.0f}"
                    f"</span> <span class='muted'>mm/24 h</span><br>{pill(t3)}</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rus, y=sw.FOS_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=rus, y=sw.FOS_lo, fill="tonexty", fillcolor="rgba(47,107,106,0.16)",
                                 line=dict(width=0), name="90 % range"))
        fig.add_trace(go.Scatter(x=rus, y=sw.FOS, line=dict(color=INK, width=3), name="AI FOS"))
        fig.add_trace(go.Scatter(x=[ru_now], y=[live.FOS], mode="markers", marker=dict(size=13, color=COL[t2]),
                                 name="today"))
        fig.add_hline(y=FOS_MIN, line_dash="dot", line_color=COL["Orange"])
        fig.add_hline(y=1.0, line_dash="dot", line_color=COL["Red"])
        fig.update_layout(**PLOT, title="FOS as water builds up in this dump", legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(title="Pore-pressure ratio r_u")
        fig.update_yaxes(title="FOS")
        st.plotly_chart(fig, **FULL)
    st.markdown("**Water and rain trigger values for this dump** (found by the AI engine)")
    st.dataframe(pd.DataFrame([
        [lv, "-" if r_ is None else f"{r_:.2f}", head(r_), "not reached" if r_ is None else f"{r_ / kappa:.0f} mm",
         f"{THR['rain'][i]:g} mm"]
        for i, (lv, r_) in enumerate([("Yellow", ru_y), ("Orange", ru_o), ("Red", ru_r)])],
        columns=["Level", "r_u", f"Piezometer head ({z_tip:g} m tip)", "7-day rain", "24-h rain (IMD)"]),
        hide_index=True, **FULL)
    if ru_r is None:
        st.caption("Red is not reached even close to full saturation: water alone is unlikely to fail this dump; "
                   "movement and field signs still apply.")

# ================================================================== 3 Movement
with tabs[2]:
    a, b = st.columns([2, 1])
    src = a.radio("Monitoring record", ["Demo B: stable dump (synthetic)", "Demo A: creep to failure (synthetic)",
                                        "Upload prism / radar / GNSS CSV"], horizontal=True)
    up = b.file_uploader("CSV: timestamp, displacement_mm", type="csv") if src.startswith("Upload") else None
    T4, T5, mv_note, h_first = "No data", "No data", "", None
    if src.startswith("Demo") or up is not None:
        d_all, tf = movement_record(src, up.getvalue() if up else b"", 1.0, 0.2)
        t_end = float(d_all.t_days.iloc[-1])
        upto = st.slider("Show the dump as it stood on day", float(d_all.t_days.iloc[min(12, len(d_all) - 1)]),
                         t_end, t_end, 0.25)
        d = d_all[d_all.t_days <= upto].reset_index(drop=True)
        cur = d.iloc[-1]
        onset = onset_of_acceleration(d, 1.3, 2.0, 0.2, 5.0)
        win = iv_window(d, onset, 0.2, fit_days=5.0) if onset is not None else None
        T4 = T.vel_trigger(cur.velocity_mm_day, THR["vel"])
        T5 = T.accel_trigger(onset, win, cur.velocity_mm_day)
        vi = classify_velocity(cur.velocity_mm_day)
        m = st.columns(4)
        m[0].metric("Velocity (T4)", f"{cur.velocity_mm_day:.1f} mm/day", T4, delta_color="off")
        m[1].metric("Survey now", vi["frequency"], vi["method"], delta_color="off")
        m[2].metric("Acceleration (T5)", "started" if onset is not None else "none",
                    f"since day {onset:.1f}" if onset is not None else T5, delta_color="off")
        m[3].metric("Failure window", f"day {win['t_early']:.0f}-{win['t_late']:.0f}" if win else "-",
                    f"{max(win['days_early'], 0):.1f} days left" if win else "not accelerating", delta_color="off")
        top = max(120.0, float(d_all.velocity_mm_day.max()) * 1.3)
        bands = [0.05] + list(THR["vel"]) + [top]
        fig = go.Figure()
        for j, lv in enumerate(T.ORDER):
            fig.add_hrect(y0=bands[j], y1=bands[j + 1], fillcolor=COL[lv], opacity=0.13, line_width=0)
        fig.add_trace(go.Scatter(x=d.t_days, y=d.velocity_mm_day.clip(lower=0.05), name="velocity",
                                 line=dict(color="#9aa5a6", width=1)))
        fig.add_trace(go.Scatter(x=d.t_days, y=d.v_sma.clip(lower=0.05), name="1-day average",
                                 line=dict(color=INK, width=2)))
        fig.add_trace(go.Scatter(x=d.t_days, y=d.v_lma.clip(lower=0.05), name="5-day average",
                                 line=dict(color=TEAL, width=2, dash="dash")))
        if onset is not None:
            fig.add_vline(x=onset, line_dash="dot", line_color=COL["Red"], annotation_text="acceleration starts")
        if tf:
            fig.add_vline(x=tf, line_color="#333", annotation_text="actual failure")
        fig.update_layout(**PLOT, title="Slope velocity against the TARP bands",
                          legend=dict(orientation="h", y=-0.2))
        fig.update_xaxes(title="Day", range=[0, max(t_end, tf or 0) + 1])
        fig.update_yaxes(type="log", title="mm/day", range=[np.log10(0.05), np.log10(top)])
        st.plotly_chart(fig, **FULL)
        h = movement_history(d_all, FOS_MIN, tuple(THR["vel"]))
        fo, fr = T.first_time(h, "Movement", "Orange"), T.first_time(h, "Movement", "Red")
        vo, vr = T.first_time(h, "T4 Velocity", "Orange"), T.first_time(h, "T4 Velocity", "Red")
        f_ = lambda x: "never" if x is None else f"day {x:.1f}"
        mv_note = (f"Velocity + acceleration: Orange {f_(fo)}, Red {f_(fr)}.  Velocity alone: Orange {f_(vo)}, "
                   f"Red {f_(vr)}." + (f"  Actual failure: day {tf:.0f}." if tf else ""))
        st.markdown(f"<div class='box'>⏱️ <b>Warning time on this record.</b> {mv_note}</div>", unsafe_allow_html=True)
        if src.startswith("Demo"):
            st.caption("Synthetic record, used to show how the triggers behave. Upload real prism or radar readings "
                       "to use your own.")
    else:
        st.info("Upload a monitoring CSV, or pick a demo record.")

# ================================================================== 4 Field signs
with tabs[3]:
    st.markdown("Tick what the shift overman or geotechnical engineer saw on the dump today.")
    seen = []
    cols = st.columns(3)
    for j, lv in enumerate(["Yellow", "Orange", "Red"]):
        cols[j].markdown(f"{pill(lv)}", unsafe_allow_html=True)
        for s_, l_ in T.VISUAL_SIGNS:
            if l_ == lv and cols[j].checkbox(s_, key=f"vs{s_}"):
                seen.append(s_)
    t6 = T.visual_trigger(seen)
    st.markdown(f"Field signs level: {pill(t6)}", unsafe_allow_html=True)
    st.caption("Signs taken from Indian failures: cracks for months before Rajmahal (2016), cracks widening from "
               "1-2 mm to 5 cm in three days at Juna Kunada (2017), toe heave at MKD-III.")

# ================================================================== combine
trig = {"T1 Design FOS": t1, "T2 Live FOS": t2, "T3 Rainfall": t3, "T4 Velocity": T4, "T5 Acceleration": T5,
        "T6 Field signs": t6}
final, gov, note, missing = T.evaluate(trig, people)
act = next(a_ for a_ in T.ACTION_TABLE if a_["Level"] == final)
with status:
    st.markdown(f"<div class='banner' style='background:{COL[final]}'><div class='lv'>{final}</div>"
                f"<div class='ln'><b>{act['Meaning']}</b>{' - ' + ', '.join(gov) if gov else ''}"
                f"{' (' + note + ')' if note else ''}<br>Action: {act['Action']}<br>Who: {act['Responsible']} · "
                f"When: {act['Response_time']}</div></div>", unsafe_allow_html=True)
    vals = {"T1 Design FOS": f"FOS {des.FOS:.2f}", "T2 Live FOS": f"FOS {live.FOS:.2f} (r_u {ru_now:.2f})",
            "T3 Rainfall": f"{rain24:.0f} mm / 24 h", "T4 Velocity": "-" if T4 == "No data" else
            f"{cur.velocity_mm_day:.1f} mm/day", "T5 Acceleration": "-" if T5 == "No data" else
            ("accelerating" if T5 != "Green" else "steady"), "T6 Field signs": f"{len(seen)} sign(s)"}
    stage = {"T1 Design FOS": "AI · design", "T2 Live FOS": "AI · water", "T3 Rainfall": "rain gauge",
             "T4 Velocity": "prisms / radar", "T5 Acceleration": "prisms / radar", "T6 Field signs": "inspection"}
    cc = st.columns(6)
    for col_, (kk, lv) in zip(cc, trig.items()):
        col_.markdown(f"<div class='card' style='border-top:5px solid {COL.get(lv, '#8a9597')}'><div class='t'>{kk}"
                      f"</div><div class='v'>{vals[kk]}</div>{pill(lv)}<div class='n'>{stage[kk]}</div></div>",
                      unsafe_allow_html=True)

# ================================================================== 5 Response plan
with tabs[4]:
    st.markdown(f"Current TARP level: {pill(final)}" + (f" &nbsp; governed by {', '.join(gov)}" if gov else "")
                + (f" &nbsp; · {note}" if note else ""), unsafe_allow_html=True)
    f2 = lambda r_: "not reached" if r_ is None else f"r_u {r_:.2f} / head {head(r_)}"
    v = THR["vel"]
    active = lambda keys: "yes" if any(s_ in have for s_ in keys) else "NO - install"
    sheet = pd.DataFrame([
        ["T1 Design FOS", vals["T1 Design FOS"], t1, f"90 % range below {FOS_MIN:.2f}", f"FOS < {FOS_MIN:.2f}",
         "FOS < 1.0", "yes"],
        ["T2 Live FOS", vals["T2 Live FOS"], t2, f2(ru_y), f2(ru_o), f2(ru_r),
         active(["Vibrating-wire piezometers", "Rain gauge (or IMD station)"])],
        ["T3 Rainfall", vals["T3 Rainfall"], t3, f"{THR['rain'][0]:g} mm/24 h", f"{THR['rain'][1]:g} mm/24 h",
         f"{THR['rain'][2]:g} mm/24 h", active(["Rain gauge (or IMD station)"])],
        ["T4 Velocity", vals["T4 Velocity"], T4, f"{v[0]:g} mm/day", f"{v[1]:g} mm/day", f"{v[2]:g} mm/day",
         active(["Total station + prisms", "Slope stability radar (SSR)", "GNSS monitors",
                 "Crack meters / extensometers", "InSAR (Sentinel-1)"])],
        ["T5 Acceleration", vals["T5 Acceleration"], T5, "acceleration starts", "failure < 7 days",
         "failure < 3 days", active(["Total station + prisms", "Slope stability radar (SSR)", "GNSS monitors"])],
        ["T6 Field signs", vals["T6 Field signs"], t6, "new cracks, seepage", "widening cracks, toe bulge",
         "scarp, visible movement", "yes"],
    ], columns=["Trigger", "Now", "Level", "Yellow", "Orange", "Red", "Instrument"])

    def colour(sv):
        return [f"background-color:{COL[x]};color:white;font-weight:600" if x in COL and x != "No data" else ""
                for x in sv]
    st.dataframe(sheet.style.apply(colour, subset=["Level"]), hide_index=True, **FULL)
    acts = pd.DataFrame(T.ACTION_TABLE).rename(columns={"Response_time": "Response time"})
    st.markdown("**What to do at each level**")
    st.dataframe(acts.style.apply(lambda r_: ["font-weight:700;background-color:#fff3d6" if r_.Level == final
                                              else "" for _ in r_], axis=1), hide_index=True, **FULL)
    st.markdown("**Readings at each level**")
    st.dataframe(pd.DataFrame([
        ["Green", "Monthly prism survey; rain gauge daily; inspection every shift"],
        ["Yellow", "Survey weekly to daily; piezometers daily; crack pegs every shift"],
        ["Orange", "Survey daily or continuous radar / GNSS; piezometers every few hours"],
        ["Red", "Continuous radar / GNSS from a safe point; nobody in the zone"],
    ], columns=["Level", "Readings"]), hide_index=True, **FULL)
    html = TP._tarp_html(site, des, t1, FOS_MIN, dtype + " dump", sheet.drop(columns=["Now", "Level"]),
                         acts, people, c, phi, g, geom, ru_d, z_tip, kappa)
    st.download_button("⬇️ Download this dump's TARP (print to PDF)", html,
                       f"TARP_{site.replace(' ', '_')}.html", "text/html")

# ================================================================== 6 Why this TARP
with tabs[5]:
    st.markdown("""
<div class='box'><b>Not only FOS.</b> Indian dumps have failed for different reasons, so the TARP watches four
stages, each with its own trigger:<br>
① <b>Before</b> - is the design safe? (AI design FOS) &nbsp; ② <b>Water</b> - is rain or pore pressure lowering it?
(AI live FOS, rain) &nbsp; ③ <b>Movement</b> - is the dump moving or speeding up? (velocity, time to failure)
&nbsp; ④ <b>Field</b> - can it be seen? (cracks, bulging, seepage)</div>""", unsafe_allow_html=True)
    with st.expander("What is new in this work", expanded=True):
        st.markdown(
            "- Indian dump studies so far give **either** a design FOS **or** a monitoring alarm. This TARP joins "
            "design FOS, live FOS with water, rainfall, velocity, acceleration and field signs, with actions, "
            "responsibilities and response times.\n"
            "- The **AI engine answers in milliseconds**, so it does what a Slide2 run cannot: it re-checks the "
            "dump at every reading and finds each dump's own trigger values (pore pressure, piezometer head, "
            "rain, height).\n"
            "- Every threshold comes from **DGMS, the NIT Rourkela scientific study, IMD or a back-analysed "
            "failure**.\n"
            "- The thresholds are built from **Indian failures** (13 events) and checked on them.")
    with st.expander("Why Indian coal-mine dumps failed (13 reported events)"):
        hh = csv("failure_history_india.csv")
        st.dataframe(hh[["date", "mine", "company", "deaths", "cause_group", "precursors", "tarp_would_fire"]]
                     .rename(columns={"cause_group": "cause", "precursors": "warning signs",
                                      "tarp_would_fire": "trigger that would fire"}), hide_index=True, **FULL)
        st.markdown(
            "1. **Two kinds of failure:** water-driven in the monsoon (Basundhara 2013, Bharatpur 2019, Kusmunda "
            "2020); faults and weak foundations in the dry season too (Jayant 2008, Rajmahal 2016, Juna Kunada "
            "2017). Rain alone would miss the second kind, so movement and field triggers are essential.\n"
            "2. **Warning signs were there** at Rajmahal (cracks for months, sliding rocks hours before) and Juna "
            "Kunada (cracks widening for three days).\n"
            "3. **Action decided the outcome:** at Juna Kunada machines were moved and crews had radios, so all 30 "
            "people escaped. At Rajmahal a stop order was not enforced, and 23 people died.\n"
            "4. **Most victims were at the toe**, which is why the plan has a consequence step and keeps people "
            "away from the toe.\n"
            "5. **Design FOS missed soft foundations** (Singareni designed at 1.49, MKD-III, Jayant).")
    with st.expander("Methods used before, and how this TARP combines them"):
        st.dataframe(pd.DataFrame([
            ["Design FOS by limit equilibrium (Slide2 / Geoslope)", "One condition, no monitoring", "Physics behind the AI; T1"],
            ["Numerical FOS (RS2, FLAC)", "Slow, one run per case", "Real correction of the AI"],
            ["Probabilistic FOS", "Design stage only", "90 % range sets Yellow"],
            ["ML / ANN FOS", "FOS only, no action", "AI engine for T1, T2 and site trigger values"],
            ["Stability rating / hazard class", "Qualitative", "Consequence step, field checklist"],
            ["Prism / radar velocity limits", "Need site calibration", "T4"],
            ["Inverse velocity (time to failure)", "Only after acceleration starts", "T5"],
            ["Rainfall thresholds", "Not tied to the dump's FOS", "T3, linked to T2 by the AI"],
            ["InSAR / LSTM forecasting", "Decision support only", "Planned forecaster"],
            ["Back-analysis of failures", "One case at a time", "Sets and checks the thresholds"],
        ], columns=["Earlier method", "Limitation alone", "Role here"]), hide_index=True, **FULL)
    with st.expander("Back-analysis: what past failures say about the thresholds"):
        st.dataframe(pd.DataFrame([
            ["Juna Kunada OC (WCL) 2017", "Cracks 1-2 mm to 50 mm in 3 days", "About 16 mm/day before failure: inside "
             "the Orange band (10-50 mm/day)", "DGMS alert 2017"],
            ["Rajmahal OCP (ECL) 2016", "Cracks for months; sliding rocks; stop order 3.5 h before", "Field Red signs "
             "came hours before; the Response failed", "Mongabay 2017"],
            ["Radar-monitored pit wall", "Alarm at about 60 mm/day", "Sounded about 12 h before collapse: supports Red "
             "at 50 mm/day", "Shellam & Coggan 2020"],
            ["WCL internal dump, 75 m at 43°", "FOS 1.07 with lab strength, yet it failed", "Field cohesion about 17 % "
             "below lab (73 vs 88.6 kPa): use back-analysed strength after a failure", "Kainthola et al. 2011"],
            ["Singareni 'Mine A' dump", "Designed at FOS 1.49, failed", "Soft black-cotton foundation: T1 needs the "
             "foundation layer", "Poulsen et al. 2014"],
            ["Coal waste dump, East Kalimantan", "353.5 mm in 17 days", "FOS 1.45 to 0.98: antecedent rain, not one "
             "day, sets T3", "Dwinagara et al. 2024"],
        ], columns=["Failure", "What was recorded", "What it means for the TARP", "Source"]), hide_index=True, **FULL)
        st.markdown("**Check a failure record of your own**: velocity 7, 3 and 1 day before failure.")
        a, b = st.columns([2, 1])
        upf = a.file_uploader("Failure record CSV (timestamp, displacement_mm)", type="csv", key="ba")
        tff = b.text_input("Failure time (YYYY-MM-DD HH:MM)", key="batf")
        if upf is not None and tff:
            ms_ = movement_summary(velocity_record(pd.read_csv(upf), tff))
            st.dataframe(pd.DataFrame([[f"{int(k_[2:-1])} day(s) before", "no data" if not np.isfinite(v_) else
                                        f"{v_:.1f} mm/day", T.vel_trigger(v_, THR["vel"]) if np.isfinite(v_) else "-"]
                                       for k_, v_ in ms_.items()], columns=["When", "Velocity", "Level now"]),
                         hide_index=True)
    with st.expander("Where every number comes from"):
        st.dataframe(pd.DataFrame([
            ["Minimum FOS 1.50 / 1.30", "DGMS circular, scientific study under Reg. 106(2) of CMR 2017, para 3.2"],
            ["TARP trigger points required", "Same circular, para 4.1; DGMS Tech. Circular 02 of 2010 and 02 of 2020"],
            ["Velocity bands 2 / 5 / 10 / 50 / 100 mm/day and survey frequency", "NIT Rourkela scientific-study notes"],
            ["Rain 64.5 / 115.6 / 204.5 mm in 24 h", "India Meteorological Department classes"],
            ["Acceleration and time to failure", "Carla et al. (2017); Fukuzono (1985)"],
            ["Field signs", "DGMS alerts; Indian failure cases"],
            ["AI training data", "3,995 Bishop solutions of Reg. 106 dumps (own solver, checked against Slide2) + 34 "
             "real FOS values: NIT Rourkela (2025) SECL, Chourasia et al. (2025) Jagannathpur, Kainthola et al. "
             "(2011) WCL"],
            ["Actions and responsibilities", "Proposed for the mine's Safety Management Plan, following the DGMS "
             "Bharatpur alert (2019)"],
        ], columns=["Item", "Source"]), hide_index=True, **FULL)
        rep = getattr(model, "report", {})
        if "loo" in rep:
            st.caption("AI accuracy on sites it never saw (leave-one-site-out): "
                       + ", ".join(f"{i_}: R² {r_.R2:.2f}, error {r_.MAPE:.0f} %"
                                   for i_, r_ in rep["loo"]["table"].iterrows()))
    with st.expander("References"):
        for grp, items in TP.REFS:
            st.markdown(f"**{grp}**")
            st.markdown("\n".join(f"- {t_}" + (f" [link]({u_})" if u_ else "") for t_, u_ in items))
