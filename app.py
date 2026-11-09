"""
DumpSafe AI - Trigger Action Response Plan (TARP) dashboard for coal-mine OB dumps.
Run:  streamlit run app.py
"""
from __future__ import annotations
import io, os, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.lem import DumpGeometry
from core.dgms import check as dgms_check
from core.training import load as load_saved, fit_model
from core.monitoring import analyse, classify_velocity, VELOCITY_TARP
from core import tarp as T

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
st.set_page_config(page_title="DumpSafe AI - TARP for dump slopes", page_icon="⛰️", layout="wide")
_v = tuple(int(x) for x in getattr(st, "__version__", "1.40.0").split(".")[:2])
FULL = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}
COL = T.COL
INK, TEAL = "#1C2B2D", "#2F6B6A"
PLOT = dict(template="simple_white", margin=dict(l=10, r=10, t=44, b=10), height=320,
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
.card {background:#fff; border:1px solid #dfe5e3; border-radius:12px; padding:10px 12px; height:128px;
       box-shadow:0 1px 2px rgba(28,43,45,0.05);}
.card .t {font-size:0.76rem; color:#4a5a5c; text-transform:uppercase; letter-spacing:0.4px;}
.card .v {font-size:1.05rem; font-weight:600; color:#1C2B2D; margin:4px 0 2px 0;}
.card .n {font-size:0.78rem; color:#5b6a6c;}
.pill {display:inline-block; border-radius:999px; padding:1px 10px; color:#fff; font-size:0.78rem; font-weight:600;}
.big {font-size:2.6rem; font-weight:700; color:#1C2B2D; line-height:1.1;}
.muted {color:#5b6a6c; font-size:0.85rem;}
.box {background:#f4f7f6; border-radius:12px; padding:12px 16px; margin:6px 0;}
.scale {display:flex; gap:6px; margin:6px 0 10px 0;}
.scale div {flex:1; border-radius:10px; padding:7px 8px; color:#fff; font-size:0.82rem; opacity:0.45;}
.scale div.on {opacity:1; outline:3px solid #1C2B2D; font-weight:600;}
.scale b {display:block; font-size:0.9rem;}
section[data-testid="stSidebar"] {background:#eef2f0;}
</style>""", unsafe_allow_html=True)


def pill(level):
    return f"<span class='pill' style='background:{COL.get(level, '#8a9597')}'>{level}</span>"


def respond(trigger, level, where=st):
    """What to do for one trigger at its current zone."""
    if level not in T.ORDER:
        where.markdown(f"<div class='box'><b>Response</b><br>No readings yet - enter the data above.</div>",
                       unsafe_allow_html=True)
        return
    a, who, when, _ = T.TRIGGER_ACTIONS[trigger][level]
    where.markdown(f"<div class='box' style='border-left:6px solid {COL[level]}'><b>Response {pill(level)}</b>"
                   f"<br>{a}<br><span class='muted'><b>Who:</b> {who} &nbsp;·&nbsp; <b>When:</b> {when}</span></div>",
                   unsafe_allow_html=True)


def scale(limits, current, where=st):
    """Four zones with their limits; the current zone is highlighted."""
    html = "".join(f"<div class='{'on' if lv == current else ''}' style='background:{COL[lv]}'><b>{lv}</b>"
                   f"{txt.replace('<', '&lt;')}</div>"
                   for lv, txt in zip(T.ORDER, limits))
    where.markdown(f"<div class='scale'>{html}</div>", unsafe_allow_html=True)


def result(label, value, unit, level, where=st):
    where.markdown(f"<div class='box'><span class='muted'>{label}</span><br><span class='big'>{value}</span> "
                   f"<span class='muted'>{unit}</span><br>{pill(level)}</div>", unsafe_allow_html=True)


SENSORS = ["Total station + prisms", "Slope stability radar (SSR)", "GNSS monitors", "Crack meters / extensometers",
           "Vibrating-wire piezometers", "Rain gauge (or IMD station)"]


@st.cache_resource(show_spinner="Loading the FOS engine ...")
def engine():
    m, _ = load_saved(False, 0.10)
    if m is None:
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
        out[nm] = dict(c=float(x.c_kPa), phi=float(x.phi_deg), g=float(x.gamma_kNm3), nd=int(x.n_decks),
                       dh=float(x.deck_height_m), da=float(x.deck_angle_deg),
                       bw=float(0 if pd.isna(x.berm_width_m) else x.berm_width_m))
    return out


@st.cache_data
def movement(raw_csv: str):
    return analyse(pd.read_csv(io.StringIO(raw_csv)))


def fos_many(model, rows):
    return model.predict(pd.DataFrame(rows), with_interval=False)["FOS"].values


def crossing(x, y, level):
    if y[0] <= level:
        return float(x[0])
    k = np.where(y <= level)[0]
    if not len(k):
        return None
    i = k[0]
    return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


model = engine()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## ⛰️ DumpSafe AI")
st.sidebar.caption("TARP for overburden dumps")
P = presets()
site = st.sidebar.selectbox("Dump / mine", list(P) + ["Custom dump"])
pre = P.get(site)
dtype = st.sidebar.radio("Type of dump", ["Permanent", "Working / temporary"])
FOS_MIN = 1.50 if dtype == "Permanent" else 1.30
people = st.sidebar.toggle("People / road / buildings below the toe", value=True)
have = st.sidebar.multiselect("Instruments at the mine", SENSORS, default=SENSORS[:1] + SENSORS[5:])

st.markdown("<h1>Trigger Action Response Plan</h1>", unsafe_allow_html=True)
status = st.container()
tabs = st.tabs(["T1 · FOS", "T2 · Rainfall", "T3 · Velocity", "T4 · Acceleration", "T5 · Field signs",
                "Response plan"])
g_lo, y_lo, r_lo = T.fos_bands(FOS_MIN)

# ================================================================== T1 FOS
with tabs[0]:
    k = site.replace(" ", "")
    L, R = st.columns([1, 1.2], gap="large")
    with L:
        a, b, c_ = st.columns(3)
        c = a.number_input("Cohesion c (kPa)", 0.0, 300.0, pre["c"] if pre else 40.0, 1.0, key=f"c{k}")
        phi = b.number_input("Friction φ (°)", 5.0, 45.0, pre["phi"] if pre else 28.0, 0.5, key=f"p{k}")
        g = c_.number_input("Unit weight γ (kN/m³)", 14.0, 28.0, pre["g"] if pre else 19.0, 0.1, key=f"g{k}")
        a, b = st.columns(2)
        nd = a.slider("Decks", 1, 4, pre["nd"] if pre and pre["nd"] <= 4 else 3, key=f"n{k}")
        dh = b.slider("Deck height (m)", 10.0, 30.0, min(pre["dh"], 30.0) if pre else 30.0, 1.0, key=f"h{k}")
        a, b = st.columns(2)
        da = a.slider("Deck angle (°)", 20.0, 37.5, min(pre["da"], 37.5) if pre else 32.0, 0.5, key=f"a{k}")
        bw = b.slider("Bench width (m)", 0.0, 40.0, pre["bw"] if pre else 30.0, 1.0, key=f"b{k}", disabled=nd == 1)
        wmode = st.radio("Water in the dump", ["Dry / drained", "Piezometer reading", "Pore-pressure ratio r_u"],
                         horizontal=True)
        z_tip, ru = 10.0, 0.0
        if wmode == "Piezometer reading":
            a, b = st.columns(2)
            z_tip = a.number_input("Tip depth (m)", 1.0, 100.0, 10.0, 0.5)
            head_now = b.number_input("Water head above tip (m)", 0.0, 100.0, 2.0, 0.1)
            ru = min(0.45, T.ru_from_piezometer(head_now, z_tip, g))
        elif wmode.startswith("Pore"):
            ru = st.slider("r_u", 0.0, 0.45, 0.10, 0.01)
    geom = DumpGeometry(int(nd), dh, da, bw if nd > 1 else 0.0)
    t0 = time.perf_counter()
    F = float(fos_many(model, [dict(c_kPa=c, phi_deg=phi, gamma_kNm3=g, r_u=ru, **geom.cols())])[0])
    ms = (time.perf_counter() - t0) * 1000
    t1 = T.fos_trigger(F, FOS_MIN)
    ok, viol = dgms_check(geom.n_decks, geom.deck_height, geom.deck_angle, geom.berm_width, geom.overall_angle,
                          H=geom.height)
    with R:
        result(f"FOS · {ms:.0f} ms · height {geom.height:.0f} m · r_u {ru:.2f}", f"{F:.2f}", "", t1)
        scale([f"≥ {g_lo:.2f}", f"{y_lo:.2f} - {g_lo:.2f}", f"{r_lo:.2f} - {y_lo:.2f}", f"< {r_lo:.2f}"], t1)
        if not ok:
            st.warning("Outside CMR 2017 Reg. 106: " + "; ".join(viol))
        respond("T1 FOS", t1)
        rus = np.round(np.arange(0.0, 0.451, 0.01), 2)
        sw = fos_many(model, [dict(c_kPa=c, phi_deg=phi, gamma_kNm3=g, r_u=r_, **geom.cols()) for r_ in rus])
        lim = [(lv, crossing(rus, sw, val)) for lv, val in [("Yellow", g_lo), ("Orange", y_lo), ("Red", r_lo)]]
        st.markdown("**Water that changes the zone**")
        st.dataframe(pd.DataFrame([[lv, "-" if r_ is None else f"{r_:.2f}",
                                    "not reached" if r_ is None else f"{r_ * g * z_tip / 9.81:.1f} m"]
                                   for lv, r_ in lim],
                                  columns=["Zone", "r_u", f"Piezometer head ({z_tip:g} m tip)"]),
                     hide_index=True, **FULL)

# ================================================================== T2 Rainfall
with tabs[1]:
    L, R = st.columns([1, 1.2], gap="large")
    with L:
        rain24 = st.number_input("Rain in the last 24 h (mm)", 0.0, 600.0, 0.0, 5.0)
        rain15 = st.number_input("Rain in the last 15 days (mm)", 0.0, 3000.0, 0.0, 10.0)
    t2 = T.rain_trigger(rain24, rain15)
    with R:
        result("Rain in 24 h", f"{rain24:.0f}", f"mm · {rain15:.0f} mm in 15 days", t2)
        y, o, r = T.RAIN_24H
        scale([f"< {y:g} mm", f"{y:g} - {o:g} mm", f"{o:g} - {r:g} mm or ≥ {T.RAIN_15D_ORANGE:g} mm in 15 d",
               f"≥ {r:g} mm"], t2)
        respond("T2 Rainfall", t2)

# ================================================================== T3 Velocity (data entered here, used by T4)
with tabs[2]:
    mode = st.radio("Movement readings", ["Upload prism / GNSS / radar CSV", "Enter readings manually"],
                    horizontal=True)
    raw = None
    if mode.startswith("Upload"):
        up = st.file_uploader("CSV columns: timestamp, displacement_mm", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
    else:
        blank = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"),
                              "displacement_mm": pd.Series(dtype=float)})
        ed = st.data_editor(blank, num_rows="dynamic", key="manual", **FULL,
                            column_config={"timestamp": st.column_config.DatetimeColumn("Date and time"),
                                           "displacement_mm": st.column_config.NumberColumn("Displacement (mm)")})
        raw = ed.dropna() if len(ed.dropna()) >= 2 else None
    t3, t4, mv = "No data", "No data", None
    if raw is not None and {"timestamp", "displacement_mm"} <= set(raw.columns) and len(raw.dropna()) >= 2:
        mv = movement(raw[["timestamp", "displacement_mm"]].to_csv(index=False))
    elif raw is not None:
        st.warning("Need at least two readings with columns timestamp and displacement_mm.")
    if mv:
        t3 = T.vel_trigger(mv["v"])
        t4 = T.accel_trigger(mv["onset"], mv["vr"], mv["window"])
        d = mv["d"]
        ts = mv["t0"] + pd.to_timedelta(d.t_days, unit="D")
    L, R = st.columns([1.3, 1], gap="large")
    with R:
        result("Velocity now", "-" if not mv else f"{mv['v']:.1f}", "mm/day", t3)
        y, o, r = T.VEL
        scale([f"< {y:g}", f"{y:g} - {o:g}", f"{o:g} - {r:g}", f"≥ {r:g} mm/day"], t3)
        respond("T3 Velocity", t3)
    with L:
        if mv:
            top = max(120.0, float(d.velocity_mm_day.max()) * 1.3)
            bands = [0.05] + list(T.VEL) + [top]
            fig = go.Figure()
            for j, lv in enumerate(T.ORDER):
                fig.add_hrect(y0=bands[j], y1=bands[j + 1], fillcolor=COL[lv], opacity=0.13, line_width=0)
            fig.add_trace(go.Scatter(x=ts, y=d.velocity_mm_day.clip(lower=0.05), mode="lines+markers",
                                     line=dict(color=INK, width=2), name="velocity"))
            fig.update_layout(**PLOT, title="Velocity", showlegend=False)
            fig.update_yaxes(type="log", title="mm/day", range=[np.log10(0.05), np.log10(top)])
            st.plotly_chart(fig, **FULL)
        else:
            st.info("Upload a survey file or type at least two readings.")

# ================================================================== T4 Acceleration
with tabs[3]:
    L, R = st.columns([1.3, 1], gap="large")
    with R:
        w = mv["window"] if mv else None
        if not mv:
            val, unit = "-", ""
        elif w:
            val, unit = f"{max(w['days_early'], 0):.1f}", "days to predicted failure"
        else:
            val, unit = (f"{mv['vr']:.2f}" if np.isfinite(mv["vr"]) else "-"), "velocity ratio (short / long)"
        result("Acceleration", val, unit, t4)
        scale(["steady", "speeding up", f"ratio ≥ {T.VR_ORANGE}", f"failure < {T.IV_RED_DAYS:g} days"], t4)
        respond("T4 Acceleration", t4)
    with L:
        if mv:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=d.v_short, name="short average (3)", line=dict(color=INK, width=2)))
            fig.add_trace(go.Scatter(x=ts, y=d.v_long, name="long average (6)",
                                     line=dict(color=TEAL, width=2, dash="dash")))
            fig.update_layout(**PLOT, title="Is the dump speeding up?", legend=dict(orientation="h", y=-0.2))
            fig.update_yaxes(title="mm/day")
            st.plotly_chart(fig, **FULL)
            if w:
                st.markdown(f"Predicted failure: **{(mv['t0'] + pd.Timedelta(days=w['t_early'])):%d %b %Y %H:%M}**")
        else:
            st.info("Uses the readings entered on T3.")

# ================================================================== T5 Field signs
with tabs[4]:
    L, R = st.columns([1.3, 1], gap="large")
    seen = []
    with L:
        for lv in ["Yellow", "Orange", "Red"]:
            st.markdown(pill(lv), unsafe_allow_html=True)
            for s_, l_ in T.VISUAL_SIGNS:
                if l_ == lv and st.checkbox(s_, key=f"vs{s_}"):
                    seen.append(s_)
    t5 = T.visual_trigger(seen)
    with R:
        result("Signs seen today", f"{len(seen)}", "", t5)
        scale(["none", "new cracks, seepage", "widening cracks, toe bulge, ponding", "scarp, sliding, movement"], t5)
        respond("T5 Field signs", t5)

# ================================================================== combined level
trig = {"T1 FOS": t1, "T2 Rainfall": t2, "T3 Velocity": t3, "T4 Acceleration": t4, "T5 Field signs": t5}
final, gov, note, missing = T.evaluate(trig, people)
act = next(a_ for a_ in T.ACTION_TABLE if a_["Level"] == final)
vals = {"T1 FOS": f"FOS {F:.2f}", "T2 Rainfall": f"{rain24:.0f} mm / 24 h",
        "T3 Velocity": "no readings" if not mv else f"{mv['v']:.1f} mm/day",
        "T4 Acceleration": "no readings" if not mv else ("speeding up" if mv["onset"] is not None else "steady"),
        "T5 Field signs": f"{len(seen)} sign(s)"}
with status:
    st.markdown(f"<div class='banner' style='background:{COL[final]}'><div class='lv'>{final}</div>"
                f"<div class='ln'>{act['Action']}<br><b>Who:</b> {act['Responsible']} · <b>When:</b> "
                f"{act['Response_time']}" + (f"<br>From: {', '.join(gov)}" if gov else "")
                + (" (raised one zone: people below)" if note else "") + "</div></div>", unsafe_allow_html=True)
    cc = st.columns(5)
    for col_, (kk, lv) in zip(cc, trig.items()):
        col_.markdown(f"<div class='card' style='border-top:5px solid {COL.get(lv, '#8a9597')}'><div class='t'>{kk}"
                      f"</div><div class='v'>{vals[kk]}</div>{pill(lv)}</div>", unsafe_allow_html=True)

# ================================================================== Response plan
with tabs[5]:
    st.markdown(f"TARP level now: {pill(final)}", unsafe_allow_html=True)
    now = [dict(Trigger=k_, Zone=lv_, Action=T.TRIGGER_ACTIONS[k_][lv_][0], Who=T.TRIGGER_ACTIONS[k_][lv_][1],
                When=T.TRIGGER_ACTIONS[k_][lv_][2]) for k_, lv_ in trig.items() if lv_ in T.ORDER and lv_ != "Green"]
    zcol = lambda s_: [f"background-color:{COL[x]};color:white;font-weight:600" if x in T.ORDER else "" for x in s_]
    st.markdown("**Responses required now**")
    if now:
        st.dataframe(pd.DataFrame(now).style.apply(zcol, subset=["Zone"]), hide_index=True, **FULL)
    else:
        st.success("All triggers Green: routine monitoring.")
    if missing:
        st.warning("No readings for " + ", ".join(missing) + ".")
    mov = ["Total station + prisms", "Slope stability radar (SSR)", "GNSS monitors", "Crack meters / extensometers"]
    need = {"T2 Rainfall": ["Rain gauge (or IMD station)"], "T3 Velocity": mov, "T4 Acceleration": mov[:3]}
    gaps = [k_ for k_, s_ in need.items() if not any(x in have for x in s_)]
    if gaps:
        st.warning("Instrument missing for " + ", ".join(gaps) + ".")
    st.markdown("**Full response plan**")
    matrix = pd.DataFrame([dict(Trigger=k_, Zone=lv_, Action=a_[0], Who=a_[1], When=a_[2])
                           for k_, d_ in T.TRIGGER_ACTIONS.items() for lv_, a_ in d_.items()])
    st.dataframe(matrix.style.apply(zcol, subset=["Zone"]), hide_index=True, **FULL, height=420)
    acts = pd.DataFrame(T.ACTION_TABLE)[["Level", "Action", "Responsible", "Response_time"]].rename(
        columns={"Level": "Zone", "Responsible": "Who", "Response_time": "When"})
    css = ("body{font-family:Arial,sans-serif;margin:28px;color:#1c2b2d}table{border-collapse:collapse;width:100%;"
           "font-size:13px}td,th{border:1px solid #999;padding:5px;text-align:left}th{background:#e7ecea}")
    sheet = pd.DataFrame([[k_, vals[k_], lv_] for k_, lv_ in trig.items()], columns=["Trigger", "Now", "Zone"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>TARP - {site}</title><style>{css}</style>"
            f"</head><body><h1>TARP - {site}</h1><p>{dtype} dump, {geom.n_decks} x {geom.deck_height:g} m decks at "
            f"{geom.deck_angle:g}°, bench {geom.berm_width:g} m. TARP level now: <b>{final}</b>.</p>"
            f"<h2>Triggers</h2>{sheet.to_html(index=False)}<h2>Responses required now</h2>"
            f"{pd.DataFrame(now).to_html(index=False) if now else '<p>None.</p>'}<h2>Whole-mine action by zone</h2>"
            f"{acts.to_html(index=False)}<h2>Full response plan</h2>{matrix.to_html(index=False)}</body></html>")
    st.download_button("⬇️ Download TARP (print to PDF)", html, f"TARP_{site.replace(' ', '_')}.html", "text/html")
