"""Pages of DumpSafe AI that explain, calibrate and document the TARP.

Each page answers one question:
  1. What is the TARP?                       -> overview_page
  2. Why have Indian coal-mine dumps failed? -> history_page
  3. Which parameters and instruments?       -> instruments_page
  4. How were the thresholds set?            -> calibration_page (with the back-analysis lab)
  8. Which data were used and which not?     -> data_page
  9. References                              -> references_page
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import tarp as T
from core.backanalysis import (velocity_record, movement_summary, rain_summary, suggest, juna_kunada,
                               LEADS, RAIN_WINDOWS)
from core.monitoring import demo_series, VELOCITY_TARP

HERE = os.path.dirname(os.path.abspath(__file__))
TD = os.path.join(HERE, "data", "tarp")


def _pill(level):
    return (f"<span style='background:{T.COL.get(level, '#8a9597')};color:white;border-radius:999px;"
            f"padding:1px 9px;font-size:0.8rem;font-weight:600'>{level}</span>")


def _box(title, body, colour="#2F6B6A"):
    return (f"<div style='border:1px solid #d9e0de;border-top:5px solid {colour};border-radius:10px;"
            f"padding:12px 14px;background:#fff;height:100%'><div style='font-weight:700;font-size:1.05rem;"
            f"margin-bottom:4px'>{title}</div><div style='font-size:0.92rem;color:#2b3a3c'>{body}</div></div>")


# ======================================================================== 1
def overview_page(FULL, model_summary=""):
    st.title("TARP for coal-mine overburden dumps")
    st.markdown("<p class='lede'>A <b>Trigger Action Response Plan</b> says, in advance, <b>what</b> is watched on "
                "the dump, <b>at what value</b> each level is reached, <b>what is done</b> at that level, <b>who</b> "
                "does it and <b>how fast</b>. This project develops that plan for Indian coal-mine OB dumps and uses "
                "AI where it improves a trigger.</p>", unsafe_allow_html=True)
    c = st.columns(3)
    c[0].markdown(_box("T - Trigger", "A measured or computed parameter with threshold values for Green, Yellow, "
                       "Orange and Red: design FOS, live FOS with today's water, rainfall, slope velocity, "
                       "acceleration, visual signs."), unsafe_allow_html=True)
    c[1].markdown(_box("A - Action", "What must be done at each level: routine survey; more frequent survey and "
                       "inspection; stop dumping and close the toe zone; stop work and evacuate.", "#d9731f"),
                  unsafe_allow_html=True)
    c[2].markdown(_box("R - Response", "Who acts and how fast: surveyor / overman (routine), geotech engineer "
                       "(24 h), Manager (same shift), Manager and Agent (immediately).", "#b03a2e"),
                  unsafe_allow_html=True)

    st.subheader("Why a TARP, and why this one")
    st.dataframe(pd.DataFrame([
        ["DGMS circular, scientific study under Reg. 106(2) of CMR 2017", "Para 3.2: minimum FOS 1.50 (permanent) / "
         "1.30 (other). Para 4.1: the study must give trigger points (warning and withdrawal levels) for a TARP.",
         "T1, T2 thresholds; the whole plan"],
        ["DGMS Tech. Circular 02 of 2010 (06.07.2010)", "Pit and dump slopes: scientific design, real-time monitoring "
         "(SSR, SAR), alarms at preset thresholds, time to withdraw men and machines.", "T4, T5; Red = withdrawal"],
        ["DGMS Tech. Circular 02 of 2020 (09.01.2020)", "Guidelines for systematic monitoring of slopes.",
         "Survey frequency per level"],
        ["DGMS alert, Bharatpur OCP (2019)", "Code of practice for real-time monitoring inside the Safety "
         "Management Plan; no work within 100 m of a backfill toe; keep people away from the toe; pre-monsoon "
         "inspection and drainage.", "Actions; consequence step; T3"],
    ], columns=["Requirement", "What it asks", "Where it sits in this TARP"]), hide_index=True, **FULL)

    st.subheader("How the plan works")
    st.markdown("""
<div style='display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;font-size:0.9rem'>
 <div style='flex:1 1 160px;border:1px solid #d9e0de;border-radius:10px;padding:10px;background:#f6f8f7'>
  <b>1. Data</b><br>Survey prisms / radar / GNSS / InSAR, crack meters, piezometers, rain gauge, shift inspection,
  dump geometry and lab strength</div>
 <div style='align-self:center;font-size:1.4rem'>&#8594;</div>
 <div style='flex:1.4 1 220px;border:1px solid #d9e0de;border-radius:10px;padding:10px;background:#fff'>
  <b>2. Six triggers</b><br>T1 Design FOS <i>(AI)</i><br>T2 Live FOS with water <i>(AI)</i><br>T3 Rainfall<br>
  T4 Velocity<br>T5 Acceleration / time to failure<br>T6 Visual signs</div>
 <div style='align-self:center;font-size:1.4rem'>&#8594;</div>
 <div style='flex:1 1 170px;border:1px solid #d9e0de;border-radius:10px;padding:10px;background:#fff'>
  <b>3. Combine</b><br>Worst trigger governs; +1 level if people, haul road or structures are in the run-out zone
  (rain and design FOS alone stop at Orange)</div>
 <div style='align-self:center;font-size:1.4rem'>&#8594;</div>
 <div style='flex:1 1 170px;border:1px solid #d9e0de;border-radius:10px;padding:10px;background:#fff'>
  <b>4. Level &#8594; Action, Responsible, Response time</b><br>
  <span style='color:#2e7d4f'>&#9632;</span> Green <span style='color:#d8a31a'>&#9632;</span> Yellow
  <span style='color:#d9731f'>&#9632;</span> Orange <span style='color:#b03a2e'>&#9632;</span> Red</div>
 <div style='align-self:center;font-size:1.4rem'>&#8634;</div>
 <div style='flex:1 1 170px;border:1px solid #d9e0de;border-radius:10px;padding:10px;background:#fff7ec'>
  <b>5. Back-analysis</b><br>After every failure or near-miss, read what each trigger showed 7, 3 and 1 day
  before and re-set the thresholds</div>
</div>""", unsafe_allow_html=True)

    st.subheader("Where AI and computation are used")
    st.dataframe(pd.DataFrame([
        ["A. FOS engine", "Neural-network ensemble trained on 3,995 Bishop solutions of DGMS-compliant dumps, "
         "corrected to real studies by a Gaussian process, with a 90 % range", "T1, T2", "Built"],
        ["B. Water link", "Learns rain -> pore-pressure ratio r_u from paired rain gauge and piezometer data, so "
         "dumps with only a rain gauge still get a live FOS", "T2", "Built (linear); needs site data"],
        ["C. Acceleration detector", "Short / long moving-average crossover marks the onset of acceleration "
         "(Carla et al. 2017)", "T5", "Built"],
        ["D. Time-to-failure window", "Inverse-velocity fits on raw, short and long series give an earliest-"
         "latest failure date", "T5", "Built"],
        ["E. Back-analysis", "Strength at failure (c-phi at FOS = 1) and trigger values 7, 3, 1 day before "
         "failure", "All thresholds", "Built"],
        ["F. Displacement forecaster", "LSTM on prism / radar / InSAR series to forecast tomorrow's velocity "
         "(Mirmazloumi et al. 2023)", "T5", "Planned - needs SECL monitoring series"],
    ], columns=["Component", "What it does", "Trigger", "Status"]), hide_index=True, **FULL)

    st.subheader("What is new here")
    st.markdown(
        "- In the papers reviewed (page 8), Indian dump studies give **either** a design FOS **or** a monitoring "
        "alarm. This plan joins design FOS, live FOS with pore pressure, rainfall, velocity, acceleration and "
        "visual signs in **one** TARP with actions and responsibilities.\n"
        "- Every threshold is traced to a **source or a back-analysed failure** (page 4), and the back-analysis "
        "lab lets the mine re-calibrate them from its own failures and near-misses.\n"
        "- Page 5 runs the AI FOS engine hundreds of times to turn a dump's geometry and material into "
        "**site-specific trigger values** (r_u, piezometer head, rain and height limits) and a downloadable TARP.\n"
        "- The live page **replays** a record day by day, so the warning time each trigger would have given "
        "is measured, not assumed.\n"
        "- The failure history (page 2) shows **why** each trigger is needed: water-driven failures in the "
        "monsoon, and fault / foundation failures in the dry season that only movement and visual triggers "
        "can catch.")
    if model_summary:
        st.caption(model_summary)


# ======================================================================== 2
def history_page(FULL, PLOT):
    st.title("Why Indian coal-mine dumps have failed")
    st.markdown("<p class='lede'>Reported OB dump and slope failures in Indian coal mines, what caused them, what "
                "warning there was, and which TARP trigger would have fired. Only facts stated in the cited source "
                "are entered; blanks mean the source does not say.</p>", unsafe_allow_html=True)
    h = pd.read_csv(os.path.join(TD, "failure_history_india.csv"))
    k = st.columns(4)
    k[0].metric("Failures documented", len(h))
    k[1].metric("Deaths in these events", int(h.deaths.sum()))
    k[2].metric("With warning signs reported", int((~h.precursors.str.startswith("Not reported")).sum()))
    k[3].metric("Rain or water involved", int(h.cause_group.str.contains("Rain|water", case=False).sum()))

    c1, c2 = st.columns(2)
    grp = h.cause_group.str.replace(r" /.*| \+.*| \(.*", "", regex=True).value_counts()
    f = go.Figure(go.Bar(x=grp.values, y=grp.index, orientation="h", marker_color="#2F6B6A"))
    f.update_layout(**PLOT, title="Main cause (as reported)")
    f.update_layout(height=320)
    c1.plotly_chart(f, **FULL)
    mon = pd.to_datetime(h.date.str[:10], errors="coerce").dt.month.dropna().astype(int)
    season = mon.map(lambda m: "Monsoon (Jun-Sep)" if 6 <= m <= 9 else "Dry season").value_counts()
    f2 = go.Figure(go.Bar(x=season.index, y=season.values, marker_color=["#7fa7c9", "#A8977A"][:len(season)]))
    f2.update_layout(**PLOT, title="Season of failure (events with a date)")
    f2.update_layout(height=320)
    c2.plotly_chart(f2, **FULL)

    st.subheader("What the history teaches the TARP")
    st.markdown("""
1. **Two families of failure.** Water-driven failures happen in the monsoon (Basundhara 2013, Bharatpur 2019,
   Kusmunda 2020). Fault and weak-foundation failures happen in the dry season too (Jayant 2008, Rajmahal 2016,
   Juna Kunada 2017). A rainfall trigger alone would have missed the second family, so **movement (T4, T5) and
   visual (T6) triggers are essential**.
2. **Warning signs were often there.** Rajmahal had cracks for months and visible sliding hours before;
   Juna Kunada had cracks widening from 1-2 mm to 5 cm over three days; Sasti had reported toe concerns.
3. **The outcome depended on Action and Response.** At Juna Kunada machines were moved and every crew had a
   radio: all 30 people escaped. At Rajmahal a stop order was given 3.5 h before the collapse but not enforced:
   23 died. The TARP therefore fixes **who** decides and **how fast**, not only the threshold.
4. **Victims were usually at the toe.** Villagers at the toe (Basundhara), crews below a dump (Sasti) or on the
   dip side of a rib holding backfill (Bharatpur). Hence the **consequence step** and the toe-exclusion action.
5. **Design FOS missed the foundation.** Singareni Mine A was designed at FOS 1.49 but failed on black-cotton
   soil; MKD-III and DGMS case (h) failed through soft ground. **T1 must include the foundation layer.**""")

    st.subheader("Event register")
    show = h[["date", "mine", "company", "state", "structure", "deaths", "cause_group", "rain", "precursors"]]
    st.dataframe(show, hide_index=True, **FULL, height=360)
    st.subheader("Each event against the TARP")
    for _, r in h.iterrows():
        with st.expander(f"{r.date} - {r.mine} ({r.company}) - {r.deaths} deaths reported"):
            st.markdown(f"**What failed:** {r.structure}  \n**Cause:** {r.cause}  \n**Rain / water:** {r.rain}  \n"
                        f"**Warning signs:** {r.precursors}  \n**TARP trigger that would fire:** {r.tarp_would_fire}  \n"
                        f"**Gap:** {r.tarp_gap}  \n**Source:** [{r.source}]({r.url})")

    st.subheader("Records with numbers before failure (India and abroad)")
    st.markdown("These are the few published records that say how much movement or rain came before a failure. "
                "They are used on page 4 to cross-check the thresholds.")
    st.dataframe(pd.read_csv(os.path.join(TD, "precursor_records.csv")), hide_index=True, **FULL)
    st.caption("No Indian dump failure with a published day-by-day prism or radar record was found. Masood et al. "
               "(2025) used SSR data from three SECL mines but did not publish the values. This is the main data gap "
               "(page 7).")


# ======================================================================== 3
def instruments_page(FULL):
    st.title("Methods, triggers and instruments")
    st.markdown("<p class='lede'>What each trigger measures, why it is needed (from the failure history), which "
                "instrument gives it, and how it is used in Indian coal mines.</p>", unsafe_allow_html=True)
    st.subheader("Methods used earlier, and how this TARP joins them")
    st.dataframe(pd.DataFrame([
        ["Limit-equilibrium design FOS (Bishop, Janbu) in Slide / Geoslope", "Kainthola et al. 2011; Behera et al. "
         "2016; most Indian scientific studies", "One FOS for one assumed condition; no monitoring link",
         "T1, and as the physics behind the AI engine"],
        ["Numerical FOS (FEM / FDM strength reduction: RS2, Phase2, FLAC)", "NIT Rourkela 2025 (SECL); Chourasia "
         "et al. 2025; Singh et al. 2022", "Slow; one run per scenario", "Field correction of the AI engine"],
        ["Probabilistic FOS / probability of failure", "Kumar et al. 2023; Read & Stacey 2009", "Design stage only",
         "90 % range in T1 (Yellow)"],
        ["ML / ANN FOS prediction", "Sahoo et al. 2025 and others", "FOS only; not linked to actions", "AI engine for "
         "T1, T2; runs hundreds of scenarios to set site thresholds"],
        ["Dump stability rating / hazard class", "Sahu, Mohanty & Jayanthu 2023", "Qualitative; static",
         "Consequence step and visual checklist"],
        ["Velocity thresholds from prisms / SSR", "NIT Rourkela notes; DGMS 02/2010; Shellam & Coggan 2020",
         "Need site calibration; no link to water or design", "T4"],
        ["Inverse-velocity time of failure", "Fukuzono 1985; Carla et al. 2017; Masood et al. 2025 (SECL SSR)",
         "Only after acceleration starts", "T5"],
        ["Rainfall thresholds", "Farina et al. 2020; IMD classes", "Not tied to the dump's FOS",
         "T3, linked to T2 through the AI r_u limits"],
        ["InSAR / LSTM forecasting", "Mirmazloumi et al. 2023", "Decision support, no TARP", "Planned forecaster"],
        ["Back-analysis of failures", "Jhanwar 2010; Dwinagara et al. 2024", "One case at a time",
         "Back-analysis lab that re-sets all thresholds"],
    ], columns=["Method", "Used by", "Limitation alone", "Role in this TARP"]), hide_index=True, **FULL)
    st.subheader("The six triggers")
    st.dataframe(pd.DataFrame([
        ["T1 Design FOS", "FOS of the planned / current geometry with design pore pressure",
         "Design errors: Sasti (height, slushy floor), WCL internal dump (43 deg), Singareni (soft foundation)",
         "Survey section, lab c, phi, gamma -> FOS engine", "When geometry or design changes"],
        ["T2 Live FOS", "Same FOS with today's pore-pressure ratio r_u", "Water behind a rib (Bharatpur); seepage at "
         "Jayant", "Vibrating-wire piezometer, or rain gauge + learned rain-to-r_u link", "Daily; hourly in monsoon"],
        ["T3 Rainfall", "24-h rain and antecedent rain index", "Basundhara (heavy rain the day before), Kusmunda "
         "(continuous rain), Bharatpur (monsoon)", "Tipping-bucket rain gauge; IMD / GPM as backup", "Hourly"],
        ["T4 Velocity", "Movement rate of crest, benches, toe (mm/day)", "Juna Kunada cracks at about 16 mm/day "
         "before failure", "Total station + prisms, SSR, GNSS, crack meter / extensometer", "Monthly (Normal) to "
         "continuous (Alarm)"],
        ["T5 Acceleration", "Onset of acceleration and inverse-velocity time to failure", "Rajmahal and Juna Kunada: "
         "movement sped up over hours to days", "Same data as T4 (needs regular readings)", "With every reading"],
        ["T6 Visual signs", "Cracks, settlement, toe bulging, seepage, debris fall", "Rajmahal (cracks for months, "
         "sliding rocks), MKD-III (toe heave)", "Shift inspection checklist, UAV photos", "Every shift"],
    ], columns=["Trigger", "Parameter", "Why (failure history)", "Instrument / input", "Reading frequency"]),
        hide_index=True, **FULL)

    st.subheader("Instruments used for dump monitoring")
    st.dataframe(pd.DataFrame([
        ["Total station + prisms", "3D displacement of discrete points", "Point only; needs line of sight; manual or "
         "robotic", "Standard in CIL opencast mines (Ministry of Coal / PIB)", "T4, T5"],
        ["Slope stability radar (SSR)", "Line-of-sight displacement over a whole face, day and night",
         "Covers whole slope; works in rain and dark; costly", "Recommended by DGMS Circular 02/2010; used at "
         "Kusmunda and other SECL mines (Masood et al. 2023, 2025)", "T4, T5"],
        ["GNSS / GPS monitors", "3D position of a point, continuous", "Good for crest and toe points; no line of sight needed", "Used on coal spoils "
         "abroad (Teck TARP)", "T4"],
        ["InSAR (Sentinel-1)", "Ground movement over large areas every 6-12 days", "Free data; too slow for Red",
         "Research use for Indian coalfields; LSTM early warning (Mirmazloumi et al. 2023)", "T4 trend, F"],
        ["UAV photogrammetry / LiDAR", "Surface models, crack maps, volume change", "Repeat surveys; geometry for T1",
         "Raniganj dumps (Chand & Koner 2023; Chand et al. 2024)", "T1, T6"],
        ["3D laser scanner", "Dense surface change", "Periodic", "Listed by Ministry of Coal (Annual Report 2023-24)",
         "T1, T4"],
        ["Crack meter / wireline extensometer", "Opening of a crack or crest movement", "Cheap, simple, continuous",
         "Crack widths reported at Juna Kunada (DGMS 2017)", "T4, T6"],
        ["Time-domain reflectometry (TDR)", "Depth of shearing in a borehole", "Finds the slip surface",
         "NCL R&D (Ministry of Coal Annual Report 2023-24)", "T1 (slip depth)"],
        ["Vibrating-wire piezometer", "Pore-water pressure in the dump / foundation", "Gives r_u directly",
         "DGMS asks for groundwater study and drainage (Bharatpur alert 2019)", "T2"],
        ["Rain gauge / IMD / GPM", "Rainfall depth and intensity", "Cheap; IMD classes give first thresholds",
         "IMD district rainfall", "T3, T2"],
    ], columns=["Instrument", "Measures", "Strengths / limits", "Indian use (source)", "Feeds"]),
        hide_index=True, **FULL)
    st.caption("Survey frequency per velocity level follows the slope-movement table of the NIT Rourkela "
               "scientific-study notes: monthly (< 2 mm/day) to continuous radar (> 50 mm/day).")
    st.dataframe(pd.DataFrame(VELOCITY_TARP, columns=["Upper limit (mm/day)", "Movement class", "Colour",
                                                      "Method", "Frequency", "Response"])
                 .drop(columns="Colour").assign(TARP=lambda x: x["Movement class"].map(T.VEL_MAP)),
                 hide_index=True, **FULL)


# ======================================================================== 4
def _register(thr):
    v, r, a = thr["vel"], thr["rain"], thr.get("ari")
    return pd.DataFrame([
        ["T1 / T2 FOS", "Orange < DGMS minimum (1.50 / 1.30); Red < 1.0; Yellow within 0.05 or 90 % range below",
         "DGMS Reg. 106(2) circular para 3.2", "WCL internal dump: Bishop 1.07 at peak, 0.72 at residual; it failed "
         "(FEM 0.80). Singareni designed at 1.49, failed on soft foundation", "Adopted (statutory)"],
        ["T3 Rain 24 h", f"Yellow {r[0]:g}, Orange {r[1]:g}, Red {r[2]:g} mm", "IMD heavy / very heavy / extremely "
         "heavy", "Basundhara failed the day after heavy rain (amount not published)", "Starting value; calibrate"],
        ["T3 Antecedent rain", "Not used" if a is None else f"Yellow {a[0]:.0f}, Orange {a[1]:.0f}, Red {a[2]:.0f} mm",
         "Back-analysis of rain before failures (lab below)", "East Kalimantan dump: 353.5 mm in 17 days turned pore "
         "pressure positive (Dwinagara et al. 2024)", "Needs daily rain for Indian failures"],
        ["T4 Velocity", f"Yellow {v[0]:g}, Orange {v[1]:g}, Red {v[2]:g} mm/day", "NIT Rourkela notes: Caution, "
         "Alert, Alarm bands", "Juna Kunada: about 16 mm/day for 3 days before failure (in Orange band). SSR alarm "
         "at about 60 mm/day came 12 h before collapse (Shellam & Coggan 2020)", "Adopted; cross-checked"],
        ["T5 Acceleration", "Yellow at onset; Orange if failure forecast < 7 days and v >= 2 mm/day; Red if < 3 days "
         "and v >= 10 mm/day", "Carla et al. (2017); Dick et al. (2015)", "40 synthetic records: no false onset, "
         "detection about 4 days after true onset; Red 5 days before failure in Demo A", "Adopted; calibrate on "
         "SECL SSR"],
        ["T6 Visual", "New cracks / settlement / seepage = Yellow; widening cracks / toe bulge / muddy outflow = "
         "Orange; scarp / visible mass movement = Red", "DGMS alerts; Rajmahal; Juna Kunada; MKD-III",
         "Rajmahal: Red signs 3.5 h before collapse; Juna Kunada: Yellow 4 days, Orange 3 days before", "Adopted"],
        ["Consequence step", "+1 level when people / haul road / structures are in the run-out zone",
         "DGMS Bharatpur alert (keep people away from the toe; 100 m rule)", "Most deaths were at the toe",
         "Adopted"],
    ], columns=["Trigger", "Current thresholds", "Basis", "Back-analysis / cross-check", "Status"])


def _read_power_csv(f):
    """Accepts a normal CSV (timestamp, rainfall_mm, displacement_mm) or a NASA POWER daily
    CSV (header block, then YEAR, MO, DY, PRECTOTCORR)."""
    raw = f.read() if hasattr(f, "read") else open(f, "rb").read()
    txt = raw.decode("utf-8", "ignore") if isinstance(raw, bytes) else raw
    if "-END HEADER-" in txt:
        txt = txt.split("-END HEADER-", 1)[1].strip()
        import io
        d = pd.read_csv(io.StringIO(txt))
        d["timestamp"] = pd.to_datetime(dict(year=d.YEAR, month=d.MO, day=d.DY))
        col = [c for c in d.columns if c.startswith("PRECTOT")][0]
        d["rainfall_mm"] = d[col].where(d[col] >= 0)
        return d[["timestamp", "rainfall_mm"]]
    import io
    return pd.read_csv(io.StringIO(txt))


RAIN_CASES = [
    ("Basundhara area (MCL), 10-08-2013", 22.05, 83.70, "2013-08-10"),
    ("Bharatpur OCP (MCL), 23-07-2019", 20.95, 85.20, "2019-07-23"),
    ("Kusmunda OC (SECL), 23-07-2020 (probable)", 22.33, 82.68, "2020-07-23"),
    ("Rajmahal OCP (ECL), 29-12-2016 - dry-season contrast", 24.85, 87.25, "2016-12-29"),
    ("Juna Kunada OC (WCL), 01-12-2017 - dry-season contrast", 20.05, 79.05, "2017-12-01"),
]


def calibration_page(FULL, PLOT, geometry_inputs, back_analyse, bishop_fos, DumpGeometry):
    st.title("How the thresholds were set")
    st.markdown("<p class='lede'>Every threshold has a source and, where a record exists, a back-analysis of a "
                "real failure. The lab below turns any failure record into threshold values and sends them to the "
                "live TARP.</p>", unsafe_allow_html=True)
    thr = st.session_state.thr
    st.subheader("Threshold register")
    st.dataframe(_register(thr), hide_index=True, **FULL)

    tabs = st.tabs(["Movement before failure", "Rain before failure", "Strength at failure",
                    "Calibration set and new thresholds"])
    # ------------------------------------------------------------ movement
    with tabs[0]:
        st.markdown("Read the velocity a record showed **7, 3 and 1 day** before failure. Upload a prism, radar, "
                    "GNSS or crack-meter record (columns `timestamp`, `displacement_mm`) with the failure time.")
        src = st.radio("Record", ["Juna Kunada OC, WCL, 2017 (real, crack width)", "Demo A (synthetic creep to "
                                  "failure)", "Upload a record"], horizontal=True, key="ba_src")
        if src.startswith("Juna"):
            df, tf = juna_kunada()
            name = "Juna Kunada 2017"
            st.caption("DGMS alert 01-12-2017: cracks 1-2 mm on 27 Nov, 5 cm by 30 Nov, failure 1 Dec 01:05. "
                       "Only two crack readings are published (time of day not given; noon assumed).")
        elif src.startswith("Demo"):
            df, td = demo_series()
            tf = str(pd.Timestamp("2026-07-01") + pd.Timedelta(days=td))
            name = "Demo A (synthetic)"
        else:
            up = st.file_uploader("Failure record CSV", type="csv", key="ba_up")
            df = pd.read_csv(up) if up else None
            tf = st.text_input("Failure date and time (YYYY-MM-DD HH:MM)", key="ba_tf")
            name = st.text_input("Case name", "My failure", key="ba_name")
        if df is not None and tf:
            d = velocity_record(df, tf)
            ms = movement_summary(d)
            k = st.columns(4)
            for j, lead in enumerate(LEADS):
                val = ms[f"v_{int(lead)}d"]
                k[j].metric(f"Velocity {int(lead)} day(s) before", "no data" if not np.isfinite(val)
                            else f"{val:.1f} mm/day", None if not np.isfinite(val) else
                            T.vel_trigger(val, thr["vel"]), delta_color="off")
            k[3].metric("Last reading before failure", f"{d.t_before.min():.2f} days")
            fig = go.Figure()
            bands = [0.05] + list(thr["vel"]) + [max(200.0, float(np.nanmax(d.velocity_mm_day.fillna(0))) * 1.5)]
            for j, lv in enumerate(["Green", "Yellow", "Orange", "Red"]):
                fig.add_hrect(y0=bands[j], y1=bands[j + 1], fillcolor=T.COL[lv], opacity=0.14, line_width=0)
            dd = d.dropna(subset=["velocity_mm_day"])
            fig.add_trace(go.Scatter(x=-dd.t_before, y=dd.velocity_mm_day.clip(lower=0.05),
                                     mode="lines+markers" if len(dd) < 30 else "lines",
                                     line=dict(color="#1C2B2D", width=2, shape="hv" if not d.attrs["dense"]
                                               else "linear"), name="velocity"))
            for lead in LEADS:
                fig.add_vline(x=-lead, line_dash="dot", line_color="#666", annotation_text=f"-{int(lead)} d")
            fig.add_vline(x=0, line_color="#b03a2e", annotation_text="failure")
            fig.update_layout(**PLOT, title=f"{name}: velocity against the current T4 bands",
                              showlegend=False)
            fig.update_xaxes(title="Days relative to failure")
            fig.update_yaxes(type="log", title="mm/day (log)")
            st.plotly_chart(fig, **FULL)
            if st.button("Add this record to the calibration set", key="ba_add_mov"):
                st.session_state.cal_cases.append(dict(case=name, kind="movement", **ms, rain_triggered=False))
                st.success("Added. See the last tab.")

    # ------------------------------------------------------------ rain
    with tabs[1]:
        st.markdown("Rain in the 1, 3, 7, 15 and 30 days before a failure, and the antecedent rain index (ARI). "
                    "Daily satellite rainfall for any past failure can be downloaded free from NASA POWER: click a "
                    "link, save the CSV, upload it here.")
        links = []
        for nm, la, lo, dte in RAIN_CASES:
            t1 = pd.Timestamp(dte)
            t0 = t1 - pd.Timedelta(days=45)
            url = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG"
                   f"&longitude={lo}&latitude={la}&start={t0:%Y%m%d}&end={t1:%Y%m%d}&format=CSV")
            links.append(f"- [{nm}]({url}) &nbsp;(grid point {la} N, {lo} E, approximate)")
        st.markdown("\n".join(links), unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        upr = c1.file_uploader("Rain CSV (NASA POWER file, or columns timestamp, rainfall_mm)", type="csv",
                               key="ba_rain")
        tfr = c2.text_input("Failure date (YYYY-MM-DD)", key="ba_rain_tf")
        nmr = c2.text_input("Case name", "Rain case", key="ba_rain_nm")
        rain_trig = c2.checkbox("Rain was a cause of this failure", value=True, key="ba_rain_trig",
                                help="Only rain-caused failures set the rain thresholds; dry-season failures are "
                                     "kept for contrast.")
        if upr is not None and tfr:
            rdf = _read_power_csv(upr)
            rs = rain_summary(rdf, tfr)
            k = st.columns(len(RAIN_WINDOWS) + 1)
            for j, w in enumerate(RAIN_WINDOWS):
                k[j].metric(f"{w}-day rain", f"{rs[f'rain_{w}d']:.0f} mm")
            k[-1].metric("ARI at failure", f"{rs['ARI_at_failure']:.0f} mm")
            r2 = rdf.copy()
            r2["timestamp"] = pd.to_datetime(r2.timestamp)
            r2["d"] = (r2.timestamp - pd.Timestamp(tfr)).dt.total_seconds() / 86400
            r2 = r2[(r2.d <= 0) & (r2.d >= -45)]
            fr = go.Figure(go.Bar(x=r2.d, y=r2.rainfall_mm, marker_color="#7fa7c9", name="daily rain"))
            fr.add_trace(go.Scatter(x=r2.d, y=r2.rainfall_mm[::-1].cumsum()[::-1], yaxis="y2",
                                    name="rain from this day to failure", line=dict(color="#1C2B2D")))
            fr.update_layout(**PLOT, title=f"{nmr}: rain before failure")
            fr.update_layout(yaxis2=dict(overlaying="y", side="right", title="cumulative (mm)", showgrid=False))
            fr.update_xaxes(title="Days relative to failure")
            st.plotly_chart(fr, **FULL)
            if st.button("Add to the calibration set", key="ba_add_rain"):
                st.session_state.cal_cases.append(dict(case=nmr, kind="rain", rain_triggered=rain_trig, **rs))
                st.success("Added.")
        st.caption("NASA POWER rainfall is a satellite / reanalysis product on a coarse grid (about 0.5 deg). Use the "
                   "mine's own rain gauge or IMD station data when available.")

    # ------------------------------------------------------------ strength
    with tabs[2]:
        st.markdown("At failure FOS = 1. Knowing the geometry and the water, the solver finds which strength "
                    "(c, phi) the dump actually had - often lower than the lab value. This corrects T1 and T2.")
        case = st.radio("Case", ["WCL internal dump, Wardha Valley (Kainthola et al. 2011)", "Your own failure"],
                        horizontal=True, key="sba_case")
        if case.startswith("WCL"):
            geom = DumpGeometry(1, 75.0, 43.0, 0.0)
            gam, ru, lab = 24.4, 0.0, [(88.6, 24.6, "lab peak"), (36.5, 21.5, "lab residual")]
            st.caption("75 m internal dump at 43 deg that failed. Lab peak c 88.6 kPa, phi 24.6 deg; residual "
                       "c 36.5 kPa, phi 21.5 deg; unit weight 24.4 kN/m3 (Kainthola et al. 2011).")
        else:
            geom = geometry_inputs("sba")
            c1, c2 = st.columns(2)
            gam = c1.number_input("Unit weight (kN/m3)", 14.0, 30.0, 20.0, 0.1, key="sba_g")
            ru = c2.number_input("r_u at failure", 0.0, 0.6, 0.0, 0.01, key="sba_ru")
            lab = []
        if st.button("Back-calculate strength at failure (about 5 s)", key="sba_go"):
            phis = [15.0, 18.0, 21.0, 24.0, 27.0, 30.0]
            ba = back_analyse(geom, gam, ru, phis)
            st.session_state.sba = (ba, lab, gam, ru, geom)
        if "sba" in st.session_state:
            ba, lab_, gam_, ru_, geom_ = st.session_state.sba
            fb = go.Figure(go.Scatter(x=ba.phi_deg, y=ba.c_kPa, mode="lines+markers", name="c-phi giving FOS = 1",
                                      line=dict(color="#b03a2e", width=3)))
            for c_, p_, lbl in lab_:
                fos = bishop_fos(geom_, c_, p_, gam_, ru_).fos
                fb.add_trace(go.Scatter(x=[p_], y=[c_], mode="markers+text", text=[f"{lbl}: FOS {fos:.2f}"],
                                        textposition="top center", marker=dict(size=11), name=lbl))
            fb.update_layout(**PLOT, title="Strength the dump had when it failed (Bishop, FOS = 1)")
            fb.update_xaxes(title="Friction angle phi (deg)")
            fb.update_yaxes(title="Cohesion c (kPa)")
            st.plotly_chart(fb, **FULL)
            st.dataframe(ba.round(1), hide_index=True)
            if lab_:
                c_need = float(np.interp(lab_[0][1], ba.phi_deg, ba.c_kPa))
                st.markdown(f"At the lab friction angle ({lab_[0][1]} deg) the dump failed with c of about "
                            f"**{c_need:.0f} kPa**, against {lab_[0][0]} kPa in the lab: field cohesion was about "
                            f"**{(1 - c_need / lab_[0][0]) * 100:.0f} % lower**. With lab peak strength the Bishop FOS "
                            "is about 1.07: above 1, so a 'FOS > 1' check would not have predicted the failure, but "
                            "below the DGMS minimum of 1.50, so T1 is Orange. After a failure, the back-analysed "
                            "strength should replace the lab strength in T1 and T2.")

    # ------------------------------------------------------------ calibration set
    with tabs[3]:
        st.markdown("Records added from the lab. Rule: **Yellow = lowest value 7 days before a failure, Orange = "
                    "lowest 3 days before, Red = lowest 1 day before** (velocity); for rain, Yellow / Orange / Red = "
                    "50 / 75 / 100 % of the lowest amount that preceded a rain-caused failure.")
        cs = pd.DataFrame(st.session_state.cal_cases)
        st.dataframe(cs, hide_index=True, **FULL)
        if len(cs):
            sg = suggest(cs[cs.get("kind", "") == "movement"]) if "kind" in cs else {}
            rain_cs = cs[(cs.get("kind", "") == "rain") & (cs.get("rain_triggered", False) == True)] \
                if "kind" in cs else pd.DataFrame()
            sr = suggest(rain_cs) if len(rain_cs) else {}
            rows, new = [], dict(thr)
            if "velocity" in sg:
                cur = list(thr["vel"])
                prop = [p if np.isfinite(p) else c_ for p, c_ in zip(sg["velocity"]["values"], cur)]
                prop = list(np.maximum.accumulate(prop))
                rows.append(["T4 velocity (mm/day)", cur, [round(x, 1) for x in prop], sg["velocity"]["n"]])
                new["vel"] = tuple(prop)
            if "ARI_at_failure" in sr:
                rows.append(["T3 antecedent rain (mm)", thr.get("ari") or "not used",
                             [round(x) for x in sr["ARI_at_failure"]["values"]], sr["ARI_at_failure"]["n"]])
                new["ari"] = tuple(sr["ARI_at_failure"]["values"])
            if rows:
                st.dataframe(pd.DataFrame(rows, columns=["Trigger", "Current (Y, O, R)", "Suggested (Y, O, R)",
                                                         "Failures used"]), hide_index=True, **FULL)
                if min(r_[3] for r_ in rows) < 3:
                    st.warning("Fewer than 3 failures: the suggestion is indicative only. Keep the source "
                               "thresholds until more failures or near-misses are added.")
                if st.button("Use the suggested thresholds in the live TARP"):
                    st.session_state.thr = new
                    st.success("Live TARP updated for this session.")
        if st.button("Reset to the source thresholds"):
            st.session_state.thr = default_thresholds()
            st.session_state.cal_cases = []
            st.rerun()


def default_thresholds():
    return dict(vel=T.VEL_DEFAULT, rain=T.RAIN_DEFAULT, ari=None)


# ======================================================================== 8
def data_page(FULL, n_physics, n_real, real):
    st.title("Data used and not used")
    st.markdown("<p class='lede'>What went into each trigger, from where, and what is still missing. Nothing in the "
                "TARP is taken from a source not listed here.</p>", unsafe_allow_html=True)
    meth = real.groupby("fos_method").size() if "fos_method" in real else pd.Series(dtype=int)
    used = pd.DataFrame([
        ["Bishop solutions of DGMS Reg. 106 dumps", f"{n_physics:,} rows", "In-house Bishop solver (checked against "
         "Slide2 and published cases)", "T1, T2 (AI FOS engine)", "Used"],
        ["Real FOS studies (SECL RS2, Jagannathpur FLAC, WCL FEM)", f"{n_real} rows", "NIT Rourkela (2025); "
         "Chourasia et al. (2025); Kainthola et al. (2011)", "T1, T2 field correction", "Used"],
        ["Sahoo et al. (2025) Mendeley dataset", "2,250 cases", "Sci. Rep. 15:40985", "Solver check", "Used"],
        ["DGMS minimum FOS", "1.50 / 1.30", "Reg. 106(2) circular para 3.2", "T1, T2 thresholds", "Used"],
        ["Velocity bands and survey frequency", "6 bands", "NIT Rourkela scientific-study notes", "T4, actions",
         "Used"],
        ["IMD rainfall classes", "3 classes", "India Meteorological Department", "T3 starting thresholds", "Used"],
        ["Indian failure history", "13 events", "DGMS circulars and alerts, papers, news (page 2)",
         "Choice of triggers, actions, consequence step", "Used"],
        ["Juna Kunada crack record", "2 readings", "DGMS alert 01-12-2017", "T4 cross-check (16 mm/day)", "Used"],
        ["Published precursor records (India and abroad)", "7 records", "Shellam & Coggan; Teck/WSP; Golder; Dwinagara",
         "Cross-check of T4, T3", "Used"],
        ["Synthetic monitoring records", "Demo A, Demo B, 40 test records", "Generated (Fukuzono creep, storm)",
         "Testing T5 and the replay only", "Used, labelled synthetic"],
    ], columns=["Data", "Size", "Source", "Used for", "Status"])
    st.subheader("Used")
    st.dataframe(used, hide_index=True, **FULL)
    st.subheader("Not used yet, and why")
    st.dataframe(pd.DataFrame([
        ["Day-by-day prism / SSR records before Indian dump failures", "Not published (Masood et al. 2025 used SECL "
         "SSR data without releasing values)", "Request SECL / NIT Rourkela radar and prism exports; load them in "
         "the back-analysis lab", "Calibrates T4, T5 for Indian dumps"],
        ["Daily rainfall before each failure", "Download blocked in the build environment", "NASA POWER links are on "
         "page 4; IMD station data from the mine", "Calibrates the antecedent-rain trigger"],
        ["Piezometer readings in Indian dumps", "Not available", "Install VW piezometers at mid-height and toe; the "
         "app learns the rain-to-r_u link from them", "Replaces the assumed r_u coefficient (T2)"],
        ["Undulating ground under the dump", "Solver assumes a flat base", "Add the surveyed ground line as the "
         "dump base in the Bishop solver", "T1 accuracy on sloping ground"],
        ["Multi-layer dump and weak foundation layer", "One material per dump at present", "Add layers with "
         "their own c, phi, gamma (Singareni, MKD-III, Jayant show why)", "T1 for soft-foundation dumps"],
        ["Seismic load", "Not included", "Pseudo-static coefficient (Chand & Koner 2023: FOS 1.24 -> 1.0 at 0.16)",
         "T1 in seismic zones"],
        ["Shift inspection logs", "Not available", "Use the T6 checklist every shift and keep the record",
         "Calibrates T6"],
        ["Near-miss records", "Not available", "Log every Orange / Red event and its outcome", "Tests false alarms"],
    ], columns=["Missing data", "Why not used", "How to get it", "What it improves"]), hide_index=True, **FULL)
    if len(meth):
        st.caption("Real FOS rows on file by method: " + "; ".join(f"{k}: {v}" for k, v in meth.items()))


# ======================================================================== 9
REFS = [
    ("Statutory and DGMS", [
        ("Coal Mines Regulations 2017, Reg. 106 (spoil banks) and Reg. 108, Govt. of India.", None),
        ("DGMS. Circular on the scientific study under Reg. 106(2) of CMR 2017, para 3.2 (minimum FOS) and para 4.1 "
         "(monitoring and TARP trigger points). Circular number and date to be added.", None),
        ("DGMS Tech. (S&T) Circular No. 02 of 2010, 06.07.2010. Design, control and monitoring of pit and dump slopes "
         "in opencast mines.", "https://elibrarywcl.wordpress.com/wp-content/uploads/2015/02/dgms-tech-st-circular-no-02-of-2010-pit-and-dump-slopes-in-opencast-mines.pdf"),
        ("DGMS Tech. Circular No. 02 of 2020, 09.01.2020. Guidelines for systematic monitoring of slopes (contents "
         "from a study-notes summary).", "https://onlineminingexam.graphy.com/blog/dgms-tech-circular-no-02-of-2020"),
        ("DGMS (2017) Alert: Juna Kunada OC, WCL, 01-12-2017.", "https://www.dgms.gov.in/writereaddata/UploadFile/Alert_Juna_Kunada_OC_WCL_DO_01-12-2017.pdf"),
        ("DGMS (2019) Alert: Bharatpur OCP, MCL, 23-07-2019.", "https://dgms.gov.in/writereaddata/UploadFile/Bharatpur%20OCP-FA-230719-alert.pdf"),
        ("Ministry of Coal (2024) Annual Report 2023-24, ch. 14 Safety in coal mines.", "https://coal.gov.in/sites/default/files/2024-07/chap14AnnualReport2024en2.pdf"),
        ("Ministry of Coal / PIB (2022) Safety measures in coal mines.", "https://www.coal.gov.in/sites/default/files/2022-01/PIB1776761.pdf"),
    ]),
    ("Indian failure cases", [
        ("Dash A.K. (2019) Current Science 117(2):304-308 - slope-failure accidents in Indian opencast mines, "
         "Table 1 (DGMS data) and a dragline-dump case study.", "https://i-scholar.in/index.php/CURS/article/download/187064/174214"),
        ("Mongabay (Feb 2017) Grief, anger and fear in the aftermath of a deadly coal mining disaster in Jharkhand state, India.",
         "https://news.mongabay.com/2017/02/grief-anger-and-fear-in-the-aftermath-of-a-deadly-coal-mining-disaster-in-jharkhand-state-india/"),
        ("Business Standard / IANS (2017) Lalmatia was not fit for mining: DGMS.", "https://business-standard.com/amp/article/news-ians/lalmatia-was-not-fit-for-mining-dgms-117011300587_1.html"),
        ("India Environment Portal (2013) Odisha coal dump crash toll rises to 14.", "https://indiaenvironmentportal.org.in/news-clippings/odisha-coal-dump-crash-toll-rises-to-14"),
        ("Deccan Herald. Landslide in coal mine in Chhattisgarh, one worker trapped (Kusmunda).", "https://www.deccanherald.com/amp/story/india%2Flandslide-in-coal-mine-in-chhattisgarh-one-worker-trapped-864840.html"),
        ("Shende et al. (2025) Phys. Chem. Earth 141:104091 - forensic investigation of a failed OB dump, central India.",
         "https://www.sciencedirect.com/science/article/abs/pii/S1474706525002414"),
        ("Poulsen B. et al. (2014) Mine overburden dump failure: a case study. Geotech. Geol. Eng. 32:297-309.",
         "https://link.springer.com/article/10.1007/s10706-013-9714-7"),
        ("Sharma S., Roy I. (2015) Slope failure of waste rock dump at Jayant opencast mine, India. Int. J. Appl. "
         "Eng. Res. 10(13):33006-33012.", None),
        ("Kainthola A., Verma D., Gupte S.S., Singh T.N. (2011) A coal mine dump stability analysis - a case study. "
         "Geomaterials 1:1-13.", "https://www.scirp.org/html/4675.html"),
        ("Jhanwar J.C. (2010) Investigation into slope failure in an opencast coal mine of Wardha Valley coalfield. "
         "Indian Mining & Engg. J. 49(4):36-40.", "https://cimfr.csircentral.net/743/"),
        ("Bishwal R.M. (2019) List of major coal mine overburden dump failures in India (1995-2019). Mendeley Data "
         "v2, doi:10.17632/ptcbv7cngh.2 (files not public).", "https://data.mendeley.com/datasets/ptcbv7cngh/2"),
    ]),
    ("Monitoring, early warning and TARP", [
        ("Carla T. et al. (2017) Guidelines on the use of inverse velocity method as a tool for setting alarm "
         "thresholds and forecasting landslides and structure collapses. Landslides 14(2):517-534.",
         "https://flore.unifi.it/handle/2158/1044054"),
        ("Fukuzono T. (1985) A new method for predicting the failure time of a slope. 4th Int. Conf. Field Workshop "
         "on Landslides, Tokyo, 145-150.", None),
        ("Dick G.J. et al. (2015) Development of an early-warning time-of-failure analysis methodology for open-pit "
         "mine slopes utilizing ground-based slope stability radar. Can. Geotech. J. 52:515-529.",
         "https://www.eoas.ubc.ca/personal/erik/e-papers/15EE_CanGeotechJ-SlopeRadar.pdf"),
        ("Shellam & Coggan (2020) Slope Stability 2020 (ACG) - SSR velocity and acceleration trends for deformation TARPs.",
         "https://papers.acg.uwa.edu.au/d/2025_10_Shellam/10_Shellam.pdf"),
        ("Masood M.M., Verma T., Raju G.Y. (2025) Development of a filtered inverse velocity method analyser. J. SAIMM "
         "125(8):477-484.", "https://www.scielo.org.za/pdf/jsaimm/v125n8/10.pdf"),
        ("Masood, Verma & Murmu (2023) ICETE 2023 - use of radar in assessing mine slope stability, Kusmunda.",
         "https://www.atlantis-press.com/article/125994195.pdf"),
        ("Mirmazloumi et al. (2023) Bull. Eng. Geol. Environ. 82:374 - InSAR time series and LSTM for early warning "
         "at mining sites.", "https://link.springer.com/content/pdf/10.1007/s10064-023-03388-w.pdf"),
        ("Farina et al. (2020) Slope Stability 2020 (ACG) - rainfall-threshold early warning system, Grasberg.",
         "https://flore.unifi.it/bitstream/2158/1217193/1/Farina%20et%20al%20ACG%202020.pdf"),
        ("WSP Canada (2023) 2022 Annual Facility Performance Report - Coal Reject Spoils, Fording River (Teck Coal).",
         "https://www.teck.com/media/2022-AFPR-FRO-Coal-Reject-Spoils.pdf"),
        ("Golder Associates & Hungr O. (1995) Runout characteristics of debris from dump failures in mountainous "
         "terrain, Stage 2. BC Mine Waste Rock Pile Research Committee.", "https://www2.gov.bc.ca/assets/gov/farming-natural-resources-and-industry/mineral-exploration-mining/documents/permitting/geotechnical-information/mined_rock_and_overburden_piles_stage_2_feb_1995.pdf"),
        ("Dwinagara B. et al. (2024) Int. J. GEOMATE 27(123):54-64 - back analysis of a rainfall-induced coal waste "
         "dump failure, East Kalimantan.", "https://geomatejournal.com/geomate/article/download/4618/3515/9006"),
        ("Chand & Koner (2023) J. Mining Environ. 14(4) - internal dump stability with UAV-based 3D modelling, Raniganj.", "https://jme.shahroodut.ac.ir/article_2900_53bae7cf957d816026b23afccfce415d.pdf"),
        ("Chand et al. (2024) Sadhana 49:277 - dump slope change detection with UAV photogrammetry.", None),
    ]),
    ("Dump stability and FOS engine", [
        ("NIT Rourkela (2025) Safety and feasibility study of filling of fly ash in eight opencast mines of SECL, "
         "Project C3/23/MN/049 (and scientific-study notes on slope-movement bands).", None),
        ("Chourasia K., Ravi S., Khare R.N. (2025) Jagannathpur OCP dump optimisation. Research Square rs-8013159.", None),
        ("Sahoo A.K., Tripathy D.P., Jayanthu S. (2025) Sci. Rep. 15:40985 and Mendeley dataset.", None),
        ("Singh U.C. et al. (2022) Indian Geotech. J. 52:735-741 - scattered soft soil in coal OB dumps (FLAC3D).",
         "https://link.springer.com/article/10.1007/s40098-022-00596-1"),
        ("Geete et al. (2023) J. Mines Metals Fuels 71(9) - review of OB dump slope stability analysis.",
         "https://informaticsjournals.co.in/index.php/jmmf/article/download/35438/22858/69257"),
        ("Nayak, Dash & Dewangan (2021) J. Mines Metals Fuels 69(7) - rainfall and groundwater effects on a coal "
         "waste dump.", "https://informaticsjournals.co.in/index.php/jmmf/article/view/28542"),
        ("Sahu, Mohanty & Jayanthu (2023) 10th Asian Mining Congress - dump slope stability rating and risk assessment.",
         "https://link.springer.com/chapter/10.1007/978-3-031-46966-4_7"),
        ("Bishop A.W. (1955) The use of the slip circle in the stability analysis of slopes. Geotechnique 5:7-17.", None),
        ("Bishop A.W., Morgenstern N. (1960) Stability coefficients for earth slopes. Geotechnique 10:129-150.", None),
        ("Read J., Stacey P. (2009) Guidelines for Open Pit Slope Design. CSIRO.", None),
        ("India Meteorological Department. Rainfall intensity classes (heavy 64.5-115.5, very heavy 115.6-204.4, "
         "extremely heavy >= 204.5 mm in 24 h).", None),
    ]),
]


def references_page():
    st.title("References")
    st.markdown("<p class='lede'>Every source used on these pages, grouped by what it supports.</p>",
                unsafe_allow_html=True)
    for grp, items in REFS:
        st.subheader(grp)
        st.markdown("\n".join(f"- {t}" + (f" [link]({u})" if u else "") for t, u in items))


# ======================================================================== 5
SENSORS = {
    "Total station + prisms": ["T4", "T5"],
    "Slope stability radar (SSR)": ["T4", "T5"],
    "GNSS monitors": ["T4", "T5"],
    "Crack meters / extensometers": ["T4", "T6"],
    "Vibrating-wire piezometers": ["T2"],
    "Rain gauge (or IMD station)": ["T3", "T2"],
    "UAV / drone survey": ["T1", "T6"],
    "InSAR (Sentinel-1)": ["T4"],
}


def _cross(x, y, level):
    """First x where y falls to `level` (y decreasing in x); None if it never does."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if y[0] <= level:
        return float(x[0])
    k = np.where(y <= level)[0]
    if not len(k):
        return None
    i = k[0]
    return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))


def site_tarp_page(FULL, PLOT, model, row, material_inputs, geometry_inputs, DumpGeometry, FOS_MIN, cons):
    st.title("TARP for your dump")
    st.markdown("<p class='lede'>Enter the dump and the instruments the mine has. The AI FOS engine is run "
                "hundreds of times to find the <b>site-specific trigger values</b>: the pore pressure, piezometer "
                "head, rainfall and dump height at which this dump reaches Yellow, Orange and Red. Together with "
                "the movement and visual triggers they form the dump's Trigger Action Response Plan, ready to "
                "download.</p>", unsafe_allow_html=True)
    with st.expander("1. The dump", expanded=True):
        site = st.text_input("Dump / mine name", "Gevra OCP, external OB dump", key="st_site")
        c, phi, g = material_inputs("st")
        geom = geometry_inputs("st")
        k1, k2, k3 = st.columns(3)
        ru_d = k1.number_input("Design r_u", 0.0, 0.6, 0.0, 0.01, key="st_rud",
                               help="Pore-pressure ratio assumed in the design (0 = drained).")
        z_tip = k2.number_input("Piezometer tip depth below the surface (m)", 1.0, 100.0, 10.0, 0.5, key="st_z")
        kappa = k3.number_input("r_u per mm of antecedent rain", 0.0001, 0.01, 0.0012, 0.0001, format="%.4f",
                                key="st_kap", help="Learned on the Live TARP page when rain and piezometer data "
                                                    "are uploaded; this default is an assumption.")
        people = st.toggle("People, haul road, buildings or a nala lie in the run-out zone below the toe",
                           value=True, key="st_people")
    with st.expander("2. Instruments available at the mine", expanded=True):
        have = st.multiselect("Tick what is installed or can be used", list(SENSORS),
                              default=["Total station + prisms", "Rain gauge (or IMD station)"], key="st_have")

    # ---------------------------------------------------------------- AI runs
    RU_MAX, H_MAX = 0.45, 120.0          # limits of the training data (beyond = extrapolation)
    rus = np.round(np.arange(0.0, RU_MAX + 0.001, 0.01), 2)
    X = pd.concat([row(c, phi, g, r_, geom) for r_ in rus], ignore_index=True)
    pr = model.predict(X)
    des = model.predict(row(c, phi, g, ru_d, geom)).iloc[0]
    t1 = T.fos_trigger(des.FOS, FOS_MIN, des.FOS_lo)
    ru_y = _cross(rus, pr.FOS_lo.values, FOS_MIN)
    ru_o = _cross(rus, pr.FOS.values, FOS_MIN)
    ru_r = _cross(rus, pr.FOS.values, 1.0)
    head = lambda r_: None if r_ is None else r_ * g * z_tip / 9.81
    ari = lambda r_: None if r_ is None else r_ / kappa
    # dump raised deck by deck at the design r_u
    ns = list(range(1, max(1, int(H_MAX // geom.deck_height)) + 1))
    Xh = pd.concat([row(c, phi, g, ru_d, DumpGeometry(n_, geom.deck_height, geom.deck_angle,
                                                       geom.berm_width if n_ > 1 else 0.0)) for n_ in ns],
                   ignore_index=True)
    ph_ = model.predict(Xh)
    ok_n = [n_ for n_, f_ in zip(ns, ph_.FOS.values) if f_ >= FOS_MIN]
    h_max = max(ok_n) * geom.deck_height if ok_n else 0.0
    # cohesion loss (softening when wet) that brings the design to the minimum
    ks = np.round(np.arange(1.0, 0.09, -0.05), 2)
    pc = model.predict(pd.concat([row(c * k_, phi, g, ru_d, geom) for k_ in ks], ignore_index=True))
    k_o = _cross(1 - ks, pc.FOS.values, FOS_MIN)

    f = lambda v, fmt: "not reached" if v is None else fmt.format(v)
    st.markdown(f"<div class='tarp' style='background:{T.COL[t1]}'><b>{site}</b><br>Design FOS {des.FOS:.2f} "
                f"(90 % range {des.FOS_lo:.2f}-{des.FOS_hi:.2f}) against the DGMS minimum {FOS_MIN:.2f} "
                f"({cons.lower()}): T1 = <b style='font-size:1rem'>{t1}</b>.</div>", unsafe_allow_html=True)
    m = st.columns(4)
    m[0].metric("Orange at r_u", f(ru_o, "{:.2f}"), f(head(ru_o), "piezometer head {:.1f} m"), delta_color="off")
    m[1].metric("Red at r_u", f(ru_r, "{:.2f}"), f(head(ru_r), "piezometer head {:.1f} m"), delta_color="off")
    top = len(ok_n) == len(ns)
    m[2].metric("Highest dump meeting the minimum", (">= " if top else "") + f"{h_max:.0f} m",
                f"{len(ok_n)} decks of {geom.deck_height:.0f} m" + (" (model limit 120 m)" if top else ""),
                delta_color="off")
    m[3].metric("Cohesion loss to reach the minimum", f(None if k_o is None else k_o * 100, "{:.0f} %"),
                "softening when wet", delta_color="off")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rus, y=pr.FOS_hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=rus, y=pr.FOS_lo, fill="tonexty", fillcolor="rgba(47,107,106,0.18)",
                             line=dict(width=0), name="90 % range"))
    fig.add_trace(go.Scatter(x=rus, y=pr.FOS, line=dict(color="#1C2B2D", width=3), name="AI FOS"))
    fig.add_hline(y=FOS_MIN, line_dash="dot", line_color=T.COL["Orange"], annotation_text=f"DGMS minimum {FOS_MIN:.2f}")
    fig.add_hline(y=1.0, line_dash="dot", line_color=T.COL["Red"], annotation_text="FOS 1.0")
    for v_, lv in [(ru_y, "Yellow"), (ru_o, "Orange"), (ru_r, "Red")]:
        if v_ is not None:
            fig.add_vline(x=v_, line_color=T.COL[lv], annotation_text=f"{lv} r_u {v_:.2f}")
    fig.update_layout(**PLOT, title="How water in the dump moves it through the TARP levels (AI FOS engine)")
    fig.update_xaxes(title="Pore-pressure ratio r_u")
    fig.update_yaxes(title="FOS")
    st.plotly_chart(fig, **FULL)
    if ru_r is None or ru_o is None:
        st.info(f"{'Red' if ru_o is not None else 'Orange and Red'} not reached up to r_u {RU_MAX} (the top of the "
                "training data, close to a fully saturated dump, where r_u is about gamma_w / gamma = "
                f"{9.81 / g:.2f}). Water alone is unlikely to take this dump to that level; movement and visual "
                "triggers still apply.")
    st.caption("Yellow: the 90 % range touches the minimum. Orange: the AI FOS equals the minimum. Red: FOS 1.0. "
               f"Piezometer head h = r_u x gamma x z / gamma_w with z = {z_tip:g} m. Rain equivalent = r_u / "
               f"{kappa:.4f} per mm of antecedent rain (assumed until learned from site data).")

    # ---------------------------------------------------------------- the TARP sheet
    thr = st.session_state.thr
    v = thr["vel"]
    rows_ = [
        ["T1 Design FOS", "AI FOS engine (survey section + lab c, phi, gamma)",
         f"90 % range below {FOS_MIN:.2f}", f"FOS below {FOS_MIN:.2f}; or dump raised above {h_max:.0f} m",
         "FOS below 1.0", True],
        ["T2 Live FOS (water)", "AI FOS with today's r_u",
         f"r_u {f(ru_y, '{:.2f}')} (head {f(head(ru_y), '{:.1f} m')})",
         f"r_u {f(ru_o, '{:.2f}')} (head {f(head(ru_o), '{:.1f} m')})",
         f"r_u {f(ru_r, '{:.2f}')} (head {f(head(ru_r), '{:.1f} m')})",
         "Vibrating-wire piezometers" in have or "Rain gauge (or IMD station)" in have],
        ["T3 Rainfall", "24-h rain (IMD classes) and antecedent rain index from the AI r_u limits",
         f"{thr['rain'][0]:g} mm/24 h or ARI {f(ari(ru_y), '{:.0f} mm')}",
         f"{thr['rain'][1]:g} mm/24 h or ARI {f(ari(ru_o), '{:.0f} mm')}",
         f"{thr['rain'][2]:g} mm/24 h or ARI {f(ari(ru_r), '{:.0f} mm')}", "Rain gauge (or IMD station)" in have],
        ["T4 Velocity", "Prism / radar / GNSS / crack meter", f"{v[0]:g} mm/day", f"{v[1]:g} mm/day",
         f"{v[2]:g} mm/day", any(s in have for s in ("Total station + prisms", "Slope stability radar (SSR)",
                                                     "GNSS monitors", "Crack meters / extensometers",
                                                     "InSAR (Sentinel-1)"))],
        ["T5 Acceleration", "Moving averages + inverse velocity", "Onset of acceleration",
         "Failure forecast < 7 days, v >= 2 mm/day", "Failure forecast < 3 days, v >= 10 mm/day",
         any(s in have for s in ("Total station + prisms", "Slope stability radar (SSR)", "GNSS monitors"))],
        ["T6 Visual", "Shift inspection checklist", "New cracks, settlement, seepage",
         "Widening cracks, toe bulge, muddy outflow", "Scarp, visible movement", True],
    ]
    sheet = pd.DataFrame(rows_, columns=["Trigger", "How it is measured", "Yellow", "Orange", "Red", "Active"])
    sheet["Active"] = sheet["Active"].map({True: "yes", False: "NO - install instrument"})
    st.subheader("Trigger values for this dump")
    st.dataframe(sheet, hide_index=True, **FULL)
    missing = sheet[sheet.Active != "yes"].Trigger.tolist()
    if missing:
        st.warning("Not active at this mine: " + ", ".join(missing) + ". Install " +
                   ("vibrating-wire piezometers or a rain gauge" if "T2 Live FOS (water)" in missing else "") +
                   ("; a rain gauge" if "T3 Rainfall" in missing else "") +
                   ("; prisms with a total station (radar for high-risk dumps)" if "T4 Velocity" in missing else "")
                   + ".")

    st.subheader("Response plan")
    acts = pd.DataFrame(T.ACTION_TABLE).rename(columns={"Response_time": "Response time"})
    if people:
        st.markdown("People, roads or structures are in the run-out zone, so **every Yellow is handled as Orange "
                    "and every Orange as Red**, except when only T1 or T3 has fired (those stop at Orange).")
    st.dataframe(acts, hide_index=True, **FULL)
    st.subheader("Monitoring schedule")
    st.dataframe(pd.DataFrame([
        ["Green", "Monthly prism survey; rain gauge daily; shift inspection", "Surveyor; overman"],
        ["Yellow", "Weekly to daily survey; piezometers daily; crack pegs read every shift",
         "Slope-stability engineer"],
        ["Orange", "Daily survey or continuous radar / GNSS; piezometers every few hours", "Assistant Manager"],
        ["Red", "Continuous radar / GNSS from a safe point; no one in the zone", "Mine Manager"],
    ], columns=["Level", "Readings", "Who reads them"]), hide_index=True, **FULL)

    st.subheader("Download the TARP")
    html = _tarp_html(site, des, t1, FOS_MIN, cons, sheet, acts, people, c, phi, g, geom, ru_d, z_tip, kappa)
    c1, c2 = st.columns(2)
    c1.download_button("TARP sheet (HTML, print to PDF)", html, f"TARP_{site.split(',')[0].replace(' ', '_')}.html",
                       "text/html")
    c2.download_button("Trigger values (CSV)", sheet.to_csv(index=False), "TARP_triggers.csv", "text/csv")

    with st.expander("How reliable is the AI FOS engine behind T1 and T2?"):
        rep = getattr(model, "report", {})
        if "surrogate_holdout" in rep:
            s_ = rep["surrogate_holdout"]
            st.markdown(f"- Against 20 % of the Bishop solutions held out of training: R2 **{s_['R2']:.3f}**, mean error "
                        f"**{s_['MAPE']:.1f} %**.")
        if "loo" in rep:
            t_ = rep["loo"]["table"]
            st.markdown("- On real studies, each dump site left out in turn (leave-one-site-out):")
            st.dataframe(t_.round(3), **FULL)
        st.markdown("- The 90 % range comes from split-conformal calibration on the real cases, so the Yellow "
                    "trigger already allows for the model's error.")


def _tarp_html(site, des, t1, fmin, cons, sheet, acts, people, c, phi, g, geom, ru_d, z, kappa):
    css = ("body{font-family:'Times New Roman',serif;margin:28px;color:#1c2b2d}h1{font-size:22px}"
           "h2{font-size:17px;margin-top:22px}table{border-collapse:collapse;width:100%;font-size:13px}"
           "td,th{border:1px solid #999;padding:5px;vertical-align:top;text-align:left}th{background:#e7ecea}"
           ".G{background:#2e7d4f;color:#fff}.Y{background:#d8a31a;color:#fff}.O{background:#d9731f;color:#fff}"
           ".R{background:#b03a2e;color:#fff}")
    def tab(df):
        h = df.to_html(index=False, escape=True).replace('style="text-align: right;"', "")
        for lv in ("Green", "Yellow", "Orange", "Red"):
            h = h.replace(f"<td>{lv}</td>", f"<td class='{lv[0]}'><b>{lv}</b></td>")
        return h
    note = ("People, roads or structures are in the run-out zone: Yellow is handled as Orange and Orange as Red "
            "(T1 or T3 alone stop at Orange)." if people else "No people or structures in the run-out zone.")
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>TARP - {site}</title><style>{css}</style>
</head><body><h1>Trigger Action Response Plan - {site}</h1>
<p>Dump: {geom.n_decks} decks x {geom.deck_height:g} m at {geom.deck_angle:g} deg, bench {geom.berm_width:g} m
(total {geom.height:g} m, overall {geom.overall_angle:.1f} deg). Material: c {c:g} kPa, phi {phi:g} deg,
gamma {g:g} kN/m3. Design r_u {ru_d:g}. Piezometer tip {z:g} m. Rain-to-r_u coefficient {kappa:.4f} per mm.</p>
<p>Design FOS (AI engine) {des.FOS:.2f}, 90 % range {des.FOS_lo:.2f}-{des.FOS_hi:.2f}; DGMS minimum {fmin:.2f}
({cons.lower()}, Reg. 106(2) circular para 3.2). T1 level: <b>{t1}</b>.</p>
<h2>Triggers</h2>{tab(sheet)}<p>The worst trigger sets the TARP level. {note}</p>
<h2>Actions and responsibilities</h2>{tab(acts)}
<p>Responsibilities and response times are proposed for adoption in the mine's Safety Management Plan.
Thresholds are re-set after every failure or near-miss by back-analysis.</p>
<p style='font-size:11px'>Generated by DumpSafe AI.</p></body></html>"""
