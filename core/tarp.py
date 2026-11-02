"""Multi-parameter TARP (Trigger Action Response Plan) for OB dump slopes.

Six triggers are each given a level (Green / Yellow / Orange / Red); the worst one
governs, and the level is raised one step when people, haul roads or structures lie
in the run-out zone below the dump (design FOS and rainfall, being causes rather than
evidence of movement, can be raised to Orange at most).

T1  Design FOS   - AI-predicted FOS of the design (dry / design r_u), DGMS para 3.2
T2  Live FOS     - the same model re-run with today's pore pressure (piezometer or rain)
T3  Rainfall     - 24 h rainfall, IMD intensity classes (site calibration needed)
T4  Velocity     - slope-movement bands of the NIT Rourkela scientific-study notes
T5  Acceleration - onset of acceleration and inverse-velocity window (Carla et al. 2017)
T6  Visual signs - shift inspection checklist (cracks, bulging, seepage, ...)

The responsibilities and response times are a PROPOSAL to be adopted through the
mine's Safety Management Plan; they are not quoted from a DGMS circular.
"""
from __future__ import annotations
import numpy as np, pandas as pd

ORDER = ["Green", "Yellow", "Orange", "Red"]
COL = {"Green": "#2e7d4f", "Yellow": "#d8a31a", "Orange": "#d9731f", "Red": "#b03a2e",
       "No data": "#8a9597"}


def worst(*levels):
    lv = [x for x in levels if x in ORDER]
    return max(lv, key=ORDER.index) if lv else "Green"


def step_up(level):
    return ORDER[min(ORDER.index(level) + 1, 3)]


# ------------------------------------------------------------------ T1 / T2 FOS
def fos_trigger(fos, fos_min, fos_lo=None):
    """Red: FOS < 1.0 (failure predicted). Orange: below the DGMS minimum.
    Yellow: within 0.05 of the minimum, or the lower end of the 90 % range below it."""
    if fos is None or not np.isfinite(fos):
        return "No data"
    if fos < 1.0:
        return "Red"
    if fos < fos_min:
        return "Orange"
    if fos < fos_min + 0.05 or (fos_lo is not None and fos_lo < fos_min):
        return "Yellow"
    return "Green"


# ------------------------------------------------------------------ T3 rainfall
# IMD daily rainfall classes: heavy 64.5-115.5, very heavy 115.6-204.4,
# extremely heavy >= 204.5 mm in 24 h.
RAIN_DEFAULT = (64.5, 115.6, 204.5)


def rain_trigger(rain_24h, thresholds=RAIN_DEFAULT):
    if rain_24h is None or not np.isfinite(rain_24h):
        return "No data"
    y, o, r = thresholds
    return "Red" if rain_24h >= r else "Orange" if rain_24h >= o else "Yellow" if rain_24h >= y else "Green"


def rolling_sum(t_days, x, width=1.0):
    t, x = np.asarray(t_days, float), np.nan_to_num(np.asarray(x, float))
    return np.array([x[(t > ti - width) & (t <= ti)].sum() for ti in t])


def antecedent_index(t_days, rain, decay=0.90):
    t, r = np.asarray(t_days, float), np.nan_to_num(np.asarray(rain, float))
    ari = np.zeros(len(t))
    for i in range(1, len(t)):
        ari[i] = ari[i - 1] * decay ** (t[i] - t[i - 1]) + r[i]
    return ari


def ru_from_piezometer(head_m, depth_m, gamma=18.6, gamma_w=9.81):
    """r_u = u / (gamma z) with u = gamma_w h (Bishop & Morgenstern 1960)."""
    return np.clip(gamma_w * np.asarray(head_m, float) / (gamma * depth_m), 0, 1)


def ari_trigger(ari, thresholds):
    """Antecedent rainfall index trigger; thresholds (Yellow, Orange, Red) in mm, or None
    until they are calibrated by back-analysis of past failures."""
    if thresholds is None or ari is None or not np.isfinite(ari):
        return "Green"
    y, o, r = thresholds
    return "Red" if ari >= r else "Orange" if ari >= o else "Yellow" if ari >= y else "Green"


# ------------------------------------------------------------------ T4 velocity
VEL_MAP = {"Normal": "Green", "Watch": "Green", "Caution": "Yellow", "Alert": "Orange",
           "Alarm": "Red", "Stop work": "Red"}
# Yellow / Orange / Red from (mm/day): Caution, Alert and Alarm bands of the NIT Rourkela notes
VEL_DEFAULT = (5.0, 10.0, 50.0)


def vel_trigger(v, thresholds=VEL_DEFAULT):
    y, o, r = thresholds
    return "Red" if v >= r else "Orange" if v >= o else "Yellow" if v >= y else "Green"


# ------------------------------------------------------------------ T5 acceleration
def accel_trigger(onset, window, velocity=None, red_days=3.0, orange_days=7.0, red_min_v=10.0,
                  orange_min_v=2.0):
    """Yellow once acceleration has started (SMA above LMA); Orange when the earliest
    inverse-velocity forecast is within `orange_days` AND the velocity is above the
    Normal band (`orange_min_v`); Red when it is within `red_days` AND the velocity is at
    least `red_min_v` mm/day (Alert band). The velocity conditions stop a noisy 1/v fit
    on a barely moving slope from raising the level on its own."""
    v_ok = (lambda lim: velocity is None or velocity >= lim)
    if onset is None:
        return "Green"
    if window is None:
        return "Yellow"
    if window["days_early"] < red_days and v_ok(red_min_v):
        return "Red"
    if window["days_early"] < orange_days and v_ok(orange_min_v):
        return "Orange"
    return "Yellow"


# ------------------------------------------------------------------ T6 visual
VISUAL_SIGNS = [
    ("New tension cracks on the crest or bench", "Yellow"),
    ("Settlement / sagging of the crest", "Yellow"),
    ("Seepage or wet patches at the toe or bench face", "Yellow"),
    ("Cracks widening or lengthening since the last shift", "Orange"),
    ("Bulging or heave at the toe", "Orange"),
    ("Muddy water or soft-soil outflow at the toe", "Orange"),
    ("Scarp formed / step in the crest, debris falling continuously", "Red"),
    ("Visible movement of the dump mass or toe pushing onto the haul road", "Red"),
]


def visual_trigger(ticked):
    return worst(*[lvl for (s, lvl) in VISUAL_SIGNS if s in ticked]) if ticked else "Green"


# ------------------------------------------------------------------ actions
ACTION_TABLE = [
    dict(Level="Green", Meaning="All triggers normal",
         Action="Routine prism / total-station survey at the frequency of the velocity band; shift "
                "inspection of crest, benches and toe; keep drains clear.",
         Responsible="Surveyor; shift overman / foreman", Response_time="Routine (as per schedule)"),
    dict(Level="Yellow", Meaning="One trigger at warning level",
         Action="Double the survey frequency; map and peg cracks; check drains and piezometers; "
                "geotechnical review of the trigger; inform the Manager.",
         Responsible="Slope-stability (geotech) engineer; Assistant Manager",
         Response_time="Within 24 h"),
    dict(Level="Orange", Meaning="A trigger at alert level",
         Action="Stop dumping on the affected deck; restrict people and vehicles below the toe and "
                "near the crest of the zone; daily or continuous monitoring (radar if available); "
                "start remedial work (unload crest, drain, toe buttress).",
         Responsible="Mine Manager (decision); Assistant Manager and geotech engineer (execution)",
         Response_time="Same shift (within 8 h)"),
    dict(Level="Red", Meaning="Failure imminent or in progress",
         Action="Stop all work in the influence zone; evacuate and barricade the run-out area; "
                "continuous watch from a safe point; report to the Agent / owner and DGMS as "
                "required under CMR 2017; no re-entry until a geotechnical review clears it.",
         Responsible="Mine Manager; Agent / owner",
         Response_time="Immediately"),
]


# T1 (design) and T3 (rain) are causes, not evidence of movement: with the consequence
# step they can raise the level to Orange at most. Red (evacuation) needs a direct trigger.
INDIRECT = ("T1 Design FOS", "T3 Rainfall")


def evaluate(triggers: dict, people_below=False):
    """triggers: {name: level or 'No data'}.
    Returns (final level, governing triggers, note, triggers without data)."""
    valid = {k: v for k, v in triggers.items() if v in ORDER}
    base = worst(*valid.values())
    gov = [k for k, v in valid.items() if v == base] if base != "Green" else []
    final, note = base, ""
    if people_below and base != "Green":
        final = step_up(base)
        if final == "Red" and base != "Red" and all(k in INDIRECT for k in gov):
            final = "Orange"
        if final != base:
            note = "raised one level: people, haul road or structures in the run-out zone"
    missing = [k for k, v in triggers.items() if v not in ORDER]
    return final, gov, note, missing


# ------------------------------------------------------------------ history (for lead time)
def level_history(d: pd.DataFrame, fos_min, rain_thr=RAIN_DEFAULT, ratio=1.3,
                  persist_days=2.0, noise_floor=0.2, red_days=3.0, orange_days=7.0,
                  red_min_v=10.0, long_days=5.0, fit_days=5.0, orange_min_v=2.0,
                  vel_thr=VEL_DEFAULT, ari_thr=None):
    """Re-runs the automatic triggers (T2-T5) at every reading as if it were 'now', so the
    page can show when each trigger would have fired and the warning lead time."""
    from core.monitoring import onset_of_acceleration, iv_window
    out = []
    for i in range(len(d)):
        sub = d.iloc[: i + 1]
        row = sub.iloc[-1]
        lv = dict(t_days=row.t_days)
        lv["T2 Live FOS"] = fos_trigger(row.get("FOS", np.nan), fos_min)
        lv["T3 Rainfall"] = worst(rain_trigger(row.get("rain_24h", np.nan), rain_thr),
                                  ari_trigger(row.get("ARI_mm", np.nan), ari_thr)) \
            if ("rain_24h" in d and np.isfinite(row.get("rain_24h", np.nan))) else "No data"
        lv["T4 Velocity"] = vel_trigger(row.velocity_mm_day, vel_thr)
        if i >= 8:
            on = onset_of_acceleration(sub, ratio, persist_days, noise_floor, long_days)
            w = iv_window(sub, on, noise_floor, fit_days=fit_days) if on is not None else None
        else:
            on, w = None, None
        lv["T5 Acceleration"] = accel_trigger(on, w, row.velocity_mm_day, red_days, orange_days,
                                              red_min_v, orange_min_v)
        lv["Overall"] = worst(lv["T2 Live FOS"], lv["T3 Rainfall"], lv["T4 Velocity"],
                              lv["T5 Acceleration"])
        out.append(lv)
    return pd.DataFrame(out)


def first_time(hist, col, level):
    """First time (days) a column reaches `level` or worse."""
    k = hist[col].map(lambda x: ORDER.index(x) if x in ORDER else -1) >= ORDER.index(level)
    return float(hist.t_days[k].iloc[0]) if k.any() else None
