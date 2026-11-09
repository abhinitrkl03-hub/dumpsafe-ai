"""Five-trigger TARP (Trigger Action Response Plan) for coal-mine OB dumps.

T1 FOS           - AI FOS of the dump with today's water condition (dry, piezometer, or r_u)
T2 Rainfall      - rain in the last 24 h and in the last 15 days
T3 Velocity      - movement rate from prisms / GNSS / radar / crack meters
T4 Acceleration  - speeding up (velocity ratio) and inverse-velocity time to failure
T5 Field signs   - cracks, settlement, bulging, seepage seen during shift inspection

Every threshold is taken from DGMS, the NIT Rourkela scientific-study notes, IMD, or from
FOS / movement / rainfall recorded before real failures (see evidence.py). The worst
trigger sets the level; it is raised one step when people, haul roads or structures are in
the run-out zone (rainfall alone stops at Orange).
"""
from __future__ import annotations
import numpy as np

ORDER = ["Green", "Yellow", "Orange", "Red"]
COL = {"Green": "#2e7d4f", "Yellow": "#d8a31a", "Orange": "#d9731f", "Red": "#b03a2e", "No data": "#8a9597"}
NAMES = ["T1 FOS", "T2 Rainfall", "T3 Velocity", "T4 Acceleration", "T5 Field signs"]


def worst(*levels):
    lv = [x for x in levels if x in ORDER]
    return max(lv, key=ORDER.index) if lv else "Green"


def step_up(level):
    return ORDER[min(ORDER.index(level) + 1, 3)]


# ------------------------------------------------------------------ T1 FOS
RED_FOS = 1.10      # every back-analysed failed dump had FOS 0.72-1.07 at failure (evidence.py)


def fos_bands(fos_min):
    """(Yellow-from, Orange-from, Red-from) as FOS values: Green >= DGMS minimum; Yellow down
    to 1.30 for permanent dumps (still meets the minimum for working dumps) or 1.20 for
    working dumps; Orange down to 1.10; Red below 1.10, the range of failed dumps."""
    return (fos_min, 1.30 if fos_min > 1.30 else 1.20, RED_FOS)


def fos_trigger(fos, fos_min):
    if fos is None or not np.isfinite(fos):
        return "No data"
    g, y, o = fos_bands(fos_min)
    return "Green" if fos >= g else "Yellow" if fos >= y else "Orange" if fos >= o else "Red"


# ------------------------------------------------------------------ T2 rainfall
RAIN_24H = (25.0, 64.5, 115.6)   # Bingham Canyon trigger; IMD heavy; IMD very heavy (mm in 24 h)
RAIN_15D_ORANGE = 350.0          # East Kalimantan dump: 353.5 mm in 17 days before failure


def rain_trigger(rain_24h, rain_15d=0.0):
    if rain_24h is None or not np.isfinite(rain_24h):
        return "No data"
    y, o, r = RAIN_24H
    lv = "Red" if rain_24h >= r else "Orange" if rain_24h >= o else "Yellow" if rain_24h >= y else "Green"
    if rain_15d is not None and rain_15d >= RAIN_15D_ORANGE:
        lv = worst(lv, "Orange")
    return lv


def ru_from_piezometer(head_m, depth_m, gamma=18.6, gamma_w=9.81):
    """r_u = u / (gamma z) with u = gamma_w h (Bishop & Morgenstern 1960)."""
    return float(np.clip(gamma_w * head_m / (gamma * depth_m), 0, 1))


# ------------------------------------------------------------------ T3 velocity
VEL = (5.0, 10.0, 50.0)   # Caution / Alert / Alarm bands of the NIT Rourkela notes (mm/day)


def vel_trigger(v):
    if v is None or not np.isfinite(v):
        return "No data"
    y, o, r = VEL
    return "Red" if v >= r else "Orange" if v >= o else "Yellow" if v >= y else "Green"


# ------------------------------------------------------------------ T4 acceleration
VR_ORANGE = 1.3       # velocity ratio trigger at Bingham Canyon (1.3-1.5; Bakken et al. 2020)
IV_RED_DAYS = 2.0     # inverse velocity predicts failure within 2 days (Bingham: 1-3 days, 36 h, 2 days)


def accel_trigger(t_onset, vr, window):
    if t_onset is None:
        return "Green"
    if window is not None and window["days_early"] <= IV_RED_DAYS:
        return "Red"
    if vr is not None and np.isfinite(vr) and vr >= VR_ORANGE:
        return "Orange"
    return "Yellow"


# ------------------------------------------------------------------ T5 field signs
VISUAL_SIGNS = [
    ("New tension cracks on the crest or bench", "Yellow"),
    ("Settlement / sagging of the crest", "Yellow"),
    ("Seepage or wet patches at the toe or bench face", "Yellow"),
    ("Cracks widening or lengthening since the last shift", "Orange"),
    ("Bulging or heave at the toe", "Orange"),
    ("Water ponding behind a rib, berm or toe", "Orange"),
    ("Scarp formed / step in the crest, rocks sliding off the face", "Red"),
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
INDIRECT = ("T2 Rainfall",)    # a cause, not evidence of movement: the consequence step stops at Orange


def evaluate(triggers: dict, people_below=False):
    """Returns (final level, governing triggers, note, triggers without data)."""
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


# ------------------------------------------------------------------ trigger-specific responses
# (action, responsible, response time, basis) for every trigger and level
TRIGGER_ACTIONS = {
    "T1 FOS": {
        "Green": ("Re-run the FOS after every survey, geometry change or piezometer reading.",
                  "Slope-stability engineer", "Routine", "DGMS para 3.2"),
        "Yellow": ("Do not raise this dump further. Revise the design (flatter decks, wider benches or lower "
                   "height) until FOS is back above the DGMS minimum; read piezometers weekly; weekly prism survey.",
                   "Slope-stability engineer; Mine Manager approves", "Before the next dumping cycle",
                   "DGMS para 3.2 (minimum 1.50 / 1.30); Singareni and East Kalimantan failed from 1.45-1.49"),
        "Orange": ("Stop dumping on this dump. Unload the crest or lower the top deck; drain the dump (toe drains, "
                   "horizontal drain holes); keep people and vehicles away from the toe; daily survey.",
                   "Mine Manager (decision); Assistant Manager (execution)", "Same shift",
                   "DGMS Bharatpur alert 2019 (drainage, drain holes, people away from the toe)"),
        "Red": ("Stop all work in the run-out zone and evacuate it; barricade; unloading or a toe buttress only "
                "from the safe side after geotechnical clearance; report to the Agent and DGMS.",
                "Mine Manager; Agent", "Immediately", "Failed dumps had FOS 0.72-1.07 (WCL, Jayant, East Kalimantan)"),
    },
    "T2 Rainfall": {
        "Green": ("Read the rain gauge daily; keep drains and berm slopes clear before the monsoon.",
                  "Shift overman", "Daily", "DGMS Bharatpur alert 2019 (pre-monsoon inspection)"),
        "Yellow": ("Inspect drains, berms and the toe after the rain; pump out any ponded water; read piezometers "
                   "and re-run T1 with the new pore pressure; no work directly below the dump during heavy rain.",
                   "Shift overman; slope-stability engineer", "Same day",
                   "Bingham Canyon TARP (25 mm in 24 h; no mining below instabilities in heavy rain)"),
        "Orange": ("Suspend dumping and all work below the dump during the rain and for 24 h after it; pump out "
                   "ponded water; daily survey; re-run T1 with piezometer readings.",
                   "Assistant Manager", "Same shift", "Basundhara 2013 and Bharatpur 2019 failed in heavy / monsoon rain"),
        "Red": ("Withdraw people and machines from the toe and run-out zone until the rain stops and an inspection "
                "clears the dump; continuous watch from a safe point.",
                "Mine Manager", "Immediately", "IMD very heavy rain; DGMS: most dumps in the region failed in the rainy season"),
    },
    "T3 Velocity": {
        "Green": ("Prism survey monthly (below 2 mm/day) or weekly (2-5 mm/day).", "Surveyor", "Routine",
                  "NIT Rourkela notes (Normal, Watch)"),
        "Yellow": ("Survey once in 2 days; walk the crest and toe for cracks; look for the cause (rain, new dumping, "
                   "toe cutting).", "Surveyor; slope-stability engineer", "Within 24 h", "NIT Rourkela notes (Caution)"),
        "Orange": ("Daily survey plus crack meters; restrict access below the dump toe; stop dumping on the moving "
                   "deck; bring in radar or GNSS if available.", "Assistant Manager; slope-stability engineer",
                   "Same shift", "NIT Rourkela notes (Alert); Bingham Canyon (pause mining, add radar)"),
        "Red": ("Evacuate the influence zone and stop all work; continuous radar or GNSS from a safe point.",
                "Mine Manager", "Immediately",
                "NIT Rourkela notes (Alarm, Stop work); 9 real collapses were above 60 mm/day 48 h before"),
    },
    "T4 Acceleration": {
        "Green": ("Keep the regular survey interval.", "Surveyor", "Routine", "-"),
        "Yellow": ("Shorten the reading interval (prisms daily, radar continuous); stop blasting and toe cutting "
                   "near the dump; find what is driving the speed-up.", "Slope-stability engineer", "Within 24 h",
                   "Bingham Canyon, Lark Quartzite (pause mining and blasting below the instability)"),
        "Orange": ("Isolate the area below the dump: no people or machines below the toe; prepare the evacuation "
                   "plan and radio alert; continuous monitoring.", "Mine Manager", "Same shift",
                   "Bingham Canyon, Jordan slide (velocity ratio above 1.3: isolate below)"),
        "Red": ("Remove all people and machines from the run-out zone; barricade; watch from a safe point until "
                "movement stops.", "Mine Manager; Agent", "Immediately",
                "Bingham Canyon (failure forecast within 1-3 days: remove personnel)"),
    },
    "T5 Field signs": {
        "Green": ("Inspect crest, benches and toe every shift and record it.", "Shift overman", "Every shift",
                  "Rajmahal and Juna Kunada cracks were first seen by people on the dump"),
        "Yellow": ("Peg, photograph and measure the cracks every shift; seal open cracks so rain cannot enter; add "
                   "prisms at the cracked area; check drains.", "Shift overman; slope-stability engineer",
                   "Same shift", "DGMS Bharatpur alert 2019 (fill open cracks); Rajmahal (cracks for months)"),
        "Orange": ("Stop dumping on that deck; move machines away from the toe and crest; drain ponded water; "
                   "install crack meters; daily survey.", "Assistant Manager", "Same shift",
                   "Juna Kunada 2017 (excavators moved when cracks widened); Bharatpur 2019 (water behind a rib)"),
        "Red": ("The shift in-charge orders everyone out at once by radio, without waiting for approval; stop all "
                "work; barricade; no re-entry until the Manager and geotechnical engineer clear it.",
                "Shift in-charge (orders withdrawal); Mine Manager", "Immediately",
                "Juna Kunada: radios, all 30 escaped. Rajmahal: stop order not enforced, 23 died"),
    },
}
