"""Movement analysis for the TARP: velocity, acceleration and time to failure.

Input: survey readings of one monitoring point (prism, GNSS, radar pixel, crack meter):
timestamp and cumulative displacement in mm. Readings may be dense (radar, GNSS, every few
minutes or hours) or sparse (daily or weekly total-station surveys, manual entry).

* Velocity (mm/day): dense records - slope of a least-squares line through the last day of
  readings; sparse records - change between successive readings.
* Velocity ratio VR = short / long moving average of velocity, over 3 and 6 readings as in
  the SECL slope-radar analyser of Masood et al. (2025). VR above 1 means the slope is
  speeding up; Bingham Canyon uses VR 1.3-1.5 as a TARP trigger (Bakken et al. 2020).
* Onset of acceleration: the short average crosses above the long one and stays above it
  (Carla et al. 2017), with the velocity above the Normal band (2 mm/day).
* Time to failure: inverse velocity 1/v fitted against time after the onset (Fukuzono 1985),
  for the raw, short and long series; the three dates give a failure window.

Velocity bands and survey frequencies: slope-movement table of the NIT Rourkela
scientific-study notes supplied for this project.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# (upper bound mm/day, class, colour, monitoring method, frequency, response)
VELOCITY_TARP = [
    (2, "Normal", "#2e7d4f", "Conventional total station", "Monthly",
     "Normal condition of slope. No response required."),
    (5, "Watch", "#8aa53a", "Conventional total station", "Weekly",
     "Initial response should start."),
    (10, "Caution", "#d8a31a", "Conventional total station (CTSM)", "Once in 2 days",
     "No failure expected within 48 h; tighten inspections."),
    (50, "Alert", "#d9731f", "CTSM + crack meters / extensometers", "Daily",
     "No failure expected within 48 h; restrict access below the dump toe."),
    (100, "Alarm", "#c0392b", "Slope stability radar", "Continuous",
     "Progressive failure indicated - evacuate the influence zone."),
    (np.inf, "Stop work", "#7b1d1d", "Slope stability radar / other systems", "Continuous",
     "Stop all work in the zone."),
]
SHORT_N, LONG_N = 3, 6          # readings in the short / long moving average (Masood et al. 2025)
NORMAL_V = 2.0                  # mm/day, upper limit of the Normal band


def classify_velocity(v):
    for ub, lvl, col, meth, freq, resp in VELOCITY_TARP:
        if v < ub:
            return dict(level=lvl, colour=col, method=meth, frequency=freq, response=resp)
    r = VELOCITY_TARP[-1]
    return dict(level=r[1], colour=r[2], method=r[3], frequency=r[4], response=r[5])


def velocity_series(df: pd.DataFrame, time_col="timestamp", disp_col="displacement_mm"):
    d = df[[time_col, disp_col]].dropna().copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).drop_duplicates(time_col).reset_index(drop=True)
    t = ((d[time_col] - d[time_col].iloc[0]).dt.total_seconds() / 86400.0).values
    x = d[disp_col].astype(float).values
    d["t_days"] = t
    dense = len(d) >= 10 and np.median(np.diff(t)) <= 0.5
    v = np.full(len(d), np.nan)
    if dense:
        for i in range(len(d)):
            m = (t >= t[i] - 1.0) & (t <= t[i])
            if m.sum() >= 3 and np.ptp(t[m]) > 0:
                v[i] = np.polyfit(t[m], x[m], 1)[0]
    elif len(d) >= 2:
        dt = np.diff(t)
        v[1:] = np.diff(x) / np.where(dt > 0, dt, np.nan)
    d["velocity_mm_day"] = np.clip(v, 0, None)
    d = d.dropna(subset=["velocity_mm_day"]).reset_index(drop=True)
    d["v_short"] = d.velocity_mm_day.rolling(SHORT_N, min_periods=1).mean()
    d["v_long"] = d.velocity_mm_day.rolling(LONG_N, min_periods=1).mean()
    d["VR"] = d.v_short / d.v_long.replace(0, np.nan)
    d.attrs["dense"] = dense
    return d


def onset(d: pd.DataFrame):
    """Time (days) at which the current run of 'short average above long average' began,
    if the slope is still in it and moving faster than the Normal band; else None."""
    if len(d) < LONG_N:
        return None
    above = (d.v_short > d.v_long) & (d.v_short >= NORMAL_V)
    if not above.iloc[-1]:
        return None
    i = len(d) - 1
    while i > 0 and above.iloc[i - 1]:
        i -= 1
    return float(d.t_days.iloc[i])


def failure_window(d: pd.DataFrame, t_on, min_points=3):
    if t_on is None:
        return None
    sub = d[d.t_days >= t_on]
    fits = {}
    for name, col in [("raw", "velocity_mm_day"), ("short", "v_short"), ("long", "v_long")]:
        s = sub[sub[col] > 0]
        if len(s) < min_points or np.ptp(s.t_days) <= 0:
            continue
        a, b = np.polyfit(s.t_days, 1.0 / s[col], 1)
        if a < 0:
            fits[name] = -b / a
    if not fits:
        return None
    now = float(d.t_days.iloc[-1])
    return dict(fits=fits, t_early=min(fits.values()), t_late=max(fits.values()),
                days_early=min(fits.values()) - now, days_late=max(fits.values()) - now)


def analyse(df: pd.DataFrame):
    """Everything the TARP needs from one monitoring point."""
    d = velocity_series(df)
    if len(d) == 0:
        return None
    t_on = onset(d)
    win = failure_window(d, t_on)
    cur = d.iloc[-1]
    return dict(d=d, onset=t_on, window=win, v=float(cur.velocity_mm_day),
                vr=float(cur.VR) if np.isfinite(cur.VR) else np.nan,
                total=float(df["displacement_mm"].astype(float).max() - df["displacement_mm"].astype(float).min()),
                t0=pd.to_datetime(df["timestamp"]).min())
