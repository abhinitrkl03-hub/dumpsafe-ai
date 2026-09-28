"""Displacement-based TARP + Fukuzono inverse-velocity forecast.

Velocity bands and monitoring frequencies follow the slope-movement table of the
NIT Rourkela scientific study notes supplied for this project. The inverse-velocity
method is Fukuzono (1985): before failure 1/v falls roughly linearly with time, and
the time where the trend meets 1/v = 0 estimates the failure time.
"""
from __future__ import annotations
import numpy as np, pandas as pd

# (upper bound mm/day, level, colour, monitoring method, frequency, response)
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
LEVEL_ORDER = [r[1] for r in VELOCITY_TARP]


def classify_velocity(v):
    for ub, lvl, col, meth, freq, resp in VELOCITY_TARP:
        if v < ub:
            return dict(level=lvl, colour=col, method=meth, frequency=freq, response=resp)
    r = VELOCITY_TARP[-1]
    return dict(level=r[1], colour=r[2], method=r[3], frequency=r[4], response=r[5])


def process_series(df: pd.DataFrame, time_col="timestamp", disp_col="displacement_mm",
                   window_days=1.0, noise_floor=0.2):
    """Velocity = slope of a least-squares line through the displacements in a trailing
    window (robust to survey noise, unlike point-to-point differences).
    Velocities below `noise_floor` (mm/day, about the resolution of total-station
    monitoring over a day) are treated as 'no measurable movement' for 1/v."""
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    t = ((d[time_col] - d[time_col].iloc[0]).dt.total_seconds() / 86400.0).values
    x = d[disp_col].astype(float).values
    v = np.full(len(d), np.nan)
    for i in range(len(d)):
        m = (t >= t[i] - window_days) & (t <= t[i])
        if m.sum() >= 3 and np.ptp(t[m]) > 0:
            v[i] = np.polyfit(t[m], x[m], 1)[0]
    d["t_days"] = t
    d["velocity_mm_day"] = pd.Series(v).bfill().clip(lower=0).values
    measurable = d["velocity_mm_day"] >= noise_floor
    d["inv_velocity"] = np.where(measurable, 1.0 / d["velocity_mm_day"].clip(lower=1e-9), np.nan)
    d["level"] = [classify_velocity(x_)["level"] for x_ in d["velocity_mm_day"]]
    return d


def inverse_velocity_forecast(d: pd.DataFrame, last_n=20):
    tail = d.dropna(subset=["inv_velocity"]).tail(last_n)
    if len(tail) < 5:
        return None
    a, b = np.polyfit(tail["t_days"], tail["inv_velocity"], 1)
    if a >= 0:
        return dict(trend="not accelerating", slope=a, intercept=b, t_fail=None)
    t_fail = -b / a
    r2 = np.corrcoef(tail["t_days"], tail["inv_velocity"])[0, 1] ** 2
    return dict(trend="accelerating", slope=a, intercept=b, t_fail=t_fail, r2=r2,
                days_left=t_fail - d["t_days"].iloc[-1])


def demo_series(days=45, seed=3, accelerate=True):
    """SYNTHETIC demo feed: steady creep, then tertiary creep in which 1/v falls linearly
    (the Fukuzono model) towards a failure at day days+3. Survey noise +/-0.3 mm."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 0.25)
    v0, t_on, t_f, A = 0.8, days * 0.55, days + 3.0, 120.0
    v = np.full(t.size, v0)
    if accelerate:
        m = t > t_on
        v[m] = v0 + A / (t_f - t[m]) - A / (t_f - t_on)
    disp = np.cumsum(v * 0.25) + rng.normal(0, 0.3, t.size)
    rain = np.clip(rng.gamma(0.4, 12, t.size) * (rng.random(t.size) < 0.2), 0, 120)
    ts = pd.Timestamp("2026-07-01") + pd.to_timedelta(t, unit="D")
    return pd.DataFrame({"timestamp": ts, "displacement_mm": disp.round(2),
                         "rainfall_mm": rain.round(1)}), days + 3.0
