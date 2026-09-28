"""Displacement-based TARP (from the slope-movement / monitoring-frequency table
used in the scientific study notes) + Fukuzono inverse-velocity forecast."""
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
                   smooth_days=1.0):
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    t_days = (d[time_col] - d[time_col].iloc[0]).dt.total_seconds() / 86400.0
    d["t_days"] = t_days
    v = np.gradient(d[disp_col].values, t_days.values)
    w = max(1, int(round(smooth_days / max(np.median(np.diff(t_days)), 1e-6))))
    d["velocity_mm_day"] = pd.Series(v).rolling(w, min_periods=1, center=False).mean().clip(lower=0)
    d["inv_velocity"] = 1.0 / d["velocity_mm_day"].replace(0, np.nan)
    d["level"] = [classify_velocity(x)["level"] for x in d["velocity_mm_day"]]
    return d


def inverse_velocity_forecast(d: pd.DataFrame, last_n=20):
    """Fukuzono (1985): 1/v decreases ~linearly towards zero before failure.
    Linear fit on the last points; failure time where the line meets 1/v = 0."""
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
    """SYNTHETIC demo feed so the page works before real prism/radar data is loaded.
    Steady creep, then tertiary creep with 1/v falling linearly (Fukuzono-type)."""
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
    return pd.DataFrame({"timestamp": ts, "displacement_mm": np.maximum.accumulate(disp).round(2),
                         "rainfall_mm": rain.round(1)})
