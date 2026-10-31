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


# ---------------------------------------------------------------- acceleration triggers
# Carla et al. (2017), Landslides 14: 517-534. The velocity record is smoothed with a
# short-term and a long-term moving average (SMA, LMA). When the SMA rises above the
# LMA and stays there, the slope has entered accelerating (tertiary) creep: this is
# the "onset of acceleration" (OOA). Inverse-velocity fits are made only on the data
# after the OOA, and fitting the raw, SMA and LMA series gives a failure WINDOW
# rather than a single date.
# Defaults (SMA/LMA ratio 1.3 held for 2 days, LMA 5 days) were chosen on 40 synthetic
# records (20 steady, 20 accelerating): no false onset, detection about 4 days after
# the true onset. They must be re-calibrated on the mine's own prism / radar records.

def _trailing_mean(t, y, width):
    out = np.full(len(t), np.nan)
    for i in range(len(t)):
        m = (t >= t[i] - width) & (t <= t[i]) & np.isfinite(y)
        if m.any():
            out[i] = y[m].mean()
    return out


def add_moving_averages(d: pd.DataFrame, short_days=1.0, long_days=5.0):
    d = d.copy()
    t, v = d["t_days"].values, d["velocity_mm_day"].values
    d["v_sma"] = _trailing_mean(t, v, short_days)
    d["v_lma"] = _trailing_mean(t, v, long_days)
    return d


def onset_of_acceleration(d: pd.DataFrame, ratio=1.3, persist_days=2.0, noise_floor=0.2,
                          long_days=5.0):
    """Time (days) of the latest SMA/LMA crossover after which the SMA stays above the
    LMA up to the last reading, with SMA/LMA >= `ratio` for at least `persist_days`
    and a rising SMA. Returns None when the slope is not accelerating."""
    if "v_sma" not in d:
        d = add_moving_averages(d)
    t = d["t_days"].values
    s, l = d["v_sma"].values, d["v_lma"].values
    # the LMA is only meaningful once a full long window of readings exists
    above = (s >= l) & (s >= noise_floor) & (t >= t[0] + long_days)
    if not above[-1]:
        return None
    i = len(above) - 1
    while i > 0 and above[i - 1]:
        i -= 1                                   # start of the current run above the LMA
    run = slice(i, len(t))
    strong = s[run] >= ratio * l[run]
    if not strong.any():
        return None
    first_strong = t[run][np.argmax(strong)]
    if t[-1] - first_strong < persist_days:
        return None
    if np.ptp(t[run]) <= 0 or np.polyfit(t[run], s[run], 1)[0] <= 0:
        return None
    return float(t[i])


def iv_window(d: pd.DataFrame, t_onset, noise_floor=0.2, min_points=5, fit_days=5.0,
              min_r2=0.5, min_span_days=2.0):
    """Inverse-velocity failure forecast from the raw, SMA and LMA velocities, each
    fitted only after the onset of acceleration and over the most recent `fit_days`
    (1/v is curved just after the onset, so the latest readings are the most linear).
    Returns the individual forecasts and the window (earliest, latest) in days since
    the start of the record."""
    if t_onset is None:
        return None
    if "v_sma" not in d:
        d = add_moving_averages(d)
    now = float(d["t_days"].iloc[-1])
    m = d["t_days"] >= max(t_onset, now - fit_days)
    fits = {}
    for name, col in [("raw", "velocity_mm_day"), ("SMA", "v_sma"), ("LMA", "v_lma")]:
        sub = d.loc[m & (d[col] >= noise_floor), ["t_days", col]]
        if len(sub) < min_points or np.ptp(sub["t_days"]) < min_span_days:
            continue
        a, b = np.polyfit(sub["t_days"], 1.0 / sub[col], 1)
        r2 = np.corrcoef(sub["t_days"], 1.0 / sub[col])[0, 1] ** 2
        if a < 0 and r2 >= min_r2:
            fits[name] = dict(t_fail=-b / a, slope=a, intercept=b, r2=r2)
    if not fits:
        return None
    tf = [f["t_fail"] for f in fits.values()]
    return dict(fits=fits, t_early=min(tf), t_late=max(tf),
                days_early=min(tf) - now, days_late=max(tf) - now, t_onset=t_onset)


def demo_storm(days=40, seed=5):
    """SYNTHETIC demo feed: a dump creeping steadily (no acceleration) that receives a
    monsoon storm of about 150 mm/day for two days from day 20. Movement stays in the
    Normal band, so only the rainfall and live-FOS triggers react. Includes a
    piezometer column (pressure head in m) that rises with the storm and drains."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 0.25)
    rain = np.clip(rng.gamma(0.4, 8, t.size) * (rng.random(t.size) < 0.15), 0, 60)
    storm = (t >= 20) & (t < 22)
    rain[storm] = rng.uniform(30, 45, storm.sum())          # ~150 mm per day
    v = 0.6 + 0.6 * np.exp(-np.clip(t - 21, 0, None) / 4) * (t >= 20)   # slight speed-up
    disp = np.cumsum(v * 0.25) + rng.normal(0, 0.3, t.size)
    ari = np.zeros(t.size)
    for i in range(1, t.size):
        ari[i] = ari[i - 1] * 0.9 ** 0.25 + rain[i]
    head = 0.5 + 0.022 * ari + rng.normal(0, 0.15, t.size)  # m of water above the tip
    ts = pd.Timestamp("2026-08-01") + pd.to_timedelta(t, unit="D")
    return pd.DataFrame({"timestamp": ts, "displacement_mm": disp.round(2),
                         "rainfall_mm": rain.round(1), "piezo_head_m": head.clip(0).round(2)}), None


def calibrate_ru(ari, ru_obs):
    """Least-squares r_u = k * ARI (through the origin) from paired antecedent rainfall
    and piezometer r_u. Returns (k, R^2). This is the learned water link (AI role B)
    in its simplest form; replace with a regression on more inputs as data grow."""
    a, r = np.asarray(ari, float), np.asarray(ru_obs, float)
    m = np.isfinite(a) & np.isfinite(r)
    if m.sum() < 5 or (a[m] ** 2).sum() == 0:
        return None, None
    k = (a[m] * r[m]).sum() / (a[m] ** 2).sum()
    ss = ((r[m] - r[m].mean()) ** 2).sum()
    r2 = 1 - ((r[m] - k * a[m]) ** 2).sum() / ss if ss > 0 else np.nan
    return float(k), float(r2)
