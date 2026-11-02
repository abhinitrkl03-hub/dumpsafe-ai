"""Back-analysis of past failures to calibrate TARP thresholds.

A failure record is any time series that ends in a known failure time: displacement
(prism, radar, crack meter, GPS) and/or rainfall. From each record we read what the
triggers were showing 7, 3 and 1 day(s) before the failure. Collected over several
failures, these become the mine's own thresholds:

    Yellow = lowest value seen 7 days before a failure
    Orange = lowest value seen 3 days before a failure
    Red    = lowest value seen 1 day before a failure

Taking the lowest value across failures is the conservative choice: every failure in
the set would have reached that level at least that early. With one or two records the
numbers are only indicative; the rule improves as failures (and near-misses) are added.
"""
from __future__ import annotations
import numpy as np, pandas as pd

LEADS = (7.0, 3.0, 1.0)              # days before failure -> Yellow, Orange, Red
RAIN_WINDOWS = (1, 3, 7, 15, 30)     # days


def _t_days(ts, t0):
    return (pd.to_datetime(ts) - t0).dt.total_seconds().values / 86400.0


def velocity_record(df, t_fail, time_col="timestamp", disp_col="displacement_mm"):
    """Velocity (mm/day) from a displacement record. Dense records (median spacing up to
    12 h and at least 10 readings) use a 1-day least-squares window; sparse records
    (e.g. crack widths read once a day or less) use the change between readings,
    held constant over each interval."""
    d = df[[time_col, disp_col]].dropna().copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d = d.sort_values(time_col).reset_index(drop=True)
    tf = pd.to_datetime(t_fail)
    d["t_before"] = -_t_days(d[time_col], tf)            # days before failure (positive)
    t = -d["t_before"].values
    x = d[disp_col].astype(float).values
    dense = len(d) >= 10 and np.median(np.diff(t)) <= 0.5
    v = np.full(len(d), np.nan)
    if dense:
        for i in range(len(d)):
            m = (t >= t[i] - 1.0) & (t <= t[i])
            if m.sum() >= 3 and np.ptp(t[m]) > 0:
                v[i] = np.polyfit(t[m], x[m], 1)[0]
        v = pd.Series(v).bfill().values
    else:
        dt = np.diff(t)
        v[1:] = np.where(dt > 0, np.diff(x) / np.where(dt > 0, dt, 1), np.nan)
    d["velocity_mm_day"] = np.clip(v, 0, None)
    d.attrs["dense"] = dense
    return d


def value_before(d, col, lead, sparse_hold=True):
    """Value of `col` at `lead` days before failure: the latest reading at or before that
    time. For sparse records the velocity of an interval applies to its whole span, so
    the interval that contains the lead time is used."""
    tb = d["t_before"].values
    if not d.attrs.get("dense", True) and sparse_hold and col == "velocity_mm_day":
        for i in range(1, len(d)):
            if tb[i - 1] >= lead >= tb[i]:
                return float(d[col].iloc[i])
        return np.nan
    m = (tb >= lead) & (tb <= lead + 0.5)          # a reading within half a day before the lead time
    if not m.any():
        return np.nan
    val = d.loc[m, col].iloc[-1]
    return float(val) if np.isfinite(val) else np.nan


def movement_summary(d):
    return {f"v_{int(k)}d": value_before(d, "velocity_mm_day", k) for k in LEADS}


def rain_summary(df, t_fail, time_col="timestamp", rain_col="rainfall_mm", decay=0.90):
    r = df[[time_col, rain_col]].dropna().copy()
    r[time_col] = pd.to_datetime(r[time_col])
    r = r.sort_values(time_col)
    tb = -_t_days(r[time_col], pd.to_datetime(t_fail))
    out = {}
    for w in RAIN_WINDOWS:
        out[f"rain_{w}d"] = float(r.loc[(tb >= 0) & (tb < w), rain_col].sum())
    t = -tb
    ari = 0.0
    for i in range(1, len(t)):
        if tb[i] < 0:
            break
        ari = ari * decay ** (t[i] - t[i - 1]) + r[rain_col].iloc[i]
    out["ARI_at_failure"] = float(ari)
    return out


def suggest(cases: pd.DataFrame):
    """cases: one row per failure with columns v_7d, v_3d, v_1d and/or rain_* / ARI.
    Returns the suggested (Yellow, Orange, Red) for velocity, and for rainfall over 3 days
    and the antecedent index (Yellow = 50 % / Orange = 75 % / Red = 100 % of the lowest
    value that preceded a failure)."""
    out = {}
    if {"v_7d", "v_3d", "v_1d"} & set(cases.columns):
        vals = [cases[c].dropna().min() if c in cases and cases[c].notna().any() else np.nan
                for c in ("v_7d", "v_3d", "v_1d")]
        out["velocity"] = dict(values=vals, n=int(cases.get("v_1d", pd.Series(dtype=float)).notna().sum()))
    for key in ("rain_3d", "ARI_at_failure"):
        if key in cases and cases[key].notna().any():
            low = float(cases[key].dropna().min())
            out[key] = dict(values=[0.5 * low, 0.75 * low, low], n=int(cases[key].notna().sum()))
    return out


# ---------------------------------------------------------------- example records
def juna_kunada():
    """Crack width at the 172 mRL bench, Juna Kunada OC (WCL), from the DGMS alert of
    01-12-2017. Only two readings are published (times of day not given; noon assumed)."""
    df = pd.DataFrame({"timestamp": ["2017-11-27 12:00", "2017-11-30 12:00"],
                       "displacement_mm": [1.5, 50.0]})
    return df, "2017-12-01 01:05"
