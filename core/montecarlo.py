"""Monte Carlo reliability analysis driven by the fast sim-to-real model,
with an optional exact-Bishop cross-check on a sub-sample."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy import stats
from .lem import DumpGeometry, bishop_fos
from .moisture import moisture_state


def sample_inputs(n, c_mean, c_cov, phi_mean, phi_cov, g_mean, g_cov, rho_c_phi=0.0,
                  w_mean=None, w_sd=0.0, ru_range=(0.0, 0.0), seed=0):
    """c ~ lognormal, phi & gamma ~ truncated normal; c-phi correlated through a
    Gaussian copula (negative correlation is commonly reported for c-phi pairs)."""
    rng = np.random.default_rng(seed)
    cov = np.array([[1, rho_c_phi], [rho_c_phi, 1]])
    z = rng.multivariate_normal([0, 0], cov, size=n)
    s_ln = np.sqrt(np.log(1 + c_cov ** 2)); mu_ln = np.log(max(c_mean, 1e-3)) - 0.5 * s_ln ** 2
    c = np.exp(mu_ln + s_ln * z[:, 0])
    phi = np.clip(phi_mean * (1 + phi_cov * z[:, 1]), 1, 60)
    g = np.clip(rng.normal(g_mean, g_cov * g_mean, n), 10, 35)
    w = None if w_mean is None else np.clip(rng.normal(w_mean, w_sd, n), 0, 40)
    ru = rng.uniform(ru_range[0], ru_range[1], n) if ru_range[1] > ru_range[0] else np.full(n, ru_range[0])
    return pd.DataFrame({"c_kPa": c, "phi_deg": phi, "gamma_kNm3": g, "r_u": ru,
                         **({"moisture_pct": w} if w is not None else {})})


def reliability(fos: np.ndarray, threshold=1.0):
    fos = np.asarray(fos)
    mu, sd = fos.mean(), fos.std(ddof=1)
    V = sd / mu
    b_norm = (mu - threshold) / sd
    b_logn = np.log(mu / threshold / np.sqrt(1 + V ** 2)) / np.sqrt(np.log(1 + V ** 2))
    pof = float(np.mean(fos < threshold))
    n = len(fos)
    ci = 1.96 * np.sqrt(max(pof * (1 - pof), 1.0 / n) / n)   # binomial CI
    return dict(mean=mu, sd=sd, cov=V, p05=np.quantile(fos, 0.05), p50=np.median(fos),
                pof=pof, pof_ci=ci, beta_normal=b_norm, beta_lognormal=b_logn,
                pof_from_beta=float(stats.norm.cdf(-b_logn)))


def run_mc(model, geom: DumpGeometry, samples: pd.DataFrame, moisture_cfg=None):
    df = samples.copy()
    if moisture_cfg is not None and "moisture_pct" in df:
        c, phi, g, ru, S = moisture_state(df.c_kPa, df.phi_deg, df.gamma_kNm3,
                                          df.moisture_pct, **moisture_cfg,
                                          ru_external=df.r_u.values)
        df["c_kPa"], df["phi_deg"], df["gamma_kNm3"], df["r_u"], df["S_r"] = c, phi, g, ru, S
    for k, v in geom.cols().items():
        df[k] = v
    pred = model.predict(df, with_interval=False)
    df["FOS"] = pred["FOS"].values
    df["FOS_physics"] = pred["FOS_physics"].values
    return df


def exact_check(geom: DumpGeometry, df: pd.DataFrame, n=80):
    """Exact Bishop on a random sub-sample to verify the physics part of the surrogate
    (the real-data correction is deliberately excluded from this comparison)."""
    sub = df.sample(min(n, len(df)), random_state=1)   # same rows as the app's sensitivity check
    exact = [bishop_fos(geom, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u).fos
             for r in sub.itertuples()]
    return pd.DataFrame({"surrogate": sub.FOS_physics.values, "exact_bishop": exact})


def rank_sensitivity(df: pd.DataFrame, cols):
    out = {}
    for c in cols:
        if c in df and df[c].std() > 0:
            out[c] = stats.spearmanr(df[c], df["FOS"]).statistic
    return pd.Series(out).sort_values(key=np.abs)
