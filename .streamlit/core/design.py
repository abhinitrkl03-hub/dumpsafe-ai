"""Reliability-based design envelope and back-analysis of failed dumps."""
from __future__ import annotations
import numpy as np, pandas as pd
from .lem import DumpGeometry, bishop_fos
from .montecarlo import sample_inputs


def design_envelope(model, c, phi, g, ru, heights, angles, berm=30.0,
                    covs=(0.3, 0.1, 0.05), n_mc=600, seed=0):
    """For each (total height, deck angle): bench the dump per CMR 2017 Reg. 106 the way
    mines do (full 30 m decks from the bottom, remainder on top, berm >= what 1V:1.5H
    requires), then predict mean FOS and PoF."""
    from .dgms import MAX_BENCH_HEIGHT, min_berm, check
    base = sample_inputs(n_mc, c, covs[0], phi, covs[1], g, covs[2], seed=seed)
    rows = []
    for h in heights:
        for a in angles:
            n_decks = max(1, int(np.ceil(h / MAX_BENCH_HEIGHT - 1e-9)))
            dh = min(h, MAX_BENCH_HEIGHT)
            top = h - (n_decks - 1) * dh
            bw = max(berm, min_berm(n_decks, dh, a, H=h)) if n_decks > 1 else 0.0
            geom = DumpGeometry(n_decks, dh, a, bw, None if abs(top - dh) < 1e-6 else top)
            ok, _ = check(n_decks, dh, a, bw, geom.overall_angle, H=h)
            d = base.copy(); d["r_u"] = ru
            for k, v in geom.cols().items():
                d[k] = v
            f = model.predict(d, with_interval=False)["FOS"].values
            rows.append(dict(H_m=h, n_decks=n_decks, berm_m=bw, deck_angle_deg=a,
                             overall_angle=geom.overall_angle, dgms_ok=ok,
                             FOS_mean=f.mean(), PoF=(f < 1).mean()))
    return pd.DataFrame(rows)


def back_analyse(geom: DumpGeometry, gamma: float, ru: float, phis, target=1.0, c_max=200.0):
    """Cohesion needed for FOS = target at each friction angle (Bishop, bisection).
    Uses the coarse circle search for speed (typically within a few % of the fine search)."""
    out = []
    for phi in phis:
        f0 = bishop_fos(geom, 0.01, phi, gamma, ru, fine=False).fos
        if f0 >= target:
            out.append(dict(phi_deg=phi, c_kPa=0.0)); continue
        lo, hi = 0.0, c_max
        if bishop_fos(geom, hi, phi, gamma, ru, fine=False).fos < target:
            out.append(dict(phi_deg=phi, c_kPa=np.nan)); continue
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if bishop_fos(geom, mid, phi, gamma, ru, fine=False).fos < target: lo = mid
            else: hi = mid
        out.append(dict(phi_deg=phi, c_kPa=0.5 * (lo + hi)))
    return pd.DataFrame(out)
