"""
Uncertainty-driven adaptive sampling (active learning) for the physics dataset.

1. Draw many candidate DGMS-compliant designs around the real coal materials.
2. Ask the surrogate ensemble for its spread (disagreement) on each candidate,
   weighted towards the safety-critical band 1.0 <= FOS <= 1.6.
3. Solve only the most uncertain candidates with the exact Bishop solver and add them
   to data/physics_dataset.csv (block "D_active_learning").

Usage: python scripts/active_learning.py [n_candidates] [n_new]
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.models import DimlessSurrogate, engineer
from core.lem import DumpGeometry, bishop_fos
from core.dgms import check
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from generate_physics_data import _compliant_geom

n_cand = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 400
rng = np.random.default_rng(99)
path = os.path.join(ROOT, "data", "physics_dataset.csv")
ph = pd.read_csv(path)
P = engineer(ph[ph.dgms_compliant])
sur = DimlessSurrogate().fit(P, np.log(P.FOS))

real = pd.read_csv(os.path.join(ROOT, "data", "real_cases.csv")).dropna(subset=["c_kPa", "phi_deg", "gamma_kNm3"])
mats = real[["c_kPa", "phi_deg", "gamma_kNm3"]].drop_duplicates().values
cand = []
for i in range(n_cand):
    c0, p0, g0 = mats[rng.integers(len(mats))] if rng.random() < 0.7 else \
        (rng.uniform(0.5, 90), rng.uniform(6, 42), rng.uniform(14, 28))
    nd, dh, bw, da = _compliant_geom(rng)
    g = DumpGeometry(nd, dh, da, bw)
    cand.append(dict(block="D_active_learning", anchor="active", c_kPa=max(0.1, c0 * rng.lognormal(0, 0.35)),
                     phi_deg=float(np.clip(rng.normal(p0, 0.12 * p0), 3, 48)),
                     gamma_kNm3=float(np.clip(rng.normal(g0, 0.06 * g0), 13, 29)),
                     r_u=(0.0 if rng.random() < 0.5 else rng.uniform(0.05, 0.45)),
                     moisture_pct=rng.uniform(4, 18), **g.cols()))
C = engineer(pd.DataFrame(cand))
mu, sd = sur.predict(C), sur.predict_std(C)
critical = np.exp(-((np.exp(mu) - 1.3) / 0.3) ** 2)       # emphasise FOS near 1.0-1.6
score = sd * (0.5 + critical)
pick = np.argsort(score)[::-1][:n_new]
t0, rows = time.time(), []
for i in pick:
    r = C.iloc[i]
    g = DumpGeometry(int(r.n_decks), r.deck_height_m, r.deck_angle_deg, r.berm_width_m)
    rows.append({**{k: cand[i][k] for k in cand[i]}, "FOS": bishop_fos(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u).fos,
                 "dgms_compliant": check(g.n_decks, g.deck_height, g.deck_angle, g.berm_width)[0]})
new = pd.DataFrame(rows)
new = new[np.isfinite(new.FOS) & (new.FOS > 0.05) & (new.FOS < 6)]
new["id"] = np.arange(len(new)) + ph["id"].max() + 1
out = pd.concat([ph, new[[c for c in ph.columns if c in new]]], ignore_index=True)
out.to_csv(path, index=False)
print(f"added {len(new)} actively chosen cases in {time.time()-t0:.0f}s; "
      f"mean ensemble spread of chosen {sd[pick].mean():.4f} vs all {sd.mean():.4f}")
