"""
Pre-computes every exact-solver reference used on the app's 'Methods and validation' page.
The app compares the CURRENT model against these references at start-up, so the numbers
shown are always for the model actually deployed.

  V1  Bishop solver vs SECL study FOS (RS2, NIT Rourkela 2025)           -> v1_secl.csv
  V2  Bishop solver vs 2,250 published Slide/Janbu cases (Sahoo 2025)     -> mendeley_solver_check.csv
  V3  Bishop solver vs closed-form infinite-slope FOS (c = 0)             -> v3_infinite_slope.csv
  V4  250 fresh exact cases for an independent surrogate test            -> independent_test_exact.csv
  V6  300 exact Monte Carlo realisations (Gevra material)                -> v6_mc_exact.csv
  V7  Exact FOS along the moisture path                                  -> v7_moisture_exact.csv
  V8  WCL failed dump: FOS with peak and residual strengths              -> v8_wcl_failure.csv
Run: python scripts/validate.py   (about 2 minutes)
"""
import os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.lem import DumpGeometry, bishop_fos
from core.moisture import moisture_state
from core.montecarlo import sample_inputs
OUT = os.path.join(ROOT, "data", "validation")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# V1 ---------------------------------------------------------------------------------
real = pd.read_csv(os.path.join(ROOT, "data", "real_cases.csv"))
secl = real[real.subsidiary == "SECL"].copy()
g = DumpGeometry(3, 30, 32, 30)
secl["FOS_bishop"] = [bishop_fos(g, r.c_kPa, r.phi_deg, r.gamma_kNm3).fos for r in secl.itertuples()]
secl["diff_pct"] = (secl.FOS_bishop - secl.FOS) / secl.FOS * 100
secl[["mine", "c_kPa", "phi_deg", "gamma_kNm3", "FOS", "FOS_bishop", "diff_pct"]].to_csv(
    os.path.join(OUT, "v1_secl.csv"), index=False)

# V3 ---------------------------------------------------------------------------------
rows = []
for phi in (30, 35):
    for beta in (20, 25, 30):
        f = bishop_fos(DumpGeometry.from_overall(60, beta), 0.01, phi, 18.0).fos
        exact = np.tan(np.radians(phi)) / np.tan(np.radians(beta))
        rows.append(dict(phi_deg=phi, beta_deg=beta, FOS_bishop=f, FOS_closed_form=exact,
                         diff_pct=(f - exact) / exact * 100))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v3_infinite_slope.csv"), index=False)

# V6 ---------------------------------------------------------------------------------
s = sample_inputs(300, 44, 0.30, 30, 0.10, 18.63, 0.05, 0.0, seed=7)
s["FOS_exact"] = [bishop_fos(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u).fos for r in s.itertuples()]
s.to_csv(os.path.join(OUT, "v6_mc_exact.csv"), index=False)

# V7 ---------------------------------------------------------------------------------
rows = []
for w in (5, 10, 15, 16, 17, 18, 19, 19.2):
    c, p, gm, ru, S = moisture_state(44, 30, 17.0, w, Gs=2.6, S_crit=0.8, w_ref=8, kc=0, kphi=0)
    rows.append(dict(w_pct=w, S=float(S), r_u=float(ru), gamma=float(gm), c_kPa=44.0, phi_deg=30.0,
                     FOS_exact=bishop_fos(g, 44, 30, float(gm), float(ru)).fos))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v7_moisture_exact.csv"), index=False)

# V8 ---------------------------------------------------------------------------------
gw = DumpGeometry.from_overall(75, 43)
rows = [dict(strength="peak (c 88.6 kPa, phi 24.6 deg)", FOS_bishop=bishop_fos(gw, 88.6, 24.6, 24.4).fos),
        dict(strength="residual (c 36.5 kPa, phi 21.5 deg)", FOS_bishop=bishop_fos(gw, 36.5, 21.5, 24.4).fos),
        dict(strength="reported by source (FEM-SRM)", FOS_bishop=0.80)]
pd.DataFrame(rows).to_csv(os.path.join(OUT, "v8_wcl_failure.csv"), index=False)
print(f"validation references written in {time.time()-t0:.0f}s")
