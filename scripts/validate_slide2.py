"""
Checks of the Slide2-procedure search (core/slide2.py):
  V12  updated: author's two Slide2 runs vs the Slide2-procedure search (whole slope and bench by bench)
  V13  Slide2-procedure search vs the training labels (bishop_fos) on 60 random training rows
Run:  python scripts/validate_slide2.py
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.lem import DumpGeometry
from core.slide2 import search, bench_by_bench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "data", "validation")

v12 = pd.read_csv(os.path.join(V, "v12_slide2.csv"))
whole, bench, crit = [], [], []
for r in v12.itertuples():
    g = DumpGeometry(int(r.n_decks), r.deck_height_m, r.deck_angle_deg, r.berm_width_m)
    whole.append(search(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u).fos)
    b = bench_by_bench(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u)
    f = [x.fos for x in b]
    bench.append(min(f)); crit.append(int(np.argmin(f)) + 1)
v12["FOS_slide2_procedure_whole"] = whole
v12["FOS_slide2_procedure_bench"] = bench
v12["critical_bench"] = crit
v12["diff_pct_procedure"] = (v12.FOS_slide2_procedure_bench - v12.FOS_slide2) / v12.FOS_slide2 * 100
v12.to_csv(os.path.join(V, "v12_slide2.csv"), index=False)
print(v12[["FOS_slide2", "FOS_bishop", "FOS_slide2_procedure_whole", "FOS_slide2_procedure_bench",
           "critical_bench", "diff_pct_procedure"]].round(3))

ph = pd.read_csv(os.path.join(ROOT, "data", "physics_dataset.csv")).sample(60, random_state=3)
rows = []
for r in ph.itertuples():
    top = r.top_deck_height_m
    g = DumpGeometry(int(r.n_decks), r.deck_height_m, r.deck_angle_deg, r.berm_width_m,
                     None if (pd.isna(top) or top <= 0) else top)
    f = search(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u).fos
    rows.append(dict(id=r.id, block=r.block, c_kPa=r.c_kPa, phi_deg=r.phi_deg, gamma_kNm3=r.gamma_kNm3,
                     n_decks=r.n_decks, deck_height_m=r.deck_height_m, deck_angle_deg=r.deck_angle_deg,
                     berm_width_m=r.berm_width_m, r_u=r.r_u, FOS_label=r.FOS, FOS_slide2_procedure=f,
                     diff_pct=(f - r.FOS) / r.FOS * 100))
v13 = pd.DataFrame(rows)
v13.to_csv(os.path.join(V, "v13_slide2_procedure_vs_labels.csv"), index=False)
d = v13.diff_pct
print(f"V13: mean {d.mean():+.2f} %, SD {d.std():.2f} %, max |diff| {d.abs().max():.2f} %")
