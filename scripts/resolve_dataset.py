"""Re-solves FOS for every row of data/physics_dataset.csv with the current solver in core/lem.py
(keeps all inputs). Used after the Oct 2026 solver update (bench-scale search, composite surfaces
off by default, verified against Slide2).
Usage: python scripts/resolve_dataset.py START END   -> data/_resolve/part_START.csv
       python scripts/resolve_dataset.py merge"""
import os, sys, glob, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.lem import DumpGeometry, bishop_fos
PATH = os.path.join(ROOT, "data", "physics_dataset.csv")
PARTS = os.path.join(ROOT, "data", "_resolve")

def geom(r):
    top = r.get("top_deck_height_m", np.nan)
    if int(r.n_decks) == 1:
        return DumpGeometry(1, float(r.H_m), float(r.deck_angle_deg), 0.0)
    return DumpGeometry(int(r.n_decks), float(r.deck_height_m), float(r.deck_angle_deg),
                        float(r.berm_width_m), None if pd.isna(top) else float(top))

if sys.argv[1] == "merge":
    new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "*.csv")))])
    ph = pd.read_csv(PATH)
    ph = ph.drop(columns=["FOS"]).merge(new[["id", "FOS"]], on="id", how="left")
    ph = ph[np.isfinite(ph.FOS) & (ph.FOS > 0.05) & (ph.FOS < 6)]
    ph.to_csv(PATH, index=False); print("rows", len(ph))
else:
    a, b = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(PARTS, exist_ok=True)
    ph = pd.read_csv(PATH).iloc[a:b]
    t = time.time(); out = []
    for _, r in ph.iterrows():
        g = geom(r)
        out.append(dict(id=r.id, FOS=bishop_fos(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u).fos))
    pd.DataFrame(out).to_csv(os.path.join(PARTS, f"part_{a:05d}.csv"), index=False)
    print(a, b, f"{time.time()-t:.0f}s", flush=True)
