"""
Imports the Sahoo et al. (2025) Mendeley dataset (2,250 Slide / Janbu-simplified cases,
data.mendeley.com/datasets/459cbkwwdr/1) in two ways.

1. solver check : every case is re-solved with the in-house Bishop solver on the SAME
                  geometry (a uniform slope of height H at overall angle beta) and compared
                  with the published FOS  -> data/mendeley_solver_check.csv
2. DGMS version : cases with overall angle <= 1V:1.5H (25 and 30 deg) are benched the way
                  mines do under CMR 2017 Reg. 106 (full 30 m decks from the bottom,
                  remainder on top, deck angle <= 37.5 deg, berms sized to keep the
                  published overall angle) and re-solved  -> block A_sahoo_benched_DGMS.
                  All 2,250 unbenched originals are kept as non-DGMS block C2 (off by default).

Usage: python scripts/import_mendeley.py check START END | bench START END | merge
"""
import os, sys, glob, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.lem import DumpGeometry, bishop_fos
from core.dgms import check, MAX_OVERALL

RAW = os.path.join(ROOT, "data", "raw", "sahoo2025_mendeley_master_datasheet.xlsx")
PARTS = os.path.join(ROOT, "data", "_mend")


def load():
    d = pd.read_excel(RAW)
    d.columns = [c.strip() for c in d.columns]
    return d.rename(columns={"Cohesion (kN/m2)": "c_kPa", "Phi (deg)": "phi_deg",
                             "Unit Weight (kN/m3)": "gamma_kNm3", "Overall Bench Height": "H_m",
                             "Overall Slope angle": "beta_overall_deg",
                             "Natural Moisture content": "moisture_pct", "FOS": "FOS_paper"})


def run(mode, a, b):
    os.makedirs(PARTS, exist_ok=True)
    d = load().iloc[a:b]
    rng = np.random.default_rng(1000 + a)
    out, t0 = [], time.time()
    for i, r in d.iterrows():
        if mode == "check":
            g = DumpGeometry.from_overall(r.H_m, r.beta_overall_deg)
        else:
            if r.beta_overall_deg > MAX_OVERALL:
                continue
            da = rng.uniform(max(r.beta_overall_deg + 3, 30.0), 37.5)
            g = DumpGeometry.dgms_benched(r.H_m, r.beta_overall_deg, deck_angle=da)
        f = bishop_fos(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, 0.0).fos
        out.append(dict(src_row=i, c_kPa=r.c_kPa, phi_deg=r.phi_deg, gamma_kNm3=r.gamma_kNm3,
                        moisture_pct=r.moisture_pct, r_u=0.0, FOS_paper=r.FOS_paper,
                        top_deck_height_m=g.top_deck_height, **g.cols(), FOS=f,
                        dgms_compliant=check(g.n_decks, g.deck_height, g.deck_angle, g.berm_width,
                                             g.overall_angle)[0]))
    pd.DataFrame(out).to_csv(os.path.join(PARTS, f"{mode}_{a:05d}.csv"), index=False)
    print(mode, a, b, len(out), f"{time.time()-t0:.0f}s")


def merge():
    chk = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "check_*.csv")))])
    ben = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "bench_*.csv")))])
    chk["diff_pct"] = (chk.FOS - chk.FOS_paper) / chk.FOS_paper * 100
    chk["beta_overall_deg"] = chk["beta_overall_deg"].round(3)
    chk.to_csv(os.path.join(ROOT, "data", "mendeley_solver_check.csv"), index=False)
    ph = pd.read_csv(os.path.join(ROOT, "data", "physics_dataset.csv"))
    ph = ph[~ph.block.isin(["A_paper_materials_DGMS", "A_sahoo_benched_DGMS", "C2_sahoo_unbenched"])]
    ben = ben.assign(block="A_sahoo_benched_DGMS", anchor="Sahoo2025_Mendeley")
    unb = chk.assign(block="C2_sahoo_unbenched", anchor="Sahoo2025_Mendeley", dgms_compliant=False)
    keep = [c for c in ph.columns if c != "id"] + ["top_deck_height_m", "FOS_paper", "src_row"]
    keep = list(dict.fromkeys(keep))
    new = pd.concat([ph, ben, unb], ignore_index=True)
    new = new[[c for c in keep if c in new.columns]]
    new.insert(0, "id", np.arange(len(new)))
    new.to_csv(os.path.join(ROOT, "data", "physics_dataset.csv"), index=False)
    print(new.groupby("block").size())
    print("solver check: mean diff %.2f %%, sd %.2f %%" % (chk.diff_pct.mean(), chk.diff_pct.std()))


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge()
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
