"""
Generates data/physics_dataset.csv with the in-house Bishop solver (core/lem.py).

All geometry follows DGMS / CMR 2017 Regulation 106 unless tagged otherwise:
benches <= 30 m, deck angle <= 37.5 deg, overall slope <= 1V:1.5H (33.7 deg).

Block A  "paper materials": material ranges of Sahoo et al. (2025) Table 2 and their dump
                            heights (60-120 m), but BENCHED to Reg. 106 (the paper used
                            unbenched single slopes up to 40 deg, which CMR 2017 does not allow).
Block B  "real-anchored"  : Monte-Carlo clouds around every REAL coal-OB material in
                            data/real_cases.csv (c, phi, gamma all known) plus
                            literature-range anchors (Kumar et al. 2023: gamma 14-20.7
                            kN/m3, phi 8-40 deg, c 0-72 kPa), 1-4 benches of 10-30 m, r_u 0-0.4.
                            COVs follow Kulhawy (1992). DGMS-compliant.
Block C  "non-compliant"  : 600 geometries that break Reg. 106 (unbenched high slopes, >30 m
                            benches, >37.5 deg decks, overall >33.7 deg), tagged
                            dgms_compliant=False. Used only if the user switches them on, to
                            learn the failure zone (the WCL and Jayant failures were such dumps).

Usage
  python scripts/generate_physics_data.py plan  [n_A] [n_B]   # sample the design
  python scripts/generate_physics_data.py run   start end     # solve a slice (resumable)
  python scripts/generate_physics_data.py merge                # write physics_dataset.csv
  python scripts/generate_physics_data.py all                  # everything in one go
"""
import sys, os, glob, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.lem import DumpGeometry, bishop_fos
from core.dgms import check, min_berm, MAX_DECK_ANGLE, MAX_BENCH_HEIGHT, MAX_OVERALL

PLAN = os.path.join(ROOT, "data", "_plan.csv")
PARTS = os.path.join(ROOT, "data", "_parts")


def _compliant_geom(rng, H=None):
    """Random DGMS-compliant geometry (CMR 2017 Reg. 106)."""
    if H is None:
        nd = int(rng.integers(1, 5))
        dh = rng.uniform(10, MAX_BENCH_HEIGHT)
    else:
        nd = max(1, int(np.ceil(H / MAX_BENCH_HEIGHT)))
        dh = H / nd
    da = rng.uniform(24, MAX_DECK_ANGLE)
    if nd == 1:                                   # single bench: overall = deck angle
        da = min(da, MAX_OVERALL)
        return nd, dh, 0.0, da
    bmin = max(10.0, min_berm(nd, dh, da))
    return nd, dh, rng.uniform(bmin, bmin + 30.0), da


def _noncompliant_geom(rng):
    """Geometry breaking at least one Reg. 106 limit (failure-zone learning)."""
    kind = rng.integers(0, 3)
    if kind == 0:                                 # unbenched high slope (e.g. WCL, Jayant)
        return 1, rng.uniform(35, 100), 0.0, rng.uniform(22, 45)
    if kind == 1:                                 # benches too high
        nd = int(rng.integers(2, 4))
        dh = rng.uniform(31, 45); da = rng.uniform(26, 40)
        return nd, dh, rng.uniform(5, 30), da
    nd = int(rng.integers(2, 5))                  # steep decks / narrow berms
    dh = rng.uniform(15, 30); da = rng.uniform(37.6, 45)
    return nd, dh, rng.uniform(3, 12), da


def plan(n_A=1200, n_B=2400, n_C=600, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    heights = (60, 80, 100, 120)
    for i in range(n_A):                          # Sahoo et al. materials, DGMS-benched
        nd, dh, bw, da = _compliant_geom(rng, H=heights[i % 4])
        rows.append(dict(block="A_paper_materials_DGMS", anchor="Sahoo2025_T2",
                         c_kPa=rng.uniform(6.5, 45.7), phi_deg=rng.uniform(13.1, 37.0),
                         gamma_kNm3=rng.uniform(15.2, 22.6), n_decks=nd, deck_height_m=dh,
                         berm_width_m=bw, deck_angle_deg=da, r_u=0.0,
                         moisture_pct=rng.uniform(5, 10)))
    real = pd.read_csv(os.path.join(ROOT, "data", "real_cases.csv"))
    anch = real.dropna(subset=["c_kPa", "phi_deg", "gamma_kNm3"])
    anch = anch[anch.material == "coal OB"].drop_duplicates(subset=["c_kPa", "phi_deg", "gamma_kNm3"])
    anchors = [(r.mine, r.c_kPa, r.phi_deg, r.gamma_kNm3) for r in anch.itertuples()]
    for k in range(12):
        anchors.append((f"Lit_range_{k+1}", rng.uniform(0.5, 72), rng.uniform(8, 40), rng.uniform(14, 20.7)))
    def material(i):
        name, c0, p0, g0 = anchors[i % len(anchors)]
        return name, max(0.1, c0 * rng.lognormal(0, 0.30)), \
            float(np.clip(rng.normal(p0, 0.08 * p0), 3, 48)), float(np.clip(rng.normal(g0, 0.05 * g0), 13, 29))
    for i in range(n_B + n_C):
        name, c, phi, g = material(i)
        comp = i < n_B
        nd, dh, bw, da = _compliant_geom(rng) if comp else _noncompliant_geom(rng)
        rows.append(dict(block="B_real_anchored_DGMS" if comp else "C_non_compliant",
                         anchor=name, c_kPa=c, phi_deg=phi, gamma_kNm3=g, n_decks=nd,
                         deck_height_m=dh, berm_width_m=bw, deck_angle_deg=da,
                         r_u=(0.0 if rng.random() < 0.65 else rng.uniform(0.05, 0.40)),
                         moisture_pct=rng.uniform(4, 18)))
    df = pd.DataFrame(rows)
    df["dgms_compliant"] = [check(int(r.n_decks), r.deck_height_m, r.deck_angle_deg, r.berm_width_m)[0]
                            for r in df.itertuples()]
    df.to_csv(PLAN, index_label="id")
    print("planned", len(df), "cases,", len(anchors), "anchors; DGMS-compliant:",
          int(df.dgms_compliant.sum()))


def run(start, end):
    os.makedirs(PARTS, exist_ok=True)
    P = pd.read_csv(PLAN, index_col="id").iloc[start:end]
    out, t0 = [], time.time()
    for i, r in P.iterrows():
        g = DumpGeometry(int(r.n_decks), r.deck_height_m, r.deck_angle_deg, r.berm_width_m)
        res = bishop_fos(g, r.c_kPa, r.phi_deg, r.gamma_kNm3, r.r_u)
        out.append({**r.to_dict(), "id": i, "H_m": g.height,
                    "beta_overall_deg": g.overall_angle, "FOS": res.fos})
    pd.DataFrame(out).to_csv(os.path.join(PARTS, f"part_{start:05d}.csv"), index=False)
    print(f"solved {start}-{end} in {time.time()-t0:.1f}s")


def merge():
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "*.csv")))])
    df = df.sort_values("id")
    df = df[np.isfinite(df.FOS) & (df.FOS > 0.05) & (df.FOS < 6)]
    df.to_csv(os.path.join(ROOT, "data", "physics_dataset.csv"), index=False)
    print("physics_dataset.csv:", len(df), "rows")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "plan":
        plan(*(int(x) for x in sys.argv[2:4]))
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "merge":
        merge()
    else:
        plan(); n = len(pd.read_csv(PLAN))
        for s in range(0, n, 500):
            run(s, min(n, s + 500))
        merge()
