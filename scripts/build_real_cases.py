"""Builds data/real_cases.csv - COAL-MINE dumps only; every row carries its source.
status: complete  -> all model inputs + FOS known (used for calibration)
        partial   -> placeholder; open the cited paper, fill blanks, set use_for_training=True
        excluded  -> complete-ish but controlled by a mechanism outside the model (e.g. weak
                     foundation interface), kept for reference only."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
nan = np.nan
rows = []
COLS = dict(mine=None, subsidiary=None, region=None, material="coal OB", c_kPa=nan, phi_deg=nan,
            gamma_kNm3=nan, n_decks=nan, deck_height_m=nan, berm_width_m=nan, deck_angle_deg=nan,
            H_m=nan, beta_overall_deg=nan, moisture_pct=nan, r_u=nan, FOS=nan, fos_method=None,
            status="partial", use_for_training=False, source=None, notes="")
def add(**k):
    k.pop("notes_geom", None)
    r = dict(COLS); r.update(k); rows.append(r)

# ---------------------------------------------------------------- SECL (8)
SECL = ("NIT Rourkela (2025) Safety and feasibility study of filling of fly ash in eight opencast mines "
        "of South Eastern Coalfields Limited. Project No. C3/23/MN/049, Department of Mining "
        "Engineering, National Institute of Technology Rourkela, July 2025")
for mine, g, c, phi, fos in [
    ("Gevra OCP", 18.63, 44, 30, 1.940), ("Kusmunda OCP", 17.65, 48, 24, 1.616),
    ("Dipka OCP", 17.65, 43, 26, 1.701), ("Manikpur OCP", 20.59, 42, 28, 1.766),
    ("Chhal OCP", 21.57, 45, 30.5, 1.926), ("Baroud OCP", 19.60, 50, 31, 2.034),
    ("Shada OCM", 15.69, 43, 28.1, 1.888), ("Kanchan OCP", 18.63, 46, 27.5, 1.799)]:
    add(mine=mine, subsidiary="SECL", region="Chhattisgarh", c_kPa=c, phi_deg=phi, gamma_kNm3=g,
        n_decks=3, deck_height_m=30, berm_width_m=30, deck_angle_deg=32, H_m=90,
        beta_overall_deg=24, r_u=0.0, FOS=fos, fos_method="RS2 (FEM shear strength reduction)",
        status="complete", use_for_training=True, source=SECL,
        notes="FOS computed in RS2 (Rocscience); r_u=0 assumed (dry analysis) - confirm from report")

# ---------------------------------------------------------------- Jagannathpur OCP, SECL (12)
JAG = ("Chourasia K., Ravi S., Khare R.N. (2025) Slope stability analysis for optimisation of OB dump "
       "in opencast coal mines - case study on Jagannathpur OCP, Chhattisgarh. Research Square "
       "preprint, doi:10.21203/rs.3.rs-8013159/v1 (Tables 2-6)")
jag = {(30, 27): 1.38, (60, 27): 1.17, (90, 27): 1.05, (30, 28): 1.40, (60, 28): 1.21, (90, 28): 1.03,
       (30, 29): 1.37, (60, 29): 1.19, (90, 29): 0.97, (30, 30): 1.29, (60, 30): 1.12, (90, 30): 0.94}
for (H, a), fos in jag.items():
    nd = H // 30
    add(mine=f"Jagannathpur OCP (H {H} m, bench {a} deg)", subsidiary="SECL (Bhatgaon Area)",
        region="Chhattisgarh", c_kPa=81.35, phi_deg=7.77, gamma_kNm3=27.0, n_decks=nd,
        deck_height_m=30, berm_width_m=(30 if nd > 1 else 0), deck_angle_deg=a, H_m=H, r_u=0.0,
        FOS=fos, fos_method="FLAC/Slope (FDM strength reduction)", status="complete",
        use_for_training=True, source=JAG,
        notes="Triaxial mean c, phi; gamma 27 as used by authors; 30 m benches; berm 30 m assumed "
              "(CMR 2017 minimum quoted by authors); phreatic surface at base (external dump)")

# ---------------------------------------------------------------- WCL failed dump (14)
KA = ("Kainthola A., Verma D., Gupte S.S., Singh T.N. (2011) A coal mine dump stability analysis - "
      "a case study. Geomaterials 1(1):1-13, doi:10.4236/gm.2011.11001 (Tables 1-3)")
for a, fos in [(43, .80), (41, .82), (39, .87), (37, .92), (35, .94), (33, 1.00), (31, 1.07),
               (29, 1.13), (27, 1.20), (25, 1.30)]:
    add(mine=f"WCL internal dump, Wardha Valley (75 m, {a} deg)", subsidiary="WCL", region="Maharashtra",
        c_kPa=88.6, phi_deg=24.6, gamma_kNm3=24.4, n_decks=1, deck_height_m=75, berm_width_m=0,
        deck_angle_deg=a, H_m=75, r_u=0.0, FOS=fos, fos_method="FEM shear-strength reduction",
        status="complete", use_for_training=True, source=KA,
        notes="Source analyses the dump as a single 75 m overall slope (internal dump). Mean peak strength of "
              "6 samples at 30% saturation; residual c 36.5 kPa, phi 21.5 deg. 43 deg row = actual failure (SRF 0.8)")
for H, fos in [(80, 1.26), (85, 1.22), (90, 1.22), (95, 1.20)]:
    add(mine=f"WCL internal dump, Wardha Valley ({H} m, 25 deg)", subsidiary="WCL", region="Maharashtra",
        c_kPa=88.6, phi_deg=24.6, gamma_kNm3=24.4, n_decks=1, deck_height_m=H, berm_width_m=0,
        deck_angle_deg=25, H_m=H, r_u=0.0, FOS=fos, fos_method="FEM shear-strength reduction",
        status="complete", use_for_training=True, source=KA,
        notes="Height sensitivity at 25 deg; source analyses a single overall slope")

# ---------------------------------------------------------------- partial / excluded coal cases
add(mine="Lakhanpur OCP", subsidiary="MCL", region="Odisha", c_kPa=90.7, phi_deg=25.17,
    gamma_kNm3=21.5, n_decks=2, deck_height_m=30, berm_width_m=25, deck_angle_deg=37, H_m=60, r_u=0.0,
    fos_method="SLIDE2 (Bishop/GLE)",
    source="Abhishek, Shamshad A., Jayanthu S. - AI enabled TARP for stability of dump slopes (ICSSMT-26 draft)",
    notes="Add the FOS from your own SLIDE2 run for this section")
JY = ("Sharma S., Roy I. (2015) Slope failure of waste rock dump at Jayant opencast mine, India. "
      "Int. J. Appl. Eng. Res. 10(13):33006-33012; strengths also in Sharma, Sengupta & Roy (2015) ARPN J. Earth Sci. 4(2)")
add(mine="Jayant OCP dragline dump (pre-failure 2008)", subsidiary="NCL", region="Madhya Pradesh",
    c_kPa=75, phi_deg=25, gamma_kNm3=20.0, n_decks=1, deck_height_m=85, H_m=85, FOS=0.94,
    fos_method="LEM (Fellenius/Bishop, with water table and interface)", status="excluded",
    notes_geom="dragline dump - a single high spoil heap as analysed in the source",
    source=JY, notes="Failure controlled by weak submerged interface (c 40 kPa, phi 21 deg) and seepage - "
                     "mechanisms not in model features; gamma ~20 kN/m3 approximate; overall angle to confirm")
add(mine="Jayant OCP dragline dump (recommended profile)", subsidiary="NCL", region="Madhya Pradesh",
    c_kPa=75, phi_deg=25, gamma_kNm3=20.0, n_decks=1, deck_height_m=85, deck_angle_deg=37, H_m=85,
    beta_overall_deg=37, FOS=1.10, fos_method="LEM (Fellenius/Bishop)", status="excluded", source=JY,
    notes="Recommended 37 deg overall for 85 m; includes interface/seepage conditions")
OU = ("Sathish Kumar M., Raj Kumar M. (2022) Evaluation of dump slope stability using Slide, Geoslope and "
      "Phase2 software. J. Machine and Computing 2(1):33-41, doi:10.53759/7669/jmc202202005")
for m, fos in [("Slide Bishop", 1.32), ("Slide Spencer", 1.30), ("Geoslope Bishop", 1.36),
               ("Geoslope Spencer", 1.35), ("Phase2 FEM", 1.28)]:
    add(mine=f"Indian opencast coal mine 3-deck dump ({m})", subsidiary="not named", region="India",
        c_kPa=82.0, phi_deg=23.0, gamma_kNm3=18.9, n_decks=3, r_u=0.0, FOS=fos, fos_method=m, source=OU,
        notes="c, phi = mean of 3 deck samples (84/82/80 kPa, 24/24/21 deg); gamma from mean MDD 1.93 g/cc; "
              "deck heights/angles not reported - ask authors or read figures")
for hw, fos in [(50, 1.07), (45, 1.13), (40, 1.19), (35, 1.24), (30, 1.27), (25, 1.28)]:
    add(mine=f"Indian opencast coal mine 3-deck dump (piezometric line {hw} m)", subsidiary="not named",
        region="India", c_kPa=82.0, phi_deg=23.0, gamma_kNm3=18.9, n_decks=3, FOS=fos,
        fos_method="Phase2 FEM-SRM", source=OU,
        notes=f"Water table {hw} m above base - convert to r_u once dump height is known")
PR = ("Pradhan S.P., Vishal V., Singh T.N., Singh V.K. (2014) Optimisation of dump slope geometry vis-a-vis "
      "flyash utilisation using numerical simulation. Am. J. Mining Metall. 2(1):1-7, doi:10.12691/ajmm-2-1-1")
for a, fos in [(28, 4.47), (32, 2.48), (34, 1.67), (36, 1.33)]:
    add(mine=f"Coal mine OB + 20% fly ash dump (60 m, {a} deg)", subsidiary="not named", region="India",
        material="coal OB + 20% fly ash", n_decks=1, deck_height_m=60, deck_angle_deg=a, H_m=60,
        beta_overall_deg=a, FOS=fos, fos_method="FLAC/Slope (FDM-SRM)", source=PR,
        notes="Source models a single unbenched 60 m slope; material strengths in the paper's property table")
MM = "Geete et al. - Stability analysis of OB dump slope, Marki Mangli-I coal mine, IGC 2021 (Springer 2023)"
for ch, state, fos in [(450, "existing", 1.77), (450, "extended", 1.37), (550, "existing", 1.60), (550, "extended", 1.45)]:
    add(mine=f"Marki Mangli-I (ch {ch} m, {state})", subsidiary="captive block (verify)", region="Maharashtra",
        c_kPa=150, phi_deg=21.1, FOS=fos, fos_method="LEM (Bishop/Spencer)", source=MM,
        notes="c, phi at OMC; add unit weight and section geometry from the paper")
add(mine="Singareni 'Mine A' dump No.1 (initial design)", subsidiary="SCCL", region="Telangana",
    c_kPa=0.0, phi_deg=26, gamma_kNm3=20.3, H_m=90, FOS=1.49, fos_method="LEM Bishop simplified",
    source="Poulsen B. et al. (2014) Mine overburden dump failure: a case study. Geotech. Geol. Eng. 32:297-309",
    notes="Design model (Fig. 6); failure was controlled by black-cotton-soil foundation (phi 6 deg) - add slope angle")
add(mine="Sonepur Bazari OCP", subsidiary="ECL", region="West Bengal", FOS=0.92, fos_method="FEM-SRM",
    source="Rajhans, Ekbote & Bhatt (2022) Materials Today: Proceedings 65:735-740",
    notes="Critical external dump; add material and geometry from paper")
add(mine="Jambad OCP", subsidiary="ECL", region="West Bengal", phi_deg=35.0, deck_angle_deg=30, r_u=0.0,
    FOS=1.30, fos_method="FEM",
    source="Assessment of OB dump and highwall slope stability for Jambad OCP using in-situ and laboratory testing",
    notes="phi = mean of 34.5/38.8/31.7 deg samples; FOS ~1.3 at 30 deg bench slope")
DH = "Kumar A. et al. (2026) Probabilistic slope stability of variably saturated OB dump slopes, Sci Rep 16:1791"
for state, ru, fos in [("dry", 0.0, 1.106), ("saturated flow", nan, 1.076)]:
    add(mine=f"Dhanbad OB dumps ({state})", subsidiary="BCCL area", region="Jharkhand", c_kPa=2.44,
        phi_deg=34.5, H_m=60, r_u=ru, FOS=fos, fos_method="FELA upper bound",
        source=DH, notes="Total height 60 m; bench layout, slope angle (25-35 deg range) and unit weight "
                         "not stated - check Fig. 5")
for H, a in [(80, 36), (87, 35)]:
    add(mine=f"Amlohri OCP internal dump ({H} m, {a} deg)", subsidiary="NCL", region="Madhya Pradesh",
        H_m=H, beta_overall_deg=a, FOS=1.20,
        fos_method="LEM (Fellenius/Bishop) design target",
        source="Stability analysis of overburden internal dump material of Amlohri opencast coal mine, India",
        notes="Total height and overall angle of recommended profile (FOS 1.2, 15 m water table); "
              "bench layout and material strength not stated")

df = pd.DataFrame(rows)
# study = who analysed it with which software; the model learns one correction per study
def _study(r):
    if r.subsidiary == "SECL": return "NIT Rourkela SECL study (RS2)"
    if r.mine.startswith("Jagannathpur"): return "Jagannathpur study (FLAC/Slope)"
    if r.mine.startswith("WCL"): return "Kainthola et al. 2011, WCL (FEM-SRM)"
    return r.source.split("(")[0].strip()[:40] + f" ({r.fos_method})"
df["study"] = [_study(r) for r in df.itertuples()]
# site = physical dump, used for leave-one-site-out validation
df.insert(2, "site", df["mine"].str.replace(r"\s*\(.*\)$", "", regex=True))
from core.dgms import check
import sys, os
def _dgms(r):
    if any(pd.isna([r.n_decks, r.deck_height_m, r.deck_angle_deg])):
        return "unknown"
    ok, v = check(int(r.n_decks), r.deck_height_m, r.deck_angle_deg,
                  0.0 if pd.isna(r.berm_width_m) else r.berm_width_m)
    return "yes" if ok else "no: " + "; ".join(v)
from core.lem import DumpGeometry
for i in df.index:     # fill overall angle from deck geometry where it is known
    r = df.loc[i]
    if pd.isna(r.beta_overall_deg) and not any(pd.isna([r.n_decks, r.deck_height_m, r.deck_angle_deg])):
        g = DumpGeometry(int(r.n_decks), r.deck_height_m, r.deck_angle_deg,
                         0.0 if pd.isna(r.berm_width_m) else r.berm_width_m)
        df.at[i, "beta_overall_deg"] = round(g.overall_angle, 2)
df["dgms_compliant"] = [_dgms(r) for r in df.itertuples()]
# Reg. 106 allows steeper/higher spoil banks when a scientific study recommends it and the
# Regional Inspector permits it by order. No source states such an order for these dumps.
df["dgms_permission"] = "not stated in source"
df.insert(0, "case_id", [f"R{i+1:03d}" for i in range(len(df))])
df.to_csv("data/real_cases.csv", index=False)
print(df.status.value_counts().to_string()); print("training rows:", int(df.use_for_training.sum()))
