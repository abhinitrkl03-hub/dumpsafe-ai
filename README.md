# DumpSafe AI: sim-to-real FOS prediction for OB dump slopes

A Streamlit platform that predicts the factor of safety (FOS) of overburden dumps in
Indian opencast mines from a small set of real cases, with uncertainty, Monte Carlo
reliability, a physics-based moisture model and a real-time TARP.

## Scope: coal-mine overburden dumps, designed to DGMS rules

Every real case is an overburden dump of an Indian opencast **coal** mine.

All simulated geometry follows **Coal Mines Regulations 2017, Regulation 106** (spoil banks):
- dump slope set by the angle of repose and not steeper than **37.5°** (steeper only if a scientific study
  recommends it and the Regional Inspector permits it by order);
- a spoil bank higher than 30 m must be benched so that **no bench exceeds 30 m**;
- **overall slope not steeper than 1V:1.5H (33.7°)**;
- toe at least 100 m from mine openings, railways, public roads and buildings (not a slope input).

Berm width is not fixed by the regulation; the code uses the smallest berm that keeps the overall slope
within 1V:1.5H (and at least 10 m). `core/dgms.py` checks every design, and the app flags any geometry
you enter that breaks these limits.

Important difference from Sahoo et al. (2025): their 2,250-case design used **unbenched** single slopes
60-120 m high at 25-40°, which Reg. 106 does not allow. Here their material ranges are kept but the
dumps are benched to Reg. 106. With the same materials, the mean FOS rises from 1.08 (their unbenched
design) to 1.41 (benched), which shows how much the DGMS benching rule adds to stability.

A separate set of 600 non-compliant simulations and the 14-row WCL failed dump (a 75 m unbenched
slope) are **off by default**. Switch on "Also train on non-DGMS geometries" in the sidebar when you
back-analyse failed dumps, because real failures (WCL, Jayant) happened on dumps outside the rules.

## What is new compared with existing work

| | Sahoo et al. 2025 (Sci Rep) | ICSSMT-26 draft (previous work) | This platform |
|---|---|---|---|
| Training labels | 2,250 Slide runs on unbenched slopes up to 40° | 65 SLIDE2 runs expanded to 5,200 with FOS-scaling formulas | 4,000 DGMS-compliant runs of an in-house Bishop solver incl. 400 chosen by active learning (plus 600 optional non-compliant), validated against 8 SECL study values |
| Model inputs | Raw c, φ, γ, H, β, moisture | Raw inputs | Dimensionless groups (Buckingham-Pi / Janbu λ) with a neural-network ensemble |
| DGMS / CMR 2017 | Not checked | Not checked | Every design checked against Reg. 106; non-compliant designs flagged and excluded by default |
| Real field cases | Not used for training | Not used | 34 cited coal-dump cases from 10 sites calibrate the model |
| Analysis software | Ignored | Ignored | Study-aware: separate corrections for the RS2 (SECL), FLAC (Jagannathpur) and FEM (WCL) studies |
| Validation | Random 80/20 split of simulated data | Random split of expanded data | Leave-one-site-out on real cases + transfer test on unseen geometry and water |
| Physics consistency | Not checked | Not checked | Dimensionless physics inputs; automatic check that FOS never falls when c, φ rise or H, r_u fall (violations < 1 %) |
| Uncertainty | None per prediction | None per prediction | Conformal interval per analysis method, widened away from the real data |
| Moisture | Input column, SHAP importance ~0 | Scaling formula | Mechanisms: bulk unit weight, saturation, pore pressure, optional lab-calibrated softening |
| Monte Carlo | One case (PoF 63%) | Independent normals | Lognormal c, correlated c-phi, reliability index, binomial CI on PoF, exact-solver cross-check |
| TARP | None | FOS only, 3 levels | Worst of FOS band, PoF band and measured velocity band, plus inverse-velocity failure forecast |
| Design | None | None | Reliability-based envelope (height x deck angle) and back-analysis of failures |

## How the model copes with very little real data

FOS is not measured in the field; it is *computed* from material strength and geometry. So the
physics can be simulated as often as needed, and the scarce real cases are used only for what
simulation cannot give. Seven techniques are combined:

1. **Physics-generated training data** from a Bishop solver validated against the SECL study values
   (MAE 0.041, MAPE 2.2 %).
2. **Dimensional analysis.** FOS of a homogeneous slope depends only on dimensionless groups
   (c/γH, Janbu's λ = c/(γH tanφ), tanφ, slope angles, bench ratios, r_u). The surrogate learns
   log(FOS/tanφ) from these, so it needs far fewer examples and extrapolates better.
3. **Neural-network ensemble** (5 members) as the surrogate; the spread between members is an
   uncertainty estimate. Physics check: FOS moves the wrong way in under 1 % of sweeps.
4. **Active learning.** 400 extra simulations were placed where the ensemble disagreed most,
   weighted to the safety-critical band FOS 1.0-1.6 (`scripts/active_learning.py`). On an
   independent test set the worst-case error fell from 13.1 % to 9.4 %.
5. **Sim-to-real correction per study.** A small Gaussian process per study (NIT Rourkela SECL study in
   RS2, Jagannathpur in FLAC, WCL in FEM) learns how reported FOS differs from the solver, using real inputs only, and is held constant
   along directions the real data never varies in (e.g. geometry for SECL).
6. **Honest uncertainty.** Conformal intervals per method, widened by the correction's own
   uncertainty and by the ensemble spread.
7. **Honest validation.** Leave-one-site-out on real data, a transfer test on unseen DGMS designs,
   and an independent exact-solver test set, instead of a random split of look-alike rows.

`data/benchmark_cases_rs2.csv` lists 20 DGMS-compliant cases built from real coal materials, with this
code's Bishop FOS. Running them in RS2 with the same settings as the SECL study and filling the empty
column gives (a) independent verification of the solver in the software the SECL study used, and
(b) RS2 results for new geometries (1-4 decks, wet cases), which removes the biggest limitation of
the SECL correction: all 8 SECL dumps share one geometry. Add the finished rows to
`data/real_cases.csv` with study = "NIT Rourkela SECL study (RS2)" and a new site name for each.

## Validation results (default: DGMS-compliant data, 20 real cases)

| Test | Result |
|---|---|
| In-house Bishop (LEM) vs SECL study FOS computed in RS2 | MAE 0.041, MAPE 2.2 % (max 5.0 %), slightly conservative |
| Surrogate vs exact solver, 20 % hold-out | **MAPE 0.73 %, R² 0.999** (gradient boosting on raw inputs: 3.3-4.4 %) |
| Monte Carlo cross-check (80 exact runs) | MAPE 0.75 % |
| Leave-one-site-out, SECL (RS2 study) | physics 3.3 %, **hybrid 0.9 %**, real-data-only GP 0.3 % (all 8 SECL share one geometry) |
| Leave-one-site-out, Jagannathpur (FLAC) | 18 % (only FLAC site, so its offset cannot be learned when held out) |
| Transfer test: SECL materials in 4 new DGMS designs vs exact Bishop | real-data-only GP **18.8 %**, physics surrogate **0.3 %**, hybrid 3.3 % (the hybrid deliberately adds the ~3 % by which the RS2 results of the SECL study exceed Bishop) |

RS2 (finite-element strength reduction) and this code's Bishop agree within about 3 % for the eight
SECL dumps, as expected for homogeneous dumps. The Jagannathpur FLAC results are about 18 % above
Bishop and the WCL FEM results about 31 % below; since RS2 and Bishop agree, these gaps come from each
study's own assumptions (boundary conditions, strength values used, strain softening), not from the
method itself. That is why the model calibrates per study rather than per method.

The model trains at app start-up (about 35 s the first time on Streamlit Cloud, then cached).

## Data and sources (coal mines only)

`data/real_cases.csv`: every row has `source`, `site`, `fos_method`, `status` and `notes` columns.

**Complete (34 rows, 10 sites); the `dgms_compliant` column says which follow Reg. 106**

| Rows | Site | Subsidiary | Method | Source |
|---|---|---|---|---|
| 8 | Gevra, Kusmunda, Dipka, Manikpur, Chhal, Baroud, Shada, Kanchan | SECL | RS2 (FEM-SRM) | NIT Rourkela (July 2025), Safety and feasibility study of filling of fly ash in eight opencast mines of South Eastern Coalfields Limited, Project No. C3/23/MN/049 |
| 12 | Jagannathpur OCP, H = 30/60/90 m x bench 27-30 deg | SECL (Bhatgaon) | FLAC/Slope | Chourasia, Ravi & Khare (2025), Research Square, doi:10.21203/rs.3.rs-8013159/v1 |
| 14 | WCL internal dump, Wardha Valley (failed at 75 m, 43 deg), 10 angles + 4 heights. **Not DGMS compliant** (unbenched), used only when non-DGMS mode is on | WCL | FEM-SRM | Kainthola et al. (2011) Geomaterials 1(1):1-13, doi:10.4236/gm.2011.11001 |

**Partial, waiting for missing values (27 rows)**: Lakhanpur OCP (MCL, add your SLIDE2 FOS); unnamed
3-deck coal dump with 5 software results and 6 water-table levels (Sathish Kumar & Raj Kumar 2022, add
geometry); coal OB + 20 % fly ash dump at 28-36 deg (Pradhan et al. 2014, add strengths); Marki Mangli-I
(4 sections); Singareni Mine A (SCCL); Sonepur Bazari (ECL); Jambad (ECL); Dhanbad dumps (BCCL area);
Amlohri (NCL, 2 profiles).

**Excluded (2 rows)**: Jayant OCP dragline dump (NCL) - failure controlled by a weak submerged interface
layer and seepage, which the model's inputs cannot represent.

To make a partial row usable: open the paper, fill the missing columns (c, phi, gamma, geometry, FOS),
and set `use_for_training` to True. Add your own cases in the same format, either by editing the CSV or
through **Data and sources → Load your own real dataset** in the app. Use one `site` value per physical
dump so validation holds out whole sites.

`data/physics_dataset.csv`: 4,596 simulated rows, column `dgms_compliant` marks each.
- Block A (1,200, compliant): material ranges of Sahoo et al. 2025 (Table 2) and their dump heights
  (60-120 m), benched to Reg. 106.
- Block B (2,396, compliant): Monte Carlo clouds around every real coal material plus 12 literature-range
  anchors (Kumar et al. 2023, coal mine OB: gamma 14-20.7 kN/m³, phi 8-40°, c 0-72 kPa); 1-4 benches of
  10-30 m, decks 24-37.5°, overall slope within 1V:1.5H, r_u 0-0.4. COVs from Kulhawy (1992).
- Block D (400, compliant): chosen by active learning where the surrogate was least certain.
- Block C (600, **non-compliant**, off by default): unbenched high slopes, benches over 30 m, decks over
  37.5°, overall slopes over 33.7°, for failure back-analysis only.

The Mendeley dataset of Sahoo et al. (data.mendeley.com/datasets/459cbkwwdr/1) can be added as extra
simulated rows through the Data page.

Regenerate after adding real materials:
```
python scripts/build_real_cases.py              # only if you edit the script instead of the CSV
python scripts/generate_physics_data.py all     # about 9 minutes
```

## Method in one paragraph (for the paper)

A Bishop simplified limit-equilibrium solver with a two-stage circular-slip search and pore-pressure
ratio generates labelled cases around real material properties. An ensemble of five neural
networks learns log(FOS/tanφ) from dimensionless groups (log c/γH, Janbu's λ = c/(γH tanφ), tanφ,
overall and deck angles, number of benches, berm-to-bench ratio, r_u); 400 of the training cases were
placed by uncertainty-driven active learning. A Gaussian process with
ARD kernel then learns the residual between reported field FOS and the surrogate using only the real
cases, fitted separately for each study (its software and assumptions), using only the raw inputs
that actually vary within that study's cases. Prediction intervals combine split-conformal residuals from leave-one-site-out validation with
the GP predictive standard deviation. Monte Carlo reliability propagates lognormal cohesion and
normal friction and unit weight (optionally correlated through a Gaussian copula) through the hybrid
model, reporting PoF with a binomial confidence interval and the lognormal reliability index.

## Pages

Overview · Data and sources · Model lab · Predict and TARP · Monte Carlo reliability ·
Moisture and rainfall · Design envelope · Back-analysis of failures · Real-time monitoring · Report

Real-time monitoring accepts a CSV (timestamp, displacement_mm, optional rainfall_mm and r_u),
an uploaded file or a published CSV link that refreshes automatically (for example Google Sheets
→ File → Share → Publish to web → CSV). Velocity bands follow the monitoring-frequency table
from the scientific study notes. The inverse-velocity forecast follows Fukuzono (1985).

## Deploy on a server

**Streamlit Community Cloud (free, simplest)**
1. Create a GitHub repository and upload everything in this folder (keep the folder structure).
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then **Deploy**.
4. You get a public link like `https://dumpsafe-ai.streamlit.app`.

**Hugging Face Spaces**: create a Space with the Docker SDK and upload the folder; the included
`Dockerfile` runs the app on port 8501 (set `app_port: 8501` in the Space README header).

**Render / Railway / any VPS**: use the `Dockerfile`, or the `Procfile` start command
`streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`.

To run on your own machine for testing: `pip install -r requirements.txt`, then `streamlit run app.py`.

## Limitations to state honestly

- The FLAC and FEM offsets each come from a single site; add more coal cases analysed with those methods.
- Reported FOS values come from different software and studies; record `fos_method` and `study`.
- The moisture-to-pore-pressure link and the rainfall-to-r_u link are simplified and must be calibrated
  with site piezometer and shear-test data. Softening is off by default.
- TARP thresholds are editable defaults, not regulatory limits. Use the values in your mine's approved
  design or DGMS permission.
- The demo monitoring feed is synthetic and labelled as such in the app.

## References

- Bishop A.W. (1955) The use of the slip circle in the stability analysis of slopes. Géotechnique 5(1):7-17.
- Bishop A.W., Morgenstern N. (1960) Stability coefficients for earth slopes. Géotechnique 10(4):129-150.
- Fukuzono T. (1985) A new method for predicting the failure time of a slope. Proc. 4th Int. Conf. and
  Field Workshop on Landslides, Tokyo, 145-150.
- Kulhawy F.H. (1992) On the evaluation of static soil properties. ASCE GSP 31:95-115.
- Kumar A., Das S.K., Nainegali L., Raviteja K.V., Reddy K.R. (2023) Probabilistic slope stability analysis
  of coal mine waste rock dump. Geotech. Geol. Eng. 41:4707-4724.
- Sahoo A.K., Tripathy D.P., Jayanthu S. (2025) Advanced machine learning techniques for predicting dump
  slope stability in Indian opencast coal mines. Sci Rep 15:40985.
- Vovk V., Gammerman A., Shafer G. (2005) Algorithmic Learning in a Random World. Springer (conformal prediction).
- Plus the case-study sources listed in `data/real_cases.csv`.
