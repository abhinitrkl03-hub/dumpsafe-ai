"""'Methods and equations' and 'Validation' pages (kept separate to keep app.py readable)."""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy import stats

from core.models import engineer, metrics, group_of
from core.lem import DumpGeometry
from core.dgms import check as dgms_check


def methods_page():
    st.title("Methods and equations")
    st.markdown("<p class='lede'>Every model, equation and rule used in the app, with its source.</p>",
                unsafe_allow_html=True)

    with st.expander("1. Limit-equilibrium solver: Bishop's simplified method", expanded=True):
        st.markdown("Factor of safety of a circular slip surface divided into vertical slices "
                    "(Bishop 1955), with pore pressure from the pore-pressure ratio (Bishop & Morgenstern 1960):")
        st.latex(r"F=\frac{\sum_i \left[c' b_i + (W_i-u_i b_i)\tan\phi'\right]/m_{\alpha,i}}{\sum_i W_i\sin\alpha_i},"
                 r"\qquad m_{\alpha,i}=\cos\alpha_i\left(1+\frac{\tan\alpha_i\tan\phi'}{F}\right),"
                 r"\qquad u_i=r_u\,\gamma\,h_i")
        st.markdown("- Solved by fixed-point iteration (tolerance 1e-4, at most 40 iterations); "
                    "m_alpha is floored at 0.2, the usual numerical guard.\n"
                    "- Training labels: 140 slices per circle; circles searched (i) on a global grid of centres "
                    "and exit points, (ii) on a separate grid for every bench, and (iii) in two refinements around "
                    "every candidate, keeping the lowest.\n"
                    "- Slip circles drawn in the app follow the Slide2 procedure step by step (core/slide2.py): "
                    "1 model (external boundary from the toe, base y = 0); 2 materials; 3 Bishop simplified with "
                    "50 slices, tolerance 0.005, 75 iterations; 4 circular surfaces by auto refine search (20 "
                    "divisions along the slope, 10 circles per division, 10 iterations, best 50 % of divisions kept), "
                    "grid search (20 x 20 centre grid, radius increment 10) or slope search (5,000 random surfaces); "
                    "5 slope limits for exit and entry; 6 invalid surfaces rejected (fewer than two slope "
                    "intersections, below the base, outside the limits, not converged); 7 the global minimum drawn "
                    "with its centre, radius lines and FOS, optionally with the lowest surfaces colour-coded by FOS. "
                    "Agreement with the author's Slide2 runs: V12; with the training labels: V13.\n"
                    "- Bench-by-bench check (Overview and Predict pages): the Slide2 search is repeated with the slope "
                    "limits moved onto each bench in turn (exit on that bench face or the half bench width in front "
                    "of its toe, entry anywhere above). The most critical bench is the one to prioritise for strengthening, drainage and "
                    "monitoring. Which bench governs depends on bench width and water: wide benches make all benches "
                    "similar, narrow benches or a wet dump make the lower benches critical.\n"
                    "- The foundation is treated as competent: circles may not pass below the dump base "
                    "(composite surfaces off, the Slide2 default). This setting was chosen after comparison "
                    "with Slide2 (V12).\n"
                    "- Geometry: n benches of height h (optionally a lower top bench), face angle beta_deck, bench width b; "
                    "overall angle = atan(H / (H cot beta_deck + (n-1) b)).")

    with st.expander("2. Physics surrogate: neural-network ensemble on dimensionless groups"):
        st.markdown("Dimensional analysis: for a homogeneous c-phi slope F depends only on dimensionless groups "
                    "(Janbu 1954; Bishop & Morgenstern 1960). The inputs are")
        st.latex(r"\mathbf{x}=\Big[\ln\tfrac{c}{\gamma H},\ \ln\tfrac{c}{\gamma H\tan\phi},\ \tan\phi,\ "
                 r"\beta,\ \beta_{deck},\ n_{decks},\ \tfrac{b}{h},\ r_u\Big],\qquad "
                 r"y=\ln\frac{F}{\tan\phi}")
        st.markdown("- Five multilayer perceptrons (128-64-32 ReLU units, Adam, learning rate 3e-3, L2 penalty 1e-4, "
                    "early stopping on a 10 % split with patience 50, standardised inputs), each with a different "
                    "random seed.\n"
                    "- Prediction = mean of the five in log space; the spread between them is an uncertainty measure:")
        st.latex(r"\ln \hat F_{phys}=\tfrac15\sum_{k=1}^{5}\big(\hat y_k+\ln\tan\phi\big),\qquad "
                 r"\sigma_{ens}=\operatorname{sd}_k\big(\hat y_k\big)")
        st.markdown("- Benchmarks trained on the same data: histogram gradient boosting on the raw inputs, with and "
                    "without monotonic constraints (scikit-learn).")

    with st.expander("3. Training data"):
        st.markdown("- **Block A** - the 1,200 published cases of Sahoo et al. (2025) at 25 and 30 deg, benched to "
                    "CMR 2017 Reg. 106 and re-solved.\n"
                    "- **Block B** - 2,396 cases around every real coal material (lognormal c, COV 30 %; normal phi, "
                    "COV 8 %; normal gamma, COV 5 %; Kulhawy 1992) plus 12 anchors from the coal-dump ranges of "
                    "Kumar et al. (2023), in random Reg. 106-compliant geometries, r_u 0-0.4.\n"
                    "- **Block D** - 400 cases chosen by active learning, score:")
        st.latex(r"s(\mathbf{x})=\sigma_{ens}(\mathbf{x})\left[0.5+\exp\!\left(-\left(\tfrac{\hat F-1.3}{0.3}\right)^2\right)\right]")
        st.markdown("- **Blocks C, C2** - 600 designs beyond Reg. 106 and the 2,250 unbenched published cases, used only "
                    "when 'DGMS permission' is switched on.\n"
                    "- **Real cases** - 20 Reg. 106-compliant coal-dump cases (8 SECL in RS2, 12 Jagannathpur in FLAC); "
                    "the 14 WCL cases are added with 'DGMS permission'.")

    with st.expander("4. Sim-to-real correction (per study)"):
        st.markdown("For each study s (same analysts, software and assumptions) with at least 3 cases, the residual "
                    "between reported and physics FOS is modelled as the study's mean offset plus a Gaussian-process "
                    "term that is only trusted inside the range covered by that study's cases:")
        st.latex(r"r_i=\ln F_i^{real}-\ln\hat F_{phys}(\mathbf{z}_i),\qquad "
                 r"\ln \hat F=\ln\hat F_{phys}+\mu_s+w(\mathbf{z})\,f_s(\mathbf{z}),\qquad "
                 r"f_s\sim\mathcal{GP}\big(0,\ \sigma_f^2 e^{-\frac12\sum_j (z_j-z'_j)^2/\ell_j^2}+\sigma_n^2\delta\big)")
        st.latex(r"w(\mathbf{z})=\exp\!\left(-d^2\right),\qquad d=\max_j \frac{\max(z_j^{min}-z_j,\ z_j-z_j^{max},\ 0)}{z_j^{max}-z_j^{min}}")
        st.markdown("- z = the raw inputs (c, phi, gamma, H, beta, beta_deck, r_u), standardised; only inputs that "
                    "vary within the study are used. Kernel hyper-parameters by maximum likelihood.\n"
                    "- A study with fewer than 3 cases, or a query for a study not in the data, gets no correction.")

    with st.expander("5. Prediction interval (conformal)"):
        st.latex(r"\hat F\,e^{\pm h},\qquad h=\sqrt{q^2+(z\,\sigma_{GP})^2+(z\,\sigma_{ens})^2},\qquad "
                 r"q=\text{the }\lceil (n+1)(1-\alpha)\rceil\text{-th smallest } |r_i^{LOSO}|")
        st.markdown("q comes from leave-one-site-out residuals of the chosen study (split-conformal prediction, "
                    "Vovk et al. 2005); z is the normal quantile for coverage 1 - alpha.")

    with st.expander("6. Monte Carlo reliability"):
        st.latex(r"c\sim\text{Lognormal}:\ \sigma_{\ln}=\sqrt{\ln(1+V_c^2)},\ \mu_{\ln}=\ln\mu_c-\tfrac12\sigma_{\ln}^2;"
                 r"\qquad \phi,\gamma\sim\text{Normal};\qquad (z_c,z_\phi)\sim\mathcal N(\mathbf 0,"
                 r"\begin{bmatrix}1&\rho\\\rho&1\end{bmatrix})")
        st.latex(r"P_f=\frac{\#\{F<1\}}{N}\pm1.96\sqrt{\frac{P_f(1-P_f)}{N}},\quad "
                 r"\beta_{N}=\frac{\mu_F-1}{\sigma_F},\quad "
                 r"\beta_{LN}=\frac{\ln\!\big(\mu_F/\sqrt{1+V_F^2}\big)}{\sqrt{\ln(1+V_F^2)}}")
        st.markdown("Sensitivity = Spearman rank correlation between each sampled input and F. "
                    "Typical COVs: unit weight 3-7 %, friction 2-13 %, cohesion 10-50 % (Kulhawy 1992).")

    with st.expander("7. Moisture and pore pressure"):
        st.latex(r"\gamma=\gamma_d(1+w),\quad e=\frac{G_s\gamma_w}{\gamma_d}-1,\quad S=\frac{wG_s}{e},\quad "
                 r"w_{sat}=\frac{e}{G_s}")
        st.latex(r"r_u=\frac{\gamma_w}{\gamma}\ \mathrm{clip}\!\left(\frac{S-S_c}{1-S_c},0,1\right),\quad "
                 r"c(w)=c\,[1-k_c(w-w_{ref})^+],\quad \phi(w)=\phi-k_\phi (w-w_{ref})^+")
        st.markdown("The first three relations are standard phase relations. S_c (default 0.8), k_c and k_phi are "
                    "assumptions to be calibrated with site data; softening is off by default.")

    with st.expander("8. Monitoring and the multi-parameter TARP"):
        st.latex(r"v(t)=\text{slope of the least-squares line through }\{(t_j,x_j):t-\Delta\le t_j\le t\}")
        st.markdown("**Acceleration (T5).** Short and long trailing averages of velocity (default 1 and 5 days). "
                    "When the short average rises above the long one and stays at least 1.3 times higher for 2 days, "
                    "the onset of acceleration (OOA) is declared (Carla et al. 2017). The inverse velocity is then "
                    "fitted only on data after the OOA, over the latest 5 days, separately for the raw, short and "
                    "long series; the three failure times give a forecast window (Fukuzono 1985).")
        st.latex(r"\frac1v=a+bt,\quad t_f=-\frac ab")
        st.markdown("**Water (T2).** Measured r_u from a piezometer, or estimated from the antecedent rainfall "
                    "index; with both, the rain-to-r_u coefficient is fitted by least squares.")
        st.latex(r"r_u=\frac{\gamma_w h_p}{\gamma z},\qquad ARI_i=ARI_{i-1}\,k^{\Delta t}+R_i,\qquad "
                 r"r_u\approx\min(r_{u,max},\ \kappa\,ARI)")
        st.markdown("**Rule.** T1 design FOS, T2 live FOS, T3 24-h rainfall (IMD heavy / very heavy / extremely "
                    "heavy), T4 velocity bands (NIT Rourkela scientific-study notes), T5 acceleration and T6 shift "
                    "inspection each give a level; the worst governs, raised one step when people or haul roads "
                    "are in the run-out zone. Defaults for T5 were chosen on 40 synthetic records (no false onset, "
                    "detection about 4 days after the true onset) and must be calibrated on the mine's own data.")

    with st.expander("9. Rules and acceptance criteria"):
        st.markdown("**CMR 2017, Regulation 106 (spoil banks):** slope not steeper than the angle of repose and "
                    "never more than 37.5 deg; spoil banks over 30 m high benched so no bench exceeds 30 m; overall "
                    "slope not steeper than 1V:1.5H (33.7 deg); toe at least 100 m from mine openings, railways, "
                    "public roads and buildings. Steeper or higher only when a scientific study recommends it and the "
                    "Regional Inspector permits it by order.")
        st.markdown("**Minimum factor of safety (DGMS circular for the scientific study under Reg. 106(2) of "
                    "CMR 2017, para 3.2):** not less than 1.50 for permanent or long-standing slopes and 1.30 "
                    "for all other cases. Para 4.1(iii) asks the study to give trigger points for the TARP.")
        st.dataframe(pd.DataFrame({"Type of dump slope": ["Permanent / long-standing", "Other (working / temporary)"],
                                   "Minimum static FoS (DGMS)": ["1.50", "1.30"],
                                   "Maximum PoF (Read & Stacey 2009)": ["5 %", "10 %"]}), hide_index=True)
        st.markdown("**TARP classes:** Green = meets the criteria; Yellow = marginal (one criterion missed, or the "
                    "prediction interval crosses the minimum FoS); Orange = unstable design (criteria missed); "
                    "Red = FOS below 1. The multi-parameter TARP page combines this class with rainfall, movement, "
                    "acceleration and visual triggers.")

    with st.expander("10. Evaluation metrics"):
        st.latex(r"R^2=1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2},\ \ RMSE=\sqrt{\tfrac1n\sum(y_i-\hat y_i)^2},"
                 r"\ \ MAE=\tfrac1n\sum|y_i-\hat y_i|,\ \ MAPE=\tfrac{100}n\sum\left|\tfrac{y_i-\hat y_i}{y_i}\right|")
        st.markdown("R^2 is negative when a model does worse than always predicting the mean of the test data. "
                    "Permutation importance = increase in RMSE when one input is randomly shuffled.")

    with st.expander("References"):
        st.markdown("""
- Bishop A.W. (1955) The use of the slip circle in the stability analysis of slopes. Géotechnique 5(1):7-17.
- Bishop A.W., Morgenstern N. (1960) Stability coefficients for earth slopes. Géotechnique 10(4):129-150.
- Janbu N. (1954) Stability analysis of slopes with dimensionless parameters. Harvard Soil Mech. Series 46.
- Carla T., Intrieri E., Di Traglia F., Nolesini T., Gigli G., Casagli N. (2017) Guidelines on the use of inverse velocity method as a tool for setting alarm thresholds and forecasting landslides and structure collapses. Landslides 14(2):517-534.
- India Meteorological Department. Rainfall intensity classes (heavy 64.5-115.5, very heavy 115.6-204.4, extremely heavy >= 204.5 mm in 24 h).
- Fukuzono T. (1985) A new method for predicting the failure time of a slope. 4th Int. Conf. Field Workshop on Landslides, Tokyo, 145-150.
- Kulhawy F.H. (1992) On the evaluation of static soil properties. ASCE GSP 31:95-115.
- DGMS. Guidelines for the scientific study under Regulation 106(2) of CMR 2017 (technical circular), para 3.2 (minimum factor of safety) and para 4.1 (monitoring and TARP).
- Read J., Stacey P. (eds) (2009) Guidelines for Open Pit Slope Design, CSIRO/CRC Press; ch. 9 Wesseloo J., Read J., Acceptance criteria.
- Vovk V., Gammerman A., Shafer G. (2005) Algorithmic Learning in a Random World. Springer.
- Coal Mines Regulations 2017, Regulation 106 (spoil banks), Govt. of India.
- NIT Rourkela (2025) Safety and feasibility study of filling of fly ash in eight opencast mines of SECL, Project C3/23/MN/049.
- Chourasia K., Ravi S., Khare R.N. (2025) Jagannathpur OCP dump optimisation, Research Square rs-8013159.
- Kainthola A. et al. (2011) A coal mine dump stability analysis - a case study. Geomaterials 1:1-13.
- Kumar A. et al. (2023) Probabilistic slope stability analysis of coal mine waste rock dump. Geotech. Geol. Eng. 41:4707-4724.
- Sahoo A.K., Tripathy D.P., Jayanthu S. (2025) Sci Rep 15:40985, and its Mendeley dataset.
""")


def _status(ok):
    return "<span class='badge ok'>pass</span>" if ok else "<span class='badge warn'>check</span>"


def validation_page(model, physics, real_train, DATA, PLOT, FULL, badge):
    st.title("Validation")
    st.markdown("<p class='lede'>Each part of the model checked against an independent reference. The numbers "
                "are recomputed for the model currently deployed.</p>", unsafe_allow_html=True)
    V = os.path.join(DATA, "validation")
    rows = []

    # V1
    v1 = pd.read_csv(os.path.join(V, "v1_secl.csv"))
    m1 = metrics(v1.FOS, v1.FOS_bishop)
    rows.append(("V1", "Bishop solver vs SECL study (RS2), 8 dumps", f"MAPE {m1['MAPE']:.1f} %, bias "
                 f"{v1.diff_pct.mean():+.1f} %", m1["MAPE"] < 5))
    # V2
    v2 = pd.read_csv(os.path.join(DATA, "mendeley_solver_check.csv"))
    r2 = np.corrcoef(v2.FOS, v2.FOS_paper)[0, 1] ** 2
    rows.append(("V2", "Bishop solver vs 2,250 published Slide/Janbu cases",
                 f"offset {v2.diff_pct.mean():+.2f} % (SD {v2.diff_pct.std():.2f} %), R2 {r2:.4f}",
                 abs(v2.diff_pct.mean()) < 6 and v2.diff_pct.std() < 2))
    # V3
    v3 = pd.read_csv(os.path.join(V, "v3_infinite_slope.csv"))
    rows.append(("V3", "Bishop solver vs closed-form infinite slope (c = 0)",
                 f"max difference {v3.diff_pct.abs().max():.2f} %", v3.diff_pct.abs().max() < 1))
    # V4
    v4 = pd.read_csv(os.path.join(V, "independent_test_exact.csv"))
    p4 = model.predict(v4, with_interval=False)["FOS_physics"].values
    m4 = metrics(v4.FOS, p4)
    rows.append(("V4", "Surrogate vs 250 fresh exact cases", f"MAPE {m4['MAPE']:.2f} %, R2 {m4['R2']:.4f}",
                 m4["MAPE"] < 2))
    # V5
    ho = model.report["surrogate_holdout"]
    rows.append(("V5", "Surrogate 20 % hold-out", f"MAPE {ho['MAPE']:.2f} %, R2 {ho['R2']:.4f}", ho["MAPE"] < 2))
    # V6 monotonic
    mv = max(model.report["monotonic_violations"].values())
    rows.append(("V6", "Physics direction (c, phi up; H, r_u down)", f"worst violation rate {mv*100:.1f} %",
                 mv < 0.02))
    # V7 LOSO per study
    loo = model.report.get("loo")
    if loo is not None:
        p = loo["pred"]
        for s_, d in p.groupby("method"):
            e_h = (abs(d.hybrid - d.FOS_real) / d.FOS_real).mean() * 100
            e_p = (abs(d.physics - d.FOS_real) / d.FOS_real).mean() * 100
            rho = stats.spearmanr(d.FOS_real, d.physics).statistic if len(d) > 2 else np.nan
            rows.append(("V7", f"Leave-one-site-out: {s_}",
                         f"hybrid {e_h:.1f} %, physics {e_p:.1f} %, rank agreement {rho:+.2f}",
                         e_h < 5))
    # V8 Monte Carlo
    v6 = pd.read_csv(os.path.join(V, "v6_mc_exact.csv"))
    g = DumpGeometry(3, 30, 32, 30)
    X = v6.assign(**g.cols())
    pm = model.predict(X, with_interval=False)["FOS_physics"].values
    mc_rows = pd.DataFrame({
        "Exact Bishop": [v6.FOS_exact.mean(), v6.FOS_exact.std(), (v6.FOS_exact < 1).mean() * 100,
                         *[stats.spearmanr(v6[c], v6.FOS_exact).statistic for c in ["c_kPa", "phi_deg", "gamma_kNm3"]]],
        "Model": [pm.mean(), pm.std(), (pm < 1).mean() * 100,
                  *[stats.spearmanr(v6[c], pm).statistic for c in ["c_kPa", "phi_deg", "gamma_kNm3"]]]},
        index=["mean FOS", "SD of FOS", "PoF (%)", "rank corr. c", "rank corr. phi", "rank corr. gamma"])
    rows.append(("V8", "Monte Carlo (300 exact runs, Gevra)",
                 f"mean {pm.mean():.3f} vs {v6.FOS_exact.mean():.3f}, SD {pm.std():.3f} vs {v6.FOS_exact.std():.3f}",
                 abs(pm.mean() - v6.FOS_exact.mean()) / v6.FOS_exact.mean() < 0.02))
    # V9 moisture
    v7 = pd.read_csv(os.path.join(V, "v7_moisture_exact.csv"))
    X7 = v7.rename(columns={"gamma": "gamma_kNm3"}).assign(**g.cols())
    p7 = model.predict(X7, with_interval=False)["FOS_physics"].values
    e7 = np.abs(p7 - v7.FOS_exact) / v7.FOS_exact * 100
    rows.append(("V9", "Moisture path vs exact solver", f"max difference {e7.max():.1f} %", e7.max() < 3))
    # V10 WCL
    v8 = pd.read_csv(os.path.join(V, "v8_wcl_failure.csv"))
    pk, rs, rep = v8.FOS_bishop.values
    rows.append(("V10", "WCL failure (75 m, 43 deg) bracketed by peak and residual strength",
                 f"peak {pk:.2f} > reported {rep:.2f} > residual {rs:.2f}", pk > rep > rs))
    # V12 Slide2
    p12 = os.path.join(V, "v12_slide2.csv")
    if os.path.exists(p12):
        v12 = pd.read_csv(p12)
        rows.append(("V12", "Bishop solver vs author's Slide2 runs (Bishop, circular)",
                     "; ".join(f"{r.FOS_bishop:.3f} vs {r.FOS_slide2:.3f} ({r.diff_pct:+.1f} %)" for r in v12.itertuples()),
                     bool((v12.diff_pct.abs() < 2).all())))
        if "FOS_slide2_procedure_bench" in v12:
            rows.append(("V12b", "Slide2-procedure search (auto refine, 50 slices, bench by bench) vs Slide2",
                         "; ".join(f"{r.FOS_slide2_procedure_bench:.3f} vs {r.FOS_slide2:.3f} "
                                   f"({r.diff_pct_procedure:+.2f} %)" for r in v12.itertuples()),
                         bool((v12.diff_pct_procedure.abs() < 1).all())))
    p13 = os.path.join(V, "v13_slide2_procedure_vs_labels.csv")
    if os.path.exists(p13):
        v13 = pd.read_csv(p13)
        rows.append(("V13", "Slide2-procedure search vs training labels (60 random rows)",
                     f"mean {v13.diff_pct.mean():+.2f} %, SD {v13.diff_pct.std():.2f} %, "
                     f"max {v13.diff_pct.abs().max():.2f} %", bool(v13.diff_pct.abs().max() < 2)))
    # V11 DGMS
    ok_secl = dgms_check(3, 30, 32, 30)[0]
    ok_wcl = dgms_check(1, 75, 43, 0)[0]
    rows.append(("V11", "DGMS Reg. 106 check on known dumps",
                 f"SECL 3 x 30 m at 32 deg compliant: {ok_secl}; WCL 75 m unbenched compliant: {ok_wcl}",
                 ok_secl and not ok_wcl))

    st.subheader("Checks against independent references")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**V1 - SECL study (RS2) vs in-house Bishop**")
        st.dataframe(v1.round(3), hide_index=True, **FULL)
        st.markdown("**V3 - closed-form check** (a cohesionless slope fails on a shallow plane: F = tan phi / tan beta)")
        st.dataframe(v3.round(3), hide_index=True, **FULL)
    with c2:
        fig = go.Figure(go.Scatter(x=v4.FOS, y=p4, mode="markers", marker=dict(size=6, color="#2F6B6A"),
                                   name="cases"))
        lim = [min(v4.FOS.min(), p4.min()), max(v4.FOS.max(), p4.max())]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dot", color="#999"), name="1:1"))
        fig.update_layout(**PLOT, title="V4 - surrogate vs exact on 250 fresh cases", showlegend=False)
        fig.update_xaxes(title="Exact Bishop FOS"); fig.update_yaxes(title="Surrogate FOS")
        st.plotly_chart(fig, **FULL)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**V8 - Monte Carlo against exact**")
        st.dataframe(mc_rows.round(3), **FULL)
    with c2:
        st.markdown("**V9 - moisture path** (c 44 kPa, phi 30 deg, dry unit weight 17 kN/m3, 3 x 30 m)")
        st.dataframe(v7.assign(FOS_model=p7, diff_pct=e7).round(3), hide_index=True, **FULL)
    if os.path.exists(p12):
        st.markdown("**V12 - author's Slide2 runs vs this app** (old two-stage search and the Slide2-procedure search)")
        st.dataframe(pd.read_csv(p12).round(3), hide_index=True, **FULL)
    if os.path.exists(p13):
        st.markdown("**V13 - Slide2-procedure search vs the training labels.** The slip circles drawn in the app "
                    "(Slide2 procedure) and the FOS values the model was trained on (two-stage search) agree, so "
                    "the training data did not need to be re-solved.")
        st.dataframe(v13.round(3), hide_index=True, **FULL)
    st.markdown("**V10 - WCL failure.** With peak strength Bishop gives 1.07 (stable), with residual strength "
                "0.72; the dump actually failed and the source's FEM analysis gave 0.80. Field strength at failure "
                "lies between peak and residual, the expected behaviour of a strain-softening spoil.")
    st.subheader("What is not validated (assumptions)")
    st.markdown(
        badge("assumption", "warn") + "Moisture softening coefficients k_c, k_phi (off by default) and the saturation "
        "threshold S_c = 0.8 - need direct-shear tests at several moisture contents and piezometer data.<br>" +
        badge("assumption", "warn") + "Rain-to-r_u coefficient on the TARP page - fitted automatically when piezometer and rain data are both "
        "uploaded; otherwise an assumed value.<br>" +
        badge("assumption", "warn") + "T5 acceleration settings and the rainfall thresholds - defaults from synthetic "
        "records and IMD classes; calibrate on the mine's own prism / radar and rain records.<br>" +
        badge("assumption", "warn") + "Jagannathpur and WCL corrections - each study is a single site, so its offset "
        "cannot be checked on another site (V7 shows physics-only error for these).<br>" +
        badge("source", "info") + "Velocity TARP bands - from the scientific-study table, not derived here.<br>" +
        badge("source", "info") + "Minimum FoS - DGMS circular (Reg. 106(2) of CMR 2017), para 3.2: 1.50 permanent / "
        "long-standing, 1.30 other cases. Maximum PoF - Read & Stacey (2009) Table 9.9.<br>" +
        badge("source", "info") + "CMR 2017 Reg. 106 limits - taken from published summaries; confirm against the "
        "gazette text.", unsafe_allow_html=True)
