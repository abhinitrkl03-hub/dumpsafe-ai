"""
Sim-to-Real FOS model
=====================
Stage 1  Physics surrogate  - ensemble of neural networks on DIMENSIONLESS groups
                              (Buckingham-Pi / Janbu lambda_c), target log(FOS / tan phi),
                              trained on the Bishop-generated dataset.
Stage 2  Real-data correction - Gaussian-process regression of the residual
                              log(FOS_real) - log(FOS_surrogate) on the REAL cases.
Stage 3  Conformal interval  - distribution-free prediction band from
                              leave-one-out residuals on the real cases.

Why: 50-60 real rows are too few to learn slope physics from scratch, but
enough to learn how real sites (and the software used to analyse them) deviate
from idealised limit-equilibrium physics.
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from dataclasses import dataclass, field
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

BASE = ["c_kPa", "phi_deg", "gamma_kNm3", "H_m", "beta_overall_deg", "deck_angle_deg", "r_u"]
ENG = ["Ns", "tan_ratio", "tan_ratio_deck"]
FEATURES = BASE + ENG
# +1 : FOS must not decrease when the feature increases, -1 : must not increase
MONO = {"c_kPa": 1, "phi_deg": 1, "gamma_kNm3": -1, "H_m": -1, "beta_overall_deg": -1,
        "deck_angle_deg": -1, "r_u": -1, "Ns": 1, "tan_ratio": 1, "tan_ratio_deck": 1}
LABELS = {"c_kPa": "Cohesion c (kPa)", "phi_deg": "Friction angle phi (deg)",
          "gamma_kNm3": "Unit weight gamma (kN/m3)", "H_m": "Dump height H (m)",
          "beta_overall_deg": "Overall slope angle (deg)", "deck_angle_deg": "Deck angle (deg)",
          "r_u": "Pore-pressure ratio r_u", "Ns": "Stability number c/(gamma H)",
          "tan_ratio": "tan(phi)/tan(beta overall)", "tan_ratio_deck": "tan(phi)/tan(deck angle)"}


METHODS = ["FDM-SRM", "FEM-SRM", "FELA"]
MIN_ROWS_PER_METHOD = 3
METHOD_LABELS = {"LEM": "Limit equilibrium (Bishop / Janbu / Spencer)",
                 "FDM-SRM": "Finite difference strength reduction (FLAC/Slope)",
                 "FEM-SRM": "Finite element strength reduction (RS2 / Phase2 / Plaxis)",
                 "FELA": "Finite element limit analysis"}


def method_class(s) -> str:
    s = str(s).upper()
    if "FLAC" in s or "FDM" in s: return "FDM-SRM"
    if "FELA" in s: return "FELA"
    if "FEM" in s or "PHASE" in s or "SRM" in s or "RS2" in s or "PLAXIS" in s: return "FEM-SRM"
    return "LEM"


def group_of(R) -> np.ndarray:
    """Calibration group of each real case: its study (same analysts, software and assumptions)
    when a 'study' column exists, otherwise its analysis-method class."""
    if "study" in R and R["study"].notna().all():
        return R["study"].astype(str).values
    return np.array([method_class(m) for m in R.get("fos_method", ["LEM"] * len(R))])


def method_onehot(classes) -> np.ndarray:
    classes = np.asarray(classes)
    return np.stack([(classes == m).astype(float) for m in METHODS], axis=1)


DIMLESS = ["logNs", "lam", "tanphi", "beta_overall_deg", "deck_angle_deg", "n_decks", "bw_ratio", "r_u"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col, default in [("n_decks", np.nan), ("deck_height_m", np.nan), ("berm_width_m", np.nan)]:
        if col not in d:
            d[col] = default
    d["deck_angle_deg"] = d["deck_angle_deg"].fillna(d["beta_overall_deg"])
    d["r_u"] = d["r_u"].fillna(0.0)
    d["n_decks"] = d["n_decks"].fillna(1.0)
    d["deck_height_m"] = d["deck_height_m"].fillna(d["H_m"] / d["n_decks"])
    d["berm_width_m"] = d["berm_width_m"].fillna(0.0)
    # dimensionless groups: FOS of a homogeneous slope depends on c/(gamma H), phi,
    # geometry ratios and r_u only (dimensional analysis; Janbu 1954, Bishop & Morgenstern 1960)
    d["tanphi"] = np.tan(np.radians(d["phi_deg"]))
    d["logNs"] = np.log(d["c_kPa"] / (d["gamma_kNm3"] * d["H_m"]) + 1e-4)
    d["lam"] = np.log(d["c_kPa"] / (d["gamma_kNm3"] * d["H_m"] * d["tanphi"]) + 1e-4)
    d["bw_ratio"] = d["berm_width_m"] / d["deck_height_m"]
    d["Ns"] = d["c_kPa"] / (d["gamma_kNm3"] * d["H_m"])
    tp = np.tan(np.radians(d["phi_deg"]))
    d["tan_ratio"] = tp / np.tan(np.radians(d["beta_overall_deg"]))
    d["tan_ratio_deck"] = tp / np.tan(np.radians(d["deck_angle_deg"]))
    return d


def _varying(X, tol=1e-6):
    """Columns that actually vary among the real cases. A correction cannot be learned
    along a direction the real data never moves in (e.g. geometry when all cases share
    one design), so the correction is kept constant along such directions."""
    cols = [j for j in range(X.shape[1]) if X[:, j].std() > tol]
    return cols or [0]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(R2=r2_score(y, p) if len(y) > 1 else np.nan,
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)),
                MAPE=float(np.mean(np.abs((y - p) / y)) * 100))


class DimlessSurrogate:
    """Ensemble of small neural networks on dimensionless groups.
    predict() returns log(FOS); predict_std() the ensemble spread (epistemic)."""
    def __init__(self, n_members=5, hidden=(128, 64, 32)):
        self.n_members, self.hidden = n_members, hidden

    def fit(self, D, logfos):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        y = np.asarray(logfos) - np.log(D["tanphi"].values)
        self.members = [make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=self.hidden, max_iter=3000, alpha=1e-4, learning_rate_init=3e-3,
            early_stopping=True, n_iter_no_change=50, random_state=seed)).fit(D[DIMLESS], y)
            for seed in range(self.n_members)]
        return self

    def _all(self, D):
        D = D if "lam" in D else engineer(D)
        return np.stack([m.predict(D[DIMLESS]) for m in self.members]) + np.log(D["tanphi"].values)

    def predict(self, D):
        return self._all(D).mean(axis=0)

    def predict_std(self, D):
        return self._all(D).std(axis=0)


def make_surrogate(monotonic=True):
    return HistGradientBoostingRegressor(
        max_iter=700, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=12,
        l2_regularization=0.5, early_stopping=False, random_state=7,
        monotonic_cst=[MONO[f] for f in FEATURES] if monotonic else None)


def make_gp(restarts=2, n_dim=None):
    n_dim = n_dim or len(FEATURES) + len(METHODS)
    k = C(0.05, (1e-4, 1.0)) * RBF(length_scale=np.ones(n_dim) * 2.0,
                                   length_scale_bounds=(0.3, 100.0)) + WhiteKernel(1e-3, (1e-6, 0.1))
    return GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=restarts,
                                    random_state=7)


@dataclass
class FOSModel:
    surrogate: object = None
    free: object = None                  # unconstrained twin, for comparison only
    gp: object = None
    scaler: StandardScaler = None
    conformal_q: float = 0.10            # half-width in log space
    q_by_method: dict = field(default_factory=dict)
    methods_seen: tuple = ()
    gps: dict = field(default_factory=dict)
    gp_cols: dict = field(default_factory=dict)
    default_method: str = "LEM"
    alpha: float = 0.10
    report: dict = field(default_factory=dict)
    n_real: int = 0

    # ------------------------------------------------------------ training
    def fit(self, phys: pd.DataFrame, real: pd.DataFrame, alpha: float = 0.10):
        self.alpha = alpha
        P = engineer(phys)
        Ptr, Pte = train_test_split(P, test_size=0.2, random_state=7)
        yte = np.log(Pte["FOS"].values)
        self.surrogate = DimlessSurrogate().fit(Ptr, np.log(Ptr["FOS"]))
        pte = np.exp(self.surrogate.predict(Pte))
        self.report["surrogate_holdout"] = metrics(np.exp(yte), pte)
        # benchmarks: tree boosters on raw inputs (what most papers do)
        mono = make_surrogate(True).fit(Ptr[FEATURES], np.log(Ptr["FOS"]))
        self.report["mono_holdout"] = metrics(np.exp(yte), np.exp(mono.predict(Pte[FEATURES])))
        self.free = make_surrogate(False).fit(Ptr[FEATURES], np.log(Ptr["FOS"]))
        self.report["free_holdout"] = metrics(np.exp(yte), np.exp(self.free.predict(Pte[FEATURES])))
        self.report["monotonic_violations"] = self.monotonic_check(Pte)
        self.surrogate = DimlessSurrogate().fit(P, np.log(P["FOS"]))   # refit on all physics data

        R = engineer(real)
        self.n_real = len(R)
        self.scaler = StandardScaler().fit(P[BASE])      # correction works on raw inputs only
        if len(R) >= 3:
            meth = group_of(R)
            resid = np.log(R["FOS"].values) - self.surrogate.predict(R)
            X = self.scaler.transform(R[BASE])
            self.gps, self.gp_cols = {}, {}
            for mth in np.unique(meth):          # one correction model per analysis method
                k = meth == mth
                if k.sum() >= MIN_ROWS_PER_METHOD:
                    cols = _varying(X[k])
                    self.gp_cols[mth] = cols
                    self.gps[mth] = make_gp(2, len(cols)).fit(X[k][:, cols], resid[k])
            self.methods_seen = tuple(sorted(self.gps))
            # default calibration = the group covering the most distinct dump sites
            sites = R["site"].values if "site" in R else np.arange(len(R))
            n_sites = {g: len(set(sites[meth == g])) for g in self.gps}
            self.default_method = max(n_sites, key=n_sites.get) if n_sites else "LEM"
            self.report["loo"] = self.leave_one_out(real)
            lo = self.report["loo"]["residuals_log"]
            lm = self.report["loo"]["pred"]["method"].values
            self.conformal_q = self._cq(lo, alpha)
            for mth in np.unique(lm):
                if (lm == mth).sum() >= 5:
                    self.q_by_method[mth] = self._cq(lo[lm == mth], alpha)
        else:   # fallback band from the physics hold-out
            self.conformal_q = float(np.quantile(np.abs(np.log(pte) - yte), 1 - alpha))
        return self

    @staticmethod
    def _cq(res, alpha):
        n = len(res)
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        return float(np.sort(np.abs(res))[k - 1])

    def _gp_X(self, D, method=None):
        if "fos_method" in D and method is None:
            cls = [method_class(m) for m in D["fos_method"]]
        else:
            cls = [method or "LEM"] * len(D)
        return np.hstack([self.scaler.transform(D[FEATURES]), method_onehot(cls)])

    # ------------------------------------------------------------ inference
    def predict(self, df: pd.DataFrame, with_interval=True, method=None):
        """FOS as it would be reported by `method` (default: the method with most real cases).
        FOS_physics is the in-house Bishop (limit-equilibrium) surrogate without correction."""
        method = method or self.default_method
        D = engineer(df)
        log_s = self.surrogate.predict(D)
        log_h = log_s.copy()
        gp_sd = np.zeros(len(D))
        gp = self.gps.get(method)
        if gp is not None:
            Xq = self.scaler.transform(D[BASE])[:, self.gp_cols[method]]
            corr, gp_sd = gp.predict(Xq, return_std=True)
            log_h = log_s + corr
        elif self.gps:        # no real case analysed with this method: physics only, wide band
            prior = max(np.sqrt(g.kernel_.k1.k1.constant_value) for g in self.gps.values())
            gp_sd = np.full(len(D), float(prior))
        out = pd.DataFrame({"FOS_physics": np.exp(log_s), "FOS": np.exp(log_h)}, index=df.index)
        if with_interval:
            # conformal half-width near the real data, widened by the GP's own
            # uncertainty when the query moves away from the real cases
            from scipy.stats import norm
            z = norm.ppf(1 - self.alpha / 2)
            q = self.q_by_method.get(method, self.conformal_q)
            ens_sd = self.surrogate.predict_std(D)
            hw = np.sqrt(q ** 2 + (z * gp_sd) ** 2 + (z * ens_sd) ** 2)
            out["FOS_lo"] = np.exp(log_h - hw)
            out["FOS_hi"] = np.exp(log_h + hw)
            out["gp_sd"] = gp_sd
        return out

    # ------------------------------------------------------------ validation
    def leave_one_out(self, real: pd.DataFrame, group_col=None):
        """Leave-one-SITE-out comparison of three strategies on real data
        (all rows of a dump are held out together, so no information leaks)."""
        R = engineer(real).reset_index(drop=True)
        if group_col is None:
            group_col = "site" if "site" in R else "mine"
        y = R["FOS"].values
        groups = R[group_col].values if group_col in R else np.arange(len(R))
        meth = group_of(R)
        p_phys = np.exp(self.surrogate.predict(R))
        p_hyb, p_real = np.zeros(len(R)), np.zeros(len(R))
        Xs = self.scaler.transform(R[BASE])
        for g in np.unique(groups):
            te = groups == g; tr = ~te
            if tr.sum() < 2:
                p_hyb[te] = p_phys[te]; p_real[te] = y[tr].mean() if tr.any() else y.mean(); continue
            for mth in np.unique(meth[te]):
                t2 = te & (meth == mth)
                k = tr & (meth == mth)
                if k.sum() >= MIN_ROWS_PER_METHOD:
                    cols = _varying(Xs[k])
                    gp = make_gp(0, len(cols)).fit(Xs[k][:, cols], np.log(y[k]) - np.log(p_phys[k]))
                    p_hyb[t2] = p_phys[t2] * np.exp(gp.predict(Xs[t2][:, cols]))
                else:
                    p_hyb[t2] = p_phys[t2]     # no other site analysed with this method
            # real-only baseline: a GP learning FOS from the few real rows alone
            sc = StandardScaler().fit(R.loc[tr, FEATURES])
            gpr = make_gp(0, len(FEATURES)).fit(sc.transform(R.loc[tr, FEATURES]),
                                                 np.log(y[tr]) - np.log(y[tr]).mean())
            p_real[te] = np.exp(gpr.predict(sc.transform(R.loc[te, FEATURES])) + np.log(y[tr]).mean())
        return {"table": pd.DataFrame({"Real-data only (GP)": metrics(y, p_real),
                                       "Physics surrogate only": metrics(y, p_phys),
                                       "Sim-to-real hybrid": metrics(y, p_hyb)}).T,
                "pred": pd.DataFrame({"case": R.get("case_id", pd.Series(range(len(R)))),
                                      "mine": groups, "method": meth, "FOS_real": y, "real_only": p_real,
                                      "physics": p_phys, "hybrid": p_hyb}),
                "residuals_log": np.log(y) - np.log(p_hyb)}

    def monotonic_check(self, D, n=150, seed=0):
        """Share of one-at-a-time sweeps in which predicted FOS moves the wrong way
        (e.g. falls when cohesion rises) by more than 0.5 %."""
        rng = np.random.default_rng(seed)
        S = D.sample(min(n, len(D)), random_state=seed)
        out = {}
        for f, sign, lo, hi in [("c_kPa", 1, 0.5, 150), ("phi_deg", 1, 5, 45),
                                ("H_m", -1, 10, 120), ("r_u", -1, 0, 0.5)]:
            bad = 0
            for _, r in S.iterrows():
                grid = pd.DataFrame([r] * 12)
                grid[f] = np.linspace(lo, hi, 12)
                if f == "H_m":
                    grid["deck_height_m"] = grid["H_m"] / grid["n_decks"]
                y = self.surrogate.predict(engineer(grid))
                bad += np.any(sign * np.diff(y) < -0.005)
            out[LABELS[f]] = bad / len(S)
        return out

    def predict_free(self, df):
        return np.exp(self.free.predict(engineer(df)[FEATURES]))

    def transfer_test(self, real: pd.DataFrame, bishop_fn, geoms):
        """Ask each model about the REAL materials in geometries / water states that
        never appear in the real data, and compare with the exact Bishop answer."""
        R = engineer(real).reset_index(drop=True)
        sc = StandardScaler().fit(R[FEATURES])
        gpr = make_gp(2, len(FEATURES)).fit(sc.transform(R[FEATURES]), np.log(R["FOS"]) - np.log(R["FOS"]).mean())
        rows = []
        for gname, (geom, ru) in geoms.items():
            for r in R.itertuples():
                d = pd.DataFrame([dict(c_kPa=r.c_kPa, phi_deg=r.phi_deg, gamma_kNm3=r.gamma_kNm3,
                                       r_u=ru, **geom.cols())])
                D = engineer(d)
                rows.append(dict(scenario=gname, mine=r.mine,
                                 exact=bishop_fn(geom, r.c_kPa, r.phi_deg, r.gamma_kNm3, ru),
                                 real_only=float(np.exp(gpr.predict(sc.transform(D[FEATURES]))[0]
                                                        + np.log(R["FOS"]).mean())),
                                 physics=float(self.predict(d, with_interval=False)["FOS_physics"].iloc[0]),
                                 hybrid=float(self.predict(d, with_interval=False)["FOS"].iloc[0])))
        out = pd.DataFrame(rows)
        tab = pd.DataFrame({"Real-data only (GP)": metrics(out.exact, out.real_only),
                            "Physics surrogate": metrics(out.exact, out.physics),
                            "Sim-to-real hybrid": metrics(out.exact, out.hybrid)}).T
        return out, tab

    def permutation_importance(self, phys: pd.DataFrame, n=800, seed=0):
        """Permute one raw input at a time (dimensionless groups are recomputed)."""
        rng = np.random.default_rng(seed)
        D = phys.sample(min(n, len(phys)), random_state=seed).reset_index(drop=True)
        y = D["FOS"].values
        base_err = np.sqrt(np.mean((np.exp(self.surrogate.predict(engineer(D))) - y) ** 2))
        imp = {}
        for f in ["c_kPa", "phi_deg", "gamma_kNm3", "H_m", "beta_overall_deg", "deck_angle_deg", "r_u"]:
            Dp = D.copy(); Dp[f] = rng.permutation(Dp[f].values)
            p = np.exp(self.surrogate.predict(engineer(Dp)))
            imp[LABELS[f]] = np.sqrt(np.mean((p - y) ** 2)) - base_err
        return pd.Series(imp).sort_values()
