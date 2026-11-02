"""Training and caching of the FOS engine (T1, T2).

Training takes about 30 s on a fast PC and a few minutes on a small cloud server, so the
two standard models (DGMS-compliant only, and with non-compliant designs) are trained once
by `python scripts/pretrain.py` and saved in models/. The app loads them in a second and
only retrains when the data files or the scikit-learn version differ from the saved ones.
"""
from __future__ import annotations
import hashlib, os
import numpy as np, pandas as pd
import sklearn

from core.lem import DumpGeometry
from core.models import FOSModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
REQ = ["c_kPa", "phi_deg", "gamma_kNm3", "H_m", "FOS"]


def training_rows(real: pd.DataFrame) -> pd.DataFrame:
    r = real.copy()
    if "use_for_training" in r:
        r = r[r["use_for_training"].astype(str).str.lower().isin(["true", "1", "yes"])]
    need = r["beta_overall_deg"].isna() & r["deck_angle_deg"].notna() & r["n_decks"].notna()
    for i in r[need].index:
        bw = r.at[i, "berm_width_m"]
        g = DumpGeometry(int(r.at[i, "n_decks"]), float(r.at[i, "deck_height_m"]),
                         float(r.at[i, "deck_angle_deg"]), 0.0 if pd.isna(bw) else float(bw))
        r.at[i, "beta_overall_deg"] = g.overall_angle
    return r.dropna(subset=REQ + ["beta_overall_deg"])


def fit_model(real: pd.DataFrame, physics: pd.DataFrame, noncompliant: bool, alpha: float = 0.10):
    tr = training_rows(real)
    ph = physics
    if not noncompliant:
        ph = ph[ph.dgms_compliant]
        if "dgms_compliant" in tr:
            tr = tr[~tr["dgms_compliant"].astype(str).str.startswith("no")]
    return FOSModel().fit(ph, tr, alpha=alpha)


def data_signature():
    h = hashlib.sha256()
    for f in ("real_cases.csv", "physics_dataset.csv"):
        with open(os.path.join(DATA, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def _path(noncompliant):
    return os.path.join(MODELS, f"fos_engine_{'all' if noncompliant else 'dgms'}.joblib")


def save(model, noncompliant, alpha):
    import joblib
    os.makedirs(MODELS, exist_ok=True)
    joblib.dump(dict(model=model, sklearn=sklearn.__version__, numpy=np.__version__,
                     data=data_signature(), alpha=alpha), _path(noncompliant), compress=3)


def load(noncompliant, alpha):
    """Saved model if it was built from the same data files, the same alpha and the same
    scikit-learn major.minor version; otherwise None (the caller then retrains)."""
    p = _path(noncompliant)
    if not os.path.exists(p):
        return None, "no saved model"
    try:
        import joblib, warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pack = joblib.load(p)
    except Exception as e:                     # incompatible pickle
        return None, f"saved model unreadable ({type(e).__name__})"
    mm = lambda v: ".".join(v.split(".")[:2])
    if mm(pack["sklearn"]) != mm(sklearn.__version__):
        return None, f"saved with scikit-learn {pack['sklearn']}, server has {sklearn.__version__}"
    if pack["data"] != data_signature() or abs(pack["alpha"] - alpha) > 1e-9:
        return None, "data files changed since the model was saved"
    return pack["model"], "loaded saved model"
