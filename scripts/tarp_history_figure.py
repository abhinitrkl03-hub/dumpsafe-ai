"""Figure: level of each TARP trigger through the two synthetic demo records.
Run from the project folder:  python scripts/tarp_history_figure.py  (writes tarp_history.png)
Trains the FOS model first (about 30 s)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from core.monitoring import demo_series, demo_storm, process_series, add_moving_averages
from core import tarp
from core.lem import DumpGeometry
from core.models import FOSModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ph = pd.read_csv(os.path.join(ROOT, "data", "physics_dataset.csv"))
ph = ph[ph.dgms_compliant]
real = pd.read_csv(os.path.join(ROOT, "data", "real_cases.csv"))
real = real[real.use_for_training.astype(str).str.lower().isin(["true", "1", "yes"])]
real = real.dropna(subset=["c_kPa", "phi_deg", "gamma_kNm3", "H_m", "FOS", "beta_overall_deg"])
real = real[~real.dgms_compliant.astype(str).str.startswith("no")]
model = FOSModel().fit(ph, real, alpha=0.1)

geom = DumpGeometry(3, 30.0, 32.0, 30.0)          # 3 x 30 m decks at 32 deg, 30 m benches
C, PHI, G, FOS_MIN = 44.0, 30.0, 18.63, 1.50      # Gevra-type spoil, permanent dump
ROWS = ["T1 Design FOS", "T2 Live FOS", "T3 Rainfall", "T4 Velocity", "T5 Acceleration", "Overall"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11})
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
cm = ListedColormap([tarp.COL[k] for k in tarp.ORDER])
cases = [("(a) Demo A: creep to failure (synthetic)", demo_series()),
         ("(b) Demo B: monsoon storm, no acceleration (synthetic)", demo_storm())]
for ax, (name, (df, t_true)) in zip(axes, cases):
    d = add_moving_averages(process_series(df))
    d["rain_24h"] = tarp.rolling_sum(d.t_days, d.rainfall_mm)
    ari = tarp.antecedent_index(d.t_days, d.rainfall_mm)
    ru = (tarp.ru_from_piezometer(d.piezo_head_m, 10.0, G).clip(0, 0.4) if "piezo_head_m" in d
          else np.minimum(0.4, 0.0012 * ari))
    X = pd.DataFrame(dict(c_kPa=C, phi_deg=PHI, gamma_kNm3=G, r_u=ru, **geom.cols()))
    d["FOS"] = model.predict(X, with_interval=False)["FOS"].values
    h = tarp.level_history(d, FOS_MIN)
    h["T1 Design FOS"] = "Green"
    h["Overall"] = [tarp.evaluate({k: r[k] for k in ROWS[:5]})[0] for _, r in h.iterrows()]
    Z = np.array([[tarp.ORDER.index(x) for x in h[r]] for r in ROWS])
    ax.imshow(Z, aspect="auto", cmap=cm, vmin=-0.5, vmax=3.5, interpolation="nearest",
              extent=[d.t_days.iloc[0], d.t_days.iloc[-1], len(ROWS) - 0.5, -0.5])
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels(ROWS)
    ax.set_title(name, loc="left", fontsize=12)
    ax.axhline(4.5, color="white", lw=3); ax.set_xlim(0, 49)
    fo, fr = tarp.first_time(h, "Overall", "Orange"), tarp.first_time(h, "Overall", "Red")
    if t_true:
        ax.axvline(t_true, color="k", ls="--")
        ax.text(t_true - 0.3, -0.3, f"failure\nday {t_true:.0f}", ha="right", va="top", fontsize=9)
        ax.text(2, 5, f"Overall: Orange from day {fo:.1f}, Red from day {fr:.1f} "
                      f"({t_true - fr:.0f} days before failure)", color="white", va="center",
                fontsize=10, fontweight="bold")
        ax.text(2, 3, f"Velocity bands alone: Orange only from day "
                      f"{tarp.first_time(h, 'T4 Velocity', 'Orange'):.1f}, never Red",
                color="white", va="center", fontsize=10)
    else:
        ax.text(1, 5, f"Orange from day {fo:.1f} (rain, live FOS)", color="white", va="center",
                fontsize=10, fontweight="bold")
        ax.text(1, 3, "Velocity bands alone: no warning", color="white", va="center", fontsize=10)
axes[1].set_xlabel("Days since start of record")
fig.legend(handles=[Patch(color=tarp.COL[k], label=k) for k in tarp.ORDER], loc="upper right", ncol=4,
           frameon=False)
plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.5)
plt.savefig(sys.argv[1] if len(sys.argv) > 1 else "tarp_history.png", dpi=200)
