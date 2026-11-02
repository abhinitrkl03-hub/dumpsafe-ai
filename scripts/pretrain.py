"""Train the two standard FOS engines once and save them in models/.
Run from the project folder:  python scripts/pretrain.py
Re-run whenever data/real_cases.csv or data/physics_dataset.csv changes, or after
changing the scikit-learn version in requirements.txt."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from core.training import fit_model, save, DATA

real = pd.read_csv(os.path.join(DATA, "real_cases.csv"))
phys = pd.read_csv(os.path.join(DATA, "physics_dataset.csv"))
for nc in (False, True):
    t = time.time()
    m = fit_model(real, phys, nc, alpha=0.10)
    save(m, nc, 0.10)
    print(f"{'all designs' if nc else 'DGMS only'}: trained in {time.time() - t:.0f} s, saved")
