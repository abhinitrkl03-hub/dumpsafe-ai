"""
Mechanistic moisture channel.

In most ML dump-stability papers moisture is fed to the model as a bare column.
Because the LEM that produced the labels never used it, the model (correctly)
learns that it does nothing - e.g. SHAP importance ~0 in Sahoo et al. (2025).

Here moisture acts on stability only through physical mechanisms:
  1. bulk unit weight      gamma = gamma_dry * (1 + w)
  2. degree of saturation  S = w*Gs / e ,  e = Gs*gamma_w/gamma_dry - 1
  3. pore pressure         r_u rises once S exceeds a threshold (perched water)
  4. optional strength softening beyond OMC, calibrated from YOUR lab tests
Each step is transparent and every coefficient is exposed in the app.
"""
from __future__ import annotations
import numpy as np

GAMMA_W = 9.81


def bulk_unit_weight(gamma_dry, w_pct):
    return np.asarray(gamma_dry) * (1.0 + np.asarray(w_pct) / 100.0)


def degree_of_saturation(gamma_dry, w_pct, Gs=2.60):
    e = Gs * GAMMA_W / np.asarray(gamma_dry) - 1.0
    e = np.clip(e, 0.05, None)
    return np.clip((np.asarray(w_pct) / 100.0) * Gs / e, 0.0, 1.0)


def ru_from_saturation(S, gamma_bulk, S_crit=0.80, max_phreatic_ratio=1.0):
    """Simplified: no positive pore pressure below S_crit; above it the
    phreatic height ratio (h_w/h) grows linearly to max at S = 1.
    r_u = (gamma_w / gamma) * (h_w / h)."""
    ratio = np.clip((np.asarray(S) - S_crit) / max(1e-6, 1.0 - S_crit), 0.0, 1.0)
    return (GAMMA_W / np.asarray(gamma_bulk)) * ratio * max_phreatic_ratio


def soften(c, phi, w_pct, w_ref, kc=0.0, kphi=0.0):
    """Strength softening beyond a reference moisture (e.g. OMC).
    kc  : fractional loss of cohesion per 1 % moisture above w_ref
    kphi: degrees of friction lost per 1 % moisture above w_ref
    Defaults are 0 (OFF) - set them from your own direct-shear tests."""
    excess = np.clip(np.asarray(w_pct) - w_ref, 0.0, None)
    c_w = np.asarray(c) * np.clip(1.0 - kc * excess, 0.05, 1.0)
    phi_w = np.clip(np.asarray(phi) - kphi * excess, 1.0, None)
    return c_w, phi_w


def moisture_state(c, phi, gamma_dry, w_pct, *, Gs=2.60, S_crit=0.80,
                   max_phreatic_ratio=1.0, w_ref=8.0, kc=0.0, kphi=0.0,
                   ru_external=0.0):
    """Returns the effective (c, phi, gamma, r_u, S) the slope actually sees."""
    gamma = bulk_unit_weight(gamma_dry, w_pct)
    S = degree_of_saturation(gamma_dry, w_pct, Gs)
    ru = ru_from_saturation(S, gamma, S_crit, max_phreatic_ratio)
    ru = np.maximum(ru, ru_external)
    c_w, phi_w = soften(c, phi, w_pct, w_ref, kc, kphi)
    return c_w, phi_w, gamma, ru, S
