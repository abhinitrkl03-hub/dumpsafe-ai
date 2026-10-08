"""
Bishop's Simplified Method (limit equilibrium) for benched OB dump slopes.

A fully vectorised circular-slip search written from scratch for this project,
so that every synthetic FOS used to train the ML surrogate is traceable to a
documented physical calculation (not to an empirical "FOS adjustment formula").

Coordinate system
-----------------
Toe of the dump at (0, 0). The dump rises to the right; the slide mass moves
to the left (down-slope). Foundation (y < 0) is treated as competent, so slip
surfaces that would dip below the base are clipped to slide along the base
(composite surface) - the usual assumption for dumps on firm ground.

Pore pressure is represented by the pore-pressure ratio r_u = u / (gamma * h)
(Bishop & Morgenstern, 1960).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------- geometry
@dataclass
class DumpGeometry:
    n_decks: int = 3
    deck_height: float = 30.0      # m, height of each full deck (bench)
    deck_angle: float = 32.0       # deg, face angle of one deck
    berm_width: float = 30.0       # m, width of berm between decks
    top_deck_height: float | None = None   # m, height of the top deck if lower (e.g. 30+30+20)

    def deck_heights(self) -> list:
        hs = [self.deck_height] * self.n_decks
        if self.top_deck_height is not None and self.n_decks >= 1:
            hs[-1] = self.top_deck_height
        return hs

    @property
    def height(self) -> float:
        return float(sum(self.deck_heights()))

    @property
    def horizontal_extent(self) -> float:
        cot = 1.0 / np.tan(np.radians(self.deck_angle))
        return self.height * cot + (self.n_decks - 1) * self.berm_width

    @property
    def overall_angle(self) -> float:
        return float(np.degrees(np.arctan(self.height / self.horizontal_extent)))

    def surface(self) -> tuple[np.ndarray, np.ndarray]:
        """Polyline of the ground surface (x, y), toe at origin."""
        cot = 1.0 / np.tan(np.radians(self.deck_angle))
        xs, ys = [-3.0 * self.height - 50.0, 0.0], [0.0, 0.0]
        x, y = 0.0, 0.0
        for i, h in enumerate(self.deck_heights()):
            x += h * cot
            y += h
            xs.append(x); ys.append(y)
            if i < self.n_decks - 1:
                x += self.berm_width
                xs.append(x); ys.append(y)
        xs.append(x + 3.0 * self.height + 50.0); ys.append(y)
        return np.array(xs), np.array(ys)

    def cols(self) -> dict:
        """Geometry columns expected by the ML model."""
        return dict(H_m=self.height, beta_overall_deg=self.overall_angle,
                    deck_angle_deg=self.deck_angle, n_decks=self.n_decks,
                    deck_height_m=self.deck_height, berm_width_m=self.berm_width)


    @staticmethod
    def dgms_benched(height: float, overall_angle: float, deck_angle: float = 35.0,
                     max_bench: float = 30.0) -> "DumpGeometry":
        """Bench a dump of given total height and overall angle the way mines do under
        CMR 2017 Reg. 106: full 30 m decks from the bottom, the remainder as the top deck,
        berms sized so the overall angle is exactly `overall_angle`."""
        n = int(np.ceil(height / max_bench - 1e-9))
        top = height - (n - 1) * max_bench
        if n == 1:
            return DumpGeometry(1, height, overall_angle, 0.0)
        horiz = height / np.tan(np.radians(overall_angle))
        faces = height / np.tan(np.radians(deck_angle))
        berm = (horiz - faces) / (n - 1)
        return DumpGeometry(n, max_bench, deck_angle, max(berm, 0.0),
                            None if abs(top - max_bench) < 1e-6 else top)

    @staticmethod
    def from_overall(height: float, overall_angle: float) -> "DumpGeometry":
        """Single uniform slope when only overall H and beta are known."""
        return DumpGeometry(n_decks=1, deck_height=height,
                            deck_angle=overall_angle, berm_width=0.0)


# ---------------------------------------------------------------- solver
@dataclass
class BishopResult:
    fos: float
    xc: float
    yc: float
    radius: float
    x_exit: float
    n_circles: int


def _bishop_batch(xc, yc, R, x_exit, xg, yg_fun, c, phi, gamma, ru, n_x=140, composite=False):
    """FOS of many circles at once. Returns array of FOS (inf where invalid)."""
    M = xc.size
    tanphi = np.tan(np.radians(phi))
    # x-sampling per circle: from exit point to right edge of circle
    t = (np.arange(n_x) + 0.5) / n_x
    x_lo = x_exit[:, None]
    x_hi = (xc + R)[:, None]
    X = x_lo + (x_hi - x_lo) * t[None, :]
    dx = ((x_hi - x_lo) / n_x)            # (M,1)
    dxc = X - xc[:, None]
    inside = np.abs(dxc) < R[:, None]
    root = np.sqrt(np.clip(R[:, None] ** 2 - dxc ** 2, 1e-9, None))
    ys = yc[:, None] - root
    alpha = np.arctan(dxc / root)
    below = ys < 0.0                      # circle reaches the dump base
    # composite=False (Slide2 default): a circle may not pass below the base -> rejected.
    # composite=True: the part below the base is replaced by sliding along the base.
    bad_circle = np.any(below & inside, axis=1) if not composite else np.zeros(M, bool)
    ys = np.where(below, 0.0, ys)
    alpha = np.where(below, 0.0, alpha)
    yg = yg_fun(X)
    h = yg - ys
    valid = inside & (h > 0)
    h = np.where(valid, h, 0.0)
    W = gamma * h * dx
    u_b = ru * gamma * h * dx             # pore force on slice base (u * b)
    drive = np.sum(W * np.sin(alpha), axis=1)
    cb = np.where(valid, c * dx, 0.0)
    num_const = cb + (W - u_b).clip(min=0) * tanphi
    ok = (drive > 1e-6) & (h.max(axis=1) > 0.5) & ~bad_circle
    F = np.full(M, 1.5)
    cos_a, tan_a = np.cos(alpha), np.tan(alpha)
    for _ in range(40):
        m_alpha = cos_a * (1.0 + tan_a * tanphi / F[:, None])
        m_alpha = np.clip(m_alpha, 0.2, None)     # standard numerical guard
        F_new = np.sum(num_const / m_alpha, axis=1) / np.where(ok, drive, 1.0)
        F_new = np.where(ok, F_new, np.inf)
        diff = np.abs(F_new[ok] - F[ok]) if ok.any() else np.zeros(1)
        if diff.max(initial=0.0) < 1e-4:
            F = F_new
            break
        F = np.where(ok, F_new, np.inf)
    return F


COMPOSITE_DEFAULT = False


def bishop_fos(geom: DumpGeometry, c: float, phi: float, gamma: float,
               ru: float = 0.0, fine: bool = True, composite: bool | None = None) -> BishopResult:
    """Minimum Bishop FOS: global grid search + bench-scale searches (one per bench),
    then two refinements around every candidate; the lowest is returned."""
    xs, ys_ = geom.surface()
    yg_fun = lambda X: np.interp(X, xs, ys_)
    H, L = geom.height, geom.horizontal_extent
    composite = COMPOSITE_DEFAULT if composite is None else composite

    def search(xc_rng, yc_rng, xe_rng, n_c, n_e):
        XC, YC, XE = np.meshgrid(np.linspace(*xc_rng, n_c),
                                 np.linspace(*yc_rng, n_c),
                                 np.linspace(*xe_rng, n_e), indexing="ij")
        XC, YC, XE = XC.ravel(), YC.ravel(), XE.ravel()
        R = np.hypot(XC - XE, YC - yg_fun(XE))
        keep = YC > yg_fun(XE) + 1.0
        XC, YC, XE, R = XC[keep], YC[keep], XE[keep], R[keep]
        if XC.size == 0:
            return np.inf, 0.0, 0.0, 0.0, 0.0, 0
        F = _bishop_batch(XC, YC, R, XE, xs, yg_fun, c, phi, gamma, ru, composite=composite)
        i = int(np.argmin(F))
        return F[i], XC[i], YC[i], R[i], XE[i], F.size

    # Stage 1a - global search: exit points from in front of the toe up to 2/3 of the profile
    xe_lo = -0.25 * H if composite else 0.0
    best = search((-0.4 * L, 1.1 * L), (0.4 * H, 2.5 * H + 0.5 * L),
                  (xe_lo, 0.66 * L), 22 if fine else 14, 12 if fine else 8)
    n = best[5]
    cands = [best]
    # Stage 1b - bench-scale search: circles cutting a single bench (or two adjacent ones).
    # Without it a coarse global grid can miss small bench circles, which in benched dumps
    # often have the lowest FOS (found when checking against Slide2).
    if geom.n_decks > 1:
        cot = 1.0 / np.tan(np.radians(geom.deck_angle))
        x0 = y0 = 0.0
        for h in geom.deck_heights():
            f = h * cot
            xt, yt, xcr, ycr = x0, y0, x0 + f, y0 + h
            cand = search((xt - 0.5 * h, xcr + 1.5 * h), (ycr - 0.2 * h, ycr + 4.0 * h),
                          (max(xt - 0.6 * h, xe_lo), xt + 0.6 * f), 16 if fine else 10, 8 if fine else 5)
            n += cand[5]
            cands.append(cand)
            x0, y0 = xcr + geom.berm_width, ycr
    cands = [c_ for c_ in cands if np.isfinite(c_[0])]
    if not cands:
        return BishopResult(float("inf"), 0.0, 0.0, 0.0, 0.0, int(n))
    # Stage 2 - refine EVERY candidate (global + one per bench) and keep the lowest.
    # Benches of equal size give nearly equal FOS; refining only the best coarse one can
    # settle on the wrong bench (e.g. the bottom bench, whose circle is cut off by the base).
    best = min(cands, key=lambda t: t[0])
    if fine:
        for cand in cands:
            F, xc, yc, R, xe, _ = cand
            for scale in (0.30, 0.10):
                dx, dy, de = scale * R + 2, scale * R + 2, 0.5 * scale * R + 1
                F2, xc2, yc2, R2, xe2, n2 = search((xc - dx, xc + dx), (max(yc - dy, 1), yc + dy),
                                                   (max(xe - de, xe_lo), xe + de), 14, 7)
                n += n2
                if F2 < F:
                    F, xc, yc, R, xe = F2, xc2, yc2, R2, xe2
            if F < best[0]:
                best = (F, xc, yc, R, xe, 0)
    F, xc, yc, R, xe, _ = best
    return BishopResult(float(F), float(xc), float(yc), float(R), float(xe), int(n))


def slip_arc(res: BishopResult, geom: DumpGeometry, n=200):
    """Coordinates of the critical slip surface for plotting."""
    xs, ys_ = geom.surface()
    X = np.linspace(res.x_exit, res.xc + res.radius, n)
    Y = res.yc - np.sqrt(np.clip(res.radius ** 2 - (X - res.xc) ** 2, 0, None))
    Y = np.maximum(Y, 0.0)
    G = np.interp(X, xs, ys_)
    m = Y <= G + 1e-6
    return X[m], Y[m]
