"""
Slide2-style slip-circle search (limit equilibrium, Bishop simplified).

This module follows the same steps a Slide2 user goes through, so that the slip circle shown
in DumpSafe AI is produced the same way as in Slide2 (Rocscience, Slide2 manual: "Surface
Options", "Grid Search", "Slope Search", "Auto Refine Search", "Slope Limits"):

1. Model     - external boundary = dump polygon from the toe (0, 0) to the right boundary,
               base at y = 0 (competent foundation, no circle may pass below it).
2. Material  - Mohr-Coulomb c, phi, unit weight; pore pressure by r_u.
3. Method    - Bishop simplified, 50 slices, tolerance 0.005, max 75 iterations (Slide2 defaults).
4. Surfaces  - circular, by one of the three Slide2 search methods:
                 * grid search        : grid of centres (default 20 x 20 intervals) above the slope,
                                        radius increment 10 at every centre, radii limited by the
                                        slope limits;
                 * slope search       : random pairs of points on the slope within the slope limits
                                        (default 5000 surfaces), circle through both points;
                 * auto refine search : slope split into 20 divisions, 10 circles per pair of
                                        divisions, 10 iterations, the best 50 % of divisions kept
                                        for the next iteration (Slide2 default method).
5. Slope limits - the exit (toe end) and entry (crown end) of every circle must lie inside the
               limits; two separate ranges can be given (Slide2 "two sets of limits"). Moving
               the limits onto one bench gives the bench-by-bench check.
6. Validity  - a surface is rejected, as in Slide2, if it has fewer than two intersections with
               the slope, passes below the external boundary (dump base), leaves through the right
               boundary, has exit / entry outside the limits, or has driving moment ~ 0.
7. Result    - the global minimum (lowest FOS) surface, plus all valid surfaces so the lowest ones
               can be drawn colour-coded, as Slide2 does with "Display all surfaces".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .lem import DumpGeometry, BishopResult, bench_points


# ------------------------------------------------------------------ settings
@dataclass
class Slide2Settings:
    method: str = "auto_refine"         # "auto_refine" | "grid" | "slope"
    n_slices: int = 50
    tolerance: float = 0.005
    max_iterations: int = 75
    # grid search
    grid_intervals: tuple = (20, 20)
    radius_increment: int = 10
    # slope search
    n_surfaces: int = 5000
    # auto refine search
    divisions: int = 20
    circles_per_division: int = 10
    iterations: int = 10
    keep_fraction: float = 0.5
    # surface filter
    min_depth: float = 0.0              # m, Slide2 "minimum depth" filter (0 = off)
    seed: int = 1


@dataclass
class Slide2Result(BishopResult):
    x_entry: float = 0.0
    n_valid: int = 0
    method: str = ""
    limits: tuple = ()
    # every valid surface (for "display all surfaces")
    all_xc: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    all_yc: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    all_r: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    all_fos: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    grid: tuple | None = None           # (x0, x1, y0, y1) of the centre grid, grid search only
    edge_warning: bool = False          # minimum found on the edge of the grid

    def lowest(self, k=50):
        o = np.argsort(self.all_fos)[:k]
        return self.all_xc[o], self.all_yc[o], self.all_r[o], self.all_fos[o]


# ------------------------------------------------------------------ model (external boundary)
class Slide2Model:
    """External boundary exactly as drawn in Slide2 / in the app: toe at (0, 0), slope, flat crest
    up to the right boundary x_r, vertical right boundary, base y = 0."""

    def __init__(self, geom: DumpGeometry):
        self.geom = geom
        L, H = geom.horizontal_extent, geom.height
        self.L, self.H = L, H
        self.x_right = L + max(0.25 * L, 30.0)
        xs, ys = geom.surface()
        m = (xs >= 0) & (xs <= L)
        self.sx = np.r_[0.0, xs[m], L, self.x_right]
        self.sy = np.r_[0.0, ys[m], H, H]
        keep = np.r_[True, np.diff(self.sx) > 1e-9]          # drop duplicate vertices
        self.sx, self.sy = self.sx[keep], self.sy[keep]
        seg = np.hypot(np.diff(self.sx), np.diff(self.sy))
        self.s_cum = np.r_[0.0, np.cumsum(seg)]               # arc length along the slope surface

    def yg(self, x):
        return np.interp(x, self.sx, self.sy)

    def x_at_s(self, s):
        return np.interp(s, self.s_cum, self.sx)

    def s_at_x(self, x):
        return np.interp(x, self.sx, self.s_cum)

    def point_at_s(self, s):
        return np.interp(s, self.s_cum, self.sx), np.interp(s, self.s_cum, self.sy)

    # -------------------------------------------------------------- circle / slope intersections
    def intersections(self, xc, yc, R):
        """Exit (left) and entry (right) x of each circle's LOWER arc with the slope surface, and the
        number of intersections."""
        P0x, P0y = self.sx[:-1][None, :], self.sy[:-1][None, :]
        dx, dy = np.diff(self.sx)[None, :], np.diff(self.sy)[None, :]
        fx, fy = P0x - xc[:, None], P0y - yc[:, None]
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        cc = fx * fx + fy * fy - (R[:, None]) ** 2
        disc = b * b - 4 * a * cc
        ok = disc >= 0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        xs_all = []
        for sgn in (-1.0, 1.0):
            t = (-b + sgn * sq) / (2 * a)
            px, py = P0x + t * dx, P0y + t * dy
            good = ok & (t >= -1e-9) & (t <= 1 + 1e-9) & (py <= yc[:, None] + 1e-9)
            xs_all.append(np.where(good, px, np.nan))
        X = np.concatenate(xs_all, axis=1)
        X = np.sort(X, axis=1)                                  # NaNs go last
        n = np.sum(np.isfinite(X), axis=1)
        x_exit = np.nanmin(np.where(np.isfinite(X), X, np.inf), axis=1)
        x_entry = np.nanmax(np.where(np.isfinite(X), X, -np.inf), axis=1)
        return x_exit, x_entry, n


# ------------------------------------------------------------------ Bishop, Slide2 settings
def bishop_slices(model: Slide2Model, xc, yc, R, x_exit, x_entry, c, phi, gamma, ru,
                  n_slices=50, tol=0.005, max_iter=75):
    """Bishop simplified FOS of many circles, sliding mass between exit and entry split into
    `n_slices` vertical slices of equal width. Slice weight by Simpson's rule on the slice height,
    slice base = chord of the arc. Iteration stops when |F_new - F| < tol (Slide2 default 0.005)."""
    M = xc.size
    tanphi = np.tan(np.radians(phi))
    e = np.linspace(0.0, 1.0, n_slices + 1)
    XE = x_exit[:, None] + (x_entry - x_exit)[:, None] * e[None, :]     # slice edges
    b = ((x_entry - x_exit) / n_slices)[:, None]
    arc = lambda X: yc[:, None] - np.sqrt(np.clip(R[:, None] ** 2 - (X - xc[:, None]) ** 2, 0, None))
    ya = arc(XE)
    XM = 0.5 * (XE[:, 1:] + XE[:, :-1])
    hE = np.clip(model.yg(XE) - ya, 0, None)
    hM = np.clip(model.yg(XM) - arc(XM), 0, None)
    W = gamma * b * (hE[:, :-1] + 4 * hM + hE[:, 1:]) / 6.0
    alpha = np.arctan2(ya[:, 1:] - ya[:, :-1], b)          # chord inclination (+ = rising to the crown)
    u_b = ru * gamma * hM * b
    drive = np.sum(W * np.sin(alpha), axis=1)
    num_const = c * b + np.clip(W - u_b, 0, None) * tanphi
    ok = drive > 1e-3
    cos_a, tan_a = np.cos(alpha), np.tan(alpha)
    F = np.ones(M)
    conv = np.zeros(M, bool)
    for _ in range(max_iter):
        m_alpha = np.clip(cos_a * (1.0 + tan_a * tanphi / F[:, None]), 0.2, None)
        F_new = np.sum(num_const / m_alpha, axis=1) / np.where(ok, drive, 1.0)
        conv = np.abs(F_new - F) < tol
        F = np.where(ok, F_new, F)
        if np.all(conv | ~ok):
            break
    F = np.where(ok & conv & (F > 0), F, np.inf)
    depth = np.max(np.clip(model.yg(XM) - arc(XM), 0, None), axis=1)
    return F, depth


# ------------------------------------------------------------------ validity + evaluation
def _evaluate(model, xc, yc, R, lim, c, phi, gamma, ru, st: Slide2Settings):
    (ex0, ex1), (en0, en1) = lim
    xe, xn, n = model.intersections(xc, yc, R)
    valid = (n >= 2) & np.isfinite(xe) & np.isfinite(xn) & (xn - xe > 0.5)
    valid &= (xe >= ex0 - 1e-6) & (xe <= ex1 + 1e-6) & (xn >= en0 - 1e-6) & (xn <= en1 + 1e-6)
    # may not pass below the external boundary (dump base y = 0)
    low = np.where((xc > xe) & (xc < xn), yc - R, np.inf)
    valid &= low >= -1e-6
    F = np.full(xc.size, np.inf)
    if valid.any():
        i = np.where(valid)[0]
        Fi, d = bishop_slices(model, xc[i], yc[i], R[i], xe[i], xn[i], c, phi, gamma, ru,
                              st.n_slices, st.tolerance, st.max_iterations)
        if st.min_depth > 0:
            Fi = np.where(d >= st.min_depth, Fi, np.inf)
        F[i] = Fi
    return F, xe, xn


def _circle_through(x1, y1, x2, y2, theta):
    """Circle through two slope points with the arc's centre ABOVE the chord; theta = half the
    central angle (rad). Returns xc, yc, R."""
    mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    d = np.hypot(x2 - x1, y2 - y1)
    R = d / (2 * np.sin(theta))
    k = np.sqrt(np.clip(R * R - (d / 2) ** 2, 0, None))
    nx, ny = -(y2 - y1) / d, (x2 - x1) / d                    # unit normal pointing up-left
    return mx + nx * k, my + ny * k, R


# ------------------------------------------------------------------ the three search methods
def _grid_search(model, lim, c, phi, gamma, ru, st, grid=None):
    (ex0, ex1), (en0, en1) = lim
    sx0, sx1 = ex0, en1
    span = max(sx1 - sx0, 1.0)
    y_ref = model.yg(np.array([en1]))[0]
    if grid is None:                                          # Slide2 auto grid, above the slope
        grid = (sx0 - 0.25 * span, sx0 + 0.9 * span,
                model.yg(np.array([ex0]))[0] + 0.2 * model.H + 1.0, y_ref + 1.6 * span)
    nx, ny = st.grid_intervals
    GX, GY = np.meshgrid(np.linspace(grid[0], grid[1], nx + 1), np.linspace(grid[2], grid[3], ny + 1),
                         indexing="ij")
    GX, GY = GX.ravel(), GY.ravel()
    # radius limits at each centre from the slope limits
    ss = np.linspace(model.s_at_x(ex0), model.s_at_x(en1), 300)
    px, py = model.point_at_s(ss)
    dist = np.hypot(px[None, :] - GX[:, None], py[None, :] - GY[:, None])
    r_min = dist.min(axis=1)
    r_max = dist.max(axis=1)                                  # circles outside the limits are rejected later
    k = np.arange(1, st.radius_increment + 1) / st.radius_increment
    R = (r_min[:, None] + (r_max - r_min)[:, None] * k[None, :])
    XC = np.repeat(GX, k.size); YC = np.repeat(GY, k.size); R = R.ravel()
    F, xe, xn = _evaluate(model, XC, YC, R, lim, c, phi, gamma, ru, st)
    # Slide2 warns when the minimum lies on the grid edge: extend the grid once in that direction
    edge = False
    if np.isfinite(F).any():
        i = int(np.argmin(F))
        gx = (XC[i] - grid[0]) / (grid[1] - grid[0]); gy = (YC[i] - grid[2]) / (grid[3] - grid[2])
        edge = min(gx, 1 - gx, gy, 1 - gy) < 1e-6
    return XC, YC, R, F, xe, xn, grid, edge


def _slope_search(model, lim, c, phi, gamma, ru, st):
    (ex0, ex1), (en0, en1) = lim
    rng = np.random.default_rng(st.seed)
    n = st.n_surfaces
    s1 = rng.uniform(model.s_at_x(ex0), model.s_at_x(ex1), n)
    s2 = rng.uniform(model.s_at_x(en0), model.s_at_x(en1), n)
    a, bb = np.minimum(s1, s2), np.maximum(s1, s2)
    x1, y1 = model.point_at_s(a); x2, y2 = model.point_at_s(bb)
    theta = np.radians(rng.uniform(5.0, 85.0, n))
    XC, YC, R = _circle_through(x1, y1, x2, y2, theta)
    F, xe, xn = _evaluate(model, XC, YC, R, lim, c, phi, gamma, ru, st)
    return XC, YC, R, F, xe, xn


def _auto_refine(model, lim, c, phi, gamma, ru, st):
    (ex0, ex1), (en0, en1) = lim
    rng = np.random.default_rng(st.seed)
    # the slope section searched: from the lowest exit limit to the highest entry limit
    regions = [(model.s_at_x(ex0), model.s_at_x(en1))]
    allc = []
    for it in range(st.iterations):
        tot = sum(b - a for a, b in regions)
        edges = []                                           # divisions of equal length over the kept regions
        for a, b in regions:
            k = max(1, int(round(st.divisions * (b - a) / tot)))
            e = np.linspace(a, b, k + 1)
            edges += list(zip(e[:-1], e[1:]))
        D = len(edges)
        lo = np.array(edges)[:, 0]; hi = np.array(edges)[:, 1]
        # only pairs whose first division can hold the exit and second division the entry
        # (when two sets of slope limits are used, e.g. bench-by-bench)
        se0, se1, sn0, sn1 = (model.s_at_x(ex0), model.s_at_x(ex1), model.s_at_x(en0), model.s_at_x(en1))
        I, J = np.triu_indices(D, k=0)
        okp = (hi[I] > se0) & (lo[I] < se1) & (hi[J] > sn0) & (lo[J] < sn1)
        I, J = I[okp], J[okp]
        I = np.repeat(I, st.circles_per_division); J = np.repeat(J, st.circles_per_division)
        s1 = rng.uniform(np.maximum(lo[I], se0), np.minimum(hi[I], se1))
        s2 = rng.uniform(np.maximum(lo[J], sn0), np.minimum(hi[J], sn1))
        s1, s2 = np.minimum(s1, s2), np.maximum(s1, s2)
        x1, y1 = model.point_at_s(s1); x2, y2 = model.point_at_s(s2)
        theta = np.radians(rng.uniform(5.0, 85.0, I.size))
        XC, YC, R = _circle_through(x1, y1, x2, y2, theta)
        F, xe, xn = _evaluate(model, XC, YC, R, lim, c, phi, gamma, ru, st)
        allc.append((XC, YC, R, F, xe, xn))
        # lowest FOS touching each division -> keep the best 50 % of divisions for the next iteration
        dmin = np.full(D, np.inf)
        np.minimum.at(dmin, I, F); np.minimum.at(dmin, J, F)
        if not np.isfinite(dmin).any():
            continue
        keep = np.argsort(dmin)[:max(2, int(np.ceil(st.keep_fraction * D)))]
        keep = np.sort(keep)
        regions = []
        for k in keep:                                       # merge adjacent kept divisions
            a, b = edges[k]
            if regions and abs(regions[-1][1] - a) < 1e-9:
                regions[-1] = (regions[-1][0], b)
            else:
                regions.append((a, b))
    cat = [np.concatenate([t[q] for t in allc]) for q in range(6)]
    return tuple(cat)


# ------------------------------------------------------------------ public API
def default_limits(geom: DumpGeometry):
    m = Slide2Model(geom)
    return ((0.0, m.x_right), (0.0, m.x_right))


def bench_limits(geom: DumpGeometry, k: int):
    """Slope limits for bench k (0 = bottom): exit on the bench face or on the half bench width in
    front of its toe; entry anywhere above the bench toe (the circle may cut the benches above)."""
    m = Slide2Model(geom)
    xt, yt, xcr, ycr, h = bench_points(geom)[k]
    lo = 0.0 if k == 0 else xt - 0.5 * geom.berm_width
    return ((lo, xcr), (xt, m.x_right))


def search(geom: DumpGeometry, c, phi, gamma, ru=0.0, settings: Slide2Settings | None = None,
           limits=None, grid=None) -> Slide2Result:
    """Run one Slide2-style search and return the global minimum surface plus all valid surfaces."""
    st = settings or Slide2Settings()
    model = Slide2Model(geom)
    lim = limits or default_limits(geom)
    edge = False
    if st.method == "grid":
        XC, YC, R, F, xe, xn, grid, edge = _grid_search(model, lim, c, phi, gamma, ru, st, grid)
        if edge:                                              # extend the grid once, like a user would
            g0 = grid; w, h = g0[1] - g0[0], g0[3] - g0[2]
            grid2 = (g0[0] - 0.5 * w, g0[1] + 0.5 * w, max(g0[2] - 0.5 * h, 1.0), g0[3] + 0.5 * h)
            XC2, YC2, R2, F2, xe2, xn2, grid, edge = _grid_search(model, lim, c, phi, gamma, ru, st, grid2)
            XC, YC, R, F, xe, xn = (np.r_[XC, XC2], np.r_[YC, YC2], np.r_[R, R2], np.r_[F, F2],
                                    np.r_[xe, xe2], np.r_[xn, xn2])
    elif st.method == "slope":
        XC, YC, R, F, xe, xn = _slope_search(model, lim, c, phi, gamma, ru, st)
    else:
        XC, YC, R, F, xe, xn = _auto_refine(model, lim, c, phi, gamma, ru, st)
    v = np.isfinite(F)
    if not v.any():
        return Slide2Result(float("inf"), 0, 0, 0, 0, int(F.size), method=st.method, limits=lim)
    i = int(np.argmin(np.where(v, F, np.inf)))
    return Slide2Result(float(F[i]), float(XC[i]), float(YC[i]), float(R[i]), float(xe[i]), int(F.size),
                        x_entry=float(xn[i]), n_valid=int(v.sum()), method=st.method, limits=lim,
                        all_xc=XC[v], all_yc=YC[v], all_r=R[v], all_fos=F[v], grid=grid, edge_warning=edge)


def bench_by_bench(geom: DumpGeometry, c, phi, gamma, ru=0.0, settings: Slide2Settings | None = None):
    """Slide2 bench-by-bench check: one search per bench with the slope limits moved onto that bench."""
    return [search(geom, c, phi, gamma, ru, settings, bench_limits(geom, k)) for k in range(geom.n_decks)]


def single_surface(geom: DumpGeometry, c, phi, gamma, ru, xc, yc, R, settings=None):
    """FOS of one given circle (Slide2 'single surface' / to re-check a circle from Slide2)."""
    st = settings or Slide2Settings()
    model = Slide2Model(geom)
    F, xe, xn = _evaluate(model, np.array([xc], float), np.array([yc], float), np.array([R], float),
                          default_limits(geom), c, phi, gamma, ru, st)
    return float(F[0]), float(xe[0]), float(xn[0])


METHOD_NAMES = {"auto_refine": "Auto refine search", "grid": "Grid search", "slope": "Slope search"}
