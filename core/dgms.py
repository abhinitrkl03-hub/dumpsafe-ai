"""
DGMS / Coal Mines Regulations 2017 checks for spoil banks (OB dumps), Regulation 106:
  * slope of a spoil bank set by the natural angle of repose, and in any case
    not steeper than 37.5 deg (steeper only if a scientific study recommends it and
    the Regional Inspector permits it by order);
  * a spoil bank higher than 30 m must be benched so that no bench exceeds 30 m;
  * overall slope not steeper than 1 vertical : 1.5 horizontal (33.69 deg);
  * toe at least 100 m from mine openings, railways, public roads, buildings.
Berm width is not fixed by the regulation; it follows from the 1V:1.5H overall-slope rule.
"""
from __future__ import annotations
import numpy as np

MAX_DECK_ANGLE = 37.5
MAX_BENCH_HEIGHT = 30.0
MAX_OVERALL = float(np.degrees(np.arctan(1 / 1.5)))   # 33.69 deg


def min_berm(n_decks, deck_height, deck_angle):
    """Smallest berm width that keeps the overall slope within 1V:1.5H."""
    if n_decks <= 1:
        return 0.0
    H = n_decks * deck_height
    need = 1.5 * H - n_decks * deck_height / np.tan(np.radians(deck_angle))
    return max(0.0, need / (n_decks - 1))


def check(n_decks, deck_height, deck_angle, berm_width, overall_angle=None):
    """Returns (compliant: bool, list of human-readable violations)."""
    v = []
    H = n_decks * deck_height
    if deck_angle > MAX_DECK_ANGLE + 1e-6:
        v.append(f"Deck angle {deck_angle:.1f} deg exceeds 37.5 deg "
                 "(allowed only with a scientific-study recommendation and DGMS order)")
    if H > MAX_BENCH_HEIGHT + 1e-6 and deck_height > MAX_BENCH_HEIGHT + 1e-6:
        v.append(f"Dump is {H:.0f} m high but a bench is {deck_height:.0f} m; "
                 "benches must not exceed 30 m")
    if overall_angle is None:
        horiz = n_decks * deck_height / np.tan(np.radians(deck_angle)) + (n_decks - 1) * berm_width
        overall_angle = float(np.degrees(np.arctan(H / horiz)))
    if overall_angle > MAX_OVERALL + 1e-6:
        v.append(f"Overall slope {overall_angle:.1f} deg is steeper than 1V:1.5H (33.7 deg)")
    return len(v) == 0, v
