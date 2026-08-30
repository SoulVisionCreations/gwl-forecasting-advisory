"""reliability.py — calibrated per-query reliability of the advisory's above/below-normal call.

The numbers here are BAKED CONSTANTS from a one-time offline calibration on the shipped test sets
(champion 3m `tft_prithvi_quarterly_rerun` + `h6_release_q` 6m; see `scripts/calibrate_clarity.py`,
a dev tool — NOT shipped). NO model, NO I/O, NO external service: at runtime this is pure arithmetic
on values the advisory has already computed (`forecast - normal` and the local `band`).

What was measured: for ~59k/57k test samples, the fraction where the advisory's **above/below-normal**
call landed on the CORRECT side of the seasonal normal, binned by the *clarity ratio*
    r = |forecast_change - normal| / band
The relationship is MONOTONIC in r and consistent across both horizons:

    r < 1        regime is "normal"  -> "typical"  : no above/below claim -> no reliability %
    1 <= r < 2   "moderate"          : ~70%  (~7 in 10)   past the band, not decisively
    r >= 2       "clear"             : ~75%  (~3 in 4)    well past the band

(Calibration note: the 3-way exact-match accuracy is only ~54% and non-monotonic — the "normal"
deadzone is genuinely hard to separate — so we deliberately calibrate & surface only the 2-way
above/below side-accuracy, which is what a farmer's crop lean actually turns on. The raw *direction*
rise/fall accuracy is stronger still, 66->88% by predicted move size; see MODEL_CARD.)
"""
from __future__ import annotations

from typing import Optional

MODERATE_RATIO = 1.0    # r below this => the call is "normal" (within band) => no above/below claim
CLEAR_RATIO = 2.0       # r at/above this => a decisive "clear" call

# above/below-normal side-accuracy (%), from scripts/calibrate_clarity.py --emit-constants.
# Calibrated on: tft_prithvi_quarterly_rerun_20260629 (3m) / h6_release_q_20260711 (6m).
# 3m and 6m came out identical (the "beat-the-normal" call is about equally hard at both horizons).
# Self-normal FLOOR — the live neighbour-normal path runs ~5-8 pts higher. Regenerate on every retrain.
_RELIABILITY = {
    "3m": {"clear": 74, "moderate": 71},
    "6m": {"clear": 74, "moderate": 71},
}


def pct_in_words(pct: Optional[int]) -> Optional[str]:
    """Plain-frequency rendering. 73-77 -> 'about 3 in 4'; else 'about N in 10'.
    Works for clarity (74->3 in 4, 71->7 in 10) AND direction (82->8 in 10, 88->9 in 10)."""
    if pct is None:
        return None
    if 73 <= pct <= 77:
        return "about 3 in 4"
    return f"about {round(pct / 10)} in 10"


def direction_of(change_m: Optional[float], horizon: str = "3m") -> Optional[dict]:
    """Absolute water-table direction (vs zero) + its calibrated reliability — the STRONGEST signal.

    change_m < 0 = water RISES (recharge / gain); >= 0 = water FALLS (depletion / loss).
    reliability_pct = the test-set direction accuracy for a move of THIS size
    (DIRECTION_ACC_BY_PRED_MOVE — bigger moves are called right more often). Returns
    {"direction","reliability_pct"} or None if change_m is missing.
    """
    if change_m is None:
        return None
    m = abs(change_m)
    tbl = DIRECTION_ACC_BY_PRED_MOVE.get(horizon, DIRECTION_ACC_BY_PRED_MOVE["3m"])
    pct = (tbl["<0.5m"] if m < 0.5 else tbl["0.5-2m"] if m < 2.0
           else tbl["2-5m"] if m < 5.0 else tbl["5m+"])
    return {"direction": "rising" if change_m < 0 else "falling", "reliability_pct": pct}

# For MODEL_CARD context only (NOT used at runtime): raw direction (rise/fall) accuracy by |predicted
# change| in metres — 3m: <0.5m 66%, 0.5-2m 82%, 2-5m 88%, 5+m 87%; 6m: 67/83/89/89.
DIRECTION_ACC_BY_PRED_MOVE = {
    "3m": {"<0.5m": 66, "0.5-2m": 82, "2-5m": 88, "5m+": 87},
    "6m": {"<0.5m": 67, "0.5-2m": 83, "2-5m": 89, "5m+": 89},
}


def clarity_of(d: Optional[float], band: Optional[float], regime: Optional[str],
               horizon: str = "3m") -> Optional[dict]:
    """Decisiveness + calibrated reliability of an above/below-normal call.

    d       = forecast_change - normal (metres; the `d_a` bin_a already returns)
    band    = the local seasonal band (metres)
    regime  = "above" | "below" | "normal" (bin_a's class — keeps this aligned with the shown regime)
    Returns {"level","reliability_pct"} or None if inputs are missing.
    A "normal" regime -> {"level":"typical","reliability_pct":None} (no above/below claim).
    """
    if d is None or band is None or band <= 0 or regime is None:
        return None
    if regime == "normal":
        return {"level": "typical", "reliability_pct": None}
    r = abs(d) / band
    level = "clear" if r >= CLEAR_RATIO else "moderate"
    pct = _RELIABILITY.get(horizon, _RELIABILITY["3m"])[level]
    return {"level": level, "reliability_pct": pct}
