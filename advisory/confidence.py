"""confidence.py — HIGH/MEDIUM/LOW from the DATA we actually have, capped by the WEAKEST.

TWO factors (weakest-link): how CLOSE the nearest known well is, and how much HISTORY backs
the normals. We deliberately do NOT score neighbour AGREEMENT: unlike NDVI pixels within one
village, GWL neighbour wells can be tens of km apart across different aquifers, so they can
legitimately have very different change values — that is spatial heterogeneity (already
reflected by distance), not forecast unreliability. A true model prediction interval would be
the right third factor; the model emits none yet, so we do NOT fake one from spatial scatter.

  - nearest_km : distance to the closest known well
  - history    : #years behind the normals + #wells with data
Neighbour spatial spread and neighbour distances are NOT surfaced in the response (they never
gated the level; a weighted-agreement coherence factor is a separate future addition). See
use_case_gwl.txt.
"""
from __future__ import annotations

_ORDER = {"good": 2, "medium": 1, "poor": 0}
_LEVEL = {2: "HIGH", 1: "MEDIUM", 0: "LOW"}


def _distance_score(nearest_km):
    if nearest_km is None:
        return "poor"
    if nearest_km <= 5:
        return "good"
    if nearest_km <= 15:
        return "medium"
    return "poor"


def _history_score(n_years, n_wells):
    n_years = n_years or 0
    n_wells = n_wells or 0
    if n_years < 4 or n_wells < 3:
        return "poor"
    if n_years >= 7 and n_wells >= 5:
        return "good"
    return "medium"


def _freshness_score(age_days):
    # how stale the anchor GWL is (days since the last real reading, forward-filled to the anchor).
    # None = no staleness reported -> fresh. NWDP normally lags ~1 quarter, so <=120d is 'good'.
    if age_days is None:
        return "good"
    if age_days <= 120:
        return "good"
    if age_days <= 270:
        return "medium"
    return "poor"


def assess(nearest_km, n_years, n_wells, freshness_days=None, verbose=False) -> dict:
    d = _distance_score(nearest_km)
    h = _history_score(n_years, n_wells)
    f = _freshness_score(freshness_days)                    # gates the level; label shown only in verbose
    level = _LEVEL[min(_ORDER[d], _ORDER[h], _ORDER[f])]   # weakest link: distance / history / freshness
    out = {"level": level, "nearest": d, "history": h,
           "nearest_km": nearest_km, "n_years": n_years, "n_wells": n_wells}
    if verbose:                                             # freshness LABEL only (data_age_days never shipped)
        out["freshness"] = f
    return out
