"""interpolate — spatially interpolate neighbour predicted-deltas to the query point.

This is the FINAL step, run after every neighbour's prediction already exists. It is
completely independent of the prediction flow: it consumes only neighbour (lat, lon),
neighbour predicted_delta, and the query (lat, lon).

IDW (inverse-distance weighting, 1/d^power) ported from Aditya's inference_idw —
pure, standalone, a KEEPER (no canonical equivalent). Distances come straight from
the StationRegistry (haversine, km) carried on each StationPrediction, so we don't
recompute them. IDW is the default and the always-available fallback.

Kriging is an OPTIONAL alternative weighting: when a per-state variogram store is
supplied (StateVariograms), we additionally solve ordinary kriging over the same
neighbour deltas and report kriging_delta alongside idw_delta. If no variogram store
is given, or kriging is unreliable for this query, kriging_delta stays None with a note
and IDW is used. Single neighbour / exact-match → IDW collapses to that neighbour's value.
"""
from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from inference.types import StationPrediction
    from inference.kriging import StateVariograms

from inference.types import InterpolationResult

_KRIGING_NOTE = (
    "kriging not enabled: no per-state variogram store supplied for this run "
    "(state_variograms.pkl); using IDW."
)

_OUTLIER_Z = 3.5          # robust (MAD) z beyond which a neighbour is an outlier candidate
_OUTLIER_ABS_M = 2.0      # ... AND it must differ from the neighbourhood median by > this (m)
_MIN_KEEP = 3             # never prune below this many surviving neighbours


def _reject_outliers(preds):
    """Flag neighbours whose predicted change is a robust outlier vs the neighbourhood.

    A single bad well can dominate IDW by MAGNITUDE even at low distance-weight (1/d^2
    down-weights distance, not magnitude). We drop a neighbour only when it is BOTH a
    relative outlier (MAD z-score > _OUTLIER_Z) AND absolutely far from the median
    (> _OUTLIER_ABS_M m) — so a tight cluster of small, plausible changes is never pruned.
    Applies to ALL wells incl. the nearest (a wild nearest is dropped too — esp. important
    when the "nearest" is far because closer wells had no data). Never prunes below
    _MIN_KEEP survivors, and is a no-op when there are < 4 neighbours or the spread is ~0.
    Median/MAD are computed once (both robust, so the outlier itself does not corrupt the fence).

    Returns (keep_mask, dropped) with dropped = [{station_code, change_m, reason}].
    """
    n = len(preds)
    keep = [True] * n
    if os.environ.get("GWL_IDW_OUTLIER_REJECT", "1").lower() in ("0", "false", "off", "no"):
        return keep, []                               # kill-switch (also used to A/B plain vs robust)
    if n < 4:
        return keep, []
    deltas = np.array([p.predicted_delta for p in preds], dtype=float)
    med = float(np.median(deltas))
    mad = float(np.median(np.abs(deltas - med)))
    if mad <= 1e-9:                                   # neighbours agree -> nothing to reject
        return keep, []
    sigma = 1.4826 * mad                              # MAD -> robust std estimate
    z = np.abs(deltas - med) / sigma
    # the nearest well is NOT exempt: a wild value is dropped even when it is the closest.
    cand = [i for i in range(n)
            if z[i] > _OUTLIER_Z and abs(deltas[i] - med) > _OUTLIER_ABS_M]
    cand.sort(key=lambda i: -z[i])                    # worst first
    dropped = []
    for i in cand:
        if sum(keep) <= _MIN_KEEP:
            break
        keep[i] = False
        dropped.append({
            "station_code": preds[i].station_code,
            "change_m": round(float(deltas[i]), 2),
            "reason": "magnitude_outlier",
        })
    return keep, dropped


def interpolate(
    lat: float,
    lon: float,
    preds: "list[StationPrediction]",
    idw_power: float = 2.0,
    top_k: int = 5,
    variograms: "Optional[StateVariograms]" = None,
) -> "InterpolationResult":
    """IDW over neighbour deltas at their (lat,lon) → delta at the query point.
    When `variograms` is supplied, also compute an ordinary-kriging delta (same
    inputs) and report it alongside; otherwise kriging_delta stays None."""
    if not preds:
        return InterpolationResult(
            idw_delta=None, kriging_delta=None,
            kriging_note="no neighbour predictions to interpolate", nn_pred=[],
            idw_current=None,
        )

    # ── robust outlier rejection FIRST: a single wild well can otherwise dominate IDW by
    # magnitude despite a low distance-weight. Guarded (see _reject_outliers) so it is a
    # no-op unless there is a clear outlier. Both IDW and kriging then run on survivors. ──
    keep_mask, outliers = _reject_outliers(preds)
    kept = [p for p, k in zip(preds, keep_mask) if k]

    dists = np.clip(np.array([p.distance_km for p in kept], dtype=float), 0.01, None)
    deltas = np.array([p.predicted_delta for p in kept], dtype=float)
    currents = np.array([p.current_gwl for p in kept], dtype=float)
    weights = 1.0 / dists ** idw_power
    weights /= weights.sum()
    idw_delta = float(np.dot(weights, deltas))
    idw_current = float(np.dot(weights, currents))   # production absolute-GWL estimate

    order = np.argsort(dists)[:top_k]
    nn_pred = [
        {
            "station_code": kept[i].station_code,
            "lat": kept[i].lat,
            "lon": kept[i].lon,
            "delta_gwl": round(float(kept[i].predicted_delta), 4),
            "dist_km": round(float(kept[i].distance_km), 3),
            "weight": round(float(weights[i]), 4),
        }
        for i in order
    ]

    # ── optional ordinary-kriging alternative (same SURVIVING neighbour deltas) ──
    kriging_delta: "Optional[float]" = None
    kriging_note = _KRIGING_NOTE
    if variograms is not None:
        nb_lats = np.array([p.lat for p in kept], dtype=float)
        nb_lons = np.array([p.lon for p in kept], dtype=float)
        kriging_delta, kriging_note = variograms.predict(
            lat, lon, nb_lats, nb_lons, deltas
        )

    return InterpolationResult(
        idw_delta=idw_delta,
        kriging_delta=kriging_delta,
        kriging_note=kriging_note,
        nn_pred=nn_pred,
        idw_current=idw_current,
        outliers=outliers,
    )
