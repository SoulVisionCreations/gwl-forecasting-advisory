"""neighbours.py — nearest WELLS + IDW weights + their full history.

GWL analog of NDVI's sampling.py. GWL does NOT sample village cropland (plots within
~4 km share the same ~10 wells, so sampling adds nothing); it selects the nearest
monitoring WELLS and IDW-weights them (1/d^2). REUSES the forecaster's StationRegistry
(KNN) and its numeric source (full per-well GWL history).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeighbourSet:
    codes: list
    weights: dict            # code -> 1/d^2
    distances: dict          # code -> km
    series: dict             # code -> pd.Series (date-indexed depth-to-water, metres)
    nearest_km: Optional[float]
    n_with_data: int
    fallback_codes: set = field(default_factory=set)   # codes whose GWL came from the WRIS backup


def gather(registry, source, lat, lon, k, current_date, exclude=None) -> NeighbourSet:
    """K nearest wells to (lat,lon), their IDW weights, and full depth history each.

    `source` is the forecaster's numeric source (LocalCsvSource / NumericFetcher); its
    .fetch(neighbours, date) returns AcquiredStation with .gwl = date-indexed GWL history.
    """
    neighbours = registry.k_nearest(lat, lon, k, exclude=exclude)
    codes = [n.station_code for n in neighbours]
    distances = {n.station_code: float(n.distance_km) for n in neighbours}
    weights = {c: 1.0 / max(distances[c], 0.01) ** 2 for c in codes}

    acquired = source.fetch(neighbours, current_date)
    series: dict = {}
    for a in acquired:
        c = a.neighbour.station_code
        g = a.gwl
        if g is not None and len(getattr(g, "index", [])):
            s = g["gwl_value"] if ("gwl_value" in getattr(g, "columns", [])) else g.iloc[:, 0]
            series[c] = s.sort_index()

    nearest = min(distances.values()) if distances else None
    fallback_codes = {a.neighbour.station_code for a in acquired
                      if getattr(a, "from_fallback", False) and a.neighbour.station_code in series}
    return NeighbourSet(codes=codes, weights=weights, distances=distances,
                        series=series, nearest_km=nearest, n_with_data=len(series),
                        fallback_codes=fallback_codes)
