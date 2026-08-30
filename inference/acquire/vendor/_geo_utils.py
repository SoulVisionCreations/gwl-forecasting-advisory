"""Minimal geo helpers (haversine + KNN) for the vendored data_fetcher.

data_fetcher imports only compute_distances + find_k_nearest from inference_idw;
extracting them here avoids pulling kriging/build_variogram/train_mlp/torch. The
full IDW + kriging is handled separately in interpolate.py (build-step d).
"""
import numpy as np

_EARTH_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_KM * np.arcsin(np.sqrt(a))


def compute_distances(farmer_lat, farmer_lon, stations):
    n = len(stations)
    s_lats = np.array([s["lat"] for s in stations])
    s_lons = np.array([s["lon"] for s in stations])
    return haversine_km(np.full(n, farmer_lat), np.full(n, farmer_lon), s_lats, s_lons)


def find_k_nearest(distances, k=20):
    k = min(k, len(distances))
    nn = np.argsort(distances)[:k]
    return nn, distances[nn]
