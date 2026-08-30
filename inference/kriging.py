"""kriging — ordinary kriging as an alternative to IDW for the FINAL interpolation step.

This module is completely independent of the prediction flow. It consumes only the
same three inputs IDW uses — neighbour (lat, lon), neighbour predicted delta, and the
query (lat, lon) — and produces a single interpolated delta at the query point. It is
NEVER involved in fetching, sample building, scaling, or model inference.

The variogram math (haversine, the 5 variogram models, apply_variogram, kriging_predict)
is a faithful inference-only port of Aditya's build_variogram.py — the SCALE/curve-fit
build path (scipy) lives there and is not needed here; we only consume the per-state
variograms it produced (slimmed into state_variograms.pkl by build_variograms.py).

`StateVariograms` loads that slim artifact and picks the right per-state variogram for a
query by a nearest-labelled-station lookup, then solves the ordinary-kriging system.
Any failure (no variogram for the region, ill-conditioned system, out-of-range result)
returns None with a note so the caller falls back to IDW.
"""
from __future__ import annotations

import pickle
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Distance (haversine, km) — ported verbatim from build_variogram.py
# ═══════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km. Works on scalars or arrays."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pairwise_distances(lats, lons):
    """Full N×N pairwise haversine distance matrix in km."""
    lats_r = np.radians(lats)
    lons_r = np.radians(lons)
    dlat = lats_r[:, None] - lats_r[None, :]
    dlon = lons_r[:, None] - lons_r[None, :]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lats_r[:, None]) * np.cos(lats_r[None, :])
         * np.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ═══════════════════════════════════════════════════════════════════════
#  Variogram models — ported verbatim from build_variogram.py
# ═══════════════════════════════════════════════════════════════════════

def spherical_model(h, nugget, sill, range_km):
    h = np.asarray(h, dtype=float)
    r = np.clip(h / max(range_km, 1e-6), 0, None)
    return np.where(h < range_km,
                    nugget + sill * (1.5 * r - 0.5 * r ** 3),
                    nugget + sill)


def gaussian_model(h, nugget, sill, range_km):
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1 - np.exp(-(h / max(range_km, 1e-6)) ** 2))


def exponential_model(h, nugget, sill, range_km):
    h = np.asarray(h, dtype=float)
    return nugget + sill * (1 - np.exp(-h / max(range_km, 1e-6)))


def linear_model(h, nugget, slope):
    h = np.asarray(h, dtype=float)
    return nugget + slope * h


def power_model(h, nugget, scale, exponent):
    h = np.asarray(h, dtype=float)
    return nugget + scale * np.power(h + 1e-10, exponent)


VARIOGRAM_MODELS = {
    "spherical":   spherical_model,
    "gaussian":    gaussian_model,
    "exponential": exponential_model,
    "linear":      linear_model,
    "power":       power_model,
}


def apply_variogram(h, model_name, params):
    """Apply a fitted variogram model to distance(s)."""
    func = VARIOGRAM_MODELS[model_name]
    if model_name in ("spherical", "gaussian", "exponential"):
        return func(h, params["nugget"], params["sill"], params["range_km"])
    elif model_name == "linear":
        return func(h, params["nugget"], params["slope"])
    elif model_name == "power":
        return func(h, params["nugget"], params["scale"], params["exponent"])
    raise ValueError(f"unknown variogram model: {model_name}")


# ═══════════════════════════════════════════════════════════════════════
#  Ordinary kriging — ported verbatim from build_variogram.kriging_predict
# ═══════════════════════════════════════════════════════════════════════

def kriging_predict(nb_lats, nb_lons, nb_values, target_lat, target_lon,
                    model_name, params):
    """Ordinary Kriging prediction using K neighbours.

    Builds the augmented kriging system:
      [Γ   1] [w]   [γ₀]
      [1ᵀ  0] [μ] = [ 1]

    Includes diagonal regularization to prevent numerical instability when stations
    are clustered (near-identical gamma rows). Returns the predicted value, or None
    on failure / unreasonable result (caller falls back to IDW).
    """
    nb_lats = np.asarray(nb_lats, dtype=float)
    nb_lons = np.asarray(nb_lons, dtype=float)
    nb_values = np.asarray(nb_values, dtype=float)
    n = len(nb_lats)
    if n < 3:
        return None

    # Station-to-station distance matrix
    dist_nn = pairwise_distances(nb_lats, nb_lons)
    # Station-to-target distances
    dist_nt = haversine_km(nb_lats, nb_lons,
                           np.full(n, target_lat), np.full(n, target_lon))

    # Gamma values
    gamma_nn = apply_variogram(dist_nn, model_name, params)
    gamma_nt = apply_variogram(dist_nt, model_name, params)

    # Diagonal regularization: add a meaningful nugget to prevent a near-singular
    # matrix when stations are clustered. max(variogram nugget, 1% of mean off-diag).
    off_diag = gamma_nn[np.triu_indices(n, k=1)]
    mean_gamma = np.mean(off_diag) if len(off_diag) > 0 else 1e-6
    nugget_reg = max(params.get("nugget", 0), mean_gamma * 0.01, 1e-10)

    # Augmented system
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = gamma_nn
    A[np.arange(n), np.arange(n)] += nugget_reg  # regularize diagonal
    A[:n, n] = 1.0
    A[n, :n] = 1.0

    b = np.zeros(n + 1)
    b[:n] = gamma_nt
    b[n] = 1.0

    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None  # still singular even with regularization

    weights = w[:n]

    # Sanity check: extreme weights indicate ill-conditioning
    if np.max(np.abs(weights)) > 10:
        return None

    pred = float(np.dot(weights, nb_values))

    # Sanity check: prediction within reasonable range of neighbour values
    val_min, val_max = np.min(nb_values), np.max(nb_values)
    val_range = max(val_max - val_min, 1e-10)
    if pred < val_min - 2 * val_range or pred > val_max + 2 * val_range:
        return None

    return pred


# ═══════════════════════════════════════════════════════════════════════
#  StateVariograms — load the slim artifact + pick the variogram for a query
# ═══════════════════════════════════════════════════════════════════════

def _safe_state(s: str) -> str:
    return str(s).lower().strip().replace(" ", "_").replace("&", "and")


class StateVariograms:
    """Per-state variograms + a nearest-labelled-station lookup to assign a query
    (or its neighbours) to a state. Built once from build_variogram's state_models
    JSONs (see build_variograms.py) and shipped as state_variograms.pkl.

    Self-contained: needs only the variograms and the (lat, lon, state) labels — it
    does NOT depend on the fetch pipeline, registry, or any prediction artifact.
    """

    def __init__(self, variograms: dict, lat, lon, state):
        # variograms: safe_state -> {"model": str, "params": dict}  (kriging states only)
        self.variograms = {_safe_state(k): v for k, v in variograms.items()}
        self._lat = np.asarray(lat, dtype=float)
        self._lon = np.asarray(lon, dtype=float)
        self._state = [_safe_state(s) for s in state]

    # ---- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"variograms": self.variograms,
                 "lat": self._lat, "lon": self._lon, "state": self._state},
                f,
            )

    @classmethod
    def load(cls, path: str) -> "StateVariograms":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["variograms"], d["lat"], d["lon"], d["state"])

    # ---- state selection ---------------------------------------------------
    def state_for_points(self, lats, lons) -> Optional[str]:
        """Majority state across the given points (each → nearest labelled station).
        Mirrors the old workflow's majority-of-neighbours state selection."""
        if not len(self._lat):
            return None
        lats = np.atleast_1d(np.asarray(lats, dtype=float))
        lons = np.atleast_1d(np.asarray(lons, dtype=float))
        states = []
        for la, lo in zip(lats, lons):
            d = haversine_km(self._lat, self._lon,
                             np.full(self._lat.shape, la), np.full(self._lon.shape, lo))
            states.append(self._state[int(np.argmin(d))])
        return max(set(states), key=states.count)

    # ---- prediction --------------------------------------------------------
    def predict(self, query_lat, query_lon, nb_lats, nb_lons, nb_deltas):
        """Krige the neighbour deltas to the query point.

        Returns (kriging_delta or None, note). None whenever kriging is unavailable
        or unreliable for this query — the caller then uses IDW.
        """
        nb_lats = np.atleast_1d(np.asarray(nb_lats, dtype=float))
        nb_lons = np.atleast_1d(np.asarray(nb_lons, dtype=float))
        nb_deltas = np.atleast_1d(np.asarray(nb_deltas, dtype=float))

        if len(nb_lats) < 3:
            return None, f"kriging needs >= 3 neighbours (have {len(nb_lats)}); using IDW"

        state = self.state_for_points(nb_lats, nb_lons)
        if state is None:
            return None, "no labelled stations for state lookup; using IDW"

        vg = self.variograms.get(state)
        if vg is None:
            return None, (f"no kriging variogram for state '{state}' "
                          f"(small-state average fallback); using IDW")

        pred = kriging_predict(nb_lats, nb_lons, nb_deltas, query_lat, query_lon,
                               vg["model"], vg["params"])
        if pred is None:
            return None, (f"kriging ill-conditioned for state '{state}' "
                          f"(variogram={vg['model']}); using IDW")
        return float(pred), f"kriging ok (variogram={vg['model']}, state={state})"
