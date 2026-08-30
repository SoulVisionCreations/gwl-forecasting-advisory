"""StationStats — per-station derived features, reconstructed for inference.

Training (`data_preparation.prepare_all_data`) computes 6 derived per-station
static features + `gwl_anomaly` from TRAIN data and bakes them into every saved
sample. For parity, inference reconstructs the per-station lookup straight from
the run's `train_samples.pkl` — the EXACT numbers the model trained on — so there
is no recompute drift and NO training-side change. Mirrors the training
station -> state -> global fallback chain (data_preparation.py ~L2511-2527).
"""
from __future__ import annotations


class StationStats:

    def __init__(self, by_station: dict, by_state: dict, global_feats: dict, month_mean: dict):
        self._by_station = by_station       # station_code -> {6 derived feats}
        self._by_state = by_state           # state -> {6 derived feats} (mean of members)
        self._global = global_feats         # {6 derived feats} (global mean)
        self._month_mean = month_mean       # (station_code, month) -> mean GWL (for gwl_anomaly)

    @classmethod
    def from_train_samples(cls, train_samples_pkl: str) -> "StationStats":
        import pickle
        from collections import defaultdict
        import numpy as np
        from gwlcore.data_preparation import DERIVED_STATION_FEATURES as _DERIVED  # shared constant

        with open(train_samples_pkl, "rb") as f:
            samples = pickle.load(f)

        by_station: dict = {}
        station_state: dict = {}
        month_acc: dict = defaultdict(list)
        for s in samples:
            code = s["station_code"]
            if code not in by_station:
                by_station[code] = {k: float(s.get(k, 0.0)) for k in _DERIVED}
                station_state[code] = s.get("state", "unknown")
            # gwl_anomaly = current_gwl - station_month_mean  =>  month_mean = current_gwl - gwl_anomaly
            month = s["current_date"].month
            month_acc[(code, month)].append(float(s["current_gwl"]) - float(s.get("gwl_anomaly", 0.0)))

        month_mean = {k: float(np.mean(v)) for k, v in month_acc.items()}

        # state-level fallback = mean of member stations' features
        state_acc: dict = defaultdict(list)
        for code, feats in by_station.items():
            state_acc[station_state[code]].append(feats)
        by_state = {
            st: {k: float(np.mean([f[k] for f in lst])) for k in _DERIVED}
            for st, lst in state_acc.items()
        }
        global_feats = {
            k: float(np.mean([f[k] for f in by_station.values()] or [0.0])) for k in _DERIVED
        }
        return cls(by_station, by_state, global_feats, month_mean)

    def save(self, path: str) -> None:
        """Serialize the small per-station tables (a few MB) so inference/production
        need not load the 1.5 GB train_samples.pkl at startup."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(
                {"by_station": self._by_station, "by_state": self._by_state,
                 "global": self._global, "month_mean": self._month_mean}, f)

    @classmethod
    def load(cls, path: str) -> "StationStats":
        import pickle
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["by_station"], d["by_state"], d["global"], d["month_mean"])

    def fill(self, sample: dict) -> None:
        """Populate the 6 derived features + gwl_anomaly on an inference sample, using
        the SAME canonical helper training uses (data_preparation.fill_derived_static_
        features) — the fallback chain + anomaly formula are no longer duplicated."""
        from gwlcore.data_preparation import fill_derived_static_features
        fill_derived_static_features(
            sample, self._by_station, self._by_state, self._global, self._month_mean
        )
