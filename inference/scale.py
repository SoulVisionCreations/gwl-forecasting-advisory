"""Scaler — thin wrapper over canonical feature_scaler (the TRAINED scalers).

No scaling logic lives here: it forwards to the fitted LSTMFeatureScaler loaded
from the run's scalers.pkl, so inference uses the exact same standardization,
label encoders and target scaler as training. tile_idx passes through untouched
(transform_sample copies it verbatim).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class Scaler:

    def __init__(self, fs):
        self._fs = fs                       # a loaded feature_scaler.LSTMFeatureScaler

    @classmethod
    def load(cls, scalers_pkl: str) -> "Scaler":
        import sys

        from gwlcore.feature_scaler import LSTMFeatureScaler, ScalerConfig

        # scalers.pkl embeds a ScalerConfig pickled from the training script running
        # as __main__; install the shim so unpickling resolves __main__.ScalerConfig.
        sys.modules["__main__"].ScalerConfig = ScalerConfig
        return cls(LSTMFeatureScaler.load(scalers_pkl))

    def transform(self, samples: "list[dict]") -> "list[dict]":
        """Canonical per-sample transform → model-ready scaled dicts (tile_idx kept)."""
        return self._fs.transform(samples)

    def transform_one(self, sample: dict) -> dict:
        """Transform a single sample (lets the engine isolate scale-stage failures)."""
        return self._fs.transform_sample(sample)

    def inverse_delta(self, scaled_delta: "np.ndarray") -> "np.ndarray":
        """Scaled regression output → absolute Δ (metres), via the target scaler."""
        return self._fs.inverse_transform_predictions(scaled_delta)
