"""Data contracts that flow between inference stages."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class InferenceError(Exception):
    """A request-level failure with a stable code + human-readable message.

    Codes: bad_request, station_not_found, no_neighbours,
    data_source_unavailable, insufficient_neighbours, internal_error.
    """

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass
class Neighbour:
    """One of the K nearest known stations to the query point."""
    station_code: str
    safe_id: str
    lat: float
    lon: float
    distance_km: float


@dataclass
class AcquiredStation:
    """Everything fetched LIVE for one neighbour. Inputs are <= current_date."""
    neighbour: Neighbour
    gwl: "pd.DataFrame"                 # GWL history over the lookback
    dynamic: "pd.DataFrame"            # rain/temp (+ any dynamic) over the lookback
    static: dict                       # elevation, stream_order, lithology, aquifer, ...
    composite_path: Optional[str]      # (safe_id, year-1, period) HLS tile, or None -> zero_idx
    data_age_days: Optional[int] = None  # age of the last REAL GWL reading before forward-fill (staleness)
    from_fallback: bool = False        # GWL filled from the WRIS backup (not the CSV) -> caps confidence


@dataclass
class StationPrediction:
    """Model output at one neighbour (+ optional ground truth when the date is historical)."""
    station_code: str
    lat: float
    lon: float
    distance_km: float
    current_date: str
    target_date: str
    current_gwl: float
    predicted_delta: float
    predicted_gwl: float
    trend: str
    trend_prob: float
    data_age_days: Optional[int] = None  # age of this well's last REAL reading (staleness) -> analog swap
    # ── validation only (historical date + ground truth available) ──
    actual_gwl: Optional[float] = None
    actual_delta: Optional[float] = None
    error: Optional[float] = None       # predicted_delta - actual_delta


@dataclass
class InterpolationResult:
    """Spatially-interpolated delta at the query point (from neighbour deltas)."""
    idw_delta: Optional[float]
    kriging_delta: Optional[float]
    kriging_note: str
    nn_pred: list                       # top-K [{lat, lon, delta_gwl, dist}]
    idw_current: Optional[float] = None  # IDW of neighbour CURRENT gwl (production absolute estimate)
    outliers: Optional[list] = None      # neighbours dropped as magnitude outliers [{station_code, change_m, reason}]
