"""Data contracts for the advisory layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Normals:
    """Interpolated historical yardsticks at the query point (Stage 3b, NO model)."""
    normal_seasonal: Optional[float]        # median seasonal change over ~N years
    normal_seasonal_6m: Optional[float]
    latest_yoy: Optional[float]             # observed year just ended
    normal_yoy: Optional[float]             # median year-over-year over ~N years
    band: Optional[float]                   # 0.5 * IQR(3m seasonal changes), floor 0.3 m  -> comparison (a)
    band_6m: Optional[float]                # 0.5 * IQR(6m seasonal changes)               -> long-crop 6m gate
    band_yoy: Optional[float]               # 0.5 * IQR(yoy changes)                        -> comparison (b)
    seasonal_changes: list = field(default_factory=list)   # per-year interpolated (for band)
    neighbour_spread: Optional[float] = None               # spatial disagreement among wells (PI proxy)
    n_years_seasonal: int = 0
    n_years_yoy: int = 0
    n_wells_used: int = 0
