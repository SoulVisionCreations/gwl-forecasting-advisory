"""InferenceConfig — paths + knobs for an inference run.

The MODEL/DATA hyperparameters (horizon, gap_days, lookback, num_timesteps,
feature set, composite_period) are NOT set here — they are LOADED from `run_dir`
at engine init, so inference uses the EXACT training config (train/inference
parity by construction). Only deployment-level paths/knobs live here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceConfig:
    run_dir: str                                  # training run: best_model.pt + data/scalers.pkl + data/<config>
    composite_cache_dir: Optional[str] = None     # HLS composite cache (default: DataConfig.composite_dir)
    station_registry_csv: Optional[str] = None    # gwl_stations.csv (default: ckpt config station_index_csv)
    k_neighbours: int = 10
    min_neighbours: int = 1            # < this many survivors → error: insufficient_neighbours
    device: str = "cpu"

    # ── numeric source ──
    local_csv: Optional[str] = None               # if set, source GWL+dynamic+static from this training CSV
                                                  # (parity, reliable) instead of live GEE+WRIS
    station_stats_pkl: Optional[str] = None       # precomputed StationStats (small); else built from
                                                  # run_dir/data/train_samples.pkl (1.5 GB)
    gwl_source: str = "wris"                       # GWL lookback source: "wris" (default) | "nwdp".
                                                  # "nwdp" swaps ONLY the GWL slice (GEE still does
                                                  # dynamic/static/composite); WRIS-down stopgap.
    nwdp_index_path: Optional[str] = None         # nwdp_station_index.pkl (default: run_dir/data/nwdp_station_index.pkl)

    # ── forecast-window source ──
    use_openmeteo: bool = True                    # try Open-Meteo seasonal forecast for rain+temp
    climatology_years: int = 1                    # climatology fallback window (1 = previous year's season)

    # ── live-fetch + interpolation knobs ──
    gee_project: Optional[str] = None             # GEE project (default: vendored fetcher's GEE_PROJECT)
    min_lookback_years: int = 12                  # numeric-history window floor
    idw_power: float = 2.0                        # IDW exponent
    pred_clamp_pct: Optional[float] = None        # |delta|<=pct*|current|; None→ckpt config→horizon default
    kriging_variograms_path: Optional[str] = None  # state_variograms.pkl; enables kriging alongside IDW
                                                   # (default: run_dir/data/state_variograms.pkl if present)

    # ── Prithvi path overrides (default: paths baked into the checkpoint config) ──
    prithvi_model_dir: Optional[str] = None
    station_index_csv: Optional[str] = None
