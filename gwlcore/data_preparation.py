"""
Data Preparation for GWL Forecasting LSTM

Implements Phase 1 from lstm_design.md:
- Load raw data from PostgreSQL
- Create horizon-aligned sequences (sampling at horizon intervals)
- Compute climatological forecasts for prediction window
- Compute historical aggregates for each timestep
- Add temporal encoding to each timestep
- Handle missing data
- Create train/val/test splits

GWL Encoding (configurable via --delta-gwl / --no-delta-gwl):
- Delta mode (default): GWL encoded as (value - current_gwl) in sequence and target.
  Removes level bias — model learns dynamics only. At inference, reconstruct:
  predicted_gwl = current_gwl + predicted_delta
- Absolute mode: Raw GWL values (original behavior)

Outlier Detection (two layers):
1. Station-level (--station-bounds / --no-station-bounds):
   Precompute IQR bounds per station from training data. Drop any sample where
   current_gwl or target_gwl falls outside bounds (bad sensor reading → bad target).
2. Per-window (--window-outliers / --no-window-outliers):
   Within each lookback window, flag GWL readings outside IQR bounds as unreliable.
   Flagged values get (delta=0, is_reliable=0) — same treatment as missing values.
   Falls back to trusting all present values if <min_outlier_points present.

Per-timestep feature vector: [gwl_encoded, is_reliable, gwl_diff, gwl_diff_reliable, sin, cos, 6 historical, cum_rain, rain_anomaly, cum_rain_anomaly] = 15 features
Forecast features: 6 (separate, for conditioning)

Reference: lstm_design.md (Phase 1: Data Preparation)
"""

import warnings
import numpy as np
import pandas as pd

# Suppress numpy nanmean warning on empty slices (expected when
# a station has no data in a particular aggregation window)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
import psycopg2
from typing import Dict, Tuple, Optional, List
from .data_util import split_by_district, split_by_date
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import os
import re
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

SAMPLE_DATA_FILE = "data_with_flag_50k.tsv"
# Training GWL dataset (data_with_flag_all.csv) — env-driven (AIKosh download -> set the path); NO
# machine-specific default. GWL_CSV_DATA, else the shared GWL_DATA_CSV, else None. Per-run overridable
# by DataConfig.csv_path (below). None here -> a clear failure if neither config nor env supplies a path.
CSV_DATA_PATH = os.environ.get("GWL_CSV_DATA") or os.environ.get("GWL_DATA_CSV") or None

# Per-feature aggregation for the historical AND forecast windows. Single source of
# truth shared with inference (SampleBuilder imports this) so the mapping can't drift.
# NDVI/SM are dropped downstream when config.drop_ndvi_sm is set.
FEATURE_AGG_FUNCTIONS = {
    "rainfall": "sum",
    "temp": "mean",
    "et": "sum",
    "runoff": "sum",
    "ndvi": "mean",
    "sm": "mean",
    "lulc": "mode",
}

# The 6 derived per-station static features baked onto every sample from TRAIN data.
DERIVED_STATION_FEATURES = (
    "mean_annual_rainfall",
    "annual_rainfall_std",
    "station_mean_gwl",
    "station_gwl_amplitude",
    "station_delta_mean",
    "station_delta_std",
)


def fill_derived_static_features(sample, by_station, by_state, global_feats, month_mean):
    """Fill the 6 derived per-station static features + gwl_anomaly on a sample via
    the station -> state -> global fallback chain. Single source of truth shared by
    training (prepare_all_data annotation) and inference (StationStats.fill), so the
    fallback logic + anomaly formula can't drift.

    by_station: {station_code -> {feat: val}}; by_state: {state -> {feat: val}};
    global_feats: {feat: val}; month_mean: {(station_code, month) -> mean GWL}.
    """
    code = sample["station_code"]
    state = sample.get("state", "unknown")
    feats = by_station.get(code) or by_state.get(state) or global_feats
    for k in DERIVED_STATION_FEATURES:
        sample[k] = feats.get(k, 0.0)
    # gwl_anomaly = current_gwl - station_month_mean (fallback: station overall mean)
    month = sample["current_date"].month
    mm = month_mean.get((code, month))
    monthly_mean = float(mm) if mm is not None else float(feats.get("station_mean_gwl", 0.0) or 0.0)
    sample["gwl_anomaly"] = float(sample["current_gwl"]) - monthly_mean


def df_gwl_readings(csv_path_or_df):
    df = (
        csv_path_or_df
        if isinstance(csv_path_or_df, pd.DataFrame)
        else pd.read_csv(csv_path_or_df)
    )
    df = df[["station_code", "date", "gwl_value", "state", "district"]]
    return df


def clean_gwl_values(df, config):
    """Canonical GWL-value cleaning — the SAME normalization
    LSTMDataPreparation.load_gwl_readings applies, factored out so EVERY inference GWL
    source (WRIS / NWDP / LocalCsv) cleans identically to training: drop NaN gwl_value,
    abs() sign-normalize (data convention is depth-to-water; the sign varies by source),
    then apply the configured hard caps [min_gwl, max_gwl]. Requires a 'gwl_value' column.
    (load_gwl_readings keeps its own inline copy with progress prints — kept byte-
    identical; this is the shared inference-facing form.)"""
    df = df[df["gwl_value"].notna()].copy()
    df["gwl_value"] = df["gwl_value"].abs()
    max_gwl = getattr(config, "max_gwl", 0) or 0
    min_gwl = getattr(config, "min_gwl", 0) or 0
    if max_gwl > 0:
        df = df[df["gwl_value"] <= max_gwl]
    if min_gwl < 0:
        df = df[df["gwl_value"] >= min_gwl]
    return df


def df_dynamic_features(csv_path_or_df):
    df = (
        csv_path_or_df
        if isinstance(csv_path_or_df, pd.DataFrame)
        else pd.read_csv(csv_path_or_df)
    )
    df = df.drop(
        columns=["rainfall", "temp", "et", "runoff", "lulc", "sm", "ndvi"],
        errors="ignore",
    )
    df = df.rename(
        columns={
            "precipitation": "rainfall",
            "temperature_2m": "temp",
            "total_evaporation_sum": "et",
            "runoff_sum": "runoff",
            "dominant_class": "lulc",
        }
    )
    df["sm"] = (df["soil_moisture_am"] + df["soil_moisture_pm"]) / 2.0
    ndvi_denom = df["sr_b5"] + df["sr_b4"]
    df["ndvi"] = np.where(ndvi_denom == 0, 0, (df["sr_b5"] - df["sr_b4"]) / ndvi_denom)
    return df


def df_static_features(csv_path_or_df):
    """
    Return static features DataFrame from CSV path or pre-loaded DataFrame.
    Columns: [station_code, elevation, well_type, aquifer_type, lithology, ...]
    """
    df = (
        csv_path_or_df
        if isinstance(csv_path_or_df, pd.DataFrame)
        else pd.read_csv(csv_path_or_df)
    )
    df = df.rename(
        columns={
            "well_aquifer_type": "aquifer_type",
            "litho_lithologic": "lithology",
        }
    )
    return df


def parse_lookback_window(value) -> int:
    """Parse a lookback-window value with optional unit suffix → days (int).

    Accepted forms:
      '7d'  → 7
      '1m'  → 30
      '30d' → 30
      '3'   → 90  (bare integer = months, ×30 for the days convention)
      3     → 90  (same)

    Months are approximated as 30 days each, matching the compute_aggregate
    days convention used throughout this module.
    """
    if value is None:
        raise ValueError("lookback_window cannot be None")
    s = str(value).strip().lower()
    if not s:
        raise ValueError("lookback_window cannot be empty")
    if s.endswith("d"):
        n = int(s[:-1])
        return n
    if s.endswith("m"):
        n = int(s[:-1])
        return n * 30
    # Bare number → months convention (multiply by 30)
    n = int(s)
    return n * 30


# ──────────────────────────────────────────────────────────────────────
# Prithvi-EO tile selection (Step 2)
#
# Each GWL sample is annotated with an integer `tile_idx` pointing at a
# pre-downloaded HLS composite (.tif) for that station/year/period. The
# resolution is PERIOD-AGNOSTIC: train on half-yearly composites now, switch
# to quarterly later via config flags only. data_preparation.py only resolves
# indices + writes a manifest; train.py loads the actual images via TileStore.
# ──────────────────────────────────────────────────────────────────────

# Day-of-year handed to Prithvi-TL per period bucket (≈ mid-period doy).
PERIOD_DOY = {
    "halfyear": {"H1": 90, "H2": 274},
    "quarter": {"Q1": 46, "Q2": 135, "Q3": 227, "Q4": 319},
}

# Filename: composite_<safe_id>_<year>_<period>.tif  (safe_id may contain '_'/'--').
# Greedy sid + end-anchored year/period correctly splits the trailing _YYYY_P.tif.
_COMPOSITE_RE = {
    "halfyear": re.compile(r"^composite_(?P<sid>.+)_(?P<year>\d{4})_(?P<p>H[12])\.tif$"),
    "quarter":  re.compile(r"^composite_(?P<sid>.+)_(?P<year>\d{4})_(?P<p>Q[1-4])\.tif$"),
}


def period_for_date(month: int, period: str) -> str:
    """Map a calendar month (1-12) to its period bucket label.

    halfyear → 'H1' (Jan-Jun) / 'H2' (Jul-Dec); quarter → 'Q1'..'Q4'.
    """
    if period == "halfyear":
        return "H1" if month <= 6 else "H2"
    if period == "quarter":
        return f"Q{(month - 1) // 3 + 1}"
    raise ValueError(f"Unknown composite_period: {period!r}")


def sanitize_safe_id(station_code) -> str:
    """Fallback raw station_code → safe_id, matching the download convention
    (space → '_', '/' → '--'). Used only when station_index_csv lacks the code.
    """
    return str(station_code).replace(" ", "_").replace("/", "--")


def parse_composite_filename(name: str, period: str):
    """Parse a composite basename → (safe_id, year:int, period_label) or None."""
    m = _COMPOSITE_RE[period].match(name)
    if not m:
        return None
    return m.group("sid"), int(m.group("year")), m.group("p")


def load_station_index(csv_path: str) -> Dict[str, str]:
    """Read the station-index CSV → {raw station_code: safe_id}.

    Requires columns 'station_code' and 'safe_id'. Empty path → {} (callers
    then fall back to sanitize_safe_id()).
    """
    if not csv_path:
        return {}
    idx = pd.read_csv(csv_path, dtype=str)
    if "station_code" not in idx.columns or "safe_id" not in idx.columns:
        raise ValueError(
            f"station_index_csv {csv_path} must have 'station_code' and "
            f"'safe_id' columns; got {list(idx.columns)}"
        )
    return dict(zip(idx["station_code"].astype(str), idx["safe_id"].astype(str)))


def build_tile_index(composite_dir: str, period: str):
    """Scan composite_dir for composite_*.tif of the given period.

    Returns (key_to_row, ordered_files, zero_idx, min_year):
      key_to_row    : {(safe_id, year, period_label): row_int}
      ordered_files : list of basenames; row == index into this list
      zero_idx      : len(ordered_files); sentinel "no tile → projector emits zeros"
      min_year      : earliest composite year seen (walk-back floor)
    """
    if period not in PERIOD_DOY:
        raise ValueError(f"Unknown composite_period: {period!r}")
    files = sorted(
        f for f in os.listdir(composite_dir)
        if f.startswith("composite_") and f.endswith(".tif")
    )
    key_to_row = {}
    ordered_files = []
    min_year = None
    skipped = 0
    for f in files:
        parsed = parse_composite_filename(f, period)
        if parsed is None:
            skipped += 1
            continue
        sid, year, p = parsed
        key = (sid, year, p)
        if key in key_to_row:
            continue  # duplicate (shouldn't happen) — keep first
        key_to_row[key] = len(ordered_files)
        ordered_files.append(f)
        min_year = year if min_year is None else min(min_year, year)
    zero_idx = len(ordered_files)
    if skipped:
        print(f"  [tile-index] skipped {skipped} non-{period} composite file(s)")
    return key_to_row, ordered_files, zero_idx, (min_year or 0)


def resolve_tile_idx(safe_id, current_date, key_to_row, zero_idx, period, min_year):
    """Resolve a sample's (safe_id, current_date) → tile_idx (row or zero_idx).

    Uses the latest COMPLETE same-period composite = (current_year-1,
    same_period): the current year's same period isn't finished yet, so year-1
    is the safe first candidate (no leakage, seasonal match). Walks back
    year-by-year to min_year; if none found → zero_idx.
    """
    p = period_for_date(current_date.month, period)
    year = current_date.year - 1
    while year >= min_year:
        row = key_to_row.get((safe_id, year, p))
        if row is not None:
            return row
        year -= 1
    return zero_idx


@dataclass
class DataConfig:
    """
    Configuration for LSTM data preparation.

    Reference: lstm_design.md (Configuration section)
    """

    # Forecast settings (configurable)

    # TODO forecast horizon and window size do not have to be the same
    forecast_horizon_months: int = 6  # Predict 6 months ahead (can be 3, 6, 12)
    lookback_years: int = 5  # 5 years of historical data (used unless lookback_months > 0)
    lookback_months: int = 0  # If > 0, overrides lookback_years (months of history)
    # Size of each lookback window — canonical unit is DAYS.
    # Default 90d ≈ 3 months. Use the parse_lookback_window() helper to convert
    # user-facing strings ('3', '1m', '7d', '90d') to this integer.
    lookback_window_days: int = 90
    # Drop NDVI and soil moisture from per-timestep aggregates and forecast features.
    # Both are 94-99% missing in raw data; dropping yields cleaner signal.
    # Default ON; set False (--no-drop-ndvi-sm) to keep them.
    drop_ndvi_sm: bool = True
    # When True: per-timestep sequence keeps ONLY GWL + temporal features
    # ([gwl_encoded, is_present, gwl_diff, gwl_diff_reliable, sin, cos] = 6 features).
    # Drops the 6 historical channels (rain, temp, ET, runoff, NDVI, SM) and the
    # 3 cumulative/anomaly rain features from the lookback. Forecast features
    # (forward-looking rain/temp) and static features are unaffected and still
    # gated by use_rain_temp / use_static_features.
    # Hypothesis: noisy historical covariates contribute to the gradient via
    # h(input,w) without contributing meaningful signal to g(actual,pred), so
    # removing them produces cleaner weight updates for the GWL→GWL relation.
    use_only_gwl: bool = False
    climatology_years: int = 12  # 12-year climatology for forecasts

    # Data availability settings
    gap_days: int = 60  # Search for GWL within ±gap_days of target date
    forward_fill_features: bool = True  # Use forward fill for missing feature values

    # Hybrid training settings (using climatology vs actual aggregates)
    use_actual_forecast_prob: float = 0.8  # Probability of using actuals in training

    # Sequence completeness: minimum fraction of timesteps with GWL present
    min_sequence_completeness: float = 0.5  # Drop sequences with < 50% GWL present

    # Minimum samples per station: stations with fewer samples are excluded entirely
    # Default 0 = disabled (we rely on min_sample_freq_days + downstream filters instead)
    min_station_samples: int = 0

    # Minimum sample frequency in days: thin each station's samples to roughly this
    # cadence using nearest-to-target selection. Reduces imbalance from
    # heavily-sampled stations. 0 = disabled.
    min_sample_freq_days: int = 14
    # Per-split overrides for min_sample_freq_days. -1 = inherit the global
    # value above. 0 = disable thinning for that split. Lets you e.g. keep
    # train un-thinned (more data, perwell loss handles imbalance) while
    # still thinning val/test for cohort stability.
    min_sample_freq_days_train: int = -1
    min_sample_freq_days_val:   int = -1
    min_sample_freq_days_test:  int = -1

    # Max samples per station in val/test splits (not applied to train).
    # Ensures no single station dominates evaluation metrics.
    max_samples_per_station_eval: int = 30

    # Minimum per-station std of target delta to keep a station.
    # Stations below this threshold ("flat wells") are excluded from train/val
    # because they lack meaningful dynamics. Test keeps all wells.
    # 0.0 = disabled (keep all stations).
    min_station_target_std: float = 0.0

    # Optional whitelist of states to keep in train/val (additive to std filter).
    # When non-empty, train+val are filtered to only include samples whose state
    # is in this set. Test stays unfiltered (cold-start view, same as std filter).
    # Empty list = disabled (keep all states).
    include_states: List[str] = field(default_factory=list)

    # Optional minimum per-station raw-observation mode-gap (in days).
    # When > 0, drops stations from train+val whose raw observations are too
    # densely sampled (mode_gap_days < threshold). Targets the dense-flat-well
    # noise-fitting failure mode. Test stays unfiltered. 0.0 = disabled.
    min_station_mode_gap_days: float = 0.0

    # Linear-interpolate the GWL channel of the LOOKBACK input sequence between
    # consecutive real observations within a station. Sample-grounding (current/
    # target) stays based on real observations — only the input sequence's GWL
    # is interpolated. is_present remains the dual encoding: 1 inside the
    # observed range (real or interpolated), 0 outside (no extrapolation).
    # Applies uniformly to train, val, test (no distribution shift).
    interpolate_lookback_gwl: bool = False

    # Maximum per-station std of target delta to keep a station.
    # When finite (< inf), stations with std >= max are also dropped — used by
    # the multi-model pipeline to slice wells into std buckets.
    # When max < inf, the filter ALSO applies to test_samples (bucket-restricted
    # test set). When max == inf (default), test is untouched, preserving
    # single-model behavior.
    max_station_target_std: float = float("inf")

    # Delta GWL encoding: predict change from current level instead of absolute
    use_delta_gwl: bool = (
        True  # Encode GWL as (value - current_gwl) in sequence & target
    )

    # Station-level outlier detection: validate current_gwl and target_gwl
    # against station historical bounds before building the sample
    validate_station_bounds: bool = True
    station_outlier_method: str = "mad"  # 'mad' (robust) or 'iqr' (legacy)
    station_iqr_multiplier: float = (
        1.5  # IQR multiplier for station bounds (used when method='iqr')
    )
    station_mad_multiplier: float = (
        3.0  # MAD multiplier for station bounds (used when method='mad')
    )

    # Hard physical cap on GWL values — applied in load_gwl_readings() before
    # any statistical outlier detection. Removes obviously corrupt sensor readings.
    # Set to 0 to disable. Applied per sign convention (positive/negative stations).
    max_gwl: float = 100.0  # Cap for positive-convention stations
    min_gwl: float = -100.0  # Cap for negative-convention stations

    # Dominant-sign threshold: fraction of non-negative readings above which a station
    # is classified as "positive convention". Range (0, 1). Default 0.5 means majority
    # vote; raise to e.g. 0.75 for a stricter majority requirement.
    station_sign_threshold: float = 0.5

    # Sign filter & hard cap analysis plots: if non-empty, plot_sign_filter_analysis()
    # and plot_hard_cap_analysis() are called from load_gwl_readings() using this as
    # the output root directory.
    sign_filter_plot_dir: str = ""
    # Per-window outlier detection: flag GWL readings that are outliers
    # within the lookback window (requires min_outlier_points present values)
    detect_window_outliers: bool = True
    min_outlier_points: int = 4  # Minimum present values to compute window IQR
    window_iqr_multiplier: float = 1.5  # IQR multiplier for window outlier detection
    outlier_min_abs_band_m: float = 1.0  # Floors the Tukey fence at ±this many meters
    # around the median. Prevents the IQR fence from collapsing to sub-meter widths
    # on flat wells (e.g. std=0.2m well gets IQR≈0.08m → fence ±0.12m → flags ~half
    # of the legitimate variation). With this floor, the band is always at least
    # ±1m wide, so real readings near the edges of a flat well's range are kept.
    # For volatile wells where IQR×1.5 > 1m, the Tukey fence dominates (unchanged).

    # Forecast feature mode: only use rainfall (sum) and temp (mean)
    use_rain_temp: bool = False

    # Derived values (computed in __post_init__)
    num_timesteps: int = None  # = lookback_total_days // lookback_window_days

    # Train/val/test split dates
    train_end_date: str = "2024-06-30"
    val_start_date: str = "2024-07-01"
    val_end_date: str = "2025-03-31"
    test_start_date: str = "2025-04-01"
    # CSV data source (overrides CSV_DATA_PATH constant if set)
    csv_path: str = ""

    # Database connection (defaults empty, resolved from env vars or CLI args)
    db_host: str = ""
    db_port: int = 5432
    db_name: str = ""
    db_user: str = ""
    db_password: str = ""
    database_table: str = "data_with_flag_500k"

    # Split strategy: "district" = station split by district,
    #                  "time" = global date cutoff,
    #                  "station_time" = per-station chronological split
    split_strategy: str = "district"

    # Per-station split ratios (only used when split_strategy="station_time")
    station_train_frac: float = 0.60
    station_val_frac: float = 0.25
    # test_frac = 1 - train_frac - val_frac (remainder)

    # Parallelism
    n_workers: int = 0  # 0 = auto (cpu_count - 1), 1 = sequential

    # ── Prithvi-EO satellite composites (Step 2: tile_idx annotation) ──
    # When True, every sample is annotated with an integer `tile_idx` that
    # points at the HLS composite tile (downloaded per station/year/period)
    # to feed the Prithvi encoder during joint fine-tuning. The actual image
    # loading happens in train.py via a manifest written by save_datasets();
    # data_preparation only resolves indices (no .tif reads here).
    use_prithvi: bool = False
    # Root dir holding composite_<safe_id>_<year>_<period>.tif files.
    composite_dir: str = ""
    # "halfyear" (H1/H2) or "quarter" (Q1-Q4). Selects bucketing, filename
    # regex, and day-of-year map. Logic is otherwise period-agnostic.
    composite_period: str = "halfyear"
    # CSV mapping raw station_code → sanitized safe_id used in tile filenames
    # (cols must include 'station_code' and 'safe_id'). Empty = derive safe_id
    # from station_code via the same sanitization the download used.
    station_index_csv: str = ""

    def __post_init__(self):
        """Compute derived values and resolve DB credentials."""
        # Effective lookback in months: lookback_months overrides lookback_years if > 0
        self.lookback_total_months = (
            self.lookback_months if self.lookback_months > 0
            else self.lookback_years * 12
        )
        # Total lookback in days (≈30 days/month convention; same as compute_aggregate)
        self.lookback_total_days = self.lookback_total_months * 30
        if self.lookback_window_days <= 0:
            raise ValueError(
                f"lookback_window_days must be > 0, got {self.lookback_window_days}"
            )
        self.num_timesteps = self.lookback_total_days // self.lookback_window_days
        if self.num_timesteps <= 0:
            raise ValueError(
                f"num_timesteps={self.num_timesteps} (lookback_total_days="
                f"{self.lookback_total_days}, lookback_window_days="
                f"{self.lookback_window_days}). "
                f"Window must be smaller than total lookback."
            )

        # Resolve DB credentials: CLI arg > env var > hardcoded default
        defaults = {
            "db_host": ("/var/run/postgresql", "GWL_DB_HOST"),
            "db_name": ("gwl", "GWL_DB_NAME"),
            "db_user": ("ubuntu", "GWL_DB_USER"),
            "db_password": ("", "GWL_DB_PASSWORD"),   # no hardcoded default — set via GWL_DB_PASSWORD / --db-password
        }
        for field, (fallback, env_key) in defaults.items():
            current = getattr(self, field)
            if not current:  # Empty string means not set via CLI
                setattr(self, field, os.environ.get(env_key, fallback))

        db_port_env = os.environ.get("GWL_DB_PORT")
        if db_port_env and self.db_port == 5432:
            self.db_port = int(db_port_env)


class LSTMDataPreparation:
    # TODO: we are already encoding each step in the time series. so why restrict sampling at regular intervals
    # TODO: all we nned is a good representation of current state (using current + historical). it may not be that important to sample at "regular" intervals for this
    """
    Prepare training data for GWL forecasting LSTM.

    This class handles:
    1. Loading raw data from PostgreSQL
    2. Creating horizon-aligned sequences   #NOTE Does not have to be horizon aligned)
    3. Computing climatological forecasts   #NOTE Avg over lookback years for the forecast duration
    4. Computing historical aggregates      #NOTE window size (does not have to equal horizon)
    5. Adding temporal encoding
    6. Creating train/val/test splits

    Reference: lstm_design.md (Input Features Design, Sequence Construction)
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.conn = None

    def connect_db(self):
        return
        """Connect to PostgreSQL database."""
        self.conn = psycopg2.connect(
            host=self.config.db_host,
            port=self.config.db_port,
            dbname=self.config.db_name,
            user=self.config.db_user,
            password=self.config.db_password,
        )

    def close_db(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    @staticmethod
    def build_interpolated_gwl_df(gwl_df: pd.DataFrame) -> pd.DataFrame:
        """Replace gwl_df with a daily-dense version per station.

        For each station: reindex onto a daily grid spanning [first_obs, last_obs],
        linearly interpolate gwl_value across the gaps, and add an `is_real`
        boolean column distinguishing original observations (True) from
        interpolated dates (False). Outside [first_obs, last_obs] no row is
        produced (preserves the is_present=0 semantics for out-of-range dates).

        Returns a new DataFrame with the same columns as the input plus
        `is_real`. The result has many more rows than the input — typical
        ratio is days-in-history / mean-cadence, e.g. 5×.

        Used only when DataConfig.interpolate_lookback_gwl is True.
        """
        if gwl_df is None or len(gwl_df) == 0:
            return gwl_df

        out_parts = []
        for code, group in gwl_df.groupby("station_code", sort=False):
            g = group.sort_values("date").drop_duplicates(subset=["date"])
            if len(g) < 2:
                # Need ≥2 points to interpolate — keep as-is, mark all real.
                g = g.copy()
                g["is_real"] = True
                out_parts.append(g)
                continue
            daily_idx = pd.date_range(
                start=g["date"].min(), end=g["date"].max(), freq="D"
            )
            real_dates = set(g["date"].dt.normalize())
            dense = (
                g.set_index("date")
                 .reindex(daily_idx)
                 .rename_axis("date")
                 .reset_index()
            )
            # Forward-fill station_code (and any other static-per-station columns)
            for col in dense.columns:
                if col in ("date", "gwl_value"):
                    continue
                dense[col] = dense[col].ffill().bfill()
            # Linear-interpolate the GWL channel only inside the observed span
            dense["gwl_value"] = dense["gwl_value"].interpolate(
                method="linear", limit_area="inside"
            )
            dense["is_real"] = dense["date"].isin(real_dates)
            out_parts.append(dense)

        if not out_parts:
            return gwl_df
        result = pd.concat(out_parts, ignore_index=True)
        return result

    def find_gwl_value_with_gap(
        self,
        station_gwl: pd.DataFrame,
        target_date: datetime,
        gap_days: int = None,
        real_only: bool = True,
    ) -> Optional[Tuple[float, datetime]]:
        """
        Find GWL value within ±gap_days of target date.

        Searches in order: exact date, then alternating -1, +1, -2, +2, etc.
        up to ±gap_days. Returns the first value found.

        When `real_only=True` (default) and station_gwl has an `is_real`
        column, interpolated rows (is_real=False) are skipped — useful for
        sample-grounding where current_gwl / target_gwl must be from real
        observations. Pass `real_only=False` to allow interpolated rows
        (used for the lookback timestep channel).

        Args:
            station_gwl: DataFrame with date index, 'gwl_value' column, and
                optionally an 'is_real' boolean column.
            target_date: Target date to search around.
            gap_days: Number of days to search on either side (uses config if None).
            real_only: If True, skip rows where is_real=False.

        Returns:
            Tuple of (gwl_value, actual_date) or None if no value found within gap.
        """
        if gap_days is None:
            gap_days = self.config.gap_days

        has_real_col = "is_real" in station_gwl.columns

        def _try(d):
            if d not in station_gwl.index:
                return None
            v = station_gwl.loc[d, "gwl_value"]
            if pd.isna(v):
                return None
            if real_only and has_real_col and not bool(station_gwl.loc[d, "is_real"]):
                return None
            return float(v), d

        # First check exact date
        hit = _try(target_date)
        if hit is not None:
            return hit

        # Search within ±gap_days
        for offset in range(1, gap_days + 1):
            # Check date - offset first (prefer earlier dates)
            hit = _try(target_date - pd.Timedelta(days=offset))
            if hit is not None:
                return hit

            # Then check date + offset
            hit = _try(target_date + pd.Timedelta(days=offset))
            if hit is not None:
                return hit

        return None

    def prepare_dynamic_features_with_fill(
        self, dynamic_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare dynamic features with forward fill for missing values.
        """
        feature_cols = ["rainfall", "temp", "et", "runoff", "ndvi", "sm", "lulc"]

        all_stations_dense = []

        for station_id, group in dynamic_df.groupby("station_code"):
            if group.empty:
                continue

            group = group.set_index("date").sort_index()

            min_date, max_date = group.index.min(), group.index.max()
            daily_index = pd.date_range(start=min_date, end=max_date, freq="D")

            dense_group = group.reindex(daily_index)[feature_cols]

            assert dense_group.index.is_monotonic_increasing, "Not in order"

            dense_group[feature_cols] = dense_group[feature_cols].ffill().infer_objects(copy=False)

            dense_group["lulc"] = dense_group["lulc"].fillna("no_data")

            for feature in feature_cols:
                if feature == "lulc":
                    continue

                col = dense_group[feature]
                if col.notna().sum() == 0:
                    dense_group[feature] = 0.0
                    continue

                dense_group[feature] = self._outlier(col)
                dense_group[feature] = dense_group[feature].fillna(0)

            dense_group["station_code"] = station_id

            all_stations_dense.append(
                dense_group.reset_index().rename(columns={"index": "date"})
            )

        if not all_stations_dense:
            return pd.DataFrame()

        _df = pd.concat(all_stations_dense, ignore_index=True)

        assert _df.isna().sum().sum() == 0, f"NaN count is {_df.isna().sum().sum()}"

        for feature in feature_cols:
            if feature == "lulc":
                continue
            print(feature, _df[feature].mean(), _df[feature].std())
        print(_df.describe(include="all"))

        return _df

    def _outlier(self, df_col: pd.Series) -> pd.Series:
        """Deal with outliers for each of the dynamic feature columns using IQR clipping."""
        q1 = df_col.quantile(0.25)
        q3 = df_col.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return df_col.clip(lower, upper)

    def load_gwl_readings(self) -> pd.DataFrame:
        """
        Load GWL readings from station_timeseries table.

        Returns:
            DataFrame with columns: [station_code, date, gwl_value]
        """
        query = f"""
        SELECT
            station_code,
            date,
            gwl_value
        FROM {self.config.database_table}
        WHERE gwl_value IS NOT NULL
        ORDER BY station_code, date
        """

        df = df_gwl_readings(self.config.csv_path or CSV_DATA_PATH)
        df = df[df["gwl_value"].notna()].sort_values(["station_code", "date"])
        df["date"] = pd.to_datetime(df["date"])
        # df = pd.read_sql(query, self.conn)        # read from file instead of db
        print(f"Loaded {len(df)} GWL readings")

        # --- Sign normalization: take abs values ---
        # Different agencies/loggers report GWL with opposite sign conventions
        # (positive = depth below surface vs negative = depth below surface).
        # Both encode the same physical quantity. Simplest robust handling: abs().
        # Hard cap at min_gwl=0 below ensures all-zero readings are dropped.
        if self.config.sign_filter_plot_dir:
            print("  Generating sign filter analysis plots...")
            plot_sign_filter_analysis(df, self.config.sign_filter_plot_dir)
        n_negative = int((df["gwl_value"] < 0).sum())
        df["gwl_value"] = df["gwl_value"].abs()
        if n_negative > 0:
            print(f"  Sign normalization: flipped {n_negative} negative readings via abs()")

        # --- Hard physical caps ---
        before_cap = len(df)
        if self.config.max_gwl > 0:
            df = df[df["gwl_value"] <= self.config.max_gwl]
        if self.config.min_gwl < 0:
            df = df[df["gwl_value"] >= self.config.min_gwl]
        removed_cap = before_cap - len(df)
        if removed_cap > 0:
            print(
                f"  Removed {removed_cap} readings outside hard caps "
                f"[{self.config.min_gwl}, {self.config.max_gwl}]m"
            )
        else:
            print(
                f"  Hard caps [{self.config.min_gwl}, {self.config.max_gwl}]m: "
                f"no readings removed"
            )

        print(f"Loaded {len(df)} GWL readings after filtering")
        df["date"] = pd.to_datetime(df["date"])
        return df

    def load_dynamic_features(self) -> pd.DataFrame:
        """
        Load dynamic features (rainfall, temperature, ET, NDVI, soil moisture, runoff, LULC).

        Returns:
            DataFrame with columns: [station_code, date, rainfall, temp, et, runoff, ndvi, sm, lulc]
        """
        query = f"""
        SELECT
            station_code,
            date,
            precipitation as rainfall,
            temperature_2m as temp,
            total_evaporation_sum as et,
            runoff_sum as runoff,
            sr_b5, sr_b4, soil_moisture_am,soil_moisture_pm,
            dominant_class as lulc
        FROM {self.config.database_table}
        ORDER BY station_code, date;
        """

        # df = pd.read_sql(query, self.conn)    # read from csv
        df = df_dynamic_features(self.config.csv_path or CSV_DATA_PATH).copy()

        # Flatten MultiIndex columns if present (can happen with duplicate CSV column names)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df["date"] = pd.to_datetime(df["date"])

        # NDVI safe computation
        # ndvi_denom = df["sr_b5"] + df["sr_b4"]
        # df["ndvi"] = np.where(
        #     ndvi_denom == 0, 0, (df["sr_b5"] - df["sr_b4"]) / ndvi_denom
        # )

        # Soil moisture safe average
        # df["sm"] = df[["soil_moisture_am", "soil_moisture_pm"]].mean(axis=1)

        df.loc[df["sm"] < 0, "sm"] = 0
        # ERA5 total_evaporation_sum is negative by convention (water leaving surface).
        # Take absolute value to get positive evaporation magnitude.
        df["et"] = df["et"].abs()
        df.loc[df["rainfall"] < 0, "rainfall"] = 0
        df.loc[df["temp"] < 0, "temp"] = 0

        # Replace inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        dynamic_df = self.prepare_dynamic_features_with_fill(df)
        return dynamic_df

    def load_static_features(self) -> pd.DataFrame:
        """
        Load static well characteristics.

        Returns:
            DataFrame with columns: [station_code, elevation, well_type,
                                    aquifer_type, lithology, stream_order,
                                    aquifer_0_aquifer, litho_supergroup,
                                    latitude, longitude]
        """
        query = f"""
        SELECT
            station_code,
            elevation,
            well_type,
            well_aquifer_type AS aquifer_type,
            litho_lithologic as lithology,
            stream_order,
            aquifer_0_aquifer,
            litho_supergroup,
            latitude,
            longitude
        FROM {self.config.database_table}
        ORDER BY station_code,date;
        """
        # df = pd.read_sql(query, self.conn)
        df = df_static_features(self.config.csv_path or CSV_DATA_PATH)

        # ── Compute well_depth (mean per station, cascade fill missing) ──
        if "depth" in df.columns:
            # Depths cannot be negative — take absolute value to fix data errors
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce").abs()
            # Per-station mean depth
            df["well_depth"] = df.groupby("station_code")["depth"].transform("mean")
            # Cascade fill: village → district → state → global median
            for level in ["village", "district", "state"]:
                if level in df.columns:
                    df["well_depth"] = df["well_depth"].fillna(
                        df.groupby(level)["well_depth"].transform("median")
                    )
            df["well_depth"] = df["well_depth"].fillna(df["well_depth"].median())
            print(f"  well_depth computed: median={df['well_depth'].median():.1f}m, "
                  f"min={df['well_depth'].min():.1f}m, max={df['well_depth'].max():.1f}m")

        static_cols = [
            "station_code",
            "elevation",
            "well_type",
            "aquifer_type",
            "lithology",
            "stream_order",
            "aquifer_0_aquifer",
            "litho_supergroup",
            "latitude",
            "longitude",
            "well_depth",
        ]
        df = df[[c for c in static_cols if c in df.columns]]
        print("Length of Static table is: ", len(df))

        numeric_cols = ["elevation", "stream_order", "latitude", "longitude", "well_depth"]
        clip_zero_cols = {"elevation", "stream_order", "well_depth"}  # Clip negatives to 0
        for col in numeric_cols:
            if col in clip_zero_cols:
                df.loc[df[col] < 0, col] = 0
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(f"  Filling {missing_count} missing values in '{col}' with 0")
                df[col] = df[col].fillna(0.0)

        categorical_cols = [
            "well_type",
            "aquifer_type",
            "lithology",
            "aquifer_0_aquifer",
            "litho_supergroup",
        ]
        for col in categorical_cols:
            if col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    print(
                        f"  Filling {missing_count} missing values in '{col}' with 'unknown'"
                    )
                df[col] = df[col].fillna("no_data")
        static_unique = df.drop_duplicates(subset=["station_code"], keep="first")
        print(f"static features count: {len(static_unique)}")
        return static_unique

    def compute_cyclical_encoding(self, month: int) -> Tuple[float, float]:
        """
        Compute cyclical encoding for month.

        Args:
            month: Month number (1-12)

        Returns:
            (month_sin, month_cos)

        Reference: lstm_design.md (Temporal Encoding)
        """
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        return month_sin, month_cos

    def compute_station_gwl_bounds(
        self,
        gwl_df: pd.DataFrame,
        start_date: str = None,
        end_date: str = None,
    ) -> Dict[int, Optional[Tuple[float, float]]]:
        """
        Compute per-station bounds for GWL outlier detection.

        Supports two methods (controlled by config.station_outlier_method):
        - 'mad': Median Absolute Deviation (robust to up to 50% contamination)
          bounds = median ± multiplier × 1.4826 × MAD
          where 1.4826 is the consistency constant for normal distributions
        - 'iqr': Interquartile Range (legacy, breaks when >25% corrupt)
          bounds = [Q1 - m×IQR, Q3 + m×IQR]

        Uses only readings within the specified date range.
        Stations with fewer than min_outlier_points readings get None (no bounds).

        Bounds are computed independently per split to handle non-stationary GWL
        (e.g. secular declining trends). Each split uses its own data to define
        what counts as a sensor error for that period.

        Args:
            gwl_df: Full GWL DataFrame with columns [station_code, date, gwl_value]
            start_date: Start of period (inclusive). None = no lower bound.
            end_date: End of period (inclusive). None = no upper bound.

        Returns:
            Dict mapping station_code → (lower_bound, upper_bound) or None
        """
        filtered = gwl_df
        if start_date is not None:
            filtered = filtered[filtered["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            filtered = filtered[filtered["date"] <= pd.to_datetime(end_date)]

        method = self.config.station_outlier_method
        bounds = {}
        n_skipped = 0

        for station, group in filtered.groupby("station_code"):
            values = group["gwl_value"].dropna().values
            if len(values) < self.config.min_outlier_points:
                bounds[station] = None
                n_skipped += 1
                continue

            if method == "mad":
                # MAD: robust to up to 50% contamination
                # 1.4826 = consistency constant (makes MAD comparable to std for normal data)
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                scaled_mad = 1.4826 * mad
                m = self.config.station_mad_multiplier
                if scaled_mad == 0:
                    # All values identical or nearly so — use tiny tolerance
                    bounds[station] = (median - 0.5, median + 0.5)
                else:
                    bounds[station] = (median - m * scaled_mad, median + m * scaled_mad)
            elif method == "iqr":
                # IQR: legacy method, breaks when >25% data is corrupt
                m = self.config.station_iqr_multiplier
                q1, q3 = np.percentile(values, [25, 75])
                iqr = q3 - q1
                bounds[station] = (q1 - m * iqr, q3 + m * iqr)
            else:
                raise ValueError(
                    f"Unknown station_outlier_method: {method}. Use 'mad' or 'iqr'."
                )

        period = f"{start_date or '...'} → {end_date or '...'}"
        method_label = (
            f"MAD×{self.config.station_mad_multiplier}"
            if method == "mad"
            else f"IQR×{self.config.station_iqr_multiplier}"
        )
        print(
            f"  Computed station GWL bounds ({method_label}) for {len(bounds) - n_skipped} stations "
            f"(skipped {n_skipped} with <{self.config.min_outlier_points} readings) "
            f"[{period}]"
        )
        return bounds

    @staticmethod
    def detect_window_outlier_flags(
        raw_values: List[Optional[float]],
        present_flags: List[bool],
        iqr_multiplier: float = 1.5,
        min_points: int = 4,
        min_abs_band_m: float = 1.0,
    ) -> List[bool]:
        """
        Detect outliers within a lookback window of GWL values.

        Computes IQR on present values. Any present value outside
        [Q1 - m*IQR, Q3 + m*IQR] is flagged as an outlier. If fewer than
        min_points values are present, returns all False (no outlier detection).

        The Tukey fence is widened to at least ±min_abs_band_m around the
        median, preventing collapse to sub-meter widths on flat wells where
        the IQR is naturally tiny (and would otherwise falsely flag
        legitimate variation).

        Args:
            raw_values: Raw GWL values per timestep (None if missing)
            present_flags: True if GWL was found for that timestep
            iqr_multiplier: IQR multiplier for bounds
            min_points: Minimum present values to attempt detection
            min_abs_band_m: Absolute-meter half-width floor around median

        Returns:
            List of bools, True = outlier (same length as raw_values)
        """
        n = len(raw_values)
        outlier_flags = [False] * n

        # Collect present values
        present_vals = [
            v for v, p in zip(raw_values, present_flags) if p and v is not None
        ]

        if len(present_vals) < min_points:
            # Not enough data for reliable detection — trust all present values
            return outlier_flags

        arr = np.array(present_vals)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        median = float(np.median(arr))
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr
        # Widen the band to at least ±min_abs_band_m around the median.
        # For flat wells, IQR collapses and Tukey alone falsely flags normal data.
        lower = min(lower, median - min_abs_band_m)
        upper = max(upper, median + min_abs_band_m)

        for i in range(n):
            if present_flags[i] and raw_values[i] is not None:
                if raw_values[i] < lower or raw_values[i] > upper:
                    outlier_flags[i] = True

        return outlier_flags

    def compute_aggregate(
        self,
        station_dynamic: pd.DataFrame,
        end_date: datetime,
        window_days: int,
        feature: str,
        agg_func: str,
    ) -> Optional[float]:
        """
        Compute aggregate for a feature over a time window (backward-looking).

        The window is [end_date - window_days, end_date), i.e., it looks backward
        from the end_date.

        Args:
            station_dynamic: DataFrame with date index for a single station (pre-indexed)
            end_date: End date of window (exclusive), window looks backward from here
            window_days: Window size in days
            feature: Feature name
            agg_func: 'sum' or 'mean'

        Returns:
            Aggregated value or None if insufficient data

        Reference: lstm_design.md (Aggregation Strategy)
        """
        start_date = end_date - relativedelta(days=window_days)

        # Use index slicing for efficient lookup on pre-indexed data
        try:
            mask = (station_dynamic.index >= start_date) & (
                station_dynamic.index < end_date
            )
            station_data = station_dynamic.loc[mask, feature]
        except KeyError:
            return np.nan

        if station_data.empty:
            return np.nan

        if agg_func == "sum":
            return station_data.sum()
        elif agg_func == "mean":
            return station_data.mean()
        elif agg_func == "mode":
            modes = station_data.mode()
            return modes.iloc[0] if not modes.empty else "no_data"
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

    def compute_climatology(
        self,
        station_dynamic: pd.DataFrame,
        reference_date: datetime,
        window_days: int,
        feature: str,
        agg_func: str,
        reference_year: int,
        num_years: int = 12,
    ) -> Optional[float]:
        """
        Compute climatological average for forecast period using only historical data.

        IMPORTANT: Only uses data from years BEFORE reference_year to avoid data leakage.

        Args:
            station_dynamic: DataFrame with date index for a single station (pre-indexed)
            reference_date: Reference datetime to extract month from (uses day=1)
            window_days: Forecast window size in days
            feature: Feature name
            agg_func: 'sum' or 'mean'
            reference_year: The year we're forecasting INTO (exclude this and later years)
            num_years: Maximum number of historical years to average over

        Returns:
            Climatological average or None if insufficient data

        Reference: lstm_design.md (Climatological Forecast section)
        """
        if station_dynamic.empty:
            return np.nan

        values = []
        years_available = station_dynamic.index.year.unique()

        # Only use years BEFORE the reference year (avoid data leakage)
        valid_years = sorted(
            [y for y in years_available if y < reference_year], reverse=True
        )

        # Limit to num_years most recent historical years
        valid_years = valid_years[:num_years]

        for year in valid_years:
            start = datetime(year, reference_date.month, 1)
            end_date = start + relativedelta(days=window_days)

            # Reuse compute_aggregate for consistent logic
            val = self.compute_aggregate(
                station_dynamic, end_date, window_days, feature, agg_func
            )
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                values.append(val)

        if not values:
            return np.nan

        # For categorical features (mode aggregation), return the overall mode across years
        if agg_func == "mode":
            return pd.Series(values).mode().iloc[0]

        return np.mean(values)

    def create_sequence_for_sample(
        self,
        station_gwl: pd.DataFrame,
        station_dynamic: pd.DataFrame,
        current_date: datetime,
        target_date: datetime,
        current_gwl: float,
        is_training: bool = True,
        external_forecast_values: Optional[Dict] = None,
        drop_reasons: Optional[Dict] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, str, List[str]]]:
        """
        Create the input sequence and forecast features for a single sample.

        Two-pass approach for GWL encoding:
        1. Collect raw GWL values + temporal/historical features across all timesteps
        2. Detect outliers within the window, then encode as (delta, is_reliable)

        GWL encoding per timestep: [delta_gwl, is_reliable]
        - delta_gwl: (raw_gwl - current_gwl) if reliable, else 0.0
        - is_reliable: 1.0 if present AND not an outlier, else 0.0

        When use_delta_gwl=False, falls back to [raw_gwl, is_present] (original behavior).

        Args:
            station_gwl: GWL DataFrame with date index for this station (pre-indexed).
            station_dynamic: Dynamic features DataFrame with date index (pre-indexed).
            current_date: Current date (t - horizon).
            target_date: Target date (t).
            current_gwl: Current GWL value (anchor for delta computation).
            is_training: Flag to indicate if in training mode.
            external_forecast_values: Optional dict of external forecast values.

        Returns:
            Tuple of (sequence, forecast_features, forecast_lulc, historical_lulc) or None.
        """
        horizon = target_date - current_date  # this is in days
        config_horizon = self.config.forecast_horizon_months
        lookback_window_days = self.config.lookback_window_days
        num_timesteps = self.config.num_timesteps

        agg_functions = dict(FEATURE_AGG_FUNCTIONS)   # module constant (shared with inference)
        # Optionally drop NDVI/SM (94-99% missing in raw data; dropping yields cleaner signal)
        if getattr(self.config, "drop_ndvi_sm", True):
            agg_functions = {k: v for k, v in agg_functions.items() if k not in ("ndvi", "sm")}

        numeric_features_to_process = [f for f in agg_functions.keys() if f != "lulc"]

        # --- Forecast Feature Generation ---
        # Always use actual future aggregates from [current_date, target_date).
        # External forecasts can still override if provided.
        forecast_values = {}
        forecast_lulc = "no_data"

        if external_forecast_values is not None:
            # Use provided external forecast (e.g., weather model output)
            for feature in numeric_features_to_process:
                forecast_values[feature] = external_forecast_values.get(feature, np.nan)
            forecast_lulc = external_forecast_values.get("lulc", "no_data")
        else:
            # Use actual future aggregates
            forecast_end_date = current_date + (horizon)
            assert forecast_end_date == target_date
            for feature, agg_func in agg_functions.items():
                val = self.compute_aggregate(
                    station_dynamic, forecast_end_date, horizon.days, feature, agg_func
                )
                if feature == "lulc":
                    forecast_lulc = (
                        val if (val is not None and not pd.isna(val)) else "no_data"
                    )
                else:
                    forecast_values[feature] = val

        if self.config.use_rain_temp:
            forecast_features = np.array(
                [
                    forecast_values["rainfall"],
                    forecast_values["temp"],
                ]
            )
        elif getattr(self.config, "drop_ndvi_sm", True):
            forecast_features = np.array(
                [
                    forecast_values["rainfall"],
                    forecast_values["temp"],
                    forecast_values["et"],
                    forecast_values["runoff"],
                ]
            )
        else:
            forecast_features = np.array(
                [
                    forecast_values["rainfall"],
                    forecast_values["temp"],
                    forecast_values["et"],
                    forecast_values["runoff"],
                    forecast_values["ndvi"],
                    forecast_values["sm"],
                ]
            )

        # --- Pass 1: Collect raw GWL values + other features per timestep ---
        raw_gwl_values = []  # Raw GWL value or None if missing
        gwl_present_flags = []  # True if GWL reading was found
        timestep_other_features = []  # [sin, cos, hist_features] per timestep
        historical_lulc = []
        timestep_dates = []  # Store dates for rainfall anomaly computation
        window_days = lookback_window_days  # canonical days unit; no ×30 conversion

        timestep_date = current_date
        # When interpolate_lookback_gwl=True, station_gwl carries an `is_real`
        # column and the lookback channel may use any in-range date — so we
        # disable the gap-based fallback (gap_days=0; exact match only) and
        # accept interpolated rows (real_only=False). When the flag is off
        # the gap is used as a search tolerance only — the timestep grid
        # stays on the clean current_date − i*window_days schedule regardless
        # of where the matched obs falls inside the gap.
        _interp_on = self.config.interpolate_lookback_gwl
        _lookback_gap = 0 if _interp_on else self.config.gap_days
        _lookback_real_only = (not _interp_on)
        for i in range(0, num_timesteps):
            # 1. GWL point value lookup. The matched date is intentionally
            # discarded: timestep_date stays anchored to the clean grid so
            # the next decrement walks back by exactly window_days. (Earlier
            # versions reassigned timestep_date to the matched date, which
            # collapsed the lookback when window_days < gap_days.)
            gwl_result = self.find_gwl_value_with_gap(
                station_gwl,
                timestep_date,
                gap_days=_lookback_gap,
                real_only=_lookback_real_only,
            )
            if gwl_result:
                gwl_value, _matched_date = gwl_result
                raw_gwl_values.append(gwl_value)
                gwl_present_flags.append(True)
            else:
                raw_gwl_values.append(None)
                gwl_present_flags.append(False)

            timestep_dates.append(timestep_date)

            # 2. Temporal encoding
            month_sin, month_cos = self.compute_cyclical_encoding(timestep_date.month)

            # 3. Historical aggregates
            hist_features = []
            for feature, agg_func in agg_functions.items():
                if feature == "lulc":
                    hist_val = self.compute_aggregate(
                        station_dynamic,
                        timestep_date,
                        window_days,
                        feature,
                        agg_func,
                    )
                    historical_lulc.append(
                        hist_val
                        if (hist_val is not None and not pd.isna(hist_val))
                        else "no_data"
                    )
                else:
                    hist_val = self.compute_aggregate(
                        station_dynamic,
                        timestep_date,
                        window_days,
                        feature,
                        agg_func,
                    )
                    hist_features.append(hist_val)

            timestep_other_features.append([month_sin, month_cos] + hist_features)
            # Decrement by same window size used for aggregates
            timestep_date -= relativedelta(days=window_days)

        # --- Pass 1b: Rainfall enrichment features ────────────────────────
        # timestep_other_features[i] layout: [sin, cos, rainfall, temp, et, runoff, ndvi, sm]
        # rainfall is at index 2 (first hist_feature after sin/cos)
        rain_idx = 2  # index in timestep_other_features

        # Collect per-window rainfall values (newest-first order, matching timesteps)
        rain_per_window = []
        rain_anomaly_per_window = []
        for i in range(num_timesteps):
            rain_val = timestep_other_features[i][rain_idx]
            rain_val_missing = (rain_val is None) or np.isnan(rain_val)

            # Compute climatological rainfall for this window
            rain_clim = self.compute_climatology(
                station_dynamic,
                timestep_dates[i],
                window_days,
                "rainfall",
                "sum",
                reference_year=timestep_dates[i].year,
                num_years=self.config.climatology_years,
            )
            rain_clim_safe = rain_clim if (rain_clim is not None and not np.isnan(rain_clim)) else 0.0

            # When rainfall is missing, impute climatology so the anomaly is 0
            # (no signal contributed) instead of −climatology, which would falsely
            # tell the model "this window was a catastrophic drought". Write the
            # imputed value back so the standalone per-timestep rainfall feature
            # stays consistent with cum_rain and rain_anomaly downstream.
            rain_val_imputed = rain_clim_safe if rain_val_missing else rain_val
            timestep_other_features[i][rain_idx] = rain_val_imputed

            rain_per_window.append(rain_val_imputed)
            rain_anomaly_per_window.append(rain_val_imputed - rain_clim_safe)

        # Compute cumulative values in chronological order (oldest first)
        # Timesteps are currently newest-first, so reverse for cumsum
        rain_chrono = rain_per_window[::-1]
        anomaly_chrono = rain_anomaly_per_window[::-1]
        cum_rain_chrono = list(np.cumsum(rain_chrono))
        cum_anomaly_chrono = list(np.cumsum(anomaly_chrono))
        # Reverse back to newest-first to match timestep order
        cum_rain = cum_rain_chrono[::-1]
        cum_anomaly = cum_anomaly_chrono[::-1]

        # Append 3 new features to each timestep's other_features
        for i in range(num_timesteps):
            timestep_other_features[i].extend([
                cum_rain[i],
                rain_anomaly_per_window[i],
                cum_anomaly[i],
            ])

        # --- Completeness Check (based on original presence, before outlier detection) ---
        num_present = sum(gwl_present_flags)
        if num_present / num_timesteps < self.config.min_sequence_completeness:
            if drop_reasons is not None:
                drop_reasons["sequence_incomplete"] += 1
                if len(drop_reasons["sequence_incomplete_details"]) < 100:
                    drop_reasons["sequence_incomplete_details"].append(
                        {
                            "num_present": num_present,
                            "num_timesteps": num_timesteps,
                            "ratio": num_present / num_timesteps,
                        }
                    )
            return None

        # --- Pass 2: Outlier detection within the window ---
        if self.config.detect_window_outliers:
            outlier_flags = self.detect_window_outlier_flags(
                raw_gwl_values,
                gwl_present_flags,
                iqr_multiplier=self.config.window_iqr_multiplier,
                min_points=self.config.min_outlier_points,
                min_abs_band_m=getattr(self.config, "outlier_min_abs_band_m", 1.0),
            )
        else:
            outlier_flags = [False] * num_timesteps

        # --- Pass 3: Encode GWL as (delta, is_reliable) or (raw, is_present) ---
        gwl_encoded_values = []
        gwl_flags = []
        for i in range(num_timesteps):
            is_present = gwl_present_flags[i]
            is_outlier = outlier_flags[i]

            if self.config.use_delta_gwl:
                # Delta encoding: is_reliable = present AND not outlier
                if is_present and not is_outlier:
                    gwl_encoded = raw_gwl_values[i] - current_gwl
                    gwl_flag = 1.0  # is_reliable
                else:
                    gwl_encoded = 0.0
                    gwl_flag = 0.0  # unreliable (missing or outlier)
            else:
                # Original encoding: raw value + is_present
                if is_present:
                    gwl_encoded = raw_gwl_values[i]
                    if is_outlier:
                        gwl_flag = 0.0  # still flag outliers as not present
                        gwl_encoded = np.nan
                    else:
                        gwl_flag = 1.0
                else:
                    gwl_encoded = np.nan
                    gwl_flag = 0.0

            gwl_encoded_values.append(gwl_encoded)
            gwl_flags.append(gwl_flag)

        # --- Pass 4: Compute gwl_diff (local trend) between consecutive timesteps ---
        # Loop is newest-first: i=0 is current_date, i+1 is one window older.
        # gwl_diff[i] = gwl_encoded[i] - gwl_encoded[i+1] = change from older to newer.
        # Both must be reliable; otherwise diff = 0.0, diff_reliable = 0.0.
        gwl_diff_values = []
        gwl_diff_flags = []
        for i in range(num_timesteps):
            if i < num_timesteps - 1 and gwl_flags[i] == 1.0 and gwl_flags[i + 1] == 1.0:
                gwl_diff_values.append(gwl_encoded_values[i] - gwl_encoded_values[i + 1])
                gwl_diff_flags.append(1.0)
            else:
                gwl_diff_values.append(0.0)
                gwl_diff_flags.append(0.0)

        # --- Assemble sequence ---
        # Default: [gwl_encoded, gwl_flag, gwl_diff, gwl_diff_flag, sin, cos,
        #  6 historical (rain,temp,et,runoff,ndvi,sm),
        #  cum_rain, rain_anomaly, cum_rain_anomaly] = 15 features
        # When use_only_gwl=True: drop the 9 other-channel features, keeping
        # only [gwl_encoded, gwl_flag, gwl_diff, gwl_diff_flag, sin, cos] = 6.
        only_gwl = getattr(self.config, "use_only_gwl", False)
        sequence = []
        for i in range(num_timesteps):
            # timestep_other_features[i] layout: [sin, cos, <historical...>, <cum/anom...>]
            other_feats = (
                timestep_other_features[i][:2]  # sin, cos only
                if only_gwl else timestep_other_features[i]
            )
            timestep_features = (
                [gwl_encoded_values[i], gwl_flags[i],
                 gwl_diff_values[i], gwl_diff_flags[i]]
                + other_feats
            )
            sequence.append(timestep_features)

        # Reverse sequence([t, t-1, t-3.... t-9]) to [t-9, t-8.... t]
        sequence = list(reversed(sequence))
        historical_lulc = list(reversed(historical_lulc))

        return np.array(sequence), forecast_features, forecast_lulc, historical_lulc

    def create_sample(
        self,
        station_gwl: pd.DataFrame,
        station_dynamic: pd.DataFrame,
        station_static: pd.Series,
        station_code: int,
        current_date: datetime,
        station_gwl_bounds: Optional[Tuple[float, float]] = None,
        is_training: bool = True,
        external_forecast_values: Optional[Dict] = None,
        drop_reasons: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Create a single training/validation/test sample for the station code.

        Pipeline:
        1. Look up current_gwl and target_gwl
        2. Validate both against station IQR bounds (if enabled) → drop if outlier
        3. Build input sequence with delta encoding + per-window outlier detection
        4. Compute target as delta (target_gwl - current_gwl) or absolute

        Args:
            station_gwl_bounds: (lower, upper) IQR bounds for this station, or None
        """
        horizon = self.config.forecast_horizon_months

        # Get current GWL (at current_date) with gap-based lookup
        current_result = self.find_gwl_value_with_gap(station_gwl, current_date)
        if current_result is None:
            if drop_reasons is not None:
                drop_reasons["no_current_gwl"] += 1
            return None
        current_gwl, _ = current_result

        # Get target GWL (at t, which is current_date + horizon) with gap-based lookup
        target_date = current_date + relativedelta(months=horizon)

        target_result = self.find_gwl_value_with_gap(station_gwl, target_date)
        if target_result is None:
            if external_forecast_values is not None:
                target_gwl = -9999
            else:
                if drop_reasons is not None:
                    drop_reasons["no_target_gwl"] += 1
                return None
        else:
            target_gwl, target_date = target_result

        # --- Station-level outlier check ---
        # Drop sample if current_gwl or target_gwl falls outside station's historical bounds
        if self.config.validate_station_bounds and station_gwl_bounds is not None:
            lower, upper = station_gwl_bounds

            if current_gwl < lower or current_gwl > upper:
                if drop_reasons is not None:
                    drop_reasons["current_gwl_outlier"] += 1
                return None

            if target_gwl != -9999 and (target_gwl < lower or target_gwl > upper):
                if drop_reasons is not None:
                    drop_reasons["target_gwl_outlier"] += 1
                return None

        # Create sequence and forecast features (pass current_gwl for delta computation)
        result = self.create_sequence_for_sample(
            station_gwl,
            station_dynamic,
            current_date,
            target_date,
            current_gwl=current_gwl,
            is_training=is_training,
            external_forecast_values=external_forecast_values,
            drop_reasons=drop_reasons,
        )

        if result is None:
            return None

        sequence, forecast_features, forecast_lulc, historical_lulc = result

        # Compute target: delta or absolute
        if self.config.use_delta_gwl and target_gwl != -9999:
            target_value = target_gwl - current_gwl
        else:
            target_value = target_gwl

        # Get state and district from GWL data (available per station)
        state = station_gwl["state"].iloc[0] if "state" in station_gwl.columns else "no_data"
        district = station_gwl["district"].iloc[0] if "district" in station_gwl.columns else "no_data"
        # Guard against NaN values
        if pd.isna(state):
            state = "no_data"
        if pd.isna(district):
            district = "no_data"

        sample = {
            "sequence": sequence,
            "forecast_features": forecast_features,
            "forecast_lulc": forecast_lulc,
            "historical_lulc": historical_lulc,
            "elevation": station_static["elevation"],
            "well_depth": station_static.get("well_depth", 0.0),
            "latitude": station_static["latitude"],
            "longitude": station_static["longitude"],
            "well_type": station_static["well_type"],
            "aquifer_type": station_static["aquifer_type"],
            "lithology": station_static["lithology"],
            "stream_order": station_static["stream_order"],
            "aquifer_0_aquifer": station_static.get("aquifer_0_aquifer", "no_data"),
            "litho_supergroup": station_static.get("litho_supergroup", "no_data"),
            # Derived per-station static features (filled by prepare_all_data after sample creation)
            "mean_annual_rainfall": 0.0,
            "annual_rainfall_std": 0.0,
            "station_mean_gwl": 0.0,
            "station_gwl_amplitude": 0.0,
            "station_delta_mean": 0.0,
            "station_delta_std": 0.0,
            "gwl_anomaly": 0.0,
            "target_gwl": target_value,  # Delta or absolute depending on config
            "target_gwl_raw": target_gwl,  # Always absolute (for reconstruction)
            "current_gwl": current_gwl,  # Always absolute (anchor for reconstruction)
            "station_code": station_code,
            "state": state,
            "district": district,
            "current_date": current_date,
            "target_date": target_date,
        }

        return sample

    def create_dataset(
        self,
        gwl_df: pd.DataFrame,
        dynamic_df: pd.DataFrame,
        static_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        station_gwl_bounds: Optional[Dict[int, Optional[Tuple[float, float]]]] = None,
        is_training: bool = True,
    ) -> List[Dict]:
        """
        Create dataset for a given date range.

        Args:
            station_gwl_bounds: Precomputed per-station IQR bounds (from training data).
                                Used to validate current_gwl and target_gwl.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Pre-index dataframes ONCE for efficient lookup
        print("Pre-indexing dataframes for efficient lookup...")
        gwl_by_station = {
            station: group.set_index("date").sort_index()
            for station, group in gwl_df.groupby("station_code")
        }
        dynamic_by_station = {
            station: group.set_index("date").sort_index()
            for station, group in dynamic_df.groupby("station_code")
        }
        static_by_station = {
            row["station_code"]: row
            for _, row in static_df.drop_duplicates("station_code").iterrows()
        }

        # Filter to date range and group by station
        filtered_gwl = gwl_df[(gwl_df["date"] >= start) & (gwl_df["date"] <= end)]
        grouped = filtered_gwl.groupby("station_code")

        samples = []
        total_stations = len(grouped)

        # Initialize drop reason counters
        drop_reasons = {
            "no_station_gwl": 0,
            "station_gwl_too_short": 0,
            "no_station_dynamic": 0,
            "no_station_static": 0,
            "no_current_gwl": 0,
            "no_target_gwl": 0,
            "current_gwl_outlier": 0,
            "target_gwl_outlier": 0,
            "sequence_incomplete": 0,
            "sequence_incomplete_details": [],
        }
        total_dates_in_range = len(filtered_gwl)
        total_after_station_filter = 0

        print(
            f"Creating samples from {start_date} to {end_date} (is_training={is_training})"
        )
        print(f"Total stations with data in range: {total_stations}")
        print(f"Horizon: {self.config.forecast_horizon_months} months")
        print(
            f"Lookback: {self.config.lookback_total_months}m ({self.config.lookback_total_days}d), window: {self.config.lookback_window_days}d ({self.config.num_timesteps} timesteps)"
        )
        if self.config.use_delta_gwl:
            print(f"GWL encoding: DELTA (value - current_gwl)")
        else:
            print(f"GWL encoding: ABSOLUTE (raw values)")
        if self.config.validate_station_bounds and station_gwl_bounds is not None:
            print(
                f"Station bounds validation: ENABLED ({len(station_gwl_bounds)} stations)"
            )
        if self.config.detect_window_outliers:
            print(
                f"Window outlier detection: ENABLED (IQR×{self.config.window_iqr_multiplier}, "
                f"min {self.config.min_outlier_points} points)"
            )

        # Resolve worker count
        n_workers = self.config.n_workers
        if n_workers == 0:
            n_workers = max(1, mp.cpu_count() - 1)
        use_parallel = n_workers > 1

        # --- Station-level filtering (fast, stays in main thread) ---
        work_items = (
            []
        )  # (config, station_code, station_gwl, station_dynamic, station_static, dates, bounds, is_training)

        for idx, (station_code, group) in enumerate(grouped):
            station_gwl = gwl_by_station.get(station_code)
            station_dynamic = dynamic_by_station.get(station_code)
            station_static = static_by_station.get(station_code)

            if station_gwl is None:
                drop_reasons["no_station_gwl"] += len(group)
                continue
            if len(station_gwl) < 2:
                drop_reasons["station_gwl_too_short"] += len(group)
                continue
            if station_dynamic is None:
                drop_reasons["no_station_dynamic"] += len(group)
                continue
            if station_static is None:
                drop_reasons["no_station_static"] += len(group)
                continue

            total_after_station_filter += len(group)
            bounds = (
                station_gwl_bounds.get(station_code) if station_gwl_bounds else None
            )

            # Candidate current_dates must be REAL observations only.
            # When interpolate_lookback_gwl=True, group has is_real flags;
            # filter them. When off, all rows are real (no flag) — pass through.
            if "is_real" in group.columns:
                candidate_dates = group.loc[group["is_real"], "date"].values
            else:
                candidate_dates = group["date"].values

            work_items.append(
                (
                    self.config,
                    station_code,
                    station_gwl,
                    station_dynamic,
                    station_static,
                    candidate_dates,
                    bounds,
                    is_training,
                )
            )

        print(f"Stations passing filters: {len(work_items)} / {total_stations}")

        # --- Process stations (parallel or sequential) ---
        if use_parallel:
            print(f"Processing with {n_workers} workers...")
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_station_worker, item): i
                    for i, item in enumerate(work_items)
                }
                completed = 0
                for future in as_completed(futures):
                    station_samples, station_drops = future.result()
                    samples.extend(station_samples)
                    # Aggregate drop reasons
                    for k, v in station_drops.items():
                        if k == "sequence_incomplete_details":
                            drop_reasons[k].extend(v)
                        else:
                            drop_reasons[k] += v
                    completed += 1
                    if completed % 200 == 0:
                        print(f"  Completed {completed}/{len(work_items)} stations...")
        else:
            print("Processing sequentially (use --workers N to parallelize)...")
            for idx, item in enumerate(work_items):
                if (idx + 1) % 200 == 0:
                    print(f"  Processing station {idx + 1}/{len(work_items)}...")
                station_samples, station_drops = _process_station_worker(item)
                samples.extend(station_samples)
                for k, v in station_drops.items():
                    if k == "sequence_incomplete_details":
                        drop_reasons[k].extend(v)
                    else:
                        drop_reasons[k] += v

        # Print drop reason summary
        station_level_drops = (
            drop_reasons["no_station_gwl"]
            + drop_reasons["station_gwl_too_short"]
            + drop_reasons["no_station_dynamic"]
            + drop_reasons["no_station_static"]
        )
        sample_level_drops = (
            drop_reasons["no_current_gwl"]
            + drop_reasons["no_target_gwl"]
            + drop_reasons["current_gwl_outlier"]
            + drop_reasons["target_gwl_outlier"]
            + drop_reasons["sequence_incomplete"]
        )
        total_dropped = station_level_drops + sample_level_drops

        print(f"\n--- SAMPLE DROP ANALYSIS ---")
        print(f"Total dates in range:           {total_dates_in_range}")
        print(f"After station-level filters:    {total_after_station_filter}")
        print(f"Samples created successfully:   {len(samples)}")
        print(f"Total dropped:                  {total_dropped}")

        print(f"\nStation-level drops (subtotal): {station_level_drops}")
        print(f"  - No station GWL data:        {drop_reasons['no_station_gwl']}")
        print(
            f"  - Station GWL too short (<2): {drop_reasons['station_gwl_too_short']}"
        )
        print(f"  - No station dynamic data:    {drop_reasons['no_station_dynamic']}")
        print(f"  - No station static data:     {drop_reasons['no_station_static']}")

        print(f"\nSample-level drops (subtotal):  {sample_level_drops}")
        print(f"  - No current GWL found:       {drop_reasons['no_current_gwl']}")
        print(f"  - No target GWL found:        {drop_reasons['no_target_gwl']}")
        print(f"  - Current GWL outlier:        {drop_reasons['current_gwl_outlier']}")
        print(f"  - Target GWL outlier:         {drop_reasons['target_gwl_outlier']}")
        print(
            f"  - Sequence incomplete (<{self.config.min_sequence_completeness:.0%}): {drop_reasons['sequence_incomplete']}"
        )

        expected_created = total_dates_in_range - total_dropped
        if expected_created != len(samples):
            print(
                f"\n  WARNING: Math doesn't add up! Expected {expected_created}, got {len(samples)}"
            )

        if drop_reasons["sequence_incomplete_details"]:
            details = drop_reasons["sequence_incomplete_details"][:5]
            print(f"\n  Sample incomplete sequence details (first {len(details)}):")
            for d in details:
                print(
                    f"    - {d['num_present']}/{d['num_timesteps']} present ({d['ratio']:.1%})"
                )

        print(f"--- END DROP ANALYSIS ---\n")

        # ── Drop stations with too few samples ────────────────────────────────
        min_station_samples = self.config.min_station_samples
        if min_station_samples > 0:
            station_buckets = defaultdict(list)
            for s in samples:
                station_buckets[s["station_code"]].append(s)

            kept, dropped_stations = [], 0
            for code, station_samples in station_buckets.items():
                if len(station_samples) >= min_station_samples:
                    kept.extend(station_samples)
                else:
                    dropped_stations += 1

            print(
                f"Min-samples filter (>= {min_station_samples}): "
                f"dropped {dropped_stations} stations "
                f"({len(samples) - len(kept)} samples removed), "
                f"{len(station_buckets) - dropped_stations} stations kept"
            )
            samples = kept

        print(f"Created {len(samples)} samples")
        return samples

    def prepare_all_data(self, shared_artifact_dir: str = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Prepare train, validation, and test datasets.

        Pipeline:
        1. Load raw data from database
        2. Compute per-station GWL bounds from training-period data (if enabled)
        3. Create train/val/test samples with delta encoding + outlier detection

        Args:
            shared_artifact_dir: If set, saves a shared dataset pickle here after
                                splitting (before sequence construction), for use
                                by both LSTM and MLP pipelines.

        Returns:
            (train_samples, val_samples, test_samples)
        """
        self.connect_db()

        try:
            # print("Loading raw data from database...")
            print("Loading raw data from CSV...")
            gwl_df = self.load_gwl_readings()
            dynamic_df = self.load_dynamic_features()
            static_df = self.load_static_features()

            print(
                f"Loaded {len(gwl_df)} GWL readings from {len(gwl_df['station_code'].unique())} stations"
            )
            print(f"Loaded {len(dynamic_df)} dynamic feature records")
            print(f"Loaded {len(static_df)} station metadata records")

            # Per-station raw-observation mode-gap (days between consecutive
            # readings, most-common value). Used by the optional
            # min_station_mode_gap_days filter and persisted for the eval-time
            # predictable cohort lookup. Computed BEFORE any filter so the
            # canonical map covers every station with ≥2 raw observations.
            # IMPORTANT: computed on RAW observations (before optional dense
            # interpolation below) — the mode-gap reflects real cadence, not
            # the dense daily grid.
            print("Computing per-station raw-observation mode-gap …")
            _gwl_sorted = gwl_df[["station_code", "date"]].dropna() \
                .sort_values(["station_code", "date"])
            _gwl_sorted["_gap"] = _gwl_sorted.groupby("station_code")["date"].diff().dt.days
            _gwl_sorted = _gwl_sorted.dropna(subset=["_gap"])
            _gwl_sorted = _gwl_sorted[_gwl_sorted["_gap"] > 0]  # drop same-day dupes
            self.station_to_mode_gap_days = (
                _gwl_sorted.groupby("station_code")["_gap"]
                           .agg(lambda s: float(s.value_counts().index[0]) if len(s) else float("nan"))
                           .to_dict()
            )
            print(f"  Built mode-gap map for {len(self.station_to_mode_gap_days)} stations")

            # Optional: replace gwl_df with daily-dense linearly-interpolated
            # version per station. Adds an `is_real` boolean column so:
            #   • sample-grounding (current/target dates) filters to is_real=True
            #   • lookback-timestep input feature accepts any in-range date
            # Default off → byte-exact pre-existing behaviour.
            if self.config.interpolate_lookback_gwl:
                print("Building dense daily-interpolated GWL series per station …")
                _before_rows = len(gwl_df)
                _before_stations = gwl_df["station_code"].nunique()
                gwl_df = self.build_interpolated_gwl_df(gwl_df)
                print(
                    f"  Dense gwl_df: {_before_rows} → {len(gwl_df)} rows "
                    f"({_before_stations} stations; "
                    f"{(gwl_df['is_real'] == False).sum()} interpolated rows)"
                )

            print(
                f"Configuration: gap_days={self.config.gap_days}, forward_fill={self.config.forward_fill_features}"
            )

            # ── Split stations/dates ──────────────────────────────────────
            strategy = self.config.split_strategy

            if strategy == "district":
                print("\nSplit strategy: district-based (station split)")
                train_stations, val_stations, test_stations = split_by_district(gwl_df)

                train_gwl_df     = gwl_df[gwl_df["station_code"].isin(train_stations)]
                train_dynamic_df = dynamic_df[dynamic_df["station_code"].isin(train_stations)]
                train_static_df  = static_df[static_df["station_code"].isin(train_stations)]

                val_gwl_df       = gwl_df[gwl_df["station_code"].isin(val_stations)]
                val_dynamic_df   = dynamic_df[dynamic_df["station_code"].isin(val_stations)]
                val_static_df    = static_df[static_df["station_code"].isin(val_stations)]

                test_gwl_df      = gwl_df[gwl_df["station_code"].isin(test_stations)]
                test_dynamic_df  = dynamic_df[dynamic_df["station_code"].isin(test_stations)]
                test_static_df   = static_df[static_df["station_code"].isin(test_stations)]

                # All splits use the full date range of their own station data
                lookback_months = self.config.lookback_total_months
                earliest_date   = gwl_df["date"].min()
                full_start_str  = (earliest_date + relativedelta(months=lookback_months)).strftime("%Y-%m-%d")
                full_end_str    = gwl_df["date"].max().strftime("%Y-%m-%d")

                train_date_range = (full_start_str, full_end_str)
                val_date_range   = (full_start_str, full_end_str)
                test_date_range  = (full_start_str, full_end_str)

            elif strategy == "time":
                print("\nSplit strategy: time-based (date split)")
                # Stations can overlap across splits; date range separates them
                test_end_date = gwl_df["date"].max().strftime("%Y-%m-%d")
                train_stations, val_stations, test_stations = split_by_date(
                    gwl_df,
                    train_end_date=self.config.train_end_date,
                    val_start_date=self.config.val_start_date,
                    val_end_date=self.config.val_end_date,
                    test_start_date=self.config.test_start_date,
                    test_end_date=test_end_date,
                )

                # Time split: all stations in each split, filtered to their date window
                train_gwl_df     = gwl_df
                train_dynamic_df = dynamic_df
                train_static_df  = static_df

                val_gwl_df       = gwl_df
                val_dynamic_df   = dynamic_df
                val_static_df    = static_df

                test_gwl_df      = gwl_df
                test_dynamic_df  = dynamic_df
                test_static_df   = static_df

                lookback_months  = self.config.lookback_total_months
                earliest_date    = gwl_df["date"].min()
                time_start_str   = (earliest_date + relativedelta(months=lookback_months)).strftime("%Y-%m-%d")

                train_date_range = (time_start_str, self.config.train_end_date)
                val_date_range   = (self.config.val_start_date, self.config.val_end_date)
                test_date_range  = (self.config.test_start_date, test_end_date)

            elif strategy == "station_time":
                t_pct = int(self.config.station_train_frac * 100)
                v_pct = int(self.config.station_val_frac * 100)
                te_pct = 100 - t_pct - v_pct
                print(f"\nSplit strategy: per-station chronological "
                      f"({t_pct}/{v_pct}/{te_pct})")
                # All stations, full date range — split happens after sample creation
                all_stations = set(gwl_df["station_code"].unique())
                train_stations = val_stations = test_stations = all_stations

                lookback_months = self.config.lookback_total_months
                earliest_date   = gwl_df["date"].min()
                full_start_str  = (earliest_date + relativedelta(months=lookback_months)).strftime("%Y-%m-%d")
                full_end_str    = gwl_df["date"].max().strftime("%Y-%m-%d")
                full_date_range = (full_start_str, full_end_str)

            else:
                raise ValueError(f"Unknown split_strategy: {strategy}")

            # Save shared artifact before any model-specific processing
            if shared_artifact_dir:
                if strategy == "district":
                    date_ranges = None
                elif strategy == "time":
                    date_ranges = {
                        "train": train_date_range,
                        "val":   val_date_range,
                        "test":  test_date_range,
                    }
                else:  # station_time
                    date_ranges = None
                self.save_shared_artifact(
                    gwl_df, dynamic_df, static_df,
                    train_stations, val_stations, test_stations,
                    shared_artifact_dir,
                    date_ranges=date_ranges,
                )

            # ── Per-station GWL bounds (outlier detection) ────────────────
            gwl_bounds = None
            if self.config.validate_station_bounds:
                print("\n" + "=" * 70)
                print("Computing per-station GWL bounds...")
                print("=" * 70)
                if strategy == "station_time":
                    # Single set of bounds from all data
                    gwl_bounds = self.compute_station_gwl_bounds(gwl_df)
                else:
                    gwl_bounds = None  # computed per-split below

            # ── Create datasets (gap-day filtering + sequence validation) ─
            if strategy == "station_time":
                # Create all samples at once, then split per-station
                print("\n" + "=" * 70)
                print("Creating samples for all stations (full date range)...")
                print("=" * 70)
                all_samples = self.create_dataset(
                    gwl_df, dynamic_df, static_df,
                    *full_date_range,
                    station_gwl_bounds=gwl_bounds,
                    is_training=True,
                )

                t_frac = self.config.station_train_frac
                v_frac = self.config.station_val_frac
                te_frac = round(1.0 - t_frac - v_frac, 2)
                print("\n" + "=" * 70)
                print(f"Splitting samples per-station chronologically "
                      f"({int(t_frac*100)}/{int(v_frac*100)}/{int(te_frac*100)})...")
                print("=" * 70)
                train_samples, val_samples, test_samples = split_samples_per_station(
                    all_samples, train_frac=t_frac, val_frac=v_frac
                )
            else:
                # District or time split: create datasets per-split
                train_gwl_bounds = None
                val_gwl_bounds   = None
                test_gwl_bounds  = None
                if self.config.validate_station_bounds:
                    if strategy == "district":
                        train_gwl_bounds = self.compute_station_gwl_bounds(train_gwl_df)
                        val_gwl_bounds   = self.compute_station_gwl_bounds(val_gwl_df)
                        test_gwl_bounds  = self.compute_station_gwl_bounds(test_gwl_df)
                    else:  # time
                        train_gwl_bounds = self.compute_station_gwl_bounds(train_gwl_df)
                        val_gwl_bounds   = self.compute_station_gwl_bounds(val_gwl_df)
                        test_gwl_bounds  = self.compute_station_gwl_bounds(test_gwl_df)

                print("\n" + "=" * 70)
                print(f"Creating training dataset ({strategy}-based split)...")
                print("=" * 70)
                train_samples = self.create_dataset(
                    train_gwl_df, train_dynamic_df, train_static_df,
                    *train_date_range,
                    station_gwl_bounds=train_gwl_bounds,
                    is_training=True,
                )

                print("\n" + "=" * 70)
                print(f"Creating validation dataset ({strategy}-based split)...")
                print("=" * 70)
                val_samples = self.create_dataset(
                    val_gwl_df, val_dynamic_df, val_static_df,
                    *val_date_range,
                    station_gwl_bounds=val_gwl_bounds,
                    is_training=False,
                )

                print("\n" + "=" * 70)
                print(f"Creating test dataset ({strategy}-based split)...")
                print("=" * 70)
                test_samples = self.create_dataset(
                    test_gwl_df, test_dynamic_df, test_static_df,
                    *test_date_range,
                    station_gwl_bounds=test_gwl_bounds,
                    is_training=False,
                )

            # ── Capture train samples BEFORE flat filter ─────────────────────
            # Static features (gwl_mean, gwl_amplitude, gwl_anomaly, delta_mean,
            # delta_std, etc.) are computed once per station from its own
            # train-period observations and reused at val/test inference time.
            # We capture the pre-filter set so flat stations excluded from
            # training (loss) still get *their own* station-specific stats —
            # instead of falling back to a generic state-level mean. This
            # decouples "drop from loss" from "drop from feature attribution",
            # which materially improves prediction calibration on flat wells.
            train_codes_for_stats = {s["station_code"] for s in train_samples}
            train_samples_for_stats = list(train_samples)

            # ── Pre-thinning: bucket / flat-station filter ──────────────────
            # min_station_target_std drops "flat" wells (std < min). With a
            # finite max_station_target_std, a STD bucket window is applied —
            # useful for multi-model pipelines where each bucket trains its
            # own model.
            #
            # Per-station bucket assignment is computed ONCE from TRAIN std
            # and applied consistently to all splits. This avoids the case
            # where a station's train_std and val_std (or test_std) land on
            # different sides of the cutoff due to sampling variance, which
            # would put the same station in different buckets across splits.
            #
            # Test handling:
            #   - max == inf (single-model): test is NOT filtered. Preserves
            #     the cold-start metric ("how does a predictable-trained model
            #     fare on flat wells?").
            #   - max < inf (multi-model): test is filtered with the same
            #     train-derived membership. Each bucket model evaluates only
            #     on its own bucket's stations.
            #
            # Stations with no train samples (e.g., district split where
            # train/val/test are disjoint) fall back to per-split filtering.
            _min_std = self.config.min_station_target_std
            _max_std = self.config.max_station_target_std
            _filter_active = (_min_std > 0) or (_max_std != float("inf"))
            if _filter_active:
                _range_str = (
                    f"[{_min_std}m, {_max_std}m)"
                    if _max_std != float("inf") else f">= {_min_std}m"
                )
                print(f"\n{'=' * 70}")
                print(f"Filtering stations by target-delta std (keep {_range_str})")
                print(f"{'=' * 70}")

                # Step 1: derive per-station bucket membership from TRAIN std
                train_in_bucket = compute_in_bucket_stations(
                    train_samples, _min_std, _max_std
                )
                train_stations_all = {s["station_code"] for s in train_samples}
                print(
                    f"  Train-derived membership: {len(train_in_bucket)} of "
                    f"{len(train_stations_all)} train stations in bucket."
                )

                # Step 2: apply membership to TRAIN
                print("Train:")
                train_samples = filter_samples_by_station_set(
                    train_samples, train_in_bucket, "    train"
                )

                # Step 3: apply same membership to VAL. For val stations not
                # present in train (district split), fall back to per-split std.
                print("Val:")
                val_stations = {s["station_code"] for s in val_samples}
                val_orphans = val_stations - train_stations_all
                if val_orphans:
                    print(
                        f"    {len(val_orphans)} val stations have no train "
                        f"samples — using per-split val_std fallback for those."
                    )
                    val_orphan_samples = [
                        s for s in val_samples if s["station_code"] in val_orphans
                    ]
                    val_orphan_in_bucket = compute_in_bucket_stations(
                        val_orphan_samples, _min_std, _max_std
                    )
                    val_membership = train_in_bucket | val_orphan_in_bucket
                else:
                    val_membership = train_in_bucket
                val_samples = filter_samples_by_station_set(
                    val_samples, val_membership, "    val"
                )

                # Step 4: TEST — bucket-mode only.
                if _max_std != float("inf"):
                    print("Test (bucket mode — same train-derived membership):")
                    test_stations = {s["station_code"] for s in test_samples}
                    test_orphans = test_stations - train_stations_all
                    if test_orphans:
                        print(
                            f"    {len(test_orphans)} test stations have no "
                            f"train samples — using per-split test_std fallback."
                        )
                        test_orphan_samples = [
                            s for s in test_samples
                            if s["station_code"] in test_orphans
                        ]
                        test_orphan_in_bucket = compute_in_bucket_stations(
                            test_orphan_samples, _min_std, _max_std
                        )
                        test_membership = train_in_bucket | test_orphan_in_bucket
                    else:
                        test_membership = train_in_bucket
                    test_samples = filter_samples_by_station_set(
                        test_samples, test_membership, "    test"
                    )
                else:
                    print("Test: (kept all — single-model cold-start view)")

            # ── State whitelist filter (additive to std filter) ──────────────
            # When include_states is non-empty, drop train+val samples whose
            # state is NOT in the whitelist. Test stays unfiltered (cold-start).
            # Both filters compose by AND — a sample must pass std AND state.
            _include_states = list(self.config.include_states or [])
            if _include_states:
                _states_set = set(_include_states)
                # Validate against actual states present in the data — catches
                # typos / case mismatches before they silently drop everything.
                _all_train_states = {s.get("state") for s in train_samples if s.get("state")}
                _unknown = _states_set - _all_train_states
                _matched = _states_set & _all_train_states
                print(f"\n{'=' * 70}")
                print(f"Filtering stations by state whitelist (n={len(_states_set)})")
                print(f"  Requested: {sorted(_states_set)}")
                if _unknown:
                    # Suggest near matches by case-insensitive contains/equals
                    _suggest = {}
                    _lc_actual = {a.lower(): a for a in _all_train_states}
                    for u in _unknown:
                        if u.lower() in _lc_actual:
                            _suggest[u] = f"'{_lc_actual[u.lower()]}' (case mismatch)"
                        else:
                            # any actual that contains the typo as substring (or vice versa)
                            cands = [a for a in _all_train_states
                                     if u.lower() in a.lower() or a.lower() in u.lower()]
                            if cands:
                                _suggest[u] = f"did you mean {cands[:3]}?"
                    raise ValueError(
                        f"INCLUDE_STATES contains states not in train data: "
                        f"{sorted(_unknown)}. "
                        f"{('Suggestions: ' + str(_suggest)) if _suggest else ''} "
                        f"Available states: {sorted(_all_train_states)}"
                    )
                print(f"  Matched (in data): {sorted(_matched)}")
                print(f"{'=' * 70}")
                _before_train = len(train_samples)
                _before_val   = len(val_samples)
                train_samples = [s for s in train_samples if s.get("state") in _states_set]
                val_samples   = [s for s in val_samples   if s.get("state") in _states_set]
                _t_stations = len({s["station_code"] for s in train_samples})
                _v_stations = len({s["station_code"] for s in val_samples})
                print(f"  train: {_before_train} → {len(train_samples)} samples ({_t_stations} stations)")
                print(f"  val:   {_before_val} → {len(val_samples)} samples ({_v_stations} stations)")
                print(f"  test: kept all {len(test_samples)} samples (cold-start view)")

            # ── Mode-gap filter (drop densely-sampled stations from train+val) ─
            # Composes with std + state filters by AND. Test stays unfiltered.
            _min_mode_gap = float(self.config.min_station_mode_gap_days or 0.0)
            if _min_mode_gap > 0:
                print(f"\n{'=' * 70}")
                print(f"Filtering stations by raw-observation mode-gap ≥ {_min_mode_gap:g} days")
                print(f"{'=' * 70}")
                _mg = self.station_to_mode_gap_days
                def _passes_mode_gap(code):
                    v = _mg.get(code)
                    return (v is not None) and (v >= _min_mode_gap)
                _before_train = len(train_samples)
                _before_val   = len(val_samples)
                train_samples = [s for s in train_samples if _passes_mode_gap(s.get("station_code"))]
                val_samples   = [s for s in val_samples   if _passes_mode_gap(s.get("station_code"))]
                _t_stations = len({s["station_code"] for s in train_samples})
                _v_stations = len({s["station_code"] for s in val_samples})
                print(f"  train: {_before_train} → {len(train_samples)} samples ({_t_stations} stations)")
                print(f"  val:   {_before_val} → {len(val_samples)} samples ({_v_stations} stations)")
                print(f"  test: kept all {len(test_samples)} samples (cold-start view)")

            # ── Post-processing: thin samples at min_sample_freq_days ───────
            # Reduces imbalance from heavily-sampled stations. Per-split
            # overrides allow e.g. keeping train un-thinned (more training
            # data + perwell loss handles imbalance) while still thinning
            # val/test for cohort stability.
            def _resolve_freq(per_split):
                return per_split if per_split >= 0 else self.config.min_sample_freq_days
            train_freq = _resolve_freq(self.config.min_sample_freq_days_train)
            val_freq   = _resolve_freq(self.config.min_sample_freq_days_val)
            test_freq  = _resolve_freq(self.config.min_sample_freq_days_test)
            if any(f and f > 0 for f in (train_freq, val_freq, test_freq)):
                print(f"\n{'=' * 70}")
                print(f"Thinning samples (per-split): "
                      f"train={train_freq}d  val={val_freq}d  test={test_freq}d")
                print(f"{'=' * 70}")
                if train_freq and train_freq > 0:
                    print("Train:")
                    train_samples = thin_samples(train_samples, train_freq)
                else:
                    print(f"Train: thinning disabled (kept {len(train_samples)} samples)")
                if val_freq and val_freq > 0:
                    print("Val:")
                    val_samples = thin_samples(val_samples, val_freq)
                else:
                    print(f"Val: thinning disabled (kept {len(val_samples)} samples)")
                if test_freq and test_freq > 0:
                    print("Test:")
                    test_samples = thin_samples(test_samples, test_freq)
                else:
                    print(f"Test: thinning disabled (kept {len(test_samples)} samples)")

            # ── Cap val and test to max_samples_per_station_eval per station ──
            # Ensures no single station dominates evaluation metrics.
            if (
                self.config.max_samples_per_station_eval
                and self.config.max_samples_per_station_eval > 0
            ):
                print(f"\n{'=' * 70}")
                print(
                    f"Capping val/test samples per station @ {self.config.max_samples_per_station_eval}"
                )
                print(f"{'=' * 70}")
                print("Val:")
                val_samples = cap_samples_per_station(
                    val_samples, self.config.max_samples_per_station_eval
                )
                print("Test:")
                test_samples = cap_samples_per_station(
                    test_samples, self.config.max_samples_per_station_eval
                )

            print(f"\nFinal sample counts: "
                  f"train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")

            # ── Compute & attach derived per-station static features ─────────
            # Done once from train data; the same value is used for all samples
            # of a station (val/test included) to avoid leakage from val/test.
            print(f"\n{'=' * 70}")
            print("Computing per-station static features from train data...")
            print(f"{'=' * 70}")
            # Use the pre-filter station set so flat stations still receive
            # their own station-specific static features (see capture above).
            train_codes = train_codes_for_stats

            # Determine train-period gwl/dynamic dataframes for stat computation
            # (avoids leakage from val/test periods in time/station_time splits).
            if strategy == "district":
                train_gwl_for_stats = gwl_df[gwl_df["station_code"].isin(train_codes)]
                train_dyn_for_stats = dynamic_df[dynamic_df["station_code"].isin(train_codes)]
            elif strategy == "time":
                train_end = pd.Timestamp(self.config.train_end_date)
                train_gwl_for_stats = gwl_df[
                    (gwl_df["station_code"].isin(train_codes)) & (gwl_df["date"] <= train_end)
                ]
                train_dyn_for_stats = dynamic_df[dynamic_df["date"] <= train_end]
            else:  # station_time: per-station chronological boundary
                # Per-station train cutoff = each station's latest train sample
                # current_date. Pooling a single global max across all stations
                # leaks each station's val/test gwl/dynamic into its own static
                # stats (long-history stations are most affected).
                per_station_cutoff: Dict[str, pd.Timestamp] = {}
                for s in train_samples:
                    code = s["station_code"]
                    d = pd.Timestamp(s["current_date"])
                    if code not in per_station_cutoff or d > per_station_cutoff[code]:
                        per_station_cutoff[code] = d

                if per_station_cutoff:
                    cutoff_df = pd.DataFrame(
                        list(per_station_cutoff.items()),
                        columns=["station_code", "_cutoff"],
                    )
                    gwl_merged = gwl_df.merge(cutoff_df, on="station_code", how="inner")
                    train_gwl_for_stats = (
                        gwl_merged[gwl_merged["date"] <= gwl_merged["_cutoff"]]
                        .drop(columns=["_cutoff"])
                        .reset_index(drop=True)
                    )
                    dyn_merged = dynamic_df.merge(cutoff_df, on="station_code", how="inner")
                    train_dyn_for_stats = (
                        dyn_merged[dyn_merged["date"] <= dyn_merged["_cutoff"]]
                        .drop(columns=["_cutoff"])
                        .reset_index(drop=True)
                    )
                else:
                    train_gwl_for_stats = gwl_df[gwl_df["station_code"].isin(train_codes)]
                    train_dyn_for_stats = dynamic_df[dynamic_df["station_code"].isin(train_codes)]

            station_static_features = compute_station_static_features(
                train_codes, train_gwl_for_stats, train_dyn_for_stats,
            )
            station_month_gwl = compute_station_month_gwl_means(
                train_codes, train_gwl_for_stats,
            )

            # Per-station delta stats (target - current) from PRE-FILTER train
            # samples — directly aligned with the prediction target. Tells the
            # model "this well's typical 3-month delta is ~mean ± std".
            delta_stats = compute_station_delta_stats(train_samples_for_stats)
            for code, ds in delta_stats.items():
                if code in station_static_features:
                    station_static_features[code].update(ds)

            # Canonical per-station train_std map: single source of truth for
            # std across the pipeline (eval predictable filter, plot bucketing,
            # CSV columns). Includes ONLY stations with real train_std
            # (compute_station_delta_stats requires ≥2 train samples). Stations
            # missing from this map are "Scenario B" cold-start — excluded from
            # predictable cohort and tagged [no train-std] in plots.
            self.station_to_train_std = {
                code: float(ds["station_delta_std"])
                for code, ds in delta_stats.items()
            }

            # Build station→state map from PRE-FILTER train samples (so flat-
            # filtered stations can still be mapped to their state for fallback).
            station_to_state = {}
            for s in train_samples_for_stats:
                station_to_state.setdefault(s["station_code"], s.get("state", "unknown"))
            state_fallbacks = compute_state_level_fallbacks(
                station_static_features, station_to_state,
            )
            global_fallback = {
                "mean_annual_rainfall": float(np.mean(
                    [v.get("mean_annual_rainfall", 0.0) for v in station_static_features.values()] or [0.0]
                )),
                "annual_rainfall_std": float(np.mean(
                    [v.get("annual_rainfall_std", 0.0) for v in station_static_features.values()] or [0.0]
                )),
                "station_mean_gwl": float(np.mean(
                    [v.get("station_mean_gwl", 0.0) for v in station_static_features.values()] or [0.0]
                )),
                "station_gwl_amplitude": float(np.mean(
                    [v.get("station_gwl_amplitude", 0.0) for v in station_static_features.values()] or [0.0]
                )),
                "station_delta_mean": float(np.mean(
                    [v.get("station_delta_mean", 0.0) for v in station_static_features.values()] or [0.0]
                )),
                "station_delta_std": float(np.mean(
                    [v.get("station_delta_std", 0.0) for v in station_static_features.values()] or [0.0]
                )),
            }
            print(f"  Computed for {len(station_static_features)} train stations")
            print(f"  (gwl_anomaly fallbacks: state-mean → global-mean for unseen wells)")

            # Annotate ALL samples (train + val + test) with derived static features + gwl_anomaly
            # Annotate ALL samples via the shared helper (same fill inference uses).
            for split_samples_list in (train_samples, val_samples, test_samples):
                for s in split_samples_list:
                    fill_derived_static_features(
                        s, station_static_features, state_fallbacks,
                        global_fallback, station_month_gwl,
                    )

            # ── Prithvi tile_idx annotation (Step 2) ──
            # Resolve each sample → integer tile_idx pointing at a pre-downloaded
            # HLS composite. Done here in the MAIN thread (post-collection), NOT
            # in _process_station_worker, so the tile index isn't shipped to
            # every worker process. tile_idx depends only on (station_code,
            # current_date) → split/thinning-agnostic, safe to annotate now.
            if self.config.use_prithvi:
                if not self.config.composite_dir:
                    raise ValueError("use_prithvi=True but composite_dir is empty")
                period = self.config.composite_period
                key_to_row, ordered_files, zero_idx, min_year = build_tile_index(
                    self.config.composite_dir, period
                )
                safe_id_map = load_station_index(self.config.station_index_csv)
                print(
                    f"\nAnnotating tile_idx (Prithvi, period={period}): "
                    f"{len(ordered_files)} composites, zero_idx={zero_idx}, "
                    f"min_year={min_year}, station_index entries={len(safe_id_map)}"
                )
                n_resolved = n_zero = 0
                fallback_codes = set()
                for split_samples_list in (train_samples, val_samples, test_samples):
                    for s in split_samples_list:
                        code = str(s["station_code"])
                        safe_id = safe_id_map.get(code)
                        if safe_id is None:
                            safe_id = sanitize_safe_id(code)
                            fallback_codes.add(code)
                        tidx = resolve_tile_idx(
                            safe_id, s["current_date"],
                            key_to_row, zero_idx, period, min_year,
                        )
                        s["tile_idx"] = tidx
                        if tidx == zero_idx:
                            n_zero += 1
                        else:
                            n_resolved += 1
                total = n_resolved + n_zero
                pct = (100.0 * n_resolved / total) if total else 0.0
                print(
                    f"  tile_idx: {n_resolved}/{total} resolved ({pct:.1f}%), "
                    f"{n_zero} → zero_idx (no composite); "
                    f"{len(fallback_codes)} station(s) used sanitized safe_id fallback"
                )
                if total and n_resolved == 0:
                    print(
                        "  [WARN] 0 samples resolved to a real tile — likely a "
                        "safe_id mismatch between samples and composite filenames "
                        "(check --station-index-csv / --composite-dir)."
                    )
                # Manifest for train.py's TileStore (row → basename). Saved by
                # save_datasets(); cached on self until then.
                self._tile_manifest = {
                    "period": period,
                    "composite_dir": self.config.composite_dir,
                    "zero_idx": zero_idx,
                    "min_year": min_year,
                    "period_doy": PERIOD_DOY[period],
                    "ordered_files": ordered_files,  # row index → basename
                }

            # Stash references so write_sample_audit() can re-derive the
            # per-timestep table (matched_date, offset, raw_gwl) without
            # re-loading from the CSV.
            self._audit_gwl_df = gwl_df
            self._audit_dynamic_df = dynamic_df

            return train_samples, val_samples, test_samples

        finally:
            self.close_db()

    def write_sample_audit(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        output_dir: str,
    ) -> None:
        """Write one plain-text audit per split (sample[0]) to <output_dir>/debug/.

        Re-derives the per-timestep table (grid_date, matched_date, offset_d,
        raw_gwl) from the saved gwl_df reference, alongside every other
        per-timestep field read directly from the sample's stored sequence.
        Lets a human verify exactly how a sample was assembled.
        """
        debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        for split_name, samples in (("train", train_samples),
                                    ("val",   val_samples),
                                    ("test",  test_samples)):
            if not samples:
                print(f"[audit] {split_name}: empty split, skipping")
                continue
            sample = samples[0]
            try:
                txt = self._render_sample_audit(sample, split_name)
            except Exception as e:
                txt = f"[audit] {split_name}: render failed: {e}\n"
            out_path = os.path.join(debug_dir, f"sample_audit_{split_name}.txt")
            with open(out_path, "w") as f:
                f.write(txt)
            print(f"[audit] wrote {out_path} ({len(txt)} bytes)")

    def _render_sample_audit(self, sample: Dict, split_name: str) -> str:
        """Build the plain-text audit string for one sample."""
        cfg = self.config
        code = sample["station_code"]
        cur_date = sample["current_date"]
        tgt_date = sample["target_date"]
        cur_gwl = float(sample["current_gwl"])
        tgt_raw = float(sample["target_gwl_raw"])
        # In delta mode sample["target_gwl"] is the delta label;
        # in absolute mode it's the absolute target.
        label = float(sample["target_gwl"])
        target_delta = tgt_raw - cur_gwl  # always-true delta

        horizon_d = (pd.Timestamp(tgt_date) - pd.Timestamp(cur_date)).days

        train_std_map = getattr(self, "station_to_train_std", {}) or {}
        mode_gap_map = getattr(self, "station_to_mode_gap_days", {}) or {}
        train_std = train_std_map.get(code)
        mode_gap = mode_gap_map.get(code)

        sequence = np.asarray(sample["sequence"])  # (T, F)
        T, F = sequence.shape
        num_timesteps = cfg.num_timesteps
        window_days = cfg.lookback_window_days

        # Re-run the lookback grid to recover matched_date / offset / raw_gwl.
        # Walk in NEWEST-FIRST order (matches create_sequence_for_sample's
        # internal loop), then reverse so output rows match the chronological
        # sequence stored in the sample.
        gwl_df = getattr(self, "_audit_gwl_df", None)
        if gwl_df is None:
            raise RuntimeError(
                "Audit requires self._audit_gwl_df, set in prepare_all_data."
            )
        station_gwl = (
            gwl_df[gwl_df["station_code"] == code]
                .set_index("date").sort_index()
        )
        _interp_on = cfg.interpolate_lookback_gwl
        _lookback_gap = 0 if _interp_on else cfg.gap_days
        _lookback_real_only = (not _interp_on)

        timestep_date = cur_date
        recovered = []  # newest-first
        for _ in range(num_timesteps):
            hit = self.find_gwl_value_with_gap(
                station_gwl, timestep_date,
                gap_days=_lookback_gap, real_only=_lookback_real_only,
            )
            if hit is not None:
                gwl_v, matched_d = hit
                offset = (pd.Timestamp(matched_d) - pd.Timestamp(timestep_date)).days
                recovered.append({
                    "grid_date": pd.Timestamp(timestep_date),
                    "matched_date": pd.Timestamp(matched_d),
                    "offset_d": int(offset),
                    "raw_gwl": float(gwl_v),
                    "found": True,
                })
            else:
                recovered.append({
                    "grid_date": pd.Timestamp(timestep_date),
                    "matched_date": None,
                    "offset_d": None,
                    "raw_gwl": None,
                    "found": False,
                })
            timestep_date = timestep_date - relativedelta(days=window_days)
        # Reverse to chronological (matches sequence row order)
        recovered.reverse()

        # Sequence column layout (after drop_ndvi_sm + rainfall enrichment):
        # 0:gwl 1:is_present 2:gwl_diff 3:gwl_diff_rel 4:sin 5:cos
        # 6:rain 7:temp 8:et 9:runoff 10:cum_rain 11:rain_anom 12:cum_anom
        col_names = (
            ["gwl_enc", "is_present", "gwl_diff", "gwl_diff_rel",
             "sin", "cos"]
            + (["rain", "temp", "et", "runoff"]
               if cfg.drop_ndvi_sm else
               ["rain", "temp", "et", "runoff", "ndvi", "sm"])
            + ["cum_rain", "rain_anom", "cum_anom"]
        )
        if F != len(col_names):
            col_names = [f"f{i}" for i in range(F)]

        # ---------- Header ----------
        L = []
        L.append(f"=== Sample audit: {split_name.upper()} split, sample 0 ===")
        L.append("")
        L.append("Identity")
        L.append(f"  station_code:           {code}")
        L.append(f"  state / district:       {sample.get('state', '?')} / "
                 f"{sample.get('district', '?')}")
        L.append(f"  current_date:           {cur_date.strftime('%Y-%m-%d')}")
        L.append(f"  target_date:            {tgt_date.strftime('%Y-%m-%d')}  "
                 f"(horizon = {cfg.forecast_horizon_months} months → "
                 f"{horizon_d} days)")
        L.append(f"  current_gwl:            {cur_gwl:.3f} m")
        L.append(f"  target_gwl_raw:         {tgt_raw:.3f} m")
        if cfg.use_delta_gwl:
            L.append(f"  target_delta (label):   {label:+.3f} m  "
                     f"(verify: target_raw - current = {target_delta:+.3f})")
        else:
            L.append(f"  target_gwl  (label):    {label:.3f} m  (absolute mode)")
        L.append("")

        # ---------- Station context ----------
        std_str = (
            f"{train_std:.3f} m   "
            + ("[passes]" if (cfg.min_station_target_std == 0
                              or (train_std is not None
                                  and train_std >= cfg.min_station_target_std))
               else f"[fails MIN_STATION_TARGET_STD={cfg.min_station_target_std}]")
            if train_std is not None else "(no train_std — cold-start station)"
        )
        mg_str = (
            f"{mode_gap:.1f} days   "
            + ("[passes]" if (cfg.min_station_mode_gap_days == 0
                              or (mode_gap is not None
                                  and mode_gap >= cfg.min_station_mode_gap_days))
               else f"[fails MIN_STATION_MODE_GAP_DAYS="
                    f"{cfg.min_station_mode_gap_days}]")
            if mode_gap is not None else "(no mode_gap — <2 raw obs)"
        )
        L.append("Station context")
        L.append(f"  train_std (target_delta):    {std_str}")
        L.append(f"  mode_gap_days (raw obs):     {mg_str}")
        L.append(f"  station_mean_gwl:            {sample.get('station_mean_gwl', 0.0):.3f}")
        L.append(f"  station_gwl_amplitude:       {sample.get('station_gwl_amplitude', 0.0):.3f}")
        L.append(f"  station_delta_mean:          {sample.get('station_delta_mean', 0.0):.3f}")
        L.append(f"  station_delta_std:           {sample.get('station_delta_std', 0.0):.3f}")
        L.append(f"  gwl_anomaly (cur − month-mean): {sample.get('gwl_anomaly', 0.0):+.3f}")
        L.append(f"  mean_annual_rainfall:        {sample.get('mean_annual_rainfall', 0.0):.3f}")
        L.append(f"  annual_rainfall_std:         {sample.get('annual_rainfall_std', 0.0):.3f}")
        L.append("")

        # ---------- Lookback config ----------
        L.append("Lookback config")
        L.append(f"  num_timesteps:        {num_timesteps}")
        L.append(f"  window_days:          {window_days}")
        L.append(f"  lookback_total_days:  {cfg.lookback_total_days}")
        L.append(f"  gap_days (lookback):  {_lookback_gap}  "
                 f"(interpolate={'yes' if _interp_on else 'no'})")
        L.append(f"  real_only:            {_lookback_real_only}")
        L.append("")

        # ---------- Per-timestep table ----------
        L.append("Per-timestep table  (chronological, oldest → newest)")
        L.append("")
        # Build header
        fixed = ["i", "grid_date", "matched", "off_d", "raw_gwl", "is_pres"]
        rest = [c for c in col_names if c not in ("gwl_enc", "is_present")]
        # Use float columns for everything in rest
        widths = {
            "i": 4, "grid_date": 10, "matched": 10, "off_d": 6,
            "raw_gwl": 8, "is_pres": 7,
        }
        for c in rest:
            widths[c] = 9
        header = " ".join(f"{h:>{widths[h]}}" for h in fixed + rest)
        L.append(header)
        L.append("-" * len(header))

        n_present = 0
        n_out_of_range = 0
        distinct_matched = set()
        for i in range(T):
            r = recovered[i]
            row = sequence[i]
            grid_str = r["grid_date"].strftime("%Y-%m-%d")
            if r["found"]:
                matched_str = r["matched_date"].strftime("%Y-%m-%d")
                off_str = f"{r['offset_d']:+d}"
                raw_gwl_str = f"{r['raw_gwl']:8.3f}"
                distinct_matched.add(matched_str)
            else:
                matched_str = "—"
                off_str = "—"
                raw_gwl_str = f"{'—':>8}"
                n_out_of_range += 1
            is_pres_val = float(row[1])
            if is_pres_val == 1.0:
                n_present += 1
            cells = [
                f"{i:>{widths['i']}}",
                f"{grid_str:>{widths['grid_date']}}",
                f"{matched_str:>{widths['matched']}}",
                f"{off_str:>{widths['off_d']}}",
                f"{raw_gwl_str:>{widths['raw_gwl']}}",
                f"{is_pres_val:>{widths['is_pres']}.1f}",
            ]
            # rest in order from col_names skipping gwl_enc,is_present
            rest_idx_map = {c: col_names.index(c) for c in rest}
            for c in rest:
                v = float(row[rest_idx_map[c]])
                cells.append(f"{v:>{widths[c]}.3f}")
            L.append(" ".join(cells))

        L.append("")
        L.append(f"Coverage:               {n_present}/{T} timesteps is_present=1 "
                 f"({100.0*n_present/max(T,1):.1f}%)")
        L.append(f"Distinct matched obs:   {len(distinct_matched)}")
        L.append(f"Out-of-range tail:      {n_out_of_range} timesteps "
                 f"(no obs found within ±gap)")
        L.append("")

        # ---------- Forecast features ----------
        L.append(f"Forecast features  (window [current_date, target_date) = {horizon_d} days)")
        ff = np.asarray(sample["forecast_features"]).astype(float)
        if cfg.use_rain_temp:
            ff_names = ["rainfall_sum", "temp_mean"]
        elif cfg.drop_ndvi_sm:
            ff_names = ["rainfall_sum", "temp_mean", "et_sum", "runoff_sum"]
        else:
            ff_names = ["rainfall_sum", "temp_mean", "et_sum", "runoff_sum",
                        "ndvi_mean", "sm_mean"]
        for name, val in zip(ff_names, ff):
            L.append(f"  {name:<14} {val:.3f}")
        L.append(f"  forecast_lulc:  {sample.get('forecast_lulc', '?')}")
        L.append("")

        # ---------- Static block ----------
        L.append("Static block")
        L.append(f"  well_depth={sample.get('well_depth', 0.0):.3f}  "
                 f"elevation={sample.get('elevation', 0.0):.3f}  "
                 f"stream_order={sample.get('stream_order', 0.0):.3f}")
        L.append(f"  lat={sample.get('latitude', 0.0):.5f}  "
                 f"lon={sample.get('longitude', 0.0):.5f}")
        L.append(f"  gwl_anomaly={sample.get('gwl_anomaly', 0.0):+.3f}")
        L.append(f"  mean_annual_rainfall={sample.get('mean_annual_rainfall', 0.0):.3f}  "
                 f"annual_rainfall_std={sample.get('annual_rainfall_std', 0.0):.3f}")
        L.append(f"  station_mean_gwl={sample.get('station_mean_gwl', 0.0):.3f}  "
                 f"station_gwl_amplitude={sample.get('station_gwl_amplitude', 0.0):.3f}")
        L.append(f"  station_delta_mean={sample.get('station_delta_mean', 0.0):+.3f}  "
                 f"station_delta_std={sample.get('station_delta_std', 0.0):.3f}")
        L.append("")

        # ---------- Categorical ----------
        L.append("Categorical")
        L.append(f"  lithology={sample.get('lithology', '?')}  "
                 f"well_type={sample.get('well_type', '?')}  "
                 f"aquifer_type={sample.get('aquifer_type', '?')}")
        L.append(f"  aquifer_0_aquifer={sample.get('aquifer_0_aquifer', '?')}  "
                 f"litho_supergroup={sample.get('litho_supergroup', '?')}")
        L.append(f"  state={sample.get('state', '?')}  "
                 f"district={sample.get('district', '?')}")
        # Historical LULC distribution
        hl = sample.get("historical_lulc", []) or []
        from collections import Counter
        lulc_counts = Counter(hl)
        L.append(f"  historical_lulc[unique]: {dict(lulc_counts.most_common())}")
        L.append("")
        return "\n".join(L)

    def save_shared_artifact(
        self,
        gwl_df,
        dynamic_df,
        static_df,
        train_stations: set,
        val_stations: set,
        test_stations: set,
        output_dir: str,
        date_ranges: dict = None,
    ) -> str:
        """
        Save a shared dataset artifact for use by both LSTM and MLP pipelines.

        Called after sign-convention cleaning and district split, but before any
        sequence construction, scaling, or model-specific encoding.

        Pickle structure:
            {
              "split":    {"train": set, "val": set, "test": set},
              "stations": {station_code: {"gwl": df, "dynamic": df, "static": dict}},
              "config":   {forecast_horizon_months, lookback_years, ...},
            }

        Returns the path of the saved file.
        """
        os.makedirs(output_dir, exist_ok=True)

        all_stations = train_stations | val_stations | test_stations

        gwl_by_station     = {s: g.reset_index(drop=True) for s, g in gwl_df.groupby("station_code") if s in all_stations}
        dynamic_by_station = {s: g.reset_index(drop=True) for s, g in dynamic_df.groupby("station_code") if s in all_stations}
        static_by_station  = {
            row["station_code"]: row.to_dict()
            for _, row in static_df.drop_duplicates("station_code").iterrows()
            if row["station_code"] in all_stations
        }

        artifact = {
            "split": {
                "train": train_stations,
                "val":   val_stations,
                "test":  test_stations,
            },
            "stations": {
                s: {
                    "gwl":     gwl_by_station.get(s),
                    "dynamic": dynamic_by_station.get(s),
                    "static":  static_by_station.get(s),
                }
                for s in all_stations
            },
            "config": {
                "split_type":                self.config.split_strategy,
                "forecast_horizon_months":   self.config.forecast_horizon_months,
                "lookback_years":            self.config.lookback_years,
                "lookback_months":           self.config.lookback_months,
                "lookback_total_months":     self.config.lookback_total_months,
                "lookback_window_days":      self.config.lookback_window_days,
                "gap_days":                  self.config.gap_days,
                "min_sequence_completeness": self.config.min_sequence_completeness,
                "date_ranges":               date_ranges,  # None for district split, dict for time split
            },
        }

        filename = (
            f"district_split"
            f"_h{self.config.forecast_horizon_months}"
            f"_lk{self.config.lookback_total_months}m"
            f"_gap{self.config.gap_days}"
            f"_v1.pkl"
        )
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(artifact, f)

        print(f"Saved shared artifact → {path}")
        print(f"  train: {len(train_stations)} | val: {len(val_stations)} | test: {len(test_stations)} stations")
        return path

    def save_datasets(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
        output_dir: str = "gwl_lstm/data",
    ):
        """
        Save prepared datasets to disk.
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving datasets to {output_dir}/...")

        with open(f"{output_dir}/train_samples.pkl", "wb") as f:
            pickle.dump(train_samples, f)
        print(f"  - Saved {len(train_samples)} training samples")

        with open(f"{output_dir}/val_samples.pkl", "wb") as f:
            pickle.dump(val_samples, f)
        print(f"  - Saved {len(val_samples)} validation samples")

        with open(f"{output_dir}/test_samples.pkl", "wb") as f:
            pickle.dump(test_samples, f)
        print(f"  - Saved {len(test_samples)} test samples")

        # Canonical per-station train_std map: single source of truth for
        # std across the pipeline. Built in prepare_all_data from PRE-thinning,
        # PRE-flat-filter train samples (≥2 samples required). Stations not in
        # this map are Scenario B cold-start.
        station_to_train_std = getattr(self, "station_to_train_std", {})
        with open(os.path.join(output_dir, "station_to_train_std.pkl"), "wb") as f:
            pickle.dump(station_to_train_std, f)
        print(f"  - Saved station_to_train_std map ({len(station_to_train_std)} stations)")

        # Per-station raw-observation mode-gap (days). Built from raw GWL data
        # in prepare_all_data, used by min_station_mode_gap_days filter and the
        # eval-time predictable cohort lookup.
        station_to_mode_gap_days = getattr(self, "station_to_mode_gap_days", {})
        with open(os.path.join(output_dir, "station_to_mode_gap_days.pkl"), "wb") as f:
            pickle.dump(station_to_mode_gap_days, f)
        print(f"  - Saved station_to_mode_gap_days map ({len(station_to_mode_gap_days)} stations)")

        # Save config for reference (matching MLP pattern)
        with open(os.path.join(output_dir, "data_config_variables.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        print(f"  - Saved data configuration")

        # Save sample metadata (matching MLP pattern)
        dataset_size = len(train_samples) + len(val_samples) + len(test_samples)
        with open(os.path.join(output_dir, "samples.pkl"), "wb") as f:
            pickle.dump(
                {
                    "samples_created_total": dataset_size,
                    "samples_dropped_total": 0,
                    "gap_days": self.config.gap_days,
                    "forecast_horizon_months": self.config.forecast_horizon_months,
                    "use_delta_gwl": self.config.use_delta_gwl,
                    "validate_station_bounds": self.config.validate_station_bounds,
                    "station_outlier_method": self.config.station_outlier_method,
                    "max_gwl": self.config.max_gwl,
                    "min_gwl": self.config.min_gwl,
                    "detect_window_outliers": self.config.detect_window_outliers,
                },
                f,
            )
        print(f"  - Saved sample metadata")

        # Prithvi tile manifest (row → composite basename) for train.py's
        # TileStore. Only written when use_prithvi annotated samples upstream.
        tile_manifest = getattr(self, "_tile_manifest", None)
        if tile_manifest is not None:
            with open(os.path.join(output_dir, "tile_manifest.pkl"), "wb") as f:
                pickle.dump(tile_manifest, f)
            print(
                f"  - Saved tile_manifest.pkl "
                f"({len(tile_manifest['ordered_files'])} composites, "
                f"period={tile_manifest['period']}, "
                f"zero_idx={tile_manifest['zero_idx']})"
            )

        print("\nDone!")


def compute_station_static_features(
    train_station_codes,
    gwl_df,
    dynamic_df,
):
    """Compute per-station derived static features from train data.

    For each station present in train_station_codes:
      - mean_annual_rainfall  : avg of (yearly rainfall sums)
      - annual_rainfall_std   : std of yearly rainfall sums
      - station_mean_gwl      : mean of GWL readings (train period)
      - station_gwl_amplitude : max(monthly mean GWL) - min(monthly mean GWL)

    Args:
        train_station_codes: iterable of station_code strings to compute for.
        gwl_df: DataFrame with [station_code, date, gwl_value] (already filtered to train period).
        dynamic_df: DataFrame with [station_code, date, rainfall, ...] (full or train period).

    Returns:
        Dict[station_code, Dict[str, float]]
    """
    codes = list(train_station_codes)
    result = {
        code: {
            "mean_annual_rainfall": 0.0,
            "annual_rainfall_std": 0.0,
            "station_mean_gwl": 0.0,
            "station_gwl_amplitude": 0.0,
        }
        for code in codes
    }

    # Annual rainfall stats — single groupby across all stations
    if dynamic_df is not None and len(dynamic_df) > 0 and "rainfall" in dynamic_df.columns:
        yearly = (
            dynamic_df.assign(_year=dynamic_df["date"].dt.year)
            .groupby(["station_code", "_year"])["rainfall"].sum()
        )
        rain_mean = yearly.groupby(level=0).mean().to_dict()
        # Single-year stations get NaN std → 0.0 (matches old `len > 1` branch).
        rain_std = yearly.groupby(level=0).std().fillna(0.0).to_dict()
        for code in codes:
            if code in rain_mean:
                result[code]["mean_annual_rainfall"] = float(rain_mean[code])
                result[code]["annual_rainfall_std"] = float(rain_std[code])

    # Per-station GWL stats — single groupby across all stations
    if gwl_df is not None and len(gwl_df) > 0 and "gwl_value" in gwl_df.columns:
        station_mean = gwl_df.groupby("station_code")["gwl_value"].mean().to_dict()
        monthly = (
            gwl_df.assign(_month=gwl_df["date"].dt.month)
            .groupby(["station_code", "_month"])["gwl_value"].mean()
        )
        amp_max = monthly.groupby(level=0).max()
        amp_min = monthly.groupby(level=0).min()
        amp_count = monthly.groupby(level=0).count()
        # Stations with <2 distinct months → 0.0 (matches old `len >= 2` branch).
        amplitude = (amp_max - amp_min).where(amp_count >= 2, 0.0).to_dict()
        for code in codes:
            if code in station_mean:
                result[code]["station_mean_gwl"] = float(station_mean[code])
            if code in amplitude:
                result[code]["station_gwl_amplitude"] = float(amplitude[code])

    return result


def compute_station_month_gwl_means(
    train_station_codes,
    gwl_df,
):
    """Compute per-(station, calendar_month) mean GWL from train data.

    Used to compute gwl_anomaly = current_gwl - station_month_mean_gwl per sample.

    Returns:
        Dict[(station_code, month_int 1-12), float]
    """
    train_set = set(train_station_codes)
    sub = gwl_df[gwl_df["station_code"].isin(train_set)]
    if len(sub) == 0:
        return {}
    monthly = (
        sub.assign(_month=sub["date"].dt.month)
        .groupby(["station_code", "_month"])["gwl_value"].mean()
    )
    return {
        (code, int(month)): float(val)
        for (code, month), val in monthly.items()
    }


def compute_station_delta_stats(samples):
    """Compute per-station mean and std of the prediction-target delta.

    Delta = (target_gwl_raw - current_gwl) per sample, where both terms are
    absolute GWL values regardless of use_delta_gwl mode. This is exactly the
    quantity the model is asked to predict (in delta mode), so feeding
    per-station summaries lets the model anchor its prediction range to the
    well's typical magnitude.

    Stations with fewer than 2 samples are skipped (std is undefined).

    Args:
        samples: list of pre-flat-filter train samples.

    Returns:
        Dict[station_code, {"station_delta_mean": float, "station_delta_std": float}]
    """
    deltas_by_station = defaultdict(list)
    for s in samples:
        # target_gwl_raw is set when use_delta_gwl=True; falls back to target_gwl
        # (which is absolute when use_delta_gwl=False).
        target_abs = s.get("target_gwl_raw", s["target_gwl"])
        current_abs = s["current_gwl"]
        deltas_by_station[s["station_code"]].append(float(target_abs) - float(current_abs))

    stats = {}
    for code, deltas in deltas_by_station.items():
        if len(deltas) < 2:
            continue
        stats[code] = {
            "station_delta_mean": float(np.mean(deltas)),
            "station_delta_std": float(np.std(deltas)),
        }
    return stats


def compute_state_level_fallbacks(
    station_static_features,
    station_to_state,
):
    """Aggregate per-station features to state-level means for fallback on unseen stations.

    Returns:
        Dict[state, Dict[str, float]]  (same keys as station_static_features values)
    """
    by_state = defaultdict(list)
    for code, feats in station_static_features.items():
        state = station_to_state.get(code, "unknown")
        by_state[state].append(feats)
    result = {}
    # Union of all keys across stations — some stations may lack optional
    # features (e.g. station_delta_mean if they had < 2 samples).
    all_keys = set()
    for feats in station_static_features.values():
        all_keys.update(feats.keys())
    for state, feat_list in by_state.items():
        if not feat_list:
            continue
        result[state] = {
            k: float(np.mean([f.get(k, 0.0) for f in feat_list])) for k in all_keys
        }
    return result


def split_samples_per_station(
    samples: List[Dict],
    train_frac: float = 0.60,
    val_frac: float = 0.25,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples per-station chronologically.

    For each station, sorts samples by current_date and assigns:
      first train_frac → train, next val_frac → val, remainder → test.
    """
    # Group by station
    by_station: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        by_station[s["station_code"]].append(s)

    train_samples, val_samples, test_samples = [], [], []
    for code, station_samples in by_station.items():
        # Sort chronologically
        station_samples.sort(key=lambda s: s["current_date"])
        n = len(station_samples)
        n_train = max(1, int(train_frac * n))
        n_val = max(1, int(val_frac * n))

        train_samples.extend(station_samples[:n_train])
        val_samples.extend(station_samples[n_train:n_train + n_val])
        test_samples.extend(station_samples[n_train + n_val:])

    n_stations = len(by_station)
    # Stations with very few samples may not contribute to all splits
    train_st = len({s["station_code"] for s in train_samples})
    val_st = len({s["station_code"] for s in val_samples})
    test_st = len({s["station_code"] for s in test_samples})
    print(f"  {len(samples)} samples from {n_stations} stations →")
    print(f"    train: {len(train_samples)} samples ({train_st} stations)")
    print(f"    val:   {len(val_samples)} samples ({val_st} stations)")
    print(f"    test:  {len(test_samples)} samples ({test_st} stations)")

    return train_samples, val_samples, test_samples


def compute_in_bucket_stations(
    samples: List[Dict], min_std: float, max_std: float = float("inf")
) -> set:
    """Return the set of station_codes whose target-delta std falls in [min_std, max_std).

    Same std calculation as filter_flat_stations, but returns membership only —
    used by callers that want to apply a single bucket assignment consistently
    across train/val/test (rather than computing std per-split independently).
    """
    targets_by_station: Dict[str, list] = defaultdict(list)
    for s in samples:
        targets_by_station[s["station_code"]].append(s["target_gwl"])
    keep = set()
    for code, vals in targets_by_station.items():
        if len(vals) < 2:
            continue
        std_val = float(np.std(vals))
        if min_std <= std_val < max_std:
            keep.add(code)
    return keep


def filter_samples_by_station_set(
    samples: List[Dict], in_bucket: set, label: str
) -> List[Dict]:
    """Keep only samples whose station_code is in `in_bucket`. Logs counts."""
    before = len(samples)
    n_stations_before = len({s["station_code"] for s in samples})
    filtered = [s for s in samples if s["station_code"] in in_bucket]
    n_stations_after = len({s["station_code"] for s in filtered})
    print(
        f"  {label}: {before} → {len(filtered)} samples "
        f"({n_stations_before} → {n_stations_after} stations)"
    )
    return filtered


def filter_flat_stations(
    samples: List[Dict], min_std: float, max_std: float = float("inf")
) -> Tuple[List[Dict], int, int]:
    """Remove stations whose target delta std falls outside [min_std, max_std).

    Computes per-station std of target_gwl (which is delta when use_delta_gwl=True)
    and keeps only stations with min_std <= std < max_std. With the default
    max_std=inf this matches the original lower-bound-only behavior.

    Note: prefer compute_in_bucket_stations + filter_samples_by_station_set
    when applying the same bucket assignment across multiple splits — that
    way val/test reuse the train-derived membership for consistency.

    Returns:
        (filtered_samples, n_stations_kept, n_stations_dropped)
    """
    targets_by_station: Dict[str, list] = defaultdict(list)
    for s in samples:
        targets_by_station[s["station_code"]].append(s["target_gwl"])

    keep_stations = set()
    drop_stations = set()
    for code, vals in targets_by_station.items():
        if len(vals) < 2:
            drop_stations.add(code)
            continue
        std_val = float(np.std(vals))
        if std_val < min_std or std_val >= max_std:
            drop_stations.add(code)
        else:
            keep_stations.add(code)

    filtered = [s for s in samples if s["station_code"] in keep_stations]
    range_str = (
        f"std in [{min_std}, {max_std}m)"
        if max_std != float("inf") else f"std < {min_std}m (drop)"
    )
    print(
        f"  Flat-well filter ({range_str}): "
        f"{len(samples)} → {len(filtered)} samples "
        f"({len(keep_stations)} stations kept, {len(drop_stations)} dropped)"
    )
    return filtered, len(keep_stations), len(drop_stations)


def thin_samples(samples: List[Dict], freq_days: int) -> List[Dict]:
    """
    Thin samples per station using nearest-to-target-interval selection.

    For each station, sort samples by current_date. Keep the first sample.
    Then iteratively target `last_kept + freq_days` and pick the remaining
    sample closest to that target (allowing gaps slightly smaller or larger
    than freq_days to maintain roughly regular cadence).

    Args:
        samples: list of sample dicts with 'station_code' and 'current_date' keys.
        freq_days: target spacing in days. 0 or None disables thinning.

    Returns:
        Thinned list of samples. Order follows original per-station ordering.
    """
    if not freq_days or freq_days <= 0 or not samples:
        return samples

    by_station = defaultdict(list)
    for s in samples:
        by_station[s["station_code"]].append(s)

    kept_all = []
    total_before = 0
    total_after = 0
    for station_code, station_samples in by_station.items():
        station_samples.sort(key=lambda s: s["current_date"])
        total_before += len(station_samples)

        kept_idx = _greedy_nearest_to_target(
            [s["current_date"] for s in station_samples], freq_days
        )
        kept_station = [station_samples[k] for k in kept_idx]
        total_after += len(kept_station)
        kept_all.extend(kept_station)

    print(
        f"  Thinning @ {freq_days} days: "
        f"{total_before} → {total_after} samples "
        f"({len(by_station)} stations, dropped {total_before - total_after})"
    )

    return kept_all


def _greedy_nearest_to_target(dates: List, freq_days: int) -> List[int]:
    """
    Given a sorted list of dates and a target spacing in days, return
    indices of kept dates using nearest-to-target-interval selection.

    Algorithm:
        keep dates[0]
        while remaining dates exist:
            target = last_kept + freq_days
            among remaining, pick the one with smallest |date - target|
            append its index, continue after it
    """
    if not dates:
        return []
    kept = [0]
    i = 1
    while i < len(dates):
        last_date = dates[kept[-1]]
        target = last_date + pd.Timedelta(days=freq_days)
        best_j = i
        best_diff = abs((dates[i] - target).days)
        j = i + 1
        # Scan forward; stop once we've clearly moved past target and are
        # getting worse (dates sorted ascending => diff is monotone past target).
        while j < len(dates):
            diff = abs((dates[j] - target).days)
            if diff < best_diff:
                best_diff = diff
                best_j = j
            elif dates[j] > target:
                break  # past target and not improving
            j += 1
        kept.append(best_j)
        i = best_j + 1
    return kept


def cap_samples_per_station(
    samples: List[Dict], max_per_station: int, seed: int = 42
) -> List[Dict]:
    """
    Cap samples per station to at most `max_per_station`, using seeded random
    subsampling. Samples that remain are sorted by current_date.

    Args:
        samples: list of sample dicts with 'station_code' and 'current_date'.
        max_per_station: max samples per station (0 = no cap).
        seed: random seed for reproducibility.

    Returns:
        Capped list of samples.
    """
    if not max_per_station or max_per_station <= 0 or not samples:
        return samples

    rng = np.random.default_rng(seed)
    by_station = defaultdict(list)
    for s in samples:
        by_station[s["station_code"]].append(s)

    kept = []
    n_capped = 0
    for station_code, station_samples in by_station.items():
        if len(station_samples) > max_per_station:
            chosen = rng.choice(len(station_samples), size=max_per_station, replace=False)
            selected = [station_samples[i] for i in chosen]
            n_capped += 1
        else:
            selected = station_samples
        selected.sort(key=lambda s: s["current_date"])
        kept.extend(selected)

    print(
        f"  Cap @ {max_per_station}/station: "
        f"{len(samples)} → {len(kept)} samples "
        f"({n_capped} stations were above the cap)"
    )
    return kept


def _process_station_worker(args):
    """
    Top-level worker function for parallel station processing.

    Processes all dates for a single station independently.
    Must be top-level (not a method) for multiprocessing pickling.

    Args:
        args: Tuple of (config, station_code, station_gwl, station_dynamic,
              station_static, dates, bounds, is_training)

    Returns:
        Tuple of (samples_list, drop_reasons_dict)
    """
    (
        config,
        station_code,
        station_gwl,
        station_dynamic,
        station_static,
        dates,
        bounds,
        is_training,
    ) = args

    # Seed random per station for reproducibility + diversity across workers
    random.seed(hash(station_code) & 0xFFFFFFFF)

    # Lightweight instance — no DB connection needed, just uses config + methods
    prep = LSTMDataPreparation(config)

    samples = []
    drop_reasons = {
        "no_current_gwl": 0,
        "no_target_gwl": 0,
        "current_gwl_outlier": 0,
        "target_gwl_outlier": 0,
        "sequence_incomplete": 0,
        "sequence_incomplete_details": [],
    }

    for date in dates:
        date_obj = pd.to_datetime(date).to_pydatetime()
        sample = prep.create_sample(
            station_gwl,
            station_dynamic,
            station_static,
            station_code,
            date_obj,
            station_gwl_bounds=bounds,
            is_training=is_training,
            drop_reasons=drop_reasons,
        )
        if sample is not None:
            samples.append(sample)

    return samples, drop_reasons


def plot_sign_filter_analysis(
    df_raw: "pd.DataFrame",
    output_dir: str,
    thresholds=(0.5, 0.75),
    n_stations: int = 4,
    seed: int = 42,
) -> None:
    """
    Visualise the per-station dominant-sign filtering for two thresholds.

    For each threshold and sign category (positive / negative) a grid of
    `n_stations` randomly-sampled stations is plotted showing the raw GWL
    time-series before and after filtering.  Figures are saved to
    ``output_dir/sign_filter_analysis/``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = os.path.join(output_dir, "sign_filter_analysis")
    os.makedirs(save_dir, exist_ok=True)

    rng = random.Random(seed)
    df_raw = df_raw.copy()
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    for threshold in thresholds:
        sign_frac = df_raw.groupby("station_code")["gwl_value"].agg(
            lambda x: (x >= 0).mean()
        )
        positive_stations = sorted(sign_frac[sign_frac > threshold].index.tolist())
        negative_stations = sorted(sign_frac[sign_frac <= threshold].index.tolist())

        for sign_label, station_pool, keep_sign in [
            ("positive", positive_stations, 1),
            ("negative", negative_stations, -1),
        ]:
            if not station_pool:
                print(
                    f"  [sign filter plot] No {sign_label} stations at threshold={threshold}, skipping."
                )
                continue

            sample_stations = rng.sample(
                station_pool, min(n_stations, len(station_pool))
            )
            n = len(sample_stations)

            fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)
            fig.suptitle(
                f"Sign filter analysis — threshold={threshold}, {sign_label} stations\n"
                f"(keep {'gwl_value >= 0' if keep_sign == 1 else 'gwl_value <= 0'})",
                fontsize=12,
                y=1.01,
            )

            for row, station in enumerate(sample_stations):
                df_station = df_raw[df_raw["station_code"] == station].sort_values(
                    "date"
                )
                if keep_sign == 1:
                    df_filtered = df_station[df_station["gwl_value"] >= 0]
                else:
                    df_filtered = df_station[df_station["gwl_value"] <= 0]

                removed = len(df_station) - len(df_filtered)
                frac = sign_frac.loc[station]

                ax_before = axes[row, 0]
                ax_before.plot(
                    df_station["date"],
                    df_station["gwl_value"],
                    color="steelblue",
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax_before.axhline(0, color="black", linewidth=0.5, linestyle="--")
                ax_before.set_title(
                    f"{station} | before  (non-neg frac={frac:.2f})", fontsize=9
                )
                ax_before.set_xlabel("Date", fontsize=8)
                ax_before.set_ylabel("GWL (m)", fontsize=8)
                ax_before.tick_params(labelsize=7)

                ax_after = axes[row, 1]
                ax_after.plot(
                    df_filtered["date"],
                    df_filtered["gwl_value"],
                    color="darkorange",
                    linewidth=0.8,
                    alpha=0.8,
                )
                ax_after.axhline(0, color="black", linewidth=0.5, linestyle="--")
                ax_after.set_title(
                    f"{station} | after  ({removed} readings removed)", fontsize=9
                )
                ax_after.set_xlabel("Date", fontsize=8)
                ax_after.set_ylabel("GWL (m)", fontsize=8)
                ax_after.tick_params(labelsize=7)

            fig.tight_layout()
            fname = f"sign_filter_{threshold}_{sign_label}.png"
            fpath = os.path.join(save_dir, fname)
            fig.savefig(fpath, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved sign filter plot → {fpath}")

    print(f"Sign filter analysis complete. Plots saved to {save_dir}/")


def plot_hard_cap_analysis(
    df_before_cap: "pd.DataFrame",
    output_dir: str,
    min_gwl: float,
    max_gwl: float,
    n_stations: int = 5,
    seed: int = 42,
) -> None:
    """
    Identify stations with hard-cap violations and plot before/after for each.

    A station is included if it has at least one reading where
    gwl_value > max_gwl OR gwl_value < min_gwl.  Up to `n_stations`
    such stations are randomly sampled and plotted as stacked before/after
    subplot pairs.

    Output: ``output_dir/hard_cap_analysis/hard_cap_outlier_stations.png``
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    violation_mask = pd.Series(False, index=df_before_cap.index)
    if max_gwl > 0:
        violation_mask |= df_before_cap["gwl_value"] > max_gwl
    if min_gwl < 0:
        violation_mask |= df_before_cap["gwl_value"] < min_gwl

    violating_stations = (
        df_before_cap.loc[violation_mask, "station_code"].unique().tolist()
    )

    if not violating_stations:
        print("  [hard cap plot] No stations with hard-cap violations found, skipping.")
        return

    rng = random.Random(seed)
    sample = rng.sample(violating_stations, min(n_stations, len(violating_stations)))

    save_dir = os.path.join(output_dir, "hard_cap_analysis")
    os.makedirs(save_dir, exist_ok=True)

    n = len(sample)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n), squeeze=False)
    fig.suptitle(
        f"Hard cap analysis  |  caps: [{min_gwl}, {max_gwl}] m\n"
        f"Showing {n} of {len(violating_stations)} stations with violations",
        fontsize=12,
        y=1.01,
    )

    for row, station in enumerate(sample):
        df_st = df_before_cap[df_before_cap["station_code"] == station].sort_values(
            "date"
        )

        df_cap = df_st.copy()
        if max_gwl > 0:
            df_cap = df_cap[df_cap["gwl_value"] <= max_gwl]
        if min_gwl < 0:
            df_cap = df_cap[df_cap["gwl_value"] >= min_gwl]

        n_violations = len(df_st) - len(df_cap)

        ax_b = axes[row, 0]
        ax_b.plot(
            df_st["date"],
            df_st["gwl_value"],
            color="steelblue",
            linewidth=0.8,
            alpha=0.8,
        )
        df_viol = df_st[
            ((max_gwl > 0) & (df_st["gwl_value"] > max_gwl))
            | ((min_gwl < 0) & (df_st["gwl_value"] < min_gwl))
        ]
        ax_b.scatter(
            df_viol["date"],
            df_viol["gwl_value"],
            color="red",
            s=20,
            zorder=5,
            label="violation",
        )
        if max_gwl > 0:
            ax_b.axhline(
                max_gwl,
                color="red",
                linewidth=0.8,
                linestyle="--",
                label=f"max={max_gwl}m",
            )
        if min_gwl < 0:
            ax_b.axhline(
                min_gwl,
                color="darkred",
                linewidth=0.8,
                linestyle="--",
                label=f"min={min_gwl}m",
            )
        ax_b.set_title(
            f"{station} | before hard cap ({n_violations} violations)", fontsize=9
        )
        ax_b.set_xlabel("Date", fontsize=8)
        ax_b.set_ylabel("GWL (m)", fontsize=8)
        ax_b.tick_params(labelsize=7)
        ax_b.legend(fontsize=7, loc="upper right")

        ax_a = axes[row, 1]
        ax_a.plot(
            df_cap["date"],
            df_cap["gwl_value"],
            color="darkorange",
            linewidth=0.8,
            alpha=0.8,
        )
        if max_gwl > 0:
            ax_a.axhline(
                max_gwl,
                color="red",
                linewidth=0.8,
                linestyle="--",
                label=f"max={max_gwl}m",
            )
        if min_gwl < 0:
            ax_a.axhline(
                min_gwl,
                color="darkred",
                linewidth=0.8,
                linestyle="--",
                label=f"min={min_gwl}m",
            )
        ax_a.set_title(f"{station} | after hard cap filtering", fontsize=9)
        ax_a.set_xlabel("Date", fontsize=8)
        ax_a.set_ylabel("GWL (m)", fontsize=8)
        ax_a.tick_params(labelsize=7)
        ax_a.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fpath = os.path.join(save_dir, "hard_cap_outlier_stations.png")
    fig.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved hard cap plot → {fpath}")
    print(
        f"Hard cap analysis complete. {len(violating_stations)} stations had violations "
        f"({n} plotted)."
    )


if __name__ == "__main__":
    """
    Run data preparation pipeline.

    Usage:
        python gwl_lstm/data_preparation.py
        python gwl_lstm/data_preparation.py --horizon 3
        python gwl_lstm/data_preparation.py --horizon 6 --lookback 5 --gap-days 30
        python gwl_lstm/data_preparation.py --no-delta-gwl --no-station-bounds
        python gwl_lstm/data_preparation.py --train-end 2022-12-31 --val-start 2023-01-01

    DB credentials via environment variables:
        export GWL_DB_HOST=/var/run/postgresql
        export GWL_DB_NAME=gwl
        export GWL_DB_USER=ubuntu
        export GWL_DB_PASSWORD=gwl
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Prepare LSTM data for GWL forecasting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data pipeline parameters ───────────────────────────────────────
    pipeline = parser.add_argument_group("pipeline parameters")
    pipeline.add_argument(
        "--horizon", type=int, default=6, help="Forecast horizon in months"
    )
    pipeline.add_argument(
        "--lookback", type=int, default=5, help="Lookback period in years (used unless --lookback-months is set)"
    )
    pipeline.add_argument(
        "--lookback-months", type=int, default=0,
        help="Lookback period in months. If > 0, overrides --lookback (years).",
    )
    pipeline.add_argument(
        "--lookback-window",
        type=str,
        default="3",
        help="Size of each lookback window. Suffix 'd' = days, 'm' = months, "
             "bare number = months (default unit). Examples: '7d' (7 days), "
             "'1m' (30 days), '3' (90 days). Total lookback (in months) is "
             "set via --lookback / --lookback-months; num_timesteps = "
             "lookback_total_days // lookback_window_days.",
    )
    pipeline.add_argument(
        "--climatology-years",
        type=int,
        default=12,
        help="Number of years for climatology",
    )
    pipeline.add_argument(
        "--gap-days",
        type=int,
        default=60,
        help="Days to search around target date for GWL",
    )
    pipeline.add_argument(
        "--no-forward-fill",
        action="store_true",
        help="Disable forward fill for missing feature values",
    )
    pipeline.add_argument(
        "--actual-forecast-prob",
        type=float,
        default=0.8,
        help="Probability of using actual (vs climatology) forecast in training",
    )
    pipeline.add_argument(
        "--rain-temp-only",
        action="store_true",
        default=False,
        help="Use only rainfall (sum) and temp (mean) as forecast features (2 instead of 6)",
    )
    ndvi_sm_grp = pipeline.add_mutually_exclusive_group()
    ndvi_sm_grp.add_argument(
        "--drop-ndvi-sm",
        action="store_true",
        default=True,
        dest="drop_ndvi_sm",
        help="Drop NDVI and soil moisture from per-timestep aggregates (default: ON; "
             "they are 94-99%% missing in raw data).",
    )
    ndvi_sm_grp.add_argument(
        "--no-drop-ndvi-sm",
        action="store_false",
        dest="drop_ndvi_sm",
        help="Keep NDVI and soil moisture features in per-timestep aggregates.",
    )
    pipeline.add_argument(
        "--use-only-gwl",
        action="store_true",
        default=False,
        help="Per-timestep sequence keeps ONLY GWL + temporal features "
             "(gwl_encoded, is_present, gwl_diff, gwl_diff_reliable, sin, cos = 6 features). "
             "Drops the 6 historical channels and 3 cumulative/anomaly features from "
             "the lookback. Forecast features (rain/temp) and static features unaffected.",
    )
    pipeline.add_argument(
        "--min-sequence-completeness",
        type=float,
        default=0.5,
        help="Minimum fraction of timesteps with GWL present (0.0-1.0)",
    )

    # ── GWL encoding ──────────────────────────────────────────────────
    gwl_group = parser.add_argument_group("GWL encoding & outlier detection")
    gwl_group.add_argument(
        "--delta-gwl",
        action="store_true",
        default=True,
        help="Encode GWL as delta from current_gwl (default: enabled)",
    )
    gwl_group.add_argument(
        "--no-delta-gwl",
        dest="delta_gwl",
        action="store_false",
        help="Use absolute GWL values instead of deltas",
    )
    gwl_group.add_argument(
        "--station-bounds",
        action="store_true",
        default=True,
        help="Validate current/target GWL against station bounds",
    )
    gwl_group.add_argument(
        "--no-station-bounds",
        dest="station_bounds",
        action="store_false",
        help="Disable station-level outlier validation",
    )
    gwl_group.add_argument(
        "--station-outlier-method",
        type=str,
        default="mad",
        choices=["mad", "iqr"],
        help="Station outlier method: mad (robust, default) or iqr (legacy)",
    )
    gwl_group.add_argument(
        "--station-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for station-level GWL bounds (when method=iqr)",
    )
    gwl_group.add_argument(
        "--station-mad-multiplier",
        type=float,
        default=3.0,
        help="MAD multiplier for station-level GWL bounds (when method=mad)",
    )
    gwl_group.add_argument(
        "--max-gwl",
        type=float,
        default=100.0,
        help="Hard cap on positive GWL values in meters (0=disabled)",
    )
    gwl_group.add_argument(
        "--min-gwl",
        type=float,
        default=-100.0,
        help="Hard floor on negative GWL values in meters (0=disabled)",
    )
    gwl_group.add_argument(
        "--station-sign-threshold",
        type=float,
        default=0.5,
        help="Fraction of non-negative readings above which a station is "
        "classified as positive-convention (default 0.5)",
    )
    gwl_group.add_argument(
        "--window-outliers",
        action="store_true",
        default=True,
        help="Detect outliers within each lookback window",
    )
    gwl_group.add_argument(
        "--no-window-outliers",
        dest="window_outliers",
        action="store_false",
        help="Disable per-window outlier detection",
    )
    gwl_group.add_argument(
        "--window-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for window-level outlier detection",
    )
    gwl_group.add_argument(
        "--min-outlier-points",
        type=int,
        default=4,
        help="Minimum present values to compute window IQR",
    )

    # ── Split strategy ─────────────────────────────────────────────────
    split = parser.add_argument_group("split strategy")
    split.add_argument(
        "--split-strategy",
        type=str,
        choices=["district", "time", "station_time"],
        default="district",
        help="Split strategy: 'district' = station split by district (default), "
             "'time' = global date cutoff, "
             "'station_time' = per-station chronological 60/25/15.",
    )
    # Keep legacy flags for backward compatibility
    split.add_argument(
        "--split-by-time",
        action="store_const",
        const="time",
        dest="split_strategy",
        help="Shortcut for --split-strategy time.",
    )
    split.add_argument(
        "--split-by-district",
        action="store_const",
        const="district",
        dest="split_strategy",
        help="Shortcut for --split-strategy district.",
    )
    split.add_argument(
        "--station-train-frac",
        type=float,
        default=0.60,
        help="Train fraction for station_time split. Default: 0.60.",
    )
    split.add_argument(
        "--station-val-frac",
        type=float,
        default=0.25,
        help="Val fraction for station_time split. Default: 0.25. "
             "Test = remainder.",
    )

    # ── Train/val/test date ranges ─────────────────────────────────────
    dates = parser.add_argument_group("split date ranges")
    dates.add_argument(
        "--train-end",
        type=str,
        default="2022-12-31",
        help="Training set end date (YYYY-MM-DD)",
    )
    dates.add_argument(
        "--val-start",
        type=str,
        default="2023-01-01",
        help="Validation set start date (YYYY-MM-DD)",
    )
    dates.add_argument(
        "--val-end",
        type=str,
        default="2024-12-31",
        help="Validation set end date (YYYY-MM-DD)",
    )
    dates.add_argument(
        "--test-start",
        type=str,
        default="2025-01-01",
        help="Test set start date (YYYY-MM-DD)",
    )

    # ── Database ───────────────────────────────────────────────────────
    db = parser.add_argument_group("database (overrides env vars GWL_DB_*)")
    db.add_argument(
        "--database-table",
        type=str,
        default="data_with_flag_500k",
        help="Database table name",
    )
    db.add_argument(
        "--db-host",
        type=str,
        default="",
        help="DB host (default: $GWL_DB_HOST or /var/run/postgresql)",
    )
    db.add_argument(
        "--db-name", type=str, default="", help="DB name (default: $GWL_DB_NAME or gwl)"
    )
    db.add_argument(
        "--db-user",
        type=str,
        default="",
        help="DB user (default: $GWL_DB_USER or ubuntu)",
    )
    db.add_argument(
        "--db-password",
        type=str,
        default="",
        help="DB password (default: $GWL_DB_PASSWORD or gwl)",
    )

    # ── Data source ────────────────────────────────────────────────────
    src = parser.add_argument_group("data source")
    src.add_argument(
        "--csv-path",
        type=str,
        default="",
        help="Path to the CSV data file (overrides CSV_DATA_PATH constant)",
    )

    # ── Output ─────────────────────────────────────────────────────────
    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir", type=str, default="gwl_lstm/data", help="Output directory"
    )
    output.add_argument(
        "--plot-dir",
        type=str,
        default=None,
        help="Directory for diagnostic plots (default: same as --output-dir)",
    )
    output.add_argument(
        "--shared-artifact-dir",
        type=str,
        default=None,
        help="If set, saves a shared dataset pickle here (for LSTM + MLP). "
             "Example: shared_data/processed",
    )
    output.add_argument(
        "--sign-filter-plot-dir",
        type=str,
        default=None,
        help="If set, generate sign-filter and hard-cap analysis plots under "
        "this directory (disabled by default)",
    )

    # ── Performance ────────────────────────────────────────────────────
    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of parallel workers (0=auto, 1=sequential)",
    )
    perf.add_argument(
        "--min-station-samples",
        type=int,
        default=0,
        help="Drop stations with fewer than this many samples (0=disabled, default)",
    )
    perf.add_argument(
        "--min-sample-freq-days",
        type=int,
        default=14,
        help="Thin each station's samples to roughly this cadence in days "
             "(nearest-to-target selection). 0=disabled. Default: 14.",
    )
    perf.add_argument(
        "--min-sample-freq-days-train",
        type=int, default=-1,
        help="Per-split override of --min-sample-freq-days for TRAIN. "
             "-1 = inherit. 0 = disable thinning for train (more data). "
             "With perwell loss weighting, this is often the right choice.",
    )
    perf.add_argument(
        "--min-sample-freq-days-val",
        type=int, default=-1,
        help="Per-split override for VAL. -1 = inherit; 0 = disable.",
    )
    perf.add_argument(
        "--min-sample-freq-days-test",
        type=int, default=-1,
        help="Per-split override for TEST. -1 = inherit; 0 = disable.",
    )
    perf.add_argument(
        "--max-samples-per-station-eval",
        type=int,
        default=30,
        help="Cap samples per station in val/test splits. 0=disabled. Default: 30.",
    )
    perf.add_argument(
        "--min-station-target-std",
        type=float,
        default=0.0,
        help="Exclude stations with target delta std below this (flat wells) "
             "from train/val. Test keeps all. 0=disabled. Default: 0.0.",
    )
    perf.add_argument(
        "--max-station-target-std",
        type=float,
        default=float("inf"),
        help="Exclude stations with target delta std >= this. Used by the "
             "multi-model pipeline to slice wells into std buckets. When < inf, "
             "test is also bucket-restricted. Default: inf (no upper bound).",
    )
    perf.add_argument(
        "--include-states",
        type=str,
        default="",
        help="Comma-separated list of states to keep in train+val (additive to "
             "std filter; both compose by AND). Test stays unfiltered (cold-start "
             "view). Empty = disabled. Example: 'Kerala,Telangana,Andhra Pradesh'.",
    )
    perf.add_argument(
        "--min-station-mode-gap-days",
        type=float,
        default=0.0,
        help="Drop stations from train+val whose RAW observation mode-gap "
             "(most-common gap days between consecutive readings) is below "
             "this threshold. Targets densely-sampled flat wells which the "
             "model fits to noise. Test stays unfiltered. 0=disabled.",
    )
    perf.add_argument(
        "--interpolate-lookback-gwl",
        action="store_true",
        help="Linear-interpolate the GWL channel of the lookback input "
             "sequence between consecutive real observations per station. "
             "Sample-grounding (current/target) stays based on real "
             "observations. Applies uniformly to train, val, test.",
    )

    # ── Prithvi-EO composites (Step 2: tile_idx) ──────────────────────
    prithvi = parser.add_argument_group("Prithvi-EO composites (tile_idx)")
    prithvi.add_argument(
        "--use-prithvi",
        action="store_true",
        default=os.environ.get("USE_PRITHVI", "").lower() in ("1", "true", "yes", "y", "on"),
        help="Annotate each sample with a tile_idx pointing at a downloaded "
             "HLS composite (for joint Prithvi fine-tuning), and write "
             "tile_manifest.pkl. Env: USE_PRITHVI=1. Requires --composite-dir.",
    )
    prithvi.add_argument(
        "--composite-dir",
        type=str,
        default=os.environ.get("COMPOSITE_DIR", ""),
        help="Dir of composite_<safe_id>_<year>_<period>.tif files. "
             "Env: COMPOSITE_DIR.",
    )
    prithvi.add_argument(
        "--composite-period",
        type=str,
        choices=["halfyear", "quarter"],
        default=os.environ.get("COMPOSITE_PERIOD", "halfyear"),
        help="Composite bucketing: halfyear (H1/H2) or quarter (Q1-Q4). "
             "Env: COMPOSITE_PERIOD. Default: halfyear.",
    )
    prithvi.add_argument(
        "--station-index-csv",
        type=str,
        default=os.environ.get("STATION_INDEX_CSV", ""),
        help="CSV mapping station_code → safe_id (cols 'station_code','safe_id'). "
             "Env: STATION_INDEX_CSV. Empty = derive safe_id by sanitizing "
             "station_code (space→_, /→--).",
    )

    args = parser.parse_args()

    start_time = time.time()

    # Create configuration
    config = DataConfig(
        forecast_horizon_months=args.horizon,
        lookback_years=args.lookback,
        lookback_months=args.lookback_months,
        lookback_window_days=parse_lookback_window(args.lookback_window),
        climatology_years=args.climatology_years,
        gap_days=args.gap_days,
        forward_fill_features=not args.no_forward_fill,
        use_actual_forecast_prob=args.actual_forecast_prob,
        min_sequence_completeness=args.min_sequence_completeness,
        # GWL encoding & outlier detection
        use_delta_gwl=args.delta_gwl,
        validate_station_bounds=args.station_bounds,
        station_outlier_method=args.station_outlier_method,
        station_iqr_multiplier=args.station_iqr_multiplier,
        station_mad_multiplier=args.station_mad_multiplier,
        max_gwl=args.max_gwl,
        min_gwl=args.min_gwl,
        detect_window_outliers=args.window_outliers,
        min_outlier_points=args.min_outlier_points,
        window_iqr_multiplier=args.window_iqr_multiplier,
        # Split strategy
        split_strategy=args.split_strategy,
        station_train_frac=args.station_train_frac,
        station_val_frac=args.station_val_frac,
        # Date ranges
        train_end_date=args.train_end,
        val_start_date=args.val_start,
        val_end_date=args.val_end,
        test_start_date=args.test_start,
        # CSV data source
        csv_path=args.csv_path,
        # --- ORIGINAL DB LOGIC (kept for future use) ---
        # database_table=args.database_table,
        # db_host=args.db_host,
        # db_port=args.db_port,
        # db_name=args.db_name,
        # db_user=args.db_user,
        # db_password=args.db_password,
        # Sign filter / hard cap analysis plots
        station_sign_threshold=args.station_sign_threshold,
        sign_filter_plot_dir=args.sign_filter_plot_dir or "",
        # Parallelism
        n_workers=args.workers,
        # Station sample threshold
        min_station_samples=args.min_station_samples,
        min_sample_freq_days=args.min_sample_freq_days,
        min_sample_freq_days_train=args.min_sample_freq_days_train,
        min_sample_freq_days_val=args.min_sample_freq_days_val,
        min_sample_freq_days_test=args.min_sample_freq_days_test,
        max_samples_per_station_eval=args.max_samples_per_station_eval,
        min_station_target_std=args.min_station_target_std,
        max_station_target_std=args.max_station_target_std,
        include_states=[s.strip() for s in args.include_states.split(",") if s.strip()] if args.include_states else [],
        min_station_mode_gap_days=args.min_station_mode_gap_days,
        interpolate_lookback_gwl=args.interpolate_lookback_gwl,
        use_rain_temp=args.rain_temp_only,
        drop_ndvi_sm=args.drop_ndvi_sm,
        use_only_gwl=args.use_only_gwl,
        # Prithvi-EO composites (Step 2)
        use_prithvi=args.use_prithvi,
        composite_dir=args.composite_dir,
        composite_period=args.composite_period,
        station_index_csv=args.station_index_csv,
    )
    plot_dir = args.plot_dir or args.output_dir

    gwl_mode = (
        "DELTA (value - current_gwl)"
        if config.use_delta_gwl
        else "ABSOLUTE (raw values)"
    )

    print("=" * 70)
    print("LSTM Data Preparation")
    print("=" * 70)
    print(f"Forecast horizon: {config.forecast_horizon_months} months")
    print(f"Lookback: {config.lookback_total_months} months")
    print(f"Lookback window: {config.lookback_window_days} days ({config.num_timesteps} timesteps)")
    print(f"Climatology years: {config.climatology_years}")
    print(f"Timesteps per sequence: {config.num_timesteps}")
    print(f"GWL gap days: ±{config.gap_days} days")
    print(f"Forward fill features: {config.forward_fill_features}")
    print(f"Actual forecast prob: {config.use_actual_forecast_prob}")
    print(f"Min sequence completeness: {config.min_sequence_completeness}")
    print()
    print(f"GWL encoding: {gwl_mode}")
    print(
        f"Hard GWL caps: [{config.min_gwl}, {config.max_gwl}]m (per-station sign convention)"
    )
    if config.validate_station_bounds:
        method = config.station_outlier_method.upper()
        if config.station_outlier_method == "mad":
            print(
                f"Station bounds validation: {method} (multiplier={config.station_mad_multiplier})"
            )
        else:
            print(
                f"Station bounds validation: {method} (multiplier={config.station_iqr_multiplier})"
            )
    else:
        print(f"Station bounds validation: disabled")
    if config.detect_window_outliers:
        print(
            f"Window outlier detection: enabled (IQR×{config.window_iqr_multiplier}, "
            f"min {config.min_outlier_points} pts)"
        )
    else:
        print(f"Window outlier detection: disabled")
    print()
    # print(f"Database table: {config.database_table}")
    # print(f"DB host: {config.db_host}")
    # print(f"DB name: {config.db_name}")
    # print(f"DB user: {config.db_user}")
    print(f"CSV data path: {CSV_DATA_PATH}")
    print(f"Train end:  {config.train_end_date}")
    print(f"Val:        {config.val_start_date} → {config.val_end_date}")
    print(f"Test start: {config.test_start_date}")
    print(f"Output directory: {args.output_dir}")
    print(f"Plot directory: {plot_dir}")
    effective_workers = (
        config.n_workers if config.n_workers > 0 else max(1, mp.cpu_count() - 1)
    )
    print(
        f"Workers: {effective_workers} ({'auto' if config.n_workers == 0 else 'manual'})"
    )
    print()
    n_hist = 4 if config.drop_ndvi_sm else 6
    n_features = 6 + n_hist + 3  # gwl(2) + diff(2) + month(2) + hist + cum_rain(3)
    n_fc = 2 if config.use_rain_temp else (4 if config.drop_ndvi_sm else 6)
    print(f"Features per timestep: {n_features}")
    print(f"  - 1 GWL {'delta' if config.use_delta_gwl else 'absolute'} value")
    print(f"  - 1 GWL is_reliable flag (0 = missing or outlier)")
    print(f"  - 1 GWL diff (local trend: change between consecutive timesteps)")
    print(f"  - 1 GWL diff_reliable flag")
    print(f"  - 2 temporal encoding (month sin/cos)")
    if config.drop_ndvi_sm:
        print(f"  - {n_hist} historical aggregates (rain,temp,et,runoff)  [NDVI/SM dropped]")
    else:
        print(f"  - {n_hist} historical aggregates (rain,temp,et,runoff,ndvi,sm)")
    print(f"  - 1 cumulative rainfall (from lookback start)")
    print(f"  - 1 rainfall anomaly (window actual - climatology)")
    print(f"  - 1 cumulative rainfall anomaly")
    print(f"Forecast features: {n_fc} (separate, for conditioning)")
    print(f"NDVI/SM dropped: {'yes' if config.drop_ndvi_sm else 'no'}")
    if config.use_prithvi:
        print(
            f"Prithvi tiles: ON (period={config.composite_period}, "
            f"dir={config.composite_dir or '<unset>'}, "
            f"station_index={config.station_index_csv or '<sanitize fallback>'})"
        )
    print()

    # Prepare data
    prep = LSTMDataPreparation(config)
    train_samples, val_samples, test_samples = prep.prepare_all_data(
        shared_artifact_dir=args.shared_artifact_dir,
    )

    # Sample-generation audit: one detailed dump per split (always-on)
    prep.write_sample_audit(
        train_samples, val_samples, test_samples, args.output_dir
    )

    # Save to disk
    prep.save_datasets(train_samples, val_samples, test_samples, args.output_dir)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Training samples:   {len(train_samples):,}")
    print(f"Validation samples: {len(val_samples):,}")
    print(f"Test samples:       {len(test_samples):,}")
    print(
        f"Total samples:      {len(train_samples) + len(val_samples) + len(test_samples):,}"
    )

    if len(train_samples) > 0:
        sample = train_samples[0]
        print(f"\nSample structure (first sample):")
        print(f"  - Sequence shape: {sample['sequence'].shape}")
        print(f"  - Forecast features shape: {sample['forecast_features'].shape}")
        print(f"  - Current GWL (raw): {sample['current_gwl']:.2f}m")
        if config.use_delta_gwl:
            print(f"  - Target GWL (delta): {sample['target_gwl']:.4f}m")
            print(f"  - Target GWL (raw):   {sample['target_gwl_raw']:.2f}m")
            print(
                f"  - Verify: current + delta = {sample['current_gwl'] + sample['target_gwl']:.2f}m "
                f"(raw target = {sample['target_gwl_raw']:.2f}m)"
            )
        else:
            print(f"  - Target GWL: {sample['target_gwl']:.2f}m")
        print(f"  - Station ID: {sample['station_code']}")
        print(f"  - Current date: {sample['current_date']}")
        print(f"  - Target date: {sample['target_date']}")

    print(f"\nTime taken: {time.time() - start_time:.1f}s")
