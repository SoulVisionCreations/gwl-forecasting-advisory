"""
Feature Scaling for GWL Forecasting LSTM

Implements normalization/scaling strategy from lstm_design.md:
- Separate scalers for different feature types
- Scale sequences while preserving structure
- StandardScaler for continuous features (fit on training data only)
- No scaling for temporal encodings (already [-1, 1])
- Save all scalers for inference

Reference: lstm_design.md (Normalization section)
"""

import copy
from collections import defaultdict
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dataclasses import dataclass


@dataclass
class ScalerConfig:
    """
    Configuration for feature scaling.

    Reference: lstm_design.md (Normalization Strategy)
    """
    # Which features to scale
    scale_gwl: bool = True
    scale_historical: bool = True  # Historical features in sequence
    scale_forecast: bool = True    # Forecast features (separate from sequence)
    scale_static: bool = True
    scale_target: bool = True

    # Temporal encodings are already [-1, 1], no scaling needed
    scale_temporal: bool = False

    # Feature indices in sequence (per timestep)
    # Default layout (NDVI/SM dropped, 13 features):
    # [gwl_val, is_present, gwl_diff, gwl_diff_reliable, sin, cos,
    #  4 historical (rain,temp,et,runoff),
    #  cum_rain, rain_anomaly, cum_rain_anomaly] = 13 features
    # When --no-drop-ndvi-sm: 15 features (ndvi, sm inserted before cum_rain).
    # `hist_end_idx` is auto-detected at fit time from sample width — keeps
    # the scaler agnostic to whether NDVI/SM are present.
    gwl_idx: int = 0
    gwl_is_present_idx: int = 1
    gwl_diff_idx: int = 2
    gwl_diff_reliable_idx: int = 3
    temporal_start_idx: int = 4
    temporal_end_idx: int = 6  # exclusive
    hist_start_idx: int = 6
    hist_end_idx: int = 13  # default: 4 base + 3 rainfall enrichment (NDVI/SM dropped)

    # Outlier detection: drop sample if ratio AND absolute diff both exceeded
    outlier_ratio: float = 10.0       # target/current or current/target > this
    outlier_abs_diff: float = 30.0    # AND abs(target - current) > this (meters)


class LSTMFeatureScaler:
    """
    Scale features for GWL forecasting LSTM.

    Implements separate scalers for different feature types:
    1. GWL values in sequences (anchor feature)
    2. Historical features in sequence (6 features per timestep)
    3. Forecast features (6 features, separate from sequence, for conditioning)
    4. Static continuous features (elevation, stream_order, latitude, longitude)
    5. Target GWL (for training only)

    Temporal encodings (sin/cos) are NOT scaled (already bounded [-1, 1])
    Embeddings are NOT scaled (learned from scratch during training)

    Note: Forecast features are stored and scaled separately from the sequence.
    They are used for conditioning the prediction, NOT fed into the LSTM.

    Reference: lstm_design.md (Normalization section)
    """

    def __init__(self, config: ScalerConfig = None):
        self.config = config or ScalerConfig()

        # Scalers for different feature types
        self.gwl_scaler = StandardScaler() if self.config.scale_gwl else None
        self.historical_scaler = StandardScaler() if self.config.scale_historical else None
        self.forecast_scaler = StandardScaler() if self.config.scale_forecast else None
        self.static_scaler = StandardScaler() if self.config.scale_static else None
        self.target_scaler = StandardScaler() if self.config.scale_target else None

        # Label encoders for categorical features
        self.well_type_encoder = LabelEncoder()
        self.aquifer_type_encoder = LabelEncoder()
        self.lithology_encoder = LabelEncoder()
        self.aquifer_0_aquifer_encoder = LabelEncoder()
        self.litho_supergroup_encoder = LabelEncoder()
        self.lulc_encoder = LabelEncoder()
        self.state_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()

        # Category-to-index mappings (needed for embeddings)
        self.embedding_metadata = {}

        # Flag to track if scalers are fitted
        self.is_fitted = False

    def _safe_encode(self, value: str, encoder: LabelEncoder) -> int:
        """
        Safely encode a categorical value, mapping unseen categories to 'unknown'.

        Args:
            value: The categorical value to encode
            encoder: The fitted LabelEncoder

        Returns:
            The encoded index (uses 'unknown' index for unseen categories)
        """
        if value in encoder.classes_:
            return encoder.transform([value])[0]
        else:
            # Map unseen category to 'unknown'
            return encoder.transform(['unknown'])[0]

    def _extract_features_from_samples(
        self,
        samples: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract feature arrays from samples for fitting scalers.

        Returns:
            (all_gwl_values, historical_values, forecast_values, static_values, lulc_values)

        Note: all_gwl_values pools sequence GWL (where present) + target GWL into
        one array. This is used for both the gwl_scaler and target_scaler (same
        physical quantity, one distribution).
        """
        n_samples = len(samples)
        num_timesteps = samples[0]['sequence'].shape[0]

        # Pool ALL GWL values: sequence (present) + target
        # These are the same physical quantity — one scaler for all
        all_gwl_values = self._collect_all_gwl(
            samples,
            gwl_idx=self.config.gwl_idx,
            is_present_idx=self.config.gwl_is_present_idx
        ).reshape(-1, 1)

        # Extract historical features from sequence (indices 4-10)
        # Shape: [n_samples * num_timesteps, 6]
        historical_values = []
        for s in samples:
            hist = s['sequence'][:, self.config.hist_start_idx:self.config.hist_end_idx]
            historical_values.append(hist)
        historical_values = np.vstack(historical_values)  # [n_samples * num_timesteps, 6]

        # Extract forecast features (separate from sequence)
        # Shape: [n_samples, 6]
        forecast_values = np.array([s['forecast_features'] for s in samples])

        # Extract static continuous features (12 dims, fixed order):
        #   0: well_depth (well_depth at index 0 so it can be extracted independently
        #      when --no-static-features is set)
        #   1: elevation, 2: stream_order, 3: latitude, 4: longitude
        #   5: gwl_anomaly (current_gwl - station/month mean from train)
        #   6: mean_annual_rainfall, 7: annual_rainfall_std
        #   8: station_mean_gwl, 9: station_gwl_amplitude
        #   10: station_delta_mean (typical 3-month delta from train period)
        #   11: station_delta_std  (volatility of 3-month deltas)
        static_values = np.array([
            [
                s.get('well_depth', 0.0),
                s['elevation'],
                s['stream_order'],
                s['latitude'],
                s['longitude'],
                s.get('gwl_anomaly', 0.0),
                s.get('mean_annual_rainfall', 0.0),
                s.get('annual_rainfall_std', 0.0),
                s.get('station_mean_gwl', 0.0),
                s.get('station_gwl_amplitude', 0.0),
                s.get('station_delta_mean', 0.0),
                s.get('station_delta_std', 0.0),
            ]
            for s in samples
        ])

        # Extract all LULC values (historical and forecast)
        lulc_values = []
        for s in samples:
            lulc_values.extend(s['historical_lulc'])
            lulc_values.append(s['forecast_lulc'])

        return all_gwl_values, historical_values, forecast_values, static_values, lulc_values

    @staticmethod
    def _collect_all_gwl(samples: List[Dict], gwl_idx: int = 0, is_present_idx: int = 1) -> np.ndarray:
        """
        Collect ALL GWL readings from a list of samples into one flat array.

        Sources:
        - sequence[:, gwl_idx] where is_present == 1.0 (historical readings)
        - target_gwl (future reading)

        Note: current_gwl is already the last present value in the sequence,
        so it's included via the sequence extraction (no double-counting).

        Returns:
            1D numpy array of all GWL values in the split
        """
        all_gwl = []
        for s in samples:
            # Sequence GWL where present
            for timestep in s['sequence']:
                if timestep[is_present_idx] == 1.0:
                    all_gwl.append(timestep[gwl_idx])
            # Target GWL
            all_gwl.append(s['target_gwl'])
        return np.array(all_gwl)

    @staticmethod
    def log_gwl_distribution(all_samples: Dict[str, List[Dict]],
                              gwl_idx: int = 0, is_present_idx: int = 1,
                              output_dir: str = None):
        """
        Print GWL distribution stats per split — both pooled and per-station.
        Optionally saves per-station CSV to output_dir.
        """
        print("\n" + "=" * 70)
        print("GWL DISTRIBUTION PER SPLIT")
        print("=" * 70)

        for split in ['train', 'val', 'test']:
            if split not in all_samples or not all_samples[split]:
                continue
            samples = all_samples[split]
            all_gwl = LSTMFeatureScaler._collect_all_gwl(
                samples, gwl_idx=gwl_idx, is_present_idx=is_present_idx
            )
            n_samples = len(samples)
            n_readings = len(all_gwl)
            q1, q3 = np.percentile(all_gwl, [25, 75])
            iqr = q3 - q1

            print(f"\n  [{split}] {n_samples} samples → {n_readings} GWL readings (pooled)")
            print(f"    mean={all_gwl.mean():.2f}  std={all_gwl.std():.2f}")
            print(f"    min={all_gwl.min():.2f}  Q1={q1:.2f}  median={np.median(all_gwl):.2f}  Q3={q3:.2f}  max={all_gwl.max():.2f}")

            # Per-station summary
            station_samples = defaultdict(list)
            for s in samples:
                station_samples[s['station_code']].append(s)

            station_means = []
            for sc, slist in station_samples.items():
                st_gwl = []
                for s in slist:
                    st_gwl.extend(
                        LSTMFeatureScaler._collect_sample_gwl(s, gwl_idx, is_present_idx)
                    )
                state = slist[0].get('state', 'unknown')
                district = slist[0].get('district', 'unknown')
                station_means.append((sc, state, district, len(slist), np.mean(st_gwl), np.std(st_gwl)))

            station_means.sort(key=lambda x: x[4])
            print(f"    {len(station_means)} stations:")
            for sc, state, district, n, m, s in station_means:
                print(f"      {sc} | {state} | {district}: {n} samples, GWL mean={m:.2f} std={s:.2f}")

            # Save per-station CSV
            if output_dir:
                import csv, os
                csv_path = os.path.join(output_dir, f"gwl_distribution_{split}.csv")
                os.makedirs(output_dir, exist_ok=True)
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["station_code", "state", "district", "n_samples", "gwl_delta_mean", "gwl_delta_std"])
                    for sc, state, district, n, m, s in station_means:
                        writer.writerow([sc, state, district, n, f"{m:.4f}", f"{s:.4f}"])
                print(f"    Saved: {csv_path}")

        print("=" * 70)

    @staticmethod
    def _collect_sample_gwl(sample: Dict, gwl_idx: int = 0, is_present_idx: int = 1) -> List[float]:
        """Collect all GWL readings from a single sample (sequence present + target)."""
        gwl_vals = []
        for timestep in sample['sequence']:
            if timestep[is_present_idx] == 1.0:
                gwl_vals.append(timestep[gwl_idx])
        gwl_vals.append(sample['target_gwl'])
        return gwl_vals

    @staticmethod
    def filter_gwl_outliers(samples: List[Dict], split_name: str = "",
                            outlier_ratio: float = 10.0,
                            outlier_abs_diff: float = 30.0) -> List[Dict]:
        """
        Remove samples with likely data errors based on current→target GWL ratio.

        Drop if BOTH conditions met:
        - ratio: target/current > outlier_ratio OR current/target > outlier_ratio
        - abs diff: abs(target - current) > outlier_abs_diff meters

        This catches sensor malfunctions (e.g. 5m → 400,000m) while preserving
        valid cross-station variability (5m well and 30m well are both fine).

        Args:
            samples: List of sample dictionaries
            split_name: Label for logging
            outlier_ratio: Max allowed ratio between current and target GWL
            outlier_abs_diff: Min absolute difference (meters) to trigger drop

        Returns:
            Filtered list of samples
        """
        if not samples:
            return samples

        label = f"[{split_name}] " if split_name else ""

        filtered = []
        dropped = 0

        for s in samples:
            current = s['current_gwl']
            target = s['target_gwl']
            abs_diff = abs(target - current)

            # Compute ratio safely (handle zero and negative values)
            abs_current = abs(current)
            abs_target = abs(target)
            if abs_current > 0 and abs_target > 0:
                ratio = max(abs_target / abs_current, abs_current / abs_target)
            elif abs_current == 0 and abs_target == 0:
                ratio = 1.0
            else:
                # One is zero, other isn't — use abs_diff only
                ratio = float('inf') if abs_diff > 0 else 1.0

            # Drop only if BOTH ratio AND abs_diff thresholds exceeded
            if ratio > outlier_ratio and abs_diff > outlier_abs_diff:
                dropped += 1
            else:
                filtered.append(s)

        print(f"  {label}Outlier filter (ratio>{outlier_ratio} AND diff>{outlier_abs_diff}m): "
              f"dropped {dropped}/{len(samples)} ({100 * dropped / len(samples):.1f}%)")
        print(f"  {label}Remaining: {len(filtered)} samples")

        return filtered

    def fit(self, all_samples: Dict[str, List[Dict]], output_dir: str = None):
        """
        Fit all scalers on appropriate data subsets.

        This method implements a separation of concerns critical for ML:
        - Numeric scalers (gwl, historical, forecast, static, target) are fitted
          ONLY on training samples to prevent data leakage
        - Categorical label encoders are fitted on ALL samples (train+val+test)
          to ensure every possible category is represented, avoiding "unseen
          category" errors during inference
        - Ratio+abs_diff outlier removal (catches data errors like 5m→400,000m
          while preserving valid cross-station variability)

        Args:
            all_samples: Dictionary with keys "train", "val", "test", each containing
                         list of samples.
                         - Numeric scalers fit only on all_samples["train"]
                         - Categorical encoders fit on all_samples["train"] + ["val"] + ["test"]

        Raises:
            ValueError: If no training samples are found.

        Reference: lstm_design.md (Normalization - Critical Rules)
        """
        print("Fitting scalers...")

        if "train" not in all_samples:
            raise ValueError("all_samples dict must have 'train' key")

        # ====================================================================
        # GWL OUTLIER REMOVAL (ratio + abs diff check per sample)
        # Drop samples where current→target jump is implausibly large,
        # catching data errors while preserving valid cross-station variability.
        # ====================================================================
        print(f"\nGWL outlier removal (ratio>{self.config.outlier_ratio} "
              f"AND diff>{self.config.outlier_abs_diff}m)...")
        for split in ['train', 'val', 'test']:
            if split in all_samples:
                before = len(all_samples[split])
                all_samples[split] = self.filter_gwl_outliers(
                    all_samples[split], split_name=split,
                    outlier_ratio=self.config.outlier_ratio,
                    outlier_abs_diff=self.config.outlier_abs_diff
                )
                after = len(all_samples[split])
                if before != after:
                    print(f"  {split}: {before} → {after}")

        # Log GWL distribution per split (after outlier removal)
        self.log_gwl_distribution(
            all_samples,
            gwl_idx=self.config.gwl_idx,
            is_present_idx=self.config.gwl_is_present_idx,
            output_dir=output_dir,
        )

        train_samples = all_samples["train"]
        static_all_samples = (
            all_samples.get("train", []) +
            all_samples.get("val", []) +
            all_samples.get("test", [])
        )

        if not train_samples:
            raise ValueError("No training samples found in all_samples['train']")

        # Auto-detect hist_end_idx from sample width (handles drop-NDVI/SM mode where
        # per-timestep is 13 features vs 15 with NDVI/SM kept). When use_only_gwl
        # is on, sequence width is 6 → hist range is empty [6,6) → no historical
        # features to scale; disable the historical scaler.
        seq_width = train_samples[0]['sequence'].shape[1]
        self.config.hist_end_idx = seq_width
        if self.config.hist_end_idx <= self.config.hist_start_idx:
            self.historical_scaler = None
            print(f"Sequence width: {seq_width} → no historical features (use_only_gwl mode); historical_scaler disabled")
        else:
            print(f"Sequence width: {seq_width} → hist range = "
                  f"[{self.config.hist_start_idx}, {self.config.hist_end_idx})")

        num_static_samples = len(static_all_samples)
        print(f"Total samples for categorical encoders: {num_static_samples}")
        print(f"Training samples (for numeric scalers): {len(train_samples)}")

        # ====================================================================
        # FIT NUMERIC SCALERS ON TRAINING DATA ONLY
        # CRITICAL: This prevents data leakage!
        # NOTE: Features may contain NaN from missing windows.
        # sklearn StandardScaler handles NaN correctly during fit
        # (computes mean/var ignoring NaN). We replace NaN → 0.0 after
        # transform in transform_sample().
        # ====================================================================
        print("\nFitting numeric scalers (on training data only)...")

        # Extract features
        all_gwl_values, historical_values, forecast_values, static_values, lulc_values = \
            self._extract_features_from_samples(train_samples)

        num_timesteps = train_samples[0]['sequence'].shape[0]
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Timesteps per sequence: {num_timesteps}")
        print(f"  All GWL values shape: {all_gwl_values.shape} (sequence present + target pooled)")
        print(f"  Historical values shape: {historical_values.shape}")
        print(f"  Forecast values shape: {forecast_values.shape}")
        print(f"  Static values shape: {static_values.shape}")

        # Log NaN counts before fitting
        nan_count_hist = np.isnan(historical_values).sum()
        nan_count_forecast = np.isnan(forecast_values).sum()
        nan_count_static = np.isnan(static_values).sum()
        if nan_count_hist > 0:
            nan_pct = 100 * nan_count_hist / historical_values.size
            print(f"  INFO: {nan_count_hist} NaN values in historical features ({nan_pct:.1f}%) — skipped during scaler fit")
        if nan_count_forecast > 0:
            nan_pct = 100 * nan_count_forecast / forecast_values.size
            print(f"  INFO: {nan_count_forecast} NaN values in forecast features ({nan_pct:.1f}%) — skipped during scaler fit")
        if nan_count_static > 0:
            nan_pct = 100 * nan_count_static / static_values.size
            print(f"  INFO: {nan_count_static} NaN values in static features ({nan_pct:.1f}%) — skipped during scaler fit")

        # Fit scalers
        if self.gwl_scaler:
            self.gwl_scaler.fit(all_gwl_values)
            print(f"  GWL scaler: mean={self.gwl_scaler.mean_[0]:.2f}, std={self.gwl_scaler.scale_[0]:.2f}"
                  f" (fitted on {len(all_gwl_values)} pooled readings)")

        if self.historical_scaler:
            self.historical_scaler.fit(historical_values)
            print(f"  Historical scaler: fitted on {historical_values.shape[1]} features")
            # Handle zero variance columns by setting scale to 1 (no scaling)
            zero_var_mask = self.historical_scaler.var_ == 0
            if zero_var_mask.any():
                self.historical_scaler.scale_[zero_var_mask] = 1.0
                print(f"  WARNING: Zero variance in {zero_var_mask.sum()} historical columns — setting scale=1")

        if self.forecast_scaler:
            self.forecast_scaler.fit(forecast_values)
            print(f"  Forecast scaler: fitted on {forecast_values.shape[1]} features")
            # Handle zero variance columns
            zero_var_mask = self.forecast_scaler.var_ == 0
            if zero_var_mask.any():
                self.forecast_scaler.scale_[zero_var_mask] = 1.0
                print(f"  WARNING: Zero variance in {zero_var_mask.sum()} forecast columns — setting scale=1")

        if self.static_scaler:
            self.static_scaler.fit(static_values)
            print(f"  Static scaler: mean={self.static_scaler.mean_}, std={self.static_scaler.scale_}")

        # Use deep copy of gwl_scaler for target_scaler (same scaling, independent object)
        if self.target_scaler:
            self.target_scaler = copy.deepcopy(self.gwl_scaler)
            print(f"  Target scaler: mean={self.target_scaler.mean_[0]:.2f}, std={self.target_scaler.scale_[0]:.2f}")

        # ====================================================================
        # FIT CATEGORICAL ENCODERS ON ALL SAMPLES
        # CRITICAL: This ensures all categories are represented, avoiding
        # "unseen category" errors during inference on val/test data
        # ====================================================================
        print("\nFitting categorical encoders (on all samples to avoid unseen categories)...")

        well_types = [s['well_type'] for s in static_all_samples] + ['unknown']
        aquifer_types = [s['aquifer_type'] for s in static_all_samples] + ['unknown']
        lithologies = [s['lithology'] for s in static_all_samples] + ['unknown']
        aquifer_0_aquifer = [s.get('aquifer_0_aquifer', 'no_data') for s in static_all_samples] + ['unknown']
        litho_supergroup = [s.get('litho_supergroup', 'no_data') for s in static_all_samples] + ['unknown']
        states = [s.get('state', 'no_data') for s in static_all_samples] + ['unknown']
        districts = [s.get('district', 'no_data') for s in static_all_samples] + ['unknown']

        # Flatten all LULC values from all windows across all samples
        lulc_cats = []
        for s in static_all_samples:
            lulc_cats.extend(s.get('historical_lulc', []))
            lulc_cats.append(s.get('forecast_lulc', 'no_data'))
        lulc_cats = list(set([v for v in lulc_cats if v is not None])) + ['unknown']

        self.well_type_encoder.fit(well_types)
        self.aquifer_type_encoder.fit(aquifer_types)
        self.lithology_encoder.fit(lithologies)
        self.aquifer_0_aquifer_encoder.fit(aquifer_0_aquifer)
        self.litho_supergroup_encoder.fit(litho_supergroup)
        self.lulc_encoder.fit(lulc_cats)
        self.state_encoder.fit(states)
        self.district_encoder.fit(districts)

        print(f"  Well types: {list(self.well_type_encoder.classes_)}")
        print(f"  Aquifer types: {list(self.aquifer_type_encoder.classes_)}")
        print(f"  Lithologies: {list(self.lithology_encoder.classes_)}")
        print(f"  Aquifer 0 Aquifer: {list(self.aquifer_0_aquifer_encoder.classes_)}")
        print(f"  Litho Supergroup: {list(self.litho_supergroup_encoder.classes_)}")
        print(f"  LULC types: {list(self.lulc_encoder.classes_)}")
        print(f"  States ({len(self.state_encoder.classes_)}): {list(self.state_encoder.classes_)}")
        print(f"  Districts ({len(self.district_encoder.classes_)}): {len(self.district_encoder.classes_)} unique")

        # Store embedding metadata
        self.embedding_metadata = {
            'lithology_to_idx': {
                cat: idx for idx, cat in enumerate(self.lithology_encoder.classes_)
            },
            'well_type_to_idx': {
                cat: idx for idx, cat in enumerate(self.well_type_encoder.classes_)
            },
            'aquifer_to_idx': {
                cat: idx for idx, cat in enumerate(self.aquifer_type_encoder.classes_)
            },
            'aquifer_0_aquifer_to_idx': {
                cat: idx for idx, cat in enumerate(self.aquifer_0_aquifer_encoder.classes_)
            },
            'litho_supergroup_to_idx': {
                cat: idx for idx, cat in enumerate(self.litho_supergroup_encoder.classes_)
            },
            'lulc_to_idx': {
                cat: idx for idx, cat in enumerate(self.lulc_encoder.classes_)
            },
            'state_to_idx': {
                cat: idx for idx, cat in enumerate(self.state_encoder.classes_)
            },
            'district_to_idx': {
                cat: idx for idx, cat in enumerate(self.district_encoder.classes_)
            },
            'num_lithology_types': len(self.lithology_encoder.classes_),
            'num_well_types': len(self.well_type_encoder.classes_),
            'num_aquifer_types': len(self.aquifer_type_encoder.classes_),
            'num_aquifer_0_aquifer_types': len(self.aquifer_0_aquifer_encoder.classes_),
            'num_litho_supergroup_types': len(self.litho_supergroup_encoder.classes_),
            'num_lulc_types': len(self.lulc_encoder.classes_),
            'num_state_types': len(self.state_encoder.classes_),
            'num_district_types': len(self.district_encoder.classes_)
        }

        self.is_fitted = True
        print("\nScalers fitted successfully!")

    def _scale_gwl_in_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Scales the GWL values in a sequence, only for timesteps where GWL is present.
        Missing GWL values are np.nan (set in data_preparation.py).
        After scaling present values, nan is replaced with 0.0 in scaled space
        (= mean = "no signal"), consistent with how NaN historical features are handled.
        """
        if not self.gwl_scaler:
            return sequence

        gwl_col = sequence[:, self.config.gwl_idx:self.config.gwl_idx+1].copy()

        # Identify missing (nan) vs present values
        nan_mask = np.isnan(gwl_col)

        # Scale only present (non-nan) GWL values
        present_mask = ~nan_mask.flatten()
        if present_mask.any():
            gwl_col[present_mask] = self.gwl_scaler.transform(gwl_col[present_mask])

        # Replace nan with 0.0 in scaled space (= mean = "no signal")
        gwl_col[nan_mask] = 0.0

        sequence[:, self.config.gwl_idx:self.config.gwl_idx+1] = gwl_col

        # Scale gwl_diff the same way (same units as gwl_delta)
        diff_col = sequence[:, self.config.gwl_diff_idx:self.config.gwl_diff_idx+1].copy()
        diff_reliable = sequence[:, self.config.gwl_diff_reliable_idx]
        diff_present = diff_reliable == 1.0
        if diff_present.any():
            diff_col[diff_present] = self.gwl_scaler.transform(diff_col[diff_present])
        diff_col[~diff_present.reshape(-1, 1)] = 0.0
        sequence[:, self.config.gwl_diff_idx:self.config.gwl_diff_idx+1] = diff_col

        return sequence

    def transform_sample(self, sample: Dict) -> Dict[str, np.ndarray]:
        """
        Transform a single sample using fitted scalers.

        Args:
            sample: Sample dictionary

        Returns:
            Dictionary with scaled features ready for model input

        Reference: lstm_design.md (Input structure)
        """
        if not self.is_fitted:
            raise RuntimeError("Scalers not fitted. Call fit() first.")

        sequence = sample['sequence'].copy()
        num_timesteps = sequence.shape[0]

        # 1. Scale GWL values in sequence (only where is_present=1.0)
        sequence = self._scale_gwl_in_sequence(sequence)

        # 2. GWL presence flag and Temporal encoding (NOT scaled)
        # Indices 1 (is_present), 2-4 (temporal) remain unchanged

        # 3. Scale historical features in sequence (NaN-aware)
        if self.historical_scaler:
            hist = sequence[:, self.config.hist_start_idx:self.config.hist_end_idx]
            # Capture NaN positions BEFORE transform
            nan_mask_hist = np.isnan(hist)
            hist_scaled = self.historical_scaler.transform(hist)
            # Replace NaN with 0.0 = mean in standardized space ("no signal")
            hist_scaled[nan_mask_hist] = 0.0
            sequence[:, self.config.hist_start_idx:self.config.hist_end_idx] = hist_scaled

        # 4. Scale forecast features (separate from sequence, NaN-aware)
        forecast_features = sample['forecast_features'].copy().reshape(1, -1)
        if self.forecast_scaler:
            nan_mask_forecast = np.isnan(forecast_features)
            forecast_features = self.forecast_scaler.transform(forecast_features)
            forecast_features[nan_mask_forecast] = 0.0

        # 5. Scale static continuous features (NaN-aware) — 12 dims, fixed order
        # See ScalerConfig docstring for layout
        static_continuous = np.array([[
            sample.get('well_depth', 0.0),
            sample['elevation'],
            sample['stream_order'],
            sample['latitude'],
            sample['longitude'],
            sample.get('gwl_anomaly', 0.0),
            sample.get('mean_annual_rainfall', 0.0),
            sample.get('annual_rainfall_std', 0.0),
            sample.get('station_mean_gwl', 0.0),
            sample.get('station_gwl_amplitude', 0.0),
            sample.get('station_delta_mean', 0.0),
            sample.get('station_delta_std', 0.0),
        ]])
        if self.static_scaler:
            nan_mask_static = np.isnan(static_continuous)
            static_continuous = self.static_scaler.transform(static_continuous)
            static_continuous[nan_mask_static] = 0.0

        # 6. Encode categorical features (map unseen categories to 'unknown')
        lithology_idx = self._safe_encode(sample['lithology'], self.lithology_encoder)
        well_type_idx = self._safe_encode(sample['well_type'], self.well_type_encoder)
        aquifer_idx = self._safe_encode(sample['aquifer_type'], self.aquifer_type_encoder)
        aquifer_0_aquifer_idx = self._safe_encode(
            sample.get('aquifer_0_aquifer', 'no_data'), self.aquifer_0_aquifer_encoder)
        litho_supergroup_idx = self._safe_encode(
            sample.get('litho_supergroup', 'no_data'), self.litho_supergroup_encoder)
        state_idx = self._safe_encode(
            sample.get('state', 'no_data'), self.state_encoder)
        district_idx = self._safe_encode(
            sample.get('district', 'no_data'), self.district_encoder)

        # Encode LULC features
        forecast_lulc_idx = self._safe_encode(sample['forecast_lulc'], self.lulc_encoder)
        historical_lulc_indices = np.array([
            self._safe_encode(val, self.lulc_encoder) for val in sample['historical_lulc']
        ])

        # 7. Scale target GWL
        target_gwl = np.array([[sample['target_gwl']]])
        if self.target_scaler:
            target_gwl = self.target_scaler.transform(target_gwl)

        # Final safety net — should not trigger after NaN handling above
        seq_flat = sequence[:, self.config.hist_start_idx:self.config.hist_end_idx]
        if np.isnan(seq_flat).any():
            nan_count = np.isnan(seq_flat).sum()
            print(f"  WARNING: Found {nan_count} residual NaN values in sequence, replacing with 0")
            sequence[:, self.config.hist_start_idx:self.config.hist_end_idx] = \
                np.nan_to_num(seq_flat, nan=0.0)
        if np.isnan(forecast_features).any():
            nan_count = np.isnan(forecast_features).sum()
            print(f"  WARNING: Found {nan_count} residual NaN values in forecast features, replacing with 0")
            forecast_features = np.nan_to_num(forecast_features, nan=0.0)

        return {
            'sequence': sequence,  # [num_timesteps, 12] - historical only
            'forecast_features': forecast_features.flatten(),  # [6] - for conditioning
            'static_continuous': static_continuous.flatten(),  # [10] - well_depth, elev, stream_ord, lat, lon, gwl_anom, mean_ann_rain, ann_rain_std, st_mean_gwl, st_gwl_amp
            'lithology_idx': lithology_idx,
            'well_type_idx': well_type_idx,
            'aquifer_idx': aquifer_idx,
            'aquifer_0_aquifer_idx': aquifer_0_aquifer_idx,
            'litho_supergroup_idx': litho_supergroup_idx,
            'state_idx': state_idx,
            'district_idx': district_idx,
            'forecast_lulc_idx': forecast_lulc_idx,
            'historical_lulc_indices': historical_lulc_indices,
            'target_gwl': target_gwl[0, 0],
            'target_gwl_raw': sample.get('target_gwl_raw', sample['target_gwl']),  # Absolute target for reconstruction
            'current_gwl_raw': sample['current_gwl'],  # For trend classification
            # Prithvi tile index: passthrough int (an index into tile_manifest.pkl,
            # NOT a feature to scale). Present only when data prep ran with
            # use_prithvi; -1 means "not annotated / Prithvi off". train.py turns
            # this into a tensor and feeds it to the live Prithvi projector.
            'tile_idx': int(sample.get('tile_idx', -1)),
            'metadata': {
                'station_code': sample['station_code'],
                'state': sample.get('state', 'unknown'),
                'district': sample.get('district', 'unknown'),
                'current_date': sample['current_date'],
                'target_date': sample['target_date']
            }
        }

    def transform(self, samples: List[Dict]) -> List[Dict[str, np.ndarray]]:
        """
        Transform multiple samples.

        Args:
            samples: List of sample dictionaries

        Returns:
            List of transformed samples

        Reference: lstm_design.md (Use SAME scaler for val/test)
        """
        return [self.transform_sample(s) for s in samples]

    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predicted GWL values.

        Args:
            predictions: Normalized predictions

        Returns:
            Denormalized predictions in original scale

        Reference: lstm_design.md (Denormalize predictions)
        """
        if self.target_scaler is None:
            return predictions

        return self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    def inverse_transform_gwl(self, gwl_values: np.ndarray) -> np.ndarray:
        """
        Inverse transform GWL values from sequences.

        Args:
            gwl_values: Normalized GWL values

        Returns:
            Denormalized GWL values
        """
        if self.gwl_scaler is None:
            return gwl_values

        return self.gwl_scaler.inverse_transform(gwl_values.reshape(-1, 1)).flatten()

    def write_transform_audit(
        self,
        sample: Dict,
        split_name: str,
        output_path: str,
    ) -> None:
        """Write a plain-text pre/post scaling audit for one sample.

        Shows fitted scaler params, per-channel sequence stats before vs after
        scaling, full pre/post rows for a few representative timesteps, and
        flags any |z| > 3 in the scaled tensor (often surfaces val/test
        samples sitting far outside the train distribution).
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not self.is_fitted:
            with open(output_path, "w") as f:
                f.write("Scaler not fitted — nothing to audit.\n")
            return

        L = []
        L.append(f"=== Scaler audit: {split_name.upper()} split, sample 0 ===")
        L.append("")
        L.append("Identity (cross-ref with sample_audit_*.txt)")
        meta_keys = ("station_code", "current_date", "target_date")
        for k in meta_keys:
            v = sample.get(k, "?")
            L.append(f"  {k}: {v}")
        L.append("")

        # ---------- Fitted scaler params ----------
        L.append("Fitted scaler params (numeric scalers fit on TRAIN only)")
        if self.gwl_scaler is not None:
            L.append(f"  gwl_scaler:        mean={self.gwl_scaler.mean_[0]:+.4f}  "
                     f"std={self.gwl_scaler.scale_[0]:.4f}")
        if self.target_scaler is not None:
            L.append(f"  target_scaler:     mean={self.target_scaler.mean_[0]:+.4f}  "
                     f"std={self.target_scaler.scale_[0]:.4f}  (= gwl_scaler)")
        # Channel labels for hist range (auto-detected at fit time)
        hist_n = self.config.hist_end_idx - self.config.hist_start_idx
        # Full sequence width tells us if NDVI/SM are present
        if hist_n == 7:
            hist_labels = ["rain", "temp", "et", "runoff",
                           "cum_rain", "rain_anom", "cum_anom"]
        elif hist_n == 9:
            hist_labels = ["rain", "temp", "et", "runoff", "ndvi", "sm",
                           "cum_rain", "rain_anom", "cum_anom"]
        else:
            hist_labels = [f"hist_{i}" for i in range(hist_n)]
        if self.historical_scaler is not None:
            L.append(f"  historical_scaler ({hist_n} dims):")
            for i, name in enumerate(hist_labels):
                L.append(f"    {name:<12}  mean={self.historical_scaler.mean_[i]:+.4f}  "
                         f"std={self.historical_scaler.scale_[i]:.4f}")
        if self.forecast_scaler is not None:
            n_fc = len(self.forecast_scaler.mean_)
            if n_fc == 2:
                fc_labels = ["rain_sum", "temp_mean"]
            elif n_fc == 4:
                fc_labels = ["rain_sum", "temp_mean", "et_sum", "runoff_sum"]
            else:
                fc_labels = [f"fc_{i}" for i in range(n_fc)]
            L.append(f"  forecast_scaler ({n_fc} dims):")
            for i, name in enumerate(fc_labels):
                L.append(f"    {name:<12}  mean={self.forecast_scaler.mean_[i]:+.4f}  "
                         f"std={self.forecast_scaler.scale_[i]:.4f}")
        if self.static_scaler is not None:
            n_st = len(self.static_scaler.mean_)
            st_labels = [
                "well_depth", "elevation", "stream_order", "lat", "lon",
                "gwl_anomaly", "mean_ann_rain", "ann_rain_std",
                "stn_mean_gwl", "stn_amp", "stn_delta_mean", "stn_delta_std",
            ][:n_st]
            L.append(f"  static_scaler ({n_st} dims):")
            for i, name in enumerate(st_labels):
                L.append(f"    {name:<14}  mean={self.static_scaler.mean_[i]:+.4f}  "
                         f"std={self.static_scaler.scale_[i]:.4f}")
        L.append("")

        # ---------- Build pre/post rows ----------
        pre_seq = np.asarray(sample["sequence"]).astype(float)  # (T, F)
        T, F = pre_seq.shape
        transformed = self.transform_sample(sample)
        post_seq = np.asarray(transformed["sequence"]).astype(float)
        pre_fc = np.asarray(sample["forecast_features"]).astype(float)
        post_fc = np.asarray(transformed["forecast_features"]).astype(float)
        pre_static = np.array([
            sample.get("well_depth", 0.0),
            sample.get("elevation", 0.0),
            sample.get("stream_order", 0.0),
            sample.get("latitude", 0.0),
            sample.get("longitude", 0.0),
            sample.get("gwl_anomaly", 0.0),
            sample.get("mean_annual_rainfall", 0.0),
            sample.get("annual_rainfall_std", 0.0),
            sample.get("station_mean_gwl", 0.0),
            sample.get("station_gwl_amplitude", 0.0),
            sample.get("station_delta_mean", 0.0),
            sample.get("station_delta_std", 0.0),
        ], dtype=float)
        post_static = np.asarray(transformed["static_continuous"]).astype(float)
        pre_target = float(sample["target_gwl"])
        post_target = float(transformed["target_gwl"])

        # ---------- Per-channel sequence stats: pre vs post ----------
        all_labels = (
            ["gwl_enc", "is_present", "gwl_diff", "gwl_diff_rel", "sin", "cos"]
            + hist_labels
        )
        if F != len(all_labels):
            all_labels = [f"c{i}" for i in range(F)]

        L.append(f"Per-channel sequence stats  (T={T} timesteps)")
        L.append(f"  {'idx':>3}  {'channel':<14}  "
                 f"{'pre_min':>10} {'pre_max':>10} {'pre_mean':>10} {'pre_std':>9}  "
                 f"{'post_min':>10} {'post_max':>10} {'post_mean':>10} {'post_std':>9}  scaled?")
        scaled_idx = set(range(self.config.hist_start_idx, self.config.hist_end_idx))
        scaled_idx.add(self.config.gwl_idx)
        for i, name in enumerate(all_labels):
            pre = pre_seq[:, i]
            post = post_seq[:, i]
            scaled = "yes" if i in scaled_idx else "no"
            # GWL channel: stats only on present timesteps (raw GWL is encoded
            # as 0 where missing, which would skew naive min/max)
            if i == self.config.gwl_idx:
                mask = pre_seq[:, self.config.gwl_is_present_idx] == 1.0
                pre = pre[mask] if mask.any() else pre
                post = post[mask] if mask.any() else post
                scaled = "yes (present-only stats)"
            if len(pre) == 0:
                pre_stats = post_stats = "  ——"
                L.append(f"  {i:>3}  {name:<14}  (no present timesteps)")
                continue
            L.append(
                f"  {i:>3}  {name:<14}  "
                f"{np.nanmin(pre):>10.3f} {np.nanmax(pre):>10.3f} "
                f"{np.nanmean(pre):>10.3f} {np.nanstd(pre):>9.3f}  "
                f"{np.nanmin(post):>10.3f} {np.nanmax(post):>10.3f} "
                f"{np.nanmean(post):>10.3f} {np.nanstd(post):>9.3f}  {scaled}"
            )
        L.append("")

        # ---------- Forecast features pre/post ----------
        L.append("Forecast features  (pre → post)")
        n_fc = len(pre_fc)
        if n_fc == 2:
            fc_labels = ["rain_sum", "temp_mean"]
        elif n_fc == 4:
            fc_labels = ["rain_sum", "temp_mean", "et_sum", "runoff_sum"]
        else:
            fc_labels = [f"fc_{i}" for i in range(n_fc)]
        for i, name in enumerate(fc_labels):
            L.append(f"  {name:<12}  {pre_fc[i]:+10.3f}  →  {post_fc[i]:+8.3f}")
        L.append("")

        # ---------- Static block pre/post ----------
        st_labels_full = [
            "well_depth", "elevation", "stream_order", "lat", "lon",
            "gwl_anomaly", "mean_ann_rain", "ann_rain_std",
            "stn_mean_gwl", "stn_amp", "stn_delta_mean", "stn_delta_std",
        ]
        L.append("Static block  (pre → post)")
        for i, name in enumerate(st_labels_full):
            L.append(f"  {name:<16}  {pre_static[i]:+12.4f}  →  {post_static[i]:+8.3f}")
        L.append("")

        # ---------- Target ----------
        L.append("Target")
        L.append(f"  target_gwl (label): pre={pre_target:+.4f}  →  post={post_target:+.4f}")
        L.append(f"  target_gwl_raw (absolute): {sample.get('target_gwl_raw', 0.0):.3f}")
        L.append(f"  current_gwl_raw (anchor):   {sample.get('current_gwl', 0.0):.3f}")
        L.append("")

        # ---------- Representative timestep rows ----------
        rep_rows = sorted({0, T // 2, T - 1})
        L.append("Representative timestep rows  (sequence[i,:], pre vs post)")
        for i in rep_rows:
            L.append(f"  i={i}  pre  = {np.array2string(pre_seq[i], precision=3, separator=', ')}")
            L.append(f"        post = {np.array2string(post_seq[i], precision=3, separator=', ')}")
        L.append("")

        # ---------- |z| > 3 flagging on scaled sequence ----------
        flagged = []
        for i in scaled_idx:
            if i >= F:
                continue
            col = post_seq[:, i]
            z = np.abs(col)
            for t in np.where(z > 3.0)[0]:
                flagged.append((int(t), int(i), all_labels[i], float(col[t])))
                if len(flagged) >= 20:
                    break
            if len(flagged) >= 20:
                break
        if flagged:
            L.append(f"|z| > 3 in scaled sequence  (first 20)")
            for t, i, name, v in flagged:
                L.append(f"  t={t:>3}  channel {i:>2} ({name:<12}) post-scale = {v:+.3f}")
        else:
            L.append("|z| > 3 in scaled sequence: none")
        L.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(L))

    def save(self, filepath: str):
        """
        Save all scalers and encoders to disk.

        Args:
            filepath: Path to save file

        Reference: lstm_design.md (Save ALL scalers for inference)
        """
        if not self.is_fitted:
            raise RuntimeError("Scalers not fitted. Call fit() first.")

        checkpoint = {
            'gwl_scaler': self.gwl_scaler,
            'historical_scaler': self.historical_scaler,
            'forecast_scaler': self.forecast_scaler,
            'static_scaler': self.static_scaler,
            'target_scaler': self.target_scaler,
            'well_type_encoder': self.well_type_encoder,
            'aquifer_type_encoder': self.aquifer_type_encoder,
            'lithology_encoder': self.lithology_encoder,
            'aquifer_0_aquifer_encoder': self.aquifer_0_aquifer_encoder,
            'litho_supergroup_encoder': self.litho_supergroup_encoder,
            'lulc_encoder': self.lulc_encoder,
            'state_encoder': self.state_encoder,
            'district_encoder': self.district_encoder,
            'embedding_metadata': self.embedding_metadata,
            'config': self.config
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"Saved scalers to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LSTMFeatureScaler':
        """
        Load scalers from disk.

        Args:
            filepath: Path to saved file

        Returns:
            LSTMFeatureScaler instance with loaded scalers
        """
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        scaler = cls(config=checkpoint['config'])
        scaler.gwl_scaler = checkpoint['gwl_scaler']
        scaler.historical_scaler = checkpoint['historical_scaler']
        scaler.forecast_scaler = checkpoint['forecast_scaler']
        scaler.static_scaler = checkpoint['static_scaler']
        scaler.target_scaler = checkpoint['target_scaler']
        scaler.well_type_encoder = checkpoint['well_type_encoder']
        scaler.aquifer_type_encoder = checkpoint['aquifer_type_encoder']
        scaler.lithology_encoder = checkpoint['lithology_encoder']
        scaler.aquifer_0_aquifer_encoder = checkpoint.get('aquifer_0_aquifer_encoder', LabelEncoder())
        scaler.litho_supergroup_encoder = checkpoint.get('litho_supergroup_encoder', LabelEncoder())
        scaler.lulc_encoder = checkpoint['lulc_encoder']
        scaler.state_encoder = checkpoint.get('state_encoder', LabelEncoder())
        scaler.district_encoder = checkpoint.get('district_encoder', LabelEncoder())
        scaler.embedding_metadata = checkpoint['embedding_metadata']
        scaler.is_fitted = True

        print(f"Loaded scalers from {filepath}")
        return scaler


if __name__ == '__main__':
    """
    Fit feature scalers on prepared data and save artifacts.

    Usage:
        python gwl_lstm/feature_scaler.py
        python gwl_lstm/feature_scaler.py --data-dir gwl_lstm/data
    """
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='Fit feature scalers for GWL LSTM forecasting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-dir', type=str, default='gwl_lstm/data',
                        help='Directory containing train/val/test pickle files')
    parser.add_argument('--outlier-ratio', type=float, default=10.0,
                        help='Max ratio between current and target GWL before flagging')
    parser.add_argument('--outlier-abs-diff', type=float, default=30.0,
                        help='Min absolute GWL diff (meters) to trigger outlier drop')
    args = parser.parse_args()

    data_dir = args.data_dir
    start = time.time()

    print("=" * 70)
    print("LSTM Feature Scaler Fitting")
    print("=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Outlier ratio: {args.outlier_ratio}")
    print(f"Outlier abs diff: {args.outlier_abs_diff}m")
    print()

    # Load prepared data
    print("Loading prepared datasets...")
    with open(os.path.join(data_dir, 'train_samples.pkl'), 'rb') as f:
        train_samples = pickle.load(f)

    with open(os.path.join(data_dir, 'val_samples.pkl'), 'rb') as f:
        val_samples = pickle.load(f)

    with open(os.path.join(data_dir, 'test_samples.pkl'), 'rb') as f:
        test_samples = pickle.load(f)

    print(f"  Training samples: {len(train_samples):,}")
    print(f"  Validation samples: {len(val_samples):,}")
    print(f"  Test samples: {len(test_samples):,}")

    # Create and fit scaler on all samples
    # Numeric scalers fit on train only, categorical encoders on all splits
    print("\n" + "=" * 70)
    print("Fitting scalers...")
    print("=" * 70)
    config = ScalerConfig(
        outlier_ratio=args.outlier_ratio,
        outlier_abs_diff=args.outlier_abs_diff
    )
    scaler = LSTMFeatureScaler(config=config)
    all_samples = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    scaler.fit(all_samples, output_dir=data_dir)

    # Save filtered samples back (outliers removed during fit)
    print("\n" + "=" * 70)
    print("Saving filtered samples...")
    print("=" * 70)
    for split in ['train', 'val', 'test']:
        path = os.path.join(data_dir, f'{split}_samples.pkl')
        with open(path, 'wb') as f:
            pickle.dump(all_samples[split], f)
        print(f"  Saved {len(all_samples[split]):,} {split} samples → {path}")

    # Transform a few samples to verify
    print("\n" + "=" * 70)
    print("Verifying transform on sample data...")
    print("=" * 70)
    train_transformed = scaler.transform(all_samples['train'][:5])
    val_transformed = scaler.transform(all_samples['val'][:5])

    print(f"  Transformed {len(train_transformed)} training samples (verification)")
    print(f"  Transformed {len(val_transformed)} validation samples (verification)")

    # Show sample structure
    sample = train_transformed[0]
    print(f"\n  Sample structure (first training sample):")
    print(f"    sequence shape: {sample['sequence'].shape} (historical only)")
    print(f"    sequence[0] (first timestep): {sample['sequence'][0][:5]}...")
    print(f"    forecast_features shape: {sample['forecast_features'].shape} (for conditioning)")
    print(f"    forecast_features: {sample['forecast_features'][:3]}...")
    print(f"    static_continuous: {sample['static_continuous']}")
    print(f"    lithology_idx: {sample['lithology_idx']}")
    print(f"    well_type_idx: {sample['well_type_idx']}")
    print(f"    aquifer_idx: {sample['aquifer_idx']}")
    print(f"    aquifer_0_aquifer_idx: {sample.get('aquifer_0_aquifer_idx', 'N/A')}")
    print(f"    litho_supergroup_idx: {sample.get('litho_supergroup_idx', 'N/A')}")
    print(f"    target_gwl (normalized): {sample['target_gwl']:.4f}")
    print(f"    current_gwl (raw): {sample['current_gwl_raw']:.2f}")

    # Scaler-transform audit: one detailed dump per split (always-on).
    # Shows fitted scaler params + per-channel pre/post stats + |z|>3 flags
    # for sample[0] of each split, so you can see how the train-fit scalers
    # actually land val/test samples.
    print("\n" + "=" * 70)
    print("Writing scaler transform audit...")
    print("=" * 70)
    debug_dir = os.path.join(data_dir, "debug")
    for split in ('train', 'val', 'test'):
        if not all_samples[split]:
            print(f"  [audit] {split}: empty split, skipping")
            continue
        out = os.path.join(debug_dir, f'scaler_audit_{split}.txt')
        scaler.write_transform_audit(all_samples[split][0], split, out)
        print(f"  [audit] wrote {out}")

    # Save scalers
    print("\n" + "=" * 70)
    print("Saving artifacts...")
    print("=" * 70)
    scaler_path = os.path.join(data_dir, 'scalers.pkl')
    scaler.save(scaler_path)

    # Test loading
    print("\n" + "=" * 70)
    print("Verifying scaler loading...")
    print("=" * 70)
    loaded_scaler = LSTMFeatureScaler.load(scaler_path)
    print("  Loaded successfully!")

    # Print embedding metadata
    print(f"\n  Embedding metadata:")
    for key, value in loaded_scaler.embedding_metadata.items():
        if isinstance(value, dict):
            print(f"    {key}: {value}")
        else:
            print(f"    {key}: {value}")

    print(f"\nAll artifacts saved to: {data_dir}/")
    print(f"Time taken: {time.time() - start:.1f}s")