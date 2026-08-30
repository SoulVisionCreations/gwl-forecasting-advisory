"""
Training Script for GWL Forecasting LSTM

This script ties together all components:
1. Load prepared data (from data_preparation.py)
2. Load fitted scalers (from feature_scaler.py)
3. Create PyTorch DataLoaders
4. Train the LSTM model (from model.py)
5. Evaluate on validation set
6. Save best model checkpoint

Supports two GWL encoding modes:
- Delta mode (default): Model predicts Δ(GWL) = target - current.
  Metrics computed on reconstructed absolute values: pred_abs = current + pred_delta.
  Trend derived from sign of predicted delta (positive = increase).
- Absolute mode: Model predicts raw GWL value. Standard denormalization for metrics.

Reference: lstm_design.md (Training Strategy section)

Usage:
    python gwl_lstm/train.py
    python gwl_lstm/train.py --batch-size 128 --lr 0.001 --epochs 100
    python gwl_lstm/train.py --device cuda --dropout 0.3
    python gwl_lstm/train.py --data-dir gwl_lstm/data --alpha 0.8 --beta 0.2
"""

import mlflow
import mlflow.pytorch
import os
import pickle
import argparse
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Import our modules
from gwlcore.model import GWLForecastLSTM, GWLForecastConditionedLSTM, MultiTaskLoss, classify_trend
from gwlcore.transformer_model import GWLForecastTransformer, GWLForecastConditionedTransformer
from gwlcore.tft_model import GWLForecastTFT
from gwl_geo_analysis import analyze_well_geography as _analyze_well_geography_fn
from gwl_plots import PlotGenerator
from gwlcore.feature_scaler import LSTMFeatureScaler, ScalerConfig

mlruns_dir = os.environ.get("MLFLOW_DIR")
print(mlruns_dir)

if mlruns_dir:
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
else:
    mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("gwl_lstm")


@dataclass
class TrainingConfig:
    """
    Configuration for training.

    Reference: lstm_design.md (Configuration section)
    """

    # Model selection: "lstm" or "transformer"
    model_type: str = "lstm"

    # If False, drop all static features (lat/lon/elevation/lithology/aquifer/etc.)
    # except well_depth. Forces the model to rely on temporal features.
    use_static_features: bool = True

    # Loss weighting / aggregation strategy:
    #   "none"    — pooled MSE (no weighting)
    #   "std"     — per-sample weight 1/std_w, sample-pooled mean (legacy "weighted")
    #   "perwell" — per-sample weight 1/var_w, mean per well, then mean across wells.
    #               Directly targets per-well R² (each well counts equally regardless
    #               of sample count or volatility).
    loss_weight_type: str = "std"
    # Legacy alias (kept for backward compat); when True maps to "std", False to "none".
    use_weighted_loss: bool = True
    # Dynamic top-trim for perwell loss: fraction (0–1) of highest-loss wells
    # to drop from the gradient each batch. 0.0 = off. Only active in "perwell"
    # mode. Idea: high-loss wells are often unlearnable noise; ignoring them
    # frees gradient for wells that follow a pattern.
    loss_trim_high: float = 0.0
    # RevIN: reversible per-sample instance normalization on the GWL channel
    # of the input sequence (using only is_present=1 timesteps for stats).
    # Target is normalized with the same per-sample (mean_s, std_s) so loss is
    # computed in normed space — equalizes gradient contribution across flat
    # and volatile wells. De-normalization happens at the metric stage so
    # reported R²/MAPE remain in real meters. Stateless: no per-well lookup.
    use_revin: bool = False
    # RevIN per-sample std floor (in scaled-delta units, ~1m real for floor=0.1).
    # Prevents catastrophic de-norm blow-up on flat-well samples where one
    # outlier or interpolation artifact in the lookback inflates std_s 100×
    # between training (~floor) and inference (~outlier-driven), causing the
    # model's trained pred_normed magnitude to de-norm to wildly wrong meters.
    # Floor caps both training target_normed and inference de-norm multiplier.
    revin_std_floor: float = 0.1
    # Minimum |target_delta| (meters) for inclusion in the *filtered* trend
    # accuracy metric. Flat-well samples with |Δ| below this contribute mostly
    # noise to trend classification (sign flips with sub-mm fluctuation), so
    # we report a second accuracy on samples where the well actually moved.
    # 0.0 = filter disabled.
    min_delta_for_trend_accuracy: float = 0.5

    # Single std threshold knob: train-derived per-station std (of target_delta).
    # Used both as the train/val data-prep flat filter AND as the eval-time
    # predictable cohort threshold (in plots, metric blocks, well_performance
    # CSVs). One value → one meaning everywhere.
    # When 0.0: no flat filter; eval prints only one (capped) metric block.
    min_station_target_std: float = 0.0

    # Optional state whitelist: when non-empty, restricts train+val to these
    # states; eval Block 2 (predictable cohort) additionally requires state
    # membership. Composes with std filter by AND. Test stays unfiltered.
    include_states: List[str] = field(default_factory=list)

    # Optional minimum per-station raw-observation mode-gap (days). When > 0,
    # train+val drop stations whose mode_gap is below threshold (densely
    # sampled flat wells); eval Block 2 requires the same. Test unfiltered.
    # Composes with std + state filters by AND. 0 = disabled.
    min_station_mode_gap_days: float = 0.0

    # Optional additive linear baseline path (residual learning).
    # When True, models add a Linear(13→1) head whose input is a fixed slice of
    # static + state + forecast features. Trains end-to-end alongside the main
    # path. The LSTM/TFT then implicitly learns the residual via the additive
    # output. Off by default for backward compatibility.
    use_linear_residual: bool = False

    # Model architecture
    num_timesteps: int = 10
    features_per_timestep: int = 13  # default: NDVI/SM dropped → 13. With --no-drop-ndvi-sm: 15. Auto-detected from data.
    num_forecast_features: int = 6  # Forecast features for conditioning
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    static_embedding_dim: int = 32
    fusion_hidden_dim: int = 64

    # Transformer-specific
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 128
    transformer_dropout: float = 0.2

    # TFT-specific
    tft_d_model: int = 64
    tft_n_heads: int = 4
    tft_lstm_layers: int = 1
    tft_dropout: float = 0.1

    # ── Prithvi-EO joint fine-tune (Step 8) ───────────────────────────
    # All inert unless use_prithvi=True (also requires model_type="tft").
    # composite_dir / zero_idx / period are read from tile_manifest.pkl in
    # data_dir; the fields below are the run-time knobs.
    use_prithvi: bool = False
    prithvi_model_dir: str = ""        # dir with Prithvi-EO .pt + prithvi_mae.py
    station_index_csv: str = ""        # gwl_stations.csv (safe_id, lat, lon)
    prithvi_proj_dim: int = 32
    lora_r: int = 16
    lora_alpha: int = 32
    ft_lr: float = 1e-4                # LoRA + projector LR
    forecaster_lr_scale: float = 10.0  # TFT (from-scratch) LR = ft_lr * this
    prithvi_n_tiles: int = 256         # unique tiles per train batch (grouped sampler)
    prithvi_samples_per_tile: int = 32  # samples per tile -> batch = n_tiles*spt
    prithvi_grad_ckpt: bool = False    # grad-checkpoint Prithvi blocks (memory)

    # Embeddings (will be updated from scaler)
    num_lithology_types: int = 8
    num_well_types: int = 5
    num_aquifer_types: int = 3
    num_aquifer_0_aquifer_types: int = 5
    num_litho_supergroup_types: int = 8
    num_lulc_types: int = 10
    num_state_types: int = 30
    num_district_types: int = 700
    # Smaller defaults: many categories have <10 stations, so high-dim embeddings overfit.
    # CLI flags can override per-experiment.
    lithology_embedding_dim: int = 3
    well_type_embedding_dim: int = 2
    aquifer_embedding_dim: int = 2
    aquifer_0_aquifer_embedding_dim: int = 2
    litho_supergroup_embedding_dim: int = 2
    lulc_embedding_dim: int = 2
    state_embedding_dim: int = 3
    district_embedding_dim: int = 3

    # Dropout
    dropout_rate: float = 0.2

    # Training hyperparameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0  # Max gradient norm (0 = disabled)

    # Loss weights
    alpha: float = 0.7  # Regression weight (MSE)
    beta: float = 0.3  # Trend weight (BCE on delta sign)
    use_uncertainty_weighting: bool = False

    # Huber loss delta: if > 0 use HuberLoss(delta) instead of MSE for regression.
    # Huber is quadratic for |error| < delta and linear beyond, reducing sensitivity
    # to large spikes without fully ignoring them.  0.0 = MSE (default).
    huber_delta: float = 0.0

    # GWL encoding mode (auto-detected from data, can be overridden)
    use_delta_gwl: bool = True  # Delta mode: model predicts Δ(GWL)

    # Forecast horizon (months). Used to derive the default prediction clamp:
    #   <=4 months  -> 20%
    #   > 4 months  -> 50%
    # Must match the horizon used at data-prep time.
    forecast_horizon_months: int = 6

    # Inference-time prediction clamp: cap |pred - current_gwl_raw| to
    # `pred_clamp_pct * |current_gwl_raw|`. Applied in validate(), saved test
    # preds, and diagnostic plots — prevents single-point spikes from
    # dominating metrics/plots. When None, the default is derived from
    # forecast_horizon_months above. Set to 0.0 to disable clamping.
    pred_clamp_pct: Optional[float] = None

    # Data loading
    batch_size: int = 64
    num_workers: int = 4
    shuffle_train: bool = True

    # Training schedule
    num_epochs: int = 100

    # Learning rate scheduler
    scheduler_mode: str = "min"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10

    # Paths
    data_dir: str = "gwl_lstm/data"
    output_dir: str = "gwl_lstm/runs"
    run_name: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    save_best_only: bool = True

    # Performance
    use_amp: bool = True  # Automatic Mixed Precision (disabled on CPU automatically)

    def __post_init__(self):
        """Generate run name and validate configuration."""
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"gwl_lstm_{timestamp}"

        self._validate_device()

    def _validate_device(self):
        """Validate and resolve the device string, and configure AMP."""
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(
                    f"WARNING: device='{self.device}' requested but CUDA is not available. Falling back to CPU."
                )
                self.device = "cpu"
            else:
                if ":" in self.device:
                    gpu_idx = int(self.device.split(":")[1])
                    if gpu_idx >= torch.cuda.device_count():
                        print(
                            f"WARNING: device='{self.device}' requested but only "
                            f"{torch.cuda.device_count()} GPU(s) available. Falling back to cuda:0."
                        )
                        self.device = "cuda:0"
                print(f"CUDA device: {torch.cuda.get_device_name(self.device)}")

        # AMP only works on CUDA
        if not self.device.startswith("cuda"):
            if self.use_amp:
                print("INFO: AMP disabled (requires CUDA device)")
                self.use_amp = False


class LSTMTensorDataset:
    """
    GPU-resident dataset for LSTM GWL forecasting.

    Pre-transfers ALL data to the target device once during construction,
    eliminating per-batch CPU→GPU transfers during training.

    Replaces the previous LSTMDataset + DataLoader(num_workers) pattern which
    created CPU tensors in worker subprocesses and transferred them every batch.

    In delta mode, carries both:
    - target_gwl: delta value (target - current) used for loss computation
    - target_gwl_raw: absolute target value used for metric computation
    """

    def __init__(
        self,
        transformed_samples: List[Dict],
        device: str = "cpu",
        use_delta_gwl: bool = True,
        use_weighted_loss: bool = True,
        loss_weight_type: str = "std",
    ):
        n = len(transformed_samples)
        self.use_delta_gwl = use_delta_gwl

        # Build contiguous arrays on CPU first (one allocation, not N small ones)
        sequence = np.array(
            [s["sequence"] for s in transformed_samples], dtype=np.float32
        )
        forecast_features = np.array(
            [s["forecast_features"] for s in transformed_samples], dtype=np.float32
        )
        static_continuous = np.array(
            [s["static_continuous"] for s in transformed_samples], dtype=np.float32
        )
        lithology = np.array(
            [s["lithology_idx"] for s in transformed_samples], dtype=np.int64
        )
        well_type = np.array(
            [s["well_type_idx"] for s in transformed_samples], dtype=np.int64
        )
        aquifer = np.array(
            [s["aquifer_idx"] for s in transformed_samples], dtype=np.int64
        )
        aq0_aq = np.array(
            [s.get("aquifer_0_aquifer_idx", 0) for s in transformed_samples],
            dtype=np.int64,
        )
        litho_sg = np.array(
            [s.get("litho_supergroup_idx", 0) for s in transformed_samples],
            dtype=np.int64,
        )
        state = np.array(
            [s.get("state_idx", 0) for s in transformed_samples],
            dtype=np.int64,
        )
        district = np.array(
            [s.get("district_idx", 0) for s in transformed_samples],
            dtype=np.int64,
        )
        hist_lulc = np.array(
            [s["historical_lulc_indices"] for s in transformed_samples], dtype=np.int64
        )
        forecast_lulc = np.array(
            [s["forecast_lulc_idx"] for s in transformed_samples], dtype=np.int64
        )
        target = np.array(
            [s["target_gwl"] for s in transformed_samples], dtype=np.float32
        )
        current_raw = np.array(
            [s["current_gwl_raw"] for s in transformed_samples], dtype=np.float32
        )
        # Prithvi tile index (Step 3 passthrough; -1 when not annotated). Inert
        # unless use_prithvi — the projector reads it only then.
        tile_idx = np.array(
            [s.get("tile_idx", -1) for s in transformed_samples], dtype=np.int64
        )

        # Extract well IDs and temporal ordering from metadata
        station_codes = [s["metadata"]["station_code"] for s in transformed_samples]
        unique_stations = sorted(set(station_codes))
        self.station_to_idx = {s: i for i, s in enumerate(unique_stations)}
        self.idx_to_station = {i: s for s, i in self.station_to_idx.items()}
        well_ids = np.array(
            [self.station_to_idx[s] for s in station_codes], dtype=np.int64
        )

        # Target dates as ordinals for chronological sorting within wells
        target_dates = [s["metadata"]["target_date"] for s in transformed_samples]
        date_ordinals = np.array(
            [
                (
                    d.toordinal()
                    if hasattr(d, "toordinal")
                    else datetime.strptime(str(d)[:10], "%Y-%m-%d").toordinal()
                )
                for d in target_dates
            ],
            dtype=np.int64,
        )

        # Single bulk transfer to device
        self.sequence = torch.from_numpy(sequence).to(device)  # (N, T, F)
        self.forecast_features = torch.from_numpy(forecast_features).to(
            device
        )  # (N, num_forecast)
        self.static_continuous = torch.from_numpy(static_continuous).to(
            device
        )  # (N, num_static)
        self.lithology_idx = torch.from_numpy(lithology).to(device)  # (N,)
        self.well_type_idx = torch.from_numpy(well_type).to(device)  # (N,)
        self.aquifer_idx = torch.from_numpy(aquifer).to(device)  # (N,)
        self.aquifer_0_aquifer_idx = torch.from_numpy(aq0_aq).to(device)  # (N,)
        self.litho_supergroup_idx = torch.from_numpy(litho_sg).to(device)  # (N,)
        self.state_idx = torch.from_numpy(state).to(device)  # (N,)
        self.district_idx = torch.from_numpy(district).to(device)  # (N,)
        self.historical_lulc_indices = torch.from_numpy(hist_lulc).to(device)  # (N, T)
        self.forecast_lulc_idx = torch.from_numpy(forecast_lulc).to(device)  # (N,)
        self.target_gwl = torch.from_numpy(target).to(device)  # (N,)
        self.current_gwl_raw = torch.from_numpy(current_raw).to(device)  # (N,)
        self.well_id = torch.from_numpy(well_ids).to(device)  # (N,)
        self.target_date_ordinal = torch.from_numpy(date_ordinals).to(device)  # (N,)
        self.tile_idx = torch.from_numpy(tile_idx).to(device)  # (N,)

        # Resolve effective loss mode (legacy use_weighted_loss=False overrides to "none")
        if not use_weighted_loss:
            effective_mode = "none"
        else:
            effective_mode = loss_weight_type
        self.loss_weight_type = effective_mode

        # Compute per-station sample weights based on mode.
        #   "none"    — weights = 1 (uniform)
        #   "std"     — 1/std_w (per-sample), normalized to mean 1
        #   "perwell" — 1/var_w (per-sample); aggregation is per-well in loss fn
        if effective_mode in ("std", "perwell"):
            station_std = {}
            for wid_val in np.unique(well_ids):
                mask = well_ids == wid_val
                std_val = target[mask].std()
                station_std[wid_val] = max(float(std_val), 0.5)  # clip floor at 0.5m
            if effective_mode == "std":
                raw_weights = np.array(
                    [1.0 / station_std[wid_val] for wid_val in well_ids], dtype=np.float32
                )
            else:  # perwell: weight by 1/var
                raw_weights = np.array(
                    [1.0 / (station_std[wid_val] ** 2) for wid_val in well_ids], dtype=np.float32
                )
            # Normalize so mean(weights) = 1.0 (keeps loss magnitude in similar range)
            raw_weights = raw_weights / raw_weights.mean()
        else:
            raw_weights = np.ones(n, dtype=np.float32)
        self.sample_weights = torch.from_numpy(raw_weights).to(device)  # (N,)

        # Delta mode: absolute target for metric computation
        # (pred_absolute = current_gwl_raw + pred_delta, compare to target_gwl_raw)
        if use_delta_gwl:
            target_raw = np.array(
                [s["target_gwl_raw"] for s in transformed_samples], dtype=np.float32
            )
            self.target_gwl_raw = torch.from_numpy(target_raw).to(device)  # (N,)
        else:
            self.target_gwl_raw = None

        self._len = n

        # Estimate memory usage
        tensors = [
            self.sequence,
            self.forecast_features,
            self.static_continuous,
            self.lithology_idx,
            self.well_type_idx,
            self.aquifer_idx,
            self.aquifer_0_aquifer_idx,
            self.litho_supergroup_idx,
            self.state_idx,
            self.district_idx,
            self.historical_lulc_indices,
            self.forecast_lulc_idx,
            self.target_gwl,
            self.current_gwl_raw,
            self.well_id,
            self.target_date_ordinal,
            self.sample_weights,
        ]
        if self.target_gwl_raw is not None:
            tensors.append(self.target_gwl_raw)
        total_bytes = sum(t.nelement() * t.element_size() for t in tensors)
        mode_label = "DELTA" if use_delta_gwl else "ABSOLUTE"
        n_wells = len(unique_stations)
        print(
            f"  LSTMTensorDataset: {n:,} samples, {n_wells:,} wells on {device} "
            f"({total_bytes / 1024**2:.1f} MB, {mode_label} mode)"
        )

    def __len__(self) -> int:
        return self._len

    def get_batch(self, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return a batch by indexing into pre-allocated device tensors. Zero-copy."""
        batch = {
            "sequence": self.sequence[indices],
            "forecast_features": self.forecast_features[indices],
            "static_continuous": self.static_continuous[indices],
            "lithology_idx": self.lithology_idx[indices],
            "well_type_idx": self.well_type_idx[indices],
            "aquifer_idx": self.aquifer_idx[indices],
            "aquifer_0_aquifer_idx": self.aquifer_0_aquifer_idx[indices],
            "litho_supergroup_idx": self.litho_supergroup_idx[indices],
            "state_idx": self.state_idx[indices],
            "district_idx": self.district_idx[indices],
            "historical_lulc_indices": self.historical_lulc_indices[indices],
            "forecast_lulc_idx": self.forecast_lulc_idx[indices],
            "target_gwl": self.target_gwl[indices],
            "current_gwl_raw": self.current_gwl_raw[indices],
            "well_id": self.well_id[indices],
            "target_date_ordinal": self.target_date_ordinal[indices],
            "sample_weights": self.sample_weights[indices],
            "tile_idx": self.tile_idx[indices],
        }
        if self.target_gwl_raw is not None:
            batch["target_gwl_raw"] = self.target_gwl_raw[indices]
        return batch


class BatchSampler:
    """
    Generates batches of indices for LSTMTensorDataset.

    Replaces DataLoader's batching — since data is already on device,
    we just need index shuffling and chunking.
    """

    def __init__(self, dataset_len: int, batch_size: int, shuffle: bool = True):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.dataset_len)
        else:
            indices = torch.arange(self.dataset_len)

        for start in range(0, self.dataset_len, self.batch_size):
            yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class Trainer:
    """
    Trainer for GWL forecasting LSTM.

    Handles training loop, validation, LR scheduling, and checkpointing.

    In delta mode:
    - Model predicts Δ(GWL), trend = sign of predicted delta
    - Loss = α·MSE(pred_delta, target_delta) + β·BCE(pred_delta, trend_label)
    - Metrics computed on reconstructed absolute values

    In absolute mode:
    - Model predicts raw GWL, trend from (pred - current) delta
    - Loss = α·MSE(pred, target) + β·BCE(pred - current, trend_label)
    - Metrics computed on denormalized absolute values
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_data: LSTMTensorDataset,
        val_data: LSTMTensorDataset,
        test_data: LSTMTensorDataset,
        scaler: LSTMFeatureScaler,
        data_metadata: Optional[Dict] = None,
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        if config.use_prithvi:
            # Tile-grouped batches so each train batch touches few unique tiles
            # (each encoded once). TRAIN ONLY — val/test stay sequential so the
            # positional test_samples[i] <-> test_data[i] mapping is preserved.
            from gwlcore.prithvi_finetune import TileGroupedBatchSampler
            self.train_sampler = TileGroupedBatchSampler(
                train_data.tile_idx.cpu().numpy(),
                n_tiles=config.prithvi_n_tiles,
                samples_per_tile=config.prithvi_samples_per_tile,
            )
        else:
            self.train_sampler = BatchSampler(
                len(train_data), config.batch_size, shuffle=config.shuffle_train
            )
        self.val_sampler = BatchSampler(len(val_data), config.batch_size, shuffle=False)
        self.test_sampler = BatchSampler(len(test_data), config.batch_size, shuffle=False)
        self.scaler = scaler
        self.data_metadata = data_metadata or {}
        self.use_delta_gwl = config.use_delta_gwl

        # Loss function
        if config.huber_delta > 0:
            regression_loss_fn = nn.HuberLoss(delta=config.huber_delta)
        else:
            regression_loss_fn = nn.MSELoss()
        self.loss_fn = MultiTaskLoss(
            alpha=config.alpha,
            beta=config.beta,
            regression_loss_fn=regression_loss_fn,
            use_uncertainty_weighting=config.use_uncertainty_weighting,
            loss_trim_high=config.loss_trim_high,
        ).to(config.device)

        # Optimizer
        if config.use_prithvi:
            # Two LR groups: Prithvi (LoRA + projector) gentle, the from-scratch
            # TFT (+ loss_fn params) fast. See prithvi_finetune.split_param_groups.
            from gwlcore.prithvi_finetune import split_param_groups
            param_groups = split_param_groups(
                self.model, config.ft_lr, config.forecaster_lr_scale
            )
            param_groups.append({
                "params": list(self.loss_fn.parameters()),
                "lr": config.ft_lr * config.forecaster_lr_scale,
            })
            self.optimizer = Adam(param_groups, weight_decay=config.weight_decay)
        else:
            self.optimizer = Adam(
                list(self.model.parameters()) + list(self.loss_fn.parameters()),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=config.scheduler_mode,
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )

        # AMP (Automatic Mixed Precision) for CUDA speedup.
        # Prithvi path uses bf16 (the 300M encoder overflows fp16) and disables
        # the GradScaler (bf16 needs no loss scaling). Non-Prithvi: fp16 + scaler.
        self.use_amp = config.use_amp
        self.amp_dtype = torch.bfloat16 if config.use_prithvi else torch.float16
        self.grad_scaler = GradScaler(
            enabled=self.use_amp and not config.use_prithvi
        )
        # One-time guard: verify Prithvi LoRA grads actually flow after backward.
        self._lora_checked = False

        # Precompute inverse-transform constants on device (absolute mode only).
        # Delta mode: target values are raw deltas (no scaling by default), no inverse needed.
        if not self.use_delta_gwl and scaler.target_scaler is not None:
            self.target_std = torch.tensor(
                scaler.target_scaler.scale_[0],
                dtype=torch.float32,
                device=config.device,
            )
            self.target_mean = torch.tensor(
                scaler.target_scaler.mean_[0], dtype=torch.float32, device=config.device
            )
        elif self.use_delta_gwl and scaler.target_scaler is not None:
            # Delta mode with target scaling enabled (rare)
            self.target_std = torch.tensor(
                scaler.target_scaler.scale_[0],
                dtype=torch.float32,
                device=config.device,
            )
            self.target_mean = torch.tensor(
                scaler.target_scaler.mean_[0], dtype=torch.float32, device=config.device
            )
        else:
            self.target_std = None
            self.target_mean = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.train_history = []
        self.val_history = []

        # Create output directory structure
        self.run_dir = os.path.join(config.output_dir, config.run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self._plot_dirs = {
            "training": os.path.join(self.run_dir, "plots", "training"),
            "gwl_val": os.path.join(self.run_dir, "plots", "gwl", "val"),
            "gwl_test": os.path.join(self.run_dir, "plots", "gwl", "test"),
            "delta_gwl_val": os.path.join(self.run_dir, "plots", "delta_gwl", "val"),
            "delta_gwl_test": os.path.join(self.run_dir, "plots", "delta_gwl", "test"),
            "geography_val": os.path.join(self.run_dir, "plots", "geography", "val"),
            "geography_test": os.path.join(self.run_dir, "plots", "geography", "test"),
        }
        self._perf_dir = os.path.join(self.run_dir, "performance")
        for d in list(self._plot_dirs.values()) + [self._perf_dir]:
            os.makedirs(d, exist_ok=True)

        # Save config and data metadata
        config_dict = asdict(config)
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        with open(os.path.join(self.run_dir, "data_metadata.json"), "w") as f:
            json.dump(self.data_metadata, f, indent=2, default=str)

        print(f"Training run: {config.run_name}")
        print(f"Output directory: {self.run_dir}")
        print(f"  Plots: {self.run_dir}/plots/{{training,gwl,delta_gwl,geography}}")
        print(f"  Performance CSVs: {self._perf_dir}")

        # Load canonical per-station train_std map (single source of truth for std).
        # Built in data prep from PRE-thinning, PRE-flat-filter train samples.
        # Stations missing from this map are Scenario B cold-start (no train data).
        station_to_train_std_path = os.path.join(
            self.config.data_dir, "station_to_train_std.pkl"
        )
        if os.path.exists(station_to_train_std_path):
            with open(station_to_train_std_path, "rb") as f:
                station_to_train_std = pickle.load(f)
            print(
                f"Loaded station_to_train_std: {len(station_to_train_std)} stations "
                f"with real train_std (rest fall back to cold-start handling)"
            )
        else:
            print(
                f"WARNING: {station_to_train_std_path} not found; "
                f"all stations will be treated as Scenario B cold-start in plots."
            )
            station_to_train_std = {}

        # Per-station raw-observation mode-gap map (mirror of train_std map).
        station_to_mode_gap_path = os.path.join(
            self.config.data_dir, "station_to_mode_gap_days.pkl"
        )
        if os.path.exists(station_to_mode_gap_path):
            with open(station_to_mode_gap_path, "rb") as f:
                station_to_mode_gap_days = pickle.load(f)
            print(
                f"Loaded station_to_mode_gap_days: {len(station_to_mode_gap_days)} stations"
            )
        else:
            station_to_mode_gap_days = {}

        # Plot generator (extracted plotting + metric-block helpers)
        self.plot_gen = PlotGenerator(
            plot_dirs=self._plot_dirs,
            run_dir=self.run_dir,
            perf_dir=self._perf_dir,
            inverse_transform_fn=self._inverse_transform_on_device,
            compute_train_seasonal_means_fn=self._compute_train_seasonal_means,
            use_amp=self.use_amp,
            device=self.config.device,
            use_delta_gwl=self.use_delta_gwl,
            model=self.model,
            min_station_target_std=config.min_station_target_std,
            include_states=config.include_states,
            min_station_mode_gap_days=config.min_station_mode_gap_days,
            station_to_train_std=station_to_train_std,
            station_to_mode_gap_days=station_to_mode_gap_days,
            train_data=self.train_data,
            val_data=self.val_data,
            use_revin=config.use_revin,
            revin_std_floor=config.revin_std_floor,
            pred_clamp_pct=self._effective_clamp_pct(),
        )


    def _plot_path(self, filename: str, split_name: str = "", is_delta: bool = False, category: str = "gwl") -> str:
        """Get the full path for saving a plot file in the organized directory structure."""
        if category == "training":
            return os.path.join(self._plot_dirs["training"], filename)
        elif category == "geography":
            key = f"geography_{split_name}"
            return os.path.join(self._plot_dirs.get(key, self.run_dir), filename)
        elif is_delta:
            key = f"delta_gwl_{split_name}"
            return os.path.join(self._plot_dirs.get(key, self.run_dir), filename)
        else:
            key = f"gwl_{split_name}"
            return os.path.join(self._plot_dirs.get(key, self.run_dir), filename)
        print(f"Device: {config.device}")
        print(f"GWL mode: {'DELTA' if self.use_delta_gwl else 'ABSOLUTE'}")
        print(f"AMP (mixed precision): {'enabled' if self.use_amp else 'disabled'}")
        print(
            f"Gradient clipping: {config.gradient_clip_norm if config.gradient_clip_norm > 0 else 'disabled'}"
        )

    def _inverse_transform_on_device(self, scaled_gwl: torch.Tensor) -> torch.Tensor:
        """Inverse-transform scaled GWL values on device (no CPU roundtrip).
        Returns raw values if no target scaler was fitted (e.g. delta mode with scale_target=False).
        """
        if self.target_std is not None:
            return scaled_gwl * self.target_std + self.target_mean
        return scaled_gwl

    def _effective_clamp_pct(self) -> float:
        """Resolve the effective prediction clamp percentage.

        Returns 0.0 (disabled) when explicitly set to 0.0. Returns the user
        override if provided; otherwise picks 0.20 for horizon <=4 months and
        0.50 for >4 months.
        """
        pct = getattr(self.config, "pred_clamp_pct", None)
        if pct is not None:
            return float(pct)
        h = getattr(self.config, "forecast_horizon_months", 6)
        return 0.20 if int(h) <= 4 else 0.50

    def _clamp_pred_to_current(
        self,
        pred_abs: torch.Tensor,
        pred_delta: torch.Tensor,
        current_gwl_raw: torch.Tensor,
    ):
        """Cap |pred - current_gwl_raw| to pct * |current_gwl_raw|.

        Returns (pred_abs_clamped, pred_delta_clamped). When the effective pct
        is 0.0, returns the inputs unchanged.
        """
        pct = self._effective_clamp_pct()
        if pct <= 0.0:
            return pred_abs, pred_delta
        max_dev = pct * torch.abs(current_gwl_raw)
        pred_delta_c = torch.clamp(pred_delta, min=-max_dev, max=max_dev)
        pred_abs_c = current_gwl_raw + pred_delta_c
        return pred_abs_c, pred_delta_c

    def _compute_train_seasonal_means(self) -> Dict:
        """Compute per-station per-target-month mean target_delta from TRAIN data.

        Used by the seasonal baseline in `_plot_vs_baselines` so the baseline
        does not leak future information from val/test.

        Returns:
            dict with two entries:
              "by_month": {(well_id, month): mean_delta}
              "by_well":  {well_id: overall_mean_delta}  (fallback when month missing)
        """
        from datetime import date as _date
        if not hasattr(self, "_cached_train_seasonal_means"):
            data = self.train_data
            target_scaled = data.target_gwl.cpu().numpy()
            well_ids = data.well_id.cpu().numpy()
            date_ords = data.target_date_ordinal.cpu().numpy()
            # Inverse-scale target delta to real meters
            if self.target_std is not None:
                t_std = float(self.target_std.item()) if hasattr(self.target_std, "item") else float(self.target_std)
                t_mean = float(self.target_mean.item()) if hasattr(self.target_mean, "item") else float(self.target_mean)
                target_delta = target_scaled * t_std + t_mean
            else:
                target_delta = target_scaled

            months = np.array([_date.fromordinal(int(d)).month for d in date_ords])

            by_month = {}
            by_well = {}
            for wid in np.unique(well_ids):
                m = well_ids == wid
                by_well[int(wid)] = float(target_delta[m].mean())
                wm = months[m]
                wd = target_delta[m]
                for mo in range(1, 13):
                    mm = wm == mo
                    if mm.any():
                        by_month[(int(wid), int(mo))] = float(wd[mm].mean())

            self._cached_train_seasonal_means = {"by_month": by_month, "by_well": by_well}
        return self._cached_train_seasonal_means

    @staticmethod
    def _rolling_mean(seq: List[float], window: int = 5) -> List[float]:
        """Trailing rolling mean — for idx i, mean of seq[max(0,i-window+1):i+1].

        Used to smooth val-loss trajectory for trend visibility. Edge-padded
        on the left (early epochs use a smaller window). Returns a list same
        length as input.
        """
        out = []
        for i in range(len(seq)):
            lo = max(0, i - window + 1)
            chunk = seq[lo : i + 1]
            out.append(sum(chunk) / len(chunk))
        return out

    def _apply_revin(
        self, sequence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        RevIN: per-sample reversible instance normalization on the GWL channel.

        Computes (mean_s, std_s) from the GWL channel using is_present=1
        timesteps only. Returns a new sequence with the GWL channel normalized
        per-sample; gap timesteps (is_present=0) stay at 0 to preserve the
        dual encoding semantics.

        Args:
            sequence: (B, T, F) input. Channel 0 = gwl, channel 1 = is_present.

        Returns:
            sequence_normed: (B, T, F) with GWL channel replaced.
            mean_s: (B,) per-sample mean from present timesteps.
            std_s: (B,) per-sample std from present timesteps.
        """
        gwl = sequence[..., 0:1]                              # (B, T, 1)
        is_present = sequence[..., 1:2]                       # (B, T, 1)
        n_present = is_present.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_s_3d = (gwl * is_present).sum(dim=1, keepdim=True) / n_present
        sq_dev = ((gwl - mean_s_3d) * is_present).pow(2)
        var_s_3d = sq_dev.sum(dim=1, keepdim=True) / n_present
        std_s_3d = (var_s_3d + 1e-5).sqrt().clamp(min=self.config.revin_std_floor)
        gwl_normed = ((gwl - mean_s_3d) / std_s_3d) * is_present
        sequence_normed = torch.cat([gwl_normed, sequence[..., 1:]], dim=-1)
        return sequence_normed, mean_s_3d.squeeze(-1).squeeze(-1), std_s_3d.squeeze(-1).squeeze(-1)

    def _compute_trend_logit_and_target(
        self, gwl_pred: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute trend logit and trend target for BCE loss. All on device.

        Delta mode:
          - trend_logit = pred (it IS the delta, sign = trend direction)
          - trend_target from sign of target_gwl (also a delta)

        Absolute mode:
          - trend_logit = pred_denorm - current_gwl_raw (compute delta for BCE)
          - trend_target from sign of (target_denorm - current_gwl_raw)
        """
        current_gwl_raw = batch["current_gwl_raw"]
        target_gwl = batch[
            "target_gwl"
        ]  # delta (delta mode) or scaled absolute (absolute mode)

        if self.use_delta_gwl:
            # Keep trend_logit in scaled space so BCE saturation matches the
            # regression-loss scale and α/β balance is unit-correct. Sign is
            # preserved by the linear scaler, so trend classification is
            # unchanged. target_delta is unscaled only because classify_trend
            # compares it against current_gwl_raw (real meters).
            target_delta = self._inverse_transform_on_device(target_gwl)

            trend_logit = gwl_pred.squeeze(1)
            trend_target = classify_trend(
                current_gwl_raw, target_delta, use_delta_gwl=True
            )
        else:
            # Absolute mode: denormalize, then compute deltas
            pred_denorm = self._inverse_transform_on_device(gwl_pred.squeeze(1))
            target_denorm = self._inverse_transform_on_device(target_gwl)

            trend_logit = pred_denorm - current_gwl_raw
            trend_target = classify_trend(
                current_gwl_raw, target_denorm, use_delta_gwl=False
            )

        return trend_logit, trend_target

    def _model_forward(self, sequence_in, batch):
        """Single call site for self.model(...). Passes tile_idx only when
        use_prithvi (the other models don't accept that kwarg)."""
        kw = dict(
            sequence=sequence_in,
            static_continuous=batch["static_continuous"],
            forecast_features=batch["forecast_features"],
            lithology_idx=batch["lithology_idx"],
            well_type_idx=batch["well_type_idx"],
            aquifer_idx=batch["aquifer_idx"],
            aquifer_0_aquifer_idx=batch["aquifer_0_aquifer_idx"],
            litho_supergroup_idx=batch["litho_supergroup_idx"],
            state_idx=batch["state_idx"],
            district_idx=batch["district_idx"],
            historical_lulc_indices=batch["historical_lulc_indices"],
            forecast_lulc_idx=batch["forecast_lulc_idx"],
        )
        if self.config.use_prithvi:
            kw["tile_idx"] = batch["tile_idx"]
        return self.model(**kw)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch. All data already on device — zero CPU↔GPU transfers."""
        self.model.train()

        epoch_loss = 0.0
        epoch_regression_loss = 0.0
        epoch_trend_loss = 0.0
        num_batches = 0

        amp_device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"

        for batch_indices in self.train_sampler:
            batch = self.train_data.get_batch(batch_indices)

            target_gwl = batch["target_gwl"].unsqueeze(1)  # (B, 1) for MSE

            # Forward pass with AMP
            with autocast(device_type=amp_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                # RevIN: per-sample normalize GWL channel before forward pass.
                if self.config.use_revin:
                    sequence_in, mean_s, std_s = self._apply_revin(batch["sequence"])
                else:
                    sequence_in = batch["sequence"]
                    mean_s = std_s = None

                gwl_pred = self._model_forward(sequence_in, batch)

                # If RevIN is active: model output is in normed space. For
                # trend classification we need the prediction in global-scaled
                # space (sign-stable for the existing classify_trend logic).
                if self.config.use_revin:
                    pred_global = (gwl_pred.squeeze(1) * std_s + mean_s).unsqueeze(1)
                    target_for_loss = (
                        (target_gwl.squeeze(1) - mean_s) / std_s
                    ).unsqueeze(1)
                    trend_logit, trend_target = self._compute_trend_logit_and_target(
                        pred_global, batch
                    )
                else:
                    target_for_loss = target_gwl
                    trend_logit, trend_target = self._compute_trend_logit_and_target(
                        gwl_pred, batch
                    )

                total_loss, loss_dict = self.loss_fn(
                    gwl_pred, target_for_loss, trend_logit, trend_target,
                    sample_weights=batch["sample_weights"],
                    well_ids=(
                        batch["well_id"]
                        if self.config.loss_weight_type == "perwell"
                        and self.config.use_weighted_loss
                        else None
                    ),
                )

            # Backward pass with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)
            self.grad_scaler.scale(total_loss).backward()

            # Gradient clipping (before optimizer step)
            if self.config.gradient_clip_norm > 0:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

            # Silent-failure guard: confirm Prithvi LoRA grads actually flow.
            # Runs once (first batch after grads exist). If zero, nothing is
            # fine-tuning Prithvi (detached projector / all-zero_idx tiles).
            if self.config.use_prithvi and not self._lora_checked:
                lora_params = [
                    p for n, p in self.model.named_parameters()
                    if n.startswith("prithvi.") and "lora_" in n and p.requires_grad
                ]
                g = sum(
                    float(p.grad.abs().sum())
                    for p in lora_params if p.grad is not None
                )
                assert g > 0, (
                    "Prithvi LoRA grads are zero after backward — encoder is NOT "
                    "being fine-tuned (check projector wiring / tile_idx coverage)."
                )
                print(f"[ft] LoRA grad check OK — Σ|grad| over "
                      f"{len(lora_params)} LoRA tensors = {g:.3e}")
                self._lora_checked = True

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            epoch_loss += loss_dict["total_loss"]
            epoch_regression_loss += loss_dict["regression_loss"]
            epoch_trend_loss += loss_dict["trend_loss"]
            num_batches += 1

        return {
            "total_loss": epoch_loss / num_batches,
            "regression_loss": epoch_regression_loss / num_batches,
            "trend_loss": epoch_trend_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self, data=None, sampler=None, split_name="val", compute_per_well: bool = True) -> Dict[str, float]:
        """
        Validate on a dataset split. Everything stays on GPU until final scalar metrics.

        Metrics are always computed on absolute GWL scale:
        - Delta mode: pred_abs = current_gwl_raw + pred_delta, target_abs = target_gwl_raw
        - Absolute mode: denormalize pred and target using target_scaler

        Args:
            data: LSTMTensorDataset to evaluate on. Defaults to self.val_data.
            sampler: BatchSampler for the dataset. Defaults to self.val_sampler.
            split_name: Name of the split (e.g. "val", "test") for reference.
            compute_per_well: Whether to compute per-well R² and KGE.
        """
        if data is None or sampler is None:
            data = self.val_data
            sampler = self.val_sampler
            split_name = "val"

        self.model.eval()

        epoch_loss = 0.0
        epoch_regression_loss = 0.0
        epoch_trend_loss = 0.0
        num_batches = 0

        all_preds_abs = []  # Reconstructed absolute predictions
        all_targets_abs = []  # Absolute targets
        all_target_deltas = []  # Target deltas (for delta-std predictable filter)
        all_trend_preds = []
        all_trend_targets = []
        all_well_ids = []  # For per-well metrics

        amp_device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"

        for batch_indices in sampler:
            batch = data.get_batch(batch_indices)

            target_gwl = batch["target_gwl"].unsqueeze(1)  # (B, 1) for MSE
            current_gwl_raw = batch["current_gwl_raw"]

            with autocast(device_type=amp_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                if self.config.use_revin:
                    sequence_in, mean_s, std_s = self._apply_revin(batch["sequence"])
                else:
                    sequence_in = batch["sequence"]
                    mean_s = std_s = None

                gwl_pred = self._model_forward(sequence_in, batch)

                # If RevIN: de-normalize prediction to global-scaled space
                # before loss + metric computation.
                if self.config.use_revin:
                    pred_global = (gwl_pred.squeeze(1) * std_s + mean_s).unsqueeze(1)
                    target_for_loss = (
                        (target_gwl.squeeze(1) - mean_s) / std_s
                    ).unsqueeze(1)
                    trend_logit, trend_target = self._compute_trend_logit_and_target(
                        pred_global, batch
                    )
                else:
                    pred_global = gwl_pred
                    target_for_loss = target_gwl
                    trend_logit, trend_target = self._compute_trend_logit_and_target(
                        gwl_pred, batch
                    )

                total_loss, loss_dict = self.loss_fn(
                    gwl_pred, target_for_loss, trend_logit, trend_target,
                    sample_weights=batch["sample_weights"],
                    well_ids=(
                        batch["well_id"]
                        if self.config.loss_weight_type == "perwell"
                        and self.config.use_weighted_loss
                        else None
                    ),
                )

            epoch_loss += loss_dict["total_loss"]
            epoch_regression_loss += loss_dict["regression_loss"]
            epoch_trend_loss += loss_dict["trend_loss"]
            num_batches += 1

            # Reconstruct absolute values for metrics (all on GPU)
            # Use pred_global (de-normalized) for inverse_transform.
            if self.use_delta_gwl:
                # Unscale predicted delta if target scaler was used
                pred_delta = self._inverse_transform_on_device(
                    pred_global.squeeze(1).float()
                )
                pred_abs = current_gwl_raw + pred_delta
                target_abs = batch["target_gwl_raw"]  # Already absolute
            else:
                pred_abs = self._inverse_transform_on_device(
                    pred_global.squeeze(1).float()
                )
                target_abs = self._inverse_transform_on_device(
                    batch["target_gwl"].float()
                )
                pred_delta = pred_abs - current_gwl_raw

            pred_abs, pred_delta = self._clamp_pred_to_current(
                pred_abs, pred_delta, current_gwl_raw
            )

            all_preds_abs.append(pred_abs)
            all_targets_abs.append(target_abs)
            # Target delta for the delta-std-based predictable filter
            all_target_deltas.append(target_abs - current_gwl_raw)

            # Trend predictions from sign of trend_logit
            trend_pred = (torch.sigmoid(trend_logit.float()) >= 0.5).long()
            all_trend_preds.append(trend_pred)
            all_trend_targets.append(trend_target)
            all_well_ids.append(batch["well_id"])

        # Concat on GPU
        all_preds_abs = torch.cat(all_preds_abs)  # (N,) absolute
        all_targets_abs = torch.cat(all_targets_abs)  # (N,) absolute
        all_target_deltas = torch.cat(all_target_deltas)  # (N,) delta

        # Derived per-sample tensors for additional per-well R² flavours:
        #   current_gwl = target_abs - target_delta
        #   pred_delta  = pred_abs - current_gwl
        #   pred_blend  = (pred_abs + current_gwl) / 2  (50/50 model+persistence)
        all_current_gwl = all_targets_abs - all_target_deltas
        all_pred_deltas = all_preds_abs - all_current_gwl
        all_preds_blend = (all_preds_abs + all_current_gwl) / 2.0
        all_trend_preds = torch.cat(all_trend_preds)  # (N,)
        all_trend_targets = torch.cat(all_trend_targets)  # (N,)
        all_well_ids = torch.cat(all_well_ids)  # (N,)

        # Compute metrics on GPU, extract as Python floats (scalar GPU→CPU, negligible)
        diff = all_preds_abs - all_targets_abs
        rmse = torch.sqrt((diff**2).mean()).item()
        mae = diff.abs().mean().item()

        # R² (equivalent to Nash-Sutcliffe Efficiency in hydrology)
        # R² = 1 - mean((obs - sim)²) / mean((obs - mean(obs))²)
        obs_mean = all_targets_abs.mean()
        mse_model = (diff**2).mean()
        mse_baseline = ((all_targets_abs - obs_mean) ** 2).mean()
        r2 = (1.0 - mse_model / (mse_baseline + 1e-8)).item()

        # KGE (Kling-Gupta Efficiency)
        # KGE = 1 - sqrt((r-1)² + (β-1)² + (γ-1)²)
        # r = correlation, β = mean_pred/mean_obs, γ = std_pred/std_obs
        pred_mean = all_preds_abs.mean()
        pred_std = all_preds_abs.std()
        obs_std = all_targets_abs.std()

        # Pearson correlation on GPU
        pred_centered = all_preds_abs - pred_mean
        obs_centered = all_targets_abs - obs_mean
        r = (pred_centered * obs_centered).mean() / (pred_std * obs_std + 1e-8)

        beta = pred_mean / (obs_mean + 1e-8)  # Bias ratio
        gamma = pred_std / (obs_std + 1e-8)  # Variability ratio

        kge = (
            1.0 - torch.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
        ).item()

        trend_accuracy = (all_trend_preds == all_trend_targets).float().mean().item()

        # Filtered trend accuracy: only samples where the target actually moved
        # (|Δ| >= threshold meters). Removes flat-well noise from the metric.
        min_delta = float(self.config.min_delta_for_trend_accuracy)
        if min_delta > 0:
            trend_mask = all_target_deltas.abs() >= min_delta
        else:
            trend_mask = torch.ones_like(all_target_deltas, dtype=torch.bool)
        n_trend_filtered = int(trend_mask.sum().item())
        if n_trend_filtered > 0:
            trend_accuracy_filtered = (
                (all_trend_preds[trend_mask] == all_trend_targets[trend_mask])
                .float().mean().item()
            )
        else:
            trend_accuracy_filtered = 0.0

        # Per-well R² — strips out between-well variance, measures within-well dynamics
        # Per-well KGE — same idea for KGE
        # Vectorised via scatter_add (no Python loop over wells)
        if compute_per_well:
            unique_wells, inverse_idx, counts = torch.unique(
                all_well_ids, return_inverse=True, return_counts=True
            )
            n_wells = len(unique_wells)
            device = all_preds_abs.device

            # Filter wells with ≥5 samples upfront. Aligned with the
            # downstream diagnostic block (gwl_plots._print_metrics_block uses
            # the same threshold). Wells with 3-4 samples produce too noisy a
            # per-well R² to include in the headline median.
            valid_count = counts >= 5

            # Per-well sums → means (scatter_add is one GPU kernel, not N)
            obs_sum = torch.zeros(n_wells, device=device).scatter_add_(
                0, inverse_idx, all_targets_abs
            )
            pred_sum = torch.zeros(n_wells, device=device).scatter_add_(
                0, inverse_idx, all_preds_abs
            )
            w_obs_mean = obs_sum / counts.float()
            w_pred_mean = pred_sum / counts.float()

            # Expand per-well means back to per-sample for residual computation
            obs_mean_exp = w_obs_mean[inverse_idx]
            pred_mean_exp = w_pred_mean[inverse_idx]

            # Per-well R²: 1 - MSE(model) / MSE(baseline)
            sq_error = (all_preds_abs - all_targets_abs) ** 2
            sq_baseline = (all_targets_abs - obs_mean_exp) ** 2
            mse_model_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, sq_error
                )
                / counts.float()
            )
            mse_base_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, sq_baseline
                )
                / counts.float()
            )

            valid_r2 = valid_count & (mse_base_pw > 1e-8)
            r2_all = 1.0 - mse_model_pw / (mse_base_pw + 1e-8)

            # Per-well R² (delta): SS_tot based on per-well mean of TARGET DELTA.
            target_d_sum_pw = torch.zeros(n_wells, device=device).scatter_add_(
                0, inverse_idx, all_target_deltas
            )
            target_d_mean_pw = target_d_sum_pw / counts.float()
            target_d_mean_exp = target_d_mean_pw[inverse_idx]
            sq_error_d = (all_pred_deltas - all_target_deltas) ** 2
            sq_base_d = (all_target_deltas - target_d_mean_exp) ** 2
            mse_model_d_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, sq_error_d
                )
                / counts.float()
            )
            mse_base_d_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, sq_base_d
                )
                / counts.float()
            )
            r2_delta_all = 1.0 - mse_model_d_pw / (mse_base_d_pw + 1e-8)
            valid_r2_delta = valid_count & (mse_base_d_pw > 1e-8)

            # Per-well R² (blend): pred_blend vs target_abs, SAME SS_tot as r2_all.
            sq_error_b = (all_preds_blend - all_targets_abs) ** 2
            mse_model_b_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, sq_error_b
                )
                / counts.float()
            )
            r2_blend_all = 1.0 - mse_model_b_pw / (mse_base_pw + 1e-8)

            # Per-well R² (persistence): pred = current_gwl (no change), SAME SS_tot.
            # Residual = current_gwl - target_abs = -target_delta → sq_error = target_delta²
            sq_error_p = all_target_deltas ** 2
            mse_model_p_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, sq_error_p
                )
                / counts.float()
            )
            r2_persist_all = 1.0 - mse_model_p_pw / (mse_base_pw + 1e-8)

            # ── MAPE-style metrics ──────────────────────────────────────────
            # For 3 predictions (model / blend / persist), compute per-sample
            # MAPE = |pred - target| / |target|  and  MAPE-floor = / max(|t|,1)
            # then mean over samples → per-well, then mean/median across wells.
            abs_t = all_targets_abs.abs()
            abs_t_floor = abs_t.clamp(min=1.0)
            abs_err_pred = (all_preds_abs - all_targets_abs).abs()
            abs_err_blend = (all_preds_blend - all_targets_abs).abs()
            abs_err_persist = all_target_deltas.abs()  # = |current - target|

            def _per_well_mean(per_sample):
                s = torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, per_sample
                )
                return s / counts.float()

            mape_pred_pw    = _per_well_mean(abs_err_pred    / (abs_t + 1e-8))
            mape_blend_pw   = _per_well_mean(abs_err_blend   / (abs_t + 1e-8))
            mape_persist_pw = _per_well_mean(abs_err_persist / (abs_t + 1e-8))
            mapef_pred_pw    = _per_well_mean(abs_err_pred    / abs_t_floor)
            mapef_blend_pw   = _per_well_mean(abs_err_blend   / abs_t_floor)
            mapef_persist_pw = _per_well_mean(abs_err_persist / abs_t_floor)

            # Per-well KGE: 1 - sqrt((r-1)² + (β-1)² + (γ-1)²)
            obs_c = all_targets_abs - obs_mean_exp
            pred_c = all_preds_abs - pred_mean_exp
            obs_var_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, obs_c**2
                )
                / counts.float()
            )
            pred_var_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, pred_c**2
                )
                / counts.float()
            )
            cross_pw = (
                torch.zeros(n_wells, device=device).scatter_add_(
                    0, inverse_idx, obs_c * pred_c
                )
                / counts.float()
            )

            w_obs_std = torch.sqrt(obs_var_pw + 1e-8)
            w_pred_std = torch.sqrt(pred_var_pw + 1e-8)
            w_r = cross_pw / (w_obs_std * w_pred_std + 1e-8)
            w_beta = w_pred_mean / (w_obs_mean + 1e-8)
            w_gamma = w_pred_std / (w_obs_std + 1e-8)
            kge_all = 1.0 - torch.sqrt(
                (w_r - 1) ** 2 + (w_beta - 1) ** 2 + (w_gamma - 1) ** 2
            )

            valid_kge = valid_count & (obs_var_pw > 1e-8) & (pred_var_pw > 1e-8)

            # Mean is clipped to [-2, 2] to prevent catastrophic outliers
            # (e.g., flat wells with R² = -1e6) from swamping the average.
            def _quartiles(t, mask, clip=(-2.0, 2.0)):
                """Return (q25, median, q75, clipped_mean) across wells in mask."""
                if not mask.any():
                    return 0.0, 0.0, 0.0, 0.0
                vals = t[mask]
                qs = torch.quantile(
                    vals, torch.tensor([0.25, 0.5, 0.75], device=vals.device)
                )
                return (
                    qs[0].item(),
                    qs[1].item(),
                    qs[2].item(),
                    vals.clamp(*clip).mean().item(),
                )

            r2_clip = (-2.0, 2.0)
            mape_clip = (0.0, 5.0)
            (per_well_r2_q25, per_well_r2,
             per_well_r2_q75, per_well_r2_mean) = _quartiles(r2_all, valid_r2, r2_clip)
            (per_well_r2_delta_q25, per_well_r2_delta,
             per_well_r2_delta_q75, per_well_r2_delta_mean) = _quartiles(
                r2_delta_all, valid_r2_delta, r2_clip)
            (per_well_r2_blend_q25, per_well_r2_blend,
             per_well_r2_blend_q75, per_well_r2_blend_mean) = _quartiles(
                r2_blend_all, valid_r2, r2_clip)
            (per_well_r2_persist_q25, per_well_r2_persist,
             per_well_r2_persist_q75, per_well_r2_persist_mean) = _quartiles(
                r2_persist_all, valid_r2, r2_clip)

            (mape_pred_q25, mape_pred_med, mape_pred_q75, mape_pred_mean) = \
                _quartiles(mape_pred_pw, valid_count, mape_clip)
            (mape_blend_q25, mape_blend_med, mape_blend_q75, mape_blend_mean) = \
                _quartiles(mape_blend_pw, valid_count, mape_clip)
            (mape_persist_q25, mape_persist_med, mape_persist_q75, mape_persist_mean) = \
                _quartiles(mape_persist_pw, valid_count, mape_clip)
            (mapef_pred_q25, mapef_pred_med, mapef_pred_q75, mapef_pred_mean) = \
                _quartiles(mapef_pred_pw, valid_count, mape_clip)
            (mapef_blend_q25, mapef_blend_med, mapef_blend_q75, mapef_blend_mean) = \
                _quartiles(mapef_blend_pw, valid_count, mape_clip)
            (mapef_persist_q25, mapef_persist_med, mapef_persist_q75, mapef_persist_mean) = \
                _quartiles(mapef_persist_pw, valid_count, mape_clip)
            per_well_kge = (
                torch.median(kge_all[valid_kge]).item() if valid_kge.any() else 0.0
            )
            per_well_kge_mean = (
                kge_all[valid_kge].clamp(-2.0, 2.0).mean().item()
                if valid_kge.any() else 0.0
            )
            n_wells_evaluated = valid_r2.sum().item()

            # Filtered per-well R²: predictable wells only (std of TARGET DELTA ≥ 1.0m, ≤ 30 samples)
            # Uses delta std (matches data-prep flat-well filter and test-eval filter).
            # Compute per-well std of target delta via scatter_add.
            target_d_sum = torch.zeros(n_wells, device=device).scatter_add_(
                0, inverse_idx, all_target_deltas
            )
            target_d_mean = target_d_sum / counts.float()
            target_d_centered = all_target_deltas - target_d_mean[inverse_idx]
            target_d_var = torch.zeros(n_wells, device=device).scatter_add_(
                0, inverse_idx, target_d_centered ** 2
            ) / counts.float()
            target_d_std = torch.sqrt(target_d_var + 1e-8)

            predictable = valid_r2 & (target_d_std >= 1.0) & (counts <= 30)
            per_well_r2_filtered = (
                torch.median(r2_all[predictable]).item() if predictable.any() else 0.0
            )
            per_well_r2_filtered_mean = (
                r2_all[predictable].clamp(-2.0, 2.0).mean().item()
                if predictable.any() else 0.0
            )
            n_wells_filtered = predictable.sum().item()
        else:
            # Carry forward last computed values (avoid misleading dips in plots)
            if self.val_history:
                last = self.val_history[-1]
                per_well_r2 = last.get("r2_per_well", 0.0)
                per_well_r2_mean = last.get("r2_per_well_mean", 0.0)
                per_well_r2_delta = last.get("r2_delta_per_well", 0.0)
                per_well_r2_delta_mean = last.get("r2_delta_per_well_mean", 0.0)
                per_well_r2_blend = last.get("r2_blend_per_well", 0.0)
                per_well_r2_blend_mean = last.get("r2_blend_per_well_mean", 0.0)
                per_well_r2_persist = last.get("r2_persist_per_well", 0.0)
                per_well_r2_persist_mean = last.get("r2_persist_per_well_mean", 0.0)
                per_well_r2_q25 = last.get("r2_q25", 0.0)
                per_well_r2_q75 = last.get("r2_q75", 0.0)
                per_well_r2_delta_q25 = last.get("r2_delta_q25", 0.0)
                per_well_r2_delta_q75 = last.get("r2_delta_q75", 0.0)
                per_well_r2_blend_q25 = last.get("r2_blend_q25", 0.0)
                per_well_r2_blend_q75 = last.get("r2_blend_q75", 0.0)
                per_well_r2_persist_q25 = last.get("r2_persist_q25", 0.0)
                per_well_r2_persist_q75 = last.get("r2_persist_q75", 0.0)
                mape_pred_q25 = last.get("mape_pred_q25", 0.0)
                mape_pred_q75 = last.get("mape_pred_q75", 0.0)
                mape_blend_q25 = last.get("mape_blend_q25", 0.0)
                mape_blend_q75 = last.get("mape_blend_q75", 0.0)
                mape_persist_q25 = last.get("mape_persist_q25", 0.0)
                mape_persist_q75 = last.get("mape_persist_q75", 0.0)
                mapef_pred_q25 = last.get("mapef_pred_q25", 0.0)
                mapef_pred_q75 = last.get("mapef_pred_q75", 0.0)
                mapef_blend_q25 = last.get("mapef_blend_q25", 0.0)
                mapef_blend_q75 = last.get("mapef_blend_q75", 0.0)
                mapef_persist_q25 = last.get("mapef_persist_q25", 0.0)
                mapef_persist_q75 = last.get("mapef_persist_q75", 0.0)
                mape_pred_med    = last.get("mape_pred_med",    0.0)
                mape_pred_mean   = last.get("mape_pred_mean",   0.0)
                mape_blend_med   = last.get("mape_blend_med",   0.0)
                mape_blend_mean  = last.get("mape_blend_mean",  0.0)
                mape_persist_med = last.get("mape_persist_med", 0.0)
                mape_persist_mean= last.get("mape_persist_mean",0.0)
                mapef_pred_med    = last.get("mapef_pred_med",    0.0)
                mapef_pred_mean   = last.get("mapef_pred_mean",   0.0)
                mapef_blend_med   = last.get("mapef_blend_med",   0.0)
                mapef_blend_mean  = last.get("mapef_blend_mean",  0.0)
                mapef_persist_med = last.get("mapef_persist_med", 0.0)
                mapef_persist_mean= last.get("mapef_persist_mean",0.0)
                per_well_kge = last.get("kge_per_well", 0.0)
                per_well_kge_mean = last.get("kge_per_well_mean", 0.0)
                n_wells_evaluated = last.get("n_wells_evaluated", 0)
                per_well_r2_filtered = last.get("r2_per_well_filtered", 0.0)
                per_well_r2_filtered_mean = last.get("r2_per_well_filtered_mean", 0.0)
                n_wells_filtered = last.get("n_wells_filtered", 0)
            else:
                per_well_r2 = 0.0
                per_well_r2_mean = 0.0
                per_well_r2_delta = 0.0
                per_well_r2_delta_mean = 0.0
                per_well_r2_blend = 0.0
                per_well_r2_blend_mean = 0.0
                per_well_r2_persist = 0.0
                per_well_r2_persist_mean = 0.0
                per_well_r2_q25 = per_well_r2_q75 = 0.0
                per_well_r2_delta_q25 = per_well_r2_delta_q75 = 0.0
                per_well_r2_blend_q25 = per_well_r2_blend_q75 = 0.0
                per_well_r2_persist_q25 = per_well_r2_persist_q75 = 0.0
                mape_pred_q25 = mape_pred_q75 = 0.0
                mape_blend_q25 = mape_blend_q75 = 0.0
                mape_persist_q25 = mape_persist_q75 = 0.0
                mapef_pred_q25 = mapef_pred_q75 = 0.0
                mapef_blend_q25 = mapef_blend_q75 = 0.0
                mapef_persist_q25 = mapef_persist_q75 = 0.0
                mape_pred_med = mape_pred_mean = 0.0
                mape_blend_med = mape_blend_mean = 0.0
                mape_persist_med = mape_persist_mean = 0.0
                mapef_pred_med = mapef_pred_mean = 0.0
                mapef_blend_med = mapef_blend_mean = 0.0
                mapef_persist_med = mapef_persist_mean = 0.0
                per_well_kge = 0.0
                per_well_kge_mean = 0.0
                n_wells_evaluated = 0
                per_well_r2_filtered = 0.0
                per_well_r2_filtered_mean = 0.0
                n_wells_filtered = 0

        return {
            "total_loss": epoch_loss / num_batches,
            "regression_loss": epoch_regression_loss / num_batches,
            "trend_loss": epoch_trend_loss / num_batches,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "r2_per_well": per_well_r2,
            "r2_per_well_mean": per_well_r2_mean,
            "r2_delta_per_well": per_well_r2_delta,
            "r2_delta_per_well_mean": per_well_r2_delta_mean,
            "r2_blend_per_well": per_well_r2_blend,
            "r2_blend_per_well_mean": per_well_r2_blend_mean,
            "r2_persist_per_well": per_well_r2_persist,
            "r2_persist_per_well_mean": per_well_r2_persist_mean,
            "r2_q25": per_well_r2_q25, "r2_q75": per_well_r2_q75,
            "r2_delta_q25": per_well_r2_delta_q25, "r2_delta_q75": per_well_r2_delta_q75,
            "r2_blend_q25": per_well_r2_blend_q25, "r2_blend_q75": per_well_r2_blend_q75,
            "r2_persist_q25": per_well_r2_persist_q25, "r2_persist_q75": per_well_r2_persist_q75,
            "mape_pred_q25": mape_pred_q25, "mape_pred_q75": mape_pred_q75,
            "mape_blend_q25": mape_blend_q25, "mape_blend_q75": mape_blend_q75,
            "mape_persist_q25": mape_persist_q25, "mape_persist_q75": mape_persist_q75,
            "mapef_pred_q25": mapef_pred_q25, "mapef_pred_q75": mapef_pred_q75,
            "mapef_blend_q25": mapef_blend_q25, "mapef_blend_q75": mapef_blend_q75,
            "mapef_persist_q25": mapef_persist_q25, "mapef_persist_q75": mapef_persist_q75,
            "mape_pred_med":    mape_pred_med,    "mape_pred_mean":    mape_pred_mean,
            "mape_blend_med":   mape_blend_med,   "mape_blend_mean":   mape_blend_mean,
            "mape_persist_med": mape_persist_med, "mape_persist_mean": mape_persist_mean,
            "mapef_pred_med":    mapef_pred_med,    "mapef_pred_mean":    mapef_pred_mean,
            "mapef_blend_med":   mapef_blend_med,   "mapef_blend_mean":   mapef_blend_mean,
            "mapef_persist_med": mapef_persist_med, "mapef_persist_mean": mapef_persist_mean,
            "r2_per_well_filtered": per_well_r2_filtered,
            "r2_per_well_filtered_mean": per_well_r2_filtered_mean,
            "kge": kge,
            "kge_per_well": per_well_kge,
            "kge_per_well_mean": per_well_kge_mean,
            "kge_r": r.item(),
            "kge_beta": beta.item(),
            "kge_gamma": gamma.item(),
            "trend_accuracy": trend_accuracy,
            "trend_accuracy_filtered": trend_accuracy_filtered,
            "n_trend_filtered": n_trend_filtered,
            "n_wells_evaluated": n_wells_evaluated,
            "n_wells_filtered": n_wells_filtered,
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint. Only called when val loss improves."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": asdict(self.config),
            "embedding_metadata": self.scaler.embedding_metadata,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

        if is_best:
            best_checkpoint_path = os.path.join(self.run_dir, "best_model.pt")
            torch.save(checkpoint, best_checkpoint_path)

    def plot_validation_loss(self):
        """Plot validation loss vs epochs (rolling-5 mean)."""
        epochs = range(1, len(self.val_history) + 1)
        val_losses = [v["total_loss"] for v in self.val_history]
        val_losses_smooth = self._rolling_mean(val_losses, window=5)

        plt.figure()
        plt.plot(epochs, val_losses_smooth, color="tab:blue", marker="o",
                 linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Loss (rolling-5)")
        plt.title("Validation Loss vs Epochs")
        plt.grid(True)

        plot_path = self._plot_path("val_loss_vs_epochs.png", category="training")
        plt.savefig(plot_path)
        plt.close()

        print(f"Validation vs Epoch plot saved to: {plot_path}")

    def format_size(self, n):
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)

    def plot_training_curves(self):
        """Create a single PNG with training diagnostics (2x2 grid)."""
        epochs = range(1, len(self.train_history) + 1)

        size_label = self.format_size(self.data_metadata.get("dataset_size", 0))
        train_samples = len(self.train_data)
        val_samples = len(self.val_data)

        train_total = [h["total_loss"] for h in self.train_history]
        train_reg = [h["regression_loss"] for h in self.train_history]
        train_trend = [h["trend_loss"] for h in self.train_history]

        val_total = [h["total_loss"] for h in self.val_history]
        val_reg = [h["regression_loss"] for h in self.val_history]
        val_trend = [h["trend_loss"] for h in self.val_history]
        val_r2 = [h["r2"] for h in self.val_history]
        val_r2_pw = [h.get("r2_per_well", h["r2"]) for h in self.val_history]
        val_r2_pw_filt = [h.get("r2_per_well_filtered", 0.0) for h in self.val_history]
        val_trend_acc = [h["trend_accuracy"] for h in self.val_history]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top-left: Train loss
        axes[0, 0].plot(epochs, train_total, label="Total Loss")
        axes[0, 0].plot(epochs, train_reg, label="Regression Loss")
        axes[0, 0].set_title(
            f"TRAIN Loss | Raw: {size_label} | Samples: {train_samples:,}"
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Bottom-left: Val loss (5-epoch rolling mean for trend clarity)
        val_total_smooth = self._rolling_mean(val_total, window=5)
        val_reg_smooth = self._rolling_mean(val_reg, window=5)
        axes[1, 0].plot(epochs, val_total_smooth, label="Total Loss")
        axes[1, 0].plot(epochs, val_reg_smooth, label="Regression Loss")
        axes[1, 0].set_title(
            f"VALIDATION Loss (rolling-5) | Raw: {size_label} | Samples: {val_samples:,}"
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Top-right: Trend loss
        axes[0, 1].plot(epochs, train_trend, label="Train Trend Loss")
        axes[0, 1].plot(epochs, val_trend, label="Val Trend Loss")
        axes[0, 1].set_title(f"Trend Loss\n{size_label}")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("BCE Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Bottom-right: R² pooled/per-well/filtered (left y-axis) and Trend Accuracy (right y-axis)
        ax_r2 = axes[1, 1]
        ln1 = ax_r2.plot(
            epochs,
            val_r2,
            marker="o",
            markersize=3,
            color="tab:blue",
            alpha=0.3,
            linestyle=":",
            label="R² (pooled)",
        )
        ln2 = ax_r2.plot(
            epochs,
            val_r2_pw,
            marker="o",
            markersize=3,
            color="tab:blue",
            alpha=0.5,
            label="R² (per-well raw)",
        )
        ln3 = ax_r2.plot(
            epochs,
            val_r2_pw_filt,
            marker="o",
            markersize=3,
            color="tab:green",
            label="R² (per-well filtered)",
        )
        ax_r2.set_xlabel("Epoch")
        ax_r2.set_ylabel("R²", color="tab:blue")
        ax_r2.tick_params(axis="y", labelcolor="tab:blue")
        ax_r2.grid(True)

        ax_acc = ax_r2.twinx()
        ln4 = ax_acc.plot(
            epochs,
            val_trend_acc,
            marker="s",
            markersize=3,
            color="tab:orange",
            label="Trend Accuracy",
        )
        ax_acc.set_ylabel("Trend Accuracy", color="tab:orange")
        ax_acc.tick_params(axis="y", labelcolor="tab:orange")
        ax_acc.set_ylim(0, 1)

        # Combined legend
        lines = ln1 + ln2 + ln3 + ln4
        labels = [l.get_label() for l in lines]
        ax_r2.legend(lines, labels, loc="lower right", fontsize=7)
        ax_r2.set_title(f"Val R² & Trend Accuracy ({size_label})")

        plt.tight_layout()
        plot_path = self._plot_path("training_diagnostics.png", category="training")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Training diagnostics saved to: {plot_path}")

    def generate_diagnostic_plots(self, data=None, sampler=None, split_name="val", n_wells_per_category: int = 5):
        """Wrapper around PlotGenerator.generate_diagnostic_plots (extracted module)."""
        if data is None or sampler is None:
            data = self.val_data
            sampler = self.val_sampler
            split_name = "val"
        return self.plot_gen.generate_diagnostic_plots(
            data=data, sampler=sampler,
            split_name=split_name,
            n_wells_per_category=n_wells_per_category,
        )

    @torch.no_grad()
    def save_test_predictions_to_pkl(self) -> None:
        """
        Run inference on the test set with the currently loaded model and save
        the absolute and delta predictions back to data/test_samples.pkl, so
        downstream tools can read predictions alongside the original sample
        contents without re-running the model.

        Each sample dict is augmented with:
            "predicted_gwl"          — absolute prediction (post-clamp, what
                                       plots/metrics see)
            "predicted_gwl_delta"    — delta prediction (post-clamp)
            "actual_pred_gwl"        — absolute prediction (pre-clamp; raw
                                       model output)
            "actual_pred_gwl_delta"  — delta prediction (pre-clamp)
        Mapping is positional: scaler.transform preserves list order and
        test_sampler has shuffle=False, so test_samples[i] ↔ self.test_data[i].
        """
        test_pkl = os.path.join(self.config.data_dir, "test_samples.pkl")
        if not os.path.exists(test_pkl):
            print(f"  save_test_predictions: {test_pkl} not found — skipping")
            return
        with open(test_pkl, "rb") as f:
            test_samples = pickle.load(f)
        if len(test_samples) != len(self.test_data.target_gwl):
            print(
                f"  save_test_predictions: length mismatch "
                f"test_samples={len(test_samples)} vs test_data={len(self.test_data.target_gwl)} — skipping"
            )
            return

        self.model.eval()
        amp_device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"

        preds_abs_clamped_l = []
        preds_delta_clamped_l = []
        preds_abs_raw_l = []
        preds_delta_raw_l = []
        for batch_indices in self.test_sampler:
            batch = self.test_data.get_batch(batch_indices)
            with autocast(device_type=amp_device_type, enabled=self.use_amp, dtype=self.amp_dtype):
                if self.config.use_revin:
                    sequence_in, mean_s, std_s = self._apply_revin(batch["sequence"])
                else:
                    sequence_in = batch["sequence"]
                    mean_s = std_s = None
                gwl_pred = self._model_forward(sequence_in, batch)
                if self.config.use_revin:
                    gwl_pred = (gwl_pred.squeeze(1) * std_s + mean_s).unsqueeze(1)
            current_gwl_raw = batch["current_gwl_raw"]
            if self.use_delta_gwl:
                pred_delta_raw = self._inverse_transform_on_device(gwl_pred.squeeze(1).float())
                pred_abs_raw = current_gwl_raw + pred_delta_raw
            else:
                pred_abs_raw = self._inverse_transform_on_device(gwl_pred.squeeze(1).float())
                pred_delta_raw = pred_abs_raw - current_gwl_raw
            pred_abs_clamped, pred_delta_clamped = self._clamp_pred_to_current(
                pred_abs_raw, pred_delta_raw, current_gwl_raw
            )
            preds_abs_clamped_l.append(pred_abs_clamped.detach().cpu().numpy())
            preds_delta_clamped_l.append(pred_delta_clamped.detach().cpu().numpy())
            preds_abs_raw_l.append(pred_abs_raw.detach().cpu().numpy())
            preds_delta_raw_l.append(pred_delta_raw.detach().cpu().numpy())

        preds_abs_clamped = np.concatenate(preds_abs_clamped_l) if preds_abs_clamped_l else np.array([])
        preds_delta_clamped = np.concatenate(preds_delta_clamped_l) if preds_delta_clamped_l else np.array([])
        preds_abs_raw = np.concatenate(preds_abs_raw_l) if preds_abs_raw_l else np.array([])
        preds_delta_raw = np.concatenate(preds_delta_raw_l) if preds_delta_raw_l else np.array([])

        for i, s in enumerate(test_samples):
            # Clamped values — the ones used in plots/metrics downstream.
            s["predicted_gwl"] = float(preds_abs_clamped[i])
            s["predicted_gwl_delta"] = float(preds_delta_clamped[i])
            # Raw model output before the val/test clamp — kept so the
            # difference (clamped vs unclamped) is inspectable.
            s["actual_pred_gwl"] = float(preds_abs_raw[i])
            s["actual_pred_gwl_delta"] = float(preds_delta_raw[i])

        with open(test_pkl, "wb") as f:
            pickle.dump(test_samples, f)
        print(f"  Saved test predictions to {test_pkl} (n={len(test_samples)})")

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 70)
        print("Starting training")
        print("=" * 70)
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")
        print(f"Training batches: {len(self.train_sampler)}")
        print(f"Validation batches: {len(self.val_sampler)}")
        print()

        total_start = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch + 1
            epoch_start = time.time()

            print(f"Epoch {self.current_epoch}/{self.config.num_epochs}")
            print("-" * 70)

            # Train
            train_metrics = self.train_epoch()
            self.train_history.append(train_metrics)

            # Validate (per-well metrics + plots only every 10 epochs or last epoch)
            is_detailed_epoch = (self.current_epoch % 10 == 0) or (
                self.current_epoch == self.config.num_epochs
            )
            print("  Validating...")
            val_metrics = self.validate(compute_per_well=is_detailed_epoch)
            self.val_history.append(val_metrics)

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["total_loss"],
                    "val_loss": val_metrics["total_loss"],
                    "val_rmse": val_metrics["rmse"],
                    "val_r2": val_metrics["r2"],
                    "val_r2_per_well": val_metrics["r2_per_well"],
                    "val_r2_delta_per_well": val_metrics["r2_delta_per_well"],
                    "val_r2_blend_per_well": val_metrics["r2_blend_per_well"],
                    "val_r2_persist_per_well": val_metrics["r2_persist_per_well"],
                    "val_kge": val_metrics["kge"],
                    "val_kge_per_well": val_metrics["kge_per_well"],
                    "trend_accuracy": val_metrics["trend_accuracy"],
                    "trend_accuracy_filtered": val_metrics["trend_accuracy_filtered"],
                },
                step=self.current_epoch,
            )

            # Print metrics
            print(
                f"  Train Loss: {train_metrics['total_loss']:.4f} "
                f"(Reg: {train_metrics['regression_loss']:.4f}, "
                f"Trend: {train_metrics['trend_loss']:.4f})"
            )
            val_loss_seq = [v["total_loss"] for v in self.val_history]
            roll5 = self._rolling_mean(val_loss_seq, window=5)[-1]
            print(
                f"  Val Loss:   {val_metrics['total_loss']:.4f} "
                f"(roll-5: {roll5:.4f}) "
                f"(Reg: {val_metrics['regression_loss']:.4f}, "
                f"Trend: {val_metrics['trend_loss']:.4f})"
            )
            print(
                f"  Val RMSE:   {val_metrics['rmse']:.2f}m, "
                f"MAE: {val_metrics['mae']:.2f}m, "
                f"R²: {val_metrics['r2']:.3f} (pooled) / "
                f"{val_metrics['r2_per_well']:.3f} med / {val_metrics['r2_per_well_mean']:.3f} mean (per-well), "
                f"KGE: {val_metrics['kge']:.3f} (pooled) / "
                f"{val_metrics['kge_per_well']:.3f} med / {val_metrics['kge_per_well_mean']:.3f} mean (per-well)"
            )
            print(
                f"  Val R²_δ:   {val_metrics['r2_delta_per_well']:.3f} med / "
                f"{val_metrics['r2_delta_per_well_mean']:.3f} mean   "
                f"R²_blend: {val_metrics['r2_blend_per_well']:.3f} med / "
                f"{val_metrics['r2_blend_per_well_mean']:.3f} mean   "
                f"R²_persist: {val_metrics['r2_persist_per_well']:.3f} med / "
                f"{val_metrics['r2_persist_per_well_mean']:.3f} mean (per-well)"
            )
            print(
                f"  Val MAPE %   pred={val_metrics['mape_pred_med']*100:.1f} med/{val_metrics['mape_pred_mean']*100:.1f} mean  "
                f"blend={val_metrics['mape_blend_med']*100:.1f}/{val_metrics['mape_blend_mean']*100:.1f}  "
                f"persist={val_metrics['mape_persist_med']*100:.1f}/{val_metrics['mape_persist_mean']*100:.1f}"
            )
            print(
                f"  Val MAPEf %  pred={val_metrics['mapef_pred_med']*100:.1f} med/{val_metrics['mapef_pred_mean']*100:.1f} mean  "
                f"blend={val_metrics['mapef_blend_med']*100:.1f}/{val_metrics['mapef_blend_mean']*100:.1f}  "
                f"persist={val_metrics['mapef_persist_med']*100:.1f}/{val_metrics['mapef_persist_mean']*100:.1f}"
            )
            print(
                f"  Val R²    quartiles: abs q25/q75={val_metrics['r2_q25']:.3f}/{val_metrics['r2_q75']:.3f}  "
                f"delta={val_metrics['r2_delta_q25']:.3f}/{val_metrics['r2_delta_q75']:.3f}  "
                f"blend={val_metrics['r2_blend_q25']:.3f}/{val_metrics['r2_blend_q75']:.3f}  "
                f"persist={val_metrics['r2_persist_q25']:.3f}/{val_metrics['r2_persist_q75']:.3f}"
            )
            print(
                f"  Val MAPE % quartiles: pred q25/q75={val_metrics['mape_pred_q25']*100:.1f}/{val_metrics['mape_pred_q75']*100:.1f}  "
                f"blend={val_metrics['mape_blend_q25']*100:.1f}/{val_metrics['mape_blend_q75']*100:.1f}  "
                f"persist={val_metrics['mape_persist_q25']*100:.1f}/{val_metrics['mape_persist_q75']*100:.1f}"
            )
            print(
                f"  Val MAPEf %quartiles: pred q25/q75={val_metrics['mapef_pred_q25']*100:.1f}/{val_metrics['mapef_pred_q75']*100:.1f}  "
                f"blend={val_metrics['mapef_blend_q25']*100:.1f}/{val_metrics['mapef_blend_q75']*100:.1f}  "
                f"persist={val_metrics['mapef_persist_q25']*100:.1f}/{val_metrics['mapef_persist_q75']*100:.1f}"
            )
            print(
                f"  KGE components: r={val_metrics['kge_r']:.3f}, "
                f"β={val_metrics['kge_beta']:.3f}, "
                f"γ={val_metrics['kge_gamma']:.3f}"
            )
            min_dt = self.config.min_delta_for_trend_accuracy
            print(
                f"  Val Trend Accuracy: {val_metrics['trend_accuracy']:.1%} (all) | "
                f"{val_metrics['trend_accuracy_filtered']:.1%} (|Δ|≥{min_dt:g}m, "
                f"N={val_metrics['n_trend_filtered']}) | "
                f"Wells evaluated: {val_metrics['n_wells_evaluated']}"
            )

            # Learning rate scheduler
            self.scheduler.step(val_metrics["total_loss"])

            # Check for improvement — save best model, run all epochs
            is_best = val_metrics["total_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["total_loss"]
                self.best_epoch = self.current_epoch
                self.save_checkpoint(is_best=True)
                print(f"  ✓ New best validation loss: {self.best_val_loss:.4f}")
            else:
                epochs_since_best = self.current_epoch - self.best_epoch
                print(
                    f"  No improvement for {epochs_since_best} epochs (best: epoch {self.best_epoch})"
                )

            # Plot curves every 10 epochs (matplotlib is slow)
            if is_detailed_epoch:
                self.plot_validation_loss()
                self.plot_training_curves()

            epoch_elapsed = time.time() - epoch_start
            print(f"  Epoch time: {epoch_elapsed:.1f}s")
            print()

        total_elapsed = time.time() - total_start
        print("=" * 70)
        print("Training complete!")
        print(
            f"Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})"
        )
        print(
            f"Total training time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)"
        )
        print(f"Final model saved to: {self.run_dir}")
        print("=" * 70)

        # Save training history
        history_path = os.path.join(self.run_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {"train": self.train_history, "val": self.val_history},
                f,
                indent=2,
                default=float,
            )
        print(f"Training history saved to: {history_path}")

        # Load best model, evaluate on test set, generate diagnostic plots
        best_path = os.path.join(self.run_dir, "best_model.pt")
        if os.path.exists(best_path):
            print(f"\nLoading best model (epoch {self.best_epoch}) from {best_path}...")
            checkpoint = torch.load(
                best_path, map_location=self.config.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])

        print("Evaluating best model on TEST set...")
        test_metrics = self.validate(self.test_data, self.test_sampler, "test")
        self.save_test_predictions_to_pkl()
        print(
            f"  Test RMSE: {test_metrics['rmse']:.2f}m, "
            f"MAE: {test_metrics['mae']:.2f}m, "
            f"R²: {test_metrics['r2']:.3f} (pooled) / {test_metrics['r2_per_well']:.3f} (per-well), "
            f"KGE: {test_metrics['kge']:.3f} (pooled) / {test_metrics['kge_per_well']:.3f} (per-well)"
        )
        print(
            f"  Test R²_δ: {test_metrics['r2_delta_per_well']:.3f} med / "
            f"{test_metrics['r2_delta_per_well_mean']:.3f} mean   "
            f"R²_blend: {test_metrics['r2_blend_per_well']:.3f} med / "
            f"{test_metrics['r2_blend_per_well_mean']:.3f} mean   "
            f"R²_persist: {test_metrics['r2_persist_per_well']:.3f} med / "
            f"{test_metrics['r2_persist_per_well_mean']:.3f} mean (per-well)"
        )
        print(
            f"  Test MAPE %   pred={test_metrics['mape_pred_med']*100:.1f} med/{test_metrics['mape_pred_mean']*100:.1f} mean  "
            f"blend={test_metrics['mape_blend_med']*100:.1f}/{test_metrics['mape_blend_mean']*100:.1f}  "
            f"persist={test_metrics['mape_persist_med']*100:.1f}/{test_metrics['mape_persist_mean']*100:.1f}"
        )
        print(
            f"  Test MAPEf %  pred={test_metrics['mapef_pred_med']*100:.1f} med/{test_metrics['mapef_pred_mean']*100:.1f} mean  "
            f"blend={test_metrics['mapef_blend_med']*100:.1f}/{test_metrics['mapef_blend_mean']*100:.1f}  "
            f"persist={test_metrics['mapef_persist_med']*100:.1f}/{test_metrics['mapef_persist_mean']*100:.1f}"
        )
        print(
            f"  Test R²    quartiles: abs q25/q75={test_metrics['r2_q25']:.3f}/{test_metrics['r2_q75']:.3f}  "
            f"delta={test_metrics['r2_delta_q25']:.3f}/{test_metrics['r2_delta_q75']:.3f}  "
            f"blend={test_metrics['r2_blend_q25']:.3f}/{test_metrics['r2_blend_q75']:.3f}  "
            f"persist={test_metrics['r2_persist_q25']:.3f}/{test_metrics['r2_persist_q75']:.3f}"
        )
        print(
            f"  Test MAPE % quartiles: pred q25/q75={test_metrics['mape_pred_q25']*100:.1f}/{test_metrics['mape_pred_q75']*100:.1f}  "
            f"blend={test_metrics['mape_blend_q25']*100:.1f}/{test_metrics['mape_blend_q75']*100:.1f}  "
            f"persist={test_metrics['mape_persist_q25']*100:.1f}/{test_metrics['mape_persist_q75']*100:.1f}"
        )
        print(
            f"  Test MAPEf %quartiles: pred q25/q75={test_metrics['mapef_pred_q25']*100:.1f}/{test_metrics['mapef_pred_q75']*100:.1f}  "
            f"blend={test_metrics['mapef_blend_q25']*100:.1f}/{test_metrics['mapef_blend_q75']*100:.1f}  "
            f"persist={test_metrics['mapef_persist_q25']*100:.1f}/{test_metrics['mapef_persist_q75']*100:.1f}"
        )
        min_dt = self.config.min_delta_for_trend_accuracy
        print(
            f"  Test Trend Accuracy: {test_metrics['trend_accuracy']:.1%} (all) | "
            f"{test_metrics['trend_accuracy_filtered']:.1%} (|Δ|≥{min_dt:g}m, "
            f"N={test_metrics['n_trend_filtered']})"
        )

        self.generate_diagnostic_plots(self.val_data, self.val_sampler, "val")
        self.generate_diagnostic_plots(self.test_data, self.test_sampler, "test")

        # Full-timeline (train + val + test) plot for the worst wells from
        # val ∪ test. Helps disentangle "model never learned this well" from
        # "model overfit / split-specific failure".
        try:
            self.plot_gen.generate_worst_full_timeline_plot(
                self.train_data, self.train_sampler,
                self.val_data, self.val_sampler,
                self.test_data, self.test_sampler,
                n_per_split=5,
            )
        except Exception as e:
            print(f"  Worst-wells full-timeline plot failed: {e}")

        # Geographic analysis of well performance
        self.analyze_well_geography(self.val_data, self.val_sampler, "val")
        self.analyze_well_geography(self.test_data, self.test_sampler, "test")

    @torch.no_grad()
    def analyze_well_geography(self, data=None, sampler=None, split_name="val"):
        """Wrapper around gwl_geo_analysis.analyze_well_geography (extracted module)."""
        if data is None or sampler is None:
            data = self.val_data
            sampler = self.val_sampler
            split_name = "val"
        _analyze_well_geography_fn(
            model=self.model,
            data=data,
            sampler=sampler,
            use_amp=self.use_amp,
            device=self.config.device,
            use_delta_gwl=self.use_delta_gwl,
            inverse_transform_fn=self._inverse_transform_on_device,
            apply_test_cap_and_filter_fn=self.plot_gen._apply_test_cap_and_filter,
            plot_path_fn=self._plot_path,
            perf_dir=self._perf_dir,
            split_name=split_name,
            station_to_train_std=self.plot_gen._station_to_train_std,
            use_revin=self.config.use_revin,
            revin_std_floor=self.config.revin_std_floor,
        )



def load_data(config: TrainingConfig) -> Dict[str, List[Dict]]:
    """Load prepared datasets from disk."""
    print("Loading prepared datasets...")

    with open(os.path.join(config.data_dir, "train_samples.pkl"), "rb") as f:
        train_samples = pickle.load(f)

    with open(os.path.join(config.data_dir, "val_samples.pkl"), "rb") as f:
        val_samples = pickle.load(f)

    with open(os.path.join(config.data_dir, "test_samples.pkl"), "rb") as f:
        test_samples = pickle.load(f)

    dataset_size = len(train_samples) + len(val_samples) + len(test_samples)

    print(f"  Dataset size (total samples): {dataset_size:,}")
    print(f"  Training samples: {len(train_samples):,}")
    print(f"  Validation samples: {len(val_samples):,}")
    print(f"  Test samples: {len(test_samples):,}")

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
        "dataset_size": dataset_size,
    }


def load_or_create_scaler(
    config: TrainingConfig, train_samples: List[Dict]
) -> LSTMFeatureScaler:
    """Load existing scaler or create and fit a new one."""
    scaler_path = os.path.join(config.data_dir, "scalers.pkl")

    if os.path.exists(scaler_path):
        print(f"Loading fitted scalers from {scaler_path}")
        scaler = LSTMFeatureScaler.load(scaler_path)
    else:
        print("No saved scalers found. Fitting new scalers on training data...")
        scaler = LSTMFeatureScaler()
        scaler.fit({"train": train_samples})
        scaler.save(scaler_path)
        print(f"Saved fitted scalers to {scaler_path}")

    return scaler


def log_split_feature_stats(train_transformed, val_transformed, output_dir):
    """
    Diagnostic: Compare feature distributions between train and val after scaling.
    Adapted for LSTM data structure (sequence, forecast_features, static_continuous).
    """
    lines = []
    lines.append("=" * 70)
    lines.append("FEATURE DISTRIBUTION COMPARISON (post-scaling)")
    lines.append("=" * 70)

    # ── Helper ─────────────────────────────────────────────────────────
    def _stats(arr):
        return arr.mean(), arr.std(), arr.min(), arr.max()

    def _flag(t_mean, t_std, v_mean, v_std):
        return (
            " ⚠️"
            if abs(v_mean - t_mean) > 1.0 or (t_std > 0 and v_std > 3 * t_std)
            else ""
        )

    # ── 1. Sequence features (N, T, F) ─────────────────────────────────
    train_seq = np.array([s["sequence"] for s in train_transformed])  # (N, T, F)
    val_seq = np.array([s["sequence"] for s in val_transformed])

    n_train, n_timesteps, n_features = train_seq.shape
    n_val = val_seq.shape[0]

    lines.append(f"\nTrain samples: {n_train:,}  |  Val samples: {n_val:,}")
    lines.append(f"Sequence shape: ({n_timesteps} timesteps, {n_features} features)")

    # Overall sequence stats
    t_m, t_s, t_mn, t_mx = _stats(train_seq)
    v_m, v_s, v_mn, v_mx = _stats(val_seq)
    flag = _flag(t_m, t_s, v_m, v_s)
    lines.append(
        f"\n  SEQUENCE (overall):  train: mean={t_m:+.3f} std={t_s:.3f}"
        f"  |  val: mean={v_m:+.3f} std={v_s:.3f}"
        f" min={v_mn:+.3f} max={v_mx:+.3f}{flag}"
    )

    # Per-feature stats (averaged across timesteps)
    lines.append(f"\n  Per-feature stats (averaged over {n_timesteps} timesteps):")
    lines.append(
        f"  {'feature_idx':>12s}  {'train_mean':>10s} {'train_std':>10s}  |  {'val_mean':>10s} {'val_std':>10s} {'val_min':>10s} {'val_max':>10s}  flag"
    )
    lines.append("  " + "-" * 100)

    for f_idx in range(n_features):
        t_col = train_seq[:, :, f_idx].flatten()
        v_col = val_seq[:, :, f_idx].flatten()
        t_m, t_s, _, _ = _stats(t_col)
        v_m, v_s, v_mn, v_mx = _stats(v_col)
        flag = _flag(t_m, t_s, v_m, v_s)
        lines.append(
            f"  {'feat_' + str(f_idx):>12s}  {t_m:+10.3f} {t_s:10.3f}  |  {v_m:+10.3f} {v_s:10.3f} {v_mn:+10.3f} {v_mx:+10.3f}{flag}"
        )

    # ── 2. Forecast features ───────────────────────────────────────────
    train_fc = np.array([s["forecast_features"] for s in train_transformed])
    val_fc = np.array([s["forecast_features"] for s in val_transformed])

    lines.append(f"\n  FORECAST FEATURES (shape: {train_fc.shape[1]}):")
    t_m, t_s, _, _ = _stats(train_fc)
    v_m, v_s, v_mn, v_mx = _stats(val_fc)
    flag = _flag(t_m, t_s, v_m, v_s)
    lines.append(
        f"    overall:  train: mean={t_m:+.3f} std={t_s:.3f}"
        f"  |  val: mean={v_m:+.3f} std={v_s:.3f}"
        f" min={v_mn:+.3f} max={v_mx:+.3f}{flag}"
    )

    for f_idx in range(train_fc.shape[1]):
        t_m, t_s, _, _ = _stats(train_fc[:, f_idx])
        v_m, v_s, v_mn, v_mx = _stats(val_fc[:, f_idx])
        flag = _flag(t_m, t_s, v_m, v_s)
        lines.append(
            f"    fc_{f_idx}:  train: mean={t_m:+.3f} std={t_s:.3f}"
            f"  |  val: mean={v_m:+.3f} std={v_s:.3f}"
            f" min={v_mn:+.3f} max={v_mx:+.3f}{flag}"
        )

    # ── 3. Static continuous features ──────────────────────────────────
    train_st = np.array([s["static_continuous"] for s in train_transformed])
    val_st = np.array([s["static_continuous"] for s in val_transformed])

    lines.append(f"\n  STATIC CONTINUOUS (shape: {train_st.shape[1]}):")
    for f_idx in range(train_st.shape[1]):
        t_m, t_s, _, _ = _stats(train_st[:, f_idx])
        v_m, v_s, v_mn, v_mx = _stats(val_st[:, f_idx])
        flag = _flag(t_m, t_s, v_m, v_s)
        lines.append(
            f"    static_{f_idx}:  train: mean={t_m:+.3f} std={t_s:.3f}"
            f"  |  val: mean={v_m:+.3f} std={v_s:.3f}"
            f" min={v_mn:+.3f} max={v_mx:+.3f}{flag}"
        )

    # ── 4. ALL GWL (pooled: sequence reliable/present + target) ──────────────────
    def _collect_all_gwl(transformed_samples, raw=False):
        """Pool all GWL readings: sequence where is_reliable=1 + target."""
        gwl_vals = []
        for s in transformed_samples:
            seq = s["sequence"]
            for t in range(seq.shape[0]):
                if seq[t, 1] == 1.0:  # is_reliable flag (index 1)
                    if raw:
                        # Unscaled: need raw value from current_gwl_raw for current timestep only
                        # For sequence GWL we don't have raw per-timestep, so skip raw for sequence
                        pass
                    else:
                        gwl_vals.append(seq[t, 0])  # scaled GWL
            gwl_vals.append(s["target_gwl"])  # target is also GWL
        return np.array(gwl_vals)

    def _collect_raw_gwl(samples):
        """Collect raw (unscaled) GWL: current_gwl_raw + target (unscaled not available post-transform)."""
        return np.array([s["current_gwl_raw"] for s in samples])

    train_gwl_scaled = _collect_all_gwl(train_transformed)
    val_gwl_scaled = _collect_all_gwl(val_transformed)
    train_gwl_raw = _collect_raw_gwl(train_transformed)
    val_gwl_raw = _collect_raw_gwl(val_transformed)

    lines.append(f"\n  ALL GWL — pooled (sequence present + target, scaled):")
    lines.append(
        f"    train: {len(train_gwl_scaled)} readings  mean={train_gwl_scaled.mean():+.3f}"
        f" std={train_gwl_scaled.std():.3f}"
        f" min={train_gwl_scaled.min():+.3f} max={train_gwl_scaled.max():+.3f}"
    )
    lines.append(
        f"    val:   {len(val_gwl_scaled)} readings  mean={val_gwl_scaled.mean():+.3f}"
        f" std={val_gwl_scaled.std():.3f}"
        f" min={val_gwl_scaled.min():+.3f} max={val_gwl_scaled.max():+.3f}"
    )
    gwl_flag = _flag(
        train_gwl_scaled.mean(),
        train_gwl_scaled.std(),
        val_gwl_scaled.mean(),
        val_gwl_scaled.std(),
    )
    if gwl_flag:
        lines.append(
            f"    {gwl_flag} GWL DISTRIBUTION MISMATCH — likely cause of high val loss"
        )

    lines.append(f"\n  CURRENT GWL (raw, unscaled — for reference):")
    lines.append(
        f"    train: mean={train_gwl_raw.mean():+.3f} std={train_gwl_raw.std():.3f}"
        f" min={train_gwl_raw.min():+.3f} max={train_gwl_raw.max():+.3f}"
    )
    lines.append(
        f"    val:   mean={val_gwl_raw.mean():+.3f} std={val_gwl_raw.std():.3f}"
        f" min={val_gwl_raw.min():+.3f} max={val_gwl_raw.max():+.3f}"
    )

    # ── 5. NaN / zero checks ──────────────────────────────────────────
    train_seq_zeros = (train_seq == 0.0).sum()
    val_seq_zeros = (val_seq == 0.0).sum()
    train_seq_total = train_seq.size
    val_seq_total = val_seq.size

    lines.append(
        f"\n  Sequence zeros (missing data): train={train_seq_zeros}/{train_seq_total}"
        f" ({100 * train_seq_zeros / train_seq_total:.1f}%)"
        f"  |  val={val_seq_zeros}/{val_seq_total}"
        f" ({100 * val_seq_zeros / val_seq_total:.1f}%)"
    )

    train_seq_nans = np.isnan(train_seq).sum()
    val_seq_nans = np.isnan(val_seq).sum()
    if train_seq_nans > 0 or val_seq_nans > 0:
        lines.append(f"  ⚠️ NaN in sequences: train={train_seq_nans} val={val_seq_nans}")

    lines.append("=" * 70)

    output = "\n".join(lines)
    print("\n" + output)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "feature_stats.txt")
    with open(filepath, "w") as f:
        f.write(output + "\n")
    print(f"Feature stats saved to: {filepath}")


def create_datasets(
    config: TrainingConfig,
    all_samples: Dict[str, List[Dict]],
    scaler: LSTMFeatureScaler,
) -> Tuple[LSTMTensorDataset, LSTMTensorDataset, LSTMTensorDataset]:
    """
    Transform samples and create GPU-resident tensor datasets.

    Data is transferred to the target device ONCE here. During training,
    batches are created by indexing into device tensors (zero-copy).
    """
    print("\nTransforming samples with fitted scalers...")

    train_samples = all_samples["train"]
    val_samples = all_samples["val"]
    test_samples = all_samples["test"]

    train_transformed = scaler.transform(train_samples)
    val_transformed = scaler.transform(val_samples)
    test_transformed = scaler.transform(test_samples)

    print(f"  Transformed {len(train_transformed):,} training samples")
    print(f"  Transformed {len(val_transformed):,} validation samples")
    print(f"  Transformed {len(test_transformed):,} test samples")

    # Log feature distribution comparison (critical for debugging train/val mismatch)
    run_dir = os.path.join(config.output_dir, config.run_name)
    perf_dir = os.path.join(run_dir, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    log_split_feature_stats(train_transformed, val_transformed, output_dir=perf_dir)

    # Build station → location/state/district/well_type/aquifer lookups from raw samples
    station_to_latlon = {}
    station_to_state = {}
    station_to_district = {}
    station_to_depth = {}
    station_to_well_type = {}
    station_to_aquifer_type = {}
    for split_samples in [train_samples, val_samples, test_samples]:
        for s in split_samples:
            code = s["station_code"]
            if code not in station_to_latlon:
                station_to_latlon[code] = (s["latitude"], s["longitude"])
            if code not in station_to_state:
                station_to_state[code] = s.get("state", "unknown")
            if code not in station_to_district:
                station_to_district[code] = s.get("district", "unknown")
            if code not in station_to_depth:
                station_to_depth[code] = s.get("well_depth", None)
            if code not in station_to_well_type:
                station_to_well_type[code] = s.get("well_type", "unknown")
            if code not in station_to_aquifer_type:
                station_to_aquifer_type[code] = s.get("aquifer_type", "unknown")

    # Pre-transfer entire datasets to device (single bulk CPU→GPU copy)
    print(f"\nPre-loading datasets to {config.device}...")
    train_data = LSTMTensorDataset(
        train_transformed, device=config.device, use_delta_gwl=config.use_delta_gwl,
        use_weighted_loss=config.use_weighted_loss,
        loss_weight_type=config.loss_weight_type,
    )
    val_data = LSTMTensorDataset(
        val_transformed, device=config.device, use_delta_gwl=config.use_delta_gwl,
        use_weighted_loss=config.use_weighted_loss,
        loss_weight_type=config.loss_weight_type,
    )
    test_data = LSTMTensorDataset(
        test_transformed, device=config.device, use_delta_gwl=config.use_delta_gwl,
        use_weighted_loss=config.use_weighted_loss,
        loss_weight_type=config.loss_weight_type,
    )

    # Attach location metadata for geographic analysis
    for dataset in [train_data, val_data, test_data]:
        dataset.station_to_latlon = station_to_latlon
        dataset.station_to_state = station_to_state
        dataset.station_to_district = station_to_district
        dataset.station_to_depth = station_to_depth
        dataset.station_to_well_type = station_to_well_type
        dataset.station_to_aquifer_type = station_to_aquifer_type

    return train_data, val_data, test_data


def main(args):
    """Main training entry point."""

    config = TrainingConfig(
        # Paths
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        # Forecast features
        num_forecast_features=2 if args.rain_temp_only else 6,
        # Training hyperparameters
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        num_epochs=args.epochs,
        device=args.device,
        forecast_horizon_months=args.forecast_horizon_months,
        pred_clamp_pct=args.pred_clamp_pct,
        # Model selection
        model_type=args.model_type,
        use_static_features=args.use_static_features,
        use_weighted_loss=args.use_weighted_loss,
        loss_weight_type=args.loss_weight_type,
        loss_trim_high=args.loss_trim_high,
        use_revin=args.use_revin,
        revin_std_floor=args.revin_std_floor,
        min_delta_for_trend_accuracy=args.min_delta_for_trend,
        min_station_target_std=args.min_station_target_std,
        include_states=[s.strip() for s in args.include_states.split(",") if s.strip()] if args.include_states else [],
        min_station_mode_gap_days=args.min_station_mode_gap_days,
        use_linear_residual=args.use_linear_residual,
        # Model architecture
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_dropout=args.lstm_dropout,
        static_embedding_dim=args.static_embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout_rate=args.dropout,
        # Transformer-specific
        transformer_d_model=args.transformer_d_model,
        transformer_nhead=args.transformer_nhead,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_dropout=args.transformer_dropout,
        # TFT-specific
        tft_d_model=args.tft_d_model,
        tft_n_heads=args.tft_n_heads,
        tft_lstm_layers=args.tft_lstm_layers,
        tft_dropout=args.tft_dropout,
        # Embedding dimensions
        lithology_embedding_dim=args.lithology_emb_dim,
        well_type_embedding_dim=args.well_type_emb_dim,
        aquifer_embedding_dim=args.aquifer_emb_dim,
        aquifer_0_aquifer_embedding_dim=args.aquifer_0_aquifer_emb_dim,
        litho_supergroup_embedding_dim=args.litho_supergroup_emb_dim,
        lulc_embedding_dim=args.lulc_emb_dim,
        state_embedding_dim=args.state_emb_dim,
        district_embedding_dim=args.district_emb_dim,
        # Loss
        alpha=args.alpha,
        beta=args.beta,
        huber_delta=args.huber_delta,
        use_uncertainty_weighting=args.uncertainty_weighting,
        # Scheduler
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        # Logging
        log_interval=args.log_interval,
        # Performance
        use_amp=args.use_amp,
        # Prithvi-EO joint fine-tune
        use_prithvi=args.use_prithvi,
        prithvi_model_dir=args.prithvi_model_dir,
        station_index_csv=args.station_index_csv,
        prithvi_proj_dim=args.prithvi_proj_dim,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        ft_lr=args.ft_lr,
        forecaster_lr_scale=args.forecaster_lr_scale,
        prithvi_n_tiles=args.prithvi_n_tiles,
        prithvi_samples_per_tile=args.prithvi_samples_per_tile,
        prithvi_grad_ckpt=args.prithvi_grad_ckpt,
    )

    print("=" * 70)
    print("GWL Forecasting LSTM Training")
    print("=" * 70)
    print(f"Configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    # Show the effective pred_clamp_pct when not explicitly set, so the printed
    # config reflects what will actually be applied during val/test inference.
    if config.pred_clamp_pct is None:
        _effective_pct = 0.20 if int(config.forecast_horizon_months) <= 4 else 0.50
        print(
            f"  pred_clamp_pct (effective): {_effective_pct} "
            f"(default for horizon={config.forecast_horizon_months}mo)"
        )
    print()

    # Load data
    all_samples = load_data(config)

    data_metadata = {
        "dataset_size": all_samples["dataset_size"],
    }
    print(f"Data metadata: {data_metadata}")

    # Load or create scaler
    scaler = load_or_create_scaler(config, all_samples["train"])

    # Auto-detect delta mode from scaler config
    if hasattr(scaler.config, "use_delta_gwl"):
        config.use_delta_gwl = scaler.config.use_delta_gwl
        print(f"\nAuto-detected from scaler: use_delta_gwl={config.use_delta_gwl}")

    # Update config with actual values from data
    if len(all_samples["train"]) > 0:
        sample = all_samples["train"][0]
        config.num_timesteps = sample["sequence"].shape[0]
        config.features_per_timestep = sample["sequence"].shape[1]
        # Derive forecast-feature count from the data too (was hardcoded to 6/2
        # via --rain-temp-only, which silently mismatched datasets with a
        # different forecast width). Keeps the model head aligned to the data.
        config.num_forecast_features = len(sample["forecast_features"])

    # Update config with actual number of categories from scaler
    config.num_lithology_types = scaler.embedding_metadata["num_lithology_types"]
    config.num_well_types = scaler.embedding_metadata["num_well_types"]
    config.num_aquifer_types = scaler.embedding_metadata["num_aquifer_types"]
    config.num_aquifer_0_aquifer_types = scaler.embedding_metadata.get(
        "num_aquifer_0_aquifer_types", 1
    )
    config.num_litho_supergroup_types = scaler.embedding_metadata.get(
        "num_litho_supergroup_types", 1
    )
    config.num_lulc_types = scaler.embedding_metadata["num_lulc_types"]
    config.num_state_types = scaler.embedding_metadata.get(
        "num_state_types", 1
    )
    config.num_district_types = scaler.embedding_metadata.get(
        "num_district_types", 1
    )

    print(f"\nUpdated config from data:")
    print(f"  GWL mode: {'DELTA' if config.use_delta_gwl else 'ABSOLUTE'}")
    print(f"  Timesteps: {config.num_timesteps}")
    print(f"  Features per timestep: {config.features_per_timestep}")
    print(f"  Forecast features: {config.num_forecast_features} (for conditioning)")
    print(f"  Lithology types: {config.num_lithology_types}")
    print(f"  Well types: {config.num_well_types}")
    print(f"  Aquifer types: {config.num_aquifer_types}")
    print(f"  Aquifer 0 Aquifer types: {config.num_aquifer_0_aquifer_types}")
    print(f"  Litho Supergroup types: {config.num_litho_supergroup_types}")
    print(f"  LULC types: {config.num_lulc_types}")
    print(f"  State types: {config.num_state_types}")
    print(f"  District types: {config.num_district_types}")

    # Create datasets (pre-loaded to device)
    train_data, val_data, test_data = create_datasets(config, all_samples, scaler)

    # Create model
    print(f"\nCreating {config.model_type} model...")
    # Shared kwargs for both architectures
    shared_model_kwargs = dict(
        num_timesteps=config.num_timesteps,
        features_per_timestep=config.features_per_timestep,
        static_embedding_dim=config.static_embedding_dim,
        fusion_hidden_dim=config.fusion_hidden_dim,
        num_lithology_types=config.num_lithology_types,
        num_well_types=config.num_well_types,
        num_aquifer_types=config.num_aquifer_types,
        num_aquifer_0_aquifer_types=max(config.num_aquifer_0_aquifer_types, 1),
        num_litho_supergroup_types=max(config.num_litho_supergroup_types, 1),
        num_lulc_types=config.num_lulc_types,
        num_state_types=max(config.num_state_types, 1),
        num_district_types=max(config.num_district_types, 1),
        lithology_embedding_dim=config.lithology_embedding_dim,
        well_type_embedding_dim=config.well_type_embedding_dim,
        aquifer_embedding_dim=config.aquifer_embedding_dim,
        aquifer_0_aquifer_embedding_dim=config.aquifer_0_aquifer_embedding_dim,
        litho_supergroup_embedding_dim=config.litho_supergroup_embedding_dim,
        lulc_embedding_dim=config.lulc_embedding_dim,
        state_embedding_dim=config.state_embedding_dim,
        district_embedding_dim=config.district_embedding_dim,
        num_forecast_features=config.num_forecast_features,
        dropout_rate=config.dropout_rate,
        use_static_features=config.use_static_features,
        use_linear_residual=config.use_linear_residual,
    )

    # Build the live Prithvi projector (only when requested; TFT-only).
    prithvi_projector = None
    if config.use_prithvi:
        if config.model_type != "tft":
            raise ValueError("--use-prithvi requires --model-type tft")
        import pickle as _pickle
        from gwlcore.prithvi_finetune import build_projector
        manifest_path = os.path.join(config.data_dir, "tile_manifest.pkl")
        with open(manifest_path, "rb") as _mf:
            _manifest = _pickle.load(_mf)
        print(f"\n  Prithvi: loaded tile_manifest.pkl "
              f"({len(_manifest['ordered_files'])} composites, "
              f"period={_manifest['period']}, zero_idx={_manifest['zero_idx']})")
        prithvi_projector = build_projector(
            _manifest,
            model_dir=config.prithvi_model_dir,
            station_csv=config.station_index_csv,
            proj_dim=config.prithvi_proj_dim,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            device=config.device,
        )
        if config.prithvi_grad_ckpt:
            enc = prithvi_projector.encoder
            if hasattr(enc, "gradient_checkpointing_enable"):
                enc.gradient_checkpointing_enable()
                print("  Prithvi: gradient checkpointing enabled")
            else:
                print("  Prithvi: WARNING grad-ckpt requested but encoder has no "
                      "gradient_checkpointing_enable(); ignoring")

    if config.model_type == "tft":
        model = GWLForecastTFT(
            **shared_model_kwargs,
            d_model=config.tft_d_model,
            n_heads=config.tft_n_heads,
            lstm_layers=config.tft_lstm_layers,
            tft_dropout=config.tft_dropout,
            use_prithvi=config.use_prithvi,
            prithvi_proj_dim=config.prithvi_proj_dim,
            prithvi_projector=prithvi_projector,
        )
    elif config.model_type == "transformer":
        model = GWLForecastTransformer(
            **shared_model_kwargs,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_encoder_layers=config.transformer_num_layers,
            dim_feedforward=config.transformer_dim_feedforward,
            transformer_dropout=config.transformer_dropout,
        )
    elif config.model_type == "conditioned_lstm":
        model = GWLForecastConditionedLSTM(
            **shared_model_kwargs,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_num_layers=config.lstm_num_layers,
            lstm_dropout=config.lstm_dropout,
        )
    elif config.model_type == "conditioned_transformer":
        model = GWLForecastConditionedTransformer(
            **shared_model_kwargs,
            d_model=config.transformer_d_model,
            nhead=config.transformer_nhead,
            num_encoder_layers=config.transformer_num_layers,
            dim_feedforward=config.transformer_dim_feedforward,
            transformer_dropout=config.transformer_dropout,
        )
    else:
        model = GWLForecastLSTM(
            **shared_model_kwargs,
            lstm_hidden_dim=config.lstm_hidden_dim,
            lstm_num_layers=config.lstm_num_layers,
            lstm_dropout=config.lstm_dropout,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        scaler=scaler,
        data_metadata=data_metadata,
    )

    with mlflow.start_run(run_name=config.run_name):

        mlflow.log_params(
            {
                # Training hyperparams
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "dropout_rate": config.dropout_rate,
                "gradient_clip_norm": config.gradient_clip_norm,
                # Loss
                "alpha": config.alpha,
                "beta": config.beta,
                "use_uncertainty_weighting": config.use_uncertainty_weighting,
                # GWL mode
                "use_delta_gwl": config.use_delta_gwl,
                # Architecture
                "model_type": config.model_type,
                "use_static_features": config.use_static_features,
                "lstm_hidden_dim": config.lstm_hidden_dim,
                "lstm_num_layers": config.lstm_num_layers,
                "lstm_dropout": config.lstm_dropout,
                "static_embedding_dim": config.static_embedding_dim,
                "fusion_hidden_dim": config.fusion_hidden_dim,
                "num_timesteps": config.num_timesteps,
                "features_per_timestep": config.features_per_timestep,
                # Scheduler
                "scheduler_factor": config.scheduler_factor,
                "scheduler_patience": config.scheduler_patience,
            }
        )

        mlflow.log_params(
            {
                "dataset_size": data_metadata["dataset_size"],
                "total_model_params": total_params,
                "trainable_params": trainable_params,
            }
        )

        # Train
        trainer.train()

        # Log artifacts to MLflow
        try:
            mlflow.pytorch.log_model(model, "model")
        except Exception as e:
            print(f"Warning: mlflow.pytorch.log_model failed: {e}")
            print(
                "  (Known setuptools/distutils compatibility issue — skipping model artifact)"
            )

        # Log entire plots and performance directories to MLflow
        for subdir in ["plots", "performance"]:
            subdir_path = os.path.join(trainer.run_dir, subdir)
            if os.path.isdir(subdir_path):
                try:
                    mlflow.log_artifacts(subdir_path, artifact_path=subdir)
                except Exception as e:
                    print(f"Warning: failed to log {subdir}: {e}")
        # Log top-level files
        for artifact_name in ["config.json", "data_metadata.json", "training_history.json"]:
            artifact_path = os.path.join(trainer.run_dir, artifact_name)
            if os.path.exists(artifact_path):
                try:
                    mlflow.log_artifact(artifact_path)
                except Exception as e:
                    print(f"Warning: failed to log {artifact_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GWL Forecasting LSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths ──────────────────────────────────────────────────────────
    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--data-dir",
        type=str,
        default="gwl_lstm/data",
        help="Directory with prepared data",
    )
    paths.add_argument(
        "--output-dir",
        type=str,
        default="gwl_lstm/runs",
        help="Output directory for checkpoints and logs",
    )
    paths.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name (auto-generated if not set)",
    )

    # ── Training hyperparameters ───────────────────────────────────────
    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=64, help="Batch size")
    training.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    training.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay (L2 regularization)",
    )
    training.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Max gradient norm (0 to disable)",
    )
    training.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    training.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    training.add_argument(
        "--forecast-horizon-months",
        type=int,
        default=6,
        help=(
            "Forecast horizon in months. Used to pick the default "
            "--pred-clamp-pct (<=4mo -> 0.20, >4mo -> 0.50)."
        ),
    )
    training.add_argument(
        "--pred-clamp-pct",
        type=float,
        default=None,
        help=(
            "Cap |pred - current_gwl_raw| to pct * |current_gwl_raw| in "
            "val/test inference, saved test preds, and diagnostic plots. "
            "If unset, defaults to 0.20 for horizon <=4mo and 0.50 for "
            ">4mo. Set 0.0 to disable."
        ),
    )

    # ── Model selection ───────────────────────────────────────────────
    parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["lstm", "transformer", "conditioned_lstm", "conditioned_transformer", "tft"],
        help="Model architecture to use (default: lstm)",
    )

    parser.add_argument(
        "--rain-temp-only",
        action="store_true",
        default=False,
        help="Use only rainfall and temp as forecast features (2 instead of 6)",
    )

    # ── Prithvi-EO joint fine-tune ────────────────────────────────────
    _pri = parser.add_argument_group("Prithvi-EO joint fine-tune (requires --model-type tft)")
    _pri.add_argument(
        "--use-prithvi",
        action="store_true",
        default=os.environ.get("USE_PRITHVI", "").lower() in ("1", "true", "yes"),
        help="Jointly fine-tune Prithvi-EO (LoRA) with the TFT. Needs "
             "tile_manifest.pkl in --data-dir (from data_preparation --use-prithvi).",
    )
    _pri.add_argument(
        "--prithvi-model-dir", type=str,
        default=os.environ.get("PRITHVI_MODEL_DIR", ""),
        help="Dir with the Prithvi-EO .pt checkpoint + prithvi_mae.py + config.json.",
    )
    _pri.add_argument(
        "--station-index-csv", type=str,
        default=os.environ.get("STATION_INDEX_CSV", ""),
        help="gwl_stations.csv with columns safe_id, lat, lon (for location coords).",
    )
    _pri.add_argument(
        "--prithvi-proj-dim", type=int,
        default=int(os.environ.get("PRITHVI_PROJ_DIM", 32)),
        help="Projected satellite-feature dim concatenated into statics (default 32).",
    )
    _pri.add_argument(
        "--lora-r", type=int, default=int(os.environ.get("LORA_R", 16)),
        help="LoRA rank on encoder attn.qkv (default 16).",
    )
    _pri.add_argument(
        "--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", 32)),
        help="LoRA alpha (default 32).",
    )
    _pri.add_argument(
        "--ft-lr", type=float, default=float(os.environ.get("FT_LR", 1e-4)),
        help="LR for Prithvi LoRA + projector (default 1e-4).",
    )
    _pri.add_argument(
        "--forecaster-lr-scale", type=float,
        default=float(os.environ.get("FORECASTER_LR_SCALE", 10.0)),
        help="TFT LR = ft_lr * this (default 10 → 1e-3 for from-scratch TFT).",
    )
    _pri.add_argument(
        "--prithvi-n-tiles", type=int,
        default=int(os.environ.get("PRITHVI_N_TILES", 256)),
        help="Unique tiles per train batch (grouped sampler; default 256).",
    )
    _pri.add_argument(
        "--prithvi-samples-per-tile", type=int,
        default=int(os.environ.get("PRITHVI_SAMPLES_PER_TILE", 32)),
        help="Samples per tile → train batch = n_tiles*spt (default 32 → 8192).",
    )
    _pri.add_argument(
        "--prithvi-grad-ckpt",
        action="store_true",
        default=os.environ.get("PRITHVI_GRAD_CKPT", "").lower() in ("1", "true", "yes"),
        help="Gradient-checkpoint Prithvi encoder blocks to save memory.",
    )

    parser.add_argument(
        "--no-static-features",
        action="store_false",
        dest="use_static_features",
        default=True,
        help="Drop all static features (lat/lon/elevation/lithology/etc) "
             "except well_depth. Forces model to rely on temporal features.",
    )
    parser.add_argument(
        "--no-weighted-loss",
        action="store_false",
        dest="use_weighted_loss",
        default=True,
        help="Disable per-station weighted loss. Default: weighted loss ON.",
    )
    parser.add_argument(
        "--loss-weight-type",
        type=str,
        choices=["std", "perwell"],
        default="std",
        help="Weighting strategy when use_weighted_loss is enabled: "
             "'std' = per-sample 1/std with pooled mean (legacy), "
             "'perwell' = per-sample 1/var with mean-per-well aggregation "
             "(directly targets per-well R²). Default: std.",
    )
    parser.add_argument(
        "--loss-trim-high",
        type=float,
        default=0.0,
        help="Dynamic top-trim for perwell loss: fraction (0-1) of highest-loss "
             "wells to drop from each batch's gradient. 0.0 = off. Only active "
             "in 'perwell' mode. Example: 0.10 drops worst 10%% of wells/batch.",
    )
    parser.add_argument(
        "--use-revin",
        action="store_true",
        help="Enable RevIN (reversible per-sample instance normalization) on "
             "the GWL channel. Equalizes gradient contribution across flat and "
             "volatile wells. Recommended pairing: loss_weight_type='none'.",
    )
    parser.add_argument(
        "--revin-std-floor",
        type=float,
        default=0.1,
        help="RevIN per-sample std floor (scaled-delta units; ~1m real for 0.1). "
             "Caps de-norm blow-up on flat-well samples where lookback outliers "
             "inflate std_s. Try 0.05 / 0.1 / 0.2 in A/B.",
    )
    parser.add_argument(
        "--min-delta-for-trend",
        type=float,
        default=0.5,
        help="Min |target_delta| (meters) for inclusion in the *filtered* trend "
             "accuracy metric. Reports a second accuracy on samples where the "
             "target actually moved, ignoring near-flat noise. 0.0 disables.",
    )
    parser.add_argument(
        "--min-station-target-std",
        type=float,
        default=0.0,
        help="Train-derived per-station std (of target_delta) threshold. Used "
             "as both the data-prep flat filter (drops stations with train_std < "
             "threshold from train+val) and the eval-time predictable cohort "
             "threshold (in plots, metric blocks, CSVs). One value, one meaning. "
             "Set to 0.0 to disable filtering and skip the predictable metric block.",
    )
    parser.add_argument(
        "--include-states",
        type=str,
        default="",
        help="Comma-separated list of states to keep in train+val. "
             "Composes with std filter by AND. Test stays unfiltered. "
             "Empty = disabled. Example: 'Kerala,Telangana,Andhra Pradesh'.",
    )
    parser.add_argument(
        "--min-station-mode-gap-days",
        type=float,
        default=0.0,
        help="Drop stations from train+val whose RAW observation mode-gap "
             "(most-common gap days between consecutive readings) is below "
             "this threshold. Targets densely-sampled flat wells which the "
             "model fits to noise. Test stays unfiltered. 0=disabled.",
    )
    parser.add_argument(
        "--use-linear-residual",
        action="store_true",
        default=False,
        help="Enable an additive linear baseline path on top of the main "
             "LSTM/TFT regression head. Linear takes a 13-dim slice of static + "
             "state + forecast features and predicts a per-well baseline; the "
             "LSTM/TFT implicitly learns the residual. Off by default.",
    )

    # ── Model architecture ─────────────────────────────────────────────
    arch = parser.add_argument_group("model architecture (shared)")
    arch.add_argument(
        "--static-embedding-dim",
        type=int,
        default=32,
        help="Static feature embedding dimension",
    )
    arch.add_argument(
        "--fusion-hidden-dim",
        type=int,
        default=64,
        help="Fusion layer hidden dimension",
    )
    arch.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # ── LSTM-specific ──────────────────────────────────────────────────
    lstm_arch = parser.add_argument_group("LSTM architecture")
    lstm_arch.add_argument(
        "--lstm-hidden-dim", type=int, default=128, help="LSTM hidden dimension"
    )
    lstm_arch.add_argument(
        "--lstm-num-layers", type=int, default=2, help="Number of LSTM layers"
    )
    lstm_arch.add_argument(
        "--lstm-dropout", type=float, default=0.2, help="LSTM dropout between layers"
    )

    # ── Transformer-specific ───────────────────────────────────────────
    tf_arch = parser.add_argument_group("Transformer architecture")
    tf_arch.add_argument(
        "--transformer-d-model", type=int, default=64, help="Transformer model dimension"
    )
    tf_arch.add_argument(
        "--transformer-nhead", type=int, default=4, help="Number of attention heads"
    )
    tf_arch.add_argument(
        "--transformer-num-layers", type=int, default=2, help="Number of transformer encoder layers"
    )
    tf_arch.add_argument(
        "--transformer-dim-feedforward", type=int, default=128, help="Transformer feedforward dimension"
    )
    tf_arch.add_argument(
        "--transformer-dropout", type=float, default=0.2, help="Transformer dropout"
    )

    # ── TFT architecture ──────────────────────────────────────────────
    tft_arch = parser.add_argument_group("TFT architecture")
    tft_arch.add_argument(
        "--tft-d-model", type=int, default=64, help="TFT internal dimension"
    )
    tft_arch.add_argument(
        "--tft-n-heads", type=int, default=4, help="TFT attention heads"
    )
    tft_arch.add_argument(
        "--tft-lstm-layers", type=int, default=1, help="TFT LSTM encoder layers"
    )
    tft_arch.add_argument(
        "--tft-dropout", type=float, default=0.1, help="TFT dropout"
    )

    # ── Embedding dimensions ───────────────────────────────────────────
    emb = parser.add_argument_group("embedding dimensions")
    emb.add_argument(
        "--lithology-emb-dim", type=int, default=4, help="Lithology embedding dim"
    )
    emb.add_argument(
        "--well-type-emb-dim", type=int, default=3, help="Well type embedding dim"
    )
    emb.add_argument(
        "--aquifer-emb-dim", type=int, default=2, help="Aquifer embedding dim"
    )
    emb.add_argument(
        "--aquifer-0-aquifer-emb-dim",
        type=int,
        default=3,
        help="Aquifer 0 aquifer embedding dim",
    )
    emb.add_argument(
        "--litho-supergroup-emb-dim",
        type=int,
        default=3,
        help="Litho supergroup embedding dim",
    )
    emb.add_argument("--lulc-emb-dim", type=int, default=3, help="LULC embedding dim")
    emb.add_argument("--state-emb-dim", type=int, default=4, help="State embedding dim")
    emb.add_argument("--district-emb-dim", type=int, default=5, help="District embedding dim")

    # ── Loss & trend ───────────────────────────────────────────────────
    loss = parser.add_argument_group("loss and trend")
    loss.add_argument(
        "--alpha", type=float, default=0.7, help="Regression loss weight (MSE)"
    )
    loss.add_argument(
        "--beta", type=float, default=0.3, help="Trend loss weight (BCE on delta sign)"
    )
    loss.add_argument(
        "--huber-delta",
        type=float,
        default=0.0,
        help="Huber loss delta (> 0 enables HuberLoss instead of MSE). "
        "Values in normalised delta space — try 0.5–2.0 to start.",
    )
    loss.add_argument(
        "--uncertainty-weighting",
        action="store_true",
        default=False,
        help="Use learnable uncertainty weighting for loss",
    )

    # ── LR scheduler ───────────────────────────────────────────────────
    sched = parser.add_argument_group("learning rate scheduler")
    sched.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
        help="LR scheduler reduction factor",
    )
    sched.add_argument(
        "--scheduler-patience",
        type=int,
        default=10,
        help="LR scheduler patience (epochs)",
    )

    # ── Logging ────────────────────────────────────────────────────────
    log = parser.add_argument_group("logging")
    log.add_argument("--log-interval", type=int, default=10, help="Log every N batches")

    # ── Performance ────────────────────────────────────────────────────
    perf = parser.add_argument_group("performance")
    perf.add_argument(
        "--no-amp",
        dest="use_amp",
        action="store_false",
        help="Disable automatic mixed precision (AMP)",
    )
    parser.set_defaults(use_amp=True)

    args = parser.parse_args()

    main(args)
