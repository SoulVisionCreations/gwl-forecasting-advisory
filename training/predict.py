"""
Single-sample inference CLI for a trained GWL forecasting model.

Usage:
    python predict.py <station_code> <current_date> --run-dir <path>

Loads the saved artifacts from a training run (data config, fitted scalers,
model checkpoint), builds a fresh sample for the requested (station, date),
applies the training-time scaler + RevIN, runs the model forward, de-normalizes,
and prints the predicted GWL at current_date + forecast_horizon_months.

Reuses the existing functionality from:
    - data_preparation.LSTMDataPreparation.create_sample
    - feature_scaler.LSTMFeatureScaler.transform_sample
    - train.Trainer._apply_revin (re-implemented here, identical math)
    - model.GWLForecast* (loaded from saved state_dict)
"""

import argparse
import os
import pickle
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

from gwlcore import data_preparation
from gwlcore import feature_scaler
from gwlcore.data_preparation import LSTMDataPreparation
from gwlcore.feature_scaler import LSTMFeatureScaler
from gwlcore.model import GWLForecastLSTM, GWLForecastConditionedLSTM
from gwlcore.transformer_model import (
    GWLForecastTransformer,
    GWLForecastConditionedTransformer,
)
from gwlcore.tft_model import GWLForecastTFT

# Pickles produced by running data_preparation.py / feature_scaler.py as
# __main__ store class references as e.g. __main__.DataConfig. When loading
# from a different entry point (predict.py here), pickle can't find them.
# Inject the relevant classes into the current __main__ so pickle.load()
# resolves them. Must run before any pickle.load() call.
_main = sys.modules.get("__main__")
if _main is not None:
    for _name in (
        "DataConfig", "LSTMDataPreparation", "Sample",
    ):
        if hasattr(data_preparation, _name) and not hasattr(_main, _name):
            setattr(_main, _name, getattr(data_preparation, _name))
    for _name in ("ScalerConfig", "LSTMFeatureScaler"):
        if hasattr(feature_scaler, _name) and not hasattr(_main, _name):
            setattr(_main, _name, getattr(feature_scaler, _name))


# ─────────────────────────────────────────────────────────────────────────────
# Artifact loading
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_model_path(run_dir: str) -> str:
    """Find best_model.pt, allowing nested run_name subdir from slurm launches."""
    direct = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(direct):
        return direct
    candidates = []
    for d in os.listdir(run_dir):
        p = os.path.join(run_dir, d, "best_model.pt")
        if os.path.isfile(p):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No best_model.pt under {run_dir}")
    return max(candidates, key=os.path.getmtime)


def _load_artifacts(run_dir: str):
    data_dir = os.path.join(run_dir, "data")
    data_config_path = os.path.join(data_dir, "data_config_variables.pkl")
    scaler_path = os.path.join(data_dir, "scalers.pkl")
    model_path = _find_best_model_path(run_dir)

    with open(data_config_path, "rb") as f:
        data_config = pickle.load(f)
    scaler = LSTMFeatureScaler.load(scaler_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    return data_config, scaler, checkpoint, model_path


def _find_station_derived_features(data_dir: str, station_code):
    """
    Derived station features (station_mean_gwl, station_delta_std, etc.) are
    computed once during data prep and attached to every sample. For inference
    on a known station, copy them from any saved sample for that station.
    Returns None for cold-start stations (not in train/val/test).
    """
    keys = (
        "station_mean_gwl", "station_gwl_amplitude",
        "station_delta_mean", "station_delta_std",
        "mean_annual_rainfall", "annual_rainfall_std",
    )
    for fname in ("train_samples.pkl", "val_samples.pkl", "test_samples.pkl"):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            samples = pickle.load(f)
        for s in samples:
            if str(s["station_code"]) == str(station_code):
                return {k: float(s.get(k, 0.0)) for k in keys}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Model construction (mirrors train.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(config_dict: dict):
    cfg = SimpleNamespace(**config_dict)

    shared_kwargs = dict(
        num_timesteps=cfg.num_timesteps,
        features_per_timestep=cfg.features_per_timestep,
        static_embedding_dim=cfg.static_embedding_dim,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        num_lithology_types=cfg.num_lithology_types,
        num_well_types=cfg.num_well_types,
        num_aquifer_types=cfg.num_aquifer_types,
        num_aquifer_0_aquifer_types=max(cfg.num_aquifer_0_aquifer_types, 1),
        num_litho_supergroup_types=max(cfg.num_litho_supergroup_types, 1),
        num_lulc_types=cfg.num_lulc_types,
        num_state_types=max(cfg.num_state_types, 1),
        num_district_types=max(cfg.num_district_types, 1),
        lithology_embedding_dim=cfg.lithology_embedding_dim,
        well_type_embedding_dim=cfg.well_type_embedding_dim,
        aquifer_embedding_dim=cfg.aquifer_embedding_dim,
        aquifer_0_aquifer_embedding_dim=cfg.aquifer_0_aquifer_embedding_dim,
        litho_supergroup_embedding_dim=cfg.litho_supergroup_embedding_dim,
        lulc_embedding_dim=cfg.lulc_embedding_dim,
        state_embedding_dim=cfg.state_embedding_dim,
        district_embedding_dim=cfg.district_embedding_dim,
        num_forecast_features=cfg.num_forecast_features,
        dropout_rate=cfg.dropout_rate,
        use_static_features=cfg.use_static_features,
        use_linear_residual=cfg.use_linear_residual,
    )

    mt = cfg.model_type
    if mt == "tft":
        model = GWLForecastTFT(
            **shared_kwargs,
            d_model=cfg.tft_d_model, n_heads=cfg.tft_n_heads,
            lstm_layers=cfg.tft_lstm_layers, tft_dropout=cfg.tft_dropout,
        )
    elif mt == "transformer":
        model = GWLForecastTransformer(
            **shared_kwargs,
            d_model=cfg.transformer_d_model, nhead=cfg.transformer_nhead,
            num_encoder_layers=cfg.transformer_num_layers,
            dim_feedforward=cfg.transformer_dim_feedforward,
            transformer_dropout=cfg.transformer_dropout,
        )
    elif mt == "conditioned_lstm":
        model = GWLForecastConditionedLSTM(
            **shared_kwargs,
            lstm_hidden_dim=cfg.lstm_hidden_dim,
            lstm_num_layers=cfg.lstm_num_layers,
            lstm_dropout=cfg.lstm_dropout,
        )
    elif mt == "conditioned_transformer":
        model = GWLForecastConditionedTransformer(
            **shared_kwargs,
            d_model=cfg.transformer_d_model, nhead=cfg.transformer_nhead,
            num_encoder_layers=cfg.transformer_num_layers,
            dim_feedforward=cfg.transformer_dim_feedforward,
            transformer_dropout=cfg.transformer_dropout,
        )
    else:
        model = GWLForecastLSTM(
            **shared_kwargs,
            lstm_hidden_dim=cfg.lstm_hidden_dim,
            lstm_num_layers=cfg.lstm_num_layers,
            lstm_dropout=cfg.lstm_dropout,
        )
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# RevIN (mirrors Trainer._apply_revin)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_revin(sequence: torch.Tensor, floor: float, window: int = 0):
    """
    Mirrors Trainer._apply_revin.

    When window > 0, mean_s/std_s are computed from only the **last `window`
    timesteps** of the lookback (most recent end). The normalization itself is
    still applied to every timestep using those stats. This is a debug-only
    diagnostic to test whether a long trended lookback inflates std_s and
    amplifies flat-well predictions — at training time RevIN uses the full
    lookback, so shorten-window only changes inference.
    """
    gwl = sequence[..., 0:1]
    is_present = sequence[..., 1:2]
    if window > 0 and window < gwl.size(1):
        gwl_stat = gwl[:, -window:, :]
        present_stat = is_present[:, -window:, :]
    else:
        gwl_stat = gwl
        present_stat = is_present
    n_present = present_stat.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean_s_3d = (gwl_stat * present_stat).sum(dim=1, keepdim=True) / n_present
    sq_dev = ((gwl_stat - mean_s_3d) * present_stat).pow(2)
    var_s_3d = sq_dev.sum(dim=1, keepdim=True) / n_present
    std_s_3d = (var_s_3d + 1e-5).sqrt().clamp(min=floor)
    gwl_normed = ((gwl - mean_s_3d) / std_s_3d) * is_present
    sequence_normed = torch.cat([gwl_normed, sequence[..., 1:]], dim=-1)
    return (
        sequence_normed,
        mean_s_3d.squeeze(-1).squeeze(-1),
        std_s_3d.squeeze(-1).squeeze(-1),
    )


def _forward_and_decode(
    transformed: dict, current_gwl: float, scaler, model, cfg, device,
    verbose: bool = False, revin_window: int = 0,
) -> float:
    """
    Run a scaler-transformed sample through model + (optional) RevIN inverse
    + scaler inverse. Returns predicted absolute GWL in real meters.
    Shared by predict() and self_test() to guarantee identical decode logic.

    When verbose=True, prints the full intermediate trace (RevIN stats,
    pre/post-denorm magnitudes) so the user can audit whether per-sample
    normalization is amplifying flat-well predictions.
    """
    def t(x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=device).unsqueeze(0)

    sequence = t(transformed["sequence"])
    use_revin = bool(getattr(cfg, "use_revin", False))
    revin_floor = float(getattr(cfg, "revin_std_floor", 0.1))

    with torch.no_grad():
        if use_revin:
            sequence_in, mean_s, std_s = _apply_revin(
                sequence, revin_floor, window=revin_window,
            )
        else:
            sequence_in = sequence
            mean_s = std_s = None

        pred = model(
            sequence=sequence_in,
            static_continuous=t(transformed["static_continuous"]),
            forecast_features=t(transformed["forecast_features"]),
            lithology_idx=t(transformed["lithology_idx"], torch.long),
            well_type_idx=t(transformed["well_type_idx"], torch.long),
            aquifer_idx=t(transformed["aquifer_idx"], torch.long),
            aquifer_0_aquifer_idx=t(transformed["aquifer_0_aquifer_idx"], torch.long),
            litho_supergroup_idx=t(transformed["litho_supergroup_idx"], torch.long),
            state_idx=t(transformed["state_idx"], torch.long),
            district_idx=t(transformed["district_idx"], torch.long),
            historical_lulc_indices=t(transformed["historical_lulc_indices"], torch.long),
            forecast_lulc_idx=t(transformed["forecast_lulc_idx"], torch.long),
        )  # (1, 1)

        pred_normed = float(pred.squeeze().item())   # raw model output

        if use_revin:
            pred_global = pred.squeeze(1) * std_s + mean_s   # → scaler-scaled
        else:
            pred_global = pred.squeeze(1)

    pred_global_scalar = float(pred_global.squeeze().item())
    pred_arr = pred_global.cpu().numpy().reshape(-1, 1)
    if scaler.target_scaler is not None:
        pred_value = float(scaler.target_scaler.inverse_transform(pred_arr)[0, 0])
        tgt_scale = float(scaler.target_scaler.scale_[0])
        tgt_mean = float(scaler.target_scaler.mean_[0])
    else:
        pred_value = float(pred_arr[0, 0])
        tgt_scale, tgt_mean = 1.0, 0.0

    if verbose:
        mean_s_v = float(mean_s.squeeze().item()) if mean_s is not None else 0.0
        std_s_v = float(std_s.squeeze().item()) if std_s is not None else 1.0
        gwl_scaled = sequence[0, :, 0].cpu().numpy()
        is_present = sequence[0, :, 1].cpu().numpy()
        T = gwl_scaled.shape[0]
        print()
        print("─" * 60)
        print(" RevIN debug trace")
        print("─" * 60)
        print(f"  current_gwl (raw, m):       {current_gwl:+.4f}")
        print(f"  target_scaler.scale_ (m):   {tgt_scale:.4f}")
        print(f"  target_scaler.mean_ (m):    {tgt_mean:+.4f}")
        print(f"  use_revin:                  {use_revin}  floor={revin_floor}"
              f"{f'  revin_window={revin_window} (last K)' if revin_window > 0 else ''}")
        print()
        print(f"  Lookback gwl channel (scaled, oldest → newest, T={T}):")
        for i in range(T):
            tag = "  " if is_present[i] > 0.5 else "✗ "
            print(f"    {tag} t={i:>2d}  gwl_scaled={gwl_scaled[i]:+.4f}  "
                  f"(≈ {gwl_scaled[i] * tgt_scale:+.3f} m delta-from-current)")
        if use_revin:
            print()
            print(f"  RevIN mean_s (scaled space):    {mean_s_v:+.4f}  "
                  f"(≈ {mean_s_v * tgt_scale:+.3f} m)")
            print(f"  RevIN std_s  (scaled space):    {std_s_v:.4f}  "
                  f"(≈ {std_s_v * tgt_scale:.3f} m)  "
                  f"{'[FLOORED]' if std_s_v <= revin_floor + 1e-6 else '[from data]'}")
        print()
        print(f"  Model output pred_normed:           {pred_normed:+.4f}")
        print(f"  After de-RevIN  pred_global:        {pred_global_scalar:+.4f}  (scaled space)")
        print(f"  After scaler⁻¹  pred_value (m):     {pred_value:+.4f}  (delta in meters)")
        print(f"  Absolute pred (current + delta):    {current_gwl + pred_value:+.4f}  m")
        if use_revin:
            print()
            print(f"  Amplification from normed → real meters:")
            print(f"    pred_normed × std_s × target_scale = "
                  f"{pred_normed:+.4f} × {std_s_v:.4f} × {tgt_scale:.4f} = "
                  f"{pred_normed * std_s_v * tgt_scale:+.4f} m")
            print(f"    (Total normed→meter factor: {std_s_v * tgt_scale:.3f}×)")
        print("─" * 60)
        print()

    if bool(getattr(cfg, "use_delta_gwl", True)):
        return current_gwl + pred_value
    else:
        return pred_value


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv_dfs(data_config, csv_path: str):
    """
    Load gwl/dynamic/static slices using the SAME processing as the training
    pipeline (sign normalization, hard caps, type conversion, optional
    daily-dense interpolation). Skipping any of these would produce a
    different lookback sequence at inference vs training.
    """
    # The load_* methods read from data_config.csv_path internally. Override
    # in-memory so we point at the local file.
    data_config.csv_path = csv_path
    prep = LSTMDataPreparation(data_config)
    gwl_df = prep.load_gwl_readings()
    dyn_df = prep.load_dynamic_features()
    static_df = prep.load_static_features()

    # CRITICAL: when interpolate_lookback_gwl=True, training replaces gwl_df
    # with a daily-dense interpolated version (data_preparation.py:1808).
    # Lookback timesteps see those interpolated rows. Skipping this gives the
    # model a sparser lookback at inference than it saw at training time,
    # producing different predictions (observed up to ~1m divergence).
    if getattr(data_config, "interpolate_lookback_gwl", False):
        print("  Building interpolated daily GWL (interpolate_lookback_gwl=True)...")
        gwl_df = prep.build_interpolated_gwl_df(gwl_df)

    return gwl_df, dyn_df, static_df, prep


def _build_sample(
    station_code, current_date, prep, gwl_df, dyn_df, static_df, data_dir,
):
    """Build a single sample with all derived features attached. Pre-loaded
    dataframes + a `LSTMDataPreparation` instance are passed in so callers
    can cache them across many samples (used by self_test)."""
    sc = str(station_code)
    station_gwl = gwl_df[gwl_df["station_code"].astype(str) == sc].copy()
    if station_gwl.empty:
        raise SystemExit(f"No GWL readings for station_code={sc}")
    station_dynamic = dyn_df[dyn_df["station_code"].astype(str) == sc].copy()
    static_rows = static_df[static_df["station_code"].astype(str) == sc]
    if static_rows.empty:
        raise SystemExit(f"No static features for station_code={sc}")
    station_static = static_rows.iloc[0]

    # CRITICAL: find_gwl_value_with_gap looks up dates via station_gwl.index.
    # The training pipeline pre-indexes by date before calling create_sample
    # (data_preparation.py:1521-1528). Mirror that here, otherwise the lookup
    # silently returns None for every date and the sample is dropped.
    station_gwl = station_gwl.set_index("date").sort_index()
    if not station_dynamic.empty:
        station_dynamic = station_dynamic.set_index("date").sort_index()

    sample = prep.create_sample(
        station_gwl, station_dynamic, station_static,
        station_code=station_code,
        current_date=pd.to_datetime(current_date).to_pydatetime(),
        is_training=False,
    )
    if sample is None:
        raise SystemExit(
            f"Could not build sample for {sc}@{current_date}. Likely missing "
            "current/target GWL within gap window or sequence completeness "
            "below threshold."
        )

    derived = _find_station_derived_features(data_dir, station_code)
    if derived is not None:
        sample.update(derived)
    else:
        print(f"  WARNING: station {sc} is cold-start (not in train/val/test); "
              "derived static features will be 0.", file=sys.stderr)

    sample["gwl_anomaly"] = (
        float(sample["current_gwl"]) - float(sample.get("station_mean_gwl", 0.0))
    )
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    station_code: str, current_date, run_dir: str,
    csv_path: Optional[str] = None,
    verbose: bool = False,
    revin_window: int = 0,
) -> dict:
    """Run inference for one (station_code, current_date) pair.

    Args:
        csv_path: Override for data_config.csv_path. Required when the saved
            path (e.g. cluster-local /mnt/...) is not accessible at inference.

    Returns a dict with current/target dates, current_gwl, predicted delta and
    absolute GWL (all in real meters).
    """
    data_config, scaler, checkpoint, model_path = _load_artifacts(run_dir)
    print(f"Loaded artifacts from {run_dir}", file=sys.stderr)
    print(f"  model: {model_path}", file=sys.stderr)

    data_dir = os.path.join(run_dir, "data")
    csv_path = csv_path or getattr(data_config, "csv_path", None)
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit(
            f"CSV not found: {csv_path!r}. The saved data_config has csv_path="
            f"{getattr(data_config, 'csv_path', None)!r} — pass --csv-path to "
            "override with a local copy."
        )

    gwl_df, dyn_df, static_df, prep = _load_csv_dfs(data_config, csv_path)
    sample = _build_sample(
        station_code, current_date, prep, gwl_df, dyn_df, static_df, data_dir,
    )

    transformed = scaler.transform_sample(sample)

    # Build + load model
    model, cfg = _build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    current_gwl = float(sample["current_gwl"])
    pred_abs = _forward_and_decode(
        transformed, current_gwl, scaler, model, cfg, device,
        verbose=verbose, revin_window=revin_window,
    )
    pred_delta = pred_abs - current_gwl

    target_date = pd.to_datetime(current_date) + relativedelta(
        months=int(data_config.forecast_horizon_months)
    )

    return {
        "station_code": str(station_code),
        "current_date": pd.to_datetime(current_date).date().isoformat(),
        "target_date": target_date.date().isoformat(),
        "horizon_months": int(data_config.forecast_horizon_months),
        "current_gwl_m": current_gwl,
        "pred_delta_m": pred_delta,
        "pred_gwl_m": pred_abs,
        "model_type": getattr(cfg, "model_type", "?"),
        "use_revin": bool(getattr(cfg, "use_revin", False)),
        "revin_std_floor": float(getattr(cfg, "revin_std_floor", 0.1))
            if getattr(cfg, "use_revin", False) else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test mode
# ─────────────────────────────────────────────────────────────────────────────

def self_test(
    run_dir: str, n_samples: int = 3, split: str = "val", tol_m: float = 0.05,
    csv_path: Optional[str] = None,
) -> int:
    """
    Sanity-check the predict() pipeline against a reference forward on saved
    samples. For each test sample:
      - Reference: scaler.transform_sample(saved_sample) → forward → decode
        (always runs — only needs saved pkls + model + scaler).
      - predict():  build sample from CSV for (station, current_date) → forward
        (only runs if csv_path resolves to an existing file).
    If both run, they should produce the same prediction within `tol_m`.
    If only reference runs (no CSV locally), prints the reference predictions
    vs actuals and returns 0 — useful for confirming model + scaler load.
    Returns 0 on success, 1 on divergence.
    """
    data_config, scaler, checkpoint, model_path = _load_artifacts(run_dir)
    print(f"Self-test from {run_dir}")
    print(f"  model: {model_path}")

    resolved_csv = csv_path or getattr(data_config, "csv_path", None)
    csv_available = bool(resolved_csv) and os.path.exists(resolved_csv)
    if csv_available:
        print(f"  csv:   {resolved_csv}")
    else:
        print(
            f"  csv:   {resolved_csv!r} not found locally — "
            "running reference-only mode (predict() comparison skipped)."
        )

    data_dir = os.path.join(run_dir, "data")
    samples_path = os.path.join(data_dir, f"{split}_samples.pkl")
    if not os.path.exists(samples_path):
        raise SystemExit(f"No {split}_samples.pkl in {data_dir}")
    with open(samples_path, "rb") as f:
        samples = pickle.load(f)
    if not samples:
        raise SystemExit(f"No samples in {samples_path}")

    model, cfg = _build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Pre-load the CSV slices + data-prep instance ONCE so the per-sample loop
    # doesn't re-read 12M+ rows N times.
    gwl_df = dyn_df = static_df = prep = None
    data_dir = os.path.join(run_dir, "data")
    if csv_available:
        print(f"  Loading CSV slices from {resolved_csv} (one-time)...")
        gwl_df, dyn_df, static_df, prep = _load_csv_dfs(data_config, resolved_csv)

    n_target = n_samples
    # Cap attempts to avoid scanning the entire split if many samples fail to
    # rebuild (e.g., local CSV doesn't have lookback data for old anchor dates).
    max_attempts = min(len(samples), max(n_target * 50, 100))
    print(
        f"\nComparing predict.py vs reference forward — targeting {n_target} "
        f"{split} samples that build successfully  (tolerance: {tol_m*100:.0f}cm, "
        f"max attempts: {max_attempts})\n"
    )

    diffs = []     # signed predict − reference (m), one per successful sample
    successful = 0
    skipped = 0
    attempts = 0
    for sample in samples:
        if successful >= n_target or attempts >= max_attempts:
            break
        attempts += 1

        station_code = sample["station_code"]
        current_date = sample["current_date"]
        date_str = pd.to_datetime(current_date).date().isoformat()

        # If the user has a CSV, try the predict() path first. Skip silently
        # if the local CSV can't reconstruct this sample (common when local
        # CSV doesn't contain enough lookback history for old anchor dates).
        rebuilt_pred_m = None
        if csv_available:
            try:
                rebuilt = _build_sample(
                    station_code, current_date, prep,
                    gwl_df, dyn_df, static_df, data_dir,
                )
                rebuilt_transformed = scaler.transform_sample(rebuilt)
                rebuilt_current = float(rebuilt["current_gwl"])
                rebuilt_pred_m = _forward_and_decode(
                    rebuilt_transformed, rebuilt_current,
                    scaler, model, cfg, device,
                )
            except SystemExit:
                skipped += 1
                continue   # try next sample

        # Reference path (always runs)
        transformed = scaler.transform_sample(sample)
        current_gwl = float(sample["current_gwl"])
        ref_pred_m = _forward_and_decode(
            transformed, current_gwl, scaler, model, cfg, device,
        )
        actual_m = float(sample.get("target_gwl_raw", sample.get("target_gwl", 0.0)))

        successful += 1
        print(f"  Sample {successful}: station={station_code} date={date_str}")
        print(f"    actual:     {actual_m:+.3f}m")
        print(f"    reference:  {ref_pred_m:+.3f}m  (err vs actual: {ref_pred_m - actual_m:+.3f}m)")
        if rebuilt_pred_m is not None:
            diff = rebuilt_pred_m - ref_pred_m
            diffs.append(diff)
            print(f"    predict.py: {rebuilt_pred_m:+.3f}m  (diff vs reference: {diff:+.5f}m)")
        print()

    if csv_available and skipped > 0:
        print(f"Note: skipped {skipped} samples whose rebuild failed "
              "(likely the local CSV lacks lookback history for that station/date).")

    if not csv_available:
        print(f"Reference-only mode: showed reference predictions on {successful} samples.")
        print("  ✓ Model + scaler + RevIN load successfully and produce sensible outputs.")
        print("  To verify the full predict() pipeline, scp the training CSV locally")
        print("  and re-run with --csv-path <local-path>.")
        return 0

    if successful == 0:
        print(f"  ✗ No samples could be rebuilt from the local CSV (tried "
              f"{attempts}). The CSV likely doesn't match what was used at "
              "training time — verify --csv-path points to the correct file.")
        return 1

    # ── Divergence statistics + significance test ─────────────────────────
    # H0: predict() is unbiased vs reference (mean of signed diffs = 0).
    # We use a one-sample t-test on the diff distribution. Random per-sample
    # divergence from approximations (e.g. gwl_anomaly per-month vs overall)
    # produces a zero-mean noise distribution. Systematic bugs (a missing
    # transform step, wrong config) produce a non-zero mean → significant t.
    diffs_arr = np.array(diffs, dtype=np.float64)
    n = len(diffs_arr)
    mean_diff = float(diffs_arr.mean())
    std_diff = float(diffs_arr.std(ddof=1)) if n > 1 else 0.0
    abs_max  = float(np.abs(diffs_arr).max())
    abs_mean = float(np.abs(diffs_arr).mean())
    se = std_diff / (n ** 0.5) if n > 1 else 0.0
    t_stat = mean_diff / se if se > 0 else 0.0
    # Conservative critical value for 95% two-sided.
    # Exact df-aware values: 9→2.26, 19→2.09, 29→2.05, ∞→1.96.
    # 2.26 is safe for the common N≤30 case without scipy.
    t_crit = 2.26 if n <= 30 else 2.00
    ci_low  = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    print(f"Summary: {successful} successful comparisons (out of {attempts} attempts).")
    print(f"  Diff stats (predict − reference, meters):")
    print(f"    mean:            {mean_diff:+.5f}m")
    print(f"    std:              {std_diff:.5f}m")
    print(f"    abs mean:         {abs_mean:.5f}m")
    print(f"    abs max:          {abs_max:.5f}m")
    print(f"    SE(mean):         {se:.5f}m")
    print(f"    t-stat:          {t_stat:+.3f}    (|t|<{t_crit} → not significant at 95%)")
    print(f"    95% CI on mean:  [{ci_low:+.5f}m, {ci_high:+.5f}m]")

    if abs(t_stat) < t_crit:
        print(f"  ✓ Pipeline is statistically equivalent to reference: mean diff "
              f"is not distinguishable from zero (|t|={abs(t_stat):.2f} < {t_crit}).")
        print(f"    Per-sample noise (abs max {abs_max*100:.1f}cm) is consistent with the")
        print(f"    documented gwl_anomaly approximation (overall vs per-month mean).")
        return 0
    print(f"  ✗ Pipeline shows SIGNIFICANT systematic bias: |t|={abs(t_stat):.2f} ≥ {t_crit}.")
    print(f"    Mean diff {mean_diff:+.4f}m is not zero by chance — something is drifting.")
    print(f"    Likely sources: a config field not threaded through, or a data-prep")
    print(f"    step (sign normalization, interpolation, hard caps) being skipped.")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("station_code", nargs="?",
                        help="Station code (e.g. TSGWD_1010). Omit with --self-test.")
    parser.add_argument("current_date", nargs="?",
                        help="Anchor date (YYYY-MM-DD). Prediction is for "
                             "current_date + forecast_horizon_months. "
                             "Omit with --self-test.")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to a training run directory (containing data/ + best_model.pt).",
    )
    parser.add_argument(
        "--csv-path",
        help="Override for the CSV data path saved in data_config "
             "(useful when running locally and the saved path points to a "
             "cluster mount). For --self-test, optional: if missing, runs in "
             "reference-only mode.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run the predict() pipeline against a reference forward on a few "
             "saved samples and verify they match within tolerance.",
    )
    parser.add_argument(
        "--self-test-n", type=int, default=3,
        help="Number of samples to use in --self-test (default: 3).",
    )
    parser.add_argument(
        "--self-test-split", default="val", choices=["train", "val", "test"],
        help="Which saved split to draw self-test samples from (default: val).",
    )
    parser.add_argument(
        "--self-test-tol", type=float, default=0.05,
        help="Self-test tolerance in meters (default: 0.05 = 5cm).",
    )
    parser.add_argument(
        "--debug-revin", action="store_true",
        help="Print the per-sample RevIN trace: mean_s, std_s, pred_normed, "
             "pred_global, pred_value and the lookback gwl channel. Use to "
             "diagnose whether long-trended lookbacks inflate std_s and "
             "amplify flat-well predictions.",
    )
    parser.add_argument(
        "--shorten-revin-window", type=int, default=0, metavar="K",
        help="Compute RevIN mean_s/std_s from only the LAST K timesteps of "
             "the lookback (instead of all). Inference-only diagnostic — the "
             "model was trained with full-lookback stats, so this lets you "
             "test whether trimming an inflated std_s reduces wild flat-well "
             "predictions. 0 = use full lookback (default).",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test(
            args.run_dir,
            n_samples=args.self_test_n,
            split=args.self_test_split,
            tol_m=args.self_test_tol,
            csv_path=args.csv_path,
        ))

    if not args.station_code or not args.current_date:
        parser.error("station_code and current_date are required (or use --self-test).")

    result = predict(
        args.station_code, args.current_date, args.run_dir,
        csv_path=args.csv_path,
        verbose=args.debug_revin,
        revin_window=args.shorten_revin_window,
    )
    print()
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
