"""
GWL diagnostic plotting + metrics aggregation.

Extracted from train.py as part of a multi-session refactor splitting
train.py into focused modules. Behavior is unchanged from the original
methods — they are simply relocated into a `PlotGenerator` class that
holds the directory layout previously stored on Trainer.

Trainer now instantiates one `PlotGenerator` and calls into it for all
diagnostic output (CSV-aware metrics blocks, per-well plots, etc.).
"""

import os
from collections import defaultdict
from datetime import date as _date
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.amp import autocast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score


class PlotGenerator:
    """Container for directory-aware plot helpers (paths + plotting + metrics).

    All `_plot_*` helpers and `generate_diagnostic_plots` were originally
    methods on Trainer. They are moved here unchanged in body — they still
    refer to `self._plot_path`, `self._plot_dirs`, etc. (now on this class).

    Args:
        plot_dirs: Mapping of category names ("training", "gwl_val", etc.) to
            directories. Same dict Trainer used.
        run_dir: Run output directory (used for fallback paths).
        perf_dir: Directory for performance CSVs.
        inverse_transform_fn: Trainer._inverse_transform_on_device.
        compute_train_seasonal_means_fn: Trainer._compute_train_seasonal_means.
        use_amp: Whether to use autocast in inference loops.
        device: e.g. "cuda" or "cpu".
        use_delta_gwl: Whether the model predicts deltas.
        model: PyTorch model (eval mode used during plotting).
    """

    def __init__(
        self,
        plot_dirs: Dict[str, str],
        run_dir: str,
        perf_dir: str,
        inverse_transform_fn,
        compute_train_seasonal_means_fn,
        use_amp: bool,
        device: str,
        use_delta_gwl: bool,
        model,
        min_station_target_std: float = 0.0,
        include_states: List[str] = None,
        min_station_mode_gap_days: float = 0.0,
        station_to_train_std: Dict[str, float] = None,
        station_to_mode_gap_days: Dict[str, float] = None,
        train_data=None,
        val_data=None,
        use_revin: bool = False,
        revin_std_floor: float = 0.1,
        pred_clamp_pct: float = 0.0,
    ):
        self._plot_dirs = plot_dirs
        self.run_dir = run_dir
        self._perf_dir = perf_dir
        self._inverse_transform_on_device = inverse_transform_fn
        self._compute_train_seasonal_means = compute_train_seasonal_means_fn
        self.use_amp = use_amp
        self.model = model
        # Mimic Trainer.config.device access used inside generate_diagnostic_plots
        self.config = type("Cfg", (), {"device": device})()
        self.use_delta_gwl = use_delta_gwl
        # Single std threshold (data-prep flat filter AND eval predictable cohort).
        # When 0.0: no flat filter, no predictable metric block.
        self._min_station_target_std = float(min_station_target_std)
        # Optional state whitelist; composes with std filter by AND.
        # When empty/None: no state filter.
        self._include_states = set(include_states) if include_states else set()
        # Optional minimum raw-observation mode-gap (days). Composes with
        # std + state filters by AND. 0 = disabled.
        self._min_station_mode_gap_days = float(min_station_mode_gap_days)
        # Canonical per-station train_std map (data prep computes from PRE-thinning,
        # PRE-filter train samples). Stations missing from the map → Scenario B
        # cold-start: tagged [no train-std] in plots, excluded from predictable.
        self._station_to_train_std = station_to_train_std or {}
        # Canonical per-station raw-observation mode-gap map (days between
        # consecutive readings, mode). Used by the mode-gap predictable filter.
        self._station_to_mode_gap_days = station_to_mode_gap_days or {}
        # Optional: source datasets for historical context overlays in best/worst plots.
        # Train history goes behind val plots; train+val history goes behind test plots.
        self._train_data = train_data
        self._val_data = val_data
        # Cache for per-well history (computed once per dataset on first request)
        self._history_cache = {}
        # When true: apply per-sample InstanceNorm to GWL channel of input
        # sequence before model.forward, then de-normalize model output before
        # inverse_transform. Mirrors Trainer._apply_revin so plot/metric paths
        # match the train-time forward semantics.
        self._use_revin = use_revin
        self._revin_std_floor = revin_std_floor
        # Inference-time prediction clamp: |pred - current| <= pct * |current|.
        # 0.0 disables. See Trainer._effective_clamp_pct for default policy.
        self._pred_clamp_pct = float(pred_clamp_pct or 0.0)

    def _clamp_pred_to_current(self, pred_abs, pred_delta, current_gwl_raw):
        """Same semantics as Trainer._clamp_pred_to_current; mirrored here so
        the plot-side inference paths produce predictions consistent with
        validate() and saved test predictions."""
        if self._pred_clamp_pct <= 0.0:
            return pred_abs, pred_delta
        max_dev = self._pred_clamp_pct * torch.abs(current_gwl_raw)
        pred_delta_c = torch.clamp(pred_delta, min=-max_dev, max=max_dev)
        pred_abs_c = current_gwl_raw + pred_delta_c
        return pred_abs_c, pred_delta_c

    def _apply_revin(self, sequence: "torch.Tensor"):
        """Same logic as Trainer._apply_revin — see train.py for docstring."""
        gwl = sequence[..., 0:1]
        is_present = sequence[..., 1:2]
        n_present = is_present.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_s_3d = (gwl * is_present).sum(dim=1, keepdim=True) / n_present
        sq_dev = ((gwl - mean_s_3d) * is_present).pow(2)
        var_s_3d = sq_dev.sum(dim=1, keepdim=True) / n_present
        std_s_3d = (var_s_3d + 1e-5).sqrt().clamp(min=self._revin_std_floor)
        gwl_normed = ((gwl - mean_s_3d) / std_s_3d) * is_present
        sequence_normed = torch.cat([gwl_normed, sequence[..., 1:]], dim=-1)
        return sequence_normed, mean_s_3d.squeeze(-1).squeeze(-1), std_s_3d.squeeze(-1).squeeze(-1)

    # =====================  Path helper  =====================
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


    # =====================  generate_diagnostic_plots  =====================
    @torch.no_grad()
    def generate_diagnostic_plots(self, data=None, sampler=None, split_name="val", n_wells_per_category: int = 5):
        """
        Generate prediction diagnostic plots on a dataset split.

        Called once after training with best model loaded. Runs a fresh inference
        pass to collect per-sample predictions with well IDs and dates.

        Args:
            data: LSTMTensorDataset to plot. Defaults to self.val_data.
            sampler: BatchSampler for the dataset. Defaults to self.val_sampler.
            split_name: Name of the split for reference (e.g. "val", "test").

        Generates 4 plots:
        1. Predicted vs Actual scatter (absolute GWL)
        2. Delta prediction quadrants (pred_delta vs target_delta)
        3. Error distribution histogram (signed errors in meters)
        4. Per-well time series: best / median / worst wells by RMSE
        """
        if data is None or sampler is None:
            data = self.val_data
            sampler = self.val_sampler
            split_name = "val"

        self.model.eval()
        amp_device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"

        # If the model has a linear residual head, also collect its standalone
        # output so plots can show "linear-only prediction" as a third line.
        has_linear_head = (
            getattr(self.model, "linear_residual_head", None) is not None
        )

        all_preds_abs = []
        all_targets_abs = []
        all_pred_deltas = []
        all_target_deltas = []
        all_well_ids = []
        all_date_ordinals = []
        all_linear_pred_deltas = [] if has_linear_head else None
        all_linear_pred_abs = [] if has_linear_head else None

        for batch_indices in sampler:
            batch = data.get_batch(batch_indices)

            with autocast(device_type=amp_device_type, enabled=self.use_amp):
                if self._use_revin:
                    sequence_in, mean_s, std_s = self._apply_revin(batch["sequence"])
                else:
                    sequence_in = batch["sequence"]
                    mean_s = std_s = None

                _kw = dict(
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
                if getattr(self.model, "use_prithvi", False):
                    _kw["tile_idx"] = batch["tile_idx"]
                gwl_pred = self.model(**_kw)
                if self._use_revin:
                    # De-normalize back to global-scaled space before inverse_transform.
                    gwl_pred = (gwl_pred.squeeze(1) * std_s + mean_s).unsqueeze(1)
                if has_linear_head:
                    linear_pred = self.model.linear_residual_head(
                        sequence=batch["sequence"],
                        static_continuous=batch["static_continuous"],
                        forecast_features=batch["forecast_features"],
                    )

            current_gwl_raw = batch["current_gwl_raw"]

            if self.use_delta_gwl:
                pred_delta = self._inverse_transform_on_device(
                    gwl_pred.squeeze(1).float()
                )
                pred_abs = current_gwl_raw + pred_delta
                target_abs = batch["target_gwl_raw"]
                target_delta = self._inverse_transform_on_device(
                    batch["target_gwl"].float()
                )
            else:
                pred_abs = self._inverse_transform_on_device(
                    gwl_pred.squeeze(1).float()
                )
                target_abs = self._inverse_transform_on_device(
                    batch["target_gwl"].float()
                )
                pred_delta = pred_abs - current_gwl_raw
                target_delta = target_abs - current_gwl_raw

            pred_abs, pred_delta = self._clamp_pred_to_current(
                pred_abs, pred_delta, current_gwl_raw
            )

            all_preds_abs.append(pred_abs.cpu().numpy())
            all_targets_abs.append(target_abs.cpu().numpy())
            all_pred_deltas.append(pred_delta.cpu().numpy())
            all_target_deltas.append(target_delta.cpu().numpy())
            all_well_ids.append(batch["well_id"].cpu().numpy())
            all_date_ordinals.append(batch["target_date_ordinal"].cpu().numpy())

            # Linear-only path predictions (same scaling treatment as full pred)
            if has_linear_head:
                if self.use_delta_gwl:
                    lin_delta = self._inverse_transform_on_device(
                        linear_pred.squeeze(1).float()
                    )
                    lin_abs = current_gwl_raw + lin_delta
                else:
                    lin_abs = self._inverse_transform_on_device(
                        linear_pred.squeeze(1).float()
                    )
                    lin_delta = lin_abs - current_gwl_raw
                all_linear_pred_deltas.append(lin_delta.cpu().numpy())
                all_linear_pred_abs.append(lin_abs.cpu().numpy())

        # Concatenate all
        preds_abs = np.concatenate(all_preds_abs)
        targets_abs = np.concatenate(all_targets_abs)
        pred_deltas = np.concatenate(all_pred_deltas)
        target_deltas = np.concatenate(all_target_deltas)
        well_ids = np.concatenate(all_well_ids)
        date_ordinals = np.concatenate(all_date_ordinals)
        if has_linear_head:
            linear_pred_deltas = np.concatenate(all_linear_pred_deltas)
            linear_pred_abs = np.concatenate(all_linear_pred_abs)
        else:
            linear_pred_deltas = None
            linear_pred_abs = None

        idx_to_station = data.idx_to_station
        station_to_state = getattr(data, "station_to_state", {}) or {}

        # For test split: split wells into predictable vs flat using the
        # canonical train_std map AND optional state whitelist (orthogonal,
        # AND-composed). Block 2 prints whenever ≥1 filter is active.
        # When BOTH filters are off (threshold=0 AND empty include_states):
        # everyone is "predictable" → only Block 1 printed.
        threshold = self._min_station_target_std
        states_set = self._include_states  # already a set
        min_mode_gap = self._min_station_mode_gap_days
        any_filter_active = (threshold > 0) or bool(states_set) or (min_mode_gap > 0)

        # Build a short label describing which filters are in effect
        def _filter_label():
            parts = []
            if threshold > 0:
                parts.append(f"train_std≥{threshold:.1f}m")
            if states_set:
                # cap to first 3 names for readability
                names = sorted(states_set)
                shown = ", ".join(names[:3]) + (f", +{len(names) - 3} more" if len(names) > 3 else "")
                parts.append(f"state ∈ {{{shown}}}")
            if min_mode_gap > 0:
                parts.append(f"mode_gap≥{min_mode_gap:g}d")
            return " + ".join(parts) if parts else "no filter"

        filter_info = None
        if split_name == "test":
            filter_info = self._apply_test_cap_and_filter(
                preds_abs, targets_abs,
                pred_deltas, target_deltas,
                well_ids, date_ordinals,
                idx_to_station,
                station_to_state=station_to_state,
                linear_pred_abs=linear_pred_abs,
                linear_pred_deltas=linear_pred_deltas,
            )

            # Counts for transparency
            print(f"\n  Test set split (filters: {_filter_label()}):")
            print(f"    Total:                       {filter_info['n_samples_raw']} samples, "
                  f"{filter_info['n_wells_raw']} wells")
            if any_filter_active:
                print(f"    Predictable (passes all filters): "
                      f"{len(filter_info['filtered_preds'])} samples, "
                      f"{filter_info['n_wells_predictable']} wells")
                print(f"    Flat (fails ≥1 filter + cold-start): "
                      f"{filter_info['n_wells_flat']} wells "
                      f"(of which {filter_info['n_wells_cold_start']} are cold-start "
                      f"with no train_std)")

            # Block 1: capped — cold-start view (all wells)
            self._print_metrics_block(
                filter_info['capped_preds'],
                filter_info['capped_targets'],
                filter_info['capped_pred_deltas'],
                filter_info['capped_target_deltas'],
                filter_info['capped_well_ids'],
                label="Test (all wells — cold-start view, includes flat)",
                idx_to_station=idx_to_station,
            )

            # Block 2: predictable cohort — when ≥1 filter is active
            if any_filter_active:
                self._print_metrics_block(
                    filter_info['filtered_preds'],
                    filter_info['filtered_targets'],
                    filter_info['filtered_pred_deltas'],
                    filter_info['filtered_target_deltas'],
                    filter_info['filtered_well_ids'],
                    label=f"Test predictable ({_filter_label()})",
                    idx_to_station=idx_to_station,
                )

            # Plots use the full test set (all wells) so flat wells show in worst panels.
            plot_preds = filter_info['capped_preds']
            plot_targets = filter_info['capped_targets']
            plot_pred_d = filter_info['capped_pred_deltas']
            plot_target_d = filter_info['capped_target_deltas']
            plot_well_ids = filter_info['capped_well_ids']
            plot_dates = filter_info['capped_dates']
            plot_linear_preds = filter_info.get('capped_linear_preds')
        else:
            # Val: single metrics block. Val is already station_std + state filtered
            # (data-prep) and capped (≤MAX_SAMPLES_PER_STATION_EVAL/well at data prep).
            label = (
                f"{split_name.capitalize()} (capped + filtered: {_filter_label()} from data prep)"
                if any_filter_active
                else f"{split_name.capitalize()} (capped, no filter)"
            )
            self._print_metrics_block(
                preds_abs, targets_abs, pred_deltas, target_deltas, well_ids,
                label=label,
                idx_to_station=idx_to_station,
            )

            # Val plots use the same data (already filtered + capped from data prep)
            plot_preds = preds_abs
            plot_targets = targets_abs
            plot_pred_d = pred_deltas
            plot_target_d = target_deltas
            plot_well_ids = well_ids
            plot_dates = date_ordinals
            plot_linear_preds = linear_pred_abs

        # ── Plot 1: Predicted vs Actual scatter ────────────────────────────
        self._plot_scatter_pred_vs_actual(plot_preds, plot_targets, split_name)

        # ── Plot 2: Delta quadrants ────────────────────────────────────────
        self._plot_delta_quadrants(plot_pred_d, plot_target_d, split_name)

        # ── Plot 3: Error distribution ─────────────────────────────────────
        self._plot_error_distribution(plot_preds, plot_targets, split_name)

        # ── Plot 4: Per-well time series (best / median / worst) ───────────
        self._plot_per_well_predictions(
            plot_preds,
            plot_targets,
            plot_well_ids,
            plot_dates,
            idx_to_station,
            n_per_category=n_wells_per_category,
            split_name=split_name,
        )

        # ── Plot 5: Per-well R² distribution (absolute) ───────────────────
        self._plot_r2_distribution(
            plot_preds, plot_targets, plot_well_ids, split_name=split_name,
        )

        # ── Plot 6: Per-well R² distribution (delta) ─────────────────────
        self._plot_r2_distribution(
            plot_pred_d, plot_target_d, plot_well_ids, split_name=split_name,
            filename_suffix="_delta",
            title_suffix=" (Delta)",
        )

        # ── Plot 7: Flat wells (test only, wells excluded by predictable filter) ──
        if split_name == "test" and filter_info is not None and filter_info['n_wells_flat'] > 0:
            self._plot_flat_wells(
                filter_info['flat_preds'],
                filter_info['flat_targets'],
                filter_info['flat_well_ids'],
                filter_info['flat_dates'],
                idx_to_station,
                split_name=split_name,
            )

        # ── Plot 8: Per-well predictions stratified by GWL level ─────────
        self._plot_per_well_by_gwl_level(
            plot_preds, plot_targets, plot_well_ids, plot_dates,
            idx_to_station, split_name=split_name,
        )

        # ── Plot 9: Delta per-well time series (best / most-sampled / worst) ──
        self._plot_per_well_predictions(
            plot_pred_d, plot_target_d, plot_well_ids, plot_dates,
            idx_to_station, n_per_category=n_wells_per_category,
            split_name=split_name,
            filename_suffix="_delta", title_suffix=" (Delta)",
            ylabel="Delta GWL (m)",
        )

        # ── Plot 10: Delta per-well by GWL level (same strata as absolute) ──
        self._plot_per_well_by_gwl_level(
            plot_pred_d, plot_target_d, plot_well_ids, plot_dates,
            idx_to_station, split_name=split_name,
            filename_suffix="_delta", title_suffix=" (Delta, by depth)",
            ylabel="Delta GWL (m)",
            stratify_targets=plot_targets,
        )

        # ── Plot 11: Delta per-well by delta volatility ──────────────────
        self._plot_per_well_by_gwl_level(
            plot_pred_d, plot_target_d, plot_well_ids, plot_dates,
            idx_to_station, split_name=split_name,
            filename_suffix="_by_volatility", title_suffix=" (by volatility)",
            ylabel="Delta GWL (m)",
            stratify_mode="mean_abs",
        )

        # ── Plot 12: R² vs depth proxy (2×2 grid) ─────────────────────────
        self._plot_r2_vs_depth(
            plot_preds, plot_targets, plot_pred_d, plot_target_d,
            plot_well_ids, idx_to_station, split_name=split_name,
        )

        # ── Plot 13: Delta residual analysis (scatter + residual) ─────────
        self._plot_delta_residuals(
            plot_pred_d, plot_target_d, split_name=split_name,
        )

        # ── Plot 14: R² delta by well volatility ─────────────────────────
        self._plot_r2_delta_by_volatility(
            plot_pred_d, plot_target_d, plot_well_ids, idx_to_station,
            split_name=split_name,
        )

        # ── Plot 15: Best/Worst wells time series (by R² delta) ─────────
        self._plot_best_worst_timeseries(
            plot_pred_d, plot_target_d,
            plot_preds, plot_targets,
            plot_well_ids, plot_dates,
            idx_to_station, n_per_category=5,
            split_name=split_name,
            linear_preds_abs=plot_linear_preds,
        )

        # ── Plot 15b/c: Best/Worst per std bucket (one well per bucket) ─
        for _mode in ("best", "worst"):
            self._plot_best_or_worst_by_std_bucket(
                plot_pred_d, plot_target_d,
                plot_preds, plot_targets,
                plot_well_ids, plot_dates,
                idx_to_station, mode=_mode,
                split_name=split_name,
                linear_preds_abs=plot_linear_preds,
            )

        # ── Plot 16: Pred vs actual delta scatter by R² quartile ────────
        self._plot_scatter_by_r2_quartile(
            plot_pred_d, plot_target_d, plot_well_ids,
            idx_to_station, split_name=split_name,
        )

        # ── Plot 16b: Pred vs actual delta scatter by std bucket ────────
        self._plot_scatter_by_std_bucket(
            plot_pred_d, plot_target_d, plot_well_ids,
            idx_to_station, split_name=split_name,
        )

        # ── Plot 17: R² delta by well attribute (state/type/aquifer/vol) ─
        station_to_state = getattr(data, "station_to_state", {})
        station_to_well_type = getattr(data, "station_to_well_type", {})
        station_to_aquifer_type = getattr(data, "station_to_aquifer_type", {})
        self._plot_r2_delta_by_attribute(
            plot_pred_d, plot_target_d, plot_well_ids,
            idx_to_station,
            station_to_state, station_to_well_type, station_to_aquifer_type,
            split_name=split_name,
        )

        # ── Plot 18: Model vs simple baselines (per-well) ────────────────
        # Seasonal baseline uses TRAIN-period statistics only (no leakage).
        train_seasonal_means = self._compute_train_seasonal_means()
        self._plot_vs_baselines(
            plot_pred_d, plot_target_d, plot_well_ids, plot_dates,
            split_name=split_name,
            train_seasonal_means=train_seasonal_means,
        )

        # ── Plot 19: Direction accuracy by |delta| magnitude ─────────────
        self._plot_direction_by_magnitude(
            plot_pred_d, plot_target_d, split_name=split_name,
        )

        # ── Plot 20: Rolling R² over time ────────────────────────────────
        self._plot_rolling_r2_over_time(
            plot_pred_d, plot_target_d, plot_dates,
            split_name=split_name,
        )

        # (Metrics already printed above for both val and test)

        print(f"Diagnostic plots saved to: {self.run_dir}/plots/")


    # =====================  _print_metrics_block  =====================
    def _print_metrics_block(
        self, preds, targets, pred_d, target_d, well_ids, label: str,
        idx_to_station: Dict[int, str] = None,
    ):
        """Print a block of metrics (pooled + per-well, absolute + delta) with a label.

        Volatility breakdown uses canonical train_std (cold-start wells get
        eval-sample fallback so they still appear in some bucket).
        """
        from sklearn.metrics import r2_score
        min_well_samples = 5

        errors = preds - targets
        rmse = np.sqrt((errors ** 2).mean()) if len(errors) else 0.0
        mae = np.abs(errors).mean() if len(errors) else 0.0
        pooled_r2_abs = r2_score(targets, preds) if len(targets) > 1 else 0.0
        pooled_r2_delta = r2_score(target_d, pred_d) if len(target_d) > 1 else 0.0
        # 50/50 model+persistence blend: pred_blend = current + pred_delta/2 = preds - pred_d/2
        preds_blend = preds - pred_d / 2.0
        pooled_r2_blend = r2_score(targets, preds_blend) if len(targets) > 1 else 0.0
        # Persistence prediction: pred = current_gwl = targets - target_d
        preds_persist = targets - target_d
        pooled_r2_persist = r2_score(targets, preds_persist) if len(targets) > 1 else 0.0

        # Per-sample MAPE inputs (avoid div-by-zero with eps; floor = max(|t|,1))
        abs_t = np.abs(targets)
        abs_t_floor = np.maximum(abs_t, 1.0)
        abs_err_pred = np.abs(preds - targets)
        abs_err_blend = np.abs(preds_blend - targets)
        abs_err_persist = np.abs(preds_persist - targets)

        per_well_r2_abs = []
        per_well_r2_delta = []
        per_well_r2_blend = []
        per_well_r2_persist = []
        per_well_std_delta = []
        # MAPE accumulators (per-well mean of per-sample MAPE)
        pw_mape_pred,    pw_mape_blend,    pw_mape_persist    = [], [], []
        pw_mapef_pred,   pw_mapef_blend,   pw_mapef_persist   = [], [], []
        for wid in np.unique(well_ids):
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            per_well_r2_abs.append(r2_score(targets[m], preds[m]))
            per_well_r2_delta.append(r2_score(target_d[m], pred_d[m]))
            per_well_r2_blend.append(r2_score(targets[m], preds_blend[m]))
            per_well_r2_persist.append(r2_score(targets[m], preds_persist[m]))
            pw_mape_pred.append((abs_err_pred[m]    / (abs_t[m] + 1e-8)).mean())
            pw_mape_blend.append((abs_err_blend[m]  / (abs_t[m] + 1e-8)).mean())
            pw_mape_persist.append((abs_err_persist[m] / (abs_t[m] + 1e-8)).mean())
            pw_mapef_pred.append((abs_err_pred[m]    / abs_t_floor[m]).mean())
            pw_mapef_blend.append((abs_err_blend[m]  / abs_t_floor[m]).mean())
            pw_mapef_persist.append((abs_err_persist[m] / abs_t_floor[m]).mean())
            # Canonical train_std for volatility bucketing; cold-start fallback
            # to eval-sample std so the well still lands in a bucket.
            train_std = None
            if idx_to_station is not None:
                station_code = idx_to_station.get(int(wid))
                train_std, source = self._get_train_std(station_code)
                if source == "eval":
                    train_std = float(target_d[m].std())
            else:
                train_std = float(target_d[m].std())
            per_well_std_delta.append(train_std)
        median_r2_abs = np.median(per_well_r2_abs) if per_well_r2_abs else 0.0
        median_r2_delta = np.median(per_well_r2_delta) if per_well_r2_delta else 0.0
        median_r2_blend = np.median(per_well_r2_blend) if per_well_r2_blend else 0.0
        median_r2_persist = np.median(per_well_r2_persist) if per_well_r2_persist else 0.0
        # Mean is clipped to [-2, 2] to avoid catastrophic outliers swamping the avg.
        mean_r2_abs = np.clip(per_well_r2_abs, -2.0, 2.0).mean() if per_well_r2_abs else 0.0
        mean_r2_delta = np.clip(per_well_r2_delta, -2.0, 2.0).mean() if per_well_r2_delta else 0.0
        mean_r2_blend = np.clip(per_well_r2_blend, -2.0, 2.0).mean() if per_well_r2_blend else 0.0
        mean_r2_persist = np.clip(per_well_r2_persist, -2.0, 2.0).mean() if per_well_r2_persist else 0.0
        pct_pos_abs = (np.array(per_well_r2_abs) > 0).mean() * 100 if per_well_r2_abs else 0.0
        pct_pos_delta = (np.array(per_well_r2_delta) > 0).mean() * 100 if per_well_r2_delta else 0.0
        pct_pos_blend = (np.array(per_well_r2_blend) > 0).mean() * 100 if per_well_r2_blend else 0.0
        pct_pos_persist = (np.array(per_well_r2_persist) > 0).mean() * 100 if per_well_r2_persist else 0.0

        def _q25_q75(lst):
            if not lst:
                return 0.0, 0.0
            a = np.array(lst)
            return float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))
        q25_abs, q75_abs         = _q25_q75(per_well_r2_abs)
        q25_delta, q75_delta     = _q25_q75(per_well_r2_delta)
        q25_blend, q75_blend     = _q25_q75(per_well_r2_blend)
        q25_persist, q75_persist = _q25_q75(per_well_r2_persist)

        print(f"\n  ── {label} metrics ──")
        print(f"    RMSE: {rmse:.3f}m  MAE: {mae:.3f}m")
        print(f"    Pooled R² (abs):   {pooled_r2_abs:.4f}   Pooled R² (delta):   {pooled_r2_delta:.4f}   Pooled R² (blend): {pooled_r2_blend:.4f}   Pooled R² (persist): {pooled_r2_persist:.4f}")
        print(f"    Per-well R² (abs):     q25={q25_abs:.4f}  median={median_r2_abs:.4f}  q75={q75_abs:.4f}  mean={mean_r2_abs:.4f}  ({pct_pos_abs:.0f}% > 0)")
        print(f"    Per-well R² (delta):   q25={q25_delta:.4f}  median={median_r2_delta:.4f}  q75={q75_delta:.4f}  mean={mean_r2_delta:.4f}  ({pct_pos_delta:.0f}% > 0)   ({len(per_well_r2_abs)} wells)")
        print(f"    Per-well R² (blend):   q25={q25_blend:.4f}  median={median_r2_blend:.4f}  q75={q75_blend:.4f}  mean={mean_r2_blend:.4f}  ({pct_pos_blend:.0f}% > 0)")
        print(f"    Per-well R² (persist): q25={q25_persist:.4f}  median={median_r2_persist:.4f}  q75={q75_persist:.4f}  mean={mean_r2_persist:.4f}  ({pct_pos_persist:.0f}% > 0)")

        def _mm(lst, clip=5.0):
            if not lst:
                return 0.0, 0.0
            arr = np.array(lst)
            return float(np.median(arr)), float(np.clip(arr, 0.0, clip).mean())
        m_med, m_mean       = _mm(pw_mape_pred)
        b_med, b_mean       = _mm(pw_mape_blend)
        p_med, p_mean       = _mm(pw_mape_persist)
        mf_med, mf_mean     = _mm(pw_mapef_pred)
        bf_med, bf_mean     = _mm(pw_mapef_blend)
        pf_med, pf_mean     = _mm(pw_mapef_persist)
        m_q25, m_q75   = _q25_q75(pw_mape_pred)
        b_q25, b_q75   = _q25_q75(pw_mape_blend)
        p_q25, p_q75   = _q25_q75(pw_mape_persist)
        mf_q25, mf_q75 = _q25_q75(pw_mapef_pred)
        bf_q25, bf_q75 = _q25_q75(pw_mapef_blend)
        pf_q25, pf_q75 = _q25_q75(pw_mapef_persist)
        print(f"    Per-well MAPE % (|t|):     pred q25/q50/q75={m_q25*100:.1f}/{m_med*100:.1f}/{m_q75*100:.1f} mean={m_mean*100:.1f}  blend={b_q25*100:.1f}/{b_med*100:.1f}/{b_q75*100:.1f} mean={b_mean*100:.1f}  persist={p_q25*100:.1f}/{p_med*100:.1f}/{p_q75*100:.1f} mean={p_mean*100:.1f}")
        print(f"    Per-well MAPE-floor % (1m): pred q25/q50/q75={mf_q25*100:.1f}/{mf_med*100:.1f}/{mf_q75*100:.1f} mean={mf_mean*100:.1f}  blend={bf_q25*100:.1f}/{bf_med*100:.1f}/{bf_q75*100:.1f} mean={bf_mean*100:.1f}  persist={pf_q25*100:.1f}/{pf_med*100:.1f}/{pf_q75*100:.1f} mean={pf_mean*100:.1f}")

        # R² delta + blend bucketed by GWL delta volatility
        if per_well_std_delta:
            arr_std = np.array(per_well_std_delta)
            arr_r2d = np.array(per_well_r2_delta)
            arr_r2b = np.array(per_well_r2_blend)
            buckets = [(0, 1.0), (1.0, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, np.inf)]
            print(f"    R² delta by volatility (std of target delta):")
            for lo, hi in buckets:
                mask = (arr_std >= lo) & (arr_std < hi)
                n = mask.sum()
                if n > 0:
                    med = np.median(arr_r2d[mask])
                    pct = (arr_r2d[mask] > 0).mean() * 100
                    hi_str = f"{hi:.0f}" if hi != np.inf else "+"
                    print(f"      std {lo:.0f}-{hi_str}m: median R² delta={med:.4f} ({pct:.0f}% > 0, {n} wells)")
            print(f"    R² blend by volatility (std of target delta):")
            for lo, hi in buckets:
                mask = (arr_std >= lo) & (arr_std < hi)
                n = mask.sum()
                if n > 0:
                    med = np.median(arr_r2b[mask])
                    pct = (arr_r2b[mask] > 0).mean() * 100
                    hi_str = f"{hi:.0f}" if hi != np.inf else "+"
                    print(f"      std {lo:.0f}-{hi_str}m: median R² blend={med:.4f} ({pct:.0f}% > 0, {n} wells)")
            arr_r2p = np.array(per_well_r2_persist)
            print(f"    R² persist by volatility (std of target delta):")
            for lo, hi in buckets:
                mask = (arr_std >= lo) & (arr_std < hi)
                n = mask.sum()
                if n > 0:
                    med = np.median(arr_r2p[mask])
                    pct = (arr_r2p[mask] > 0).mean() * 100
                    hi_str = f"{hi:.0f}" if hi != np.inf else "+"
                    print(f"      std {lo:.0f}-{hi_str}m: median R² persist={med:.4f} ({pct:.0f}% > 0, {n} wells)")


    # =====================  _plot_scatter_pred_vs_actual  =====================
    def _plot_scatter_pred_vs_actual(self, preds_abs, targets_abs, split_name="val"):
        """Predicted vs actual scatter with 45° line and error coloring."""
        errors = np.abs(preds_abs - targets_abs)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(
            targets_abs,
            preds_abs,
            c=errors,
            cmap="RdYlGn_r",
            alpha=0.4,
            s=8,
            edgecolors="none",
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label("Absolute Error (m)")

        # 45° reference line
        lo = min(targets_abs.min(), preds_abs.min())
        hi = max(targets_abs.max(), preds_abs.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            linewidth=1,
            alpha=0.5,
            label="Perfect prediction",
        )

        # Global metrics
        rmse = np.sqrt(((preds_abs - targets_abs) ** 2).mean())
        mae = np.abs(preds_abs - targets_abs).mean()
        ax.set_xlabel("Actual GWL (m)")
        ax.set_ylabel("Predicted GWL (m)")
        ax.set_title(
            f"{split_name.capitalize()}: Predicted vs Actual GWL\n"
            f"RMSE={rmse:.2f}m, MAE={mae:.2f}m, N={len(preds_abs):,}"
        )
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_scatter_pred_vs_actual.png", split_name),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_delta_quadrants  =====================
    def _plot_delta_quadrants(self, pred_deltas, target_deltas, split_name="val"):
        """Delta prediction plot with trend-correctness quadrants."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Color by trend correctness
        correct_trend = (pred_deltas >= 0) == (target_deltas >= 0)
        ax.scatter(
            target_deltas[correct_trend],
            pred_deltas[correct_trend],
            alpha=0.3,
            s=8,
            c="tab:green",
            label="Correct trend",
            edgecolors="none",
        )
        ax.scatter(
            target_deltas[~correct_trend],
            pred_deltas[~correct_trend],
            alpha=0.3,
            s=8,
            c="tab:red",
            label="Wrong trend",
            edgecolors="none",
        )

        # Quadrant lines
        lo = min(target_deltas.min(), pred_deltas.min())
        hi = max(target_deltas.max(), pred_deltas.max())
        margin = (hi - lo) * 0.05
        ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            linewidth=1,
            alpha=0.3,
        )

        trend_acc = correct_trend.mean()
        ax.set_xlabel("Actual Δ(GWL) (m)")
        ax.set_ylabel("Predicted Δ(GWL) (m)")
        ax.set_title(
            f"{split_name.capitalize()}: Delta Predictions — Trend Accuracy: {trend_acc:.1%}\n"
            f"Green=correct trend, Red=wrong trend"
        )
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_delta_quadrants.png", split_name),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_error_distribution  =====================
    def _plot_error_distribution(self, preds_abs, targets_abs, split_name="val"):
        """Histogram of signed prediction errors."""
        errors = preds_abs - targets_abs

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            errors,
            bins=80,
            color="steelblue",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(
            0, color="red", linewidth=1.5, linestyle="--", alpha=0.7, label="Zero error"
        )
        ax.axvline(
            errors.mean(),
            color="orange",
            linewidth=1.5,
            linestyle="-",
            alpha=0.7,
            label=f"Mean bias: {errors.mean():.2f}m",
        )

        p5, p95 = np.percentile(errors, [5, 95])
        ax.axvline(p5, color="gray", linewidth=1, linestyle=":", alpha=0.5)
        ax.axvline(
            p95,
            color="gray",
            linewidth=1,
            linestyle=":",
            alpha=0.5,
            label=f"90% range: [{p5:.1f}, {p95:.1f}]m",
        )

        ax.set_xlabel("Prediction Error (m) — Predicted minus Actual")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{split_name.capitalize()} Error Distribution\n"
            f"Mean={errors.mean():.2f}m, Std={errors.std():.2f}m, "
            f"Median={np.median(errors):.2f}m"
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_error_distribution.png", split_name),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_per_well_predictions  =====================
    def _plot_per_well_predictions(
        self,
        preds_abs,
        targets_abs,
        well_ids,
        date_ordinals,
        idx_to_station,
        n_per_category=3,
        split_name="val",
        filename_suffix="",
        title_suffix="",
        ylabel="GWL (m)",
    ):
        """
        Time series plots for best / most-sampled / worst wells by R².

        For each category, plots actual vs predicted over time,
        sorted chronologically within each well. X-axis shows real dates.
        """
        from sklearn.metrics import r2_score
        import matplotlib.dates as mdates
        from datetime import date as _date

        # Compute per-well R² and RMSE
        unique_wells = np.unique(well_ids)
        min_well_samples = 5
        well_r2 = {}
        well_rmses = {}
        for wid in unique_wells:
            mask = well_ids == wid
            if mask.sum() < min_well_samples:
                continue
            well_r2[wid] = r2_score(targets_abs[mask], preds_abs[mask])
            well_errors = preds_abs[mask] - targets_abs[mask]
            well_rmses[wid] = np.sqrt((well_errors**2).mean())

        if len(well_r2) < 3:
            print("  Warning: too few wells for per-well diagnostic plots")
            return

        # Sort wells by R² (best = highest R²)
        sorted_wells = sorted(well_r2.keys(), key=lambda w: well_r2[w], reverse=True)
        n_wells = len(sorted_wells)

        # Select wells: best (highest R²), most-sampled, worst (lowest R²)
        best_wells = sorted_wells[:n_per_category]
        worst_wells = sorted_wells[-n_per_category:]

        # Most-sampled wells (excluding already selected best/worst)
        selected = set(best_wells + worst_wells)
        well_sample_counts = {
            wid: (well_ids == wid).sum()
            for wid in well_r2
            if wid not in selected
        }
        most_sampled_wells = sorted(
            well_sample_counts.keys(), key=lambda w: well_sample_counts[w], reverse=True
        )[:n_per_category]

        categories = [
            ("BEST", best_wells, "#0D88ED"),
            ("Most Sampled", most_sampled_wells, "tab:blue"),
            ("WORST", worst_wells, "#FF5722"),
        ]

        n_rows = len(categories)
        n_cols = n_per_category
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
        )

        for row, (category_label, wells, color) in enumerate(categories):
            for col, wid in enumerate(wells):
                ax = axes[row, col]
                mask = well_ids == wid
                dates = date_ordinals[mask]
                preds = preds_abs[mask]
                targets = targets_abs[mask]

                # Sort chronologically
                sort_idx = np.argsort(dates)
                dates_sorted = dates[sort_idx]
                preds_sorted = preds[sort_idx]
                targets_sorted = targets[sort_idx]

                # Convert ordinals to matplotlib date floats
                x_axis = [
                    mdates.date2num(_date.fromordinal(int(d)))
                    for d in dates_sorted
                ]

                ax.plot(x_axis, targets_sorted, "b-", linewidth=1.2,
                        label="Actual", alpha=0.8)
                ax.plot(x_axis, preds_sorted, "r--", linewidth=1.2,
                        label="Predicted", alpha=0.8)

                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                         fontsize=7)

                station_code = idx_to_station.get(wid, f"well_{wid}")
                r2 = well_r2[wid]
                rmse = well_rmses[wid]
                ax.set_title(
                    f"{category_label} | {station_code}\n"
                    f"R²={r2:.4f}  RMSE={rmse:.2f}m  ({mask.sum()} samples)",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel(ylabel)

        plt.suptitle(
            f"Per-Well {split_name.capitalize()} Predictions{title_suffix} — {n_wells:,} wells total",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_per_well_predictions{filename_suffix}.png", split_name, is_delta="_delta" in filename_suffix),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


    # =====================  _get_train_std  =====================
    def _get_train_std(self, station_code: str) -> Tuple[Optional[float], str]:
        """
        Look up canonical train_std for a station.

        Returns (value, source) where source ∈ {"train", "eval"}:
          - "train": station has real train_std (Scenario A)
          - "eval":  station has no train_std (Scenario B cold-start);
                    value is None — caller decides on fallback (eval-sample std)
                    and tagging behavior.
        """
        if station_code in self._station_to_train_std:
            return float(self._station_to_train_std[station_code]), "train"
        return None, "eval"


    # =====================  _apply_test_cap_and_filter  =====================
    def _apply_test_cap_and_filter(
        self,
        preds_abs,
        targets_abs,
        pred_deltas,
        target_deltas,
        well_ids,
        date_ordinals,
        idx_to_station: Dict[int, str],
        station_to_state: Dict[str, str] = None,
        linear_pred_abs=None,
        linear_pred_deltas=None,
    ):
        """
        Sort test samples per well by date and split wells into predictable vs flat
        using the canonical train_std map AND optional state whitelist.

        Predictable (Scenario A, passes ALL active filters):
            station has real train_std AND
              train_std >= MIN_STATION_TARGET_STD AND
              (state ∈ INCLUDE_STATES or INCLUDE_STATES is empty)
        Flat (Scenario A, fails any filter):
            station has real train_std but fails std threshold or state whitelist.
        Cold-start (Scenario B):
            station has NO train_std (e.g., district split, or <2 train samples).
            Always classified flat — excluded from predictable cohort regardless
            of any eval-sample std.

        Capping is no longer done here — data prep already caps val+test at
        max_samples_per_station_eval. The original "cap" step would just be
        a no-op given that. The function name is kept for backward compat
        with external callers; semantically it's "sort + predictable split".

        Output dict keys are unchanged (capped_*, filtered_*, flat_*) so existing
        callers in train.py and gwl_geo_analysis.py keep working.
        """
        threshold = self._min_station_target_std
        states_set = self._include_states  # already a set; empty = no filter
        min_mode_gap = self._min_station_mode_gap_days
        s2s = station_to_state or {}
        s2mg = self._station_to_mode_gap_days
        unique_wells = np.unique(well_ids)

        # Sort each well's samples by date (predictable order for plotting)
        sort_indices = []
        for wid in unique_wells:
            idx = np.where(well_ids == wid)[0]
            idx = idx[np.argsort(date_ordinals[idx])]
            sort_indices.append(idx)
        sorted_idx = (
            np.concatenate(sort_indices) if sort_indices else np.array([], dtype=np.int64)
        )

        s_preds = preds_abs[sorted_idx]
        s_targets = targets_abs[sorted_idx]
        s_pred_d = pred_deltas[sorted_idx]
        s_target_d = target_deltas[sorted_idx]
        s_well_ids = well_ids[sorted_idx]
        s_dates = date_ordinals[sorted_idx]

        s_linear_pred_abs = (
            linear_pred_abs[sorted_idx] if linear_pred_abs is not None else None
        )
        s_linear_pred_d = (
            linear_pred_deltas[sorted_idx] if linear_pred_deltas is not None else None
        )

        # Predictable / flat / cold-start split via canonical train_std map
        # AND optional state whitelist AND optional mode-gap filter. All filters
        # compose by AND; any can be vacuous (off when threshold=0, states_set
        # is empty, or min_mode_gap=0).
        predictable_wells = set()
        flat_wells = set()
        cold_start_wells = set()
        for wid in unique_wells:
            station_code = idx_to_station.get(int(wid))
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                cold_start_wells.add(int(wid))
                flat_wells.add(int(wid))  # excluded from predictable
                continue
            std_ok   = (train_std >= threshold)
            state_ok = (not states_set) or (s2s.get(station_code) in states_set)
            if min_mode_gap > 0:
                mg = s2mg.get(station_code)
                mode_gap_ok = (mg is not None) and (mg >= min_mode_gap)
            else:
                mode_gap_ok = True
            if std_ok and state_ok and mode_gap_ok:
                predictable_wells.add(int(wid))
            else:
                flat_wells.add(int(wid))

        pred_mask = np.isin(s_well_ids, list(predictable_wells))
        flat_mask = np.isin(s_well_ids, list(flat_wells))

        out = {
            "capped_preds": s_preds,
            "capped_targets": s_targets,
            "capped_pred_deltas": s_pred_d,
            "capped_target_deltas": s_target_d,
            "capped_well_ids": s_well_ids,
            "capped_dates": s_dates,
            "filtered_preds": s_preds[pred_mask],
            "filtered_targets": s_targets[pred_mask],
            "filtered_pred_deltas": s_pred_d[pred_mask],
            "filtered_target_deltas": s_target_d[pred_mask],
            "filtered_well_ids": s_well_ids[pred_mask],
            "filtered_dates": s_dates[pred_mask],
            "flat_preds": s_preds[flat_mask],
            "flat_targets": s_targets[flat_mask],
            "flat_well_ids": s_well_ids[flat_mask],
            "flat_dates": s_dates[flat_mask],
            "n_wells_raw": len(unique_wells),
            "n_wells_capped": len(unique_wells),
            "n_wells_predictable": len(predictable_wells),
            "n_wells_flat": len(flat_wells),
            "n_wells_cold_start": len(cold_start_wells),
            "n_samples_raw": len(preds_abs),
            "n_samples_capped": len(s_preds),
            "predictable_well_ids": predictable_wells,
            "flat_well_ids_set": flat_wells,
            "cold_start_well_ids": cold_start_wells,
        }
        if s_linear_pred_abs is not None:
            out["capped_linear_preds"] = s_linear_pred_abs
            out["capped_linear_pred_deltas"] = s_linear_pred_d
            out["filtered_linear_preds"] = s_linear_pred_abs[pred_mask]
            out["filtered_linear_pred_deltas"] = s_linear_pred_d[pred_mask]
        return out


    # =====================  _plot_flat_wells  =====================
    def _plot_flat_wells(
        self,
        preds,
        targets,
        well_ids,
        date_ordinals,
        idx_to_station,
        split_name: str = "test",
        n_per_row: int = 5,
    ):
        """
        Plot wells excluded by the predictable filter (std < 1.0m).

        3 rows x n_per_row grid:
          - Row 1: flattest wells (lowest std)
          - Row 2: most-sampled flat wells
          - Row 3: borderline flat wells (highest std but still below threshold)

        No R² in titles (meaningless for flat wells).
        """
        import matplotlib.dates as mdates
        from datetime import date as _date

        unique_wells = np.unique(well_ids)
        if len(unique_wells) < 1:
            return

        # Compute std and n per well, skip wells with < 20 samples
        well_stats = {}
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < 20:
                continue
            well_stats[wid] = {
                "std": float(targets[m].std()),
                "n": int(m.sum()),
                "rmse": float(np.sqrt(((preds[m] - targets[m]) ** 2).mean())),
            }

        if len(well_stats) < n_per_row:
            print(f"  Too few flat wells for flat-wells plot ({len(well_stats)} < {n_per_row})")
            return

        sorted_by_std = sorted(well_stats.keys(), key=lambda w: well_stats[w]["std"])
        flattest = sorted_by_std[:n_per_row]

        # Row 2: most-sampled (excluding already selected)
        selected = set(flattest)
        remaining = [w for w in well_stats if w not in selected]
        most_sampled = sorted(remaining, key=lambda w: well_stats[w]["n"], reverse=True)[:n_per_row]

        # Row 3: borderline flat (highest std but still < threshold)
        selected.update(most_sampled)
        remaining2 = [w for w in well_stats if w not in selected]
        borderline = sorted(remaining2, key=lambda w: well_stats[w]["std"], reverse=True)[:n_per_row]

        categories = [
            ("Flattest (lowest std)", flattest, "#9b59b6"),
            ("Most-sampled flat", most_sampled, "#1abc9c"),
            ("Borderline (std near 1.0m)", borderline, "#e67e22"),
        ]

        rows = [c for c in categories if len(c[1]) > 0]
        if not rows:
            return

        n_rows = len(rows)
        n_cols = n_per_row
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
        )

        for row_i, (label, wells, color) in enumerate(rows):
            for col_i in range(n_cols):
                ax = axes[row_i, col_i]
                if col_i >= len(wells):
                    ax.axis("off")
                    continue
                wid = wells[col_i]
                m = well_ids == wid
                dates = date_ordinals[m]
                p = preds[m]
                t = targets[m]
                sort_idx = np.argsort(dates)
                dates_s = dates[sort_idx]
                p_s = p[sort_idx]
                t_s = t[sort_idx]

                x = [mdates.date2num(_date.fromordinal(int(d))) for d in dates_s]
                ax.plot(x, t_s, "b-", linewidth=1.2, label="Actual", alpha=0.8)
                ax.plot(x, p_s, "r--", linewidth=1.2, label="Predicted", alpha=0.8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

                station_code = idx_to_station.get(wid, f"well_{wid}")
                stats = well_stats[wid]
                ax.set_title(
                    f"{label} | {station_code}\n"
                    f"std={stats['std']:.2f}m  RMSE={stats['rmse']:.2f}m  ({stats['n']} samples)",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
                if col_i == 0:
                    ax.set_ylabel("GWL (m)")

        plt.suptitle(
            f"Flat Wells Excluded from Predictable Filter — {split_name.capitalize()} "
            f"(std < 1.0m, {len(well_stats)} wells)",
            fontsize=13,
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_flat_wells_predictions.png", split_name),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_per_well_by_gwl_level  =====================
    def _plot_per_well_by_gwl_level(
        self,
        preds_abs,
        targets_abs,
        well_ids,
        date_ordinals,
        idx_to_station,
        n_per_side: int = 3,
        split_name: str = "val",
        filename_suffix: str = "",
        title_suffix: str = "",
        ylabel: str = "GWL (m)",
        stratify_targets=None,
        stratify_mode: str = "mean",
    ):
        """
        Per-well prediction plots stratified by GWL level.

        3 strata (high/medium/low) × 6 columns (3 best + 3 worst by R²).

        Args:
            stratify_targets: If provided, use these (e.g., absolute targets) for
                computing strata instead of targets_abs. Useful when plotting deltas
                but stratifying by absolute GWL level.
            stratify_mode: "mean" = stratify by mean(values) (default),
                          "mean_abs" = stratify by mean(|values|) (for delta volatility).
        """
        from sklearn.metrics import r2_score
        import matplotlib.dates as mdates
        from datetime import date as _date

        min_well_samples = 5
        unique_wells = np.unique(well_ids)
        strat_data = stratify_targets if stratify_targets is not None else targets_abs

        # Compute per-well stats
        well_stats = {}
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            t = targets_abs[m]
            p = preds_abs[m]
            s = strat_data[m]
            if stratify_mode == "mean_abs":
                strat_val = float(np.abs(s).mean())
            else:
                strat_val = float(s.mean())
            well_stats[wid] = {
                "strat_val": strat_val,
                "mean_gwl": float(s.mean()),
                "r2": float(r2_score(t, p)),
                "rmse": float(np.sqrt(((p - t) ** 2).mean())),
                "n": int(m.sum()),
            }

        if len(well_stats) < 15:
            print(f"  Too few wells for GWL-level stratified plot ({len(well_stats)})")
            return

        # Sort by stratification value, split into quintiles
        sorted_wids = sorted(well_stats.keys(), key=lambda w: well_stats[w]["strat_val"])
        n = len(sorted_wids)
        q20 = max(1, int(n * 0.2))
        q40 = max(q20 + 1, int(n * 0.4))
        q60 = max(q40 + 1, int(n * 0.6))
        q80 = max(q60 + 1, int(n * 0.8))

        low_wids = sorted_wids[:q20]          # lowest mean GWL (shallowest)
        mid_wids = sorted_wids[q40:q60]       # middle
        high_wids = sorted_wids[q80:]         # highest mean GWL (deepest)

        def pick_best_worst(wids, n_pick):
            by_r2 = sorted(wids, key=lambda w: well_stats[w]["r2"], reverse=True)
            best = by_r2[:n_pick]
            worst = by_r2[-n_pick:] if len(by_r2) >= 2 * n_pick else by_r2[n_pick:]
            return best, worst

        if stratify_mode == "mean_abs":
            strata_labels = [
                ("High volatility", high_wids, "#0D88ED", "#FF5722"),
                ("Medium volatility", mid_wids, "#0D88ED", "#FF5722"),
                ("Low volatility", low_wids, "#0D88ED", "#FF5722"),
            ]
        else:
            strata_labels = [
                ("High GWL (deep)", high_wids, "#0D88ED", "#FF5722"),
                ("Medium GWL", mid_wids, "#0D88ED", "#FF5722"),
                ("Low GWL (shallow)", low_wids, "#0D88ED", "#FF5722"),
            ]

        strata = []
        for label, wids, color_best, color_worst in strata_labels:
            if len(wids) < 2 * n_per_side:
                continue
            best, worst = pick_best_worst(wids, n_per_side)
            strata.append((label, best, worst, color_best, color_worst))

        if not strata:
            print("  Not enough wells in each stratum for GWL-level plot")
            return

        n_rows = len(strata)
        n_cols = 2 * n_per_side
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

        for row_i, (label, best, worst, cb, cw) in enumerate(strata):
            wells_in_row = list(best) + list(worst)
            colors = [cb] * len(best) + [cw] * len(worst)
            sublabels = ["BEST"] * len(best) + ["WORST"] * len(worst)

            for col_i, (wid, color, sublabel) in enumerate(zip(wells_in_row, colors, sublabels)):
                if col_i >= n_cols:
                    break
                ax = axes[row_i, col_i]
                m = well_ids == wid
                dates = date_ordinals[m]
                p = preds_abs[m]
                t = targets_abs[m]

                sort_idx = np.argsort(dates)
                dates_s = dates[sort_idx]
                p_s = p[sort_idx]
                t_s = t[sort_idx]

                x = [mdates.date2num(_date.fromordinal(int(d))) for d in dates_s]
                ax.plot(x, t_s, "b-", linewidth=1.2, label="Actual", alpha=0.8)
                ax.plot(x, p_s, color=color, linestyle="--", linewidth=1.2,
                        label="Predicted", alpha=0.8)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

                station_code = idx_to_station.get(wid, f"well_{wid}")
                stats = well_stats[wid]
                ax.set_title(
                    f"{sublabel} | {station_code}\n"
                    f"mean={stats['mean_gwl']:.1f}m  R2={stats['r2']:.3f}  "
                    f"RMSE={stats['rmse']:.2f}m  ({stats['n']})",
                    fontsize=8, fontweight="bold", color=color,
                )
                ax.legend(fontsize=6)
                ax.grid(True, alpha=0.3)
                if col_i == 0:
                    ax.set_ylabel(f"{label}\n{ylabel}")

        plt.suptitle(
            f"Per-Well Predictions by GWL Level{title_suffix} — {split_name.capitalize()} "
            f"({len(well_stats)} wells)",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_per_well_by_gwl_level{filename_suffix}.png", split_name, is_delta="_delta" in filename_suffix),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_r2_vs_depth  =====================
    def _plot_r2_vs_depth(
        self,
        preds_abs,
        targets_abs,
        pred_deltas,
        target_deltas,
        well_ids,
        idx_to_station,
        split_name: str = "val",
    ):
        """
        2×2 scatter grid: R² vs mean GWL (depth proxy).

        Layout:
          Top-left:  R² (abs) vs mean GWL
          Top-right: R² (abs) vs mean(|delta|)
          Bot-left:  R² (delta) vs mean GWL
          Bot-right: R² (delta) vs mean(|delta|)

        Each dot = one well. Horizontal line at R²=0.
        """
        from sklearn.metrics import r2_score

        min_well_samples = 5
        unique_wells = np.unique(well_ids)

        mean_gwls = []
        mean_abs_deltas = []
        r2_abs_list = []
        r2_delta_list = []
        labels = []

        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            t_abs = targets_abs[m]
            p_abs = preds_abs[m]
            t_d = target_deltas[m]
            p_d = pred_deltas[m]

            mean_gwls.append(float(t_abs.mean()))
            mean_abs_deltas.append(float(np.abs(t_d).mean()))
            r2_abs_list.append(float(r2_score(t_abs, p_abs)))
            r2_delta_list.append(float(r2_score(t_d, p_d)))
            labels.append(idx_to_station.get(wid, f"well_{wid}"))

        if len(mean_gwls) < 5:
            print(f"  Too few wells for R² vs depth plot ({len(mean_gwls)})")
            return

        mean_gwls = np.array(mean_gwls)
        mean_abs_deltas = np.array(mean_abs_deltas)
        r2_abs = np.array(r2_abs_list)
        r2_delta = np.array(r2_delta_list)

        # Clip R² for display
        r2_abs_clip = np.clip(r2_abs, -2, 1)
        r2_delta_clip = np.clip(r2_delta, -2, 1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        configs = [
            (axes[0, 0], mean_gwls, r2_abs_clip, "R² (abs) vs Mean GWL", "Mean GWL (m)"),
            (axes[0, 1], mean_abs_deltas, r2_abs_clip, "R² (abs) vs Mean |Delta|", "Mean |Delta| (m)"),
            (axes[1, 0], mean_gwls, r2_delta_clip, "R² (delta) vs Mean GWL", "Mean GWL (m)"),
            (axes[1, 1], mean_abs_deltas, r2_delta_clip, "R² (delta) vs Mean |Delta|", "Mean |Delta| (m)"),
        ]

        for ax, x, y, title, xlabel in configs:
            # Color by R² value
            sc = ax.scatter(x, y, c=y, cmap="RdYlGn", s=20, alpha=0.7,
                           edgecolors="none", vmin=-2, vmax=1)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", alpha=0.4)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("R²")
            ax.set_title(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-2.1, 1.1)

        fig.colorbar(sc, ax=axes, label="R²", shrink=0.6)

        n_wells = len(mean_gwls)
        plt.suptitle(
            f"R² vs Depth/Volatility — {split_name.capitalize()} ({n_wells} wells)",
            fontsize=13, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_r2_vs_depth.png", split_name),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_best_worst_timeseries  =====================
    # =====================  _collect_per_well_history  =====================
    def _collect_per_well_history(self, data, cache_key: str):
        """
        Pull (date_ordinal, target_gwl_raw) per well from a dataset, keyed by
        station_code (string). Used to overlay train (and val) history behind
        val/test best-worst panels.

        IMPORTANT: must key by station_code, NOT well_id. Each LSTMTensorDataset
        builds its own station_to_idx from its unique stations, so well_id=5
        in train_data is a different station than well_id=5 in val_data.

        No inference — just reads tensors directly. Cached by `cache_key`.
        """
        if data is None:
            return {}
        if cache_key in self._history_cache:
            return self._history_cache[cache_key]

        well_ids = data.well_id.cpu().numpy()
        date_ords = data.target_date_ordinal.cpu().numpy()
        if data.target_gwl_raw is not None:
            gwl_raw = data.target_gwl_raw.cpu().numpy()
        else:
            gwl_raw = data.target_gwl.cpu().numpy()
        # Local well_id → station_code map for THIS dataset
        local_idx_to_station = data.idx_to_station

        history = {}
        for wid in np.unique(well_ids):
            mask = well_ids == wid
            d = date_ords[mask]
            g = gwl_raw[mask]
            sort_idx = np.argsort(d)
            station_code = local_idx_to_station.get(int(wid))
            if station_code is None:
                continue
            history[station_code] = (d[sort_idx], g[sort_idx])

        self._history_cache[cache_key] = history
        return history


    # =====================  _render_well_timeseries_panel  =====================
    def _render_well_timeseries_panel(
        self,
        ax,
        wid,
        well_ids,
        preds_abs,
        targets_abs,
        date_ordinals,
        idx_to_station,
        train_hist,
        val_hist,
        split_name,
        linear_preds_abs,
        title_prefix,
        color,
        r2_d,
        n_samples,
        well_std_val,
        std_source: str = "train",
        show_ylabel: bool = True,
    ):
        """
        Render one well's time series into `ax`: history truth + eval truth +
        predicted overlay (+ optional linear-only baseline). Used by both
        _plot_best_worst_timeseries and _plot_best_or_worst_by_std_bucket.
        """
        import matplotlib.dates as mdates
        from datetime import date as _date

        m = well_ids == wid
        d_ord = date_ordinals[m]
        preds = preds_abs[m]
        targets = targets_abs[m]
        sort_idx = np.argsort(d_ord)
        d_ord = d_ord[sort_idx]
        preds = preds[sort_idx]
        targets = targets[sort_idx]
        linear_preds = None
        if linear_preds_abs is not None:
            linear_preds = linear_preds_abs[m][sort_idx]

        station_code_lookup = idx_to_station.get(wid, None)
        if station_code_lookup is not None:
            t_d, t_g = train_hist.get(station_code_lookup, (np.array([]), np.array([])))
            v_d, v_g = val_hist.get(station_code_lookup, (np.array([]), np.array([])))
        else:
            t_d, t_g = np.array([]), np.array([])
            v_d, v_g = np.array([]), np.array([])

        def _ord_to_x(arr):
            return [mdates.date2num(_date.fromordinal(int(d))) for d in arr]

        x_eval = _ord_to_x(d_ord)
        x_train = _ord_to_x(t_d)
        x_val = _ord_to_x(v_d)

        eval_start = min(x_eval) if x_eval else None
        val_start = min(x_val) if x_val else None

        if x_train and eval_start is not None:
            end = val_start if val_start is not None else eval_start
            ax.axvspan(min(x_train), end, color="0.92", zorder=0)
        if x_val and eval_start is not None:
            ax.axvspan(val_start, eval_start, color="0.85", zorder=0)

        all_x = list(x_train) + list(x_val) + list(x_eval)
        all_y = list(t_g) + list(v_g) + list(targets)
        if all_x:
            order = np.argsort(all_x)
            all_x_arr = np.array(all_x)[order]
            all_y_arr = np.array(all_y)[order]
            ax.plot(all_x_arr, all_y_arr, "b-", linewidth=1.0,
                    label="Actual", alpha=0.7)

        ax.plot(x_eval, preds, "r--", linewidth=1.3,
                label="Predicted", alpha=0.9)
        if linear_preds is not None:
            ax.plot(x_eval, linear_preds, color="#2E8B57", linestyle=":",
                    linewidth=1.0, label="Linear only", alpha=0.7)

        if val_start is not None:
            ax.axvline(val_start, color="gray", linestyle=":",
                       linewidth=0.7, alpha=0.6)
        if eval_start is not None and x_train:
            ax.axvline(eval_start, color="gray", linestyle=":",
                       linewidth=0.7, alpha=0.6)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=30, ha="right", fontsize=7)

        station_code = idx_to_station.get(wid, f"well_{wid}")
        missing_tags = []
        if len(t_d) == 0:
            missing_tags.append("no train")
        if split_name == "test" and len(v_d) == 0:
            missing_tags.append("no val")
        if std_source == "eval":
            missing_tags.append("no train-std")
        missing_str = f"  [{', '.join(missing_tags)}]" if missing_tags else ""
        ax.set_title(
            f"{title_prefix} | {station_code}{missing_str}\n"
            f"R² delta = {r2_d:.3f}  (n={n_samples}, std={well_std_val:.2f}m)",
            fontsize=9, fontweight="bold", color=color,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if show_ylabel:
            ax.set_ylabel("GWL (m)")


    # =====================  _plot_best_or_worst_by_std_bucket  =====================
    def _plot_best_or_worst_by_std_bucket(
        self,
        pred_deltas,
        target_deltas,
        preds_abs,
        targets_abs,
        well_ids,
        date_ordinals,
        idx_to_station,
        mode: str = "best",
        split_name: str = "val",
        linear_preds_abs=None,
    ):
        """
        Per-std-bucket time series of one representative well per bucket.

        Buckets match `_plot_scatter_by_std_bucket`: [0-1, 1-3, 3-8, 8-15, 15+].
        For each bucket, restrict to wells with n_samples >= 0.8 * max(n in bucket)
        and pick the well with highest (mode="best") or lowest (mode="worst")
        R² delta.

        2x3 grid (last cell empty). Same panel rendering as
        _plot_best_worst_timeseries (history overlay, period shading, optional
        linear-only baseline, std/n/R²Δ in title).
        """
        from sklearn.metrics import r2_score

        if split_name == "test":
            train_hist = self._collect_per_well_history(self._train_data, "train")
            val_hist = self._collect_per_well_history(self._val_data, "val")
        elif split_name == "val":
            train_hist = self._collect_per_well_history(self._train_data, "train")
            val_hist = {}
        else:
            train_hist = {}
            val_hist = {}

        bucket_defs = [
            ("0-1m",   0.0,  1.0,  "#FF5722"),
            ("1-3m",   1.0,  3.0,  "#FF9800"),
            ("3-8m",   3.0,  8.0,  "#4CAF50"),
            ("8-15m",  8.0,  15.0, "#0D88ED"),
            ("15m+",   15.0, np.inf, "#673AB7"),
        ]

        min_well_samples = 5
        unique_wells = np.unique(well_ids)
        well_r2_d, well_n, well_std, well_std_source = {}, {}, {}, {}
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            station_code = idx_to_station.get(int(wid))
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                # Scenario B cold-start: fall back to eval-sample std for placement
                # only; mark with source so panel title can flag it.
                train_std = float(target_deltas[m].std())
            well_r2_d[wid] = float(r2_score(target_deltas[m], pred_deltas[m]))
            well_n[wid] = int(m.sum())
            well_std[wid] = train_std
            well_std_source[wid] = source

        if not well_r2_d:
            print(f"  Too few wells for {mode}-by-std-bucket plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes_flat = axes.flatten()
        pick_extreme = max if mode == "best" else min

        for ax, (label, lo, hi, color) in zip(axes_flat, bucket_defs):
            wells_in = [w for w in well_r2_d if lo <= well_std[w] < hi]
            if not wells_in:
                ax.text(0.5, 0.5, "no wells", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_title(f"std {label}  (0 wells)", fontsize=10, color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            max_n = max(well_n[w] for w in wells_in)
            threshold = max(min_well_samples, int(0.8 * max_n))
            candidates = [w for w in wells_in if well_n[w] >= threshold]
            if not candidates:
                threshold = max(min_well_samples, int(0.5 * max_n))
                candidates = [w for w in wells_in if well_n[w] >= threshold]
            if not candidates:
                candidates = wells_in

            wid = pick_extreme(candidates, key=lambda w: well_r2_d[w])
            self._render_well_timeseries_panel(
                ax, wid, well_ids, preds_abs, targets_abs,
                date_ordinals, idx_to_station, train_hist, val_hist,
                split_name, linear_preds_abs,
                title_prefix=f"std {label}", color=color,
                r2_d=well_r2_d[wid], n_samples=well_n[wid],
                well_std_val=well_std[wid],
                std_source=well_std_source[wid],
                show_ylabel=True,
            )

        for extra_ax in axes_flat[len(bucket_defs):]:
            extra_ax.axis("off")

        pretty_mode = "Best" if mode == "best" else "Worst"
        plt.suptitle(
            f"{pretty_mode} Well Per std Bucket — {split_name.capitalize()}",
            fontsize=13, y=1.005,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_{mode}_by_std_bucket.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_best_worst_timeseries  =====================
    def _plot_best_worst_timeseries(
        self,
        pred_deltas,
        target_deltas,
        preds_abs,
        targets_abs,
        well_ids,
        date_ordinals,
        idx_to_station,
        n_per_category: int = 5,
        split_name: str = "val",
        linear_preds_abs=None,
    ):
        """
        Side-by-side time series plot of best vs worst wells (by R² delta).

        2 columns x n_per_category rows. Left = best, right = worst.
        Each panel shows actual vs predicted absolute GWL over time.

        For val plots: train period truth is overlaid behind the val window.
        For test plots: train + val period truth is overlaid behind the test window.
        Periods are distinguished by background shading.

        If `linear_preds_abs` is provided (model has linear residual head enabled),
        a third line shows the linear-only prediction. Useful to see how much
        of the prediction comes from the linear baseline vs the LSTM/TFT path.
        """
        from sklearn.metrics import r2_score
        import matplotlib.dates as mdates
        from datetime import date as _date

        # Load history depending on split.
        if split_name == "test":
            train_hist = self._collect_per_well_history(self._train_data, "train")
            val_hist = self._collect_per_well_history(self._val_data, "val")
        elif split_name == "val":
            train_hist = self._collect_per_well_history(self._train_data, "train")
            val_hist = {}
        else:
            train_hist = {}
            val_hist = {}

        min_well_samples = 5
        unique_wells = np.unique(well_ids)

        # Compute per-well R² delta + std for ALL wells with enough samples.
        # No std-based filter here — flat wells are now allowed so the worst panels
        # can expose cold-start failures (input data is already capped @ data-prep).
        # std uses canonical train_std (or eval-sample fallback for cold-start).
        well_r2_d = {}
        well_n = {}
        well_std = {}
        well_std_source = {}
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            station_code = idx_to_station.get(int(wid))
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                train_std = float(target_deltas[m].std())
            well_r2_d[wid] = float(r2_score(target_deltas[m], pred_deltas[m]))
            well_n[wid] = int(m.sum())
            well_std[wid] = train_std
            well_std_source[wid] = source

        if len(well_r2_d) < 2 * n_per_category:
            print(f"  Too few wells for best/worst plot ({len(well_r2_d)})")
            return

        # Restrict candidates to wells with high sample counts so time-series plots
        # have enough points to be readable. Threshold = 0.8 × max observed n;
        # fall back to 0.5 × max if too few candidates.
        max_n = max(well_n.values())
        threshold = max(min_well_samples, int(0.8 * max_n))
        candidates = [w for w, n in well_n.items() if n >= threshold]
        if len(candidates) < 2 * n_per_category:
            threshold = max(min_well_samples, int(0.5 * max_n))
            candidates = [w for w, n in well_n.items() if n >= threshold]
        # Last-resort fallback: take all wells if still too few
        if len(candidates) < 2 * n_per_category:
            candidates = list(well_r2_d.keys())

        sorted_wells = sorted(candidates, key=lambda w: well_r2_d[w])
        worst = sorted_wells[:n_per_category]
        best = sorted_wells[-n_per_category:][::-1]  # highest R² first
        print(f"  Best/worst plot: candidates with n≥{threshold} (max_n={max_n}): {len(candidates)} wells")

        # Cache selected worst station codes by split_name so the
        # full-timeline diagnostic can plot the SAME stations shown here.
        if not hasattr(self, "_last_worst_stations"):
            self._last_worst_stations = {}
        self._last_worst_stations[split_name] = [
            (idx_to_station.get(int(w)), float(well_r2_d[w]), int(well_n[w]))
            for w in worst
            if idx_to_station.get(int(w)) is not None
        ]

        fig, axes = plt.subplots(
            n_per_category, 2,
            figsize=(14, 3.2 * n_per_category),
            squeeze=False,
        )

        for col, (label, wells, color) in enumerate([
            ("BEST", best, "#0D88ED"),
            ("WORST", worst, "#FF5722"),
        ]):
            for row, wid in enumerate(wells):
                ax = axes[row, col]
                self._render_well_timeseries_panel(
                    ax, wid, well_ids, preds_abs, targets_abs,
                    date_ordinals, idx_to_station, train_hist, val_hist,
                    split_name, linear_preds_abs,
                    title_prefix=label, color=color,
                    r2_d=well_r2_d[wid], n_samples=well_n[wid],
                    well_std_val=well_std[wid],
                    std_source=well_std_source[wid],
                    show_ylabel=(col == 0),
                )

        plt.suptitle(
            f"Best vs Worst Wells (by R² delta) — {split_name.capitalize()}",
            fontsize=13, y=1.005,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_best_worst_timeseries.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  generate_worst_full_timeline_plot  =====================
    @torch.no_grad()
    def generate_worst_full_timeline_plot(
        self,
        train_data, train_sampler,
        val_data, val_sampler,
        test_data, test_sampler,
        n_per_split: int = 5,
    ):
        """
        Plot worst-performing wells (by per-station R²δ in val and test)
        across their full train+val+test timeline.

        For each station: actual GWL over the full history (target_gwl_raw at
        target dates of every saved sample across the 3 splits) + the model's
        predictions at the same dates. Background shading distinguishes
        train / val / test regions per station.

        Output: plots/delta_gwl/test/diag_worst_full_timeline.png
        """
        import matplotlib.dates as mdates
        from datetime import date as _date

        self.model.eval()
        amp_device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"

        def _run_split(data, sampler, clamp_split=True):
            preds_abs_l, targets_abs_l = [], []
            pred_deltas_l, target_deltas_l = [], []
            well_ids_l, date_ords_l = [], []
            for batch_indices in sampler:
                batch = data.get_batch(batch_indices)
                with autocast(device_type=amp_device_type, enabled=self.use_amp):
                    if self._use_revin:
                        sequence_in, mean_s, std_s = self._apply_revin(batch["sequence"])
                    else:
                        sequence_in = batch["sequence"]
                        mean_s = std_s = None
                    _kw = dict(
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
                    if getattr(self.model, "use_prithvi", False):
                        _kw["tile_idx"] = batch["tile_idx"]
                    gwl_pred = self.model(**_kw)
                    if self._use_revin:
                        gwl_pred = (gwl_pred.squeeze(1) * std_s + mean_s).unsqueeze(1)
                current_gwl_raw = batch["current_gwl_raw"]
                if self.use_delta_gwl:
                    pred_delta = self._inverse_transform_on_device(gwl_pred.squeeze(1).float())
                    pred_abs = current_gwl_raw + pred_delta
                    target_abs = batch["target_gwl_raw"]
                    target_delta = self._inverse_transform_on_device(batch["target_gwl"].float())
                else:
                    pred_abs = self._inverse_transform_on_device(gwl_pred.squeeze(1).float())
                    target_abs = self._inverse_transform_on_device(batch["target_gwl"].float())
                    pred_delta = pred_abs - current_gwl_raw
                    target_delta = target_abs - current_gwl_raw
                if clamp_split:
                    pred_abs, pred_delta = self._clamp_pred_to_current(
                        pred_abs, pred_delta, current_gwl_raw
                    )
                preds_abs_l.append(pred_abs.cpu().numpy())
                targets_abs_l.append(target_abs.cpu().numpy())
                pred_deltas_l.append(pred_delta.cpu().numpy())
                target_deltas_l.append(target_delta.cpu().numpy())
                well_ids_l.append(batch["well_id"].cpu().numpy())
                date_ords_l.append(batch["target_date_ordinal"].cpu().numpy())
            return {
                "preds_abs": np.concatenate(preds_abs_l) if preds_abs_l else np.array([]),
                "targets_abs": np.concatenate(targets_abs_l) if targets_abs_l else np.array([]),
                "pred_deltas": np.concatenate(pred_deltas_l) if pred_deltas_l else np.array([]),
                "target_deltas": np.concatenate(target_deltas_l) if target_deltas_l else np.array([]),
                "well_ids": np.concatenate(well_ids_l) if well_ids_l else np.array([]),
                "date_ords": np.concatenate(date_ords_l) if date_ords_l else np.array([]),
                "idx_to_station": data.idx_to_station,
            }

        train_res = _run_split(train_data, train_sampler, clamp_split=False)
        val_res = _run_split(val_data, val_sampler, clamp_split=True)
        test_res = _run_split(test_data, test_sampler, clamp_split=True)

        # Pull the exact stations selected by `_plot_best_worst_timeseries`
        # for val and test (cached on self._last_worst_stations). This
        # guarantees the full-timeline panels show the SAME stations as the
        # worst columns of diag_{val,test}_best_worst_timeseries.png.
        cached = getattr(self, "_last_worst_stations", {}) or {}
        worst_val = (cached.get("val") or [])[:n_per_split]
        worst_test = (cached.get("test") or [])[:n_per_split]

        if not worst_val and not worst_test:
            print(
                "  No cached worst stations from val/test best-worst plots; "
                "ensure those plots are generated before this one."
            )
            return

        worst_codes_ordered = []
        seen = set()
        for sc, _, _ in worst_val + worst_test:
            if sc not in seen:
                seen.add(sc)
                worst_codes_ordered.append(sc)

        val_r2_map = {sc: r2 for sc, r2, _ in worst_val}
        test_r2_map = {sc: r2 for sc, r2, _ in worst_test}

        def _station_preds(res, sc):
            station_to_idx = {v: k for k, v in res["idx_to_station"].items()}
            wid = station_to_idx.get(sc)
            if wid is None or len(res["well_ids"]) == 0:
                return None
            m = res["well_ids"] == wid
            if not m.any():
                return None
            d = res["date_ords"][m]
            p = res["preds_abs"][m]
            t = res["targets_abs"][m]
            sort_idx = np.argsort(d)
            return d[sort_idx], p[sort_idx], t[sort_idx]

        train_hist = self._collect_per_well_history(train_data, "train_ft")
        val_hist = self._collect_per_well_history(val_data, "val_ft")
        test_hist = self._collect_per_well_history(test_data, "test_ft")

        n_panels = len(worst_codes_ordered)
        fig, axes = plt.subplots(
            n_panels, 1, figsize=(14, 3.0 * n_panels), squeeze=False,
        )

        for i, sc in enumerate(worst_codes_ordered):
            ax = axes[i, 0]
            train_dates = train_hist.get(sc, (np.array([]), np.array([])))
            val_dates = val_hist.get(sc, (np.array([]), np.array([])))
            test_dates = test_hist.get(sc, (np.array([]), np.array([])))

            hx = np.concatenate([train_dates[0], val_dates[0], test_dates[0]])
            hy = np.concatenate([train_dates[1], val_dates[1], test_dates[1]])
            if len(hx) > 0:
                sort_idx = np.argsort(hx)
                hx_sorted, hy_sorted = hx[sort_idx], hy[sort_idx]
                hx_dt = [_date.fromordinal(int(x)) for x in hx_sorted]
                ax.plot(hx_dt, hy_sorted, color="#0D88ED", linewidth=1.2,
                        alpha=0.85, label="Actual")

            for split_name, res, color in [
                ("train", train_res, "#888888"),
                ("val", val_res, "#FF9800"),
                ("test", test_res, "#D32F2F"),
            ]:
                out = _station_preds(res, sc)
                if out is None:
                    continue
                px, py, _ = out
                px_dt = [_date.fromordinal(int(x)) for x in px]
                ax.plot(px_dt, py, "o--", color=color, markersize=3,
                        linewidth=0.9, alpha=0.75,
                        label=f"Pred ({split_name}, n={len(px)})")

            train_end = float(train_dates[0].max()) if len(train_dates[0]) else None
            val_end = float(val_dates[0].max()) if len(val_dates[0]) else None
            if train_end is not None:
                xlim_lo_f, xlim_hi_f = ax.get_xlim()
                # Convert matplotlib's float xlim back to datetime.date for axvspan
                # (axvspan can't mix datetime.date with raw floats).
                xlim_lo_dt = mdates.num2date(xlim_lo_f).date()
                xlim_hi_dt = mdates.num2date(xlim_hi_f).date()
                te_dt = _date.fromordinal(int(train_end))
                ax.axvspan(xlim_lo_dt, te_dt, color="gray", alpha=0.06, zorder=0)
                if val_end is not None and val_end > train_end:
                    ve_dt = _date.fromordinal(int(val_end))
                    ax.axvspan(te_dt, ve_dt, color="gray", alpha=0.14, zorder=0)
                    ax.axvspan(ve_dt, xlim_hi_dt, color="gray", alpha=0.24, zorder=0)
                else:
                    ax.axvspan(te_dt, xlim_hi_dt, color="gray", alpha=0.24, zorder=0)

            vr2 = val_r2_map.get(sc)
            tr2 = test_r2_map.get(sc)
            vr2_s = f"val R²δ={vr2:.3f}" if vr2 is not None else "val R²δ=—"
            tr2_s = f"test R²δ={tr2:.3f}" if tr2 is not None else "test R²δ=—"
            ax.set_title(f"{sc}  |  {vr2_s},  {tr2_s}",
                         fontsize=10, color="#D32F2F")
            ax.set_ylabel("GWL (m)")
            ax.grid(alpha=0.3)
            ax.legend(loc="best", fontsize=7, ncol=2)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

        plt.suptitle(
            f"Worst Wells Full Timeline (val ∪ test, by R²δ) — top {n_per_split} per split",
            fontsize=13, y=1.005,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path("diag_worst_full_timeline.png", "test", is_delta=True),
            dpi=150, bbox_inches="tight",
        )
        plt.close()
        print(f"  Worst-wells full-timeline plot saved ({n_panels} stations)")


    # =====================  _plot_scatter_by_r2_quartile  =====================
    def _plot_scatter_by_r2_quartile(
        self,
        pred_deltas,
        target_deltas,
        well_ids,
        idx_to_station,
        split_name: str = "val",
    ):
        """
        2x2 grid of pred vs actual delta scatter, stratified by R² delta quartile.

        Top-left: best 25% wells (highest R² delta)
        Top-right: 50-75%
        Bottom-left: 25-50%
        Bottom-right: worst 25% (lowest R² delta)

        Reveals failure mode: flat-lining (vertical band), bias (off-diagonal),
        or noise (random scatter).

        Filter wells by canonical train_std threshold (self._min_station_target_std)
        — cold-start excluded since train_std unknown.
        """
        from sklearn.metrics import r2_score

        min_well_samples = 5
        threshold = self._min_station_target_std
        unique_wells = np.unique(well_ids)

        well_r2_d = {}
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            station_code = idx_to_station.get(int(wid))
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                continue  # cold-start excluded
            if train_std < threshold:
                continue
            well_r2_d[wid] = float(r2_score(target_deltas[m], pred_deltas[m]))

        if len(well_r2_d) < 8:
            print(f"  Too few predictable wells for R² quartile scatter ({len(well_r2_d)})")
            return

        sorted_wells = sorted(well_r2_d.keys(), key=lambda w: well_r2_d[w])
        n = len(sorted_wells)
        q1 = sorted_wells[: n // 4]
        q2 = sorted_wells[n // 4 : n // 2]
        q3 = sorted_wells[n // 2 : 3 * n // 4]
        q4 = sorted_wells[3 * n // 4 :]

        # (axis_label, wells, panel_color)
        panels = [
            ("Top 25% (best R²)", q4, "#0D88ED"),
            ("50-75%", q3, "#4CAF50"),
            ("25-50%", q2, "#FF9800"),
            ("Bottom 25% (worst R²)", q1, "#FF5722"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        # Determine common axis limits from all data so panels share scale
        all_t = target_deltas
        all_p = pred_deltas
        lim = float(np.percentile(np.abs(np.concatenate([all_t, all_p])), 99))

        for ax, (label, wells, color) in zip(axes.flatten(), panels):
            mask = np.isin(well_ids, wells)
            t = target_deltas[mask]
            p = pred_deltas[mask]
            r2s = [well_r2_d[w] for w in wells]
            r2_min, r2_max = min(r2s), max(r2s)

            ax.scatter(t, p, s=8, alpha=0.4, color=color, edgecolors="none")
            ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=1,
                    label="y = x")
            ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
            ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("True delta (m)")
            ax.set_ylabel("Predicted delta (m)")
            ax.set_title(
                f"{label}  ({len(wells)} wells, n={mask.sum()} samples)\n"
                f"R² delta range: [{r2_min:.2f}, {r2_max:.2f}]",
                fontsize=10, color=color,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.suptitle(
            f"Predicted vs Actual Delta by R² Quartile — {split_name.capitalize()}",
            fontsize=13, y=1.0,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_scatter_by_r2_quartile.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_scatter_by_std_bucket  =====================
    def _plot_scatter_by_std_bucket(
        self,
        pred_deltas,
        target_deltas,
        well_ids,
        idx_to_station,
        split_name: str = "val",
    ):
        """
        Pred vs actual delta scatter, stratified by canonical train_std per well.

        Buckets: [0-1, 1-3, 3-8, 8-15, 15+]. Cold-start wells (no train_std) fall
        back to eval-sample std for placement; tracked separately in counts but
        kept in panels (still informative).
        """
        from sklearn.metrics import r2_score

        min_well_samples = 5
        unique_wells = np.unique(well_ids)

        # Per-well train_std (canonical) and R²Δ
        well_std = {}
        well_r2_d = {}
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            station_code = idx_to_station.get(int(wid))
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                train_std = float(target_deltas[m].std())
            well_std[wid] = train_std
            well_r2_d[wid] = float(r2_score(target_deltas[m], pred_deltas[m]))

        if len(well_std) < 5:
            print(f"  Too few wells for std-bucket scatter ({len(well_std)})")
            return

        # Bucket edges + labels (closed on left, open on right; last bucket is open-ended)
        bucket_defs = [
            ("0-1m",   0.0,  1.0,  "#FF5722"),
            ("1-3m",   1.0,  3.0,  "#FF9800"),
            ("3-8m",   3.0,  8.0,  "#4CAF50"),
            ("8-15m",  8.0,  15.0, "#0D88ED"),
            ("15m+",   15.0, np.inf, "#673AB7"),
        ]

        # Common axis limits so panels are comparable
        lim = float(np.percentile(np.abs(np.concatenate([target_deltas, pred_deltas])), 99))

        # 2×3 grid (last cell stays empty)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes_flat = axes.flatten()

        for ax, (label, lo, hi, color) in zip(axes_flat, bucket_defs):
            wells_in = [w for w, s in well_std.items() if lo <= s < hi]
            mask = np.isin(well_ids, wells_in)
            t = target_deltas[mask]
            p = pred_deltas[mask]

            if len(wells_in) == 0:
                ax.text(0.5, 0.5, "no wells", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
                ax.set_title(f"std {label}  (0 wells)", fontsize=10, color=color)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            r2_vals = [well_r2_d[w] for w in wells_in]
            median_r2 = float(np.median(r2_vals))
            pct_pos = (np.array(r2_vals) > 0).mean() * 100

            ax.scatter(t, p, s=8, alpha=0.4, color=color, edgecolors="none")
            ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, linewidth=1,
                    label="y = x")
            ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
            ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_xlabel("True delta (m)")
            ax.set_ylabel("Predicted delta (m)")
            ax.set_title(
                f"std {label}  ({len(wells_in)} wells, {mask.sum()} samples)\n"
                f"median per-well R²Δ = {median_r2:.3f}  ({pct_pos:.0f}% > 0)",
                fontsize=10, color=color,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        # Hide the unused last cell
        for extra_ax in axes_flat[len(bucket_defs):]:
            extra_ax.axis("off")

        plt.suptitle(
            f"Predicted vs Actual Delta by std(target_delta) Bucket — {split_name.capitalize()}",
            fontsize=13, y=1.0,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_scatter_by_std_bucket.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_r2_delta_by_attribute  =====================
    def _plot_r2_delta_by_attribute(
        self,
        pred_deltas,
        target_deltas,
        well_ids,
        idx_to_station,
        station_to_state,
        station_to_well_type,
        station_to_aquifer_type,
        split_name: str = "val",
    ):
        """
        2x2 boxplots of per-well R² delta grouped by attributes:
          (state, well_type, aquifer_type, volatility bucket).

        Categories with fewer than min_wells_per_cat are excluded.
        """
        from sklearn.metrics import r2_score

        min_well_samples = 5
        min_wells_per_cat = 5
        unique_wells = np.unique(well_ids)

        threshold = self._min_station_target_std
        rows = []
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            station_code = idx_to_station.get(int(wid), f"well_{int(wid)}")
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                continue  # cold-start excluded — train_std unknown
            if train_std < threshold:
                continue
            r2_d = float(r2_score(target_deltas[m], pred_deltas[m]))
            rows.append({
                "r2": r2_d,
                "std": train_std,
                "state": station_to_state.get(station_code, "unknown"),
                "well_type": station_to_well_type.get(station_code, "unknown"),
                "aquifer_type": station_to_aquifer_type.get(station_code, "unknown"),
            })

        if len(rows) < 10:
            print(f"  Too few predictable wells for attribute boxplots ({len(rows)})")
            return

        # Volatility bucket label
        def vol_bucket(s):
            if s < 1: return "0-1m"
            if s < 3: return "1-3m"
            if s < 8: return "3-8m"
            if s < 15: return "8-15m"
            return "15+m"

        for r in rows:
            r["vol"] = vol_bucket(r["std"])

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))

        def boxplot_by(ax, key, title, ordered_cats=None):
            # Group by key
            groups = {}
            for r in rows:
                groups.setdefault(r[key], []).append(np.clip(r["r2"], -2, 1))
            # Filter to categories with enough wells
            groups = {k: v for k, v in groups.items() if len(v) >= min_wells_per_cat}
            if not groups:
                ax.text(0.5, 0.5, f"No category with ≥{min_wells_per_cat} wells",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
                return
            # Order: by median descending, or use provided order
            if ordered_cats is None:
                cats = sorted(groups.keys(), key=lambda k: -np.median(groups[k]))
            else:
                cats = [c for c in ordered_cats if c in groups]
            data = [groups[c] for c in cats]
            counts = [len(groups[c]) for c in cats]
            labels = [f"{c}\n(n={n})" for c, n in zip(cats, counts)]

            bp = ax.boxplot(data, labels=labels, showmeans=True,
                            patch_artist=True,
                            meanprops=dict(marker='D', markerfacecolor='black',
                                           markeredgecolor='black', markersize=4))
            for patch in bp['boxes']:
                patch.set_facecolor("#cce6ff")
                patch.set_alpha(0.7)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_ylabel("R² delta (clipped to [-2, 1])")
            ax.set_ylim(-2.1, 1.1)
            ax.set_title(title, fontsize=11)
            ax.grid(True, axis="y", alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

        boxplot_by(axes[0, 0], "state", "R² delta by State")
        boxplot_by(axes[0, 1], "well_type", "R² delta by Well Type")
        boxplot_by(axes[1, 0], "aquifer_type", "R² delta by Aquifer Type")
        boxplot_by(
            axes[1, 1], "vol", "R² delta by Volatility Bucket",
            ordered_cats=["0-1m", "1-3m", "3-8m", "8-15m", "15+m"],
        )

        plt.suptitle(
            f"R² Delta by Well Attributes — {split_name.capitalize()} "
            f"({len(rows)} predictable wells)",
            fontsize=13, y=1.0,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_r2_delta_by_attribute.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_vs_baselines  =====================
    def _plot_vs_baselines(
        self,
        pred_deltas,
        target_deltas,
        well_ids,
        date_ordinals,
        split_name: str = "val",
        train_seasonal_means: Optional[Dict] = None,
    ):
        """
        Per-well comparison of model R² delta vs simple baselines:
          - persistence (predict delta=0)
          - seasonal mean: predict the well's calendar-month mean delta
            computed from TRAIN data only (no future leakage). When the
            well/month combo isn't seen in train, falls back to the well's
            train-period overall mean, then to 0.

        Plot: scatter of model R² delta (x) vs best baseline R² delta (y).
        Dots above the diagonal => baseline beats model for that well.
        """
        from sklearn.metrics import r2_score
        from datetime import date as _date

        min_well_samples = 5
        unique_wells = np.unique(well_ids)

        if train_seasonal_means is None:
            train_seasonal_means = {"by_month": {}, "by_well": {}}
        sm_by_month = train_seasonal_means.get("by_month", {})
        sm_by_well = train_seasonal_means.get("by_well", {})

        rows = []
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            t = target_deltas[m]
            p = pred_deltas[m]
            std_t = float(t.std())
            if std_t < 1.0:
                continue
            d_ord = date_ordinals[m]
            months = np.array([
                _date.fromordinal(int(d)).month for d in d_ord
            ])

            # Model R²
            r2_model = float(r2_score(t, p))

            # Persistence: predict 0
            ss_res = float(np.sum((t - 0) ** 2))
            ss_tot = float(np.sum((t - t.mean()) ** 2))
            r2_persistence = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

            # Seasonal mean from TRAIN data only.
            # For each sample, look up (well, target_month) in train.
            # Fallback chain: month-specific → well overall → 0.
            wid_int = int(wid)
            well_fallback = sm_by_well.get(wid_int, 0.0)
            seasonal_pred = np.array([
                sm_by_month.get((wid_int, int(mo)), well_fallback) for mo in months
            ])
            r2_seasonal = float(r2_score(t, seasonal_pred)) if ss_tot > 1e-8 else 0.0

            # Best baseline
            best_baseline = max(r2_persistence, r2_seasonal)

            rows.append({
                "wid": wid,
                "model": r2_model,
                "persistence": r2_persistence,
                "seasonal": r2_seasonal,
                "best_baseline": best_baseline,
                "model_wins": r2_model > best_baseline,
            })

        if len(rows) < 10:
            print(f"  Too few predictable wells for baseline comparison ({len(rows)})")
            return

        n = len(rows)
        n_wins = sum(1 for r in rows if r["model_wins"])
        pct_wins = 100.0 * n_wins / n

        # Clip for plotting (preserve R² < -2 as -2 for display)
        x = np.clip([r["model"] for r in rows], -2, 1)
        y = np.clip([r["best_baseline"] for r in rows], -2, 1)
        wins = np.array([r["model_wins"] for r in rows])

        fig, ax = plt.subplots(figsize=(10, 9))

        # Below-diagonal (model wins) in green, above-diagonal in red
        ax.scatter(x[wins], y[wins], s=18, alpha=0.55, color="#2ecc71",
                   edgecolors="none", label=f"Model wins ({n_wins})")
        ax.scatter(x[~wins], y[~wins], s=18, alpha=0.55, color="#e74c3c",
                   edgecolors="none", label=f"Baseline wins ({n - n_wins})")

        # Diagonal
        ax.plot([-2, 1], [-2, 1], "k--", alpha=0.5, linewidth=1, label="y = x")
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.set_xlim(-2.05, 1.05)
        ax.set_ylim(-2.05, 1.05)
        ax.set_xlabel("Model R² delta")
        ax.set_ylabel("Best baseline R² delta (max of persistence, seasonal mean)")
        ax.set_title(
            f"Model vs Best Baseline per Well — {split_name.capitalize()}\n"
            f"Model beats baseline on {pct_wins:.1f}% of wells ({n_wins}/{n})",
            fontsize=12,
        )
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_model_vs_baselines.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

        # Also print headline summary
        print(f"  Baseline comparison ({split_name}): "
              f"model beats best baseline on {pct_wins:.1f}% of wells "
              f"({n_wins}/{n})")
        print(f"    Median per-well R²: model={np.median([r['model'] for r in rows]):.3f}, "
              f"persistence={np.median([r['persistence'] for r in rows]):.3f}, "
              f"seasonal={np.median([r['seasonal'] for r in rows]):.3f}")


    # =====================  _plot_direction_by_magnitude  =====================
    def _plot_direction_by_magnitude(
        self,
        pred_deltas,
        target_deltas,
        split_name: str = "val",
    ):
        """
        Bar chart of direction prediction accuracy by |target_delta| magnitude.

        Headline trend accuracy hides whether the model predicts direction on
        big (real) changes vs small (noisy) ones. Buckets:
          0-0.5m, 0.5-2m, 2-5m, 5+m
        """
        bucket_edges = [0.0, 0.5, 2.0, 5.0, np.inf]
        bucket_labels = ["0-0.5m", "0.5-2m", "2-5m", "5+m"]
        abs_t = np.abs(target_deltas)

        # sign-correct: same sign or both zero (rare exact 0)
        # treat 0 as positive to avoid degenerate cases
        sgn_t = np.sign(target_deltas)
        sgn_p = np.sign(pred_deltas)
        # Map 0 → +1 to avoid cases where exact zero confuses sign comparison
        sgn_t = np.where(sgn_t == 0, 1, sgn_t)
        sgn_p = np.where(sgn_p == 0, 1, sgn_p)
        correct = sgn_t == sgn_p

        accuracies = []
        counts = []
        ci_widths = []
        for i in range(len(bucket_labels)):
            m = (abs_t >= bucket_edges[i]) & (abs_t < bucket_edges[i + 1])
            n = int(m.sum())
            if n > 0:
                acc = float(correct[m].mean())
                # Approx 95% CI (binomial)
                ci = 1.96 * np.sqrt(acc * (1 - acc) / max(n, 1))
            else:
                acc, ci = 0.0, 0.0
            accuracies.append(acc)
            counts.append(n)
            ci_widths.append(ci)

        overall_acc = float(correct.mean()) if len(correct) else 0.0

        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(bucket_labels))
        colors = ["#bdbdbd", "#9be7b3", "#4abf73", "#1a8a40"]
        bars = ax.bar(x_pos, accuracies, yerr=ci_widths, capsize=6,
                      color=colors, edgecolor="black", linewidth=0.7)

        ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label="Random (50%)")
        ax.axhline(overall_acc, color="black", linestyle=":", linewidth=1, alpha=0.7,
                   label=f"Overall ({overall_acc:.1%})")

        for bar, acc, n in zip(bars, accuracies, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{acc:.1%}\n(n={n:,})",
                    ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(bucket_labels)
        ax.set_xlabel("|target delta| bucket")
        ax.set_ylabel("Direction accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Direction accuracy by |target delta| — {split_name.capitalize()}\n"
            f"Bigger changes are easier to predict directionally",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_direction_by_magnitude.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

        # Print summary
        print(f"  Direction accuracy by |delta| ({split_name}):")
        for label, acc, n in zip(bucket_labels, accuracies, counts):
            print(f"    {label:8s}: {acc:.1%}  (n={n:,})")


    # =====================  _plot_rolling_r2_over_time  =====================
    def _plot_rolling_r2_over_time(
        self,
        pred_deltas,
        target_deltas,
        date_ordinals,
        split_name: str = "val",
        window_days: int = 90,
    ):
        """
        Rolling pooled R² delta over target_date. Reveals temporal drift,
        seasonal performance variation, and concept-shift patterns.
        """
        from sklearn.metrics import r2_score
        import matplotlib.dates as mdates
        from datetime import date as _date

        if len(date_ordinals) < 200:
            print(f"  Too few samples for rolling R² plot ({len(date_ordinals)})")
            return

        sort_idx = np.argsort(date_ordinals)
        d_ord = date_ordinals[sort_idx]
        t = target_deltas[sort_idx]
        p = pred_deltas[sort_idx]

        # Rolling window centred at each unique day
        unique_days = np.unique(d_ord)
        # Sample every ~10 days for speed if range is large
        step = max(1, len(unique_days) // 100)
        sample_days = unique_days[::step]

        centers = []
        r2_vals = []
        n_in_window = []
        half = window_days // 2
        for day in sample_days:
            mask = (d_ord >= day - half) & (d_ord <= day + half)
            if mask.sum() < 30:
                continue
            t_w = t[mask]
            p_w = p[mask]
            if t_w.std() < 1e-6:
                continue
            r2 = r2_score(t_w, p_w)
            centers.append(day)
            r2_vals.append(r2)
            n_in_window.append(int(mask.sum()))

        if len(centers) < 5:
            print(f"  Too few rolling windows for R² over time plot")
            return

        # Convert ordinals to dates
        x = [mdates.date2num(_date.fromordinal(int(d))) for d in centers]
        r2_clipped = np.clip(r2_vals, -1, 1)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, r2_clipped, "-o", markersize=3, linewidth=1.3,
                color="#2c5aa0", alpha=0.85)

        # Shade monsoon months (Jun-Sep) across years
        if x:
            d_min = mdates.num2date(min(x))
            d_max = mdates.num2date(max(x))
            for year in range(d_min.year, d_max.year + 1):
                start = mdates.date2num(_date(year, 6, 1))
                end = mdates.date2num(_date(year, 9, 30))
                ax.axvspan(start, end, color="grey", alpha=0.15)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        ax.set_xlabel("target_date (window center)")
        ax.set_ylabel(f"Rolling R² delta ({window_days}-day window, clipped to [-1, 1])")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(
            f"Rolling R² delta over time — {split_name.capitalize()}\n"
            f"(monsoon Jun-Sep shaded; window={window_days}d)",
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self._plot_path(
                f"diag_{split_name}_rolling_r2_over_time.png",
                split_name, is_delta=True,
            ),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_r2_delta_by_volatility  =====================
    def _plot_r2_delta_by_volatility(
        self,
        pred_deltas,
        target_deltas,
        well_ids,
        idx_to_station,
        split_name: str = "val",
    ):
        """
        Scatter + binned median of R² delta vs canonical train_std per well.

        X-axis: train_std of target delta (well volatility).
        Y-axis: R² delta per well.
        Black line: binned median trend.

        Cold-start wells (no train_std) are excluded — their volatility is unknown.
        """
        from sklearn.metrics import r2_score

        min_well_samples = 5
        unique_wells = np.unique(well_ids)

        stds = []
        r2s = []
        for wid in unique_wells:
            m = well_ids == wid
            if m.sum() < min_well_samples:
                continue
            station_code = idx_to_station.get(int(wid))
            train_std, source = self._get_train_std(station_code)
            if source == "eval":
                continue
            stds.append(train_std)
            r2s.append(float(r2_score(target_deltas[m], pred_deltas[m])))

        if len(stds) < 5:
            print(f"  Too few wells for R² delta by volatility plot ({len(stds)})")
            return

        stds = np.array(stds)
        r2s = np.array(r2s)
        r2s_clip = np.clip(r2s, -2, 1)

        fig, ax = plt.subplots(figsize=(10, 7))

        # Scatter
        sc = ax.scatter(stds, r2s_clip, c=r2s_clip, cmap="RdYlGn", s=20,
                        alpha=0.6, edgecolors="none", vmin=-2, vmax=1)
        plt.colorbar(sc, ax=ax, label="R² delta")

        # Binned median trend line
        n_bins = 15
        bin_edges = np.linspace(stds.min(), stds.max(), n_bins + 1)
        bin_centers = []
        bin_medians = []
        bin_counts = []
        for i in range(n_bins):
            mask = (stds >= bin_edges[i]) & (stds < bin_edges[i + 1])
            if mask.sum() >= 3:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_medians.append(np.median(r2s_clip[mask]))
                bin_counts.append(mask.sum())
        if bin_centers:
            ax.plot(bin_centers, bin_medians, "k-o", linewidth=2, markersize=5,
                    alpha=0.8, label="Binned median")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0.5, color="grey", linewidth=0.5, linestyle=":", alpha=0.4)

        # Bucket boundaries
        for boundary in [1.0, 3.0, 8.0, 15.0]:
            if boundary < stds.max():
                ax.axvline(boundary, color="grey", linewidth=0.5, linestyle=":", alpha=0.4)

        ax.set_xlabel("Std of Target Delta (well volatility, m)")
        ax.set_ylabel("R² Delta (clipped to [-2, 1])")
        ax.set_ylim(-2.1, 1.1)
        ax.set_title(
            f"R² Delta vs Well Volatility — {split_name.capitalize()} ({len(stds)} wells)\n"
            f"Overall median R² delta: {np.median(r2s):.4f}",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_r2_delta_by_volatility.png", split_name, is_delta=True),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_delta_residuals  =====================
    def _plot_delta_residuals(
        self,
        pred_deltas,
        target_deltas,
        split_name: str = "val",
    ):
        """
        1×2 plot analyzing delta prediction quality across the range of true deltas.

        Left: pred_delta vs true_delta scatter (45° line = perfect).
        Right: residual (pred - true) vs true_delta — reveals systematic patterns:
          - Fan shape → model struggles at extremes (regression to mean)
          - Flat near 0 → model is well-calibrated
          - Slope → systematic over/under-prediction at one end

        Highlights extreme tails (top/bottom 5%) in red.
        """
        errors = pred_deltas - target_deltas

        # Identify extreme tails (top/bottom 5% by |true_delta|)
        abs_delta = np.abs(target_deltas)
        p95 = np.percentile(abs_delta, 95)
        is_extreme = abs_delta >= p95
        is_normal = ~is_extreme

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: pred vs true delta scatter
        ax = axes[0]
        ax.scatter(target_deltas[is_normal], pred_deltas[is_normal],
                   s=8, alpha=0.3, c="#4a90d9", edgecolors="none", label="Normal")
        ax.scatter(target_deltas[is_extreme], pred_deltas[is_extreme],
                   s=15, alpha=0.7, c="#e74c3c", edgecolors="none",
                   label=f"Extreme (|delta| >= {p95:.1f}m, top 5%)")
        lo = min(target_deltas.min(), pred_deltas.min())
        hi = max(target_deltas.max(), pred_deltas.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "k--", linewidth=0.8, alpha=0.5, label="Perfect (45°)")
        ax.set_xlabel("True Delta GWL (m)")
        ax.set_ylabel("Predicted Delta GWL (m)")
        ax.set_title("Predicted vs True Delta")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Right: residual vs true delta
        ax = axes[1]
        ax.scatter(target_deltas[is_normal], errors[is_normal],
                   s=8, alpha=0.3, c="#4a90d9", edgecolors="none", label="Normal")
        ax.scatter(target_deltas[is_extreme], errors[is_extreme],
                   s=15, alpha=0.7, c="#e74c3c", edgecolors="none",
                   label=f"Extreme (top 5%)")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        # Add binned mean error line to show trend
        n_bins = 20
        bin_edges = np.linspace(target_deltas.min(), target_deltas.max(), n_bins + 1)
        bin_centers = []
        bin_means = []
        for i in range(n_bins):
            mask = (target_deltas >= bin_edges[i]) & (target_deltas < bin_edges[i + 1])
            if mask.sum() >= 5:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_means.append(errors[mask].mean())
        if bin_centers:
            ax.plot(bin_centers, bin_means, "k-", linewidth=2, alpha=0.8,
                    label="Binned mean error")

        ax.set_xlabel("True Delta GWL (m)")
        ax.set_ylabel("Prediction Error (pred - true) (m)")
        ax.set_title("Residual vs True Delta")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Summary stats for extremes
        if is_extreme.sum() > 0:
            ext_rmse = np.sqrt((errors[is_extreme] ** 2).mean())
            norm_rmse = np.sqrt((errors[is_normal] ** 2).mean())
            ext_bias = errors[is_extreme].mean()
            norm_bias = errors[is_normal].mean()
        else:
            ext_rmse = norm_rmse = ext_bias = norm_bias = 0.0

        plt.suptitle(
            f"Delta Residual Analysis — {split_name.capitalize()} "
            f"({len(target_deltas):,} samples)\n"
            f"Normal: RMSE={norm_rmse:.2f}m, bias={norm_bias:.2f}m | "
            f"Extreme (5%): RMSE={ext_rmse:.2f}m, bias={ext_bias:.2f}m",
            fontsize=11, y=1.02,
        )
        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_delta_residuals.png", split_name, is_delta=True),
            dpi=150, bbox_inches="tight",
        )
        plt.close()


    # =====================  _plot_r2_distribution  =====================
    def _plot_r2_distribution(
        self,
        preds_abs,
        targets_abs,
        well_ids,
        split_name="val",
        filename_suffix="",
        title_suffix="",
    ):
        """
        Histogram of per-well R² values with tier breakdown.

        Tiers: Good (>0.8), Moderate (0.5-0.8), Weak (0-0.5), Failing (<0).
        """
        from sklearn.metrics import r2_score

        unique_wells = np.unique(well_ids)
        min_well_samples = 5
        well_r2s = []
        for wid in unique_wells:
            mask = well_ids == wid
            if mask.sum() < min_well_samples:
                continue
            well_r2s.append(r2_score(targets_abs[mask], preds_abs[mask]))

        if len(well_r2s) < 3:
            print("  Warning: too few wells for R² distribution plot")
            return

        well_r2s = np.array(well_r2s)

        # Tier counts
        n_good = (well_r2s > 0.8).sum()
        n_moderate = ((well_r2s > 0.5) & (well_r2s <= 0.8)).sum()
        n_weak = ((well_r2s >= 0) & (well_r2s <= 0.5)).sum()
        n_failing = (well_r2s < 0).sum()
        n_total = len(well_r2s)
        median_r2 = np.median(well_r2s)

        fig, ax = plt.subplots(figsize=(10, 6))

        # Clip for display (extremely negative R² values compress the histogram)
        display_r2 = np.clip(well_r2s, -2, 1)
        ax.hist(display_r2, bins=50, edgecolor="black", linewidth=0.5,
                color="#4a90d9", alpha=0.8)

        # Median line
        ax.axvline(median_r2, color="red", linestyle="--", linewidth=1.5,
                   label=f"Median: {median_r2:.3f}")

        # Tier boundaries
        for thresh, label in [(0.8, "Good"), (0.5, "Moderate"), (0.0, "Weak/Failing")]:
            ax.axvline(thresh, color="grey", linestyle=":", linewidth=1, alpha=0.6)
            ax.text(thresh, ax.get_ylim()[1] * 0.95, f" {label}", fontsize=7,
                    color="grey", va="top")

        ax.set_xlabel(f"Per-Well R²{title_suffix}")
        ax.set_ylabel("Number of Wells")
        ax.set_title(
            f"{split_name.capitalize()}: Per-Well R²{title_suffix} Distribution ({n_total} wells)\n"
            f"Good (>0.8): {n_good} | Moderate (0.5-0.8): {n_moderate} | "
            f"Weak (0-0.5): {n_weak} | Failing (<0): {n_failing}",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self._plot_path(f"diag_{split_name}_r2_distribution{filename_suffix}.png", split_name, is_delta="_delta" in filename_suffix),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


