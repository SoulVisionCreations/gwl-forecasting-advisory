"""
Geographic per-well analysis for GWL forecasting.

Extracted from train.py (Trainer.analyze_well_geography) as part of a
multi-session refactor splitting train.py into focused modules.

Behavior is unchanged from the original method — this is a pure mechanical
move with `self.X` access replaced by explicit function arguments.
"""

from collections import defaultdict
import csv
import os
from typing import Dict, Optional

import numpy as np
import torch
from torch.amp import autocast
import matplotlib.pyplot as plt


def analyze_well_geography(
    model,
    data,
    sampler,
    *,
    use_amp: bool,
    device: str,
    use_delta_gwl: bool,
    inverse_transform_fn,
    apply_test_cap_and_filter_fn,
    plot_path_fn,
    perf_dir: str,
    split_name: str = "val",
    station_to_train_std: dict = None,
    use_revin: bool = False,
    revin_std_floor: float = 0.1,
):
    """Analyze per-well prediction performance by geographic location.

    Computes per-well RMSE and R², then writes:
      - well_performance_<split>.csv
      - state_performance_<split>.csv
      - district_performance_<split>.csv
      - plots/geography/<split>/well_geography_<split>.png (3-panel map)

    Also prints state tiers, worst/best states/districts/wells, and a
    similarity report on the worst stations.

    Args:
        model: PyTorch model in eval mode.
        data: LSTMTensorDataset for the split (provides batches + station maps).
        sampler: BatchSampler iterating over `data`.
        use_amp: Whether to wrap inference in autocast.
        device: e.g., "cuda" or "cpu".
        use_delta_gwl: True if model predicts delta; False if absolute.
        inverse_transform_fn: callable(scaled_tensor) → unscaled_tensor (Trainer._inverse_transform_on_device).
        apply_test_cap_and_filter_fn: callable that filters test set to capped + predictable wells.
        plot_path_fn: callable(filename, split_name=..., category=...) → str path for saving plots.
        perf_dir: directory for performance CSVs.
        split_name: "val" | "test" | etc.
    """
    model.eval()
    amp_device_type = "cuda" if device.startswith("cuda") else "cpu"

    all_preds = []
    all_targets = []
    all_pred_deltas = []
    all_target_deltas = []
    all_well_ids = []
    all_date_ordinals = []

    for batch_indices in sampler:
        batch = data.get_batch(batch_indices)
        with autocast(device_type=amp_device_type, enabled=use_amp):
            if use_revin:
                # RevIN: per-sample InstanceNorm on GWL channel (mirrors Trainer._apply_revin).
                gwl_ch = batch["sequence"][..., 0:1]
                is_present = batch["sequence"][..., 1:2]
                n_present = is_present.sum(dim=1, keepdim=True).clamp(min=1.0)
                mean_s_3d = (gwl_ch * is_present).sum(dim=1, keepdim=True) / n_present
                sq_dev = ((gwl_ch - mean_s_3d) * is_present).pow(2)
                var_s_3d = sq_dev.sum(dim=1, keepdim=True) / n_present
                std_s_3d = (var_s_3d + 1e-5).sqrt().clamp(min=revin_std_floor)
                gwl_normed = ((gwl_ch - mean_s_3d) / std_s_3d) * is_present
                sequence_in = torch.cat([gwl_normed, batch["sequence"][..., 1:]], dim=-1)
                mean_s = mean_s_3d.squeeze(-1).squeeze(-1)
                std_s = std_s_3d.squeeze(-1).squeeze(-1)
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
            if getattr(model, "use_prithvi", False):
                _kw["tile_idx"] = batch["tile_idx"]
            gwl_pred = model(**_kw)
            if use_revin:
                # De-normalize before inverse_transform.
                gwl_pred = (gwl_pred.squeeze(1) * std_s + mean_s).unsqueeze(1)

        current_gwl_raw = batch["current_gwl_raw"]

        if use_delta_gwl:
            pred_delta = inverse_transform_fn(gwl_pred.squeeze(1).float())
            pred_abs = current_gwl_raw + pred_delta
            target_abs = batch["target_gwl_raw"] if data.target_gwl_raw is not None else batch["target_gwl"]
            target_delta = inverse_transform_fn(batch["target_gwl"].float())
        else:
            pred_abs = inverse_transform_fn(gwl_pred.squeeze(1).float())
            target_abs = inverse_transform_fn(batch["target_gwl"].float())
            pred_delta = pred_abs - current_gwl_raw
            target_delta = target_abs - current_gwl_raw

        all_preds.append(pred_abs.cpu())
        all_targets.append(target_abs.cpu())
        all_pred_deltas.append(pred_delta.cpu())
        all_target_deltas.append(target_delta.cpu())
        all_well_ids.append(batch["well_id"].cpu())
        all_date_ordinals.append(batch["target_date_ordinal"].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_pred_deltas = torch.cat(all_pred_deltas).numpy()
    all_target_deltas = torch.cat(all_target_deltas).numpy()
    all_well_ids = torch.cat(all_well_ids).numpy()
    all_date_ordinals = torch.cat(all_date_ordinals).numpy()

    idx_to_station = data.idx_to_station
    if station_to_train_std is None:
        station_to_train_std = {}

    if split_name == "test":
        station_to_state_local = getattr(data, "station_to_state", {}) or {}
        filter_info = apply_test_cap_and_filter_fn(
            all_preds, all_targets,
            all_pred_deltas, all_target_deltas,
            all_well_ids, all_date_ordinals,
            idx_to_station,
            station_to_state=station_to_state_local,
        )
        # Use sorted set (all wells; data prep already capped). CSV includes
        # std_target_delta (train_std) + std_source per row so the user can
        # filter in pandas (df[df.std_target_delta >= 2.0]).
        all_preds = filter_info['capped_preds']
        all_targets = filter_info['capped_targets']
        all_pred_deltas = filter_info['capped_pred_deltas']
        all_target_deltas = filter_info['capped_target_deltas']
        all_well_ids = filter_info['capped_well_ids']

    # Compute per-well metrics
    station_to_latlon = getattr(data, "station_to_latlon", {})
    station_to_state = getattr(data, "station_to_state", {})
    station_to_district = getattr(data, "station_to_district", {})
    station_to_depth = getattr(data, "station_to_depth", {})
    unique_wells = np.unique(all_well_ids)

    min_well_samples = 5
    rows = []
    for wid in unique_wells:
        mask = all_well_ids == wid
        n = mask.sum()
        if n < min_well_samples:
            continue

        preds = all_preds[mask]
        targets = all_targets[mask]
        pred_d = all_pred_deltas[mask]
        target_d = all_target_deltas[mask]
        errors = preds - targets

        rmse = np.sqrt((errors ** 2).mean())
        mae = np.abs(errors).mean()
        ss_res = (errors ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

        ss_res_d = ((pred_d - target_d) ** 2).sum()
        ss_tot_d = ((target_d - target_d.mean()) ** 2).sum()
        r2_delta = 1 - ss_res_d / ss_tot_d if ss_tot_d > 1e-8 else 0.0

        # 50/50 blend of model prediction and persistence:
        # pred_blend = (pred_abs + current_gwl) / 2 = current_gwl + pred_delta / 2
        # If r2_blend > r2, model overshoots magnitude (right direction, too aggressive).
        current_gwl = targets - target_d
        pred_blend = (preds + current_gwl) / 2.0
        ss_res_b = ((pred_blend - targets) ** 2).sum()
        r2_blend = 1 - ss_res_b / ss_tot if ss_tot > 1e-8 else 0.0

        # Persistence prediction: pred = current_gwl (no change). Natural baseline.
        ss_res_p = ((current_gwl - targets) ** 2).sum()  # = sum(target_d**2)
        r2_persist = 1 - ss_res_p / ss_tot if ss_tot > 1e-8 else 0.0

        # Per-well MAPE: mean over samples of |pred - target| / denom
        abs_t = np.abs(targets)
        abs_t_floor = np.maximum(abs_t, 1.0)
        abs_err_pred    = np.abs(preds - targets)
        abs_err_blend   = np.abs(pred_blend - targets)
        abs_err_persist = np.abs(current_gwl - targets)
        mape_pred    = float((abs_err_pred    / (abs_t + 1e-8)).mean())
        mape_blend   = float((abs_err_blend   / (abs_t + 1e-8)).mean())
        mape_persist = float((abs_err_persist / (abs_t + 1e-8)).mean())
        mapef_pred    = float((abs_err_pred    / abs_t_floor).mean())
        mapef_blend   = float((abs_err_blend   / abs_t_floor).mean())
        mapef_persist = float((abs_err_persist / abs_t_floor).mean())

        station_code = idx_to_station.get(int(wid), f"well_{int(wid)}")
        lat, lon = station_to_latlon.get(station_code, (None, None))
        state = station_to_state.get(station_code, "unknown")
        district = station_to_district.get(station_code, "unknown")
        well_depth = station_to_depth.get(station_code, None)

        # Canonical train_std lookup; cold-start (Scenario B) falls back to
        # eval-sample std with std_source="eval" so the row is still usable
        # but flagged.
        if station_code in station_to_train_std:
            std_target_delta = float(station_to_train_std[station_code])
            std_source = "train"
        else:
            std_target_delta = float(target_d.std())
            std_source = "eval"

        rows.append({
            "station_code": station_code,
            "state": state,
            "district": district,
            "lat": lat,
            "lon": lon,
            "well_depth": well_depth,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "r2_delta": r2_delta,
            "r2_blend": r2_blend,
            "r2_persist": r2_persist,
            "mape_pred":    mape_pred,
            "mape_blend":   mape_blend,
            "mape_persist": mape_persist,
            "mapef_pred":    mapef_pred,
            "mapef_blend":   mapef_blend,
            "mapef_persist": mapef_persist,
            "n_samples": int(n),
            "mean_target": targets.mean(),
            "std_target": targets.std(),
            "std_target_delta": std_target_delta,
            "std_source": std_source,
        })

    if not rows:
        print(f"  No wells with enough samples for geographic analysis ({split_name})")
        return

    csv_path = os.path.join(perf_dir, f"well_performance_{split_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def _aggregate_by(rows, key):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        agg_rows = []
        for name, wells in groups.items():
            rmses = [w["rmse"] for w in wells]
            r2s = [w["r2"] for w in wells]
            r2_deltas = [w["r2_delta"] for w in wells]
            r2_blends = [w["r2_blend"] for w in wells]
            r2_persists = [w["r2_persist"] for w in wells]
            def _q25(arr): return float(np.quantile(arr, 0.25))
            def _q75(arr): return float(np.quantile(arr, 0.75))
            def _mm(field, clip=5.0):
                arr = np.array([w[field] for w in wells])
                return (
                    _q25(arr), float(np.median(arr)), _q75(arr),
                    float(np.clip(arr, 0.0, clip).mean()),
                )
            mape_pred_q25, mape_pred_med, mape_pred_q75, mape_pred_mean = _mm("mape_pred")
            mape_blend_q25, mape_blend_med, mape_blend_q75, mape_blend_mean = _mm("mape_blend")
            mape_persist_q25, mape_persist_med, mape_persist_q75, mape_persist_mean = _mm("mape_persist")
            mapef_pred_q25, mapef_pred_med, mapef_pred_q75, mapef_pred_mean = _mm("mapef_pred")
            mapef_blend_q25, mapef_blend_med, mapef_blend_q75, mapef_blend_mean = _mm("mapef_blend")
            mapef_persist_q25, mapef_persist_med, mapef_persist_q75, mapef_persist_mean = _mm("mapef_persist")
            agg_rows.append({
                key: name,
                "n_wells": len(wells),
                "n_samples": sum(w["n_samples"] for w in wells),
                "rmse_median": np.median(rmses),
                "rmse_mean": np.mean(rmses),
                "rmse_max": np.max(rmses),
                "r2_q25": _q25(r2s), "r2_median": np.median(r2s), "r2_q75": _q75(r2s),
                "r2_mean": np.mean(r2s),
                "r2_min": np.min(r2s),
                "r2_delta_q25": _q25(r2_deltas), "r2_delta_median": np.median(r2_deltas), "r2_delta_q75": _q75(r2_deltas),
                "r2_delta_mean": np.mean(r2_deltas),
                "r2_blend_q25": _q25(r2_blends), "r2_blend_median": np.median(r2_blends), "r2_blend_q75": _q75(r2_blends),
                "r2_blend_mean": np.mean(r2_blends),
                "r2_persist_q25": _q25(r2_persists), "r2_persist_median": np.median(r2_persists), "r2_persist_q75": _q75(r2_persists),
                "r2_persist_mean": np.mean(r2_persists),
                "mape_pred_q25": mape_pred_q25, "mape_pred_median": mape_pred_med, "mape_pred_q75": mape_pred_q75, "mape_pred_mean": mape_pred_mean,
                "mape_blend_q25": mape_blend_q25, "mape_blend_median": mape_blend_med, "mape_blend_q75": mape_blend_q75, "mape_blend_mean": mape_blend_mean,
                "mape_persist_q25": mape_persist_q25, "mape_persist_median": mape_persist_med, "mape_persist_q75": mape_persist_q75, "mape_persist_mean": mape_persist_mean,
                "mapef_pred_q25": mapef_pred_q25, "mapef_pred_median": mapef_pred_med, "mapef_pred_q75": mapef_pred_q75, "mapef_pred_mean": mapef_pred_mean,
                "mapef_blend_q25": mapef_blend_q25, "mapef_blend_median": mapef_blend_med, "mapef_blend_q75": mapef_blend_q75, "mapef_blend_mean": mapef_blend_mean,
                "mapef_persist_q25": mapef_persist_q25, "mapef_persist_median": mapef_persist_med, "mapef_persist_q75": mapef_persist_q75, "mapef_persist_mean": mapef_persist_mean,
            })
        return agg_rows

    state_rows = _aggregate_by(rows, "state")
    state_csv_path = os.path.join(perf_dir, f"state_performance_{split_name}.csv")
    with open(state_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=state_rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(state_rows, key=lambda r: r["rmse_median"], reverse=True))

    district_rows = _aggregate_by(rows, "district")
    district_csv_path = os.path.join(perf_dir, f"district_performance_{split_name}.csv")
    with open(district_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=district_rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(district_rows, key=lambda r: r["rmse_median"], reverse=True))

    # State tiers
    MIN_WELLS_FOR_TIER = 5
    tierable_states = [r for r in state_rows if r["n_wells"] >= MIN_WELLS_FOR_TIER]
    small_states = [r for r in state_rows if r["n_wells"] < MIN_WELLS_FOR_TIER]

    if len(tierable_states) >= 3:
        sorted_for_tier = sorted(tierable_states, key=lambda r: r["rmse_median"])
        n_tier = len(sorted_for_tier)
        p25 = max(1, int(n_tier * 0.25))
        p75 = min(n_tier - 1, int(n_tier * 0.75))

        for i, r in enumerate(sorted_for_tier):
            if i < p25:
                r["tier"] = "Good"
            elif i < p75:
                r["tier"] = "Moderate"
            else:
                r["tier"] = "Poor"
    else:
        for r in tierable_states:
            r["tier"] = "Unclassified"

    for r in small_states:
        r["tier"] = "Too few wells"

    with open(state_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=state_rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(state_rows, key=lambda r: r["rmse_median"], reverse=True))

    rmses = [r["rmse"] for r in rows]
    r2s = [r["r2"] for r in rows]
    n_wells = len(rows)

    print(f"\n  Geographic well analysis ({split_name}): "
          f"{n_wells} wells, {len(state_rows)} states, {len(district_rows)} districts")
    print(f"    RMSE — median: {np.median(rmses):.2f}m, "
          f"mean: {np.mean(rmses):.2f}m, max: {np.max(rmses):.2f}m")
    print(f"    R²   — median: {np.median(r2s):.3f}, "
          f"mean: {np.mean(r2s):.3f}, min: {np.min(r2s):.3f}")

    if len(tierable_states) >= 3:
        print(f"\n    State performance tiers ({split_name}, states with >= {MIN_WELLS_FOR_TIER} wells):")
        for tier in ["Good", "Moderate", "Poor"]:
            tier_states = sorted(
                [r for r in state_rows if r.get("tier") == tier],
                key=lambda r: r["rmse_median"],
            )
            if not tier_states:
                continue
            tier_rmses = [r["rmse_median"] for r in tier_states]
            tier_r2s = [r["r2_median"] for r in tier_states]
            state_list = ", ".join(
                f"{r['state']} ({r['n_wells']})" for r in tier_states
            )
            print(f"      {tier:10s} — {len(tier_states)} states | "
                  f"median RMSE: {np.median(tier_rmses):.2f}m | "
                  f"median R²: {np.median(tier_r2s):.3f}")
            print(f"        {state_list}")

        if small_states:
            small_names = ", ".join(
                f"{r['state']} ({r['n_wells']})" for r in small_states
            )
            print(f"      Excluded (< {MIN_WELLS_FOR_TIER} wells): {small_names}")

    reliable_states = [r for r in state_rows if r["n_wells"] >= MIN_WELLS_FOR_TIER]
    sorted_states = sorted(reliable_states, key=lambda r: r["rmse_median"], reverse=True)
    print(f"\n    Worst 5 states by median RMSE (>= {MIN_WELLS_FOR_TIER} wells):")
    for r in sorted_states[:5]:
        print(f"      {r['state']:25s} | wells={r['n_wells']:3d} | "
              f"RMSE={r['rmse_median']:7.2f}m (median) | "
              f"R²={r['r2_median']:6.3f} (median)")

    best_states = sorted(reliable_states, key=lambda r: r["rmse_median"])
    print(f"\n    Best 5 states by median RMSE (>= {MIN_WELLS_FOR_TIER} wells):")
    for r in best_states[:5]:
        print(f"      {r['state']:25s} | wells={r['n_wells']:3d} | "
              f"RMSE={r['rmse_median']:7.2f}m (median) | "
              f"R²={r['r2_median']:6.3f} (median)")

    MIN_WELLS_FOR_DISTRICT = 3
    reliable_districts = [r for r in district_rows if r["n_wells"] >= MIN_WELLS_FOR_DISTRICT]
    sorted_districts = sorted(reliable_districts, key=lambda r: r["rmse_median"], reverse=True)
    print(f"\n    Worst 10 districts by median RMSE (>= {MIN_WELLS_FOR_DISTRICT} wells):")
    for r in sorted_districts[:10]:
        print(f"      {r['district']:25s} | wells={r['n_wells']:3d} | "
              f"RMSE={r['rmse_median']:7.2f}m (median) | "
              f"R²={r['r2_median']:6.3f} (median)")

    best_districts = sorted(reliable_districts, key=lambda r: r["rmse_median"])
    print(f"\n    Best 10 districts by median RMSE (>= {MIN_WELLS_FOR_DISTRICT} wells):")
    for r in best_districts[:10]:
        print(f"      {r['district']:25s} | wells={r['n_wells']:3d} | "
              f"RMSE={r['rmse_median']:7.2f}m (median) | "
              f"R²={r['r2_median']:6.3f} (median)")

    sorted_rows = sorted(rows, key=lambda r: r["rmse"], reverse=True)
    print(f"\n    Worst 10 wells by RMSE:")
    for r in sorted_rows[:10]:
        print(f"      {r['station_code']:20s} | {r['state']:15s} | {r['district']:20s} | "
              f"RMSE={r['rmse']:7.2f}m | R²={r['r2']:6.3f} | n={r['n_samples']}")

    # Worst-station similarity report
    MIN_SAMPLES_SIMILARITY = 50
    eligible_rows = [r for r in rows if r["n_samples"] >= MIN_SAMPLES_SIMILARITY]
    if len(eligible_rows) < 2:
        print(f"\n  Warning: too few wells with >= {MIN_SAMPLES_SIMILARITY} samples "
              f"for similarity analysis (found {len(eligible_rows)})")
        eligible_rows = rows
    N_WORST = min(10, len(eligible_rows))
    worst_by_r2 = sorted(eligible_rows, key=lambda r: r["r2"])[:N_WORST]

    print(f"\n  ── Worst {N_WORST} Stations by R²: Similarity Analysis ─────────────────")

    state_counts = defaultdict(int)
    for r in worst_by_r2:
        state_counts[r["state"]] += 1
    print(f"    State distribution:")
    for state, cnt in sorted(state_counts.items(), key=lambda x: -x[1]):
        print(f"      {state:25s}: {cnt}/{N_WORST} wells")

    worst_mean_gwl = np.mean([r["mean_target"] for r in worst_by_r2])
    worst_std_gwl = np.mean([r["std_target"] for r in worst_by_r2])
    all_mean_gwl = np.mean([r["mean_target"] for r in rows])
    all_std_gwl = np.mean([r["std_target"] for r in rows])
    print(f"    Mean GWL level   — worst: {worst_mean_gwl:7.2f}m  |  all wells: {all_mean_gwl:7.2f}m")
    print(f"    GWL variability  — worst: {worst_std_gwl:7.2f}m  |  all wells: {all_std_gwl:7.2f}m  "
          f"({'more variable' if worst_std_gwl > all_std_gwl else 'less variable'} than avg)")

    worst_lats = [r["lat"] for r in worst_by_r2 if r["lat"] is not None]
    worst_lons = [r["lon"] for r in worst_by_r2 if r["lon"] is not None]
    all_lats = [r["lat"] for r in rows if r["lat"] is not None]
    all_lons = [r["lon"] for r in rows if r["lon"] is not None]
    if worst_lats and all_lats:
        w_lat_rng = max(worst_lats) - min(worst_lats)
        w_lon_rng = max(worst_lons) - min(worst_lons)
        a_lat_rng = max(all_lats) - min(all_lats)
        a_lon_rng = max(all_lons) - min(all_lons)
        pct_lat = 100 * w_lat_rng / a_lat_rng if a_lat_rng > 0 else 0
        pct_lon = 100 * w_lon_rng / a_lon_rng if a_lon_rng > 0 else 0
        clustered = pct_lat < 40 or pct_lon < 40
        spread_arrow = "CLUSTERED" if clustered else "distributed"
        print(f"    Geographic spread -- lat: {w_lat_rng:.2f} deg ({pct_lat:.0f}% of total), "
              f"lon: {w_lon_rng:.2f} deg ({pct_lon:.0f}% of total)"
              f"  -> {spread_arrow}")

    header = f"    {'Station':20s} {'State':20s} {'District':20s} {'R2':6s} {'Mean GWL':9s} {'Std GWL':8s} {'n':5s}"
    print(f"\n{header}")
    sep = "    " + "-" * 95
    print(sep)
    for r in worst_by_r2:
        print(f"    {r['station_code']:20s} {r['state']:20s} {r['district']:20s} "
              f"{r['r2']:6.3f} {r['mean_target']:9.2f}m {r['std_target']:8.2f}m {r['n_samples']:5d}")

    # Plots
    wells_with_coords = [r for r in rows if r["lat"] is not None and r["lon"] is not None]
    if len(wells_with_coords) < 5:
        print(f"    Skipping map plot — too few wells with coordinates")
        return

    lats = [r["lat"] for r in wells_with_coords]
    lons = [r["lon"] for r in wells_with_coords]
    rmse_vals = [r["rmse"] for r in wells_with_coords]
    r2_vals = [r["r2"] for r in wells_with_coords]

    fig, axes = plt.subplots(1, 3, figsize=(26, 8))

    sc1 = axes[0].scatter(lons, lats, c=rmse_vals, cmap="RdYlGn_r",
                          s=15, alpha=0.7, edgecolors="none")
    plt.colorbar(sc1, ax=axes[0], label="RMSE (m)")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].set_title(f"{split_name.capitalize()}: Per-Well RMSE ({n_wells} wells)")
    axes[0].grid(True, alpha=0.3)

    sc2 = axes[1].scatter(lons, lats, c=r2_vals, cmap="RdYlGn",
                          s=15, alpha=0.7, edgecolors="none",
                          vmin=-1, vmax=1)
    plt.colorbar(sc2, ax=axes[1], label="R²")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_title(f"{split_name.capitalize()}: Per-Well R² ({n_wells} wells)")
    axes[1].grid(True, alpha=0.3)

    eligible_coords = [r for r in wells_with_coords if r["n_samples"] >= MIN_SAMPLES_SIMILARITY]
    if len(eligible_coords) < 4:
        eligible_coords = wells_with_coords
    N_HIGHLIGHT = min(10, len(eligible_coords) // 2)
    sorted_by_r2 = sorted(eligible_coords, key=lambda r: r["r2"])
    worst_hl = sorted_by_r2[:N_HIGHLIGHT]
    best_hl = sorted_by_r2[-N_HIGHLIGHT:]

    axes[2].scatter(
        lons, lats,
        c="#cccccc", s=10, alpha=0.5, label="All wells",
    )
    axes[2].scatter(
        [r["lon"] for r in best_hl],
        [r["lat"] for r in best_hl],
        c="#2ecc71", s=80, marker="*", zorder=5, label=f"Best {N_HIGHLIGHT} R²",
    )
    axes[2].scatter(
        [r["lon"] for r in worst_hl],
        [r["lat"] for r in worst_hl],
        c="#e74c3c", s=80, marker="X", zorder=5, label=f"Worst {N_HIGHLIGHT} R²",
    )
    for r in worst_hl:
        axes[2].annotate(
            r["station_code"], (r["lon"], r["lat"]),
            fontsize=6, color="#c0392b",
            xytext=(4, 4), textcoords="offset points",
        )
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].set_title(
        f"{split_name.capitalize()}: Worst vs Best {N_HIGHLIGHT} Stations by R²"
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        plot_path_fn(f"well_geography_{split_name}.png", split_name=split_name, category="geography"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"    Saved: performance/well_performance_{split_name}.csv, "
          f"performance/state_performance_{split_name}.csv, "
          f"performance/district_performance_{split_name}.csv, "
          f"plots/geography/{split_name}/well_geography_{split_name}.png")
