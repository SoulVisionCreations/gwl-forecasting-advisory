"""
new_data_fetcher.py — Batch-GEE drop-in replacement for data_fetcher.py

Instead of N stations × M collections GEE round-trips, batches all station
coordinates into a single ee.FeatureCollection and makes ONE getRegion() call
per collection.  Typical reduction: ~100 GEE calls → ~7.

Usage:
    # In inference_engine.py, swap:
    #   from data_fetcher import DataFetcher
    # with:
    #   from new_data_fetcher import DataFetcher
"""

from __future__ import annotations

import logging
import time as _time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import ee
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from inference.acquire.vendor.data_fetcher import (
    DataFetcher as _OriginalDataFetcher,
    GEE_DYNAMIC,
    GEE_ASSETS,
    LOOKBACK_YEARS,
    FORECAST_DAYS,
    WORKERS as _BASE_WORKERS,
    _DYNAMIC_COLS,
    _init_gee,
    _gee_call_with_retry,
    _to_date_str,
    _to_datetime,
    _find_trained_station_match,
    get_forecast_features_with_api,
)

logger = logging.getLogger(__name__)

IS_EVAL = False
WORKERS = _BASE_WORKERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_station_indices(df: pd.DataFrame, stations: list[dict]) -> None:
    """Vectorized nearest-station assignment for getRegion results.

    getRegion returns longitude/latitude columns — match each row to the
    closest station by Euclidean distance in degree-space.  With ≤10 stations
    this is O(rows × stations) but fully vectorized with numpy.
    """
    lats = df["latitude"].values.astype(float)
    lons = df["longitude"].values.astype(float)
    s_lats = np.array([s["lat"] for s in stations])
    s_lons = np.array([s["lon"] for s in stations])
    dlat = lats[:, None] - s_lats[None, :]
    dlon = lons[:, None] - s_lons[None, :]
    df["station_idx"] = np.argmin(dlat**2 + dlon**2, axis=1)


def _make_points_fc(stations: list[dict]):
    """Build an ee.FeatureCollection of Points from station dicts."""
    return ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([s["lon"], s["lat"]])) for s in stations]
    )


def _collection_final_features(cfg) -> set:
    """Final dynamic-feature names a GEE_DYNAMIC collection contributes, AFTER rename +
    the sm/ndvi derivations in _merge_collection_results. Used to skip collections whose
    output the model drops (config-driven)."""
    rename = cfg.get("rename", {})
    bands = list(cfg.get("bands", []))
    feats = set(rename.get(b, b) for b in bands)          # renamed band outputs
    if {"soil_moisture_am", "soil_moisture_pm"} & set(bands):
        feats.add("sm")                                   # smap -> sm (see _merge_collection_results)
    if {"sr_b4", "sr_b5"} & feats:
        feats.add("ndvi")                                 # landsat -> ndvi
    return feats


def _used_dynamic_features(data_config) -> set:
    """Dynamic features the model actually consumes, per its data_config — mirrors
    create_sequence_for_sample: FEATURE_AGG_FUNCTIONS minus 'lulc', minus ndvi/sm when
    drop_ndvi_sm is set. (drop_ndvi_sm defaults True, matching data_preparation.)"""
    from inference.acquire.vendor.data_fetcher import GEE_DYNAMIC as _GD  # noqa: F401 (keep import local)
    try:
        from gwlcore.data_preparation import FEATURE_AGG_FUNCTIONS
        feats = set(FEATURE_AGG_FUNCTIONS.keys()) - {"lulc"}
    except Exception:  # noqa: BLE001 — fall back to the known dynamic feature set
        feats = {"rainfall", "temp", "et", "runoff", "ndvi", "sm"}
    if getattr(data_config, "drop_ndvi_sm", True):
        feats -= {"ndvi", "sm"}
    return feats


# ---------------------------------------------------------------------------
# Batch GEE — dynamic features
# ---------------------------------------------------------------------------


def _batch_fetch_gee_bands(
    collection_id: str,
    bands: list[str],
    scale: float,
    stations: list[dict],
    points_fc,
    start_date: str,
    end_date: str,
    chunk_months: int = 60,
    collection_start: str | None = None,
) -> dict[int, pd.DataFrame]:
    """Fetch bands from one ImageCollection for ALL stations in one GEE call.

    Returns {station_index: DataFrame[date, band1, band2, …]}.
    """
    n = len(stations)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    if collection_start:
        col_start = datetime.strptime(collection_start, "%Y-%m-%d")
        if start < col_start:
            logger.debug(
                "Clamping start %s → %s for %s", start_date, collection_start, collection_id
            )
            start = col_start

    if start >= end:
        return {i: pd.DataFrame(columns=["date"] + bands) for i in range(n)}

    all_chunks: list[pd.DataFrame] = []
    current = start

    while current < end:
        chunk_end = min(current + relativedelta(months=chunk_months), end)
        s_str = current.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")

        try:
            col = (
                ee.ImageCollection(collection_id)
                .filterDate(s_str, e_str)
                .select(bands)
            )
            region_data = _gee_call_with_retry(
                lambda c=col: c.getRegion(points_fc, scale).getInfo()
            )
            if region_data and len(region_data) >= 2:
                df = pd.DataFrame(region_data[1:], columns=region_data[0])
                df["date"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
                all_chunks.append(df)
            else:
                logger.debug(
                    "No batch GEE data: %s %s (%s→%s)", collection_id, bands, s_str, e_str
                )
        except Exception as e:
            logger.warning(
                "Batch GEE chunk failed — %s %s (%s→%s): %s",
                collection_id, bands, s_str, e_str, e,
            )

        current = chunk_end

    if not all_chunks:
        logger.warning("No batch GEE data at all: %s %s", collection_id, bands)
        return {i: pd.DataFrame(columns=["date"] + bands) for i in range(n)}

    merged = pd.concat(all_chunks, ignore_index=True)
    _assign_station_indices(merged, stations)

    result: dict[int, pd.DataFrame] = {}
    for i in range(n):
        sdf = merged[merged["station_idx"] == i].copy()
        if not sdf.empty:
            sdf = sdf.set_index("date")[bands]
            sdf = sdf[~sdf.index.duplicated(keep="first")].sort_index()
            result[i] = sdf.reset_index()
        else:
            result[i] = pd.DataFrame(columns=["date"] + bands)

    return result


def _batch_get_historical_features(
    stations: list[dict],
    points_fc,
    start_date: str,
    end_date: str,
    keep_features: set | None = None,
) -> dict[int, pd.DataFrame]:
    """Batch fetch all dynamic GEE features for every station.

    One getRegion call per collection (chirps, era5, smap, landsat) = 4 calls
    instead of 4 × N.
    """
    _init_gee()
    n = len(stations)

    def _fetch_collection(key_cfg):
        key, cfg = key_cfg
        logger.info(
            "Batch GEE dynamic '%s' bands %s for %d stations (%s→%s)",
            cfg["collection"], cfg["bands"], n, start_date, end_date,
        )
        t0 = _time.time()
        per_station = _batch_fetch_gee_bands(
            cfg["collection"],
            cfg["bands"],
            cfg["scale"],
            stations,
            points_fc,
            start_date,
            end_date,
            chunk_months=cfg.get("chunk_months", 60),
            collection_start=cfg.get("collection_start"),
        )
        if cfg["rename"]:
            for i in range(n):
                if not per_station[i].empty:
                    per_station[i] = per_station[i].rename(columns=cfg["rename"])
        logger.info("Batch GEE dynamic '%s' done: %.1fs", cfg["collection"], _time.time() - t0)
        return per_station

    cfgs = list(GEE_DYNAMIC.items())
    if keep_features is not None:
        # Skip collections whose output features the model drops (e.g. SMAP/sm, Landsat/ndvi
        # when drop_ndvi_sm=True) — nothing downstream reads them, so this is parity-safe.
        cfgs = [(k, c) for (k, c) in cfgs if _collection_final_features(c) & keep_features]
    with ThreadPoolExecutor(max_workers=len(cfgs) if not IS_EVAL else WORKERS) as pool:
        collection_results = list(pool.map(_fetch_collection, cfgs))

    return _merge_collection_results(collection_results, n)


def _merge_collection_results(collection_results, n: int) -> dict[int, pd.DataFrame]:
    """Merge per-collection {station_idx: df} results into per-station dynamic dfs.

    Extracted verbatim from _batch_get_historical_features so the stock path and the
    minimal range-based path (_batch_get_ranges) share EXACTLY the same merge/derive
    logic — guaranteeing identical output columns and values."""
    output: dict[int, pd.DataFrame] = {}
    for i in range(n):
        frames = []
        for coll_result in collection_results:
            sdf = coll_result.get(i, pd.DataFrame())
            if not sdf.empty and "date" in sdf.columns:
                frames.append(sdf.set_index("date"))

        if not frames:
            output[i] = pd.DataFrame(columns=_DYNAMIC_COLS)
            continue

        merged = frames[0].join(frames[1:], how="outer")

        if "soil_moisture_am" in merged.columns and "soil_moisture_pm" in merged.columns:
            merged["sm"] = (merged["soil_moisture_am"] + merged["soil_moisture_pm"]) / 2.0
            merged = merged.drop(columns=["soil_moisture_am", "soil_moisture_pm"])

        if "sr_b4" in merged.columns and "sr_b5" in merged.columns:
            denom = merged["sr_b5"] + merged["sr_b4"]
            merged["ndvi"] = np.where(
                denom == 0, 0, (merged["sr_b5"] - merged["sr_b4"]) / denom
            )
            merged = merged.drop(columns=["sr_b4", "sr_b5"])

        merged["lulc"] = "no_data"
        merged = merged.reset_index()
        existing = [c for c in _DYNAMIC_COLS if c in merged.columns]
        output[i] = merged[existing]

    return output


# ---------------------------------------------------------------------------
# Batch GEE — MINIMAL fetch (Phase 1): only the date ranges the model reads
# (lookback year + per-year climatology horizon slices), not a continuous 12yr
# daily series. Gated by GWL_FETCH_MINIMAL=1 in numeric_fetcher.fetch. Output is
# identical in shape/columns to _batch_get_historical_features; downstream
# aggregation (create_sequence_for_sample / compute_climatology) is UNCHANGED, so
# it reads the same rows and produces the same values — just far less is fetched.
# ---------------------------------------------------------------------------


def _minimal_ranges(current_date, data_config, buffer_days: int = 45, pad_days: int = 7):
    """The minimal (start, end) datetime ranges the model actually reads:

    * lookback: [current - (lookback_total_days + gap_days + buffer), current] — covers
      the 60x6-day OBSERVED sequence create_sequence_for_sample builds.
    * climatology: for each of `climatology_years` prior years (leakage-free, year <
      current.year), the horizon window [datetime(y, target_month, 1), +window_days]
      that compute_climatology reads, padded a few days each side so the half-open
      [end-window, end) slice never clips.

    Ranges are unclamped here; each collection clamps to its own collection_start."""
    dc = data_config
    lb_days = getattr(dc, "lookback_total_days", None)
    if not lb_days:
        lb_days = (int(getattr(dc, "num_timesteps", 60) or 60)
                   * int(getattr(dc, "lookback_window_days", 6) or 6))
    gap = int(getattr(dc, "gap_days", 30) or 30)
    ranges = [(current_date - timedelta(days=int(lb_days) + gap + buffer_days), current_date)]

    horizon_m = int(getattr(dc, "forecast_horizon_months", 3) or 3)
    target_date = current_date + relativedelta(months=horizon_m)
    window_days = (target_date - current_date).days
    clim_years = int(getattr(dc, "climatology_years", 12) or 12)
    tmonth = target_date.month
    for y in range(current_date.year - clim_years, current_date.year):
        anchor = datetime(y, tmonth, 1)
        ranges.append((anchor - timedelta(days=pad_days),
                       anchor + timedelta(days=window_days + pad_days)))
    return ranges


def _batch_fetch_gee_bands_ranges(
    collection_id: str,
    bands: list[str],
    scale: float,
    stations: list[dict],
    points_fc,
    ranges,
    collection_start: str | None = None,
) -> dict[int, pd.DataFrame]:
    """Like _batch_fetch_gee_bands but fetches an explicit list of (start,end) datetime
    ranges (each a small getRegion -> avoids the huge single-call stall) and merges into
    per-station daily dfs (concat + dedup by date, keep first). Same per-station output
    shape as _batch_fetch_gee_bands."""
    n = len(stations)
    col_start = datetime.strptime(collection_start, "%Y-%m-%d") if collection_start else None
    per_station_frames: dict[int, list] = {i: [] for i in range(n)}
    for (rs, re) in ranges:
        s = rs
        if col_start and s < col_start:
            s = col_start
        if s >= re:
            continue
        # chunk_months=12 keeps every getInfo <= ~1yr of daily rows (NDVI-sized).
        r = _batch_fetch_gee_bands(
            collection_id, bands, scale, stations, points_fc,
            s.strftime("%Y-%m-%d"), re.strftime("%Y-%m-%d"),
            chunk_months=12, collection_start=collection_start,
        )
        for i in range(n):
            if not r[i].empty:
                per_station_frames[i].append(r[i])
    result: dict[int, pd.DataFrame] = {}
    for i in range(n):
        if per_station_frames[i]:
            m = pd.concat(per_station_frames[i], ignore_index=True)
            m = (m.drop_duplicates(subset="date", keep="first")
                   .sort_values("date").reset_index(drop=True))
            result[i] = m
        else:
            result[i] = pd.DataFrame(columns=["date"] + bands)
    return result


def _batch_get_ranges(stations: list[dict], points_fc, ranges,
                      keep_features: set | None = None) -> dict[int, pd.DataFrame]:
    """Minimal-fetch analogue of _batch_get_historical_features: fetch ONLY `ranges`
    (lookback year + per-year climatology slices) instead of a continuous 12-year daily
    series. Reuses _merge_collection_results so output is identical to the stock path."""
    _init_gee()
    n = len(stations)

    def _fetch_collection(key_cfg):
        key, cfg = key_cfg
        logger.info(
            "Minimal GEE dynamic '%s' bands %s for %d stations (%d ranges)",
            cfg["collection"], cfg["bands"], n, len(ranges),
        )
        t0 = _time.time()
        per_station = _batch_fetch_gee_bands_ranges(
            cfg["collection"], cfg["bands"], cfg["scale"], stations, points_fc,
            ranges, collection_start=cfg.get("collection_start"),
        )
        if cfg["rename"]:
            for i in range(n):
                if not per_station[i].empty:
                    per_station[i] = per_station[i].rename(columns=cfg["rename"])
        logger.info("Minimal GEE dynamic '%s' done: %.1fs", cfg["collection"], _time.time() - t0)
        return per_station

    cfgs = list(GEE_DYNAMIC.items())
    if keep_features is not None:
        # Skip collections whose output features the model drops (e.g. SMAP/sm, Landsat/ndvi
        # when drop_ndvi_sm=True) — nothing downstream reads them, so this is parity-safe.
        cfgs = [(k, c) for (k, c) in cfgs if _collection_final_features(c) & keep_features]
    with ThreadPoolExecutor(max_workers=len(cfgs) if not IS_EVAL else WORKERS) as pool:
        collection_results = list(pool.map(_fetch_collection, cfgs))

    return _merge_collection_results(collection_results, n)


# ---------------------------------------------------------------------------
# Batch GEE — CLIMATOLOGY-ONLY fetch. With use_only_gwl=True the lookback is
# GWL-only, so the ONLY thing the dynamic fetch feeds is the forecast-window
# climatology (compute_climatology) for the forecast features. This fetches JUST
# those features' bands over JUST the per-year horizon slices — no lookback, no
# unused collections/bands. GWL_FETCH_CLIMATOLOGY_ONLY=1. Config-driven & parity-
# safe: it delivers the exact rows compute_climatology reads.
# ---------------------------------------------------------------------------


def _forecast_features(data_config) -> set:
    """Dynamic features used as FORECAST-window inputs, mirroring
    create_sequence_for_sample's forecast_features selection:
      use_rain_temp -> {rainfall, temp}; else {rainfall,temp,et,runoff} (drop_ndvi_sm)
      or all six."""
    if getattr(data_config, "use_rain_temp", False):
        return {"rainfall", "temp"}
    try:
        from gwlcore.data_preparation import FEATURE_AGG_FUNCTIONS
        feats = set(FEATURE_AGG_FUNCTIONS.keys()) - {"lulc"}
    except Exception:  # noqa: BLE001
        feats = {"rainfall", "temp", "et", "runoff", "ndvi", "sm"}
    if getattr(data_config, "drop_ndvi_sm", True):
        feats -= {"ndvi", "sm"}
    return feats


def _climatology_ranges(current_date, data_config, pad_days: int = 7):
    """Only the per-year horizon slices compute_climatology reads (NO lookback):
    for each prior year (leakage-free), [datetime(y, target_month, 1), +window_days],
    padded a few days so the half-open [end-window, end) slice never clips."""
    horizon_m = int(getattr(data_config, "forecast_horizon_months", 3) or 3)
    target_date = current_date + relativedelta(months=horizon_m)
    window_days = (target_date - current_date).days
    clim_years = int(getattr(data_config, "climatology_years", 12) or 12)
    tmonth = target_date.month
    ranges = []
    for y in range(current_date.year - clim_years, current_date.year):
        anchor = datetime(y, tmonth, 1)
        ranges.append((anchor - timedelta(days=pad_days),
                       anchor + timedelta(days=window_days + pad_days)))
    return ranges


def _feature_source_specs(needed_features: set):
    """Map needed final features -> per-collection fetch specs with ONLY the bands that
    produce a needed feature (band-level selection). Returns list of
    (key, cfg, bands_subset, rename_subset)."""
    specs = []
    for key, cfg in GEE_DYNAMIC.items():
        rename = cfg.get("rename", {})
        bands = [b for b in cfg["bands"] if rename.get(b, b) in needed_features]
        # derived combos need BOTH source bands
        if "sm" in needed_features and {"soil_moisture_am", "soil_moisture_pm"} <= set(cfg["bands"]):
            bands = list(cfg["bands"])
        if "ndvi" in needed_features and {"SR_B4", "SR_B5"} <= set(cfg["bands"]):
            bands = list(cfg["bands"])
        if bands:
            sub_rename = {b: rename[b] for b in bands if b in rename}
            specs.append((key, cfg, bands, sub_rename))
    return specs


def _batch_get_climatology(stations: list[dict], points_fc, ranges,
                           needed_features: set) -> dict[int, pd.DataFrame]:
    """Fetch ONLY `needed_features` (band-level) over the climatology `ranges`, merged
    exactly like the other paths. Same per-station output shape."""
    _init_gee()
    n = len(stations)
    specs = _feature_source_specs(needed_features)

    def _fetch(spec):
        key, cfg, bands, sub_rename = spec
        logger.info("Climatology GEE '%s' bands %s for %d stations (%d slices)",
                    cfg["collection"], bands, n, len(ranges))
        t0 = _time.time()
        per = _batch_fetch_gee_bands_ranges(cfg["collection"], bands, cfg["scale"],
                                            stations, points_fc, ranges,
                                            collection_start=cfg.get("collection_start"))
        if sub_rename:
            for i in range(n):
                if not per[i].empty:
                    per[i] = per[i].rename(columns=sub_rename)
        logger.info("Climatology GEE '%s' done: %.1fs", cfg["collection"], _time.time() - t0)
        return per

    if not specs:
        return {i: pd.DataFrame(columns=_DYNAMIC_COLS) for i in range(n)}
    with ThreadPoolExecutor(max_workers=len(specs)) as pool:
        results = list(pool.map(_fetch, specs))
    return _merge_collection_results(results, n)


# ---------------------------------------------------------------------------
# Batch GEE — fallback rainfall + temp
# ---------------------------------------------------------------------------


def _batch_get_rainfall_temp_history(
    stations: list[dict],
    points_fc,
    start_date: str,
    end_date: str,
) -> dict[int, pd.DataFrame]:
    """Batch fetch CHIRPS rainfall + ERA5 temperature for fallback forecast.

    2 getRegion calls (one per collection) instead of 2 × N.
    """
    _init_gee()
    n = len(stations)
    chirps_cfg = GEE_DYNAMIC["chirps"]
    era5_cfg = GEE_DYNAMIC["era5"]

    def _fetch_chirps():
        return _batch_fetch_gee_bands(
            chirps_cfg["collection"],
            chirps_cfg["bands"],
            chirps_cfg["scale"],
            stations,
            points_fc,
            start_date,
            end_date,
            chirps_cfg["chunk_months"],
            chirps_cfg.get("collection_start"),
        )

    def _fetch_era5():
        return _batch_fetch_gee_bands(
            era5_cfg["collection"],
            ["temperature_2m"],
            era5_cfg["scale"],
            stations,
            points_fc,
            start_date,
            end_date,
            era5_cfg["chunk_months"],
            era5_cfg.get("collection_start"),
        )

    with ThreadPoolExecutor(max_workers=2 if not IS_EVAL else WORKERS) as pool:
        f_c = pool.submit(_fetch_chirps)
        f_e = pool.submit(_fetch_era5)
        chirps_all = f_c.result()
        era5_all = f_e.result()

    output: dict[int, pd.DataFrame] = {}
    for i in range(n):
        c_df = chirps_all.get(
            i, pd.DataFrame(columns=["date"] + chirps_cfg["bands"])
        )
        e_df = era5_all.get(
            i, pd.DataFrame(columns=["date", "temperature_2m"])
        )

        c_df = c_df.rename(columns=chirps_cfg["rename"])
        e_df = e_df.rename(columns={"temperature_2m": "temp"})

        if c_df.empty and e_df.empty:
            output[i] = pd.DataFrame(columns=["date", "rainfall", "temp"])
            continue

        df = (
            c_df.set_index("date")
            .join(e_df.set_index("date"), how="outer")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        output[i] = df

    return output


# ---------------------------------------------------------------------------
# Batch GEE — static features
# ---------------------------------------------------------------------------


def _batch_get_all_static(stations: list[dict], points_fc) -> list[dict]:
    """Fetch ALL static features for ALL stations in ONE GEE call.

    Uses ee.FeatureCollection.map() to run terrain, stream-order, lithology,
    and aquifer lookups server-side for every point in a single round-trip.
    """
    _init_gee()

    terrain_img = ee.Image(GEE_ASSETS["terrain"]).select("constant")
    stream_img = ee.Image(GEE_ASSETS["stream_order"])
    litho_src = ee.FeatureCollection(GEE_ASSETS["lithology"])
    aquifer_src = ee.FeatureCollection(GEE_ASSETS["aquifer"])

    empty_litho = ee.Feature(
        None, {"LITHOLOGIC": "no_data", "SUPERGROUP": "no_data"}
    )
    empty_aquifer = ee.Feature(
        None, {"Major_Aqui": "no_data", "aquifer": "no_data"}
    )

    def _lookup(feature):
        point = feature.geometry()

        elev = terrain_img.reduceRegion(ee.Reducer.first(), point, 30)
        so = stream_img.reduceRegion(ee.Reducer.first(), point, 90)

        lit_col = litho_src.filterBounds(point)
        lit = ee.Feature(
            ee.Algorithms.If(lit_col.size(), lit_col.first(), empty_litho)
        )

        aq_col = aquifer_src.filterBounds(point)
        aq = ee.Feature(
            ee.Algorithms.If(aq_col.size(), aq_col.first(), empty_aquifer)
        )

        return feature.set(
            {
                "elevation": ee.Algorithms.If(
                    elev.get("constant"), elev.get("constant"), 0
                ),
                "stream_order": ee.Algorithms.If(so.get("b1"), so.get("b1"), 0),
                "lithology": lit.get("LITHOLOGIC"),
                "litho_supergroup": lit.get("SUPERGROUP"),
                "aquifer_type": aq.get("Major_Aqui"),
                "aquifer_0_aquifer": aq.get("aquifer"),
            }
        )

    logger.info("Batch static GEE fetch for %d stations (single call)", len(stations))
    t0 = _time.time()
    result_fc = _gee_call_with_retry(lambda: points_fc.map(_lookup).getInfo())
    logger.info("Batch static GEE done: %.1fs", _time.time() - t0)

    statics: list[dict] = []
    for feat in result_fc["features"]:
        props = feat["properties"]
        statics.append(
            {
                "elevation": props.get("elevation", 0) or 0,
                "stream_order": props.get("stream_order", 0) or 0,
                "lithology": props.get("lithology", "no_data") or "no_data",
                "litho_supergroup": props.get("litho_supergroup", "no_data") or "no_data",
                "aquifer_type": props.get("aquifer_type", "no_data") or "no_data",
                "aquifer_0_aquifer": props.get("aquifer_0_aquifer", "no_data") or "no_data",
            }
        )

    return statics


# ---------------------------------------------------------------------------
# DataFetcher — drop-in replacement
# ---------------------------------------------------------------------------


class DataFetcher(_OriginalDataFetcher):
    """Same interface as data_fetcher.DataFetcher but batches GEE calls.

    Inherits station discovery, WRIS API calls, and admin-detail lookup
    from the original class.  Only get_Data is overridden.
    """

    def get_Data(
        self,
        lat: float,
        lon: float,
        current_date: str,
        lookback_years: int = LOOKBACK_YEARS,
        forecast_horizon_months: int = FORECAST_DAYS // 30,
        fallback_years: int = 5,
        min_distance: float = None,
        is_eval=False,
    ) -> tuple[list[dict], list[pd.DataFrame]]:
        global IS_EVAL
        IS_EVAL = is_eval
        self.flat = lat
        self.flon = lon

        target_dt = _to_datetime(current_date)
        start_date = (target_dt - timedelta(days=365 * lookback_years)).strftime(
            "%Y-%m-%d"
        )
        fallback_start = (target_dt - timedelta(days=365 * fallback_years)).strftime(
            "%Y-%m-%d"
        )
        end_date = current_date
        forecast_end = (
            target_dt + timedelta(days=forecast_horizon_months * 30)
        ).strftime("%Y-%m-%d")

        # WRIS GWL needs to be fetched a bit past model_date so that
        # preprocess.find_gwl_with_gap can search ±gap_days symmetrically.
        # Without this widening, the +N day side of the search is always
        # empty (we never fetched those rows), and sparse stations whose
        # nearest reading sits AFTER model_date fail with NO_GWL.
        # GEE dynamic / fallback are deliberately NOT widened — those
        # must remain capped at model_date to avoid future-leakage into
        # the LSTM input window.
        _WRIS_END_BUFFER_DAYS = 35
        try:
            wris_end_date = (
                _to_datetime(end_date) + timedelta(days=_WRIS_END_BUFFER_DAYS)
            ).strftime("%Y-%m-%d")
        except Exception:
            wris_end_date = end_date
        print(f"[DEBUG] new_data_fetcher.get_Data  start_date={start_date}  "
              f"end_date={end_date}  wris_end_date={wris_end_date}")

        # ── station discovery (unchanged) ────────────────────────────────
        state, district = self._get_admin_details(lat, lon)
        # Stash admin for the QUERY lat/lon so the caller can surface it
        # back to the API consumer (district/state of the user-supplied point).
        self.admin = {"state": state, "district": district}

        if min_distance is not None:
            exact_match = None
        else:
            exact_match = _find_trained_station_match(
                lat, lon, state=state, district=district
            )

        if exact_match:
            logger.info(
                "Exact trained-station match: %s at %.4f km — single-station mode",
                exact_match["station_code"],
                exact_match["distance_km"],
            )
            stations = [
                {
                    "station_code": exact_match["station_code"],
                    "lat": exact_match["lat"],
                    "lon": exact_match["lon"],
                }
            ]
        else:
            stations = self.get_nearest_stations(
                lat, lon, n=10, state=state, district=district,
                min_distance=min_distance,
            )
        self.station_info = stations

        if not stations:
            logger.warning("No stations found — returning empty results")
            return [], [], None

        # ── build shared FeatureCollection ───────────────────────────────
        _init_gee()
        points_fc = _make_points_fc(stations)

        # ── parallel: batch GEE + per-station WRIS ───────────────────────
        n_workers = len(stations) + 3  # 3 batch GEE tasks + N WRIS tasks
        logger.info(
            "Batch data fetch: %d stations, %d workers (3 GEE batch + %d WRIS)",
            len(stations), n_workers, len(stations),
        )
        t0 = _time.time()

        with ThreadPoolExecutor(
            max_workers=n_workers if not is_eval else WORKERS
        ) as pool:
            f_dynamic = pool.submit(
                _batch_get_historical_features,
                stations, points_fc, start_date, end_date,
            )
            f_fallback = pool.submit(
                _batch_get_rainfall_temp_history,
                stations, points_fc, fallback_start, end_date,
            )
            f_static = pool.submit(
                _batch_get_all_static, stations, points_fc,
            )

            f_wris = [
                pool.submit(
                    self.download_from_station,
                    s["station_code"], start_date, wris_end_date,
                )
                for s in stations
            ]

            dynamic_results = f_dynamic.result()
            fallback_results = f_fallback.result()
            static_results = f_static.result()
            wris_results = [f.result() for f in f_wris]

        logger.info("Batch data fetch done: %.1fs", _time.time() - t0)

        # ── assemble per-station dicts ───────────────────────────────────
        inputs: list[dict] = []
        for i, station in enumerate(stations):
            scode = station["station_code"]
            slat = station["lat"]
            slon = station["lon"]

            raw = wris_results[i]
            gwl_df = self._parse_gwl_timeseries(
                raw["timeseries"], raw["metadata"], scode
            )

            print(f"gwl_df: ", len(gwl_df))
            dynamic_df = dynamic_results.get(i, pd.DataFrame(columns=_DYNAMIC_COLS))
            if not dynamic_df.empty:
                dynamic_df.insert(0, "station_code", scode)

            fallback_df = fallback_results.get(
                i, pd.DataFrame(columns=["date", "rainfall", "temp"])
            )

            m = raw["metadata"][0] if raw["metadata"] else {}
            well_type = (
                m.get("well_type", m.get("welltype", "no_data")) or "no_data"
            )
            try:
                well_depth = float(
                    m.get("well_depth", m.get("welldepth", m.get("depth", 0))) or 0
                )
            except (TypeError, ValueError):
                well_depth = 0.0

            gee_static = (
                static_results[i] if i < len(static_results) else {}
            )

            static = {
                "station_code": scode,
                "latitude": slat,
                "longitude": slon,
                "well_type": well_type,
                "well_depth": well_depth,
                **gee_static,
            }

            logger.info(
                "[%s] GWL rows: %d | Dynamic rows: %d | Fallback rows: %d",
                scode, len(gwl_df), len(dynamic_df), len(fallback_df),
            )

            inputs.append(
                {
                    "gwl": gwl_df,
                    "dynamic": dynamic_df,
                    "static": static,
                    "rainfall_temp_history": fallback_df,
                }
            )

        # ── forecasts (Open-Meteo, per-station — not GEE) ───────────────
        t0 = _time.time()
        with ThreadPoolExecutor(
            max_workers=len(stations) if not is_eval else WORKERS
        ) as pool:
            future_to_idx = {
                pool.submit(
                    get_forecast_features_with_api,
                    s["lat"], s["lon"], current_date, forecast_end,
                ): i
                for i, s in enumerate(stations)
            }
            forecasts = [None] * len(stations)
            for future in future_to_idx:
                forecasts[future_to_idx[future]] = future.result()

        logger.info(
            "Forecasts for %d stations: %.1fs", len(stations), _time.time() - t0,
        )

        return inputs, forecasts, exact_match
