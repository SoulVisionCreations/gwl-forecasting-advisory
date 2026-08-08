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
    with ThreadPoolExecutor(max_workers=len(cfgs) if not IS_EVAL else WORKERS) as pool:
        collection_results = list(pool.map(_fetch_collection, cfgs))

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
