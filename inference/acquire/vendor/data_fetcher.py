from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Union

import ee
import numpy as np
import pandas as pd
import requests
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOOKBACK_YEARS: int = 1
FORECAST_DAYS: int = 90

WORKERS = 1
IS_EVAL = False
GEE_PROJECT = os.environ.get("GEE_PROJECT") or os.environ.get("GWL_GEE_PROJECT") or ""

# Dynamic feature collections — grouped by source so multi-band fetches are batched
GEE_DYNAMIC = {
    "chirps": {
        "collection": "UCSB-CHG/CHIRPS/DAILY",
        "bands": ["precipitation"],
        "scale": 5566,
        "rename": {"precipitation": "rainfall"},
        "chunk_months": 60,
        "collection_start": "1981-01-01",  # CHIRPS starts 1981
    },
    "era5": {
        "collection": "ECMWF/ERA5_LAND/DAILY_AGGR",
        "bands": ["temperature_2m", "total_evaporation_sum", "runoff_sum"],
        "scale": 11132,
        "rename": {
            "temperature_2m": "temp",
            "total_evaporation_sum": "et",
            "runoff_sum": "runoff",
        },
        "chunk_months": 60,
        "collection_start": "1950-01-01",  # ERA5-Land starts 1950
    },
    "smap": {
        "collection": "NASA/SMAP/SPL3SMP_E/006",
        "bands": ["soil_moisture_am", "soil_moisture_pm"],
        "scale": 9000,
        "rename": {},
        "chunk_months": 60,
        "collection_start": "2015-04-01",  # SMAP launched April 2015
    },
    # "modis_ndvi": {
    #     "collection":       "MODIS/061/MOD13A2",
    #     "bands":            ["NDVI"],
    #     "scale":            1000,
    #     "rename":           {"NDVI": "ndvi"},
    #     "chunk_months":     60,
    #     "collection_start": "2000-02-18",
    # },
    "landsat": {
        "collection": "LANDSAT/LC08/C02/T1_L2",
        "bands": ["SR_B4", "SR_B5"],
        "scale": 500,  # FIX: was 30 — caused "User memory limit exceeded"
        "rename": {"SR_B4": "sr_b4", "SR_B5": "sr_b5"},
        "chunk_months": 60,  # Landsat has many scenes; use smaller chunks
        "collection_start": "2013-04-11",  # Landsat-8 launch date
    },
}

# Static GEE assets (from gee_data_extraction.py)
GEE_ASSETS = {
    "terrain": "projects/corestack-datasets/assets/datasets/terrain/pan_india_terrain_raster",
    "stream_order": "projects/corestack-datasets/assets/datasets/Stream_Order_Raster_India",
    "lithology": "projects/ext-datasets/assets/datasets/Lithology_pan_india",
    "aquifer": "projects/ext-datasets/assets/datasets/Aquifer_vector",
}

WRIS_BASE_URL = "https://indiawris.gov.in"
WRIS_DATASET_CODE = "GWATERLVL"
WRIS_HEADERS = {
    "Content-Type": "application/json",
    "Origin": "https://indiawris.gov.in",
    "Referer": "https://indiawris.gov.in/dataSet/",
}

# WRIS hardening for inference (the live service can't tolerate a single slow call
# silently yielding no GWL). Longer timeout + exponential-backoff retries.
_WRIS_TIMEOUT = 60      # was a single 30s shot
_WRIS_MAX_RETRIES = 4   # total attempts per endpoint
_WRIS_BACKOFF = 3       # seconds; waits 3, 6, 12s between attempts

CORESTACK_BASE_URL = "https://geoserver.core-stack.org/api/v1"
CORESTACK_API_KEY = os.environ.get("CORESTACK_API_KEY", "")

# GWL history CSV (data_with_flag_all.csv) — the SAME dataset training uses, needed here for the WRIS
# station-index / 10-yr lookback. Env-driven (AIKosh download -> set the path); NO machine-specific
# default. GWL_STATION_CSV, else the shared GWL_DATA_CSV, else "" (a clear failure only if actually used).
STATION_CSV_PATH = os.environ.get("GWL_STATION_CSV") or os.environ.get("GWL_DATA_CSV") or ""


from inference.acquire.vendor._geo_utils import compute_distances, find_k_nearest


# Load unique station index once at import time (lat, lon, state, district only)
def _load_station_index(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            csv_path,
            usecols=["station_code", "latitude", "longitude", "state", "district"],
        )
        df = df.drop_duplicates(subset="station_code").reset_index(drop=True)
        df["state_lower"] = df["state"].str.lower().str.strip()
        df["district_lower"] = df["district"].str.lower().str.strip()
        logger.info(
            "Loaded station index: %d unique stations from %s", len(df), csv_path
        )
        return df
    except Exception as e:
        logger.error("Failed to load station index from %s: %s", csv_path, e)
        return pd.DataFrame(
            columns=[
                "station_code",
                "latitude",
                "longitude",
                "state",
                "district",
                "state_lower",
                "district_lower",
            ]
        )


_STATION_INDEX: pd.DataFrame = None  # lazy-loaded on first use

_TRAINED_STATION_INDEX: pd.DataFrame = None  # deduplicated lat/lon from training data
_EXACT_MATCH_THRESHOLD_KM = 0.2


def _load_trained_station_index() -> pd.DataFrame:
    global _TRAINED_STATION_INDEX
    if _TRAINED_STATION_INDEX is not None:
        return _TRAINED_STATION_INDEX
    try:
        df = (
            pd.read_csv(
                STATION_CSV_PATH,
                usecols=["station_code", "latitude", "longitude", "state", "district"],
            )
            .drop_duplicates(subset="station_code")
            .reset_index(drop=True)
        )
        df["state_lower"] = df["state"].str.lower().str.strip()
        df["district_lower"] = df["district"].str.lower().str.strip()
        logger.info("Loaded trained station index: %d stations", len(df))
        _TRAINED_STATION_INDEX = df
        return df
    except Exception as e:
        logger.error("Failed to load trained station index: %s", e)
        return pd.DataFrame(
            columns=[
                "station_code",
                "latitude",
                "longitude",
                "state",
                "district",
                "state_lower",
                "district_lower",
            ]
        )


def _find_trained_station_match(
    lat: float, lon: float, state: str = "", district: str = ""
) -> dict | None:
    """Return the nearest trained station within _EXACT_MATCH_THRESHOLD_KM, else None.

    Tries district → state → global, accepting the first subset that produces a
    within-threshold match.  This handles the (common) case where the admin API
    labels a border station with a different district than the catalog, e.g.
    admin says "TIRUPPUR" while data_with_flag_all.csv has it under "Erode" —
    a hard district filter would silently miss the station and force fallback
    to interpolation.  State subsets are ~500-2000 rows so the extra haversine
    pass is microseconds.
    """
    df = _load_trained_station_index()
    if df.empty:
        return None

    candidates: list[pd.DataFrame] = []
    if district:
        candidates.append(df[df["district_lower"] == district.lower().strip()])
    if state:
        candidates.append(df[df["state_lower"] == state.lower().strip()])
    candidates.append(df)  # last resort: full catalog

    for subset in candidates:
        if subset.empty:
            continue
        station_list = [
            {"lat": float(r["latitude"]), "lon": float(r["longitude"])}
            for r in subset[["latitude", "longitude"]].to_dict("records")
        ]
        distances = compute_distances(lat, lon, station_list)
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])
        if min_dist <= _EXACT_MATCH_THRESHOLD_KM:
            row = subset.iloc[min_idx]
            return {
                "station_code": str(row["station_code"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
                "distance_km": round(min_dist, 4),
            }
    return None


_DYNAMIC_COLS = ["date", "rainfall", "temp", "et", "runoff", "ndvi", "sm", "lulc"]
_STATIC_KEYS = [
    "station_code",
    "latitude",
    "longitude",
    "elevation",
    "stream_order",
    "well_type",
    "well_depth",
    "lithology",
    "litho_supergroup",
    "aquifer_type",
    "aquifer_0_aquifer",
]

# Caps concurrent GEE HTTP requests to stay within the urllib3 pool size (10).
# Without this, parallel station + collection threads saturate the pool and spam warnings.
_GEE_SEM = threading.Semaphore(8)

_GEE_MAX_RETRIES = 5


def _gee_call_with_retry(fn):
    """
    Call fn() (a GEE .getInfo() lambda) with exponential backoff on rate-limit errors.
    Waits 1, 2s between attempts. Raises on final failure.
    """
    logger.info("GEE call initiated (max_retries=%d)", _GEE_MAX_RETRIES)
    for attempt in range(_GEE_MAX_RETRIES):
        try:
            with _GEE_SEM:
                return fn()
        except Exception as e:
            msg = str(e).lower()
            # Retry on rate-limits AND transient network/server errors (not just 429).
            transient = any(k in msg for k in (
                "too many requests", "429", "timed out", "timeout", "deadline",
                "connection", "reset", "temporarily", "unavailable", "ssl",
                "500", "502", "503", "504",
            ))
            if transient and attempt < _GEE_MAX_RETRIES - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8s
                logger.warning(
                    "GEE transient error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, _GEE_MAX_RETRIES, str(e)[:80], wait,
                )
                _time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# GEE initialisation
# ---------------------------------------------------------------------------
_gee_initialised = False

# HIGH-VOLUME endpoint for the numeric getRegion/getInfo calls — built for automated heavy access and far
# less throttled than the STANDARD endpoint on a noncommercial project (the standard endpoint is where heavy
# getRegion calls stall/queue). Same endpoint the composite downloader already uses.
_HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"
# Per-request deadline (seconds) so a slow/throttled getRegion FAILS FAST (raises -> retryable) instead of
# hanging for minutes. Tunable via GWL_GEE_TIMEOUT_S.
try:
    _GEE_TIMEOUT_S = int(os.environ.get("GWL_GEE_TIMEOUT_S", "120"))
except ValueError:
    _GEE_TIMEOUT_S = 120


def _init_gee() -> None:
    global _gee_initialised
    if _gee_initialised:
        return

    def _do_init():
        if GEE_PROJECT:
            ee.Initialize(project=GEE_PROJECT, opt_url=_HIGH_VOLUME)
        else:
            ee.Initialize(opt_url=_HIGH_VOLUME)

    try:
        _do_init()
    except ee.EEException:
        ee.Authenticate()
        _do_init()
    # Bound every GEE request so a throttled getRegion times out (retryable) rather than blocking.
    try:
        ee.data.setDeadline(_GEE_TIMEOUT_S * 1000)   # milliseconds
    except Exception:  # noqa: BLE001 — older ee without setDeadline: skip, endpoint fix still applies
        pass
    _gee_initialised = True


# ---------------------------------------------------------------------------
# GEE dynamic helpers
# ---------------------------------------------------------------------------


def _fetch_gee_bands(
    collection_id: str,
    bands: list[str],
    scale: float,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    chunk_months: int = 60,
    collection_start: str | None = None,
) -> pd.DataFrame:
    """
    Fetch one or more bands from a single ImageCollection at a point.

    Splits the date range into chunks of `chunk_months` to avoid GEE memory
    limits. If `collection_start` is provided, the effective start date is
    clamped to that date so chunks before satellite/collection availability
    are skipped silently instead of emitting noisy warnings.
    """
    point = ee.Geometry.Point([lon, lat])

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Clamp start to the collection's actual availability window
    if collection_start:
        col_start_dt = datetime.strptime(collection_start, "%Y-%m-%d")
        if start < col_start_dt:
            logger.debug(
                "Clamping start from %s to collection_start %s for %s",
                start_date,
                collection_start,
                collection_id,
            )
            start = col_start_dt

    if start >= end:
        logger.info(
            "Date range entirely before collection_start (%s) for %s — skipping",
            collection_start,
            collection_id,
        )
        return pd.DataFrame(columns=["date"] + bands)

    all_dfs = []
    current = start

    while current < end:  # this loop runs only once since chunk_months = end- start
        chunk_end = min(current + relativedelta(months=chunk_months), end)
        s_str = current.strftime("%Y-%m-%d")
        e_str = chunk_end.strftime("%Y-%m-%d")

        try:
            collection = (
                ee.ImageCollection(collection_id).filterDate(s_str, e_str).select(bands)
            )
            region_data = _gee_call_with_retry(
                lambda: collection.getRegion(point, scale).getInfo()
            )

            if region_data and len(region_data) >= 2:
                df = pd.DataFrame(region_data[1:], columns=region_data[0])
                df["date"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
                df = df.set_index("date")[bands]
                df = df[~df.index.duplicated(keep="first")]
                all_dfs.append(df)
            else:
                logger.debug(
                    "No GEE data (empty result): %s %s (%s → %s)",
                    collection_id,
                    bands,
                    s_str,
                    e_str,
                )

        except Exception as e:
            logger.warning(
                "GEE chunk failed — %s %s (%s → %s): %s",
                collection_id,
                bands,
                s_str,
                e_str,
                e,
            )

        current = chunk_end

    if not all_dfs:
        logger.warning(
            "No GEE data at all: %s %s (%s → %s)",
            collection_id,
            bands,
            start_date,
            end_date,
        )
        return pd.DataFrame(columns=["date"] + bands)

    merged = pd.concat(all_dfs)
    merged = merged[~merged.index.duplicated(keep="first")].sort_index()
    return merged.reset_index()


def get_historical_features(
    lat: float,
    lon: float,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> pd.DataFrame:
    """
    Fetch all dynamic features for a point from GEE.

    Returns DataFrame: [date, rainfall, temp, et, runoff, ndvi, sm]
    """
    _init_gee()
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)

    def _fetch_one(args):
        i, cfg = args
        _time.sleep(
            1
        )  # stagger: collection 0 fires immediately, 1 after 1s, 2 after 2s, …
        logger.info(
            "Fetching GEE collection '%s' bands %s for (%.4f, %.4f) from %s to %s",
            cfg["collection"],
            cfg["bands"],
            lat,
            lon,
            start,
            end,
        )
        timer_start = _time.time()

        # fetch historical features from bands
        df = _fetch_gee_bands(
            cfg["collection"],
            cfg["bands"],
            cfg["scale"],
            lat,
            lon,
            start,
            end,
            chunk_months=cfg.get("chunk_months", 60),
            collection_start=cfg.get("collection_start"),
        )
        if not df.empty and cfg["rename"]:
            df = df.rename(columns=cfg["rename"])
        logger.info(
            "Fetched %d rows from GEE collection '%s' for (%.4f, %.4f)",
            len(df),
            cfg["collection"],
            lat,
            lon,
        )
        timer_end = _time.time()
        logger.info(
            "Time taken for collection '%s': %s",
            cfg["collection"],
            timer_end - timer_start,
        )
        return df.set_index("date")

    cfgs = list(GEE_DYNAMIC.values())

    # fetch dynamic historical
    with ThreadPoolExecutor(max_workers=len(cfgs) if not IS_EVAL else WORKERS) as pool:
        frames = list(pool.map(_fetch_one, enumerate(cfgs)))

    if not frames:
        return pd.DataFrame(columns=_DYNAMIC_COLS)

    merged = frames[0].join(frames[1:], how="outer")

    # sm = mean(soil_moisture_am, soil_moisture_pm)
    if "soil_moisture_am" in merged.columns and "soil_moisture_pm" in merged.columns:
        merged["sm"] = (merged["soil_moisture_am"] + merged["soil_moisture_pm"]) / 2.0
        merged = merged.drop(columns=["soil_moisture_am", "soil_moisture_pm"])

    # ndvi = (SR_B5 - SR_B4) / (SR_B5 + SR_B4) — matches data_preparation.py
    if "sr_b4" in merged.columns and "sr_b5" in merged.columns:
        denom = merged["sr_b5"] + merged["sr_b4"]
        merged["ndvi"] = np.where(
            denom == 0, 0, (merged["sr_b5"] - merged["sr_b4"]) / denom
        )
        merged = merged.drop(columns=["sr_b4", "sr_b5"])

    merged["lulc"] = "no_data"

    merged = merged.reset_index()
    existing = [c for c in _DYNAMIC_COLS if c in merged.columns]
    return merged[existing]


# ---------------------------------------------------------------------------
# GEE static helper
# ---------------------------------------------------------------------------


def get_static_features(lat: float, lon: float) -> dict:
    """
    Sample static GEE assets at a point (terrain, stream order, lithology, aquifer).
    Mirrors extract_static_data() from gee_data_extraction.py.
    """
    _init_gee()
    point = ee.Geometry.Point([lon, lat])

    def _fetch_elevation():
        terrain = ee.Image(GEE_ASSETS["terrain"]).select("constant")
        elev = _gee_call_with_retry(
            lambda: terrain.reduceRegion(ee.Reducer.first(), point, 30).getInfo()
        )
        return elev.get("constant") or 0

    def _fetch_stream_order():
        stream = ee.Image(GEE_ASSETS["stream_order"])
        so = _gee_call_with_retry(
            lambda: stream.reduceRegion(ee.Reducer.first(), point, 90).getInfo()
        )
        return so.get("b1") or 0

    def _fetch_lithology():
        litho_fc = (
            ee.FeatureCollection(GEE_ASSETS["lithology"]).filterBounds(point).first()
        )
        litho_info = _gee_call_with_retry(
            lambda: ee.Algorithms.If(
                litho_fc, litho_fc.toDictionary(), ee.Dictionary({})
            ).getInfo()
        )
        lithology = litho_info.get("LITHOLOGIC", "no_data") if litho_info else "no_data"
        litho_supergroup = (
            litho_info.get("SUPERGROUP", "no_data") if litho_info else "no_data"
        )
        return lithology, litho_supergroup

    def _fetch_aquifer():
        aquifer_fc = (
            ee.FeatureCollection(GEE_ASSETS["aquifer"]).filterBounds(point).first()
        )
        aquifer_info = _gee_call_with_retry(
            lambda: ee.Algorithms.If(
                aquifer_fc, aquifer_fc.toDictionary(), ee.Dictionary({})
            ).getInfo()
        )
        aquifer_type = (
            aquifer_info.get("Major_Aqui", "no_data") if aquifer_info else "no_data"
        )
        aquifer_0_aquifer = (
            aquifer_info.get("aquifer", "no_data") if aquifer_info else "no_data"
        )
        return aquifer_type, aquifer_0_aquifer

    with ThreadPoolExecutor(max_workers=4 if not IS_EVAL else WORKERS) as pool:
        f_elev = pool.submit(_fetch_elevation)
        f_stream = pool.submit(_fetch_stream_order)
        f_litho = pool.submit(_fetch_lithology)
        f_aquifer = pool.submit(_fetch_aquifer)
        elevation = f_elev.result()
        stream_order = f_stream.result()
        lithology, litho_supergroup = f_litho.result()
        aquifer_type, aquifer_0_aquifer = f_aquifer.result()

    return {
        "elevation": elevation,
        "stream_order": stream_order,
        "lithology": lithology,
        "litho_supergroup": litho_supergroup,
        "aquifer_type": aquifer_type,
        "aquifer_0_aquifer": aquifer_0_aquifer,
    }


# ---------------------------------------------------------------------------
# Fallback forecast: lightweight rainfall + temp only
# ---------------------------------------------------------------------------


def get_rainfall_temp_history(
    lat: float,
    lon: float,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> pd.DataFrame:
    """
    Fetch only CHIRPS rainfall and ERA5 temperature for the fallback forecast.
    Much faster than get_historical_features — skips SMAP, Landsat, and extra ERA5 bands.
    Returns DataFrame: [date, rainfall, temp]
    """
    _init_gee()
    start = _to_date_str(start_date)
    end = _to_date_str(end_date)

    chirps_cfg = GEE_DYNAMIC["chirps"]
    era5_cfg = GEE_DYNAMIC["era5"]

    def _fetch_chirps():
        return _fetch_gee_bands(
            chirps_cfg["collection"],
            chirps_cfg["bands"],
            chirps_cfg["scale"],
            lat,
            lon,
            start,
            end,
            chunk_months=chirps_cfg["chunk_months"],
            collection_start=chirps_cfg.get("collection_start"),
        ).rename(columns=chirps_cfg["rename"])

    def _fetch_era5():
        return _fetch_gee_bands(
            era5_cfg["collection"],
            ["temperature_2m"],
            era5_cfg["scale"],
            lat,
            lon,
            start,
            end,
            chunk_months=era5_cfg["chunk_months"],
            collection_start=era5_cfg.get("collection_start"),
        ).rename(columns={"temperature_2m": "temp"})

    with ThreadPoolExecutor(max_workers=2 if not IS_EVAL else WORKERS) as pool:
        f_chirps = pool.submit(_fetch_chirps)
        f_era5 = pool.submit(_fetch_era5)
        chirps = f_chirps.result()
        era5 = f_era5.result()

    if chirps.empty and era5.empty:
        return pd.DataFrame(columns=["date", "rainfall", "temp"])

    df = (
        chirps.set_index("date")
        .join(era5.set_index("date"), how="outer")
        .reset_index()
        .rename(columns={"index": "date"})
    )
    return df


# ---------------------------------------------------------------------------
# Forecast features — Open-Meteo seasonal API + archive baseline
# ---------------------------------------------------------------------------


def _parse_daily_response(resp) -> tuple[pd.DatetimeIndex, object]:
    """Parse Daily() block from an openmeteo response into (dates, daily)."""
    daily = resp.Daily()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).tz_localize(None)
    return dates, daily


def get_forecast_features_with_api(
    lat: float,
    lon: float,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> pd.DataFrame:
    """
    Fetch 90-day forecast using Open-Meteo seasonal API.

    Seasonal API returns rainfall (mm/day) and temperature as an anomaly (°C
    departure from 1991-2020 climatology).  The anomaly is converted to absolute
    Kelvin by adding per-month ERA5 archive baselines fetched in parallel:

        temp_K = anomaly_C + archive_monthly_mean_C + 273.15

    Returns DataFrame [date, rainfall, temp(K)], or empty DataFrame on failure
    (preprocess climatology fallback will kick in automatically).
    """
    try:
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry

        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        cache_session.request = lambda method, url, **kwargs: (
            type(cache_session).request(
                cache_session, method, url, timeout=kwargs.pop("timeout", 8), **kwargs
            )
        )
        retry_session = retry(cache_session, retries=1, backoff_factor=0.1)
        client = openmeteo_requests.Client(session=retry_session)

        start_dt = _to_datetime(start_date)
        end_dt = _to_datetime(end_date)

        # ── parallel: seasonal forecast + ERA5 archive baseline ───────────
        def _fetch_seasonal():
            return client.weather_api(
                "https://seasonal-api.open-meteo.com/v1/seasonal",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ["rain_sum", "temperature_2m_mean"],
                    "start_date": _to_date_str(start_dt),
                    "end_date": _to_date_str(end_dt),
                },
            )[0]

        def _fetch_archive():
            return client.weather_api(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": ["temperature_2m_mean"],
                    "start_date": (start_dt - relativedelta(years=5)).strftime(
                        "%Y-%m-%d"
                    ),
                    "end_date": (end_dt - relativedelta(years=1)).strftime("%Y-%m-%d"),
                },
            )[0]

        with ThreadPoolExecutor(max_workers=2 if not IS_EVAL else WORKERS) as pool:
            f_s = pool.submit(_fetch_seasonal)
            f_a = pool.submit(_fetch_archive)
            s_resp = f_s.result()
            a_resp = f_a.result()

        # ── parse seasonal ────────────────────────────────────────────────
        s_dates, s_daily = _parse_daily_response(s_resp)
        df = pd.DataFrame(
            {
                "date": s_dates,
                "rainfall": s_daily.Variables(0).ValuesAsNumpy(),
                "temp_anomaly": s_daily.Variables(1).ValuesAsNumpy(),
            }
        )

        # ── parse archive → per-month mean temperature (°C) ──────────────
        a_dates, a_daily = _parse_daily_response(a_resp)
        arch_df = pd.DataFrame(
            {"date": a_dates, "temp": a_daily.Variables(0).ValuesAsNumpy()}
        )

        forecast_months = set(df["date"].dt.month.tolist())
        monthly_normals = (
            arch_df[arch_df["date"].dt.month.isin(forecast_months)]
            .groupby(
                arch_df.loc[
                    arch_df["date"].dt.month.isin(forecast_months), "date"
                ].dt.month
            )["temp"]
            .mean()
            .to_dict()
        )

        # anomaly(°C) + historical_mean(°C) + 273.15 → Kelvin (matches ERA5-Land training scale)
        df["temp"] = (
            df["temp_anomaly"]
            + df["date"].dt.month.map(monthly_normals).fillna(0.0)
            + 273.15
        )
        df = df.drop(columns=["temp_anomaly"])

        if df[["rainfall", "temp"]].isna().any().any():
            logger.warning("Seasonal forecast has NaN values — fallback will be used")
            return pd.DataFrame(columns=["date", "rainfall", "temp"])

        logger.info(
            "Seasonal forecast: %d rows (rain_sum=%.1f mm, temp_mean=%.1f K)",
            len(df),
            df["rainfall"].sum(),
            df["temp"].mean(),
        )
        return df[["date", "rainfall", "temp"]]

    except Exception as e:
        logger.warning("Seasonal forecast fetch failed (%s) — fallback will be used", e)
        return pd.DataFrame(columns=["date", "rainfall", "temp"])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class DataFetcher:

    def __init__(self):
        self.flat = None
        self.flon = None
        self.station_info = []

    def _get_admin_details(self, lat: float, lon: float) -> tuple[str, str]:
        """Call CoreStack geoserver to get state and district for a lat/lon.

        Retried with exponential backoff (transient timeouts/5xx) like the other
        live APIs; returns ('', '') only after exhausting attempts.
        """
        last = None
        for attempt in range(_WRIS_MAX_RETRIES):
            try:
                resp = requests.get(
                    f"{CORESTACK_BASE_URL}/get_admin_details_by_latlon/",
                    params={"latitude": lat, "longitude": lon},
                    headers={"X-API-Key": CORESTACK_API_KEY},
                    timeout=_WRIS_TIMEOUT,
                )
                d = resp.json()
                state = d.get("State") or d.get("state") or d.get("state_name", "")
                district = (
                    d.get("District") or d.get("district") or d.get("district_name", "")
                )
                logger.info(
                    "Admin details for (%.4f, %.4f): state=%s, district=%s",
                    lat, lon, state, district,
                )
                return str(state), str(district)
            except Exception as e:
                last = f"{type(e).__name__}: {str(e)[:80]}"
                if attempt < _WRIS_MAX_RETRIES - 1:
                    wait = _WRIS_BACKOFF * (2 ** attempt)
                    logger.warning(
                        "CoreStack admin attempt %d/%d failed (%s) — retrying in %ds",
                        attempt + 1, _WRIS_MAX_RETRIES, last, wait,
                    )
                    _time.sleep(wait)
        logger.warning("Failed to get admin details after %d attempts: %s",
                       _WRIS_MAX_RETRIES, last)
        return "", ""

    def _list_wris_stations(self, state: str = "", district: str = "") -> list[dict]:
        """Return stations from the local CSV index, filtered by district or state."""
        global _STATION_INDEX
        if _STATION_INDEX is None:
            _STATION_INDEX = _load_station_index(STATION_CSV_PATH)

        if district:
            mask = _STATION_INDEX["district_lower"] == district.lower().strip()
            label = f"district='{district}'"
        elif state:
            mask = _STATION_INDEX["state_lower"] == state.lower().strip()
            label = f"state='{state}'"
        else:
            logger.warning("No state or district provided — cannot list stations")
            return []

        subset = _STATION_INDEX[mask]
        logger.info("Station index filter %s → %d stations", label, len(subset))

        stations = []
        for _, row in subset.iterrows():
            try:
                stations.append(
                    {
                        "station_code": str(row["station_code"]),
                        "lat": float(row["latitude"]),
                        "lon": float(row["longitude"]),
                    }
                )
            except (TypeError, ValueError):
                continue
        return stations

    def get_nearest_stations(
        self,
        lat: float,
        lon: float,
        n: int = 10,
        state: str = "",
        district: str = "",
        min_distance: float = None,
    ) -> list[dict]:
        """
        Find n nearest WRIS stations to (lat, lon).
        1. Get state + district from CoreStack geoserver (skipped if already provided).
        2. List WRIS stations in that district; fall back to state if < n found.
        3. Return the n closest by haversine distance.
        """
        if not state and not district:
            state, district = self._get_admin_details(lat, lon)

        stations = self._list_wris_stations(district=district, state=state)
        if len(stations) < n and state:
            logger.info(
                "Only %d stations in district '%s', falling back to state '%s'",
                len(stations),
                district,
                state,
            )
            stations = self._list_wris_stations(state=state)

        if not stations:
            logger.warning(
                "No WRIS stations found for state='%s' district='%s' — "
                "falling back to global nearest-station search",
                state,
                district,
            )
            global _STATION_INDEX
            if _STATION_INDEX is None:
                _STATION_INDEX = _load_station_index(STATION_CSV_PATH)
            idx = _STATION_INDEX.dropna(subset=["latitude", "longitude"])
            stations = [
                {
                    "station_code": str(r["station_code"]),
                    "lat": float(r["latitude"]),
                    "lon": float(r["longitude"]),
                }
                for r in idx[["station_code", "latitude", "longitude"]].to_dict(
                    "records"
                )
            ]
            if not stations:
                return []

        distances = compute_distances(lat, lon, stations)

        if min_distance is not None:
            filtered = [(s, d) for s, d in zip(stations, distances) if d > min_distance]
            if not filtered:
                logger.warning(
                    "No stations found beyond min_distance=%.1f km for (%.4f, %.4f)",
                    min_distance,
                    lat,
                    lon,
                )
                return []
            stations, distances = zip(*filtered)
            stations = list(stations)
            distances = np.array(
                distances
            )  # find_k_nearest needs np.ndarray for fancy indexing

        indices, _ = find_k_nearest(distances, n)
        print(stations[i] for i in indices)
        return [stations[i] for i in indices]

    def api_call(self, endpoint: str, payload: dict) -> list:
        """POST to a WRIS endpoint with exponential-backoff retries.

        Retries on transient failures (timeouts, connection errors, non-200
        statusCode — WRIS returns transient 5xx under load). A clean 200 with
        empty data returns [] immediately (legitimately no data, not an error).
        Returns [] only after exhausting retries.
        """
        last = None
        for attempt in range(_WRIS_MAX_RETRIES):
            try:
                resp = requests.post(
                    f"{WRIS_BASE_URL}{endpoint}",
                    headers=WRIS_HEADERS,
                    json=payload,
                    timeout=_WRIS_TIMEOUT,
                )
                data = resp.json()
                if data.get("statusCode") == 200:
                    return data.get("data", [])
                last = f"statusCode={data.get('statusCode')} http={resp.status_code}"
            except Exception as e:  # timeout / connection / JSON decode
                last = f"{type(e).__name__}: {str(e)[:80]}"
            if attempt < _WRIS_MAX_RETRIES - 1:
                wait = _WRIS_BACKOFF * (2 ** attempt)
                logger.warning(
                    "WRIS %s attempt %d/%d failed (%s) — retrying in %ds",
                    endpoint, attempt + 1, _WRIS_MAX_RETRIES, last, wait,
                )
                _time.sleep(wait)
        logger.error(
            "WRIS %s failed after %d attempts (%s)", endpoint, _WRIS_MAX_RETRIES, last
        )
        return []

    def download_from_station(
        self, station_code: str, start_date: str, end_date: str
    ) -> dict:
        start = _time.time()
        with ThreadPoolExecutor(max_workers=2 if not IS_EVAL else WORKERS) as pool:
            f_meta = pool.submit(
                self.api_call,
                "/stationMaster/getMasterStationsList",
                {"stationcode": station_code, "datasetcode": WRIS_DATASET_CODE},
            )
            f_ts = pool.submit(
                self.api_call,
                "/CommonDataSetMasterAPI/getCommonDataSetByStationCode",
                {
                    "station_code": station_code,
                    "starttime": start_date,
                    "endtime": end_date,
                    "dataset": WRIS_DATASET_CODE,
                },
            )
            metadata = f_meta.result()
            timeseries = f_ts.result()
        end = _time.time()
        logger.info(
            "WRIS API call for station %s took %.1f seconds", station_code, end - start
        )
        return {
            "station_code": station_code,
            "metadata": metadata,
            "timeseries": timeseries,
            "query_params": {"start_date": start_date, "end_date": end_date},
        }

    def _fetch_station_data(
        self,
        station: dict,
        start_date: str,
        end_date: str,
        fallback_start: str,
    ) -> dict:

        slat = station["lat"]
        slon = station["lon"]
        scode = station["station_code"]
        logger.info("Processing station %s (%.4f, %.4f)", scode, slat, slon)
        start = _time.time()
        # NOTE: the live engine path uses new_data_fetcher.DataFetcher.get_Data,
        # which overrides this whole flow. Keeping the same WRIS widening here
        # for any caller that does hit _fetch_station_data directly.
        _WRIS_END_BUFFER_DAYS = 35
        try:
            wris_end_date = (
                datetime.strptime(end_date, "%Y-%m-%d")
                + timedelta(days=_WRIS_END_BUFFER_DAYS)
            ).strftime("%Y-%m-%d")
        except Exception:
            wris_end_date = end_date
        with ThreadPoolExecutor(max_workers=4 if not IS_EVAL else WORKERS) as pool:
            f_wris = pool.submit(
                self.download_from_station, scode, start_date, wris_end_date
            )
            f_dynamic = pool.submit(
                get_historical_features, slat, slon, start_date, end_date
            )
            f_fallback = pool.submit(
                get_rainfall_temp_history, slat, slon, fallback_start, end_date
            )
            f_gee_static = pool.submit(get_static_features, slat, slon)

            raw = f_wris.result()
            dynamic_df = f_dynamic.result()
            rainfall_temp_history = f_fallback.result()
            gee_static = f_gee_static.result()

        gwl_df = self._parse_gwl_timeseries(raw["timeseries"], raw["metadata"], scode)
        if not dynamic_df.empty:
            dynamic_df.insert(0, "station_code", scode)

        m = raw["metadata"][0] if raw["metadata"] else {}
        well_type = m.get("well_type", m.get("welltype", "no_data")) or "no_data"
        try:
            well_depth = float(
                m.get("well_depth", m.get("welldepth", m.get("depth", 0))) or 0
            )
        except (TypeError, ValueError):
            well_depth = 0.0

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
            scode,
            len(gwl_df),
            len(dynamic_df),
            len(rainfall_temp_history),
        )
        end = _time.time()
        logger.info(
            "Total time for GEE fetch station %s: %.1f seconds", scode, end - start
        )
        return {
            "gwl": gwl_df,
            "dynamic": dynamic_df,
            "static": static,
            "rainfall_temp_history": rainfall_temp_history,
        }

    @staticmethod
    def _parse_gwl_timeseries(
        timeseries: list, metadata: list, station_code: str
    ) -> pd.DataFrame:
        """Convert raw WRIS timeseries → DataFrame [station_code, date, gwl_value, state, district]."""
        if not timeseries:
            return pd.DataFrame(
                columns=["station_code", "date", "gwl_value", "state", "district"]
            )

        df = pd.DataFrame(timeseries)

        # WRIS returns 'dataTime' (sub-daily) — match on 'date' OR 'time'
        date_col = next(
            (c for c in df.columns if "date" in c.lower() or "time" in c.lower()), None
        )
        if date_col:
            df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
            if date_col != "date":
                df = df.drop(columns=[date_col])

        # WRIS value column is 'dataValue'
        val_col = next(
            (c for c in df.columns if c != "date" and "value" in c.lower()), None
        )
        if val_col and val_col != "gwl_value":
            df = df.rename(columns={val_col: "gwl_value"})

        # Aggregate sub-daily readings to one row per day.
        # WRIS sometimes returns dataValue as a string (e.g. "3.2") or with
        # sentinel values like "" / "NA", which makes the column object-dtype
        # and breaks .mean(). Coerce + drop NaNs before the groupby.
        df["gwl_value"] = pd.to_numeric(df["gwl_value"], errors="coerce")
        df = df.dropna(subset=["gwl_value"])
        df = df.groupby("date", as_index=False)["gwl_value"].mean()

        # ── DEBUG: what does the engine actually see from WRIS live? ─────
        print(f"\n[DEBUG] _parse_gwl_timeseries  station={station_code}  daily_rows={len(df)}")
        if len(df):
            print(f"[DEBUG]    date range: {df['date'].min().date()} → {df['date'].max().date()}")
            try:
                _anchor = pd.Timestamp("2024-10-30")
                _near = df[(df['date'] >= _anchor - pd.Timedelta(days=35))
                         & (df['date'] <= _anchor + pd.Timedelta(days=35))].sort_values("date")
                print(f"[DEBUG]    rows within ±35d of {_anchor.date()}: {len(_near)}")
                for _, r in _near.iterrows():
                    print(f"[DEBUG]      {r['date'].date()}  gwl={r['gwl_value']:.4f}  "
                          f"({(r['date'] - _anchor).days:+d}d)")
            except Exception as _e:
                print(f"[DEBUG]    (anchor-window inspection failed: {_e})")

        m = metadata[0] if metadata else {}
        df["station_code"] = station_code
        df["state"] = m.get("state", m.get("statename", "no_data"))
        df["district"] = m.get("district", m.get("districtname", "no_data"))

        return df[["station_code", "date", "gwl_value", "state", "district"]]

    @staticmethod
    def _parse_static(
        metadata: list, station_code: str, lat: float, lon: float
    ) -> dict:
        """Combine WRIS metadata + GEE static assets → static feature dict."""
        m = metadata[0] if metadata else {}

        well_type = m.get("well_type", m.get("welltype", "no_data")) or "no_data"
        try:
            well_depth = float(
                m.get("well_depth", m.get("welldepth", m.get("depth", 0))) or 0
            )
        except (TypeError, ValueError):
            well_depth = 0.0

        gee = get_static_features(lat, lon)

        return {
            "station_code": station_code,
            "latitude": lat,
            "longitude": lon,
            "well_type": well_type,
            "well_depth": well_depth,
            **gee,
        }

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
        """
        Returns
        -------
        inputs : list of 10 dicts, each:
            {
              "gwl":     DataFrame [station_code, date, gwl_value, state, district],
              "dynamic": DataFrame [station_code, date, rainfall, temp, et, runoff, ndvi, sm],
              "static":  dict      {station_code, latitude, longitude, elevation,
                                   stream_order, well_type, well_depth, lithology,
                                   litho_supergroup, aquifer_type, aquifer_0_aquifer},
            }

        forecasts : list of 10 DataFrames [date, rainfall, temp, et, runoff, ndvi, sm] #TODO: DICT OF DICTS AND KEY IS STATION CODE
        """
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

        state, district = self._get_admin_details(lat, lon)

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
                lat,
                lon,
                n=10,
                state=state,
                district=district,
                min_distance=min_distance,
            )
        self.station_info = stations

        if not stations:
            logger.warning("No stations found — returning empty results")
            return [], []

        _init_gee()

        logger.info("Fetching data for %d stations in parallel", len(stations))
        with ThreadPoolExecutor(
            max_workers=len(stations) if not is_eval else WORKERS
        ) as pool:
            inputs = list(
                pool.map(
                    lambda s: self._fetch_station_data(
                        s, start_date, end_date, fallback_start
                    ),
                    stations,
                )
            )

        start = _time.time()
        with ThreadPoolExecutor(
            max_workers=len(stations) if not is_eval else WORKERS
        ) as pool:
            future_to_idx = {
                pool.submit(
                    get_forecast_features_with_api,
                    s["lat"],
                    s["lon"],
                    current_date,
                    forecast_end,
                ): i
                for i, s in enumerate(stations)
            }
            forecasts = [None] * len(stations)
            for future in future_to_idx:
                forecasts[future_to_idx[future]] = future.result()

        end = _time.time()
        logger.info(
            "Fetched forecasts for %d stations in %.1f seconds (parallel)",
            len(stations),
            end - start,
        )
        return inputs, forecasts, exact_match


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------


def _to_date_str(d: Union[str, datetime]) -> str:
    if isinstance(d, str):
        return d
    return d.strftime("%Y-%m-%d")


def _to_datetime(d: Union[str, datetime]) -> datetime:
    if isinstance(d, datetime):
        return d
    return datetime.strptime(d, "%Y-%m-%d")


if __name__ == "__main__":
    current_date = "2025-01-01"

    STATIONS = [
        {"station_code": "PBGW-8", "lat": 32.3686, "lon": 75.6133},
        {"station_code": "PBGW-5", "lat": 32.2883, "lon": 75.7433},
    ]

    _init_gee()

    target_dt = _to_datetime(current_date)
    start_date = (target_dt - timedelta(days=365 * LOOKBACK_YEARS)).strftime("%Y-%m-%d")
    forecast_end = (target_dt + timedelta(days=FORECAST_DAYS)).strftime("%Y-%m-%d")

    inf = DataFetcher()
    inputs = []
    forecasts = []

    for station in STATIONS:
        scode = station["station_code"]
        slat = station["lat"]
        slon = station["lon"]

        print(f"\n{'='*60}")
        print(f"Station: {scode}  ({slat}, {slon})")
        print(f"{'='*60}")

        raw = inf.download_from_station(scode, start_date, current_date)
        gwl_df = inf._parse_gwl_timeseries(raw["timeseries"], raw["metadata"], scode)
        print(f"GWL rows      : {len(gwl_df)}")
        print(gwl_df.head(3))

        dynamic_df = get_historical_features(slat, slon, start_date, current_date)
        if not dynamic_df.empty:
            dynamic_df.insert(0, "station_code", scode)
        print(f"\nDynamic rows  : {len(dynamic_df)}")
        print(dynamic_df.head(3))

        static = inf._parse_static(raw["metadata"], scode, slat, slon)
        print(f"\nStatic features:")
        for k, v in static.items():
            print(f"  {k}: {v}")

        forecast_df = get_forecast_features_with_api(
            slat, slon, current_date, forecast_end
        )
        print(f"\nForecast rows : {len(forecast_df)}")
        print(forecast_df.head(3))

        inputs.append({"gwl": gwl_df, "dynamic": dynamic_df, "static": static})
        forecasts.append(forecast_df)

    print(f"\n{'='*60}")
    print(f"DONE — {len(inputs)} station dicts, {len(forecasts)} forecast DataFrames")
    print(f"{'='*60}")
