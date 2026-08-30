"""Open-Meteo seasonal forecast for the inference forecast window.

Ports `get_forecast_features_with_api` from the inference-eval branch: fetches a
rain + temperature forecast over the horizon window from Open-Meteo's SEASONAL
API, and de-anomalizes the temperature (which the seasonal API returns as a °C
departure from 1991-2020 climatology) to absolute KELVIN using a per-month ERA5
archive baseline — matching the ERA5-Land temperature scale the model trained on:

    temp_K = anomaly_C + archive_monthly_mean_C + 273.15

This is the PRIMARY forecast source. On any failure/empty (API down, deps absent,
or a clearly-historical window the seasonal API won't forecast) it returns {},
and the caller (SampleBuilder) falls back to climatology.
"""
from __future__ import annotations

from datetime import datetime
from typing import Union


def _to_date_str(d: "Union[str, datetime]") -> str:
    return d if isinstance(d, str) else d.strftime("%Y-%m-%d")


def _to_datetime(d: "Union[str, datetime]") -> datetime:
    return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")


def _parse_daily_response(resp):
    import pandas as pd

    daily = resp.Daily()
    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).tz_localize(None)
    return dates, daily


def _forecast_df(lat, lon, start_date, end_date):
    """Open-Meteo seasonal rain + temp(K) over [start, end] -> DataFrame
    [date, rainfall, temp], or None on any failure/empty."""
    try:
        from concurrent.futures import ThreadPoolExecutor

        import pandas as pd
        import openmeteo_requests
        import requests_cache
        from retry_requests import retry
        from dateutil.relativedelta import relativedelta

        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        cache_session.request = lambda method, url, **kwargs: (
            type(cache_session).request(
                cache_session, method, url, timeout=kwargs.pop("timeout", 8), **kwargs
            )
        )
        client = openmeteo_requests.Client(
            session=retry(cache_session, retries=1, backoff_factor=0.1)
        )
        start_dt = _to_datetime(start_date)
        end_dt = _to_datetime(end_date)

        def _seasonal():
            return client.weather_api(
                "https://seasonal-api.open-meteo.com/v1/seasonal",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": ["rain_sum", "temperature_2m_mean"],
                    "start_date": _to_date_str(start_dt), "end_date": _to_date_str(end_dt),
                },
            )[0]

        def _archive():
            return client.weather_api(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat, "longitude": lon,
                    "daily": ["temperature_2m_mean"],
                    "start_date": (start_dt - relativedelta(years=5)).strftime("%Y-%m-%d"),
                    "end_date": (end_dt - relativedelta(years=1)).strftime("%Y-%m-%d"),
                },
            )[0]

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_s = pool.submit(_seasonal)
            f_a = pool.submit(_archive)
            s_resp = f_s.result()
            a_resp = f_a.result()

        s_dates, s_daily = _parse_daily_response(s_resp)
        df = pd.DataFrame({
            "date": s_dates,
            "rainfall": s_daily.Variables(0).ValuesAsNumpy(),
            "temp_anomaly": s_daily.Variables(1).ValuesAsNumpy(),
        })
        if df.empty:
            return None

        a_dates, a_daily = _parse_daily_response(a_resp)
        arch = pd.DataFrame({"date": a_dates, "temp": a_daily.Variables(0).ValuesAsNumpy()})
        fmonths = set(df["date"].dt.month.tolist())
        m = arch[arch["date"].dt.month.isin(fmonths)]
        normals = m.groupby(m["date"].dt.month)["temp"].mean().to_dict()

        # anomaly(°C) + historical_mean(°C) + 273.15 -> Kelvin (ERA5-Land training scale)
        df["temp"] = df["temp_anomaly"] + df["date"].dt.month.map(normals).fillna(0.0) + 273.15
        return df.drop(columns=["temp_anomaly"])
    except Exception:
        return None


def fetch_openmeteo_rain_temp(lat, lon, start_date, end_date) -> dict:
    """Aggregate the Open-Meteo forecast over the window ->
    {'rainfall': sum, 'temp': mean}. Empty dict on any failure/empty response."""
    df = _forecast_df(lat, lon, start_date, end_date)
    out: dict = {}
    if df is not None and len(df):
        if "rainfall" in df.columns:
            out["rainfall"] = float(df["rainfall"].sum())
        if "temp" in df.columns:
            out["temp"] = float(df["temp"].mean())
    return out
