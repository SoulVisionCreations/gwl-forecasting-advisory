"""wris_gwl_provider.py — LIVE GWL lookback from India-WRIS BY station_code (robust: -k + depth filter).

The mirror of NwdpGwlProvider for the `gwl_provider` seam, but it queries WRIS's
getCommonDataSetByStationCode by the well's OWN `station_code` (NO nearest-neighbour search — the
advisory already does its own IDW over neighbours), with verify=false (WRIS's broken TLS chain), the
depth-only datatype filter (drop AMSL elevation rows), and the canonical clean_gwl_values (abs + caps).
This is the SAME robust WRIS path as the WRIS-primary normals (reuses parse_wris_records). It returns
the WRIS-compatible (gwl_df, meta) contract with `data_age_days`, via the SHARED resolve_anchor_gwl
policy (fresh -> unchanged; stale -> forward-fill to the anchor; anchor never moved).

Why it exists: the vendored `GWL_SOURCE=wris` path predates the stopgap and lacks verify=false + the
depth filter (which is why it broke and NWDP was adopted). This is the robust replacement.
"""
from __future__ import annotations

import threading
import time as _time
from datetime import timedelta

from inference.acquire.csv_wris_fallback_source import (
    _ENDPOINT,
    _WRIS_HEADERS,
    WRIS_BASE_URL,
    WRIS_DATASET_CODE,
    parse_wris_records,
)


class WrisGwlProvider:
    """GWL-only seam (like NwdpGwlProvider). `.gwl_and_meta(station_code, lat, lon, current_date)`
    returns (gwl_df, meta) — gwl_df = DataFrame('date','gwl_value'); meta has data_age_days."""

    def __init__(self, data_config=None, verify=False, lookback_days=400,
                 retries=2, timeout=30):
        self.data_config = data_config
        self.verify = verify
        self.lookback_days = int(lookback_days)
        self.retries = retries
        self.timeout = timeout
        self._notes: "list[str]" = []
        self._notes_lock = threading.Lock()   # gwl_and_meta may run on many threads (concurrent fetch)
        self._down = False                     # per-request WRIS circuit breaker: a transport failure on
        #                                        ANY neighbour skips WRIS for the rest; reset_breaker() clears.

    def reset_breaker(self):
        """Clear the transport-failure breaker at the START of a fetch (called by NumericFetcher).
        Mirrors WrisWithCsvFallbackSource's per-request `_down_this_fetch`."""
        self._down = False

    def _forward_buffer_days(self) -> int:
        hm = getattr(self.data_config, "forecast_horizon_months", 3) or 3
        gap = getattr(self.data_config, "gap_days", 30) or 30
        return int(hm) * 31 + int(gap) + 5

    def gwl_and_meta(self, station_code, lat, lon, current_date):
        import pandas as pd
        from inference.acquire.anchor import resolve_anchor_gwl

        meta = {"well_type": "no_data", "well_depth": 0.0, "data_age_days": None}
        if self._down:                          # a prior neighbour hit a WRIS transport failure this request
            return None, meta                   # -> skip the (dead) endpoint; caller uses the fallback
        start = current_date - timedelta(days=self.lookback_days)
        end = current_date + timedelta(days=self._forward_buffer_days())
        recs = self._post(station_code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        if recs is None:
            return None, meta
        try:
            s = parse_wris_records(recs, station_code, self.data_config)   # _is_depth + per-day mean + clean
        except Exception:  # noqa: BLE001 — a bad well never sinks the request
            s = None
        if s is None or not len(s):
            return None, meta
        df = s.reset_index()                                              # DatetimeIndex 'date' + 'gwl_value'
        df.columns = ["date", "gwl_value"][: len(df.columns)] if len(df.columns) == 2 else df.columns
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp(end.date())]
        if df.empty:
            return None, meta
        gap_days = getattr(self.data_config, "gap_days", 30) or 30
        df, fill = resolve_anchor_gwl(df, current_date, gap_days)         # fresh / forward-fill / (never dropped)
        if df is None:
            return None, meta
        if fill is not None:
            self._note(f"WRIS: {station_code} current GWL carried forward from "
                       f"{fill['last_date']} ({fill['age_days']}d stale; live-source lag)")
        meta["data_age_days"] = fill["age_days"] if fill else 0
        return df, meta

    def _note(self, msg):
        with self._notes_lock:
            self._notes.append(msg)

    def drain_notes(self) -> "list[str]":
        with self._notes_lock:
            n = list(dict.fromkeys(self._notes))
            self._notes = []
        return n

    def _post(self, code, start, end):
        import requests
        if not self.verify:
            import urllib3
            urllib3.disable_warnings()
        payload = {"station_code": code, "starttime": start, "endtime": end, "dataset": WRIS_DATASET_CODE}
        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(WRIS_BASE_URL + _ENDPOINT, headers=_WRIS_HEADERS, json=payload,
                                  timeout=self.timeout, verify=self.verify)
                d = r.json()
                if d.get("statusCode") == 200:
                    return d.get("data", []) or []
                last = f"statusCode={d.get('statusCode')} http={r.status_code}"
            except Exception as e:  # timeout / connection / SSL / JSON = transport failure = outage
                last = type(e).__name__
                self._down = True   # fail-fast: skip WRIS for the rest of THIS request (concurrent fetch)
            if attempt < self.retries - 1:
                _time.sleep(1.0)
        self._note(f"WRIS live unavailable for {code} ({last})")
        return None
