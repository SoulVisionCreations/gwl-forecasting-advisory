"""NwdpGwlProvider — GWL lookback from the NWDP CKAN portal (WRIS drop-in).

A STOPGAP for the WRIS GWL API while it is down. It supplies ONLY the groundwater
history slice; dynamic climate, static features, composites and the forecast still
come from GEE / Open-Meteo exactly as before. It plugs into NumericFetcher via the
optional `gwl_provider` seam, so the rest of the inference flow is unchanged and the
WRIS path stays byte-identical (default). Flip back with one flag when WRIS recovers.

All the NWDP-specific work lives here:
  - map the query point to its NEAREST NWDP station by coordinates (prefer telemetry
    = 6-hourly = freshest; fall back to manual quarterly), within a distance cutoff;
  - resolve which time-range resource(s) cover the lookback window;
  - datastore_search that station's series (no API key), parse DD-MM-YYYY HH:MM +
    the '...(meter)' value column;
  - clean to match training (drop NaN -> abs() sign-normalisation);
  - ANCHOR POLICY (shared with the WRIS path via inference.acquire.anchor): NWDP
    publishes ~1 quarter behind, so a well's tail is usually stale. resolve_anchor_gwl
    keeps fresh series (<= gap_days) as-is, forward-fills (LOCF) stale ones up to the
    anchor with a note, and treats > max_staleness_days as unusable (caller tries the
    next nearest well). The anchor date is never moved.

Returns the SAME contract WRIS's _parse_gwl_timeseries produces:
  gwl_df = DataFrame(DatetimeIndex, column 'gwl_value'), and meta = {well_type, well_depth}.
"""
from __future__ import annotations

import json
import pickle
import re
from datetime import timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

_API = "https://nwdp.nwic.gov.in/api/3/action"
_VALUE_COL = re.compile(r"Groundwater Level.*\(meter\)", re.IGNORECASE)


def _haversine_km(lat1, lon1, lat2, lon2):
    import numpy as np
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


class NwdpGwlProvider:

    def __init__(self, index_path: str, data_config=None, max_km: float = 25.0,
                 lookback_days: int = 400, max_staleness_days: int = 100,  # = anchor.DEFAULT_MAX_STALENESS_DAYS
                 max_candidates: int = 8, timeout: int = 60):
        import numpy as np
        with open(index_path, "rb") as f:
            idx = pickle.load(f)
        self._idx = idx
        self.data_config = data_config
        self.max_km = float(max_km)                    # search radius for a usable well
        self.lookback_days = int(lookback_days)
        self.max_staleness_days = int(max_staleness_days)  # skip wells whose latest reading is older
        self.max_candidates = int(max_candidates)      # nearest wells to try before giving up
        self.timeout = int(timeout)
        self._lat = np.asarray(idx["lats"], dtype=float)
        self._lon = np.asarray(idx["lons"], dtype=float)
        self._mode = np.asarray(idx["modes"])
        self._tel_mask = self._mode == "telemetry"
        self._notes: list[str] = []
        self._series_cache: dict = {}          # resource_id|station -> raw DataFrame
        self._session = None

    # ------------------------------------------------------------------ seam entry
    def gwl_and_meta(self, station_code: str, lat: float, lon: float, current_date: "datetime"):
        """Return (gwl_df, meta) for the query point, WRIS-compatible. meta has
        well_type/well_depth (NWDP lacks them -> defaults). gwl_df or None if no
        NWDP station within the cutoff (-> neighbour dropped, like WRIS empty)."""
        meta = {"well_type": "no_data", "well_depth": 0.0, "data_age_days": None}
        start = current_date - timedelta(days=self.lookback_days)
        # Fetch a forward buffer PAST the anchor so a PAST-date validation can reach the
        # +horizon target reading for scoring. Production/today queries have no NWDP data
        # past the anchor, so this is a no-op there. The anchor is NEVER moved
        # (resolve_anchor_gwl anchors at current_date) — post-anchor rows only feed the
        # ground-truth lookup, never the sequence or the anchor value.
        fetch_end = current_date + timedelta(days=self._forward_buffer_days())
        # Try nearest wells in order; use the FIRST with usable data. ALSO record how OLD that well's
        # latest REAL reading is (data_age_days) -> the advisory swaps a stale well's model forecast for
        # a seasonal analog (docs/stale_data_seasonal_analog). We do NOT reach out for freshness here.
        import pandas as pd
        anchor = pd.Timestamp(current_date).normalize()
        for i, dist_km in self._candidates(lat, lon):
            name = self._idx["names"][i]
            key = self._idx["group_keys"][i]
            records = self._fetch_station(name, key, start, fetch_end)
            gwl_df = self._to_gwl_df(records, start, fetch_end, current_date, name, dist_km)
            if gwl_df is not None:
                latest = self._latest_real_date(records, anchor)
                meta["data_age_days"] = None if latest is None else int((anchor - latest).days)
                return gwl_df, meta
        return None, meta

    def _latest_real_date(self, records, anchor):
        """Newest real 'Data Acquisition Time' on/before the anchor (pre-forward-fill), else None."""
        import pandas as pd
        if records is None or not len(records) or "Data Acquisition Time" not in records.columns:
            return None
        dts = pd.to_datetime(records["Data Acquisition Time"], format="%d-%m-%Y %H:%M",
                             errors="coerce").dropna()
        dts = dts[dts.dt.normalize() <= anchor]
        return dts.max().normalize() if len(dts) else None

    def _forward_buffer_days(self) -> int:
        """Days to fetch PAST the anchor so a past-date validation can reach the +horizon
        target (+gap tolerance). Today/production queries have no future data -> no-op."""
        horizon_m = getattr(self.data_config, "forecast_horizon_months", 3) or 3
        gap = getattr(self.data_config, "gap_days", 30) or 30
        return int(horizon_m) * 31 + int(gap) + 5

    def drain_notes(self) -> "list[str]":
        n = list(dict.fromkeys(self._notes))   # dedup, keep order
        self._notes = []
        return n

    # ------------------------------------------------------------------ internals
    def _candidates(self, lat: float, lon: float):
        """The nearest wells within max_km, nearest first, capped to max_candidates.
        (The caller keeps the first with recent-enough data — telemetry wells, being
        6-hourly, usually win on recency; a nearby well is preferred spatially.)"""
        import numpy as np
        d = _haversine_km(lat, lon, self._lat, self._lon)
        out = []
        for j in np.argsort(d):
            if d[j] > self.max_km or len(out) >= self.max_candidates:
                break
            out.append((int(j), float(d[j])))
        return out

    def _resources_for(self, group_key: str, start: "datetime", end: "datetime"):
        """The resource ids whose [y0,y1] overlaps [start.year, end.year]."""
        out = []
        for y0, y1, rid in self._idx["groups"].get(group_key, []):
            if not (y1 < start.year or y0 > end.year):
                out.append(rid)
        return out

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()   # trusts *_proxy env (big-10 Squid)
        return self._session

    def _fetch_station(self, name: str, group_key: str, start: "datetime", end: "datetime"):
        """datastore_search each covering resource for this station (cached)."""
        import pandas as pd
        session = self._get_session()
        frames = []
        for rid in self._resources_for(group_key, start, end):
            ck = f"{rid}|{name}"
            if ck in self._series_cache:
                frames.append(self._series_cache[ck]); continue
            recs = []
            try:
                r = session.get(f"{_API}/datastore_search",
                                params={"resource_id": rid,
                                        "filters": json.dumps({"Station": name}),
                                        "limit": 100000},
                                timeout=self.timeout)
                r.raise_for_status()
                j = r.json()
                if j.get("success"):
                    recs = j["result"].get("records", [])
            except Exception:  # noqa: BLE001 — soft: missing resource -> no rows
                recs = []
            dfp = pd.DataFrame(recs)
            self._series_cache[ck] = dfp
            frames.append(dfp)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def _to_gwl_df(self, records, start: "datetime", window_end: "datetime",
                   current_date: "datetime", name: str, dist_km: float):
        """records -> DataFrame(DatetimeIndex, 'gwl_value'), cleaned like training. The
        series is bounded ABOVE by window_end (= anchor + forward buffer) so a past-date
        validation can find the +horizon target; it is NOT lower-bounded, so a well whose
        newest reading is far older than nwdp_lookback_days still survives (the anchor,
        current_date, is forward-filled per the shared policy and NEVER moved)."""
        import pandas as pd

        if records is None or not len(records):
            return None
        cols = list(records.columns)
        vcol = next((c for c in cols if _VALUE_COL.search(str(c))), None)
        if vcol is None or "Data Acquisition Time" not in cols:
            return None

        df = pd.DataFrame({
            "date": pd.to_datetime(records["Data Acquisition Time"],
                                   format="%d-%m-%Y %H:%M", errors="coerce"),
            "gwl_value": pd.to_numeric(records[vcol], errors="coerce"),
        }).dropna(subset=["date", "gwl_value"])
        if df.empty:
            return None

        # training-parity cleaning (shared): drop NaN → abs() sign-norm → hard caps.
        from gwlcore.data_preparation import clean_gwl_values
        df = clean_gwl_values(df, self.data_config)
        if df.empty:
            return None

        # daily series (last obs per day), sorted. Upper-bound to window_end (the post-anchor
        # buffer feeds the +horizon validation target). NO lower bound: NWDP lags badly, so a
        # well's newest reading can be far older than nwdp_lookback_days; we still keep it so
        # resolve_anchor_gwl can forward-fill it to the anchor. Truly no staleness cap — the
        # advisory degrades CONFIDENCE by age instead of dropping the well (see anchor.py).
        df = (df.sort_values("date")
                .assign(day=lambda d: d["date"].dt.normalize())
                .drop_duplicates("day", keep="last"))
        df = df[df["day"] <= pd.Timestamp(window_end.date())]
        if df.empty:
            return None
        df = df[["day", "gwl_value"]].rename(columns={"day": "date"}).reset_index(drop=True)

        # Shared anchor policy (identical to the WRIS path): fresh -> unchanged;
        # stale -> forward-fill (LOCF) to the anchor + note; too stale -> drop (caller
        # tries the next nearest well). Anchor date is never moved.
        from inference.acquire.anchor import resolve_anchor_gwl
        gap_days = getattr(self.data_config, "gap_days", 30) or 30
        df, fill = resolve_anchor_gwl(df, current_date, gap_days, self.max_staleness_days)
        if df is None:
            return None
        if fill is not None:
            self._notes.append(
                f"NWDP: current GWL carried forward from {fill['last_date']} "
                f"({fill['age_days']}d stale; nearest well '{name}' {dist_km:.1f} km; live-source lag)")

        return df.set_index("date")[["gwl_value"]]
