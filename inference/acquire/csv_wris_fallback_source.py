"""csv_wris_fallback_source.py — CSV-primary normals source with a WRIS per-well BACKUP.

The advisory's normals / consensus / analog read ~10 yr of per-neighbour GWL history from a
LOCAL SNAPSHOT (LocalCsvSource over data_with_flag_all.csv). This wrapper keeps that CSV as the
DEFAULT and, ONLY for a neighbour the CSV has NO data for at all, fetches that ONE well's history
LIVE from WRIS (historical GWATERLVL) — so a coverage gap yields a confidence-capped answer instead
of a decline. A well with ANY CSV history keeps the CSV (the backup never second-guesses the snapshot).

Guarantees (why it "doesn't change things much"):
  - A query whose neighbours are ALL in the CSV is byte-identical to LocalCsvSource (no WRIS call).
  - The WRIS path is ADDITIVE: it only fills a well the CSV could not, never overrides a CSV well.
  - WRIS records are DATATYPE-FILTERED to depth-to-water (the AMSL *elevation* records WRIS returns
    for the same station — e.g. -534 m — are dropped; averaging them in would be garbage), then run
    through the SAME canonical clean_gwl_values (abs() sign-norm + caps) as the CSV -> consistent.
  - Wells filled from WRIS are TAGGED (AcquiredStation.from_fallback) so the advisory caps
    confidence at MEDIUM (a backup-sourced normal is less validated than the training snapshot).

Enabled by GWL_NORMALS_WRIS_FALLBACK=1 ; verify by GWL_WRIS_VERIFY (default False, for the broken
WRIS cert chain — mirrors curl -k). Wired in serve_advisory. Endpoint + probe findings:
reference_gwl_source_alternatives.
"""
from __future__ import annotations

import time as _time

from inference.acquire.vendor.data_fetcher import WRIS_BASE_URL, WRIS_DATASET_CODE

_ENDPOINT = "/CommonDataSetMasterAPI/getCommonDataSetByStationCode"
_WRIS_HEADERS = {"Origin": WRIS_BASE_URL, "Referer": WRIS_BASE_URL + "/dataSet/",
                 "User-Agent": "Mozilla/5.0"}
_MIN_ROWS = 4          # a WRIS series with fewer readings isn't worth injecting
_LOOKBACK_YEARS = 12   # fetch this many years back from the anchor (normals need ~10)
_RETRIES = 3
_TIMEOUT = 60


def _is_depth(desc) -> bool:
    """Keep depth-to-water datatypes (HGZ/GGZ '...Water Level'); DROP the 'AMSL' elevation
    records WRIS returns for the same station (metres above sea level, not depth)."""
    d = str(desc or "")
    return ("Water Level" in d) and ("AMSL" not in d)


def parse_wris_records(records, station_code, data_config):
    """Raw WRIS timeseries -> cleaned per-day GWL DataFrame (DatetimeIndex + 'gwl_value'), or None.

    Depth-only datatypes; sub-daily collapsed to a per-day mean; then the canonical
    clean_gwl_values (abs() + configured caps) for parity with the CSV/training. Standalone
    (no network) so it is unit-testable.
    """
    import pandas as pd

    from gwlcore.data_preparation import clean_gwl_values

    rows = [(r.get("dataTime"), r.get("dataValue")) for r in (records or [])
            if _is_depth(r.get("datatypeDescription"))]
    rows = [(t, v) for (t, v) in rows if t is not None and v is not None]
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "gwl_value"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["gwl_value"] = pd.to_numeric(df["gwl_value"], errors="coerce")
    df = df.dropna(subset=["date", "gwl_value"])
    if df.empty:
        return None
    df = df.groupby("date", as_index=False)["gwl_value"].mean()   # collapse sub-daily readings
    df["station_code"] = str(station_code)
    df = clean_gwl_values(df.sort_values(["station_code", "date"]), data_config)  # abs + caps (parity)
    if df is None or not len(df):
        return None
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["gwl_value"]].sort_index()


class CsvWithWrisFallbackSource:
    """LocalCsvSource primary + WRIS per-well backup. Same `.fetch(neighbours, current_date)`
    interface as LocalCsvSource / NumericFetcher, so it drops straight into the advisory's
    normals_source seam. Only the GWL series is filled (normals never use dynamic/static)."""

    def __init__(self, primary, data_config, verify=False, min_rows=_MIN_ROWS,
                 lookback_years=_LOOKBACK_YEARS):
        self.primary = primary                 # LocalCsvSource (the DEFAULT 10-yr snapshot)
        self.data_config = data_config
        self.verify = verify
        self.min_rows = min_rows
        self.lookback_years = lookback_years
        self._cache: dict = {}                  # (code, anchor_year) -> gwl_df | None  (history is static)
        self._notes: "list[str]" = []           # surfaced like other sources' drain_notes()

    def fetch(self, neighbours, current_date):
        acquired = self.primary.fetch(neighbours, current_date)
        for a in acquired:
            g = a.gwl
            if g is not None and len(getattr(g, "index", [])) > 0:
                continue                                     # CSV has ANY data for this well -> untouched (DEFAULT)
            code = a.neighbour.station_code
            series = self._wris_series(code, current_date)   # try the backup for THIS well only
            if series is not None and len(series) >= self.min_rows:
                a.gwl = series
                a.from_fallback = True                       # -> advisory caps confidence at MEDIUM
                self._notes.append(f"normals: {code} filled from WRIS backup ({len(series)} readings)")
        return acquired

    def drain_notes(self):
        n = list(self._notes)
        self._notes.clear()
        return n

    # ------------------------------------------------------------------ WRIS
    def _wris_series(self, code, current_date):
        key = (code, current_date.year)
        if key in self._cache:
            return self._cache[key]
        records = self._post(code, f"{current_date.year - self.lookback_years}-01-01",
                             current_date.strftime("%Y-%m-%d"))
        series = None
        if records is not None:
            try:
                series = parse_wris_records(records, code, self.data_config)
            except Exception as e:  # noqa: BLE001 — a bad well must never sink the request
                self._notes.append(f"normals: WRIS parse failed for {code} ({type(e).__name__})")
        self._cache[key] = series
        return series

    def _post(self, code, start, end):
        import requests
        if not self.verify:
            import urllib3
            urllib3.disable_warnings()
        payload = {"station_code": code, "starttime": start, "endtime": end, "dataset": WRIS_DATASET_CODE}
        last = None
        for attempt in range(_RETRIES):
            try:
                r = requests.post(WRIS_BASE_URL + _ENDPOINT, headers=_WRIS_HEADERS, json=payload,
                                  timeout=_TIMEOUT, verify=self.verify)
                d = r.json()
                if d.get("statusCode") == 200:
                    return d.get("data", []) or []
                last = f"statusCode={d.get('statusCode')} http={r.status_code}"
            except Exception as e:  # timeout / connection / SSL / JSON
                last = type(e).__name__
            if attempt < _RETRIES - 1:
                _time.sleep(1.5 * (2 ** attempt))
        self._notes.append(f"normals: WRIS backup unavailable for {code} ({last})")
        return None
