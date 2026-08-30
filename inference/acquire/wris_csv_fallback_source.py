"""wris_csv_fallback_source.py — WRIS-PRIMARY normals source with a CSV snapshot BACKUP.

The MIRROR IMAGE of CsvWithWrisFallbackSource: instead of "CSV first, WRIS to fill a gap", this
does "WRIS first, CSV to fill a gap". For EACH neighbour it fetches ~10 yr of GWL history LIVE
from WRIS; a well WRIS cannot serve — or can only serve a thin series when the CSV is materially
richer — falls back to the LocalCsvSource training snapshot. Same `.fetch(neighbours, current_date)`
interface as LocalCsvSource, so it drops straight into the advisory's `normals_source` seam.

WHY (branch wris-primary-normals):
  Prefer FRESH live WRIS history over the static training snapshot for the seasonal / year-over-year
  NORMALS — the CSV snapshot ages (a distributed release is never re-exported), whereas WRIS
  self-freshens with no operator action. The forecast (Stage 2) is UNTOUCHED — it always uses the
  engine's own live source (NWDP) — so this changes ONLY the normals + regime bins, never
  `forecast_change_m`.

QUALITY-AWARE (never DOWNGRADE):
  A blind flip would replace a rich CSV history with a THIN WRIS one for any well WRIS can serve
  (>= min_rows). Instead, per well: use WRIS only when it is usable AND either (a) rich enough on its
  own (>= min_good_readings) OR (b) not sparser than the CSV (wris_n >= csv_n). Otherwise keep the
  richer CSV. So WRIS wins on the wells that matter (fresh + adequate) but a sparse WRIS series never
  displaces a solid CSV one. The `min_good_readings` escape hatch keeps WRIS for a well that is
  adequate on its own even when the CSV is a very dense telemetry series (density beyond ~seasonal
  resolution adds nothing to a 10-yr normal).

FAIL-FAST (WRIS outage never stalls a request):
  The hot path uses a SINGLE attempt + a short timeout (no 3x exponential backoff — that is only for
  the CSV-primary BACKUP role). A network/timeout failure trips a per-request circuit breaker: WRIS is
  skipped for the remaining wells THIS request (they take the CSV snapshot), so a WRIS outage costs one
  short timeout, not one per neighbour. A non-200 statusCode (station-level "no data") does NOT trip
  the breaker — only a transport failure does. Worst case degrades to exactly the CSV-primary behavior.

PARITY / SAFETY:
  - REUSES the exact WRIS parse of CsvWithWrisFallbackSource (depth-only datatype filter dropping AMSL
    elevation rows, per-day mean, canonical clean_gwl_values abs()+caps). Nothing re-derived.
  - Additive & self-contained: does NOT modify CsvWithWrisFallbackSource, so the CSV-primary
    ("arm A" / 8101) path stays byte-identical.

CONFIDENCE:
  WRIS-primary wells are deliberately NOT tagged `from_fallback` (WRIS is the INTENDED primary here;
  the CSV backup is the most-validated source) — confidence stays driven by consensus + freshness.

Enabled by GWL_NORMALS_WRIS_PRIMARY=1 (needs GWL_NORMALS_CSV for the backup). verify by
GWL_WRIS_VERIFY (default False, for the broken WRIS cert chain). Wired in serve_advisory.
"""
from __future__ import annotations

import os
import threading
import time as _time

from inference.acquire.csv_wris_fallback_source import (
    _ENDPOINT,
    _LOOKBACK_YEARS,
    _MIN_ROWS,
    _WRIS_HEADERS,
    WRIS_BASE_URL,
    WRIS_DATASET_CODE,
    parse_wris_records,
)

_MIN_GOOD_READINGS = 20   # WRIS "rich enough on its own" floor: >= this -> use WRIS even if CSV denser
_HOT_RETRIES = 1          # fail-fast: single attempt on the primary hot path (backoff is a backup-only luxury)
_HOT_TIMEOUT = 15         # fail-fast: short per-well timeout
_MAX_WORKERS = 4          # concurrent WRIS fetches (one POST/neighbour, independent); GWL_WRIS_CONCURRENCY overrides


def _n_rows(df) -> int:
    """Row count of a GWL DataFrame (WRIS series or CSV history), 0 if None/empty."""
    if df is None:
        return 0
    idx = getattr(df, "index", None)
    return len(idx) if idx is not None else 0


class WrisWithCsvFallbackSource:
    """WRIS-primary + CSV (training-snapshot) per-well backup, quality-aware + fail-fast. Only the GWL
    series is sourced from WRIS; dynamic/static ride along from the CSV primary (normals never use
    dynamic/static)."""

    def __init__(self, primary, data_config, verify=False, min_rows=_MIN_ROWS,
                 lookback_years=_LOOKBACK_YEARS, min_good_readings=_MIN_GOOD_READINGS,
                 retries=_HOT_RETRIES, timeout=_HOT_TIMEOUT, max_workers=None):
        self.primary = primary                 # LocalCsvSource (the BACKUP now)
        self.data_config = data_config
        self.verify = verify
        self.min_rows = min_rows               # WRIS series smaller than this is never usable
        self.min_good_readings = min_good_readings   # ... but >= this is "adequate on its own"
        self.lookback_years = lookback_years
        self.retries = retries                 # fail-fast hot path
        self.timeout = timeout
        self.max_workers = int(max_workers if max_workers is not None
                               else os.environ.get("GWL_WRIS_CONCURRENCY", _MAX_WORKERS))
        self._cache: dict = {}                  # (code, anchor_year) -> gwl_df | None  (history is static)
        self._notes: "list[str]" = []           # surfaced like other sources' drain_notes()
        self._notes_lock = threading.Lock()     # _notes.append from worker threads (concurrent fetch)
        self._down_this_fetch = False           # per-request WRIS circuit breaker (reset each fetch)

    def fetch(self, neighbours, current_date):
        acquired = self.primary.fetch(neighbours, current_date)   # CSV backup (also carries dyn/static)
        self._down_this_fetch = False                             # fresh breaker per request
        # CONCURRENT: each neighbour is an INDEPENDENT WRIS POST (mutates its own AcquiredStation in
        # place), so K wells fetch in parallel -> wall-time ~= the slowest single call, not their sum.
        # The circuit breaker is PRESERVED: the first transport failure trips `_down_this_fetch`, and
        # every worker that has not yet started skips WRIS -> CSV snapshot. So a WRIS outage costs about
        # ONE timeout wave (<= max_workers calls) rather than one-per-neighbour. Order is irrelevant
        # here (each worker touches only its own `a`); `_note`/breaker guarded for thread-safety.
        if self.max_workers > 1 and len(acquired) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(acquired))) as pool:
                list(pool.map(lambda a: self._resolve_one(a, current_date), acquired))
        else:
            for a in acquired:
                self._resolve_one(a, current_date)
        return acquired

    def _resolve_one(self, a, current_date):
        """Resolve ONE neighbour's normals GWL: WRIS-primary, CSV backup, quality-aware. Mutates a.gwl
        in place. Safe to run concurrently across neighbours (independent `a`; shared breaker + notes
        guarded). Preserves the exact serial decision logic."""
        code = a.neighbour.station_code
        csv_n = _n_rows(a.gwl)                                 # what the CSV snapshot has for this well
        if self._down_this_fetch:                             # WRIS already failed hard this request
            if csv_n:
                self._note(f"normals: {code} WRIS down (skipped) -> CSV snapshot")
            return
        try:
            series = self._wris_series(code, current_date)    # try WRIS FIRST
        except Exception as e:  # noqa: BLE001 — belt-and-suspenders: no WRIS problem fails a request
            self._note(f"normals: {code} WRIS error ({type(e).__name__}) -> CSV snapshot")
            series = None
        wris_n = _n_rows(series)
        # QUALITY-AWARE: prefer WRIS, but never downgrade to a thin WRIS series when the CSV is
        # richer. Use WRIS if usable AND (adequate on its own OR at least as rich as the CSV).
        if wris_n >= self.min_rows and (wris_n >= self.min_good_readings or wris_n >= csv_n):
            a.gwl = series                                    # WRIS is primary here
            self._note(f"normals: {code} from WRIS live ({wris_n} readings; csv={csv_n})")
        elif csv_n:                                           # keep the CSV snapshot (no downgrade)
            if wris_n >= self.min_rows:
                self._note(f"normals: {code} WRIS thinner ({wris_n} < csv {csv_n}) -> kept CSV")
            elif wris_n > 0:
                self._note(f"normals: {code} WRIS too few ({wris_n} < min {self.min_rows}) -> CSV")
            else:
                self._note(f"normals: {code} WRIS empty -> CSV snapshot fallback")
        # else: neither WRIS nor CSV has data -> a.gwl stays None (well dropped downstream)

    def _note(self, msg):
        with self._notes_lock:
            self._notes.append(msg)

    def drain_notes(self):
        with self._notes_lock:
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
        if records is None:
            # Transport failure (breaker tripped) -> do NOT cache, so WRIS is retried next request once
            # it recovers. A station-level "no data" (non-200, breaker NOT tripped) is cached as a miss.
            if not self._down_this_fetch:
                self._cache[key] = None
            return None
        series = None
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
        for attempt in range(self.retries):
            try:
                r = requests.post(WRIS_BASE_URL + _ENDPOINT, headers=_WRIS_HEADERS, json=payload,
                                  timeout=self.timeout, verify=self.verify)
                d = r.json()
                if d.get("statusCode") == 200:
                    return d.get("data", []) or []
                last = f"statusCode={d.get('statusCode')} http={r.status_code}"   # station-level; NOT an outage
            except Exception as e:  # timeout / connection / SSL / JSON -> transport failure = outage
                last = type(e).__name__
                self._down_this_fetch = True   # fail-fast: skip WRIS for the rest of THIS request
            if attempt < self.retries - 1:
                _time.sleep(0.5)
        self._notes.append(f"normals: WRIS primary unavailable for {code} ({last})")
        return None
