"""Layer-1 (offline, deterministic) tests for the CONCURRENT WRIS fetch + circuit breaker.

No network: the one I/O method (`_post` / `_wris_series` / provider.gwl_and_meta) is monkeypatched with
a fake that injects latency and failures. We assert: (1) correctness + order are unchanged vs serial,
(2) the threadpool actually overlaps (wall-time ~= slowest call, not the sum), (3) the circuit breaker
still bounds a WRIS outage to ~one wave (<= max_workers calls), everything else falling back to CSV.

Run:  python -m unittest tests.test_wris_concurrent -v      (from the repo root)
"""
import hashlib
import os
import threading
import time
import types
import unittest
from datetime import datetime

import pandas as pd

from inference.acquire.numeric_fetcher import NumericFetcher
from inference.acquire.wris_csv_fallback_source import WrisWithCsvFallbackSource
from inference.acquire.wris_gwl_provider import WrisGwlProvider


def _rowsig(obj):
    """Order-PRESERVING signature of a GWL Series/DataFrame (rows as-is, not sorted)."""
    if obj is None:
        return None
    if hasattr(obj, "columns"):
        vals = obj["gwl_value"] if "gwl_value" in obj.columns else obj.iloc[:, 0]
        idx = obj["date"] if "date" in obj.columns else obj.index
        return tuple((str(d)[:10], round(float(v), 4)) for d, v in zip(list(idx), list(vals)))
    return tuple((str(d)[:10], round(float(v), 4)) for d, v in zip(list(obj.index), list(obj.values)))


def _uniq_series(code, n=20):
    """A UNIQUE-per-station date-indexed series (seeded by the code)."""
    seed = int(hashlib.md5(code.encode()).hexdigest()[:6], 16)
    idx = pd.date_range("2016-01-01", periods=n, freq="45D")
    return pd.Series([round(1.0 + (seed % 1000) / 100.0 + i * 0.1, 3) for i in range(n)],
                     index=idx, name="gwl_value")


def _series(n):
    """A date-indexed GWL series of length n (values irrelevant; only the row count drives the
    quality-aware WRIS-vs-CSV decision)."""
    if n <= 0:
        return None
    idx = pd.date_range("2015-01-01", periods=n, freq="30D")
    return pd.Series(range(n), index=idx, name="gwl_value")


def _station(code, csv_rows):
    """A fake AcquiredStation carrying the CSV-snapshot backup for one well."""
    return types.SimpleNamespace(
        neighbour=types.SimpleNamespace(station_code=code, lat=13.0, lon=76.0),
        gwl=_series(csv_rows),
    )


class _FakePrimary:
    """Stands in for LocalCsvSource: returns the CSV-snapshot AcquiredStations for the neighbours."""
    def __init__(self, csv_rows_by_code):
        self.csv_rows_by_code = csv_rows_by_code

    def fetch(self, neighbours, current_date):
        return [_station(c, self.csv_rows_by_code[c]) for c in [n.station_code for n in neighbours]]


def _neighbours(codes):
    return [types.SimpleNamespace(station_code=c, lat=13.0, lon=76.0) for c in codes]


# ----------------------------------------------------------------------------- normals source
class TestNormalsConcurrency(unittest.TestCase):

    def _make(self, csv_rows_by_code, max_workers):
        src = WrisWithCsvFallbackSource(_FakePrimary(csv_rows_by_code), data_config=None,
                                        max_workers=max_workers)
        return src

    def test_correctness_and_order_parallel_equals_serial(self):
        # 4 wells exercising every branch of the quality-aware decision:
        #   A: WRIS rich (30) > CSV thin (5)         -> WRIS wins (>= min_good_readings=20)
        #   B: WRIS thin (3 < min_rows=4), CSV (50)   -> CSV kept (no downgrade)
        #   C: WRIS empty, CSV (10)                   -> CSV
        #   D: WRIS medium (10), CSV (8): 10 >= csv 8 -> WRIS wins (not sparser than CSV)
        csv = {"A": 5, "B": 50, "C": 10, "D": 8}
        wris = {"A": 30, "B": 3, "C": 0, "D": 10}
        expect_source = {"A": "wris", "B": "csv", "C": "csv", "D": "wris"}
        expect_len = {"A": 30, "B": 50, "C": 10, "D": 10}
        nbs = _neighbours(["A", "B", "C", "D"])

        def run(max_workers):
            src = self._make(csv, max_workers)
            src._wris_series = lambda code, cur: _series(wris[code])   # mock the WRIS layer
            out = src.fetch(nbs, None)
            return {a.neighbour.station_code: (None if a.gwl is None else len(a.gwl)) for a in out}, \
                   [a.neighbour.station_code for a in out]

        par_lens, par_order = run(4)
        ser_lens, ser_order = run(1)
        self.assertEqual(par_order, ["A", "B", "C", "D"], "order must be preserved")
        self.assertEqual(par_order, ser_order)
        self.assertEqual(par_lens, ser_lens, "parallel result must equal serial result")
        for c in "ABCD":
            self.assertEqual(par_lens[c], expect_len[c], f"{c}: wrong series ({expect_source[c]} expected)")

    def test_threadpool_actually_overlaps(self):
        # 8 wells, each WRIS call takes 0.2s. Serial would be 1.6s; parallel@4 must be ~2 waves (~0.4s).
        codes = [f"W{i}" for i in range(8)]
        csv = {c: 5 for c in codes}
        src = self._make(csv, max_workers=4)

        def slow_wris(code, cur):
            time.sleep(0.2)
            return _series(30)     # rich -> WRIS wins, so the fetch really used the (slow) WRIS path
        src._wris_series = slow_wris

        t0 = time.time()
        out = src.fetch(_neighbours(codes), None)
        elapsed = time.time() - t0
        self.assertTrue(all(len(a.gwl) == 30 for a in out), "all wells should have taken WRIS")
        self.assertGreater(elapsed, 0.2, "not instant — the slow path really ran")
        self.assertLess(elapsed, 0.6 * (8 * 0.2), f"parallel {elapsed:.2f}s must beat serial 1.6s")

    def test_circuit_breaker_bounds_outage(self):
        # WRIS is DOWN: every _post is a transport failure. The breaker must skip WRIS for wells beyond
        # the first wave -> <= max_workers actual _post calls, wall-time ~= one timeout, all -> CSV.
        codes = [f"D{i}" for i in range(10)]
        csv = {c: 12 for c in codes}
        src = self._make(csv, max_workers=4)
        calls = {"n": 0}
        lock = threading.Lock()

        def dead_post(code, start, end):
            with lock:
                calls["n"] += 1
            time.sleep(0.2)                 # the connect-timeout wait
            src._down_this_fetch = True     # transport failure trips the breaker (as real _post does)
            return None
        src._post = dead_post

        t0 = time.time()
        out = src.fetch(_neighbours(codes), None)
        elapsed = time.time() - t0
        self.assertLessEqual(calls["n"], 4, f"breaker should cap _post at max_workers; got {calls['n']}")
        self.assertLess(elapsed, 0.6, f"outage bounded to ~one wave; got {elapsed:.2f}s")
        self.assertTrue(all(len(a.gwl) == 12 for a in out), "every well must fall back to the CSV snapshot")


# ----------------------------------------------------------------------------- forecast provider
class TestForecastProviderBreaker(unittest.TestCase):

    def test_breaker_skips_after_transport_failure_serial(self):
        prov = WrisGwlProvider(data_config=None)
        calls = {"n": 0}

        def dead_post(code, start, end):
            calls["n"] += 1
            prov._down = True     # transport failure (as the real _post's except does)
            return None
        prov._post = dead_post

        import datetime as dt
        cur = dt.datetime(2026, 8, 17)
        prov.reset_breaker()
        results = [prov.gwl_and_meta(f"S{i}", 13.0, 76.0, cur) for i in range(6)]
        self.assertEqual(calls["n"], 1, "after the first transport failure, the rest skip _post")
        self.assertTrue(all(df is None for df, _ in results), "all return None (caller uses fallback)")

    def test_breaker_bounds_outage_parallel(self):
        # Mirror how NumericFetcher calls it: many concurrent gwl_and_meta with WRIS down. First wave
        # (<= max_workers) hit _post; the rest see the tripped breaker and skip.
        from concurrent.futures import ThreadPoolExecutor
        import datetime as dt
        prov = WrisGwlProvider(data_config=None)
        calls = {"n": 0}
        lock = threading.Lock()

        def dead_post(code, start, end):
            with lock:
                calls["n"] += 1
            time.sleep(0.2)
            prov._down = True
            return None
        prov._post = dead_post
        prov.reset_breaker()

        cur = dt.datetime(2026, 8, 17)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda i: prov.gwl_and_meta(f"S{i}", 13.0, 76.0, cur), range(10)))
        elapsed = time.time() - t0
        self.assertLessEqual(calls["n"], 4, f"breaker should cap _post at max_workers; got {calls['n']}")
        self.assertLess(elapsed, 0.6, f"outage bounded to ~one wave; got {elapsed:.2f}s")

    def test_reset_breaker_clears(self):
        prov = WrisGwlProvider(data_config=None)
        prov._down = True
        prov.reset_breaker()
        self.assertFalse(prov._down)


# ----------------------------------------------------------------- serial==concurrent parity (extra)
class TestSerialConcurrentParity(unittest.TestCase):
    """Stronger identity checks: a WRIS series is UNORDERED on arrival and we sort it, so serial and
    concurrent must yield byte-identical, same-order data per station — every time, in every regime."""

    @staticmethod
    def _outsig(out):
        """(station_code, order-preserving series signature) in OUTPUT ORDER -> catches data AND
        row-order AND station-position differences in one comparison."""
        return [(a.neighbour.station_code, _rowsig(a.gwl)) for a in out]

    def test_stress_concurrent_equals_serial_over_many_runs(self):
        # A single concurrent run passing can be luck; run it 40x and demand it equals the serial
        # baseline EVERY time. Tiny sleep in the WRIS layer widens the interleave/race window.
        codes = [f"S{i:03d}" for i in range(24)]
        nbs = _neighbours(codes)

        def make(mw):
            src = WrisWithCsvFallbackSource(_FakePrimary({c: 0 for c in codes}), None, max_workers=mw)
            def slow(code, cur):
                time.sleep(0.003)
                return _uniq_series(code)
            src._wris_series = slow
            return src

        baseline = self._outsig(make(1).fetch(nbs, None))
        self.assertEqual(len(baseline), 24)
        self.assertEqual(len({s for _, s in baseline}), 24, "each station must have its own unique series")
        for run in range(40):
            got = self._outsig(make(8).fetch(nbs, None))
            self.assertEqual(got, baseline, f"concurrent run {run} diverged from serial (data/order/position)")

    def test_mixed_wris_csv_quality_is_concurrency_invariant(self):
        # (csv_rows, wris_rows) chosen to hit each quality-aware branch: WRIS-wins / CSV-kept / empties.
        spec = {"A": (5, 30), "B": (50, 3), "C": (10, 0), "D": (8, 10), "E": (0, 0), "F": (40, 40)}
        nbs = _neighbours(list(spec))

        def build(mw):
            src = WrisWithCsvFallbackSource(_FakePrimary({c: spec[c][0] for c in spec}), None, max_workers=mw)
            src._wris_series = lambda code, cur: _series(spec[code][1])
            return [(a.neighbour.station_code, None if a.gwl is None else len(a.gwl))
                    for a in src.fetch(nbs, None)]

        self.assertEqual(build(8), build(1), "the WRIS-vs-CSV decision must not depend on concurrency")

    def test_downcase_fallback_is_concurrency_invariant(self):
        # WRIS down: both serial & concurrent must fall back to the SAME per-station CSV series + order.
        codes = [f"D{i}" for i in range(12)]
        nbs = _neighbours(codes)

        def build(mw):
            src = WrisWithCsvFallbackSource(_FakePrimary({c: 9 for c in codes}), None, max_workers=mw)
            def dead(code, start, end):
                time.sleep(0.002)
                src._down_this_fetch = True
                return None
            src._post = dead
            return [(a.neighbour.station_code, None if a.gwl is None else len(a.gwl))
                    for a in src.fetch(nbs, None)]

        self.assertEqual(build(8), build(1))

    def test_numeric_fetcher_static_dynamic_gwl_stay_aligned(self):
        # The forecast-path risk: worker for station i must read dynamic_results[i] + static_results[i]
        # (not some other thread's). Station i is tagged with value i across gwl / dynamic / static /
        # meta; assert alignment holds identically at max_workers=1 and 8.
        import inference.acquire.vendor.new_data_fetcher as ndf
        saved = {k: getattr(ndf, k, None) for k in
                 ("_init_gee", "_make_points_fc", "_batch_get_historical_features",
                  "_batch_get_all_static", "DataFetcher")}
        env_saved = {k: os.environ.get(k) for k in
                     ("GWL_WRIS_CONCURRENCY", "GWL_FETCH_CLIMATOLOGY_ONLY", "GWL_FETCH_MINIMAL",
                      "GWL_SKIP_UNUSED_COLLECTIONS")}

        def restore():
            for k, v in saved.items():
                if v is not None:
                    setattr(ndf, k, v)
            for k, v in env_saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.addCleanup(restore)

        ndf._init_gee = lambda: None
        ndf._make_points_fc = lambda stations: None
        ndf._batch_get_historical_features = lambda stations, fc, s, e, keep_features=None: {
            i: pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2), "rainfall": [i, i]})
            for i in range(len(stations))}
        ndf._batch_get_all_static = lambda stations, fc: [{"elev": i} for i in range(len(stations))]
        ndf.DataFetcher = lambda: types.SimpleNamespace()
        for k in ("GWL_FETCH_CLIMATOLOGY_ONLY", "GWL_FETCH_MINIMAL", "GWL_SKIP_UNUSED_COLLECTIONS"):
            os.environ.pop(k, None)

        class _FakeProv:
            def __init__(self):
                self._down = False
            def reset_breaker(self):
                self._down = False
            def gwl_and_meta(self, code, lat, lon, cur):
                time.sleep(0.002)
                i = int(code[1:])
                df = pd.DataFrame({"date": pd.date_range("2021-01-01", periods=3), "gwl_value": [i, i, i]})
                return df, {"data_age_days": i, "well_type": "bore"}

        dc = types.SimpleNamespace(gap_days=30, forecast_horizon_months=3)
        nbs = [types.SimpleNamespace(station_code=f"N{i:03d}", lat=13.0, lon=76.0) for i in range(20)]

        def run(mw):
            os.environ["GWL_WRIS_CONCURRENCY"] = str(mw)
            nf = NumericFetcher(gwl_provider=_FakeProv(), data_config=dc)
            out = nf.fetch(nbs, datetime(2026, 8, 17))
            return [(a.neighbour.station_code,
                     float(a.gwl["gwl_value"].iloc[0]),
                     None if a.dynamic.empty else float(a.dynamic["rainfall"].iloc[0]),
                     a.static.get("elev"),
                     a.data_age_days) for a in out]

        serial = run(1)
        concurrent = run(8)
        self.assertEqual(concurrent, serial, "serial vs concurrent differ (numeric_fetcher)")
        for i, (code, g, rain, elev, age) in enumerate(concurrent):
            self.assertEqual((code, g, rain, elev, age), (f"N{i:03d}", float(i), float(i), i, i),
                             f"station {i} mis-aligned: gwl/dynamic/static/meta must all be i")


if __name__ == "__main__":
    unittest.main(verbosity=2)
