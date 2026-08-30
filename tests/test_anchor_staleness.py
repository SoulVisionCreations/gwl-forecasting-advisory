"""Staleness drop + next-well fallthrough (re-enabled at 100 days).

resolve_anchor_gwl is the shared anchor policy for BOTH WRIS and NWDP. Its freshness tiers:
  age <= gap_days           -> fresh, series unchanged
  gap_days < age <= 100      -> stale, forward-fill (LOCF) to the anchor
  age > 100                  -> TOO STALE -> (None, None): drop the well
  no reading at/behind anchor -> (None, None): unusable
When every candidate well is dropped, the caller has no usable live reading -> seasonal analog.

Run:  python -m unittest tests.test_anchor_staleness -v
"""
import types
import unittest
from datetime import datetime, timedelta

import pandas as pd

from inference.acquire.anchor import resolve_anchor_gwl, DEFAULT_MAX_STALENESS_DAYS
from inference.acquire.nwdp_source import NwdpGwlProvider

ANCHOR = datetime(2026, 8, 17)


def _df_ending(days_before, n=5):
    """A GWL DataFrame whose NEWEST reading is `days_before` days before the anchor."""
    last = ANCHOR - timedelta(days=days_before)
    return pd.DataFrame({"date": pd.date_range(end=last, periods=n, freq="10D"),
                         "gwl_value": [5.0] * n})


class TestResolveAnchorTiers(unittest.TestCase):

    def test_default_threshold_is_100(self):
        self.assertEqual(DEFAULT_MAX_STALENESS_DAYS, 100)

    def test_fresh_within_gap_returned_unchanged(self):
        df = _df_ending(10)                       # 10d <= gap_days(30)
        out, fill = resolve_anchor_gwl(df, ANCHOR, gap_days=30, max_staleness_days=100)
        self.assertIsNotNone(out)
        self.assertIsNone(fill)                   # no forward-fill note
        self.assertEqual(len(out), len(df))       # untouched

    def test_stale_but_usable_is_forward_filled(self):
        df = _df_ending(60)                       # 30 < 60 <= 100
        out, fill = resolve_anchor_gwl(df, ANCHOR, gap_days=30, max_staleness_days=100)
        self.assertIsNotNone(out)
        self.assertIsNotNone(fill)
        self.assertEqual(fill["age_days"], 60)
        # forward-filled up to the anchor
        self.assertEqual(pd.Timestamp(out["date"].max()).normalize(), pd.Timestamp(ANCHOR).normalize())

    def test_too_stale_is_dropped(self):
        df = _df_ending(150)                      # 150 > 100 -> DROP
        out, fill = resolve_anchor_gwl(df, ANCHOR, gap_days=30, max_staleness_days=100)
        self.assertIsNone(out)
        self.assertIsNone(fill)

    def test_boundary_100_kept_101_dropped(self):
        out100, _ = resolve_anchor_gwl(_df_ending(100), ANCHOR, gap_days=30, max_staleness_days=100)
        out101, _ = resolve_anchor_gwl(_df_ending(101), ANCHOR, gap_days=30, max_staleness_days=100)
        self.assertIsNotNone(out100, "age == max_staleness is still usable (forward-filled)")
        self.assertIsNone(out101, "age == max_staleness + 1 is dropped")

    def test_no_reading_behind_anchor_dropped(self):
        future = pd.DataFrame({"date": pd.date_range(start=ANCHOR + timedelta(days=5), periods=3),
                               "gwl_value": [5.0, 5.0, 5.0]})
        out, fill = resolve_anchor_gwl(future, ANCHOR, gap_days=30, max_staleness_days=100)
        self.assertIsNone(out)
        self.assertIsNone(fill)


class TestNwdpFallthrough(unittest.TestCase):
    """The nearest well being too stale must not shadow a fresher next-nearest well."""

    def _provider(self, fetch_map, candidates):
        p = NwdpGwlProvider.__new__(NwdpGwlProvider)     # bypass pickle load
        p._idx = {"names": [c[2] for c in candidates],
                  "group_keys": ["g"] * len(candidates)}
        p.data_config = types.SimpleNamespace(gap_days=30, forecast_horizon_months=3,
                                              max_gwl=0, min_gwl=0)
        p.max_staleness_days = 100
        p.lookback_days = 400
        p._candidates = lambda lat, lon: [(i, dist) for i, (dist, _, _) in enumerate(candidates)]
        p._fetch_station = lambda name, key, s, e: fetch_map[name]
        return p

    @staticmethod
    def _recs(date_str, value):
        return pd.DataFrame({"Data Acquisition Time": [date_str],
                             "Groundwater Level Telemetry 6 Hourly (meter)": [str(value)]})

    def test_stale_nearest_falls_through_to_fresh_next(self):
        candidates = [(1.9, "manual", "StaleWell"), (10.9, "tele", "FreshWell")]
        fetch = {"StaleWell": self._recs("25-08-2024 12:00", 7.0),   # ~722d stale -> dropped
                 "FreshWell": self._recs("15-08-2026 12:00", 4.5)}   # ~2d fresh -> used
        p = self._provider(fetch, candidates)
        df, meta = p.gwl_and_meta(None, 13.5, 76.0, ANCHOR)
        self.assertIsNotNone(df, "should fall through to the fresh well, not go None")
        self.assertLessEqual(meta["data_age_days"], 3, "must have used the FRESH well")
        self.assertAlmostEqual(float(df["gwl_value"].iloc[-1]), 4.5, places=3)

    def test_all_stale_returns_none_for_analog(self):
        candidates = [(1.9, "manual", "StaleA"), (10.9, "tele", "StaleB")]
        fetch = {"StaleA": self._recs("01-01-2024 12:00", 7.0),
                 "StaleB": self._recs("10-02-2024 12:00", 4.5)}     # both > 100d stale
        p = self._provider(fetch, candidates)
        df, meta = p.gwl_and_meta(None, 13.5, 76.0, ANCHOR)
        self.assertIsNone(df, "no fresh well anywhere -> None -> advisory seasonal analog")


if __name__ == "__main__":
    unittest.main(verbosity=2)
