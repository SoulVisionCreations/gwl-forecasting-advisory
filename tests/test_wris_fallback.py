"""Unit tests for the CSV-primary / WRIS-backup normals source (no network).

Covers the two correctness-critical bits: the datatype filter (drop AMSL elevation, keep
depth-to-water) + abs()/per-day cleaning, and the per-well fallback wiring (CSV present ->
untouched ; CSV missing -> filled from WRIS + tagged from_fallback). Plus the confidence cap.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from inference.acquire.csv_wris_fallback_source import (
    CsvWithWrisFallbackSource, parse_wris_records, _is_depth,
)
from inference.types import AcquiredStation, Neighbour

DC = SimpleNamespace(max_gwl=100.0, min_gwl=0.0)   # abs + cap at 100 m (parity stand-in)

# mixed WRIS payload: HGZ/GGZ depth-to-water (keep) + an AMSL elevation record (drop)
RECS = [
    {"datatypeDescription": "MANUAL-Water Level", "dataTime": "2013-01-05T06:00:00", "dataValue": 5.7},
    {"datatypeDescription": "GPRS-Water Level",   "dataTime": "2013-04-05T06:00:00", "dataValue": -6.0},
    {"datatypeDescription": "Manual Water Level - AMSL", "dataTime": "2013-04-05T06:00:00", "dataValue": -534.23},
    {"datatypeDescription": "MANUAL-Water Level", "dataTime": "2013-07-05T00:00:00", "dataValue": 4.0},
    {"datatypeDescription": "MANUAL-Water Level", "dataTime": "2013-07-05T12:00:00", "dataValue": 6.0},
    {"datatypeDescription": "MANUAL-Water Level", "dataTime": "2013-10-05T06:00:00", "dataValue": 3.0},
]


def test_is_depth_filters_amsl():
    assert _is_depth("MANUAL-Water Level") and _is_depth("GPRS-Water Level")
    assert not _is_depth("Manual Water Level - AMSL")   # elevation, not depth
    assert not _is_depth(None) and not _is_depth("Rainfall")


def test_parse_drops_amsl_absolutes_and_daymeans():
    s = parse_wris_records(RECS, "TEST", DC)
    assert s is not None and list(s.columns) == ["gwl_value"]
    vals = {str(d.date()): v for d, v in s["gwl_value"].items()}
    assert vals["2013-01-05"] == 5.7
    assert vals["2013-04-05"] == 6.0    # abs(-6.0); the AMSL -534 on the SAME day is dropped, NOT averaged
    assert vals["2013-07-05"] == 5.0    # per-day mean of 4 and 6
    assert vals["2013-10-05"] == 3.0
    assert len(s) == 4                  # four distinct days (AMSL record excluded)
    assert s["gwl_value"].max() <= 100  # -534 never leaks in
    assert (s["gwl_value"] > 0).all()   # abs() sign-normalized


def test_parse_empty_when_only_amsl():
    only_amsl = [{"datatypeDescription": "Manual Water Level - AMSL", "dataTime": "2013-01-05", "dataValue": -534.0}]
    assert parse_wris_records(only_amsl, "T", DC) is None
    assert parse_wris_records([], "T", DC) is None


class _FakePrimary:
    """Stand-in LocalCsvSource: returns whatever gwl is registered per code (None = CSV gap)."""
    def __init__(self, gwl_by_code):
        self.g = gwl_by_code

    def fetch(self, neighbours, current_date):
        return [AcquiredStation(neighbour=n, gwl=self.g.get(n.station_code),
                                dynamic=None, static=None, composite_path=None) for n in neighbours]


def _nb(code):
    return Neighbour(station_code=code, safe_id=code, lat=0.0, lon=0.0, distance_km=1.0)


def test_fallback_only_fills_the_gap_and_tags():
    import pandas as pd
    idx = pd.to_datetime(["2013-01-01", "2013-04-01", "2013-07-01", "2013-10-01", "2014-01-01"])
    csv_df = pd.DataFrame({"gwl_value": [3.0, 4.0, 5.0, 4.5, 3.5]}, index=idx)   # A is in the CSV
    src = CsvWithWrisFallbackSource(_FakePrimary({"A": csv_df, "B": None}), DC, verify=False)
    src._post = lambda code, start, end: RECS if code == "B" else None            # no network

    out = {a.neighbour.station_code: a for a in src.fetch([_nb("A"), _nb("B")], datetime(2024, 2, 15))}
    assert out["A"].gwl is csv_df and out["A"].from_fallback is False              # CSV well untouched
    assert out["B"].gwl is not None and out["B"].from_fallback is True             # gap filled from WRIS
    assert len(out["B"].gwl) == 4


def test_confidence_cap_helper():
    from advisory import consensus as cons
    assert cons.cap_level("HIGH", "MEDIUM") == "MEDIUM"    # backup normals can't read HIGH
    assert cons.cap_level("LOW", "MEDIUM") == "LOW"        # already weaker -> unchanged
    assert cons.cap_level("MEDIUM", "MEDIUM") == "MEDIUM"
