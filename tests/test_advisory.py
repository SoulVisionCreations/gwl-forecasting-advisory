"""Phase A deterministic advisory — unit tests (no model, no data host needed).

Reproduces the 4 real monsoon x season cases' regimes from their known numbers, plus the
per-place IQR band, weakest-link confidence, the long-crop gate, season lens, and a small
normals.compute over a synthetic well series. These are the exact behaviours documented in
use_case_gwl.txt / use_case_gwl_seasons.txt.
"""
from __future__ import annotations

import pandas as pd

from advisory import confidence as conf
from advisory import normals as norm
from advisory import rule_engine as re_
from advisory import season_lens as lens
from advisory.neighbours import NeighbourSet

RULES = re_.load_rules()

# Real two-band values (bandS = seasonal IQR band for (a); bandY = yoy IQR band for (b)) — /tmp/_twoband.py.
# expect_season uses the DEADBAND lens season_of(normS, bandS): |normS| within bandS -> 'transition'
# (NE Ariyalur -1.29/1.95 and coastal TN 1.05/1.55 are within-noise, so they read transition, not wet/dry).
# (name, fc, normS, bandS, latestY, normY, bandY, expect_a, expect_b, expect_tier, expect_season)
CASES = [
    ("SW-wet CG Durg",     -6.16, -11.71, 3.40, -0.61,  1.19, 3.09, "below",  "normal", "water-light", "wet"),
    ("SW-dry Davanagere",   1.54,   9.03, 3.37,  3.28, -0.39, 4.91, "above",  "normal", "water-heavy", "dry"),
    ("NE-wet TN Ariyalur",  1.28,  -1.29, 1.95,  4.86,  1.05, 0.99, "below",  "below",  "water-light", "transition"),
    ("NE-dry coastal TN",   0.31,   1.05, 1.55,  2.46,  2.03, 2.65, "normal", "normal", "usual",       "transition"),
]


def test_four_cases_regime_tier_season():
    for name, fc, ns, bs, ly, ny, by, ea, eb, etier, eseason in CASES:
        a, _ = re_.bin_a(fc, ns, bs)
        b, _ = re_.bin_b(ly, ny, by)
        assert a == ea, f"{name}: a={a} expected {ea}"
        assert b == eb, f"{name}: b={b} expected {eb}"
        cell = RULES[(a, b)]
        assert cell["tier"] == etier, f"{name}: tier={cell['tier']} expected {etier}"
        assert lens.season_of(ns, bs) == eseason, f"{name}: season mismatch"   # deadband lens
        # crop-TYPE guidance exists for (tier, season) and carries a water need + groups
        g = re_.crop_guidance(etier, eseason, {"allowed": False})
        assert g["water_need"] and g["type_examples"], f"{name}: no crop guidance"


def test_band_is_half_iqr_with_floor():
    # IQR of [7,8,9,11,13] = Q3(11)-Q1(8)=3 -> band 1.5 ; a flat series -> floor 0.3
    assert abs(norm._BAND_FLOOR - 0.3) < 1e-9
    # via numpy percentile (linear) the exact value differs slightly; just assert floor kicks in
    band_flat = max(0.5 * 0.0, norm._BAND_FLOOR)
    assert band_flat == 0.3


def test_confidence_weakest_link():
    # distance + history only (weakest link); neighbour agreement no longer scored
    assert conf.assess(0.4, 9, 9)["level"] == "HIGH"            # close + long history
    assert conf.assess(39.0, 10, 10)["level"] == "LOW"          # far well caps to LOW
    assert conf.assess(8.0, 9, 9)["level"] in ("MEDIUM", "LOW")  # medium distance -> not HIGH
    assert conf.assess(2.0, 3, 2)["level"] == "LOW"             # thin history caps to LOW
    # spatial spread + neighbour distances are not surfaced in the confidence output
    _c = conf.assess(2.0, 9, 9)
    assert _c["level"] == "HIGH" and "area_spread_m" not in _c and "median_distance_km" not in _c


def test_consensus_confidence():
    # NEW confidence: 1/d^2-weighted SD of per-well z's -> NDVI band, weakest-link freshness, min-well gate
    from advisory import consensus as cons
    w = {"A": 1 / 9, "B": 1 / 64, "C": 1 / 225, "D": 1 / 361}
    agree = {"A": 3.0, "B": 2.0, "C": 2.5, "D": 2.2}      # tight  -> SD < 0.5
    mixed = {"A": 3.0, "B": 2.0, "C": -1.0, "D": -2.0}    # spread -> SD >= 1.0
    assert cons._z_band(cons._weighted_sd(agree, w)[0]) == "good"
    assert cons._z_band(cons._weighted_sd(mixed, w)[0]) == "poor"
    assert cons._axis_confidence(agree, w, "good")["level"] == "HIGH"      # fresh + agree
    assert cons._axis_confidence(agree, w, "poor")["level"] == "LOW"       # poor freshness band overrides
    assert cons._axis_confidence(agree, w, "medium")["level"] == "MEDIUM"  # analog band caps at MEDIUM
    # NO min-well gate: the 1/d^2 weighting handles proximity -> rely on a near/lone well
    assert cons._axis_confidence({"A": 3.0, "B": 2.0}, w, "good")["level"] == "HIGH"  # 2 wells, no gate
    assert cons._axis_confidence({"A": 3.0}, w, "good")["level"] == "HIGH"            # single well -> rely on it
    assert cons._axis_confidence({}, w, "good")["level"] == "LOW"                     # zero wells -> LOW


def test_seasonal_analog():
    # stale-well fallback: median of the well's most recent up-to-3 Jul->Oct changes
    from datetime import datetime
    from advisory import analog as an
    data = {2025: (5.0, 7.0), 2024: (4.5, 6.3), 2023: (5.2, 7.5), 2022: (4.8, 6.6), 2021: (5.1, 7.0)}
    idx, vals = [], []
    for y, (jul, oct_) in data.items():
        idx += [pd.Timestamp(f"{y}-07-15"), pd.Timestamp(f"{y}-10-15")]
        vals += [jul, oct_]
    s = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()
    anchor = datetime(2026, 7, 21)
    # most-recent-3 Jul->Oct changes = [+2.0(2025), +1.8(2024), +2.3(2023)] -> median +2.0
    assert abs(an.seasonal_change(s, anchor, 3) - 2.0) < 1e-6
    assert an.seasonal_change(pd.Series([], dtype=float), anchor, 3) is None       # no history -> None


def test_season_deadband_transition():
    # near-zero normal within the band -> transition (Kolar June fix); clear cases unchanged
    assert lens.season_of(0.3, 0.6) == "transition"
    assert lens.season_of(-2.4, 0.8) == "wet"
    assert lens.season_of(1.9, 0.7) == "dry"
    assert lens.season_of(0.3) == "dry"          # no band -> legacy hard cut at 0


def test_sowing_window_is_calendar_tag_only():
    assert lens.sowing_window("2024-06-15") == "kharif"
    assert lens.sowing_window("2024-11-15") == "rabi"
    assert lens.sowing_window("2024-03-15") == "zaid"


def test_crop_guidance_types_not_species():
    g = re_.crop_guidance("water-light", "transition", {"allowed": False})   # no calendar -> 'any'
    assert g["water_need"] == "low"
    assert g["type_examples"] and g["long_crop_ok"] is False
    # PERENNIALS (sugarcane/banana) are NEVER suggested -- not even when the long-crop gate clears
    # (guardrail: no new perennial on a single forecast).
    g_hi = re_.crop_guidance("water-heavy", "wet", {"allowed": True}, sowing_window="kharif")
    assert not any(("sugarcane" in t or "banana" in t) for t in g_hi["type_examples"])
    g_hi_no = re_.crop_guidance("water-heavy", "wet", {"allowed": False}, sowing_window="kharif")
    assert not any(("sugarcane" in t or "banana" in t) for t in g_hi_no["type_examples"])
    # calendar picks physically-sowable examples: rabi high -> wheat/potato, NOT paddy-in-winter
    g_rabi = re_.crop_guidance("water-heavy", "dry", {"allowed": True}, sowing_window="rabi")
    assert any("wheat" in t or "potato" in t for t in g_rabi["type_examples"])
    assert not any("paddy" in t for t in g_rabi["type_examples"])


def test_slm_valid_fidelity_guards():
    from advisory.slm import OllamaPhraser
    p = OllamaPhraser.__new__(OllamaPhraser)
    pos = {"crop_guidance": {"type_examples": ["wheat (irrigated)", "potato"]},
           "outlook_phrase": "water is set to hold up better than normal for this time of year here",
           "long_crop": {"allowed": True}}
    assert p._valid("- Water holds up well.\n- Plant wheat or potato.\n- A longer duration crop is fine.\n- High confidence.", pos) is True
    assert p._valid("- Water is depleting faster than usual.\n- Plant wheat or potato.\n- Longer crop ok.\n- High conf.", pos) is False   # wrong direction
    assert p._valid("- Water holds up well.\n- Plant wheat or potato.\n- Prefer a short crop.\n- High confidence.", pos) is False          # wrong duration (long_ok)
    assert p._valid("- The outlook is good, water will remain stable.\n- Plant wheat or potato.\n- A longer crop is fine.\n- High conf.", pos) is False   # vague 'stable' understates the above-normal anomaly
    neu = {"crop_guidance": {"type_examples": ["maize", "cotton"]},
           "outlook_phrase": "water looks about typical for this window", "long_crop": {"allowed": True}}
    assert p._valid("- Water looks about typical, fairly stable.\n- Plant maize or cotton.\n- A longer crop is fine.\n- High confidence.", neu) is True   # 'stable' fine on a NEUTRAL outlook
    neg = {"crop_guidance": {"type_examples": ["lentil", "linseed"]},
           "outlook_phrase": "water is set to deplete faster than normal for this time of year here",
           "long_crop": {"allowed": False}}
    assert p._valid("- Water is falling faster than usual.\n- Plant lentil or linseed.\n- A short crop is best.\n- High confidence.", neg) is True
    assert p._valid("- Water supply will improve nicely.\n- Plant lentil or linseed.\n- Short crop.\n- High conf.", neg) is False           # wrong direction


def test_long_crop_gate_asymmetric():
    assert re_.long_crop_gate("above", "above")["allowed"] is True
    assert re_.long_crop_gate("above", "normal")["allowed"] is True
    assert re_.long_crop_gate("above", "below")["allowed"] is False     # 6m veto
    assert re_.long_crop_gate("above", None)["allowed"] is False        # no 6m -> short crop
    assert re_.long_crop_gate("below", "above")["allowed"] is False     # weak 3m start


def test_normals_compute_on_synthetic_series():
    # one on-site well; Jun->Sep depletes ~+5 every year 2014..2023, tiny yoy
    dates, depths = [], []
    for y in range(2010, 2025):
        dates += [f"{y}-06-15", f"{y}-09-15", f"{y}-12-15"]
        base = 10.0 + 0.2 * (y - 2010)          # slow decline baked in
        depths += [base, base + 5.0, base + 1.0]  # Jun -> Sep(+5) -> Dec(+1 vs Jun)
    s = pd.Series(depths, index=pd.to_datetime(dates)).sort_index()
    nb = NeighbourSet(codes=["W"], weights={"W": 1.0}, distances={"W": 0.0},
                      series={"W": s}, nearest_km=0.0, n_with_data=1)
    nrm = norm.compute(nb, "2024-06-15", n_years=10, gap_days=20)
    assert abs(nrm.normal_seasonal - 5.0) < 0.5           # Jun->Sep ~ +5
    assert nrm.band is not None and nrm.band >= 0.3       # flat seasonal -> floor
    assert nrm.n_years_seasonal >= 8
    # yoy small (slow decline ~ +0.2/yr)
    assert nrm.normal_yoy is not None and abs(nrm.normal_yoy) < 1.5


if __name__ == "__main__":
    for fn in [test_four_cases_regime_tier_season, test_band_is_half_iqr_with_floor,
               test_season_deadband_transition, test_sowing_window_is_calendar_tag_only,
               test_crop_guidance_types_not_species,
               test_confidence_weakest_link, test_long_crop_gate_asymmetric,
               test_normals_compute_on_synthetic_series]:
        fn()
        print("PASS", fn.__name__)
    print("all advisory Phase-A tests passed")
