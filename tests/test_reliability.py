"""Tests for advisory/reliability.py (clarity level + calibrated reliability) and its phraser wiring."""
from advisory import reliability as R
from advisory import phraser as P


# ---- clarity_of ---------------------------------------------------------------

def test_normal_regime_is_typical_no_pct():
    c = R.clarity_of(d=0.1, band=0.4, regime="normal")
    assert c == {"level": "typical", "reliability_pct": None}


def test_clear_call_above():
    # |d|/band = 1.0/0.3 = 3.33 >= CLEAR_RATIO -> clear. Assert against the constant (survives recalibration).
    c = R.clarity_of(d=-1.0, band=0.3, regime="above")
    assert c["level"] == "clear" and c["reliability_pct"] == R._RELIABILITY["3m"]["clear"]


def test_moderate_call_below():
    # |d|/band = 0.6/0.4 = 1.5 -> moderate
    c = R.clarity_of(d=0.6, band=0.4, regime="below")
    assert c["level"] == "moderate" and c["reliability_pct"] == R._RELIABILITY["3m"]["moderate"]


def test_boundary_ratio_exactly_2_is_clear():
    c = R.clarity_of(d=0.8, band=0.4, regime="above")   # ratio 2.0
    assert c["level"] == "clear"


def test_missing_inputs_return_none():
    assert R.clarity_of(None, 0.3, "above") is None
    assert R.clarity_of(1.0, None, "above") is None
    assert R.clarity_of(1.0, 0.0, "above") is None      # band 0 guarded
    assert R.clarity_of(1.0, 0.3, None) is None


def test_6m_horizon_lookup():
    c = R.clarity_of(d=-1.0, band=0.3, regime="above", horizon="6m")
    assert c["level"] == "clear" and c["reliability_pct"] == R._RELIABILITY["6m"]["clear"]


# ---- pct_in_words -------------------------------------------------------------

def test_pct_in_words():
    assert R.pct_in_words(74) == "about 3 in 4"     # 74 = shipped "clear"
    assert R.pct_in_words(71) == "about 7 in 10"    # 71 = shipped "moderate"
    assert R.pct_in_words(72) == "about 7 in 10"    # boundary: <73 -> N-in-10
    assert R.pct_in_words(None) is None


# ---- direction_of (absolute axis) --------------------------------------------

def test_direction_of_rising_big_move():
    d = R.direction_of(-3.0, "3m")     # negative = rising; |3.0| in the 2-5m bucket
    assert d["direction"] == "rising" and d["reliability_pct"] == R.DIRECTION_ACC_BY_PRED_MOVE["3m"]["2-5m"]


def test_direction_of_falling_small_move():
    d = R.direction_of(0.3, "3m")      # positive = falling; |0.3| < 0.5
    assert d["direction"] == "falling" and d["reliability_pct"] == R.DIRECTION_ACC_BY_PRED_MOVE["3m"]["<0.5m"]


def test_direction_of_none():
    assert R.direction_of(None) is None


def test_pct_in_words_direction():
    assert R.pct_in_words(82) == "about 8 in 10"
    assert R.pct_in_words(88) == "about 9 in 10"


# ---- phraser wiring (two-axis outlook) ---------------------------------------

def _adv(outlook, **extra):
    base = {"status": "ok", "confidence": {"level": "HIGH"}, "last_year": "normal",
            "crop_guidance": {"water_need": "high", "type_examples": ["paddy/rice"]},
            "long_crop": {"allowed": False}, "outlook": outlook}
    base.update(extra)
    return base


def test_water_trend_bullet_rising():
    b = P._water_trend_bullet(_adv({"water_trend": {"direction": "rising", "reliability_pct": 82}}))
    assert b and "rise (recharge)" in b and "gain water" in b and "about 8 in 10" in b


def test_water_trend_bullet_falling():
    b = P._water_trend_bullet(_adv({"water_trend": {"direction": "falling", "reliability_pct": 66}}))
    assert b and "fall (depletion)" in b and "lose water" in b


def test_vs_normal_bullet_above_moderate():
    b = P._vs_normal_bullet(_adv({"vs_normal": {"level": "above", "clarity": "moderate", "reliability_pct": 71}}))
    assert b and "better than a normal year" in b and "moderate call" in b and "about 7 in 10" in b


def test_vs_normal_bullet_typical_no_number():
    b = P._vs_normal_bullet(_adv({"vs_normal": {"level": "normal", "clarity": "typical", "reliability_pct": None}}))
    assert b and "typical year" in b and "in 10" not in b and "in 4" not in b


def test_outlook_bullets_absent():
    assert P._water_trend_bullet({"outlook": {}}) is None
    assert P._vs_normal_bullet({}) is None


def test_phrase_v0_two_axis():
    adv = _adv({"water_trend": {"direction": "rising", "reliability_pct": 82},
                "vs_normal": {"level": "above", "clarity": "moderate", "reliability_pct": 71}},
               crop_lean="water-heavy; can expand", sowing_window="kharif")
    msg = P.phrase_v0(adv)
    assert "Water this season" in msg and "rise (recharge)" in msg and "about 8 in 10" in msg
    assert "Versus a normal year" in msg and "moderate call" in msg
    assert "Suggestion" in msg and "Context: last year was about normal" in msg
    assert "expand confidently" not in msg   # over-claim removed
