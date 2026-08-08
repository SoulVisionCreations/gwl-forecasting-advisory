"""rule_engine.py — facts -> matched cell + gates (deterministic; the ONLY decider).

Bins the two change-vs-change comparisons against the LOCAL band, matches the 3x3
regime -> (net_read, crop_lean, tier) from rules/gwl_advisory_rules.csv, and applies
the gates. Instead of naming specific crops (region/soil/market dependent — outside a
groundwater lens), it emits crop-TYPE guidance via crop_guidance(): the water NEED (from
the tier), a duration hint (from the wet/dry/transition regime), the long-crop gate, and
a few BROAD crop GROUPS as illustration. The farmer picks the specific crop locally.

Gates:
  - long-crop : allow a long-duration crop ONLY if BOTH 3m and 6m are above/normal;
                a 6m 'below' is a FIRM veto (asymmetric: easy to say no, cautious to say yes).
  - data-poor : handled by the orchestrator (advisory.py) before calling match().
"""
from __future__ import annotations

import csv
import os

_RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")


def load_rules(path=None):
    path = path or os.path.join(_RULES_DIR, "gwl_advisory_rules.csv")
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(row["a"].strip(), row["b"].strip())] = {
                "net_read": row["net_read"].strip(),
                "crop_lean": row["crop_lean"].strip(),
                "tier": row["tier"].strip(),
            }
    return out


# --- crop-TYPE guidance ---------------------------------------------------------------------
# The TIER sets the water NEED; the long-crop gate sets the DURATION. Example crop names are picked by
# the CALENDAR TAG (sowing_window, from the anchor month) so they are physically sowable NOW — kharif
# crops in the monsoon window, rabi crops in winter — WITHOUT the calendar ever inferring wetness (that
# stays the season lens). So a DRY-rabi message names rabi crops (wheat/potato), never "paddy in winter".
# PERENNIALS (sugarcane/banana) are NEVER suggested: the guardrail is 'no new perennial on a single
# forecast' (they need a multi-year trend), and the long-crop gate is only a single forecast.
_WATER_NEED = {"water-heavy": "high", "usual": "normal",
               "usual-to-light": "moderate", "water-light": "low"}
_EXAMPLES = {
    "kharif": {
        "high":     ["paddy/rice", "irrigated vegetables", "fodder maize"],
        "normal":   ["maize", "cotton", "groundnut", "soybean"],
        "moderate": ["pigeonpea (tur)", "groundnut", "bajra", "short-duration maize"],
        "low":      ["ragi (finger millet)", "bajra", "jowar", "horse gram"],
    },
    "rabi": {
        "high":     ["wheat (irrigated)", "potato", "onion", "garlic"],
        "normal":   ["wheat", "mustard", "barley", "chickpea (irrigated)"],
        "moderate": ["chickpea (gram)", "lentil (masur)", "safflower", "coriander"],
        "low":      ["lentil / horse gram (residual)", "linseed", "rainfed pulses"],
    },
    "zaid": {
        "high":     ["summer paddy", "fodder maize", "irrigated vegetables"],
        "normal":   ["green gram (moong)", "summer groundnut", "sunflower"],
        "moderate": ["green gram (moong)", "cowpea", "sesame (til)"],
        "low":      ["cowpea", "sesame", "fodder"],
    },
    "any": {        # calendar unknown -> water-class groups only (no season-bound species)
        "high":     ["a water-intensive crop", "irrigated vegetables"],
        "normal":   ["your usual mixed crop", "groundnut"],
        "moderate": ["pulses", "oilseeds", "millets"],
        "low":      ["millets", "pulses", "oilseeds"],
    },
}
_DURATION = {
    "wet":        "a full-season crop is supportable water-wise",
    "dry":        "lean to a shorter-duration, less thirsty crop",
    "transition": "keep it flexible — a short-duration crop is safest",
}


def crop_guidance(tier, season, long_crop, sowing_window=None):
    """Deterministic crop-TYPE rule. water need (tier) + duration (long-crop gate) + example crops
    picked by the sowing CALENDAR (physically sowable now), never by wetness."""
    need = _WATER_NEED.get(tier, "normal")
    long_ok = bool((long_crop or {}).get("allowed"))
    cal = sowing_window if sowing_window in _EXAMPLES else "any"
    groups = list(_EXAMPLES[cal].get(need, _EXAMPLES["any"][need]))   # never perennials (removed above)
    # duration: the long-crop gate (3m+6m) is authoritative; the season hint applies only
    # when the gate hasn't cleared a long crop (avoids "short is safest" AND "long is workable").
    # Either way, a NEW perennial is never OK on one forecast -> say so in both branches.
    if long_ok:
        dur_txt = "a longer-duration crop is workable — keep a margin (not a new perennial on one forecast)"
    else:
        dur_txt = (_DURATION.get(season, _DURATION["transition"])
                   + "; do not start a new perennial on a single forecast")
    rule = f"Water budget is {need}; {dur_txt}. Pick the specific crop from what you grow locally."
    return {
        "water_need": need,                    # high | normal | moderate | low
        "duration_hint": "long-ok" if long_ok else "short",
        "long_crop_ok": long_ok,
        "type_examples": groups[:4],           # calendar-appropriate examples, not prescriptions
        "rule": rule,
    }


def bin_a(forecast, normal, band):
    """(a) coming window: forecast - normal. < -band = ABOVE (more water), > +band = BELOW."""
    if forecast is None or normal is None or band is None:
        return None, None
    d = round(forecast - normal, 2)
    return ("above" if d < -band else "below" if d > band else "normal"), d


def bin_b(latest, normal, band):
    """(b) year just ended: latest_yoy - normal_yoy. > +band = BELOW (deficit), < -band = ABOVE (wetter)."""
    if latest is None or normal is None or band is None:
        return None, None
    d = round(latest - normal, 2)
    return ("below" if d > band else "above" if d < -band else "normal"), d


def long_crop_gate(a3, a6):
    """Long crop allowed ONLY if both 3m and 6m are above/normal; 6m below vetoes."""
    if a3 in ("above", "normal") and a6 in ("above", "normal"):
        return {"allowed": True,
                "reason": "3m and 6m both at/above normal — workable; keep a margin (6m is the shakiest signal)"}
    if a6 == "below":
        return {"allowed": False, "reason": "6m outlook below normal — water may not last; prefer a short crop"}
    if a6 is None:
        return {"allowed": False, "reason": "no 6m outlook — cannot clear a long crop; prefer a short crop"}
    return {"allowed": False, "reason": "3m below normal — weak start for a long crop; prefer a short crop"}
