"""phraser.py — turn the structured advisory into a farmer message (bullet points).

v0 = deterministic TEMPLATE (this file; always available — the fallback + the validation
oracle). v1 = a small language model (Gemma, Phase B) that ONLY rephrases the same
structured DECISION fields, added behind the same phrase() interface with a template
fallback. The rule engine is the sole decider; the phraser NEVER computes or flips the sign.

The fixed CAVEATS (normal != sustainable; groundwater-lens) are appended DETERMINISTICALLY to
both paths — the SLM never rephrases them, so they can't be diluted or hallucinated.

SIGN GUARD: change_m > 0 = water FELL (worse); < 0 = recharge (better). Any message must
preserve the direction implied by the structured advisory. The SLM (Phase B) is handed this
note and its output is validated against the structured fields before use.
"""
from __future__ import annotations

SIGN_NOTE = ("change_m > 0 = water table FELL (worse); < 0 = recharge (better). Never invert this.")

# keyed on adv["last_year"] (plain words set in advisory._LAST_YEAR): "wetter" / "normal" / "drier"
_PAST_YEAR = {"wetter": "wetter than normal", "normal": "about normal", "drier": "a deficit"}

# Fixed closing caveats — appended to every OK message (template or SLM), never rephrased.
_CAUTION_LINE = ("Caution: 'normal' means typical of recent years — if your area's water table "
                 "is falling year on year, even a normal season keeps depleting it.")
_LENS_LINE = ("Note: groundwater signal only — crop types are guidance; pick the specific crop "
              "locally and also weigh rainfall, price and soil.")


def _closing_bullets():
    return [f"• {_CAUTION_LINE}", f"• {_LENS_LINE}"]


# SOFT confidence sentences (farmer message). The internal HIGH/MEDIUM/LOW is a spatial-consensus
# proxy, NOT a real prediction interval, so the message never says "HIGH confidence" (an over-claim
# that reads as certainty / can alarm). One gentle line per level; ALWAYS deterministic — the SLM is
# told not to phrase confidence, so this line can't be softened away or over-stated by the model.
_CONF_SENTENCE = {
    "HIGH":   "This is a reasonably well-supported read for your area.",
    "MEDIUM": "Treat this as a rough guide — the local signals are mixed.",
    "LOW":    "Local data is thin here — treat this as a weak hint; check with your KVK.",
}


def _confidence_bullet(level):
    return "• " + _CONF_SENTENCE.get(level, _CONF_SENTENCE["LOW"])


# Two-axis outlook lines (calibrated & deterministic — the SLM never rephrases these; the numbers
# must survive). Read from adv["outlook"] (built in advisory.py):
#   water_trend = {direction: rising|falling, reliability_pct}   ← vs ZERO (gain/lose), strongest signal
#   vs_normal   = {level: above|below|normal, clarity, reliability_pct}   ← vs the NORMAL (better/worse)
def _water_trend_bullet(adv):
    from advisory.reliability import pct_in_words
    wt = (adv.get("outlook") or {}).get("water_trend")
    if not wt:
        return None
    rising = wt.get("direction") == "rising"
    verb = "rise (recharge)" if rising else "fall (depletion)"
    gain = "gain" if rising else "lose"
    w = pct_in_words(wt.get("reliability_pct"))
    tail = f" ({w} for a move this size)" if w else ""
    return f"• Water this season: the table looks set to {verb} — you should {gain} water{tail}."


_VS_NORMAL_CORE = {
    "above": "a bit better than a normal year (more water than usual)",
    "below": "worse than a normal year (less water than usual)",
    "normal": "about a typical year — no strong signal either way",
}


def _vs_normal_bullet(adv):
    from advisory.reliability import pct_in_words
    vn = (adv.get("outlook") or {}).get("vs_normal")
    if not vn:
        return None
    core = _VS_NORMAL_CORE.get(vn.get("level"))
    if not core:
        return None
    clar, w = vn.get("clarity"), pct_in_words(vn.get("reliability_pct"))
    tail = f" — a {clar} call ({w})" if clar in ("clear", "moderate") and w else ""
    return f"• Versus a normal year: {core}{tail}."


# WHY we couldn't advise — one plain lead sentence per machine reason_code (advisory.py:_decline).
# Kept in sync with the reason_code enum there; a missing/unknown code falls back to the generic line.
_INSUFFICIENT_LEAD = {
    "out_of_network": "You're outside our monitored well network here — the nearest well is too far to base a local groundwater call on.",
    "sparse_history": "There are wells nearby, but they don't have enough years of records to work out what's normal for this season.",
    "no_recent_reading": "There are wells nearby, but none has a recent reading and there isn't enough past history to estimate from instead.",
    "no_seasonal_history": "There are wells nearby, but not enough seasonal history to estimate a groundwater outlook here.",
    "bad_request": "That request can't be answered as-is — check the date is today or earlier and the location is valid.",
    "forecast_failed": "We couldn't produce a groundwater forecast for this point right now.",
}
_INSUFFICIENT_ACTION = ("• For now, rely on your own well/borewell reading and local KVK advice; a district "
                        "groundwater bulletin (CGWB) can also help. Try again once nearby wells report.")


def _insufficient_message(adv: dict) -> str:
    """Actionable decline message: a plain WHY (from reason_code) + the concrete numbers
    (from details) + what the farmer can do instead."""
    rc = adv.get("reason_code")
    det = adv.get("details") or {}
    lead = _INSUFFICIENT_LEAD.get(rc)
    if lead is None:      # unknown/absent code -> fall back to the free-text reason
        reason = adv.get("reason", "")
        lead = "Not enough nearby well data for a reliable call here" + (f" ({reason})" if reason else "")
    bits = []
    nk = det.get("nearest_km")
    if isinstance(nk, (int, float)):
        bits.append(f"nearest monitored well ~{nk:.0f} km away")
    uy = det.get("usable_years")
    if isinstance(uy, int):
        bits.append(f"only {uy} year{'' if uy == 1 else 's'} of usable seasonal history")
    detail = f" ({'; '.join(bits)})" if bits else ""
    return "\n".join([f"• {lead}{detail}.", _INSUFFICIENT_ACTION])


def phrase_v0(adv: dict) -> str:
    """Deterministic template — the message as '• ' bullet points, one per line."""
    if adv.get("status") == "insufficient_data":
        return _insufficient_message(adv)

    conf = (adv.get("confidence") or {}).get("level")   # _confidence_bullet handles None -> LOW line
    lean = adv.get("crop_lean")
    g = adv.get("crop_guidance") or {}
    need = g.get("water_need", "")
    groups = ", ".join(str(c) for c in g.get("type_examples", [])[:3])
    grp_txt = f" (e.g. {groups})" if groups else ""
    sow = adv.get("sowing_window")
    sow_txt = f" ({sow} window)" if sow else ""
    past = _PAST_YEAR.get(adv.get("last_year"), "")
    long_ok = (adv.get("long_crop") or {}).get("allowed")
    dur = ("A longer-duration crop is workable — keep a margin (not a new perennial on one forecast)." if long_ok
           else "Prefer a short-duration crop; don't start a new perennial on a single forecast.")

    bullets = []
    for line in (_water_trend_bullet(adv), _vs_normal_bullet(adv)):     # (1) gain/lose  (2) vs a normal year
        if line:
            bullets.append(line)
    bullets.append("• " + (f"Suggestion{sow_txt}: {lean} — a {need}-water crop{grp_txt}." if lean
                           else f"Suggestion{sow_txt}: a {need}-water crop{grp_txt}."))
    bullets.append(f"• Duration: {dur}")
    bullets.append(_confidence_bullet(conf))
    if past:
        bullets.append(f"• Context: last year was {past}.")
    return "\n".join(bullets + _closing_bullets())


def phrase(adv: dict, slm=None) -> str:
    """Phase A deterministic template — authoritative for the two-axis outlook shape.

    NEVER raises: a phrasing bug must not sink an otherwise-good advisory. If the template build
    fails for ANY reason we log it and return a safe line, so the caller still gets the structured
    numbers/outlook fields instead of a hard error.

    NOTE: the Phase-B SLM (Gemma) rephrasing is NOT yet re-integrated for the new outlook shape —
    it must leave the calibrated water_trend / vs_normal numbers untouched (rephrase only the crop
    suggestion). Until that's wired, we render the deterministic template. TODO(slm-two-axis): when
    re-adding it, wrap `slm.rephrase(...)` in the same try/except -> phrase_v0 fallback.
    """
    try:
        return phrase_v0(adv)
    except Exception:   # noqa: BLE001 — degrade to a safe message; phrasing must never crash the advisory
        import traceback as _tb
        _tb.print_exc()
        if adv.get("status") == "insufficient_data":
            return ("• Not enough nearby well data for a reliable call here.\n"
                    "• Treat any number as indicative and check locally (KVK).")
        return ("• A groundwater outlook was computed — see the numbers/outlook fields.\n"
                "• The plain-language summary couldn't be built here; treat it as indicative and "
                "check locally (KVK).")
