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

_PAST_YEAR = {"above": "wetter than normal", "normal": "about normal", "below": "a deficit"}

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


def phrase_v0(adv: dict) -> str:
    """Deterministic template — the message as '• ' bullet points, one per line."""
    if adv.get("status") == "insufficient_data":
        reason = adv.get("reason", "")
        return "\n".join([
            "• Not enough nearby well data for a reliable call here"
            + (f" ({reason})" if reason else "") + ".",
            "• Treat any number as indicative and check locally (KVK).",
        ])

    r = adv["regime"]
    conf = adv["confidence"]["level"]
    lean = adv["crop_lean"]
    g = adv.get("crop_guidance") or {}
    need = g.get("water_need", "")
    groups = ", ".join(str(c) for c in g.get("type_examples", [])[:3])
    outlook = adv.get("outlook_phrase", "")
    past = _PAST_YEAR.get(r.get("b"), "")
    sow = adv.get("sowing_window")
    sow_txt = f" ({sow} sowing window)" if sow else ""
    grp_txt = f" (e.g. {groups})" if groups else ""
    long_ok = (adv.get("long_crop") or {}).get("allowed")
    dur = ("A longer-duration crop is workable — keep a margin (not a new perennial on one forecast)." if long_ok
           else "Prefer a short-duration crop; don't start a new perennial on a single forecast.")
    lines = [
        f"Outlook{sow_txt}: {outlook}.",
        f"Past year: {past}." if past else None,
        f"Suggestion: {lean} — a {need}-water crop{grp_txt}.",
        f"Duration: {dur}",
        _CONF_SENTENCE.get(conf, _CONF_SENTENCE["LOW"]),   # soft, never "Confidence: HIGH"
    ]
    bullets = ["• " + ln for ln in lines if ln]
    return "\n".join(bullets + _closing_bullets())


def phrase(adv: dict, slm=None) -> str:
    """Phase A: template. Phase B: SLM for the DECISION bullets + fixed closing caveats,
    with a full-template fallback whenever the SLM errors or fails validation."""
    if slm is None or adv.get("status") != "ok":
        return phrase_v0(adv)
    try:
        msg = slm.rephrase(adv, sign_note=SIGN_NOTE)      # Phase B interface (Gemma)
    except Exception:
        msg = None
    if not msg:
        return phrase_v0(adv)
    # append the SOFT confidence line + fixed caveats deterministically (the SLM phrases neither)
    conf = (adv.get("confidence") or {}).get("level")
    return "\n".join([msg, _confidence_bullet(conf)] + _closing_bullets())
