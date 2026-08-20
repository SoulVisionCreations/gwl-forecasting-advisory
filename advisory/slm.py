"""slm.py — Ollama-backed Gemma phraser (Phase B). It ONLY rephrases; it never decides.

Calls a local Ollama endpoint (gemma3n:e2b by default) to turn the structured advisory into
a natural farmer message. HARD-constrained by a system prompt (sign-guard + "phrase only")
and VALIDATED against the structured fields; on any error / empty output / failed validation
it returns None, so phraser.phrase() falls back to the deterministic v0 template. The rule
engine stays the sole decider.

Runs on big-10 (GPU). The model is pulled on `big` (registry) onto the shared FS; big-10 only
serves it. See feedback_no_code_on_big.
"""
from __future__ import annotations

import json
import os
import re

# Fidelity guards for _valid(): the rephrase must not carry the OPPOSITE direction/duration of the
# decision. POS/NEG_DIR are direction words (not the neutral noun 'recharge', which appears in both a
# strong and a weak monsoon). Duration patterns match the ADVICE ('short/longer duration', 'short
# crop'); crop-type GROUP names (e.g. 'short-duration maize') are STRIPPED in _valid before the check
# so they can't false-match. On a mismatch _valid returns False -> phraser falls back to the template.
_POS_DIR = re.compile(r"\bbetter\b|surplus|abundan|plenty|holds? up better|more water|improv|"
                      r"stronger|above normal|hold up well", re.I)
_NEG_DIR = re.compile(r"deplet|falling faster|fall faster|weaker|worse|deficit|draw down|less water|"
                      r"running low|drying|scarce|declin|faster than usual", re.I)
_SHORT_ADVICE = re.compile(r"short(?:er)?[- ]duration|\bshort(?:er)? crop\b|prefer a short", re.I)
_LONG_ADVICE = re.compile(r"long(?:er)?[- ]duration|\blong(?:er)? crop\b", re.I)
# Vague-stability wording UNDER-states a real anomaly (above/below normal). Rejected on an anomaly
# decision (kept for a neutral/'about typical' one) -> falls back to the faithful template.
_VAGUE = re.compile(r"\bstable\b|\bsteady\b|unchanged|remain the same|stay the same|remain constant|"
                    r"\bno change\b|holds? steady", re.I)

_SYSTEM = """You are a groundwater crop advisor. You ONLY rephrase a pre-computed decision into \
a short, simple message for an Indian smallholder farmer. You must NOT change any decision, crop \
type, direction, or confidence.
SIGN RULE: {sign_note}
Rules:
- Use the water need, the crop-type groups, the crop lean, and the confidence EXACTLY as given. Do \
NOT name a specific crop variety.
- Do NOT include ANY digits or numbers (no months, no metres, no counts) — the decision has none.
- Format as EXACTLY 3 short bullet points, each on its OWN line, each starting with "- ". No \
paragraph and no closing note (fixed notes are added separately). Keep each bullet a clean, complete phrase.
- The points, in order: (1) the water OUTLOOK; (2) the crop guidance = the water need plus 1-2 of \
the given crop-type groups; (3) the duration advice.
- Do NOT mention confidence, certainty, reliability, or how sure the advice is — a separate line covers that.
- OUTLOOK must state the direction FAITHFULLY, matching the given outlook phrase: if water holds up \
BETTER than normal say 'better than usual' (do NOT say 'stable', 'steady', or 'the same'); if it \
DEPLETES / falls FASTER say so; if it is about typical say 'about typical'. Do NOT soften or reverse it.
- DURATION: if a long crop is allowed, say a longer crop is workable but keep a margin; if NOT allowed, \
prefer a short-duration crop. Never say 'continue with the current duration'. Never suggest starting a \
NEW perennial (e.g. sugarcane) on a single forecast.
- Do NOT flip the sign (recharge vs fall)."""


class OllamaPhraser:
    """Phase-B phraser. Pass an instance as AdvisoryEngine(slm=...); phraser falls back to
    the v0 template whenever this returns None."""

    def __init__(self, host=None, model="gemma3n:e2b", timeout=60, keep_alive="30m",
                 temperature=0.0, num_predict=160):
        host = host or os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
        self.host = host if host.startswith("http") else "http://" + host
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.num_predict = num_predict

    def rephrase(self, adv, sign_note=""):
        # decline / non-ok stay exact (deterministic) -> let the template handle them
        if adv.get("status") != "ok":
            return None
        import requests

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM.format(sign_note=sign_note)},
                {"role": "user", "content": self._user_payload(adv)},
            ],
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "num_predict": self.num_predict},
        }
        r = requests.post(self.host + "/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        msg = ((r.json().get("message") or {}).get("content") or "")
        msg = self._format_bullets(msg)
        return msg if self._valid(msg, adv) else None

    # Gemma often appends its OWN generic disclaimer ("consult a local expert...", "for informational
    # purposes...") -- we already append fixed caveats, so drop those duplicate lines.
    _DROP_LINE = re.compile(r"consult|professional|informational|local expert|licensed|as needed|"
                            r"replace|disclaimer|general guidance|confidence|certaint|reliab", re.I)

    @classmethod
    def _format_bullets(cls, msg):
        """Normalise the model output to '• ' bullet lines (KEEP newlines; collapse only intra-line
        whitespace; strip a leading bullet/number marker; drop Gemma's self-added disclaimer lines)."""
        msg = msg.replace("▁", " ")            # gemma sentencepiece '_' leak -> space
        out = []
        for ln in msg.splitlines():
            ln = re.sub(r"[ \t]+", " ", ln).strip()
            ln = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", ln)   # drop one leading bullet / list number
            ln = ln.rstrip("*").strip()                          # drop a trailing '*' artifact
            if ln and not cls._DROP_LINE.search(ln):
                out.append("• " + ln)
        return "\n".join(out)

    def _user_payload(self, adv):
        g = adv.get("crop_guidance") or {}
        return json.dumps({
            "water_outlook": adv.get("outlook_phrase"),
            "year_just_ended": {"wetter": "wetter than normal", "normal": "about normal",
                                "drier": "a deficit"}.get(adv.get("last_year"), ""),
            "crop_lean": adv.get("crop_lean"),
            "water_need": g.get("water_need"),
            "crop_type_groups": g.get("type_examples"),
            "long_crop_allowed": (adv.get("long_crop") or {}).get("allowed"),
        }, ensure_ascii=False)

    def _valid(self, msg, adv):
        """Trust the message ONLY if it carries the decision — it must name at least one of the
        given crop-type GROUPS and carry NO digits (our advisory is number-free, so any digit is a
        hallucination). Else fall back to the v0 template. (Water-need words like 'high'/'low' are
        NOT accepted alone — they collide with 'high confidence' etc.)"""
        if not msg or len(msg) < 20:
            return False
        if re.search(r"\d", msg):          # number-free by design → a digit = hallucination
            return False
        low = msg.lower()
        g = adv.get("crop_guidance") or {}
        tokens = set()
        for c in g.get("type_examples", []):
            for t in re.split(r"[^a-z]+", str(c).lower()):
                if len(t) >= 4:
                    tokens.add(t)
        if not (tokens and any(t in low for t in tokens)):     # must name a given crop group
            return False
        # DIRECTION fidelity: reject the OPPOSITE of the decision's direction (a softening that keeps
        # the right side is fine; a reversal is not). Direction is read off the outlook phrase.
        outlook = (adv.get("outlook_phrase") or "").lower()
        pos = any(w in outlook for w in ("better", "above", "hold up", "holds up", "running above"))
        neg = any(w in outlook for w in ("deplet", "weaker", "draw down", "faster", "below"))
        if pos and _NEG_DIR.search(low):
            return False
        if neg and _POS_DIR.search(low):
            return False
        if (pos or neg) and _VAGUE.search(low):     # vague 'stable' understates a real anomaly
            return False
        # DURATION fidelity: reject a SHORTER-crop message when a long crop is OK (and a LONGER-crop one
        # when it isn't). Strip the crop-type GROUP names first (normalise hyphens->spaces) so a crop
        # NAME like "short-duration maize" is not mistaken for "short-duration" DURATION advice.
        low_da = low.replace("-", " ")
        for c in g.get("type_examples", []):
            low_da = low_da.replace(str(c).lower().replace("-", " "), " ")
        long_ok = bool((adv.get("long_crop") or {}).get("allowed"))
        if long_ok and _SHORT_ADVICE.search(low_da):
            return False
        if (not long_ok) and _LONG_ADVICE.search(low_da):
            return False
        return True
