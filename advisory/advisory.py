"""advisory.py — orchestrator: (lat, lon, date) -> structured advisory + message.

REUSES the existing forecaster (TFTInferenceEngine.predict for change_m) and its
StationRegistry + numeric source (neighbour histories). Adds only the deterministic
advisory layer: normals + band + confidence + season lens + rule engine + phraser.
The 6m forecast comes from an OPTIONAL second engine (h6 run-dir); absent -> long crops
are gated off. Nothing here modifies the forecaster.
"""
from __future__ import annotations

from advisory import analog as _analog
from advisory import consensus as _consensus
from advisory import neighbours as _nb
from advisory import normals as _norm
from advisory import phraser as _phr
from advisory import rule_engine as _rules
from advisory import season_lens as _lens

_CAVEATS = [
    "'normal' is the 10-yr median; where the aquifer is declining, that baseline is already "
    "depleted — 'normal' means typical-of-a-falling-decade, not safe/sustainable",
    "groundwater lens only; crop TYPES are guidance — pick the specific crop locally (soil/season/price/KVK)",
]

# Plain-language, VALUE-AWARE key for the structured fields — added under verbose=True only (default
# responses stay pure data). Explains what THIS response's values mean, keyed by field name; a client
# reading the raw JSON gets a human glossary without the machine payload carrying prose by default.
_EX_A = {"above": "wetter than the seasonal normal (more recharge than usual)",
         "normal": "about the seasonal normal", "below": "drier than the seasonal normal (less recharge)"}
_EX_B = {"above": "wetter than its year-over-year normal", "normal": "about its year-over-year normal",
         "below": "a deficit vs its year-over-year normal"}
_EX_WN = {"high": "water is plentiful — a high-water (thirsty) crop like paddy or irrigated veg is workable",
          "normal": "typical water — grow your usual crops",
          "moderate": "water is a bit tight — lean to lighter, shorter-duration crops",
          "low": "water is scarce — stick to drought-tolerant crops (millets/pulses)"}
_EX_FR = {"good": "based on recent live well readings nearby",
          "medium": "based on the seasonal pattern (no recent live reading nearby)",
          "poor": "based on the 10-yr seasonal normal (little recent data nearby)"}
_EX_LVL = {"well-supported": "fairly trustworthy — nearby wells broadly agree",
           "mixed": "moderate — some disagreement or staleness among nearby wells",
           "thin": "low — a weak hint only; nearby wells disagree or the data is sparse/old"}


def _explain(adv: dict) -> dict:
    """Build the `_explanation` block for a status=ok advisory (verbose only)."""
    r = adv.get("regime") or {}
    nums = adv.get("numbers") or {}
    cg = adv.get("crop_guidance") or {}
    lc = adv.get("long_crop") or {}
    conf = adv.get("confidence") or {}
    out: dict = {}
    if r.get("a") in _EX_A:
        out["regime.a"] = f"{r['a']} — next 3 months look {_EX_A[r['a']]}"
    if r.get("a_6m") in _EX_A:
        out["regime.a_6m"] = f"{r['a_6m']} — next 6 months look {_EX_A[r['a_6m']]}"
    if r.get("b") in _EX_B:
        out["regime.b"] = f"{r['b']} — the past year was {_EX_B[r['b']]}"
    if isinstance(nums.get("forecast_change_m"), (int, float)):
        v = nums["forecast_change_m"]
        dirn = "rise (recharge)" if v < 0 else "fall (deplete)" if v > 0 else "hold"
        out["numbers.forecast_change_m"] = (f"{v:+.2f} m over 3 months → water table set to {dirn} "
                                            "(sign: <0 rises/recharge, >0 falls/depletion)")
    if cg.get("water_need") in _EX_WN:
        out["crop_guidance.water_need"] = (f"'{cg['water_need']}' = how much water the land can SUPPLY "
                                           f"(a budget, NOT a shortage) — {_EX_WN[cg['water_need']]}")
    if "allowed" in lc:
        out["long_crop"] = ("allowed — 3m and 6m both at/above normal, so a longer crop is workable"
                            if lc["allowed"]
                            else "not advised — " + (lc.get("reason") or "weak or short outlook"))
    # confidence = ONE clear line: how much to trust this call (well agreement + data recency), plus the
    # basis (live vs seasonal pattern vs 10-yr normal). Explicitly NOT a probability.
    lvl, fr = conf.get("level"), conf.get("freshness")
    if lvl in _EX_LVL:
        s = f"how much to trust this call — '{lvl}': {_EX_LVL[lvl]}"
        if fr in _EX_FR:
            s += f"; {_EX_FR[fr]}"
        out["confidence"] = s + " — a rough guide, not a probability"
    return out


class AdvisoryEngine:

    def __init__(self, engine, engine_6m=None, n_years=10, gap_days=45,
                 rules_path=None, slm=None, normals_source=None):
        self.engine = engine
        self.engine_6m = engine_6m
        # HYBRID SOURCES: Stage 3 (normals) needs ~10 yr of history, but the live forecast source
        # (NWDP) only has a recent lookback. So the normals read from a SEPARATE historical source
        # (a LocalCsvSource / GWL history DB). None -> fall back to the engine's own source (fine
        # when that is already local-csv). The forecast (Stage 2) always uses the engine's source.
        self.normals_source = normals_source
        self.n_years = n_years
        self.gap_days = gap_days
        self.rules = _rules.load_rules(rules_path)
        self.slm = slm

    # ------------------------------------------------------------------ advise
    def advise(self, lat=None, lon=None, date=None, station_code=None, leave_one_out=False,
               verbose=False) -> dict:
        eng = self.engine
        anchor = eng._resolve_date(date)

        # 3m forecast — reuse the forecaster verbatim. predict_with_neighbours also returns the
        # per-well predictions (preds3) in the SAME run — used by the consensus confidence layer.
        fc, preds3 = eng.predict_with_neighbours(lat=lat, lon=lon, date=date,
                                                 station_code=station_code, leave_one_out=leave_one_out)
        if fc.get("status") == "error":
            return self._decline(lat, lon, anchor,
                                 reason=(fc.get("error") or {}).get("message", "forecast failed"))
        loc = fc.get("location", {})
        qlat = loc.get("lat", lat)
        qlon = loc.get("lon", lon)
        forecast_3m = fc.get("change_m")
        forecast_date = fc.get("forecast_date")

        # optional 6m forecast (second engine) — also keep its per-well predictions (preds6)
        forecast_6m = None
        preds6 = []
        if self.engine_6m is not None:
            f6, preds6 = self.engine_6m.predict_with_neighbours(
                lat=lat, lon=lon, date=date, station_code=station_code, leave_one_out=leave_one_out)
            if f6.get("status") != "error":
                forecast_6m = f6.get("change_m")

        # neighbours + histories -> normals + band  (reuse registry + numeric source)
        exclude = station_code if (station_code and leave_one_out) else None
        nb = _nb.gather(eng.registry, self.normals_source or eng._numeric, qlat, qlon,
                        eng.cfg.k_neighbours, anchor, exclude=exclude)
        nrm = _norm.compute(nb, anchor, n_years=self.n_years, gap_days=self.gap_days)

        # insufficient-data gate: decline ONLY when there is genuinely no usable normal (no seasonal
        # or yoy history at all -> a None median) or no well within range. A THIN history (2-3 yrs, or
        # even 1) is now WORKED WITH rather than declined: the median normals hold with few points and
        # _band falls back to the interannual range (>=2 yrs) or the spatial spread (1 yr) instead of
        # None. (Dropped the old `min(n_years) < 3` decline; band is None now only when 0 yrs, already
        # caught by normal_seasonal is None.)
        if (nrm.normal_seasonal is None or nrm.latest_yoy is None or nrm.normal_yoy is None
                or nrm.band is None
                or (nb.nearest_km is not None and nb.nearest_km > 15)):
            if nb.nearest_km is not None and nb.nearest_km > 15:
                _reason = (f"no monitored well within 15 km (nearest is {nb.nearest_km:.0f} km) — "
                           f"this point is outside our well network here")            # COVERAGE gap
            else:
                _reason = (f"wells are nearby ({nb.nearest_km:.0f} km) but lack enough history to form a "
                           f"seasonal normal (usable years: {nrm.n_years_seasonal})")   # HISTORY gap
            return self._decline(qlat, qlon, anchor, forecast_date=forecast_date, reason=_reason)

        # STALE wells: swap the model forecast for the well's SEASONAL ANALOG (median recent seasonal
        # change), then re-IDW to Q. Freshness becomes METHOD-based (good=fresh / medium=analog /
        # poor=climatology). No swap when every well is fresh (byte-identical live path). See
        # docs/stale_data_seasonal_analog_DISCUSSION.txt. The per-well changes also feed the consensus.
        import re as _re
        _ages = [int(x) for w in (fc.get("warnings") or []) for x in _re.findall(r"(\d+)\s*d stale", str(w))]
        freshness_days = min(_ages) if _ages else None
        forecast_3m, fc3_by_code, freshness_band, basis = _analog.blend(
            preds3, nb, anchor, 3, nrm.normal_seasonal, forecast_3m, freshness_days, gap_days=self.gap_days)
        forecast_6m, fc6_by_code, _fb6, _b6 = _analog.blend(
            preds6, nb, anchor, 6, nrm.normal_seasonal_6m, forecast_6m, freshness_days, gap_days=self.gap_days)

        # binning + regime + gates — each comparison uses the band of its OWN change series
        band_yoy = nrm.band_yoy or nrm.band                 # yoy band; fall back to seasonal if too few yoy yrs
        band_6m = nrm.band_6m or nrm.band
        a3, d_a = _rules.bin_a(forecast_3m, nrm.normal_seasonal, nrm.band)
        b, d_b = _rules.bin_b(nrm.latest_yoy, nrm.normal_yoy, band_yoy)
        a6, _ = _rules.bin_a(forecast_6m, nrm.normal_seasonal_6m, band_6m)
        cell = self.rules.get((a3, b), {})
        tier = cell.get("tier")
        season = _lens.season_of(nrm.normal_seasonal, nrm.band)   # wet/dry/transition (deadband)
        sowing = _lens.sowing_window(anchor)                      # calendar tag only (kharif/rabi/zaid)
        long_crop = _rules.long_crop_gate(a3, a6)
        guidance = _rules.crop_guidance(tier, season, long_crop, sowing_window=sowing)  # types, calendar-appropriate

        # confidence — spatial CONSENSUS (per-well z-spread) weakest-linked with the METHOD-based
        # freshness band computed above (good=fresh / medium=analog / poor=climatology). The per-well
        # changes (fc3/fc6_by_code) are the LIVE-or-ANALOG values actually used (from the blend).
        conf = _consensus.assess(nb, fc3_by_code, fc6_by_code, anchor, freshness_band,
                                 n_years=self.n_years, gap_days=self.gap_days)
        if getattr(nb, "fallback_codes", None):   # normals came from the WRIS backup -> less validated
            conf["level"] = _consensus.cap_level(conf["level"], "MEDIUM")

        adv = {
            "status": "ok",
            "location": {"lat": qlat, "lon": qlon},
            "anchor_date": str(anchor)[:10],
            "forecast_date": forecast_date,
            "numbers": {
                "forecast_change_m": forecast_3m,
                "normal_seasonal_change_m": nrm.normal_seasonal,
                "latest_yoy_change_m": nrm.latest_yoy,
                "normal_yoy_change_m": nrm.normal_yoy,
                "forecast_change_m_6m": forecast_6m,
                "normal_seasonal_change_m_6m": nrm.normal_seasonal_6m,
                "band": nrm.band,
            },
            "season": season,
            "sowing_window": sowing,
            "regime": {"a": a3, "b": b, "d_a": d_a, "d_b": d_b, "a_6m": a6},
            "net_read": cell.get("net_read"),
            "crop_lean": cell.get("crop_lean"),
            "crop_tier": tier,
            "crop_guidance": guidance,
            "long_crop": long_crop,
            "confidence": conf,
            "outlook_phrase": _lens.phrase_outlook(nrm.normal_seasonal, a3, nrm.band),
            "caveats": list(_CAVEATS),
        }
        adv["message"] = _phr.phrase(adv, slm=self.slm)
        if basis in ("analog", "climatology"):     # LABEL the stale-data fallback honestly (deterministic)
            _bnote = ("seasonal estimate from recent years (no recent live reading here)"
                      if basis == "analog"
                      else "estimate from the 10-yr seasonal normal (no recent data here)")
            adv["message"] = adv["message"] + f"\n• Basis: {_bnote} — treat as indicative, not a live forecast."
        # SOFTEN the surfaced confidence: internal HIGH/MEDIUM/LOW is a spatial-CONSENSUS proxy (well
        # agreement + freshness), NOT a real prediction interval — so the OUTPUT never claims a strong
        # "HIGH confidence". The message already rendered a gentle sentence (phraser); relabel the
        # structured field to the matching soft word so nothing downstream says HIGH/MEDIUM/LOW.
        adv["confidence"]["level"] = _consensus.SOFT_LABEL.get(
            adv["confidence"]["level"], adv["confidence"]["level"])
        # ALWAYS-HIDDEN internals — NEVER shipped, even with verbose=True. These are used ONLY to BUILD
        # the message above (the caution/outlook text still reaches the farmer inside `message`).
        adv["numbers"].pop("band", None)
        adv["regime"].pop("d_a", None)
        adv["regime"].pop("d_b", None)
        for _k in ("crop_lean", "crop_tier", "caveats", "outlook_phrase",
                   "season", "sowing_window", "net_read"):
            adv.pop(_k, None)
        # Default (verbose=False) additionally strips the remaining internal reasoning; verbose=True
        # keeps the full normals + a/b/a_6m regime bins + confidence.freshness for debugging.
        if not verbose:
            adv["numbers"] = {"forecast_change_m": forecast_3m, "forecast_change_m_6m": forecast_6m}
            adv["regime"] = {"a": a3, "b": b, "a_6m": a6}
            adv["confidence"].pop("freshness", None)
        else:
            adv["_explanation"] = _explain(adv)   # plain-language key for the fields (verbose only)
        # Surface ONLY the imagery rate-limit note (if any) as `warnings` — the one data issue worth
        # flagging to the caller. Other internal diagnostics (stale forward-fills, per-well availability,
        # single-neighbour) stay OUT of the response to keep it digestible; `status`/`confidence` already
        # carry reliability. Present ONLY when there was a rate-limit; a clean run omits `warnings`.
        _img = [str(w) for w in (fc.get("warnings") or []) if str(w).startswith("imagery:")]
        if _img:
            adv["warnings"] = _img
        return adv

    # ------------------------------------------------------------------ decline
    def _decline(self, lat, lon, anchor, forecast_date=None, reason=""):
        adv = {
            "status": "insufficient_data",
            "location": {"lat": lat, "lon": lon},
            "anchor_date": str(anchor)[:10],
            "forecast_date": forecast_date,
            "reason": reason,
            "confidence": {"level": _consensus.SOFT_LABEL["LOW"]},
            "crop_guidance": None,
        }
        adv["message"] = _phr.phrase(adv, slm=self.slm)
        return adv
