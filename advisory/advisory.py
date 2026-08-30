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
from advisory import reliability as _reliability
from advisory import rule_engine as _rules
from advisory import season_lens as _lens

_CAVEATS = [
    "'normal' is the 10-yr median; where the aquifer is declining, that baseline is already "
    "depleted — 'normal' means typical-of-a-falling-decade, not safe/sustainable",
    "groundwater lens only; crop TYPES are guidance — pick the specific crop locally (soil/season/price/KVK)",
]

# Last-year (year-over-year) context, in plain words. From bin_b's above/normal/below where, for
# depth-to-water, "above" = wetter than its yoy normal and "below" = a deficit. Surfaced only under
# verbose (the farmer message always mentions it); field name `last_year`, not the old `regime.b`.
_LAST_YEAR = {"above": "wetter", "normal": "normal", "below": "drier"}


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
        # A "data_source_unavailable" error = wells ARE nearby but none has a usable LIVE reading (e.g.
        # WRIS empty), so the forecaster couldn't run. Don't decline yet — fall through to a normals/CSV
        # -driven SEASONAL-ANALOG fallback (guarded below by coverage <=15 km AND >=2 yrs of seasonal
        # history), still useful to the farmer. Any OTHER error (bad_request / no_neighbours / internal)
        # declines immediately.
        analog_only = False
        if fc.get("status") == "error":
            _err = fc.get("error") or {}
            if _err.get("code") != "data_source_unavailable":
                _code = _err.get("code")
                # Map the forecaster's error code to a stable advisory reason_code:
                #   no_neighbours / insufficient_neighbours -> out_of_network (coverage — no/too-few wells)
                #   bad_request / station_not_found         -> bad_request   (usage — future date, bad location)
                #   anything else (internal_error, ...)     -> forecast_failed
                _rc = {"no_neighbours": "out_of_network",
                       "insufficient_neighbours": "out_of_network",
                       "bad_request": "bad_request",
                       "station_not_found": "bad_request"}.get(_code, "forecast_failed")
                return self._decline(
                    lat, lon, anchor, reason=_err.get("message", "forecast failed"),
                    reason_code=_rc, details={"error_code": _code})
            analog_only = True

        from dateutil.relativedelta import relativedelta as _rd
        loc = fc.get("location", {}) or {}
        qlat = loc.get("lat", lat)
        qlon = loc.get("lon", lon)
        if (qlat is None or qlon is None) and station_code:      # the error path may omit location
            _hit = eng.registry.latlon_of(station_code)
            if _hit:
                qlat, qlon = _hit
        forecast_date = (fc.get("forecast_date")
                         or str(anchor + _rd(months=getattr(eng.data_config, "forecast_horizon_months", 3)))[:10])
        forecast_3m = None if analog_only else fc.get("change_m")

        # optional 6m forecast (second engine) — also keep its per-well predictions (preds6).
        # Skipped in the analog-only path (that engine would also lack any live GWL).
        forecast_6m = None
        preds6 = []
        if self.engine_6m is not None and not analog_only:
            f6, preds6 = self.engine_6m.predict_with_neighbours(
                lat=lat, lon=lon, date=date, station_code=station_code, leave_one_out=leave_one_out)
            if f6.get("status") != "error":
                forecast_6m = f6.get("change_m")

        # neighbours + histories -> normals + band  (reuse registry + numeric source)
        exclude = station_code if (station_code and leave_one_out) else None
        nb = _nb.gather(eng.registry, self.normals_source or eng._numeric, qlat, qlon,
                        eng.cfg.k_neighbours, anchor, exclude=exclude)
        # CONSISTENCY (live path): compute the normals over the SAME wells the FORECAST IDW'd — the
        # fresh-anchor survivors in preds3 — so `forecast vs normal` is an apples-to-apples spatial
        # blend (identical stations + 1/d^2 weights), not a comparison across two different well sets.
        # Only wells that ALSO have normals history survive (intersection); if that leaves nothing usable,
        # keep the full set rather than manufacture a decline. The ANALOG path is already consistent
        # (forecast_from_history + normals both run over `nb`), so it is left untouched.
        if not analog_only and preds3:
            _kept = {getattr(p, "station_code", None) for p in preds3}
            _kept.discard(None)
            if _kept:
                nb_fc = _nb.restrict(nb, _kept)
                if nb_fc.n_with_data:
                    nb = nb_fc
        nrm = _norm.compute(nb, anchor, n_years=self.n_years, gap_days=self.gap_days)

        # insufficient-data gate: decline ONLY when there is genuinely no usable normal (no seasonal
        # or yoy history at all -> a None median) or no well within range. A THIN history (2-3 yrs, or
        # even 1) is now WORKED WITH rather than declined: the median normals hold with few points and
        # _band falls back to the interannual range (>=2 yrs) or the spatial spread (1 yr) instead of
        # None. (Dropped the old `min(n_years) < 3` decline; band is None now only when 0 yrs, already
        # caught by normal_seasonal is None.)
        # COVERAGE, tiered by the nearest-well distance:
        #   <= 15 km   -> normal advisory (local, unchanged).
        #   15-40 km   -> EXTENDED: still advise, but it is a broad REGIONAL read — capped to `thin`
        #                 confidence + a distance caveat on the message (both applied below). The forecast
        #                 math is unchanged (far wells already self-down-weight via 1/d^2); we only relabel.
        #   > 40 km    -> decline: genuinely outside the monitored network (e.g. offshore / no wells).
        _COVERAGE_KM, _EXTENDED_KM = 15.0, 40.0
        far = nb.nearest_km is not None and _COVERAGE_KM < nb.nearest_km <= _EXTENDED_KM
        if nb.nearest_km is not None and nb.nearest_km > _EXTENDED_KM:
            return self._decline(qlat, qlon, anchor, forecast_date=forecast_date,
                                 reason=(f"no monitored well within {_EXTENDED_KM:.0f} km (nearest is "
                                         f"{nb.nearest_km:.0f} km) — this point is outside our well network here"),
                                 reason_code="out_of_network",
                                 details={"nearest_km": round(nb.nearest_km, 1)})
        # HISTORY gate: a usable SEASONAL normal + band is ALWAYS required. The year-over-year normal is
        # required for the LIVE path (it drives the 'past year' regime b); the ANALOG-only path may
        # proceed SEASONAL-only (regime b becomes unknown) so a >=2yr-seasonal well isn't lost for want
        # of a yoy pair. THIN seasonal history (2-3 yrs, even 1) is worked with, not declined.
        if (nrm.normal_seasonal is None or nrm.band is None
                or (not analog_only and (nrm.latest_yoy is None or nrm.normal_yoy is None))):
            return self._decline(qlat, qlon, anchor, forecast_date=forecast_date,
                                 reason=(f"wells are nearby ({nb.nearest_km:.0f} km) but lack enough history to "
                                         f"form a seasonal normal (usable years: {nrm.n_years_seasonal})"),
                                 reason_code="sparse_history",
                                 details={"nearest_km": round(nb.nearest_km, 1),
                                          "usable_years": nrm.n_years_seasonal})

        # ANALOG-ONLY fallback (no live reading anywhere -> no model ran): require >=2 yrs of seasonal
        # history to make the estimate meaningful; otherwise decline honestly.
        if analog_only and (nrm.n_years_seasonal or 0) < 2:
            return self._decline(qlat, qlon, anchor, forecast_date=forecast_date,
                                 reason=(f"wells are nearby ({nb.nearest_km:.0f} km) but have no recent live "
                                         f"reading and <2 yrs of seasonal history to estimate from"),
                                 reason_code="no_recent_reading",
                                 details={"nearest_km": round(nb.nearest_km, 1),
                                          "usable_years": nrm.n_years_seasonal})

        # STALE wells: swap the model forecast for the well's SEASONAL ANALOG (median recent seasonal
        # change), then re-IDW to Q. Freshness becomes METHOD-based (good=fresh / medium=analog /
        # poor=climatology). No swap when every well is fresh (byte-identical live path). See
        # docs/stale_data_seasonal_analog_DISCUSSION.txt. The per-well changes also feed the consensus.
        import re as _re
        _ages = [int(x) for w in (fc.get("warnings") or []) for x in _re.findall(r"(\d+)\s*d stale", str(w))]
        freshness_days = min(_ages) if _ages else None
        if analog_only:
            # NO model ran: pure seasonal analog per well (median same-season change), IDW to Q; the
            # basis note + THIN confidence cap (below) flag it as an estimate, not a live forecast.
            forecast_3m, fc3_by_code, freshness_band, basis = _analog.forecast_from_history(
                nb, anchor, 3, nrm.normal_seasonal, gap_days=self.gap_days)
            forecast_6m, fc6_by_code, _fb6, _b6 = _analog.forecast_from_history(
                nb, anchor, 6, nrm.normal_seasonal_6m, gap_days=self.gap_days)
            if forecast_3m is None:                       # nothing usable even from history
                return self._decline(qlat, qlon, anchor, forecast_date=forecast_date,
                                     reason=(f"wells are nearby ({nb.nearest_km:.0f} km) but have no usable "
                                             f"seasonal history to estimate from"),
                                     reason_code="no_seasonal_history",
                                     details={"nearest_km": round(nb.nearest_km, 1),
                                              "usable_years": nrm.n_years_seasonal})
        else:
            forecast_3m, fc3_by_code, freshness_band, basis = _analog.blend(
                preds3, nb, anchor, 3, nrm.normal_seasonal, forecast_3m, freshness_days, gap_days=self.gap_days)
            forecast_6m, fc6_by_code, _fb6, _b6 = _analog.blend(
                preds6, nb, anchor, 6, nrm.normal_seasonal_6m, forecast_6m, freshness_days, gap_days=self.gap_days)

        # EXTENDED-band plausibility clamp: a 15-40 km read often rests on ONE nearby well, so a single
        # anomalous well can push the blended forecast far past what the LOCAL history supports. Keep it
        # within `normal +/- k*band` (k=3; the same band the regime bins use), self-calibrating to the
        # area's own variability. Applies to whatever the band produced (live blend OR analog). The
        # <=15 km path (far=False) is UNTOUCHED.
        if far:
            _K = 3.0
            def _clamp(v, nrmv, bnd):
                if v is None or nrmv is None or bnd is None:
                    return v
                return max(nrmv - _K * bnd, min(nrmv + _K * bnd, v))
            forecast_3m = _clamp(forecast_3m, nrm.normal_seasonal, nrm.band)
            forecast_6m = _clamp(forecast_6m, nrm.normal_seasonal_6m, nrm.band_6m or nrm.band)

        # binning + regime + gates — each comparison uses the band of its OWN change series
        band_yoy = nrm.band_yoy or nrm.band                 # yoy band; fall back to seasonal if too few yoy yrs
        band_6m = nrm.band_6m or nrm.band
        a3, d_a = _rules.bin_a(forecast_3m, nrm.normal_seasonal, nrm.band)
        b, d_b = _rules.bin_b(nrm.latest_yoy, nrm.normal_yoy, band_yoy)
        a6, _ = _rules.bin_a(forecast_6m, nrm.normal_seasonal_6m, band_6m)
        # TWO-AXIS DRIVERS (advisory/reliability.py; pure arithmetic, no model/service):
        #   water_trend = rising/falling (forecast sign vs ZERO) + direction accuracy — the STRONGEST signal
        #   clarity     = how decisive the vs-normal (a3) call is + its calibrated reliability
        _clar = _reliability.clarity_of(d_a, nrm.band, a3)             # {level, reliability_pct} or None
        _wt = _reliability.direction_of(forecast_3m, horizon="3m")    # {direction, reliability_pct} or None
        _wt_dir = (_wt or {}).get("direction")
        # crop tier is driven by (water_trend x vs_normal) — reflects BOTH "is there water?" and "vs usual?"
        # so "above normal" in a DEPLETION season (still falling) can't map to water-heavy. yoy no longer
        # drives the tier — it is context only.
        cell = self.rules.get((_wt_dir, a3), {})
        tier = cell.get("tier")
        season = _lens.season_of(nrm.normal_seasonal, nrm.band)   # wet/dry/transition (deadband)
        sowing = _lens.sowing_window(anchor)                      # calendar tag only (kharif/rabi/zaid)
        long_crop = _rules.long_crop_gate(a3, a6, water_trend=_wt_dir, clarity=(_clar or {}).get("level"))
        guidance = _rules.crop_guidance(tier, season, long_crop, sowing_window=sowing)  # types, calendar-appropriate

        # confidence — spatial CONSENSUS (per-well z-spread) weakest-linked with the METHOD-based
        # freshness band computed above (good=fresh / medium=analog / poor=climatology). The per-well
        # changes (fc3/fc6_by_code) are the LIVE-or-ANALOG values actually used (from the blend).
        conf = _consensus.assess(nb, fc3_by_code, fc6_by_code, anchor, freshness_band,
                                 n_years=self.n_years, gap_days=self.gap_days)
        if getattr(nb, "fallback_codes", None):   # normals came from the WRIS backup -> less validated
            conf["level"] = _consensus.cap_level(conf["level"], "MEDIUM")
        if analog_only:                           # no live reading anywhere -> never above THIN ("thin")
            conf["level"] = _consensus.cap_level(conf["level"], "LOW")
        if far:                                   # 15-40 km: regional read from distant wells -> never above THIN
            conf["level"] = _consensus.cap_level(conf["level"], "LOW")

        # OUTLOOK for the response — reuse the two-axis drivers computed above (water_trend + clarity).
        outlook = {}
        if _wt:
            outlook["water_trend"] = _wt
        if a3:
            _vn = {"level": a3, "clarity": (_clar or {}).get("level")}
            _rp = (_clar or {}).get("reliability_pct")
            if _rp is not None:              # above/below call -> a real calibrated side-accuracy (74/71)
                _vn["reliability_pct"] = _rp
            else:                            # "typical" (within band): no honest above/below % exists, so
                _vn["note"] = ("within the normal band — no above/below-normal call to rate; "
                               "go by the water-trend read (rise/fall) instead")   # a note, NOT a null pct
            outlook["vs_normal"] = _vn

        adv = {
            "status": "ok",
            "location": {"lat": qlat, "lon": qlon},
            "anchor_date": str(anchor)[:10],
            "forecast_date": forecast_date,
            "numbers": {
                "forecast_change_m": forecast_3m,
                "normal_seasonal_change_m": nrm.normal_seasonal,       # verbose: what the 3m is compared to
                "forecast_change_m_6m": forecast_6m,
                "normal_seasonal_change_m_6m": nrm.normal_seasonal_6m,  # verbose
            },
            "outlook": outlook,                        # the two calibrated axes (water_trend + vs_normal)
            "last_year": _LAST_YEAR.get(b),            # verbose: yoy context (was regime.b) — "wetter"/"normal"/"drier"
            "crop_guidance": guidance,
            "long_crop": long_crop,
            "confidence": conf,
            # --- internal-only (used to BUILD the message, then popped; never shipped) ---
            "season": season, "sowing_window": sowing, "net_read": cell.get("net_read"),
            "crop_lean": cell.get("crop_lean"), "crop_tier": tier, "caveats": list(_CAVEATS),
            "outlook_phrase": _lens.phrase_outlook(nrm.normal_seasonal, a3, nrm.band),
        }
        adv["message"] = _phr.phrase(adv, slm=self.slm)
        if basis in ("analog", "climatology"):     # LABEL the stale-data fallback honestly (deterministic)
            _bnote = ("seasonal estimate from recent years (no recent live reading here)"
                      if basis == "analog"
                      else "estimate from the 10-yr seasonal normal (no recent data here)")
            adv["message"] = adv["message"] + f"\n• Basis: {_bnote} — treat as indicative, not a live forecast."
        if far:                                    # 15-40 km: honest distance caveat (regional, not local)
            adv["message"] = adv["message"] + (
                f"\n• Heads-up: the nearest monitored well is {nb.nearest_km:.0f} km away — this is a broad "
                f"regional read for your area, not a well-specific forecast; treat it as indicative and "
                f"check locally (KVK).")
        # SOFTEN the surfaced confidence: internal HIGH/MEDIUM/LOW is a spatial-CONSENSUS proxy (well
        # agreement + freshness), NOT a real prediction interval — so the OUTPUT never claims a strong
        # "HIGH confidence". The message already rendered a gentle sentence (phraser); relabel the
        # structured field to the matching soft word so nothing downstream says HIGH/MEDIUM/LOW.
        adv["confidence"]["level"] = _consensus.SOFT_LABEL.get(
            adv["confidence"]["level"], adv["confidence"]["level"])
        # crop_guidance.long_crop_ok / duration_hint just MIRROR long_crop.allowed -> drop them; the
        # top-level long_crop {allowed, reason} is the single source of truth for duration.
        for _k in ("long_crop_ok", "duration_hint"):
            (adv.get("crop_guidance") or {}).pop(_k, None)
        # ALWAYS-HIDDEN internals — used ONLY to BUILD the message above; never shipped (the wording still
        # reaches the farmer inside `message`).
        for _k in ("crop_lean", "crop_tier", "caveats", "outlook_phrase",
                   "season", "sowing_window", "net_read"):
            adv.pop(_k, None)
        # Default (verbose=False) strips the debug detail; verbose=True keeps the normals it was compared
        # against, the `last_year` yoy context, and confidence.freshness.
        if not verbose:
            adv["numbers"] = {"forecast_change_m": forecast_3m, "forecast_change_m_6m": forecast_6m}
            adv.pop("last_year", None)               # yoy context is a verbose-only detail (still in message)
            adv["confidence"].pop("freshness", None)
        # Surface ONLY the imagery rate-limit note (if any) as `warnings` — the one data issue worth
        # flagging to the caller. Other internal diagnostics (stale forward-fills, per-well availability,
        # single-neighbour) stay OUT of the response to keep it digestible; `status`/`confidence` already
        # carry reliability. Present ONLY when there was a rate-limit; a clean run omits `warnings`.
        _img = [str(w) for w in (fc.get("warnings") or []) if str(w).startswith("imagery:")]
        if _img:
            adv["warnings"] = _img
        return adv

    # ------------------------------------------------------------------ decline
    def _decline(self, lat, lon, anchor, forecast_date=None, reason="",
                 reason_code=None, details=None):
        """Structured 'insufficient_data' response.

        reason_code = a machine-stable enum of WHY we declined (out_of_network |
        sparse_history | no_recent_reading | no_seasonal_history | forecast_failed);
        details = the concrete numbers behind it ({nearest_km, usable_years, ...}) so a
        client can tell "no wells nearby" from "wells nearby but too little history".
        `reason` stays the human free-text; the phraser renders an actionable message
        from reason_code + details.
        """
        adv = {
            "status": "insufficient_data",
            "location": {"lat": lat, "lon": lon},
            "anchor_date": str(anchor)[:10],
            "forecast_date": forecast_date,
            "reason": reason,
            "reason_code": reason_code,
            "details": {k: v for k, v in (details or {}).items() if v is not None},
            "confidence": {"level": _consensus.SOFT_LABEL["LOW"]},
            "crop_guidance": None,
        }
        adv["message"] = _phr.phrase(adv, slm=self.slm)
        return adv
