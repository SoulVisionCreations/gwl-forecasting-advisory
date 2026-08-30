"""build_response — assemble the JSON the CLI prints / the API returns.

Minimal default, rich details. The model's genuine output is the 3-month CHANGE
(`change_m`). The absolute forecast and "current" are SPATIALLY INTERPOLATED (IDW) from
the neighbouring wells — there is no sensor at an off-station query point — so they are
easy to over-read and are NOT surfaced by default. Everything beyond the change (absolute
forecast/current, confidence counts, kriging, per-well breakdown, validation extras) lives
under `details`, returned ONLY when `details=True`.

Default keys: status, [error], location, as_of_date, forecast_date, change_m, warnings.

The CLI (`python -m inference`) and FastAPI `/predict` return the IDENTICAL dict — both
serialize exactly what this function builds, so the shape stays in lockstep.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inference.types import StationPrediction, InterpolationResult

# GWL is depth-to-water (metres), always >= 0 by the training abs() sign convention
# (data_preparation.py:753). The DELTA CLAMP is the single guard: with |delta| <=
# pct*|current| and pct in (0,1), result = current + delta >= (1-pct)*current >= 0, so
# it can't go negative. Decision Jul 3 2026: dropped the redundant positivity floor
# (kept only the delta clamp; it also caps spikes and matches the trained/reported
# metrics). If the clamp is ever disabled (pct=0) or pct>=1, positivity is NOT
# guaranteed — the engine warns at init; re-add a floor there if that config is used.


def _clamp_delta(delta: Optional[float], ref_current: Optional[float], pct: float) -> Optional[float]:
    """Final clamp of the interpolated delta by the QUERY point's reference current:
    |delta| <= pct*|ref_current|. Mirrors the per-prediction training clamp, applied
    at the query so the interpolated GWL can't deviate wildly from the query's level."""
    if delta is None or ref_current is None or not pct or pct <= 0.0:
        return delta
    max_dev = pct * abs(ref_current)
    return max(-max_dev, min(max_dev, delta))


def _fmt_date(d) -> str:
    return d.strftime("%Y-%m-%d") if isinstance(d, datetime) else str(d)[:10]


def build_response(
    status: str,
    query: dict,
    preds: "list[StationPrediction]",
    interp: "Optional[InterpolationResult]",
    dropped: "list[dict]",
    warnings: "list[str]",
    s1: Optional[dict] = None,
    error: Optional[dict] = None,
    clamp_pct: float = 0.0,
    details: bool = False,
) -> dict:
    """Assemble the response. `query` carries lat/lon/current_date/target_date.
    clamp_pct (>0) clamps the interpolated delta by the query current. DEFAULT surfaces
    only the predicted change; details=True adds the interpolated absolutes, confidence,
    kriging, per-well breakdown and validation extras under `details`."""
    out: dict = {"status": status}                       # ok | partial | error
    if error is not None:
        out["error"] = error                             # {code, message}; omitted when None
    out.update({
        "location": {"lat": query.get("lat"), "lon": query.get("lon")},
        "as_of_date": _fmt_date(query["current_date"]) if query.get("current_date") else None,
        "forecast_date": _fmt_date(query["target_date"]) if query.get("target_date") else None,
        "change_m": None,                # THE answer: predicted 3-month change (the model output)
        "warnings": warnings,
    })

    # All absolute quantities are IDW estimates from neighbours -> `details`, not default.
    idw_gwl = kriging_gwl = k_delta = None
    idw_current = interp.idw_current if interp is not None else None
    if interp is not None and interp.idw_delta is not None:
        delta = _clamp_delta(interp.idw_delta, idw_current, clamp_pct)
        idw_gwl = None if idw_current is None else round(idw_current + delta, 2)
        k_delta = _clamp_delta(interp.kriging_delta, idw_current, clamp_pct)
        kriging_gwl = (None if (idw_current is None or k_delta is None)
                       else round(idw_current + k_delta, 2))
        out["change_m"] = None if delta is None else round(delta, 2)

    # nearby_wells + nearest_well_km are opt-in — built inside the `details` block below.

    # ── details (opt-in): interpolated absolutes + confidence + kriging + per-well + validation ──
    if details:
        weight_by = {w["station_code"]: w.get("weight") for w in (interp.nn_pred if interp else [])}
        wells = []
        for p in preds:
            wells.append({
                "code": p.station_code,
                "km": round(p.distance_km, 2),
                "weight": weight_by.get(p.station_code),
                "current_gwl_m": round(p.current_gwl, 2),
                "change_m": round(p.predicted_delta, 2),
                "pred_gwl_m": round(p.predicted_gwl, 2),
                "actual_gwl_m": None if p.actual_gwl is None else round(p.actual_gwl, 2),
                "trend": p.trend,
            })
        # stable candidate set (used/outlier/no_data) — presentation view of the neighbourhood.
        # Same geographically-nearest wells on any date; only statuses/values change. used feeds
        # the headline, outlier has data but was rejected, no_data had no reading at this anchor.
        _outlier_codes = {o["station_code"] for o in (interp.outliers or [])} if interp is not None else set()
        _cands = []
        for p in preds:
            _cands.append((p.distance_km, {
                "km": round(p.distance_km, 2),
                "change_m": round(p.predicted_delta, 2),
                "water_level": ("stable" if p.predicted_delta == 0
                                else "falling" if p.predicted_delta > 0 else "rising"),
                "status": "outlier" if p.station_code in _outlier_codes else "used",
            }))
        for _d in dropped:
            _dk = _d.get("distance_km")
            if _dk is None:
                continue
            _cands.append((_dk, {"km": round(_dk, 2), "change_m": None,
                                 "water_level": None, "status": "no_data"}))
        _cands.sort(key=lambda c: c[0])
        _used_km = [dist for dist, row in _cands if row["status"] == "used"]
        _actual = s1.get("actual_gwl") if s1 is not None else None
        det: dict = {
            # absolute estimates are SPATIALLY INTERPOLATED (IDW of neighbour currents) — no
            # sensor exists at the query point, so treat these as estimates, not measurements.
            "forecast_gwl_m": idw_gwl,                                             # current_gwl_m + change_m
            "current_gwl_m": None if idw_current is None else round(idw_current, 2),
            "n_wells_used": len(preds),
            "nearest_well_km": round(min(_used_km), 2) if _used_km else None,    # nearest well feeding the estimate
            "nearby_wells": [row for _, row in _cands],                          # stable candidate set w/ status
            "method": "idw (primary) + kriging (alternative)",
            "forecast_gwl_m_raw": (None if (interp is None or interp.idw_delta is None or idw_current is None)
                                   else round(idw_current + interp.idw_delta, 2)),   # pre-clamp IDW
            "kriging_gwl_m": kriging_gwl,                                          # clamped kriging forecast
            "kriging_change_m": None if k_delta is None else round(k_delta, 2),    # clamped kriging change
            "kriging_gwl_m_raw": (None if (interp is None or interp.kriging_delta is None or idw_current is None)
                                  else round(idw_current + interp.kriging_delta, 2)),  # pre-clamp kriging
            "kriging_note": interp.kriging_note if interp is not None else None,
            "wells": wells,
            "dropped": dropped,                          # [{station_code, stage, reason}] (data-collection)
            "outliers_excluded": (interp.outliers or []) if interp is not None else [],  # magnitude outliers cut from IDW/kriging
        }
        # ── validation extras (known station_code): compare against the real target ──
        if s1 is not None:
            det["validation_station"] = s1.get("station_code")
            det["actual_gwl_m"] = None if _actual is None else round(_actual, 2)
            det["forecast_error_m"] = (None if (idw_gwl is None or _actual is None)
                                       else round(idw_gwl - _actual, 2))
            det["kriging_error_m"] = (None if (kriging_gwl is None or _actual is None)
                                      else round(kriging_gwl - _actual, 2))
        out["details"] = det

    return out
