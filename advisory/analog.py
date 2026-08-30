"""analog.py — seasonal-analog fallback for a STALE well.

When a well's live GWL is > STALE_DAYS old, its model forecast is built on a flat forward-filled
sequence and is unreliable. Instead we use the well's OWN recent seasonal behaviour: the MEDIAN of
its most recent up-to-N same-window seasonal CHANGES (anchor-month -> anchor+horizon-month) from its
history. This IS a change, so it drops straight into the change-based decision layer in place of the
model forecast_change_m. No same-season history -> None (caller falls back to the climatology normal).

See docs/stale_data_seasonal_analog_DISCUSSION.txt (final spec). Per-well; the caller IDWs to Q.
"""
from __future__ import annotations

from dateutil.relativedelta import relativedelta

from advisory.normals import _depth_at, _window

STALE_DAYS = 180          # live reading older than this (>6 mo) -> use the analog, not the model
N_ANALOG_YEARS = 3        # median of up to this many most-recent complete same-window changes
_LOOKBACK_YEARS = 12      # how far back to search for those complete windows


def seasonal_change(series, anchor, horizon_months, gap_days=45):
    """MEDIAN of the well's most recent up-to-N same-window seasonal CHANGES (depth(end)-depth(start),
    window = anchor.month -> +horizon_months) from `series` (date-indexed depth). Most recent first,
    only windows that END on/before the anchor. Returns a float, or None if it has no such window."""
    import numpy as np
    if series is None or not len(series):
        return None
    mf = anchor.month
    mt = (anchor + relativedelta(months=horizon_months)).month
    changes = []
    for y in range(anchor.year, anchor.year - _LOOKBACK_YEARS, -1):    # most recent year first
        a, b = _window(mf, mt, y)
        if b > anchor:                                                 # window must be fully in the past
            continue
        x = _depth_at(series, a, gap_days)
        z = _depth_at(series, b, gap_days)
        if x is not None and z is not None:
            changes.append(z - x)
            if len(changes) >= N_ANALOG_YEARS:
                break
    return float(np.median(changes)) if changes else None


def blend(preds, nb, anchor, horizon_months, normal_change, live_forecast, freshness_days, gap_days=45):
    """Resolve each neighbour's per-well change nearest->farthest, fresh->stale, then IDW(1/d^2) to Q:
        fresh (<= STALE_DAYS)  -> the model's live predicted change
        stale (> STALE_DAYS)   -> the well's seasonal ANALOG (median recent seasonal change)
        stale + no analog      -> the 10-yr climatology normal_change
    Returns (change_at_Q, per_well_change_by_code, freshness_band, basis) where band/basis come from an
    ANY-FRESH rule: if >= 1 contributing well is fresh -> "live"/good; only when EVERY well is stale
    does the label fall to the DOMINANT (max 1/d^2 weight) stale well's analog/climatology. If NO well
    is stale, returns the engine's live_forecast unchanged (byte-identical fresh path) + the age-based
    band. `preds` = StationPredictions (need .data_age_days).
    """
    from advisory.consensus import _freshness_band

    by_code = {}
    if not any((p.data_age_days or 0) > STALE_DAYS for p in preds):     # all fresh -> unchanged
        for p in preds:
            by_code[p.station_code] = p.predicted_delta
        return live_forecast, by_code, _freshness_band(freshness_days), "live"

    num = den = 0.0
    contribs = []                                                      # (weight, band, method)
    for p in preds:
        w = 1.0 / max(p.distance_km, 0.01) ** 2
        age = p.data_age_days
        if age is not None and age > STALE_DAYS:                       # STALE well
            ch = seasonal_change(nb.series.get(p.station_code), anchor, horizon_months, gap_days)
            if ch is None:
                ch, band, method = normal_change, "poor", "climatology"
            else:
                band, method = "medium", "analog"
        else:                                                          # FRESH well -> model live
            ch, band, method = p.predicted_delta, _freshness_band(age), "live"
        if ch is None:
            continue
        by_code[p.station_code] = ch
        num += w * ch
        den += w
        contribs.append((w, band, method))
    if not contribs:
        return live_forecast, by_code, "poor", "climatology"
    # Overall label: if ANY contributing well is FRESH (live) — i.e. there is at least one recent live
    # reading nearby — present the call as live/good. A stale well's NUMBER already falls back to its
    # own seasonal analog in the IDW; this only governs the confidence/freshness LABEL, so one fresh
    # neighbour is enough to drop the "no recent live reading" caveat. Only when EVERY well is stale
    # (no fresh reading anywhere) do we fall to the dominant (nearest) well's analog/climatology label.
    n_fresh = sum(1 for (_w, _b, method) in contribs if method == "live")
    if n_fresh >= 1:
        live_ages = [p.data_age_days for p in preds
                     if p.data_age_days is not None and p.data_age_days <= STALE_DAYS]
        return num / den, by_code, _freshness_band(min(live_ages) if live_ages else None), "live"
    dom = max(contribs, key=lambda t: t[0])                            # all stale -> nearest well
    return num / den, by_code, dom[1], dom[2]


def forecast_from_history(nb, anchor, horizon_months, normal_change, gap_days=45):
    """NO-MODEL seasonal-analog fallback — used when NO neighbour had a usable LIVE reading, so the
    forecaster couldn't run at all (e.g. WRIS empty), but the normals source (CSV) still holds history.

    Computed by the SAME recipe as the NORMAL (normals._changes_over_years: IDW-per-year, then median),
    but over the most recent N_ANALOG_YEARS instead of the full ~10 — so the fallback forecast and the
    normal are one consistent method (both = median over years of the per-year IDW change), differing
    only in the window length. Falls back to the 10-yr climatology `normal_change` if no recent year
    has a usable IDW pair.

    Returns (change_at_Q, per_well_change_by_code, band, basis): basis='analog' if the recent-years
    recipe produced a value else 'climatology'; band is ALWAYS 'poor' (no live reading anywhere).
    by_code = each well's OWN recent-median change (for the consensus spread only, not the blend).
    change_at_Q is None when nothing usable at all -> the caller declines.
    """
    import numpy as np
    from advisory.normals import _changes_over_years   # identical per-year IDW recipe used by the normal

    mf = anchor.month
    mt = (anchor + relativedelta(months=horizon_months)).month
    y0 = anchor.year
    recent = range(y0 - N_ANALOG_YEARS, y0)              # most recent N years, leakage-free (all < y0)
    per_year = _changes_over_years(nb, mf, mt, recent, gap_days)   # per-year IDW change to Q (== normal's)

    by_code = {}                                          # per-well recent medians -> consensus spread only
    for code in nb.codes:
        v = seasonal_change(nb.series.get(code), anchor, horizon_months, gap_days)
        if v is not None:
            by_code[code] = v

    if per_year:
        return float(np.median(per_year)), by_code, "poor", "analog"
    if normal_change is not None:                         # no recent IDW pair -> 10-yr climatology
        return normal_change, by_code, "poor", "climatology"
    return None, by_code, "poor", "climatology"
