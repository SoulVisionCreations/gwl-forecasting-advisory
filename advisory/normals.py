"""normals.py — interpolated historical yardsticks + the LOCAL band (Stage 3b).

GWL analog of NDVI's aggregate.py. From the neighbour wells' RECORDED history, IDW-
interpolate the per-year seasonal change and the year-over-year change to the query
point; the medians are the "normal" yardsticks and the IQR of the seasonal changes
sets the local BAND = 0.5 x IQR(seasonal changes) (floor 0.3 m) — so "above/below
normal" travels across aquifer types instead of a fixed 1.5 m. NO model here.

All quantities are change_m = depth(later) - depth(earlier) on abs(gwl) = depth-to-water.
"""
from __future__ import annotations

from datetime import datetime

from dateutil.relativedelta import relativedelta

from advisory.types import Normals

_BAND_FLOOR = 0.3


def _depth_at(series, when, gap_days):
    """Nearest recorded depth within +/- gap_days of `when`, else None."""
    import numpy as np
    import pandas as pd

    if series is None or not len(series):
        return None
    dd = (series.index - pd.Timestamp(when)).days.values
    m = np.abs(dd) <= gap_days
    if not m.any():
        return None
    return float(series.values[m][np.abs(dd[m]).argmin()])


def _idw_change(nb, a, b, gap_days):
    """IDW over neighbours of (depth(b) - depth(a)); only wells with BOTH endpoints."""
    num = den = 0.0
    for c in nb.codes:
        x = _depth_at(nb.series.get(c), a, gap_days)
        y = _depth_at(nb.series.get(c), b, gap_days)
        if x is None or y is None:
            continue
        w = nb.weights[c]
        num += w * (y - x)
        den += w
    return (num / den) if den else None


def _window(mf, mt, y):
    """(start, end) datetimes for month mf->mt anchored in year y (crosses year if mt<=mf)."""
    return datetime(y, mf, 15), datetime(y if mt > mf else y + 1, mt, 15)


def _changes_over_years(nb, mf, mt, years, gap_days):
    out = []
    for y in years:
        a, b = _window(mf, mt, y)
        c = _idw_change(nb, a, b, gap_days)
        if c is not None:
            out.append(c)
    return out


def compute(nb, anchor_date, n_years=10, gap_days=45) -> Normals:
    """The four normals + the local band, for the anchor's season window.

    Both the 3m (anchor+3) and 6m (anchor+6) seasonal normals are computed regardless of
    the forecaster's own horizon, so the advisory can gate long crops on the 6m outlook.
    """
    import numpy as np

    anchor = (datetime.strptime(anchor_date[:10], "%Y-%m-%d")
              if isinstance(anchor_date, str) else anchor_date)
    mf = anchor.month
    m3 = (anchor + relativedelta(months=3)).month
    m6 = (anchor + relativedelta(months=6)).month
    y0 = anchor.year

    seas_years = range(y0 - n_years, y0)                 # e.g. 2014..2023 for a 2024 anchor
    seasonal_changes = _changes_over_years(nb, mf, m3, seas_years, gap_days)
    seasonal6_changes = _changes_over_years(nb, mf, m6, seas_years, gap_days)

    yoy_years = range(y0 - 1 - n_years, y0 - 1)          # windows y->y+1 with end <= y0-1 (exclude observed)
    yoy_changes = _changes_over_years(nb, mf, mf, yoy_years, gap_days)
    # latest_yoy = the most recent OBSERVABLE year-over-year change (depth now vs ~1 yr earlier).
    # Ideally measured AT the anchor, but the historical normals source lags (NWDP publishes ~1-2
    # quarters behind), so a reading exactly at a near-today anchor is usually missing. Rather than
    # decline, walk the endpoint back from the anchor to the most recent date the IDW blend supports
    # (bounded to 18 months). A historical anchor that HAS data at the anchor resolves on the first
    # step -> identical to before; only near-today anchors fall back to the latest observable pair.
    # Staleness is not lost — the advisory degrades CONFIDENCE by anchor age elsewhere.
    latest_yoy = None
    t = anchor
    for _ in range(18):                                  # up to 18 months back from the anchor
        c = _idw_change(nb, t - relativedelta(years=1), t, gap_days)
        if c is not None:
            latest_yoy = c
            break
        t = t - relativedelta(months=1)

    # neighbour_spread = SPATIAL disagreement among the wells (PI proxy, not interannual noise):
    # each well's OWN most-recent complete seasonal change; std across wells. Tight = the area is
    # spatially coherent -> interpolation trustworthy; wide = heterogeneous -> shakier.
    per_well = []
    for c in nb.codes:
        s = nb.series.get(c)
        if s is None:
            continue
        for yy in range(y0 - 1, y0 - 1 - n_years, -1):   # most recent year this well has both endpoints
            a, b = _window(mf, m3, yy)
            x = _depth_at(s, a, gap_days)
            y = _depth_at(s, b, gap_days)
            if x is not None and y is not None:
                per_well.append(y - x)
                break
    neighbour_spread = float(np.std(per_well)) if len(per_well) >= 2 else None

    def _med(v):
        return round(float(np.median(v)), 2) if v else None

    def _band(v):
        # Local band = how far the forecast must beat the normal to count as above/below — tied to
        # THAT series' own spread, so the anomaly means the same real thing across aquifers. Each
        # comparison uses the band of its OWN change series (seasonal for (a), yoy for (b)).
        # Few-history fallback (so a thin well still answers instead of declining):
        #   >= 4 yrs -> 0.5 * IQR   (robust interannual spread)
        #   2-3 yrs  -> 0.5 * (max - min)   (rough interannual spread from the range we have)
        #   1 yr     -> the spatial neighbour_spread (no interannual info to go on), else the floor
        # floored at _BAND_FLOOR; None ONLY when there are 0 years (no normal exists either).
        if not v:
            return None
        if len(v) >= 4:
            iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
            return round(max(0.5 * iqr, _BAND_FLOOR), 2)
        if len(v) >= 2:
            return round(max(0.5 * (max(v) - min(v)), _BAND_FLOOR), 2)
        return round(max(neighbour_spread or 0.0, _BAND_FLOOR), 2)

    return Normals(
        normal_seasonal=_med(seasonal_changes),
        normal_seasonal_6m=_med(seasonal6_changes),
        latest_yoy=None if latest_yoy is None else round(latest_yoy, 2),
        normal_yoy=_med(yoy_changes),
        band=_band(seasonal_changes),
        band_6m=_band(seasonal6_changes),
        band_yoy=_band(yoy_changes),
        seasonal_changes=[round(v, 2) for v in seasonal_changes],
        neighbour_spread=None if neighbour_spread is None else round(neighbour_spread, 2),
        n_years_seasonal=len(seasonal_changes),
        n_years_yoy=len(yoy_changes),
        n_wells_used=nb.n_with_data,
    )
