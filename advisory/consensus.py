"""consensus.py — spatial-CONSENSUS confidence for the advisory.

Do the neighbour wells AGREE about the trend at the query point? For each axis
(seasonal 3m, seasonal 6m, year-over-year) each well is scored as a z against ITS OWN
history (so aquifer/scale differences wash out), the z's are combined into a 1/d^2-
weighted SD, and the SD -> a HIGH/MEDIUM/LOW band (NDVI cutoffs). Weakest-link with
freshness (no min well-count gate). See docs/confidence_measure_design.txt.

The forecast VALUE at Q is unchanged (IDW of neighbour change_m, done in the engine);
this module only produces the CONFIDENCE in it.
"""
from __future__ import annotations

from datetime import datetime

from dateutil.relativedelta import relativedelta

from advisory.normals import _depth_at, _window

# provisional knobs (calibrate later — see docs/confidence_measure_design.txt)
MIN_YEARS = 2            # a well needs this many years of history to give a valid z (mean/std).
                         # Kept low on purpose: data is sparse, so include more wells (rough mean/std
                         # is fine — STD_FLOOR guards a tiny 2-point std). Provisional -> calibrate.
STD_FLOOR = 0.2
# NOTE: no min-WELL-count gate. The 1/d^2 weighting already handles proximity (a near well
# dominates -> weighted SD -> 0 -> good), so being on/near ONE good well is enough. The
# 'nearest well too far' case is handled upstream by the advisory's >40 km decline.
_Z_HI, _Z_LO = 0.5, 1.0            # SD < HI -> good ; < LO -> medium ; else poor
_FRESH_HI, _FRESH_LO = 120, 270    # days

_ORDER = {"good": 2, "medium": 1, "poor": 0}
_LEVEL = {2: "HIGH", 1: "MEDIUM", 0: "LOW"}
_LVL_ORDER = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}

# Public-facing SOFT labels. Internally the level stays HIGH/MEDIUM/LOW, but that is a spatial-
# CONSENSUS proxy (do the wells agree + is the data fresh), NOT a real prediction interval — so the
# OUTPUT must never claim "HIGH confidence" (over-claim / false alarm). The advisory relabels the
# surfaced field to these gentle words at the output boundary; the message uses phraser._CONF_SENTENCE.
SOFT_LABEL = {"HIGH": "well-supported", "MEDIUM": "mixed", "LOW": "thin"}


def cap_level(level, ceiling):
    """Return the WEAKER of `level` and `ceiling` (both HIGH/MEDIUM/LOW). Used to cap confidence
    when the normals came from the WRIS backup (less validated than the CSV snapshot)."""
    return level if _LVL_ORDER.get(level, 0) <= _LVL_ORDER.get(ceiling, 2) else ceiling


def _z_band(sd):
    if sd is None:
        return "poor"
    if sd < _Z_HI:
        return "good"
    if sd < _Z_LO:
        return "medium"
    return "poor"


def _freshness_band(age_days):
    if age_days is None or age_days <= _FRESH_HI:
        return "good"
    if age_days <= _FRESH_LO:
        return "medium"
    return "poor"


def _mean_std(vals):
    import numpy as np
    if len(vals) < MIN_YEARS:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def _seasonal_changes(series, mf, mt, years, gap_days):
    """That well's OWN change (depth(end)-depth(start)) for the mf->mt window, one per
    year it has both endpoints. mf==mt gives the year-over-year (y -> y+1) window."""
    out = []
    for y in years:
        a, b = _window(mf, mt, y)
        x = _depth_at(series, a, gap_days)
        z = _depth_at(series, b, gap_days)
        if x is not None and z is not None:
            out.append(z - x)
    return out


def _latest_yoy(series, anchor, gap_days, months_back=18):
    """Most recent observable year-over-year change (depth(t) - depth(t-1yr)), walking
    the endpoint back from the anchor up to months_back months, else None."""
    t = anchor
    for _ in range(months_back):
        a = _depth_at(series, t - relativedelta(years=1), gap_days)
        b = _depth_at(series, t, gap_days)
        if a is not None and b is not None:
            return b - a
        t = t - relativedelta(months=1)
    return None


def _weighted_sd(z_by_code, weights):
    """1/d^2-weighted population SD of the z's + the count of contributing wells."""
    codes = [c for c in z_by_code if z_by_code[c] is not None and weights.get(c)]
    if len(codes) < 2:
        return None, len(codes)
    ws = [weights[c] for c in codes]
    zs = [z_by_code[c] for c in codes]
    w_tot = sum(ws)
    zbar = sum(w * z for w, z in zip(ws, zs)) / w_tot
    var = sum(w * (z - zbar) ** 2 for w, z in zip(ws, zs)) / w_tot
    return var ** 0.5, len(codes)


def _axis_confidence(z_by_code, weights, freshness_band):
    """z-spread band, weakest-linked with the (already-resolved) freshness band. NO min-well gate:
    the 1/d^2 weighting already makes proximity work (a near well dominates -> weighted SD -> 0 ->
    good). A lone well has no spread to measure -> rely on it (treat as agreement). Zero wells -> LOW."""
    sd, n = _weighted_sd(z_by_code, weights)
    if n == 0:
        return {"level": "LOW", "z_sd": None, "n_wells": 0}
    sd = 0.0 if sd is None else sd                     # single well (n == 1) has zero spread by convention
    level = _LEVEL[min(_ORDER[_z_band(sd)], _ORDER[freshness_band])]
    return {"level": level, "z_sd": round(sd, 2), "n_wells": n}


def assess(nb, fc3_by_code, fc6_by_code, anchor, freshness_band,
           n_years=10, gap_days=45) -> dict:
    """Consensus confidence for the query.

    nb            : NeighbourSet (nb.codes / nb.series / nb.weights = 1/d^2).
    fc3/fc6_by_code: {station_code: per-well change_m} for 3m / 6m (model live OR seasonal analog).
    freshness_band: "good" | "medium" | "poor" — METHOD-based, resolved by the advisory
                    (good=fresh live, medium=analog, poor=climatology). Weakest-linked per axis.

    Returns {level, seasonal_3m, seasonal_6m, yoy, z_sd_3m/6m/yoy, freshness}.
    level (overall) = weakest of the axes the main call rests on = min(3m, yoy).
    """
    anchor = (datetime.strptime(anchor[:10], "%Y-%m-%d")
              if isinstance(anchor, str) else anchor)
    mf = anchor.month
    m3 = (anchor + relativedelta(months=3)).month
    m6 = (anchor + relativedelta(months=6)).month
    y0 = anchor.year
    seas_years = range(y0 - n_years, y0)
    yoy_years = range(y0 - 1 - n_years, y0 - 1)

    z3, z6, zy = {}, {}, {}
    for c in nb.codes:
        s = nb.series.get(c)
        if s is None or not len(s):
            continue
        m, sd = _mean_std(_seasonal_changes(s, mf, m3, seas_years, gap_days))   # 3m forecast
        v = fc3_by_code.get(c)
        if m is not None and v is not None:
            z3[c] = (v - m) / max(sd, STD_FLOOR)
        m, sd = _mean_std(_seasonal_changes(s, mf, m6, seas_years, gap_days))   # 6m forecast
        v = fc6_by_code.get(c)
        if m is not None and v is not None:
            z6[c] = (v - m) / max(sd, STD_FLOOR)
        m, sd = _mean_std(_seasonal_changes(s, mf, mf, yoy_years, gap_days))    # yoy observed
        v = _latest_yoy(s, anchor, gap_days)
        if m is not None and v is not None:
            zy[c] = (v - m) / max(sd, STD_FLOOR)

    w = nb.weights
    a3 = _axis_confidence(z3, w, freshness_band)
    ay = _axis_confidence(zy, w, freshness_band)
    _ = _axis_confidence(z6, w, freshness_band)   # 6m consensus computed but DELIBERATELY excluded
                                                  # from `level` (shakiest signal) and not surfaced.
    # overall = min(3m, yoy) — the axes the MAIN call rests on. The per-axis levels + z_sds are
    # computed internally but NOT surfaced; the single `level` is the headline.
    overall = _LEVEL[min(_LVL_ORDER[a3["level"]], _LVL_ORDER[ay["level"]])]

    return {"level": overall, "freshness": freshness_band}
