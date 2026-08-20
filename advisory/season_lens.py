"""season_lens.py — WET/DRY water regime from the SIGN of the LOCAL historical normal.

`normal_seasonal` is the ~10-year MEDIAN of the OBSERVED seasonal change (normals.py) —
never the forecast, so a wrong forecast can't flip the regime. Its sign classifies the
window here: < 0 = WET (recharge), > 0 = DRY (deplete). This absorbs the SW/NE monsoon
geography from the DATA (the same month is wet in one belt, dry in another).

DEADBAND: when |normal_seasonal| is within the local band (0.5xIQR), the history itself
is ambiguous (recharge ~ depletion, nets to ~0) -> 'transition' rather than a forced
wet/dry call. This fixes the near-zero flip (e.g. Kolar June, normal ~ +0.3).

The regime drives the message WORDING and the crop-type DURATION hint only. The crop
CALENDAR tag (kharif/rabi/zaid) is a separate, purely-descriptive month label — it does
NOT infer wetness (which would be north-centric and wrong in the NE-monsoon south).
See use_case_gwl_seasons.txt.
"""
from __future__ import annotations

from datetime import datetime


def season_of(normal_seasonal, band=None):
    """Water regime from the local historical seasonal normal, with a deadband.

    'wet' if normal < -band, 'dry' if normal > +band, else 'transition' (too close to
    zero to call). band None/0 -> legacy hard cut at 0 (wet/dry only)."""
    if normal_seasonal is None:
        return None
    b = band if (band is not None and band > 0) else 0.0
    if normal_seasonal < -b:
        return "wet"
    if normal_seasonal > b:
        return "dry"
    return "transition"


def sowing_window(anchor_date):
    """Indian cropping-CALENDAR tag from the sowing month (metadata ONLY — does NOT pick
    crops or infer wetness). kharif = Jun-Sep, rabi = Oct-Jan, zaid (summer) = Feb-May."""
    if anchor_date is None:
        return None
    m = (datetime.strptime(anchor_date[:10], "%Y-%m-%d")
         if isinstance(anchor_date, str) else anchor_date).month
    if 6 <= m <= 9:
        return "kharif"
    if m >= 10 or m <= 1:
        return "rabi"
    return "zaid"


def phrase_outlook(normal_seasonal, a_class, band=None):
    """Coming-window outlook (recharge vs deplete). PLACE-NEUTRAL wording: it never names a calendar
    season (monsoon/winter/dry-season), so it can't clash with the sowing_window tag in the NE-monsoon
    belt (where a rabi window can recharge and a kharif window can deplete). wet vs dry is still carried
    by 'recharge' (wet) vs 'hold up / deplete' (dry)."""
    season = season_of(normal_seasonal, band)
    if season == "transition":
        # near-break-even normal: talk about vs a typical year
        if a_class == "above":
            return "water is set to hold up better than a typical year here"
        if a_class == "below":
            return "water is set to draw down more than a typical year here"
        return "water looks about typical for this window"
    wet = season == "wet"
    if a_class == "above":
        return ("recharge is running above normal for this time of year here" if wet
                else "water is set to hold up better than normal for this time of year here")
    if a_class == "below":
        return ("recharge is running weaker than normal for this time of year here" if wet
                else "water is set to deplete faster than normal for this time of year here")
    return ("recharge is about normal for this time of year here" if wet
            else "depletion is about normal for this time of year here")
