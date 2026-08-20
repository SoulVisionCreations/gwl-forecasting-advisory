"""resolve_anchor_gwl — bring a neighbour's GWL series up to the ONE shared anchor
date (= the query's current_date) so the spatial interpolation blends every neighbour
on a common temporal frame: same anchor, same target window, same as-of level.

Used by BOTH live GWL sources — the WRIS path (NumericFetcher) and the NWDP provider
— so they behave identically. This is INFERENCE-ONLY: training never forward-fills to
a synthetic anchor (its current_date is always a real reading). The anchor date is
NEVER moved; only each neighbour's tail is filled up to it, or the neighbour is
dropped when its newest reading is hopelessly stale.

Freshness tiers (age = anchor - newest reading on/before the anchor):
  age <= gap_days               fresh      -> series returned UNCHANGED; the canonical
                                              ±gap_days lookup in create_sample finds
                                              the reading. No note.
  gap_days < age <= max_stale    stale      -> forward-fill (LOCF) daily rows up to the
                                              anchor so `current_gwl` exists at the
                                              shared anchor. Returns a note.
  age > max_stale                too stale  -> (None, None): the caller drops the
                                              neighbour (a 5-month-old reading must not
                                              pollute the spatial blend).

Readings AFTER the anchor (WRIS fetches a +gap_days target buffer) are preserved
untouched — they feed the validation target lookup, never the forward-fill.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

DEFAULT_MAX_STALENESS_DAYS = 100


def resolve_anchor_gwl(gwl_df, current_date: "datetime", gap_days: int = 30,
                       max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS):
    """gwl_df: DataFrame with a 'date' column (or DatetimeIndex) + 'gwl_value'.

    Returns (gwl_df_or_None, fill_or_None) where fill = {"last_date": date,
    "age_days": int} when a stale forward-fill was applied (else None). The caller
    formats its own source-specific note from `fill`. Schema-preserving: synthetic
    fill rows replicate the last real row (so per-station columns like state/district
    survive) with `is_real` set to 0 when that column exists.
    """
    import pandas as pd

    if gwl_df is None or not len(gwl_df):
        return None, None

    has_date_col = "date" in getattr(gwl_df, "columns", [])
    d = gwl_df if has_date_col else gwl_df.reset_index().rename(
        columns={(gwl_df.index.name or "index"): "date"})
    anchor = pd.Timestamp(current_date).normalize()
    days = pd.to_datetime(d["date"]).dt.normalize()

    on_or_before = days[days <= anchor]
    if on_or_before.empty:
        return None, None                      # nothing at/behind the anchor -> unusable
    last_date = on_or_before.max()
    age = int((anchor - last_date).days)

    if age <= gap_days:
        return gwl_df, None                    # fresh -> unchanged (gap lookup finds it)
    if age > max_staleness_days:
        return None, None                      # TOO STALE -> drop this well. NWDP's loop then tries the
        #                                        next-nearest station; WRIS's neighbour drops out of the
        #                                        IDW blend. If EVERY well is too stale -> no usable live
        #                                        reading -> the advisory's seasonal-analog fallback.

    # stale-but-usable (gap_days < age <= max_staleness): forward-fill (LOCF) last_date+1 .. anchor.
    last_row = d.loc[days == last_date].iloc[-1]
    last_val = float(last_row["gwl_value"])
    fill_days = pd.date_range(last_date + pd.Timedelta(days=1), anchor, freq="D")
    if not len(fill_days):
        return gwl_df, None
    fill = pd.DataFrame([last_row.to_dict()] * len(fill_days))
    fill["date"] = fill_days.values
    fill["gwl_value"] = last_val
    if "is_real" in fill.columns:
        fill["is_real"] = 0
    out = pd.concat([d, fill], ignore_index=True).sort_values("date").reset_index(drop=True)
    if not has_date_col:
        out = out.set_index("date")
    return out, {"last_date": last_date.date(), "age_days": age}
