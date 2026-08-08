"""NumericFetcher — live numeric inputs (GWL + dynamic + static) for neighbours.

Drives the VENDORED optimized fetcher (inference.acquire.vendor.new_data_fetcher)
with OUR StationRegistry neighbours — so neighbours are trained stations that have
composites + per-station stats — rather than the fetcher's own discovery. Reuses
its batched-GEE primitives (_make_points_fc, _batch_get_historical_features,
_batch_get_all_static) + WRIS (download_from_station, _parse_gwl_timeseries).

Config-agnostic: delivers EVERY dynamic column the fetcher produces (rainfall,
temp, et, runoff, ndvi, sm, lulc); canonical create_sample selects per training
config. Skips the fetcher's forecast fetch (we use canonical climatology) and its
discovery. composite_path is left None here — LiveFetcher fills it via CompositeFetcher.

NOTE (revisit for fidelity): when config.interpolate_lookback_gwl is True, the GWL
series should be densified via LSTMDataPreparation.build_interpolated_gwl_df before
sequence building; handled in SampleBuilder, not here.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from inference.types import Neighbour, AcquiredStation


class NumericFetcher:

    def __init__(self, lookback_years: int = 12, gee_project: Optional[str] = None,
                 gwl_provider=None, data_config=None):
        self.lookback_years = lookback_years
        self.gee_project = gee_project
        # Optional GWL-slice override (e.g. NwdpGwlProvider when WRIS is down). When
        # None, the WRIS path below runs (now with training-parity GWL cleaning). GEE
        # dynamic/static are ALWAYS fetched here regardless of the GWL source.
        self.gwl_provider = gwl_provider
        self.data_config = data_config   # for GWL abs()/caps parity + gap_days buffer
        self._df = None   # lazily-built vendored DataFetcher
        self._notes: "list[str]" = []    # anchor forward-fill / staleness notes (WRIS path)

    def drain_notes(self) -> "list[str]":
        """Anchor-resolution notes accumulated during fetch (dedup, then cleared).
        The engine surfaces these as response warnings — same mechanism NWDP uses."""
        n = list(dict.fromkeys(self._notes))
        self._notes = []
        return n

    def fetch(self, neighbours: "list[Neighbour]", current_date: "datetime") -> "list[AcquiredStation]":
        import pandas as pd

        from inference.types import AcquiredStation
        from inference.acquire.vendor import new_data_fetcher as ndf
        from inference.acquire.anchor import resolve_anchor_gwl
        from gwlcore.data_preparation import clean_gwl_values

        if self._df is None:
            self._df = ndf.DataFetcher()
        df = self._df

        stations = [{"station_code": n.station_code, "lat": n.lat, "lon": n.lon} for n in neighbours]
        start_date = (current_date - timedelta(days=365 * self.lookback_years)).strftime("%Y-%m-%d")
        end_date = current_date.strftime("%Y-%m-%d")
        gap_days = getattr(self.data_config, "gap_days", 30) or 30    # match training gap; was hardcoded 35
        horizon_m = getattr(self.data_config, "forecast_horizon_months", 3) or 3
        # Forward buffer to the +horizon target so PAST-date validation can score against
        # the real target reading (today/production: WRIS has no future data -> no-op).
        # (WRIS is down -> this extension is UNTESTED; mirrors the NWDP forward buffer.)
        wris_end = (current_date + timedelta(days=int(horizon_m) * 31 + gap_days + 5)).strftime("%Y-%m-%d")

        ndf._init_gee()
        points_fc = ndf._make_points_fc(stations)
        dynamic_results = ndf._batch_get_historical_features(stations, points_fc, start_date, end_date)
        static_results = ndf._batch_get_all_static(stations, points_fc)

        out: "list[AcquiredStation]" = []
        for i, n in enumerate(neighbours):
            scode = n.station_code
            if self.gwl_provider is None:
                raw = df.download_from_station(scode, start_date, wris_end)
                gwl_df = df._parse_gwl_timeseries(raw["timeseries"], raw["metadata"], scode)
                # Training-parity: the vendored WRIS parse omits abs()/caps that
                # load_gwl_readings applies — normalize identically (data_fetcher.py:1109).
                gwl_df = clean_gwl_values(gwl_df, self.data_config)
                # Shared anchor policy (identical to NWDP): fresh -> unchanged; stale ->
                # forward-fill (LOCF) to the anchor + note; too stale -> drop (gwl_df=None).
                gwl_df, fill = resolve_anchor_gwl(gwl_df, current_date, gap_days)
                if fill is not None:
                    self._notes.append(
                        f"WRIS: {scode} current GWL carried forward from {fill['last_date']} "
                        f"({fill['age_days']}d stale; live-source lag)")
                meta = raw["metadata"][0] if raw.get("metadata") else {}
                data_age_days = fill["age_days"] if fill else 0   # <=gap -> ~0; forward-filled -> its age
            else:  # GWL from the injected provider (e.g. NWDP); GEE dynamic/static unchanged
                gwl_df, meta = self.gwl_provider.gwl_and_meta(scode, n.lat, n.lon, current_date)
                data_age_days = meta.get("data_age_days")         # age of the well's latest REAL reading

            dyn = dynamic_results.get(i, pd.DataFrame())

            well_type = meta.get("well_type", meta.get("welltype", "no_data")) or "no_data"
            try:
                well_depth = float(meta.get("well_depth", meta.get("welldepth", meta.get("depth", 0))) or 0)
            except (TypeError, ValueError):
                well_depth = 0.0
            gee_static = static_results[i] if i < len(static_results) else {}
            static = {
                "station_code": scode,
                "latitude": n.lat,
                "longitude": n.lon,
                "well_type": well_type,
                "well_depth": well_depth,
                **gee_static,
            }

            # canonical methods expect a DatetimeIndex
            if gwl_df is not None and "date" in getattr(gwl_df, "columns", []):
                gwl_df = gwl_df.set_index("date")
            if not dyn.empty and "date" in dyn.columns:
                dyn = dyn.set_index("date")

            out.append(
                AcquiredStation(
                    neighbour=n,
                    gwl=gwl_df,
                    dynamic=dyn,
                    static=pd.Series(static),
                    composite_path=None,   # filled by LiveFetcher via CompositeFetcher
                    data_age_days=data_age_days,
                )
            )
        return out
