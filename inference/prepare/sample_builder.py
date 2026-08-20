"""SampleBuilder — build canonical model-input samples from live-fetched data.

REUSES `data_preparation.LSTMDataPreparation.create_sample` /
`create_sequence_for_sample` / `compute_climatology` directly (they are already
single-station, DataFrame-driven — no bulk-DB coupling, no refactor needed). This
is the single source of truth shared with training; it replaces the forked
`Inference_LSTM/preprocess.py`.

Fetcher contract (build-step b): each AcquiredStation supplies
  - gwl     : DataFrame, DatetimeIndex, column 'gwl_value' (+ 'is_real','state',
              'district' if available) — same shape create_sample/find_gwl expect.
  - dynamic : DataFrame, DatetimeIndex, the dynamic feature columns the training
              config uses (rainfall, temp, et, runoff, ...).
  - static  : pd.Series with elevation, well_depth, latitude, longitude, well_type,
              aquifer_type, lithology, stream_order, aquifer_0_aquifer, litho_supergroup.

Inference-specific vs training:
  - forecast features come from CLIMATOLOGY (the future is unknown at inference),
    passed as external_forecast_values; create_sample then sets target_gwl=-9999.
  - derived per-station stats + gwl_anomaly are filled from StationStats (parity).
  - tile_idx is a per-request index into the fetched composites (not the training
    manifest); zero_idx for neighbours whose composite is missing.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference.types import AcquiredStation
    from inference.prepare.station_stats import StationStats


class SampleBuilder:

    def __init__(self, data_config, station_stats: "StationStats",
                 climatology_years: int = 1, use_openmeteo: bool = True):
        self.data_config = data_config
        self.station_stats = station_stats
        self.climatology_years = int(climatology_years)   # default 1 = previous year's season
        self.use_openmeteo = bool(use_openmeteo)
        # Reuse the canonical per-sample engine (cheap: __init__ just stores config).
        # FEATURE_AGG_FUNCTIONS is the SAME mapping create_sample uses — imported, not
        # copied, so the forecast/historical aggregations can't drift from training.
        from gwlcore.data_preparation import LSTMDataPreparation, FEATURE_AGG_FUNCTIONS
        self._dp = LSTMDataPreparation(data_config)
        self._feature_agg = FEATURE_AGG_FUNCTIONS

    def build(self, acquired: "list[AcquiredStation]", current_date: datetime):
        """-> (samples: list[dict], ordered_composite_paths: list[str], zero_idx: int,
                skipped: list[dict]).

        Per neighbour: climatology forecast -> canonical create_sample(is_training
        =False) -> StationStats.fill -> per-request tile_idx. The TileStore over
        `ordered_composite_paths` is built downstream (build-step c) and indexed by
        the tile_idx values set here.
        """
        # 1) Build the per-request ordered composite list (dedup) for tile_idx.
        ordered_paths: list[str] = []
        path_to_idx: dict[str, int] = {}
        for a in acquired:
            p = a.composite_path
            if p and p not in path_to_idx:
                path_to_idx[p] = len(ordered_paths)
                ordered_paths.append(p)
        zero_idx = len(ordered_paths)   # sentinel: projector emits zeros

        samples: list[dict] = []
        skipped: list[dict] = []
        for a in acquired:
            code = a.neighbour.station_code
            try:
                station_gwl = self._prepare_gwl(a.gwl, code)
                forecast = self._forecast(a.dynamic, current_date, a.neighbour)
                sample = self._dp.create_sample(
                    station_gwl=station_gwl,
                    station_dynamic=a.dynamic,
                    station_static=a.static,
                    station_code=code,
                    current_date=current_date,
                    is_training=False,
                    external_forecast_values=forecast,   # Open-Meteo forecast; climatology fallback
                )
                if sample is None:
                    skipped.append({"station_code": code, "reason": "create_sample_none"})
                    continue
                self.station_stats.fill(sample)                       # derived feats + gwl_anomaly (parity)
                sample["tile_idx"] = path_to_idx.get(a.composite_path, zero_idx)
                samples.append(sample)
            except Exception as e:  # noqa: BLE001 — per-neighbour isolation
                skipped.append({"station_code": code, "reason": f"{type(e).__name__}: {str(e)[:120]}"})
        return samples, ordered_paths, zero_idx, skipped

    def _prepare_gwl(self, gwl, code: str):
        """Densify the neighbour's GWL exactly as training does when
        interpolate_lookback_gwl=True (data_preparation.py:1968-1972): reuse
        build_interpolated_gwl_df so the lookback grid (gap_days=0, exact match,
        real_only=False) finds a daily-dense series with an `is_real` column.
        Operates on a COPY (a.gwl stays the raw real observations, used for actuals).
        No-op when the flag is off or the series is empty."""
        if gwl is None or not len(getattr(gwl, "index", [])):
            return gwl
        if not getattr(self.data_config, "interpolate_lookback_gwl", False):
            return gwl
        g = gwl.reset_index()
        if "date" not in g.columns:
            g = g.rename(columns={g.columns[0]: "date"})
        if "station_code" not in g.columns:
            g["station_code"] = code
        g = self._dp.build_interpolated_gwl_df(g)
        return g.set_index("date")

    def _forecast(self, station_dynamic, current_date: datetime, neighbour) -> dict:
        """Forecast-window features: climatology baseline, with rain+temp OVERRIDDEN
        by the Open-Meteo seasonal forecast when available (a real forecast ~ the
        'actual future' training mostly saw; climatology ~ its fallback regime)."""
        from dateutil.relativedelta import relativedelta

        forecast = self._climatology(station_dynamic, current_date)
        if self.use_openmeteo:
            target_date = current_date + relativedelta(
                months=self.data_config.forecast_horizon_months)
            if self._is_forecastable(target_date):
                from inference.acquire.forecast_fetcher import fetch_openmeteo_rain_temp
                om = fetch_openmeteo_rain_temp(neighbour.lat, neighbour.lon, current_date, target_date)
                for k, v in om.items():
                    if v is not None and v == v:   # skip None / NaN
                        forecast[k] = v
        return forecast

    @staticmethod
    def _is_forecastable(target_date: datetime) -> bool:
        """Open-Meteo seasonal is a real-time forecast; skip clearly-historical
        windows (validation on past dates) so we don't pay dead API calls."""
        try:
            from dateutil.relativedelta import relativedelta
            now = datetime.now()
            return target_date >= datetime(now.year, now.month, now.day) - relativedelta(months=1)
        except Exception:
            return True

    def _climatology(self, station_dynamic, current_date: datetime) -> dict:
        """Climatology over the horizon window. Reuses data_preparation.
        compute_climatology (years < current year, leakage-free). num_years defaults
        to 1 -> the previous year's same-season values (the lookback already covers
        that year); raise climatology_years for a multi-year average."""
        from dateutil.relativedelta import relativedelta

        horizon = self.data_config.forecast_horizon_months
        target_date = current_date + relativedelta(months=horizon)
        window_days = (target_date - current_date).days
        clim: dict = {}
        for feature, agg in self._feature_agg.items():
            if feature != "lulc" and feature not in getattr(station_dynamic, "columns", []):
                continue   # feature not fetched (e.g. dropped ndvi/sm) -> create_sequence ignores it
            clim[feature] = self._dp.compute_climatology(
                station_dynamic, target_date, window_days, feature, agg,
                reference_year=current_date.year, num_years=self.climatology_years,
            )
        return clim
