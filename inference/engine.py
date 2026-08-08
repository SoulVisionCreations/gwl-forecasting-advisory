"""TFTInferenceEngine — the single reusable inference core.

Both the CLI (`python -m inference`) and a future FastAPI app call `predict()`.
Heavy objects (model, scaler, projector, registry, fetchers, station-stats) are
built ONCE in __init__: per-run for the CLI, once-at-startup for the API. No flow
is duplicated between CLI and API, and every stage delegates to a thin module that
reuses canonical training code (data_preparation / feature_scaler / tft_model /
prithvi_finetune).

Two modes, same path:
  - PRODUCTION  : predict(lat, lon[, date])                 — arbitrary point.
  - VALIDATION  : predict(station_code=…, leave_one_out=True) — derive that
                  station's lat/lon, exclude it from its own neighbours, and score
                  the interpolated prediction against its real GWL at target_date.
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings
from datetime import datetime
from typing import Optional

from dateutil.relativedelta import relativedelta

from inference.config import InferenceConfig
from inference.types import StationPrediction


def _load_data_config(run_dir: str):
    """Unpickle the training DataConfig (data_config_variables.pkl).

    It was pickled from a script running as __main__, so the class resolves to
    __main__.DataConfig; install the shim, then load. Gives train/inference parity
    for gap_days, horizon, num_timesteps, composite_period, interpolate_lookback_gwl…
    """
    from gwlcore import data_preparation as dp

    sys.modules["__main__"].DataConfig = dp.DataConfig  # unpickle shim
    path = os.path.join(run_dir, "data", "data_config_variables.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


class TFTInferenceEngine:

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg

        from inference.scale import Scaler
        from inference.model.tft_loader import TFTModelRunner
        from inference.prepare.station_stats import StationStats
        from inference.prepare.sample_builder import SampleBuilder
        from inference.acquire.station_registry import StationRegistry
        from inference.acquire.numeric_fetcher import NumericFetcher
        from inference.acquire.composite_fetcher import CompositeFetcher
        from inference.acquire import LiveFetcher
        from gwlcore.data_preparation import LSTMDataPreparation

        # ── single-folder package resolution ──
        # When an artifact isn't given explicitly, look for it as a sibling under
        # run_dir, so a self-contained package folder "just runs" with only run_dir:
        #   run_dir/best_model.pt, run_dir/data/{scalers,data_config,tile_manifest,
        #   station_stats}.pkl, run_dir/gwl_stations.csv, run_dir/prithvi_base/
        def _in_rd(*p):
            path = os.path.join(cfg.run_dir, *p)
            return path if os.path.exists(path) else None
        prithvi_dir = cfg.prithvi_model_dir or _in_rd("prithvi_base")
        # .tsv preferred (AIKosh Model asset rejects .csv); .csv still works (dev packages).
        registry_csv = cfg.station_registry_csv or _in_rd("gwl_stations.tsv") or _in_rd("gwl_stations.csv")
        stats_pkl = cfg.station_stats_pkl or _in_rd("data", "station_stats.pkl")

        # ── parity config + trained artifacts ──
        self.data_config = _load_data_config(cfg.run_dir)
        self.scaler = Scaler.load(os.path.join(cfg.run_dir, "data", "scalers.pkl"))
        self.model = TFTModelRunner.load(
            cfg.run_dir, device=cfg.device,
            model_dir=prithvi_dir,
            station_csv=cfg.station_index_csv or registry_csv,
        )
        self.use_prithvi = self.model.use_prithvi

        # Prediction clamp (train.py _clamp_pred_to_current / _effective_clamp_pct):
        # cap |delta| <= pct*|current_gwl|. Precedence: explicit cfg → ckpt config →
        # horizon default (0.20 for <=4mo, 0.50 otherwise).
        pct = cfg.pred_clamp_pct
        if pct is None:
            pct = self.model.cfg.get("pred_clamp_pct")
        if pct is None:
            pct = 0.20 if self.data_config.forecast_horizon_months <= 4 else 0.50
        self.clamp_pct = float(pct) if (pct and pct > 0) else 0.0
        # Positivity now rides on the delta clamp alone (the positivity floor was
        # removed Jul 3 2026 — decision: delta clamp only). With 0 < pct < 1 and
        # current >= 0, current+delta >= (1-pct)*current >= 0, so it can't go negative.
        # Outside that range it CAN → warn so a misconfig isn't silent.
        if not (0.0 < self.clamp_pct < 1.0):
            warnings.warn(
                f"pred_clamp_pct={self.clamp_pct} is outside (0,1): the delta clamp no "
                "longer guarantees GWL>=0 and the positivity floor was removed — "
                "predicted GWL may be negative.", RuntimeWarning, stacklevel=2)

        # ── station stats (parity-exact derived feats) + spatial registry ──
        if stats_pkl:
            self.station_stats = StationStats.load(stats_pkl)                    # small precomputed table
        else:
            stats_src = os.path.join(cfg.run_dir, "data", "train_samples.pkl")   # 1.5 GB fallback
            self.station_stats = StationStats.from_train_samples(stats_src)
        registry_csv = registry_csv or self.model.cfg.get("station_index_csv")   # else ckpt path
        self.registry = StationRegistry(registry_csv)

        # ── optional kriging variograms (FINAL-step alternative to IDW) ──
        # Independent of the prediction flow: loaded here only so the final
        # interpolate() can offer a kriging delta alongside IDW. Absent → IDW only.
        vario_path = cfg.kriging_variograms_path or _in_rd("data", "state_variograms.pkl")
        self.variograms = None
        if vario_path:
            from inference.kriging import StateVariograms
            self.variograms = StateVariograms.load(vario_path)

        # ── canonical sample builder + gap-GWL lookup (for validation truth) ──
        self.builder = SampleBuilder(
            self.data_config, self.station_stats,
            climatology_years=cfg.climatology_years, use_openmeteo=cfg.use_openmeteo,
        )
        self._dp = LSTMDataPreparation(self.data_config)

        # ── numeric source: local training CSV (reliable, parity) or live GEE+WRIS ──
        gee_project = cfg.gee_project
        gwl_provider = None
        if cfg.local_csv:
            from inference.acquire.local_csv_source import LocalCsvSource

            numeric = LocalCsvSource(cfg.local_csv, self.data_config)
        else:
            lookback_years = max(1, int(round(self.data_config.num_timesteps *
                                              getattr(self.data_config, "lookback_window_days", 90) / 365)) + 1)
            lookback_years = max(lookback_years, cfg.min_lookback_years)
            # GWL source: WRIS (default) or NWDP (swaps ONLY the GWL slice; GEE still
            # supplies dynamic/static/composite). Heavy lifting is all in the provider.
            _src = getattr(cfg, "gwl_source", "wris")
            if _src == "nwdp":
                from inference.acquire.nwdp_source import NwdpGwlProvider

                idx_path = cfg.nwdp_index_path or os.path.join(cfg.run_dir, "data", "nwdp_station_index.pkl")
                gwl_provider = NwdpGwlProvider(idx_path, data_config=self.data_config)
            elif _src == "wris":
                # Robust WRIS-by-station_code: verify=false for WRIS's broken TLS chain +
                # the depth-only datatype filter + canonical clean. Replaces the vendored
                # WRIS path (numeric_fetcher's gwl_provider=None branch), which lacked both
                # and is why NWDP was adopted as a stopgap. WRIS-empty wells return None ->
                # the seasonal-analog / normals layer supplies the CSV fallback.
                from inference.acquire.wris_gwl_provider import WrisGwlProvider

                _verify = os.environ.get("GWL_WRIS_VERIFY", "").strip().lower() in ("1", "true", "yes", "on")
                gwl_provider = WrisGwlProvider(data_config=self.data_config, verify=_verify)
            numeric = NumericFetcher(lookback_years=lookback_years, gee_project=gee_project,
                                     gwl_provider=gwl_provider, data_config=self.data_config)

        # ── composite tiles (only when use_prithvi; disk-cached, live GEE on miss) ──
        composite = None
        if self.use_prithvi:
            # Cache-dir precedence: explicit cfg → package-local run_dir/composites →
            # the training composite dir if it exists (e.g. big-10) → else create the
            # package-local dir for live GEE downloads. (The baked-in DataConfig path
            # may be unwritable off the training host.)
            cache_dir = cfg.composite_cache_dir
            if not cache_dir:
                rd_comp = os.path.join(cfg.run_dir, "composites")
                dc_comp = getattr(self.data_config, "composite_dir", None)
                if os.path.isdir(rd_comp):
                    cache_dir = rd_comp
                elif dc_comp and os.path.isdir(dc_comp):
                    cache_dir = dc_comp
                else:
                    cache_dir = rd_comp   # created on first live download
            composite = CompositeFetcher(
                cache_dir=cache_dir,
                composite_period=getattr(self.data_config, "composite_period", "quarter"),
                gee_project=gee_project,
            )
        self._numeric = numeric
        self._gwl_provider = gwl_provider   # for surfacing NWDP forward-fill notes
        self.fetcher = LiveFetcher(numeric, composite) if composite is not None else None

    # ------------------------------------------------------------------ predict
    def predict(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        date: Optional[str] = None,
        station_code: Optional[str] = None,
        leave_one_out: bool = False,
        details: bool = False,
    ) -> dict:
        """End-to-end inference. Never raises: returns a response dict whose
        `status` is ok | partial | error. Station-level failures drop that station
        (tagged with the stage it died at) and the request continues; request-level
        failures return status=error with a {code, message}."""
        from inference.postprocess import build_response
        from inference.types import InferenceError

        return self._predict_core(lat, lon, date, station_code, leave_one_out, details)[0]

    def predict_with_neighbours(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        date: Optional[str] = None,
        station_code: Optional[str] = None,
        leave_one_out: bool = False,
        details: bool = False,
    ):
        """Same as predict(), but ALSO returns the raw per-neighbour predictions in ONE
        pipeline run: (response_dict, preds). `preds` is a list of StationPrediction
        (each with .station_code, .distance_km, .predicted_delta, ...) — used by the
        advisory's consensus-confidence layer. `preds` is [] on any error / no survivors.
        No extra model calls: it is exactly predict()'s run, keeping the survivors."""
        return self._predict_core(lat, lon, date, station_code, leave_one_out, details)

    def _predict_core(self, lat, lon, date, station_code, leave_one_out, details):
        """Shared body of predict()/predict_with_neighbours(). Returns (response, preds)."""
        from inference.postprocess import build_response
        from inference.types import InferenceError

        dropped: "list[dict]" = []
        warnings: "list[str]" = []
        query: dict = {}
        try:
            lat, lon, current_date, target_date = self._resolve_query(lat, lon, date, station_code)
            query = {"lat": lat, "lon": lon, "current_date": current_date, "target_date": target_date}
            preds, interp, s1 = self._run(
                lat, lon, current_date, target_date, station_code, leave_one_out, dropped, warnings,
            )
        except InferenceError as e:
            return build_response("error", query, [], None, dropped, warnings,
                                  error={"code": e.code, "message": e.message}, details=details), []
        except Exception as e:  # unexpected → internal_error (never leak a traceback to the caller)
            return build_response("error", query, [], None, dropped, warnings,
                                  error={"code": "internal_error",
                                         "message": f"{type(e).__name__}: {str(e)[:200]}"}, details=details), []

        if len(preds) == 1:
            warnings.append("single-neighbour interpolation: low confidence")
        status = "ok" if not dropped else "partial"
        return build_response(status, query, preds, interp, dropped, warnings,
                              s1=s1, clamp_pct=self.clamp_pct, details=details), preds

    # ----------------------------------------------------------- staged pipeline
    def _resolve_query(self, lat, lon, date, station_code):
        """Validate input → (lat, lon, current_date, target_date). Raises InferenceError."""
        from inference.types import InferenceError

        try:
            current_date = self._resolve_date(date)
        except Exception:
            raise InferenceError("bad_request", f"unparseable date {date!r}; expected YYYY-MM-DD")

        # The anchor is the "current" date — GWL cannot exist for a future date, so a
        # future anchor would silently run on forward-filled/imputed inputs. Reject it:
        # to forecast ahead, pass the latest date you have data for (or omit for today);
        # the forecast is always anchor + forecast_horizon_months.
        now = datetime.now()
        today = datetime(now.year, now.month, now.day)
        if current_date > today:
            raise InferenceError(
                "bad_request",
                f"date {current_date.date()} is in the future; it must be today "
                f"({today.date()}) or earlier.")

        if lat is None or lon is None:
            if station_code is None:
                raise InferenceError("bad_request", "provide (lat, lon) or station_code")
            hit = self.registry.latlon_of(station_code)
            if hit is None:
                raise InferenceError("station_not_found",
                                     f"station_code {station_code!r} not in the station registry")
            lat, lon = hit
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            raise InferenceError("bad_request", f"invalid lat/lon: {lat!r}, {lon!r}")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise InferenceError("bad_request", f"lat/lon out of range: {lat}, {lon}")

        target_date = current_date + relativedelta(months=self.data_config.forecast_horizon_months)
        return lat, lon, current_date, target_date

    def _run(self, lat, lon, current_date, target_date, station_code, leave_one_out, dropped, warnings):
        from inference.interpolate import interpolate
        from inference.types import InferenceError

        # KNN — known stations near the query point (always uses the registry catalogue)
        exclude = station_code if (station_code and leave_one_out) else None
        neighbours = self.registry.k_nearest(lat, lon, self.cfg.k_neighbours, exclude=exclude)
        if not neighbours:
            raise InferenceError("no_neighbours", "no known stations near the query point")

        # Acquire — a total source outage (e.g. GEE batch / CSV unreadable) is request-level
        try:
            if self.fetcher is not None:
                acquired = self.fetcher.fetch(neighbours, current_date)
            else:
                acquired = self._numeric.fetch(neighbours, current_date)
        except Exception as e:
            raise InferenceError("data_source_unavailable",
                                 f"data source failed for all stations: {type(e).__name__}: {str(e)[:160]}")

        # Surface anchor forward-fill / staleness notes from whichever GWL source ran (NWDP provider or
        # the WRIS NumericFetcher), plus the LiveFetcher's imagery notes (e.g. GEE rate-limiting).
        for _src in (self._gwl_provider, self._numeric, self.fetcher):
            if _src is not None and hasattr(_src, "drain_notes"):
                warnings.extend(_src.drain_notes())

        # Drop neighbours with no usable GWL (stage=fetch); composite-missing is a soft warning
        usable = []
        for a in acquired:
            code = a.neighbour.station_code
            if a.gwl is None or not len(getattr(a.gwl, "index", [])):
                dropped.append({"station_code": code, "stage": "fetch",
                                "reason": "no GWL returned (api timeout / not found / empty)"})
                continue
            if self.use_prithvi and a.composite_path is None:
                warnings.append(f"composite missing for {code} (zero-filled)")
            usable.append(a)
        if not usable:
            # Wells WERE found nearby (KNN succeeded) but none returned a usable reading — describe the
            # DATA gap precisely (vs the coverage gap handled in the advisory when the nearest is too far).
            # Radius = the FARTHEST of the k returned (the circle enclosing all of them), not the nearest —
            # "N wells within X km" reads as "all N lie within X km". Format with one decimal so a sub-km
            # (or on-station) distance shows as e.g. "0.3 km" instead of collapsing to "0 km".
            _radius = neighbours[-1].distance_km if neighbours else None
            raise InferenceError(
                "data_source_unavailable",
                (f"{len(neighbours)} well(s) within {_radius:.1f} km, but none has recent water-level data "
                 f"(no live reading in the lookback window — likely inactive / stale wells)")
                if _radius is not None else "no station returned usable GWL data")

        # Build samples — canonical create_sample; per-station failures → stage=sample
        samples, ordered_paths, zero_idx, skipped = self.builder.build(usable, current_date)
        for s in skipped:
            dropped.append({"station_code": s["station_code"], "stage": "sample", "reason": s["reason"]})

        # Scale — per-sample isolation so one bad sample can't sink the batch (stage=scale)
        scaled, kept = [], []
        for samp in samples:
            try:
                scaled.append(self.scaler.transform_one(samp))
                kept.append(samp)
            except Exception as e:
                dropped.append({"station_code": samp.get("station_code"), "stage": "scale",
                                "reason": f"{type(e).__name__}: {str(e)[:100]}"})

        # Predict — batch, falling back to per-sample isolation on a batch error (stage=predict)
        acq_by_code = {a.neighbour.station_code: a for a in usable}
        preds = self._predict_safe(scaled, kept, ordered_paths, zero_idx, acq_by_code, current_date, dropped)

        # Survivor floor → request-level error
        floor = max(1, self.cfg.min_neighbours)
        if len(preds) < floor:
            code = "no_neighbours" if not preds else "insufficient_neighbours"
            raise InferenceError(
                code, f"only {len(preds)} usable neighbour(s) after data collection; need >= {floor}",
            )

        # Validation ground truth (soft — never blocks the prediction)
        s1 = None
        if station_code is not None:
            s1 = self._station_ground_truth(station_code, current_date, target_date)
            if s1 is None or s1.get("actual_gwl") is None:
                warnings.append(f"validation ground truth unavailable for {station_code}")

        interp = interpolate(lat, lon, preds, idw_power=self.cfg.idw_power,
                             variograms=self.variograms)
        if self.variograms is not None and interp.kriging_delta is None:
            warnings.append(interp.kriging_note)
        # Tag each dropped neighbour with its distance so the response can render the FULL
        # stable candidate set (used + outlier + no_data), not just the survivors — the same
        # geographically-nearest wells show on any date, only their status changes.
        _dist_by_code = {n.station_code: n.distance_km for n in neighbours}
        for _d in dropped:
            _d.setdefault("distance_km", _dist_by_code.get(_d["station_code"]))
        return preds, interp, s1

    def _resolve_date(self, date: Optional[str]) -> datetime:
        if date is None:
            now = datetime.now()
            return datetime(now.year, now.month, now.day)
        return datetime.strptime(date[:10], "%Y-%m-%d")

    def _predict_safe(self, scaled, kept, ordered_paths, zero_idx, acq_by_code, current_date, dropped):
        """Batch predict; if the batch raises, isolate each sample so only the
        genuinely-bad station(s) are dropped (stage=predict)."""
        if not scaled:
            return []
        try:
            out = self.model.predict(scaled, ordered_paths, zero_idx)
            deltas = self.scaler.inverse_delta(out["delta_scaled"])
            return [
                self._one_prediction(kept[i], float(deltas[i]), float(out["trend_prob"][i]),
                                     acq_by_code, current_date)
                for i in range(len(kept))
            ]
        except Exception:
            preds = []
            for i, samp in enumerate(kept):
                try:
                    out = self.model.predict([scaled[i]], ordered_paths, zero_idx)
                    delta = float(self.scaler.inverse_delta(out["delta_scaled"])[0])
                    preds.append(self._one_prediction(samp, delta, float(out["trend_prob"][0]),
                                                      acq_by_code, current_date))
                except Exception as e:
                    dropped.append({"station_code": samp.get("station_code"), "stage": "predict",
                                    "reason": f"{type(e).__name__}: {str(e)[:100]}"})
            return preds

    def _one_prediction(self, s, delta, trend_prob, acq_by_code, current_date) -> "StationPrediction":
        code = s["station_code"]
        current_gwl = float(s["current_gwl"])
        delta = float(delta)
        if self.clamp_pct > 0.0:                       # cap unrealistic deltas (parity w/ training eval)
            max_dev = self.clamp_pct * abs(current_gwl)
            delta = max(-max_dev, min(max_dev, delta))
        predicted_gwl = current_gwl + delta
        a = acq_by_code.get(code)
        nbr = a.neighbour if a is not None else None
        target_date = s["target_date"]
        # neighbour actual @ target_date (historical date → per-station score)
        actual_gwl = actual_delta = error = None
        if a is not None and a.gwl is not None and len(getattr(a.gwl, "index", [])):
            hit = self._dp.find_gwl_value_with_gap(a.gwl, target_date)
            if hit is not None:
                actual_gwl = float(hit[0])
                actual_delta = actual_gwl - current_gwl
                error = delta - actual_delta
        return StationPrediction(
            station_code=code,
            lat=nbr.lat if nbr else float("nan"),
            lon=nbr.lon if nbr else float("nan"),
            distance_km=nbr.distance_km if nbr else float("nan"),
            current_date=str(current_date)[:10],
            target_date=str(target_date)[:10],
            current_gwl=round(current_gwl, 4),
            predicted_delta=round(delta, 4),
            predicted_gwl=round(predicted_gwl, 4),
            trend="up" if delta >= 0 else "down",
            trend_prob=round(trend_prob, 4),
            data_age_days=(a.data_age_days if a is not None else None),
            actual_gwl=None if actual_gwl is None else round(actual_gwl, 4),
            actual_delta=None if actual_delta is None else round(actual_delta, 4),
            error=None if error is None else round(error, 4),
        )

    def _station_ground_truth(self, station_code, current_date, target_date) -> Optional[dict]:
        """Fetch the query station's own GWL and gap-look-up current + actual values."""
        # Resolve the station to a Neighbour (self @ 0 km) and fetch its GWL only.
        self_nbr = self.registry.get(station_code)
        if self_nbr is None:
            return None
        acquired = self._numeric.fetch([self_nbr], current_date)
        if not acquired or acquired[0].gwl is None:
            return {"station_code": station_code, "current_gwl": None, "actual_gwl": None}
        gwl = acquired[0].gwl
        cur = self._dp.find_gwl_value_with_gap(gwl, current_date)
        act = self._dp.find_gwl_value_with_gap(gwl, target_date)
        return {
            "station_code": station_code,
            "current_gwl": None if cur is None else float(cur[0]),
            "current_match_date": None if cur is None else str(cur[1])[:10],
            "actual_gwl": None if act is None else float(act[0]),
            "actual_match_date": None if act is None else str(act[1])[:10],
        }
