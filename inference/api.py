"""FastAPI wrapper over TFTInferenceEngine — a THIN HTTP adapter, no new logic.

Builds ONE TFTInferenceEngine at startup (from an env-configured InferenceConfig) and
forwards every request to engine.predict(). The engine already returns a structured,
JSON-serializable dict (status ok|partial|error + prediction/validation/dropped/warnings)
and never raises, so the endpoints only map that status / error-code onto an HTTP code.
No fetching, sampling, scaling, model or interpolation logic lives here — it all comes
from the inference package (same core the CLI `python -m inference` calls).

Run (single-folder package → only GWL_RUN_DIR needed):
    GWL_RUN_DIR=/path/to/inference_pkg \
    [GWL_LOCAL_CSV=... GWL_DEVICE=cuda GWL_COMPOSITE_CACHE_DIR=...] \
        uvicorn inference.api:app --host 0.0.0.0 --port 8000

    # live GWL from NWDP (WRIS-down stopgap; GEE still supplies dynamic/static/composite):
    GWL_RUN_DIR=... GWL_SOURCE=nwdp GWL_DEVICE=cuda uvicorn inference.api:app --port 8000

GWL source precedence: GWL_LOCAL_CSV (if set) wins → else GWL_SOURCE ("wris" default | "nwdp").

Config via env (mirrors InferenceConfig / the CLI flags):
    GWL_RUN_DIR (required), GWL_SOURCE (wris|nwdp), GWL_NWDP_INDEX, GWL_LOCAL_CSV,
    GWL_COMPOSITE_CACHE_DIR, GWL_DEVICE, GWL_K_NEIGHBOURS, GWL_MIN_NEIGHBOURS,
    GWL_STATION_STATS, GWL_STATION_REGISTRY_CSV, GWL_GEE_PROJECT, GWL_IDW_POWER,
    GWL_PRED_CLAMP_PCT, GWL_KRIGING_VARIOGRAMS, GWL_PRITHVI_MODEL_DIR,
    GWL_USE_OPENMETEO (default true), GWL_CLIMATOLOGY_YEARS.

Endpoints:
    GET  /health   → engine-loaded status
    POST /predict  → {lat, lon, date?}   (public production forecast — the only advertised body)

Hidden/internal request fields (accepted via extra="allow", NOT shown in the schema; kept for
our own validation use and described in the docs, not the API): station_code, leave_one_out
(DEFAULT True), details. The public response surfaces status/location/dates/change_m (+ `error`
on failure); internal notes/warnings are stripped at the API boundary, and a `partial` result
(some neighbour wells dropped) is presented as `ok` — the per-well drops stay in the details
block, all still available via the CLI. So the public `status` is effectively ok | error.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field   # BaseModel/Field are stable across pydantic v1 & v2

# HTTP status per engine InferenceError code (engine.predict returns status="error"
# with error.code; we map that here — the body is always the engine's response dict).
_CODE_HTTP = {
    "bad_request": 400,
    "station_not_found": 404,
    "no_neighbours": 422,
    "insufficient_neighbours": 422,
    "data_source_unavailable": 503,
    "internal_error": 500,
    "init_error": 503,
}


def _bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _as_bool(v, default: bool) -> bool:
    """Coerce a hidden extra-field value (bool or bool-ish string) to bool."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _build_config():
    """InferenceConfig from env vars — only sets fields that are present so the
    engine's defaults + single-folder auto-resolution still apply."""
    from inference.config import InferenceConfig

    run_dir = os.environ.get("GWL_RUN_DIR")
    if not run_dir:
        raise RuntimeError("GWL_RUN_DIR is required (path to the inference package / run dir)")

    kw: dict = {"run_dir": run_dir}
    str_fields = {
        "GWL_LOCAL_CSV": "local_csv",
        "GWL_SOURCE": "gwl_source",                  # "wris" (default) | "nwdp" (WRIS-down stopgap)
        "GWL_NWDP_INDEX": "nwdp_index_path",         # nwdp_station_index.pkl (else run_dir/data/…)
        "GWL_COMPOSITE_CACHE_DIR": "composite_cache_dir",
        "GWL_DEVICE": "device",
        "GWL_STATION_STATS": "station_stats_pkl",
        "GWL_STATION_REGISTRY_CSV": "station_registry_csv",
        "GWL_GEE_PROJECT": "gee_project",
        "GWL_KRIGING_VARIOGRAMS": "kriging_variograms_path",
        "GWL_PRITHVI_MODEL_DIR": "prithvi_model_dir",
    }
    for env, field in str_fields.items():
        v = os.environ.get(env)
        if v:
            kw[field] = v
    for env, field, cast in (
        ("GWL_K_NEIGHBOURS", "k_neighbours", int),
        ("GWL_MIN_NEIGHBOURS", "min_neighbours", int),
        ("GWL_IDW_POWER", "idw_power", float),
        ("GWL_PRED_CLAMP_PCT", "pred_clamp_pct", float),
        ("GWL_CLIMATOLOGY_YEARS", "climatology_years", int),
    ):
        v = os.environ.get(env)
        if v:
            kw[field] = cast(v)
    kw["use_openmeteo"] = _bool_env("GWL_USE_OPENMETEO", True)
    return InferenceConfig(**kw)


_STATE: dict = {"engine": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the engine ONCE (loads the 1.3 GB model + scalers + registry). On failure,
    # keep the process up so /health surfaces the error rather than crash-looping.
    try:
        from inference.engine import TFTInferenceEngine

        _STATE["engine"] = TFTInferenceEngine(_build_config())
    except Exception as e:  # noqa: BLE001
        _STATE["error"] = f"{type(e).__name__}: {e}"
    yield
    _STATE["engine"] = None


app = FastAPI(title="GWL Forecast (Prithvi+TFT) Inference", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Public request body — only lat/lon/date are advertised (production forecast).

    Hidden internal fields are still ACCEPTED (extra="allow") but kept OUT of the schema;
    they are for our own validation use and are described in the docs, not here:
      • station_code (str)   — validate against a known station instead of lat/lon
      • leave_one_out (bool) — DEFAULT True; exclude that station from its own neighbours
      • details (bool)       — DEFAULT False; add the per-well `details` block
    """
    model_config = {"extra": "allow"}   # accept the hidden fields above without listing them

    lat: Optional[float] = Field(None, description="query latitude")
    lon: Optional[float] = Field(None, description="query longitude")
    date: Optional[str] = Field(None, description="anchor/current date YYYY-MM-DD (default: today; must be today or earlier)")
    # The "provide (lat, lon) or station_code" check is enforced by the engine (bad_request → 400).


@app.get("/")
def root():
    return {"service": "gwl-forecast-prithvi-tft", "version": "1.0",
            "endpoints": ["/health", "/predict"], "docs": "/docs"}


@app.get("/health")
def health(response: Response):
    eng = _STATE["engine"]
    if eng is None:
        response.status_code = 503
        return {"status": "error", "model_loaded": False, "error": _STATE["error"]}
    return {"status": "ok", "model_loaded": True, "use_prithvi": eng.use_prithvi,
            "run_dir": eng.cfg.run_dir, "device": eng.cfg.device}


@app.post("/predict")
def predict(req: PredictRequest, response: Response):
    """Forward to TFTInferenceEngine.predict — the SAME core `python -m inference` runs."""
    eng = _STATE["engine"]
    if eng is None:
        response.status_code = 503
        return {"status": "error",
                "error": {"code": "init_error", "message": _STATE["error"] or "engine not loaded"}}
    # Hidden/internal fields (extra="allow", not in the advertised schema): validation
    # against a known station, with leave-one-out ON by default; details opt-in.
    station_code = getattr(req, "station_code", None)
    leave_one_out = _as_bool(getattr(req, "leave_one_out", True), True)
    details = _as_bool(getattr(req, "details", False), False)
    result = eng.predict(
        lat=req.lat, lon=req.lon, date=req.date,
        station_code=station_code, leave_one_out=leave_one_out,
        details=details,
    )
    # Hide internal notes/warnings from the public response (still produced for the CLI /
    # details block); errors are always surfaced via `error`.
    result.pop("warnings", None)
    result.pop("notes", None)
    # Public status is simply ok | error: anything that isn't an error is reported as `ok`
    # (this folds in `partial`, where some neighbour wells were dropped). Whether a station
    # was dropped is captured by the internal notes/warnings + details block, not the status.
    if result.get("status") == "error":
        code = (result.get("error") or {}).get("code", "internal_error")
        response.status_code = _CODE_HTTP.get(code, 500)
    else:
        result["status"] = "ok"
    return result
