"""serve_advisory.py — thin /advisory endpoint (Phase A).

Builds ONE forecaster engine (+ an OPTIONAL 6m engine) and an AdvisoryEngine at startup,
then forwards (lat, lon, date) to advise(). It REUSES the existing forecaster (the same
core `/predict` and `python -m inference` call) — no forecasting logic here.

Run:
    GWL_RUN_DIR=/path/to/3m_pkg  [GWL_RUN_DIR_6M=/path/to/6m_run]  [GWL_LOCAL_CSV=... GWL_DEVICE=cuda]
        uvicorn advisory.serve_advisory:app --host 0.0.0.0 --port 8100
Env: GWL_RUN_DIR (3m, required), GWL_RUN_DIR_6M (optional), + the same GWL_* vars as inference.api.
"""
from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

_STATE: dict = {"adv": None, "error": None}

# Farmer-safe fallback text for an UNEXPECTED internal failure (a bug or an uncaught dependency error
# on an otherwise well-formed request). Known can't-forecast-here cases already degrade inside advise()
# to status="insufficient_data" with their own reason/message — this is only the last-resort net.
_SAFE_FAIL_MSG = ("We couldn't produce an advisory right now. Please try again shortly, or check with "
                  "your local agriculture office (KVK).")


def _build_engine(run_dir_env: str):
    """Build a TFTInferenceEngine from a run-dir env var, reusing inference.api._build_config
    (so all the other GWL_* knobs apply). Returns None if the env var is unset."""
    run_dir = os.environ.get(run_dir_env)
    if not run_dir:
        return None
    from inference.api import _build_config
    from inference.engine import TFTInferenceEngine

    prev = os.environ.get("GWL_RUN_DIR")
    os.environ["GWL_RUN_DIR"] = run_dir
    try:
        return TFTInferenceEngine(_build_config())
    finally:
        if prev is not None:
            os.environ["GWL_RUN_DIR"] = prev
        elif run_dir_env != "GWL_RUN_DIR":
            os.environ.pop("GWL_RUN_DIR", None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from advisory.advisory import AdvisoryEngine

        eng = _build_engine("GWL_RUN_DIR")
        if eng is None:
            raise RuntimeError("GWL_RUN_DIR is required (path to the 3m inference package / run dir)")
        eng6 = _build_engine("GWL_RUN_DIR_6M")     # optional
        # Phase-B SLM phraser (env-gated). GWL_ADVISORY_SLM=1 enables the Ollama Gemma phraser;
        # GWL_OLLAMA_HOST / GWL_OLLAMA_MODEL override the endpoint/model. Absent -> v0 template.
        slm = None
        if os.environ.get("GWL_ADVISORY_SLM", "").strip().lower() in ("1", "true", "yes", "on"):
            from advisory.slm import OllamaPhraser
            slm = OllamaPhraser(host=os.environ.get("GWL_OLLAMA_HOST"),
                                model=os.environ.get("GWL_OLLAMA_MODEL", "gemma3n:e2b"))
        # HYBRID SOURCES: the forecast uses the engine's live source (e.g. GWL_SOURCE=nwdp), but the
        # NORMALS (Stage 3, ~10 yr history) read from a historical CSV -- GWL_NORMALS_CSV (else the
        # legacy GWL_LOCAL_CSV). Absent -> normals fall back to the engine's source.
        normals_source = None
        normals_csv = os.environ.get("GWL_NORMALS_CSV") or os.environ.get("GWL_LOCAL_CSV")
        if normals_csv:
            from inference.acquire.local_csv_source import LocalCsvSource
            normals_source = LocalCsvSource(normals_csv, eng.data_config)
            _verify = os.environ.get("GWL_WRIS_VERIFY", "").strip().lower() in ("1", "true", "yes", "on")
            # WRIS-PRIMARY (experiment): flip the order — fetch each neighbour's ~10-yr NORMALS history
            # LIVE from WRIS first, fall back to the CSV snapshot only for a well WRIS can't serve. The
            # forecast is untouched. GWL_NORMALS_WRIS_PRIMARY=1 (needs the CSV for the backup). Takes
            # precedence over the CSV-primary WRIS backup below.
            if os.environ.get("GWL_NORMALS_WRIS_PRIMARY", "").strip().lower() in ("1", "true", "yes", "on"):
                from inference.acquire.wris_csv_fallback_source import WrisWithCsvFallbackSource
                normals_source = WrisWithCsvFallbackSource(normals_source, eng.data_config, verify=_verify)
            # Optional per-well BACKUP: for a neighbour the CSV genuinely lacks, fill its ~10-yr
            # history LIVE from WRIS (verify=False for the broken cert chain). CSV stays the default;
            # any query fully covered by the CSV is untouched. GWL_NORMALS_WRIS_FALLBACK=1 to enable.
            elif os.environ.get("GWL_NORMALS_WRIS_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on"):
                from inference.acquire.csv_wris_fallback_source import CsvWithWrisFallbackSource
                normals_source = CsvWithWrisFallbackSource(normals_source, eng.data_config, verify=_verify)
        _STATE["adv"] = AdvisoryEngine(eng, engine_6m=eng6, slm=slm, normals_source=normals_source)
    except Exception as e:  # noqa: BLE001
        _STATE["error"] = f"{type(e).__name__}: {e}"
        # FAIL-FAST: a load failure (bad/missing GWL_RUN_DIR / GWL_RUN_DIR_6M / GWL_PRITHVI_MODEL_DIR /
        # GWL_NORMALS_CSV path, unreadable checkpoint, bad config) must STOP the service — not bring it up
        # "loaded:false" and 503 every request. Re-raise so uvicorn aborts startup with a clear error.
        import sys
        print(f"[serve_advisory] STARTUP FAILED — engine/model did not load: {_STATE['error']}\n"
              f"  check the GWL_RUN_DIR / GWL_RUN_DIR_6M / GWL_PRITHVI_MODEL_DIR / "
              f"GWL_STATION_REGISTRY_CSV / GWL_NORMALS_CSV paths.", file=sys.stderr, flush=True)
        raise
    yield
    _STATE["adv"] = None


app = FastAPI(title="GWL Farmer Advisory", version="0.1", lifespan=lifespan)


class AdvisoryRequest(BaseModel):
    model_config = {"extra": "allow"}   # station_code / leave_one_out accepted, not advertised
    lat: Optional[float] = Field(None, description="query latitude")
    lon: Optional[float] = Field(None, description="query longitude")
    date: Optional[str] = Field(None, description="anchor date YYYY-MM-DD (default: today; must be today or earlier)")
    verbose: bool = Field(False, description="include debug detail (the normals the forecast is compared "
                                             "against, last-year context, confidence freshness) in the response")


@app.get("/health")
def health(response: Response):
    if _STATE["adv"] is None:
        response.status_code = 503
        return {"status": "error", "loaded": False, "error": _STATE["error"]}
    return {"status": "ok", "loaded": True, "has_6m": _STATE["adv"].engine_6m is not None}


@app.post("/advisory")
def advisory(req: AdvisoryRequest, response: Response):
    adv = _STATE["adv"]
    if adv is None:
        response.status_code = 503
        return {"status": "error", "error": {"code": "init_error", "message": _STATE["error"]}}
    station_code = getattr(req, "station_code", None)
    leave_one_out = bool(getattr(req, "leave_one_out", False))
    try:
        return adv.advise(lat=req.lat, lon=req.lon, date=req.date,
                          station_code=station_code, leave_one_out=leave_one_out,
                          verbose=bool(getattr(req, "verbose", False)))
    except Exception as e:  # noqa: BLE001 — NEVER leak a 500; always a graceful, structured reply
        traceback.print_exc()   # full traceback to the service log for ops; safe message to the caller
        return {"status": "error",
                "location": {"lat": req.lat, "lon": req.lon},
                "reason": f"internal error: {type(e).__name__}",
                "error": {"code": "internal_error", "message": str(e)[:300]},
                "crop_guidance": None,
                "message": _SAFE_FAIL_MSG}


# ---- OpenAI-compatible shim (Open WebUI) ---------------------------------------------
# Lets Open WebUI (or any OpenAI-API client) drive the advisory as a "chat model": it POSTs
# a chat message, we parse lat/lon/date out of it, run advise(), and return the bulleted
# message as the assistant reply. Purely additive: reuses the already-loaded _STATE["adv"],
# so ONE service now speaks /advisory AND /v1/chat/completions (no second model load).
import json as _json
import re as _re
import time as _time
from fastapi.responses import StreamingResponse

_MODEL_ID = os.environ.get("GWL_OPENAI_MODEL_ID", "gwl-advisory")
_NUM = _re.compile(r"(-?\d{1,3}\.\d+)")        # decimals -> first two = lat, lon
_DATE = _re.compile(r"(\d{4}-\d{2}-\d{2})")     # optional ISO date


def _parse_query(text):
    """Pull (lat, lon, date) from free chat text: 'advisory for 14.71, 75.78 on 2023-11-15'."""
    nums = _NUM.findall(text or "")
    m = _DATE.search(text or "")
    if len(nums) < 2:
        return None
    return float(nums[0]), float(nums[1]), (m.group(1) if m else None)


def _run_advisory_text(text):
    adv = _STATE["adv"]
    if adv is None:
        return "Advisory service is not ready (model still loading or misconfigured)."
    q = _parse_query(text)
    if q is None:
        return ("Please give a location as lat, lon (and an optional date), e.g.\n"
                "`advisory for 14.71, 75.78 on 2023-11-15`")
    lat, lon, date = q
    try:
        out = adv.advise(lat=lat, lon=lon, date=date)
        return out.get("message") or _json.dumps(out)
    except Exception:  # noqa: BLE001 — chat path also degrades to a safe message, never a 500
        traceback.print_exc()
        return _SAFE_FAIL_MSG


@app.get("/v1/models")                          # Open WebUI calls this to fill the model dropdown
def list_models():
    return {"object": "list",
            "data": [{"id": _MODEL_ID, "object": "model", "created": 0, "owned_by": "gwl"}]}


@app.post("/v1/chat/completions")               # ...and this to chat
def chat_completions(body: dict):
    msgs = body.get("messages") or []
    user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    content = _run_advisory_text(user if isinstance(user, str) else str(user))
    now = int(_time.time())
    cid = f"chatcmpl-{now}"
    if body.get("stream"):                       # Open WebUI streams by default
        def sse():
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": _MODEL_ID,
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                  "finish_reason": None}]}
            yield f"data: {_json.dumps(chunk)}\n\n"
            end = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": _MODEL_ID,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {_json.dumps(end)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    return {"id": cid, "object": "chat.completion", "created": now, "model": _MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
