# GWL Forecasting — Groundwater Level 3‑ & 6‑Month Forecast

From-scratch **Temporal Fusion Transformer (TFT)** conditioned on a **frozen Prithvi-EO-2.0-300M**
satellite-imagery encoder (LoRA fine-tuned) + a small projector. Forecasts the **3‑month (and 6‑month)
change in groundwater level** (depth-to-water, metres below ground) at any point in India, blending
nearby well predictions by inverse-distance weighting — with a farmer-facing crop/water **advisory** on top.

> Status: package under construction (migrated from the research repo, imports-only; inference parity verified).

The published **Model** is a combined **3‑month + 6‑month** bundle sharing one frozen Prithvi backbone
(`gwl_model_3m6m_v1.zip` → `gwl_model_v1/{model_3m, model_6m, prithvi_base}`).

> ▶ **Running the farmer advisory (the main use-case)?** Go straight to **[`docs/ADVISORY.md`](docs/ADVISORY.md)** —
> complete from-scratch setup: clone this repo + download the **Model + Dataset** from AIKosh, then run.

## Three entry points

| You want to… | Go to | Command |
|---|---|---|
| **Run a forecast** (User A) | [`docs/INFERENCE.md`](docs/INFERENCE.md) | `python -m inference --run-dir gwl_model_v1/model_3m --prithvi-model-dir gwl_model_v1/prithvi_base --lat 12.97 --lon 77.59` |
| **Run the farmer advisory** (User A) | [`docs/ADVISORY.md`](docs/ADVISORY.md) | `uvicorn advisory.serve_advisory:app` (3m + 6m + normals + Gemma) |
| **Reproduce / fine-tune** (User B) | [`docs/TRAINING.md`](docs/TRAINING.md) | `python training/train.py --help` |

All sit on top of [`gwlcore/`](gwlcore/) — the shared engine (data prep, scaling, model
architectures, Prithvi fine-tune, composite download).

## Layout

```
gwlcore/    shared engine imported by all entry points
inference/  User A — forecaster CLI (python -m inference) + FastAPI (inference/api.py)
advisory/   User A — farmer crop/water advisory service (advisory/serve_advisory.py)
training/   User B — train / fine-tune driver + eval tools
scripts/    build/index utilities (build_package, build_nwdp_index, build_variograms,
            extract_gwl_stations, eval_interpolation)
tests/      unit + parity tests
weights/    Model asset (downloaded from AIKosh; not committed — see weights/README.md)
data/       Dataset asset (downloaded from AIKosh; not committed — see data/README.md)
docs/       MODEL_CARD · SETUP · INFERENCE · ADVISORY · TRAINING · DATA_SOURCES
```

## Quickstart (advisory — the use-case)

```bash
# 1. code + env
git clone https://github.com/SoulVisionCreations/gwl-forecasting-advisory.git && cd gwl-forecasting-advisory
uv venv --python 3.11 && uv pip install -e ".[api]"       # pinned deps (torch 2.10.0) — see docs/SETUP.md
# 2. credentials + artefacts
earthengine authenticate                                  # per-user OAuth (live imagery/features via GEE)
earthengine set_project <your-project-id>                 # default project = same id as GWL_GEE_PROJECT (below)
#    Model:   aikosh.indiaai.gov.in/web/models/details/ground_water_level_forcasting_model.html -> "Download Model" -> gwl_model_3m6m_v1.zip
#    Dataset: aikosh.indiaai.gov.in/web/datasets/details/ground_water_level_all.html -> "Download Dataset" -> ground_water_level_all_v1.zip
#    put both zips in THIS folder ($PWD), then extract:
unzip gwl_model_3m6m_v1.zip                               # -> ./gwl_model_v1/{prithvi_base,model_3m,model_6m}
unzip ground_water_level_all_v1.zip                      # -> ./gwl_data.csv
# 3. run the advisory (3m + 6m + normals + confidence + message)
W=$PWD/gwl_model_v1
GWL_RUN_DIR=$W/model_3m GWL_RUN_DIR_6M=$W/model_6m GWL_PRITHVI_MODEL_DIR=$W/prithvi_base \
GWL_NORMALS_CSV=$PWD/gwl_data.csv GWL_DATA_CSV=$PWD/gwl_data.csv \
GWL_SOURCE=wris GWL_NORMALS_WRIS_PRIMARY=1 GWL_WRIS_VERIFY=false \
GWL_GEE_PROJECT=your-gee-project-id GWL_DEVICE=cuda \
uvicorn advisory.serve_advisory:app --host 0.0.0.0 --port 8100
# query it (curl):
curl -s -X POST :8100/advisory -H 'Content-Type: application/json' \
     -d '{"lat":12.97,"lon":77.59,"date":"2025-01-30"}'
# ...or in a BROWSER (no curl): open  http://localhost:8100/docs  — Swagger UI:
#   POST /advisory -> "Try it out" -> fill lat/lon/date -> Execute.  (/advisory is POST-only.)
```

**Full advisory walkthrough** (GEE headless auth, optional Gemma/Ollama phraser, CPU-only option, complete
env-var reference): **[`docs/ADVISORY.md`](docs/ADVISORY.md)**. Runtime data is fetched live per request: GWL
from NWDP/WRIS, satellite composites + features from Google Earth Engine, forecast weather from Open-Meteo.
A raw single-horizon forecast (no advisory) is in [`docs/INFERENCE.md`](docs/INFERENCE.md).

## License

Code & model weights: CC-BY-4.0. Training data: GODL-India. The Prithvi-EO base is Apache-2.0
(NASA-IBM); see `NOTICE`.
