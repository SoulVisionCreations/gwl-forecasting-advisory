# Setup

Install the package, get the Model asset, and authenticate the live data sources. One environment serves
the **forecaster** and the **advisory** (User A) and **training / fine‑tuning** (User B). For the farmer
advisory specifically, [`ADVISORY.md`](ADVISORY.md) has the complete end‑to‑end walkthrough.

---

## 1. Code + Python environment

**Get the code:** clone the public repository, then `cd` into it:
`git clone https://github.com/SoulVisionCreations/gwl-forecasting-advisory.git && cd gwl-forecasting-advisory`
*(No Git? Use GitHub's "Code → Download ZIP" on the repo.)*

Python **3.11** is recommended (the release was built and verified on 3.11.14). We pin dependencies in
`pyproject.toml` so published numbers reproduce exactly.

Using [`uv`](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv                       # creates .venv
uv pip install -e ".[api]"    # inference + FastAPI server
# or, for training / fine-tuning:
uv pip install -e ".[train]"  # adds mlflow, matplotlib, seaborn, tqdm
```

Using plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[api]"        # or ".[train]"
```

> **Why pinned?** `pyproject.toml` fixes `torch==2.10.0` and the rest of the numeric/model stack to the exact
> versions the model was trained with. Unpinned installs pull newer CUDA builds that introduce a benign but
> non‑bit‑exact ~1e‑4 drift. Ship and run with the pinned environment.

Extras:
- `.[api]` — FastAPI + uvicorn (the HTTP server).
- `.[train]` — MLflow, matplotlib, seaborn, tqdm (training + plots).

---

## 2. Get the Model asset

The trained model is published as the **Model** artefact on AIKosh (`gwl_model_3m6m_v1.zip`) and is **not**
committed to this repo. It is a **combined 3‑month + 6‑month bundle sharing one frozen Prithvi‑EO backbone**.
Download it (login‑gated) and unzip (it already contains a top‑level `gwl_model_v1/`):

```bash
mkdir -p ~/gwl_advisory && cd ~/gwl_advisory   # (optional) a base folder to keep artefacts together
unzip gwl_model_3m6m_v1.zip                     # -> ./gwl_model_v1/  (zip carries its own top-level dir)
```
> Using `unzip -d <folder>`? Create it FIRST — `unzip -d` does **not** make a missing directory:
> `mkdir -p <folder> && unzip gwl_model_3m6m_v1.zip -d <folder>`.

```
gwl_model_v1/
  prithvi_base/    # SHARED frozen Prithvi-EO-2.0-300M backbone (~1.3 GB), shipped ONCE
  model_3m/        # 3-month forecaster — slim ckpt (~23 MB) + config.json + gwl_stations.tsv + data/*.pkl
  model_6m/        # 6-month forecaster — same layout, its own scalers/station_stats
```

**Where the trained weights live:** each `best_model.pt` (~23 MB) holds the model's *trainable delta* —
the **Prithvi LoRA adapters** (rank-16 / α-32 on the encoder `qkv`, ~1.57M params) + the **1024→32
projector** + the **from-scratch TFT + embeddings** (~1.9M trainable in total). The big
`prithvi_base/Prithvi_EO_V2_300M_TL.pt` holds **only the frozen 330.7M Prithvi backbone** — shared by both
horizons and byte-identical to the public NASA-IBM weights (so it ships once). There is **no separate LoRA
file**: full model = frozen base (from `prithvi_base/`) + the delta (from each `best_model.pt`).

Each `best_model.pt` is **slim**: the frozen backbone is stripped and **rebuilt at load from `prithvi_base/`**
(nothing is fetched from Hugging Face). **Because the backbone is shared (not inside each model dir), you must
point the loader at it** with `--prithvi-model-dir` (CLI) / `GWL_PRITHVI_MODEL_DIR` (API):

```bash
# forecaster, single horizon (CLI)
python -m inference --run-dir gwl_model_v1/model_3m --prithvi-model-dir gwl_model_v1/prithvi_base --lat … --lon …
```

For the **farmer advisory** (runs 3m + 6m together, adds normals/rules/confidence + a Gemma message), see
[`ADVISORY.md`](ADVISORY.md). Full model contents: [`weights/README.md`](../weights/README.md).

---

## 3. Authenticate Google Earth Engine (live inputs)

Satellite composites and numerical/static features are fetched **live from Google Earth Engine** per request.
GEE access needs a **project id** (with Earth Engine enabled) *and* a login token.

**First (one‑time, in a browser):** sign in at <https://earthengine.google.com> → **Get Started** →
create/pick a **Google Cloud project** and **register it for Earth Engine** — **noncommercial/research** is
**free** (approval usually quick), **commercial** needs a **paid plan + billing**. This enables the
**"Earth Engine API"** on the project; note its **Project ID**. (Without a registered, API‑enabled project,
GEE calls fail with a permission/project error regardless of auth.)

**Then authenticate + set the project as default:**
```bash
earthengine authenticate                 # opens a browser; saves your token to ~/.config/earthengine/credentials
earthengine set_project <PROJECT_ID>     # set the default (same id as GWL_GEE_PROJECT / --gee-project)
```
`earthengine authenticate` only logs **you** in — it does not create the project or enable the API (that's
the step above). **Headless / server (no browser):** `earthengine authenticate --auth-mode=notebook` (paste
the token), or copy an already-authenticated `~/.config/earthengine/credentials` onto the server, or use a
**service account** (`ee.ServiceAccountCredentials`) registered for Earth Engine.

Pass your GEE Cloud project via `--gee-project` / `GWL_GEE_PROJECT` (this is the **only mandatory
credential**; NWDP/WRIS/Open-Meteo are public and need no key — see [`ADVISORY.md`](ADVISORY.md) → *Credentials at a glance*).

### Credentials & IDs via environment variables

The repo hardcodes **no** keys or project IDs — supply them via the environment:

| Variable | Needed for | Notes |
|---|---|---|
| `GWL_GEE_PROJECT` (or `GEE_PROJECT`) | GEE access | your Earth Engine Cloud project; else the credential's default project is used |
| `CORESTACK_API_KEY` | *optional* | only for the legacy CoreStack admin-lookup path; the default GEE static-feature path does not need it. To obtain a key: <https://core-stack.org/use-apis/> |
| `GWL_DB_PASSWORD` (+ `GWL_DB_HOST`/`GWL_DB_NAME`/`GWL_DB_USER`) | **training only** | Postgres source DB for data-prep (User B); not used at inference |

```bash
export GWL_GEE_PROJECT=your-gee-project-id
# export CORESTACK_API_KEY=...        # only if you use the CoreStack admin lookup
```

---

## 4. (Training only) Get the Dataset + composites

Plain inference does **not** need the dataset. For fine‑tuning / reproduction (User B):

1. Download the **Dataset** artefact (`gwl_data.csv`) into `data/` (see [`data/README.md`](../data/README.md)).
   (The **advisory** also needs this file — for the ~10‑yr normals — see [`ADVISORY.md`](ADVISORY.md).)
2. Regenerate the quarterly composite tiles from Earth Engine (they are not shipped):

```bash
python scripts/extract_gwl_stations.py                 # CSV -> gwl_stations.csv (code, lat, lon)
python -m gwlcore.download_gwl_quarterly_composites \
    --stations_csv gwl_stations.csv --project <gee> --years 2023 2024 --quarters Q1 Q2 Q3 Q4 \
    --output_dir quarterly_composites
```

Full training walkthrough: [`TRAINING.md`](TRAINING.md).

---

## 5. Smoke test

```bash
# a known station, leave-one-out (imagery fetched live from Earth Engine)
M=gwl_model_v1
python -m inference --run-dir $M/model_3m --prithvi-model-dir $M/prithvi_base \
    --station-code 095022077204101 --leave-one-out \
    --date 2025-01-30 --gwl-source wris --device cuda
# -> status=ok, change_m=0.7   (2 decimals)
```

If that returns a `change_m`, the environment, weights and Earth Engine auth are all working. Next:
[`INFERENCE.md`](INFERENCE.md) (forecaster) or [`ADVISORY.md`](ADVISORY.md) (farmer advisory).

### Browse the API in a browser (`/docs`)

Running the HTTP service instead of the CLI (the advisory — see [`ADVISORY.md`](ADVISORY.md) — or the
inference API)? Once it's up, open **`http://<host>:<port>/docs`** for the interactive **Swagger UI**:
expand an endpoint → **Try it out** → fill the body → **Execute** — no `curl` needed. `/redoc` is a
read-only reference and `/health` is the liveness check. `POST` endpoints like `/advisory` can't be
called from the address bar (a `?lat=…&lon=…` URL returns `405`) — use `/docs`. Remote host?
`ssh -N -L <port>:127.0.0.1:<port> <server>`, then browse `http://127.0.0.1:<port>/docs`.

---

## Notes

- **GPU vs CPU:** the default is `--device cpu` (CLI **and** API); pass `--device cuda` / `GWL_DEVICE=cuda` to use a
  GPU. The model itself is tiny at inference (~0.2s); latency is dominated by the live Earth Engine fetch
  (~1–3 min cold, seconds warm).
- **GWL source:** `--gwl-source wris` (default) or `nwdp`. WRIS may be temporarily unavailable; **NWDP** is a
  drop‑in live source. A `--local-csv` is available for offline/parity runs.
