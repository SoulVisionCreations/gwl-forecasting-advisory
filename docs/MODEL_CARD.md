# Model Card — GWL Forecasting (Prithvi-EO + TFT)

**Groundwater-level 3‑month change forecast for any point in India.**
Given a location (latitude/longitude) and an "as‑of" date, the model predicts the **change in
groundwater level over the next 3 months** — expressed as depth‑to‑water in metres below ground.

> Scope of this card: **architecture · data sources · usage · license & attribution · intended use.**

---

## Architecture

A from‑scratch **Temporal Fusion Transformer (TFT)** conditioned on a **frozen Prithvi‑EO‑2.0‑300M**
satellite‑imagery encoder, joint fine‑tuned end‑to‑end.

| Component | Detail |
|---|---|
| Imagery encoder | **Prithvi‑EO‑2.0‑300M** (NASA–IBM), **frozen**, adapted with **LoRA** (rank 16, α 32, on the attention `qkv` projections) |
| Projector | Linear **1024 → 32**, maps the Prithvi embedding into the TFT conditioning space |
| Temporal model | **TFT trained from scratch** over the well's history + engineered features |
| Normalisation | **RevIN** (reversible instance norm) on the GWL sequence |
| Output guard | **delta‑clamp**: the predicted change is bounded to `|Δ| ≤ pct · |current|` (default `pct = 0.6`) |
| Horizon | fixed **3 months** ahead of the anchor date |
| Parameters | **~1.9M trainable** (LoRA 1.57M + projector 0.03M + TFT 0.31M) over a **~330.7M frozen** base — **332,643,180 total** |

**Inputs** (per well):
- **GWL history** — a sequence of past depth‑to‑water readings with presence indicators over a lookback window.
- **Static attributes** — station properties (e.g. elevation, stream order, well/aquifer/lithology type) plus
  **training‑derived per‑station statistics** and a `gwl_anomaly` baseline.
- **Dynamic weather** — rainfall and temperature aggregates.
- **Forecast‑window conditioning** — seasonal **climatology**, overridden by a live **Open‑Meteo** forecast when the
  target window is near‑real‑time.
- **Satellite imagery** — a **Prithvi‑EO composite** at the location (6‑band HLS, 224×224, quarterly).

**Output:** a single scalar — the **3‑month change in depth‑to‑water (metres)**.
Sign convention: **`change < 0` = water table rises (recharge)**, **`change > 0` = water table falls**.

**Spatial interpolation:** for an arbitrary query point the engine runs the model on the **K nearest known wells**
and blends their predictions to the query location by **inverse‑distance weighting (IDW)** (kriging is available as an
alternative and reported alongside). See [`INFERENCE.md`](INFERENCE.md).

> Imagery ablation note: an imagery‑free TFT baseline performs comparably, i.e. the satellite encoder is roughly
> **neutral** on aggregate skill; the full Prithvi+TFT model is the published release.

### How the satellite imagery enters the forecaster

The imagery is consumed as a **static covariate**, not as a time series:

1. Each sample is matched to **one quarterly HLS composite tile** (6‑band, 224×224) at the well location.
2. The **frozen Prithvi‑EO‑2.0‑300M encoder** (adapted with LoRA on the attention `qkv`) encodes the tile; the
   **1024 → 32 projector** then compresses that embedding to a 32‑dim vector.
3. That vector is **concatenated onto the static‑feature block** (appended after elevation, stream order, the
   `gwl_anomaly` baseline, etc.).
4. As a static covariate it feeds the TFT's **four static context vectors**, which (a) condition the
   **per‑timestep variable selection**, (b) **initialise the LSTM encoder's hidden & cell states**, and
   (c) **enrich** the temporal features — so the satellite embedding conditions the *entire* temporal forecast
   rather than acting as a plain input channel.

The model enforces this wiring: `use_prithvi=True` requires `use_static_features=True` — the Prithvi vector is only
ever added inside the static block (see `gwlcore/tft_model.py`, `forward()` Step 0).

---

## Data sources

**Training**

| Source | Used for | Provider / License |
|---|---|---|
| Groundwater observations (`gwl_data.csv`, ~3.3M readings, 10,411 stations) | GWL history + targets | CGWB / India‑WRIS · **GODL‑India** |
| Quarterly HLS composites (Prithvi imagery tiles) | imagery encoder input | NASA/USGS **HLS** via Google Earth Engine |
| Rainfall / temperature (CHIRPS, ERA5) | dynamic weather features | via Google Earth Engine |
| Static geospatial layers | static features | via Google Earth Engine |

**Inference (fetched live per request, nothing pre‑stored):**

| Signal | Source |
|---|---|
| Current & historical GWL | India‑WRIS, or **NWDP** (`--gwl-source nwdp`), or a local CSV (parity) |
| Satellite composites, numerical & static features | Google Earth Engine |
| Forecast‑window weather | Open‑Meteo |

The model ships only the trained weights and small **lookups / training‑derived statistics** (scalers, per‑station
stats, variograms, station registry, tile manifest). It does **not** ship or require the raw observational datasets at
inference time. Full detail: [`DATA_SOURCES.md`](DATA_SOURCES.md).

---

## Usage

Install, download & unzip the Model asset (`gwl_model_3m6m_v1.zip` → `gwl_model_v1/`), authenticate Earth
Engine, then (the shared Prithvi backbone is passed via `--prithvi-model-dir` / `GWL_PRITHVI_MODEL_DIR`):

```bash
# CLI — production 3-month forecast at a point (use model_6m for 6-month)
python -m inference --run-dir gwl_model_v1/model_3m --prithvi-model-dir gwl_model_v1/prithvi_base \
    --lat 12.9716 --lon 77.5946 --date 2025-01-30 --gwl-source wris

# API — same engine over HTTP
GWL_RUN_DIR=gwl_model_v1/model_3m GWL_PRITHVI_MODEL_DIR=gwl_model_v1/prithvi_base \
    GWL_SOURCE=wris uvicorn inference.api:app --port 8000
curl -X POST :8000/predict -H 'Content-Type: application/json' \
     -d '{"lat":12.9716,"lon":77.5946,"date":"2025-01-30"}'
# -> {"status":"ok","location":{...},"as_of_date":"2025-01-30","forecast_date":"2025-04-30","change_m":-0.56}
```

The `date` is the **anchor** (default: today; a future date is rejected). The forecast is always **anchor + 3 months**.
Full CLI/API reference — request fields, response schema, validation mode — in [`INFERENCE.md`](INFERENCE.md).
To reproduce or fine‑tune the model, see [`TRAINING.md`](TRAINING.md).

---

## License & Attribution

| Artefact | License |
|---|---|
| Model weights & code | **Apache‑2.0** |
| Training dataset | **GODL‑India** (Government Open Data License – India) |

**Attribution:**
- **Prithvi‑EO‑2.0‑300M** foundation model — **NASA & IBM** (Apache‑2.0). The architecture code (`prithvi_mae.py`) is
  vendored with the weights.
- **HLS** (Harmonized Landsat Sentinel‑2) imagery — **NASA / USGS** and **Copernicus / ESA**.
- **Open‑Meteo** forecast weather — **CC‑BY 4.0**.
- **Groundwater observations** — **Central Ground Water Board (CGWB)** / India‑WRIS / NWDP, under **GODL‑India**.

Model outputs are **not** encumbered by Google Earth Engine terms (GEE is used only to fetch the public input layers).
See the repository `NOTICE` for the full attribution text.

---

## Intended use

**Intended for**
- **3‑month‑ahead groundwater‑level change** estimates at a point of interest, for **planning, monitoring and
  decision support** (water resource management, agriculture, research) across India.
- Screening / triage — flagging where the water table is likely to **rise (recharge)** or **fall (depletion)** over the
  coming season, and relative comparison across locations.

**Not intended for**
- A **gridded map product** — the model produces a **point forecast**; area estimates away from wells are spatial
  interpolations, not measurements.
- A substitute for **field measurement** or statutory/regulatory determinations.
- Horizons other than **3 months**, or **future anchor dates** (the "as‑of" date must be today or earlier).
- Regions or regimes outside the **Indian** training distribution.

Absolute forecast levels away from a well are **spatially interpolated** from neighbouring wells (no sensor exists at
an arbitrary query point) and should be read as estimates; the model's genuine output is the **change** (`change_m`).
