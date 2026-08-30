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

### How the LoRA weights load & adapt the encoder

`best_model.pt` ships the **trained** LoRA adapters (plus the projector and TFT head). The encoder is adapted with
**HuggingFace `peft`**, in two distinct steps at load time:

1. **Rebuild the structure** — `gwlcore/prithvi_finetune.py:build_projector` instantiates the frozen
   `PrithviMAE`, loads the pretrained base, then `peft.get_peft_model(model, LoraConfig(r=16, α=32,
   target_modules=[encoder.blocks.*.attn.qkv]))` **injects the LoRA slots** on every block's fused `qkv` and
   **rewires the forward**. LoRA is init so the delta is **zero** (`B=0`) → at this point the encoder is
   byte‑identical to the pristine base.
2. **Fill the weights** — `inference/model/tft_loader.py` then `model.load_state_dict(best_model.pt)` **pours the
   trained `A`/`B` values into those slots** (and the projector + TFT). Now the delta is the learned one.

**Adapted forward** at each attention block:
```
qkv(x) = W_frozen · x  +  (α / r) · B · (A · x)     # here (32/16)·B·(A·x)
```
`W_frozen` (never updated) is the pretrained weight; `B·A` is the low‑rank (rank‑16) learned delta.

**Checkpoint keys** — 24 encoder blocks × `lora_A` + `lora_B` = **48 tensors, 1,572,864 params**:
```
prithvi.encoder.base_model.model.encoder.blocks.{0..23}.attn.qkv.lora_A.default.weight   (16, 1024)  = (r, in)
prithvi.encoder.base_model.model.encoder.blocks.{0..23}.attn.qkv.lora_B.default.weight   (3072, 16)  = (out, r)
```
`out = 3 × 1024` because `qkv` is **fused** (one adapter covers q, k, v). Identical in `model_3m` and `model_6m`.
The slim bundle strips the frozen base (reconstructed from the Prithvi dir); the fat bundle bakes it in, so a
`strict` load reproduces every key (base + LoRA + projector + TFT).

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
    --lat 12.9716 --lon 77.5946 --date 2025-01-30 --gwl-source nwdp

# API — same engine over HTTP
GWL_RUN_DIR=gwl_model_v1/model_3m GWL_PRITHVI_MODEL_DIR=gwl_model_v1/prithvi_base \
    GWL_SOURCE=nwdp uvicorn inference.api:app --port 8000
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

### How to trust the advisory

The forecaster's skill is modest‑but‑real: **per‑well median R²_δ ≈ 0.25** on the *change* (0.275 at 6m), and its
**direction of change (rise vs fall) is ~86–88% correct on moves larger than 2 m, but only ~60% on sub‑0.5 m
moves.** So the **trustworthy unit is the *direction + tier + confidence*, not the exact number** — the advisory
turns the noisy value into a noise‑scaled, spatially‑blended, band‑binned, confidence‑gated call.

**Three signals accompany every advisory** — two calibrated *outlook* axes + a data‑quality label:
- **`outlook.water_trend`** — *will the water rise or fall?* (forecast vs **zero**). The model's **strongest**
  signal (~66% for tiny moves → ~88% for big ones) — what a farmer most needs: are you *gaining or losing* water.
- **`outlook.vs_normal`** — *is that better or worse than a normal year?* (forecast vs the seasonal normal).
  Calibrated on the test set: the above/below‑normal call is right **~7 in 10** when *moderate* and **~3 in 4**
  when *clear*; a *typical* (within‑band) call makes no above/below claim. (The 3‑way exact call — telling
  "normal" from "slightly above/below" — is only ~54%, so we deliberately do **not** headline it.) These %s are a
  **sign match** (does the *actual* land on the same side of normal as the forecast, not how close the value is) —
  which is why they hold at R²_δ ≈ 0.25; the grading (gap vs band, thresholds) is in
  [DECISION_LAYER.md → Reading the outlook](DECISION_LAYER.md#reading-the-outlook-levels-clarity-and-reliability).
- **`confidence.level`** (`well‑supported / mixed / thin`) — is the *data* solid? (a fresh reading × how well the
  neighbouring wells agree). Separate from whether the *call* is right.

| Situation | Trust the crop **tier**? |
|---|---|
| `well‑supported` + a *clear/moderate* `vs_normal` call + a sizeable rise/fall | **Yes** — act on the tier |
| `mixed`, or `vs_normal` is `typical` (within band) | **Direction only** — no crop switch |
| `thin` / seasonal analog / nearest well far / tiny <0.5 m move | **Indicative only** — verify locally (KVK) |

- A **long/perennial‑crop clearance** is the most conservative signal (needs *both* 3m and 6m at/above normal);
  respect a **refusal**.
- The tier is a **water‑budget lean, not a yield guarantee** — weigh rainfall, price, soil, pick the specific crop
  locally.

**For reference** — raw direction (rise/fall) accuracy by *predicted* move size (test set): `<0.5 m ≈66%`,
`0.5–2 m ≈82%`, `2–5 m ≈88%`, `5 m+ ≈87%` (3m; 6m similar). Reliability climbs steeply once the move exceeds
~0.5 m — which is why tiny sub‑0.5 m moves are reported as **normal**, not a strong call.

**Regional variation.** Skill is **not uniform** across India. On the test set the above/below‑normal call ranged
~**62–75%** by state and the per‑well median R²_δ swung from **+0.63 (Kerala)** to **−1.35 (Bihar)**. It is
strongest in parts of **Kerala, Telangana, Chhattisgarh** and weakest in parts of **Tamil Nadu, Karnataka and
Bihar** (where R²_δ can be **negative** — at or below a persistence baseline). Weight the advisory accordingly in
those regions; a runtime per‑region reliability flag is a candidate enhancement.
