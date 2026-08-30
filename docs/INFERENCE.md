# Inference

Run a groundwater‑level forecast — via the CLI (`python -m inference`) or the FastAPI server
(`inference/api.py`). **Both use the exact same engine**, so results are identical.

Prerequisites: [`SETUP.md`](SETUP.md) (installed package, `weights/` downloaded, Earth Engine authenticated).

---

## Concept

- You give a **location** and an **anchor date** (`date`). The model forecasts the **change in groundwater level
  over the next 3 months** (`change_m`, metres of depth‑to‑water).
- The `date` is the **"as‑of"/current** date — **not** the target. The horizon is fixed at **anchor + 3 months**
  (`forecast_date`). To forecast further out you must wait for newer data; you cannot anchor in the future.
- **Sign:** `change_m < 0` → water table **rises** (recharge); `change_m > 0` → water table **falls**.
- For an arbitrary point the engine finds the **K nearest known wells**, runs the model on each, and blends them to
  your location by **inverse‑distance weighting (IDW)**.

---

## CLI

The Model artefact unzips to `gwl_model_v1/{model_3m, model_6m, prithvi_base}`. Pick the horizon you want as
`--run-dir` (`model_3m` or `model_6m`) and always pass the shared backbone via `--prithvi-model-dir`.

```bash
# production 3-month forecast at a point (lat/lon)
M=gwl_model_v1
python -m inference --run-dir $M/model_3m --prithvi-model-dir $M/prithvi_base \
    --lat 12.9716 --lon 77.5946 --date 2025-01-30 \
    --gwl-source nwdp --device cuda   # nwdp is the default; shown here explicitly
# 6-month: swap --run-dir $M/model_6m (same --prithvi-model-dir)
```

Key flags:

| Flag | Meaning |
|---|---|
| `--run-dir` | the horizon's Model folder (`gwl_model_v1/model_3m` or `.../model_6m`) — **required** |
| `--prithvi-model-dir` | shared Prithvi backbone (`gwl_model_v1/prithvi_base`) — **required** (slim ckpt rebuilds the base from here) |
| `--lat` `--lon` | query location (omit when using `--station-code`) |
| `--date` | anchor date `YYYY-MM-DD` (default: today; **must be today or earlier**) |
| `--gwl-source` | `nwdp` (default — WRIS API is unreliable) \| `wris` |
| `--nwdp-index` | NWDP well index (`gwl_model_v1/model_3m/data/nwdp_station_index.pkl`) when `--gwl-source nwdp` |
| `--local-csv` | use a local GWL CSV instead of a live source (offline / parity) |
| `--composite-cache-dir` | reuse cached imagery tiles. **Omit ⇒ imagery is fetched LIVE from Earth Engine** (known stations are fast when a cache is provided) |
| `--gee-project` | your Google Earth Engine Cloud project (if your account requires one) |
| `--k` | number of nearest neighbours (default 10) |
| `--min-neighbours` | minimum usable neighbours (default 1) |
| `--device` | `cpu` \| `cuda` |
| `--station-code` `--leave-one-out` `--details` | **validation mode** — see below |

---

## API server

```bash
M=gwl_model_v1
GWL_RUN_DIR=$M/model_3m GWL_PRITHVI_MODEL_DIR=$M/prithvi_base \
  GWL_SOURCE=nwdp \
  GWL_DEVICE=cuda \
  uvicorn inference.api:app --host 0.0.0.0 --port 8000
# (omit GWL_COMPOSITE_CACHE_DIR to fetch imagery live; set it to a folder to cache/reuse tiles)
```

Endpoints: `GET /health` · `GET /` · `POST /predict`.

```bash
curl -X POST :8000/predict -H 'Content-Type: application/json' \
     -d '{"lat":12.9716,"lon":77.5946,"date":"2025-01-30"}'
```

Every CLI flag has an environment‑variable equivalent: `GWL_RUN_DIR` (required), `GWL_SOURCE`, `GWL_NWDP_INDEX`,
`GWL_LOCAL_CSV`, `GWL_COMPOSITE_CACHE_DIR`, `GWL_DEVICE`, `GWL_K_NEIGHBOURS`, `GWL_MIN_NEIGHBOURS`,
`GWL_STATION_STATS`, `GWL_STATION_REGISTRY_CSV`, `GWL_GEE_PROJECT`, `GWL_IDW_POWER`, `GWL_PRED_CLAMP_PCT`, `GWL_KRIGING_VARIOGRAMS`,
`GWL_PRITHVI_MODEL_DIR`, `GWL_USE_OPENMETEO`, `GWL_CLIMATOLOGY_YEARS`.
Source precedence: `GWL_LOCAL_CSV` → else `GWL_SOURCE` (**`nwdp` default** | `wris`).

> **Live source defaults to NWDP.** The India‑WRIS API has recurring outages and is currently unreliable,
> so the **forecast** live GWL defaults to **NWDP** telemetry (CKAN — reachable, fresher, 6‑hourly) rather
> than WRIS. The **normals** (Stage 3, ~10‑yr history) read from the **CSV snapshot** (`GWL_NORMALS_CSV`,
> e.g. `gwl_data.csv`) — *not* WRIS — for the same reason; NWDP's telemetry history is too short (~2021→) to
> form a 10‑yr normal. To force the old behaviour set `GWL_SOURCE=wris` (and, for WRIS‑backed normals,
> `GWL_NORMALS_WRIS_PRIMARY=1`), but expect timeouts while WRIS is down.

### GWL anchoring & staleness

Each neighbour well's readings are sparse and end on different dates, so before the spatial blend every
well's series is brought onto the **one shared anchor** (the request `date`, which is **never moved**) by
`resolve_anchor_gwl` — the **same policy for WRIS and NWDP**. With `age = anchor − (newest reading on/before
the anchor)`:

| age | tier | behaviour |
|---|---|---|
| `age ≤ gap_days` (30) | **fresh** | series used as‑is |
| `gap_days < age ≤ 100` | **stale** | **forward‑fill** — carry the last real value flat up to the anchor (confidence is lowered by age) |
| `age > 100` (`max_staleness_days`) | **too stale** | **drop the well** → NWDP tries the next‑nearest station; a WRIS neighbour drops out of the IDW blend |
| no reading on/before the anchor | — | drop |

Within a well, gaps **between** real readings are **linearly interpolated** onto a daily grid
(`interpolate_lookback_gwl`, applied identically in training and inference). Each timestep then carries two
flags the model reads:

- **`is_present`** — *does a GWL value exist here?* `1` for a real, interpolated, **or** forward‑filled day;
  `0` for dates **before the first reading** (no backward extrapolation — the model treats these as *unknown*,
  not as "GWL = 0").
- **`is_real`** — *is that value an actual observation (`1`) or a synthetic fill (`0`)?* Bookkeeping only: the
  sample's `current`/`target` are always grounded on real observations; the input sequence may use fills.

The anchor's `current` value is the last real reading **carried forward** (LOCF), never interpolated toward a
later reading. If **every** nearby well is too stale (or none has live data), there is no usable current
reading and the advisory degrades to the **seasonal analog** (see [ADVISORY.md](ADVISORY.md)). Normals are
computed over the **same wells the forecast used**, so `forecast vs normal` is a like‑for‑like spatial blend.

### Request body

The **public** request is just three fields:

| Field | Type | Notes |
|---|---|---|
| `lat` | float | query latitude |
| `lon` | float | query longitude |
| `date` | string | anchor date `YYYY-MM-DD` (default today; must be today or earlier) |

### Response

**Success:**

```json
{
  "status": "ok",
  "location": {"lat": 12.9716, "lon": 77.5946},
  "as_of_date": "2025-01-30",
  "forecast_date": "2025-04-30",
  "change_m": -0.56
}
```

- `change_m` is **the** answer — the predicted 3‑month change in depth‑to‑water (negative = rises).
- `forecast_date` = `as_of_date` + 3 months.

**Error:**

```json
{"status": "error", "error": {"code": "bad_request", "message": "date 2027-03-08 is in the future; it must be today (2026-07-08) or earlier."}}
```

Error codes map to HTTP status: `bad_request` → 400, `station_not_found` → 404,
`no_neighbours` / `insufficient_neighbours` → 422, `data_source_unavailable` → 503, `internal_error` → 500.

> The API keeps the response clean: internal notes/warnings are omitted and status is either **`ok`** or
> **`error`**. (The CLI additionally prints warnings, e.g. forward‑fill/staleness notes, for debugging.)

---

## The anchor date & the future‑date guard

`date` is the **anchor** — the date you have "current" groundwater data for. A **future** anchor is rejected:

```bash
curl -X POST :8000/predict -d '{"lat":12.97,"lon":77.59,"date":"2027-03-08"}'
# -> 400  {"status":"error","error":{"code":"bad_request","message":"date 2027-03-08 is in the future; it must be today (…) or earlier."}}
```

This is deliberate: a future anchor has no real "current" reading, so the forecast would silently run on
forward‑filled/stale inputs. To look ahead, anchor on the **latest date you have data for** and read `forecast_date`
(= anchor + 3 months).

---

## Validation mode (advanced)

To score against a **known station** (rather than an arbitrary lat/lon), the engine accepts three additional,
**undocumented‑in‑schema** request fields (they still work; they're just kept out of the public body):

| Field | Default | Meaning |
|---|---|---|
| `station_code` | — | validate against this known station instead of `lat`/`lon` |
| `leave_one_out` | **`true`** | exclude that station from its own neighbour set (honest validation) |
| `details` | `false` | include the per‑well breakdown block |

```bash
# CLI
python -m inference --run-dir gwl_model_v1/model_3m --prithvi-model-dir gwl_model_v1/prithvi_base \
    --station-code 095022077204101 --leave-one-out --date 2025-01-30 --gwl-source nwdp --details
# API
curl -X POST :8000/predict -d '{"station_code":"095022077204101","date":"2025-01-30"}'
# -> {"status":"ok", …, "change_m":0.7}   (values rounded to 2 decimals)
```

With `details=true` the response also includes (absolute quantities are **spatially interpolated** — there is no
sensor at a query point):

```
forecast_gwl_m, current_gwl_m, n_wells_used, nearest_well_km, method,
forecast_gwl_m_raw, kriging_gwl_m / kriging_change_m / kriging_gwl_m_raw / kriging_note,
wells[], dropped[]
# and, when validating a known station:
validation_station, actual_gwl_m, forecast_error_m, kriging_error_m
```

---

## GWL data sources

| Source | Flag / env | Notes |
|---|---|---|
| **NWDP** | `--gwl-source nwdp` (**default**) | live CKAN telemetry; coord‑nearest well with a staleness guard + forward‑fill. Index auto‑resolves from `model_3m/data/nwdp_station_index.pkl` |
| **India‑WRIS** | `--gwl-source wris` | official CGWB portal; **recurring outages — currently unreliable**, hence not the default |
| **Local CSV** | `--local-csv <file>` | offline / parity; takes precedence when set |

Live sources publish about a quarter behind, so a well's latest reading is often stale. The engine forward‑fills
each neighbour's last reading up to the shared anchor (up to **200 days**; older → that neighbour is dropped) so all
neighbours are blended on one common "as‑of" level. Satellite, numerical and static features come from Earth Engine;
forecast‑window weather from Open‑Meteo. See [`DATA_SOURCES.md`](DATA_SOURCES.md).

---

## Performance

The model itself is ~0.2 s. End‑to‑end latency is dominated by the **live Earth Engine fetch**: ~1–3 minutes cold for
a new location, seconds when composites are cached. Use `--composite-cache-dir` to reuse tiles.
