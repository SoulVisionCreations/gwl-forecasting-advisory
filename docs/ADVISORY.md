# Advisory — Groundwater Crop/Water Advisory (setup & run)

The **advisory** is a deterministic decision layer on top of the GWL forecaster. Given a location
(`lat`, `lon`) and an as-of `date`, it returns a farmer-facing crop/water advisory: whether groundwater
is set to **rise (recharge)** or **fall (deplete)** over the next 3 **and** 6 months versus the *local
seasonal normal*, whether a **long-duration crop** is workable, a **confidence** level, and a short
plain-language message.

It runs BOTH a 3-month and a 6-month forecaster (each a TFT + frozen Prithvi-EO + LoRA), compares the
forecasts against ~10-year historical normals, applies a rule table, and (optionally) phrases the result
with a small language model. The advisory itself does not run the neural net — it consumes the forecasts.

> Sign convention (GWL = depth-to-water): `change < 0` = water table **rises** (recharge) · `change > 0` =
> water table **falls** (depletion).

---

## Dependencies

| Dependency | What it is | Where it comes from |
|---|---|---|
| **Model bundle** | `gwl_model_3m6m_v1.zip` → extracts to `gwl_model_v1/{prithvi_base, model_3m, model_6m}` (3m + 6m forecasters sharing one frozen Prithvi base) | AIKosh **Model** — [ground_water_level_forcasting_model](https://aikosh.indiaai.gov.in/web/models/details/ground_water_level_forcasting_model.html) → *Download Model* → `gwl_model_3m6m_v1.zip` |
| **Dataset** | `gwl_data.csv` — ~10-yr per-station GWL; the **fallback** for the seasonal / year-over-year **normals** (WRIS is primary — see live services), and the **station index** the WRIS source uses to locate wells | AIKosh **Dataset** — [ground_water_level_all](https://aikosh.indiaai.gov.in/web/datasets/details/ground_water_level_all.html) → *Download Dataset* → `ground_water_level_all_v1.zip` (unzips to `gwl_data.csv`) |
| **Gemma `gemma3:4b`** *(optional)* | phrases the farmer message; **skip it** to use a deterministic template (decision + numbers are identical) | Ollama (`ollama pull`) — not shipped, under the Gemma Terms of Use |
| **Live services** | **India-WRIS** (anchor GWL by station_code + ~10-yr normals history, WRIS-primary) · Google Earth Engine (satellite composites + numerical/static features) · Open-Meteo (forecast weather, automatic) · NWDP (optional anchor-GWL fallback) | fetched per request |

The Model bundle is self-contained: the frozen Prithvi base is shipped once in `prithvi_base/` and
reconstructed at load; each model's `data/` carries its own scalers + `station_stats`.

---

## Setup (Linux)

**1. Code + environment**

Get the code — **clone the public repository**, then set up the pinned environment:

```bash
git clone https://github.com/SoulVisionCreations/gwl-forecasting-advisory.git && cd gwl-forecasting-advisory
uv venv --python 3.11               # 3.11.14 is the verified build; a bare `uv venv` may pick a newer
                                    # python (3.13 works, but pin 3.11 to match the release)
uv pip install -e ".[api]"          # torch 2.10 + earthengine + openmeteo (base) + fastapi/uvicorn ([api])
```
(No Git? Use GitHub's **Code → Download ZIP** on the repo, unzip, and `cd` into the folder.)

**2. Google Earth Engine** (live imagery + numerical features) — **the only mandatory credential.**
Earth Engine access is tied to a **Google Cloud project registered for Earth Engine**. You need **two
things**: a **project id** (with Earth Engine enabled) and your **login token**. Do the project setup FIRST
— `earthengine authenticate` / `set_project` below **assume the project already exists**.

**2a. One‑time: register for Earth Engine & get a project id** *(browser; do this before the commands)*
1. Sign in at **<https://earthengine.google.com>** with a Google account → **"Get Started"**.
2. **Create or pick a Google Cloud project** when prompted (or create one at
   <https://console.cloud.google.com/projectcreate>) and note its **Project ID** (e.g. `my-gee-project`).
3. **Register that project for Earth Engine** and choose a use type:
   - **Noncommercial / research** — **free** (academic, nonprofit, personal/eval). Approval is usually
     instant or quick.
   - **Commercial** — needs a **paid** Earth Engine plan with **billing enabled** on the project.
4. The signup **enables the "Earth Engine API"** on the project. If you created the project yourself,
   confirm it's on at **Cloud console → APIs & Services → search "Earth Engine API" → Enable**.

   → You now have a **PROJECT_ID** with Earth Engine enabled. Use it in **2b** *and* as `GWL_GEE_PROJECT`
   in **Run**. Without a registered, API‑enabled project, the GEE calls fail with a *permission / project*
   error no matter how you authenticate.

**2b. Authenticate & set that project as default**
```bash
earthengine authenticate                 # opens a browser → sign in → saves your token to ~/.config/earthengine/credentials
earthengine set_project <PROJECT_ID>     # make it the default (SAME id you pass as GWL_GEE_PROJECT)
```
- `earthengine authenticate` only logs **you** in (identity + token) — it does **not** create a project or
  enable the API; that's **2a**.
- **Headless / server (no browser):** `earthengine authenticate --auth-mode=notebook` (paste the token),
  **or** copy an already-authenticated `~/.config/earthengine/credentials` to the server, **or** use a
  **service account** (`ee.ServiceAccountCredentials`) whose email is registered for Earth Engine.

> ### Credentials at a glance
> | Service | Auth needed? | How |
> |---|---|---|
> | **Google Earth Engine** | **YES (required)** | `earthengine authenticate` + `GWL_GEE_PROJECT` (above) |
> | **India-WRIS** (anchor GWL + normals, WRIS-primary) | no — public API | nothing (set `GWL_WRIS_VERIFY=false` for their broken cert) |
> | **NWDP** (optional anchor-GWL fallback) | no — public API | nothing; index ships in the bundle |
> | **Open-Meteo** (forecast weather) | no — free API | nothing |
> | **CoreStack** (`CORESTACK_API_KEY`) | **no — optional, not required** | leave unset; the advisory runs and reproduces the reference output without it (it only fills the query point's state/district display). If you *do* want it, request a key per <https://core-stack.org/use-apis/> |
> | **Ollama / Gemma** | no — local | `ollama pull gemma3:4b` (no account) |
> | **AIKosh** (Model + Dataset download) | yes — portal login | download the two artefacts from your AIKosh account |

**3. Gemma via Ollama** *(optional — the message phraser)*

```bash
curl -fsSL https://ollama.com/install.sh | sh   # install Ollama
ollama pull gemma3:4b                           # ~3.3 GB
ollama serve                                    # serves on 127.0.0.1:11434 (skip if already running)
```
> **Already running?** `ollama serve` printing `address already in use` is **harmless** — it just means
> the daemon is already up (check with `curl -s 127.0.0.1:11434/api/tags` or `ollama ps`). Ollama runs
> ONE server for ALL your models; the advisory selects gemma3:4b **by name** via `GWL_OLLAMA_MODEL`
> (below), so having other models pulled is fine.

Skip this to use the built-in template (`GWL_ADVISORY_SLM` unset); the advisory's decision and all
numbers are identical either way.

**4. Model + Dataset** — download the two AIKosh artefacts, then unzip both into **one folder**.

1. **Get the Model** — open
   **<https://aikosh.indiaai.gov.in/web/models/details/ground_water_level_forcasting_model.html>** and click
   **"Download Model"** → **`gwl_model_3m6m_v1.zip`** (unzips to a top-level `gwl_model_v1/`).
2. **Get the Dataset** — open
   **<https://aikosh.indiaai.gov.in/web/datasets/details/ground_water_level_all.html>** and click
   **"Download Dataset"** → **`ground_water_level_all_v1.zip`** (unzips to a single `gwl_data.csv`, ~760 MB).
3. **Unzip both** into one folder. Run the `unzip` from *inside* that folder — each zip carries its own
   top-level name, so they land cleanly (no `-d` needed):

```bash
mkdir -p ~/gwl_advisory && cd ~/gwl_advisory     # a base folder to hold both artefacts
# move the two downloaded zips into this folder, then extract:
unzip gwl_model_3m6m_v1.zip           # -> ~/gwl_advisory/gwl_model_v1/{prithvi_base, model_3m, model_6m}
unzip ground_water_level_all_v1.zip   # -> ~/gwl_advisory/gwl_data.csv
ls gwl_model_v1 && ls -l gwl_data.csv # sanity check
```
> **Extracting with `unzip -d <folder>` instead?** Create the folder FIRST — `unzip -d` does **not** make a
> missing directory: `mkdir -p <folder> && unzip gwl_model_3m6m_v1.zip -d <folder>`.

You now have the two absolute paths used in **Run** below:
`~/gwl_advisory/gwl_model_v1` (the Model bundle) and `~/gwl_advisory/gwl_data.csv` (the Dataset).

---

## Run

```bash
W=~/gwl_advisory/gwl_model_v1        # the unzipped Model bundle (Setup step 4)
DATA=~/gwl_advisory/gwl_data.csv     # the unzipped Dataset      (Setup step 4)

GWL_RUN_DIR=$W/model_3m   GWL_RUN_DIR_6M=$W/model_6m \
GWL_PRITHVI_MODEL_DIR=$W/prithvi_base \
GWL_NWDP_INDEX=$W/model_3m/data/nwdp_station_index.pkl \
GWL_STATION_REGISTRY_CSV=$W/model_3m/gwl_stations.tsv \
GWL_NORMALS_CSV=$DATA   GWL_DATA_CSV=$DATA \
GWL_NORMALS_WRIS_PRIMARY=1   GWL_WRIS_VERIFY=false \
GWL_SOURCE=wris   GWL_GEE_PROJECT=<your-gee-project> \
GWL_COMPOSITE_CACHE_DIR=~/gwl_advisory/composites \
GWL_ADVISORY_SLM=1   GWL_OLLAMA_HOST=127.0.0.1:11434   GWL_OLLAMA_MODEL=gemma3:4b \
GWL_DEVICE=cuda \
uvicorn advisory.serve_advisory:app --host 0.0.0.0 --port 8100
```

### Environment variables

| Variable | Required? | Meaning |
|---|---|---|
| `GWL_RUN_DIR` | yes | 3-month forecaster dir (`model_3m`) |
| `GWL_RUN_DIR_6M` | recommended | 6-month forecaster dir (`model_6m`) — enables the long-crop gate |
| `GWL_PRITHVI_MODEL_DIR` | yes | shared Prithvi base dir (`prithvi_base`) — the slim checkpoints reconstruct the base from here |
| `GWL_SOURCE` | — | live anchor GWL source: **`wris`** (default — queries each well by its own `station_code`, `verify=false` + depth-datatype filter + clean; training-parity, no neighbour substitution) or `nwdp` (fallback stopgap — maps to the nearest telemetry well by coordinates) |
| `GWL_NWDP_INDEX` | with `nwdp` only | NWDP well index (ships inside `model_3m/data/`); unused when `GWL_SOURCE=wris` |
| `GWL_STATION_REGISTRY_CSV` | — | station registry (`model_3m/gwl_stations.tsv`) |
| `GWL_NORMALS_CSV` | yes | the Dataset CSV — source for the ~10-yr **normals** |
| **`GWL_DATA_CSV`** | **yes, with either WRIS mode** | the **same file** as `GWL_NORMALS_CSV`, in a second role — the WRIS source reads it to build a station index (code→lat/lon) to **locate wells**. In code this is the internal `STATION_CSV_PATH` (= `GWL_STATION_CSV` or `GWL_DATA_CSV`; no default after path-hardening) — you set `GWL_DATA_CSV`, never `STATION_CSV_PATH`. **Required whenever `GWL_NORMALS_WRIS_PRIMARY=1` or `GWL_NORMALS_WRIS_FALLBACK=1`** |
| `GWL_NORMALS_WRIS_PRIMARY` | **recommended `1` (deployed default)** | **WRIS-primary normals**: fetch each neighbour's ~10-yr history LIVE from WRIS *first*, per well; the shipped CSV is the fallback (used when WRIS lacks the well, returns too few readings, or a materially thinner series than the CSV — never downgrades a solid CSV well). **Fail-fast**: a WRIS outage trips a per-request circuit breaker and the request degrades to *exactly* the CSV path. Takes **precedence** over `GWL_NORMALS_WRIS_FALLBACK`. Needs `GWL_DATA_CSV`; `GWL_WRIS_VERIFY` (default `false`) skips WRIS's broken TLS. The forecast is unaffected (still `GWL_SOURCE`) |
| `GWL_NORMALS_WRIS_FALLBACK` | alternative `1` | CSV-**primary** with a WRIS back-fill: keep the CSV snapshot as the normals default and only fill a well the CSV **lacks** from WRIS (caps confidence). **Ignored if `GWL_NORMALS_WRIS_PRIMARY=1` is also set.** Its **partner is `GWL_DATA_CSV`**. `GWL_WRIS_VERIFY` (default `false`) = skip WRIS TLS verify (their cert chain is broken) |
| `GWL_GEE_PROJECT` | yes | Earth Engine project (live imagery + features) |
| `GWL_ADVISORY_SLM` | — | `1` → Gemma phrasing; unset → deterministic template |
| `GWL_OLLAMA_HOST` / `GWL_OLLAMA_MODEL` | with SLM | Ollama endpoint (`127.0.0.1:11434`) / model tag (`gemma3:4b`) |
| `GWL_DEVICE` | — | `cuda` or `cpu` |
| `GWL_COMPOSITE_CACHE_DIR` | **recommended** | one folder to cache the Prithvi imagery tiles. **Set it** — the advisory runs **both** the 3m and 6m forecasters, and they need the **same** tiles (keyed by station·year‑1·quarter, no horizon dependency). A **shared** cache means each tile is downloaded from Earth Engine **once** and reused by the other forecaster (and stays warm across later requests: cold ⇒ warm = minutes ⇒ seconds). **Unset ⇒** each forecaster falls back to its **own** `<run_dir>/composites`, so the same ~10 tiles are fetched **twice** per cold request. Tiles are written atomically (temp‑file + rename); the two forecasters run sequentially, so a shared dir has **no race** (the 6m just reads the 3m's freshly‑downloaded tiles) |
| `GWL_COMPOSITE_WORKERS` | optional (default `6`) | how many neighbour imagery tiles to download **concurrently** from Earth Engine per request. **Rate limits are auto-handled + visible:** each tile retries with exponential backoff, a GEE `429`/quota error is **logged as a `WARNING`** in the server output, and any tile the concurrent burst loses is **retried serially** within the same request (serializing clears the burst). Only lower this — or set **`1` = fully serial** — if that WARNING keeps recurring |
| `GWL_STATION_STATS` | **leave unset** | leave UNSET so each engine self-resolves its own `data/station_stats.pkl` (3m→3m's, 6m→6m's). Setting it forces both engines onto one table — the wrong wiring for the 6m |

> **`GWL_NORMALS_CSV` vs `GWL_DATA_CSV`** — both point at the **same** file (`gwl_data.csv`) but play
> two roles: `GWL_NORMALS_CSV` supplies the ~10‑yr **normals** (the CSV is the fallback under WRIS-primary,
> or the default under WRIS-fallback); `GWL_DATA_CSV` supplies the **station index** the WRIS source uses to
> locate wells. They are **partners, not alternatives** — with either WRIS mode on, set **both**.

---

## API

- `GET /health` → `{status, loaded, has_6m}`
- `POST /advisory` `{lat, lon, date, verbose?}` →
  ```json
  {
    "status": "ok",
    "location": {"lat": ..., "lon": ...},
    "anchor_date": "...", "forecast_date": "...",
    "numbers": {"forecast_change_m": <3m>, "forecast_change_m_6m": <6m>,
                "normal_seasonal_change_m": ..., "normal_seasonal_change_m_6m": ...,
                "latest_yoy_change_m": ..., "normal_yoy_change_m": ...},
    "regime": {"a": "above|normal|below", "b": "...", "a_6m": "..."},
    "long_crop": {"allowed": true, "reason": "..."},
    "confidence": {"level": "well-supported|mixed|thin", "freshness": "..."},
    "crop_guidance": {"water_need": "...", "duration_hint": "...", "long_crop_ok": true,
                      "type_examples": [...], "rule": "..."},
    "message": "...",
    "warnings": ["imagery: N satellite tile(s) hit an Earth Engine rate limit ..."]  // ONLY if imagery was rate-limited
  }
  ```
  `verbose=true` adds the internal normals / band / z-spreads.

  > **Checking for issues.** The `message` is the *farmer-facing* summary; for programmatic reliability
  > checks use **`status`** (`ok` vs `insufficient_data`/`error`) and **`confidence`** (`level` +
  > `freshness`). A **`warnings`** array appears **only when the satellite imagery was rate-limited by
  > Earth Engine** (and retried) — a clean run omits it. (Other internal data notes, e.g. stale readings
  > carried forward, are kept out of the response to keep it digestible.)
  **Graceful degradation:** for a future date, no/too‑far wells, or missing data the response is
  **HTTP 200 with `status: "insufficient_data"`** — a `reason` (e.g. *"date … is in the future"* or
  *"nearest well … km"*) plus a safe farmer `message`; `numbers`/`regime`/`crop_guidance` are null.
  It never hard‑errors on the query. `status: "error"` covers two cases: the service isn't loaded
  (HTTP 503, `error.code=init_error`); or an **unexpected internal failure** on a well‑formed request —
  caught and returned at **HTTP 200** with `error.code=internal_error` and a safe farmer `message`, so
  a bug or a transient dependency error never surfaces as a raw 500. The traceback is logged
  server‑side for ops.
- `GET /v1/models` · `POST /v1/chat/completions` — OpenAI-compatible shim (for Open WebUI).

```bash
curl -s -X POST :8100/advisory -H 'Content-Type: application/json' \
     -d '{"lat":12.9716,"lon":77.5946,"date":"2025-01-30","verbose":true}'
```

**Example response** — a real run for the request above (Bengaluru, anchor `2025-01-30`). This uses a
**historical** anchor, so it's reproducible; a *today* anchor (omit `date`) will of course differ, and
live values shift as new well readings land:

```json
{
  "status": "ok",
  "location": {"lat": 12.9716, "lon": 77.5946},
  "anchor_date": "2025-01-30", "forecast_date": "2025-04-30",
  "numbers": {"forecast_change_m": 3.49, "forecast_change_m_6m": 1.35,
              "normal_seasonal_change_m": 1.14, "normal_seasonal_change_m_6m": 0.32,
              "latest_yoy_change_m": 9.22, "normal_yoy_change_m": 0.31},
  "regime": {"a": "below", "b": "below", "a_6m": "normal"},
  "long_crop": {"allowed": false, "reason": "3m below normal — weak start for a long crop; prefer a short crop"},
  "confidence": {"level": "mixed", "freshness": "good"},
  "crop_guidance": {"water_need": "low", "duration_hint": "short", "long_crop_ok": false,
                    "type_examples": ["lentil / horse gram (residual)", "linseed", "rainfed pulses"]},
  "message": "• Outlook (rabi sowing window): water is set to draw down more than a typical year here. • Past year: a deficit. • Suggestion: water-light; conserve — a low-water crop … (abbreviated)"
}
```

Reading it: `forecast_change_m > 0` = the water table is set to **fall** (deplete) over the next 3 months —
here `+3.49 m` vs a seasonal normal of `+1.14 m`, so `regime.a = below` (drier than usual). Values are
bounded to a ±60%-of-current plausibility band, so `forecast_change_m` may sit exactly on that clamp.

### Browse the API in a browser (`/docs`)

The server ships FastAPI's interactive docs — **no `curl` needed**:

| URL | What it is |
|---|---|
| `http://<host>:8100/docs` | **Swagger UI** — run `/advisory` straight from the browser |
| `http://<host>:8100/redoc` | ReDoc — clean read-only API reference |
| `http://<host>:8100/health` | liveness JSON (`{status, loaded, has_6m}`) |
| `http://<host>:8100/openapi.json` | raw OpenAPI schema |

**Run an advisory from the browser:** open **`/docs`** → expand **`POST /advisory`** → **Try it out** →
put `{"lat":12.9716,"lon":77.5946,"date":"2025-01-30","verbose":true}` in the request body → **Execute**.
The JSON response (and the equivalent `curl`) appears inline.

`/advisory` is **POST-only** — a plain `…/advisory?lat=…&lon=…` address-bar URL returns **`405`**; use
`/docs` (or the `curl` POST above). Remote server? Tunnel first —
`ssh -N -L 8100:127.0.0.1:8100 <server>` — then open `http://127.0.0.1:8100/docs` locally.

---

## Notes

- **Latency** — a request is dominated by live Earth Engine + WRIS fetching, not the model. On a **GPU**
  a fully-live request is **~1 min** (warm imagery ≈ **55 s**; a cold request adds the imagery download,
  now fetched **concurrently** — ~10 neighbour tiles in ~8 s instead of ~50 s one-by-one). On **CPU**
  it's **~2–3 min** cold. A warm composite cache (`GWL_COMPOSITE_CACHE_DIR`) turns the imagery into a
  disk read; download concurrency is tunable via `GWL_COMPOSITE_WORKERS` (default 6). The remaining time
  is the numeric Earth Engine feature batch + the per-neighbour WRIS reads.
- **Earth Engine quota** — heavy or parallel querying can trip GEE's **noncommercial compute quota**
  ("your project … is in restricted mode"): requests then throttle hard or time out, though the service
  stays up and recovers as load drops. **Pace requests** (one at a time), reuse tiles via
  `GWL_COMPOSITE_CACHE_DIR`, or use a registered/paid Earth Engine project for sustained use.
- The anchor `date` must be today or earlier. The 3-month forecast is `anchor + 3 months`; the 6-month is
  `anchor + 6 months`.
- **SLM is optional** — the template fallback gives an identical decision; Gemma only rephrases.
- **GPU is optional** — `GWL_DEVICE=cpu` runs with no GPU (validated on a laptop). The forecast matches a
  GPU run to within **~0.01 m** (float rounding); the **decision, normals and confidence are identical**.
  CPU is slower per request (Prithvi runs on CPU) but otherwise equivalent.
- **Licensing:** code & model weights Apache-2.0 · dataset GODL-India · Gemma under the Gemma Terms of Use
  (referenced via Ollama, not shipped). See `NOTICE`.
