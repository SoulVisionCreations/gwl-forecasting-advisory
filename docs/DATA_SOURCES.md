# Data Sources

The model follows a **"download everything live, store almost nothing"** principle for observational data.
At inference time it fetches the inputs it needs from public APIs per request; it ships only the trained weights
and a few small lookups / training‑derived statistics.

---

## What is fetched live (per request, not pre‑stored)

| Signal | Source | How |
|---|---|---|
| **Current & historical GWL** | **NWDP** (default) **or** India‑WRIS **or** a local CSV | `--gwl-source nwdp\|wris` / `--local-csv` |
| **Satellite composites** (Prithvi imagery) | Google Earth Engine (HLS) | server‑side compositing, 6‑band 224×224 |
| **Numerical / dynamic** (rainfall, temperature) | Google Earth Engine (CHIRPS, ERA5) | aggregated over the lookback window |
| **Static** geospatial layers | Google Earth Engine | at the query location |
| **Forecast‑window weather** | Open‑Meteo | seasonal forecast; falls back to climatology when out of range |

Nothing above is cached to disk across requests (within a single request, imagery tiles are held in working memory).

### Groundwater level (GWL) sources

- **NWDP** (`--gwl-source nwdp`, **default**) — live CKAN telemetry. Finds the coordinate‑nearest well, applies a
  staleness guard (drops wells whose latest reading is > 100 days old → next‑nearest), and forward‑fills (LOCF) the
  last reading up to the anchor. Default because it is reachable and fresher than WRIS.
- **India‑WRIS** (`--gwl-source wris`) — the official CGWB water‑resources portal; **recurring outages make it
  currently unreliable**, so it is no longer the default. The ~10‑yr **normals** read from the **CSV** snapshot
  (`gwl_data.csv`), not WRIS.
- **Local CSV** (`--local-csv`) — offline / parity runs; takes precedence when set.

**Freshness handling (all sources, shared logic):** live portals publish ~a quarter behind, so a well's newest
reading is usually older than the anchor date. Per neighbour, with `age = anchor − newest reading ≤ anchor`:

| Age | Behaviour |
|---|---|
| ≤ 30 days | fresh — used as‑is |
| 30–200 days | **stale** — forward‑filled (LOCF) up to the anchor, with a note |
| > 200 days | **too stale** — that neighbour is dropped |

This keeps every neighbour on **one common anchor** so the spatial (IDW) blend mixes comparable "as‑of" levels.

### Earth Engine

Satellite composites and the numerical/static feature layers come from **Google Earth Engine**. External users
authenticate with per‑user OAuth (`earthengine authenticate`) and supply their own Cloud project
(`--gee-project` / `GWL_GEE_PROJECT`). GEE is used only to fetch **public** input layers — **model outputs are not
encumbered by GEE terms**.

---

## What is shipped (from disk — the Model asset, not data downloads)

| File | Size | Why it must ship |
|---|---|---|
| `best_model.pt` | ~1.3 GB (23 MB slim) | the trained weights (Prithvi backbone baked in + LoRA + projector + TFT) |
| `data/scalers.pkl` | — | fitted feature/target scalers |
| `data/data_config_variables.pkl` | — | horizon / timesteps / gap / composite period … |
| `data/station_stats.pkl` | 1.8 MB | **training‑derived** per‑station statistics + `gwl_anomaly` baseline — **cannot be recomputed live** |
| `data/state_variograms.pkl` | 0.13 MB | training‑derived kriging variograms (kriging is the alternative to the default IDW) |
| `data/tile_manifest.pkl` | 17 MB | composite tile manifest (a lookup) |
| `data/nwdp_station_index.pkl` | 2.8 MB | NWDP well index — a **location** lookup, not GWL values |
| `gwl_stations.tsv` | — | station registry: code, lat, lon (a **location** lookup) |
| `prithvi_base/{prithvi_mae.py, config.json}` | — | Prithvi architecture code + normalisation (loaded at startup) |

These lookups hold **locations and training‑derived statistics**, not observational values. `station_stats.pkl` and
`state_variograms.pkl` are the only genuinely "stored data" a forecast consumes; everything observational is live.

The large training samples (`samples.pkl`, ~2.3 GB) are **not** shipped.

---

## Training data (the Dataset asset)

For fine‑tuning / reproduction (User B) only:

- **`gwl_data.csv`** — groundwater readings + dynamic (rain/temp) + static features; ~3.3M readings across
  10,411 stations. Published as the **Dataset** artefact.
- **Quarterly HLS composites** — ~485k 6‑band tiles; **regenerable, not distributed** (rebuild via
  `gwlcore.download_gwl_quarterly_composites`; see [`TRAINING.md`](TRAINING.md)).

---

## Licensing & attribution

| Data | Provider | License |
|---|---|---|
| Groundwater observations | Central Ground Water Board (CGWB) / India‑WRIS / NWDP | **GODL‑India** |
| HLS satellite imagery | NASA / USGS · Copernicus / ESA | open (NASA/USGS · Copernicus) |
| Rainfall / temperature | CHIRPS · ERA5 (via GEE) | respective open terms |
| Forecast weather | Open‑Meteo | **CC‑BY 4.0** |
| Prithvi‑EO‑2.0‑300M | NASA · IBM | **Apache‑2.0** |

Model weights & code are **Apache‑2.0**; the training dataset is **GODL‑India**. See the repository `NOTICE` and the
[`MODEL_CARD.md`](MODEL_CARD.md) for full attribution.
