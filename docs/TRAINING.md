# Training & Fine‑tuning

Reproduce or re‑train the champion model — a from‑scratch **TFT** jointly fine‑tuned with a **frozen
Prithvi‑EO‑2.0‑300M** encoder (LoRA). This is **User B**'s path; it is **not** needed for plain inference.

Prerequisites: [`SETUP.md`](SETUP.md) with the **`.[train]`** extra installed:

```bash
uv pip install -e ".[train]"     # adds mlflow, matplotlib, seaborn, tqdm
```

---

## Inputs you need

| Input | Where | Notes |
|---|---|---|
| GWL dataset | `data/gwl_data.csv` | the **Dataset** artefact (download; see [`data/README.md`](../data/README.md)) |
| Station list | `gwl_stations.csv` | derive with `python scripts/extract_gwl_stations.py` |
| Quarterly composites | `quarterly_composites/` | regenerate from Earth Engine (not shipped — see below) |
| Prithvi base | `Prithvi-EO-2.0-300M-TL/` | the foundation model (NASA–IBM, Apache‑2.0) |
| Earth Engine auth | `~/.config/earthengine/credentials` | `earthengine authenticate` |

### 1. Derive the station list

```bash
python scripts/extract_gwl_stations.py            # gwl_data.csv -> gwl_stations.csv (code, lat, lon)
```

### 2. Download the imagery composites

The ~485k HLS composite tiles are **regenerable, not distributed**. The same downloader the inference engine
uses server‑side produces them:

```bash
python -m gwlcore.download_gwl_quarterly_composites \
    --stations_csv gwl_stations.csv --project <gee> \
    --years 2023 2024 --quarters Q1 Q2 Q3 Q4 \
    --output_dir quarterly_composites [--workers 48]
# -> per-station 6-band 224x224 int16 GeoTIFFs under quarterly_composites/
```

Because training and serving share one tile recipe, the composites the model trains on are exactly what it consumes
at inference time.

---

## Run the pipeline

`slurm/run_big10.sh` orchestrates the three stages end‑to‑end:

```
prep (01_data_preparation) → scale (02_feature_scaler) → train (03_train) → reliability-calibration
```

> The final **reliability-calibration** step recomputes the advisory's test-set reliability constants and
> writes a paste-ready block to `<run_dir>/reliability_constants.txt` — see
> [After training: refresh the advisory reliability constants](#after-training-refresh-the-advisory-reliability-constants) below. It is non-fatal (a failure never aborts the run).

```bash
export CSV_PATH=data/gwl_data.csv COMPOSITE_DIR=quarterly_composites COMPOSITE_PERIOD=quarter
export PRITHVI_MODEL_DIR=Prithvi-EO-2.0-300M-TL STATION_INDEX_CSV=gwl_stations.csv
CUDA_VISIBLE_DEVICES=0 USE_PRITHVI=yes RUN_NAME=my_run EPOCHS=50 bash slurm/run_big10.sh
```

Output goes to `output/prithvi/my_run_<timestamp>/`:

```
best_model.pt      # the trained bundle
config.json        # the run config
data/              # scalers, data_config, station_stats, tile_manifest, variograms …
plots/             # training curves + diagnostics
```

Everything is **override‑friendly** — every setting is an environment variable with a default (`${VAR:-default}`),
so you change behavior by exporting the var before the run. No code edits needed. The full list with defaults is at
the top of [`slurm/run_big10.sh`](../slurm/run_big10.sh); the ones you'll actually touch:

**Data & forecast window**

| Env var | Default | Meaning |
|---|---|---|
| `HORIZON` | `3` | **forecast horizon in months** — how far ahead to predict (e.g. `HORIZON=6` → a 6‑month model). Re‑preps the targets. |
| `LOOKBACK` | `1` | history window fed to the model, in **years** |
| `LOOKBACK_WINDOW` | `6d` | spacing of the lookback timesteps (`6d` → 60 steps/yr) |
| `GAP_DAYS` | `30` | ± days to match a GWL reading to a timestep |
| `COMPOSITE_PERIOD` | `halfyear` | imagery cadence: `halfyear` \| `quarter` (**release = `quarter`**) |
| `COMPOSITE_DIR` | …/half_yearly | folder of composite tiles — **must match `COMPOSITE_PERIOD`** |
| `CSV_PATH` | gwl_data.csv | training dataset |
| `STATION_INDEX_CSV` | gwl_stations.csv | station registry (code, lat, lon) |

**Features (what the model sees)**

| Env var | Default | Meaning |
|---|---|---|
| `USE_PRITHVI` | `yes` | satellite‑imagery encoder on/off (`no` → the imagery‑free TFT baseline) |
| `USE_RAIN_TEMP` | `yes` | rainfall + temperature dynamic features |
| `USE_STATIC_FEATURES` | `yes` | static well/aquifer/lithology + derived per‑station stats |
| `DROP_NDVI_SM` | `yes` | drop NDVI + soil‑moisture channels (champion drops them) |
| `USE_REVIN` | `yes` | reversible instance‑norm on the GWL sequence |
| `INTERPOLATE_LOOKBACK_GWL` | `yes` | densify the lookback with interpolation |

**Model**

| Env var | Default | Meaning |
|---|---|---|
| `MODEL_TYPE` | `tft` | `tft` \| `transformer` \| `conditioned_lstm` \| `conditioned_transformer` \| `lstm` (**Prithvi requires `tft`**) |
| `PRED_CLAMP_PCT` | `0.6` | clamp the predicted change to \|Δ\| ≤ pct·\|current\| |
| `TFT_D_MODEL`,`TFT_N_HEADS`,`TFT_DROPOUT` | `64`,`4`,`0.1` | TFT size |

**Prithvi / LoRA**

| Env var | Default | Meaning |
|---|---|---|
| `LORA_R`, `LORA_ALPHA` | `16`, `32` | LoRA rank / α on the frozen encoder |
| `PRITHVI_PROJ_DIM` | `32` | projector output dim (1024 → this) |
| `FT_LR` | `1e-4` | LoRA + projector learning rate |
| `FORECASTER_LR_SCALE` | `10` | TFT LR = `FT_LR` × this (→ 1e‑3) |
| `PRITHVI_N_TILES` × `PRITHVI_SAMPLES_PER_TILE` | `64`×`32` | train batch (= 2048) |
| `PRITHVI_GRAD_CKPT` | `no` | gradient checkpointing (less GPU memory, slower) |

**Optimisation**

| Env var | Default | Meaning |
|---|---|---|
| `EPOCHS` | `50` | training epochs |
| `BATCH_SIZE` | `2048` | val/test batch (train batch = n_tiles×spt under Prithvi) |
| `LR`, `WEIGHT_DECAY`, `DROPOUT` | `1e-3`, `1e-5`, `0.2` | optimiser knobs |
| `SCHEDULER_FACTOR`, `SCHEDULER_PATIENCE` | `0.5`, `10` | LR‑on‑plateau schedule |

**Split, filters & orchestration**

| Env var | Default | Meaning |
|---|---|---|
| `SPLIT_STRATEGY` | `station_time` | train/val/test split scheme |
| `TRAIN_END`/`VAL_START`/`VAL_END`/`TEST_START` | 2024‑12‑31 / 2025‑01‑01 / 2025‑08‑31 / 2025‑09‑01 | date boundaries |
| `INCLUDE_STATES` | (all) | comma‑list to restrict training to given states |
| `RUN_NAME` | (timestamp) | run label (a timestamp is appended) |
| `CUDA_VISIBLE_DEVICES` | — | GPU id (vary it + `RUN_NAME` to run several at once) |
| `REUSE_DATA_DIR` | — | skip prep+scale, reuse an already‑prepared `data/` dir (ablations) |
| `PY` | `python` | Python to use — point it at the repo venv, e.g. `.venv/bin/python` |

You can also run the three stages directly (each reads the same env vars):
`python -m gwlcore.data_preparation`, `python -m gwlcore.feature_scaler`, `python training/train.py` (all support `--help`).

---

## After training: refresh the advisory reliability constants

**⚠️ Don't skip this on a retrain.** The farmer advisory surfaces **calibrated test‑set reliability** — the "about N in 10" figures on
`outlook.water_trend` (rise/fall **direction accuracy**) and `outlook.vs_normal` (above/below‑normal
**side‑accuracy**). Those percentages are **baked constants** in `advisory/reliability.py`
(`_RELIABILITY` and `DIRECTION_ACC_BY_PRED_MOVE`), calibrated on the **shipped champion's** test set.
**Train a new model and they go stale** — the advisory would keep quoting the *old* model's reliability for
the *new* model's forecasts. So refreshing them is part of every retrain, not an afterthought.

`run_big10.sh` already **recomputes them automatically** as the last pipeline step — **no model inference, no
new artifact**, just pandas over the run's own `data/test_samples.pkl` (the `predicted_gwl_delta` already
stored there). It writes a **paste‑ready block** to:

```
<run_dir>/reliability_constants.txt
```

**The one manual step (by design):** open that file and paste its `_RELIABILITY["<Nm>"]` +
`DIRECTION_ACC_BY_PRED_MOVE["<Nm>"]` blocks into `advisory/reliability.py`, then **commit** it. A GPU run
never rewrites source — the constants stay human‑reviewed **in code** (they are *not* a shipped model/dataset
artifact). Do this for **each horizon** you retrain: the `3m` and `6m` models each own a block (`"3m"` /
`"6m"`), so a dual‑model release pastes both.

To (re)generate the block by hand for an existing run (e.g. if the auto‑step was skipped):

```bash
PYTHONPATH=. python scripts/calibrate_clarity.py \
    --pkl <run_dir>/data/test_samples.pkl --horizon <3|6> --gwl_csv data/gwl_data.csv \
    --state_csv <run_dir>/performance/state_performance_test.csv --emit-constants
```

The emitted numbers are a **self‑normal floor** (the live neighbour‑normal path runs a few points higher), so
shipping them as‑is stays conservative. How they're surfaced to the farmer is documented in
[`RESPONSE_FIELDS.md`](RESPONSE_FIELDS.md) (the `outlook` fields) and [`MODEL_CARD.md`](MODEL_CARD.md)
("How to trust the advisory").

---

## Champion configuration

The published model's hyperparameters are the **defaults** of `run_big10.sh` / `train.py`:

| Group | Values |
|---|---|
| Forecast | horizon **3 months**, `num_timesteps` 60, lookback 6d, gap 30d |
| GWL encoding | `use_delta_gwl`, `use_revin` (std floor 0.1), delta‑clamp `pct` 0.6, `interpolate_lookback` |
| Features | `only_gwl` + `rain_temp`, `drop_ndvi_sm`, static features on |
| Prithvi LoRA | rank 16, α 32, on `qkv`; projector 1024→32 |
| Imagery | `n_tiles` 64, subtiles‑per‑tile 32, quarterly composites |
| Optim | batch 2048; LoRA/projector lr 1e‑4, TFT lr 1e‑3 |
| Split | `station_time` |

Trainable parameters ≈ **0.474%** (LoRA 1.573M + projector 0.03M + TFT 0.31M) over a **~330.7M frozen** Prithvi base
(**332,643,180 total**).

---

## Recipes — example runs

Set the shared inputs once, then pick a recipe (each is a single line):

```bash
export CSV_PATH=data/gwl_data.csv
export COMPOSITE_DIR=quarterly_composites COMPOSITE_PERIOD=quarter
export STATION_INDEX_CSV=gwl_stations.csv PRITHVI_MODEL_DIR=Prithvi-EO-2.0-300M-TL
export PY=.venv/bin/python          # the repo's own venv
```

```bash
# 1) CHAMPION — the release config (3-month, quarterly, Prithvi + TFT)
CUDA_VISIBLE_DEVICES=0 USE_PRITHVI=yes RUN_NAME=champion EPOCHS=50 bash slurm/run_big10.sh

# 2) 6-MONTH horizon instead of 3 — just set HORIZON=6 (re-preps 6-month targets, then trains)
CUDA_VISIBLE_DEVICES=0 HORIZON=6 RUN_NAME=h6 EPOCHS=50 bash slurm/run_big10.sh

# 3) Imagery-free BASELINE (no Prithvi — faster, TFT only)
CUDA_VISIBLE_DEVICES=1 USE_PRITHVI=no RUN_NAME=baseline EPOCHS=50 bash slurm/run_big10.sh

# 4) Longer history — 2 years of lookback instead of 1
CUDA_VISIBLE_DEVICES=0 LOOKBACK=2 RUN_NAME=lookback2y EPOCHS=50 bash slurm/run_big10.sh

# 5) Quick SMOKE test — one state, 1 epoch (verifies the whole pipeline in minutes)
CUDA_VISIBLE_DEVICES=0 INCLUDE_STATES=Karnataka EPOCHS=1 RUN_NAME=smoke bash slurm/run_big10.sh

# 6) ABLATION reusing already-prepped data (skips prep+scale → much faster iteration)
CUDA_VISIBLE_DEVICES=0 REUSE_DATA_DIR=output/prithvi/champion_<ts>/data RUN_NAME=ablation EPOCHS=30 bash slurm/run_big10.sh

# 7) A different MODEL family (imagery-free conditioned LSTM)
CUDA_VISIBLE_DEVICES=1 USE_PRITHVI=no MODEL_TYPE=conditioned_lstm RUN_NAME=clstm EPOCHS=50 bash slurm/run_big10.sh
```

Each writes to `output/prithvi/<RUN_NAME>_<timestamp>/`.
- Changing `HORIZON`, `LOOKBACK`, `LOOKBACK_WINDOW`, `GAP_DAYS`, feature toggles or the split **re-runs data prep**
  (the samples change). Changing only optimiser/model knobs can reuse a prepared `data/` via `REUSE_DATA_DIR`.
- A 6-month model trained this way is **served identically** — the inference engine reads the horizon from the
  packaged config, so `python -m inference --run-dir weights_h6 …` returns a 6-month forecast with no flag changes.

---

## Package the trained model for release

Turn a trained run into the **fat, self‑contained** Model asset (`weights/`) — backbone baked in, zero Hugging Face:

```bash
python scripts/build_package.py \
    --run-dir output/prithvi/my_run_<ts> \
    --prithvi-dir Prithvi-EO-2.0-300M-TL \
    --registry-csv gwl_stations.csv \
    --out weights --vendor-prithvi-arch
# (--slim instead of --vendor-prithvi-arch → a ~23 MB delta checkpoint that references the public Prithvi base)
```

`build_package.py` copies `best_model.pt` + the inference artifacts (scalers, data_config, station_stats, tile
manifest, variograms, NWDP index), writes the station registry as **`gwl_stations.tsv`**, and vendors the Prithvi
architecture (`prithvi_mae.py` + `config.json`). Every file is an AIKosh‑Model‑accepted format.

**Verify the package** with the exact query that produced a known result during training, via either
`python -m inference --run-dir weights …` (see [`INFERENCE.md`](INFERENCE.md)) or the API — the `change_m` must match
the source run.

---

## Notes

- **Reproducibility:** train and serve on the pinned environment (`pyproject.toml`). The fresh pinned venv reproduces
  the published numbers exactly; unpinned CUDA builds add a benign ~1e‑4 drift.
- **MLflow / plots:** the `.[train]` extra pins `mlflow==3.11.1` and `matplotlib==3.10.8` (newer versions break the
  file‑store backend / boxplot labels used by the training plots).
- **Compute:** the champion trains on a single high‑memory GPU; the frozen 330.7M base means only ~1.9M params get
  gradients, so memory is dominated by the (frozen) backbone forward pass over imagery tiles.
