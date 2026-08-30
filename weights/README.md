# weights/ — the Model asset (download target)

The model files are **not committed** to this repo. They are published as the **Model** artefact on
AIKosh (`gwl_model_3m6m_v1.zip`) and downloaded here (login-gated; download via the AIKosh portal).

The artefact is a **combined 3‑month + 6‑month bundle that shares one frozen Prithvi‑EO backbone**.
Unzip it anywhere (it already contains a top-level `gwl_model_v1/` folder):

```bash
unzip gwl_model_3m6m_v1.zip           # -> ./gwl_model_v1/
```

## Contents (self-contained — zero Hugging Face dependency at load)

```
gwl_model_v1/
  prithvi_base/                       # SHARED frozen Prithvi-EO-2.0-300M backbone (shipped ONCE)
    Prithvi_EO_V2_300M_TL.pt          #   ~1.3 GB base weights (Apache-2.0, NASA-IBM)
    prithvi_mae.py                    #   Prithvi architecture code — imported at load
    config.json                       #   base normalisation (mean/std) + dims
  model_3m/                           # 3-month-horizon forecaster (slim delta)
    best_model.pt                     #   ~23 MB — LoRA + projector + TFT (base STRIPPED, rebuilt from prithvi_base)
    config.json  data_metadata.json
    gwl_stations.tsv                  #   station registry (code, lat, lon). TSV — AIKosh Model assets reject .csv
    data/
      scalers.pkl                     #   fitted feature/target scalers
      data_config_variables.pkl       #   horizon / num_timesteps / gap / composite_period ...
      station_stats.pkl               #   per-station derived features + gwl_anomaly baseline (THIS model's own)
      tile_manifest.pkl               #   composite tile manifest
      state_variograms.pkl            #   per-state kriging variograms (IDW default; kriging reported alongside)
      nwdp_station_index.pkl          #   NWDP well index (enables --gwl-source nwdp)
  model_6m/                           # 6-month-horizon forecaster — same layout (its OWN station_stats etc.)
```

Each `best_model.pt` is a **slim** checkpoint: the frozen backbone is stripped and **reconstructed at load
from `prithvi_base/`** (public pretrained weights, byte-identical across both models). The loaded model is
identical to a fat checkpoint. **Because the backbone lives in the shared `prithvi_base/` (not inside each
model dir), you must point the loader at it** via `GWL_PRITHVI_MODEL_DIR` (API) / `--prithvi-model-dir` (CLI).

## Using it

- **Forecaster (single horizon)** — see [`../docs/INFERENCE.md`](../docs/INFERENCE.md):
  ```bash
  python -m inference --run-dir gwl_model_v1/model_3m \
      --prithvi-model-dir gwl_model_v1/prithvi_base --lat 12.9716 --lon 77.5946
  ```
- **Advisory (3m + 6m together)** — see [`../docs/ADVISORY.md`](../docs/ADVISORY.md)
  (`GWL_RUN_DIR=…/model_3m`, `GWL_RUN_DIR_6M=…/model_6m`, `GWL_PRITHVI_MODEL_DIR=…/prithvi_base`).

Build the bundle from trained runs with `scripts/build_package.py`.
