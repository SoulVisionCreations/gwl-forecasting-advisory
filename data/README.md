# data/ — the Dataset asset (download target)

Training data is **not committed** to this repo. It is published as the **Dataset** artefact on
AIKosh: open **<https://aikosh.indiaai.gov.in/web/datasets/details/ground_water_level_all.html>** and click
**"Download Dataset"** → **`ground_water_level_all_v1.zip`**, which unzips to a single **`gwl_data.csv`**
(~760 MB). It is used by (a) **fine-tuning / reproduction** (User B) and (b) the **advisory** — which reads
it for the ~10‑yr seasonal / year‑over‑year **normals** and (with the WRIS fallback) as a station index.
The **plain forecaster** (`python -m inference`) does **not** need it.

## Contents

```
data/
  gwl_data.csv                # numerical inputs: GWL readings + dynamic (rain/temp) + static features
                              # (25 columns; ~3.3M rows). See docs/ADVISORY.md for how the advisory uses it.
```

## Satellite composites (regenerable — NOT shipped)

The ~485k quarterly HLS composite tiles are **not** distributed. Regenerate them from the station
list using the shared downloader (needs Google Earth Engine access):

```bash
# 1) derive the station list from the CSV
python scripts/extract_gwl_stations.py            # -> gwl_stations.csv (code, safe_id, lat, lon)

# 2) download quarterly composites (server-side compositing via GEE; ~6-band 224x224 int16 GeoTIFFs)
python -m gwlcore.download_gwl_quarterly_composites   # (station, year, quarter) -> composite_<id>_<year>_<Q>.tif
```

`gwlcore.download_gwl_quarterly_composites` is the same module the inference engine reuses for its
per-request composite fetch, so training and serving share one tile recipe. License: GODL-India.
