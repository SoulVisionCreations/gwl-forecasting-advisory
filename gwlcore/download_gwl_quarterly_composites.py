"""
download_gwl_quarterly_composites.py
====================================
QUARTERLY HLS composites for GWL station locations (Prithvi static encoding).

Same machinery as download_gwl_composites.py (the half-yearly GWL downloader) and
download_annual_composite.py (GEE high-volume endpoint + getDownloadURL + thread
pool, all compositing server-side) EXCEPT the year is split into FOUR calendar
quarters instead of two halves:

    Q1 = Jan 1 .. Apr 1   (winter / Rabi)              doy ~ 46
    Q2 = Apr 1 .. Jul 1   (pre-monsoon / summer)       doy ~ 135
    Q3 = Jul 1 .. Oct 1   (monsoon / Kharif)           doy ~ 227
    Q4 = Oct 1 .. Jan 1   (post-monsoon)               doy ~ 319

    (station, year, quarter) -> median(cloud-masked HLS scenes in window)
                             -> one (6, 224, 224) int16 GeoTIFF (reflectance x 10000)

Output: "composite_<safe_id>_<year>_<Q1|Q2|Q3|Q4>.tif"
    6 bands: BLUE, GREEN, RED, NIR, SWIR1, SWIR2
    224 x 224 pixels, 30 m native (6720 m box), reflectance x 10000 (int16)
    0 = nodata

Coverage note
-------------
A quarter is a 3-month window, shorter than the half-year (6 mo) and annual (12 mo)
composites. The monsoon quarter Q3 (Jul-Oct) is both the shortest AND cloudiest
window, so its valid-pixel fraction can be noticeably lower than the half-year H2
(which got >0.91). min_coverage stays at 0.05 (an empty-region guard, NOT a cloud
filter) for consistency, but RUN THE PROBE FIRST and inspect Q3 coverage before
committing to the full ~500k-composite job.

Tile selection at inference / data-prep time
---------------------------------------------
Given a GWL sample with current_date, two reasonable rules (decide at data-prep):
  (a) SAME-QUARTER (seasonal match, like the half-yearly rule):
        quarter = quarter_for_date(current_date.month)
        use (current_year - 1, same_quarter); walk back year-by-year if missing.
        Matches seasonal landscape character but is up to ~1 year stale.
  (b) MOST-RECENT-COMPLETE quarter (freshest):
        use the latest quarter that ENDS on/before current_date.
        e.g. current_date 2023-06-22 -> Q1 2023 (ended Apr 1) is freshest.
        Fresher, but mixes seasons.
If nothing found -> zero embedding (no Prithvi contribution). The download fetches
ALL quarters x ALL years, so either rule is supported downstream.

Temporal leakage note (identical to NDVI pipeline Fix A):
  Never use a quarter that overlaps or follows the prediction cutoff. Rule (a)'s
  (current_year - 1) candidate is always safe; rule (b) requires the quarter to
  end strictly before current_date.

Example
-------
  python download_gwl_quarterly_composites.py \
      --stations_csv gwl_stations.csv \
      --project your-gee-project-id \
      --years 2020 2021 2022 2023 2024 \
      --output_dir gwl_quarterly_composites \
      --workers 48
"""

import os
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import requests
import pandas as pd

HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"

LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7"]
SENTINEL_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]
COMMON_BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"]
REFLECTANCE_SCALE = 10000

QUARTERS = ("Q1", "Q2", "Q3", "Q4")
QUARTER_DOY = {"Q1": 46, "Q2": 135, "Q3": 227, "Q4": 319}   # window midpoints


def quarter_window(year, quarter):
    if quarter == "Q1":
        return f"{year}-01-01", f"{year}-04-01"
    if quarter == "Q2":
        return f"{year}-04-01", f"{year}-07-01"
    if quarter == "Q3":
        return f"{year}-07-01", f"{year}-10-01"
    return f"{year}-10-01", f"{year + 1}-01-01"


def quarter_for_date(month):
    """Which calendar quarter a month belongs to."""
    return f"Q{(month - 1) // 3 + 1}"


# ── GEE masking / harmonization (identical to NDVI + half-yearly pipeline) ─────

def mask_clouds_hls(image):
    fmask = image.select("Fmask")
    mask = (fmask.bitwiseAnd(1 << 1).eq(0)
            .And(fmask.bitwiseAnd(1 << 2).eq(0))
            .And(fmask.bitwiseAnd(1 << 3).eq(0)))
    return image.updateMask(mask)


def rename_landsat(image):
    return image.select(LANDSAT_BANDS + ["Fmask"], COMMON_BANDS + ["Fmask"])


def rename_sentinel(image):
    return image.select(SENTINEL_BANDS + ["Fmask"], COMMON_BANDS + ["Fmask"])


def utm_epsg(lat, lon):
    zone = int((lon + 180) / 6) + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def hls_composite(region, start_date, end_date):
    landsat = (ee.ImageCollection("NASA/HLS/HLSL30/v002")
               .filterBounds(region).filterDate(start_date, end_date)
               .map(rename_landsat).map(mask_clouds_hls))
    sentinel = (ee.ImageCollection("NASA/HLS/HLSS30/v002")
                .filterBounds(region).filterDate(start_date, end_date)
                .map(rename_sentinel).map(mask_clouds_hls))
    return landsat.merge(sentinel).select(COMMON_BANDS).median()


# ── Single composite download ─────────────────────────────────────────────────

# Hard client-side timeout for the getDownloadURL GEE call. It can hang indefinitely on a
# throttled/flaky backend (no timeout on this transport; the subsequent requests.get IS
# bounded). Bounding it in a daemon thread turns a hang into a retryable error. Tunable via
# GWL_GEE_CALL_TIMEOUT_S (shared with the numeric path's default).
try:
    _COMPOSITE_GEE_TIMEOUT_S = int(os.environ.get("GWL_GEE_CALL_TIMEOUT_S", "60"))
except ValueError:
    _COMPOSITE_GEE_TIMEOUT_S = 60


def _call_with_timeout(fn, timeout_s):
    """Run fn() in a daemon thread; raise RuntimeError('...timed out...') if it exceeds
    timeout_s (orphaned thread abandoned — the caller's retry loop issues a fresh call)."""
    box = {}

    def _run():
        try:
            box["val"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise RuntimeError(f"getDownloadURL timed out after {timeout_s}s (abandoned, will retry)")
    if "err" in box:
        raise box["err"]
    return box.get("val")


def fetch_composite(station_code, lat, lon, year, quarter, out_dir,
                    patch_radius_m, patch_px, min_coverage, retries=5):
    """Download one (station, year, quarter) composite. Returns (status, name, valid_frac)."""
    file_name = f"composite_{station_code}_{year}_{quarter}"
    out_path = os.path.join(out_dir, file_name + ".tif")
    if os.path.exists(out_path):
        return ("skip-exists", file_name, None)

    start_date, end_date = quarter_window(year, quarter)
    region = ee.Geometry.Point([lon, lat]).buffer(patch_radius_m).bounds()
    epsg = utm_epsg(lat, lon)
    img = (hls_composite(region, start_date, end_date)
           .multiply(REFLECTANCE_SCALE).toInt16().unmask(0))
    params = {
        "region": region, "crs": epsg,
        "dimensions": f"{patch_px}x{patch_px}",
        "format": "GEO_TIFF", "bands": COMMON_BANDS,
    }

    data = None
    last_err = None
    for attempt in range(retries):
        try:
            url = _call_with_timeout(lambda: img.getDownloadURL(params), _COMPOSITE_GEE_TIMEOUT_S)
            resp = requests.get(url, timeout=300)
            if resp.status_code == 200:
                data = resp.content
                break
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)[:120]
        time.sleep(2 ** attempt)

    if data is None:
        return ("error", f"{file_name}: {last_err}", None)

    try:
        from rasterio.io import MemoryFile
        with MemoryFile(data) as mem, mem.open() as src:
            arr = src.read()
    except Exception as e:
        return ("error", f"{file_name}: decode {str(e)[:80]}", None)

    if arr.shape != (len(COMMON_BANDS), patch_px, patch_px):
        return ("empty", file_name, 0.0)

    valid = float((arr[COMMON_BANDS.index("NIR")] > 0).mean())
    if valid < min_coverage:
        return ("empty", file_name, valid)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_path)
    return ("ok", file_name, valid)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stations_csv", required=True,
                   help="CSV with columns: station_code, safe_id, lat, lon")
    p.add_argument("--project", required=True,
                   help="GEE cloud project id.")
    p.add_argument("--years", nargs="+", type=int,
                   default=[2020, 2021, 2022, 2023, 2024],
                   help="Composite years. Use year Y to serve GWL samples in year Y+1. "
                        "e.g. for samples 2021-2025 pass --years 2020 2021 2022 2023 2024")
    p.add_argument("--quarters", nargs="+", choices=["Q1", "Q2", "Q3", "Q4"],
                   default=["Q1", "Q2", "Q3", "Q4"],
                   help="Which quarters to download (default: all four).")
    p.add_argument("--output_dir", default="gwl_quarterly_composites")
    p.add_argument("--patch_radius", type=int, default=3360,
                   help="Buffer radius in metres around station centre (default 3360 -> 6720 m box).")
    p.add_argument("--patch_px", type=int, default=224)
    p.add_argument("--min_coverage", type=float, default=0.05,
                   help="Empty-region guard (NIR > 0 fraction). NOT a cloud filter. "
                        "Q3 (monsoon) coverage can be low — inspect the probe first.")
    p.add_argument("--workers", type=int, default=48,
                   help="Concurrent download threads.")
    p.add_argument("--retries", type=int, default=5)
    p.add_argument("--limit_stations", type=int, default=None,
                   help="Process only the first N stations (for testing).")
    args = p.parse_args()

    ee.Initialize(project=args.project, opt_url=HIGH_VOLUME)
    os.makedirs(args.output_dir, exist_ok=True)

    stations_df = pd.read_csv(args.stations_csv, sep=None, engine="python")   # sniff .csv / .tsv
    required = {"station_code", "safe_id", "lat", "lon"}
    missing = required - set(stations_df.columns)
    if missing:
        raise ValueError(
            f"stations_csv missing columns: {missing}. "
            f"Run extract_gwl_stations.py first to generate this file."
        )

    if args.limit_stations:
        stations_df = stations_df.head(args.limit_stations)

    # Use safe_id for filenames (handles spaces, slashes in station_code)
    jobs = []
    for r in stations_df.itertuples():
        for year in args.years:
            for quarter in args.quarters:
                jobs.append((str(r.safe_id), float(r.lat), float(r.lon), year, quarter))

    total = len(jobs)
    print(f"{len(stations_df)} stations x {len(args.years)} yr x {len(args.quarters)} quarters "
          f"= {total} composites -> {args.output_dir}/  ({args.workers} workers)")

    counts, lock, t0, done = {}, threading.Lock(), time.time(), 0
    cov_rows = []
    err_log = os.path.join(args.output_dir, "_errors.log")

    def submit(j):
        station_code, lat, lon, year, quarter = j
        return fetch_composite(
            station_code, lat, lon, year, quarter,
            args.output_dir, args.patch_radius,
            args.patch_px, args.min_coverage, args.retries
        ), j

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(submit, j) for j in jobs]
        for fut in as_completed(futures):
            (status, name, valid), (station_code, lat, lon, year, quarter) = fut.result()
            with lock:
                counts[status] = counts.get(status, 0) + 1
                done += 1
                if status in ("error", "empty"):
                    with open(err_log, "a") as ef:
                        ef.write(f"{status}\t{name}\n")
                if valid is not None:
                    cov_rows.append({
                        "station_code": station_code, "lat": lat, "lon": lon,
                        "year": year, "quarter": quarter, "valid_frac": valid,
                    })
                if done % 25 == 0 or done == total:
                    rate = done / (time.time() - t0) * 60
                    print(f"  {done}/{total} | {rate:.1f} comp/min | "
                          + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                          flush=True)

    dt = (time.time() - t0) / 60
    print(f"\nDone in {dt:.1f} min. "
          + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if cov_rows:
        cov = pd.DataFrame(cov_rows)
        cov.to_csv(os.path.join(args.output_dir, "coverage_check.csv"), index=False)
        print("\n=== valid-pixel fraction of downloaded tiles (NIR > 0) ===")
        print(cov.groupby("quarter")["valid_frac"].agg(["mean", "min", "median"]).round(4))

        for q in args.quarters:
            sub = cov[cov.quarter == q]
            if len(sub):
                print(f"\n{q}: min={sub.valid_frac.min():.4f} | "
                      f">0.70 in {(sub.valid_frac > 0.7).mean()*100:.0f}% | "
                      f">0.90 in {(sub.valid_frac > 0.9).mean()*100:.0f}%")

        worst = cov.nsmallest(5, "valid_frac")
        print("\nworst 5 tiles:")
        for r in worst.itertuples():
            print(f"  {r.station_code} {r.year} {r.quarter}: {r.valid_frac:.4f}")

        print(f"\nWrote {args.output_dir}/coverage_check.csv")

    print(f"Composites in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
