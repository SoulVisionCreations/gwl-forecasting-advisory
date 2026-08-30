"""CLI for GWL Prithvi+TFT inference.

    # production: arbitrary point
    python -m inference --run-dir <run> --lat 32.288 --lon 75.743 [--date 2023-06-06]

    # validation: a known station, left out of its own neighbours, scored vs truth
    python -m inference --run-dir <run> --station-code 030406 --leave-one-out --date 2024-02-15

A thin adapter over TFTInferenceEngine.predict() — the SAME core a future FastAPI
app would call. (We use `python -m inference` rather than a root inference.py to
avoid clashing with the `inference/` package name.)
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser(prog="inference", description="GWL Prithvi+TFT inference")
    ap.add_argument("--run-dir", required=True, help="training run dir (best_model.pt + scalers + config)")
    ap.add_argument("--lat", type=float, default=None, help="query latitude (omit when --station-code given)")
    ap.add_argument("--lon", type=float, default=None, help="query longitude (omit when --station-code given)")
    ap.add_argument("--date", default=None, help="current/anchor date YYYY-MM-DD (default: today; must be today or earlier — forecast is anchor + horizon)")
    ap.add_argument("--station-code", default=None, help="treat query as this known station (validation)")
    ap.add_argument("--leave-one-out", action="store_true", help="exclude that station from its neighbours")
    ap.add_argument("--details", action="store_true", help="include the per-well `details` block in the response")
    ap.add_argument("--k", type=int, default=10, help="number of nearest-neighbour stations")
    ap.add_argument("--min-neighbours", type=int, default=1,
                    help="fewer surviving neighbours than this → status=error (insufficient_neighbours)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--composite-cache-dir", default=None, help="HLS composite cache dir (default: training composite_dir)")
    ap.add_argument("--station-registry-csv", default=None, help="gwl_stations.csv (default: ckpt station_index_csv)")
    ap.add_argument("--local-csv", default=None,
                    help="source GWL+dynamic+static from this training CSV (parity, reliable) instead of live GEE+WRIS")
    ap.add_argument("--gwl-source", default="nwdp", choices=["wris", "nwdp"],
                    help="live GWL lookback source: nwdp (DEFAULT — WRIS API is unreliable/down) or wris; "
                         "swaps ONLY the GWL slice, GEE still does dynamic/static/composite")
    ap.add_argument("--nwdp-index", default=None,
                    help="nwdp_station_index.pkl (default: run_dir/data/nwdp_station_index.pkl)")
    ap.add_argument("--station-stats", default=None,
                    help="precomputed StationStats pkl (small); else built from run_dir/data/train_samples.pkl (1.5 GB)")
    ap.add_argument("--gee-project", default=None, help="GEE project id (default: vendored fetcher default)")
    ap.add_argument("--idw-power", type=float, default=2.0)
    ap.add_argument("--kriging-variograms", default=None,
                    help="state_variograms.pkl → also report an ordinary-kriging delta alongside IDW "
                         "(default: run_dir/data/state_variograms.pkl if present)")
    ap.add_argument("--prithvi-model-dir", default=None, help="override Prithvi dir baked in the checkpoint")
    ap.add_argument("--out", default=None, help="write the JSON response to this path (also printed)")
    args = ap.parse_args()

    if (args.lat is None or args.lon is None) and args.station_code is None:
        ap.error("provide either --lat and --lon, or --station-code (validation).")

    # Imported lazily so `--help` / argparse errors don't pay the heavy import cost.
    from inference.config import InferenceConfig
    from inference.engine import TFTInferenceEngine

    cfg = InferenceConfig(
        run_dir=args.run_dir,
        composite_cache_dir=args.composite_cache_dir,
        station_registry_csv=args.station_registry_csv,
        k_neighbours=args.k,
        min_neighbours=args.min_neighbours,
        device=args.device,
        local_csv=args.local_csv,
        gwl_source=args.gwl_source,
        nwdp_index_path=args.nwdp_index,
        station_stats_pkl=args.station_stats,
        gee_project=args.gee_project,
        idw_power=args.idw_power,
        kriging_variograms_path=args.kriging_variograms,
        prithvi_model_dir=args.prithvi_model_dir,
    )
    # Engine construction is infra (model/scaler/registry load) — surface failures
    # as a structured error too, not a raw traceback.
    try:
        engine = TFTInferenceEngine(cfg)
        result = engine.predict(
            lat=args.lat,
            lon=args.lon,
            date=args.date,
            station_code=args.station_code,
            leave_one_out=args.leave_one_out,
            details=args.details,
        )
    except Exception as e:  # init failure
        result = {"status": "error",
                  "error": {"code": "init_error", "message": f"{type(e).__name__}: {e}"}}

    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)

    raise SystemExit(1 if result.get("status") == "error" else 0)


if __name__ == "__main__":
    main()
