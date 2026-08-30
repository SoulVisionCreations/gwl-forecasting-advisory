"""build_variograms — slim the per-state variogram JSONs into one inference artifact.

Aditya's build_variogram.py writes one JSON per state under outputs/state_models/, each
holding a fitted `variogram` (kriging states) OR method="average" (small states), plus a
bulky `stations` list (build-time only). For inference we need just two things:

  1. {safe_state -> {"model", "params"}}   — the kriging variograms (20 of 26 states)
  2. (lat, lon, safe_state) labels         — to assign a query/neighbours to a state by
                                             nearest labelled station (tiny vs the JSONs)

This produces state_variograms.pkl, which StateVariograms.load() consumes. It is a
one-time, offline step — nothing here runs at inference time, and it does NOT touch any
prediction artifact.

    python -m inference.build_variograms \
        --state-models-dir <.../outputs/state_models> \
        --out <run_dir>/data/state_variograms.pkl
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def build(state_models_dir: str, out_path: str) -> dict:
    from inference.kriging import StateVariograms, _safe_state

    files = sorted(glob.glob(os.path.join(state_models_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"no *.json in {state_models_dir}")

    variograms: dict = {}
    lats: list = []
    lons: list = []
    states: list = []
    n_kriging = n_average = 0

    for fp in files:
        with open(fp) as f:
            m = json.load(f)
        state = _safe_state(m.get("state", os.path.splitext(os.path.basename(fp))[0]))

        vg = m.get("variogram")
        if m.get("method") == "kriging" and vg and vg.get("model") and vg.get("params"):
            variograms[state] = {"model": vg["model"], "params": vg["params"]}
            n_kriging += 1
        else:
            n_average += 1

        # collect (lat, lon, state) labels from EVERY state (incl. average ones, so a
        # query in a no-variogram region maps to that state → IDW fallback, not a wrong one)
        for s in m.get("stations", []):
            la, lo = s.get("lat"), s.get("lon")
            if la is None or lo is None:
                continue
            lats.append(float(la))
            lons.append(float(lo))
            states.append(state)

    sv = StateVariograms(variograms, lats, lons, states)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sv.save(out_path)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"states: {len(files)}  (kriging={n_kriging}, average/idw={n_average})")
    print(f"labelled stations: {len(lats):,}")
    print(f"variogram states: {sorted(variograms)}")
    print(f"wrote {out_path}  ({size_mb:.2f} MB)")
    return {"n_states": len(files), "n_kriging": n_kriging,
            "n_labels": len(lats), "out": out_path}


def main():
    ap = argparse.ArgumentParser(prog="inference.build_variograms")
    ap.add_argument("--state-models-dir", required=True,
                    help="dir of per-state variogram JSONs (build_variogram outputs/state_models)")
    ap.add_argument("--out", required=True, help="output state_variograms.pkl path")
    args = ap.parse_args()
    build(args.state_models_dir, args.out)


if __name__ == "__main__":
    main()
