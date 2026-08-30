"""Assemble a self-contained GWL inference package folder.

Gathers the inference artifacts into ONE directory (dropping the bulky
train/val/test_samples.pkl), so the engine runs from a single --run-dir: the
engine auto-resolves prithvi_base/, gwl_stations.csv and data/station_stats.pkl
as siblings of run_dir.

    python -m inference.build_package \
        --run-dir     <trained run dir> \
        --prithvi-dir <Prithvi-EO-2.0-300M-TL dir> \
        --registry-csv <gwl_stations.csv> \
        --out         <package dir> [--link-prithvi]

Produces:
    <out>/best_model.pt                                   (model)
    <out>/config.json, experiment_summary.txt            (reference)
    <out>/data/scalers.pkl
    <out>/data/data_config_variables.pkl
    <out>/data/tile_manifest.pkl
    <out>/data/station_stats.pkl                          (built from train_samples if absent)
    <out>/data/nwdp_station_index.pkl                      (if present; enables --gwl-source nwdp)
    <out>/gwl_stations.tsv                                (registry; TSV, AIKosh Model-compatible)
    <out>/prithvi_base/                                   (Prithvi base; copy or --link-prithvi symlink)

Then run inference with just:  python -m inference --run-dir <out> ...
"""
from __future__ import annotations

import argparse
import os
import shutil


def _slim_ckpt(rd, prithvi_dir, registry_csv, out_path):
    """Write best_model.pt with the FROZEN Prithvi base stripped from the state_dict.

    The base (330.7M params) is public pretrained weights, byte-identical to what
    build_projector reloads from the Prithvi dir at load time. We keep only the
    trained delta — TFT + LoRA adapters + projector (~1.9M, ~8MB) — so the shipped
    ckpt shrinks 1.3GB -> ~8MB. The loaded model is IDENTICAL either way, so
    predictions and metrics are unchanged. Requires the Prithvi dir at load time.
    """
    import torch
    ckpt = torch.load(os.path.join(rd, "best_model.pt"), map_location="cpu", weights_only=False)
    if not ckpt.get("config", {}).get("use_prithvi"):
        torch.save(ckpt, out_path)          # baseline TFT-only: nothing to strip
        print("  (not a Prithvi model — copied full ckpt, no slimming)")
        return
    # Rebuild the model exactly as inference does, to read requires_grad truthfully.
    from inference.model.tft_loader import TFTModelRunner
    runner = TFTModelRunner.load(rd, device="cpu", model_dir=prithvi_dir, station_csv=registry_csv)
    trainable = {n for n, p in runner.model.named_parameters() if p.requires_grad}
    full_sd = ckpt["model_state_dict"]
    # Keep TFT (not under prithvi.), LoRA (.lora_), pre_norm/proj (prithvi.{pre_norm,proj}).
    # Drop only the frozen base under prithvi.encoder.* that isn't a LoRA adapter.
    keep = lambda k: (not k.startswith("prithvi.encoder.")) or (".lora_" in k)
    slim_sd = {k: v for k, v in full_sd.items() if keep(k)}
    dropped_trainable = trainable - set(slim_sd)
    if dropped_trainable:                   # safety: never drop a trained param
        raise RuntimeError(f"slim would drop trained params: {sorted(dropped_trainable)[:5]}")
    ckpt["model_state_dict"] = slim_sd
    ckpt["slim"] = True
    ckpt["slim_note"] = (f"frozen Prithvi base stripped; rebuilt from Prithvi dir at load. "
                         f"kept {len(slim_sd)}/{len(full_sd)} tensors.")
    torch.save(ckpt, out_path)
    print(f"  slim ckpt: kept {len(slim_sd)}/{len(full_sd)} tensors "
          f"(dropped {len(full_sd) - len(slim_sd)} frozen-base)")


def main():
    ap = argparse.ArgumentParser(prog="inference.build_package")
    ap.add_argument("--run-dir", required=True, help="trained run dir (best_model.pt + data/)")
    ap.add_argument("--prithvi-dir", required=True, help="Prithvi-EO-2.0-300M-TL dir")
    ap.add_argument("--registry-csv", required=True, help="gwl_stations.csv (station registry)")
    ap.add_argument("--out", required=True, help="output package dir")
    ap.add_argument("--link-prithvi", action="store_true",
                    help="symlink the ~1.3GB Prithvi base instead of copying it")
    ap.add_argument("--vendor-prithvi-arch", action="store_true",
                    help="ship ONLY prithvi_mae.py + config.json (the Prithvi arch), NOT the ~1.3G "
                         "base .pt — makes a FAT self-contained package: the baked-in backbone in "
                         "best_model.pt + the exclude_ckpt loader load with zero external Prithvi files.")
    ap.add_argument("--slim", action="store_true",
                    help="strip the frozen Prithvi base from best_model.pt (1.3GB -> ~8MB); "
                         "rebuilt from the Prithvi dir at load. Model/metrics unchanged.")
    ap.add_argument("--composites-dir", default=None,
                    help="optional: include a composites/ cache (offline/region package); "
                         "else the engine downloads tiles live from GEE into run_dir/composites")
    ap.add_argument("--link-composites", action="store_true",
                    help="symlink composites instead of copying")
    ap.add_argument("--variograms", default=None,
                    help="optional: existing state_variograms.pkl to include (enables kriging)")
    ap.add_argument("--state-models-dir", default=None,
                    help="optional: build state_variograms.pkl from this build_variogram outputs/state_models dir")
    args = ap.parse_args()

    rd, out = args.run_dir, args.out
    os.makedirs(os.path.join(out, "data"), exist_ok=True)

    # 1) model + reference files
    if args.slim:
        print("  slimming best_model.pt (strip frozen Prithvi base) ...")
        _slim_ckpt(rd, args.prithvi_dir, args.registry_csv, os.path.join(out, "best_model.pt"))
    else:
        shutil.copy(os.path.join(rd, "best_model.pt"), os.path.join(out, "best_model.pt"))
    # NB: experiment_summary.txt intentionally NOT copied — .txt is a Dataset format, not an
    # AIKosh Model format. Only Model-accepted formats (.json/.pt/.pkl/.tsv/.py) ship here.
    for f in ("config.json", "data_metadata.json"):
        src = os.path.join(rd, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, f))

    # 2) inference data artifacts (NOT the bulky train/val/test_samples.pkl)
    for f in ("scalers.pkl", "data_config_variables.pkl", "tile_manifest.pkl"):
        shutil.copy(os.path.join(rd, "data", f), os.path.join(out, "data", f))

    # 3) station_stats.pkl — build the slim table from train_samples if not present
    stats = os.path.join(rd, "data", "station_stats.pkl")
    if not os.path.exists(stats):
        from inference.prepare.station_stats import StationStats
        print("  building station_stats.pkl from train_samples.pkl ...")
        StationStats.from_train_samples(os.path.join(rd, "data", "train_samples.pkl")).save(stats)
    shutil.copy(stats, os.path.join(out, "data", "station_stats.pkl"))

    # 3b) kriging variograms (optional; enables the kriging alternative to IDW)
    vario_dst = os.path.join(out, "data", "state_variograms.pkl")
    vario_src = os.path.join(rd, "data", "state_variograms.pkl")
    if args.variograms:
        shutil.copy(args.variograms, vario_dst)
    elif args.state_models_dir:
        from build_variograms import build
        print("  building state_variograms.pkl from state_models ...")
        build(args.state_models_dir, vario_dst)
    elif os.path.exists(vario_src):
        shutil.copy(vario_src, vario_dst)

    # 3c) NWDP station index (optional; enables --gwl-source nwdp as a WRIS drop-in
    #     without a live index rebuild). Ship it if the run dir already has one.
    nwdp_src = os.path.join(rd, "data", "nwdp_station_index.pkl")
    if os.path.exists(nwdp_src):
        shutil.copy(nwdp_src, os.path.join(out, "data", "nwdp_station_index.pkl"))
        print("  packed nwdp_station_index.pkl")

    # 4) station registry -> gwl_stations.TSV (AIKosh Model asset rejects .csv; .tsv is accepted,
    #    and the engine/loader readers sniff comma-or-tab, so .csv still works elsewhere).
    import pandas as _pd
    _reg = _pd.read_csv(args.registry_csv, sep=None, engine="python", dtype={"station_code": str})
    _reg.to_csv(os.path.join(out, "gwl_stations.tsv"), sep="\t", index=False)
    print(f"  wrote gwl_stations.tsv ({len(_reg)} stations; TSV = AIKosh Model-compatible)")

    # 5) Prithvi base (copy or symlink)
    dst = os.path.join(out, "prithvi_base")
    if os.path.islink(dst) or os.path.exists(dst):
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            shutil.rmtree(dst)
    if args.vendor_prithvi_arch:
        # FAT self-contained: ship only the arch (prithvi_mae.py + config.json), no base .pt.
        os.makedirs(dst, exist_ok=True)
        for f in ("prithvi_mae.py", "config.json"):
            src = os.path.join(args.prithvi_dir, f)
            if not os.path.exists(src):
                raise SystemExit(f"--vendor-prithvi-arch: {f} not found in {args.prithvi_dir}")
            shutil.copy(src, os.path.join(dst, f))
        print("  vendored Prithvi arch only (prithvi_mae.py + config.json; NO 1.3G base .pt)")
    elif args.link_prithvi:
        os.symlink(os.path.abspath(args.prithvi_dir), dst)
    else:
        shutil.copytree(args.prithvi_dir, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 6) optional composites cache (engine auto-uses run_dir/composites)
    if args.composites_dir:
        cdst = os.path.join(out, "composites")
        if os.path.islink(cdst) or os.path.exists(cdst):
            os.unlink(cdst) if os.path.islink(cdst) else shutil.rmtree(cdst)
        if args.link_composites:
            os.symlink(os.path.abspath(args.composites_dir), cdst)
        else:
            shutil.copytree(args.composites_dir, cdst)

    # report
    def _sz(p):
        if os.path.islink(p):
            return "symlink"
        if os.path.isdir(p):
            t = sum(os.path.getsize(os.path.join(r, f))
                    for r, _, fs in os.walk(p) for f in fs)
        else:
            t = os.path.getsize(p)
        return f"{t / 1024 / 1024:.1f} MB"

    print(f"\nPackage assembled at: {out}")
    for rel in ("best_model.pt", "data/scalers.pkl", "data/data_config_variables.pkl",
                "data/tile_manifest.pkl", "data/station_stats.pkl", "data/state_variograms.pkl",
                "data/nwdp_station_index.pkl", "gwl_stations.tsv", "prithvi_base"):
        p = os.path.join(out, rel)
        print(f"  {rel:35s} {_sz(p) if os.path.exists(p) or os.path.islink(p) else 'MISSING'}")
    print(f"\nRun:  python -m inference --run-dir {out} --station-code <code> --local-csv <csv> ...")


if __name__ == "__main__":
    main()
