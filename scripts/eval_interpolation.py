"""Evaluate IDW vs kriging interpolation (+ persistence baseline) at scale.

Leave-one-out over N stations × M dates: for each (station, date) we predict at the
station's location from its NEIGHBOURS ONLY, then compare the IDW and kriging forecasts
to the station's stored true target GWL. IDW & kriging share the same current baseline
(current_gwl_m), so they differ only in the predicted change — meaning the discriminating
metrics are head-to-head win-rate, MAE, and delta-R² (pooled absolute R² is baseline-
dominated and near-identical for both; reported for reference). Persistence (Δ=0, i.e.
current_gwl_m) is included so we know whether either interpolation of the delta adds value.

Subcommands:
  build   : pick samples from test/val_samples.pkl → samples.pkl
  worker  : run engine.predict over one shard → shard CSV   (own engine → <=1 GEE call)
  metrics : compute the comparison from the merged CSV
  run     : build → fork W workers (default 4, GEE cap) → merge → metrics  [one entry point]

Engine reuse: worker builds a TFTInferenceEngine (same core the CLI/API use) and calls
predict(station_code, leave_one_out=True). No new inference logic.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import subprocess
import sys
from collections import defaultdict


# ───────────────────────────── build ─────────────────────────────
def cmd_build(args):
    """Select up to n_stations distinct stations × up to n_dates dates each, from
    test_samples.pkl (+ val_samples.pkl), current_date in the requested years."""
    years = {int(y) for y in args.years.split(",")}
    pool = []
    for name in ("test_samples.pkl", "val_samples.pkl"):
        p = os.path.join(args.samples_dir, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                pool.extend(pickle.load(f))
    if not pool:
        raise SystemExit(f"no test/val_samples.pkl under {args.samples_dir}")

    by_station: "dict[str, dict]" = defaultdict(dict)   # code -> {date_str: sample}
    for s in pool:
        code = s.get("station_code"); cd = s.get("current_date")
        tgt = s.get("target_gwl_raw", s.get("target_gwl"))
        if not code or cd is None or tgt is None:
            continue
        yr = cd.year if hasattr(cd, "year") else int(str(cd)[:4])
        if yr not in years:
            continue
        date = cd.strftime("%Y-%m-%d") if hasattr(cd, "strftime") else str(cd)[:10]
        rec = {
            "station_code": code, "date": date,
            "target_gwl": float(tgt),
            "current_gwl": float(s.get("current_gwl_raw", s.get("current_gwl", 0.0)) or 0.0),
            "state": s.get("state", "unknown"),
        }
        by_station[code].setdefault(date, rec)   # dedup dates per station

    # select stations (>= min_dates); balanced = round-robin across states so no single
    # state dominates, else most-dates-first (which skews to well-sampled southern states)
    eligible = {c: d for c, d in by_station.items() if len(d) >= args.min_dates}
    picks = []
    if getattr(args, "balanced", False):
        by_state: "dict[str, list]" = defaultdict(list)
        for code, dmap in eligible.items():
            by_state[next(iter(dmap.values()))["state"]].append((code, dmap))
        for st in by_state:
            by_state[st].sort(key=lambda kv: -len(kv[1]))     # best-sampled wells first
        idx, states = {st: 0 for st in by_state}, sorted(by_state)
        while len(picks) < args.n_stations:
            added = False
            for st in states:
                if idx[st] < len(by_state[st]):
                    code, dmap = by_state[st][idx[st]]; idx[st] += 1
                    picks.append((code, sorted(dmap.values(), key=lambda r: r["date"])[: args.n_dates]))
                    added = True
                    if len(picks) >= args.n_stations:
                        break
            if not added:
                break
    else:
        for code, dmap in sorted(eligible.items(), key=lambda kv: -len(kv[1])):
            picks.append((code, sorted(dmap.values(), key=lambda r: r["date"])[: args.n_dates]))
            if len(picks) >= args.n_stations:
                break

    samples = [r for _, dates in picks for r in dates]
    with open(args.out, "wb") as f:
        pickle.dump(samples, f)
    n_states = len({r["state"] for r in samples})
    print(f"built {len(samples)} samples: {len(picks)} stations "
          f"(>= {args.min_dates} dates, <= {args.n_dates} each), {n_states} states → {args.out}")
    # quick per-state count
    sc = defaultdict(int)
    for r in samples:
        sc[r["state"]] += 1
    top = sorted(sc.items(), key=lambda kv: -kv[1])[:10]
    print("  top states:", ", ".join(f"{s}={n}" for s, n in top))


# ───────────────────────────── worker ─────────────────────────────
def _build_engine(args):
    from inference.config import InferenceConfig
    from inference.engine import TFTInferenceEngine
    cfg = InferenceConfig(
        run_dir=args.package_dir,
        local_csv=args.local_csv or None,
        composite_cache_dir=args.composite_cache_dir or None,
        device=args.device,
        k_neighbours=args.k,
    )
    return TFTInferenceEngine(cfg)


_FIELDS = ["station_code", "date", "state", "target_gwl", "current_gwl_m",
           "idw_gwl", "idw_abs_err", "kriging_gwl", "kriging_abs_err",
           "persistence_abs_err", "n_wells", "kriging_engaged", "status"]


def cmd_worker(args):
    with open(args.samples, "rb") as f:
        samples = pickle.load(f)
    shard = samples[args.shard:: args.nshards]
    print(f"[worker {args.shard}/{args.nshards}] {len(shard)} samples", flush=True)
    eng = _build_engine(args)
    print(f"[worker {args.shard}] engine ready", flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS); w.writeheader()
        for i, s in enumerate(shard):
            code, date, tgt = s["station_code"], s["date"], s["target_gwl"]
            try:
                # details=True: absolute forecast/current/kriging/n_wells live under `details`
                # now (default response = change_m only)
                r = eng.predict(station_code=code, leave_one_out=True, date=date, details=True)
            except Exception as e:  # noqa: BLE001
                w.writerow({"station_code": code, "date": date, "state": s.get("state"),
                            "target_gwl": tgt, "status": f"exc:{type(e).__name__}"})
                continue
            det = r.get("details") or {}
            idw = det.get("forecast_gwl_m"); krig = det.get("kriging_gwl_m")
            cur = det.get("current_gwl_m")
            w.writerow({
                "station_code": code, "date": date, "state": s.get("state"),
                "target_gwl": round(tgt, 4),
                "current_gwl_m": None if cur is None else round(cur, 4),
                "idw_gwl": None if idw is None else round(idw, 4),
                "idw_abs_err": None if idw is None else round(abs(idw - tgt), 4),
                "kriging_gwl": None if krig is None else round(krig, 4),
                "kriging_abs_err": None if krig is None else round(abs(krig - tgt), 4),
                "persistence_abs_err": None if cur is None else round(abs(cur - tgt), 4),
                "n_wells": det.get("n_wells_used"),
                "kriging_engaged": krig is not None,
                "status": r.get("status"),
            })
            if (i + 1) % 200 == 0:
                print(f"[worker {args.shard}] {i+1}/{len(shard)}", flush=True)
    print(f"[worker {args.shard}] done → {args.out}", flush=True)


# ───────────────────────────── metrics ─────────────────────────────
def _r2(pred, true):
    import numpy as np
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return None if ss_tot <= 1e-9 else 1.0 - ss_res / ss_tot


def cmd_metrics(args):
    import numpy as np
    rows = list(csv.DictReader(open(args.csv)))
    def fnum(v):
        return None if v in ("", "None", None) else float(v)
    for r in rows:
        for k in ("target_gwl", "current_gwl_m", "idw_gwl", "idw_abs_err",
                  "kriging_gwl", "kriging_abs_err", "persistence_abs_err"):
            r[k] = fnum(r.get(k))

    ok = [r for r in rows if r["idw_gwl"] is not None]
    keng = [r for r in ok if r["kriging_engaged"] in ("True", True) and r["kriging_gwl"] is not None]
    print(f"rows={len(rows)}  with_prediction={len(ok)}  kriging_engaged={len(keng)} "
          f"({100*len(keng)/max(len(ok),1):.0f}%)")

    def err_stats(rows_, key):
        e = [r[key] for r in rows_ if r[key] is not None]
        if not e: return None
        e = np.asarray(e)
        return dict(n=len(e), mae=float(e.mean()), median=float(np.median(e)),
                    rmse=float(np.sqrt((e**2).mean())), p90=float(np.percentile(e, 90)),
                    mx=float(e.max()))

    print("\n── pooled error (all predictions) ──")
    for label, key, data in [("IDW", "idw_abs_err", ok),
                             ("KRIGING", "kriging_abs_err", ok),
                             ("PERSISTENCE", "persistence_abs_err", ok)]:
        st = err_stats(data, key)
        if st:
            print(f"  {label:11s} n={st['n']:6d}  MAE={st['mae']:.3f}  median={st['median']:.3f}  "
                  f"RMSE={st['rmse']:.3f}  p90={st['p90']:.3f}  max={st['mx']:.2f}")

    print("\n── on the KRIGING-ENGAGED subset (where they differ) ──")
    for label, key in [("IDW", "idw_abs_err"), ("KRIGING", "kriging_abs_err"),
                       ("PERSISTENCE", "persistence_abs_err")]:
        st = err_stats(keng, key)
        if st:
            print(f"  {label:11s} n={st['n']:6d}  MAE={st['mae']:.3f}  median={st['median']:.3f}  RMSE={st['rmse']:.3f}")
    # head-to-head on engaged subset
    hh = [(r["idw_abs_err"], r["kriging_abs_err"]) for r in keng
          if r["idw_abs_err"] is not None and r["kriging_abs_err"] is not None]
    kw = sum(1 for i, k in hh if k < i - 1e-9); iw = sum(1 for i, k in hh if i < k - 1e-9)
    print(f"  head-to-head: kriging better {kw} ({100*kw/max(len(hh),1):.0f}%) | "
          f"idw better {iw} ({100*iw/max(len(hh),1):.0f}%) | ties {len(hh)-kw-iw}")

    # pooled R² (absolute) + delta-R²
    print("\n── R² (all predictions) ──")
    for label, gk in [("IDW", "idw_gwl"), ("KRIGING", "kriging_gwl")]:
        rr = [r for r in ok if r[gk] is not None and r["current_gwl_m"] is not None]
        r2_abs = _r2([r[gk] for r in rr], [r["target_gwl"] for r in rr])
        pd_ = [r[gk] - r["current_gwl_m"] for r in rr]
        ad_ = [r["target_gwl"] - r["current_gwl_m"] for r in rr]
        r2_delta = _r2(pd_, ad_)
        print(f"  {label:8s} pooled R²(abs)={r2_abs:.4f}  R²(Δ, change)={r2_delta:.4f}  (n={len(rr)})")

    # per-station median/mean R²
    print("\n── per-station R² (median / mean across wells with >=3 dates) ──")
    by_st = defaultdict(list)
    for r in ok:
        by_st[r["station_code"]].append(r)
    for label, gk in [("IDW", "idw_gwl"), ("KRIGING", "kriging_gwl")]:
        per = []
        for code, rs in by_st.items():
            rs2 = [r for r in rs if r[gk] is not None]
            if len(rs2) < 3: continue
            v = _r2([r[gk] for r in rs2], [r["target_gwl"] for r in rs2])
            if v is not None: per.append(v)
        if per:
            per = np.asarray(per)
            print(f"  {label:8s} median={float(np.median(per)):.4f}  mean={float(per.mean()):.4f}  (n_wells={len(per)})")

    # per-state MAE
    print("\n── per-state MAE (idw vs kriging, engaged subset, top 12 by count) ──")
    st_rows = defaultdict(list)
    for r in keng:
        st_rows[r["state"]].append(r)
    for state, rs in sorted(st_rows.items(), key=lambda kv: -len(kv[1]))[:12]:
        ie = np.mean([r["idw_abs_err"] for r in rs if r["idw_abs_err"] is not None])
        ke = np.mean([r["kriging_abs_err"] for r in rs if r["kriging_abs_err"] is not None])
        win = "KRIG" if ke < ie else "IDW"
        print(f"  {state[:22]:22s} n={len(rs):5d}  idw={ie:.2f}  krig={ke:.2f}  → {win}")


# ───────────────────────────── run (orchestrate) ─────────────────────────────
def cmd_run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    samples_pkl = os.path.join(args.out_dir, "samples.pkl")
    # 1) build
    cmd_build(argparse.Namespace(samples_dir=args.samples_dir, years=args.years,
                                 n_stations=args.n_stations, n_dates=args.n_dates,
                                 min_dates=args.min_dates, out=samples_pkl,
                                 balanced=args.balanced))
    # 2) fork W workers (each own engine → <=W concurrent GEE calls)
    procs, shard_csvs = [], []
    for i in range(args.workers):
        out_csv = os.path.join(args.out_dir, f"shard_{i}.csv")
        shard_csvs.append(out_csv)
        cmd = [sys.executable, "-m", "inference.eval_interpolation", "worker",
               "--samples", samples_pkl, "--shard", str(i), "--nshards", str(args.workers),
               "--out", out_csv, "--package-dir", args.package_dir, "--device", args.device,
               "--k", str(args.k)]
        if args.local_csv: cmd += ["--local-csv", args.local_csv]
        if args.composite_cache_dir: cmd += ["--composite-cache-dir", args.composite_cache_dir]
        log = open(os.path.join(args.out_dir, f"shard_{i}.log"), "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
        print(f"launched worker {i} → {out_csv}")
    rc = [p.wait() for p in procs]
    print("workers exit codes:", rc)
    # 3) merge
    merged = os.path.join(args.out_dir, "eval.csv")
    with open(merged, "w", newline="") as out:
        w = None
        for cf in shard_csvs:
            if not os.path.exists(cf): continue
            for r in csv.DictReader(open(cf)):
                if w is None:
                    w = csv.DictWriter(out, fieldnames=list(r.keys())); w.writeheader()
                w.writerow(r)
    print(f"merged → {merged}")
    # 4) metrics
    cmd_metrics(argparse.Namespace(csv=merged))


def main():
    ap = argparse.ArgumentParser(prog="inference.eval_interpolation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--samples-dir", required=True); b.add_argument("--out", required=True)
    b.add_argument("--n-stations", type=int, default=2000); b.add_argument("--n-dates", type=int, default=15)
    b.add_argument("--min-dates", type=int, default=3); b.add_argument("--years", default="2024,2025")
    b.add_argument("--balanced", action="store_true", help="round-robin across states (balanced), not most-dates-first")
    b.set_defaults(func=cmd_build)

    w = sub.add_parser("worker")
    w.add_argument("--samples", required=True); w.add_argument("--out", required=True)
    w.add_argument("--shard", type=int, required=True); w.add_argument("--nshards", type=int, required=True)
    w.add_argument("--package-dir", required=True); w.add_argument("--local-csv", default=None)
    w.add_argument("--composite-cache-dir", default=None); w.add_argument("--device", default="cuda")
    w.add_argument("--k", type=int, default=10)
    w.set_defaults(func=cmd_worker)

    m = sub.add_parser("metrics"); m.add_argument("--csv", required=True); m.set_defaults(func=cmd_metrics)

    r = sub.add_parser("run")
    r.add_argument("--samples-dir", required=True); r.add_argument("--package-dir", required=True)
    r.add_argument("--out-dir", required=True); r.add_argument("--local-csv", default=None)
    r.add_argument("--composite-cache-dir", default=None); r.add_argument("--device", default="cuda")
    r.add_argument("--workers", type=int, default=4); r.add_argument("--k", type=int, default=10)
    r.add_argument("--n-stations", type=int, default=2000); r.add_argument("--n-dates", type=int, default=15)
    r.add_argument("--min-dates", type=int, default=3); r.add_argument("--years", default="2024,2025")
    r.add_argument("--balanced", action="store_true", help="round-robin across states (balanced state mix)")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
