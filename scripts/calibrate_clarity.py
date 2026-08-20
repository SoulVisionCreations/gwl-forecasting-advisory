#!/usr/bin/env python3
"""calibrate_clarity.py — DEV ANALYSIS (not shipped, no artifact produced).

Measures how well a per-query reliability signal predicts whether the advisory's
regime call (above/normal/below vs the seasonal normal) is CORRECT, using the
champion test-set predictions. NO model inference — reuses `predicted_gwl_delta`
already stored in test_samples.pkl. NO external services — local pkl + gwl_data.csv.

For each test sample (station S, anchor date, horizon h):
  predicted_delta = predicted_gwl_delta                 (raw m, post-clamp = served change_m)
  actual_delta    = target_gwl_raw - current_gwl        (raw m; == stored target_gwl)
  normal, band    = advisory.normals.compute on S's OWN history from gwl_data.csv
                    (1-well neighbour-set -> exact production math; self-normal, the
                     dominant term of the IDW normal at an on-station query)
  predicted_regime = bin_a(predicted_delta, normal, band)   # above/normal/below
  actual_regime    = bin_a(actual_delta,    normal, band)
  correct          = predicted_regime == actual_regime

Then it bins `correct` by a MENU of candidate signals and prints accuracy per bucket
(+ n) so we can pick the best-separating one. Run from the repo root:
    PYTHONPATH=. python scripts/calibrate_clarity.py \
        --pkl <run_dir>/data/test_samples.pkl --horizon {3|6} --gwl_csv data/gwl_data.csv
"""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

from advisory.neighbours import NeighbourSet
from advisory import normals as NORM
from advisory.rule_engine import bin_a


def load_station_series(gwl_csv):
    """station_code -> pd.Series(depth-to-water m = abs(gwl_value), date-indexed, sorted)."""
    df = pd.read_csv(gwl_csv, usecols=["station_code", "date", "gwl_value"],
                     dtype={"station_code": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "gwl_value"])
    df["depth"] = df["gwl_value"].abs()                       # advisory convention: depth-to-water >= 0
    out = {}
    for code, g in df.groupby("station_code"):
        s = pd.Series(g["depth"].values, index=g["date"].values)
        out[code] = s[~s.index.duplicated(keep="last")].sort_index()
    return out


def bucket_table(signal, correct, edges, labels):
    """Print n + accuracy per bucket for one candidate signal."""
    signal = np.asarray(signal, float)
    correct = np.asarray(correct, bool)
    rows = []
    for i, lab in enumerate(labels):
        m = (signal >= edges[i]) & (signal < edges[i + 1])
        n = int(m.sum())
        acc = float(correct[m].mean()) if n else float("nan")
        rows.append((lab, n, acc))
    spread = (max(r[2] for r in rows if r[1] > 30) - min(r[2] for r in rows if r[1] > 30)) \
        if sum(1 for r in rows if r[1] > 30) >= 2 else float("nan")
    for lab, n, acc in rows:
        print(f"    {lab:>10s} : n={n:7d}  acc={acc*100:5.1f}%")
    mono = all(rows[i][2] <= rows[i + 1][2] + 1e-9 for i in range(len(rows) - 1)
               if rows[i][1] > 30 and rows[i + 1][1] > 30)
    print(f"    -> spread(top-bottom, n>30) = {spread*100:4.1f} pts   monotone_up={mono}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--horizon", type=int, required=True, choices=[3, 6])
    ap.add_argument("--gwl_csv", default="data/gwl_data.csv")
    ap.add_argument("--n_years", type=int, default=10)
    ap.add_argument("--gap_days", type=int, default=45)
    ap.add_argument("--min_years", type=int, default=4, help="require >= this many seasonal years for a clean normal")
    ap.add_argument("--state_csv", default=None, help="optional state_performance_test.csv to merge r2_delta_median")
    ap.add_argument("--emit-constants", dest="emit_constants", action="store_true",
                    help="print a PASTE-READY block for advisory/reliability.py (for the training hook)")
    args = ap.parse_args()

    print(f"=== loading test samples: {args.pkl} ===")
    d = pickle.load(open(args.pkl, "rb"))
    print(f"    n samples: {len(d)}")
    print(f"=== loading station history: {args.gwl_csv} ===")
    series = load_station_series(args.gwl_csv)
    print(f"    stations with history: {len(series)}")

    norm_attr = "normal_seasonal" if args.horizon == 3 else "normal_seasonal_6m"
    band_attr = "band" if args.horizon == 3 else "band_6m"
    ny_attr = "n_years_seasonal"

    cache = {}   # (code, month, year) -> (normal, band, n_years)

    def normal_band(code, anchor):
        key = (code, anchor.month, anchor.year)
        if key in cache:
            return cache[key]
        s = series.get(code)
        if s is None or not len(s):
            cache[key] = (None, None, 0)
            return cache[key]
        nb = NeighbourSet(codes=[code], weights={code: 1.0}, distances={code: 0.0},
                          series={code: s}, nearest_km=0.0, n_with_data=1)
        nrm = NORM.compute(nb, anchor, n_years=args.n_years, gap_days=args.gap_days)
        res = (getattr(nrm, norm_attr), getattr(nrm, band_attr), getattr(nrm, ny_attr))
        cache[key] = res
        return res

    # per-sample signals
    clar_ratio, abs_anom, abs_change, abs_actual = [], [], [], []
    reg3_ok, reg2_ok, dir_ok, states = [], [], [], []
    pred_reg, act_reg = [], []
    n_skip_hist = n_skip_norm = 0

    for s in d:
        code = str(s["station_code"])
        anchor = pd.Timestamp(s["current_date"]).to_pydatetime()
        pdl = s["predicted_gwl_delta"]
        adl = s["target_gwl_raw"] - s["current_gwl"]
        normal, band, nyr = normal_band(code, anchor)
        if normal is None or band is None:
            n_skip_hist += 1
            continue
        if nyr < args.min_years:
            n_skip_norm += 1
            continue
        preg, _ = bin_a(pdl, normal, band)
        areg, _ = bin_a(adl, normal, band)
        if preg is None or areg is None:
            n_skip_norm += 1
            continue
        clar_ratio.append(abs(pdl - normal) / band)
        abs_anom.append(abs(pdl - normal))
        abs_change.append(abs(pdl))
        abs_actual.append(abs(adl))
        reg3_ok.append(preg == areg)                                     # 3-way regime match
        # 2-way: which side of normal (drop the deadzone) — sign(pred-normal)==sign(actual-normal)
        reg2_ok.append(np.sign(pdl - normal) == np.sign(adl - normal))
        dir_ok.append((np.sign(pdl) if pdl != 0 else 1) == (np.sign(adl) if adl != 0 else 1))
        pred_reg.append(preg); act_reg.append(areg)
        states.append(s.get("state"))

    n = len(reg3_ok)
    print(f"\n=== usable samples: {n}   (skipped: no-history {n_skip_hist}, thin-normal {n_skip_norm}) ===")
    if n == 0:
        print("NO usable samples — abort."); return
    print(f"OVERALL: 3-way regime {np.mean(reg3_ok)*100:.1f}% | 2-way above/below vs normal "
          f"{np.mean(reg2_ok)*100:.1f}% | raw direction {np.mean(dir_ok)*100:.1f}%")
    from collections import Counter
    print(f"predicted regime mix: {dict(Counter(pred_reg))}   actual regime mix: {dict(Counter(act_reg))}")

    print(f"\n########## A) clarity ratio |pred-normal|/band  vs  3-WAY regime ##########")
    bucket_table(clar_ratio, reg3_ok, [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf], ["<0.5","0.5-1","1-1.5","1.5-2","2-3","3+"])
    print(f"\n########## B) clarity ratio  vs  2-WAY above/below-normal ##########")
    bucket_table(clar_ratio, reg2_ok, [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf], ["<0.5","0.5-1","1-1.5","1.5-2","2-3","3+"])
    print(f"\n########## C) |PREDICTED change| (m)  vs  raw DIRECTION  (KEY: inference-available) ##########")
    bucket_table(abs_change, dir_ok, [0, 0.5, 2.0, 5.0, np.inf], ["<0.5","0.5-2","2-5","5+"])
    print(f"\n########## D) |ACTUAL change| (m)  vs  raw DIRECTION  (reproduces the known table) ##########")
    bucket_table(abs_actual, dir_ok, [0, 0.5, 2.0, 5.0, np.inf], ["<0.5","0.5-2","2-5","5+"])
    print(f"\n########## E) |PREDICTED change| (m)  vs  2-WAY above/below-normal ##########")
    bucket_table(abs_change, reg2_ok, [0, 0.5, 2.0, 5.0, np.inf], ["<0.5","0.5-2","2-5","5+"])

    print(f"\n########## F) per-STATE reliability (>=200 samples): 2-way above/below | direction | per-well r2d ##########")
    st_r2 = {}
    if args.state_csv:
        try:
            sp = pd.read_csv(args.state_csv)
            st_r2 = dict(zip(sp["state"].astype(str), sp["r2_delta_median"]))
        except Exception as e:
            print("    (state_perf csv not read:", e, ")")
    agg = defaultdict(lambda: [0, 0, 0])   # n, reg2_ok, dir_ok
    for st, r2c, dc in zip(states, reg2_ok, dir_ok):
        a = agg[st]; a[0] += 1; a[1] += int(r2c); a[2] += int(dc)
    print(f"    {'state':>18s} {'n':>6s} {'2way%':>7s} {'dir%':>7s} {'r2d_med':>8s}")
    rows = [(st, c, r2ok, dok) for st, (c, r2ok, dok) in agg.items() if c >= 200]
    for st, cnt, r2ok, dok in sorted(rows, key=lambda r: -r[2] / max(r[1], 1)):   # sort by 2-way acc
        r2d = st_r2.get(str(st))
        r2s = "" if r2d is None else f"{r2d:.3f}"
        print(f"    {str(st)[:18]:>18s} {cnt:6d} {r2ok/cnt*100:7.1f} {dok/cnt*100:7.1f} {r2s:>8s}")
    v2 = [r2ok / cnt * 100 for _, cnt, r2ok, _ in rows]
    vd = [dok / cnt * 100 for _, cnt, _, dok in rows]
    if len(v2) >= 2:
        print(f"    -> across states (n>=200): 2-way spread {max(v2)-min(v2):.1f} pts "
              f"[{min(v2):.0f}-{max(v2):.0f}]   direction spread {max(vd)-min(vd):.1f} pts [{min(vd):.0f}-{max(vd):.0f}]")

    if args.emit_constants:
        rr = np.asarray(clar_ratio); r2 = np.asarray(reg2_ok, bool)
        dch = np.asarray(abs_change); do = np.asarray(dir_ok, bool)
        def _acc(sig, ok, lo, hi):
            m = (sig >= lo) & (sig < hi)
            return int(round(ok[m].mean() * 100)) if m.sum() > 30 else None
        h = f"{args.horizon}m"
        clear, mod = _acc(rr, r2, 2.0, np.inf), _acc(rr, r2, 1.0, 2.0)
        d0, d1, d2, d3 = (_acc(dch, do, 0, 0.5), _acc(dch, do, 0.5, 2.0),
                          _acc(dch, do, 2.0, 5.0), _acc(dch, do, 5.0, np.inf))
        print("\n# ============ PASTE-READY → advisory/reliability.py (horizon %s) ============" % h)
        print(f"# calibrated on: {args.pkl}")
        print(f'#   _RELIABILITY["{h}"] = {{"clear": {clear}, "moderate": {mod}}}')
        print(f'#   DIRECTION_ACC_BY_PRED_MOVE["{h}"] = '
              f'{{"<0.5m": {d0}, "0.5-2m": {d1}, "2-5m": {d2}, "5m+": {d3}}}')
        print("# NOTE: self-normal FLOOR (production/neighbour-normal runs ~+5-8 pts higher). "
              "Sanity-check before committing.")


if __name__ == "__main__":
    main()
