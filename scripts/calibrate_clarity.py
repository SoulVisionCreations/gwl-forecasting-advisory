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
(+ n) so we can pick the best-separating one.

Normal basis (which "normal" the forecast is scored against — the two arms match our design discussion):
  - DEFAULT  SELF-normal (the station's OWN history) — the shipped FLOOR (clear ~74 / moderate ~71).
  - --neighbour  INTERPOLATED leave-one-out neighbour-normal (k nearest OTHER wells, IDW 1/d^2) — exactly
    what the live advisory does at an off-station plot; runs ~+5-8 pts higher (≈ clear 82 / moderate 78).
    Needs latitude/longitude in the gwl_csv (+ scipy); use --limit to subsample the heavier pass.

Run from the repo root:
    PYTHONPATH=. python scripts/calibrate_clarity.py --pkl <run_dir>/data/test_samples.pkl --horizon {3|6} --gwl_csv data/gwl_data.csv
    PYTHONPATH=. python scripts/calibrate_clarity.py ... --neighbour --limit 12500   # interpolated (production) basis
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


def load_station_coords(gwl_csv):
    """station_code -> (lat, lon), one per station — for the INTERPOLATED (neighbour) normal."""
    df = pd.read_csv(gwl_csv, usecols=["station_code", "latitude", "longitude"],
                     dtype={"station_code": str})
    df = df.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_code")
    return {str(r.station_code): (float(r.latitude), float(r.longitude))
            for r in df.itertuples(index=False)}


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


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
    ap.add_argument("--neighbour", action="store_true",
                    help="use the INTERPOLATED leave-one-out neighbour-normal (k nearest OTHER wells, IDW; "
                         "production-like, runs ~+5-8 pts higher) instead of the self-normal FLOOR")
    ap.add_argument("--k", type=int, default=8, help="neighbours for --neighbour (leave-one-out IDW)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #samples processed (handy for the heavier --neighbour pass; the earlier NN "
                         "validation used ~12500)")
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

    # INTERPOLATED (production-like) normal: leave-one-out over the k nearest OTHER wells, IDW(1/d^2) —
    # exactly what the live advisory does at an off-station plot. Overrides normal_band when --neighbour.
    if args.neighbour:
        from scipy.spatial import cKDTree
        _coords = load_station_coords(args.gwl_csv)
        _cc = [c for c in _coords if c in series]                 # stations with BOTH coords and history
        _arr = np.array([_coords[c] for c in _cc])
        _lr, _gr = np.radians(_arr[:, 0]), np.radians(_arr[:, 1])
        _xyz = np.c_[np.cos(_lr) * np.cos(_gr), np.cos(_lr) * np.sin(_gr), np.sin(_lr)]  # unit sphere -> exact NN
        _tree = cKDTree(_xyz)
        _idx_of = {c: i for i, c in enumerate(_cc)}
        print(f"    NEIGHBOUR (interpolated, leave-one-out) mode: k={args.k}, {len(_cc)} stations indexed")

        def normal_band(code, anchor):     # noqa: F811 — override with the LOO neighbour-normal
            key = (code, anchor.month, anchor.year)
            if key in cache:
                return cache[key]
            i = _idx_of.get(code)
            if i is None:
                cache[key] = (None, None, 0); return cache[key]
            _, nn = _tree.query(_xyz[i], k=args.k + 1)            # +1 so we can drop the station itself
            lat0, lon0 = _coords[code]
            codes, w, dist, ser = [], {}, {}, {}
            for j in np.atleast_1d(nn):
                c2 = _cc[int(j)]
                if c2 == code:                                    # leave-one-out: exclude the query station
                    continue
                km = float(_haversine_km(lat0, lon0, *_coords[c2]))
                w[c2] = 1.0 / max(km, 0.1) ** 2; dist[c2] = km; ser[c2] = series[c2]; codes.append(c2)
                if len(codes) >= args.k:
                    break
            if not codes:
                cache[key] = (None, None, 0); return cache[key]
            nb = NeighbourSet(codes=codes, weights=w, distances=dist, series=ser,
                              nearest_km=min(dist.values()), n_with_data=len(codes))
            nrm = NORM.compute(nb, anchor, n_years=args.n_years, gap_days=args.gap_days)
            res = (getattr(nrm, norm_attr), getattr(nrm, band_attr), getattr(nrm, ny_attr))
            cache[key] = res; return res

    # per-sample signals
    clar_ratio, abs_anom, abs_change, abs_actual = [], [], [], []
    reg3_ok, reg2_ok, dir_ok, states = [], [], [], []
    pred_reg, act_reg = [], []
    n_skip_hist = n_skip_norm = 0

    _samples = d if not args.limit else d[::max(1, len(d) // args.limit)][:args.limit]   # even, deterministic
    for s in _samples:
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
    _basis = "NEIGHBOUR interpolated (leave-one-out, ~production)" if args.neighbour else "SELF-normal FLOOR"
    print(f"\n=== usable samples: {n}   (skipped: no-history {n_skip_hist}, thin-normal {n_skip_norm})"
          f"   | normal basis: {_basis} ===")
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
        _b = ("NEIGHBOUR interpolated (leave-one-out, k=%d)" % args.k) if args.neighbour else "self-normal FLOOR"
        print("\n# ============ PASTE-READY → advisory/reliability.py (horizon %s) ============" % h)
        print(f"# calibrated on: {args.pkl}  |  basis: {_b}")
        print(f'#   _RELIABILITY["{h}"] = {{"clear": {clear}, "moderate": {mod}}}')
        print(f'#   DIRECTION_ACC_BY_PRED_MOVE["{h}"] = '
              f'{{"<0.5m": {d0}, "0.5-2m": {d1}, "2-5m": {d2}, "5m+": {d3}}}')
        if args.neighbour:
            print("# NOTE: these are the INTERPOLATED / production-like numbers (a few pts ABOVE the shipped "
                  "self-normal FLOOR). Only paste if deliberately switching reliability.py to the neighbour basis.")
        else:
            print("# NOTE: self-normal FLOOR (the interpolated/production basis via --neighbour runs ~+5-8 pts "
                  "higher). Sanity-check before committing.")


if __name__ == "__main__":
    main()
