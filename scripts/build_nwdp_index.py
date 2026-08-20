"""Build the NWDP station index (one-time setup, like build_variograms.py).

NWDP (National Water Data Portal, nwdp.nwic.gov.in) is a CKAN portal whose
groundwater-level datasets (CGWB + state departments, telemetry 6-hourly + manual
quarterly) are split per state, per mode, and per time-range CSV resource
(..._1991_2020, ..._2021_2025, ..._2026_2030). For our ~1-year lookback we only
ever need the 2021_2025 and 2026_2030 splits.

This script enumerates the current stations + their resource ids and writes a small
pickle the runtime NwdpGwlProvider loads to (a) map a query point to its NEAREST
NWDP station by coordinates and (b) know which resource(s) to datastore_search for
that station's series. No API key. Read-only.

Output pickle schema (nwdp_station_index.pkl):
  {
    'names':      [str, ...],           # NWDP "Station" value (exact, for filters)
    'lats':       [float, ...],
    'lons':       [float, ...],
    'states':     [str, ...],
    'modes':      [str, ...],           # 'telemetry' | 'manual'
    'group_keys': [str, ...],           # -> groups[key]
    'groups': { key: [[y0, y1, resource_id], ...] },   # time-range -> resource
  }

Run (on big-10, proxy in env):
  python -m inference.build_nwdp_index --out <run_dir>/data/nwdp_station_index.pkl
  # optional: --limit-datasets N (smoke), --modes telemetry manual
"""
from __future__ import annotations

import argparse
import pickle
import re
import time

_API = "https://nwdp.nwic.gov.in/api/3/action"
_YEAR_RES = re.compile(r"(19|20)\d{2}")   # find a 4-digit year in a resource name


def _get(session, action, params, retries=3, timeout=60):
    url = f"{_API}/{action}"
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            j = r.json()
            if not j.get("success"):
                raise RuntimeError(f"CKAN success=False: {str(j.get('error'))[:120]}")
            return j["result"]
        except Exception:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{action} failed after {retries} retries")


def _year_range(resource_name: str):
    """Extract (y0, y1) from a resource name like '... (2021 - 2025) ...' or
    '..._2026_2030.csv'. Returns None if not a 2021+/2026+ split we care about."""
    years = [int(m.group()) for m in _YEAR_RES.finditer(resource_name or "")]
    years = [y for y in years if 1990 <= y <= 2035]
    if len(years) >= 2:
        y0, y1 = min(years), max(years)
        return (y0, y1)
    return None


def _distinct_stations(session, resource_id):
    """All distinct (Station, Latitude, Longitude, State) rows in a resource."""
    res = _get(session, "datastore_search", {
        "resource_id": resource_id,
        "fields": "Station,Latitude,Longitude,State",
        "distinct": "true",
        "limit": 100000,
    })
    return res.get("records", []) if res else []


def build(out_path: str, limit_datasets: int = 0, modes=("telemetry", "manual")):
    import requests
    session = requests.Session()   # trusts *_proxy env vars (big-10 Squid)

    print("enumerating GWL datasets ...")
    ps = _get(session, "package_search", {"q": "Ground Water Level", "rows": 400})
    datasets = ps["results"]
    print(f"  {len(datasets)} datasets (of {ps['count']})")

    # key = "dataset|state|mode" (dataset MUST be in the key: several datasets — CGWB
    # + individual state departments — cover the same state/mode/year-range, so keying
    # on (state,mode) alone lets one dataset's resource OVERWRITE another's while its
    # stations remain, pairing a station with a resource that doesn't contain it).
    groups: dict = {}   # key -> {'state','mode','y':{(y0,y1):rid}, 'stations':{name:(lat,lon)}}
    processed = 0
    for ds in datasets:
        title = ds.get("title", "")
        ds_id = ds.get("name") or ds.get("id") or "ds"
        mode = "telemetry" if "Telemetry" in title else ("manual" if "Manual" in title else None)
        if mode is None or mode not in modes:
            continue
        for res in ds.get("resources", []):
            yr = _year_range(res.get("name", ""))
            if yr is None or yr[0] < 2021:      # only 2021_2025 / 2026_2030 (skip 1991_2020)
                continue
            rid = res["id"]
            recs = _distinct_stations(session, rid)
            if not recs:
                continue
            # one resource = one state; take the state from the records (majority non-empty)
            states = [r.get("State") for r in recs if r.get("State")]
            state = max(set(states), key=states.count) if states else "Unknown"
            key = f"{ds_id}|{state}|{mode}"
            g = groups.setdefault(key, {"state": state, "mode": mode, "y": {}, "stations": {}})
            g["y"][yr] = rid
            for r in recs:
                name = r.get("Station")
                try:
                    lat = float(r.get("Latitude")); lon = float(r.get("Longitude"))
                except (TypeError, ValueError):
                    continue
                if not name or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                g["stations"].setdefault(name, (lat, lon))
            print(f"    {state:22s} {mode:9s} {str(yr):14s} {len(recs):5d} stations  ({rid[:8]})")
        processed += 1
        if limit_datasets and processed >= limit_datasets:
            print(f"  (stopped at --limit-datasets {limit_datasets})")
            break

    # flatten
    names, lats, lons, states_l, modes_l, gkeys = [], [], [], [], [], []
    out_groups: dict = {}
    for key, g in groups.items():
        out_groups[key] = sorted([[y0, y1, rid] for (y0, y1), rid in g["y"].items()])
        for name, (lat, lon) in g["stations"].items():
            names.append(name); lats.append(lat); lons.append(lon)
            states_l.append(g["state"]); modes_l.append(g["mode"]); gkeys.append(key)

    index = {"names": names, "lats": lats, "lons": lons, "states": states_l,
             "modes": modes_l, "group_keys": gkeys, "groups": out_groups}
    with open(out_path, "wb") as f:
        pickle.dump(index, f)
    print(f"\nWROTE {out_path}: {len(names)} stations, {len(out_groups)} (state,mode) groups")
    tel = sum(1 for m in modes_l if m == "telemetry")
    print(f"  telemetry stations: {tel} | manual: {len(names) - tel}")


def main():
    ap = argparse.ArgumentParser(prog="inference.build_nwdp_index")
    ap.add_argument("--out", required=True, help="output nwdp_station_index.pkl path")
    ap.add_argument("--limit-datasets", type=int, default=0, help="smoke: process only N datasets")
    ap.add_argument("--modes", nargs="+", default=["telemetry", "manual"],
                    choices=["telemetry", "manual"])
    args = ap.parse_args()
    build(args.out, limit_datasets=args.limit_datasets, modes=tuple(args.modes))


if __name__ == "__main__":
    main()
