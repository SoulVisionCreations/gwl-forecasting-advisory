"""
extract_gwl_stations.py
=======================
Extract unique (station_code, lat, lon) from data_with_flag_all.csv and
assign a filesystem-safe station_id for use in composite filenames.

Sanitization rules (deterministic, reversible via station_index.csv):
  - strip leading/trailing whitespace
  - spaces -> underscore
  - forward slashes -> double-dash  (KLM/01 -> KLM--01)
  - all other characters kept as-is (hyphens, dots, alphanumeric are safe)

Output files:
  gwl_stations.csv      -- station_code, safe_id, lat, lon, earliest_date, latest_date
  station_index.csv     -- station_code <-> safe_id lookup (same data, explicit name)

The safe_id is used in composite filenames:
  composite_<safe_id>_<year>_<H1|H2>.tif

Usage:
    python extract_gwl_stations.py \
        --csv "$GWL_DATA_CSV" \
        --output gwl_stations.csv
"""

import argparse
import os
import pandas as pd


def sanitize(station_code: str) -> str:
    """Deterministic station_code -> filesystem-safe id."""
    s = str(station_code).strip()
    s = s.replace("/", "--")      # forward slash -> double-dash
    s = s.replace(" ", "_")       # space -> underscore
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=os.environ.get("GWL_DATA_CSV"),
                   help="GWL dataset CSV (data_with_flag_all.csv); default: $GWL_DATA_CSV")
    p.add_argument("--output", default=os.environ.get("GWL_STATIONS_OUT", "gwl_stations.csv"),
                   help="output stations CSV; default: $GWL_STATIONS_OUT or ./gwl_stations.csv")
    p.add_argument("--station_col", default="station_code")
    p.add_argument("--lat_col",     default="latitude")
    p.add_argument("--lon_col",     default="longitude")
    p.add_argument("--date_col",    default="date")
    args = p.parse_args()
    if not args.csv:
        p.error("--csv is required (or set $GWL_DATA_CSV)")

    print(f"Reading: {args.csv}")
    df = pd.read_csv(
        args.csv,
        usecols=[args.station_col, args.lat_col, args.lon_col, args.date_col],
        low_memory=False,
    )
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    stations = (
        df.groupby(args.station_col)
        .agg(
            lat=(args.lat_col, "first"),
            lon=(args.lon_col, "first"),
            earliest_date=(args.date_col, "min"),
            latest_date=(args.date_col, "max"),
        )
        .reset_index()
        .rename(columns={args.station_col: "station_code"})
    )
    stations = stations.dropna(subset=["lat", "lon"])

    # Fix swapped lat/lon: India bounds lat 6-38, lon 66-98.
    # If a station has lat in [66,98] and lon in [6,38], the columns are swapped.
    INDIA_LAT = (6, 38)
    INDIA_LON = (66, 98)
    swapped = (
        stations["lat"].between(INDIA_LON[0], INDIA_LON[1]) &
        stations["lon"].between(INDIA_LAT[0], INDIA_LAT[1])
    )
    if swapped.any():
        print(f"Auto-correcting {swapped.sum()} stations with swapped lat/lon.")
        stations.loc[swapped, ["lat", "lon"]] = (
            stations.loc[swapped, ["lon", "lat"]].values
        )

    # Drop anything still outside India bounds after correction
    in_bounds = (
        stations["lat"].between(INDIA_LAT[0], INDIA_LAT[1]) &
        stations["lon"].between(INDIA_LON[0], INDIA_LON[1])
    )
    n_dropped = (~in_bounds).sum()
    if n_dropped:
        print(f"Dropping {n_dropped} stations still outside India bounds after correction.")
        stations = stations[in_bounds]

    # Assign safe_id and verify no collisions
    stations["safe_id"] = stations["station_code"].apply(sanitize)

    dupes = stations[stations.duplicated("safe_id", keep=False)]
    if len(dupes):
        print("\nWARNING: safe_id collisions detected:")
        print(dupes[["station_code", "safe_id"]].to_string())
        # Disambiguate by appending a counter suffix
        seen = {}
        new_ids = []
        for sid in stations["safe_id"]:
            if stations["safe_id"].eq(sid).sum() > 1:
                count = seen.get(sid, 0)
                new_ids.append(f"{sid}___{count}")
                seen[sid] = count + 1
            else:
                new_ids.append(sid)
        stations["safe_id"] = new_ids
        print("Resolved with numeric suffix (___N).")

    # Composite years: HLS starts ~2013; need lag-1 composite for each sample year
    HLS_START = 2013
    earliest_year = max(HLS_START, int(stations["earliest_date"].dt.year.min()))
    latest_year   = int(stations["latest_date"].dt.year.max()) - 1

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stations.to_csv(args.output, index=False)

    # Also write an explicit index file for lookup at data-prep time
    index_path = os.path.join(out_dir, "station_index.csv") if out_dir else "station_index.csv"
    stations[["station_code", "safe_id"]].to_csv(index_path, index=False)

    print(f"\n{len(stations):,} unique stations")
    print(f"  -> {args.output}")
    print(f"  -> {index_path}")
    print(f"\nStation name issues:")
    print(f"  spaces   : {stations['station_code'].str.contains(' ').sum()}")
    print(f"  slashes  : {stations['station_code'].str.contains('/').sum()}")
    print(f"\nSample mappings (first 10 with non-trivial safe_id):")
    changed = stations[stations["station_code"] != stations["safe_id"]].head(10)
    print(changed[["station_code", "safe_id"]].to_string(index=False))

    print(f"\nLat range : {stations.lat.min():.4f} .. {stations.lat.max():.4f}")
    print(f"Lon range : {stations.lon.min():.4f} .. {stations.lon.max():.4f}")
    print(f"Data range: {stations.earliest_date.min().date()} -> {stations.latest_date.max().date()}")

    year_list = list(range(earliest_year, latest_year + 1))
    n_years = len(year_list)
    n_composites = n_years * 2 * len(stations)
    print(f"\nRecommended --years for download_gwl_composites.py:")
    print(f"  {' '.join(str(y) for y in year_list)}")
    print(f"  {n_years} years x 2 halves x {len(stations):,} stations = {n_composites:,} composites")
    print(f"  (stations before {HLS_START} get zero embedding — HLS coverage starts ~2013)")


if __name__ == "__main__":
    main()
