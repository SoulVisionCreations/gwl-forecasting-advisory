"""StationRegistry — known GWL stations + spatial KNN.

Source: gwl_stations.csv (station_code, lat, lon, earliest_date, latest_date,
safe_id). Resolves a query (lat,lon) to its K nearest known stations via
vectorised haversine distance; supports leave-one-out exclusion by station_code
(validation). Carries `safe_id` — needed downstream for composite lookup/download.
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inference.types import Neighbour

_EARTH_KM = 6371.0088


class StationRegistry:

    def __init__(self, registry_csv: str):
        import numpy as np
        import pandas as pd

        # sniff comma or tab (.csv or .tsv — AIKosh Model asset takes .tsv, not .csv);
        # keep station_code as str so all-numeric codes don't lose leading zeros.
        df = pd.read_csv(registry_csv, sep=None, engine="python", dtype={"station_code": str})
        self._code = df["station_code"].astype(str).to_numpy()
        self._safe = df["safe_id"].astype(str).to_numpy()
        self._lat = df["lat"].astype(float).to_numpy()
        self._lon = df["lon"].astype(float).to_numpy()
        self._lat_rad = np.radians(self._lat)
        self._lon_rad = np.radians(self._lon)

    def __len__(self) -> int:
        return len(self._code)

    def _row_of(self, station_code: str):
        import numpy as np

        hits = np.nonzero(self._code == str(station_code))[0]
        return int(hits[0]) if len(hits) else None

    def latlon_of(self, station_code: str) -> "Optional[tuple[float, float]]":
        """(lat, lon) of a known station, or None if not in the registry."""
        i = self._row_of(station_code)
        return (float(self._lat[i]), float(self._lon[i])) if i is not None else None

    def get(self, station_code: str) -> "Optional[Neighbour]":
        """The station itself as a Neighbour (distance 0), or None if unknown."""
        from inference.types import Neighbour

        i = self._row_of(station_code)
        if i is None:
            return None
        return Neighbour(
            station_code=str(self._code[i]),
            safe_id=str(self._safe[i]),
            lat=float(self._lat[i]),
            lon=float(self._lon[i]),
            distance_km=0.0,
        )

    def k_nearest(self, lat: float, lon: float, k: int, exclude: Optional[str] = None) -> "list[Neighbour]":
        """K nearest known stations to (lat,lon). `exclude` drops a station_code (leave-one-out)."""
        import numpy as np

        from inference.types import Neighbour

        d = self._haversine_km(lat, lon)
        order = np.argsort(d, kind="stable")
        ex = str(exclude) if exclude is not None else None

        out: "list[Neighbour]" = []
        for idx in order:
            code = str(self._code[idx])
            if ex is not None and code == ex:
                continue
            out.append(
                Neighbour(
                    station_code=code,
                    safe_id=str(self._safe[idx]),
                    lat=float(self._lat[idx]),
                    lon=float(self._lon[idx]),
                    distance_km=float(d[idx]),
                )
            )
            if len(out) >= k:
                break
        return out

    def _haversine_km(self, lat: float, lon: float):
        import numpy as np

        lat1 = np.radians(lat)
        lon1 = np.radians(lon)
        dlat = self._lat_rad - lat1
        dlon = self._lon_rad - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(self._lat_rad) * np.sin(dlon / 2.0) ** 2
        return 2.0 * _EARTH_KM * np.arcsin(np.sqrt(a))
