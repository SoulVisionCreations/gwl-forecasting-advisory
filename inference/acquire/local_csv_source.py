"""LocalCsvSource — numeric inputs (GWL + dynamic + static) from the TRAINING CSV.

Drop-in alternative to NumericFetcher (same `.fetch(neighbours, current_date)` →
list[AcquiredStation] interface), for when the live WRIS endpoint is unreachable
from the run host, or when we want EXACT training parity. The neighbours are
trained stations, so every value they need is already in data_with_flag_all.csv —
the very file training ingested.

Parity by reuse: the column mapping is the canonical
  data_preparation.df_gwl_readings / df_dynamic_features / df_static_features,
and the GWL cleaning mirrors LSTMDataPreparation.load_gwl_readings (notna → sort →
abs() sign-normalization → min/max caps). Nothing is re-derived.

Extraction is a single grep over the (sorted-by-nothing) 3.8 GB CSV — ~1.5 s for
all K neighbours since station_code is column 1 and we anchor `^code,`. Results are
cached per station_code so the validation ground-truth lookup doesn't re-scan.
"""
from __future__ import annotations

import io
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from inference.types import Neighbour, AcquiredStation

_DYN_COLS = ["rainfall", "temp", "et", "runoff", "ndvi", "sm", "lulc"]


class LocalCsvSource:

    def __init__(self, csv_path: str, data_config):
        self.csv_path = csv_path
        self.data_config = data_config
        self._header = None
        self._cache: dict = {}   # station_code -> (gwl_df, dynamic_df, static_series)

    def fetch(self, neighbours: "list[Neighbour]", current_date: "datetime") -> "list[AcquiredStation]":
        from inference.types import AcquiredStation

        need = [n.station_code for n in neighbours if n.station_code not in self._cache]
        if need:
            self._load_codes(need)

        out: "list[AcquiredStation]" = []
        for n in neighbours:
            gwl, dyn, static = self._cache.get(n.station_code, (None, None, None))
            out.append(AcquiredStation(neighbour=n, gwl=gwl, dynamic=dyn, static=static, composite_path=None))
        return out

    # ------------------------------------------------------------------ internals
    def _header_line(self) -> str:
        if self._header is None:
            with open(self.csv_path) as f:
                self._header = f.readline().rstrip("\n")
        return self._header

    def _load_codes(self, codes: "list[str]") -> None:
        import pandas as pd

        from gwlcore.data_preparation import df_gwl_readings, df_dynamic_features, df_static_features

        pattern = "^(" + "|".join(re.escape(c) for c in codes) + "),"
        proc = subprocess.run(
            ["grep", "-E", pattern, self.csv_path],
            capture_output=True, text=True,
        )
        body = proc.stdout
        if not body.strip():
            for c in codes:
                self._cache[c] = (None, None, None)
            return

        # Force station_code to str: an all-numeric single-station batch is otherwise
        # inferred as int64 (leading zeros stripped), so the string filter below — and
        # the registry's string codes — match nothing (gwl=None). station_code is a
        # string id by design, so str is the correct dtype everywhere.
        raw = pd.read_csv(io.StringIO(self._header_line() + "\n" + body), low_memory=False,
                          dtype={"station_code": str})

        # Canonical column maps (the parity-critical transforms) reused verbatim.
        gwl_all = self._clean_gwl(df_gwl_readings(raw))
        dyn_all = df_dynamic_features(raw)
        sta_all = df_static_features(raw)

        for c in codes:
            g = gwl_all[gwl_all["station_code"] == c].copy()
            if len(g):
                g["date"] = pd.to_datetime(g["date"])
                g = g.set_index("date")
            d = dyn_all[dyn_all["station_code"] == c].copy()
            if len(d):
                d["date"] = pd.to_datetime(d["date"])
                d = d.set_index("date")[[col for col in _DYN_COLS if col in d.columns]]
            s = self._static_series(sta_all, c)
            self._cache[c] = (
                g if len(g) else None,
                d if len(d) else None,
                s,
            )

    def _clean_gwl(self, df):
        """Sort, then apply the canonical GWL cleaning shared with training and the
        WRIS/NWDP sources: data_preparation.clean_gwl_values (drop NaN → abs() sign-
        normalization → configured hard caps). Single source of truth = no drift."""
        from gwlcore.data_preparation import clean_gwl_values
        return clean_gwl_values(df.sort_values(["station_code", "date"]), self.data_config)

    def _static_series(self, sta_all, code: str):
        import pandas as pd

        rows = sta_all[sta_all["station_code"] == code]
        if not len(rows):
            return pd.Series({"station_code": code, "well_depth": 0.0})
        r = rows.iloc[0]

        def _f(key):
            try:
                v = r.get(key, 0.0)
                return float(v) if v == v and v is not None else 0.0   # NaN-safe
            except (TypeError, ValueError):
                return 0.0

        def _s(key):
            v = r.get(key, "no_data")
            return v if (isinstance(v, str) and v) else "no_data"

        return pd.Series({
            "station_code": code,
            "latitude": _f("latitude"),
            "longitude": _f("longitude"),
            "well_type": _s("well_type"),
            "well_depth": 0.0,                 # 'depth' is empty for all rows (well_depth unusable)
            "elevation": _f("elevation"),
            "stream_order": _f("stream_order"),
            "lithology": _s("lithology"),       # df_static renamed litho_lithologic -> lithology
            "litho_supergroup": _s("litho_supergroup"),
            "aquifer_type": _s("aquifer_type"), # df_static renamed well_aquifer_type -> aquifer_type
            "aquifer_0_aquifer": _s("aquifer_0_aquifer"),
        })
