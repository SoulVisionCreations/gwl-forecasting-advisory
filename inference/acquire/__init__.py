"""Live data acquisition for the K neighbour stations (per request).

LiveFetcher combines the two fetchers into one AcquiredStation list:
- NumericFetcher   : GWL history + dynamic (rain/temp/...) + static   (drives vendored new_data_fetcher)
- CompositeFetcher : the (safe_id, year-1, period) HLS tile           (reuse fetch_composite)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime

    from inference.types import Neighbour, AcquiredStation
    from inference.acquire.numeric_fetcher import NumericFetcher
    from inference.acquire.composite_fetcher import CompositeFetcher


class LiveFetcher:

    def __init__(self, numeric: "NumericFetcher", composite: "CompositeFetcher"):
        self.numeric = numeric
        self.composite = composite
        self._notes: "list[str]" = []

    def drain_notes(self) -> "list[str]":
        """Human notes from the last fetch (e.g. imagery rate-limiting) — surfaced in the response."""
        n = list(dict.fromkeys(self._notes))
        self._notes = []
        return n

    def fetch(self, neighbours: "list[Neighbour]", current_date: "datetime") -> "list[AcquiredStation]":
        """Fetch numeric inputs for all neighbours (one batched GEE call), then resolve each
        neighbour's composite tile CONCURRENTLY. Each tile is still built by the training-exact
        `fetch_composite` (per-station raster patches can't fold into one FeatureCollection call the
        way point-features do); parallelising the ~K downloads is the parity-preserving speedup on
        cache-miss requests (cache hits are just disk reads).

        Rate limits: the downloads hit GEE's HIGH-VOLUME endpoint (built for concurrent getDownloadURL),
        and `fetch_composite` already retries each tile up to 5x with exponential backoff (absorbs a
        transient 429 burst per-tile). Concurrency is capped modestly by GWL_COMPOSITE_WORKERS
        (default 6); set it to 1 for a fully SERIAL fallback if GEE ever pushes back hard."""
        import os
        from concurrent.futures import ThreadPoolExecutor

        acquired = self.numeric.fetch(neighbours, current_date)
        if not acquired:
            return acquired

        def _resolve(a):
            n = a.neighbour
            try:
                return self.composite.fetch_tile(n.safe_id, n.lat, n.lon, current_date)
            except Exception:                     # a bad tile never sinks the request
                return None                       # composite-missing -> soft warning -> zero_idx

        try:
            cap = int(os.environ.get("GWL_COMPOSITE_WORKERS", "6"))
        except ValueError:
            cap = 6
        workers = max(1, min(len(acquired), cap))
        drain = getattr(self.composite, "drain_rate_limits", None)

        n_rl = 0
        if workers == 1:
            paths = [_resolve(a) for a in acquired]
            n_rl = drain() if drain else 0
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                paths = list(pool.map(_resolve, acquired))   # order preserved -> correct station<->tile
            # Self-heal: a rate-limited burst -> retry the MISSING tiles SERIALLY (concurrency is the
            # cause, so serializing clears it). Logged so ops can see it happened.
            n_rl = drain() if drain else 0
            missing = [i for i, p in enumerate(paths) if p is None]
            if n_rl and missing:
                _log.warning("GEE rate-limited %d imagery download(s) this request; retrying %d missing "
                             "tile(s) serially. If this recurs, set GWL_COMPOSITE_WORKERS=1 (serial).",
                             n_rl, len(missing))
                for i in missing:
                    paths[i] = _resolve(acquired[i])
                if drain:
                    n_rl += drain()                          # include the serial-retry hits

        if n_rl:   # surface a response note (companion to the server WARNING logged above)
            still = sum(1 for p in paths if p is None)
            self._notes.append(
                f"imagery: {n_rl} satellite tile(s) hit an Earth Engine rate limit and were retried"
                + (f"; {still} still unavailable (zero-filled)." if still else " and recovered."))

        for a, path in zip(acquired, paths):
            a.composite_path = path
        return acquired
