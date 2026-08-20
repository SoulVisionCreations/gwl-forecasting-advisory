"""CompositeFetcher — live HLS composite per neighbour, reusing the bulk download tool.

Resolves the leakage-free tile = (safe_id, year-1, same-quarter as current_date),
walking back to the latest available prior year (the rule training uses; documented
in download_gwl_quarterly_composites.py). REUSES `fetch_composite` from that script
(no duplication) — which is itself cache-first: if the tile already exists in
`cache_dir` (the pre-downloaded 2013-2024 set, or a prior request) it returns
'skip-exists' with no GEE call. So historical-date queries cost only disk lookups;
only genuinely new years (e.g. 2025 for a 2026 query) hit GEE. Returns the tile path,
or None when no imagery exists across the walk-back -> the sample gets zero_idx.

Defaults (patch_radius/patch_px/min_coverage) MUST match the values training's
composites were built with, else inference tiles won't match the training distribution.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

_log = logging.getLogger(__name__)

# Signals of a GEE rate-limit / quota error in fetch_composite's returned message.
_RATE_LIMIT_SIGNS = ("429", "too many requests", "rate limit", "rate_limit", "quota", "resource_exhausted")


def _looks_rate_limited(msg) -> bool:
    m = str(msg).lower()
    return any(s in m for s in _RATE_LIMIT_SIGNS)


class CompositeFetcher:

    def __init__(
        self,
        cache_dir: str,
        composite_period: str = "quarter",
        gee_project: Optional[str] = None,
        patch_radius: int = 3360,        # must match training download
        patch_px: int = 224,
        min_coverage: float = 0.05,
        walk_back_years: int = 12,       # composite set spans 2013-2024
    ):
        if composite_period != "quarter":
            raise NotImplementedError(
                f"composite_period={composite_period!r}: only 'quarter' wired so far "
                "(half-yearly would reuse download_gwl_composites.py analogously)."
            )
        self.cache_dir = cache_dir
        self.composite_period = composite_period
        self.gee_project = gee_project
        self.patch_radius = patch_radius
        self.patch_px = patch_px
        self.min_coverage = min_coverage
        self.walk_back_years = walk_back_years
        self._gee_ready = False
        import threading
        self._gee_lock = threading.Lock()   # fetch_tile may run on many threads (parallel composites)
        self._rl_lock = threading.Lock()
        self._rate_limits = 0               # GEE 429/quota hits since the last drain (observability)

    def fetch_tile(self, safe_id: str, lat: float, lon: float, current_date: datetime) -> Optional[str]:
        """Cached/downloaded (safe_id, year-1, quarter) tile path, or None (-> zero_idx)."""
        import os

        from gwlcore.download_gwl_quarterly_composites import quarter_for_date, fetch_composite

        os.makedirs(self.cache_dir, exist_ok=True)
        quarter = quarter_for_date(current_date.month)
        year = current_date.year - 1
        min_year = year - self.walk_back_years

        while year > min_year:
            path = os.path.join(self.cache_dir, f"composite_{safe_id}_{year}_{quarter}.tif")
            # cache hit -> return WITHOUT initialising GEE (the common historical case)
            if os.path.exists(path):
                return path
            # not cached -> download this (safe_id, year, quarter) live
            self._ensure_gee()
            status, name, _valid = fetch_composite(
                safe_id, lat, lon, year, quarter, self.cache_dir,
                self.patch_radius, self.patch_px, self.min_coverage,
            )
            if status in ("ok", "skip-exists"):
                return os.path.join(self.cache_dir, name + ".tif")
            # 'error' from a GEE rate-limit (after fetch_composite's own 5x backoff) -> record +
            # log it so it's VISIBLE (server can't act on an invisible 429). 'empty' = no imagery.
            if status == "error" and _looks_rate_limited(name):
                with self._rl_lock:
                    self._rate_limits += 1
                _log.warning("GEE rate-limited imagery for %s (%s); backed off + retried. If frequent, "
                             "lower GWL_COMPOSITE_WORKERS (=1 for serial).", safe_id, str(name)[-70:])
            # 'empty' or 'error' -> walk back a year
            year -= 1
        return None

    def drain_rate_limits(self) -> int:
        """Count of GEE rate-limit (429/quota) hits since the last call; resets the counter."""
        with self._rl_lock:
            n = self._rate_limits
            self._rate_limits = 0
            return n

    def _ensure_gee(self) -> None:
        if self._gee_ready:
            return
        with self._gee_lock:                 # double-checked: only ONE thread initialises GEE
            if self._gee_ready:
                return
            import ee

            from gwlcore.download_gwl_quarterly_composites import HIGH_VOLUME

            try:
                ee.Initialize(project=self.gee_project, opt_url=HIGH_VOLUME)
            except Exception:
                ee.Initialize(opt_url=HIGH_VOLUME)   # already initialised / implicit project
            self._gee_ready = True
