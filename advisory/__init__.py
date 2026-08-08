"""Farmer-advisory layer (Phase A, deterministic) — built ON TOP of the forecaster.

The forecaster (inference/) produces change_m; this package turns that + interpolated
historical normals into a cropping advisory via a DETERMINISTIC rule engine. A small
language model (Phase B, phraser.py) only PHRASES the structured result — it never
decides. Lives on the `use_case` branch only; it REUSES the forecaster, never edits it.
See advisory_build_plan.txt / use_case_gwl.txt.
"""
