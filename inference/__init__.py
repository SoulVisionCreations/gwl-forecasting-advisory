"""GWL Prithvi+TFT inference package.

A THIN orchestration layer over the canonical training modules. Heavy logic
(sample building, scaling, model + projector, composite download) is REUSED from
`data_preparation` / `feature_scaler` / `tft_model` / `prithvi_finetune` /
`download_*composites*` — never re-implemented (single source of truth with
training). See memory: project_prithvi_inference_plan.

Entry points:
- `python -m inference --lat .. --lon .. --run-dir ..`   (CLI; see __main__.py)
- `TFTInferenceEngine.predict(lat, lon, date)`           (the reusable core; a
  future FastAPI app calls the exact same method)
"""
