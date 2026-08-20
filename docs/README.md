# docs/

Package documentation.

- [`ADVISORY.md`](ADVISORY.md) — **the farmer crop/water advisory (the release use-case)**: complete from-scratch setup + run + API. **Start here to run the advisory.**
- [`MODEL_CARD.md`](MODEL_CARD.md) — architecture, data sources, usage, license & attribution, intended use
- [`SETUP.md`](SETUP.md) — install, pinned environment, per-user GEE OAuth, downloading the Model/Dataset assets
- [`INFERENCE.md`](INFERENCE.md) — CLI + FastAPI usage, request/response schema, anchor/date semantics, GWL sources
- [`DECISION_LAYER.md`](DECISION_LAYER.md) — how the advisory turns a forecast into advice (the pipeline, the rule table) + **how to customize** it
- [`RESPONSE_FIELDS.md`](RESPONSE_FIELDS.md) — **every response field explained, one by one** (numbers, outlook, confidence, crop_guidance, reason_code/details)
- [`RESPONSE_STATES.md`](RESPONSE_STATES.md) — every response state (ok / insufficient_data / error), its trigger, `reason_code` and an example
- [`TRAINING.md`](TRAINING.md) — reproduce / fine-tune (composites download, `run_big10`, champion config, packaging)
- [`DATA_SOURCES.md`](DATA_SOURCES.md) — live feeds (WRIS/NWDP, GEE, Open-Meteo) + shipped artefacts, freshness, licensing
- [`openapi.yaml`](openapi.yaml) — OpenAPI 3.0 spec for the advisory API (`/advisory` + `/health`)

New here? To run the **advisory** (the main use-case) → **ADVISORY.md**. For a raw forecast → **SETUP** → **INFERENCE**.
Start with **MODEL_CARD** for the what; **TRAINING** to re-train.
