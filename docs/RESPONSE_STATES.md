# Response states & failure modes

Every `/advisory` response carries a top-level **`status`**: `ok`, `insufficient_data`, or `error`.
HTTP is **200** for all of them **except** a service that failed to load (**503**). An `ok` response can
still be **degraded** (thin confidence / poor freshness / an estimate rather than a live forecast) — so
always read `confidence` and the message's `Basis:` line, not just `status`.

This page lists every non-clean output, its trigger, the `reason_code` + `reason`/`error` you'll see, and
an example. Every field on a response is explained one-by-one in [RESPONSE_FIELDS.md](RESPONSE_FIELDS.md).

---

## A. `status: "ok"` — a forecast was produced

| | When | What you see |
|---|---|---|
| **A1 — Live forecast** | ≥1 well within 15 km has a **recent live reading**, the model runs, and full normals (seasonal + year‑over‑year) exist | `numbers.forecast_change_m` (3m) + `_6m`, `outlook` (`water_trend` + `vs_normal`), `crop_guidance`, `long_crop`, `confidence.level = well-supported`, **no** `Basis:` line. `verbose` adds `last_year` + the seasonal normals + `confidence.freshness = good` |
| **A2 — Analog estimate** | No neighbour has a usable **live** reading (the model can't run), **but** the normals hold **≥ 2 years** of seasonal history | `status: ok`, `confidence.level = "thin"` (hard‑capped), `freshness = poor`, and the message ends with a **`Basis:`** line ("seasonal estimate from recent years … treat as indicative, not a live forecast") |
| **A3 — Regional read** | the nearest well is **15–40 km** away (no closer well) — still answered, but as a broad *regional* read, not a local forecast | `status: ok`, `confidence.level = "thin"` (hard‑capped), a **"Heads‑up: nearest monitored well is N km away … regional read"** line in `message`; the forecast is plausibility‑clamped to the local band |

- **A1 example:** `{"lat":12.9716,"lon":77.5946}` (Bengaluru) → live forecast, freshness `good`.
- **A2 example:** `{"lat":18.2711,"lon":78.9617}` (Telangana) → `ok`, analog 3‑month ≈ −4.8 m, `thin`.
- **A2 variant — seasonal‑only:** a well with ≥2 yr seasonal history but no year‑over‑year pair returns the
  estimate with the **"Past year"** line omitted. Example `{"lat":23.2506,"lon":88.47}` (West Bengal).
- **A3 example:** a point ~20–30 km from the nearest monitored well → `ok`, `thin`, with the distance heads‑up line.

> An **A2** number is an **estimate from past seasons**, not a live model call. Treat it as a weak hint.

---

## B. `status: "insufficient_data"` — declined, with a `reason` (HTTP 200)

Every decline carries a machine‑stable **`reason_code`** and a **`details`** object (the concrete numbers
behind it), alongside the human `reason` and the farmer `message`. Two of these are genuine **data** gaps
(B1 spatial, B2/B3 temporal); the rest are input/usage cases that also surface here.

**`reason_code` catalog** (branch on this, not the free‑text `reason`):

| `reason_code` | Meaning | `details` |
|---|---|---|
| `out_of_network` | no monitored well near enough (> 40 km, or none / too few found) | `nearest_km` *(or `error_code`)* |
| `sparse_history` | wells nearby but too few years for a seasonal normal (or no yoy pair) | `nearest_km`, `usable_years` |
| `no_recent_reading` | no live reading anywhere **and** < 2 yr history to estimate from | `nearest_km`, `usable_years` |
| `no_seasonal_history` | analog path found nothing usable to estimate from | `nearest_km`, `usable_years` |
| `bad_request` | usage error — future/unparseable date, or invalid/unknown location | `error_code` |
| `forecast_failed` | an unexpected forecaster error (rare; logged server‑side) | `error_code` |

### B1 — Coverage gap *(spatial — the point is outside the well network)*
- **Trigger:** the nearest monitored well is **> 40 km** away. (15–40 km is still answered as a thin
  *regional* read — **A3**; only beyond 40 km do we decline rather than guess.)
- **`reason_code`:** `out_of_network` · **`details`:** `{"nearest_km": <N>}`
- **Reason:** `no monitored well within 40 km (nearest is <N> km) — this point is outside our well network here`
- **Example:** `{"lat":19.0,"lon":70.5}` (offshore) → nearest well ≈ 222 km.

### B2 — History gap *(temporal — nearby wells, too few years)*
- **Trigger:** wells **are** nearby but there's no usable seasonal normal — the analog path needs
  **≥ 2 years** of seasonal history; the live path additionally needs the year‑over‑year normal; and a
  seasonal normal can't be formed from 0 usable years.
- **`reason_code`:** `sparse_history` (no usable seasonal normal / missing yoy pair) **or** `no_recent_reading`
  (no live reading anywhere **and** < 2 yr history to fall back on) · **`details`:** `{"nearest_km": <N>, "usable_years": <k>}`
- **Reason:** `wells are nearby (<N> km) but lack enough history to form a seasonal normal (usable years: <k>)`
  (or `…no recent live reading and <2 yrs of seasonal history to estimate from`).
- **Example:** a well with only ~1 year of readings. (Points with ≥2 yr now resolve to an **A2** estimate.)

### B3 — No usable seasonal history *(analog path, nothing to estimate from)*
- **Trigger:** the analog fallback ran but neither a per‑well seasonal change nor the climatology normal
  could be computed for any neighbour (e.g. their history doesn't cover the anchor's season window).
- **`reason_code`:** `no_seasonal_history` · **`details`:** `{"nearest_km": <N>, "usable_years": <k>}`
- **Reason:** `wells are nearby (<N> km) but have no usable seasonal history to estimate from`

### B4 — Future or bad date *(usage)*
- **Trigger:** `date` is **after today** (a future anchor would run on forward‑filled inputs), or the date
  is unparseable. The date may be **today or any past date**; only strictly‑future is rejected.
- **`reason_code`:** `bad_request` · **`details`:** `{"error_code": "bad_request"}`
- **Reason:** `date <YYYY-MM-DD> is in the future; it must be today (<today>) or earlier.` /
  `unparseable date '<x>'; expected YYYY-MM-DD`
- **Example:** `{"lat":12.9716,"lon":77.5946,"date":"2999-01-01"}`.

### B5 — No wells at all / too few survivors *(coverage, extreme)*
- **Trigger:** no known station near the point, or fewer than the required floor survived data collection.
- **`reason_code`:** `out_of_network` · **`details`:** `{"error_code": "no_neighbours" | "insufficient_neighbours"}`
- **Reason:** `no known stations near the query point` / `only <k> usable neighbour(s) after data collection; need >= <floor>`
- **Example:** a point far outside the monitored region (no registry wells at all).

### B6 — Invalid location / missing input *(usage)*
- **Trigger:** lat/lon out of range, both lat/lon **and** station_code missing, or an unknown station_code.
- **`reason_code`:** `bad_request` · **`details`:** `{"error_code": "bad_request" | "station_not_found"}`
- **Reason:** `lat/lon out of range: …` / `provide (lat, lon) or station_code` / `station_code '<x>' not in the station registry`
- **Example:** `{"lat":999,"lon":0}`.

---

## C. `status: "error"` — a service problem, not a data problem

| | When | What you see |
|---|---|---|
| **C1 — Service not loaded** (HTTP **503**) | the engine/model failed to load at startup (a wrong model/data path or unreadable checkpoint). The service **fails fast** and refuses to start rather than come up "healthy but empty" | `/health → {loaded:false}`; `/advisory → 503 {status:"error", error.code:"init_error"}` |
| **C2 — Unexpected internal error** (HTTP **200**) | an unhandled exception on a well‑formed request. It never leaks a raw 500 | `{status:"error", error:{code:"internal_error", message:"…"}}` plus a safe farmer `message`; the traceback is logged server‑side |

---

## D. Quality / degradation flags on an `ok` (read these — not failures)

- **`confidence.level = "thin"`** — weak signal. Either an analog estimate (A2, always thin), the nearby
  live wells **disagree** spatially, or only a single well contributed.
- **`confidence.freshness = "poor" | "medium"`** — stale data. `medium` = a stale‑but‑present well fell back
  to its seasonal analog; `poor` = climatology / the no‑live‑reading A2 path. (`good` = a recent live reading nearby.)
- **`Basis:` line in `message`** — the number is an **estimate** (analog/climatology), not a live model forecast.
- **`warnings: ["imagery: N satellite tile(s) hit an Earth Engine rate limit …"]`** — some satellite tiles
  were throttled by Earth Engine and retried (or zero‑filled if still missing). This is the **only** note
  surfaced in `warnings`, on purpose. More likely on a fresh / low‑quota Earth Engine project; it self‑heals
  via serial retry.

---

## E. Data‑source fallbacks (degrade gracefully — usually invisible)

- **Live GWL (WRIS) outage** → a per‑request breaker trips and the normals fall back to the bundled dataset
  snapshot; the request still answers (confidence may cap lower). A full outage never hard‑fails a request.
- **Earth Engine numeric fetch throttled/stalled** → a hard per‑call timeout + retry on the high‑volume
  endpoint abandons a stalled call and retries; it never hangs indefinitely. Sustained **quota exhaustion**,
  though, makes the fetch fail for all stations → the analog fallback (A2) or a B‑decline.
- **Satellite tile missing** after retries → zero‑filled for that neighbour (a soft warning).
- **Stale well reading** → forward‑filled to the anchor for **any** age (no staleness drop); confidence is
  degraded by age instead, so a stale‑but‑present well still contributes with a lower freshness label.

---

## Quick branch map for callers

```
status == "ok"                -> use it, BUT check confidence.level + the "Basis:" line
                                 ("thin" / an analog estimate = weak signal).
status == "insufficient_data" -> no number; branch on `reason_code`, show `reason` + `details`.
                                 out_of_network (B1/B5) vs sparse_history/no_recent_reading/
                                 no_seasonal_history (B2/B3) vs bad_request (B4/B6).
status == "error"             -> our problem, not the data. 503 = service down (C1);
                                 200 + internal_error = transient/bug (C2).
```

**The two genuine data declines:**
- **Coverage** (`out_of_network`) — nearest well > 40 km (point outside the network); 15–40 km still
  answers as a thin *regional* read (A3). *Spatial.*
- **History** (`sparse_history` / `no_recent_reading` / `no_seasonal_history`) — wells nearby but too few
  years of usable seasonal history. *Temporal.*

Everything else is a usage error (bad/future input), a service error, or a **degraded‑but‑usable** `ok`.
