# Response fields — every field explained

The `POST /advisory` response is a JSON object. This page explains **every field**, one by one, for each
`status`. For the *states* (when you get each `status`) and their triggers, see
[RESPONSE_STATES.md](RESPONSE_STATES.md); for how the decision is made, see
[DECISION_LAYER.md](DECISION_LAYER.md).

> **Sign convention** (GWL = depth‑to‑water): `change_m < 0` = water table **rises** (recharge / gain);
> `change_m > 0` = water table **falls** (depletion / loss).

---

## 1. `status: "ok"` — a forecast (or a labelled estimate) was produced

### Worked example (default, `verbose=false`)

```json
{
  "status": "ok",
  "location": {"lat": 17.3334, "lon": 78.4093},
  "anchor_date": "2026-08-20",
  "forecast_date": "2026-11-20",
  "numbers": {"forecast_change_m": -2.14, "forecast_change_m_6m": -3.02},
  "outlook": {
    "water_trend": {"direction": "rising", "reliability_pct": 88},
    "vs_normal":   {"level": "above", "clarity": "clear", "reliability_pct": 74}
  },
  "long_crop": {"allowed": true, "reason": "3m and 6m at/above normal, water not falling, call decisive — workable; keep a margin"},
  "confidence": {"level": "well-supported"},
  "crop_guidance": {
    "water_need": "high",
    "type_examples": ["paddy/rice", "irrigated vegetables", "fodder maize"],
    "rule": "Water budget is high; a longer-duration crop is workable — keep a margin ..."
  },
  "message": "• Water this season: the table looks set to rise (recharge) — you should gain water (about 9 in 10 for a move this size).\n• Versus a normal year: a bit better than a normal year (more water than usual) — a clear call (about 3 in 4).\n• Suggestion (kharif window): water-heavy; can expand — a high-water crop (e.g. paddy/rice, irrigated vegetables, fodder maize).\n• Duration: A longer-duration crop is workable — keep a margin (not a new perennial on one forecast).\n• This is a reasonably well-supported read for your area.\n• Context: last year was about normal.\n• Caution: 'normal' means typical of recent years ...\n• Note: groundwater signal only ..."
}
```

### Top‑level fields

| Field | Type | Meaning |
|---|---|---|
| `status` | string | `ok` — a forecast or a clearly‑labelled estimate is present. (Still read `confidence` + the message's `Basis:` line — an `ok` can be **degraded**.) |
| `location` | `{lat, lon}` | the resolved query point (echoed back; may be snapped to a station if `station_code` was given) |
| `anchor_date` | `YYYY-MM-DD` | the "as‑of" date the forecast is anchored to (today or a past date) |
| `forecast_date` | `YYYY-MM-DD` | the 3‑month target date (`anchor + 3 months`) |
| `numbers` | object | the raw forecast changes — see below |
| `outlook` | object | the two calibrated axes — **the heart of the advisory** — see below |
| `long_crop` | `{allowed, reason}` | whether a long‑duration / perennial crop is cleared, and why/why‑not |
| `confidence` | `{level}` | how much to trust the call, from the **data** (not the model) — see below |
| `crop_guidance` | object | the crop‑TYPE guidance (water need + example crops) — see below |
| `message` | string | the farmer‑facing bulleted message (`\n`‑separated); a plain‑language render of everything above |
| `warnings` | string[] | **present only** if satellite imagery was rate‑limited by Earth Engine (and retried). A clean run omits it |
| `last_year` | string | *(verbose only)* last year vs its own normal, in plain words: `wetter` / `normal` / `drier` — the "Context: last year was …" line |

### `numbers`

| Field | Type | Meaning |
|---|---|---|
| `forecast_change_m` | float (m) | 3‑month predicted change in depth‑to‑water. `< 0` rises (recharge), `> 0` falls (depletion). Bounded to a ±60%‑of‑current plausibility clamp |
| `forecast_change_m_6m` | float (m) | the 6‑month predicted change (present when the 6m forecaster is loaded) |

*`verbose=true` adds the two normals the forecast is compared against:* `normal_seasonal_change_m`
and `normal_seasonal_change_m_6m` (the 3m / 6m seasonal yardsticks).

### `outlook` — two calibrated axes (each answers a **different** question)

The advisory deliberately separates *"is water gaining or losing?"* from *"is that better or worse than
usual?"*. Each axis carries its **own** test‑set reliability. Both are **separate** from `confidence`.

> **How `level` and `clarity` are computed** — the gap vs the band, the `clear`/`moderate`/`typical` thresholds,
> why `below`‑normal can still be recharging, and why the 74/71 % hold up at R²_δ ≈ 0.25 — is in
> [DECISION_LAYER.md → Reading the outlook](DECISION_LAYER.md#reading-the-outlook-levels-clarity-and-reliability).

**`outlook.water_trend`** — *will the water table rise or fall?* (vs **zero**; the model's strongest signal)

| Field | Type | Meaning |
|---|---|---|
| `direction` | `rising` \| `falling` | `rising` = the 3m forecast is a recharge/gain (`change_m < 0`); `falling` = a drawdown/loss |
| `reliability_pct` | int | test‑set **direction accuracy for a move of this size** — bigger moves are called right more often (~66% for a <0.5 m move up to ~88% for a 2–5 m move). Rendered in the message as "about N in 10" |

**`outlook.vs_normal`** — *is that better or worse than a normal year?* (vs the seasonal **normal ± band**)

| Field | Type | Meaning |
|---|---|---|
| `level` | `above` \| `normal` \| `below` | `above` = wetter than the seasonal normal (more recharge / less depletion than usual); `below` = drier than usual; `normal` = within the local band |
| `clarity` | `clear` \| `moderate` \| `typical` | how **decisive** the call is: `clear` = well past the band (ratio ≥ 2); `moderate` = just past it (1 ≤ ratio < 2); `typical` = within the band (a `normal` level → no above/below claim) |
| `reliability_pct` | int | test‑set **side‑accuracy** of the above/below call at this clarity (`clear ≈ 74`, `moderate ≈ 71`). **Present only for an `above` / `below` call** (never null) |
| `note` | string | present **instead of** `reliability_pct` when `level = normal` (`clarity = typical`): a short "within the normal band — no above/below call to rate; go by the water‑trend read" — the forecast sits in the deadzone where an above/below % would be a coin‑flip, so we give a note, not a number |

> All `reliability_pct` numbers are **baked constants** from `scripts/calibrate_clarity.py` (re‑computed at
> each retrain) — there is **no** model or service call at runtime. They are self‑normal **floors**; the live
> neighbour‑normal path runs a few points higher. See MODEL_CARD → *How to trust the advisory*.

### `confidence` — how good is the *data* (not the call)

| Field | Type | Meaning |
|---|---|---|
| `level` | `well-supported` \| `mixed` \| `thin` | a fresh reading nearby **×** how well the neighbouring wells agree. `thin` also results from an analog estimate, a lone/disagreeing well, or a far (15–40 km) regional read. It is a **rough guide, not a probability** |

*`verbose=true` adds* `confidence.freshness` (`good` = a recent live reading nearby · `medium` = a
stale‑but‑present well fell back to its seasonal analog · `poor` = climatology / no live reading).

> **`confidence` vs `outlook.*.reliability_pct`** — `confidence` is about the **inputs** (is the data fresh and
> in agreement?); `reliability_pct` is about the **call** (historically, how often is a call like this right?).
> A call can be *well‑supported* (good data) yet only *moderate* clarity (borderline vs normal), or vice‑versa.

### `crop_guidance` — crop **types**, never named prescriptions

| Field | Type | Meaning |
|---|---|---|
| `water_need` | `high` \| `normal` \| `moderate` \| `low` | the water **budget** the land can supply this season (from the `tier`) — a budget, **not** a shortage |
| `type_examples` | string[] | up to 4 **calendar‑appropriate** example crops for the water need — sowable *now* per the kharif/rabi/zaid window. Illustration, not a prescription (perennials are never suggested) |
| `rule` | string | the deterministic one‑line rationale (water need + duration) |

*(Duration lives on `long_crop`, not here — the old `crop_guidance.duration_hint` / `long_crop_ok` fields
were dropped as duplicates of `long_crop.allowed`.)*

### `long_crop`

| Field | Type | Meaning |
|---|---|---|
| `allowed` | bool | is a long‑duration / perennial crop cleared? (Only if 3m & 6m at/above normal, water **not falling**, and the above‑call is not marginal — see [DECISION_LAYER.md](DECISION_LAYER.md#c-change-the-long-crop-rule--rule_enginelong_crop_gatea3-a6-water_trend-clarity)) |
| `reason` | string | plain‑language why / why‑not |

### `verbose=true` extras

Setting `{"verbose": true}` keeps everything above and adds a little debug detail:

| Field | Meaning |
|---|---|
| `last_year` | last year vs its own normal, in plain words: `wetter` / `normal` / `drier` (the "Context: last year …" line — this replaces the old `regime.b`) |
| `numbers.*` normals | `normal_seasonal_change_m`, `normal_seasonal_change_m_6m` — the 3m / 6m seasonal yardsticks the forecast is compared against |
| `confidence.freshness` | `good` / `medium` / `poor` (data recency basis) |

*(There is no `regime` object or `_explanation` block anymore — `regime.a` was just `outlook.vs_normal.level`,
`regime.b` became `last_year`, and every field is explained here instead of inline.)*

---

## 2. `status: "insufficient_data"` — declined (HTTP 200)

The advisory never hard‑errors on a well‑formed query; when it can't stand behind a number it declines
**with structure** so a client can tell *why* and react.

```json
{
  "status": "insufficient_data",
  "location": {"lat": 25.3579, "lon": 77.4263},
  "anchor_date": "2026-08-20",
  "forecast_date": "2026-11-20",
  "reason": "wells are nearby (23 km) but lack enough history to form a seasonal normal (usable years: 1)",
  "reason_code": "sparse_history",
  "details": {"nearest_km": 23.0, "usable_years": 1},
  "confidence": {"level": "thin"},
  "crop_guidance": null,
  "message": "• There are wells nearby, but they don't have enough years of records to work out what's normal for this season (nearest monitored well ~23 km away; only 1 year of usable seasonal history).\n• For now, rely on your own well/borewell reading and local KVK advice; a district groundwater bulletin (CGWB) can also help. Try again once nearby wells report."
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | string | `insufficient_data` |
| `location`, `anchor_date`, `forecast_date` | — | echoed as in the `ok` case (`forecast_date` may be null) |
| `reason` | string | **human‑readable** free text — good for logs/UI, not for branching |
| `reason_code` | string | **machine‑stable** enum to branch on — see the catalog below |
| `details` | object | the concrete numbers behind the decline; keys depend on the code (`nearest_km`, `usable_years`, `error_code`). Empty‑value keys are omitted |
| `confidence.level` | string | always `thin` on a decline |
| `crop_guidance` | null | no guidance when we decline |
| `message` | string | a plain‑language *why* + what the farmer can do instead |

**`reason_code` catalog:**

| `reason_code` | Meaning | `details` keys |
|---|---|---|
| `out_of_network` | no monitored well near enough (> 40 km, or none / too few found) | `nearest_km` *(or `error_code`)* |
| `sparse_history` | wells nearby but too few years for a seasonal normal (or no year‑over‑year pair) | `nearest_km`, `usable_years` |
| `no_recent_reading` | no live reading anywhere **and** < 2 yr history to estimate from | `nearest_km`, `usable_years` |
| `no_seasonal_history` | the analog path found nothing usable to estimate from | `nearest_km`, `usable_years` |
| `bad_request` | usage error — future/unparseable date, or invalid/unknown location | `error_code` |
| `forecast_failed` | an unexpected forecaster error (rare; logged server‑side) | `error_code` |

*(15–40 km still returns `status: ok` as a thin **regional** read — it does **not** decline. Only > 40 km
gives `out_of_network`.)*

---

## 3. `status: "error"` — a service problem, not a data problem

| Field | Type | Meaning |
|---|---|---|
| `status` | string | `error` |
| `error.code` | string | `init_error` (service not loaded — **HTTP 503**) or `internal_error` (an unexpected exception on a well‑formed request — **HTTP 200**, never a raw 500) |
| `error.message` | string | short diagnostic; the full traceback is logged server‑side, not returned |
| `message` | string | a safe farmer‑facing fallback message (on `internal_error`) |

---

## Quick branch map for callers

```
status == "ok"                -> use numbers + outlook, BUT check confidence.level and the message "Basis:" line
                                 (thin / an analog estimate = weak signal). outlook.water_trend = gain/lose;
                                 outlook.vs_normal = better/worse than usual (+ reliability_pct on each).
status == "insufficient_data" -> no number. Branch on reason_code; show reason + details in the UI.
status == "error"             -> our problem. 503 = service down (init_error); 200 = transient/bug (internal_error).
```
