# The advisory decision layer (and how to customize it)

The forecaster (frozen Prithvi‑EO + LoRA + TFT) only outputs a **number**: the predicted 3‑ and 6‑month
change in depth‑to‑water. The **decision layer** (everything in `advisory/`) turns that number — plus the
local well history — into a farmer‑facing advisory. It is **fully deterministic** (the only optional
non‑determinism is the SLM message phraser). Nothing here changes the forecaster.

---

## Pipeline at a glance

| # | Stage | Module | What it does |
|---|---|---|---|
| 1 | **Forecast** (input) | the model | 3m & 6m `change_m` = IDW over the nearest wells' model predictions |
| 2 | **Normals** | `normals.py` | seasonal normal + year‑over‑year normal + a local **band**, each = `median over ~10 yrs ( IDW‑over‑wells of that year's change )` |
| 3 | **Stale / no‑live‑data** | `analog.py` | a stale well (or a no‑live‑data point) falls back to the **seasonal analog** = the *same recipe as the normal* over the recent ~3 yrs |
| 4 | **Coverage** | `advisory.py` | `≤15 km` normal · `15–40 km` extended (thin + caveat + plausibility clamp) · `>40 km` decline |
| 5 | **Binning** | `rule_engine.bin_a/bin_b` | internal bins vs normal (±band): 3m → `outlook.vs_normal` · yoy → `last_year` (context) · 6m → the long‑crop gate |
| 5b | **Two‑axis drivers** | `reliability.py` | `water_trend` = 3m sign vs **zero** (rising/falling) + its direction reliability · `clarity` = how decisive the `vs_normal` call is + its reliability |
| 6 | **Rule table** | `rules/gwl_advisory_rules.csv` | **`(water_trend, vs_normal)`** → `net_read`, `crop_lean`, `tier` (yoy `b` is **context only**, no longer a tier driver) |
| 7 | **Long‑crop gate** | `rule_engine.long_crop_gate` | allowed only if 3m & 6m at/above normal, water **not falling**, and the above‑call is **not marginal** |
| 8 | **Crop guidance** | `rule_engine.crop_guidance` | `tier` → water‑need + sowing‑calendar crop examples (duration comes from the long‑crop gate, surfaced on `long_crop`) |
| 9 | **Confidence** | `consensus.py` | spatial agreement (per‑well z‑spread) ⊗ freshness → `well‑supported / mixed / thin`; capped for fallback/analog/far |
| 10 | **Message** | `phraser.py` | deterministic template, or optional Gemma SLM (phrasing only — the decision is identical) |

The default **response** carries `numbers`, **`outlook`** (`water_trend` + `vs_normal`), `crop_guidance`,
`long_crop`, `confidence`, and the `message`; `verbose=true` adds `last_year` (the yoy context), the two
seasonal normals the forecast is compared against, and `confidence.freshness`. Every field is explained one‑by‑one in
[RESPONSE_FIELDS.md](RESPONSE_FIELDS.md); every response *state* — ok / insufficient_data / error — is
catalogued in [RESPONSE_STATES.md](RESPONSE_STATES.md).

> **Why two axes drive the tier.** The old table keyed `(a = forecast‑vs‑normal, b = past‑year‑vs‑normal)`.
> That let *"above normal"* in a **depletion** season (water still falling, just less than usual) map to a
> **water‑heavy** budget — a contradiction. The tier is now keyed on **`water_trend`** (is the table actually
> gaining or losing water?) × **`vs_normal`** (is that better or worse than usual?), so a falling‑but‑above
> case correctly reads *"drawing down — go easy on water"* (usual‑to‑light), not water‑heavy.

---

## Reading the outlook: levels, clarity, and reliability

Both `outlook` axes are graded from the **same 3‑month forecast**, via **two different comparisons**:

**`water_trend` — forecast vs ZERO** (absolute). `forecast_change_m < 0` = table **rises** (recharge → `rising`);
`> 0` = **falls** (depletion → `falling`). Its `reliability_pct` = the test‑set **direction accuracy for a move
of that size** (~66% under 0.5 m → ~88% at 2–5 m).

**`vs_normal` — forecast vs the seasonal NORMAL** (relative). It emits two things from the gap
`d = forecast − normal_seasonal`.

**`level`** = the **sign of the gap**, binned against the local **band**:

| gap | `level` | meaning |
|---|---|---|
| `d < −band` | **`above`** | *less depletion / more recharge than usual* → **better than a normal year** |
| `d > +band` | **`below`** | *more depletion / less recharge than usual* → **worse than a normal year** |
| `−band ≤ d ≤ +band` | **`normal`** | within the band |

**`clarity`** = **how big the gap is, in bands** — `ratio = abs(d) / band`:

| `ratio` | `clarity` | `reliability_pct` |
|---|---|---|
| `≥ 2` | **`clear`** (well past the band) | **≈ 74** |
| `1 – 2` | **`moderate`** (just past) | **≈ 71** |
| `< 1` | **`typical`** (inside the band) | *(none — a `note` instead)* |

> **`vs_normal` is NOT "discharge vs recharge".** `level` compares the forecast to the **normal**, not to zero,
> so a `below` (drier‑than‑usual) location can still be **recharging** — just less than a normal monsoon.
> *Example:* normal −5 m, forecast −2 m, band 1 m → gap **+3 m** → `ratio 3` → **clear `below`** (74 %), yet the
> forecast is −2 m so `water_trend = rising` (recharge). Absolute rise/fall lives in `water_trend`;
> better/worse‑than‑usual lives in `vs_normal`. `clarity` is only the gap **magnitude**, never its direction —
> a **clear** call can be clear‑above *or* clear‑below.

**The band** is the local ± tolerance — ≈ ½·IQR of the neighbour wells' year‑to‑year spread, floored at 0.3 m
(`advisory/normals.py`). Everything above is **relative to it**: a 1 m gap is "clear" where wells barely move and
"typical" where they swing several metres.

### What the reliability % measures (and why it holds at R²_δ ≈ 0.25)

The score is a **sign match** — it says nothing about the exact metres:

```
correct  ⟺  sign(actual − normal) == sign(prediction − normal)
```

i.e. *did the actual land on the same side of the normal — better or worse than usual — as the forecast said?*
We are **not** claiming the actual is close to the predicted value; it needn't be. The value can be well off and
the call still counts, **as long as the actual doesn't cross back over the normal**.

That is exactly why a **far‑from‑normal (`clear`) prediction is more trustworthy**: the bigger the predicted gap,
the larger a model error would have to be to push the actual back across the normal — so it usually stays on the
predicted side (it just has to stay above, or below; it need not also be *far*). Near the normal a tiny error flips
the side — a coin‑flip — which is why we return a `note` there, not a number.

So a modest **R²_δ ≈ 0.25** and an honest **~74 %** side‑accuracy sit together fine: **R²_δ grades *magnitude*
(regression — a hard bar, "how many metres"); the reliability % grades the *side of the normal* (classification — an
easier bar, "which way vs usual"), and we only assert it where the gap earns it (≥ 1 band).** In one line:
**clarity buys side‑agreement, not value‑closeness.**

The numbers are baked, self‑normal **floor** constants (`advisory/reliability.py`; the live neighbour‑normal path
runs a few points higher) — see [MODEL_CARD.md](MODEL_CARD.md) → *How to trust the advisory*.

---

## Worked example — a query walked end to end

Inputs **after interpolation** over the neighbour wells (a farmer's plot, monsoon anchor):

| quantity | value |
|---|---|
| 3‑month forecast `change_m` | **−4.0 m** (`< 0` → water **rises** / recharge) |
| 6‑month forecast | −3.0 m |
| seasonal **normal** (3m) | **−1.5 m** · **band** ±1.0 m |
| last year (yoy) | ≈ normal |
| neighbour wells | tight agreement, recent readings |

Derivation (two axes from the *same* forecast, then the tier, then confidence):

| step | computation | result |
|---|---|---|
| **water_trend** (forecast vs **zero**) | −4.0 < 0 → rising; move of 4 m falls in the 2–5 m bucket | **rising**, reliability **88%** |
| gap vs normal | `d = −4.0 − (−1.5) = −2.5 m` | |
| **clarity** | `abs(d) / band = 2.5 / 1.0 = 2.5  (≥ 2)` | **clear** |
| **vs_normal level** | `d < −band` → wetter than usual | **above**, reliability **74%** |
| **crop tier** | `rules[(rising, above)]` | **water‑heavy → `water_need: high`** (plenty available) |
| **long‑crop gate** | 3m & 6m at/above normal, water not falling, call not marginal | **allowed** |
| **confidence** | neighbour wells agree × fresh readings | **well‑supported** |

→ the farmer message:

```
• Water this season: the table looks set to rise (recharge) — you should gain water (about 9 in 10 for a move this size).
• Versus a normal year: a bit better than a normal year (more water than usual) — a clear call (about 3 in 4).
• Suggestion (kharif window): water-heavy; can expand — a high-water crop (e.g. paddy/rice, irrigated vegetables, fodder maize).
• Duration: A longer-duration crop is workable — keep a margin (not a new perennial on one forecast).
• This is a reasonably well-supported read for your area.
• Context: last year was about normal.
• Caution / Note … (normal ≠ sustainable; groundwater-only lens).
```

> **Contrast — a within‑band case.** Keep the same normal/band but make the forecast **−2.0 m** → `d = −0.5 m`,
> `abs(d)/band = 0.5 < 1` → **typical**. The table is still **rising (recharge)**, but it is *not* distinguishably
> better than a normal season, so `vs_normal` carries a **note instead of a %**: *"about a typical year — no strong
> signal either way."* This is exactly why the two axes stay separate: **gaining water ≠ better‑than‑usual.**

---

## How to customize the advisory

Everything a farmer *sees* — the read, the crop suggestion, the water budget, the wording, the confidence
label — is **data + a handful of constants in `advisory/`**. None of it touches the model or the forecast
number, so you can change the advice safely and deterministically. From easiest (a CSV) to deepest (the
message template):

### A. Change the advice for a regime — the rule table (a CSV, **no code**)
**File:** `advisory/rules/gwl_advisory_rules.csv` — **6 rows**, one per `water_trend`×`vs_normal` combination.
**Format:** `water_trend,vs_normal,net_read,crop_lean,tier`

| column | values | meaning |
|---|---|---|
| `water_trend` | `rising` / `falling` | is the water table set to **gain or lose** water (3m forecast sign vs **zero**) |
| `vs_normal` | `above` / `normal` / `below` | is that **better or worse** than a normal year (3m vs the seasonal normal ±band) |
| `net_read` | free text | the plain‑language read shown to the farmer |
| `crop_lean` | free text | the suggestion |
| `tier` | `water-heavy` / `usual` / `usual-to-light` / `water-light` | the water budget → picks the crop list (see B) |

The shipped table (`water_trend × vs_normal → tier`). `water_need` (last column) is the `_WATER_NEED[tier]`
mapping in B — shown here for reference:

| `water_trend` | `vs_normal` | `net_read` | `tier` | water_need |
|---|---|---|---|---|
| rising | above | gaining, better than usual | `water-heavy` | high |
| rising | normal | gaining, about as usual | `usual` | normal |
| rising | below | gaining but weak | `usual-to-light` | moderate |
| falling | above | losing, but less than usual | `usual-to-light` | moderate |
| falling | normal | losing, about as usual | `usual` | normal |
| falling | below | losing, worse than usual | `water-light` | low |

Read it as *"is there water (rising/falling), and is that above/below what's usual?"* — the diagonal
(gaining‑and‑better → heavy; losing‑and‑worse → light) is the intuition; the mixed cells (gaining‑but‑weak,
losing‑but‑less) stay **moderate**. The past‑year yoy (`b`) is **context only** now, not a tier input.

Example — make a *falling / below* year advise harder conservation (edit that row):
```
falling,below,losing water and worse than usual,water-light; conserve moisture — consider fallow,water-light
```
**Apply without editing the shipped file:** pass your own CSV to the engine —
`AdvisoryEngine(rules_path="/path/to/my_rules.csv")` — no code change.

### B. Localise the crop suggestions — `advisory/rule_engine.py`
- **`_EXAMPLES`** — a `season → water‑need → [crops]` dict. Seasons: `kharif` / `rabi` / `zaid` / `any`;
  water‑needs: `high` / `normal` / `moderate` / `low`. Replace the lists with what your district grows:
  ```python
  _EXAMPLES = {
      "kharif": { "high": ["paddy/rice", ...], "moderate": ["pigeonpea (tur)", "green gram", ...], ... },
      ...
  }
  ```
- **`_WATER_NEED`** — maps a rule `tier` → the water‑need label
  (`water-heavy→high`, `usual→normal`, `usual-to-light→moderate`, `water-light→low`).
- **`_DURATION`** — the duration wording per regime (`wet` / `dry` / `transition`).

### C. Change the long‑crop rule — `rule_engine.long_crop_gate(a3, a6, water_trend, clarity)`
A long/perennial crop is cleared **only if all hold**: (1) both 3m and 6m at/above normal (a `below` 6m
vetoes; no 6m ⇒ not cleared), (2) the water is **not falling** (`water_trend != "falling"` — a perennial
needs the table to hold or rise, not draw down), and (3) an `above` call is **not marginal**
(`clarity != "moderate"` — a barely‑above read is too weak to commit a year+). Edit this one function to
loosen or tighten any of the three (e.g. allow on 3m alone, or drop the falling veto).

### D. Make the above/normal/below call more/less sensitive — the band
**File:** `advisory/normals.py`, `_BAND_FLOOR = 0.3` (metres). The regime is `(forecast − normal)` vs `±band`:
**raise** the floor → more results land on **normal** (less twitchy); **lower** it → more **above/below**
calls. (The band also widens with the wells' interannual spread — the IQR multiplier inside `normals._band`.)

### E. The message wording — `advisory/phraser.py` (+ env)
The deterministic **bullet template** (bullet order, caveats) lives in `phraser.py`. Setting
`GWL_ADVISORY_SLM=1` swaps in a local Gemma model for **phrasing only** — the numbers and the decision are
identical either way, so the SLM is safe to leave off.

### F. Confidence label & caps — `advisory/consensus.py`
The `well-supported / mixed / thin` label = spatial well‑agreement (z‑spread) ⊗ data freshness, then softened
and **capped** (analog / no‑live → `thin`; extended‑distance band → `thin`). Tune the cutoffs or the caps here.

> **Not the decision layer** (these shape the *inputs*, not the crop/water read — see
> [INFERENCE.md](INFERENCE.md) / [ADVISORY.md](ADVISORY.md)): the GWL **source** (`GWL_SOURCE`, default
> `nwdp`), the **staleness** drop (`anchor.py`), fetch **concurrency** (`GWL_WRIS_CONCURRENCY`), the normals
> **history window** (`n_years`), and the **coverage** distance tiers.

---

## What to leave alone (unless you mean to)
- **The forecaster** (Prithvi + LoRA + TFT) — that's the *model*, not the decision layer. Retraining is a
  separate path (see [TRAINING.md](TRAINING.md)).
- **The normal / analog recipe** (`median‑over‑years( IDW‑over‑wells )`) — it's the yardstick the whole
  decision layer compares against, and the analog is deliberately computed the *same* way so the two are
  apples‑to‑apples. Change it and every regime call shifts.

---

## One‑line mental model
> **forecast** = IDW over wells · **normal / analog** = median over years of (IDW over wells) · the rule
> table turns *(is water rising or falling, and is that above/below normal)* into a crop/water read ·
> confidence and the distance/freshness caveats say how much to trust it.
