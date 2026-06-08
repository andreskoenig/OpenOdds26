# Experiments & ideas

A running log of what we've tried (with results) and what's next. Keeps negative
results so we don't re-litigate them.

## Tried & rejected

- **xG (StatsBomb open-data).** Added per-match team xG for the 6 senior intl
  tournaments (WC2018/22, Euro20/24, Copa24, AFCON23) and blended into the prior.
  Coverage is too sparse — only tournaments have free xG (~4% of the 2022 window,
  ~8% for 2026), so ~95% of matches fall back to goals. WC2022 validation:
  log-loss **1.0240 with xG == 1.0240 pure goals**. No measurable gain → dropped
  from the model (scripts kept: `fetch_statsbomb_xg.py`, `xg_coverage_report.py`).

- **Drop friendlies.** Hypothesis: warm-ups are noise. Result: **worse** — WC2022
  log-loss 1.0240 → **1.0411** (friendlies are 32% of the data; removing them
  starves the fit). Rejected.

- **Down-weight friendlies** (`friendly_weight` knob). Swept 1.0 → 0.0; log-loss
  is **monotonic** (1.0240 → 1.0254 → 1.0275 → 1.0309 → 1.0411). No sweet spot
  below 1.0. Friendlies carry net signal even near a tournament. Knob kept as a
  diagnostic; production stays at 1.0.

**Pattern:** three data-quality interventions all failed to help. The model
already extracts the available signal from historical results; the residual gap
to Bet365's closing line is forward-looking info (injuries, form, lineups) that
historical-data cleaning can't recover.

## Next: "momentum" (in-tournament weighting)

Idea: results **within the same World Cup** should carry extra weight for
predicting later games **in that same tournament** — beyond the normal time
decay. A team can arrive in form/peaking (or the opposite) and the early WC games
reveal it fast.

- **Use case:** Morocco 2022 — pre-tournament ratings badly underrated them;
  their group-stage results were the signal that they were a different side that
  month. A momentum term would let R16/QF predictions react to the group stage.
- **Sketch:** add a weight multiplier (or a short-horizon form term) for matches
  in the *current* tournament when simulating/predicting later rounds of it; or a
  per-team in-tournament form adjustment updated after each round.
- **Caveat:** can't be validated on a *forward* forecast cleanly; would test on
  2022 by predicting knockout rounds using group-stage results with vs without the
  momentum term.
