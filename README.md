# OpenOdds26 — World Cup 2026 Prediction Model

A from-scratch football forecasting model built on **free, public data**. It
fits a time-weighted, opponent-adjusted **Dixon–Coles goals model**, blends in a
**betting-market prior** (Polymarket), and runs a **20,000-tournament Monte Carlo
simulation** of the real 48-team 2026 bracket to produce win probabilities,
round-reach odds, and group-stage predictions.

Validated on the 2022 World Cup against Bet365's closing odds. Honest headline:
**it gets close to the world's sharpest book but does not beat the closing line.**

## Current 2026 prediction

<!-- PREDICTIONS:START -->
**Model v1.1** · last run **2026-06-11 11:59** · 20000 simulations, as-of 2026-06-10

| # | Team | Model P(win) | Market (Polymarket) |
|---|------|-------------:|--------------------:|
| 1 | Spain | 19.3% | 16.3% |
| 2 | Argentina | 18.1% | 9.0% |
| 3 | France | 11.7% | 15.4% |
| 4 | England | 10.6% | 10.4% |
| 5 | Portugal | 9.1% | 10.4% |
| 6 | Brazil | 7.9% | 8.1% |
| 7 | Germany | 4.4% | 5.0% |
| 8 | Netherlands | 3.4% | 3.9% |
| 9 | Belgium | 3.1% | 2.1% |
| 10 | Colombia | 2.3% | 1.8% |

_Auto-generated from `data/forecast_2026.json` by `scripts/update_readme.py`. Market = de-vigged-free Polymarket winner odds (a model input, not an independent benchmark)._
<!-- PREDICTIONS:END -->

### Sensitivity band

<!-- SENSITIVITY:START -->
How robust is the forecast to its two judgment knobs? Re-run over market-prior weight c_m ∈ [0.0, 0.35, 0.7] × recency half-life ∈ [1.5, 3.0, 5.0]y (5000 sims/cell, 2026-06-11 12:31). Narrow band = config-robust; wide band = the number is an opinion of the knob settings.

| Team | Headline | Range across configs |
|------|---------:|---------------------:|
| Spain | 19.5% | 12.3% – 23.9% |
| Argentina | 18.7% | 14.0% – 20.7% |
| France | 12.0% | 6.9% – 13.7% |
| England | 10.6% | 7.2% – 12.3% |
| Portugal | 8.4% | 5.9% – 10.3% |
| Brazil | 7.5% | 6.8% – 15.6% |
| Germany | 4.0% | 2.7% – 5.5% |
| Netherlands | 3.4% | 2.7% – 4.1% |
| Belgium | 2.7% | 1.9% – 4.3% |
| Colombia | 2.4% | 1.3% – 5.3% |
<!-- SENSITIVITY:END -->

> Design rule (SPEC §2): **agents fetch and clean; code computes and predicts.**
> No probability or prediction comes from free-form LLM reasoning — every number
> is produced by the deterministic, unit-tested Python core.

## How it works

1. **Inputs (all free):**
   - Match results — `martj42/international_results` (every men's international since 1872, ~49k matches)
   - FIFA ranking points — community mirrors + the official FIFA PDF (point-in-time)
   - Squad market value — Transfermarkt (talent-pool prior)
   - Market prior — Polymarket "World Cup Winner" odds (normalized, **de-pathed** to strip bracket-draw luck)
2. **Engine — Dixon–Coles:** each team gets an attack and a defense rating;
   expected goals `≈ exp(μ + atk_home − def_away + γ·home)`, with a low-score
   (ρ) correction. Fit by **ridge-penalized, time-weighted MLE** (recent matches
   weighted more; half-life **1.5y**, hard truncation at **10y**). Honest framing:
   out-of-sample validation showed the half-life curve is statistically **flat**
   (its point optimum was actually 5y); 1.5y is a deliberate **recency prior**
   chosen within that flat band, not a validated optimum. Priors come from FIFA
   points, squad value, and the de-pathed market.
3. **Simulation:** 20,000 Monte Carlo runs of the group stage + explicit 48-team
   knockout bracket (best-thirds allocation, host advantage, extra time/penalties).

## Results

- **2022 validation** (64 WC games, match-level 1X2 log-loss; lower is better):
  shipped config **≈1.024** vs de-vigged **Bet365 0.9986**. Optimal model+market
  blend put weight **0** on the model — the closing line is sharper. Caveat: the
  64-game set was consulted across many experiments during development, so treat
  this as a **development-set** estimate (qualitative conclusion robust; the
  point estimate is soft by ~±0.01). Only match-level 1X2 is validated — the
  tournament-simulation layer (P(win), reach-round) has not been backtested.
- **2026 forecast** (top of the field): Spain · Argentina · France · England ·
  Portugal · Brazil. The model leans on recent form (high on Spain/Argentina);
  the market favors France.
- **xG** (StatsBomb open-data) was explored but **dropped**: free international
  xG only covers major tournaments (~4–8% of matches), so it added nothing
  measurable to the 2022 validation.

## Run it

```bash
python -m pytest tests/                       # 72 tests
python scripts/run_pipeline.py                # re-fetch fast data + full re-run
python scripts/run_pipeline.py --no-fetch     # recompute on existing data
python scripts/run_forecast_2026.py           # the 20k forecast on its own
python scripts/run_backtest_2022.py           # 2022 validation vs Bet365
```

`run_pipeline.py` re-fetches Polymarket + match results (Transfermarkt stays
cached), then recomputes de-path → forecast → group-stage → CSV, writing a
provenance manifest to `data/pipeline_run.json`.

## Layout

```
wc_model/        deterministic core (pure Python, unit-tested)
  schemas.py     data contracts (TypedDicts)
  devig.py       de-vig market odds
  features.py    atk/def + market/squad priors (opponent-adjusted, time-truncated)
  model.py       Dixon–Coles fit + scoreline matrix
  market.py      market 1X2 -> scoreline calibration
  simulate.py    48-team Monte Carlo tournament
  evaluate.py    log-loss / Brier / RPS scoring
  pipeline.py    run_prediction / backtest harness
scripts/         data fetchers, tuners, forecast/backtest runners, run_pipeline
config/          tournament_config_2022.json, tournament_config_2026.json
data/            inputs + outputs (forecast_2026.json, predict_groupstage_2026.csv, ...)
tests/           72 unit tests
SPEC.md          original design spec
```

## Configuration

The API key for one (now-unused) odds provider is read from a gitignored `.env`
(`STATSAPI_KEY`) — never committed. All other sources need no credentials.

Locked hyperparameters: `xi=0.0012651` (t½ 1.5y), `max_history_years=10`,
`lambda_reg=8.0`, `c_a=c_d=0.30`, `c_x=c_y=0.10`, `c_v=0.1` (squad),
`c_m=0.35` (market), `opponent_adjust=True`.

## Versioning

Current version lives in `VERSION`. The prediction table and version stamp above
are refreshed on every substantial change via `python scripts/update_readme.py`
(run it after each forecast). Git tags mark releases.

- **v1.1** — fix penalty-shootout strength sign (`atk + def_`, was `atk − def_`;
  strong defenses were penalized in simulated shootouts); honest relabeling of
  the half-life (recency prior, not validated optimum) and the 2022 number
  (development-set grade); new **sensitivity band** published over c_m × half-life.
- **v1.0** — Dixon–Coles + de-pathed Polymarket prior + opponent-adjusted prior,
  1.5y half-life / 10y truncation, squad-value prior. Friendlies **kept**
  (dropping them worsened 2022 log-loss 1.024 → 1.041).
