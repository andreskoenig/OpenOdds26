# OpenOdds26 — World Cup 2026 Prediction Model

A from-scratch football forecasting model built on **free, public data**. It
fits a time-weighted, opponent-adjusted **Dixon–Coles goals model**, blends in a
**betting-market prior** (Polymarket), and runs a **20,000-tournament Monte Carlo
simulation** of the real 48-team 2026 bracket to produce win probabilities,
round-reach odds, and group-stage predictions.

Validated on the 2022 World Cup against Bet365's closing odds. Honest headline:
**it gets close to the world's sharpest book but does not beat the closing line.**

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
   weighted more; half-life **1.5y**, hard truncation at **10y** — both chosen by
   out-of-sample validation). Priors come from FIFA points, squad value, and the
   de-pathed market.
3. **Simulation:** 20,000 Monte Carlo runs of the group stage + explicit 48-team
   knockout bracket (best-thirds allocation, host advantage, extra time/penalties).

## Results

- **2022 validation** (64 WC games, match-level 1X2 log-loss; lower is better):
  model **≈1.02** vs de-vigged **Bet365 0.9986**. Optimal model+market blend put
  weight **0** on the model — the closing line is sharper.
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
