# World Cup 2026 Prediction Model — Build Spec

## 1. Objective
Predict the 2026 FIFA World Cup — the tournament winner, each team's
round-by-round progression probabilities, and the full scoreline
distribution of every match — using a time-weighted Dixon–Coles goals
model with a two-part, market-derived "surprise" factor, evaluated by
Monte Carlo simulation of the bracket.

## 2. System overview
Two layers:

- **Agentic layer (LLM).** An orchestrator (Claude Code, Opus 4.8) plus
  three data-fetch subagents (Sonnet 4.6). Job: acquire and clean
  heterogeneous web data into the schemas in §7.
- **Deterministic layer (pure Python, zero LLM calls).** De-vigging,
  feature construction, model fit, scoreline matrix, tournament
  simulation, evaluation. Must be reproducible and unit-tested.

Hard rule: **agents fetch and clean; code computes and predicts.** No
statistic, probability, or prediction is ever produced by free-form LLM
reasoning.

## 3. The match engine (Dixon–Coles)
For a match between home team `i` and away team `j`:

    log λ_home = μ + atk_i − def_j + γ·H_i
    log λ_away = μ + atk_j − def_i

where
- `atk_x` = attacking strength of team x (higher → scores more)
- `def_x` = defensive strength of team x (higher → concedes fewer)
- `μ`     = global baseline log goal rate
- `γ`     = home advantage; `H_i = 1` only if team i is a tournament host
            playing in its own country, else 0. All non-host matches are
            neutral (H = 0 for both sides).

Goals are modeled jointly. Independent-Poisson base:

    P0(x, y) = Pois(x; λ_home) · Pois(y; λ_away)

Dixon–Coles low-score dependence correction with parameter `ρ`:

    P(x, y) = τ(x, y) · P0(x, y)
    τ(0,0) = 1 − λ_home·λ_away·ρ
    τ(0,1) = 1 + λ_home·ρ
    τ(1,0) = 1 + λ_away·ρ
    τ(1,1) = 1 − ρ
    τ(x,y) = 1   otherwise

Compute P(x, y) over x, y ∈ {0, …, 10} and renormalize to sum to 1. This
**scoreline matrix is the model's only output primitive.**

> **NOTE (formulas unchanged):** `ρ` is fit from data and is typically
> *negative* for football (≈ −0.1 to −0.15). With these τ formulas it is that
> negative `ρ` that raises the 0-0 and 1-1 draw mass and lowers 1-0/0-1 — the
> regime the fitted model actually operates in. (A positive `ρ` does the
> reverse.) The fit must keep `ρ` within
> `[max(−1/λ_home, −1/λ_away), min(1/(λ_home·λ_away), 1)]` so that no
> τ-adjusted cell can go negative.

### Derived markets (all are sums over the matrix)
- Home win = Σ_{x>y} P(x,y);  Draw = Σ_{x=y};  Away win = Σ_{x<y}
- Over 2.5 = Σ_{x+y≥3};  BTTS = Σ_{x≥1, y≥1}
- Correct score (x,y) = P(x,y) directly — this is the "all intermediate
  scores" deliverable.
- Output **true probabilities**; do NOT add an overround/vig. You are
  forecasting, not booking.

## 4. Ratings construction (the international-football fix)
National teams play few, weakly-connected fixtures, so freely fitted
atk/def ratings are unstable. Anchor them on priors and regularize.

Per team, at the as-of date:
- FIFA strength prior:  `z = (fifa_points − mean) / sd`  (over the field)
- Attack index `a` = standardized blend of xG-for-per-match and
  goals-for-per-match (weight xG higher; it is less noisy):
  `a_raw = w·xGf + (1−w)·GFpm`, then standardize.
- Defense index `d` = standardized blend of xG-against and goals-against,
  sign-inverted so higher = better defense.

Priors:
    atk_prior = c_a·z + c_x·a
    def_prior = c_d·z + c_y·d        (coefficients tuned by backtest)

Fit by maximizing the time-weighted Dixon–Coles log-likelihood with a
ridge penalty pulling ratings toward the priors:

    maximize  Σ_m w(t_m)·log P(result_m)
              − λ_reg · Σ_team [ (atk − atk_prior)² + (def − def_prior)² ]

    time weight  w(t) = exp(−ξ·t),   t = days from match m to the as-of date

`ξ` controls how fast old results decay — **this parameter is "form".**
There is no separate form covariate. Fit μ, γ, ρ and per-team atk/def
jointly; ξ and ρ may be fixed by grid search / backtest. Use only matches
strictly dated before the as-of date.

## 5. The surprise factor
Built entirely from pre-match closing odds. For each historical match
involving team T:

1. **De-vig.** Decimal odds → raw `q_k = 1/o_k` → overround `O = Σ q_k` →
   de-vigged `p_k = q_k / O`. Average across the ≥3 books → consensus
   `(p_win, p_draw, p_loss)` from T's perspective. (Basic normalization;
   Shin / power method optional later.)
2. `p_realized` = consensus probability of the outcome that actually
   happened.
3. **Surprisal**  `S_m = −ln(p_realized)`.
4. Expected points `EP_m = 3·p_win + 1·p_draw`; actual points
   `AP_m ∈ {3,1,0}`.

Two features over T's last N matches, time-weighted with the same `w(t)`:

    Upset propensity        U_T = Σ w·S_m / Σ w          (≥0, direction-agnostic)
    Market-adj. performance M_T = Σ w·(AP_m − EP_m) / Σ w (signed)

### Where each enters the engine
- **U_T → dispersion (variance), not mean.** For team T, replace the
  Poisson goal count with a negative binomial whose dispersion rises with
  U_T, e.g. NB size `r_T = r0 / (1 + κ·(U_T − Ū))`. Higher U_T → fatter
  tails → more upsets in *both* directions in the sim. `κ` tuned by
  backtest; `κ = 0` recovers pure Poisson.
- **M_T → small, regularized mean nudge.** `atk_prior_T += θ·M_T`, with
  `θ` small and defaulting near 0. Raise it only if backtest shows it
  improves calibration — it is an implicit claim of edge over the market.

## 6. Tournament simulation
Format (confirm exact rules in config against the official FIFA source):
48 teams, 12 groups of 4; top 2 of each group plus the 8 best
third-placed teams → 32-team knockout (R32 → R16 → QF → SF → final).

Per Monte Carlo run (N ≥ 10,000):
- **Group stage.** Sample a scoreline from each match's matrix; award
  3/1/0; rank groups and select best thirds by the FIFA tiebreak rules in
  config.
- **Knockouts.** Sample a scoreline; if drawn, simulate extra time with λ
  scaled by 30/90, then penalties (≈50/50 with a small tilt toward the
  higher-rated side).
- **Host advantage** applies only to USA/CAN/MEX in matches played in
  their own country.

Aggregate across runs: P(win tournament), P(reach each round), and
expected / modal scorelines per fixture.

## 7. Data schemas (each agent's output contract)
All keyed on canonical `team_id`. Every time series carries dates so the
deterministic layer can enforce an as-of cutoff — **no look-ahead.**

    teams              team_id, canonical_name, aliases[], confederation
    match_results      match_id, date, home_team_id, away_team_id,
                       venue_country, neutral(bool), competition,
                       home_goals, away_goals
    fifa_ratings       team_id, as_of_date, fifa_points, fifa_rank
                       (point-in-time snapshots)
    team_xg            match_id, team_id, xg_for, xg_against
    match_odds         match_id, bookmaker, odds_home, odds_draw,
                       odds_away, captured_at  (pre-match close, ≥3 books)
    tournament_config  groups, fixture schedule, host_cities,
                       group tiebreak rules, best-thirds rule
                       (hand-maintained, not fetched)

## 8. Agents
**Orchestrator — Claude Code, Opus 4.8, effort high.** Coordinates the
run, passes each subagent the team list + as-of date + path to
`teams.json`, then calls the deterministic tools. Does no fetching itself.

**Three fetch subagents — Sonnet 4.6, narrow tools** (WebFetch, WebSearch,
Read, Write). Each: fetch raw data only, resolve names to canonical
`team_id`, enforce the as-of cutoff, validate, write its schema, report
coverage gaps.
- `results-fifa-fetcher` → match_results + fifa_ratings
- `xg-fetcher`           → team_xg
- `odds-fetcher`         → match_odds (≥3 books, closing prices)

**Deterministic tools** (Python; exposed via MCP or called directly):
`devig`, `build_features`, `fit_model`, `score_matrix`,
`simulate_tournament`, `evaluate`.

## 9. Build order
1. Data contracts (this document).
2. Deterministic core on hand-entered sample data — `score_matrix`,
   `simulate_tournament`, `fit_model`. Unit-test before any agent exists.
3. Validate: backtest on past qualifiers/tournaments; report log-loss +
   Brier + calibration; assert zero look-ahead.
4. Wrap the core as tools.
5. Build one fetch subagent end-to-end; clone the pattern.
6. Add the orchestrator; run the full loop.

## 10. Validation gates
- Match model scored by ranked-probability score / log-loss against
  market closing odds on a held-out set; matching or beating the
  de-vigged market is the bar.
- Calibration plots for 1X2 and over/under.
- Reproducibility: identical seed + inputs → identical output.
- Look-ahead audit: every feature for match m uses only data dated < m.

## 11. Data sources (free)
All sources below are free; pull once and cache permanently — historical
results, ratings, and closing odds are immutable and the footprint is small.

- **match_results** — martj42 `international_results` dataset
  (GitHub / Kaggle). Free; covers 1872→present; includes the `neutral` flag.
  Maps directly to the `match_results` schema (§7).
- **fifa_ratings** — the fifa.com ranking archive (point-in-time by release
  date) or a Kaggle historical mirror. Free; take the snapshot in force at the
  as-of date (point-in-time, no look-ahead).
- **team_xg** — FBref / StatsBomb xG where freely available (recent major
  tournaments and big-confederation qualifiers). Understat is club-only and
  unusable for national teams. Where xG is absent, fall back to goals via the
  `build_features` blend (the blend already degrades to goals-only).
- **match_odds** — no clean free bulk dataset exists for internationals, so
  split the need:
  - *(a) tournament-match odds + market benchmark* — the BALLDONTLIE free World
    Cup API (2018 / 2022 / 2026).
  - *(b) form-window odds* — OddsPapi free tier (rate-limited; verify
    international historical coverage), or graceful degradation
    (`build_features` already skips matches with no odds), with a World Football
    Elo-implied probability proxy only as a last-resort gap filler.
  - Do **not** use api-football's free tier for historical odds — it exposes
    only the last 7 days.
- **Sharp anchor** — prefer the sharpest available book's closing line
  (Pinnacle) where the free data has it; consensus fallback otherwise. On free
  international data a sharp book is often unavailable, so consensus / proxy
  usually dominates, but the prefer-sharp-else-consensus logic still applies.

### Leakage role of tournament data
For a run with as-of date **T**, only data dated **strictly before T** are model
inputs. The predicted tournament's own results and pre-match odds are used
**only** as scoring labels and the market benchmark at evaluation (§10) —
**never** as features. A tournament's data therefore flips role by run: the 2022
World Cup is labels-only for the backtest, but a legitimate historical input for
the 2026 run.