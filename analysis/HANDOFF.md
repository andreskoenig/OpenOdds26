# WC2026 dataset export

Generated 2026-07-20 08:49 by scripts/export_wc2026_dataset.py from the OpenOdds26 model
(https://github.com/andreskoenig/OpenOdds26). 104 matches: 72 group + 32 knockout.

## wc2026_matches.csv / .json — one row per match

- match_id, stage (group|R32|R16|QF|SF|third_place|final), group, date (kickoff, ESPN scoreboard day)
- home_id/away_id/home_name/away_name — home = nominal home side of the fixture
- actual_home_goals/actual_away_goals — final recorded score (group: 90'; knockout: after extra time if played)
- status — FT (decided in 90'), AET (decided in extra time), PEN (penalty shootout)
- outcome_90 — home/draw/away at regulation (AET/PEN games = draw). Bookmaker 1X2 settles on this.
- outcome_120 — home/draw/away at the end of extra time (PEN games = draw). Group games: same as outcome_90.
- advancer_id — knockout only: the team that progressed (resolves shootouts)
- model_p_*_90 — model's pre-match regulation 1X2 probabilities
- model_p_*_120 — knockout only: model's 1X2 by end of extra time (draw leg = goes to penalties)
- model_p_home_advance/model_p_away_advance — knockout only: P(progress), extra time + penalties folded in
- model_xg_home/away — model expected goals (90')
- model_modal_home/away (+ model_modal_p) — most likely scoreline (group: 90'; knockout: 120' distribution)
- model_pred_outcome — argmax of the model 1X2 (group: 90'; knockout: 120')
- bet365_*/pinnacle_* — closing decimal odds (last pre-kickoff snapshot, OddsPapi) + de-vigged
  probabilities (proportional normalization). Settle on outcome_90.
- model_1x2_correct — group: pred vs outcome_90; knockout: pred vs outcome_120
- model_exact_correct — modal scoreline == actual score
- model_advance_correct — knockout: model favourite by P(advance) == advancer_id
- model_log_loss / model_brier — model 1X2 scored on its basis (group 90' / knockout 120')

All predictions were FROZEN before each match kicked off (walk-forward; knockout
rounds frozen the day the previous round completed). No look-ahead.

## wc2026_pwin_history.csv

P(win tournament) snapshot after each completed round (knockout-conditioned
bracket propagation from R32 on). Columns: date, round, team_id, team_name, p_win.
Eliminated teams are absent from later snapshots (i.e. 0).

## wc2026_kpis.json

Closing aggregate metrics as shown on the dashboard:
- kpis — headline over all 104 games: accuracy_1x2, model_exact_rate,
  mean_log_loss, mean_brier (group games scored at 90'; knockout at 120').
- kpis_market — per book (Bet365, Pinnacle): n matched games, the book's
  accuracy/log-loss/brier, the model's SAME-GAME numbers (model_*), and deltas
  (d_* = book minus model). All on the 90' basis (books settle regulation time).
- knockout_summary — knockout-only: n ties, advance accuracy, 1X2@120 accuracy,
  modal (exact score) rate.
- calibration — favourite-probability buckets vs realized favourite win rate.

## wc2026_market_odds_raw.json

Verbatim copy of the closing-odds archive (OddsPapi /v4/historical-odds;
closing = last snapshot at or before kickoff). Per fixture: kickoff (UTC date,
can sit +1 day vs the ESPN scoreboard date), home_id/away_id in the PROVIDER'S
orientation (may be flipped vs the match dataset - the CSV already reorients),
and raw decimal 1X2 per book. Kept so the de-vig step is reproducible.

## Reading guide / caveats for analysis

- Two scoring bases coexist by design. Headline model metrics score knockout
  games at 120' ("did the model call the game"); every model-vs-book comparison
  is at 90' because that is what a 1X2 market settles on. Use outcome_90 with
  the book probabilities, outcome_120 with model_p_*_120.
- A knockout game with status PEN is a draw at BOTH bases (level after 120);
  AET games are a draw at 90' and decisive at 120'. The final (Spain 1-0
  Argentina, AET) is a 90'-draw / 120'-home-win.
- Model exact-score is the modal scoreline of a full bivariate distribution -
  the modal probability (model_modal_p) is the fair yardstick, typically 9-19%.
- The Polymarket winner prior is a MODEL INPUT (weight c_m=0.35), so
  wc2026_pwin_history is not fully independent of the market. Bet365/Pinnacle
  per-match odds are independent benchmarks - the model never saw them.
- The 2022 backtest (see repo README) is development-set grade; this 2026 file
  is the only true out-of-sample record of the model.
- Suggested analyses: model vs book log-loss/Brier with paired tests (same 104
  games), calibration curves from the probability columns, skill by stage
  (group vs knockout), upset analysis (low-probability outcomes that landed),
  and P(win) trajectory vs the market over the 9 snapshots.
