# WC2026 dataset export

Generated 2026-07-20 08:36 by scripts/export_wc2026_dataset.py from the OpenOdds26 model
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
