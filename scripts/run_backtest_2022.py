"""Baseline World Cup 2022 backtest (deterministic; reads cached free data only).

FREE data only: martj42 results + historical FIFA ranking points (fetched
separately into data/). team_xg and match_odds are EMPTY this run -> goals-based
attack/defense indices, surprise factor OFF (kappa=0, theta=0). No fetching here.

as-of date = 2022-11-19 (day before the opener); the harness uses only data
strictly before it as inputs, and the 2022 World Cup matches purely as scoring
labels.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # make the wc_model package importable

from wc_model.pipeline import backtest
from wc_model.schemas import Hyperparams
AS_OF = "2022-11-19"
CUTOFF = date.fromisoformat(AS_OF)
WC_END = date(2022, 12, 31)
MIN_MATCHES = 50   # fit ratings only for teams with this many pre-cutoff games
N_SIMS = 10000
SEED = 20221119

# Base hyperparameters from the FREE validation sweep (scripts/tune_hparams_2022.py,
# val log-loss 0.9577): FIFA-heavy prior + stronger shrinkage + slower decay.
# Surprise factor OFF (kappa/theta sweep found no generalizable gain).
# Squad-value prior c_v=0.1 = best of scripts/tune_squad_value_2022.py (val 0.9573).
# Override with --kappa / --theta / --cv.
HP = Hyperparams(
    xi=0.0008,        # slower time decay (more history)
    lambda_reg=8.0,   # stronger ridge pull toward priors (compresses spread)
    c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,   # FIFA-prior heavy vs goals index
    theta=0.0, kappa=0.0, c_v=0.1, blend_weight=0.7, n_recent=10,   # squad-value prior ON
    opponent_adjust=False, max_history_years=0.0,   # off by default (old shipped config)
)


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    config = _load("config/tournament_config_2022.json")
    # All fetched Bet365 closing odds (form-window + the 64 WC). build_features uses
    # only rows strictly before the as-of date (form-window) as the U/M input; the
    # post-cutoff WC rows are auto-excluded there and used solely as the benchmark.
    all_odds = _load("data/match_odds.json")
    squad_values = _load("data/squad_values.json")   # point-in-time squad-value prior

    # Levers: chosen defaults in HP, overridable via --kappa / --theta / --cv.
    if "--kappa" in sys.argv:
        HP.kappa = float(sys.argv[sys.argv.index("--kappa") + 1])
    if "--theta" in sys.argv:
        HP.theta = float(sys.argv[sys.argv.index("--theta") + 1])
    if "--cv" in sys.argv:
        HP.c_v = float(sys.argv[sys.argv.index("--cv") + 1])
    if "--xi" in sys.argv:
        HP.xi = float(sys.argv[sys.argv.index("--xi") + 1])
    if "--trunc" in sys.argv:
        HP.max_history_years = float(sys.argv[sys.argv.index("--trunc") + 1])
    if "--oppadj" in sys.argv:
        HP.opponent_adjust = True
    if "--friendly-weight" in sys.argv:
        HP.friendly_weight = float(sys.argv[sys.argv.index("--friendly-weight") + 1])
    drop_friendlies = "--no-friendlies" in sys.argv
    if drop_friendlies:
        before = len(matches)
        matches = [m for m in matches if m["competition"] != "Friendly"]
        print(f"--no-friendlies: dropped {before - len(matches)} friendlies "
              f"(fit uses {len(matches)} matches; WC2022 labels unaffected)")
    team_xg = []
    if "--xg" in sys.argv:
        team_xg = _load("data/team_xg.json")["rows"]   # StatsBomb xG joined to our match_ids
    n_sims = N_SIMS
    if "--sims" in sys.argv:
        n_sims = int(sys.argv[sys.argv.index("--sims") + 1])
    surprise_on = HP.kappa != 0.0 or HP.theta != 0.0

    # Config now carries canonical slug team_ids directly (no remap needed).
    team_ids_all = {t["team_id"] for t in teams_all}
    cfg_ids = {t for g in config["groups"].values() for t in g}
    bad = sorted(t for t in cfg_ids if t not in team_ids_all)
    if bad:
        raise SystemExit(f"FATAL: config group team_ids not in teams.json: {bad}")

    # Eligible teams for the rating fit: enough pre-cutoff fixtures. Always keep
    # the 32 World Cup teams. (All 336 teams are loaded; we just don't fit a
    # meaningless rating for obscure/defunct/sub-threshold entities.)
    pre = [m for m in matches if date.fromisoformat(m["date"]) < CUTOFF]
    cnt = Counter()
    for m in pre:
        cnt[m["home_team_id"]] += 1
        cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES} | cfg_ids
    teams = [t for t in teams_all if t["team_id"] in eligible]

    # Actual labels: the 2022 World Cup matches (after the cutoff, in 2022).
    actual = [
        m for m in matches
        if m["competition"] == "FIFA World Cup"
        and CUTOFF < date.fromisoformat(m["date"]) <= WC_END
    ]
    wc_ids = {m["match_id"] for m in actual}
    wc_odds = [r for r in all_odds if r["match_id"] in wc_ids]          # 64-game benchmark
    form_odds = [r for r in all_odds if r["match_id"] not in wc_ids]    # U/M input (pre-cutoff)

    bits = []
    if surprise_on:
        bits.append("surprise ON")
    if HP.c_v != 0.0:
        bits.append(f"squad-value prior c_v={HP.c_v}")
    mode = "; ".join(bits) if bits else "tuned baseline (no surprise, no squad prior)"
    print("=" * 78)
    print(f"World Cup 2022 backtest (free data; goals-based; {mode})")
    print("=" * 78)
    print(f"as-of date           : {AS_OF}")
    print(f"teams loaded         : {len(teams_all)}  | fitted (>= {MIN_MATCHES} games): {len(teams)}")
    print(f"match_results        : {len(matches)}  | strictly pre-cutoff: {len(pre)}")
    print(f"fifa_ratings snaps   : {len(fifa)}")
    print(f"actual WC22 matches  : {len(actual)} (scoring labels)")
    print(f"form-window odds     : {len(form_odds)} rows -> build_features U/M input (Bet365, pre-cutoff)")
    print(f"benchmark odds       : {len(wc_odds)} rows -> WC closing (de-vigged in backtest)")
    print(f"squad-value snaps    : {len(squad_values)} rows -> z_squad_value (c_v prior)")
    print(f"team_xg rows         : {len(team_xg)} ({'StatsBomb xG ON' if team_xg else 'OFF -> pure goals'})")
    print(f"hyperparams          : xi={HP.xi} lambda_reg={HP.lambda_reg} "
          f"c_a/c_x/c_d/c_y={HP.c_a}/{HP.c_x}/{HP.c_d}/{HP.c_y} "
          f"theta={HP.theta} kappa={HP.kappa} c_v={HP.c_v} "
          f"opp_adj={HP.opponent_adjust} trunc={HP.max_history_years}y")
    print(f"sims                 : {N_SIMS} (seed {SEED})")
    print("\nfitting + simulating (this takes a few minutes) ...\n")

    report = backtest(
        AS_OF, teams, config, matches, fifa,
        team_xg=team_xg,
        match_odds=all_odds,   # U/M input; build_features filters to pre-cutoff (form only)
        actual_results=actual,
        tournament_market_odds=wc_odds,   # de-vigged Bet365 closing = benchmark
        hyperparams=HP, squad_values=squad_values, n_sims=n_sims, seed=SEED,
    )

    p = report.prediction.params
    print(f"fitted globals       : mu={p.mu:.4f}  gamma={p.gamma:.4f}  rho={p.rho:.4f}")

    mm = report.match_metrics
    mk = mm.get("market")
    print("\n" + "-" * 78)
    print(f"MATCH-LEVEL HEAD-TO-HEAD over {len(actual)} World Cup games (lower is better)")
    print("-" * 78)
    print(f"  {'metric':<10}{'model':>12}{'market':>12}{'gap (model-market)':>22}")
    if mk is not None:
        for key, label in (("log_loss", "log-loss"), ("brier", "Brier"), ("rps", "RPS")):
            gap = mm[key] - mk[key]
            print(f"  {label:<10}{mm[key]:>12.4f}{mk[key]:>12.4f}{gap:>+22.4f}")
        print(f"\n  (uniform 1X2 log-loss baseline = {math.log(3):.4f}; "
              f"market = de-vigged Bet365 closing)")
    else:
        print("  market metrics unavailable (no aligned odds) -- model only:")
        for key, label in (("log_loss", "log-loss"), ("brier", "Brier"), ("rps", "RPS")):
            print(f"  {label:<10}{mm[key]:>12.4f}")

    print("\n" + "-" * 78)
    print("SIMULATED P(win) -- top 10")
    print("-" * 78)
    ranked = sorted(report.p_win.items(), key=lambda kv: kv[1], reverse=True)
    for i, (tid, pw) in enumerate(ranked[:10], 1):
        print(f"  {i:>2}. {tid:<20} {pw * 100:5.1f}%")

    arg = report.p_win.get("argentina")
    print("\n" + "-" * 78)
    print("SANITY")
    print("-" * 78)
    print(f"  Argentina P(win)   : {arg * 100:.1f}%" if arg is not None
          else "  Argentina P(win)   : (not in field)")
    if report.champion_p_win is not None:
        print(f"  actual champion    : {report.champion} "
              f"(predicted P(win) {report.champion_p_win * 100:.1f}%)")
    else:
        print(f"  actual champion    : {report.champion}")


if __name__ == "__main__":
    main()
