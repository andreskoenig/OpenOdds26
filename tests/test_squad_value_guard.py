"""Correctness guard: c_v=0 must reproduce the tuned surprise-OFF baseline.

With the squad-value prior weight c_v=0, the squad-value term drops out of the
fit, so the WC2022 match-level log-loss must equal the known tuned baseline
(1.0235). If it does not, the squad-value wiring leaked into the c_v=0 path.

This runs the real fit (slow, ~1 min) with n_sims=1 (log-loss is independent of
the simulation). It is skipped if the assembled data files are not present.
"""

import json
import os
from datetime import date

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = ["data/teams.json", "data/match_results.json", "data/fifa_ratings.json",
            "data/match_odds.json", "data/squad_values.json",
            "config/tournament_config_2022.json"]

pytestmark = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ROOT, p)) for p in REQUIRED),
    reason="assembled data files not present",
)


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def test_cv_zero_reproduces_tuned_baseline_logloss():
    from wc_model.pipeline import backtest
    from wc_model.schemas import Hyperparams

    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    all_odds = _load("data/match_odds.json")
    squad_values = _load("data/squad_values.json")
    config = _load("config/tournament_config_2022.json")

    as_of = "2022-11-19"
    cutoff = date.fromisoformat(as_of)
    cfg_ids = {t for g in config["groups"].values() for t in g}

    from collections import Counter
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cutoff:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= 50} | cfg_ids
    teams = [t for t in teams_all if t["team_id"] in eligible]

    actual = [m for m in matches
              if m["competition"] == "FIFA World Cup"
              and cutoff < date.fromisoformat(m["date"]) <= date(2022, 12, 31)]

    # Tuned base hyperparameters with the squad-value prior OFF (c_v=0).
    hp = Hyperparams(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30,
                     c_y=0.10, theta=0.0, kappa=0.0, c_v=0.0)

    report = backtest(
        as_of, teams, config, matches, fifa,
        team_xg=[], match_odds=all_odds, actual_results=actual,
        tournament_market_odds=None,
        hyperparams=hp, squad_values=squad_values, n_sims=1, seed=20221119,
    )
    assert abs(report.match_metrics["log_loss"] - 1.0235) < 1e-3
