"""End-to-end pipeline / backtest tests (SPEC §9-§10). Synthetic data only."""

import math
from datetime import date, timedelta

import numpy as np

from wc_model.model import result_probs, score_matrix
from wc_model.pipeline import backtest, run_prediction
from wc_model.schemas import Hyperparams


# --- known generating model (t0 clearly strongest) -------------------------

TEAM_IDS = [f"t{i}" for i in range(8)]
TRUE_ATK = dict(zip(TEAM_IDS, [0.60, 0.20, 0.10, 0.00, -0.10, -0.20, -0.20, -0.40]))
TRUE_DEF = dict(zip(TEAM_IDS, [0.50, 0.10, 0.05, 0.00, -0.05, -0.10, -0.20, -0.30]))
TRUE_MU = math.log(1.3)
TRUE_GAMMA = 0.20
TRUE_RHO = -0.05

AS_OF = "2025-01-01"
CONFIG = {
    "groups": {"A": TEAM_IDS[0:4], "B": TEAM_IDS[4:8]},
    "advance_per_group": 2,
    "best_thirds": 0,
}

HP = Hyperparams(xi=0.0, lambda_reg=1e-3, c_a=0.0, c_x=0.0, c_d=0.0, c_y=0.0)


def _true_matrix(home, away, home_flag):
    return score_matrix(
        TRUE_ATK[home], TRUE_DEF[home], TRUE_ATK[away], TRUE_DEF[away],
        TRUE_MU, TRUE_GAMMA, TRUE_RHO, home=home_flag,
    )


def _true_probs(home, away, home_flag):
    return result_probs(_true_matrix(home, away, home_flag))


def _sample(p, rng):
    idx = rng.choice(p.size, p=p.ravel())
    return divmod(int(idx), p.shape[1])


def _match(mid, dte, home, away, hg, ag, neutral):
    return {
        "match_id": mid, "date": dte, "home_team_id": home, "away_team_id": away,
        "venue_country": "X", "neutral": neutral, "competition": "F",
        "home_goals": int(hg), "away_goals": int(ag),
    }


def _fair_odds(mid, p_home, p_draw, p_away, book="b1"):
    return {
        "match_id": mid, "bookmaker": book,
        "odds_home": 1.0 / p_home, "odds_draw": 1.0 / p_draw, "odds_away": 1.0 / p_away,
        "captured_at": "2000-01-01",
    }


def build_world(rounds=8, seed=7):
    """Pre-cutoff history sampled from the true model, plus fifa/xg/odds tables."""
    rng = np.random.default_rng(seed)
    teams = [{"team_id": t} for t in TEAM_IDS]
    results, team_xg, match_odds = [], [], []
    base = date(2018, 1, 1)
    mid = 0
    for _ in range(rounds):
        for i, home in enumerate(TEAM_IDS):
            for j, away in enumerate(TEAM_IDS):
                if i == j:
                    continue
                p = _true_matrix(home, away, True)
                hg, ag = _sample(p, rng)
                m_id = f"m{mid}"
                dte = (base + timedelta(days=mid)).isoformat()
                results.append(_match(m_id, dte, home, away, hg, ag, neutral=False))
                team_xg.append({"match_id": m_id, "team_id": home, "xg_for": float(hg), "xg_against": float(ag)})
                team_xg.append({"match_id": m_id, "team_id": away, "xg_for": float(ag), "xg_against": float(hg)})
                ph, pd, pa = result_probs(p)
                match_odds.append(_fair_odds(m_id, ph, pd, pa))
                mid += 1
    fifa_ratings = [
        {"team_id": t, "as_of_date": "2024-12-01",
         "fifa_points": 1000.0 + 500.0 * (TRUE_ATK[t] + TRUE_DEF[t]), "fifa_rank": 1}
        for t in TEAM_IDS
    ]
    return teams, results, fifa_ratings, team_xg, match_odds


WORLD = build_world()


def _run(seed=0, n_sims=200, world=None):
    teams, results, fifa, xg, odds = world if world is not None else WORLD
    return run_prediction(
        AS_OF, teams, CONFIG, results, fifa, xg, odds,
        hyperparams=HP, n_sims=n_sims, seed=seed,
    )


# --- tests -----------------------------------------------------------------

def test_end_to_end_smoke_p_win_sums_to_one():
    res = _run(seed=1, n_sims=200)
    assert set(res.p_win) == set(TEAM_IDS)
    assert abs(sum(res.p_win.values()) - 1.0) < 1e-9
    # predict_match returns a usable scoreline + normalized 1X2.
    mp = res.predict_match("t0", "t7", home_flag=False)
    assert mp["matrix"].shape == (11, 11)
    assert abs(sum(mp["probs"]) - 1.0) < 1e-9


def test_integration_leak_guard_post_cutoff_rows_change_nothing():
    teams, results, fifa, xg, odds = WORLD

    # Post-cutoff rows that WOULD shift the fit/features if they leaked through.
    poison_results = list(results) + [
        _match("future1", "2025-06-01", "t7", "t0", 9, 0, neutral=False)
    ]
    poison_xg = list(xg) + [
        {"match_id": "future1", "team_id": "t7", "xg_for": 9.0, "xg_against": 0.0},
        {"match_id": "future1", "team_id": "t0", "xg_for": 0.0, "xg_against": 9.0},
    ]
    poison_odds = list(odds) + [_fair_odds("future1", 0.01, 0.04, 0.95)]
    poison_fifa = list(fifa) + [
        {"team_id": "t7", "as_of_date": "2025-03-01", "fifa_points": 99999.0, "fifa_rank": 1}
    ]

    clean = run_prediction(AS_OF, teams, CONFIG, results, fifa, xg, odds,
                           hyperparams=HP, n_sims=150, seed=42)
    poisoned = run_prediction(AS_OF, teams, CONFIG, poison_results, poison_fifa,
                              poison_xg, poison_odds, hyperparams=HP, n_sims=150, seed=42)

    assert poisoned.params == clean.params
    assert poisoned.p_win == clean.p_win
    assert poisoned.progression == clean.progression


def test_stronger_team_has_highest_simulated_p_win():
    res = _run(seed=3, n_sims=400)
    best = max(res.p_win, key=res.p_win.get)
    assert best == "t0"


def test_reproducibility_same_seed_identical_results():
    a = _run(seed=11, n_sims=200)
    b = _run(seed=11, n_sims=200)
    assert a.p_win == b.p_win
    assert a.progression == b.progression


def test_hyperparams_threading_same_xi_reaches_features_and_fit():
    hp = Hyperparams(xi=0.037, lambda_reg=1e-2, c_a=0.1, c_x=0.1, c_d=0.1, c_y=0.1)
    # The single object projects the SAME xi into both stages.
    assert hp.feature_kwargs()["xi"] == hp.fit_kwargs()["xi"] == hp.xi
    teams, results, fifa, xg, odds = WORLD
    res = run_prediction(AS_OF, teams, CONFIG, results, fifa, xg, odds,
                         hyperparams=hp, n_sims=50, seed=0)
    # fit_model records the xi it weighted the likelihood with; it must match.
    assert res.params.xi == hp.xi


# --- backtest: match-conditional scoring vs market -------------------------

def _build_actual_and_markets(n=200, seed=99):
    """Neutral actual fixtures sampled from the true model + true/bad markets."""
    rng = np.random.default_rng(seed)
    canonical = {"home": (1, 0), "draw": (1, 1), "away": (0, 1)}
    actual, good_market, bad_market = [], [], []
    for k in range(n):
        i, j = rng.choice(len(TEAM_IDS), size=2, replace=False)
        home, away = TEAM_IDS[i], TEAM_IDS[j]
        ph, pd, pa = _true_probs(home, away, False)
        outcome = rng.choice(["home", "draw", "away"], p=[ph, pd, pa])
        hg, ag = canonical[outcome]
        mid = f"a{k}"
        actual.append(_match(mid, "2025-02-01", home, away, hg, ag, neutral=True))
        good_market.append(_fair_odds(mid, ph, pd, pa))         # near-optimal
        bad_market.append(_fair_odds(mid, 0.8, 0.1, 0.1))       # miscalibrated
    return actual, good_market, bad_market


def test_backtest_model_is_in_market_ballpark_and_beats_bad_market():
    teams, results, fifa, xg, odds = WORLD
    actual, good_market, bad_market = _build_actual_and_markets()

    common = dict(
        as_of_date=AS_OF, teams=teams, tournament_config=CONFIG,
        match_results=results, fifa_ratings=fifa, team_xg=xg, match_odds=odds,
        actual_results=actual, hyperparams=HP, n_sims=50, seed=0,
    )
    good = backtest(tournament_market_odds=good_market, **common)
    bad = backtest(tournament_market_odds=bad_market, **common)

    model_ll = good.match_metrics["log_loss"]
    good_market_ll = good.match_metrics["market"]["log_loss"]
    bad_market_ll = bad.match_metrics["market"]["log_loss"]

    assert math.isfinite(model_ll)
    # Same model in both runs -> identical model metrics.
    assert bad.match_metrics["log_loss"] == model_ll
    # The model is in the same ballpark as the near-optimal (true) market...
    assert abs(model_ll - good_market_ll) < 0.2
    # ...and clearly beats a deliberately miscalibrated market.
    assert model_ll < bad_market_ll
    # Sanity figure is a valid probability.
    assert 0.0 <= good.champion_p_win <= 1.0
