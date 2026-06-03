"""Unit tests for fit_model (SPEC §4). Synthetic data generated from known params."""

import math

import numpy as np

from wc_model.model import fit_model, matchup_matrix, score_matrix


# --- helpers ---------------------------------------------------------------

def _team(tid):
    return {"team_id": tid}


def _match(mid, dte, home, away, hg, ag, neutral):
    return {
        "match_id": mid,
        "date": dte,
        "home_team_id": home,
        "away_team_id": away,
        "venue_country": "X",
        "neutral": neutral,
        "competition": "F",
        "home_goals": int(hg),
        "away_goals": int(ag),
    }


def _zero_features(team_ids):
    return [
        {
            "team_id": t,
            "z_fifa": 0.0,
            "attack_index": 0.0,
            "defense_index": 0.0,
            "upset_propensity": 0.0,
            "market_adj_perf": 0.0,
        }
        for t in team_ids
    ]


def _sample(p, rng):
    idx = rng.choice(p.size, p=p.ravel())
    return divmod(int(idx), p.shape[1])


# --- known generating model ------------------------------------------------

TEAM_IDS = [f"t{i}" for i in range(6)]
TRUE_ATK = dict(zip(TEAM_IDS, [0.30, 0.20, 0.00, -0.10, -0.15, -0.25]))  # sum 0
TRUE_DEF = dict(zip(TEAM_IDS, [0.25, -0.05, 0.10, -0.20, 0.05, -0.15]))  # sum 0
TRUE_MU = math.log(1.3)
TRUE_GAMMA = 0.25
TRUE_RHO = -0.06


def _generate_round_robin(rounds, seed):
    rng = np.random.default_rng(seed)
    results = []
    mid = 0
    base = np.datetime64("2018-01-01")
    for r in range(rounds):
        for i, home in enumerate(TEAM_IDS):
            for j, away in enumerate(TEAM_IDS):
                if i == j:
                    continue
                p = score_matrix(
                    TRUE_ATK[home], TRUE_DEF[home], TRUE_ATK[away], TRUE_DEF[away],
                    TRUE_MU, TRUE_GAMMA, TRUE_RHO, home=True,
                )
                hg, ag = _sample(p, rng)
                dte = str((base + np.timedelta64(mid, "D")))[:10]
                results.append(_match(f"m{mid}", dte, home, away, hg, ag, neutral=False))
                mid += 1
    return results


_RECOVERY_RESULTS = _generate_round_robin(rounds=15, seed=12345)
_AS_OF = "2030-01-01"


def _fit_recovery(**overrides):
    kw = dict(xi=0.0, lambda_reg=1e-3, c_a=0.0, c_x=0.0, c_d=0.0, c_y=0.0, theta=0.0)
    kw.update(overrides)
    return fit_model(_AS_OF, [_team(t) for t in TEAM_IDS], _RECOVERY_RESULTS,
                     _zero_features(TEAM_IDS), **kw)


# --- tests -----------------------------------------------------------------

def test_generative_parameter_recovery():
    params = _fit_recovery()

    # atk/def are identified only up to a global location; the ridge-to-0 prior
    # picks the centered gauge, so re-center both sides before comparing.
    atk = np.array([params.atk[t] for t in TEAM_IDS])
    deff = np.array([params.def_[t] for t in TEAM_IDS])
    atk -= atk.mean()
    deff -= deff.mean()
    true_atk = np.array([TRUE_ATK[t] for t in TEAM_IDS])
    true_def = np.array([TRUE_DEF[t] for t in TEAM_IDS])

    assert np.max(np.abs(atk - true_atk)) < 0.15
    assert np.max(np.abs(deff - true_def)) < 0.15
    assert abs(params.gamma - TRUE_GAMMA) < 0.12
    assert abs(params.rho - TRUE_RHO) < 0.08
    assert abs(params.mu - TRUE_MU) < 0.2


def test_rho_in_bounds_and_score_matrix_assert_never_trips():
    params = _fit_recovery()
    assert -0.2 <= params.rho <= 0.2
    # Building every matchup must not trip score_matrix's negative-cell assert.
    for h in TEAM_IDS:
        for a in TEAM_IDS:
            if h == a:
                continue
            p = matchup_matrix(params, h, a, home_flag=True)
            assert abs(p.sum() - 1.0) < 1e-9


def test_deterministic_same_inputs_same_params():
    a = _fit_recovery()
    b = _fit_recovery()
    assert a.mu == b.mu and a.gamma == b.gamma and a.rho == b.rho
    assert a.atk == b.atk and a.def_ == b.def_


def test_high_lambda_reg_pulls_fit_toward_priors():
    teams = [_team("A"), _team("B")]
    feats = [
        {"team_id": "A", "z_fifa": 1.0, "attack_index": 0.0, "defense_index": 0.5,
         "upset_propensity": 0.0, "market_adj_perf": 0.0},
        {"team_id": "B", "z_fifa": -1.0, "attack_index": 0.0, "defense_index": -0.5,
         "upset_propensity": 0.0, "market_adj_perf": 0.0},
    ]
    results = [
        _match("m0", "2020-01-01", "A", "B", 1, 0, neutral=True),
        _match("m1", "2020-01-02", "B", "A", 2, 1, neutral=True),
    ]
    params = fit_model("2021-01-01", teams, results, feats,
                       xi=0.0, lambda_reg=1e6, c_a=1.0, c_x=0.0, c_d=1.0, c_y=1.0)
    # Priors: atk = c_a*z = +/-1; def = c_d*z + c_y*defense_index = +/-1.5.
    assert abs(params.atk["A"] - 1.0) < 1e-2
    assert abs(params.atk["B"] - (-1.0)) < 1e-2
    assert abs(params.def_["A"] - 1.5) < 1e-2
    assert abs(params.def_["B"] - (-1.5)) < 1e-2


def test_higher_xi_shifts_fit_toward_recent_results():
    teams = [_team("A"), _team("B")]
    feats = _zero_features(["A", "B"])
    results = []
    mid = 0
    # Old block: B dominates A (venue-symmetric so home advantage cancels).
    for _ in range(8):
        results.append(_match(f"o{mid}", "2020-01-15", "B", "A", 3, 0, neutral=True)); mid += 1
        results.append(_match(f"o{mid}", "2020-01-16", "A", "B", 0, 3, neutral=True)); mid += 1
    # Recent block: A dominates B.
    for _ in range(8):
        results.append(_match(f"r{mid}", "2020-12-15", "A", "B", 3, 0, neutral=True)); mid += 1
        results.append(_match(f"r{mid}", "2020-12-16", "B", "A", 0, 3, neutral=True)); mid += 1

    as_of = "2021-01-01"
    common = dict(lambda_reg=1e-3, c_a=0.0, c_x=0.0, c_d=0.0, c_y=0.0)
    low = fit_model(as_of, teams, results, feats, xi=0.0, **common)
    high = fit_model(as_of, teams, results, feats, xi=0.03, **common)

    d_low = low.atk["A"] - low.atk["B"]
    d_high = high.atk["A"] - high.atk["B"]
    assert abs(d_low) < 0.1          # equal weighting -> symmetric
    assert d_high > d_low            # recent (A strong) pulls A above B


def test_positive_theta_with_positive_M_raises_rating():
    teams = [_team("A"), _team("B")]
    feats = [
        {"team_id": "A", "z_fifa": 0.0, "attack_index": 0.0, "defense_index": 0.0,
         "upset_propensity": 0.0, "market_adj_perf": 1.0},
        {"team_id": "B", "z_fifa": 0.0, "attack_index": 0.0, "defense_index": 0.0,
         "upset_propensity": 0.0, "market_adj_perf": 0.0},
    ]
    # Symmetric, neutral draws so the likelihood does not favor either team.
    results = [
        _match("m0", "2020-01-01", "A", "B", 1, 1, neutral=True),
        _match("m1", "2020-01-02", "B", "A", 1, 1, neutral=True),
    ]
    common = dict(as_of_date="2021-01-01", teams=teams, match_results=results,
                  feature_records=feats, xi=0.0, lambda_reg=1.0,
                  c_a=0.0, c_x=0.0, c_d=0.0, c_y=0.0)
    no_theta = fit_model(**common, theta=0.0)
    with_theta = fit_model(**common, theta=0.5)
    assert with_theta.atk["A"] > no_theta.atk["A"]
