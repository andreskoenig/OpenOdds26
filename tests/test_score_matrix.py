"""Unit tests for the Dixon–Coles scoreline matrix (SPEC §3, §5).

Hand-entered synthetic ratings only (no real teams).
"""

import math

import numpy as np

from wc_model.model import (
    btts,
    correct_score,
    over_under,
    result_probs,
    score_matrix,
)

GOALS = np.arange(11)


def _home_marginal(p):
    return p.sum(axis=1)


def _away_marginal(p):
    return p.sum(axis=0)


def _mean(marginal):
    return float((GOALS * marginal).sum())


def _var(marginal):
    m = _mean(marginal)
    return float((GOALS * GOALS * marginal).sum()) - m * m


def test_matrix_sums_to_one_across_rating_combos():
    combos = [
        dict(atk_i=0.0, def_i=0.0, atk_j=0.0, def_j=0.0),
        dict(atk_i=0.5, def_i=0.2, atk_j=-0.3, def_j=0.1),
        dict(atk_i=-0.4, def_i=0.4, atk_j=0.6, def_j=-0.2),
    ]
    for c in combos:
        for rho in (0.0, 0.05, 0.1):
            for home in (False, True):
                p = score_matrix(mu=0.1, gamma=0.3, rho=rho, home=home, **c)
                assert p.shape == (11, 11)
                assert p.min() >= 0.0
                assert abs(p.sum() - 1.0) < 1e-9


def test_identical_teams_on_neutral_ground_are_symmetric():
    # Same atk/def, neutral (home=False so gamma is not applied).
    p = score_matrix(
        atk_i=0.3, def_i=0.1, atk_j=0.3, def_j=0.1,
        mu=0.0, gamma=0.4, rho=0.05, home=False,
    )
    home, draw, away = result_probs(p)
    assert abs(home - away) < 1e-12
    assert draw > 0.0


def test_stronger_attack_raises_win_prob_and_expected_goals():
    base = dict(def_i=0.0, atk_j=0.0, def_j=0.0, mu=0.0, gamma=0.0, rho=0.0, home=False)
    p_weak = score_matrix(atk_i=0.0, **base)
    p_strong = score_matrix(atk_i=0.8, **base)

    home_weak, _, _ = result_probs(p_weak)
    home_strong, _, _ = result_probs(p_strong)
    assert home_strong > home_weak

    assert _mean(_home_marginal(p_strong)) > _mean(_home_marginal(p_weak))


def test_marginal_means_recover_lambdas():
    mu, atk_i, def_j, atk_j, def_i = 0.0, 0.2, 0.1, -0.1, 0.05
    p = score_matrix(
        atk_i=atk_i, def_i=def_i, atk_j=atk_j, def_j=def_j,
        mu=mu, gamma=0.0, rho=0.0, home=False,
    )
    lam_home = math.exp(mu + atk_i - def_j)
    lam_away = math.exp(mu + atk_j - def_i)
    assert abs(_mean(_home_marginal(p)) - lam_home) < 1e-3
    assert abs(_mean(_away_marginal(p)) - lam_away) < 1e-3


def test_rho_shifts_low_score_mass_per_spec_signs():
    # SPEC §3: tau(0,0)=1-lam_h*lam_a*rho, tau(1,1)=1-rho  -> both < 1 for rho>0,
    # while tau(0,1)=1+lam_h*rho and tau(1,0)=1+lam_a*rho are > 1.
    # Therefore rho>0 LOWERS P(0,0) and P(1,1) and RAISES P(0,1) and P(1,0).
    # (The task prose described the opposite direction; SPEC §3 is the source of
    # truth, so the assertions follow the spec's tau signs.)
    kw = dict(atk_i=0.1, def_i=0.0, atk_j=0.0, def_j=0.05, mu=0.0, gamma=0.0, home=False)
    p0 = score_matrix(rho=0.0, **kw)
    pr = score_matrix(rho=0.1, **kw)

    assert correct_score(pr, 0, 0) < correct_score(p0, 0, 0)
    assert correct_score(pr, 1, 1) < correct_score(p0, 1, 1)
    assert correct_score(pr, 0, 1) > correct_score(p0, 0, 1)
    assert correct_score(pr, 1, 0) > correct_score(p0, 1, 0)


def test_negative_rho_raises_draw_mass_lowers_one_nil():
    # The fitted regime: rho < 0 (typically ~ -0.1 to -0.15 for football).
    # With SPEC §3's tau signs, negative rho RAISES P(0,0) and P(1,1) and LOWERS
    # P(1,0) and P(0,1) -- the draw-inflating Dixon-Coles effect.
    kw = dict(atk_i=0.1, def_i=0.0, atk_j=0.0, def_j=0.05, mu=0.0, gamma=0.0, home=False)
    p0 = score_matrix(rho=0.0, **kw)
    pn = score_matrix(rho=-0.1, **kw)

    assert correct_score(pn, 0, 0) > correct_score(p0, 0, 0)
    assert correct_score(pn, 1, 1) > correct_score(p0, 1, 1)
    assert correct_score(pn, 1, 0) < correct_score(p0, 1, 0)
    assert correct_score(pn, 0, 1) < correct_score(p0, 0, 1)


def test_dispersion_increases_goal_variance_same_mean():
    kw = dict(atk_i=0.2, def_i=0.0, atk_j=0.0, def_j=0.0, mu=0.0, gamma=0.0, rho=0.0, home=False)
    p_poisson = score_matrix(**kw)
    p_disp = score_matrix(dispersion_home=0.6, **kw)  # NB over-dispersion alpha

    m_pois = _home_marginal(p_poisson)
    m_disp = _home_marginal(p_disp)

    assert _var(m_disp) > _var(m_pois)
    # Same mean (negative binomial of equal mean), within truncation tolerance.
    assert abs(_mean(m_disp) - _mean(m_pois)) < 1e-2


def test_derived_markets_are_consistent_sums():
    p = score_matrix(
        atk_i=0.3, def_i=0.1, atk_j=0.1, def_j=0.2,
        mu=0.0, gamma=0.0, rho=0.05, home=False,
    )
    home, draw, away = result_probs(p)
    assert abs((home + draw + away) - 1.0) < 1e-9

    over, under = over_under(p, line=2.5)
    assert abs((over + under) - 1.0) < 1e-9  # no integer total equals 2.5

    assert 0.0 <= btts(p) <= 1.0
    assert abs(correct_score(p, 0, 0) - p[0, 0]) < 1e-15
