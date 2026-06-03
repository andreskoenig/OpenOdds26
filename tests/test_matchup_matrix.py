"""Unit tests for the matchup_matrix adapter (SPEC §3, §5)."""

import numpy as np

from wc_model.model import matchup_matrix
from wc_model.schemas import ModelParams


GOALS = np.arange(11)


def _home_marginal_var(p):
    m = p.sum(axis=1)
    mean = float((GOALS * m).sum())
    return float((GOALS * GOALS * m).sum()) - mean * mean


def _params(u_map):
    return ModelParams(
        mu=0.3,
        gamma=0.25,
        rho=-0.05,
        atk={"H": 0.2, "L": -0.1, "M": 0.0},
        def_={"H": 0.0, "L": 0.1, "M": 0.0},
        xi=0.0,
        U=u_map,
    )


def test_fitted_params_produce_valid_matrix():
    params = _params({"H": 1.0, "L": 1.0, "M": 1.0})
    p = matchup_matrix(params, "H", "L", home_flag=True)
    assert p.shape == (11, 11)
    assert p.min() >= 0.0
    assert abs(p.sum() - 1.0) < 1e-9


def test_kappa_widens_variance_for_high_u_team():
    # H has the highest U (above the field mean); kappa>0 must fatten its tails.
    params = _params({"H": 2.0, "L": 0.0, "M": 1.0})  # mean U = 1.0
    p_poisson = matchup_matrix(params, "H", "L", home_flag=False, kappa=0.0)
    p_dispersed = matchup_matrix(params, "H", "L", home_flag=False, kappa=1.0)

    assert _home_marginal_var(p_dispersed) > _home_marginal_var(p_poisson)
    assert abs(p_dispersed.sum() - 1.0) < 1e-9
