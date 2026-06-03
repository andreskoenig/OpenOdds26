"""Tests for the market -> scoreline core (wc_model.market)."""

import numpy as np

from wc_model.market import market_scoreline

RHO = -0.045  # a representative fitted rho


def test_reproduces_target_1x2():
    for ph, pd, pa in [(0.50, 0.27, 0.23), (0.80, 0.13, 0.07), (0.20, 0.25, 0.55),
                       (0.34, 0.33, 0.33), (0.62, 0.22, 0.16)]:
        out = market_scoreline(ph, pd, pa, RHO)
        rh, rd, ra = out["probs"]
        assert abs(rh - ph) < 1e-3, f"home {rh} vs {ph}"
        assert abs(ra - pa) < 1e-3, f"away {ra} vs {pa}"
        assert abs(rd - pd) < 1e-3, f"draw {rd} vs {pd}"  # draw implied by the other two


def test_matrix_is_valid_distribution():
    out = market_scoreline(0.55, 0.25, 0.20, RHO)
    P = out["matrix"]
    assert P.shape == (11, 11)
    assert P.min() >= 0.0
    assert abs(P.sum() - 1.0) < 1e-9


def test_derived_markets_consistent():
    out = market_scoreline(0.45, 0.30, 0.25, RHO)
    assert len(out["top_scores"]) == 6
    # top score is the global argmax
    assert out["top_scores"][0][0] == out["most_likely_score"]
    # top_scores sorted descending
    probs = [p for _, p in out["top_scores"]]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 <= out["over_2_5"] <= 1.0
    assert 0.0 <= out["btts"] <= 1.0
    assert out["lambda_home"] > 0 and out["lambda_away"] > 0


def test_stronger_home_market_gives_higher_home_lambda():
    weak = market_scoreline(0.30, 0.30, 0.40, RHO)
    strong = market_scoreline(0.70, 0.20, 0.10, RHO)
    assert strong["lambda_home"] > weak["lambda_home"]
    assert strong["probs"][0] > weak["probs"][0]
