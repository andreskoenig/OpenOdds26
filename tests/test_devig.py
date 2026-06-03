"""Unit tests for de-vigging (SPEC §5 step 1). Hand-entered synthetic odds."""

import math

import pytest

from wc_model.devig import consensus_probs, devig


def _fair_odds(p_home, p_draw, p_away):
    """Decimal odds for a fair book (raw implied probs already sum to 1)."""
    return (1.0 / p_home, 1.0 / p_draw, 1.0 / p_away)


def test_fair_book_returned_unchanged():
    p = (0.50, 0.30, 0.20)
    out = devig(*_fair_odds(*p))
    assert math.isclose(sum(out), 1.0, abs_tol=1e-12)
    for got, want in zip(out, p):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)


def test_overround_book_sums_to_one_and_preserves_ratios():
    # Raw implied probs 0.6/0.3/0.2 sum to 1.1 (10% overround).
    odds = (1 / 0.6, 1 / 0.3, 1 / 0.2)
    p_home, p_draw, p_away = devig(*odds)
    assert math.isclose(p_home + p_draw + p_away, 1.0, abs_tol=1e-12)
    # Proportional normalization preserves the ratios of the raw implieds.
    assert math.isclose(p_home / p_draw, 0.6 / 0.3, rel_tol=1e-12)
    assert math.isclose(p_draw / p_away, 0.3 / 0.2, rel_tol=1e-12)


def test_consensus_averages_three_books():
    books = [
        _fair_odds(0.50, 0.30, 0.20),
        _fair_odds(0.40, 0.35, 0.25),
        _fair_odds(0.60, 0.25, 0.15),
    ]
    out = consensus_probs(books)
    expected = (
        (0.50 + 0.40 + 0.60) / 3,
        (0.30 + 0.35 + 0.25) / 3,
        (0.20 + 0.25 + 0.15) / 3,
    )
    assert math.isclose(sum(out), 1.0, abs_tol=1e-12)
    for got, want in zip(out, expected):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)


def test_lower_odds_give_higher_probability():
    # Home is the shortest price -> must be the most likely outcome.
    p_home, p_draw, p_away = devig(1.5, 4.0, 7.0)
    assert p_home > p_draw > p_away


def test_unknown_method_raises():
    with pytest.raises(NotImplementedError):
        devig(2.0, 3.0, 4.0, method="shin")
