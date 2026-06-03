"""Unit tests for the Monte Carlo tournament simulation (SPEC §6).

Hand-entered synthetic field only (no real teams). Uses a small, reduced
bracket so runs are fast.
"""

import numpy as np

from wc_model.model import score_matrix
from wc_model.simulate import simulate_tournament

# 4 groups of 4 = 16 teams; top 2 advance, no best-thirds -> 8 qualifiers
# (a power of two): knockout QF -> SF -> final.
TEAM_IDS = [f"t{i:02d}" for i in range(16)]
GROUPS = {
    "A": TEAM_IDS[0:4],
    "B": TEAM_IDS[4:8],
    "C": TEAM_IDS[8:12],
    "D": TEAM_IDS[12:16],
}
CONFIG = {"groups": GROUPS, "advance_per_group": 2, "best_thirds": 0}


def _flat_teams(rating=0.0):
    return {t: {"rating": rating, "host": False} for t in TEAM_IDS}


def _degenerate_matrix_fn(super_team, max_goals=10):
    """Matrix function where ``super_team`` beats everyone 3-0; others draw 1-1."""

    def fn(home, away, _flag):
        p = np.zeros((max_goals + 1, max_goals + 1))
        if super_team == home:
            p[3, 0] = 1.0
        elif super_team == away:
            p[0, 3] = 1.0
        else:
            p[1, 1] = 1.0
        return p

    return fn


def _synthetic_matrix_fn():
    """A genuinely stochastic matrix function over distinct synthetic ratings."""
    ratings = {
        t: {"atk": 0.15 * (i % 5) - 0.3, "def": 0.1 * (i % 3) - 0.1}
        for i, t in enumerate(TEAM_IDS)
    }

    def fn(home, away, flag):
        return score_matrix(
            atk_i=ratings[home]["atk"],
            def_i=ratings[home]["def"],
            atk_j=ratings[away]["atk"],
            def_j=ratings[away]["def"],
            mu=0.0,
            gamma=0.3,
            rho=0.05,
            home=flag,
        )

    return fn, {t: {"rating": r["atk"] - r["def"], "host": False} for t, r in ratings.items()}


def test_dominant_team_wins_with_probability_one():
    super_team = TEAM_IDS[0]
    result = simulate_tournament(
        _flat_teams(),
        _degenerate_matrix_fn(super_team),
        CONFIG,
        n_runs=100,
        seed=1,
    )
    # The dominant team never draws, so it wins every match in every run.
    assert result[super_team]["winner"] == 1.0
    for t in TEAM_IDS:
        if t != super_team:
            assert result[t]["winner"] == 0.0


def test_same_seed_and_inputs_give_identical_results():
    matrix_fn, teams = _synthetic_matrix_fn()
    a = simulate_tournament(teams, matrix_fn, CONFIG, n_runs=200, seed=7)
    b = simulate_tournament(teams, matrix_fn, CONFIG, n_runs=200, seed=7)
    assert a == b


def test_win_probabilities_sum_to_one():
    matrix_fn, teams = _synthetic_matrix_fn()
    result = simulate_tournament(teams, matrix_fn, CONFIG, n_runs=300, seed=3)
    total = sum(result[t]["winner"] for t in TEAM_IDS)
    assert abs(total - 1.0) < 1e-9


def test_reach_round_probabilities_are_monotonic_and_bounded():
    matrix_fn, teams = _synthetic_matrix_fn()
    result = simulate_tournament(teams, matrix_fn, CONFIG, n_runs=300, seed=5)
    # Qualifier count is 8 -> rounds QF, SF, final, plus "winner".
    for t in TEAM_IDS:
        qf = result[t]["QF"]
        sf = result[t]["SF"]
        final = result[t]["final"]
        win = result[t]["winner"]
        assert 0.0 <= win <= final <= sf <= qf <= 1.0


# --- knockout bracket crossing (config-driven seeding) ---------------------

import json
from pathlib import Path

_CONFIG_2022 = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "tournament_config_2022.json")
    .read_text(encoding="utf-8")
)


def _deterministic_world(groups):
    """Strengths strictly decreasing in listed order; the stronger team always wins.

    With distinct strengths and no draws, each group's ranking equals its listed
    order, so position ``p`` of group ``g`` is ``groups[g][p-1]`` (1st, 2nd, ...).
    """
    all_teams = [t for ts in groups.values() for t in ts]
    strength = {t: len(all_teams) - i for i, t in enumerate(all_teams)}
    teams = {t: {"rating": float(strength[t]), "host": False} for t in all_teams}
    return strength, teams


def _spy_matrix_fn(strength, max_goals=10):
    """Deterministic matrix_fn that records every (home, away) it is asked about."""
    calls = []

    def fn(home, away, _flag):
        calls.append((home, away))
        p = np.zeros((max_goals + 1, max_goals + 1))
        if strength[home] > strength[away]:
            p[1, 0] = 1.0  # home wins 1-0
        else:
            p[0, 1] = 1.0  # away wins 0-1
        return p

    return fn, calls


def _first_round_pairs(groups, config, n_first_round):
    """Run one deterministic tournament; return the first knockout round's
    (home, away) pairings, captured as the first cross-group matrix_fn calls."""
    strength, teams = _deterministic_world(groups)
    fn, calls = _spy_matrix_fn(strength)
    simulate_tournament(teams, fn, config, n_runs=1, seed=0)
    group_of = {t: g for g, ts in groups.items() for t in ts}
    cross = [(h, a) for (h, a) in calls if group_of[h] != group_of[a]]
    return cross[:n_first_round]


def test_knockout_follows_config_crossing_2022():
    groups = _CONFIG_2022["groups"]
    advance = _CONFIG_2022["advance_per_group"]

    def slot(s):  # '1A' -> winner of A, '2B' -> runner-up of B, ...
        return groups[s[1:]][int(s[0]) - 1]

    # 16 qualifiers -> first round (R16) has 8 matches; standard 2022 crossing.
    r16 = _first_round_pairs(groups, _CONFIG_2022, 8)
    crossing = [("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
                ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G")]
    expected = [(slot(h), slot(a)) for h, a in crossing]

    assert advance == 2
    assert r16 == expected
    # Spot-check the (group, position) intent: 1A meets 2B, 1C meets 2D.
    assert {r16[0][0], r16[0][1]} == {slot("1A"), slot("2B")}
    assert {r16[1][0], r16[1][1]} == {slot("1C"), slot("2D")}


def test_no_knockout_bracket_seeds_sequentially():
    groups = {
        "A": [f"a{i}" for i in range(4)],
        "B": [f"b{i}" for i in range(4)],
        "C": [f"c{i}" for i in range(4)],
        "D": [f"d{i}" for i in range(4)],
    }
    config = {"groups": groups, "advance_per_group": 2, "best_thirds": 0}

    # 8 qualifiers, no bracket -> sequential seeding [1A,1B,1C,1D,2A,2B,2C,2D],
    # so the first round pairs winners with winners, runners-up with runners-up.
    pairs = _first_round_pairs(groups, config, 4)
    expected = [
        (groups["A"][0], groups["B"][0]),
        (groups["C"][0], groups["D"][0]),
        (groups["A"][1], groups["B"][1]),
        (groups["C"][1], groups["D"][1]),
    ]
    assert pairs == expected
    # Sequential (not a crossing): 1A's opponent is 1B (a fellow winner), not 2B.
    assert pairs[0] == (groups["A"][0], groups["B"][0])
