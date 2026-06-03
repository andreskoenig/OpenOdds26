"""Validity tests for the 48-team 2026 explicit knockout bracket (Phase 3)."""

import json
from collections import defaultdict
from pathlib import Path

from wc_model.model import score_matrix
from wc_model.simulate import simulate_tournament

CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "tournament_config_2026.json")
    .read_text(encoding="utf-8")
)
TEAM_IDS = [t for g in CONFIG["groups"].values() for t in g]


def _world():
    """Synthetic distinct ratings + a stochastic matrix_fn over the 48 teams."""
    ratings = {t: {"atk": 0.12 * ((i % 7) - 3), "def": 0.1 * ((i % 5) - 2)}
               for i, t in enumerate(TEAM_IDS)}
    hosts = set(CONFIG["host_team_ids"])
    teams = {t: {"rating": ratings[t]["atk"] - ratings[t]["def"], "host": t in hosts}
             for t in TEAM_IDS}

    def fn(home, away, flag):
        return score_matrix(ratings[home]["atk"], ratings[home]["def"],
                            ratings[away]["atk"], ratings[away]["def"],
                            mu=0.1, gamma=0.25, rho=-0.05, home=flag)

    return teams, fn


def test_single_run_bracket_is_structurally_valid():
    teams, fn = _world()
    prog, extras = simulate_tournament(teams, fn, CONFIG, n_runs=1, seed=7,
                                       collect_extras=True)
    ko = extras["sample"]["knockout"]
    by_round = defaultdict(list)
    for m in ko:
        if m["label"]:
            by_round[m["label"]].extend([m["home"], m["away"]])

    # exactly 32 distinct in R32, halving to one champion; no team twice per round
    expected = {"R32": 32, "R16": 16, "QF": 8, "SF": 4, "final": 2}
    for label, n in expected.items():
        teams_in = by_round[label]
        assert len(teams_in) == n, f"{label} had {len(teams_in)} slots"
        assert len(set(teams_in)) == n, f"{label} has a duplicate team"
        assert set(teams_in) <= set(TEAM_IDS)

    champ = extras["sample"]["champion"]
    assert champ in set(by_round["final"])
    # winners propagate: each round's teams are a subset of the previous round's
    for a, b in [("R16", "R32"), ("QF", "R16"), ("SF", "QF"), ("final", "SF")]:
        assert set(by_round[a]) <= set(by_round[b])


def test_aggregate_one_champion_and_monotonic_reach():
    teams, fn = _world()
    prog = simulate_tournament(teams, fn, CONFIG, n_runs=200, seed=3)
    assert set(prog) == set(TEAM_IDS)
    assert abs(sum(prog[t]["winner"] for t in TEAM_IDS) - 1.0) < 1e-9
    # exactly 32 teams reach R32 each run on average
    assert abs(sum(prog[t]["R32"] for t in TEAM_IDS) - 32.0) < 1e-9
    for t in TEAM_IDS:
        r32, r16, qf, sf, fin, win = (prog[t]["R32"], prog[t]["R16"], prog[t]["QF"],
                                      prog[t]["SF"], prog[t]["final"], prog[t]["winner"])
        assert 0.0 <= win <= fin <= sf <= qf <= r16 <= r32 <= 1.0


def test_reproducible_under_fixed_seed():
    teams, fn = _world()
    a = simulate_tournament(teams, fn, CONFIG, n_runs=60, seed=11)
    b = simulate_tournament(teams, fn, CONFIG, n_runs=60, seed=11)
    assert a == b


def test_best_thirds_eight_distinct_groups_in_r32():
    # The 8 third-place teams in R32 come from 8 distinct groups.
    teams, fn = _world()
    _, extras = simulate_tournament(teams, fn, CONFIG, n_runs=1, seed=5, collect_extras=True)
    # group winners/runners-up of each group from the sampled standings
    standings = extras["sample"]["groups"]
    top2 = {row["team"] for g in standings.values() for row in g[:2]}
    r32 = set()
    for m in extras["sample"]["knockout"]:
        if m["label"] == "R32":
            r32.update([m["home"], m["away"]])
    thirds = r32 - top2
    assert len(thirds) == 8
    # each third is the 3rd-placed team of its group -> 8 distinct groups
    third_groups = [g for g, rows in standings.items() if rows[2]["team"] in thirds]
    assert len(third_groups) == 8
