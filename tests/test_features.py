"""Unit tests for build_features (SPEC §4, §5). Hand-entered synthetic data."""

import math

import numpy as np

from wc_model.features import build_features


# --- tiny builders ---------------------------------------------------------

def team(tid):
    return {"team_id": tid}


def match(match_id, dte, home, away, hg, ag):
    return {
        "match_id": match_id,
        "date": dte,
        "home_team_id": home,
        "away_team_id": away,
        "venue_country": "NEU",
        "neutral": True,
        "competition": "Friendly",
        "home_goals": hg,
        "away_goals": ag,
    }


def fair_odds(p_home, p_draw, p_away):
    return 1.0 / p_home, 1.0 / p_draw, 1.0 / p_away


def odds(match_id, p_home, p_draw, p_away, book="b1"):
    oh, od, oa = fair_odds(p_home, p_draw, p_away)
    return {
        "match_id": match_id,
        "bookmaker": book,
        "odds_home": oh,
        "odds_draw": od,
        "odds_away": oa,
        "captured_at": "2000-01-01",
    }


def xg(match_id, tid, xg_for, xg_against):
    return {"match_id": match_id, "team_id": tid, "xg_for": xg_for, "xg_against": xg_against}


def fifa(tid, dte, points, rank=1):
    return {"team_id": tid, "as_of_date": dte, "fifa_points": points, "fifa_rank": rank}


def _rec(records, tid):
    return next(r for r in records if r["team_id"] == tid)


# --- §5 surprisal ----------------------------------------------------------

def test_surprisal_large_for_underdog_win_small_for_favorite_win():
    teams = [team("UD"), team("FAV")]
    results = [
        match("m1", "2020-01-01", "UD", "o1", 1, 0),   # underdog UD wins
        match("m2", "2020-01-01", "FAV", "o2", 1, 0),  # favorite FAV wins
    ]
    odds_rows = [
        odds("m1", 0.05, 0.15, 0.80),  # UD home -> p_win = 0.05
        odds("m2", 0.90, 0.05, 0.05),  # FAV home -> p_win = 0.90
    ]
    recs = build_features("2020-06-01", teams, results, [], [], odds_rows, xi=0.0)

    # xi=0 -> single-match U equals the surprisal S = -ln(p_realized).
    assert math.isclose(_rec(recs, "UD")["upset_propensity"], -math.log(0.05), rel_tol=1e-9)
    assert math.isclose(_rec(recs, "FAV")["upset_propensity"], -math.log(0.90), rel_tol=1e-9)
    assert _rec(recs, "UD")["upset_propensity"] > _rec(recs, "FAV")["upset_propensity"]


# --- §5 M sign -------------------------------------------------------------

def test_market_adj_perf_sign_beats_loses_and_chalk():
    teams = [team("BEAT"), team("LOSE"), team("CHALK")]
    results = [
        match("b1", "2020-01-01", "BEAT", "o", 1, 0),   # underdog wins -> M>0
        match("l1", "2020-01-01", "LOSE", "o", 0, 1),   # favorite loses -> M<0
        # CHALK: 2 wins, 1 draw, 1 loss at p=(0.5,0.25,0.25); mean AP == EP=1.75.
        match("c1", "2020-01-01", "CHALK", "o", 1, 0),
        match("c2", "2020-01-02", "CHALK", "o", 1, 0),
        match("c3", "2020-01-03", "CHALK", "o", 1, 1),
        match("c4", "2020-01-04", "CHALK", "o", 0, 1),
    ]
    odds_rows = [
        odds("b1", 0.05, 0.15, 0.80),
        odds("l1", 0.90, 0.05, 0.05),
        odds("c1", 0.50, 0.25, 0.25),
        odds("c2", 0.50, 0.25, 0.25),
        odds("c3", 0.50, 0.25, 0.25),
        odds("c4", 0.50, 0.25, 0.25),
    ]
    recs = build_features("2020-06-01", teams, results, [], [], odds_rows, xi=0.0)
    assert _rec(recs, "BEAT")["market_adj_perf"] > 0
    assert _rec(recs, "LOSE")["market_adj_perf"] < 0
    assert abs(_rec(recs, "CHALK")["market_adj_perf"]) < 1e-9


# --- §5 U is direction-agnostic --------------------------------------------

def test_upset_propensity_is_direction_agnostic():
    teams = [team("A"), team("B")]
    results = [
        match("a1", "2020-01-01", "A", "o", 1, 0),  # upset win (underdog wins)
        match("a2", "2020-01-01", "A", "o", 0, 1),  # upset loss (favorite loses)
        match("b1", "2020-01-01", "B", "o", 1, 0),  # on-form win (favorite wins)
        match("b2", "2020-01-01", "B", "o", 0, 1),  # on-form loss (underdog loses)
    ]
    odds_rows = [
        odds("a1", 0.05, 0.05, 0.90),  # A underdog, wins
        odds("a2", 0.90, 0.05, 0.05),  # A favorite, loses
        odds("b1", 0.90, 0.05, 0.05),  # B favorite, wins
        odds("b2", 0.05, 0.05, 0.90),  # B underdog, loses
    ]
    recs = build_features("2020-06-01", teams, results, [], [], odds_rows, xi=0.0)
    assert _rec(recs, "A")["upset_propensity"] > _rec(recs, "B")["upset_propensity"]


# --- §4/§5 time decay ------------------------------------------------------

def test_more_recent_surprise_produces_larger_feature():
    # Identical histories (one surprise + one calm match) with dates swapped.
    teams = [team("R"), team("O")]
    recent, old = "2020-12-22", "2020-03-07"
    results = [
        match("r_surp", recent, "R", "o", 1, 0),  # R: surprise is recent
        match("r_calm", old, "R", "o", 1, 0),
        match("o_surp", old, "O", "o", 1, 0),     # O: surprise is old
        match("o_calm", recent, "O", "o", 1, 0),
    ]
    odds_rows = [
        odds("r_surp", 0.05, 0.15, 0.80), odds("r_calm", 0.90, 0.05, 0.05),
        odds("o_surp", 0.05, 0.15, 0.80), odds("o_calm", 0.90, 0.05, 0.05),
    ]
    recs = build_features("2021-01-01", teams, results, [], [], odds_rows, xi=0.01)
    assert _rec(recs, "R")["upset_propensity"] > _rec(recs, "O")["upset_propensity"]


# --- §4 z_fifa -------------------------------------------------------------

def test_z_fifa_standardized_and_monotonic():
    teams = [team("L"), team("M"), team("H")]
    ratings = [
        fifa("L", "2020-01-01", 1000.0),
        fifa("M", "2020-01-01", 1500.0),
        fifa("H", "2020-01-01", 2000.0),
    ]
    recs = build_features("2020-06-01", teams, [], ratings, [], [], xi=0.0)
    zs = np.array([_rec(recs, t)["z_fifa"] for t in ("L", "M", "H")])
    assert abs(zs.mean()) < 1e-9
    assert abs(zs.std() - 1.0) < 1e-9
    assert zs[0] < zs[1] < zs[2]  # more points -> higher z


def test_point_in_time_fifa_uses_latest_snapshot_before_cutoff():
    teams = [team("X"), team("Y")]
    ratings = [
        fifa("X", "2019-01-01", 1000.0),
        fifa("X", "2020-03-01", 1900.0),  # latest valid snapshot for X
        fifa("X", "2025-01-01", 50.0),    # post-cutoff: must be ignored
        fifa("Y", "2020-02-01", 1100.0),
    ]
    recs = build_features("2020-06-01", teams, [], ratings, [], [], xi=0.0)
    # X (1900) > Y (1100) -> X has the higher z; the post-cutoff 50.0 was ignored.
    assert _rec(recs, "X")["z_fifa"] > _rec(recs, "Y")["z_fifa"]


# --- §4 attack / defense indices -------------------------------------------

def test_indices_track_xg_with_defense_sign_inversion():
    teams = [team("T1"), team("T2"), team("T3")]
    results = [
        match("m1", "2020-01-01", "T1", "o1", 0, 0),
        match("m2", "2020-01-01", "T2", "o2", 1, 1),
        match("m3", "2020-01-01", "T3", "o3", 3, 3),
    ]
    team_xg = [
        xg("m1", "T1", 0.5, 0.5),
        xg("m2", "T2", 1.5, 1.5),
        xg("m3", "T3", 3.0, 3.0),
    ]
    recs = build_features("2020-06-01", teams, results, [], team_xg, [], xi=0.0)
    a = {t: _rec(recs, t)["attack_index"] for t in ("T1", "T2", "T3")}
    d = {t: _rec(recs, t)["defense_index"] for t in ("T1", "T2", "T3")}
    # Higher xG-for -> higher attack_index.
    assert a["T1"] < a["T2"] < a["T3"]
    # Higher xG-against -> lower defense_index (sign inversion holds).
    assert d["T1"] > d["T2"] > d["T3"]


def test_missing_xg_falls_back_to_goals():
    # No xG provided: attack/defense fall back to goals; ordering still holds.
    teams = [team("A"), team("B")]
    results = [
        match("m1", "2020-01-01", "A", "o", 0, 3),  # A: weak attack, leaky defense
        match("m2", "2020-01-01", "B", "o", 3, 0),  # B: strong attack, tight defense
    ]
    recs = build_features("2020-06-01", teams, results, [], [], [], xi=0.0)
    assert _rec(recs, "B")["attack_index"] > _rec(recs, "A")["attack_index"]
    assert _rec(recs, "B")["defense_index"] > _rec(recs, "A")["defense_index"]


# --- look-ahead guard ------------------------------------------------------

def test_lookahead_guard_post_cutoff_data_is_ignored():
    teams = [team("A"), team("B")]
    base_results = [
        match("m1", "2020-01-01", "A", "B", 1, 0),
        match("m2", "2020-02-01", "B", "A", 2, 1),
    ]
    base_xg = [xg("m1", "A", 1.2, 0.4), xg("m2", "B", 1.8, 1.0)]
    base_odds = [odds("m1", 0.5, 0.3, 0.2), odds("m2", 0.45, 0.30, 0.25)]
    base_fifa = [fifa("A", "2019-12-01", 1500.0), fifa("B", "2019-12-01", 1400.0)]

    as_of = "2020-06-01"
    base = build_features(as_of, teams, base_results, base_fifa, base_xg, base_odds, xi=0.01)

    # A post-cutoff match (with xG, odds, and a fifa snapshot) that WOULD change
    # the features if used. It must be silently ignored.
    poison_results = base_results + [match("future", "2020-09-01", "A", "B", 9, 0)]
    poison_xg = base_xg + [xg("future", "A", 9.0, 0.0)]
    poison_odds = base_odds + [odds("future", 0.01, 0.04, 0.95)]  # huge upset if used
    poison_fifa = base_fifa + [fifa("A", "2020-08-01", 9999.0)]
    poisoned = build_features(
        as_of, teams, poison_results, poison_fifa, poison_xg, poison_odds, xi=0.01
    )

    assert poisoned == base


# --- numerics & sparse history ---------------------------------------------

def test_p_realized_near_zero_does_not_blow_up():
    teams = [team("Z")]
    results = [match("m1", "2020-01-01", "Z", "o", 0, 1)]  # Z loses
    # Z is an overwhelming favorite; the realized (loss) prob is ~1e-13 (< eps).
    odds_rows = [odds("m1", 0.999999998, 1e-9, 1e-9)]
    recs = build_features("2020-06-01", teams, results, [], [], odds_rows, xi=0.0)
    u = _rec(recs, "Z")["upset_propensity"]
    assert math.isfinite(u)
    assert u > 0


def test_team_with_fewer_than_n_recent_matches():
    teams = [team("A")]
    results = [
        match("m1", "2020-01-01", "A", "o", 1, 0),
        match("m2", "2020-02-01", "A", "o", 0, 1),
    ]
    odds_rows = [odds("m1", 0.4, 0.3, 0.3), odds("m2", 0.6, 0.2, 0.2)]
    recs = build_features("2020-06-01", teams, results, [], [], odds_rows, xi=0.05, n_recent=10)
    rec = _rec(recs, "A")
    assert math.isfinite(rec["upset_propensity"])
    assert math.isfinite(rec["market_adj_perf"])
