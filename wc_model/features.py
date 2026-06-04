"""Feature construction (SPEC §4 ratings priors and §5 surprise factor).

Pure deterministic computation. Builds the FIFA strength prior, attack/defense
indices, and the market-derived surprise features (upset propensity U_T and
market-adjusted performance M_T). Every feature for the build uses only data
dated strictly before the as-of date (no look-ahead).

This is a pure data transform: it emits RAW features only. It does NOT combine
them into atk/def priors — the prior coefficients (c_a, c_x, ...) and the
dispersion/mean mappings (kappa, theta) belong to fit_model and the predict
step.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np

from .devig import consensus_probs
from .schemas import (
    FeatureRecord,
    FifaRating,
    MatchOdds,
    MatchResult,
    TeamXg,
)

_EPS = 1e-12


def _as_date(s: str) -> date:
    return date.fromisoformat(s)


def _team_id(team) -> str:
    return team["team_id"] if isinstance(team, dict) else team


def time_weight(days_to_as_of: float, xi: float) -> float:
    """Exponential time-decay weight ``w(t) = exp(-xi * t)`` (SPEC §4)."""
    return math.exp(-xi * days_to_as_of)


def _standardize_map(values: Dict[str, float], team_ids: Sequence[str]) -> Dict[str, float]:
    """Standardize ``values`` across the field; absent teams default to 0.0.

    Mean / sd (population, ddof=0) are computed over the teams that have a value;
    a team without a value (or when sd == 0) gets 0.0 (the field-neutral z).
    """
    present = np.array([values[t] for t in team_ids if t in values], dtype=float)
    if present.size == 0:
        return {t: 0.0 for t in team_ids}
    mean = float(present.mean())
    sd = float(present.std())
    out: Dict[str, float] = {}
    for t in team_ids:
        if t in values and sd > 0:
            out[t] = (values[t] - mean) / sd
        else:
            out[t] = 0.0
    return out


def build_features(
    as_of_date: str,
    teams: Sequence,
    match_results: Sequence[MatchResult],
    fifa_ratings: Sequence[FifaRating],
    team_xg: Sequence[TeamXg],
    match_odds: Sequence[MatchOdds],
    *,
    xi: float,
    blend_weight: float = 0.7,
    n_recent: int = 10,
    squad_values: Optional[Sequence[dict]] = None,
    market_probs: Optional[dict] = None,
) -> List[FeatureRecord]:
    """Build one raw feature record per team (SPEC §4, §5).

    As-of cutoff (hard rule, enforced here regardless of caller input): only
    data strictly dated before ``as_of_date`` is used. ``match_results``,
    ``team_xg``, and ``match_odds`` are filtered by match date; ``fifa_ratings``
    uses the most recent snapshot with ``as_of_date <= as_of_date`` (point-in-
    time).

    Returns a ``FeatureRecord`` per team (in ``teams`` order) with: ``z_fifa``,
    ``attack_index``, ``defense_index``, ``upset_propensity`` (U), and
    ``market_adj_perf`` (M).
    """
    as_of = _as_date(as_of_date)
    team_ids = [_team_id(t) for t in teams]
    team_set = set(team_ids)

    match_by_id = {m["match_id"]: m for m in match_results}

    def match_date(match_id: str) -> Optional[date]:
        m = match_by_id.get(match_id)
        return _as_date(m["date"]) if m is not None else None

    # --- As-of cutoff filtering (strictly before as_of) --------------------
    pre_matches = [m for m in match_results if _as_date(m["date"]) < as_of]

    xg_by_key: Dict[tuple, TeamXg] = {}
    for r in team_xg:
        md = match_date(r["match_id"])
        if md is not None and md < as_of:
            xg_by_key[(r["match_id"], r["team_id"])] = r

    odds_by_match: Dict[str, List[tuple]] = {}
    for o in match_odds:
        md = match_date(o["match_id"])
        if md is not None and md < as_of:
            odds_by_match.setdefault(o["match_id"], []).append(
                (o["odds_home"], o["odds_draw"], o["odds_away"])
            )

    # --- §4: FIFA strength prior (point-in-time snapshot) ------------------
    latest_fifa: Dict[str, tuple] = {}  # team_id -> (snapshot_date, fifa_points)
    for r in fifa_ratings:
        snap = _as_date(r["as_of_date"])
        if snap <= as_of:
            cur = latest_fifa.get(r["team_id"])
            if cur is None or snap > cur[0]:
                latest_fifa[r["team_id"]] = (snap, r["fifa_points"])
    fifa_points = {t: latest_fifa[t][1] for t in team_ids if t in latest_fifa}
    z_fifa = _standardize_map(fifa_points, team_ids)

    # --- §4 (extension): squad-value (talent-pool) prior -------------------
    # Point-in-time like fifa_ratings: latest snapshot with as_of_date <= cutoff.
    # Standardize log(total_value_eur) across the field; impute z=0 (neutral) when
    # n_players < 10 so thin-coverage minnows are not distorted.
    latest_sv: Dict[str, tuple] = {}  # team_id -> (snap_date, total_value, n_players)
    for r in (squad_values or []):
        snap = _as_date(r["as_of_date"])
        if snap <= as_of:
            cur = latest_sv.get(r["team_id"])
            if cur is None or snap > cur[0]:
                latest_sv[r["team_id"]] = (snap, r["total_value_eur"], r["n_players"])
    sv_log = {
        t: math.log(latest_sv[t][1])
        for t in team_ids
        if t in latest_sv and latest_sv[t][2] >= 10 and latest_sv[t][1] > 0
    }
    z_squad_value = _standardize_map(sv_log, team_ids)

    # --- Market (Polymarket winner) prior ---------------------------------
    # Standardize log(market P(win)) across priced teams; unpriced -> z=0.
    # (Outcome-level market info folded into the team strength prior.)
    mkt_log = {
        t: math.log(market_probs[t])
        for t in team_ids
        if market_probs and t in market_probs and market_probs[t] > 0
    }
    z_market = _standardize_map(mkt_log, team_ids)

    # --- §4: attack / defense indices --------------------------------------
    gf: Dict[str, list] = {t: [] for t in team_ids}
    ga: Dict[str, list] = {t: [] for t in team_ids}
    xgf: Dict[str, list] = {t: [] for t in team_ids}
    xga: Dict[str, list] = {t: [] for t in team_ids}

    for m in pre_matches:
        mid = m["match_id"]
        for side in ("home", "away"):
            tid = m["home_team_id"] if side == "home" else m["away_team_id"]
            if tid not in team_set:
                continue
            goals_for = m["home_goals"] if side == "home" else m["away_goals"]
            goals_against = m["away_goals"] if side == "home" else m["home_goals"]
            gf[tid].append(goals_for)
            ga[tid].append(goals_against)
            xr = xg_by_key.get((mid, tid))
            # Fall back to actual goals when xG is missing for this match.
            xgf[tid].append(xr["xg_for"] if xr is not None else goals_for)
            xga[tid].append(xr["xg_against"] if xr is not None else goals_against)

    a_raw: Dict[str, float] = {}
    d_raw: Dict[str, float] = {}
    for t in team_ids:
        if gf[t]:
            a_raw[t] = blend_weight * float(np.mean(xgf[t])) + (1 - blend_weight) * float(np.mean(gf[t]))
            d_raw[t] = blend_weight * float(np.mean(xga[t])) + (1 - blend_weight) * float(np.mean(ga[t]))

    attack_index = _standardize_map(a_raw, team_ids)
    defense_z = _standardize_map(d_raw, team_ids)
    # Sign-invert: conceding fewer (lower d_raw) -> higher defense_index.
    defense_index = {t: -defense_z[t] for t in team_ids}

    # --- §5: surprise features (U, M) over recent matches with odds --------
    team_rows: Dict[str, list] = {t: [] for t in team_ids}
    for m in pre_matches:
        mid = m["match_id"]
        if mid not in odds_by_match:
            continue  # skip matches with no odds
        p_home, p_draw, p_away = consensus_probs(odds_by_match[mid])
        for side in ("home", "away"):
            tid = m["home_team_id"] if side == "home" else m["away_team_id"]
            if tid not in team_set:
                continue
            if side == "home":
                p_win, p_loss = p_home, p_away
                team_goals, opp_goals = m["home_goals"], m["away_goals"]
            else:
                p_win, p_loss = p_away, p_home
                team_goals, opp_goals = m["away_goals"], m["home_goals"]
            if team_goals > opp_goals:
                actual_points, p_realized = 3, p_win
            elif team_goals == opp_goals:
                actual_points, p_realized = 1, p_draw
            else:
                actual_points, p_realized = 0, p_loss
            expected_points = 3 * p_win + 1 * p_draw
            team_rows[tid].append((_as_date(m["date"]), actual_points, expected_points, p_realized))

    upset_propensity: Dict[str, float] = {}
    market_adj_perf: Dict[str, float] = {}
    for t in team_ids:
        rows = sorted(team_rows[t], key=lambda r: r[0], reverse=True)[:n_recent]
        sum_w = sum_ws = sum_wm = 0.0
        for match_dt, actual_points, expected_points, p_realized in rows:
            days = (as_of - match_dt).days
            w = time_weight(days, xi)
            surprisal = -math.log(min(max(p_realized, _EPS), 1.0))
            sum_w += w
            sum_ws += w * surprisal
            sum_wm += w * (actual_points - expected_points)
        upset_propensity[t] = sum_ws / sum_w if sum_w > 0 else 0.0
        market_adj_perf[t] = sum_wm / sum_w if sum_w > 0 else 0.0

    return [
        FeatureRecord(
            team_id=t,
            z_fifa=z_fifa[t],
            attack_index=attack_index[t],
            defense_index=defense_index[t],
            upset_propensity=upset_propensity[t],
            market_adj_perf=market_adj_perf[t],
            z_squad_value=z_squad_value[t],
            z_market=z_market[t],
        )
        for t in team_ids
    ]
