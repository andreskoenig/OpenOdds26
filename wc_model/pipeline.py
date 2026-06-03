"""End-to-end backtest harness (SPEC §9 build order, §10 validation gates).

Pure deterministic Python — no agents, no network, no fetching. Chains the
deterministic pieces: ``build_features`` -> ``fit_model`` -> a ``matchup_matrix``
closure -> ``simulate_tournament``, and scores each real pre-match prediction
against reality with ``evaluate``.

The as-of cutoff (no look-ahead, SPEC §7/§10) is enforced by the underlying
``build_features`` / ``fit_model`` filtering; this module re-uses the raw input
tables as given and does not re-leak post-cutoff rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .devig import consensus_probs
from .evaluate import evaluate
from .features import build_features
from .model import fit_model, matchup_matrix, result_probs
from .schemas import (
    FifaRating,
    Hyperparams,
    MatchOdds,
    MatchResult,
    ModelParams,
)
from .simulate import simulate_tournament


def _team_id(team) -> str:
    return team["team_id"] if isinstance(team, dict) else team


def _outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if home_goals < away_goals:
        return "away"
    return "draw"


@dataclass
class PredictionResult:
    """Output of ``run_prediction`` (SPEC §6 tournament view + match predictor)."""

    params: ModelParams
    p_win: Dict[str, float]
    progression: Dict[str, Dict[str, float]]
    kappa: float = 0.0
    r0: float = 8.0

    def predict_match(
        self, home: str, away: str, home_flag: bool
    ) -> Dict[str, object]:
        """Pre-match scoreline matrix + 1X2 for any pairing from fitted params.

        Returns ``{"matrix": P, "probs": (p_home, p_draw, p_away), "home":,
        "draw":, "away":}`` using the same §5 dispersion (``kappa``) as the sim.
        """
        matrix = matchup_matrix(
            self.params, home, away, home_flag, kappa=self.kappa, r0=self.r0
        )
        p_home, p_draw, p_away = result_probs(matrix)
        return {
            "matrix": matrix,
            "probs": (p_home, p_draw, p_away),
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        }


@dataclass
class BacktestReport:
    """Output of ``backtest`` (SPEC §10): match-level metrics + tournament view."""

    match_metrics: Dict[str, object]
    p_win: Dict[str, float]
    champion: Optional[str]
    champion_p_win: Optional[float]
    prediction: PredictionResult


def run_prediction(
    as_of_date: str,
    teams: Sequence,
    tournament_config: dict,
    match_results: Sequence[MatchResult],
    fifa_ratings: Sequence[FifaRating],
    team_xg: Sequence,
    match_odds: Sequence[MatchOdds],
    *,
    hyperparams: Hyperparams,
    squad_values: Optional[Sequence[dict]] = None,
    n_sims: int = 10000,
    seed: int = 0,
) -> PredictionResult:
    """Features -> fit -> simulate, threading one ``Hyperparams`` (SPEC §4-§6).

    Every input table is used only for rows strictly before ``as_of_date`` (the
    cutoff is enforced inside ``build_features`` / ``fit_model``). The fitted
    params drive both the tournament simulation and ``predict_match``.
    """
    hp = hyperparams

    features = build_features(
        as_of_date,
        teams,
        match_results,
        fifa_ratings,
        team_xg,
        match_odds,
        squad_values=squad_values,
        **hp.feature_kwargs(),
    )
    params = fit_model(as_of_date, teams, match_results, features, **hp.fit_kwargs())

    def matrix_fn(home: str, away: str, home_flag: bool) -> np.ndarray:
        return matchup_matrix(params, home, away, home_flag, kappa=hp.kappa)

    team_ids = [_team_id(t) for t in teams]
    host_ids = set(
        tournament_config.get("host_team_ids", tournament_config.get("host_cities", []))
    )
    sim_teams = {
        tid: {
            "rating": params.atk.get(tid, 0.0) - params.def_.get(tid, 0.0),
            "host": tid in host_ids,
        }
        for tid in team_ids
    }

    progression = simulate_tournament(
        sim_teams, matrix_fn, tournament_config, n_runs=n_sims, seed=seed
    )
    p_win = {tid: progression[tid]["winner"] for tid in progression}

    return PredictionResult(
        params=params, p_win=p_win, progression=progression, kappa=hp.kappa
    )


def backtest(
    as_of_date: str,
    teams: Sequence,
    tournament_config: dict,
    match_results: Sequence[MatchResult],
    fifa_ratings: Sequence[FifaRating],
    team_xg: Sequence,
    match_odds: Sequence[MatchOdds],
    actual_results: Sequence[MatchResult],
    tournament_market_odds: Optional[Sequence[MatchOdds]] = None,
    *,
    hyperparams: Hyperparams,
    squad_values: Optional[Sequence[dict]] = None,
    n_sims: int = 10000,
    seed: int = 0,
) -> BacktestReport:
    """Predict on pre-cutoff inputs, then score each REAL pairing (SPEC §10).

    Match-conditional scoring: for every actual played fixture, the model's
    pre-match 1X2 for that exact pairing (via ``predict_match``) is scored
    against the real outcome with ``evaluate``. The simulated bracket is NOT
    scored against reality — it is the tournament-level view only. If
    ``tournament_market_odds`` is supplied it is de-vigged and passed as
    ``evaluate``'s market for a direct model-vs-market comparison.
    """
    prediction = run_prediction(
        as_of_date,
        teams,
        tournament_config,
        match_results,
        fifa_ratings,
        team_xg,
        match_odds,
        hyperparams=hyperparams,
        squad_values=squad_values,
        n_sims=n_sims,
        seed=seed,
    )

    odds_by_match: Dict[str, List[Tuple[float, float, float]]] = {}
    if tournament_market_odds is not None:
        for o in tournament_market_odds:
            odds_by_match.setdefault(o["match_id"], []).append(
                (o["odds_home"], o["odds_draw"], o["odds_away"])
            )

    predicted: List[Tuple[float, float, float]] = []
    actual: List[str] = []
    market: List[Tuple[float, float, float]] = []
    market_ok = tournament_market_odds is not None

    for m in actual_results:
        home, away = m["home_team_id"], m["away_team_id"]
        home_flag = not m.get("neutral", True)
        predicted.append(prediction.predict_match(home, away, home_flag)["probs"])
        actual.append(_outcome(m["home_goals"], m["away_goals"]))
        if market_ok:
            books = odds_by_match.get(m["match_id"])
            if books:
                market.append(consensus_probs(books))
            else:
                market_ok = False  # incomplete market -> cannot compare fairly

    market_arg = market if (market_ok and market) else None
    match_metrics = evaluate(predicted, actual, market=market_arg)

    # Sanity figure: the eventual champion's predicted P(win). The final is taken
    # as the latest-dated actual fixture.
    champion: Optional[str] = None
    champion_p_win: Optional[float] = None
    if actual_results:
        final = max(actual_results, key=lambda m: m["date"])
        oc = _outcome(final["home_goals"], final["away_goals"])
        champion = final["away_team_id"] if oc == "away" else final["home_team_id"]
        champion_p_win = prediction.p_win.get(champion)

    return BacktestReport(
        match_metrics=match_metrics,
        p_win=prediction.p_win,
        champion=champion,
        champion_p_win=champion_p_win,
        prediction=prediction,
    )
