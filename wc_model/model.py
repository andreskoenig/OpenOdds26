"""The match engine: Dixon–Coles fit and scoreline matrix (SPEC §3, §4).

Pure deterministic computation. ``fit_model`` maximizes the time-weighted,
ridge-penalized Dixon–Coles log-likelihood (SPEC §4); ``score_matrix`` produces
the renormalized scoreline probability matrix that is the model's only output
primitive; ``matchup_matrix`` adapts fitted params (incl. the §5 U→dispersion
mapping) into the callable ``simulate_tournament`` expects.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, poisson

from .schemas import FeatureRecord, MatchResult, ModelParams

MAX_GOALS = 10

# Conservative fixed bound on rho for the fit (SPEC §3 NOTE).
# TODO: replace with exact per-fixture bounds
# [max(-1/lam_home, -1/lam_away), min(1/(lam_home*lam_away), 1)] evaluated across
# the fitted lambdas. The score_matrix negative-cell assert is the backstop.
_RHO_BOUND = 0.2


def _as_date(s: str) -> date:
    return date.fromisoformat(s)


def _team_id(team) -> str:
    return team["team_id"] if isinstance(team, dict) else team


def fit_model(
    as_of_date: str,
    teams: Sequence,
    match_results: Sequence[MatchResult],
    feature_records: Sequence[FeatureRecord],
    *,
    xi: float,
    lambda_reg: float,
    c_a: float,
    c_x: float,
    c_d: float,
    c_y: float,
    theta: float = 0.0,
    c_v: float = 0.0,
    c_m: float = 0.0,
    max_history_years: float = 0.0,
    friendly_weight: float = 1.0,
) -> ModelParams:
    """Fit μ, γ, ρ and per-team atk/def by ridge-penalized weighted MLE (SPEC §4).

    Priors are built here (not in ``build_features``) from the feature records:

        atk_prior_t = c_a·z_fifa_t + c_x·attack_index_t + theta·M_t + c_v·z_squad_value_t
        def_prior_t = c_d·z_fifa_t + c_y·defense_index_t + c_v·z_squad_value_t

    Objective (maximized):

        Σ_m w(t_m)·log P_raw(scoreline_m)
          − lambda_reg · Σ_t [(atk_t − atk_prior_t)² + (def_t − def_prior_t)²]

    where ``w(t)=exp(-xi·t)``, ``t`` = days from match m to ``as_of_date``, and
    only matches strictly dated before ``as_of_date`` are used. The per-match
    term is the RAW Dixon–Coles pmf at the single observed scoreline
    ``τ(x,y)·Pois(x;λ_home)·Pois(y;λ_away)`` — NOT the grid-normalized
    ``score_matrix`` (its truncated-grid renormalization is a predict-time
    convenience only).

    Home advantage applies to the home side of non-neutral matches (``H_i=1``).
    ``rho`` is box-constrained to keep τ cells positive across the fitted
    lambdas. Each team's U is carried into the returned ``ModelParams``.
    """
    as_of = _as_date(as_of_date)
    team_ids = [_team_id(t) for t in teams]
    idx = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)

    feat_by = {r["team_id"]: r for r in feature_records}
    atk_prior = np.zeros(n)
    def_prior = np.zeros(n)
    u_vec = np.zeros(n)
    for t, i in idx.items():
        r = feat_by.get(t)
        if r is not None:
            z_sv = r.get("z_squad_value", 0.0)
            z_mkt = r.get("z_market", 0.0)
            atk_prior[i] = (c_a * r["z_fifa"] + c_x * r["attack_index"]
                            + theta * r["market_adj_perf"] + c_v * z_sv + c_m * z_mkt)
            def_prior[i] = (c_d * r["z_fifa"] + c_y * r["defense_index"]
                            + c_v * z_sv + c_m * z_mkt)
            u_vec[i] = r["upset_propensity"]

    # --- Assemble training matches (strictly before as_of) -----------------
    # Optional hard truncation: drop matches older than max_history_years (a clean
    # backstop on top of the time decay; at the shipped half-life such matches
    # already carry <~1% weight).
    lo = (as_of - timedelta(days=max_history_years * 365.25)
          if max_history_years and max_history_years > 0 else None)
    hi, ai, xx, yy, ww, hh = [], [], [], [], [], []
    for m in match_results:
        md = _as_date(m["date"])
        if md >= as_of:
            continue
        if lo is not None and md < lo:
            continue
        if m["home_team_id"] not in idx or m["away_team_id"] not in idx:
            continue
        hi.append(idx[m["home_team_id"]])
        ai.append(idx[m["away_team_id"]])
        xx.append(m["home_goals"])
        yy.append(m["away_goals"])
        days = (as_of - md).days
        w = math.exp(-xi * days)
        if friendly_weight != 1.0 and m.get("competition") == "Friendly":
            w *= friendly_weight
        ww.append(w)
        hh.append(0.0 if m.get("neutral", False) else 1.0)

    hi = np.asarray(hi, dtype=int)
    ai = np.asarray(ai, dtype=int)
    xx = np.asarray(xx, dtype=float)
    yy = np.asarray(yy, dtype=float)
    ww = np.asarray(ww, dtype=float)
    hh = np.asarray(hh, dtype=float)

    is00 = (xx == 0) & (yy == 0)
    is01 = (xx == 0) & (yy == 1)
    is10 = (xx == 1) & (yy == 0)
    is11 = (xx == 1) & (yy == 1)
    lgx = gammaln(xx + 1.0)
    lgy = gammaln(yy + 1.0)

    def neg_objective(v: np.ndarray) -> float:
        mu, gamma, rho = v[0], v[1], v[2]
        atk = v[3 : 3 + n]
        deff = v[3 + n : 3 + 2 * n]
        penalty = lambda_reg * float(
            np.sum((atk - atk_prior) ** 2) + np.sum((deff - def_prior) ** 2)
        )
        if hi.size == 0:
            return penalty

        lam_home = np.exp(mu + atk[hi] - deff[ai] + gamma * hh)
        lam_away = np.exp(mu + atk[ai] - deff[hi])
        if not (np.all(np.isfinite(lam_home)) and np.all(np.isfinite(lam_away))):
            return 1e12

        log_pois = (
            xx * np.log(lam_home) - lam_home - lgx
            + yy * np.log(lam_away) - lam_away - lgy
        )

        tau = np.ones_like(lam_home)
        tau[is00] = 1.0 - lam_home[is00] * lam_away[is00] * rho
        tau[is01] = 1.0 + lam_home[is01] * rho
        tau[is10] = 1.0 + lam_away[is10] * rho
        tau[is11] = 1.0 - rho
        if np.any(tau <= 0):
            return 1e12  # out-of-range rho: invalid pmf, push optimizer away

        loglik = float(np.sum(ww * (log_pois + np.log(tau))))
        return -(loglik - penalty)

    mu0 = math.log(max(float(np.mean(np.concatenate([xx, yy]))), 1e-3)) if xx.size else 0.0
    v0 = np.concatenate([[mu0, 0.1, 0.0], atk_prior, def_prior])
    bounds = (
        [(-3.0, 3.0), (-2.0, 2.0), (-_RHO_BOUND, _RHO_BOUND)]
        + [(-3.0, 3.0)] * n
        + [(-3.0, 3.0)] * n
    )

    res = minimize(neg_objective, v0, method="L-BFGS-B", bounds=bounds)
    v = res.x

    mu, gamma, rho = float(v[0]), float(v[1]), float(v[2])
    atk = {t: float(v[3 + idx[t]]) for t in team_ids}
    deff = {t: float(v[3 + n + idx[t]]) for t in team_ids}
    u_map = {t: float(u_vec[idx[t]]) for t in team_ids}

    return ModelParams(
        mu=mu,
        gamma=gamma,
        rho=rho,
        atk=atk,
        def_=deff,
        xi=xi,
        U=u_map,
        lambda_reg=lambda_reg,
        c_a=c_a,
        c_x=c_x,
        c_d=c_d,
        c_y=c_y,
        theta=theta,
    )


def _goal_marginal(mean: float, max_goals: int, dispersion: Optional[float]) -> np.ndarray:
    """Goal-count pmf vector over ``0..max_goals`` for one side.

    ``dispersion`` (NB over-dispersion α, var = mean + α·mean²) is None or 0 for
    a pure Poisson marginal; α > 0 gives a negative binomial of the *same mean*
    with fatter tails (SPEC §5). Internally α maps to NB size ``r = 1/α`` and
    success prob ``p = r / (r + mean)``; as α → 0, r → ∞ and the NB → Poisson.
    """
    k = np.arange(max_goals + 1)
    if dispersion is None or dispersion == 0:
        return poisson.pmf(k, mean)
    if dispersion < 0:
        raise ValueError("dispersion (over-dispersion alpha) must be >= 0")
    r = 1.0 / dispersion
    p = r / (r + mean)
    return nbinom.pmf(k, r, p)


def score_matrix(
    atk_i: float,
    def_i: float,
    atk_j: float,
    def_j: float,
    mu: float,
    gamma: float,
    rho: float,
    home: bool = False,
    max_goals: int = MAX_GOALS,
    dispersion_home: Optional[float] = None,
    dispersion_away: Optional[float] = None,
) -> np.ndarray:
    """Dixon–Coles scoreline probability matrix for one fixture (SPEC §3).

    Team ``i`` is home, team ``j`` is away. Goal rates follow §3:

        log λ_home = μ + atk_i − def_j + γ·H_i      (H_i = 1 iff ``home``)
        log λ_away = μ + atk_j − def_i

    Builds the independent base ``P0[x, y]`` from each side's goal marginal,
    applies the Dixon–Coles τ low-score correction to the four cells (0,0),
    (0,1), (1,0), (1,1) exactly as in §3, and renormalizes to sum to 1. Returns
    a ``(max_goals+1) x (max_goals+1)`` numpy array indexed ``P[home_goals,
    away_goals]``.

    ``dispersion_home`` / ``dispersion_away`` (SPEC §5): optional NB
    over-dispersion α for that side's marginal (same mean, fatter tails).
    Default None = pure Poisson.
    """
    h = 1.0 if home else 0.0
    lam_home = math.exp(mu + atk_i - def_j + gamma * h)
    lam_away = math.exp(mu + atk_j - def_i)

    marg_home = _goal_marginal(lam_home, max_goals, dispersion_home)
    marg_away = _goal_marginal(lam_away, max_goals, dispersion_away)

    # Independent base P0[x, y] = P(home=x) · P(away=y).
    p0 = np.outer(marg_home, marg_away)

    # Dixon–Coles low-score dependence correction (τ uses the Poisson means).
    tau = np.ones_like(p0)
    tau[0, 0] = 1.0 - lam_home * lam_away * rho
    tau[0, 1] = 1.0 + lam_home * rho
    tau[1, 0] = 1.0 + lam_away * rho
    tau[1, 1] = 1.0 - rho

    p = tau * p0
    # An in-range rho keeps every tau-adjusted cell non-negative (SPEC §3 NOTE).
    # Assert before renormalizing so an out-of-range rho fails loudly instead of
    # renormalizing garbage.
    if p.min() < 0:
        raise ValueError(
            "negative scoreline cell after tau correction: rho is out of the "
            "valid range [max(-1/lam_home, -1/lam_away), "
            "min(1/(lam_home*lam_away), 1)]"
        )
    total = p.sum()
    if total <= 0:
        raise ValueError("scoreline matrix has non-positive total mass")
    return p / total


def matchup_matrix(
    params: ModelParams,
    home_id: str,
    away_id: str,
    home_flag: bool,
    *,
    kappa: float = 0.0,
    r0: float = 8.0,
) -> np.ndarray:
    """Adapter: fitted params -> scoreline matrix for a fixture (SPEC §3, §5).

    Pulls atk/def/μ/γ/ρ from ``params`` and maps each side's upset propensity U
    to a goal-dispersion via the §5 mapping ``r_T = r0 / (1 + kappa·(U_T − Ū))``
    (NB size). The over-dispersion passed to ``score_matrix`` is ``α = 1/r_T``,
    so higher U → larger α → fatter tails. ``kappa = 0`` → pure Poisson.

    This is the callable signature ``simulate_tournament`` expects
    (``matrix_fn(home_id, away_id, home_flag)``); bind ``kappa``/``r0`` via a
    closure or ``functools.partial`` to plug fitted params into the simulator.
    """
    atk_i, def_i = params.atk[home_id], params.def_[home_id]
    atk_j, def_j = params.atk[away_id], params.def_[away_id]

    if kappa == 0:
        disp_home = disp_away = None
    else:
        u_values = np.array(list(params.U.values()), dtype=float)
        u_mean = float(u_values.mean()) if u_values.size else 0.0

        def _alpha(team_id: str) -> float:
            u = params.U.get(team_id, u_mean)
            denom = max(1.0 + kappa * (u - u_mean), 1e-6)
            r = r0 / denom
            return 1.0 / r  # over-dispersion alpha = 1 / NB size

        disp_home = _alpha(home_id)
        disp_away = _alpha(away_id)

    return score_matrix(
        atk_i, def_i, atk_j, def_j,
        params.mu, params.gamma, params.rho,
        home=home_flag,
        dispersion_home=disp_home,
        dispersion_away=disp_away,
    )


# --- Derived markets: pure sums over the matrix (SPEC §3) -------------------


def _index_grids(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(p.shape)
    return rows, cols  # rows = home goals, cols = away goals


def result_probs(p: np.ndarray) -> Tuple[float, float, float]:
    """1X2 probabilities ``(home_win, draw, away_win)`` (SPEC §3)."""
    x, y = _index_grids(p)
    home = float(p[x > y].sum())
    draw = float(p[x == y].sum())
    away = float(p[x < y].sum())
    return home, draw, away


def over_under(p: np.ndarray, line: float = 2.5) -> Tuple[float, float]:
    """Total-goals ``(over, under)`` for a given ``line`` (default 2.5, SPEC §3)."""
    x, y = _index_grids(p)
    total = x + y
    over = float(p[total > line].sum())
    under = float(p[total < line].sum())
    return over, under


def btts(p: np.ndarray) -> float:
    """Both-teams-to-score probability ``Σ_{x≥1, y≥1} P(x, y)`` (SPEC §3)."""
    x, y = _index_grids(p)
    return float(p[(x >= 1) & (y >= 1)].sum())


def correct_score(p: np.ndarray, x: int, y: int) -> float:
    """Exact-score probability ``P(x, y)`` directly (SPEC §3)."""
    return float(p[x, y])
