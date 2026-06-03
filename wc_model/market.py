"""Market -> scoreline core: calibrate a Dixon-Coles matrix to market 1X2.

Outcomes are MARKET-DRIVEN (w=0): given de-vigged 1X2 probabilities and a FIXED
rho (the model's fitted low-score dependence), solve for (lambda_home,
lambda_away) so the Dixon-Coles scoreline matrix reproduces the market's home-win
and away-win probabilities (draw is implied). The model contributes ONLY the
scoreline shape (rho + the Poisson/DC structure), never the outcome.

Reuses wc_model.model (score_matrix and the derived-market helpers).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import fsolve

from .model import btts, correct_score, over_under, result_probs, score_matrix

MAX_GOALS = 10


def _matrix(lam_home: float, lam_away: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Dixon-Coles matrix with explicit lambdas and fixed rho (mu=gamma=0)."""
    # score_matrix computes lam = exp(atk_i - def_j); set logs so lam == target.
    return score_matrix(math.log(lam_home), 0.0, math.log(lam_away), 0.0,
                        mu=0.0, gamma=0.0, rho=rho, home=False, max_goals=max_goals)


def calibrate_lambdas(p_home: float, p_draw: float, p_away: float, rho: float,
                      max_goals: int = MAX_GOALS) -> Tuple[float, float]:
    """Solve (lambda_home, lambda_away) so the DC matrix hits (p_home, p_away).

    2 targets (home-win, away-win), 2 unknowns; scipy root-find in log-space so
    the lambdas stay positive. rho is held fixed.
    """
    s = p_home + p_draw + p_away
    ph, pa = p_home / s, p_away / s

    def resid(x):
        lh, la = math.exp(x[0]), math.exp(x[1])
        h, _, a = result_probs(_matrix(lh, la, rho, max_goals))
        return [h - ph, a - pa]

    x0 = [math.log(0.6 + 1.6 * ph), math.log(0.6 + 1.6 * pa)]
    sol, info, ier, _ = fsolve(resid, x0, full_output=True)
    if ier != 1:
        # Retry from a neutral start if the heuristic init struggled.
        sol, info, ier, _ = fsolve(resid, [math.log(1.3), math.log(1.1)], full_output=True)
    return math.exp(sol[0]), math.exp(sol[1])


def market_scoreline(p_home: float, p_draw: float, p_away: float, rho: float,
                     max_goals: int = MAX_GOALS) -> Dict[str, object]:
    """Calibrate a DC scoreline matrix to market 1X2 (fixed rho) and derive markets.

    Returns a dict with: ``matrix`` (11x11), ``lambda_home``/``lambda_away``,
    ``probs`` (reproduced p_home, p_draw, p_away), ``most_likely_score`` (x, y),
    ``over_2_5``, ``btts``, and ``top_scores`` (top-6 exact scores with prob).
    """
    lh, la = calibrate_lambdas(p_home, p_draw, p_away, rho, max_goals)
    P = _matrix(lh, la, rho, max_goals)
    flat_idx = int(np.argmax(P))
    mx, my = divmod(flat_idx, P.shape[1])
    over, _ = over_under(P, 2.5)
    order = np.argsort(P, axis=None)[::-1][:6]
    top = [((int(i // P.shape[1]), int(i % P.shape[1])), float(P[i // P.shape[1], i % P.shape[1]]))
           for i in order]
    return {
        "matrix": P,
        "lambda_home": lh,
        "lambda_away": la,
        "probs": result_probs(P),
        "most_likely_score": (int(mx), int(my)),
        "over_2_5": float(over),
        "btts": btts(P),
        "top_scores": top,
    }
