"""Backtest evaluation and validation gates (SPEC §10).

Pure deterministic computation. Scores 1X2 forecasts by log-loss, multiclass
Brier, ranked-probability score (RPS over the ordered home→draw→away outcomes),
and a calibration summary. When market probabilities are supplied, the same
metrics are computed for the market so model-vs-market is directly comparable on
the same fixtures (matching or beating the market is the bar).
"""

from __future__ import annotations

import math
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schemas import MatchResult

_OUTCOMES = ("home", "draw", "away")
_OUTCOME_IDX = {o: i for i, o in enumerate(_OUTCOMES)}
_EPS = 1e-15


def _pred_array(predicted: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    return np.asarray(predicted, dtype=float).reshape(-1, 3)


def _onehot(actual: Sequence[str]) -> np.ndarray:
    y = np.array([_OUTCOME_IDX[o] for o in actual], dtype=int)
    oh = np.zeros((len(y), 3), dtype=float)
    oh[np.arange(len(y)), y] = 1.0
    return oh


def log_loss(
    predicted: Sequence[Tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Mean multiclass log-loss of 1X2 predictions (SPEC §10)."""
    p = _pred_array(predicted)
    y = np.array([_OUTCOME_IDX[o] for o in outcomes], dtype=int)
    p_actual = p[np.arange(len(y)), y]
    return float(-np.mean(np.log(np.clip(p_actual, _EPS, 1.0))))


def brier_score(
    predicted: Sequence[Tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Mean multiclass Brier score of 1X2 predictions (SPEC §10)."""
    p = _pred_array(predicted)
    oh = _onehot(outcomes)
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def ranked_probability_score(
    predicted: Sequence[Tuple[float, float, float]],
    outcomes: Sequence[str],
) -> float:
    """Mean ranked-probability score over ordered home→draw→away (SPEC §10).

    Per forecast: ``(1/(K-1)) · Σ_{i=1}^{K-1} (CumP_i − CumO_i)²`` with K=3, so a
    perfect forecast scores 0 and the worst scores 1.
    """
    p = _pred_array(predicted)
    oh = _onehot(outcomes)
    cum_p = np.cumsum(p, axis=1)[:, :2]
    cum_o = np.cumsum(oh, axis=1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (3 - 1)))


def calibration_data(
    predicted: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> Dict[str, object]:
    """Reliability bins + expected calibration error for a binary series (SPEC §10).

    ``predicted`` are probabilities of an event; ``outcomes`` are the matching
    0/1 indicators. Returns ``{"bins": [...], "calibration_error": ECE}`` where
    each bin holds its range, mean predicted prob, observed frequency, and count,
    and ECE is the count-weighted mean ``|mean_pred − mean_obs|``.
    """
    preds = np.asarray(predicted, dtype=float)
    obs = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = preds.size
    bins: List[dict] = []
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        count = int(mask.sum())
        if count > 0:
            mean_pred = float(preds[mask].mean())
            mean_obs = float(obs[mask].mean())
            ece += (count / total) * abs(mean_pred - mean_obs)
        else:
            mean_pred = mean_obs = 0.0
        bins.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "mean_pred": mean_pred,
                "mean_obs": mean_obs,
                "count": count,
            }
        )
    return {"bins": bins, "calibration_error": float(ece)}


def _calibration_multiclass(p: np.ndarray, oh: np.ndarray, n_bins: int) -> Dict[str, object]:
    # Flatten the (sample x class) predicted probs vs their 0/1 indicators.
    return calibration_data(p.ravel(), oh.ravel(), n_bins=n_bins)


def _metrics(
    predicted: Sequence[Tuple[float, float, float]],
    actual: Sequence[str],
    n_bins: int,
) -> Dict[str, object]:
    p = _pred_array(predicted)
    oh = _onehot(actual)
    return {
        "log_loss": log_loss(predicted, actual),
        "brier": brier_score(predicted, actual),
        "rps": ranked_probability_score(predicted, actual),
        "calibration": _calibration_multiclass(p, oh, n_bins),
    }


def lookahead_audit(results: Sequence[MatchResult], as_of_date: str) -> bool:
    """Return True iff no result is dated on/after ``as_of_date`` (SPEC §10).

    A standalone guard for the zero-look-ahead gate: every record fed to a build
    for ``as_of_date`` must be strictly earlier than it.
    """
    cutoff = date.fromisoformat(as_of_date)
    return all(date.fromisoformat(m["date"]) < cutoff for m in results)


def evaluate(
    predicted: Sequence[Tuple[float, float, float]],
    actual: Sequence[str],
    market: Optional[Sequence[Tuple[float, float, float]]] = None,
    *,
    scoreline_pred: Optional[Sequence[np.ndarray]] = None,
    actual_scorelines: Optional[Sequence[Tuple[int, int]]] = None,
    n_bins: int = 10,
) -> Dict[str, object]:
    """Score 1X2 forecasts against actual outcomes (SPEC §10).

    ``predicted`` / ``market`` are sequences of ``(p_home, p_draw, p_away)``;
    ``actual`` is a sequence of outcomes in ``{"home","draw","away"}``. Returns
    log-loss, Brier, RPS, and a calibration summary. If ``market`` is given, the
    same metrics are returned under ``"market"`` for direct comparison on the
    same fixtures.

    If both ``scoreline_pred`` (matrices) and ``actual_scorelines`` (``(x, y)``)
    are supplied, a correct-score log-loss is added under
    ``"correct_score_log_loss"``.
    """
    result = _metrics(predicted, actual, n_bins)
    if market is not None:
        result["market"] = _metrics(market, actual, n_bins)
    if scoreline_pred is not None and actual_scorelines is not None:
        vals = [
            -math.log(max(float(mat[x, y]), _EPS))
            for mat, (x, y) in zip(scoreline_pred, actual_scorelines)
        ]
        result["correct_score_log_loss"] = float(np.mean(vals)) if vals else 0.0
    return result
