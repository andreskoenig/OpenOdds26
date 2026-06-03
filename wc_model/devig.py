"""De-vigging of bookmaker odds into consensus probabilities (SPEC §5 step 1).

Pure deterministic computation. Converts decimal odds to raw implied
probabilities, removes the overround, and averages de-vigged probabilities
across books into a consensus ``(p_home, p_draw, p_away)``. Basic normalization
for now; Shin / power methods are reserved behind the ``method`` hook.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def devig(
    odds_home: float,
    odds_draw: float,
    odds_away: float,
    method: str = "normalize",
) -> Tuple[float, float, float]:
    """De-vig one book's 1X2 decimal odds (SPEC §5 step 1).

    Raw implied ``q_k = 1/o_k``; overround ``O = Σ q_k``; de-vigged
    ``p_k = q_k / O``. Returns ``(p_home, p_draw, p_away)`` summing to 1.

    ``method`` is a hook for richer de-vig schemes (Shin / power) later; only
    ``"normalize"`` (basic proportional normalization) is implemented now.
    """
    if method != "normalize":
        raise NotImplementedError(f"devig method {method!r} not implemented")
    q = np.array([1.0 / odds_home, 1.0 / odds_draw, 1.0 / odds_away], dtype=float)
    overround = q.sum()
    p = q / overround
    return float(p[0]), float(p[1]), float(p[2])


def consensus_probs(
    books: Sequence[Tuple[float, float, float]],
    method: str = "normalize",
) -> Tuple[float, float, float]:
    """Consensus across books: de-vig each, then average (SPEC §5 step 1).

    ``books`` is a sequence of ``(odds_home, odds_draw, odds_away)`` triplets.
    Each is de-vigged to a probability triplet (each summing to 1); the
    element-wise mean is returned (also summing to 1).
    """
    if len(books) == 0:
        raise ValueError("consensus_probs requires at least one book")
    devigged = np.array([devig(*b, method=method) for b in books], dtype=float)
    mean = devigged.mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2])
