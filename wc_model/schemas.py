"""Data schemas — the agents' output contracts (SPEC §7).

One TypedDict per table. All series keyed on canonical ``team_id`` and carry
dates so the deterministic layer can enforce an as-of cutoff (no look-ahead).
These are structural contracts only; no logic lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TypedDict


class Team(TypedDict):
    """A canonical national team (table: ``teams``)."""

    team_id: str
    canonical_name: str
    aliases: List[str]
    confederation: str


class MatchResult(TypedDict):
    """A single historical match (table: ``match_results``)."""

    match_id: str
    date: str
    home_team_id: str
    away_team_id: str
    venue_country: str
    neutral: bool
    competition: str
    home_goals: int
    away_goals: int


class FifaRating(TypedDict):
    """Point-in-time FIFA ranking snapshot (table: ``fifa_ratings``)."""

    team_id: str
    as_of_date: str
    fifa_points: float
    fifa_rank: int


class TeamXg(TypedDict):
    """Per-match expected goals for a team (table: ``team_xg``)."""

    match_id: str
    team_id: str
    xg_for: float
    xg_against: float


class MatchOdds(TypedDict):
    """Pre-match closing 1X2 decimal odds from one book (table: ``match_odds``).

    At least three books per match.
    """

    match_id: str
    bookmaker: str
    odds_home: float
    odds_draw: float
    odds_away: float
    captured_at: str


class FeatureRecord(TypedDict):
    """Per-team raw features produced by ``build_features`` (SPEC §4, §5).

    Raw features only — these are NOT yet combined into atk/def priors (that,
    along with c_a/c_x/... and the kappa/theta mappings, belongs to fit_model
    and the predict step).
    """

    team_id: str
    z_fifa: float
    attack_index: float
    defense_index: float
    upset_propensity: float  # U_T  (>= 0, direction-agnostic)
    market_adj_perf: float  # M_T  (signed)
    z_squad_value: float  # standardized log squad/talent-pool value (0 = neutral/thin)
    z_market: float  # standardized log market (Polymarket winner) prob (0 = neutral/unpriced)


class TournamentConfig(TypedDict):
    """Hand-maintained tournament configuration (table: ``tournament_config``).

    Not fetched. Confirm against the official FIFA source.
    """

    groups: dict
    schedule: list
    host_cities: List[str]
    tiebreak_rules: dict
    best_thirds_rule: dict


@dataclass
class ModelParams:
    """Fitted Dixon–Coles parameters (SPEC §3, §4) + predict-time carry-overs.

    ``mu``/``gamma``/``rho`` are the global baseline, home advantage, and
    low-score dependence; ``atk``/``def_`` are per-team attacking/defensive
    ratings. ``xi`` is the time-decay used in the fit (kept so predict/sim use
    the same "form"). ``U`` carries each team's upset propensity for the §5
    predict-time dispersion mapping. The remaining fields record the
    hyperparameters used to build the priors and nudges.
    """

    mu: float
    gamma: float
    rho: float
    atk: Dict[str, float]
    def_: Dict[str, float]
    xi: float
    U: Dict[str, float] = field(default_factory=dict)
    lambda_reg: float = 0.0
    c_a: float = 0.0
    c_x: float = 0.0
    c_d: float = 0.0
    c_y: float = 0.0
    theta: float = 0.0


@dataclass
class Hyperparams:
    """The single bundle of knobs threaded through the whole chain.

    One object flows into ``build_features``, ``fit_model``, and
    ``matchup_matrix`` so a value (notably ``xi`` — the SPEC §4 "form" decay) can
    never drift between the feature step and the fit step. The ``*_kwargs``
    helpers project this object onto each stage's keyword arguments, drawing
    every shared value from the same field.
    """

    xi: float
    lambda_reg: float
    c_a: float
    c_x: float
    c_d: float
    c_y: float
    theta: float = 0.0
    kappa: float = 0.0
    c_v: float = 0.0   # squad-value prior weight (atk_prior/def_prior += c_v * z_squad_value)
    c_m: float = 0.0   # market (Polymarket winner) prior weight (+= c_m * z_market)
    blend_weight: float = 0.7
    n_recent: int = 10

    def feature_kwargs(self) -> Dict[str, float]:
        """Keyword args for ``build_features`` (xi shared with the fit)."""
        return {"xi": self.xi, "blend_weight": self.blend_weight, "n_recent": self.n_recent}

    def fit_kwargs(self) -> Dict[str, float]:
        """Keyword args for ``fit_model`` (xi shared with the feature step)."""
        return {
            "xi": self.xi,
            "lambda_reg": self.lambda_reg,
            "c_a": self.c_a,
            "c_x": self.c_x,
            "c_d": self.c_d,
            "c_y": self.c_y,
            "theta": self.theta,
            "c_v": self.c_v,
            "c_m": self.c_m,
        }
