"""Monte Carlo tournament simulation (SPEC §6).

Pure deterministic computation (seeded RNG for reproducibility). Samples
scorelines from each match's matrix, runs the group stage and knockouts under
the configured FIFA rules, and aggregates progression probabilities across
runs.

Inputs (see ``simulate_tournament``):
- ``teams``: mapping ``team_id -> {"rating": float, "host": bool}``. ``rating``
  is used only for the small penalty-shootout tilt; ``host`` enables the home
  flag (SPEC §6, host advantage for USA/CAN/MEX in their own country).
- ``matrix_fn(home_id, away_id, home_flag) -> np.ndarray``: returns the §3
  scoreline matrix for any fixture; it closes over the global params and per-
  team atk/def ratings.
- ``config``: tournament_config with ``groups`` and (optionally)
  ``advance_per_group``, ``best_thirds``, ``tiebreak_rules``.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Knockout round labels keyed by the number of teams entering the round.
_ROUND_LABELS = {64: "R64", 32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "final"}

# Extra time is 30 of 90 minutes (SPEC §6).
_ET_SCALE = 30.0 / 90.0


def _round_label(size: int) -> str:
    if size not in _ROUND_LABELS:
        raise ValueError(f"unsupported knockout round size: {size}")
    return _ROUND_LABELS[size]


def expected_goals(p: np.ndarray) -> Tuple[float, float]:
    """Marginal expected goals ``(E[home], E[away])`` of a scoreline matrix."""
    rows, cols = np.indices(p.shape)
    return float((rows * p).sum()), float((cols * p).sum())


def sample_scoreline(p: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    """Sample one ``(home_goals, away_goals)`` from the matrix ``P[x, y]``."""
    flat = p.ravel()
    idx = rng.choice(flat.size, p=flat)
    return divmod(int(idx), p.shape[1])  # (row, col) = (home_goals, away_goals)


def _orient(a: str, b: str, teams: Dict[str, dict]) -> Tuple[str, str, bool]:
    """Decide home/away and the home flag for a fixture between ``a`` and ``b``.

    Host advantage (home flag) applies only when exactly one side is a host;
    otherwise the match is neutral (SPEC §6). ``a`` is the nominal home side on
    a neutral fixture for determinism.

    NOTE: without venue data we approximate "in their own country" as "a host
    always plays at home"; replace with schedule-driven venue logic when the
    config schedule is populated.
    """
    host_a = bool(teams.get(a, {}).get("host", False))
    host_b = bool(teams.get(b, {}).get("host", False))
    if host_a and not host_b:
        return a, b, True
    if host_b and not host_a:
        return b, a, True
    return a, b, False


def _match_goals(
    a: str,
    b: str,
    teams: Dict[str, dict],
    matrix_fn: Callable[[str, str, bool], np.ndarray],
    rng: np.random.Generator,
) -> Tuple[int, int, np.ndarray]:
    """Sample regulation goals for ``a`` and ``b``. Returns ``(ga, gb, P)``."""
    home, away, flag = _orient(a, b, teams)
    p = matrix_fn(home, away, flag)
    hg, ag = sample_scoreline(p, rng)
    if home == a:
        return hg, ag, p
    return ag, hg, p


def _knockout_winner(
    a: str,
    b: str,
    teams: Dict[str, dict],
    matrix_fn: Callable[[str, str, bool], np.ndarray],
    rng: np.random.Generator,
) -> str:
    """Resolve a knockout tie to a single winner (SPEC §6).

    Regulation scoreline; on a draw, extra time with λ scaled by 30/90; if still
    level, penalties (~50/50 with a small tilt toward the higher-rated side).
    """
    ga, gb, p = _match_goals(a, b, teams, matrix_fn, rng)
    if ga != gb:
        return a if ga > gb else b

    # Extra time: independent Poisson goals at 30/90 of the match goal rate.
    home, away, _ = _orient(a, b, teams)
    eg_home, eg_away = expected_goals(p)
    et_home = int(rng.poisson(eg_home * _ET_SCALE))
    et_away = int(rng.poisson(eg_away * _ET_SCALE))
    if home == a:
        ga += et_home
        gb += et_away
    else:
        ga += et_away
        gb += et_home
    if ga != gb:
        return a if ga > gb else b

    # Penalties: ~50/50 with a small tilt toward the higher-rated side.
    rating_a = float(teams.get(a, {}).get("rating", 0.0))
    rating_b = float(teams.get(b, {}).get("rating", 0.0))
    tilt = float(np.clip(0.05 * (rating_a - rating_b), -0.15, 0.15))
    return a if rng.random() < 0.5 + tilt else b


def _rank_group(group: List[str], stats: Dict[str, dict]) -> List[str]:
    """Rank a group best-to-worst.

    Fallback tiebreak: points -> goal difference -> goals scored, then original
    group order for determinism.

    TODO: this fallback MUST be replaced with the official FIFA 2026 group
    tiebreak rules (head-to-head, fair-play, drawing of lots, ...) once
    ``config["tiebreak_rules"]`` is populated and confirmed against the official
    FIFA source.
    """
    return sorted(
        group,
        key=lambda t: (
            -stats[t]["pts"],
            -stats[t]["gd"],
            -stats[t]["gf"],
            group.index(t),
        ),
    )


def _play_group(
    group: List[str],
    teams: Dict[str, dict],
    matrix_fn: Callable[[str, str, bool], np.ndarray],
    rng: np.random.Generator,
) -> Tuple[List[str], Dict[str, dict]]:
    """Round-robin a group; award 3/1/0; return (ranked teams, stats)."""
    stats = {t: {"pts": 0, "gd": 0, "gf": 0} for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            ga, gb, _ = _match_goals(a, b, teams, matrix_fn, rng)
            stats[a]["gf"] += ga
            stats[b]["gf"] += gb
            stats[a]["gd"] += ga - gb
            stats[b]["gd"] += gb - ga
            if ga > gb:
                stats[a]["pts"] += 3
            elif gb > ga:
                stats[b]["pts"] += 3
            else:
                stats[a]["pts"] += 1
                stats[b]["pts"] += 1
    return _rank_group(group, stats), stats


def _resolve_slot(slot: str, ranked_by_group: Dict[str, List[str]], advance: int) -> str:
    """Map a bracket slot like ``'1A'`` / ``'2B'`` to its (position, group) team.

    Position is 1-based; the trailing letters are the group name. Only the clean
    top-two case (``position <= advance``) is supported.

    TODO: positions above ``advance`` denote best third-placed qualifiers, whose
    bracket placement follows FIFA's third-place allocation table (2026 format,
    fiddly). Implement that table when the 2026 config is built; the 2022 config
    has best_thirds = 0, so only positions 1 and 2 occur here.
    """
    position = int(slot[0])
    group = slot[1:]
    if position > advance:
        raise NotImplementedError(
            f"bracket slot {slot!r} references a best third-placed team; FIFA's "
            "third-place allocation table is not yet implemented (best_thirds>0)"
        )
    return ranked_by_group[group][position - 1]


def _bracket_seed_slots(config: dict, n_qualifiers: int) -> Optional[List[str]]:
    """First knockout round's slot order from ``config['knockout_bracket']``.

    Returns a flat list of slot strings (e.g. ``['1A','2B','1C','2D', ...]``) in
    seeding order, so adjacent pairs form the crossing's first-round matches and
    the existing forward-play loop reproduces the whole tree. Returns None when
    the config has no usable bracket, so seeding falls back to sequential order.

    The first round is the list of match dicts whose length is ``n_qualifiers/2``
    and whose slots are (position, group) references (they start with a digit);
    later rounds reference winners/losers (``'W ...'`` / ``'L ...'``).
    """
    bracket = config.get("knockout_bracket")
    if not isinstance(bracket, dict):
        return None
    target = n_qualifiers // 2
    for value in bracket.values():
        if not (isinstance(value, list) and len(value) == target):
            continue
        if not all(isinstance(m, dict) and "home" in m and "away" in m for m in value):
            continue
        if not str(value[0]["home"])[:1].isdigit():
            continue  # a later round (winner/loser slots), not the seeding round
        slots: List[str] = []
        for m in value:
            slots.append(m["home"])
            slots.append(m["away"])
        return slots
    return None


def simulate_tournament(
    teams: Dict[str, dict],
    matrix_fn: Callable[[str, str, bool], np.ndarray],
    config: dict,
    n_runs: int = 10000,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Run N Monte Carlo tournaments and aggregate progression (SPEC §6).

    Per run: group stage (3/1/0, ranked, best third-placed teams selected), then
    a single-elimination knockout (R32 → … → final) with extra time and
    penalties on draws and host advantage for hosts. The RNG is seeded so
    identical seed + inputs give identical output.

    Returns ``team_id -> {round_label: P(reach round), ..., "winner": P(win)}``,
    where round labels cover every knockout round from the first one down to
    ``"final"``.
    """
    groups: Dict[str, List[str]] = config["groups"]
    advance = int(config.get("advance_per_group", 2))
    best_thirds = int(config.get("best_thirds", 8))

    group_names = list(groups.keys())
    all_team_ids = [t for g in groups.values() for t in g]

    n_qualifiers = len(group_names) * advance + best_thirds
    if n_qualifiers < 2 or (n_qualifiers & (n_qualifiers - 1)) != 0:
        raise ValueError(
            f"qualifier count must be a power of two >= 2, got {n_qualifiers}"
        )

    # Knockout round sizes from the first round down to the final (size 2).
    sizes = []
    s = n_qualifiers
    while s >= 2:
        sizes.append(s)
        s //= 2
    round_labels = [_round_label(s) for s in sizes]

    counts: Dict[str, Dict[str, float]] = {
        t: {label: 0 for label in round_labels} for t in all_team_ids
    }
    for t in all_team_ids:
        counts[t]["winner"] = 0

    # Bracket crossing (constant across runs); None -> sequential fallback.
    bracket_slots = _bracket_seed_slots(config, n_qualifiers)

    rng = np.random.default_rng(seed)

    for _ in range(n_runs):
        # --- Group stage ---------------------------------------------------
        ranked_by_group: Dict[str, List[str]] = {}
        third_candidates: List[Tuple[str, dict, int]] = []
        for order_idx, gname in enumerate(group_names):
            ranked, stats = _play_group(groups[gname], teams, matrix_fn, rng)
            ranked_by_group[gname] = ranked
            if advance < len(ranked):
                third = ranked[advance]  # team finishing just below the cut
                third_candidates.append((third, stats[third], order_idx))

        # Best third-placed teams (same fallback tiebreak; group order breaks ties).
        third_candidates.sort(
            key=lambda c: (-c[1]["pts"], -c[1]["gd"], -c[1]["gf"], c[2])
        )
        best_third_teams = [c[0] for c in third_candidates[:best_thirds]]

        # Seed the first knockout round.
        if bracket_slots is not None:
            # Follow the config's crossing: map each (position, group) slot to its
            # team; adjacent pairs are the first-round matches (e.g. 1A v 2B).
            qualifiers = [_resolve_slot(s, ranked_by_group, advance) for s in bracket_slots]
        else:
            # Sequential fallback (unchanged): all 1st places, then 2nds, ..., thirds.
            qualifiers = []
            for pos in range(advance):
                for gname in group_names:
                    qualifiers.append(ranked_by_group[gname][pos])
            qualifiers.extend(best_third_teams)

        # --- Knockout bracket ---------------------------------------------
        current = qualifiers
        size = len(current)
        while size >= 2:
            label = _round_label(size)
            for t in current:
                counts[t][label] += 1
            winners: List[str] = []
            for k in range(0, size, 2):
                winners.append(
                    _knockout_winner(current[k], current[k + 1], teams, matrix_fn, rng)
                )
            current = winners
            size //= 2

        counts[current[0]]["winner"] += 1

    return {
        t: {label: counts[t][label] / n_runs for label in (*round_labels, "winner")}
        for t in all_team_ids
    }
