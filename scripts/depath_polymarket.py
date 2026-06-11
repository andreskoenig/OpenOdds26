"""De-path the Polymarket 'Winner' odds -> strength-only market signal.

Polymarket P(win) = strength x bracket-path difficulty. Because the 2026 draw is
fixed and published, that path component is a STRUCTURAL bias (share a half with a
superpower -> your winner odds drop for reasons that are schedule, not ability).
Folding raw winner odds into a *strength* prior imports that bias.

Fix: estimate each team's path factor from our own bracket engine, using a
BASELINE rating set (c_m = 0, so this is not circular) and host advantage OFF on
both sides (so the ratio isolates draw difficulty only):

    path_factor[t] = P_win(actual draw) / mean_k P_win(random draw k)
    depathed[t]    = market[t] / path_factor[t]      (renormalized over priced teams)

A team in an unusually hard bracket has path_factor < 1 -> depathed > market
(we credit it back the strength the schedule was hiding), and vice versa.

Writes data/polymarket_winner_2026_depathed.json.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.features import build_features
from wc_model.model import fit_model, matchup_matrix
from wc_model.simulate import simulate_tournament

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AS_OF = "2026-06-10"
MIN_MATCHES = 50
_FAST = bool(os.environ.get("WC_FAST"))  # pipeline --quick: cheap smoke-test sims
N_ACTUAL = 1500 if _FAST else 6000   # sims for the real-draw P(win)
N_DRAW = 400 if _FAST else 1500      # sims per random draw
K_DRAWS = 12 if _FAST else 40        # random draws averaged for the neutral reference
SEED = 20260610

# Baseline hyperparams: market prior OFF (c_m=0) so path geometry is independent
# of the market signal we are about to correct. Everything else as shipped.
HP = dict(xi=0.0012651, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,
          theta=0.0, c_v=0.1, c_m=0.0, opponent_adjust=True, max_history_years=10.0)


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    pm = _load("data/polymarket_winner_2026.json")
    market = pm["p_market"]
    config = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams_all}

    cfg_ids = {t for g in config["groups"].values() for t in g}
    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES} | cfg_ids
    teams = [t for t in teams_all if t["team_id"] in eligible]

    print("=" * 70)
    print("DE-PATH POLYMARKET — strip fixed-bracket schedule from winner odds")
    print("=" * 70)
    print(f"as-of {AS_OF} | fitted {len(teams)} teams | baseline c_m=0, "
          f"opponent_adjust={HP['opponent_adjust']}, host OFF")
    print(f"actual draw: {N_ACTUAL} sims | neutral: {K_DRAWS} random draws x "
          f"{N_DRAW} sims\n", flush=True)

    # --- Baseline ratings (market prior OFF) -------------------------------
    feats = build_features(AS_OF, teams, matches, fifa, [], [], squad_values=squad,
                           market_probs=None, xi=HP["xi"], blend_weight=0.7, n_recent=10,
                           opponent_adjust=HP["opponent_adjust"],
                           max_history_years=HP["max_history_years"])
    params = fit_model(AS_OF, teams, matches, feats, xi=HP["xi"],
                       lambda_reg=HP["lambda_reg"], c_a=HP["c_a"], c_x=HP["c_x"],
                       c_d=HP["c_d"], c_y=HP["c_y"], theta=HP["theta"],
                       c_v=HP["c_v"], c_m=HP["c_m"],
                       max_history_years=HP["max_history_years"])

    def matrix_fn(home, away, home_flag):
        return matchup_matrix(params, home, away, home_flag, kappa=0.0)

    # host OFF on both sides so the ratio isolates DRAW difficulty, not the
    # hosts' always-home bonus.
    wc48 = [t for g in config["groups"].values() for t in g]
    # strength = atk + def_ (higher def_ = better defense in this parameterization)
    sim_teams = {t: {"rating": params.atk.get(t, 0.0) + params.def_.get(t, 0.0),
                     "host": False} for t in wc48}

    # --- P(win) under the ACTUAL draw --------------------------------------
    prog = simulate_tournament(sim_teams, matrix_fn, config, n_runs=N_ACTUAL, seed=SEED)
    p_actual = {t: prog[t]["winner"] for t in wc48}

    # --- P(win) averaged over RANDOM draws (neutral reference) -------------
    rng = np.random.default_rng(SEED)
    group_names = list(config["groups"].keys())
    gsize = len(next(iter(config["groups"].values())))
    p_neutral_sum = {t: 0.0 for t in wc48}
    for k in range(K_DRAWS):
        perm = list(rng.permutation(wc48))
        shuffled_groups = {
            g: perm[i * gsize:(i + 1) * gsize] for i, g in enumerate(group_names)
        }
        cfg_k = dict(config)
        cfg_k["groups"] = shuffled_groups
        prog_k = simulate_tournament(sim_teams, matrix_fn, cfg_k,
                                     n_runs=N_DRAW, seed=SEED + 1 + k)
        for t in wc48:
            p_neutral_sum[t] += prog_k[t]["winner"]
        if (k + 1) % 10 == 0:
            print(f"  ... {k + 1}/{K_DRAWS} random draws done", flush=True)
    p_neutral = {t: p_neutral_sum[t] / K_DRAWS for t in wc48}

    # --- path factor + de-pathed market ------------------------------------
    # A path factor is a ratio of two simulated win probs; for the long tail
    # (teams that ~never win) that ratio is pure noise and explodes when divided
    # out. So we only de-path teams with a meaningful neutral win prob (>= FLOOR)
    # and clip the factor to a sane band; everyone else keeps the raw market.
    EPS = 1e-9
    FLOOR = 0.005          # 0.5% neutral win prob: below this the factor is noise
    CLIP = (0.5, 2.0)

    def pf(t):
        if p_neutral[t] < FLOOR:
            return 1.0
        f = (p_actual[t] + EPS) / (p_neutral[t] + EPS)
        return min(max(f, CLIP[0]), CLIP[1])

    path_factor = {t: pf(t) for t in wc48}

    priced = [t for t in market]
    depathed_raw = {t: market[t] / path_factor.get(t, 1.0) for t in priced}
    tot = sum(depathed_raw.values())
    depathed = {t: v / tot for t, v in depathed_raw.items()}

    out = {
        "as_of": AS_OF,
        "source": "polymarket world-cup-winner, de-pathed via own bracket engine",
        "method": (f"depathed = market / (P_actual / mean P_random); "
                   f"baseline c_m=0, host OFF; {K_DRAWS} random draws x {N_DRAW} sims"),
        "p_market": depathed,           # <- drop-in replacement for the raw field
        "p_market_raw": market,
        "path_factor": {t: round(path_factor[t], 4) for t in priced},
        "p_actual": {t: round(p_actual[t], 5) for t in priced},
        "p_neutral": {t: round(p_neutral[t], 5) for t in priced},
        "team_names": {t: name[t] for t in priced},
    }
    pj = os.path.join(ROOT, "data", "polymarket_winner_2026_depathed.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # --- report: biggest schedule effects ----------------------------------
    print(f"\n  {'team':<18}{'path':>7}{'raw%':>8}{'depath%':>9}{'shift':>8}")
    order = sorted(priced, key=lambda t: -market[t])
    for t in order[:16]:
        pf = path_factor[t]
        raw = market[t] * 100
        dp = depathed[t] * 100
        tag = "  (hard draw)" if pf < 0.93 else ("  (easy draw)" if pf > 1.07 else "")
        print(f"  {name[t]:<18}{pf:>7.2f}{raw:>7.1f}%{dp:>8.1f}%{dp - raw:>+7.1f}{tag}")
    print(f"\nwrote {pj}")
    print("path<1 = harder-than-average bracket -> credited strength back (depath>raw)")


if __name__ == "__main__":
    main()
