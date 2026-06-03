"""FREE hyperparameter tuning pass (no fetching, surprise OFF, WC2022 untouched).

Leakage-free validation:
  - Fit as-of 2022-08-01 (uses ONLY data strictly before that date).
  - Score every NON-World-Cup international played in [2022-08-01, 2022-11-19]
    with that single fit (no per-match refit). Objective = mean 1X2 log-loss.
  - The 64 World Cup 2022 games are NOT used here (they stay the clean test).

Each grid point does build_features -> fit_model -> per-match matchup_matrix
prediction -> evaluate. No tournament simulation (not needed for match scoring),
so a fit is fast. Results stream to scripts/tune_results.json.

Run:  python scripts/tune_hparams_2022.py [--smoke N]
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.evaluate import evaluate
from wc_model.features import build_features
from wc_model.model import fit_model, matchup_matrix, result_probs
from wc_model.schemas import Hyperparams

AS_OF_VAL = "2022-08-01"
VAL_END = "2022-11-19"
MIN_MATCHES = 50
RESULTS_PATH = os.path.join(ROOT, "scripts", "tune_results.json")

# Grid: vary shrinkage (lambda_reg), prior composition (FIFA c_a/c_d vs index
# c_x/c_y), and time decay (xi). Small, targeted at the over-wide rating spread.
#            xi,     lambda_reg, c_fifa, c_index
GRID = [
    (0.0015, 1.0, 0.20, 0.20),    # current baseline
    (0.0015, 4.0, 0.20, 0.20),    # more shrinkage
    (0.0015, 16.0, 0.20, 0.20),   # strong shrinkage
    (0.0015, 4.0, 0.30, 0.10),    # FIFA-heavy prior
    (0.0015, 4.0, 0.10, 0.30),    # index-heavy prior
    (0.0015, 8.0, 0.30, 0.10),    # strong shrinkage + FIFA-heavy
    (0.0008, 8.0, 0.30, 0.10),    # + slower decay (more history)
    (0.0025, 8.0, 0.30, 0.10),    # + faster decay (recent form)
]


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _outcome(m):
    if m["home_goals"] > m["away_goals"]:
        return "home"
    if m["home_goals"] < m["away_goals"]:
        return "away"
    return "draw"


def main():
    smoke = None
    if "--smoke" in sys.argv:
        smoke = int(sys.argv[sys.argv.index("--smoke") + 1])

    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")

    cut = date.fromisoformat(AS_OF_VAL)
    val_end = date.fromisoformat(VAL_END)

    # Eligible teams: >= MIN_MATCHES strictly before the fit's as-of date.
    pre = [m for m in matches if date.fromisoformat(m["date"]) < cut]
    cnt = Counter()
    for m in pre:
        cnt[m["home_team_id"]] += 1
        cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES}
    teams = [t for t in teams_all if t["team_id"] in eligible]

    # Validation set: NON-WC internationals in (cut, val_end], both teams fit-able.
    val = [
        m for m in matches
        if m["competition"] != "FIFA World Cup"
        and cut <= date.fromisoformat(m["date"]) <= val_end
        and m["home_team_id"] in eligible
        and m["away_team_id"] in eligible
    ]

    print(f"fit as-of {AS_OF_VAL} | pre-cutoff matches {len(pre)} | eligible teams {len(teams)}")
    print(f"validation: {len(val)} non-WC internationals in [{AS_OF_VAL}, {VAL_END}]")
    print(f"sweeping {len(GRID) if smoke is None else min(smoke, len(GRID))} settings\n")

    grid = GRID if smoke is None else GRID[:smoke]
    results = []
    best = None
    for i, (xi, lam, c_fifa, c_index) in enumerate(grid, 1):
        hp = Hyperparams(xi=xi, lambda_reg=lam, c_a=c_fifa, c_x=c_index,
                         c_d=c_fifa, c_y=c_index, theta=0.0, kappa=0.0,
                         blend_weight=0.7, n_recent=10)
        feats = build_features(AS_OF_VAL, teams, matches, fifa, [], [], **hp.feature_kwargs())
        params = fit_model(AS_OF_VAL, teams, matches, feats, **hp.fit_kwargs())

        predicted, actual = [], []
        for m in val:
            hf = not m.get("neutral", False)
            P = matchup_matrix(params, m["home_team_id"], m["away_team_id"], hf, kappa=hp.kappa)
            predicted.append(result_probs(P))
            actual.append(_outcome(m))
        metrics = evaluate(predicted, actual)
        ll = metrics["log_loss"]

        row = {"xi": xi, "lambda_reg": lam, "c_fifa": c_fifa, "c_index": c_index,
               "val_log_loss": ll, "val_brier": metrics["brier"],
               "val_rps": metrics["rps"], "n_scored": len(actual),
               "mu": params.mu, "gamma": params.gamma, "rho": params.rho}
        results.append(row)
        if best is None or ll < best["val_log_loss"]:
            best = row
        print(f"  [{i}/{len(grid)}] xi={xi:<7} lambda_reg={lam:<5} c_fifa={c_fifa} "
              f"c_index={c_index}  ->  val log-loss={ll:.4f}  (Brier={metrics['brier']:.4f}, "
              f"n={len(actual)})", flush=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"validation_window": [AS_OF_VAL, VAL_END], "results": results,
                       "best": best}, f, indent=2)

    print("\n" + "=" * 70)
    print("BEST (validation log-loss):")
    print(f"  xi={best['xi']} lambda_reg={best['lambda_reg']} "
          f"c_a=c_d={best['c_fifa']} c_x=c_y={best['c_index']}  "
          f"-> val log-loss {best['val_log_loss']:.4f}")
    print("=" * 70)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
