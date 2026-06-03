"""Tune the squad-value prior weight c_v with the base hyperparameters FIXED.

Leakage-free validation, identical setup to the prior sweeps:
  - Fit as-of 2022-08-01 (only data strictly before that date; squad-value uses
    the latest snapshot <= 2022-08-01).
  - Score the 224 NON-World-Cup internationals played [2022-08-01, 2022-11-19].
  - 64 WC2022 games NOT used.

Base hyperparameters HELD FIXED:
  xi=0.0008, lambda_reg=8.0, c_a=c_d=0.30, c_x=c_y=0.10, theta=0, kappa=0.
Only c_v is swept. c_v=0 recovers the surprise-OFF baseline, so the sweep cannot
worsen the final model.
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

AS_OF_VAL = "2022-08-01"
VAL_END = "2022-11-19"
MIN_MATCHES = 50
RESULTS_PATH = os.path.join(ROOT, "scripts", "tune_squad_value_results.json")

BASE = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,
            theta=0.0, blend_weight=0.7, n_recent=10)
C_V_GRID = [0.0, 0.1, 0.2, 0.3]


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
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad_values = _load("data/squad_values.json")

    cut = date.fromisoformat(AS_OF_VAL)
    val_end = date.fromisoformat(VAL_END)

    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES}
    teams = [t for t in teams_all if t["team_id"] in eligible]

    val = [m for m in matches
           if m["competition"] != "FIFA World Cup"
           and cut <= date.fromisoformat(m["date"]) <= val_end
           and m["home_team_id"] in eligible and m["away_team_id"] in eligible]

    print(f"fit as-of {AS_OF_VAL} | eligible teams {len(teams)} | validation matches {len(val)}")
    print(f"base FIXED: {BASE}\n")

    # z_squad_value is computed in build_features (independent of c_v) -> build once.
    feats = build_features(AS_OF_VAL, teams, matches, fifa, [], [],
                           squad_values=squad_values, xi=BASE["xi"],
                           blend_weight=BASE["blend_weight"], n_recent=BASE["n_recent"])
    nz = sum(1 for f in feats if f["z_squad_value"] != 0.0)
    print(f"teams with non-zero z_squad_value (n_players>=10, snapshot<=cutoff): {nz}/{len(feats)}\n")

    results, best = [], None
    for c_v in C_V_GRID:
        params = fit_model(AS_OF_VAL, teams, matches, feats,
                           xi=BASE["xi"], lambda_reg=BASE["lambda_reg"],
                           c_a=BASE["c_a"], c_x=BASE["c_x"], c_d=BASE["c_d"],
                           c_y=BASE["c_y"], theta=BASE["theta"], c_v=c_v)
        predicted, actual = [], []
        for m in val:
            hf = not m.get("neutral", False)
            P = matchup_matrix(params, m["home_team_id"], m["away_team_id"], hf, kappa=0.0)
            predicted.append(result_probs(P))
            actual.append(_outcome(m))
        mm = evaluate(predicted, actual)
        row = {"c_v": c_v, "val_log_loss": mm["log_loss"], "val_brier": mm["brier"],
               "val_rps": mm["rps"], "n": len(actual)}
        results.append(row)
        if best is None or mm["log_loss"] < best["val_log_loss"]:
            best = row
        tag = "  <- baseline" if c_v == 0.0 else ""
        print(f"  c_v={c_v:<4} -> val log-loss={mm['log_loss']:.4f} (Brier={mm['brier']:.4f}){tag}",
              flush=True)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump({"base": BASE, "results": results, "best": best}, f, indent=2)

    base_ll = next(r["val_log_loss"] for r in results if r["c_v"] == 0.0)
    print("\n" + "=" * 60)
    print(f"BEST: c_v={best['c_v']} -> val log-loss {best['val_log_loss']:.4f}")
    print(f"(baseline c_v=0 = {base_ll:.4f}; improvement = {base_ll - best['val_log_loss']:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
