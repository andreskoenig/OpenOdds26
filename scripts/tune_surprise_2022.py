"""Tune the surprise factor (kappa, theta) with the base hyperparameters FIXED.

Leakage-free validation, identical setup to scripts/tune_hparams_2022.py:
  - Fit as-of 2022-08-01 (only data strictly before that date).
  - Score the 224 NON-World-Cup internationals played [2022-08-01, 2022-11-19].
  - 64 WC2022 games NOT used.

Base hyperparameters are HELD FIXED (from the prior free tuning pass):
  xi=0.0008, lambda_reg=8.0, c_a=c_d=0.30, c_x=c_y=0.10.
Only kappa (U -> goal dispersion, SPEC sec.5) and theta (M -> atk_prior nudge)
are swept. kappa=theta=0 recovers the surprise-OFF baseline, so the sweep cannot
do worse than that.

U/M come from form-window Bet365 closing odds in data/match_odds.json;
build_features de-vigs and skips matches with no odds (graceful degradation).
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
RESULTS_PATH = os.path.join(ROOT, "scripts", "tune_surprise_results.json")

# FIXED base hyperparameters.
BASE = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,
            blend_weight=0.7, n_recent=10)

KAPPAS = [0.0, 0.5, 1.0, 2.0]
THETAS = [0.0, 0.05, 0.1]


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
    odds = _load("data/match_odds.json")   # form-window + WC; build_features filters by date

    cut = date.fromisoformat(AS_OF_VAL)
    val_end = date.fromisoformat(VAL_END)

    pre = [m for m in matches if date.fromisoformat(m["date"]) < cut]
    cnt = Counter()
    for m in pre:
        cnt[m["home_team_id"]] += 1
        cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES}
    teams = [t for t in teams_all if t["team_id"] in eligible]

    val = [m for m in matches
           if m["competition"] != "FIFA World Cup"
           and cut <= date.fromisoformat(m["date"]) <= val_end
           and m["home_team_id"] in eligible and m["away_team_id"] in eligible]

    # Form-window odds available strictly before the fit date (what U/M can use).
    pre_odds_mids = {r["match_id"] for r in odds}
    n_pre_odds = sum(1 for m in pre if m["match_id"] in pre_odds_mids)
    print(f"fit as-of {AS_OF_VAL} | eligible teams {len(teams)} | validation matches {len(val)}")
    print(f"odds rows total {len(odds)} | pre-Aug-2022 matches with odds (U/M input): {n_pre_odds}")
    print(f"base FIXED: {BASE}\n")

    # Features depend on the as-of date + odds, NOT on kappa/theta -> build once.
    feats = build_features(AS_OF_VAL, teams, matches, fifa, [], odds,
                           xi=BASE["xi"], blend_weight=BASE["blend_weight"],
                           n_recent=BASE["n_recent"])
    nz_u = sum(1 for f in feats if f["upset_propensity"] != 0.0)
    nz_m = sum(1 for f in feats if f["market_adj_perf"] != 0.0)
    print(f"teams with non-zero U: {nz_u}/{len(feats)} | non-zero M: {nz_m}/{len(feats)}\n")

    results = []
    best = None
    for theta in THETAS:
        params = fit_model(AS_OF_VAL, teams, matches, feats,
                           xi=BASE["xi"], lambda_reg=BASE["lambda_reg"],
                           c_a=BASE["c_a"], c_x=BASE["c_x"], c_d=BASE["c_d"],
                           c_y=BASE["c_y"], theta=theta)
        for kappa in KAPPAS:
            predicted, actual = [], []
            for m in val:
                hf = not m.get("neutral", False)
                P = matchup_matrix(params, m["home_team_id"], m["away_team_id"], hf, kappa=kappa)
                predicted.append(result_probs(P))
                actual.append(_outcome(m))
            mm = evaluate(predicted, actual)
            row = {"kappa": kappa, "theta": theta, "val_log_loss": mm["log_loss"],
                   "val_brier": mm["brier"], "val_rps": mm["rps"], "n": len(actual)}
            results.append(row)
            if best is None or mm["log_loss"] < best["val_log_loss"]:
                best = row
            tag = "  <- baseline" if (kappa == 0.0 and theta == 0.0) else ""
            print(f"  kappa={kappa:<4} theta={theta:<5} -> val log-loss={mm['log_loss']:.4f} "
                  f"(Brier={mm['brier']:.4f}){tag}", flush=True)
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump({"base": BASE, "results": results, "best": best}, f, indent=2)

    print("\n" + "=" * 64)
    print(f"BEST: kappa={best['kappa']} theta={best['theta']} "
          f"-> val log-loss {best['val_log_loss']:.4f}")
    base_ll = next(r["val_log_loss"] for r in results if r["kappa"] == 0 and r["theta"] == 0)
    print(f"(surprise-OFF baseline = {base_ll:.4f}; improvement = {base_ll - best['val_log_loss']:+.4f})")
    print("=" * 64)


if __name__ == "__main__":
    main()
