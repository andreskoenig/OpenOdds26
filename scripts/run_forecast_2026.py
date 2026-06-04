"""PHASE 4: end-to-end 2026 World Cup forecast (forward prediction; no benchmark).

Locked hyperparameters (xi=0.0008, lambda_reg=8.0, c_a=c_d=0.30, c_x=c_y=0.10,
theta=0, kappa=0, c_v=0.1). As-of cutoff 2026-06-10 (day before the opener) — only
data strictly before it is used. Surprise OFF, no odds, no market benchmark.

Saves data/forecast_2026.json and prints: P(win) for all 48 teams; reach-round
probabilities; per-group expected points / P(advance) / modal standings; ONE
sampled full bracket (a single plausible realization); the model's most-likely
final scoreline.
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

from wc_model.pipeline import run_prediction
from wc_model.schemas import Hyperparams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AS_OF = "2026-06-10"
N_SIMS = 20000
SEED = 20260610
MIN_MATCHES = 50

HP = Hyperparams(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,
                 theta=0.0, kappa=0.0, c_v=0.1, c_m=0.35, blend_weight=0.7, n_recent=10)


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    market = _load("data/polymarket_winner_2026.json")["p_market"]  # Polymarket prior
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

    print("=" * 84)
    print("FIFA WORLD CUP 2026 — forward forecast (free data; surprise OFF; squad prior ON)")
    print("=" * 84)
    print(f"as-of {AS_OF} | teams loaded {len(teams_all)} | fitted {len(teams)} "
          f"| sims {N_SIMS} (seed {SEED})")
    print(f"hyperparams: xi={HP.xi} lambda_reg={HP.lambda_reg} "
          f"c_a/c_x/c_d/c_y={HP.c_a}/{HP.c_x}/{HP.c_d}/{HP.c_y} "
          f"theta={HP.theta} kappa={HP.kappa} c_v={HP.c_v} c_m={HP.c_m} (market prior ON)")
    print("\nfitting + simulating 20,000 tournaments (several minutes) ...\n", flush=True)

    pred = run_prediction(
        AS_OF, teams, config, matches, fifa, team_xg=[], match_odds=[],
        hyperparams=HP, squad_values=squad, market_probs=market,
        n_sims=N_SIMS, seed=SEED, collect_extras=True,
    )
    p = pred.params
    extras = pred.extras
    print(f"fitted globals: mu={p.mu:.4f} gamma={p.gamma:.4f} rho={p.rho:.4f}\n", flush=True)

    nm = lambda tid: name.get(tid, tid)

    # ---- P(win), all 48 sorted ----
    p_win = sorted(pred.p_win.items(), key=lambda kv: kv[1], reverse=True)
    print("-" * 84)
    print("P(WIN TOURNAMENT) — all 48 teams")
    print("-" * 84)
    for i, (tid, pw) in enumerate(p_win, 1):
        print(f"  {i:>2}. {nm(tid):<24} {pw * 100:5.1f}%")

    # ---- reach-round (top 16 by P(win)) ----
    print("\n" + "-" * 84)
    print("P(REACH ROUND) — top 16 by P(win)   [R32  R16   QF   SF  Final  Win]")
    print("-" * 84)
    for tid, _ in p_win[:16]:
        pr = pred.progression[tid]
        print(f"  {nm(tid):<24} {pr['R32']*100:5.1f} {pr['R16']*100:5.1f} "
              f"{pr['QF']*100:5.1f} {pr['SF']*100:5.1f} {pr['final']*100:5.1f} {pr['winner']*100:5.1f}")

    # ---- per-group: expected points, P(advance), modal standings ----
    print("\n" + "-" * 84)
    print("GROUP STAGE — expected points, P(advance to R32), modal final standings")
    print("-" * 84)
    ep, pa = extras["expected_points"], extras["p_advance"]
    for g, members in config["groups"].items():
        print(f"  Group {g}:")
        for tid in sorted(members, key=lambda t: -ep[t]):
            print(f"     {nm(tid):<22} xPts {ep[tid]:4.2f}   P(adv) {pa[tid]*100:5.1f}%")
        modal = extras["modal_standings"][g]
        print(f"     modal order: {' > '.join(nm(t) for t in modal)}")

    # ---- one sampled bracket ----
    s = extras["sample"]
    print("\n" + "-" * 84)
    print("ONE SAMPLED BRACKET (a single plausible realization — NOT the forecast)")
    print("-" * 84)
    print("  Group winners / runners-up (this realization):")
    for g, rows in s["groups"].items():
        w, r = rows[0]["team"], rows[1]["team"]
        print(f"     {g}: 1.{nm(w)}  2.{nm(r)}")
    label_name = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
                  "SF": "Semi-finals", "final": "FINAL", None: "Third place"}
    cur = None
    for m in s["knockout"]:
        if m["label"] != cur:
            cur = m["label"]
            print(f"\n  {label_name.get(cur, cur)}:")
        extra = "" if m["decided_by"] == "regulation" else f" ({m['decided_by']})"
        print(f"     {nm(m['home'])} {m['home_goals']}-{m['away_goals']} {nm(m['away'])}"
              f"{extra}  -> {nm(m['winner'])}")
    print(f"\n  SAMPLED CHAMPION: {nm(s['champion'])}")

    # ---- model's most-likely scoreline for that sampled final ----
    fin = s["knockout"][-1] if s["knockout"][-1]["label"] == "final" else \
        next(m for m in s["knockout"] if m["label"] == "final")
    fh, fa = fin["home"], fin["away"]
    matrix = pred.predict_match(fh, fa, home_flag=False)["matrix"]
    xy = int(np.argmax(matrix))
    mx, my = divmod(xy, matrix.shape[1])
    print(f"\n  Model's most-likely scoreline for that final ({nm(fh)} vs {nm(fa)}): "
          f"{mx}-{my}  (p={matrix[mx, my]*100:.1f}%)")

    # ---- save JSON ----
    forecast = {
        "as_of": AS_OF, "n_sims": N_SIMS, "seed": SEED,
        "hyperparams": {"xi": HP.xi, "lambda_reg": HP.lambda_reg, "c_a": HP.c_a,
                        "c_x": HP.c_x, "c_d": HP.c_d, "c_y": HP.c_y, "theta": HP.theta,
                        "kappa": HP.kappa, "c_v": HP.c_v},
        "fitted_globals": {"mu": p.mu, "gamma": p.gamma, "rho": p.rho},
        "p_win": {tid: pred.p_win[tid] for tid, _ in p_win},
        "progression": pred.progression,
        "group_expected_points": ep,
        "p_advance": pa,
        "modal_standings": extras["modal_standings"],
        "sampled_bracket": s,
        "sampled_final_modal_scoreline": {"home": fh, "away": fa, "score": [mx, my],
                                          "prob": float(matrix[mx, my])},
        "team_names": {tid: nm(tid) for tid in pred.p_win},
    }
    out = os.path.join(ROOT, "data", "forecast_2026.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(forecast, f, ensure_ascii=False, indent=2)

    # ---- human-readable headline ----
    print("\n" + "=" * 84)
    print("HEADLINE — P(win) top 15")
    print("=" * 84)
    for i, (tid, pw) in enumerate(p_win[:15], 1):
        print(f"  {i:>2}. {nm(tid):<24} {pw * 100:5.1f}%")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
