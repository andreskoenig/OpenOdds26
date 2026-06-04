"""Matchday-1 (group-stage first round) predictions for the 2026 World Cup.

No public free 1X2 lines exist for unplayed 2026 games (0/72 coverage) and the
provider is currently blocking, so these are MODEL predictions (shipped config,
as-of 2026-06-10), CLEARLY MARKED — not market-driven.

Matchday-1 pairings follow the standard FIFA pattern within each group:
  game 1 = seed1 v seed2,  game 2 = seed3 v seed4
(verified against the 2022 schedule: Group A MD1 was Qatar[1]-Ecuador[2] and
Senegal[3]-Netherlands[4]). The position-1 side is the nominal home team; host
advantage applies only to Mexico/Canada/USA in their own opener.
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
from wc_model.model import btts, fit_model, matchup_matrix, over_under, result_probs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HP = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0, c_v=0.1)
AS_OF = "2026-06-10"


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    cfg = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    hosts = set(cfg["host_team_ids"])
    groups = cfg["groups"]

    # Matchday 1 from seeded order: seed1 v seed2, seed3 v seed4 (standard pattern).
    md1 = []
    for g, m in groups.items():
        md1.append({"group": g, "home": m[0], "away": m[1]})
        md1.append({"group": g, "home": m[2], "away": m[3]})

    # Fit the shipped 2026 model.
    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for mm in matches:
        if date.fromisoformat(mm["date"]) < cut:
            cnt[mm["home_team_id"]] += 1
            cnt[mm["away_team_id"]] += 1
    all_group = {t for ms in groups.values() for t in ms}
    eligible = {t for t, c in cnt.items() if c >= 50} | all_group
    tlist = [t for t in teams if t["team_id"] in eligible]
    feats = build_features(AS_OF, tlist, matches, fifa, [], [], squad_values=squad,
                           xi=HP["xi"], blend_weight=0.7, n_recent=10)
    params = fit_model(AS_OF, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"],
                       theta=HP["theta"], c_v=HP["c_v"])

    print("=" * 86)
    print("2026 WORLD CUP — GROUP STAGE, MATCHDAY 1 (first round) — MODEL prediction")
    print("=" * 86)
    print("No free 2026 market odds exist (0/72) and the provider is blocking, so these")
    print(f"are the DIXON-COLES MODEL's predictions (as-of {AS_OF}, rho={params.rho:.4f}) — NOT")
    print("market-driven. MD1 pairings: seed1 v seed2, seed3 v seed4 (standard FIFA pattern).\n")

    out = []
    for g in sorted(groups):
        print(f"-- Group {g} --")
        for game in [x for x in md1 if x["group"] == g]:
            h, a = game["home"], game["away"]
            hf = h in hosts
            P = matchup_matrix(params, h, a, home_flag=hf, kappa=0.0)
            ph, pd, pa = result_probs(P)
            ml = max((("home", ph), ("draw", pd), ("away", pa)), key=lambda kv: kv[1])[0]
            res = {"home": f"{nm(h)} win", "draw": "Draw", "away": f"{nm(a)} win"}[ml]
            sx, sy = divmod(int(np.argmax(P)), P.shape[1])
            ov, _ = over_under(P, 2.5)
            bt = btts(P)
            tag = "  (host)" if hf else ""
            print(f"   {nm(h)} vs {nm(a)}{tag}")
            print(f"      1X2: {nm(h)} {ph*100:.0f}% | Draw {pd*100:.0f}% | {nm(a)} {pa*100:.0f}%"
                  f"  ->  {res}; most-likely {sx}-{sy}  (O2.5 {ov*100:.0f}%, BTTS {bt*100:.0f}%)")
            out.append({"group": g, "home": h, "away": a, "host_advantage": hf,
                        "p_home": ph, "p_draw": pd, "p_away": pa, "most_likely_result": ml,
                        "most_likely_score": [int(sx), int(sy)], "over_2_5": float(ov), "btts": float(bt)})
        print()

    pj = os.path.join(ROOT, "data", "predict_md1_2026.json")
    json.dump({"as_of": AS_OF, "rho": params.rho, "source": "model (no market odds available)",
               "md1_rule": "seed1 v seed2, seed3 v seed4", "games": out,
               "team_names": {t: nm(t) for t in all_group}},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {pj}  ({len(out)} matchday-1 games)")


if __name__ == "__main__":
    main()
