"""Full group-stage predictions (all 72 games, 3 matchdays) for the 2026 World Cup.

No public free 1X2 lines exist for unplayed 2026 games (0/72) and the provider is
blocking, so these are MODEL predictions (shipped config, as-of 2026-06-10),
CLEARLY MARKED — not market-driven.

The 6 round-robin games per group are organised into matchdays by the standard
FIFA pattern (seed indices 0-3), verified against the 2022 schedule:
  MD1: (0 v 1), (2 v 3)   MD2: (0 v 2), (3 v 1)   MD3: (1 v 2), (0 v 3)
The seed-1 host is the nominal home side in all three of its games (so host
advantage — Mexico/Canada/USA only — applies throughout their group).
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

# (home_seed_idx, away_seed_idx) per matchday — standard FIFA group pattern.
MATCHDAYS = {
    1: [(0, 1), (2, 3)],
    2: [(0, 2), (3, 1)],
    3: [(1, 2), (0, 3)],
}


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

    print("=" * 90)
    print("2026 WORLD CUP — FULL GROUP STAGE (all 72 games) — MODEL prediction")
    print("=" * 90)
    print("No free 2026 market odds (0/72) and provider blocking -> DIXON-COLES MODEL")
    print(f"predictions (as-of {AS_OF}, rho={params.rho:.4f}), NOT market-driven. Matchdays")
    print("follow the standard FIFA pattern; host advantage = Mexico/Canada/USA only.\n")

    out = []
    for g in sorted(groups):
        members = groups[g]
        print(f"================ GROUP {g} ================")
        for md in (1, 2, 3):
            print(f"  Matchday {md}:")
            for hi, ai in MATCHDAYS[md]:
                h, a = members[hi], members[ai]
                hf = h in hosts
                P = matchup_matrix(params, h, a, home_flag=hf, kappa=0.0)
                ph, pd, pa = result_probs(P)
                ml = max((("home", ph), ("draw", pd), ("away", pa)), key=lambda kv: kv[1])[0]
                res = {"home": nm(h), "draw": "Draw", "away": nm(a)}[ml]
                sx, sy = divmod(int(np.argmax(P)), P.shape[1])
                ov, _ = over_under(P, 2.5)
                bt = btts(P)
                tag = " (host)" if hf else ""
                print(f"     {nm(h)}{tag} vs {nm(a)}:  {ph*100:.0f}/{pd*100:.0f}/{pa*100:.0f}  "
                      f"-> {res} {sx}-{sy}  (O2.5 {ov*100:.0f}%, BTTS {bt*100:.0f}%)")
                out.append({"group": g, "matchday": md, "home": h, "away": a, "host_advantage": hf,
                            "p_home": ph, "p_draw": pd, "p_away": pa, "most_likely_result": ml,
                            "most_likely_score": [int(sx), int(sy)], "over_2_5": float(ov), "btts": float(bt)})
        print()

    pj = os.path.join(ROOT, "data", "predict_groupstage_2026.json")
    json.dump({"as_of": AS_OF, "rho": params.rho, "source": "model (no market odds available)",
               "matchday_pattern": {str(k): v for k, v in MATCHDAYS.items()},
               "games": out, "team_names": {t: nm(t) for t in all_group}},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {pj}  ({len(out)} group-stage games)")


if __name__ == "__main__":
    main()
