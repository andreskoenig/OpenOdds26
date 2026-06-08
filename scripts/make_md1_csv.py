"""Build the group-stage FIRST ROUND (matchday 1) CSV from the by-date predictions.

Matchday 1 = each team's opening match = the first two chronological games of each
of the 12 groups (24 fixtures). Same columns as the full group-stage CSV:
expected goals, a rounded headline score, and the top-3 likeliest scorelines.
"""

import csv, json, os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "data", "predict_groupstage_by_date_2026.json"), encoding="utf-8"))
nm = d["team_names"]

by_group = defaultdict(list)
for g in d["games"]:
    by_group[g["group"]].append(g)

md1 = []
for grp, games in by_group.items():
    md1 += sorted(games, key=lambda x: x["date"])[:2]   # first 2 games per group = MD1
md1.sort(key=lambda x: (x["date"], x["group"]))

rows = []
for i, g in enumerate(md1, 1):
    ml = g["most_likely_result"]
    res = {"home": f"{nm[g['home']]} win", "away": f"{nm[g['away']]} win", "draw": "Draw"}[ml]
    exg = f"{g['exp_goals_home']:.2f}-{g['exp_goals_away']:.2f}"
    rounded = f"{round(g['exp_goals_home'])}-{round(g['exp_goals_away'])}"
    top3 = ", ".join(f"{h}-{a} {round(pr * 100)}%" for h, a, pr in g["top3_scores"])
    rows.append([i, g["date"], g["group"], nm[g["home"]], nm[g["away"]],
                 round(g["p_home"] * 100), round(g["p_draw"] * 100), round(g["p_away"] * 100),
                 res, exg, rounded, top3])

out = os.path.join(ROOT, "data", "predict_md1_2026.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["game", "date", "group", "home", "away", "p_home", "p_draw", "p_away",
                "predicted_result", "exp_goals", "rounded_score", "top3_scores"])
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} matchday-1 fixtures)")
