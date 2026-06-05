"""Build the group-stage CSV from the by-date predictions JSON.

Columns: model 1X2 (integer %), predicted_result (from the 1X2 probabilities,
NOT the modal score), expected goals per side (marginal means of the score
matrix), and the top-3 most-likely exact scorelines with probabilities. No single
"exact score" headline -- the scoreline information lives in exp_goals + top3.
"""
import csv, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "data", "predict_groupstage_by_date_2026.json"), encoding="utf-8"))
nm = d["team_names"]

rows = []
for i, g in enumerate(d["games"], 1):
    # Headline result from the 1X2 probabilities (robust), not the modal score.
    ml = g["most_likely_result"]
    res = {"home": f"{nm[g['home']]} win", "away": f"{nm[g['away']]} win", "draw": "Draw"}[ml]
    exg = f"{g['exp_goals_home']:.2f}-{g['exp_goals_away']:.2f}"
    top3 = ", ".join(f"{h}-{a} {round(pr * 100)}%" for h, a, pr in g["top3_scores"])
    rows.append([i, g["date"], g["group"], nm[g["home"]], nm[g["away"]],
                 round(g["p_home"] * 100), round(g["p_draw"] * 100), round(g["p_away"] * 100),
                 res, exg, top3])

out = os.path.join(ROOT, "data", "predict_groupstage_2026.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["game", "date", "group", "home", "away", "p_home", "p_draw", "p_away",
                "predicted_result", "exp_goals", "top3_scores"])
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} games)")
