"""Build the group-stage CSV from the by-date predictions JSON.

predicted_result is taken from the MODAL SCORE: 'Draw' if the most-likely exact
score is level, otherwise the side the modal score favours. No exact scores in
the CSV. Probabilities are the model 1X2 (integer %).
"""
import csv, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "data", "predict_groupstage_by_date_2026.json"), encoding="utf-8"))
nm = d["team_names"]
rows = []
for i, g in enumerate(d["games"], 1):
    sx, sy = g["most_likely_score"]
    if sx > sy:
        res = f"{nm[g['home']]} win"
    elif sx < sy:
        res = f"{nm[g['away']]} win"
    else:
        res = "Draw"
    rows.append([i, g["date"], g["group"], nm[g["home"]], nm[g["away"]],
                 round(g["p_home"] * 100), round(g["p_draw"] * 100), round(g["p_away"] * 100), res])

out = os.path.join(ROOT, "data", "predict_groupstage_2026.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["game", "date", "group", "home", "away", "p_home", "p_draw", "p_away", "predicted_result"])
    w.writerows(rows)
print(f"wrote {out} ({len(rows)} games)")
draws = sum(1 for r in rows if r[8] == "Draw")
print(f"draws (modal score level): {draws}")
