"""Export the model's most-probable (modal) scoreline for every game -> CSV.

Group stage (72) from data/predict_groupstage_by_date_2026.json (most_likely_score)
plus the Round of 32 (16) from data/knockout_2026.json (modal 90' score). Writes
data/modal_scores_2026.csv, sorted by date.
"""

from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def _load(rel):
    with open(os.path.join(DATA, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    nm = {t["team_id"]: t["canonical_name"] for t in _load("teams.json")}
    name = lambda t: nm.get(t, t)
    rows = []

    g = _load("predict_groupstage_by_date_2026.json")
    for x in g.get("games", []):
        s = x.get("most_likely_score", [None, None])
        rows.append(("Group " + (x.get("group") or ""), x.get("date", ""),
                     name(x["home"]), name(x["away"]),
                     f"{s[0]}-{s[1]}" if s[0] is not None else ""))

    if os.path.exists(os.path.join(DATA, "knockout_2026.json")):
        k = _load("knockout_2026.json")
        for x in k.get("rounds", {}).get("R32", []):
            s = x.get("modal", [None, None])
            rows.append(("Round of 32", x.get("date") or "",
                         x.get("home_name", x["home"]), x.get("away_name", x["away"]),
                         f"{s[0]}-{s[1]}" if s[0] is not None else ""))

    rows.sort(key=lambda r: (r[1], r[0]))
    out = os.path.join(DATA, "modal_scores_2026.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "date", "home", "away", "modal_score"])
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} games)")


if __name__ == "__main__":
    main()
