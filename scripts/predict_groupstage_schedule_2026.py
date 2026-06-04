"""Full 2026 group stage ordered BY MATCH DATE — MODEL prediction (not market).

The real day-by-day calendar (June 11-27, 2026) is taken from the published
schedule (ESPN), since the odds/fixtures API is blocking this session. Each game
is predicted with the fitted Dixon-Coles model. Host advantage (Mexico/Canada/USA
only) is applied to the host in EVERY game it plays (it plays in its own country),
regardless of the schedule's nominal home/away label.
"""

from __future__ import annotations

import json
import os
import re
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

# Published group-stage calendar (home, away as listed). Source: ESPN schedule.
SCHEDULE = [
    ("2026-06-11", [("Mexico", "South Africa"), ("South Korea", "Czechia")]),
    ("2026-06-12", [("Canada", "Bosnia and Herzegovina"), ("United States", "Paraguay")]),
    ("2026-06-13", [("Qatar", "Switzerland"), ("Brazil", "Morocco"),
                    ("Haiti", "Scotland"), ("Australia", "Turkiye")]),
    ("2026-06-14", [("Germany", "Curacao"), ("Netherlands", "Japan"),
                    ("Ivory Coast", "Ecuador"), ("Sweden", "Tunisia")]),
    ("2026-06-15", [("Spain", "Cape Verde"), ("Belgium", "Egypt"),
                    ("Saudi Arabia", "Uruguay"), ("Iran", "New Zealand")]),
    ("2026-06-16", [("France", "Senegal"), ("Iraq", "Norway"),
                    ("Argentina", "Algeria"), ("Austria", "Jordan")]),
    ("2026-06-17", [("Portugal", "DR Congo"), ("England", "Croatia"),
                    ("Ghana", "Panama"), ("Uzbekistan", "Colombia")]),
    ("2026-06-18", [("Czechia", "South Africa"), ("Switzerland", "Bosnia and Herzegovina"),
                    ("Canada", "Qatar"), ("Mexico", "South Korea")]),
    ("2026-06-19", [("United States", "Australia"), ("Scotland", "Morocco"),
                    ("Brazil", "Haiti"), ("Turkiye", "Paraguay")]),
    ("2026-06-20", [("Netherlands", "Sweden"), ("Germany", "Ivory Coast"),
                    ("Ecuador", "Curacao"), ("Tunisia", "Japan")]),
    ("2026-06-21", [("Spain", "Saudi Arabia"), ("Belgium", "Iran"),
                    ("Uruguay", "Cape Verde"), ("New Zealand", "Egypt")]),
    ("2026-06-22", [("Argentina", "Austria"), ("France", "Iraq"),
                    ("Norway", "Senegal"), ("Jordan", "Algeria")]),
    ("2026-06-23", [("Portugal", "Uzbekistan"), ("England", "Ghana"),
                    ("Panama", "Croatia"), ("Colombia", "DR Congo")]),
    ("2026-06-24", [("Switzerland", "Canada"), ("Bosnia and Herzegovina", "Qatar"),
                    ("Scotland", "Brazil"), ("Morocco", "Haiti"),
                    ("Czechia", "Mexico"), ("South Africa", "South Korea")]),
    ("2026-06-25", [("Ecuador", "Germany"), ("Curacao", "Ivory Coast"),
                    ("Japan", "Sweden"), ("Tunisia", "Netherlands"),
                    ("Turkiye", "United States"), ("Paraguay", "Australia")]),
    ("2026-06-26", [("Norway", "France"), ("Senegal", "Iraq"),
                    ("Cape Verde", "Saudi Arabia"), ("Uruguay", "Spain"),
                    ("Egypt", "Iran"), ("New Zealand", "Belgium")]),
    ("2026-06-27", [("Panama", "England"), ("Croatia", "Ghana"),
                    ("Colombia", "Portugal"), ("DR Congo", "Uzbekistan"),
                    ("Algeria", "Austria"), ("Jordan", "Argentina")]),
]
NAME_OVERRIDE = {"czechia": "czech_republic", "turkiye": "turkey", "curacao": "curacao"}


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    cfg = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    hosts = set(cfg["host_team_ids"])
    group_of = {t: g for g, ms in cfg["groups"].items() for t in ms}

    lookup = {}
    for t in teams:
        for x in [t["canonical_name"], *(t.get("aliases") or [])]:
            if x:
                lookup[x.lower()] = t["team_id"]
                lookup[_slug(x)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(x):
        return NAME_OVERRIDE.get(_slug(x)) or lookup.get(x.lower()) or lookup.get(_slug(x))

    # resolve + validate the schedule
    sched = []
    bad = []
    for d, gms in SCHEDULE:
        for hn, an in gms:
            h, a = resolve(hn), resolve(an)
            if not h or not a:
                bad.append((hn, an))
            sched.append((d, h, a))
    if bad:
        sys.exit(f"FATAL: unresolved schedule teams: {bad}")
    assert len(sched) == 72, f"expected 72 games, got {len(sched)}"

    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for mm in matches:
        if date.fromisoformat(mm["date"]) < cut:
            cnt[mm["home_team_id"]] += 1
            cnt[mm["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= 50} | set(group_of)
    tlist = [t for t in teams if t["team_id"] in eligible]
    feats = build_features(AS_OF, tlist, matches, fifa, [], [], squad_values=squad,
                           xi=HP["xi"], blend_weight=0.7, n_recent=10)
    params = fit_model(AS_OF, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"],
                       theta=HP["theta"], c_v=HP["c_v"])

    def predict(home, away):
        """1X2 + modal score in (home, away) display order; host gets advantage."""
        if home in hosts:
            P = matchup_matrix(params, home, away, True, kappa=0.0)
        elif away in hosts:
            P = matchup_matrix(params, away, home, True, kappa=0.0).T
        else:
            P = matchup_matrix(params, home, away, False, kappa=0.0)
        ph, pd, pa = result_probs(P)
        sx, sy = divmod(int(np.argmax(P)), P.shape[1])
        ov, _ = over_under(P, 2.5)
        return ph, pd, pa, sx, sy, ov, btts(P)

    print("=" * 92)
    print("2026 WORLD CUP — GROUP STAGE BY MATCH DATE — MODEL prediction (NOT market)")
    print("=" * 92)
    print(f"as-of {AS_OF}, rho={params.rho:.4f}. No free 2026 odds (0/72) -> Dixon-Coles model.")
    print("Calendar from published schedule; host advantage = Mexico/Canada/USA in all their games.")
    print("Format: [Grp] Home vs Away  H/D/A%  -> result, modal score (O2.5, BTTS).\n")

    out = []
    cur = None
    for d, h, a in sched:
        if d != cur:
            cur = d
            print(f"\n--- {d} ---")
        ph, pd, pa, sx, sy, ov, bt = predict(h, a)
        ml = max((("home", ph), ("draw", pd), ("away", pa)), key=lambda kv: kv[1])[0]
        res = {"home": nm(h), "draw": "Draw", "away": nm(a)}[ml]
        hostmark = " (host)" if (h in hosts or a in hosts) else ""
        print(f"   [{group_of[h]}] {nm(h)} vs {nm(a)}{hostmark}:  {ph*100:.0f}/{pd*100:.0f}/{pa*100:.0f}  "
              f"-> {res} {sx}-{sy}  (O2.5 {ov*100:.0f}%, BTTS {bt*100:.0f}%)")
        out.append({"date": d, "group": group_of[h], "home": h, "away": a,
                    "p_home": ph, "p_draw": pd, "p_away": pa, "most_likely_result": ml,
                    "most_likely_score": [int(sx), int(sy)], "over_2_5": float(ov), "btts": float(bt)})

    pj = os.path.join(ROOT, "data", "predict_groupstage_by_date_2026.json")
    json.dump({"as_of": AS_OF, "rho": params.rho, "source": "model (no market odds); calendar=published schedule",
               "games": out, "team_names": {t: nm(t) for t in group_of}},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {pj}  ({len(out)} games, by date)")


if __name__ == "__main__":
    main()
