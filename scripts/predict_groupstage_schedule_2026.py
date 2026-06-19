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

HP = dict(xi=0.0012651, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0,
          c_v=0.1, c_m=0.35, opponent_adjust=True, max_history_years=10.0)
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


def _played_pairs():
    """Set of frozenset({home_id, away_id}) for WC2026 group games already played.

    Sourced from the model's results (martj42) plus the dashboard ESPN layer, so a
    game counts as played the moment either source has a final score. Group pairs
    are unique (each pair meets once), so the unordered pair identifies the game.
    """
    pairs = set()
    for rel in ("data/match_results.json", "dashboard/live_results.json"):
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            continue
        data = json.load(open(p, encoding="utf-8"))
        recs = data if isinstance(data, list) else data.get("results", [])
        for r in recs:
            if r.get("competition") != "FIFA World Cup" or r.get("date", "") < "2026-06-11":
                continue
            h, a = r.get("home_team_id"), r.get("away_team_id")
            if h and a and r.get("home_goals") is not None:
                pairs.add(frozenset((h, a)))
    return pairs


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    market = _load("data/polymarket_winner_2026_depathed.json")["p_market"]
    cfg = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    hosts = set(cfg["host_team_ids"])
    group_of = {t: g for g, ms in cfg["groups"].items() for t in ms}

    # --live: walk-forward update. Advance the cutoff to today so UNPLAYED games
    # get post-MD1 predictions; already-played games are kept frozen (no
    # look-ahead, their scoring is untouched). Default: full pre-tournament regen.
    live = "--live" in sys.argv
    as_of = date.today().isoformat() if live else AS_OF

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

    cut = date.fromisoformat(as_of)
    cnt = Counter()
    for mm in matches:
        if date.fromisoformat(mm["date"]) < cut:
            cnt[mm["home_team_id"]] += 1
            cnt[mm["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= 50} | set(group_of)
    tlist = [t for t in teams if t["team_id"] in eligible]
    feats = build_features(as_of, tlist, matches, fifa, [], [], squad_values=squad,
                           market_probs=market, xi=HP["xi"], blend_weight=0.7, n_recent=10,
                           opponent_adjust=HP["opponent_adjust"],
                           max_history_years=HP["max_history_years"])
    params = fit_model(as_of, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"],
                       theta=HP["theta"], c_v=HP["c_v"], c_m=HP["c_m"],
                       max_history_years=HP["max_history_years"])

    def predict(home, away):
        """1X2, expected goals + top-3 scorelines in (home, away) display order.

        Expected goals = marginal means of the score matrix (E[home], E[away]);
        top-3 = the three most-likely exact scorelines with their probabilities.
        Host gets the home advantage regardless of nominal listing.
        """
        if home in hosts:
            P = matchup_matrix(params, home, away, True, kappa=0.0)
        elif away in hosts:
            P = matchup_matrix(params, away, home, True, kappa=0.0).T
        else:
            P = matchup_matrix(params, home, away, False, kappa=0.0)
        ph, pd, pa = result_probs(P)
        sx, sy = divmod(int(np.argmax(P)), P.shape[1])
        ov, _ = over_under(P, 2.5)
        n, m = P.shape
        exh = float(sum(i * P[i, :].sum() for i in range(n)))   # E[home goals]
        exa = float(sum(j * P[:, j].sum() for j in range(m)))   # E[away goals]
        cells = sorted(((float(P[i, j]), i, j) for i in range(n) for j in range(m)),
                       reverse=True)[:3]
        top3 = [[i, j, pr] for pr, i, j in cells]               # [home, away, prob]
        return ph, pd, pa, sx, sy, ov, btts(P), exh, exa, top3

    print("=" * 92)
    print("2026 WORLD CUP — GROUP STAGE BY MATCH DATE — MODEL prediction (NOT market)")
    print("=" * 92)
    print(f"as-of {as_of}{' [LIVE walk-forward]' if live else ''}, rho={params.rho:.4f}. "
          f"No free 2026 odds (0/72) -> Dixon-Coles model.")
    print("Calendar from published schedule; host advantage = Mexico/Canada/USA in all their games.")
    print("Format: [Grp] Home vs Away  H/D/A%  -> result, modal score (O2.5, BTTS).\n")

    out = []
    cur = None
    for d, h, a in sched:
        if d != cur:
            cur = d
            print(f"\n--- {d} ---")
        ph, pd, pa, sx, sy, ov, bt, exh, exa, top3 = predict(h, a)
        ml = max((("home", ph), ("draw", pd), ("away", pa)), key=lambda kv: kv[1])[0]
        res = {"home": nm(h), "draw": "Draw", "away": nm(a)}[ml]
        hostmark = " (host)" if (h in hosts or a in hosts) else ""
        t3 = ", ".join(f"{i}-{j} {pr*100:.0f}%" for i, j, pr in top3)
        print(f"   [{group_of[h]}] {nm(h)} vs {nm(a)}{hostmark}:  {ph*100:.0f}/{pd*100:.0f}/{pa*100:.0f}  "
              f"-> {res}, E[goals] {exh:.2f}-{exa:.2f}  [{t3}]  (O2.5 {ov*100:.0f}%, BTTS {bt*100:.0f}%)")
        out.append({"date": d, "group": group_of[h], "home": h, "away": a,
                    "p_home": ph, "p_draw": pd, "p_away": pa, "most_likely_result": ml,
                    "most_likely_score": [int(sx), int(sy)],
                    "exp_goals_home": exh, "exp_goals_away": exa, "top3_scores": top3,
                    "over_2_5": float(ov), "btts": float(bt), "pred_as_of": as_of})

    pj = os.path.join(ROOT, "data", "predict_groupstage_by_date_2026.json")

    # --live: keep already-played games frozen at their original prediction; only
    # the upcoming games carry the fresh as_of. Each game records its own
    # pred_as_of so the dashboard can show when each matchday was last predicted.
    if live and os.path.exists(pj):
        played = _played_pairs()
        oldj = json.load(open(pj, encoding="utf-8"))
        old_as_of = oldj.get("as_of", AS_OF)
        old = {frozenset((g["home"], g["away"])): g for g in oldj.get("games", [])}
        merged, n_frozen, n_updated = [], 0, 0
        for g in out:
            key = frozenset((g["home"], g["away"]))
            if key in played and key in old:
                e = dict(old[key])
                e.setdefault("pred_as_of", old_as_of)
                merged.append(e)
                n_frozen += 1
            else:
                merged.append(g)
                n_updated += 1
        out = merged
        print(f"\nLIVE merge: {n_frozen} played games kept frozen, "
              f"{n_updated} upcoming games re-predicted as-of {as_of}")

    json.dump({"as_of": as_of, "rho": params.rho,
               "source": "model (no market odds); calendar=published schedule"
                         + ("; LIVE walk-forward (played games frozen)" if live else ""),
               "games": out, "team_names": {t: nm(t) for t in group_of}},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {pj}  ({len(out)} games, by date)")


if __name__ == "__main__":
    main()
