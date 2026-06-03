"""PART 1: 2022 World Cup full bracket from cached de-vigged Bet365 closing odds.

w=0: the MARKET 1X2 is the outcome truth; the DC model contributes ONLY scoreline
shape (via the fitted rho). For each of the 64 WC2022 games: de-vig the cached
Bet365 closing odds -> market 1X2 -> market_scoreline -> most-likely scoreline.
Lay out as a bracket; report 1X2 and exact-score accuracy vs the known results.
Knockout 1X2 = 90-minute market result (draws -> extra time); the recorded score
may include extra time. Cached data only, no fetching.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.devig import consensus_probs
from wc_model.features import build_features
from wc_model.market import market_scoreline
from wc_model.model import fit_model

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HP = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0, c_v=0.1)
AS_OF = "2022-11-19"

ROUND_BY_DATE = [  # (round_key, lo, hi)
    ("Group stage", "2022-11-20", "2022-12-02"),
    ("Round of 16", "2022-12-03", "2022-12-06"),
    ("Quarter-finals", "2022-12-09", "2022-12-10"),
    ("Semi-finals", "2022-12-13", "2022-12-14"),
    ("Third place", "2022-12-17", "2022-12-17"),
    ("Final", "2022-12-18", "2022-12-18"),
]


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _outcome(hg, ag):
    return "home" if hg > ag else ("away" if hg < ag else "draw")


def _round_of(d):
    for name, lo, hi in ROUND_BY_DATE:
        if lo <= d <= hi:
            return name
    return "Other"


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    odds = _load("data/match_odds.json")
    cfg = _load("config/tournament_config_2022.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    group_of = {t: g for g, ms in cfg["groups"].items() for t in ms}

    odds_by_match = {}
    for o in odds:
        odds_by_match.setdefault(o["match_id"], []).append(
            (o["odds_home"], o["odds_draw"], o["odds_away"]))

    # Fit the shipped model once just to obtain the fitted rho (scoreline shape).
    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    wc_ids = {t for g in cfg["groups"].values() for t in g}
    eligible = {t for t, c in cnt.items() if c >= 50} | wc_ids
    tlist = [t for t in teams if t["team_id"] in eligible]
    feats = build_features(AS_OF, tlist, matches, fifa, [], [], squad_values=squad,
                           xi=HP["xi"], blend_weight=0.7, n_recent=10)
    params = fit_model(AS_OF, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"],
                       theta=HP["theta"], c_v=HP["c_v"])
    rho = params.rho

    wc = [m for m in matches if m["competition"] == "FIFA World Cup"
          and "2022-11-19" < m["date"] <= "2022-12-31"]

    games = []
    for m in wc:
        books = odds_by_match.get(m["match_id"])
        if not books:
            continue
        ph, pd, pa = consensus_probs(books)
        ms = market_scoreline(ph, pd, pa, rho)
        ml_1x2 = max((("home", ph), ("draw", pd), ("away", pa)), key=lambda kv: kv[1])[0]
        actual = _outcome(m["home_goals"], m["away_goals"])
        games.append({
            "round": _round_of(m["date"]),
            "group": group_of.get(m["home_team_id"]) if _round_of(m["date"]) == "Group stage" else None,
            "date": m["date"], "home": m["home_team_id"], "away": m["away_team_id"],
            "market_1x2": {"home": round(ph, 3), "draw": round(pd, 3), "away": round(pa, 3)},
            "market_most_likely_1x2": ml_1x2,
            "market_most_likely_score": list(ms["most_likely_score"]),
            "actual_score": [m["home_goals"], m["away_goals"]],
            "actual_1x2": actual,
            "hit_1x2": ml_1x2 == actual,
            "hit_score": list(ms["most_likely_score"]) == [m["home_goals"], m["away_goals"]],
        })

    # accuracy
    def acc(subset):
        if not subset:
            return (0, 0, 0.0, 0.0)
        n = len(subset)
        h1 = sum(g["hit_1x2"] for g in subset)
        hs = sum(g["hit_score"] for g in subset)
        return (n, h1, 100 * h1 / n, hs, 100 * hs / n)

    grp = [g for g in games if g["round"] == "Group stage"]
    ko = [g for g in games if g["round"] != "Group stage"]
    n_all, h1_all, p1_all, hs_all, ps_all = acc(games)
    n_g, h1_g, p1_g, hs_g, ps_g = acc(grp)
    n_k, h1_k, p1_k, hs_k, ps_k = acc(ko)

    # ---- print ----
    print("=" * 82)
    print("PART 1 — 2022 WORLD CUP, MARKET-DRIVEN (w=0) BRACKET (cached Bet365 closing)")
    print("=" * 82)
    print(f"outcomes = de-vigged Bet365 1X2 (market truth); scoreline shape = DC model")
    print(f"fitted rho (scoreline shape only) = {rho:.4f} | games with odds: {len(games)}/64\n")

    order = ["Group stage", "Round of 16", "Quarter-finals", "Semi-finals", "Third place", "Final"]
    for rnd in order:
        rgames = [g for g in games if g["round"] == rnd]
        if not rgames:
            continue
        print("-" * 82)
        print(f"{rnd}" + ("  (1X2 = 90-min market; recorded score may include extra time)"
                          if rnd not in ("Group stage",) else ""))
        print("-" * 82)
        if rnd == "Group stage":
            for gl in sorted(set(g["group"] for g in rgames)):
                print(f"  Group {gl}:")
                for g in [x for x in rgames if x["group"] == gl]:
                    _print_game(g, nm)
        else:
            for g in rgames:
                _print_game(g, nm)
        print()

    print("=" * 82)
    print("ACCURACY (market most-likely vs actual)")
    print("=" * 82)
    print(f"  ALL 64        : 1X2 {h1_all}/{n_all} = {p1_all:.0f}%   exact-score {hs_all}/{n_all} = {ps_all:.0f}%")
    print(f"  Group stage   : 1X2 {h1_g}/{n_g} = {p1_g:.0f}%   exact-score {hs_g}/{n_g} = {ps_g:.0f}%")
    print(f"  Knockout (90') : 1X2 {h1_k}/{n_k} = {p1_k:.0f}%   exact-score {hs_k}/{n_k} = {ps_k:.0f}%")
    print("  NOTE: exact-score accuracy is expected to be low (modal score ~1-0/1-1);")
    print("  2022 was upset-heavy, so even the sharp market's most-likely 1X2 misses often.")

    # ---- write ----
    out_json = os.path.join(ROOT, "data", "odds_bracket_2022.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rho": rho, "n_games": len(games),
                   "accuracy": {"all": {"n": n_all, "x12": p1_all, "score": ps_all},
                                "group": {"n": n_g, "x12": p1_g, "score": ps_g},
                                "knockout": {"n": n_k, "x12": p1_k, "score": ps_k}},
                   "games": games}, f, ensure_ascii=False, indent=2)

    out_md = os.path.join(ROOT, "data", "odds_bracket_2022.md")
    _write_md(out_md, rho, games, order, nm,
              (n_all, p1_all, ps_all), (n_g, p1_g, ps_g), (n_k, p1_k, ps_k))
    print(f"\nwrote {out_json}\nwrote {out_md}")


def _print_game(g, nm):
    s = g["market_most_likely_score"]
    a = g["actual_score"]
    mark = "OK " if g["hit_1x2"] else "x  "
    sc = "S" if g["hit_score"] else " "
    print(f"     [{mark}{sc}] {nm(g['home'])} vs {nm(g['away'])}: "
          f"market {g['market_most_likely_1x2']:<4} pred {s[0]}-{s[1]} | "
          f"actual {a[0]}-{a[1]} ({g['actual_1x2']})")


def _write_md(path, rho, games, order, nm, allacc, gacc, kacc):
    L = ["# 2022 World Cup — market-driven bracket (w=0)", "",
         "Outcomes are the **de-vigged Bet365 closing 1X2** (market truth). The "
         "Dixon-Coles model contributes **only scoreline shape** via the fitted "
         f"`rho = {rho:.4f}`. Knockout 1X2 is the 90-minute market; recorded scores "
         "may include extra time.", "",
         f"**Accuracy** — ALL: 1X2 {allacc[1]:.0f}%, exact-score {allacc[2]:.0f}% "
         f"(n={allacc[0]}); Group: 1X2 {gacc[1]:.0f}%, score {gacc[2]:.0f}%; "
         f"Knockout: 1X2 {kacc[1]:.0f}%, score {kacc[2]:.0f}%.", ""]
    for rnd in order:
        rg = [g for g in games if g["round"] == rnd]
        if not rg:
            continue
        L.append(f"## {rnd}")
        if rnd == "Group stage":
            for gl in sorted(set(g["group"] for g in rg)):
                L.append(f"### Group {gl}")
                L.append("| match | market 1X2 | pred score | actual | hit |")
                L.append("|---|---|---|---|---|")
                for g in [x for x in rg if x["group"] == gl]:
                    L.append(_md_row(g, nm))
                L.append("")
        else:
            L.append("| match | market 1X2 | pred score | actual | hit |")
            L.append("|---|---|---|---|---|")
            for g in rg:
                L.append(_md_row(g, nm))
            L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def _md_row(g, nm):
    s, a = g["market_most_likely_score"], g["actual_score"]
    hit = ("1X2" if g["hit_1x2"] else "") + ("+score" if g["hit_score"] else "")
    return (f"| {nm(g['home'])} v {nm(g['away'])} | {g['market_most_likely_1x2']} "
            f"({g['market_1x2']['home']:.2f}/{g['market_1x2']['draw']:.2f}/{g['market_1x2']['away']:.2f}) "
            f"| {s[0]}-{s[1]} | {a[0]}-{a[1]} ({g['actual_1x2']}) | {hit or '-'} |")


if __name__ == "__main__":
    main()
