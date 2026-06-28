"""Knockout-stage predictions (2026 World Cup) — single P(advance) per tie.

Once the group stage is complete, the R32 bracket is fixed. This resolves the
config knockout_bracket slots (1A, 2B, T:... thirds) against the REAL final
standings, then for each R32 tie computes ONE number: P(each team advances),
with extra time + penalties folded in (mirrors simulate.py):

    P(A adv) = P(A win 90)
             + P(draw 90) * [ P(A win ET) + P(ET draw) * (0.5 + pen_tilt) ]

  ET: independent Poisson with rate = (marginal E[goals in 90]) * 30/90
  penalties: ~coin flip with a small rating tilt, clipped to +/-0.15

There is no separate "120-min 1X2" — the 90-min 1X2 (regulation, 3-way) is kept
only as `p_1x2_90` for the bookmaker comparison; the bracket prediction is
P(advance). Later rounds (R16+) are emitted as TBD slots (teams depend on R32).

Writes data/knockout_2026.json. Use --live to fit as-of today (default).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import date, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.features import build_features
from wc_model.model import fit_model, matchup_matrix, result_probs
from wc_model.simulate import _allocate_thirds, _third_slot_specs, expected_goals
from wc_model.pipeline import team_strength

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HP = dict(xi=0.0012651, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0,
          c_v=0.1, c_m=0.35, opponent_adjust=True, max_history_years=10.0)
ET_SCALE = 30.0 / 90.0
PEN_TILT = 0.05
PEN_CLIP = 0.15
ROUND_LABELS = [("round_of_32", "R32"), ("round_of_16", "R16"),
                ("quarter_finals", "QF"), ("semi_finals", "SF"), ("final", "Final")]


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


# ESPN spellings -> our team_ids (knockout fixtures use the official bracket, so we
# align the third-place ties to reality rather than our allocation approximation).
_ESPN_OVERRIDES = {
    "czechia": "czech_republic", "korea republic": "south_korea", "south korea": "south_korea",
    "ir iran": "iran", "turkiye": "turkey", "türkiye": "turkey", "usa": "united_states",
    "cote d'ivoire": "ivory_coast", "côte d'ivoire": "ivory_coast", "ivory coast": "ivory_coast",
    "cape verde": "cape_verde", "cabo verde": "cape_verde", "dr congo": "dr_congo",
    "congo dr": "dr_congo", "curacao": "curacao", "curaçao": "curacao",
    "bosnia-herzegovina": "bosnia_and_herzegovina", "bosnia and herzegovina": "bosnia_and_herzegovina",
}
_ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={}"


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def fetch_espn_r32(teams):
    """Official R32 from ESPN -> {team_id: (opponent_id, date)}. Empty dict on failure."""
    lookup = {}
    for t in teams:
        for x in [t["canonical_name"], *(t.get("aliases") or [])]:
            if x:
                lookup[x.lower()] = t["team_id"]; lookup[_slug(x)] = t["team_id"]
    def resolve(n):
        return _ESPN_OVERRIDES.get(n.lower().strip()) or lookup.get(n.lower()) or lookup.get(_slug(n))
    out = {}
    d = date(2026, 6, 28)
    while d <= date(2026, 7, 4):
        try:
            data = json.loads(urllib.request.urlopen(
                urllib.request.Request(_ESPN_URL.format(d.strftime("%Y%m%d")),
                                       headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())
        except Exception:
            d += timedelta(days=1); continue
        for ev in data.get("events", []):
            cs = ev["competitions"][0]["competitors"]
            try:
                h = resolve(next(c for c in cs if c["homeAway"] == "home")["team"]["displayName"])
                a = resolve(next(c for c in cs if c["homeAway"] == "away")["team"]["displayName"])
            except StopIteration:
                continue
            if h and a:
                out[h] = (a, d.isoformat()); out[a] = (h, d.isoformat())
        d += timedelta(days=1)
    return out


def _pois(lam, K=15):
    return np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(K + 1)])


def et_outcome(lam_h, lam_a):
    """(P home wins ET, P ET draw, P away wins ET) from independent Poisson goals."""
    M = np.outer(_pois(lam_h), _pois(lam_a))   # M[m, n] = P(home m, away n)
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


def real_standings(group, by_pair):
    """Final group table from real results -> (ranked team_ids, stats)."""
    st = {t: {"pts": 0, "gd": 0, "gf": 0} for t in group}
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            a, b = group[i], group[j]
            r = by_pair.get(frozenset((a, b)))
            if not r:
                continue
            hid, hg, ag = r
            ga, gb = (hg, ag) if hid == a else (ag, hg)
            st[a]["gf"] += ga; st[b]["gf"] += gb
            st[a]["gd"] += ga - gb; st[b]["gd"] += gb - ga
            if ga > gb:
                st[a]["pts"] += 3
            elif gb > ga:
                st[b]["pts"] += 3
            else:
                st[a]["pts"] += 1; st[b]["pts"] += 1
    ranked = sorted(group, key=lambda t: (-st[t]["pts"], -st[t]["gd"], -st[t]["gf"], group.index(t)))
    return ranked, st


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    market = _load("data/polymarket_winner_2026_depathed.json")["p_market"]
    cfg = _load("config/tournament_config_2026.json")
    kb = cfg["knockout_bracket"]
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    hosts = set(cfg["host_team_ids"])
    groups = cfg["groups"]
    group_of = {t: g for g, ms in groups.items() for t in ms}

    # real group results (martj42 + ESPN gap-filler), oriented (home_id, hg, ag)
    by_pair = {}
    def add(r):
        if r.get("competition") != "FIFA World Cup" or not ("2026-06-11" <= r.get("date", "") <= "2026-06-27"):
            return
        h, a, hg, ag = r.get("home_team_id"), r.get("away_team_id"), r.get("home_goals"), r.get("away_goals")
        if h and a and hg is not None and group_of.get(h) == group_of.get(a):
            by_pair.setdefault(frozenset((h, a)), (h, int(hg), int(ag)))
    for r in matches:
        add(r)
    lp = os.path.join(ROOT, "dashboard", "live_results.json")
    if os.path.exists(lp):
        for r in _load("dashboard/live_results.json")["results"]:
            add(r)

    # final standings + best-thirds allocation
    ranked = {g: real_standings(groups[g], by_pair)[0] for g in groups}
    stats = {g: real_standings(groups[g], by_pair)[1] for g in groups}
    advance_per = int(cfg.get("advance_per_group", 2))
    thirds = sorted(((ranked[g][advance_per], stats[g][ranked[g][advance_per]], g) for g in groups),
                    key=lambda c: (-c[1]["pts"], -c[1]["gd"], -c[1]["gf"]))
    best_third_groups = [c[2] for c in thirds[:int(cfg.get("best_thirds", 8))]]
    alloc = _allocate_thirds(best_third_groups, _third_slot_specs(kb))
    third_team = {sid: ranked[g][advance_per] for sid, g in alloc.items()}

    def resolve(slot):
        if slot.startswith("T:"):
            return third_team.get(slot)
        pos, g = int(slot[0]), slot[1:]
        return ranked[g][pos - 1]

    # fit model as-of today (same config as the live forecast)
    as_of = date.today().isoformat()
    cut = date.fromisoformat(as_of)
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1; cnt[m["away_team_id"]] += 1
    elig = {t for t, c in cnt.items() if c >= 50} | set(group_of)
    tlist = [t for t in teams if t["team_id"] in elig]
    feats = build_features(as_of, tlist, matches, fifa, [], [], squad_values=squad,
                           market_probs=market, xi=HP["xi"], blend_weight=0.7, n_recent=10,
                           opponent_adjust=HP["opponent_adjust"], max_history_years=HP["max_history_years"])
    params = fit_model(as_of, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"], theta=HP["theta"],
                       c_v=HP["c_v"], c_m=HP["c_m"], max_history_years=HP["max_history_years"])

    def tie(a, b):
        """One knockout tie a vs b. Returns P(advance), the 90' and 120' 1X2,
        expected goals, and the modal 90' scoreline (all in a/home vs b/away order).
        The 120' 1X2 is the single full-time result: a wins by 120 / level after
        120 (-> penalties) / b wins by 120."""
        if a in hosts and b not in hosts:
            P = matchup_matrix(params, a, b, True, kappa=0.0)
        elif b in hosts and a not in hosts:
            P = matchup_matrix(params, b, a, True, kappa=0.0).T
        else:
            P = matchup_matrix(params, a, b, False, kappa=0.0)
        ph, pd, pa = result_probs(P)                       # 90-min 1X2 (a, draw, b)
        eg_a, eg_b = expected_goals(P)
        et_a, et_d, et_b = et_outcome(eg_a * ET_SCALE, eg_b * ET_SCALE)
        tilt = float(np.clip(PEN_TILT * (team_strength(params, a) - team_strength(params, b)), -PEN_CLIP, PEN_CLIP))
        p_a_adv = ph + pd * (et_a + et_d * (0.5 + tilt))
        h120, d120, a120 = ph + pd * et_a, pd * et_d, pa + pd * et_b   # result by end of ET
        sx, sy = divmod(int(np.argmax(P)), P.shape[1])     # modal 90' score (a-b)
        return {"p_a_adv": float(p_a_adv), "p_b_adv": float(1 - p_a_adv),
                "x90": [round(float(ph), 4), round(float(pd), 4), round(float(pa), 4)],
                "x120": [round(float(h120), 4), round(float(d120), 4), round(float(a120), 4)],
                "xg": [round(float(eg_a), 2), round(float(eg_b), 2)],
                "modal": [int(sx), int(sy)], "modal_p": round(float(P[sx, sy]), 4)}

    # R32: resolve teams, then align to ESPN's official bracket (fixes the
    # third-place allocation, which our stand-in doesn't match exactly). Each tie
    # has a determinate (non-third) team as anchor; ESPN gives its real opponent.
    espn = fetch_espn_r32(teams)
    n_fixed = 0
    r32 = []
    for m in kb["round_of_32"]:
        a, b = resolve(m["home"]), resolve(m["away"])
        date_str = None
        anchor = a if not m["home"].startswith("T:") else b
        if anchor in espn:
            real_opp, date_str = espn[anchor]
            other = b if anchor == a else a
            if real_opp != other:
                n_fixed += 1
            a, b = anchor, real_opp
        t = tie(a, b)
        pa_adv, pb_adv = t["p_a_adv"], t["p_b_adv"]
        r32.append({"match": m["match"], "home": a, "away": b, "date": date_str,
                    "home_name": nm(a), "away_name": nm(b),
                    "p_home_adv": round(pa_adv, 4), "p_away_adv": round(pb_adv, 4),
                    "fav": a if pa_adv >= pb_adv else b, "fav_name": nm(a if pa_adv >= pb_adv else b),
                    "fav_p": round(max(pa_adv, pb_adv), 4),
                    "p_1x2_90": t["x90"], "p_1x2_120": t["x120"], "xg": t["xg"],
                    "modal": t["modal"], "modal_p": t["modal_p"]})

    # later rounds: TBD slots (teams unknown until R32 plays) — keep the tree refs
    later = {}
    for key, lab in ROUND_LABELS[1:]:
        ms = kb.get(key, [])
        ms = ms if isinstance(ms, list) else [ms]
        later[lab] = [{"match": m["match"], "home_ref": m["home"], "away_ref": m["away"]} for m in ms]
    tp = kb.get("third_place")
    if tp:
        later["third_place"] = [{"match": tp["match"], "home_ref": tp["home"], "away_ref": tp["away"]}]

    out = {"as_of": as_of, "source": "model knockout P(advance) (ET + penalties folded in)",
           "rounds": {"R32": r32, **later},
           "team_names": {t: nm(t) for t in group_of}}
    pj = os.path.join(ROOT, "data", "knockout_2026.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"as-of {as_of} | R32 resolved ({len(r32)} ties) | {n_fixed} third-place ties aligned to ESPN\n")
    for t in r32:
        print(f"  [{t['match']}] {t['home_name']:>16} vs {t['away_name']:<16}  "
              f"{t['fav_name']} adv {t['fav_p']*100:.0f}%  (90' 1X2 "
              f"{t['p_1x2_90'][0]*100:.0f}/{t['p_1x2_90'][1]*100:.0f}/{t['p_1x2_90'][2]*100:.0f})")
    print(f"\nwrote {pj}")


if __name__ == "__main__":
    main()
