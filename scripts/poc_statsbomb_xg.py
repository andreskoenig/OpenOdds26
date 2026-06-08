"""POC: per-match team xG for recent international tournaments (StatsBomb open data).

After FotMob's stats API turned out to be gated (x-mas header + Cloudflare
Turnstile -> 403 even in a headless browser) and FBref blocks this network's IP,
the clean win is StatsBomb's FREE open-data on GitHub (raw.githubusercontent.com,
reachable here). It carries the gold-standard StatsBomb xG that FBref itself
licenses, with full event data: every Shot event has shot.statsbomb_xg. Summing
per team per match yields exactly our team_xg schema -- no scraping, no anti-bot,
officially open.

Covered senior international tournaments (competition_id / season_id):
  FIFA World Cup 2022   43 / 106
  UEFA Euro 2024        55 / 282
  Copa America 2024    223 / 282
  AFCON 2023          1267 / 107

Output: data/xg_poc.json with team_xg rows {match_id, date, team_id, opponent_id,
xg_for, xg_against} plus a readable table. To wire into the model later, join
these to data/match_results.json on (date, team pair) and feed build_features as
the team_xg input (it blends xG with goals at blend_weight=0.7).

Run:  python scripts/poc_statsbomb_xg.py [comp_id season_id] ...
Default: Euro 2024 (55 282).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; FIFAWC-research/1.0)"}

TOURNAMENTS = {  # friendly name -> (competition_id, season_id)
    "wc2022": (43, 106), "euro2024": (55, 282),
    "copa2024": (223, 282), "afcon2023": (1267, 107),
}


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _gj(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90))


def build_resolver():
    """Map StatsBomb team names -> our canonical team_ids (alias/slug aware)."""
    with open(os.path.join(ROOT, "data", "teams.json"), encoding="utf-8") as f:
        teams = json.load(f)
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]
    # StatsBomb spellings that differ from our canonical names
    extra = {"czech republic": "czechia", "korea republic": "south_korea",
             "ir iran": "iran", "china pr": "china", "usa": "united_states",
             "united states": "united_states", "côte d'ivoire": "ivory_coast",
             "ivory coast": "ivory_coast"}
    for k, v in extra.items():
        lookup.setdefault(k, lookup.get(v, v))

    valid = {t["team_id"] for t in teams}

    def resolve(name):
        return lookup.get(name.lower()) or lookup.get(_slug(name)) or _slug(name)
    return resolve, valid


def match_team_xg(match_id):
    """Sum shot.statsbomb_xg per team for one match -> {team_name: xg}."""
    events = _gj(f"{BASE}/events/{match_id}.json")
    xg = defaultdict(float)
    for e in events:
        if e.get("type", {}).get("name") == "Shot":
            xg[e["team"]["name"]] += e.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
    return xg


def main():
    # parse comp/season args (pairs) or default to Euro 2024
    args = sys.argv[1:]
    pairs = []
    if args:
        for i in range(0, len(args) - 1, 2):
            pairs.append((int(args[i]), int(args[i + 1])))
    if not pairs:
        pairs = [TOURNAMENTS["euro2024"]]

    resolve, valid_ids = build_resolver()
    rows, table, unresolved = [], [], set()

    for comp_id, season_id in pairs:
        matches = _gj(f"{BASE}/matches/{comp_id}/{season_id}.json")
        comp_name = matches[0]["competition"]["competition_name"]
        season = matches[0]["season"]["season_name"]
        print(f"\n=== {comp_name} {season}  ({comp_id}/{season_id}) — {len(matches)} matches ===",
              flush=True)
        for m in sorted(matches, key=lambda x: x["match_date"]):
            mid = m["match_id"]
            hn = m["home_team"]["home_team_name"]
            an = m["away_team"]["away_team_name"]
            try:
                xg = match_team_xg(mid)
            except Exception as e:
                print(f"  [{mid}] events fetch error: {type(e).__name__}")
                continue
            xh = round(xg.get(hn, 0.0), 3)
            xa = round(xg.get(an, 0.0), 3)
            hid, aid = resolve(hn), resolve(an)
            for nm, rid in ((hn, hid), (an, aid)):
                if rid not in valid_ids:   # genuinely unmapped to teams.json
                    unresolved.add(f"{nm} -> {rid}")
            d = m["match_date"]
            rows.append({"match_id": f"sb_{mid}", "date": d, "team_id": hid,
                         "opponent_id": aid, "xg_for": xh, "xg_against": xa})
            rows.append({"match_id": f"sb_{mid}", "date": d, "team_id": aid,
                         "opponent_id": hid, "xg_for": xa, "xg_against": xh})
            table.append((d, hn, m["home_score"], m["away_score"], an, xh, xa))
            print(f"  {d}  {hn[:14]:<14} {m['home_score']}-{m['away_score']} "
                  f"{an[:14]:<14}  xG {xh:.2f}-{xa:.2f}", flush=True)

    out = os.path.join(ROOT, "data", "xg_poc.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "StatsBomb open-data (github statsbomb/open-data)",
                   "competitions": pairs, "n_matches": len(table),
                   "team_xg_rows": rows}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print(f"POC RESULT — {len(table)} matches, {len(rows)} team_xg rows")
    print("=" * 66)
    # quick sanity: tournament xG leaders (avg xG created per match)
    agg = defaultdict(lambda: [0.0, 0])
    for r in rows:
        agg[r["team_id"]][0] += r["xg_for"]
        agg[r["team_id"]][1] += 1
    top = sorted(((tid, v[0] / v[1], v[1]) for tid, v in agg.items()),
                 key=lambda x: -x[1])[:8]
    print("top teams by avg xG created/match:")
    for tid, avg, n in top:
        print(f"  {tid:<16}{avg:>5.2f} xG/match  ({n} games)")
    if unresolved:
        print(f"\nname-resolve fallbacks (slugged): {sorted(unresolved)[:12]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
