"""Fetch per-match team xG for the last ~10 years of senior men's international
tournaments (StatsBomb open data) and JOIN to our match_results.

Source: github statsbomb/open-data (free, official). Sum each Shot's
shot.statsbomb_xg per team per match. Then join every StatsBomb match to our
data/match_results.json on (date +/-1, {home_id, away_id}) so each xG row carries
OUR match_id -- exactly what build_features keys team_xg on (match_id, team_id).

Tournaments (senior men's international, past ~10y):
  WC 2018 (43/3), WC 2022 (43/106), Euro 2020 (55/43), Euro 2024 (55/282),
  Copa America 2024 (223/282), AFCON 2023 (1267/107)

Writes data/team_xg.json (rows usable by the model) + prints a join report.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections import defaultdict
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
UA = {"User-Agent": "Mozilla/5.0 (compatible; FIFAWC-research/1.0)"}

COMPS = {  # name -> (competition_id, season_id)
    "WC 2018": (43, 3), "WC 2022": (43, 106),
    "Euro 2020": (55, 43), "Euro 2024": (55, 282),
    "Copa America 2024": (223, 282), "AFCON 2023": (1267, 107),
}


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _gj(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90))


def build_resolver():
    with open(os.path.join(ROOT, "data", "teams.json"), encoding="utf-8") as f:
        teams = json.load(f)
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]
    extra = {"czech republic": "czechia", "korea republic": "south_korea",
             "ir iran": "iran", "china pr": "china", "usa": "united_states",
             "united states": "united_states", "côte d'ivoire": "ivory_coast",
             "ivory coast": "ivory_coast", "north macedonia": "north_macedonia",
             "congo dr": "dr_congo", "congo_dr": "dr_congo", "dr congo": "dr_congo",
             "cape verde islands": "cape_verde", "cape_verde_islands": "cape_verde",
             "cape verde": "cape_verde"}
    valid = {t["team_id"] for t in teams}
    for k, v in extra.items():
        lookup.setdefault(k, v)

    def resolve(name):
        return lookup.get(name.lower()) or lookup.get(_slug(name)) or _slug(name)
    return resolve, valid


def build_match_index():
    """(date, frozenset{home,away}) -> our match_id, for date and +/-1 day."""
    with open(os.path.join(ROOT, "data", "match_results.json"), encoding="utf-8") as f:
        matches = json.load(f)
    idx = {}
    for m in matches:
        idx[(m["date"], frozenset((m["home_team_id"], m["away_team_id"])))] = m["match_id"]
    return idx, matches


def lookup_match(idx, d, pair):
    base = date.fromisoformat(d)
    for delta in (0, -1, 1):
        key = ((base + timedelta(days=delta)).isoformat(), pair)
        if key in idx:
            return idx[key]
    return None


def match_team_xg(match_id):
    events = _gj(f"{BASE}/events/{match_id}.json")
    xg = defaultdict(float)
    for e in events:
        if e.get("type", {}).get("name") == "Shot":
            xg[e["team"]["name"]] += e.get("shot", {}).get("statsbomb_xg", 0.0) or 0.0
    return xg


def main():
    resolve, valid = build_resolver()
    idx, _ = build_match_index()

    rows = []                 # usable team_xg rows (our match_id)
    matched = unmatched = 0
    unmatched_ex = []
    per_comp = {}

    for name, (cid, sid) in COMPS.items():
        ms = _gj(f"{BASE}/matches/{cid}/{sid}.json")
        cm = cu = 0
        for m in sorted(ms, key=lambda x: x["match_date"]):
            mid = m["match_id"]
            hn, an = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
            hid, aid = resolve(hn), resolve(an)
            d = m["match_date"]
            our = lookup_match(idx, d, frozenset((hid, aid)))
            try:
                xg = match_team_xg(mid)
            except Exception as e:
                print(f"  events err {mid}: {type(e).__name__}")
                continue
            xh = round(xg.get(hn, 0.0), 3)
            xa = round(xg.get(an, 0.0), 3)
            if our is None:
                unmatched += 1
                cu += 1
                if len(unmatched_ex) < 12:
                    unmatched_ex.append(f"{d} {hid} vs {aid} ({name})")
                continue
            matched += 1
            cm += 1
            rows.append({"match_id": our, "team_id": hid, "xg_for": xh, "xg_against": xa})
            rows.append({"match_id": our, "team_id": aid, "xg_for": xa, "xg_against": xh})
        per_comp[name] = (cm, cu, len(ms))
        print(f"  {name:<20} matched {cm:>3} / {len(ms):>3}  (unmatched {cu})", flush=True)

    out = os.path.join(ROOT, "data", "team_xg.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "StatsBomb open-data (statsbomb/open-data)",
                   "competitions": list(COMPS.keys()),
                   "n_rows": len(rows), "n_matched_matches": matched,
                   "n_unmatched_matches": unmatched, "rows": rows},
                  f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"JOINED {matched} matches ({len(rows)} team_xg rows); {unmatched} unmatched")
    print("=" * 60)
    for name, (cm, cu, tot) in per_comp.items():
        print(f"  {name:<20} {cm}/{tot} joined")
    if unmatched_ex:
        print("\nunmatched examples (not found in match_results +/-1d):")
        for e in unmatched_ex:
            print("  -", e)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
