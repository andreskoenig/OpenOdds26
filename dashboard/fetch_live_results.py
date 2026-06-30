"""DASHBOARD-ONLY fast results layer (ESPN, no API key).

The model pipeline sources results from martj42 (GitHub), which lags ~a day.
For the live dashboard we additionally pull finished World Cup results from
ESPN's public scoreboard API (no key), so the scoreboard updates within minutes
of full-time instead of waiting on martj42. This NEVER feeds the model — it only
writes dashboard/live_results.json, which build_performance.py merges in as a
gap-filler (martj42 stays authoritative when it has the same game).

Stdlib only. Writes dashboard/live_results.json.

Run:  python dashboard/fetch_live_results.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
UA = {"User-Agent": "Mozilla/5.0 (compatible; OpenOdds-dashboard/1.0)"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={}"
START = dt.date(2026, 6, 11)
WC_COMPETITION = "FIFA World Cup"

# ESPN spellings that don't slug-match our team_ids
OVERRIDES = {
    "czechia": "czech_republic", "czech republic": "czech_republic",
    "bosnia-herzegovina": "bosnia_and_herzegovina",
    "bosnia and herzegovina": "bosnia_and_herzegovina",
    "ir iran": "iran", "iran": "iran", "korea republic": "south_korea",
    "south korea": "south_korea", "turkiye": "turkey", "türkiye": "turkey",
    "usa": "united_states", "united states": "united_states",
    "cape verde islands": "cape_verde", "cabo verde": "cape_verde",
    "dr congo": "dr_congo", "congo dr": "dr_congo", "curacao": "curacao",
    "curaçao": "curacao", "cote d'ivoire": "ivory_coast",
    "côte d'ivoire": "ivory_coast", "ivory coast": "ivory_coast",
}


def slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def build_resolver():
    with open(os.path.join(DATA, "teams.json"), encoding="utf-8") as f:
        teams = json.load(f)
    lookup, valid = {}, {t["team_id"] for t in teams}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(name):
        low = name.lower().strip()
        if low in OVERRIDES:
            return OVERRIDES[low]
        return lookup.get(low) or lookup.get(slug(name)) or slug(name)
    return resolve, valid


def fetch_day(date_str):
    url = ESPN.format(date_str)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def main():
    resolve, valid = build_resolver()
    today = dt.date.today()
    end = max(today, START)
    records, unresolved = [], set()
    d = START
    while d <= end:
        ds = d.strftime("%Y%m%d")
        try:
            data = fetch_day(ds)
        except Exception as e:
            print(f"  {d} fetch error: {type(e).__name__}")
            d += dt.timedelta(days=1)
            continue
        for ev in data.get("events", []):
            comp = ev["competitions"][0]
            status = ev["status"]["type"]["name"]
            # Accept any FINISHED status. Knockout ties decided in extra time
            # (STATUS_FINAL_AET) or penalties (STATUS_FINAL_PEN) are NOT
            # STATUS_FULL_TIME — without these, ET/pen knockouts get dropped.
            if status not in ("STATUS_FULL_TIME", "STATUS_FINAL_AET",
                              "STATUS_FINAL_PEN", "STATUS_FINAL"):
                continue
            cs = comp["competitors"]
            try:
                h = next(c for c in cs if c["homeAway"] == "home")
                a = next(c for c in cs if c["homeAway"] == "away")
            except StopIteration:
                continue
            hid = resolve(h["team"]["displayName"])
            aid = resolve(a["team"]["displayName"])
            for nm, tid in ((h["team"]["displayName"], hid), (a["team"]["displayName"], aid)):
                if tid not in valid:
                    unresolved.add(f"{nm} -> {tid}")
            # who advanced (ESPN winner flag) — decides penalty ties our goal data can't
            adv = None
            if h.get("winner"):
                adv = hid
            elif a.get("winner"):
                adv = aid
            records.append({
                "date": d.isoformat(),
                "home_team_id": hid, "away_team_id": aid,
                "home_goals": int(h["score"]), "away_goals": int(a["score"]),
                "competition": WC_COMPETITION, "source": "espn",
                "status": status, "advancer": adv,
            })
        d += dt.timedelta(days=1)

    out = {"generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "source": "ESPN public scoreboard (fifa.world), finished matches only",
           "results": records}
    path = os.path.join(HERE, "live_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"live_results.json: {len(records)} finished WC matches "
          f"({START} .. {end})")
    for r in records:
        print(f"  {r['date']} {r['home_team_id']} {r['home_goals']}-{r['away_goals']} {r['away_team_id']}")
    if unresolved:
        print(f"UNRESOLVED names (check OVERRIDES): {sorted(unresolved)}")


if __name__ == "__main__":
    main()
