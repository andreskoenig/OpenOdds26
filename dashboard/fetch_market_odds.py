"""DASHBOARD market layer: WC2026 closing 1X2 odds via OddsPapi (free tier).

OddsPapi's /v4/historical-odds returns a per-outcome price time-series, so for a
finished game the last snapshot at/before kickoff is the CLOSING line. We capture
bet365 + Pinnacle closing for every played game into data/research/
market_odds_wc2026.json; build_performance.py then scores the books' log-loss /
Brier / 1X2 accuracy against ours.

Played games' historical odds are static, so a game is fetched once and cached
(the daily refresh only spends quota on newly-finished games).

SECURITY: ODDSPAPI_KEY is read from the environment (a GitHub Actions secret in
CI, or .env locally) and sent only as the apiKey query param — never printed,
logged, or committed.

Run:  python dashboard/fetch_market_odds.py   (no-op if ODDSPAPI_KEY is unset)
Stdlib only (python-dotenv used if available, optional).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from dotenv import load_dotenv  # optional; in CI the key is an env secret
except Exception:
    load_dotenv = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
ARCHIVE = os.path.join(DATA, "research", "market_odds_wc2026.json")

BASE = "https://api.oddspapi.io"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
TOURNAMENT_ID = 16                 # OddsPapi "World Cup" (men's); avoids a resolve call
SEASON_FROM, SEASON_TO = "2026-06-11", "2026-07-20"
M_1X2 = "101"
OUT = {"101": "home", "102": "draw", "103": "away"}
WANT_BOOKS = ("bet365", "pinnacle")
FINISHED = {"finished", "ended", "after extra time", "after penalties"}

OVERRIDES = {
    "czechia": "czech_republic", "czech republic": "czech_republic",
    "korea republic": "south_korea", "south korea": "south_korea",
    "ir iran": "iran", "iran": "iran", "turkiye": "turkey", "türkiye": "turkey",
    "turkey": "turkey", "usa": "united_states", "united states": "united_states",
    "ivory coast": "ivory_coast", "cote d'ivoire": "ivory_coast",
    "côte d'ivoire": "ivory_coast", "cape verde": "cape_verde",
    "cabo verde": "cape_verde", "dr congo": "dr_congo", "congo dr": "dr_congo",
    "curacao": "curacao", "curaçao": "curacao",
    "bosnia and herzegovina": "bosnia_and_herzegovina",
}
_n_req = 0


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _get(path, params):
    global _n_req
    q = dict(params)
    q["apiKey"] = os.environ["ODDSPAPI_KEY"]
    url = BASE + path + "?" + urllib.parse.urlencode(q)
    for attempt in range(4):
        _n_req += 1
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < 3:
                time.sleep(3 * (attempt + 1))
                continue
            if e.code == 404:
                return None
            print(f"OddsPapi HTTP {e.code} on {path}: {e.read().decode('utf-8','replace')[:140]}")
            return None
    return None


def build_resolver():
    teams = json.load(open(os.path.join(DATA, "teams.json"), encoding="utf-8"))
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(name):
        low = (name or "").lower().strip()
        return OVERRIDES.get(low) or lookup.get(low) or lookup.get(_slug(name or "")) or _slug(name or "")
    return resolve


def closing(series, kickoff):
    pre = [s for s in series if s.get("createdAt") and s["createdAt"] <= kickoff
           and s.get("active", True) and s.get("price")]
    pick = pre or [s for s in series if s.get("price")]
    return float(pick[-1]["price"]) if pick else None


def fixture_closing_books(fixture_id, kickoff):
    od = _get("/v4/historical-odds", {"fixtureId": fixture_id, "bookmakers": ",".join(WANT_BOOKS)})
    if not od or "bookmakers" not in od:
        return {}
    books = {}
    for bname, bk in od["bookmakers"].items():
        mkt = (bk.get("markets") or {}).get(M_1X2)
        if not mkt:
            continue
        trip = {}
        for oid, side in OUT.items():
            o = (mkt.get("outcomes") or {}).get(oid)
            if not o:
                break
            price = closing(o.get("players", {}).get("0", []), kickoff)
            if not price or price <= 1.0:
                break
            trip[side] = price
        if len(trip) == 3:
            books[bname] = trip
    return books


def main():
    if load_dotenv:
        load_dotenv(os.path.join(ROOT, ".env"))
    if not os.environ.get("ODDSPAPI_KEY"):
        print("ODDSPAPI_KEY unset — skipping market odds fetch (dashboard market stats "
              "use the existing archive).")
        return
    resolve = build_resolver()
    fixtures = _get("/v4/fixtures", {"tournamentId": TOURNAMENT_ID,
                                     "from": SEASON_FROM, "to": SEASON_TO})
    fixtures = fixtures if isinstance(fixtures, list) else (fixtures or {}).get("data", [])

    archive = {}
    if os.path.exists(ARCHIVE):
        archive = {e["fixture_id"]: e for e in json.load(open(ARCHIVE, encoding="utf-8")).get("fixtures", [])}

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_new = n_skip = 0
    for f in fixtures or []:
        if f.get("statusName", "").strip().lower() not in FINISHED:
            continue
        fid = f["fixtureId"]
        if fid in archive and archive[fid].get("books"):
            n_skip += 1
            continue
        kickoff = f.get("trueStartTime") or f.get("startTime")
        books = fixture_closing_books(fid, kickoff)
        if not books:
            continue
        archive[fid] = {
            "fixture_id": fid, "kickoff": (kickoff or "")[:10], "captured_at": now,
            "home_id": resolve(f.get("participant1Name")), "away_id": resolve(f.get("participant2Name")),
            "home_name": f.get("participant1Name"), "away_name": f.get("participant2Name"),
            "books": {b.capitalize(): t for b, t in books.items()},
        }
        n_new += 1

    out = {"generated_at": now, "tournament_id": TOURNAMENT_ID,
           "source": "OddsPapi /v4/historical-odds (closing = last snapshot <= kickoff)",
           "fixtures": sorted(archive.values(), key=lambda e: e["kickoff"])}
    os.makedirs(os.path.dirname(ARCHIVE), exist_ok=True)
    json.dump(out, open(ARCHIVE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"market odds: {_n_req} requests | archive {len(archive)} games ({n_new} new, {n_skip} cached)")


if __name__ == "__main__":
    main()
