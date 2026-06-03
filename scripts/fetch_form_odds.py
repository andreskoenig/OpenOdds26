"""Fetch FORM-WINDOW Bet365 closing 1X2 odds for the surprise factor (SPEC sec.5).

Bounded, cached, resumable, browser-UA, key from .env. Only international matches
in the ~24 months before 2022-11-19 that involve a team in EITHER the validation
set (non-WC, Aug 1 - Nov 19 2022) or the WC2022 field -- enough to compute U/M
as-of 2022-08-01 (validation) and 2022-11-19 (test). Raw decimal odds, single book
(Bet365). Appends to data/match_odds.json (keeps the 64 WC2022 benchmark rows).

build_features skips matches with no odds, so partial coverage degrades
gracefully (teams without odds get U=M=0 = baseline).

Run:  python scripts/fetch_form_odds.py [--plan]      (--plan = network-free sizing)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://api.thestatsapi.com/api/football"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
MIN_INTERVAL = 3.0
MAX_RETRIES = 4
MAX_ODDS_CALLS = 750            # bound on per-match odds requests (covers all WC-involving)
MAX_CONSEC_FAIL = 4            # circuit breaker: abort+save if quota looks exhausted

WINDOW_FROM = "2020-11-19"      # ~24 months before the test cutoff
WINDOW_TO = "2022-11-19"
AS_OF_VAL = "2022-08-01"
MIN_MATCHES = 50

OUTPUT = ROOT / "data" / "match_odds.json"

# Provider competition selection (national-team internationals only).
POS_KW = ["nations league", "world championship qual", "world cup qual", "friendl",
          "euro", "copa america", "copa am", "gold cup", "africa cup", "asian cup",
          "confederations", "arab cup", "finalissima"]
NEG_KW = ["champions league", "libertadores", "sudamericana", "europa", "conference",
          "club", "women", "youth", "futsal", "beach", "u-1", "u-2", "u17", "u19",
          "u20", "u21", "u23", " w ", "(w)"]


def _load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", s.lower())).strip("_")


def _outcome(m):
    return "home" if m["home_goals"] > m["away_goals"] else ("away" if m["home_goals"] < m["away_goals"] else "draw")


def compute_targets():
    """Network-free: the set of our match_results rows that need form-window odds."""
    matches = _load("data/match_results.json")
    teams = _load("data/teams.json")
    cfg = _load("config/tournament_config_2022.json")
    conf_of = {t["team_id"]: t.get("confederation") for t in teams}

    cut_aug = date.fromisoformat(AS_OF_VAL)
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut_aug:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES}

    val_teams = set()
    for m in matches:
        if (m["competition"] != "FIFA World Cup"
                and cut_aug <= date.fromisoformat(m["date"]) <= date.fromisoformat(WINDOW_TO)
                and m["home_team_id"] in eligible and m["away_team_id"] in eligible):
            val_teams.add(m["home_team_id"])
            val_teams.add(m["away_team_id"])
    wc_teams = {t for g in cfg["groups"].values() for t in g}
    relevant = val_teams | wc_teams

    lo, hi = date.fromisoformat(WINDOW_FROM), date.fromisoformat(WINDOW_TO)
    targets = {}
    for m in matches:
        d = date.fromisoformat(m["date"])
        if (m["competition"] != "FIFA World Cup" and lo <= d < hi
                and (m["home_team_id"] in relevant or m["away_team_id"] in relevant)):
            h, a = m["home_team_id"], m["away_team_id"]
            targets[(m["date"], min(h, a), max(h, a))] = m
    return targets, relevant, wc_teams, conf_of


def _conf_bucket(m, relevant, conf_of):
    tid = m["home_team_id"] if m["home_team_id"] in relevant else m["away_team_id"]
    return conf_of.get(tid) or "none"


def plan():
    targets, relevant, wc_teams, conf_of = compute_targets()
    by_conf = Counter(_conf_bucket(m, relevant, conf_of) for m in targets.values())
    wc_involved = sum(1 for m in targets.values()
                      if m["home_team_id"] in wc_teams or m["away_team_id"] in wc_teams)
    print(f"relevant teams: {len(relevant)} (incl. {len(wc_teams)} WC)")
    print(f"form-window target matches ({WINDOW_FROM}..{WINDOW_TO}, non-WC, >=1 relevant): {len(targets)}")
    print(f"  involving a WC team: {wc_involved}")
    print("  by confederation of the relevant team:")
    for c, n in by_conf.most_common():
        print(f"    {c:<10} {n}")
    print(f"\nodds-call bound: {MAX_ODDS_CALLS} (prioritise WC-team + recent matches)")


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------
load_dotenv(ROOT / ".env")
_API_KEY = os.environ.get("STATSAPI_KEY")
_last = 0.0


def _get(path, params=None):
    global _last
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(MAX_RETRIES + 1):
        gap = time.monotonic() - _last
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", "Bearer " + _API_KEY)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _UA)
        _last = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                ra = e.headers.get("Retry-After")
                wait = min(float(ra) if (ra and str(ra).isdigit()) else 20.0 * (attempt + 1), 90.0)
                print(f"    429; backoff {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code} for {url}") from e
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__} for {url}") from e
    raise RuntimeError(f"retries exhausted for {url}")


def build_resolver(teams):
    r = {}
    for t in teams:
        for nm in [t["canonical_name"], *t.get("aliases", [])]:
            r[nm.lower()] = t["team_id"]
            r[_slug(nm)] = t["team_id"]
    r.update({"korea republic": "south_korea", "usa": "united_states",
              "united states": "united_states", "ir iran": "iran",
              "czechia": "czech_republic", "cote d'ivoire": "ivory_coast",
              "côte d'ivoire": "ivory_coast"})
    return r


def resolve(name, r):
    return r.get(name.lower()) or r.get(_slug(name))


def select_competitions():
    comps, page, total = [], 1, 1
    while page <= total:
        d = _get("/competitions", {"per_page": 100, "page": page})
        comps.extend(d.get("data", []))
        total = int((d.get("meta") or {}).get("total_pages", 1) or 1)
        page += 1
    chosen = []
    for c in comps:
        nl = str(c.get("name", "")).lower()
        if any(k in nl for k in POS_KW) and not any(k in nl for k in NEG_KW):
            chosen.append(c)
    return chosen


def extract_closing(mo, outcome):
    v = mo.get(outcome)
    if not isinstance(v, dict):
        return None
    for k in ("last_seen", "opening"):
        x = v.get(k)
        if x not in (None, "", "null"):
            try:
                return float(x)
            except (ValueError, TypeError):
                pass
    return None


def full():
    if not _API_KEY:
        print("FATAL: STATSAPI_KEY not set", file=sys.stderr)
        sys.exit(1)

    targets, relevant, wc_teams, conf_of = compute_targets()
    teams = _load("data/teams.json")
    resolver = build_resolver(teams)

    # Resume: keep existing rows (incl. the 64 WC benchmark); skip done match_ids.
    existing = []
    done = set()
    if OUTPUT.exists():
        try:
            existing = json.loads(OUTPUT.read_text(encoding="utf-8"))
            done = {r["match_id"] for r in existing}
        except (ValueError, OSError):
            existing, done = [], set()
    print(f"existing odds rows: {len(existing)} (resuming)", flush=True)

    comps = select_competitions()
    print(f"selected {len(comps)} international competitions:", flush=True)
    for c in comps:
        print(f"    {c.get('id')}  {c.get('name')}  odds={c.get('odds_available')}", flush=True)

    # Enumerate provider matches in window -> map to our target matches.
    to_fetch = {}  # our_match_id -> (provider_id, our_match)
    for c in comps:
        if c.get("odds_available") is False:
            continue
        page, total = 1, 1
        while page <= total:
            d = _get("/matches", {"competition_id": c["id"], "date_from": WINDOW_FROM,
                                  "date_to": WINDOW_TO, "per_page": 100, "page": page})
            for pm in d.get("data", []):
                ht = resolve((pm.get("home_team") or {}).get("name", ""), resolver)
                at = resolve((pm.get("away_team") or {}).get("name", ""), resolver)
                if not ht or not at:
                    continue
                dt = (pm.get("utc_date") or "")[:10]
                key = (dt, min(ht, at), max(ht, at))
                m = targets.get(key)
                if m and m["match_id"] not in done and m["match_id"] not in to_fetch:
                    to_fetch[m["match_id"]] = (pm.get("id"), m, ht)  # ht = provider home id
            total = int((d.get("meta") or {}).get("total_pages", 1) or 1)
            page += 1
    print(f"matched {len(to_fetch)} target fixtures to provider matches "
          f"(of {len(targets)} targets)", flush=True)

    # Prioritise WC-team and recent matches under the call bound.
    order = sorted(
        to_fetch.items(),
        key=lambda kv: (
            (kv[1][1]["home_team_id"] in wc_teams or kv[1][1]["away_team_id"] in wc_teams),
            kv[1][1]["date"],
        ),
        reverse=True,
    )[:MAX_ODDS_CALLS]

    new_rows = []
    obtained = set()
    consec_fail = 0
    for i, (our_mid, (prov_id, m, prov_home_id)) in enumerate(order, 1):
        try:
            resp = _get(f"/matches/{prov_id}/odds")
            consec_fail = 0
        except RuntimeError as e:
            # Only quota exhaustion (429 retries exhausted) trips the breaker.
            # A 404 just means this match has no odds resource -> skip, no penalty.
            if "exhausted" in str(e):
                consec_fail += 1
                print(f"  [{i}/{len(order)}] {our_mid} QUOTA {e}", flush=True)
                if consec_fail >= MAX_CONSEC_FAIL:
                    print(f"  circuit breaker: {consec_fail} consecutive quota failures "
                          f"-- saving partial and stopping.", flush=True)
                    break
            continue
        bms = (resp.get("data") or {}).get("bookmakers", [])
        b365 = next((b for b in bms if "bet365" in str(b.get("bookmaker", "")).lower()), None)
        if not b365:
            continue
        mo = (b365.get("markets") or {}).get("match_odds")
        if not isinstance(mo, dict):
            continue
        ph, pd, pa = extract_closing(mo, "home"), extract_closing(mo, "draw"), extract_closing(mo, "away")
        if ph is None or pd is None or pa is None:
            continue
        # Orientation: align provider home/away to OUR fixture orientation.
        oh, od, oa = ph, pd, pa
        if prov_home_id == m["away_team_id"]:
            oh, oa = pa, ph
        new_rows.append({"match_id": our_mid, "bookmaker": "Bet365",
                         "odds_home": oh, "odds_draw": od, "odds_away": oa,
                         "captured_at": m["date"]})
        obtained.add(our_mid)
        if i % 25 == 0:
            (OUTPUT).write_text(json.dumps(existing + new_rows, indent=2), encoding="utf-8")
            print(f"  [{i}/{len(order)}] saved checkpoint ({len(new_rows)} new)", flush=True)

    all_rows = existing + new_rows
    OUTPUT.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    # Coverage report by confederation (target vs obtained).
    tgt_conf = Counter(_conf_bucket(m, relevant, conf_of) for m in targets.values())
    got_conf = Counter(_conf_bucket(targets[k], relevant, conf_of)
                       for k in [kk for kk in targets
                                 if targets[kk]["match_id"] in (obtained | done)])
    print("\n--- FORM-WINDOW ODDS COVERAGE (by confederation of relevant team) ---")
    print(f"{'conf':<10}{'target':>8}{'with_odds':>11}")
    for c in sorted(tgt_conf, key=lambda x: -tgt_conf[x]):
        print(f"{c:<10}{tgt_conf[c]:>8}{got_conf.get(c, 0):>11}")
    print(f"\ntotal target {len(targets)} | newly fetched {len(new_rows)} | "
          f"total odds rows now {len(all_rows)} (incl. 64 WC benchmark)")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    if "--max" in sys.argv:
        MAX_ODDS_CALLS = int(sys.argv[sys.argv.index("--max") + 1])
    if "--plan" in sys.argv:
        plan()
    else:
        full()
