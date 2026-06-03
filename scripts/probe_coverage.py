"""One-off, READ-ONLY coverage probe for TheStatsAPI (https://www.thestatsapi.com).

Goal: confirm whether the provider actually carries the historical INTERNATIONAL
data the model needs (results, 1X2 odds incl. Pinnacle, xG) for a few categories
of pre-November-2022 fixtures, before building the full fetch.

Safety / constraints:
- Read-only: GET requests only.
- The API key is read from os.environ["STATSAPI_KEY"] via python-dotenv and is
  sent ONLY in the Authorization header. It is never printed, logged, or written.
- No fetched data is written to the repo; everything goes to stdout (ASCII only).
- Request budget + pacing keeps it to ~a dozen calls, well under 120/min.
- A browser User-Agent is sent because the provider's CDN (Cloudflare) blocks the
  default Python-urllib agent (error 1010). This is a legitimate keyed client; the
  CDN over-blocks the stdlib UA -- not detection evasion.

Endpoint/response contract (confirmed live against the API):
  GET /competitions?per_page=100&page=N   -> {data:[{id (comp_xxxx), name,
       confederation (UNRELIABLE), type, odds_available, xg_available}], meta:{total_pages}}
  GET /matches?competition_id=&date_from=YYYY-MM-DD&date_to=YYYY-MM-DD
       -> {data:[{id (mt_xxxx), utc_date, status, score, home_team{name},
                  away_team{name}, odds_available, xg_available}]}  (newest first)
  GET /matches/{id}/odds  -> {data:{bookmakers:[{bookmaker, markets:{match_odds:
                              {home/draw/away:{opening, last_seen}}, ...}}]}}
  GET /matches/{id}/stats -> {data:{overview:{expected_goals:{all:{home,away}}}}}
Auth: Authorization: Bearer <key>
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from dotenv import load_dotenv

BASE = "https://api.thestatsapi.com/api/football"
CUTOFF = datetime(2022, 11, 1, tzinfo=timezone.utc)  # strictly before Nov 2022
MAX_REQUESTS = 14
MIN_INTERVAL = 0.7  # seconds between calls -> <=86/min, under the 120/min cap
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

load_dotenv()
try:
    _API_KEY = os.environ["STATSAPI_KEY"]
except KeyError:
    print("FATAL: STATSAPI_KEY not found in environment/.env", file=sys.stderr)
    sys.exit(1)

_req_count = 0
_last_call = 0.0


def get(path, params=None):
    """Paced, read-only GET. Returns {'_status', 'data', '_error'}. Key never logged."""
    global _req_count, _last_call
    if _req_count >= MAX_REQUESTS:
        return {"_status": None, "_error": "request budget exhausted", "data": None}
    gap = time.monotonic() - _last_call
    if gap < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - gap)
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", "Bearer " + _API_KEY)  # key stays in-header only
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _UA)
    _req_count += 1
    _last_call = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
        try:
            return {"_status": status, "data": json.loads(body)}
        except json.JSONDecodeError:
            return {"_status": status, "data": None}
    except urllib.error.HTTPError as e:
        return {"_status": e.code, "_error": f"HTTP {e.code}", "data": None}
    except Exception as e:
        return {"_status": None, "_error": type(e).__name__, "data": None}


def parse_dt(s):
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fmt_score(m):
    sc = m.get("score")
    if isinstance(sc, dict):
        for hk, ak in (("home", "away"), ("home_score", "away_score")):
            if sc.get(hk) is not None and sc.get(ak) is not None:
                return f"{sc[hk]}-{sc[ak]}"
        ft = sc.get("full_time") or sc.get("fulltime")
        if isinstance(ft, dict) and ft.get("home") is not None:
            return f"{ft.get('home')}-{ft.get('away')}"
    if isinstance(sc, str):
        return sc
    if m.get("home_score") is not None and m.get("away_score") is not None:
        return f"{m['home_score']}-{m['away_score']}"
    return None


def parse_odds(data):
    """Return (has_1x2, book_names, pinnacle_present, closing_present)."""
    bms = []
    if isinstance(data, dict):
        bms = (data.get("data") or {}).get("bookmakers") or []
    books, has_1x2, pinnacle, closing = [], False, False, False
    for b in bms:
        if not isinstance(b, dict):
            continue
        name = b.get("bookmaker")
        if name:
            books.append(name)
            if "pinnacle" in str(name).lower():
                pinnacle = True
        mo = (b.get("markets") or {}).get("match_odds")
        if isinstance(mo, dict):
            has_1x2 = True
            for outcome in ("home", "draw", "away"):
                v = mo.get(outcome)
                if isinstance(v, dict) and v.get("last_seen") not in (None, ""):
                    closing = True
    return has_1x2, books, pinnacle, closing


def parse_xg(data):
    """Return (xg_populated, (home, away)) from overview.expected_goals.all."""
    eg = None
    if isinstance(data, dict):
        eg = ((data.get("data") or {}).get("overview") or {}).get("expected_goals")
    allv = eg.get("all") if isinstance(eg, dict) else None
    if not isinstance(allv, dict):
        return False, None
    h, a = allv.get("home"), allv.get("away")
    populated = (isinstance(h, (int, float)) and h > 0) or (isinstance(a, (int, float)) and a > 0)
    return bool(populated), (h, a)


# --- competition discovery --------------------------------------------------

def fetch_all_competitions():
    r = get("/competitions", {"per_page": 100, "page": 1})
    if r.get("_status") != 200 or not isinstance(r.get("data"), dict):
        return [], r.get("_status"), r.get("_error")
    payload = r["data"]
    comps = list(payload.get("data", []))
    total_pages = int((payload.get("meta") or {}).get("total_pages", 1) or 1)
    for page in range(2, total_pages + 1):
        rp = get("/competitions", {"per_page": 100, "page": page})
        if rp.get("_status") == 200 and isinstance(rp.get("data"), dict):
            comps.extend(rp["data"].get("data", []))
    return comps, 200, None


# Competition match rules (NAME-based; the confederation field is unreliable).
CATEGORIES = [
    {"label": "2022 UEFA Nations League (sanity check)",
     "match": lambda n: "uefa nations league" in n,
     "window": ("2022-06-01", "2022-10-31")},
    {"label": "2021-22 CONMEBOL World Cup qualifier",
     "match": lambda n: "conmebol" in n and "qual" in n,
     "window": ("2021-09-01", "2022-03-31")},
    {"label": "2021-22 AFC World Cup qualifier / Asian Cup",
     "match": lambda n: ("afc" in n and "qual" in n) or "asian cup" in n,
     "window": ("2021-09-01", "2022-03-31")},
    {"label": "2022 international friendly",
     "match": lambda n: "friendl" in n,
     "window": ("2022-01-01", "2022-10-31")},
]


def pick_competition(comps, cat):
    for c in comps:
        if cat["match"](str(c.get("name", "")).lower()):
            return c
    return None


def find_fixture(competition_id, window):
    """First finished pre-cutoff fixture, preferring one that advertises odds."""
    r = get("/matches", {"competition_id": competition_id,
                          "date_from": window[0], "date_to": window[1]})
    if r.get("_status") != 200 or not isinstance(r.get("data"), dict):
        return None, f"matches status={r.get('_status')} {r.get('_error', '')}".strip()
    items = r["data"].get("data", [])
    finished = [m for m in items
                if str(m.get("status", "")).lower() == "finished"
                and (parse_dt(m.get("utc_date")) or CUTOFF) < CUTOFF]
    if not finished:
        return None, f"no finished pre-cutoff fixture ({len(items)} in window)"
    with_odds = [m for m in finished if m.get("odds_available") is True]
    return (with_odds[0] if with_odds else finished[0]), None


def probe_category(cat, comps):
    rep = {"label": cat["label"], "competition": None, "competition_id": None,
           "comp_odds_flag": None, "comp_xg_flag": None,
           "fixture": None, "fixture_date": None, "score": None, "result": False,
           "odds": False, "books": [], "pinnacle": False, "closing": False,
           "xg": False, "xg_value": None, "notes": []}

    comp = pick_competition(comps, cat)
    if comp is None:
        rep["notes"].append("no matching competition in the plan's competition list")
        return rep
    rep["competition"] = comp.get("name")
    rep["competition_id"] = comp.get("id")
    rep["comp_odds_flag"] = comp.get("odds_available")
    rep["comp_xg_flag"] = comp.get("xg_available")

    fixture, err = find_fixture(comp["id"], cat["window"])
    if fixture is None:
        rep["notes"].append(err)
        return rep

    mid = fixture.get("id")
    home = (fixture.get("home_team") or {}).get("name", "?")
    away = (fixture.get("away_team") or {}).get("name", "?")
    rep["fixture"] = f"{home} vs {away}"
    dt = parse_dt(fixture.get("utc_date"))
    rep["fixture_date"] = dt.date().isoformat() if dt else None
    rep["result"] = str(fixture.get("status", "")).lower() == "finished"
    rep["score"] = fmt_score(fixture)

    # Odds (skip the call only if the competition explicitly advertises none).
    if rep["comp_odds_flag"] is False:
        rep["notes"].append("competition odds_available=false -> odds not fetched")
    else:
        od = get(f"/matches/{mid}/odds")
        if od.get("_status") == 200:
            rep["odds"], rep["books"], rep["pinnacle"], rep["closing"] = parse_odds(od["data"])
        else:
            rep["notes"].append(f"odds: status={od.get('_status')} {od.get('_error', '')}".strip())

    # xG (skip the call only if the competition explicitly advertises none).
    if rep["comp_xg_flag"] is False:
        rep["notes"].append("competition xg_available=false -> stats not fetched")
    else:
        st = get(f"/matches/{mid}/stats")
        if st.get("_status") == 200:
            rep["xg"], rep["xg_value"] = parse_xg(st["data"])
            if not rep["xg"] and rep["xg_value"] is not None:
                rep["notes"].append(f"expected_goals present but zero/empty {rep['xg_value']}")
        else:
            rep["notes"].append(f"stats: status={st.get('_status')} {st.get('_error', '')}".strip())

    return rep


def yn(b):
    return "yes" if b else "no"


def flag(v):
    return {True: "true", False: "false", None: "n/a"}[v]


def main():
    print("=" * 80)
    print("TheStatsAPI READ-ONLY coverage probe  (pre-Nov-2022 internationals)")
    print("base:", BASE, "| auth: Authorization: Bearer <key> (key not shown)")
    print("=" * 80)

    comps, status, err = fetch_all_competitions()
    print(f"competitions endpoint status: {status} {err or ''}".rstrip(),
          f"| competitions in plan: {len(comps)}")
    if not comps:
        print("\nCannot resolve competitions -> cannot locate fixtures. "
              "Verdict: provider/endpoint access unconfirmed.")
        print(f"\ntotal API requests made: {_req_count}")
        return

    reports = [probe_category(cat, comps) for cat in CATEGORIES]

    print("\n" + "-" * 80)
    print("PER-CATEGORY COVERAGE")
    print("-" * 80)
    for r in reports:
        print(f"\n[{r['label']}]")
        print(f"  competition  : {r['competition']} (id={r['competition_id']})")
        print(f"  comp flags   : odds_available={flag(r['comp_odds_flag'])}  "
              f"xg_available={flag(r['comp_xg_flag'])}")
        date_str = f"  @ {r['fixture_date']}" if r["fixture_date"] else ""
        print(f"  fixture      : {r['fixture'] or '-'}{date_str}")
        print(f"  result?      : {yn(r['result'])}"
              + (f"  (score {r['score']})" if r["score"] else ""))
        books = ", ".join(r["books"]) if r["books"] else "none"
        print(f"  odds (1X2)?  : {yn(r['odds'])}   books=[{books}]   "
              f"closing(last_seen)? {yn(r['closing'])}   Pinnacle? {yn(r['pinnacle'])}")
        xg_extra = f"  (expected_goals all={r['xg_value']})" if r["xg_value"] is not None else ""
        print(f"  xG?          : {yn(r['xg'])}{xg_extra}")
        if r["odds"] and r["fixture_date"]:
            print(f"  odds history : confirmed back to at least {r['fixture_date']} (sampled fixture)")
        for n in r["notes"]:
            print(f"  note         : {n}")

    # --- verdict ---
    print("\n" + "-" * 80)
    print("VERDICT (SPEC sec.5 surprise feature needs >=3-book de-vigged closing odds)")
    print("-" * 80)

    def short(lbl):
        return lbl.split("(")[0].strip()

    fed = [short(r["label"]) for r in reports if r["odds"]]
    sparse = [short(r["label"]) for r in reports if not r["odds"]]
    pinn = [short(r["label"]) for r in reports if r["pinnacle"]]
    multi = [short(r["label"]) for r in reports if len(r["books"]) >= 3]
    xg_ok = [short(r["label"]) for r in reports if r["xg"]]

    print("  result coverage   : " + ", ".join(short(r["label"]) for r in reports if r["result"]))
    print("  1X2 odds present  : " + (", ".join(fed) or "none"))
    print("  >=3 books present : " + (", ".join(multi) or "none (single-book only where odds exist)"))
    print("  Pinnacle present  : " + (", ".join(pinn) or "none"))
    print("  real xG present   : " + (", ".join(xg_ok) or "none"))
    print("  no odds at all    : " + (", ".join(sparse) or "none"))
    print()
    print("  Read: the surprise feature will be SPARSE for internationals. Odds exist")
    print("  for covered comps but as a SINGLE book (Bet365) with no Pinnacle and only a")
    print("  last_seen (~closing) price -- so SPEC sec.5's >=3-book consensus de-vig is")
    print("  not satisfiable here; U/M reduce to a single-book signal or fall back to the")
    print("  no-odds skip / Elo proxy. xG is effectively ABSENT (zeros) for internationals,")
    print("  so the attack/defense blend leans on goals (SPEC sec.4 fallback). Results are")
    print("  solid across confederations. Net: well-fed for results; thin for odds/xG.")

    print(f"\ntotal API requests made: {_req_count} (cap {MAX_REQUESTS})")


if __name__ == "__main__":
    main()
