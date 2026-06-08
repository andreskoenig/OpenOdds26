"""POC: scrape per-match xG from FotMob for international matches.

FotMob gates its stats API behind a signed `x-mas` header (direct GET -> 403).
The robust workaround: drive a headless browser to the match page, let FotMob's
OWN JavaScript call the gated endpoint (it computes x-mas), and intercept the
`matchDetails` response. We then pull team xG and emit rows in our existing
team_xg schema: {match_id, team_id, xg_for, xg_against}.

This is a PROOF OF CONCEPT (small N), not the production fetcher. It proves the
data is systematically obtainable. Enumeration of match IDs uses the FREE,
ungated page HTML (__NEXT_DATA__ / inline ids); only the xG fetch needs the browser.

Run:  python scripts/poc_fotmob_xg.py [TEAM_ID] [MAX_MATCHES]
Default: Colombia (8258), up to 6 matches with xG.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

TEAM_ID = sys.argv[1] if len(sys.argv) > 1 else "8258"          # Colombia
MAX_MATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 6


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def enumerate_match_ids(team_id):
    """FREE (no header): pull candidate match ids from the team overview page."""
    url = f"https://www.fotmob.com/teams/{team_id}/overview"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=40).read().decode("utf-8", "replace")
    # candidate match ids: 6-8 digit ints that appear as match links / ids
    ids = re.findall(r'(?:matchId\\?"?:\\?"?|/match/|#)(\d{6,8})', html)
    seen, out = set(), []
    for m in ids:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def find_xg(obj):
    """Recursively locate the 'Expected goals (xG)' [home, away] pair in matchDetails."""
    hits = []

    def walk(o):
        if isinstance(o, dict):
            title = str(o.get("title", "")).lower()
            if "expected goals" in title or title == "xg":
                hits.append(o.get("stats") or o.get("value"))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    for val in hits:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            try:
                return float(val[0]), float(val[1])
            except (TypeError, ValueError):
                continue
    return None


def find_meta(md):
    """Best-effort home/away team names + date + matchId from matchDetails."""
    g = md.get("general", {}) if isinstance(md, dict) else {}
    ht = (g.get("homeTeam") or {}).get("name")
    at = (g.get("awayTeam") or {}).get("name")
    mid = g.get("matchId") or g.get("id")
    dt = (g.get("matchTimeUTC") or g.get("matchTimeUTCDate") or "")[:10] if g else ""
    return ht, at, str(mid) if mid else None, dt


def main():
    from playwright.sync_api import sync_playwright

    print(f"enumerating match ids for team {TEAM_ID} (free HTTP) ...", flush=True)
    ids = enumerate_match_ids(TEAM_ID)
    print(f"  found {len(ids)} candidate ids; will scrape until {MAX_MATCHES} have xG\n", flush=True)

    rows = []          # team_xg schema rows
    table = []         # for display
    captured = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=UA)
        # speed up: drop images/media/fonts (we only need the JSON XHR)
        page.route("**/*", lambda r: (
            r.abort() if r.request.resource_type in ("image", "media", "font")
            else r.continue_()))

        def on_response(resp):
            if "matchdetails" in resp.url.lower():
                try:
                    captured["md"] = resp.json()
                except Exception:
                    pass

        page.on("response", on_response)

        for mid in ids:
            captured.clear()
            try:
                page.goto(f"https://www.fotmob.com/match/{mid}",
                          wait_until="domcontentloaded", timeout=45000)
            except Exception as e:
                print(f"  [{mid}] nav error: {type(e).__name__}")
                continue
            # poll up to ~15s for the intercepted matchDetails XHR
            for _ in range(30):
                if captured.get("md"):
                    break
                page.wait_for_timeout(500)

            md = captured.get("md")
            if not md:
                print(f"  [{mid}] no matchDetails captured")
                continue
            xg = find_xg(md)
            ht, at, real_mid, dt = find_meta(md)
            if not xg:
                print(f"  [{mid}] {ht} vs {at}: no xG (likely not played)")
                continue

            xh, xa = xg
            hid, aid = _slug(ht or "home"), _slug(at or "away")
            key = real_mid or mid
            rows.append({"match_id": f"fotmob_{key}", "team_id": hid,
                         "xg_for": xh, "xg_against": xa})
            rows.append({"match_id": f"fotmob_{key}", "team_id": aid,
                         "xg_for": xa, "xg_against": xh})
            table.append((dt, ht, at, xh, xa))
            print(f"  [{mid}] OK  {dt}  {ht} {xh:.2f} - {xa:.2f} {at}", flush=True)
            if len(table) >= MAX_MATCHES:
                break

        browser.close()

    out = os.path.join(ROOT, "data", "xg_poc.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"source": "fotmob matchDetails via headless browser (POC)",
                   "team_id": TEAM_ID, "n_matches": len(table),
                   "team_xg_rows": rows}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"POC RESULT — {len(table)} matches with xG  (team_xg rows: {len(rows)})")
    print("=" * 64)
    print(f"  {'date':<12}{'home':<16}{'xG':>6}  {'xG':<6}{'away'}")
    for dt, ht, at, xh, xa in table:
        print(f"  {dt:<12}{(ht or '')[:15]:<16}{xh:>6.2f}  {xa:<6.2f}{at or ''}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
