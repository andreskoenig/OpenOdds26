"""Fetch 1X2 closing odds for the 64 FIFA World Cup 2022 matches from TheStatsAPI.

Safety / constraints:
- Read-only: GET requests only. Writes ONLY data/match_odds.json.
- The API key is read from os.environ["STATSAPI_KEY"] via python-dotenv and is
  sent ONLY in the Authorization header. It is NEVER printed, logged, or written.
- Only fetches 1X2 odds (match_odds market). No xG, no stats, no other data.
- Only fetches the 64 WC2022 matches from comp_6107. No other competitions.
- Bookmaker: Bet365 only (the sole book present per prior probe).
- Closing price = last_seen value; falls back to opening if last_seen is null.
- Orientation: aligns provider home/away to OUR fixture's home_team_id/away_team_id,
  swapping if the provider's home maps to our away_team_id.
- Paced at ~0.6 s between requests (under 120/min).
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
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE = "https://api.thestatsapi.com/api/football"
COMP_ID = "comp_6107"          # FIFA World Cup (verified: odds_available=true)
DATE_FROM = "2022-11-20"
DATE_TO = "2022-12-18"
MIN_INTERVAL = 4.0             # seconds between calls (free tier rate-limits bursts)
MAX_RETRIES = 6                # retries on HTTP 429, with backoff

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MATCH_RESULTS_PATH = REPO_ROOT / "data" / "match_results.json"
TEAMS_PATH = REPO_ROOT / "data" / "teams.json"
OUTPUT_PATH = REPO_ROOT / "data" / "match_odds.json"

# ---------------------------------------------------------------------------
# Auth setup (key NEVER printed/logged)
# ---------------------------------------------------------------------------
load_dotenv(REPO_ROOT / ".env")
try:
    _API_KEY = os.environ["STATSAPI_KEY"]
except KeyError:
    print("FATAL: STATSAPI_KEY not found in environment/.env", file=sys.stderr)
    sys.exit(1)

_last_call = 0.0


def _get(path: str, params: dict | None = None) -> dict:
    """Paced GET with 429 backoff. Returns parsed JSON body or raises on failure."""
    global _last_call
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    for attempt in range(MAX_RETRIES + 1):
        gap = time.monotonic() - _last_call
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", "Bearer " + _API_KEY)   # key only in header
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _UA)
        _last_call = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < MAX_RETRIES:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if (ra and str(ra).isdigit()) else 20.0 * (attempt + 1)
                wait = min(wait, 90.0)
                print(f"    429 rate-limited; backing off {wait:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code} for {url}") from e
        except Exception as e:
            raise RuntimeError(f"{type(e).__name__} for {url}") from e
    raise RuntimeError(f"exhausted retries for {url}")


# ---------------------------------------------------------------------------
# Team name resolution
# ---------------------------------------------------------------------------
# Extra hard-coded aliases for names known to differ in TheStatsAPI
EXTRA_ALIASES: dict[str, str] = {
    "korea republic": "south_korea",
    "republic of korea": "south_korea",
    "south korea": "south_korea",
    "usa": "united_states",
    "united states": "united_states",
    "united states of america": "united_states",
    "ir iran": "iran",
    "iran (islamic republic of)": "iran",
    "ivory coast": "ivory_coast",
    "cote d'ivoire": "ivory_coast",
    "côte d'ivoire": "ivory_coast",
}


def _slug(name: str) -> str:
    """Lowercase, non-alphanumeric -> underscores, collapse runs."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", name.lower())).strip("_")


def build_resolver(teams: list[dict]) -> dict[str, str]:
    """Map any name variant -> team_id."""
    resolver: dict[str, str] = {}
    for t in teams:
        tid = t["team_id"]
        for name in [t["canonical_name"]] + t.get("aliases", []):
            resolver[name.lower()] = tid
            resolver[_slug(name)] = tid
    # Add hard-coded extras (override if already present)
    resolver.update(EXTRA_ALIASES)
    return resolver


def resolve_team(name: str, resolver: dict[str, str]) -> str | None:
    raw = name.lower()
    if raw in resolver:
        return resolver[raw]
    slug = _slug(name)
    if slug in resolver:
        return resolver[slug]
    return None


# ---------------------------------------------------------------------------
# Fixture alignment
# ---------------------------------------------------------------------------

def load_wc2022_fixtures(
    path: Path,
) -> tuple[dict[str, dict], dict[tuple[str, str, str], str]]:
    """Return (match_by_id, lookup).

    lookup: (date_str, team_id_a, team_id_b) -> match_id
      where a < b alphabetically so the key is order-independent.
    """
    with path.open() as f:
        all_matches = json.load(f)

    wc22 = [
        m for m in all_matches
        if m.get("competition") == "FIFA World Cup"
        and m["date"] > "2022-11-19"
        and m["date"] <= "2022-12-31"
    ]

    match_by_id: dict[str, dict] = {m["match_id"]: m for m in wc22}
    lookup: dict[tuple[str, str, str], str] = {}
    for m in wc22:
        h, a = m["home_team_id"], m["away_team_id"]
        key = (m["date"], min(h, a), max(h, a))
        lookup[key] = m["match_id"]

    return match_by_id, lookup


# ---------------------------------------------------------------------------
# Odds extraction
# ---------------------------------------------------------------------------

def extract_closing(market: dict, outcome: str) -> float | None:
    """Return closing (last_seen, else opening) decimal price, or None."""
    v = market.get(outcome)
    if not isinstance(v, dict):
        return None
    ls = v.get("last_seen")
    if ls not in (None, "", "null"):
        try:
            return float(ls)
        except (ValueError, TypeError):
            pass
    op = v.get("opening")
    if op not in (None, "", "null"):
        try:
            return float(op)
        except (ValueError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    teams: list[dict] = json.loads(TEAMS_PATH.read_text(encoding="utf-8"))
    resolver = build_resolver(teams)

    match_by_id, lookup = load_wc2022_fixtures(MATCH_RESULTS_PATH)
    our_ids = set(match_by_id)
    print(f"WC2022 fixtures in our data: {len(our_ids)}", flush=True)

    # Resume: keep any odds rows already fetched; only fetch the missing fixtures.
    existing_rows: list[dict] = []
    done: set[str] = set()
    if OUTPUT_PATH.exists():
        try:
            existing_rows = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
            done = {r["match_id"] for r in existing_rows if r.get("match_id") in our_ids}
            existing_rows = [r for r in existing_rows if r.get("match_id") in our_ids]
        except (ValueError, OSError):
            existing_rows, done = [], set()
    print(f"already fetched (resuming): {len(done)}", flush=True)

    # --- 1. Fetch all 64 WC matches from provider ---
    print(f"Fetching matches for {COMP_ID} ({DATE_FROM} -> {DATE_TO}) ...", flush=True)
    r = _get(
        "/matches",
        {
            "competition_id": COMP_ID,
            "date_from": DATE_FROM,
            "date_to": DATE_TO,
            "per_page": 100,
        },
    )
    provider_matches = r.get("data", [])
    print(f"Provider returned {len(provider_matches)} matches.", flush=True)

    if len(provider_matches) != 64:
        print(
            f"WARNING: expected 64 matches, got {len(provider_matches)}.",
            file=sys.stderr,
        )

    # --- 2. Align provider matches to our fixtures ---
    # For each provider match, resolve team names -> team_ids -> find our fixture.
    unresolved_names: list[str] = []
    unmatched_fixtures: list[str] = []
    aligned: list[tuple[str, str, str, dict]] = []  # (our_match_id, prov_id, date, fixture)

    for pm in provider_matches:
        prov_id = pm.get("id")
        utc_date = pm.get("utc_date", "")
        # Date: take just the YYYY-MM-DD portion
        date_str = utc_date[:10] if utc_date else None
        ht_name = (pm.get("home_team") or {}).get("name", "")
        at_name = (pm.get("away_team") or {}).get("name", "")

        ht_id = resolve_team(ht_name, resolver)
        at_id = resolve_team(at_name, resolver)

        if ht_id is None:
            unresolved_names.append(ht_name)
        if at_id is None:
            unresolved_names.append(at_name)
        if ht_id is None or at_id is None:
            continue

        key = (date_str, min(ht_id, at_id), max(ht_id, at_id))
        our_mid = lookup.get(key)
        if our_mid is None:
            unmatched_fixtures.append(
                f"{date_str} {ht_name}({ht_id}) vs {at_name}({at_id})"
            )
            continue

        aligned.append((our_mid, prov_id, date_str, match_by_id[our_mid]))

    print(
        f"Aligned {len(aligned)}/{len(our_ids)} fixtures to provider matches.",
        flush=True,
    )
    if unresolved_names:
        print(f"UNRESOLVED team names: {list(set(unresolved_names))}", file=sys.stderr)
    if unmatched_fixtures:
        print(f"UNMATCHED fixtures: {unmatched_fixtures}", file=sys.stderr)

    # --- 3. Fetch odds for each aligned match ---
    rows: list[dict] = []
    missing_odds: list[str] = []

    for i, (our_mid, prov_id, date_str, our_fixture) in enumerate(aligned):
        if our_mid in done:
            continue  # already have this fixture's odds (resume)
        print(
            f"  [{i+1}/{len(aligned)}] {prov_id} -> {our_mid} ...",
            end=" ",
            flush=True,
        )
        try:
            odds_resp = _get(f"/matches/{prov_id}/odds")
        except RuntimeError as e:
            print(f"ERROR: {e}", flush=True)
            missing_odds.append(f"{our_mid} ({e})")
            continue

        bookmakers = (odds_resp.get("data") or {}).get("bookmakers", [])

        # Find Bet365
        bet365 = None
        for bm in bookmakers:
            if isinstance(bm, dict) and "bet365" in str(bm.get("bookmaker", "")).lower():
                bet365 = bm
                break

        if bet365 is None:
            all_books = [b.get("bookmaker") for b in bookmakers if isinstance(b, dict)]
            print(f"no Bet365 (books: {all_books})", flush=True)
            missing_odds.append(f"{our_mid} (no Bet365; books={all_books})")
            continue

        mo = (bet365.get("markets") or {}).get("match_odds")
        if not isinstance(mo, dict):
            print("no match_odds market", flush=True)
            missing_odds.append(f"{our_mid} (no match_odds market)")
            continue

        # Extract raw closing prices (provider perspective: home, draw, away)
        prov_home = extract_closing(mo, "home")
        prov_draw = extract_closing(mo, "draw")
        prov_away = extract_closing(mo, "away")

        if prov_home is None or prov_draw is None or prov_away is None:
            print(f"missing prices h={prov_home} d={prov_draw} a={prov_away}", flush=True)
            missing_odds.append(f"{our_mid} (missing prices)")
            continue

        # --- Orientation fix ---
        # Resolve provider's home team to our team_id
        prov_ht_name = (
            # find the original provider match entry
            next(
                (pm.get("home_team", {}).get("name", "") for pm in provider_matches if pm.get("id") == prov_id),
                "",
            )
        )
        prov_ht_id = resolve_team(prov_ht_name, resolver)

        our_home_id = our_fixture["home_team_id"]
        our_away_id = our_fixture["away_team_id"]

        if prov_ht_id == our_away_id:
            # Provider's home is our away -> swap
            odds_home = prov_away
            odds_draw = prov_draw
            odds_away = prov_home
            orientation_note = "SWAPPED"
        else:
            odds_home = prov_home
            odds_draw = prov_draw
            odds_away = prov_away
            orientation_note = "aligned"

        print(
            f"OK [{orientation_note}] Bet365 h={odds_home} d={odds_draw} a={odds_away}",
            flush=True,
        )

        rows.append(
            {
                "match_id": our_mid,
                "bookmaker": "Bet365",
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
                "captured_at": date_str,
            }
        )

    # --- 4. Validate and write output (merge resumed + newly fetched) ---
    all_rows = existing_rows + rows
    output_match_ids = {r["match_id"] for r in all_rows}
    missing_from_output = our_ids - output_match_ids

    print(f"\n--- Summary ---")
    print(f"WC2022 fixtures:          {len(our_ids)}")
    print(f"Provider matches fetched: {len(provider_matches)}")
    print(f"Aligned:                  {len(aligned)}")
    print(f"Previously saved:         {len(existing_rows)}")
    print(f"Newly fetched this run:   {len(rows)}")
    print(f"Total rows with 1X2:      {len(all_rows)}")
    print(f"Missing odds:             {len(missing_odds)}")
    if missing_odds:
        print("  Missing:")
        for m in missing_odds:
            print(f"    {m}")
    if missing_from_output:
        print(f"Fixtures with NO row in output ({len(missing_from_output)}):")
        for mid in sorted(missing_from_output):
            fx = match_by_id[mid]
            print(f"    {mid}  {fx['date']} {fx['home_team_id']} vs {fx['away_team_id']}")

    # Validate every match_id in output is a real WC2022 fixture
    spurious = output_match_ids - our_ids
    if spurious:
        print(f"ERROR: spurious match_ids in output: {spurious}", file=sys.stderr)
        sys.exit(1)

    # Write output
    OUTPUT_PATH.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_PATH}")
    print("Fetched ONLY 1X2 odds for these 64 WC2022 matches. No xG. No other data.")
    print("API key was never printed or logged.")


if __name__ == "__main__":
    main()
