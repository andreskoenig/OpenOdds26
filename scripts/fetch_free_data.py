"""
fetch_free_data.py
==================
Downloads free international football data for the FIFA WC 2022 baseline backtest.

Sources:
  A) martj42/international_results  — match results since 1872
  B) hericlibong/FifaRankingScraper — historical FIFA world rankings 1992-2023

Outputs (under data/):
  teams.json         list of {team_id, canonical_name, aliases, confederation}
  match_results.json list of {match_id, date, home_team_id, away_team_id,
                              venue_country, neutral, competition, home_goals, away_goals}
  fifa_ratings.json  list of {team_id, as_of_date, fifa_points, fifa_rank}

NO odds, NO xG, NO wc_model modifications.
"""

import csv
import io
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# Force UTF-8 output on Windows to avoid codec errors with Unicode team names
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

URL_RESULTS = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
URL_RANKINGS = (
    "https://raw.githubusercontent.com/hericlibong/FifaRankingScraper/main/"
    "historicalmenranking/historicalmenranking/spiders/data.csv"
)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Alias table  (martj42 name  ->  canonical name)
# Where the two sources use different spellings we normalise to ONE canonical.
# martj42 name is the canonical; aliases captures ALL known variant spellings.
# ---------------------------------------------------------------------------

# Each entry: (canonical_name, [additional_aliases])
# canonical = martj42 spelling (or best-known English form)
ALIAS_PAIRS = [
    # martj42 canonical → FIFA ranking variants
    ("United States",           ["USA", "United States of America", "US"]),
    ("South Korea",             ["Korea Republic", "Korea DPR", "Republic of Korea"]),
    ("North Korea",             ["DPR Korea", "Korea DPR"]),
    ("Iran",                    ["IR Iran", "Islamic Republic of Iran"]),
    ("China",                   ["China PR", "People's Republic of China"]),
    ("Czech Republic",          ["Czechia", "Czech Rep.", "Czechia"]),
    ("Ivory Coast",             ["Côte d'Ivoire", "Cote d'Ivoire", "Cote dIvoire",
                                  "Côte d'Ivoire"]),
    ("DR Congo",                ["Congo DR", "Congo, DR", "Democratic Republic of the Congo",
                                  "DR Congo"]),
    ("Cape Verde",              ["Cabo Verde"]),
    ("Curacao",                 ["Curaçao", "Curaçao"]),
    ("Bosnia and Herzegovina",  ["Bosnia & Herzegovina", "Bosnia-Herzegovina"]),
    ("Republic of Ireland",     ["Ireland", "Rep. of Ireland"]),
    ("North Macedonia",         ["Macedonia", "Macedonia, FYR", "FYR Macedonia",
                                  "Republic of North Macedonia"]),
    ("Eswatini",                ["Swaziland"]),
    ("Kosovo",                  ["Kosovo"]),
    ("Saint Kitts and Nevis",   ["Saint Kitts & Nevis", "St. Kitts and Nevis",
                                  "St Kitts and Nevis"]),
    ("Saint Vincent and the Grenadines",
                                ["St. Vincent and the Grenadines",
                                  "St Vincent and the Grenadines",
                                  "Saint Vincent & the Grenadines"]),
    ("Saint Lucia",             ["St. Lucia", "St Lucia"]),
    ("Sao Tome and Principe",   ["São Tomé and Príncipe",
                                  "São Tomé and Príncipe",
                                  "São Tomé e Príncipe",
                                  "São Tomé & Príncipe",
                                  "Sao Tome & Principe"]),
    # FIFA ranking uses "Korea Republic" and "IR Iran" — these map to South Korea / Iran above.
    # Additional variants from the FIFA rankings source (country field):
    ("Syria",                   ["Syrian Arab Republic"]),
    ("Russia",                  ["Russian Federation"]),
    ("Bolivia",                 ["Bolivia, Plurinational State of"]),
    ("Tanzania",                ["Tanzania, United Republic of"]),
    ("Congo",                   ["Congo (Brazzaville)", "Republic of the Congo",
                                  "Congo, Republic of"]),
    ("Kyrgyzstan",              ["Kyrgyz Republic"]),
    ("Moldova",                 ["Republic of Moldova"]),
    ("Vietnam",                 ["Viet Nam"]),
    ("Laos",                    ["Lao PDR", "Lao People's Democratic Republic"]),
    ("Macedonia",               ["North Macedonia"]),   # handled above; extra guard
    ("South Korea",             ["Korea Republic"]),    # duplicate guard
    ("Macau",                   ["Macao"]),
    ("Palestine",               ["Palestinian Territory"]),
    ("Trinidad and Tobago",     ["Trinidad & Tobago"]),
    ("Antigua and Barbuda",     ["Antigua & Barbuda"]),
    ("Turks and Caicos Islands",["Turks & Caicos Islands"]),
    ("Sint Maarten",            ["St. Maarten"]),
    ("Timor-Leste",             ["East Timor"]),
    ("Brunei",                  ["Brunei Darussalam"]),
    ("Hong Kong",               ["Hong Kong, China"]),
    ("British Virgin Islands",  ["BVI"]),
    ("US Virgin Islands",       ["United States Virgin Islands"]),
    ("Guam",                    ["Guam"]),
    ("American Samoa",          ["American Samoa"]),
    ("Northern Mariana Islands",["Northern Mariana Is."]),
    ("Dominica",                ["Dominica"]),
    ("Grenada",                 ["Grenada"]),
    ("Bermuda",                 ["Bermuda"]),
    ("Cayman Islands",          ["Cayman Is."]),
    ("Bahamas",                 ["The Bahamas"]),
    ("Guinea-Bissau",           ["Guinea Bissau"]),
    ("Equatorial Guinea",       ["Equatorial Guinea"]),
    # FIFA ranking source-specific variants
    ("Cape Verde",              ["Cape Verde Islands"]),
    ("Chinese Taipei",          ["Chinese Taipei"]),
    ("Sao Tome and Principe",   ["Sao Tome e Principe", "São Tomé e Príncipe"]),
    ("Saint Vincent and the Grenadines",
                                ["St. Vincent / Grenadines", "St. Vincent and the Grenadines"]),
    ("Gambia",                  ["The Gambia"]),
    ("Turkey",                  ["Turkiye", "Türkiye", "Tırkiye"]),
]

def make_team_id(name: str) -> str:
    """Convert a canonical team name to a slug team_id."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug


def build_alias_map():
    """Return a dict: any_spelling.lower() -> canonical_name."""
    alias_map = {}
    for canonical, aliases in ALIAS_PAIRS:
        # Map canonical to itself
        alias_map[canonical.lower()] = canonical
        # Map each alias to the canonical
        for a in aliases:
            alias_map[a.lower()] = canonical
    return alias_map


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Step 1: Download match results
# ---------------------------------------------------------------------------
print("=" * 60)
print("Downloading match results from martj42...")
results_csv = fetch_url(URL_RESULTS)
results_reader = csv.DictReader(io.StringIO(results_csv))
results_rows = list(results_reader)
print(f"  Raw rows downloaded: {len(results_rows)}")
print(f"  Columns: {results_reader.fieldnames}")

# ---------------------------------------------------------------------------
# Step 2: Download FIFA rankings
# ---------------------------------------------------------------------------
print("\nDownloading FIFA rankings from hericlibong/FifaRankingScraper...")
rankings_csv = fetch_url(URL_RANKINGS)
rankings_reader = csv.DictReader(io.StringIO(rankings_csv))
rankings_rows = list(rankings_reader)
print(f"  Raw rows downloaded: {len(rankings_rows)}")
print(f"  Columns: {rankings_reader.fieldnames}")

# ---------------------------------------------------------------------------
# Step 3: Build teams.json
#   - Seed from ALL teams in match_results
#   - Set confederation from FIFA rankings where matched, else null
# ---------------------------------------------------------------------------
print("\nBuilding teams.json...")

alias_map = build_alias_map()

# Collect all team names from match results
all_team_names: set = set()
for row in results_rows:
    all_team_names.add(row["home_team"])
    all_team_names.add(row["away_team"])

# Build canonical lookup:  original_name -> canonical_name
def get_canonical(name: str) -> str:
    return alias_map.get(name.lower(), name)

# Build teams dict: canonical_name -> {team_id, canonical_name, aliases, confederation}
teams_dict: dict = {}

# Reverse alias map: canonical -> set of aliases
canonical_to_aliases: dict = {}
for variant, canonical in alias_map.items():
    if canonical not in canonical_to_aliases:
        canonical_to_aliases[canonical] = set()
    canonical_to_aliases[canonical].add(variant)

for name in sorted(all_team_names):
    canonical = get_canonical(name)
    tid = make_team_id(canonical)
    if canonical not in teams_dict:
        teams_dict[canonical] = {
            "team_id": tid,
            "canonical_name": canonical,
            "aliases": [],
            "confederation": None,
        }
    # Add this original name as alias if different from canonical
    if name != canonical and name not in teams_dict[canonical]["aliases"]:
        teams_dict[canonical]["aliases"].append(name)

# Populate confederation from FIFA rankings source
# FIFA source columns: date, country, rank, previousRank, totalPoints, previousPoints, ...conf
# Build country->conf from most recent ranking entry
country_conf: dict = {}
for row in rankings_rows:
    c = row["country"].strip()
    conf = row["conf"].strip() if row.get("conf") else None
    country_conf[c] = conf

# Map FIFA country names to canonical, then populate confederation
for fifa_country, conf in country_conf.items():
    canonical = get_canonical(fifa_country)
    if canonical in teams_dict and conf:
        teams_dict[canonical]["confederation"] = conf

# Convert to list, sort by team_id
teams_list = sorted(teams_dict.values(), key=lambda t: t["team_id"])

# Ensure aliases is a sorted list without duplicates
for t in teams_list:
    t["aliases"] = sorted(set(t["aliases"]))

print(f"  Total teams: {len(teams_list)}")
teams_conf_count = sum(1 for t in teams_list if t["confederation"] is not None)
print(f"  Teams with confederation set: {teams_conf_count}")

# ---------------------------------------------------------------------------
# Step 4: Build match_results.json
# ---------------------------------------------------------------------------
print("\nBuilding match_results.json...")

def parse_score(s: str):
    """Return int or None."""
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None

# Build team lookup: canonical_name -> team_id
name_to_id: dict = {}
for t in teams_list:
    name_to_id[t["canonical_name"]] = t["team_id"]
    for a in t["aliases"]:
        name_to_id[a] = t["team_id"]
# Also cover raw names via alias_map
for variant, canonical in alias_map.items():
    if canonical in name_to_id:
        name_to_id[variant] = name_to_id[canonical]

def get_team_id(name: str) -> str | None:
    canonical = get_canonical(name)
    return name_to_id.get(canonical) or name_to_id.get(name)

match_results = []
dropped = 0
dropped_reasons = []

for i, row in enumerate(results_rows):
    home_goals = parse_score(row.get("home_score", ""))
    away_goals = parse_score(row.get("away_score", ""))
    if home_goals is None or away_goals is None:
        dropped += 1
        if len(dropped_reasons) < 10:
            dropped_reasons.append(
                f"Row {i}: home={row.get('home_score')!r} away={row.get('away_score')!r} "
                f"teams={row.get('home_team')} vs {row.get('away_team')} date={row.get('date')}"
            )
        continue

    date = row["date"].strip()
    home_name = row["home_team"].strip()
    away_name = row["away_team"].strip()

    home_id = get_team_id(home_name)
    away_id = get_team_id(away_name)
    if home_id is None:
        home_id = make_team_id(get_canonical(home_name))
    if away_id is None:
        away_id = make_team_id(get_canonical(away_name))

    neutral_raw = row.get("neutral", "FALSE").strip().upper()
    neutral = neutral_raw == "TRUE"

    match_id = f"{i:05d}_{date}_{home_id}_{away_id}"

    match_results.append({
        "match_id": match_id,
        "date": date,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "venue_country": row.get("country", "").strip(),
        "neutral": neutral,
        "competition": row.get("tournament", "").strip(),
        "home_goals": home_goals,
        "away_goals": away_goals,
    })

print(f"  Rows kept: {len(match_results)}")
print(f"  Rows dropped (non-integer scores): {dropped}")
if dropped_reasons:
    print(f"  Sample dropped rows:")
    for r in dropped_reasons:
        print(f"    {r}")

# Stats
from collections import Counter
competitions = Counter(m["competition"] for m in match_results)
wc_total = competitions.get("FIFA World Cup", 0)
wc_2022 = sum(
    1 for m in match_results
    if m["competition"] == "FIFA World Cup" and m["date"] >= "2022-11-01" and m["date"] <= "2022-12-31"
)
distinct_teams = set()
for m in match_results:
    distinct_teams.add(m["home_team_id"])
    distinct_teams.add(m["away_team_id"])
all_dates = [m["date"] for m in match_results]
print(f"  Date range: {min(all_dates)} to {max(all_dates)}")
print(f"  Distinct team IDs: {len(distinct_teams)}")
print(f"  Rows with competition=='FIFA World Cup': {wc_total}")
print(f"  2022 WC matches (Nov-Dec 2022): {wc_2022}")

# ---------------------------------------------------------------------------
# Step 5: Build fifa_ratings.json
#   Keep snapshots with as_of_date in [2018-01-01, 2022-11-19]
# ---------------------------------------------------------------------------
print("\nBuilding fifa_ratings.json...")

FIFA_DATE_MIN = "2018-01-01"
FIFA_DATE_MAX = "2022-11-19"

fifa_ratings = []
skipped_date = 0
skipped_points = 0
unmatched_names: set = set()

for row in rankings_rows:
    as_of_date = row.get("date", "").strip()
    if not as_of_date:
        continue
    if as_of_date < FIFA_DATE_MIN or as_of_date > FIFA_DATE_MAX:
        skipped_date += 1
        continue

    country_raw = row.get("country", "").strip()
    try:
        fifa_points = float(row.get("totalPoints", "").strip())
    except (ValueError, AttributeError):
        skipped_points += 1
        continue
    try:
        fifa_rank = int(row.get("rank", "").strip())
    except (ValueError, AttributeError):
        skipped_points += 1
        continue

    canonical = get_canonical(country_raw)
    tid = name_to_id.get(canonical) or name_to_id.get(country_raw)
    if tid is None:
        # Try to match via make_team_id
        candidate_id = make_team_id(canonical)
        # Check if this id exists in teams
        tid_exists = any(t["team_id"] == candidate_id for t in teams_list)
        if tid_exists:
            tid = candidate_id
        else:
            unmatched_names.add(country_raw)
            # Still create entry with generated id
            tid = candidate_id

    fifa_ratings.append({
        "team_id": tid,
        "as_of_date": as_of_date,
        "fifa_points": fifa_points,
        "fifa_rank": fifa_rank,
    })

print(f"  Total fifa_ratings snapshots kept: {len(fifa_ratings)}")
print(f"  Rows skipped (outside date range): {skipped_date}")
print(f"  Rows skipped (bad points/rank): {skipped_points}")
distinct_rating_dates = sorted(set(r["as_of_date"] for r in fifa_ratings))
distinct_rating_teams = sorted(set(r["team_id"] for r in fifa_ratings))
print(f"  Distinct release dates: {len(distinct_rating_dates)}")
print(f"  Date range: {min(distinct_rating_dates) if distinct_rating_dates else 'N/A'} to {max(distinct_rating_dates) if distinct_rating_dates else 'N/A'}")
print(f"  Distinct teams: {len(distinct_rating_teams)}")
if unmatched_names:
    print(f"  FIFA country names with no match_results team_id ({len(unmatched_names)} total):")
    for n in sorted(unmatched_names)[:30]:
        print(f"    {n!r}")

# ---------------------------------------------------------------------------
# Step 6: Reconciliation — check 32 WC 2022 teams have FIFA points near Oct/Nov 2022
# ---------------------------------------------------------------------------
print("\nReconciliation — 32 WC 2022 teams coverage...")

WC2022_TEAMS = [
    "Qatar", "Ecuador", "Senegal", "Netherlands",
    "England", "Iran", "United States", "Wales",
    "Argentina", "Saudi Arabia", "Mexico", "Poland",
    "France", "Australia", "Denmark", "Tunisia",
    "Spain", "Costa Rica", "Germany", "Japan",
    "Belgium", "Canada", "Morocco", "Croatia",
    "Brazil", "Serbia", "Switzerland", "Cameroon",
    "Portugal", "Ghana", "Uruguay", "South Korea",
]

# Find team_id for each WC team
wc_ids = {}
for team in WC2022_TEAMS:
    canonical = get_canonical(team)
    tid = name_to_id.get(canonical) or name_to_id.get(team)
    if tid is None:
        tid = make_team_id(canonical)
    wc_ids[team] = tid

# Check which have ratings in Oct 2022
oct_2022_ratings = {r["team_id"] for r in fifa_ratings if r["as_of_date"] == "2022-10-06"}
missing_wc = []
for team, tid in wc_ids.items():
    if tid not in oct_2022_ratings:
        missing_wc.append(team)

print(f"  WC 2022 teams with Oct-2022 FIFA points: {len(WC2022_TEAMS) - len(missing_wc)}/{len(WC2022_TEAMS)}")
if missing_wc:
    print(f"  Missing WC teams from Oct-2022 snapshot: {missing_wc}")
else:
    print("  ALL 32 WC 2022 teams have FIFA points for 2022-10-06.")

# ---------------------------------------------------------------------------
# Coverage-gap: teams in match_results that never appear in fifa_ratings
# ---------------------------------------------------------------------------
rated_team_ids = set(r["team_id"] for r in fifa_ratings)
match_team_ids = set()
for m in match_results:
    match_team_ids.add(m["home_team_id"])
    match_team_ids.add(m["away_team_id"])

unrated_match_teams = match_team_ids - rated_team_ids
print(f"\nTeams in match_results with no FIFA rating row: {len(unrated_match_teams)}")
# Map back to canonical names for reporting
id_to_canonical = {t["team_id"]: t["canonical_name"] for t in teams_list}
unrated_sample = sorted(
    id_to_canonical.get(tid, tid) for tid in unrated_match_teams
)[:50]
print(f"  Sample (up to 50): {unrated_sample}")

# ---------------------------------------------------------------------------
# Step 7: Write output files
# ---------------------------------------------------------------------------
print("\nWriting output files...")

teams_path = DATA_DIR / "teams.json"
results_path = DATA_DIR / "match_results.json"
ratings_path = DATA_DIR / "fifa_ratings.json"

with open(teams_path, "w", encoding="utf-8") as f:
    json.dump(teams_list, f, ensure_ascii=False, indent=2)
print(f"  Wrote {teams_path} ({len(teams_list)} teams)")

with open(results_path, "w", encoding="utf-8") as f:
    json.dump(match_results, f, ensure_ascii=False, indent=2)
print(f"  Wrote {results_path} ({len(match_results)} matches)")

with open(ratings_path, "w", encoding="utf-8") as f:
    json.dump(fifa_ratings, f, ensure_ascii=False, indent=2)
print(f"  Wrote {ratings_path} ({len(fifa_ratings)} rating snapshots)")

# ---------------------------------------------------------------------------
# Step 8: Validate output files parse correctly and have expected fields
# ---------------------------------------------------------------------------
print("\nValidating output files...")

with open(teams_path, encoding="utf-8") as f:
    t_check = json.load(f)
assert isinstance(t_check, list) and len(t_check) > 0
assert all(set(["team_id", "canonical_name", "aliases", "confederation"]) <= set(x.keys()) for x in t_check[:5])
print(f"  teams.json: valid JSON, {len(t_check)} entries, required fields present")

with open(results_path, encoding="utf-8") as f:
    r_check = json.load(f)
assert isinstance(r_check, list) and len(r_check) > 0
required_r = {"match_id", "date", "home_team_id", "away_team_id", "venue_country", "neutral", "competition", "home_goals", "away_goals"}
assert all(required_r <= set(x.keys()) for x in r_check[:5])
print(f"  match_results.json: valid JSON, {len(r_check)} entries, required fields present")

with open(ratings_path, encoding="utf-8") as f:
    f_check = json.load(f)
assert isinstance(f_check, list) and len(f_check) > 0
required_f = {"team_id", "as_of_date", "fifa_points", "fifa_rank"}
assert all(required_f <= set(x.keys()) for x in f_check[:5])
print(f"  fifa_ratings.json: valid JSON, {len(f_check)} entries, required fields present")

print("\n========== COVERAGE-GAP REPORT ==========")
print(f"match_results:")
print(f"  Rows kept:              {len(match_results):,}")
print(f"  Rows dropped:           {dropped}")
print(f"  Date range:             {min(all_dates)} to {max(all_dates)}")
print(f"  Distinct teams:         {len(distinct_teams)}")
print(f"  FIFA World Cup rows:    {wc_total}")
print(f"  2022 WC matches:        {wc_2022}")
print(f"")
print(f"fifa_ratings:")
print(f"  Total snapshots:        {len(fifa_ratings):,}")
print(f"  Distinct release dates: {len(distinct_rating_dates)}")
print(f"  Date range:             {min(distinct_rating_dates) if distinct_rating_dates else 'N/A'} to {max(distinct_rating_dates) if distinct_rating_dates else 'N/A'}")
print(f"  Distinct teams:         {len(distinct_rating_teams)}")
print(f"")
print(f"Reconciliation:")
print(f"  match_results teams with no FIFA rating: {len(unrated_match_teams)}")
print(f"  WC 2022 teams missing Oct-2022 FIFA pts: {len(missing_wc)}")
if missing_wc:
    print(f"  Missing: {missing_wc}")
print("==========================================")
