"""Assemble a point-in-time squad-value (talent-pool) prior from Transfermarkt.

Clean-CSV source (no scraping): dcaribou/transfermarkt-datasets R2 CDN.
Deterministic, cached (raw CSVs kept under data/raw/ so re-runs resume), and
leakage-safe (each snapshot uses only valuations dated <= the as-of date).

Outputs (written BEFORE any wc_model change so a later hiccup keeps the data):
  data/squad_values.json            rows (team_id, as_of_date, total_value_eur, n_players)
  data/squad_values_coverage.txt    coverage by confederation + top-20 + sanity

Run:  python scripts/fetch_squad_values.py
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CDN = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/"
FILES = {"players": "players.csv.gz", "player_valuations": "player_valuations.csv.gz"}
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

AS_OF_DATES = ["2022-08-01", "2022-11-19", "2026-06-02"]  # 2026-06-02 = "current"
TOP_N = 26
SANITY_TEAMS = ["france", "brazil", "england", "spain", "portugal"]

# Transfermarkt citizenship spellings that differ from our martj42-slug team_ids.
TM_OVERRIDES = {
    "korea, south": "south_korea", "south korea": "south_korea",
    "korea, north": "north_korea",
    "cote d'ivoire": "ivory_coast", "côte d'ivoire": "ivory_coast",
    "china": "china_pr", "chinese taipei": "chinese_taipei",
    "turkey": "turkey", "türkiye": "turkey",
    "bosnia-herzegovina": "bosnia_and_herzegovina",
    "the gambia": "gambia", "cape verde": "cape_verde",
    "congo": "congo", "dr congo": "dr_congo", "congo dr": "dr_congo",
    "ireland": "republic_of_ireland", "republic of ireland": "republic_of_ireland",
    "north macedonia": "north_macedonia", "macedonia": "north_macedonia",
    "united states": "united_states", "usa": "united_states",
    "united arab emirates": "united_arab_emirates",
    "curacao": "curacao", "curaçao": "curacao",
    "czech republic": "czech_republic", "czechia": "czech_republic",
    "iran": "iran", "russia": "russia", "venezuela": "venezuela",
}


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", s.lower())).strip("_")


def download():
    os.makedirs(RAW, exist_ok=True)
    for table, fn in FILES.items():
        dest = os.path.join(RAW, fn)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"cached: {fn} ({os.path.getsize(dest) // 1024} KB)", flush=True)
            continue
        url = CDN + fn
        print(f"downloading {url} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  saved {fn} ({os.path.getsize(dest) // 1024} KB)", flush=True)


def _open_csv(table):
    path = os.path.join(RAW, FILES[table])
    return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")


def build_resolver(teams):
    id_set = {t["team_id"] for t in teams}
    lookup = {}
    for t in teams:
        for nm in [t["team_id"], t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
    for k, v in TM_OVERRIDES.items():
        if v in id_set:
            lookup[k] = v

    def resolve(name):
        if not name:
            return None
        return lookup.get(name.lower()) or lookup.get(_slug(name))

    return resolve, id_set


def main():
    download()
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json"), encoding="utf-8"))
    conf_of = {t["team_id"]: t.get("confederation") for t in teams}
    resolve, id_set = build_resolver(teams)

    # --- players: player_id -> team_id (by citizenship) ---
    with _open_csv("players") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        for need in ("player_id", "name", "country_of_citizenship", "position"):
            if need not in cols:
                sys.exit(f"FATAL: players missing column {need!r}; have {cols}")
        player_team = {}
        unmatched = Counter()
        for row in rdr:
            cit = (row.get("country_of_citizenship") or "").strip()
            tid = resolve(cit)
            if tid is None:
                if cit:
                    unmatched[cit] += 1
                continue
            player_team[row["player_id"]] = tid
    print(f"players mapped to a team_id: {len(player_team)} | unmatched citizenships: {len(unmatched)}",
          flush=True)

    # --- valuations: per mapped player, list of (date, value) ---
    with _open_csv("player_valuations") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        for need in ("player_id", "date", "market_value_in_eur"):
            if need not in cols:
                sys.exit(f"FATAL: player_valuations missing column {need!r}; have {cols}")
        vals = defaultdict(list)
        n_rows = 0
        for row in rdr:
            pid = row["player_id"]
            if pid not in player_team:
                continue
            v = row.get("market_value_in_eur")
            d = row.get("date")
            if not v or not d:
                continue
            try:
                vals[pid].append((d[:10], float(v)))
            except ValueError:
                continue
            n_rows += 1
    print(f"valuation rows kept (mapped players): {n_rows} across {len(vals)} players", flush=True)

    # --- snapshots: top-26 most-recent-value (<= as_of) per team ---
    snapshots = []  # (team_id, as_of, total, n)
    by_asof_team_total = {}
    for as_of in AS_OF_DATES:
        team_player_value = defaultdict(list)  # team -> [player_best_value]
        for pid, series in vals.items():
            best = None
            for d, v in series:
                if d <= as_of and (best is None or d > best[0]):
                    best = (d, v)
            if best is not None:
                team_player_value[player_team[pid]].append(best[1])
        for tid, plist in team_player_value.items():
            plist.sort(reverse=True)
            top = plist[:TOP_N]
            total = float(sum(top))
            snapshots.append({"team_id": tid, "as_of_date": as_of,
                              "total_value_eur": total, "n_players": len(top)})
            by_asof_team_total[(as_of, tid)] = (total, len(top))
        print(f"  snapshot {as_of}: {sum(1 for s in snapshots if s['as_of_date'] == as_of)} teams",
              flush=True)

    out = os.path.join(ROOT, "data", "squad_values.json")
    json.dump(snapshots, open(out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {out} ({len(snapshots)} rows)", flush=True)

    # --- coverage + sanity report ---
    lines = []
    lines.append("SQUAD-VALUE COVERAGE + SANITY REPORT")
    lines.append("=" * 60)
    if unmatched:
        lines.append(f"\nUnmatched citizenships (top 25 of {len(unmatched)}):")
        for nm, c in unmatched.most_common(25):
            lines.append(f"  {nm:<28} {c}")

    snap_2211 = {s["team_id"]: s for s in snapshots if s["as_of_date"] == "2022-11-19"}
    by_conf = Counter(conf_of.get(t) or "none" for t in snap_2211)
    lines.append("\nTeams covered at 2022-11-19, by confederation:")
    for c, n in by_conf.most_common():
        lines.append(f"  {c:<10} {n}")

    top20 = sorted(snap_2211.values(), key=lambda s: s["total_value_eur"], reverse=True)[:20]
    lines.append("\nTop 20 teams by 2022-11-19 squad value (top-26 talent pool):")
    for i, s in enumerate(top20, 1):
        lines.append(f"  {i:>2}. {s['team_id']:<20} EUR {s['total_value_eur']/1e6:8.1f}m  (n={s['n_players']})")

    top20_ids = [s["team_id"] for s in top20]
    missing = [t for t in SANITY_TEAMS if t not in top20_ids]
    lines.append("")
    if missing:
        lines.append(f"!! SANITY FLAG: expected near-top teams NOT in top 20: {missing}")
        lines.append("   -> citizenship map or date filter is likely WRONG; investigate.")
    else:
        ranks = {t: top20_ids.index(t) + 1 for t in SANITY_TEAMS}
        lines.append(f"SANITY OK: {ranks} all in top 20.")

    report = "\n".join(lines)
    print("\n" + report)
    cov = os.path.join(ROOT, "data", "squad_values_coverage.txt")
    open(cov, "w", encoding="utf-8").write(report + "\n")
    print(f"\nwrote {cov}")


if __name__ == "__main__":
    main()
