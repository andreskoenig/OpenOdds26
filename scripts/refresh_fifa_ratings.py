"""PHASE 1b: append the most-recent free FIFA ranking snapshot to fifa_ratings.json.

Existing file stops at 2022-10 (2022 baseline). FIFA.com is anti-bot and Kaggle
needs auth, so the best free DIRECT source is Dato-Futbol/fifa-ranking (scraped
from fifa.com, through 2024-09). We append its latest snapshot as the current
prior (rank derived by points), preserving all existing rows, and report the
snapshot date + any staleness clearly. Verifies all 48 WC2026 teams are covered.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}

# (label, url, date_col, team_col, points_col, rank_col|None)  most-recent wins.
SOURCES = [
    ("Dato-Futbol/fifa-ranking",
     "https://raw.githubusercontent.com/Dato-Futbol/fifa-ranking/master/ranking_fifa_historical.csv",
     "date", "team", "total_points", None),
    ("hericlibong/FifaRankingScraper",
     "https://raw.githubusercontent.com/hericlibong/FifaRankingScraper/main/"
     "historicalmenranking/historicalmenranking/spiders/data.csv",
     "date", "country", "totalPoints", "rank"),
]


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams = _load("data/teams.json")
    id_set = {t["team_id"] for t in teams}
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    # FIFA/Dato-Futbol spellings not stored in teams.json aliases.
    fifa_overrides = {
        "cabo verde": "cape_verde", "czechia": "czech_republic",
        "congo dr": "dr_congo", "ir iran": "iran",
        "côte d'ivoire": "ivory_coast", "cote d'ivoire": "ivory_coast",
        "aotearoa new zealand": "new_zealand", "korea republic": "south_korea",
        "korea dpr": "north_korea", "türkiye": "turkey", "turkiye": "turkey",
        "usa": "united_states", "china pr": "china", "chinese taipei": "chinese_taipei",
        "st kitts and nevis": "saint_kitts_and_nevis",
        "st lucia": "saint_lucia",
        "st vincent and the grenadines": "saint_vincent_and_the_grenadines",
    }

    def resolve(name):
        name = re.sub(r"\s*\(unranked\)\s*$", "", name).strip()
        if name.lower() in fifa_overrides and fifa_overrides[name.lower()] in id_set:
            return fifa_overrides[name.lower()]
        return lookup.get(name.lower()) or lookup.get(_slug(name))

    cfg = _load("config/tournament_config_2026.json")
    wc48 = {t for g in cfg["groups"].values() for t in g}

    best = None  # (max_date, label, rows, dcol, tcol, pcol, rcol)
    for label, url, dcol, tcol, pcol, rcol in SOURCES:
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                rows = list(csv.DictReader(io.StringIO(r.read().decode("utf-8", "replace"))))
        except Exception as e:
            print(f"source {label}: failed ({type(e).__name__})", flush=True)
            continue
        ds = [row.get(dcol, "").strip() for row in rows if row.get(dcol, "").strip()]
        mx = max(ds) if ds else ""
        print(f"source {label}: rows={len(rows)} max date {mx}", flush=True)
        if best is None or mx > best[0]:
            best = (mx, label, rows, dcol, tcol, pcol, rcol)

    if best is None:
        print("FATAL: no FIFA source reachable", file=sys.stderr)
        sys.exit(2)

    latest, label, rows, dcol, tcol, pcol, rcol = best
    snap = []
    unresolved = set()
    for row in rows:
        if row.get(dcol, "").strip() != latest:
            continue
        tid = resolve(row.get(tcol, "").strip())
        try:
            pts = float(row.get(pcol, "").strip())
        except (ValueError, AttributeError):
            continue
        if tid is None or tid not in id_set:
            unresolved.add(row.get(tcol, "").strip())
            continue
        snap.append((tid, pts, row.get(rcol, "").strip() if rcol else None))

    # Derive rank by points (desc) if the source has no rank column.
    snap.sort(key=lambda x: -x[1])
    new_rows = []
    for i, (tid, pts, raw_rank) in enumerate(snap, 1):
        try:
            rk = int(raw_rank) if raw_rank else i
        except ValueError:
            rk = i
        new_rows.append({"team_id": tid, "as_of_date": latest,
                         "fifa_points": pts, "fifa_rank": rk})

    # Idempotent: drop any existing rows for this snapshot date, then add fresh
    # (keeps ranks/points internally consistent across re-runs).
    existing = [r for r in _load("data/fifa_ratings.json") if r["as_of_date"] != latest]
    appended = new_rows
    merged = existing + appended
    out = os.path.join(ROOT, "data", "fifa_ratings.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    covered = {r["team_id"] for r in new_rows}
    missing = sorted(wc48 - covered)
    print(f"\nchosen source: {label} | snapshot date: {latest}")
    if latest < "2025-01-01":
        print(f"  NOTE: latest free FIFA data is {latest} (~not 2026). FIFA.com is "
              f"anti-bot and Kaggle needs auth; this is the most recent free DIRECT "
              f"source. z_fifa is one prior of several (squad-value is current 2026).")
    print(f"appended {len(appended)} rows; total fifa_ratings rows now {len(merged)}")
    print(f"WC2026 teams with a current FIFA value: {48 - len(missing)}/48")
    if missing:
        print(f"  MISSING: {missing}")
    if unresolved:
        print(f"  unresolved source team names (sample): {sorted(unresolved)[:15]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
