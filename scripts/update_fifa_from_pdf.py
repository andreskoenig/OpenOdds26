"""Parse the official FIFA ranking PDF and append a current snapshot to fifa_ratings.

Robust parse: FIFA ranks are strictly sequential (1,2,3,...) and points strictly
descending, which disambiguates the rank lines from stray digits in the messy PDF
layout. Validates monotonic points + 48-WC coverage before writing.

Usage: python scripts/update_fifa_from_pdf.py [--write]   (default = dry run)
"""

from __future__ import annotations

import json
import os
import re
import sys

import pypdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = r"C:\Users\andres.koenig\Downloads\FIFA_Coca-Cola Men's World Ranking.pdf"
SNAPSHOT_DATE = "2026-06-04"

# FIFA-spelling -> our team_id (extends data resolution for known mismatches).
FIFA_OVERRIDES = {
    "korea republic": "south_korea", "korea dpr": "north_korea",
    "ir iran": "iran", "usa": "united_states", "china pr": "china",
    "congo dr": "dr_congo", "cabo verde": "cape_verde", "cape verde": "cape_verde",
    "czechia": "czech_republic", "turkiye": "turkey", "turkey": "turkey",
    "republic of ireland": "republic_of_ireland",
    "chinese taipei": "chinese_taipei",
    "north macedonia": "north_macedonia", "the gambia": "gambia",
}


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def parse_pdf():
    reader = pypdf.PdfReader(PDF)
    lines = []
    for pg in reader.pages:
        lines.extend(pg.extract_text().split("\n"))

    entries = []           # (rank, raw_team_name, points)
    expected = 1
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^(\d{1,3})(?:\s+\d+)?$", line)
        if m and int(m.group(1)) == expected:
            # team name = next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            team = lines[j].strip() if j < len(lines) else ""
            team = re.sub(r"\s+\d{3,4}\.\d{2}.*$", "", team).strip()  # drop same-line points
            # points = first NNNN.NN within the next few lines of this block
            pts = None
            for k in range(j, min(j + 9, len(lines))):
                pm = re.search(r"(\d{3,4}\.\d{2})", lines[k])
                if pm:
                    pts = float(pm.group(1))
                    break
            entries.append((expected, team, pts))
            expected += 1
            i = j + 1
        else:
            i += 1
    return entries


def main():
    write = "--write" in sys.argv
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json"), encoding="utf-8"))
    id_set = {t["team_id"] for t in teams}
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(raw):
        raw = re.sub(r"^[^A-Za-z]+", "", raw).strip()  # drop leading flag/space
        low = raw.lower()
        if low in FIFA_OVERRIDES and FIFA_OVERRIDES[low] in id_set:
            return FIFA_OVERRIDES[low]
        # fuzzy for mis-decoded accents in the WC field
        if "ivoire" in low:
            return "ivory_coast"
        if "rkiye" in low or low.startswith("t") and "rki" in low:
            return "turkey"
        if "cura" in low:
            return "curacao"
        return lookup.get(low) or lookup.get(_slug(raw))

    entries = parse_pdf()
    print(f"parsed entries: {len(entries)}")
    # monotonic points check
    pts_seq = [p for _, _, p in entries if p is not None]
    mono = all(pts_seq[i] >= pts_seq[i + 1] for i in range(len(pts_seq) - 1))
    n_nopts = sum(1 for _, _, p in entries if p is None)
    print(f"points present: {len(pts_seq)} | missing points: {n_nopts} | strictly descending: {mono}")
    print("\nfirst 12:")
    for r, nm, p in entries[:12]:
        print(f"  {r:>3} {nm:<26} {p}")
    print("last 6:")
    for r, nm, p in entries[-6:]:
        print(f"  {r:>3} {nm:<26} {p}")

    # resolve
    rows, unresolved = [], []
    for r, nm, p in entries:
        if p is None:
            continue
        tid = resolve(nm)
        if tid is None or tid not in id_set:
            unresolved.append(nm)
            continue
        rows.append({"team_id": tid, "as_of_date": SNAPSHOT_DATE, "fifa_points": p, "fifa_rank": r})
    # dedupe (keep best rank per team_id)
    seen = {}
    for row in rows:
        if row["team_id"] not in seen:
            seen[row["team_id"]] = row
    rows = list(seen.values())

    cfg = json.load(open(os.path.join(ROOT, "config", "tournament_config_2026.json"), encoding="utf-8"))
    wc48 = {t for g in cfg["groups"].values() for t in g}
    covered = {row["team_id"] for row in rows}
    missing = sorted(wc48 - covered)
    print(f"\nresolved rows: {len(rows)} | unresolved names: {len(set(unresolved))}")
    if unresolved:
        print("  unresolved sample:", sorted(set(unresolved))[:25])
    print(f"WC2026 teams covered: {48 - len(missing)}/48")
    if missing:
        print("  MISSING:", missing)
    # show the 48 WC teams' new points
    print("\nWC48 new FIFA points:")
    by_id = {row["team_id"]: row for row in rows}
    for tid in sorted(wc48, key=lambda t: -(by_id[t]["fifa_points"] if t in by_id else 0)):
        if tid in by_id:
            print(f"  {tid:<24} rank {by_id[tid]['fifa_rank']:>3}  {by_id[tid]['fifa_points']}")

    if not write:
        print("\n[DRY RUN] re-run with --write to append to data/fifa_ratings.json")
        return
    if missing or not mono:
        print("\nNOT WRITING: coverage/monotonicity check failed.")
        sys.exit(1)
    path = os.path.join(ROOT, "data", "fifa_ratings.json")
    existing = [x for x in json.load(open(path, encoding="utf-8")) if x["as_of_date"] != SNAPSHOT_DATE]
    merged = existing + rows
    json.dump(merged, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nWROTE {path}: appended {len(rows)} rows for {SNAPSHOT_DATE} (total {len(merged)})")


if __name__ == "__main__":
    main()
