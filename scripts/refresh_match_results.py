"""PHASE 1a: refresh match_results by MERGING new martj42 rows onto the existing file.

Every existing row is preserved byte-for-byte (key order untouched), protecting the
validated 2022 baseline AND the match_odds join. Only matches NOT already present
(by (date, home_team_id, away_team_id)) are appended, with new match_ids continuing
the index. Team names resolve to the EXISTING canonical team_ids in data/teams.json.

Deterministic. Run, then run the c_v=0 WC2022 guard test (must report 1.0235).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import urllib.request
from datetime import date

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
BASELINE_CUTOFF = "2022-11-19"


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams = _load("data/teams.json")
    existing = _load("data/match_results.json")

    # Resolver from existing canonical team_ids (martj42 spellings + aliases + slugs).
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(name):
        return lookup.get(name.lower()) or lookup.get(_slug(name)) or _slug(name)

    existing_keys = {(r["date"], r["home_team_id"], r["away_team_id"]) for r in existing}
    max_idx = 0
    for r in existing:
        try:
            max_idx = max(max_idx, int(str(r["match_id"]).split("_", 1)[0]))
        except ValueError:
            pass
    existing_max_date = max(r["date"] for r in existing)
    print(f"existing rows: {len(existing)} | max index {max_idx} | max date {existing_max_date}",
          flush=True)

    print(f"downloading {URL} ...", flush=True)
    req = urllib.request.Request(URL, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read().decode("utf-8", "replace")
    rows = list(csv.DictReader(io.StringIO(raw)))
    print(f"fresh rows: {len(rows)}", flush=True)

    new_rows = []
    new_teams = set()
    idx = max_idx
    for row in rows:
        hs, as_ = row.get("home_score", "").strip(), row.get("away_score", "").strip()
        if not hs or not as_:
            continue
        try:
            hg, ag = int(hs), int(as_)
        except ValueError:
            continue
        d = row["date"].strip()
        hid = resolve(row["home_team"].strip())
        aid = resolve(row["away_team"].strip())
        if hid not in {t["team_id"] for t in teams}:
            new_teams.add(row["home_team"].strip())
        if aid not in {t["team_id"] for t in teams}:
            new_teams.add(row["away_team"].strip())
        key = (d, hid, aid)
        if key in existing_keys:
            continue
        idx += 1
        new_rows.append({
            "match_id": f"{idx:05d}_{d}_{hid}_{aid}",
            "date": d,
            "home_team_id": hid,
            "away_team_id": aid,
            "venue_country": row.get("country", "").strip(),
            "neutral": row.get("neutral", "FALSE").strip().upper() == "TRUE",
            "competition": row.get("tournament", "").strip(),
            "home_goals": hg,
            "away_goals": ag,
        })
        existing_keys.add(key)

    merged = existing + new_rows

    # Byte-for-byte guard: confirm the <= baseline-cutoff slice is unchanged.
    pre_before = [r for r in existing if r["date"] <= BASELINE_CUTOFF]
    pre_after = [r for r in merged if r["date"] <= BASELINE_CUTOFF]
    assert pre_before == pre_after, "FATAL: baseline-cutoff rows changed!"

    out = os.path.join(ROOT, "data", "match_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nappended {len(new_rows)} new matches | merged total {len(merged)}")
    if new_rows:
        nd = [r["date"] for r in new_rows]
        print(f"new match date range: {min(nd)} .. {max(nd)}")
        # show the most recent additions
        for r in sorted(new_rows, key=lambda x: x["date"])[-8:]:
            print(f"  {r['date']}  {r['home_team_id']} {r['home_goals']}-{r['away_goals']} {r['away_team_id']}  ({r['competition']})")
    if new_teams:
        print(f"unresolved/new team names (made fresh slug ids): {sorted(new_teams)}")
    print(f"baseline rows (<= {BASELINE_CUTOFF}) preserved byte-for-byte: {len(pre_after)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
