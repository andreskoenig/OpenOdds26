"""PHASE 2: build config/tournament_config_2026.json from the real 5-Dec-2025 draw.

Groups + bracket are encoded from cross-checked Wikipedia data (main article,
per-group pages, knockout-stage page). Team NAMES are resolved here to canonical
team_ids in data/teams.json. A HARD GATE asserts structure; on any failure it
STOPS and reports (no guessing).

Slot grammar consumed by simulate.py (Phase 3):
  "1A"/"2B"  = group A winner / group B runner-up (by standings)
  "T:A,B,C,D,F" = a best-third allocated from one of those eligible groups
  "W:M74"/"L:M101" = winner / loser of a prior match
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "config", "tournament_config_2026.json")

# --- 12 groups in seeded positional order (hosts in position 1) -------------
GROUPS_BY_NAME = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}
HOST_NAMES = {"A": "Mexico", "B": "Canada", "D": "United States"}  # position 1

# --- official R32 bracket (matches 73-88); third slots carry eligible groups -
ROUND_OF_32 = [
    ("M73", "2A", "2B"),
    ("M74", "1E", "T:A,B,C,D,F"),
    ("M75", "1F", "2C"),
    ("M76", "1C", "2F"),
    ("M77", "1I", "T:C,D,F,G,H"),
    ("M78", "2E", "2I"),
    ("M79", "1A", "T:C,E,F,H,I"),
    ("M80", "1L", "T:E,H,I,J,K"),
    ("M81", "1D", "T:B,E,F,I,J"),
    ("M82", "1G", "T:A,E,H,I,J"),
    ("M83", "2K", "2L"),
    ("M84", "1H", "2J"),
    ("M85", "1B", "T:E,F,G,I,J"),
    ("M86", "1J", "2H"),
    ("M87", "1K", "T:D,E,I,J,L"),
    ("M88", "2D", "2G"),
]
ROUND_OF_16 = [("M89", "W:M74", "W:M77"), ("M90", "W:M73", "W:M75"),
               ("M91", "W:M76", "W:M78"), ("M92", "W:M79", "W:M80"),
               ("M93", "W:M83", "W:M84"), ("M94", "W:M81", "W:M82"),
               ("M95", "W:M86", "W:M88"), ("M96", "W:M85", "W:M87")]
QUARTER_FINALS = [("M97", "W:M89", "W:M90"), ("M98", "W:M93", "W:M94"),
                  ("M99", "W:M91", "W:M92"), ("M100", "W:M95", "W:M96")]
SEMI_FINALS = [("M101", "W:M97", "W:M98"), ("M102", "W:M99", "W:M100")]
THIRD_PLACE = ("M103", "L:M101", "L:M102")
FINAL = ("M104", "W:M101", "W:M102")

KEY_TEAMS = ["spain", "argentina", "france", "england", "brazil"]


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _matches(rows):
    return [{"match": m, "home": h, "away": a} for m, h, a in rows]


def main():
    teams = json.load(open(os.path.join(ROOT, "data", "teams.json"), encoding="utf-8"))
    id_set = {t["team_id"] for t in teams}
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(name):
        return lookup.get(name.lower()) or lookup.get(_slug(name))

    failures = []
    groups_ids = {}
    unresolved = []
    for g, names in GROUPS_BY_NAME.items():
        ids = []
        for nm in names:
            tid = resolve(nm)
            if tid is None or tid not in id_set:
                unresolved.append((g, nm))
                ids.append(None)
            else:
                ids.append(tid)
        groups_ids[g] = ids

    host_ids = []
    for g, nm in HOST_NAMES.items():
        tid = resolve(nm)
        host_ids.append(tid)
        if groups_ids.get(g, [None])[0] != tid:
            failures.append(f"host {nm} not in {g}1 (got {groups_ids.get(g, ['?'])[0]})")

    all_ids = [t for ids in groups_ids.values() for t in ids if t]
    # ---- HARD GATE ----
    if unresolved:
        failures.append(f"unresolved team names: {unresolved}")
    if len(all_ids) != 48 or len(set(all_ids)) != 48:
        failures.append(f"expected 48 unique team_ids, got {len(all_ids)} ({len(set(all_ids))} unique)")
    if len(GROUPS_BY_NAME) != 12 or any(len(v) != 4 for v in GROUPS_BY_NAME.values()):
        failures.append("not 12 groups of 4")
    for kt in KEY_TEAMS:
        if kt not in set(all_ids):
            failures.append(f"key team missing: {kt}")
    expected_hosts = {"mexico", "canada", "united_states"}
    if set(host_ids) != expected_hosts:
        failures.append(f"host ids {host_ids} != {expected_hosts}")

    if failures:
        print("HARD GATE FAILED — NOT writing config:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    config = {
        "_comment": ("2026 FIFA World Cup (USA/Canada/Mexico). 48 teams, 12 groups "
                     "A-L of 4; top 2 + 8 best thirds -> Round of 32. Groups/bracket "
                     "from the 5-Dec-2025 draw (Wikipedia, cross-checked). Team values "
                     "are canonical slug team_ids from data/teams.json. Hosts (host flag "
                     "only): Mexico A1, Canada B1, USA D1."),
        "edition": "2026",
        "host_team_ids": ["mexico", "canada", "united_states"],
        "host_cities": ["mexico", "canada", "united_states"],
        "advance_per_group": 2,
        "best_thirds": 8,
        "best_thirds_rule": {
            "count": 8,
            "note": ("8 of 12 third-placed teams advance; each R32 third-place slot "
                     "(home/away 'T:<groups>') is open only to thirds from the listed "
                     "groups, and a third never meets its own group winner. The full "
                     "FIFA Annex C combination table (495 rows) is not transcribed; "
                     "simulate.py assigns the 8 qualifying thirds to these slots by a "
                     "deterministic constraint-respecting matching."),
        },
        "groups": groups_ids,
        "tiebreak_rules": {
            "_comment": "Official 2026 group ranking order.",
            "order": ["points", "goal_difference", "goals_scored",
                      "head_to_head_points", "head_to_head_goal_difference",
                      "head_to_head_goals_scored", "fair_play_points", "drawing_of_lots"],
            "unsimulatable_resolution": "seeded_random",
        },
        "schedule": [],
        "knockout_bracket": {
            "_comment": ("Official 2026 bracket (matches 73-104), verified twice vs the "
                         "Wikipedia knockout-stage page. Slots: 1X/2X = group winner/"
                         "runner-up; 'T:...' = allocated best-third; 'W:Mxx'/'L:Mxx' = "
                         "winner/loser of a prior match."),
            "round_of_32": _matches(ROUND_OF_32),
            "round_of_16": _matches(ROUND_OF_16),
            "quarter_finals": _matches(QUARTER_FINALS),
            "semi_finals": _matches(SEMI_FINALS),
            "third_place": {"match": THIRD_PLACE[0], "home": THIRD_PLACE[1], "away": THIRD_PLACE[2]},
            "final": {"match": FINAL[0], "home": FINAL[1], "away": FINAL[2]},
        },
    }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("HARD GATE PASSED.")
    print(f"  48 unique teams across 12 groups of 4; hosts mexico/canada/united_states in A1/B1/D1.")
    print(f"  key teams present: {KEY_TEAMS}")
    for g in GROUPS_BY_NAME:
        print(f"  Group {g}: {groups_ids[g]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
