"""Report xG coverage (StatsBomb) vs goals, for WC2022 and WC2026 squads.

xG exists only for the 6 senior international tournaments StatsBomb has released
(WC2018/2022, Euro2020/2024, Copa2024, AFCON2023). This quantifies how much of
each squad's match history (in a 10-year window) is xG-covered vs goals-only --
the context for whether xG can move the model.

Windows (10 years back from each tournament's as-of):
  WC2026 teams: 2016-06-10 .. 2026-06-10
  WC2022 teams: 2012-11-19 .. 2022-11-19   (the validation window)
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    matches = _load("data/match_results.json")
    txg = _load("data/team_xg.json")["rows"]
    teams = {t["team_id"]: t["canonical_name"] for t in _load("data/teams.json")}
    wc22 = {t for g in _load("config/tournament_config_2022.json")["groups"].values() for t in g}
    wc26 = {t for g in _load("config/tournament_config_2026.json")["groups"].values() for t in g}

    mdate = {m["match_id"]: m["date"] for m in matches}
    xg_mids = {r["match_id"] for r in txg}            # match_ids that have xG
    xg_dates = sorted(mdate[m] for m in xg_mids if m in mdate)

    # per-team list of (date, match_id) from goals history
    by_team = defaultdict(list)
    for m in matches:
        for tid in (m["home_team_id"], m["away_team_id"]):
            by_team[tid].append((m["date"], m["match_id"]))

    def cov(team_ids, start, end):
        """Per-team (goals_n, xg_n) within [start, end); plus totals."""
        rows = {}
        gt = xt = 0
        for tid in team_ids:
            g = [(d, mid) for (d, mid) in by_team.get(tid, []) if start <= d < end]
            x = [mid for (d, mid) in g if mid in xg_mids]
            rows[tid] = (len(g), len(x))
            gt += len(g)
            xt += len(x)
        return rows, gt, xt

    print("=" * 68)
    print("xG COVERAGE (StatsBomb) vs GOALS")
    print("=" * 68)
    print(f"team_xg rows: {len(txg)} | distinct xG matches joined: {len(xg_mids)}")
    if xg_dates:
        print(f"xG date range: {xg_dates[0]} .. {xg_dates[-1]}")
        pre22 = sum(1 for d in xg_dates if d < '2022-11-19')
        print(f"xG matches BEFORE WC2022 cutoff (usable in the 2022 validation): {pre22}")
        print(f"xG matches after: {len(xg_dates) - pre22}")

    for label, ids, start, end in [
        ("WC2026 squads (48)", wc26, "2016-06-10", "2026-06-10"),
        ("WC2022 squads (32)", wc22, "2012-11-19", "2022-11-19"),
    ]:
        rows, gt, xt = cov(ids, start, end)
        print("\n" + "-" * 68)
        print(f"{label}   window {start} .. {end}")
        print("-" * 68)
        pct = 100 * xt / gt if gt else 0
        print(f"  AGGREGATE: {xt} xG matches / {gt} goal matches = {pct:.1f}% xG coverage")
        print(f"  {'team':<18}{'goals':>7}{'xG':>6}{'cov%':>7}")
        for tid in sorted(ids, key=lambda t: -(rows[t][1] / rows[t][0] if rows[t][0] else 0)):
            g, x = rows[tid]
            c = 100 * x / g if g else 0
            print(f"  {teams.get(tid, tid):<18}{g:>7}{x:>6}{c:>6.0f}%")


if __name__ == "__main__":
    main()
