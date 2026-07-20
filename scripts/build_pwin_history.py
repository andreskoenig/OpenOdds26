"""Mine git history for P(win) snapshots -> data/pwin_history.json.

Every end-of-round re-run commits data/forecast_live_2026.json; its git history
is therefore a free time series of the tournament forecast. This script walks
those commits (plus the frozen pre-tournament data/forecast_2026.json as the
baseline point and the current working tree as the latest point) and writes one
compact JSON the dashboard can chart.

Snapshots are deduped per day (latest wins). P(win) is stored for every team in
each snapshot (teams absent from a snapshot are eliminated -> the frontend may
render them at 0).

Run:  python scripts/build_pwin_history.py     (after each round's forecast)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE_REL = "data/forecast_live_2026.json"
OUT = os.path.join(ROOT, "data", "pwin_history.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# date -> display label (round the forecast was generated after)
LABELS = {
    "2026-06-11": "Pre",
    "2026-06-19": "MD1",
    "2026-06-24": "MD2",
    "2026-06-28": "Groups",
    "2026-07-04": "R32",
    "2026-07-08": "R16",
    "2026-07-12": "QF",
    "2026-07-13": "QF",
    "2026-07-16": "SF",
    "2026-07-19": "Final",
    "2026-07-20": "Final",
}


def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8").stdout


def _load_str(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def snapshot_date(fc, fallback):
    """Prefer the file's own stamp (generated_at 'YYYY-MM-DD ..' or as_of)."""
    for k in ("generated_at", "as_of"):
        v = fc.get(k)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    return fallback


def main():
    snaps = {}  # date -> {"date", "label", "source", "p_win"}

    # 1) frozen pre-tournament baseline (the working-tree file IS frozen)
    with open(os.path.join(ROOT, "data", "forecast_2026.json"), encoding="utf-8") as f:
        base = json.load(f)
    d = snapshot_date(base, "2026-06-11")
    snaps[d] = {"date": d, "label": LABELS.get(d, d[5:]), "source": "pre-tournament",
                "p_win": base.get("p_win", {})}

    # 2) every committed forecast_live snapshot (no --follow: rename detection
    #    would bleed into forecast_2026.json's earlier history)
    log = _git("log", "--format=%H %cs", "--", LIVE_REL)
    for line in log.splitlines():
        if not line.strip():
            continue
        sha, cdate = line.split()
        fc = _load_str(_git("show", f"{sha}:{LIVE_REL}"))
        if not fc or "p_win" not in fc:
            continue
        d = snapshot_date(fc, cdate)
        if d < "2026-06-12":          # pre-tournament noise
            continue
        if d not in snaps:            # log is newest-first; keep the latest per day
            snaps[d] = {"date": d, "label": LABELS.get(d, d[5:]), "source": sha[:7],
                        "p_win": fc["p_win"]}

    # 3) current working tree (today's run, possibly not yet committed)
    live_path = os.path.join(ROOT, LIVE_REL)
    if os.path.exists(live_path):
        with open(live_path, encoding="utf-8") as f:
            cur = json.load(f)
        d = snapshot_date(cur, "")
        if d and "p_win" in cur:
            snaps[d] = {"date": d, "label": LABELS.get(d, d[5:]), "source": "working-tree",
                        "p_win": cur["p_win"]}

    ordered = [snaps[k] for k in sorted(snaps)]

    # round to 4 decimals to keep the file small
    for s in ordered:
        s["p_win"] = {t: round(p, 4) for t, p in s["p_win"].items()}

    # team display names from teams.json
    with open(os.path.join(ROOT, "data", "teams.json"), encoding="utf-8") as f:
        names = {t["team_id"]: t["canonical_name"] for t in json.load(f)}
    seen = {t for s in ordered for t in s["p_win"]}

    out = {"snapshots": ordered,
           "team_names": {t: names.get(t, t) for t in seen}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"pwin_history.json: {len(ordered)} snapshots "
          f"({ordered[0]['date']} .. {ordered[-1]['date']})")
    for s in ordered:
        top = sorted(s["p_win"].items(), key=lambda kv: -kv[1])[:4]
        print(f"  {s['date']} [{s['label']:>6}] " +
              "  ".join(f"{names.get(t, t)} {p*100:.1f}%" for t, p in top))


if __name__ == "__main__":
    main()
