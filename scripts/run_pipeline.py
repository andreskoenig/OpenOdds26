"""On-demand full re-run pipeline for the 2026 World Cup forecast.

Re-extracts the FAST-CHANGING free data, then re-runs the ENTIRE prediction
chain on it. Transfermarkt squad values are NEVER re-fetched (slow-moving talent
pool, cached in data/squad_values.json).

STAGES
  fetch   Polymarket winner odds      (fetch_polymarket_winner.py)   [default ON]
          match results (martj42)     (refresh_match_results.py)     [default ON]
          FIFA ranking points         (refresh_fifa_ratings.py)      [default OFF*]
          Transfermarkt squad values  -- CACHED, never re-fetched
  compute de-path Polymarket          (depath_polymarket.py)
          20k tournament forecast     (run_forecast_2026.py)
          group-stage by-date preds   (predict_groupstage_schedule_2026.py)
          group-stage CSV             (make_groupstage_csv.py)

  * FIFA auto-refresh is OFF by default: the authoritative 2026 points were
    loaded from the official FIFA PDF (update_fifa_from_pdf.py), while the free
    auto-source only reaches 2024-09. Enable with --fetch-fifa if you want it.

USAGE
  python scripts/run_pipeline.py                 # fetch (PM + results) + full re-run
  python scripts/run_pipeline.py --quick         # same, but cheap smoke-test sims
  python scripts/run_pipeline.py --no-fetch       # recompute on existing data only
  python scripts/run_pipeline.py --fetch-fifa     # also refresh FIFA from free source
  python scripts/run_pipeline.py --skip-results   # don't refresh match results
  python scripts/run_pipeline.py --dry-run        # print the plan, run nothing

Writes data/pipeline_run.json (provenance manifest + final top-8).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # run children under the same interpreter/venv as the pipeline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _mtime(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    return datetime.fromtimestamp(os.path.getmtime(p)).isoformat(timespec="seconds")


def run_stage(name, script, env_extra=None, script_args=()):
    """Run one child script live (inherited stdout); fail fast on nonzero exit."""
    print("\n" + "=" * 78)
    print(f"STAGE: {name}   ->   scripts/{script} {' '.join(script_args)}".rstrip())
    print("=" * 78, flush=True)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    rc = subprocess.call([PY, "-u", os.path.join(ROOT, "scripts", script), *script_args],
                         cwd=ROOT, env=env)
    dt = time.time() - t0
    status = "ok" if rc == 0 else f"FAILED (rc={rc})"
    print(f"\n[{name}] {status} in {dt:.0f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"\nPIPELINE ABORTED at stage '{name}' (exit {rc}). "
                         f"Earlier stages already wrote their outputs.")
    return {"stage": name, "script": script, "seconds": round(dt, 1)}


def build_plan(args):
    plan = []  # (name, script, env_extra, script_args)
    if not args.no_fetch:
        plan.append(("fetch: Polymarket winner odds", "fetch_polymarket_winner.py", None, ()))
        if not args.skip_results:
            # allow upstream-added OLD matches; existing baseline rows still guarded.
            plan.append(("fetch: match results (martj42)", "refresh_match_results.py",
                         None, ("--allow-baseline-additions",)))
        if args.fetch_fifa:
            plan.append(("fetch: FIFA ranking points", "refresh_fifa_ratings.py", None, ()))
    env = {"WC_FAST": "1", "WC_NSIMS": "2000"} if args.quick else None
    plan.append(("compute: de-path Polymarket", "depath_polymarket.py", env, ()))
    plan.append(("compute: 20k forecast", "run_forecast_2026.py", env, ()))
    plan.append(("compute: group-stage by-date", "predict_groupstage_schedule_2026.py", None, ()))
    plan.append(("compute: group-stage CSV", "make_groupstage_csv.py", None, ()))
    return plan


def provenance():
    """Snapshot of every data input/output (mtime + key stats) for the manifest."""
    prov = {}
    try:
        m = _load("data/match_results.json")
        prov["match_results"] = {"mtime": _mtime("data/match_results.json"),
                                 "rows": len(m), "latest": max(x["date"] for x in m)}
    except Exception as e:
        prov["match_results"] = {"error": str(e)}
    try:
        f = _load("data/fifa_ratings.json")
        prov["fifa_ratings"] = {"mtime": _mtime("data/fifa_ratings.json"),
                                "rows": len(f), "latest_snapshot": max(x["as_of_date"] for x in f)}
    except Exception as e:
        prov["fifa_ratings"] = {"error": str(e)}
    try:
        s = _load("data/squad_values.json")
        prov["squad_values_CACHED"] = {"mtime": _mtime("data/squad_values.json"),
                                       "rows": len(s),
                                       "note": "never re-fetched (Transfermarkt)"}
    except Exception as e:
        prov["squad_values_CACHED"] = {"error": str(e)}
    try:
        pm = _load("data/polymarket_winner_2026.json")
        prov["polymarket"] = {"mtime": _mtime("data/polymarket_winner_2026.json"),
                              "as_of": pm.get("as_of"), "n_priced": pm.get("n_teams_priced"),
                              "overround": round(pm.get("raw_price_sum", 0) - 1, 4)}
    except Exception as e:
        prov["polymarket"] = {"error": str(e)}
    return prov


def main():
    ap = argparse.ArgumentParser(description="Full re-run pipeline (re-fetch fast data, recompute).")
    ap.add_argument("--no-fetch", action="store_true", help="skip all fetching; recompute only")
    ap.add_argument("--skip-results", action="store_true", help="don't refresh match results")
    ap.add_argument("--fetch-fifa", action="store_true", help="also refresh FIFA (free source, ~2024-09)")
    ap.add_argument("--quick", action="store_true", help="cheap smoke-test sims (fast, not for real)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    plan = build_plan(args)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print("=" * 78)
    print("FIFA WC 2026 — FULL RE-RUN PIPELINE")
    print("=" * 78)
    print(f"started {started}" + ("   [QUICK smoke-test sims]" if args.quick else ""))
    print("Transfermarkt squad values: CACHED (never re-fetched).")
    if args.no_fetch:
        print("Fetch: SKIPPED (--no-fetch) — recomputing on existing data.")
    if not args.fetch_fifa and not args.no_fetch:
        print("FIFA: kept as-is (official PDF snapshot); use --fetch-fifa to refresh.")
    print("\nPLAN:")
    for i, (name, script, _e, _a) in enumerate(plan, 1):
        print(f"  {i}. {name:<34} ({script})")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    timings = [run_stage(name, script, env, sargs) for (name, script, env, sargs) in plan]
    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- summary + manifest ----------------------------------------------
    prov = provenance()
    top = []
    try:
        pw = _load("data/forecast_2026.json")["p_win"]
        nm = {t["team_id"]: t["canonical_name"] for t in _load("data/teams.json")}
        top = [(nm.get(t, t), round(pw[t] * 100, 1))
               for t in sorted(pw, key=lambda x: -pw[x])[:8]]
    except Exception as e:
        top = [("error", str(e))]

    manifest = {"started_utc": started, "finished_utc": finished, "quick": args.quick,
                "args": vars(args), "stages": timings, "provenance": prov,
                "forecast_top8": top}
    with open(os.path.join(ROOT, "data", "pipeline_run.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print("=" * 78)
    total = sum(t["seconds"] for t in timings)
    print(f"total {total:.0f}s across {len(timings)} stages")
    print(f"match_results: {prov['match_results'].get('rows')} rows "
          f"(latest {prov['match_results'].get('latest')})")
    print(f"polymarket: {prov['polymarket'].get('n_priced')} priced, "
          f"overround {prov['polymarket'].get('overround')}, snapshot {prov['polymarket'].get('as_of')}")
    print(f"squad_values: CACHED ({prov['squad_values_CACHED'].get('rows')} rows, not re-fetched)")
    print("\nfinal P(win) top 8:")
    for nm_, p in top:
        print(f"  {nm_:<16}{p:>6}%")
    print(f"\nwrote data/pipeline_run.json")


if __name__ == "__main__":
    main()
