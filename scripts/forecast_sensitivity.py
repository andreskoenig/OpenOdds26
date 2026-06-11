"""Forecast sensitivity band: how much do the JUDGMENT knobs move P(win)?

The 2026 forecast's two least-validated parameters are the market-prior weight
(c_m, chosen "moderate" by judgment) and the recency half-life (1.5y, a recency
prior picked within a statistically flat validation band whose point optimum was
5y). This script re-runs the forecast over a 3x3 grid:

    c_m  in {0.0, 0.35, 0.7}   x   t-half in {1.5y, 3y, 5y}

with truncation derived per half-life (6.64 * t-half, the weight<1% rule) and
fewer sims per cell (the band needs ~0.5pp precision, not 0.2pp). Publishes the
min-max P(win) range per team so readers see which claims are config-robust and
which are artifacts of one knob setting.

Writes data/forecast_sensitivity_2026.json. Run AFTER depath (needs the
de-pathed market file). ~30-40 min.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.pipeline import run_prediction
from wc_model.schemas import Hyperparams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AS_OF = "2026-06-10"
N_SIMS = 5000
SEED = 20260610
MIN_MATCHES = 50
DAY = 365.25

HALF_LIVES = [1.5, 3.0, 5.0]
CMS = [0.0, 0.35, 0.7]
HEADLINE = (1.5, 0.35)  # the shipped config


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    market = _load("data/polymarket_winner_2026_depathed.json")["p_market"]
    config = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams_all}

    cfg_ids = {t for g in config["groups"].values() for t in g}
    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES} | cfg_ids
    teams = [t for t in teams_all if t["team_id"] in eligible]

    print("=" * 72)
    print("FORECAST SENSITIVITY BAND — c_m x half-life grid")
    print("=" * 72)
    print(f"{len(HALF_LIVES) * len(CMS)} configs x {N_SIMS} sims | as-of {AS_OF}\n", flush=True)

    cells = {}
    for hl in HALF_LIVES:
        xi = math.log(2.0) / (hl * DAY)
        trunc = round(hl * math.log(100.0, 2.0), 1)  # weight<1% rule, ~6.64*t-half
        for cm in CMS:
            hp = Hyperparams(xi=xi, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30,
                             c_y=0.10, theta=0.0, kappa=0.0, c_v=0.1, c_m=cm,
                             blend_weight=0.7, n_recent=10, opponent_adjust=True,
                             max_history_years=trunc)
            pred = run_prediction(
                AS_OF, teams, config, matches, fifa, team_xg=[], match_odds=[],
                hyperparams=hp, squad_values=squad, market_probs=market if cm > 0 else None,
                n_sims=N_SIMS, seed=SEED,
            )
            key = f"t{hl}_cm{cm}"
            cells[key] = {"half_life_y": hl, "c_m": cm, "trunc_y": trunc,
                          "p_win": pred.p_win}
            top3 = sorted(pred.p_win, key=lambda t: -pred.p_win[t])[:3]
            print(f"  [t1/2={hl}y cm={cm}] top: "
                  + ", ".join(f"{name[t]} {pred.p_win[t]*100:.1f}%" for t in top3),
                  flush=True)

    # band per team: min / max / headline across configs
    all_ids = sorted({t for c in cells.values() for t in c["p_win"]})
    hl_key = f"t{HEADLINE[0]}_cm{HEADLINE[1]}"
    band = {}
    for t in all_ids:
        vals = [c["p_win"].get(t, 0.0) for c in cells.values()]
        band[t] = {"min": min(vals), "max": max(vals),
                   "headline": cells[hl_key]["p_win"].get(t, 0.0)}

    out = {
        "as_of": AS_OF, "n_sims_per_cell": N_SIMS, "seed": SEED,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "grid": {"half_lives_y": HALF_LIVES, "c_m": CMS},
        "headline_config": {"half_life_y": HEADLINE[0], "c_m": HEADLINE[1]},
        "cells": cells, "band": band,
        "team_names": {t: name.get(t, t) for t in all_ids},
    }
    pj = os.path.join(ROOT, "data", "forecast_sensitivity_2026.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\nBAND (top 10 by headline):")
    top = sorted(all_ids, key=lambda t: -band[t]["headline"])[:10]
    print(f"  {'team':<16}{'headline':>9}{'min':>7}{'max':>7}")
    for t in top:
        b = band[t]
        print(f"  {name.get(t, t):<16}{b['headline']*100:>8.1f}%{b['min']*100:>6.1f}%"
              f"{b['max']*100:>6.1f}%")
    print(f"\nwrote {pj}")


if __name__ == "__main__":
    main()
