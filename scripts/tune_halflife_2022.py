"""Choose the time-decay half-life by OUT-OF-SAMPLE log-loss (not by guessing).

We sweep xi (the SPEC-4 decay) over a fine grid of half-lives and score each on
TWO leakage-free validation folds, then pick the half-life by the one-standard-
error rule (smallest/most-recent half-life whose pooled log-loss is within 1 SE
of the best). Truncation is then DERIVED from the chosen half-life as the age at
which a match's weight drops below 1% (~6.6 half-lives) -- not a second guess.

Folds (both score predict_match on real fixtures; no simulation needed):
  A) WC2022     -- one fit as-of the day before the opener, score the 64 games.
  B) rolling    -- quarterly origins through 2021-2022; each fit predicts the
                   NEXT quarter's non-WC internationals (rolling-origin CV).

Engine under test = the shipped RESULTS engine (opponent_adjust ON, squad prior
ON, market prior OFF -- we are setting the results memory; the market prior is
added on top later and does not change the optimal results half-life).

Writes scripts/tune_halflife_results.json.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.features import build_features
from wc_model.model import fit_model, matchup_matrix, result_probs
from wc_model.schemas import Hyperparams

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIN_MATCHES = 50
DAY = 365.25
EPS = 1e-12
RESULTS_PATH = os.path.join(ROOT, "scripts", "tune_halflife_results.json")

# Half-lives (years) to test -> xi = ln2 / (years * 365.25).
HALF_LIVES = [1.0, 1.5, 2.0, 3.0, 5.0]

# Fit window: only matches within this many years before the as-of are fed to the
# fit (older matches carry <6% weight even at the longest half-life tested, so
# this is fair across the grid and cuts each fit's cost ~2x).
FIT_WINDOW_Y = 20.0

# Fold A: WC2022.
AS_OF_A = "2022-11-20"          # opener was 2022-11-20; strictly-before excludes it
WC_START, WC_END = "2022-11-20", "2022-12-18"

# Fold B: rolling origins; each predicts [origin, next_origin).
ORIGINS_B = ["2021-06-01", "2021-12-01", "2022-06-01", "2022-10-01"]
B_END = "2022-12-31"

# Shipped results-engine prior weights (xi is what we vary).
BASE = dict(lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,
            theta=0.0, kappa=0.0, c_v=0.1, c_m=0.0, opponent_adjust=True,
            blend_weight=0.7, n_recent=10)


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _outcome_idx(m):
    if m["home_goals"] > m["away_goals"]:
        return 0
    if m["home_goals"] < m["away_goals"]:
        return 2
    return 1


def _eligible(matches, cut):
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    return {t for t, c in cnt.items() if c >= MIN_MATCHES}


def _window(matches, as_of_str, years):
    """Matches within `years` before `as_of` (the fit window / truncation)."""
    cut = date.fromisoformat(as_of_str)
    lo = cut - timedelta(days=years * DAY)
    return [m for m in matches if lo <= date.fromisoformat(m["date"]) < cut]


def _fit(as_of, teams, matches, fifa, squad, xi):
    hp = Hyperparams(xi=xi, **BASE)
    feats = build_features(as_of, teams, matches, fifa, [], [], squad_values=squad,
                           market_probs=None, **hp.feature_kwargs())
    return fit_model(as_of, teams, matches, feats, **hp.fit_kwargs()), hp


def _losses(params, hp, val):
    """Per-match 1X2 log-losses for a fitted model over fixtures `val`."""
    out = []
    for m in val:
        hf = not m.get("neutral", False)
        P = matchup_matrix(params, m["home_team_id"], m["away_team_id"], hf, kappa=hp.kappa)
        p = result_probs(P)
        out.append(-math.log(max(p[_outcome_idx(m)], EPS)))
    return out


def _agg(losses):
    n = len(losses)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "se": float("nan")}
    mean = sum(losses) / n
    var = sum((x - mean) ** 2 for x in losses) / n
    return {"n": n, "mean": mean, "se": math.sqrt(var / n)}


def main():
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    tn = {t["team_id"]: t["canonical_name"] for t in teams_all}

    # --- Fold A setup: WC2022 ---------------------------------------------
    cutA = date.fromisoformat(AS_OF_A)
    eligA = _eligible(matches, cutA)
    teamsA = [t for t in teams_all if t["team_id"] in eligA]
    matchesA_fit = _window(matches, AS_OF_A, FIT_WINDOW_Y)
    valA = [m for m in matches
            if m["competition"] == "FIFA World Cup"
            and WC_START <= m["date"] <= WC_END
            and m["home_team_id"] in eligA and m["away_team_id"] in eligA]

    # --- Fold B setup: rolling non-WC internationals ----------------------
    foldB = []  # (origin_str, teams_list, fit_matches, val_list)
    for i, o in enumerate(ORIGINS_B):
        nxt = ORIGINS_B[i + 1] if i + 1 < len(ORIGINS_B) else B_END
        cut = date.fromisoformat(o)
        elig = _eligible(matches, cut)
        tl = [t for t in teams_all if t["team_id"] in elig]
        mfit = _window(matches, o, FIT_WINDOW_Y)
        val = [m for m in matches
               if m["competition"] != "FIFA World Cup"
               and o <= m["date"] < nxt
               and m["home_team_id"] in elig and m["away_team_id"] in elig]
        foldB.append((o, tl, mfit, val))

    print("=" * 74)
    print("HALF-LIFE SWEEP — out-of-sample 1X2 log-loss (lower = better)")
    print("=" * 74)
    print(f"Fold A (WC2022): fit as-of {AS_OF_A}, score {len(valA)} WC games")
    nB = sum(len(v) for *_, v in foldB)
    print(f"Fold B (rolling): {len(ORIGINS_B)} quarterly origins, "
          f"{nB} non-WC internationals total")
    print(f"engine: opponent_adjust ON, squad prior ON, market prior OFF\n", flush=True)

    rows = []
    for hl in HALF_LIVES:
        xi = math.log(2.0) / (hl * DAY)
        # Fold A: single fit
        pA, hpA = _fit(AS_OF_A, teamsA, matchesA_fit, fifa, squad, xi)
        lossA = _losses(pA, hpA, valA)
        # Fold B: rolling fits
        lossB = []
        for (o, tl, mfit, val) in foldB:
            if not val:
                continue
            pB, hpB = _fit(o, tl, mfit, fifa, squad, xi)
            lossB += _losses(pB, hpB, val)
        pooled = lossA + lossB
        rows.append({"half_life_y": hl, "xi": xi,
                     "A": _agg(lossA), "B": _agg(lossB), "pooled": _agg(pooled)})
        a, b, p = rows[-1]["A"], rows[-1]["B"], rows[-1]["pooled"]
        print(f"  t½={hl:>4}y (xi={xi:.6f})  A={a['mean']:.4f}  B={b['mean']:.4f}  "
              f"pooled={p['mean']:.4f} ±{p['se']:.4f}", flush=True)
        json.dump({"half_lives": HALF_LIVES, "rows": rows}, open(RESULTS_PATH, "w"), indent=2)

    # --- selection: one-SE rule on pooled, leaning to shorter half-life ----
    best = min(rows, key=lambda r: r["pooled"]["mean"])
    thresh = best["pooled"]["mean"] + best["pooled"]["se"]
    within = [r for r in rows if r["pooled"]["mean"] <= thresh]
    chosen = min(within, key=lambda r: r["half_life_y"])  # most recent within 1 SE
    trunc_y = chosen["half_life_y"] * math.log(100.0, 2.0)  # weight < 1%

    print("\n" + "=" * 74)
    print(f"BEST pooled log-loss: t½={best['half_life_y']}y "
          f"({best['pooled']['mean']:.4f} ±{best['pooled']['se']:.4f})")
    print(f"1-SE threshold: {thresh:.4f}  -> within-1SE half-lives: "
          f"{[r['half_life_y'] for r in within]}")
    print(f"CHOSEN (most-recent within 1 SE): t½={chosen['half_life_y']}y  "
          f"xi={chosen['xi']:.6f}")
    print(f"DERIVED truncation (weight<1%, 6.64*t½): {trunc_y:.1f} years")
    print("=" * 74)
    summary = {"best": best, "chosen": chosen, "truncation_years": trunc_y,
               "one_se_threshold": thresh}
    full = {"half_lives": HALF_LIVES, "rows": rows, "summary": summary}
    json.dump(full, open(RESULTS_PATH, "w"), indent=2)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
