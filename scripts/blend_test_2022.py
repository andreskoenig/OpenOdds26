"""Blend test: does the model add signal beyond the de-vigged Bet365 market?

Cached data only, no fetching, NO model changes. Shipped config is locked:
xi=0.0008, lambda_reg=8.0, c_a=c_d=0.30, c_x=c_y=0.10, theta=0, kappa=0, c_v=0.1.

VALIDATION (tune): the 224 non-WC internationals Aug 1 - Nov 19 2022, fit as-of
2022-08-01. WC TEST (apply once): the 64 WC2022 games, fit as-of 2022-11-19.
Sweep w in {0..1} for a linear pool and a log-linear (geometric) pool; pick
(pool, w*) by best VALIDATION log-loss; apply once to the WC games.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from datetime import date

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.devig import consensus_probs
from wc_model.evaluate import brier_score, log_loss, ranked_probability_score
from wc_model.features import build_features
from wc_model.model import fit_model, matchup_matrix, result_probs

MIN_MATCHES = 50
WS = [round(w, 1) for w in np.arange(0.0, 1.01, 0.1)]
HP = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0, c_v=0.1)
_EPS = 1e-12


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _outcome(m):
    if m["home_goals"] > m["away_goals"]:
        return "home"
    if m["home_goals"] < m["away_goals"]:
        return "away"
    return "draw"


def _linear(model, market, w):
    return tuple(w * mo + (1 - w) * ma for mo, ma in zip(model, market))


def _loglinear(model, market, w):
    raw = [max(mo, _EPS) ** w * max(ma, _EPS) ** (1 - w) for mo, ma in zip(model, market)]
    s = sum(raw)
    return tuple(r / s for r in raw)


def main():
    teams_all = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    odds = _load("data/match_odds.json")
    id_to_team = {t["team_id"]: t for t in teams_all}

    odds_by_match = {}
    for o in odds:
        odds_by_match.setdefault(o["match_id"], []).append(
            (o["odds_home"], o["odds_draw"], o["odds_away"]))

    def fit(as_of, extra_ids=frozenset()):
        cut = date.fromisoformat(as_of)
        cnt = Counter()
        for m in matches:
            if date.fromisoformat(m["date"]) < cut:
                cnt[m["home_team_id"]] += 1
                cnt[m["away_team_id"]] += 1
        eligible = {t for t, c in cnt.items() if c >= MIN_MATCHES} | set(extra_ids)
        tlist = [t for t in teams_all if t["team_id"] in eligible]
        feats = build_features(as_of, tlist, matches, fifa, [], [], squad_values=squad,
                               xi=HP["xi"], blend_weight=0.7, n_recent=10)
        params = fit_model(as_of, tlist, matches, feats, xi=HP["xi"],
                           lambda_reg=HP["lambda_reg"], c_a=HP["c_a"], c_x=HP["c_x"],
                           c_d=HP["c_d"], c_y=HP["c_y"], theta=HP["theta"], c_v=HP["c_v"])
        return params, eligible

    def pair_probs(params, fixtures, eligible):
        """Return (model_list, market_list, outcomes) for fixtures with BOTH; coverage."""
        model, market, outs = [], [], []
        total = with_odds = 0
        for m in fixtures:
            h, a = m["home_team_id"], m["away_team_id"]
            if h not in eligible or a not in eligible:
                continue
            total += 1
            books = odds_by_match.get(m["match_id"])
            if not books:
                continue
            with_odds += 1
            P = matchup_matrix(params, h, a, home_flag=not m.get("neutral", True), kappa=0.0)
            model.append(result_probs(P))
            market.append(consensus_probs(books))
            outs.append(_outcome(m))
        return model, market, outs, total, with_odds

    # ---- VALIDATION: non-WC internationals Aug 1 - Nov 19 2022 ----
    params_val, elig_val = fit("2022-08-01")
    val_fixtures = [m for m in matches
                    if m["competition"] != "FIFA World Cup"
                    and date(2022, 8, 1) <= date.fromisoformat(m["date"]) <= date(2022, 11, 19)]
    v_model, v_market, v_out, v_total, v_cov = pair_probs(params_val, val_fixtures, elig_val)

    # ---- WC TEST: 64 WC2022 games ----
    wc_ids = {t for g in _load("config/tournament_config_2022.json")["groups"].values() for t in g}
    params_wc, elig_wc = fit("2022-11-19", extra_ids=wc_ids)
    wc_fixtures = [m for m in matches
                   if m["competition"] == "FIFA World Cup"
                   and date(2022, 11, 19) < date.fromisoformat(m["date"]) <= date(2022, 12, 31)]
    w_model, w_market, w_out, w_total, w_cov = pair_probs(params_wc, wc_fixtures, elig_wc)

    print("=" * 78)
    print("BLEND TEST — model vs de-vigged Bet365 market (cached only)")
    print("=" * 78)
    print(f"VALIDATION coverage: {v_cov}/{v_total} non-WC internationals have Bet365 odds "
          f"({100*v_cov/max(v_total,1):.0f}%)")
    print(f"WC TEST coverage   : {w_cov}/{w_total} WC games have odds")
    print(f"sanity — model alone val log-loss={log_loss(v_model, v_out):.4f}; "
          f"WC model alone={log_loss(w_model, w_out):.4f} (shipped 1.0202), "
          f"WC market alone={log_loss(w_market, w_out):.4f} (shipped 0.9986)")

    # ---- VALIDATION sweep ----
    print("\n" + "-" * 78)
    print("VALIDATION sweep (log-loss; w = weight on MODEL)   [w=0 market, w=1 model]")
    print("-" * 78)
    print(f"   {'w':>4} | {'linear':>9} | {'log-linear':>10}")
    best = None  # (loss, pool, w)
    for w in WS:
        ll_lin = log_loss([_linear(mo, ma, w) for mo, ma in zip(v_model, v_market)], v_out)
        ll_log = log_loss([_loglinear(mo, ma, w) for mo, ma in zip(v_model, v_market)], v_out)
        print(f"   {w:>4} | {ll_lin:>9.4f} | {ll_log:>10.4f}")
        for pool, ll in (("linear", ll_lin), ("log-linear", ll_log)):
            if best is None or ll < best[0]:
                best = (ll, pool, w)
    best_ll, best_pool, best_w = best
    print(f"\nBEST validation: pool={best_pool}  w*={best_w}  val log-loss={best_ll:.4f}")

    # ---- apply ONCE to WC ----
    pool_fn = _linear if best_pool == "linear" else _loglinear
    w_blend = [pool_fn(mo, ma, best_w) for mo, ma in zip(w_model, w_market)]

    def row(name, pred):
        return (name, log_loss(pred, w_out), brier_score(pred, w_out),
                ranked_probability_score(pred, w_out))

    rows = [row("market alone (w=0)", w_market), row("model alone (w=1)", w_model),
            row(f"blend {best_pool} w*={best_w}", w_blend)]
    print("\n" + "-" * 78)
    print("WC2022 TEST (64 games; lower is better)")
    print("-" * 78)
    print(f"  {'':<26}{'log-loss':>10}{'Brier':>10}{'RPS':>9}")
    for nm, ll, br, rps in rows:
        print(f"  {nm:<26}{ll:>10.4f}{br:>10.4f}{rps:>9.4f}")

    mkt_ll = rows[0][1]
    blend_ll = rows[2][1]
    delta = mkt_ll - blend_ll
    print("\nVERDICT:")
    if best_w == 0.0:
        print("  w*=0 -> the blend is just the market; the MODEL ADDS NOTHING beyond the")
        print("  odds on validation. (No independent signal detectable here.)")
    else:
        better = "BEATS" if delta > 0 else "does NOT beat"
        print(f"  w*={best_w}>0 chosen on validation -> model carries some independent signal.")
        print(f"  On the held-out WC games the blend {better} the market by "
              f"{delta:+.4f} log-loss ({mkt_ll:.4f} -> {blend_ll:.4f}).")

    # ---- thin-coverage guard: in-sample WC blend curve (overfit-prone) ----
    if v_cov < 112:  # < half of 224
        print("\n" + "-" * 78)
        print(f"WARNING: validation odds coverage thin ({v_cov}/224); w* may be unreliable.")
        print("In-sample WC blend curve (OVERFIT-PRONE — not a fair estimate):")
        print("-" * 78)
        for w in WS:
            ll_lin = log_loss([_linear(mo, ma, w) for mo, ma in zip(w_model, w_market)], w_out)
            ll_log = log_loss([_loglinear(mo, ma, w) for mo, ma in zip(w_model, w_market)], w_out)
            print(f"   w={w:>4} | linear {ll_lin:.4f} | log-linear {ll_log:.4f}")

    out = os.path.join(ROOT, "data", "blend_test_2022.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"validation_coverage": [v_cov, v_total], "wc_coverage": [w_cov, w_total],
                   "best_pool": best_pool, "best_w": best_w, "best_val_log_loss": best_ll,
                   "wc": {nm: {"log_loss": ll, "brier": br, "rps": rps}
                          for nm, ll, br, rps in rows}}, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
