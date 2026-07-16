"""Knockout-conditioned live forecast (2026 World Cup).

Once the group stage AND some knockout rounds are decided, the frozen/group-only
`run_forecast --live` is wrong: it re-simulates the whole bracket from group
standings, so eliminated teams keep positive P(win). This conditions on the REAL
knockout results instead.

Method (deterministic bracket propagation, no Monte Carlo needed):
  * Each decided R32 tie pins its winner distribution to {advancer: 1.0}.
  * For every later match, propagate a distribution over which team fills each
    slot: winner_dist[M][t] = sum over feeder pairings of
        P(t in home slot) * P(o in away slot) * P(t beats o),
    where P(t beats o) is the model's single-tie P(advance) (ET + penalties
    folded in, exactly as predict_knockouts_2026.py computes it).
  * P(win tournament) = winner distribution of the Final. Eliminated teams -> 0.

Also fills R16 prediction cards (teams are known once R32 is complete) back into
data/knockout_2026.json WITHOUT touching the frozen R32 predictions, and writes
data/forecast_live_2026.json (drop-in for the dashboard forecast panel).

Run:  python scripts/forecast_live_knockout.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.request
from collections import defaultdict
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.features import build_features
from wc_model.model import fit_model, matchup_matrix, result_probs
from wc_model.simulate import expected_goals
from wc_model.pipeline import team_strength

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Same hyperparameters as the production forecast / predict_knockouts.
HP = dict(xi=0.0012651, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10,
          theta=0.0, c_v=0.1, c_m=0.35, opponent_adjust=True, max_history_years=10.0)
ET_SCALE = 30.0 / 90.0
PEN_TILT = 0.05
PEN_CLIP = 0.15
# ET draw calibration. Independent pro-rata Poissons underprice "still level
# after 120": they ignore the strong positive correlation of ET goals (score
# effects: cagey play, trailing team equalizes) that regulation handles via the
# Dixon-Coles rho. Empirically, of major-tournament knockout games reaching ET:
# WC2018 4/5, Euro2020 3/6, WC2022 5/5, WC2026-so-far 3/5 went to penalties
# -> 15/21 = 0.71 vs the raw model's ~0.54. Shrinking the empirical rate toward
# the model (Beta, ~10 pseudo-obs) gives a target of ~0.66-0.68, i.e. a draw
# inflation factor of ~1.25 on the matchup-specific et_d (capped for safety).
# Applied multiplicatively so stronger/high-xG matchups keep lower pens prob.
ET_DRAW_INFLATION = 1.25
ET_DRAW_CAP = 0.90
# config round key -> (json round label, match-id range order in the tree)
ROUND_KEYS = [("round_of_32", "R32"), ("round_of_16", "R16"),
              ("quarter_finals", "QF"), ("semi_finals", "SF"), ("final", "Final")]


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _pois(lam, K=15):
    return np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(K + 1)])


def et_outcome(lam_h, lam_a):
    M = np.outer(_pois(lam_h), _pois(lam_a))
    return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())


_ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard?dates={}"


def espn_kickoff_dates(teams, d0, d1):
    """ESPN scheduled fixtures d0..d1 -> {frozenset(pair): iso date}. Best effort."""
    lookup = {"usa": "united_states", "united states": "united_states",
              "turkiye": "turkey", "türkiye": "turkey", "czechia": "czech_republic",
              "korea republic": "south_korea", "ir iran": "iran",
              "cote d'ivoire": "ivory_coast", "côte d'ivoire": "ivory_coast"}
    for t in teams:
        for x in [t["canonical_name"], *(t.get("aliases") or [])]:
            if x:
                lookup.setdefault(x.lower(), t["team_id"])
    out = {}
    d = d0
    while d <= d1:
        try:
            data = json.loads(urllib.request.urlopen(
                urllib.request.Request(_ESPN_URL.format(d.strftime("%Y%m%d")),
                                       headers={"User-Agent": "Mozilla/5.0"}), timeout=20).read())
        except Exception:
            d += timedelta(days=1)
            continue
        for ev in data.get("events", []):
            try:
                cs = ev["competitions"][0]["competitors"]
                ids = [lookup.get(c["team"]["displayName"].lower()) for c in cs]
            except Exception:
                continue
            if all(ids):
                out[frozenset(ids)] = d.isoformat()
        d += timedelta(days=1)
    return out


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    # --raw-market: use the raw normalized Polymarket winner odds as the market
    # prior instead of the de-pathed version. Correct late in the knockout: with
    # only the final (+ 3rd place) left, there is no bracket path luck to strip —
    # P(win WC) IS the finalists' per-game probability, so de-path = identity.
    raw_market = "--raw-market" in sys.argv
    market = _load("data/polymarket_winner_2026.json" if raw_market
                   else "data/polymarket_winner_2026_depathed.json")["p_market"]
    cfg = _load("config/tournament_config_2026.json")
    kb = cfg["knockout_bracket"]
    hosts = set(cfg["host_team_ids"])
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)

    knockout = _load("data/knockout_2026.json")     # frozen R32 predictions + tree
    live = _load("dashboard/live_results.json")["results"]

    # advancer per team-pair (ESPN winner flag resolves penalty ties).
    # KNOCKOUT window only: ESPN also flags winners of decisive GROUP games, and
    # a pair can rematch in the knockout (e.g. a third-place tie) — a group-game
    # winner must never pin a knockout tie.
    adv_by_pair = {}
    for r in live:
        if r.get("advancer") and r.get("date", "") >= "2026-06-28":
            adv_by_pair[frozenset((r["home_team_id"], r["away_team_id"]))] = r["advancer"]

    # ---- fit the model as-of today (absorbs all played results) ----
    # Same eligibility as predict_knockouts_2026: teams with >=50 historical
    # matches, plus every tournament team (so all 48 are always fit).
    as_of = date.today().isoformat()
    cut = date.fromisoformat(as_of)
    group_ids = {t for g in cfg["groups"].values() for t in g}
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    elig = {t for t, c in cnt.items() if c >= 50} | group_ids
    tlist = [t for t in teams if t["team_id"] in elig]
    feats = build_features(as_of, tlist, matches, fifa, [], [], squad_values=squad,
                           market_probs=market, xi=HP["xi"], blend_weight=0.7, n_recent=10,
                           opponent_adjust=HP["opponent_adjust"], max_history_years=HP["max_history_years"])
    params = fit_model(as_of, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"], theta=HP["theta"],
                       c_v=HP["c_v"], c_m=HP["c_m"], max_history_years=HP["max_history_years"])

    _tie_cache = {}

    def tie(a, b):
        """One knockout tie a(home) vs b(away): P(a advances) + 90'/120' 1X2, xG,
        modal 120' scoreline. Identical model to predict_knockouts_2026.tie()."""
        if (a, b) in _tie_cache:
            return _tie_cache[(a, b)]
        if a in hosts and b not in hosts:
            P = matchup_matrix(params, a, b, True, kappa=0.0)
        elif b in hosts and a not in hosts:
            P = matchup_matrix(params, b, a, True, kappa=0.0).T
        else:
            P = matchup_matrix(params, a, b, False, kappa=0.0)
        ph, pd, pa = result_probs(P)
        eg_a, eg_b = expected_goals(P)
        et_a, et_d, et_b = et_outcome(eg_a * ET_SCALE, eg_b * ET_SCALE)
        # calibrate P(pens | ET): inflate the ET draw, renormalize the decisive
        # mass so et_a:et_b keeps the model's relative strength ordering.
        et_d_adj = min(ET_DRAW_INFLATION * et_d, ET_DRAW_CAP)
        dec_scale = (1.0 - et_d_adj) / (1.0 - et_d) if et_d < 1.0 else 0.0
        f_draw = et_d_adj / et_d if et_d > 0 else 1.0
        et_a, et_b, et_d = et_a * dec_scale, et_b * dec_scale, et_d_adj
        tilt = float(np.clip(PEN_TILT * (team_strength(params, a) - team_strength(params, b)),
                             -PEN_CLIP, PEN_CLIP))
        p_a_adv = ph + pd * (et_a + et_d * (0.5 + tilt))
        h120, d120, a120 = ph + pd * et_a, pd * et_d, pa + pd * et_b
        poh, poa = _pois(eg_a * ET_SCALE), _pois(eg_b * ET_SCALE)
        Q = {}
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                p = float(P[i, j])
                if p < 1e-12:
                    continue
                if i != j:
                    Q[(i, j)] = Q.get((i, j), 0.0) + p
                else:
                    # same calibration inside the 120' scoreline distribution:
                    # level ET cells (-> pens) up-weighted, decisive down-weighted,
                    # so the modal score stays consistent with the 120' 1X2.
                    for ei, pi in enumerate(poh):
                        for ej, pj in enumerate(poa):
                            w = f_draw if ei == ej else dec_scale
                            Q[(i + ei, j + ej)] = Q.get((i + ei, j + ej), 0.0) + p * pi * pj * w
        (mi, mj), mp = max(Q.items(), key=lambda kv: kv[1])
        out = {"p_a_adv": float(p_a_adv),
               "x90": [round(float(ph), 4), round(float(pd), 4), round(float(pa), 4)],
               "x120": [round(float(h120), 4), round(float(d120), 4), round(float(a120), 4)],
               "xg": [round(float(eg_a), 2), round(float(eg_b), 2)],
               "modal": [int(mi), int(mj)], "modal_p": round(float(mp), 4)}
        _tie_cache[(a, b)] = out
        return out

    # ---- tree from config: match_id -> (round_label, home_ref, away_ref) ----
    node = {}
    order = {lab: [] for _, lab in ROUND_KEYS}
    for key, lab in ROUND_KEYS:
        ms = kb.get(key, [])
        ms = ms if isinstance(ms, list) else [ms]
        for m in ms:
            node[m["match"]] = (lab, m.get("home"), m.get("away"))
            order[lab].append(m["match"])

    # R32 winner distributions pinned to the real advancer (all 16 decided).
    r32_by_id = {t["match"]: t for t in knockout["rounds"]["R32"]}
    base = {}
    for mid in order["R32"]:
        t = r32_by_id[mid]
        adv = adv_by_pair.get(frozenset((t["home"], t["away"])))
        # decided -> pin the real advancer; undecided -> the frozen pre-match
        # P(advance) (NOT 50/50), so a partially played round still propagates.
        base[mid] = ({adv: 1.0} if adv else
                     {t["home"]: t["p_home_adv"], t["away"]: t["p_away_adv"]})

    def _ref_id(ref):
        return ref[2:] if ref and ref[:2] in ("W:", "L:") else None

    memo = {}

    def winner_dist(mid):
        if mid in memo:
            return memo[mid]
        lab, hr, ar = node[mid]
        if lab == "R32":
            memo[mid] = base[mid]
            return base[mid]
        H = winner_dist(_ref_id(hr))
        A = winner_dist(_ref_id(ar))
        # If this tie's pairing is certain AND its result is in, pin the real
        # advancer — at EVERY round, not just R32. Without this, a rerun after
        # R16/QF results land would keep eliminated teams alive in p_win.
        if len(H) == 1 and len(A) == 1:
            a = next(iter(H))
            b = next(iter(A))
            adv = adv_by_pair.get(frozenset((a, b)))
            if adv:
                memo[mid] = {adv: 1.0}
                return memo[mid]
        out = defaultdict(float)
        for ta, pta in H.items():
            for tb, ptb in A.items():
                w = pta * ptb
                if w <= 0:
                    continue
                padv = tie(ta, tb)["p_a_adv"]      # ta as home
                out[ta] += w * padv
                out[tb] += w * (1.0 - padv)
        memo[mid] = dict(out)
        return memo[mid]

    final_id = order["Final"][0]
    champ = winner_dist(final_id)
    tot = sum(champ.values()) or 1.0
    p_win = {t: v / tot for t, v in sorted(champ.items(), key=lambda kv: -kv[1])}

    # ---- fill prediction cards for every round whose teams are now known ----
    # (R16 today; QF/SF/Final/third_place as their feeders resolve on later
    # runs). Frozen R32 predictions are never touched.
    # kickoff dates from ESPN's published schedule (next ~10 days, best effort)
    kick = espn_kickoff_dates(teams, date.today(), date.today() + timedelta(days=10))

    def loser_dist(mid):
        """P(team loses match mid) = P(reaches mid) - P(wins mid). Needed for
        the third-place tie, whose refs are L:Mxx (semi-final losers)."""
        lab, hr, ar = node[mid]
        W = winner_dist(mid)
        reach = defaultdict(float)
        for ref in (hr, ar):
            rid = _ref_id(ref)
            if rid:
                for t, p in winner_dist(rid).items():
                    reach[t] += p
            elif ref:  # literal team id (R32 slots are pre-resolved)
                reach[ref] += 1.0
        out = {t: p - W.get(t, 0.0) for t, p in reach.items() if p - W.get(t, 0.0) > 1e-12}
        return out

    def ref_dist(ref):
        rid = _ref_id(ref)
        if not rid:
            return {}
        return loser_dist(rid) if ref.startswith("L:") else winner_dist(rid)

    n_filled = n_frozen = 0
    for rname in ("R16", "QF", "SF", "Final", "third_place"):
        for e in knockout["rounds"].get(rname, []) or []:
            hd = ref_dist(e.get("home_ref"))
            ad = ref_dist(e.get("away_ref"))
            # both slots certain -> a real, known pairing
            if not (len(hd) == 1 and len(ad) == 1
                    and max(hd.values()) >= 1.0 - 1e-9 and max(ad.values()) >= 1.0 - 1e-9):
                continue
            a = next(iter(hd)); b = next(iter(ad))
            # FREEZE GUARD: once this tie has a result, its card is immutable
            # (and if it was never predicted pre-match, it stays unpredicted) —
            # a fit that has seen the outcome must not write "predictions".
            if frozenset((a, b)) in adv_by_pair:
                n_frozen += 1
                continue
            r = tie(a, b)
            pa_adv = r["p_a_adv"]; pb_adv = 1.0 - pa_adv
            e.update({"home": a, "away": b, "home_name": nm(a), "away_name": nm(b),
                      "date": kick.get(frozenset((a, b))) or e.get("date"),
                      "p_home_adv": round(pa_adv, 4), "p_away_adv": round(pb_adv, 4),
                      "fav": a if pa_adv >= pb_adv else b,
                      "fav_name": nm(a if pa_adv >= pb_adv else b),
                      "fav_p": round(max(pa_adv, pb_adv), 4),
                      "p_1x2_90": r["x90"], "p_1x2_120": r["x120"], "xg": r["xg"],
                      "modal": r["modal"], "modal_p": r["modal_p"]})
            n_filled += 1
    knockout["as_of"] = as_of
    with open(os.path.join(ROOT, "data", "knockout_2026.json"), "w", encoding="utf-8") as f:
        json.dump(knockout, f, ensure_ascii=False, indent=2)

    # ---- write the knockout-conditioned live forecast ----
    n_cond = sum(1 for r in live if r["date"] >= "2026-06-11")
    through = max((r["date"] for r in live), default=None)
    out = {
        "as_of": as_of,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "is_live": True,
        "source": "knockout-conditioned bracket propagation (decided ties pinned)",
        "market_prior": "raw polymarket winner odds (no de-path; identity with <=2 games left)"
                        if raw_market else "de-pathed polymarket winner odds",
        "games_conditioned": n_cond,
        "conditioned_through": through,
        "team_names": {t: nm(t) for t in p_win},
        "p_win": p_win,
    }
    with open(os.path.join(ROOT, "data", "forecast_live_2026.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- report ----
    print(f"as-of {as_of} | conditioned on {n_cond} WC games through {through}")
    print(f"prediction cards filled: {n_filled} | frozen (already decided): {n_frozen}\n")
    print("P(win tournament) — remaining teams:")
    for t, p in list(p_win.items())[:16]:
        print(f"  {nm(t):<18} {p*100:5.1f}%")


if __name__ == "__main__":
    main()
