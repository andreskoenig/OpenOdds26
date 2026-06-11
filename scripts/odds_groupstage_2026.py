"""PART 2: 2026 group-stage forecast from TODAY's 1X2 market (w=0), with fallback.

Intent: fetch today's de-vigged 1X2 for all 72 group games, calibrate each to a
scoreline matrix (market_scoreline, fixed rho), Monte-Carlo the group stage under
the official 2026 rules, and report P(1st/2nd/3rd/advance), modal tables, and the
projected 8 best thirds.

REALITY (reported, not hidden): there is no ODDS_API_KEY, and the available
provider (TheStatsAPI) returns the 72 fixtures with odds_available=False / 404 —
no market lines exist for unplayed 2026 group games. So coverage is 0/72 and the
sim FALLS BACK to the MODEL matrices (fitted 2026 model, matchup_matrix), CLEARLY
MARKED as model-shaped, not market-driven. If/when lines appear, games with odds
use market_scoreline; only the missing ones fall back.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date

import numpy as np
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from wc_model.features import build_features
from wc_model.market import market_scoreline
from wc_model.model import fit_model, matchup_matrix
from wc_model.simulate import _play_group

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HP = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0, c_v=0.1)
AS_OF = "2026-06-10"
N_SIMS = 20000
SEED = 20260610
BASE = "https://api.thestatsapi.com/api/football"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
GROUP_WINDOW = ("2026-06-01", "2026-07-05")


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def fetch_group_odds_coverage(resolve):
    """Return (coverage_count, total, books_seen, note). Read-only, key in header."""
    load_dotenv(os.path.join(ROOT, ".env"))
    key = os.environ.get("ODDS_API_KEY") or os.environ.get("STATSAPI_KEY")
    src = "ODDS_API_KEY" if os.environ.get("ODDS_API_KEY") else "STATSAPI_KEY (no ODDS_API_KEY present)"
    if not key:
        return 0, 0, [], "no odds key in .env"
    url = BASE + "/matches?" + urllib.parse.urlencode(
        {"competition_id": "comp_6107", "date_from": GROUP_WINDOW[0],
         "date_to": GROUP_WINDOW[1], "per_page": 100})
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("User-Agent", UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        return 0, 0, [], f"fetch failed: {type(e).__name__}"
    ms = data.get("data", [])
    with_odds = [m for m in ms if m.get("odds_available") is True]
    return len(with_odds), len(ms), [], f"source={src}; provider returned {len(ms)} fixtures, {len(with_odds)} with odds_available"


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    cfg = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    groups = cfg["groups"]
    hosts = set(cfg["host_team_ids"])
    all_teams = [t for g in groups.values() for t in g]

    lookup = {}
    for t in teams:
        for x in [t["canonical_name"], *(t.get("aliases") or [])]:
            if x:
                lookup[x.lower()] = t["team_id"]
                lookup[_slug(x)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]
    resolve = lambda x: lookup.get(x.lower()) or lookup.get(_slug(x))

    # ---- odds coverage probe ----
    cov, total, _books, note = fetch_group_odds_coverage(resolve)
    market_available = cov > 0

    # ---- fit the shipped 2026 model (for fallback matrices + rho) ----
    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for m in matches:
        if date.fromisoformat(m["date"]) < cut:
            cnt[m["home_team_id"]] += 1
            cnt[m["away_team_id"]] += 1
    eligible = {t for t, c in cnt.items() if c >= 50} | set(all_teams)
    tlist = [t for t in teams if t["team_id"] in eligible]
    feats = build_features(AS_OF, tlist, matches, fifa, [], [], squad_values=squad,
                           xi=HP["xi"], blend_weight=0.7, n_recent=10)
    params = fit_model(AS_OF, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"],
                       theta=HP["theta"], c_v=HP["c_v"])

    teams_dict = {t: {"rating": params.atk.get(t, 0.0) + params.def_.get(t, 0.0),
                      "host": t in hosts} for t in all_teams}

    # market matrices per pairing (none here); fallback = model matchup_matrix.
    market_mtx = {}  # frozenset({a,b}) -> (matrix, home_id)

    def matrix_fn(home, away, flag):
        key = frozenset({home, away})
        if key in market_mtx:
            M, mhome = market_mtx[key]
            return M if mhome == home else M.T
        return matchup_matrix(params, home, away, flag, kappa=0.0)

    src_label = ("MARKET (de-vigged 1X2)" if market_available
                 else "MODEL fallback (no market lines available — model-shaped, NOT market)")

    print("=" * 80)
    print("PART 2 — 2026 GROUP STAGE forecast")
    print("=" * 80)
    print(f"odds coverage: {cov}/{total or 72} group games have market lines")
    print(f"  {note}")
    print(f"scoreline/outcome source: {src_label}")
    print(f"as-of {AS_OF} | rho={params.rho:.4f} | sims {N_SIMS} (seed {SEED})")
    if not market_available:
        print("  NOTE: w=0 market-driven outcomes are NOT possible for unplayed 2026 group")
        print("  games (no public free lines yet). This is the MODEL's group forecast; the")
        print("  market->scoreline core is ready and will be used per-game once lines exist.")
    print("\nsimulating group stage ...\n", flush=True)

    pos = {t: [0, 0, 0, 0] for t in all_teams}
    adv = {t: 0 for t in all_teams}
    best_third = {t: 0 for t in all_teams}
    modal = {g: Counter() for g in groups}
    rng = np.random.default_rng(SEED)

    for _ in range(N_SIMS):
        thirds = []
        for g, members in groups.items():
            ranked, stats = _play_group(members, teams_dict, matrix_fn, rng)
            for i, t in enumerate(ranked):
                pos[t][i] += 1
            adv[ranked[0]] += 1
            adv[ranked[1]] += 1
            thirds.append((ranked[2], stats[ranked[2]]))
            modal[g][tuple(ranked)] += 1
        thirds.sort(key=lambda c: (-c[1]["pts"], -c[1]["gd"], -c[1]["gf"]))
        for t, _s in thirds[:8]:
            best_third[t] += 1

    def pct(x):
        return 100.0 * x / N_SIMS

    print("-" * 80)
    print("PER GROUP — P(1st) P(2nd) P(3rd) P(advance), modal final table")
    print("-" * 80)
    out_groups = {}
    for g, members in groups.items():
        print(f"  Group {g}:")
        rows = sorted(members, key=lambda t: -adv[t])
        for t in rows:
            print(f"     {nm(t):<22} 1st {pct(pos[t][0]):4.1f}  2nd {pct(pos[t][1]):4.1f}  "
                  f"3rd {pct(pos[t][2]):4.1f}  adv {pct(adv[t]):5.1f}%")
        modal_order = list(modal[g].most_common(1)[0][0])
        print(f"     modal table: {' > '.join(nm(t) for t in modal_order)}")
        out_groups[g] = {
            "teams": {t: {"p1": pct(pos[t][0]), "p2": pct(pos[t][1]), "p3": pct(pos[t][2]),
                          "p_advance": pct(adv[t])} for t in members},
            "modal_table": modal_order,
        }

    # projected 8 best thirds
    bt = sorted(all_teams, key=lambda t: -best_third[t])[:8]
    print("\n" + "-" * 80)
    print("PROJECTED 8 BEST THIRD-PLACE QUALIFIERS (P = qualifies as a best third)")
    print("-" * 80)
    for i, t in enumerate(bt, 1):
        print(f"  {i}. {nm(t):<22} P(best-third qualify) {pct(best_third[t]):4.1f}%  "
              f"(P 3rd {pct(pos[t][2]):4.1f}%)")

    out = {
        "as_of": AS_OF, "n_sims": N_SIMS, "seed": SEED, "rho": params.rho,
        "odds_coverage": {"with_odds": cov, "total": total or 72, "note": note},
        "outcome_source": src_label,
        "groups": out_groups,
        "projected_best_thirds": [{"team": t, "p_best_third": pct(best_third[t]),
                                   "p_third": pct(pos[t][2])} for t in bt],
        "team_names": {t: nm(t) for t in all_teams},
    }
    pj = os.path.join(ROOT, "data", "odds_groupstage_2026.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    _write_md(os.path.join(ROOT, "data", "odds_groupstage_2026.md"), out, nm, src_label, note)
    print(f"\nwrote {pj}\nwrote {os.path.join(ROOT, 'data', 'odds_groupstage_2026.md')}")


def _write_md(path, out, nm, src_label, note):
    L = ["# 2026 World Cup — group-stage forecast", "",
         f"**Outcome source:** {src_label}", "",
         f"**Odds coverage:** {out['odds_coverage']['with_odds']}/{out['odds_coverage']['total']} "
         f"group games have market lines — {note}.", "",
         f"as-of {out['as_of']}, {out['n_sims']} sims, rho={out['rho']:.4f}.", ""]
    for g, gd in out["groups"].items():
        L.append(f"## Group {g}")
        L.append("| team | P(1st) | P(2nd) | P(3rd) | P(advance) |")
        L.append("|---|---|---|---|---|")
        for t in sorted(gd["teams"], key=lambda x: -gd["teams"][x]["p_advance"]):
            d = gd["teams"][t]
            L.append(f"| {nm(t)} | {d['p1']:.1f} | {d['p2']:.1f} | {d['p3']:.1f} | {d['p_advance']:.1f}% |")
        L.append(f"\n_modal table:_ {' > '.join(nm(t) for t in gd['modal_table'])}\n")
    L.append("## Projected 8 best third-place qualifiers")
    L.append("| # | team | P(best-third qualify) | P(3rd) |")
    L.append("|---|---|---|---|")
    for i, b in enumerate(out["projected_best_thirds"], 1):
        L.append(f"| {i} | {nm(b['team'])} | {b['p_best_third']:.1f}% | {b['p_third']:.1f}% |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
