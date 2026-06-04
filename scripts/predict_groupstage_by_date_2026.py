"""Full group-stage predictions ordered BY MATCH DATE (2026 World Cup).

Fetches the real 72 group fixtures (dates + home/away) from TheStatsAPI, then
predicts each with the fitted Dixon-Coles model (no free 2026 odds exist; MODEL,
not market). Output is sorted chronologically by kickoff.

If the fixtures fetch is blocked, exits non-zero so the by-group/matchday script
(predict_groupstage_2026.py) remains the fallback.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
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
from wc_model.model import btts, fit_model, matchup_matrix, over_under, result_probs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HP = dict(xi=0.0008, lambda_reg=8.0, c_a=0.30, c_x=0.10, c_d=0.30, c_y=0.10, theta=0.0, c_v=0.1)
AS_OF = "2026-06-10"
BASE = "https://api.thestatsapi.com/api/football"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
CACHE = os.path.join(ROOT, "data", "raw", "wc2026_fixtures.json")


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def fetch_fixtures():
    if os.path.exists(CACHE):  # cache so we never depend on a flaky provider twice
        return json.load(open(CACHE, encoding="utf-8"))
    load_dotenv(os.path.join(ROOT, ".env"))
    key = os.environ["STATSAPI_KEY"]
    url = BASE + "/matches?" + urllib.parse.urlencode(
        {"competition_id": "comp_6107", "date_from": "2026-06-01",
         "date_to": "2026-07-05", "per_page": 100})
    last = None
    for attempt in range(7):
        req = urllib.request.Request(url, method="GET")
        req.add_header("Authorization", "Bearer " + key)
        req.add_header("User-Agent", UA)
        req.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8", "replace")).get("data", [])
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            json.dump(data, open(CACHE, "w", encoding="utf-8"))
            return data
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (403, 429, 503) and attempt < 6:
                wait = 10 * (attempt + 1)
                print(f"  fixtures fetch {e.code}; backoff {wait}s (attempt {attempt+1}/7)", flush=True)
                time.sleep(wait)
                continue
            raise
    raise last


def main():
    teams = _load("data/teams.json")
    matches = _load("data/match_results.json")
    fifa = _load("data/fifa_ratings.json")
    squad = _load("data/squad_values.json")
    cfg = _load("config/tournament_config_2026.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    nm = lambda t: name.get(t, t)
    hosts = set(cfg["host_team_ids"])
    group_of = {t: g for g, ms in cfg["groups"].items() for t in ms}

    lookup = {}
    for t in teams:
        for x in [t["canonical_name"], *(t.get("aliases") or [])]:
            if x:
                lookup[x.lower()] = t["team_id"]
                lookup[_slug(x)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]
    resolve = lambda x: lookup.get(x.lower()) or lookup.get(_slug(x))

    print("fetching 2026 fixtures (dates + home/away) ...", flush=True)
    fixtures = fetch_fixtures()
    games = []
    for m in fixtures:
        h = resolve((m.get("home_team") or {}).get("name", ""))
        a = resolve((m.get("away_team") or {}).get("name", ""))
        if not h or not a or h not in group_of or a not in group_of:
            continue
        if group_of[h] != group_of[a]:
            continue  # group-stage games only (same group)
        games.append({"utc": m.get("utc_date", ""), "home": h, "away": a, "group": group_of[h]})
    games.sort(key=lambda g: (g["utc"], g["group"]))
    print(f"  group fixtures resolved: {len(games)}", flush=True)

    cut = date.fromisoformat(AS_OF)
    cnt = Counter()
    for mm in matches:
        if date.fromisoformat(mm["date"]) < cut:
            cnt[mm["home_team_id"]] += 1
            cnt[mm["away_team_id"]] += 1
    all_group = set(group_of)
    eligible = {t for t, c in cnt.items() if c >= 50} | all_group
    tlist = [t for t in teams if t["team_id"] in eligible]
    feats = build_features(AS_OF, tlist, matches, fifa, [], [], squad_values=squad,
                           xi=HP["xi"], blend_weight=0.7, n_recent=10)
    params = fit_model(AS_OF, tlist, matches, feats, xi=HP["xi"], lambda_reg=HP["lambda_reg"],
                       c_a=HP["c_a"], c_x=HP["c_x"], c_d=HP["c_d"], c_y=HP["c_y"],
                       theta=HP["theta"], c_v=HP["c_v"])

    print("\n" + "=" * 90)
    print("2026 WORLD CUP — FULL GROUP STAGE BY MATCH DATE — MODEL prediction (not market)")
    print("=" * 90)
    print(f"as-of {AS_OF}, rho={params.rho:.4f}. Format: 1X2 H/D/A -> result, modal score.\n")

    out = []
    cur_date = None
    for g in games:
        d = g["utc"][:10]
        if d != cur_date:
            cur_date = d
            print(f"\n--- {d} ---")
        h, a = g["home"], g["away"]
        hf = h in hosts
        P = matchup_matrix(params, h, a, home_flag=hf, kappa=0.0)
        ph, pd, pa = result_probs(P)
        ml = max((("home", ph), ("draw", pd), ("away", pa)), key=lambda kv: kv[1])[0]
        res = {"home": nm(h), "draw": "Draw", "away": nm(a)}[ml]
        sx, sy = divmod(int(np.argmax(P)), P.shape[1])
        ov, _ = over_under(P, 2.5)
        bt = btts(P)
        tag = " (host)" if hf else ""
        print(f"   [{g['group']}] {nm(h)}{tag} vs {nm(a)}:  {ph*100:.0f}/{pd*100:.0f}/{pa*100:.0f}  "
              f"-> {res} {sx}-{sy}  (O2.5 {ov*100:.0f}%, BTTS {bt*100:.0f}%)")
        out.append({"date": d, "utc": g["utc"], "group": g["group"], "home": h, "away": a,
                    "host_advantage": hf, "p_home": ph, "p_draw": pd, "p_away": pa,
                    "most_likely_result": ml, "most_likely_score": [int(sx), int(sy)],
                    "over_2_5": float(ov), "btts": float(bt)})

    pj = os.path.join(ROOT, "data", "predict_groupstage_by_date_2026.json")
    json.dump({"as_of": AS_OF, "rho": params.rho, "source": "model (no market odds)",
               "games": out, "team_names": {t: nm(t) for t in all_group}},
              open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {pj}  ({len(out)} games, ordered by date)")


if __name__ == "__main__":
    main()
