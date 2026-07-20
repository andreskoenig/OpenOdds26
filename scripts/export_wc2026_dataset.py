"""Export the full WC2026 dataset -> data/export/ (CSV + JSON).

One row per match (104 = 72 group + 32 knockout) joining:
  - ground truth (score, 90'/120' outcome, ET/pens status, advancer)
  - model predictions (frozen pre-match: 1X2, modal score, xG, P(advance))
  - bookmaker closing odds (Bet365/Pinnacle raw decimals + de-vigged probs)
  - per-match model scoring (correct flags, log-loss, Brier)

Plus the tournament P(win) time series in long format, and a column README.
Everything reads the already-built artifacts (performance.json, the market
archive, live_results) — no model code runs here. Stdlib only.

Run:  python scripts/export_wc2026_dataset.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import date, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "analysis")
PERF = os.path.join(ROOT, "dashboard", "performance.json")
MARKET = os.path.join(ROOT, "data", "research", "market_odds_wc2026.json")
LIVE = os.path.join(ROOT, "dashboard", "live_results.json")
PWIN = os.path.join(ROOT, "data", "pwin_history.json")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def devig(oh, od, oa):
    q = [1.0 / oh, 1.0 / od, 1.0 / oa]
    s = sum(q)
    return [round(x / s, 4) for x in q]


def build_market_lookup():
    """(pair, near-date) -> reoriented odds per book (same rule as the scorer)."""
    idx = {}
    for e in load(MARKET).get("fixtures", []):
        idx.setdefault(frozenset((e["home_id"], e["away_id"])), []).append(e)

    def get(home, away, near):
        best, bd = None, None
        for e in idx.get(frozenset((home, away)), []):
            try:
                delta = abs((date.fromisoformat(e["kickoff"]) - date.fromisoformat(near)).days)
            except Exception:
                continue
            if delta > 2:
                continue
            if bd is None or delta < bd:
                best, bd = e, delta
        if not best:
            return {}
        out = {}
        for bname, o in (best.get("books") or {}).items():
            if best["home_id"] == home:
                oh, od, oa = o.get("home"), o.get("draw"), o.get("away")
            else:
                oh, od, oa = o.get("away"), o.get("draw"), o.get("home")
            if None in (oh, od, oa):
                continue
            out[bname] = {"odds": [oh, od, oa], "probs": devig(oh, od, oa)}
        return out
    return get


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    perf = load(PERF)
    market = build_market_lookup()
    status_by_pair = {}
    for r in load(LIVE).get("results", []):
        key = (frozenset((r["home_team_id"], r["away_team_id"])), r["date"])
        status_by_pair[key] = r.get("status", "")

    def status_of(home, away, d):
        for (pair, rd), st in status_by_pair.items():
            if pair == frozenset((home, away)) and abs(
                    (date.fromisoformat(rd) - date.fromisoformat(d)).days) <= 1:
                return {"STATUS_FULL_TIME": "FT", "STATUS_FINAL": "FT",
                        "STATUS_FINAL_AET": "AET", "STATUS_FINAL_PEN": "PEN"}.get(st, st)
        return ""

    rows = []

    # ---- 72 group games -------------------------------------------------
    for m in perf["matches"]:
        x = m.get("pred_1x2") or {}
        me = market(m["home_id"], m["away_id"], m["date"])
        b365, pin = me.get("Bet365"), me.get("Pinnacle")
        rows.append({
            "match_id": f"G{len(rows)+1:02d}", "stage": "group", "group": m.get("group"),
            "date": m["date"], "home_id": m["home_id"], "away_id": m["away_id"],
            "home_name": m["home"], "away_name": m["away"],
            "actual_home_goals": (m.get("actual_score") or [None, None])[0],
            "actual_away_goals": (m.get("actual_score") or [None, None])[1],
            "status": status_of(m["home_id"], m["away_id"], m["date"]) or "FT",
            "outcome_90": m.get("actual_outcome"), "outcome_120": m.get("actual_outcome"),
            "advancer_id": None,
            "model_p_home_90": x.get("h"), "model_p_draw_90": x.get("d"), "model_p_away_90": x.get("a"),
            "model_p_home_120": None, "model_p_draw_120": None, "model_p_away_120": None,
            "model_p_home_advance": None, "model_p_away_advance": None,
            "model_xg_home": (m.get("exp_goals") or [None, None])[0],
            "model_xg_away": (m.get("exp_goals") or [None, None])[1],
            "model_modal_home": (m.get("pred_score") or [None, None])[0],
            "model_modal_away": (m.get("pred_score") or [None, None])[1],
            "model_modal_p": None, "model_pred_outcome": m.get("pred_outcome"),
            "bet365_home_odds": b365 and b365["odds"][0], "bet365_draw_odds": b365 and b365["odds"][1],
            "bet365_away_odds": b365 and b365["odds"][2],
            "bet365_p_home": b365 and b365["probs"][0], "bet365_p_draw": b365 and b365["probs"][1],
            "bet365_p_away": b365 and b365["probs"][2],
            "pinnacle_home_odds": pin and pin["odds"][0], "pinnacle_draw_odds": pin and pin["odds"][1],
            "pinnacle_away_odds": pin and pin["odds"][2],
            "pinnacle_p_home": pin and pin["probs"][0], "pinnacle_p_draw": pin and pin["probs"][1],
            "pinnacle_p_away": pin and pin["probs"][2],
            "model_1x2_correct": m.get("outcome_correct"),
            "model_exact_correct": m.get("exact_correct"),
            "model_advance_correct": None,
            "model_log_loss": m.get("log_loss"), "model_brier": m.get("brier"),
        })

    # ---- 32 knockout ties ----------------------------------------------
    STAGE = {"R32": "R32", "R16": "R16", "QF": "QF", "SF": "SF",
             "third_place": "third_place", "Final": "final"}
    ko = perf["knockout"]["rounds"]
    for rnd in ("R32", "R16", "QF", "SF", "third_place", "Final"):
        for t in ko.get(rnd, []) or []:
            if not t.get("home"):
                continue
            x90 = t.get("p_1x2_90") or [None] * 3
            x120 = t.get("p_1x2_120") or [None] * 3
            st = status_of(t["home"], t["away"], t.get("date") or "2026-06-28")
            oc120 = t.get("actual_outcome")
            oc90 = "draw" if st in ("AET", "PEN") else oc120
            me = market(t["home"], t["away"], t.get("date") or "2026-06-28")
            b365, pin = me.get("Bet365"), me.get("Pinnacle")
            rows.append({
                "match_id": t.get("match"), "stage": STAGE[rnd], "group": None,
                "date": t.get("date"), "home_id": t["home"], "away_id": t["away"],
                "home_name": t.get("home_name"), "away_name": t.get("away_name"),
                "actual_home_goals": (t.get("actual_score") or [None, None])[0],
                "actual_away_goals": (t.get("actual_score") or [None, None])[1],
                "status": st,
                "outcome_90": oc90, "outcome_120": oc120,
                "advancer_id": t.get("actual_advancer"),
                "model_p_home_90": x90[0], "model_p_draw_90": x90[1], "model_p_away_90": x90[2],
                "model_p_home_120": x120[0], "model_p_draw_120": x120[1], "model_p_away_120": x120[2],
                "model_p_home_advance": t.get("p_home_adv"), "model_p_away_advance": t.get("p_away_adv"),
                "model_xg_home": (t.get("xg") or [None, None])[0],
                "model_xg_away": (t.get("xg") or [None, None])[1],
                "model_modal_home": (t.get("modal") or [None, None])[0],
                "model_modal_away": (t.get("modal") or [None, None])[1],
                "model_modal_p": t.get("modal_p"),
                "model_pred_outcome": None if x120[0] is None else
                    ("home", "draw", "away")[max(range(3), key=lambda i: x120[i])],
                "bet365_home_odds": b365 and b365["odds"][0], "bet365_draw_odds": b365 and b365["odds"][1],
                "bet365_away_odds": b365 and b365["odds"][2],
                "bet365_p_home": b365 and b365["probs"][0], "bet365_p_draw": b365 and b365["probs"][1],
                "bet365_p_away": b365 and b365["probs"][2],
                "pinnacle_home_odds": pin and pin["odds"][0], "pinnacle_draw_odds": pin and pin["odds"][1],
                "pinnacle_away_odds": pin and pin["odds"][2],
                "pinnacle_p_home": pin and pin["probs"][0], "pinnacle_p_draw": pin and pin["probs"][1],
                "pinnacle_p_away": pin and pin["probs"][2],
                "model_1x2_correct": t.get("x1x2_correct"),
                "model_exact_correct": t.get("exact_correct"),
                "model_advance_correct": t.get("advance_correct"),
                "model_log_loss": t.get("log_loss"), "model_brier": t.get("brier"),
            })

    cols = list(rows[0].keys())
    csv_path = os.path.join(OUT_DIR, "wc2026_matches.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_DIR, "wc2026_matches.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    # ---- P(win) time series, long format --------------------------------
    pw = load(PWIN)
    with open(os.path.join(OUT_DIR, "wc2026_pwin_history.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "round", "team_id", "team_name", "p_win"])
        for s in pw["snapshots"]:
            for t, p in sorted(s["p_win"].items(), key=lambda kv: -kv[1]):
                w.writerow([s["date"], s["label"], t, pw["team_names"].get(t, t), p])

    # ---- closing KPIs + calibration + market comparison ------------------
    kpis_out = {
        "generated_at": perf.get("generated_at"),
        "tournament": perf.get("tournament"),
        "kpis": perf.get("kpis"),
        "kpis_market": perf.get("kpis_market"),
        "knockout_summary": perf.get("knockout_summary"),
        "calibration": perf.get("calibration"),
    }
    with open(os.path.join(OUT_DIR, "wc2026_kpis.json"), "w", encoding="utf-8") as f:
        json.dump(kpis_out, f, ensure_ascii=False, indent=1)

    # ---- raw closing-odds archive (verbatim copy for provenance) ---------
    with open(os.path.join(OUT_DIR, "wc2026_market_odds_raw.json"), "w", encoding="utf-8") as f:
        json.dump(load(MARKET), f, ensure_ascii=False, indent=1)

    # ---- handoff document ------------------------------------------------
    readme = """# WC2026 dataset export

Generated {now} by scripts/export_wc2026_dataset.py from the OpenOdds26 model
(https://github.com/andreskoenig/OpenOdds26). 104 matches: 72 group + 32 knockout.

## wc2026_matches.csv / .json — one row per match

- match_id, stage (group|R32|R16|QF|SF|third_place|final), group, date (kickoff, ESPN scoreboard day)
- home_id/away_id/home_name/away_name — home = nominal home side of the fixture
- actual_home_goals/actual_away_goals — final recorded score (group: 90'; knockout: after extra time if played)
- status — FT (decided in 90'), AET (decided in extra time), PEN (penalty shootout)
- outcome_90 — home/draw/away at regulation (AET/PEN games = draw). Bookmaker 1X2 settles on this.
- outcome_120 — home/draw/away at the end of extra time (PEN games = draw). Group games: same as outcome_90.
- advancer_id — knockout only: the team that progressed (resolves shootouts)
- model_p_*_90 — model's pre-match regulation 1X2 probabilities
- model_p_*_120 — knockout only: model's 1X2 by end of extra time (draw leg = goes to penalties)
- model_p_home_advance/model_p_away_advance — knockout only: P(progress), extra time + penalties folded in
- model_xg_home/away — model expected goals (90')
- model_modal_home/away (+ model_modal_p) — most likely scoreline (group: 90'; knockout: 120' distribution)
- model_pred_outcome — argmax of the model 1X2 (group: 90'; knockout: 120')
- bet365_*/pinnacle_* — closing decimal odds (last pre-kickoff snapshot, OddsPapi) + de-vigged
  probabilities (proportional normalization). Settle on outcome_90.
- model_1x2_correct — group: pred vs outcome_90; knockout: pred vs outcome_120
- model_exact_correct — modal scoreline == actual score
- model_advance_correct — knockout: model favourite by P(advance) == advancer_id
- model_log_loss / model_brier — model 1X2 scored on its basis (group 90' / knockout 120')

All predictions were FROZEN before each match kicked off (walk-forward; knockout
rounds frozen the day the previous round completed). No look-ahead.

## wc2026_pwin_history.csv

P(win tournament) snapshot after each completed round (knockout-conditioned
bracket propagation from R32 on). Columns: date, round, team_id, team_name, p_win.
Eliminated teams are absent from later snapshots (i.e. 0).

## wc2026_kpis.json

Closing aggregate metrics as shown on the dashboard:
- kpis — headline over all 104 games: accuracy_1x2, model_exact_rate,
  mean_log_loss, mean_brier (group games scored at 90'; knockout at 120').
- kpis_market — per book (Bet365, Pinnacle): n matched games, the book's
  accuracy/log-loss/brier, the model's SAME-GAME numbers (model_*), and deltas
  (d_* = book minus model). All on the 90' basis (books settle regulation time).
- knockout_summary — knockout-only: n ties, advance accuracy, 1X2@120 accuracy,
  modal (exact score) rate.
- calibration — favourite-probability buckets vs realized favourite win rate.

## wc2026_market_odds_raw.json

Verbatim copy of the closing-odds archive (OddsPapi /v4/historical-odds;
closing = last snapshot at or before kickoff). Per fixture: kickoff (UTC date,
can sit +1 day vs the ESPN scoreboard date), home_id/away_id in the PROVIDER'S
orientation (may be flipped vs the match dataset - the CSV already reorients),
and raw decimal 1X2 per book. Kept so the de-vig step is reproducible.

## Reading guide / caveats for analysis

- Two scoring bases coexist by design. Headline model metrics score knockout
  games at 120' ("did the model call the game"); every model-vs-book comparison
  is at 90' because that is what a 1X2 market settles on. Use outcome_90 with
  the book probabilities, outcome_120 with model_p_*_120.
- A knockout game with status PEN is a draw at BOTH bases (level after 120);
  AET games are a draw at 90' and decisive at 120'. The final (Spain 1-0
  Argentina, AET) is a 90'-draw / 120'-home-win.
- Model exact-score is the modal scoreline of a full bivariate distribution -
  the modal probability (model_modal_p) is the fair yardstick, typically 9-19%.
- The Polymarket winner prior is a MODEL INPUT (weight c_m=0.35), so
  wc2026_pwin_history is not fully independent of the market. Bet365/Pinnacle
  per-match odds are independent benchmarks - the model never saw them.
- The 2022 backtest (see repo README) is development-set grade; this 2026 file
  is the only true out-of-sample record of the model.
- Suggested analyses: model vs book log-loss/Brier with paired tests (same 104
  games), calibration curves from the probability columns, skill by stage
  (group vs knockout), upset analysis (low-probability outcomes that landed),
  and P(win) trajectory vs the market over the 9 snapshots.
""".format(now=datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(os.path.join(OUT_DIR, "HANDOFF.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    n_books = sum(1 for r in rows if r["bet365_p_home"] is not None)
    print(f"export: {len(rows)} matches -> {csv_path}")
    print(f"  with Bet365 odds: {n_books} | knockout rows: {sum(1 for r in rows if r['stage'] != 'group')}")
    print(f"  pwin snapshots: {len(pw['snapshots'])}")


if __name__ == "__main__":
    main()
