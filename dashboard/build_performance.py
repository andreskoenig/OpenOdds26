#!/usr/bin/env python3
"""Build dashboard/performance.json for the OpenOdds26 live-performance dashboard.

SCORING ONLY. This script never re-runs the forecast/model. It compares the
FROZEN pre-tournament predictions against actual results as they arrive in
data/match_results.json, and writes a compact JSON the dashboard front-end polls.

Standard library ONLY (json, csv, math, datetime, os) -- runs on a Raspberry Pi
with no pip installs.

Inputs (relative to project root):
  data/predict_groupstage_by_date_2026.json  per-game frozen predictions
  data/pool_picks_groupstage.csv             the user's EV-optimal pool picks
  data/match_results.json                    actual results (refreshed upstream)
  data/forecast_2026.json                    tournament P(win) + generated_at

Output:
  dashboard/performance.json

A prediction is matched to an actual result by date (within +/-1 day) AND the
unordered team pair {home, away}. Actual goals are re-oriented to the
prediction's home/away orientation before scoring.

Pool scoring (group-stage rule, identical for both of the user's pools):
  exact score  -> 3 points (implies correct outcome)
  correct outcome only -> 1 point
  else -> 0 points
"""

import csv
import datetime
import json
import math
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

PRED_PATH = os.path.join(DATA, "predict_groupstage_by_date_2026.json")
PICKS_PATH = os.path.join(DATA, "pool_picks_groupstage.csv")
RESULTS_PATH = os.path.join(DATA, "match_results.json")
FORECAST_PATH = os.path.join(DATA, "forecast_2026.json")
# Conditional mid-tournament forecast (pins played group games); preferred for
# the forecast panel when present. The frozen pre-tournament forecast above is
# kept untouched so the per-match scoring stays honest.
LIVE_FORECAST_PATH = os.path.join(DATA, "forecast_live_2026.json")
OUT_PATH = os.path.join(HERE, "performance.json")

GAMES_TOTAL_GROUP = 72
TOURNAMENT_START = "2026-06-11"
WC_COMPETITION = "FIFA World Cup"
EPS = 1e-15


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_date(s):
    return datetime.date(*(int(x) for x in s.split("-")))


def outcome_from_goals(hg, ag):
    """Return 'home' / 'draw' / 'away' for goals in the home/away orientation."""
    if hg > ag:
        return "home"
    if hg < ag:
        return "away"
    return "draw"


def parse_pick(pick):
    """'2-1' -> (2, 1). Returns None if unparseable."""
    try:
        h, a = pick.strip().split("-")
        return int(h), int(a)
    except Exception:
        return None


def safe_div(num, den):
    return (num / den) if den else None


# ----------------------------------------------------------------------------
# Load inputs (degrade gracefully if optional files are missing)
# ----------------------------------------------------------------------------
def load_pool_picks():
    """Map (date, home_id, away_id) -> pick info, keyed by canonical lower ids."""
    picks = {}
    if not os.path.exists(PICKS_PATH):
        return picks
    pred = load_json(PRED_PATH)
    name_to_id = {v: k for k, v in pred.get("team_names", {}).items()}
    with open(PICKS_PATH, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            home_id = name_to_id.get(row["home"], row["home"])
            away_id = name_to_id.get(row["away"], row["away"])
            key = (row["date"], home_id, away_id)
            picks[key] = {
                "pick": row.get("pick", ""),
                "exp_points": row.get("exp_points", ""),
                "p_exact": row.get("p_exact", ""),
                "p_outcome": row.get("p_outcome", ""),
            }
    return picks


def index_actuals():
    """Index played WC2026 results by frozenset team pair -> list of records.

    Only includes competition == 'FIFA World Cup' and date >= 2026-06-11.
    """
    index = {}
    # martj42 (model source, authoritative) + ESPN live layer (dashboard-only
    # gap-filler so results show before martj42 catches up). Same record shape.
    sources = []
    if os.path.exists(RESULTS_PATH):
        sources.extend(load_json(RESULTS_PATH))
    live_path = os.path.join(HERE, "live_results.json")
    if os.path.exists(live_path):
        sources.extend(load_json(live_path).get("results", []))
    for rec in sources:
        if rec.get("competition") != WC_COMPETITION:
            continue
        date = rec.get("date", "")
        if date < TOURNAMENT_START:
            continue
        home = rec.get("home_team_id")
        away = rec.get("away_team_id")
        if home is None or away is None:
            continue
        if rec.get("home_goals") is None or rec.get("away_goals") is None:
            continue
        pair = frozenset((home, away))
        index.setdefault(pair, []).append(rec)
    return index


def find_actual(pred_home, pred_away, pred_date, actuals_index):
    """Find an actual result for this prediction by team pair + date (+/-1 day).

    Returns goals re-oriented to the prediction's (home, away) orientation:
    (home_goals, away_goals) or None.
    """
    pair = frozenset((pred_home, pred_away))
    candidates = actuals_index.get(pair)
    if not candidates:
        return None
    pdate = parse_date(pred_date)
    best = None
    best_delta = None
    for rec in candidates:
        try:
            delta = abs((parse_date(rec["date"]) - pdate).days)
        except Exception:
            continue
        if delta > 1:
            continue
        if best_delta is None or delta < best_delta:
            best, best_delta = rec, delta
    if best is None:
        return None
    # Re-orient to prediction's home/away.
    if best["home_team_id"] == pred_home:
        return int(best["home_goals"]), int(best["away_goals"])
    else:
        return int(best["away_goals"]), int(best["home_goals"])


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
CAL_BUCKETS = [
    (0.33, 0.45),
    (0.45, 0.55),
    (0.55, 0.70),
    (0.70, 1.01),  # upper inclusive of 1.0
]


def bucket_index(p):
    for i, (lo, hi) in enumerate(CAL_BUCKETS):
        if lo <= p < hi:
            return i
    return None


# ----------------------------------------------------------------------------
# Main build
# ----------------------------------------------------------------------------
def build():
    pred = load_json(PRED_PATH)
    games = pred.get("games", [])
    team_names = pred.get("team_names", {})

    actuals_index = index_actuals()

    # Prefer the conditional live forecast (pins played group games) for the
    # forecast panel; fall back to the frozen pre-tournament forecast.
    forecast = {}
    if os.path.exists(LIVE_FORECAST_PATH):
        forecast = load_json(LIVE_FORECAST_PATH)
    elif os.path.exists(FORECAST_PATH):
        forecast = load_json(FORECAST_PATH)
    f_team_names = forecast.get("team_names", team_names)

    matches = []
    # Accumulators
    n_played = 0
    n_outcome_correct = 0
    n_model_exact = 0
    sum_logloss = 0.0
    sum_brier = 0.0
    # Calibration accumulators: per bucket [sum_pred_prob, n_correct, n]
    cal = [[0.0, 0, 0] for _ in CAL_BUCKETS]

    def disp(tid):
        return team_names.get(tid, tid)

    for g in games:
        home, away = g["home"], g["away"]
        p_home = g.get("p_home", 0.0)
        p_draw = g.get("p_draw", 0.0)
        p_away = g.get("p_away", 0.0)
        ml_score = g.get("most_likely_score", [None, None])

        # Predicted outcome = the model's highest-probability 1X2 pick.
        # NOTE: this is intentionally NOT the modal exact score's implied
        # result -- the modal score can be 1-1 (draw) while the 1X2 argmax
        # favours a win. "Out" is scored against this probabilistic pick.
        pred_outcome = max(
            (("home", p_home), ("draw", p_draw), ("away", p_away)),
            key=lambda kv: kv[1],
        )[0]

        actual = find_actual(home, away, g["date"], actuals_index)
        played = actual is not None

        entry = {
            "date": g["date"],
            "group": g.get("group"),
            "home": disp(home),
            "away": disp(away),
            "pred_outcome": pred_outcome,
            "pred_score": list(ml_score),
            "pred_1x2": {"h": round(p_home, 4), "d": round(p_draw, 4),
                         "a": round(p_away, 4)},
            "exp_goals": [round(g.get("exp_goals_home", 0.0), 2),
                          round(g.get("exp_goals_away", 0.0), 2)],
            "played": played,
            "actual_score": None,
            "actual_outcome": None,
            "outcome_correct": None,
            "exact_correct": None,
            "log_loss": None,
            "brier": None,
        }

        if played:
            n_played += 1
            ahg, aag = actual
            actual_outcome = outcome_from_goals(ahg, aag)
            entry["actual_score"] = [ahg, aag]
            entry["actual_outcome"] = actual_outcome

            # (a) is the model's highest-probability 1X2 pick correct?
            outcome_correct = (pred_outcome == actual_outcome)
            entry["outcome_correct"] = outcome_correct
            if outcome_correct:
                n_outcome_correct += 1

            # (b) model modal exact score correct?
            exact_correct = (list(ml_score) == [ahg, aag])
            entry["exact_correct"] = exact_correct
            if exact_correct:
                n_model_exact += 1

            # (c) 1X2 log-loss and Brier on the model probabilities
            p_assigned = {"home": p_home, "draw": p_draw,
                          "away": p_away}[actual_outcome]
            p_assigned = min(max(p_assigned, EPS), 1.0)
            log_loss = -math.log(p_assigned)
            sum_logloss += log_loss
            entry["log_loss"] = round(log_loss, 4)

            ind = {"home": 0.0, "draw": 0.0, "away": 0.0}
            ind[actual_outcome] = 1.0
            brier = ((p_home - ind["home"]) ** 2 +
                     (p_draw - ind["draw"]) ** 2 +
                     (p_away - ind["away"]) ** 2)
            sum_brier += brier
            entry["brier"] = round(brier, 4)

            # Calibration: favorite probability vs realized win for favorite.
            p_fav = max(p_home, p_draw, p_away)
            fav = max((("home", p_home), ("draw", p_draw), ("away", p_away)),
                      key=lambda kv: kv[1])[0]
            bi = bucket_index(p_fav)
            if bi is not None:
                cal[bi][0] += p_fav
                cal[bi][2] += 1
                if fav == actual_outcome:
                    cal[bi][1] += 1

        matches.append(entry)

    # Chronological order (date, then group) so the matchday separators in the
    # table land cleanly. Played games are always earlier dates than upcoming
    # ones, so this still shows played-first in practice; row styling
    # (played/upcoming) distinguishes them.
    matches.sort(key=lambda m: (m["date"], m["group"] or ""))

    # KPIs
    if n_played > 0:
        kpis = {
            "n_played": n_played,
            "accuracy_1x2": round(n_outcome_correct / n_played, 4),
            "model_exact_rate": round(n_model_exact / n_played, 4),
            "mean_log_loss": round(sum_logloss / n_played, 4),
            "mean_brier": round(sum_brier / n_played, 4),
        }
    else:
        kpis = {
            "n_played": 0,
            "accuracy_1x2": None,
            "model_exact_rate": None,
            "mean_log_loss": None,
            "mean_brier": None,
        }

    calibration = []
    for i, (lo, hi) in enumerate(CAL_BUCKETS):
        s_pred, n_corr, n = cal[i]
        hi_label = 1.0 if hi > 1.0 else hi
        calibration.append({
            "bucket": "{:.2f}-{:.2f}".format(lo, hi_label),
            "lo": lo,
            "hi": hi_label,
            "n": n,
            "predicted": round(s_pred / n, 4) if n else None,
            "realized": round(n_corr / n, 4) if n else None,
        })

    # Forecast top-10 P(win), with movement vs the FROZEN pre-tournament forecast
    # (the baseline): per team we expose the probability delta (percentage points)
    # and the rank delta (positive = moved up the table). When the displayed
    # forecast IS the frozen one, baseline == current so all deltas are 0.
    baseline_pw, pretournament_gen = {}, None
    if os.path.exists(FORECAST_PATH):
        try:
            _frozen = load_json(FORECAST_PATH)
            baseline_pw = _frozen.get("p_win", {})
            pretournament_gen = _frozen.get("generated_at")
        except Exception:
            pass

    # Per-matchday prediction generation dates (distinct pred_as_of per matchday).
    def _md(d):
        return 1 if d <= "2026-06-17" else (2 if d <= "2026-06-23" else 3)
    _mdsets = {1: set(), 2: set(), 3: set()}
    for g in games:
        _mdsets[_md(g["date"])].add(g.get("pred_as_of") or pred.get("as_of"))
    matchday_pred_dates = {k: sorted(v) for k, v in _mdsets.items()}
    baseline_rank = {
        tid: r for r, (tid, _) in enumerate(
            sorted(baseline_pw.items(), key=lambda kv: kv[1], reverse=True), start=1)
    }

    forecast_top10 = []
    p_win = forecast.get("p_win", {})
    cur_sorted = sorted(p_win.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (tid, p) in enumerate(cur_sorted[:10], start=1):
        entry = {"team": f_team_names.get(tid, tid), "pct": round(p * 100, 2)}
        if tid in baseline_pw:
            entry["delta_pct"] = round((p - baseline_pw[tid]) * 100, 2)
            entry["rank_delta"] = baseline_rank.get(tid, rank) - rank  # + = up
        else:
            entry["delta_pct"] = None   # not in the baseline field (new entrant)
            entry["rank_delta"] = None
        forecast_top10.append(entry)

    payload = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tournament": {
            "games_played": n_played,
            "games_total": GAMES_TOTAL_GROUP,
            "start_date": TOURNAMENT_START,
        },
        "kpis": kpis,
        "calibration": calibration,
        "matches": matches,
        "predictions_as_of": pred.get("as_of"),
        "matchday_pred_dates": matchday_pred_dates,
        "pretournament_forecast_generated_at": pretournament_gen,
        "forecast_generated_at": forecast.get("generated_at"),
        "forecast_is_live": forecast.get("is_live", False),
        "forecast_games_conditioned": forecast.get("games_conditioned", 0),
        "forecast_conditioned_through": forecast.get("conditioned_through"),
        "forecast_top10": forecast_top10,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return payload


def main():
    payload = build()
    k = payload["kpis"]
    n = payload["tournament"]["games_played"]
    total = payload["tournament"]["games_total"]
    if n > 0:
        print(
            "performance.json: {}/{} played | 1X2 acc {:.0%} | model exact {:.0%} "
            "| logloss {:.3f} | brier {:.3f}".format(
                n, total, k["accuracy_1x2"], k["model_exact_rate"],
                k["mean_log_loss"], k["mean_brier"],
            )
        )
    else:
        print(
            "performance.json: 0/{} played | awaiting first result "
            "(tournament starts {})".format(total, TOURNAMENT_START)
        )


if __name__ == "__main__":
    main()
