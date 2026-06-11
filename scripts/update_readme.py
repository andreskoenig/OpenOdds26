"""Refresh the README's "Current 2026 prediction" table from forecast_2026.json.

Injects a top-10 P(win) table (model vs Polymarket) plus the run timestamp and
model version between the <!-- PREDICTIONS:START/END --> markers. Run after every
forecast so the GitHub homepage always shows the latest numbers.

Run:  python scripts/update_readme.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
START = "<!-- PREDICTIONS:START -->"
END = "<!-- PREDICTIONS:END -->"
S_START = "<!-- SENSITIVITY:START -->"
S_END = "<!-- SENSITIVITY:END -->"


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


def main():
    fc = _load("data/forecast_2026.json")
    pw = fc["p_win"]
    market = _load("data/polymarket_winner_2026.json")["p_market"]
    names = {t["team_id"]: t["canonical_name"] for t in _load("data/teams.json")}
    version = "dev"
    vpath = os.path.join(ROOT, "VERSION")
    if os.path.exists(vpath):
        with open(vpath, encoding="utf-8") as f:
            version = f.read().strip()

    # timestamp: prefer the stamped field, else the file's mtime
    ts = fc.get("generated_at")
    if not ts:
        mt = os.path.getmtime(os.path.join(ROOT, "data/forecast_2026.json"))
        ts = datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M")

    top = sorted(pw, key=lambda t: -pw[t])[:10]
    lines = [
        f"**Model v{version}** · last run **{ts}** · "
        f"{fc.get('n_sims', '?')} simulations, as-of {fc.get('as_of', '?')}",
        "",
        "| # | Team | Model P(win) | Market (Polymarket) |",
        "|---|------|-------------:|--------------------:|",
    ]
    for i, t in enumerate(top, 1):
        mk = market.get(t)
        mk_s = f"{mk * 100:.1f}%" if mk is not None else "—"
        lines.append(f"| {i} | {names.get(t, t)} | {pw[t] * 100:.1f}% | {mk_s} |")
    lines += ["",
              "_Auto-generated from `data/forecast_2026.json` by "
              "`scripts/update_readme.py`. Market = de-vigged-free Polymarket "
              "winner odds (a model input, not an independent benchmark)._"]
    block = "\n".join(lines)

    readme_path = os.path.join(ROOT, "README.md")
    with open(readme_path, encoding="utf-8") as f:
        text = f.read()
    if START not in text or END not in text:
        raise SystemExit("README is missing the PREDICTIONS markers.")
    pre = text.split(START)[0]
    post = text.split(END)[1]
    new = f"{pre}{START}\n{block}\n{END}{post}"

    # ---- sensitivity band (optional; rendered when the file exists) -------
    sens_path = os.path.join(ROOT, "data", "forecast_sensitivity_2026.json")
    if os.path.exists(sens_path) and S_START in new and S_END in new:
        with open(sens_path, encoding="utf-8") as f:
            sens = json.load(f)
        band, snames = sens["band"], sens["team_names"]
        grid = sens["grid"]
        stop = sorted(band, key=lambda t: -band[t]["headline"])[:10]
        sl = [
            f"How robust is the forecast to its two judgment knobs? Re-run over "
            f"market-prior weight c_m ∈ {grid['c_m']} × recency half-life ∈ "
            f"{grid['half_lives_y']}y ({sens['n_sims_per_cell']} sims/cell, "
            f"{sens['generated_at']}). Narrow band = config-robust; wide band = "
            f"the number is an opinion of the knob settings.",
            "",
            "| Team | Headline | Range across configs |",
            "|------|---------:|---------------------:|",
        ]
        for t in stop:
            b = band[t]
            sl.append(f"| {snames.get(t, t)} | {b['headline']*100:.1f}% "
                      f"| {b['min']*100:.1f}% – {b['max']*100:.1f}% |")
        s_block = "\n".join(sl)
        s_pre = new.split(S_START)[0]
        s_post = new.split(S_END)[1]
        new = f"{s_pre}{S_START}\n{s_block}\n{S_END}{s_post}"

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"README updated: top-10 table, v{version}, run {ts}")


if __name__ == "__main__":
    main()
