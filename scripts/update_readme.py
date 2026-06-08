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
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"README updated: top-10 table, v{version}, run {ts}")


if __name__ == "__main__":
    main()
