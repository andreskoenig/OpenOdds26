"""PHASE 1c: confirm the current (2026) squad-value snapshot covers the 48 WC teams."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


sv = _load("data/squad_values.json")
cfg = _load("config/tournament_config_2026.json")
wc48 = {t for g in cfg["groups"].values() for t in g}

# pick the current (latest) snapshot date
dates = sorted({r["as_of_date"] for r in sv})
current = dates[-1]
cur = {r["team_id"]: r for r in sv if r["as_of_date"] == current}
print(f"squad_values snapshot dates: {dates}")
print(f"current snapshot: {current} ({len(cur)} teams)")

missing = sorted(t for t in wc48 if t not in cur)
low = sorted(t for t in wc48 if t in cur and cur[t]["n_players"] < 10)
print(f"WC2026 teams in current squad snapshot: {48 - len(missing)}/48")
if missing:
    print(f"  MISSING (no squad value): {missing}")
if low:
    print(f"  thin (<10 players, will impute z=0): {low}")
covered = [t for t in wc48 if t in cur and cur[t]['n_players'] >= 10]
print(f"  full-coverage (n_players>=10): {len(covered)}/48")
