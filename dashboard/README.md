# OpenOdds26 — Live Performance Dashboard

A self-contained dashboard that tracks how the **frozen** pre-tournament World
Cup 2026 predictions are scoring as real results come in. It is **scoring
only** — it never re-runs the (deliberately slow, frozen) forecast/model.

- **No build step, no installs.** Front-end is React 18 via CDN; back-end is
  Python 3 **standard library only** (no numpy/scipy/flask/requests/pip).
- **Read-only w.r.t. the model.** The refresh trigger only refreshes results and
  re-scores; it never re-forecasts.

## Files

| File | Purpose |
|------|---------|
| `build_performance.py` | Compares frozen predictions + EV pool picks against `data/match_results.json`, writes `performance.json`. Stdlib only. |
| `performance.json` | Generated scoreboard the front-end polls. |
| `index.html` | Single-file React-via-CDN dashboard. Fetches `./performance.json` on load and every 60s. |
| `serve.py` | Stdlib `http.server` that serves this directory on `0.0.0.0:8080` and exposes `/api/refresh`. |

## Run it on the Raspberry Pi

### One-time setup
1. Copy/clone the repo to the Pi, e.g. to `/home/pi/FIFAWC`.
2. Ensure Python 3 is present: `python3 --version` (3.7+). No `pip install` needed.

### Serve the dashboard
From the **project root**:
```bash
python3 dashboard/serve.py
```
Then open `http://<pi-ip>:8080/` from any device on the same network
(find the Pi's IP with `hostname -I`). The page auto-polls `performance.json`
every 60s, so new results appear without a manual reload.

Optional overrides: `DASHBOARD_PORT` (default 8080), `DASHBOARD_HOST`
(default 0.0.0.0).

## Keeping it up to date

### Auto-update via cron (recommended)
Refresh results and rebuild the scoreboard every 15 minutes. Edit with
`crontab -e` and add:
```cron
*/15 * * * * cd /home/pi/FIFAWC && /usr/bin/python3 scripts/refresh_match_results.py --allow-baseline-additions >/dev/null 2>&1; /usr/bin/python3 dashboard/fetch_live_results.py >/dev/null 2>&1; /usr/bin/python3 dashboard/build_performance.py >/dev/null 2>&1
```
Three steps, `;`-separated so a transient failure of one source doesn't block
the others: (1) martj42 results for the model, (2) the ESPN live-results layer
(dashboard-only, no key — fills the gap before martj42 catches up), (3) rebuild
the scoreboard. The browser's 60s poll picks up the freshly written
`performance.json` automatically — no need to restart `serve.py` or reload.

(Adjust the repo path and `python3` path to match your Pi; check with
`which python3`.)

### Manual / instant update
Trigger a refresh + rebuild on demand (runs the same two steps as the cron job):
```bash
curl -X POST http://<pi-ip>:8080/api/refresh
# GET also works for convenience:
curl http://<pi-ip>:8080/api/refresh
```
Returns JSON `{"ok": true, "summary": "..."}` (or `ok:false` with the error).

## Run serve.py under systemd (survives reboots)

Create `/etc/systemd/system/openodds26.service`:
```ini
[Unit]
Description=OpenOdds26 live-performance dashboard
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/FIFAWC
ExecStart=/usr/bin/python3 /home/pi/FIFAWC/dashboard/serve.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now openodds26.service
sudo systemctl status openodds26.service
```

## How scoring works (brief)

- Each frozen prediction is matched to an actual result by **date (±1 day)** and
  the **unordered team pair**. Actual goals are **re-oriented** to the
  prediction's home/away before scoring (upstream may list teams in the opposite
  orientation).
- Per game: 1X2 outcome correct?, model modal exact-score correct?, your EV pool
  pick exact/outcome, per-match **log-loss** and **Brier**.
- Pool points (group-stage rule, both pools identical): **exact = 3 pts**
  (implies outcome), **outcome only = 1 pt**, else 0 — scored on the EV pick.
- Aggregates: 1X2 accuracy, model & pool exact rates, mean log-loss, mean Brier,
  cumulative pool points, and a favorite-probability **calibration** breakdown.

## Limitations / assumptions

- **Group stage only (72 games).** Knockout-round predictions are not scored —
  those predictions don't exist until the bracket teams are known.
- **No market-vs-model comparison.** There are no free per-match 2026 odds, so
  that comparison is omitted.
- Only results with `competition == "FIFA World Cup"` and `date >= 2026-06-11`
  whose both team IDs appear in our predictions are counted as played.
- Pre-tournament the dashboard runs cleanly with 0 played games (KPIs show "—",
  all matches flagged upcoming).
