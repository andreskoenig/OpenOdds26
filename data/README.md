# data/

Holds the fetch subagents' outputs, all **keyed on canonical `team_id`**
(SPEC §7). `teams.json` is the canonical team registry (`team_id`,
`canonical_name`, `aliases[]`, `confederation`) that every fetcher resolves
names against; it starts empty (`[]`).

Other tables written here by the agents (one file/dataset per contract):

- `match_results` — historical matches
- `fifa_ratings` — point-in-time FIFA ranking snapshots
- `team_xg` — per-match expected goals
- `match_odds` — pre-match closing 1X2 odds (≥3 books)

All series carry dates so the deterministic layer can enforce an as-of cutoff
(no look-ahead).
