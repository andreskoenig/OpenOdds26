# config/

`tournament_config.json` is the `tournament_config` table from SPEC §7. It is
**hand-maintained, not fetched**, and every value **must be confirmed against
the official FIFA source** before any real run.

Fields (all stubs for now):

- `groups` — the 12 groups of 4 (48 teams)
- `schedule` — the full fixture schedule
- `host_cities` — host nations / cities; seeded `["USA", "CAN", "MEX"]`.
  Host advantage (γ) applies only to USA/CAN/MEX matches played in their own
  country.
- `tiebreak_rules` — group-stage tiebreak rules
- `best_thirds_rule` — selection of the 8 best third-placed teams

Confirm the exact format and rules against the official FIFA source (SPEC §6).
