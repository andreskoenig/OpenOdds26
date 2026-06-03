# FIFAWC — World Cup 2026 Prediction Model

A time-weighted Dixon–Coles goals model with a market-derived "surprise"
factor, evaluated by Monte Carlo simulation of the 48-team bracket.

**[SPEC.md](SPEC.md) is the source of truth.** This README only describes the
layout. The architecture splits into two layers (SPEC §2):

- **Agentic layer (LLM):** an orchestrator (Opus 4.8) plus three Sonnet 4.6
  fetch subagents that acquire and clean web data into the schemas in SPEC §7.
- **Deterministic layer (pure Python, zero LLM calls):** de-vigging, features,
  model fit, scoreline matrix, simulation, evaluation. Reproducible and tested.

> Hard rule: **agents fetch and clean; code computes and predicts.** No
> statistic, probability, or prediction comes from free-form LLM reasoning.

## Layout

```
FIFAWC/
├── SPEC.md                     # source of truth — read first
├── README.md                   # this file
├── .gitignore
├── .claude/
│   ├── settings.json           # orchestrator: Opus 4.8, effort high; subagents -> Sonnet 4.6
│   └── agents/                 # three narrow fetch subagents (SPEC §8)
│       ├── results-fifa-fetcher.md   # -> match_results + fifa_ratings
│       ├── xg-fetcher.md             # -> team_xg
│       └── odds-fetcher.md           # -> match_odds (>=3 books, closing)
├── wc_model/                   # deterministic layer (stubs only)
│   ├── __init__.py
│   ├── schemas.py              # TypedDicts for every table in SPEC §7
│   ├── devig.py                # devig (SPEC §5.1)
│   ├── features.py             # build_features (SPEC §4, §5)
│   ├── model.py                # fit_model + score_matrix (SPEC §3, §4)
│   ├── simulate.py             # simulate_tournament (SPEC §6)
│   └── evaluate.py             # evaluate / validation gates (SPEC §10)
├── data/                       # fetched outputs keyed by team_id
│   ├── teams.json              # canonical team registry (starts [])
│   └── README.md
├── config/
│   ├── tournament_config.json  # hand-maintained stub — confirm vs official FIFA
│   └── README.md
└── tests/
    ├── test_score_matrix.py    # skeleton (TODO markers)
    └── test_simulate.py        # skeleton (TODO markers)
```

## Status

Scaffold only. The deterministic modules are stubs (`raise
NotImplementedError`); tests are empty skeletons. Per SPEC §9, the next step is
to implement and unit-test the deterministic core (`score_matrix`,
`simulate_tournament`, `fit_model`) on hand-entered sample data before any
agent runs. Nothing fetches or computes yet.
