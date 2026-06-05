# Handoff — Twitter/X thread about the World Cup prediction model

**Purpose:** brief for another LLM to draft a short thread. All numbers below are
from this project's pipeline and are accurate as stated. Flags mark anything the
writer should double-check or attribute carefully. Tone target: smart-but-
accessible, honest (we do NOT claim to beat the bookies), a little fun.

---

## 0. One-line pitch
A from-scratch football forecasting model built only on **free, public data**,
validated on the 2022 World Cup against Bet365's closing odds, then pointed at
2026 with a Polymarket "wisdom-of-the-market" signal folded in.

---

## 1. POST 1 idea — how bookmakers actually price a match
Plain-English background for the opener:

- A bookmaker posts **decimal odds** (e.g. 2.50). The naive implied probability is
  just `1 / odds` (2.50 → 40%).
- Add up the implied probabilities for home/draw/away and they sum to **more than
  100%** — typically 103–107%. That extra slice is the **overround** (a.k.a. the
  "vig" or margin): the bookmaker's built-in edge.
- To recover the bookmaker's *fair* view you **de-vig**: rescale the three
  implied probabilities so they sum to 100%.
- Odds aren't a pure forecast — books also shade them to **balance their book**
  (limit how much they can lose). But the **closing odds** (right before kickoff)
  are considered extremely sharp because they've absorbed all the late money and
  news. Beating the closing line is famously hard.
- **Polymarket is different.** It's a prediction *market*: traders buy/sell
  shares, and the **price is directly the probability**. No bookmaker margin — we
  observed only ~3% overround on the 2026 "World Cup Winner" market, which we just
  normalize to 100% (no de-vig needed).

---

## 2. POST 2–3 idea — what the model is and what we feed it (semi-technical)

**The engine: a Dixon–Coles goals model.**
- Treats each team's goals in a match as a (roughly) Poisson count.
- Every team gets two numbers: an **attack rating** and a **defense rating**.
- Expected goals for A vs B ≈ `exp(baseline + attack_A − defense_B + home_advantage)`,
  with a small correction (the ρ term) for low-scoring/correlated scores (0-0, 1-1).
- Ratings are fit by **maximum likelihood over the whole match history**, with
  **recent matches weighted more** (exponential time decay) and **ridge
  regularization** pulling ratings toward sensible priors.
- The fit is **opponent-adjusted**: scoring 3 vs a strong defense counts more than
  3 vs a weak one (that's the whole point of fitting attack & defense jointly).
- We then run **20,000 Monte Carlo simulations** of the entire tournament (group
  stage + the real 48-team 2026 bracket) to turn ratings into P(win the cup),
  P(reach each round), group tables, etc.

**The priors (what nudges a team before results speak):**
1. **FIFA ranking points** (standardized).
2. **Squad market value** from Transfermarkt — a talent-pool proxy.
3. **(2026 only) the Polymarket winner odds** — a forward-looking "market" prior.

**Two model versions to mention:**
- **Original (validated on 2022):** results + FIFA prior + squad-value prior.
  No market input, recency half-life ~2.4 years. This is the clean test.
- **2026 version:** same engine **plus** three upgrades —
  (a) a **Polymarket "World Cup Winner" prior** (moderate weight),
  (b) **opponent-adjusted goal priors** using point-in-time FIFA strength,
  (c) a **recency half-life of 1.5 years** (chosen by out-of-sample validation,
      not guessed) with a **10-year hard cutoff** on history.
  We also **"de-pathed" the Polymarket odds** — stripped out the luck of the
  bracket draw — so the market prior measures *strength*, not an easy/hard route.

**Data collection — all free / public (good "open-source" angle):**
- **Match results:** `martj42/international_results` (GitHub) — every men's
  international since **1872** (~49,000 matches).
- **FIFA ranking points:** community scrapers (FifaRankingScraper, Dato-Futbol)
  plus the **official FIFA ranking PDF** for the latest 2026 snapshot.
- **Squad values:** Transfermarkt clean-CSV dataset (`dcaribou/transfermarkt-datasets`).
- **2022 odds:** Bet365 closing 1X2 (form-window games + the 64 World Cup games).
- **2026 market:** Polymarket "World Cup Winner" via their **free public API**.
- No paid data feeds anywhere.

---

## 3. POST 4 idea — RESULTS, 2022 World Cup (the honest validation)

Scored on the **64 actual WC2022 games**, match-level 1X2 **log-loss** (lower =
better; uniform-guess baseline = 1.099):

| | log-loss |
|---|---|
| **Our model** | **1.0202** |
| **Bet365 (de-vigged closing odds)** | **0.9986** |

- Honest headline: **Bet365 narrowly won.** When we tuned the best blend of model
  + market, the optimal weight on our model was **0** — the closing line was
  sharper. A free-data, open-method model gets *close to* the world's sharpest
  book but doesn't beat it. (That's the honest, and arguably more interesting,
  story.)

**Model's pre-tournament top 5 to win the cup (our simulation):**
1. Brazil 27.3%
2. **Argentina 13.1%**  ← actual eventual champion (model had them 2nd)
3. Spain 8.1%
4. England 7.5%
5. Portugal 7.2%

- Nice narrative beat: the model's favorite **Brazil** went out in the quarters;
  its #2 **Argentina** actually won it.
- ⚠️ FLAG for writer: Bet365's pre-2022 *outright winner* favorites (publicly:
  Brazil, France, Argentina, England, Spain) are **not** from our pipeline — our
  rigorous market comparison is the **match-level log-loss above**. If you want a
  market "top 5 to win" line, either cite public Bet365 outrights with a source or
  skip it and stick to the log-loss + our model's top 5.

---

## 4. POST 5–6 idea — RESULTS, 2026 World Cup

**Model (final, with Polymarket prior) — top 5 to win:**
1. **Spain 21.4%**
2. Argentina 15.1%
3. France 13.5%
4. England 10.8%
5. Portugal 8.9%

**Polymarket (the market) — top 5 to win:**
1. France 16.3%
2. Spain 15.6%
3. England 11.2%
4. Portugal 9.1%
5. Argentina 8.7%

**The interesting tension (good post material):**
- The model is **higher on Spain (+5.9pts) and Argentina (+6.4pts)** than the
  market, and a touch lower on France.
- Why: shortening the memory to ~1.5 years makes **recent form** dominate, and the
  model rewards **Spain** (Euro 2024 winners, dominant recent run) and **Argentina**
  (reigning champions) more than the bookies do. The market prefers **France**.
- So 2026 is **not** a clean "model vs market" bake-off — the market is one of the
  model's *inputs*. It's a blend that leans market-ward but keeps the model's own
  recent-form convictions. Be transparent about this.

**Group-stage predictions (72 games):** each game gets a 1X2 probability, the
model's **expected goals** per side, a **rounded headline score**, and the **top-3
most-likely exact scorelines** with probabilities. Illustrative lines:

- Germany **4–0** Curaçao (expected goals 4.19–0.44; top score 4-0 @13%)
- Qatar **1–3** Switzerland (0.59–3.04)
- Mexico **2–1** South Africa (2.33–0.51), 78% Mexico win
- Netherlands **1–1** Japan (1.29–1.03) — a genuine coin-flip (42/29/29)

**Teaching moment for a post (the "why is every score 1-0?" insight):**
- The single *most-likely exact score* in football is almost always low (1-0, 1-1,
  0-0) — that's just true (those ARE the most common real scores), and any one
  scoreline is only ~13–17% likely.
- So we report **expected goals** (uses the *whole* probability distribution)
  instead of one misleading "modal" score. That's why mismatches correctly show
  4-0 / 0-3 while even games show 1-1.

---

## 5. Honesty guardrails (please keep these in the thread's spirit)
- We **do not beat the closing line** — and we say so. The hook is "how close can
  free data + open methods get," not "we cracked the bookies."
- 2026 model **uses** the market as an input → don't frame 2026 model-vs-Polymarket
  as the model independently out-predicting the market.
- Numbers are point-in-time (model as-of ~10 June 2026 cutoff; Polymarket snapshot
  early June 2026). Probabilities shift as squads/injuries/odds move.
- Expected goals ≠ post-match xG (shots); here it's the model's *forecast* scoring
  rate.

## 6. Suggested thread skeleton (writer can reorder)
1. Hook + "how do bookies price a game?" (overround / de-vig / closing line).
2. "I built my own from free data" — the Dixon–Coles engine in 3 sentences.
3. Inputs + free data sources (results since 1872, FIFA points, Transfermarkt,
   Polymarket).
4. 2022 reality check vs Bet365 (log-loss table + model top 5 + Argentina beat).
5. 2026 top 5: model vs Polymarket (the Spain/Argentina-vs-France disagreement).
6. Group-stage flavor (a few scorelines) + the "why expected goals, not a single
   score" insight.
7. Honest close: didn't beat the closing line; here's what's open/repeatable.

## 7. Handy phrasings (optional)
- "Implied probability = 1 / decimal odds — but the bookie's numbers sum to ~105%;
  that 5% is their cut."
- "Closing odds are the final boss of forecasting. We got within a whisker — and
  lost on points."
- "The model's pick to *win* 2022 was Brazil. Its silver medal pick, Argentina,
  lifted the trophy."
- "For 2026 the model is a Spain/Argentina believer; the market backs France."
