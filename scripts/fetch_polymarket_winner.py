"""Fetch Polymarket 'World Cup Winner' market -> normalized P(win) per team.

Polymarket is a prediction market: each team's share price IS its implied
probability (no bookmaker vig). We do NOT de-vig; we only NORMALIZE the 48-team
field to sum to 1 (handles spread/overround). Price = bid/ask midpoint (fallback
last trade). Placeholder/untraded markets (no real team) are dropped.

Writes data/polymarket_winner_2026.json and prints model vs market side-by-side.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}
AS_OF = "2026-06-04"
OVERRIDES = {"usa": "united_states", "united states": "united_states",
             "korea republic": "south_korea", "south korea": "south_korea",
             "ir iran": "iran", "turkiye": "turkey", "czechia": "czech_republic",
             "cote d'ivoire": "ivory_coast", "ivory coast": "ivory_coast",
             "cabo verde": "cape_verde", "dr congo": "dr_congo", "congo dr": "dr_congo",
             "curacao": "curacao", "bosnia": "bosnia_and_herzegovina",
             "bosnia-herzegovina": "bosnia_and_herzegovina",
             "bosnia and herzegovina": "bosnia_and_herzegovina"}


def _slug(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", s.lower())).strip("_")


def _load(p):
    with open(os.path.join(ROOT, p), encoding="utf-8") as f:
        return json.load(f)


def _price(m):
    bb, ba = m.get("bestBid"), m.get("bestAsk")
    try:
        if bb is not None and ba is not None:
            bb, ba = float(bb), float(ba)
            # ignore untraded placeholder books (e.g. bid 0 / ask 1 -> mid 0.5)
            if ba - bb < 0.5:
                return (bb + ba) / 2
    except (TypeError, ValueError):
        pass
    op = m.get("outcomePrices")
    if op:
        try:
            return float(json.loads(op)[0])
        except Exception:
            pass
    ltp = m.get("lastTradePrice")
    return float(ltp) if ltp is not None else None


def main():
    teams = _load("data/teams.json")
    name = {t["team_id"]: t["canonical_name"] for t in teams}
    id_set = {t["team_id"] for t in teams}
    lookup = {}
    for t in teams:
        for nm in [t["canonical_name"], *(t.get("aliases") or [])]:
            if nm:
                lookup[nm.lower()] = t["team_id"]
                lookup[_slug(nm)] = t["team_id"]
        lookup[t["team_id"]] = t["team_id"]

    def resolve(x):
        low = x.lower()
        if low in OVERRIDES and OVERRIDES[low] in id_set:
            return OVERRIDES[low]
        return lookup.get(low) or lookup.get(_slug(x))

    cfg = _load("config/tournament_config_2026.json")
    wc48 = {t for g in cfg["groups"].values() for t in g}

    url = "https://gamma-api.polymarket.com/events?" + urllib.parse.urlencode({"slug": "world-cup-winner"})
    req = urllib.request.Request(url, headers=UA)
    ev = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace"))[0]

    raw = {}        # team_id -> price
    unresolved = []
    for m in ev.get("markets", []):
        title = m.get("groupItemTitle") or ""
        tid = resolve(title)
        p = _price(m)
        if p is None:
            continue
        if tid is None or tid not in wc48:
            if title and not title.lower().startswith("team ") and title.lower() != "other":
                unresolved.append((title, round(p, 3)))
            continue
        raw[tid] = p

    tot = sum(raw.values())
    market = {tid: p / tot for tid, p in raw.items()}  # normalize to sum 1

    out = {
        "as_of": AS_OF, "source": "polymarket world-cup-winner (normalized, not de-vigged)",
        "raw_price_sum": tot, "n_teams_priced": len(raw),
        "p_market": market,
        "raw_price": raw,
        "team_names": {tid: name[tid] for tid in raw},
    }
    pj = os.path.join(ROOT, "data", "polymarket_winner_2026.json")
    json.dump(out, open(pj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # side-by-side vs model
    model = _load("data/forecast_2026.json")["p_win"]
    print("=" * 66)
    print("MODEL vs POLYMARKET — P(win 2026 World Cup)")
    print("=" * 66)
    print(f"polymarket priced {len(raw)}/48 WC teams | raw price sum {tot:.3f} "
          f"(overround {100*(tot-1):+.0f}%) -> normalized to 100%")
    if unresolved:
        print(f"non-WC names priced by Polymarket (ignored): {unresolved[:10]}")
    missing = sorted(wc48 - set(raw))
    if missing:
        print(f"WC teams NOT priced by Polymarket ({len(missing)}): {[name[t] for t in missing]}")
    print(f"\n  {'team':<18}{'model':>8}{'market':>9}{'diff(mkt-mdl)':>15}")
    order = sorted(wc48, key=lambda t: -market.get(t, 0))
    for t in order[:16]:
        mdl = model.get(t, 0) * 100
        mkt = market.get(t, 0) * 100
        print(f"  {name[t]:<18}{mdl:>7.1f}%{mkt:>8.1f}%{mkt-mdl:>+14.1f}")
    print(f"\nwrote {pj}")


if __name__ == "__main__":
    main()
