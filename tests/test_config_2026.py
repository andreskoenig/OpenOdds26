"""Structural gate for the 2026 World Cup config (real Dec-5-2025 draw)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "tournament_config_2026.json"


def _load():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _team_ids():
    with open(ROOT / "data" / "teams.json", encoding="utf-8") as f:
        return {t["team_id"] for t in json.load(f)}


def test_parses():
    assert isinstance(_load(), dict)


def test_twelve_groups_of_four_48_unique_teams():
    cfg = _load()
    groups = cfg["groups"]
    assert sorted(groups) == list("ABCDEFGHIJKL")
    for g, members in groups.items():
        assert len(members) == 4, f"group {g} not size 4"
    flat = [t for m in groups.values() for t in m]
    assert len(flat) == 48
    assert len(set(flat)) == 48


def test_all_teams_resolve_to_real_slug_ids():
    cfg = _load()
    ids = _team_ids()
    for members in cfg["groups"].values():
        for tid in members:
            assert tid == tid.lower(), f"{tid} not lowercase slug"
            assert tid in ids, f"{tid} not in teams.json"


def test_hosts_and_advancement():
    cfg = _load()
    assert cfg["host_team_ids"] == ["mexico", "canada", "united_states"]
    assert cfg["groups"]["A"][0] == "mexico"
    assert cfg["groups"]["B"][0] == "canada"
    assert cfg["groups"]["D"][0] == "united_states"
    assert cfg["advance_per_group"] == 2
    assert cfg["best_thirds"] == 8


def test_key_teams_present():
    cfg = _load()
    flat = {t for m in cfg["groups"].values() for t in m}
    for kt in ("spain", "argentina", "france", "england", "brazil"):
        assert kt in flat


def test_knockout_bracket_shape():
    kb = _load()["knockout_bracket"]
    assert len(kb["round_of_32"]) == 16
    assert len(kb["round_of_16"]) == 8
    assert len(kb["quarter_finals"]) == 4
    assert len(kb["semi_finals"]) == 2
    assert kb["third_place"]["match"] and kb["final"]["match"]
    # R32 has exactly 8 third-place slots (T:...) and they reference groups.
    third_slots = [s for m in kb["round_of_32"] for s in (m["home"], m["away"])
                   if str(s).startswith("T:")]
    assert len(third_slots) == 8
