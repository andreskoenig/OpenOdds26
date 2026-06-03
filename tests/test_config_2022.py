"""Structural checks for the 2022 World Cup tournament config (SPEC §6 format)."""

import json
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "tournament_config_2022.json"


def _load():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_config_parses():
    cfg = _load()
    assert isinstance(cfg, dict)


def test_eight_groups_of_four_with_32_unique_teams():
    cfg = _load()
    groups = cfg["groups"]
    assert len(groups) == 8
    for name, members in groups.items():
        assert len(members) == 4, f"group {name} must have 4 teams"
    all_teams = [t for members in groups.values() for t in members]
    assert len(all_teams) == 32
    assert len(set(all_teams)) == 32  # all unique


def test_advancement_is_top_two_and_no_best_thirds():
    cfg = _load()
    assert cfg["advance_per_group"] == 2
    assert cfg["best_thirds"] == 0


def test_host_is_qatar():
    cfg = _load()
    assert cfg["host_team_ids"] == ["qatar"]
    assert "qatar" in cfg["host_cities"]
    assert "qatar" in cfg["groups"]["A"]


def test_group_teams_are_canonical_slug_ids_present_in_teams_json():
    cfg = _load()
    teams_path = Path(__file__).resolve().parents[1] / "data" / "teams.json"
    with open(teams_path, encoding="utf-8") as f:
        team_ids = {t["team_id"] for t in json.load(f)}
    for members in cfg["groups"].values():
        for tid in members:
            assert tid == tid.lower(), f"{tid} is not a lowercase slug id"
            assert tid in team_ids, f"{tid} not found in data/teams.json"
