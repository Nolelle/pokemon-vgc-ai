"""Tests for curated team-level metagame knowledge."""

from __future__ import annotations

from dataclasses import dataclass

from vgc.meta import (
    known_nature,
    known_set_for_pokemon,
    load_meta_teams,
    recognize_meta_team,
    validate_meta_teams,
)


@dataclass
class _FakePokemon:
    species: str
    item: str | None = None


def test_video_data_contains_ten_complete_teams_and_valid_champions_ids() -> None:
    teams = load_meta_teams()

    assert len(teams) == 10
    assert sum(len(team["sets"]) for team in teams) == 60
    assert validate_meta_teams(teams) == []


def test_recognizes_an_exact_six_species_team_in_any_order() -> None:
    aerozard = next(team for team in load_meta_teams() if team["id"] == "aerozard")
    observed = [_FakePokemon(entry["species_id"]) for entry in reversed(aerozard["sets"])]

    assert recognize_meta_team(observed) == aerozard


def test_does_not_guess_from_only_a_partial_core() -> None:
    observed = [
        _FakePokemon("charizard"),
        _FakePokemon("garchomp"),
        _FakePokemon("kingambit"),
        _FakePokemon("sylveon"),
    ]

    assert recognize_meta_team(observed) is None


def test_known_nature_accepts_an_already_mega_evolved_species() -> None:
    team = next(team for team in load_meta_teams() if team["id"] == "gardevoir_maushold")
    gardevoir = _FakePokemon("gardevoirmega", "gardevoirite")

    assert known_set_for_pokemon(team, gardevoir)["species_id"] == "gardevoir"
    assert known_nature(team, gardevoir) == "modest"


def test_spoken_corrections_are_encoded_instead_of_graphic_errors() -> None:
    teams = {team["id"]: team for team in load_meta_teams()}
    aero_sylveon = known_set_for_pokemon(teams["aerozard"], _FakePokemon("sylveon"))
    pex = known_set_for_pokemon(teams["charizard_y_toxapex"], _FakePokemon("toxapex"))

    assert aero_sylveon["ability_id"] == "pixilate"
    assert pex["ability_id"] == "regenerator"
