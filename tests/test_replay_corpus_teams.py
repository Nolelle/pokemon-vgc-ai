"""Legal packed teams used by the varied decision-replay corpus."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.config import FORMAT_ID, SHOWDOWN_REPO, TEAMS_DIR
from vgc.damage import FieldState, PokemonState, damage_range, to_id
from vgc.data import load_moves
from vgc.node import find_node
from vgc.stats import calculate_stats

CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "replay_corpus"
CORPUS_TEAMS = sorted(CORPUS_DIR.glob("*.packed.txt"))
STAT_IDS = ("hp", "atk", "def", "spa", "spd", "spe")
SCRIPTED_MOVE_IDS = (
    "eruption",
    "protect",
    "electricterrain",
    "charge",
    "trickroom",
)


def _packed_entries(path: Path):
    return list(Teambuilder.parse_packed_team(path.read_text().strip()))


def _species_id(entry) -> str:
    return to_id(entry.species or entry.nickname)


def _sp_spread(entry) -> dict[str, int]:
    return dict(zip(STAT_IDS, entry.evs, strict=True))


def _pokemon_state(entry) -> PokemonState:
    return PokemonState(
        species_id=_species_id(entry),
        sp_spread=_sp_spread(entry),
        nature=to_id(entry.nature) or "serious",
        item=entry.item,
        ability=entry.ability,
    )


@pytest.mark.parametrize(
    "path", CORPUS_TEAMS, ids=lambda path: path.name.removesuffix(".packed.txt")
)
def test_replay_corpus_teams_are_legal(path: Path) -> None:
    packed = path.read_text().strip()
    parsed = list(Teambuilder.parse_packed_team(packed))
    assert len(parsed) == 6
    assert all(sum(entry.evs) == 66 and max(entry.evs) <= 32 for entry in parsed)

    if not SHOWDOWN_REPO.exists():
        pytest.skip(f"Showdown checkout missing at {SHOWDOWN_REPO}")

    result = subprocess.run(
        [find_node(), "pokemon-showdown", "validate-team", FORMAT_ID],
        cwd=SHOWDOWN_REPO,
        input=packed + "\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stdout.strip()


def test_scripted_corpus_moves_never_miss() -> None:
    moves = load_moves()
    for move_id in SCRIPTED_MOVE_IDS:
        accuracy = moves[move_id]["accuracy"]
        assert accuracy is True or accuracy == 100, (move_id, accuracy)


def test_nuke_leads_are_ohkoed_by_eruption_at_min_roll() -> None:
    torkoal = next(
        entry
        for entry in _packed_entries(TEAMS_DIR / "dev.packed.txt")
        if _species_id(entry) == "torkoal"
    )
    leads = _packed_entries(CORPUS_DIR / "frail_leads.packed.txt")[:2]
    assert [_species_id(entry) for entry in leads] == ["jolteon", "rotommow"]
    attacker = _pokemon_state(torkoal)
    field = FieldState(weather="sun", is_doubles=True, num_targets=2)
    for lead in leads:
        defender = _pokemon_state(lead)
        result = damage_range(attacker, defender, "eruption", field)
        hp = defender.stats()["hp"]
        assert result.min_damage >= hp, (lead.nickname, result.min_damage, hp)


def test_terrain_setter_outspeeds_every_opposing_pokemon() -> None:
    jolteon = _packed_entries(CORPUS_DIR / "frail_leads.packed.txt")[0]
    assert _species_id(jolteon) == "jolteon"
    setter_speed = calculate_stats(
        "jolteon", _sp_spread(jolteon), to_id(jolteon.nature) or "serious"
    )["spe"]
    opposing = [
        calculate_stats(_species_id(entry), _sp_spread(entry), to_id(entry.nature) or "serious")[
            "spe"
        ]
        for entry in _packed_entries(TEAMS_DIR / "dev.packed.txt")
    ]
    assert setter_speed > max(opposing), (setter_speed, opposing)
