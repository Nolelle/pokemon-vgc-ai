"""Ground-truth checks for the reusable Regulation M-B rain team."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.config import FORMAT_ID, SHOWDOWN_REPO
from vgc.node import find_node

from sim_harness import run_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]
PASTE_PATH = REPO_ROOT / "teams" / "regmb_rain_balance.paste.txt"
PACKED_PATH = REPO_ROOT / "teams" / "regmb_rain_balance.packed.txt"
STAT_IDS = ("hp", "atk", "def", "spa", "spd", "spe")


def _team_objects() -> list[dict]:
    result = []
    for entry in Teambuilder.parse_packed_team(PACKED_PATH.read_text().strip()):
        result.append(
            {
                "species": entry.species or entry.nickname,
                "name": entry.nickname,
                "item": entry.item or "",
                "ability": entry.ability or "",
                "moves": entry.moves,
                "nature": entry.nature or "Serious",
                "evs": dict(zip(STAT_IDS, entry.evs, strict=True)),
                "ivs": dict(zip(STAT_IDS, entry.ivs, strict=True)),
                "level": entry.level or 50,
                "gender": entry.gender or "",
            }
        )
    return result


def _decoded_eval(result: dict, index: int) -> dict:
    value = result["evalResults"][index]
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return json.loads(value)


def test_packed_fixture_preserves_all_own_team_information() -> None:
    team = list(Teambuilder.parse_packed_team(PACKED_PATH.read_text().strip()))

    assert len(team) == 6
    assert all(sum(entry.evs) == 66 and max(entry.evs) <= 32 for entry in team)
    assert [(entry.nickname, entry.item, entry.ability) for entry in team] == [
        ("Pelipper", "SitrusBerry", "Drizzle"),
        ("Swampert", "Swampertite", "Torrent"),
        ("Archaludon", "Leftovers", "Stamina"),
        ("Sinistcha", "CobaBerry", "Hospitality"),
        ("Grimmsnarl", "LightClay", "Prankster"),
        ("Metagross", "Metagrossite", "ClearBody"),
    ]


@pytest.mark.integration
def test_paste_is_legal_in_regulation_mb() -> None:
    result = subprocess.run(
        [find_node(), "pokemon-showdown", "validate-team", FORMAT_ID],
        cwd=SHOWDOWN_REPO,
        input=PASTE_PATH.read_text(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not result.stdout.strip()


@pytest.mark.integration
def test_rain_mega_and_second_mega_rejection_match_showdown() -> None:
    team = _team_objects()
    result = run_scenario(
        FORMAT_ID,
        team,
        team,
        [
            "p1 team 2613",
            "p2 team 1435",
            "p1 move 4 mega, move 4",
            "p2 move 4, move 4",
            'eval JSON.stringify({turn:battle.turn,weather:battle.field.weather,'
            'p1:battle.p1.active.map(p=>({species:p.species.id,ability:p.ability,'
            'speed:p.getStat("spe")}))})',
            "p1 move 4, move 4 mega",
            "p2 move 4, move 4",
            'eval JSON.stringify({turn:battle.turn,'
            'p1:battle.p1.active.map(p=>({species:p.species.id,ability:p.ability}))})',
        ],
    )
    after_first = _decoded_eval(result, 0)
    after_rejected_second = _decoded_eval(result, 1)

    assert after_first == {
        "turn": 2,
        "weather": "raindance",
        "p1": [
            {"species": "swampertmega", "ability": "swiftswim", "speed": 244},
            {"species": "metagross", "ability": "clearbody", "speed": 99},
        ],
    }
    assert after_rejected_second["turn"] == 2
    assert after_rejected_second["p1"][1] == {
        "species": "metagross",
        "ability": "clearbody",
    }


@pytest.mark.integration
def test_wide_guard_electro_shot_and_stamina_match_showdown() -> None:
    team = _team_objects()
    result = run_scenario(
        FORMAT_ID,
        team,
        team,
        [
            "p1 team 1342",
            "p2 team 1263",
            "p1 move 4, move 3 1",
            "p2 move 1 2, move 2",
            'eval JSON.stringify({weather:battle.field.weather,'
            'boosts:battle.p1.active[1].boosts})',
        ],
    )
    state = _decoded_eval(result, 0)
    log = result["log"]

    assert state["weather"] == "raindance"
    assert state["boosts"]["def"] == 1
    assert state["boosts"]["spa"] == 1
    assert any("|move|p1b: Archaludon|Electro Shot||[still]" in line for line in log)
    assert sum("|move: Wide Guard" in line for line in log) == 2


@pytest.mark.integration
def test_hospitality_and_rage_powder_match_showdown() -> None:
    team = _team_objects()
    result = run_scenario(
        FORMAT_ID,
        team,
        team,
        [
            "p1 team 1243",
            "p2 team 1362",
            "eval battle.p1.active[1].damage(100)",
            "p1 switch 3, move 4",
            "p2 move 4, move 4",
            "p1 move 2, move 4",
            "p2 move 2 2, move 1 2",
        ],
    )
    log = result["log"]

    assert any("[from] ability: Hospitality" in line for line in log)
    assert any("|move|p2a: Pelipper|Hurricane|p1a: Sinistcha" in line for line in log)
    assert any("|move|p2b: Archaludon|Dragon Pulse|p1a: Sinistcha" in line for line in log)


@pytest.mark.integration
def test_light_clay_and_tailwind_durations_match_showdown() -> None:
    team = _team_objects()
    result = run_scenario(
        FORMAT_ID,
        team,
        team,
        [
            "p1 team 5123",
            "p2 team 1263",
            "p1 move 3, move 3",
            "p2 move 4, move 4",
            'eval JSON.stringify({reflect:battle.p1.sideConditions.reflect.duration,'
            'tailwind:battle.p1.sideConditions.tailwind.duration})',
        ],
    )

    # One end-of-turn decrement has happened: Light Clay's Reflect started at 8 turns,
    # while Tailwind started at 4.
    assert _decoded_eval(result, 0) == {"reflect": 7, "tailwind": 3}
