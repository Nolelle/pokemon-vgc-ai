"""Regression tests for the second battle-parsing audit pass (slots, reveals, disguises, Megas).

Line shapes are copied from audit traces; helpers come from `test_poke_env_repairs`.
"""

from __future__ import annotations

import pytest
from poke_env.battle import Field
from test_poke_env_repairs import LEADS, foe, play, without

from vgc.mechanics_state import snapshot_battle
from vgc.poke_env_compat import normalize_for_poke_env

BASE = LEADS[:1] + ["|switch|p2a: Alcremie|Alcremie, L50, F|100/100"]


def test_block_line_reveals_the_holders_ability() -> None:
    lines = BASE + ["|-block|p2a: Alcremie|ability: Aroma Veil|[of] p2a: Alcremie"]
    assert foe(play(lines), "Alcremie").ability == "aromaveil"
    with without("ability_reveal_of_holder"):
        assert foe(play(lines), "Alcremie").ability != "aromaveil"


def test_trace_line_reveals_the_traced_foes_ability() -> None:
    lines = LEADS[:1] + [
        "|switch|p2a: Gardevoir|Gardevoir, L50, F|100/100",
        "|switch|p2b: Alcremie|Alcremie, L50, F|100/100",
        "|-ability|p1a: Heracross|Adaptability|Trace|[from] ability: Trace|[of] p2b: Alcremie",
    ]
    assert foe(play(lines), "Alcremie").ability == "adaptability"


@pytest.mark.parametrize(
    ("line", "item"),
    [
        ("|-activate|p2a: Alcremie|item: Quick Claw", "quickclaw"),
        ("|-heal|p2a: Alcremie|50/100|[from] item: Shell Bell|[of] p1a: Heracross", "shellbell"),
        ("|-mega|p2a: Alcremie|Alcremie|Alcremite", "alcremite"),
    ],
)
def test_effect_lines_reveal_a_foes_held_item(line: str, item: str) -> None:
    assert foe(play(BASE + [line]), "Alcremie").item == item
    with without("item_effect_reveal", "mega_stone_known"):
        assert foe(play(BASE + [line]), "Alcremie").item != item


def test_trick_shows_the_received_item_and_a_consumed_white_herb_stays_gone() -> None:
    shown = play(
        BASE
        + [
            "|move|p2a: Alcremie|Trick|p1a: Heracross",
            "|-activate|p2a: Alcremie|move: Trick|[of] p1a: Heracross",
            "|-item|p2a: Alcremie|Sitrus Berry|[from] move: Trick",
        ]
    )
    assert foe(shown, "Alcremie").item == "sitrusberry"
    herb = play(
        BASE
        + [
            "|move|p2a: Alcremie|Trick|p1a: Heracross",
            "|-activate|p2a: Alcremie|move: Trick|[of] p1a: Heracross",
            "|-enditem|p2a: Alcremie|White Herb",
            "|-item|p2a: Alcremie|White Herb|[from] move: Trick",
        ]
    )
    assert foe(herb, "Alcremie").item is None


def test_trick_or_treat_adds_a_third_type() -> None:
    lines = BASE + ["|-start|p2a: Alcremie|typeadd|Ghost|[from] move: Trick-or-Treat"]
    assert {t.name for t in foe(play(lines), "Alcremie").types} == {"FAIRY", "GHOST"}
    with without("typeadd_third_type"):
        assert {t.name for t in foe(play(lines), "Alcremie").types} == {"FAIRY"}


def test_illusion_break_gives_the_disguise_back_its_own_state() -> None:
    lines = LEADS[:1] + [
        "|switch|p2a: Torkoal|Torkoal, L50, F|100/100",
        "|switch|p2b: Zoroark|Zoroark, L50, F|100/100",
        "|switch|p2b: Alcremie|Alcremie, L50, F|100/100",  # a Zoroark disguised as Alcremie
        "|move|p2b: Alcremie|Nasty Plot|p2b: Alcremie",
        "|-damage|p2b: Alcremie|40/100",
        "|replace|p2b: Zoroark|Zoroark, L50, F",
        "|-end|p2b: Zoroark|Illusion",
    ]
    battle = play(lines)
    assert "nastyplot" not in foe(battle, "Alcremie").moves
    assert "nastyplot" in foe(battle, "Zoroark").moves
    assert foe(battle, "Zoroark").current_hp_fraction == pytest.approx(0.4)
    with without("illusion_restore_disguise"):
        assert "nastyplot" in foe(play(lines), "Alcremie").moves


def test_meganium_is_not_mistaken_for_a_mega_forme() -> None:
    def snap(extra: list[str]):
        battle = play(LEADS[:1] + ["|switch|p2a: Meganium|Meganium, L50, M|100/100", *extra])
        return next(
            m for m in snapshot_battle(battle).opponent_side.pokemon if m.name == "Meganium"
        )

    plain = snap([])
    assert plain.species_id == "meganium" and not plain.mega_evolved
    mega = snap(
        [
            "|detailschange|p2a: Meganium|Meganium-Mega, L50, M",
            "|-mega|p2a: Meganium|Meganium|Meganiumite",
        ]
    )
    assert mega.species_id == "meganiummega" and mega.mega_evolved


def test_fairy_lock_is_recorded_and_ends() -> None:
    battle = play(LEADS + ["|-fieldactivate|move: Fairy Lock", "|upkeep", "|turn|2"])
    assert Field.FAIRY_LOCK in battle.fields
    for turn in (3, 4):
        for line in ("|upkeep", f"|turn|{turn}"):
            battle.parse_message(normalize_for_poke_env(line.split("|")))
    assert Field.FAIRY_LOCK not in battle.fields
