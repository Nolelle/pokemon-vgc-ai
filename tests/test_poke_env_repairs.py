"""Regression tests for the poke-env repairs `offline/audit_battle_parsing.py` found.

Each test replays real protocol line shapes (copied from audit traces) through a poke-env
`DoubleBattle` and checks the state against what Showdown does. Where it is cheap the same
lines are replayed with the repair switched off, to show the test fails without it.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import pytest
from poke_env.battle import DoubleBattle, Effect, Pokemon

from vgc import poke_env_compat as compat
from vgc.poke_env_compat import normalize_for_poke_env
from vgc.mechanics_state import snapshot_battle

HEADER = [
    "|player|p1|me||",
    "|player|p2|foe||",
]


def play(lines: list[str], *, role: str = "p1") -> DoubleBattle:
    battle = DoubleBattle(
        "battle-test", "me" if role == "p1" else "foe", logging.getLogger("t"), gen=9
    )
    battle.player_role = role
    for line in HEADER + lines:
        battle.parse_message(normalize_for_poke_env(line.split("|")))
    return battle


@contextmanager
def without(*fixes: str):
    compat.set_disabled_fixes(fixes)
    try:
        yield
    finally:
        compat.set_disabled_fixes([])


def foe(battle: DoubleBattle, name: str) -> Pokemon:
    return battle.opponent_team[f"p2: {name}"]


def own(battle: DoubleBattle, name: str) -> Pokemon:
    return battle.team[f"p1: {name}"]


LEADS = [
    "|switch|p1a: Heracross|Heracross, L50, F|100/100",
    "|switch|p1b: Vivillon|Vivillon, L50, M|100/100",
    "|switch|p2a: Alcremie|Alcremie, L50, F|100/100",
    "|switch|p2b: Torkoal|Torkoal, L50, F|100/100",
    "|turn|1",
]


# --- Champions Mega abilities -------------------------------------------------------------


@pytest.mark.parametrize(
    ("forme", "ability"),
    [
        ("Golisopod-Mega", "toughclaws"),
        ("Garchomp-Mega-Z", "levitate"),
        ("Staraptor-Mega", "contrary"),
        ("Lucario-Mega-Z", "auraguard"),
    ],
)
def test_champions_mega_ability_overrides_poke_envs_vanilla_dex(forme: str, ability: str) -> None:
    base = forme.split("-Mega")[0]
    battle = play(
        LEADS[:1]
        + [
            f"|switch|p2a: {base}|{base}, L50, F|100/100",
            f"|detailschange|p2a: {base}|{forme}, L50, F",
        ]
    )
    assert foe(battle, base).ability == ability
    with without("champions_mega_ability"):
        battle = play(
            LEADS[:1]
            + [
                f"|switch|p2a: {base}|{base}, L50, F|100/100",
                f"|detailschange|p2a: {base}|{forme}, L50, F",
            ]
        )
        assert foe(battle, base).ability != ability


def test_snapshot_recognises_a_mega_whose_ability_differs_from_poke_env() -> None:
    battle = play(
        LEADS[:1]
        + [
            "|switch|p2a: Golisopod|Golisopod, L50, M|100/100",
            "|detailschange|p2a: Golisopod|Golisopod-Mega, L50, M",
        ]
    )
    mon = next(m for m in snapshot_battle(battle).opponent_side.pokemon if m.name == "Golisopod")
    assert mon.species_id == "golisopodmega" and mon.mega_evolved


def test_floette_mega_is_found_even_though_it_changes_from_floette_eternal() -> None:
    battle = play(
        LEADS[:1]
        + [
            "|switch|p2a: Floette|Floette-Eternal, L50, F|100/100",
            "|detailschange|p2a: Floette|Floette-Mega, L50, F",
        ]
    )
    mon = next(m for m in snapshot_battle(battle).opponent_side.pokemon if m.name == "Floette")
    assert mon.species_id == "floettemega"


# --- single-turn effects and momentary markers --------------------------------------------


def test_singleturn_effects_end_with_the_turn() -> None:
    battle = play(
        LEADS
        + [
            "|move|p2a: Alcremie|Endure|p2a: Alcremie",
            "|-singleturn|p2a: Alcremie|move: Endure",
            "|move|p2b: Torkoal|Helping Hand|p2a: Alcremie",
            "|-singleturn|p2a: Alcremie|Helping Hand|[of] p2b: Torkoal",
        ]
    )
    assert {Effect.ENDURE, Effect.HELPING_HAND} <= set(foe(battle, "Alcremie").effects)
    battle.parse_message(["", "upkeep"])
    battle.parse_message(["", "turn", "2"])
    assert not set(foe(battle, "Alcremie").effects) & {Effect.ENDURE, Effect.HELPING_HAND}


def test_destiny_bond_ends_when_its_user_next_acts() -> None:
    battle = play(
        LEADS
        + [
            "|move|p2a: Alcremie|Destiny Bond|p2a: Alcremie",
            "|-singlemove|p2a: Alcremie|Destiny Bond",
            "|upkeep",
            "|turn|2",
        ]
    )
    assert Effect.DESTINY_BOND in foe(battle, "Alcremie").effects  # survives the turn
    battle.parse_message("|move|p2a: Alcremie|Moonblast|p1a: Heracross".split("|"))
    assert Effect.DESTINY_BOND not in foe(battle, "Alcremie").effects


def test_activation_markers_do_not_outlive_the_turn_but_traps_do() -> None:
    lines = LEADS + [
        "|-activate|p2a: Alcremie|move: Struggle",
        "|-activate|p2a: Alcremie|item: Quick Claw",
        "|-activate|p1a: Heracross|move: Whirlpool|[of] p2b: Torkoal",
        "|upkeep",
        "|turn|2",
    ]
    battle = play(lines)
    assert not set(foe(battle, "Alcremie").effects) & {Effect.STRUGGLE, Effect.QUICK_CLAW}
    assert Effect.WHIRLPOOL in own(battle, "Heracross").effects
    with without("momentary_activate_effects"):
        stuck = play(lines)
        assert Effect.STRUGGLE in foe(stuck, "Alcremie").effects


def test_snapshot_reports_showdown_volatile_ids_not_poke_env_markers() -> None:
    battle = play(
        LEADS
        + [
            "|-activate|p1a: Heracross|move: Whirlpool|[of] p2b: Torkoal",
            "|-start|p2a: Alcremie|typechange|Ghost|[from] move: Trick-or-Treat",
            "|-start|p2b: Torkoal|move: Future Sight",
        ]
    )
    ids = {e.id for m in snapshot_battle(battle).our_side.pokemon for e in m.effects}
    assert "partiallytrapped" in ids and "whirlpool" not in ids
    foe_ids = {e.id for m in snapshot_battle(battle).opponent_side.pokemon for e in m.effects}
    assert not foe_ids & {"typechange", "futuresight"}


# --- charge moves ------------------------------------------------------------------------


def test_a_skipped_charge_turn_clears_preparing() -> None:
    lines = LEADS + [
        "|move|p2a: Alcremie|Solar Beam||[still]",
        "|-prepare|p2a: Alcremie|Solar Beam",
        "|-anim|p2a: Alcremie|Solar Beam|p1a: Heracross",
    ]
    assert not foe(play(lines), "Alcremie").preparing
    with without("skipped_charge"):
        assert foe(play(lines), "Alcremie").preparing


def test_a_real_charge_turn_stays_preparing_and_cant_cancels_it() -> None:
    charging = LEADS + [
        "|move|p2a: Alcremie|Dig||[still]",
        "|-prepare|p2a: Alcremie|Dig",
    ]
    assert foe(play(charging), "Alcremie").preparing
    interrupted = charging + ["|cant|p2a: Alcremie|par"]
    assert not foe(play(interrupted), "Alcremie").preparing


# --- abilities ---------------------------------------------------------------------------


def test_worry_seed_result_is_temporary_and_names_the_real_ability() -> None:
    lines = LEADS + [
        "|move|p1a: Heracross|Worry Seed|p2a: Alcremie",
        "|-ability|p2a: Alcremie|Insomnia|Sweet Veil|[from] move: Worry Seed",
    ]
    battle = play(lines)
    mon = foe(battle, "Alcremie")
    assert mon.ability == "insomnia"
    assert mon.base_ability == "sweetveil"  # the line names the OLD ability
    battle.parse_message("|switch|p2a: Torkoal2|Torkoal, L50, F|100/100".split("|"))
    assert foe(battle, "Alcremie").ability == "sweetveil"
    with without("move_changed_ability"):
        stuck = play(lines)
        stuck.parse_message("|switch|p2a: Torkoal2|Torkoal, L50, F|100/100".split("|"))
        assert foe(stuck, "Alcremie").ability == "insomnia"


@pytest.mark.parametrize(
    ("line", "mon", "ability"),
    [
        ("|-weather|SunnyDay|[from] ability: Drought|[of] p2b: Torkoal", "Torkoal", "drought"),
        (
            "|-fieldstart|move: Grassy Terrain|[from] ability: Grassy Surge|[of] p2a: Alcremie",
            "Alcremie",
            "grassysurge",
        ),
        ("|-immune|p2a: Alcremie|[from] ability: Levitate", "Alcremie", "levitate"),
        (
            "|-damage|p1a: Heracross|80/100|[from] ability: Rough Skin|[of] p2b: Torkoal",
            "Torkoal",
            "roughskin",
        ),
        (
            "|-item|p1a: Heracross|Leftovers|[from] ability: Frisk|[of] p2a: Alcremie",
            "Alcremie",
            "frisk",
        ),
        (
            "|-item|p2a: Alcremie|Sitrus Berry|[from] ability: Pickpocket|[of] p1a: Heracross",
            "Alcremie",
            "pickpocket",
        ),
        (
            "|-heal|p2a: Alcremie|100/100|[from] ability: Water Absorb|[of] p1a: Heracross",
            "Alcremie",
            "waterabsorb",
        ),
        (
            "|-heal|p2a: Alcremie|100/100|[from] ability: Hospitality|[of] p2b: Torkoal",
            "Torkoal",
            "hospitality",
        ),
        ("|-activate|p2b: Torkoal|ability: Pressure", "Torkoal", "pressure"),
    ],
)
def test_ability_is_revealed_by_who_it_acted_on(line: str, mon: str, ability: str) -> None:
    # Torkoal and Alcremie have several abilities, so poke-env cannot infer them.
    battle = play(LEADS + [line])
    mon_obj = foe(battle, mon)
    if len(mon_obj.possible_abilities) > 1:
        assert mon_obj.ability == ability
    else:
        assert mon_obj.ability == mon_obj.possible_abilities[0]


def test_reveal_never_touches_our_own_pokemon_or_attackers_of_an_absorb() -> None:
    battle = play(
        LEADS + ["|-heal|p2a: Alcremie|100/100|[from] ability: Water Absorb|[of] p1a: Heracross"]
    )
    assert own(battle, "Heracross").ability != "waterabsorb"


def test_ability_revealed_after_skill_swap_does_not_outlive_the_switch() -> None:
    battle = play(
        LEADS
        + [
            "|move|p2b: Torkoal|Skill Swap|p2a: Alcremie",
            "|-activate|p2b: Torkoal|Skill Swap|||[of] p2a: Alcremie",
            "|-item|p1a: Heracross|Leftovers|[from] ability: Frisk|[of] p2a: Alcremie",
        ]
    )
    mon = foe(battle, "Alcremie")
    assert mon.ability == "frisk"
    battle.parse_message("|switch|p2a: Torkoal2|Torkoal, L50, F|100/100".split("|"))
    assert foe(battle, "Alcremie").ability != "frisk"


def test_gastro_acid_marks_the_target() -> None:
    battle = play(
        LEADS + ["|move|p1a: Heracross|Gastro Acid|p2a: Alcremie", "|-endability|p2a: Alcremie"]
    )
    assert Effect.GASTRO_ACID in foe(battle, "Alcremie").effects


# --- Protect counter, sleep and toxic counters ---------------------------------------------


def test_confusion_self_hit_resets_the_protect_chain() -> None:
    lines = LEADS + [
        "|move|p2a: Alcremie|Protect|p2a: Alcremie",
        "|-singleturn|p2a: Alcremie|Protect",
        "|upkeep",
        "|turn|2",
        "|-activate|p2a: Alcremie|confusion",
        "|-damage|p2a: Alcremie|80/100|[from] confusion",
    ]
    assert foe(play(lines), "Alcremie").protect_counter == 0
    with without("confusion_resets_protect"):
        assert foe(play(lines), "Alcremie").protect_counter == 1


def test_a_reflected_move_does_not_break_the_protect_chain() -> None:
    battle = play(
        LEADS
        + [
            "|move|p2a: Alcremie|Protect|p2a: Alcremie",
            "|-singleturn|p2a: Alcremie|Protect",
            "|move|p2a: Alcremie|Stealth Rock|p1a: Heracross|[from] ability: Magic Bounce",
        ]
    )
    assert foe(battle, "Alcremie").protect_counter == 1


@pytest.mark.parametrize("called", [False, True])
def test_sleep_talk_and_snore_count_one_sleep_turn(called: bool) -> None:
    lines = LEADS + [
        "|-status|p2a: Alcremie|slp",
        "|upkeep",
        "|turn|2",
        "|cant|p2a: Alcremie|slp",
    ]
    if called:
        lines += [
            "|move|p2a: Alcremie|Sleep Talk|p2a: Alcremie",
            "|move|p2a: Alcremie|Protect||[from] move: Sleep Talk|[still]",
        ]
    else:
        lines += ["|move|p2a: Alcremie|Snore|p1a: Heracross"]
    assert foe(play(lines), "Alcremie").status_counter == 1
    with without("sleep_counter"):
        assert foe(play(lines), "Alcremie").status_counter > 1


def test_a_new_status_starts_its_own_counter() -> None:
    battle = play(
        LEADS
        + [
            "|-status|p2a: Alcremie|tox",
            "|upkeep",
            "|turn|2",
            "|upkeep",
            "|turn|3",
            "|move|p2a: Alcremie|Rest|p2a: Alcremie",
            "|-status|p2a: Alcremie|slp|[from] move: Rest",
        ]
    )
    assert foe(battle, "Alcremie").status_counter == 0


def test_toxic_replacement_after_the_residual_phase_gets_no_tick() -> None:
    lines = LEADS + [
        "|upkeep",
        "|switch|p2a: Dragonite|Dragonite, L50, M|100/100",
        "|-status|p2a: Dragonite|tox",
        "|turn|2",
    ]
    assert foe(play(lines), "Dragonite").status_counter == 0
    with without("toxic_replacement"):
        assert foe(play(lines), "Dragonite").status_counter == 1


# --- boosts handed over, Baton Pass, Illusion ---------------------------------------------


def test_baton_pass_hands_over_boosts_and_volatiles() -> None:
    lines = LEADS + [
        "|-boost|p2a: Alcremie|atk|2",
        "|-start|p2a: Alcremie|Substitute",
        "|move|p2a: Alcremie|Baton Pass|p2a: Alcremie",
        "|switch|p2a: Dragonite|Dragonite, L50, M|60/100|[from] Baton Pass",
    ]
    battle = play(lines)
    mon = foe(battle, "Dragonite")
    assert mon.boosts["atk"] == 2 and Effect.SUBSTITUTE in mon.effects
    assert foe(battle, "Alcremie").boosts["atk"] == 0
    with without("baton_pass"):
        assert foe(play(lines), "Dragonite").boosts["atk"] == 0


def test_an_ordinary_switch_still_clears_boosts() -> None:
    battle = play(
        LEADS
        + [
            "|-boost|p2a: Alcremie|atk|2",
            "|switch|p2a: Dragonite|Dragonite, L50, M|100/100",
        ]
    )
    assert foe(battle, "Dragonite").boosts["atk"] == 0


def test_psych_up_also_copies_the_crit_stage_volatiles() -> None:
    lines = LEADS + [
        "|-start|p2a: Alcremie|move: Focus Energy",
        "|-start|p1a: Heracross|move: Dragon Cheer",
        "|move|p1a: Heracross|Psych Up|p2a: Alcremie",
        "|-copyboost|p1a: Heracross|p2a: Alcremie|[from] move: Psych Up",
    ]
    battle = play(lines)
    assert Effect.FOCUS_ENERGY in own(battle, "Heracross").effects
    assert Effect.DRAGON_CHEER not in own(battle, "Heracross").effects  # silently replaced


def test_illusion_break_keeps_the_boosts_earned_under_the_disguise() -> None:
    battle = play(
        LEADS
        + [
            "|-unboost|p2a: Alcremie|atk|1",
            "|replace|p2a: Zoroark|Zoroark, L50, M",
        ]
    )
    assert foe(battle, "Zoroark").boosts["atk"] == -1


# --- Flash Fire, Regenerator, formes -----------------------------------------------------


def test_flash_fire_boost_outlasts_the_first_fire_move() -> None:
    lines = LEADS + [
        "|-start|p2b: Torkoal|ability: Flash Fire",
        "|move|p2b: Torkoal|Eruption|p1a: Heracross",
    ]
    assert Effect.FLASH_FIRE in foe(play(lines), "Torkoal").effects
    with without("flash_fire_persists"):
        assert Effect.FLASH_FIRE not in foe(play(lines), "Torkoal").effects


def test_regenerator_heal_is_not_applied_twice() -> None:
    lines = LEADS + [
        "|-damage|p2a: Alcremie|40/100",
        "|-heal|p2a: Alcremie|73/100|[from] ability: Regenerator|[silent]",
        "|switch|p2a: Dragonite|Dragonite, L50, M|100/100",
    ]
    battle = play(lines)
    foe(battle, "Alcremie")._ability = "regenerator"  # poke-env needs it known to heal itself
    battle = play(lines[:-1])
    foe(battle, "Alcremie")._ability = "regenerator"
    battle.parse_message(lines[-1].split("|"))
    assert foe(battle, "Alcremie").current_hp == 73


def test_permanent_forme_change_updates_the_species() -> None:
    battle = play(
        LEADS
        + [
            "|switch|p2a: Palafin|Palafin, L50, M|100/100",
            "|detailschange|p2a: Palafin|Palafin-Hero, L50, M",
        ]
    )
    assert foe(battle, "Palafin").species == "palafinhero"
    with without("forme_species"):
        stale = play(
            LEADS
            + [
                "|switch|p2a: Palafin|Palafin, L50, M|100/100",
                "|detailschange|p2a: Palafin|Palafin-Hero, L50, M",
            ]
        )
        assert foe(stale, "Palafin").species == "palafin"


def test_temporary_forme_change_reverts_on_switch_out() -> None:
    battle = play(
        LEADS
        + [
            "|switch|p2a: Aegislash|Aegislash, L50, M|100/100",
            "|-formechange|p2a: Aegislash|Aegislash-Blade|[from] ability: Stance Change",
        ]
    )
    assert foe(battle, "Aegislash").species == "aegislashblade"
    battle.parse_message("|switch|p2a: Dragonite|Dragonite, L50, M|100/100".split("|"))
    assert foe(battle, "Aegislash").species == "aegislash"


def test_mega_species_is_not_renamed_by_the_forme_repair() -> None:
    battle = play(
        LEADS
        + [
            "|switch|p2a: Garchomp|Garchomp, L50, F|100/100",
            "|detailschange|p2a: Garchomp|Garchomp-Mega-Z, L50, F",
        ]
    )
    assert foe(battle, "Garchomp").species == "garchomp"


# --- exact state / memory -----------------------------------------------------------------


def test_layered_side_conditions_report_layers_not_a_start_turn() -> None:
    from vgc.condition_clock import observe_condition_line

    battle = play(LEADS)
    for turn in (2, 3, 4):
        for line in ("|upkeep", f"|turn|{turn}"):
            observe_condition_line(battle, line.split("|"))
            battle.parse_message(line.split("|"))
    for line in (
        "|move|p2a: Alcremie|Toxic Spikes|p1a: Heracross",
        "|-sidestart|p1: me|move: Toxic Spikes",
        "|move|p2a: Alcremie|Toxic Spikes|p1a: Heracross",
        "|-sidestart|p1: me|move: Toxic Spikes",
    ):
        observe_condition_line(battle, line.split("|"))
        battle.parse_message(line.split("|"))
    for line in ("|upkeep", "|turn|5", "|upkeep", "|turn|6"):
        observe_condition_line(battle, line.split("|"))
        battle.parse_message(line.split("|"))
    spikes = {e.id: e for e in snapshot_battle(battle).our_side.side_conditions}["toxicspikes"]
    assert spikes.counter_kind == "layers" and spikes.turns == 2
    with without("snapshot_layers"):
        legacy = {e.id: e for e in snapshot_battle(battle).our_side.side_conditions}["toxicspikes"]
        assert legacy.turns != 2


def test_light_clay_screens_last_eight_turns_for_our_own_setter() -> None:
    from vgc.condition_clock import observe_condition_line, remaining_turns

    battle = play(LEADS)
    own(battle, "Heracross").item = "lightclay"
    for line in (
        "|move|p1a: Heracross|Reflect|p1a: Heracross",
        "|-sidestart|p1: me|Reflect",
        "|upkeep",
        "|turn|2",
    ):
        observe_condition_line(battle, line.split("|"))
        battle.parse_message(line.split("|"))
    assert remaining_turns(battle, "side", "reflect", "p1") == 7
    with without("screen_clock"):
        assert remaining_turns(battle, "side", "reflect", "p1") != 7


def test_a_foe_item_that_is_gone_reads_as_consumed_even_with_nothing_else_revealed() -> None:
    battle = play(LEADS + ["|-enditem|p2a: Alcremie|Sitrus Berry"])
    mon = next(m for m in snapshot_battle(battle).opponent_side.pokemon if m.name == "Alcremie")
    assert mon.item_state == "consumed"
    with without("snapshot_foe_item"):
        legacy = next(
            m for m in snapshot_battle(battle).opponent_side.pokemon if m.name == "Alcremie"
        )
        assert legacy.item_state == "unknown"


def test_battle_memory_does_not_record_a_changed_ability_as_the_base() -> None:
    from vgc.battle_memory import BattleMemory

    memory = BattleMemory(battle_tag="t", our_role="p1")
    memory.observe_protocol(
        [
            "|switch|p2a: Serperior|Serperior, L50, F|100/100".split("|"),
            "|-ability|p2a: Serperior|Insomnia|Overgrow|[from] move: Worry Seed".split("|"),
        ]
    )
    assert memory.opponent_abilities.get("serperior") == "overgrow"
    memory.observe_protocol(
        [
            "|switch|p2b: Gardevoir|Gardevoir, L50, F|100/100".split("|"),
            "|-ability|p2b: Gardevoir|Intimidate|[from] ability: Trace|[of] p1a: Heracross".split(
                "|"
            ),
        ]
    )
    assert memory.opponent_abilities.get("gardevoir") == "trace"


# --- round-2 audit findings -------------------------------------------------------------


def test_baton_pass_carries_throat_chop() -> None:
    lines = LEADS + [
        "|-start|p2a: Alcremie|Throat Chop",
        "|move|p2a: Alcremie|Baton Pass|p2a: Alcremie",
        "|switch|p2a: Dragonite|Dragonite, L50, M|60/100|[from] Baton Pass",
    ]
    assert Effect.THROAT_CHOP in foe(play(lines), "Dragonite").effects
    ordinary = lines[:-1] + ["|switch|p2a: Dragonite|Dragonite, L50, M|60/100"]
    assert Effect.THROAT_CHOP not in foe(play(ordinary), "Dragonite").effects


def _alcremie_with_shell_bell(extra: list[str]) -> DoubleBattle:
    battle = play(LEADS)
    foe(battle, "Alcremie")._item = "shellbell"
    for line in extra:
        battle.parse_message(line.split("|"))
    return battle


def test_cud_chew_replaying_a_berry_does_not_consume_the_held_item() -> None:
    lines = [
        "|-activate|p2a: Alcremie|ability: Cud Chew",
        "|-enditem|p2a: Alcremie|Pecha Berry|[eat]",
    ]
    assert foe(_alcremie_with_shell_bell(lines), "Alcremie").item == "shellbell"
    with without("cud_chew_keeps_item"):
        assert foe(_alcremie_with_shell_bell(lines), "Alcremie").item is None


def test_a_stolen_white_herb_logged_eaten_before_received_is_gone() -> None:
    lines = LEADS + [
        "|move|p2b: Torkoal|Thief|p1a: Heracross",
        "|-enditem|p2b: Torkoal|White Herb",
        "|-clearnegativeboost|p2b: Torkoal|[silent]",
        "|-enditem|p1a: Heracross|White Herb|[silent]|[from] move: Thief|[of] p2b: Torkoal",
        "|-item|p2b: Torkoal|White Herb|[from] move: Thief|[of] p1a: Heracross",
    ]
    assert foe(play(lines), "Torkoal").item is None
    with without("stolen_item_eaten"):
        assert foe(play(lines), "Torkoal").item == "whiteherb"


def test_ally_skill_swap_twice_returns_the_abilities_even_when_one_fainted() -> None:
    battle = play(LEADS)
    heracross, vivillon = own(battle, "Heracross"), own(battle, "Vivillon")
    heracross._ability, vivillon._ability = "wanderingspirit", "scrappy"
    for line in (
        "|-activate|p1b: Vivillon|Skill Swap|||[of] p1a: Heracross",
        "|-damage|p1b: Vivillon|0 fnt",
        "|-activate|p1a: Heracross|Skill Swap|||[of] p1b: Vivillon",
    ):
        battle.parse_message(line.split("|"))
    assert heracross.ability == "wanderingspirit"


def test_symbiosis_item_logged_before_the_berry_it_replaces_survives() -> None:
    lines = [
        "|-activate|p1a: Heracross|ability: Symbiosis|Shell Bell|[of] p2a: Alcremie",
        "|-enditem|p2a: Alcremie|Shuca Berry|[weaken]",
    ]
    battle = play(LEADS)
    foe(battle, "Alcremie")._item = "shucaberry"
    for line in lines[:1]:
        battle.parse_message(line.split("|"))
    assert foe(battle, "Alcremie").item == "shellbell"
    battle.parse_message(lines[1].split("|"))
    assert foe(battle, "Alcremie").item == "shellbell"


def test_armor_tail_cant_line_does_not_reset_the_holders_protect_chain() -> None:
    lines = LEADS + [
        "|move|p2a: Alcremie|Protect|p2a: Alcremie",
        "|-singleturn|p2a: Alcremie|Protect",
        "|cant|p2a: Alcremie|ability: Armor Tail|Sucker Punch|[of] p1a: Heracross",
    ]
    assert foe(play(lines), "Alcremie").protect_counter == 1
    with without("blocked_priority_is_not_cant"):
        assert foe(play(lines), "Alcremie").protect_counter == 0


def test_shed_tail_hands_the_substitute_to_the_replacement() -> None:
    lines = LEADS + [
        "|move|p2a: Alcremie|Shed Tail|p2a: Alcremie",
        "|switch|p2a: Dragonite|Dragonite, L50, M|50/100|[from] Shed Tail",
    ]
    assert Effect.SUBSTITUTE in foe(play(lines), "Dragonite").effects
    with without("shed_tail"):
        assert Effect.SUBSTITUTE not in foe(play(lines), "Dragonite").effects


def test_toxic_stage_caps_at_fifteen() -> None:
    lines = LEADS + ["|-status|p2a: Alcremie|tox"]
    for turn in range(2, 22):
        lines += ["|upkeep", f"|turn|{turn}"]
    assert foe(play(lines), "Alcremie").status_counter == 15
