from __future__ import annotations

from types import SimpleNamespace

from vgc.mechanics_state import BOOST_IDS, snapshot_battle


def _move(move_id: str, *, pp: int, max_pp: int, disabled: bool = False):
    return SimpleNamespace(
        id=move_id,
        current_pp=pp,
        max_pp=max_pp,
        entry={"disabled": disabled, "disabledSource": "Taunt" if disabled else None},
        is_last_used=False,
    )


def _pokemon(species: str, *, opponent: bool = False):
    return SimpleNamespace(
        species=species,
        base_species=species,
        name=species,
        level=50,
        gender="F",
        types=("Fairy",),
        base_types=("Fairy",),
        current_hp=73,
        max_hp=150,
        fainted=False,
        active=True,
        revealed=not opponent,
        selected_in_teampreview=True,
        stats={"atk": 80, "def": 100, "spa": 130, "spd": 110, "spe": 90},
        evs=(0, 0, 22, 20, 9, 15),
        ivs=(31, 31, 31, 31, 31, 31),
        nature="Modest",
        boosts={"spa": 2, "accuracy": -1, "evasion": 1},
        status="par",
        status_counter=3,
        effects={"substitute": 2, "confusion": 1},
        item=None if opponent else "Fairy Feather",
        ability=None if opponent else "Pixilate",
        base_ability=None if opponent else "Pixilate",
        temporary_ability=None,
        forme_change_ability=None,
        base_moves=("Hyper Voice", "Protect"),
        moves={
            "hypervoice": _move("hypervoice", pp=7, max_pp=16),
            "protect": _move("protect", pp=8, max_pp=8, disabled=True),
        },
        last_move=SimpleNamespace(id="Hyper Voice"),
        mimic_move=None,
        preparing_move=None,
        preparing_target=None,
        must_recharge=True,
        protect_counter=2,
        first_turn=False,
        transformed=False,
        is_terastallized=False,
        tera_type=None,
        weight=23.5,
    )


def test_snapshot_preserves_all_observable_mechanics_state() -> None:
    ours = _pokemon("Sylveon")
    opponent = _pokemon("Gengar", opponent=True)
    battle = SimpleNamespace(
        format="gen9championsvgc2026regmb",
        gen=9,
        turn=8,
        max_team_size=6,
        team_size=4,
        teampreview=False,
        commanding=False,
        reviving=False,
        wait=False,
        finished=False,
        won=False,
        lost=False,
        fields={"trickroom": 3, "gravity": 2},
        weather={"sunnyday": 4},
        team={"p1: Sylveon": ours},
        opponent_team={"p2: Gengar": opponent},
        active_pokemon=[ours, None],
        opponent_active_pokemon=[opponent, None],
        side_conditions={"tailwind": 2, "reflect": 4},
        opponent_side_conditions={"safeguard": 3},
        force_switch=[False, True],
        opponent_force_switch=[False, False],
        trapped=[True, False],
        opponent_trapped=[False, False],
        maybe_trapped=[True, False],
        opponent_maybe_trapped=[False, True],
        can_mega_evolve=[False, False],
        opponent_can_mega_evolve=[True, False],
        used_mega_evolve=True,
        opponent_used_mega_evolve=False,
        can_dynamax=False,
        opponent_can_dynamax=False,
        used_dynamax=False,
        opponent_used_dynamax=False,
        can_tera=False,
        opponent_can_tera=False,
        used_tera=False,
        opponent_used_tera=False,
        can_z_move=False,
        opponent_can_z_move=False,
        used_z_move=False,
        opponent_used_z_move=False,
        available_moves=[[ours.moves["hypervoice"]], []],
        available_switches=[[opponent], []],
        valid_orders=("one", "two"),
        teampreview_team=[ours],
        teampreview_opponent_team=[opponent],
    )

    state = snapshot_battle(battle)
    mon = state.our_side.pokemon[0]

    assert state.turn == 8
    assert {effect.id: effect.turns for effect in state.fields} == {"gravity": 2, "trickroom": 3}
    assert {effect.id: effect.turns for effect in state.our_side.side_conditions} == {
        "reflect": 4,
        "tailwind": 2,
    }
    assert dict(mon.boosts) == dict.fromkeys(BOOST_IDS, 0) | {
        "spa": 2,
        "accuracy": -1,
        "evasion": 1,
    }
    assert {effect.id for effect in mon.effects} == {"confusion", "substitute"}
    assert mon.status == "par"
    assert mon.status_counter == 3
    assert mon.must_recharge
    assert mon.protect_counter == 2
    assert mon.moves[0].id == "hypervoice"
    assert mon.moves[1].disabled
    assert mon.moves[1].disabled_reason == "Taunt"
    assert dict(mon.individual_values or ()) == dict.fromkeys(
        ("hp", "atk", "def", "spa", "spd", "spe"), 31
    )
    assert mon.base_move_ids == ("hypervoice", "protect")
    assert state.our_side.force_switch == (False, True)
    assert state.our_side.trapped == (True, False)
    assert state.available_moves == (("hypervoice",), ())
    assert state.available_switches == (("gengar",), ())
    assert state.valid_order_count == 2
    assert state.our_preview_species == ("sylveon",)
    assert state.opponent_preview_species == ("gengar",)


def test_opponent_unknown_item_ability_and_moves_stay_unknown() -> None:
    opponent = _pokemon("Gengar", opponent=True)
    opponent.moves = {}
    battle = SimpleNamespace(
        opponent_team={"p2: Gengar": opponent},
        opponent_active_pokemon=[opponent, None],
    )

    mon = snapshot_battle(battle).opponent_side.pokemon[0]

    assert mon.item_id is None and not mon.item_known
    assert mon.ability_id is None and not mon.ability_known
    assert mon.moves == ()
