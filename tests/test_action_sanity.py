"""Wasted-action guard (vgc.action_sanity): three blunders, plus no-false-positive cases."""

from __future__ import annotations

from types import SimpleNamespace

from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from vgc.action_sanity import wasted_action_reasons


def _mon(species: str, ability: str | None = None, item: str | None = None) -> Pokemon:
    mon = Pokemon(gen=9, species=species)
    mon._ability = ability  # revealed ability; None = hidden
    mon._item = item
    return mon


def _move(move_id: str, target: int = 0) -> SingleBattleOrder:
    return SingleBattleOrder(Move(move_id, gen=9), move_target=target)


def _battle(ours, theirs, fields=(), available=None):
    return SimpleNamespace(
        active_pokemon=list(ours),
        opponent_active_pokemon=list(theirs),
        fields={f: 1 for f in fields},
        available_moves=available or [[1, 2, 3, 4], [1, 2, 3, 4]],
        last_request={},
    )


def _order(a, b) -> DoubleBattleOrder:
    return DoubleBattleOrder(first_order=a, second_order=b)


def test_fake_out_into_hidden_armor_tail_is_flagged_only_when_blocking() -> None:
    ours = [_mon("sneasler"), _mon("salamence")]
    # Farigiraf's ability is hidden; usage says Armor Tail >= 50% (or it is unknowable).
    battle = _battle(ours, [_mon("farigiraf"), _mon("incineroar")])
    # Usage share of Armor Tail on Farigiraf is ~98%, so the hidden ability counts, and it
    # also covers its Incineroar partner.
    for target in (1, 2):
        reasons = wasted_action_reasons(battle, _order(_move("fakeout", target), _move("hypervoice")))
        assert reasons and reasons[0].startswith("priority_blocked:fakeout")
    # Revealed Armor Tail certainly blocks; revealed Cud Chew does not.
    armored = _battle(ours, [_mon("farigiraf", ability="armortail"), _mon("incineroar")])
    assert wasted_action_reasons(armored, _order(_move("fakeout", 1), _move("hypervoice")))
    chewy = _battle(ours, [_mon("farigiraf", ability="cudchew"), _mon("incineroar")])
    assert not wasted_action_reasons(chewy, _order(_move("fakeout", 1), _move("hypervoice")))
    # Non-priority moves and Mold Breaker attackers are fine.
    assert not wasted_action_reasons(armored, _order(_move("closecombat", 1), _move("hypervoice")))


def test_fake_out_into_ordinary_target_not_flagged() -> None:
    battle = _battle(
        [_mon("sneasler"), _mon("salamence")], [_mon("incineroar"), _mon("rillaboom")]
    )
    assert not wasted_action_reasons(battle, _order(_move("fakeout", 1), _move("hypervoice")))


def test_psychic_terrain_blocks_priority_on_grounded_only() -> None:
    ours = [_mon("sneasler"), _mon("salamence")]
    grounded = _battle(ours, [_mon("incineroar"), _mon("rillaboom")], [Field.PSYCHIC_TERRAIN])
    assert wasted_action_reasons(grounded, _order(_move("fakeout", 1), _move("hypervoice")))
    flyer = _battle(ours, [_mon("salamence"), _mon("rillaboom")], [Field.PSYCHIC_TERRAIN])
    assert not wasted_action_reasons(flyer, _order(_move("fakeout", 1), _move("hypervoice")))


def test_helping_hand_wasted_unless_partner_attacks() -> None:
    ours = [_mon("indeedeef"), _mon("gardevoir")]
    battle = _battle(ours, [_mon("incineroar"), _mon("rillaboom")])
    assert wasted_action_reasons(battle, _order(_move("helpinghand"), _move("protect")))
    switch = SingleBattleOrder(_mon("basculegion"))
    assert wasted_action_reasons(battle, _order(_move("helpinghand"), switch))
    assert not wasted_action_reasons(battle, _order(_move("helpinghand"), _move("moonblast", 1)))
    absent = _battle([ours[0], None], [_mon("incineroar"), _mon("rillaboom")])
    assert wasted_action_reasons(absent, _order(_move("helpinghand"), None))


def test_choice_scarf_status_move_flagged_but_attacks_and_locked_are_not() -> None:
    ours = [_mon("indeedee", item="choicescarf"), _mon("sneasler")]
    battle = _battle(ours, [_mon("incineroar"), _mon("rillaboom")])
    assert wasted_action_reasons(battle, _order(_move("protect"), _move("direclaw", 1)))
    assert not wasted_action_reasons(battle, _order(_move("psychic", 1), _move("direclaw", 1)))
    assert not wasted_action_reasons(battle, _order(_move("trick", 1), _move("direclaw", 1)))
    locked = _battle(ours, [_mon("incineroar"), _mon("rillaboom")], available=[[1], [1, 2]])
    assert not wasted_action_reasons(locked, _order(_move("protect"), _move("direclaw", 1)))
    plain = [_mon("indeedee", item="lifeorb"), _mon("sneasler")]
    battle = _battle(plain, [_mon("incineroar"), _mon("rillaboom")])
    assert not wasted_action_reasons(battle, _order(_move("protect"), _move("direclaw", 1)))


def test_own_surge_switch_in_changes_terrain_before_moves() -> None:
    ours = [_mon("indeedeef"), _mon("sneasler")]
    theirs = [_mon("incineroar"), _mon("rillaboom")]
    psychic = _battle(ours, theirs, [Field.PSYCHIC_TERRAIN])
    grassy_in = SingleBattleOrder(_mon("rillaboom", ability="grassysurge"))
    # Grassy Surge replaces Psychic Terrain before Fake Out: legal.
    assert not wasted_action_reasons(psychic, _order(grassy_in, _move("fakeout", 1)))
    assert wasted_action_reasons(psychic, _order(_move("protect"), _move("fakeout", 1)))
    # And Psychic Surge arriving on a clear field makes it illegal.
    clear = _battle(ours, theirs)
    psy_in = SingleBattleOrder(_mon("indeedee", ability="psychicsurge"))
    assert wasted_action_reasons(clear, _order(psy_in, _move("fakeout", 1)))


def test_costs_are_strategic_and_far_below_a_game_result() -> None:
    from vgc.action_sanity import wasted_action_cost
    from vgc.models import PolicyConfig

    config = PolicyConfig()
    ours = [_mon("sneasler"), _mon("salamence")]
    battle = _battle(ours, [_mon("farigiraf", ability="armortail"), _mon("incineroar")])
    cost, reasons = wasted_action_cost(battle, _order(_move("fakeout", 1), _move("hypervoice")), config)
    assert reasons and 0 < cost == config.wasted_action_penalty < 1_000  # win is +10,000
    scarf = [_mon("indeedee", item="choicescarf"), _mon("sneasler")]
    choice_cost, _ = wasted_action_cost(
        _battle(scarf, [_mon("incineroar"), _mon("rillaboom")]),
        _order(_move("protect"), _move("direclaw", 1)),
        config,
    )
    assert choice_cost == config.choice_lock_status_penalty < config.wasted_action_penalty


def test_trick_onto_itemless_unburden_ally_is_costed() -> None:
    ours = [_mon("indeedee", item="choicescarf"), _mon("sneasler", ability="unburden")]
    battle = _battle(ours, [_mon("incineroar"), _mon("rillaboom")])
    reasons = wasted_action_reasons(battle, _order(_move("trick", -2), _move("direclaw", 1)))
    assert reasons and reasons[0].startswith("trick_to_ally")
    ours[1]._item = "lifeorb"
    assert not wasted_action_reasons(battle, _order(_move("trick", -2), _move("direclaw", 1)))
