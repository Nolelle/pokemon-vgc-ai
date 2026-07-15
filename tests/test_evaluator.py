"""Unit tests for `vgc.evaluator`'s building blocks, using directly-constructed
`PokemonState`s (no live battle/server needed -- see tests/integration/ for the
end-to-end path). Mirrors tests/test_damage.py's "hand-computed expected values"
style.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move

from vgc.damage import FieldState, PokemonState, damage_range
from vgc.data import load_moves
from vgc.evaluator import (
    _Context,
    _ThreatInfo,
    _resolve_targets,
    _score_attack_order,
    _score_protect,
    effective_speed,
    guaranteed_ko,
    likely_ko,
    mega_species_id,
    mega_evolved_state,
    resolves_before,
)
from vgc.models import PolicyConfig


def _garchomp(**kwargs) -> PokemonState:
    defaults = dict(sp_spread={"hp": 2, "atk": 32, "spe": 32}, nature="jolly")
    defaults.update(kwargs)
    return PokemonState("garchomp", **defaults)


def _klefki(**kwargs) -> PokemonState:
    defaults = dict(sp_spread={"hp": 32, "def": 32, "spd": 2}, nature="bold")
    defaults.update(kwargs)
    return PokemonState("klefki", **defaults)


# --- effective_speed ----------------------------------------------------------------


def test_effective_speed_choice_scarf_multiplies_by_1_5() -> None:
    plain = _garchomp()
    scarfed = _garchomp(item="choicescarf")
    assert effective_speed(scarfed) == pytest.approx(effective_speed(plain) * 1.5)


def test_effective_speed_paralysis_halves_speed() -> None:
    plain = _garchomp()
    paralyzed = _garchomp(status="par")
    assert effective_speed(paralyzed) == pytest.approx(effective_speed(plain) * 0.5)


def test_effective_speed_positive_boost_stage_increases_speed() -> None:
    plain = _garchomp()
    boosted = _garchomp(boosts={"spe": 2})
    assert effective_speed(boosted) > effective_speed(plain)


# --- resolves_before (priority brackets + Trick Room inversion) ---------------------


def test_resolves_before_priority_beats_raw_speed() -> None:
    # Fake Out (priority 3) vs a much faster but priority-0 attacker.
    assert resolves_before(3, 50.0, 0, 400.0, trick_room=False) is True
    assert resolves_before(0, 400.0, 3, 50.0, trick_room=False) is False


def test_resolves_before_same_priority_faster_first_without_trick_room() -> None:
    assert resolves_before(0, 150.0, 0, 100.0, trick_room=False) is True
    assert resolves_before(0, 100.0, 0, 150.0, trick_room=False) is False


def test_resolves_before_trick_room_inverts_speed_order() -> None:
    # Under Trick Room the SLOWER Pokemon acts first within the same priority bracket.
    assert resolves_before(0, 100.0, 0, 150.0, trick_room=True) is True
    assert resolves_before(0, 150.0, 0, 100.0, trick_room=True) is False


def test_resolves_before_trick_room_does_not_touch_priority() -> None:
    # Trick Room never reorders priority brackets, only same-bracket speed ties.
    assert resolves_before(3, 50.0, 0, 400.0, trick_room=True) is True


# --- guaranteed_ko / likely_ko --------------------------------------------------------


def test_guaranteed_ko_true_when_min_roll_clears_hp() -> None:
    attacker = _garchomp(item="lifeorb")
    defender = _klefki(current_hp=1)
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    assert guaranteed_ko(result, defender.hp_or_max()) is True


def test_guaranteed_ko_false_and_likely_ko_true_for_borderline_hp() -> None:
    attacker = _garchomp()
    defender = _klefki()
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    # Pick an HP value strictly between the min and expected roll so min doesn't KO but
    # the expected (mean) roll does.
    borderline_hp = int((result.min_damage + result.expected_damage) / 2) or 1
    assert result.min_damage < borderline_hp <= result.expected_damage
    assert guaranteed_ko(result, borderline_hp) is False
    assert likely_ko(result, borderline_hp) is True


def test_guaranteed_and_likely_ko_false_when_far_from_ko() -> None:
    attacker = _garchomp()
    defender = _klefki()
    result = damage_range(attacker, defender, "earthquake", FieldState(is_doubles=False))
    huge_hp = result.max_damage * 10
    assert guaranteed_ko(result, huge_hp) is False
    assert likely_ko(result, huge_hp) is False


# --- mega_species_id -------------------------------------------------------------------


def test_mega_species_id_resolves_from_base_species_and_item() -> None:
    assert mega_species_id("charizard", "charizarditey") == "charizardmegay"
    assert mega_species_id("charizard", "charizarditex") == "charizardmegax"


def test_mega_species_id_none_for_wrong_item_or_no_item() -> None:
    assert mega_species_id("charizard", "leftovers") is None
    assert mega_species_id("charizard", None) is None


def test_mega_species_id_none_for_species_with_no_mega() -> None:
    assert mega_species_id("klefki", "leftovers") is None


def test_mega_evolved_state_uses_mega_stats_and_ability() -> None:
    state = PokemonState(
        species_id="charizard",
        item="charizarditey",
        ability="blaze",
        sp_spread={"spa": 32, "spe": 24, "hp": 10},
        nature="modest",
    )

    mega = mega_evolved_state(state)

    assert mega.species_id == "charizardmegay"
    assert mega.ability == "drought"
    assert mega.stats()["spa"] > state.stats()["spa"]


# --- _resolve_targets: allAdjacent (hits ally) vs allAdjacentFoes (foes only) --------


@dataclass
class _FakeActive:
    fainted: bool = False


def _two_slot_ctx(our_alive=(True, True), opp_alive=(True, True)) -> _Context:
    our_pokemon = [_FakeActive(fainted=not alive) for alive in our_alive]
    opp_pokemon = [_FakeActive(fainted=not alive) for alive in opp_alive]
    return _Context(
        battle=SimpleNamespace(side_conditions=[]),
        trick_room=False,
        weather=None,
        terrain=None,
        our_side_screens=frozenset(),
        opp_side_screens=frozenset(),
        our_pokemon=our_pokemon,
        opp_pokemon=opp_pokemon,
        our_states=[None, None],
        opp_states=[None, None],
        our_speed=[0.0, 0.0],
        opp_speed=[0.0, 0.0],
        threat_on_us=[_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
    )


def test_resolve_targets_all_adjacent_includes_live_ally() -> None:
    ctx = _two_slot_ctx()
    move_data = load_moves()["earthquake"]
    assert move_data["target"] == "allAdjacent"
    targets = set(_resolve_targets(move_data, actor_slot=0, move_target=0, ctx=ctx))
    assert targets == {(0, False), (1, False), (1, True)}


def test_resolve_targets_all_adjacent_foes_excludes_ally() -> None:
    ctx = _two_slot_ctx()
    move_data = load_moves()["muddywater"]
    assert move_data["target"] == "allAdjacentFoes"
    targets = set(_resolve_targets(move_data, actor_slot=0, move_target=0, ctx=ctx))
    assert targets == {(0, False), (1, False)}


def test_resolve_targets_all_adjacent_skips_fainted_ally() -> None:
    ctx = _two_slot_ctx(our_alive=(True, False))
    move_data = load_moves()["earthquake"]
    targets = set(_resolve_targets(move_data, actor_slot=0, move_target=0, ctx=ctx))
    assert targets == {(0, False), (1, False)}


# --- _score_attack_order: Earthquake's ally-damage penalty ---------------------------


def _attack_ctx(*, ally_state: PokemonState | None, opp_state: PokemonState) -> _Context:
    our_states = [_garchomp(), ally_state]
    opp_states = [opp_state, None]
    our_pokemon = [
        SimpleNamespace(first_turn=False, fainted=False),
        SimpleNamespace(first_turn=False, fainted=False) if ally_state else None,
    ]
    opp_pokemon = [SimpleNamespace(ability=opp_state.ability, fainted=False), None]
    return _Context(
        battle=SimpleNamespace(side_conditions=[]),
        trick_room=False,
        weather=None,
        terrain=None,
        our_side_screens=frozenset(),
        opp_side_screens=frozenset(),
        our_pokemon=our_pokemon,
        opp_pokemon=opp_pokemon,
        our_states=our_states,
        opp_states=opp_states,
        our_speed=[effective_speed(our_states[0]), effective_speed(ally_state) if ally_state else 0.0],
        opp_speed=[effective_speed(opp_state), 0.0],
        threat_on_us=[_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
    )


def _earthquake_single(move_target: int = 0) -> tuple[Move, SimpleNamespace]:
    move = Move("earthquake", gen=9)
    single = SimpleNamespace(order=move, mega=False, move_target=move_target)
    return move, single


def test_earthquake_penalizes_score_when_ally_takes_damage() -> None:
    config = PolicyConfig()
    opp_state = _klefki()

    move, single = _earthquake_single()
    ctx_with_ally = _attack_ctx(ally_state=_garchomp(), opp_state=opp_state)
    score_with_ally, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_ally, config
    )

    ctx_with_immune_ally = _attack_ctx(ally_state=_garchomp(ability="levitate"), opp_state=opp_state)
    score_with_immune_ally, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_immune_ally, config
    )

    # Same opponent-side damage in both cases (ally is present -- and thus the spread
    # 0.75x modifier applies -- either way); the ONLY difference is whether the ally
    # itself takes (and gets penalized for) damage.
    assert score_with_ally < score_with_immune_ally

    attacker_state = ctx_with_ally.our_states[0]
    ally_result = damage_range(attacker_state, _garchomp(), "earthquake", FieldState(is_doubles=True, num_targets=2))
    expected_penalty = ally_result.expected_percent * config.damage_percent_weight * config.ally_damage_penalty_weight
    assert (score_with_immune_ally - score_with_ally) == pytest.approx(expected_penalty, rel=1e-6)


def test_single_target_move_has_no_ally_penalty() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move = Move("dragonclaw", gen=9)
    move_data = load_moves()["dragonclaw"]
    assert move_data["target"] == "normal"
    single = SimpleNamespace(order=move, mega=False, move_target=1)  # opponent slot 1

    ctx = _attack_ctx(ally_state=_garchomp(), opp_state=opp_state)
    score, raw, _ = _score_attack_order(move, move_data, single, 0, ctx, config)
    # Pure single-target damage score: no ally involved at all, so score == raw damage
    # contribution (no KO bonus expected against a bulky Klefki with a neutral hit).
    assert score == pytest.approx(raw, rel=1e-6)


# --- _score_protect: threat-scaled value, repeat-use and low-threat penalties -------


def test_protect_value_scales_with_threat_percent() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=80.0, move_id="earthquake", priority=0)
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=0)
    score, info = _score_protect(0, ctx, config)
    assert score == pytest.approx(80.0 * config.protect_threat_weight)
    assert info["protect_counter"] == 0


def test_protect_repeat_use_is_penalized() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=80.0, move_id="earthquake", priority=0)
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=1)
    score, _ = _score_protect(0, ctx, config)
    expected = 80.0 * config.protect_threat_weight - config.protect_repeat_penalty
    assert score == pytest.approx(expected)


def test_protect_with_no_real_threat_is_penalized_and_negative() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=5.0, move_id="tackle", priority=0)
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=0)
    score, _ = _score_protect(0, ctx, config)
    expected = 5.0 * config.protect_threat_weight - config.protect_low_threat_penalty
    assert score == pytest.approx(expected)
    assert score < 0


def test_protect_beats_low_threat_penalty_when_threat_is_high() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=90.0, move_id="earthquake", priority=0)
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=0)
    score, _ = _score_protect(0, ctx, config)
    assert score > 0
