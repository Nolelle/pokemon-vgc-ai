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
    _cross_slot_adjustments,
    _opp_protect_probability,
    _opp_switch_probability,
    _resolve_targets,
    _score_attack_order,
    _score_protect,
    effective_speed,
    field_effective_speed,
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


def test_field_effective_speed_applies_matching_weather_ability() -> None:
    excadrill = PokemonState("excadrill", ability="sandrush", nature="jolly")

    assert field_effective_speed(excadrill, weather="sand") == pytest.approx(
        effective_speed(excadrill) * 2.0
    )
    assert field_effective_speed(excadrill, weather="sun") == effective_speed(excadrill)


def test_field_effective_speed_applies_tailwind_and_weather_together() -> None:
    venusaur = PokemonState("venusaur", ability="chlorophyll", nature="modest")

    assert field_effective_speed(venusaur, weather="sun", tailwind=True) == pytest.approx(
        effective_speed(venusaur) * 4.0
    )


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
        opp_protect_prob=[0.0, 0.0],
        opp_switch_prob=[0.0, 0.0],
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
        our_speed=[
            effective_speed(our_states[0]),
            effective_speed(ally_state) if ally_state else 0.0,
        ],
        opp_speed=[effective_speed(opp_state), 0.0],
        threat_on_us=[_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
        opp_protect_prob=[0.0, 0.0],
        opp_switch_prob=[0.0, 0.0],
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

    ctx_with_immune_ally = _attack_ctx(
        ally_state=_garchomp(ability="levitate"), opp_state=opp_state
    )
    score_with_immune_ally, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_immune_ally, config
    )

    # Same opponent-side damage in both cases (ally is present -- and thus the spread
    # 0.75x modifier applies -- either way); the ONLY difference is whether the ally
    # itself takes (and gets penalized for) damage.
    assert score_with_ally < score_with_immune_ally

    attacker_state = ctx_with_ally.our_states[0]
    ally_result = damage_range(
        attacker_state, _garchomp(), "earthquake", FieldState(is_doubles=True, num_targets=2)
    )
    expected_penalty = (
        ally_result.expected_percent
        * config.damage_percent_weight
        * config.ally_damage_penalty_weight
    )
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


# --- joint targeting: reward useful focus fire, penalize redundant overkill --------


def _target_info(
    *,
    expected: float,
    hp: float = 100.0,
    guaranteed: bool = False,
    guarded: bool = False,
    protect_prob: float = 0.0,
    target_slot: int = 0,
) -> dict:
    return {
        "move_id": "tackle",
        "raw_damage_score": expected,
        "single_target_slot": target_slot,
        "expected_percent_by_target": {target_slot: expected},
        "current_hp_percent_by_target": {target_slot: hp},
        "guaranteed_ko_slots": [target_slot] if guaranteed else [],
        "survival_guard_slots": [target_slot] if guarded else [],
        "protect_prob_by_target": {target_slot: protect_prob},
    }


def test_cross_slot_penalizes_second_attack_after_guaranteed_ko() -> None:
    config = PolicyConfig()
    first = _target_info(expected=120.0, guaranteed=True)
    second = _target_info(expected=40.0)

    adjustment = _cross_slot_adjustments(first, second, config)

    assert adjustment == -config.redundant_ko_target_penalty


def test_cross_slot_preserves_double_target_into_full_hp_focus_sash() -> None:
    config = PolicyConfig()
    first = _target_info(expected=120.0, guaranteed=True, guarded=True)
    second = _target_info(expected=40.0, guarded=True)

    assert _cross_slot_adjustments(first, second, config) == 0.0


def test_cross_slot_rewards_combined_ko_that_neither_move_gets_alone() -> None:
    config = PolicyConfig()
    first = _target_info(expected=60.0)
    second = _target_info(expected=45.0)

    adjustment = _cross_slot_adjustments(first, second, config)

    assert adjustment == config.focus_fire_ko_bonus


# --- cross-slot stacking penalty: two single-target attacks into one Protect-capable
# --- slot forfeit the smaller attack's EV with probability protect_prob -------------


def test_cross_slot_stacking_penalty_scales_with_min_expected_and_protect_prob() -> None:
    config = PolicyConfig()
    first = _target_info(expected=60.0, protect_prob=0.4)
    second = _target_info(expected=45.0, protect_prob=0.4)

    adjustment = _cross_slot_adjustments(first, second, config)

    # Both attacks stay below the shared target's HP, so focus_fire_ko_bonus also
    # applies (60 + 45 >= 100) -- the stacking penalty is additive with it, not a
    # replacement, and scales off the SMALLER (redundant) attack's expected damage.
    expected_stack_penalty = 0.4 * min(60.0, 45.0) * config.protect_stack_penalty_weight
    assert adjustment == pytest.approx(config.focus_fire_ko_bonus - expected_stack_penalty)


def test_cross_slot_stacking_penalty_zero_when_protect_prob_is_zero() -> None:
    config = PolicyConfig()
    first = _target_info(expected=60.0, protect_prob=0.0)
    second = _target_info(expected=45.0, protect_prob=0.0)

    adjustment = _cross_slot_adjustments(first, second, config)

    # No Protect chance -> no stacking penalty -> pure focus-fire bonus, unchanged from
    # the pre-Protect-anticipation behavior.
    assert adjustment == config.focus_fire_ko_bonus


def test_cross_slot_stacking_penalty_zero_when_targets_differ() -> None:
    config = PolicyConfig()
    first = _target_info(expected=60.0, protect_prob=0.9, target_slot=0)
    second = _target_info(expected=45.0, protect_prob=0.9, target_slot=1)

    # Different single_target_slot values -> the shared-target branch never runs, so no
    # stacking penalty (and no redundant-KO/focus-fire adjustment either).
    assert _cross_slot_adjustments(first, second, config) == 0.0


def test_cross_slot_stacking_penalty_applies_alongside_redundant_ko_penalty() -> None:
    config = PolicyConfig()
    first = _target_info(expected=120.0, guaranteed=True, protect_prob=0.5)
    second = _target_info(expected=40.0, protect_prob=0.5)

    adjustment = _cross_slot_adjustments(first, second, config)

    expected_stack_penalty = 0.5 * min(120.0, 40.0) * config.protect_stack_penalty_weight
    assert adjustment == pytest.approx(-config.redundant_ko_target_penalty - expected_stack_penalty)


def test_cross_slot_stacking_penalty_falls_back_to_second_infos_protect_prob() -> None:
    config = PolicyConfig()
    first = _target_info(expected=60.0, protect_prob=0.0)
    second = _target_info(expected=45.0, protect_prob=0.3)
    # Simulate first_info missing the key entirely (defensive fallback path).
    del first["protect_prob_by_target"]

    adjustment = _cross_slot_adjustments(first, second, config)

    expected_stack_penalty = 0.3 * min(60.0, 45.0) * config.protect_stack_penalty_weight
    assert adjustment == pytest.approx(config.focus_fire_ko_bonus - expected_stack_penalty)


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


# --- _opp_protect_probability: opponent Protect anticipation -------------------------


def test_opp_protect_probability_zero_without_self_protect_move() -> None:
    config = PolicyConfig()
    prob = _opp_protect_probability(
        has_self_protect_move=False, pressure_percent=100.0, protect_counter=0, config=config
    )
    assert prob == 0.0


def test_opp_protect_probability_positive_with_self_protect_move() -> None:
    config = PolicyConfig()
    prob = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=0.0, protect_counter=0, config=config
    )
    assert prob == pytest.approx(config.opp_protect_base_prob)
    assert prob > 0.0


def test_opp_protect_probability_scales_up_with_pressure() -> None:
    config = PolicyConfig()
    low_pressure = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=20.0, protect_counter=0, config=config
    )
    high_pressure = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=80.0, protect_counter=0, config=config
    )
    assert high_pressure > low_pressure
    assert high_pressure == pytest.approx(
        config.opp_protect_base_prob + config.opp_protect_pressure_scale * 0.8
    )


def test_opp_protect_probability_capped() -> None:
    config = PolicyConfig()
    prob = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=1000.0, protect_counter=0, config=config
    )
    assert prob == pytest.approx(config.opp_protect_prob_cap)


def test_opp_protect_probability_collapsed_by_repeat_factor() -> None:
    config = PolicyConfig()
    fresh = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=100.0, protect_counter=0, config=config
    )
    repeated = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=100.0, protect_counter=1, config=config
    )
    assert repeated == pytest.approx(fresh * config.opp_protect_repeat_factor)
    assert repeated < fresh


def test_opp_protect_probability_zero_when_base_and_scale_zeroed() -> None:
    config = PolicyConfig(opp_protect_base_prob=0.0, opp_protect_pressure_scale=0.0)
    prob = _opp_protect_probability(
        has_self_protect_move=True, pressure_percent=100.0, protect_counter=0, config=config
    )
    assert prob == 0.0


# --- _opp_switch_probability: opponent switch anticipation ----------------------------


def test_opp_switch_probability_requires_pressure_floor() -> None:
    config = PolicyConfig()
    # Weak output, but pressure below the floor -- no switch incentive.
    prob = _opp_switch_probability(pressure_percent=30.0, opp_output_percent=10.0, config=config)
    assert prob == 0.0


def test_opp_switch_probability_requires_output_below_ceiling() -> None:
    config = PolicyConfig()
    # Heavy pressure, but their own output is still strong -- no switch incentive.
    prob = _opp_switch_probability(pressure_percent=90.0, opp_output_percent=50.0, config=config)
    assert prob == 0.0


def test_opp_switch_probability_triggers_when_both_conditions_met() -> None:
    config = PolicyConfig()
    prob = _opp_switch_probability(pressure_percent=90.0, opp_output_percent=10.0, config=config)
    assert prob == pytest.approx(config.opp_switch_prob)


def test_opp_switch_probability_zero_when_config_zeroed() -> None:
    config = PolicyConfig(opp_switch_prob=0.0)
    prob = _opp_switch_probability(pressure_percent=90.0, opp_output_percent=10.0, config=config)
    assert prob == 0.0


# --- _score_attack_order: opponent Protect/switch dampening ---------------------------


def test_score_attack_order_protect_prob_leaves_pure_damage_score_untouched() -> None:
    """No KO bonus at all (full-HP target, neither guaranteed nor likely) -- Protect
    probability must have NO effect on the score, confirming raw expected damage really
    is left undiscounted (a blocked hit costs the OPPONENT their turn too, so that EV
    loss is roughly tempo-neutral; only a denied KO is a real cost).
    """
    config = PolicyConfig()
    opp_state = _klefki()  # full HP: Ground-vs-Steel Earthquake deals real % but no KO
    move, single = _earthquake_single()

    ctx_no_protect = _attack_ctx(ally_state=None, opp_state=opp_state)
    score_no_protect, _, breakdown_no_protect = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_protect, config
    )
    assert breakdown_no_protect["guaranteed_ko_slots"] == []
    assert breakdown_no_protect["protect_prob_by_target"][0] == 0.0
    assert score_no_protect > 0.0

    ctx_with_protect = _attack_ctx(ally_state=None, opp_state=opp_state)
    ctx_with_protect.opp_protect_prob[0] = 0.4
    score_with_protect, _, breakdown_with_protect = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_protect, config
    )
    assert breakdown_with_protect["protect_prob_by_target"][0] == pytest.approx(0.4)
    assert score_with_protect == pytest.approx(score_no_protect)


def test_score_attack_order_protect_prob_dampens_only_ko_bonus_not_raw_damage() -> None:
    config = PolicyConfig()
    opp_state = _klefki(current_hp=1)  # 1 HP: Earthquake guarantee-KOs
    move, single = _earthquake_single()

    ctx_no_protect = _attack_ctx(ally_state=None, opp_state=opp_state)
    score_no_protect, _, breakdown_no_protect = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_protect, config
    )
    assert 0 in breakdown_no_protect["guaranteed_ko_slots"]
    assert breakdown_no_protect["protect_prob_by_target"][0] == 0.0

    ctx_with_protect = _attack_ctx(ally_state=None, opp_state=opp_state)
    ctx_with_protect.opp_protect_prob[0] = 0.4
    score_with_protect, _, breakdown_with_protect = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_protect, config
    )
    assert breakdown_with_protect["protect_prob_by_target"][0] == pytest.approx(0.4)

    # Protect probability dampens ONLY the KO-dependent bonuses (guaranteed_ko_bonus and
    # outspeed_ko_bonus here -- this slot outspeeds by default since threat_on_us is
    # unset); the raw damage term is untouched, so the score only drops by that share.
    ko_bonus_total = config.guaranteed_ko_bonus + config.outspeed_ko_bonus
    expected_drop = ko_bonus_total * 0.4
    assert score_with_protect == pytest.approx(score_no_protect - expected_drop)


def test_score_attack_order_ko_bonus_receives_both_discounts_multiplicatively() -> None:
    config = PolicyConfig()
    opp_state = _klefki(current_hp=1)  # 1 HP: Earthquake guarantee-KOs
    move, single = _earthquake_single()

    ctx_baseline = _attack_ctx(ally_state=None, opp_state=opp_state)
    score_baseline, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_baseline, config
    )

    ctx_both = _attack_ctx(ally_state=None, opp_state=opp_state)
    ctx_both.opp_protect_prob[0] = 0.4
    ctx_both.opp_switch_prob[0] = 0.5
    score_both, _, breakdown_both = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_both, config
    )
    assert breakdown_both["protect_prob_by_target"][0] == pytest.approx(0.4)
    assert breakdown_both["switch_prob_by_target"][0] == pytest.approx(0.5)

    # Both discounts apply to ko_bonus MULTIPLICATIVELY (1 - switch) * (1 - protect), not
    # additively -- e.g. 0.5 + 0.4 = 0.9 would leave only 10% of the bonus, but the
    # correct combined retention here is 0.5 * 0.6 = 0.3 (30%).
    ko_bonus_total = config.guaranteed_ko_bonus + config.outspeed_ko_bonus
    dampened_ko_bonus = ko_bonus_total * (1.0 - 0.5) * (1.0 - 0.4)
    expected_score = score_baseline - ko_bonus_total + dampened_ko_bonus
    assert score_both == pytest.approx(expected_score)


def test_score_attack_order_switch_prob_dampens_only_ko_bonus() -> None:
    config = PolicyConfig()
    opp_state = _klefki(current_hp=1)  # 1 HP: Earthquake guarantee-KOs
    move, single = _earthquake_single()

    ctx_no_switch = _attack_ctx(ally_state=None, opp_state=opp_state)
    score_no_switch, _, breakdown_no_switch = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_switch, config
    )
    assert 0 in breakdown_no_switch["guaranteed_ko_slots"]
    assert breakdown_no_switch["switch_prob_by_target"][0] == 0.0

    ctx_with_switch = _attack_ctx(ally_state=None, opp_state=opp_state)
    ctx_with_switch.opp_switch_prob[0] = 1.0
    score_with_switch, _, breakdown_with_switch = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_switch, config
    )
    assert breakdown_with_switch["switch_prob_by_target"][0] == pytest.approx(1.0)

    # A full switch probability zeroes out BOTH KO-dependent bonuses (guaranteed_ko_bonus
    # and outspeed_ko_bonus -- this slot outspeeds by default since threat_on_us is unset)
    # but leaves the raw damage term untouched.
    expected_drop = config.guaranteed_ko_bonus + config.outspeed_ko_bonus
    assert score_with_switch == pytest.approx(score_no_switch - expected_drop)


def test_score_attack_order_zero_probs_match_across_configs_differing_only_in_new_fields() -> None:
    """Regression guard: `_score_attack_order` only ever reads `ctx.opp_protect_prob`/
    `ctx.opp_switch_prob` (never the PolicyConfig fields directly), so with those ctx
    arrays at their zero default, two configs differing only in the new opponent
    Protect/switch fields must score identically -- confirming the multipliers really are
    exactly 1.0 and the new feature doesn't change ANY existing scoring path.
    """
    default_config = PolicyConfig()
    alt_config = PolicyConfig(
        opp_protect_base_prob=0.9,
        opp_protect_pressure_scale=0.9,
        opp_protect_prob_cap=0.9,
        opp_protect_repeat_factor=0.9,
        opp_switch_prob=0.9,
        opp_switch_pressure_floor=1.0,
        opp_switch_output_ceiling=99.0,
    )
    opp_state = _klefki()
    move, single = _earthquake_single()
    ctx = _attack_ctx(ally_state=None, opp_state=opp_state)

    score_default, _, breakdown_default = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx, default_config
    )
    score_alt, _, breakdown_alt = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx, alt_config
    )

    assert score_default == score_alt
    assert breakdown_default["protect_prob_by_target"][0] == 0.0
    assert breakdown_default["switch_prob_by_target"][0] == 0.0
    assert breakdown_alt["protect_prob_by_target"][0] == 0.0
    assert breakdown_alt["switch_prob_by_target"][0] == 0.0
