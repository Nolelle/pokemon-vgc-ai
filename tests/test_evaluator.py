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
    _score_status_mega,
    _score_status_move,
    _score_switch,
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


def test_mega_is_delayed_when_it_does_not_change_the_current_turn() -> None:
    config = PolicyConfig()
    ctx = _attack_ctx(
        ally_state=None,
        opp_state=_klefki(),
        attacker_state=_garchomp(item="garchompite", ability="roughskin"),
    )
    score, info = _score_status_mega(0, ctx, config)
    assert info["mega_material"] is False
    assert score == -config.mega_unnecessary_penalty


def test_weather_setting_mega_is_material_even_beside_status_move() -> None:
    config = PolicyConfig()
    charizard = PokemonState("charizard", item="charizarditey", ability="blaze")
    ctx = _attack_ctx(
        ally_state=None,
        opp_state=_klefki(),
        attacker_state=charizard,
    )
    score, info = _score_status_mega(0, ctx, config)
    assert info["mega_weather_change"] is True
    assert info["mega_material"] is True
    assert score == 0.0


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
        priors={},
        gameplan=None,
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


def _attack_ctx(
    *,
    ally_state: PokemonState | None,
    opp_state: PokemonState,
    gameplan=None,
    weather: str | None = None,
    threat_on_us: list | None = None,
    attacker_state: PokemonState | None = None,
) -> _Context:
    our_states = [attacker_state if attacker_state is not None else _garchomp(), ally_state]
    opp_states = [opp_state, None]
    our_pokemon = [
        SimpleNamespace(first_turn=False, fainted=False, species=our_states[0].species_id),
        SimpleNamespace(first_turn=False, fainted=False, species=ally_state.species_id)
        if ally_state
        else None,
    ]
    opp_pokemon = [
        SimpleNamespace(ability=opp_state.ability, fainted=False, species=opp_state.species_id),
        None,
    ]
    return _Context(
        battle=SimpleNamespace(side_conditions=[]),
        trick_room=False,
        weather=weather,
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
        threat_on_us=threat_on_us if threat_on_us is not None else [_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
        opp_protect_prob=[0.0, 0.0],
        opp_switch_prob=[0.0, 0.0],
        priors={},
        gameplan=gameplan,
    )


def _earthquake_single(move_target: int = 0) -> tuple[Move, SimpleNamespace]:
    move = Move("earthquake", gen=9)
    single = SimpleNamespace(order=move, mega=False, move_target=move_target)
    return move, single


def _single_for(
    move_id: str, move_target: int = 0, mega: bool = False
) -> tuple[Move, SimpleNamespace]:
    move = Move(move_id, gen=9)
    single = SimpleNamespace(order=move, mega=mega, move_target=move_target)
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


def test_cross_slot_different_targets_earn_dual_pressure_bonus() -> None:
    config = PolicyConfig()
    first = _target_info(expected=60.0, protect_prob=0.9, target_slot=0)
    second = _target_info(expected=45.0, protect_prob=0.9, target_slot=1)

    # Threatening both opposing slots makes a single Protect insufficient.
    assert _cross_slot_adjustments(first, second, config) == config.dual_target_pressure_bonus


def test_fake_out_plus_setup_has_explicit_action_economy_synergy() -> None:
    config = PolicyConfig()
    first = {"move_id": "fakeout"}
    second = {"move_id": "swordsdance", "utility_kind": "setup"}
    assert _cross_slot_adjustments(first, second, config) == config.fake_out_setup_bonus


def test_redirection_plus_setup_has_explicit_pressure_pairing_synergy() -> None:
    config = PolicyConfig()
    first = {"move_id": "ragepowder", "utility_kind": "redirection"}
    second = {"move_id": "trickroom", "utility_kind": "setup"}
    assert _cross_slot_adjustments(first, second, config) == config.redirection_setup_bonus


def test_protect_plus_partner_ko_of_threat_has_action_denial_synergy() -> None:
    config = PolicyConfig()
    first = {
        "move_id": "protect",
        "utility_kind": "protect",
        "actor_slot": 0,
        "threat_source_slot": 1,
    }
    second = {"move_id": "earthquake", "ko_slots": [1]}
    assert _cross_slot_adjustments(first, second, config) == config.protect_partner_cleanup_bonus


def test_tailwind_that_immediately_enables_partner_ko_gets_bonus() -> None:
    config = PolicyConfig()
    first = {"move_id": "tailwind", "utility_kind": "speed_control"}
    second = {
        "move_id": "dragonclaw",
        "ko_slots": [0],
        "actor_speed": 100.0,
        "target_speeds": {0: 150.0},
    }
    assert _cross_slot_adjustments(first, second, config) == config.speed_control_immediate_ko_bonus


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


def test_protect_repeat_use_decays_geometrically() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=80.0, move_id="earthquake", priority=0)
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=1)
    score, info = _score_protect(0, ctx, config)
    expected = 80.0 * config.protect_threat_weight * config.protect_success_decay
    assert score == pytest.approx(expected)
    assert info["success_prob"] == pytest.approx(config.protect_success_decay)


def test_protect_score_ratio_across_counters_is_1_third_ninth() -> None:
    """Real Gen 9 mechanics: each consecutive Protect-family use divides success odds
    by ~3 (counter 0 = 100%, 1 = ~33%, 2 = ~11%) -- the threat-avoidance term (not the
    low-threat penalty, which is counter-independent) should decay in exactly that
    1 : 1/3 : 1/9 ratio.
    """
    config = PolicyConfig()
    # High enough threat that protect_low_threat_penalty never applies -- isolates the
    # geometric decay term from that separate, counter-independent penalty.
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=90.0, move_id="earthquake", priority=0)

    scores = []
    for counter in (0, 1, 2):
        ctx.our_pokemon[0] = SimpleNamespace(protect_counter=counter)
        score, _ = _score_protect(0, ctx, config)
        scores.append(score)

    assert scores[0] == pytest.approx(90.0 * config.protect_threat_weight)
    assert scores[1] == pytest.approx(scores[0] / 3.0)
    assert scores[2] == pytest.approx(scores[0] / 9.0)


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


def test_protect_values_information_when_opponent_set_is_unknown() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=50.0, move_id="earthquake", priority=0)
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=0)
    ctx.opp_uncertainty = [3, 2]
    score, info = _score_protect(0, ctx, config)
    baseline = 50.0 * config.protect_threat_weight
    assert score > baseline
    assert info["information_value"] == 4 * config.protect_information_per_unknown


def test_combined_double_target_threat_can_justify_protect() -> None:
    config = PolicyConfig()
    ctx = _two_slot_ctx()
    ctx.threat_on_us[0] = _ThreatInfo(percent=40.0, move_id="tackle", priority=0)
    ctx.double_target_threat[0] = 200.0
    ctx.our_pokemon[0] = SimpleNamespace(protect_counter=0)
    _, info = _score_protect(0, ctx, config)
    assert info["combined_threat_percent"] == pytest.approx(
        200.0 * config.double_target_threat_weight
    )


def test_taunt_explicitly_values_denial_of_opponent_setup() -> None:
    config = PolicyConfig()
    ctx = _attack_ctx(ally_state=None, opp_state=_klefki())
    ctx.opp_pokemon[0].moves = {"trickroom": None, "protect": None, "psychic": None}
    ctx.opp_control_threat[0] = 40.0
    move = Move("taunt", gen=9)
    single = SimpleNamespace(order=move, move_target=1, mega=False)
    score, info = _score_status_move("taunt", load_moves()["taunt"], single, 0, ctx, config)
    assert score > config.taunt_base_value
    assert info["utility_kind"] == "action_denial"


def test_setup_is_penalized_when_user_is_in_immediate_danger() -> None:
    config = PolicyConfig()
    ctx = _attack_ctx(ally_state=None, opp_state=_klefki())
    ctx.threat_on_us[0] = _ThreatInfo(percent=100.0, move_id="psychic")
    move = Move("swordsdance", gen=9)
    single = SimpleNamespace(order=move, move_target=0, mega=False)
    score, info = _score_status_move(
        "swordsdance", load_moves()["swordsdance"], single, 0, ctx, config
    )
    assert score == config.setup_base_value - config.unsafe_setup_penalty
    assert info["danger"] == 100.0


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


# --- charge/recharge move discounts ---------------------------------------------------


def test_charge_move_discount_applies_without_sun() -> None:
    config = PolicyConfig()
    full_strength = PolicyConfig(charge_move_discount=1.0)
    opp_state = _klefki()
    move, single = _single_for("solarbeam")

    ctx = _attack_ctx(ally_state=None, opp_state=opp_state, weather=None)
    score, raw, _ = _score_attack_order(move, load_moves()["solarbeam"], single, 0, ctx, config)
    score_full, raw_full, _ = _score_attack_order(
        move, load_moves()["solarbeam"], single, 0, ctx, full_strength
    )

    assert score_full > 0.0  # sanity: solarbeam actually does something to Klefki
    assert score == pytest.approx(score_full * config.charge_move_discount)
    assert raw == pytest.approx(raw_full * config.charge_move_discount)


def test_charge_move_discount_applies_in_rain_too() -> None:
    """Only SUN skips Solar Beam's charge turn -- any other weather (including rain,
    which is otherwise a real mechanical modifier on plenty of moves) still gets the
    full discount.
    """
    config = PolicyConfig()
    full_strength = PolicyConfig(charge_move_discount=1.0)
    opp_state = _klefki()
    move, single = _single_for("solarbeam")

    ctx = _attack_ctx(ally_state=None, opp_state=opp_state, weather="rain")
    score, _, _ = _score_attack_order(move, load_moves()["solarbeam"], single, 0, ctx, config)
    score_full, _, _ = _score_attack_order(
        move, load_moves()["solarbeam"], single, 0, ctx, full_strength
    )

    assert score == pytest.approx(score_full * config.charge_move_discount)


def test_charge_move_discount_does_not_apply_to_solarbeam_in_sun() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move, single = _single_for("solarbeam")

    ctx = _attack_ctx(ally_state=None, opp_state=opp_state, weather="sun")
    score, raw, _ = _score_attack_order(move, load_moves()["solarbeam"], single, 0, ctx, config)

    full_strength = PolicyConfig(charge_move_discount=1.0)
    score_full, raw_full, _ = _score_attack_order(
        move, load_moves()["solarbeam"], single, 0, ctx, full_strength
    )

    # Sun skips the charge turn for Solar Beam -- no discount should apply at all.
    assert score == pytest.approx(score_full)
    assert raw == pytest.approx(raw_full)


def test_recharge_move_discount_applies_regardless_of_weather() -> None:
    config = PolicyConfig()
    full_strength = PolicyConfig(recharge_move_discount=1.0)
    opp_state = _klefki()
    move, single = _single_for("hyperbeam")

    for weather in (None, "sun", "rain"):
        ctx = _attack_ctx(ally_state=None, opp_state=opp_state, weather=weather)
        score, raw, _ = _score_attack_order(move, load_moves()["hyperbeam"], single, 0, ctx, config)
        score_full, raw_full, _ = _score_attack_order(
            move, load_moves()["hyperbeam"], single, 0, ctx, full_strength
        )
        assert score == pytest.approx(score_full * config.recharge_move_discount)
        assert raw == pytest.approx(raw_full * config.recharge_move_discount)


# --- game-plan layer consumption: win_con_preservation / plan_breaker / collapsed ------
# --- matchup -- vgc.gameplan's own derivation is tested in tests/test_gameplan.py; ------
# --- these test ONLY how vgc.evaluator READS a (hand-built) GamePlan. -----------------


def _minimal_gameplan(
    our_species: tuple[str, ...] = (),
    primary_win_con_idx: int | None = None,
    plan_breakers: frozenset = frozenset(),
    matrix: dict | None = None,
):
    from vgc.gameplan import GamePlan

    return GamePlan(
        our_species=our_species,
        opp_species=(),
        win_con_scores=tuple(1.0 for _ in our_species),
        their_threat_scores=(),
        primary_win_con_idx=primary_win_con_idx,
        primary_threat_idx=None,
        answers={},
        plan_breakers=plan_breakers,
        matrix=matrix or {},
    )


def test_win_con_preservation_penalizes_staying_in_under_lethal_threat() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move, single = _earthquake_single()
    # 100%+ threat, and earthquake here does NOT guarantee/likely-KO Klefki (verified:
    # ~88% expected, min roll 134 < Klefki's 164 max HP) -- so
    # resolves_threat_before_it_lands stays False and the penalty should fire.
    threat = [_ThreatInfo(percent=100.0, move_id="tackle", priority=0), _ThreatInfo()]

    ctx_no_plan = _attack_ctx(
        ally_state=None, opp_state=opp_state, threat_on_us=threat, gameplan=None
    )
    score_no_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_plan, config
    )

    plan = _minimal_gameplan(our_species=("garchomp",), primary_win_con_idx=0)
    ctx_with_plan = _attack_ctx(
        ally_state=None, opp_state=opp_state, threat_on_us=threat, gameplan=plan
    )
    score_with_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_plan, config
    )

    assert score_with_plan == pytest.approx(
        score_no_plan - config.win_con_preservation_weight * 100.0
    )


def test_win_con_preservation_does_not_fire_below_100_percent_threat() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move, single = _earthquake_single()
    threat = [_ThreatInfo(percent=99.9, move_id="tackle", priority=0), _ThreatInfo()]

    plan = _minimal_gameplan(our_species=("garchomp",), primary_win_con_idx=0)
    ctx_no_plan = _attack_ctx(
        ally_state=None, opp_state=opp_state, threat_on_us=threat, gameplan=None
    )
    ctx_with_plan = _attack_ctx(
        ally_state=None, opp_state=opp_state, threat_on_us=threat, gameplan=plan
    )
    score_no_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_plan, config
    )
    score_with_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_plan, config
    )
    assert score_with_plan == pytest.approx(score_no_plan)


def test_win_con_preservation_does_not_fire_for_a_non_win_con_slot() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move, single = _earthquake_single()
    threat = [_ThreatInfo(percent=100.0, move_id="tackle", priority=0), _ThreatInfo()]

    # primary_win_con_idx points at a DIFFERENT species than the actor (garchomp).
    plan = _minimal_gameplan(our_species=("torkoal",), primary_win_con_idx=0)
    ctx_no_plan = _attack_ctx(
        ally_state=None, opp_state=opp_state, threat_on_us=threat, gameplan=None
    )
    ctx_with_plan = _attack_ctx(
        ally_state=None, opp_state=opp_state, threat_on_us=threat, gameplan=plan
    )
    score_no_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_plan, config
    )
    score_with_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_plan, config
    )
    assert score_with_plan == pytest.approx(score_no_plan)


def test_plan_breaker_target_bonus_added_to_contribution() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move, single = _earthquake_single()

    ctx_no_plan = _attack_ctx(ally_state=None, opp_state=opp_state, gameplan=None)
    score_no_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_plan, config
    )

    plan = _minimal_gameplan(plan_breakers=frozenset({"klefki"}))
    ctx_with_plan = _attack_ctx(ally_state=None, opp_state=opp_state, gameplan=plan)
    score_with_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_plan, config
    )

    assert score_with_plan == pytest.approx(score_no_plan + config.plan_breaker_target_bonus)


def test_plan_breaker_target_bonus_not_added_when_target_is_not_a_plan_breaker() -> None:
    config = PolicyConfig()
    opp_state = _klefki()
    move, single = _earthquake_single()

    plan = _minimal_gameplan(plan_breakers=frozenset({"some-other-mon"}))
    ctx_no_plan = _attack_ctx(ally_state=None, opp_state=opp_state, gameplan=None)
    ctx_with_plan = _attack_ctx(ally_state=None, opp_state=opp_state, gameplan=plan)
    score_no_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_no_plan, config
    )
    score_with_plan, _, _ = _score_attack_order(
        move, load_moves()["earthquake"], single, 0, ctx_with_plan, config
    )
    assert score_with_plan == pytest.approx(score_no_plan)


# --- collapsed-matchup switch bonus (the Charizard-in-rain fix) -----------------------


def _switch_ctx(
    *,
    outgoing_state: PokemonState,
    outgoing_moves: list[str],
    opp_state: PokemonState,
    opp_moves: list[str],
    weather: str | None = None,
    gameplan=None,
) -> _Context:
    our_pokemon = [
        SimpleNamespace(
            species=outgoing_state.species_id, moves=dict.fromkeys(outgoing_moves), fainted=False
        ),
        None,
    ]
    opp_pokemon = [
        SimpleNamespace(
            species=opp_state.species_id,
            moves=dict.fromkeys(opp_moves),
            ability=opp_state.ability,
            fainted=False,
        ),
        None,
    ]
    return _Context(
        battle=SimpleNamespace(side_conditions=[], opponent_side_conditions=[]),
        trick_room=False,
        weather=weather,
        terrain=None,
        our_side_screens=frozenset(),
        opp_side_screens=frozenset(),
        our_pokemon=our_pokemon,
        opp_pokemon=opp_pokemon,
        our_states=[outgoing_state, None],
        opp_states=[opp_state, None],
        our_speed=[effective_speed(outgoing_state), 0.0],
        opp_speed=[effective_speed(opp_state), 0.0],
        threat_on_us=[_ThreatInfo(), _ThreatInfo()],
        opp_threat_score=[0.0, 0.0],
        opp_protect_prob=[0.0, 0.0],
        opp_switch_prob=[0.0, 0.0],
        priors={},
        gameplan=gameplan,
    )


def test_collapsed_matchup_switch_bonus_fires_on_weather_flip() -> None:
    from poke_env.battle.pokemon import Pokemon

    from vgc.gameplan import _best_move_percent

    config = PolicyConfig()
    charizard = PokemonState(
        "charizard", sp_spread={"hp": 2, "spa": 32, "spe": 32}, nature="modest"
    )
    klefki = _klefki()
    # The gameplan table's NEUTRAL-field expectation for this exact matchup -- computed
    # via the same helper build_gameplan itself uses, so this isn't a hand-guessed float.
    table_pct = _best_move_percent(charizard, ["heatwave"], klefki)
    plan = _minimal_gameplan(matrix={("charizard", "klefki"): table_pct})

    incoming = Pokemon(gen=9, species="incineroar")
    incoming._current_hp = 100

    ctx_rain = _switch_ctx(
        outgoing_state=charizard,
        outgoing_moves=["heatwave"],
        opp_state=klefki,
        opp_moves=["playrough"],
        weather="rain",
        gameplan=plan,
    )
    score_rain, info_rain = _score_switch(incoming, 0, ctx_rain, config)

    ctx_neutral = _switch_ctx(
        outgoing_state=charizard,
        outgoing_moves=["heatwave"],
        opp_state=klefki,
        opp_moves=["playrough"],
        weather=None,
        gameplan=plan,
    )
    score_neutral, info_neutral = _score_switch(incoming, 0, ctx_neutral, config)

    # Rain roughly halves Heat Wave's output vs the same neutral-field table value --
    # below collapsed_matchup_floor -- so real switch pressure should appear in rain...
    assert info_rain["collapsed_matchup_bonus"] > 0.0
    # ...and NOT under the exact conditions the table was built for (current == table).
    assert info_neutral["collapsed_matchup_bonus"] == pytest.approx(0.0)
    assert score_rain > score_neutral


def test_collapsed_matchup_switch_bonus_absent_without_a_gameplan() -> None:
    config = PolicyConfig()
    charizard = PokemonState(
        "charizard", sp_spread={"hp": 2, "spa": 32, "spe": 32}, nature="modest"
    )
    klefki = _klefki()

    from poke_env.battle.pokemon import Pokemon

    incoming = Pokemon(gen=9, species="incineroar")
    incoming._current_hp = 100

    ctx = _switch_ctx(
        outgoing_state=charizard,
        outgoing_moves=["heatwave"],
        opp_state=klefki,
        opp_moves=["playrough"],
        weather="rain",
        gameplan=None,
    )
    _score, info = _score_switch(incoming, 0, ctx, config)
    assert info["collapsed_matchup_bonus"] == 0.0
