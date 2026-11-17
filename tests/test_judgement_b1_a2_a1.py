"""Unit tests for B1 (human_like opponent), A2 (smooth KO/threat pricing),
A4-lite (tactical opponent replies), and A1 (next-turn first-strike threat leaf).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from poke_env.battle.move import Move
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from vgc.baselines import BASELINES, make_player
from vgc.damage import DamageResult, PokemonState
from vgc.evaluator import (
     ScoredOrder,
    _Context,
    _ThreatInfo,
    _continuous_threat_severity,
    _cross_slot_adjustments,
    _score_protect,
    _score_screen,
    _score_trick_room,
    _smooth_ko_bonus,
    ko_roll_probability,
)
from vgc.models import PolicyConfig
from vgc.next_turn_threat import (
    FOCUS_FIRE_PROTECT_DISCOUNT,
    SINGLE_THREAT_PROTECT_DISCOUNT,
    _side_first_strike_threat,
    next_turn_threat_value,
)
from vgc.rl.exact_search import _opponent_replies
from vgc.stats import calculate_stats


def _dummy_context(
    *,
    turn: int = 1,
    trick_room: bool = False,
    our_speed: list[float] | None = None,
    opp_speed: list[float] | None = None,
    threat_pcts: list[float] | None = None,
    double_threat_pcts: list[float] | None = None,
    protect_counter: int = 0,
    our_hp_frac: float = 1.0,
) -> _Context:
    our_speed = our_speed or [110.0, 105.0]
    opp_speed = opp_speed or [150.0, 140.0]
    threat_pcts = threat_pcts or [90.0, 40.0]
    double_threat_pcts = double_threat_pcts or [95.0, 50.0]

    our_state_0 = PokemonState(species_id="incineroar")
    our_state_0.current_hp = max(1, round(our_hp_frac * our_state_0.max_hp()))
    our_state_1 = PokemonState(species_id="rillaboom")
    opp_state_0 = PokemonState(species_id="dragapult")
    opp_state_1 = PokemonState(species_id="garchomp")

    our_mon_0 = SimpleNamespace(species="incineroar", fainted=False, protect_counter=protect_counter)
    our_mon_1 = SimpleNamespace(species="rillaboom", fainted=False, protect_counter=0)
    opp_mon_0 = SimpleNamespace(species="dragapult", fainted=False, protect_counter=0)
    opp_mon_1 = SimpleNamespace(species="garchomp", fainted=False, protect_counter=0)

    battle = SimpleNamespace(
        turn=turn,
        side_conditions=set(),
        opponent_side_conditions=set(),
        available_switches=[[], []],
    )
    threats = [
        _ThreatInfo(percent=threat_pcts[0], move_id="dragonclaw", priority=0, source_slot=0),
        _ThreatInfo(percent=threat_pcts[1], move_id="earthquake", priority=0, source_slot=1),
    ]
    return _Context(
        battle=battle,
        trick_room=trick_room,
        weather=None,
        terrain=None,
        our_side_screens=frozenset(),
        opp_side_screens=frozenset(),
        our_pokemon=[our_mon_0, our_mon_1],
        opp_pokemon=[opp_mon_0, opp_mon_1],
        our_states=[our_state_0, our_state_1],
        opp_states=[opp_state_0, opp_state_1],
        our_speed=our_speed,
        opp_speed=opp_speed,
        threat_on_us=threats,
        opp_threat_score=[85.0, 50.0],
        opp_protect_prob=[0.1, 0.1],
        opp_switch_prob=[0.0, 0.0],
        priors={},
        gameplan=None,
        double_target_threat=double_threat_pcts,
    )


def test_b1_human_like_baseline_registered_and_configurable():
    assert "human_like" in BASELINES
    team = (Path(__file__).resolve().parents[1] / "teams" / "dev.packed.txt").read_text().strip()
    player = make_player("human_like", team=team, start_listening=False)
    assert player.config.human_like_tactical_mode is True
    assert player.config.continuous_expected_death is True
    assert player.config.smooth_ko_roll_bonus is True


def test_b1_early_speed_control_priority_turns_1_and_2():
    ctx_t1 = _dummy_context(turn=1, our_speed=[100.0, 95.0], opp_speed=[150.0, 140.0])
    ctx_t3 = _dummy_context(turn=3, our_speed=[100.0, 95.0], opp_speed=[150.0, 140.0])
    base_cfg = PolicyConfig(human_like_tactical_mode=False)
    human_cfg = PolicyConfig(
        human_like_tactical_mode=True,
        human_like_early_speed_control_bonus=55.0,
    )

    tw_base, _ = _score_screen("tailwind", ctx_t1, base_cfg, actor_slot=0)
    tw_human_t1, info_t1 = _score_screen("tailwind", ctx_t1, human_cfg, actor_slot=0)
    tw_human_t3, info_t3 = _score_screen("tailwind", ctx_t3, human_cfg, actor_slot=0)
    assert tw_human_t1 - tw_base == 55.0
    assert info_t1["human_like_early_speed_control_bonus"] == 55.0
    assert tw_human_t3 == tw_base
    assert info_t3["human_like_early_speed_control_bonus"] == 0.0

    tr_base, _ = _score_trick_room(ctx_t1, base_cfg, actor_slot=0)
    tr_human_t1, tr_info = _score_trick_room(ctx_t1, human_cfg, actor_slot=0)
    assert tr_human_t1 - tr_base == 55.0
    assert tr_info["human_like_early_speed_control_bonus"] == 55.0


def test_b1_focus_fire_and_defensive_protect():
    ctx = _dummy_context(turn=1, threat_pcts=[85.0, 30.0], double_threat_pcts=[90.0, 30.0])
    base_cfg = PolicyConfig(human_like_tactical_mode=False)
    human_cfg = PolicyConfig(
        human_like_tactical_mode=True,
        human_like_focus_fire_bonus=40.0,
        human_like_defensive_protect_bonus=35.0,
        human_like_protect_threat_ratio=0.80,
    )

    p_base, _ = _score_protect(0, ctx, base_cfg)
    p_human, p_info = _score_protect(0, ctx, human_cfg)
    assert p_human - p_base == 35.0
    assert p_info["human_like_defensive_protect_bonus"] == 35.0

    # Repeat protect (protect_counter == 1) never gets the human_like defensive Protect bonus
    ctx_repeat = _dummy_context(
        turn=2, threat_pcts=[85.0, 30.0], double_threat_pcts=[90.0, 30.0], protect_counter=1
    )
    _, p_rep_info = _score_protect(0, ctx_repeat, human_cfg)
    assert p_rep_info["human_like_defensive_protect_bonus"] == 0.0

    # Focus fire on target 0 (the highest-threat opposing slot: 1.25x bonus)
    slot0_info = {
        "single_target_slot": 0,
        "guaranteed_ko_slots": (),
        "survival_guard_slots": (),
        "expected_percent_by_target": {0: 55.0},
        "current_hp_percent_by_target": {0: 100.0},
        "protect_prob_by_target": {0: 0.0},
    }
    slot1_info = {
        "single_target_slot": 0,
        "guaranteed_ko_slots": (),
        "survival_guard_slots": (),
        "expected_percent_by_target": {0: 50.0},
        "current_hp_percent_by_target": {0: 100.0},
        "protect_prob_by_target": {0: 0.0},
    }
    cross_base = _cross_slot_adjustments(slot0_info, slot1_info, base_cfg, ctx=ctx)
    cross_human = _cross_slot_adjustments(slot0_info, slot1_info, human_cfg, ctx=ctx)
    assert cross_human - cross_base == 40.0 * 1.25


def test_a2_smooth_ko_roll_probability_and_continuous_threat():
    cfg = PolicyConfig(
        smooth_ko_roll_bonus=True,
        likely_ko_bonus=20.0,
        guaranteed_ko_bonus=35.0,
    )
    rolls = list(range(85, 101))  # 16 rolls: 85..100
    res = DamageResult(
        min_damage=85,
        max_damage=100,
        expected_damage=92.5,
        min_percent=85.0,
        max_percent=100.0,
        expected_percent=92.5,
        breakdown={"move_supported": True, "immune": False, "rolls": rolls},
    )
    assert ko_roll_probability(res, 101) == 0.0
    assert ko_roll_probability(res, 85) == 1.0
    assert ko_roll_probability(res, 93) == 8 / 16

    assert _smooth_ko_bonus(0.0, cfg) == 0.0
    assert _smooth_ko_bonus(0.5, cfg) == 20.0
    assert _smooth_ko_bonus(1.0, cfg) == 35.0
    # Smoothly monotone across all 17 roll counts
    bonuses = [_smooth_ko_bonus(k / 16.0, cfg) for k in range(17)]
    for a, b in zip(bonuses, bonuses[1:], strict=False):
        assert a < b

    # Continuous sub-100% threat severity
    assert _continuous_threat_severity(60.0, 100.0, 0.65) == 0.0
    sev_80 = _continuous_threat_severity(80.0, 100.0, 0.65)
    sev_95 = _continuous_threat_severity(95.0, 100.0, 0.65)
    sev_100 = _continuous_threat_severity(100.0, 100.0, 0.65)
    assert 0.0 < sev_80 < sev_95 < sev_100 == 1.0


def test_a4_lite_injects_speed_control_and_focus_fire_replies():
    def _make_scored(move_a: str, target_a: int, move_b: str, target_b: int, score: float, exp_a=None, exp_b=None):
        order = DoubleBattleOrder(
            SingleBattleOrder(Move(move_a, gen=9), move_target=target_a),
            SingleBattleOrder(Move(move_b, gen=9), move_target=target_b),
        )
        breakdown = {
            "slot0": {
                "expected_percent_by_target": exp_a or {},
                "current_hp_percent_by_target": {0: 100.0, 1: 100.0},
            },
            "slot1": {
                "expected_percent_by_target": exp_b or {},
                "current_hp_percent_by_target": {0: 100.0, 1: 100.0},
            },
        }
        return ScoredOrder(order=order, score=score, breakdown=breakdown)

    candidates = [
        _make_scored("shadowball", 1, "dragonclaw", 2, 100.0),
        _make_scored("shadowball", 2, "dragonclaw", 1, 98.0),
        _make_scored("thunderbolt", 1, "dragonclaw", 2, 96.0),
        _make_scored("thunderbolt", 2, "dragonclaw", 1, 94.0),
        _make_scored("tailwind", 0, "dragonclaw", 1, 88.0),
        _make_scored("shadowball", 1, "dragonclaw", 1, 86.0, exp_a={0: 55.0}, exp_b={0: 52.0}),
    ]
    cfg_off = PolicyConfig(search_opp_candidates=4, exact_search_tactical_replies=False)
    cfg_on = PolicyConfig(search_opp_candidates=4, exact_search_tactical_replies=True)

    replies_off = _opponent_replies(candidates, cfg_off)
    replies_on = _opponent_replies(candidates, cfg_on)
    assert len(replies_off) == 4
    assert len(replies_on) == 4
    messages_on = {r.order.message for r in replies_on}
    assert candidates[0].order.message in messages_on
    assert candidates[4].order.message in messages_on  # Tailwind reply injected
    assert candidates[5].order.message in messages_on  # Focus-fire KO reply injected


def _mech_mon(
    species: str,
    moves: tuple[str, ...],
    *,
    ours: bool = True,
    hp_frac: float = 1.0,
    protect_counter: int = 0,
    spe_sp: int = 0,
    atk_sp: int = 0,
    spa_sp: int = 0,
):
    sp = {"hp": 0, "atk": atk_sp, "def": 0, "spa": spa_sp, "spd": 0, "spe": spe_sp}
    stats = calculate_stats(species, sp=sp, nature="serious")
    max_hp = stats["hp"]
    cur_hp = max(1, round(hp_frac * max_hp))
    return SimpleNamespace(
        species_id=species,
        base_species_id=species,
        fainted=False,
        active=True,
        current_hp=cur_hp,
        max_hp=max_hp,
        ability_id=None,
        ability_known=ours,
        item_id=None,
        item_known=ours,
        item_lost_this_stint=False,
        status=None,
        must_recharge=False,
        protect_counter=protect_counter,
        effects=(),
        boosts=(("atk", 0), ("def", 0), ("spa", 0), ("spd", 0), ("spe", 0)),
        stats=tuple(stats.items()),
        stat_points=tuple(sp.items()) if ours else None,
        nature="serious",
        last_move_id=None,
        moves=tuple(
            SimpleNamespace(id=m, disabled=False, current_pp=8, max_pp=8) for m in moves
        ),
    )


def _mech_state(ours, theirs, *, turn=4, weather=(), fields=(), our_cond=(), their_cond=()):
    return SimpleNamespace(
        turn=turn,
        won=False,
        lost=False,
        finished=False,
        weather=tuple(weather),
        fields=tuple(fields),
        our_side=SimpleNamespace(pokemon=tuple(ours), side_conditions=tuple(our_cond)),
        opponent_side=SimpleNamespace(pokemon=tuple(theirs), side_conditions=tuple(their_cond)),
    )


def test_a1_doubles_contention_and_protect_discount_and_no_double_count():
    cfg = PolicyConfig(
        exact_search_next_turn_threat=True,
        exact_search_next_turn_threat_weight=0.5,
        exact_search_alive_weight=90.0,
    )
    # One fast attacker (Dragapult) with a single-target OHKO against two low-HP slower
    # defenders (Torkoal and Ursaluna at 20% HP, no Protect).
    fast_foe = _mech_mon("dragapult", ("shadowball",), ours=False, spe_sp=32, spa_sp=32)
    def0_low = _mech_mon("torkoal", ("flamethrower",), ours=True, hp_frac=0.20, spe_sp=0)
    def1_low = _mech_mon("ursaluna", ("headlongrush",), ours=True, hp_frac=0.20, spe_sp=0)

    state_one_def = _mech_state([def0_low], [fast_foe])
    state_two_defs = _mech_state([def0_low, def1_low], [fast_foe])

    threat_one = _side_first_strike_threat(
        state_one_def.opponent_side,
        state_one_def.our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    threat_two = _side_first_strike_threat(
        state_two_defs.opponent_side,
        state_two_defs.our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    # Marginal value at 20% HP is ~100*0.20 + 90 = 110.
    # Doubles contention: one single-target foe can only KO ONE of the two slots on t+1!
    assert abs(threat_one - threat_two) < 1e-6
    assert 105.0 <= threat_one <= 115.0

    # No double-counting: a full-HP defender that gets OHKO'd has marginal at-risk value
    # 100*1.0 + 90 = 190 > 110 (the 20%-HP defender already lost 80 HP on turn t).
    chandelure_20 = _mech_mon("chandelure", ("willowisp",), ours=True, hp_frac=0.20, spe_sp=0)
    chandelure_50 = _mech_mon("chandelure", ("willowisp",), ours=True, hp_frac=0.50, spe_sp=0)
    t_20 = _side_first_strike_threat(
        _mech_state([chandelure_20], [fast_foe]).opponent_side,
        _mech_state([chandelure_20], [fast_foe]).our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    t_50 = _side_first_strike_threat(
        _mech_state([chandelure_50], [fast_foe]).opponent_side,
        _mech_state([chandelure_50], [fast_foe]).our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    assert t_50 > t_20

    # Protect discount: when the defender has Protect and protect_counter == 0, single-foe
    # threat is discounted by SINGLE_THREAT_PROTECT_DISCOUNT (0.50), vs 1.0 when counter == 1,
    # and FOCUS_FIRE_PROTECT_DISCOUNT (0.85) when two foes focus-fire it.
    def_with_protect_fresh = _mech_mon(
        "torkoal", ("flamethrower", "protect"), ours=True, hp_frac=0.20, protect_counter=0, spe_sp=0
    )
    def_with_protect_used = _mech_mon(
        "torkoal", ("flamethrower", "protect"), ours=True, hp_frac=0.20, protect_counter=1, spe_sp=0
    )
    t_fresh = _side_first_strike_threat(
        _mech_state([def_with_protect_fresh], [fast_foe]).opponent_side,
        _mech_state([def_with_protect_fresh], [fast_foe]).our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    t_used = _side_first_strike_threat(
        _mech_state([def_with_protect_used], [fast_foe]).opponent_side,
        _mech_state([def_with_protect_used], [fast_foe]).our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    assert abs(t_fresh - SINGLE_THREAT_PROTECT_DISCOUNT * t_used) < 1e-6

    fast_foe_2 = _mech_mon("greninja", ("hydropump",), ours=False, spe_sp=32, spa_sp=32)
    t_focus = _side_first_strike_threat(
        _mech_state([def_with_protect_fresh], [fast_foe, fast_foe_2]).opponent_side,
        _mech_state([def_with_protect_fresh], [fast_foe, fast_foe_2]).our_side,
        attacker_is_ours=False,
        att_tailwind=False,
        def_tailwind=False,
        weather=None,
        terrain=None,
        trick_room=False,
        config=cfg,
    )
    assert abs(t_focus - FOCUS_FIRE_PROTECT_DISCOUNT * t_used) < 1e-6


def test_a1_symmetry_and_trick_room_flip():
    cfg = PolicyConfig(
        exact_search_next_turn_threat=True,
        exact_search_next_turn_threat_weight=0.5,
        exact_search_alive_weight=90.0,
    )
    our_slow = _mech_mon("ursaluna", ("headlongrush",), ours=True, hp_frac=0.25, spe_sp=0, atk_sp=32)
    opp_fast = _mech_mon("dragapult", ("shadowball",), ours=False, hp_frac=0.25, spe_sp=32, spa_sp=32)

    # Without Trick Room, Dragapult moves first and threatens Ursaluna -> negative value for us.
    bare_state = _mech_state([our_slow], [opp_fast], turn=4)
    val_bare = next_turn_threat_value(bare_state, cfg)
    assert val_bare < 0.0

    # With Trick Room active (started turn 3), Ursaluna moves first and threatens Dragapult -> positive!
    tr_effect = SimpleNamespace(id="trickroom", turns=3)
    tr_state = _mech_state([our_slow], [opp_fast], turn=4, fields=(tr_effect,))
    val_tr = next_turn_threat_value(tr_state, cfg)
    assert val_tr > 0.0
