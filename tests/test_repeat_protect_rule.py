"""`PolicyConfig.forbid_repeat_protect`: no back-to-back Protect, with two narrow exceptions.

Hand-built contexts and orders (no live battle). The rule sits in
`vgc.evaluator.apply_repeat_protect_rule`, called from `score_joint_orders`, which is where
the fast search, the exact judge's judged set and the clock guard's cheap order all get their
candidates.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from poke_env.battle.pokemon import Pokemon

import vgc.evaluator as ev
from poke_env.battle.move import Move
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from tests.test_search import _build_ctx, _garchomp, _klefki, _mon
from vgc.decision_trace import finish_trace, get_last_trace, start_trace
from vgc.evaluator import ScoredOrder, _ThreatInfo, apply_repeat_protect_rule
from vgc.models import PolicyConfig

CONFIG = PolicyConfig()


def _fake_single(move_id: str, move_target: int = 0) -> SingleBattleOrder:
    return SingleBattleOrder(Move(move_id, gen=9), move_target=move_target)


def _fake_order(first, second) -> DoubleBattleOrder:
    return DoubleBattleOrder(first, second)


def _switch() -> SingleBattleOrder:
    return SingleBattleOrder(Pokemon(gen=9, species="pikachu"))


def _entry(first, second, partner_ko=None, score=0.0) -> ScoredOrder:
    """Scored order; ``partner_ko`` is the slot-1 info the evaluator would record."""
    breakdown = {"slot0": {"kind": "move"}, "slot1": {"kind": "move"}}
    if partner_ko is not None:
        breakdown["slot1"] = {
            "kind": "move",
            "ko_slots": partner_ko.get("ko_slots", []),
            "hit_probability_by_target": partner_ko.get("hit", {}),
            "protect_prob_by_target": partner_ko.get("protect", {}),
            "actor_priority": partner_ko.get("priority", 0),
            "actor_speed": partner_ko.get("speed", 200.0),
        }
    return ScoredOrder(order=_fake_order(first, second), score=score, breakdown=breakdown)


def _ctx(counter=1, partner_counter=0, threat=60.0, threat2=0.0, opp_speed=(100.0, 100.0)):
    ctx = _build_ctx(
        our_states=[_garchomp(), _garchomp()],
        opp_states=[_klefki(), _klefki()],
        our_pokemon=[_mon(protect_counter=counter), _mon(protect_counter=partner_counter)],
        opp_pokemon=[_mon(), _mon()],
    )
    ctx.opp_speed = list(opp_speed)
    ctx.threat_matrix[0][0] = _ThreatInfo(percent=threat, move_id="psychic", source_slot=0)
    if threat2:
        ctx.threat_matrix[0][1] = _ThreatInfo(percent=threat2, move_id="psychic", source_slot=1)
    return ctx


def _protect():
    return _fake_single("protect")


def _attack():
    return _fake_single("tackle", 1)


def test_repeat_protect_is_removed() -> None:
    entries = [
        _entry(_protect(), _attack()),
        _entry(_attack(), _attack()),
        _entry(_switch(), _attack()),
    ]
    kept = apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG)
    assert kept == entries[1:]


def test_first_protect_is_untouched() -> None:
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, _ctx(counter=0), CONFIG) == entries


def test_only_the_repeating_slot_is_restricted() -> None:
    # Slot 0 Protected last turn; slot 1 did not, so slot 1 may Protect.
    entries = [
        _entry(_attack(), _protect()),
        _entry(_protect(), _protect()),
        _entry(_attack(), _attack()),
    ]
    kept = apply_repeat_protect_rule(entries, _ctx(counter=1, partner_counter=0), CONFIG)
    assert kept == [entries[0], entries[2]]


def test_wide_guard_is_not_a_stall_roll_repeat() -> None:
    entries = [_entry(_fake_single("wideguard"), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG) == entries


@pytest.mark.parametrize(
    "move", ["spikyshield", "kingsshield", "banefulbunker", "silktrap", "endure", "detect"]
)
def test_other_protect_family_moves_are_dropped_too(move) -> None:
    entries = [_entry(_fake_single(move), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, _ctx(counter=2), CONFIG) == entries[1:]


def test_forced_protect_is_kept() -> None:
    # Encored into Protect: every option for slot 0 is a Protect-family move.
    entries = [_entry(_protect(), _attack()), _entry(_protect(), _switch())]
    kept = apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG)
    assert kept == entries
    assert kept[0].breakdown["repeat_protect_exception"] == "forced"


def test_not_forced_when_a_switch_exists() -> None:
    entries = [_entry(_protect(), _attack()), _entry(_switch(), _attack())]
    assert apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG) == entries[1:]


KO = {"ko_slots": [0], "hit": {0: 1.0}, "protect": {0: 0.0}, "speed": 200.0}


def test_partner_ko_of_the_only_threat_keeps_the_protect() -> None:
    entries = [_entry(_protect(), _attack(), partner_ko=KO), _entry(_attack(), _attack())]
    kept = apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG)
    assert kept == entries
    assert kept[0].breakdown["repeat_protect_exception"] == "partner_ko"


@pytest.mark.parametrize(
    "ko",
    [
        {**KO, "ko_slots": []},  # no KO
        {**KO, "hit": {0: 0.4}},  # unreliable hit
        {**KO, "protect": {0: 0.8}},  # foe likely Protects
        {**KO, "speed": 50.0},  # foe moves first and still hits us
    ],
)
def test_partner_that_does_not_really_remove_the_threat_is_no_exception(ko) -> None:
    entries = [_entry(_protect(), _attack(), partner_ko=ko), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG) == entries[1:]


def test_partner_must_ko_every_threatening_foe() -> None:
    # Two foes threaten slot 0; the partner KOs only foe 0.
    entries = [_entry(_protect(), _attack(), partner_ko=KO), _entry(_attack(), _attack())]
    ctx = _ctx(counter=1, threat2=40.0)
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries[1:]
    # A foe whose best hit is below the threat floor does not count as a threat.
    ctx = _ctx(counter=1, threat2=CONFIG.protect_low_threat_floor - 1.0)
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries


def test_trick_room_flips_who_resolves_first() -> None:
    entries = [
        _entry(_protect(), _attack(), partner_ko={**KO, "speed": 50.0}),
        _entry(_attack(), _attack()),
    ]
    ctx = _ctx(counter=1)
    ctx.trick_room = True  # the slower partner now goes first
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries


def test_no_threat_at_all_is_no_exception() -> None:
    entries = [_entry(_protect(), _attack(), partner_ko=KO), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, _ctx(counter=1, threat=0.0), CONFIG) == entries[1:]


def test_empty_result_falls_back_to_the_unfiltered_list() -> None:
    # Each order puts a repeat Protect on a different slot and neither slot is forced.
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _protect())]
    ctx = _ctx(counter=1, partner_counter=1)
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries


def test_missing_slot_never_triggers() -> None:
    entries = [_entry(_protect(), _attack())]
    ctx = _ctx(counter=1)
    ctx.our_pokemon[0] = None
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries


def test_trace_note_when_orders_are_removed(monkeypatch) -> None:
    monkeypatch.setenv("VGC_TRACE", "1")
    token = start_trace()
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    apply_repeat_protect_rule(entries, _ctx(counter=1), CONFIG)
    finish_trace(token)
    note = get_last_trace().notes["repeat_protect_rule"]
    assert note["removed"] == 1 and note["kept"] == 1


def _patched_score(monkeypatch, entries, ctx):
    monkeypatch.setattr(ev, "enumerate_joint_orders", lambda b: [e.order for e in entries])
    monkeypatch.setattr(ev, "build_context", lambda b, c: ctx)
    monkeypatch.setattr(ev, "score_joint_orders_in_context", lambda o, c, cfg: list(entries))
    return SimpleNamespace()


def test_knob_and_opponent_flag_control_score_joint_orders(monkeypatch) -> None:
    """`score_joint_orders` feeds the fast search, the judge's set and the cheap fallback."""
    entries = [_entry(_protect(), _attack(), score=9.0), _entry(_attack(), _attack(), score=1.0)]
    battle = _patched_score(monkeypatch, entries, _ctx(counter=1))
    assert ev.score_joint_orders(battle, CONFIG) == entries[1:]
    off = replace(CONFIG, forbid_repeat_protect=False)
    assert ev.score_joint_orders(battle, off) == entries  # knob off = legacy
    assert ev.score_joint_orders(battle, CONFIG, repeat_protect_rule=False) == entries


# -- exception (c): end-of-turn residual damage faints every threatening foe ---------------


def _frail(status: str | None, frac: float):
    state = _klefki(status=status)
    state.current_hp = max(1, int(state.max_hp() * frac))
    return state


def test_residual_ko_keeps_repeat_protect() -> None:
    ctx = _ctx(counter=1)
    ctx.opp_states[0] = _frail("psn", 0.10)  # poison takes 1/8 >= 10%
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    kept = apply_repeat_protect_rule(entries, ctx, CONFIG)
    assert kept == entries
    assert entries[0].breakdown["repeat_protect_exception"] == "residual_ko"


def test_residual_that_does_not_ko_still_drops_protect() -> None:
    ctx = _ctx(counter=1)
    ctx.opp_states[0] = _frail("psn", 0.25)  # 1/8 < 25%
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries[1:]


def test_second_surviving_threat_still_drops_protect() -> None:
    ctx = _ctx(counter=1, threat2=60.0)
    ctx.opp_states[0] = _frail("psn", 0.10)  # slot 1 foe is at full HP, unstatused
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries[1:]


def test_both_threats_residual_ko_keeps_protect() -> None:
    ctx = _ctx(counter=1, threat2=60.0)
    ctx.opp_states[0] = _frail("psn", 0.10)
    ctx.opp_states[1] = _frail("brn", 0.05)
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries


def test_residual_exception_knob_off_is_legacy() -> None:
    ctx = _ctx(counter=1)
    ctx.opp_states[0] = _frail("psn", 0.10)
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    off = replace(CONFIG, repeat_protect_residual_exception=False)
    assert apply_repeat_protect_rule(entries, ctx, off) == entries[1:]


def test_magic_guard_foe_takes_no_residual() -> None:
    ctx = _ctx(counter=1)
    ctx.opp_states[0] = _frail("psn", 0.10)
    ctx.opp_states[0].ability = "magicguard"
    entries = [_entry(_protect(), _attack()), _entry(_attack(), _attack())]
    assert apply_repeat_protect_rule(entries, ctx, CONFIG) == entries[1:]
