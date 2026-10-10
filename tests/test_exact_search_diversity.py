"""Passive look-ahead candidate selection and diverse opponent replies (pure logic)."""

from __future__ import annotations

from dataclasses import replace

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import DoubleBattleOrder, PassBattleOrder, SingleBattleOrder

from vgc.evaluator import ScoredOrder
from vgc.exact_judge import judge_config, narrow_for_lookahead
from vgc.models import PolicyConfig
from vgc.rl.exact_search import _opponent_replies, is_passive_order


def _single(move_id: str, target: int = 0) -> SingleBattleOrder:
    return SingleBattleOrder(Move(move_id, gen=9), move_target=target)


def _order(first, second) -> DoubleBattleOrder:
    return DoubleBattleOrder(first_order=first, second_order=second)


def _entry(first, second, score: float) -> ScoredOrder:
    return ScoredOrder(_order(first, second), score, {})


def test_passive_means_every_acting_slot_protects() -> None:
    assert is_passive_order(_order(_single("protect"), _single("protect")))
    assert is_passive_order(_order(_single("protect"), PassBattleOrder()))
    assert not is_passive_order(_order(_single("protect"), _single("earthquake")))
    assert not is_passive_order(_order(PassBattleOrder(), _single("tailwind")))
    switch = SingleBattleOrder(Pokemon(gen=9, species="pikachu"))
    assert not is_passive_order(_order(_single("protect"), switch))


def test_lookahead_narrows_only_when_a_stall_turn_competes_with_an_attack() -> None:
    cfg = PolicyConfig(exact_judge_passive_lookahead=True)
    stall = _entry(_single("protect"), _single("protect"), 5.0)
    attacks = [_entry(_single("earthquake", 1), _single("protect"), 9.0 - i) for i in range(4)]
    fast = [attacks[0], stall, *attacks[1:]]

    kept, fires = narrow_for_lookahead(fast, cfg)
    assert fires
    assert kept == [attacks[0], stall, attacks[1]]  # fast order, 2 alternatives, fast pick kept

    assert narrow_for_lookahead(attacks, cfg) == (attacks, False)  # nothing passive
    assert narrow_for_lookahead([stall], cfg) == ([stall], False)  # nothing to compare with
    off = replace(cfg, exact_judge_passive_lookahead=False)
    assert narrow_for_lookahead(fast, off) == (fast, False)
    assert PolicyConfig().exact_judge_passive_lookahead is False  # ships off until measured

    assert judge_config(cfg, 3).exact_search_passive_lookahead is True
    narrow = replace(cfg, exact_judge_passive_lookahead_samples=1)
    assert judge_config(narrow, 3, lookahead=True).exact_search_future_samples == 1
    assert judge_config(narrow, 3).exact_search_future_samples == cfg.exact_judge_future_samples


def test_diverse_replies_spread_across_plans_and_keep_protect_and_switch() -> None:
    # Six target variations of one plan outscore a Protect, a switch and a second plan.
    variants = [
        _entry(_single("woodhammer", t), _single("shadowball", u), 100.0 - i)
        for i, (t, u) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2), (1, 0), (2, 0)])
    ]
    protect = _entry(_single("woodhammer", 1), _single("protect"), 80.0)
    switch = _entry(
        _single("woodhammer", 1), SingleBattleOrder(Pokemon(gen=9, species="pikachu")), 70.0
    )
    other = _entry(_single("grassyglide", 2), _single("makeitrain"), 60.0)
    scored = [*variants, protect, switch, other]

    base = PolicyConfig(search_opp_candidates=4)
    legacy = _opponent_replies(scored, base)
    assert legacy == scored[:4]  # the top four are all one plan

    diverse = _opponent_replies(scored, replace(base, exact_search_diverse_replies=True))
    assert len(diverse) == 4
    assert diverse[0] is variants[0]  # the top reply stays
    assert protect in diverse and switch in diverse
    assert any(entry is other for entry in diverse)

    # A reply the evaluator rates hopeless is not searched just to look different.
    hopeless = _entry(_single("grassyglide", 2), _single("shadowball", 1), -500.0)
    strict = replace(base, exact_search_diverse_replies=True)
    # Nothing plausible is left for the last slots: the legacy top-N order fills them.
    moves = ["grassyglide", "makeitrain", "earthquake", "knockoff", "icepunch"]
    thin = [variants[0]] + [
        _entry(_single(m, 1), _single("shadowball", 1), -600.0 - i) for i, m in enumerate(moves)
    ]
    assert _opponent_replies(thin, strict) == thin[:4]
    plausible = _opponent_replies([*variants, protect, hopeless], strict)
    assert hopeless not in plausible
