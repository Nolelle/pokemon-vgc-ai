"""End-of-turn residual simulation behind the repeat-Protect residual exception.

`vgc.evaluator.residual_hp` follows Showdown's residual order with per-effect integer
rounding (see its docstring); these are the audit's repros plus a sure-KO path test.
"""

from __future__ import annotations

import vgc.evaluator as ev
from tests.test_repeat_protect_rule import _ctx, _frail
from vgc.evaluator import ResidualInputs, residual_hp


def test_grassy_heal_comes_before_burn() -> None:
    # Snorlax 235 max HP at 5 HP: heal 14 (-> 19), burn 14 (-> 5). Survives.
    inp = ResidualInputs(status="brn", grounded_terrain=True)
    assert residual_hp(5, 235, inp) == 5


def test_each_residual_effect_rounds_separately() -> None:
    # 235 max HP at 29 HP: sand 14, burn 14 -> 1 HP left, not a KO.
    inp = ResidualInputs(status="brn", weather="sand")
    assert residual_hp(29, 235, inp) == 1
    assert residual_hp(28, 235, inp) <= 0


def test_heatproof_halves_burn() -> None:
    inp = ResidualInputs(status="brn", ability="heatproof")
    assert residual_hp(6, 142, inp) == 2  # 142 // 16 = 8, halved to 4
    assert residual_hp(6, 142, ResidualInputs(status="brn")) <= 0


def test_poison_heal_only_changes_poison() -> None:
    assert residual_hp(10, 200, ResidualInputs(status="psn", ability="poisonheal")) == 35
    # An unpoisoned Poison Heal mon still faints to sand and burn.
    assert residual_hp(10, 200, ResidualInputs(weather="sand", ability="poisonheal")) <= 0
    assert residual_hp(10, 200, ResidualInputs(status="brn", ability="poisonheal")) <= 0


def test_magic_guard_and_sand_immunities() -> None:
    guarded = ResidualInputs(status="psn", weather="sand", ability="magicguard")
    assert residual_hp(1, 200, guarded) == 1
    for inp in (
        ResidualInputs(weather="sand", types=frozenset({"Rock"})),
        ResidualInputs(weather="sand", ability="overcoat"),
        ResidualInputs(weather="sand", item="safetygoggles"),
    ):
        assert residual_hp(1, 200, inp) == 1


def test_salt_cure_is_champions_fractions() -> None:
    assert residual_hp(100, 160, ResidualInputs(salt_cure=True)) == 90  # 1/16
    assert residual_hp(100, 160, ResidualInputs(salt_cure=True, types=frozenset({"Water"}))) == 80


def test_leftovers_heals_before_poison() -> None:
    inp = ResidualInputs(status="psn", item="leftovers")  # +10 then -20 at 160 max HP
    assert residual_hp(15, 160, inp) == 5
    assert residual_hp(10, 160, inp) <= 0


def test_sure_residual_ko_through_the_evaluator_path() -> None:
    ctx = _ctx(counter=1)
    ctx.opp_states[0] = _frail("psn", 0.05)
    assert ev._residual_kos(ctx.opp_states[0], ctx.opp_pokemon[0], ctx)
    ctx.opp_states[0] = _frail("brn", 0.10)  # 1/16 = 6.25% < 10%
    assert not ev._residual_kos(ctx.opp_states[0], ctx.opp_pokemon[0], ctx)
