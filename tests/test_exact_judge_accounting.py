"""The exact judge's bookkeeping: things that must NOT move the score.

Each case was a verified defect (2026-10-05) that silently skewed every exact-search
ranking, teacher label, and position grade. `exact_search_consistent_accounting=False`
keeps the legacy scorecard for A/Bs, so each test also pins the old behavior.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from vgc.evaluator import ScoredOrder
from vgc.models import PolicyConfig
from vgc.rl.exact_search import _position_value, combine_belief_rankings

FIXED = PolicyConfig()
LEGACY = replace(FIXED, exact_search_consistent_accounting=False)


def _mon(hp: float = 1.0, *, fainted: bool = False, atk: int = 0):
    return SimpleNamespace(
        fainted=fainted,
        current_hp=0 if fainted else int(hp * 100),
        max_hp=100,
        status=None,
        boosts=(("atk", atk),),
        effects=(),
    )


def _state(ours, theirs, *, team_size=4, finished=False):
    return SimpleNamespace(
        won=False,
        lost=False,
        finished=finished,
        team_size=team_size,
        our_side=SimpleNamespace(pokemon=tuple(ours), side_conditions=()),
        opponent_side=SimpleNamespace(pokemon=tuple(theirs), side_conditions=()),
    )


def test_first_reveal_of_a_full_health_reserve_does_not_move_the_score() -> None:
    ours = [_mon(), _mon(), _mon(), _mon()]
    before = _state(ours, [_mon(), _mon()])
    after = _state(ours, [_mon(), _mon(), _mon()])  # a third, untouched, now seen
    assert _position_value(after, FIXED) == pytest.approx(_position_value(before, FIXED))
    assert _position_value(after, LEGACY) - _position_value(before, LEGACY) == -100.0


def test_a_fainted_pokemon_keeps_no_value_from_its_leftover_boosts() -> None:
    boosted_dead = _state([_mon()], [_mon(fainted=True, atk=2)], team_size=None)
    plain_dead = _state([_mon()], [_mon(fainted=True)], team_size=None)
    assert _position_value(boosted_dead, FIXED) == _position_value(plain_dead, FIXED)
    assert _position_value(boosted_dead, LEGACY) < _position_value(plain_dead, LEGACY)


def test_a_finished_game_without_a_winner_scores_zero() -> None:
    draw = _state([_mon()], [_mon(0.2)], team_size=None, finished=True)
    assert _position_value(draw, FIXED) == 0.0
    assert _position_value(draw, LEGACY) == pytest.approx(80.0)


def test_combined_beliefs_report_averaged_parts_not_the_modal_beliefs() -> None:
    order = SimpleNamespace(message="move 1, move 1")

    def ranking(score: float, exchange: float):
        return [ScoredOrder(order, score, {"exchange_value": exchange, "myopic_score": 10.0})]

    combined = combine_belief_rankings([(0.6, ranking(10.0, 0.0)), (0.4, ranking(60.0, 50.0))])
    assert combined[0].score == pytest.approx(30.0)
    assert combined[0].breakdown["exchange_value"] == pytest.approx(20.0)
    assert combined[0].breakdown["myopic_score"] == pytest.approx(10.0)
