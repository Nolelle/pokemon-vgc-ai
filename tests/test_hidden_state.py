"""Beliefs over privately rolled timers must stay legal, weighted, and complete."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vgc.mechanics_state import EffectSnapshot
from vgc.models import PolicyConfig
from vgc.rl.exact_search import combine_belief_rankings
from vgc.rl.hidden_state import (
    CONFUSION_DURATION_PRIOR,
    SLEEP_DURATION_PRIOR,
    HiddenStateHypothesis,
    enumerate_hidden_state_hypotheses,
)
from tests.mechanics_state_helpers import make_battle_state, make_pokemon_state


def _sleep_times(hypotheses, side: str, species: str) -> dict[int, float]:
    return {
        hypothesis.payload[side][species]["sleepTime"]: hypothesis.weight
        for hypothesis in hypotheses
    }


def test_no_hidden_timer_yields_one_certain_branch() -> None:
    state = make_battle_state()

    hypotheses = enumerate_hidden_state_hypotheses(state)

    assert hypotheses == [HiddenStateHypothesis(weight=1.0)]


def test_fresh_sleep_keeps_both_champions_durations_with_their_real_prior() -> None:
    state = make_battle_state(
        our_pokemon=[make_pokemon_state("amoonguss", status="slp", status_counter=0)]
    )

    times = _sleep_times(enumerate_hidden_state_hypotheses(state), "our", "amoonguss")

    # `sample([2, 3, 3])`: two opportunities one time in three, otherwise three.
    assert times == pytest.approx({2: 1.0 / 3.0, 3: 2.0 / 3.0})
    assert dict(SLEEP_DURATION_PRIOR) == pytest.approx({2: 1.0 / 3.0, 3: 2.0 / 3.0})


def test_one_denied_opportunity_shortens_every_surviving_duration() -> None:
    state = make_battle_state(
        our_pokemon=[make_pokemon_state("amoonguss", status="slp", status_counter=1)]
    )

    times = _sleep_times(enumerate_hidden_state_hypotheses(state), "our", "amoonguss")

    assert times == pytest.approx({1: 1.0 / 3.0, 2: 2.0 / 3.0})


def test_two_denied_opportunities_leave_only_the_three_turn_roll() -> None:
    state = make_battle_state(
        our_pokemon=[make_pokemon_state("amoonguss", status="slp", status_counter=2)]
    )

    hypotheses = enumerate_hidden_state_hypotheses(state)

    assert _sleep_times(hypotheses, "our", "amoonguss") == pytest.approx({1: 1.0})


def test_early_bird_burns_two_sleep_ticks_per_opportunity() -> None:
    state = make_battle_state(
        our_pokemon=[
            make_pokemon_state(
                "dodrio", status="slp", status_counter=1, ability_id="earlybird"
            )
        ]
    )

    hypotheses = enumerate_hidden_state_hypotheses(state)

    # Both a 2-roll and a 3-roll are spent after one Early Bird opportunity, so the
    # only legal remaining state is "wakes at its next chance".
    assert _sleep_times(hypotheses, "our", "dodrio") == pytest.approx({1: 1.0})


def test_confusion_enumerates_its_uniform_two_to_five_roll() -> None:
    state = make_battle_state(
        opponent_pokemon=[
            make_pokemon_state(
                "incineroar", effects=(EffectSnapshot("confusion", turns=0),)
            )
        ]
    )

    hypotheses = enumerate_hidden_state_hypotheses(
        state, replace(PolicyConfig(), exact_search_state_hypotheses=8)
    )

    times = {
        hypothesis.payload["opponent"]["incineroar"]["confusionTime"]: hypothesis.weight
        for hypothesis in hypotheses
    }
    assert times == pytest.approx({2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25})
    assert len(CONFUSION_DURATION_PRIOR) == 4


def test_fainted_pokemon_contribute_no_belief_branches() -> None:
    state = make_battle_state(
        our_pokemon=[
            make_pokemon_state("amoonguss", status="slp", status_counter=0, fainted=True)
        ]
    )

    assert enumerate_hidden_state_hypotheses(state) == [HiddenStateHypothesis(weight=1.0)]


def test_branches_are_capped_and_renormalised_to_a_real_distribution() -> None:
    state = make_battle_state(
        our_pokemon=[
            make_pokemon_state("amoonguss", status="slp", status_counter=0),
            make_pokemon_state(
                "rillaboom", effects=(EffectSnapshot("confusion", turns=0),)
            ),
        ]
    )

    hypotheses = enumerate_hidden_state_hypotheses(
        state, replace(PolicyConfig(), exact_search_state_hypotheses=3)
    )

    assert len(hypotheses) == 3
    assert sum(hypothesis.weight for hypothesis in hypotheses) == pytest.approx(1.0)
    # Most likely first, so a truncated search keeps the branches that matter most.
    assert hypotheses == sorted(hypotheses, key=lambda h: h.weight, reverse=True)


def test_zero_branches_is_rejected_rather_than_silently_skipping_the_search() -> None:
    state = make_battle_state()

    with pytest.raises(ValueError, match="at least 1"):
        enumerate_hidden_state_hypotheses(
            state, replace(PolicyConfig(), exact_search_state_hypotheses=0)
        )


class _FakeOrder:
    def __init__(self, message: str) -> None:
        self.message = message


def _ranking(scores: dict[str, float]):
    from vgc.evaluator import ScoredOrder

    return [
        ScoredOrder(_FakeOrder(message), score, {"searched": True})
        for message, score in scores.items()
    ]


def test_belief_rankings_average_by_prior_weight() -> None:
    combined = combine_belief_rankings(
        [
            (0.25, _ranking({"move a": 0.0, "move b": 100.0})),
            (0.75, _ranking({"move a": 100.0, "move b": 0.0})),
        ]
    )

    scores = {entry.order.message: entry.score for entry in combined}
    assert scores == pytest.approx({"move a": 75.0, "move b": 25.0})
    assert combined[0].order.message == "move a"
    assert combined[0].breakdown["belief_branches"] == 2
    assert combined[0].breakdown["belief_weighted"] is True


def test_a_single_certain_belief_is_passed_through_unweighted() -> None:
    combined = combine_belief_rankings([(1.0, _ranking({"move a": 7.0}))])

    assert combined[0].score == pytest.approx(7.0)
    assert combined[0].breakdown["belief_weighted"] is False


def test_mismatched_belief_rankings_fail_instead_of_dropping_orders() -> None:
    with pytest.raises(ValueError, match="identical set of legal orders"):
        combine_belief_rankings(
            [(0.5, _ranking({"move a": 1.0})), (0.5, _ranking({"move b": 1.0}))]
        )
