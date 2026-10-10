"""The decision trace must explain the order that is actually sent."""

from __future__ import annotations

from types import SimpleNamespace

from tests.test_belief_scoring import _crossing_battle
from vgc.actions import describe_order
from vgc.agent import VgcPlayer
from vgc.decision_trace import finish_trace, start_trace
from vgc.evaluator import breakdown_for_order, score_joint_orders
from vgc.models import PolicyConfig


def test_chosen_breakdown_follows_the_final_order_not_the_evaluators_top_pick(monkeypatch):
    """Ladder game 2695880700 turn 8: the note described Trick while a double Protect was
    sent, because the search ranks through `score_joint_orders` again and a later ranking
    overwrote the note. `decide` re-records it for the order it returns."""
    monkeypatch.setenv("VGC_TRACE", "1")
    config = PolicyConfig(model_move_accuracy=False)
    battle = _crossing_battle()
    token = start_trace()
    ranked = score_joint_orders(battle, config)
    final = ranked[3]  # e.g. what the search/judge preferred over the myopic top pick
    assert describe_order(final.order) != describe_order(ranked[0].order)

    VgcPlayer._record_final_choice(SimpleNamespace(config=config), battle, final)
    trace = finish_trace(token)

    assert trace is not None
    notes = trace.notes
    assert notes["chosen_breakdown_order"] == describe_order(final.order)
    assert notes["final_choice"]["order"] == describe_order(final.order)
    assert notes["chosen_breakdown"] == final.breakdown
    assert notes["chosen_breakdown"] != ranked[0].breakdown
    # The myopic ranking stays visible, labelled as such.
    assert notes["top_candidates"][0]["order"] == describe_order(ranked[0].order)
    assert "not the final choice" in notes["top_candidates_source"]


def test_unfinished_trace_still_names_the_order_its_breakdown_belongs_to(monkeypatch):
    monkeypatch.setenv("VGC_TRACE", "1")
    battle = _crossing_battle()
    token = start_trace()
    ranked = score_joint_orders(battle, PolicyConfig(model_move_accuracy=False))
    trace = finish_trace(token)
    assert trace.notes["chosen_breakdown_order"] == describe_order(ranked[0].order)


def test_breakdown_for_order_matches_the_ranked_breakdown():
    config = PolicyConfig(model_move_accuracy=False)
    battle = _crossing_battle()
    ranked = score_joint_orders(battle, config)
    assert breakdown_for_order(battle, ranked[5].order, config) == ranked[5].breakdown
