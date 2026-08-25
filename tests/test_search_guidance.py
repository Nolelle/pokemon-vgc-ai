from __future__ import annotations

from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

import vgc.search as search_module
from vgc.actions import describe_order
from vgc.evaluator import ScoredOrder
from vgc.models import PolicyConfig
from vgc.rl.search_guidance import (
    NeuralRanking,
    select_neural_guided_candidates,
    summarize_decision_records,
)
from vgc.search import ExchangeResult, OppResponse, _OppSlotAction


def _single(target, *, move_target: int = 0):
    return SimpleNamespace(
        order=target,
        move_target=move_target,
        mega=False,
        z_move=False,
        dynamax=False,
        terastallize=False,
    )


def _joint(first, second):
    return SimpleNamespace(first_order=first, second_order=second)


def _move_order(first: str, second: str):
    return _joint(_single(Move(first, gen=9), move_target=1), _single(Move(second, gen=9)))


def _ranking(entries: list[ScoredOrder]) -> NeuralRanking:
    descriptions = tuple(describe_order(entry.order) for entry in reversed(entries))
    count = len(descriptions)
    return NeuralRanking(
        descriptions=descriptions,
        probabilities=tuple(1.0 / count for _ in descriptions),
        logits=tuple(float(index) for index in range(count)),
        state_value=0.0,
        elapsed_ms=1.0,
    )


def _strategic_entries() -> list[ScoredOrder]:
    switch = Pokemon(gen=9, species="incineroar")
    orders = [
        _move_order("heatwave", "dragonclaw"),
        _joint(_single(switch), _single(Move("protect", gen=9))),
        _move_order("protect", "protect"),
        _move_order("tailwind", "protect"),
        _move_order("dragonclaw", "moonblast"),
        _move_order("psychic", "protect"),
        _move_order("thunderbolt", "protect"),
        _move_order("earthquake", "protect"),
        _move_order("moonblast", "protect"),
        _move_order("heatwave", "protect"),
        _move_order("dragonclaw", "protect"),
        _move_order("psychic", "moonblast"),
    ]
    return [
        ScoredOrder(order=order, score=100.0 - index, breakdown={})
        for index, order in enumerate(orders)
    ]


def test_neural_guidance_keeps_exact_k_and_reserves_safety_inside_budget() -> None:
    entries = _strategic_entries()
    selected, tail, safety = select_neural_guided_candidates(
        entries,
        PolicyConfig(search_our_candidates=10),
        _ranking(entries),
        requested_safety_slots=4,
    )

    assert len(selected) == 10
    assert len(tail) == 2
    assert len({id(entry) for entry in [*selected, *tail]}) == len(entries)
    assert [item["reason"] for item in safety] == ["heuristic_top", "switch", "control"]
    assert selected[0] is entries[0]


def test_hybrid_five_caps_safety_at_half_the_budget() -> None:
    entries = _strategic_entries()
    selected, _tail, safety = select_neural_guided_candidates(
        entries,
        PolicyConfig(search_our_candidates=5),
        _ranking(entries),
        requested_safety_slots=4,
    )

    assert len(selected) == 5
    assert len(safety) == 2


def test_search_rejects_experimental_selector_that_expands_budget(monkeypatch) -> None:
    entries = _strategic_entries()[:3]
    monkeypatch.setattr(search_module, "score_joint_orders", lambda _battle, _config: entries)
    monkeypatch.setattr(search_module, "build_context", lambda _battle, _config: object())
    response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda _ctx, _config: [response]
    )
    monkeypatch.setattr(
        search_module,
        "resolve_exchange",
        lambda *_args: ExchangeResult(),
    )

    with pytest.raises(ValueError, match="expected 2"):
        search_module.search_joint_orders(
            object(),
            PolicyConfig(search_our_candidates=2, use_rolling_horizon=False),
            candidate_selector=lambda myopic, _config: (myopic, []),
        )


def test_leaf_observer_sees_baseline_results_without_changing_ranking(monkeypatch) -> None:
    entries = _strategic_entries()[:2]
    context = SimpleNamespace()
    response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(search_module, "score_joint_orders", lambda _battle, _config: entries)
    monkeypatch.setattr(search_module, "build_context", lambda _battle, _config: context)
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda _ctx, _config: [response]
    )
    monkeypatch.setattr(
        search_module,
        "resolve_exchange",
        lambda *_args: ExchangeResult(opp_hp_lost_pct=20.0),
    )
    observed = []
    config = PolicyConfig(
        search_our_candidates=2,
        use_rolling_horizon=False,
        search_hp_weight=1.0,
    )

    baseline = search_module.search_joint_orders(object(), config)
    shadow = search_module.search_joint_orders(
        object(), config, leaf_observer=observed.append
    )

    assert [entry.score for entry in shadow] == [entry.score for entry in baseline]
    assert [describe_order(entry.order) for entry in shadow] == [
        describe_order(entry.order) for entry in baseline
    ]
    assert len(observed) == 1
    assert observed[0].context is context
    assert len(observed[0].actions) == 2
    assert observed[0].actions[0].leaves[0].baseline_value == 20.0


def test_leaf_value_adjuster_changes_only_searched_leaf_values(monkeypatch) -> None:
    entries = _strategic_entries()[:2]
    context = SimpleNamespace()
    response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(search_module, "score_joint_orders", lambda _battle, _config: entries)
    monkeypatch.setattr(search_module, "build_context", lambda _battle, _config: context)
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda _ctx, _config: [response]
    )
    monkeypatch.setattr(search_module, "resolve_exchange", lambda *_args: ExchangeResult())
    config = PolicyConfig(search_our_candidates=2, use_rolling_horizon=False)

    adjusted = search_module.search_joint_orders(
        object(),
        config,
        leaf_value_adjuster=lambda batch: [[-100.0], [100.0]],
    )

    assert adjusted[0].order is entries[1].order
    assert adjusted[0].breakdown["learned_leaf_adjustment"] is True
    assert adjusted[0].breakdown["baseline_search_score"] == entries[1].score


def test_leaf_value_adjuster_rejects_wrong_shape(monkeypatch) -> None:
    entries = _strategic_entries()[:2]
    response = OppResponse(
        slot0=_OppSlotAction(kind="none"), slot1=_OppSlotAction(kind="none")
    )
    monkeypatch.setattr(search_module, "score_joint_orders", lambda _battle, _config: entries)
    monkeypatch.setattr(search_module, "build_context", lambda _battle, _config: object())
    monkeypatch.setattr(
        search_module, "_enumerate_opp_responses", lambda _ctx, _config: [response]
    )
    monkeypatch.setattr(search_module, "resolve_exchange", lambda *_args: ExchangeResult())

    with pytest.raises(ValueError, match="wrong number of actions"):
        search_module.search_joint_orders(
            object(),
            PolicyConfig(search_our_candidates=2, use_rolling_horizon=False),
            leaf_value_adjuster=lambda _batch: [[0.0]],
        )


def test_shadow_summary_clusters_recall_by_team_and_reports_cost() -> None:
    records = [
        {
            "team_id": "team-a",
            "battle_id": "a-1",
            "retention": {
                "neural_at_10": {"retained": True, "known_regret": None}
            },
            "latency_ms": {"search": 10.0, "neural_ranking": 2.0, "total": 12.0},
            "search_work": {"exchange_count": 120, "forecast_count": 120},
        },
        {
            "team_id": "team-a",
            "battle_id": "a-2",
            "retention": {
                "neural_at_10": {"retained": False, "known_regret": 40.0}
            },
            "latency_ms": {"search": 20.0, "neural_ranking": 4.0, "total": 24.0},
            "search_work": {"exchange_count": 100, "forecast_count": 100},
        },
        {
            "team_id": "team-b",
            "battle_id": "b-1",
            "retention": {
                "neural_at_10": {"retained": True, "known_regret": None}
            },
            "latency_ms": {"search": 30.0, "neural_ranking": 6.0, "total": 36.0},
            "search_work": {"exchange_count": 80, "forecast_count": 80},
        },
    ]

    summary = summarize_decision_records(records)

    metric = summary["retention"]["neural_at_10"]
    assert metric["recall"] == pytest.approx(2 / 3)
    assert metric["known_regret_mean"] == 40.0
    assert summary["teams"] == 2
    assert summary["latency_ms"]["total"]["p50"] == 24.0
    assert summary["search_work"]["exchange_count_mean"] == 100.0


# --- upset arbitration -------------------------------------------------------------------


def _scored_actions(pairs):
    """pairs: list of (description, score, searched). Orders are opaque stubs."""

    return [
        ScoredOrder(order=SimpleNamespace(description=description), score=score, breakdown={"searched": searched})
        for description, score, searched in pairs
    ]


def _describe_stub(order):
    return order.description


def test_upset_below_margin_falls_back_to_best_default(monkeypatch):
    from vgc.rl import search_guidance as sg

    monkeypatch.setattr(sg, "describe_order", _describe_stub)
    scored = _scored_actions(
        [
            ("newcomer", 100.0, True),   # hybrid-only discovery
            ("default_best", 95.0, True),
            ("other_default", 90.0, True),
            ("tail", 10.0, False),
        ]
    )
    reheaded, audit = sg._arbitrate_guided_upset(scored, {"default_best", "other_default"}, 10.0)

    assert _describe_stub(reheaded[0].order) == "default_best"
    assert audit["upset_arbitrated"] is True
    assert audit["upset_gap"] == pytest.approx(5.0)
    assert len({id(e) for e in reheaded}) == 4  # still a complete permutation


def test_upset_clearing_margin_stands(monkeypatch):
    from vgc.rl import search_guidance as sg

    monkeypatch.setattr(sg, "describe_order", _describe_stub)
    scored = _scored_actions(
        [
            ("newcomer", 120.0, True),
            ("default_best", 95.0, True),
        ]
    )
    reheaded, audit = sg._arbitrate_guided_upset(
        scored, {"default_best"}, 10.0
    )
    assert _describe_stub(reheaded[0].order) == "newcomer"
    assert audit["upset_arbitrated"] is False
    assert audit["upset_gap"] == pytest.approx(25.0)


def test_default_winner_and_zero_margin_are_untouched(monkeypatch):
    from vgc.rl import search_guidance as sg

    monkeypatch.setattr(sg, "describe_order", _describe_stub)
    scored = _scored_actions([("default_a", 50.0, True), ("default_b", 40.0, True)])
    same, audit = sg._arbitrate_guided_upset(scored, {"default_a", "default_b"}, 10.0)
    assert same is scored and audit["upset_arbitrated"] is False

    upset = _scored_actions([("newcomer", 99.0, True), ("default_a", 40.0, True)])
    untouched, audit2 = sg._arbitrate_guided_upset(upset, {"default_a"}, 0.0)
    assert _describe_stub(untouched[0].order) == "newcomer" and audit2["upset_arbitrated"] is False


def test_no_default_searched_entries_leaves_list_alone(monkeypatch):
    from vgc.rl import search_guidance as sg

    monkeypatch.setattr(sg, "describe_order", _describe_stub)
    scored = _scored_actions([("newcomer", 99.0, True)])
    same, audit = sg._arbitrate_guided_upset(scored, set(), 10.0)
    assert same is scored and audit["upset_arbitrated"] is False
