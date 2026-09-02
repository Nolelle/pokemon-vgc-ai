"""Mechanics-exact one-turn search for offline teachers and direct battles.

The strategic score is deliberately simple and configurable.  The important contract
is that every candidate/response transition is executed by the official local Pokemon
Showdown engine. Accuracy, sleep, critical hits, target redirection, switching, items,
abilities, residual effects, form changes, and ordering therefore come from one source
of truth rather than from Python approximations.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import defaultdict
from typing import Callable, Sequence

from vgc.belief_scoring import belief_ordered_candidates
from vgc.evaluator import ScoredOrder, score_joint_orders
from vgc.mechanics_state import BattleMechanicsState, PokemonMechanicsState, snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import DirectBattle, SIDES, choice_string
from vgc.rl.mechanics_oracle import evaluate_exact_branches
from vgc.search import _select_search_candidates, _validate_selected_partition

ExactCandidateSelector = Callable[
    [list[ScoredOrder], PolicyConfig], tuple[list[ScoredOrder], list[ScoredOrder]]
]


def _future_seeds(battle_id: str, turn: int, count: int) -> list[tuple[int, ...]]:
    if count <= 0:
        raise ValueError("exact_search_future_samples must be positive")
    seeds: list[tuple[int, ...]] = []
    for sample in range(count):
        digest = hashlib.sha256(f"{battle_id}:{turn}:{sample}".encode()).digest()
        seeds.append(
            tuple(
                max(1, int.from_bytes(digest[offset : offset + 4], "big"))
                for offset in range(0, 16, 4)
            )
        )
    return seeds


def _hp_fraction(mon: PokemonMechanicsState) -> float:
    if mon.fainted:
        return 0.0
    if mon.current_hp is None or not mon.max_hp:
        return 1.0
    return max(0.0, min(1.0, mon.current_hp / mon.max_hp))


def _side_position(side, config: PolicyConfig) -> float:
    score = 0.0
    hard_control = {"slp", "frz"}
    for mon in side.pokemon:
        score += 100.0 * _hp_fraction(mon)
        if mon.status:
            weight = (
                config.exact_search_hard_control_weight
                if mon.status in hard_control
                else config.exact_search_status_weight
            )
            score -= weight
        score += config.exact_search_boost_weight * sum(stage for _name, stage in mon.boosts)
        score += config.exact_search_effect_weight * len(mon.effects)
    score += config.exact_search_effect_weight * len(side.side_conditions)
    return score


def _position_value(state: BattleMechanicsState, config: PolicyConfig) -> float:
    if state.won:
        return 10_000.0
    if state.lost:
        return -10_000.0
    return _side_position(state.our_side, config) - _side_position(
        state.opponent_side, config
    )


def _softmax_weights(scored: list[ScoredOrder], temperature: float) -> list[float]:
    if not scored:
        return [1.0]
    scale = max(float(temperature), 1e-6)
    maximum = max(entry.score for entry in scored)
    raw = [math.exp((entry.score - maximum) / scale) for entry in scored]
    total = sum(raw)
    return [value / total for value in raw]


def search_joint_orders_exact(
    root: DirectBattle,
    side: str,
    config: PolicyConfig | None = None,
    *,
    candidate_selector: ExactCandidateSelector | None = None,
) -> list[ScoredOrder]:
    """Rank a direct battle's legal orders using exact Showdown transitions."""

    started_at = time.perf_counter()
    if side not in SIDES:
        raise ValueError(f"unknown side {side!r}")
    config = config or PolicyConfig()
    battle = getattr(root, "_decision_battles", {}).get(side, root.battles[side])
    myopic = score_joint_orders(battle, config)
    if not myopic:
        return []
    ranked = belief_ordered_candidates(battle, myopic, config)
    if candidate_selector is None:
        searched, unsearched = _select_search_candidates(ranked, config)
    else:
        searched, unsearched = candidate_selector(list(ranked), config)
        _validate_selected_partition(ranked, searched, unsearched, config)

    expected = root.sides_to_move()
    other = "p2" if side == "p1" else "p1"
    if side not in expected:
        raise ValueError(f"Showdown is not waiting for {side}")
    opponent_scored = (
        score_joint_orders(root.battles[other], config)[: config.search_opp_candidates]
        if other in expected
        else []
    )
    opponent_orders = [entry.order for entry in opponent_scored] or [None]
    opponent_weights = _softmax_weights(
        opponent_scored, config.search_response_temperature
    )

    joint_choices: list[dict[str, str]] = []
    choice_owner: list[tuple[int, int]] = []
    for our_index, entry in enumerate(searched):
        for opponent_index, opponent_order in enumerate(opponent_orders):
            choices = {side: choice_string(entry.order)}
            if opponent_order is not None:
                choices[other] = choice_string(opponent_order)
            joint_choices.append(choices)
            choice_owner.append((our_index, opponent_index))

    # This is already the player's fog-safe state; no simulator internals escape.
    before = _position_value(snapshot_battle(battle), config)
    future_seeds = _future_seeds(
        root.battle_id,
        int(getattr(battle, "turn", 0) or 0),
        config.exact_search_future_samples,
    )
    branches = evaluate_exact_branches(
        root,
        joint_choices,
        future_seeds=future_seeds,
        branch_prefix=f"exact-{side}-{getattr(battle, 'turn', 0)}",
    )
    values: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_choice = len(future_seeds)
    for branch_index, branch in enumerate(branches):
        choice_index = branch_index // per_choice
        owner = choice_owner[choice_index]
        values[owner].append(_position_value(branch.state_for(side), config) - before)

    results: list[ScoredOrder] = []
    searched_finals: list[float] = []
    for our_index, entry in enumerate(searched):
        response_means = [
            sum(values[(our_index, opponent_index)]) / len(values[(our_index, opponent_index)])
            for opponent_index in range(len(opponent_orders))
        ]
        worst = min(response_means)
        expectation = sum(
            weight * value
            for weight, value in zip(opponent_weights, response_means, strict=True)
        )
        exact_delta = (
            config.search_worst_case_weight * worst
            + (1.0 - config.search_worst_case_weight) * expectation
        )
        final = (
            config.search_myopic_weight * entry.score
            + config.search_position_weight * exact_delta
        )
        breakdown = dict(entry.breakdown)
        breakdown.update(
            {
                "myopic_score": entry.score,
                "exchange_value": exact_delta,
                "searched": True,
                "mechanics_source": "official_showdown_clone",
                "exact_random_samples": len(future_seeds),
                "exact_opponent_responses": len(opponent_orders),
                "n_responses": len(opponent_orders),
                "approximate_transition": False,
            }
        )
        results.append(ScoredOrder(entry.order, final, breakdown))
        searched_finals.append(final)

    floor = min(searched_finals)
    for tail_index, entry in enumerate(unsearched):
        breakdown = dict(entry.breakdown)
        breakdown.update(
            {
                "myopic_score": entry.score,
                "searched": False,
                "mechanics_source": "unsearched_tail",
                "approximate_transition": None,
            }
        )
        results.append(ScoredOrder(entry.order, floor - 1.0 - tail_index, breakdown))
    ranked = sorted(results, key=lambda entry: entry.score, reverse=True)
    metrics = {
        "searched_actions": len(searched),
        "opponent_responses": len(opponent_orders),
        "exchange_count": len(joint_choices) * len(future_seeds),
        "forecast_count": 0,
        "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
        "mechanics_source": "official_showdown_clone",
    }
    for entry in ranked:
        entry.breakdown["search_metrics"] = metrics
    return ranked


def combine_belief_rankings(
    rankings: Sequence[tuple[float, list[ScoredOrder]]],
) -> list[ScoredOrder]:
    """Average one ranking per hidden-state belief into a single ranking.

    Every branch inside a ranking is already an exact Showdown transition. What differs
    between rankings is only which legal value a privately rolled timer took, so the
    honest summary is the prior-weighted mean of each order's score rather than the
    score under whichever duration happened to be most likely.
    """

    if not rankings:
        raise ValueError("cannot combine an empty set of belief rankings")
    total_weight = sum(weight for weight, _ranking in rankings)
    if total_weight <= 0.0:
        raise ValueError("belief weights must sum to a positive number")
    key_sets = [{entry.order.message for entry in ranking} for _weight, ranking in rankings]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("belief rankings must cover an identical set of legal orders")

    best_weight, best_ranking = max(rankings, key=lambda row: row[0])
    totals: dict[str, float] = defaultdict(float)
    for weight, ranking in rankings:
        for entry in ranking:
            totals[entry.order.message] += weight * entry.score

    combined: list[ScoredOrder] = []
    for entry in best_ranking:
        breakdown = dict(entry.breakdown)
        breakdown.update(
            {
                "belief_branches": len(rankings),
                "belief_modal_weight": best_weight / total_weight,
                "belief_weighted": len(rankings) > 1,
            }
        )
        combined.append(
            ScoredOrder(entry.order, totals[entry.order.message] / total_weight, breakdown)
        )
    return sorted(combined, key=lambda entry: entry.score, reverse=True)
