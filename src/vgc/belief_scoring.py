"""Belief-mixture scoring of joint orders over hidden opponent Stat Point spreads.

`vgc.evaluator.score_joint_orders` scores every legal joint order against one point
estimate per opponent active (`vgc.sets.opponent_state`). This module scores the same
enumerated list against a capped, renormalised posterior over those hidden spreads --
the same joint-hypothesis construction `vgc.rl.live_mirror` uses for Showdown roots,
keyed here by active slot rather than species id, and without importing `vgc.rl`.

Nothing here is wired into a play path. `PolicyConfig.shortlist_belief_hypotheses`
ships at 1 (the mode) until a later rung selects the shortlist against the mixture.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import DoubleBattleOrder

from vgc.actions import enumerate_joint_orders
from vgc.battle_memory import BattleMemory
from vgc.damage import PokemonState, to_id
from vgc.evaluator import (
    ScoredOrder,
    _record_trace,
    build_context,
    score_joint_orders_in_context,
)
from vgc.meta import known_nature, recognize_meta_team
from vgc.models import PolicyConfig
from vgc.opponent_belief import build_opponent_beliefs
from vgc.sets import load_usage_spreads, opponent_spread_hypotheses, opponent_state

_SpreadAssignment = dict[int, tuple[dict[str, int], str]]
_JointHypothesis = tuple[float, _SpreadAssignment]


def _active_opponent_slots(battle) -> list[tuple[int, object]]:
    actives = list(getattr(battle, "opponent_active_pokemon", None) or ())
    return [
        (idx, mon)
        for idx, mon in enumerate(actives)
        if mon is not None and not getattr(mon, "fainted", False)
    ]


def _preview_team(battle) -> list:
    preview_team = list(getattr(battle, "teampreview_opponent_team", None) or [])
    if len(preview_team) != 6:
        opponent_team = getattr(battle, "opponent_team", {}) or {}
        preview_team = list(opponent_team.values())
    return preview_team


def _species_hypotheses(
    species_id: str,
    posterior: dict | None,
    limit: int,
) -> list[tuple[dict[str, int], str, float]]:
    """Top-`limit` spreads for one species, renormalised. Posterior when available."""

    if posterior is not None:
        belief = posterior.get(species_id)
        if belief is not None and belief.hypotheses:
            kept = sorted(belief.hypotheses, key=lambda entry: -entry.probability)[:limit]
            total = sum(entry.probability for entry in kept)
            if total > 0.0:
                return [
                    (dict(entry.sp), entry.nature, entry.probability / total)
                    for entry in kept
                ]
    return [
        (dict(spread), nature, probability)
        for spread, nature, probability in opponent_spread_hypotheses(
            species_id, limit=limit
        )
    ]


def joint_spread_hypotheses(
    battle,
    memory: BattleMemory | None,
    config: PolicyConfig | None,
    *,
    limit: int,
) -> list[_JointHypothesis]:
    """Weighted joint Stat Point/nature beliefs over the opponent's ACTIVE slots.

    Per active, non-fainted opponent slot: take the top-`limit` spreads (the battle's
    posterior via `build_opponent_beliefs` when `memory` is set, else the corpus prior
    from `vgc.sets.opponent_spread_hypotheses`), renormalise, then take the cross
    product across slots. Sorted most likely first; joint weights renormalised to 1.
    `limit=1` is exactly one hypothesis -- each slot's mode.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit!r}")
    config = config or PolicyConfig()
    slots = _active_opponent_slots(battle)
    if not slots:
        return [(1.0, {})]

    posterior = None
    if memory is not None:
        posterior = {
            belief.species_id: belief
            for belief in build_opponent_beliefs(battle, memory, config)
        }

    per_slot = [
        (idx, _species_hypotheses(to_id(getattr(mon, "species", None)), posterior, limit))
        for idx, mon in slots
    ]
    combined: list[_JointHypothesis] = []
    for choice in itertools.product(*(hypotheses for _idx, hypotheses in per_slot)):
        weight = 1.0
        assignment: _SpreadAssignment = {}
        for (idx, _hypotheses), (spread, nature, probability) in zip(
            per_slot, choice, strict=True
        ):
            weight *= probability
            assignment[idx] = (spread, nature)
        combined.append((weight, assignment))
    combined.sort(key=lambda row: row[0], reverse=True)
    total = sum(weight for weight, _assignment in combined)
    if total <= 0.0:
        return [(1.0, combined[0][1])] if combined else [(1.0, {})]
    return [(weight / total, assignment) for weight, assignment in combined]


def _scores_in_enumerate_order(
    joint_orders: Sequence[DoubleBattleOrder], scored: Sequence[ScoredOrder]
) -> list[float]:
    """Align a sorted `ScoredOrder` list back to the shared enumerate positions."""

    score_by_order_id = {id(entry.order): entry.score for entry in scored}
    return [score_by_order_id[id(order)] for order in joint_orders]


def _opp_states_for_hypothesis(
    battle,
    assignment: _SpreadAssignment,
    usage: dict,
    meta_team,
) -> dict[int, PokemonState]:
    actives = list(getattr(battle, "opponent_active_pokemon", None) or [])
    override: dict[int, PokemonState] = {}
    for idx, (spread, nature) in assignment.items():
        if idx >= len(actives):
            continue
        mon = actives[idx]
        if mon is None or getattr(mon, "fainted", False):
            continue
        # A recognised meta team tells us the nature outright; only the Stat Points stay
        # hypothetical in that case.
        meta_nature = known_nature(meta_team, mon)
        state = opponent_state(mon, usage=usage, nature_override=meta_nature)
        state.sp_spread = dict(spread)
        state.nature = meta_nature or nature
        override[idx] = state
    return override


def score_joint_orders_under_beliefs(
    battle: DoubleBattle,
    config: PolicyConfig | None = None,
    *,
    memory: BattleMemory | None = None,
) -> list[ScoredOrder]:
    """Score each legal joint order as a probability-weighted mean over spread beliefs.

    Enumerates once, scores the same list under each joint hypothesis, and returns one
    `ScoredOrder` per order whose `score` is the mixture. Not wired into play.
    """
    config = config or PolicyConfig()
    limit = config.shortlist_belief_hypotheses
    if limit < 1:
        raise ValueError(
            f"shortlist_belief_hypotheses must be at least 1, got {limit!r}"
        )
    if memory is None:
        memory = getattr(battle, "_vgc_battle_memory", None)

    joint_orders = enumerate_joint_orders(battle)
    if not joint_orders:
        return []

    hypotheses = joint_spread_hypotheses(battle, memory, config, limit=limit)
    usage = load_usage_spreads()
    meta_team = recognize_meta_team(_preview_team(battle))

    weights: list[float] = []
    per_hypothesis_aligned: list[list[float]] = []
    for weight, assignment in hypotheses:
        override = _opp_states_for_hypothesis(battle, assignment, usage, meta_team)
        ctx = build_context(
            battle, config, opp_state_override=override if override else None
        )
        scored = score_joint_orders_in_context(joint_orders, ctx, config)
        weights.append(weight)
        per_hypothesis_aligned.append(_scores_in_enumerate_order(joint_orders, scored))

    mixed: list[ScoredOrder] = []
    for index, order in enumerate(joint_orders):
        per_hypothesis_scores = [row[index] for row in per_hypothesis_aligned]
        score = sum(
            weight * value for weight, value in zip(weights, per_hypothesis_scores, strict=True)
        )
        mixed.append(
            ScoredOrder(
                order=order,
                score=score,
                breakdown={
                    "belief_mixture": {
                        "hypotheses": len(hypotheses),
                        "weights": list(weights),
                        "per_hypothesis_scores": per_hypothesis_scores,
                    }
                },
            )
        )
    mixed.sort(key=lambda scored_order: scored_order.score, reverse=True)
    _record_trace(mixed, config)
    return mixed
