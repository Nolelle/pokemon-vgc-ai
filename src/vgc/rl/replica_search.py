"""Bounded simultaneous one-turn search over publicly constructed possible worlds.

This module never constructs a world from a live simulator. Callers must construct
worlds from the player's observation and must supply only each side's own observation
to the action selector. The common root strategy cannot depend on which world was
sampled; the opponent may respond differently in each world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Sequence, TypeVar

import numpy as np

A = TypeVar("A")
Obs = TypeVar("Obs")
W = TypeVar("W")


@dataclass(frozen=True)
class SampledWorld(Generic[W, Obs]):
    """A reconstructed world and the opponent-visible observation in that world."""

    state: W
    opponent_observation: Obs
    weight: float = 1.0


@dataclass(frozen=True)
class ReplicaSearchResult(Generic[A]):
    actions: tuple[A, ...]
    probabilities: tuple[float, ...]
    value: float
    opponent_probabilities: tuple[tuple[float, ...], ...]
    opponent_actions: tuple[tuple[A, ...], ...]
    exploitability: float


def _actions(observation, select, key: Callable, limit: int) -> tuple:
    # Selector orders legal joint actions by its network score, best first.
    selected = tuple(select(observation))[:limit]
    if not selected:
        raise ValueError("action selector returned no legal actions")
    keys = [key(action) for action in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("action keys must be unique within a candidate list")
    return selected


def _solve(matrices: Sequence[np.ndarray], weights: np.ndarray, iterations: int):
    """Solve one common root strategy against world-specific opponent responses.

    Deterministic regret matching provides an approximate solution. The reported gap
    bounds how far the returned strategies are from equilibrium.
    """
    n = matrices[0].shape[0]
    if iterations < 1:
        raise ValueError("iterations must be positive")
    regrets = np.zeros(n)
    opp_regrets = [np.zeros(m.shape[1]) for m in matrices]
    p_sum = np.zeros(n)
    q_sums = [np.zeros(m.shape[1]) for m in matrices]
    for _ in range(iterations):
        positive = np.maximum(regrets, 0)
        p = positive / positive.sum() if positive.sum() else np.full(n, 1 / n)
        qs = []
        for r in opp_regrets:
            positive = np.maximum(r, 0)
            qs.append(positive / positive.sum() if positive.sum() else np.full(len(r), 1 / len(r)))
        action_values = sum(w * (m @ q) for w, m, q in zip(weights, matrices, qs))
        regrets += action_values - p @ action_values
        for m, q, r in zip(matrices, qs, opp_regrets):
            values = p @ m
            r += q @ values - values
        p_sum += p
        for total, q in zip(q_sums, qs):
            total += q
    p = p_sum / iterations
    qs = [q / iterations for q in q_sums]
    lower = sum(w * np.min(p @ m) for w, m in zip(weights, matrices))
    upper = np.max(sum(w * (m @ q) for w, m, q in zip(weights, matrices, qs)))
    return p, qs, float((lower + upper) / 2), float(max(0, upper - lower))


def search_sampled_worlds(
    public_observation: Obs,
    worlds: Sequence[SampledWorld[W, Obs]],
    select_actions: Callable[[Obs], Sequence[A]],
    evaluate_matrix: Callable[[W, tuple[A, ...], tuple[A, ...]], Sequence[Sequence[float]]],
    *,
    action_key: Callable[[A], Hashable],
    max_actions: int = 8,
    max_worlds: int = 16,
    iterations: int = 20000,
    worst_case_weight: float = 0.5,
) -> ReplicaSearchResult[A]:
    """Find a shared root strategy from at most 16 possible worlds and 8 actions.

    The selector is called once on the real player's public observation and once on
    each reconstructed opponent observation. ``evaluate_matrix`` must advance one
    turn for each action pair and return values from the root player's perspective;
    it is responsible for hiding private state from any value network it uses.
    Each matrix entry must average several independent engine seeds. Opponent replies
    have a uniform prior; ``worst_case_weight`` blends the worst response with that
    average (default 50/50). When every world has the same reply keys, matrices
    are averaged before solving, so the opponent cannot identify the sampled world.
    No worlds are silently discarded: callers must sample or limit them beforehand.
    """
    if not 0 <= worst_case_weight <= 1:
        raise ValueError("worst_case_weight must be between 0 and 1")
    if not 1 <= max_actions <= 8 or not 1 <= max_worlds <= 16:
        raise ValueError("limits must be between 1 and 8 actions / 16 worlds")
    if not 1 <= len(worlds) <= max_worlds:
        raise ValueError("world count outside configured limit")
    weights = np.asarray([world.weight for world in worlds], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("world weights must be finite, nonnegative and have positive sum")
    weights /= weights.sum()
    actions = _actions(public_observation, select_actions, action_key, max_actions)
    opponents, matrices = [], []
    for world in worlds:
        opponent = _actions(world.opponent_observation, select_actions, action_key, max_actions)
        matrix = np.asarray(evaluate_matrix(world.state, actions, opponent), dtype=float)
        if matrix.shape != (len(actions), len(opponent)) or not np.all(np.isfinite(matrix)):
            raise ValueError("payoff matrix has invalid shape or non-finite values")
        opponents.append(opponent)
        matrices.append(matrix)
    # A row's mean is the expected payoff under a uniform reply prior. Adding it
    # to each column preserves the simultaneous game's dimensions and blends the
    # expected and worst-response objectives without exposing the sampled world.
    matrices = [
        worst_case_weight * m + (1 - worst_case_weight) * m.mean(axis=1, keepdims=True)
        for m in matrices
    ]
    reply_keys = [tuple(map(action_key, o)) for o in opponents]
    aligned = all(set(keys) == set(reply_keys[0]) for keys in reply_keys)
    if aligned:
        # Selector rankings can differ across worlds even when the legal replies
        # are identical. Align by identity, then restore each world's own order.
        permutations = [[keys.index(k) for k in reply_keys[0]] for keys in reply_keys]
        combined = sum(w * m[:, perm] for w, m, perm in zip(weights, matrices, permutations))
        p, shared_q, value, gap = _solve([combined], np.array([1.0]), iterations)
        qs = []
        for perm in permutations:
            q = np.zeros(len(perm))
            q[perm] = shared_q[0]
            qs.append(q)
    else:
        p, qs, value, gap = _solve(matrices, weights, iterations)
    return ReplicaSearchResult(
        actions,
        tuple(map(float, p)),
        value,
        tuple(tuple(map(float, q)) for q in qs),
        tuple(opponents),
        gap,
    )
