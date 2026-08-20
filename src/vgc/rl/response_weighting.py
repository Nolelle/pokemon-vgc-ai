"""How much does weighting the two collected opponent responses actually matter?

VGC turns are simultaneous: the opponent commits their action without seeing ours, so
the quantity we would deploy is

    Q(o, a) = sum_b P(b | o) * Q(o, a, b)

with ``P(b | o)`` -- NOT ``P(b | o, a)``.  The counterfactual collector
(``offline/evaluate_counterfactual_value.py``) already matches that structure: it picks
the same top-K opponent responses at a root and pairs each of them against every one of
our candidate actions, so ``b`` is chosen independently of ``a``.

Today every collected root uses ``P(b | o) = uniform`` over the opponent's top TWO search
responses.  This module measures whether a better ``P(b | o)`` could buy anything, by
sweeping the single free parameter that a two-response distribution has: ``weight``, the
probability mass on ``b0`` (the opponent's higher-scoring search response).

Two scopes that are easy to conflate, kept separate throughout:

* **Truth-side sensitivity / headroom** (``sensitivity_summary``, ``headroom_summary``)
  needs no learned model at all.  It asks: as the true response weight ``p`` moves across
  [0, 1], does the identity of our best action change, and how much value does committing
  to uniform leave on the table?  If the best action is the same for every ``p``, then no
  opponent model over these two responses can change a single decision, and the question
  is closed.
* **Deployment rules** (``evaluate_rule``) ask a narrower question: given a model's
  predicted ``Q(o, a, b)``, does aggregating with something other than 50/50 pick better
  actions?

Both are strictly limited to the top-two responses the collector stored.  Nothing here
can say whether a response OUTSIDE the top two matters -- that needs new data.

**"Oracle" means an optimistic upper bound computed from outcomes we would not have in a
real battle.**  ``headroom_summary`` scores each root against the true ``p`` for that
root; a deployed policy never gets that.  Treat the numbers as a ceiling, not a forecast.

Every estimate here rests on 4 exact continuations per (action, response) pair, so each
``Q(o, a, b)`` is a mean of four +/-1 outcomes and lands on a multiple of 0.5.  That is
coarse enough to manufacture apparent headroom out of nothing, which is what
``null_resample`` exists to measure: it rebuilds the dataset under "the response genuinely
does not matter" and re-runs the same statistic, giving the noise floor any real effect
has to clear.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# The two responses are ordered by opponent search score, so index 0 is always the
# opponent's preferred reply.  ``weight`` throughout is the probability mass on index 0.
RESPONSE_COUNT = 2

# Ranking ties are compared with a tolerance rather than exact equality because the
# aggregates are floating-point combinations of quarter-integer outcomes.
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class RootTable:
    """One root's exact-continuation payoff matrix over (our action, their response)."""

    root_id: str
    turn_bucket: str
    team_id: str
    opponent_team_id: str
    own_actions: tuple[str, ...]
    responses: tuple[str, ...]
    outcomes: np.ndarray  # (actions, 2) mean continuation outcome, our point of view
    rollout_counts: np.ndarray  # (actions, 2)
    search_scores: np.ndarray  # (actions,) our search score, constant across responses
    response_scores: np.ndarray  # (2,) opponent search score per response

    @property
    def action_count(self) -> int:
        return len(self.own_actions)


def build_root_tables(
    samples: Iterable[Any],
) -> tuple[list[RootTable], dict[str, int]]:
    """Group Q samples into per-root payoff matrices, skipping unusable roots.

    Returns the tables plus a count of why roots were skipped.  A root is usable only if
    it has at least two of our actions and exactly two opponent responses, with every
    (action, response) cell present -- a ragged root has no well-defined weighted
    aggregate.
    """

    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[sample.root_id].append(sample)
    tables: list[RootTable] = []
    skipped: dict[str, int] = defaultdict(int)
    for root_id in sorted(grouped):
        rows = grouped[root_id]
        responses = sorted(
            {(row.opponent_search_score, row.opponent_action) for row in rows},
            key=lambda item: (-item[0], item[1]),
        )
        actions = sorted(
            {(row.search_score, row.own_action) for row in rows},
            key=lambda item: (-item[0], item[1]),
        )
        if len({name for _score, name in actions}) != len(actions):
            skipped["action_score_disagreement"] += 1
            continue
        if len(responses) != RESPONSE_COUNT:
            skipped[f"responses_{len(responses)}"] += 1
            continue
        if len(actions) < 2:
            skipped["single_action"] += 1
            continue
        cells = {(row.own_action, row.opponent_action): row for row in rows}
        if len(cells) != len(actions) * RESPONSE_COUNT:
            skipped["ragged_matrix"] += 1
            continue
        action_names = tuple(name for _score, name in actions)
        response_names = tuple(name for _score, name in responses)
        try:
            matrix = [[cells[(a, b)] for b in response_names] for a in action_names]
        except KeyError:
            skipped["ragged_matrix"] += 1
            continue
        first = rows[0]
        tables.append(
            RootTable(
                root_id=root_id,
                turn_bucket=str(
                    getattr(first, "root_metadata", {}).get("turn_bucket", "unlabeled")
                ),
                team_id=first.team_id,
                opponent_team_id=first.opponent_team_id,
                own_actions=action_names,
                responses=response_names,
                outcomes=np.asarray(
                    [[cell.mean_outcome for cell in row] for row in matrix],
                    dtype=np.float64,
                ),
                rollout_counts=np.asarray(
                    [[cell.rollout_count for cell in row] for row in matrix], dtype=np.int64
                ),
                search_scores=np.asarray(
                    [matrix[index][0].search_score for index in range(len(action_names))],
                    dtype=np.float64,
                ),
                response_scores=np.asarray(
                    [matrix[0][index].opponent_search_score for index in range(RESPONSE_COUNT)],
                    dtype=np.float64,
                ),
            )
        )
    return tables, dict(skipped)


def weight_grid(steps: int) -> np.ndarray:
    """A 0..1 grid that always contains the 0.5 baseline (``steps`` must be even)."""

    if steps < 2 or steps % 2 != 0:
        raise ValueError("weight grid needs an even step count so 0.5 is on the grid")
    return np.linspace(0.0, 1.0, steps + 1)


def aggregate_values(outcomes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine the two responses at every weight: returns an (actions, weights) matrix."""

    weights = np.asarray(weights, dtype=np.float64)
    mixture = np.stack([weights, 1.0 - weights])  # (2, weights)
    return np.asarray(outcomes, dtype=np.float64) @ mixture


def _argmax(values: np.ndarray) -> np.ndarray:
    """Argmax along the action axis; ties break toward the higher search score.

    Actions are stored in descending search-score order, so ``np.argmax``'s
    first-wins behaviour already implements that tie-break.
    """

    return np.argmax(values, axis=0)


def regret_surface(table: RootTable, weights: np.ndarray) -> np.ndarray:
    """``regret[i, j]`` = value lost by deploying ``weights[i]`` when truth is ``weights[j]``.

    Both axes use the same grid, so row ``i`` is "commit to this weight" and column ``j``
    is "the opponent's real response distribution".  The diagonal is zero by construction.
    """

    values = aggregate_values(table.outcomes, weights)  # (actions, weights)
    chosen = _argmax(values)  # (weights,) action chosen at each deployed weight
    best = values.max(axis=0)  # (weights,) best attainable under each true weight
    return best[None, :] - values[chosen, :]


def sensitivity_summary(tables: Sequence[RootTable], weights: np.ndarray) -> dict[str, Any]:
    """Does the response weight change which of our actions looks best?

    Pure geometry of the collected payoffs -- no model, no oracle.  ``stable_fraction``
    is the share of roots whose best action is identical for every weight in [0, 1]; on
    those roots an opponent model over these two responses provably cannot change the
    decision.
    """

    if not tables:
        raise ValueError("sensitivity summary needs at least one root")
    baseline_index = int(np.argmin(np.abs(weights - 0.5)))
    distinct: list[int] = []
    differs_from_uniform: list[float] = []
    crossovers: list[float] = []
    for table in tables:
        values = aggregate_values(table.outcomes, weights)
        chosen = _argmax(values)
        distinct.append(len(set(chosen.tolist())))
        differs_from_uniform.append(float(np.mean(chosen != chosen[baseline_index])))
        changes = np.nonzero(np.diff(chosen))[0]
        crossovers.extend(float(weights[index + 1]) for index in changes)
    distinct_array = np.asarray(distinct, dtype=np.float64)
    return {
        "roots": len(tables),
        "stable_fraction": float(np.mean(distinct_array == 1)),
        "flip_fraction": float(np.mean(distinct_array > 1)),
        "mean_distinct_best_actions": float(distinct_array.mean()),
        "mean_weight_share_differing_from_uniform": float(np.mean(differs_from_uniform)),
        "crossover_weights": {
            "count": len(crossovers),
            "min": float(min(crossovers)) if crossovers else None,
            "median": float(np.median(crossovers)) if crossovers else None,
            "max": float(max(crossovers)) if crossovers else None,
        },
    }


def headroom_summary(tables: Sequence[RootTable], weights: np.ndarray) -> dict[str, Any]:
    """Oracle ceiling on knowing ``P(b | o)`` better, over the top two responses only.

    ``uniform_regret_mean_over_p`` is the average value uniform weighting gives up when
    the true weight is drawn uniformly from [0, 1] and an oracle could have picked the
    right action for it.  ``uniform_regret_worst_over_p`` is the worst case over ``p``.

    ``best_fixed_weight`` is reported as a consistency check, not as a result: the value
    of an action is linear in ``p``, so its average over a uniform prior on ``p`` is
    exactly its value at ``p = 0.5``.  Deploying 0.5 therefore already maximises the
    prior-averaged value, and this field must come back 0.5 on any input.  A reweighting
    can only pay off if the true ``p`` is systematically away from 0.5 -- which is why
    ``minimax_fixed_weight`` (the weight minimising the WORST case over ``p``) is the
    column that can actually move.  Weights that select the same actions tie exactly, so
    the identity is asserted on the achieved regret
    (``linearity_check_uniform_achieves_best_mean_over_p``) rather than on which of the
    tied weights ``argmin`` happens to report.

    All of these are optimistic: they score against the true ``p`` for the root, which a
    deployed policy never sees.
    """

    if not tables:
        raise ValueError("headroom summary needs at least one root")
    baseline_index = int(np.argmin(np.abs(weights - 0.5)))
    surfaces = np.stack([regret_surface(table, weights) for table in tables])
    mean_surface = surfaces.mean(axis=0)  # (deployed, true)
    uniform_row = mean_surface[baseline_index]
    average_over_p = mean_surface.mean(axis=1)  # uniform prior over the true weight
    worst_over_p = mean_surface.max(axis=1)  # minimax over the true weight
    best_index = int(np.argmin(average_over_p))
    minimax_index = int(np.argmin(worst_over_p))
    per_root_uniform_mean = surfaces[:, baseline_index, :].mean(axis=1)
    per_root_uniform_worst = surfaces[:, baseline_index, :].max(axis=1)
    return {
        "roots": len(tables),
        "uniform_regret_mean_over_p": float(uniform_row.mean()),
        "uniform_regret_worst_over_p": float(uniform_row.max()),
        "uniform_regret_by_true_weight": [float(value) for value in uniform_row],
        "per_root_uniform_regret_mean": {
            "mean": float(per_root_uniform_mean.mean()),
            "standard_error": _standard_error(per_root_uniform_mean),
            "zero_fraction": float(np.mean(per_root_uniform_mean <= TIE_TOLERANCE)),
            "max": float(per_root_uniform_mean.max()),
        },
        "per_root_uniform_regret_worst": {
            "mean": float(per_root_uniform_worst.mean()),
            "standard_error": _standard_error(per_root_uniform_worst),
            "zero_fraction": float(np.mean(per_root_uniform_worst <= TIE_TOLERANCE)),
            "max": float(per_root_uniform_worst.max()),
        },
        "best_fixed_weight": float(weights[best_index]),
        "best_fixed_weight_regret_mean_over_p": float(average_over_p[best_index]),
        "uniform_regret_mean_over_p_minus_best_fixed": float(
            average_over_p[baseline_index] - average_over_p[best_index]
        ),
        # Many weights select the same action and therefore tie on regret; what the
        # linearity argument pins down is the achieved value, not which weight reports it.
        "linearity_check_uniform_achieves_best_mean_over_p": bool(
            average_over_p[baseline_index] <= average_over_p.min() + 1e-9
        ),
        "minimax_fixed_weight": float(weights[minimax_index]),
        "minimax_fixed_weight_worst_regret": float(worst_over_p[minimax_index]),
        "uniform_worst_regret_minus_minimax": float(
            worst_over_p[baseline_index] - worst_over_p[minimax_index]
        ),
        "regret_mean_over_p_by_deployed_weight": [float(value) for value in average_over_p],
        "regret_worst_over_p_by_deployed_weight": [float(value) for value in worst_over_p],
    }


def _standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1) / math.sqrt(values.size))


# --- deployment rules -------------------------------------------------------------


def equal_weight(_table: RootTable) -> float:
    """The current baseline: 50/50 over the opponent's top two search responses."""

    return 0.5


def top_response_weight(_table: RootTable) -> float:
    """Assume the opponent always plays their higher-scoring search response."""

    return 1.0


def search_proportional_weight(table: RootTable) -> float:
    """Split in proportion to the raw opponent search scores.

    Search scores are large positive numbers whose two top entries are usually within a
    few percent of each other, so this rule sits very close to 50/50 by construction --
    it is included because it is the obvious first thing to try, not because it has much
    range.
    """

    scores = table.response_scores
    shifted = scores - min(0.0, float(scores.min()))
    total = float(shifted.sum())
    if not math.isfinite(total) or total <= 0.0:
        return 0.5
    return float(shifted[0] / total)


def softmax_weight_fn(temperature: float) -> Callable[[RootTable], float]:
    """Convert the opponent's search-score gap into a probability with a temperature.

    Lower temperature means "the opponent almost certainly plays their top response";
    very high temperature converges back to 50/50.
    """

    if temperature <= 0.0 or not math.isfinite(temperature):
        raise ValueError("softmax temperature must be finite and positive")

    def rule(table: RootTable) -> float:
        gap = float(table.response_scores[0] - table.response_scores[1])
        return float(1.0 / (1.0 + math.exp(-gap / temperature)))

    return rule


def _rule_scores(
    predictions: np.ndarray, table: RootTable, rule: str, weight_fn: Callable[[RootTable], float]
) -> np.ndarray:
    if rule == "worst_case":
        # Minimax: judge each of our actions by its worse response, no weighting at all.
        return predictions.min(axis=1)
    weight = weight_fn(table)
    return aggregate_values(predictions, np.asarray([weight]))[:, 0]


def evaluate_rule(
    tables: Sequence[RootTable],
    predictions: dict[str, np.ndarray],
    *,
    rule: str,
    weight_fn: Callable[[RootTable], float] = equal_weight,
    truth_weight: float = 0.5,
) -> dict[str, Any]:
    """Score a deployment rule's action picks against the truth at a fixed ``truth_weight``.

    ``predictions`` maps root id to an (actions, 2) matrix aligned with
    ``RootTable.own_actions`` / ``RootTable.responses``.  Passing the true outcomes as
    predictions yields the oracle upper bound for that rule.
    """

    truth = np.asarray([truth_weight], dtype=np.float64)
    correct: list[float] = []
    regrets: list[float] = []
    weights_used: list[float] = []
    for table in tables:
        predicted = predictions.get(table.root_id)
        if predicted is None:
            continue
        if predicted.shape != table.outcomes.shape:
            raise ValueError(f"prediction shape mismatch at root {table.root_id}")
        scores = _rule_scores(predicted, table, rule, weight_fn)
        chosen = int(np.argmax(scores))
        actual = aggregate_values(table.outcomes, truth)[:, 0]
        best = float(actual.max())
        correct.append(float(abs(actual[chosen] - best) <= TIE_TOLERANCE))
        regrets.append(best - float(actual[chosen]))
        weights_used.append(0.5 if rule == "worst_case" else weight_fn(table))
    if not correct:
        raise ValueError("no roots had predictions available")
    regret_array = np.asarray(regrets, dtype=np.float64)
    return {
        "rule": rule,
        "roots": len(correct),
        "truth_weight": float(truth_weight),
        "top_action_accuracy": float(np.mean(correct)),
        "mean_regret": float(regret_array.mean()),
        "regret_standard_error": _standard_error(regret_array),
        "mean_deployed_weight": float(np.mean(weights_used)),
        "deployed_weight_spread": float(np.max(weights_used) - np.min(weights_used)),
    }


def evaluate_rule_over_truth_grid(
    tables: Sequence[RootTable],
    predictions: dict[str, np.ndarray],
    truth_weights: np.ndarray,
    *,
    rule: str,
    weight_fn: Callable[[RootTable], float] = equal_weight,
) -> dict[str, Any]:
    """Score a rule at every plausible true response distribution, not just 50/50.

    Judging a deployment rule only at ``truth_weight=0.5`` quietly assumes the answer --
    the 50/50 rule wins by construction.  Averaging over the whole grid instead asks
    "which rule is best when we do not know ``P(b | o)``", which is the situation we are
    actually in.  ``mean_regret_over_truth`` uses a uniform prior over the true weight;
    ``worst_regret_over_truth`` is the minimax view.
    """

    per_truth = [
        evaluate_rule(
            tables,
            predictions,
            rule=rule,
            weight_fn=weight_fn,
            truth_weight=float(truth_weight),
        )
        for truth_weight in truth_weights
    ]
    regrets = np.asarray([row["mean_regret"] for row in per_truth], dtype=np.float64)
    accuracies = np.asarray([row["top_action_accuracy"] for row in per_truth], dtype=np.float64)
    at_uniform = min(per_truth, key=lambda row: abs(row["truth_weight"] - 0.5))
    return {
        "rule": rule,
        "roots": at_uniform["roots"],
        "mean_deployed_weight": at_uniform["mean_deployed_weight"],
        "deployed_weight_spread": at_uniform["deployed_weight_spread"],
        "mean_regret_over_truth": float(regrets.mean()),
        "worst_regret_over_truth": float(regrets.max()),
        "mean_accuracy_over_truth": float(accuracies.mean()),
        "regret_at_uniform_truth": at_uniform["mean_regret"],
        "accuracy_at_uniform_truth": at_uniform["top_action_accuracy"],
        "regret_standard_error_at_uniform_truth": at_uniform["regret_standard_error"],
        "by_truth_weight": per_truth,
    }


def sweep_fixed_weights(
    tables: Sequence[RootTable],
    predictions: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    truth_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate every constant deployment weight on the grid, against a fixed truth."""

    return [
        evaluate_rule(
            tables,
            predictions,
            rule=f"fixed_{weight:.2f}",
            weight_fn=lambda _table, value=float(weight): value,
            truth_weight=truth_weight,
        )
        for weight in weights
    ]


# --- null control -----------------------------------------------------------------


def null_resample(tables: Sequence[RootTable], rng: np.random.Generator) -> list[RootTable]:
    """Rebuild the payoffs under "the opponent's response genuinely does not matter".

    For each root and action the two observed cells are pooled into one true win
    probability, then both cells are re-drawn independently with the same rollout count.
    Any headroom the sweep reports on the resampled data is manufactured purely by having
    4 continuations per cell, which is the floor the real measurement has to clear.
    """

    resampled: list[RootTable] = []
    for table in tables:
        pooled = table.outcomes.mean(axis=1, keepdims=True)
        probability = np.clip((1.0 + pooled) / 2.0, 0.0, 1.0)
        probability = np.broadcast_to(probability, table.outcomes.shape)
        counts = np.maximum(table.rollout_counts, 1)
        wins = rng.binomial(counts, probability)
        outcomes = (2.0 * wins / counts) - 1.0
        resampled.append(replace(table, outcomes=outcomes))
    return resampled


def null_distribution(
    tables: Sequence[RootTable],
    weights: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Run the sensitivity/headroom statistics under the no-response-effect null."""

    if draws <= 0:
        raise ValueError("null control needs at least one draw")
    rng = np.random.default_rng(seed)
    flip_fractions: list[float] = []
    uniform_regrets: list[float] = []
    worst_regrets: list[float] = []
    for _draw in range(draws):
        drawn = null_resample(tables, rng)
        flip_fractions.append(sensitivity_summary(drawn, weights)["flip_fraction"])
        headroom = headroom_summary(drawn, weights)
        uniform_regrets.append(headroom["uniform_regret_mean_over_p"])
        worst_regrets.append(headroom["uniform_regret_worst_over_p"])
    return {
        "draws": draws,
        "seed": seed,
        "flip_fraction": _null_stats(np.asarray(flip_fractions)),
        "uniform_regret_mean_over_p": _null_stats(np.asarray(uniform_regrets)),
        "uniform_regret_worst_over_p": _null_stats(np.asarray(worst_regrets)),
    }


def _null_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "percentile_5": float(np.percentile(values, 5)),
        "percentile_95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def null_exceedance(observed: float, values: dict[str, float]) -> dict[str, float]:
    """Express an observed statistic in null standard deviations (0 if the null is a point)."""

    deviation = values.get("standard_deviation", 0.0)
    return {
        "observed": float(observed),
        "null_mean": float(values["mean"]),
        "excess_over_null_mean": float(observed - values["mean"]),
        "null_standard_deviations": (
            float((observed - values["mean"]) / deviation) if deviation > 0.0 else 0.0
        ),
    }


def group_by_stratum(tables: Sequence[RootTable]) -> dict[str, list[RootTable]]:
    grouped: dict[str, list[RootTable]] = defaultdict(list)
    for table in tables:
        grouped[table.turn_bucket].append(table)
    return dict(sorted(grouped.items()))
