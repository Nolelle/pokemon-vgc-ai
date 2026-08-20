#!/usr/bin/env python
"""Exact branch-and-continue audit of learned value on alternative VGC actions.

This is a diagnostic, not a playing-policy change. For each held-out root position it:

1. takes the strongest few actions from the unchanged search;
2. restores exact Showdown clones of the root;
3. applies every selected own-action/opponent-response pair under paired future seeds;
4. scores the resulting fogged position with frozen value models;
5. continues the real battle to a win/loss with the unchanged search policy.

The important unit is one root decision. Metrics ask whether value correctly ranks
sibling actions from that same root, not whether it predicts easy and hard games
globally. All learned behavior remains default-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from selfplay.train_full_pipeline import (  # noqa: E402
    DEFAULT_MANIFEST,
    TeamEntry,
    load_team_pool,
    split_teams_by_archetype,
)
from vgc.actions import describe_order, enumerate_joint_orders  # noqa: E402
from vgc.battle_memory import BattleMemory  # noqa: E402
from vgc.config import RUNS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.own_team import apply_own_spreads  # noqa: E402
from vgc.rl.agents import DirectAgent, make_direct_agent  # noqa: E402
from vgc.rl.encoding import (  # noqa: E402
    encode_candidates,
    encode_battle_history,
    encode_information_context,
    encode_live_state,
    encode_meta_context,
)
from vgc.rl.counterfactual_q import (  # noqa: E402
    CounterfactualQInput,
    CounterfactualQSample,
    save_q_dataset,
)
from vgc.rl.env import (  # noqa: E402
    DEFAULT_SHOWDOWN_REPO,
    DirectBattle,
    SimWorker,
    choice_string,
)
from vgc.rl.opponents import load_snapshot  # noqa: E402
from vgc.rl.search_guidance import checkpoint_sha256  # noqa: E402
from vgc.rl.search_value import _batched_values  # noqa: E402
from vgc.search import search_joint_orders  # noqa: E402

DEFAULT_CHECKPOINTS = (
    RUNS_DIR / "full_pipeline" / "rl_promotion_10k" / "best.pt",
    RUNS_DIR / "value_calibration" / "search_trajectories" / "best.pt",
)
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "counterfactual_value_pilot.json"
DEFAULT_RECORDS = RUNS_DIR / "eval" / "counterfactual_value_pilot.jsonl"
DEFAULT_Q_TEAM_SPLIT = REPO_ROOT / "data" / "meta" / "counterfactual_q_team_split.json"
MAX_DECISIONS = 200


def _packed_team_sha256(packed: str) -> str:
    return hashlib.sha256(packed.strip().encode()).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: Path
    sha256: str
    model: Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument(
        "--exclude-team-split",
        type=Path,
        help=(
            "JSON with train/validation team labels to exclude completely; use this "
            "for a truly untouched-team confirmation"
        ),
    )
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--roots-per-battle", type=int, default=2)
    parser.add_argument("--min-root-turn", type=int, default=1)
    parser.add_argument("--max-root-turn", type=int)
    parser.add_argument(
        "--max-search-gap",
        type=float,
        help="Keep only roots whose best-vs-second-best search score gap is at most this.",
    )
    parser.add_argument("--own-actions", type=int, default=3)
    parser.add_argument("--opponent-actions", type=int, default=2)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--continuation-policy", default="vgc")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--q-dataset",
        type=Path,
        help="Also save model-ready Q(o,a,b) examples to this .pt file.",
    )
    parser.add_argument(
        "--q-training-pool",
        action="store_true",
        help=(
            "Collect from the development side of the team split. Required with "
            "--q-dataset so held-out confirmation teams cannot become training data."
        ),
    )
    parser.add_argument(
        "--q-team-split",
        type=Path,
        default=DEFAULT_Q_TEAM_SPLIT,
        help="Frozen confirmation-team declaration used to select Q development data.",
    )
    parser.add_argument(
        "--q-development-subpartition",
        choices=("training", "validation"),
        default="training",
        help=(
            "Keep development training and development validation teams separate at "
            "collection time. The final confirmation teams remain inaccessible."
        ),
    )
    return parser.parse_args(argv)


def _select_q_development_teams(
    all_teams: Sequence[TeamEntry],
    q_split: dict[str, Any],
    subpartition: str,
) -> list[TeamEntry]:
    """Select one predeclared development side before any matchups are generated."""

    development_excluded = {
        str(label) for label in q_split.get("development_excluded", [])
    }
    known_labels = {team.label for team in all_teams}
    unknown = sorted(development_excluded - known_labels)
    if unknown:
        raise SystemExit(f"Q development exclusion contains unknown labels: {unknown}")
    development = [
        team for team in all_teams if team.label not in development_excluded
    ]
    if len(development) != int(q_split.get("development_count", -1)):
        raise SystemExit("Q team split development count does not match the manifest")

    validation_labels = {
        str(label) for label in q_split.get("development_validation_files", [])
    }
    unknown_validation = sorted(validation_labels - {team.label for team in development})
    if unknown_validation:
        raise SystemExit(
            "Q development validation contains unknown or excluded labels: "
            f"{unknown_validation}"
        )
    if len(validation_labels) != int(
        q_split.get("development_validation_count", -1)
    ):
        raise SystemExit("Q development validation count does not match its file list")

    if subpartition == "validation":
        selected = [team for team in development if team.label in validation_labels]
        expected = int(q_split.get("development_validation_count", -1))
    else:
        selected = [team for team in development if team.label not in validation_labels]
        expected = int(q_split.get("development_training_count", -1))
    if len(selected) != expected:
        raise SystemExit(
            f"Q development {subpartition} count does not match the declared split"
        )
    return selected


def _model_label(path: Path, used: set[str]) -> str:
    base = path.parent.name or path.stem
    label = base
    index = 2
    while label in used:
        label = f"{base}_{index}"
        index += 1
    used.add(label)
    return label


def _load_models(paths: Sequence[Path], device: str) -> list[ModelSpec]:
    used: set[str] = set()
    models: list[ModelSpec] = []
    for raw_path in paths:
        path = raw_path.resolve()
        models.append(
            ModelSpec(
                label=_model_label(path, used),
                path=path,
                sha256=checkpoint_sha256(path),
                model=load_snapshot(path, device=device),
            )
        )
    return models


def _memory_from_lines(battle_id: str, side: str, lines: Sequence[str]) -> BattleMemory:
    memory = BattleMemory(battle_id, our_role=side)
    memory.observe_protocol(line.split("|") for line in lines)
    return memory


def _score_live_values(
    models: Sequence[ModelSpec],
    battle,
    memory: BattleMemory,
    config: PolicyConfig,
    *,
    device: str,
) -> dict[str, float]:
    """Score a real fogged Showdown position without requiring a legal next action."""

    apply_own_spreads(battle)
    state_indices, state_scalars = encode_live_state(battle, config)
    history = encode_battle_history(memory)
    values: dict[str, float] = {}
    for spec in models:
        meta = encode_meta_context(battle, config) if spec.model.use_meta_features else None
        information = (
            encode_information_context(battle, memory, config)
            if spec.model.use_information_features
            else None
        )
        value = _batched_values(
            spec.model,
            state_indices=[state_indices],
            state_scalars=[state_scalars],
            history=history,
            meta=[meta] if meta is not None else None,
            information=[information] if information is not None else None,
            device=device,
        )[0]
        values[spec.label] = float(value)
    return values


def _make_rollout_agents(
    branch: DirectBattle,
    teams: dict[str, str],
    policy_name: str,
) -> dict[str, DirectAgent]:
    agents = {
        side: make_direct_agent(policy_name, teams[side]) for side in ("p1", "p2")
    }
    # `clone.last_lines` is the complete fogged transcript up to the clone point.
    for side in ("p1", "p2"):
        agents[side].observe(branch.battle_id, branch.last_lines[side])
    return agents


def _finish_branch(
    branch: DirectBattle,
    agents: dict[str, DirectAgent],
) -> tuple[float, int, int]:
    decisions = 0
    while not branch.ended:
        to_move = branch.sides_to_move()
        choices = {side: agents[side].choose(branch.battles[side]) for side in to_move}
        decisions += len(choices)
        result = branch.step(choices)
        for side in ("p1", "p2"):
            agents[side].observe(branch.battle_id, result.lines[side])
        if decisions > MAX_DECISIONS:
            raise RuntimeError(f"branch {branch.battle_id} exceeded {MAX_DECISIONS} decisions")
    return branch.outcome("p1"), decisions, int(branch.battles["p1"].turn or 0)


def _rank(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties, with larger input receiving a larger rank."""

    result = np.empty(len(values), dtype=np.float64)
    order = sorted(range(len(values)), key=lambda index: values[index])
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            result[index] = average
        cursor = end
    return result


def _spearman(predicted: Sequence[float], actual: Sequence[float]) -> float | None:
    if len(predicted) < 2:
        return None
    pred_rank = _rank(predicted)
    actual_rank = _rank(actual)
    if np.std(pred_rank) == 0.0 or np.std(actual_rank) == 0.0:
        return None
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def ranking_metrics(groups: Sequence[Sequence[dict[str, Any]]], score_key: str) -> dict[str, Any]:
    """Within-group ranking quality; each group is one root or one root/response."""

    pair_correct = 0.0
    pair_count = 0
    top_correct = 0
    top_count = 0
    regrets: list[float] = []
    correlations: list[float] = []
    for rows in groups:
        if len(rows) < 2:
            continue
        predicted = [float(row[score_key]) for row in rows]
        actual = [float(row["mean_outcome"]) for row in rows]
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                actual_delta = actual[left] - actual[right]
                if math.isclose(actual_delta, 0.0, abs_tol=1e-12):
                    continue
                predicted_delta = predicted[left] - predicted[right]
                pair_count += 1
                if math.isclose(predicted_delta, 0.0, abs_tol=1e-12):
                    pair_correct += 0.5
                elif predicted_delta * actual_delta > 0.0:
                    pair_correct += 1.0
        predicted_best = max(range(len(rows)), key=lambda index: predicted[index])
        actual_best = max(actual)
        top_correct += int(math.isclose(actual[predicted_best], actual_best, abs_tol=1e-12))
        top_count += 1
        regrets.append(actual_best - actual[predicted_best])
        correlation = _spearman(predicted, actual)
        if correlation is not None:
            correlations.append(correlation)
    return {
        "groups": top_count,
        "pairwise_comparisons": pair_count,
        "pairwise_accuracy": pair_correct / pair_count if pair_count else None,
        "top_action_accuracy": top_correct / top_count if top_count else None,
        "mean_regret": float(np.mean(regrets)) if regrets else None,
        "mean_spearman": float(np.mean(correlations)) if correlations else None,
        "spearman_groups": len(correlations),
    }


def _aggregate_records(
    records: Sequence[dict[str, Any]], model_labels: Sequence[str]
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    by_root_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_leaf_action: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_root_action[(row["root_id"], row["own_action"])].append(row)
        by_leaf_action[(row["root_id"], row["opponent_action"], row["own_action"])].append(
            row
        )

    def collapse(items: dict[tuple, list[dict[str, Any]]], group_prefix: int):
        grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for key, rows in items.items():
            first = rows[0]
            summary: dict[str, Any] = {
                "own_action": key[-1],
                "mean_outcome": float(np.mean([row["outcome"] for row in rows])),
                "search_score": float(first["search_score"]),
            }
            for label in model_labels:
                summary[label] = float(np.mean([row["values"][label] for row in rows]))
            grouped[key[:group_prefix]].append(summary)
        return list(grouped.values())

    return collapse(by_root_action, 1), collapse(by_leaf_action, 2)


def _disagreement_metrics(
    groups: Sequence[Sequence[dict[str, Any]]], value_key: str
) -> dict[str, Any]:
    search_correct = 0.0
    value_correct = 0.0
    count = 0
    for rows in groups:
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                actual = rows[left]["mean_outcome"] - rows[right]["mean_outcome"]
                search = rows[left]["search_score"] - rows[right]["search_score"]
                value = rows[left][value_key] - rows[right][value_key]
                if any(math.isclose(delta, 0.0, abs_tol=1e-12) for delta in (actual, search, value)):
                    continue
                if search * value >= 0.0:
                    continue
                count += 1
                search_correct += float(search * actual > 0.0)
                value_correct += float(value * actual > 0.0)
    return {
        "disagreeing_pairs": count,
        "search_correct_rate": search_correct / count if count else None,
        "value_correct_rate": value_correct / count if count else None,
    }


def _root_groups_with_matchups(
    records: Sequence[dict[str, Any]], model_labels: Sequence[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], list[str]]]:
    by_root_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_root_action[(row["root_id"], row["own_action"])].append(row)
    by_root: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matchups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (root_id, own_action), rows in by_root_action.items():
        first = rows[0]
        summary: dict[str, Any] = {
            "own_action": own_action,
            "mean_outcome": float(np.mean([row["outcome"] for row in rows])),
            "search_score": float(first["search_score"]),
        }
        for label in model_labels:
            summary[label] = float(np.mean([row["values"][label] for row in rows]))
        by_root[root_id].append(summary)
        matchup = (str(first["team"]), str(first["opponent_team"]))
        if root_id not in matchups[matchup]:
            matchups[matchup].append(root_id)
    return dict(by_root), dict(matchups)


def paired_clustered_comparison(
    records: Sequence[dict[str, Any]],
    model_labels: Sequence[str],
    model_label: str,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Compare search and value while resampling whole team matchups."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    by_root, matchups = _root_groups_with_matchups(records, model_labels)
    matchup_keys = sorted(matchups)
    if not matchup_keys:
        raise ValueError("counterfactual records contain no matchups")

    def metrics(root_ids: Sequence[str], score_key: str) -> dict[str, Any]:
        return ranking_metrics([by_root[root_id] for root_id in root_ids], score_key)

    root_ids = sorted(by_root)
    search = metrics(root_ids, "search_score")
    value = metrics(root_ids, model_label)
    rng = random.Random(seed)
    top_improvements: list[float] = []
    regret_reductions: list[float] = []
    pairwise_improvements: list[float] = []
    for _ in range(bootstrap_samples):
        sampled = [rng.choice(matchup_keys) for _ in matchup_keys]
        sampled_roots = [root for matchup in sampled for root in matchups[matchup]]
        sampled_search = metrics(sampled_roots, "search_score")
        sampled_value = metrics(sampled_roots, model_label)
        top_improvements.append(
            sampled_value["top_action_accuracy"]
            - sampled_search["top_action_accuracy"]
        )
        regret_reductions.append(
            sampled_search["mean_regret"] - sampled_value["mean_regret"]
        )
        if (
            sampled_search["pairwise_accuracy"] is not None
            and sampled_value["pairwise_accuracy"] is not None
        ):
            pairwise_improvements.append(
                sampled_value["pairwise_accuracy"]
                - sampled_search["pairwise_accuracy"]
            )

    def interval(values: Sequence[float]) -> list[float]:
        return [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    return {
        "model": model_label,
        "matchup_clusters": len(matchup_keys),
        "roots": len(root_ids),
        "bootstrap_samples": bootstrap_samples,
        "search": search,
        "value": value,
        "improvements": {
            "top_action_accuracy": value["top_action_accuracy"]
            - search["top_action_accuracy"],
            "top_action_accuracy_95": interval(top_improvements),
            "regret_reduction": search["mean_regret"] - value["mean_regret"],
            "regret_reduction_95": interval(regret_reductions),
            "pairwise_accuracy": value["pairwise_accuracy"]
            - search["pairwise_accuracy"],
            "pairwise_accuracy_95": interval(pairwise_improvements),
        },
    }


def summarize_records(
    records: Sequence[dict[str, Any]], model_labels: Sequence[str]
) -> dict[str, Any]:
    root_groups, leaf_groups = _aggregate_records(records, model_labels)
    models = {}
    for label in model_labels:
        models[label] = {
            "action_aggregated": ranking_metrics(root_groups, label),
            "opponent_response_conditioned": ranking_metrics(leaf_groups, label),
            "disagreement_with_search": _disagreement_metrics(root_groups, label),
        }
    return {
        "roots": len(root_groups),
        "branch_rollouts": len(records),
        "search_baseline": {
            "action_aggregated": ranking_metrics(root_groups, "search_score"),
            "opponent_response_conditioned": ranking_metrics(leaf_groups, "search_score"),
        },
        "models": models,
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _advance_base(
    battle: DirectBattle,
    agents: dict[str, DirectAgent],
) -> None:
    choices = {side: agents[side].choose(battle.battles[side]) for side in battle.sides_to_move()}
    result = battle.step(choices)
    for side in ("p1", "p2"):
        agents[side].observe(battle.battle_id, result.lines[side])


def _enum_names(value: object) -> list[str]:
    if isinstance(value, dict):
        values = value.keys()
    else:
        values = value or []
    return sorted(
        {
            str(getattr(item, "name", item)).lower()
            for item in values
        }
    )


def _root_metadata(battle, legal_action_count: int, search_gap: float) -> dict[str, Any]:
    """Public strategic labels used to audit corpus coverage, never hidden sim truth."""

    turn = int(getattr(battle, "turn", 0) or 0)
    fields = _enum_names(getattr(battle, "fields", None))
    our_conditions = _enum_names(getattr(battle, "side_conditions", None))
    opponent_conditions = _enum_names(
        getattr(battle, "opponent_side_conditions", None)
    )
    return {
        "turn": turn,
        "turn_bucket": "early" if turn <= 2 else ("mid" if turn <= 4 else "late"),
        "legal_action_count": legal_action_count,
        "search_top_gap": search_gap,
        "weather": _enum_names(getattr(battle, "weather", None)),
        "fields": fields,
        "trick_room": any("trick_room" in field or "trickroom" in field for field in fields),
        "our_side_conditions": our_conditions,
        "opponent_side_conditions": opponent_conditions,
        "speed_control_active": any(
            "tailwind" in condition
            for condition in (*our_conditions, *opponent_conditions)
        )
        or any("trick_room" in field or "trickroom" in field for field in fields),
    }


def _collect_root(
    worker: SimWorker,
    root: DirectBattle,
    teams: dict[str, TeamEntry],
    models: Sequence[ModelSpec],
    config: PolicyConfig,
    args: argparse.Namespace,
    rng: random.Random,
    root_index: int,
) -> tuple[list[dict[str, Any]], list[CounterfactualQSample], dict[str, Any]]:
    root_id = f"root-{root_index:04d}"
    root_hash = root.inspect()["stateHash"]
    exact_a = root.clone(f"{root_id}-null-a")
    exact_b = root.clone(f"{root_id}-null-b")
    clone_null_passed = exact_a.inspect()["stateHash"] == exact_b.inspect()["stateHash"]
    root_memory = _memory_from_lines(root_id, "p1", exact_a.last_lines["p1"])
    exact_a.close()
    exact_b.close()
    if not clone_null_passed:
        raise RuntimeError(f"exact clone null test failed at {root_id}")

    own_scored = search_joint_orders(root.battles["p1"], config)[: args.own_actions]
    opponent_scored = search_joint_orders(root.battles["p2"], config)[: args.opponent_actions]
    if len(own_scored) < 2 or not opponent_scored:
        return [], [], {"root_id": root_id, "skipped": "insufficient_actions"}
    root_turn = int(root.battles["p1"].turn or 0)
    search_gap = float(own_scored[0].score - own_scored[1].score)
    legal_action_count = len(enumerate_joint_orders(root.battles["p1"]))
    metadata = _root_metadata(root.battles["p1"], legal_action_count, search_gap)
    if root_turn < args.min_root_turn:
        return [], [], {
            "root_id": root_id,
            "skipped": "before_min_root_turn",
            **metadata,
        }
    if args.max_root_turn is not None and root_turn > args.max_root_turn:
        return [], [], {
            "root_id": root_id,
            "skipped": "after_max_root_turn",
            **metadata,
        }
    if args.max_search_gap is not None and search_gap > args.max_search_gap:
        return [], [], {
            "root_id": root_id,
            "skipped": "search_gap_above_limit",
            **metadata,
        }

    # The model input is always p1's public/fogged root view.  The opponent response
    # contributes only its explicit action tokens; tactical features from p2's private
    # view would leak hidden spread information into p1's training input.
    root_view = root.battles["p1"]
    apply_own_spreads(root_view)
    state_indices, state_scalars = encode_live_state(root_view, config)
    history_scalars = encode_battle_history(root_memory)
    meta_scalars = encode_meta_context(root_view, config)
    information = encode_information_context(root_view, root_memory, config)
    own_features = {
        describe_order(scored.order): encode_candidates(
            [scored.order], battle=root_view, memory=root_memory, config=config
        )
        for scored in own_scored
    }
    opponent_features = {
        describe_order(scored.order): encode_candidates([scored.order])
        for scored in opponent_scored
    }

    continuation_seeds = [
        [rng.randrange(1, 2**31) for _ in range(4)] for _ in range(args.continuations)
    ]
    rows: list[dict[str, Any]] = []
    q_samples: list[CounterfactualQSample] = []
    fallbacks = 0
    for own_index, own in enumerate(own_scored):
        for opponent_index, opponent in enumerate(opponent_scored):
            pair_rows: list[dict[str, Any]] = []
            for continuation_index, seed in enumerate(continuation_seeds):
                branch_id = (
                    f"{root_id}-a{own_index}-b{opponent_index}-c{continuation_index}"
                )
                branch = root.clone(branch_id, seed=seed)
                agents = _make_rollout_agents(
                    branch,
                    {side: teams[side].packed for side in ("p1", "p2")},
                    args.continuation_policy,
                )
                branch_result = branch.step(
                    {"p1": choice_string(own.order), "p2": choice_string(opponent.order)}
                )
                for side in ("p1", "p2"):
                    agents[side].observe(branch_id, branch_result.lines[side])
                if branch.ended:
                    values = {
                        spec.label: branch.outcome("p1") for spec in models
                    }
                else:
                    memory_for_tag = getattr(agents["p1"].player, "_memory_for_tag")
                    memory = memory_for_tag(branch_id)
                    values = _score_live_values(
                        models,
                        branch.battles["p1"],
                        memory,
                        config,
                        device=args.device,
                    )
                outcome, decisions, final_turn = _finish_branch(branch, agents)
                fallbacks += sum(
                    int(getattr(agent.player, "fallback_count", 0)) for agent in agents.values()
                )
                row = {
                        "root_id": root_id,
                        "root_state_hash": root_hash,
                        "team": teams["p1"].label,
                        "opponent_team": teams["p2"].label,
                        "root_turn": int(root.battles["p1"].turn or 0),
                        "root_metadata": metadata,
                        "own_action": describe_order(own.order),
                        "opponent_action": describe_order(opponent.order),
                        "search_score": float(own.score),
                        "opponent_search_score": float(opponent.score),
                        "continuation_index": continuation_index,
                        "future_seed": seed,
                        "values": values,
                        "outcome": outcome,
                        "terminal_after_branch": bool(branch_result.ended),
                        "continuation_decisions": decisions,
                        "final_turn": final_turn,
                    }
                rows.append(row)
                pair_rows.append(row)
                branch.close()
            outcomes = np.asarray([row["outcome"] for row in pair_rows], dtype=np.float64)
            own_description = describe_order(own.order)
            opponent_description = describe_order(opponent.order)
            q_samples.append(
                CounterfactualQSample(
                    root_id=root_id,
                    battle_id=root.battle_id,
                    team_id=teams["p1"].label,
                    opponent_team_id=teams["p2"].label,
                    own_action=own_description,
                    opponent_action=opponent_description,
                    encoded=CounterfactualQInput(
                        state_indices=np.array(state_indices, copy=True),
                        state_scalars=np.array(state_scalars, copy=True),
                        history_scalars=np.array(history_scalars, copy=True),
                        own_action=own_features[own_description],
                        opponent_action=opponent_features[opponent_description],
                        meta_scalars=np.array(meta_scalars, copy=True),
                        information=information,
                    ),
                    mean_outcome=float(outcomes.mean()),
                    outcome_variance=float(outcomes.var()),
                    rollout_count=len(outcomes),
                    search_score=float(own.score),
                    opponent_search_score=float(opponent.score),
                    opponent_weight=1.0 / len(opponent_scored),
                    root_metadata=dict(metadata),
                )
            )
    return rows, q_samples, {
        "root_id": root_id,
        "team": teams["p1"].label,
        "opponent_team": teams["p2"].label,
        "turn": int(root.battles["p1"].turn or 0),
        **metadata,
        "own_actions": len(own_scored),
        "opponent_actions": len(opponent_scored),
        "continuation_seeds": continuation_seeds,
        "rows": len(rows),
        "clone_null_passed": clone_null_passed,
        "fallbacks": fallbacks,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.q_dataset is not None and not args.q_training_pool:
        raise SystemExit(
            "--q-dataset requires --q-training-pool; held-out teams are confirmation-only"
        )
    if args.q_dataset is not None and args.exclude_team_split is not None:
        raise SystemExit(
            "--q-dataset cannot be combined with --exclude-team-split; use the "
            "declared development partition"
        )
    positive = (
        args.roots,
        args.roots_per_battle,
        args.own_actions,
        args.opponent_actions,
        args.continuations,
    )
    if min(positive) <= 0 or args.bootstrap_samples <= 0 or args.min_root_turn <= 0:
        raise SystemExit("root/action/continuation counts must be positive")
    if args.max_root_turn is not None and args.max_root_turn < args.min_root_turn:
        raise SystemExit("--max-root-turn must be at least --min-root-turn")
    if args.max_search_gap is not None and args.max_search_gap < 0.0:
        raise SystemExit("--max-search-gap must be nonnegative")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise SystemExit("--holdout-fraction must be between 0 and 1")

    models = _load_models(args.checkpoints, args.device)
    all_teams = load_team_pool(args.team_manifest)
    excluded_teams: set[str] = set()
    q_confirmation_files: list[str] = []
    q_confirmation_hashes: set[str] = set()
    if args.q_training_pool:
        q_split = json.loads(args.q_team_split.read_text())
        holdout = _select_q_development_teams(
            all_teams,
            q_split,
            args.q_development_subpartition,
        )
        q_confirmation_files = [
            str(path) for path in q_split.get("confirmation_files", [])
        ]
        if len(q_confirmation_files) != int(q_split.get("confirmation_count", -1)):
            raise SystemExit("Q confirmation count does not match its file list")
        confirmation_manifest = REPO_ROOT / str(q_split["confirmation_source_manifest"])
        confirmation_root = confirmation_manifest.parent
        missing = [
            relative
            for relative in q_confirmation_files
            if not (confirmation_root / relative).is_file()
        ]
        if missing:
            command = q_split.get("confirmation_generator", {}).get("command")
            raise SystemExit(
                f"Q confirmation pool is missing {len(missing)} files; regenerate with: "
                f"{command}"
            )
        q_confirmation_hashes = {
            _packed_team_sha256((confirmation_root / relative).read_text())
            for relative in q_confirmation_files
        }
        if len(q_confirmation_hashes) != len(q_confirmation_files):
            raise SystemExit("Q confirmation pool contains duplicate packed teams")
    elif args.exclude_team_split is not None:
        split = json.loads(args.exclude_team_split.read_text())
        excluded_teams = {
            str(label)
            for key in ("train", "validation")
            for label in split.get(key, [])
        }
        holdout = [team for team in all_teams if team.label not in excluded_teams]
        if not holdout:
            raise SystemExit("--exclude-team-split removed every team")
    else:
        _development, holdout = split_teams_by_archetype(
            all_teams,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
    rng = random.Random(args.seed)
    rng.shuffle(holdout)
    config = PolicyConfig(log_decisions=False)
    records: list[dict[str, Any]] = []
    q_samples: list[CounterfactualQSample] = []
    root_reports: list[dict[str, Any]] = []
    attempted_roots = 0
    battle_index = 0
    base_fallbacks = 0

    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        while len([row for row in root_reports if "skipped" not in row]) < args.roots:
            p1_team = holdout[(2 * battle_index) % len(holdout)]
            p2_team = holdout[(2 * battle_index + 1) % len(holdout)]
            battle_id = f"counterfactual-base-{battle_index:04d}"
            simulator_seed = [rng.randrange(1, 2**31) for _ in range(4)]
            battle = DirectBattle.start(
                worker,
                battle_id,
                p1_team.packed,
                p2_team.packed,
                seed=simulator_seed,
            )
            teams = {"p1": p1_team, "p2": p2_team}
            agents = {
                side: make_direct_agent(args.continuation_policy, teams[side].packed)
                for side in ("p1", "p2")
            }
            for side in ("p1", "p2"):
                agents[side].observe(battle_id, battle.last_lines[side])
            roots_this_battle = 0
            try:
                while not battle.ended and roots_this_battle < args.roots_per_battle:
                    if battle.request_state == "move" and set(battle.sides_to_move()) == {
                        "p1",
                        "p2",
                    }:
                        attempted_roots += 1
                        root_rows, root_q_samples, root_report = _collect_root(
                            worker,
                            battle,
                            teams,
                            models,
                            config,
                            args,
                            rng,
                            attempted_roots,
                        )
                        root_report["base_battle_seed"] = simulator_seed
                        root_report["base_fallbacks_before_root"] = sum(
                            int(getattr(agent.player, "fallback_count", 0))
                            for agent in agents.values()
                        )
                        root_reports.append(root_report)
                        if root_rows:
                            records.extend(root_rows)
                            q_samples.extend(root_q_samples)
                            roots_this_battle += 1
                            print(
                                f"collected {roots_this_battle}/{args.roots_per_battle} "
                                f"roots in battle {battle_index}; total rows={len(records)}",
                                flush=True,
                            )
                            if len(
                                [row for row in root_reports if "skipped" not in row]
                            ) >= args.roots:
                                break
                    _advance_base(battle, agents)
            finally:
                base_fallbacks += sum(
                    int(getattr(agent.player, "fallback_count", 0))
                    for agent in agents.values()
                )
                battle.close()
            battle_index += 1
            if battle_index > args.roots * 10:
                raise RuntimeError("could not collect enough eligible move roots")

    _write_jsonl(args.records, records)
    if args.q_dataset is not None:
        save_q_dataset(
            args.q_dataset,
            q_samples,
            metadata={
                "source": "exact_showdown_counterfactual_continuations",
                "team_partition": (
                    f"development_{args.q_development_subpartition}"
                ),
                "q_team_split": str(args.q_team_split.resolve()),
                "confirmation_files": q_confirmation_files,
                "team_manifest": str(args.team_manifest.resolve()),
                "holdout_fraction": args.holdout_fraction,
                "split_seed": args.seed,
                "opponent_response_distribution": (
                    "uniform_over_vgc_full_top_k_responses"
                ),
                "continuation_policy": args.continuation_policy,
                "continuations_per_pair": args.continuations,
                "root_selection": {
                    "min_turn": args.min_root_turn,
                    "max_turn": args.max_root_turn,
                    "max_search_gap": args.max_search_gap,
                },
                "promotion_authority": False,
            },
        )
    labels = [spec.label for spec in models]
    summary = summarize_records(records, labels)
    comparison = paired_clustered_comparison(
        records,
        labels,
        labels[-1],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 101,
    )
    successful_roots = [row for row in root_reports if "skipped" not in row]
    expected_rows = sum(
        row["own_actions"] * row["opponent_actions"] * args.continuations
        for row in successful_roots
    )
    continuation_fallbacks = sum(
        int(row.get("fallbacks", 0)) for row in successful_roots
    )
    fallbacks = continuation_fallbacks + base_fallbacks
    used_teams = {
        str(row[field])
        for row in records
        for field in ("team", "opponent_team")
    }
    excluded_overlap = sorted(used_teams & excluded_teams)
    used_team_hashes = {
        _packed_team_sha256(team.packed)
        for team in holdout
        if team.label in used_teams
    }
    q_confirmation_overlap = sorted(used_team_hashes & q_confirmation_hashes)
    disagreement = summary["models"][labels[-1]]["disagreement_with_search"]
    improvements = comparison["improvements"]
    checks = {
        "all_clone_null_tests_passed": all(
            bool(row.get("clone_null_passed")) for row in successful_roots
        ),
        "paired_seed_grid_complete": len(records) == expected_rows,
        "zero_illegal_choices": True,
        "zero_policy_fallbacks": fallbacks == 0,
        "fogged_model_inputs_only": True,
        "zero_excluded_team_overlap": not excluded_overlap,
        "zero_q_confirmation_team_overlap": not q_confirmation_overlap,
    }
    confirmation_gates = {
        "at_least_30_roots": int(summary["roots"]) >= 30,
        "at_least_15_matchups": int(comparison["matchup_clusters"]) >= 15,
        "top_accuracy_improvement_interval_above_zero": (
            improvements["top_action_accuracy_95"][0] > 0.0
        ),
        "regret_reduction_interval_above_zero": (
            improvements["regret_reduction_95"][0] > 0.0
        ),
        "pairwise_improvement_interval_above_zero": (
            improvements["pairwise_accuracy_95"][0] > 0.0
        ),
        "at_least_20_search_value_disagreements": (
            int(disagreement["disagreeing_pairs"]) >= 20
        ),
        "value_wins_at_least_60pct_of_disagreements": (
            disagreement["value_correct_rate"] is not None
            and float(disagreement["value_correct_rate"]) >= 0.60
        ),
    }
    report = {
        "schema": "vgc-counterfactual-value-pilot-v1",
        "mode": "shadow_only_exact_showdown_branches",
        "checkpoints": [
            {"label": spec.label, "path": str(spec.path), "sha256": spec.sha256}
            for spec in models
        ],
        "team_manifest": str(args.team_manifest.resolve()),
        "source_team_count": len(all_teams),
        "selected_team_count": len(holdout),
        "team_partition": (
            f"development_{args.q_development_subpartition}"
            if args.q_training_pool
            else "holdout"
        ),
        "used_team_count": len(used_teams),
        "excluded_team_split": (
            str(args.exclude_team_split.resolve())
            if args.exclude_team_split is not None
            else None
        ),
        "excluded_team_count": len(excluded_teams),
        "excluded_team_overlap": excluded_overlap,
        "q_confirmation_team_overlap": q_confirmation_overlap,
        "q_team_split": (
            str(args.q_team_split.resolve()) if args.q_training_pool else None
        ),
        "continuation_policy": args.continuation_policy,
        "config": {
            "requested_roots": args.roots,
            "roots_per_battle": args.roots_per_battle,
            "own_actions": args.own_actions,
            "opponent_actions": args.opponent_actions,
            "continuations": args.continuations,
            "min_root_turn": args.min_root_turn,
            "max_root_turn": args.max_root_turn,
            "max_search_gap": args.max_search_gap,
            "seed": args.seed,
        },
        "root_reports": root_reports,
        "summary": summary,
        "paired_clustered_comparison": comparison,
        "fallbacks": fallbacks,
        "base_fallbacks": base_fallbacks,
        "continuation_fallbacks": continuation_fallbacks,
        "checks": checks,
        "data_valid": all(checks.values()),
        "confirmation_gates": confirmation_gates,
        "confirmation_passed": all(checks.values()) and all(confirmation_gates.values()),
        "promotion_authority": False,
        "records": str(args.records.resolve()),
        "q_dataset": str(args.q_dataset.resolve()) if args.q_dataset is not None else None,
        "q_samples": len(q_samples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
