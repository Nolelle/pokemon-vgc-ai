#!/usr/bin/env python
"""Oracle sweep: can weighting the two collected opponent responses buy anything?

Answers, on the existing counterfactual dataset and without touching gameplay:

1. How often does moving the response weight from 0% to 100% change which of our actions
   looks best?
2. How much of uniform weighting's regret could an oracle that knew ``P(b | o)`` remove?
3. What is the best achievable action accuracy under any single weighting?
4. Do practical rules (equal / opponent search score / softmax / worst case) beat 50/50?

Rules and temperatures are selected on the 350 development-training roots; the 120
team-separated development-validation roots are scored once with the selection already
fixed.  The protected confirmation teams are never loaded.

Every number is limited to the top TWO opponent responses the collector stored, and the
"oracle" rows are optimistic upper bounds computed from outcomes a live battle would not
have.  See ``vgc.rl.response_weighting``'s module docstring for the framing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vgc.config import RUNS_DIR
from vgc.rl.response_weighting import (
    RootTable,
    build_root_tables,
    equal_weight,
    evaluate_rule,
    evaluate_rule_over_truth_grid,
    group_by_stratum,
    headroom_summary,
    null_distribution,
    null_exceedance,
    search_proportional_weight,
    sensitivity_summary,
    softmax_weight_fn,
    sweep_fixed_weights,
    top_response_weight,
    weight_grid,
)

DEFAULT_Q_TEAM_SPLIT = (
    Path(__file__).resolve().parents[1] / "data" / "meta" / "counterfactual_q_team_split.json"
)
DEFAULT_OUT = RUNS_DIR / "counterfactual_q" / "response_weight_sweep" / "sweep.json"

# Softmax temperatures span "opponent always plays their top response" (small) through
# "search-score gaps carry no information" (large); the gaps themselves run to ~50.
DEFAULT_TEMPERATURES = (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset-strata", nargs="+")
    parser.add_argument("--validation-dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-dataset-strata", nargs="+")
    parser.add_argument("--q-team-split", type=Path, default=DEFAULT_Q_TEAM_SPLIT)
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional trained Q checkpoint; enables the model-side deployment rules.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--weight-steps",
        type=int,
        default=100,
        help="Even number of steps on the 0..1 weight grid, so 0.5 stays on it.",
    )
    parser.add_argument(
        "--truth-steps",
        type=int,
        default=20,
        help="Even step count for the coarser grid of true response distributions.",
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--null-seed", type=int, default=20260824)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _packed_team_sha256(packed: str) -> str:
    """Hash a packed team the same way the collector does, so the two agree."""

    return hashlib.sha256(packed.strip().encode()).hexdigest()


def _pool_path(split: dict[str, Any], manifest_key: str, relative: str) -> Path:
    root = (Path(__file__).resolve().parents[1] / str(split[manifest_key])).parent
    candidate = root / relative
    return candidate if candidate.is_file() else root / f"{relative}.txt"


def _confirmation_team_hashes(split: dict[str, Any]) -> set[str]:
    files = [str(label) for label in split.get("confirmation_files", [])]
    if len(files) != int(split.get("confirmation_count", -1)):
        raise SystemExit("confirmation team declaration is inconsistent")
    hashes: set[str] = set()
    for relative in files:
        path = _pool_path(split, "confirmation_source_manifest", relative)
        if not path.is_file():
            raise SystemExit(f"confirmation pool is missing {path}")
        hashes.add(_packed_team_sha256(path.read_text()))
    return hashes


def _leaked_confirmation_teams(
    teams: set[str], split: dict[str, Any], confirmation_hashes: set[str]
) -> list[str]:
    leaked: list[str] = []
    for team in sorted(teams):
        path = _pool_path(split, "development_source_manifest", team)
        if not path.is_file():
            raise SystemExit(f"cannot resolve development team {team!r} to {path}")
        if _packed_team_sha256(path.read_text()) in confirmation_hashes:
            leaked.append(team)
    return leaked


def _load_shards(
    paths: list[Path], strata: list[str] | None, *, prefix_offset: int, expected_partition: str
) -> tuple[list[Any], list[dict[str, Any]]]:
    from vgc.rl.counterfactual_q import load_q_dataset

    loaded: list[Any] = []
    metadata_rows: list[dict[str, Any]] = []
    for local_index, dataset_path in enumerate(paths):
        shard_samples, shard_metadata = load_q_dataset(dataset_path)
        partition = str(shard_metadata.get("team_partition", ""))
        if not partition.startswith(expected_partition):
            raise SystemExit(
                f"refusing to analyse {dataset_path}: partition {partition!r} is not "
                f"{expected_partition!r}"
            )
        prefix = f"shard-{prefix_offset + local_index:02d}"
        declared = strata[local_index] if strata is not None else None
        loaded.extend(
            replace(
                sample,
                root_id=f"{prefix}:{sample.root_id}",
                battle_id=f"{prefix}:{sample.battle_id}",
                root_metadata={
                    **getattr(sample, "root_metadata", {}),
                    **({"turn_bucket": declared} if declared is not None else {}),
                },
            )
            for sample in shard_samples
        )
        metadata_rows.append(
            {
                "path": str(dataset_path.resolve()),
                "sha256": _sha256(dataset_path),
                "samples": len(shard_samples),
                "team_partition": partition,
                "opponent_response_distribution": shard_metadata.get(
                    "opponent_response_distribution"
                ),
                "continuations_per_pair": shard_metadata.get("continuations_per_pair"),
            }
        )
    return loaded, metadata_rows


def _model_predictions(
    model_path: Path, samples: list[Any], tables: list[RootTable], *, device: str
) -> dict[str, np.ndarray]:
    """Predict Q(o, a, b) for every collected cell, laid out to match each RootTable."""

    import torch

    from vgc.rl.counterfactual_q import ActionResponseValueNet, predict_q
    from vgc.rl.opponents import load_snapshot

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "vgc-action-response-value-v1":
        raise SystemExit(f"unsupported Q checkpoint format {payload.get('format')!r}")
    backbone = load_snapshot(Path(payload["base_checkpoint"]), device=device)
    model = ActionResponseValueNet(backbone).to(device)
    model.load_state_dict(payload["model_state_dict"])
    values = predict_q(model, samples, batch_size=512, device=device)
    lookup = {
        (sample.root_id, sample.own_action, sample.opponent_action): float(values[index])
        for index, sample in enumerate(samples)
    }
    predictions: dict[str, np.ndarray] = {}
    for table in tables:
        predictions[table.root_id] = np.asarray(
            [
                [lookup[(table.root_id, action, response)] for response in table.responses]
                for action in table.own_actions
            ],
            dtype=np.float64,
        )
    return predictions


def _oracle_block(
    tables: list[RootTable], weights: np.ndarray, *, null_draws: int, null_seed: int
) -> dict[str, Any]:
    """Model-free sensitivity + headroom, with the sampling-noise null beside it."""

    sensitivity = sensitivity_summary(tables, weights)
    headroom = headroom_summary(tables, weights)
    null = null_distribution(tables, weights, draws=null_draws, seed=null_seed)
    return {
        "sensitivity": sensitivity,
        "headroom": headroom,
        "null_control": null,
        "versus_null": {
            "flip_fraction": null_exceedance(sensitivity["flip_fraction"], null["flip_fraction"]),
            "uniform_regret_mean_over_p": null_exceedance(
                headroom["uniform_regret_mean_over_p"], null["uniform_regret_mean_over_p"]
            ),
            "uniform_regret_worst_over_p": null_exceedance(
                headroom["uniform_regret_worst_over_p"], null["uniform_regret_worst_over_p"]
            ),
        },
        "by_stratum": {
            label: {
                "sensitivity": sensitivity_summary(subset, weights),
                "headroom": headroom_summary(subset, weights),
            }
            for label, subset in group_by_stratum(tables).items()
        },
    }


def _candidate_rules(temperatures: list[float]) -> list[tuple[str, Any]]:
    rules: list[tuple[str, Any]] = [
        ("equal", equal_weight),
        ("top_response_only", top_response_weight),
        ("search_proportional", search_proportional_weight),
        ("worst_case", equal_weight),
    ]
    rules.extend(
        (f"softmax_t{temperature:g}", softmax_weight_fn(temperature))
        for temperature in temperatures
    )
    return rules


def _rule_table(
    tables: list[RootTable],
    predictions: dict[str, np.ndarray],
    truth_weights: np.ndarray,
    temperatures: list[float],
) -> list[dict[str, Any]]:
    return [
        evaluate_rule_over_truth_grid(
            tables, predictions, truth_weights, rule=name, weight_fn=weight_fn
        )
        for name, weight_fn in _candidate_rules(temperatures)
    ]


def _resolve_rule(name: str, temperatures: list[float]) -> Any:
    for candidate, weight_fn in _candidate_rules(temperatures):
        if candidate == name:
            return weight_fn
    raise SystemExit(f"unknown rule {name!r}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.null_draws < 0:
        raise SystemExit("--null-draws must be nonnegative")
    weights = weight_grid(args.weight_steps)
    truth_weights = weight_grid(args.truth_steps)

    split = json.loads(args.q_team_split.read_text())
    reserved = {str(label) for label in split.get("development_validation_files", [])}
    if len(reserved) != int(split.get("development_validation_count", -1)):
        raise SystemExit("development-validation team declaration is inconsistent")
    confirmation_hashes = _confirmation_team_hashes(split)

    train_samples, train_metadata = _load_shards(
        args.dataset, args.dataset_strata, prefix_offset=0, expected_partition="development"
    )
    train_samples = [
        sample
        for sample in train_samples
        if sample.team_id not in reserved and sample.opponent_team_id not in reserved
    ]
    validation_samples, validation_metadata = _load_shards(
        args.validation_dataset,
        args.validation_dataset_strata,
        prefix_offset=len(args.dataset),
        expected_partition="development_validation",
    )
    stray = sorted(
        {
            team
            for sample in validation_samples
            for team in (sample.team_id, sample.opponent_team_id)
            if team not in reserved
        }
    )
    if stray:
        raise SystemExit(f"validation shards contain non-reserved teams: {stray}")

    train_tables, train_skipped = build_root_tables(train_samples)
    validation_tables, validation_skipped = build_root_tables(validation_samples)
    if not train_tables or not validation_tables:
        raise SystemExit("no usable roots after grouping")
    all_teams = {
        team
        for sample in [*train_samples, *validation_samples]
        for team in (sample.team_id, sample.opponent_team_id)
    }
    # The confirmation pool reuses the development pool's file NAMES (both come from
    # tools/build_archetype_pool.py), so overlap has to be checked on packed-team content
    # the way the collector does it, not on labels.
    leaked = sorted(_leaked_confirmation_teams(all_teams, split, confirmation_hashes))
    if leaked:
        raise SystemExit(f"confirmation teams leaked into the sweep: {leaked}")
    train_teams = {
        team for sample in train_samples for team in (sample.team_id, sample.opponent_team_id)
    }
    validation_teams = {
        team for sample in validation_samples for team in (sample.team_id, sample.opponent_team_id)
    }
    overlap = sorted(train_teams & validation_teams)
    if overlap:
        raise SystemExit(f"train and validation share teams: {overlap}")

    report: dict[str, Any] = {
        "schema": "vgc-response-weight-sweep-v1",
        "question": (
            "Does a better P(b|o) over the two collected opponent responses change which "
            "action we should pick?"
        ),
        "simultaneous_move_note": (
            "The opponent cannot condition on our sampled action, so the deployable "
            "quantity is Q(o,a) = sum_b P(b|o) Q(o,a,b), not P(b|o,a)."
        ),
        "scope_limit": "Top two search responses only; says nothing about responses outside them.",
        "datasets": {"training": train_metadata, "validation": validation_metadata},
        "weight_grid_steps": args.weight_steps,
        "truth_grid_steps": args.truth_steps,
        "roots": {"training": len(train_tables), "validation": len(validation_tables)},
        "skipped_roots": {"training": train_skipped, "validation": validation_skipped},
        "teams": {"training": len(train_teams), "validation": len(validation_teams)},
        "team_overlap": overlap,
        "confirmation_teams_touched": leaked,
        "search_integration": False,
        "promotion_authority": False,
    }

    report["oracle"] = {
        "training": _oracle_block(
            train_tables, weights, null_draws=args.null_draws, null_seed=args.null_seed
        ),
        "validation": _oracle_block(
            validation_tables, weights, null_draws=args.null_draws, null_seed=args.null_seed + 1
        ),
    }

    # A perfect Q model still has to aggregate the two responses, so run the rule table
    # on the true outcomes first: that isolates "the weighting is wrong" from "the model
    # is wrong".
    perfect_train = {table.root_id: table.outcomes for table in train_tables}
    perfect_validation = {table.root_id: table.outcomes for table in validation_tables}
    report["perfect_model_rules"] = {
        "note": (
            "Predictions are the exact collected outcomes, so any gap here is caused by "
            "the response weighting alone."
        ),
        "training": _rule_table(train_tables, perfect_train, truth_weights, args.temperatures),
        "validation": _rule_table(
            validation_tables, perfect_validation, truth_weights, args.temperatures
        ),
    }

    if args.model is not None:
        model_train = _model_predictions(
            args.model, train_samples, train_tables, device=args.device
        )
        model_validation = _model_predictions(
            args.model, validation_samples, validation_tables, device=args.device
        )
        training_rules = _rule_table(train_tables, model_train, truth_weights, args.temperatures)
        fixed_sweep = sweep_fixed_weights(train_tables, model_train, weights)
        selected = min(training_rules, key=lambda row: row["mean_regret_over_truth"])
        selected_fn = _resolve_rule(selected["rule"], args.temperatures)
        report["model_rules"] = {
            "checkpoint": str(args.model.resolve()),
            "checkpoint_sha256": _sha256(args.model),
            "training": training_rules,
            "training_fixed_weight_sweep_at_uniform_truth": fixed_sweep,
            "selection": {
                "criterion": "lowest mean regret over a uniform prior on the true weight",
                "selected_rule": selected["rule"],
                "training_mean_regret_over_truth": selected["mean_regret_over_truth"],
                "training_regret_at_uniform_truth": selected["regret_at_uniform_truth"],
            },
            "validation_reported_once": {
                "selected": evaluate_rule_over_truth_grid(
                    validation_tables,
                    model_validation,
                    truth_weights,
                    rule=selected["rule"],
                    weight_fn=selected_fn,
                ),
                "equal_baseline": evaluate_rule_over_truth_grid(
                    validation_tables,
                    model_validation,
                    truth_weights,
                    rule="equal",
                    weight_fn=equal_weight,
                ),
            },
            "validation_all_rules_for_transparency": _rule_table(
                validation_tables, model_validation, truth_weights, args.temperatures
            ),
            "validation_search_baseline_at_uniform_truth": evaluate_rule(
                validation_tables,
                {
                    table.root_id: np.repeat(table.search_scores[:, None], 2, axis=1)
                    for table in validation_tables
                },
                rule="search_score",
                weight_fn=equal_weight,
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _print_summary(report)
    print(f"\nwrote {args.out}")
    return 0


def _print_summary(report: dict[str, Any]) -> None:
    for partition in ("training", "validation"):
        block = report["oracle"][partition]
        sensitivity = block["sensitivity"]
        headroom = block["headroom"]
        versus = block["versus_null"]
        print(f"[{partition}] roots={sensitivity['roots']}")
        print(
            f"  best action unchanged across all weights: "
            f"{sensitivity['stable_fraction']:.1%} "
            f"(flips {sensitivity['flip_fraction']:.1%}, "
            f"null {versus['flip_fraction']['null_mean']:.1%})"
        )
        print(
            f"  oracle regret removable from uniform: "
            f"{headroom['uniform_regret_mean_over_p']:.4f} mean over p, "
            f"{headroom['uniform_regret_worst_over_p']:.4f} worst "
            f"(null mean {versus['uniform_regret_mean_over_p']['null_mean']:.4f}, "
            f"{versus['uniform_regret_mean_over_p']['null_standard_deviations']:+.1f} sd)"
        )
        print(
            f"  best single fixed weight: {headroom['best_fixed_weight']:.2f} "
            f"(mean regret {headroom['best_fixed_weight_regret_mean_over_p']:.4f} vs "
            f"uniform {headroom['uniform_regret_mean_over_p']:.4f})"
        )
    if "model_rules" in report:
        selection = report["model_rules"]["selection"]
        chosen = report["model_rules"]["validation_reported_once"]["selected"]
        baseline = report["model_rules"]["validation_reported_once"]["equal_baseline"]
        print(f"[model] selected on training: {selection['selected_rule']}")
        print(
            f"  validation mean regret over p: {chosen['mean_regret_over_truth']:.4f} "
            f"vs equal {baseline['mean_regret_over_truth']:.4f}"
        )
        print(
            f"  validation accuracy at uniform truth: "
            f"{chosen['accuracy_at_uniform_truth']:.1%} vs equal "
            f"{baseline['accuracy_at_uniform_truth']:.1%}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
