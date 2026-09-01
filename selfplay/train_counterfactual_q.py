#!/usr/bin/env python
"""Train the default-off action-and-response value model on exact continuations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - command requires train extra
    raise SystemExit("run `uv sync --extra train` before Q training") from exc

from vgc.config import RUNS_DIR
from vgc.rl.counterfactual_q import (
    ActionResponseValueNet,
    QTrainingConfig,
    evaluate_q_model,
    load_q_dataset,
    split_q_samples_team_disjoint,
    train_q_model,
)
from vgc.rl.opponents import load_snapshot

DEFAULT_DATASET = RUNS_DIR / "counterfactual_q" / "development.pt"
DEFAULT_CHECKPOINT = RUNS_DIR / "full_pipeline" / "rl_promotion_10k" / "best.pt"
DEFAULT_OUT_DIR = RUNS_DIR / "counterfactual_q" / "model"
DEFAULT_Q_TEAM_SPLIT = (
    Path(__file__).resolve().parents[1] / "data" / "meta" / "counterfactual_q_team_split.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        default=[DEFAULT_DATASET],
        help="One or more development Q datasets; root IDs are namespaced before merging.",
    )
    parser.add_argument(
        "--dataset-strata",
        nargs="+",
        help="Optional label per --dataset, such as: early mid late.",
    )
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        nargs="+",
        help=(
            "Predeclared development-validation shards. When supplied, reserved teams "
            "are filtered from --dataset instead of making a split after matchmaking."
        ),
    )
    parser.add_argument(
        "--validation-dataset-strata",
        nargs="+",
        help="Optional label per --validation-dataset.",
    )
    parser.add_argument(
        "--q-team-split",
        type=Path,
        default=DEFAULT_Q_TEAM_SPLIT,
        help="Declaration of reserved development-validation and confirmation teams.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260819,
        help="Fixed team-split seed, separate from model initialization/training seed.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Also update the shared observation/action encoder; default trains only Q head.",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        help=(
            "Slower learning rate for the shared encoder under --train-backbone. "
            "Omit to move the encoder at the head's rate, which overfit in an early pilot."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.epochs, args.patience) <= 0:
        raise SystemExit("--epochs and --patience must be positive")
    if args.lr <= 0.0 or args.weight_decay < 0.0 or args.ranking_weight < 0.0:
        raise SystemExit("learning rate must be positive; weights must be nonnegative")
    if args.backbone_lr is not None:
        if args.backbone_lr < 0.0:
            raise SystemExit("--backbone-lr must be nonnegative")
        if not args.train_backbone:
            raise SystemExit("--backbone-lr requires --train-backbone")
    if args.dataset_strata is not None and len(args.dataset_strata) != len(args.dataset):
        raise SystemExit("--dataset-strata must provide exactly one label per --dataset")
    if args.validation_dataset_strata is not None and (
        args.validation_dataset is None
        or len(args.validation_dataset_strata) != len(args.validation_dataset)
    ):
        raise SystemExit(
            "--validation-dataset-strata must provide exactly one label per --validation-dataset"
        )
    torch.manual_seed(args.seed)
    samples = []
    dataset_metadata = []

    def load_shards(paths, strata, *, prefix_offset: int):
        loaded = []
        metadata_rows = []
        for local_index, dataset_path in enumerate(paths):
            shard_samples, shard_metadata = load_q_dataset(dataset_path)
            partition = str(shard_metadata.get("team_partition", ""))
            if not partition.startswith("development"):
                raise SystemExit(
                    f"refusing to train: {dataset_path} is not marked as development data"
                )
            prefix = f"shard-{prefix_offset + local_index:02d}"
            declared_stratum = strata[local_index] if strata is not None else None
            loaded.extend(
                replace(
                    sample,
                    root_id=f"{prefix}:{sample.root_id}",
                    battle_id=f"{prefix}:{sample.battle_id}",
                    root_metadata={
                        **getattr(sample, "root_metadata", {}),
                        **(
                            {"turn_bucket": declared_stratum}
                            if declared_stratum is not None
                            else {}
                        ),
                    },
                )
                for sample in shard_samples
            )
            metadata_rows.append(
                {
                    "path": str(dataset_path.resolve()),
                    "sha256": _sha256(dataset_path),
                    "samples": len(shard_samples),
                    "metadata": shard_metadata,
                }
            )
        return loaded, metadata_rows

    samples, dataset_metadata = load_shards(
        args.dataset,
        args.dataset_strata,
        prefix_offset=0,
    )
    explicit_validation = args.validation_dataset is not None
    if explicit_validation:
        q_split = json.loads(args.q_team_split.read_text())
        reserved = {str(label) for label in q_split.get("development_validation_files", [])}
        if len(reserved) != int(q_split.get("development_validation_count", -1)):
            raise SystemExit("development-validation team declaration is inconsistent")
        unfiltered_count = len(samples)
        train_samples = [
            sample
            for sample in samples
            if sample.team_id not in reserved and sample.opponent_team_id not in reserved
        ]
        if not train_samples:
            raise SystemExit("reserved-team filtering removed every training sample")
        validation_samples, validation_metadata = load_shards(
            args.validation_dataset,
            args.validation_dataset_strata,
            prefix_offset=len(args.dataset),
        )
        invalid_validation = sorted(
            {
                team
                for sample in validation_samples
                for team in (sample.team_id, sample.opponent_team_id)
                if team not in reserved
            }
        )
        if invalid_validation:
            raise SystemExit(
                f"explicit validation data contains non-reserved teams: {invalid_validation}"
            )
        for row in validation_metadata:
            if row["metadata"].get("team_partition") != "development_validation":
                raise SystemExit("explicit validation shards must be marked development_validation")
        dataset_metadata.extend(validation_metadata)
        filtered_training_samples = unfiltered_count - len(train_samples)
    else:
        train_samples, validation_samples = split_q_samples_team_disjoint(
            samples,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed,
        )
        filtered_training_samples = 0
    backbone = load_snapshot(args.checkpoint, device=args.device)
    if not backbone.use_mechanics_features:
        raise SystemExit(
            "counterfactual Q training requires a complete-mechanics backbone checkpoint"
        )
    model = ActionResponseValueNet(backbone).to(args.device)
    config = QTrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        ranking_weight=args.ranking_weight,
        patience=args.patience,
        seed=args.seed,
        train_backbone=args.train_backbone,
        backbone_learning_rate=args.backbone_lr,
    )
    before = {
        "train": evaluate_q_model(model, train_samples, batch_size=512, device=args.device),
        "validation": evaluate_q_model(
            model, validation_samples, batch_size=512, device=args.device
        ),
    }
    training = train_q_model(
        model,
        train_samples,
        validation_samples,
        config,
        device=args.device,
    )
    after = {
        "train": evaluate_q_model(model, train_samples, batch_size=512, device=args.device),
        "validation": evaluate_q_model(
            model, validation_samples, batch_size=512, device=args.device
        ),
    }

    def by_stratum(selected_samples):
        grouped = {}
        labels = sorted(
            {
                getattr(sample, "root_metadata", {}).get("turn_bucket", "unlabeled")
                for sample in selected_samples
            }
        )
        for label in labels:
            subset = [
                sample
                for sample in selected_samples
                if getattr(sample, "root_metadata", {}).get("turn_bucket", "unlabeled") == label
            ]
            grouped[label] = evaluate_q_model(model, subset, batch_size=512, device=args.device)
        return grouped

    after["validation_by_stratum"] = by_stratum(validation_samples)
    train_teams = sorted(
        {team for sample in train_samples for team in (sample.team_id, sample.opponent_team_id)}
    )
    validation_teams = sorted(
        {
            team
            for sample in validation_samples
            for team in (sample.team_id, sample.opponent_team_id)
        }
    )
    overlap = sorted(set(train_teams) & set(validation_teams))
    if overlap:
        raise RuntimeError(f"team-disjoint split leaked teams: {overlap}")
    report = {
        "schema": "vgc-counterfactual-q-training-v1",
        "datasets": dataset_metadata,
        "base_checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": _sha256(args.checkpoint),
        "samples": len(train_samples) + len(validation_samples),
        "roots": len({sample.root_id for sample in [*train_samples, *validation_samples]}),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_teams": train_teams,
        "validation_teams": validation_teams,
        "team_overlap": overlap,
        "config": asdict(config),
        "split_seed": args.split_seed,
        "validation_source": (
            "predeclared_development_validation_teams"
            if explicit_validation
            else "post_collection_connected_components"
        ),
        "filtered_reserved_training_samples": filtered_training_samples,
        "q_team_split": (str(args.q_team_split.resolve()) if explicit_validation else None),
        "before": before,
        "training": training,
        "after": after,
        "search_integration": False,
        "promotion_authority": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "format": "vgc-action-response-value-v1",
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(args.checkpoint.resolve()),
            "base_checkpoint_sha256": report["base_checkpoint_sha256"],
            "training_report": report,
            "promotion_authority": False,
        },
        args.out_dir / "best.pt",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from vgc.mechanics_gate import enforce_mechanics_gate_for_cli
    from vgc.battle_state_gate import enforce_battle_state_gate_for_cli

    enforce_mechanics_gate_for_cli("model training")
    enforce_battle_state_gate_for_cli("model training")
    raise SystemExit(main())
