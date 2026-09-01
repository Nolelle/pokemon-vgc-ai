#!/usr/bin/env python
"""Calibrate only the RL value head on real full-search trajectory outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - command requires train extra
    raise SystemExit("run `uv sync --extra train` before value calibration") from exc

from vgc.config import RUNS_DIR
from vgc.rl.opponents import load_snapshot
from vgc.rl.value_calibration import (
    ValueCalibrationConfig,
    calibrate_value_head,
    evaluate_value_model,
    split_samples_by_team_archetype,
)

DEFAULT_DATASET = RUNS_DIR / "eval" / "neural_search_train_teams.pt"
DEFAULT_CHECKPOINT = RUNS_DIR / "full_pipeline" / "rl_promotion_10k" / "best.pt"
DEFAULT_OUT_DIR = RUNS_DIR / "value_calibration" / "search_trajectories"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_samples(path: Path) -> list:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "vgc-neural-search-guidance-v2-exact-mechanics":
        raise ValueError(f"unsupported search trajectory dataset: {payload.get('format')!r}")
    samples = list(payload.get("samples") or [])
    if not samples:
        raise ValueError("search trajectory dataset is empty")
    if any(sample.outcome is None for sample in samples):
        raise ValueError("every search trajectory sample must have a terminal outcome")
    return samples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.epochs, args.batch_size, args.patience) <= 0:
        raise SystemExit("--epochs, --batch-size, and --patience must be positive")
    if args.lr <= 0.0 or args.weight_decay < 0.0:
        raise SystemExit("--lr must be positive and --weight-decay nonnegative")
    samples = _load_samples(args.dataset)
    train_samples, validation_samples = split_samples_by_team_archetype(
        samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    model = load_snapshot(args.checkpoint, device=args.device)
    original_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    config = ValueCalibrationConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
    )
    before_train = evaluate_value_model(
        model, train_samples, batch_size=args.batch_size, device=args.device
    )
    before_validation = evaluate_value_model(
        model, validation_samples, batch_size=args.batch_size, device=args.device
    )
    training = calibrate_value_head(
        model,
        train_samples,
        validation_samples,
        config,
        device=args.device,
    )
    after_train = evaluate_value_model(
        model, train_samples, batch_size=args.batch_size, device=args.device
    )
    after_validation = evaluate_value_model(
        model, validation_samples, batch_size=args.batch_size, device=args.device
    )
    unchanged_keys = [
        key
        for key, value in model.state_dict().items()
        if key.startswith("value_head.")
        or torch.equal(value.detach().cpu(), original_state[key])
    ]
    policy_frozen = len(unchanged_keys) == len(model.state_dict())
    if not policy_frozen:
        changed = sorted(set(model.state_dict()) - set(unchanged_keys))
        raise RuntimeError(f"value-only training changed shared/policy parameters: {changed}")

    train_teams = sorted({str(sample.team_id) for sample in train_samples})
    validation_teams = sorted({str(sample.team_id) for sample in validation_samples})
    metrics = {
        "schema": "vgc-search-value-calibration-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset),
        "base_checkpoint": str(args.checkpoint.resolve()),
        "base_checkpoint_sha256": _sha256(args.checkpoint),
        "sample_count": len(samples),
        "battle_count": len({sample.battle_id for sample in samples}),
        "team_count": len({sample.team_id for sample in samples}),
        "train_teams": train_teams,
        "validation_teams": validation_teams,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "before": {"train": before_train, "validation": before_validation},
        "training": training,
        "after": {"train": after_train, "validation": after_validation},
        "policy_parameters_frozen": policy_frozen,
        "config": asdict(config),
        "promotion_authority": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    split_path = args.out_dir / "team_split.json"
    split_path.write_text(
        json.dumps(
            {"train": train_teams, "validation": validation_teams},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    source_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibrated = dict(source_checkpoint)
    calibrated.update(
        {
            "model_state_dict": model.state_dict(),
            "value_output_transform": model.value_output_transform,
            "stage": "search_trajectory_value_calibration",
            "value_calibration": metrics,
        }
    )
    checkpoint_path = args.out_dir / "best.pt"
    torch.save(calibrated, checkpoint_path)
    print(json.dumps({**metrics, "checkpoint": str(checkpoint_path.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    from vgc.mechanics_gate import enforce_mechanics_gate_for_cli
    from vgc.battle_state_gate import enforce_battle_state_gate_for_cli

    enforce_mechanics_gate_for_cli("model training")
    enforce_battle_state_gate_for_cli("model training")
    raise SystemExit(main())
