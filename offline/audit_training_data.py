#!/usr/bin/env python
"""Audit one public-information teacher dataset before model training.

This is deliberately a small, fail-closed check. It reports useful counts and rejects
missing fields, duplicate decision ids, malformed or non-member actions, broken split
fingerprints, and battle/team overlap between training and validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="default: split_manifest.json beside the (single) --dataset",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--show-examples",
        type=int,
        default=0,
        help="print this many compact examples for manual inspection",
    )
    return parser.parse_args(argv)


def _split_group(sample, group_by: str) -> str:
    return sample.battle_id if group_by == "battle" else str(sample.team_sha256)


def audit(
    dataset_paths: Path | list[Path], split_manifest_path: Path
) -> dict[str, Any]:
    import torch

    from vgc.rl.demonstrations import (
        DEMONSTRATION_FORMAT_VERSION,
        SPLIT_MANIFEST_VERSION,
        file_sha256,
        validate_sample,
        canonical_samples,
        validate_source_compatibility,
        partition_errors,
    )

    if isinstance(dataset_paths, Path):
        dataset_paths = [dataset_paths]
    label = (
        str(dataset_paths[0].resolve())
        if len(dataset_paths) == 1
        else f"{len(dataset_paths)} datasets starting with {dataset_paths[0].resolve()}"
    )

    def _fail(message: str) -> dict[str, Any]:
        return {
            "schema": "vgc-training-data-audit-v1",
            "dataset": label,
            "verdict": "FAIL",
            "errors": [message],
            "counts": {},
            "examples": [],
        }

    errors: list[str] = []
    payloads: list[dict] = []
    for dataset_path in dataset_paths:
        try:
            payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - the audit should report, not traceback
            return _fail(f"dataset could not be read: {type(exc).__name__}: {exc}")
        if not isinstance(payload, dict):
            return _fail(
                f"dataset payload must be a mapping, got {type(payload).__name__}"
            )
        payloads.append(payload)

    for payload in payloads:
        if payload.get("format") != DEMONSTRATION_FORMAT_VERSION:
            errors.append(
                f"unsupported dataset format {payload.get('format')!r}; "
                f"expected {DEMONSTRATION_FORMAT_VERSION!r}"
            )
    metadata = dict(payloads[0].get("metadata") or {})
    try:
        validate_source_compatibility([dict(p.get("metadata") or {}) for p in payloads])
    except ValueError as exc:
        errors.append(str(exc))

    samples = []
    seen_files = set()
    for path, payload in zip(dataset_paths, payloads):
        digest = file_sha256(path)
        if digest in seen_files:
            errors.append("duplicate dataset source file")
        seen_files.add(digest)
        samples.extend(canonical_samples(payload.get("samples") or [], str((payload.get("metadata") or {}).get("collection_id") or digest)))

    def _payload_counts(payload: dict) -> dict[str, int]:
        file_samples = list(payload.get("samples") or [])
        return {
            "sample_count": len(file_samples),
            "battle_count": len(
                {
                    sample.battle_id
                    for sample in file_samples
                    if isinstance(getattr(sample, "battle_id", None), str)
                }
            ),
            "team_count": len(
                {
                    sample.team_sha256
                    for sample in file_samples
                    if isinstance(getattr(sample, "team_sha256", None), str)
                }
            ),
        }

    for path, payload in zip(dataset_paths, payloads):
        for field, actual in _payload_counts(payload).items():
            if payload.get(field) != actual:
                errors.append(
                    f"{path.name}: stored {field}={payload.get(field)!r}, actual={actual}"
                )
    invalid_examples = 0
    valid_samples = []
    for index, sample in enumerate(samples):
        try:
            validate_sample(sample)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            invalid_examples += 1
            if invalid_examples <= 20:
                errors.append(f"sample {index}: {exc}")
        else:
            valid_samples.append(sample)
    if invalid_examples > 20:
        errors.append(f"{invalid_examples - 20} additional invalid samples omitted")

    decision_keys = []
    for sample in samples:
        try:
            decision_keys.append(
                (sample.battle_id, int(sample.decision_index), str(sample.team_sha256))
            )
        except (AttributeError, TypeError, ValueError):
            continue
    duplicate_decisions = sum(count - 1 for count in Counter(decision_keys).values())
    if duplicate_decisions:
        errors.append(f"{duplicate_decisions} duplicate battle/decision ids")

    # Combined counts are reported below; each file's stored counts were
    # already checked against its own samples above.
    recorded_counts = {
        "sample_count": len(samples),
        "battle_count": len(
            {
                sample.battle_id
                for sample in samples
                if isinstance(getattr(sample, "battle_id", None), str)
            }
        ),
        "team_count": len(
            {
                sample.team_sha256
                for sample in samples
                if isinstance(getattr(sample, "team_sha256", None), str)
            }
        ),
    }

    split_counts: dict[str, int] = {}
    split_team_counts: dict[str, int] = {}
    if not split_manifest_path.is_file():
        errors.append(f"split manifest is missing: {split_manifest_path}")
    else:
        try:
            manifest = json.loads(split_manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"split manifest could not be read: {type(exc).__name__}: {exc}")
            manifest = {}
        if manifest.get("schema") != SPLIT_MANIFEST_VERSION:
            errors.append(
                f"unsupported split manifest {manifest.get('schema')!r}; "
                f"expected {SPLIT_MANIFEST_VERSION!r}"
            )
        manifest_dataset_field = manifest.get("dataset")
        single_file_manifest = isinstance(manifest_dataset_field, str)
        manifest_datasets = (
            [manifest_dataset_field] if single_file_manifest else (manifest_dataset_field or [])
        )
        if sorted(str(entry) for entry in manifest_datasets) != sorted(
            str(path.resolve()) for path in dataset_paths
        ):
            errors.append("split manifest dataset list does not match the audited files")
        else:
            manifest_shas = manifest.get("dataset_sha256")
            for dataset_path in dataset_paths:
                if single_file_manifest:
                    expected_sha = manifest_shas
                else:
                    expected_sha = (manifest_shas or {}).get(str(dataset_path.resolve()))
                if expected_sha != file_sha256(dataset_path):
                    errors.append(
                        f"split manifest fingerprint does not match {dataset_path.name}"
                    )
        group_by = manifest.get("group_by")
        if group_by not in ("battle", "team"):
            errors.append(f"split manifest has invalid group_by {group_by!r}")
        else:
            train_groups = set(manifest.get("train_groups") or [])
            validation_groups = set(manifest.get("validation_groups") or [])
            overlap = train_groups & validation_groups
            if overlap:
                errors.append(f"{len(overlap)} groups appear in both training and validation")
            assigned: dict[str, list] = {"train": [], "validation": []}
            for sample in valid_samples:
                group = _split_group(sample, group_by)
                destinations = [
                    name
                    for name, groups in (
                        ("train", train_groups),
                        ("validation", validation_groups),
                    )
                    if group in groups
                ]
                if len(destinations) != 1:
                    errors.append(
                        f"sample {sample.battle_id} turn {sample.turn} belongs to "
                        f"{len(destinations)} split groups"
                    )
                    continue
                assigned[destinations[0]].append(sample)

            errors.extend(partition_errors(
                assigned["train"], assigned["validation"], group_by=group_by
            ))
            team_hashes = {
                name: {
                    fingerprint
                    for sample in partition
                    for fingerprint in (sample.team_sha256, sample.opponent_team_sha256)
                }
                for name, partition in assigned.items()
            }
            split_counts = {name: len(partition) for name, partition in assigned.items()}
            split_team_counts = {
                name: len(fingerprints) for name, fingerprints in team_hashes.items()
            }
            for name in ("train", "validation"):
                expected = manifest.get(f"{name}_samples")
                if expected != split_counts[name]:
                    errors.append(
                        f"manifest {name}_samples={expected!r}, actual={split_counts[name]}"
                    )

    action_counts = [int(sample.legal_action_count) for sample in valid_samples]
    counts = {
        **recorded_counts,
        "opponent_team_count": len(
            {sample.opponent_team_sha256 for sample in valid_samples}
        ),
        "invalid_examples": invalid_examples,
        "duplicate_decisions": duplicate_decisions,
        "turn_min": min((int(sample.turn) for sample in valid_samples), default=None),
        "turn_max": max((int(sample.turn) for sample in valid_samples), default=None),
        "legal_actions_min": min(action_counts, default=None),
        "legal_actions_max": max(action_counts, default=None),
        "split_samples": split_counts,
        "split_team_fingerprints": split_team_counts,
    }
    examples = [
        {
            "battle_id": sample.battle_id,
            "turn": int(sample.turn),
            "decision_index": int(sample.decision_index),
            "request_kind": sample.request_kind,
            "team_id": sample.team_id,
            "opponent_team_id": sample.opponent_team_id,
            "legal_actions": int(sample.legal_action_count),
            "teacher_action": sample.teacher_action_description,
            "teacher_index": int(sample.teacher_action_index),
            "teacher_source": sample.source_id,
        }
        for sample in valid_samples[:20]
    ]
    return {
        "schema": "vgc-training-data-audit-v1",
        "dataset": (
            str(dataset_paths[0].resolve())
            if len(dataset_paths) == 1
            else sorted(str(path.resolve()) for path in dataset_paths)
        ),
        "dataset_sha256": (
            file_sha256(dataset_paths[0])
            if len(dataset_paths) == 1
            else {
                str(path.resolve()): file_sha256(path) for path in dataset_paths
            }
        ),
        "split_manifest": str(split_manifest_path.resolve()),
        "verdict": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": counts,
        "metadata": metadata,
        "examples": examples,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.split_manifest is not None:
        split_manifest = args.split_manifest
    elif len(args.dataset) == 1:
        split_manifest = args.dataset[0].parent / "split_manifest.json"
    else:
        raise SystemExit("--split-manifest is required with several --dataset files")
    report = audit(args.dataset, split_manifest)
    visible = dict(report)
    visible["examples"] = report["examples"][: max(0, args.show_examples)]
    print(json.dumps(visible, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
