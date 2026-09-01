"""Persistent complete-joint-action examples for strategic imitation training."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Literal

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.demonstrations requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.rl.distill import DistillationSample, PUBLIC_TEACHER_SOURCE_ID

DEMONSTRATION_FORMAT_VERSION = "vgc-joint-demonstrations-v3-public-information"
SPLIT_MANIFEST_VERSION = "vgc-demonstration-split-v1"
INFORMATION_CONTRACT_VERSION = "public_observation_only_v1"
SplitGroup = Literal["battle", "team"]

REQUIRED_METADATA_FIELDS = (
    "created_at_utc",
    "repository_commit",
    "repository_dirty",
    "showdown_commit",
    "showdown_dirty",
    "format_id",
    "collector",
    "requested_games",
    "seed",
    "team_source",
    "opponents",
    "policy_config",
    "information_contract",
    "teacher_source",
)


def validate_metadata(metadata: dict[str, object]) -> None:
    missing = [name for name in REQUIRED_METADATA_FIELDS if metadata.get(name) in (None, "")]
    if missing:
        raise ValueError(f"demonstration metadata is missing required fields: {missing}")
    if metadata["information_contract"] != INFORMATION_CONTRACT_VERSION:
        raise ValueError(
            "demonstration metadata has the wrong information contract: "
            f"{metadata['information_contract']!r}"
        )
    if metadata["teacher_source"] != PUBLIC_TEACHER_SOURCE_ID:
        raise ValueError(
            f"demonstration metadata has unsafe teacher source {metadata['teacher_source']!r}"
        )
    if not isinstance(metadata["requested_games"], int) or metadata["requested_games"] < 1:
        raise ValueError("demonstration metadata requested_games must be a positive integer")
    if not isinstance(metadata["seed"], int):
        raise ValueError("demonstration metadata seed must be an integer")
    for field in ("repository_dirty", "showdown_dirty"):
        if not isinstance(metadata[field], bool):
            raise ValueError(f"demonstration metadata {field} must be a boolean")


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _numeric_array_is_valid(array) -> bool:
    return getattr(array, "ndim", 0) > 0 and all(
        math.isfinite(float(value)) for value in array.reshape(-1)
    )


def validate_sample(sample: DistillationSample) -> None:
    if len(sample.candidates) <= 0:
        raise ValueError("a demonstration must contain at least one legal candidate")
    if not 0 <= sample.teacher_action_index < len(sample.candidates):
        raise ValueError("teacher action does not identify a legal joint candidate")
    if sample.information is None:
        raise ValueError("complete-context demonstrations require information features")
    if sample.mechanics is None:
        raise ValueError("complete-context demonstrations require a full mechanics snapshot")
    if sample.source_id != PUBLIC_TEACHER_SOURCE_ID:
        raise ValueError(
            "training demonstrations must come from public-information exact Showdown "
            "branches; got "
            f"{sample.source_id!r}"
        )
    if sample.candidates.tactical is None:
        raise ValueError("complete-context demonstrations require tactical features")
    if not sample.battle_id or int(sample.turn) < 1:
        raise ValueError("a demonstration requires a battle id and positive turn")
    if int(sample.decision_index) < 1 or sample.request_kind not in ("move", "switch"):
        raise ValueError(
            "a demonstration requires a positive decision index and move/switch request kind"
        )
    if not _valid_sha256(sample.team_sha256) or not _valid_sha256(
        sample.opponent_team_sha256
    ):
        raise ValueError("a demonstration requires SHA-256 fingerprints for both teams")
    count = len(sample.candidates)
    if sample.legal_action_count != count:
        raise ValueError("legal_action_count does not match the candidate feature rows")
    descriptions = sample.candidate_descriptions
    if descriptions is None or len(descriptions) != count:
        raise ValueError("canonical descriptions are required for every legal candidate")
    if len(set(descriptions)) != count or any(not description for description in descriptions):
        raise ValueError("legal candidate descriptions must be non-empty and unique")
    if sample.teacher_action_description != descriptions[sample.teacher_action_index]:
        raise ValueError("teacher action does not match its legal candidate description")
    candidate_arrays = (
        sample.candidates.move_indices,
        sample.candidates.target_indices,
        sample.candidates.switch_species_indices,
        sample.candidates.flags,
        sample.candidates.tactical,
    )
    if any(array.shape[0] != count for array in candidate_arrays):
        raise ValueError("candidate feature arrays do not have the same row count")
    numeric_arrays = (
        sample.state_indices,
        sample.state_scalars,
        sample.history_scalars,
        sample.information.indices,
        sample.information.scalars,
        sample.mechanics.tokens,
        *candidate_arrays,
    )
    if any(not _numeric_array_is_valid(array) for array in numeric_arrays):
        raise ValueError("a demonstration contains an empty, malformed, or non-finite array")
    aligned_optional = (
        ("candidate_myopic_ranks", sample.candidate_myopic_ranks),
        ("candidate_tags", sample.candidate_tags),
        ("search_scores", sample.search_scores),
        ("searched_mask", sample.searched_mask),
    )
    for name, array in aligned_optional:
        if array is None or array.shape[0] != count:
            raise ValueError(f"{name} is required and must align with the candidates")
        if not _numeric_array_is_valid(array):
            raise ValueError(f"{name} contains invalid values")


def annotate_samples(
    samples: list[DistillationSample],
    *,
    team_id: str,
    opponent_team_id: str,
    team_sha256: str,
    opponent_team_sha256: str,
    source_id: str = PUBLIC_TEACHER_SOURCE_ID,
) -> list[DistillationSample]:
    return [
        replace(
            sample,
            source_id=source_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            team_sha256=team_sha256,
            opponent_team_sha256=opponent_team_sha256,
        )
        for sample in samples
    ]


def save_demonstrations(
    path: Path,
    samples: list[DistillationSample],
    *,
    metadata: dict[str, object],
) -> None:
    validate_metadata(metadata)
    for sample in samples:
        validate_sample(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": DEMONSTRATION_FORMAT_VERSION,
            "metadata": metadata,
            "samples": samples,
            "sample_count": len(samples),
            "battle_count": len({sample.battle_id for sample in samples}),
            "team_count": len({sample.team_id for sample in samples if sample.team_id}),
        },
        path,
    )


def load_demonstration_dataset(
    path: Path,
) -> tuple[list[DistillationSample], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != DEMONSTRATION_FORMAT_VERSION:
        raise ValueError(
            f"unsupported demonstration format {payload.get('format')!r}; "
            f"expected {DEMONSTRATION_FORMAT_VERSION!r}. Older v2 files used a private "
            "simulator root and are intentionally blocked from new training."
        )
    metadata = dict(payload.get("metadata") or {})
    validate_metadata(metadata)
    samples = list(payload.get("samples") or [])
    for sample in samples:
        validate_sample(sample)
    return samples, metadata


def load_demonstrations(path: Path) -> list[DistillationSample]:
    samples, _metadata = load_demonstration_dataset(path)
    return samples


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_group(sample: DistillationSample, group_by: SplitGroup) -> str:
    return sample.battle_id if group_by == "battle" else str(sample.team_sha256)


def save_split_manifest(
    path: Path,
    *,
    dataset_path: Path,
    train: list[DistillationSample],
    validation: list[DistillationSample],
    group_by: SplitGroup,
    seed: int,
    val_fraction: float,
) -> None:
    train_groups = sorted({_split_group(sample, group_by) for sample in train})
    validation_groups = sorted({_split_group(sample, group_by) for sample in validation})
    if set(train_groups) & set(validation_groups):
        raise ValueError("training and validation groups overlap")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": SPLIT_MANIFEST_VERSION,
                "dataset": str(dataset_path.resolve()),
                "dataset_sha256": file_sha256(dataset_path),
                "group_by": group_by,
                "seed": seed,
                "validation_fraction": val_fraction,
                "train_groups": train_groups,
                "validation_groups": validation_groups,
                "train_samples": len(train),
                "validation_samples": len(validation),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def split_samples_grouped(
    samples: list[DistillationSample],
    *,
    val_fraction: float,
    seed: int,
    group_by: SplitGroup = "battle",
) -> tuple[list[DistillationSample], list[DistillationSample]]:
    """Split whole battles or whole teams so adjacent states cannot leak into validation."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if group_by == "team" and any(not _valid_sha256(sample.team_sha256) for sample in samples):
        raise ValueError("team-grouped validation requires a team fingerprint on every sample")
    def key(sample: DistillationSample) -> str:
        return _split_group(sample, group_by)

    groups = sorted({key(sample) for sample in samples})
    if len(groups) < 2:
        raise ValueError(f"{group_by}-grouped validation requires at least two groups")
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_count = max(1, min(len(groups) - 1, round(len(groups) * val_fraction)))
    validation_groups = set(groups[:val_count])
    train = [sample for sample in samples if key(sample) not in validation_groups]
    validation = [sample for sample in samples if key(sample) in validation_groups]
    return train, validation
