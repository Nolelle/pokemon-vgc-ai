"""Persistent complete-joint-action examples for strategic imitation training."""

from __future__ import annotations

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

from vgc.rl.distill import DistillationSample

DEMONSTRATION_FORMAT_VERSION = "vgc-joint-demonstrations-v2-exact-mechanics"
SplitGroup = Literal["battle", "team"]


def validate_sample(sample: DistillationSample) -> None:
    if len(sample.candidates) <= 0:
        raise ValueError("a demonstration must contain at least one legal candidate")
    if not 0 <= sample.teacher_action_index < len(sample.candidates):
        raise ValueError("teacher action does not identify a legal joint candidate")
    if sample.information is None:
        raise ValueError("complete-context demonstrations require information features")
    if sample.mechanics is None:
        raise ValueError("complete-context demonstrations require a full mechanics snapshot")
    if sample.source_id != "exact_showdown_teacher_v1":
        raise ValueError(
            "training demonstrations must come from exact Showdown branches; got "
            f"{sample.source_id!r}"
        )
    if sample.candidates.tactical is None:
        raise ValueError("complete-context demonstrations require tactical features")


def annotate_samples(
    samples: list[DistillationSample],
    *,
    team_id: str,
    opponent_team_id: str,
    source_id: str = "exact_showdown_teacher_v1",
) -> list[DistillationSample]:
    return [
        replace(
            sample,
            source_id=source_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
        )
        for sample in samples
    ]


def save_demonstrations(path: Path, samples: list[DistillationSample]) -> None:
    for sample in samples:
        validate_sample(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": DEMONSTRATION_FORMAT_VERSION,
            "samples": samples,
            "sample_count": len(samples),
            "battle_count": len({sample.battle_id for sample in samples}),
            "team_count": len({sample.team_id for sample in samples if sample.team_id}),
        },
        path,
    )


def load_demonstrations(path: Path) -> list[DistillationSample]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != DEMONSTRATION_FORMAT_VERSION:
        raise ValueError(
            f"unsupported demonstration format {payload.get('format')!r}; "
            f"expected {DEMONSTRATION_FORMAT_VERSION!r}"
        )
    samples = list(payload.get("samples") or [])
    for sample in samples:
        validate_sample(sample)
    return samples


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
    if group_by == "team" and any(sample.team_id is None for sample in samples):
        raise ValueError("team-grouped validation requires team_id on every sample")
    key = (
        (lambda sample: sample.battle_id)
        if group_by == "battle"
        else (lambda sample: str(sample.team_id))
    )
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
