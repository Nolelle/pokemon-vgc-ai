"""Value-head-only training on real full-search trajectories.

Every target is the final +/-1 result of the battle that actually produced the state.
No unplayed hypothetical action receives that outcome. All shared representation and
policy parameters stay frozen; only ``CandidatePolicyValueNet.value_head`` changes.
"""

from __future__ import annotations

import copy
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.value_calibration requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.bc.encoding import MOVE_TO_IDX, SPECIES_TO_IDX, TARGET_TO_IDX
from vgc.rl.encoding import TACTICAL_FEATURE_DIM
from vgc.rl.mechanics_encoding import pad_mechanics_features


@dataclass(frozen=True)
class ValueCalibrationConfig:
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 8
    seed: int = 20260817


def split_samples_by_team_archetype(
    samples: list,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list, list]:
    """Hold out whole teams, with at least one team per represented archetype."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if any(sample.team_id is None for sample in samples):
        raise ValueError("team-disjoint value calibration requires team_id on every sample")
    teams_by_archetype: dict[str, list[str]] = defaultdict(list)
    for team_id in sorted({str(sample.team_id) for sample in samples}):
        archetype = team_id.split("/", 1)[0]
        teams_by_archetype[archetype].append(team_id)
    validation_teams: set[str] = set()
    for index, archetype in enumerate(sorted(teams_by_archetype)):
        teams = list(teams_by_archetype[archetype])
        if len(teams) < 2:
            raise ValueError(f"archetype {archetype!r} needs at least two teams")
        random.Random(seed + index).shuffle(teams)
        count = max(1, min(len(teams) - 1, round(len(teams) * validation_fraction)))
        validation_teams.update(teams[:count])
    train = [sample for sample in samples if str(sample.team_id) not in validation_teams]
    validation = [sample for sample in samples if str(sample.team_id) in validation_teams]
    return train, validation


def _tensor_batch(samples: list, device: str) -> dict[str, torch.Tensor]:
    count = len(samples)
    moves = np.full((count, 1, 2), MOVE_TO_IDX["<pass>"], dtype=np.int64)
    targets = np.full((count, 1, 2), TARGET_TO_IDX["<none>"], dtype=np.int64)
    species = np.full((count, 1, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64)
    mechanics = [getattr(sample.encoded, "mechanics", None) for sample in samples]
    if any(feature is None for feature in mechanics):
        raise ValueError("value calibration requires complete mechanics snapshots")
    mechanics_tokens, mechanics_mask = pad_mechanics_features(mechanics)
    return {
        "state_indices": torch.as_tensor(
            np.stack([sample.encoded.state_indices for sample in samples]),
            dtype=torch.long,
            device=device,
        ),
        "state_scalars": torch.as_tensor(
            np.stack([sample.encoded.state_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "history_scalars": torch.as_tensor(
            np.stack([sample.encoded.history_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "move_indices": torch.as_tensor(moves, dtype=torch.long, device=device),
        "target_indices": torch.as_tensor(targets, dtype=torch.long, device=device),
        "switch_species_indices": torch.as_tensor(species, dtype=torch.long, device=device),
        "flags": torch.zeros((count, 1, 2, 4), dtype=torch.float32, device=device),
        "candidate_mask": torch.ones((count, 1), dtype=torch.bool, device=device),
        "meta_scalars": torch.as_tensor(
            np.stack([sample.encoded.meta_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "information_indices": torch.as_tensor(
            np.stack([sample.encoded.information.indices for sample in samples]),
            dtype=torch.long,
            device=device,
        ),
        "information_scalars": torch.as_tensor(
            np.stack([sample.encoded.information.scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "tactical_features": torch.zeros(
            (count, 1, TACTICAL_FEATURE_DIM), dtype=torch.float32, device=device
        ),
        "mechanics_tokens": torch.as_tensor(
            mechanics_tokens, dtype=torch.long, device=device
        ),
        "mechanics_mask": torch.as_tensor(
            mechanics_mask, dtype=torch.bool, device=device
        ),
        "targets": torch.as_tensor(
            [float(sample.outcome) for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
    }


def _values(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    _logits, values = model(
        batch["state_indices"],
        batch["state_scalars"],
        batch["history_scalars"],
        batch["move_indices"],
        batch["target_indices"],
        batch["switch_species_indices"],
        batch["flags"],
        batch["candidate_mask"],
        meta_scalars=batch["meta_scalars"] if model.use_meta_features else None,
        information_indices=(
            batch["information_indices"] if model.use_information_features else None
        ),
        information_scalars=(
            batch["information_scalars"] if model.use_information_features else None
        ),
        tactical_features=(
            batch["tactical_features"] if model.use_tactical_features else None
        ),
        mechanics_tokens=(
            batch["mechanics_tokens"] if model.use_mechanics_features else None
        ),
        mechanics_mask=(
            batch["mechanics_mask"] if model.use_mechanics_features else None
        ),
    )
    return values


def evaluate_value_model(
    model: nn.Module,
    samples: list,
    *,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    if not samples:
        raise ValueError("cannot evaluate an empty value sample set")
    model.eval()
    predictions: list[float] = []
    targets: list[float] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _tensor_batch(samples[start : start + batch_size], device)
            predictions.extend(_values(model, batch).cpu().tolist())
            targets.extend(batch["targets"].cpu().tolist())
    prediction_array = np.asarray(predictions, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    squared = (prediction_array - target_array) ** 2
    by_battle: dict[str, list[float]] = defaultdict(list)
    by_team: dict[str, list[float]] = defaultdict(list)
    for sample, error in zip(samples, squared, strict=True):
        by_battle[str(sample.battle_id)].append(float(error))
        by_team[str(sample.team_id)].append(float(error))
    probabilities = np.clip((prediction_array + 1.0) / 2.0, 0.0, 1.0)
    wins = (target_array + 1.0) / 2.0
    bins: list[dict[str, object]] = []
    for low in np.linspace(0.0, 0.8, 5):
        high = low + 0.2
        mask = (probabilities >= low) & (
            probabilities <= high if high >= 1.0 else probabilities < high
        )
        bins.append(
            {
                "range": [float(low), float(high)],
                "count": int(mask.sum()),
                "mean_prediction": float(probabilities[mask].mean()) if mask.any() else None,
                "observed_win_rate": float(wins[mask].mean()) if mask.any() else None,
            }
        )
    return {
        "samples": len(samples),
        "battles": len(by_battle),
        "teams": len(by_team),
        "decision_mse": float(squared.mean()),
        "battle_equal_mse": float(
            np.mean([np.mean(errors) for errors in by_battle.values()])
        ),
        "team_equal_mse": float(np.mean([np.mean(errors) for errors in by_team.values()])),
        "win_probability_brier": float(np.mean((probabilities - wins) ** 2)),
        "value_min": float(prediction_array.min()),
        "value_max": float(prediction_array.max()),
        "calibration_bins": bins,
    }


def calibrate_value_head(
    model: nn.Module,
    train_samples: list,
    validation_samples: list,
    config: ValueCalibrationConfig,
    *,
    device: str,
) -> dict[str, object]:
    if not train_samples or not validation_samples:
        raise ValueError("value calibration requires non-empty train and validation sets")
    if any(sample.outcome not in (-1.0, 0.0, 1.0) for sample in [*train_samples, *validation_samples]):
        raise ValueError("value calibration requires terminal outcomes in {-1, 0, +1}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.value_head.parameters():
        parameter.requires_grad_(True)
    model.value_output_transform = "tanh"
    optimizer = torch.optim.AdamW(
        model.value_head.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    battle_counts = Counter(str(sample.battle_id) for sample in train_samples)
    rng = np.random.default_rng(config.seed)
    best_state = copy.deepcopy(model.value_head.state_dict())
    best_epoch = 0
    best_metric = float("inf")
    patience_used = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        indices = rng.permutation(len(train_samples))
        model.eval()  # shared representation is frozen; keep inference deterministic
        losses: list[float] = []
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            samples = [train_samples[int(index)] for index in selected]
            batch = _tensor_batch(samples, device)
            predictions = _values(model, batch)
            weights = torch.as_tensor(
                [1.0 / battle_counts[str(sample.battle_id)] for sample in samples],
                dtype=torch.float32,
                device=device,
            )
            loss = (weights * (predictions - batch["targets"]).square()).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_value_model(
            model,
            validation_samples,
            batch_size=config.batch_size,
            device=device,
        )
        metric = float(validation["battle_equal_mse"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": sum(losses) / len(losses),
                "validation_battle_equal_mse": metric,
            }
        )
        if metric < best_metric - 1e-8:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.value_head.state_dict())
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= config.patience:
                break
    model.value_head.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_battle_equal_mse": best_metric,
        "history": history,
        "trainable_parameters": ["value_head.weight", "value_head.bias"],
        "value_output_transform": model.value_output_transform,
    }
