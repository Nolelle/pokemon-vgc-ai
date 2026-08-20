"""Teach the candidate policy to imitate the existing search policy before PPO.

This is a bootstrap, not the final objective. The teacher supplies a competent legal
joint action for each state; PPO later remains free to improve beyond that teacher from
actual win/loss rewards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from poke_env.battle.double_battle import DoubleBattle

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.distill requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.actions import describe_order
from vgc.agent import VgcPlayer
from vgc.evaluator import score_joint_orders
from vgc.rl.encoding import (
    META_SCALAR_DIM,
    CandidateFeatures,
    InformationFeatures,
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    encode_battle_history,
    encode_candidates,
    encode_live_state,
    encode_information_context,
    encode_meta_context,
    pad_candidate_features,
    pad_candidate_tactical_features,
)
from vgc.search import search_joint_orders


@dataclass(frozen=True)
class DistillationConfig:
    epochs: int = 8
    batch_size: int = 128
    max_grad_norm: float = 1.0
    val_fraction: float = 0.2
    seed: int = 0
    # Consecutive epochs with no improvement in validation accuracy (only checked when
    # distill_policy is called with val_samples) before training stops early. 0 (the
    # default) disables early stopping entirely -- distill_policy always runs the full
    # `epochs` and never restores a "best" checkpoint, current behavior byte-for-byte.
    early_stopping_patience: int = 0


@dataclass(frozen=True)
class DistillationSample:
    battle_id: str
    state_indices: np.ndarray
    state_scalars: np.ndarray
    history_scalars: np.ndarray
    candidates: CandidateFeatures
    teacher_action_index: int
    # Always recorded (regardless of whether the model being distilled uses meta
    # features) -- cheap and lets --meta-features be combined with --bootstrap-games
    # without needing a separate teacher-recording pass.
    meta_scalars: np.ndarray | None = None
    information: InformationFeatures | None = None
    source_id: str = "simulator_teacher"
    team_id: str | None = None
    opponent_team_id: str | None = None


class TeacherRecordingPlayer(VgcPlayer):
    """Play the existing policy while recording its choice among every legal order."""

    def __init__(self, **player_kwargs) -> None:
        self.distillation_samples: list[DistillationSample] = []
        super().__init__(**player_kwargs)

    def decide(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        memory = self._memory_for(battle)
        state_indices, state_scalars = encode_live_state(battle, self.config)
        history_scalars = encode_battle_history(memory)
        if self.config.use_two_ply_search:
            scored = search_joint_orders(battle, self.config)
        elif self.config.use_heuristic_evaluator:
            scored = score_joint_orders(battle, self.config)
        else:
            scored = []
        if not scored:
            return self.choose_random_move(battle)

        # search/evaluator returns the complete legal set in score order. Candidate
        # position is never encoded, so putting the teacher's choice at index 0 cannot
        # leak the answer to the network.
        orders = [entry.order for entry in scored]
        self.distillation_samples.append(
            DistillationSample(
                battle_id=battle.battle_tag,
                state_indices=np.array(state_indices, copy=True),
                state_scalars=np.array(state_scalars, copy=True),
                history_scalars=np.array(history_scalars, copy=True),
                candidates=encode_candidates(
                    orders, battle=battle, memory=memory, config=self.config
                ),
                teacher_action_index=0,
                meta_scalars=encode_meta_context(battle, self.config),
                information=encode_information_context(battle, memory, self.config),
            )
        )
        chosen = orders[0]
        memory.record_choice(int(getattr(battle, "turn", 0) or 0), describe_order(chosen))
        return chosen

    def completed_samples(self) -> list[DistillationSample]:
        completed = {tag for tag, battle in self.battles.items() if battle.finished}
        return [sample for sample in self.distillation_samples if sample.battle_id in completed]


def teacher_action_index(battle, config, orders) -> int | None:
    """Return the existing policy's preferred action within ``orders``.

    The network's candidate order is intentionally independent of the evaluator's
    score order, so matching uses the complete Showdown order description, including
    targets and transformation flags.
    """

    if config.use_two_ply_search:
        scored = search_joint_orders(battle, config)
    elif config.use_heuristic_evaluator:
        scored = score_joint_orders(battle, config)
    else:
        return None
    if not scored:
        return None
    teacher_description = describe_order(scored[0].order)
    matches = [
        index for index, order in enumerate(orders) if describe_order(order) == teacher_description
    ]
    return matches[0] if len(matches) == 1 else None


def split_samples_by_battle(
    samples: list[DistillationSample],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[DistillationSample], list[DistillationSample]]:
    """Game-disjoint train/validation split so adjacent turns cannot leak across it."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    battle_ids = sorted({sample.battle_id for sample in samples})
    if len(battle_ids) < 2:
        raise ValueError("distillation requires at least two completed battles")
    rng = random.Random(seed)
    rng.shuffle(battle_ids)
    val_count = max(1, min(len(battle_ids) - 1, round(len(battle_ids) * val_fraction)))
    val_ids = set(battle_ids[:val_count])
    train = [sample for sample in samples if sample.battle_id not in val_ids]
    val = [sample for sample in samples if sample.battle_id in val_ids]
    return train, val


def _tensor_batch(samples: list[DistillationSample], device: str) -> dict[str, torch.Tensor]:
    moves, targets, species, flags, mask = pad_candidate_features(
        [sample.candidates for sample in samples]
    )
    tactical = pad_candidate_tactical_features([sample.candidates for sample in samples])
    batch = {
        "state_indices": torch.as_tensor(
            np.stack([sample.state_indices for sample in samples]),
            dtype=torch.long,
            device=device,
        ),
        "state_scalars": torch.as_tensor(
            np.stack([sample.state_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "history_scalars": torch.as_tensor(
            np.stack([sample.history_scalars for sample in samples]),
            dtype=torch.float32,
            device=device,
        ),
        "move_indices": torch.as_tensor(moves, dtype=torch.long, device=device),
        "target_indices": torch.as_tensor(targets, dtype=torch.long, device=device),
        "switch_species_indices": torch.as_tensor(species, dtype=torch.long, device=device),
        "flags": torch.as_tensor(flags, dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "tactical_features": torch.as_tensor(
            tactical, dtype=torch.float32, device=device
        ),
        "teacher_actions": torch.as_tensor(
            [sample.teacher_action_index for sample in samples],
            dtype=torch.long,
            device=device,
        ),
        # Hand-built samples (e.g. in tests) may not set meta_scalars -- fall back to a
        # well-formed zero vector rather than requiring every caller to populate it.
        "meta_scalars": torch.as_tensor(
            np.stack(
                [
                    sample.meta_scalars
                    if sample.meta_scalars is not None
                    else np.zeros(META_SCALAR_DIM, dtype=np.float32)
                    for sample in samples
                ]
            ),
            dtype=torch.float32,
            device=device,
        ),
    }
    batch["information_indices"] = torch.as_tensor(
        np.stack(
            [
                sample.information.indices
                if sample.information is not None
                else np.zeros(INFORMATION_INDEX_DIM, dtype=np.int64)
                for sample in samples
            ]
        ),
        dtype=torch.long,
        device=device,
    )
    batch["information_scalars"] = torch.as_tensor(
        np.stack(
            [
                sample.information.scalars
                if sample.information is not None
                else np.zeros(INFORMATION_SCALAR_DIM, dtype=np.float32)
                for sample in samples
            ]
        ),
        dtype=torch.float32,
        device=device,
    )
    return batch


def evaluate_agreement(
    model: nn.Module,
    samples: list[DistillationSample],
    *,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    if not samples:
        raise ValueError("cannot evaluate an empty distillation sample set")
    model.eval()
    correct = 0
    top3_correct = 0
    total = 0
    losses: list[float] = []
    entropies: list[float] = []
    teacher_probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = _tensor_batch(samples[start : start + batch_size], device)
            logits, _values = model(
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
            )
            loss = nn.functional.cross_entropy(logits, batch["teacher_actions"])
            distribution = Categorical(logits=logits)
            probabilities = distribution.probs
            teacher_probabilities.extend(
                probabilities.gather(1, batch["teacher_actions"].unsqueeze(1))
                .squeeze(1)
                .cpu()
                .tolist()
            )
            ranked = logits.topk(min(3, logits.shape[-1]), dim=-1).indices
            correct += int((ranked[:, 0] == batch["teacher_actions"]).sum().item())
            top3_correct += int(
                (ranked == batch["teacher_actions"].unsqueeze(1)).any(dim=1).sum().item()
            )
            total += logits.shape[0]
            losses.append(float(loss))
            entropies.append(float(distribution.entropy().mean()))
    return {
        "accuracy": correct / total,
        "top3_accuracy": top3_correct / total,
        "teacher_probability": sum(teacher_probabilities) / len(teacher_probabilities),
        "loss": sum(losses) / len(losses),
        "entropy": sum(entropies) / len(entropies),
        "samples": float(total),
    }


def distill_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_samples: list[DistillationSample],
    config: DistillationConfig,
    *,
    device: str,
    val_samples: list[DistillationSample] | None = None,
) -> dict[str, float]:
    """Train `model` to imitate the teacher for `config.epochs` epochs.

    When `val_samples` is given, evaluates honest validation agreement after EVERY
    epoch, tracks the best epoch by validation accuracy, and restores that best
    state_dict into `model` before returning -- this is the early-stopping / best-
    epoch-selection lever described in the module docstring's overfitting rationale.
    `config.early_stopping_patience > 0` additionally stops training once that many
    consecutive epochs pass with no improvement over the best. When `val_samples` is
    None (the default caller shape), behavior is exactly as before this argument
    existed: fixed `config.epochs`, no extra evaluation cost, no restore.
    """

    if not train_samples:
        raise ValueError("cannot train from an empty distillation sample set")
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    grad_norms: list[float] = []
    best_epoch = 0
    best_val_accuracy = -1.0
    # Validation cross-entropy is the primary checkpoint selector. Exact top-1
    # agreement is very coarse on a small held-out set: a model can assign much more
    # probability to the teacher across nearly every state while flipping one marginal
    # argmax and appearing "worse" by accuracy. Loss is the proper scoring rule for the
    # full distribution; teacher probability and then accuracy break ties.
    best_val_key = (float("-inf"), -1.0, -1.0)
    best_val_teacher_probability = 0.0
    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_since_improvement = 0
    val_history: list[dict[str, float]] = []
    epochs_run = 0
    for epoch in range(1, config.epochs + 1):
        indices = rng.permutation(len(train_samples))
        model.train()
        for start in range(0, len(indices), config.batch_size):
            selected = indices[start : start + config.batch_size]
            batch = _tensor_batch([train_samples[int(index)] for index in selected], device)
            logits, _values = model(
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
            )
            loss = nn.functional.cross_entropy(logits, batch["teacher_actions"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
        epochs_run = epoch

        if val_samples is not None:
            # evaluate_agreement puts the model in eval mode internally; return to
            # train mode afterward so the next epoch's dropout/BN behavior is
            # unaffected by having just evaluated.
            epoch_metrics = evaluate_agreement(
                model, val_samples, batch_size=config.batch_size, device=device
            )
            model.train()
            val_history.append(
                {
                    "epoch": float(epoch),
                    "accuracy": epoch_metrics["accuracy"],
                    "teacher_probability": epoch_metrics["teacher_probability"],
                    "loss": epoch_metrics["loss"],
                }
            )
            candidate_key = (
                -epoch_metrics["loss"],
                epoch_metrics["teacher_probability"],
                epoch_metrics["accuracy"],
            )
            if candidate_key > best_val_key:
                best_val_key = candidate_key
                best_val_accuracy = epoch_metrics["accuracy"]
                best_val_teacher_probability = epoch_metrics["teacher_probability"]
                best_val_loss = epoch_metrics["loss"]
                best_epoch = epoch
                best_state_dict = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if (
                    config.early_stopping_patience > 0
                    and epochs_since_improvement >= config.early_stopping_patience
                ):
                    break

    result: dict[str, float] = {
        "loss": sum(losses) / len(losses),
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "samples": float(len(train_samples)),
        "epochs": float(config.epochs),
    }
    if val_samples is not None:
        assert best_state_dict is not None  # at least one epoch always runs
        model.load_state_dict(best_state_dict)
        result["best_epoch"] = float(best_epoch)
        result["best_val_accuracy"] = best_val_accuracy
        result["best_val_teacher_probability"] = best_val_teacher_probability
        result["best_val_loss"] = best_val_loss
        result["epochs_run"] = float(epochs_run)
        result["val_history"] = val_history
    return result
