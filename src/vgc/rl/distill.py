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
    CandidateFeatures,
    encode_battle_history,
    encode_candidates,
    encode_live_state,
    pad_candidate_features,
)
from vgc.search import search_joint_orders


@dataclass(frozen=True)
class DistillationConfig:
    epochs: int = 8
    batch_size: int = 128
    max_grad_norm: float = 1.0
    val_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class DistillationSample:
    battle_id: str
    state_indices: np.ndarray
    state_scalars: np.ndarray
    history_scalars: np.ndarray
    candidates: CandidateFeatures
    teacher_action_index: int


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
                candidates=encode_candidates(orders),
                teacher_action_index=0,
            )
        )
        chosen = orders[0]
        memory.record_choice(int(getattr(battle, "turn", 0) or 0), describe_order(chosen))
        return chosen

    def completed_samples(self) -> list[DistillationSample]:
        completed = {tag for tag, battle in self.battles.items() if battle.finished}
        return [sample for sample in self.distillation_samples if sample.battle_id in completed]


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
    return {
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
        "teacher_actions": torch.as_tensor(
            [sample.teacher_action_index for sample in samples],
            dtype=torch.long,
            device=device,
        ),
    }


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
) -> dict[str, float]:
    if not train_samples:
        raise ValueError("cannot train from an empty distillation sample set")
    rng = np.random.default_rng(config.seed)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _epoch in range(config.epochs):
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
            )
            loss = nn.functional.cross_entropy(logits, batch["teacher_actions"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm.detach()))
    return {
        "loss": sum(losses) / len(losses),
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "samples": float(len(train_samples)),
        "epochs": float(config.epochs),
    }
