"""Action-and-response value data and a default-off Q model.

The generic PPO value head answers "how good is this state?".  Search needs a more
local comparison: "from this exact state, how good is our action against this possible
opponent response?"  ``ActionResponseValueNet`` predicts that Q(o, a, b) quantity.

Nothing in this module is connected to the shipped player or search scorer.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.counterfactual_q requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.rl.encoding import (
    CandidateFeatures,
    InformationFeatures,
    pad_candidate_features,
    pad_candidate_tactical_features,
)
from vgc.rl.model import ACTION_HIDDEN_DIM, CandidatePolicyValueNet

COUNTERFACTUAL_Q_FORMAT = "vgc-counterfactual-q-v1"


@dataclass(frozen=True)
class CounterfactualQInput:
    """One fog-safe root observation plus one simultaneous action pair."""

    state_indices: np.ndarray
    state_scalars: np.ndarray
    history_scalars: np.ndarray
    own_action: CandidateFeatures
    opponent_action: CandidateFeatures
    meta_scalars: np.ndarray
    information: InformationFeatures


@dataclass(frozen=True)
class CounterfactualQSample:
    """Continuation estimate for one Q(o, a, b) training target."""

    root_id: str
    battle_id: str
    team_id: str
    opponent_team_id: str
    own_action: str
    opponent_action: str
    encoded: CounterfactualQInput
    mean_outcome: float
    outcome_variance: float
    rollout_count: int
    search_score: float
    opponent_search_score: float
    opponent_weight: float
    root_metadata: dict[str, Any] = field(default_factory=dict)


def validate_q_sample(sample: CounterfactualQSample) -> None:
    if len(sample.encoded.own_action) != 1:
        raise ValueError("Q samples require exactly one own action")
    if len(sample.encoded.opponent_action) != 1:
        raise ValueError("Q samples require exactly one opponent response")
    if not -1.0 <= sample.mean_outcome <= 1.0:
        raise ValueError("mean outcome must be in [-1, +1]")
    if sample.outcome_variance < 0.0 or not math.isfinite(sample.outcome_variance):
        raise ValueError("outcome variance must be finite and nonnegative")
    if sample.rollout_count <= 0:
        raise ValueError("rollout count must be positive")
    if sample.opponent_weight <= 0.0 or not math.isfinite(sample.opponent_weight):
        raise ValueError("opponent response weight must be finite and positive")
    arrays = (
        sample.encoded.state_indices,
        sample.encoded.state_scalars,
        sample.encoded.history_scalars,
        sample.encoded.meta_scalars,
        sample.encoded.information.scalars,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Q sample contains non-finite observation features")


def save_q_dataset(
    path: Path,
    samples: list[CounterfactualQSample],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not samples:
        raise ValueError("cannot save an empty Q dataset")
    for sample in samples:
        validate_q_sample(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": COUNTERFACTUAL_Q_FORMAT,
            "samples": samples,
            "sample_count": len(samples),
            "root_count": len({sample.root_id for sample in samples}),
            "team_count": len(
                {team for sample in samples for team in (sample.team_id, sample.opponent_team_id)}
            ),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_q_dataset(path: Path) -> tuple[list[CounterfactualQSample], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != COUNTERFACTUAL_Q_FORMAT:
        raise ValueError(
            f"unsupported Q dataset {payload.get('format')!r}; expected {COUNTERFACTUAL_Q_FORMAT!r}"
        )
    samples = list(payload.get("samples") or [])
    if not samples:
        raise ValueError("Q dataset is empty")
    for sample in samples:
        validate_q_sample(sample)
    return samples, dict(payload.get("metadata") or {})


def _team_components(samples: list[CounterfactualQSample]) -> list[set[str]]:
    """Return connected team groups so no roster can cross the train/dev boundary."""

    parent: dict[str, str] = {}

    def find(team: str) -> str:
        parent.setdefault(team, team)
        if parent[team] != team:
            parent[team] = find(parent[team])
        return parent[team]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for sample in samples:
        union(sample.team_id, sample.opponent_team_id)
    components: dict[str, set[str]] = defaultdict(set)
    for team in parent:
        components[find(team)].add(team)
    return list(components.values())


def split_q_samples_team_disjoint(
    samples: list[CounterfactualQSample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CounterfactualQSample], list[CounterfactualQSample]]:
    """Split whole connected matchups; neither side's team can leak across the split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    components = _team_components(samples)
    if len(components) < 2:
        raise ValueError(
            "team-disjoint validation requires at least two disconnected matchup groups"
        )
    rng = random.Random(seed)
    rng.shuffle(components)
    validation_count = max(
        1, min(len(components) - 1, round(len(components) * validation_fraction))
    )
    validation_teams = set().union(*components[:validation_count])
    validation = [
        sample
        for sample in samples
        if sample.team_id in validation_teams or sample.opponent_team_id in validation_teams
    ]
    validation_ids = {id(sample) for sample in validation}
    train = [sample for sample in samples if id(sample) not in validation_ids]
    train_teams = {team for sample in train for team in (sample.team_id, sample.opponent_team_id)}
    observed_validation_teams = {
        team for sample in validation for team in (sample.team_id, sample.opponent_team_id)
    }
    if train_teams & observed_validation_teams:
        raise RuntimeError("team-disjoint split leaked a roster across train and validation")
    return train, validation


class ActionResponseValueNet(nn.Module):
    """Predict Q(o, a, b) from a policy network's fog-safe shared representation."""

    def __init__(self, backbone: CandidatePolicyValueNet) -> None:
        super().__init__()
        self.backbone = backbone
        self.q_head = nn.Sequential(
            nn.Linear(ACTION_HIDDEN_DIM * 6, ACTION_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(ACTION_HIDDEN_DIM, 1),
        )

    def forward(
        self,
        state_indices: torch.Tensor,
        state_scalars: torch.Tensor,
        history_scalars: torch.Tensor,
        own_move_indices: torch.Tensor,
        own_target_indices: torch.Tensor,
        own_switch_species_indices: torch.Tensor,
        own_flags: torch.Tensor,
        opponent_move_indices: torch.Tensor,
        opponent_target_indices: torch.Tensor,
        opponent_switch_species_indices: torch.Tensor,
        opponent_flags: torch.Tensor,
        pair_mask: torch.Tensor,
        *,
        meta_scalars: torch.Tensor | None = None,
        information_indices: torch.Tensor | None = None,
        information_scalars: torch.Tensor | None = None,
        own_tactical_features: torch.Tensor | None = None,
        opponent_tactical_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pair_mask.ndim != 2 or not torch.all(pair_mask.any(dim=1)):
            raise ValueError("each batch row must contain at least one action pair")
        context = self.backbone.encode_context(
            state_indices,
            state_scalars,
            history_scalars,
            meta_scalars=meta_scalars,
            information_indices=information_indices,
            information_scalars=information_scalars,
        )
        own_hidden = self.backbone.encode_actions(
            own_move_indices,
            own_target_indices,
            own_switch_species_indices,
            own_flags,
            tactical_features=own_tactical_features,
        )
        opponent_hidden = self.backbone.encode_actions(
            opponent_move_indices,
            opponent_target_indices,
            opponent_switch_species_indices,
            opponent_flags,
            tactical_features=opponent_tactical_features,
        )
        if own_hidden.shape != opponent_hidden.shape:
            raise ValueError("own actions and opponent responses must have matching shapes")
        projected = self.backbone.state_projection(context).unsqueeze(1).expand_as(own_hidden)
        combined = torch.cat(
            (
                projected,
                own_hidden,
                opponent_hidden,
                projected * own_hidden,
                projected * opponent_hidden,
                own_hidden * opponent_hidden,
            ),
            dim=-1,
        )
        values = torch.tanh(self.q_head(combined).squeeze(-1))
        return values.masked_fill(~pair_mask.bool(), 0.0)


def _tensor_batch(
    samples: list[CounterfactualQSample], device: str
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    own = [sample.encoded.own_action for sample in samples]
    opponent = [sample.encoded.opponent_action for sample in samples]
    own_moves, own_targets, own_species, own_flags, own_mask = pad_candidate_features(own)
    opp_moves, opp_targets, opp_species, opp_flags, opp_mask = pad_candidate_features(opponent)
    pair_mask = own_mask & opp_mask
    inputs = {
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
        "own_move_indices": torch.as_tensor(own_moves, dtype=torch.long, device=device),
        "own_target_indices": torch.as_tensor(own_targets, dtype=torch.long, device=device),
        "own_switch_species_indices": torch.as_tensor(own_species, dtype=torch.long, device=device),
        "own_flags": torch.as_tensor(own_flags, dtype=torch.float32, device=device),
        "opponent_move_indices": torch.as_tensor(opp_moves, dtype=torch.long, device=device),
        "opponent_target_indices": torch.as_tensor(opp_targets, dtype=torch.long, device=device),
        "opponent_switch_species_indices": torch.as_tensor(
            opp_species, dtype=torch.long, device=device
        ),
        "opponent_flags": torch.as_tensor(opp_flags, dtype=torch.float32, device=device),
        "pair_mask": torch.as_tensor(pair_mask, dtype=torch.bool, device=device),
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
        "own_tactical_features": torch.as_tensor(
            pad_candidate_tactical_features(own), dtype=torch.float32, device=device
        ),
        "opponent_tactical_features": torch.as_tensor(
            pad_candidate_tactical_features(opponent), dtype=torch.float32, device=device
        ),
    }
    targets = torch.as_tensor(
        [sample.mean_outcome for sample in samples], dtype=torch.float32, device=device
    )
    return inputs, targets


def predict_q(
    model: ActionResponseValueNet,
    samples: list[CounterfactualQSample],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    predictions: list[float] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            inputs, _targets = _tensor_batch(samples[start : start + batch_size], device)
            if not model.backbone.use_meta_features:
                inputs["meta_scalars"] = None
            if not model.backbone.use_information_features:
                inputs["information_indices"] = None
                inputs["information_scalars"] = None
            if not model.backbone.use_tactical_features:
                inputs["own_tactical_features"] = None
                inputs["opponent_tactical_features"] = None
            predictions.extend(model(**inputs)[:, 0].cpu().tolist())
    return np.asarray(predictions, dtype=np.float64)


def _ranking_metrics(
    samples: list[CounterfactualQSample], predictions: np.ndarray
) -> dict[str, Any]:
    by_root_response: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_root_action: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        by_root_response[(sample.root_id, sample.opponent_action)].append(index)
        by_root_action[(sample.root_id, sample.own_action)].append(index)

    def score(groups: list[list[tuple[float, float]]]) -> dict[str, Any]:
        pairs = 0
        correct = 0.0
        tops = 0
        top_correct = 0
        regrets: list[float] = []
        for rows in groups:
            if len(rows) < 2:
                continue
            for left in range(len(rows)):
                for right in range(left + 1, len(rows)):
                    actual_delta = rows[left][1] - rows[right][1]
                    if math.isclose(actual_delta, 0.0, abs_tol=1e-12):
                        continue
                    predicted_delta = rows[left][0] - rows[right][0]
                    pairs += 1
                    if math.isclose(predicted_delta, 0.0, abs_tol=1e-12):
                        correct += 0.5
                    elif predicted_delta * actual_delta > 0.0:
                        correct += 1.0
            predicted_best = max(range(len(rows)), key=lambda index: rows[index][0])
            actual_best = max(row[1] for row in rows)
            top_correct += int(math.isclose(rows[predicted_best][1], actual_best, abs_tol=1e-12))
            tops += 1
            regrets.append(actual_best - rows[predicted_best][1])
        return {
            "groups": tops,
            "pairwise_comparisons": pairs,
            "pairwise_accuracy": correct / pairs if pairs else None,
            "top_action_accuracy": top_correct / tops if tops else None,
            "mean_regret": float(np.mean(regrets)) if regrets else None,
        }

    conditioned = [
        [(float(predictions[index]), samples[index].mean_outcome) for index in indices]
        for indices in by_root_response.values()
    ]
    root_actions: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (root_id, _action), indices in by_root_action.items():
        weights = np.asarray(
            [samples[index].opponent_weight for index in indices], dtype=np.float64
        )
        weights /= weights.sum()
        root_actions[root_id].append(
            (
                float(np.sum(predictions[indices] * weights)),
                float(
                    np.sum(np.asarray([samples[index].mean_outcome for index in indices]) * weights)
                ),
            )
        )
    return {
        "opponent_response_conditioned": score(conditioned),
        "action_aggregated": score(list(root_actions.values())),
    }


def evaluate_q_model(
    model: ActionResponseValueNet,
    samples: list[CounterfactualQSample],
    *,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot evaluate an empty Q sample set")
    predictions = predict_q(model, samples, batch_size=batch_size, device=device)
    targets = np.asarray([sample.mean_outcome for sample in samples], dtype=np.float64)
    result = _ranking_metrics(samples, predictions)
    search = _ranking_metrics(
        samples,
        np.asarray([sample.search_score for sample in samples], dtype=np.float64),
    )
    q_action = result["action_aggregated"]
    search_action = search["action_aggregated"]
    result.update(
        {
            "samples": len(samples),
            "roots": len({sample.root_id for sample in samples}),
            "teams": len(
                {team for sample in samples for team in (sample.team_id, sample.opponent_team_id)}
            ),
            "mean_squared_error": float(np.mean((predictions - targets) ** 2)),
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
            "search_baseline": search,
            "improvement_over_search": {
                "pairwise_accuracy": (
                    q_action["pairwise_accuracy"] - search_action["pairwise_accuracy"]
                    if q_action["pairwise_accuracy"] is not None
                    and search_action["pairwise_accuracy"] is not None
                    else None
                ),
                "top_action_accuracy": (
                    q_action["top_action_accuracy"] - search_action["top_action_accuracy"]
                    if q_action["top_action_accuracy"] is not None
                    and search_action["top_action_accuracy"] is not None
                    else None
                ),
                "regret_reduction": (
                    search_action["mean_regret"] - q_action["mean_regret"]
                    if q_action["mean_regret"] is not None
                    and search_action["mean_regret"] is not None
                    else None
                ),
            },
        }
    )
    return result


@dataclass(frozen=True)
class QTrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    ranking_weight: float = 0.5
    patience: int = 8
    seed: int = 20260819
    train_backbone: bool = False
    backbone_learning_rate: float | None = None
    """Separate, slower rate for the shared encoder when ``train_backbone`` is set.

    The encoder was fitted for policy/value, not for ranking counterfactual action
    outcomes, so letting it move is a genuinely different hypothesis from fitting a head
    on top of it.  But it also carries far more parameters than the head and a much
    smaller Q dataset to justify them, and an early pilot overfitted when it was
    unfrozen at the head's rate.  ``None`` reproduces that old behaviour exactly (one
    rate for everything); a small value here is the conservative version -- the head
    still learns fast while the representation drifts slowly.
    """


def train_q_model(
    model: ActionResponseValueNet,
    train_samples: list[CounterfactualQSample],
    validation_samples: list[CounterfactualQSample],
    config: QTrainingConfig,
    *,
    device: str,
) -> dict[str, Any]:
    """Fit expected return plus within-root ordering, with early stopping on regret."""

    if not train_samples or not validation_samples:
        raise ValueError("Q training requires non-empty train and validation sets")
    if config.backbone_learning_rate is not None and config.backbone_learning_rate < 0.0:
        raise ValueError("backbone learning rate must be nonnegative")
    if config.backbone_learning_rate is not None and not config.train_backbone:
        raise ValueError("backbone_learning_rate requires train_backbone")
    head_parameters = list(model.q_head.parameters())
    head_ids = {id(parameter) for parameter in head_parameters}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in head_ids
    ]
    for parameter in backbone_parameters:
        parameter.requires_grad_(config.train_backbone)
    for parameter in head_parameters:
        parameter.requires_grad_(True)
    groups: list[dict[str, Any]] = [{"params": head_parameters, "lr": config.learning_rate}]
    backbone_lr = (
        config.learning_rate
        if config.backbone_learning_rate is None
        else config.backbone_learning_rate
    )
    if config.train_backbone and backbone_parameters and backbone_lr > 0.0:
        groups.append({"params": backbone_parameters, "lr": backbone_lr})
    optimizer = torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    by_root: dict[str, list[CounterfactualQSample]] = defaultdict(list)
    for sample in train_samples:
        by_root[sample.root_id].append(sample)
    root_ids = sorted(by_root)
    rng = random.Random(config.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_regret = float("inf")
    best_epoch = 0
    patience_used = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        rng.shuffle(root_ids)
        model.train()
        losses: list[float] = []
        for root_id in root_ids:
            samples = by_root[root_id]
            inputs, targets = _tensor_batch(samples, device)
            if not model.backbone.use_meta_features:
                inputs["meta_scalars"] = None
            if not model.backbone.use_information_features:
                inputs["information_indices"] = None
                inputs["information_scalars"] = None
            if not model.backbone.use_tactical_features:
                inputs["own_tactical_features"] = None
                inputs["opponent_tactical_features"] = None
            predictions = model(**inputs)[:, 0]
            regression = F.smooth_l1_loss(predictions, targets)
            differences = targets[:, None] - targets[None, :]
            triangle = torch.triu(torch.ones_like(differences, dtype=torch.bool), diagonal=1)
            same_response = torch.as_tensor(
                [
                    [left.opponent_action == right.opponent_action for right in samples]
                    for left in samples
                ],
                dtype=torch.bool,
                device=device,
            )
            comparable = triangle & same_response & (differences.abs() > 1e-6)
            ranking_terms: list[torch.Tensor] = []
            if comparable.any():
                predicted_differences = predictions[:, None] - predictions[None, :]
                ranking_terms.append(
                    F.softplus(
                        -differences[comparable].sign() * predicted_differences[comparable]
                    ).mean()
                )
            # Search ultimately compares our actions after combining the plausible
            # opponent responses, so train that exact within-root comparison too.
            action_indices: dict[str, list[int]] = defaultdict(list)
            for index, sample in enumerate(samples):
                action_indices[sample.own_action].append(index)
            if len(action_indices) >= 2:
                action_predictions: list[torch.Tensor] = []
                action_targets: list[torch.Tensor] = []
                for indices in action_indices.values():
                    weights = torch.as_tensor(
                        [samples[index].opponent_weight for index in indices],
                        dtype=torch.float32,
                        device=device,
                    )
                    weights = weights / weights.sum()
                    action_predictions.append((predictions[indices] * weights).sum())
                    action_targets.append((targets[indices] * weights).sum())
                aggregate_predictions = torch.stack(action_predictions)
                aggregate_targets = torch.stack(action_targets)
                aggregate_differences = aggregate_targets[:, None] - aggregate_targets[None, :]
                aggregate_mask = torch.triu(
                    torch.ones_like(aggregate_differences, dtype=torch.bool), diagonal=1
                ) & (aggregate_differences.abs() > 1e-6)
                if aggregate_mask.any():
                    predicted_differences = (
                        aggregate_predictions[:, None] - aggregate_predictions[None, :]
                    )
                    ranking_terms.append(
                        F.softplus(
                            -aggregate_differences[aggregate_mask].sign()
                            * predicted_differences[aggregate_mask]
                        ).mean()
                    )
            ranking = (
                torch.stack(ranking_terms).mean() if ranking_terms else predictions.new_zeros(())
            )
            loss = regression + config.ranking_weight * ranking
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_q_model(model, validation_samples, batch_size=512, device=device)
        regret = validation["action_aggregated"]["mean_regret"]
        metric = float(regret) if regret is not None else float("inf")
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "validation_action_regret": metric,
            }
        )
        if metric < best_regret - 1e-8:
            best_regret = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_used = 0
        else:
            patience_used += 1
            if patience_used >= config.patience:
                break
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_action_regret": best_regret,
        "history": history,
        "train_backbone": config.train_backbone,
        "backbone_learning_rate": (backbone_lr if config.train_backbone else 0.0),
        "head_parameters": sum(p.numel() for p in head_parameters),
        "backbone_parameters": sum(p.numel() for p in backbone_parameters),
    }
