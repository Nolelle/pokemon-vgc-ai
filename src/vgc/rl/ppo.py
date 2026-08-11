"""Masked PPO rollout storage, GAE returns, and optimization.

Rewards are sparse by design: intermediate turns receive zero and the final decision
receives +1 for a win, -1 for a loss, or 0 for a draw. GAE propagates that battle
result backward through the decisions that created it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch import nn
    from torch.distributions import Categorical
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.ppo requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.rl.encoding import CandidateFeatures, pad_candidate_features


@dataclass(frozen=True)
class PpoConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    # Small imitation guardrail: retain probability on the proven search action while
    # PPO remains free to prefer a different action when win/loss evidence supports it.
    teacher_anchor_weight: float = 0.05
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 128
    # Potential-based reward-shaping coefficient (see vgc.rl.rewards.board_potential
    # and RolloutBuffer.finish_episode). 0.0 (default) disables shaping entirely --
    # rewards/advantages are byte-for-byte identical to the sparse-only behavior.
    reward_shaping_coef: float = 0.0


@dataclass
class RolloutStep:
    state_indices: np.ndarray
    state_scalars: np.ndarray
    history_scalars: np.ndarray
    candidates: CandidateFeatures
    action_index: int
    old_log_prob: float
    old_value: float
    teacher_action_index: int | None = None
    # Populated only when the model was built with use_meta_features=True (see
    # vgc.rl.encoding.encode_meta_context); left None otherwise so non-meta training is
    # byte-for-byte unaffected.
    meta_scalars: np.ndarray | None = None
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    return_value: float = 0.0
    # Board potential Phi(s_t) at decision time (vgc.rl.rewards.board_potential),
    # populated whenever PpoConfig.reward_shaping_coef > 0 and left at the default 0.0
    # otherwise. Only ever consumed by finish_episode's shaping step.
    state_potential: float = 0.0


class RolloutBuffer:
    """Completed on-policy episodes ready for one or more PPO epochs."""

    def __init__(self) -> None:
        self.steps: list[RolloutStep] = []
        self._episode_start = 0

    def add(self, step: RolloutStep) -> None:
        if step.action_index < 0 or step.action_index >= len(step.candidates):
            raise ValueError("action_index does not identify a legal candidate")
        if step.teacher_action_index is not None and not (
            0 <= step.teacher_action_index < len(step.candidates)
        ):
            raise ValueError("teacher_action_index does not identify a legal candidate")
        self.steps.append(step)

    def finish_episode(
        self, outcome: float, config: PpoConfig, *, terminal_potential: float = 0.0
    ) -> None:
        episode = self.steps[self._episode_start :]
        if not episode:
            return
        if config.reward_shaping_coef > 0.0:
            # Potential-based shaping (Ng, Harada & Russell 1999): adding
            # coef * (gamma * Phi(s_{t+1}) - Phi(s_t)) to every step's reward
            # telescopes to coef * (gamma * Phi(terminal) - Phi(s_0)) over the whole
            # episode, so it densifies credit assignment without changing which
            # policy is optimal. `terminal_potential` is Phi of the finished battle's
            # final state, computed by the caller (see PpoVgcPlayer._battle_finished_
            # callback) since this buffer has no battle object of its own.
            last_index = len(episode) - 1
            for index, step in enumerate(episode):
                next_potential = (
                    episode[index + 1].state_potential
                    if index < last_index
                    else terminal_potential
                )
                step.reward += config.reward_shaping_coef * (
                    config.gamma * next_potential - step.state_potential
                )
        # When reward_shaping_coef == 0.0, every step's reward is still exactly 0.0
        # here (the shaping loop above never ran), so this line is byte-for-byte the
        # same assignment as before shaping existed.
        episode[-1].reward += float(outcome)
        episode[-1].done = True
        gae = 0.0
        next_value = 0.0
        for step in reversed(episode):
            continuation = 0.0 if step.done else 1.0
            delta = step.reward + config.gamma * next_value * continuation - step.old_value
            gae = delta + config.gamma * config.gae_lambda * continuation * gae
            step.advantage = gae
            step.return_value = gae + step.old_value
            next_value = step.old_value
        self._episode_start = len(self.steps)

    def clear(self) -> None:
        self.steps.clear()
        self._episode_start = 0

    def discard_unfinished_episode(self) -> int:
        """Drop only the in-flight tail after a failed/interrupted simulator game."""

        discarded = len(self.steps) - self._episode_start
        del self.steps[self._episode_start :]
        return discarded

    def extend_finished(self, other: "RolloutBuffer") -> None:
        """Merge a worker buffer after verifying it has no partial episode."""

        if other._episode_start != len(other.steps):
            raise ValueError("cannot merge a rollout buffer with an unfinished episode")
        self.steps.extend(other.steps)
        self._episode_start = len(self.steps)

    def __len__(self) -> int:
        return len(self.steps)


def select_action(
    model: nn.Module,
    state_indices: torch.Tensor,
    state_scalars: torch.Tensor,
    history_scalars: torch.Tensor,
    move_indices: torch.Tensor,
    target_indices: torch.Tensor,
    switch_species_indices: torch.Tensor,
    flags: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    deterministic: bool = False,
    meta_scalars: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (or greedily choose) only among the mask's legal candidates.

    `generator` makes the SAMPLING reproducible without touching torch's global RNG,
    which is what `docs/rl_roadmap.md`'s deterministic-replay criterion needs: the
    simulator seed alone does not pin an episode, because the policy is stochastic by
    design. Leaving it None keeps the previous global-RNG behavior byte for byte.
    """

    logits, values = model(
        state_indices,
        state_scalars,
        history_scalars,
        move_indices,
        target_indices,
        switch_species_indices,
        flags,
        candidate_mask,
        meta_scalars=meta_scalars,
    )
    distribution = Categorical(logits=logits)
    if deterministic:
        actions = logits.argmax(dim=-1)
    elif generator is None:
        actions = distribution.sample()
    else:
        # Categorical.sample() takes no generator, so draw from its own normalized
        # probs -- identical distribution, just from a stream we control. Masked
        # candidates carry ~0 probability here exactly as they do in .sample().
        actions = torch.multinomial(distribution.probs, 1, generator=generator).squeeze(-1)
    return actions, distribution.log_prob(actions), values


def _tensor_batch(steps: list[RolloutStep], device: str) -> dict[str, torch.Tensor]:
    moves, targets, species, flags, mask = pad_candidate_features(
        [step.candidates for step in steps]
    )
    batch = {
        "state_indices": torch.as_tensor(
            np.stack([step.state_indices for step in steps]), dtype=torch.long, device=device
        ),
        "state_scalars": torch.as_tensor(
            np.stack([step.state_scalars for step in steps]), dtype=torch.float32, device=device
        ),
        "history_scalars": torch.as_tensor(
            np.stack([step.history_scalars for step in steps]),
            dtype=torch.float32,
            device=device,
        ),
        "move_indices": torch.as_tensor(moves, dtype=torch.long, device=device),
        "target_indices": torch.as_tensor(targets, dtype=torch.long, device=device),
        "switch_species_indices": torch.as_tensor(species, dtype=torch.long, device=device),
        "flags": torch.as_tensor(flags, dtype=torch.float32, device=device),
        "candidate_mask": torch.as_tensor(mask, dtype=torch.bool, device=device),
        "actions": torch.as_tensor(
            [step.action_index for step in steps], dtype=torch.long, device=device
        ),
        "teacher_actions": torch.as_tensor(
            [
                step.teacher_action_index if step.teacher_action_index is not None else -1
                for step in steps
            ],
            dtype=torch.long,
            device=device,
        ),
        "old_log_probs": torch.as_tensor(
            [step.old_log_prob for step in steps], dtype=torch.float32, device=device
        ),
        "old_values": torch.as_tensor(
            [step.old_value for step in steps], dtype=torch.float32, device=device
        ),
        "advantages": torch.as_tensor(
            [step.advantage for step in steps], dtype=torch.float32, device=device
        ),
        "returns": torch.as_tensor(
            [step.return_value for step in steps], dtype=torch.float32, device=device
        ),
    }
    # meta_scalars is populated on every step when the model uses meta features (see
    # PpoVgcPlayer.decide) and None on every step otherwise -- a batch is always
    # homogeneous within one training run, so checking the first step is sufficient.
    if steps[0].meta_scalars is not None:
        batch["meta_scalars"] = torch.as_tensor(
            np.stack([step.meta_scalars for step in steps]), dtype=torch.float32, device=device
        )
    return batch


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    config: PpoConfig = PpoConfig(),
    *,
    device: str = "cpu",
) -> dict[str, float]:
    """Run clipped PPO epochs over completed trajectories and return mean metrics."""

    if not buffer.steps:
        raise ValueError("cannot update from an empty rollout buffer")
    if buffer._episode_start != len(buffer.steps):
        raise ValueError("rollout buffer contains an unfinished episode")

    advantages = np.asarray([step.advantage for step in buffer.steps], dtype=np.float32)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for step, advantage in zip(buffer.steps, advantages):
        step.advantage = float(advantage)

    rng = np.random.default_rng()
    metric_rows: list[dict[str, float]] = []
    model.train()
    for _ in range(config.epochs):
        indices = rng.permutation(len(buffer.steps))
        for start in range(0, len(indices), config.minibatch_size):
            chosen = indices[start : start + config.minibatch_size]
            batch = _tensor_batch([buffer.steps[int(index)] for index in chosen], device)
            logits, values = model(
                batch["state_indices"],
                batch["state_scalars"],
                batch["history_scalars"],
                batch["move_indices"],
                batch["target_indices"],
                batch["switch_species_indices"],
                batch["flags"],
                batch["candidate_mask"],
                meta_scalars=batch.get("meta_scalars"),
            )
            distribution = Categorical(logits=logits)
            log_probs = distribution.log_prob(batch["actions"])
            ratio = (log_probs - batch["old_log_probs"]).exp()
            unclipped = ratio * batch["advantages"]
            clipped = ratio.clamp(1.0 - config.clip_ratio, 1.0 + config.clip_ratio)
            policy_loss = -torch.minimum(unclipped, clipped * batch["advantages"]).mean()

            clipped_values = batch["old_values"] + (values - batch["old_values"]).clamp(
                -config.value_clip_ratio, config.value_clip_ratio
            )
            value_unclipped = (values - batch["returns"]).square()
            value_clipped = (clipped_values - batch["returns"]).square()
            value_loss = 0.5 * torch.maximum(value_unclipped, value_clipped).mean()
            entropy = distribution.entropy().mean()
            teacher_mask = batch["teacher_actions"] >= 0
            if teacher_mask.any():
                teacher_logits = logits[teacher_mask]
                teacher_actions = batch["teacher_actions"][teacher_mask]
                teacher_anchor_loss = nn.functional.cross_entropy(
                    teacher_logits,
                    teacher_actions,
                )
                teacher_probabilities = teacher_logits.softmax(dim=-1).gather(
                    1,
                    teacher_actions.unsqueeze(1),
                )
                teacher_probability = teacher_probabilities.mean()
                teacher_agreement = (
                    (teacher_logits.argmax(dim=-1) == teacher_actions).float().mean()
                )
            else:
                # A plain zero tensor, not `logits.sum() * 0.0` -- if `logits` were
                # ever non-finite (model divergence), multiplying by zero would still
                # propagate NaN/Inf (`0.0 * nan == nan` in IEEE 754) into `loss` below
                # even though `config.teacher_anchor_weight` may itself be 0.0 (e.g.
                # the end of a --teacher-anchor-final-weight anneal), silently
                # corrupting an otherwise-unrelated update.
                teacher_anchor_loss = logits.new_zeros(())
                teacher_probability = logits.new_zeros(())
                teacher_agreement = logits.new_zeros(())
            loss = (
                policy_loss
                + config.value_loss_weight * value_loss
                - config.entropy_weight * entropy
                + config.teacher_anchor_weight * teacher_anchor_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            approx_kl = (batch["old_log_probs"] - log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > config.clip_ratio).float().mean()
            metric_rows.append(
                {
                    "loss": float(loss.detach()),
                    "policy_loss": float(policy_loss.detach()),
                    "value_loss": float(value_loss.detach()),
                    "entropy": float(entropy.detach()),
                    "teacher_anchor_loss": float(teacher_anchor_loss.detach()),
                    "teacher_probability": float(teacher_probability.detach()),
                    "teacher_agreement": float(teacher_agreement.detach()),
                    "teacher_coverage": float(teacher_mask.float().mean().detach()),
                    "approx_kl": float(approx_kl.detach()),
                    "clip_fraction": float(clip_fraction.detach()),
                    "grad_norm": float(grad_norm.detach()),
                }
            )

    return {
        key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in metric_rows[0]
    } | {"steps": float(len(buffer.steps))}
