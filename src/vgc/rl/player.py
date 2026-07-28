"""poke-env player that collects on-policy trajectories from real Showdown battles."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise ImportError(
        "vgc.rl.player requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.rl.encoding import (
    encode_battle_history,
    encode_candidates,
    encode_live_state,
    encode_meta_context,
    pad_candidate_features,
)
from vgc.rl.distill import teacher_action_index
from vgc.rl.ppo import PpoConfig, RolloutBuffer, RolloutStep, select_action


class PpoVgcPlayer(VgcPlayer):
    """Choose legal joint orders with a network and record them for PPO updates."""

    def __init__(
        self,
        *,
        model,
        rollout_buffer: RolloutBuffer | None = None,
        ppo_config: PpoConfig = PpoConfig(),
        device: str = "cpu",
        deterministic: bool = False,
        **player_kwargs,
    ) -> None:
        self.model = model.to(device)
        self.rollout_buffer = rollout_buffer
        self.ppo_config = ppo_config
        self.device = device
        self.deterministic = deterministic
        self._battle_step_counts: dict[str, int] = {}
        super().__init__(**player_kwargs)

    def decide(self, battle):
        orders = enumerate_joint_orders(battle)
        if not orders:
            return self.choose_random_move(battle)
        memory = self._memory_for(battle)
        state_indices, state_scalars = encode_live_state(battle, self.config)
        history_scalars = encode_battle_history(memory)
        candidates = encode_candidates(orders)
        moves, targets, species, flags, mask = pad_candidate_features([candidates])
        meta_scalars = (
            encode_meta_context(battle, self.config) if self.model.use_meta_features else None
        )
        meta_tensor = (
            torch.as_tensor(meta_scalars[None, :], dtype=torch.float32, device=self.device)
            if meta_scalars is not None
            else None
        )
        self.model.eval()
        with torch.no_grad():
            actions, log_probs, values = select_action(
                self.model,
                torch.as_tensor(state_indices[None, :], dtype=torch.long, device=self.device),
                torch.as_tensor(state_scalars[None, :], dtype=torch.float32, device=self.device),
                torch.as_tensor(history_scalars[None, :], dtype=torch.float32, device=self.device),
                torch.as_tensor(moves, dtype=torch.long, device=self.device),
                torch.as_tensor(targets, dtype=torch.long, device=self.device),
                torch.as_tensor(species, dtype=torch.long, device=self.device),
                torch.as_tensor(flags, dtype=torch.float32, device=self.device),
                torch.as_tensor(mask, dtype=torch.bool, device=self.device),
                deterministic=self.deterministic,
                meta_scalars=meta_tensor,
            )
        action_index = int(actions.item())
        if self.rollout_buffer is not None:
            teacher_index = (
                teacher_action_index(battle, self.config, orders)
                if self.ppo_config.teacher_anchor_weight > 0.0
                else None
            )
            self.rollout_buffer.add(
                RolloutStep(
                    state_indices=np.array(state_indices, copy=True),
                    state_scalars=np.array(state_scalars, copy=True),
                    history_scalars=np.array(history_scalars, copy=True),
                    candidates=candidates,
                    action_index=action_index,
                    old_log_prob=float(log_probs.item()),
                    old_value=float(values.item()),
                    teacher_action_index=teacher_index,
                    meta_scalars=(
                        np.array(meta_scalars, copy=True) if meta_scalars is not None else None
                    ),
                )
            )
            tag = battle.battle_tag
            self._battle_step_counts[tag] = self._battle_step_counts.get(tag, 0) + 1
        memory.record_choice(
            int(getattr(battle, "turn", 0) or 0), describe_order(orders[action_index])
        )
        return orders[action_index]

    def _battle_finished_callback(self, battle) -> None:
        count = self._battle_step_counts.pop(battle.battle_tag, 0)
        if self.rollout_buffer is not None and count:
            outcome = 1.0 if battle.won else (-1.0 if battle.lost else 0.0)
            self.rollout_buffer.finish_episode(outcome, self.ppo_config)
