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
    encode_information_context,
    encode_live_state,
    encode_meta_context,
    pad_candidate_features,
    pad_candidate_tactical_features,
)
from vgc.rl.distill import teacher_action_index
from vgc.rl.ppo import PpoConfig, RolloutBuffer, RolloutStep, select_action
from vgc.rl.rewards import board_potential


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
        policy_seed: int | None = None,
        **player_kwargs,
    ) -> None:
        self.model = model.to(device)
        self.rollout_buffer = rollout_buffer
        self.ppo_config = ppo_config
        self.device = device
        self.deterministic = deterministic
        # A private sampling stream, so one player's action draws are reproducible and
        # independent of anything else that touches torch's global RNG (data shuffling,
        # dropout, another player in the same process). None keeps the global RNG.
        self.policy_generator: torch.Generator | None = None
        if policy_seed is not None:
            self.policy_generator = torch.Generator(device=device)
            self.policy_generator.manual_seed(int(policy_seed))
        self._battle_step_counts: dict[str, int] = {}
        super().__init__(**player_kwargs)

    def decide(self, battle):
        orders = enumerate_joint_orders(battle)
        if not orders:
            return self.choose_random_move(battle)
        memory = self._memory_for(battle)
        state_indices, state_scalars = encode_live_state(battle, self.config)
        history_scalars = encode_battle_history(memory)
        candidates = encode_candidates(
            orders,
            battle=battle if self.model.use_tactical_features else None,
            memory=memory if self.model.use_tactical_features else None,
            config=self.config,
        )
        moves, targets, species, flags, mask = pad_candidate_features([candidates])
        tactical = pad_candidate_tactical_features([candidates])
        meta_scalars = (
            encode_meta_context(battle, self.config) if self.model.use_meta_features else None
        )
        meta_tensor = (
            torch.as_tensor(meta_scalars[None, :], dtype=torch.float32, device=self.device)
            if meta_scalars is not None
            else None
        )
        information = (
            encode_information_context(battle, memory, self.config)
            if self.model.use_information_features
            else None
        )
        information_indices = (
            torch.as_tensor(
                information.indices[None, :], dtype=torch.long, device=self.device
            )
            if information is not None
            else None
        )
        information_scalars = (
            torch.as_tensor(
                information.scalars[None, :], dtype=torch.float32, device=self.device
            )
            if information is not None
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
                information_indices=information_indices,
                information_scalars=information_scalars,
                tactical_features=(
                    torch.as_tensor(tactical, dtype=torch.float32, device=self.device)
                    if self.model.use_tactical_features
                    else None
                ),
                generator=self.policy_generator,
            )
        action_index = int(actions.item())
        if self.rollout_buffer is not None:
            teacher_index = (
                teacher_action_index(battle, self.config, orders)
                if self.ppo_config.teacher_anchor_weight > 0.0
                else None
            )
            # Gate the compute (not just the use) on coef > 0 -- board_potential is
            # cheap, but there is no reason to pay it every decision when reward
            # shaping is off (the default).
            state_potential = (
                board_potential(battle) if self.ppo_config.reward_shaping_coef > 0.0 else 0.0
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
                    information=information,
                    state_potential=state_potential,
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
            # Phi(absorbing terminal state) MUST be 0, not board_potential(battle).
            #
            # Ng, Harada & Russell's policy-invariance result holds for an arbitrary
            # Phi only because the shaping telescopes to gamma*Phi(s_T) - Phi(s_0), and
            # that is policy-independent (hence harmless) ONLY when Phi(s_T) is a
            # constant -- conventionally 0 -- for every terminal state. Passing the
            # finished board's potential instead makes the episode's total shaping
            # depend on HOW we won: coef * gamma * Phi(final board) is larger for a win
            # with more HP/board resources left. At the documented
            # --reward-shaping-coef 0.3 that is up to ~30% of the terminal signal
            # quietly optimizing "win cleanly" rather than "win", which is exactly the
            # reward misspecification shaping is supposed to avoid.
            self.rollout_buffer.finish_episode(outcome, self.ppo_config, terminal_potential=0.0)
