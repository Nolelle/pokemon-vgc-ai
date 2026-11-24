"""High-throughput Rust VecEnv + MPS/GPU PPO league trainer for Champions Reg M-C.

Drives `pikapro.rl.VecEnv(..., observation="rich", masks="factored")` across hundreds of
parallel Rust battles with zero `poke-env` text parsing, using an autoregressive factored
transformer (`RustFactoredPolicy`) on Apple Silicon MPS / CUDA / CPU.

Supports:
- Self-play + prioritized historical snapshot league + native Rust anchor opponents
- Per-trajectory GAE advantages, calibrated BCE win-probability value loss, KL early stop
- Cosine learning-rate schedule, atomic `trainer_state.pt` checkpoint + `--resume`
- Fast held-out evaluation on unseen teams against Rust `max_base_power`,
  `simple_heuristics`, and (optionally) Python `vgc` 2-ply search
- Weights & Biases logging
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pikapro
import pikapro.rl as rl
import torch
import torch.nn.functional as F

from vgc.action_gate import enforce_action_gate_for_cli
from vgc.battle_state_gate import enforce_battle_state_gate_for_cli
from vgc.config import FORMAT_ID
from vgc.mechanics_gate import enforce_mechanics_gate_for_cli
from vgc.rl.rust_policy import (
    RICH_FIELD_TURN_IDX,
    RICH_OBS_SIZE,
    RustFactoredPolicy,
    load_rust_policy,
    masked_log_prob_and_entropy,
    save_rust_policy,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "data/selfplay/v3_clean_train_manifest.json"
DEFAULT_HOLDOUT_MANIFEST = PROJECT / "data/selfplay/mc_sheet_pool_v2/holdout_manifest.json"
TRAINER_STATE_FILE = "trainer_state.pt"


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def scheduled_lr(
    completed: int,
    total_games: int,
    *,
    max_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    if schedule == "constant":
        return max_lr
    if schedule == "cosine":
        progress = min(max(completed / max(total_games, 1), 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


def load_teams(manifest: Path, *, use_weights: bool = False) -> list[str]:
    rows = json.loads(manifest.read_text())
    unique_weights: dict[str, int] = {}
    for row in rows:
        packed = (manifest.parent / row["file"]).read_text().strip()
        w = max(1, int(row.get("weight", 1))) if use_weights else 1
        if packed in unique_weights:
            unique_weights[packed] = max(unique_weights[packed], w)
        else:
            unique_weights[packed] = w
    if len(unique_weights) < 2 or any(not team for team in unique_weights):
        raise ValueError(f"need at least two different nonempty packed teams in {manifest}")
    teams: list[str] = []
    for packed, w in unique_weights.items():
        teams.extend([packed] * w)
    return teams


@dataclass
class Snapshot:
    game_count: int
    path: Path
    games: float = 0.0
    learner_points: float = 0.0

    def record_game(self, pts: float, *, max_effective_games: float = 200.0) -> None:
        """EMA-capped PFSP stat update so priority tracks current win rate, not ancient history."""
        if self.games >= max_effective_games:
            decay = (max_effective_games - 1.0) / max_effective_games
            self.games *= decay
            self.learner_points *= decay
        self.games += 1.0
        self.learner_points += float(pts)

    @property
    def priority(self) -> float:
        win_rate = (1.0 + self.learner_points) / (2.0 + self.games)
        return (1.0 - win_rate) ** 2


def build_optimizer(
    model: RustFactoredPolicy, *, lr: float, weight_decay: float
) -> torch.optim.AdamW:
    """Build AdamW with weight decay applied only to 2D+ non-embedding linear weights.

    Embeddings (`species_emb`, `item_emb`, `ability_emb`, `move_emb`), LayerNorm weights,
    and biases receive `weight_decay=0.0` so rare moves/items/species are never decayed
    toward zero between appearances.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    emb_param_ids = {
        id(p)
        for mod in model.modules()
        if isinstance(mod, (torch.nn.Embedding, torch.nn.LayerNorm))
        for p in mod.parameters(recurse=False)
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            id(param) in emb_param_ids
            or param.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "emb" in name.lower()
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": float(weight_decay)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=float(lr),
    )


@dataclass
class StepRecord:
    obs: np.ndarray  # (1124,) float32
    mask: np.ndarray  # (mask_width,) bool
    min_switches: int
    is_preview: bool
    a0: int
    a1: int
    log_prob: float
    value: float


@dataclass
class SampleBatch:
    obs: np.ndarray
    mask: np.ndarray
    min_switches: np.ndarray
    is_preview: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    old_log_prob: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    values: np.ndarray


def _finish_trajectory(
    steps: list[StepRecord],
    reward: float,
    *,
    gamma: float,
    gae_lambda: float,
    out_records: list[tuple[StepRecord, float, float]],
) -> None:
    n = len(steps)
    if n == 0:
        return
    advs = [0.0] * n
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_v = 0.0 if t == n - 1 else steps[t + 1].value
        r = reward if t == n - 1 else 0.0
        delta = r + gamma * next_v - steps[t].value
        gae = delta + gamma * gae_lambda * gae
        advs[t] = gae
    for t in range(n):
        out_records.append((steps[t], reward, advs[t]))


def _collate_samples(records: list[tuple[StepRecord, float, float]]) -> SampleBatch:
    n = len(records)
    obs = np.empty((n, RICH_OBS_SIZE), dtype=np.float32)
    mask_w = records[0][0].mask.shape[0]
    mask = np.empty((n, mask_w), dtype=bool)
    min_sw = np.empty(n, dtype=np.uint8)
    is_pv = np.empty(n, dtype=bool)
    a0 = np.empty(n, dtype=np.int64)
    a1 = np.empty(n, dtype=np.int64)
    old_lp = np.empty(n, dtype=np.float32)
    rets = np.empty(n, dtype=np.float32)
    advs = np.empty(n, dtype=np.float32)
    vals = np.empty(n, dtype=np.float32)
    for i, (rec, ret, adv) in enumerate(records):
        obs[i] = rec.obs
        mask[i] = rec.mask
        min_sw[i] = rec.min_switches
        is_pv[i] = rec.is_preview
        a0[i] = rec.a0
        a1[i] = rec.a1
        old_lp[i] = rec.log_prob
        rets[i] = ret
        advs[i] = adv
        vals[i] = rec.value
    return SampleBatch(obs, mask, min_sw, is_pv, a0, a1, old_lp, rets, advs, vals)


@torch.no_grad()
def _select_batch_actions(
    model: RustFactoredPolicy,
    space: pikapro.ActionSpace,
    obs_np: np.ndarray,
    masks_np: np.ndarray,
    min_switches_np: np.ndarray,
    *,
    deterministic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched GPU/CPU inference for active `(env, player)` decisions.

    Returns ``(a0, a1, log_prob, value, is_preview)`` as 1D numpy arrays.
    """
    n = len(obs_np)
    s = model.slot_size
    dev = model.device
    a0_out = np.full(n, -1, dtype=np.int64)
    a1_out = np.full(n, -1, dtype=np.int64)
    lp_out = np.zeros(n, dtype=np.float32)

    x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(dev)
    enc = model.encode_board(x)
    val_t, _ = model.value_from_context(enc.context)
    val_out = val_t.cpu().numpy().astype(np.float32)

    pv_masks = masks_np[:, 2 * s :]
    is_pv = pv_masks.any(axis=-1)

    if is_pv.any():
        pv_idx = np.flatnonzero(is_pv)
        pv_idx_t = torch.from_numpy(pv_idx).to(dev)
        pv_logits = model.preview_logits(enc.index(pv_idx_t))
        pv_m_t = torch.from_numpy(np.ascontiguousarray(pv_masks[pv_idx], dtype=bool)).to(dev)
        pv_logits = pv_logits.masked_fill(~pv_m_t, -1e9)
        pv_lp = F.log_softmax(pv_logits, dim=-1)
        if deterministic:
            pv_a = pv_logits.argmax(dim=-1)
        else:
            pv_a = torch.distributions.Categorical(logits=pv_logits).sample()
        pv_chosen_lp = pv_lp.gather(-1, pv_a.unsqueeze(-1)).squeeze(-1)
        a0_out[pv_idx] = pv_a.cpu().numpy()
        lp_out[pv_idx] = pv_chosen_lp.cpu().numpy()

    turn_idx = np.flatnonzero(~is_pv)
    if len(turn_idx) > 0:
        t_masks = masks_np[turn_idx]
        m0_np = t_masks[:, :s]
        has_m0 = m0_np.any(axis=-1)
        if has_m0.any():
            valid_local = np.flatnonzero(has_m0)
            v_global = turn_idx[valid_local]
            v_t = torch.from_numpy(v_global).to(dev)
            enc_v = enc.index(v_t)
            m0_t = torch.from_numpy(np.ascontiguousarray(m0_np[valid_local], dtype=bool)).to(dev)
            l0 = model.slot0_logits(enc_v).masked_fill(~m0_t, -1e9)
            lp0_all = F.log_softmax(l0, dim=-1)
            if deterministic:
                a0_t = l0.argmax(dim=-1)
            else:
                a0_t = torch.distributions.Categorical(logits=l0).sample()
            lp0 = lp0_all.gather(-1, a0_t.unsqueeze(-1)).squeeze(-1)
            a0_np = a0_t.cpu().numpy()

            m1_np = space.pair_mask(
                t_masks[valid_local],
                min_switches_np[v_global],
                a0_np,
            )
            has_m1 = m1_np.any(axis=-1)
            a1_np = np.full(len(valid_local), -1, dtype=np.int64)
            lp1_np = np.zeros(len(valid_local), dtype=np.float32)
            if has_m1.any():
                m1_ok = np.flatnonzero(has_m1)
                m1_ok_t = torch.from_numpy(m1_ok).to(dev)
                m1_t = torch.from_numpy(np.ascontiguousarray(m1_np[m1_ok], dtype=bool)).to(dev)
                l1 = model.slot1_logits(enc_v.index(m1_ok_t), a0_t[m1_ok_t]).masked_fill(
                    ~m1_t, -1e9
                )
                lp1_all = F.log_softmax(l1, dim=-1)
                if deterministic:
                    a1_t = l1.argmax(dim=-1)
                else:
                    a1_t = torch.distributions.Categorical(logits=l1).sample()
                lp1 = lp1_all.gather(-1, a1_t.unsqueeze(-1)).squeeze(-1)
                a1_np[m1_ok] = a1_t.cpu().numpy()
                lp1_np[m1_ok] = lp1.cpu().numpy()

            stuck = ~has_m1
            if stuck.any():
                a0_np[stuck] = rl.RANDOM_ACTION
                a1_np[stuck] = rl.RANDOM_ACTION

            a0_out[v_global] = a0_np
            a1_out[v_global] = a1_np
            lp_out[v_global] = lp0.cpu().numpy() + lp1_np

    return a0_out, a1_out, lp_out, val_out, is_pv


def ppo_update(
    model: RustFactoredPolicy,
    optimizer: torch.optim.Optimizer,
    space: pikapro.ActionSpace,
    batch: SampleBatch,
    rng: random.Random,
    *,
    epochs: int,
    batch_size: int,
    target_kl: float,
    entropy_coef: float = 0.01,
) -> dict[str, float]:
    dev = model.device
    s = model.slot_size
    n = len(batch.obs)
    if n == 0:
        raise ValueError("cannot run PPO update on empty batch")

    adv = batch.advantages.astype(np.float32)
    adv = (adv - adv.mean()) / max(float(adv.std()), 1e-6)

    # Precompute legal m1 masks for all turn samples in one vectorized Rust call!
    turn_all = np.flatnonzero(~batch.is_preview & (batch.a0 >= 0) & (batch.a1 >= 0))
    pv_all = np.flatnonzero(batch.is_preview & (batch.a0 >= 0))
    m1_by_sample: dict[int, np.ndarray] = {}
    if len(turn_all) > 0:
        m1_arr = space.pair_mask(
            batch.mask[turn_all], batch.min_switches[turn_all], batch.a0[turn_all]
        )
        m1_full = np.zeros((n, s), dtype=bool)
        m1_full[turn_all] = m1_arr
    else:
        m1_full = np.zeros((n, s), dtype=bool)
    del m1_by_sample

    # Move entire rollout batch to GPU/MPS once for zero per-minibatch host transfer!
    obs_t = torch.from_numpy(batch.obs).to(dev)
    mask_t = torch.from_numpy(batch.mask).to(dev)
    m1_t = torch.from_numpy(m1_full).to(dev)
    a0_t = torch.from_numpy(batch.a0).to(dev)
    a1_t = torch.from_numpy(batch.a1).to(dev)
    old_lp_t = torch.from_numpy(batch.old_log_prob).to(dev)
    ret_t = torch.from_numpy(batch.returns).to(dev)
    adv_t = torch.from_numpy(adv).to(dev)

    loss_sum = 0.0
    policy_sum = 0.0
    value_sum = 0.0
    entropy_sum = 0.0
    kl_sum = 0.0
    updates = 0
    epochs_run = 0

    model.train()
    for _ in range(epochs):
        epochs_run += 1
        epoch_kl_sum = 0.0
        epoch_updates = 0
        groups = [turn_all.tolist(), pv_all.tolist()]
        minibatches: list[tuple[bool, list[int]]] = []
        for is_pv_group, idxs in ((False, groups[0]), (True, groups[1])):
            rng.shuffle(idxs)
            chunks = [
                idxs[start : start + batch_size]
                for start in range(0, len(idxs), batch_size)
                if idxs[start : start + batch_size]
            ]
            # Merge tiny remainder tail batches (< batch_size // 2) into the preceding chunk
            # so small preview/turn remainders never take noisy full-LR gradient steps.
            if len(chunks) >= 2 and len(chunks[-1]) < batch_size // 2:
                chunks[-2].extend(chunks.pop())
            for chunk in chunks:
                minibatches.append((is_pv_group, chunk))
        rng.shuffle(minibatches)

        for is_pv_group, rows in minibatches:
            idx_t = torch.tensor(rows, dtype=torch.long, device=dev)
            enc = model.encode_board(obs_t[idx_t])
            _, value_logit = model.value_from_context(enc.context)

            if is_pv_group:
                pv_logits = model.preview_logits(enc)
                pv_mask = mask_t[idx_t, 2 * s :]
                new_lp, ent = masked_log_prob_and_entropy(pv_logits, pv_mask, a0_t[idx_t])
            else:
                l0 = model.slot0_logits(enc)
                m0 = mask_t[idx_t, :s]
                lp0, ent0 = masked_log_prob_and_entropy(l0, m0, a0_t[idx_t])
                l1 = model.slot1_logits(enc, a0_t[idx_t])
                m1 = m1_t[idx_t]
                lp1, ent1 = masked_log_prob_and_entropy(l1, m1, a1_t[idx_t])
                new_lp = lp0 + lp1
                ent = ent0 + ent1

            log_ratio = new_lp - old_lp_t[idx_t]
            ratio = log_ratio.exp()
            mb_adv = adv_t[idx_t]
            policy_loss = -torch.minimum(
                ratio * mb_adv, ratio.clamp(0.8, 1.2) * mb_adv
            ).mean()
            win_prob_target = (ret_t[idx_t] + 1.0) * 0.5
            value_loss = F.binary_cross_entropy_with_logits(value_logit, win_prob_target)
            entropy = ent.mean()
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                raise RuntimeError("training loss was not finite")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()

            updates += 1
            epoch_updates += 1
            loss_sum += float(loss.detach().item())
            policy_sum += float(policy_loss.detach().item())
            value_sum += float(value_loss.detach().item())
            entropy_sum += float(entropy.detach().item())
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                kl_val = float(approx_kl.item())
                kl_sum += kl_val
                epoch_kl_sum += kl_val
        if target_kl > 0 and epoch_updates > 0 and (epoch_kl_sum / epoch_updates) > 1.5 * target_kl:
            break
    model.eval()
    return {
        "loss": loss_sum / max(updates, 1),
        "policy_loss": policy_sum / max(updates, 1),
        "value_loss": value_sum / max(updates, 1),
        "entropy": entropy_sum / max(updates, 1),
        "approx_kl": kl_sum / max(updates, 1),
        "epochs_run": float(epochs_run),
    }


@torch.no_grad()
def eval_rust_opponents(
    model: RustFactoredPolicy,
    holdout_pool: rl.TeamPool,
    opponents: list[str],
    *,
    games_per_opponent: int,
    threads: int,
    seed: int,
) -> dict[str, float]:
    """Fast native Rust evaluation on unseen holdout teams (100 games in <1s)."""
    results: dict[str, float] = {}
    batch_n = min(64, max(1, games_per_opponent))
    # Assign exact per-slot quotas so slow games are never dropped (zero fast-game bias).
    base_q, rem = divmod(games_per_opponent, batch_n)
    quotas = np.asarray([base_q + (1 if i < rem else 0) for i in range(batch_n)], dtype=np.int64)
    for opp_idx, opp_name in enumerate(opponents):
        if opp_name not in rl.BUILTIN_POLICIES:
            continue
        env = rl.VecEnv(
            "championsvgc",
            n=batch_n,
            teams=holdout_pool,
            threads=threads,
            seed=seed + opp_idx * 1009,
            opponent=opp_name,
            ots=0.0,
            preview="learned",
            masks="factored",
            observation="rich",
            side_randomization=True,
        )
        step = env.reset()
        done_per_env = np.zeros(batch_n, dtype=np.int64)
        points = 0.0
        while int(done_per_env.sum()) < games_per_opponent:
            needs = step.needs_action[:, 0] & (done_per_env < quotas)
            act = np.full((batch_n, 2), -1, dtype=np.int64)
            idx = np.flatnonzero(needs)
            if len(idx) > 0:
                a0, a1, _, _, _ = _select_batch_actions(
                    model,
                    env.space,
                    step.obs[idx, 0],
                    step.masks[idx, 0],
                    step.min_switches[idx, 0],
                    deterministic=True,
                )
                act[idx, 0] = a0
                act[idx, 1] = a1
            # Still step completed envs with random action if they wait for slower peers.
            idle_needs = step.needs_action[:, 0] & (done_per_env >= quotas)
            if idle_needs.any():
                act[idle_needs, 0] = rl.RANDOM_ACTION
                act[idle_needs, 1] = rl.RANDOM_ACTION
            step = env.step(act)
            finished = np.flatnonzero(step.done)
            for i in finished:
                if done_per_env[i] >= quotas[i]:
                    continue
                points += (float(step.reward[i, 0]) + 1.0) * 0.5
                done_per_env[i] += 1
        results[f"eval_win_rate_{opp_name}"] = round(points / max(games_per_opponent, 1), 4)
    return results


@torch.no_grad()
def eval_vs_python_vgc(
    model: RustFactoredPolicy,
    holdout_teams: list[str],
    *,
    games: int,
    seed: int,
) -> float:
    """Evaluate `RustFactoredPolicy` against our Python `vgc` 2-ply heuristic search."""
    from vgc.rl.agents import make_direct_agent
    from vgc.rl.env import DirectBattle, choice_string
    from vgc.rl.replica_engine import ReplicaRustWorker

    eval_rng = random.Random(seed)
    space = pikapro.ActionSpace("championsvgc")
    points = 0.0
    with ReplicaRustWorker() as worker:
        for game_idx in range(games):
            t_a, t_b = eval_rng.sample(holdout_teams, k=2)
            learner_is_p1 = game_idx % 2 == 0
            p1_team, p2_team = (t_a, t_b) if learner_is_p1 else (t_b, t_a)
            learner_side = "p1" if learner_is_p1 else "p2"
            opp_side = "p2" if learner_is_p1 else "p1"
            opp_team = p2_team if learner_is_p1 else p1_team
            learner_team = p1_team if learner_is_p1 else p2_team
            sim_seed = [eval_rng.randrange(2**16) for _ in range(4)]

            opp = make_direct_agent("vgc", opp_team)
            sess = rl.ChannelSession(
                "championsvgc",
                observation="rich",
                masks="factored",
                team=learner_team,
                username=learner_side,
            )
            battle = DirectBattle.start(
                worker, f"rust-eval-vgc-{game_idx}", p1_team, p2_team, seed=sim_seed
            )
            try:
                opp.observe(f"rust-eval-vgc-{game_idx}", battle.last_lines[opp_side])
                sess.feed_many(battle.last_lines[learner_side])
                while not battle.ended:
                    to_move = battle.sides_to_move()
                    choices: dict[str, str] = {}
                    for side in to_move:
                        if side == opp_side:
                            view = battle.battles[side]
                            view._vgc_direct_root = battle
                            view._vgc_direct_side = side
                            choices[side] = opp.choose(view)
                        else:
                            if not sess.needs_action:
                                choices[side] = "default"
                                continue
                            mask = sess.mask
                            a0, a1, _, _, is_pv = _select_batch_actions(
                                model,
                                space,
                                sess.obs[None, :],
                                mask[None, :],
                                np.asarray([sess.min_switches], dtype=np.uint8),
                                deterministic=True,
                            )
                            if is_pv[0]:
                                choices[side] = choice_string(sess.choice([int(a0[0])]))
                            else:
                                choices[side] = choice_string(sess.choice([int(a0[0]), int(a1[0])]))
                    res = battle.step(choices)
                    opp.observe(f"rust-eval-vgc-{game_idx}", res.lines[opp_side])
                    sess.feed_many(res.lines[learner_side])
                if battle.winner is None:
                    points += 0.5
                elif battle.winner == learner_side:
                    points += 1.0
            finally:
                battle.close()
    return round(points / max(games, 1), 4)


def save_trainer_state(
    out: Path,
    *,
    model: RustFactoredPolicy,
    optimizer: torch.optim.Optimizer,
    completed: int,
    snapshot_index: int,
    league: list[Snapshot],
    rng: random.Random,
    wandb_run_id: str | None,
) -> Path:
    path = out / TRAINER_STATE_FILE
    tmp = path.with_suffix(".tmp")
    payload = {
        "policy_version": model.policy_version,
        "ffn_mult": model.ffn_mult,
        "width": model.width,
        "layers": model.layers,
        "dex_sizes": model.dex_sizes,
        "completed": completed,
        "snapshot_index": snapshot_index,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "league": [
            {
                "game_count": s.game_count,
                "path": s.path.name,
                "games": float(s.games),
                "learner_points": float(s.learner_points),
            }
            for s in league
        ],
        "wandb_run_id": wandb_run_id,
    }
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def train(args: argparse.Namespace) -> None:
    enforce_mechanics_gate_for_cli("model training")
    enforce_battle_state_gate_for_cli("model training")
    enforce_action_gate_for_cli("model training")

    train_teams = load_teams(args.team_manifest, use_weights=args.use_manifest_weights)
    train_pool = rl.TeamPool(train_teams, "championsvgc")
    holdout_teams = load_teams(args.holdout_manifest) if args.eval_every > 0 else []
    holdout_pool = rl.TeamPool(holdout_teams, "championsvgc") if holdout_teams else None

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # Resolve policy_version & ffn_mult consistently for fresh, --init-from, and --resume runs.
    resolved_ver = int(args.policy_version)
    resolved_ffn = int(args.ffn_mult) if int(args.ffn_mult) > 0 else None
    resume_state: dict[str, Any] | None = None
    init_model: RustFactoredPolicy | None = None
    if args.resume:
        state_path = args.out / TRAINER_STATE_FILE
        if not state_path.is_file():
            raise FileNotFoundError(f"cannot resume: missing {state_path}")
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if resume_state["width"] != args.width or resume_state["layers"] != args.layers:
            raise ValueError("resume width/layers mismatch")
        resolved_ver = int(resume_state.get("policy_version", 2))
        resolved_ffn = int(resume_state.get("ffn_mult", 3 if resolved_ver >= 3 else 2))
    elif args.init_from is not None:
        init_model = load_rust_policy(args.init_from, device=device)
        if init_model.width != args.width or init_model.layers != args.layers:
            raise ValueError(
                f"--init-from width/layers ({init_model.width}x{init_model.layers}) "
                f"does not match ({args.width}x{args.layers})"
            )
        resolved_ver = init_model.policy_version
        resolved_ffn = init_model.ffn_mult

    model = (
        RustFactoredPolicy(
            args.width,
            args.layers,
            policy_version=resolved_ver,
            ffn_mult=resolved_ffn,
        )
        .to(device)
        .eval()
    )
    optimizer = build_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)

    wandb_run_id: str | None = None
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        try:
            optimizer.load_state_dict(resume_state["optimizer"])
        except ValueError:
            # Backwards-compat: older trainer_state had 1 AdamW param group instead of 2.
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            optimizer.load_state_dict(resume_state["optimizer"])
        for g in optimizer.param_groups:
            g["lr"] = args.learning_rate
        rng.setstate(resume_state["rng_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        completed = int(resume_state["completed"])
        snapshot_index = int(resume_state["snapshot_index"])
        wandb_run_id = resume_state.get("wandb_run_id")
        league = [
            Snapshot(
                game_count=int(e["game_count"]),
                path=args.out / e["path"],
                games=float(e["games"]),
                learner_points=float(e["learner_points"]),
            )
            for e in resume_state["league"]
        ]
        metrics_mode = "a"
    else:
        if init_model is not None:
            model.load_state_dict(init_model.state_dict())
            del init_model
        args.out.mkdir(parents=True, exist_ok=False)
        snap0 = save_rust_policy(args.out / f"policy_{0:07d}.pt", model)
        league = [Snapshot(0, snap0)]
        completed = 0
        snapshot_index = 0
        metrics_mode = "w"

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.out.name,
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            mode=args.wandb_mode,
            config={
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "resolved_device": device,
                "format": FORMAT_ID,
                "pikapro_version": version("pikapro"),
                "team_count": len(train_pool),
                "parameter_count": sum(p.numel() for p in model.parameters()),
            },
        )
        wandb_run_id = wandb_run.id

    if not args.resume:
        save_trainer_state(
            args.out,
            model=model,
            optimizer=optimizer,
            completed=completed,
            snapshot_index=snapshot_index,
            league=league,
            rng=rng,
            wandb_run_id=wandb_run_id,
        )
        args.out.joinpath("settings.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "resolved_device": device,
                    "format": FORMAT_ID,
                    "pikapro_version": version("pikapro"),
                    "team_count": len(train_pool),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "wandb_run_id": wandb_run_id,
                },
                default=str,
                indent=2,
            )
            + "\n"
        )

    # Partition parallel VecEnv slots: self-play, prioritized snapshot, native Rust anchor.
    num_envs = int(args.num_envs)
    n_anchor = int(round(num_envs * args.anchor_fraction)) if args.anchor_opponent else 0
    n_rem = num_envs - n_anchor
    n_self = n_rem // 2
    n_snap = n_rem - n_self
    self_slice = slice(0, n_self)
    snap_slice = slice(n_self, n_self + n_snap)
    anchor_slice = slice(n_self + n_snap, num_envs)

    env = rl.VecEnv(
        "championsvgc",
        n=num_envs,
        teams=train_pool,
        threads=args.threads,
        seed=args.seed,
        opponent="external",
        ots=0.0,
        preview="learned",
        masks="factored",
        observation="rich",
        autoreset="next_step",
        side_randomization=True,
    )
    if n_anchor > 0:
        env.set_opponent(args.anchor_opponent, which=list(range(anchor_slice.start, anchor_slice.stop)))

    step = env.reset()
    trajs_p0: list[list[StepRecord]] = [[] for _ in range(num_envs)]
    trajs_p1: list[list[StepRecord]] = [[] for _ in range(num_envs)]
    snap_cache: dict[str, RustFactoredPolicy] = {}
    snap_cache_cap = max(12, int(args.snapshots_per_wave) * 2)

    def _get_snapshot_model(snap: Snapshot) -> RustFactoredPolicy:
        key = str(snap.path)
        m = snap_cache.get(key)
        if m is None:
            m = load_rust_policy(snap.path, device=device)
            if len(snap_cache) >= snap_cache_cap:
                snap_cache.pop(next(iter(snap_cache)))
            snap_cache[key] = m
        return m

    started = time.monotonic()
    last_eval = completed
    last_vgc_eval = completed

    try:
        with args.out.joinpath("metrics.jsonl").open(metrics_mode) as metrics_f:
            while completed < args.games:
                step_started = time.monotonic()
                target_batch_games = min(args.games - completed, args.games_per_update)

                # Partition `snap_slice` across K prioritized snapshots per update wave
                # (instead of 1 single snapshot for all 115 league envs) to reduce gradient variance.
                snap_groups: list[tuple[slice, Snapshot, RustFactoredPolicy]] = []
                env_to_snap: dict[int, Snapshot] = {}
                if n_snap > 0 and league:
                    k_snaps = max(1, min(int(args.snapshots_per_wave), len(league), n_snap))
                    picked = rng.choices(league, weights=[s.priority for s in league], k=k_snaps)
                    # Deduplicate while preserving multi-slot weight proportionally.
                    by_path: dict[str, tuple[Snapshot, int]] = {}
                    for sp_item in picked:
                        pk = str(sp_item.path)
                        if pk in by_path:
                            by_path[pk] = (sp_item, by_path[pk][1] + 1)
                        else:
                            by_path[pk] = (sp_item, 1)
                    offset = snap_slice.start
                    rem_slots = n_snap
                    rem_weight = k_snaps
                    for sp_item, w_cnt in by_path.values():
                        n_alloc = (
                            rem_slots
                            if w_cnt >= rem_weight
                            else max(1, round(rem_slots * (w_cnt / rem_weight)))
                        )
                        n_alloc = min(n_alloc, rem_slots)
                        sub_sl = slice(offset, offset + n_alloc)
                        snap_groups.append((sub_sl, sp_item, _get_snapshot_model(sp_item)))
                        for ei_s in range(sub_sl.start, sub_sl.stop):
                            env_to_snap[ei_s] = sp_item
                        offset += n_alloc
                        rem_slots -= n_alloc
                        rem_weight -= w_cnt
                        if rem_slots <= 0:
                            break

                records: list[tuple[StepRecord, float, float]] = []
                batch_games_done = 0
                vs_self_games = 0
                vs_snap_games = 0
                vs_snap_points = 0.0
                vs_anchor_games = 0
                vs_anchor_points = 0.0
                draws = 0
                turns_sum = 0

                while batch_games_done < target_batch_games:
                    actions = np.full((num_envs, 2, 2), -1, dtype=np.int64)

                    # 1. Batch all learner decisions: player 0 everywhere + player 1 in self-play.
                    p0_idx = np.flatnonzero(step.needs_action[:, 0])
                    p1_self_idx = (
                        np.flatnonzero(step.needs_action[self_slice, 1]) if n_self > 0 else np.empty(0, dtype=np.int64)
                    )
                    n0 = len(p0_idx)
                    n1 = len(p1_self_idx)
                    if n0 + n1 > 0:
                        obs_list = []
                        mask_list = []
                        ms_list = []
                        if n0 > 0:
                            obs_list.append(step.obs[p0_idx, 0])
                            mask_list.append(step.masks[p0_idx, 0])
                            ms_list.append(step.min_switches[p0_idx, 0])
                        if n1 > 0:
                            obs_list.append(step.obs[p1_self_idx, 1])
                            mask_list.append(step.masks[p1_self_idx, 1])
                            ms_list.append(step.min_switches[p1_self_idx, 1])
                        obs_cat = np.concatenate(obs_list, axis=0)
                        mask_cat = np.concatenate(mask_list, axis=0)
                        ms_cat = np.concatenate(ms_list, axis=0)
                        a0, a1, lp, val, is_pv = _select_batch_actions(
                            model, env.space, obs_cat, mask_cat, ms_cat, deterministic=False
                        )
                        if n0 > 0:
                            actions[p0_idx, 0, 0] = a0[:n0]
                            actions[p0_idx, 0, 1] = a1[:n0]
                            for k, ei in enumerate(p0_idx):
                                trajs_p0[ei].append(
                                    StepRecord(
                                        obs=obs_cat[k].copy(),
                                        mask=mask_cat[k].copy(),
                                        min_switches=int(ms_cat[k]),
                                        is_preview=bool(is_pv[k]),
                                        a0=int(a0[k]),
                                        a1=int(a1[k]),
                                        log_prob=float(lp[k]),
                                        value=float(val[k]),
                                    )
                                )
                        if n1 > 0:
                            actions[p1_self_idx, 1, 0] = a0[n0:]
                            actions[p1_self_idx, 1, 1] = a1[n0:]
                            for k, ei in enumerate(p1_self_idx):
                                idx_k = n0 + k
                                trajs_p1[ei].append(
                                    StepRecord(
                                        obs=obs_cat[idx_k].copy(),
                                        mask=mask_cat[idx_k].copy(),
                                        min_switches=int(ms_cat[idx_k]),
                                        is_preview=bool(is_pv[idx_k]),
                                        a0=int(a0[idx_k]),
                                        a1=int(a1[idx_k]),
                                        log_prob=float(lp[idx_k]),
                                        value=float(val[idx_k]),
                                    )
                                )

                    # 2. Batch all historical snapshot opponent decisions (per snapshot group).
                    for sub_sl, _sp_obj, sp_model in snap_groups:
                        snap_local = np.flatnonzero(step.needs_action[sub_sl, 1])
                        if len(snap_local) > 0:
                            snap_idx = snap_local + sub_sl.start
                            sa0, sa1, _, _, _ = _select_batch_actions(
                                sp_model,
                                env.space,
                                step.obs[snap_idx, 1],
                                step.masks[snap_idx, 1],
                                step.min_switches[snap_idx, 1],
                                deterministic=False,
                            )
                            actions[snap_idx, 1, 0] = sa0
                            actions[snap_idx, 1, 1] = sa1

                    step = env.step(actions)
                    finished = np.flatnonzero(step.done)
                    for ei in finished:
                        r0 = float(step.reward[ei, 0])
                        r1 = float(step.reward[ei, 1])
                        _finish_trajectory(
                            trajs_p0[ei],
                            r0,
                            gamma=args.gamma,
                            gae_lambda=args.gae_lambda,
                            out_records=records,
                        )
                        trajs_p0[ei].clear()
                        if ei < n_self:
                            _finish_trajectory(
                                trajs_p1[ei],
                                r1,
                                gamma=args.gamma,
                                gae_lambda=args.gae_lambda,
                                out_records=records,
                            )
                            trajs_p1[ei].clear()
                            vs_self_games += 1
                        elif ei < n_self + n_snap:
                            pts = (r0 + 1.0) * 0.5
                            vs_snap_games += 1
                            vs_snap_points += pts
                            snap_owner = env_to_snap.get(int(ei))
                            if snap_owner is not None:
                                snap_owner.record_game(pts)
                        else:
                            vs_anchor_games += 1
                            vs_anchor_points += (r0 + 1.0) * 0.5
                        draws += int(r0 == 0.0)
                        # Approximate turn count from field.turn column (= turn / 100)
                        turns_sum += int(round(float(step.obs[ei, 0, RICH_FIELD_TURN_IDX]) * 100.0))
                        batch_games_done += 1

                batch = _collate_samples(records)
                non_draws = batch.returns != 0.0
                value_acc = (
                    float(np.mean((batch.values[non_draws] > 0.0) == (batch.returns[non_draws] > 0.0)))
                    if np.any(non_draws)
                    else None
                )

                current_lr = scheduled_lr(
                    completed,
                    args.games,
                    max_lr=args.learning_rate,
                    min_lr=args.min_learning_rate,
                    schedule=args.lr_schedule,
                )
                for group in optimizer.param_groups:
                    group["lr"] = current_lr

                ppo = ppo_update(
                    model,
                    optimizer,
                    env.space,
                    batch,
                    rng,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    target_kl=args.target_kl,
                    entropy_coef=args.entropy_coef,
                )
                completed += batch_games_done

                if completed >= snapshot_index + args.snapshot_games or completed >= args.games:
                    snap_path = save_rust_policy(args.out / f"policy_{completed:07d}.pt", model)
                    league.append(Snapshot(completed, snap_path))
                    league = league[-args.league_size :]
                    snapshot_index = completed
                    save_trainer_state(
                        args.out,
                        model=model,
                        optimizer=optimizer,
                        completed=completed,
                        snapshot_index=snapshot_index,
                        league=league,
                        rng=rng,
                        wandb_run_id=wandb_run_id,
                    )

                eval_metrics: dict[str, float] = {}
                if (
                    holdout_pool is not None
                    and args.eval_every > 0
                    and (completed - last_eval >= args.eval_every or completed >= args.games)
                ):
                    eval_opps = [o.strip() for o in args.eval_opponents.split(",") if o.strip()]
                    eval_metrics.update(
                        eval_rust_opponents(
                            model,
                            holdout_pool,
                            eval_opps,
                            games_per_opponent=args.eval_games,
                            threads=args.threads,
                            seed=args.seed + 900001,
                        )
                    )
                    last_eval = completed

                if (
                    holdout_teams
                    and args.vgc_eval_every > 0
                    and (completed - last_vgc_eval >= args.vgc_eval_every or completed >= args.games)
                ):
                    eval_metrics["eval_win_rate_vgc"] = eval_vs_python_vgc(
                        model,
                        holdout_teams,
                        games=args.vgc_eval_games,
                        seed=args.seed + 910001,
                    )
                    last_vgc_eval = completed

                step_elapsed = max(time.monotonic() - step_started, 1e-6)
                row: dict[str, object] = {
                    "games": completed,
                    "batch_games": batch_games_done,
                    "vs_current": vs_self_games,
                    "vs_snapshots_games": vs_snap_games,
                    "vs_snapshots_score": round(vs_snap_points / vs_snap_games, 4)
                    if vs_snap_games
                    else None,
                    "vs_anchors": vs_anchor_games,
                    "vs_anchors_score": round(vs_anchor_points / vs_anchor_games, 4)
                    if vs_anchor_games
                    else None,
                    "draws": draws,
                    "mean_turns": round(turns_sum / max(batch_games_done, 1), 2),
                    "decisions": len(batch.obs),
                    "preview_decisions": int(batch.is_preview.sum()),
                    "training_loss": ppo["loss"],
                    "policy_loss": ppo["policy_loss"],
                    "value_loss": ppo["value_loss"],
                    "entropy": ppo["entropy"],
                    "approx_kl": ppo["approx_kl"],
                    "ppo_epochs_run": int(ppo["epochs_run"]),
                    "learning_rate": current_lr,
                    "value_accuracy": round(value_acc, 4) if value_acc is not None else None,
                    "games_per_sec": round(batch_games_done / step_elapsed, 2),
                    "decisions_per_sec": round(len(batch.obs) / step_elapsed, 1),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    **eval_metrics,
                }
                line = json.dumps(row, sort_keys=True)
                print(line, flush=True)
                metrics_f.write(line + "\n")
                metrics_f.flush()

                if wandb_run is not None:
                    payload: dict[str, float] = {
                        "train/loss": ppo["loss"],
                        "train/policy_loss": ppo["policy_loss"],
                        "train/value_loss": ppo["value_loss"],
                        "train/entropy": ppo["entropy"],
                        "train/approx_kl": ppo["approx_kl"],
                        "train/ppo_epochs_run": ppo["epochs_run"],
                        "train/learning_rate": current_lr,
                        "train/mean_turns": turns_sum / max(batch_games_done, 1),
                        "perf/games_per_sec": batch_games_done / step_elapsed,
                        "perf/decisions_per_sec": len(batch.obs) / step_elapsed,
                    }
                    if value_acc is not None:
                        payload["train/value_accuracy"] = value_acc
                    if vs_snap_games:
                        payload["train/vs_snapshots_score"] = vs_snap_points / vs_snap_games
                    if vs_anchor_games:
                        payload["train/vs_anchors_score"] = vs_anchor_points / vs_anchor_games
                    for k, v in eval_metrics.items():
                        payload[f"eval/{k.removeprefix('eval_')}"] = v
                    wandb_run.log(payload, step=completed)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="warm-start weights from an existing RustFactoredPolicy checkpoint into a new --out run",
    )
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--use-manifest-weights",
        action="store_true",
        help="respect optional integer `weight` fields in --team-manifest for meta-frequency sampling",
    )
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--games", type=int, default=1_000_000)
    parser.add_argument("--games-per-update", type=int, default=1024)
    parser.add_argument("--snapshot-games", type=int, default=16384)
    parser.add_argument("--league-size", type=int, default=24)
    parser.add_argument(
        "--snapshots-per-wave",
        type=int,
        default=2,
        help="number of prioritized league snapshots sampled in parallel per update wave",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument(
        "--policy-version",
        type=int,
        default=3,
        choices=(2, 3),
        help="RustFactoredPolicy version (3 = static move meta + GPU speed cues + symmetric preview)",
    )
    parser.add_argument(
        "--ffn-mult",
        type=int,
        default=0,
        help="Transformer FFN width multiplier (0 = auto: 3 for v3, 2 for v2)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.025)
    parser.add_argument("--anchor-opponent", default="simple_heuristics")
    parser.add_argument("--anchor-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=37301)
    parser.add_argument("--eval-every", type=int, default=16384)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-opponents", default="max_base_power,simple_heuristics")
    parser.add_argument("--vgc-eval-every", type=int, default=65536)
    parser.add_argument("--vgc-eval-games", type=int, default=40)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    raise SystemExit(main())
