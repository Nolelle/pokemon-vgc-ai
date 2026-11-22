"""Train a fresh Champions policy and value network by playing its own league."""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import math
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vgc.actions import enumerate_joint_orders
from vgc.action_gate import enforce_action_gate_for_cli
from vgc.battle_state_gate import enforce_battle_state_gate_for_cli
from vgc.config import FORMAT_ID
from vgc.mechanics_gate import enforce_mechanics_gate_for_cli
from vgc.rl.agents import make_direct_agent
from vgc.rl.env import DirectBattle, SimWorker, choice_string
from vgc.rl.match import MAX_DECISIONS, play_battle
from vgc.rl.replication_policy import (
    CHECKPOINT_VERSION,
    EncodedDecision,
    ReplicationPolicy,
    encode_decision,
    load_replication_policy,
    preview_order,
    stack_decisions,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "data/selfplay/mc_sheet_pool_v2/train_manifest.json"
DEFAULT_HOLDOUT_MANIFEST = PROJECT / "data/selfplay/mc_sheet_pool_v2/holdout_manifest.json"
TRAINER_STATE_FILE = "trainer_state.pt"

_WORKER_STATE: dict[str, object] = {}


def resolve_profile(
    profile: str,
    *,
    workers: int | None = None,
    threads: int | None = None,
    cpu_count: int | None = None,
) -> tuple[int, int, int]:
    """Return ``(workers, threads, nice)`` for ``desktop`` or ``afk`` mode."""
    cpus = max(1, cpu_count or os.cpu_count() or 4)
    if profile == "desktop":
        default_workers = max(1, cpus // 2)
        default_threads = min(2, cpus)
        nice = 10
    elif profile == "afk":
        default_workers = max(1, cpus - 2)
        default_threads = min(4, cpus)
        nice = 2
    else:
        raise ValueError(f"unknown resource profile {profile!r}")
    return (
        workers if workers is not None else default_workers,
        threads if threads is not None else default_threads,
        nice,
    )


def scheduled_lr(
    completed: int,
    total_games: int,
    *,
    max_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    """Compute the learning rate for the current game count."""
    if schedule == "constant":
        return max_lr
    if schedule == "cosine":
        progress = min(max(completed / max(total_games, 1), 0.0), 1.0)
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"unknown lr_schedule {schedule!r}")


@dataclass(frozen=True)
class Decision:
    encoded: EncodedDecision
    index: int
    log_prob: float
    value: float


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return requested


def compute_trajectory_samples(
    steps: list[Decision],
    reward: float,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> list[tuple[Decision, float, float]]:
    """Compute per-step (decision, terminal_return, gae_advantage) for one player's episode."""
    n = len(steps)
    if n == 0:
        return []
    advantages = [0.0] * n
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_value = 0.0 if t == n - 1 else steps[t + 1].value
        step_reward = reward if t == n - 1 else 0.0
        delta = step_reward + gamma * next_value - steps[t].value
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
    return [(steps[t], reward, advantages[t]) for t in range(n)]


class LeagueAgent:
    def __init__(self, name, model, seed, *, record, deterministic=False):
        self.name = name
        self.model = model
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.steps: list[Decision] = []
        self.record = record
        self.deterministic = deterministic

    @torch.no_grad()
    def choose(self, battle):
        preview = battle.teampreview
        orders = None if preview else enumerate_joint_orders(battle)
        if orders == []:
            return "default"
        encoded = encode_decision(battle, orders, preview=preview, version=self.model.version)
        logits, value = self.model(*stack_decisions([encoded], self.model.device), preview=preview)
        log_probs = logits[0].log_softmax(dim=0)
        if self.deterministic:
            index = int(logits[0].argmax().item())
        else:
            index = int(torch.multinomial(log_probs.exp().cpu(), 1, generator=self.rng).item())
        if self.record:
            self.steps.append(
                Decision(encoded, index, float(log_probs[index]), float(value.item()))
            )
        return choice_string(preview_order(encoded.candidates[index]) if preview else orders[index])

    def observe(self, battle_tag, lines):
        pass

    def finish(self, battle):
        pass


@dataclass
class Snapshot:
    game_count: int
    path: Path
    model: ReplicationPolicy | None = None
    games: int = 0
    learner_points: float = 0.0

    @property
    def priority(self):
        # A Beta(1, 1) starting point avoids discarding new opponents before play.
        win_rate = (1 + self.learner_points) / (2 + self.games)
        return (1 - win_rate) ** 2


def _make_worker(engine: str):
    if engine == "pikapro":
        from vgc.rl.replica_engine import ReplicaRustWorker

        return ReplicaRustWorker()
    return SimWorker()


def _init_rollout_worker(engine: str, width: int, layers: int, version_num: int, nice: int) -> None:
    if nice > 0 and hasattr(os, "nice"):
        try:
            os.nice(nice)
        except OSError:
            pass
    torch.set_num_threads(1)
    worker = _make_worker(engine)
    worker.__enter__()
    atexit.register(worker.__exit__, None, None, None)
    _WORKER_STATE.clear()
    _WORKER_STATE["worker"] = worker
    _WORKER_STATE["learner"] = ReplicationPolicy(width, layers, version=version_num).eval()
    _WORKER_STATE["snapshots"] = {}


def _load_cached_snapshot(path_str: str) -> ReplicationPolicy:
    cache: dict[str, ReplicationPolicy] = _WORKER_STATE["snapshots"]  # type: ignore[assignment]
    model = cache.get(path_str)
    if model is None:
        model = load_replication_policy(path_str, device="cpu")
        if len(cache) >= 32:
            cache.pop(next(iter(cache)))
        cache[path_str] = model
    return model


@dataclass
class _ActiveGame:
    battle_id: str
    battle: DirectBattle
    learner_side: str
    opp_side: str
    opp_kind: str  # "self", "snapshot", or "anchor"
    opp_model_key: str | None  # None for learner/self, path for snapshot
    opp_anchor: object | None
    snap_game_count: int | None
    anchor_name: str | None
    learner_rng: torch.Generator
    opp_rng: torch.Generator
    learner_steps: list[Decision]
    opp_steps: list[Decision]
    decisions: int = 0


@torch.no_grad()
def _play_rollout_jobs(
    worker,
    learner_model: ReplicationPolicy,
    jobs: list[tuple],
    snapshot_loader,
    *,
    rollout_batch_size: int = 16,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[tuple[Decision, float, float]], list[tuple[float, int, int | None, str | None]]]:
    """Run up to ``rollout_batch_size`` battles concurrently and batch neural forward passes."""
    samples: list[tuple[Decision, float, float]] = []
    summaries: list[tuple[float, int, int | None, str | None]] = []
    batch_limit = max(1, int(rollout_batch_size))
    queue = list(jobs)
    active: list[_ActiveGame] = []

    def _start_next() -> None:
        (
            game_idx,
            team_p1,
            team_p2,
            learner_is_p1,
            sim_seed,
            learner_seed,
            opponent_seed,
            snap_game_count,
            snap_path,
            anchor_name,
        ) = queue.pop(0)
        battle_id = f"replication-{game_idx}"
        learner_side = "p1" if learner_is_p1 else "p2"
        opp_side = "p2" if learner_is_p1 else "p1"
        battle = DirectBattle.start(worker, battle_id, team_p1, team_p2, seed=sim_seed)
        if anchor_name is not None:
            opp_kind = "anchor"
            opp_team = team_p2 if learner_is_p1 else team_p1
            opp_anchor = make_direct_agent(anchor_name, opp_team)
            opp_anchor.observe(battle_id, battle.last_lines[opp_side])
            opp_model_key = None
        elif snap_path is not None:
            opp_kind = "snapshot"
            opp_anchor = None
            opp_model_key = snap_path
        else:
            opp_kind = "self"
            opp_anchor = None
            opp_model_key = None
        active.append(
            _ActiveGame(
                battle_id=battle_id,
                battle=battle,
                learner_side=learner_side,
                opp_side=opp_side,
                opp_kind=opp_kind,
                opp_model_key=opp_model_key,
                opp_anchor=opp_anchor,
                snap_game_count=snap_game_count,
                anchor_name=anchor_name,
                learner_rng=torch.Generator(device="cpu").manual_seed(learner_seed),
                opp_rng=torch.Generator(device="cpu").manual_seed(opponent_seed),
                learner_steps=[],
                opp_steps=[],
            )
        )

    try:
        while queue or active:
            while queue and len(active) < batch_limit:
                _start_next()

            # Collect all pending choices across active games, batching neural calls by
            # (model_key, is_preview).
            choices_by_game: list[dict[str, str]] = [{} for _ in active]
            neural_groups: dict[tuple[str | None, bool], list[tuple]] = {}

            for g_idx, game in enumerate(active):
                to_move = game.battle.sides_to_move()
                for side in to_move:
                    view = game.battle.battles[side]
                    is_learner = side == game.learner_side
                    if not is_learner and game.opp_kind == "anchor":
                        view._vgc_direct_root = game.battle
                        view._vgc_direct_side = side
                        choices_by_game[g_idx][side] = game.opp_anchor.choose(view)  # type: ignore[union-attr]
                        continue
                    preview = bool(view.teampreview)
                    orders = None if preview else enumerate_joint_orders(view)
                    if orders == []:
                        choices_by_game[g_idx][side] = "default"
                        continue
                    model_key = None if (is_learner or game.opp_kind == "self") else game.opp_model_key
                    model = learner_model if model_key is None else snapshot_loader(model_key)
                    encoded = encode_decision(view, orders, preview=preview, version=model.version)
                    record = is_learner or game.opp_kind == "self"
                    rng = game.learner_rng if is_learner else game.opp_rng
                    steps_list = game.learner_steps if is_learner else game.opp_steps
                    neural_groups.setdefault((model_key, preview), []).append(
                        (g_idx, side, encoded, orders, record, rng, steps_list)
                    )

            for (model_key, preview), items in neural_groups.items():
                model = learner_model if model_key is None else snapshot_loader(model_key)
                batch_encoded = [item[2] for item in items]
                logits, values = model(
                    *stack_decisions(batch_encoded, model.device), preview=preview
                )
                logits_cpu = logits.cpu()
                values_cpu = values.cpu()
                for row_idx, (g_idx, side, encoded, orders, record, rng, steps_list) in enumerate(
                    items
                ):
                    n_cands = len(encoded.candidates)
                    row_logits = logits_cpu[row_idx, :n_cands]
                    log_probs = row_logits.log_softmax(dim=0)
                    probs = log_probs.exp()
                    index = int(torch.multinomial(probs, 1, generator=rng).item())
                    if record:
                        steps_list.append(
                            Decision(
                                encoded,
                                index,
                                float(log_probs[index].item()),
                                float(values_cpu[row_idx].item()),
                            )
                        )
                    choice = (
                        preview_order(encoded.candidates[index]) if preview else orders[index]
                    )
                    choices_by_game[g_idx][side] = choice_string(choice)

            still_active: list[_ActiveGame] = []
            for g_idx, game in enumerate(active):
                choices = choices_by_game[g_idx]
                game.decisions += len(choices)
                result = game.battle.step(choices)
                if game.opp_anchor is not None:
                    game.opp_anchor.observe(game.battle_id, result.lines[game.opp_side])  # type: ignore[union-attr]
                if game.decisions > MAX_DECISIONS:
                    raise RuntimeError(
                        f"battle {game.battle_id} exceeded {MAX_DECISIONS} decisions"
                    )
                if game.battle.ended:
                    if game.opp_anchor is not None:
                        game.opp_anchor.finish(game.battle.battles[game.opp_side])  # type: ignore[union-attr]
                    winner = game.battle.winner
                    if winner is None:
                        reward = 0.0
                    else:
                        reward = 1.0 if winner == game.learner_side else -1.0
                    turns = int(getattr(game.battle.battles["p1"], "turn", 0) or 0)
                    samples.extend(
                        compute_trajectory_samples(
                            game.learner_steps, reward, gamma=gamma, gae_lambda=gae_lambda
                        )
                    )
                    if game.opp_steps:
                        samples.extend(
                            compute_trajectory_samples(
                                game.opp_steps, -reward, gamma=gamma, gae_lambda=gae_lambda
                            )
                        )
                    summaries.append((reward, turns, game.snap_game_count, game.anchor_name))
                    game.battle.close()
                else:
                    still_active.append(game)
            active = still_active
    finally:
        for game in active:
            game.battle.close()

    return samples, summaries


def _run_rollout_chunk(
    learner_state: dict[str, torch.Tensor],
    jobs: list[tuple],
    rollout_batch_size: int = 16,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[tuple[Decision, float, float]], list[tuple[float, int, int | None, str | None]]]:
    learner_model: ReplicationPolicy = _WORKER_STATE["learner"]  # type: ignore[assignment]
    learner_model.load_state_dict(learner_state)
    learner_model.eval()
    return _play_rollout_jobs(
        _WORKER_STATE["worker"],
        learner_model,
        jobs,
        _load_cached_snapshot,
        rollout_batch_size=rollout_batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def _play_eval_jobs(
    worker,
    learner_model: ReplicationPolicy,
    jobs: list[tuple],
) -> list[tuple[str, float, int]]:
    outcomes: list[tuple[str, float, int]] = []
    for eval_idx, opponent_name, team_p1, team_p2, learner_is_p1, sim_seed in jobs:
        learner = LeagueAgent("learner", learner_model, 0, record=False, deterministic=True)
        opp_team = team_p2 if learner_is_p1 else team_p1
        opponent = make_direct_agent(opponent_name, opp_team)
        pair = (learner, opponent) if learner_is_p1 else (opponent, learner)
        result = play_battle(
            worker,
            f"eval-{opponent_name}-{eval_idx}",
            {"p1": pair[0], "p2": pair[1]},
            {"p1": team_p1, "p2": team_p2},
            seed=sim_seed,
        )
        outcomes.append((opponent_name, result.result_for("learner"), result.turns))
    return outcomes


def _run_eval_chunk(
    learner_state: dict[str, torch.Tensor],
    jobs: list[tuple],
) -> list[tuple[str, float, int]]:
    learner_model: ReplicationPolicy = _WORKER_STATE["learner"]  # type: ignore[assignment]
    learner_model.load_state_dict(learner_state)
    learner_model.eval()
    return _play_eval_jobs(_WORKER_STATE["worker"], learner_model, jobs)


def _split_chunks(items: list, n_chunks: int) -> list[list]:
    n = max(1, min(n_chunks, len(items)))
    return [items[i::n] for i in range(n) if items[i::n]]


def update_policy_with_metrics(
    model,
    optimizer,
    samples,
    rng,
    *,
    epochs,
    batch_size,
    target_kl: float = 0.025,
) -> dict[str, float]:
    """PPO with clipped likelihood ratios, GAE advantages, calibrated value loss, and KL stop."""
    if not samples:
        raise ValueError("cannot train without recorded decisions")
    device = model.device
    normalized_samples = [
        (item[0], float(item[1]), float(item[2]))
        if len(item) == 3
        else (item[0], float(item[1]), float(item[1]) - item[0].value)
        for item in samples
    ]
    advantages = np.asarray([adv for _, _, adv in normalized_samples], dtype=np.float32)
    advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
    groups = [
        [idx for idx, (step, _, _) in enumerate(normalized_samples) if step.encoded.preview == preview]
        for preview in (False, True)
    ]
    loss_sum = 0.0
    policy_sum = 0.0
    value_sum = 0.0
    entropy_sum = 0.0
    kl_sum = 0.0
    updates = 0
    epochs_run = 0
    use_bce_value = getattr(model, "version", 1) >= 2
    model.train()
    for _ in range(epochs):
        epochs_run += 1
        epoch_kl_sum = 0.0
        epoch_updates = 0
        batches = []
        for group in groups:
            rng.shuffle(group)
            batches.extend(
                group[start : start + batch_size] for start in range(0, len(group), batch_size)
            )
        rng.shuffle(batches)
        for rows in batches:
            steps = [normalized_samples[i][0] for i in rows]
            rewards = [normalized_samples[i][1] for i in rows]
            states = [step.encoded for step in steps]
            logits, value, value_logit = model(
                *stack_decisions(states, device),
                preview=states[0].preview,
                return_value_logit=True,
            )
            distribution = torch.distributions.Categorical(logits=logits)
            indices = torch.tensor([step.index for step in steps], dtype=torch.long, device=device)
            old = torch.tensor(
                [step.log_prob for step in steps], dtype=torch.float32, device=device
            )
            returns = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantage = torch.from_numpy(advantages[rows]).to(device)
            new_log_prob = distribution.log_prob(indices)
            log_ratio = new_log_prob - old
            ratio = log_ratio.exp()
            policy_loss = -torch.minimum(
                ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
            ).mean()
            if use_bce_value and value_logit is not None:
                win_prob_target = (returns + 1.0) * 0.5
                value_loss = F.binary_cross_entropy_with_logits(value_logit, win_prob_target)
            else:
                value_loss = F.mse_loss(value, returns)
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
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
        "loss": loss_sum / updates,
        "policy_loss": policy_sum / updates,
        "value_loss": value_sum / updates,
        "entropy": entropy_sum / updates,
        "approx_kl": kl_sum / updates,
        "epochs_run": float(epochs_run),
    }


def update_policy(model, optimizer, samples, rng, *, epochs, batch_size) -> float:
    """Backwards-compatible wrapper returning the scalar PPO training loss."""
    return update_policy_with_metrics(
        model, optimizer, samples, rng, epochs=epochs, batch_size=batch_size, target_kl=0.0
    )["loss"]


def load_teams(manifest: Path) -> list[str]:
    rows = json.loads(manifest.read_text())
    teams = list(dict.fromkeys((manifest.parent / row["file"]).read_text().strip() for row in rows))
    if len(teams) < 2 or any(not team for team in teams):
        raise ValueError(f"need at least two different nonempty packed teams in {manifest}")
    return teams


def save_snapshot(out: Path, model: ReplicationPolicy, games: int) -> Path:
    path = out / f"policy_{games:07d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(model.checkpoint(), temporary)
    temporary.replace(path)
    return path


def save_trainer_state(
    out: Path,
    *,
    model: ReplicationPolicy,
    optimizer: torch.optim.Optimizer,
    completed: int,
    snapshot_index: int,
    league: list[Snapshot],
    rng: random.Random,
    wandb_run_id: str | None,
) -> Path:
    path = out / TRAINER_STATE_FILE
    temporary = path.with_suffix(".tmp")
    payload = {
        "width": model.width,
        "layers": model.layers,
        "version": model.version,
        "completed": completed,
        "snapshot_index": snapshot_index,
        "model": _cpu_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "league": [
            {
                "game_count": s.game_count,
                "path": s.path.name,
                "games": s.games,
                "learner_points": s.learner_points,
            }
            for s in league
        ],
        "wandb_run_id": wandb_run_id,
    }
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _cpu_state_dict(model: ReplicationPolicy) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def _run_holdout_eval(
    *,
    model: ReplicationPolicy,
    holdout_teams: list[str],
    opponents: list[str],
    eval_games: int,
    eval_seed: int,
    pool: ProcessPoolExecutor | None,
    direct_worker,
    workers: int,
) -> dict[str, float]:
    eval_rng = random.Random(eval_seed)
    jobs: list[tuple] = []
    idx = 0
    for opponent_name in opponents:
        for pair_idx in range(eval_games):
            team_a, team_b = eval_rng.sample(holdout_teams, k=2)
            learner_is_p1 = pair_idx % 2 == 0
            team_p1, team_p2 = (team_a, team_b) if learner_is_p1 else (team_b, team_a)
            sim_seed = [eval_rng.randrange(2**16) for _ in range(4)]
            jobs.append((idx, opponent_name, team_p1, team_p2, learner_is_p1, sim_seed))
            idx += 1

    if pool is not None:
        learner_state = _cpu_state_dict(model)
        chunks = _split_chunks(jobs, workers)
        futures = [pool.submit(_run_eval_chunk, learner_state, chunk) for chunk in chunks]
        outcomes = [row for future in futures for row in future.result()]
    else:
        outcomes = _play_eval_jobs(direct_worker, model, jobs)

    metrics: dict[str, float] = {}
    for opponent_name in opponents:
        opp_rows = [row for row in outcomes if row[0] == opponent_name]
        if not opp_rows:
            continue
        score = sum((reward + 1.0) / 2.0 for _, reward, _ in opp_rows) / len(opp_rows)
        metrics[f"eval_win_rate_{opponent_name}"] = round(score, 4)
    return metrics


def train(args):
    enforce_mechanics_gate_for_cli("model training")
    enforce_battle_state_gate_for_cli("model training")
    enforce_action_gate_for_cli("model training")
    teams = load_teams(args.team_manifest)
    eval_opponents = [name.strip() for name in args.eval_opponents.split(",") if name.strip()]
    anchor_opponents = [name.strip() for name in args.anchor_opponents.split(",") if name.strip()]
    holdout_teams: list[str] = []
    if args.eval_every > 0 and args.eval_games > 0 and eval_opponents:
        holdout_teams = load_teams(args.holdout_manifest)

    workers, threads, nice = resolve_profile(
        args.profile, workers=args.workers, threads=args.threads
    )
    resolved_device = resolve_device(args.device)
    if nice > 0 and hasattr(os, "nice"):
        try:
            os.nice(nice)
        except OSError:
            pass
    torch.set_num_threads(threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    wandb_run_id: str | None = None
    if args.resume:
        state_path = args.out / TRAINER_STATE_FILE
        if not state_path.is_file():
            raise FileNotFoundError(f"cannot resume: missing {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["width"] != args.width or state["layers"] != args.layers:
            raise ValueError(
                f"resume mismatch: saved model is width={state['width']} layers={state['layers']}, "
                f"requested width={args.width} layers={args.layers}"
            )
        model_version = int(state.get("version", 1))
        model = (
            ReplicationPolicy(args.width, args.layers, version=model_version)
            .to(resolved_device)
            .eval()
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        rng.setstate(state["rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        completed = int(state["completed"])
        snapshot_index = int(state["snapshot_index"])
        wandb_run_id = state.get("wandb_run_id")
        league = []
        for entry in state["league"]:
            snap_path = args.out / entry["path"]
            snap_model = (
                load_replication_policy(snap_path, device=resolved_device)
                if workers == 1
                else None
            )
            league.append(
                Snapshot(
                    game_count=int(entry["game_count"]),
                    path=snap_path,
                    model=snap_model,
                    games=int(entry["games"]),
                    learner_points=float(entry["learner_points"]),
                )
            )
        metrics_mode = "a"
    else:
        model_version = int(args.model_version)
        model = (
            ReplicationPolicy(args.width, args.layers, version=model_version)
            .to(resolved_device)
            .eval()
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        args.out.mkdir(parents=True, exist_ok=False)
        snap0 = save_snapshot(args.out, model, 0)
        league = [Snapshot(0, snap0, copy.deepcopy(model) if workers == 1 else None)]
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
                "resolved_device": resolved_device,
                "model_version": model_version,
                "resolved_workers": workers,
                "resolved_threads": threads,
                "resolved_nice": nice,
                "format": FORMAT_ID,
                "pikapro_version": version("pikapro") if args.engine == "pikapro" else None,
                "team_count": len(teams),
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
                    "resolved_device": resolved_device,
                    "model_version": model_version,
                    "resolved_workers": workers,
                    "resolved_threads": threads,
                    "resolved_nice": nice,
                    "format": FORMAT_ID,
                    "pikapro_version": version("pikapro") if args.engine == "pikapro" else None,
                    "team_count": len(teams),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "wandb_run_id": wandb_run_id,
                },
                default=str,
                indent=2,
            )
            + "\n"
        )

    started = time.monotonic()
    last_eval = completed
    pool: ProcessPoolExecutor | None = None
    direct_worker = None
    if workers > 1:
        ctx = multiprocessing.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_rollout_worker,
            initargs=(args.engine, args.width, args.layers, model_version, nice),
        )
    else:
        direct_worker = _make_worker(args.engine)
        direct_worker.__enter__()

    try:
        with args.out.joinpath("metrics.jsonl").open(metrics_mode) as metrics:
            while completed < args.games:
                step_started = time.monotonic()
                games = min(args.games - completed, args.games_per_update)
                jobs: list[tuple] = []
                snapshot_by_count = {s.game_count: s for s in league}
                for game in range(completed, completed + games):
                    snapshot = None
                    anchor_name = None
                    if anchor_opponents and args.anchor_fraction > 0 and rng.random() < args.anchor_fraction:
                        anchor_name = rng.choice(anchor_opponents)
                    elif game % 2 == 1:
                        snapshot = rng.choices(league, weights=[s.priority for s in league], k=1)[0]
                    learner_is_p1 = (game // 2) % 2 == 0
                    team_p1, team_p2 = rng.sample(teams, k=2)
                    sim_seed = [rng.randrange(2**16) for _ in range(4)]
                    jobs.append(
                        (
                            game,
                            team_p1,
                            team_p2,
                            learner_is_p1,
                            sim_seed,
                            rng.getrandbits(32),
                            rng.getrandbits(32),
                            None if snapshot is None else snapshot.game_count,
                            None if snapshot is None else str(snapshot.path),
                            anchor_name,
                        )
                    )

                if pool is not None:
                    learner_state = _cpu_state_dict(model)
                    chunks = _split_chunks(jobs, workers)
                    futures = [
                        pool.submit(
                            _run_rollout_chunk,
                            learner_state,
                            chunk,
                            args.rollout_batch_size,
                            args.gamma,
                            args.gae_lambda,
                        )
                        for chunk in chunks
                    ]
                    samples: list[tuple[Decision, float, float]] = []
                    summaries: list[tuple[float, int, int | None, str | None]] = []
                    for future in futures:
                        chunk_samples, chunk_summaries = future.result()
                        samples.extend(chunk_samples)
                        summaries.extend(chunk_summaries)
                else:
                    path_to_model = {str(s.path): s.model for s in league}
                    samples, summaries = _play_rollout_jobs(
                        direct_worker,
                        model,
                        jobs,
                        lambda path_str: path_to_model[path_str],
                        rollout_batch_size=args.rollout_batch_size,
                        gamma=args.gamma,
                        gae_lambda=args.gae_lambda,
                    )

                opponents_played: dict[int, int] = {}
                current_games = 0
                anchor_games = 0
                anchor_score = 0.0
                historical_score = 0.0
                draws = 0
                mean_turns = 0
                for reward, turns, snap_game_count, anchor_name in summaries:
                    draws += int(reward == 0)
                    mean_turns += turns
                    if anchor_name is not None:
                        anchor_games += 1
                        anchor_score += (reward + 1.0) / 2.0
                    elif snap_game_count is None:
                        current_games += 1
                    else:
                        points = (reward + 1.0) / 2.0
                        historical_score += points
                        snap_obj = snapshot_by_count[snap_game_count]
                        snap_obj.learner_points += points
                        snap_obj.games += 1
                        opponents_played[snap_game_count] = (
                            opponents_played.get(snap_game_count, 0) + 1
                        )

                non_draws = [(step, reward) for step, reward, _ in samples if reward != 0]
                value_accuracy = (
                    sum((step.value > 0) == (reward > 0) for step, reward in non_draws)
                    / len(non_draws)
                    if non_draws
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

                ppo = update_policy_with_metrics(
                    model,
                    optimizer,
                    samples,
                    rng,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    target_kl=args.target_kl,
                )
                completed += games
                if completed >= snapshot_index + args.snapshot_games or completed == args.games:
                    snap_path = save_snapshot(args.out, model, completed)
                    league.append(
                        Snapshot(
                            completed,
                            snap_path,
                            copy.deepcopy(model) if workers == 1 else None,
                        )
                    )
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
                    holdout_teams
                    and args.eval_every > 0
                    and (completed - last_eval >= args.eval_every or completed == args.games)
                ):
                    eval_metrics = _run_holdout_eval(
                        model=model,
                        holdout_teams=holdout_teams,
                        opponents=eval_opponents,
                        eval_games=args.eval_games,
                        eval_seed=args.seed + 900001,
                        pool=pool,
                        direct_worker=direct_worker,
                        workers=workers,
                    )
                    last_eval = completed

                step_elapsed = max(time.monotonic() - step_started, 1e-6)
                historical_games = games - current_games - anchor_games
                row: dict[str, object] = {
                    "games": completed,
                    "batch_games": games,
                    "vs_current": current_games,
                    "vs_anchors": anchor_games,
                    "vs_anchors_score": round(anchor_score / anchor_games, 4)
                    if anchor_games
                    else None,
                    "vs_snapshots": opponents_played,
                    "vs_snapshots_score": historical_score / historical_games
                    if historical_games
                    else None,
                    "draws": draws,
                    "mean_turns": mean_turns / games,
                    "decisions": len(samples),
                    "preview_decisions": sum(step.encoded.preview for step, _, _ in samples),
                    "training_loss": ppo["loss"],
                    "policy_loss": ppo["policy_loss"],
                    "value_loss": ppo["value_loss"],
                    "entropy": ppo["entropy"],
                    "approx_kl": ppo["approx_kl"],
                    "ppo_epochs_run": int(ppo["epochs_run"]),
                    "learning_rate": current_lr,
                    "value_accuracy": round(value_accuracy, 4)
                    if value_accuracy is not None
                    else None,
                    "games_per_sec": round(games / step_elapsed, 2),
                    "decisions_per_sec": round(len(samples) / step_elapsed, 1),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    **eval_metrics,
                }
                line = json.dumps(row, sort_keys=True)
                print(line, flush=True)
                metrics.write(line + "\n")
                metrics.flush()

                if wandb_run is not None:
                    log_payload: dict[str, float] = {
                        "train/loss": ppo["loss"],
                        "train/policy_loss": ppo["policy_loss"],
                        "train/value_loss": ppo["value_loss"],
                        "train/entropy": ppo["entropy"],
                        "train/approx_kl": ppo["approx_kl"],
                        "train/ppo_epochs_run": ppo["epochs_run"],
                        "train/learning_rate": current_lr,
                        "train/mean_turns": mean_turns / games,
                        "perf/games_per_sec": games / step_elapsed,
                        "perf/decisions_per_sec": len(samples) / step_elapsed,
                    }
                    if value_accuracy is not None:
                        log_payload["train/value_accuracy"] = value_accuracy
                    if historical_games:
                        log_payload["train/vs_snapshots_score"] = (
                            historical_score / historical_games
                        )
                    if anchor_games:
                        log_payload["train/vs_anchors_score"] = anchor_score / anchor_games
                    for key, value in eval_metrics.items():
                        log_payload[f"eval/{key.removeprefix('eval_')}"] = value
                    wandb_run.log(log_payload, step=completed)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if direct_worker is not None:
            direct_worker.__exit__(None, None, None)
        if wandb_run is not None:
            wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted or finished run from <out>/trainer_state.pt",
    )
    parser.add_argument(
        "--profile",
        choices=("desktop", "afk"),
        default="desktop",
        help=(
            "'desktop' leaves half the CPU free at low priority for interactive use; "
            "'afk' uses most cores for unattended/overnight runs"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="override parallel battle worker count (default comes from --profile)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="override PyTorch CPU threads for updates (default comes from --profile)",
    )
    parser.add_argument("--engine", choices=("pikapro", "showdown"), default="pikapro")
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-manifest", type=Path, default=DEFAULT_HOLDOUT_MANIFEST)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--games-per-update", type=int, default=64)
    parser.add_argument("--snapshot-games", type=int, default=256)
    parser.add_argument("--league-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--model-version",
        type=int,
        default=CHECKPOINT_VERSION,
        help="replication architecture/feature version (1=legacy, 2=timers+protect+bce)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="training device: auto (uses MPS/CUDA if available), cpu, mps, or cuda",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=16,
        help="concurrent battles stepped in lockstep per worker for batched neural inference",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="learning-rate schedule across total --games (default: constant)",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1.5e-5,
        help="final learning rate for --lr-schedule cosine",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="GAE discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.025,
        help="PPO KL early-stopping threshold (0 disables)",
    )
    parser.add_argument(
        "--anchor-opponents",
        default="",
        help="comma-separated fixed anchor opponents mixed into training (e.g. heuristic,vgc)",
    )
    parser.add_argument(
        "--anchor-fraction",
        type=float,
        default=0.0,
        help="fraction of training games played against anchor opponents (0..1)",
    )
    parser.add_argument("--seed", type=int, default=37101)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=256,
        help="run held-out baseline eval every N games (0 disables)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=20,
        help="held-out eval games per baseline opponent",
    )
    parser.add_argument(
        "--eval-opponents",
        default="maxpower,heuristic",
        help="comma-separated baseline opponent names for held-out eval",
    )
    parser.add_argument("--wandb-project", default=None, help="Weights & Biases project name")
    parser.add_argument("--wandb-entity", default=None, help="Weights & Biases entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="Weights & Biases run display name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases logging mode",
    )
    args = parser.parse_args()
    positive = [
        args.games,
        args.games_per_update,
        args.snapshot_games,
        args.league_size,
        args.batch_size,
        args.epochs,
        args.width,
        args.layers,
        args.learning_rate,
    ]
    if args.workers is not None:
        positive.append(args.workers)
    if args.threads is not None:
        positive.append(args.threads)
    if min(positive) <= 0 or args.width % 4 or args.eval_every < 0 or args.eval_games < 0:
        parser.error(
            "counts and learning rate must be positive, eval counts non-negative, "
            "and width must be divisible by four"
        )
    train(args)


if __name__ == "__main__":
    raise SystemExit(main())
