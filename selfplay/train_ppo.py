#!/usr/bin/env python
"""Train the candidate-aware policy/value network from real local Showdown games.

Learners play complete battles against a mixture of the fixed VGC heuristic and a
bounded pool of historical policy snapshots, receive only final win/draw/loss reward,
then update between game batches. Independent poke-env clients collect workers in
parallel while sharing one frozen learner model for that rollout batch. It remains
default-off and writes checkpoints under ``runs/ppo``.

Example smoke run::

    .venv/bin/python selfplay/train_ppo.py --iterations 1 --games-per-iteration 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import secrets
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402

from selfplay.run_selfplay import _safe_stop_listening, start_local_server  # noqa: E402
from vgc.agent import VgcPlayer  # noqa: E402
from vgc.config import FORMAT_ID, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.model import CandidatePolicyValueNet  # noqa: E402
from vgc.rl.opponents import (  # noqa: E402
    RL_ARCHITECTURE_VERSION,
    OpponentSpec,
    choose_opponent,
    discover_snapshots,
    load_snapshot,
    save_snapshot,
)
from vgc.rl.player import PpoVgcPlayer  # noqa: E402
from vgc.rl.ppo import PpoConfig, RolloutBuffer, ppo_update  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "ppo"
DEFAULT_BC_CHECKPOINT = REPO_ROOT / "data" / "models" / "bc_policy_v4_selfplay.pt"
WORKER_TIMEOUT_PER_GAME_SECONDS = 120.0


def save_checkpoint(
    path: Path,
    model: CandidatePolicyValueNet,
    optimizer: torch.optim.Optimizer,
    *,
    iteration: int,
    games_seen: int,
    ppo_config: PpoConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "games_seen": games_seen,
            "ppo_config": asdict(ppo_config),
            "architecture": RL_ARCHITECTURE_VERSION,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    model: CandidatePolicyValueNet,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
) -> tuple[int, int, PpoConfig]:
    """Restore training state and return iteration, game count, and PPO settings."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != RL_ARCHITECTURE_VERSION:
        raise ValueError(
            f"unsupported PPO checkpoint architecture {architecture!r}; "
            f"expected {RL_ARCHITECTURE_VERSION!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved_ppo_config = PpoConfig(**checkpoint.get("ppo_config", {}))
    return (
        int(checkpoint.get("iteration", 0)),
        int(checkpoint.get("games_seen", 0)),
        saved_ppo_config,
    )


async def _collect_worker(
    model: CandidatePolicyValueNet,
    *,
    worker_id: int,
    games: int,
    team: str,
    device: str,
    ppo_config: PpoConfig,
    opponent_spec: OpponentSpec,
) -> tuple[dict[str, object], RolloutBuffer]:
    token = secrets.token_hex(3)
    buffer = RolloutBuffer()
    common_config = replace(
        PolicyConfig(),
        accept_open_team_sheet=False,
        use_rolling_horizon=False,
    )
    learner = PpoVgcPlayer(
        model=model,
        rollout_buffer=buffer,
        ppo_config=ppo_config,
        device=device,
        config=common_config,
        team=team,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"ppol{worker_id}-{token}", None),
    )
    if opponent_spec.kind == "snapshot":
        if opponent_spec.checkpoint_path is None:
            raise ValueError("snapshot opponent is missing checkpoint_path")
        opponent = PpoVgcPlayer(
            model=load_snapshot(opponent_spec.checkpoint_path, device=device),
            rollout_buffer=None,
            device=device,
            deterministic=False,
            config=common_config,
            team=team,
            battle_format=FORMAT_ID,
            account_configuration=AccountConfiguration(f"ppos{worker_id}-{token}", None),
        )
    else:
        opponent = VgcPlayer(
            config=common_config,
            team=team,
            battle_format=FORMAT_ID,
            account_configuration=AccountConfiguration(f"ppoh{worker_id}-{token}", None),
        )
    error: str | None = None
    try:
        await asyncio.wait_for(
            learner.battle_against(opponent, n_battles=games),
            timeout=max(1, games) * WORKER_TIMEOUT_PER_GAME_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - one worker must not erase other rollouts
        error = f"{type(exc).__name__}: {exc}"
        buffer.discard_unfinished_episode()
    finally:
        await _safe_stop_listening(learner)
        await _safe_stop_listening(opponent)
    return {
        "worker": worker_id,
        "requested": games,
        "games": learner.n_finished_battles,
        "wins": learner.n_won_battles,
        "losses": learner.n_lost_battles,
        "steps": len(buffer),
        "opponent": opponent_spec.label,
        "error": error,
    }, buffer


def allocate_games(games: int, jobs: int) -> list[int]:
    """Balanced positive worker chunks that sum exactly to ``games``."""

    if games <= 0 or jobs <= 0:
        raise ValueError("games and jobs must be positive")
    worker_count = min(games, jobs)
    base, remainder = divmod(games, worker_count)
    return [base + int(index < remainder) for index in range(worker_count)]


async def collect_games(
    model: CandidatePolicyValueNet,
    buffer: RolloutBuffer,
    *,
    games: int,
    jobs: int,
    team: str,
    device: str,
    ppo_config: PpoConfig,
    snapshots: list[Path],
    heuristic_fraction: float,
    seed: int,
) -> dict[str, object]:
    allocations = allocate_games(games, jobs)
    specs = [
        choose_opponent(
            snapshots,
            heuristic_fraction=heuristic_fraction,
            rng=random.Random(seed + worker_id),
        )
        for worker_id in range(len(allocations))
    ]
    worker_results = await asyncio.gather(
        *(
            _collect_worker(
                model,
                worker_id=worker_id,
                games=worker_games,
                team=team,
                device=device,
                ppo_config=ppo_config,
                opponent_spec=specs[worker_id],
            )
            for worker_id, worker_games in enumerate(allocations)
        )
    )
    rows: list[dict[str, object]] = []
    opponent_counts: dict[str, int] = {}
    for row, worker_buffer in worker_results:
        buffer.extend_finished(worker_buffer)
        rows.append(row)
        label = str(row["opponent"])
        opponent_counts[label] = opponent_counts.get(label, 0) + int(row["games"])
    return {
        "games": sum(int(row["games"]) for row in rows),
        "wins": sum(int(row["wins"]) for row in rows),
        "losses": sum(int(row["losses"]) for row in rows),
        "steps": len(buffer),
        "worker_errors": sum(row["error"] is not None for row in rows),
        "opponents": opponent_counts,
        "workers": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--team", type=Path, default=TEAMS_DIR / "meta1.packed.txt")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume model, optimizer, iteration, and game count from a PPO checkpoint",
    )
    parser.add_argument(
        "--snapshot-pool-size",
        type=int,
        default=8,
        help="bounded number of rotating historical opponent snapshots",
    )
    parser.add_argument(
        "--heuristic-opponent-fraction",
        type=float,
        default=0.25,
        help="worker-level probability of using the fixed VGC heuristic anchor",
    )
    parser.add_argument(
        "--bc-checkpoint",
        type=Path,
        default=DEFAULT_BC_CHECKPOINT,
        help="warm-start the state encoder; pass a nonexistent path for random init",
    )
    parser.add_argument(
        "--skip-server-start",
        action="store_true",
        help="assume the local Showdown server is already running",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0 or args.games_per_iteration <= 0 or args.jobs <= 0:
        raise SystemExit("iterations, games-per-iteration, and jobs must be positive")
    if args.snapshot_pool_size <= 0:
        raise SystemExit("snapshot-pool-size must be positive")
    if not 0.0 <= args.heuristic_opponent_fraction <= 1.0:
        raise SystemExit("heuristic-opponent-fraction must be between 0 and 1")
    if not args.team.exists():
        raise SystemExit(f"team does not exist: {args.team}")
    torch.manual_seed(args.seed)

    model = CandidatePolicyValueNet().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    last_iteration = 0
    games_seen = 0
    ppo_config = PpoConfig()
    if args.resume is not None:
        if not args.resume.exists():
            raise SystemExit(f"resume checkpoint does not exist: {args.resume}")
        last_iteration, games_seen, ppo_config = load_training_checkpoint(
            args.resume, model, optimizer, device=args.device
        )
        print(f"resumed {args.resume}: iteration={last_iteration} games_seen={games_seen}")
    elif args.bc_checkpoint.exists():
        loaded = model.warm_start_state_encoder(args.bc_checkpoint)
        print(
            f"warm-started state encoder: {loaded['loaded']}/{loaded['available']} tensors "
            f"from {args.bc_checkpoint}"
        )
    else:
        print(f"BC warm-start skipped; checkpoint not found: {args.bc_checkpoint}")
    team = args.team.read_text().strip()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    pool_dir = args.out_dir / "opponent_pool"

    server_process = None
    if not args.skip_server_start:
        print("starting local Showdown server...")
        server_process = start_local_server()
        print("server ready")

    try:
        for offset in range(1, args.iterations + 1):
            iteration = last_iteration + offset
            started = time.time()
            snapshot_path = save_snapshot(
                pool_dir,
                model,
                generation=iteration - 1,
                max_snapshots=args.snapshot_pool_size,
            )
            snapshots = discover_snapshots(pool_dir)
            buffer = RolloutBuffer()
            result = asyncio.run(
                collect_games(
                    model,
                    buffer,
                    games=args.games_per_iteration,
                    jobs=args.jobs,
                    team=team,
                    device=args.device,
                    ppo_config=ppo_config,
                    snapshots=snapshots,
                    heuristic_fraction=args.heuristic_opponent_fraction,
                    seed=args.seed + iteration * 10_000,
                )
            )
            if not buffer.steps:
                raise RuntimeError("no PPO decisions were recorded from completed games")
            update_metrics = ppo_update(model, optimizer, buffer, ppo_config, device=args.device)
            games_seen += result["games"]
            row = {
                "iteration": iteration,
                "games_seen": games_seen,
                "batch": result,
                "ppo": update_metrics,
                "snapshot": str(snapshot_path),
                "snapshot_pool_size": len(snapshots),
                "elapsed_seconds": time.time() - started,
            }
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
            checkpoint_path = args.out_dir / "latest.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                iteration=iteration,
                games_seen=games_seen,
                ppo_config=ppo_config,
            )
            print(
                f"iteration {iteration} (+{offset}/{args.iterations} this run): "
                f"games={result['games']} wins={result['wins']} steps={result['steps']} "
                f"loss={update_metrics['loss']:.4f} entropy={update_metrics['entropy']:.4f}"
            )
    finally:
        if server_process is not None:
            server_process.kill()
            server_process.wait(timeout=10)

    print(f"checkpoint: {args.out_dir / 'latest.pt'}")
    print(f"metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
