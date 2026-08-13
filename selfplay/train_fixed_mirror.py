"""Minimal fixed-mirror PPO trainer for Phase 2 RL experiments.

Same team, same preview order, fogged direct env, terminal ±1 reward. One baseline
opponent at a time; side alternation matches ``vgc.rl.match.run_series``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "train_fixed_mirror requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.config import FORMAT_ID, REPO_ROOT
from vgc.rl.agents import PHASE2_PREVIEW_ORDER, DirectAgent, make_direct_agent
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.match import BattleOutcome, play_battle, summarize
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.ppo import PpoConfig, RolloutBuffer, ppo_update

DEFAULT_TEAM = REPO_ROOT / "teams" / "phase2_mirror.packed.txt"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "ppo" / "fixed_mirror"
LEARNER_NAME = "ppo"


def build_ppo_config() -> PpoConfig:
    """Experiment protocol knobs shared by every fixed-mirror run."""

    return PpoConfig(
        gamma=1.0,
        teacher_anchor_weight=0.0,
        reward_shaping_coef=0.0,
        entropy_weight=0.01,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games-per-iteration", type=int, default=32)
    parser.add_argument("--eval-games", type=int, default=0)
    parser.add_argument("--eval-every-iterations", type=int, default=1)
    parser.add_argument("--opponent", default="random")
    parser.add_argument(
        "--eval-opponents",
        default="",
        help="comma-separated baseline names; empty uses --opponent",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bc-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="load model weights only (curriculum); does not restore optimizer or games_seen",
    )
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


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
            "use_meta_features": model.use_meta_features,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CandidatePolicyValueNet,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
) -> tuple[int, int, PpoConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != RL_ARCHITECTURE_VERSION:
        raise ValueError(
            f"unsupported checkpoint architecture {architecture!r}; "
            f"expected {RL_ARCHITECTURE_VERSION!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved = checkpoint.get("ppo_config", {})
    return (
        int(checkpoint.get("iteration", 0)),
        int(checkpoint.get("games_seen", 0)),
        PpoConfig(**saved),
    )


def _parse_eval_opponents(raw: str, default_opponent: str) -> list[str]:
    if not raw.strip():
        return [default_opponent]
    return [name.strip() for name in raw.split(",") if name.strip()]


def _save_iteration_snapshot(out_dir: Path, model: CandidatePolicyValueNet, iteration: int) -> Path:
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"iter_{iteration:04d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": RL_ARCHITECTURE_VERSION,
            "iteration": iteration,
            "use_meta_features": model.use_meta_features,
        },
        path,
    )
    return path


def _play_training_game(
    worker: SimWorker,
    *,
    model: CandidatePolicyValueNet,
    buffer: RolloutBuffer,
    team: str,
    opponent_name: str,
    ppo_config: PpoConfig,
    device: str,
    battle_tag: str,
    learner_side: str,
    sim_seed: list[int],
    policy_seed: int,
    opponent_seed: int,
) -> tuple[float, int]:
    random.seed(opponent_seed)
    learner = PpoVgcPlayer(
        model=model,
        rollout_buffer=buffer,
        ppo_config=ppo_config,
        device=device,
        policy_seed=policy_seed,
        start_listening=False,
        team=team,
        battle_format=FORMAT_ID,
    )
    learner_agent = DirectAgent(
        learner,
        name=LEARNER_NAME,
        preview_order=PHASE2_PREVIEW_ORDER,
    )
    opponent_agent = make_direct_agent(
        opponent_name,
        team,
        preview_order=PHASE2_PREVIEW_ORDER,
    )
    opponent_side = "p2" if learner_side == "p1" else "p1"
    agents = {learner_side: learner_agent, opponent_side: opponent_agent}
    steps_before = len(buffer.steps)
    outcome = play_battle(
        worker,
        battle_tag,
        agents,
        {"p1": team, "p2": team},
        battle_format=FORMAT_ID,
        seed=sim_seed,
    )
    steps_added = len(buffer.steps) - steps_before
    return outcome.result_for(LEARNER_NAME), steps_added


def _run_eval(
    worker: SimWorker,
    *,
    model: CandidatePolicyValueNet,
    team: str,
    opponent_name: str,
    games: int,
    ppo_config: PpoConfig,
    device: str,
    seed: int,
    iteration: int,
) -> dict[str, object]:
    model.eval()
    rng = random.Random(
        seed + iteration * 20_000 + sum(ord(char) for char in opponent_name)
    )
    outcomes: list[BattleOutcome] = []
    for game_idx in range(games):
        learner_side = "p1" if game_idx % 2 == 0 else "p2"
        opponent_side = "p2" if learner_side == "p1" else "p1"
        sim_seed = [rng.randrange(1, 2**31) for _ in range(4)]
        random.seed(seed + iteration * 20_000 + game_idx)
        learner = PpoVgcPlayer(
            model=model,
            rollout_buffer=None,
            ppo_config=ppo_config,
            device=device,
            deterministic=False,
            policy_seed=seed + iteration * 20_000 + game_idx,
            start_listening=False,
            team=team,
            battle_format=FORMAT_ID,
        )
        learner_agent = DirectAgent(
            learner,
            name=LEARNER_NAME,
            preview_order=PHASE2_PREVIEW_ORDER,
        )
        opponent_agent = make_direct_agent(
            opponent_name,
            team,
            preview_order=PHASE2_PREVIEW_ORDER,
        )
        agents = {learner_side: learner_agent, opponent_side: opponent_agent}
        outcomes.append(
            play_battle(
                worker,
                f"eval-{iteration}-{opponent_name}-{game_idx}",
                agents,
                {"p1": team, "p2": team},
                battle_format=FORMAT_ID,
                seed=sim_seed,
            )
        )
    return summarize(outcomes, LEARNER_NAME, opponent_name)


def _record_evals(
    worker: SimWorker,
    *,
    model: CandidatePolicyValueNet,
    team: str,
    opponents: list[str],
    games: int,
    ppo_config: PpoConfig,
    device: str,
    seed: int,
    iteration: int,
    games_seen: int,
    eval_path: Path,
) -> None:
    for opponent_name in opponents:
        summary = _run_eval(
            worker,
            model=model,
            team=team,
            opponent_name=opponent_name,
            games=games,
            ppo_config=ppo_config,
            device=device,
            seed=seed,
            iteration=iteration,
        )
        eval_row = {
            "iteration": iteration,
            "games_seen": games_seen,
            "opponent": opponent_name,
            "games": summary["games"],
            "wins": summary["p1_wins"],
            "losses": summary["p2_wins"],
            "draws": summary["draws"],
            "win_rate": summary["p1_win_rate"],
            "wilson": summary["p1_wilson"],
        }
        with eval_path.open("a") as eval_file:
            eval_file.write(json.dumps(eval_row, sort_keys=True) + "\n")
        print(
            f"  eval vs {opponent_name} @ {games_seen} games: "
            f"{summary['p1_wins']}/{summary['games']} "
            f"win_rate={summary['p1_win_rate']:.3f} "
            f"wilson={summary['p1_wilson']}"
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers != 1:
        raise SystemExit("train_fixed_mirror v1 supports --workers 1 only")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.games_per_iteration <= 0:
        raise SystemExit("--games-per-iteration must be positive")
    if args.eval_games < 0:
        raise SystemExit("--eval-games must be non-negative")
    if args.eval_every_iterations <= 0:
        raise SystemExit("--eval-every-iterations must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    team = DEFAULT_TEAM.read_text().strip()
    eval_opponents = _parse_eval_opponents(args.eval_opponents, args.opponent)
    ppo_config = build_ppo_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    eval_path = args.out_dir / "eval.jsonl"

    model = CandidatePolicyValueNet(use_meta_features=False).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.bc_checkpoint is not None and args.bc_checkpoint.exists():
        loaded = model.warm_start_state_encoder(args.bc_checkpoint)
        print(
            f"warm-started state encoder: {loaded['loaded']}/{loaded['available']} tensors "
            f"from {args.bc_checkpoint}"
        )
    if args.init_from is not None:
        if not args.init_from.is_file():
            raise SystemExit(f"--init-from not found: {args.init_from}")
        checkpoint = torch.load(args.init_from, map_location=args.device, weights_only=False)
        architecture = checkpoint.get("architecture")
        if architecture != RL_ARCHITECTURE_VERSION:
            raise SystemExit(
                f"unsupported --init-from architecture {architecture!r}; "
                f"expected {RL_ARCHITECTURE_VERSION!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"initialized weights from {args.init_from}")

    games_seen = 0
    sim_repo = DEFAULT_SHOWDOWN_REPO
    if not (sim_repo / "dist" / "sim" / "index.js").exists():
        raise SystemExit(f"no built showdown sim at {sim_repo}")

    with SimWorker(sim_repo) as worker:
        if args.eval_games > 0:
            print(f"eval at 0 games ({args.eval_games} per opponent)", flush=True)
            _record_evals(
                worker,
                model=model,
                team=team,
                opponents=eval_opponents,
                games=args.eval_games,
                ppo_config=ppo_config,
                device=args.device,
                seed=args.seed,
                iteration=0,
                games_seen=0,
                eval_path=eval_path,
            )
        for iteration in range(1, args.iterations + 1):
            started = time.time()
            snapshot_path = _save_iteration_snapshot(args.out_dir, model, iteration)
            buffer = RolloutBuffer()
            rng = random.Random(args.seed + iteration * 10_000)
            wins = losses = draws = steps = 0
            for game_idx in range(args.games_per_iteration):
                learner_side = "p1" if game_idx % 2 == 0 else "p2"
                sim_seed = [rng.randrange(1, 2**31) for _ in range(4)]
                policy_seed = args.seed + iteration * 10_000 + game_idx
                opponent_seed = args.seed + iteration * 10_000 + game_idx + 1
                result, game_steps = _play_training_game(
                    worker,
                    model=model,
                    buffer=buffer,
                    team=team,
                    opponent_name=args.opponent,
                    ppo_config=ppo_config,
                    device=args.device,
                    battle_tag=f"train-{iteration}-{game_idx}",
                    learner_side=learner_side,
                    sim_seed=sim_seed,
                    policy_seed=policy_seed,
                    opponent_seed=opponent_seed,
                )
                steps += game_steps
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    draws += 1

            if not buffer.steps:
                raise RuntimeError("no PPO decisions were recorded from completed training games")
            update_metrics = ppo_update(
                model, optimizer, buffer, ppo_config, device=args.device
            )
            games_seen += args.games_per_iteration
            row = {
                "iteration": iteration,
                "games_seen": games_seen,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "steps": steps,
                "ppo": update_metrics,
                "snapshot": str(snapshot_path),
                "elapsed_seconds": time.time() - started,
            }
            with metrics_path.open("a") as metrics_file:
                metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
            save_checkpoint(
                args.out_dir / "latest.pt",
                model,
                optimizer,
                iteration=iteration,
                games_seen=games_seen,
                ppo_config=ppo_config,
            )
            print(
                f"iteration {iteration}: games={args.games_per_iteration} "
                f"wins={wins} losses={losses} draws={draws} steps={steps} "
                f"loss={update_metrics['loss']:.4f} entropy={update_metrics['entropy']:.4f}"
            )

            if args.eval_games > 0 and iteration % args.eval_every_iterations == 0:
                _record_evals(
                    worker,
                    model=model,
                    team=team,
                    opponents=eval_opponents,
                    games=args.eval_games,
                    ppo_config=ppo_config,
                    device=args.device,
                    seed=args.seed,
                    iteration=iteration,
                    games_seen=games_seen,
                    eval_path=eval_path,
                )

    print(f"checkpoint: {args.out_dir / 'latest.pt'}")
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
