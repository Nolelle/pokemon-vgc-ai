"""Minimal fixed-mirror PPO trainer for Phase 2 RL experiments.

Same team, same preview order, fogged direct env, terminal ±1 reward. Side
alternation matches ``vgc.rl.match.run_series``. Training can be one baseline
(``--opponent``) or a per-battle mixture (``--mix name=weight,...``). Evaluation
always scores listed opponents separately and reports pool mean / min win rate.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "train_fixed_mirror requires the 'train' extra (torch) -- run `uv sync --extra train`."
    ) from exc

from vgc.baselines import BASELINES
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
# Mixed-opponent experiment: keep VGC-family mass while putting enough weight on
# SimpleHeuristic that sequential training forgot.
DEFAULT_MIX = (
    "random=0.1,maxpower=0.1,heuristic=0.2,"
    "vgc_myopic=0.2,vgc_shallow=0.2,vgc=0.2"
)


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
        "--mix",
        default="",
        help=(
            "comma-separated name=weight mix sampled independently each training "
            "battle; empty uses --opponent for every battle. Weights are normalized."
        ),
    )
    parser.add_argument(
        "--eval-opponents",
        default="",
        help="comma-separated baseline names; empty uses --mix names, else --opponent",
    )
    parser.add_argument(
        "--eval-at-games",
        default="",
        help=(
            "comma-separated game counts at which to eval all --eval-opponents "
            "(in addition to the 0-game eval). When set, --eval-every-iterations "
            "is ignored."
        ),
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
    parser.add_argument(
        "--information-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable complete own-team and fog-safe opponent-belief model inputs",
    )
    parser.add_argument(
        "--tactical-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable first-principles damage, KO, Speed, threat, and synergy inputs",
    )
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
            "use_information_features": model.use_information_features,
            "use_tactical_features": model.use_tactical_features,
            "head_dropout": model.head_dropout_p,
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
    checkpoint_information = bool(checkpoint.get("use_information_features", False))
    if checkpoint_information != model.use_information_features:
        raise ValueError(
            f"checkpoint use_information_features={checkpoint_information} does not match "
            f"model use_information_features={model.use_information_features}"
        )
    checkpoint_tactical = bool(checkpoint.get("use_tactical_features", False))
    if checkpoint_tactical != model.use_tactical_features:
        raise ValueError(
            f"checkpoint use_tactical_features={checkpoint_tactical} does not match "
            f"model use_tactical_features={model.use_tactical_features}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved = checkpoint.get("ppo_config", {})
    return (
        int(checkpoint.get("iteration", 0)),
        int(checkpoint.get("games_seen", 0)),
        PpoConfig(**saved),
    )


def parse_opponent_mix(raw: str) -> list[tuple[str, float]]:
    """Parse ``name=weight,...`` into normalized ``(name, probability)`` pairs."""

    if not raw.strip():
        return []
    pairs: list[tuple[str, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"mix entry {part!r} must be name=weight")
        name, weight_s = part.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"mix entry {part!r} has an empty name")
        try:
            weight = float(weight_s)
        except ValueError as exc:
            raise ValueError(f"mix entry {part!r} has a non-numeric weight") from exc
        if weight < 0:
            raise ValueError(f"mix entry {part!r} has a negative weight")
        pairs.append((name, weight))
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        raise ValueError("mix weights must sum to a positive number")
    return [(name, weight / total) for name, weight in pairs]


def sample_opponent(mix: Sequence[tuple[str, float]], rng: random.Random) -> str:
    names = [name for name, _ in mix]
    weights = [weight for _, weight in mix]
    return rng.choices(names, weights=weights, k=1)[0]


def parse_eval_at_games(raw: str) -> list[int]:
    if not raw.strip():
        return []
    targets: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        target = int(part)
        if target <= 0:
            raise ValueError(f"eval-at-games value {part!r} must be a positive integer")
        targets.append(target)
    return sorted(set(targets))


def _parse_eval_opponents(
    raw: str,
    default_opponent: str,
    mix_names: Sequence[str] | None = None,
) -> list[str]:
    if raw.strip():
        return [name.strip() for name in raw.split(",") if name.strip()]
    if mix_names:
        return list(mix_names)
    return [default_opponent]


def pool_eval_metrics(eval_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """Mean and min win rate across separately evaluated opponents."""

    rates = {str(row["opponent"]): float(row["win_rate"]) for row in eval_rows}
    mean_wr = sum(rates.values()) / len(rates)
    min_opponent = min(rates, key=rates.__getitem__)
    return {
        "kind": "pool",
        "mean_win_rate": mean_wr,
        "min_win_rate": rates[min_opponent],
        "min_opponent": min_opponent,
        "win_rates": rates,
    }


def _save_iteration_snapshot(
    out_dir: Path,
    model: CandidatePolicyValueNet,
    iteration: int,
    *,
    games_seen: int,
) -> Path:
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"iter_{iteration:04d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": RL_ARCHITECTURE_VERSION,
            "iteration": iteration,
            "games_seen": games_seen,
            "use_meta_features": model.use_meta_features,
            "use_information_features": model.use_information_features,
            "use_tactical_features": model.use_tactical_features,
            "head_dropout": model.head_dropout_p,
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
) -> dict[str, object]:
    eval_rows: list[dict[str, object]] = []
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
        eval_rows.append(eval_row)
        with eval_path.open("a") as eval_file:
            eval_file.write(json.dumps(eval_row, sort_keys=True) + "\n")
        print(
            f"  eval vs {opponent_name} @ {games_seen} games: "
            f"{summary['p1_wins']}/{summary['games']} "
            f"win_rate={summary['p1_win_rate']:.3f} "
            f"wilson={summary['p1_wilson']}"
        )
    pool = pool_eval_metrics(eval_rows)
    pool_row = {
        "iteration": iteration,
        "games_seen": games_seen,
        **pool,
    }
    with eval_path.open("a") as eval_file:
        eval_file.write(json.dumps(pool_row, sort_keys=True) + "\n")
    print(
        f"  pool @ {games_seen} games: mean={pool['mean_win_rate']:.3f} "
        f"min={pool['min_win_rate']:.3f} ({pool['min_opponent']})"
    )
    return pool


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers != 1:
        raise SystemExit("train_fixed_mirror v1 supports --workers 1 only")
    if args.iterations < 0:
        raise SystemExit("--iterations must be non-negative")
    if args.iterations == 0 and args.eval_games <= 0:
        raise SystemExit("--iterations 0 is eval-only and requires --eval-games > 0")
    if args.games_per_iteration <= 0:
        raise SystemExit("--games-per-iteration must be positive")
    if args.eval_games < 0:
        raise SystemExit("--eval-games must be non-negative")
    if args.eval_every_iterations <= 0:
        raise SystemExit("--eval-every-iterations must be positive")
    try:
        mix = parse_opponent_mix(args.mix)
        eval_at_games = parse_eval_at_games(args.eval_at_games)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown = [name for name, _ in mix if name not in BASELINES]
    if unknown:
        available = ", ".join(sorted(BASELINES))
        raise SystemExit(f"unknown mix opponent {unknown!r}; available: {available}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    team = DEFAULT_TEAM.read_text().strip()
    eval_opponents = _parse_eval_opponents(
        args.eval_opponents,
        args.opponent,
        mix_names=[name for name, _ in mix],
    )
    unknown_eval = [name for name in eval_opponents if name not in BASELINES]
    if unknown_eval:
        available = ", ".join(sorted(BASELINES))
        raise SystemExit(
            f"unknown eval opponent {unknown_eval!r}; available: {available}"
        )
    ppo_config = build_ppo_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    eval_path = args.out_dir / "eval.jsonl"

    model = CandidatePolicyValueNet(
        use_meta_features=False,
        use_information_features=args.information_features,
        use_tactical_features=args.tactical_features,
    ).to(args.device)
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
        checkpoint_information = bool(checkpoint.get("use_information_features", False))
        if checkpoint_information != model.use_information_features:
            raise SystemExit(
                "--init-from information-feature setting does not match; pass "
                "--information-features only for a checkpoint trained with that input"
            )
        checkpoint_tactical = bool(checkpoint.get("use_tactical_features", False))
        if checkpoint_tactical != model.use_tactical_features:
            raise SystemExit(
                "--init-from tactical-feature setting does not match; pass "
                "--tactical-features only for a checkpoint trained with that input"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"initialized weights from {args.init_from}")

    games_seen = 0
    sim_repo = DEFAULT_SHOWDOWN_REPO
    if not (sim_repo / "dist" / "sim" / "index.js").exists():
        raise SystemExit(f"no built showdown sim at {sim_repo}")

    with SimWorker(sim_repo) as worker:
        if mix:
            mix_s = ", ".join(f"{name}={weight:.2f}" for name, weight in mix)
            print(f"training mix: {mix_s}", flush=True)
        else:
            print(f"training opponent: {args.opponent}", flush=True)
        evaled_targets: set[int] = set()
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
            buffer = RolloutBuffer()
            rng = random.Random(args.seed + iteration * 10_000)
            wins = losses = draws = steps = 0
            opponent_counts: Counter[str] = Counter()
            for game_idx in range(args.games_per_iteration):
                learner_side = "p1" if game_idx % 2 == 0 else "p2"
                opponent_name = sample_opponent(mix, rng) if mix else args.opponent
                opponent_counts[opponent_name] += 1
                sim_seed = [rng.randrange(1, 2**31) for _ in range(4)]
                policy_seed = args.seed + iteration * 10_000 + game_idx
                opponent_seed = args.seed + iteration * 10_000 + game_idx + 1
                result, game_steps = _play_training_game(
                    worker,
                    model=model,
                    buffer=buffer,
                    team=team,
                    opponent_name=opponent_name,
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
            # Save AFTER the update. Previously iter_0040 represented the model before
            # iteration 40 while its filename and metrics row implied the opposite.
            snapshot_path = _save_iteration_snapshot(
                args.out_dir,
                model,
                iteration,
                games_seen=games_seen,
            )
            row = {
                "iteration": iteration,
                "games_seen": games_seen,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "steps": steps,
                "opponent_counts": dict(opponent_counts),
                "mix": {name: weight for name, weight in mix} if mix else {args.opponent: 1.0},
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

            due_targets = (
                [target for target in eval_at_games if games_seen >= target and target not in evaled_targets]
                if eval_at_games
                else []
            )
            eval_by_interval = (
                not eval_at_games
                and args.eval_games > 0
                and iteration % args.eval_every_iterations == 0
            )
            if args.eval_games > 0 and (due_targets or eval_by_interval):
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
                evaled_targets.update(due_targets)

    if args.iterations > 0:
        print(f"checkpoint: {args.out_dir / 'latest.pt'}")
        print(f"metrics: {metrics_path}")
    if args.eval_games > 0:
        print(f"eval: {eval_path}")


if __name__ == "__main__":
    main()
