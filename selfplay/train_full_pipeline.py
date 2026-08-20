#!/usr/bin/env python
"""Direct-environment PPO over varied teams, mixed baselines, and policy snapshots."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "train_full_pipeline requires the train extra -- run `uv sync --extra train`."
    ) from exc

from vgc.baselines import BASELINES
from vgc.config import FORMAT_ID, REPO_ROOT
from vgc.evaluation import clustered_interval, wilson_interval
from vgc.rl.agents import DirectAgent, make_direct_agent
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.match import play_battle
from vgc.rl.opponents import (
    RL_ARCHITECTURE_VERSION,
    discover_snapshots,
    load_snapshot,
    save_snapshot,
)
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.ppo import PpoConfig, RolloutBuffer, ppo_update

DEFAULT_MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json"
DEFAULT_INIT = REPO_ROOT / "runs" / "full_pipeline" / "imitation" / "best.pt"
DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "full_pipeline" / "rl"
DEFAULT_BASELINE_MIX = (
    "random=0.1,maxpower=0.1,heuristic=0.2,"
    "vgc_myopic=0.2,vgc_shallow=0.2,vgc=0.2"
)


@dataclass(frozen=True)
class TeamEntry:
    label: str
    archetype: str
    packed: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=64)
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--baseline-mix", default=DEFAULT_BASELINE_MIX)
    parser.add_argument("--selfplay-fraction", type=float, default=0.35)
    parser.add_argument("--mirror-fraction", type=float, default=0.25)
    parser.add_argument("--max-snapshots", type=int, default=8)
    parser.add_argument("--eval-games", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--teacher-anchor-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def parse_mix(raw: str) -> list[tuple[str, float]]:
    pairs: list[tuple[str, float]] = []
    for part in raw.split(","):
        name, separator, raw_weight = part.strip().partition("=")
        if not separator or not name:
            raise ValueError(f"baseline mix entry {part!r} must be name=weight")
        weight = float(raw_weight)
        if weight < 0.0:
            raise ValueError("baseline mix weights must be nonnegative")
        pairs.append((name, weight))
    total = sum(weight for _name, weight in pairs)
    if total <= 0.0:
        raise ValueError("baseline mix weights must sum to a positive value")
    return [(name, weight / total) for name, weight in pairs]


def sample_mix(mix: list[tuple[str, float]], rng: random.Random) -> str:
    return rng.choices(
        [name for name, _weight in mix],
        weights=[weight for _name, weight in mix],
        k=1,
    )[0]


def load_team_pool(manifest: Path) -> list[TeamEntry]:
    records = json.loads(manifest.read_text())
    teams = [
        TeamEntry(
            label=f"{record['archetype']}/{Path(record['file']).stem}",
            archetype=record["archetype"],
            packed=(manifest.parent / record["file"]).read_text().strip(),
        )
        for record in records
    ]
    if not teams:
        raise ValueError(f"team manifest is empty: {manifest}")
    return teams


def split_teams_by_archetype(
    teams: list[TeamEntry], *, holdout_fraction: float, seed: int
) -> tuple[list[TeamEntry], list[TeamEntry]]:
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    grouped: dict[str, list[TeamEntry]] = defaultdict(list)
    for team in teams:
        grouped[team.archetype].append(team)
    train: list[TeamEntry] = []
    holdout: list[TeamEntry] = []
    for index, archetype in enumerate(sorted(grouped)):
        choices = sorted(grouped[archetype], key=lambda team: team.label)
        random.Random(seed + index).shuffle(choices)
        count = max(1, min(len(choices) - 1, round(len(choices) * holdout_fraction)))
        holdout.extend(choices[:count])
        train.extend(choices[count:])
    return train, holdout


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    *,
    iteration: int,
    games_seen: int,
    ppo_config: PpoConfig,
    evaluation: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "architecture": RL_ARCHITECTURE_VERSION,
            "use_meta_features": model.use_meta_features,
            "use_information_features": model.use_information_features,
            "use_tactical_features": model.use_tactical_features,
            "head_dropout": model.head_dropout_p,
            "value_output_transform": model.value_output_transform,
            "iteration": iteration,
            "games_seen": games_seen,
            "ppo_config": asdict(ppo_config),
            "evaluation": evaluation,
        },
        path,
    )


def _policy_agent(model, team: str, *, device: str, seed: int, deterministic: bool):
    return DirectAgent(
        PpoVgcPlayer(
            model=model,
            rollout_buffer=None,
            ppo_config=PpoConfig(teacher_anchor_weight=0.0),
            device=device,
            deterministic=deterministic,
            policy_seed=seed,
            team=team,
            battle_format=FORMAT_ID,
            start_listening=False,
        ),
        name="policy",
    )


def evaluate_policy(
    worker: SimWorker,
    *,
    model,
    teams: list[TeamEntry],
    opponents: list[str],
    games: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    model.eval()
    rng = random.Random(seed)
    schedule_rng = random.Random(seed + 17)
    pairings = [(team, opponent) for team in teams for opponent in opponents]
    schedule_rng.shuffle(pairings)
    schedule = [
        (team, opponent, learner_side)
        for team, opponent in pairings
        for learner_side in ("p1", "p2")
    ]
    by_team: dict[str, list[float]] = defaultdict(list)
    by_opponent: dict[str, list[float]] = defaultdict(list)
    fallback_count = 0
    for game_index in range(games):
        if game_index and game_index % len(schedule) == 0:
            schedule_rng.shuffle(pairings)
            schedule = [
                (team, opponent, learner_side)
                for team, opponent in pairings
                for learner_side in ("p1", "p2")
            ]
        team, opponent_name, learner_side = schedule[game_index % len(schedule)]
        # poke-env's built-in random/max-power tie breaks use Python's module-level
        # RNG. Pin it separately from simulator and policy randomness so rerunning a
        # frozen gate reproduces the exact same battles.
        random.seed(seed + game_index * 1_000_003)
        learner = _policy_agent(
            model,
            team.packed,
            device=device,
            seed=seed + game_index,
            deterministic=True,
        )
        opponent = make_direct_agent(opponent_name, team.packed)
        other_side = "p2" if learner_side == "p1" else "p1"
        outcome = play_battle(
            worker,
            f"full-eval-{game_index}",
            {learner_side: learner, other_side: opponent},
            {learner_side: team.packed, other_side: team.packed},
            seed=[rng.randrange(1, 2**31) for _ in range(4)],
        )
        result = outcome.result_for("policy")
        win = 1.0 if result > 0 else 0.0
        by_team[team.label].append(win)
        by_opponent[opponent_name].append(win)
        fallback_count += learner.player.fallback_count
    clusters = [
        (int(sum(results)), len(results)) for results in by_team.values() if results
    ]
    total_games = sum(len(results) for results in by_team.values())
    total_wins = sum(sum(results) for results in by_team.values())
    opponent_results = {
        name: {
            "games": len(results),
            "wins": int(sum(results)),
            "win_rate": sum(results) / len(results),
            "wilson_interval": wilson_interval(int(sum(results)), len(results)),
        }
        for name, results in sorted(by_opponent.items())
    }
    opponent_rates = {
        name: float(result["win_rate"]) for name, result in opponent_results.items()
    }
    return {
        "games": total_games,
        "teams": len(clusters),
        "wins": int(total_wins),
        "win_rate": total_wins / total_games,
        "clustered_interval": clustered_interval(clusters),
        "by_opponent": opponent_rates,
        "opponent_results": opponent_results,
        "mean_opponent_win_rate": sum(opponent_rates.values()) / len(opponent_rates),
        "min_opponent_win_rate": min(opponent_rates.values()),
        "fallback_count": fallback_count,
    }


def _training_game(
    worker: SimWorker,
    *,
    model,
    buffer: RolloutBuffer,
    learner_team: TeamEntry,
    opponent_team: TeamEntry,
    opponent_kind: str,
    opponent_label: str,
    opponent_model,
    learner_side: str,
    ppo_config: PpoConfig,
    seed: int,
    rng: random.Random,
    device: str,
) -> float:
    random.seed(seed + 2)
    learner = DirectAgent(
        PpoVgcPlayer(
            model=model,
            rollout_buffer=buffer,
            ppo_config=ppo_config,
            device=device,
            deterministic=False,
            policy_seed=seed,
            team=learner_team.packed,
            battle_format=FORMAT_ID,
            start_listening=False,
        ),
        name="learner",
    )
    if opponent_kind == "snapshot":
        opponent = _policy_agent(
            opponent_model,
            opponent_team.packed,
            device=device,
            seed=seed + 1,
            deterministic=False,
        )
        opponent.name = "opponent"
    else:
        opponent = make_direct_agent(opponent_label, opponent_team.packed)
        opponent.name = "opponent"
    other_side = "p2" if learner_side == "p1" else "p1"
    outcome = play_battle(
        worker,
        f"full-train-{seed}",
        {learner_side: learner, other_side: opponent},
        {learner_side: learner_team.packed, other_side: opponent_team.packed},
        seed=[rng.randrange(1, 2**31) for _ in range(4)],
    )
    return outcome.result_for("learner")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.iterations <= 0 or args.games_per_iteration <= 0:
        raise SystemExit("--iterations and --games-per-iteration must be positive")
    if args.eval_games <= 0 or args.eval_every <= 0:
        raise SystemExit("--eval-games and --eval-every must be positive")
    if not 0.0 <= args.selfplay_fraction <= 1.0:
        raise SystemExit("--selfplay-fraction must be between 0 and 1")
    if not 0.0 <= args.mirror_fraction <= 1.0:
        raise SystemExit("--mirror-fraction must be between 0 and 1")
    mix = parse_mix(args.baseline_mix)
    unknown = sorted({name for name, _weight in mix} - set(BASELINES))
    if unknown:
        raise SystemExit(f"unknown baselines: {unknown}; available: {sorted(BASELINES)}")
    all_teams = load_team_pool(args.team_manifest)
    train_teams, holdout_teams = split_teams_by_archetype(
        all_teams, holdout_fraction=args.holdout_fraction, seed=args.seed
    )
    model = load_snapshot(args.init, device=args.device)
    if not model.use_information_features or not model.use_tactical_features:
        raise SystemExit("--init must be a complete-context information+tactical checkpoint")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ppo_config = PpoConfig(
        gamma=1.0,
        teacher_anchor_weight=args.teacher_anchor_weight,
        reward_shaping_coef=0.0,
        entropy_weight=0.01,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = args.out_dir / "snapshot_pool"
    metrics_path = args.out_dir / "metrics.jsonl"
    eval_path = args.out_dir / "eval.jsonl"
    # Generation zero is the frozen imitation policy, so self-play is available from
    # the first PPO iteration and always retains a pre-RL anchor in the ring initially.
    save_snapshot(pool_dir, model, generation=0, max_snapshots=args.max_snapshots)
    games_seen = 0
    best_key = (-1.0, -1.0)
    eval_opponents = [name for name, _weight in mix]

    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        initial_eval = evaluate_policy(
            worker,
            model=model,
            teams=holdout_teams,
            opponents=eval_opponents,
            games=args.eval_games,
            seed=args.seed + 9_000_000,
            device=args.device,
        )
        initial_row = {"iteration": 0, "games_seen": 0, **initial_eval}
        eval_path.write_text(json.dumps(initial_row, sort_keys=True) + "\n")
        save_training_checkpoint(
            args.out_dir / "best.pt",
            model,
            optimizer,
            iteration=0,
            games_seen=0,
            ppo_config=ppo_config,
            evaluation=initial_eval,
        )
        best_key = (
            float(initial_eval["min_opponent_win_rate"]),
            float(initial_eval["mean_opponent_win_rate"]),
        )
        print(f"eval 0: {initial_eval}", flush=True)

        for iteration in range(1, args.iterations + 1):
            started = time.time()
            rng = random.Random(args.seed + iteration * 100_000)
            snapshots = discover_snapshots(pool_dir)
            snapshot_cache = {
                path: load_snapshot(path, device=args.device) for path in snapshots
            }
            buffer = RolloutBuffer()
            results: list[float] = []
            opponent_counts: Counter[str] = Counter()
            learner_team_counts: Counter[str] = Counter()
            for game_index in range(args.games_per_iteration):
                learner_team = rng.choice(train_teams)
                opponent_team = (
                    learner_team
                    if rng.random() < args.mirror_fraction
                    else rng.choice(train_teams)
                )
                use_snapshot = bool(snapshots) and rng.random() < args.selfplay_fraction
                if use_snapshot:
                    snapshot_path = rng.choice(snapshots)
                    opponent_kind = "snapshot"
                    opponent_label = snapshot_path.stem
                    opponent_model = snapshot_cache[snapshot_path]
                else:
                    opponent_kind = "baseline"
                    opponent_label = sample_mix(mix, rng)
                    opponent_model = None
                opponent_counts[opponent_label] += 1
                learner_team_counts[learner_team.label] += 1
                results.append(
                    _training_game(
                        worker,
                        model=model,
                        buffer=buffer,
                        learner_team=learner_team,
                        opponent_team=opponent_team,
                        opponent_kind=opponent_kind,
                        opponent_label=opponent_label,
                        opponent_model=opponent_model,
                        learner_side="p1" if game_index % 2 == 0 else "p2",
                        ppo_config=ppo_config,
                        seed=args.seed + iteration * 100_000 + game_index,
                        rng=rng,
                        device=args.device,
                    )
                )
            update = ppo_update(model, optimizer, buffer, ppo_config, device=args.device)
            games_seen += args.games_per_iteration
            snapshot_path = save_snapshot(
                pool_dir,
                model,
                generation=iteration,
                max_snapshots=args.max_snapshots,
            )
            row = {
                "iteration": iteration,
                "games_seen": games_seen,
                "wins": sum(result > 0 for result in results),
                "losses": sum(result < 0 for result in results),
                "draws": sum(result == 0 for result in results),
                "opponent_counts": dict(opponent_counts),
                "learner_team_counts": dict(learner_team_counts),
                "ppo": update,
                "snapshot": str(snapshot_path),
                "elapsed_seconds": time.time() - started,
            }
            with metrics_path.open("a") as file:
                file.write(json.dumps(row, sort_keys=True) + "\n")
            save_training_checkpoint(
                args.out_dir / "latest.pt",
                model,
                optimizer,
                iteration=iteration,
                games_seen=games_seen,
                ppo_config=ppo_config,
            )
            print(
                f"iteration {iteration}: wins={row['wins']} losses={row['losses']} "
                f"steps={int(update['steps'])} loss={update['loss']:.4f}",
                flush=True,
            )
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                evaluation = evaluate_policy(
                    worker,
                    model=model,
                    teams=holdout_teams,
                    opponents=eval_opponents,
                    games=args.eval_games,
                    seed=args.seed + 9_000_000,
                    device=args.device,
                )
                eval_row = {"iteration": iteration, "games_seen": games_seen, **evaluation}
                with eval_path.open("a") as file:
                    file.write(json.dumps(eval_row, sort_keys=True) + "\n")
                candidate_key = (
                    float(evaluation["min_opponent_win_rate"]),
                    float(evaluation["mean_opponent_win_rate"]),
                )
                if candidate_key > best_key:
                    best_key = candidate_key
                    save_training_checkpoint(
                        args.out_dir / "best.pt",
                        model,
                        optimizer,
                        iteration=iteration,
                        games_seen=games_seen,
                        ppo_config=ppo_config,
                        evaluation=evaluation,
                    )
                print(f"eval {iteration}: {evaluation}", flush=True)

    print(f"best checkpoint: {args.out_dir / 'best.pt'}")
    print(f"latest checkpoint: {args.out_dir / 'latest.pt'}")
    print(f"metrics: {metrics_path}")
    print(f"evaluations: {eval_path}")


if __name__ == "__main__":
    main()
