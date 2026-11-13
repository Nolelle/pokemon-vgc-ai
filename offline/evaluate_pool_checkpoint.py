#!/usr/bin/env python
"""Score one PPO checkpoint on a team-pool manifest against one baseline, in-process.

Eval wrapper for the many-teams warm-start experiment
(``docs/prereg/rl_multiteam_warmstart.md``). Unlike
``offline/evaluate_learned_policy.py`` -- which re-splits a manifest into its own
train/holdout arms -- this tool plays EVERY scheduled game on the manifest it is
given, so a run can be pointed at the pre-registered ``train_manifest.json`` /
``holdout_manifest.json`` split directly and compared checkpoint-to-checkpoint.

Games run on the in-process simulator (``SimWorker`` + ``play_battle`` +
``make_direct_agent``, same seam as ``selfplay/train_full_pipeline.py``'s
``evaluate_policy``) -- no Showdown server. ``--jobs`` shards game indices across
spawned worker processes, each with its own ``SimWorker`` and model load.

Determinism contract: everything about game ``g`` -- team draws, side, simulator
seed, the poke-env tie-break seed, and the policy's sampling seed -- is derived
from ``(seed, g)`` alone. Rows are therefore identical for any ``--jobs`` value,
and two checkpoints evaluated with the same ``--seed`` face the SAME games, which
is what makes checkpoint pairs comparable as paired games rather than two
independent win rates.

``--ablate trunkc`` freezes the state trunk to a constant (the ``trunkc`` arm of
``runs/pretrain_review/ablate_eval.py``), but with a wrinkle the shard layout
forces: the constant must be identical in every worker process, so each process
first replays game 0 of the schedule with a capturing patch, keeps the first
``encode_hidden`` output as the constant, discards that game, then plays its
assigned games with the frozen constant. Game 0's row is still produced (played
under the frozen trunk like every other game).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover - train extra is optional
    raise SystemExit(
        "evaluate_pool_checkpoint requires the train extra -- run `uv sync --extra train`."
    ) from exc

from selfplay.train_full_pipeline import TeamEntry, load_team_pool  # noqa: E402
from vgc.baselines import BASELINES  # noqa: E402
from vgc.config import FORMAT_ID  # noqa: E402
from vgc.evaluation import clustered_interval  # noqa: E402
from vgc.rl.agents import DirectAgent, make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402
from vgc.rl.match import play_battle  # noqa: E402
from vgc.rl.opponents import load_snapshot  # noqa: E402
from vgc.rl.player import PpoVgcPlayer  # noqa: E402
from vgc.rl.ppo import PpoConfig  # noqa: E402

ABLATIONS = ("none", "trunkc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "team-pool manifest (train_full_pipeline.load_team_pool format), e.g. "
            "data/selfplay/mc_sheet_pool_v2/holdout_manifest.json"
        ),
    )
    parser.add_argument(
        "--opponent",
        required=True,
        help=f"vgc.baselines name: {', '.join(sorted(BASELINES))}",
    )
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ablate", choices=ABLATIONS, default="none")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        while chunk := checkpoint_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def game_plan(
    seed: int, game_index: int, teams: list[TeamEntry]
) -> tuple[TeamEntry, TeamEntry, str, list[int]]:
    """The fixed schedule for game ``game_index``: both teams, the learner's side,
    and the simulator seed, all derived from ``(seed, game_index)`` only."""

    rng = random.Random(f"{seed}:{game_index}")
    learner_team = rng.choice(teams)
    opponent_team = rng.choice(teams)
    learner_side = "p1" if game_index % 2 == 0 else "p2"
    sim_seed = [rng.randrange(1, 2**31) for _ in range(4)]
    return learner_team, opponent_team, learner_side, sim_seed


def play_eval_game(
    worker: SimWorker,
    model,
    *,
    opponent: str,
    ablate: str,
    seed: int,
    game_index: int,
    teams: list[TeamEntry],
    battle_tag: str | None = None,
) -> dict[str, object]:
    """Play scheduled game ``game_index`` and return its per-game row.

    The learner is the checkpoint's stochastic seeded policy (deterministic=False,
    matching the trained policy and exp1's protocol); team preview is the player's
    own heuristic ``build_team_order`` for BOTH sides (no ``preview_order``),
    matching training.
    """

    learner_team, opponent_team, learner_side, sim_seed = game_plan(seed, game_index, teams)
    row: dict[str, object] = {
        "game_index": game_index,
        "learner_team": learner_team.label,
        "learner_archetype": learner_team.archetype,
        "opponent_team": opponent_team.label,
        "opponent_archetype": opponent_team.archetype,
        "side": learner_side,
        "opponent": opponent,
        "ablate": ablate,
    }
    try:
        # poke-env's built-in random/max-power tie breaks use Python's module-level
        # RNG; pin it per game so sharding cannot change a result.
        random.seed(seed * 1_000_003 + game_index)
        learner = DirectAgent(
            PpoVgcPlayer(
                model=model,
                rollout_buffer=None,
                ppo_config=PpoConfig(teacher_anchor_weight=0.0),
                deterministic=False,
                policy_seed=seed * 1000 + game_index,
                start_listening=False,
                team=learner_team.packed,
                battle_format=FORMAT_ID,
            ),
            name="policy",
        )
        opponent_agent = make_direct_agent(opponent, opponent_team.packed)
        other_side = "p2" if learner_side == "p1" else "p1"
        outcome = play_battle(
            worker,
            battle_tag or f"pool-eval-{game_index}",
            {learner_side: learner, other_side: opponent_agent},
            {learner_side: learner_team.packed, other_side: opponent_team.packed},
            battle_format=FORMAT_ID,
            seed=sim_seed,
        )
        result = outcome.result_for("policy")
        row.update(
            result=result,
            win=1 if result > 0 else 0,
            turns=outcome.turns,
            fallback_count=learner.player.fallback_count,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - one crashed game must not lose the shard
        row.update(
            result=None,
            win=0,
            turns=None,
            fallback_count=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    return row


def capture_trunk_constant(
    worker: SimWorker,
    model,
    *,
    opponent: str,
    seed: int,
    teams: list[TeamEntry],
) -> torch.Tensor:
    """Replay scheduled game 0 with a capturing patch and return the first
    ``encode_hidden`` output (``out[:1]``). The game's result is discarded -- every
    worker process runs this identical replay so all shards freeze the trunk to the
    SAME constant board encoding."""

    original = model.state_encoder.encode_hidden
    captured: dict[str, torch.Tensor] = {}

    def capturing(indices, scalars):
        out = original(indices, scalars)
        if "value" not in captured:
            captured["value"] = out[:1].detach().clone()
        return out

    model.state_encoder.encode_hidden = capturing
    try:
        play_eval_game(
            worker,
            model,
            opponent=opponent,
            ablate="trunkc",
            seed=seed,
            game_index=0,
            teams=teams,
            battle_tag="pool-eval-trunkref",
        )
    finally:
        model.state_encoder.encode_hidden = original
    if "value" not in captured:
        raise RuntimeError("trunkc: the game-0 replay produced no encode_hidden call")
    return captured["value"]


def install_trunkc(model, constant: torch.Tensor) -> None:
    """Freeze ``model.state_encoder.encode_hidden`` to ``constant`` for every call."""

    original = model.state_encoder.encode_hidden

    def patched(indices, scalars):
        out = original(indices, scalars)
        return constant.expand_as(out)

    model.state_encoder.encode_hidden = patched


def run_shard(job: dict[str, object]) -> list[dict[str, object]]:
    """One worker process: load the model, open one SimWorker, play its games."""

    torch.set_num_threads(1)
    teams = load_team_pool(Path(str(job["manifest"])))
    model = load_snapshot(Path(str(job["checkpoint"])), device="cpu")
    model.eval()
    opponent = str(job["opponent"])
    ablate = str(job["ablate"])
    seed = int(job["seed"])
    rows: list[dict[str, object]] = []
    with SimWorker() as worker:
        if ablate == "trunkc":
            constant = capture_trunk_constant(
                worker, model, opponent=opponent, seed=seed, teams=teams
            )
            install_trunkc(model, constant)
        for game_index in job["indices"]:
            rows.append(
                play_eval_game(
                    worker,
                    model,
                    opponent=opponent,
                    ablate=ablate,
                    seed=seed,
                    game_index=int(game_index),
                    teams=teams,
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    if args.games <= 0 or args.jobs <= 0:
        raise SystemExit("--games and --jobs must be positive")
    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest does not exist: {args.manifest}")
    if args.opponent not in BASELINES:
        raise SystemExit(
            f"unknown opponent {args.opponent!r}; available: {', '.join(sorted(BASELINES))}"
        )
    # Validate the manifest in the parent too so a bad file fails before any
    # worker spawns, and load the payload once for the summary's provenance.
    load_team_pool(args.manifest)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    worker_count = min(args.jobs, args.games)
    jobs = [
        {
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "opponent": args.opponent,
            "ablate": args.ablate,
            "seed": args.seed,
            "indices": list(range(worker_id, args.games, worker_count)),
        }
        for worker_id in range(worker_count)
    ]
    started = time.time()
    ctx = multiprocessing.get_context("spawn")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as pool:
        for shard_rows in pool.map(run_shard, jobs):
            rows.extend(shard_rows)
    wall_seconds = time.time() - started
    rows.sort(key=lambda row: int(row["game_index"]))

    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_out.open("w") as rows_file:
        for row in rows:
            rows_file.write(json.dumps(row, sort_keys=True) + "\n")

    completed = [row for row in rows if row["error"] is None]
    wins = sum(int(row["win"]) for row in completed)
    fallback_total = sum(int(row["fallback_count"]) for row in completed)
    by_learner_team: dict[str, list[int]] = {}
    for row in completed:
        cluster = by_learner_team.setdefault(str(row["learner_team"]), [0, 0])
        cluster[0] += int(row["win"])
        cluster[1] += 1
    clusters = list(by_learner_team.values())
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "iteration": int(payload.get("iteration", 0)),
        "games_seen": int(payload.get("games_seen", 0)),
        "manifest": str(args.manifest),
        "opponent": args.opponent,
        "seed": args.seed,
        "ablate": args.ablate,
        "games": args.games,
        "completed": len(completed),
        "wins": wins,
        "win_rate": wins / len(completed) if completed else 0.0,
        "errors": args.games - len(completed),
        "fallback_count": fallback_total,
        "clustered_interval": list(clustered_interval(clusters)),
        "wall_seconds": round(wall_seconds, 3),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"{args.opponent} ({args.ablate}): {wins}/{len(completed)} wins "
        f"({summary['win_rate']:.3f}), errors={summary['errors']}, "
        f"{wall_seconds:.1f}s -> {args.summary_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
