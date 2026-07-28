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
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402

from selfplay.run_selfplay import _safe_stop_listening, start_local_server  # noqa: E402
from vgc.agent import VgcPlayer  # noqa: E402
from vgc.config import FORMAT_ID, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.distill import (  # noqa: E402
    DistillationConfig,
    DistillationSample,
    TeacherRecordingPlayer,
    distill_policy,
    evaluate_agreement,
    split_samples_by_battle,
)
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
DEFAULT_TEAM_POOL_DIR = REPO_ROOT / "data" / "selfplay" / "pool"
DEFAULT_DEV_TEAM = TEAMS_DIR / "dev.packed.txt"
WORKER_TIMEOUT_PER_GAME_SECONDS = 120.0
# Opt-in "archetype pool mode" defaults (see `--archetype-pool` below). Unrelated to
# the flat `--holdout-team-fraction` path, which stays untouched.
DEFAULT_HOLDOUT_TEAMS_PER_ARCHETYPE = 2


@dataclass(frozen=True)
class OpponentTeamChoice:
    label: str
    packed: str
    group: str  # "mirror" or "diverse"


def load_diverse_opponent_teams(pool_dir: Path, dev_team_path: Path) -> list[OpponentTeamChoice]:
    paths = [dev_team_path]
    if pool_dir.exists():
        paths.extend(sorted(pool_dir.glob("*.packed.txt")))
    choices: list[OpponentTeamChoice] = []
    for path in paths:
        if not path.exists():
            continue
        packed = path.read_text().strip()
        if packed:
            choices.append(OpponentTeamChoice(label=path.stem, packed=packed, group="diverse"))
    if not choices:
        raise ValueError("no diverse opponent teams were found")
    return choices


@dataclass(frozen=True)
class PoolTeam:
    """One team from the archetype-tagged pool (`tools/build_archetype_pool.py`'s
    ``manifest.json``). Unlike ``OpponentTeamChoice``, which only ever describes an
    OPPONENT's team, a ``PoolTeam`` can be used as either side -- archetype-pool mode
    draws both the learner's and the opponent's team from the same list of these.
    """

    label: str
    packed: str
    archetype: str
    source: str  # "anchor" or "variant"


def load_archetype_pool(manifest_path: Path) -> list[PoolTeam]:
    """Load ``tools/build_archetype_pool.py``'s archetype-tagged team pool.

    ``manifest_path`` is the ``manifest.json`` written by that tool: a JSON list of
    ``{"file": <path relative to manifest_path's directory>, "archetype": <label>,
    "source": "anchor"|"variant", "species": [...]}`` records. Each referenced packed
    team file is read and validated non-empty here so a caller gets one clear
    ``SystemExit`` up front instead of a worker crashing deep into a run.
    """

    if not manifest_path.exists():
        raise SystemExit(f"archetype pool manifest does not exist: {manifest_path}")
    try:
        records = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"archetype pool manifest is not valid JSON: {manifest_path}: {exc}")
    manifest_dir = manifest_path.parent
    pool_teams: list[PoolTeam] = []
    for record in records:
        team_path = manifest_dir / record["file"]
        if not team_path.exists():
            raise SystemExit(f"archetype pool team file does not exist: {team_path}")
        packed = team_path.read_text().strip()
        if not packed:
            raise SystemExit(f"archetype pool team file is empty: {team_path}")
        pool_teams.append(
            PoolTeam(
                label=Path(record["file"]).stem,
                packed=packed,
                archetype=record["archetype"],
                source=record["source"],
            )
        )
    if not pool_teams:
        raise SystemExit(f"archetype pool manifest has no teams: {manifest_path}")
    return pool_teams


def split_holdout_teams(
    diverse_teams: list[OpponentTeamChoice],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[OpponentTeamChoice], list[OpponentTeamChoice]]:
    """Deterministically split ``diverse_teams`` into (train, held-out) subsets.

    Held-out teams are never used as training opponents; evaluating separately
    against seen vs. held-out teams measures whether the policy generalizes beyond
    the specific opponent matchups it trained on. ``holdout_fraction == 0.0`` is a
    no-op that preserves current behavior exactly (returns all teams as train, no
    held-out teams).
    """

    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0.0 (inclusive) and 1.0 (exclusive)")
    if holdout_fraction == 0.0:
        return list(diverse_teams), []
    if len(diverse_teams) < 2:
        raise ValueError("need at least 2 diverse teams to hold any out")
    shuffled = list(diverse_teams)
    random.Random(seed).shuffle(shuffled)
    holdout_count = round(holdout_fraction * len(shuffled))
    holdout_count = min(max(holdout_count, 1), len(shuffled) - 1)
    train_teams = shuffled[:-holdout_count]
    holdout_teams = shuffled[-holdout_count:]
    return train_teams, holdout_teams


def split_holdout_by_archetype(
    pool_teams: list[PoolTeam],
    *,
    holdout_per_archetype: int,
    seed: int,
) -> tuple[list[PoolTeam], list[PoolTeam]]:
    """WITHIN-archetype held-out split over the archetype-tagged pool.

    Unlike ``split_holdout_teams`` (a flat fraction over one undifferentiated list),
    this holds out ``holdout_per_archetype`` teams from EACH archetype separately, so
    generalization can be measured "did it learn this archetype" rather than "did it
    memorize this specific team out of an arbitrary cross-archetype pool". The split is
    deterministic (a per-archetype seeded shuffle) and clamps down to
    ``len(archetype_teams) - 1`` so at least one TRAIN team always survives per
    archetype; an archetype with only 1 team therefore contributes 0 held-out / 1
    train. Returns ``(train_teams, holdout_teams)``, each flattened across archetypes
    in sorted-label order (independent of ``pool_teams``' input order).
    """

    if holdout_per_archetype < 0:
        raise ValueError("holdout_per_archetype must be nonnegative")
    if not pool_teams:
        raise ValueError("pool_teams must be non-empty")
    by_archetype: dict[str, list[PoolTeam]] = {}
    for team in pool_teams:
        by_archetype.setdefault(team.archetype, []).append(team)
    train_teams: list[PoolTeam] = []
    holdout_teams: list[PoolTeam] = []
    for archetype in sorted(by_archetype):
        archetype_teams = list(by_archetype[archetype])
        random.Random(f"{seed}:{archetype}").shuffle(archetype_teams)
        max_holdout = max(len(archetype_teams) - 1, 0)
        holdout_count = min(holdout_per_archetype, max_holdout)
        holdout_teams.extend(archetype_teams[:holdout_count])
        train_teams.extend(archetype_teams[holdout_count:])
    return train_teams, holdout_teams


def build_opponent_team_schedule(
    worker_count: int,
    *,
    learner_team: str,
    diverse_teams: list[OpponentTeamChoice],
    mirror_fraction: float,
    seed: int,
    pair_groups: bool = False,
) -> list[OpponentTeamChoice]:
    """Exact mirror/diverse worker mix with deterministic diverse-team rotation."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not diverse_teams:
        raise ValueError("diverse_teams must be non-empty")
    if not 0.0 <= mirror_fraction <= 1.0:
        raise ValueError("mirror_fraction must be between 0 and 1")
    rng = random.Random(seed)
    mirror_count = round(worker_count * mirror_fraction)
    flags = [True] * mirror_count + [False] * (worker_count - mirror_count)
    if not pair_groups:
        rng.shuffle(flags)
    diverse_order = list(diverse_teams)
    rng.shuffle(diverse_order)
    diverse_index = 0
    schedule: list[OpponentTeamChoice] = []
    for mirror in flags:
        if mirror:
            schedule.append(
                OpponentTeamChoice(label="meta1-mirror", packed=learner_team, group="mirror")
            )
        else:
            schedule.append(diverse_order[diverse_index % len(diverse_order)])
            diverse_index += 1
    return schedule


def sample_learner_and_opponent(
    pool_teams: list[PoolTeam],
    *,
    rng: random.Random,
    mirror_fraction: float,
) -> tuple[PoolTeam, OpponentTeamChoice]:
    """Draw one worker's (learner, opponent) team pair from ``pool_teams``.

    With probability ``mirror_fraction`` the opponent is the SAME team as the learner
    (``group="mirror"``); otherwise the opponent is an independently drawn team from
    the same list (``group="diverse"``), which may coincidentally equal the learner's
    team by chance -- that's fine, it just isn't the deliberate mirror case. ``rng`` is
    threaded in (rather than seeded here) so a caller can draw a whole worker schedule
    from one shared, seeded ``random.Random`` for determinism across workers.
    """

    if not pool_teams:
        raise ValueError("pool_teams must be non-empty")
    if not 0.0 <= mirror_fraction <= 1.0:
        raise ValueError("mirror_fraction must be between 0 and 1")
    learner = rng.choice(pool_teams)
    mirrored = rng.random() < mirror_fraction
    opponent_pool_team = learner if mirrored else rng.choice(pool_teams)
    opponent = OpponentTeamChoice(
        label=f"{opponent_pool_team.archetype}:{opponent_pool_team.label}",
        packed=opponent_pool_team.packed,
        group="mirror" if mirrored else "diverse",
    )
    return learner, opponent


def build_pool_worker_assignments(
    worker_count: int,
    pool_teams: list[PoolTeam],
    *,
    mirror_fraction: float,
    seed: int,
) -> list[tuple[PoolTeam, OpponentTeamChoice]]:
    """One ``sample_learner_and_opponent`` draw per worker, all fed from a single
    seeded ``random.Random`` so the whole worker schedule is deterministic and
    reproducible for a given ``seed``. This is what replaces the fixed-learner
    ``build_opponent_team_schedule`` in archetype-pool mode: every worker gets its own
    independently-sampled learner team, not just its own opponent team.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not pool_teams:
        raise ValueError("pool_teams must be non-empty")
    rng = random.Random(seed)
    return [
        sample_learner_and_opponent(pool_teams, rng=rng, mirror_fraction=mirror_fraction)
        for _ in range(worker_count)
    ]


def build_same_archetype_pool_worker_assignments(
    worker_count: int,
    pool_teams: list[PoolTeam],
    *,
    mirror_fraction: float,
    seed: int,
) -> list[tuple[PoolTeam, OpponentTeamChoice]]:
    """Like ``build_pool_worker_assignments``, but each worker draws BOTH its learner
    and opponent team from a single archetype's subset of ``pool_teams``, iterating
    archetypes round-robin across workers (worker ``i`` uses the
    ``i % archetype_count``-th archetype in sorted-label order).

    Design choice: this exists specifically so the held-out generalization arm's
    per-archetype win-rate breakdown is meaningful (see
    ``evaluate_with_generalization_by_archetype``'s ``holdout_by_archetype``). A worker
    whose learner plays a `charizard_sun_offense` held-out team against an unrelated
    `sand_offense` held-out team would tell you nothing about whether the policy
    generalized WITHIN either archetype -- pairing same-archetype teams on both sides
    is what makes "win rate for archetype X" a coherent number.
    """

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not pool_teams:
        raise ValueError("pool_teams must be non-empty")
    by_archetype: dict[str, list[PoolTeam]] = {}
    for team in pool_teams:
        by_archetype.setdefault(team.archetype, []).append(team)
    archetype_labels = sorted(by_archetype)
    rng = random.Random(seed)
    assignments: list[tuple[PoolTeam, OpponentTeamChoice]] = []
    for worker_id in range(worker_count):
        archetype = archetype_labels[worker_id % len(archetype_labels)]
        assignments.append(
            sample_learner_and_opponent(
                by_archetype[archetype], rng=rng, mirror_fraction=mirror_fraction
            )
        )
    return assignments


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


def load_training_checkpoint(
    path: Path,
    model: CandidatePolicyValueNet,
    optimizer: torch.optim.Optimizer,
    *,
    device: str,
) -> tuple[int, int, PpoConfig]:
    """Restore training state and return iteration, game count, and PPO settings.

    Raises if the checkpoint's `use_meta_features` disagrees with `model`'s -- a
    meta-on checkpoint must never be silently loaded into a meta-off model or vice
    versa, since the two have different parameter shapes (`context_encoder`'s input
    width, presence/absence of `meta_encoder`).
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture != RL_ARCHITECTURE_VERSION:
        raise ValueError(
            f"unsupported PPO checkpoint architecture {architecture!r}; "
            f"expected {RL_ARCHITECTURE_VERSION!r}"
        )
    checkpoint_use_meta = bool(checkpoint.get("use_meta_features", False))
    if checkpoint_use_meta != model.use_meta_features:
        raise ValueError(
            f"checkpoint use_meta_features={checkpoint_use_meta} does not match "
            f"requested model use_meta_features={model.use_meta_features}; pass/omit "
            "--meta-features to match the checkpoint it was trained with"
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
    learner_team: str,
    opponent_team: OpponentTeamChoice,
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
        team=learner_team,
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
            team=opponent_team.packed,
            battle_format=FORMAT_ID,
            account_configuration=AccountConfiguration(f"ppos{worker_id}-{token}", None),
        )
    else:
        opponent = VgcPlayer(
            config=common_config,
            team=opponent_team.packed,
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
        "opponent_team": opponent_team.label,
        "team_group": opponent_team.group,
        "error": error,
    }, buffer


async def _collect_teacher_worker(
    *,
    worker_id: int,
    games: int,
    learner_team: str,
    opponent_team: OpponentTeamChoice,
) -> tuple[dict[str, object], list[DistillationSample]]:
    token = secrets.token_hex(3)
    teacher_config = replace(
        PolicyConfig(),
        accept_open_team_sheet=False,
        use_rolling_horizon=False,
    )
    teacher = TeacherRecordingPlayer(
        config=teacher_config,
        team=learner_team,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"teach{worker_id}-{token}", None),
    )
    opponent = VgcPlayer(
        config=teacher_config,
        team=opponent_team.packed,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"teachopp{worker_id}-{token}", None),
    )
    error: str | None = None
    try:
        await asyncio.wait_for(
            teacher.battle_against(opponent, n_battles=games),
            timeout=max(1, games) * WORKER_TIMEOUT_PER_GAME_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - other bootstrap workers remain useful
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await _safe_stop_listening(teacher)
        await _safe_stop_listening(opponent)
    samples = teacher.completed_samples()
    return {
        "worker": worker_id,
        "requested": games,
        "games": teacher.n_finished_battles,
        "samples": len(samples),
        "opponent_team": opponent_team.label,
        "team_group": opponent_team.group,
        "error": error,
    }, samples


async def _evaluate_worker(
    model: CandidatePolicyValueNet,
    *,
    worker_id: int,
    games: int,
    learner_team: str,
    opponent_team: OpponentTeamChoice,
    device: str,
    learner_challenges: bool,
) -> dict[str, object]:
    token = secrets.token_hex(3)
    common_config = replace(
        PolicyConfig(),
        accept_open_team_sheet=False,
        use_rolling_horizon=False,
    )
    learner = PpoVgcPlayer(
        model=model,
        rollout_buffer=None,
        device=device,
        deterministic=True,
        config=common_config,
        team=learner_team,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"evalrl{worker_id}-{token}", None),
    )
    opponent = VgcPlayer(
        config=common_config,
        team=opponent_team.packed,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"evalh{worker_id}-{token}", None),
    )
    error: str | None = None
    try:
        challenger, receiver = (learner, opponent) if learner_challenges else (opponent, learner)
        await asyncio.wait_for(
            challenger.battle_against(receiver, n_battles=games),
            timeout=max(1, games) * WORKER_TIMEOUT_PER_GAME_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - other evaluation workers remain useful
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await _safe_stop_listening(learner)
        await _safe_stop_listening(opponent)
    return {
        "worker": worker_id,
        "requested": games,
        "games": learner.n_finished_battles,
        "wins": learner.n_won_battles,
        "losses": learner.n_lost_battles,
        "side": "challenger" if learner_challenges else "receiver",
        "opponent_team": opponent_team.label,
        "team_group": opponent_team.group,
        "error": error,
    }


def allocate_games(games: int, jobs: int) -> list[int]:
    """Balanced positive worker chunks that sum exactly to ``games``."""

    if games <= 0 or jobs <= 0:
        raise ValueError("games and jobs must be positive")
    worker_count = min(games, jobs)
    base, remainder = divmod(games, worker_count)
    return [base + int(index < remainder) for index in range(worker_count)]


async def collect_teacher_samples(
    *,
    games: int,
    jobs: int,
    learner_team: str,
    diverse_teams: list[OpponentTeamChoice],
    mirror_fraction: float,
    seed: int,
) -> tuple[list[DistillationSample], dict[str, object]]:
    allocations = allocate_games(games, jobs)
    teams = build_opponent_team_schedule(
        len(allocations),
        learner_team=learner_team,
        diverse_teams=diverse_teams,
        mirror_fraction=mirror_fraction,
        seed=seed,
    )
    worker_results = await asyncio.gather(
        *(
            _collect_teacher_worker(
                worker_id=worker_id,
                games=count,
                learner_team=learner_team,
                opponent_team=teams[worker_id],
            )
            for worker_id, count in enumerate(allocations)
        )
    )
    samples: list[DistillationSample] = []
    rows: list[dict[str, object]] = []
    for row, worker_samples in worker_results:
        rows.append(row)
        samples.extend(worker_samples)
    return samples, {
        "games": sum(int(row["games"]) for row in rows),
        "samples": len(samples),
        "worker_errors": sum(row["error"] is not None for row in rows),
        "workers": rows,
    }


async def collect_teacher_samples_from_pool(
    *,
    games: int,
    jobs: int,
    pool_teams: list[PoolTeam],
    mirror_fraction: float,
    seed: int,
) -> tuple[list[DistillationSample], dict[str, object]]:
    """Archetype-pool analogue of ``collect_teacher_samples``: each worker's learner
    team is ALSO drawn from ``pool_teams`` (not one fixed team shared by every
    worker), via ``build_pool_worker_assignments``. Callers must pass only TRAIN pool
    teams (never held-out) -- same requirement as the non-pool bootstrap path.
    """

    allocations = allocate_games(games, jobs)
    assignments = build_pool_worker_assignments(
        len(allocations), pool_teams, mirror_fraction=mirror_fraction, seed=seed
    )
    worker_results = await asyncio.gather(
        *(
            _collect_teacher_worker(
                worker_id=worker_id,
                games=count,
                learner_team=learner.packed,
                opponent_team=opponent,
            )
            for worker_id, (count, (learner, opponent)) in enumerate(
                zip(allocations, assignments)
            )
        )
    )
    samples: list[DistillationSample] = []
    rows: list[dict[str, object]] = []
    for (row, worker_samples), (learner, _opponent) in zip(worker_results, assignments):
        rows.append(
            {**row, "learner_team": learner.label, "learner_archetype": learner.archetype}
        )
        samples.extend(worker_samples)
    return samples, {
        "games": sum(int(row["games"]) for row in rows),
        "samples": len(samples),
        "worker_errors": sum(row["error"] is not None for row in rows),
        "workers": rows,
    }


async def evaluate_frozen_policy(
    model: CandidatePolicyValueNet,
    *,
    games: int,
    jobs: int,
    learner_team: str,
    diverse_teams: list[OpponentTeamChoice],
    mirror_fraction: float,
    seed: int,
    device: str,
) -> dict[str, object]:
    allocations = allocate_games(games, jobs)
    teams = build_opponent_team_schedule(
        len(allocations),
        learner_team=learner_team,
        diverse_teams=diverse_teams,
        mirror_fraction=mirror_fraction,
        seed=seed,
        pair_groups=True,
    )
    rows = await asyncio.gather(
        *(
            _evaluate_worker(
                model,
                worker_id=worker_id,
                games=count,
                learner_team=learner_team,
                opponent_team=teams[worker_id],
                device=device,
                learner_challenges=worker_id % 2 == 0,
            )
            for worker_id, count in enumerate(allocations)
        )
    )
    completed = sum(int(row["games"]) for row in rows)
    wins = sum(int(row["wins"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    by_team_group: dict[str, dict[str, float | int]] = {}
    for group in ("mirror", "diverse"):
        group_rows = [row for row in rows if row["team_group"] == group]
        group_games = sum(int(row["games"]) for row in group_rows)
        group_wins = sum(int(row["wins"]) for row in group_rows)
        by_team_group[group] = {
            "games": group_games,
            "wins": group_wins,
            "win_rate": group_wins / group_games if group_games else 0.0,
        }
    return {
        "games": completed,
        "wins": wins,
        "losses": losses,
        "draws": completed - wins - losses,
        "win_rate": wins / completed if completed else 0.0,
        "worker_errors": sum(row["error"] is not None for row in rows),
        "deterministic": True,
        "opponent": "vgc-shallow-search",
        "by_team_group": by_team_group,
        "workers": rows,
    }


async def evaluate_pool_policy(
    model: CandidatePolicyValueNet,
    *,
    games: int,
    jobs: int,
    pool_teams: list[PoolTeam],
    mirror_fraction: float,
    seed: int,
    device: str,
    same_archetype_pairing: bool = False,
) -> dict[str, object]:
    """Archetype-pool analogue of ``evaluate_frozen_policy``: each worker's learner AND
    opponent team are BOTH drawn from ``pool_teams`` (see
    ``sample_learner_and_opponent``) instead of one fixed learner team facing a
    schedule of opponent-only teams.

    When ``same_archetype_pairing`` is True, each worker draws both teams from a
    single archetype's subset of ``pool_teams`` (round-robin across workers -- see
    ``build_same_archetype_pool_worker_assignments``), which is what makes the
    returned ``by_archetype`` breakdown meaningful; the held-out arm of
    ``evaluate_with_generalization_by_archetype`` uses this so a low/high win rate can
    be attributed to a specific archetype instead of averaged across an arbitrary
    cross-archetype matchup.
    """

    allocations = allocate_games(games, jobs)
    if same_archetype_pairing:
        assignments = build_same_archetype_pool_worker_assignments(
            len(allocations), pool_teams, mirror_fraction=mirror_fraction, seed=seed
        )
    else:
        assignments = build_pool_worker_assignments(
            len(allocations), pool_teams, mirror_fraction=mirror_fraction, seed=seed
        )
    rows = await asyncio.gather(
        *(
            _evaluate_worker(
                model,
                worker_id=worker_id,
                games=count,
                learner_team=learner.packed,
                opponent_team=opponent,
                device=device,
                learner_challenges=worker_id % 2 == 0,
            )
            for worker_id, (count, (learner, opponent)) in enumerate(
                zip(allocations, assignments)
            )
        )
    )
    enriched_rows: list[dict[str, object]] = []
    by_archetype: dict[str, dict[str, float | int]] = {}
    for row, (learner, _opponent) in zip(rows, assignments):
        enriched_rows.append(
            {**row, "learner_team": learner.label, "learner_archetype": learner.archetype}
        )
        entry = by_archetype.setdefault(learner.archetype, {"games": 0, "wins": 0})
        entry["games"] = int(entry["games"]) + int(row["games"])
        entry["wins"] = int(entry["wins"]) + int(row["wins"])
    for entry in by_archetype.values():
        games_played = int(entry["games"])
        entry["win_rate"] = entry["wins"] / games_played if games_played else 0.0
    completed = sum(int(row["games"]) for row in rows)
    wins = sum(int(row["wins"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    by_team_group: dict[str, dict[str, float | int]] = {}
    for group in ("mirror", "diverse"):
        group_rows = [row for row in rows if row["team_group"] == group]
        group_games = sum(int(row["games"]) for row in group_rows)
        group_wins = sum(int(row["wins"]) for row in group_rows)
        by_team_group[group] = {
            "games": group_games,
            "wins": group_wins,
            "win_rate": group_wins / group_games if group_games else 0.0,
        }
    return {
        "games": completed,
        "wins": wins,
        "losses": losses,
        "draws": completed - wins - losses,
        "win_rate": wins / completed if completed else 0.0,
        "worker_errors": sum(row["error"] is not None for row in rows),
        "deterministic": True,
        "opponent": "vgc-shallow-search",
        "by_team_group": by_team_group,
        "by_archetype": by_archetype,
        "workers": enriched_rows,
    }


async def evaluate_with_generalization(
    model: CandidatePolicyValueNet,
    *,
    eval_games: int,
    eval_jobs: int,
    learner_team: str,
    diverse_teams: list[OpponentTeamChoice],
    train_teams: list[OpponentTeamChoice],
    holdout_teams: list[OpponentTeamChoice],
    eval_mirror_fraction: float,
    generalization_games: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Run the standard frozen evaluation, plus a seen-vs-held-out check.

    The primary evaluation is unchanged (same ``diverse_teams``/``mirror_fraction`` as
    before held-out support existed), so ``evaluation["win_rate"]``/structure stays
    byte-for-byte identical when ``holdout_teams`` is empty. When teams are held out,
    two extra diverse-only (``mirror_fraction=0.0``) evaluations run against the train
    subset ("seen") and the held-out subset, and their win rates/gap are attached under
    ``evaluation["generalization"]``. Factored out so the periodic (``--eval-every-
    games``) and final evaluation call sites can't drift apart.
    """

    evaluation = await evaluate_frozen_policy(
        model,
        games=eval_games,
        jobs=eval_jobs,
        learner_team=learner_team,
        diverse_teams=diverse_teams,
        mirror_fraction=eval_mirror_fraction,
        seed=seed,
        device=device,
    )
    if not holdout_teams:
        return evaluation
    if generalization_games:
        # Explicit per-arm budget: the seen-vs-held-out gap is a difference of two win
        # rates, so its noise is wider than either arm's. Sizing both arms directly is
        # the only way to make the gap statistically meaningful (this repo measures
        # +/-4-6 points of run-to-run variance at n=100-300 -- see CLAUDE.md).
        seen_games = generalization_games
        holdout_games = generalization_games
    else:
        # Keep the generalization check cheap relative to the primary evaluation: it's a
        # diagnostic, not the main eval signal, so it gets half of --eval-games (or
        # --eval-jobs, whichever is larger, so every worker still gets at least one game)
        # split across the seen and held-out arms rather than a full --eval-games budget
        # for each.
        generalization_budget = max(eval_jobs, eval_games // 2)
        seen_games = max(1, generalization_budget // 2)
        holdout_games = max(1, generalization_budget - seen_games)
    seen_eval, holdout_eval = await asyncio.gather(
        evaluate_frozen_policy(
            model,
            games=seen_games,
            jobs=eval_jobs,
            learner_team=learner_team,
            diverse_teams=train_teams,
            mirror_fraction=0.0,
            seed=seed,
            device=device,
        ),
        evaluate_frozen_policy(
            model,
            games=holdout_games,
            jobs=eval_jobs,
            learner_team=learner_team,
            diverse_teams=holdout_teams,
            mirror_fraction=0.0,
            seed=seed,
            device=device,
        ),
    )
    evaluation["generalization"] = {
        "seen_win_rate": seen_eval["win_rate"],
        "holdout_win_rate": holdout_eval["win_rate"],
        "generalization_gap": seen_eval["win_rate"] - holdout_eval["win_rate"],
        "seen_games": seen_eval["games"],
        "holdout_games": holdout_eval["games"],
        "seen": seen_eval,
        "holdout": holdout_eval,
    }
    return evaluation


async def evaluate_with_generalization_by_archetype(
    model: CandidatePolicyValueNet,
    *,
    eval_games: int,
    eval_jobs: int,
    train_pool_teams: list[PoolTeam],
    holdout_pool_teams: list[PoolTeam],
    eval_mirror_fraction: float,
    generalization_games: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Archetype-pool analogue of ``evaluate_with_generalization``.

    The headline evaluation (``evaluation["win_rate"]``) runs across TRAIN pool teams
    with BOTH learner and opponent drawn per-worker (``evaluate_pool_policy``), since
    in archetype-pool mode the learner has no single fixed team to evaluate -- this is
    what keeps the headline number reflecting varied-team play rather than meta1.

    When ``holdout_pool_teams`` is non-empty (``--holdout-teams-per-archetype > 0``),
    also runs a seen-vs-held-out check using the exact same
    ``evaluation["generalization"]`` schema ``evaluate_with_generalization`` produces
    (``seen_win_rate``/``holdout_win_rate``/``generalization_gap``/``seen_games``/
    ``holdout_games``/``seen``/``holdout``), so ``_tracked_win_rate`` and best-model
    tracking keep working unchanged. Additionally attaches
    ``generalization["holdout_by_archetype"]`` from the held-out arm's
    same-archetype-paired ``by_archetype`` breakdown (see ``evaluate_pool_policy``'s
    ``same_archetype_pairing``) -- a diagnostic since per-archetype game counts are
    small; the pooled ``holdout_win_rate`` remains the primary number.
    """

    evaluation = await evaluate_pool_policy(
        model,
        games=eval_games,
        jobs=eval_jobs,
        pool_teams=train_pool_teams,
        mirror_fraction=eval_mirror_fraction,
        seed=seed,
        device=device,
    )
    if not holdout_pool_teams:
        return evaluation
    if generalization_games:
        # Same explicit per-arm budget rationale as `evaluate_with_generalization`.
        seen_games = generalization_games
        holdout_games = generalization_games
    else:
        generalization_budget = max(eval_jobs, eval_games // 2)
        seen_games = max(1, generalization_budget // 2)
        holdout_games = max(1, generalization_budget - seen_games)
    seen_eval, holdout_eval = await asyncio.gather(
        evaluate_pool_policy(
            model,
            games=seen_games,
            jobs=eval_jobs,
            pool_teams=train_pool_teams,
            mirror_fraction=0.0,
            seed=seed,
            device=device,
            # Both arms use same-archetype pairing so the seen-vs-holdout gap isolates
            # seen-vs-unseen teams ONLY -- if the seen arm free-mixed archetypes while
            # the holdout arm paired within-archetype, the gap would also reflect the
            # difference in matchup structure, not just memorization.
            same_archetype_pairing=True,
        ),
        evaluate_pool_policy(
            model,
            games=holdout_games,
            jobs=eval_jobs,
            pool_teams=holdout_pool_teams,
            mirror_fraction=0.0,
            seed=seed,
            device=device,
            same_archetype_pairing=True,
        ),
    )
    evaluation["generalization"] = {
        "seen_win_rate": seen_eval["win_rate"],
        "holdout_win_rate": holdout_eval["win_rate"],
        "generalization_gap": seen_eval["win_rate"] - holdout_eval["win_rate"],
        "seen_games": seen_eval["games"],
        "holdout_games": holdout_eval["games"],
        "seen": seen_eval,
        "holdout": holdout_eval,
        "holdout_by_archetype": holdout_eval["by_archetype"],
    }
    return evaluation


async def collect_games(
    model: CandidatePolicyValueNet,
    buffer: RolloutBuffer,
    *,
    games: int,
    jobs: int,
    learner_team: str,
    diverse_teams: list[OpponentTeamChoice],
    mirror_fraction: float,
    device: str,
    ppo_config: PpoConfig,
    snapshots: list[Path],
    heuristic_fraction: float,
    seed: int,
) -> dict[str, object]:
    allocations = allocate_games(games, jobs)
    teams = build_opponent_team_schedule(
        len(allocations),
        learner_team=learner_team,
        diverse_teams=diverse_teams,
        mirror_fraction=mirror_fraction,
        seed=seed + 50_000,
    )
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
                learner_team=learner_team,
                opponent_team=teams[worker_id],
                device=device,
                ppo_config=ppo_config,
                opponent_spec=specs[worker_id],
            )
            for worker_id, worker_games in enumerate(allocations)
        )
    )
    rows: list[dict[str, object]] = []
    opponent_counts: dict[str, int] = {}
    opponent_team_counts: dict[str, int] = {}
    for row, worker_buffer in worker_results:
        buffer.extend_finished(worker_buffer)
        rows.append(row)
        label = str(row["opponent"])
        opponent_counts[label] = opponent_counts.get(label, 0) + int(row["games"])
        team_label = str(row["opponent_team"])
        opponent_team_counts[team_label] = opponent_team_counts.get(team_label, 0) + int(
            row["games"]
        )
    return {
        "games": sum(int(row["games"]) for row in rows),
        "wins": sum(int(row["wins"]) for row in rows),
        "losses": sum(int(row["losses"]) for row in rows),
        "steps": len(buffer),
        "worker_errors": sum(row["error"] is not None for row in rows),
        "opponents": opponent_counts,
        "opponent_teams": opponent_team_counts,
        "workers": rows,
    }


async def collect_games_from_pool(
    model: CandidatePolicyValueNet,
    buffer: RolloutBuffer,
    *,
    games: int,
    jobs: int,
    pool_teams: list[PoolTeam],
    mirror_fraction: float,
    device: str,
    ppo_config: PpoConfig,
    snapshots: list[Path],
    heuristic_fraction: float,
    seed: int,
) -> dict[str, object]:
    """Archetype-pool analogue of ``collect_games``: each worker's learner team is
    drawn from ``pool_teams`` per-worker (``build_pool_worker_assignments``) instead of
    one fixed ``learner_team`` shared by every worker -- this is the actual "learner
    plays varied teams" behavior change from the fixed-meta1-learner design. Adds a
    ``learner_teams`` count (analogous to the existing ``opponent_teams`` count) to the
    returned dict so ``metrics.jsonl`` shows learner-team variety directly.
    """

    allocations = allocate_games(games, jobs)
    assignments = build_pool_worker_assignments(
        len(allocations), pool_teams, mirror_fraction=mirror_fraction, seed=seed + 50_000
    )
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
                learner_team=learner.packed,
                opponent_team=opponent,
                device=device,
                ppo_config=ppo_config,
                opponent_spec=specs[worker_id],
            )
            for worker_id, (worker_games, (learner, opponent)) in enumerate(
                zip(allocations, assignments)
            )
        )
    )
    rows: list[dict[str, object]] = []
    opponent_counts: dict[str, int] = {}
    opponent_team_counts: dict[str, int] = {}
    learner_team_counts: dict[str, int] = {}
    for (row, worker_buffer), (learner, _opponent) in zip(worker_results, assignments):
        buffer.extend_finished(worker_buffer)
        rows.append(
            {**row, "learner_team": learner.label, "learner_archetype": learner.archetype}
        )
        label = str(row["opponent"])
        opponent_counts[label] = opponent_counts.get(label, 0) + int(row["games"])
        team_label = str(row["opponent_team"])
        opponent_team_counts[team_label] = opponent_team_counts.get(team_label, 0) + int(
            row["games"]
        )
        learner_label = f"{learner.archetype}:{learner.label}"
        learner_team_counts[learner_label] = learner_team_counts.get(learner_label, 0) + int(
            row["games"]
        )
    return {
        "games": sum(int(row["games"]) for row in rows),
        "wins": sum(int(row["wins"]) for row in rows),
        "losses": sum(int(row["losses"]) for row in rows),
        "steps": len(buffer),
        "worker_errors": sum(row["error"] is not None for row in rows),
        "opponents": opponent_counts,
        "opponent_teams": opponent_team_counts,
        "learner_teams": learner_team_counts,
        "workers": rows,
    }


def _tracked_win_rate(evaluation: dict[str, object]) -> float:
    """Win rate used for best-model tracking and resume recovery.

    Prefers the held-out generalization win rate when a ``generalization`` block is
    present (the honest measure of skill vs. unseen opponents), falling back to the
    plain ``win_rate`` when held-out evaluation is disabled -- so behavior is unchanged
    when ``--holdout-team-fraction`` is 0.0.
    """

    generalization = evaluation.get("generalization")
    if generalization:
        return float(generalization["holdout_win_rate"])
    return float(evaluation["win_rate"])


def _print_generalization(evaluation: dict[str, object]) -> None:
    generalization = evaluation.get("generalization")
    if not generalization:
        return
    print(
        "  generalization: "
        f"seen={generalization['seen_win_rate']:.3f} ({generalization['seen_games']} games) "
        f"holdout={generalization['holdout_win_rate']:.3f} ({generalization['holdout_games']} games) "
        f"gap={generalization['generalization_gap']:.3f}"
    )
    # Archetype-pool mode only: `holdout_by_archetype` is absent in the flat-holdout
    # path, so this is a no-op there.
    holdout_by_archetype = generalization.get("holdout_by_archetype")
    if holdout_by_archetype:
        breakdown = ", ".join(
            f"{archetype}={stats['win_rate']:.2f} ({stats['games']}g)"
            for archetype, stats in sorted(holdout_by_archetype.items())
        )
        print(f"  holdout by archetype: {breakdown}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iteration", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="frozen deterministic games vs the heuristic after training",
    )
    parser.add_argument(
        "--eval-every-games",
        type=int,
        default=0,
        help="also run the frozen evaluation whenever this many training games are reached",
    )
    parser.add_argument(
        "--eval-jobs",
        type=int,
        default=0,
        help="parallel evaluation workers; defaults to --jobs",
    )
    parser.add_argument("--team", type=Path, default=TEAMS_DIR / "meta1.packed.txt")
    parser.add_argument(
        "--opponent-team-pool",
        type=Path,
        default=DEFAULT_TEAM_POOL_DIR,
        help="directory of varied packed teams used by training opponents",
    )
    parser.add_argument(
        "--dev-opponent-team",
        type=Path,
        default=DEFAULT_DEV_TEAM,
        help="additional varied opponent team",
    )
    parser.add_argument(
        "--mirror-team-fraction",
        type=float,
        default=0.25,
        help="worker fraction whose opponent uses the learner's team during training",
    )
    parser.add_argument(
        "--eval-mirror-team-fraction",
        type=float,
        default=0.50,
        help="worker fraction using the learner's team in frozen evaluation",
    )
    parser.add_argument(
        "--holdout-team-fraction",
        type=float,
        default=0.0,
        help=(
            "fraction of the diverse opponent pool held out of training and evaluated "
            "separately to measure generalization"
        ),
    )
    parser.add_argument(
        "--generalization-eval-games",
        type=int,
        default=0,
        help=(
            "games per arm (seen and held-out) for the generalization check; "
            "0 derives a cheap budget from --eval-games"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--teacher-anchor-weight",
        type=float,
        default=0.05,
        help="strength of the search-policy guardrail during PPO updates; 0 disables it",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=PpoConfig().entropy_weight,
        help=(
            "exploration bonus during PPO updates; lower values make the policy commit "
            "to its preferred action instead of staying diffuse"
        ),
    )
    parser.add_argument(
        "--teacher-anchor-final-weight",
        type=float,
        default=None,
        help=(
            "if set, linearly anneal the teacher-anchor weight from --teacher-anchor-"
            "weight (start) to this value (end) across the run's iterations, applied "
            "to both that iteration's game collection (so the player's >0 gate matches) "
            "and its ppo_update. Default (unset) keeps the weight constant at "
            "--teacher-anchor-weight, current behavior."
        ),
    )
    parser.add_argument(
        "--reward-shaping-coef",
        type=float,
        default=0.0,
        help=(
            "potential-based reward-shaping coefficient added to the sparse terminal "
            "outcome before GAE (see vgc.rl.rewards.board_potential); 0.0 (default) "
            "disables shaping and leaves rewards/advantages byte-for-byte unchanged"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-games",
        type=int,
        default=0,
        help="teacher-play games used to warm up the joint-action policy before PPO",
    )
    parser.add_argument("--bootstrap-epochs", type=int, default=30)
    parser.add_argument("--bootstrap-lr", type=float, default=1e-3)
    parser.add_argument("--bootstrap-val-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-min-val-accuracy", type=float, default=0.40)
    parser.add_argument("--bootstrap-min-improvement", type=float, default=0.10)
    parser.add_argument("--bootstrap-min-teacher-probability", type=float, default=0.20)
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
    parser.add_argument(
        "--archetype-pool",
        type=Path,
        default=None,
        help=(
            "manifest.json from tools/build_archetype_pool.py; opts into archetype-pool "
            "mode where BOTH the learner and opponent teams are drawn per-worker from "
            "the archetype-tagged pool (instead of --team being the sole fixed learner "
            "team) and held-out teams are split WITHIN each archetype instead of flat "
            "over --opponent-team-pool. Default (unset) leaves all existing behavior "
            "byte-for-byte unchanged."
        ),
    )
    parser.add_argument(
        "--holdout-teams-per-archetype",
        type=int,
        default=DEFAULT_HOLDOUT_TEAMS_PER_ARCHETYPE,
        help="teams held out of training per archetype in --archetype-pool mode",
    )
    parser.add_argument(
        "--meta-features",
        action="store_true",
        help=(
            "add the deployment-available meta-context branch (opponent/our archetype "
            "+ set-prior reveal scalars, see vgc.rl.encoding.encode_meta_context) to "
            "the network. Default off leaves the architecture byte-for-byte unchanged. "
            "A checkpoint's use_meta_features must match this flag on --resume."
        ),
    )
    return parser.parse_args()


def teacher_anchor_weight_for_iteration(
    global_iteration: int,
    *,
    total_iterations: int,
    start: float,
    final: float | None,
) -> float:
    """Teacher-anchor weight for one training iteration under linear annealing.

    `global_iteration` is the absolute, resume-aware iteration index (1-based:
    `last_iteration + offset`, never just this invocation's `offset`), and
    `total_iterations` is the absolute iteration index the FULL intended run reaches
    (`last_iteration + args.iterations` -- last_iteration is 0 on a fresh run). This is
    the resume-scheduling choice made here: each invocation's `--iterations` is treated
    as "how many more iterations to run," and the anneal's total span is whatever that
    adds up to including iterations already completed by prior --resume invocations, so
    resuming continues the same start->final ramp instead of restarting it from
    `start`. `start` should be `ppo_config.teacher_anchor_weight` (already resolved to
    either `--teacher-anchor-weight` on a fresh run or the checkpoint's saved value on
    `--resume`) so that `final=None` reproduces the exact pre-annealing constant-weight
    behavior in both cases, byte-for-byte. `final=None` means annealing is off: always
    return `start`. A single-iteration span (`total_iterations <= 1`, e.g. a fresh run
    with `--iterations 1`) has nothing to interpolate over and always uses `start`.
    """

    if final is None or total_iterations <= 1:
        return start
    progress = (global_iteration - 1) / (total_iterations - 1)
    progress = max(0.0, min(1.0, progress))
    return start + (final - start) * progress


def main() -> int:
    args = parse_args()
    if args.iterations < 0 or args.games_per_iteration <= 0 or args.jobs <= 0:
        raise SystemExit("iterations must be nonnegative; games-per-iteration/jobs positive")
    if args.eval_games < 0 or args.eval_jobs < 0:
        raise SystemExit("eval-games and eval-jobs must be nonnegative")
    if args.iterations == 0 and args.eval_games == 0:
        raise SystemExit("eval-games must be nonnegative and some training/evaluation is required")
    if args.eval_every_games < 0:
        raise SystemExit("eval-every-games must be nonnegative")
    if args.eval_every_games and not args.eval_games:
        raise SystemExit("eval-every-games requires positive eval-games")
    if args.bootstrap_games < 0 or args.bootstrap_epochs <= 0 or args.bootstrap_lr <= 0:
        raise SystemExit(
            "bootstrap-games must be nonnegative; bootstrap-epochs/lr must be positive"
        )
    if not 0.0 < args.bootstrap_val_fraction < 1.0:
        raise SystemExit("bootstrap-val-fraction must be between 0 and 1")
    if not 0.0 <= args.bootstrap_min_val_accuracy <= 1.0:
        raise SystemExit("bootstrap-min-val-accuracy must be between 0 and 1")
    if not 0.0 <= args.bootstrap_min_improvement <= 1.0:
        raise SystemExit("bootstrap-min-improvement must be between 0 and 1")
    if not 0.0 <= args.bootstrap_min_teacher_probability <= 1.0:
        raise SystemExit("bootstrap-min-teacher-probability must be between 0 and 1")
    if args.resume is not None and args.bootstrap_games:
        raise SystemExit("bootstrap-games cannot be combined with --resume")
    if args.snapshot_pool_size <= 0:
        raise SystemExit("snapshot-pool-size must be positive")
    if not 0.0 <= args.heuristic_opponent_fraction <= 1.0:
        raise SystemExit("heuristic-opponent-fraction must be between 0 and 1")
    if not 0.0 <= args.mirror_team_fraction <= 1.0:
        raise SystemExit("mirror-team-fraction must be between 0 and 1")
    if not 0.0 <= args.eval_mirror_team_fraction <= 1.0:
        raise SystemExit("eval-mirror-team-fraction must be between 0 and 1")
    if not 0.0 <= args.holdout_team_fraction < 1.0:
        raise SystemExit("holdout-team-fraction must be in [0.0, 1.0)")
    if args.generalization_eval_games < 0:
        raise SystemExit("generalization-eval-games must be nonnegative")
    if args.teacher_anchor_weight < 0.0:
        raise SystemExit("teacher-anchor-weight must be nonnegative")
    if args.entropy_weight < 0.0:
        raise SystemExit("entropy-weight must be nonnegative")
    if args.teacher_anchor_final_weight is not None and args.teacher_anchor_final_weight < 0.0:
        raise SystemExit("teacher-anchor-final-weight must be nonnegative")
    if args.reward_shaping_coef < 0.0:
        raise SystemExit("reward-shaping-coef must be nonnegative")
    if args.archetype_pool is not None and args.holdout_teams_per_archetype < 0:
        raise SystemExit("holdout-teams-per-archetype must be nonnegative")
    if not args.team.exists():
        raise SystemExit(f"team does not exist: {args.team}")
    torch.manual_seed(args.seed)

    model = CandidatePolicyValueNet(use_meta_features=args.meta_features).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    last_iteration = 0
    games_seen = 0
    ppo_config = PpoConfig(
        teacher_anchor_weight=args.teacher_anchor_weight,
        reward_shaping_coef=args.reward_shaping_coef,
        entropy_weight=args.entropy_weight,
    )
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
    print(
        f"teacher anchor weight: {ppo_config.teacher_anchor_weight}"
        + (
            f" -> {args.teacher_anchor_final_weight} (annealed over the run)"
            if args.teacher_anchor_final_weight is not None
            else " (constant)"
        )
    )
    print(f"reward shaping coef: {ppo_config.reward_shaping_coef}")
    team = args.team.read_text().strip()
    eval_jobs = args.eval_jobs or args.jobs
    diverse_teams = load_diverse_opponent_teams(
        args.opponent_team_pool,
        args.dev_opponent_team,
    )
    print(
        f"opponent teams: {len(diverse_teams)} varied + "
        f"{args.mirror_team_fraction:.0%} mirror workers"
    )
    train_teams, holdout_teams = split_holdout_teams(
        diverse_teams,
        holdout_fraction=args.holdout_team_fraction,
        seed=args.seed,
    )
    print(f"held-out split: {len(train_teams)} train / {len(holdout_teams)} held-out teams")

    # Archetype-pool mode (opt-in via --archetype-pool). Left empty/unused otherwise, so
    # the fixed-learner path above is completely untouched when the flag is absent.
    archetype_pool_teams: list[PoolTeam] = []
    archetype_train_teams: list[PoolTeam] = []
    archetype_holdout_teams: list[PoolTeam] = []
    if args.archetype_pool is not None:
        archetype_pool_teams = load_archetype_pool(args.archetype_pool)
        archetype_train_teams, archetype_holdout_teams = split_holdout_by_archetype(
            archetype_pool_teams,
            holdout_per_archetype=args.holdout_teams_per_archetype,
            seed=args.seed,
        )
        pool_archetype_counts = Counter(team.archetype for team in archetype_pool_teams)
        train_archetype_counts = Counter(team.archetype for team in archetype_train_teams)
        holdout_archetype_counts = Counter(team.archetype for team in archetype_holdout_teams)
        print(
            f"archetype pool: {len(archetype_pool_teams)} teams across "
            f"{len(pool_archetype_counts)} archetypes; learner-team randomization: on"
        )
        for archetype in sorted(pool_archetype_counts):
            print(
                f"  {archetype}: {train_archetype_counts.get(archetype, 0)} train / "
                f"{holdout_archetype_counts.get(archetype, 0)} held-out "
                f"(of {pool_archetype_counts[archetype]})"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    bootstrap_path = args.out_dir / "bootstrap.json"
    evaluation_path = args.out_dir / "evaluation.json"
    evaluation_history_path = args.out_dir / "evaluation_history.jsonl"
    best_checkpoint_path = args.out_dir / "best.pt"
    best_evaluation_path = args.out_dir / "best_evaluation.json"
    pool_dir = args.out_dir / "opponent_pool"
    best_eval_win_rate = -1.0
    if (
        args.resume is not None
        and evaluation_history_path.exists()
        and best_checkpoint_path.exists()
    ):
        for line in evaluation_history_path.read_text().splitlines():
            if line.strip():
                best_eval_win_rate = max(
                    best_eval_win_rate,
                    _tracked_win_rate(json.loads(line)),
                )

    def record_evaluation(evaluation: dict[str, object], *, iteration: int) -> bool:
        nonlocal best_eval_win_rate
        evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        with evaluation_history_path.open("a") as evaluation_history:
            evaluation_history.write(json.dumps(evaluation, sort_keys=True) + "\n")
        win_rate = _tracked_win_rate(evaluation)
        improved = win_rate > best_eval_win_rate
        if improved:
            best_eval_win_rate = win_rate
            best_evaluation_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
            save_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                iteration=iteration,
                games_seen=int(evaluation["training_games"]),
                ppo_config=ppo_config,
            )
        return improved

    def run_evaluation() -> dict[str, object]:
        """Frozen + generalization evaluation, branching on archetype-pool mode.

        Kept as one closure so the two call sites (periodic and final) below can't
        drift apart on which arguments they pass.
        """
        if args.archetype_pool is not None:
            return asyncio.run(
                evaluate_with_generalization_by_archetype(
                    model,
                    eval_games=args.eval_games,
                    eval_jobs=eval_jobs,
                    train_pool_teams=archetype_train_teams,
                    holdout_pool_teams=archetype_holdout_teams,
                    eval_mirror_fraction=args.eval_mirror_team_fraction,
                    generalization_games=args.generalization_eval_games,
                    seed=args.seed,
                    device=args.device,
                )
            )
        return asyncio.run(
            evaluate_with_generalization(
                model,
                eval_games=args.eval_games,
                eval_jobs=eval_jobs,
                learner_team=team,
                diverse_teams=diverse_teams,
                train_teams=train_teams,
                holdout_teams=holdout_teams,
                eval_mirror_fraction=args.eval_mirror_team_fraction,
                generalization_games=args.generalization_eval_games,
                seed=args.seed,
                device=args.device,
            )
        )

    server_process = None
    if not args.skip_server_start:
        print("starting local Showdown server...")
        server_process = start_local_server()
        print("server ready")

    try:
        if args.bootstrap_games:
            bootstrap_started = time.time()
            if args.archetype_pool is not None:
                samples, collection = asyncio.run(
                    collect_teacher_samples_from_pool(
                        games=args.bootstrap_games,
                        jobs=args.jobs,
                        # Bootstrap is a training phase, so it must never touch held-out
                        # teams (same rationale as the non-pool path below).
                        pool_teams=archetype_train_teams,
                        mirror_fraction=args.mirror_team_fraction,
                        seed=args.seed,
                    )
                )
            else:
                samples, collection = asyncio.run(
                    collect_teacher_samples(
                        games=args.bootstrap_games,
                        jobs=args.jobs,
                        learner_team=team,
                        # Bootstrap is a training phase, so it must never touch held-out
                        # teams -- otherwise the warm-started encoder has already adapted to
                        # states arising from "unseen" opponents and the generalization gap
                        # is contaminated. With held-out disabled, train_teams == diverse_teams.
                        diverse_teams=train_teams,
                        mirror_fraction=args.mirror_team_fraction,
                        seed=args.seed,
                    )
                )
            train_samples, val_samples = split_samples_by_battle(
                samples,
                val_fraction=args.bootstrap_val_fraction,
                seed=args.seed,
            )
            distill_config = DistillationConfig(
                epochs=args.bootstrap_epochs,
                val_fraction=args.bootstrap_val_fraction,
                seed=args.seed,
            )
            before = evaluate_agreement(
                model,
                val_samples,
                batch_size=distill_config.batch_size,
                device=args.device,
            )
            training = distill_policy(
                model,
                torch.optim.Adam(model.parameters(), lr=args.bootstrap_lr),
                train_samples,
                distill_config,
                device=args.device,
            )
            train_after = evaluate_agreement(
                model,
                train_samples,
                batch_size=distill_config.batch_size,
                device=args.device,
            )
            val_after = evaluate_agreement(
                model,
                val_samples,
                batch_size=distill_config.batch_size,
                device=args.device,
            )
            improvement = val_after["accuracy"] - before["accuracy"]
            passed = (
                val_after["accuracy"] >= args.bootstrap_min_val_accuracy
                and improvement >= args.bootstrap_min_improvement
                and val_after["teacher_probability"] >= args.bootstrap_min_teacher_probability
            )
            bootstrap_result = {
                "collection": collection,
                "train_battles": len({sample.battle_id for sample in train_samples}),
                "val_battles": len({sample.battle_id for sample in val_samples}),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "before": before,
                "training": training,
                "train_after": train_after,
                "val_after": val_after,
                "val_accuracy_improvement": improvement,
                "min_val_accuracy": args.bootstrap_min_val_accuracy,
                "min_improvement": args.bootstrap_min_improvement,
                "min_teacher_probability": args.bootstrap_min_teacher_probability,
                "passed": passed,
                "elapsed_seconds": time.time() - bootstrap_started,
            }
            bootstrap_path.write_text(json.dumps(bootstrap_result, indent=2, sort_keys=True) + "\n")
            print(
                "teacher bootstrap: "
                f"games={collection['games']} samples={len(samples)} "
                f"val_agreement={before['accuracy']:.3f}->{val_after['accuracy']:.3f} "
                f"teacher_prob={val_after['teacher_probability']:.3f} "
                f"passed={passed}"
            )
            if not passed:
                raise RuntimeError(
                    f"teacher bootstrap gate failed; PPO was not started. See {bootstrap_path}"
                )

        next_eval_at = (
            ((games_seen // args.eval_every_games) + 1) * args.eval_every_games
            if args.eval_every_games
            else None
        )
        last_evaluated_games: int | None = None
        total_planned_iterations = last_iteration + args.iterations
        for offset in range(1, args.iterations + 1):
            iteration = last_iteration + offset
            current_teacher_anchor_weight = teacher_anchor_weight_for_iteration(
                iteration,
                total_iterations=total_planned_iterations,
                start=ppo_config.teacher_anchor_weight,
                final=args.teacher_anchor_final_weight,
            )
            iteration_ppo_config = replace(
                ppo_config, teacher_anchor_weight=current_teacher_anchor_weight
            )
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
                collect_games_from_pool(
                    model,
                    buffer,
                    games=args.games_per_iteration,
                    jobs=args.jobs,
                    pool_teams=archetype_train_teams,
                    mirror_fraction=args.mirror_team_fraction,
                    device=args.device,
                    ppo_config=iteration_ppo_config,
                    snapshots=snapshots,
                    heuristic_fraction=args.heuristic_opponent_fraction,
                    seed=args.seed + iteration * 10_000,
                )
                if args.archetype_pool is not None
                else collect_games(
                    model,
                    buffer,
                    games=args.games_per_iteration,
                    jobs=args.jobs,
                    learner_team=team,
                    diverse_teams=train_teams,
                    mirror_fraction=args.mirror_team_fraction,
                    device=args.device,
                    ppo_config=iteration_ppo_config,
                    snapshots=snapshots,
                    heuristic_fraction=args.heuristic_opponent_fraction,
                    seed=args.seed + iteration * 10_000,
                )
            )
            if not buffer.steps:
                raise RuntimeError("no PPO decisions were recorded from completed games")
            update_metrics = ppo_update(
                model, optimizer, buffer, iteration_ppo_config, device=args.device
            )
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
                f"loss={update_metrics['loss']:.4f} entropy={update_metrics['entropy']:.4f} "
                f"teacher_anchor_weight={current_teacher_anchor_weight:.4f}"
            )

            if next_eval_at is not None and games_seen >= next_eval_at:
                evaluation = run_evaluation()
                evaluation["training_games"] = games_seen
                improved = record_evaluation(evaluation, iteration=iteration)
                last_evaluated_games = games_seen
                print(
                    f"frozen evaluation at {games_seen} games: "
                    f"wins={evaluation['wins']}/{evaluation['games']} "
                    f"win_rate={evaluation['win_rate']:.3f} "
                    f"errors={evaluation['worker_errors']} best={improved}"
                )
                _print_generalization(evaluation)
                while next_eval_at <= games_seen:
                    next_eval_at += args.eval_every_games

        if args.eval_games and last_evaluated_games != games_seen:
            evaluation = run_evaluation()
            evaluation["training_games"] = games_seen
            improved = record_evaluation(
                evaluation,
                iteration=last_iteration + args.iterations,
            )
            print(
                "frozen evaluation: "
                f"wins={evaluation['wins']}/{evaluation['games']} "
                f"win_rate={evaluation['win_rate']:.3f} "
                f"errors={evaluation['worker_errors']} best={improved}"
            )
            _print_generalization(evaluation)
    finally:
        if server_process is not None:
            server_process.kill()
            server_process.wait(timeout=10)

    print(f"checkpoint: {args.out_dir / 'latest.pt'}")
    print(f"metrics: {metrics_path}")
    if args.bootstrap_games:
        print(f"bootstrap: {bootstrap_path}")
    if args.eval_games:
        print(f"evaluation: {evaluation_path}")
        print(f"evaluation history: {evaluation_history_path}")
        print(f"best checkpoint: {best_checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
