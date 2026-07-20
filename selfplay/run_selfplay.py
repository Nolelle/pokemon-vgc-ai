#!/usr/bin/env python
"""Self-play data generation: grows the BC value head's training data ~10x beyond the
downloaded human replay corpus (52,879 decision records, 2939 replays) with FREE local
bot-vs-bot games, and is also the thing that decides whether the encoder-v4 preview-
context features (`vgc.bc.encoding`'s `bc-encoding-v4`) actually pay off at a larger
data scale than the ~53k-record corpus where they measured flat-to-negative.

Starts the local Showdown server itself (`node pokemon-showdown start --no-security`,
same subprocess pattern `tests/integration/test_local_battles.py`'s `local_server`
fixture uses), then plays `--n-games` sequential (v1: `--jobs` > 1 is not yet
implemented -- see `parse_args`) 1-battle-at-a-time games between two fresh
`vgc.bc.selfplay.RecordingVgcPlayer`s, each built from an independently-sampled
`(team, PolicyConfig variant)` pair (see `build_game_specs`/`build_config_variants`).
Every completed battle's buffered decision records are appended (crash-safe: written
only once that battle finishes, see `RecordingVgcPlayer`'s own docstring) to
`--out` (default `data/selfplay/records.jsonl`, gitignored).

## Diversity

- **Configs** (`build_config_variants`): `"myopic"` (the diagnostic one-turn evaluator),
  `"search"` (the real shipped default -- `use_two_ply_search=True`, matching
  `ladder/run_ladder.py`'s plain `--search`/default session), and a handful of
  weight-jittered variants of `"search"` (+/-20% on `JITTER_FIELDS`, deterministic per
  `--seed`). EVERY variant forces `accept_open_team_sheet=False` -- see
  `vgc.bc.selfplay`'s module docstring for why self-play must NOT mutually accept OTS.
- **Teams** (`build_game_specs`): `teams/meta1.packed.txt` (our actual ladder team) is
  guaranteed to appear on at least one side of >= `--meta1-fraction` (default 0.5) of all
  games -- the value net needs to be sharpest on OUR positions, not diluted across
  arbitrary teams. The rest of the games (and the non-meta1 side of a meta1 game) draw
  from `teams/dev.packed.txt` plus the generated corpus-derived opponent pool
  (`data/selfplay/pool/*.packed.txt`, see `tools/build_selfplay_pool.py` -- run that
  first if the pool directory is empty; this script warns and falls back to
  dev-team-only diversity if so, rather than failing outright).

Usage:
    # One-time (or whenever you want a fresh pool): validate + build the opponent pool.
    .venv/bin/python tools/build_selfplay_pool.py

    # Start the local server yourself first (this script does NOT start it for you if
    # one is already listening on localhost:8000 -- see start_local_server's docstring):
    cd /Users/edmundyu/code/projects/pokemon-showdown && node pokemon-showdown start --no-security &

    .venv/bin/python selfplay/run_selfplay.py --n-games 1500 --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402

from vgc.bc.selfplay import RecordingVgcPlayer  # noqa: E402
from vgc.config import FORMAT_ID, SHOWDOWN_REPO, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.node import find_node, node_environment  # noqa: E402

_LOGGER = logging.getLogger(__name__)

DEFAULT_OUT_PATH = REPO_ROOT / "data" / "selfplay" / "records.jsonl"
DEFAULT_POOL_DIR = REPO_ROOT / "data" / "selfplay" / "pool"
DEFAULT_META1_PATH = TEAMS_DIR / "meta1.packed.txt"
DEFAULT_DEV_PATH = TEAMS_DIR / "dev.packed.txt"
SERVER_READY_TIMEOUT_SECONDS = 30
GAME_TIMEOUT_SECONDS = 120.0
SUMMARY_EVERY = 50

# Weight-jitter targets: key PolicyConfig weights with real documented behavioral impact
# (see runs/experiments.jsonl's campaign postmortems -- e.g. search_worst_case_weight's
# 0.25->0.4 tuning, speed_control_flip_weight's Trick Room fix), jittered +/-20% per
# variant via a seeded RNG on top of the "search" base config.
JITTER_FIELDS: tuple[str, ...] = (
    "search_worst_case_weight",
    "win_con_preservation_weight",
    "expected_death_cost_weight",
    "speed_control_flip_weight",
    "damage_percent_weight",
)
JITTER_RANGE = 0.2  # +/- 20%
N_JITTER_VARIANTS = 3


def build_config_variants(seed: int = 0) -> dict[str, PolicyConfig]:
    """Named `PolicyConfig` variants for self-play decision-policy diversity -- see
    module docstring's "Diversity" section. Deterministic for a given `seed` (only the
    jittered variants depend on it; `"myopic"`/`"search"` are always identical).
    """
    base_myopic = replace(PolicyConfig(), use_two_ply_search=False, accept_open_team_sheet=False)
    base_search = replace(PolicyConfig(), use_two_ply_search=True, accept_open_team_sheet=False)

    variants: dict[str, PolicyConfig] = {"myopic": base_myopic, "search": base_search}

    rng = random.Random(seed)
    for index in range(N_JITTER_VARIANTS):
        overrides = {}
        for field_name in JITTER_FIELDS:
            current = getattr(base_search, field_name)
            multiplier = 1.0 + rng.uniform(-JITTER_RANGE, JITTER_RANGE)
            overrides[field_name] = current * multiplier
        variants[f"jitter-{index}"] = replace(base_search, **overrides)
    return variants


@dataclass(frozen=True)
class GameSpec:
    index: int
    p1_team_path: Path
    p1_config_name: str
    p2_team_path: Path
    p2_config_name: str


def discover_team_pool(pool_dir: Path, dev_team_path: Path) -> list[Path]:
    """Non-meta1 team choices: `dev_team_path` plus every `*.packed.txt` under
    `pool_dir` (sorted for determinism). `pool_dir` missing/empty just means fewer
    choices (dev-only) -- callers should warn, not fail, since the pool is a
    `tools/build_selfplay_pool.py` prerequisite that's easy to forget to (re)run.
    """
    pool_paths = sorted(pool_dir.glob("*.packed.txt")) if pool_dir.exists() else []
    return [dev_team_path, *pool_paths]


def build_game_specs(
    n_games: int,
    *,
    seed: int,
    meta1_team_path: Path,
    other_team_paths: list[Path],
    config_names: list[str],
    meta1_min_fraction: float = 0.5,
) -> list[GameSpec]:
    """`n_games` deterministic `GameSpec`s. `meta1_team_path` is placed on (a randomly
    chosen) one side of exactly `ceil(n_games * meta1_min_fraction)` games (a hard
    guarantee by construction, not a probabilistic target) -- the other side of a
    meta1 game, and BOTH sides of a non-meta1 game, are drawn from `other_team_paths`
    (which may itself legally include meta1 mirror matches if the caller passes it in
    there too -- this function doesn't do that itself, keeping the >= fraction exact and
    easy to verify). Each side's `PolicyConfig` variant is sampled independently from
    `config_names`. Returns `[]` for `n_games <= 0`.
    """
    if n_games <= 0:
        return []
    if not other_team_paths:
        raise ValueError("other_team_paths must be non-empty")
    if not config_names:
        raise ValueError("config_names must be non-empty")

    rng = random.Random(seed)
    n_meta1_games = math.ceil(n_games * meta1_min_fraction)
    meta1_flags = [True] * n_meta1_games + [False] * (n_games - n_meta1_games)
    rng.shuffle(meta1_flags)

    specs: list[GameSpec] = []
    for index, has_meta1 in enumerate(meta1_flags):
        if has_meta1:
            meta1_side = rng.choice(("p1", "p2"))
            other_team = rng.choice(other_team_paths)
            p1_team = meta1_team_path if meta1_side == "p1" else other_team
            p2_team = other_team if meta1_side == "p1" else meta1_team_path
        else:
            p1_team = rng.choice(other_team_paths)
            p2_team = rng.choice(other_team_paths)
        specs.append(
            GameSpec(
                index=index,
                p1_team_path=p1_team,
                p1_config_name=rng.choice(config_names),
                p2_team_path=p2_team,
                p2_config_name=rng.choice(config_names),
            )
        )
    return specs


async def _safe_stop_listening(player, timeout_seconds: float = 15.0) -> None:
    """Best-effort websocket teardown -- mirrors ladder/run_ladder.py's
    `_safe_stop_listening` (a connection that already died server-side must never block
    or abort the run's own bookkeeping).
    """
    try:
        await asyncio.wait_for(player.ps_client.stop_listening(), timeout=timeout_seconds)
    except Exception:  # noqa: BLE001 - teardown of a dead client must never propagate
        _LOGGER.debug("stop_listening failed during teardown; discarding client", exc_info=True)


async def run_one_game(
    spec: GameSpec,
    config_variants: dict[str, PolicyConfig],
    out_path: Path,
    token: str,
) -> dict[str, object]:
    """Plays exactly one game for `spec`, returns a small per-game result dict (never
    raises -- a single bad game must not abort a 1500-game run, so any exception is
    caught and reported as `"error"` in the result instead).
    """
    p1_team = spec.p1_team_path.read_text().strip()
    p2_team = spec.p2_team_path.read_text().strip()
    p1 = RecordingVgcPlayer(
        out_path=out_path,
        replay_tag=f"{spec.p1_config_name}-{spec.p1_team_path.stem}",
        config=config_variants[spec.p1_config_name],
        team=p1_team,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"sp{spec.index}a-{token}", None),
    )
    p2 = RecordingVgcPlayer(
        out_path=out_path,
        replay_tag=f"{spec.p2_config_name}-{spec.p2_team_path.stem}",
        config=config_variants[spec.p2_config_name],
        team=p2_team,
        battle_format=FORMAT_ID,
        account_configuration=AccountConfiguration(f"sp{spec.index}b-{token}", None),
    )
    error: str | None = None
    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=GAME_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - one bad game must not abort the whole run
        error = f"{type(exc).__name__}: {exc}"
        _LOGGER.warning("game %d failed", spec.index, exc_info=True)
    finally:
        await _safe_stop_listening(p1)
        await _safe_stop_listening(p2)

    p1_battle = next(iter(p1.battles.values()), None)
    p2_battle = next(iter(p2.battles.values()), None)
    return {
        "index": spec.index,
        "p1_config": spec.p1_config_name,
        "p2_config": spec.p2_config_name,
        "p1_team": spec.p1_team_path.stem,
        "p2_team": spec.p2_team_path.stem,
        "p1_won": p1_battle.won if p1_battle is not None else None,
        "p2_won": p2_battle.won if p2_battle is not None else None,
        "turns": p1_battle.turn if p1_battle is not None else None,
        "records_written": p1.records_written + p2.records_written,
        "error": error,
    }


def start_local_server() -> subprocess.Popen:
    """Starts `node pokemon-showdown start --no-security` and blocks until it reports
    ready (or raises) -- same pattern as `tests/integration/test_local_battles.py`'s
    `local_server` fixture. Caller is responsible for killing the returned process.
    """
    node = find_node()
    process = subprocess.Popen(
        [node, "pokemon-showdown", "start", "--no-security"],
        cwd=str(SHOWDOWN_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=node_environment(node),
    )
    deadline = time.monotonic() + SERVER_READY_TIMEOUT_SECONDS
    ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"showdown server exited early:\n{output}")
        line = process.stdout.readline() if process.stdout else ""
        if "listening on" in line.lower():
            ready = True
            break
    if not ready:
        process.kill()
        raise TimeoutError("showdown server did not report ready in time")
    time.sleep(1.0)
    return process


async def run_selfplay(
    n_games: int,
    *,
    seed: int,
    out_path: Path,
    pool_dir: Path,
    meta1_path: Path,
    dev_path: Path,
    meta1_fraction: float,
) -> dict[str, object]:
    config_variants = build_config_variants(seed)
    other_team_paths = discover_team_pool(pool_dir, dev_path)
    if len(other_team_paths) <= 1:
        print(
            f"WARNING: no generated pool teams found under {pool_dir} -- run "
            "tools/build_selfplay_pool.py first for real team diversity; falling back "
            "to dev-team-only for the non-meta1 side.",
            file=sys.stderr,
        )
    specs = build_game_specs(
        n_games,
        seed=seed,
        meta1_team_path=meta1_path,
        other_team_paths=other_team_paths,
        config_names=list(config_variants),
        meta1_min_fraction=meta1_fraction,
    )

    games_completed = 0
    games_failed = 0
    p1_wins = 0
    total_records = 0
    start = time.time()
    for spec in specs:
        token = f"{spec.index:05d}"
        result = await run_one_game(spec, config_variants, out_path, token)
        games_completed += 1
        total_records += result["records_written"]
        if result["error"] is not None:
            games_failed += 1
        elif result["p1_won"] is True:
            p1_wins += 1

        if games_completed % SUMMARY_EVERY == 0 or games_completed == n_games:
            elapsed = time.time() - start
            print(
                f"[{games_completed}/{n_games}] records={total_records} "
                f"failed={games_failed} p1_win_rate={p1_wins / games_completed:.3f} "
                f"elapsed={elapsed:.0f}s ({elapsed / games_completed:.2f}s/game)"
            )

    return {
        "games_requested": n_games,
        "games_completed": games_completed,
        "games_failed": games_failed,
        "records_written": total_records,
        "elapsed_seconds": time.time() - start,
        "out_path": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-games", type=int, default=1500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--meta1-team", type=Path, default=DEFAULT_META1_PATH)
    parser.add_argument("--dev-team", type=Path, default=DEFAULT_DEV_PATH)
    parser.add_argument(
        "--meta1-fraction",
        type=float,
        default=0.5,
        help="minimum fraction of games where teams/meta1.packed.txt appears on a side",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel game workers -- v1 only supports 1 (sequential); >1 is future work",
    )
    parser.add_argument(
        "--skip-server-start",
        action="store_true",
        help="assume a local server is already running instead of starting one",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jobs != 1:
        print("--jobs > 1 is not implemented yet (v1 is sequential-only)", file=sys.stderr)
        return 1
    if not args.meta1_team.exists():
        print(f"{args.meta1_team} does not exist", file=sys.stderr)
        return 1
    if not args.dev_team.exists():
        print(f"{args.dev_team} does not exist", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    server_process = None
    if not args.skip_server_start:
        print("starting local Showdown server...")
        server_process = start_local_server()
        print("server ready")

    try:
        result = asyncio.run(
            run_selfplay(
                args.n_games,
                seed=args.seed,
                out_path=args.out,
                pool_dir=args.pool_dir,
                meta1_path=args.meta1_team,
                dev_path=args.dev_team,
                meta1_fraction=args.meta1_fraction,
            )
        )
    finally:
        if server_process is not None:
            server_process.kill()
            server_process.wait(timeout=10)

    print("\nself-play summary:")
    print(f"  games completed:  {result['games_completed']}/{result['games_requested']}")
    print(f"  games failed:     {result['games_failed']}")
    print(f"  records written:  {result['records_written']}")
    print(f"  elapsed:          {result['elapsed_seconds']:.1f}s")
    print(f"  output:           {result['out_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
