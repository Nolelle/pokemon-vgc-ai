#!/usr/bin/env python
"""Record saved decision positions from seeded offline games.

Plays our team(s) against opponent teams from a pool manifest on the direct (no-server)
environment with the shipped engine on both sides, records OUR side's player-view message
stream (``vgc.battle_state_replay`` decision-replay bundle), samples real decisions
(no team preview, no forced switch, no single-option turn), and writes::

    runs/positions/<run>/positions.jsonl     one row per position
    runs/positions/<run>/bundles/<id>.json   the bundle CUT at that decision (no future)
    runs/positions/<run>/meta.json           arguments, commit, timings

Each row carries: position id, game seed, our team, opponent team id, turn, the engine's
top-K ranked orders (canonical ``describe_order`` strings + wire form + score) and the
order it chose, and a team-disjoint ``split`` ("tune" / "test") keyed on the OPPONENT
TEAM so a grader never tunes and tests on the same opponent team.

Each position is verified by rebuilding it from the saved bundle and comparing the
state / legal-action fingerprints (``--no-verify`` to skip). Grade with
``offline/grade_positions.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.baselines import BASELINES  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.positions import (  # noqa: E402
    POSITION_SCHEMA,
    play_recorded_game,
    sampleable_decisions,
    split_membership,
    team_split,
    trimmed_bundle,
    verify_position,
)
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402

MAIN_CHECKOUT = Path("/Users/edmundyu/code/projects/pokemon-vgc-ai")
DEFAULT_OUR_TEAMS = [
    REPO_ROOT / "teams" / "owner" / "psyspam_sand.packed.txt",
    REPO_ROOT / "teams" / "owner" / "salamence_tw.packed.txt",
]
MANIFEST_REL = Path("data/selfplay/mc_sheet_pool_v2/holdout_manifest.json")


def default_manifest() -> Path:
    local = REPO_ROOT / MANIFEST_REL
    return local if local.exists() else MAIN_CHECKOUT / MANIFEST_REL


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_opponents(manifest: Path) -> list[dict]:
    entries = json.loads(manifest.read_text())
    teams = []
    for raw in entries:
        path = manifest.parent / str(raw["file"])
        if path.is_file():
            teams.append({"file": str(raw["file"]), "team": path.read_text().strip()})
    if not teams:
        raise SystemExit(f"no readable team files next to {manifest}")
    return teams


def make_opponent(name: str, team: str):
    if name not in BASELINES:
        raise SystemExit(f"unknown opponent {name!r}; one of {sorted(BASELINES)}")
    if name.startswith("vgc"):
        return make_direct_agent(name, team, config=PolicyConfig(format_id=FORMAT_ID))
    return make_direct_agent(name, team)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=None, help="opponent pool manifest")
    ap.add_argument("--split-file", type=Path, default=None, help="split.json (default: beside manifest)")
    ap.add_argument(
        "--require-split",
        action="store_true",
        help="refuse to run if split.json is missing or does not name every opponent team "
        "(otherwise the fallback hashes the team's six-species set)",
    )
    ap.add_argument("--our-teams", type=Path, nargs="+", default=DEFAULT_OUR_TEAMS)
    ap.add_argument("--positions", type=int, default=40, help="stop after this many positions")
    ap.add_argument("--max-games", type=int, default=60)
    ap.add_argument("--per-game", type=int, default=3, help="positions sampled per game")
    ap.add_argument("--min-turn", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=6, help="engine candidates recorded per position")
    ap.add_argument("--opponent", default="vgc", help="baseline playing the opponent team")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--run", default=None, help="run name (default: timestamp)")
    ap.add_argument("--out-dir", type=Path, default=RUNS_DIR / "positions")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    manifest = args.manifest or default_manifest()
    split_file = args.split_file or manifest.parent / "split.json"
    opponents = load_opponents(manifest)
    unnamed = [o["file"] for o in opponents if split_membership(o["file"], split_file) is None]
    if unnamed and args.require_split:
        raise SystemExit(
            f"--require-split: {split_file} is missing or does not name {len(unnamed)} "
            f"opponent team(s), e.g. {unnamed[:3]}"
        )
    if unnamed:
        print(
            f"warning: {len(unnamed)}/{len(opponents)} opponent teams not in {split_file}; "
            "splitting them by a hash of their six-species set",
            flush=True,
        )
    our_teams = [(p.name.removesuffix(".packed.txt"), p.read_text().strip()) for p in args.our_teams]
    run = args.run or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_dir / run
    (run_dir / "bundles").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    order = list(range(len(opponents)))
    rng.shuffle(order)

    rows: list[dict] = []
    stats = {"games": 0, "verify_failed": 0, "game_errors": 0, "game_seconds": 0.0}
    started = time.perf_counter()
    out_path = run_dir / "positions.jsonl"
    with SimWorker() as worker, out_path.open("w") as out:
        for game_index in range(args.max_games):
            if len(rows) >= args.positions:
                break
            opp = opponents[order[game_index % len(order)]]
            our_name, our_team = our_teams[game_index % len(our_teams)]
            seat = "p1" if (game_index // len(our_teams)) % 2 == 0 else "p2"
            game_seed = [rng.randrange(1, 2**31) for _ in range(4)]
            # A room-tag-shaped id: the replay parser creates the battle from it.
            game_id = f"battle-{FORMAT_ID}-{rng.randrange(10**8, 10**9)}"
            label = f"{run}-g{game_index:03d}"
            t0 = time.perf_counter()
            try:
                game = play_recorded_game(
                    worker,
                    game_id,
                    our_team=our_team,
                    opp_team=opp["team"],
                    opp_agent=make_opponent(args.opponent, opp["team"]),
                    seed=game_seed,
                    seat=seat,
                    top_k=args.top_k,
                )
            except Exception as exc:  # noqa: BLE001 - one bad game must not end the run
                stats["game_errors"] += 1
                print(f"[{label}] game failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            stats["games"] += 1
            stats["game_seconds"] += time.perf_counter() - t0
            candidates = [
                i
                for i in sampleable_decisions(game)
                if int(game.bundle["decisions"][i]["turn"]) >= args.min_turn
            ]
            rng.shuffle(candidates)
            kept = 0
            for index in candidates:
                if kept >= args.per_game or len(rows) >= args.positions:
                    break
                position_id = f"{label}-d{index:03d}"
                cut = trimmed_bundle(game.bundle, index)
                if not args.no_verify:
                    check = verify_position(cut, index)
                    if not check.ready:
                        stats["verify_failed"] += 1
                        print(f"[{position_id}] verify failed: {check.mismatches[:2]}", flush=True)
                        continue
                bundle_rel = f"bundles/{position_id}.json"
                (run_dir / bundle_rel).write_text(json.dumps(cut))
                decision = game.bundle["decisions"][index]
                ranking = game.rankings[index]
                row = {
                    "schema": POSITION_SCHEMA,
                    "position_id": position_id,
                    "run": run,
                    "game_id": game_id,
                    "game_seed": game_seed,
                    "seat": seat,
                    "our_team": our_name,
                    "our_team_sha256": cut["own_team_sha256"],
                    "opp_team_id": opp["file"],
                    "split": team_split(opp["file"], split_file, packed_team=opp["team"]),
                    "opponent_agent": args.opponent,
                    "turn": int(decision["turn"]),
                    "decision_index": index,
                    "n_legal": len(decision["legal_actions"]),
                    "state_sha256": decision["state_sha256"],
                    "bundle_path": bundle_rel,
                    "engine_chosen": decision["chosen_order"],
                    "engine_chosen_wire": decision["chosen_order_wire"],
                    "engine_top_k": ranking,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                rows.append(row)
                kept += 1
            print(
                f"[{label}] {our_name} ({seat}) vs {opp['file']}: "
                f"{len(game.bundle['decisions'])} decisions, {kept} positions kept "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    by_split: dict[str, int] = {}
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    meta = {
        "schema": POSITION_SCHEMA,
        "run": run,
        "created_at": datetime.now(UTC).isoformat(),
        "commit": git_commit(),
        "format": FORMAT_ID,
        "manifest": str(manifest),
        "split_file": str(split_file),
        "opponents_without_split_membership": len(unnamed),
        "our_teams": [name for name, _ in our_teams],
        "positions": len(rows),
        "positions_by_split": by_split,
        "distinct_opponent_teams": len({r["opp_team_id"] for r in rows}),
        "elapsed_seconds": round(elapsed, 1),
        **stats,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in (
        "positions", "positions_by_split", "distinct_opponent_teams", "games",
        "verify_failed", "game_errors", "elapsed_seconds")}), flush=True)
    print(f"wrote {out_path}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
