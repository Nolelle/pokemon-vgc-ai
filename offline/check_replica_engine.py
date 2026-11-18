"""Compare Rust and pinned Showdown on the same teams, choices and random seeds."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np

from vgc.actions import enumerate_joint_orders
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SIDES, DirectBattle, SimWorker, choice_string
from vgc.rl.replica_engine import ReplicaRustWorker
from vgc.rl.replication_policy import encode_decision


def health(game):
    return {
        side: {key: mon.current_hp for key, mon in game.battles[side].team.items()}
        for side in SIDES
    }


def decision_state(game):
    return {
        "ended": game.ended,
        "winner": game.winner,
        "phase": game.request_state,
        "sides": game.sides_to_move(),
        "rejections": getattr(game, "hidden_trap_rejections", 0),
    }


def player_view_difference(first, second, side):
    """Compare what the transformer sees, including its legal doubles choices."""
    observations = []
    order_lists = []
    for game in (first, second):
        view = game.battles[side]
        orders = (
            None if view.teampreview else sorted(enumerate_joint_orders(view), key=choice_string)
        )
        order_lists.append([choice_string(order) for order in orders or ()])
        observations.append(encode_decision(view, orders, preview=view.teampreview))
    if order_lists[0] != order_lists[1]:
        return {
            "side": side,
            "difference": "legal choices",
            "showdown_only": sorted(set(order_lists[0]) - set(order_lists[1])),
            "rust_only": sorted(set(order_lists[1]) - set(order_lists[0])),
        }
    for field in ("pokemon", "scalars", "field", "candidates"):
        left, right = (getattr(observation, field) for observation in observations)
        if left.shape != right.shape:
            return {"side": side, "difference": field, "shapes": [left.shape, right.shape]}
        if not np.array_equal(left, right):
            index = tuple(np.argwhere(left != right)[0])
            return {
                "side": side,
                "difference": field,
                "index": list(map(int, index)),
                "showdown": left[index].item(),
                "rust": right[index].item(),
            }
    return None


def compare(official, rust, seed, teams, *, random_preview=False):
    rng = random.Random(seed)
    team_a, team_b = rng.choices(teams, k=2)
    battle_seed = [rng.randrange(65536) for _ in range(4)]
    games = []
    try:
        for worker in (official, rust):
            games.append(DirectBattle.start(worker, str(seed), team_a, team_b, seed=battle_seed))
        first, second = games
        choices = {}
        for step in range(1000):
            if decision_state(first) != decision_state(second) or health(first) != health(second):
                return {
                    "difference": "result, requested side, or health",
                    "seed": seed,
                    "step": step,
                    "last_choices": choices,
                    "showdown_state": decision_state(first),
                    "rust_state": decision_state(second),
                    "showdown_hp": health(first),
                    "rust_hp": health(second),
                }
            if first.ended:
                return None
            for side in first.sides_to_move():
                difference = player_view_difference(first, second, side)
                if difference:
                    return {"seed": seed, "step": step, "last_choices": choices, **difference}
            choices = {}
            for side in first.sides_to_move():
                if first.request_state == "teampreview":
                    pick = rng.sample("123456", 4) if random_preview else "1234"
                    choices[side] = "team " + "".join(pick)
                else:
                    choices[side] = choice_string(
                        rng.choice(enumerate_joint_orders(first.battles[side]))
                    )
            for engine, game in zip(("showdown", "rust"), games, strict=True):
                try:
                    game.step(choices)
                except (ValueError, RuntimeError, KeyError) as exc:
                    return {
                        "seed": seed,
                        "step": step + 1,
                        "difference": "step failed",
                        "engine": engine,
                        "last_choices": choices,
                        "error": str(exc),
                    }
        raise RuntimeError(f"seed {seed} exceeded 1000 steps")
    finally:
        for game in games:
            game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument(
        "--random-preview", action="store_true", help="vary the four brought and their leads"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--team-manifest",
        type=Path,
        default=Path("data/selfplay/mc_sheet_pool_v2/train_manifest.json"),
    )
    args = parser.parse_args()
    if args.games < 1:
        parser.error("games must be positive")
    manifest = args.team_manifest
    teams = [
        (manifest.parent / row["file"]).read_text().strip()
        for row in json.loads(manifest.read_text())
    ]
    differences = []
    with SimWorker() as official, ReplicaRustWorker() as rust:
        for seed in range(args.seed, args.seed + args.games):
            difference = compare(official, rust, seed, teams, random_preview=args.random_preview)
            if difference:
                differences.append(difference)
    print(
        json.dumps(
            {
                "rust_package": version("pikapro"),
                "showdown_revision": subprocess.check_output(
                    ["git", "-C", str(DEFAULT_SHOWDOWN_REPO), "rev-parse", "HEAD"], text=True
                ).strip(),
                "random_preview": args.random_preview,
                "team_manifest": str(manifest),
                "first_seed": args.seed,
                "games": args.games,
                "matched": args.games - len(differences),
                "differences": differences,
            },
            indent=2,
        )
    )
    return int(bool(differences))


if __name__ == "__main__":
    raise SystemExit(main())
