"""Controlled public-reveal experiment; uses the certified core via PYTHONPATH.

The research-only blind adapter removes Counter/Mirror Coat reveals from both
public protocol memory and the parsed opponent move dictionary. It never changes
the simulator. Oracle choices are diagnostics, never learning targets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from poke_env.battle.move import MoveSet

from research_full_confirmation import load_model
from research_tactical_tasks import (
    CONFIG, damage_bounds, mon, new_model, root_from_fixture, selected_order, team, wish,
)
from vgc.actions import enumerate_joint_orders
from vgc.config import FORMAT_ID, REPO_ROOT, SHOWDOWN_REPO
from vgc.node import find_node, node_environment
from vgc.rl.agents import DirectAgent
from vgc.rl.env import SimWorker, choice_string
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.ppo import RolloutBuffer, _tensor_batch, ppo_update

REVEALS = {"counter", "mirrorcoat"}


def fixtures():
    helper = mon("Garganacl", "PurifyingSalt", ["Explosion", "Protect"])
    own = team(wish("Clefable"), wish("Gardevoir"), helper,
               mon("Primarina", "Torrent", ["Liquidation", "Moonblast", "Protect"],
                   item="LifeOrb", nature="Lonely", spread=",32,2,32,,"))
    rows = []
    for move in ("Counter", "MirrorCoat"):
        foe = team(wish("Gardevoir"), wish("Chimecho"), helper,
                   mon("Swampert", "Torrent", [move, "Protect"], spread=",2,,32,,32"))
        rows.append({"name": move.lower(), "teams": {"p1": own, "p2": foe},
                     "seed": [13]*4, "prefix": [
                         {"p1": "team 1234", "p2": "team 1234"},
                         {"p1": "move healingwish, move healingwish",
                          "p2": "move healingwish, move healingwish"},
                         {"p1": "switch 3, switch 4", "p2": "switch 3, switch 4"},
                         {"p1": "move explosion, move protect",
                          "p2": "move explosion, move protect"},
                         {"p1": "pass, move protect", "p2": f"pass, move {move.lower()}"},
                     ]})
    return rows


class RevealBlindAgent(DirectAgent):
    def observe(self, battle_tag, lines):
        filtered = []
        for line in lines:
            fields = line.split("|")
            if (len(fields) > 3 and fields[1] == "move" and fields[2].startswith("p2")
                    and fields[3].lower().replace(" ", "") in REVEALS):
                continue
            filtered.append(line)
        super().observe(battle_tag, filtered)

    def choose(self, battle):
        saved = [(p, p._moves) for p in battle.opponent_team.values()]
        try:
            for p, moves in saved:
                p._moves = MoveSet({key: value for key, value in moves.moves.items()
                                    if key not in REVEALS})
            return super().choose(battle)
        finally:
            for p, moves in saved:
                p._moves = moves


def agent_for(model, fixture, seed, blind=False, buffer=None):
    cls = RevealBlindAgent if blind else DirectAgent
    return cls(PpoVgcPlayer(model=model, rollout_buffer=buffer, ppo_config=CONFIG,
                           policy_seed=seed, team=fixture["teams"]["p1"], start_listening=False))


def episode(root, fixture, seed, *, model=None, blind=False, buffer=None, fixed_move=None):
    rng = random.Random(seed)
    battle = root.clone(f"{root.battle_id}-episode", seed=[rng.randrange(1, 2**31) for _ in range(4)])
    agent = agent_for(model, fixture, seed, blind, buffer) if model is not None else None
    decisions = 0
    first_choice = None
    try:
        if agent:
            agent.observe(battle.battle_id, battle.last_lines["p1"])
        while not battle.ended:
            if decisions >= 100:
                raise RuntimeError("information task exceeded decision cap")
            choices = {}
            for side in battle.sides_to_move():
                view = battle.battles[side]
                if side == "p1":
                    decisions += 1
                    choices[side] = (agent.choose(view) if agent else
                                     selected_order(view, {1: (fixed_move, 2)}) if fixed_move else
                                     choice_string(rng.choice(enumerate_joint_orders(view))))
                    if first_choice is None:
                        first_choice = choices[side]
                else:
                    available = {m.id for m in view.available_moves[1]}
                    move = fixture["name"] if fixture["name"] in available else (
                        "protect" if "protect" in available else "struggle")
                    choices[side] = selected_order(view, {1: (move, None)})
            result = battle.step(choices)
            if agent:
                agent.observe(battle.battle_id, result.lines["p1"])
        if agent:
            agent.finish(battle.battles["p1"])
            if agent.player.fallback_count:
                raise RuntimeError("actor fallback invalidates evidence")
        return {"outcome": battle.outcome("p1"), "decisions": decisions, "first_choice": first_choice}
    finally:
        battle.close()


def encoded_root(root, fixture, model, *, blind):
    branch = root.clone(f"{root.battle_id}-encoding")
    try:
        buffer = RolloutBuffer()
        actor = agent_for(model, fixture, 101, blind, buffer)
        actor.observe(branch.battle_id, branch.last_lines["p1"])
        actor.choose(branch.battles["p1"])
        assert not actor.player.fallback_count and len(buffer) == 1
        batch = _tensor_batch(buffer.steps, "cpu")
        # Only actual network inputs; PPO labels/returns are not observations.
        inputs = {k: v for k, v in batch.items() if k not in {
            "actions", "old_log_probs", "old_values", "returns", "advantages", "teacher_actions"}}
        return inputs
    finally:
        branch.close()


def input_difference(a, b):
    assert a.keys() == b.keys()
    return [k for k in a if not torch.equal(a[k], b[k])]


def evaluate(roots, fs, model, blind, games, base):
    rows = []
    for i, (root, fixture) in enumerate(zip(roots, fs, strict=True)):
        # Same policy/randomness schedule in the two variants; variant is never an input.
        records = [episode(root, fixture, base+j, model=model, blind=blind) for j in range(games)]
        rows.append({"task": fixture["name"], "wins": sum(r["outcome"] == 1 for r in records),
                     "games": games, "records": records})
    return rows


def context():
    return {"source_snapshot": str(REPO_ROOT), "source_commit": subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(json.dumps(fixtures(), sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probe", "train", "confirm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--future-seed", type=int, default=22000000)
    args = parser.parse_args()
    if args.output.exists() or min(args.games, args.iterations, args.eval_games) < 1:
        parser.error("positive budgets and fresh output required")
    from vgc.mechanics_gate import assert_mechanics_ready
    from vgc.battle_state_gate import assert_battle_state_ready
    from vgc.action_gate import assert_action_ready
    for gate in (assert_mechanics_ready, assert_battle_state_ready, assert_action_ready):
        gate("revealed-information research")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    fs = fixtures()
    node = find_node()
    for packed in {t for f in fs for t in f["teams"].values()}:
        validated = subprocess.run([node, "pokemon-showdown", "validate-team", FORMAT_ID],
                                   cwd=SHOWDOWN_REPO, input=packed, text=True, capture_output=True,
                                   env=node_environment(node))
        if validated.returncode:
            raise RuntimeError(validated.stdout + validated.stderr)
    report = {"status": "diagnostic_unapproved", "complete": False, "provenance": context(),
              "seed": args.seed, "blind": args.blind, "config": asdict(CONFIG),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
              if args.checkpoint else None, "fixtures": fs}
    with SimWorker() as worker:
        roots = [root_from_fixture(worker, f, f"reveal-{i}") for i, f in enumerate(fs)]
        try:
            model = load_model(args.checkpoint) if args.checkpoint else new_model(args.seed)
            if args.mode == "probe":
                report["blind_input_differences"] = input_difference(*[
                    encoded_root(r, f, model, blind=True) for r, f in zip(roots, fs, strict=True)])
                report["aware_input_differences"] = input_difference(*[
                    encoded_root(r, f, model, blind=False) for r, f in zip(roots, fs, strict=True)])
                assert not report["blind_input_differences"], report["blind_input_differences"]
                assert report["aware_input_differences"]
                hidden = []
                for i, fixture in enumerate(fs):
                    f = copy.deepcopy(fixture)
                    f["prefix"] = f["prefix"][:-1]
                    root = root_from_fixture(worker, f, f"hidden-{i}")
                    try:
                        hidden.append(encoded_root(root, f, model, blind=False))
                    finally:
                        root.close()
                report["unrevealed_private_input_differences"] = input_difference(*hidden)
                assert not report["unrevealed_private_input_differences"]
                report["damage_bounds"] = damage_bounds(roots[0], [
                    {"source": [0, 1], "target": [1, 1], "move": move}
                    for move in ("liquidation", "moonblast")])
                report["policies"] = []
                for i, (root, fixture) in enumerate(zip(roots, fs, strict=True)):
                    for move in ("liquidation", "moonblast", None):
                        results = [episode(root, fixture, 21000000+j, fixed_move=move)
                                   for j in range(args.games)]
                        report["policies"].append({"task": fixture["name"], "move": move,
                            "wins": sum(r["outcome"] == 1 for r in results), "games": args.games})
                print(report["policies"], flush=True)
            elif args.mode == "confirm":
                if args.checkpoint is None:
                    parser.error("confirmation requires checkpoint")
                report["tasks"] = evaluate(roots, fs, model, args.blind, args.games, args.future_seed)
            else:
                if args.checkpoint is None:
                    parser.error("training requires an explicit initial checkpoint")
                args.output.mkdir(parents=True)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
                rng = np.random.default_rng(args.seed)
                report.update({"learning_rate": 0.0003, "games_per_update": 32,
                               "selection": "fixed endpoint", "updates": [], "evaluations": []})
                for iteration in range(args.iterations+1):
                    if iteration:
                        buffer = RolloutBuffer()
                        for j in range(32):
                            i = j % 2
                            episode(roots[i], fs[i], args.seed*1000000 + iteration*100+j,
                                    model=model, blind=args.blind, buffer=buffer)
                        metrics = ppo_update(model, optimizer, buffer, CONFIG, rng=rng)
                        assert all(np.isfinite(v) for v in metrics.values())
                        report["updates"].append({"iteration": iteration, "decisions": len(buffer), **metrics})
                    if iteration % 8 == 0 or iteration == args.iterations:
                        rows = evaluate(roots, fs, model, args.blind, args.eval_games, 21500000)
                        report["evaluations"].append({"iteration": iteration, "tasks": rows})
                        print("eval", args.seed, args.blind, iteration,
                              [(r["task"], r["wins"]) for r in rows], flush=True)
                        (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
                saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                saved.update({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                              "rng_state": copy.deepcopy(rng.bit_generator.state), "torch_rng": torch.get_rng_state(),
                              "research_provenance": context(), "research_seed": args.seed,
                              "research_blind": args.blind, "research_iterations": args.iterations,
                              "status": "diagnostic_unapproved"})
                torch.save(saved, args.output / "endpoint.pt")
            report.update({"complete": True, "elapsed_seconds": time.monotonic()-started})
            target = args.output / "report.json" if args.mode == "train" else args.output
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2)+"\n")
            print("completed", args.mode, report["elapsed_seconds"], flush=True)
        finally:
            for root in roots:
                root.close()


if __name__ == "__main__":
    main()
