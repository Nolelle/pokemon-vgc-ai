"""Learn terminal outcomes from small endgames reached by real game prefixes.

No simulator-state patches, teacher labels, search shortlists, or private actor
features. The fixture's empirical maxpower result is a commissioning reference,
not an optimal-strategy oracle. Learning near the upper bound of one is the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from vgc.config import REPO_ROOT
from vgc.rl.agents import DirectAgent, make_direct_agent
from vgc.rl.env import DirectBattle, SimWorker
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.ppo import PpoConfig, RolloutBuffer, ppo_update

CONFIG = PpoConfig(gamma=1.0, teacher_anchor_weight=0.0, reward_shaping_coef=0.0,
                   epochs=4, minibatch_size=64)
TEAM = (REPO_ROOT / "teams/phase2_mirror.packed.txt").read_text().strip()


def root_from_fixture(worker, fixture, tag):
    root = DirectBattle.start(worker, tag, TEAM, TEAM, seed=fixture["seed"])
    try:
        for choice in fixture["prefix"]:
            root.step(choice)
    except Exception:
        root.close()
        raise
    return root


def episode(root, side, seed, *, model=None, buffer=None, policy="maxpower"):
    random.seed(seed)
    rng = random.Random(seed)
    battle = root.clone(f"{root.battle_id}-episode",
                        seed=[rng.randrange(1, 2**31) for _ in range(4)])
    other = "p2" if side == "p1" else "p1"
    if model is None:
        actor = make_direct_agent(policy, TEAM)
    else:
        actor = DirectAgent(PpoVgcPlayer(
            model=model, rollout_buffer=buffer, ppo_config=CONFIG, policy_seed=seed,
            team=TEAM, start_listening=False,
        ))
    actors = {side: actor, other: make_direct_agent("maxpower", TEAM)}
    decisions = 0
    try:
        for s, agent in actors.items():
            agent.observe(battle.battle_id, battle.last_lines[s])
        while not battle.ended:
            if decisions > 100:
                raise RuntimeError("endgame did not terminate within diagnostic cap")
            choices = {s: actors[s].choose(battle.battles[s]) for s in battle.sides_to_move()}
            decisions += int(side in choices)
            result = battle.step(choices)
            for s, agent in actors.items():
                agent.observe(battle.battle_id, result.lines[s])
        actor.finish(battle.battles[side])
        if getattr(actor.player, "fallback_count", 0):
            raise RuntimeError("actor fallback invalidated endgame trajectory")
        return battle.outcome(side), decisions
    finally:
        battle.close()


def mine(worker, output):
    fixtures = []
    for seed in range(1, 801):
        random.seed(1000 + seed)
        root = DirectBattle.start(worker, f"mine-{seed}", TEAM, TEAM, seed=[seed] * 4)
        agents = {s: make_direct_agent("random", TEAM, preview_order="team 1234")
                  for s in ("p1", "p2")}
        prefix = []
        for s, agent in agents.items():
            agent.observe(root.battle_id, root.last_lines[s])
        checked = set()
        try:
            while not root.ended:
                for side in root.sides_to_move():
                    view = root.battles[side]
                    alive = sum(not p.fainted for p in view.team.values()) - 2
                    foes = list(view.opponent_team.values())
                    if (side in checked or view.teampreview or alive != 1 or len(foes) != 4
                            or sum(not p.fainted for p in foes) != 1 or any(view.force_switch)):
                        continue
                    checked.add(side)
                    baseline = [episode(root, side, 5000 + i) for i in range(8)]
                    random_results = [episode(root, side, 6000 + i, policy="random")[0]
                                      for i in range(8)]
                    # Freeze tasks before learning. Do not select on learned-policy results.
                    if all(r == 1 for r, _ in baseline) and sum(r == 1 for r in random_results) <= 5:
                        fixtures.append({"seed": [seed] * 4, "side": side,
                                         "prefix": list(prefix), "maxpower": baseline,
                                         "random": random_results})
                        print("fixture", len(fixtures), "seed", seed, side, flush=True)
                        if len(fixtures) == 8:
                            output.parent.mkdir(parents=True, exist_ok=True)
                            output.write_text(json.dumps({"team_sha256": hashlib.sha256(
                                TEAM.encode()).hexdigest(), "fixtures": fixtures}, indent=2) + "\n")
                            return
                # Probing episodes seed Python RNG, so restore an independent prefix stream.
                random.seed(seed * 1000 + len(prefix))
                choice = {s: agents[s].choose(root.battles[s]) for s in root.sides_to_move()}
                prefix.append(choice)
                result = root.step(choice)
                for s, agent in agents.items():
                    agent.observe(root.battle_id, result.lines[s])
        finally:
            root.close()
    raise RuntimeError(f"only found {len(fixtures)} eligible tasks; no suite written")


def train(worker, fixtures_path, output, seed, iterations, lr, task, hard_task,
          games_per_update=32, eval_every=16, checkpoint_selection="endpoint"):
    suite = json.loads(fixtures_path.read_text())
    assert suite["team_sha256"] == hashlib.sha256(TEAM.encode()).hexdigest()
    if task is not None:
        suite["fixtures"] = [suite["fixtures"][task]]
    roots = [root_from_fixture(worker, f, f"task-{i}") for i, f in enumerate(suite["fixtures"])]
    torch.manual_seed(seed)
    model = CandidatePolicyValueNet(use_information_features=True,
                                    use_mechanics_features=True, use_tactical_features=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed": seed, "learning_rate": lr, "config": asdict(CONFIG),
              "task_subset": task,
              "hard_task": hard_task,
              "games_per_update": games_per_update, "eval_every_updates": eval_every,
              "checkpoint_selection": checkpoint_selection,
              "architecture": RL_ARCHITECTURE_VERSION,
              "status": "diagnostic_unapproved",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(
        fixtures_path.read_bytes()).hexdigest(), "evaluations": [], "updates": []}

    best_worst_task = -1

    def save(path, iteration):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": asdict(CONFIG), "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "seed": seed,
                    "iteration": iteration, "games_seen": iteration * games_per_update,
                    "architecture": RL_ARCHITECTURE_VERSION,
                    "status": "diagnostic_unapproved",
                    "source_sha256": report["source_sha256"],
                    "fixture_sha256": report["fixture_sha256"]}, path)

    def evaluate(iteration):
        nonlocal best_worst_task
        rows = []
        for i, (root, fixture) in enumerate(zip(roots, suite["fixtures"], strict=True)):
            results = [episode(root, fixture["side"], 900000 + 1000 * i + j,
                               model=model)[0] for j in range(64)]
            rows.append({"task": i, "wins": results.count(1.0), "games": len(results)})
        row = {"iteration": iteration, "tasks": rows,
               "win_rate": sum(r["wins"] for r in rows) / sum(r["games"] for r in rows)}
        report["evaluations"].append(row)
        worst = min(r["wins"] for r in rows)
        if checkpoint_selection == "worst-task" and worst > best_worst_task:
            best_worst_task = worst
            report["selected_iteration"] = iteration
            report["selected_worst_task_wins"] = worst
            save(output / "selected_checkpoint.pt", iteration)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("eval", seed, row, flush=True)

    try:
        evaluate(0)
        for iteration in range(1, iterations + 1):
            buffer = RolloutBuffer()
            for game in range(games_per_update):
                index = game % len(roots)
                if hard_task is not None:
                    index = hard_task if game % 2 == 0 else (game // 2) % len(roots)
                game_index = (iteration - 1) * games_per_update + game
                episode(roots[index], suite["fixtures"][index]["side"],
                        # Same per-game randomness as the historical 32-game run;
                        # only update grouping changes in the larger-batch trial.
                        seed * 1000000 + (game_index // 32 + 1) * 100 + game_index % 32,
                        model=model, buffer=buffer)
            metrics = ppo_update(model, optimizer, buffer, CONFIG, rng=rng)
            report["updates"].append({"iteration": iteration, **metrics})
            if iteration % eval_every == 0 or iteration == iterations:
                evaluate(iteration)
        save(output / "checkpoint.pt", iterations)
    finally:
        for root in roots:
            root.close()


def confirm(worker, fixtures_path, checkpoint, output, task, future_seed):
    """Fresh future seeds; no updates or checkpoint selection on these outcomes."""
    suite = json.loads(fixtures_path.read_text())
    assert suite["team_sha256"] == hashlib.sha256(TEAM.encode()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = CandidatePolicyValueNet(use_information_features=True,
                                    use_mechanics_features=True, use_tactical_features=True)
    model.load_state_dict(saved["model"], strict=True)
    rows = []
    for i, fixture in enumerate(suite["fixtures"]):
        if task is not None and i != task:
            continue
        root = root_from_fixture(worker, fixture, f"confirm-{i}")
        try:
            learned = [episode(root, fixture["side"], future_seed + 1000 * i + j,
                               model=model)[0] for j in range(128)]
            reference = [episode(root, fixture["side"], future_seed + 1000 * i + j)[0]
                         for j in range(128)]
            rows.append({"task": i, "wins": learned.count(1.0), "games": 128,
                         "reference_wins": reference.count(1.0)})
        finally:
            root.close()
    report = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "fixture_sha256": hashlib.sha256(fixtures_path.read_bytes()).hexdigest(),
              "future_seed_base": future_seed, "tasks": rows,
              "passed_point_target": all(r["wins"] / r["games"] >= 0.95 for r in rows)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mine", "train", "confirm"))
    parser.add_argument("--fixtures", type=Path, default=Path("data/research/endgames_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/rl_sections_1_8/endgames"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--games-per-update", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--checkpoint-selection", choices=("endpoint", "worst-task"),
                        default="endpoint")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--task", type=int, choices=range(8))
    parser.add_argument("--hard-task", type=int, choices=range(8))
    parser.add_argument("--future-seed", type=int, default=1900000)
    args = parser.parse_args()
    if args.games_per_update < 1 or args.eval_every < 1 or args.iterations < 1:
        parser.error("games-per-update, eval-every, and iterations must be positive")
    if args.task is not None and args.hard_task is not None:
        parser.error("task subset and hard-task sampling are separate experiments")
    destination = args.fixtures if args.mode == "mine" else args.output
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        parser.error("refusing to overwrite existing experiment evidence; use a new output path")
    torch.set_num_threads(1)
    started = time.monotonic()
    with SimWorker() as worker:
        if args.mode == "mine":
            mine(worker, args.fixtures)
        elif args.mode == "train":
            train(worker, args.fixtures, args.output, args.seed, args.iterations, args.lr,
                  args.task, args.hard_task, args.games_per_update, args.eval_every,
                  args.checkpoint_selection)
        else:
            if args.checkpoint is None:
                parser.error("confirm requires --checkpoint")
            confirm(worker, args.fixtures, args.checkpoint, args.output, args.task, args.future_seed)
    print("elapsed", time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()
