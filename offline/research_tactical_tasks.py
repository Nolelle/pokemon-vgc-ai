"""Legal, scripted-prefix tactical tasks with explicit engine-checked witnesses.

Healing Wish is used only in actual legal game prefixes to reach small states.
The learner receives ordinary fogged requests and public history. Oracle policies
are diagnostic controls, never PPO labels or private actor features.
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

import torch
import numpy as np

from commission_rl import forward_batch
from vgc.actions import enumerate_joint_orders
from vgc.config import FORMAT_ID, REPO_ROOT, SHOWDOWN_REPO
from vgc.node import find_node, node_environment
from vgc.model_vocabulary import require_current_vocabulary
from vgc.rl.agents import DirectAgent
from vgc.rl.env import DirectBattle, SimWorker, choice_string
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION
from vgc.rl.ppo import PpoConfig, RolloutBuffer, _tensor_batch, ppo_update

CONFIG = PpoConfig(gamma=1.0, teacher_anchor_weight=0.0, reward_shaping_coef=0.0,
                   epochs=4, minibatch_size=64)


def mon(species, ability, moves, *, item="", nature="Hardy", spread=""):
    return f"{species}||{item}|{ability}|{','.join(moves)}|{nature}|{spread}||||50|"


def team(*members):
    # Two legal unbrought sets keep the actual bring-six/pick-four rules intact.
    unused = Path("teams/phase2_mirror.packed.txt").read_text().strip().split("]")
    return "]".join([*members, unused[1], unused[3]])


def wish(species):
    abilities = {"Clefable": "MagicGuard", "Gardevoir": "Trace", "Chimecho": "Levitate"}
    return mon(species, abilities[species], ["HealingWish", "Protect"])


def garchomp(*, setup=False):
    return mon("Garchomp", "RoughSkin",
               ["Earthquake" if setup else "DragonClaw", "DragonClaw" if setup else "PoisonJab",
                "SwordsDance", "Protect"], nature="Adamant", spread="2,32,,,,32")


def build_fixtures():
    slowbro = mon("Slowbro", "OwnTempo", ["SlackOff", "IceBeam", "CalmMind", "Protect"],
                  nature="Bold", spread="32,,32,,2,")
    starmie = mon("Starmie", "NaturalCure", ["IceBeam", "CosmicPower", "Recover", "Protect"],
                  nature="Modest", spread=",,2,32,32,")
    helper = mon("Clefable", "MagicGuard", ["HelpingHand", "LifeDew"])
    exploder = mon("Garganacl", "PurifyingSalt", ["Explosion", "Protect"])
    own_setup = team(wish("Clefable"), wish("Gardevoir"), exploder, garchomp(setup=True))
    foe_setup = team(wish("Gardevoir"), wish("Chimecho"), exploder, slowbro)
    foe_targets = team(wish("Gardevoir"), wish("Chimecho"), starmie, helper)
    preview = {"p1": "team 1234", "p2": "team 1234"}
    wishes = {"p1": "move healingwish, move healingwish",
              "p2": "move healingwish, move healingwish"}
    replacements = {"p1": "switch 3, switch 4", "p2": "switch 3, switch 4"}
    setup_prefix = [preview, wishes, replacements,
                    {"p1": "move explosion, move protect",
                     "p2": "move explosion, move protect"}]
    target_prefix = [preview, wishes, replacements,
                     {"p1": "move memento 2, move dragonclaw 1",
                      "p2": "move cosmicpower, move helpinghand -1"}]
    metagross = mon("Metagross", "ClearBody", ["Explosion", "Protect"], item="LifeOrb",
                    nature="Adamant", spread="32,32,,,,2")
    return [
        {"name": "setup", "teams": {"p1": own_setup, "p2": foe_setup},
         "prefix": setup_prefix, "opponent": "recovery"},
        {"name": "replacement", "teams": {"p1": own_setup, "p2": foe_setup},
         "prefix": [preview,
                    {"p1": "move healingwish, move protect", **{"p2": wishes["p2"]}},
                    {"p1": "switch 3, pass", "p2": replacements["p2"]},
                    {"p1": "move explosion, move healingwish",
                     "p2": "move explosion, move protect"}],
         "opponent": "recovery"},
        {"name": "target", "teams": {
            "p1": team(wish("Clefable"), wish("Gardevoir"),
                       mon("Ninetales", "FlashFire", ["Memento", "Protect"]), garchomp()),
            "p2": foe_targets}, "prefix": target_prefix, "opponent": "icebeam"},
        {"name": "sacrifice", "teams": {
            "p1": team(wish("Clefable"), wish("Gardevoir"), metagross, garchomp()),
            "p2": foe_targets}, "prefix": [preview, wishes, replacements],
         "opponent": "icebeam"},
    ]


def root_from_fixture(worker, fixture, tag):
    teams = fixture["teams"]
    root = DirectBattle.start(worker, tag, teams["p1"], teams["p2"], seed=fixture["seed"])
    try:
        for choices in fixture["prefix"]:
            root.step(choices)
        if root.ended:
            raise ValueError(f"prefix ended task {fixture['name']}")
        return root
    except Exception:
        root.close()
        raise


def selected_order(battle, wanted):
    orders = enumerate_joint_orders(battle)
    for order in orders:
        matches = True
        for slot, (move, target) in wanted.items():
            single = order.first_order if slot == 0 else order.second_order
            if getattr(single.order, "id", None) != move:
                matches = False
            if target is not None and single.move_target != target:
                matches = False
        if matches:
            return choice_string(order)
    raise ValueError(f"script unavailable: {wanted}; legal={[choice_string(o) for o in orders]}")


def opponent_order(battle, kind):
    wanted = {}
    for slot, pokemon in enumerate(battle.active_pokemon):
        if pokemon is None or pokemon.fainted:
            continue
        moves = {m.id for m in battle.available_moves[slot]}
        if pokemon.species == "clefable":
            partner = battle.active_pokemon[1 - slot]
            move = ("helpinghand" if partner is not None and not partner.fainted
                    and "helpinghand" in moves else "lifedew" if "lifedew" in moves
                    else "helpinghand" if "helpinghand" in moves else "struggle")
            wanted[slot] = (move, None)
        elif kind == "recovery" and "slackoff" in moves:
            wanted[slot] = ("slackoff", None)
        else:
            targets = [(i + 1, p) for i, p in enumerate(battle.opponent_active_pokemon)
                       if p is not None and not p.fainted]
            target = next((i for i, p in targets if p.species == "garchomp"), targets[0][0])
            wanted[slot] = ("icebeam" if "icebeam" in moves else "struggle", target)
    return selected_order(battle, wanted)


def witness_order(battle, kind):
    if any(battle.force_switch):
        return choice_string(enumerate_joint_orders(battle)[0])
    if kind in ("setup", "replacement"):
        slot, pokemon = next((i, p) for i, p in enumerate(battle.active_pokemon)
                             if p is not None and not p.fainted)
        move = "swordsdance" if pokemon.boosts["atk"] < 6 else "earthquake"
        return selected_order(battle, {slot: (move, None)})
    if kind == "target":
        foe = battle.opponent_active_pokemon[0]
        slot = next(i for i, p in enumerate(battle.active_pokemon) if p is not None and not p.fainted)
        return selected_order(battle, {slot: ("dragonclaw", 1)} if foe and not foe.fainted
                              else {slot: ("poisonjab", 2)})
    return selected_order(battle, {0: ("explosion", None), 1: ("protect", None)})


def episode(root, fixture, seed, *, model=None, buffer=None, random_actor=False):
    rng = random.Random(seed)
    battle = root.clone(f"{root.battle_id}-episode", seed=[rng.randrange(1, 2**31) for _ in range(4)])
    learner = None if model is None else DirectAgent(PpoVgcPlayer(
        model=model, rollout_buffer=buffer, ppo_config=CONFIG, policy_seed=seed,
        team=fixture["teams"]["p1"], start_listening=False))
    decisions = 0
    try:
        if learner:
            learner.observe(battle.battle_id, battle.last_lines["p1"])
        while not battle.ended:
            if decisions > 100:
                raise RuntimeError("tactical task exceeded decision cap")
            choices = {}
            for side in battle.sides_to_move():
                view = battle.battles[side]
                if side == "p2":
                    choices[side] = opponent_order(view, fixture["opponent"])
                else:
                    decisions += 1
                    choices[side] = (learner.choose(view) if learner else
                                     choice_string(rng.choice(enumerate_joint_orders(view)))
                                     if random_actor else witness_order(view, fixture["name"]))
            result = battle.step(choices)
            if learner:
                learner.observe(battle.battle_id, result.lines["p1"])
        if learner:
            learner.finish(battle.battles["p1"])
            if learner.player.fallback_count:
                raise RuntimeError("learner fallback invalidated tactical trajectory")
        return battle.outcome("p1"), decisions
    finally:
        battle.close()


def damage_bounds(root, queries):
    state = root.worker.request({"cmd": "dump", "id": root.battle_id})["state"]
    node = find_node()
    result = subprocess.run([node, str(REPO_ROOT / "tools/research_damage_bounds.mjs"),
                             str(SHOWDOWN_REPO)], input=json.dumps({"state": state, "queries": queries}),
                            text=True, capture_output=True, check=True, env=node_environment(node))
    bounds = json.loads(result.stdout)
    for bound in bounds:
        context = bound["context"]
        assert context["accuracy"] == 100
        assert context["sourceBoosts"]["accuracy"] == context["targetBoosts"]["evasion"] == 0
        assert context["sourceStatus"] == ""
        assert not {"confusion", "flinch", "attract"}.intersection(context["sourceVolatiles"])
    return bounds


def oracle_evidence(root, fixture):
    """Native damage bounds plus the real scripted setup; never actor input."""
    name = fixture["name"]
    if name in ("setup", "replacement"):
        branch = root.clone(f"{root.battle_id}-proof")
        try:
            while any(branch.battles["p1"].force_switch):
                branch.step({"p1": witness_order(branch.battles["p1"], name)})
            slot = next(i for i, p in enumerate(branch.battles["p1"].active_pokemon)
                        if p is not None and not p.fainted)
            query = {"source": [0, slot], "target": [1, 1], "move": "earthquake"}
            unboosted = damage_bounds(branch, [query])[0]
            for _ in range(3):
                branch.step({"p1": witness_order(branch.battles["p1"], name),
                             "p2": opponent_order(branch.battles["p2"], "recovery")})
            boosted = damage_bounds(branch, [query])[0]
            foe = branch.battles["p2"].active_pokemon[1]
            minimum = min(d["damage"] for d in boosted["damages"])
            assert minimum >= foe.current_hp, (minimum, foe.current_hp)
            return {"unboosted": unboosted, "boosted": boosted,
                    "foe_hp_before_payoff": foe.current_hp,
                    "minimum_boosted_damage": minimum,
                    "proof_scope": "three real Swords Dances, then guaranteed KO on all native rolls"}
        finally:
            branch.close()
    if name == "target":
        queries = [{"source": [0, 1], "target": [1, 0], "move": "dragonclaw"},
                   {"source": [0, 1], "target": [1, 1], "move": "poisonjab"},
                   {"source": [1, 0], "target": [0, 1], "move": "icebeam"}]
        bounds = damage_bounds(root, queries)
        own = root.battles["p1"].active_pokemon[1]
        foes = root.battles["p2"].active_pokemon
        assert min(d["damage"] for d in bounds[0]["damages"]) >= foes[0].current_hp
        assert bounds[0]["context"]["sourceSpeed"] > bounds[0]["context"]["targetSpeed"]
        # Life Dew heals one quarter between hits; two Poison Jabs suffice even
        # at minimum damage. This is not a claimed one-hit KO.
        minimum_poison = min(d["damage"] for d in bounds[1]["damages"])
        assert 2 * minimum_poison >= foes[1].current_hp + foes[1].max_hp // 4
        assert min(d["damage"] for d in bounds[2]["damages"]) >= own.current_hp
        return {"bounds": bounds, "hp": [own.current_hp, *[p.current_hp for p in foes]],
                "proof_scope": "KO Starmie before lethal Ice Beam; two Poison Jabs beat Clefable's Life Dew"}
    bounds = damage_bounds(root, [{"source": [0, 0], "target": [1, i],
                                  "move": "explosion", "spread": True} for i in range(2)] +
                           [{"source": [1, 0], "target": [0, 1], "move": "icebeam"}])
    foes = root.battles["p2"].active_pokemon
    assert all(min(d["damage"] for d in b["damages"]) >= p.current_hp
               for b, p in zip(bounds[:2], foes, strict=True))
    assert "stall" not in bounds[2]["context"]["targetVolatiles"]
    assert min(d["damage"] for d in bounds[2]["damages"]) >= bounds[2]["context"]["targetHP"]
    return {"bounds": bounds, "foe_hp": [p.current_hp for p in foes],
            "proof_scope": "first-use Protect shields Garchomp; Explosion KOs both foes on every roll"}


def new_model(seed):
    torch.manual_seed(seed)
    return CandidatePolicyValueNet(use_information_features=True,
                                    use_mechanics_features=True, use_tactical_features=True)


def experiment_context():
    files = sorted((REPO_ROOT / "src/vgc").rglob("*.py")) + [
        Path(__file__), REPO_ROOT / "tools/sim_worker.mjs",
        REPO_ROOT / "tools/research_damage_bounds.mjs"]
    hashes = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files}
    showdown_head = subprocess.check_output(
        ["git", "-C", str(SHOWDOWN_REPO), "rev-parse", "HEAD"], text=True).strip()
    return {"source_bundle_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "showdown_head": showdown_head,
            "catalog_sha256": hashlib.sha256((REPO_ROOT / "data/champions/mechanics_catalog.json").read_bytes()).hexdigest(),
            "torch_version": str(torch.__version__)}


def represent(roots, fixtures, args):
    """Overfit only verified winning root choices; this is not RL evidence."""
    model = new_model(args.seed)
    samples, labels = [], []
    for root, fixture in zip(roots, fixtures, strict=True):
        buffer = RolloutBuffer()
        agent = DirectAgent(PpoVgcPlayer(model=model, rollout_buffer=buffer,
                            ppo_config=CONFIG, policy_seed=args.seed,
                            team=fixture["teams"]["p1"], start_listening=False))
        desired = witness_order(root.battles["p1"], fixture["name"])
        orders = enumerate_joint_orders(root.battles["p1"])
        labels.append([choice_string(o) for o in orders].index(desired))
        agent.observe(root.battle_id, root.last_lines["p1"])
        agent.choose(root.battles["p1"])
        if len(buffer) != 1 or agent.player.fallback_count:
            raise RuntimeError("representability sample did not use production encoder")
        samples.append(buffer.steps[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _tensor_batch(samples, "cpu")
    targets = torch.tensor(labels)
    curve = []
    for iteration in range(201):
        model.eval()
        logits, _ = forward_batch(model, batch)
        if iteration in (0, 50, 100, 200):
            probabilities = logits.softmax(-1)[torch.arange(len(labels)), targets]
            curve.append({"iteration": iteration, "winning_action_probabilities":
                          probabilities.detach().tolist()})
        if iteration == 200:
            break
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {"kind": "supervised_representability_only", "seed": args.seed,
            "curve": curve, "passed": min(curve[-1]["winning_action_probabilities"]) >= 0.99}


def learn(roots, fixtures, args, fixture_sha):
    model = new_model(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(args.seed)
    report = {"status": "diagnostic_unapproved", "seed": args.seed, "config": asdict(CONFIG),
              "learning_rate": 0.001, "games_per_update": 32,
              "fixture_sha256": fixture_sha, "evaluations": [], "updates": [],
              "selection": "highest minimum development wins, earliest tie",
              "provenance": args.context}
    args.output.mkdir(parents=True)
    best = -1

    def save(path, iteration):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": asdict(CONFIG), "rng": copy.deepcopy(rng.bit_generator.state),
                    "torch_rng": torch.get_rng_state(), "architecture": RL_ARCHITECTURE_VERSION,
                    "fixture_sha256": fixture_sha, "iteration": iteration, "seed": args.seed,
                    "provenance": args.context,
                    "status": "diagnostic_unapproved"}, path)

    def evaluate(iteration):
        nonlocal best
        rows = []
        for i, (root, fixture) in enumerate(zip(roots, fixtures, strict=True)):
            results = [episode(root, fixture, 8100000 + i*1000 + j, model=model)[0]
                       for j in range(args.eval_games)]
            rows.append({"task": fixture["name"], "wins": results.count(1.0),
                         "draws": results.count(0.0), "games": len(results)})
        row = {"iteration": iteration, "tasks": rows,
               "win_rate": sum(r["wins"] for r in rows) / sum(r["games"] for r in rows)}
        report["evaluations"].append(row)
        minimum = min(r["wins"] for r in rows)
        if minimum > best:
            best = minimum
            report["selected_iteration"] = iteration
            save(args.output / "selected_checkpoint.pt", iteration)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("eval", args.seed, row, flush=True)

    evaluate(0)
    for iteration in range(1, args.iterations + 1):
        buffer = RolloutBuffer()
        decisions = 0
        for game in range(32):
            i = game % len(fixtures)
            _, count = episode(roots[i], fixtures[i], args.seed*1000000 + iteration*100 + game,
                               model=model, buffer=buffer)
            decisions += count
        metrics = ppo_update(model, optimizer, buffer, CONFIG, rng=rng)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise RuntimeError("nonfinite tactical update")
        report["updates"].append({"iteration": iteration, "decisions": decisions, **metrics})
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            evaluate(iteration)
    save(args.output / "checkpoint.pt", args.iterations)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "probe", "represent", "train", "confirm"))
    parser.add_argument("--fixtures", type=Path, default=Path("data/research/tactical_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--future-seed", type=int, default=9100000)
    args = parser.parse_args()
    if min(args.games, args.iterations, args.eval_every, args.eval_games) < 1:
        parser.error("game counts, iterations and evaluation interval must be positive")
    if args.output.exists() or (args.mode == "build" and args.fixtures.exists()):
        parser.error("refusing to overwrite evidence")
    torch.set_num_threads(1)
    args.context = experiment_context()
    started = time.monotonic()
    fixtures = build_fixtures() if args.mode == "build" else json.loads(args.fixtures.read_text())["fixtures"]
    node = find_node()
    if args.mode == "build":
        for packed in {t for f in fixtures for t in f["teams"].values()}:
            result = subprocess.run([node, "pokemon-showdown", "validate-team", FORMAT_ID],
                                    cwd=SHOWDOWN_REPO, env=node_environment(node), input=packed,
                                    text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"illegal research team: {result.stdout} {result.stderr}")
        for fixture in fixtures:
            fixture["seed"] = [13, 13, 13, 13]
    report = {"status": "oracle_diagnostic_only", "games_per_task": args.games, "tasks": [],
              "provenance": args.context,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.mode in ("represent", "train", "confirm"):
        with SimWorker() as worker:
            roots = [root_from_fixture(worker, f, f"tactical-{i}") for i, f in enumerate(fixtures)]
            try:
                fixture_sha = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
                if args.mode == "represent":
                    result = represent(roots, fixtures, args)
                elif args.mode == "train":
                    result = learn(roots, fixtures, args, fixture_sha)
                else:
                    if args.checkpoint is None:
                        parser.error("confirm requires checkpoint")
                    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                    if saved["architecture"] != RL_ARCHITECTURE_VERSION:
                        raise ValueError("checkpoint architecture mismatch")
                    canonical = "model_state_dict" in saved
                    if canonical:
                        require_current_vocabulary(saved, args.checkpoint)
                    elif saved["fixture_sha256"] != fixture_sha:
                        raise ValueError("checkpoint fixture identity mismatch")
                    model = new_model(args.seed)
                    model.load_state_dict(saved["model_state_dict"] if canonical else saved["model"],
                                          strict=True)
                    rows = []
                    for i, (root, fixture) in enumerate(zip(roots, fixtures, strict=True)):
                        outcomes = [episode(root, fixture, args.future_seed + 1000*i + j, model=model)[0]
                                    for j in range(args.games)]
                        rows.append({"task": fixture["name"], "wins": outcomes.count(1.0),
                                     "draws": outcomes.count(0.0), "games": len(outcomes)})
                    result = {"tasks": rows, "passed_point_target": all(
                        r["wins"]/r["games"] >= 0.95 for r in rows),
                        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                        "future_seed_base": args.future_seed,
                        "training_fixture_sha256": saved.get("fixture_sha256")}
                result.update({"source_sha256": report["source_sha256"],
                               "provenance": args.context,
                               "fixture_sha256": fixture_sha,
                               "elapsed_seconds": time.monotonic() - started})
                target = args.output / "report.json" if args.mode == "train" else args.output
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(result, indent=2) + "\n")
                print("completed", args.mode, "elapsed", result["elapsed_seconds"], flush=True)
                return
            finally:
                for root in roots:
                    root.close()
    with SimWorker() as worker:
        for i, fixture in enumerate(fixtures):
            root = root_from_fixture(worker, fixture, f"tactical-{i}")
            try:
                witnesses = [episode(root, fixture, 7100000 + i*1000 + j)
                             for j in range(args.games)]
                random_results = [episode(root, fixture, 7200000 + i*1000 + j, random_actor=True)[0]
                                  for j in range(args.games)]
                row = {"name": fixture["name"], "witness_wins": sum(r == 1 for r, _ in witnesses),
                       "random_wins": random_results.count(1.0), "random_draws": random_results.count(0.0),
                       "witness_decisions": sorted({n for _, n in witnesses})}
                row["native_oracle_evidence"] = oracle_evidence(root, fixture)
                report["tasks"].append(row)
                print({k: v for k, v in row.items() if k != "native_oracle_evidence"}, flush=True)
            finally:
                root.close()
    if args.mode == "build":
        args.fixtures.parent.mkdir(parents=True, exist_ok=True)
        args.fixtures.write_text(json.dumps({"format": FORMAT_ID, "fixtures": fixtures}, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
