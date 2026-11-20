#!/usr/bin/env python
"""Probe one local decision with sampled public reconstructions; never plays online.

The engine samples and network are experimental. This is a wiring check, not a
measurement of ladder strength. Sixteen worlds, eight actions and four random futures
are the defaults; use smaller limits for a fast smoke check.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np

from vgc.actions import enumerate_joint_orders
from vgc.models import PolicyConfig
from vgc.rl.env import DirectBattle, SimWorker, choice_string
from vgc.rl.live_mirror import LiveExactMirror, mirror_side
from vgc.rl.replica_search import SampledWorld, search_sampled_worlds


def resolve_rejected_choices(clone, choices, select, *, max_retries=8):
    """Resolve hidden-information rejections, stopping at the first completed step.

    A rejected switch/disabled move reveals an updated request. Retry only the sides
    now asked to choose, using their own updated views; never advance another turn.
    """
    attempted = {side: {choice} for side, choice in choices.items()}
    retries = 0
    while True:
        before = getattr(clone, "hidden_trap_rejections", 0)
        turn_before = tuple(getattr(clone.battles[side], "turn", None) for side in ("p1", "p2"))
        clone.step(choices)
        after = getattr(clone, "hidden_trap_rejections", 0)
        if after == before:
            return retries
        turn_after = tuple(getattr(clone.battles[side], "turn", None) for side in ("p1", "p2"))
        if clone.ended or turn_after != turn_before:
            raise RuntimeError("rejection unexpectedly advanced or ended the battle")
        retries += 1
        if retries > max_retries:
            raise RuntimeError("hidden-information retry limit exceeded")
        waiting = clone.sides_to_move()
        if not waiting:
            raise RuntimeError("rejection did not produce an updated request")
        choices = {}
        for side in waiting:
            legal = select(clone.battles[side])
            previous = attempted.setdefault(side, set())
            replacement = next(
                (choice_string(order) for order in legal if choice_string(order) not in previous),
                None,
            )
            if replacement is None:
                raise RuntimeError(f"no new legal choice after rejection for {side}")
            choices[side] = replacement
            previous.add(replacement)


def search_public_decision(
    battle,
    packed_team,
    policy,
    *,
    worlds=16,
    actions=8,
    seeds=4,
    iterations=20000,
    seed=1,
    mirror=None,
):
    """Search one player-view decision and return original legal orders and statistics.

    The mirror guesses hidden information using only this player's observation and
    team. Network calls see only the appropriate player's parsed observation, never
    the simulator object. A forced-switch request may require only one player's input.
    Pass a reusable ``mirror`` to avoid spawning a new worker on every turn.
    """
    if not 1 <= worlds <= 16 or not 1 <= actions <= 8 or seeds < 1:
        raise ValueError("worlds/actions/seeds must be within 1..16/1..8/positive")
    config = dataclasses.replace(PolicyConfig(), exact_search_total_hypotheses=worlds)
    side = mirror_side(battle)
    other = "p2" if side == "p1" else "p1"
    rng = np.random.default_rng(seed)
    future_seeds = [tuple(map(int, rng.integers(0, 2**16, size=4))) for _ in range(seeds)]
    counter = itertools.count()
    roots = []
    retry_count = 0
    # Offline harnesses sometimes attach the true private simulator to battle; remove
    # that authority before passing the observation to the mirror or model.
    private = {
        key: vars(battle).pop(key)
        for key in ("_vgc_direct_root", "_vgc_direct_side")
        if key in vars(battle)
    }
    owns_mirror = mirror is None
    try:
        if owns_mirror:
            mirror = LiveExactMirror(packed_team, config)
        hypotheses = mirror.hypotheses(battle)
        sampled = []
        for hypothesis in hypotheses:
            root = mirror.build(battle, hypothesis)
            roots.append(root)
            if side not in root.sides_to_move():
                raise ValueError("player is not asked to choose in reconstructed world")
            opponent_view = root.battles[other] if other in root.sides_to_move() else None
            sampled.append(SampledWorld(root, opponent_view, hypothesis.weight))

        def select(view):
            if view is None:
                return [None]
            legal = enumerate_joint_orders(view)
            if not legal:
                raise ValueError("no legal joint choices in player view")
            logits, _ = policy.score(view, legal)
            logits = np.asarray(logits, dtype=float).reshape(-1)
            if len(logits) != len(legal) or not np.isfinite(logits).all():
                raise ValueError("network returned invalid action scores")
            return [legal[i] for i in np.argsort(-logits, kind="stable")]

        def key(order):
            return "<wait>" if order is None else choice_string(order)

        def matrix(root, our_orders, opponent_orders):
            nonlocal retry_count
            cells = [[[] for _ in opponent_orders] for _ in our_orders]
            pending_positions = []
            pending_battles = []
            for i, our_order in enumerate(our_orders):
                for j, opponent_order in enumerate(opponent_orders):
                    for future_seed in future_seeds:
                        clone = root.clone(f"replica-probe-{next(counter)}", seed=future_seed)
                        try:
                            choices = {side: choice_string(our_order)}
                            if opponent_order is not None:
                                choices[other] = choice_string(opponent_order)
                            start_turn = clone.battles[side].turn
                            retry_count += resolve_rejected_choices(clone, choices, select)
                            # A pivot or a faint may interrupt move resolution. Choose
                            # replacements before judging the next turn's position.
                            for _ in range(8):
                                if clone.ended or clone.battles[side].turn > start_turn:
                                    break
                                if clone.request_state != "switch":
                                    raise RuntimeError(
                                        "one-turn branch stopped outside a switch request"
                                    )
                                forced = {
                                    who: choice_string(select(clone.battles[who])[0])
                                    for who in clone.sides_to_move()
                                }
                                retry_count += resolve_rejected_choices(clone, forced, select)
                            else:
                                raise RuntimeError(
                                    "one-turn search exceeded eight replacement requests"
                                )
                            if clone.ended:
                                cells[i][j].append(clone.outcome(side))
                            else:
                                pending_positions.append((i, j))
                                pending_battles.append(clone.battles[side])
                        finally:
                            clone.close()
            if pending_battles:
                batch_values = (
                    policy.values(pending_battles)
                    if hasattr(policy, "values")
                    else [float(policy.value(b)) for b in pending_battles]
                )
                for (i, j), value in zip(pending_positions, batch_values, strict=True):
                    if not np.isfinite(value):
                        raise ValueError("network returned a non-finite value")
                    cells[i][j].append(float(value))
            result = np.empty((len(our_orders), len(opponent_orders)))
            for i in range(len(our_orders)):
                for j in range(len(opponent_orders)):
                    result[i, j] = float(np.mean(cells[i][j]))
            return result

        result = search_sampled_worlds(
            battle,
            sampled,
            select,
            matrix,
            action_key=key,
            max_actions=actions,
            max_worlds=worlds,
            iterations=iterations,
        )
        return result, {**mirror.last_hypothesis_audit, "rejected_choice_retries": retry_count}
    finally:
        for root in roots:
            root.close()
        if owns_mirror and mirror is not None:
            mirror.close()
        vars(battle).update(private)


def paired_match(policy, team, opponent_team, *, pairs, worlds, actions, seeds, iterations, seed):
    """Play seat-swapped local pairs on pikapro with reusable search mirrors."""
    from vgc.rl.replica_engine import ReplicaRustWorker

    config = dataclasses.replace(PolicyConfig(), exact_search_total_hypotheses=worlds)
    mirrors = {
        "p1": LiveExactMirror(team, config),
        "p2": LiveExactMirror(opponent_team, config),
    }
    outcomes = []
    try:
        with ReplicaRustWorker() as worker:
            for pair in range(pairs):
                sim_rng = np.random.default_rng(seed + pair)
                sim_seed = tuple(map(int, sim_rng.integers(0, 2**16, size=4)))
                for search_side in ("p1", "p2"):
                    teams = {"p1": team, "p2": opponent_team}
                    policy_rng = np.random.default_rng(seed + pair)
                    counters = {
                        "search_decisions": 0,
                        "actual_worlds": 0,
                        "rejected_choice_retries": 0,
                        "errors": 0,
                    }
                    battle = DirectBattle.start(
                        worker,
                        f"replica-pair-{pair}-{search_side}",
                        teams["p1"],
                        teams["p2"],
                        seed=sim_seed,
                    )
                    try:
                        decisions = 0
                        while not battle.ended:
                            choices = {}
                            for side in battle.sides_to_move():
                                view = battle.battles[side]
                                if view.teampreview:
                                    choices[side] = choice_string(policy.choose_preview(view))
                                elif side == search_side:
                                    result, audit = search_public_decision(
                                        view,
                                        teams[side],
                                        policy,
                                        worlds=worlds,
                                        actions=actions,
                                        seeds=seeds,
                                        iterations=iterations,
                                        seed=int(policy_rng.integers(0, 2**31)),
                                        mirror=mirrors[side],
                                    )
                                    selected = int(
                                        policy_rng.choice(
                                            len(result.actions), p=result.probabilities
                                        )
                                    )
                                    choices[side] = choice_string(result.actions[selected])
                                    counters["search_decisions"] += 1
                                    counters["actual_worlds"] += audit["searched"]
                                    counters["rejected_choice_retries"] += audit[
                                        "rejected_choice_retries"
                                    ]
                                else:
                                    legal = enumerate_joint_orders(view)
                                    if not legal:
                                        raise ValueError("plain policy has no legal joint choices")
                                    logits, _ = policy.score(view, legal)
                                    choices[side] = choice_string(legal[int(np.argmax(logits))])
                            battle.step(choices)
                            decisions += 1
                            if decisions > 1000:
                                raise RuntimeError("local game exceeded 1000 decisions")
                        outcomes.append(
                            {
                                "pair": pair,
                                "search_side": search_side,
                                "sim_seed": sim_seed,
                                "outcome": battle.outcome(search_side),
                                "decisions": decisions,
                                **counters,
                            }
                        )
                    except Exception as exc:
                        counters["errors"] += 1
                        outcomes.append(
                            {
                                "pair": pair,
                                "search_side": search_side,
                                "sim_seed": sim_seed,
                                "outcome": None,
                                "decisions": decisions,
                                "error": str(exc),
                                **counters,
                            }
                        )
                    finally:
                        battle.close()
    finally:
        for mirror in mirrors.values():
            mirror.close()
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--team", type=Path, default=Path("teams/phase2_mirror.packed.txt"))
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--actions", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--paired-games",
        type=int,
        default=0,
        help="number of seat-swapped local game pairs (default: opening only)",
    )
    parser.add_argument("--opponent-team", type=Path, help="packed holdout team for paired games")
    args = parser.parse_args()
    import torch

    from vgc.rl.replication_policy import load_replication_policy

    torch.set_num_threads(1)
    policy = load_replication_policy(args.checkpoint, device="cpu")
    team = args.team.read_text().strip()
    if args.paired_games:
        if args.paired_games < 0 or args.opponent_team is None:
            parser.error("positive --paired-games requires --opponent-team")
        opponent_team = args.opponent_team.read_text().strip()
        results = paired_match(
            policy,
            team,
            opponent_team,
            pairs=args.paired_games,
            worlds=args.worlds,
            actions=args.actions,
            seeds=args.seeds,
            iterations=args.iterations,
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "p1_team": str(args.team),
                    "p2_team": str(args.opponent_team),
                    "checkpoint": str(args.checkpoint),
                    "results": results,
                },
                indent=2,
            )
        )
        if any(row["errors"] for row in results):
            raise SystemExit(1)
        return
    with SimWorker() as worker:
        battle = DirectBattle.start(worker, "replica-probe-source", team, team, seed=[1, 2, 3, 4])
        try:
            battle.step({"p1": "team 1234", "p2": "team 1234"})
            result, audit = search_public_decision(
                battle.battles["p1"],
                team,
                policy,
                worlds=args.worlds,
                actions=args.actions,
                seeds=args.seeds,
                iterations=args.iterations,
                seed=args.seed,
            )
            print(
                json.dumps(
                    {
                        "choices": [choice_string(a) for a in result.actions],
                        "probabilities": result.probabilities,
                        "value": result.value,
                        "gap": result.exploitability,
                        "worlds": audit,
                    },
                    indent=2,
                )
            )
        finally:
            battle.close()


if __name__ == "__main__":
    main()
