"""Exact counterfactual branches through the official Showdown engine.

This module is intentionally separate from a live policy. It receives a `DirectBattle`
owned by an offline collector, clones Showdown's complete internal state, executes legal
joint choices, and returns only each player's fogged public observation. The policy never
receives the omniscient clone or hidden opponent data.

Use this to create mechanics-correct teacher targets. Do not label a choice with
`vgc.search.resolve_exchange` when an exact Showdown branch is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vgc.mechanics_state import BattleMechanicsState, snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import SIDES, DirectBattle, InvalidChoice, choice_string, step_many


@dataclass(frozen=True)
class ExactMechanicsBranch:
    branch_id: str
    choices: tuple[tuple[str, str], ...]
    future_seed: tuple[int, ...] | None
    request_state: str
    ended: bool
    winner: str | None
    public_states: tuple[tuple[str, BattleMechanicsState], ...]
    public_lines: tuple[tuple[str, tuple[str, ...]], ...]
    # Multi-turn continuation bookkeeping (all zero/False when continuation is off).
    continuation_turns_completed: int = 0
    continuation_steps: int = 0
    continuation_ended_early: bool = False  # the battle ended before all turns were played
    continuation_truncated: bool = False  # step budget ran out / no side could move

    def state_for(self, side: str) -> BattleMechanicsState:
        return dict(self.public_states)[side]


def _battle_turn(clone: DirectBattle) -> int:
    return max(int(getattr(clone.battles[side], "turn", 0) or 0) for side in SIDES)


def _continuation_orders(battle, policy: str, config: PolicyConfig):
    """Ranked orders of the continuation policy from ONE side's own fogged view."""

    if policy == "myopic":
        from vgc.evaluator import score_joint_orders

        return score_joint_orders(battle, config)
    if policy == "search":
        from vgc.search import search_joint_orders

        return search_joint_orders(battle, config)
    raise ValueError(f"unknown exact_search_continuation_policy {policy!r}")


def _continue_branches(
    clones: list[DirectBattle],
    base_turn: int,
    turns: int,
    policy: str,
    config: PolicyConfig,
) -> list[dict[str, object]]:
    """Advance every clone `turns` more COMPLETED turns, all clones stepped together.

    Turns are counted from the clone's battle turn number, not from requests: a faint's
    forced replacement is an extra step INSIDE a turn. The searched step already moved the
    clone to `base_turn + 1` (or left it on a replacement request inside `base_turn`).
    Each side that owes a choice picks the top order of the continuation policy from its
    own fogged view; a side on a `wait` request is not in `sides_to_move()` so is skipped.

    Hidden-trap rejection inside a continuation is NOT fatal (unlike the searched step):
    Showdown re-requests only the rejected side, whose view now knows it is trapped, so
    the next loop pass simply re-ranks and sends a legal order; after two rejections in
    a row that side sends "default". The searched step stays strict because there an
    unresolved switch would be scored as a free exchange; here the turn is merely replayed.
    """

    target = base_turn + 1 + turns
    budget = 4 * turns + 4
    info: list[dict[str, object]] = [{"steps": 0, "streak": 0, "truncated": False} for _ in clones]
    worker = clones[0].worker
    while True:
        pending: list[tuple[DirectBattle, dict[str, str]]] = []
        owners: list[int] = []
        for index, clone in enumerate(clones):
            if clone.ended or _battle_turn(clone) >= target or info[index]["truncated"]:
                continue
            sides = clone.sides_to_move()
            if not sides or int(info[index]["steps"]) >= budget:
                info[index]["truncated"] = True
                continue
            choices: dict[str, str] = {}
            for side in sides:
                if int(info[index]["streak"]) >= 2:
                    choices[side] = "default"
                    continue
                ranked = _continuation_orders(clone.battles[side], policy, config)
                choices[side] = choice_string(ranked[0].order) if ranked else "default"
            pending.append((clone, choices))
            owners.append(index)
        if not pending:
            return info
        before = [getattr(clone, "hidden_trap_rejections", 0) for clone, _ in pending]
        step_many(worker, pending)
        for (clone, _choices), index, count in zip(pending, owners, before):
            info[index]["steps"] = int(info[index]["steps"]) + 1
            rejected = getattr(clone, "hidden_trap_rejections", 0) != count
            info[index]["streak"] = int(info[index]["streak"]) + 1 if rejected else 0


def evaluate_exact_branches(
    root: DirectBattle,
    joint_choices: Sequence[dict[str, str]],
    *,
    future_seeds: Sequence[Sequence[int] | None] = (None,),
    branch_prefix: str = "mechanics",
    config: PolicyConfig | None = None,
) -> list[ExactMechanicsBranch]:
    """Execute every `(joint choice, future seed)` branch in exact Showdown clones.

    `joint_choices` must contain exactly the sides Showdown is waiting for. Supplying
    multiple future seeds turns accuracy, damage, critical-hit, secondary-effect, wake,
    thaw, and Speed-tie randomness into explicit samples without changing the shared
    past. The root battle is never mutated.

    With `config.exact_search_continuation_turns > 0` each clone keeps playing that many
    more completed turns (both sides on `exact_search_continuation_policy`, same seeded
    RNG stream) before its board is snapshotted. Without `config` (or with 0) the
    behaviour is exactly the one-turn judge.
    """

    if not joint_choices:
        raise ValueError("at least one joint choice is required")
    if not future_seeds:
        raise ValueError("at least one future seed is required")
    turns = int(config.exact_search_continuation_turns) if config is not None else 0
    policy = config.exact_search_continuation_policy if config is not None else "myopic"
    expected_sides = set(root.sides_to_move())
    base_turn = _battle_turn(root)
    # (partial branch kwargs, clone) kept open only while a continuation still needs them.
    live: list[tuple[dict[str, object], DirectBattle]] = []
    branches: list[ExactMechanicsBranch] = []

    def snapshot(kwargs: dict[str, object], clone: DirectBattle, **extra) -> None:
        branches.append(
            ExactMechanicsBranch(
                **kwargs,
                request_state=clone.request_state,
                ended=clone.ended,
                winner=clone.winner,
                public_states=tuple((side, snapshot_battle(clone.battles[side])) for side in SIDES),
                **extra,
            )
        )

    try:
        for choice_index, choices in enumerate(joint_choices):
            if set(choices) != expected_sides:
                raise ValueError(
                    f"root expects choices from {sorted(expected_sides)}, got {sorted(choices)}"
                )
            for seed_index, raw_seed in enumerate(future_seeds):
                seed = tuple(int(v) for v in raw_seed) if raw_seed is not None else None
                branch_id = f"{branch_prefix}-{choice_index}-{seed_index}"
                clone = root.clone(branch_id, seed=seed)
                keep = False
                try:
                    rejections_before = getattr(clone, "hidden_trap_rejections", 0)
                    result = clone.step(dict(choices))
                    if getattr(clone, "hidden_trap_rejections", 0) != rejections_before:
                        # DirectBattle lets a live game retry after Showdown's hidden-trap
                        # rejection, but this branch never resolved its turn: scoring its
                        # unchanged board would read the rejected switch as a free exchange.
                        raise InvalidChoice(
                            f"branch {branch_id}: switch rejected by a hidden trap; the "
                            "turn did not resolve"
                        )
                    kwargs = {
                        "branch_id": branch_id,
                        "choices": tuple(sorted(choices.items())),
                        "future_seed": seed,
                        "public_lines": tuple((side, tuple(result.lines[side])) for side in SIDES),
                    }
                    if turns > 0 and not clone.ended:
                        live.append((kwargs, clone))
                        keep = True
                    else:
                        snapshot(kwargs, clone)
                finally:
                    if not keep:
                        clone.close()
        if live:
            assert config is not None
            clones = [clone for _kwargs, clone in live]
            info = _continue_branches(clones, base_turn, turns, policy, config)
            for (kwargs, clone), i in zip(live, info, strict=True):
                done = max(0, min(turns, _battle_turn(clone) - base_turn - 1))
                snapshot(
                    kwargs,
                    clone,
                    continuation_turns_completed=done,
                    continuation_steps=int(i["steps"]),
                    continuation_ended_early=bool(clone.ended and done < turns),
                    continuation_truncated=bool(i["truncated"]),
                )
    finally:
        for _kwargs, clone in live:
            clone.close()
    # Branch order must equal (choice, seed) order, as before: continuation branches were
    # appended after the one-turn ones, so restore it.
    order = {
        f"{branch_prefix}-{c}-{s}": c * len(future_seeds) + s
        for c in range(len(joint_choices))
        for s in range(len(future_seeds))
    }
    branches.sort(key=lambda branch: order[branch.branch_id])
    return branches
