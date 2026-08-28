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
from vgc.rl.env import DirectBattle, SIDES


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

    def state_for(self, side: str) -> BattleMechanicsState:
        return dict(self.public_states)[side]


def evaluate_exact_branches(
    root: DirectBattle,
    joint_choices: Sequence[dict[str, str]],
    *,
    future_seeds: Sequence[Sequence[int] | None] = (None,),
    branch_prefix: str = "mechanics",
) -> list[ExactMechanicsBranch]:
    """Execute every `(joint choice, future seed)` branch in exact Showdown clones.

    `joint_choices` must contain exactly the sides Showdown is waiting for. Supplying
    multiple future seeds turns accuracy, damage, critical-hit, secondary-effect, wake,
    thaw, and Speed-tie randomness into explicit samples without changing the shared
    past. The root battle is never mutated.
    """

    if not joint_choices:
        raise ValueError("at least one joint choice is required")
    if not future_seeds:
        raise ValueError("at least one future seed is required")
    expected_sides = set(root.sides_to_move())
    branches: list[ExactMechanicsBranch] = []
    for choice_index, choices in enumerate(joint_choices):
        if set(choices) != expected_sides:
            raise ValueError(
                f"root expects choices from {sorted(expected_sides)}, got "
                f"{sorted(choices)}"
            )
        for seed_index, raw_seed in enumerate(future_seeds):
            seed = tuple(int(value) for value in raw_seed) if raw_seed is not None else None
            branch_id = f"{branch_prefix}-{choice_index}-{seed_index}"
            clone = root.clone(branch_id, seed=seed)
            try:
                result = clone.step(dict(choices))
                branches.append(
                    ExactMechanicsBranch(
                        branch_id=branch_id,
                        choices=tuple(sorted(choices.items())),
                        future_seed=seed,
                        request_state=result.request_state,
                        ended=result.ended,
                        winner=result.winner,
                        public_states=tuple(
                            (side, snapshot_battle(clone.battles[side])) for side in SIDES
                        ),
                        public_lines=tuple(
                            (side, tuple(result.lines[side])) for side in SIDES
                        ),
                    )
                )
            finally:
                clone.close()
    return branches
