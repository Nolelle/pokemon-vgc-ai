"""Fixed-size state and legal-joint-action encoding for candidate-aware RL.

The state half reuses the exact schema-v4 BC encoder used by replay training and live
inference. The action half encodes each *complete* doubles order (both active slots),
including move/switch/pass identity, resolved target, switch species, and once-per-turn
mechanic flags. A policy therefore scores legal joint actions directly rather than
combining two independently predicted moves after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from poke_env.player.battle_order import DoubleBattleOrder

from vgc.bc.encoding import (
    MOVE_TO_IDX,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    TARGET_TO_IDX,
    encode_state,
    flatten_state,
)
from vgc.bc.policy import battle_state_record
from vgc.bc.selfplay import order_to_action_dict
from vgc.battle_memory import BattleMemory

NUM_ORDER_SLOTS = 2
NUM_ACTION_FLAGS = 4  # mega, z-move, dynamax, tera
HISTORY_SCALAR_DIM = 16


@dataclass(frozen=True)
class CandidateFeatures:
    """Unpadded features for all legal candidates in one decision state."""

    move_indices: np.ndarray  # (candidates, 2)
    target_indices: np.ndarray  # (candidates, 2)
    switch_species_indices: np.ndarray  # (candidates, 2)
    flags: np.ndarray  # (candidates, 2, 4)

    def __len__(self) -> int:
        return int(self.move_indices.shape[0])


def encode_live_state(battle, config=None) -> tuple[np.ndarray, np.ndarray]:
    """Return BC-compatible ``(index_array, scalars+zero-slot-marker)`` for a battle.

    BC historically appends a two-value active-slot marker. A joint-order policy has
    no single deciding slot, so both values are zero. Retaining the width allows the
    full BC state trunk to be warm-started without changing its first linear layer.
    """

    record = battle_state_record(battle, config)
    index_array, scalars = flatten_state(encode_state(record))
    return index_array, np.concatenate((scalars, np.zeros(SLOT_FEATURE_DIM, dtype=np.float32)))


def _mean_hp(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values) if values else 0.0


def encode_battle_history(memory: BattleMemory) -> np.ndarray:
    """Compact longitudinal context accumulated before the current decision.

    Values are bounded to roughly ``[-1, 1]`` or ``[0, 1]`` so this branch can train
    stably beside the existing normalized BC scalars. It captures sequence information
    the current board alone cannot: repeated lines, observed opponent habits, target
    preferences, recent HP momentum, weather changes, and active-slot turnover.
    """

    move_count = sum(sum(counts.values()) for counts in memory.opponent_moves.values())
    unique_moves = len({move for counts in memory.opponent_moves.values() for move in counts})
    protect_count = sum(memory.opponent_protects.values())
    target_count = sum(memory.opponent_targets.values())
    target_concentration = (
        max(memory.opponent_targets.values(), default=0) / target_count if target_count else 0.0
    )
    order_count = len(memory.our_orders)
    unique_orders = len({order for _turn, order in memory.our_orders})
    repeated_last_order = float(
        order_count >= 2 and memory.our_orders[-1][1] == memory.our_orders[-2][1]
    )
    move_concentration = (
        max(
            (count for counts in memory.opponent_moves.values() for count in counts.values()),
            default=0,
        )
        / move_count
        if move_count
        else 0.0
    )
    switch_count = sum(memory.opponent_switches.values())

    weather_changes = 0
    our_turnover = 0
    opp_turnover = 0
    for previous, current in zip(memory.turns, memory.turns[1:]):
        weather_changes += int(previous.weather != current.weather)
        our_turnover += len(set(previous.our_active) - set(current.our_active))
        opp_turnover += len(set(previous.opponent_active) - set(current.opponent_active))

    our_momentum = 0.0
    opp_momentum = 0.0
    if len(memory.turns) >= 2:
        previous, current = memory.turns[-2], memory.turns[-1]
        our_momentum = (_mean_hp(current.our_hp) - _mean_hp(previous.our_hp)) / 100.0
        opp_momentum = (_mean_hp(current.opponent_hp) - _mean_hp(previous.opponent_hp)) / 100.0

    values = np.asarray(
        [
            min(len(memory.turns), 20) / 20.0,
            min(move_count, 24) / 24.0,
            min(unique_moves, 16) / 16.0,
            protect_count / move_count if move_count else 0.0,
            min(switch_count, 12) / 12.0,
            target_concentration,
            min(order_count, 20) / 20.0,
            repeated_last_order,
            min(unique_orders, 12) / 12.0,
            move_concentration,
            min(len(memory.opponent_switches), 6) / 6.0,
            min(len(memory.opponent_targets), 4) / 4.0,
            min(weather_changes, 6) / 6.0,
            min(our_turnover, 8) / 8.0,
            min(opp_turnover, 8) / 8.0,
            float(np.clip(opp_momentum - our_momentum, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert values.shape == (HISTORY_SCALAR_DIM,)
    return values


def _single_features(single, action: dict[str, object]) -> tuple[int, int, int, list[float]]:
    kind = action.get("kind")
    if kind == "move":
        move_idx = MOVE_TO_IDX.get(str(action.get("move_id") or ""), MOVE_TO_IDX["<unk>"])
        raw_target = str(action.get("target_slot") or "<none>")
        target_token = "self_or_field" if raw_target == "self" else raw_target
        target_idx = TARGET_TO_IDX.get(target_token, TARGET_TO_IDX["<none>"])
        switch_species_idx = SPECIES_TO_IDX["<pad>"]
    elif kind == "switch":
        move_idx = MOVE_TO_IDX["<switch>"]
        target_idx = TARGET_TO_IDX["<none>"]
        switch_species_idx = SPECIES_TO_IDX.get(
            str(action.get("switch_species") or ""), SPECIES_TO_IDX["<unk>"]
        )
    else:
        move_idx = MOVE_TO_IDX["<pass>"]
        target_idx = TARGET_TO_IDX["<none>"]
        switch_species_idx = SPECIES_TO_IDX["<pad>"]

    flags = [
        float(bool(getattr(single, "mega", False))),
        float(bool(getattr(single, "z_move", False))),
        float(bool(getattr(single, "dynamax", False))),
        float(bool(getattr(single, "terastallize", False))),
    ]
    return move_idx, target_idx, switch_species_idx, flags


def encode_candidates(orders: Sequence[DoubleBattleOrder]) -> CandidateFeatures:
    """Encode every legal joint order in enumeration order."""

    move_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    species_rows: list[list[int]] = []
    flag_rows: list[list[list[float]]] = []
    for order in orders:
        action = order_to_action_dict(order)
        singles = (order.first_order, order.second_order)
        encoded = [
            _single_features(single, action[f"slot{slot}"]) for slot, single in enumerate(singles)
        ]
        move_rows.append([entry[0] for entry in encoded])
        target_rows.append([entry[1] for entry in encoded])
        species_rows.append([entry[2] for entry in encoded])
        flag_rows.append([entry[3] for entry in encoded])

    count = len(orders)
    return CandidateFeatures(
        move_indices=np.asarray(move_rows, dtype=np.int64).reshape(count, NUM_ORDER_SLOTS),
        target_indices=np.asarray(target_rows, dtype=np.int64).reshape(count, NUM_ORDER_SLOTS),
        switch_species_indices=np.asarray(species_rows, dtype=np.int64).reshape(
            count, NUM_ORDER_SLOTS
        ),
        flags=np.asarray(flag_rows, dtype=np.float32).reshape(
            count, NUM_ORDER_SLOTS, NUM_ACTION_FLAGS
        ),
    )


def pad_candidate_features(
    candidates: Sequence[CandidateFeatures],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad a variable-size candidate list batch and return its legality mask."""

    if not candidates:
        raise ValueError("candidate batch must be non-empty")
    if any(len(entry) == 0 for entry in candidates):
        raise ValueError("every state must have at least one legal candidate")
    batch = len(candidates)
    width = max(len(entry) for entry in candidates)
    moves = np.full((batch, width, NUM_ORDER_SLOTS), MOVE_TO_IDX["<pass>"], dtype=np.int64)
    targets = np.full((batch, width, NUM_ORDER_SLOTS), TARGET_TO_IDX["<none>"], dtype=np.int64)
    species = np.full((batch, width, NUM_ORDER_SLOTS), SPECIES_TO_IDX["<pad>"], dtype=np.int64)
    flags = np.zeros((batch, width, NUM_ORDER_SLOTS, NUM_ACTION_FLAGS), dtype=np.float32)
    mask = np.zeros((batch, width), dtype=bool)
    for row, entry in enumerate(candidates):
        size = len(entry)
        moves[row, :size] = entry.move_indices
        targets[row, :size] = entry.target_indices
        species[row, :size] = entry.switch_species_indices
        flags[row, :size] = entry.flags
        mask[row, :size] = True
    return moves, targets, species, flags, mask
