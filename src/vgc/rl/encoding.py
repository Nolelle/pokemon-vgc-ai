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

NUM_ORDER_SLOTS = 2
NUM_ACTION_FLAGS = 4  # mega, z-move, dynamax, tera


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
    return index_array, np.concatenate(
        (scalars, np.zeros(SLOT_FEATURE_DIM, dtype=np.float32))
    )


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
            _single_features(single, action[f"slot{slot}"])
            for slot, single in enumerate(singles)
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
    targets = np.full(
        (batch, width, NUM_ORDER_SLOTS), TARGET_TO_IDX["<none>"], dtype=np.int64
    )
    species = np.full(
        (batch, width, NUM_ORDER_SLOTS), SPECIES_TO_IDX["<pad>"], dtype=np.int64
    )
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
