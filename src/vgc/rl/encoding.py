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

from vgc.archetypes import ARCHETYPES, classify_team
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
from vgc.damage import to_id
from vgc.models import PolicyConfig
from vgc.sets import load_set_priors, opponent_move_ids

NUM_ORDER_SLOTS = 2
NUM_ACTION_FLAGS = 4  # mega, z-move, dynamax, tera
HISTORY_SCALAR_DIM = 16

# --- Meta-aware context (deployment-available only -- see encode_meta_context below) --

# One archetype-one-hot slot per label in vgc.archetypes.ARCHETYPES, plus one trailing
# "unknown" slot for when classify_team can't match any archetype's defining core.
_ARCHETYPE_LABELS: tuple[str, ...] = tuple(archetype.label for archetype in ARCHETYPES)
_ARCHETYPE_LABEL_TO_INDEX: dict[str, int] = {
    label: index for index, label in enumerate(_ARCHETYPE_LABELS)
}
META_ARCHETYPE_SLOTS = len(_ARCHETYPE_LABELS) + 1  # +1 unknown slot

# Doubles: up to two opponent Pokemon are ever active at once.
META_ACTIVE_SLOTS = 2
# Per active opponent slot: (1) a presence flag (this slot has a live, known opponent
# Pokemon -- 0.0 when fainted/absent/battle hasn't started), (2) the fraction of that
# species' corpus set-prior move-frequency mass already covered by moves it has
# actually REVEALED this battle, and (3) how many additional prior-likely moves
# `vgc.sets.opponent_move_ids` would fill in beyond what's revealed, normalized by
# `PolicyConfig.set_prior_max_moves` (a coarse "how much is still unknown" signal).
META_PRIOR_SCALARS_PER_SLOT = 3
META_SCALAR_DIM = META_ARCHETYPE_SLOTS * 2 + META_ACTIVE_SLOTS * META_PRIOR_SCALARS_PER_SLOT


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


def _archetype_one_hot(species_ids) -> np.ndarray:
    """One-hot over `_ARCHETYPE_LABELS` + trailing unknown slot for `species_ids`."""

    vector = np.zeros(META_ARCHETYPE_SLOTS, dtype=np.float32)
    label = classify_team(species_ids)
    vector[_ARCHETYPE_LABEL_TO_INDEX[label] if label in _ARCHETYPE_LABEL_TO_INDEX else -1] = 1.0
    return vector


def _opponent_species_ids(battle) -> list[str]:
    """The opponent's species, deployment-available only.

    Prefers the full previewed 6 (`battle.teampreview_opponent_team`, populated once
    during team preview and never cleared -- see poke-env's
    `AbstractBattle._register_teampreview_pokemon`), since all 6 species are visible at
    team preview regardless of whether Open Team Sheets triggered. Falls back to
    whatever is currently revealed mid-battle (`battle.opponent_team`) if team preview
    data isn't available for some reason (e.g. a synthetic/stub battle in tests).
    """

    preview = list(getattr(battle, "teampreview_opponent_team", None) or [])
    if preview:
        return [pokemon.species for pokemon in preview if getattr(pokemon, "species", None)]
    opponent_team = getattr(battle, "opponent_team", None) or {}
    return [
        pokemon.species for pokemon in opponent_team.values() if getattr(pokemon, "species", None)
    ]


def _our_species_ids(battle) -> list[str]:
    """Our own team's species -- always fully known, we built the team."""

    team = getattr(battle, "team", None) or {}
    return [pokemon.species for pokemon in team.values() if getattr(pokemon, "species", None)]


def _prior_scalars_for_active(
    pokemon, priors: dict[str, object], config: PolicyConfig
) -> tuple[float, float, float]:
    """`(presence, revealed_fraction, remaining_normalized)` for one active opponent
    Pokemon -- see `META_PRIOR_SCALARS_PER_SLOT`'s comment for what each means. Uses
    only `pokemon.moves` (moves it has actually revealed by using/being tracked this
    battle) and the corpus prior tables -- never a ground-truth hidden moveset.
    """

    if pokemon is None or getattr(pokemon, "fainted", False):
        return 0.0, 0.0, 0.0

    revealed = [move_id for move_id in (to_id(raw) for raw in (pokemon.moves or {})) if move_id]

    # Reuses vgc.sets.opponent_move_ids (revealed + top corpus-prior moves filled in up
    # to config.set_prior_max_moves) rather than reimplementing the ranking/threshold
    # logic; the count it adds beyond what's actually revealed is "how many more moves
    # the prior thinks are likely but we haven't seen yet".
    filled = opponent_move_ids(pokemon, priors, config)
    max_moves = max(config.set_prior_max_moves, 1)
    remaining_normalized = min(1.0, max(0.0, (len(filled) - len(revealed)) / max_moves))

    revealed_fraction = 0.0
    species_id = to_id(pokemon.species)
    entry = (priors.get("species") or {}).get(species_id) if priors else None
    if entry and entry.get("appearances", 0) >= config.set_prior_min_games:
        move_counts: dict[str, int] = entry.get("moves") or {}
        total_mass = sum(move_counts.values())
        if total_mass > 0:
            revealed_mass = sum(move_counts.get(move_id, 0) for move_id in revealed)
            revealed_fraction = revealed_mass / total_mass

    return 1.0, revealed_fraction, remaining_normalized


def _encode_meta_context(battle, config: PolicyConfig) -> np.ndarray:
    opponent_archetype = _archetype_one_hot(_opponent_species_ids(battle))
    our_archetype = _archetype_one_hot(_our_species_ids(battle))

    priors = load_set_priors() if config.use_set_priors else {}
    active_opponents = list(getattr(battle, "opponent_active_pokemon", None) or [])
    active_opponents = (active_opponents + [None, None])[:META_ACTIVE_SLOTS]
    prior_scalars: list[float] = []
    for pokemon in active_opponents:
        prior_scalars.extend(_prior_scalars_for_active(pokemon, priors, config))

    vector = np.concatenate(
        (
            opponent_archetype,
            our_archetype,
            np.asarray(prior_scalars, dtype=np.float32),
        )
    ).astype(np.float32)
    assert vector.shape == (META_SCALAR_DIM,)
    return vector


def encode_meta_context(battle, config: PolicyConfig | None = None) -> np.ndarray:
    """Deployment-available meta context: opponent/our archetype one-hots plus a few
    compact set-prior/reveal scalars for the currently active opponent Pokemon.

    Every input is something a real ladder opponent would actually show us -- species
    (team preview or in-battle reveals), our own team, revealed moves, and corpus
    set-priors (`vgc.sets.load_set_priors`). Never touches ground-truth hidden Stat
    Points, exact secret movesets, or anything else the sim knows but a real opponent
    wouldn't reveal (see this module's and the RL meta-features design doc's warnings).

    Guarded end-to-end: any missing/malformed battle data (a stub battle in tests, a
    battle before team preview has populated anything, a species not in the archetype/
    prior corpora) falls back to a well-formed all-zero/"unknown" vector rather than
    raising -- this must never throw during a live battle.
    """

    config = config or PolicyConfig()
    try:
        return _encode_meta_context(battle, config)
    except Exception:
        return np.zeros(META_SCALAR_DIM, dtype=np.float32)


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
