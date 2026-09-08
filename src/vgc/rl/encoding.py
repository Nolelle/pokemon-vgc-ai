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
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.player.battle_order import DoubleBattleOrder

from vgc.archetypes import ARCHETYPES, classify_team
from vgc.bc.encoding import (
    ABILITY_TO_IDX,
    ITEM_TO_IDX,
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
from vgc.damage import PokemonState, damage_range, to_id
from vgc.data import load_moves, load_natures
from vgc.evaluator import (
    _attacker_state_for,
    _our_pokemon_state,
    _resolve_targets,
    build_context,
    field_effective_speed,
    resolves_before,
)
from vgc.models import PolicyConfig
from vgc.opponent_belief import (
    MAX_SPREAD_HYPOTHESES,
    PokemonBelief,
    build_opponent_beliefs,
)
from vgc.own_team import spread_index
from vgc.principles import utility_kind
from vgc.sets import load_set_priors, opponent_move_ids
from vgc.stats import STAT_IDS, calculate_stats

NUM_ORDER_SLOTS = 2
NUM_ACTION_FLAGS = 4  # mega, z-move, dynamax, tera
HISTORY_SCALAR_DIM = 16
TACTICAL_FEATURE_DIM = 42

# Full-information contract for the new learned pipeline.  The old BC-shaped state is
# retained above for checkpoint compatibility; this parallel branch supplies what it
# omitted: all six exact own sets and a fog-safe probability estimate for all six
# opposing preview species.
INFORMATION_TEAM_SLOTS = 6
INFORMATION_MOVES_PER_MON = 4
OWN_SCALARS_PER_MON = 28
OPP_SCALARS_PER_MON = 36
INFORMATION_SCALAR_DIM = (
    INFORMATION_TEAM_SLOTS * OWN_SCALARS_PER_MON
    + INFORMATION_TEAM_SLOTS * OPP_SCALARS_PER_MON
)
INFORMATION_INDEX_DIM = INFORMATION_TEAM_SLOTS * (3 + INFORMATION_MOVES_PER_MON) * 2
_STATUS_IDS = ("none", "brn", "par", "psn", "tox", "slp", "frz")
_STATUS_INDEX = {status: index for index, status in enumerate(_STATUS_IDS)}
_NORMALIZED_STAT_CEILING = 500.0


@dataclass(frozen=True)
class InformationFeatures:
    """Complete own-team facts plus fog-safe opponent probabilities."""

    indices: np.ndarray  # (84,) categorical ids
    scalars: np.ndarray  # (384,) normalized factual/probability values

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
    tactical: np.ndarray | None = None  # (candidates, 42), zero when battle omitted

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


def _distribution_uncertainty(probabilities: dict[str, float]) -> float:
    if not probabilities:
        return 1.0
    return float(1.0 - max(probabilities.values()))


def _status_vector(status) -> list[float]:
    status_id = getattr(status, "name", str(status or "none")).lower()
    vector = [0.0] * len(_STATUS_IDS)
    vector[_STATUS_INDEX.get(status_id, 0)] = 1.0
    return vector


def _own_information(battle) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    team = list((getattr(battle, "team", None) or {}).items())[:INFORMATION_TEAM_SLOTS]
    active_ids = {
        id(mon)
        for mon in (getattr(battle, "active_pokemon", None) or [])
        if mon is not None
    }
    builders = spread_index(getattr(battle, "teambuilder_team", None) or [])
    species_indices: list[int] = []
    item_indices: list[int] = []
    ability_indices: list[int] = []
    move_indices: list[int] = []
    scalars: list[float] = []
    natures = load_natures()
    for ident, mon in team:
        name = str(ident).split(": ", 1)[-1]
        builder = builders.get(name)
        species_id = to_id(getattr(mon, "species", None))
        species_indices.append(SPECIES_TO_IDX.get(species_id, SPECIES_TO_IDX["<unk>"]))
        original_item = to_id(getattr(builder, "item", None)) if builder is not None else None
        item_id = original_item or to_id(getattr(mon, "item", None))
        item_indices.append(ITEM_TO_IDX.get(item_id, ITEM_TO_IDX["<unk>"]))
        original_ability = (
            to_id(getattr(builder, "ability", None)) if builder is not None else None
        )
        ability_id = original_ability or to_id(getattr(mon, "ability", None))
        ability_indices.append(ABILITY_TO_IDX.get(ability_id, ABILITY_TO_IDX["<unk>"]))
        raw_moves = (
            list(getattr(builder, "moves", None) or [])
            if builder is not None
            else list((getattr(mon, "moves", None) or {}).keys())
        )
        encoded_moves = [
            MOVE_TO_IDX.get(to_id(move), MOVE_TO_IDX["<unk>"])
            for move in raw_moves[:INFORMATION_MOVES_PER_MON]
        ]
        encoded_moves += [MOVE_TO_IDX["<pad>"]] * (
            INFORMATION_MOVES_PER_MON - len(encoded_moves)
        )
        move_indices.extend(encoded_moves)

        try:
            if builder is not None and getattr(builder, "evs", None) is not None:
                spread = dict(zip(STAT_IDS, builder.evs, strict=True))
                nature = str(getattr(builder, "nature", None) or "serious").lower()
                stats = calculate_stats(species_id, spread, nature)
            else:
                state = _our_pokemon_state(mon)
                spread = state.sp_spread or {}
                nature = state.nature or "serious"
                stats = state.stats()
        except (AttributeError, KeyError, TypeError, ValueError):
            spread = dict.fromkeys(STAT_IDS, 0)
            nature = "serious"
            stats = {stat: 0 for stat in STAT_IDS}
        nature_data = natures.get(nature) or {}
        nature_vector = [
            1.0 if nature_data.get("plus") == stat else (-1.0 if nature_data.get("minus") == stat else 0.0)
            for stat in STAT_IDS[1:]
        ]
        scalars.extend(
            [
                1.0,
                float(id(mon) in active_ids),
                float(getattr(mon, "current_hp_fraction", 0.0) or 0.0),
                float(bool(getattr(mon, "fainted", False))),
                *_status_vector(getattr(mon, "status", None)),
                *[float(spread.get(stat, 0)) / 32.0 for stat in STAT_IDS],
                *[float(stats.get(stat, 0)) / _NORMALIZED_STAT_CEILING for stat in STAT_IDS],
                *nature_vector,
            ]
        )

    missing = INFORMATION_TEAM_SLOTS - len(team)
    species_indices.extend([SPECIES_TO_IDX["<pad>"]] * missing)
    item_indices.extend([ITEM_TO_IDX["<unk>"]] * missing)
    ability_indices.extend([ABILITY_TO_IDX["<unk>"]] * missing)
    move_indices.extend(
        [MOVE_TO_IDX["<pad>"]] * (missing * INFORMATION_MOVES_PER_MON)
    )
    scalars.extend([0.0] * (missing * OWN_SCALARS_PER_MON))
    return species_indices, item_indices, ability_indices, move_indices, scalars


def _opponent_information(
    battle, memory: BattleMemory, config: PolicyConfig
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    beliefs = build_opponent_beliefs(battle, memory, config)[:INFORMATION_TEAM_SLOTS]
    species_indices: list[int] = []
    item_indices: list[int] = []
    ability_indices: list[int] = []
    move_indices: list[int] = []
    scalars: list[float] = []
    for belief in beliefs:
        species_indices.append(
            SPECIES_TO_IDX.get(belief.species_id, SPECIES_TO_IDX["<unk>"])
        )
        item_indices.append(
            ITEM_TO_IDX.get(belief.known_item, ITEM_TO_IDX["<unk>"])
            if belief.known_item
            else ITEM_TO_IDX["<unk>"]
        )
        ability_indices.append(
            ABILITY_TO_IDX.get(belief.known_ability, ABILITY_TO_IDX["<unk>"])
            if belief.known_ability
            else ABILITY_TO_IDX["<unk>"]
        )
        encoded_moves = [
            MOVE_TO_IDX.get(move_id, MOVE_TO_IDX["<unk>"])
            for move_id in belief.revealed_moves[:INFORMATION_MOVES_PER_MON]
        ]
        encoded_moves += [MOVE_TO_IDX["<pad>"]] * (
            INFORMATION_MOVES_PER_MON - len(encoded_moves)
        )
        move_indices.extend(encoded_moves)
        scalars.extend(
            [
                1.0,
                float(belief.appeared),
                float(belief.active),
                belief.hp_fraction,
                float(belief.fainted),
                float(belief.known_item is not None),
                float(belief.known_ability is not None),
                len(belief.revealed_moves) / INFORMATION_MOVES_PER_MON,
                _distribution_uncertainty(belief.move_probabilities),
                _distribution_uncertainty(belief.item_probabilities),
                _distribution_uncertainty(belief.ability_probabilities),
                belief.spread_entropy,
                belief.speed_min / _NORMALIZED_STAT_CEILING,
                belief.speed_max / _NORMALIZED_STAT_CEILING,
                belief.speed_mean / _NORMALIZED_STAT_CEILING,
            ]
        )
        for index in range(MAX_SPREAD_HYPOTHESES):
            if index < len(belief.hypotheses):
                hypothesis = belief.hypotheses[index]
                stats = hypothesis.stats(belief.species_id)
                scalars.extend(
                    [
                        hypothesis.probability,
                        *[
                            float(stats[stat]) / _NORMALIZED_STAT_CEILING
                            for stat in STAT_IDS
                        ],
                    ]
                )
            else:
                scalars.extend([0.0] * 7)

    missing = INFORMATION_TEAM_SLOTS - len(beliefs)
    species_indices.extend([SPECIES_TO_IDX["<pad>"]] * missing)
    item_indices.extend([ITEM_TO_IDX["<unk>"]] * missing)
    ability_indices.extend([ABILITY_TO_IDX["<unk>"]] * missing)
    move_indices.extend(
        [MOVE_TO_IDX["<pad>"]] * (missing * INFORMATION_MOVES_PER_MON)
    )
    scalars.extend([0.0] * (missing * OPP_SCALARS_PER_MON))
    return species_indices, item_indices, ability_indices, move_indices, scalars


def encode_information_context(
    battle, memory: BattleMemory, config: PolicyConfig | None = None
) -> InformationFeatures:
    """Encode facts available to a real player, never private simulator truth."""

    config = config or PolicyConfig()
    own_species, own_items, own_abilities, own_moves, own_scalars = _own_information(battle)
    opp_species, opp_items, opp_abilities, opp_moves, opp_scalars = _opponent_information(
        battle, memory, config
    )
    indices = np.asarray(
        [
            *own_species,
            *own_items,
            *own_abilities,
            *own_moves,
            *opp_species,
            *opp_items,
            *opp_abilities,
            *opp_moves,
        ],
        dtype=np.int64,
    )
    scalars = np.asarray([*own_scalars, *opp_scalars], dtype=np.float32)
    assert indices.shape == (INFORMATION_INDEX_DIM,)
    assert scalars.shape == (INFORMATION_SCALAR_DIM,)
    return InformationFeatures(indices=indices, scalars=scalars)


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


def _active_beliefs(battle, beliefs: list[PokemonBelief]) -> dict[int, PokemonBelief]:
    by_species = {belief.species_id: belief for belief in beliefs}
    result: dict[int, PokemonBelief] = {}
    for slot, mon in enumerate((getattr(battle, "opponent_active_pokemon", None) or [])[:2]):
        if mon is not None:
            belief = by_species.get(to_id(getattr(mon, "species", None)))
            if belief is not None:
                result[slot] = belief
    return result


def _hypothesis_state(belief: PokemonBelief, hypothesis) -> PokemonState:
    state = PokemonState(
        species_id=belief.species_id,
        sp_spread=hypothesis.sp,
        nature=hypothesis.nature,
        item=belief.known_item,
        ability=belief.known_ability,
    )
    state.current_hp = round(state.max_hp() * belief.hp_fraction)
    return state


def _ko_roll_probability(result, target_hp: int) -> float:
    rolls = result.breakdown.get("rolls") or []
    return sum(float(damage >= target_hp) for damage in rolls) / len(rolls) if rolls else 0.0


def _clip_percent(value: float, ceiling: float = 200.0) -> float:
    return float(np.clip(value / ceiling, 0.0, 1.0))


def _move_tactical_features(
    single,
    actor_slot: int,
    ctx,
    active_beliefs: dict[int, PokemonBelief],
) -> tuple[list[float], dict[str, object]]:
    move_id = to_id(single.order.id)
    move_data = load_moves().get(move_id) or {}
    attacker = _attacker_state_for(single, actor_slot, ctx)
    targets = _resolve_targets(move_data, actor_slot, single.move_target, ctx)
    opponent_targets = [slot for slot, ally in targets if not ally]
    ally_targets = [slot for slot, ally in targets if ally]
    incoming = max(
        ctx.threat_on_us[actor_slot].percent,
        ctx.double_target_threat[actor_slot],
    )
    actor_state = ctx.our_states[actor_slot]
    actor_hp = (
        actor_state.hp_or_max() / actor_state.max_hp() if actor_state is not None else 0.0
    )
    accuracy = move_data.get("accuracy", 100)
    accuracy_value = 1.0 if accuracy is True else float(accuracy or 0.0) / 100.0
    priority = int(move_data.get("priority", 0) or 0)
    kind = utility_kind(move_id)
    is_status = move_data.get("category") == "Status"

    expected_values: list[float] = []
    minimum_values: list[float] = []
    maximum_values: list[float] = []
    ko_probability = 0.0
    move_first_probability = 0.0
    unsupported_probability = 0.0
    target_hp_fraction = 0.0
    target_slots: set[int] = set()
    for target_slot in opponent_targets:
        belief = active_beliefs.get(target_slot)
        if belief is None:
            unsupported_probability += 1.0
            continue
        target_slots.add(target_slot)
        target_hp_fraction += belief.hp_fraction
        target_expected = 0.0
        target_min = float("inf")
        target_max = 0.0
        target_ko = 0.0
        target_first = 0.0
        target_unsupported = 0.0
        for hypothesis in belief.hypotheses:
            defender = _hypothesis_state(belief, hypothesis)
            field = ctx.field_state(
                defender_is_ours=False,
                num_targets=max(1, len(targets)),
            )
            result = damage_range(attacker, defender, move_id, field)
            probability = hypothesis.probability
            if not result.breakdown.get("move_supported", False):
                target_unsupported += probability
                continue
            target_expected += probability * result.expected_percent
            target_min = min(target_min, result.min_percent)
            target_max = max(target_max, result.max_percent)
            target_ko += probability * _ko_roll_probability(result, defender.hp_or_max())
            opponent_speed = field_effective_speed(
                defender,
                weather=ctx.weather,
                tailwind=SideCondition.TAILWIND
                in (getattr(ctx.battle, "opponent_side_conditions", {}) or {}),
            )
            target_first += probability * float(
                resolves_before(
                    priority,
                    ctx.our_speed[actor_slot],
                    0,
                    opponent_speed,
                    ctx.trick_room,
                )
            )
        expected_values.append(target_expected)
        minimum_values.append(0.0 if target_min == float("inf") else target_min)
        maximum_values.append(target_max)
        ko_probability += target_ko
        move_first_probability += target_first
        unsupported_probability += target_unsupported

    target_count = max(1, len(opponent_targets))
    expected = sum(expected_values)
    minimum = sum(minimum_values)
    maximum = sum(maximum_values)
    ko_probability /= target_count
    move_first_probability = (
        move_first_probability / target_count if opponent_targets else 1.0
    )
    unsupported_probability /= target_count
    target_hp_fraction = (
        target_hp_fraction / target_count if opponent_targets else 0.0
    )

    ally_damage = 0.0
    for target_slot in ally_targets:
        defender = ctx.our_states[target_slot]
        if defender is not None:
            result = damage_range(
                attacker,
                defender,
                move_id,
                ctx.field_state(defender_is_ours=True, num_targets=max(1, len(targets))),
            )
            ally_damage += result.expected_percent

    vector = [
        _clip_percent(expected),
        _clip_percent(minimum),
        _clip_percent(maximum),
        float(np.clip(ko_probability, 0.0, 1.0)),
        float(np.clip(move_first_probability, 0.0, 1.0)),
        _clip_percent(incoming),
        actor_hp,
        target_hp_fraction,
        accuracy_value,
        float(np.clip(priority / 7.0, -1.0, 1.0)),
        float(kind == "protect"),
        0.0,
        float(kind == "speed_control"),
        float(kind in {"redirection", "action_denial", "wide_defense", "burn"}),
        float(kind in {"setup", "recovery"}),
        float(np.clip(unsupported_probability, 0.0, 1.0)),
    ]
    return vector, {
        "expected": expected,
        "minimum": minimum,
        "maximum": maximum,
        "incoming": incoming,
        "targets": target_slots,
        "ally_damage": ally_damage,
        "kind": kind,
        "is_attack": not is_status,
    }


def _switch_tactical_features(
    incoming_mon,
    actor_slot: int,
    ctx,
    active_beliefs: dict[int, PokemonBelief],
) -> tuple[list[float], dict[str, object]]:
    incoming = _our_pokemon_state(incoming_mon)
    outgoing_values: list[float] = []
    incoming_values: list[float] = []
    move_first_values: list[float] = []
    for target_slot, belief in active_beliefs.items():
        for hypothesis in belief.hypotheses:
            opponent = _hypothesis_state(belief, hypothesis)
            outgoing_best = 0.0
            for move_id in (getattr(incoming_mon, "moves", None) or {}):
                result = damage_range(
                    incoming,
                    opponent,
                    move_id,
                    ctx.field_state(defender_is_ours=False, num_targets=1),
                )
                if result.breakdown.get("move_supported", False):
                    outgoing_best = max(outgoing_best, result.expected_percent)
            outgoing_values.append(hypothesis.probability * outgoing_best)

            incoming_best = 0.0
            for move_id, move_probability in belief.move_probabilities.items():
                result = damage_range(
                    opponent,
                    incoming,
                    move_id,
                    ctx.field_state(defender_is_ours=True, num_targets=1),
                )
                if result.breakdown.get("move_supported", False):
                    incoming_best = max(
                        incoming_best, move_probability * result.expected_percent
                    )
            incoming_values.append(hypothesis.probability * incoming_best)
            opponent_speed = field_effective_speed(
                opponent,
                weather=ctx.weather,
                tailwind=SideCondition.TAILWIND
                in (getattr(ctx.battle, "opponent_side_conditions", {}) or {}),
            )
            move_first_values.append(
                hypothesis.probability
                * float(
                    resolves_before(
                        0,
                        field_effective_speed(incoming, weather=ctx.weather),
                        0,
                        opponent_speed,
                        ctx.trick_room,
                    )
                )
            )
    expected = sum(outgoing_values)
    incoming_threat = sum(incoming_values)
    count = max(1, len(active_beliefs))
    expected /= count
    incoming_threat /= count
    move_first = sum(move_first_values) / count
    hp_fraction = incoming.hp_or_max() / incoming.max_hp()
    vector = [
        _clip_percent(expected),
        _clip_percent(min(outgoing_values, default=0.0)),
        _clip_percent(max(outgoing_values, default=0.0)),
        0.0,
        float(np.clip(move_first, 0.0, 1.0)),
        _clip_percent(incoming_threat),
        hp_fraction,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    return vector, {
        "expected": expected,
        "minimum": min(outgoing_values, default=0.0),
        "maximum": max(outgoing_values, default=0.0),
        "incoming": incoming_threat,
        "targets": set(),
        "ally_damage": 0.0,
        "kind": "switch",
        "is_attack": False,
    }


def _single_tactical_features(single, actor_slot, ctx, active_beliefs):
    if single is None:
        return [0.0] * 16, {
            "expected": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "incoming": 0.0,
            "targets": set(),
            "ally_damage": 0.0,
            "kind": "pass",
            "is_attack": False,
        }
    if isinstance(single.order, Move):
        return _move_tactical_features(single, actor_slot, ctx, active_beliefs)
    if isinstance(single.order, Pokemon):
        return _switch_tactical_features(single.order, actor_slot, ctx, active_beliefs)
    return _single_tactical_features(None, actor_slot, ctx, active_beliefs)


def encode_candidate_tactical_features(
    battle,
    memory: BattleMemory,
    orders: Sequence[DoubleBattleOrder],
    config: PolicyConfig | None = None,
) -> np.ndarray:
    """First-principles facts for every legal joint order, integrated over beliefs."""

    config = config or PolicyConfig()
    ctx = build_context(battle, config)
    beliefs = build_opponent_beliefs(battle, memory, config)
    active_beliefs = _active_beliefs(battle, beliefs)
    rows: list[list[float]] = []
    for order in orders:
        first, first_meta = _single_tactical_features(
            order.first_order, 0, ctx, active_beliefs
        )
        second, second_meta = _single_tactical_features(
            order.second_order, 1, ctx, active_beliefs
        )
        targets = set(first_meta["targets"]) | set(second_meta["targets"])
        shared_targets = set(first_meta["targets"]) & set(second_meta["targets"])
        both_attack = bool(first_meta["is_attack"] and second_meta["is_attack"])
        joint = [
            _clip_percent(float(first_meta["expected"]) + float(second_meta["expected"])),
            len(targets) / 2.0,
            float(bool(shared_targets)),
            _clip_percent(
                float(first_meta["ally_damage"]) + float(second_meta["ally_damage"]),
                100.0,
            ),
            _clip_percent(float(first_meta["minimum"]) + float(second_meta["minimum"])),
            _clip_percent(float(first_meta["maximum"]) + float(second_meta["maximum"])),
            _clip_percent(max(float(first_meta["incoming"]), float(second_meta["incoming"]))),
            float(
                (first_meta["kind"] == "protect" and second_meta["is_attack"])
                or (second_meta["kind"] == "protect" and first_meta["is_attack"])
            ),
            float(
                (not first_meta["is_attack"] and second_meta["is_attack"])
                or (not second_meta["is_attack"] and first_meta["is_attack"])
            ),
            float(both_attack),
        ]
        rows.append([*first, *second, *joint])
    return np.asarray(rows, dtype=np.float32).reshape(len(orders), TACTICAL_FEATURE_DIM)


def encode_candidates(
    orders: Sequence[DoubleBattleOrder],
    *,
    battle=None,
    memory: BattleMemory | None = None,
    config: PolicyConfig | None = None,
) -> CandidateFeatures:
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
    tactical = (
        encode_candidate_tactical_features(battle, memory, orders, config)
        if battle is not None and memory is not None
        else np.zeros((count, TACTICAL_FEATURE_DIM), dtype=np.float32)
    )
    return CandidateFeatures(
        move_indices=np.asarray(move_rows, dtype=np.int64).reshape(count, NUM_ORDER_SLOTS),
        target_indices=np.asarray(target_rows, dtype=np.int64).reshape(count, NUM_ORDER_SLOTS),
        switch_species_indices=np.asarray(species_rows, dtype=np.int64).reshape(
            count, NUM_ORDER_SLOTS
        ),
        flags=np.asarray(flag_rows, dtype=np.float32).reshape(
            count, NUM_ORDER_SLOTS, NUM_ACTION_FLAGS
        ),
        tactical=tactical,
    )


def pad_candidate_tactical_features(candidates: Sequence[CandidateFeatures]) -> np.ndarray:
    """Pad candidate tactical rows to the same width as ``pad_candidate_features``."""

    if not candidates:
        raise ValueError("candidate batch must be non-empty")
    width = max(len(entry) for entry in candidates)
    tactical = np.zeros((len(candidates), width, TACTICAL_FEATURE_DIM), dtype=np.float32)
    for row, entry in enumerate(candidates):
        if entry.tactical is not None:
            tactical[row, : len(entry)] = entry.tactical
    return tactical


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
