"""Complete observable battle-state snapshot for the Champions decision engine.

`vgc.damage.PokemonState` and `FieldState` are intentionally small inputs to one damage
calculation. They are not a full battle state. This module is the canonical snapshot for
mechanics-aware decisions: it preserves every piece of public state exposed by poke-env,
including counters and generic effects that we do not understand yet.

Unknown opponent information stays unknown. A complete state representation must not
cheat by filling private moves, spreads, or abilities from simulator-only data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

from vgc.condition_clock import elapsed_ticks
from vgc.damage import to_id
from vgc.data import load_species
from vgc.poke_env_compat import fix_enabled
from vgc.sets import normalize_item, normalize_status

BOOST_IDS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STAT_POINT_IDS = ("hp", "atk", "def", "spa", "spd", "spe")


def _effect_id(value: Any) -> str:
    """Stable id for poke-env enums, strings, and effect-like objects."""

    if value is None:
        return ""
    for attribute in ("id", "name", "value"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str) and candidate:
            return to_id(candidate)
    return to_id(str(value).split(".")[-1])


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    return int(value)


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    """Read a public property that may be unavailable during an early request."""

    try:
        return getattr(value, name, default)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return default


@dataclass(frozen=True)
class EffectSnapshot:
    """One status, volatile, side, field, weather, terrain, or slot effect."""

    id: str
    turns: int | None = None
    raw_value: str | int | float | bool | None = None
    counter_kind: str = "unknown"


_LAYERED_SIDE_CONDITIONS = frozenset({"spikes", "toxicspikes", "stealthrock", "stickyweb"})

# poke-env's `Pokemon.effects` mixes real Showdown volatiles with its own bookkeeping. Showdown
# knows these under other ids, or not at all; `vgc/rl/mechanics_oracle` and the exact mirror
# write each id straight into a Showdown Pokemon's `volatiles`, so an id Showdown does not
# use becomes a bogus condition there.
_EFFECT_ALIASES = {
    **{
        trap: "partiallytrapped"
        for trap in (
            "bind", "clamp", "firespin", "infestation", "magmastorm", "sandtomb", "snaptrap",
            "thundercage", "whirlpool", "wrap",
        )
    },
    **{f"stockpile{n}": "stockpile" for n in (1, 2, 3)},
    **{f"perish{n}": "perishsong" for n in (0, 1, 2, 3)},
}  # fmt: skip
# Not volatiles in Showdown: a type change is recorded in the Pokemon's `types`, which the
# snapshot already carries.
# (Future Sight lives in a slot condition; Supreme Overlord's count lives in the ability state.)
_NON_VOLATILE_EFFECTS = frozenset(
    {"typechange", "typeadd", "futuresight", "doomdesire", "fallen", *(f"fallen{n}" for n in range(1, 6))}
)


def _effect_snapshots(
    values: Any,
    *,
    counter_kind: str,
    battle: Any = None,
    clock_kind: str | None = None,
    clock_side: str | None = None,
    canonical_volatiles: bool = False,
) -> tuple[EffectSnapshot, ...]:
    if values is None:
        return ()
    entries: Iterable[tuple[Any, Any]]
    if isinstance(values, Mapping):
        entries = values.items()
    else:
        entries = ((value, None) for value in values)
    result: list[EffectSnapshot] = []
    seen: set[str] = set()
    for effect, raw in entries:
        effect_id = _effect_id(effect)
        if not effect_id:
            continue
        if canonical_volatiles and fix_enabled("snapshot_effects"):
            if effect_id in _NON_VOLATILE_EFFECTS:
                continue
            effect_id = _EFFECT_ALIASES.get(effect_id, effect_id)
            if effect_id in seen:
                continue
            seen.add(effect_id)
        serializable_raw = raw if isinstance(raw, (str, int, float, bool)) else None
        turns = _optional_int(raw)
        layered = (
            counter_kind == "side_start_turn"
            and effect_id in _LAYERED_SIDE_CONDITIONS
            and fix_enabled("snapshot_layers")
        )
        if clock_kind is not None and battle is not None and not layered:
            # poke-env's start turn is wrong for weather (restamped every upkeep) and
            # for switch-in setters; report the start implied by Showdown's real
            # duration ticks instead (vgc.condition_clock).
            ticks = elapsed_ticks(battle, clock_kind, effect_id, clock_side)
            if ticks is not None:
                turns = int(getattr(battle, "turn", 0) or 0) - ticks
                serializable_raw = turns
        result.append(
            EffectSnapshot(
                id=effect_id,
                turns=turns,
                raw_value=serializable_raw,
                counter_kind=(
                    "layers"
                    if counter_kind == "side_start_turn" and effect_id in _LAYERED_SIDE_CONDITIONS
                    else counter_kind
                ),
            )
        )
    return tuple(sorted(result, key=lambda effect: effect.id))


@dataclass(frozen=True)
class MoveSnapshot:
    id: str
    current_pp: int | None
    max_pp: int | None
    disabled: bool
    disabled_reason: str | None
    last_used: bool


def _move_snapshot(move_id: Any, move: Any, *, unavailable: bool = False) -> MoveSnapshot:
    entry = getattr(move, "entry", None) or {}
    disabled = bool(
        unavailable or getattr(move, "disabled", False) or entry.get("disabled", False)
    )
    reason = entry.get("disabledSource") or entry.get("disabled_reason")
    if unavailable and not reason:
        # The request did not offer this known move (choice lock, Encore, Disable,
        # PP exhaustion, recharge...). The reason is deliberately generic: the
        # request is the authority that it is unavailable, not why.
        reason = "request"
    return MoveSnapshot(
        id=to_id(getattr(move, "id", None) or move_id),
        current_pp=_optional_int(getattr(move, "current_pp", None)),
        max_pp=_optional_int(getattr(move, "max_pp", None)),
        disabled=disabled,
        disabled_reason=str(reason) if reason else None,
        last_used=bool(getattr(move, "is_last_used", False)),
    )


@dataclass(frozen=True)
class PokemonMechanicsState:
    """All observable per-Pokemon state needed by Showdown mechanics."""

    species_id: str
    base_species_id: str | None
    name: str | None
    level: int | None
    gender: str | None
    types: tuple[str, ...]
    base_types: tuple[str, ...]
    current_hp: int | None
    max_hp: int | None
    fainted: bool
    active: bool
    revealed: bool
    selected_in_preview: bool
    stats: tuple[tuple[str, int | None], ...]
    stat_points: tuple[tuple[str, int | None], ...] | None
    individual_values: tuple[tuple[str, int | None], ...] | None
    nature: str | None
    boosts: tuple[tuple[str, int], ...]
    status: str | None
    status_counter: int
    effects: tuple[EffectSnapshot, ...]
    item_id: str | None
    # ``known`` / ``consumed`` / ``unknown`` for opponents; ``none`` is only reachable
    # on our own side (we always know our registered team). Opponent ``item is None``
    # without a recorded original item is never ``none`` -- holding nothing is not
    # publicly observable except via Open Team Sheets, which poke-env does not distinguish
    # from ``-enditem`` consumption without extra memory.
    item_state: str
    item_known: bool
    ability_id: str | None
    base_ability_id: str | None
    temporary_ability_id: str | None
    forme_change_ability_id: str | None
    ability_known: bool
    base_move_ids: tuple[str, ...]
    moves: tuple[MoveSnapshot, ...]
    last_move_id: str | None
    mimic_move_id: str | None
    preparing_move_id: str | None
    preparing_target: str | int | None
    preparing: bool
    must_recharge: bool
    protect_counter: int
    first_turn: bool
    transformed: bool
    mega_evolved: bool
    terastallized: bool
    tera_type: str | None
    weight: float | None
    # Public-protocol fact for Unburden: the item went away DURING the current stint (since the
    # last switch-in). ``None`` = unknown (no battle memory), the consumer falls back to a guess.
    item_lost_this_stint: bool | None = None


def _mega_forme_id(species_id: str, forme_change_ability_id: str | None) -> str | None:
    """Mega forme poke-env applied without renaming ``species``.

    ``Pokemon.mega_evolve`` / ``forme_change`` load mega dex data with
    ``store_species=False``, so ``species`` stays the base id. The mega ability is
    stashed on ``forme_change_ability``; that is the public signal an opponent Mega
    actually happened.
    """

    if species_id and "mega" in species_id:
        return species_id
    if not species_id or not forme_change_ability_id:
        return None
    species = load_species().get(species_id)
    if not species:
        return None
    forme_ability = forme_change_ability_id
    candidates = [to_id(name) for name in species.get("otherFormes") or ()]
    if fix_enabled("mega_changes_from"):
        # Floette-Mega is listed under Floette but changes from Floette-Eternal, which has no
        # `otherFormes` of its own.
        candidates += [
            forme_id
            for forme_id, forme in load_species().items()
            if forme.get("isMega") and forme.get("changesFrom") == species.get("name")
        ]
    for forme_id in candidates:
        if not forme_id:
            continue
        forme = load_species().get(forme_id)
        if not forme or not forme.get("isMega"):
            continue
        mega_ability = to_id((forme.get("abilities") or {}).get("0"))
        if mega_ability == forme_ability:
            return forme_id
    return None


def _revealed_items_for(battle: Any) -> dict[str, str] | None:
    memory = getattr(battle, "_vgc_battle_memory", None)
    if memory is None:
        return None
    return memory.opponent_items


def _resolve_item_state(
    pokemon: Any,
    *,
    opponent: bool,
    revealed_items: Mapping[str, str] | None,
) -> tuple[str, str | None]:
    """Classify item knowledge: known, consumed, none (own side only), or unknown."""

    raw_item = getattr(pokemon, "item", None)
    item_id = normalize_item(raw_item)
    species_id = to_id(getattr(pokemon, "species", None)) or ""

    if not opponent:
        return ("known", item_id) if item_id is not None else ("none", None)

    if raw_item is None and fix_enabled("snapshot_foe_item"):
        # A hidden foe item is poke-env's "unknown_item" sentinel; `None` is only reached
        # through `-enditem` (consumed, knocked off, Fling, Trick-away), which is public.
        # The old test below needed some OTHER item to have been revealed first, so a foe
        # whose only item was just consumed read as "unknown" and the exact mirror kept
        # its guessed item (a Focus Sash, a Sitrus Berry) alive.
        return "consumed", None
    if raw_item == "unknown_item" or item_id is None and not revealed_items:
        return "unknown", None
    if item_id is not None:
        return "known", item_id
    if revealed_items and species_id in revealed_items:
        return "consumed", None
    return "unknown", None


def _request_confirmed_preparing(
    *,
    opponent: bool,
    preparing: bool,
    preparing_move_id: str | None,
    available_move_ids: tuple[str, ...] | None,
) -> tuple[bool, str | None]:
    """Whether a two-turn lock is actually in force on this observation.

    poke-env can leave ``Pokemon.preparing`` set after a sun Solar Beam that skipped
    its charge turn, while ``available_moves`` still lists every move. The REQUEST is
    the public observation of a genuine lock: our active slot is preparing only when
    that slot's available moves are exactly the charging move. Opponent Pokemon have
    no request on this battle object, so their poke-env flag is kept as-is.
    """

    if opponent:
        return preparing, preparing_move_id if preparing else None
    if not preparing or not preparing_move_id or available_move_ids is None:
        return False, None
    offered = tuple(move_id for move_id in available_move_ids if move_id)
    if offered == (preparing_move_id,):
        return True, preparing_move_id
    return False, None


def _request_confirmed_must_recharge(
    *,
    opponent: bool,
    must_recharge: bool,
    available_move_ids: tuple[str, ...] | None,
) -> bool:
    """Whether a Hyper Beam-style recharge turn is actually in force.

    Same rule as `_request_confirmed_preparing`: poke-env can leave
    ``Pokemon.must_recharge`` set when Showdown did not require the recharge (seen on a
    Choice Scarf Heliolisk whose request still offered Hyper Beam). Materialised in the
    mirror, the stale flag made exact search choose ``recharge``, which reached the real
    battle as ``move 1`` -- a disabled Thunderbolt. Our active slot must recharge only
    when its request offers exactly ``recharge``; opponents keep the flag.
    """

    if opponent:
        return must_recharge
    if not must_recharge or available_move_ids is None:
        return False
    return tuple(move_id for move_id in available_move_ids if move_id) == ("recharge",)


def snapshot_pokemon(
    pokemon: Any,
    *,
    opponent: bool,
    revealed_items: Mapping[str, str] | None = None,
    available_move_ids: tuple[str, ...] | None = None,
    item_lost_this_stint: bool | None = None,
) -> PokemonMechanicsState:
    moves = getattr(pokemon, "moves", None) or {}
    move_entries = (
        moves.items() if isinstance(moves, Mapping) else ((move.id, move) for move in moves)
    )
    item_state, item = _resolve_item_state(
        pokemon, opponent=opponent, revealed_items=revealed_items
    )
    ability = to_id(getattr(pokemon, "ability", None)) or None
    base_ability = to_id(getattr(pokemon, "base_ability", None)) or None
    species_id = to_id(getattr(pokemon, "species", None)) or ""
    base_species_id = to_id(getattr(pokemon, "base_species", None)) or None
    forme_change_ability_id = to_id(getattr(pokemon, "forme_change_ability", None)) or None
    mega_forme_id = _mega_forme_id(species_id, forme_change_ability_id)
    if mega_forme_id:
        species_id = mega_forme_id
    raw_stats = getattr(pokemon, "stats", None) or {}
    raw_boosts = getattr(pokemon, "boosts", None) or {}
    raw_stat_points = getattr(pokemon, "evs", None)
    raw_individual_values = getattr(pokemon, "ivs", None)
    stat_points = (
        tuple(
            (stat_id, _optional_int(value))
            for stat_id, value in zip(STAT_POINT_IDS, raw_stat_points, strict=True)
        )
        if raw_stat_points is not None
        else None
    )
    preparing_target = getattr(pokemon, "preparing_target", None)
    if preparing_target is not None and not isinstance(preparing_target, (str, int)):
        preparing_target = str(preparing_target)
    preparing_move_id = (
        to_id(
            getattr(getattr(pokemon, "preparing_move", None), "id", None)
            or getattr(pokemon, "preparing_move", None)
        )
        or None
    )
    preparing, preparing_move_id = _request_confirmed_preparing(
        opponent=opponent,
        preparing=bool(getattr(pokemon, "preparing", False)),
        preparing_move_id=preparing_move_id,
        available_move_ids=available_move_ids,
    )
    if not preparing:
        preparing_target = None
    return PokemonMechanicsState(
        species_id=species_id,
        base_species_id=base_species_id,
        name=getattr(pokemon, "name", None),
        level=_optional_int(getattr(pokemon, "level", None)),
        gender=(
            _effect_id(getattr(pokemon, "gender", None)) or None
            if getattr(pokemon, "gender", None) is not None
            else None
        ),
        types=tuple(_effect_id(value) for value in (getattr(pokemon, "types", None) or ())),
        base_types=tuple(
            _effect_id(value) for value in (getattr(pokemon, "base_types", None) or ())
        ),
        current_hp=_optional_int(getattr(pokemon, "current_hp", None)),
        max_hp=_optional_int(getattr(pokemon, "max_hp", None)),
        fainted=bool(getattr(pokemon, "fainted", False)),
        active=bool(getattr(pokemon, "active", False)),
        revealed=bool(getattr(pokemon, "revealed", not opponent)),
        selected_in_preview=bool(getattr(pokemon, "selected_in_teampreview", False)),
        stats=tuple(sorted((str(key), _optional_int(value)) for key, value in raw_stats.items())),
        stat_points=stat_points,
        individual_values=(
            tuple(
                (stat_id, _optional_int(value))
                for stat_id, value in zip(STAT_POINT_IDS, raw_individual_values, strict=True)
            )
            if raw_individual_values is not None
            else None
        ),
        nature=to_id(getattr(pokemon, "nature", None)) or None,
        boosts=tuple((boost, int(raw_boosts.get(boost, 0))) for boost in BOOST_IDS),
        status=normalize_status(getattr(pokemon, "status", None)),
        status_counter=int(getattr(pokemon, "status_counter", 0) or 0),
        effects=_effect_snapshots(
            getattr(pokemon, "effects", None),
            counter_kind="elapsed_actions",
            canonical_volatiles=True,
        ),
        item_id=item,
        item_state=item_state,
        item_known=item_state != "unknown",
        ability_id=ability,
        base_ability_id=base_ability,
        temporary_ability_id=to_id(getattr(pokemon, "temporary_ability", None)) or None,
        forme_change_ability_id=forme_change_ability_id,
        ability_known=not opponent or ability is not None,
        base_move_ids=tuple(
            sorted(to_id(move) for move in (getattr(pokemon, "base_moves", None) or ()))
        ),
        moves=tuple(
            sorted(
                (
                    _move_snapshot(
                        key,
                        move,
                        unavailable=(
                            available_move_ids is not None
                            and to_id(key) not in set(available_move_ids)
                        ),
                    )
                    for key, move in move_entries
                ),
                key=lambda move: move.id,
            )
        ),
        last_move_id=to_id(getattr(getattr(pokemon, "last_move", None), "id", None)) or None,
        mimic_move_id=to_id(
            getattr(getattr(pokemon, "mimic_move", None), "id", None)
            or getattr(pokemon, "mimic_move", None)
        )
        or None,
        preparing_move_id=preparing_move_id,
        preparing_target=preparing_target,
        preparing=preparing,
        must_recharge=_request_confirmed_must_recharge(
            opponent=opponent,
            must_recharge=bool(getattr(pokemon, "must_recharge", False)),
            available_move_ids=available_move_ids,
        ),
        protect_counter=int(getattr(pokemon, "protect_counter", 0) or 0),
        first_turn=bool(getattr(pokemon, "first_turn", False)),
        transformed=bool(getattr(pokemon, "transformed", False)),
        mega_evolved=bool(mega_forme_id)
        or ("mega" in species_id and species_id != (base_species_id or species_id)),
        terastallized=bool(getattr(pokemon, "is_terastallized", False)),
        tera_type=to_id(getattr(pokemon, "tera_type", None)) or None,
        weight=float(getattr(pokemon, "weight", 0.0))
        if getattr(pokemon, "weight", None) is not None
        else None,
        item_lost_this_stint=item_lost_this_stint,
    )


@dataclass(frozen=True)
class SideMechanicsState:
    pokemon: tuple[PokemonMechanicsState, ...]
    active_species: tuple[str | None, ...]
    side_conditions: tuple[EffectSnapshot, ...]
    force_switch: tuple[bool | None, ...]
    trapped: tuple[bool | None, ...]
    maybe_trapped: tuple[bool | None, ...]
    can_mega_evolve: tuple[bool | None, ...]
    used_mega_evolution: bool
    can_dynamax: tuple[bool | None, ...]
    used_dynamax: bool
    can_tera: tuple[bool | None, ...]
    used_tera: bool
    can_z_move: tuple[bool | None, ...]
    used_z_move: bool


def _bool_tuple(value: Any, length: int = 2) -> tuple[bool | None, ...]:
    if value is None:
        return tuple(None for _ in range(length))
    if isinstance(value, (list, tuple)):
        return tuple(bool(entry) for entry in value)
    return tuple(bool(value) for _ in range(length))


def _active_index(active: list[Any], pokemon: Any) -> int | None:
    for index, candidate in enumerate(active):
        if candidate is pokemon:
            return index
    return None


def _slot_available_move_ids(
    *,
    opponent: bool,
    active: list[Any],
    pokemon: Any,
    available_move_ids: tuple[tuple[str, ...], ...],
) -> tuple[str, ...] | None:
    if opponent:
        return None
    slot = _active_index(active, pokemon)
    if slot is None or slot >= len(available_move_ids):
        return None
    return available_move_ids[slot]


def _side_role(battle: Any, *, opponent: bool) -> str | None:
    role = getattr(battle, "player_role", None)
    if role not in ("p1", "p2"):
        return None
    return ("p2" if role == "p1" else "p1") if opponent else role


def _side_snapshot(
    battle: Any,
    *,
    opponent: bool,
    available_move_ids: tuple[tuple[str, ...], ...] = (),
    opponent_mega_unknown: bool = False,
) -> SideMechanicsState:
    prefix = "opponent_" if opponent else ""
    team = getattr(battle, f"{prefix}team", None) or {}
    pokemon_values = team.values() if isinstance(team, Mapping) else team
    active = list(getattr(battle, f"{prefix}active_pokemon", None) or ())
    while len(active) < 2:
        active.append(None)
    revealed_items = _revealed_items_for(battle) if opponent else None
    memory = getattr(battle, "_vgc_battle_memory", None)
    memory_role = None
    if memory is not None:
        memory_role = memory.opponent_role if opponent else memory.our_role
    pokemon_snapshots = tuple(
        snapshot_pokemon(
            mon,
            opponent=opponent,
            revealed_items=revealed_items,
            item_lost_this_stint=(
                memory.item_lost_this_stint(memory_role, getattr(mon, "species", None))
                if memory is not None
                else None
            ),
            available_move_ids=_slot_available_move_ids(
                opponent=opponent,
                active=active,
                pokemon=mon,
                available_move_ids=available_move_ids,
            ),
        )
        for mon in pokemon_values
    )
    used_mega_evolution = bool(getattr(battle, f"{prefix}used_mega_evolve", False))
    can_mega_evolve = _bool_tuple(getattr(battle, f"{prefix}can_mega_evolve", False))
    if opponent and opponent_mega_unknown:
        # poke-env has no `opponent_can_mega_evolve`: the foe's request is private, so
        # "cannot Mega" would be a guess dressed as a fact. Until the foe has used its one
        # Mega (the -mega line, or a Pokemon already shown in Mega forme) the answer is
        # unknown; afterwards it is a definite no.
        used_mega_evolution = used_mega_evolution or any(
            mon.mega_evolved for mon in pokemon_snapshots
        )
        can_mega_evolve = _bool_tuple(False if used_mega_evolution else None)
    return SideMechanicsState(
        pokemon=pokemon_snapshots,
        active_species=tuple(
            to_id(getattr(mon, "species", None)) if mon is not None else None for mon in active
        ),
        side_conditions=_effect_snapshots(
            getattr(battle, f"{prefix}side_conditions", None),
            counter_kind="side_start_turn",
            battle=battle,
            clock_kind="side",
            clock_side=_side_role(battle, opponent=opponent),
        ),
        force_switch=_bool_tuple(getattr(battle, f"{prefix}force_switch", False)),
        trapped=_bool_tuple(getattr(battle, f"{prefix}trapped", False)),
        maybe_trapped=_bool_tuple(getattr(battle, f"{prefix}maybe_trapped", False)),
        can_mega_evolve=can_mega_evolve,
        used_mega_evolution=used_mega_evolution,
        can_dynamax=_bool_tuple(getattr(battle, f"{prefix}can_dynamax", False)),
        used_dynamax=bool(getattr(battle, f"{prefix}used_dynamax", False)),
        can_tera=_bool_tuple(getattr(battle, f"{prefix}can_tera", False)),
        used_tera=bool(getattr(battle, f"{prefix}used_tera", False)),
        can_z_move=_bool_tuple(getattr(battle, f"{prefix}can_z_move", False)),
        used_z_move=bool(getattr(battle, f"{prefix}used_z_move", False)),
    )


@dataclass(frozen=True)
class BattleMechanicsState:
    """Lossless public snapshot consumed by future exact mechanics handlers."""

    format_id: str
    generation: int
    game_type: str
    turn: int
    max_team_size: int | None
    team_size: int | None
    team_preview: bool
    commanding: bool
    reviving: bool
    waiting: bool
    finished: bool
    won: bool
    lost: bool
    fields: tuple[EffectSnapshot, ...]
    weather: tuple[EffectSnapshot, ...]
    available_moves: tuple[tuple[str, ...], ...]
    available_switches: tuple[tuple[str, ...], ...]
    valid_order_count: int | None
    last_request_json: str | None
    our_preview_species: tuple[str, ...]
    opponent_preview_species: tuple[str, ...]
    our_side: SideMechanicsState
    opponent_side: SideMechanicsState


def snapshot_battle(battle: Any, *, opponent_mega_unknown: bool = False) -> BattleMechanicsState:
    """Copy every observable mechanics field from a poke-env battle object.

    ``opponent_mega_unknown`` reports the opponent's ``can_mega_evolve`` as ``None`` (not
    observable) until it has used its Mega, instead of the poke-env default of False. Only
    the live mirror asks for it; the default keeps every stored/encoded snapshot unchanged.
    """

    try:
        raw_available_moves = getattr(battle, "available_moves", None) or ()
    except (AttributeError, RuntimeError, ValueError):
        raw_available_moves = ()
    try:
        raw_available_switches = getattr(battle, "available_switches", None) or ()
    except (AttributeError, RuntimeError, ValueError):
        raw_available_switches = ()
    try:
        raw_valid_orders = getattr(battle, "valid_orders", None)
    except (AttributeError, RuntimeError, ValueError):
        raw_valid_orders = None
    try:
        raw_last_request = getattr(battle, "last_request", None)
        last_request_json = (
            json.dumps(raw_last_request, sort_keys=True, default=str)
            if raw_last_request is not None
            else None
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        last_request_json = None
    available_moves = tuple(
        tuple(to_id(getattr(move, "id", move)) for move in slot_moves)
        for slot_moves in raw_available_moves
    )
    return BattleMechanicsState(
        format_id=to_id(getattr(battle, "format", None)),
        generation=int(getattr(battle, "gen", 9) or 9),
        game_type="doubles",
        turn=int(getattr(battle, "turn", 0) or 0),
        max_team_size=_optional_int(_safe_attr(battle, "max_team_size")),
        team_size=_optional_int(_safe_attr(battle, "team_size")),
        team_preview=bool(getattr(battle, "teampreview", False)),
        commanding=bool(getattr(battle, "commanding", False)),
        reviving=bool(getattr(battle, "reviving", False)),
        waiting=bool(getattr(battle, "wait", False)),
        finished=bool(getattr(battle, "finished", False)),
        won=bool(getattr(battle, "won", False)),
        lost=bool(getattr(battle, "lost", False)),
        fields=_effect_snapshots(
            getattr(battle, "fields", None),
            counter_kind="start_turn",
            battle=battle,
            clock_kind="field",
        ),
        weather=_effect_snapshots(
            getattr(battle, "weather", None),
            counter_kind="start_turn",
            battle=battle,
            clock_kind="weather",
        ),
        available_moves=available_moves,
        available_switches=tuple(
            tuple(to_id(getattr(mon, "species", mon)) for mon in slot_switches)
            for slot_switches in raw_available_switches
        ),
        valid_order_count=len(raw_valid_orders) if raw_valid_orders is not None else None,
        last_request_json=last_request_json,
        our_preview_species=tuple(
            to_id(getattr(mon, "species", mon))
            for mon in (getattr(battle, "teampreview_team", None) or ())
        ),
        opponent_preview_species=tuple(
            to_id(getattr(mon, "species", mon))
            for mon in (getattr(battle, "teampreview_opponent_team", None) or ())
        ),
        our_side=_side_snapshot(
            battle, opponent=False, available_move_ids=available_moves
        ),
        opponent_side=_side_snapshot(
            battle, opponent=True, opponent_mega_unknown=opponent_mega_unknown
        ),
    )
