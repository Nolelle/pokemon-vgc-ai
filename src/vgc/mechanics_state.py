"""Complete observable battle-state snapshot for the Champions decision engine.

`vgc.damage.PokemonState` and `FieldState` are intentionally small inputs to one damage
calculation. They are not a full battle state. This module is the canonical snapshot for
mechanics-aware decisions: it preserves every piece of public state exposed by poke-env,
including counters and generic effects that we do not understand yet.

Unknown opponent information stays unknown. A complete state representation must not
cheat by filling private moves, spreads, or abilities from simulator-only data.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, Mapping

from vgc.damage import to_id
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


@dataclass(frozen=True)
class EffectSnapshot:
    """One status, volatile, side, field, weather, terrain, or slot effect."""

    id: str
    turns: int | None = None
    raw_value: str | int | float | bool | None = None


def _effect_snapshots(values: Any) -> tuple[EffectSnapshot, ...]:
    if values is None:
        return ()
    entries: Iterable[tuple[Any, Any]]
    if isinstance(values, Mapping):
        entries = values.items()
    else:
        entries = ((value, None) for value in values)
    result: list[EffectSnapshot] = []
    for effect, raw in entries:
        effect_id = _effect_id(effect)
        if not effect_id:
            continue
        serializable_raw = raw if isinstance(raw, (str, int, float, bool)) else None
        result.append(
            EffectSnapshot(
                id=effect_id,
                turns=_optional_int(raw),
                raw_value=serializable_raw,
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


def _move_snapshot(move_id: Any, move: Any) -> MoveSnapshot:
    entry = getattr(move, "entry", None) or {}
    disabled = bool(getattr(move, "disabled", False) or entry.get("disabled", False))
    reason = entry.get("disabledSource") or entry.get("disabled_reason")
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
    nature: str | None
    boosts: tuple[tuple[str, int], ...]
    status: str | None
    status_counter: int
    effects: tuple[EffectSnapshot, ...]
    item_id: str | None
    item_known: bool
    ability_id: str | None
    base_ability_id: str | None
    ability_known: bool
    moves: tuple[MoveSnapshot, ...]
    last_move_id: str | None
    preparing_move_id: str | None
    preparing_target: str | int | None
    must_recharge: bool
    protect_counter: int
    first_turn: bool
    transformed: bool
    mega_evolved: bool
    terastallized: bool
    tera_type: str | None
    weight: float | None


def snapshot_pokemon(pokemon: Any, *, opponent: bool) -> PokemonMechanicsState:
    moves = getattr(pokemon, "moves", None) or {}
    move_entries = moves.items() if isinstance(moves, Mapping) else ((move.id, move) for move in moves)
    item = normalize_item(getattr(pokemon, "item", None))
    ability = to_id(getattr(pokemon, "ability", None)) or None
    base_ability = to_id(getattr(pokemon, "base_ability", None)) or None
    species_id = to_id(getattr(pokemon, "species", None))
    base_species_id = to_id(getattr(pokemon, "base_species", None)) or None
    raw_stats = getattr(pokemon, "stats", None) or {}
    raw_boosts = getattr(pokemon, "boosts", None) or {}
    raw_stat_points = getattr(pokemon, "evs", None)
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
    return PokemonMechanicsState(
        species_id=species_id,
        base_species_id=base_species_id,
        name=getattr(pokemon, "name", None),
        level=_optional_int(getattr(pokemon, "level", None)),
        gender=getattr(pokemon, "gender", None),
        types=tuple(str(value) for value in (getattr(pokemon, "types", None) or ())),
        base_types=tuple(str(value) for value in (getattr(pokemon, "base_types", None) or ())),
        current_hp=_optional_int(getattr(pokemon, "current_hp", None)),
        max_hp=_optional_int(getattr(pokemon, "max_hp", None)),
        fainted=bool(getattr(pokemon, "fainted", False)),
        active=bool(getattr(pokemon, "active", False)),
        revealed=bool(getattr(pokemon, "revealed", not opponent)),
        selected_in_preview=bool(getattr(pokemon, "selected_in_teampreview", False)),
        stats=tuple(
            sorted((str(key), _optional_int(value)) for key, value in raw_stats.items())
        ),
        stat_points=stat_points,
        nature=to_id(getattr(pokemon, "nature", None)) or None,
        boosts=tuple((boost, int(raw_boosts.get(boost, 0))) for boost in BOOST_IDS),
        status=normalize_status(getattr(pokemon, "status", None)),
        status_counter=int(getattr(pokemon, "status_counter", 0) or 0),
        effects=_effect_snapshots(getattr(pokemon, "effects", None)),
        item_id=item,
        item_known=not opponent or item is not None,
        ability_id=ability,
        base_ability_id=base_ability,
        ability_known=not opponent or ability is not None,
        moves=tuple(sorted((_move_snapshot(key, move) for key, move in move_entries), key=lambda move: move.id)),
        last_move_id=to_id(getattr(getattr(pokemon, "last_move", None), "id", None)) or None,
        preparing_move_id=to_id(
            getattr(getattr(pokemon, "preparing_move", None), "id", None)
            or getattr(pokemon, "preparing_move", None)
        )
        or None,
        preparing_target=preparing_target,
        must_recharge=bool(getattr(pokemon, "must_recharge", False)),
        protect_counter=int(getattr(pokemon, "protect_counter", 0) or 0),
        first_turn=bool(getattr(pokemon, "first_turn", False)),
        transformed=bool(getattr(pokemon, "transformed", False)),
        mega_evolved="mega" in species_id and species_id != (base_species_id or species_id),
        terastallized=bool(getattr(pokemon, "is_terastallized", False)),
        tera_type=to_id(getattr(pokemon, "tera_type", None)) or None,
        weight=float(getattr(pokemon, "weight", 0.0))
        if getattr(pokemon, "weight", None) is not None
        else None,
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


def _side_snapshot(battle: Any, *, opponent: bool) -> SideMechanicsState:
    prefix = "opponent_" if opponent else ""
    team = getattr(battle, f"{prefix}team", None) or {}
    pokemon_values = team.values() if isinstance(team, Mapping) else team
    active = list(getattr(battle, f"{prefix}active_pokemon", None) or ())
    while len(active) < 2:
        active.append(None)
    return SideMechanicsState(
        pokemon=tuple(snapshot_pokemon(mon, opponent=opponent) for mon in pokemon_values),
        active_species=tuple(
            to_id(getattr(mon, "species", None)) if mon is not None else None for mon in active
        ),
        side_conditions=_effect_snapshots(getattr(battle, f"{prefix}side_conditions", None)),
        force_switch=_bool_tuple(getattr(battle, f"{prefix}force_switch", False)),
        trapped=_bool_tuple(getattr(battle, f"{prefix}trapped", False)),
        maybe_trapped=_bool_tuple(getattr(battle, f"{prefix}maybe_trapped", False)),
        can_mega_evolve=_bool_tuple(getattr(battle, f"{prefix}can_mega_evolve", False)),
        used_mega_evolution=bool(getattr(battle, f"{prefix}used_mega_evolve", False)),
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
    team_preview: bool
    waiting: bool
    finished: bool
    won: bool
    lost: bool
    fields: tuple[EffectSnapshot, ...]
    weather: tuple[EffectSnapshot, ...]
    available_moves: tuple[tuple[str, ...], ...]
    available_switches: tuple[tuple[str, ...], ...]
    valid_order_count: int | None
    our_side: SideMechanicsState
    opponent_side: SideMechanicsState


def snapshot_battle(battle: Any) -> BattleMechanicsState:
    """Copy every observable mechanics field from a poke-env battle object."""

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
    return BattleMechanicsState(
        format_id=to_id(getattr(battle, "format", None)),
        generation=int(getattr(battle, "gen", 9) or 9),
        game_type="doubles",
        turn=int(getattr(battle, "turn", 0) or 0),
        team_preview=bool(getattr(battle, "teampreview", False)),
        waiting=bool(getattr(battle, "wait", False)),
        finished=bool(getattr(battle, "finished", False)),
        won=bool(getattr(battle, "won", False)),
        lost=bool(getattr(battle, "lost", False)),
        fields=_effect_snapshots(getattr(battle, "fields", None)),
        weather=_effect_snapshots(getattr(battle, "weather", None)),
        available_moves=tuple(
            tuple(to_id(getattr(move, "id", move)) for move in slot_moves)
            for slot_moves in raw_available_moves
        ),
        available_switches=tuple(
            tuple(to_id(getattr(mon, "species", mon)) for mon in slot_switches)
            for slot_switches in raw_available_switches
        ),
        valid_order_count=len(raw_valid_orders) if raw_valid_orders is not None else None,
        our_side=_side_snapshot(battle, opponent=False),
        opponent_side=_side_snapshot(battle, opponent=True),
    )
