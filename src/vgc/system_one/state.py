"""Public-board state serialization for System One (no private simulator roots)."""

from __future__ import annotations

from poke_env.battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather

from vgc.battle_memory import BattleMemory
from vgc.damage import to_id
from vgc.models import PolicyConfig


_WEATHER_TO_STR = {
    Weather.SUNNYDAY: "sun",
    Weather.DESOLATELAND: "sun",
    Weather.RAINDANCE: "rain",
    Weather.PRIMORDIALSEA: "rain",
    Weather.SANDSTORM: "sand",
    Weather.HAIL: "snow",
    Weather.SNOWSCAPE: "snow",
}
_TERRAIN_TO_STR = {
    Field.ELECTRIC_TERRAIN: "electric",
    Field.GRASSY_TERRAIN: "grassy",
    Field.PSYCHIC_TERRAIN: "psychic",
    Field.MISTY_TERRAIN: "misty",
}


def _weather_str(battle: DoubleBattle) -> str | None:
    for weather in battle.weather:
        mapped = _WEATHER_TO_STR.get(weather)
        if mapped:
            return mapped
    return None


def _terrain_str(battle: DoubleBattle) -> str | None:
    for field_ in battle.fields:
        mapped = _TERRAIN_TO_STR.get(field_)
        if mapped:
            return mapped
    return None


def _active_public(mon) -> dict[str, object] | None:
    if mon is None or mon.fainted:
        return None
    return {
        "species": to_id(mon.species),
        "hp_percent": round(float(mon.current_hp_fraction) * 100.0, 1),
        "status": str(mon.status).split(".")[-1].lower() if mon.status else None,
        "boosts": dict(mon.boosts or {}),
        "item_known": mon.item is not None,
        "item": to_id(mon.item) if mon.item else None,
        "ability_known": mon.ability is not None,
        "ability": to_id(mon.ability) if mon.ability else None,
        "revealed_moves": [to_id(move) for move in (mon.moves or {}).keys()],
    }


def build_public_board_state(
    battle: DoubleBattle,
    memory: BattleMemory,
    config: PolicyConfig,
) -> dict[str, object]:
    """Fog-safe snapshot: own team sheet plus public observations and beliefs."""

    # Lazy import avoids evaluator <-> opponent_belief cycle at module load.
    from vgc.opponent_belief import information_boundary_summary

    boundary = information_boundary_summary(battle, memory, config)
    trick_room = Field.TRICK_ROOM in battle.fields
    our_tailwind = SideCondition.TAILWIND in battle.side_conditions
    opp_tailwind = SideCondition.TAILWIND in battle.opponent_side_conditions
    return {
        "format": config.format_id,
        "turn": int(getattr(battle, "turn", 0) or 0),
        "weather": _weather_str(battle),
        "terrain": _terrain_str(battle),
        "trick_room": trick_room,
        "our_tailwind": our_tailwind,
        "opponent_tailwind": opp_tailwind,
        "our_active": [
            _active_public(mon) for mon in (battle.active_pokemon or [None, None])[:2]
        ],
        "opponent_active": [
            _active_public(mon)
            for mon in (battle.opponent_active_pokemon or [None, None])[:2]
        ],
        "information_boundary": boundary,
    }
