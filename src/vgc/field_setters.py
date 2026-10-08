"""Weather/terrain SETTERS for the fast search and the myopic evaluator.

Behind `PolicyConfig.search_model_field_setters`. Sunny Day / Rain Dance / Sandstorm /
Snowscape / Chilly Reception / the four terrains and the switch-in abilities that do the
same (Drought, Drizzle, Sand Stream, Snow Warning, Orichalcum Pulse, the four Surges, Hadron
Engine) used to be invisible: no `utility_kind`, a score of 0 as a status move, nothing
applied in the exchange. This module holds the shared facts and the valuation:

* `MOVE_CONDITIONS` / `ABILITY_CONDITIONS`: which `(kind, value)` a move/ability sets, with
  `kind` in {"weather", "terrain"} and `value` in `vgc.field_control`'s names
  (sun/rain/sand/snow, electric/grassy/psychic/misty).
* `setter_value`: the team-plan payoff of starting a condition THIS turn, in the evaluator's
  currency (about 1 point per 1% of max HP, the same units as the exact judge's
  `field_control_value`). Computed as

      V(board with the condition started now) - V(board as it is)

  where V is `vgc.field_control.field_control_value` over `search_field_setter_horizon`
  projected turns, so it inherits everything that function knows: weather-speed abilities,
  Trick Room, the Showdown-MEASURED plan gains (`exact_search_field_measured_plan`: rain
  firing Electro Shot, Psychic Terrain Expanding Force...) for our side, and the estimated
  opponent fit. A condition that REPLACES an active one removes the old one's value from
  the "after" board, so replacing our own helpful weather is negative by construction. The
  same condition already up means the move/ability fails, so it earns exactly 0.

  For a switch-in setter the comparison is between the board with the swap but no
  condition and the board with the swap plus the condition, so the credit is the
  condition's value for the team that would be on the field, not the swap itself (which the
  switch score already prices).

The modified board is a `dataclasses.replace` copy of the `BattleMechanicsState` snapshot;
nothing here touches a live battle.
"""

from __future__ import annotations

from dataclasses import replace

from vgc.field_control import _PERMANENT_WEATHER, _TERRAIN_IDS, _WEATHER_IDS, field_control_value
from vgc.mechanics_state import BattleMechanicsState, EffectSnapshot, SideMechanicsState
from vgc.models import PolicyConfig

# Engine duration of a weather/terrain started on a turn, counting that turn.
SETTER_DURATION = 5

MOVE_CONDITIONS: dict[str, tuple[str, str]] = {
    "sunnyday": ("weather", "sun"),
    "raindance": ("weather", "rain"),
    "sandstorm": ("weather", "sand"),
    "snowscape": ("weather", "snow"),
    "hail": ("weather", "snow"),
    "chillyreception": ("weather", "snow"),
    "electricterrain": ("terrain", "electric"),
    "grassyterrain": ("terrain", "grassy"),
    "psychicterrain": ("terrain", "psychic"),
    "mistyterrain": ("terrain", "misty"),
}

ABILITY_CONDITIONS: dict[str, tuple[str, str]] = {
    "drought": ("weather", "sun"),
    "orichalcumpulse": ("weather", "sun"),
    "drizzle": ("weather", "rain"),
    "sandstream": ("weather", "sand"),
    "snowwarning": ("weather", "snow"),
    "electricsurge": ("terrain", "electric"),
    "hadronengine": ("terrain", "electric"),
    "grassysurge": ("terrain", "grassy"),
    "psychicsurge": ("terrain", "psychic"),
    "mistysurge": ("terrain", "misty"),
}

_WEATHER_EFFECT_ID = {"sun": "sunnyday", "rain": "raindance", "sand": "sandstorm", "snow": "snowscape"}
_TERRAIN_EFFECT_ID = {
    "electric": "electricterrain",
    "grassy": "grassyterrain",
    "psychic": "psychicterrain",
    "misty": "mistyterrain",
}


def move_condition(move_id: str | None) -> tuple[str, str] | None:
    return MOVE_CONDITIONS.get(move_id or "")


def ability_condition(ability_id: str | None) -> tuple[str, str] | None:
    return ABILITY_CONDITIONS.get(ability_id or "")


def current_condition(state: BattleMechanicsState, kind: str) -> str | None:
    """The weather/terrain name now on the board (field_control's names), or None."""
    if kind == "weather":
        for effect in state.weather:
            if effect.id in _WEATHER_IDS:
                return _WEATHER_IDS[effect.id]
        return None
    for effect in state.fields:
        if effect.id in _TERRAIN_IDS:
            return _TERRAIN_IDS[effect.id]
    return None


def _blocked_by_permanent_weather(state: BattleMechanicsState) -> bool:
    return any(effect.id in _PERMANENT_WEATHER for effect in state.weather)


def with_condition(state: BattleMechanicsState, kind: str, value: str) -> BattleMechanicsState:
    """The snapshot with ``value`` started on the snapshot's turn, replacing any other."""
    start = int(state.turn or 0)
    if kind == "weather":
        effect = EffectSnapshot(
            id=_WEATHER_EFFECT_ID[value], turns=start, raw_value=start, counter_kind="start_turn"
        )
        return replace(state, weather=(effect,))
    effect = EffectSnapshot(
        id=_TERRAIN_EFFECT_ID[value], turns=start, raw_value=start, counter_kind="start_turn"
    )
    fields = tuple(f for f in state.fields if f.id not in _TERRAIN_IDS) + (effect,)
    return replace(state, fields=tuple(sorted(fields, key=lambda f: f.id)))


def with_switch_in(
    state: BattleMechanicsState, slot: int, incoming_species_id: str
) -> BattleMechanicsState:
    """The snapshot with our ``slot`` replaced on the field by ``incoming_species_id``."""
    side: SideMechanicsState = state.our_side
    outgoing = side.active_species[slot] if slot < len(side.active_species) else None
    pokemon = tuple(
        replace(mon, active=False)
        if (mon.active and outgoing is not None and mon.species_id == outgoing)
        else replace(mon, active=True)
        if mon.species_id == incoming_species_id
        else mon
        for mon in side.pokemon
    )
    active = list(side.active_species)
    if slot < len(active):
        active[slot] = incoming_species_id
    return replace(state, our_side=replace(side, pokemon=pokemon, active_species=tuple(active)))


def setter_value(
    state: BattleMechanicsState | None,
    kind: str,
    value: str,
    config: PolicyConfig,
    *,
    switch_in: tuple[int, str] | None = None,
) -> float:
    """Team-plan payoff of starting ``value`` this turn, for us minus them (see module)."""
    if state is None or state.finished:
        return 0.0
    if current_condition(state, kind) == value or (
        kind == "weather" and _blocked_by_permanent_weather(state)
    ):
        return 0.0  # already up (or blocked): the move/ability fails
    horizon = max(1, int(config.search_field_setter_horizon))
    field_config = replace(config, exact_search_field_horizon=horizon)
    base = with_switch_in(state, *switch_in) if switch_in is not None else state
    before = field_control_value(base, field_config)
    after = field_control_value(with_condition(base, kind, value), field_config)
    return config.search_field_setter_weight * (after - before)

