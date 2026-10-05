"""Speed-control and weather/terrain-control value for the exact judge.

`field_control_value(state, config)` returns OUR advantage minus THEIR advantage, in the
same currency as `vgc.rl.exact_search._position_value` (one full-HP Pokemon = 100 HP points
+ 90 alive points), from conditions that outlast the current turn:

    V = sum_{t=1..H} decay^(t-1) * (speed_order_weight * S_t + field_fit_weight * F_t)

* `S_t` in [-1, 1]: how often our Pokemon move before theirs on projected turn t, given
  Tailwind / Trick Room / weather-speed abilities that are still up at t.
* `F_t` in %-of-max-HP per turn: our weather/terrain offence gain + chip/heal minus theirs.

Only DIFFERENCES between boards matter (the exact search subtracts the root's value), so
the module is built to be consistent across boards, not to be an absolute calibration.

Projected turn t=1 is the turn the board is about to play (`state.turn`): for a branch's
final board that is the next turn, for the root it is the turn being decided.

Durations (verified against the real engine, 2026-10-05, ~60 direct battles)
---------------------------------------------------------------------------
A snapshot condition's `turns` is the poke-env turn on which it was set, and the engine
removes it at the end of the turn on which its duration runs out. Observed: Tailwind set
on turn s is visible at decisions s+1..s+3 (last_active = s+3, duration 4 counting the
setting turn). Terrains set on turn s>=1 are visible through s+4 (duration 5), and a
terrain set by a turn-0 switch-in (s=0, no residual before turn 1) through turn 5. Rule:

    last_active_turn = s + D - 1    (s >= 1)        last_active_turn = D    (s == 0)

with D = 4 (Tailwind), 5 (Trick Room, terrain; same engine mechanism, Trick Room itself
only observed up to s+3 in the probe, never contradicting the rule). A condition covers
projected turn t iff `state.turn + t - 1 <= last_active_turn`.

WEATHER IS DIFFERENT: poke-env overwrites the weather's `turns` with the current turn on
every `-weather ... [upkeep]` line (abstract_battle.py), so a snapshot cannot say when the
weather began. Weather therefore is assumed to last `WEATHER_ASSUMED_REMAINING` more turns
(permanent Desolate Land / Primordial Sea: unlimited). Whether weather is up at all, and
which one, is exact; only its expiry is a flat guess.

Not modeled (known gaps)
------------------------
Duration extenders (Heat/Damp/Smooth/Icy Rock, Terrain Extender: 8 turns) because the
setter's item is not recorded; priority moves; Psychic Terrain priority block, Misty/Electric
Terrain status blocks, Grassy Terrain's Earthquake halving; Sand / Snow defensive boosts;
Protosynthesis / Quark Drive / Solar Power / Sand Force; speed-tie randomness beyond 0.5;
unrevealed opponent Pokemon; hidden opponent abilities (used only when the species has a
single possible ability); future switches (bench Pokemon count at half weight for speed,
not at all for field fit).
"""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from vgc.damage import (
    FieldState,
    PokemonState,
    _apply_stage,
    _is_grounded,
    _terrain_modifier,
    _weather_modifier,
    to_id,
)
from vgc.data import load_moves, load_species
from vgc.models import PolicyConfig
from vgc.sets import opponent_move_ids, opponent_spread_hypotheses, set_priors_for
from vgc.sets import usage_spreads_for
from vgc.stats import calculate_stats, default_opponent_nature, default_opponent_spread

# --- constants (conversions and engine facts, not strategic weights) -------------------

# Engine durations, counting the turn the condition is set (see module docstring).
TAILWIND_DURATION = 4
TRICK_ROOM_DURATION = 5
TERRAIN_DURATION = 5

# Weather expiry is unknowable from poke-env (see docstring): assume this many more turns.
# A 5-turn weather observed at a random point has ~2-3 turns left; set-up turns skew high.
WEATHER_ASSUMED_REMAINING = 3
# Same fallback for a Trick Room / Tailwind / terrain whose start turn is missing.
UNKNOWN_START_REMAINING = 2

# Conversion from "relative gain in best expected attack power" to %-of-max-HP per turn.
# A strong attack does roughly 35% to a neutral target in doubles; a +30% power boost is
# therefore worth ~10% HP per turn. This is a unit conversion; the strategic weight is
# `PolicyConfig.exact_search_field_fit_weight`.
NOMINAL_DAMAGE_PCT = 35.0

SAND_CHIP_PCT = 6.25
GRASSY_HEAL_PCT = 6.25
_SAND_IMMUNE_TYPES = frozenset({"rock", "ground", "steel"})
_SAND_IMMUNE_ABILITIES = frozenset(
    {"overcoat", "sandveil", "sandrush", "sandforce", "magicguard"}
)
_BENCH_PAIR_WEIGHT = 0.5
_SPREAD_HYPOTHESES = 6  # top-N opponent spread beliefs used for the Speed distribution

_WEATHER_IDS = {
    "sunnyday": "sun",
    "desolateland": "sun",
    "raindance": "rain",
    "primordialsea": "rain",
    "sandstorm": "sand",
    "snowscape": "snow",
    "snow": "snow",
    "hail": "snow",
}
_PERMANENT_WEATHER = frozenset({"desolateland", "primordialsea"})
_TERRAIN_IDS = {
    "electricterrain": "electric",
    "grassyterrain": "grassy",
    "psychicterrain": "psychic",
    "mistyterrain": "misty",
}
_WEATHER_SPEED_ABILITY = {
    "sun": "chlorophyll",
    "rain": "swiftswim",
    "sand": "sandrush",
    "snow": "slushrush",
}
_WEATHER_BALL_TYPE = {"sun": "fire", "rain": "water", "sand": "rock", "snow": "ice"}


# --- condition durations ----------------------------------------------------------------


def _last_active_turn(start: int | None, duration: int) -> int | None:
    """Last turn on which a condition set on turn `start` still applies (see docstring)."""
    if start is None:
        return None
    return duration if start == 0 else start + duration - 1


def _turns_covered(turn: int, start: int | None, duration: int, horizon: int) -> int:
    """How many of projected turns 1..horizon the condition covers (a prefix)."""
    last = _last_active_turn(start, duration)
    if last is None:
        return min(horizon, UNKNOWN_START_REMAINING)
    return max(0, min(horizon, last - turn + 1))


class _Conditions:
    """Field conditions per projected turn t = 1..H (index t-1)."""

    __slots__ = ("weather", "terrain", "trick_room", "our_tailwind", "their_tailwind")

    def __init__(self, state: Any, horizon: int) -> None:
        turn = int(state.turn or 0)
        weather_ids = [effect.id for effect in state.weather if effect.id in _WEATHER_IDS]
        weather_id = weather_ids[0] if weather_ids else None
        if weather_id is None:
            weather_turns = 0
        elif weather_id in _PERMANENT_WEATHER:
            weather_turns = horizon
        else:
            weather_turns = min(horizon, WEATHER_ASSUMED_REMAINING)
        weather_name = _WEATHER_IDS.get(weather_id) if weather_id else None
        self.weather = [weather_name if t < weather_turns else None for t in range(horizon)]

        terrain_name, terrain_turns = None, 0
        trick_turns = 0
        for effect in state.fields:
            if effect.id in _TERRAIN_IDS:
                terrain_name = _TERRAIN_IDS[effect.id]
                terrain_turns = _turns_covered(turn, effect.turns, TERRAIN_DURATION, horizon)
            elif effect.id == "trickroom":
                trick_turns = _turns_covered(turn, effect.turns, TRICK_ROOM_DURATION, horizon)
        self.terrain = [terrain_name if t < terrain_turns else None for t in range(horizon)]
        self.trick_room = [t < trick_turns for t in range(horizon)]
        self.our_tailwind = self._tailwind(state.our_side, turn, horizon)
        self.their_tailwind = self._tailwind(state.opponent_side, turn, horizon)

    @staticmethod
    def _tailwind(side: Any, turn: int, horizon: int) -> list[bool]:
        covered = 0
        for effect in side.side_conditions:
            if effect.id == "tailwind":
                covered = _turns_covered(turn, effect.turns, TAILWIND_DURATION, horizon)
        return [t < covered for t in range(horizon)]


# --- cached lookups -----------------------------------------------------------------------


@lru_cache(maxsize=1024)
def _opponent_speed_distribution(
    species_id: str, usage_file: str, config: PolicyConfig
) -> tuple[tuple[int, float], ...]:
    """Weighted raw Speed stats for a hidden-spread opponent: ((speed, probability), ...)."""
    try:
        hypotheses = opponent_spread_hypotheses(
            species_id, usage_spreads_for(config), limit=_SPREAD_HYPOTHESES
        )
        merged: dict[int, float] = {}
        for spread, nature, weight in hypotheses:
            speed = calculate_stats(species_id, sp=spread, nature=nature)["spe"]
            merged[speed] = merged.get(speed, 0.0) + weight
    except KeyError:
        return ()
    return tuple(sorted(merged.items()))


@lru_cache(maxsize=1024)
def _default_speed(species_id: str) -> int | None:
    try:
        return calculate_stats(
            species_id,
            sp=default_opponent_spread(species_id),
            nature=default_opponent_nature(species_id),
        )["spe"]
    except KeyError:
        return None


@lru_cache(maxsize=1024)
def _sole_ability(species_id: str) -> str | None:
    species = load_species().get(species_id) or {}
    abilities = {to_id(name) for name in (species.get("abilities") or {}).values()}
    return next(iter(abilities)) if len(abilities) == 1 else None


@lru_cache(maxsize=1024)
def _opponent_moves(
    species_id: str, base_species_id: str, revealed: tuple[str, ...], config: PolicyConfig
) -> tuple[str, ...]:
    pokemon = SimpleNamespace(species=base_species_id or species_id, moves=dict.fromkeys(revealed))
    return tuple(opponent_move_ids(pokemon, set_priors_for(config), config))


@lru_cache(maxsize=None)
def _grounded_dummy() -> PokemonState:
    species = load_species()
    for species_id, data in species.items():
        if "Flying" not in data.get("types", ()) and not species_id.endswith("mega"):
            return PokemonState(species_id=species_id)
    raise RuntimeError("no grounded species in data")


@lru_cache(maxsize=4096)
def _best_attack(
    species_id: str,
    ability: str | None,
    moves: tuple[str, ...],
    weather: str | None,
    terrain: str | None,
) -> float:
    """Best expected damaging-move power (BP x STAB x accuracy x weather/terrain)."""
    try:
        attacker = PokemonState(species_id=species_id, ability=ability)
        own_types = {t.lower() for t in attacker.types()}
    except KeyError:
        return 0.0
    field = FieldState(weather=weather, terrain=terrain)
    defender = _grounded_dummy()
    move_data = load_moves()
    best = 0.0
    for move_id in moves:
        move = move_data.get(move_id)
        if not move or move.get("category") == "Status":
            continue
        power = float(move.get("basePower") or 0)
        move_type = str(move.get("type") or "Normal")
        if move_id == "weatherball":
            if weather in _WEATHER_BALL_TYPE:
                power *= 2.0
                move_type = _WEATHER_BALL_TYPE[weather].capitalize()
        if power <= 0:
            continue
        accuracy = move.get("accuracy")
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
            power *= accuracy / 100.0
        if move_type.lower() in own_types:
            power *= 1.5
        power *= _weather_modifier(field, move_type)
        power *= _terrain_modifier(field, move_type, attacker, defender)
        best = max(best, power)
    return best


@lru_cache(maxsize=8192)
def _field_pct(
    species_id: str,
    ability: str | None,
    moves: tuple[str, ...],
    weather: str | None,
    terrain: str | None,
) -> float:
    """%-of-max-HP per turn this Pokemon gains from the field vs. a bare field."""
    if weather is None and terrain is None:
        return 0.0
    try:
        state = PokemonState(species_id=species_id, ability=ability)
        types = {t.lower() for t in state.types()}
        grounded = _is_grounded(state)
    except KeyError:
        return 0.0
    total = 0.0
    bare = _best_attack(species_id, ability, moves, None, None)
    if bare > 0:
        total += (_best_attack(species_id, ability, moves, weather, terrain) / bare - 1.0) * (
            NOMINAL_DAMAGE_PCT
        )
    if weather == "sand" and not (types & _SAND_IMMUNE_TYPES) and (
        ability not in _SAND_IMMUNE_ABILITIES
    ):
        total -= SAND_CHIP_PCT
    if terrain == "grassy" and grounded:
        total += GRASSY_HEAL_PCT
    return total


# --- per-Pokemon views -------------------------------------------------------------------


class _Mon:
    """Everything the two sub-scores need from one Pokemon, computed once per call."""

    __slots__ = ("active", "speeds", "speed_ability", "species_id", "ability", "moves")

    def __init__(self, mon: Any, *, ours: bool, config: PolicyConfig) -> None:
        self.active = bool(mon.active)
        self.species_id = mon.species_id
        ability = mon.ability_id if (ours or mon.ability_known) else None
        if ability is None and not ours:
            ability = _sole_ability(mon.species_id)
        self.ability = ability
        self.speed_ability = ability if ability in _WEATHER_SPEED_ABILITY.values() else None

        if ours:
            stats = dict(mon.stats)
            base = stats.get("spe") or _default_speed(mon.species_id)
            distribution = ((float(base), 1.0),) if base else ()
        else:
            distribution = tuple(
                (float(speed), weight)
                for speed, weight in _opponent_speed_distribution(
                    mon.species_id, config.usage_spreads_file, config
                )
            )
            if not distribution:
                default = _default_speed(mon.species_id)
                distribution = ((float(default), 1.0),) if default else ()

        stage = dict(mon.boosts).get("spe", 0) if self.active else 0
        multiplier = 1.0
        if mon.item_id == "choicescarf" and (ours or mon.item_known):
            multiplier *= 1.5
        if mon.status == "par":
            multiplier *= 0.5
        self.speeds = tuple((_apply_stage(int(speed), stage) * multiplier, w) for speed, w in distribution)

        if ours:
            move_ids = tuple(sorted(m.id for m in mon.moves))
        else:
            revealed = tuple(sorted(m.id for m in mon.moves))
            move_ids = _opponent_moves(
                mon.species_id, mon.base_species_id or "", revealed, config
            )
        self.moves = move_ids


def _alive_mons(side: Any, *, ours: bool, config: PolicyConfig) -> list[_Mon]:
    result = []
    for mon in side.pokemon:
        if mon.fainted or not mon.species_id:
            continue
        view = _Mon(mon, ours=ours, config=config)
        result.append(view)
    return result


def _speed_at(mon: _Mon, weather: str | None, tailwind: bool) -> list[tuple[float, float]]:
    factor = 1.0
    if tailwind:
        factor *= 2.0
    if mon.speed_ability is not None and weather is not None:
        if _WEATHER_SPEED_ABILITY.get(weather) == mon.speed_ability:
            factor *= 2.0
    return [(speed * factor, weight) for speed, weight in mon.speeds]


def _p_before(ours: list[tuple[float, float]], theirs: list[tuple[float, float]], tr: bool) -> float:
    total = 0.0
    for sa, wa in ours:
        for sb, wb in theirs:
            if sa == sb:
                value = 0.5
            else:
                value = 1.0 if ((sa < sb) if tr else (sa > sb)) else 0.0
            total += wa * wb * value
    return total


def _speed_order(
    ours: list[_Mon], theirs: list[_Mon], conditions: _Conditions, horizon: int
) -> list[float]:
    result = []
    for t in range(horizon):
        weather = conditions.weather[t]
        tr = conditions.trick_room[t]
        our_speeds = [_speed_at(m, weather, conditions.our_tailwind[t]) for m in ours]
        their_speeds = [_speed_at(m, weather, conditions.their_tailwind[t]) for m in theirs]
        weighted = 0.0
        weight_sum = 0.0
        for a, a_speeds in zip(ours, our_speeds):
            if not a_speeds:
                continue
            for b, b_speeds in zip(theirs, their_speeds):
                if not b_speeds:
                    continue
                weight = 1.0 if (a.active and b.active) else _BENCH_PAIR_WEIGHT
                weighted += weight * (2.0 * _p_before(a_speeds, b_speeds, tr) - 1.0)
                weight_sum += weight
        result.append(weighted / weight_sum if weight_sum else 0.0)
    return result


def _field_fit(
    ours: list[_Mon], theirs: list[_Mon], conditions: _Conditions, horizon: int
) -> list[float]:
    result = []
    for t in range(horizon):
        weather, terrain = conditions.weather[t], conditions.terrain[t]
        if weather is None and terrain is None:
            result.append(0.0)
            continue
        total = 0.0
        for mon in ours:
            if mon.active:
                total += _field_pct(mon.species_id, mon.ability, mon.moves, weather, terrain)
        for mon in theirs:
            if mon.active:
                total -= _field_pct(mon.species_id, mon.ability, mon.moves, weather, terrain)
        result.append(total)
    return result


def field_control_value(state: Any, config: PolicyConfig) -> float:
    """Our speed/field-control advantage minus theirs, in exact-judge points (see module)."""
    horizon = max(0, int(config.exact_search_field_horizon))
    if horizon == 0 or state.finished:
        return 0.0
    conditions = _Conditions(state, horizon)
    ours = _alive_mons(state.our_side, ours=True, config=config)
    theirs = _alive_mons(state.opponent_side, ours=False, config=config)
    speed = _speed_order(ours, theirs, conditions, horizon)
    fit = _field_fit(ours, theirs, conditions, horizon)
    value = 0.0
    scale = 1.0
    for t in range(horizon):
        value += scale * (
            config.exact_search_speed_order_weight * speed[t]
            + config.exact_search_field_fit_weight * fit[t]
        )
        scale *= config.exact_search_field_decay
    return value
