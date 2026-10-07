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
A snapshot condition's `turns` is its start turn as implied by Showdown's real duration
ticks (`vgc.condition_clock`: poke-env's own stamp restarted weather every upkeep and
charged switch-in setters a turn they never lost). With that, one rule holds for moves,
switch-in setters and weather alike, verified 266/266 against the live simulator's
remaining durations on 16 direct games (2026-10-05):

    last_active_turn = s + D - 1

with D = 4 (Tailwind), 5 (Trick Room, terrain, weather). A condition covers projected
turn t iff `state.turn + t - 1 <= last_active_turn`. Item extensions (8 turns) are only
known once a condition outlives 5 turns; `vgc.condition_clock` then reports it net of the
extension. Desolate Land / Primordial Sea are permanent.

Measured plan value for OUR side (`PolicyConfig.exact_search_field_measured_plan`)
----------------------------------------------------------------------------------
`F_t`'s attack term is a guess (base power x generic weather/terrain modifiers); it cannot
see what a condition enables (rain firing Electro Shot in one turn, Expanding Force hitting
both foes...). With the flag on, OUR side's attack term is instead the damage gain MEASURED by
the real engine (`vgc.plan_value`), per surviving active Pokemon, and per surviving brought
bench Pokemon at `exact_search_field_reserve_weight`, with the two largest contributions
counted. Sand chip and Grassy healing stay as separate additive terms. The opponent keeps the
estimate (its sets are hidden), so the term is deliberately asymmetric in phase 1. A set with
no cache entry falls back to the estimate and is counted in `plan_value.FALLBACKS`.

Measured speed payoff (`PolicyConfig.exact_search_field_measured_speed`)
------------------------------------------------------------------------
`S_t` treats Tailwind / Trick Room as "who moves first, averaged over pairings", blind to what
moving first is worth for a given set. With the flag on, such a condition is taken OUT of
`S_t` and valued by the engine-measured payoff of `vgc.speed_payoff` -- %HP per Pokemon per turn
dealt minus taken against a panel of real M-C sets -- added to the fit term (so
`exact_search_field_fit_weight` applies). Active Pokemon count fully, brought bench at
`exact_search_field_reserve_weight`; weather-speed abilities stay in `S_t`. The rules:

* **Each payoff is already a NET duel number** (damage dealt minus damage taken), so the same
  exchange shows up once in our side's numbers and once, negated, in the opponent's. The two
  views are AVERAGED, never summed: Trick Room = (ours.tr - theirs.tr) / 2; our Tailwind =
  (ours.tw - theirs.tw_against) / 2; their Tailwind = (ours.tw_against - theirs.tw) / 2 (a cost).
  Worked example: Trick Room with our two slow sets at +20 each and their two fast ones at -10
  each is (40 - (-20)) / 2 = +30 %HP per turn, not 60. If only one side's Pokemon are cached
  (ours = every surviving Pokemon of ours; theirs = every ACTIVE foe), that single view is used
  as is; with neither cached the condition keeps the generic term.
* **The probe measured ONE control on an otherwise bare field**, so the measured value is used
  only on a projected turn on which exactly one of {our Tailwind, their Tailwind, Trick Room} is
  up. When two or more overlap, that turn is valued by the generic `S_t`, which orders every
  combination exactly: two Tailwinds double both sides' Speed and cancel (no credit to either
  side), and Tailwind inside Trick Room makes the Tailwind side slower, never a benefit.

Not modeled (known gaps)
------------------------
Duration extenders (Heat/Damp/Smooth/Icy Rock, Terrain Extender: 8 turns) until the
condition outlives its base duration; priority moves; Psychic Terrain priority block, Misty/Electric
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

from vgc import plan_value, speed_payoff
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
WEATHER_DURATION = 5

# Fallback for a Trick Room / Tailwind / terrain whose start turn is missing.
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
    # The snapshot's start turn already reflects Showdown's real duration ticks
    # (vgc.condition_clock), so one rule covers moves, switch-in setters and weather.
    return start + duration - 1


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
            start = next(effect.turns for effect in state.weather if effect.id == weather_id)
            weather_turns = _turns_covered(turn, start, WEATHER_DURATION, horizon)
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
def _chip_heal_pct(species_id: str, ability: str | None, weather: str | None, terrain: str | None) -> float:
    """%-of-max-HP per turn of sand chip (negative) and Grassy healing (positive)."""
    try:
        state = PokemonState(species_id=species_id, ability=ability)
        types = {t.lower() for t in state.types()}
        grounded = _is_grounded(state)
    except KeyError:
        return 0.0
    total = 0.0
    if weather == "sand" and not (types & _SAND_IMMUNE_TYPES) and (
        ability not in _SAND_IMMUNE_ABILITIES
    ):
        total -= SAND_CHIP_PCT
    if terrain == "grassy" and grounded:
        total += GRASSY_HEAL_PCT
    return total


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
    total = 0.0
    bare = _best_attack(species_id, ability, moves, None, None)
    if bare > 0:
        total += (_best_attack(species_id, ability, moves, weather, terrain) / bare - 1.0) * (
            NOMINAL_DAMAGE_PCT
        )
    return total + _chip_heal_pct(species_id, ability, weather, terrain)


# --- per-Pokemon views -------------------------------------------------------------------


class _Mon:
    """Everything the two sub-scores need from one Pokemon, computed once per call."""

    __slots__ = (
        "active", "speeds", "multiplier", "speed_ability", "species_id", "ability", "moves",
        "plan", "types", "speed",
    )

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
        # Integer staged Speed; Scarf/paralysis are applied with Tailwind and weather in
        # `_speed_at` as ONE chained modifier, the way Showdown rounds them.
        self.speeds = tuple((_apply_stage(int(speed), stage), w) for speed, w in distribution)
        self.multiplier = multiplier

        if ours:
            move_ids = tuple(sorted(m.id for m in mon.moves))
        else:
            revealed = tuple(sorted(m.id for m in mon.moves))
            move_ids = _opponent_moves(
                mon.species_id, mon.base_species_id or "", revealed, config
            )
        self.moves = move_ids

        # Our own measured plan value (vgc.plan_value), or None: not requested, or this set
        # was not in the cache (counted, and `_measured_side_fit` falls back to the estimate).
        self.plan = None
        if ours and config.exact_search_field_measured_plan:
            self.plan = plan_value.lookup(mon.species_id, move_ids)
            if self.plan is None:
                plan_value.FALLBACKS[mon.species_id] += 1
        self.types = None

        # Measured Tailwind / Trick Room payoff (vgc.speed_payoff): our own set, or the
        # opponent species' most common set. None when not requested or not cached.
        self.speed = None
        if config.exact_search_field_measured_speed:
            self.speed = (
                speed_payoff.lookup_own(mon.species_id, move_ids)
                if ours
                else speed_payoff.lookup_opponent(mon.species_id)
            )


def _alive_mons(side: Any, *, ours: bool, config: PolicyConfig) -> list[_Mon]:
    result = []
    for mon in side.pokemon:
        if mon.fainted or not mon.species_id:
            continue
        view = _Mon(mon, ours=ours, config=config)
        result.append(view)
    return result


def _showdown_modify(value: int, factor: float) -> int:
    """Showdown's `modify`: a 4096-based chained modifier, rounded half down.

    Fractional speeds would turn exact ties (e.g. Scarfed 101 vs 151) into a sure first
    move; Showdown compares these integers.
    """

    modifier = round(factor * 4096)
    return (value * modifier + 2047) // 4096


def _speed_at(mon: _Mon, weather: str | None, tailwind: bool) -> list[tuple[float, float]]:
    factor = mon.multiplier
    if tailwind:
        factor *= 2.0
    if mon.speed_ability is not None and weather is not None:
        if _WEATHER_SPEED_ABILITY.get(weather) == mon.speed_ability:
            factor *= 2.0
    return [(float(_showdown_modify(int(speed), factor)), weight) for speed, weight in mon.speeds]


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


class _Measured:
    """Per projected turn t: which speed controls are valued by measurement, not the generic term.

    At most one of the three is True on any turn (the probe measured one control alone; see
    the module docstring). `ours` / `theirs` say which sides' payoffs are available.
    """

    __slots__ = ("our_tailwind", "their_tailwind", "trick_room", "ours", "theirs", "cancel")

    def __init__(self, horizon: int = 0, ours: bool = False, theirs: bool = False) -> None:
        self.our_tailwind = [False] * horizon
        self.their_tailwind = [False] * horizon
        self.trick_room = [False] * horizon
        self.cancel = [False] * horizon  # both Tailwinds up: the generic term cancels them
        self.ours = ours
        self.theirs = theirs


def _speed_order(
    ours: list[_Mon],
    theirs: list[_Mon],
    conditions: _Conditions,
    horizon: int,
    measured: _Measured | None = None,
) -> list[float]:
    result = []
    for t in range(horizon):
        weather = conditions.weather[t]
        tr = conditions.trick_room[t]
        our_tw = conditions.our_tailwind[t]
        their_tw = conditions.their_tailwind[t]
        if measured is not None:
            # A condition valued by `_measured_speed_fit` is removed from the generic term so it
            # is not counted twice. Two Tailwinds double both sides and cancel exactly.
            if measured.cancel[t]:
                our_tw = their_tw = False
            tr = tr and not measured.trick_room[t]
            our_tw = our_tw and not measured.our_tailwind[t]
            their_tw = their_tw and not measured.their_tailwind[t]
        our_speeds = [_speed_at(m, weather, our_tw) for m in ours]
        their_speeds = [_speed_at(m, weather, their_tw) for m in theirs]
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


def _is_airborne(mon: _Mon) -> bool:
    """Flying-type or Levitate (the ability is only known when revealed or unique)."""
    if mon.ability == "levitate":
        return True
    if mon.types is None:
        data = load_species().get(mon.species_id) or {}
        mon.types = frozenset(str(t).lower() for t in data.get("types", ()))
    return "flying" in mon.types


def _plan_profile(theirs: list[_Mon]) -> int:
    """Which reference foe side to read our measured gains against.

    Two opposing Pokemon in play -> the two-foe profile (spread moves and Expanding Force hit
    both, so their gain is real). One -> the one-foe profile, or the airborne one when that
    single foe is Flying/Levitate (grounded-only terrain effects vanish against it). The
    measured gains already hold the best move per profile, so no "which is higher" test is
    needed: the two-foe gain is never below the one-foe gain for the same set.
    """
    actives = [mon for mon in theirs if mon.active]
    if len(actives) >= 2:
        return plan_value.PROFILE_INDEX["double"]
    if len(actives) == 1 and _is_airborne(actives[0]):
        return plan_value.PROFILE_INDEX["airborne"]
    return plan_value.PROFILE_INDEX["single"]


def _measured_side_fit(
    ours: list[_Mon], profile: int, weather: str | None, terrain: str | None, reserve_weight: float
) -> float:
    """Our side's field fit from cached engine measurements (see `vgc.plan_value`).

    Each surviving ACTIVE Pokemon contributes its measured gain plus the sand-chip /
    Grassy-heal term (the probe excludes chip); each surviving brought BENCH Pokemon
    contributes its measured gain at `reserve_weight` (it can still come in). Only the two
    largest contributions count: a side has two Pokemon on the field, and a plan that needs
    a third is not a plan the next turn can use. An active Pokemon with no cache entry falls
    back to the old estimate (and was counted in `plan_value.FALLBACKS` when built).
    """
    contributions = []
    for mon in ours:
        if mon.plan is None:
            if mon.active:
                contributions.append(_field_pct(mon.species_id, mon.ability, mon.moves, weather, terrain))
            continue
        gain = mon.plan.gain_at(weather, terrain, profile)
        if mon.active:
            contributions.append(gain + _chip_heal_pct(mon.species_id, mon.ability, weather, terrain))
        else:
            contributions.append(reserve_weight * gain)
    contributions.sort(reverse=True)
    return sum(contributions[:2])


def _speed_side_value(mons: list[_Mon], field: str, reserve_weight: float) -> float:
    """A side's measured speed payoff: every ACTIVE Pokemon counts fully and with its sign,
    plus the average bench payoff at the reserve weight.

    Not "the two best": Trick Room on our side hurts a fast active Pokemon right now even if
    a slow teammate on the bench would love it, and taking the two largest values hid that.
    """

    active = [getattr(mon.speed, field) for mon in mons if mon.speed is not None and mon.active]
    bench = [getattr(mon.speed, field) for mon in mons if mon.speed is not None and not mon.active]
    value = sum(active)
    if bench:
        value += reserve_weight * sum(bench) / len(bench)
    return value


def _measured_speed_flags(
    ours: list[_Mon], theirs: list[_Mon], conditions: _Conditions, horizon: int
) -> _Measured:
    """Which turns' speed control can be valued by measurement (see the module docstring).

    Ours need every surviving Pokemon of ours cached; the opponent needs every ACTIVE foe
    cached (its bench is mostly unrevealed). A turn is measured only when exactly one control
    is up and at least one side's payoffs are available; two Tailwinds cancel; any other overlap
    keeps the generic term.
    """
    ours_ok = bool(ours) and all(mon.speed is not None for mon in ours)
    theirs_ok = all(mon.speed is not None for mon in theirs if mon.active) and any(
        mon.active for mon in theirs
    )
    measured = _Measured(horizon, ours_ok, theirs_ok)
    for t in range(horizon):
        our_tw, their_tw = conditions.our_tailwind[t], conditions.their_tailwind[t]
        room = conditions.trick_room[t]
        measured.cancel[t] = our_tw and their_tw
        if not (ours_ok or theirs_ok) or (our_tw + their_tw + room) != 1:
            continue
        measured.our_tailwind[t] = our_tw
        measured.their_tailwind[t] = their_tw
        measured.trick_room[t] = room
    return measured


def _mean_views(views: list[float]) -> float:
    return sum(views) / len(views) if views else 0.0


def _measured_speed_fit(
    ours: list[_Mon],
    theirs: list[_Mon],
    conditions: _Conditions,
    horizon: int,
    measured: _Measured,
    config: PolicyConfig,
) -> list[float]:
    """%HP per turn the measured Tailwind / Trick Room payoff adds for us (see vgc.speed_payoff).

    Every payoff is a net duel number, so the same exchange appears in our sets' numbers and
    (negated) in theirs: the views are averaged, not summed (see the module docstring).
    """
    reserve = config.exact_search_field_reserve_weight
    scale = config.exact_search_speed_payoff_scale

    def side(mons: list[_Mon], field: str) -> float:
        return _speed_side_value(mons, field, reserve)

    our_tailwind_value = _mean_views(
        ([side(ours, "tw")] if measured.ours else [])
        + ([-side(theirs, "tw_against")] if measured.theirs else [])
    )
    their_tailwind_value = _mean_views(
        ([side(ours, "tw_against")] if measured.ours else [])
        + ([-side(theirs, "tw")] if measured.theirs else [])
    )
    trick_room_value = _mean_views(
        ([side(ours, "tr")] if measured.ours else [])
        + ([-side(theirs, "tr")] if measured.theirs else [])
    )
    result = []
    for t in range(horizon):
        total = 0.0
        if measured.our_tailwind[t]:
            total = our_tailwind_value
        elif measured.their_tailwind[t]:
            total = their_tailwind_value
        elif measured.trick_room[t]:
            total = trick_room_value
        result.append(scale * total)
    return result


def _field_fit(
    ours: list[_Mon],
    theirs: list[_Mon],
    conditions: _Conditions,
    horizon: int,
    config: PolicyConfig,
) -> list[float]:
    measured = config.exact_search_field_measured_plan
    profile = _plan_profile(theirs) if measured else 0
    result = []
    for t in range(horizon):
        weather, terrain = conditions.weather[t], conditions.terrain[t]
        if weather is None and terrain is None:
            result.append(0.0)
            continue
        total = 0.0
        if measured:
            total += _measured_side_fit(
                ours, profile, weather, terrain, config.exact_search_field_reserve_weight
            )
        else:
            for mon in ours:
                if mon.active:
                    total += _field_pct(mon.species_id, mon.ability, mon.moves, weather, terrain)
        # The opponent keeps the estimate in both modes: its sets are not ours to measure.
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
    measured = None
    if config.exact_search_field_measured_speed:
        measured = _measured_speed_flags(ours, theirs, conditions, horizon)
    speed = _speed_order(ours, theirs, conditions, horizon, measured)
    fit = _field_fit(ours, theirs, conditions, horizon, config)
    if measured is not None:
        extra = _measured_speed_fit(ours, theirs, conditions, horizon, measured, config)
        fit = [a + b for a, b in zip(fit, extra)]
    value = 0.0
    scale = 1.0
    for t in range(horizon):
        value += scale * (
            config.exact_search_speed_order_weight * speed[t]
            + config.exact_search_field_fit_weight * fit[t]
        )
        scale *= config.exact_search_field_decay
    return value
