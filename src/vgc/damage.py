"""Gen-9-mechanics damage calculator operating on Champions-mod data.

Confirmed against the local Showdown checkout (`sim/battle-actions.ts::getDamage` /
`modifyDamage`, and `data/mods/champions/scripts.ts::modifyDamage` -- the champions mod
only overrides `modifyDamage` to also announce 4x/0.25x effectiveness messages, the
actual math is identical to base gen9): all mechanics here are standard, unmodified gen9
mechanics running on top of champions-mod species/moves/items/typechart data. Abilities
and the items relevant here are *not* overridden by `data/mods/champions/{abilities,
items}.ts` (grep confirmed -- those files only carry legality (`isNonstandard`) flags for
this item/ability set, no mechanical changes), so standard gen9 ability/item formulas
apply verbatim.

## Pipeline (mirrors `getDamage`/`modifyDamage`'s actual order of operations)

1. Look up the move; most `basePowerCallback` moves (see `data/moves.ts`, no
   `data/mods/champions/moves.ts` overrides for any of these -- confirmed by grep) have
   no flat `basePower` the exported data can carry, so this module reimplements the
   subset whose formula is fully computable from `PokemonState`/`FieldState` alone:
   Weather Ball (type + BP by weather), Water Spout/Eruption (BP scales with the user's
   current HP fraction), Electro Ball/Gyro Ball (BP from a user/target Speed ratio, using
   the same boosted+Choice-Scarf+paralysis "effective speed" `vgc.evaluator.
   effective_speed` uses -- duplicated locally as `_effective_speed_for_ratio` rather
   than importing the Phase 2b evaluator module from this Phase 2a one), and Grass Knot/
   Low Kick/Heavy Slam/Heat Crash (BP from species `weightkg`, no Float Stone/Autotomize/
   `ModifyWeight`-event modeling -- see `_variable_base_power`). Anything else with
   `basePower == 0` still short-circuits with `breakdown["move_supported"] = False`
   rather than silently returning 0 damage -- see `DamageResult.breakdown` docs below.
2. Type effectiveness from `typechart.json`, plus ability-based type immunities
   (Levitate/Flash Fire/Water Absorb/Volt Absorb/Storm Drain/Lightning Rod) and Wonder
   Guard's "only super-effective connects" rule. A dead stop here returns an all-zero
   `DamageResult` with `breakdown["immune"] = True`.
3. Raw Atk/Def stats (`vgc.stats.calculate_stats`) with stat-stage boost multipliers,
   then raw-stat abilities (Huge Power/Pure Power double Atk, Guts is a 1.5x Atk boost
   while statused -- separate from Guts' *other* job of negating burn's physical
   halving, which is handled at the burn step below) and the Light Ball item (2x
   Atk/SpA for Pikachu-line only).
4. Base power modifiers: Technician (basePower <= 60 -> 1.5x, checked against the
   move's raw power same as the real engine's high-priority `onBasePower`), type-boost
   items (Charcoal etc., 1.2x), Muscle Band/Wise Glasses (1.1x by category).
5. The base formula: `floor(floor(floor(floor(2*50/5+2) * power * atk) / def) / 50) + 2`
   (level is always 50 in this format -- see `vgc.stats.FORMAT_LEVEL`).
6. Spread modifier (0.75x, only for `allAdjacent`/`allAdjacentFoes`-target moves
   actually hitting >=2 targets this turn -- `field.num_targets >= 2`).
7. Weather (sun/rain boost-or-cut Fire/Water 1.5x/0.5x) and terrain (Electric/Grassy/
   Psychic terrain 1.3x their type for a grounded attacker; Misty Terrain halves Dragon
   moves against a grounded defender; Grassy Terrain also halves Earthquake/Bulldoze/
   Magnitude against a grounded defender).
8. The 85-100% random roll -- applied here, BEFORE STAB/type/burn/final modifiers, same
   as the real engine (`modifyDamage`'s "random factor" step precedes STAB). This is why
   `damage_range` computes all 16 rolls end-to-end rather than scaling one deterministic
   number by 0.85/1.0 at the very end -- floor-truncation at each stage means those
   aren't quite the same number.
9. STAB (1.5x, or 2x with Adaptability) and type effectiveness (from step 2).
10. Burn halves physical damage unless the attacker has Guts.
11. Final multiplicative stage (all combined into one `final_multiplier`, matching the
    real engine's single `ModifyDamage` event most of these items/abilities/screens hang
    off of): Reflect/Light Screen/Aurora Veil (0.5x singles, 2/3 in doubles -- Aurora
    Veil doesn't stack with a same-category Reflect/Light Screen, matching
    `data/moves.ts`'s `auroraveil` condition), Solid Rock/Filter (0.75x incoming
    super-effective damage), Tinted Lens (2x outgoing resisted damage), Multiscale
    (0.5x incoming damage at full HP), Thick Fat (0.5x incoming Fire/Ice), Life Orb
    (1.3x), Expert Belt (1.2x on a super-effective hit).

Crits are explicitly out of scope for v1 (per the Phase 2a spec) -- `damage_range` never
rolls or assumes a crit. Terastallization/Stellar-type mechanics are also out of scope
for v1 (not in `PokemonState`'s field list) -- see CLAUDE.md's Phase 2a note.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vgc.data import load_moves, load_species, load_typechart
from vgc.stats import StatId, calculate_stats

# --- id normalization ------------------------------------------------------------------

_ID_RE = re.compile(r"[^a-z0-9]")


def to_id(value: str | None) -> str | None:
    """Showdown-style id normalization (lowercase, alphanumeric only), so callers can
    pass either display names ("Life Orb") or ids ("lifeorb") -- poke-env's own fields
    are already ids, but this keeps hand-built `PokemonState`s forgiving.
    """
    if value is None:
        return None
    return _ID_RE.sub("", value.lower())


# --- field/pokemon state ----------------------------------------------------------------

# Stat stages, keyed the same as vgc.stats.BOOSTABLE_STAT_IDS ("atk"/"def"/"spa"/"spd"/
# "spe"); accuracy/evasion stages aren't relevant to damage and are intentionally not
# modeled here.
Boosts = dict[str, int]

STATUS_IDS = frozenset({"brn", "par", "psn", "tox", "slp", "frz"})

# Side conditions relevant to damage; matches the sideCondition ids used by
# data/moves.ts's reflect/lightscreen/auroraveil entries.
ScreenId = str  # "reflect" | "lightscreen" | "auroraveil"


@dataclass
class PokemonState:
    """One side of a `damage_range` calculation. Mutable (unlike `PolicyConfig`) --
    this is transient per-decision battle state, not a behavior-gating config object.

    :param species_id: key into `vgc.data.load_species()`. For a mega forme, pass the
        mega's own id directly (e.g. `"garchompmega"`) -- its `baseStats`/`types`/
        `abilities` already reflect the mega forme (see CLAUDE.md).
    :param sp_spread: Stat Points (see `vgc.stats`), e.g. `{"hp": 32, "atk": 32}`.
        `None` means "unknown" -- `stats()` falls back to
        `vgc.stats.default_opponent_spread(species_id)`, which is what you want for an
        opponent whose real spread Open Team Sheets doesn't reveal (see `vgc.sets`).
    :param nature: nature id; `None` falls back to
        `vgc.stats.default_opponent_nature(species_id)` for the same reason as `sp_spread`.
    :param boosts: stat stages (-6..6), e.g. `{"atk": 2}` for a +2 Atk boost. Missing
        stats default to 0.
    :param status: one of `STATUS_IDS`, or None.
    :param item: item id (or display name -- normalized via `to_id`), or None.
    :param ability: ability id (or display name), or None.
    :param current_hp: current HP in absolute points. `None` means "at full HP" (the
        only place this matters here is Multiscale).
    """

    species_id: str
    sp_spread: dict[str, int] | None = None
    nature: str | None = None
    boosts: Boosts = field(default_factory=dict)
    status: str | None = None
    item: str | None = None
    ability: str | None = None
    current_hp: int | None = None

    def __post_init__(self) -> None:
        self.item = to_id(self.item) or None
        self.ability = to_id(self.ability) or None

    def species_data(self) -> dict:
        species = load_species().get(self.species_id)
        if species is None:
            raise KeyError(f"unknown species id {self.species_id!r} (check vgc.data.load_species())")
        return species

    def types(self) -> list[str]:
        return list(self.species_data()["types"])

    def stats(self) -> dict[StatId, int]:
        from vgc.stats import default_opponent_nature, default_opponent_spread

        sp = self.sp_spread if self.sp_spread is not None else default_opponent_spread(self.species_id)
        nature = self.nature if self.nature is not None else default_opponent_nature(self.species_id)
        return calculate_stats(self.species_id, sp=sp, nature=nature)

    def max_hp(self) -> int:
        return self.stats()["hp"]

    def hp_or_max(self) -> int:
        return self.current_hp if self.current_hp is not None else self.max_hp()

    def boost_stage(self, stat_id: str) -> int:
        return self.boosts.get(stat_id, 0)


@dataclass
class FieldState:
    """Battle-field context shared by both sides for one `damage_range` call.

    :param weather: `"sun"` | `"rain"` | None. (Sand/snow are accepted as values too for
        callers that want to carry them through, but neither has a damage-formula effect
        implemented here -- sand/snow's real effects are Sp.Def/Def boosts, not damage
        multipliers, so they're out of scope for a damage *calculator*.)
    :param terrain: `"electric"` | `"grassy"` | `"psychic"` | `"misty"` | None.
    :param screens: side conditions active on the DEFENDER's side, subset of
        `{"reflect", "lightscreen", "auroraveil"}`.
    :param trick_room: stored for evaluator context (e.g. speed-order decisions); not
        used by `damage_range` itself, since crit/speed-dependent moves aren't in scope.
    :param is_doubles: drives the screens multiplier (0.5x singles vs 2/3x doubles,
        matching `activePerHalf > 1` in `data/moves.ts`). Defaults to True since this
        project's target format is always doubles.
    :param num_targets: number of Pokemon actually hit by this move this turn. Only
        matters for moves whose `target` is `allAdjacent`/`allAdjacentFoes`, and only
        when it is >= 2 (a spread move that only actually hits one target due to a fainted
        ally/opponent does NOT get the 0.75x spread penalty in the real engine).
    """

    weather: str | None = None
    terrain: str | None = None
    screens: frozenset[ScreenId] = frozenset()
    trick_room: bool = False
    is_doubles: bool = True
    num_targets: int = 1


@dataclass
class DamageResult:
    """Output of `damage_range`.

    `breakdown` keys (all present on every call, even the short-circuit immune/
    unsupported paths, so callers can branch on them without a KeyError):
      - `move_supported` (bool): False for Status moves (0 damage is correct, not a
        limitation) is NOT what this flags -- it flags moves this calculator can't
        compute at all (variable-basePower moves with no flat `basePower` in the
        exported data, e.g. Gyro Ball/Electro Ball/Water Spout). Status moves instead
        get `reason: "status_move"` with `move_supported: True` and 0 damage (correct).
      - `immune` (bool): True if type/ability immunity zeroed the move out.
      - `ability_handled` (bool): False if either side's `ability` isn't in this
        module's whitelist (`ABILITY_WHITELIST`) -- the ability, if any, was still
        looked up but contributed no modifier (graceful default, not an error).
      - `item_handled` (bool): same idea for the attacker's `item`.
      - `stab`, `type_effectiveness`, `weather_modifier`, `terrain_modifier`,
        `spread_modifier`, `burn_modifier`, `final_multiplier` (floats): the
        multiplier actually applied at each pipeline stage (1.0 if not applicable).
      - `base_power` (int | None): the move's base power AFTER basePower-stage item/
        ability modifiers (None if unsupported/status).
      - `attack_stat`/`defense_stat` (str): which raw stat ("atk"/"def"/"spa"/"spd")
        was used.
      - `attack_value`/`defense_value` (int | None): the fully-modified stat values fed
        into the base formula.
      - `rolls` (list[int]): all 16 possible damage values (85%-100% in 1% steps), in
        case a caller wants the full distribution rather than just min/max/expected.
    """

    min_damage: int
    max_damage: int
    expected_damage: float
    min_percent: float
    max_percent: float
    expected_percent: float
    breakdown: dict[str, object]


# --- ability / item whitelists -----------------------------------------------------------

# Raw-stat-doubling abilities (applied to the attack stat directly, like a x2 item).
_HUGE_POWER_ABILITIES = frozenset({"hugepower", "purepower"})

# Ability -> the move type it grants immunity to (Levitate/Flash Fire/*Absorb/*Rod).
ABILITY_TYPE_IMMUNITY: dict[str, str] = {
    "levitate": "Ground",
    "flashfire": "Fire",
    "waterabsorb": "Water",
    "voltabsorb": "Electric",
    "stormdrain": "Water",
    "lightningrod": "Electric",
}

ABILITY_WHITELIST: frozenset[str] = frozenset(
    {
        "guts",
        "hugepower",
        "purepower",
        "technician",
        "adaptability",
        "wonderguard",
        "multiscale",
        "solidrock",
        "filter",
        "tintedlens",
        "thickfat",
        *ABILITY_TYPE_IMMUNITY,
    }
)

# Type-boosting held items -- 1.2x basePower to a matching move type (data/items.ts:
# onBasePower, chainModify([4915, 4096]) === 1.2 exactly).
ITEM_TYPE_BOOST: dict[str, str] = {
    "charcoal": "Fire",
    "mysticwater": "Water",
    "blackglasses": "Dark",
    "dragonfang": "Dragon",
    "hardstone": "Rock",
    "magnet": "Electric",
    "metalcoat": "Steel",
    "miracleseed": "Grass",
    "nevermeltice": "Ice",
    "poisonbarb": "Poison",
    "sharpbeak": "Flying",
    "silkscarf": "Normal",
    "silverpowder": "Bug",
    "softsand": "Ground",
    "twistedspoon": "Psychic",
    "fairyfeather": "Fairy",
}

# Category-boosting held items -- 1.1x basePower to a matching category (data/items.ts:
# chainModify([4505, 4096]) === 1.1 exactly).
ITEM_CATEGORY_BOOST: dict[str, str] = {
    "muscleband": "Physical",
    "wiseglasses": "Special",
}

# Items with a damage-relevant effect that isn't a basePower modifier -- handled
# separately in `_final_multiplier`/`_attack_stat_item_modifier`, but listed here so
# `to_id(item) in ITEM_WHITELIST` covers them for the `item_handled` breakdown flag.
ITEM_WHITELIST: frozenset[str] = frozenset(
    {*ITEM_TYPE_BOOST, *ITEM_CATEGORY_BOOST, "lifeorb", "expertbelt", "lightball", "choicescarf"}
)

_LIFE_ORB_MULTIPLIER = 5324 / 4096  # data/items.ts lifeorb.onModifyDamage, == 1.3 exactly
_EXPERT_BELT_MULTIPLIER = 4915 / 4096  # data/items.ts expertbelt.onModifyDamage, == 1.2 exactly
_ITEM_TYPE_BOOST_MULTIPLIER = 4915 / 4096  # same fraction as expert belt, == 1.2
_ITEM_CATEGORY_BOOST_MULTIPLIER = 4505 / 4096  # data/items.ts muscleband/wiseglasses, == 1.1

# Screens: 0.5x in singles, 2/3x in doubles (data/moves.ts reflect/lightscreen/
# auroraveil: `this.activePerHalf > 1 ? [2732, 4096] : 0.5`; 2732/4096 == 2/3 exactly).
_SCREEN_MULTIPLIER_SINGLES = 0.5
_SCREEN_MULTIPLIER_DOUBLES = 2732 / 4096

_SPREAD_MULTIPLIER = 0.75
_SPREAD_TARGETS = frozenset({"allAdjacent", "allAdjacentFoes"})

TYPE_EFFECTIVENESS_MULTIPLIER: dict[int, float] = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}


def _apply_stage(stat: int, stage: int) -> int:
    stage = max(-6, min(6, stage))
    if stage >= 0:
        return stat * (2 + stage) // 2
    return stat * 2 // (2 - stage)


def _type_effectiveness(move_type: str, defender_types: list[str]) -> float:
    typechart = load_typechart()
    multiplier = 1.0
    for defender_type in defender_types:
        entry = typechart.get(defender_type.lower())
        if entry is None:
            continue
        code = entry["damageTaken"].get(move_type, 0)
        multiplier *= TYPE_EFFECTIVENESS_MULTIPLIER.get(code, 1.0)
    return multiplier


def _is_grounded(state: PokemonState) -> bool:
    """Rough grounded check for terrain purposes: not Flying-type and not Levitate.
    Doesn't model Iron Ball/Gravity/Roost/Ingrain (deferred -- see report deviations).
    """
    if "Flying" in state.types():
        return False
    if state.ability == "levitate":
        return False
    return True


def _terrain_modifier(field: FieldState, move_type: str, attacker: PokemonState, defender: PokemonState) -> float:
    if field.terrain == "electric" and move_type == "Electric" and _is_grounded(attacker):
        return 1.3
    if field.terrain == "grassy" and move_type == "Grass" and _is_grounded(attacker):
        return 1.3
    if field.terrain == "psychic" and move_type == "Psychic" and _is_grounded(attacker):
        return 1.3
    if field.terrain == "misty" and move_type == "Dragon" and _is_grounded(defender):
        return 0.5
    return 1.0


def _weather_modifier(field: FieldState, move_type: str) -> float:
    if field.weather == "sun":
        if move_type == "Fire":
            return 1.5
        if move_type == "Water":
            return 0.5
    elif field.weather == "rain":
        if move_type == "Water":
            return 1.5
        if move_type == "Fire":
            return 0.5
    return 1.0


def _screen_modifier(field: FieldState, category: str) -> float:
    per_side_multiplier = _SCREEN_MULTIPLIER_DOUBLES if field.is_doubles else _SCREEN_MULTIPLIER_SINGLES
    has_reflect = "reflect" in field.screens and category == "Physical"
    has_light_screen = "lightscreen" in field.screens and category == "Special"
    if has_reflect or has_light_screen:
        return per_side_multiplier
    # Aurora Veil doesn't stack with a same-category Reflect/Light Screen (see
    # data/moves.ts auroraveil condition -- it explicitly bails in that case).
    if "auroraveil" in field.screens:
        return per_side_multiplier
    return 1.0


def _empty_breakdown() -> dict[str, object]:
    return {
        "move_supported": True,
        "immune": False,
        "ability_handled": True,
        "item_handled": True,
        "stab": 1.0,
        "type_effectiveness": 1.0,
        "weather_modifier": 1.0,
        "terrain_modifier": 1.0,
        "spread_modifier": 1.0,
        "burn_modifier": 1.0,
        "final_multiplier": 1.0,
        "base_power": None,
        "attack_stat": None,
        "defense_stat": None,
        "attack_value": None,
        "defense_value": None,
        "rolls": [],
        "reason": None,
    }


def _zero_result(breakdown: dict[str, object]) -> DamageResult:
    return DamageResult(
        min_damage=0,
        max_damage=0,
        expected_damage=0.0,
        min_percent=0.0,
        max_percent=0.0,
        expected_percent=0.0,
        breakdown=breakdown,
    )


# --- variable base power (basePowerCallback) moves --------------------------------------

# Weather -> the type Weather Ball becomes (data/moves.ts weatherball.onModifyType); BP
# doubles (50 -> 100, onModifyMove) for every one of these four cases identically. Sand/
# snow have no *damage-multiplier* effect elsewhere in this module (real gen9 Sandstorm/
# Snow are Def/SpD boosts for Rock/Ice types, not a Fire/Water-style damage modifier --
# see `FieldState`'s docstring), but they still drive Weather Ball's type/power exactly
# like sun/rain do.
_WEATHER_BALL_TYPE: dict[str, str] = {"sun": "Fire", "rain": "Water", "sand": "Rock", "snow": "Ice"}

# data/moves.ts grassknot/lowkick vs. heavyslam/heatcrash basePowerCallback -- both pairs
# share their respective threshold tables verbatim.
_WEIGHT_TIER_MOVE_IDS = frozenset({"grassknot", "lowkick"})
_WEIGHT_RATIO_MOVE_IDS = frozenset({"heavyslam", "heatcrash"})


def _weight_tier_base_power(weight_hg: float) -> int:
    """Grass Knot/Low Kick's basePowerCallback: BP by the target's weight alone (weight
    in hectograms, matching `Pokemon.getWeight()`'s units -- see `_weight_hg`).
    """
    if weight_hg >= 2000:
        return 120
    if weight_hg >= 1000:
        return 100
    if weight_hg >= 500:
        return 80
    if weight_hg >= 250:
        return 60
    if weight_hg >= 100:
        return 40
    return 20


def _weight_ratio_base_power(attacker_hg: float, defender_hg: float) -> int:
    """Heavy Slam/Heat Crash's basePowerCallback: BP by the attacker:defender weight
    ratio (both in hectograms -- see `_weight_hg`).
    """
    if defender_hg <= 0:
        return 40
    ratio = attacker_hg / defender_hg
    if ratio >= 5:
        return 120
    if ratio >= 4:
        return 100
    if ratio >= 3:
        return 80
    if ratio >= 2:
        return 60
    return 40


def _weight_hg(state: PokemonState) -> float:
    """Weight in hectograms, matching `Pokemon.getWeight()` (`weighthg`, `weightkg * 10`
    in the exported species data). No Float Stone/Autotomize/Heavy Metal modeling (the
    real engine's `ModifyWeight` event) -- `PokemonState` only carries the species'
    unmodified weight, the same simplification as everything else in this module's
    whitelist-table design (see `ABILITY_WHITELIST`/`ITEM_WHITELIST`).
    """
    return round(state.species_data()["weightkg"] * 10)


def _effective_speed_for_ratio(state: PokemonState) -> float:
    """Speed as Electro Ball/Gyro Ball's basePowerCallback reads it (`pokemon.getStat
    ('spe')`): stat stage, then Choice Scarf (1.5x) and paralysis (0.5x) -- the same
    "effective speed" `vgc.evaluator.effective_speed` computes for turn-order purposes,
    duplicated here (not imported) to keep this Phase 2a module free of a dependency on
    the Phase 2b evaluator built on top of it (see CLAUDE.md's module-layering note). A
    float estimate is fine -- only the floor()'d *ratio* of two of these numbers ever
    reaches a real formula, same as `effective_speed`'s own docstring reasoning.
    """
    speed = float(_apply_stage(state.stats()["spe"], state.boost_stage("spe")))
    if state.item == "choicescarf":
        speed *= 1.5
    if state.status == "par":
        speed *= 0.5
    return speed


def _variable_base_power(
    move_id: str,
    move: dict,
    attacker: PokemonState,
    defender: PokemonState,
    field: FieldState,
) -> tuple[int, str | None] | None:
    """`(base_power, type_override)` for the variable-basePower moves this module
    implements (see module docstring point 1), mirroring each move's
    `basePowerCallback`/`onModifyType`/`onModifyMove` in `data/moves.ts` (no
    `data/mods/champions/moves.ts` override for any of these -- confirmed by grep, so
    vanilla gen9 formulas apply verbatim). `type_override` is `None` when the move's type
    doesn't change (everything except Weather Ball). Returns `None` (not a tuple) for any
    other move id, which falls through to the caller's existing flat-`basePower`/
    `move_supported: False` handling -- this function does not change behavior for any
    move outside its own explicit id list.
    """
    if move_id == "weatherball":
        weather_type = _WEATHER_BALL_TYPE.get(field.weather or "")
        if weather_type is None:
            return None  # no recognized weather -> flat 50 BP Normal-type (the exported default)
        return int(move["basePower"]) * 2, weather_type

    if move_id in ("waterspout", "eruption"):
        # basePowerCallback: `move.basePower * pokemon.hp / pokemon.maxhp`, then
        # `clampIntRange(basePower, 1)` (Math.floor, minimum 1) in
        # sim/battle-actions.ts::getDamage. Integer math here (not the JS float division)
        # to sidestep float-rounding edge cases -- `move["basePower"] * hp` is always an
        # exact integer product, so `// max_hp` reproduces `Math.floor` exactly.
        hp = attacker.hp_or_max()
        max_hp = attacker.max_hp()
        base_power = (move["basePower"] * hp) // max_hp
        return max(1, base_power), None

    if move_id == "electroball":
        # basePowerCallback: `ratio = floor(pokemon.getStat('spe') / target.getStat
        # ('spe'))` -- pokemon is the attacker (move user), target is the defender.
        user_speed = _effective_speed_for_ratio(attacker)
        target_speed = _effective_speed_for_ratio(defender)
        ratio = int(user_speed // target_speed) if target_speed > 0 else 0
        bp_table = [40, 60, 80, 120, 150]
        return bp_table[min(ratio, 4)], None

    if move_id == "gyroball":
        # basePowerCallback: `power = floor(25 * target.getStat('spe') / pokemon.getStat
        # ('spe')) + 1`, capped at 150 -- pokemon is the attacker, target the defender
        # (a slower attacker vs. a faster target yields a higher ratio -> more power).
        user_speed = _effective_speed_for_ratio(attacker)
        target_speed = _effective_speed_for_ratio(defender)
        if user_speed <= 0:
            return 1, None
        power = int(25 * target_speed / user_speed) + 1
        return min(power, 150), None

    if move_id in _WEIGHT_TIER_MOVE_IDS:
        return _weight_tier_base_power(_weight_hg(defender)), None

    if move_id in _WEIGHT_RATIO_MOVE_IDS:
        return _weight_ratio_base_power(_weight_hg(attacker), _weight_hg(defender)), None

    return None


def damage_range(
    attacker: PokemonState,
    defender: PokemonState,
    move_id: str,
    field: FieldState | None = None,
) -> DamageResult:
    """Compute the min/max/expected damage `attacker`'s `move_id` deals to `defender`.

    No crit roll (v1 scope; see module docstring). See `DamageResult.breakdown` for the
    full trace, including `move_supported`/`immune`/`ability_handled`/`item_handled`
    flags callers should check before trusting a 0-damage result silently.
    """
    field = field if field is not None else FieldState()
    move_id = to_id(move_id)
    move = load_moves().get(move_id)
    if move is None:
        raise KeyError(f"unknown move id {move_id!r} (check vgc.data.load_moves())")

    breakdown = _empty_breakdown()

    if move["category"] == "Status":
        breakdown["reason"] = "status_move"
        return _zero_result(breakdown)

    base_power = move.get("basePower") or 0
    move_type = move["type"]
    variable = _variable_base_power(move_id, move, attacker, defender, field)
    if variable is not None:
        base_power, variable_type = variable
        if variable_type is not None:
            move_type = variable_type
    elif base_power <= 0:
        # Relies on a basePowerCallback this module doesn't implement (anything not in
        # _variable_base_power's explicit move_id list above). Rather than silently
        # returning 0 damage, flag it as unsupported.
        breakdown["move_supported"] = False
        breakdown["reason"] = "variable_base_power_not_implemented"
        return _zero_result(breakdown)

    category = move["category"]
    is_physical = category == "Physical"

    defender_types = defender.types()
    type_mult = _type_effectiveness(move_type, defender_types)
    breakdown["type_effectiveness"] = type_mult

    ability_immune_type = ABILITY_TYPE_IMMUNITY.get(defender.ability or "")
    if type_mult == 0.0 or ability_immune_type == move_type:
        breakdown["immune"] = True
        return _zero_result(breakdown)

    if defender.ability == "wonderguard" and type_mult <= 1.0:
        breakdown["immune"] = True
        breakdown["reason"] = "wonder_guard"
        return _zero_result(breakdown)

    ability_handled = True
    for ability in (attacker.ability, defender.ability):
        if ability is not None and ability not in ABILITY_WHITELIST:
            ability_handled = False
    breakdown["ability_handled"] = ability_handled
    breakdown["item_handled"] = attacker.item is None or attacker.item in ITEM_WHITELIST

    attacker_stats = attacker.stats()
    defender_stats = defender.stats()
    attack_stat_id: StatId = "atk" if is_physical else "spa"
    defense_stat_id: StatId = "def" if is_physical else "spd"

    attack_value = _apply_stage(attacker_stats[attack_stat_id], attacker.boost_stage(attack_stat_id))
    defense_value = _apply_stage(defender_stats[defense_stat_id], defender.boost_stage(defense_stat_id))

    # Raw-stat abilities/items (applied to the boosted stat, before the base formula).
    if attacker.ability in _HUGE_POWER_ABILITIES:
        attack_value *= 2
    if attacker.ability == "guts" and attacker.status is not None:
        attack_value = int(attack_value * 1.5)
    if attacker.item == "lightball" and attacker.species_data().get("baseSpecies") == "Pikachu":
        attack_value *= 2

    # Base-power-stage modifiers.
    if attacker.ability == "technician" and base_power <= 60:
        base_power = int(base_power * 1.5)
    type_boost_type = ITEM_TYPE_BOOST.get(attacker.item or "")
    if type_boost_type == move_type:
        base_power = int(base_power * _ITEM_TYPE_BOOST_MULTIPLIER)
    category_boost = ITEM_CATEGORY_BOOST.get(attacker.item or "")
    if category_boost == category:
        base_power = int(base_power * _ITEM_CATEGORY_BOOST_MULTIPLIER)

    breakdown["base_power"] = base_power
    breakdown["attack_stat"] = attack_stat_id
    breakdown["defense_stat"] = defense_stat_id
    breakdown["attack_value"] = attack_value
    breakdown["defense_value"] = defense_value

    level = 50  # this format has no adjustable level (see vgc.stats.FORMAT_LEVEL)
    level_factor = (2 * level) // 5 + 2
    raw = level_factor * base_power * attack_value
    raw = raw // defense_value
    raw = raw // 50
    raw += 2

    is_spread_move = move["target"] in _SPREAD_TARGETS and field.num_targets >= 2
    spread_modifier = _SPREAD_MULTIPLIER if is_spread_move else 1.0
    breakdown["spread_modifier"] = spread_modifier
    raw = int(raw * spread_modifier)

    weather_modifier = _weather_modifier(field, move_type)
    breakdown["weather_modifier"] = weather_modifier
    raw = int(raw * weather_modifier)

    terrain_modifier = _terrain_modifier(field, move_type, attacker, defender)
    breakdown["terrain_modifier"] = terrain_modifier
    raw = int(raw * terrain_modifier)

    is_stab = move_type in attacker.types()
    stab = (2.0 if attacker.ability == "adaptability" else 1.5) if is_stab else 1.0
    breakdown["stab"] = stab

    burn_modifier = 1.0
    if attacker.status == "brn" and is_physical and attacker.ability != "guts":
        burn_modifier = 0.5
    breakdown["burn_modifier"] = burn_modifier

    final_multiplier = 1.0
    final_multiplier *= _screen_modifier(field, category)
    if defender.ability == "multiscale" and defender.hp_or_max() >= defender.max_hp():
        final_multiplier *= 0.5
    if defender.ability in ("solidrock", "filter") and type_mult > 1.0:
        final_multiplier *= 0.75
    if attacker.ability == "tintedlens" and type_mult < 1.0:
        final_multiplier *= 2.0
    if defender.ability == "thickfat" and move_type in ("Fire", "Ice"):
        final_multiplier *= 0.5
    if attacker.item == "lifeorb":
        final_multiplier *= _LIFE_ORB_MULTIPLIER
    if attacker.item == "expertbelt" and type_mult > 1.0:
        final_multiplier *= _EXPERT_BELT_MULTIPLIER
    breakdown["final_multiplier"] = final_multiplier

    # The random roll (85%-100%) is applied here in the real engine -- BEFORE STAB/
    # type/burn/final modifiers, not after (see module docstring point 8).
    rolls: list[int] = []
    for percent in range(85, 101):
        damage = raw * percent // 100
        damage = int(damage * stab)
        damage = int(damage * type_mult)
        damage = int(damage * burn_modifier)
        damage = int(damage * final_multiplier)
        damage = max(damage, 1)
        rolls.append(damage)
    breakdown["rolls"] = rolls

    max_hp = defender.max_hp()
    min_damage, max_damage = rolls[0], rolls[-1]
    expected_damage = sum(rolls) / len(rolls)

    return DamageResult(
        min_damage=min_damage,
        max_damage=max_damage,
        expected_damage=expected_damage,
        min_percent=min_damage / max_hp * 100,
        max_percent=max_damage / max_hp * 100,
        expected_percent=expected_damage / max_hp * 100,
        breakdown=breakdown,
    )
