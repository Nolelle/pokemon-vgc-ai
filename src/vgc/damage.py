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
6b. Weather defence boosts, applied to the (staged) defending stat before the base formula,
   exactly where the real engine's ModifySpD/ModifyDef events run: Sandstorm gives Rock-type
   defenders 1.5x Sp. Def, Snow gives Ice-type defenders 1.5x Def (ground-truth tested in
   `tests/test_damage_ground_truth.py`).
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

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from vgc.config import DATA_DIR
from vgc.data import load_items, load_moves, load_species, load_typechart
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
    :param item_lost: the holder's item was consumed/removed this stint (Unburden trigger).
    """

    species_id: str
    sp_spread: dict[str, int] | None = None
    nature: str | None = None
    boosts: Boosts = field(default_factory=dict)
    status: str | None = None
    item: str | None = None
    ability: str | None = None
    current_hp: int | None = None
    # True once the holder has lost or consumed its item this stint on the field, so an
    # Unburden holder has its Speed doubled (data/abilities.ts unburden). Set only by the
    # state builders when `PolicyConfig.model_unburden` is on; inert for every other ability.
    item_lost: bool = False

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

    :param weather: `"sun"` | `"rain"` | `"sand"` | `"snow"` | None. Sun/rain scale Fire/Water
        damage; sand raises Rock-type defenders' Sp. Def and snow raises Ice-type defenders'
        Def by 1.5x (see `_weather_defense_value`). Cloud Nine/Air Lock are not modelled.
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
    # Abilities of the two partners, for effects that cross Pokemon (Friend Guard, Armor Tail /
    # Queenly Majesty / Dazzling priority blocks, Lightning Rod / Storm Drain redirection,
    # Steely Spirit, the Auras). None = unknown / none.
    attacker_ally_ability: str | None = None
    defender_ally_ability: str | None = None


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


# --- tables extracted from the Showdown engine --------------------------------------------------
# Everything below mirrors `data/abilities.ts` / `data/items.ts` / `data/conditions.ts` /
# `data/moves.ts` of the local checkout (champions mod does not override these). Each modifier is
# the engine's own 4096ths numerator and the priority is its `on*Priority`, so the chained
# fixed-point rounding is reproduced exactly (`tests/test_damage_ground_truth.py` and the full
# sweep in `offline/sweep_calculator_vs_engine.py` are the bar).

# Raw-stat-doubling abilities (ModifyAtk, x2, physical only).
_HUGE_POWER_ABILITIES = frozenset({"hugepower", "purepower"})

# Ability -> the move type it grants immunity to (absorbing abilities, Levitate).
ABILITY_TYPE_IMMUNITY: dict[str, str] = {
    "levitate": "Ground",
    "eelevate": "Ground",
    "flashfire": "Fire",
    "waterabsorb": "Water",
    "voltabsorb": "Electric",
    "stormdrain": "Water",
    "lightningrod": "Electric",
    "sapsipper": "Grass",
    "dryskin": "Water",
    "eartheater": "Ground",
    "motordrive": "Electric",
}
# Ability -> the move flag it is immune to.
ABILITY_FLAG_IMMUNITY: dict[str, str] = {"soundproof": "sound", "bulletproof": "bullet"}
# Abilities that stop the opposing side's positive-priority moves from reaching the holder.
_PRIORITY_BLOCK_ABILITIES = frozenset({"armortail", "queenlymajesty", "dazzling"})
_MOLD_BREAKER_ABILITIES = frozenset({"moldbreaker", "teravolt", "turboblaze"})
_WEATHER_NEGATING_ABILITIES = frozenset({"cloudnine", "airlock"})

# "-ate" abilities: Normal-type moves become the ability's type and get x4915/4096 base power
# (data/abilities.ts aerilate/pixilate/refrigerate/galvanize/dragonize; the Champions mod
# only un-hides Dragonize). Applied in ModifyType (before immunity/effectiveness/STAB), so
# the changed type is the move's type for the rest of the pipeline. The excluded moves keep
# their Normal type (data/abilities.ts `noModifyType`).
_ATE_ABILITY_TYPE: dict[str, str] = {
    "aerilate": "Flying",
    "pixilate": "Fairy",
    "refrigerate": "Ice",
    "galvanize": "Electric",
    "dragonize": "Dragon",
}
_ATE_EXCLUDED_MOVES = frozenset(
    {
        "judgment",
        "multiattack",
        "naturalgift",
        "revelationdance",
        "technoblast",
        "terrainpulse",
        "weatherball",
    }
)
# Liquid Voice: sound moves become Water (no power boost).
_LIQUID_VOICE_TYPE = "Water"

# Moves that read another stat than their category's default (data/moves.ts override*
# fields; the data export does not carry them). Body Press attacks with the USER's Defense
# (and Defense stage); Foul Play with the TARGET's Attack (and the target's Attack stage);
# Psyshock/Psystrike/Secret Sword are Special attacks aimed at the target's Defense.
# Wonder Room swaps are not modelled.
_OFFENSIVE_STAT_OVERRIDE: dict[str, StatId] = {"bodypress": "def"}
_OFFENSIVE_FROM_TARGET = frozenset({"foulplay"})
_DEFENSIVE_STAT_OVERRIDE: dict[str, StatId] = {
    "psyshock": "def",
    "psystrike": "def",
    "secretsword": "def",
}
# Moves that ignore the target's defensive stat stages (`ignoreDefensive`).
_IGNORE_DEFENSIVE_BOOSTS = frozenset({"darkestlariat", "sacredsword"})
# Moves that always land a critical hit (`willCrit: true`): x1.5 after the weather step, the
# target's positive Defense stages and the user's negative Attack stages ignored, screens ignored.
_ALWAYS_CRIT_MOVES = frozenset({"flowertrick", "frostbreath", "stormthrow", "surgingstrikes", "wickedblow"})
# Moves that shatter Reflect / Light Screen / Aurora Veil before they hit.
_SCREEN_BREAKERS = frozenset({"brickbreak", "psychicfangs", "ragingbull"})
# Moves that raise the user's Special Attack by one stage before they deal damage.
_SELF_SPA_PLUS_ONE = frozenset({"electroshot", "meteorbeam"})
# Moves that count as recoil / crash-damage moves for Reckless.
_RECOIL_MOVES = frozenset(
    {
        "bravebird", "doubleedge", "flareblitz", "headsmash", "lightofruin", "volttackle",
        "wavecrash", "wildcharge", "woodhammer", "highjumpkick", "supercellslam", "axekick",
    }
)
# `hasSheerForceBoost` moves get Sheer Force's x1.3 although they have no secondary to remove.
_SHEER_FORCE_BOOST_MOVES = frozenset({"electroshot", "ragingfury"})

# Type-boosting held items -- onBasePower priority 15, chainModify([4915, 4096]).
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
    "blackbelt": "Fighting",
    "spelltag": "Ghost",
}
# Category-boosting held items -- onBasePower priority 16, chainModify([4505, 4096]).
ITEM_CATEGORY_BOOST: dict[str, str] = {
    "muscleband": "Physical",
    "wiseglasses": "Special",
}
# Resist berries: the type they halve (the Pokemon must be hit super effectively; Chilan Berry
# halves any Normal hit). Eaten on the hit, so the first hit is the one that matters.
ITEM_RESIST_BERRY: dict[str, str] = {
    "occaberry": "Fire", "passhoberry": "Water", "wacanberry": "Electric", "rindoberry": "Grass",
    "yacheberry": "Ice", "chopleberry": "Fighting", "kebiaberry": "Poison", "shucaberry": "Ground",
    "cobaberry": "Flying", "payapaberry": "Psychic", "tangaberry": "Bug", "chartiberry": "Rock",
    "kasibberry": "Ghost", "habanberry": "Dragon", "colburberry": "Dark", "babiriberry": "Steel",
    "roseliberry": "Fairy", "chilanberry": "Normal",
}
# Terrain-boosted seeds only change stats after the terrain starts; no damage effect here.
ITEM_NO_DAMAGE_EFFECT_LISTED = frozenset(
    {
        "airballoon", "bigroot", "bindingband", "brightpowder", "damprock", "ejectbutton",
        "electricseed", "focusband", "focussash", "grassyseed", "heatrock", "icyrock",
        "kingsrock", "leek", "leftovers", "leppaberry", "lightclay", "lumberry", "mentalherb",
        "metronome", "mistyseed", "oranberry", "persimberry", "psychicseed", "quickclaw",
        "redcard", "rockyhelmet", "scopelens", "shedshell", "shellbell", "sitrusberry",
        "smoothrock", "terrainextender", "whiteherb", "widelens", "zoomlens", "aspearberry",
        "cheriberry", "chestoberry", "pechaberry", "rawstberry", "choicescarf", "ironball",
    }
)

# Abilities this module reads (an ability outside this set contributes nothing to damage).
ABILITY_WHITELIST: frozenset[str] = frozenset(
    {
        *_ATE_ABILITY_TYPE, "liquidvoice", "guts", "hugepower", "purepower", "technician",
        "adaptability", "wonderguard", "multiscale", "solidrock", "filter", "tintedlens",
        "thickfat", *ABILITY_TYPE_IMMUNITY, *ABILITY_FLAG_IMMUNITY, *_PRIORITY_BLOCK_ABILITIES,
        *_MOLD_BREAKER_ABILITIES, *_WEATHER_NEGATING_ABILITIES, "toughclaws", "sheerforce",
        "ironfist", "sharpness", "megalauncher", "strongjaw", "reckless", "punkrock",
        "fairyaura", "darkaura", "aurabreak", "sandforce", "heatproof", "purifyingsalt",
        "waterbubble", "furcoat", "marvelscale", "grasspelt", "fluffy", "hustle", "blaze",
        "overgrow", "torrent", "swarm", "solarpower", "firemane", "steelworker", "transistor",
        "dragonsmaw", "protean", "libero", "unaware", "friendguard", "steelyspirit", "longreach",
        "megasol", "scrappy", "infiltrator", "ripen", "levitate",
    }
)
ITEM_WHITELIST: frozenset[str] = frozenset(
    {
        *ITEM_TYPE_BOOST, *ITEM_CATEGORY_BOOST, *ITEM_RESIST_BERRY, "lifeorb", "expertbelt",
        "lightball", "choicescarf", "normalgem", "airballoon", "ironball",
        *ITEM_NO_DAMAGE_EFFECT_LISTED,
    }
)

# --- fixed-point helpers (sim/battle.ts modify / chainModify) ------------------------------------

_Q = 4096


def _modify(value: int, modifier: int) -> int:
    """``battle.modify(value, modifier / 4096)``: round half DOWN in 4096ths."""
    return (value * modifier + 2047) >> 12


def _chain_mod(previous: int, numerator: int) -> int:
    """``battle.chainModify`` on a modifier held in 4096ths."""
    return (previous * numerator + 2048) >> 12


class _Chain:
    """One engine event's ``chainModify`` calls, applied in the engine's handler order
    (higher ``on*Priority`` first, then registration order)."""

    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: list[tuple[int, int, int]] = []

    def add(self, priority: int, numerator: int) -> None:
        self.items.append((-priority, len(self.items), numerator))

    def modifier(self) -> int:
        modifier = _Q
        for _priority, _index, numerator in sorted(self.items):
            modifier = _chain_mod(modifier, numerator)
        return modifier

    def apply(self, value: int) -> int:
        if not self.items:
            return value
        return _modify(value, self.modifier())


TYPE_EFFECTIVENESS_MULTIPLIER: dict[int, float] = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}

# Screens: 0.5x in singles, 2732/4096 in doubles (data/moves.ts reflect/lightscreen/auroraveil).
_SCREEN_NUMERATOR_SINGLES = 2048
_SCREEN_NUMERATOR_DOUBLES = 2732

_SPREAD_TARGETS = frozenset({"allAdjacent", "allAdjacentFoes"})


def _apply_stage(stat: int, stage: int) -> int:
    stage = max(-6, min(6, stage))
    if stage >= 0:
        return stat * (2 + stage) // 2
    return stat * 2 // (2 - stage)


def _type_code(move_type: str, defender_type: str) -> int:
    entry = load_typechart().get(defender_type.lower())
    if entry is None:
        return 0
    return entry["damageTaken"].get(move_type, 0)


def _type_effectiveness(move_type: str, defender_types: list[str]) -> float:
    multiplier = 1.0
    for defender_type in defender_types:
        multiplier *= TYPE_EFFECTIVENESS_MULTIPLIER.get(_type_code(move_type, defender_type), 1.0)
    return multiplier


def _is_grounded(state: PokemonState) -> bool:
    """Grounded check for terrain / Ground-move purposes: not Flying-type, not Levitate, not
    holding an Air Balloon -- unless it holds an Iron Ball. Gravity/Roost/Ingrain are not modelled.
    """
    if state.item == "ironball" and not state.item_lost:
        return True
    if "Flying" in state.types():
        return False
    if state.ability in ("levitate", "eelevate"):
        return False
    if state.item == "airballoon" and not state.item_lost:
        return False
    return True


def _has_item(state: PokemonState) -> bool:
    return bool(state.item) and not state.item_lost


def _positive_boosts(state: PokemonState) -> int:
    return sum(stage for stage in state.boosts.values() if stage > 0)


# Status-curing berries eat themselves the moment the status lands, so the holder never carries it
# into a hit; terrain seeds raise a stat once their terrain is up.
_STATUS_CURE_BERRY: dict[str, frozenset[str]] = {
    "cheriberry": frozenset({"par"}),
    "chestoberry": frozenset({"slp"}),
    "pechaberry": frozenset({"psn", "tox"}),
    "rawstberry": frozenset({"brn"}),
    "aspearberry": frozenset({"frz"}),
    "lumberry": frozenset({"brn", "par", "psn", "tox", "slp", "frz"}),
}
_TERRAIN_SEED: dict[str, tuple[str, str]] = {
    "electricseed": ("electric", "def"),
    "grassyseed": ("grassy", "def"),
    "mistyseed": ("misty", "spd"),
    "psychicseed": ("psychic", "spd"),
}


def _with_item_effects(state: PokemonState, field: FieldState) -> PokemonState:
    """``state`` as the hit sees it after its berry / terrain seed has already triggered."""
    if state.item is None or state.item_lost:
        return state
    from dataclasses import replace

    cures = _STATUS_CURE_BERRY.get(state.item)
    seed = _TERRAIN_SEED.get(state.item)
    if cures is not None and state.status in cures:
        return replace(state, status=None, item_lost=True)  # eaten: no longer holds an item
    if seed is not None and field.terrain == seed[0]:
        boosts = dict(state.boosts)
        boosts[seed[1]] = min(6, boosts.get(seed[1], 0) + 1)
        return replace(state, boosts=boosts, item_lost=True)
    if state.item in ("sitrusberry", "oranberry") and state.hp_or_max() * 2 <= state.max_hp():
        return replace(state, item_lost=True)  # eaten at half HP (the heal itself is not modelled)
    return state


def partner_ability(state: PokemonState | None) -> str | None:
    """The ability a standing partner contributes to damage (None when absent or fainted)."""
    if state is None or state.hp_or_max() <= 0:
        return None
    return state.ability


def with_partners(
    field: FieldState, attacker_partner: PokemonState | None, defender_partner: PokemonState | None
) -> FieldState:
    """``field`` carrying both partners' abilities (Friend Guard, Armor Tail / Queenly Majesty,
    Lightning Rod / Storm Drain redirection, Steely Spirit, the Auras)."""
    from dataclasses import replace

    return replace(
        field,
        attacker_ally_ability=partner_ability(attacker_partner),
        defender_ally_ability=partner_ability(defender_partner),
    )


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
        "move_type": None,
        "type_changed_from": None,
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


@lru_cache(maxsize=1)
def _breakable_abilities() -> frozenset[str]:
    """Abilities Mold Breaker (and Teravolt / Turboblaze) bypass: the engine's ``breakable`` flag."""
    path = DATA_DIR / "mechanics_catalog.json"
    try:
        with path.open() as file:
            catalog = json.load(file)
        return frozenset(catalog["mechanics_surface"]["ability_flags"]["breakable"])
    except (OSError, KeyError, ValueError):
        return frozenset()


# --- weather / terrain -------------------------------------------------------------------------------


def _effective_weather(field: FieldState, attacker: PokemonState, defender: PokemonState) -> str | None:
    for ability in (
        attacker.ability, defender.ability, field.attacker_ally_ability, field.defender_ally_ability
    ):
        if ability in _WEATHER_NEGATING_ABILITIES:
            return None
    return field.weather


def _weather_defense_value(
    defense_value: int, weather: str | None, defender: PokemonState, defense_stat_id: str
) -> int:
    """Sandstorm: Rock-type Sp. Def x1.5; Snow: Ice-type Def x1.5 (data/conditions.ts).

    The engine applies it with ``battle.modify(stat, 1.5)`` -- 4096-based fixed point whose
    rounding sends exact halves DOWN. Runs on the already-staged stat, before the base formula.
    """

    if weather == "sand" and defense_stat_id == "spd" and "Rock" in defender.types():
        return _modify(defense_value, 6144)
    if weather == "snow" and defense_stat_id == "def" and "Ice" in defender.types():
        return _modify(defense_value, 6144)
    return defense_value


# --- variable base power (basePowerCallback) moves --------------------------------------

# Weather -> the type Weather Ball becomes (data/moves.ts weatherball.onModifyType); BP
# doubles (50 -> 100, onModifyMove) for every one of these four cases identically.
_WEATHER_BALL_TYPE: dict[str, str] = {"sun": "Fire", "rain": "Water", "sand": "Rock", "snow": "Ice"}
_TERRAIN_PULSE_TYPE: dict[str, str] = {
    "electric": "Electric",
    "grassy": "Grass",
    "misty": "Fairy",
    "psychic": "Psychic",
}

# data/moves.ts grassknot/lowkick vs. heavyslam/heatcrash basePowerCallback.
_WEIGHT_TIER_MOVE_IDS = frozenset({"grassknot", "lowkick"})
_WEIGHT_RATIO_MOVE_IDS = frozenset({"heavyslam", "heatcrash"})

# Fixed-damage moves: damage regardless of stats (`damage: 'level'`, `damageCallback`).
_LEVEL_DAMAGE_MOVES = frozenset({"seismictoss", "nightshade"})
_HALF_TARGET_HP_MOVES = frozenset({"superfang", "naturesmadness", "ruination"})
_OHKO_MOVES = frozenset({"fissure", "guillotine", "horndrill", "sheercold"})


def _weight_tier_base_power(weight_hg: float) -> int:
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
    """Weight in hectograms (`weightkg * 10`). No Float Stone/Autotomize/Heavy Metal modelling."""
    weight = round(state.species_data()["weightkg"] * 10)
    if state.ability == "heavymetal":
        weight *= 2
    elif state.ability == "lightmetal":
        weight //= 2
    return weight


def _effective_speed_for_ratio(state: PokemonState) -> float:
    """Speed as Electro Ball/Gyro Ball's basePowerCallback reads it (`pokemon.getStat('spe')`):
    stat stage, then Choice Scarf (1.5x), Iron Ball (0.5x) and paralysis (0.5x) -- the same
    "effective speed" `vgc.evaluator.effective_speed` computes for turn-order purposes,
    duplicated here to keep this Phase 2a module free of a dependency on the evaluator.
    """
    speed = float(_apply_stage(state.stats()["spe"], state.boost_stage("spe")))
    if state.item == "choicescarf":
        speed *= 1.5
    if state.item == "ironball":
        speed *= 0.5
    if state.status == "par":
        speed *= 0.5
    if state.item_lost and state.ability == "unburden":
        speed *= 2.0
    return speed


def _flail_base_power(attacker: PokemonState) -> int:
    ratio = max((attacker.hp_or_max() * 48) // attacker.max_hp(), 1)
    if ratio < 2:
        return 200
    if ratio < 5:
        return 150
    if ratio < 10:
        return 100
    if ratio < 17:
        return 80
    if ratio < 33:
        return 40
    return 20


def _hard_press_base_power(defender: PokemonState) -> int:
    hp, max_hp = defender.hp_or_max(), defender.max_hp()
    scaled = (hp * 4096) // max_hp
    return ((100 * (100 * scaled) + 2047) // 4096) // 100 or 1


def _variable_base_power(
    move_id: str,
    move: dict,
    attacker: PokemonState,
    defender: PokemonState,
    field: FieldState,
    weather: str | None,
) -> tuple[int, str | None] | None:
    """`(base_power, type_override)` for the basePowerCallback / onModifyMove moves this module
    implements, mirroring `data/moves.ts`. `type_override` is `None` when the move's type does
    not change. Returns `None` for any other move id (the caller then uses the flat power).
    """
    base = int(move.get("basePower") or 0)
    if move_id == "weatherball":
        weather_type = _WEATHER_BALL_TYPE.get(weather or "")
        if weather_type is None:
            return None
        return base * 2, weather_type

    if move_id == "terrainpulse":
        if field.terrain in _TERRAIN_PULSE_TYPE and _is_grounded(attacker):
            return base * 2, _TERRAIN_PULSE_TYPE[field.terrain]
        return None

    if move_id in ("waterspout", "eruption"):
        hp = attacker.hp_or_max()
        max_hp = attacker.max_hp()
        return max(1, (base * hp) // max_hp), None

    if move_id == "electroball":
        user_speed = _effective_speed_for_ratio(attacker)
        target_speed = _effective_speed_for_ratio(defender)
        ratio = int(user_speed // target_speed) if target_speed > 0 else 0
        return [40, 60, 80, 120, 150][min(ratio, 4)], None

    if move_id == "gyroball":
        user_speed = _effective_speed_for_ratio(attacker)
        target_speed = _effective_speed_for_ratio(defender)
        if user_speed <= 0:
            return 1, None
        return min(int(25 * target_speed / user_speed) + 1, 150), None

    if move_id in _WEIGHT_TIER_MOVE_IDS:
        return _weight_tier_base_power(_weight_hg(defender)), None

    if move_id in _WEIGHT_RATIO_MOVE_IDS:
        return _weight_ratio_base_power(_weight_hg(attacker), _weight_hg(defender)), None

    if move_id in ("flail", "reversal"):
        return _flail_base_power(attacker), None

    if move_id == "hardpress":
        return _hard_press_base_power(defender), None

    if move_id in ("powertrip", "storedpower"):
        return base + 20 * _positive_boosts(attacker), None

    if move_id == "acrobatics":
        return (base * 2 if not _has_item(attacker) else base), None

    if move_id in ("hex", "infernalparade"):
        if defender.status is not None or defender.ability == "comatose":
            return base * 2, None
        return base, None

    if move_id == "risingvoltage":
        if field.terrain == "electric" and _is_grounded(defender):
            return base * 2, None
        return base, None

    return None


def _move_bp_chain_effects(
    move_id: str,
    attacker: PokemonState,
    defender: PokemonState,
    field: FieldState,
    weather: str | None,
    chain: _Chain,
) -> None:
    """The moves' own ``onBasePower`` handlers (priority 0, so they run last in the BasePower chain)."""
    if move_id == "knockoff":
        if _has_item(defender) and not _holds_own_mega_stone(defender):
            chain.add(0, 6144)
    elif move_id == "facade":
        if attacker.status is not None and attacker.status != "slp":
            chain.add(0, 8192)
    elif move_id in ("venoshock", "barbbarrage"):
        if defender.status in ("psn", "tox"):
            chain.add(0, 8192)
    elif move_id == "brine":
        if defender.hp_or_max() * 2 <= defender.max_hp():
            chain.add(0, 8192)
    elif move_id == "expandingforce":
        if field.terrain == "psychic" and _is_grounded(attacker):
            chain.add(0, 6144)
    elif move_id == "mistyexplosion":
        if field.terrain == "misty" and _is_grounded(attacker):
            chain.add(0, 6144)
    elif move_id in ("solarbeam", "solarblade"):
        if weather in ("rain", "sand", "snow"):
            chain.add(0, 2048)


def _holds_own_mega_stone(state: PokemonState) -> bool:
    item = (load_items().get(state.item or "") or {}) if state.item else {}
    if not item.get("megaStone"):
        return False
    base = state.species_data().get("baseSpecies")
    return any(to_id(user) == to_id(base) or to_id(user) == state.species_id for user in item.get("itemUser") or [])


def _terrain_bp(field: FieldState, move_id: str, move_type: str, attacker: PokemonState, defender: PokemonState, chain: _Chain) -> float:
    """Terrain's ``onBasePower`` effects (priority 6). Returns the display multiplier."""
    terrain = field.terrain
    shown = 1.0
    if terrain == "electric" and move_type == "Electric" and _is_grounded(attacker):
        chain.add(6, 5325)
        shown = 1.3
    elif terrain == "grassy" and move_type == "Grass" and _is_grounded(attacker):
        chain.add(6, 5325)
        shown = 1.3
    elif terrain == "psychic" and move_type == "Psychic" and _is_grounded(attacker):
        chain.add(6, 5325)
        shown = 1.3
    elif terrain == "misty" and move_type == "Dragon" and _is_grounded(defender):
        chain.add(6, 2048)
        shown = 0.5
    if terrain == "grassy" and move_id in ("earthquake", "bulldoze", "magnitude") and _is_grounded(defender):
        chain.add(6, 2048)
        shown *= 0.5
    return shown


def damage_range(
    attacker: PokemonState,
    defender: PokemonState,
    move_id: str,
    field: FieldState | None = None,
) -> DamageResult:
    """Compute the min/max/expected damage `attacker`'s `move_id` deals to `defender`.

    No crit roll (v1 scope; see module docstring) -- except the few moves that always crit. See
    `DamageResult.breakdown` for the full trace, including `move_supported`/`immune`/
    `ability_handled`/`item_handled` flags callers should check before trusting a 0-damage result
    silently.
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

    attacker = _with_item_effects(attacker, field)
    defender = _with_item_effects(defender, field)
    weather = _effective_weather(field, attacker, defender)
    if attacker.ability == "megasol":
        weather = "sun"  # the holder's moves resolve as if in harsh sunlight

    move_type = move["type"]
    category = move["category"]
    flags = move.get("flags") or {}
    if move_id == "shellsidearm":
        # Physical (and contact) when the raw Attack would do more than the raw Sp. Atk.
        raw_attacker, raw_defender = attacker.stats(), defender.stats()
        atk = _apply_stage(raw_attacker["atk"], attacker.boost_stage("atk"))
        spa = _apply_stage(raw_attacker["spa"], attacker.boost_stage("spa"))
        def_ = _apply_stage(raw_defender["def"], defender.boost_stage("def"))
        spd = _apply_stage(raw_defender["spd"], defender.boost_stage("spd"))
        physical = (22 * 90 * atk // def_) // 50
        special = (22 * 90 * spa // spd) // 50
        if physical > special:
            category = "Physical"
            flags = {**flags, "contact": 1}
    base_power = int(move.get("basePower") or 0)
    variable = _variable_base_power(move_id, move, attacker, defender, field, weather)
    if variable is not None:
        base_power, variable_type = variable
        if variable_type is not None:
            move_type = variable_type
    fixed_damage = (
        move_id in _LEVEL_DAMAGE_MOVES or move_id in _HALF_TARGET_HP_MOVES or move_id in _OHKO_MOVES
    )
    if base_power <= 0 and not fixed_damage:
        breakdown["move_supported"] = False
        breakdown["reason"] = "variable_base_power_not_implemented"
        return _zero_result(breakdown)

    original_type = move_type
    ate_type = _ATE_ABILITY_TYPE.get(attacker.ability or "")
    ate_boosted = False
    if ate_type is not None and move_type == "Normal" and move_id not in _ATE_EXCLUDED_MOVES:
        move_type = ate_type
        ate_boosted = True
    elif attacker.ability == "liquidvoice" and flags.get("sound"):
        move_type = _LIQUID_VOICE_TYPE
    breakdown["move_type"] = move_type
    breakdown["type_changed_from"] = original_type if move_type != original_type else None

    # Defender-side abilities are bypassed by Mold Breaker and friends (the `breakable` flag).
    defender_ability = defender.ability
    if attacker.ability in _MOLD_BREAKER_ABILITIES and defender_ability in _breakable_abilities():
        defender_ability = None

    # --- type effectiveness and immunities -----------------------------------------------------
    defender_types = defender.types()
    type_steps = 0  # the engine's typeMod: +1 per super-effective factor, -1 per resisted one
    immune_by_type = False
    scrappy = attacker.ability == "scrappy"
    effect_pairs = [(move_type, defender_types)]
    if move_id == "flyingpress":
        effect_pairs.append(("Flying", defender_types))
    for pair_type, pair_defenders in effect_pairs:
        for defender_type in pair_defenders:
            code = _type_code(pair_type, defender_type)
            if move_id == "freezedry" and pair_type == move_type and defender_type == "Water":
                code = 1
            if code == 3 and scrappy and pair_type in ("Normal", "Fighting") and defender_type == "Ghost":
                code = 0
            if pair_type == "Ground" and defender_type == "Flying" and defender.item == "ironball":
                code = 0
            if code == 3:
                immune_by_type = True
            elif code == 1:
                type_steps += 1
            elif code == 2:
                type_steps -= 1
    type_mult = 0.0 if immune_by_type else 2.0**type_steps
    breakdown["type_effectiveness"] = type_mult

    ability_immune_type = ABILITY_TYPE_IMMUNITY.get(defender_ability or "")
    ungrounded = (
        "Flying" in defender_types
        and not (defender.item == "ironball" and not defender.item_lost)
    ) or defender_ability in ("levitate", "eelevate") or (
        defender.item == "airballoon" and not defender.item_lost
    )
    ground_immune = move_type == "Ground" and ungrounded and not (
        defender.item == "ironball" and not defender.item_lost
    )
    flag_immune = ABILITY_FLAG_IMMUNITY.get(defender_ability or "")
    if (
        immune_by_type
        or ability_immune_type == move_type
        or ground_immune
        or (flag_immune is not None and flags.get(flag_immune))
    ):
        breakdown["immune"] = True
        return _zero_result(breakdown)

    if defender_ability == "wonderguard" and type_mult <= 1.0:
        breakdown["immune"] = True
        breakdown["reason"] = "wonder_guard"
        return _zero_result(breakdown)

    ally_blocker = field.defender_ally_ability
    if attacker.ability in _MOLD_BREAKER_ABILITIES and ally_blocker in _breakable_abilities():
        ally_blocker = None
    if defender_ability in _PRIORITY_BLOCK_ABILITIES or ally_blocker in _PRIORITY_BLOCK_ABILITIES:
        from vgc.priority_rules import effective_priority

        at_full = attacker.hp_or_max() >= attacker.max_hp()
        if effective_priority(move, attacker.ability, at_full) > 0:
            breakdown["immune"] = True
            breakdown["reason"] = "priority_block"
            return _zero_result(breakdown)

    for redirector in (field.defender_ally_ability, field.attacker_ally_ability):
        if (
            (redirector == "lightningrod" and move_type == "Electric")
            or (redirector == "stormdrain" and move_type == "Water")
        ) and move["target"] in ("normal", "any", "adjacentFoe"):
            if attacker.ability not in _MOLD_BREAKER_ABILITIES:
                breakdown["immune"] = True
                breakdown["reason"] = "redirected_to_partner"
                return _zero_result(breakdown)

    ability_handled = True
    for ability in (attacker.ability, defender.ability):
        if ability is not None and ability not in ABILITY_WHITELIST:
            ability_handled = False
    breakdown["ability_handled"] = ability_handled
    breakdown["item_handled"] = attacker.item is None or attacker.item in ITEM_WHITELIST

    # --- fixed-damage moves ----------------------------------------------------------------------
    if fixed_damage:
        if move_id in _LEVEL_DAMAGE_MOVES:
            fixed = 50
        elif move_id in _OHKO_MOVES:
            if move_id == "sheercold" and "Ice" in defender_types:
                breakdown["immune"] = True
                return _zero_result(breakdown)
            fixed = defender.max_hp()
        else:
            fixed = max(1, defender.hp_or_max() // 2)
        breakdown["base_power"] = None
        breakdown["rolls"] = [fixed] * 16
        max_hp = defender.max_hp()
        return DamageResult(
            min_damage=fixed,
            max_damage=fixed,
            expected_damage=float(fixed),
            min_percent=fixed / max_hp * 100,
            max_percent=fixed / max_hp * 100,
            expected_percent=fixed / max_hp * 100,
            breakdown=breakdown,
        )

    # --- stats ------------------------------------------------------------------------------------------
    is_physical = category == "Physical"
    always_crit = move_id in _ALWAYS_CRIT_MOVES and defender_ability not in ("shellarmor", "battlearmor")
    attacker_stats = attacker.stats()
    defender_stats = defender.stats()
    attack_stat_id: StatId = _OFFENSIVE_STAT_OVERRIDE.get(move_id) or ("atk" if is_physical else "spa")
    defense_stat_id: StatId = _DEFENSIVE_STAT_OVERRIDE.get(move_id) or ("def" if is_physical else "spd")
    from_target = move_id in _OFFENSIVE_FROM_TARGET
    offensive_mon = defender if from_target else attacker
    offensive_stats = defender_stats if from_target else attacker_stats

    attack_stage = offensive_mon.boost_stage(attack_stat_id)
    if move_id in _SELF_SPA_PLUS_ONE and attack_stat_id == "spa":
        attack_stage = min(6, attack_stage + 1)
    defense_stage = defender.boost_stage(defense_stat_id)
    if defender_ability == "unaware" and offensive_mon is attacker:
        attack_stage = 0
    if attacker.ability == "unaware":
        defense_stage = 0
    if move_id in _IGNORE_DEFENSIVE_BOOSTS:
        defense_stage = 0
    if always_crit:
        attack_stage = max(attack_stage, 0)
        defense_stage = min(defense_stage, 0)
    attack_value = _apply_stage(offensive_stats[attack_stat_id], attack_stage)
    defense_value = _apply_stage(defender_stats[defense_stat_id], defense_stage)
    defense_value = _weather_defense_value(defense_value, weather, defender, defense_stat_id)

    # ModifyAtk / ModifyDef chains (ability priority 5-6, Light Ball 1).
    atk_chain = _Chain()
    ab = attacker.ability
    if ab in _HUGE_POWER_ABILITIES and is_physical:
        atk_chain.add(5, 8192)
    if ab == "guts" and attacker.status is not None and is_physical:
        atk_chain.add(5, 6144)
    if ab == "hustle" and is_physical:
        atk_chain.add(5, 6144)
    if ab == "solarpower" and not is_physical and weather == "sun":
        atk_chain.add(5, 6144)
    hp_third = attacker.hp_or_max() * 3 <= attacker.max_hp()
    if hp_third and (
        (ab == "blaze" and move_type == "Fire")
        or (ab == "overgrow" and move_type == "Grass")
        or (ab == "torrent" and move_type == "Water")
        or (ab == "swarm" and move_type == "Bug")
    ):
        atk_chain.add(5, 6144)
    if (ab == "firemane" and move_type == "Fire") or (ab == "steelworker" and move_type == "Steel") or (
        ab == "dragonsmaw" and move_type == "Dragon"
    ):
        atk_chain.add(5, 6144)
    if ab == "transistor" and move_type == "Electric":
        atk_chain.add(5, 5325)
    if ab == "waterbubble" and move_type == "Water":
        atk_chain.add(5, 8192)
    if attacker.item == "lightball" and attacker.species_data().get("baseSpecies") == "Pikachu":
        atk_chain.add(1, 8192)
    da = defender_ability
    if da == "thickfat" and move_type in ("Fire", "Ice"):
        atk_chain.add(6, 2048)
    if da == "heatproof" and move_type == "Fire":
        atk_chain.add(6 if is_physical else 5, 2048)
    if da == "purifyingsalt" and move_type == "Ghost":
        atk_chain.add(6 if is_physical else 5, 2048)
    if da == "waterbubble" and move_type == "Fire":
        atk_chain.add(5, 2048)
    attack_value = atk_chain.apply(attack_value)

    def_chain = _Chain()
    if da == "furcoat" and defense_stat_id == "def" and is_physical:
        def_chain.add(6, 8192)
    if da == "marvelscale" and defender.status is not None and defense_stat_id == "def":
        def_chain.add(6, 6144)
    if da == "grasspelt" and field.terrain == "grassy" and defense_stat_id == "def":
        def_chain.add(6, 6144)
    defense_value = def_chain.apply(defense_value)

    # --- base power chain ------------------------------------------------------------------------------
    bp_chain = _Chain()
    if ab == "technician" and base_power <= 60:
        bp_chain.add(30, 6144)
    if ab == "rivalry":
        pass  # gender is not tracked
    if ab in ("ironfist",) and flags.get("punch"):
        bp_chain.add(23, 4915)
    if ab == "reckless" and move_id in _RECOIL_MOVES:
        bp_chain.add(23, 4915)
    if ate_boosted:
        bp_chain.add(23, 4915)
    if ab == "sheerforce" and (move.get("secondaries") or move_id in _SHEER_FORCE_BOOST_MOVES):
        bp_chain.add(21, 5325)
    if ab == "toughclaws" and flags.get("contact") and ab != "longreach":
        bp_chain.add(21, 5325)
    if ab == "sandforce" and weather == "sand" and move_type in ("Rock", "Ground", "Steel"):
        bp_chain.add(21, 5325)
    if ab == "sharpness" and flags.get("slicing"):
        bp_chain.add(19, 6144)
    if ab == "megalauncher" and flags.get("pulse"):
        bp_chain.add(19, 6144)
    if ab == "strongjaw" and flags.get("bite"):
        bp_chain.add(19, 6144)
    if ab == "punkrock" and flags.get("sound"):
        bp_chain.add(7, 5325)
    if ab == "steelyspirit" and move_type == "Steel":
        bp_chain.add(22, 6144)
    if field.attacker_ally_ability == "steelyspirit" and move_type == "Steel":
        bp_chain.add(22, 6144)
    aura_breaker = "aurabreak" in (attacker.ability, defender.ability, field.defender_ally_ability, field.attacker_ally_ability)
    for aura, aura_type in (("fairyaura", "Fairy"), ("darkaura", "Dark")):
        if move_type == aura_type and aura in (
            attacker.ability, defender.ability, field.attacker_ally_ability, field.defender_ally_ability,
        ):
            bp_chain.add(20, 3072 if aura_breaker else 5448)
    if da == "dryskin" and move_type == "Fire":
        bp_chain.add(17, 5120)
    if attacker.item in ITEM_TYPE_BOOST and ITEM_TYPE_BOOST[attacker.item] == move_type and not attacker.item_lost:
        bp_chain.add(15, 4915)
    if attacker.item == "normalgem" and move_type == "Normal" and not attacker.item_lost:
        bp_chain.add(14, 5325)
    if attacker.item in ITEM_CATEGORY_BOOST and ITEM_CATEGORY_BOOST[attacker.item] == category:
        bp_chain.add(16, 4505)
    terrain_shown = _terrain_bp(field, move_id, move_type, attacker, defender, bp_chain)
    _move_bp_chain_effects(move_id, attacker, defender, field, weather, bp_chain)
    base_power = max(1, bp_chain.apply(base_power))

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

    expanding_spread = (
        move_id == "expandingforce" and field.terrain == "psychic" and _is_grounded(attacker)
    )
    is_spread_move = (move["target"] in _SPREAD_TARGETS or expanding_spread) and field.num_targets >= 2
    spread_modifier = 0.75 if is_spread_move else 1.0
    breakdown["spread_modifier"] = spread_modifier
    if is_spread_move:
        raw = _modify(raw, 3072)

    weather_numerator = _Q
    if weather == "sun":
        weather_numerator = 6144 if move_type == "Fire" else 2048 if move_type == "Water" else _Q
    elif weather == "rain":
        weather_numerator = 6144 if move_type == "Water" else 2048 if move_type == "Fire" else _Q
    breakdown["weather_modifier"] = weather_numerator / _Q
    if weather_numerator != _Q:
        raw = _modify(raw, weather_numerator)
    breakdown["terrain_modifier"] = terrain_shown

    if always_crit:
        raw = int(raw * 1.5)

    attacker_types = [move_type] if attacker.ability in ("protean", "libero") else attacker.types()
    is_stab = move_type in attacker_types
    stab_numerator = (8192 if attacker.ability == "adaptability" else 6144) if is_stab else _Q
    breakdown["stab"] = stab_numerator / _Q

    burn_applies = attacker.status == "brn" and is_physical and attacker.ability != "guts" and move_id != "facade"
    breakdown["burn_modifier"] = 0.5 if burn_applies else 1.0

    # ModifyDamage chain (the final multiplier): screens, defensive and offensive abilities/items.
    final_chain = _Chain()
    has_screen = (
        ("reflect" in field.screens and category == "Physical")
        or ("lightscreen" in field.screens and category == "Special")
        or "auroraveil" in field.screens
    )
    if (
        has_screen
        and not always_crit
        and attacker.ability != "infiltrator"
        and move_id not in _SCREEN_BREAKERS
    ):
        final_chain.add(
            0, _SCREEN_NUMERATOR_DOUBLES if field.is_doubles else _SCREEN_NUMERATOR_SINGLES
        )
    if da == "multiscale" and defender.hp_or_max() >= defender.max_hp():
        final_chain.add(0, 2048)
    if da in ("solidrock", "filter") and type_mult > 1.0:
        final_chain.add(0, 3072)
    if da == "fluffy":
        contact = bool(flags.get("contact")) and attacker.ability != "longreach"
        if contact:
            final_chain.add(0, 2048)
        if move_type == "Fire":
            final_chain.add(0, 8192)
    if da == "punkrock" and flags.get("sound"):
        final_chain.add(0, 2048)
    if field.defender_ally_ability == "friendguard":
        final_chain.add(0, 3072)
    if attacker.ability == "tintedlens" and type_mult < 1.0:
        final_chain.add(0, 8192)
    resist_type = ITEM_RESIST_BERRY.get(defender.item or "") if not defender.item_lost else None
    if resist_type is not None and resist_type == move_type and (type_mult > 1.0 or resist_type == "Normal"):
        final_chain.add(-1, 1024 if defender_ability == "ripen" else 2048)
    if attacker.item == "lifeorb" and not attacker.item_lost:
        final_chain.add(0, 5324)
    if attacker.item == "expertbelt" and type_mult > 1.0 and not attacker.item_lost:
        final_chain.add(0, 4915)
    final_modifier = final_chain.modifier() if final_chain.items else _Q
    breakdown["final_multiplier"] = final_modifier / _Q

    # The random roll (85%-100%) is applied BEFORE STAB/type/burn/final modifiers, exactly as in
    # the engine; all 16 rolls are computed end to end because every step truncates.
    steps = type_steps
    rolls: list[int] = []
    for percent in range(85, 101):
        damage = (raw * percent) // 100
        if stab_numerator != _Q:
            damage = _modify(damage, stab_numerator)
        if steps > 0:
            damage *= 2**steps
        elif steps < 0:
            for _ in range(-steps):
                damage //= 2
        if burn_applies:
            damage = _modify(damage, 2048)
        if final_chain.items:
            damage = _modify(damage, final_modifier)
        rolls.append(max(damage, 1))
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


# --- small multipliers other modules (vgc.field_control) read directly -----------------------------


def _terrain_modifier(
    field: FieldState, move_type: str, attacker: PokemonState, defender: PokemonState
) -> float:
    """Terrain's base-power multiplier for a move type (display / estimation use)."""
    return _terrain_bp(field, "", move_type, attacker, defender, _Chain())


def _weather_modifier(field: FieldState, move_type: str) -> float:
    """Sun/rain damage multiplier for a move type."""
    if field.weather == "sun":
        return 1.5 if move_type == "Fire" else 0.5 if move_type == "Water" else 1.0
    if field.weather == "rain":
        return 1.5 if move_type == "Water" else 0.5 if move_type == "Fire" else 1.0
    return 1.0
