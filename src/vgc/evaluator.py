"""Phase 2b myopic 1-turn evaluator: scores every legal joint order for the current
doubles turn and hands back a ranked list.

`score_joint_orders(battle, config)` is the entry point `vgc.agent.VgcPlayer.decide()`
calls (see vgc/agent.py) -- it builds one `_Context` snapshot of the turn (our states,
the opponent's estimated states via `vgc.sets.opponent_state`, field conditions, and a
few precomputed threat figures), then scores each `DoubleBattleOrder` from
`vgc.actions.enumerate_joint_orders` against that snapshot.

This is deliberately "myopic": every order is scored assuming the turn's damage/status
outcomes resolve and nothing more (no 2-ply search, no opponent-response modeling beyond
"their single best known move against us this turn"). Every weight below lives on
`vgc.models.PolicyConfig` (individually commented there) so it's A/B-able from
offline/run_gates.py without a code change -- this module should never have a bare
numeric literal standing in for a strategic judgment call; if you need one, add a
PolicyConfig field.

## What's scored, and how (see PolicyConfig's comments for the exact weights)

- **Offense**: expected damage (%) of the chosen move onto each target it actually hits,
  with bonuses for a guaranteed KO (`min_damage >= target_hp`) or a merely-expected one,
  a speed-order discount when a target can plausibly KO the attacker first, and an
  outspeed bonus for KOing before a real threat gets to act. Spread moves
  (data/champions/moves.json `target: "allAdjacent"`, e.g. Earthquake) that also hit our
  own ally have that ally damage subtracted (scaled up, not just zeroed) unless the ally
  is immune (Levitate/Flying -- `damage_range` already returns 0 for that).
- **Speed/turn order**: `effective_speed` (boosted stat, Choice Scarf 1.5x, paralysis
  0.5x) and `resolves_before` (priority bracket, then Trick-Room-aware speed comparison)
  are the two building blocks; both are pure functions over `PokemonState`s so they're
  unit-testable without a real battle.
- **Defense/Protect**: `threat_on_us` (built once per turn in `_Context`) is each of our
  slots' best estimated incoming damage % from either opponent active's best known move;
  Protect's value scales with that, is penalized when `Pokemon.protect_counter` shows it
  was already used last turn, and is penalized again when the threat is negligible (this
  is what keeps the evaluator from reflexively double-Protecting when neither slot is
  actually threatened -- both penalties make an unthreatened Protect score below almost
  any attacking alternative, without a special-cased "not both" rule).
- **Opponent Protect/switch anticipation**: a `_SELF_PROTECT_MOVES` member in a slot's
  known kit makes Protect a real possibility instead of a guess -- "known" means Open
  Team Sheets when they're available, PLUS (since OTS essentially never triggers on the
  real public ladder -- see `vgc.replay_parse`'s module docstring) `vgc.sets.
  opponent_move_ids`'s replay-corpus-frequency fill for whatever's still unrevealed
  (`build_context` loads `vgc.sets.load_set_priors()` once per turn into `_Context.
  priors`, same pattern as `load_usage_spreads`). `_opp_protect_probability` estimates
  Protect's chance from a base rate plus how close our best single attack onto that slot
  comes to a KO (capped, and
  collapsed when `protect_counter` shows they already Protected last turn), and
  `_opp_switch_probability` estimates a pivot-out chance when that slot is under heavy
  pressure but its own output is weak. In `_score_attack_order`, only the KO-dependent
  bonuses (guaranteed/likely/outspeed) and the Fake Out flinch bonus are dampened by
  Protect probability -- raw expected damage is deliberately left undiscounted, since a
  blocked hit also costs the OPPONENT their turn and is roughly tempo-neutral, whereas a
  denied KO is a real EV loss (discounting the whole contribution instead broke the
  calibration of every other weight -- screens, switches, own Protect -- and regressed
  win rate in gate testing). Switch probability dampens only the KO-dependent bonuses too
  (our damage still lands on whatever replaces them). Separately, `_cross_slot_adjustments`
  penalizes STACKING two single-target attacks into the same Protect-capable slot: if it
  Protects, both attacks blank, so the smaller (redundant) one's expected value -- which
  could have gone at the other opposing slot instead -- is forfeited with probability
  `protect_prob` (`protect_stack_penalty_weight`).
- **Status/utility**: Trick Room, Fake Out (folded into the normal attack-scoring path
  since it IS a damaging move, not a status move), Sleep Powder, Helping Hand (scored at
  the joint-order level since its value depends on the PARTNER slot's chosen move),
  Parting Shot, and Reflect/Light Screen/Aurora Veil/Tailwind all get individually-hard-
  coded heuristics (see `_score_status_move` and friends) rather than a generic
  effect-interpreter -- the exported move data
  (data/champions/moves.json) doesn't carry boosts/selfSwitch/sideCondition fields to
  drive one generically (see module docstring deviations in CLAUDE.md).
- **Switches**: matchup delta (our best expected % onto the field's actives minus their
  best expected % onto the incoming mon) minus a flat tempo cost, plus an Intimidate
  re-trigger bonus.
- **Mega**: no explicit "should I mega" search -- `_attacker_state_for` swaps in the mega
  forme's stats/ability whenever `order.mega` is True, so a mega order's damage output is
  simply already-better in this scoring pass (mega_evolve_asap only breaks exact ties).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.damage import DamageResult, FieldState, PokemonState, damage_range, to_id
from vgc.data import load_moves, load_species
from vgc.decision_trace import record_note
from vgc.gameplan import GamePlan, build_gameplan
from vgc.meta import known_nature, recognize_meta_team
from vgc.models import PolicyConfig
from vgc.principles import (
    BURN_MOVES,
    DIRECT_DENIAL_MOVES,
    RECOVERY_MOVES,
    REDIRECTION_MOVES,
    SETUP_MOVES,
    SLEEP_MOVES,
    SPEED_CONTROL_MOVES,
    WIDE_DEFENSE_MOVES,
    detect_team_signals,
    is_speed_drop_attack,
    normalized_move_ids,
    utility_kind,
)
from vgc.sets import (
    load_set_priors,
    load_usage_spreads,
    normalize_item,
    normalize_status,
    opponent_move_ids,
    opponent_state,
)
from vgc.stats import STAT_IDS

# --- pure building blocks (unit tested directly against hand-built PokemonStates) ------

# Sentinel distinguishing "no override passed" from "explicitly override to None (no
# weather)" in _Context.field_state's `weather` kwarg.
_UNSET = object()

_PARALYSIS_SPEED_MULTIPLIER = 0.5
_CHOICE_SCARF_SPEED_MULTIPLIER = 1.5


def effective_speed(state: PokemonState) -> float:
    """Effective Speed stat for turn-order purposes: boost stage, then Choice Scarf
    (1.5x) and paralysis (0.5x). A float estimate (not the engine's exact 4096ths-chain
    integer math) -- fine for comparing two Pokemon's turn order, which is all this is
    used for; nothing here feeds back into `vgc.damage`'s own (exact) calculations.
    """
    from vgc.damage import _apply_stage  # reuse the exact stage-multiplier formula

    speed = float(_apply_stage(state.stats()["spe"], state.boost_stage("spe")))
    if state.item == "choicescarf":
        speed *= _CHOICE_SCARF_SPEED_MULTIPLIER
    if state.status == "par":
        speed *= _PARALYSIS_SPEED_MULTIPLIER
    return speed


def resolves_before(
    a_priority: int, a_speed: float, b_priority: int, b_speed: float, trick_room: bool
) -> bool:
    """True if actor A acts before actor B this turn: higher priority bracket always
    wins regardless of Trick Room (priority is untouched by it in the real engine);
    within the same bracket, faster goes first normally and SLOWER goes first under
    Trick Room. Ties favor A -- a heuristic evaluator doesn't need to model Showdown's
    random speed-tie coin flip, just a deterministic order for scoring.
    """
    if a_priority != b_priority:
        return a_priority > b_priority
    if trick_room:
        return a_speed <= b_speed
    return a_speed >= b_speed


def guaranteed_ko(result: DamageResult, target_hp: int) -> bool:
    """True if even the WORST damage roll already KOs -- no risk involved."""
    return target_hp > 0 and result.min_damage >= target_hp


def likely_ko(result: DamageResult, target_hp: int) -> bool:
    """True if the EXPECTED (mean) damage roll KOs -- real but roll-dependent."""
    return target_hp > 0 and result.expected_damage >= target_hp


def _opp_protect_probability(
    has_self_protect_move: bool, pressure_percent: float, protect_counter: int, config: PolicyConfig
) -> float:
    """Estimated probability an opponent slot Protects this turn.

    0.0 unless Open Team Sheets confirm the slot holds a `_SELF_PROTECT_MOVES` member --
    otherwise a base rate (`opp_protect_base_prob`) scaled up by how close our best single
    attack onto that slot comes to a KO (`pressure_percent`, capped at
    `opp_protect_prob_cap`), then collapsed toward zero if `protect_counter` shows they
    already Protected last turn (`opp_protect_repeat_factor`).
    """
    if not has_self_protect_move:
        return 0.0
    prob = min(
        config.opp_protect_prob_cap,
        config.opp_protect_base_prob
        + config.opp_protect_pressure_scale * (pressure_percent / 100.0),
    )
    if protect_counter >= 1:
        prob *= config.opp_protect_repeat_factor
    return prob


def _opp_switch_probability(
    pressure_percent: float, opp_output_percent: float, config: PolicyConfig
) -> float:
    """Flat probability an opponent slot switches out this turn: it only applies when the
    slot is under heavy incoming pressure (`pressure_percent >= opp_switch_pressure_floor`)
    AND its own best expected output is weak (`opp_output_percent < opp_switch_output_ceiling`)
    -- see `PolicyConfig`'s `opp_switch_*` comments for the reasoning behind each bound.
    """
    if (
        pressure_percent >= config.opp_switch_pressure_floor
        and opp_output_percent < config.opp_switch_output_ceiling
    ):
        return config.opp_switch_prob
    return 0.0


# --- move-id tables for the hand-coded status/utility heuristics ------------------------

# Every move whose primary job is "block this turn's incoming hits" -- scored uniformly
# via threat-on-this-slot (see _score_protect). Side-wide variants (Wide Guard, Quick
# Guard, Crafty Shield, Mat Block) are folded in too since none of our target team/eval
# scope needs to distinguish "blocks status" vs "blocks everything" for v1.
_PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "spikyshield",
        "banefulbunker",
        "kingsshield",
        "silktrap",
        "obstruct",
        "burningbulwark",
        "wideguard",
        "quickguard",
        "matblock",
        "craftyshield",
    }
)

# Protect-family moves that shield the move's USER from single-target attacks -- used to
# ask whether an OPPONENT slot might Protect against OUR incoming single-target attack
# this turn (see `_opp_protect_probability`). Deliberately narrower than `_PROTECT_MOVES`
# above (which drives scoring OUR OWN Protect choice and rightly includes the side-wide
# variants): Wide Guard/Quick Guard/Mat Block/Crafty Shield block spread moves for the
# whole side, not a single-target attack onto this one slot, so they're out of scope for
# the "will THIS slot dodge THIS hit" question this set answers.
_SELF_PROTECT_MOVES = frozenset(
    {
        "protect",
        "detect",
        "spikyshield",
        "banefulbunker",
        "kingsshield",
        "silktrap",
        "obstruct",
        "burningbulwark",
    }
)

# Abilities that block flinch (a secondary effect) outright -- Fake Out's bonus doesn't
# apply against these.
_FLINCH_IMMUNE_ABILITIES = frozenset({"innerfocus", "shielddust"})

# Abilities that block Intimidate's Attack drop outright (a simplified but representative
# subset -- Mind's Eye/Guard Dog invert or ignore it too, included for completeness).
_INTIMIDATE_IMMUNE_ABILITIES = frozenset(
    {
        "clearbody",
        "whitesmoke",
        "hypercutter",
        "owntempo",
        "oblivious",
        "innerfocus",
        "scrappy",
        "fullmetalbody",
        "mindseye",
        "guarddog",
    }
)

_SCREEN_MOVE_TO_SIDE_CONDITION = {
    "reflect": "reflect",
    "lightscreen": "lightscreen",
    "auroraveil": "auroraveil",
}

_SPREAD_TARGETS_HITTING_ALLY = frozenset({"allAdjacent"})
_SPREAD_TARGETS_FOES_ONLY = frozenset({"allAdjacentFoes"})
_SINGLE_TARGETS = frozenset({"normal", "any", "adjacentFoe"})

# `charge`-flag moves whose charge turn is skipped under sun (data/mods/champions's
# `onBasePower`/`beforeMove` conditionals for these two -- confirmed against
# data/champions/moves.json's `flags.charge`). Power Herb ALSO skips a charge move's
# turn (once, then consumed) but isn't modeled here -- see charge_move_discount's
# comment in vgc/models.py for that gap.
_SUN_SKIPS_CHARGE_TURN = frozenset({"solarbeam", "solarblade"})

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
_SIDE_CONDITION_TO_SCREEN = {
    SideCondition.REFLECT: "reflect",
    SideCondition.LIGHT_SCREEN: "lightscreen",
    SideCondition.AURORA_VEIL: "auroraveil",
}

# Weather a mega forme's OWN ability sets immediately on evolving (before this turn's
# moves resolve, per the real engine) -- e.g. Charizard-Mega-Y's Drought. `_Context.weather`
# is a snapshot of `battle.weather` from BEFORE this turn's (hypothetical) mega evolution,
# so a mega order that evolves into one of these abilities needs its own moves (including
# the mega's own turn) scored against the NEW weather, not the stale pre-mega one -- see
# `_score_attack_order`'s `weather_for_this_order` computation.
_ABILITY_WEATHER = {"drought": "sun", "drizzle": "rain"}
_WEATHER_SPEED_ABILITY = {
    "sun": "chlorophyll",
    "rain": "swiftswim",
    "sand": "sandrush",
    "snow": "slushrush",
}


def field_effective_speed(
    state: PokemonState, *, weather: str | None = None, tailwind: bool = False
) -> float:
    """Effective in-battle Speed including field multipliers.

    ``effective_speed`` handles modifiers carried by the Pokemon itself (boost stages,
    Choice Scarf, paralysis). This wrapper adds the active weather ability and the
    Pokemon's side of Tailwind, which are properties of the current field instead.
    """

    speed = effective_speed(state)
    if _WEATHER_SPEED_ABILITY.get(weather) == state.ability:
        speed *= 2.0
    if tailwind:
        speed *= 2.0
    return speed


def mega_species_id(base_species_id: str, item_id: str | None) -> str | None:
    """The mega forme's species id if `base_species_id` holding `item_id` can mega
    evolve into one of its `otherFormes`, else None. Pure lookup over
    `vgc.data.load_species()` -- no battle state involved.
    """
    if not item_id:
        return None
    species = load_species().get(base_species_id)
    if species is None:
        return None
    for forme_name in species.get("otherFormes") or ():
        forme_id = to_id(forme_name)
        forme_data = load_species().get(forme_id)
        if (
            forme_data
            and forme_data.get("isMega")
            and to_id(forme_data.get("requiredItem")) == item_id
        ):
            return forme_id
    return None


def mega_evolved_state(state: PokemonState) -> PokemonState:
    """Return the Mega form implied by ``state.item``, or ``state`` if none exists."""

    mega_id = mega_species_id(state.species_id, state.item)
    if mega_id is None:
        return state
    mega_species = load_species().get(mega_id) or {}
    mega_ability = to_id((mega_species.get("abilities") or {}).get("0"))
    return PokemonState(
        species_id=mega_id,
        sp_spread=state.sp_spread,
        nature=state.nature,
        boosts=dict(state.boosts),
        status=state.status,
        item=state.item,
        ability=mega_ability,
        current_hp=state.current_hp,
    )


# --- ScoredOrder / top-level entry point -------------------------------------------------


@dataclass
class ScoredOrder:
    order: DoubleBattleOrder
    score: float
    breakdown: dict[str, object] = field(default_factory=dict)


def score_joint_orders(
    battle: DoubleBattle, config: PolicyConfig | None = None
) -> list[ScoredOrder]:
    """Score every legal joint order for the current turn, best first.

    Returns `[]` if `vgc.actions.enumerate_joint_orders` has nothing legal (mirrors that
    function's own documented empty-list contract) -- callers should fall back to a
    random/default move in that case, same as `vgc.agent.VgcPlayer`'s exception-safe
    wrapper already does for any other evaluator failure.
    """
    config = config or PolicyConfig()
    joint_orders = enumerate_joint_orders(battle)
    if not joint_orders:
        return []

    ctx = build_context(battle, config)
    scored: list[ScoredOrder] = []
    for order in joint_orders:
        first_info = _score_single(order.first_order, 0, ctx, config)
        second_info = _score_single(order.second_order, 1, ctx, config)
        cross = _cross_slot_adjustments(first_info, second_info, config)
        total = first_info["score"] + second_info["score"] + cross
        scored.append(
            ScoredOrder(
                order=order,
                score=total,
                breakdown={"slot0": first_info, "slot1": second_info, "cross_slot": cross},
            )
        )
    scored.sort(key=lambda scored_order: scored_order.score, reverse=True)
    _record_trace(scored, config)
    return scored


def _record_trace(scored: list[ScoredOrder], config: PolicyConfig) -> None:
    if not scored:
        return
    top_k = max(1, config.trace_top_k)
    record_note(
        "top_candidates",
        [
            {"order": describe_order(entry.order), "score": round(entry.score, 3)}
            for entry in scored[:top_k]
        ],
    )
    record_note("chosen_breakdown", scored[0].breakdown)
    if len(scored) > 1:
        record_note("score_margin", round(scored[0].score - scored[1].score, 3))


# --- battle context: one snapshot per score_joint_orders() call -------------------------


@dataclass
class _ThreatInfo:
    percent: float = 0.0
    move_id: str | None = None
    priority: int = 0
    source_slot: int | None = None


@dataclass
class _Context:
    battle: DoubleBattle
    trick_room: bool
    weather: str | None
    terrain: str | None
    our_side_screens: frozenset[str]
    opp_side_screens: frozenset[str]
    our_pokemon: list[Pokemon | None]
    opp_pokemon: list[Pokemon | None]
    our_states: list[PokemonState | None]
    opp_states: list[PokemonState | None]
    our_speed: list[float]
    opp_speed: list[float]
    # Best estimated incoming damage % (from either opponent active's best known move)
    # onto each of OUR slots -- drives Protect's value.
    threat_on_us: list[_ThreatInfo]
    # Best estimated outgoing damage % (from either of our actives) onto each OPPONENT
    # slot -- a rough "how scary is this opposing mon overall" figure for Fake Out.
    opp_threat_score: list[float]
    # Estimated probability each OPPONENT slot Protects this turn (0.0 unless a
    # `_SELF_PROTECT_MOVES` member is in its known kit -- "known" via Open Team Sheets
    # OR `vgc.sets.opponent_move_ids`'s replay-corpus prior fill, see `priors` below) --
    # see `_opp_protect_probability`; dampens the ENTIRE per-target contribution in
    # `_score_attack_order` since Protect blocks damage, KO bonuses, and Fake Out alike.
    opp_protect_prob: list[float]
    # Estimated probability each OPPONENT slot switches out this turn under heavy
    # pressure with weak own output -- see `_opp_switch_probability`; dampens only the
    # KO-dependent bonuses in `_score_attack_order` (our damage still lands on whatever
    # replaces them).
    opp_switch_prob: list[float]
    # Loaded once per turn (`vgc.sets.load_set_priors()`, `{}` if set_priors.json is
    # missing) and threaded through `_Context` so `vgc.search._opp_slot_candidates` can
    # call `opponent_move_ids` with the SAME loaded dict instead of re-reading the file
    # -- mirrors how `opponent_state`'s own `usage=` parameter is threaded through.
    priors: dict[str, object]
    # Built once per turn by `vgc.gameplan.build_gameplan` (see that module) from our
    # full known team + the opponent's previewed/known team -- the win-condition
    # framework (`primary_win_con`, `plan_breakers`, ...) that `win_con_preservation_
    # weight`/`plan_breaker_target_bonus`/`collapsed_matchup_switch_bonus` read.  `None`
    # only when there's nothing to build it from (e.g. our own team came back empty,
    # shouldn't normally happen) -- every read site null-checks this.
    gameplan: GamePlan | None
    # Full joint danger if both opposing slots target the same one of ours, rather than
    # only the single strongest hit. Used for survival and Protect decisions.
    double_target_threat: list[float] = field(default_factory=lambda: [0.0, 0.0])
    # Missing move/item/ability facts per opposing active. Protect can earn information
    # value instead of being treated solely as damage prevention.
    opp_uncertainty: list[int] = field(default_factory=lambda: [0, 0])
    # Strategic non-damage threat (TR/Tailwind/setup/redirection/denial) per opposing slot.
    opp_control_threat: list[float] = field(default_factory=lambda: [0.0, 0.0])
    opp_engine_enabler_slots: frozenset[int] = frozenset()
    preview_plan: object | None = None
    endgame: bool = False

    def field_state(
        self, defender_is_ours: bool, num_targets: int, weather: str | None = _UNSET
    ) -> FieldState:
        screens = self.our_side_screens if defender_is_ours else self.opp_side_screens
        return FieldState(
            weather=self.weather if weather is _UNSET else weather,
            terrain=self.terrain,
            screens=screens,
            trick_room=self.trick_room,
            is_doubles=True,
            num_targets=num_targets,
        )

    def our_alive(self) -> list[int]:
        return [
            i for i in (0, 1) if self.our_pokemon[i] is not None and not self.our_pokemon[i].fainted
        ]

    def opp_alive(self) -> list[int]:
        return [
            i for i in (0, 1) if self.opp_pokemon[i] is not None and not self.opp_pokemon[i].fainted
        ]


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


def _screens_from(side_conditions) -> frozenset[str]:
    return frozenset(
        _SIDE_CONDITION_TO_SCREEN[sc] for sc in side_conditions if sc in _SIDE_CONDITION_TO_SCREEN
    )


def _our_pokemon_state(pokemon: Pokemon) -> PokemonState:
    """Build a PokemonState for one of OUR OWN Pokemon. Unlike `vgc.sets.opponent_state`,
    our own Stat Points/nature are genuinely known (poke-env parses them straight from
    the Teambuilder team we supplied, see `Pokemon.evs`/`Pokemon.nature`'s docstrings) --
    only fall back to the same usage-based guess opponent_state uses if poke-env didn't
    populate them for some reason (defensive; shouldn't happen for our own team).
    """
    species_id = to_id(pokemon.species)
    sp_spread = dict(zip(STAT_IDS, pokemon.evs, strict=True)) if pokemon.evs else None
    nature = pokemon.nature
    boosts = {
        stat: value
        for stat, value in (pokemon.boosts or {}).items()
        if stat in ("atk", "def", "spa", "spd", "spe") and value
    }
    state = PokemonState(
        species_id=species_id,
        sp_spread=sp_spread,
        nature=nature,
        boosts=boosts,
        status=normalize_status(pokemon.status),
        item=normalize_item(pokemon.item),
        ability=pokemon.ability or None,
    )
    if sp_spread is None or nature is None:
        from vgc.stats import default_opponent_nature, default_opponent_spread

        state.sp_spread = (
            state.sp_spread if state.sp_spread is not None else default_opponent_spread(species_id)
        )
        state.nature = (
            state.nature if state.nature is not None else default_opponent_nature(species_id)
        )
    state.current_hp = pokemon.current_hp
    return state


def _best_attacking_move(
    attacker: PokemonState, move_ids: list[str], defender: PokemonState, field_state: FieldState
) -> tuple[float, str | None, int]:
    """The attacker's single best (highest expected %) supported, non-immune attacking
    move against `defender` from `move_ids`. Returns (0.0, None, 0) if none qualify.
    """
    moves_data = load_moves()
    best_pct, best_move_id, best_priority = 0.0, None, 0
    for raw_move_id in move_ids:
        move_id = to_id(raw_move_id)
        data = moves_data.get(move_id)
        if data is None or data["category"] == "Status":
            continue
        result = damage_range(attacker, defender, move_id, field_state)
        if not result.breakdown["move_supported"] or result.breakdown["immune"]:
            continue
        if result.expected_percent > best_pct:
            best_pct = result.expected_percent
            best_move_id = move_id
            best_priority = int(data.get("priority", 0))
    return best_pct, best_move_id, best_priority


def build_context(battle: DoubleBattle, config: PolicyConfig) -> _Context:
    usage = load_usage_spreads()
    priors = load_set_priors()
    preview_team = list(getattr(battle, "teampreview_opponent_team", None) or [])
    if len(preview_team) != 6:
        opponent_team = getattr(battle, "opponent_team", {}) or {}
        preview_team = list(opponent_team.values())
    meta_team = recognize_meta_team(preview_team)
    if meta_team is not None:
        record_note("opponent_meta_team", {"id": meta_team["id"], "name": meta_team["name"]})

    trick_room = Field.TRICK_ROOM in battle.fields
    weather = _weather_str(battle)
    terrain = _terrain_str(battle)
    our_side_screens = _screens_from(battle.side_conditions)
    opp_side_screens = _screens_from(battle.opponent_side_conditions)

    our_pokemon = list(battle.active_pokemon or [None, None])
    opp_pokemon = list(battle.opponent_active_pokemon or [None, None])
    while len(our_pokemon) < 2:
        our_pokemon.append(None)
    while len(opp_pokemon) < 2:
        opp_pokemon.append(None)

    our_states = [
        _our_pokemon_state(mon) if mon is not None and not mon.fainted else None
        for mon in our_pokemon
    ]
    opp_states = [
        opponent_state(mon, usage=usage, nature_override=known_nature(meta_team, mon))
        if mon is not None and not mon.fainted
        else None
        for mon in opp_pokemon
    ]

    our_tailwind = SideCondition.TAILWIND in battle.side_conditions
    opp_tailwind = SideCondition.TAILWIND in battle.opponent_side_conditions
    our_speed = [
        field_effective_speed(state, weather=weather, tailwind=our_tailwind)
        if state is not None
        else 0.0
        for state in our_states
    ]
    opp_speed = [
        field_effective_speed(state, weather=weather, tailwind=opp_tailwind)
        if state is not None
        else 0.0
        for state in opp_states
    ]

    field_vs_us = FieldState(
        weather=weather,
        terrain=terrain,
        screens=our_side_screens,
        trick_room=trick_room,
        is_doubles=True,
    )
    # Mirrors field_vs_us but from the opponent's side of the field (their own screens) --
    # used below to estimate OUR best pressure onto each opponent slot for Protect/switch
    # anticipation.
    field_vs_opp = FieldState(
        weather=weather,
        terrain=terrain,
        screens=opp_side_screens,
        trick_room=trick_room,
        is_doubles=True,
    )

    threat_on_us = [_ThreatInfo(), _ThreatInfo()]
    double_target_threat = [0.0, 0.0]
    opp_threat_score = [0.0, 0.0]
    for our_idx in (0, 1):
        our_state = our_states[our_idx]
        if our_state is None:
            continue
        threats_from_each_opp: list[float] = []
        for opp_idx in (0, 1):
            opp_state = opp_states[opp_idx]
            opp_mon = opp_pokemon[opp_idx]
            if opp_state is None or opp_mon is None:
                continue
            move_ids = opponent_move_ids(opp_mon, priors=priors, config=config)
            pct, move_id, priority = _best_attacking_move(
                opp_state, move_ids, our_state, field_vs_us
            )
            threats_from_each_opp.append(pct)
            if pct > threat_on_us[our_idx].percent:
                threat_on_us[our_idx] = _ThreatInfo(
                    percent=pct, move_id=move_id, priority=priority, source_slot=opp_idx
                )
            opp_threat_score[opp_idx] = max(opp_threat_score[opp_idx], pct)
        double_target_threat[our_idx] = sum(threats_from_each_opp)

    # Our best expected % onto each opponent slot (the mirror image of threat_on_us) --
    # drives both Protect pressure and switch-incentive pressure below.
    pressure_on_opp = [0.0, 0.0]
    for opp_idx in (0, 1):
        opp_state = opp_states[opp_idx]
        if opp_state is None:
            continue
        for our_idx in (0, 1):
            our_state = our_states[our_idx]
            our_mon = our_pokemon[our_idx]
            if our_state is None or our_mon is None:
                continue
            our_move_ids = list(our_mon.moves.keys()) if our_mon.moves else []
            pct, _, _ = _best_attacking_move(our_state, our_move_ids, opp_state, field_vs_opp)
            pressure_on_opp[opp_idx] = max(pressure_on_opp[opp_idx], pct)

    opp_protect_prob = [0.0, 0.0]
    opp_switch_prob = [0.0, 0.0]
    opp_uncertainty = [0, 0]
    opp_control_threat = [0.0, 0.0]
    for opp_idx in (0, 1):
        opp_state = opp_states[opp_idx]
        opp_mon = opp_pokemon[opp_idx]
        if opp_state is None or opp_mon is None:
            continue
        known_move_ids = set(opponent_move_ids(opp_mon, priors=priors, config=config))
        has_self_protect_move = bool(known_move_ids & _SELF_PROTECT_MOVES)
        protect_counter = getattr(opp_mon, "protect_counter", 0)
        opp_protect_prob[opp_idx] = _opp_protect_probability(
            has_self_protect_move, pressure_on_opp[opp_idx], protect_counter, config
        )
        opp_switch_prob[opp_idx] = _opp_switch_probability(
            pressure_on_opp[opp_idx], opp_threat_score[opp_idx], config
        )
        revealed_move_count = len(getattr(opp_mon, "moves", None) or {})
        opp_uncertainty[opp_idx] = max(0, 4 - revealed_move_count)
        if not getattr(opp_mon, "item", None):
            opp_uncertainty[opp_idx] += 1
        if not getattr(opp_mon, "ability", None):
            opp_uncertainty[opp_idx] += 1
        # Open Team Sheets do not reveal this format's Stat Point spread/nature.
        opp_uncertainty[opp_idx] += 1
        control_ids = known_move_ids & (
            DIRECT_DENIAL_MOVES | REDIRECTION_MOVES | SPEED_CONTROL_MOVES | SETUP_MOVES
        )
        opp_control_threat[opp_idx] = min(100.0, 20.0 * len(control_ids))

    # Game-plan layer (vgc.gameplan): built from our FULL known team (battle.team --
    # always exactly known, unlike the opponent's) and their previewed/known team
    # (`preview_team`, computed above -- teampreview_opponent_team when the full 6 is
    # available, else whatever's been revealed so far via opponent_team). v1 deliberately
    # does NOT try to fill in opponent team-preview mons we have no state for beyond that
    # -- see vgc.gameplan's module docstring for why a pure function over already-built
    # states is the right boundary here.
    our_team_full = [
        mon
        for mon in (getattr(battle, "team", None) or {}).values()
        if mon is not None and not mon.fainted
    ]
    our_gameplan_states = [_our_pokemon_state(mon) for mon in our_team_full]
    our_gameplan_move_ids = [list(mon.moves.keys()) if mon.moves else [] for mon in our_team_full]

    opp_team_full = [
        mon for mon in preview_team if mon is not None and not getattr(mon, "fainted", False)
    ]
    opp_gameplan_states = [
        opponent_state(mon, usage=usage, nature_override=known_nature(meta_team, mon))
        for mon in opp_team_full
    ]
    opp_gameplan_move_ids = [
        opponent_move_ids(mon, priors=priors, config=config) for mon in opp_team_full
    ]

    preview_plan = getattr(battle, "_vgc_preview_plan", None)
    preferred_closer = getattr(preview_plan, "our_closer_species", None)
    gameplan = None
    if our_gameplan_states:
        gameplan = build_gameplan(
            our_gameplan_states,
            opp_gameplan_states,
            our_gameplan_move_ids,
            opp_gameplan_move_ids,
            config,
            preferred_win_con_species=preferred_closer,
        )
        record_note("gameplan", gameplan.summary())

    opp_signals = detect_team_signals(opp_team_full)
    enabler_species = {
        to_id(opp_team_full[idx].species)
        for idx in opp_signals.engine_enabler_indices
        if idx < len(opp_team_full)
    }
    opp_engine_enabler_slots = frozenset(
        idx
        for idx, mon in enumerate(opp_pokemon)
        if mon is not None and to_id(mon.species) in enabler_species
    )
    selected_ours = [
        mon
        for mon in (getattr(battle, "team", None) or {}).values()
        if mon is not None
        and not mon.fainted
        and (
            getattr(mon, "selected_in_teampreview", False)
            or getattr(mon, "_selected_in_teampreview", False)
        )
    ]
    if not selected_ours:
        selected_ours = our_team_full
    alive_opponents = [mon for mon in opp_team_full if not getattr(mon, "fainted", False)]
    endgame = len(selected_ours) <= 2 or len(alive_opponents) <= 2
    record_note(
        "principles_turn",
        {
            "our_speed": [round(speed, 3) for speed in our_speed],
            "opponent_speed": [round(speed, 3) for speed in opp_speed],
            "double_target_threat": [round(value, 3) for value in double_target_threat],
            "opponent_uncertainty": opp_uncertainty,
            "opponent_control_threat": opp_control_threat,
            "engine_enabler_slots": sorted(opp_engine_enabler_slots),
            "endgame": endgame,
        },
    )

    return _Context(
        battle=battle,
        trick_room=trick_room,
        weather=weather,
        terrain=terrain,
        our_side_screens=our_side_screens,
        opp_side_screens=opp_side_screens,
        our_pokemon=our_pokemon,
        opp_pokemon=opp_pokemon,
        our_states=our_states,
        opp_states=opp_states,
        our_speed=our_speed,
        opp_speed=opp_speed,
        threat_on_us=threat_on_us,
        opp_threat_score=opp_threat_score,
        opp_protect_prob=opp_protect_prob,
        opp_switch_prob=opp_switch_prob,
        priors=priors,
        gameplan=gameplan,
        double_target_threat=double_target_threat,
        opp_uncertainty=opp_uncertainty,
        opp_control_threat=opp_control_threat,
        opp_engine_enabler_slots=opp_engine_enabler_slots,
        preview_plan=preview_plan,
        endgame=endgame,
    )


# Backwards-compatible private alias (this module's own code used the underscore-
# prefixed name before Phase 2c's vgc.search needed to build the same _Context itself --
# same promotion pattern as vgc.sets.normalize_item/normalize_status).
_build_context = build_context


# --- per-slot scoring ---------------------------------------------------------------------


def _score_single(
    single: SingleBattleOrder | None, actor_slot: int, ctx: _Context, config: PolicyConfig
) -> dict:
    info: dict[str, object] = {
        "kind": "pass",
        "score": 0.0,
        "move_id": None,
        "raw_damage_score": 0.0,
    }
    if single is None:
        return info
    target = single.order
    if isinstance(target, Move):
        move_id = to_id(target.id)
        info["kind"] = "move"
        info["move_id"] = move_id
        move_data = load_moves().get(move_id)
        if move_data is None:
            info["reason"] = "unknown_move"
            return info
        if move_data["category"] == "Status":
            score, extra = _score_status_move(move_id, move_data, single, actor_slot, ctx, config)
            if getattr(single, "mega", False):
                mega_adjustment, mega_info = _score_status_mega(actor_slot, ctx, config)
                score += mega_adjustment
                extra.update(mega_info)
            extra.setdefault("actor_slot", actor_slot)
            extra.setdefault("utility_kind", utility_kind(move_id))
            info["score"] = score
            info.update(extra)
            return info
        score, raw_damage_score, extra = _score_attack_order(
            target, move_data, single, actor_slot, ctx, config
        )
        info["score"] = score
        info["raw_damage_score"] = raw_damage_score
        info.update(extra)
        return info
    if isinstance(target, Pokemon):
        info["kind"] = "switch"
        score, extra = _score_switch(target, actor_slot, ctx, config)
        info["score"] = score
        info.update(extra)
        return info
    # PassBattleOrder / str message orders (e.g. forced pass) -- neutral score.
    return info


def _score_status_mega(
    actor_slot: int, ctx: _Context, config: PolicyConfig
) -> tuple[float, dict[str, object]]:
    """Mega-resource adjustment for non-damaging orders such as Protect + Mega."""
    base_state = ctx.our_states[actor_slot]
    actor_mon = ctx.our_pokemon[actor_slot]
    if base_state is None:
        return -config.mega_unnecessary_penalty, {"mega_material": False}
    mega_state = mega_evolved_state(base_state)
    weather_change = _ABILITY_WEATHER.get(mega_state.ability, ctx.weather) != ctx.weather
    speed_flip = any(
        effective_speed(base_state) <= ctx.opp_speed[idx] < effective_speed(mega_state)
        for idx in ctx.opp_alive()
    )
    material = weather_change or speed_flip
    score = 0.0 if material else -config.mega_unnecessary_penalty
    planned_mega = getattr(ctx.preview_plan, "default_mega_species", None)
    actor_species = to_id(getattr(actor_mon, "species", None))
    if planned_mega is not None:
        score += (
            config.default_mega_bonus
            if actor_species == planned_mega
            else -config.alternate_mega_penalty
        )
    return score, {
        "mega_material": material,
        "mega_weather_change": weather_change,
        "mega_speed_flip": speed_flip,
    }


def _attacker_state_for(single: SingleBattleOrder, actor_slot: int, ctx: _Context) -> PokemonState:
    base_state = ctx.our_states[actor_slot]
    if getattr(single, "mega", False):
        return mega_evolved_state(base_state)
    return base_state


def _resolve_targets(
    move_data: dict, actor_slot: int, move_target: int, ctx: _Context
) -> list[tuple[int, bool]]:
    """Which of our/the opponent's alive active slots `move_data` actually hits, as
    `(slot_index, is_ally)` pairs. `move_target` is the showdown target position poke-env
    attached to the order (-1/-2 = our slot 1/2, 1/2 = opponent slot 1/2, 0 = "no single
    target", per `DoubleBattle`'s POKEMON_*_POSITION/OPPONENT_*_POSITION constants) --
    only meaningful for single-target moves; spread/self/field moves ignore it and use
    the move's own `target` string instead.
    """
    target_kind = move_data.get("target")
    opp_alive = ctx.opp_alive()
    our_allies_alive = [i for i in ctx.our_alive() if i != actor_slot]

    if target_kind in _SPREAD_TARGETS_FOES_ONLY:
        return [(i, False) for i in opp_alive]
    if target_kind in _SPREAD_TARGETS_HITTING_ALLY:
        return [(i, False) for i in opp_alive] + [(i, True) for i in our_allies_alive]
    if target_kind in _SINGLE_TARGETS:
        if move_target in (1, 2):
            idx = move_target - 1
            return [(idx, False)] if idx in opp_alive else []
        if move_target in (-1, -2):
            idx = -move_target - 1
            return [(idx, True)] if idx in our_allies_alive else []
        # No explicit target attached (shouldn't normally happen for these target
        # kinds) -- fall back to the first alive opposing slot.
        return [(opp_alive[0], False)] if opp_alive else []
    return []  # self / allySide / foeSide / all / adjacentAlly -- handled per-move below


def _score_attack_order(
    move: Move,
    move_data: dict,
    single: SingleBattleOrder,
    actor_slot: int,
    ctx: _Context,
    config: PolicyConfig,
) -> tuple[float, float, dict]:
    move_id = to_id(move.id)
    attacker_state = _attacker_state_for(single, actor_slot, ctx)
    targets = _resolve_targets(move_data, actor_slot, single.move_target, ctx)
    if not targets:
        return 0.0, 0.0, {"reason": "no_target"}

    opp_targets = [idx for idx, is_ally in targets if not is_ally]
    ally_targets = [idx for idx, is_ally in targets if is_ally]
    num_hit = len(opp_targets) + len(ally_targets)
    actor_priority = int(move_data.get("priority", 0))
    actor_mon = ctx.our_pokemon[actor_slot]

    # A mega evolution that grants a weather-setting ability (Drought/Drizzle) takes
    # effect before this turn's moves resolve in the real engine -- score THIS order's
    # own damage against that new weather, not the pre-mega snapshot in ctx.weather.
    weather_for_this_order = ctx.weather
    if getattr(single, "mega", False):
        weather_for_this_order = _ABILITY_WEATHER.get(attacker_state.ability, ctx.weather)

    score = 0.0
    raw_damage_score = 0.0
    expected_percent_by_target: dict[int, float] = {}
    current_hp_percent_by_target: dict[int, float] = {}
    guaranteed_ko_slots: list[int] = []
    ko_slots: list[int] = []
    survival_guard_slots: list[int] = []
    protect_prob_by_target: dict[int, float] = {}
    switch_prob_by_target: dict[int, float] = {}
    ally_expected_percent_by_target: dict[int, float] = {}
    # True once any target in this order gets a guaranteed/likely KO that resolves
    # BEFORE ctx.threat_on_us[actor_slot] can land -- see win_con_preservation_weight
    # below: a win con that removes its own threat this turn doesn't need "preserving".
    resolves_threat_before_it_lands = False
    speed_drop_targets: list[int] = []
    flinch_targets: list[int] = []
    field_vs_opp = ctx.field_state(
        defender_is_ours=False, num_targets=num_hit, weather=weather_for_this_order
    )
    for idx in opp_targets:
        defender_state = ctx.opp_states[idx]
        if defender_state is None:
            continue
        result = damage_range(attacker_state, defender_state, move_id, field_vs_opp)
        expected_percent_by_target[idx] = result.expected_percent
        current_hp_percent_by_target[idx] = (
            100.0 * defender_state.hp_or_max() / defender_state.max_hp()
        )
        base_damage = result.expected_percent * config.damage_percent_weight
        raw_damage_score += base_damage

        target_hp = defender_state.hp_or_max()
        is_guaranteed = guaranteed_ko(result, target_hp)
        is_likely = likely_ko(result, target_hp)
        ko_bonus = 0.0
        if is_guaranteed:
            guaranteed_ko_slots.append(idx)
            ko_slots.append(idx)
            ko_bonus += config.guaranteed_ko_bonus
        elif is_likely:
            ko_slots.append(idx)
            ko_bonus += config.likely_ko_bonus

        damage_term = base_damage
        threat = ctx.threat_on_us[actor_slot]
        we_move_first = True
        if threat.move_id is not None:
            opp_moves_first = resolves_before(
                threat.priority,
                ctx.opp_speed[idx],
                actor_priority,
                ctx.our_speed[actor_slot],
                ctx.trick_room,
            )
            if opp_moves_first and threat.percent >= 100.0:
                damage_term *= config.threatened_output_discount
                ko_bonus *= config.threatened_output_discount
            we_move_first = resolves_before(
                actor_priority,
                ctx.our_speed[actor_slot],
                threat.priority,
                ctx.opp_speed[idx],
                ctx.trick_room,
            )
        if we_move_first and (is_guaranteed or is_likely):
            ko_bonus += config.outspeed_ko_bonus
            resolves_threat_before_it_lands = True

        # An opponent pivoting out under heavy pressure with weak own output still eats
        # our damage on the replacement, but any KO-dependent bonus evaporates -- only the
        # KO-related share of this target's contribution is dampened by switch odds.
        ko_bonus *= 1.0 - ctx.opp_switch_prob[idx]
        # Protect denies the KO itself, which IS a real EV loss -- removing a threat from
        # the field is not "neutral" just because they also burned a turn to Protect. Raw
        # chip damage is different: a blocked hit costs the OPPONENT their turn too, so
        # that EV loss is roughly tempo-neutral and is deliberately left UNDISCOUNTED
        # (damage_term below never sees opp_protect_prob) -- multiplying the whole
        # per-target contribution by (1 - protect_prob) previously cut guaranteed-KO plays
        # so hard it broke the calibration of every other weight (screens, switches, own
        # Protect) and regressed the bot's win rate against SimpleHeuristicsPlayer.
        ko_bonus *= 1.0 - ctx.opp_protect_prob[idx]
        contribution = damage_term + ko_bonus

        # Removing a piece that specifically invalidates our win con re-enables the
        # whole game plan (the Chandelure-vs-Torkoal case) -- worth a flat bonus on top
        # of whatever this target's damage/KO math already contributes.
        if ctx.gameplan is not None:
            opp_mon_for_idx = ctx.opp_pokemon[idx]
            opp_species_id = to_id(opp_mon_for_idx.species) if opp_mon_for_idx is not None else None
            if opp_species_id is not None and opp_species_id in ctx.gameplan.plan_breakers:
                contribution += config.plan_breaker_target_bonus
            if opp_species_id == ctx.gameplan.primary_threat_species:
                contribution += config.primary_threat_target_bonus
        if idx in ctx.opp_engine_enabler_slots:
            contribution += config.engine_enabler_target_bonus

        if is_speed_drop_attack(move_id):
            reduced_speed = ctx.opp_speed[idx] * (2.0 / 3.0)
            flips_order = any(
                ctx.our_speed[our_idx] < ctx.opp_speed[idx]
                and ctx.our_speed[our_idx] >= reduced_speed
                for our_idx in ctx.our_alive()
            )
            speed_value = config.speed_drop_target_value * (1.0 + ctx.opp_threat_score[idx] / 100.0)
            if flips_order:
                speed_value += config.speed_control_immediate_ko_bonus * 0.5
            contribution += speed_value
            speed_drop_targets.append(idx)

        secondary = move_data.get("secondary") or {}
        if (
            secondary.get("volatileStatus") == "flinch"
            and we_move_first
            and to_id(getattr(ctx.opp_pokemon[idx], "ability", None))
            not in _FLINCH_IMMUNE_ABILITIES
        ):
            flinch_chance = float(secondary.get("chance", 0.0)) / 100.0
            contribution += (
                flinch_chance
                * config.generic_flinch_weight
                * (1.0 + ctx.opp_threat_score[idx] / 100.0)
            )
            flinch_targets.append(idx)

        if move_id == "fakeout" and getattr(actor_mon, "first_turn", False):
            opp_mon = ctx.opp_pokemon[idx]
            opp_ability = to_id(opp_mon.ability) if opp_mon and opp_mon.ability else None
            if opp_ability not in _FLINCH_IMMUNE_ABILITIES:
                fake_out_bonus = config.fake_out_weight * (1.0 + ctx.opp_threat_score[idx] / 100.0)
                # Protect blocks the flinch too -- no hit lands at all if they Protect.
                contribution += fake_out_bonus * (1.0 - ctx.opp_protect_prob[idx])

        protect_prob_by_target[idx] = ctx.opp_protect_prob[idx]
        switch_prob_by_target[idx] = ctx.opp_switch_prob[idx]

        score += contribution

        at_full_hp = defender_state.hp_or_max() >= defender_state.max_hp()
        if at_full_hp and (
            defender_state.item == "focussash" or defender_state.ability == "sturdy"
        ):
            survival_guard_slots.append(idx)

    field_vs_us = ctx.field_state(
        defender_is_ours=True, num_targets=num_hit, weather=weather_for_this_order
    )
    for idx in ally_targets:
        ally_state = ctx.our_states[idx]
        if ally_state is None:
            continue
        result = damage_range(attacker_state, ally_state, move_id, field_vs_us)
        ally_expected_percent_by_target[idx] = result.expected_percent
        score -= (
            result.expected_percent
            * config.damage_percent_weight
            * config.ally_damage_penalty_weight
        )

    mega_material = False
    mega_gain = 0.0
    if getattr(single, "mega", False):
        base_state = ctx.our_states[actor_slot]
        if base_state is not None:
            base_field = ctx.field_state(defender_is_ours=False, num_targets=num_hit)
            base_expected = 0.0
            for idx in opp_targets:
                defender_state = ctx.opp_states[idx]
                if defender_state is not None:
                    base_expected += damage_range(
                        base_state, defender_state, move_id, base_field
                    ).expected_percent
            mega_gain = raw_damage_score - base_expected * config.damage_percent_weight
            speed_flip = any(
                effective_speed(base_state) <= ctx.opp_speed[idx] < effective_speed(attacker_state)
                for idx in opp_targets
            )
            weather_change = weather_for_this_order != ctx.weather
            mega_material = (
                mega_gain >= config.mega_material_gain_floor or speed_flip or weather_change
            )
            if not mega_material:
                score -= config.mega_unnecessary_penalty
        planned_mega = getattr(ctx.preview_plan, "default_mega_species", None)
        actor_species = to_id(getattr(actor_mon, "species", None))
        if planned_mega is not None:
            if actor_species == planned_mega:
                score += config.default_mega_bonus
            else:
                score -= config.alternate_mega_penalty
        if config.mega_evolve_asap:
            score += 1e-3

    # Charge/recharge moves telegraph a slow, punishable turn (Solar Beam et al) or
    # forfeit the FOLLOWING turn after landing (Hyper Beam et al) -- discount the
    # order's entire contribution rather than trying to model the extra turn's own
    # opportunity cost or the free hit the opponent gets on it (v1 scope; see
    # search.py's module docstring for the search-side gap this leaves).
    move_flags = move_data.get("flags") or {}
    if move_flags.get("charge") and not (
        move_id in _SUN_SKIPS_CHARGE_TURN and weather_for_this_order == "sun"
    ):
        score *= config.charge_move_discount
        raw_damage_score *= config.charge_move_discount
    elif move_flags.get("recharge"):
        score *= config.recharge_move_discount
        raw_damage_score *= config.recharge_move_discount

    # Win-con preservation: our primary win condition shouldn't be traded away on a turn
    # where it's facing a near-certain KO and retreating (switch/Protect -- neither of
    # which passes through this function) was available. Only fires when this order
    # ITSELF doesn't already remove the threat first (resolves_threat_before_it_lands).
    win_con_preservation_penalty = 0.0
    if (
        ctx.gameplan is not None
        and ctx.gameplan.primary_win_con_species is not None
        and actor_mon is not None
        and to_id(actor_mon.species) == ctx.gameplan.primary_win_con_species
        and max(
            ctx.threat_on_us[actor_slot].percent,
            ctx.double_target_threat[actor_slot] * config.double_target_threat_weight,
        )
        >= 100.0
        and not resolves_threat_before_it_lands
    ):
        preservation_threat = max(
            ctx.threat_on_us[actor_slot].percent,
            ctx.double_target_threat[actor_slot] * config.double_target_threat_weight,
        )
        win_con_preservation_penalty = config.win_con_preservation_weight * preservation_threat
        score -= win_con_preservation_penalty

    single_target_slot = (
        opp_targets[0]
        if move_data.get("target") in _SINGLE_TARGETS and len(opp_targets) == 1
        else None
    )
    return (
        score,
        raw_damage_score,
        {
            "single_target_slot": single_target_slot,
            "expected_percent_by_target": expected_percent_by_target,
            "current_hp_percent_by_target": current_hp_percent_by_target,
            "guaranteed_ko_slots": guaranteed_ko_slots,
            "ko_slots": ko_slots,
            "survival_guard_slots": survival_guard_slots,
            "protect_prob_by_target": protect_prob_by_target,
            "switch_prob_by_target": switch_prob_by_target,
            "ally_expected_percent_by_target": ally_expected_percent_by_target,
            "actor_slot": actor_slot,
            "actor_species": to_id(getattr(actor_mon, "species", None)),
            "actor_speed": ctx.our_speed[actor_slot],
            "actor_priority": actor_priority,
            "target_speeds": {idx: ctx.opp_speed[idx] for idx in opp_targets},
            "speed_drop_targets": speed_drop_targets,
            "flinch_targets": flinch_targets,
            "mega_material": mega_material,
            "mega_gain": mega_gain,
            "win_con_preservation_penalty": win_con_preservation_penalty,
        },
    )


def _score_status_move(
    move_id: str,
    move_data: dict,
    single: SingleBattleOrder,
    actor_slot: int,
    ctx: _Context,
    config: PolicyConfig,
) -> tuple[float, dict]:
    if move_id in _PROTECT_MOVES:
        return _score_protect(actor_slot, ctx, config)
    if move_id == "trickroom":
        return _score_trick_room(ctx, config)
    if move_id in SLEEP_MOVES:
        return _score_sleep_move(move_id, move_data, single, actor_slot, ctx, config)
    if move_id == "helpinghand":
        return 0.0, {}  # cross-slot value only -- see _cross_slot_adjustments
    if move_id == "partingshot":
        return _score_parting_shot(actor_slot, ctx, config)
    if move_id in _SCREEN_MOVE_TO_SIDE_CONDITION or move_id == "tailwind":
        return _score_screen(move_id, ctx, config)
    if move_id in REDIRECTION_MOVES:
        partner_slot = 1 - actor_slot
        partner_threat = max(
            ctx.threat_on_us[partner_slot].percent,
            ctx.double_target_threat[partner_slot] * config.double_target_threat_weight,
        )
        return config.redirection_base_value + partner_threat * 0.4, {
            "utility_kind": "redirection",
            "partner_slot": partner_slot,
            "partner_threat": partner_threat,
        }
    if move_id in {"taunt", "encore", "yawn"}:
        return _score_targeted_denial(move_id, move_data, single, actor_slot, ctx, config)
    if move_id in SETUP_MOVES:
        danger = max(
            ctx.threat_on_us[actor_slot].percent,
            ctx.double_target_threat[actor_slot] * config.double_target_threat_weight,
        )
        score = config.setup_base_value
        if danger >= 80.0:
            score -= config.unsafe_setup_penalty
        return score, {"utility_kind": "setup", "danger": danger}
    if move_id in BURN_MOVES:
        return _score_burn(move_data, single, actor_slot, ctx, config)
    if move_id in WIDE_DEFENSE_MOVES:
        spread_threats = 0
        for mon in ctx.opp_pokemon:
            for known_id in normalized_move_ids(mon) if mon is not None else ():
                known_data = load_moves().get(known_id) or {}
                if known_data.get("target") in _SPREAD_TARGETS_FOES_ONLY:
                    spread_threats += 1
        return config.wide_defense_base_value * min(2, spread_threats), {
            "utility_kind": "wide_defense",
            "spread_threats": spread_threats,
        }
    if move_id in RECOVERY_MOVES:
        state = ctx.our_states[actor_slot]
        missing = 100.0 * (1.0 - state.hp_or_max() / state.max_hp()) if state is not None else 0.0
        return config.recovery_base_value * min(1.0, missing / 50.0), {
            "utility_kind": "recovery",
            "missing_hp_percent": missing,
        }
    return 0.0, {"reason": "unmodeled_status_move"}


def _score_protect(actor_slot: int, ctx: _Context, config: PolicyConfig) -> tuple[float, dict]:
    threat = ctx.threat_on_us[actor_slot]
    pokemon = ctx.our_pokemon[actor_slot]
    protect_counter = getattr(pokemon, "protect_counter", 0) if pokemon is not None else 0

    # Real Gen 9 mechanics: consecutive protect-family uses divide the move's actual
    # success chance by ~3 each time (counter 0 = 100%, 1 = ~33%, 2 = ~11%, ...) --
    # multiplying the threat-avoidance term by that same geometric factor makes the
    # SCORE decay the way the move's real value does (a Protect that's very likely to
    # just fail avoids very little expected damage), superseding the old flat
    # `protect_repeat_penalty` subtraction (see that field's comment in vgc/models.py).
    success_prob = config.protect_success_decay**protect_counter
    combined_threat = max(
        threat.percent,
        ctx.double_target_threat[actor_slot] * config.double_target_threat_weight,
    )
    score = combined_threat * config.protect_threat_weight * success_prob
    if threat.percent < config.protect_low_threat_floor:
        score -= config.protect_low_threat_penalty
    information_value = (
        min(4, sum(ctx.opp_uncertainty)) * config.protect_information_per_unknown
        if combined_threat > 0.0
        else 0.0
    )
    stall_reasons = 0
    if SideCondition.TAILWIND in getattr(ctx.battle, "opponent_side_conditions", ()):
        stall_reasons += 1
    if ctx.trick_room:
        our_alive, opp_alive = ctx.our_alive(), ctx.opp_alive()
        if our_alive and opp_alive:
            our_avg = mean(ctx.our_speed[idx] for idx in our_alive)
            opp_avg = mean(ctx.opp_speed[idx] for idx in opp_alive)
            if our_avg > opp_avg:  # Trick Room is helping their slower side.
                stall_reasons += 1
    if ctx.weather is not None:
        stall_reasons += 1
    if any(state is not None and state.status is not None for state in ctx.opp_states):
        stall_reasons += 1
    stall_value = stall_reasons * config.protect_field_stall_per_turn
    available_switches = getattr(ctx.battle, "available_switches", None) or [[], []]
    has_reposition = actor_slot < len(available_switches) and bool(available_switches[actor_slot])
    reposition_value = config.protect_reposition_bonus if has_reposition else 0.0
    score += information_value + stall_value + reposition_value
    return score, {
        "threat_percent": threat.percent,
        "combined_threat_percent": combined_threat,
        "protect_counter": protect_counter,
        "success_prob": success_prob,
        "information_value": information_value,
        "stall_value": stall_value,
        "reposition_value": reposition_value,
        "utility_kind": "protect",
        "actor_slot": actor_slot,
        "threat_source_slot": threat.source_slot,
    }


def _speed_control_flip_value(
    ctx: _Context,
    config: PolicyConfig,
    hypothetical_our_speed,  # Callable[[int], float] -- the slot's speed AFTER setting this
    trick_room_hypothetical: bool,  # ctx.trick_room's value AFTER setting this
) -> tuple[float, dict]:
    """Speed control's value should scale with its DAMAGE-ORDER consequence, not just
    the raw average-speed gap `_score_trick_room`'s base term already prices in (see
    that function's docstring for why the base term alone is far too small to ever win
    the argmax against a real attack -- confirmed against a reconstructed postmortem
    trace: Venusaur+Farigiraf vs Sneasler+Blaziken, no field, scored the base term at
    34.2 points while a single attacking move scored 260+).

    For each of our alive slots, checks whether setting this (Trick Room or Tailwind)
    would flip that slot from LOSING the speed race against its biggest known threat
    (`ctx.threat_on_us[slot]`) to WINNING it -- i.e. we stop eating
    `threatened_output_discount` and start landing our own hits (or a KO) before that
    threat resolves. A threat with priority > 0 never flips (priority brackets are
    untouched by both Trick Room and Tailwind, matching `resolves_before`'s own
    modeling) -- correctly excludes Fake Out-class pressure from inflating this term.

    `value = speed_control_flip_weight * (sum over BENEFITING slots of
    min(100, threat_on_us[slot].percent)) * (benefiting slot count / alive slot count)`
    -- the sum rewards flipping order against a THREATENING slot specifically (a flip
    that doesn't matter for any real incoming threat contributes nothing), and the
    fraction-of-team term keeps a flip that only helps one of two alive slots worth less
    than one that helps the whole team, without any per-turn counter/state (a pure
    function of this turn's `_Context` alone, by design -- see PolicyConfig's comment).
    """
    our_alive = ctx.our_alive()
    if not our_alive:
        return 0.0, {"benefiting_slots": []}
    benefiting_slots: list[int] = []
    benefiting_pct_sum = 0.0
    for slot in our_alive:
        threat = ctx.threat_on_us[slot]
        if threat.move_id is None or threat.source_slot is None:
            continue
        opp_speed = ctx.opp_speed[threat.source_slot]
        currently_first = resolves_before(
            0, ctx.our_speed[slot], threat.priority, opp_speed, ctx.trick_room
        )
        if currently_first:
            continue  # already winning the race against this threat -- no flip to earn
        hypothetically_first = resolves_before(
            0, hypothetical_our_speed(slot), threat.priority, opp_speed, trick_room_hypothetical
        )
        if hypothetically_first:
            benefiting_slots.append(slot)
            benefiting_pct_sum += min(100.0, threat.percent)
    fraction_benefiting = len(benefiting_slots) / len(our_alive)
    value = config.speed_control_flip_weight * benefiting_pct_sum * fraction_benefiting
    return value, {
        "benefiting_slots": benefiting_slots,
        "benefiting_pct_sum": round(benefiting_pct_sum, 3),
        "fraction_benefiting": round(fraction_benefiting, 3),
        "flip_value": round(value, 3),
    }


def _score_trick_room(ctx: _Context, config: PolicyConfig) -> tuple[float, dict]:
    if ctx.trick_room:
        return -config.trick_room_teardown_penalty, {"reason": "already_active"}
    our_alive, opp_alive = ctx.our_alive(), ctx.opp_alive()
    our_avg = mean([ctx.our_speed[i] for i in our_alive]) if our_alive else 0.0
    opp_avg = mean([ctx.opp_speed[i] for i in opp_alive]) if opp_alive else 0.0
    base = (opp_avg - our_avg) * config.trick_room_setup_weight
    # Trick Room flips the FULL speed comparison (same raw speeds, inverted direction --
    # not a multiplier the way Tailwind is), so the hypothetical speed per slot is just
    # its CURRENT effective speed.
    flip_value, flip_info = _speed_control_flip_value(
        ctx,
        config,
        hypothetical_our_speed=lambda slot: ctx.our_speed[slot],
        trick_room_hypothetical=True,
    )
    return base + flip_value, {
        "our_avg_speed": our_avg,
        "opp_avg_speed": opp_avg,
        "base_value": round(base, 3),
        **flip_info,
    }


def _score_sleep_move(
    move_id: str,
    move_data: dict,
    single: SingleBattleOrder,
    actor_slot: int,
    ctx: _Context,
    config: PolicyConfig,
) -> tuple[float, dict]:
    targets = _resolve_targets(move_data, actor_slot, single.move_target, ctx)
    opp_targets = [idx for idx, is_ally in targets if not is_ally]
    if not opp_targets:
        return 0.0, {"reason": "no_target"}
    idx = opp_targets[0]
    state = ctx.opp_states[idx]
    if state is None:
        return 0.0, {"reason": "no_target"}
    if state.status is not None:
        return 0.0, {"reason": "already_statused"}
    if move_id in {"sleeppowder", "spore"} and (
        "Grass" in state.types() or state.ability == "overcoat"
    ):
        return 0.0, {"reason": "powder_immune"}
    accuracy = move_data.get("accuracy", 100)
    accuracy_factor = 1.0 if accuracy is True else float(accuracy) / 100.0
    base = config.sleep_powder_weight if move_id == "sleeppowder" else config.generic_sleep_value
    value = base * accuracy_factor * (1.0 + ctx.opp_speed[idx] / 200.0)
    return value, {"target_slot": idx, "utility_kind": "action_denial"}


def _score_targeted_denial(
    move_id: str,
    move_data: dict,
    single: SingleBattleOrder,
    actor_slot: int,
    ctx: _Context,
    config: PolicyConfig,
) -> tuple[float, dict]:
    targets = _resolve_targets(move_data, actor_slot, single.move_target, ctx)
    opp_targets = [idx for idx, is_ally in targets if not is_ally]
    if not opp_targets:
        return 0.0, {"reason": "no_target"}
    idx = opp_targets[0]
    mon = ctx.opp_pokemon[idx]
    state = ctx.opp_states[idx]
    known_ids = set(normalized_move_ids(mon)) if mon is not None else set()
    known_status = sum(
        (load_moves().get(known_id) or {}).get("category") == "Status" for known_id in known_ids
    )
    accuracy = move_data.get("accuracy", 100)
    accuracy_factor = 1.0 if accuracy is True else float(accuracy) / 100.0
    if move_id == "taunt":
        value = config.taunt_base_value * min(1.0, known_status / 2.0)
    elif move_id == "encore":
        protect_counter = getattr(mon, "protect_counter", 0) if mon is not None else 0
        value = config.encore_base_value * (
            1.0 if protect_counter else min(1.0, known_status / 2.0)
        )
    else:  # Yawn: delayed but near-certain action denial or a forced switch.
        if state is None or state.status is not None:
            return 0.0, {"reason": "ineligible_target"}
        value = config.yawn_base_value
    value *= accuracy_factor
    value *= 1.0 + ctx.opp_control_threat[idx] / 100.0
    return value, {
        "target_slot": idx,
        "utility_kind": "action_denial",
        "known_status_moves": known_status,
    }


def _score_burn(
    move_data: dict,
    single: SingleBattleOrder,
    actor_slot: int,
    ctx: _Context,
    config: PolicyConfig,
) -> tuple[float, dict]:
    targets = _resolve_targets(move_data, actor_slot, single.move_target, ctx)
    opp_targets = [idx for idx, is_ally in targets if not is_ally]
    if not opp_targets:
        return 0.0, {"reason": "no_target"}
    idx = opp_targets[0]
    state = ctx.opp_states[idx]
    mon = ctx.opp_pokemon[idx]
    if state is None or state.status is not None or "Fire" in state.types():
        return 0.0, {"reason": "burn_ineligible"}
    known_ids = normalized_move_ids(mon) if mon is not None else frozenset()
    physical = sum(
        (load_moves().get(move_id) or {}).get("category") == "Physical" for move_id in known_ids
    )
    damaging = sum(
        (load_moves().get(move_id) or {}).get("category") != "Status" for move_id in known_ids
    )
    physical_share = physical / max(1, damaging)
    accuracy = move_data.get("accuracy", 100)
    accuracy_factor = 1.0 if accuracy is True else float(accuracy) / 100.0
    value = config.burn_base_value * physical_share * accuracy_factor
    value *= 1.0 + ctx.opp_threat_score[idx] / 100.0
    return value, {"target_slot": idx, "utility_kind": "burn", "physical_share": physical_share}


def _score_parting_shot(actor_slot: int, ctx: _Context, config: PolicyConfig) -> tuple[float, dict]:
    threat = ctx.threat_on_us[actor_slot]
    # Rough split: a fraction of screen_setup_weight standing in for the -1 Atk/-1 SpA
    # debuff's value (not separately modeled -- moves.json doesn't export boosts data),
    # plus a pivot term scaled by how threatened this slot already is (escaping a bad
    # matchup is worth more the worse that matchup is).
    debuff_value = config.screen_setup_weight * 0.3
    pivot_value = threat.percent * 0.4
    return debuff_value + pivot_value, {"threat_percent": threat.percent}


def _score_screen(move_id: str, ctx: _Context, config: PolicyConfig) -> tuple[float, dict]:
    if move_id == "tailwind":
        already_active = SideCondition.TAILWIND in ctx.battle.side_conditions
        if already_active:
            return -config.screen_setup_weight * 0.5, {"reason": "already_active"}
        # Tailwind flips the speed race the same way Trick Room does but via a 2x
        # multiplier on OUR side's speed rather than inverting the comparison -- same
        # `_speed_control_flip_value` helper, different hypothetical-speed function (see
        # `field_effective_speed`'s own `tailwind` handling for why 2.0 is the real
        # mechanical multiplier, not a separately-tuned literal here).
        flip_value, flip_info = _speed_control_flip_value(
            ctx,
            config,
            hypothetical_our_speed=lambda slot: ctx.our_speed[slot] * 2.0,
            trick_room_hypothetical=ctx.trick_room,
        )
        return config.screen_setup_weight + flip_value, {
            "utility_kind": "speed_control",
            **flip_info,
        }
    already_active = _SCREEN_MOVE_TO_SIDE_CONDITION[move_id] in ctx.our_side_screens
    if already_active:
        return -config.screen_setup_weight * 0.5, {"reason": "already_active"}
    return config.screen_setup_weight, {}


def _intimidate_immune(pokemon: Pokemon | None) -> bool:
    if pokemon is None:
        return True
    ability = to_id(pokemon.ability) if pokemon.ability else None
    return ability in _INTIMIDATE_IMMUNE_ABILITIES


def _score_switch(
    incoming: Pokemon, actor_slot: int, ctx: _Context, config: PolicyConfig
) -> tuple[float, dict]:
    incoming_state = _our_pokemon_state(incoming)
    incoming_move_ids = list(incoming.moves.keys()) if incoming.moves else []
    opp_alive = ctx.opp_alive()

    field_vs_opp = ctx.field_state(defender_is_ours=False, num_targets=1)
    field_vs_us = ctx.field_state(defender_is_ours=True, num_targets=1)

    our_best = 0.0
    their_best = 0.0
    incoming_by_opponent: list[float] = []
    for idx in opp_alive:
        opp_state = ctx.opp_states[idx]
        opp_mon = ctx.opp_pokemon[idx]
        if opp_state is None or opp_mon is None:
            continue
        pct_out, _, _ = _best_attacking_move(
            incoming_state, incoming_move_ids, opp_state, field_vs_opp
        )
        our_best = max(our_best, pct_out)
        opp_move_ids = opponent_move_ids(opp_mon, priors=ctx.priors, config=config)
        pct_in, _, _ = _best_attacking_move(opp_state, opp_move_ids, incoming_state, field_vs_us)
        their_best = max(their_best, pct_in)
        incoming_by_opponent.append(pct_in)

    matchup = (our_best - their_best) * config.switch_matchup_weight
    score = matchup - config.switch_tempo_cost
    if incoming.ability and to_id(incoming.ability) == "intimidate":
        eligible = [idx for idx in opp_alive if not _intimidate_immune(ctx.opp_pokemon[idx])]
        score += len(eligible) * config.intimidate_switch_bonus
    incoming_ability = to_id(getattr(incoming, "ability", None))
    if incoming_ability in {"drizzle", "drought", "sandstream", "snowwarning", "hospitality"}:
        score += config.switch_activation_bonus
    safe_from_both = bool(incoming_by_opponent) and all(
        pct < config.opp_switch_output_ceiling for pct in incoming_by_opponent
    )
    if safe_from_both:
        score += config.switch_safe_both_bonus

    # Collapsed-matchup pressure (the Charizard-in-rain fix): compare the OUTGOING mon's
    # CURRENT best expected % onto the field's actives (real weather, via the same
    # field_vs_opp/_best_attacking_move math as above) against its gameplan-table
    # (neutral-field) expectation for those same opponents. A mon whose role has
    # genuinely collapsed this turn (current well below what the matchup "should" do)
    # generates real switch pressure beyond whatever the plain current-field matchup
    # term above already captures, since that term only sees "less damage than usual",
    # not "this matchup fundamentally isn't working under these conditions".
    collapsed_bonus = 0.0
    outgoing_mon = ctx.our_pokemon[actor_slot]
    outgoing_state = ctx.our_states[actor_slot]
    if ctx.gameplan is not None and outgoing_mon is not None and outgoing_state is not None:
        outgoing_species = to_id(outgoing_mon.species)
        outgoing_move_ids = list(outgoing_mon.moves.keys()) if outgoing_mon.moves else []
        current_pct = 0.0
        table_pct = 0.0
        for idx in opp_alive:
            opp_state = ctx.opp_states[idx]
            opp_mon = ctx.opp_pokemon[idx]
            if opp_state is None or opp_mon is None:
                continue
            pct, _, _ = _best_attacking_move(
                outgoing_state, outgoing_move_ids, opp_state, field_vs_opp
            )
            current_pct = max(current_pct, pct)
            table_pct = max(
                table_pct, ctx.gameplan.table_percent(outgoing_species, to_id(opp_mon.species))
            )
        if table_pct > 0.0 and current_pct < config.collapsed_matchup_floor * table_pct:
            collapsed_bonus = config.collapsed_matchup_switch_bonus * (table_pct - current_pct)
    score += collapsed_bonus

    endgame_bonus = 0.0
    if ctx.gameplan is not None and ctx.gameplan.primary_win_con_species is not None:
        outgoing_mon = ctx.our_pokemon[actor_slot]
        outgoing_is_closer = (
            outgoing_mon is not None
            and to_id(outgoing_mon.species) == ctx.gameplan.primary_win_con_species
        )
        incoming_is_closer = to_id(incoming.species) == ctx.gameplan.primary_win_con_species
        if outgoing_is_closer and not incoming_is_closer:
            endgame_bonus += config.switch_endgame_preservation_bonus
        elif ctx.endgame and incoming_is_closer and our_best > their_best:
            endgame_bonus += config.switch_endgame_preservation_bonus
    score += endgame_bonus

    return score, {
        "matchup": matchup,
        "our_best_percent": our_best,
        "their_best_percent": their_best,
        "incoming_percent_by_opponent": incoming_by_opponent,
        "safe_from_both": safe_from_both,
        "activation_ability": incoming_ability,
        "collapsed_matchup_bonus": collapsed_bonus,
        "endgame_bonus": endgame_bonus,
    }


def _cross_slot_adjustments(first_info: dict, second_info: dict, config: PolicyConfig) -> float:
    bonus = 0.0
    if first_info.get("move_id") == "helpinghand" and second_info.get("raw_damage_score"):
        bonus += float(second_info["raw_damage_score"]) * config.helping_hand_weight
    if second_info.get("move_id") == "helpinghand" and first_info.get("raw_damage_score"):
        bonus += float(first_info["raw_damage_score"]) * config.helping_hand_weight

    # Spread damage beside Protect is a deliberate pressure pairing: the ally does not
    # take the spread hit in the real turn, so refund the evaluator's ally-damage penalty.
    for protect_info, attack_info in ((first_info, second_info), (second_info, first_info)):
        if protect_info.get("utility_kind") == "protect":
            protected_slot = protect_info.get("actor_slot")
            ally_damage = float(
                attack_info.get("ally_expected_percent_by_target", {}).get(protected_slot, 0.0)
            )
            bonus += ally_damage * config.damage_percent_weight * config.ally_damage_penalty_weight

    utility_pairs = (
        (first_info.get("move_id"), second_info.get("utility_kind")),
        (second_info.get("move_id"), first_info.get("utility_kind")),
    )
    if any(move_id == "fakeout" and kind == "setup" for move_id, kind in utility_pairs):
        bonus += config.fake_out_setup_bonus
    if any(
        redirect.get("utility_kind") == "redirection" and partner.get("utility_kind") == "setup"
        for redirect, partner in ((first_info, second_info), (second_info, first_info))
    ):
        bonus += config.redirection_setup_bonus

    # Protect is strongest when the partner removes the exact attacker committing into
    # the protected slot, converting one enemy action into nothing while we still act.
    for protect_info, partner_info in ((first_info, second_info), (second_info, first_info)):
        source_slot = protect_info.get("threat_source_slot")
        if (
            protect_info.get("utility_kind") == "protect"
            and source_slot is not None
            and (
                source_slot in partner_info.get("ko_slots", ())
                or source_slot in partner_info.get("flinch_targets", ())
                or (
                    partner_info.get("utility_kind") == "action_denial"
                    and partner_info.get("target_slot") == source_slot
                )
            )
        ):
            bonus += config.protect_partner_cleanup_bonus

    # Tailwind/Trick Room/speed drops earn their largest value only when they immediately
    # flip the partner into a KO before the target can act.
    for control_info, attack_info in ((first_info, second_info), (second_info, first_info)):
        ko_slots = attack_info.get("ko_slots", ())
        actor_speed = float(attack_info.get("actor_speed", 0.0))
        target_speeds = attack_info.get("target_speeds", {})
        enables_ko = False
        if control_info.get("move_id") == "tailwind":
            enables_ko = any(
                actor_speed < float(target_speeds.get(slot, 0.0)) <= actor_speed * 2.0
                for slot in ko_slots
            )
        elif control_info.get("move_id") == "trickroom":
            enables_ko = any(actor_speed < float(target_speeds.get(slot, 0.0)) for slot in ko_slots)
        elif control_info.get("speed_drop_targets"):
            enables_ko = any(
                slot in control_info.get("speed_drop_targets", ())
                and actor_speed < float(target_speeds.get(slot, 0.0))
                and actor_speed >= float(target_speeds.get(slot, 0.0)) * (2.0 / 3.0)
                for slot in ko_slots
            )
        if enables_ko:
            bonus += config.speed_control_immediate_ko_bonus

    first_target = first_info.get("single_target_slot")
    second_target = second_info.get("single_target_slot")
    if first_target is not None and second_target is not None and first_target != second_target:
        bonus += config.dual_target_pressure_bonus
    if first_target is not None and first_target == second_target:
        target = int(first_target)
        first_guarantees = target in first_info.get("guaranteed_ko_slots", ())
        second_guarantees = target in second_info.get("guaranteed_ko_slots", ())
        has_survival_guard = target in first_info.get(
            "survival_guard_slots", ()
        ) or target in second_info.get("survival_guard_slots", ())
        first_expected = float(first_info.get("expected_percent_by_target", {}).get(target, 0.0))
        second_expected = float(second_info.get("expected_percent_by_target", {}).get(target, 0.0))

        # Stacking penalty: if the shared target Protects, BOTH single-target attacks
        # blank this turn -- the smaller of the two was the redundant one, and it could
        # have gone at the OTHER opposing slot instead, so that expected value is
        # forfeited with probability protect_prob. This is additive with (not a
        # replacement for) the guaranteed-KO/focus-fire sub-cases below, since it's a
        # distinct cost (a wasted move slot) from either of those.
        protect_prob = float(
            first_info.get("protect_prob_by_target", {}).get(
                target, second_info.get("protect_prob_by_target", {}).get(target, 0.0)
            )
        )
        bonus -= (
            protect_prob
            * min(first_expected, second_expected)
            * config.protect_stack_penalty_weight
        )

        if (first_guarantees or second_guarantees) and not has_survival_guard:
            bonus -= config.redundant_ko_target_penalty
        elif not first_guarantees and not second_guarantees:
            target_hp = float(
                first_info.get("current_hp_percent_by_target", {}).get(
                    target,
                    second_info.get("current_hp_percent_by_target", {}).get(target, 100.0),
                )
            )
            if first_expected < target_hp and second_expected < target_hp:
                if first_expected + second_expected >= target_hp:
                    bonus += config.focus_fire_ko_bonus

    # Spread-plus-cleanup is the other canonical focus-fire pattern. It never enters the
    # same-single-target branch above because a spread move has no single_target_slot.
    first_expected_map = first_info.get("expected_percent_by_target", {})
    second_expected_map = second_info.get("expected_percent_by_target", {})
    if first_target is None or second_target is None:
        for target in set(first_expected_map).intersection(second_expected_map):
            first_expected = float(first_expected_map[target])
            second_expected = float(second_expected_map[target])
            target_hp = float(
                first_info.get("current_hp_percent_by_target", {}).get(
                    target,
                    second_info.get("current_hp_percent_by_target", {}).get(target, 100.0),
                )
            )
            if (
                first_expected < target_hp
                and second_expected < target_hp
                and first_expected + second_expected >= target_hp
            ):
                bonus += config.focus_fire_ko_bonus
    return bonus
