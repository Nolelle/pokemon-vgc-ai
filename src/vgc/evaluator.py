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
- **Opponent Protect/switch anticipation**: Open Team Sheets reveal the opponent's full
  movesets, so a `_SELF_PROTECT_MOVES` member in a slot's known kit makes Protect a real
  possibility instead of a guess -- `_opp_protect_probability` estimates it from a base
  rate plus how close our best single attack onto that slot comes to a KO (capped, and
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
from vgc.meta import known_nature, recognize_meta_team
from vgc.models import PolicyConfig
from vgc.sets import load_usage_spreads, normalize_item, normalize_status, opponent_state
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

    ctx = _build_context(battle, config)
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
    # Estimated probability each OPPONENT slot Protects this turn (0.0 unless Open Team
    # Sheets confirm a `_SELF_PROTECT_MOVES` member in its known kit) -- see
    # `_opp_protect_probability`; dampens the ENTIRE per-target contribution in
    # `_score_attack_order` since Protect blocks damage, KO bonuses, and Fake Out alike.
    opp_protect_prob: list[float]
    # Estimated probability each OPPONENT slot switches out this turn under heavy
    # pressure with weak own output -- see `_opp_switch_probability`; dampens only the
    # KO-dependent bonuses in `_score_attack_order` (our damage still lands on whatever
    # replaces them).
    opp_switch_prob: list[float]

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


def _build_context(battle: DoubleBattle, config: PolicyConfig) -> _Context:
    usage = load_usage_spreads()
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
    opp_threat_score = [0.0, 0.0]
    for our_idx in (0, 1):
        our_state = our_states[our_idx]
        if our_state is None:
            continue
        for opp_idx in (0, 1):
            opp_state = opp_states[opp_idx]
            opp_mon = opp_pokemon[opp_idx]
            if opp_state is None or opp_mon is None:
                continue
            move_ids = list(opp_mon.moves.keys()) if opp_mon.moves else []
            pct, move_id, priority = _best_attacking_move(
                opp_state, move_ids, our_state, field_vs_us
            )
            if pct > threat_on_us[our_idx].percent:
                threat_on_us[our_idx] = _ThreatInfo(percent=pct, move_id=move_id, priority=priority)
            opp_threat_score[opp_idx] = max(opp_threat_score[opp_idx], pct)

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
    for opp_idx in (0, 1):
        opp_state = opp_states[opp_idx]
        opp_mon = opp_pokemon[opp_idx]
        if opp_state is None or opp_mon is None:
            continue
        known_move_ids = {
            to_id(move_id) for move_id in (opp_mon.moves.keys() if opp_mon.moves else [])
        }
        has_self_protect_move = bool(known_move_ids & _SELF_PROTECT_MOVES)
        protect_counter = getattr(opp_mon, "protect_counter", 0)
        opp_protect_prob[opp_idx] = _opp_protect_probability(
            has_self_protect_move, pressure_on_opp[opp_idx], protect_counter, config
        )
        opp_switch_prob[opp_idx] = _opp_switch_probability(
            pressure_on_opp[opp_idx], opp_threat_score[opp_idx], config
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
    )


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
    survival_guard_slots: list[int] = []
    protect_prob_by_target: dict[int, float] = {}
    switch_prob_by_target: dict[int, float] = {}
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
            ko_bonus += config.guaranteed_ko_bonus
        elif is_likely:
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
        score -= (
            result.expected_percent
            * config.damage_percent_weight
            * config.ally_damage_penalty_weight
        )

    if config.mega_evolve_asap and getattr(single, "mega", False):
        score += 1e-3  # tie-breaker nudge only -- mega stats already drive the real gain

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
            "survival_guard_slots": survival_guard_slots,
            "protect_prob_by_target": protect_prob_by_target,
            "switch_prob_by_target": switch_prob_by_target,
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
    if move_id == "sleeppowder":
        return _score_sleep_powder(move_data, single, actor_slot, ctx, config)
    if move_id == "helpinghand":
        return 0.0, {}  # cross-slot value only -- see _cross_slot_adjustments
    if move_id == "partingshot":
        return _score_parting_shot(actor_slot, ctx, config)
    if move_id in _SCREEN_MOVE_TO_SIDE_CONDITION or move_id == "tailwind":
        return _score_screen(move_id, ctx, config)
    return 0.0, {"reason": "unmodeled_status_move"}


def _score_protect(actor_slot: int, ctx: _Context, config: PolicyConfig) -> tuple[float, dict]:
    threat = ctx.threat_on_us[actor_slot]
    pokemon = ctx.our_pokemon[actor_slot]
    protect_counter = getattr(pokemon, "protect_counter", 0) if pokemon is not None else 0

    score = threat.percent * config.protect_threat_weight
    if protect_counter >= 1:
        score -= config.protect_repeat_penalty
    if threat.percent < config.protect_low_threat_floor:
        score -= config.protect_low_threat_penalty
    return score, {"threat_percent": threat.percent, "protect_counter": protect_counter}


def _score_trick_room(ctx: _Context, config: PolicyConfig) -> tuple[float, dict]:
    if ctx.trick_room:
        return -config.trick_room_teardown_penalty, {"reason": "already_active"}
    our_alive, opp_alive = ctx.our_alive(), ctx.opp_alive()
    our_avg = mean([ctx.our_speed[i] for i in our_alive]) if our_alive else 0.0
    opp_avg = mean([ctx.opp_speed[i] for i in opp_alive]) if opp_alive else 0.0
    return (opp_avg - our_avg) * config.trick_room_setup_weight, {
        "our_avg_speed": our_avg,
        "opp_avg_speed": opp_avg,
    }


def _score_sleep_powder(
    move_data: dict, single: SingleBattleOrder, actor_slot: int, ctx: _Context, config: PolicyConfig
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
    if "Grass" in state.types() or state.ability == "overcoat":
        return 0.0, {"reason": "powder_immune"}
    value = config.sleep_powder_weight * (1.0 + ctx.opp_speed[idx] / 200.0)
    return value, {"target_slot": idx}


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
    else:
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
    for idx in opp_alive:
        opp_state = ctx.opp_states[idx]
        opp_mon = ctx.opp_pokemon[idx]
        if opp_state is None or opp_mon is None:
            continue
        pct_out, _, _ = _best_attacking_move(
            incoming_state, incoming_move_ids, opp_state, field_vs_opp
        )
        our_best = max(our_best, pct_out)
        opp_move_ids = list(opp_mon.moves.keys()) if opp_mon.moves else []
        pct_in, _, _ = _best_attacking_move(opp_state, opp_move_ids, incoming_state, field_vs_us)
        their_best = max(their_best, pct_in)

    matchup = (our_best - their_best) * config.switch_matchup_weight
    score = matchup - config.switch_tempo_cost
    if incoming.ability and to_id(incoming.ability) == "intimidate":
        eligible = [idx for idx in opp_alive if not _intimidate_immune(ctx.opp_pokemon[idx])]
        score += len(eligible) * config.intimidate_switch_bonus
    return score, {
        "matchup": matchup,
        "our_best_percent": our_best,
        "their_best_percent": their_best,
    }


def _cross_slot_adjustments(first_info: dict, second_info: dict, config: PolicyConfig) -> float:
    bonus = 0.0
    if first_info.get("move_id") == "helpinghand" and second_info.get("raw_damage_score"):
        bonus += float(second_info["raw_damage_score"]) * config.helping_hand_weight
    if second_info.get("move_id") == "helpinghand" and first_info.get("raw_damage_score"):
        bonus += float(first_info["raw_damage_score"]) * config.helping_hand_weight

    first_target = first_info.get("single_target_slot")
    second_target = second_info.get("single_target_slot")
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
    return bonus
