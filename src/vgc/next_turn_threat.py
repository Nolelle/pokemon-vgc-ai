"""Next-turn first-strike KO threat leaf term (A1) for the exact search.

`next_turn_threat_value(state, config)` prices imminent turn-(t+1) first-strike KO risk
at the end of a simulated turn in `vgc.rl.exact_search._position_value`:

    V_A1(s) = w_A1 * (Threat(our -> opp) - Threat(opp -> our))

The formulation enforces four invariants:
1. **No double-counting with turn-t damage**: a surviving active slot i with post-turn
   HP fraction h_i has marginal at-risk value `M_i = 100 * h_i + exact_search_alive_weight`.
   HP lost during turn t is already absent from `h_i`.
2. **Doubles contention**: each active attacker j chooses at most one move/target
   assignment on turn t+1 (single-target slot 0, single-target slot 1, or a spread move
   hitting both with the 0.75x spread multiplier). One faster foe with a single-target
   move cannot KO two slots simultaneously.
3. **Protect discount**: when `protect_counter == 0` and the defender has a self-Protect
   move available, a single-attacker threat (`k_i == 1`) is discounted by
   `SINGLE_THREAT_PROTECT_DISCOUNT` (0.50), whereas a two-attacker focus-fire threat
   (`k_i >= 2`) uses `FOCUS_FIRE_PROTECT_DISCOUNT` (0.85), and a slot with
   `protect_counter > 0` or no Protect receives no discount (1.0).
4. **Symmetry under post-turn speed order**: both sides are evaluated with the same
   `_side_first_strike_threat` function using `resolves_before` under the post-turn
   weather, terrain, Tailwind, and Trick Room state from `_Conditions(state, 1)`.
"""

from __future__ import annotations

from vgc.accuracy import hit_probability
from vgc.damage import FieldState, PokemonState, _is_grounded, damage_range
from vgc.data import load_moves
from vgc.evaluator import field_effective_speed, psychic_terrain_blocks, resolves_before
from vgc.field_control import _Conditions, _opponent_moves, _sole_ability
from vgc.mechanics_state import BattleMechanicsState, PokemonMechanicsState, SideMechanicsState
from vgc.models import PolicyConfig
from vgc.priority_rules import action_priority
from vgc.principles import SELF_PROTECT_MOVES
from vgc.sets import usage_spreads_for
from vgc.stats import default_opponent_nature, default_opponent_spread

# Threat severity floor (fraction of remaining HP) below which next-turn poke is not
# treated as an imminent KO threat.
THREAT_SEVERITY_FLOOR = 0.70

# Protect discount factors when `protect_counter == 0` and the defender can Protect.
SINGLE_THREAT_PROTECT_DISCOUNT = 0.50
FOCUS_FIRE_PROTECT_DISCOUNT = 0.85

_FIRST_TURN_ONLY_MOVES = frozenset({"fakeout", "firstimpression"})
_SCREEN_IDS = frozenset({"reflect", "lightscreen", "auroraveil"})
_CHOICE_ITEMS = frozenset({"choicescarf", "choiceband", "choicespecs"})


def _hp_fraction(mon: PokemonMechanicsState) -> float:
    if mon.fainted:
        return 0.0
    if mon.current_hp is None or not mon.max_hp:
        return 1.0
    return max(0.0, min(1.0, mon.current_hp / mon.max_hp))


def _active_mons(side: SideMechanicsState) -> list[PokemonMechanicsState]:
    return [mon for mon in side.pokemon if mon.active and not mon.fainted and mon.species_id][:2]


def _build_pokemon_state(
    mon: PokemonMechanicsState, *, ours: bool, config: PolicyConfig
) -> PokemonState:
    species_id = mon.species_id
    boosts = {name: int(stage) for name, stage in mon.boosts if stage}
    item = mon.item_id if (ours or mon.item_known) else None
    ability = mon.ability_id if (ours or mon.ability_known) else _sole_ability(species_id)
    sp_spread: dict[str, int] | None = None
    nature: str | None = None
    if ours and mon.stat_points:
        sp_spread = {k: int(v or 0) for k, v in mon.stat_points}
        nature = mon.nature or default_opponent_nature(species_id)
    else:
        usage = usage_spreads_for(config)
        entry = usage.get(species_id)
        spreads = entry.get("spreads") if isinstance(entry, dict) else None
        if spreads and isinstance(spreads[0], dict) and isinstance(spreads[0].get("sp"), dict):
            sp_spread = {k: int(v) for k, v in spreads[0]["sp"].items()}
            nature = spreads[0].get("nature") or default_opponent_nature(species_id)
        else:
            sp_spread = default_opponent_spread(species_id)
            nature = default_opponent_nature(species_id)
    state = PokemonState(
        species_id=species_id,
        sp_spread=sp_spread,
        nature=nature,
        boosts=boosts,
        status=mon.status,
        item=item,
        ability=ability,
        item_lost=bool(mon.item_lost_this_stint),
    )
    max_hp = state.max_hp()
    state.current_hp = max(1, round(_hp_fraction(mon) * max_hp))
    return state


def _usable_moves_next_turn(
    mon: PokemonMechanicsState, *, ours: bool, config: PolicyConfig
) -> list[str]:
    if ours:
        moves = [
            m.id
            for m in mon.moves
            if not m.disabled and (m.current_pp is None or m.current_pp > 0)
        ]
        if mon.item_id in _CHOICE_ITEMS and not mon.item_lost_this_stint and mon.last_move_id:
            locked = [m for m in moves if m == mon.last_move_id]
            if locked:
                return locked
        return moves

    if mon.item_known and mon.item_id in _CHOICE_ITEMS and mon.last_move_id:
        return [mon.last_move_id]
    revealed = tuple(
        sorted(
            m.id
            for m in mon.moves
            if not m.disabled and (m.current_pp is None or m.current_pp > 0)
        )
    )
    disabled_ids = {
        m.id for m in mon.moves if m.disabled or (m.current_pp is not None and m.current_pp <= 0)
    }
    inferred = _opponent_moves(mon.species_id, mon.base_species_id or "", revealed, config)
    return [m for m in inferred if m not in disabled_ids]


def _can_protect_next_turn(
    mon: PokemonMechanicsState, move_ids: list[str]
) -> bool:
    if mon.protect_counter > 0:
        return False
    if mon.status in ("slp", "frz") or mon.must_recharge:
        return False
    if any(eff.id in ("taunt", "encore", "choicelock") for eff in mon.effects):
        return False
    return any(m in SELF_PROTECT_MOVES for m in move_ids)


def _fastest_attack_priority(
    mon: PokemonMechanicsState,
    pstate: PokemonState,
    move_ids: list[str],
    terrain: str | None,
    config: PolicyConfig,
) -> float:
    moves_data = load_moves()
    best: float | None = None
    full_hp = pstate.hp_or_max() >= pstate.max_hp()
    grounded = _is_grounded(pstate)
    for move_id in move_ids:
        # An active Pokemon at the end of turn t will not be on its first turn at t+1.
        if move_id in _FIRST_TURN_ONLY_MOVES:
            continue
        data = moves_data.get(move_id)
        if data is None or data.get("category") == "Status":
            continue
        prio = action_priority(
            data,
            pstate.ability,
            full_hp,
            terrain,
            effective=config.effective_move_priority,
            grounded=grounded,
        )
        if best is None or prio > best:
            best = prio
    return 0.0 if best is None else best


def _threat_severity(damage_fraction: float) -> float:
    """Smooth severity in [0, 1] for next-turn first-strike damage fraction of remaining HP."""
    if damage_fraction >= 1.0:
        return 1.0
    if damage_fraction <= THREAT_SEVERITY_FLOOR:
        return 0.0
    normalized = (damage_fraction - THREAT_SEVERITY_FLOOR) / (1.0 - THREAT_SEVERITY_FLOOR)
    return normalized * normalized


def _attacker_action_vectors(
    att_mon: PokemonMechanicsState,
    att_state: PokemonState,
    att_speed: float,
    att_moves: list[str],
    att_ally_ability: str | None,
    def_mons: list[PokemonMechanicsState],
    def_states: list[PokemonState],
    def_speeds: list[float],
    def_priorities: list[float],
    def_hp_pcts: list[float],
    def_screens: frozenset[str],
    weather: str | None,
    terrain: str | None,
    trick_room: bool,
    config: PolicyConfig,
) -> list[tuple[float, ...]]:
    """Return candidate next-turn first-strike fractional damage vectors for one attacker.

    Each vector has length `len(def_mons)` and represents ONE legal action choice by
    `att_mon`: either a single-target attack onto one defender slot or a spread attack
    hitting all defender slots simultaneously.
    """
    n_def = len(def_mons)
    zero = tuple(0.0 for _ in range(n_def))
    if att_mon.status in ("slp", "frz") or att_mon.must_recharge:
        return [zero]

    moves_data = load_moves()
    best_single = [0.0 for _ in range(n_def)]
    best_spread: list[float] | None = None
    best_spread_sum = 0.0

    full_hp = att_state.hp_or_max() >= att_state.max_hp()
    grounded = _is_grounded(att_state)

    for move_id in att_moves:
        if move_id in _FIRST_TURN_ONLY_MOVES:
            continue
        data = moves_data.get(move_id)
        if data is None or data.get("category") == "Status":
            continue
        prio = action_priority(
            data,
            att_state.ability,
            full_hp,
            terrain,
            effective=config.effective_move_priority,
            grounded=grounded,
        )
        is_spread = data.get("target") in ("allAdjacentFoes", "allAdjacent") and n_def > 1
        num_targets = 2 if is_spread else 1
        if is_spread:
            vec = [0.0 for _ in range(n_def)]
            for i in range(n_def):
                if config.psychic_terrain_blocks_priority and psychic_terrain_blocks(
                    data, att_state, def_states[i], terrain
                ):
                    continue
                if not resolves_before(prio, att_speed, def_priorities[i], def_speeds[i], trick_room):
                    continue
                def_ally_ab = (
                    def_states[1 - i].ability
                    if n_def == 2 and config.partner_abilities_in_damage
                    else None
                )
                field = FieldState(
                    weather=weather,
                    terrain=terrain,
                    screens=def_screens,
                    trick_room=trick_room,
                    is_doubles=True,
                    num_targets=num_targets,
                    attacker_ally_ability=att_ally_ability if config.partner_abilities_in_damage else None,
                    defender_ally_ability=def_ally_ab,
                )
                res = damage_range(att_state, def_states[i], move_id, field)
                if not res.breakdown["move_supported"] or res.breakdown["immune"]:
                    continue
                hit_p = (
                    hit_probability(data, att_state, def_states[i], weather)
                    if config.model_move_accuracy
                    else 1.0
                )
                dmg_pct = res.expected_percent * hit_p
                vec[i] = min(1.0, dmg_pct / max(1.0, def_hp_pcts[i]))
            if sum(vec) > best_spread_sum:
                best_spread_sum = sum(vec)
                best_spread = vec
        else:
            for i in range(n_def):
                if config.psychic_terrain_blocks_priority and psychic_terrain_blocks(
                    data, att_state, def_states[i], terrain
                ):
                    continue
                if not resolves_before(prio, att_speed, def_priorities[i], def_speeds[i], trick_room):
                    continue
                def_ally_ab = (
                    def_states[1 - i].ability
                    if n_def == 2 and config.partner_abilities_in_damage
                    else None
                )
                field = FieldState(
                    weather=weather,
                    terrain=terrain,
                    screens=def_screens,
                    trick_room=trick_room,
                    is_doubles=True,
                    num_targets=1,
                    attacker_ally_ability=att_ally_ability if config.partner_abilities_in_damage else None,
                    defender_ally_ability=def_ally_ab,
                )
                res = damage_range(att_state, def_states[i], move_id, field)
                if not res.breakdown["move_supported"] or res.breakdown["immune"]:
                    continue
                hit_p = (
                    hit_probability(data, att_state, def_states[i], weather)
                    if config.model_move_accuracy
                    else 1.0
                )
                dmg_pct = res.expected_percent * hit_p
                frac = min(1.0, dmg_pct / max(1.0, def_hp_pcts[i]))
                if frac > best_single[i]:
                    best_single[i] = frac

    options: list[tuple[float, ...]] = [zero]
    for i in range(n_def):
        if best_single[i] > 0.0:
            single_vec = [0.0 for _ in range(n_def)]
            single_vec[i] = best_single[i]
            options.append(tuple(single_vec))
    if best_spread is not None and best_spread_sum > 0.0:
        options.append(tuple(best_spread))
    return options


def _side_first_strike_threat(
    att_side: SideMechanicsState,
    def_side: SideMechanicsState,
    *,
    attacker_is_ours: bool,
    att_tailwind: bool,
    def_tailwind: bool,
    weather: str | None,
    terrain: str | None,
    trick_room: bool,
    config: PolicyConfig,
) -> float:
    att_mons = _active_mons(att_side)
    def_mons = _active_mons(def_side)
    if not att_mons or not def_mons:
        return 0.0

    att_states = [
        _build_pokemon_state(mon, ours=attacker_is_ours, config=config) for mon in att_mons
    ]
    def_states = [
        _build_pokemon_state(mon, ours=not attacker_is_ours, config=config) for mon in def_mons
    ]
    att_speeds = [
        field_effective_speed(s, weather=weather, tailwind=att_tailwind, terrain=terrain)
        for s in att_states
    ]
    def_speeds = [
        field_effective_speed(s, weather=weather, tailwind=def_tailwind, terrain=terrain)
        for s in def_states
    ]
    att_moves = [
        _usable_moves_next_turn(mon, ours=attacker_is_ours, config=config) for mon in att_mons
    ]
    def_moves = [
        _usable_moves_next_turn(mon, ours=not attacker_is_ours, config=config) for mon in def_mons
    ]
    def_priorities = [
        _fastest_attack_priority(mon, st, moves, terrain, config)
        for mon, st, moves in zip(def_mons, def_states, def_moves, strict=True)
    ]
    def_hp_fracs = [_hp_fraction(mon) for mon in def_mons]
    def_hp_pcts = [100.0 * h for h in def_hp_fracs]
    def_marginal_values = [
        100.0 * h + config.exact_search_alive_weight for h in def_hp_fracs
    ]
    def_can_protect = [
        _can_protect_next_turn(mon, moves)
        for mon, moves in zip(def_mons, def_moves, strict=True)
    ]
    def_screens = frozenset(
        eff.id for eff in def_side.side_conditions if eff.id in _SCREEN_IDS
    )

    attacker_options: list[list[tuple[float, ...]]] = []
    for j, (att_mon, att_state, att_speed, moves) in enumerate(
        zip(att_mons, att_states, att_speeds, att_moves, strict=True)
    ):
        ally_ability = att_states[1 - j].ability if len(att_states) == 2 else None
        attacker_options.append(
            _attacker_action_vectors(
                att_mon,
                att_state,
                att_speed,
                moves,
                ally_ability,
                def_mons,
                def_states,
                def_speeds,
                def_priorities,
                def_hp_pcts,
                def_screens,
                weather,
                terrain,
                trick_room,
                config,
            )
        )

    if len(attacker_options) == 1:
        joint_assignments = [(v0,) for v0 in attacker_options[0]]
    else:
        joint_assignments = [
            (v0, v1) for v0 in attacker_options[0] for v1 in attacker_options[1]
        ]

    best_threat = 0.0
    n_def = len(def_mons)
    for assignment in joint_assignments:
        total = 0.0
        for i in range(n_def):
            combined_frac = min(1.0, sum(vec[i] for vec in assignment))
            severity = _threat_severity(combined_frac)
            if severity <= 0.0:
                continue
            contributors = sum(1 for vec in assignment if vec[i] > 0.0)
            if def_can_protect[i]:
                discount = (
                    SINGLE_THREAT_PROTECT_DISCOUNT
                    if contributors <= 1
                    else FOCUS_FIRE_PROTECT_DISCOUNT
                )
            else:
                discount = 1.0
            total += def_marginal_values[i] * severity * discount
        if total > best_threat:
            best_threat = total
    return best_threat


def next_turn_threat_value(state: BattleMechanicsState, config: PolicyConfig) -> float:
    """Symmetric next-turn first-strike KO risk delta (ours onto foe minus foe onto us)."""
    if state.won or state.lost or state.finished:
        return 0.0
    cond = _Conditions(state, 1)
    weather = cond.weather[0]
    terrain = cond.terrain[0]
    trick_room = cond.trick_room[0]
    our_tw = cond.our_tailwind[0]
    their_tw = cond.their_tailwind[0]

    our_threat_on_opp = _side_first_strike_threat(
        state.our_side,
        state.opponent_side,
        attacker_is_ours=True,
        att_tailwind=our_tw,
        def_tailwind=their_tw,
        weather=weather,
        terrain=terrain,
        trick_room=trick_room,
        config=config,
    )
    opp_threat_on_us = _side_first_strike_threat(
        state.opponent_side,
        state.our_side,
        attacker_is_ours=False,
        att_tailwind=their_tw,
        def_tailwind=our_tw,
        weather=weather,
        terrain=terrain,
        trick_room=trick_room,
        config=config,
    )
    return config.exact_search_next_turn_threat_weight * (our_threat_on_opp - opp_threat_on_us)
