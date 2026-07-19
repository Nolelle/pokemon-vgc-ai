"""Phase 2c shallow 2-ply search: anticipates the opponent's best responses instead of
scoring our move as if they stand still.

`search_joint_orders(battle, config)` is `vgc.evaluator.score_joint_orders`'s sibling
entry point -- `vgc.agent.VgcPlayer.decide()` calls this one instead when
`PolicyConfig.use_two_ply_search` is set (the default; see models.py). It reuses the
myopic evaluator for two things: the initial ranking (which orders are even worth
searching) and the shared `_Context` battle snapshot (`vgc.evaluator.build_context`,
promoted from private for exactly this reuse -- same pattern as `vgc.sets.normalize_item`
before it).

## Pipeline

1. `vgc.evaluator.score_joint_orders(battle, config)` for the myopic ranking. Empty in,
   empty out (mirrors that function's own documented contract).
2. Prune to the top `config.search_our_candidates` myopic orders -- searching every legal
   order (dozens, once switches/megas/targeting variants are enumerated) would multiply
   an already nontrivial per-order cost by too much; orders outside the myopic top-K are
   essentially never actually best, so they keep their myopic score (scaled by
   `search_myopic_weight`) and the returned list stays complete and consistently
   comparable (see `search_joint_orders`'s docstring for the exact blend).
3. `_enumerate_opp_responses(ctx, config)` builds a capped set of plausible opponent
   joint responses: per opponent slot, top known damaging moves/targets, Protect,
   strategic utility/control (setup, speed control, denial, redirection, screens), and
   the safest previewed defensive switches. "Known" moves come from
   `vgc.sets.opponent_move_ids` (revealed set plus replay-corpus priors). The per-slot
   choices are cross-producted and pruned to `search_opp_candidates` by a cheap
   enumeration-time score; Protect and switch likelihoods are calibrated separately.
4. `resolve_exchange(our_order, opp_response, ctx, config)` simulates ONE turn: our
   switches/mega resolve first (like the real engine), then every remaining move/Protect
   action (ours and theirs) executes in priority/speed order (Trick-Room-aware, reusing
   `resolves_before`'s comparison semantics), computing fresh `damage_range` against each
   action's CURRENT (possibly already-damaged, possibly Protected-into-0) defender state.
   Never mutates the shared `_Context`'s states -- everything operates on
   `dataclasses.replace`d copies.
5. Each candidate order's per-response exchange values are aggregated as a
   response-LIKELIHOOD-weighted expectation blended with a lighter worst case:
   `w * min(values) + (1 - w) * sum(p_i * values_i)` (`search_worst_case_weight` = w,
   dropped from 0.7 to 0.25 after v1 gate testing -- see below). Each response's weight
   `p_i` starts from a softmax over its cheap enumeration-time score
   (`p_i ∝ exp(enum_score_i / search_response_temperature)`), then gets multiplied by
   `ctx.opp_protect_prob[s]` for every Protect action it contains on opponent slot `s` --
   a hypothetical "they Protect" response is only as credible as the evaluator's own
   estimate that they actually would, not a certainty just because it survived
   enumeration. v1's pure `w * min + (1-w) * mean` treated a Protect-heavy worst case as
   equally likely regardless of how implausible that Protect actually was, which made the
   search systematically pessimistic about attacking (observed in gate traces: the bot
   preferred its own Protect over a 44-point-better attack, and dodged imaginary
   Protects by picking worse attack targets) -- this is why `search_worst_case_weight`'s
   default also dropped: the worst case is now a tail-risk hedge on top of an
   already-plausibility-weighted expectation, not the dominant term it was when the
   "expectation" term was a flat, unweighted mean.
6. Final score = `search_myopic_weight * myopic_score + search_position_weight *
   aggregated_exchange_value` -- but see `search_joint_orders`'s docstring for a
   ranking-safety fix on top of this: an unsearched (myopic-tail) order can never end up
   ranked above every searched order, regardless of what its bare myopic score is.

## v1 scope gaps (documented, not oversights)

- **Opponent switches use preview candidates.** The search retains the safest previewed
  bench states and weights them by pressure-derived switch probability. It cannot know
  which four were actually brought, and it does not predict the switch-in's following-
  turn move at this depth.
- **Only ONE ply of opponent response** -- this is a 2-ply search (our move, then their
  best response), not a full minimax tree. No modeling of what WE would do on the
  following turn.
- **No opponent mega evolution.** We don't know the opponent's revealed mega item is
  necessarily going to be used this exact turn, and modeling it would double the
  response-candidate space for a v1 feature; `vgc.evaluator.opp_threat_score` already
  factors the opponent's raw damage output (mega or not, whichever poke-env currently
  reports) into the myopic half.
- **Charge/recharge moves (Solar Beam, Hyper Beam, ...) are simulated naively.**
  `resolve_exchange` calls `damage_range` directly for whatever move a response/order
  names, with no notion that a charge move's damage this turn is conditional on
  actually having charged, or that a recharge move forfeits the FOLLOWING turn --
  `vgc.evaluator._score_attack_order`'s `charge_move_discount`/`recharge_move_discount`
  already reshape what reaches this search (a bad Solar Beam order is discounted before
  ever making the myopic top-K cut, so it's rarely even a search candidate), so this is
  an accepted v1 gap rather than a silent blind spot: the search's own per-exchange
  values for a charge/recharge move that DOES get searched are still optimistic (full
  damage, no "wasted the charge turn" or "no follow-up next turn" modeling), same
  reasoning as the no-opponent-switches gap above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import cmp_to_key, partial

from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.player.battle_order import DoubleBattleOrder

from vgc.actions import describe_order
from vgc.bc.policy import (
    exchange_state_record,
    load_bc_policy,
    position_value,
    position_values_batch,
)
from vgc.damage import FieldState, PokemonState, damage_range, to_id
from vgc.data import load_moves
from vgc.decision_trace import record_note
from vgc.evaluator import (
    _ABILITY_WEATHER,
    _PROTECT_MOVES,
    _SELF_PROTECT_MOVES,
    _SPREAD_TARGETS_FOES_ONLY,
    _SPREAD_TARGETS_HITTING_ALLY,
    _Context,
    _best_attacking_move,
    _our_pokemon_state,
    _resolve_targets,
    ScoredOrder,
    build_context,
    field_effective_speed,
    mega_evolved_state,
    resolves_before,
    score_joint_orders,
)
from vgc.models import PolicyConfig
from vgc.principles import REDIRECTION_MOVES, utility_kind
from vgc.sets import load_usage_spreads, opponent_move_ids, opponent_state

# --- opponent response candidates ---------------------------------------------------------


@dataclass(frozen=True)
class _OppSlotAction:
    """One hypothetical action for a single opponent slot this turn.

    :param kind: ``"move"`` | ``"protect"`` | ``"none"`` (fainted/empty slot, or no
        legal candidate found for it -- treated as a pass).
    :param target_our_slot: for a single-target move, which of OUR alive slots it hits;
        `None` for a spread move (hits every eligible target the move's own `target`
        string implies) or for `kind != "move"`.
    :param value: cheap enumeration-time score used only to rank/prune joint response
        candidates in `_enumerate_opp_responses` -- NOT reused by `resolve_exchange`,
        which recomputes real damage fresh against live (possibly already-damaged)
        states.
    """

    kind: str
    move_id: str | None = None
    target_our_slot: int | None = None
    value: float = 0.0
    switch_state: PokemonState | None = None
    switch_species: str | None = None
    utility_value: float = 0.0


@dataclass(frozen=True)
class OppResponse:
    """One hypothetical joint opponent response (`_OppSlotAction` per opponent slot,
    index-aligned with `_Context.opp_pokemon`/`opp_states`).

    :param enum_score: the cheap enumeration-time joint score `_enumerate_opp_responses`
        already sorts/prunes by (sum of each slot action's `value`) -- threaded onto the
        response itself so `_aggregate_exchange_values` can turn it into a response-
        likelihood weight without re-deriving it (see that function's docstring).
    """

    slot0: _OppSlotAction
    slot1: _OppSlotAction
    enum_score: float = 0.0

    def describe(self) -> str:
        return f"{_describe_opp_slot_action(self.slot0)} / {_describe_opp_slot_action(self.slot1)}"


def _describe_opp_slot_action(action: _OppSlotAction) -> str:
    if action.kind == "none":
        return "pass"
    if action.kind == "protect":
        return action.move_id or "protect"
    if action.kind == "switch":
        return f"switch->{action.switch_species or 'unknown'}"
    target = f"@our{action.target_our_slot}" if action.target_our_slot is not None else ""
    return f"{action.move_id}{target}"


def _our_pressure_on_opp_slot(ctx: _Context, opp_idx: int) -> float:
    """Our best expected % onto `opp_idx` from either of our alive actives.

    Mirrors `vgc.evaluator.build_context`'s own (locally-scoped, not exported)
    `pressure_on_opp` computation; recomputed here instead of reading it off `_Context`
    to keep that dataclass's public contract unchanged for this feature -- a handful of
    extra `damage_range` calls, not a hot path (only called once per opponent slot that
    has a Protect candidate).
    """
    opp_state = ctx.opp_states[opp_idx]
    if opp_state is None:
        return 0.0
    field_vs_opp = ctx.field_state(defender_is_ours=False, num_targets=1)
    best = 0.0
    for our_idx in ctx.our_alive():
        our_state = ctx.our_states[our_idx]
        our_mon = ctx.our_pokemon[our_idx]
        if our_state is None or our_mon is None:
            continue
        our_move_ids = list(our_mon.moves.keys()) if our_mon.moves else []
        pct, _, _ = _best_attacking_move(our_state, our_move_ids, opp_state, field_vs_opp)
        best = max(best, pct)
    return best


def _opp_slot_candidates(opp_idx: int, ctx: _Context, config: PolicyConfig) -> list[_OppSlotAction]:
    """Candidate actions for one opponent slot: the top `search_opp_moves_per_slot`
    known damaging (move, target) pairs by expected damage, plus an always-included
    Protect candidate when available and unused last turn. See module docstring point 3.
    """
    opp_mon = ctx.opp_pokemon[opp_idx]
    opp_state = ctx.opp_states[opp_idx]
    if opp_mon is None or opp_state is None or opp_idx not in ctx.opp_alive():
        return [_OppSlotAction(kind="none")]

    move_ids = opponent_move_ids(opp_mon, priors=ctx.priors, config=config)
    our_alive = ctx.our_alive()
    field_vs_us_single = ctx.field_state(defender_is_ours=True, num_targets=1)
    field_vs_us_spread = ctx.field_state(defender_is_ours=True, num_targets=max(1, len(our_alive)))

    scored: list[tuple[float, _OppSlotAction]] = []
    for raw_move_id in move_ids:
        move_id = to_id(raw_move_id)
        move_data = load_moves().get(move_id)
        if move_data is None or move_data["category"] == "Status":
            continue  # only "known damaging move" candidates -- see module docstring
        target_kind = move_data.get("target")
        if target_kind in _SPREAD_TARGETS_FOES_ONLY or target_kind in _SPREAD_TARGETS_HITTING_ALLY:
            total_pct = 0.0
            supported = False
            for our_idx in our_alive:
                our_state = ctx.our_states[our_idx]
                if our_state is None:
                    continue
                result = damage_range(opp_state, our_state, move_id, field_vs_us_spread)
                if result.breakdown["move_supported"] and not result.breakdown["immune"]:
                    supported = True
                total_pct += result.expected_percent
            if supported:
                scored.append(
                    (total_pct, _OppSlotAction(kind="move", move_id=move_id, value=total_pct))
                )
        else:
            for our_idx in our_alive:
                our_state = ctx.our_states[our_idx]
                if our_state is None:
                    continue
                result = damage_range(opp_state, our_state, move_id, field_vs_us_single)
                if not result.breakdown["move_supported"] or result.breakdown["immune"]:
                    continue
                scored.append(
                    (
                        result.expected_percent,
                        _OppSlotAction(
                            kind="move",
                            move_id=move_id,
                            target_our_slot=our_idx,
                            value=result.expected_percent,
                        ),
                    )
                )

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top_k = max(0, config.search_opp_moves_per_slot)
    candidates = [action for _, action in scored[:top_k]]

    # move_ids is already to_id-normalized (opponent_move_ids' own contract).
    protect_move_id = next(iter(sorted(set(move_ids) & _SELF_PROTECT_MOVES)), None)
    protect_counter = getattr(opp_mon, "protect_counter", 0)
    if protect_move_id is not None and protect_counter == 0:
        protect_value = _our_pressure_on_opp_slot(ctx, opp_idx) * config.protect_threat_weight
        candidates.append(
            _OppSlotAction(kind="protect", move_id=protect_move_id, value=protect_value)
        )

    utility_actions: list[_OppSlotAction] = []
    for move_id in move_ids:
        kind = utility_kind(move_id)
        if kind is None or kind == "protect":
            continue
        kind_scale = {
            "speed_control": 1.0,
            "action_denial": 1.0,
            "redirection": 0.9,
            "setup": 0.9,
            "screen": 0.7,
            "burn": 0.7,
            "wide_defense": 0.8,
            "recovery": 0.5,
            "pivot": 0.6,
        }.get(kind, 0.5)
        strategic_value = config.search_opp_utility_weight * kind_scale
        utility_actions.append(
            _OppSlotAction(
                kind="utility",
                move_id=move_id,
                value=strategic_value,
                utility_value=strategic_value,
            )
        )
    utility_actions.sort(key=lambda action: action.value, reverse=True)
    candidates.extend(utility_actions[: max(0, config.search_opp_utility_per_slot)])

    # Previewed-but-not-active Pokemon are plausible defensive pivots. Retain the switch-
    # ins that take the least estimated damage from our current board; response weighting
    # below scales them by the evaluator's pressure-derived switch probability.
    preview = list(getattr(ctx.battle, "teampreview_opponent_team", None) or [])
    active_species = {
        to_id(mon.species) for mon in ctx.opp_pokemon if mon is not None and not mon.fainted
    }
    usage = load_usage_spreads()
    switch_actions: list[_OppSlotAction] = []
    for bench_mon in preview:
        species_id = to_id(getattr(bench_mon, "species", None))
        if not species_id or species_id in active_species or getattr(bench_mon, "fainted", False):
            continue
        bench_state = opponent_state(bench_mon, usage=usage)
        worst_incoming = 0.0
        for our_idx in ctx.our_alive():
            our_state = ctx.our_states[our_idx]
            our_mon = ctx.our_pokemon[our_idx]
            if our_state is None or our_mon is None:
                continue
            pct, _, _ = _best_attacking_move(
                our_state,
                list(our_mon.moves.keys()) if our_mon.moves else [],
                bench_state,
                ctx.field_state(defender_is_ours=False, num_targets=1),
            )
            worst_incoming = max(worst_incoming, pct)
        switch_value = max(0.0, 100.0 - worst_incoming)
        switch_actions.append(
            _OppSlotAction(
                kind="switch",
                value=switch_value,
                switch_state=bench_state,
                switch_species=species_id,
            )
        )
    switch_actions.sort(key=lambda action: action.value, reverse=True)
    candidates.extend(switch_actions[: max(0, config.search_opp_switches_per_slot)])

    if not candidates:
        candidates.append(_OppSlotAction(kind="none"))
    return candidates


def _enumerate_opp_responses(ctx: _Context, config: PolicyConfig) -> list[OppResponse]:
    """Cross product of each slot's `_opp_slot_candidates`, pruned to the top
    `search_opp_candidates` by a cheap enumeration-time joint score (sum of each slot
    action's `value`). See module docstring point 3 and its v1-scope-gaps section.
    """
    per_slot = [_opp_slot_candidates(0, ctx, config), _opp_slot_candidates(1, ctx, config)]

    joint: list[tuple[float, OppResponse]] = []
    for slot0_action in per_slot[0]:
        for slot1_action in per_slot[1]:
            if (
                slot0_action.kind == "switch"
                and slot1_action.kind == "switch"
                and slot0_action.switch_species == slot1_action.switch_species
            ):
                continue
            joint_score = slot0_action.value + slot1_action.value
            joint.append(
                (
                    joint_score,
                    OppResponse(slot0=slot0_action, slot1=slot1_action, enum_score=joint_score),
                )
            )
    joint.sort(key=lambda pair: pair[0], reverse=True)
    cap = max(1, config.search_opp_candidates)
    return [response for _, response in joint[:cap]]


# --- exchange resolution ---------------------------------------------------------------


@dataclass
class ExchangeResult:
    """Per-side totals from one simulated exchange (see `resolve_exchange`).

    `hp_lost_pct` sums, over that side's mons, the damage each took THIS exchange as a
    percent of ITS OWN max HP (capped at whatever HP it had left when hit, so overkill
    damage past 0 doesn't inflate the total). `faints` counts mons that crossed from
    HP > 0 to HP <= 0 during this exchange.

    `our_states`/`opp_states`/`weather` are the POST-exchange snapshot (copies -- see
    `resolve_exchange`'s "never mutate ctx" contract) -- only populated for
    `PolicyConfig.use_value_head`'s benefit (`vgc.bc.policy.exchange_state_record`
    builds a value-head-ready record from them); every other caller only reads the
    HP/faint totals above, same as before this field existed.
    """

    our_hp_lost_pct: float = 0.0
    our_faints: int = 0
    opp_hp_lost_pct: float = 0.0
    opp_faints: int = 0
    our_states: list[PokemonState | None] = field(default_factory=list)
    opp_states: list[PokemonState | None] = field(default_factory=list)
    weather: str | None = None
    our_utility_value: float = 0.0
    opp_utility_value: float = 0.0


@dataclass
class _Action:
    """One resolved (move or Protect) action pending in an exchange's turn order.
    Switches/mega evolution are NOT actions here -- they resolve immediately while
    building this list (see `_build_our_actions`), before any speed/priority ordering.
    """

    side: str  # "our" | "opp"
    slot: int
    kind: str  # "move" | "protect" | "utility"
    move_id: str | None
    targets: list[tuple[str, int, bool]]  # (side, idx, is_ally_of_the_actor)
    priority: int
    speed: float = 0.0  # filled in once the exchange's weather/mega state is settled
    utility_value: float = 0.0
    spread: bool = False


def _copy_state(state: PokemonState | None) -> PokemonState | None:
    """A `dataclasses.replace` copy with its OWN `boosts` dict (never share the mutable
    dict with `_Context`'s original -- see module docstring point 4's "never mutate ctx"
    contract).
    """
    if state is None:
        return None
    return replace(state, boosts=dict(state.boosts))


def _build_our_actions(
    our_order: DoubleBattleOrder,
    ctx: _Context,
    our_states: list[PokemonState | None],
    config: PolicyConfig,
) -> tuple[list[_Action], str | None]:
    """Resolve OUR switches/mega evolution immediately (mutating `our_states` in place,
    exactly as the real engine resolves them before any move executes), then return the
    move/Protect actions for slots that actually act this turn, plus the weather
    override implied by a mega evolution granting Drought/Drizzle (`None` if no override
    -- mirrors `vgc.evaluator._score_attack_order`'s `weather_for_this_order`).
    """
    actions: list[_Action] = []
    weather_override: str | None = None
    orders = (our_order.first_order, our_order.second_order)
    for slot, single in enumerate(orders):
        if single is None:
            continue
        target = single.order
        if isinstance(target, Pokemon):
            our_states[slot] = _our_pokemon_state(target)  # switch: no action this turn
            continue
        if not isinstance(target, Move):
            continue  # PassBattleOrder / str message order -- no-op
        move_id = to_id(target.id)
        move_data = load_moves().get(move_id)
        if move_data is None:
            continue
        if getattr(single, "mega", False):
            state = our_states[slot]
            if state is not None:
                our_states[slot] = mega_evolved_state(state)
                if our_states[slot].ability in _ABILITY_WEATHER:
                    weather_override = _ABILITY_WEATHER[our_states[slot].ability]
        if move_data["category"] == "Status":
            if move_id in _PROTECT_MOVES:
                actions.append(
                    _Action(
                        side="our",
                        slot=slot,
                        kind="protect",
                        move_id=move_id,
                        targets=[],
                        priority=int(move_data.get("priority", 0)),
                    )
                )
            else:
                kind = utility_kind(move_id)
                if kind is not None:
                    utility_scale = {
                        "speed_control": 1.0,
                        "action_denial": 1.0,
                        "redirection": 0.9,
                        "setup": 0.9,
                        "screen": 0.7,
                        "burn": 0.7,
                        "wide_defense": 0.8,
                        "recovery": 0.5,
                        "pivot": 0.6,
                    }.get(kind, 0.5)
                    actions.append(
                        _Action(
                            side="our",
                            slot=slot,
                            kind="utility",
                            move_id=move_id,
                            targets=[],
                            priority=int(move_data.get("priority", 0)),
                            utility_value=config.search_opp_utility_weight * utility_scale,
                        )
                    )
            continue
        targets = _resolve_targets(move_data, slot, single.move_target, ctx)
        side_tagged = [("opp", idx, False) for idx, is_ally in targets if not is_ally] + [
            ("our", idx, True) for idx, is_ally in targets if is_ally
        ]
        actions.append(
            _Action(
                side="our",
                slot=slot,
                kind="move",
                move_id=move_id,
                targets=side_tagged,
                priority=int(move_data.get("priority", 0)),
                spread=move_data.get("target")
                in (_SPREAD_TARGETS_FOES_ONLY | _SPREAD_TARGETS_HITTING_ALLY),
            )
        )
    return actions, weather_override


def _build_opp_actions(opp_response: OppResponse, ctx: _Context) -> list[_Action]:
    """Turn non-switch response choices into executable move/Protect/utility actions."""
    actions: list[_Action] = []
    for slot, slot_action in enumerate((opp_response.slot0, opp_response.slot1)):
        if slot_action.kind in {"none", "switch"}:
            continue
        move_data = load_moves().get(slot_action.move_id) if slot_action.move_id else None
        if slot_action.kind == "protect":
            actions.append(
                _Action(
                    side="opp",
                    slot=slot,
                    kind="protect",
                    move_id=slot_action.move_id,
                    targets=[],
                    priority=int((move_data or {}).get("priority", 0)),
                )
            )
            continue
        if slot_action.kind == "utility":
            if move_data is None:
                continue
            actions.append(
                _Action(
                    side="opp",
                    slot=slot,
                    kind="utility",
                    move_id=slot_action.move_id,
                    targets=[],
                    priority=int(move_data.get("priority", 0)),
                    utility_value=slot_action.utility_value,
                )
            )
            continue
        if move_data is None:
            continue
        if slot_action.target_our_slot is not None:
            targets = [("our", slot_action.target_our_slot, False)]
        else:
            # Spread move -- resolve via the same target-kind sets the evaluator uses,
            # against the CURRENT (pre-exchange) alive sets (stable until actions apply).
            our_alive = ctx.our_alive()
            opp_allies_alive = [i for i in ctx.opp_alive() if i != slot]
            target_kind = move_data.get("target")
            if target_kind in _SPREAD_TARGETS_HITTING_ALLY:
                targets = [("our", i, False) for i in our_alive] + [
                    ("opp", i, True) for i in opp_allies_alive
                ]
            else:  # _SPREAD_TARGETS_FOES_ONLY
                targets = [("our", i, False) for i in our_alive]
        actions.append(
            _Action(
                side="opp",
                slot=slot,
                kind="move",
                move_id=slot_action.move_id,
                targets=targets,
                priority=int(move_data.get("priority", 0)),
                spread=move_data.get("target")
                in (_SPREAD_TARGETS_FOES_ONLY | _SPREAD_TARGETS_HITTING_ALLY),
            )
        )
    return actions


def _action_order_cmp(a: _Action, b: _Action, trick_room: bool) -> int:
    """`functools.cmp_to_key` comparator building the exchange's total action order from
    `resolves_before`'s pairwise "does A act before B" semantics.
    """
    a_before_b = resolves_before(a.priority, a.speed, b.priority, b.speed, trick_room)
    b_before_a = resolves_before(b.priority, b.speed, a.priority, a.speed, trick_room)
    if a_before_b and not b_before_a:
        return -1
    if b_before_a and not a_before_b:
        return 1
    # A genuine tie (equal priority AND speed -- resolves_before's own "ties favor the
    # first argument" rule makes both directions True here) -- break it our-side-first,
    # applying that same "favor the first/our side" spirit one level up.
    if a.side != b.side:
        return -1 if a.side == "our" else 1
    return 0


def _apply_action(
    action: _Action,
    our_states: list[PokemonState | None],
    opp_states: list[PokemonState | None],
    our_protected: list[bool],
    opp_protected: list[bool],
    our_redirector: list[int | None],
    opp_redirector: list[int | None],
    weather_for_exchange: str | None,
    ctx: _Context,
    result: ExchangeResult,
) -> None:
    actor_states = our_states if action.side == "our" else opp_states
    actor_state = actor_states[action.slot]
    if actor_state is None or actor_state.hp_or_max() <= 0:
        return  # a fainted actor (from an earlier action this exchange) does not act

    if action.kind == "protect":
        protected = our_protected if action.side == "our" else opp_protected
        protected[action.slot] = True
        return
    if action.kind == "utility":
        if action.side == "our":
            result.our_utility_value += action.utility_value
            if action.move_id in REDIRECTION_MOVES:
                our_redirector[0] = action.slot
        else:
            result.opp_utility_value += action.utility_value
            if action.move_id in REDIRECTION_MOVES:
                opp_redirector[0] = action.slot
        return

    num_targets = len(action.targets)
    field_vs_us = FieldState(
        weather=weather_for_exchange,
        terrain=ctx.terrain,
        screens=ctx.our_side_screens,
        trick_room=ctx.trick_room,
        is_doubles=True,
        num_targets=num_targets,
    )
    field_vs_opp = FieldState(
        weather=weather_for_exchange,
        terrain=ctx.terrain,
        screens=ctx.opp_side_screens,
        trick_room=ctx.trick_room,
        is_doubles=True,
        num_targets=num_targets,
    )
    for side, original_idx, _is_ally in action.targets:
        idx = original_idx
        if not action.spread and side != action.side:
            redirector = our_redirector[0] if side == "our" else opp_redirector[0]
            if redirector is not None:
                idx = redirector
        defender_states = our_states if side == "our" else opp_states
        defender_state = defender_states[idx]
        if defender_state is None or defender_state.hp_or_max() <= 0:
            continue  # already fainted earlier this exchange -- no damage to deal
        protected = our_protected if side == "our" else opp_protected
        if protected[idx]:
            continue  # Protect blanks single-target AND spread hits alike (v1 scope)
        field = field_vs_us if side == "our" else field_vs_opp
        damage_result = damage_range(actor_state, defender_state, action.move_id, field)
        before = defender_state.hp_or_max()
        actual_loss = min(damage_result.expected_damage, before)
        defender_state.current_hp = max(0.0, before - actual_loss)
        max_hp = defender_state.max_hp()
        loss_pct = (actual_loss / max_hp * 100.0) if max_hp else 0.0
        newly_fainted = before > 0 and defender_state.current_hp <= 0
        if side == "our":
            result.our_hp_lost_pct += loss_pct
            if newly_fainted:
                result.our_faints += 1
        else:
            result.opp_hp_lost_pct += loss_pct
            if newly_fainted:
                result.opp_faints += 1


def resolve_exchange(
    our_order: DoubleBattleOrder, opp_response: OppResponse, ctx: _Context, config: PolicyConfig
) -> ExchangeResult:
    """Simulate one turn: `our_order` vs one hypothetical `opp_response`, starting from
    `ctx`'s snapshot. Operates entirely on copies of `ctx`'s `PokemonState`s -- `ctx`
    itself is never mutated, so the same `_Context` is reused across every candidate
    order and every opponent response in one `search_joint_orders` call.

    ``config`` is accepted (per this module's documented entry-point signature) but not
    read here -- every weight this function's OWN mechanics need (none; it just adds up
    damage/faints) lives one level up in `_exchange_value`, matching how `damage_range`
    itself doesn't take a `PolicyConfig` either.
    """
    our_states = [_copy_state(state) for state in ctx.our_states]
    opp_states = [_copy_state(state) for state in ctx.opp_states]

    our_actions, weather_override = _build_our_actions(our_order, ctx, our_states, config)
    weather_for_exchange = weather_override if weather_override is not None else ctx.weather
    for slot, slot_action in enumerate((opp_response.slot0, opp_response.slot1)):
        if slot_action.kind == "switch" and slot_action.switch_state is not None:
            opp_states[slot] = _copy_state(slot_action.switch_state)
    opp_actions = _build_opp_actions(opp_response, ctx)

    our_tailwind = SideCondition.TAILWIND in ctx.battle.side_conditions
    opp_tailwind = SideCondition.TAILWIND in ctx.battle.opponent_side_conditions
    for action in our_actions:
        state = our_states[action.slot]
        action.speed = (
            field_effective_speed(state, weather=weather_for_exchange, tailwind=our_tailwind)
            if state is not None
            else 0.0
        )
    for action in opp_actions:
        state = opp_states[action.slot]
        action.speed = (
            field_effective_speed(state, weather=weather_for_exchange, tailwind=opp_tailwind)
            if state is not None
            else 0.0
        )

    all_actions = our_actions + opp_actions
    all_actions.sort(key=cmp_to_key(partial(_action_order_cmp, trick_room=ctx.trick_room)))

    result = ExchangeResult()
    our_protected = [False, False]
    opp_protected = [False, False]
    our_redirector: list[int | None] = [None]
    opp_redirector: list[int | None] = [None]
    for action in all_actions:
        _apply_action(
            action,
            our_states,
            opp_states,
            our_protected,
            opp_protected,
            our_redirector,
            opp_redirector,
            weather_for_exchange,
            ctx,
            result,
        )
    result.our_states = our_states
    result.opp_states = opp_states
    result.weather = weather_for_exchange
    return result


def _response_weights(
    responses: list[OppResponse], ctx: _Context, config: PolicyConfig
) -> list[float]:
    """Response-likelihood weights for `_aggregate_exchange_values`'s expectation term.

    Starts from a softmax over each response's cheap enumeration-time score
    (`p_i ∝ exp(enum_score_i / search_response_temperature)`, computed with the
    standard max-subtraction trick for numerical stability), then multiplies each
    response's raw weight by `ctx.opp_protect_prob[s]` for every Protect action it
    contains on opponent slot `s` -- a response where a slot Protects is only as
    credible as the evaluator's OWN estimate that they'd actually do that (already
    computed from Open Team Sheets + pressure in `vgc.evaluator.build_context`), not a
    certainty just because it's in the enumerated candidate set. Renormalized to sum to
    1; falls back to a uniform distribution if every weight comes out <= 0 (e.g. every
    response contains a nearly-impossible Protect).
    """
    if not responses:
        return []
    temperature = config.search_response_temperature or 1e-6
    scaled_scores = [response.enum_score / temperature for response in responses]
    max_score = max(scaled_scores)
    raw_weights = [math.exp(score - max_score) for score in scaled_scores]

    for i, response in enumerate(responses):
        for slot_idx, slot_action in enumerate((response.slot0, response.slot1)):
            if slot_action.kind == "protect":
                raw_weights[i] *= ctx.opp_protect_prob[slot_idx]
            elif slot_action.kind == "switch":
                raw_weights[i] *= max(0.05, ctx.opp_switch_prob[slot_idx])

    total = sum(raw_weights)
    if total <= 0.0:
        uniform = 1.0 / len(responses)
        return [uniform] * len(responses)
    return [weight / total for weight in raw_weights]


def _aggregate_exchange_values(
    values: list[float], responses: list[OppResponse], ctx: _Context, config: PolicyConfig
) -> float:
    """Aggregate one candidate order's per-response exchange values (`values`, index-
    aligned with `responses`) into a single number: `w * worst_case + (1 - w) *
    sum(p_i * values_i)`, `w = search_worst_case_weight`, `p_i` from `_response_weights`.

    NOT a pure minimax (`w` well below 1.0 by default) -- a real ladder opponent isn't a
    perfect adversarial optimizer, and v1's flat, unweighted mean let every Protect-heavy
    worst case drag every attacking order down equally regardless of how plausible that
    Protect actually was (see module docstring point 5) -- the worst-case term is now a
    tail-risk hedge on top of a plausibility-weighted expectation, not competing against
    an equally-naive average. Returns 0.0 for an empty `values` list (no responses).
    """
    if not values:
        return 0.0
    worst_case = min(values)
    weights = _response_weights(responses, ctx, config)
    expectation = sum(weight * value for weight, value in zip(weights, values, strict=True))
    return (
        config.search_worst_case_weight * worst_case
        + (1.0 - config.search_worst_case_weight) * expectation
    )


def _exchange_value(result: ExchangeResult, config: PolicyConfig) -> float:
    """Exchange value = (their loss) - (our loss), in `damage_percent_weight`'s points
    currency. Deliberately a DELTA, not an absolute score: neither side's unknown bench
    reserves are modeled (see module docstring's scope gaps), so an absolute score would
    implicitly assume "no more Pokemon exist for either side" -- wrong for both sides
    EQUALLY, which is exactly what cancels out by only ever looking at the relative
    outcome of this one simulated exchange.
    """
    opp_loss = (
        result.opp_hp_lost_pct * config.search_hp_weight
        + result.opp_faints * config.search_faint_weight
    )
    our_loss = (
        result.our_hp_lost_pct * config.search_hp_weight
        + result.our_faints * config.search_faint_weight
    )
    return opp_loss - our_loss + result.our_utility_value - result.opp_utility_value


def _value_head_delta(v_after: float | None, v_before: float | None, config: PolicyConfig) -> float:
    """`config.value_head_weight * 100 * (v_after - v_before)` -- the outcome value
    head's opinion of how much an exchange's resulting position improved/worsened our
    win probability, in the same percent-of-HP currency `_exchange_value` already uses
    (see `PolicyConfig.value_head_weight`'s comment for the 100x scale). Returns 0.0 (a
    true no-op, not just "small") whenever either probability is unavailable -- `v_after`/
    `v_before` being `None` covers every disabled/degraded case upstream (flag off,
    checkpoint missing, checkpoint has no value head -- see `vgc.bc.policy.position_value`
    and `position_values_batch`'s own None-means-disabled contracts), so this function
    itself doesn't need to know WHY a probability is missing, only that it is. Pure
    arithmetic (no policy/state lookups of its own) so `search_joint_orders` can call
    `vgc.bc.policy.position_values_batch` ONCE for every (candidate, response) pair
    instead of once per pair -- see that function's docstring for the latency this
    batching fixes (~40ms/decision unbatched vs a <5ms budget).
    """
    if v_after is None or v_before is None:
        return 0.0
    return config.value_head_weight * 100.0 * (v_after - v_before)


# --- top-level entry point ---------------------------------------------------------------


def search_joint_orders(
    battle: DoubleBattle, config: PolicyConfig | None = None
) -> list[ScoredOrder]:
    """Score every legal joint order for the current turn using the shallow 2-ply
    search, best first. See the module docstring for the full pipeline.

    Returns `[]` under the exact same conditions `vgc.evaluator.score_joint_orders`
    does (nothing legal this turn) -- callers should fall back the same way.

    Ranking-safety fix (gate regression 2026-07: search v1 dropped vs-heuristic win rate
    from 81.7% to 71.3%, traced partly to this): a myopic-tail ("unsearched") order keeps
    only its bare `myopic_score * search_myopic_weight`, while every SEARCHED order's
    final score includes `search_position_weight * aggregated_exchange_value`, which is
    usually negative (a real opponent almost always has SOME punishing response). That
    let a mediocre order that escaped scrutiny outrank every order the search actually
    vetted -- observed live: an unsearched order (myopic 116) beat the searched myopic-
    best (myopic 219). Every unsearched order's score is therefore capped BELOW the
    lowest searched final score (`min(searched finals) - 1.0 - tail_index`, strictly
    descending so the tail's own myopic order is preserved beneath the searched block,
    never above it) instead of using its own myopic-derived score directly.
    """
    config = config or PolicyConfig()
    myopic = score_joint_orders(battle, config)
    if not myopic:
        return []

    ctx = build_context(battle, config)
    responses = _enumerate_opp_responses(ctx, config)

    # Outcome value head (opt-in, see PolicyConfig.use_value_head's comment): v_before
    # is the CURRENT position's value, computed ONCE and reused across every candidate x
    # response pair below (it doesn't depend on either) -- only loaded/computed at all
    # when the flag is on, so this is a true no-op (not even a checkpoint load attempt)
    # when it's off, matching every other opt-in knob's graceful-degradation contract.
    # `record_cache` is shared across every exchange_state_record call this decision --
    # bench/side-conditions/field are ctx-derived and (almost always) identical across
    # every exchange, see that function's `cache` param docstring for the latency this
    # avoids recomputing buys back.
    value_policy = None
    v_before: float | None = None
    record_cache: dict = {}
    if config.use_value_head:
        value_policy = load_bc_policy(config.bc_checkpoint_path)
        if value_policy is not None and value_policy.has_value_head:
            v_before = position_value(
                value_policy,
                exchange_state_record(ctx.our_states, ctx.opp_states, ctx, cache=record_cache),
            )

    cutoff = max(1, config.search_our_candidates)
    searched, unsearched = myopic[:cutoff], myopic[cutoff:]

    # Every (candidate, response) exchange is resolved FIRST, across the whole searched
    # block, so the value head (if active) can be scored in ONE batched forward pass
    # over all of them (vgc.bc.policy.position_values_batch) instead of one call per
    # pair -- see that function's docstring for the latency this fixes.
    exchanges_by_entry: list[list[ExchangeResult]] = [
        [resolve_exchange(entry.order, response, ctx, config) for response in responses]
        for entry in searched
    ]
    v_after_by_entry: list[list[float | None]]
    if v_before is not None:
        flat_records = [
            exchange_state_record(exchange.our_states, exchange.opp_states, ctx, cache=record_cache)
            for exchanges in exchanges_by_entry
            for exchange in exchanges
        ]
        flat_v_after = position_values_batch(value_policy, flat_records)
        v_after_by_entry = []
        cursor = 0
        for exchanges in exchanges_by_entry:
            v_after_by_entry.append(flat_v_after[cursor : cursor + len(exchanges)])
            cursor += len(exchanges)
    else:
        v_after_by_entry = [[None] * len(exchanges) for exchanges in exchanges_by_entry]

    scored: list[ScoredOrder] = []
    searched_finals: list[float] = []
    for entry, exchanges, v_afters in zip(
        searched, exchanges_by_entry, v_after_by_entry, strict=True
    ):
        values: list[float] = []
        response_values: list[tuple[OppResponse, float]] = []
        for response, exchange, v_after in zip(responses, exchanges, v_afters, strict=True):
            value = _exchange_value(exchange, config) + _value_head_delta(v_after, v_before, config)
            values.append(value)
            response_values.append((response, value))

        aggregated = _aggregate_exchange_values(values, responses, ctx, config)
        if response_values:
            worst_response, worst_value = min(response_values, key=lambda pair: pair[1])
        else:
            worst_response, worst_value = None, 0.0

        final_score = (
            config.search_myopic_weight * entry.score + config.search_position_weight * aggregated
        )
        breakdown = dict(entry.breakdown)
        breakdown["myopic_score"] = entry.score
        breakdown["exchange_value"] = aggregated
        breakdown["worst_response"] = (
            worst_response.describe() if worst_response is not None else None
        )
        breakdown["worst_response_value"] = worst_value
        breakdown["n_responses"] = len(responses)
        breakdown["searched"] = True
        scored.append(ScoredOrder(order=entry.order, score=final_score, breakdown=breakdown))
        searched_finals.append(final_score)

    # Always non-empty here: `searched` has at least one entry whenever `myopic` is
    # non-empty (cutoff = max(1, ...)), and we already returned early for empty myopic.
    min_searched_final = min(searched_finals)
    for tail_index, entry in enumerate(unsearched):
        breakdown = dict(entry.breakdown)
        breakdown["myopic_score"] = entry.score
        breakdown["exchange_value"] = None
        breakdown["worst_response"] = None
        breakdown["worst_response_value"] = None
        breakdown["n_responses"] = 0
        breakdown["searched"] = False
        tail_score = min_searched_final - 1.0 - tail_index
        scored.append(ScoredOrder(order=entry.order, score=tail_score, breakdown=breakdown))

    scored.sort(key=lambda scored_order: scored_order.score, reverse=True)
    _record_search_trace(scored, config)
    return scored


def _record_search_trace(scored: list[ScoredOrder], config: PolicyConfig) -> None:
    if not scored:
        return
    top_k = max(1, config.trace_top_k)
    record_note(
        "search_top_candidates",
        [
            {
                "order": describe_order(entry.order),
                "score": round(entry.score, 3),
                "myopic_score": round(float(entry.breakdown.get("myopic_score", 0.0)), 3),
                "exchange_value": (
                    round(float(entry.breakdown["exchange_value"]), 3)
                    if entry.breakdown.get("exchange_value") is not None
                    else None
                ),
                "worst_response": entry.breakdown.get("worst_response"),
            }
            for entry in scored[:top_k]
        ],
    )
