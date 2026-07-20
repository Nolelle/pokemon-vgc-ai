"""Team preview: pick 4-of-6 (+ lead order) against the opponent's previewed 6.

`build_team_order(battle, config)` is `vgc.agent.VgcPlayer.decide_teampreview()`'s real
implementation (see vgc/agent.py) -- it scores every `C(6,4)` pick times `C(4,2)` lead
pair (90 candidates total) and returns poke-env's expected `"/team XXXX"` order string
(first two digits are the leads, matching `Player.random_teampreview`'s own format --
see that method's docstring in poke-env for the wire format).

Every candidate is scored on three PolicyConfig-weighted terms (see vgc/models.py for
the exact weights):

  - **Damage exchange**: for each of the 4 picked mons against each of the opponent's
    previewed 6, our best expected-damage % onto them minus their best expected-damage %
    onto us (both via `vgc.sets.opponent_state`'s usage-spread-informed guess for the
    opponent, since real Stat Points aren't known pre-battle either), averaged over all
    24 pairs. This is a net-advantage PROXY for a true ratio (`our% / their%`) -- a real
    ratio blows up/is undefined whenever `their%` is near 0, which happens constantly
    (immunities, resists), so a bounded difference is used instead; it captures the same
    "who wins this exchange" signal without that failure mode.
  - **Speed**: mean effective Speed of the two LEADS (the two mons that actually matter
    for turn 1) minus the opponent's previewed 6's mean Speed. Lead-set weather includes
    Chlorophyll/Swift Swim/Sand Rush/Slush Rush multipliers.
  - **Trick Room coherence**: 0 unless the pick includes a mon with Trick Room in its
    kit, in which case it rewards ALSO bringing genuinely slow attackers alongside the
    setter (mean opponent Speed minus the mean Speed of the picked non-setter mons) --
    a TR pick with fast attackers alongside the setter doesn't get this bonus, since that
    isn't really "TR mode" so much as "brought Trick Room as tech".

The `our(6) x opponent(<=6)` matchup matrix is computed once per possible lead weather,
then reused across all 90 candidates -- see `_build_matchup_matrix`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import mean

from poke_env.battle.abstract_battle import AbstractBattle

from vgc.damage import FieldState, to_id
from vgc.decision_trace import record_note
from vgc.evaluator import (
    _best_attacking_move,
    _our_pokemon_state,
    effective_speed,
    mega_evolved_state,
    mega_species_id,
)
from vgc.gameplan import build_gameplan
from vgc.meta import known_nature, recognize_meta_team
from vgc.models import PolicyConfig
from vgc.principles import TeamSignals, detect_team_signals
from vgc.sets import load_usage_spreads, opponent_state

_LEADS_COUNT = 2
_PICK_COUNT = 4
_TEAM_SIZE = 6

_WEATHER_BY_ABILITY = {
    "drought": "sun",
    "drizzle": "rain",
    "sandstream": "sand",
    "snowwarning": "snow",
}
_SPEED_ABILITY_BY_WEATHER = {
    "sun": "chlorophyll",
    "rain": "swiftswim",
    "sand": "sandrush",
    "snow": "slushrush",
}
_PREVIEW_WEATHERS = (None, "sun", "rain", "sand", "snow")


@dataclass(frozen=True)
class PreviewPlan:
    """Strategic commitments made before turn one and carried into battle scoring."""

    our_closer_species: str | None
    opponent_closer_species: str | None
    default_mega_species: str | None
    opponent_engines: tuple[str, ...]
    picked_species: tuple[str, ...]
    lead_species: tuple[str, ...]
    lead_functions: tuple[str, ...]
    speed_modes: tuple[str, ...]
    balanced_structure: dict[str, bool]
    lead_covers_engine: bool
    back_has_second_speed_mode: bool
    # Iteration 6 (vgc.preview_predict.predict_preview_hybrid): the opponent's
    # single-highest-probability predicted lead pair, species ids -- `None` when
    # `PolicyConfig.use_preview_prediction` is off or prediction failed (e.g. an
    # opponent preview with fewer than 6 species). Seeds turn-1 context: available to
    # any downstream reader via `_Context.preview_plan` the same way
    # `our_closer_species`/`default_mega_species` already are, and recorded for tracing
    # regardless (see `.summary()`) so predicted-vs-actual accuracy is inspectable from
    # a real session's traces.
    predicted_opponent_leads: tuple[str, ...] | None = None

    def summary(self) -> dict[str, object]:
        return {
            "our_closer": self.our_closer_species,
            "opponent_closer": self.opponent_closer_species,
            "default_mega": self.default_mega_species,
            "opponent_engines": list(self.opponent_engines),
            "picked_species": list(self.picked_species),
            "lead_species": list(self.lead_species),
            "lead_functions": list(self.lead_functions),
            "speed_modes": list(self.speed_modes),
            "balanced_structure": dict(self.balanced_structure),
            "lead_covers_engine": self.lead_covers_engine,
            "back_has_second_speed_mode": self.back_has_second_speed_mode,
            "predicted_opponent_leads": (
                list(self.predicted_opponent_leads)
                if self.predicted_opponent_leads is not None
                else None
            ),
        }


def build_team_order(battle: AbstractBattle, config: PolicyConfig | None = None) -> str:
    config = config or PolicyConfig()
    our_team = list(battle.team.values())
    opp_team = list(battle.teampreview_opponent_team or [])

    if len(our_team) < _PICK_COUNT:
        # Defensive: a legal VGC battle always has 6, but never crash the exception-safe
        # wrapper's caller over a malformed/short team.
        order = list(range(1, len(our_team) + 1))
        return "/team " + "".join(str(i) for i in order)
    if not opp_team:
        order = list(range(1, _PICK_COUNT + 1))
        return "/team " + "".join(str(i) for i in order)

    usage = load_usage_spreads()
    meta_team = recognize_meta_team(opp_team)
    # A held Mega stone is public information. Preview the matchup value of the potential
    # Mega form (including weather abilities); the turn evaluator separately decides
    # whether evolving now materially changes damage, Speed, weather, or survival.
    our_base_states = [_our_pokemon_state(mon) for mon in our_team]
    our_states = [mega_evolved_state(state) for state in our_base_states]
    opp_states = [
        mega_evolved_state(
            opponent_state(mon, usage=usage, nature_override=known_nature(meta_team, mon))
        )
        for mon in opp_team
    ]
    our_move_id_lists = [
        [to_id(move_id) for move_id in mon.moves.keys()] if mon.moves else [] for mon in our_team
    ]
    opp_move_id_lists = [
        [to_id(move_id) for move_id in mon.moves.keys()] if mon.moves else [] for mon in opp_team
    ]
    our_signals = detect_team_signals(our_team)
    opp_signals = detect_team_signals(opp_team)
    gameplan = build_gameplan(our_states, opp_states, our_move_id_lists, opp_move_id_lists, config)
    closer_idx = gameplan.primary_win_con_idx
    opponent_closer_idx = gameplan.primary_threat_idx
    mega_indices = frozenset(
        idx
        for idx, state in enumerate(our_base_states)
        if mega_species_id(state.species_id, state.item) is not None
    )
    default_mega_idx = None
    if mega_indices:
        default_mega_idx = max(mega_indices, key=lambda idx: gameplan.win_con_scores[idx])

    opp_speed = [effective_speed(state) for state in opp_states]
    opp_avg_speed = mean(opp_speed) if opp_speed else 0.0

    matchup_by_weather = {
        weather: _build_matchup_matrix(
            our_states,
            our_move_id_lists,
            opp_states,
            opp_move_id_lists,
            weather=weather,
        )
        for weather in _PREVIEW_WEATHERS
    }

    # Iteration 6 (docs/preview_prediction_plan.md): predict the OPPONENT's own bring-4
    # + leads (vgc.preview_predict.predict_preview_hybrid -- usage-driven bring-4,
    # matchup-conditioned leads, backtest-gated -- see PolicyConfig.
    # use_preview_prediction's comment) and score our candidates against THAT
    # distribution instead of assuming all 6 previewed opponent mons are equally likely
    # to be brought. Deferred import: vgc.preview_predict imports FROM this module
    # (_build_matchup_matrix/_score_choice/etc, to reuse these internals without
    # duplicating them), so importing it back at module level here would be circular --
    # this local import only ever runs at call time, by which point both modules are
    # fully loaded.
    opp_bring_weights = None
    opp_worst_case_subset = None
    predicted_opponent_leads: tuple[str, ...] | None = None
    predicted_preview_summary: dict[str, object] | None = None
    if config.use_preview_prediction:
        from vgc.preview_predict import (
            bring4_distribution,
            lead_distribution,
            predict_preview_hybrid,
        )

        opp_species_ids = [to_id(mon.species) for mon in opp_team]
        our_species_ids = [to_id(mon.species) for mon in our_team]
        hybrid_candidates = predict_preview_hybrid(opp_species_ids, our_species_ids, config)
        if hybrid_candidates:
            bring4_ranked = bring4_distribution(hybrid_candidates)
            opp_bring_weights = [0.0] * len(opp_team)
            for subset, prob in bring4_ranked:
                for idx in subset:
                    opp_bring_weights[idx] += prob
            opp_worst_case_subset = bring4_ranked[0][0]
            lead_ranked = lead_distribution(hybrid_candidates)
            if lead_ranked:
                predicted_opponent_leads = tuple(
                    to_id(opp_team[idx].species) for idx in lead_ranked[0][0]
                )
            predicted_preview_summary = {
                "predicted_bring4": [
                    [to_id(opp_team[i].species) for i in subset] for subset, _p in bring4_ranked[:3]
                ],
                "predicted_bring4_probs": [round(p, 4) for _s, p in bring4_ranked[:3]],
                "predicted_leads": [
                    [to_id(opp_team[i].species) for i in pair] for pair, _p in lead_ranked[:3]
                ],
                "predicted_leads_probs": [round(p, 4) for _pair, p in lead_ranked[:3]],
            }

    best_score = float("-inf")
    best_order: tuple[int, ...] = tuple(range(_PICK_COUNT))
    best_breakdown: dict[str, object] = {}
    for pick in combinations(range(len(our_team)), _PICK_COUNT):
        for leads in combinations(pick, _LEADS_COUNT):
            bench = tuple(i for i in pick if i not in leads)
            order = leads + bench
            score, breakdown = _score_choice(
                order=order,
                picked=pick,
                our_states=our_states,
                our_move_id_lists=our_move_id_lists,
                matchup_by_weather=matchup_by_weather,
                opp_avg_speed=opp_avg_speed,
                n_opp=len(opp_states),
                our_signals=our_signals,
                opp_signals=opp_signals,
                closer_idx=closer_idx,
                opponent_closer_idx=opponent_closer_idx,
                mega_indices=mega_indices,
                default_mega_idx=default_mega_idx,
                config=config,
                opp_bring_weights=opp_bring_weights,
                opp_worst_case_subset=opp_worst_case_subset,
            )
            if score > best_score:
                best_score = score
                best_order = order
                best_breakdown = breakdown

    order_string = "/team " + "".join(str(i + 1) for i in best_order)
    lead_functions = tuple(best_breakdown.get("lead_functions", ()))
    preview_plan = PreviewPlan(
        our_closer_species=(
            to_id(our_team[closer_idx].species) if closer_idx is not None else None
        ),
        opponent_closer_species=(
            to_id(opp_team[opponent_closer_idx].species)
            if opponent_closer_idx is not None
            else None
        ),
        default_mega_species=(
            to_id(our_team[default_mega_idx].species) if default_mega_idx is not None else None
        ),
        opponent_engines=tuple(sorted(opp_signals.engines)),
        picked_species=tuple(to_id(our_team[idx].species) for idx in best_order),
        lead_species=tuple(to_id(our_team[idx].species) for idx in best_order[:2]),
        lead_functions=lead_functions,
        speed_modes=tuple(best_breakdown.get("speed_modes", ())),
        balanced_structure=dict(best_breakdown.get("balanced_structure", {})),
        lead_covers_engine=bool(best_breakdown.get("lead_covers_engine", False)),
        back_has_second_speed_mode=bool(best_breakdown.get("back_has_second_speed_mode", False)),
        predicted_opponent_leads=predicted_opponent_leads,
    )
    # poke-env battle objects are mutable throughout a battle. Carry the preview choice
    # forward so turn scoring preserves the closer and default Mega we actually selected
    # instead of silently inventing a new plan every turn.
    try:
        setattr(battle, "_vgc_preview_plan", preview_plan)
    except (AttributeError, TypeError):
        pass
    for idx in best_order:
        # Mirrors poke-env's own Player.random_teampreview contract (see its docstring):
        # mark picked Pokemon so anything downstream reading
        # Pokemon.selected_in_teampreview sees a consistent picture.
        our_team[idx]._selected_in_teampreview = True
    record_note(
        "team_preview_choice",
        {
            "order": order_string,
            "score": round(best_score, 3),
            **best_breakdown,
            "plan": preview_plan.summary(),
        },
    )
    if predicted_preview_summary is not None:
        record_note("predicted_opponent_preview", predicted_preview_summary)
    if meta_team is not None:
        record_note(
            "opponent_meta_team",
            {
                "id": meta_team["id"],
                "name": meta_team["name"],
                "common_leads": meta_team["common_leads"],
            },
        )
    return order_string


def _build_matchup_matrix(
    our_states, our_move_id_lists, opp_states, opp_move_id_lists, *, weather=None
):
    field = FieldState(is_doubles=True, weather=weather)
    n_our = len(our_states)
    n_opp = len(opp_states)
    our_onto_them = [[0.0] * n_opp for _ in range(n_our)]
    them_onto_us = [[0.0] * n_opp for _ in range(n_our)]
    for i in range(n_our):
        for j in range(n_opp):
            our_onto_them[i][j], _, _ = _best_attacking_move(
                our_states[i], our_move_id_lists[i], opp_states[j], field
            )
            them_onto_us[i][j], _, _ = _best_attacking_move(
                opp_states[j], opp_move_id_lists[j], our_states[i], field
            )
    return our_onto_them, them_onto_us


def _lead_weather(leads, our_states) -> str | None:
    for index in leads:
        weather = _WEATHER_BY_ABILITY.get(our_states[index].ability)
        if weather is not None:
            return weather
    return None


def _preview_speed(state, weather: str | None) -> float:
    speed = effective_speed(state)
    if _SPEED_ABILITY_BY_WEATHER.get(weather) == state.ability:
        speed *= 2.0
    return speed


def _score_choice(
    *,
    order,
    picked,
    our_states,
    our_move_id_lists,
    matchup_by_weather,
    opp_avg_speed,
    n_opp,
    our_signals: TeamSignals,
    opp_signals: TeamSignals,
    closer_idx,
    opponent_closer_idx,
    mega_indices,
    default_mega_idx,
    config,
    opp_bring_weights=None,
    opp_worst_case_subset=None,
) -> tuple[float, dict[str, object]]:
    """`opp_bring_weights`/`opp_worst_case_subset` (both `None` by default, giving the
    EXACT pre-iteration-6 flat-average-over-all-6 behavior): when
    `PolicyConfig.use_preview_prediction` is on, `build_team_order` passes
    `opp_bring_weights` (per-opponent-mon marginal predicted bring probability, from
    `vgc.preview_predict.predict_preview_hybrid`) and `opp_worst_case_subset` (the
    single highest-probability predicted opponent bring-4) so `exchange_score` reflects
    "how do we do against what they'll ACTUALLY bring" instead of "how do we do against
    all 6 previewed mons equally" -- see `PolicyConfig.team_preview_opponent_worst_case_
    weight`'s comment for the blend.
    """
    leads = order[:_LEADS_COUNT]
    weather = _lead_weather(leads, our_states)
    our_onto_them, them_onto_us = matchup_by_weather[weather]

    if opp_bring_weights is None:
        exchange_terms = [
            our_onto_them[i][j] - them_onto_us[i][j] for i in picked for j in range(n_opp)
        ]
        exchange_score = mean(exchange_terms) if exchange_terms else 0.0
    else:
        weighted_sum = 0.0
        total_weight = 0.0
        for i in picked:
            for j in range(n_opp):
                weight = opp_bring_weights[j]
                weighted_sum += weight * (our_onto_them[i][j] - them_onto_us[i][j])
                total_weight += weight
        expectation_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        if opp_worst_case_subset:
            worst_terms = [
                our_onto_them[i][j] - them_onto_us[i][j]
                for i in picked
                for j in opp_worst_case_subset
            ]
            worst_case_score = mean(worst_terms) if worst_terms else expectation_score
            hedge = config.team_preview_opponent_worst_case_weight
            exchange_score = (1.0 - hedge) * expectation_score + hedge * worst_case_score
        else:
            exchange_score = expectation_score

    lead_speed = mean(_preview_speed(our_states[i], weather) for i in leads)
    speed_score = lead_speed - opp_avg_speed

    has_tr_setter = any("trickroom" in our_move_id_lists[i] for i in picked)
    tr_score = 0.0
    if has_tr_setter:
        non_setters = [i for i in picked if "trickroom" not in our_move_id_lists[i]]
        if non_setters:
            attacker_speed = mean(_preview_speed(our_states[i], weather) for i in non_setters)
            tr_score = opp_avg_speed - attacker_speed

    lead_functions: set[str] = set()
    lead_roles = [our_signals.roles_by_index[i] for i in leads]
    if any(role.attacker for role in lead_roles):
        lead_functions.add("immediate_pressure")
    if any(role.speed_control for role in lead_roles):
        lead_functions.add("speed_control")
    if any(role.action_denial for role in lead_roles):
        lead_functions.add("deny_setup")
    if any(role.redirection or role.protection for role in lead_roles):
        lead_functions.add("protect_partner")
    if any(
        max((our_onto_them[i][j] - them_onto_us[i][j] for j in range(n_opp)), default=0.0) >= 40.0
        for i in leads
    ):
        lead_functions.add("force_switch")
    if any(role.protection for role in lead_roles) and any(role.attacker for role in lead_roles):
        lead_functions.add("safe_information")
    if any(our_onto_them[i][j] >= 100.0 for i in leads for j in range(n_opp)):
        lead_functions.add("immediate_ko")

    lead_function_score = min(4, len(lead_functions)) * config.team_preview_lead_function_bonus
    passive_penalty = config.team_preview_passive_lead_penalty if len(lead_functions) < 2 else 0.0

    # Their engine enablers and closer must have at least one favorable answer among our
    # selected four. Each covered strategic target earns one explicit bonus.
    strategic_targets = set(opp_signals.engine_enabler_indices)
    if opponent_closer_idx is not None:
        strategic_targets.add(opponent_closer_idx)
    covered_targets = 0
    for opp_idx in strategic_targets:
        if any(our_onto_them[i][opp_idx] > them_onto_us[i][opp_idx] for i in picked):
            covered_targets += 1
    engine_answer_score = covered_targets * config.team_preview_engine_answer_bonus
    lead_covers_engine = any(
        our_onto_them[lead_idx][opp_idx] > them_onto_us[lead_idx][opp_idx]
        for lead_idx in leads
        for opp_idx in strategic_targets
    ) or any(role.action_denial for role in lead_roles)
    lead_engine_denial_score = (
        config.team_preview_lead_engine_denial_bonus if lead_covers_engine else 0.0
    )

    closer_score = 0.0
    if closer_idx is not None and closer_idx in picked:
        closer_score += config.team_preview_closer_pick_bonus
        if closer_idx in order[2:]:
            closer_score += config.team_preview_closer_back_bonus

    # A useful backline can enter into attacks aimed at either lead. Use its worst
    # previewed incoming hit as the safety measure, then keep the safer of the two backs
    # for each lead. This directly rewards position repair rather than four would-be leads.
    bench = order[2:]
    bench_safety_terms = [
        max(
            (
                100.0 - max((them_onto_us[bench_idx][j] for j in range(n_opp)), default=0.0)
                for bench_idx in bench
            ),
            default=0.0,
        )
        for _lead_idx in leads
    ]
    backline_safety = mean(bench_safety_terms) if bench_safety_terms else 0.0

    picked_roles = [our_signals.roles_by_index[i] for i in picked]
    covered_roles = {
        role_name
        for role_name in ("attacker", "speed_control", "action_denial", "pivot", "protection")
        if any(getattr(role, role_name) for role in picked_roles)
    }
    role_coverage_score = len(covered_roles) * config.team_preview_role_coverage_bonus

    speed_modes: set[str] = set()
    if weather is not None:
        speed_modes.add(f"weather:{weather}")
    if any("tailwind" in our_move_id_lists[i] for i in picked):
        speed_modes.add("tailwind")
    if any("trickroom" in our_move_id_lists[i] for i in picked):
        speed_modes.add("trick_room")
    if any(our_signals.roles_by_index[i].priority_pressure for i in picked):
        speed_modes.add("priority")
    back_has_second_speed_mode = bool(
        len(speed_modes) >= 2
        and any(
            our_signals.roles_by_index[i].speed_control
            or our_signals.roles_by_index[i].priority_pressure
            for i in bench
        )
    )
    second_speed_mode_score = (
        config.team_preview_second_speed_mode_bonus if back_has_second_speed_mode else 0.0
    )

    balanced_checks = {
        "speed_control": any(role.speed_control for role in picked_roles),
        "redirection_or_fake_out": any(
            role.redirection or role.action_denial for role in picked_roles
        ),
        "two_attackers": sum(role.attacker for role in picked_roles) >= 2,
        "defensive_pivot": any(role.pivot for role in picked_roles),
        "closer": closer_idx in picked if closer_idx is not None else False,
    }
    balanced_structure_score = (
        sum(balanced_checks.values()) * config.team_preview_balanced_structure_bonus
    )

    mega_score = 0.0
    picked_megas = mega_indices.intersection(picked)
    if default_mega_idx is not None and default_mega_idx in picked:
        mega_score += config.team_preview_default_mega_bonus
    if len(picked_megas) > 1:
        mega_score -= (len(picked_megas) - 1) * config.team_preview_extra_mega_penalty

    total = (
        exchange_score * config.team_preview_damage_weight
        + speed_score * config.team_preview_speed_weight
        + tr_score * config.team_preview_tr_coherence_weight
        + lead_function_score
        - passive_penalty
        + engine_answer_score
        + lead_engine_denial_score
        + closer_score
        + backline_safety * config.team_preview_backline_safety_weight
        + role_coverage_score
        + second_speed_mode_score
        + balanced_structure_score
        + mega_score
    )
    return total, {
        "exchange_score": round(exchange_score, 3),
        "speed_score": round(speed_score, 3),
        "tr_score": round(tr_score, 3),
        "has_tr_setter": has_tr_setter,
        "lead_weather": weather,
        "opponent_engines": sorted(opp_signals.engines),
        "lead_functions": sorted(lead_functions),
        "passive_lead_penalty": round(passive_penalty, 3),
        "engine_targets_covered": covered_targets,
        "lead_covers_engine": lead_covers_engine,
        "closer_picked": closer_idx in picked if closer_idx is not None else False,
        "closer_in_back": closer_idx in bench if closer_idx is not None else False,
        "backline_safety": round(backline_safety, 3),
        "role_coverage": sorted(covered_roles),
        "speed_modes": sorted(speed_modes),
        "back_has_second_speed_mode": back_has_second_speed_mode,
        "balanced_structure": balanced_checks,
        "default_mega_picked": default_mega_idx in picked
        if default_mega_idx is not None
        else False,
    }
