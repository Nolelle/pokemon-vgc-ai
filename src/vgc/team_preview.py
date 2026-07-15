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
)
from vgc.meta import known_nature, recognize_meta_team
from vgc.models import PolicyConfig
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
    # A held Mega stone is public information and this evaluator has no strategic reason
    # to save a Mega for later, so preview the form that will actually battle. This also
    # exposes weather-setting Mega abilities such as Drought to the lead scorer.
    our_states = [mega_evolved_state(_our_pokemon_state(mon)) for mon in our_team]
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
                config=config,
            )
            if score > best_score:
                best_score = score
                best_order = order
                best_breakdown = breakdown

    order_string = "/team " + "".join(str(i + 1) for i in best_order)
    for idx in best_order:
        # Mirrors poke-env's own Player.random_teampreview contract (see its docstring):
        # mark picked Pokemon so anything downstream reading
        # Pokemon.selected_in_teampreview sees a consistent picture.
        our_team[idx]._selected_in_teampreview = True
    record_note(
        "team_preview_choice", {"order": order_string, "score": round(best_score, 3), **best_breakdown}
    )
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
    config,
) -> tuple[float, dict[str, object]]:
    leads = order[:_LEADS_COUNT]
    weather = _lead_weather(leads, our_states)
    our_onto_them, them_onto_us = matchup_by_weather[weather]

    exchange_terms = [
        our_onto_them[i][j] - them_onto_us[i][j] for i in picked for j in range(n_opp)
    ]
    exchange_score = mean(exchange_terms) if exchange_terms else 0.0

    lead_speed = mean(_preview_speed(our_states[i], weather) for i in leads)
    speed_score = lead_speed - opp_avg_speed

    has_tr_setter = any("trickroom" in our_move_id_lists[i] for i in picked)
    tr_score = 0.0
    if has_tr_setter:
        non_setters = [i for i in picked if "trickroom" not in our_move_id_lists[i]]
        if non_setters:
            attacker_speed = mean(_preview_speed(our_states[i], weather) for i in non_setters)
            tr_score = opp_avg_speed - attacker_speed

    total = (
        exchange_score * config.team_preview_damage_weight
        + speed_score * config.team_preview_speed_weight
        + tr_score * config.team_preview_tr_coherence_weight
    )
    return total, {
        "exchange_score": round(exchange_score, 3),
        "speed_score": round(speed_score, 3),
        "tr_score": round(tr_score, 3),
        "has_tr_setter": has_tr_setter,
        "lead_weather": weather,
    }
