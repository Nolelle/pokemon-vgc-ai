"""Unit tests for `vgc.belief_scoring` -- mixture scoring over opponent-spread beliefs.

Hand-built battles (no live server), same SimpleNamespace conventions as
`tests/test_evaluator.py`, `tests/test_opponent_belief.py` (via `tests/test_battle_memory.py`
helpers), and `tests/test_live_mirror_beliefs.py`.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from poke_env.battle.move import Move
from poke_env.player.battle_order import SingleBattleOrder

import vgc.search as search_module
from vgc.actions import describe_order, enumerate_joint_orders
from vgc.battle_memory import BattleMemory, SpeedObservation
from vgc.belief_scoring import (
    belief_ordered_candidates,
    joint_spread_hypotheses,
    score_joint_orders_under_beliefs,
)
from vgc.damage import PokemonState
from vgc.decision_trace import finish_trace, start_trace
from vgc.evaluator import build_context, score_joint_orders, score_joint_orders_in_context
from vgc.models import PolicyConfig
from vgc.search import ExchangeResult
from vgc.sets import opponent_spread_hypotheses, opponent_state
from vgc.stats import calculate_stats

_TWO_SPREADS = [
    ({"atk": 32, "spe": 32}, "jolly", 0.75),
    ({"hp": 32, "def": 32}, "impish", 0.25),
]
_THREE_SPREADS = [
    ({"spe": 32}, "jolly", 0.5),
    ({"spe": 16}, "jolly", 0.3),
    ({"spe": 0}, "brave", 0.2),
]


def _opp_mon(
    species: str,
    *,
    moves: tuple[str, ...] = ("heatwave", "protect"),
    item: str | None = None,
    ability: str | None = None,
    fainted: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        moves={move: None for move in moves},
        item=item,
        ability=ability,
        boosts={},
        status=None,
        current_hp_fraction=1.0,
        current_hp=100,
        fainted=fainted,
        protect_counter=0,
    )


def _our_mon(
    species: str,
    *,
    evs: list[int] | None,
    nature: str | None,
    moves: tuple[str, ...] = ("earthquake", "protect"),
    item: str | None = None,
    ability: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        evs=evs,
        nature=nature,
        boosts={},
        status=None,
        item=item,
        ability=ability,
        current_hp=200,
        fainted=False,
        moves={move: None for move in moves},
        first_turn=False,
        protect_counter=0,
    )


def _battle(
    *,
    our_active: list,
    opp_active: list,
    valid_orders=None,
) -> SimpleNamespace:
    while len(our_active) < 2:
        our_active = [*our_active, None]
    while len(opp_active) < 2:
        opp_active = [*opp_active, None]
    live_opp = [mon for mon in opp_active if mon is not None]
    battle = SimpleNamespace(
        active_pokemon=our_active,
        opponent_active_pokemon=opp_active,
        team={
            f"p1: {mon.species}": mon for mon in our_active if mon is not None
        },
        opponent_team={
            f"p2: {mon.species}": mon for mon in live_opp
        },
        teampreview_opponent_team=list(live_opp),
        side_conditions=[],
        opponent_side_conditions=[],
        fields=[],
        weather=[],
        turn=1,
    )
    if valid_orders is not None:
        battle.valid_orders = valid_orders
    return battle


def _move_order(move_id: str, move_target: int = 0) -> SingleBattleOrder:
    return SingleBattleOrder(Move(move_id, gen=9), move_target=move_target)


def _our_garchomp() -> SimpleNamespace:
    # Adamant 32 Spe Garchomp is Spe 154 -- between corpus Charizard 144 and 167.
    return _our_mon(
        "garchomp",
        evs=[2, 32, 0, 0, 0, 32],
        nature="adamant",
        item="lifeorb",
        ability="roughskin",
        moves=("earthquake", "protect"),
    )


def _our_incineroar() -> SimpleNamespace:
    return _our_mon(
        "incineroar",
        evs=[32, 0, 14, 0, 20, 0],
        nature="careful",
        item="sitrusberry",
        ability="intimidate",
        moves=("protect", "fakeout"),
    )


def _straddle_battle() -> SimpleNamespace:
    """Our Garchomp at Spe 154 between Charizard corpus 144 and 167."""

    return _battle(
        our_active=[_our_garchomp(), _our_incineroar()],
        opp_active=[_opp_mon("charizard", moves=("heatwave", "protect"), ability="blaze")],
        valid_orders=(
            [_move_order("earthquake", 1), _move_order("protect")],
            [_move_order("protect")],
        ),
    )


_CROSSING_K = 15
_CROSSING_ORDER = "protect / protect"


def _crossing_battle() -> SimpleNamespace:
    """Straddle fixture with enough legal orders that the mixture crosses K=15.

    Corpus Charizard Spe 144/152/167 vs our Garchomp Spe 154. Mixture lifts
    `protect / protect` from myopic rank 15 into the K=15 shortlist (belief rank 14),
    displacing `swordsdance / knockoff@1`.
    """

    garchomp = _our_garchomp()
    garchomp.moves = {
        move: None
        for move in ("earthquake", "protect", "rockslide", "dragonclaw", "swordsdance")
    }
    incineroar = _our_incineroar()
    incineroar.moves = {
        move: None
        for move in ("protect", "fakeout", "flareblitz", "partingshot", "knockoff")
    }
    return _battle(
        our_active=[garchomp, incineroar],
        opp_active=[
            _opp_mon(
                "charizard",
                moves=("heatwave", "protect", "hurricane"),
                ability="blaze",
            )
        ],
        valid_orders=(
            [
                _move_order("earthquake", 1),
                _move_order("protect"),
                _move_order("rockslide"),
                _move_order("dragonclaw", 1),
                _move_order("swordsdance"),
            ],
            [
                _move_order("protect"),
                _move_order("fakeout", 1),
                _move_order("flareblitz", 1),
                _move_order("partingshot", 1),
                _move_order("knockoff", 1),
            ],
        ),
    )




def test_no_memory_limit_1_is_a_single_certain_hypothesis(monkeypatch) -> None:
    def fake(species_id, usage=None, limit=None):
        return _TWO_SPREADS[: limit or len(_TWO_SPREADS)]

    monkeypatch.setattr("vgc.belief_scoring.opponent_spread_hypotheses", fake)
    battle = _battle(
        our_active=[_our_mon("garchomp", evs=None, nature=None)],
        opp_active=[_opp_mon("garchomp"), _opp_mon("incineroar")],
    )
    hypotheses = joint_spread_hypotheses(battle, None, PolicyConfig(), limit=1)
    assert len(hypotheses) == 1
    assert hypotheses[0][0] == pytest.approx(1.0)
    assignment = hypotheses[0][1]
    assert assignment[0] == (_TWO_SPREADS[0][0], _TWO_SPREADS[0][1])
    assert assignment[1] == (_TWO_SPREADS[0][0], _TWO_SPREADS[0][1])


def test_no_memory_limit_3_two_actives_is_the_renormalised_cross_product(
    monkeypatch,
) -> None:
    def fake(species_id, usage=None, limit=None):
        return _THREE_SPREADS[: limit or len(_THREE_SPREADS)]

    monkeypatch.setattr("vgc.belief_scoring.opponent_spread_hypotheses", fake)
    battle = _battle(
        our_active=[_our_mon("garchomp", evs=None, nature=None)],
        opp_active=[_opp_mon("garchomp"), _opp_mon("incineroar")],
    )
    hypotheses = joint_spread_hypotheses(battle, None, PolicyConfig(), limit=3)
    assert len(hypotheses) == 9
    weights = [weight for weight, _assignment in hypotheses]
    assert weights == sorted(weights, reverse=True)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.5 * 0.5)


def test_no_memory_matches_the_corpus_prior_for_a_real_species() -> None:
    battle = _battle(
        our_active=[_our_mon("garchomp", evs=None, nature=None)],
        opp_active=[_opp_mon("charizard"), None],
    )
    hypotheses = joint_spread_hypotheses(battle, None, PolicyConfig(), limit=3)
    prior = opponent_spread_hypotheses("charizard", limit=3)
    assert len(hypotheses) == len(prior)
    assert sum(weight for weight, _assignment in hypotheses) == pytest.approx(1.0)
    for (weight, assignment), (spread, nature, probability) in zip(
        hypotheses, prior, strict=True
    ):
        assert weight == pytest.approx(probability)
        assert assignment[0][0] == spread
        assert assignment[0][1] == nature


def test_speed_observation_moves_the_top_joint_hypothesis_off_the_prior() -> None:
    battle = _battle(
        our_active=[_our_mon("garchomp", evs=None, nature=None)],
        opp_active=[_opp_mon("charizard"), None],
    )
    prior = joint_spread_hypotheses(battle, None, PolicyConfig(), limit=3)
    if len(prior) < 2:
        pytest.skip("charizard has a single corpus spread in this checkout")

    def speed_of(assignment):
        spread, nature = assignment[0]
        return calculate_stats("charizard", spread, nature)["spe"]

    speeds = sorted(speed_of(assignment) for _weight, assignment in prior)
    # Mode is the fastest corpus Charizard (timid 167). Observing they were no
    # faster than something between the slow and fast hypotheses contradicts it.
    cutoff = (speeds[0] + speeds[-1]) / 2.0
    memory = BattleMemory(battle_tag="test")
    memory.speed_observations.append(
        SpeedObservation(
            opponent_species="charizard", threshold=cutoff, relation="at_most"
        )
    )
    posterior = joint_spread_hypotheses(battle, memory, PolicyConfig(), limit=3)
    assert posterior[0][1] != prior[0][1]
    assert speed_of(posterior[0][1]) <= cutoff


# --- mixture scoring contract -------------------------------------------------------


def test_belief_mixture_differs_from_the_mode_when_spreads_straddle_our_speed() -> None:
    battle = _straddle_battle()
    our_speed = calculate_stats(
        "garchomp", {"hp": 2, "atk": 32, "spe": 32}, "adamant"
    )["spe"]
    charizard_prior = opponent_spread_hypotheses("charizard", limit=3)
    hyp_speeds = [
        calculate_stats("charizard", spread, nature)["spe"]
        for spread, nature, _probability in charizard_prior
    ]
    assert min(hyp_speeds) < our_speed < max(hyp_speeds)

    mode = score_joint_orders(battle)
    mixture = score_joint_orders_under_beliefs(
        battle, PolicyConfig(shortlist_belief_hypotheses=3)
    )
    assert mode
    assert mixture
    mode_by_desc = {describe_order(entry.order): entry.score for entry in mode}
    mix_by_desc = {describe_order(entry.order): entry.score for entry in mixture}
    mode_rank = [describe_order(entry.order) for entry in mode]
    mix_rank = [describe_order(entry.order) for entry in mixture]
    score_moved = any(
        abs(mode_by_desc[key] - mix_by_desc[key]) > 1e-6 for key in mode_by_desc
    )
    rank_moved = mode_rank != mix_rank
    assert score_moved or rank_moved


def test_limit_1_without_memory_matches_mode_when_top_posterior_is_opponent_state() -> None:
    """(c)(ii): mixture@1 equals the shipped scorer iff the top prior is the mode.

    `opponent_state` picks `max(usage, key=weight)` (`vgc.sets._usage_spread_for_species`).
    `opponent_spread_hypotheses(limit=1)` is the same entry for Charizard -- unique
    weights, timid 32 Spe -- so the scores must match.
    """
    battle = _straddle_battle()
    charizard = battle.opponent_active_pokemon[0]
    mode_state = opponent_state(charizard)
    top = opponent_spread_hypotheses("charizard", limit=1)[0]
    assert mode_state.sp_spread == top[0]
    assert mode_state.nature == top[1]

    mode = score_joint_orders(battle)
    mixture = score_joint_orders_under_beliefs(
        battle, PolicyConfig(shortlist_belief_hypotheses=1)
    )
    assert len(mode) == len(mixture)
    mode_by_desc = {describe_order(entry.order): entry.score for entry in mode}
    mix_by_desc = {describe_order(entry.order): entry.score for entry in mixture}
    assert mode_by_desc.keys() == mix_by_desc.keys()
    for key, score in mode_by_desc.items():
        assert mix_by_desc[key] == pytest.approx(score)


def test_build_context_override_changes_only_the_overridden_slot_speed() -> None:
    battle = _battle(
        our_active=[_our_garchomp(), _our_incineroar()],
        opp_active=[
            _opp_mon("charizard", moves=("heatwave",), ability="blaze"),
            _opp_mon("incineroar", moves=("fakeout",), ability="intimidate"),
        ],
    )
    config = PolicyConfig()
    baseline = build_context(battle, config)
    slow = PokemonState(
        "charizard",
        sp_spread={"hp": 10, "spa": 32, "spe": 24},
        nature="modest",
        ability="blaze",
    )
    overridden = build_context(battle, config, opp_state_override={0: slow})
    assert overridden.opp_speed[0] != pytest.approx(baseline.opp_speed[0])
    assert overridden.opp_speed[0] < baseline.opp_speed[0]
    assert overridden.opp_speed[1] == pytest.approx(baseline.opp_speed[1])
    assert overridden.our_speed == baseline.our_speed


def test_score_joint_orders_matches_in_context_helper() -> None:
    battle = _straddle_battle()
    config = PolicyConfig()
    via_entry = score_joint_orders(battle, config)
    via_helper = score_joint_orders_in_context(
        enumerate_joint_orders(battle), build_context(battle, config), config
    )
    assert [describe_order(entry.order) for entry in via_entry] == [
        describe_order(entry.order) for entry in via_helper
    ]
    for left, right in zip(via_entry, via_helper, strict=True):
        assert left.score == pytest.approx(right.score)
        assert left.breakdown == right.breakdown


def test_empty_joint_orders_yield_empty_mixture(monkeypatch) -> None:
    monkeypatch.setattr("vgc.belief_scoring.enumerate_joint_orders", lambda _battle: [])
    battle = _straddle_battle()
    assert score_joint_orders_under_beliefs(battle) == []


def test_shortlist_belief_hypotheses_below_one_is_rejected() -> None:
    battle = _straddle_battle()
    with pytest.raises(ValueError, match="at least 1"):
        score_joint_orders_under_beliefs(
            battle, PolicyConfig(shortlist_belief_hypotheses=0)
        )


# --- Rung 2b: belief-ordered shortlist ------------------------------------------------


def test_belief_ordered_candidates_hypotheses_1_returns_the_same_list() -> None:
    battle = _straddle_battle()
    myopic = score_joint_orders(battle)
    ranked = belief_ordered_candidates(
        battle, myopic, PolicyConfig(shortlist_belief_hypotheses=1)
    )
    assert ranked is myopic
    assert all("belief_mixture" not in entry.breakdown for entry in ranked)


def test_belief_mixture_crosses_the_shortlist_boundary() -> None:
    battle = _crossing_battle()
    myopic = score_joint_orders(battle)
    scores_before = [entry.score for entry in myopic]
    ranked = belief_ordered_candidates(
        battle, myopic, PolicyConfig(shortlist_belief_hypotheses=3)
    )
    assert len(ranked) == len(myopic)
    assert {id(entry) for entry in ranked} == {id(entry) for entry in myopic}
    for entry, original in zip(myopic, scores_before, strict=True):
        assert entry.score == original
    assert [id(entry) for entry in ranked] != [id(entry) for entry in myopic]

    k = _CROSSING_K
    myopic_descs = [describe_order(entry.order) for entry in myopic]
    ranked_descs = [describe_order(entry.order) for entry in ranked]
    assert _CROSSING_ORDER not in myopic_descs[:k]
    assert _CROSSING_ORDER in ranked_descs[:k]
    assert myopic_descs.index(_CROSSING_ORDER) == k
    assert ranked_descs.index(_CROSSING_ORDER) == k - 1
    for entry in ranked:
        mixture = entry.breakdown["belief_mixture"]
        assert mixture["hypotheses"] == 3
        assert "mixture_score" in mixture
        assert mixture["myopic_rank"] == myopic_descs.index(describe_order(entry.order))


def test_belief_ordered_candidates_does_not_rescore_the_mode_hypothesis(
    monkeypatch,
) -> None:
    battle = _straddle_battle()
    myopic = score_joint_orders(battle)
    calls = {"n": 0}
    real = score_joint_orders_in_context

    def wrapped(orders, ctx, config):
        calls["n"] += 1
        return real(orders, ctx, config)

    monkeypatch.setattr("vgc.belief_scoring.score_joint_orders_in_context", wrapped)
    belief_ordered_candidates(
        battle, myopic, PolicyConfig(shortlist_belief_hypotheses=3)
    )
    # Three Charizard spreads; the timid mode matches opponent_state and is skipped.
    assert calls["n"] == 2


def test_search_shortlist_follows_belief_rank_and_keeps_opponent_responses(
    monkeypatch,
) -> None:
    battle = _crossing_battle()
    seen_responses: list[list[str]] = []
    original_enumerate = search_module._enumerate_opp_responses

    def spy_responses(ctx, config):
        result = original_enumerate(ctx, config)
        seen_responses.append([response.describe() for response in result])
        return result

    monkeypatch.setattr(search_module, "_enumerate_opp_responses", spy_responses)
    monkeypatch.setattr(
        search_module, "resolve_exchange", lambda *_args, **_kwargs: ExchangeResult()
    )

    config_one = PolicyConfig(
        shortlist_belief_hypotheses=1,
        search_our_candidates=_CROSSING_K,
        search_diverse_candidates=False,
        use_rolling_horizon=False,
        search_opp_candidates=4,
    )
    config_three = PolicyConfig(
        shortlist_belief_hypotheses=3,
        search_our_candidates=_CROSSING_K,
        search_diverse_candidates=False,
        use_rolling_horizon=False,
        search_opp_candidates=4,
    )
    result_one = search_module.search_joint_orders(battle, config_one)
    result_three = search_module.search_joint_orders(battle, config_three)

    searched_one = {
        describe_order(entry.order)
        for entry in result_one
        if entry.breakdown.get("searched")
    }
    searched_three = {
        describe_order(entry.order)
        for entry in result_three
        if entry.breakdown.get("searched")
    }
    assert searched_one != searched_three
    assert _CROSSING_ORDER not in searched_one
    assert _CROSSING_ORDER in searched_three
    assert len(seen_responses) == 2
    assert seen_responses[0] == seen_responses[1]


def test_belief_shortlist_trace_note_appears_once_per_decision(monkeypatch) -> None:
    monkeypatch.setenv("VGC_TRACE", "1")
    battle = _crossing_battle()
    myopic = score_joint_orders(battle)
    token = start_trace()
    belief_ordered_candidates(
        battle, myopic, PolicyConfig(shortlist_belief_hypotheses=3)
    )
    trace = finish_trace(token)
    assert trace is not None
    note = trace.notes["belief_shortlist"]
    assert note["hypotheses"] == 3
    assert note["reordered"] >= 1
    assert note["top_changed"] is False

    token = start_trace()
    monkeypatch.setattr(
        search_module, "resolve_exchange", lambda *_args, **_kwargs: ExchangeResult()
    )
    search_module.search_joint_orders(
        battle,
        PolicyConfig(
            shortlist_belief_hypotheses=3,
            search_our_candidates=_CROSSING_K,
            search_diverse_candidates=False,
            use_rolling_horizon=False,
            search_opp_candidates=2,
        ),
    )
    search_trace = finish_trace(token)
    assert search_trace is not None
    assert "belief_shortlist" in search_trace.notes
    assert search_trace.notes["belief_shortlist"]["hypotheses"] == 3
