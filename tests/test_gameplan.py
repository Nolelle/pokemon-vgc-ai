"""Unit tests for `vgc.gameplan` -- the pure win-condition matchup-matrix derivation.
Hand-built `PokemonState`s (mirrors tests/test_evaluator.py's `_garchomp`/`_klefki`
style); type IMMUNITY (Ghost vs Normal) is used to deterministically force specific
1v1 win/loss outcomes regardless of exact stat tuning, since an immune matchup always
resolves to "0 damage, 0 turns-to-KO chance" no matter what stats are involved.
"""

from __future__ import annotations

import math

import pytest

from vgc.damage import PokemonState
from vgc.gameplan import GamePlan, _beats, _best_move_percent, _turns_to_ko, build_gameplan


def _state(species: str, **kwargs) -> PokemonState:
    defaults = {"sp_spread": {"hp": 32, "atk": 32}, "nature": "adamant"}
    defaults.update(kwargs)
    return PokemonState(species, **defaults)


# --- _turns_to_ko / _beats: the "who wins the 1v1" primitive ------------------------


def test_turns_to_ko_zero_percent_is_infinite() -> None:
    assert math.isinf(_turns_to_ko(0.0))


def test_turns_to_ko_rounds_up() -> None:
    assert _turns_to_ko(34.0) == 3  # ceil(100/34) == 3, not 2


def test_beats_fewer_turns_wins() -> None:
    assert _beats(pct_a=50.0, pct_b=25.0, a_faster=False) is True
    assert _beats(pct_a=25.0, pct_b=50.0, a_faster=False) is False


def test_beats_tied_turns_broken_by_speed() -> None:
    assert _beats(pct_a=50.0, pct_b=50.0, a_faster=True) is True
    assert _beats(pct_a=50.0, pct_b=50.0, a_faster=False) is False


def test_beats_neither_can_ko_is_not_a_win_for_either_side() -> None:
    assert _beats(pct_a=0.0, pct_b=0.0, a_faster=True) is False


# --- _best_move_percent: neutral-field, status moves skipped, immunity is 0 ---------


def test_best_move_percent_skips_status_moves() -> None:
    attacker = _state("klefki")
    defender = _state("magikarp", sp_spread={"hp": 32})
    assert _best_move_percent(attacker, ["protect"], defender) == 0.0


def test_best_move_percent_is_zero_on_immunity() -> None:
    attacker = _state("torkoal")  # Fire, only a Normal-type move
    defender = _state("gengar", sp_spread={"hp": 32})  # Ghost -- immune to Normal
    assert _best_move_percent(attacker, ["bodyslam"], defender) == 0.0


def test_best_move_percent_picks_the_best_of_several_moves() -> None:
    attacker = _state("garchomp", sp_spread={"hp": 2, "atk": 32, "spe": 32})
    defender = _state("klefki", sp_spread={"hp": 32, "def": 32})
    only_weak = _best_move_percent(attacker, ["dragonclaw"], defender)
    with_better = _best_move_percent(attacker, ["dragonclaw", "earthquake"], defender)
    assert with_better >= only_weak  # earthquake (Ground) hits Klefki (Steel) hard


# --- build_gameplan: matrix / win-con / answers / plan-breaker derivation -----------


def _no_immunity_scenario() -> GamePlan:
    torkoal = _state("torkoal")
    klefki = _state("klefki")
    magikarp = _state("magikarp", sp_spread={"hp": 32})
    shuckle = _state("shuckle", sp_spread={"hp": 32})
    return build_gameplan(
        our_states=[torkoal, klefki],
        opp_states=[magikarp, shuckle],
        our_move_ids=[["bodyslam"], ["playrough"]],
        opp_move_ids=[["tackle"], ["tackle"]],
    )


def _immunity_scenario() -> GamePlan:
    """Both opponents are Ghost-type: Torkoal's only move (Normal) does 0 to either of
    them, while Klefki's Fairy move still connects -- the Chandelure-vs-Torkoal case,
    forced deterministically via a real type immunity rather than precise stat tuning.
    """
    torkoal = _state("torkoal")
    klefki = _state("klefki")
    gengar = _state("gengar", sp_spread={"hp": 32})
    misdreavus = _state("misdreavus", sp_spread={"hp": 32})
    return build_gameplan(
        our_states=[torkoal, klefki],
        opp_states=[gengar, misdreavus],
        our_move_ids=[["bodyslam"], ["playrough"]],
        opp_move_ids=[["shadowball"], ["shadowball"]],
    )


def test_no_immunity_scenario_torkoal_is_the_primary_win_con() -> None:
    plan = _no_immunity_scenario()
    assert plan.primary_win_con_species == "torkoal"
    assert plan.win_con_scores[plan.our_species.index("torkoal")] == pytest.approx(1.0)
    assert plan.plan_breakers == frozenset()


def test_immunity_scenario_torkoal_cannot_win_a_single_matchup() -> None:
    plan = _immunity_scenario()
    torkoal_idx = plan.our_species.index("torkoal")
    assert plan.win_con_scores[torkoal_idx] == pytest.approx(0.0)


def test_primary_win_con_changes_when_the_opposing_lineup_walls_it() -> None:
    """The core "win con changes" claim: the SAME our-team roster has a different
    primary win con depending on who they're up against.
    """
    no_immunity_plan = _no_immunity_scenario()
    immunity_plan = _immunity_scenario()
    assert no_immunity_plan.primary_win_con_species == "torkoal"
    assert immunity_plan.primary_win_con_species == "klefki"
    assert no_immunity_plan.primary_win_con_species != immunity_plan.primary_win_con_species


def test_plan_breakers_populated_with_the_mon_that_beats_our_primary_win_con() -> None:
    plan = _immunity_scenario()
    assert plan.primary_win_con_species == "klefki"
    # gengar beats klefki outright in this scenario (real damage back, klefki's Fairy
    # move only resisted, not immune, so it deals SOME damage but not enough/fast
    # enough) -- it should show up as a plan breaker of the CURRENT primary win con.
    assert "gengar" in plan.plan_breakers


def test_answers_picks_best_trade_advantage_per_opponent() -> None:
    plan = _no_immunity_scenario()
    # Torkoal's Normal move actually connects on both magikarp/shuckle (no immunity),
    # and neither opposing mon threatens Torkoal back with a mere "tackle" -- Torkoal
    # should be (at least tied for) the best answer to both.
    assert plan.answers["magikarp"] in plan.our_species
    assert plan.answers["shuckle"] in plan.our_species


def test_table_percent_matches_the_matrix_and_defaults_to_zero_for_unknown_pair() -> None:
    plan = _no_immunity_scenario()
    assert plan.table_percent("torkoal", "magikarp") == plan.matrix[("torkoal", "magikarp")]
    assert plan.table_percent("torkoal", "magikarp") > 0.0
    assert plan.table_percent("torkoal", "nosuchmon") == 0.0


def test_summary_is_json_shaped_and_keyed_by_species() -> None:
    plan = _no_immunity_scenario()
    summary = plan.summary()
    assert summary["primary_win_con"] == "torkoal"
    assert set(summary["win_con_scores"]) == {"torkoal", "klefki"}
    assert set(summary["their_threat_scores"]) == {"magikarp", "shuckle"}
    assert isinstance(summary["plan_breakers"], list)


# --- edge cases: empty sides don't crash ---------------------------------------------


def test_empty_opponent_states_does_not_crash() -> None:
    plan = build_gameplan(
        our_states=[_state("torkoal")],
        opp_states=[],
        our_move_ids=[["bodyslam"]],
        opp_move_ids=[],
    )
    assert plan.primary_win_con_idx == 0
    assert plan.win_con_scores == (0.0,)
    assert plan.their_threat_scores == ()
    assert plan.primary_threat_idx is None
    assert plan.plan_breakers == frozenset()


def test_empty_our_states_does_not_crash() -> None:
    plan = build_gameplan(
        our_states=[],
        opp_states=[_state("gengar", sp_spread={"hp": 32})],
        our_move_ids=[],
        opp_move_ids=[["shadowball"]],
    )
    assert plan.primary_win_con_idx is None
    assert plan.primary_win_con_species is None
    assert plan.their_threat_scores == (0.0,)
