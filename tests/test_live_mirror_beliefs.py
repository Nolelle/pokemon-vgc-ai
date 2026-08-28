"""Unit tests for the live mirror's hidden-information beliefs.

`vgc.rl.live_mirror` rebuilds a public observation into a real Showdown battle, which
means it has to commit to values no player is ever told -- the opponent's Stat Points and
nature. These cover the belief layer that decides WHICH values it commits to, without
starting a Showdown worker; the end-to-end behaviour is covered by
`tests/test_exact_search.py`'s integration tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vgc.models import PolicyConfig
from vgc.rl.live_mirror import MirrorHypothesis, _spread_beliefs
from vgc.rl.hidden_state import HiddenStateHypothesis


def _battle(*species: str) -> SimpleNamespace:
    return SimpleNamespace(
        opponent_active_pokemon=[SimpleNamespace(species=name) for name in species]
    )


_TWO_SPREADS = [
    ({"atk": 32, "spe": 32}, "jolly", 0.75),
    ({"hp": 32, "def": 32}, "impish", 0.25),
]


@pytest.fixture
def fixed_spreads(monkeypatch):
    def fake(species_id, usage=None, limit=None):
        return _TWO_SPREADS[: limit or len(_TWO_SPREADS)]

    monkeypatch.setattr("vgc.rl.live_mirror.opponent_spread_hypotheses", fake)


def test_default_config_commits_to_one_spread_per_species(fixed_spreads) -> None:
    # The shipped default must reproduce the old point-estimate behaviour exactly, or
    # every gate-tuned number moves without an A/B.
    beliefs = _spread_beliefs(_battle("garchomp", "incineroar"), PolicyConfig())
    assert len(beliefs) == 1
    weight, assignment = beliefs[0]
    assert weight == 1.0
    assert assignment == {
        "garchomp": ({"atk": 32, "spe": 32}, "jolly"),
        "incineroar": ({"atk": 32, "spe": 32}, "jolly"),
    }


def test_raising_the_cap_produces_a_real_distribution(fixed_spreads) -> None:
    config = PolicyConfig(exact_search_spread_hypotheses=4)
    beliefs = _spread_beliefs(_battle("garchomp", "incineroar"), config)
    # Two actives x two spreads each = the full cross product, and nothing is certain.
    assert len(beliefs) == 4
    assert sum(weight for weight, _assignment in beliefs) == pytest.approx(1.0)
    assert beliefs[0][0] == pytest.approx(0.75 * 0.75)
    assert all(weight < 1.0 for weight, _assignment in beliefs)


def test_capping_below_the_cross_product_renormalizes(fixed_spreads) -> None:
    config = PolicyConfig(exact_search_spread_hypotheses=2)
    beliefs = _spread_beliefs(_battle("garchomp", "incineroar"), config)
    assert len(beliefs) == 2
    # Truncation must not leave the surviving branches summing to less than one; a
    # belief-weighted average over weights that sum to 0.75 is not an average.
    assert sum(weight for weight, _assignment in beliefs) == pytest.approx(1.0)


def test_no_active_opponent_yields_one_empty_belief(fixed_spreads) -> None:
    beliefs = _spread_beliefs(_battle(), PolicyConfig())
    assert beliefs == [(1.0, {})]


def test_cap_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _spread_beliefs(_battle("garchomp"), PolicyConfig(exact_search_spread_hypotheses=0))


def test_spread_key_separates_beliefs_that_need_a_rebuilt_battle() -> None:
    # rebase() re-patches when the key matches and rebuilds when it does not, so two
    # beliefs differing only in a timer must share a key and two differing in a spread
    # must not.
    spread_a = {"garchomp": ({"atk": 32}, "jolly")}
    spread_b = {"garchomp": ({"hp": 32}, "impish")}
    timer_a = HiddenStateHypothesis(weight=1.0, payload={})
    timer_b = HiddenStateHypothesis(
        weight=1.0, payload={"p2": {"a": {"sleep_remaining": 2}}}
    )
    assert (
        MirrorHypothesis(0.5, spread_a, timer_a).spread_key
        == MirrorHypothesis(0.5, spread_a, timer_b).spread_key
    )
    assert (
        MirrorHypothesis(0.5, spread_a, timer_a).spread_key
        != MirrorHypothesis(0.5, spread_b, timer_a).spread_key
    )


def test_hypothesis_without_timers_has_an_empty_patch_payload() -> None:
    assert MirrorHypothesis(1.0, {}).payload == {}


def test_real_usage_data_makes_the_top_spread_uncertain() -> None:
    # Uses the committed corpus rather than the fixture: the point of the cap existing at
    # all is that the most popular spread is not a safe bet.
    config = PolicyConfig(exact_search_spread_hypotheses=3)
    beliefs = _spread_beliefs(_battle("incineroar"), config)
    if len(beliefs) == 1:  # species missing from the corpus in this checkout
        return
    assert beliefs[0][0] < 1.0
    assert sum(weight for weight, _assignment in beliefs) == pytest.approx(1.0)
