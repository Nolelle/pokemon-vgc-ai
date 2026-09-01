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
from vgc.battle_memory import BattleMemory, SpeedObservation
from vgc.rl.live_mirror import LiveExactMirror, MirrorHypothesis, _spread_beliefs
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


def test_default_config_keeps_multiple_material_spread_possibilities(fixed_spreads) -> None:
    beliefs = _spread_beliefs(_battle("garchomp", "incineroar"), PolicyConfig())
    assert len(beliefs) == 2
    assert sum(weight for weight, _assignment in beliefs) == pytest.approx(1.0)
    assert all(0.0 < weight < 1.0 for weight, _assignment in beliefs)


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


# --- beliefs must narrow as the battle reveals things -------------------------------
#
# vgc.opponent_belief already reweights the corpus prior by observed move order and
# damage. These check the mirror consumes that POSTERIOR rather than the flat prior, so
# "they outsped my Pokemon" actually shrinks the hypothesis set the search branches over.


def _battle_with_preview(*species: str) -> SimpleNamespace:
    mons = [SimpleNamespace(species=name, fainted=False, moves={}, item=None,
                            ability=None, status=None, current_hp_fraction=1.0)
            for name in species]
    return SimpleNamespace(
        opponent_active_pokemon=mons,
        teampreview_opponent_team=mons,
        opponent_team={name: mon for name, mon in zip(species, mons, strict=True)},
    )


def _speed_evidence(species_id: str, threshold: float) -> BattleMemory:
    memory = BattleMemory(battle_tag="test")
    memory.speed_observations.append(
        SpeedObservation(
            opponent_species=species_id, threshold=threshold, relation="at_least"
        )
    )
    return memory


def test_observing_that_they_outsped_us_reweights_toward_fast_spreads() -> None:
    config = PolicyConfig(exact_search_spread_hypotheses=3)
    battle = _battle_with_preview("charizard")

    prior = _spread_beliefs(battle, config)
    if len(prior) < 2:
        pytest.skip("charizard has a single corpus spread in this checkout")

    def speed_of(assignment):
        from vgc.stats import calculate_stats

        spread, nature = assignment["charizard"]
        return calculate_stats("charizard", spread, nature)["spe"]

    speeds = sorted(speed_of(assignment) for _weight, assignment in prior)
    # Say we watched it move before something faster than its slowest hypothesis.
    memory = _speed_evidence("charizard", float(speeds[-1]))
    posterior = _spread_beliefs(battle, config, memory)

    prior_top = max(prior, key=lambda row: row[0])
    posterior_top = max(posterior, key=lambda row: row[0])
    # The surviving belief must be one that can actually reach the observed Speed.
    assert speed_of(posterior_top[1]) >= speeds[-1]
    assert posterior_top[0] > prior_top[0]
    assert sum(weight for weight, _assignment in posterior) == pytest.approx(1.0)


def test_contradicted_spreads_are_suppressed_but_not_deleted() -> None:
    # vgc.opponent_belief keeps a mismatch at 5% rather than zero, because speed ties and
    # crits make a single observation noisy. The mirror must not launder that into
    # certainty.
    config = PolicyConfig(exact_search_spread_hypotheses=3)
    battle = _battle_with_preview("charizard")
    prior = _spread_beliefs(battle, config)
    if len(prior) < 2:
        pytest.skip("charizard has a single corpus spread in this checkout")
    memory = _speed_evidence("charizard", 9999.0)  # nothing can be this fast
    posterior = _spread_beliefs(battle, config, memory)
    assert len(posterior) == len(prior)
    assert all(0.0 < weight < 1.0 for weight, _assignment in posterior)


def test_no_memory_falls_back_to_the_flat_prior(fixed_spreads) -> None:
    config = PolicyConfig(exact_search_spread_hypotheses=2)
    battle = _battle("garchomp")
    assert _spread_beliefs(battle, config, None) == _spread_beliefs(battle, config)


def test_combined_hidden_configurations_are_capped_and_renormalized(monkeypatch) -> None:
    config = PolicyConfig(exact_search_total_hypotheses=8)
    mirror = object.__new__(LiveExactMirror)
    mirror.config = config
    mirror.last_hypothesis_audit = {}
    monkeypatch.setattr("vgc.rl.live_mirror.snapshot_battle", lambda _battle: object())
    monkeypatch.setattr(
        "vgc.rl.live_mirror.enumerate_hidden_state_hypotheses",
        lambda _state, _config: [
            HiddenStateHypothesis(0.6),
            HiddenStateHypothesis(0.4, {"opponent": {"x": {"sleepTime": 2}}}),
        ],
    )
    monkeypatch.setattr(
        "vgc.rl.live_mirror._spread_beliefs",
        lambda _battle, _config, _memory: [(0.7, {"a": ({"spe": 32}, "jolly")}), (0.3, {})],
    )
    monkeypatch.setattr(
        "vgc.rl.live_mirror._set_beliefs",
        lambda _battle, _config, _memory: [(0.8, {"a": {"moves": ("protect",)}}), (0.2, {})],
    )
    monkeypatch.setattr(
        "vgc.rl.live_mirror._bring_beliefs",
        lambda _battle, _config: [(0.75, ("a", "b", "c", "d")), (0.25, ("a", "b", "c", "e"))],
    )

    hypotheses = mirror.hypotheses(object())

    assert len(hypotheses) == 8
    assert sum(entry.weight for entry in hypotheses) == pytest.approx(1.0)
    assert mirror.last_hypothesis_audit["total_before_cap"] == 16
    assert mirror.last_hypothesis_audit["searched"] == 8
    assert mirror.last_hypothesis_audit["retained_probability_mass"] == pytest.approx(1.0)
    assert 0.0 < mirror.last_hypothesis_audit["direct_representative_mass"] < 1.0
    assert mirror.last_hypothesis_audit["compressed"] is True
