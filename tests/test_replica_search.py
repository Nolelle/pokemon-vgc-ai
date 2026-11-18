import numpy as np
import pytest

from vgc.rl.replica_search import SampledWorld, search_sampled_worlds


def test_common_root_strategy_and_world_specific_responses():
    # The two worlds favour opposite actions. A player must choose without knowing
    # which world is real, so solving each world separately would be misleading.
    seen = []
    worlds = [SampledWorld([[1], [-1]], ["x"], 0.5), SampledWorld([[-1], [1]], ["y"], 0.5)]

    def select(observation):
        seen.append(observation)
        return observation

    result = search_sampled_worlds(
        ["a", "b"], worlds, select, lambda state, _a, _b: state, action_key=str, iterations=1000
    )
    assert seen == [["a", "b"], ["x"], ["y"]]
    assert result.actions == ("a", "b")
    assert result.value == pytest.approx(0, abs=0.01)
    assert result.exploitability < 0.01
    assert sum(result.probabilities) == pytest.approx(1)


def test_matching_pennies_and_limits():
    worlds = [SampledWorld([[1, -1], [-1, 1]], ["c", "d"])]
    result = search_sampled_worlds(
        ["a", "b"],
        worlds,
        lambda obs: obs,
        lambda state, _a, _b: state,
        action_key=str,
        iterations=1000,
    )
    assert result.probabilities == pytest.approx((0.5, 0.5), abs=0.02)
    assert result.exploitability < 0.03
    with pytest.raises(ValueError, match="world count"):
        search_sampled_worlds(
            ["a"], worlds * 17, lambda obs: obs, lambda state, _a, _b: state, action_key=str
        )
    with pytest.raises(ValueError, match="non-finite"):
        search_sampled_worlds(
            ["a"],
            [SampledWorld([[np.nan]], ["b"])],
            lambda obs: obs,
            lambda state, _a, _b: state,
            action_key=str,
        )


def test_same_replies_in_different_order_are_aligned_before_solving():
    worlds = [
        SampledWorld([[2, 0], [0, 1]], ["x", "y"]),
        SampledWorld([[0, 2], [1, 0]], ["y", "x"]),
    ]
    result = search_sampled_worlds(
        ["a", "b"],
        worlds,
        lambda obs: obs,
        lambda state, _a, _b: state,
        action_key=str,
        worst_case_weight=1,
        iterations=10000,
    )
    assert result.probabilities == pytest.approx((1 / 3, 2 / 3), abs=0.02)
    assert result.opponent_probabilities[0] == pytest.approx(
        tuple(reversed(result.opponent_probabilities[1])), abs=0.001
    )
    assert result.exploitability < 0.02


def test_rejected_switch_is_retried_once_without_an_extra_step():
    from types import SimpleNamespace

    from offline.replica_search_probe import resolve_rejected_choices

    class Battle:
        ended = False
        hidden_trap_rejections = 0
        battles = {"p1": SimpleNamespace(turn=3), "p2": SimpleNamespace(turn=3)}

        def __init__(self):
            self.calls = []

        def step(self, choices):
            self.calls.append(choices.copy())
            if len(self.calls) == 1:
                self.hidden_trap_rejections += 1
            else:
                self.battles["p1"].turn = 4
                self.battles["p2"].turn = 4

        def sides_to_move(self):
            return ["p1"]

    clone = Battle()
    retries = resolve_rejected_choices(
        clone,
        {"p1": "switch 3", "p2": "move 1"},
        lambda view: ["switch 3", "move 2"],
    )
    assert retries == 1
    assert clone.calls == [{"p1": "switch 3", "p2": "move 1"}, {"p1": "move 2"}]
