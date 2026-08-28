from __future__ import annotations

from dataclasses import replace

import pytest

from vgc.config import REPO_ROOT
from vgc.models import PolicyConfig
from vgc.mechanics_state import snapshot_battle
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.exact_search import combine_belief_rankings, search_joint_orders_exact
from vgc.rl.hidden_state import HiddenStateHypothesis, enumerate_hidden_state_hypotheses
from vgc.rl.live_mirror import LiveExactMirror
from vgc.actions import enumerate_joint_orders
from vgc.rl.env import choice_string


@pytest.mark.integration
def test_exact_search_scores_cloned_showdown_branches_without_mutating_root() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = (REPO_ROOT / "teams" / "phase2_mirror.packed.txt").read_text().strip()
    config = replace(
        PolicyConfig(),
        search_our_candidates=2,
        search_opp_candidates=2,
        exact_search_future_samples=2,
        use_rolling_horizon=False,
        use_value_head=False,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = DirectBattle.start(worker, "exact-search", team, team, seed=[7, 8, 9, 10])
        root.step({"p1": "team 1234", "p2": "team 1234"})
        before = worker.request({"cmd": "inspect", "id": root.battle_id})["stateHash"]
        scored = search_joint_orders_exact(root, "p1", config)
        after = worker.request({"cmd": "inspect", "id": root.battle_id})["stateHash"]
        root.close()

    assert scored
    assert before == after
    searched = [entry for entry in scored if entry.breakdown.get("searched")]
    assert len(searched) == 2
    assert all(
        entry.breakdown["mechanics_source"] == "official_showdown_clone"
        for entry in searched
    )
    assert all(entry.breakdown["approximate_transition"] is False for entry in searched)
    assert all(entry.breakdown["exact_random_samples"] == 2 for entry in searched)


@pytest.mark.integration
def test_later_turn_public_state_can_be_rebuilt_into_an_exact_live_root() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    config = replace(
        PolicyConfig(),
        search_our_candidates=1,
        search_opp_candidates=1,
        exact_search_future_samples=1,
        use_rolling_horizon=False,
        use_value_head=False,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(source_worker, "live-source", team, team, seed=[3, 4, 5, 6])
        source.step({"p1": "team 1234", "p2": "team 1234"})
        source.step(
            {
                "p1": choice_string(enumerate_joint_orders(source.battles["p1"])[0]),
                "p2": choice_string(enumerate_joint_orders(source.battles["p2"])[0]),
            }
        )
        assert source.battles["p1"].turn >= 2
        mirror = LiveExactMirror(team, config)
        try:
            root = mirror.build(source.battles["p1"])
            try:
                scored = search_joint_orders_exact(root, "p1", config)
            finally:
                root.close()
        finally:
            mirror.close()
            source.close()

    assert scored
    assert scored[0].breakdown["mechanics_source"] == "official_showdown_clone"
    assert scored[0].breakdown["approximate_transition"] is False


@pytest.mark.integration
def test_hidden_sleep_beliefs_reach_showdown_as_different_legal_states() -> None:
    """A belief branch must change the real engine state, not just a Python number."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = DirectBattle.start(worker, "belief-patch", team, team, seed=[11, 12, 13, 14])
        root.step({"p1": "team 1234", "p2": "team 1234"})
        observed = snapshot_battle(root.battles["p1"])
        asleep = replace(observed.our_side.pokemon[0], status="slp", status_counter=0)
        state = replace(
            observed,
            our_side=replace(
                observed.our_side,
                pokemon=(asleep, *observed.our_side.pokemon[1:]),
            ),
        )
        hypotheses = enumerate_hidden_state_hypotheses(state)

        remaining = []
        for hypothesis in hypotheses:
            root.patch_public_state(state, perspective="p1", hidden_hypothesis=hypothesis.payload)
            dumped = worker.request({"cmd": "dump", "id": root.battle_id})["state"]
            sleeper = next(
                mon
                for mon in dumped["sides"][0]["pokemon"]
                if mon.get("status") == "slp"
            )
            remaining.append(sleeper["statusState"]["time"])
        root.close()

    # Champions rolls `sample([2, 3, 3])`, so both durations must survive as branches.
    assert sorted(remaining) == [2, 3]
    assert sorted(round(h.weight, 6) for h in hypotheses) == [
        round(1.0 / 3.0, 6),
        round(2.0 / 3.0, 6),
    ]


@pytest.mark.integration
def test_one_mirror_root_is_reused_across_belief_branches() -> None:
    """Rebasing a mirror must survive having already had branches cloned from it."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    config = replace(
        PolicyConfig(),
        search_our_candidates=1,
        search_opp_candidates=1,
        exact_search_future_samples=1,
        use_rolling_horizon=False,
        use_value_head=False,
    )
    beliefs = [
        HiddenStateHypothesis(weight=0.6, payload={}),
        HiddenStateHypothesis(weight=0.4, payload={}),
    ]
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(source_worker, "belief-loop", team, team, seed=[9, 1, 4, 2])
        source.step({"p1": "team 1234", "p2": "team 1234"})
        observation = source.battles["p1"]
        mirror = LiveExactMirror(team, config)
        root = None
        rankings = []
        try:
            for belief in beliefs:
                root = (
                    mirror.rebase(root, observation, belief)
                    if root is not None
                    else mirror.build(observation, belief)
                )
                rankings.append(
                    (belief.weight, search_joint_orders_exact(root, "p1", config))
                )
        finally:
            if root is not None:
                root.close()
            mirror.close()
            source.close()

    combined = combine_belief_rankings(rankings)

    assert combined
    assert combined[0].breakdown["belief_branches"] == 2
    assert combined[0].breakdown["mechanics_source"] == "official_showdown_clone"
    # Identical beliefs must not drift: the combined score is the shared branch score.
    assert combined[0].score == pytest.approx(rankings[0][1][0].score)
