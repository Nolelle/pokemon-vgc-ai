"""Multi-turn exact continuation: turn counting, forced switches, N=0 equivalence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from vgc.actions import enumerate_joint_orders
from vgc.config import REPO_ROOT
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.exact_search import _position_value, search_joint_orders_exact
from vgc.rl.mechanics_oracle import evaluate_exact_branches

pytestmark = pytest.mark.integration

SEEDS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)]


def _root(worker: SimWorker, name: str) -> DirectBattle:
    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    root = DirectBattle.start(worker, name, team, team, seed=[7, 8, 9, 10])
    root.step({"p1": "team 1234", "p2": "team 1234"})
    return root


def _choices(root: DirectBattle) -> list[dict[str, str]]:
    first = [choice_string(o) for o in enumerate_joint_orders(root.battles["p1"])[:3]]
    opp = choice_string(enumerate_joint_orders(root.battles["p2"])[0])
    return [{"p1": c, "p2": opp} for c in first]


def _turn(clone_state) -> int:
    return clone_state.turn


def test_continuation_one_turn_and_root_untouched() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    cfg0 = PolicyConfig()
    cfg1 = replace(cfg0, exact_search_continuation_turns=1)
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = _root(worker, "cont-1")
        base = root.battles["p1"].turn
        before = worker.request({"cmd": "inspect", "id": root.battle_id})["stateHash"]
        choices = _choices(root)
        zero = evaluate_exact_branches(root, choices, future_seeds=SEEDS, config=cfg0)
        one = evaluate_exact_branches(root, choices, future_seeds=SEEDS, config=cfg1)
        after = worker.request({"cmd": "inspect", "id": root.battle_id})["stateHash"]
        root.close()
    assert before == after
    assert [b.branch_id for b in zero] == [b.branch_id for b in one]
    assert all(b.continuation_turns_completed == 0 for b in zero)
    for z, o in zip(zero, one, strict=True):
        if o.ended:
            assert o.continuation_ended_early or o.continuation_turns_completed == 1
        else:
            assert o.continuation_turns_completed == 1
            assert o.state_for("p1").turn == z.state_for("p1").turn + 1 == base + 2
    cfg = PolicyConfig()
    assert any(
        _position_value(o.state_for("p1"), cfg) != _position_value(z.state_for("p1"), cfg)
        for z, o in zip(zero, one, strict=True)
    )


def test_continuation_resolves_forced_switches_and_keeps_going() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    cfg = replace(PolicyConfig(), exact_search_continuation_turns=4)
    seeds = [(s, s + 1, s + 2, s + 3) for s in range(1, 40, 4)]
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = _root(worker, "cont-faint")
        branches = evaluate_exact_branches(root, _choices(root)[:1], future_seeds=seeds, config=cfg)
        root.close()
    assert not any(b.continuation_truncated for b in branches)
    # A forced replacement is an extra step inside a turn: steps exceed completed turns.
    assert any(b.continuation_steps > b.continuation_turns_completed for b in branches)
    assert all(b.continuation_turns_completed == 4 or b.ended for b in branches)


def test_zero_continuation_ranking_matches_no_config_path() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    cfg = replace(
        PolicyConfig(),
        search_our_candidates=3,
        search_opp_candidates=2,
        exact_search_future_samples=2,
        use_rolling_horizon=False,
        use_value_head=False,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = _root(worker, "cont-0")
        choices = _choices(root)
        legacy = evaluate_exact_branches(root, choices, future_seeds=SEEDS)
        new = evaluate_exact_branches(root, choices, future_seeds=SEEDS, config=cfg)
        ranked = search_joint_orders_exact(root, "p1", cfg)
        root.close()
    # public_lines is excluded: the legacy path itself reorders protocol chunks run to run.
    assert [replace(b, public_lines=()) for b in legacy] == [
        replace(b, public_lines=()) for b in new
    ]
    assert ranked[0].breakdown["search_metrics"]["continuation_turns"] == 0


def test_search_continuation_mode_values_and_root_untouched() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    cfg0 = PolicyConfig()
    cfgp = replace(cfg0, exact_search_continuation_turns=1)
    cfgs = replace(cfgp, exact_search_continuation_mode="search")

    def score(state):
        return _position_value(state, cfg0)

    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        root = _root(worker, "cont-search")
        before = worker.request({"cmd": "inspect", "id": root.battle_id})["stateHash"]
        choices = _choices(root)
        pol = evaluate_exact_branches(root, choices, future_seeds=SEEDS, config=cfgp)
        sea = evaluate_exact_branches(
            root, choices, future_seeds=SEEDS, config=cfgs, our_side="p1", score_state=score
        )
        after = worker.request({"cmd": "inspect", "id": root.battle_id})["stateHash"]
        root.close()
    assert before == after
    assert [b.branch_id for b in pol] == [b.branch_id for b in sea]
    live = [b for b in sea if not b.ended]
    assert live and all(b.continuation_value is not None for b in live)
    assert all(b.continuation_value is None for b in pol)
    assert any(
        s.continuation_value != _position_value(p.state_for("p1"), cfg0)
        for p, s in zip(pol, sea, strict=True)
        if s.continuation_value is not None
    )
