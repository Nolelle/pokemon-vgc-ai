"""LiveExactMirror roots must actually execute Showdown branches after patchPublic.

A patched battle whose BattleQueue was replaced with a plain array used to throw inside
the next `go()`, which the worker drains swallowed. Every clone.step then returned no
protocol, no request, and an unchanged public state -- so exact search scored every
candidate with the same exchange_value. These tests fail closed on that signature.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest

from vgc.actions import enumerate_joint_orders
from vgc.config import REPO_ROOT
from vgc.mechanics_state import BattleMechanicsState, snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.exact_search import search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror
from vgc.rl.mechanics_oracle import evaluate_exact_branches

pytestmark = pytest.mark.integration

POOL = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150"
OWN_TEAM_PATH = POOL / "rain_offense" / "team_11.packed.txt"
OPP_TEAM_PATH = POOL / "triple_setup_balance" / "team_02.packed.txt"

COMPACT_CONFIG = replace(
    PolicyConfig(),
    search_our_candidates=3,
    search_opp_candidates=2,
    search_opp_moves_per_slot=1,
    exact_search_future_samples=1,
    exact_search_state_hypotheses=1,
    exact_search_spread_hypotheses=2,
    exact_search_set_hypotheses=1,
    exact_search_bring_hypotheses=1,
    exact_search_total_hypotheses=2,
    use_rolling_horizon=False,
    use_value_head=False,
)

_CLONE_IDS = itertools.count()


def _require_showdown_and_pool() -> tuple[str, str]:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    if not OWN_TEAM_PATH.is_file() or not OPP_TEAM_PATH.is_file():
        pytest.skip("archetype_pool_150 teams are unavailable")
    return OWN_TEAM_PATH.read_text().strip(), OPP_TEAM_PATH.read_text().strip()


def _progress_signature(state: BattleMechanicsState) -> tuple:
    hp = tuple(
        (mon.species_id, mon.current_hp, mon.fainted)
        for mon in (*state.our_side.pokemon, *state.opponent_side.pokemon)
    )
    return state.turn, hp


def _joint_choices(root: DirectBattle, count: int) -> list[dict[str, str]]:
    expected = set(root.sides_to_move())
    assert expected == {"p1", "p2"}, (
        f"need a both-sides-to-move turn, not a forced switch; waiting on {sorted(expected)}"
    )
    orders = {side: enumerate_joint_orders(root.battles[side]) for side in expected}
    assert all(orders[side] for side in expected)
    p2_choice = choice_string(orders["p2"][0])
    choices: list[dict[str, str]] = []
    seen: set[str] = set()
    for order in orders["p1"]:
        encoded = choice_string(order)
        if encoded in seen:
            continue
        seen.add(encoded)
        choices.append({"p1": encoded, "p2": p2_choice})
        if len(choices) >= count:
            break
    if len(choices) < count and len(orders["p2"]) > 1:
        choices.append(
            {"p1": choice_string(orders["p1"][0]), "p2": choice_string(orders["p2"][1])}
        )
    assert len(choices) >= count, "not enough distinct joint choices on this turn"
    return choices[:count]


def _assert_clone_step_advances(
    root: DirectBattle, choices: dict[str, str], *, label: str
) -> None:
    before = snapshot_battle(root.battles["p1"])
    clone = root.clone(f"{root.battle_id}-{label}-{next(_CLONE_IDS)}")
    try:
        result = clone.step(dict(choices))
        assert result.lines["p1"], f"{label}: clone.step returned no p1 protocol lines"
        assert result.request_state, f"{label}: clone.step returned an empty request_state"
        after = snapshot_battle(clone.battles["p1"])
        assert _progress_signature(after) != _progress_signature(before), (
            f"{label}: clone snapshot did not advance (turn or HP unchanged)"
        )
    finally:
        clone.close()


@contextmanager
def _mirror_turn1(
    config: PolicyConfig = COMPACT_CONFIG,
) -> Iterator[tuple[DirectBattle, LiveExactMirror, DirectBattle]]:
    own_team, opp_team = _require_showdown_and_pool()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-branch-source",
            own_team,
            opp_team,
            seed=[7, 8, 9, 10],
        )
        mirror: LiveExactMirror | None = None
        root: DirectBattle | None = None
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            assert set(source.sides_to_move()) == {"p1", "p2"}
            mirror = LiveExactMirror(own_team, config)
            root = mirror.build(source.battles["p1"])
            assert set(root.sides_to_move()) == {"p1", "p2"}
            yield source, mirror, root
        finally:
            if root is not None:
                root.close()
            if mirror is not None:
                mirror.close()
            source.close()


def test_mirror_clone_step_emits_protocol_and_advances_public_state() -> None:
    with _mirror_turn1() as (source, mirror, root):
        choices = _joint_choices(root, 1)[0]
        _assert_clone_step_advances(root, choices, label="built")
        hypotheses = mirror.hypotheses(source.battles["p1"])
        if len(hypotheses) <= 1:
            # Only one hidden-information hypothesis; rebase sub-assert skipped.
            return
        rebased = mirror.rebase(root, source.battles["p1"], hypotheses[1])
        try:
            rebase_choices = _joint_choices(rebased, 1)[0]
            _assert_clone_step_advances(rebased, rebase_choices, label="rebased")
        finally:
            # `_mirror_turn1` closes the original root; rebase may have replaced it.
            if rebased is not root:
                rebased.close()


def test_mirror_exact_branches_diverge_from_each_other_and_the_root() -> None:
    with _mirror_turn1() as (_source, _mirror, root):
        pair = _joint_choices(root, 2)
        branches = evaluate_exact_branches(root, pair)
        assert len(branches) == 2
        root_state = snapshot_battle(root.battles["p1"])
        first = branches[0].state_for("p1")
        second = branches[1].state_for("p1")
        assert first != root_state
        assert second != root_state
        assert first != second


def test_mirror_exact_search_exchange_values_are_not_constant() -> None:
    with _mirror_turn1() as (_source, _mirror, root):
        scored = search_joint_orders_exact(root, "p1", COMPACT_CONFIG)
        searched = [entry for entry in scored if entry.breakdown.get("searched")]
        assert len(searched) >= 3
        exchanges = [float(entry.breakdown["exchange_value"]) for entry in searched]
        assert statistics.pstdev(exchanges) > 0
        assert len({round(value, 9) for value in exchanges}) >= 2
        offsets = [
            float(entry.score) - float(entry.breakdown["myopic_score"]) for entry in searched
        ]
        assert len({round(value, 9) for value in offsets}) >= 2, (
            "exact search collapsed to a constant offset of the myopic score"
        )


def test_patch_public_leaves_a_playable_request_and_null_stream_error() -> None:
    with _mirror_turn1() as (_source, _mirror, root):
        inspected = root.inspect()
        assert inspected["requestState"] in {"move", "switch"}
        assert "error" in inspected
        assert inspected["error"] is None
        choices = _joint_choices(root, 1)[0]
        clone = root.clone(f"{root.battle_id}-worker-contract-{next(_CLONE_IDS)}")
        try:
            result = clone.step(dict(choices))
            assert result.error is None
            assert result.lines["p1"]
            assert result.lines["p2"]
        finally:
            clone.close()
