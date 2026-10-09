"""Repeat Protect-family rolls are enumerated exactly, not sampled.

A protect-family move used while the user still holds Showdown's `stall` volatile succeeds
with probability 1/counter (1/3, then 1/9). That roll is one PRNG draw, so with the judge's
two future samples a 1/3 Protect was represented as 0, 1 or 2 successes out of 2. With
`PolicyConfig.exact_search_exact_stall_odds` the oracle runs both forced outcomes of every
repeat roll (every combination when several Pokemon repeat) and weights them by the true
odds. Ladder evidence behind the fix: 25 back-to-back Protects in 46 games, ~2/3 failed.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from tests.test_live_mirror_branches import COMPACT_CONFIG, _require_showdown_and_pool
from vgc.config import REPO_ROOT
from vgc.exact_judge import judge_config
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.exact_search import _position_value, search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror
from vgc.rl.mechanics_oracle import (
    evaluate_exact_branches,
    repeat_stall_rolls,
    stall_outcomes,
)

# Own team (rain_offense team_11) leads Pelipper + Archaludon; move 4 is Tailwind for
# Pelipper and Protect for Archaludon. Used on turn 1, it leaves Archaludon at counter 3.
BOTH_MOVE_4 = "move 4, move 4"
PROTECT_LINE = "|-singleturn|p1b: Archaludon|Protect"
EXACT = replace(COMPACT_CONFIG, exact_search_exact_stall_odds=True, exact_search_future_samples=2)
LEGACY = replace(EXACT, exact_search_exact_stall_odds=False)


def _staller(side="p1", position="b", counter=3.0, moves=("tackle", "protect", "wideguard")):
    return {
        "side": side,
        "position": position,
        "counter": counter,
        "moves": list(moves),
        "disabled": [False] * len(moves),
        "rolls": [m in ("protect", "detect", "kingsshield") for m in moves],
    }


# -- pure helpers (no simulator) ------------------------------------------------------------


def test_repeat_rolls_only_for_rolling_moves() -> None:
    stallers = [_staller()]
    assert repeat_stall_rolls(stallers, {"p1": "move tackle 1, move protect"}) == [("p1b", 1 / 3)]
    assert repeat_stall_rolls(stallers, {"p1": "move tackle 1, move 2"}) == [("p1b", 1 / 3)]
    assert repeat_stall_rolls(stallers, {"p1": "move tackle 1, move 1 1"}) == []  # attacks
    assert repeat_stall_rolls(stallers, {"p1": "move 2, move wideguard"}) == []  # no roll
    assert repeat_stall_rolls(stallers, {"p1": "move 2, switch 3"}) == []
    assert repeat_stall_rolls(stallers, {"p2": "move protect, move protect"}) == []
    # Slot a vs b is read from the comma-separated order.
    assert repeat_stall_rolls([_staller(position="a", counter=9.0)],
                              {"p1": "move protect, move tackle 1"}) == [("p1a", 1 / 9)]


def test_stall_outcomes_enumerate_every_combination_with_true_odds() -> None:
    assert stall_outcomes([]) == [({}, 1.0, "")]
    single = stall_outcomes([("p1b", 1 / 3)])
    assert [(f, round(w, 12)) for f, w, _ in single] == [
        ({"p1b": True}, round(1 / 3, 12)),
        ({"p1b": False}, round(2 / 3, 12)),
    ]
    both = stall_outcomes([("p1b", 1 / 3), ("p2a", 1 / 9)])
    assert len(both) == 4 and len({suffix for *_, suffix in both}) == 4
    assert sum(weight for _f, weight, _s in both) == pytest.approx(1.0)
    by_force = {tuple(sorted(f.items())): w for f, w, _ in both}
    assert by_force[(("p1b", True), ("p2a", False))] == pytest.approx(1 / 3 * 8 / 9)
    assert by_force[(("p1b", False), ("p2a", False))] == pytest.approx(2 / 3 * 8 / 9)


def test_exact_stall_knob_ships_on_and_judge_config_passes_it_through() -> None:
    assert PolicyConfig().exact_search_exact_stall_odds is True
    assert judge_config(PolicyConfig(), 6).exact_search_exact_stall_odds is True
    off = replace(PolicyConfig(), exact_search_exact_stall_odds=False)
    assert judge_config(off, 6).exact_search_exact_stall_odds is False


# -- real simulator ------------------------------------------------------------------------


@pytest.mark.integration
def test_worker_roll_list_matches_showdown_source() -> None:
    """The worker's list of rolling moves must be every move that runs `StallMove`."""

    moves_ts = DEFAULT_SHOWDOWN_REPO / "data" / "moves.ts"
    if not moves_ts.is_file():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    rolling: set[str] = set()
    current = ""
    for line in moves_ts.read_text().splitlines():
        header = re.match(r"^\t([a-z0-9]+): \{$", line)
        if header:
            current = header.group(1)
        elif "runEvent('StallMove'" in line:
            rolling.add(current)
    worker_src = (REPO_ROOT / "tools" / "sim_worker.mjs").read_text()
    block = re.search(r"STALL_ROLL_MOVES = new Set\(\[(.*?)\]\)", worker_src, re.S)
    assert block is not None
    assert set(re.findall(r"'([a-z0-9]+)'", block.group(1))) == rolling


def _untimed(lines) -> list[str]:
    return [line for line in lines if not line.startswith("|t:|")]


def _turn_two_root(worker: SimWorker, label: str):
    own_team, opp_team = _require_showdown_and_pool()
    source = DirectBattle.start(worker, f"{label}-source", own_team, opp_team, seed=[1, 2, 3, 4])
    mirror = LiveExactMirror(own_team, EXACT)
    source.step({"p1": "team 1234", "p2": "team 1234"})
    source.step({"p1": BOTH_MOVE_4, "p2": "default"})
    assert set(source.sides_to_move()) == {"p1", "p2"}, "need a normal turn 2"
    root = mirror.build(source.battles["p1"])
    return source, mirror, root


@pytest.mark.integration
def test_forced_stall_outcome_is_deterministic_and_matches_the_natural_roll() -> None:
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source, mirror, root = _turn_two_root(worker, "stall-force")
        try:
            (staller,) = [s for s in root.stall_info() if s["species"] == "archaludon"]
            assert staller["counter"] == 3 and staller["rolls"][3] is True
            choices = {"p1": BOTH_MOVE_4, "p2": "default"}
            natural_successes = 0
            for i in range(24):
                seed = [i + 1, 7, 11, 13]
                lines = {}
                for label, force in (("natural", None), ("win", True), ("fail", False)):
                    clone = root.clone(f"{root.battle_id}-f{i}-{label}", seed=seed,
                                       stall_force=None if force is None else {"p1b": force})
                    try:
                        lines[label] = clone.step(dict(choices)).lines
                    finally:
                        clone.close()
                assert PROTECT_LINE in "\n".join(lines["win"]["p1"]), f"seed {i}"
                assert PROTECT_LINE not in "\n".join(lines["fail"]["p1"]), f"seed {i}"
                natural = PROTECT_LINE in "\n".join(lines["natural"]["p1"])
                natural_successes += natural
                # The forced roll still takes the stock PRNG draw, so forcing the outcome
                # the seed would have produced anyway changes nothing at all.
                same = lines["win" if natural else "fail"]
                for side in ("p1", "p2"):  # ignore the wall-clock `|t:|` stamps
                    assert _untimed(same[side]) == _untimed(lines["natural"][side]), f"seed {i}"
            assert 1 <= natural_successes < 20  # 1/3 odds, sanity only
            with pytest.raises(Exception, match="no stall volatile"):
                root.clone(f"{root.battle_id}-bad", stall_force={"p2a": True})
        finally:
            root.close()
            mirror.close()
            source.close()


@pytest.mark.integration
def test_exact_branches_weight_both_outcomes_by_the_true_odds() -> None:
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source, mirror, root = _turn_two_root(worker, "stall-weights")
        try:
            choices = [{"p1": BOTH_MOVE_4, "p2": "default"}]
            before = _position_value(snapshot_battle(root.battles["p1"]), EXACT)
            for seeds in ([(1, 2, 3, 4), (5, 6, 7, 8)], [(9, 9, 9, 9), (3, 1, 4, 1)]):
                branches = evaluate_exact_branches(
                    root, choices, future_seeds=seeds, branch_prefix="odds", config=EXACT
                )
                assert len(branches) == 4
                for sample in (0, 1):
                    group = [b for b in branches if b.sample_index == sample]
                    assert sorted(round(b.weight, 9) for b in group) == [
                        round(1 / 3, 9), round(2 / 3, 9)
                    ]
                    assert sum(b.weight for b in group) == pytest.approx(1.0)
                    for b in group:
                        text = "\n".join("\n".join(lines) for _side, lines in b.public_lines)
                        # The stall part is identical whatever the random key: success
                        # always shows the Protect, failure never does.
                        assert (PROTECT_LINE in text) == dict(b.stall_force)["p1b"]
                # Weighted value == 1/3 * success value + 2/3 * failure value, per sample.
                for sample, seed in enumerate(seeds):
                    manual = {}
                    for force in (True, False):
                        clone = root.clone(f"{root.battle_id}-m{sample}{force}", seed=seed,
                                           stall_force={"p1b": force})
                        try:
                            clone.step(dict(choices[0]))
                            state = snapshot_battle(clone.battles["p1"])
                        finally:
                            clone.close()
                        manual[force] = _position_value(state, EXACT) - before
                    weighted = sum(
                        b.weight * (_position_value(b.state_for("p1"), EXACT) - before)
                        for b in branches if b.sample_index == sample
                    )
                    assert weighted == pytest.approx(manual[True] / 3 + manual[False] * 2 / 3)
            # Legacy: one sampled branch per (choice, seed), nothing forced.
            legacy = evaluate_exact_branches(
                root, choices, future_seeds=[(1, 2, 3, 4), (5, 6, 7, 8)], branch_prefix="old",
                config=LEGACY,
            )
            assert [(b.weight, b.stall_force) for b in legacy] == [(1.0, ())] * 2
            assert [b.branch_id for b in legacy] == ["old-0-0", "old-0-1"]
        finally:
            root.close()
            mirror.close()
            source.close()


@pytest.mark.integration
def test_exact_search_counts_the_extra_outcome_branches_only_with_the_knob() -> None:
    def protect_both(ranked, config):
        keep = [e for e in ranked if e.order.message.count("move protect") >= 1][:2]
        keep += [e for e in ranked if "protect" not in e.order.message][:1]
        rest = [e for e in ranked if all(e is not k for k in keep)]
        return keep, rest

    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source, mirror, root = _turn_two_root(worker, "stall-search")
        try:
            metrics = {}
            for label, config in (("on", EXACT), ("off", LEGACY)):
                ranked = search_joint_orders_exact(
                    root, "p1", config, candidate_selector=protect_both, randomness_key="k"
                )
                searched = [e for e in ranked if e.breakdown.get("searched")]
                assert len(searched) == 3
                metrics[label] = (
                    ranked[0].breakdown["search_metrics"]["exchange_count"],
                    searched[0].breakdown["exact_stall_outcome_branches"],
                )
            # Searched orders that repeat Protect double their branches only with the knob.
            assert metrics["off"][1] == 0
            assert metrics["on"][1] > 0
            assert metrics["on"][0] > metrics["off"][0]
        finally:
            root.close()
            mirror.close()
            source.close()
