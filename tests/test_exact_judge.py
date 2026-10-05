"""Live exact-search judge (`vgc.exact_judge`): off is inert, on only ever re-ranks, errors
and timeouts keep the fast search's pick.

The exact engine is stubbed in the unit tests (its own contracts live in
test_public_search.py / test_exact_search.py); the `integration` tests drive the real
public mirror against the local Showdown checkout, no server needed.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vgc.agent as agent_module
import vgc.exact_judge as judge_module
from vgc.actions import describe_order
from vgc.agent import VgcPlayer
from vgc.clock import bind_deadline
from vgc.evaluator import ScoredOrder
from vgc.exact_judge import ExactJudge, judge_budget_s, select_judged
from vgc.models import PolicyConfig
from vgc.rl.agents import make_direct_agent
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.match import play_battle

TAGS = "abcdefgh"


def _order(tag: str) -> SimpleNamespace:
    single = SimpleNamespace(
        order=tag, mega=False, z_move=False, dynamax=False, terastallize=False, move_target=0
    )
    return SimpleNamespace(first_order=single, second_order=single, message=f"/choose move {tag}")


def _scored(searched: int = 6) -> list[ScoredOrder]:
    """Fast-search output, best first: ``searched`` searched entries, then the skipped tail."""

    rows = []
    for i, tag in enumerate(TAGS):
        is_searched = i < searched
        rows.append(
            ScoredOrder(
                _order(tag), 100.0 - 2 * i, {"searched": is_searched, "myopic_score": 50.0 - i}
            )
        )
    return rows


def _config(**overrides) -> PolicyConfig:
    return PolicyConfig(exact_judge_live=True, **overrides)


def _battle(**kw) -> SimpleNamespace:
    return SimpleNamespace(battle_tag="battle-x", turn=3, force_switch=False, **kw)


def _stub(values: dict[str, float]):
    return lambda battle, memory, team, cfg, wanted, metric: dict(values)


def test_off_runs_only_the_fast_search(monkeypatch) -> None:
    scored = _scored()
    monkeypatch.setattr(agent_module, "search_joint_orders", lambda battle, config, **kw: scored)

    def boom(*_a, **_k):
        raise AssertionError("the judge must not run when exact_judge_live is off")

    monkeypatch.setattr(ExactJudge, "rerank", boom)
    # Every other judge knob set, only the master switch off.
    config = replace(PolicyConfig(), exact_judge_top_k=2, exact_judge_budget_s=0.1)
    player = VgcPlayer.__new__(VgcPlayer)
    player.config = config
    assert player._search(_battle(), SimpleNamespace()) is scored
    assert player.exact_judge_log == []


def test_on_plays_the_stubbed_exact_best_and_it_is_one_of_the_legal_candidates(
    monkeypatch,
) -> None:
    scored = _scored()
    monkeypatch.setattr(agent_module, "search_joint_orders", lambda battle, config, **kw: scored)
    # "c" (fast rank 3) is exact-best; "g" is unsearched and outside top-K so is never judged.
    values = {_order(t).message: v for t, v in zip("abcdef", [10.0, 20.0, 90.0, 5.0, 0.0, 1.0])}
    values[_order("g").message] = 999.0
    monkeypatch.setattr(judge_module, "default_ranker", _stub(values))
    player = VgcPlayer.__new__(VgcPlayer)
    player.config = _config(exact_judge_top_k=6)
    player._own_packed_team = "packed-team"

    result = player._search(_battle(), SimpleNamespace())

    assert result[0].order is scored[2].order
    assert {e.order.message for e in result} == {e.order.message for e in scored}  # legal set
    assert result[0].order.message != _order("g").message  # an unjudged order cannot win
    assert result[0].breakdown["exact_judge"] is True
    assert [e.score for e in result] == sorted((e.score for e in result), reverse=True)
    record = player.exact_judge_log[0]
    assert record["status"] == "ok" and record["overturned"] is True
    assert record["fast_pick"] == describe_order(scored[0].order)
    assert record["exact_pick"] == describe_order(scored[2].order)


def test_extra_myopic_widens_the_judged_set_and_the_margin_guards_the_fast_pick() -> None:
    scored = _scored(searched=4)
    judged = select_judged(scored, top_k=2, extra_myopic=2)
    assert [e.order.message for e in judged] == [_order(t).message for t in "abef"]

    values = {_order("a").message: 50.0, _order("b").message: 58.0}
    for margin, expected in ((0.0, "b"), (10.0, "a")):  # a gain of 8 only clears margin 0
        judge = ExactJudge(
            _config(exact_judge_top_k=2, exact_judge_overturn_margin=margin),
            "team",
            ranker=_stub(values),
        )
        assert judge.rerank(_battle(), None, scored)[0].order.message == _order(expected).message


@pytest.mark.parametrize("failure", ["raises", "empty", "unscored_fast_pick", "timeout"])
def test_any_exact_failure_keeps_the_fast_pick(failure) -> None:
    scored = _scored()

    def ranker(battle, memory, team, cfg, wanted, metric):
        if failure == "raises":
            raise RuntimeError("worker died")
        if failure == "empty":
            return {}
        if failure == "timeout":
            time.sleep(1.5)
        return {_order("c").message: 99.0}  # the fast pick "a" has no exact value

    judge = ExactJudge(_config(exact_judge_budget_s=0.3), "team", ranker=ranker)
    started = time.monotonic()
    result = judge.rerank(_battle(), None, scored)
    assert result is scored  # the very same fast ranking, untouched
    assert judge.log[0]["status"] in ("error", "timeout")
    assert judge.log[0]["overturned"] is False
    if failure == "timeout":
        assert judge.log[0]["status"] == "timeout"
        assert time.monotonic() - started < 1.2  # did not wait for the abandoned search


def test_budget_follows_the_clock_guard_and_no_time_means_no_judge() -> None:
    config = _config(exact_judge_budget_s=3.0, exact_judge_margin_s=1.0)
    assert judge_budget_s(config) == 3.0  # no timer: the offline default

    def with_left(seconds: float) -> float:
        def run() -> float:
            bind_deadline(time.monotonic() + seconds)
            return judge_budget_s(config)

        return contextvars.copy_context().run(run)

    assert with_left(10.0) == pytest.approx(3.0, abs=0.05)
    assert with_left(2.5) == pytest.approx(1.5, abs=0.05)

    calls = []
    judge = ExactJudge(config, "team", ranker=lambda *a: calls.append(a) or {})

    def tight() -> list[ScoredOrder]:
        bind_deadline(time.monotonic() + 1.1)  # 1.1 s left - 1.0 s margin < the minimum
        return judge.rerank(_battle(), None, _scored())

    contextvars.copy_context().run(tight)
    assert judge.log[0]["status"] == "skipped_no_time" and calls == []


def test_forced_switches_and_missing_team_are_skipped_without_a_search() -> None:
    calls = []
    ranker = lambda *a: calls.append(a) or {}  # noqa: E731
    forced = ExactJudge(_config(), "team", ranker=ranker)
    forced.rerank(SimpleNamespace(battle_tag="t", turn=1, force_switch=[True, False]), None, _scored())
    no_team = ExactJudge(_config(), None, ranker=ranker)
    no_team.rerank(_battle(), None, _scored())
    assert [forced.log[0]["status"], no_team.log[0]["status"]] == [
        "skipped_forced_switch",
        "skipped_no_team",
    ]
    assert calls == []


# --- real battles through the public mirror ----------------------------------------------

TEAM_PATH = Path(__file__).resolve().parents[1] / "teams" / "meta1.packed.txt"


@pytest.fixture(scope="module")
def worker():
    if not (DEFAULT_SHOWDOWN_REPO / "dist" / "sim" / "index.js").exists():
        pytest.skip(f"no built showdown sim at {DEFAULT_SHOWDOWN_REPO}")
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as running:
        yield running


def _play(worker, config: PolicyConfig, seed: int) -> tuple[list[str], VgcPlayer, str | None]:
    """One seeded vgc mirror; returns every choice string sent, our player, the winner."""

    team = TEAM_PATH.read_text().strip()
    sent: list[str] = []
    agents = {
        "p1": make_direct_agent("vgc", team, config=config),
        "p2": make_direct_agent("vgc", team, config=PolicyConfig()),
    }
    agents["p1"].name, agents["p2"].name = "ours", "opp"
    for side, agent in agents.items():
        original = agent.choose
        agent.choose = lambda battle, _orig=original, _side=side: (  # noqa: E731
            sent.append(f"{_side}:{(msg := _orig(battle))}") or msg
        )
    outcome = play_battle(
        worker, f"judge-{seed}", agents, {"p1": team, "p2": team}, seed=[seed] * 4
    )
    return sent, agents["p1"].player, outcome.winner


@pytest.mark.integration
def test_knob_off_plays_the_identical_seeded_game(worker) -> None:
    default_choices, _, default_winner = _play(worker, PolicyConfig(), 11)
    perturbed = replace(
        PolicyConfig(), exact_judge_top_k=2, exact_judge_budget_s=0.05, exact_judge_metric="score"
    )
    off_choices, player, off_winner = _play(worker, perturbed, 11)
    assert off_choices == default_choices and off_winner == default_winner
    assert player.exact_judge_log == []


@pytest.mark.integration
def test_knob_on_completes_a_real_game_with_exact_rankings_and_no_fallback(worker) -> None:
    choices, player, winner = _play(worker, _config(), 11)
    assert winner in {"p1", "p2"}  # every judged choice was legal: an illegal one raises
    assert player.fallback_count == 0
    ok = [r for r in player.exact_judge_log if r["status"] == "ok"]
    assert ok, player.exact_judge_log
    for record in ok:
        assert record["judged"] >= 2 and record["elapsed_ms"] > 0
        assert record["exact_pick"] in {row["order"] for row in record["ranking"]}
