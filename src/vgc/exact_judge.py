"""Live exact-search judge: re-rank the fast search's best candidates with Showdown.

The shipped decision path (`vgc.search.search_joint_orders`, ~50 ms) scores each candidate
with a Python approximation of the next turn.  Exact grading (`offline/grade_positions.py`)
found a better move among the candidates that search ranked lower or skipped on a sizeable
share of positions.  This module lets the exact engine act as the JUDGE over the fast
search's shortlist, inside the per-turn clock budget:

1. the fast search runs exactly as before and produces its ranking;
2. the top ``exact_judge_top_k`` entries (plus ``exact_judge_extra_myopic`` skipped ones)
   are handed to `vgc.rl.public_search.public_information_exact_search` -- the same public
   reconstruction `NeuralSearchPlayer` and the offline grader use.  It sees only what a
   real player sees plus our own packed team; it never touches a private simulator root;
3. the exact-best order is played if it beats the fast pick by more than
   ``exact_judge_overturn_margin`` on ``exact_judge_metric``.

Every failure (no team, no time, a mirror error, an exact search that outruns its budget)
keeps the fast search's pick.  The exact search runs on a helper thread that owns its own
temporary Showdown worker; on timeout the thread is abandoned (it cleans up after itself)
and at most one abandoned search is ever left running per player.

``PolicyConfig.exact_judge_live`` defaults to False, in which case none of this runs.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from vgc import plan_value, speed_payoff
from vgc.actions import describe_order
from vgc.clock import cancelled, time_left
from vgc.decision_trace import record_note
from vgc.evaluator import ScoredOrder
from vgc.models import PolicyConfig

# Below this the exact search (mirror start-up alone is ~0.1-0.3 s) cannot finish usefully.
MIN_BUDGET_S = 0.25
METRICS = ("exchange_value", "score")

# (battle, memory, own packed team, judge config, wanted order messages, metric)
#   -> {order message: exact value}
Ranker = Callable[[Any, Any, str, PolicyConfig, "frozenset[str]", str], Mapping[str, float]]


# --- pure helpers -------------------------------------------------------------------------


def select_judged(
    scored: Sequence[ScoredOrder], top_k: int, extra_myopic: int
) -> list[ScoredOrder]:
    """The fast search's top-K searched entries, then its best-myopic skipped ones.

    ``scored`` is the fast search's output (best first; searched entries precede the
    unsearched tail).  The fast order is preserved so ties resolve towards the fast pick.
    """

    searched = [e for e in scored if e.breakdown.get("searched", True) is not False]
    skipped = [e for e in scored if e.breakdown.get("searched", True) is False]
    judged = searched[: max(1, int(top_k))]
    if extra_myopic > 0:
        skipped.sort(key=lambda e: -float(e.breakdown.get("myopic_score", e.score)))
        judged += skipped[: int(extra_myopic)]
    # LLM proposals (vgc.llm.proposer) are always judged, wherever the fast search
    # ranked them: the exact judge, not the fast search, decides whether they are better.
    for entry in scored:
        if entry.breakdown.get("llm_proposed") and all(entry is not j for j in judged):
            judged.append(entry)
    return judged


def judge_config(config: PolicyConfig, n_candidates: int) -> PolicyConfig:
    """``config`` with the exact search sized for the judge (`exact_judge_*` widths)."""

    hypotheses = max(1, int(config.exact_judge_hypotheses))
    return replace(
        config,
        search_our_candidates=max(1, n_candidates),
        search_opp_candidates=config.exact_judge_opp_candidates,
        exact_search_future_samples=config.exact_judge_future_samples,
        exact_search_state_hypotheses=hypotheses,
        exact_search_spread_hypotheses=hypotheses,
        exact_search_set_hypotheses=hypotheses,
        exact_search_bring_hypotheses=hypotheses,
        exact_search_total_hypotheses=hypotheses,
    )


def with_exact_timers(judge_cfg: PolicyConfig, battle: Any, config: PolicyConfig) -> PolicyConfig:
    """``judge_cfg`` widened so every hidden-timer branch of ``battle`` is searched.

    The judge runs with one hypothesis, which kept only the most likely remaining sleep
    (3 action opportunities, p=2/3) and discarded the shorter one (p=1/3), so a real wake
    chance read as zero. Timers are the only axis widened (spread/set/bring stay at one
    hypothesis), so the root count is just the number of consistent timer assignments,
    capped by ``exact_search_state_hypotheses``. Positions with nobody asleep or confused
    are unchanged. Costs one extra search per extra branch on those positions only.
    """

    if not config.exact_judge_exact_timers:
        return judge_cfg
    from vgc.mechanics_state import snapshot_battle
    from vgc.rl.hidden_state import enumerate_hidden_state_hypotheses

    cap = max(1, int(config.exact_search_state_hypotheses))
    branches = len(
        enumerate_hidden_state_hypotheses(
            snapshot_battle(battle), replace(config, exact_search_state_hypotheses=cap)
        )
    )
    if branches <= 1:
        return judge_cfg
    return replace(
        judge_cfg,
        exact_search_state_hypotheses=cap,
        exact_search_total_hypotheses=max(judge_cfg.exact_search_total_hypotheses, branches),
    )


def judge_budget_s(config: PolicyConfig) -> float:
    """Seconds this decision may spend judging: the cap, or what the clock guard leaves."""

    budget = float(config.exact_judge_budget_s)
    left = time_left()
    if left is not None:
        budget = min(budget, left - float(config.exact_judge_margin_s))
    return budget


def exact_values(rankings: Sequence[tuple[float, list[ScoredOrder]]], metric: str) -> dict[str, float]:
    """Belief-weighted exact value per searched order message.

    ``exchange_value`` is averaged across the per-belief rankings (not read off the
    combined ranking, whose breakdown carries only the most likely belief's value).
    """

    if metric not in METRICS:
        raise ValueError(f"exact_judge_metric must be one of {METRICS}, got {metric!r}")
    total = sum(weight for weight, _ranking in rankings)
    if total <= 0.0:
        raise ValueError("belief weights must sum to a positive number")
    values: dict[str, float] = defaultdict(float)
    seen: dict[str, int] = defaultdict(int)
    for weight, ranking in rankings:
        for entry in ranking:
            if not entry.breakdown.get("searched"):
                continue
            value = (
                float(entry.breakdown["exchange_value"])
                if metric == "exchange_value"
                else float(entry.score)
            )
            values[entry.order.message] += weight * value / total
            seen[entry.order.message] += 1
    return {key: value for key, value in values.items() if seen[key] == len(rankings)}


def rerank(
    scored: list[ScoredOrder],
    judged: Sequence[ScoredOrder],
    values: Mapping[str, float],
    margin: float,
) -> tuple[list[ScoredOrder], dict[str, Any]]:
    """Put the exact-best judged order first when it clearly beats the fast pick.

    Returns the (possibly re-ordered) list and a report fragment.  The fast pick must have
    an exact value; if it does not, the comparison is meaningless and ``ValueError`` makes
    the caller keep the fast pick.  With no overturn the SAME list object comes back.
    """

    fast = scored[0]
    if fast.order.message not in values:
        raise ValueError("the exact search did not score the fast search's own pick")
    scorable = [e for e in judged if e.order.message in values]
    best = max(scorable, key=lambda e: values[e.order.message])  # first max = fast order wins ties
    gain = values[best.order.message] - values[fast.order.message]
    overturned = best is not fast and gain > margin
    fragment: dict[str, Any] = {
        "judged": len(scorable),
        "fast_pick": describe_order(fast.order),
        "exact_pick": describe_order(best.order if overturned else fast.order),
        "overturned": overturned,
        "gain": round(gain, 3),
        "ranking": [
            {
                "order": describe_order(e.order),
                "value": round(values[e.order.message], 3),
                "fast_rank": next(i + 1 for i, s in enumerate(scored) if s is e),
            }
            for e in sorted(scorable, key=lambda e: -values[e.order.message])
        ],
    }
    if not overturned:
        return scored, fragment
    breakdown = dict(best.breakdown)
    breakdown.update(
        exact_judge=True,
        exact_judge_value=values[best.order.message],
        exact_judge_overturned=describe_order(fast.order),
    )
    # Keep the list sorted best-first for downstream score blending (BC policy).
    top = ScoredOrder(order=best.order, score=max(best.score, fast.score), breakdown=breakdown)
    return [top] + [e for e in scored if e is not best], fragment


# --- the exact backend --------------------------------------------------------------------


def default_ranker(
    battle: Any,
    memory: Any,
    own_packed_team: str,
    config: PolicyConfig,
    wanted: frozenset[str],
    metric: str,
) -> dict[str, float]:
    """Exact values for exactly the ``wanted`` orders, via the public mirror."""

    from vgc.rl.public_search import public_information_exact_search

    def selector(ranked, _config):
        searched = [e for e in ranked if e.order.message in wanted]
        return searched, [e for e in ranked if e.order.message not in wanted]

    rankings: list[tuple[float, list[ScoredOrder]]] = []
    public_information_exact_search(
        battle,
        config,
        own_packed_team,
        memory=memory,
        candidate_selector=selector,
        rankings_out=rankings,
    )
    return exact_values(rankings, metric)


def _is_forced_switch(battle: Any) -> bool:
    forced = getattr(battle, "force_switch", False)
    return bool(any(forced) if isinstance(forced, (list, tuple)) else forced)


# --- the judge ----------------------------------------------------------------------------


class ExactJudge:
    """Per-player judge: owns the in-memory decision log and the one-abandoned-search rule."""

    def __init__(
        self,
        config: PolicyConfig,
        own_packed_team: str | None,
        ranker: Ranker | None = None,
    ) -> None:
        self.config = config
        self.own_packed_team = own_packed_team
        if config.exact_search_field_measured_plan:
            plan_value.register_own_team(own_packed_team)
        if config.exact_search_field_measured_speed:
            speed_payoff.register_own_team(own_packed_team)
        self._ranker = ranker
        self._inflight: threading.Thread | None = None
        self.log: list[dict[str, Any]] = []

    def rerank(self, battle: Any, memory: Any, scored: list[ScoredOrder]) -> list[ScoredOrder]:
        """``scored`` with the exact-best judged order first, or ``scored`` unchanged."""

        started = time.perf_counter()
        report: dict[str, Any] = {
            "battle_tag": getattr(battle, "battle_tag", None),
            "turn": getattr(battle, "turn", None),
            "status": "ok",
            "budget_s": None,
            "judged": 0,
            "overturned": False,
            "fast_pick": describe_order(scored[0].order) if scored else None,
            "exact_pick": None,
            "gain": None,
            "ranking": [],
            "error": None,
        }
        result = scored
        try:
            result = self._rerank(battle, memory, scored, report)
        except Exception as exc:  # noqa: BLE001 - the fast search's pick is always the fallback
            report["status"] = "error"
            report["error"] = f"{type(exc).__name__}: {exc}"
            result = scored
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        self._record(report)
        return result

    def _rerank(
        self, battle: Any, memory: Any, scored: list[ScoredOrder], report: dict[str, Any]
    ) -> list[ScoredOrder]:
        config = self.config
        if config.exact_judge_metric not in METRICS:
            raise ValueError(f"exact_judge_metric must be one of {METRICS}")
        if _is_forced_switch(battle):
            report["status"] = "skipped_forced_switch"
            return scored
        judged = select_judged(
            scored, config.exact_judge_top_k, config.exact_judge_extra_myopic
        )
        if len(judged) < 2:
            report["status"] = "skipped_few_candidates"
            return scored
        if not self.own_packed_team:
            report["status"] = "skipped_no_team"
            return scored
        if cancelled():
            report["status"] = "skipped_cancelled"
            return scored
        budget = judge_budget_s(config)
        report["budget_s"] = round(budget, 2)
        if budget < MIN_BUDGET_S:
            report["status"] = "skipped_no_time"
            return scored
        if self._inflight is not None and self._inflight.is_alive():
            report["status"] = "skipped_busy"  # an abandoned search is still winding down
            return scored

        wanted = frozenset(e.order.message for e in judged)
        values = self._run_bounded(
            battle,
            memory,
            with_exact_timers(judge_config(config, len(wanted)), battle, config),
            wanted,
            budget,
            report,
        )
        if values is None:
            return scored
        new_scored, fragment = rerank(
            scored, judged, values, float(config.exact_judge_overturn_margin)
        )
        report.update(fragment)
        return new_scored

    def _run_bounded(
        self,
        battle: Any,
        memory: Any,
        judge_cfg: PolicyConfig,
        wanted: frozenset[str],
        budget: float,
        report: dict[str, Any],
    ) -> Mapping[str, float] | None:
        """Run the exact search on a helper thread; None (and a status) on timeout/error."""

        ranker = self._ranker or default_ranker
        box: dict[str, Any] = {}
        done = threading.Event()
        metric = self.config.exact_judge_metric
        team = self.own_packed_team or ""

        def work() -> None:
            try:
                box["values"] = ranker(battle, memory, team, judge_cfg, wanted, metric)
            except BaseException as exc:  # noqa: BLE001 - reported to the caller
                box["error"] = exc
            finally:
                done.set()

        # A fresh empty context: the helper must never write into the decision trace or
        # inherit the guard's cancel flag/deadline (it may outlive both).
        ctx = contextvars.Context()
        thread = threading.Thread(
            target=lambda: ctx.run(work), name="vgc-exact-judge", daemon=True
        )
        self._inflight = thread
        thread.start()
        if not done.wait(budget):
            report["status"] = "timeout"
            return None
        if "error" in box:
            err = box["error"]
            report["status"] = "error"
            report["error"] = f"{type(err).__name__}: {err}"
            report["traceback"] = "".join(traceback.format_exception(err))[-4000:]
            return None
        values = box["values"]
        if not values:
            report["status"] = "error"
            report["error"] = "empty exact ranking"
            return None
        return values

    def _record(self, report: dict[str, Any]) -> None:
        self.log.append(report)
        record_note("exact_judge", report)
        path = self.config.exact_judge_log_path
        if not path:
            return
        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a") as handle:
                handle.write(json.dumps(report, sort_keys=True, default=str) + "\n")
        except OSError:
            pass  # logging must never affect a decision
