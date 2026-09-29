#!/usr/bin/env python
"""Per-stage decision-time breakdown on live local games (diagnostic, no decision changes).

Sibling of `measure_search_latency.py` (which only times whole decisions). This script
wraps the existing pipeline functions from the OUTSIDE (monkeypatching module attributes
for the duration of the run) and attributes each decision's wall time to stages using a
stack, so every stage is reported as EXCLUSIVE (self) time and the stages plus
`unattributed` add up to the decision total. Nothing in src/ is edited and tracing
(`VGC_TRACE`) is left off, so no trace bookkeeping is added to the timings; the search
work counters come from the `search_metrics` dicts the search already returns.

Paths:
  default  -- shipped `vgc.search.search_joint_orders` (2-ply search + rolling horizon).
  hybrid   -- `--checkpoint X.pt`: `vgc.rl.search_guidance.NeuralSearchPlayer` in hybrid
              mode (neural shortlist; mechanics-aware checkpoints route through the exact
              Showdown mirror, see `vgc.rl.public_search`).

Server must already be running (see CLAUDE.md). Example:

    .venv/bin/python offline/measure_decision_stages.py --n 5 \
        --team teams/meta1.packed.txt --out runs/eval/decision_stage_latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.agent import VgcPlayer  # noqa: E402
from vgc.baselines import make_player  # noqa: E402
from vgc.config import FORMAT_ID, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402

# (module, attribute path, stage name). Attribute path may be "Class.method".
# Stage names are what appears in the report; `search_*` are the default 2-ply path,
# `exact_*` the Showdown-mirror path, `preview_*` team preview.
STAGES: list[tuple[str, str, str]] = [
    # --- default 2-ply search (vgc.search) ---
    ("vgc.search", "search_joint_orders", "search_total"),
    ("vgc.agent", "search_joint_orders", "search_total"),  # agent imported the name directly
    ("vgc.rl.search_guidance", "search_joint_orders", "search_total"),
    ("vgc.search", "score_joint_orders", "search_myopic_score"),
    ("vgc.search", "belief_ordered_candidates", "search_belief_reorder"),
    ("vgc.search", "build_context", "search_build_context"),
    ("vgc.search", "_enumerate_opp_responses", "search_opp_response_enum"),
    ("vgc.search", "_select_search_candidates", "search_select_candidates"),
    ("vgc.search", "resolve_exchange", "search_resolve_exchange"),
    ("vgc.search", "forecast_position", "search_rolling_horizon_forecast"),
    ("vgc.search", "_aggregate_exchange_values", "search_aggregate"),
    # --- exact Showdown-mirror search (hybrid checkpoints with mechanics features) ---
    ("vgc.rl.public_search", "public_information_exact_search", "exact_total"),
    ("vgc.rl.search_guidance", "public_information_exact_search", "exact_total"),
    ("vgc.rl.public_search", "search_joint_orders_exact", "exact_search_fn"),
    ("vgc.rl.public_search", "combine_belief_rankings", "exact_combine_beliefs"),
    ("vgc.rl.public_search", "LiveExactMirror", "exact_mirror_start"),
    ("vgc.rl.live_mirror", "LiveExactMirror.hypotheses", "exact_hypotheses"),
    ("vgc.rl.live_mirror", "LiveExactMirror.build", "exact_mirror_build_root"),
    ("vgc.rl.live_mirror", "LiveExactMirror.rebase", "exact_mirror_rebase_root"),
    ("vgc.rl.live_mirror", "LiveExactMirror.close", "exact_mirror_close"),
    ("vgc.rl.exact_search", "score_joint_orders", "exact_myopic_score"),
    ("vgc.rl.exact_search", "belief_ordered_candidates", "exact_belief_reorder"),
    ("vgc.rl.exact_search", "_select_search_candidates", "exact_select_candidates"),
    ("vgc.rl.exact_search", "evaluate_exact_branches", "exact_showdown_branches"),
    ("vgc.rl.exact_search", "snapshot_battle", "exact_snapshot"),
    # --- neural shortlist ---
    ("vgc.rl.search_guidance", "rank_legal_orders", "neural_ranking"),
    ("vgc.rl.search_guidance", "select_neural_guided_candidates", "neural_select"),
    ("vgc.rl.search_guidance", "_select_search_candidates", "search_select_candidates"),
    # --- team preview ---
    ("vgc.agent", "build_team_order", "preview_build_team_order"),
    ("vgc.team_preview", "_build_matchup_matrix", "preview_matchup_matrix"),
    ("vgc.team_preview", "_score_choice", "preview_score_choice"),
]
# Stages that are only wrapped to show up as their own line; "*_total" style wrappers are
# containers -- exclusive time makes their children not double count.


class _Decision:
    __slots__ = ("excl", "incl", "calls", "stack", "counters")

    def __init__(self) -> None:
        self.excl: dict[str, float] = defaultdict(float)
        self.incl: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.stack: list[list] = []  # [name, start, child_time]
        self.counters: dict[str, float] = {}


_active: _Decision | None = None


def _wrap(fn, stage: str):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        d = _active
        if d is None:
            return fn(*args, **kwargs)
        frame = [stage, time.perf_counter(), 0.0]
        d.stack.append(frame)
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - frame[1]
            d.stack.pop()
            d.excl[stage] += elapsed - frame[2]
            d.incl[stage] += elapsed
            d.calls[stage] += 1
            if d.stack:
                d.stack[-1][2] += elapsed
        _capture_metrics(d, stage, result)
        return result

    wrapper.__wrapped_stage__ = stage  # type: ignore[attr-defined]
    return wrapper


def _capture_metrics(d: _Decision, stage: str, result) -> None:
    """Pull existing work counters out of the search's own `search_metrics` dict."""

    if stage not in ("search_total", "exact_search_fn") or not result:
        return
    metrics = result[0].breakdown.get("search_metrics") or {}
    for key in (
        "legal_actions",
        "searched_actions",
        "opponent_responses",
        "exchange_count",
        "forecast_count",
    ):
        if key in metrics:
            d.counters[key] = d.counters.get(key, 0) + metrics[key]
    if "elapsed_ms" in metrics:
        d.counters["search_elapsed_ms_reported"] = (
            d.counters.get("search_elapsed_ms_reported", 0.0) + metrics["elapsed_ms"]
        )


def install_patches() -> list[str]:
    installed: list[str] = []
    for module_name, path, stage in STAGES:
        try:
            module = importlib.import_module(module_name)
            if "." in path:
                cls_name, attr = path.split(".")
                owner = getattr(module, cls_name)
            else:
                owner, attr = module, path
            original = getattr(owner, attr)
        except (ImportError, AttributeError):
            continue
        if isinstance(original, type):
            # Constructor of a class: wrap __init__ so instances are still that class.
            init = original.__init__
            original.__init__ = _wrap(init, stage)  # type: ignore[method-assign]
        else:
            setattr(owner, attr, _wrap(original, stage))
        installed.append(stage)
    return installed


class TimingMixin:
    """Times whole choose_move()/teampreview() calls and attributes them to stages."""

    def _init_timing(self) -> None:
        self.stage_records: list[dict[str, object]] = []

    def _timed(self, kind: str, call, battle):
        global _active
        d = _Decision()
        _active = d
        started = time.perf_counter()
        n_fallbacks = self.fallback_count
        try:
            return call(battle)
        finally:
            total = time.perf_counter() - started
            _active = None
            self.stage_records.append(
                {
                    "kind": kind,
                    "turn": int(getattr(battle, "turn", 0) or 0),
                    "total_s": total,
                    "excl_s": dict(d.excl),
                    "incl_s": dict(d.incl),
                    "calls": dict(d.calls),
                    "counters": dict(d.counters),
                    "fallback": self.fallback_count > n_fallbacks,
                }
            )

    def choose_move(self, battle):
        return self._timed("move", super().choose_move, battle)

    def teampreview(self, battle):
        return self._timed("preview", super().teampreview, battle)


class TimedDefaultPlayer(TimingMixin, VgcPlayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_timing()


def _make_hybrid_class():
    from vgc.rl.search_guidance import NeuralSearchPlayer

    class TimedHybridPlayer(TimingMixin, NeuralSearchPlayer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._init_timing()

    return TimedHybridPlayer


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(pct * (len(s) - 1))))]


def _dist(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "p50": _pct(values, 0.50),
        "p90": _pct(values, 0.90),
        "p99": _pct(values, 0.99),
        "max": max(values) if values else 0.0,
    }


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"decisions": 0}
    totals = [r["total_s"] * 1000 for r in records]  # type: ignore[operator]
    grand_total = sum(totals)
    names = sorted({k for r in records for k in r["excl_s"]})  # type: ignore[union-attr]
    stages: dict[str, object] = {}
    attributed = [0.0] * len(records)
    for name in names:
        per = [r["excl_s"].get(name, 0.0) * 1000 for r in records]  # type: ignore[union-attr]
        incl = [r["incl_s"].get(name, 0.0) * 1000 for r in records]  # type: ignore[union-attr]
        for i, v in enumerate(per):
            attributed[i] += v
        stages[name] = {
            "exclusive_ms": _dist(per),
            "inclusive_ms": _dist(incl),
            "share_of_total": sum(per) / grand_total if grand_total else 0.0,
            "calls_per_decision_mean": sum(r["calls"].get(name, 0) for r in records)  # type: ignore[union-attr]
            / len(records),
            "decisions_with_stage": sum(1 for v in per if v > 0),
        }
    unattributed = [max(0.0, t - a) for t, a in zip(totals, attributed, strict=True)]
    stages["unattributed"] = {
        "exclusive_ms": _dist(unattributed),
        "share_of_total": sum(unattributed) / grand_total if grand_total else 0.0,
    }
    counter_names = sorted({k for r in records for k in r["counters"]})  # type: ignore[union-attr]
    counters = {
        k: _dist([r["counters"].get(k, 0) for r in records])  # type: ignore[union-attr]
        for k in counter_names
    }
    return {
        "decisions": len(records),
        "fallbacks": sum(1 for r in records if r["fallback"]),
        "total_ms": _dist(totals),
        "total_ms_sum": grand_total,
        "stages": stages,
        "work_counters": counters,
    }


def _table(title: str, summary: dict[str, object]) -> str:
    if not summary.get("decisions"):
        return f"{title}: no decisions"
    lines = [
        f"{title}: n={summary['decisions']} fallbacks={summary['fallbacks']}",
        f"  {'stage (exclusive ms)':34s}{'p50':>9s}{'p90':>9s}{'p99':>9s}{'max':>9s}{'share':>8s}{'calls':>8s}",
    ]
    t = summary["total_ms"]
    lines.append(
        f"  {'TOTAL':34s}{t['p50']:9.1f}{t['p90']:9.1f}{t['p99']:9.1f}{t['max']:9.1f}{'100.0%':>8s}"  # type: ignore[index]
    )
    stages = summary["stages"]
    for name, s in sorted(stages.items(), key=lambda kv: -kv[1]["share_of_total"]):  # type: ignore[union-attr]
        e = s["exclusive_ms"]
        lines.append(
            f"  {name:34s}{e['p50']:9.1f}{e['p90']:9.1f}{e['p99']:9.1f}{e['max']:9.1f}"
            f"{s['share_of_total'] * 100:7.1f}%{s.get('calls_per_decision_mean', 0):8.1f}"
        )
    for k, v in summary["work_counters"].items():  # type: ignore[union-attr]
        lines.append(f"  counter {k}: mean {v['mean']:.1f} p50 {v['p50']:.0f} max {v['max']:.0f}")
    return "\n".join(lines)


async def _play(
    n: int, team: str, opponent: str, checkpoint: Path | None, device: str
) -> tuple[object, dict[str, object]]:
    config = PolicyConfig(format_id=FORMAT_ID)
    assert config.use_two_ply_search and config.use_rolling_horizon
    if checkpoint is None:
        p1 = TimedDefaultPlayer(config=config, team=team, battle_format=FORMAT_ID)
    else:
        from vgc.rl.opponents import load_snapshot

        model = load_snapshot(checkpoint, device=device)
        p1 = _make_hybrid_class()(
            model=model,
            checkpoint_path=checkpoint,
            mode="hybrid",
            device=device,
            config=config,
            team=team,
            battle_format=FORMAT_ID,
        )
    p2 = make_player(opponent, team, FORMAT_ID)
    started = time.perf_counter()
    try:
        await asyncio.wait_for(p1.battle_against(p2, n_battles=n), timeout=120 * n)
    finally:
        await p1.ps_client.stop_listening()
        await p2.ps_client.stop_listening()
    meta = {
        "games_requested": n,
        "games_finished": p1.n_finished_battles,
        "wins": p1.n_won_battles,
        "fallbacks": p1.fallback_count,
        "wall_s": time.perf_counter() - started,
    }
    return p1, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--team", default=str(TEAMS_DIR / "meta1.packed.txt"))
    parser.add_argument("--opponent", default="heuristic")
    parser.add_argument("--checkpoint", default=None, help="enable hybrid neural-shortlist path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="runs/eval/decision_stage_latency.json")
    args = parser.parse_args()

    team = Path(args.team).read_text().strip()
    installed = install_patches()
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    p1, meta = asyncio.run(_play(args.n, team, args.opponent, checkpoint, args.device))
    records = p1.stage_records  # type: ignore[attr-defined]
    moves = [r for r in records if r["kind"] == "move"]
    previews = [r for r in records if r["kind"] == "preview"]
    report = {
        "path": "hybrid" if checkpoint else "default",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "opponent": args.opponent,
        "team_file": args.team,
        **meta,
        "instrumented_stages": installed,
        "move_decisions": summarize(moves),
        "team_preview": summarize(previews),
        "notes": [
            "Stage times are EXCLUSIVE (self) time from a call stack; stages + unattributed = total.",
            "unattributed = time inside choose_move()/teampreview() outside any wrapped function "
            "(BattleMemory, replay recorder, order construction, glue).",
            "Nothing measures time outside choose_move() (poke-env parsing, websocket, sim server).",
        ],
    }
    print(_table("MOVE DECISIONS", report["move_decisions"]))
    print(_table("TEAM PREVIEW", report["team_preview"]))
    print(json.dumps({k: v for k, v in meta.items()}, indent=2))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    return 0 if meta["games_finished"] == args.n and meta["fallbacks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
