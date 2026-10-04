#!/usr/bin/env python
"""Proposer screen, step 1: ask the LLM for moves on saved positions, plus a fair control.

Question this feeds (see ``offline/grade_positions.py --compare``): on saved real positions,
does GPT-6 Luna propose good moves that the engine's live search did NOT already consider,
more often than a control that adds the same NUMBER of the engine's own next-ranked
candidates?  Both arms get extra orders the shipped search did not score, so any difference
is about WHICH orders were added, not how many.

For each position written by ``offline/record_positions.py`` this

1. rebuilds the public battle from the saved player-view bundle (``vgc.positions``),
2. recomputes the engine's candidate list exactly as live play does:
   ``score_joint_orders`` -> ``belief_ordered_candidates`` -> ``_select_search_candidates``
   (the "searched set" is what the shortlist keeps; the rest is the unsearched tail).  The
   searched set used to define "novel" is that recomputed set UNION the set recorded at game
   time (they differ at the shortlist edge, because the offline game's engine saw an enriched
   opponent while the rebuild sees the public view),
3. builds the packet with ``vgc.llm.facts.build_packet`` (same call the live proposer makes,
   our team plan from ``teams/owner/plans`` via ``load_team_plan``), and calls
   ``vgc.llm.harness.advise`` at ``--level`` with the real client (or ``--fake``),
4. maps the option IDs back to canonical ``describe_order`` strings and keeps the NOVEL ones
   (not in the searched set),
5. writes the control: the first ``n`` unsearched orders in myopic-rank order, with ``n`` =
   the number of Luna's novel proposals clamped to 1..3.

Outputs, appended next to the positions (one row per position, resumable)::

    <positions_dir>/proposals_luna[_<tag>].jsonl
    <positions_dir>/proposals_control[_<tag>].jsonl

Row: ``{position_id, source, level, orders, novel_orders, searched_orders, call: {...}}``.
``orders`` are canonical (``"rockslide / earthquake"``) and gradeable with
``offline/grade_positions.py`` (``--proposals`` or ``--compare``).

Money: the real client spends against its OWN spend file, ``runs/llm/screen_spend.json``,
capped by ``--max-usd`` (default 0.25), never the live-play one.  When the cap would be
exceeded the run stops cleanly; rerun the same command to continue (finished positions are
skipped).  ``--fake SCENARIO`` uses ``FakeLLMClient`` ($0, no network); the extra scenario
``novel`` (script-local, smoke only) proposes unsearched options so the grader has novel
orders to score.

    PYTHONPATH=src .venv/bin/python offline/propose_positions.py runs/positions/<run> \\
        --split tune --level none --max-usd 0.25
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.actions import describe_order  # noqa: E402
from vgc.belief_scoring import belief_ordered_candidates  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR  # noqa: E402
from vgc.evaluator import score_joint_orders  # noqa: E402
from vgc.llm.client import SCENARIOS, FakeLLMClient, OpenAIResponsesClient, _answer  # noqa: E402
from vgc.llm.config import LEVELS, LLMConfig  # noqa: E402
from vgc.llm.facts import build_packet, load_team_plan  # noqa: E402
from vgc.llm.harness import advise  # noqa: E402
from vgc.llm.packet import new_request_id  # noqa: E402
from vgc.llm.spend import SpendMeter, shared_meter  # noqa: E402
from vgc.llm.types import RawResult  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.positions import load_bundle, rebuild_position  # noqa: E402
from vgc.search import _select_search_candidates  # noqa: E402

LLM_DIR = RUNS_DIR / "llm"
CONTROL_MIN, CONTROL_MAX = 1, 3


class NovelFakeClient(FakeLLMClient):
    """Smoke-only fake: answers with options OUTSIDE the searched set (``self.ids``)."""

    def __init__(self) -> None:
        super().__init__("valid")
        self.ids: list[str] = []

    def complete(self, packet, level, max_output_tokens, timeout_s):  # noqa: ANN001
        self.calls.append(packet)
        return RawResult(
            text=_answer(self.ids or list(packet.option_ids[:1])),
            input_tokens=len(packet.full_text) // 4,
            output_tokens=80,
            request_id=packet.request_id,
        )


def engine_config() -> PolicyConfig:
    """The config the positions were recorded under (``record_positions.py``)."""
    return PolicyConfig(format_id=FORMAT_ID)


def candidate_split(battle, memory, config: PolicyConfig):
    """``(ranked, searched, unsearched)`` exactly as ``search_joint_orders`` builds them."""
    myopic = score_joint_orders(battle, config)
    ranked = belief_ordered_candidates(battle, myopic, config, memory=memory)
    searched, unsearched = _select_search_candidates(ranked, config)
    return ranked, searched, unsearched


def _read_calls(log_path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not log_path.exists():
        return [], offset
    with log_path.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    rows = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows, offset + len(data)


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            out.add(str(json.loads(line)["position_id"]))
    return out


def propose_one(
    position: dict[str, Any],
    run_dir: Path,
    *,
    client,
    meter: SpendMeter,
    level: str,
    budget_s: float,
    log_path: Path,
    llm_cfg: LLMConfig,
    config: PolicyConfig,
    skip_clear_gap: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Return ``(luna_row, control_row, hit_spend_cap)`` for one position."""

    started = time.perf_counter()
    bundle = load_bundle(position, run_dir)
    battle, memory = rebuild_position(bundle, int(position["decision_index"]))
    ranked, searched, unsearched = candidate_split(battle, memory, config)
    searched_keys = [describe_order(e.order) for e in searched]
    recorded = position.get("engine_searched")
    # "Already considered" = scored by the shortlist on this public rebuild OR scored by the
    # engine at game time.  The two can differ at the shortlist's edge: the offline game's
    # engine saw an enriched opponent, the rebuild sees what ladder play sees.  A novel order
    # must be neither, so a proposal can never get credit for something the engine scored.
    searched_set = set(searched_keys) | set(recorded or [])
    unsearched = [e for e in unsearched if describe_order(e.order) not in searched_set]
    chosen_ok = position["engine_chosen"] in searched_keys or any(
        position.get("engine_chosen_wire") == e.order.message for e in searched
    )
    rebuild_s = time.perf_counter() - started

    by_text = {str(e.order.message): e for e in ranked}
    turn = int(getattr(battle, "turn", 0) or 0)
    packet, options = build_packet(
        battle,
        ranked,
        llm_config=llm_cfg,
        team_plan=load_team_plan(position["our_team"]),
        memory=memory,
        policy_config=config,
        request_id=new_request_id(turn),
    )
    option_order = {o.id: o.order for o in options}
    gap = float(ranked[0].score) - float(ranked[1].score) if len(ranked) > 1 else 0.0

    if isinstance(client, NovelFakeClient):
        # Smoke only: skip the 3 best unsearched options, then take two (deliberately not
        # the control's picks, so the paired comparison has something to compare).
        novel_ids = [
            o.id
            for o in options
            if o.order in by_text and describe_order(by_text[o.order].order) not in searched_set
        ]
        client.ids = novel_ids[3:5] or novel_ids[:2]

    offset = log_path.stat().st_size if log_path.exists() else 0
    status, call = "skipped_clear_gap", {"request_id": packet.request_id}
    orders: list[str] = []
    hit_cap = False
    if not unsearched:
        # Every legal order was already searched: nothing novel exists, so no paid call.
        status = "skipped_no_unsearched"
    elif not (skip_clear_gap and gap >= llm_cfg.clear_gap):
        t0 = time.perf_counter()
        advice = advise(packet, client, level, budget_s, meter, log_path, config=llm_cfg)
        wall = time.perf_counter() - t0
        records, _ = _read_calls(log_path, offset)
        mine = [r for r in records if r.get("request_id") == packet.request_id]
        last = mine[-1] if mine else {}
        status = str(last.get("status", "no_record"))
        hit_cap = any(r.get("status") == "spend_cap" for r in mine)
        call = {
            "request_id": packet.request_id,
            "status": status,
            "attempts": len(mine),
            "cost_usd": round(sum(float(r.get("cost_usd", 0.0)) for r in mine), 6),
            "latency_s": round(wall, 3),
            "input_tokens": int(last.get("input_tokens", 0)),
            "output_tokens": int(last.get("output_tokens", 0)),
            "invalid_ids": int(last.get("invalid_ids", 0)),
        }
        if advice is not None:
            for proposal in advice.proposals:
                entry = by_text.get(option_order.get(proposal.id, ""))
                if entry is None:
                    continue
                key = describe_order(entry.order)
                if key not in orders:
                    orders.append(key)
    novel = [o for o in orders if o not in searched_set]

    n_control = min(CONTROL_MAX, max(CONTROL_MIN, len(novel)))
    control = [describe_order(e.order) for e in unsearched[:n_control]]

    common = {
        "position_id": position["position_id"],
        "opp_team_id": position["opp_team_id"],
        "split": position["split"],
        "level": level,
        "searched_orders": sorted(searched_set),
        "n_ranked": len(ranked),
        "searched_matches_recorded": None
        if recorded is None
        else set(recorded) == set(searched_keys),
        "searched_orders_recorded": recorded,
        "engine_chosen_in_searched": chosen_ok,
        "gap_top2": round(gap, 2),
        "prep_seconds": round(rebuild_s, 2),
    }
    luna_row = {
        **common,
        "source": "luna",
        "orders": orders,
        "novel_orders": novel,
        "status": status,
        "call": call,
    }
    control_row = {
        **common,
        "source": "control",
        "orders": control,
        "novel_orders": control,
        "matched_to_luna_novel": len(novel),
        "status": status,
    }
    return luna_row, control_row, hit_cap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("positions_dir", type=Path, help="runs/positions/<run>")
    ap.add_argument("--split", choices=("all", "tune", "test"), default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--level", choices=LEVELS, default="none", help="reasoning effort")
    ap.add_argument("--fake", default=None, metavar="SCENARIO",
                    help=f"FakeLLMClient scenario ({', '.join(SCENARIOS)}) or 'novel'; no network")
    ap.add_argument("--max-usd", type=float, default=0.25, help="hard cap for this screen")
    ap.add_argument("--budget-s", type=float, default=10.0, help="per-call deadline")
    ap.add_argument("--spend-file", type=Path, default=LLM_DIR / "screen_spend.json")
    ap.add_argument("--log", type=Path, default=LLM_DIR / "screen_calls.jsonl")
    ap.add_argument("--model", default=LLMConfig().model)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--skip-clear-gap", action="store_true",
                    help="skip positions the live proposer would skip (engine top-2 gap >= clear_gap)")
    args = ap.parse_args()

    run_dir = args.positions_dir
    positions = [
        json.loads(line)
        for line in (run_dir / "positions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if args.split != "all":
        positions = [p for p in positions if p["split"] == args.split]
    if args.limit:
        positions = positions[: args.limit]

    llm_cfg = LLMConfig(model=args.model, spend_cap_usd=args.max_usd)
    if args.fake:
        if args.fake == "novel":
            client: Any = NovelFakeClient()
        elif args.fake in SCENARIOS:
            client = FakeLLMClient(args.fake)
        else:
            raise SystemExit(f"unknown --fake scenario {args.fake!r}")
        meter = SpendMeter(None, args.max_usd, args.model)  # in memory: never touches a file
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set; use --fake SCENARIO for a $0 run")
        client = OpenAIResponsesClient(args.model)
        meter = shared_meter(args.spend_file, args.max_usd, args.model)
        print(f"real client {args.model}; spend file {args.spend_file}, cap ${meter.cap_usd:.2f}, "
              f"already committed ${meter.committed_usd:.4f}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    luna_path = run_dir / f"proposals_luna{suffix}.jsonl"
    control_path = run_dir / f"proposals_control{suffix}.jsonl"
    done = _done_ids(luna_path) & _done_ids(control_path)
    todo = [p for p in positions if p["position_id"] not in done]
    print(f"{len(positions)} positions, {len(done)} already done, {len(todo)} to do", flush=True)

    config = engine_config()
    seconds: list[float] = []
    spent = 0.0
    stopped = None
    n_done = n_novel = 0
    with luna_path.open("a") as luna_out, control_path.open("a") as control_out:
        for index, position in enumerate(todo, start=1):
            t0 = time.perf_counter()
            try:
                luna, control, hit_cap = propose_one(
                    position, run_dir, client=client, meter=meter, level=args.level,
                    budget_s=args.budget_s, log_path=args.log, llm_cfg=llm_cfg, config=config,
                    skip_clear_gap=args.skip_clear_gap,
                )
            except Exception as exc:  # noqa: BLE001 - one bad position must not end the run
                print(f"[{index}/{len(todo)}] {position['position_id']}: ERROR "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            if hit_cap:
                stopped = "spend cap reached"
                print(f"[{index}/{len(todo)}] {position['position_id']}: spend cap reached "
                      f"(${meter.committed_usd:.4f} of ${meter.cap_usd:.2f}); stopping cleanly. "
                      "Rerun the same command to continue.", flush=True)
                break
            luna_out.write(json.dumps(luna) + "\n")
            control_out.write(json.dumps(control) + "\n")
            luna_out.flush()
            control_out.flush()
            elapsed = time.perf_counter() - t0
            seconds.append(elapsed)
            spent += float(luna["call"].get("cost_usd", 0.0))
            n_done += 1
            n_novel += 1 if luna["novel_orders"] else 0
            print(
                f"[{index}/{len(todo)}] {position['position_id']}: {luna['status']}, "
                f"{len(luna['orders'])} proposed / {len(luna['novel_orders'])} novel, "
                f"control {len(control['orders'])}, "
                f"${luna['call'].get('cost_usd', 0.0):.4f}, {elapsed:.1f}s",
                flush=True,
            )
    summary = {
        "positions_done_this_run": n_done,
        "positions_with_novel_proposal": n_novel,
        "spent_this_run_usd": round(spent, 4),
        "mean_seconds_per_position": round(sum(seconds) / len(seconds), 2) if seconds else None,
        "stopped": stopped,
        "luna_file": str(luna_path),
        "control_file": str(control_path),
    }
    print(json.dumps(summary, indent=2))
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
