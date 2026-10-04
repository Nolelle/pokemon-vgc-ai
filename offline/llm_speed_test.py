#!/usr/bin/env python3
"""Measure real per-call latency, tokens, cost and answer validity for the LLM advisor.

Stage 1 (no network): play a few offline games on the direct environment, with a
recording player of ours (owner teams) against the normal `vgc` bot on real ladder teams.
At each of our decisions it captures the engine's ranking and builds a packet with
`vgc.llm.facts.build_packet`. Packets are cached to disk, so a re-run does not replay games.

Stage 2: call the client for `--calls-per-level` packets at each `--levels` value through
`vgc.llm.harness.advise` (the production path: strict ID validation, spend meter, JSONL
call log) and report per level: n, valid-answer rate, p50/p95/max latency, mean input /
cached / output tokens, total and per-call cost, incomplete count.

`--fake` uses `FakeLLMClient` ($0, no network). Without it the real GPT-6 Luna client is
used, which needs OPENAI_API_KEY and spends real money, capped locally by `--max-usd`
(separate spend file `runs/llm/speed_spend.json`, so it never touches the live-play cap).

    PYTHONPATH=src .venv/bin/python offline/llm_speed_test.py --fake --calls-per-level 5
    PYTHONPATH=src .venv/bin/python offline/llm_speed_test.py --levels none low
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import FORMAT_ID, RUNS_DIR  # noqa: E402
from vgc.llm.client import FakeLLMClient, OpenAIResponsesClient  # noqa: E402
from vgc.llm.config import LEVELS, LLMConfig  # noqa: E402
from vgc.llm.facts import build_packet, load_team_plan  # noqa: E402
from vgc.llm.harness import advise  # noqa: E402
from vgc.llm.packet import new_request_id  # noqa: E402
from vgc.llm.spend import SpendMeter  # noqa: E402
from vgc.llm.types import ContextPacket  # noqa: E402

DEFAULT_OUR_TEAMS = (
    REPO_ROOT / "teams" / "owner" / "psyspam_sand.packed.txt",
    REPO_ROOT / "teams" / "owner" / "salamence_tw.packed.txt",
)
LLM_DIR = RUNS_DIR / "llm"
MANIFEST_REL = Path("data") / "selfplay" / "mc_sheet_pool_v2" / "holdout_manifest.json"


# ---------------------------------------------------------------------------------------
# Stage 1: positions
# ---------------------------------------------------------------------------------------


def find_manifest(explicit: Path | None) -> Path:
    """The opponent-team manifest: --manifest, this checkout, or the main checkout (a
    worktree does not carry the gitignored data/selfplay tree)."""
    candidates = [explicit] if explicit else []
    candidates.append(REPO_ROOT / MANIFEST_REL)
    try:
        common = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=REPO_ROOT, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
        candidates.append((REPO_ROOT / common).resolve().parent / MANIFEST_REL)
    except Exception:
        pass
    for path in candidates:
        if path is not None and Path(path).is_file():
            return Path(path)
    raise FileNotFoundError(f"no opponent manifest found; tried {[str(c) for c in candidates]}")


def load_opponents(manifest: Path) -> list[dict[str, str]]:
    rows = json.loads(manifest.read_text())
    out = []
    for row in rows:
        path = manifest.parent / str(row["file"])
        if path.is_file():
            out.append({"file": str(row["file"]), "team": path.read_text().strip()})
    if not out:
        raise ValueError(f"no usable team files next to {manifest}")
    return out


def make_recorder_class() -> type:
    """A VgcPlayer that logs a packet at every one of our (non-trivial) decisions."""
    from vgc.actions import describe_order, enumerate_joint_orders
    from vgc.agent import VgcPlayer
    from vgc.evaluator import score_joint_orders
    from vgc.search import search_joint_orders

    class RecorderPlayer(VgcPlayer):
        def __init__(self, *args: Any, sink: list, team_plan: str, meta: dict, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._sink, self._plan, self._meta = sink, team_plan, meta

        def decide(self, battle):  # noqa: ANN001
            memory = self._memory_for(battle)
            try:
                scored = search_joint_orders(battle, self.config)
            except Exception:
                scored = score_joint_orders(battle, self.config)
            if not scored:
                return self.choose_random_move(battle)
            memory.record_choice(int(getattr(battle, "turn", 0) or 0), describe_order(scored[0].order))
            if len(scored) > 1 and not any(getattr(battle, "force_switch", None) or []):
                packet, options = build_packet(
                    battle, scored, llm_config=LLMConfig(), team_plan=self._plan,
                    memory=memory, request_id=new_request_id(getattr(battle, "turn", None)),
                )
                self._sink.append(
                    {**dataclasses.asdict(packet), **self._meta, "n_options": len(options),
                     "n_legal": len(enumerate_joint_orders(battle))}
                )
            return scored[0].order

    return RecorderPlayer


def collect_positions(
    n_positions: int, our_teams: list[Path], manifest: Path, seed: int, max_games: int
) -> list[dict[str, Any]]:
    from vgc.models import PolicyConfig
    from vgc.rl.agents import DirectAgent, make_direct_agent
    from vgc.rl.env import SimWorker
    from vgc.rl.match import play_battle

    recorder_cls = make_recorder_class()
    opponents = load_opponents(manifest)
    rng = random.Random(seed)
    sink: list[dict[str, Any]] = []
    config = PolicyConfig(format_id=FORMAT_ID)
    with SimWorker() as worker:
        for game in range(max_games):
            if len(sink) >= n_positions * 2:  # keep a surplus, then sample evenly
                break
            ours_path = our_teams[game % len(our_teams)]
            ours = ours_path.read_text().strip()
            opp = opponents[rng.randrange(len(opponents))]
            meta = {"our_team": ours_path.name.removesuffix(".txt").removesuffix(".packed"),
                    "opp_team": opp["file"], "game": game}
            player = recorder_cls(
                config=config, team=ours, battle_format=FORMAT_ID, start_listening=False,
                sink=sink, team_plan=load_team_plan(ours_path), meta=meta,
            )
            agents = {
                "ours": DirectAgent(player, name="ours"),
                "opp": make_direct_agent("vgc", opp["team"]),
            }
            agents["opp"].name = "opp"
            first, second = ("ours", "opp") if game % 2 == 0 else ("opp", "ours")
            t0 = time.monotonic()
            play_battle(
                worker, f"speed{game}", {"p1": agents[first], "p2": agents[second]},
                {"p1": ours if first == "ours" else opp["team"],
                 "p2": ours if second == "ours" else opp["team"]},
                seed=[rng.randrange(1, 2**31) for _ in range(4)],
            )
            print(f"  game {game}: {len(sink)} positions so far ({time.monotonic() - t0:.1f}s)",
                  file=sys.stderr)
    if len(sink) > n_positions:  # even spread across games and turns, deterministic
        step = len(sink) / n_positions
        sink = [sink[int(i * step)] for i in range(n_positions)]
    return sink


def load_or_collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    cache: Path = args.positions_cache
    if cache.is_file() and not args.regen:
        positions = json.loads(cache.read_text())
        if len(positions) >= args.positions:
            print(f"using {len(positions)} cached positions from {cache}", file=sys.stderr)
            return positions[: args.positions]
    manifest = find_manifest(args.manifest)
    print(f"collecting {args.positions} positions (opponents: {manifest})", file=sys.stderr)
    positions = collect_positions(
        args.positions, [Path(p) for p in args.our_teams], manifest, args.seed, args.max_games
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(positions))
    return positions


def packet_size_report(positions: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[int]) -> dict[str, float]:
        return {"mean": round(statistics.fmean(values), 1), "max": max(values)} if values else {}

    sizes = [len((p["fixed_text"] + p["turn_text"]).encode()) for p in positions]
    return {
        "n_positions": len(positions),
        "total_bytes": stats(sizes),
        "fixed_bytes": stats([len(p["fixed_text"].encode()) for p in positions]),
        "turn_bytes": stats([len(p["turn_text"].encode()) for p in positions]),
        "est_total_tokens_at_4_bytes": stats([s // 4 for s in sizes]),
        "options": stats([p["n_options"] for p in positions]),
    }


# ---------------------------------------------------------------------------------------
# Stage 2: calls
# ---------------------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize_level(level: str, rows: list[dict[str, Any]], n_calls: int) -> dict[str, Any]:
    first = [r for r in rows if r["attempt"] == 0]
    sent = [r for r in first if r["status"] != "spend_cap"]
    lat = [r["latency_s"] for r in sent if r["latency_s"] > 0]
    done = [r for r in sent if r["input_tokens"] > 0]
    total_cost = sum(r["cost_usd"] for r in rows)
    statuses: dict[str, int] = {}
    for r in first:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    valid = sum(1 for r in first if r["status"] == "ok")

    def mean(key: str) -> float:
        return round(statistics.fmean([r[key] for r in done]), 1) if done else 0.0

    return {
        "level": level,
        "n": n_calls,
        "valid_rate": round(valid / n_calls, 4) if n_calls else 0.0,
        "valid": valid,
        "incomplete": statuses.get("incomplete", 0),
        "statuses": statuses,
        "latency_s": {"p50": round(percentile(lat, 0.5), 3), "p95": round(percentile(lat, 0.95), 3),
                      "max": round(max(lat), 3) if lat else 0.0},
        "mean_input_tokens": mean("input_tokens"),
        "mean_cached_tokens": mean("cached_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_call_usd": round(total_cost / n_calls, 6) if n_calls else 0.0,
        "cache_hit_share_of_input": round(
            sum(r["cached_tokens"] for r in done) / max(1, sum(r["input_tokens"] for r in done)), 3
        ),
    }


def run_calls(args: argparse.Namespace, positions: list[dict[str, Any]]) -> dict[str, Any]:
    cfg = LLMConfig(model=args.model)
    if args.fake:
        client: Any = FakeLLMClient("valid", sleep_s=0.02)
        meter = SpendMeter(None, cap_usd=args.max_usd, model=cfg.model)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (use --fake for a $0 dry run)")
        client = OpenAIResponsesClient(cfg.model)
        meter = SpendMeter(args.spend_file, cap_usd=args.max_usd, model=cfg.model)
    log_path = args.out.with_suffix(".calls.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    results: dict[str, Any] = {}
    for level in args.levels:
        start_n = len(log_path.read_text().splitlines()) if log_path.exists() else 0
        n_valid_ids = 0
        for i in range(args.calls_per_level):
            raw = positions[i % len(positions)]
            packet = ContextPacket(
                fixed_text=raw["fixed_text"], turn_text=raw["turn_text"],
                option_ids=tuple(raw["option_ids"]),
                request_id=new_request_id(raw.get("turn")), turn=raw.get("turn"),
            )
            advice = advise(packet, client, level, args.budget_s, meter, log_path, config=cfg)
            n_valid_ids += advice is not None
            if meter.committed_usd >= meter.cap_usd:
                print(f"  spend cap reached at {level} call {i + 1}", file=sys.stderr)
        rows = [json.loads(line) for line in log_path.read_text().splitlines()[start_n:]]
        results[level] = summarize_level(level, rows, args.calls_per_level)
        results[level]["valid_by_harness"] = n_valid_ids
        s = results[level]
        print(f"{level:>6}: n={s['n']} valid={s['valid_rate']:.1%} "
              f"p50={s['latency_s']['p50']}s p95={s['latency_s']['p95']}s "
              f"max={s['latency_s']['max']}s in={s['mean_input_tokens']} "
              f"cached={s['mean_cached_tokens']} out={s['mean_output_tokens']} "
              f"cost=${s['total_cost_usd']:.4f} (${s['cost_per_call_usd']:.5f}/call) "
              f"incomplete={s['incomplete']}")
    return {"levels": results, "spent_usd_local_meter": round(meter.spent_usd, 6),
            "calls_log": str(log_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--levels", nargs="+", default=["none", "low"], choices=list(LEVELS))
    p.add_argument("--calls-per-level", type=int, default=150)
    p.add_argument("--fake", action="store_true", help="FakeLLMClient: no network, $0")
    p.add_argument("--max-usd", type=float, default=1.00, help="local spend cap for this run")
    p.add_argument("--spend-file", type=Path, default=LLM_DIR / "speed_spend.json")
    p.add_argument("--out", type=Path, default=LLM_DIR / "speed_test.json")
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--budget-s", type=float, default=60.0, help="per-call deadline (seconds)")
    p.add_argument("--positions", type=int, default=100, help="distinct positions to capture")
    p.add_argument("--positions-cache", type=Path, default=LLM_DIR / "speed_positions.json")
    p.add_argument("--regen", action="store_true", help="ignore the cached positions")
    p.add_argument("--our-teams", nargs="+", type=Path, default=list(DEFAULT_OUR_TEAMS))
    p.add_argument("--manifest", type=Path, default=None, help="opponent team manifest.json")
    p.add_argument("--max-games", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261003)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    positions = load_or_collect(args)
    if not positions:
        raise SystemExit("no positions captured")
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fake": args.fake, "model": args.model, "max_usd": args.max_usd,
        "calls_per_level": args.calls_per_level,
        "packet_size": packet_size_report(positions),
    }
    print("packet size:", json.dumps(report["packet_size"]))
    report.update(run_calls(args, positions))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
