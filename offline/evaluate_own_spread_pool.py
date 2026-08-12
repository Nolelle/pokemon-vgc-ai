"""Confirm the accurate-spread Protect retune across the archetype team pool.

The candidate is the shipped heuristic: true own-team spreads and Protect weight 0.6.
The incumbent is the legacy behavior it must replace: fake own-team spreads and Protect
weight 0.8. Each matchup is a same-team mirror, battle sides alternate, and every team
gets an even number of games. Unlike Phase 2's single fixed mirror, team preview is not
pinned to ``team 1234`` because that would force arbitrary pool teams to bring whichever
four happen to occupy their first four file positions.

Pass conditions are declared before the run:

* the overall 95% Wilson interval lower bound must exceed 50%; and
* no archetype may be clearly losing (its interval upper bound must remain at least 50%).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.config import FORMAT_ID, RUNS_DIR  # noqa: E402
from vgc.evaluation import wilson_interval  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402
from vgc.rl.match import run_series, summarize  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool" / "manifest.json"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "own_spread_archetype_pool_gate.json"
CANDIDATE_NAME = "true_spread_protect_060"
INCUMBENT_NAME = "legacy_fake_spread_protect_080"


def load_pool(manifest_path: Path) -> list[dict[str, Any]]:
    """Load and validate the generated archetype-pool manifest."""

    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{manifest_path} must contain a non-empty JSON list")
    entries = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict) or not raw.get("file") or not raw.get("archetype"):
            raise ValueError(f"invalid team entry at index {index}: {raw!r}")
        team_path = manifest_path.parent / str(raw["file"])
        if not team_path.is_file():
            raise FileNotFoundError(team_path)
        entries.append(
            {
                "index": index,
                "file": str(raw["file"]),
                "archetype": str(raw["archetype"]),
                "team": team_path.read_text().strip(),
            }
        )
    return entries


def balanced_chunks(entries: Sequence[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    """Split teams deterministically across worker processes."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    chunks = [[] for _ in range(min(workers, len(entries)))]
    for index, entry in enumerate(entries):
        chunks[index % len(chunks)].append(entry)
    return chunks


def _evaluate_chunk(
    entries: Sequence[dict[str, Any]], games_per_team: int, base_seed: int
) -> list[dict[str, Any]]:
    candidate_config = PolicyConfig(
        format_id=FORMAT_ID,
        use_own_team_spreads=True,
        protect_threat_weight=0.6,
    )
    incumbent_config = PolicyConfig(
        format_id=FORMAT_ID,
        use_own_team_spreads=False,
        protect_threat_weight=0.8,
    )
    results = []
    with SimWorker() as worker:
        for entry in entries:
            team = str(entry["team"])
            factories = {
                CANDIDATE_NAME: partial(
                    make_direct_agent,
                    "vgc",
                    team,
                    config=candidate_config,
                ),
                INCUMBENT_NAME: partial(
                    make_direct_agent,
                    "vgc",
                    team,
                    config=incumbent_config,
                ),
            }
            outcomes = run_series(
                worker,
                factories,
                {CANDIDATE_NAME: team, INCUMBENT_NAME: team},
                games_per_team,
                seed=base_seed + int(entry["index"]) * 1009,
            )
            result = summarize(outcomes, CANDIDATE_NAME, INCUMBENT_NAME)
            result.update(
                {
                    "archetype": entry["archetype"],
                    "team_file": entry["file"],
                    "seed": base_seed + int(entry["index"]) * 1009,
                }
            )
            results.append(result)
    return results


def aggregate(results: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    games = sum(int(result["games"]) for result in results)
    wins = sum(int(result["p1_wins"]) for result in results)
    losses = sum(int(result["p2_wins"]) for result in results)
    draws = sum(int(result["draws"]) for result in results)
    low, high = wilson_interval(wins, games)
    mean_turns = (
        sum(float(result["mean_turns"]) * int(result["games"]) for result in results) / games
        if games
        else 0.0
    )
    return {
        "label": label,
        "teams": len(results),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games if games else 0.0,
        "wilson": [low, high],
        "mean_turns": mean_turns,
    }


def build_report(team_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the run and apply its predeclared pass conditions."""

    overall = aggregate(team_results, "overall")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in team_results:
        grouped[str(result["archetype"])].append(result)
    by_archetype = {
        archetype: aggregate(results, archetype)
        for archetype, results in sorted(grouped.items())
    }
    overall_pass = float(overall["wilson"][0]) > 0.5
    clearly_losing = [
        name for name, result in by_archetype.items() if float(result["wilson"][1]) < 0.5
    ]
    return {
        "passed": overall_pass and not clearly_losing,
        "pass_conditions": {
            "overall": "95% Wilson lower bound > 0.50",
            "archetype_guardrail": "no archetype 95% Wilson upper bound < 0.50",
        },
        "overall_passed": overall_pass,
        "clearly_losing_archetypes": clearly_losing,
        "overall": overall,
        "by_archetype": by_archetype,
        "team_results": sorted(team_results, key=lambda result: str(result["team_file"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--games-per-team",
        type=int,
        default=18,
        help="must be even so each bot gets each battle side equally (58 x 18 = 1044)",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.games_per_team < 2 or args.games_per_team % 2:
        raise SystemExit("--games-per-team must be a positive even number of at least 2")
    entries = load_pool(args.manifest)
    chunks = balanced_chunks(entries, args.workers)
    total_games = len(entries) * args.games_per_team
    print(
        f"running {total_games} games across {len(entries)} teams with {len(chunks)} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(_evaluate_chunk, chunk, args.games_per_team, args.seed)
            for chunk in chunks
        ]
        team_results = []
        for index, future in enumerate(futures, start=1):
            chunk_results = future.result()
            team_results.extend(chunk_results)
            print(
                f"worker {index}/{len(futures)} complete: {len(chunk_results)} teams",
                flush=True,
            )

    report = build_report(team_results)
    report.update(
        {
            "candidate": {
                "name": CANDIDATE_NAME,
                "use_own_team_spreads": True,
                "protect_threat_weight": 0.6,
            },
            "incumbent": {
                "name": INCUMBENT_NAME,
                "use_own_team_spreads": False,
                "protect_threat_weight": 0.8,
            },
            "manifest": str(args.manifest),
            "seed": args.seed,
            "games_per_team": args.games_per_team,
            "workers": len(chunks),
            "team_preview": "each bot's normal heuristic preview",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    overall = report["overall"]
    print(
        f"overall: {overall['wins']}/{overall['games']} = {overall['win_rate']:.3f}, "
        f"95% CI [{overall['wilson'][0]:.3f}, {overall['wilson'][1]:.3f}]",
        flush=True,
    )
    for name, result in report["by_archetype"].items():
        print(
            f"  {name}: {result['wins']}/{result['games']} = {result['win_rate']:.3f}, "
            f"95% CI [{result['wilson'][0]:.3f}, {result['wilson'][1]:.3f}]",
            flush=True,
        )
    print(f"passed: {report['passed']}; wrote {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
