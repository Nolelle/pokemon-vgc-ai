"""Measure what the training-time exact search's information leak actually costs.

`vgc.rl.exact_search.search_joint_orders_exact` enumerates the opponent's likely replies
from `root.battles[other]` -- their PRIVATE battle object, with true moves/item/stats.
`vgc.rl.live_mirror.LiveExactMirror` rebuilds the same decision from only what a real
ladder game reveals. Both currently exist; only the first is used to mint teacher labels
(see CLAUDE.md's "Exact Showdown mechanics" section).

This script answers the two questions that decide whether closing that gap is worth its
cost, without training anything:

1. AGREEMENT: on real positions, how often does the search's top pick change between the
   true root ("peek") and the reconstructed root ("guess")? Rare disagreement means the
   leak is mostly harmless. Frequent disagreement means teacher labels are currently
   advising moves the deployed agent can never justify from what it actually sees.
2. TIMING: how much slower is a decision through the reconstructed root? The ladder has a
   turn clock; a live-mirror hybrid is only viable if this is affordable per decision.

Both roots run through the SAME search at a reduced (but identical) candidate width, so
the comparison isolates the information source, not search thoroughness.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.config import REPO_ROOT
from vgc.evaluation import wilson_interval
from vgc.evaluator import score_joint_orders
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.exact_search import search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror

DEFAULT_MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json"
DEFAULT_OUTPUT = REPO_ROOT / "runs" / "eval" / "opponent_information_leak.json"

# Deliberately narrow: the question is whether the INFORMATION source changes the
# answer, not how thorough the search is. Both roots run this identically.
DIAGNOSTIC_CONFIG = PolicyConfig(
    search_our_candidates=4,
    search_opp_candidates=4,
    exact_search_future_samples=1,
    exact_search_state_hypotheses=1,
    exact_search_spread_hypotheses=1,
    use_rolling_horizon=False,
    use_value_head=False,
)


@dataclass
class Comparison:
    team_pair: str
    turn: int
    agree: bool
    peek_order: str
    guess_order: str
    peek_seconds: float
    guess_seconds: float


def _load_team_files(manifest_path: Path, count: int, seed: int) -> list[Path]:
    """One team per distinct archetype where possible -- cluster-by-team, not by game."""

    manifest = json.loads(manifest_path.read_text())
    by_archetype: dict[str, list[str]] = {}
    for entry in manifest:
        by_archetype.setdefault(entry["archetype"], []).append(entry["file"])
    rng = random.Random(seed)
    archetypes = sorted(by_archetype)
    rng.shuffle(archetypes)
    chosen: list[Path] = []
    pool_dir = manifest_path.parent
    index = 0
    while len(chosen) < count:
        archetype = archetypes[index % len(archetypes)]
        files = by_archetype[archetype]
        chosen.append(pool_dir / rng.choice(files))
        index += 1
    return chosen


def _pick_myopic(battle, config: PolicyConfig, rng: random.Random):
    """Cheap move selection to drive the game between checkpoints -- no exact search."""

    scored = score_joint_orders(battle, config)
    if scored:
        return scored[0].order
    orders = enumerate_joint_orders(battle)
    return rng.choice(orders) if orders else None


def _compare_at_checkpoint(
    root: DirectBattle,
    own_team: str,
    team_pair_label: str,
    turn: int,
) -> Comparison | None:
    if "p1" not in root.sides_to_move():
        return None

    t0 = time.perf_counter()
    peek_scored = search_joint_orders_exact(root, "p1", DIAGNOSTIC_CONFIG)
    peek_seconds = time.perf_counter() - t0
    if not peek_scored:
        return None

    mirror = LiveExactMirror(own_team, DIAGNOSTIC_CONFIG)
    try:
        mirror_root = mirror.build(root.battles["p1"])
        try:
            t0 = time.perf_counter()
            guess_scored = search_joint_orders_exact(mirror_root, "p1", DIAGNOSTIC_CONFIG)
            guess_seconds = time.perf_counter() - t0
        finally:
            mirror_root.close()
    finally:
        mirror.close()
    if not guess_scored:
        return None

    peek_order = describe_order(peek_scored[0].order)
    guess_order = describe_order(guess_scored[0].order)
    return Comparison(
        team_pair=team_pair_label,
        turn=turn,
        agree=peek_order == guess_order,
        peek_order=peek_order,
        guess_order=guess_order,
        peek_seconds=peek_seconds,
        guess_seconds=guess_seconds,
    )


def run(
    *,
    manifest_path: Path,
    pairs: int,
    checkpoints: tuple[int, ...],
    seed: int,
) -> list[Comparison]:
    team_files = _load_team_files(manifest_path, pairs * 2, seed)
    comparisons: list[Comparison] = []
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        for pair_index in range(pairs):
            own_path = team_files[2 * pair_index]
            opp_path = team_files[2 * pair_index + 1]
            own_team = own_path.read_text().strip()
            opp_team = opp_path.read_text().strip()
            label = f"{own_path.parent.name}_vs_{opp_path.parent.name}"
            rng = random.Random(seed + pair_index)

            root = DirectBattle.start(
                worker,
                f"leak-probe-{pair_index}",
                own_team,
                opp_team,
                seed=[rng.randint(1, 2**31 - 1) for _ in range(4)],
            )
            try:
                root.step({"p1": "team 1234", "p2": "team 1234"})
                next_checkpoint = 0
                for turn in range(1, max(checkpoints) + 1):
                    if root.ended:
                        break
                    if next_checkpoint < len(checkpoints) and turn == checkpoints[next_checkpoint]:
                        try:
                            comparison = _compare_at_checkpoint(root, own_team, label, turn)
                        except Exception as exc:  # noqa: BLE001 -- a rare worker-side
                            # race (a request answered between snapshot and patch) should
                            # skip this one checkpoint, not abort a 15-pair run over it.
                            print(f"[{label} turn {turn}] SKIPPED ({exc})")
                            comparison = None
                        if comparison is not None:
                            comparisons.append(comparison)
                            print(
                                f"[{label} turn {turn}] "
                                f"{'AGREE' if comparison.agree else 'DISAGREE'} "
                                f"peek={comparison.peek_seconds:.2f}s guess={comparison.guess_seconds:.2f}s"
                            )
                        next_checkpoint += 1
                    choices = {}
                    for side in root.sides_to_move():
                        order = _pick_myopic(root.battles[side], DIAGNOSTIC_CONFIG, rng)
                        if order is None:
                            continue
                        choices[side] = choice_string(order)
                    if not choices:
                        break
                    root.step(choices)
            finally:
                root.close()
    return comparisons


def summarize(comparisons: list[Comparison]) -> dict:
    if not comparisons:
        return {"comparisons": 0}
    agreements = sum(1 for c in comparisons if c.agree)
    total = len(comparisons)
    low, high = wilson_interval(agreements, total)
    peek_times = sorted(c.peek_seconds for c in comparisons)
    guess_times = sorted(c.guess_seconds for c in comparisons)

    def median(values: list[float]) -> float:
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    return {
        "comparisons": total,
        "agreements": agreements,
        "agreement_rate": agreements / total,
        "agreement_wilson_low": low,
        "agreement_wilson_high": high,
        "peek_seconds_median": median(peek_times),
        "guess_seconds_median": median(guess_times),
        "slowdown_factor_median": median(guess_times) / median(peek_times)
        if median(peek_times) > 0
        else None,
        "disagreements": [
            {
                "team_pair": c.team_pair,
                "turn": c.turn,
                "peek_order": c.peek_order,
                "guess_order": c.guess_order,
            }
            for c in comparisons
            if not c.agree
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs", type=int, default=15, help="number of team-pair games")
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[2, 4, 6],
        help="turn numbers to compare peek vs guess at, per game",
    )
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    comparisons = run(
        manifest_path=args.manifest,
        pairs=args.pairs,
        checkpoints=tuple(sorted(args.checkpoints)),
        seed=args.seed,
    )
    summary = summarize(comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== opponent information leak: peek (true root) vs guess (live mirror) ===")
    if summary["comparisons"] == 0:
        print("no usable comparisons -- check that the local Showdown checkout is available")
        return
    print(f"comparisons: {summary['comparisons']}")
    print(
        f"agreement: {summary['agreements']}/{summary['comparisons']} "
        f"= {summary['agreement_rate']:.1%} "
        f"(Wilson low {summary['agreement_wilson_low']:.1%})"
    )
    print(f"peek median: {summary['peek_seconds_median']:.2f}s")
    print(f"guess median: {summary['guess_seconds_median']:.2f}s")
    print(f"slowdown factor: {summary['slowdown_factor_median']:.1f}x")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
