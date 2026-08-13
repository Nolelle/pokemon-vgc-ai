"""Confirm a heuristic change across the archetype team pool.

The candidate defaults to the shipped heuristic: true own-team spreads and Protect
weight 0.6. The incumbent defaults to the legacy behavior it must replace: fake own-team
spreads and Protect weight 0.8. Each matchup is a same-team mirror, battle sides
alternate, and every team gets an even number of games. Unlike Phase 2's single fixed
mirror, team preview is not pinned to ``team 1234`` because that would force arbitrary
pool teams to bring whichever four happen to occupy their first four file positions.

Pass conditions are declared before the run:

* the overall 95% CLUSTER-ROBUST interval lower bound must exceed 50%; and
* no archetype may be clearly losing after a Holm family-wise correction.

## Why both pass conditions changed

The first two runs of this gate (``runs/eval/own_spread_archetype_pool*.json``) used a
pooled Wilson interval and six uncorrected archetype checks. Both understate uncertainty
in ways that matter here:

* **Pooling.** Games are clustered in teams and teams genuinely differ -- the measured
  between-team SD is ~0.10 win rate, which inflates the pooled variance by ~1.8x. The
  reported ``[0.476, 0.537]`` was really ``[0.466, 0.547]``. See `vgc.evaluation`.
* **Six subgroups.** Six independent uncorrected 95% checks give a ~14% chance of at
  least one false alarm under a true null. That is what happened: ``gardevoir_maushold``
  was flagged at 40.7% and then measured 54.9% on the next seed, and a policy change was
  built to chase it before the pool showed it was worth +0.31 wins per team (t = 0.74).

``--null-test`` runs both arms with the SAME config. It must land at 50%; anything else
means the harness itself favors a seat or an arm, which would contaminate every result
this gate has ever produced.
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
from vgc.evaluation import (  # noqa: E402
    clustered_interval,
    holm_rejections,
    student_t_cdf,
    variance_components,
    wilson_interval,
)
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402
from vgc.rl.match import run_series, summarize  # noqa: E402

DEFAULT_MANIFEST = REPO_ROOT / "data" / "selfplay" / "archetype_pool" / "manifest.json"
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "own_spread_archetype_pool_gate.json"
CANDIDATE_NAME = "true_spread_protect_060"
INCUMBENT_NAME = "legacy_fake_spread_protect_080"
# Both arms of --null-test. Distinct names because BattleOutcome attributes results by
# name, so a mirror needs two labels even when the two configs are identical.
NULL_A_NAME = "null_arm_a"
NULL_B_NAME = "null_arm_b"

CANDIDATE_ARM = {"use_own_team_spreads": True, "protect_threat_weight": 0.6}
INCUMBENT_ARM = {"use_own_team_spreads": False, "protect_threat_weight": 0.8}


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
    entries: Sequence[dict[str, Any]],
    games_per_team: int,
    base_seed: int,
    arms: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Play every team in `entries`. `arms` maps arm name -> PolicyConfig overrides.

    Insertion order matters: the first arm is the one whose win rate the report is
    stated in terms of. Under `--null-test` the two override dicts are equal, and the
    only asymmetry left in the run is which seat each arm starts in -- which
    `run_series` alternates away.
    """

    (first_name, first_overrides), (second_name, second_overrides) = arms.items()
    first_config = PolicyConfig(format_id=FORMAT_ID, **first_overrides)
    second_config = PolicyConfig(format_id=FORMAT_ID, **second_overrides)
    results = []
    with SimWorker() as worker:
        for entry in entries:
            team = str(entry["team"])
            factories = {
                first_name: partial(make_direct_agent, "vgc", team, config=first_config),
                second_name: partial(make_direct_agent, "vgc", team, config=second_config),
            }
            outcomes = run_series(
                worker,
                factories,
                {first_name: team, second_name: team},
                games_per_team,
                seed=base_seed + int(entry["index"]) * 1009,
            )
            result = summarize(outcomes, first_name, second_name)
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
    """Pool one group of teams, reporting BOTH intervals.

    `wilson` is kept so older reports stay comparable, but it is not what the gate
    decides on -- it treats the group's games as independent when they are clustered in
    teams. `clustered` is the honest interval and the one the pass conditions use.
    """

    games = sum(int(result["games"]) for result in results)
    wins = sum(int(result["p1_wins"]) for result in results)
    losses = sum(int(result["p2_wins"]) for result in results)
    draws = sum(int(result["draws"]) for result in results)
    clusters = [(int(result["p1_wins"]), int(result["games"])) for result in results]
    low, high = wilson_interval(wins, games)
    clustered_low, clustered_high = clustered_interval(clusters)
    components = variance_components(clusters)
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
        "clustered": [clustered_low, clustered_high],
        "team_effect_sd": components.tau,
        "naive_se": components.naive_se,
        "clustered_se": components.clustered_se,
        "design_effect": components.design_effect,
        "se_floor": components.se_floor,
        "minimum_detectable_effect": components.minimum_detectable_effect,
        "floor_detectable_effect": components.floor_detectable_effect,
        "mean_turns": mean_turns,
    }


def archetype_guardrail(
    by_archetype: dict[str, dict[str, Any]], alpha: float = 0.05
) -> dict[str, Any]:
    """Flag archetypes the change clearly HURTS, corrected for testing several at once.

    Each archetype gets a one-sided cluster-robust t test of `win rate < 0.50`, then Holm
    holds the family-wise false-alarm rate at `alpha` across all archetypes. All three
    pieces matter: without clustering the per-archetype SE is too small, without the t
    (df = teams - 1, since the SE is estimated from ~9 teams) an estimated SE is treated
    as a known one, and without Holm six subgroup checks trip on noise about one run in
    seven.
    """

    names = sorted(by_archetype)
    pvalues = []
    for name in names:
        result = by_archetype[name]
        se = float(result["clustered_se"])
        rate = float(result["win_rate"])
        teams = int(result["teams"])
        # No spread between teams means no usable SE; treat as non-significant.
        if se <= 0 or teams < 2:
            pvalues.append(1.0)
        else:
            pvalues.append(student_t_cdf((rate - 0.5) / se, teams - 1))
    rejected = holm_rejections(pvalues, alpha)
    return {
        "alpha": alpha,
        "method": (
            "one-sided cluster-robust t test per archetype (df = teams - 1), "
            "Holm-corrected across archetypes"
        ),
        "pvalues": dict(zip(names, pvalues)),
        "clearly_losing": [name for name, flag in zip(names, rejected) if flag],
        # Kept visible so the cost of the correction is auditable rather than implicit.
        "uncorrected_would_flag": [name for name, p in zip(names, pvalues) if p <= alpha],
    }


def build_report(team_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the run and apply its predeclared pass conditions."""

    overall = aggregate(team_results, "overall")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in team_results:
        grouped[str(result["archetype"])].append(result)
    by_archetype = {
        archetype: aggregate(results, archetype) for archetype, results in sorted(grouped.items())
    }
    overall_pass = float(overall["clustered"][0]) > 0.5
    guardrail = archetype_guardrail(by_archetype)
    clearly_losing = guardrail["clearly_losing"]
    return {
        "passed": overall_pass and not clearly_losing,
        "pass_conditions": {
            "overall": "95% cluster-robust (by team) lower bound > 0.50",
            "archetype_guardrail": (
                "no archetype clearly below 0.50 by a one-sided cluster-robust test, "
                "Holm-corrected across archetypes at family-wise alpha 0.05"
            ),
        },
        "overall_passed": overall_pass,
        "clearly_losing_archetypes": clearly_losing,
        "guardrail": guardrail,
        "power": {
            "note": (
                "minimum_detectable_effect is the smallest true edge this run could "
                "certify. floor_detectable_effect is the smallest edge THIS TEAM POOL "
                "could certify at any number of games -- below it, only more teams help."
            ),
            "minimum_detectable_effect": overall["minimum_detectable_effect"],
            "floor_detectable_effect": overall["floor_detectable_effect"],
            "team_effect_sd": overall["team_effect_sd"],
            "design_effect": overall["design_effect"],
        },
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
    parser.add_argument(
        "--null-test",
        action="store_true",
        help=(
            "A/A sanity run: give BOTH arms the candidate config. A correct harness "
            "returns 50%%; anything else is a seat or arm bias that invalidates every "
            "A/B this gate has produced."
        ),
    )
    return parser.parse_args()


def build_arms(null_test: bool) -> dict[str, dict[str, Any]]:
    """Arm name -> PolicyConfig overrides, in report order (first arm is the subject)."""

    if null_test:
        return {NULL_A_NAME: dict(CANDIDATE_ARM), NULL_B_NAME: dict(CANDIDATE_ARM)}
    return {CANDIDATE_NAME: dict(CANDIDATE_ARM), INCUMBENT_NAME: dict(INCUMBENT_ARM)}


def main() -> int:
    args = parse_args()
    if args.games_per_team < 2 or args.games_per_team % 2:
        raise SystemExit("--games-per-team must be a positive even number of at least 2")
    entries = load_pool(args.manifest)
    chunks = balanced_chunks(entries, args.workers)
    arms = build_arms(args.null_test)
    total_games = len(entries) * args.games_per_team
    mode = "A/A NULL TEST (both arms identical)" if args.null_test else "A/B gate"
    print(
        f"{mode}: {total_games} games across {len(entries)} teams with {len(chunks)} workers",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        futures = [
            executor.submit(_evaluate_chunk, chunk, args.games_per_team, args.seed, arms)
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
    (first_name, first_overrides), (second_name, second_overrides) = arms.items()
    report.update(
        {
            "null_test": args.null_test,
            "candidate": {"name": first_name, **first_overrides},
            "incumbent": {"name": second_name, **second_overrides},
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
        f"overall: {overall['wins']}/{overall['games']} = {overall['win_rate']:.3f}\n"
        f"  cluster-robust 95% CI [{overall['clustered'][0]:.3f}, "
        f"{overall['clustered'][1]:.3f}]  <- the gate decides on this\n"
        f"  naive Wilson    95% CI [{overall['wilson'][0]:.3f}, {overall['wilson'][1]:.3f}]"
        f"  (design effect {overall['design_effect']:.2f}x)",
        flush=True,
    )
    print(
        f"  team-effect SD {overall['team_effect_sd']:.3f}; this run can certify an edge "
        f">= +{overall['minimum_detectable_effect'] * 100:.1f}pts, and this POOL can never "
        f"certify below +{overall['floor_detectable_effect'] * 100:.1f}pts at any game count",
        flush=True,
    )
    for name, result in report["by_archetype"].items():
        print(
            f"  {name}: {result['wins']}/{result['games']} = {result['win_rate']:.3f}, "
            f"cluster-robust [{result['clustered'][0]:.3f}, {result['clustered'][1]:.3f}]",
            flush=True,
        )
    guardrail = report["guardrail"]
    if guardrail["uncorrected_would_flag"]:
        print(
            f"  guardrail: flagged {guardrail['clearly_losing'] or 'none'} after Holm; "
            f"uncorrected would have flagged {guardrail['uncorrected_would_flag']}",
            flush=True,
        )
    if args.null_test:
        low, high = overall["clustered"]
        verdict = (
            "OK" if low <= 0.5 <= high else "HARNESS BIAS -- investigate before trusting any A/B"
        )
        print(f"null test: 50% {'inside' if low <= 0.5 <= high else 'OUTSIDE'} CI -> {verdict}")
        return 0 if low <= 0.5 <= high else 1
    print(f"passed: {report['passed']}; wrote {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
