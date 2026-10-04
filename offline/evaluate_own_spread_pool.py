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

## Asymmetric mode, checkpoints, seats (2026-10-03)

``--our-teams A.packed.txt [B.packed.txt ...]`` switches from same-team mirrors to
OUR team(s) vs each pool (opponent) team. A head-to-head between two arms on two
DIFFERENT teams is confounded by which team is stronger, so each (our team, opponent
team) pair is played in BOTH orientations -- half the games the candidate arm pilots our
team and the incumbent pilots the opponent's, half the reverse -- and seats alternate
inside each orientation. Team strength then cancels in expectation: an A/A run lands at
50% by construction, and the win rate is the candidate arm's skill edge on these teams.

**Clustering.** The CI clusters by OPPONENT team (pooled over our teams), not by
(our team, opponent) pair. The same opponent team appears in every pair it is part of, so
its games are correlated across pairs; treating pairs as independent would understate the
variance by exactly that correlation. Our teams are fixed and few (2), so they are a
fixed effect (reported per our-team in ``by_our_team``), not a population we generalise
over. Consequently the power floor is set by the number of OPPONENT teams.

``--checkpoint-games N`` plays in rounds of about N games and rewrites the output JSON
after each round (``interim: true`` until the last). ``--futility-min-effect X`` may stop
the run early ONLY when the CI upper bound is below ``0.5 + X`` (the target gain is no
longer reachable). It never stops for success: checking every N games and declaring a
win the first time the lower bound clears 50% is a repeated significance test and
inflates false positives; the success decision is made once, at the fixed end.
Futility stops cannot raise false positives (they only cost a little power).
Win rates are also reported by seat (candidate arm as p1 vs as p2).
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
    SEAT_KEYS,
    clustered_interval,
    futility_stop,
    holm_rejections,
    merge_cluster_results,
    plan_rounds,
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


ROUND_SEED_STRIDE = 1_000_003
OUR_TEAM_SEED_STRIDE = 104_729


def _seat_counts(outcomes: Sequence[Any], subject: str) -> dict[str, int]:
    """Games `subject` played as p1 / p2 and how many it won from each seat."""

    counts = dict.fromkeys(SEAT_KEYS, 0)
    for outcome in outcomes:
        seat = outcome.agent_sides[subject]
        counts[f"{seat}_seat_games"] += 1
        if outcome.winner == seat:
            counts[f"{seat}_seat_wins"] += 1
    return counts


def _evaluate_chunk(
    entries: Sequence[dict[str, Any]],
    games_per_team: int,
    base_seed: int,
    arms: dict[str, dict[str, Any]],
    round_index: int = 0,
) -> list[dict[str, Any]]:
    """Play every unit in `entries`. `arms` maps arm name -> PolicyConfig overrides.

    Insertion order matters: the first arm is the one whose win rate the report is
    stated in terms of. Under `--null-test` the two override dicts are equal, and the
    only asymmetry left in the run is which seat each arm starts in -- which
    `run_series` alternates away.

    A plain pool entry is a same-team mirror (original behaviour). An entry carrying
    `our_team`/`our_name` is an asymmetric pair: half the games the first arm pilots our
    team vs the opponent's, half the reverse (see the module docstring for why).
    `round_index` only varies the seed, so each checkpoint round plays fresh games;
    round 0 reproduces the pre-checkpoint seeds exactly.
    """

    (first_name, first_overrides), (second_name, second_overrides) = arms.items()
    first_config = PolicyConfig(format_id=FORMAT_ID, **first_overrides)
    second_config = PolicyConfig(format_id=FORMAT_ID, **second_overrides)
    results = []
    with SimWorker() as worker:
        for entry in entries:
            seed = (
                base_seed
                + int(entry["index"]) * 1009
                + int(entry.get("our_index", 0)) * OUR_TEAM_SEED_STRIDE
                + round_index * ROUND_SEED_STRIDE
            )
            opp_team = str(entry["team"])
            extra: dict[str, Any] = {}
            if "our_team" not in entry:
                orientations = [(opp_team, opp_team, games_per_team)]
            else:
                our_team = str(entry["our_team"])
                half = games_per_team // 2
                # (first arm's team, second arm's team, games)
                orientations = [(our_team, opp_team, half), (opp_team, our_team, half)]
            outcomes = []
            on_ours_games = on_ours_wins = 0
            for orientation, (first_team, second_team, games) in enumerate(orientations):
                series = run_series(
                    worker,
                    {
                        first_name: partial(
                            make_direct_agent, "vgc", first_team, config=first_config
                        ),
                        second_name: partial(
                            make_direct_agent, "vgc", second_team, config=second_config
                        ),
                    },
                    {first_name: first_team, second_name: second_team},
                    games,
                    seed=seed + orientation * 7,
                )
                outcomes.extend(series)
                if "our_team" in entry and orientation == 0:
                    on_ours_games = len(series)
                    on_ours_wins = sum(1 for o in series if o.result_for(first_name) > 0)
            result = summarize(outcomes, first_name, second_name)
            result.update(_seat_counts(outcomes, first_name))
            if "our_team" in entry:
                extra = {
                    "our_team_name": str(entry["our_name"]),
                    # Candidate-arm results while piloting OUR team (diagnostic only;
                    # the headline rate pools both orientations).
                    "on_our_team_games": on_ours_games,
                    "on_our_team_wins": on_ours_wins,
                }
            result.update(
                {
                    "archetype": entry["archetype"],
                    "team_file": entry["file"],
                    "unit_id": entry.get("unit_id", entry["file"]),
                    "seed": seed,
                    **extra,
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
    by_seat = seat_breakdown(results)
    report = {
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
    if by_seat:
        report["by_seat"] = by_seat
    return report


def seat_breakdown(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Candidate-arm win rate when seated as p1 vs p2, with cluster-robust intervals.

    Seats alternate game by game, so the two seats should each hold ~half the games. A
    seat rate far from the overall rate is a seat-dependent bias (or a real p1/p2
    asymmetry in doubles) that the alternation cancels out of the headline number but
    that is worth seeing. Empty for results without per-seat counters (older reports).
    """

    out: dict[str, Any] = {}
    for seat in ("p1", "p2"):
        clusters = [
            (int(r.get(f"{seat}_seat_wins", 0)), int(r.get(f"{seat}_seat_games", 0)))
            for r in results
        ]
        games = sum(g for _, g in clusters)
        if not games:
            return {}
        wins = sum(w for w, _ in clusters)
        low, high = clustered_interval(clusters)
        out[seat] = {
            "games": games,
            "wins": wins,
            "win_rate": wins / games,
            "clustered": [low, high],
        }
    return out


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


def build_report(team_results: Sequence[dict[str, Any]], min_gain: float = 0.0) -> dict[str, Any]:
    """Aggregate the run and apply its predeclared pass conditions.

    `min_gain` (default 0, the original gate) additionally requires the POINT estimate
    to be at least `0.5 + min_gain` -- a lower bound just above 50% on a tiny edge is
    not the effect the experiment was built to find.
    """

    overall = aggregate(team_results, "overall")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in team_results:
        grouped[str(result["archetype"])].append(result)
    by_archetype = {
        archetype: aggregate(results, archetype) for archetype, results in sorted(grouped.items())
    }
    overall_pass = float(overall["clustered"][0]) > 0.5 and float(overall["win_rate"]) >= (
        0.5 + min_gain
    )
    guardrail = archetype_guardrail(by_archetype)
    clearly_losing = guardrail["clearly_losing"]
    return {
        "passed": overall_pass and not clearly_losing,
        "pass_conditions": {
            "overall": (
                "95% cluster-robust (by team) lower bound > 0.50"
                + (f" and point win rate >= {0.5 + min_gain:.3f}" if min_gain else "")
            ),
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
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        metavar="FIELD=VALUE",
        help=(
            "PolicyConfig override for the candidate arm (repeatable). When given, BOTH "
            "arms start from PolicyConfig defaults instead of the own-spread arm pair: the "
            "candidate applies these overrides, the incumbent applies only --incumbent."
        ),
    )
    parser.add_argument(
        "--incumbent",
        action="append",
        default=None,
        metavar="FIELD=VALUE",
        help="PolicyConfig override for the incumbent arm (repeatable); see --candidate.",
    )
    parser.add_argument(
        "--our-teams",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "asymmetric mode: packed team file(s) of OURS; each plays every --manifest "
            "team (the opponents) in both orientations. --games-per-team is then games "
            "per (our team, opponent) pair and must be a multiple of 4. CI clusters by "
            "opponent team."
        ),
    )
    parser.add_argument(
        "--checkpoint-games",
        type=int,
        default=None,
        help=(
            "play in rounds of about this many TOTAL games (rounded to whole per-unit "
            "chunks) and rewrite --output after each round with an interim report"
        ),
    )
    parser.add_argument(
        "--futility-min-effect",
        type=float,
        default=None,
        metavar="X",
        help=(
            "futility-only early stop at checkpoints: stop when the cluster-robust CI "
            "upper bound is below 0.5 + X. Never stops for success (repeated looks "
            "inflate false positives); requires --checkpoint-games"
        ),
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.0,
        help=(
            "also require the point win rate >= 0.5 + this at the fixed end (default 0 = "
            "original gate). LLM-protocol runs pass 0.02, see docs/llm_test_protocol.md"
        ),
    )
    return parser.parse_args()


def parse_overrides(pairs: Sequence[str] | None) -> dict[str, Any]:
    """`FIELD=VALUE` strings -> PolicyConfig kwargs, typed from the field's default."""

    overrides: dict[str, Any] = {}
    defaults = PolicyConfig()
    for pair in pairs or ():
        field, _, raw = pair.partition("=")
        if not _ or not hasattr(defaults, field):
            raise SystemExit(f"unknown PolicyConfig override {pair!r}")
        default = getattr(defaults, field)
        if isinstance(default, bool):
            overrides[field] = raw.lower() in ("1", "true", "yes")
        elif isinstance(default, int):
            overrides[field] = int(raw)
        elif isinstance(default, float):
            overrides[field] = float(raw)
        else:
            overrides[field] = raw
    return overrides


def _arm_name(prefix: str, overrides: dict[str, Any]) -> str:
    if not overrides:
        return f"{prefix}_defaults"
    return prefix + "_" + "_".join(f"{key}_{value}" for key, value in sorted(overrides.items()))


def build_arms(
    null_test: bool,
    candidate: Sequence[str] | None = None,
    incumbent: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Arm name -> PolicyConfig overrides, in report order (first arm is the subject)."""

    if candidate is not None or incumbent is not None:
        candidate_arm = parse_overrides(candidate)
        incumbent_arm = parse_overrides(incumbent)
        if null_test:
            return {NULL_A_NAME: dict(candidate_arm), NULL_B_NAME: dict(candidate_arm)}
        return {
            _arm_name("candidate", candidate_arm): candidate_arm,
            _arm_name("incumbent", incumbent_arm): incumbent_arm,
        }
    if null_test:
        return {NULL_A_NAME: dict(CANDIDATE_ARM), NULL_B_NAME: dict(CANDIDATE_ARM)}
    return {CANDIDATE_NAME: dict(CANDIDATE_ARM), INCUMBENT_NAME: dict(INCUMBENT_ARM)}


def load_our_teams(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Read our packed team files; the name is the file stem without `.packed`."""

    teams = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        teams.append(
            {
                "index": index,
                "name": path.name.removesuffix(".txt").removesuffix(".packed"),
                "team": path.read_text().strip(),
            }
        )
    return teams


def build_units(
    entries: Sequence[dict[str, Any]], our_teams: Sequence[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Mirror mode: one unit per pool team. Asymmetric: one per (opponent, our team)."""

    if not our_teams:
        return [dict(entry) for entry in entries]
    return [
        {
            **entry,
            "our_index": ours["index"],
            "our_name": ours["name"],
            "our_team": ours["team"],
            "unit_id": f"{entry['file']}|{ours['name']}",
        }
        for entry in entries
        for ours in our_teams
    ]


def cluster_rows(unit_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster results: per pool/opponent team (collapses our-team pairs in asym mode)."""

    return merge_cluster_results(unit_rows, "team_file")


def asymmetric_extras(unit_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-our-team results (each clustered by opponent team) for asymmetric runs."""

    names = sorted({str(row["our_team_name"]) for row in unit_rows})
    return {
        "by_our_team": {
            name: aggregate(
                [row for row in unit_rows if row["our_team_name"] == name], f"our:{name}"
            )
            for name in names
        },
        "pair_results": sorted(unit_rows, key=lambda row: str(row["unit_id"])),
    }


def checkpoint_summary(report: dict[str, Any], round_index: int) -> dict[str, Any]:
    overall = report["overall"]
    return {
        "round": round_index,
        "games": overall["games"],
        "win_rate": overall["win_rate"],
        "clustered": overall["clustered"],
        "by_seat": overall.get("by_seat", {}),
    }


def main() -> int:
    args = parse_args()
    asymmetric = args.our_teams is not None
    granule = 4 if asymmetric else 2
    if args.games_per_team < granule or args.games_per_team % granule:
        raise SystemExit(
            f"--games-per-team must be a positive multiple of {granule}"
            + (" in asymmetric mode (both orientations x both seats)" if asymmetric else "")
        )
    if args.futility_min_effect is not None and args.checkpoint_games is None:
        raise SystemExit("--futility-min-effect needs --checkpoint-games")
    entries = load_pool(args.manifest)
    our_teams = load_our_teams(args.our_teams) if asymmetric else None
    units = build_units(entries, our_teams)
    chunks = balanced_chunks(units, args.workers)
    arms = build_arms(args.null_test, args.candidate, args.incumbent)
    chunk_games = None
    if args.checkpoint_games is not None:
        per_unit = args.checkpoint_games // len(units)
        chunk_games = max(granule, per_unit - per_unit % granule)
    rounds = plan_rounds(args.games_per_team, chunk_games, granule)
    total_games = len(units) * args.games_per_team
    mode = "A/A NULL TEST (both arms identical)" if args.null_test else "A/B gate"
    if asymmetric:
        mode += f", asymmetric: {len(our_teams)} our team(s) x {len(entries)} opponent teams"
    print(
        f"{mode}: {total_games} games across {len(units)} units with {len(chunks)} workers"
        f" in {len(rounds)} round(s)",
        flush=True,
    )
    (first_name, first_overrides), (second_name, second_overrides) = arms.items()
    meta = {
        "null_test": args.null_test,
        "candidate": {"name": first_name, **first_overrides},
        "incumbent": {"name": second_name, **second_overrides},
        "manifest": str(args.manifest),
        "our_teams": [str(path) for path in args.our_teams] if asymmetric else None,
        "seed": args.seed,
        "games_per_team": args.games_per_team,
        "round_games_per_unit": rounds,
        "workers": len(chunks),
        "team_preview": "each bot's normal heuristic preview",
        "clustering": ("opponent team (pooled over our teams)" if asymmetric else "pool team"),
        "futility_min_effect": args.futility_min_effect,
        "min_gain": args.min_gain,
    }
    all_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    stopped_for_futility = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        for round_index, games in enumerate(rounds):
            futures = [
                executor.submit(_evaluate_chunk, chunk, games, args.seed, arms, round_index)
                for chunk in chunks
            ]
            for future in futures:
                all_rows.extend(future.result())
            unit_rows = merge_cluster_results(
                all_rows, "unit_id", ("on_our_team_games", "on_our_team_wins")
            )
            clusters_now = cluster_rows(unit_rows)
            report = build_report(clusters_now, args.min_gain)
            history.append(checkpoint_summary(report, round_index + 1))
            last = round_index == len(rounds) - 1
            overall = report["overall"]
            print(
                f"round {round_index + 1}/{len(rounds)}: {overall['wins']}/{overall['games']}"
                f" = {overall['win_rate']:.3f}, cluster-robust "
                f"[{overall['clustered'][0]:.3f}, {overall['clustered'][1]:.3f}]",
                flush=True,
            )
            if not last and args.futility_min_effect is not None:
                stopped_for_futility = futility_stop(
                    [(int(r["p1_wins"]), int(r["games"])) for r in clusters_now],
                    args.futility_min_effect,
                )
            final = last or stopped_for_futility
            report.update(meta)
            report.update(
                {
                    "interim": not final,
                    "stopped_for_futility": stopped_for_futility,
                    "checkpoints": history,
                    # Success is only ever declared at the fixed end of the run.
                    "passed": bool(report["passed"]) and last and not stopped_for_futility,
                }
            )
            if asymmetric:
                report.update(asymmetric_extras(unit_rows))
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
            if stopped_for_futility:
                print(
                    f"FUTILITY STOP after round {round_index + 1}: CI upper bound "
                    f"{overall['clustered'][1]:.3f} < {0.5 + args.futility_min_effect:.3f}",
                    flush=True,
                )
                break

    print(
        f"overall: {overall['wins']}/{overall['games']} = {overall['win_rate']:.3f}\n"
        f"  cluster-robust 95% CI [{overall['clustered'][0]:.3f}, "
        f"{overall['clustered'][1]:.3f}]  <- the gate decides on this\n"
        f"  naive Wilson    95% CI [{overall['wilson'][0]:.3f}, {overall['wilson'][1]:.3f}]"
        f"  (design effect {overall['design_effect']:.2f}x)",
        flush=True,
    )
    for seat, seat_result in overall.get("by_seat", {}).items():
        print(
            f"  as {seat}: {seat_result['wins']}/{seat_result['games']} = "
            f"{seat_result['win_rate']:.3f}, cluster-robust "
            f"[{seat_result['clustered'][0]:.3f}, {seat_result['clustered'][1]:.3f}]",
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
    for name, result in report.get("by_our_team", {}).items():
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
