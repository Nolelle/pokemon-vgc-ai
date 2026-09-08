#!/usr/bin/env python
"""Classify the pure-RL scaling curve: still climbing, plateaued, or not transferring.

Consumes the aggregate JSONL written by `offline/evaluate_scaling_milestones.py` (plus,
for paired comparisons, the per-game record files it references) and applies the
PRE-REGISTERED decision rules fixed before launch:

- CASE A (keep scaling): holdout win rate vs full `vgc` gains >= +3 percentage points
  from `--from-games` to `--to-games`, with a team-bootstrap 95% interval excluding
  zero, replicated in at least `--required-seeds` seeds.
- CASE B (plateau): every search rung (`vgc_myopic`, `vgc_shallow`, `vgc`) within
  +/-2 points of its earlier value at the later anchor across ALL seeds.
- CASE C (generalization wall): overall TRAIN win rate gains >= +5 points while the
  HOLDOUT overall gain stays <= +1 point (learning more, transferring none).

Paired deltas join the two checkpoints' per-game rows on game index under the frozen
evaluation schedule, which removes most of the schedule variance an unpaired
comparison would drown in. Team-bootstrap resampling supplies the cluster-aware
uncertainty (games within one team are not independent coin flips).
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

SEARCH_RUNGS = ("vgc_myopic", "vgc_shallow", "vgc")
INCUMBENT_RUNG = "vgc"


def load_rows(paths: Path | list[Path]) -> list[dict]:
    """Load one or more per-seed milestone-evaluation JSONL files."""

    if isinstance(paths, Path):
        paths = [paths]
    rows = []
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nearest_level(levels: list[int], target: int, tolerance: int) -> int | None:
    """Closest evaluated games count to `target`, or None outside `tolerance`."""

    if not levels:
        return None
    best = min(levels, key=lambda level: abs(level - target))
    return best if abs(best - target) <= tolerance else None


def load_opponent_records(path: Path, opponent: str) -> dict[int, tuple[str, str, float]]:
    """game_index -> (team, side, win) for one opponent from a per-game record file."""

    if not path.exists():
        raise SystemExit(f"missing game-records file needed for paired deltas: {path}")
    records: dict[int, tuple[str, str, float]] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["opponent"] != opponent:
            continue
        records[int(row["game_index"])] = (
            row["team"],
            row["side"],
            float(row["win"]),
        )
    return records


def paired_delta_by_team(
    early_records: dict[int, tuple[str, str, float]],
    later_records: dict[int, tuple[str, str, float]],
) -> dict[str, list[float]]:
    """Per-team win-difference lists joined on game index (schedule must be identical)."""

    diffs: dict[str, list[float]] = defaultdict(list)
    for game_index, (team, side, early_win) in early_records.items():
        later = later_records.get(game_index)
        if later is None:
            continue
        later_team, later_side, later_win = later
        if later_team != team or later_side != side:
            raise SystemExit(
                f"frozen schedule mismatch at game {game_index}: "
                f"{(team, side)} vs {(later_team, later_side)}"
            )
        diffs[team].append(later_win - early_win)
    if not diffs:
        raise SystemExit("no joinable paired games between the two checkpoints")
    return dict(diffs)


def bootstrap_delta_interval(
    diffs_by_team: dict[str, list[float]],
    *,
    replicates: int = 2000,
    seed: int = 20260815,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Mean paired delta with a team-clustered bootstrap percentile interval."""

    teams = sorted(diffs_by_team)
    flat = [diff for values in diffs_by_team.values() for diff in values]
    point = sum(flat) / len(flat)
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        sample = [
            diff
            for _team in range(len(teams))
            for diff in diffs_by_team[teams[rng.randrange(len(teams))]]
        ]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    low = means[max(0, int(math.ceil(alpha * replicates)) - 1)]
    high = means[min(replicates - 1, int(math.floor((1 - alpha) * replicates)))]
    return point, low, high


def fit_log_slope(points: list[tuple[float, float]]) -> tuple[float, float]:
    """OLS slope (win-rate change per decade of games) with its standard error."""

    n = len(points)
    if n < 3:
        return float("nan"), float("nan")
    xs = [math.log10(games) for games, _rate in points]
    ys = [rate for _games, rate in points]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(value**2 for value in residuals)
    variance = sse / (n - 2) if n > 2 else 0.0
    se = math.sqrt(variance / sxx) if sxx > 0 else float("nan")
    return slope, se


def group_rates(
    rows: list[dict],
) -> dict[tuple[str, str], dict[int, dict[str, float]]]:
    """(split, opponent) -> games_seen -> seed_label -> win rate."""

    grouped: dict[tuple[str, str], dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        split = row["split"]
        games = int(row["games_seen"])
        label = row.get("seed_label") or row["checkpoint"]
        for opponent, rate in row["result"]["by_opponent"].items():
            grouped[(split, opponent)][games][label] = float(rate)
    return grouped


def print_table(grouped: dict, seed_labels: set[str]) -> None:
    for (split, opponent), by_games in sorted(grouped.items()):
        print(f"[{split}] {opponent}")
        for games in sorted(by_games):
            rates = by_games[games]
            pooled = sum(rates.values()) / len(rates)
            detail = ", ".join(
                f"{label}={rates[label]:.3f}" for label in sorted(seed_labels) if label in rates
            )
            print(f"    {games:>7}  pooled={pooled:.3f}  ({detail})")


def paired_anchor_delta(
    row_early: dict,
    row_late: dict,
    opponent: str,
) -> tuple[float, float, float]:
    """Point/low/high for one seed's paired delta between two anchor checkpoints."""

    early = load_opponent_records(Path(row_early["game_records"]), opponent)
    late = load_opponent_records(Path(row_late["game_records"]), opponent)
    diffs = paired_delta_by_team(early, late)
    return bootstrap_delta_interval(diffs)


def find_row(rows: list[dict], seed_label: str, split: str, games: int) -> dict | None:
    for row in rows:
        if (
            (row.get("seed_label") or "") == seed_label
            and row["split"] == split
            and int(row["games_seen"]) == games
        ):
            return row
    return None


def classify(
    rows: list[dict],
    *,
    from_games: int,
    to_games: int,
    required_seeds: int,
    case_a_points: float,
    case_b_tolerance: float,
    tolerance_games: int,
) -> dict:
    seed_labels = sorted({row.get("seed_label") or "" for row in rows})
    findings: dict[str, object] = {
        "anchors": {"from_games": from_games, "to_games": to_games},
        "seed_labels": seed_labels,
    }

    # --- Case A: paired holdout-vs-incumbent improvement, replicated ---
    case_a_seeds: dict[str, dict] = {}
    for label in seed_labels:
        early = find_row(rows, label, "holdout", from_games)
        late = find_row(rows, label, "holdout", to_games)
        if early is None or late is None:
            continue
        if INCUMBENT_RUNG not in early["result"]["by_opponent"]:
            continue
        try:
            point, low, high = paired_anchor_delta(early, late, INCUMBENT_RUNG)
        except SystemExit as error:
            case_a_seeds[label] = {"error": str(error)}
            continue
        case_a_seeds[label] = {
            "delta": point,
            "interval_low": low,
            "interval_high": high,
            "significant_gain": bool(point >= case_a_points and low > 0.0),
        }
    significant = [
        label
        for label, value in case_a_seeds.items()
        if isinstance(value, dict) and value.get("significant_gain")
    ]
    case_a = len(significant) >= required_seeds
    findings["case_a"] = {
        "rule": (
            f"holdout vs {INCUMBENT_RUNG} paired delta >= +{case_a_points:.2f} with "
            f"95% interval excluding 0, in >= {required_seeds} seeds"
        ),
        "per_seed": case_a_seeds,
        "significant_seeds": significant,
        "matched": case_a,
    }

    # --- Case B: every search rung flat between anchors, in all seeds ---
    grouped = group_rates(rows)
    case_b_seeds: dict[str, dict] = {}
    for label in seed_labels:
        per_rung = {}
        flat_all = True
        for rung in SEARCH_RUNGS:
            early_rate = grouped.get(("holdout", rung), {}).get(from_games, {}).get(label)
            late_rate = grouped.get(("holdout", rung), {}).get(to_games, {}).get(label)
            if early_rate is None or late_rate is None:
                flat_all = False
                continue
            delta = round(late_rate - early_rate, 6)
            per_rung[rung] = delta
            if abs(delta) > case_b_tolerance:
                flat_all = False
        if per_rung:
            case_b_seeds[label] = {"deltas": per_rung, "flat_all_rungs": flat_all}
    case_b = bool(case_b_seeds) and all(value["flat_all_rungs"] for value in case_b_seeds.values())
    findings["case_b"] = {
        "rule": (
            f"every search rung within +/-{case_b_tolerance:.2f} between "
            f"{from_games} and {to_games} in ALL seeds"
        ),
        "per_seed": case_b_seeds,
        "matched": case_b,
    }

    # --- Case C: training-team learning without unseen-team transfer ---
    def overall_gain(split: str) -> float | None:
        deltas = []
        for label in seed_labels:
            early = find_row(rows, label, split, from_games)
            late = find_row(rows, label, split, to_games)
            if early is None or late is None:
                continue
            # Rounded so 0.31 - 0.30 counts as exactly +0.01, not +0.01000...009.
            deltas.append(round(late["result"]["win_rate"] - early["result"]["win_rate"], 6))
        return sum(deltas) / len(deltas) if deltas else None

    train_gain = overall_gain("train")
    holdout_gain = overall_gain("holdout")
    case_c = (
        train_gain is not None
        and holdout_gain is not None
        and train_gain >= 0.05
        and holdout_gain <= 0.01
        and (train_gain - holdout_gain) >= 0.04
    )
    findings["case_c"] = {
        "rule": "overall train gain >= +0.05 while overall holdout gain <= +0.01",
        "train_gain": train_gain,
        "holdout_gain": holdout_gain,
        "matched": case_c,
    }

    if case_a:
        verdict = "CASE_A_KEEP_SCALING"
    elif case_c:
        verdict = "CASE_C_GENERALIZATION_WALL"
    elif case_b:
        verdict = "CASE_B_PLATEAU"
    else:
        verdict = "MIXED_OR_INSUFFICIENT_EVIDENCE"
    findings["verdict"] = verdict

    # Descriptive log-log slopes per split/rung (not the basis of the verdict).
    slopes = {}
    for (split, opponent), by_games in grouped.items():
        points = [
            (games, sum(rates.values()) / len(rates))
            for games, rates in by_games.items()
            if games > 0
        ]
        slope, se = fit_log_slope(points)
        slopes[f"{split}/{opponent}"] = {
            "slope_per_decade": None if math.isnan(slope) else round(slope, 4),
            "slope_se": None if math.isnan(se) else round(se, 4),
        }
    findings["descriptive_log_slopes"] = slopes
    findings["tolerance_note"] = f"anchor levels resolved within +/-{tolerance_games} games"
    return findings


def resolve_anchor_levels(
    rows: list[dict], from_target: int, to_target: int, tolerance: int
) -> tuple[int, int]:
    levels = sorted({int(row["games_seen"]) for row in rows})
    start = nearest_level(levels, from_target, tolerance)
    end = nearest_level(levels, to_target, tolerance)
    if start is None or end is None or end <= start:
        raise SystemExit(
            f"cannot resolve anchors {from_target}->{to_target} within +/-{tolerance}; "
            f"available levels: {levels}"
        )
    return start, end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        nargs="+",
        default=Path("runs/full_pipeline/rl_scale/milestone_eval.jsonl"),
        help="one or more per-seed milestone-evaluation JSONL files",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--from-games", type=int, default=25_000)
    parser.add_argument("--to-games", type=int, default=100_000)
    parser.add_argument(
        "--tolerance-games",
        type=int,
        default=8192,
        help="max distance between a requested anchor and an evaluated milestone",
    )
    parser.add_argument("--required-seeds", type=int, default=2)
    parser.add_argument(
        "--case-a-points",
        type=float,
        default=0.03,
        help="minimum holdout-vs-incumbent gain (fraction) for Case A",
    )
    parser.add_argument(
        "--case-b-tolerance",
        type=float,
        default=0.02,
        help="max |gain| per search rung (fraction) consistent with Case B",
    )
    args = parser.parse_args(argv)

    rows = load_rows(args.eval_jsonl)
    if not rows:
        raise SystemExit(f"no evaluation rows found in {args.eval_jsonl}")
    from_games, to_games = resolve_anchor_levels(
        rows, args.from_games, args.to_games, args.tolerance_games
    )

    grouped = group_rates(rows)
    seed_labels = {row.get("seed_label") or "" for row in rows}
    print_table(grouped, seed_labels)

    findings = classify(
        rows,
        from_games=from_games,
        to_games=to_games,
        required_seeds=args.required_seeds,
        case_a_points=args.case_a_points,
        case_b_tolerance=args.case_b_tolerance,
        tolerance_games=args.tolerance_games,
    )
    print("\n=== scaling-curve verdict ===")
    print(json.dumps(findings, indent=2, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(findings, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
