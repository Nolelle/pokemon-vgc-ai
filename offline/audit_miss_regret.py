#!/usr/bin/env python
"""Quantify what shortlist misses actually cost, from recorded shadow/hybrid decisions.

Recall@K counts every miss equally, but a miss whose best retained action scores within
noise of the search winner costs nothing at deployment time. The decision records
written by `NeuralSearchPlayer` (`offline/evaluate_neural_search.py`) already carry that
number -- `known_regret`, the search-score gap between the search winner and the best
action the shortlist actually kept. Scores use the evaluator's currency
(`damage_percent_weight` points per 1% HP), so single-digit regrets are small and
three-digit regrets are blunders. This script aggregates those gaps so the promotion
rule can be argued in value lost per turn, not just set membership.

Censoring note: `known_regret` is computed only against actions the DEFAULT selector
also searched -- a kept-but-unsearched action's true score is unknown, so regret is
measured against whatever overlap exists and therefore UPPER-bounds the kept set's
quality loss... i.e. these numbers are pessimistic, never optimistic. When nothing
overlaps, regret reads as unknown rather than invented.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.evaluation import clustered_interval  # noqa: E402
from vgc.rl.distill import turn_bucket  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "runs" / "eval" / "miss_regret_audit.json"
DEFAULT_THRESHOLDS = (1.0, 5.0, 10.0, 20.0)
_METRIC_RE = re.compile(r"^(neural|guided)_at_(\d+)$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--metric",
        default="neural_at_10",
        help="which retention flavor to audit, e.g. neural_at_10 or guided_at_10",
    )
    parser.add_argument(
        "--negligible-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="regret values at or below which a miss is treated as free",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def action_count_bucket(count: int) -> str:
    """Same bins the offline recall screen reports, so both audits compare directly."""

    if count <= 0:
        return "unknown"
    if count <= 25:
        return "0-25"
    if count <= 100:
        return "26-100"
    if count <= 200:
        return "101-200"
    return "201+"


def _clustered(hits: list[bool], teams: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for hit, team in zip(hits, teams):
        grouped[team].append(hit)
    clusters = [(sum(values), len(values)) for values in grouped.values()]
    low, high = clustered_interval(clusters)
    return {
        "rate": sum(hits) / len(hits) if hits else 0.0,
        "decisions": len(hits),
        "teams": len(clusters),
        "clustered_lower_bound": low,
    }


def summarize_regret(
    records: list[dict[str, Any]],
    *,
    metric: str,
    thresholds: list[float],
) -> dict[str, Any]:
    """Aggregate miss severity for one retention metric over decision records.

    Every rate is reported twice: among misses (is the damage concentrated?) and among
    ALL decisions ("effective retention" -- retained OR cheaply missed). The second is
    the promotion-relevant one: a 4% miss rate where most misses are free is a very
    different object from a clean 4%.
    """

    if not _METRIC_RE.match(metric):
        raise ValueError(
            f"--metric must look like 'neural_at_10' or 'guided_at_8', got {metric!r}"
        )
    teams: list[str] = []
    retained_flags: list[bool] = []
    regrets: list[float | None] = []
    buckets: list[tuple[str, str]] = []  # (action-count bucket, phase bucket)
    skipped_no_metric = 0
    for record in records:
        payload = (record.get("retention") or {}).get(metric)
        if payload is None:
            skipped_no_metric += 1
            continue
        retained_flags.append(bool(payload.get("retained")))
        raw_regret = payload.get("known_regret")
        regrets.append(None if raw_regret is None else float(raw_regret))
        teams.append(str(record.get("team_id") or record.get("battle_id") or "unknown"))
        buckets.append(
            (
                action_count_bucket(int(record.get("legal_action_count") or 0)),
                f"phase:{turn_bucket(int(record.get("turn") or 0))}",
            )
        )

    values = sorted(
        value
        for value, retained in zip(regrets, retained_flags)
        if not retained and value is not None
    )

    def _percentile(percentile: float) -> float | None:
        if not values:
            return None
        position = (len(values) - 1) * percentile / 100.0
        low = int(position)
        high = min(low + 1, len(values) - 1)
        weight = position - low
        return values[low] * (1.0 - weight) + values[high] * weight

    def _negligible(threshold: float) -> dict[str, Any]:
        # A decision is "effectively retained" when the teacher action survived the
        # shortlist OR the best kept action was within `threshold` of it.
        # Unknown-regret misses cannot be excused, so they stay misses here.
        effective = [
            retained or (regret is not None and regret <= threshold)
            for retained, regret in zip(retained_flags, regrets)
        ]
        miss_teams = [team for team, retained in zip(teams, retained_flags) if not retained]
        miss_effective = [
            hit for hit, retained in zip(effective, retained_flags) if not retained
        ]
        return {
            "misses_within_threshold": int(
                sum(
                    1
                    for regret, retained in zip(regrets, retained_flags)
                    if not retained and regret is not None and regret <= threshold
                )
            ),
            "effective_retention_among_misses": _clustered(miss_effective, miss_teams),
            "effective_retention_overall": _clustered(effective, teams),
        }

    summary: dict[str, Any] = {
        "schema": "vgc-miss-regret-audit-v1",
        "metric": metric,
        "decisions": len(retained_flags),
        "records_without_metric": skipped_no_metric,
        "teams": len(set(teams)),
        "retained": int(sum(retained_flags)),
        "misses": len(retained_flags) - int(sum(retained_flags)),
        "misses_with_known_regret": len(values),
        "misses_with_unknown_regret": sum(
            1
            for regret, retained in zip(regrets, retained_flags)
            if not retained and regret is None
        ),
        "regret_distribution": {
            "mean": sum(values) / len(values) if values else None,
            "p50": _percentile(50.0),
            "p95": _percentile(95.0),
            "max": values[-1] if values else None,
        },
        "negligible_thresholds": {},
        "by_action_count_bucket": {},
        "by_phase_bucket": {},
    }
    for threshold in thresholds:
        summary["negligible_thresholds"][f"le_{threshold:g}"] = _negligible(threshold)

    for label_key, out_key in ((0, "by_action_count_bucket"), (1, "by_phase_bucket")):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, bucket in enumerate(buckets):
            groups[bucket[label_key]].append(index)
        for label, indices in sorted(groups.items()):
            sub_retained = [retained_flags[index] for index in indices]
            sub_misses = [index for index in indices if not retained_flags[index]]
            sub_values = sorted(
                regrets[index] for index in sub_misses if regrets[index] is not None
            )
            block: dict[str, Any] = {
                "decisions": len(indices),
                "retained": int(sum(sub_retained)),
                "recall": sum(sub_retained) / len(indices) if indices else 0.0,
                "known_regret_p95": (
                    sub_values[min(len(sub_values) - 1, round(0.95 * (len(sub_values) - 1)))]
                    if sub_values
                    else None
                ),
            }
            for threshold in thresholds:
                free_misses = sum(
                    1
                    for index in sub_misses
                    if regrets[index] is not None and regrets[index] <= threshold
                )
                block[f"le_{threshold:g}_share_of_misses"] = (
                    free_misses / len(sub_misses) if sub_misses else None
                )
            summary[out_key][label] = block
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if any(threshold < 0 for threshold in args.negligible_thresholds):
        raise SystemExit("--negligible-thresholds must be non-negative")
    records = load_records(args.records)
    if not records:
        raise SystemExit("no decision records found")
    summary = summarize_regret(records, metric=args.metric, thresholds=args.negligible_thresholds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    distribution = summary["regret_distribution"]
    print(
        f"{summary['decisions']} decisions / {summary['teams']} teams | "
        f"{summary['retained']} retained, {summary['misses']} missed "
        f"({summary['misses_with_unknown_regret']} without known regret)"
    )
    print(
        f"miss regret: mean {distribution['mean']:.2f}, p50 {distribution['p50']:.2f}, "
        f"p95 {distribution['p95']:.2f}, max {distribution['max']:.2f}"
    )
    print("\neffectively-retained rate (retained OR regret <= threshold):")
    print(f"{'threshold':>10s} {'overall':>18s} {'among misses':>18s}")
    for label, block in summary["negligible_thresholds"].items():
        overall = block["effective_retention_overall"]
        among = block["effective_retention_among_misses"]
        print(
            f"{label:>10s} {overall['rate']:>8.2%} [LCB {overall['clustered_lower_bound']:.3f}] "
            f"{among['rate']:>8.2%} [LCB {among['clustered_lower_bound']:.3f}]"
        )
    print("\nhardest action-count buckets by recall:")
    ranked = sorted(
        summary["by_action_count_bucket"].items(), key=lambda item: item[1]["recall"]
    )
    for label, block in ranked[:4]:
        print(f"  {label:>24s}: recall {block['recall']:.1%} over {block['decisions']} decisions")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
