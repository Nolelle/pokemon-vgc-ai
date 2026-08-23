#!/usr/bin/env python
"""Measure whether a policy can replace the hand-written search shortlist.

The shipped search already prunes ~262 legal joint orders down to
`PolicyConfig.search_our_candidates` (10) with the myopic evaluator, then searches
those. So neural candidate guidance does not remove a pruning step -- it changes WHO
picks the ten. The question it has to answer first is therefore retention, not speed:
if the teacher's action is ranked 11th by the network, search never sees it and no
amount of search quality afterwards recovers it.

Two retention flavors are reported:

``by_k``
    Pure top-K of the network's ranking. Computable on every dataset; this is the fast
    screen.

``by_k_guided`` (--simulate-guided)
    Retention through the DEPLOYED selector (`vgc.rl.guided_selection`), which reserves
    up to half the budget for heuristic safety picks before filling from neural ranks.
    This is the number hybrid mode actually experiences -- a teacher action at neural
    rank 8 can be dropped by the reserve, or one at rank 40 kept via the myopic-leader
    slot. Requires schema-v2 collections that recorded per-candidate myopic ranks and
    tags; older datasets report those decisions as unknown rather than guessing.

Recall@K is reported with a team-clustered interval because decisions inside one battle
(and one roster) are not independent evidence -- the same reason the multi-team gates
and the counterfactual Q work cluster by team. Pooling 9,000 decisions as if they were
independent understates the error badly.

Scope: this measures recall on states a TEACHER visited. The shadow evaluation
(`offline/evaluate_neural_search.py`) measures it on states the guided player itself
reaches, which is the distribution that actually matters at play time. Treat this as the
fast screen and the shadow run as the gate -- a gap between the two is itself the
finding that teacher-state distribution is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vgc.config import RUNS_DIR
from vgc.evaluation import clustered_interval, wilson_interval

DEFAULT_OUT = RUNS_DIR / "eval" / "shortlist_recall.json"
# The shadow gate's bar. Recall below this means the network cannot be trusted as the
# sole shortlist authority, however good its top-3 ordering is.
RECALL_TARGET = 0.98
DEFAULT_KS = (1, 3, 5, 10)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--holdout-teams", type=Path,
        help="JSON list of team labels to score; omit to score every loaded sample.",
    )
    parser.add_argument(
        "--simulate-guided",
        action="store_true",
        help="also replay the deployed safety-slotted selector (needs schema-v2 samples)",
    )
    parser.add_argument(
        "--safety-slots",
        type=int,
        default=4,
        help="safety reserve passed to the guided replay (deployment default: 4)",
    )
    parser.add_argument("--recall-target", type=float, default=RECALL_TARGET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def _clustered(hits: list[bool], teams: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for hit, team in zip(hits, teams):
        grouped[team].append(hit)
    clusters = [(sum(values), len(values)) for values in grouped.values()]
    pooled = float(np.mean(hits)) if hits else 0.0
    low, high = clustered_interval(clusters)
    naive_low, naive_high = wilson_interval(sum(hits), len(hits))
    return {
        "recall": pooled,
        "teams": len(clusters),
        "decisions": len(hits),
        "clustered_interval_95": [low, high],
        "clustered_lower_bound": low,
        "naive_wilson_interval_95": [naive_low, naive_high],
        "clustering_widened_interval_by": (
            (high - low) / (naive_high - naive_low) if naive_high > naive_low else 1.0
        ),
    }


def _strata(samples: list[Any]) -> dict[str, list[int]]:
    from vgc.rl.distill import turn_bucket

    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        turn = int(getattr(sample, "turn", 0) or 0)
        # Turn 0 means the dataset predates the metadata, not "turn zero".
        phase = turn_bucket(turn) if turn > 0 else "unknown"
        groups[f"phase:{phase}"].append(index)
        count = int(getattr(sample, "legal_action_count", 0) or 0)
        if count <= 0:
            label = "unknown"
        elif count <= 25:
            label = "0-25"
        elif count <= 100:
            label = "26-100"
        elif count <= 200:
            label = "101-200"
        else:
            label = "201+"
        groups[f"legal_actions:{label}"].append(index)
    return dict(sorted(groups.items()))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.recall_target <= 0.0 or args.recall_target > 1.0:
        raise SystemExit("--recall-target must be in (0, 1]")

    from vgc.rl.demonstrations import load_demonstrations
    from vgc.rl.distill import recall_at_k
    from vgc.rl.opponents import load_snapshot

    samples: list[Any] = []
    for path in args.dataset:
        samples.extend(load_demonstrations(path))
    if args.holdout_teams is not None:
        keep = {str(label) for label in json.loads(args.holdout_teams.read_text())}
        samples = [s for s in samples if s.team_id in keep]
    if not samples:
        raise SystemExit("no samples to score")
    missing_metadata = sum(1 for s in samples if not getattr(s, "legal_action_count", 0))
    model = load_snapshot(args.checkpoint, device=args.device)
    result = recall_at_k(model, samples, ks=args.ks, batch_size=args.batch_size, device=args.device)
    teams = [str(s.team_id) for s in samples]

    report: dict[str, Any] = {
        "schema": "vgc-shortlist-recall-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "datasets": [str(path.resolve()) for path in args.dataset],
        "decisions": result["samples"],
        "teams": len(set(teams)),
        "samples_missing_stratification_metadata": missing_metadata,
        "teacher_rank": result["teacher_rank"],
        "recall_target": args.recall_target,
        "measured_on": "teacher-visited states (fast screen, not the shadow gate)",
        "by_k": {},
        "by_stratum": {},
    }
    for k in args.ks:
        block = _clustered(result["hits"][str(k)], teams)
        block["trivial_decisions_with_at_most_k_legal_actions"] = result["trivial"][str(k)]
        block["meets_target"] = bool(block["clustered_lower_bound"] >= args.recall_target)
        report["by_k"][str(k)] = block

    if args.simulate_guided:
        from vgc.rl.distill import simulate_guided_hits

        guided_hits = simulate_guided_hits(
            samples, result["ranked_indices"], ks=args.ks, safety_slots=args.safety_slots
        )
        report["by_k_guided"] = {}
        report["guided_safety_slots"] = args.safety_slots
        for k in args.ks:
            flavor = [hit for hit in guided_hits[k] if hit is not None]
            block = _clustered(flavor, teams) if flavor else {
                "recall": 0.0, "teams": 0, "decisions": 0,
                "clustered_interval_95": [0.0, 0.0], "clustered_lower_bound": 0.0,
                "naive_wilson_interval_95": [0.0, 0.0], "clustering_widened_interval_by": 1.0,
            }
            block["decisions_without_guidance_metadata"] = len(guided_hits[k]) - len(flavor)
            block["trivial_decisions_with_at_most_k_legal_actions"] = result["trivial"][str(k)]
            # A pass computed on a metadata-covered SUBSET is not a pass for the gate
            # when the excluded remainder can fail at all -- especially since trivially
            # retained decisions are exactly the ones metadata-less replay CAN score.
            if not flavor or block["decisions_without_guidance_metadata"]:
                block["verdict"] = "INDETERMINATE"
            elif block["clustered_lower_bound"] >= args.recall_target:
                block["verdict"] = "PASS"
            else:
                block["verdict"] = "FAIL"
            block["meets_target"] = block["verdict"] == "PASS"
            report["by_k_guided"][str(k)] = block

    strata = _strata(samples)
    for label, indices in strata.items():
        report["by_stratum"][label] = {
            str(k): _clustered(
                [result["hits"][str(k)][i] for i in indices], [teams[i] for i in indices]
            )
            for k in args.ks
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if missing_metadata:
        print(
            f"note: {missing_metadata}/{len(samples)} samples predate stratification "
            f"metadata -- their phase/action-count strata read as 'unknown'"
        )
    print(
        f"{report['decisions']} decisions / {report['teams']} teams | "
        f"teacher rank median {result['teacher_rank']['median']:.0f}, "
        f"p95 {result['teacher_rank']['p95']:.0f}, max {result['teacher_rank']['max']:.0f}"
    )
    print(f"\n{'K':>4s} {'recall':>8s} {'clustered 95% CI':>22s} {'LCB':>7s} {'trivial':>8s}")
    for k in args.ks:
        b = report["by_k"][str(k)]
        lo, hi = b["clustered_interval_95"]
        print(
            f"{k:>4d} {b['recall']:>8.1%} [{lo:>8.3f},{hi:>8.3f}] {b['clustered_lower_bound']:>7.3f} "
            f"{b['trivial_decisions_with_at_most_k_legal_actions']:>8d}"
        )
    top = max(args.ks)
    verdict = report["by_k"][str(top)]
    print(
        f"\ndecision rule (pure top-{top}): LCB >= {args.recall_target:.2f} -> "
        f"{'PASS' if verdict['meets_target'] else 'FAIL'} "
        f"(LCB {verdict['clustered_lower_bound']:.3f})"
    )
    guided_blocks = report.get("by_k_guided")
    if guided_blocks:
        gverdict = guided_blocks[str(top)]
        unknown = gverdict["decisions_without_guidance_metadata"]
        suffix = (
            f", {unknown} decisions without metadata excluded -- INDETERMINATE"
            if gverdict["verdict"] == "INDETERMINATE"
            else (f", {unknown} without metadata excluded" if unknown else "")
        )
        print(
            f"decision rule (guided, safety_slots={report['guided_safety_slots']}): "
            f"LCB(Recall@{top}) >= {args.recall_target:.2f} -> {gverdict['verdict']} "
            f"(LCB {gverdict['clustered_lower_bound']:.3f}{suffix})"
        )
    print("\nhardest strata by recall@%d:" % top)
    ranked = sorted(report["by_stratum"].items(), key=lambda item: item[1][str(top)]["recall"])
    for label, block in ranked[:4]:
        b = block[str(top)]
        print(f"  {label:>24s}: {b['recall']:.1%} over {b['decisions']} decisions")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
