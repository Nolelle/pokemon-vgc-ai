#!/usr/bin/env python
"""Test whether logit-averaging checkpoints beats every member on shortlist recall.

Ensembling is the cheapest possible recall lever -- no retraining, just K forward
passes averaged per decision. If two models disagree mostly on near-ties, their mean
ranks the teacher's action more stably; if they fail on the same states, it changes
nothing. This script answers that in one pass over the team-held-out decisions,
reporting each member and the ensemble through the same recall_at_k + clustered-interval
machinery as the single-model screen.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from vgc.evaluation import clustered_interval  # noqa: E402
from vgc.rl.distill import recall_at_k  # noqa: E402
from vgc.rl.opponents import load_snapshot  # noqa: E402


class LogitAveragedEnsemble:
    """Same forward contract as CandidatePolicyValueNet; logits/values are member means."""

    def __init__(self, members: list) -> None:
        flags = {
            ("use_meta_features", m.use_meta_features) for m in members
        } | {("use_information_features", m.use_information_features) for m in members} | {
            ("use_tactical_features", m.use_tactical_features) for m in members
        }
        if len(flags) != 3:
            raise ValueError("ensemble members must agree on feature flags")
        self.members = torch.nn.ModuleList(members)
        self.use_meta_features = members[0].use_meta_features
        self.use_information_features = members[0].use_information_features
        self.use_tactical_features = members[0].use_tactical_features
        self.eval()

    def eval(self) -> None:
        for member in self.members:
            member.eval()

    def forward(self, *args, **kwargs):
        logits = []
        values = []
        for member in self.members:
            member_logits, member_values = member(*args, **kwargs)
            logits.append(member_logits)
            values.append(member_values)
        return torch.stack(logits).mean(dim=0), torch.stack(values).mean(dim=0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--holdout-teams", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--recall-target", type=float, default=0.98)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def _clustered(hits: list[bool], teams: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for hit, team in zip(hits, teams):
        grouped[team].append(hit)
    low, high = clustered_interval([(sum(v), len(v)) for v in grouped.values()])
    top = max(len(v) for v in grouped.values())
    return {
        "recall": sum(hits) / len(hits),
        "decisions": len(hits),
        "teams": len(grouped),
        "clustered_lower_bound": low,
        "max_cluster_decisions": top,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from vgc.rl.demonstrations import load_demonstrations

    keep = set(json.loads(args.holdout_teams.read_text()))
    samples = [
        s for s in load_demonstrations(args.dataset) if str(s.team_id) in keep
    ]
    if not samples:
        raise SystemExit("holdout filter matched no decisions")
    teams = [str(s.team_id) for s in samples]

    members = [load_snapshot(path, device=args.device) for path in args.checkpoint]
    report: dict[str, Any] = {
        "schema": "vgc-shortlist-ensemble-recall-v1",
        "members": [str(path) for path in args.checkpoint],
        "decisions": len(samples),
        "teams": len(set(teams)),
        "by_member": {},
        "ensemble": {},
    }

    for path, member in zip(args.checkpoint, members):
        result = recall_at_k(member, samples, ks=args.ks, batch_size=args.batch_size, device=args.device)
        report["by_member"][path.name] = {
            k: _clustered(result["hits"][k], teams) for k in result["hits"]
        }
        report["by_member"][path.name]["teacher_rank"] = result["teacher_rank"]

    ensemble = LogitAveragedEnsemble(members)
    result = recall_at_k(ensemble, samples, ks=args.ks, batch_size=args.batch_size, device=args.device)
    report["ensemble"] = {
        k: _clustered(result["hits"][k], teams) for k in result["hits"]
    }
    report["ensemble"]["teacher_rank"] = result["teacher_rank"]

    top_k = str(max(args.ks))
    print(f"{len(samples)} decisions / {len(set(teams))} teams | Recall@{top_k}:")
    for name, block in report["by_member"].items():
        b = block[top_k]
        print(f"  member {name:>24s}: {b['recall']:.1%} (LCB {b['clustered_lower_bound']:.3f})")
        del b
    ens = report["ensemble"][top_k]
    verdict = "PASS" if ens["clustered_lower_bound"] >= args.recall_target else "FAIL"
    print(
        f"  {'ENSEMBLE':>24s}: {ens['recall']:.1%} (LCB {ens['clustered_lower_bound']:.3f}) "
        f"-> {verdict} vs target {args.recall_target:.2f}"
    )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
