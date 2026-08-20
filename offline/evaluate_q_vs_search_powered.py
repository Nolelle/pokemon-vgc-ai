#!/usr/bin/env python
"""Compare the action-value model against existing search on a powered validation set.

The original 24-team development-validation reserve has a cluster-robust standard-error
floor of ~0.035 in regret units -- roughly six times the effect it was being asked to
certify -- so every previous "mixed result" was as likely to be the yardstick as the
model.  This scores the same comparison on a larger, hash-disjoint team reserve and
reports cluster-robust intervals plus the power floor, so a null result can be read as
"no effect" rather than "no resolution".

Regret is per root: the value of the best available action minus the value of the action
a method picked, both under the collected uniform-response aggregate.  Lower is better.
Positive `search minus model` means the model is the better chooser.

Nothing here is wired into gameplay; the model remains default-off.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vgc.config import RUNS_DIR
from vgc.evaluation import clustered_mean, holm_rejections, student_t_cdf
from vgc.rl.response_weighting import build_root_tables

DEFAULT_OUT = RUNS_DIR / "counterfactual_q" / "powered_validation" / "q_vs_search.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, nargs="+", required=True)
    parser.add_argument("--label", nargs="+", help="One label per --model.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


def _load(paths: list[Path]) -> list[Any]:
    from vgc.rl.counterfactual_q import load_q_dataset

    samples: list[Any] = []
    for index, path in enumerate(paths):
        shard, metadata = load_q_dataset(path)
        if metadata.get("team_partition") != "development_validation":
            raise SystemExit(f"{path} is not development_validation data")
        samples.extend(
            replace(
                sample,
                root_id=f"p{index:02d}:{sample.root_id}",
                battle_id=f"p{index:02d}:{sample.battle_id}",
            )
            for sample in shard
        )
    return samples


def _predictions(model_path: Path, samples: list[Any], *, device: str) -> dict[tuple, float]:
    import torch

    from vgc.rl.counterfactual_q import ActionResponseValueNet, predict_q
    from vgc.rl.opponents import load_snapshot

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "vgc-action-response-value-v1":
        raise SystemExit(f"unsupported checkpoint {payload.get('format')!r}")
    model = ActionResponseValueNet(load_snapshot(Path(payload["base_checkpoint"]), device=device))
    model.load_state_dict(payload["model_state_dict"])
    values = predict_q(model, samples, batch_size=512, device=device)
    return {
        (sample.root_id, sample.own_action, sample.opponent_action): float(values[index])
        for index, sample in enumerate(samples)
    }


def _paired_report(differences: list[float], keys: list[str], label: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, key in zip(differences, keys):
        grouped[key].append(value)
    summary = clustered_mean(sorted(grouped.items()))
    low, high = summary.interval()
    t_statistic = summary.mean / summary.clustered_se if summary.clustered_se > 0 else 0.0
    p_value = 2.0 * (1.0 - student_t_cdf(abs(t_statistic), max(summary.clusters - 1, 1)))
    return {
        "cluster_unit": label,
        "clusters": summary.clusters,
        "roots": summary.items,
        "mean_difference": summary.mean,
        "clustered_se": summary.clustered_se,
        "naive_se": summary.naive_se,
        "design_effect": summary.design_effect,
        "confidence_interval_95": [low, high],
        "t_statistic": t_statistic,
        "p_value": p_value,
        "significant": bool(low > 0.0 or high < 0.0),
        "between_cluster_sd_tau": summary.tau,
        "within_cluster_sd": summary.within_sd,
        "se_floor_at_this_cluster_count": summary.se_floor,
        "minimum_detectable_effect": summary.minimum_detectable_effect,
        "floor_detectable_effect": summary.floor_detectable_effect,
        "clusters_needed_for_0.03": summary.clusters_needed(0.03, 4),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.label is not None and len(args.label) != len(args.model):
        raise SystemExit("--label must provide exactly one label per --model")
    samples = _load(args.dataset)
    tables, skipped = build_root_tables(samples)
    if not tables:
        raise SystemExit("no usable roots")

    truths = {t.root_id: t.outcomes.mean(axis=1) for t in tables}
    search_pick = {t.root_id: int(t.search_scores.argmax()) for t in tables}
    search_regret = [
        float(truths[t.root_id].max() - truths[t.root_id][search_pick[t.root_id]]) for t in tables
    ]
    search_correct = [
        bool(truths[t.root_id][search_pick[t.root_id]] >= truths[t.root_id].max() - 1e-12)
        for t in tables
    ]

    report: dict[str, Any] = {
        "schema": "vgc-q-vs-search-powered-v1",
        "datasets": [str(path.resolve()) for path in args.dataset],
        "roots": len(tables),
        "skipped_roots": skipped,
        "teams": len({team for t in tables for team in (t.team_id, t.opponent_team_id)}),
        "matchups": len({tuple(sorted((t.team_id, t.opponent_team_id))) for t in tables}),
        "search_baseline": {
            "mean_regret": float(np.mean(search_regret)),
            "top_action_accuracy": float(np.mean(search_correct)),
        },
        "search_integration": False,
        "promotion_authority": False,
        "models": [],
    }

    pvalues: list[float] = []
    for index, model_path in enumerate(args.model):
        label = args.label[index] if args.label else model_path.parent.name
        lookup = _predictions(model_path, samples, device=args.device)
        regrets: list[float] = []
        correct: list[bool] = []
        differences: list[float] = []
        for table in tables:
            predicted = np.asarray(
                [
                    [lookup[(table.root_id, action, response)] for response in table.responses]
                    for action in table.own_actions
                ]
            )
            truth = truths[table.root_id]
            pick = int(predicted.mean(axis=1).argmax())
            regrets.append(float(truth.max() - truth[pick]))
            correct.append(bool(truth[pick] >= truth.max() - 1e-12))
            differences.append(float(truth.max() - truth[search_pick[table.root_id]]) - regrets[-1])
        by_team = _paired_report(differences, [t.team_id for t in tables], "our team")
        by_matchup = _paired_report(
            differences,
            [
                f"{min(t.team_id, t.opponent_team_id)}|{max(t.team_id, t.opponent_team_id)}"
                for t in tables
            ],
            "matchup",
        )
        disagree = [i for i in range(len(tables)) if correct[i] != search_correct[i]]
        strata: dict[str, Any] = {}
        for bucket in sorted({t.turn_bucket for t in tables}):
            picked = [i for i, t in enumerate(tables) if t.turn_bucket == bucket]
            strata[bucket] = _paired_report(
                [differences[i] for i in picked],
                [tables[i].team_id for i in picked],
                "our team",
            )
        pvalues.append(by_team["p_value"])
        report["models"].append(
            {
                "label": label,
                "checkpoint": str(model_path.resolve()),
                "mean_regret": float(np.mean(regrets)),
                "top_action_accuracy": float(np.mean(correct)),
                "paired_vs_search_by_team": by_team,
                "paired_vs_search_by_matchup": by_matchup,
                "paired_vs_search_by_stratum": strata,
                "disagreements": {
                    "roots": len(disagree),
                    "model_correct": sum(1 for i in disagree if correct[i]),
                    "search_correct": sum(1 for i in disagree if search_correct[i]),
                },
            }
        )

    # Three seeds tested against one baseline is a family; correct for it.
    rejections = holm_rejections(pvalues)
    for entry, rejected in zip(report["models"], rejections):
        entry["holm_significant_across_seeds"] = bool(rejected)
    report["holm_family_size"] = len(pvalues)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    base = report["search_baseline"]
    print(
        f"roots={report['roots']} teams={report['teams']} matchups={report['matchups']}\n"
        f"search: regret {base['mean_regret']:.4f}, accuracy {base['top_action_accuracy']:.1%}"
    )
    for entry in report["models"]:
        team = entry["paired_vs_search_by_team"]
        print(
            f"\n{entry['label']}: regret {entry['mean_regret']:.4f}, "
            f"accuracy {entry['top_action_accuracy']:.1%}\n"
            f"  search - model = {team['mean_difference']:+.4f} "
            f"[{team['confidence_interval_95'][0]:+.4f}, {team['confidence_interval_95'][1]:+.4f}] "
            f"p={team['p_value']:.3f} {'SIGNIFICANT' if team['significant'] else 'not significant'}\n"
            f"  power: this run can certify {team['minimum_detectable_effect']:.4f}; "
            f"{team['clusters']}-team floor is {team['floor_detectable_effect']:.4f}\n"
            f"  disagreements: model {entry['disagreements']['model_correct']} / "
            f"search {entry['disagreements']['search_correct']} of {entry['disagreements']['roots']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
