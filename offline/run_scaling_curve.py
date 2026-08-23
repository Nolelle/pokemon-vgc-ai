#!/usr/bin/env python
"""Run the data-scaling curve behind the shortlist-recall gate.

Trains identical models on nested subsets of one demonstration collection (5%, ...,
100%), then scores every checkpoint on the SAME team-held-out decision set with
`vgc.rl.distill.recall_at_k` and team-clustered intervals. The point of the curve is to
distinguish "recall has plateaued -- stop collecting" from "still climbing -- here is
what more data costs": fitting miss rate ~ N^-alpha in log-log space turns the slope
into a projected data requirement, with the huge caveat that only a couple of degrees
of freedom stand behind it.

Subsets are NESTED (one seeded shuffle of battle ids, prefixes taken), so consecutive
points differ only by added battles. Training is delegated to
`selfplay/train_imitation.py` exactly as a manual run would be; extra flags can be
forwarded verbatim via --extra-train-args (e.g. --select-metric recall_at_k).
"""

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.evaluation import clustered_interval  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "runs" / "full_pipeline" / "scaling_curve"
DEFAULT_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
DEFAULT_KS = (1, 3, 5, 10)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--holdout-dataset", type=Path, required=True)
    parser.add_argument("--holdout-teams", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--extra-train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="forwarded verbatim to train_imitation.py, e.g. "
        "-- --select-metric recall_at_k --hard-example-weight 2.0",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="reuse an existing checkpoint/metrics for a fraction instead of retraining",
    )
    parser.add_argument("--target-recall", type=float, default=0.989,
        help="point estimate the fit projects a data requirement for (LCB target plus "
        "the holdout's own interval gap)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def materialize_nested_subsets(
    dataset_path: Path,
    fractions: list[float],
    *,
    seed: int,
    out_dir: Path,
) -> dict[float, Path]:
    """Write one nested demonstrations file per fraction (prefixes of one shuffle)."""

    from vgc.rl.demonstrations import load_demonstrations, save_demonstrations

    samples = load_demonstrations(dataset_path)
    battle_ids = sorted({sample.battle_id for sample in samples})
    rng = random.Random(seed)
    rng.shuffle(battle_ids)

    paths: dict[float, Path] = {}
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for fraction in fractions:
        keep = set(battle_ids[: max(1, round(len(battle_ids) * fraction))])
        subset = [sample for sample in samples if sample.battle_id in keep]
        path = data_dir / f"frac_{int(round(fraction * 100)):03d}.pt"
        if not path.exists():
            save_demonstrations(path, subset)
        paths[fraction] = path
        print(f"fraction {fraction:.2f}: {len(subset)} decisions / {len(keep)} battles")
    return paths


def train_fraction(
    subset_path: Path,
    model_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path | None, dict]:
    """Train via train_imitation.py; returns (checkpoint, its metrics.json)."""

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / "best.pt"
    metrics_path = model_dir / "metrics.json"
    if args.skip_existing and checkpoint.exists() and metrics_path.exists():
        return checkpoint, json.loads(metrics_path.read_text())

    command = [
        sys.executable,
        str(REPO_ROOT / "selfplay" / "train_imitation.py"),
        "--dataset",
        str(subset_path),
        "--out-dir",
        str(model_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        *args.extra_train_args,
    ]
    print("running:", shlex.join(command))
    # train_imitation exits non-zero when validation teacher probability does not
    # improve, but it SAVES the checkpoint and metrics first -- judge by artifacts.
    subprocess.run(command, cwd=REPO_ROOT, check=False)
    if not checkpoint.exists():
        raise SystemExit(f"training produced no checkpoint in {model_dir}")
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return checkpoint, metrics


def evaluate_checkpoint(
    checkpoint: Path,
    holdout_samples: list,
    *,
    ks: list[int],
    device: str,
) -> dict:
    from vgc.evaluation import wilson_interval
    from vgc.rl.distill import recall_at_k
    from vgc.rl.opponents import load_snapshot

    model = load_snapshot(checkpoint, device=device)
    result = recall_at_k(model, holdout_samples, ks=ks, batch_size=256, device=device)
    teams = [str(sample.team_id) for sample in holdout_samples]
    grouped: dict[str, list[bool]] = defaultdict(list)
    for team, hit in zip(teams, result["hits"][str(max(ks))]):
        grouped[team].append(hit)

    top_block: dict = {}
    low, high = clustered_interval([(sum(v), len(v)) for v in grouped.values()])
    naive_low, naive_high = wilson_interval(sum(map(bool, result["hits"][str(max(ks))])), len(holdout_samples))
    top_block.update(
        recall=result["recall"][str(max(ks))],
        clustered_lower_bound=low,
        clustered_upper_bound=high,
        naive_wilson_lower_bound=naive_low,
    )
    return {
        "decisions": len(holdout_samples),
        "teams": len(grouped),
        "teacher_rank": result["teacher_rank"],
        "by_k": {
            str(k): {
                "recall": result["recall"][str(k)],
                "trivial": result["trivial"][str(k)],
            }
            for k in ks
        },
        "top_k_clustered": top_block,
    }


def fit_power_law(points: list[tuple[float, float]]) -> dict:
    """Log-log OLS of miss rate against decisions -> alpha, R^2, projection."""

    import math

    usable = [(n, miss) for n, miss in points if n > 0 and miss > 0]
    if len(usable) < 3:
        return {"slope_alpha": None, "r_squared": None}
    xs = [math.log(n) for n, _ in usable]
    ys = [math.log(miss) for _, miss in usable]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        # Every point at the same data volume -- nothing to fit (e.g. someone passed
        # holdout sizes instead of training sizes).
        return {"slope_alpha": None, "r_squared": None}
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    ss_res = sum(r**2 for r in residuals)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    return {
        "slope_alpha": slope,
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
        "points_used": len(usable),
        "note": "a handful of nested, correlated points; treat projections as order of "
        "magnitude only, never as a budget",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.target_recall < 1:
        raise SystemExit("--target-recall must be in (0, 1)")
    # REMAINDER swallows everything after --extra-train-args, including any later
    # driver-owned flags; make that failure loud instead of silently retargeting the
    # training runs.
    reserved = {"--dataset", "--out-dir", "--seed", "--device"}
    for flag in reserved:
        if flag in args.extra_train_args:
            raise SystemExit(
                f"{flag} must be given to run_scaling_curve itself, not forwarded "
                f"via --extra-train-args (extras must come last on the command line)"
            )

    from vgc.rl.demonstrations import load_demonstrations

    holdout_teams = set(json.loads(args.holdout_teams.read_text()))
    holdout_samples = [
        sample
        for sample in load_demonstrations(args.holdout_dataset)
        if str(sample.team_id) in holdout_teams
    ]
    if not holdout_samples:
        raise SystemExit("holdout filter matched no decisions")
    print(f"holdout: {len(holdout_samples)} decisions / {len(holdout_teams)} teams")

    subsets = materialize_nested_subsets(
        args.dataset, sorted(args.fractions), seed=args.seed, out_dir=args.out_dir
    )

    curve: list[dict] = []
    for fraction in sorted(args.fractions):
        label = f"frac_{int(round(fraction * 100)):03d}"
        checkpoint, train_metrics = train_fraction(
            subsets[fraction], args.out_dir / f"model_{label}", args
        )
        assert checkpoint is not None
        evaluation = evaluate_checkpoint(checkpoint, holdout_samples, ks=args.ks, device=args.device)
        entry = {
            "fraction": fraction,
            # The scaling axis is how much data the model TRAINED on -- every
            # checkpoint is scored on the same holdout, so holdout size carries no
            # information about the curve.
            "trained_decisions": train_metrics.get("sample_count"),
            "checkpoint": str(checkpoint),
            "training_best_val_recall_at_10": (
                train_metrics.get("training", {}).get("best_val_recall_at_10")
                if isinstance(train_metrics.get("training"), dict)
                else None
            ),
            **evaluation,
        }
        curve.append(entry)
        top = entry["top_k_clustered"]
        print(
            f"{label}: trained on {entry['trained_decisions']} decisions | "
            f"R@{max(args.ks)} {top['recall']:.1%} "
            f"(LCB {top['clustered_lower_bound']:.3f}) over {entry['decisions']} holdout decisions"
        )

    top_k = str(max(args.ks))
    points = [
        (entry["trained_decisions"] or 0, 1.0 - entry["by_k"][top_k]["recall"])
        for entry in curve
    ]
    fit = fit_power_law(points)
    final_miss = 1.0 - curve[-1]["by_k"][top_k]["recall"]
    if fit.get("slope_alpha") is not None and final_miss > 0:
        needed_ratio = ((1.0 - args.target_recall) / final_miss) ** (1.0 / fit["slope_alpha"])
        fit["multiplier_to_target"] = needed_ratio
        fit["projected_decisions_to_target"] = (
            curve[-1]["trained_decisions"] * needed_ratio
        )
        fit["target_recall"] = args.target_recall

    report = {
        "schema": "vgc-shortlist-scaling-curve-v1",
        "dataset": str(args.dataset.resolve()),
        "holdout_dataset": str(args.holdout_dataset.resolve()),
        "holdout_teams": str(args.holdout_teams.resolve()),
        "train_command_tail": args.extra_train_args,
        "curve": curve,
        "power_law_fit": fit,
    }
    out_path = args.out_dir / "scaling_curve.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print("\ntrained_on    holdout   R@10     LCB     misses")
    for entry in curve:
        top = entry["top_k_clustered"]
        print(
            f"{entry['trained_decisions'] or 0:>10d}  {entry['decisions']:>8d}   "
            f"{top['recall']:>7.1%}  {top['clustered_lower_bound']:.3f}"
            f"   {1.0 - entry['by_k'][top_k]['recall']:.1%}"
        )
    if fit.get("slope_alpha") is not None:
        print(
            f"\nfit: miss ~ N^{fit['slope_alpha']:.2f} (R^2 {fit['r_squared']:.3f})"
        )
        if "projected_decisions_to_target" in fit:
            print(
                f"projection to {args.target_recall:.1%}: x{fit['multiplier_to_target']:.1f} "
                f"data (~{fit['projected_decisions_to_target']:.0f} decisions)"
            )
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
