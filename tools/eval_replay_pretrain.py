#!/usr/bin/env python
"""Score trained replay-pretraining checkpoints on a held-out split.

Runs the trainer's own evaluators (evaluate_turn / evaluate_preview) plus a
value bias-by-turn table on the validation split repeatably, or on the sealed
test split via the deliberate `--split test --final-test` double gate (which
also appends an access-log line). Test caches are never written to disk.

    .venv/bin/python tools/eval_replay_pretrain.py \
        --prepared data/bc/pretraining_regmc_s15 --run runs/pilot_s15_2 \
        --split validation --out runs/pilot_s15_2/eval_validation.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from train_replay_pretrain import (  # noqa: E402
    build_baselines,
    build_cache,
    evaluate_preview,
    evaluate_turn,
)
from vgc.bc.encoding import trunk_inputs  # noqa: E402
from vgc.bc.replay_pretrain import PreviewPretrainNet, TurnPretrainNet  # noqa: E402
from vgc.bc.replay_pretraining import file_hash, iter_task_examples  # noqa: E402
from vgc.model_vocabulary import VOCABULARY_KEY, current_vocabulary  # noqa: E402

_BIAS_BATCH = 4096


def _load_checkpoint(path: Path, device, summary_sha: str) -> dict:
    ck = torch.load(path, map_location=device)
    assert ck.get(VOCABULARY_KEY) == current_vocabulary(), (
        f"checkpoint vocabulary mismatch: {path.name}")
    prov = ck.get("provenance") or {}
    assert prov.get("prepared_summary_sha256") == summary_sha, (
        f"checkpoint was not trained on this prepared dataset: {path.name}")
    return ck


@torch.no_grad()
def _bias_by_turn(nets: dict, prepared: Path, split: str, device, outcome_floor) -> dict:
    """Value-head calibration by (turn, decision kind): n, mean predicted win
    prob, observed win rate. Re-derives turns by iterating outcome examples
    (the cache keeps no per-unit turn) and encoding with trunk_inputs, in
    batches -- one pass feeds every checkpoint."""
    groups = {name: collections.defaultdict(lambda: [0, 0.0, 0.0]) for name in nets}
    idx_l, sc_l, meta_l = [], [], []

    def flush():
        if not idx_l:
            return
        idx = torch.tensor(np.stack(idx_l), device=device)
        sc = torch.tensor(np.stack(sc_l), device=device)
        for name, net in nets.items():
            probs = torch.sigmoid(net.value_logit(net.hidden(idx, sc))).squeeze(-1)
            for (kind, turn, won), prob in zip(meta_l, probs.cpu().numpy()):
                bucket = str(turn) if turn <= 11 else "12+"
                g = groups[name][(kind, bucket)]
                g[0] += 1
                g[1] += float(prob)
                g[2] += won
        idx_l.clear()
        sc_l.clear()
        meta_l.clear()

    for ex in iter_task_examples(prepared, "outcome", split=split,
                                 final_evaluation=(split == "test")):
        meta = ex["meta"]
        if meta.get("decision_kind") == "teampreview":
            continue
        if outcome_floor is not None and (meta.get("pre_rating") or 0) < outcome_floor:
            continue
        i_arr, s_arr = trunk_inputs(ex["input"])
        idx_l.append(i_arr)
        sc_l.append(s_arr)
        meta_l.append((meta.get("decision_kind"), int(meta.get("turn") or 0),
                       1.0 if ex["target"] == "win" else 0.0))
        if len(idx_l) >= _BIAS_BATCH:
            flush()
    flush()
    order = {str(t): t for t in range(1, 12)} | {"12+": 12}
    out = {}
    for name, gmap in groups.items():
        table = {}
        for (kind, bucket), (n, pred_sum, win_sum) in sorted(
                gmap.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 99))):
            table.setdefault(kind, {})[bucket] = {
                "n": n, "pred": pred_sum / n, "observed": win_sum / n}
        out[name] = table
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True,
                    help="trainer output dir (turn_net.pt, turn_net_broad.pt, preview_net.pt)")
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final-test", action="store_true",
                    help="required acknowledgement to open the sealed test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final_test:
        ap.error("--split test requires --final-test")
    if args.final_test and args.split != "test":
        ap.error("--final-test is only for --split test")
    if args.out.exists():
        ap.error(f"output file already exists: {args.out}")

    device = torch.device("cpu")
    t0 = time.time()
    train_cache = build_cache(args.prepared, "train")
    eval_cache = build_cache(args.prepared, args.split, None,
                             final_evaluation=(args.split == "test"))
    eval_cache["_train_lineups"] = {tuple(x) for x in train_cache["lineup"]}
    baselines = build_baselines(train_cache)
    del train_cache
    print(f"caches ready in {time.time() - t0:.1f}s", flush=True)

    sampling = json.loads((args.prepared / "summary.json").read_text())["sampling"]
    policy_floor = sampling["broad"].get("broad_min_pre_rating")
    outcome_floor = sampling["broad"].get("broad_outcome_min_pre_rating")
    summary_sha = hashlib.sha256(
        (args.prepared / "summary.json").read_bytes()).hexdigest()

    def load_turn(name):
        net = TurnPretrainNet().to(device)
        ck = _load_checkpoint(args.run / f"{name}.pt", device, summary_sha)
        net.state_encoder.load_state_dict(ck["model_state_dict"])
        net.load_state_dict(ck["heads_state_dict"], strict=False)
        net.eval()
        return net

    def turn_metrics(net):
        return {
            "full": evaluate_turn(net, eval_cache, device, baselines,
                                  only_1600=False, policy_floor=policy_floor,
                                  outcome_floor=outcome_floor),
            "only_1600": evaluate_turn(net, eval_cache, device, baselines,
                                       only_1600=True),
        }

    nets = {name: load_turn(name) for name in ("turn_net", "turn_net_broad")}
    pnet = PreviewPretrainNet().to(device)
    ck = _load_checkpoint(args.run / "preview_net.pt", device, summary_sha)
    pnet.load_state_dict(ck["model_state_dict"])
    pnet.eval()

    result = {
        "split": args.split,
        "run": str(args.run),
        "checkpoint_sha256": {
            name: file_hash(args.run / f"{name}.pt")
            for name in ("turn_net", "turn_net_broad", "preview_net")},
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=REPO_ROOT).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                     capture_output=True, text=True,
                                     cwd=REPO_ROOT).stdout.strip()),
        "turn_net": turn_metrics(nets["turn_net"]),
        "turn_net_broad": turn_metrics(nets["turn_net_broad"]),
        "preview": evaluate_preview(pnet, eval_cache, device, baselines,
                                    policy_floor=policy_floor),
        "bias_by_turn": _bias_by_turn(nets, args.prepared, args.split, device,
                                      outcome_floor),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("wrote", args.out)
    if args.split == "test":
        log_path = REPO_ROOT / "runs" / "test_access_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(json.dumps({
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "prepared": str(args.prepared.resolve()),
                "run": str(args.run.resolve()),
                "out": str(args.out.resolve()),
                "git": result["git"]}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
