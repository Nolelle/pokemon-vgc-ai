"""Measure task interference and policy movement on a frozen endgame checkpoint.

Uses fresh, balanced real trajectories and identical data/Adam state for every
update comparison. These are training diagnostics, not a strength certificate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from commission_rl import forward_batch
from research_endgames import CONFIG, episode, root_from_fixture
from vgc.rl.env import SimWorker
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.ppo import RolloutBuffer, _tensor_batch, ppo_update


def predictions(model, steps):
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(steps), 32):
            batch = _tensor_batch(steps[start:start + 32], "cpu")
            logits, values = forward_batch(model, batch)
            for i in range(len(values)):
                count = int(batch["candidate_mask"][i].sum())
                rows.append((logits[i, :count].log_softmax(-1).clone(), float(values[i])))
    return rows


def measure(before, after, steps):
    kls, shifts = [], []
    for (old_log, _), (new_log, _) in zip(before, after, strict=True):
        kls.append(float((old_log.exp() * (old_log - new_log)).sum()))
        shifts.append(float((old_log.exp() - new_log.exp()).abs().sum() / 2))
    return {
        "decisions": len(steps), "mean_exact_kl": float(np.mean(kls)),
        "max_exact_kl": max(kls), "mean_total_variation": float(np.mean(shifts)),
        "value_mse_before": float(np.mean([(v - s.return_value)**2
                                           for (_, v), s in zip(before, steps, strict=True)])),
        "value_mse_after": float(np.mean([(v - s.return_value)**2
                                          for (_, v), s in zip(after, steps, strict=True)])),
    }


def gradient(model, steps, advantages, *, value=False):
    parameters = list(model.parameters())
    result = [torch.zeros_like(p) for p in parameters]
    for start in range(0, len(steps), 16):
        selected = steps[start:start + 16]
        batch = _tensor_batch(selected, "cpu")
        logits, values = forward_batch(model, batch)
        if value:
            loss = 0.5 * ((values - batch["returns"])**2).sum() / len(steps)
        else:
            adv = torch.tensor(advantages[start:start + 16], dtype=torch.float32)
            loss = -(Categorical(logits=logits).log_prob(batch["actions"]) * adv).sum() / len(steps)
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        for dest, grad in zip(result, grads, strict=True):
            if grad is not None:
                dest.add_(grad.detach())
    return torch.cat([g.flatten() for g in result])


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path("data/research/endgames_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=32)
    parser.add_argument("--future-seed", type=int, default=4100000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite diagnostic evidence")
    torch.set_num_threads(1)
    started = time.monotonic()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CandidatePolicyValueNet(use_information_features=True,
                                    use_mechanics_features=True, use_tactical_features=True)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    buffer = RolloutBuffer()
    spans, collected = [], []
    with SimWorker() as worker:
        for i, fixture in enumerate(fixtures):
            root = root_from_fixture(worker, fixture, f"diagnose-{i}")
            start = len(buffer)
            try:
                outcomes = [episode(root, fixture["side"], args.future_seed + i*1000 + j,
                                    model=model, buffer=buffer)[0]
                            for j in range(args.episodes_per_task)]
            finally:
                root.close()
            spans.append((start, len(buffer)))
            collected.append({"task": i, "wins": outcomes.count(1.0), "games": len(outcomes)})
            print("collected", collected[-1], flush=True)
    advantages = np.array([s.advantage for s in buffer.steps])
    normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    before = predictions(model, buffer.steps)
    policy_grads, value_grads = [], []
    for start, end in spans:
        policy_grads.append(gradient(model, buffer.steps[start:end], normalized[start:end]))
        value_grads.append(gradient(model, buffer.steps[start:end], normalized[start:end], value=True))
    report = {
        "status": "diagnostic_unapproved", "checkpoint_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "future_seed": args.future_seed, "collection": collected,
        "policy_gradient_cosines": [[cosine(a, b) for b in policy_grads] for a in policy_grads],
        "policy_value_gradient_cosines": [[cosine(a, b) for b in value_grads] for a in policy_grads],
        "policy_gradient_norms": [float(g.norm()) for g in policy_grads],
        "value_gradient_norms": [float(g.norm()) for g in value_grads], "comparisons": [],
    }
    for epochs in (4, 1):
        candidate = copy.deepcopy(model)
        optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)
        optimizer.load_state_dict(saved["optimizer"])
        config = replace(CONFIG, epochs=epochs)
        trial_buffer = copy.deepcopy(buffer)
        metrics = ppo_update(candidate, optimizer, trial_buffer, config,
                             rng=np.random.default_rng(110064))
        after = predictions(candidate, buffer.steps)
        row = {"config": asdict(config), "optimizer_lr": optimizer.param_groups[0]["lr"],
               "metrics": metrics, "tasks": [measure(before[s:e], after[s:e], buffer.steps[s:e])
                                               for s, e in spans]}
        report["comparisons"].append(row)
        print("comparison", epochs, row["tasks"], flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
