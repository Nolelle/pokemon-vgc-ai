"""Bounded PPO commissioning: a synthetic bandit and real fogged battle rollouts.

This is a diagnostic, not a strength experiment or release certificate. The bandit
uses the production candidate model and PPO code; its rewards are NOT Pokemon
labels. Real-game collection tests probability consistency without training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from vgc.bc.encoding import (
    INDEX_DIM, MOVE_TO_IDX, SLOT_FEATURE_DIM, STATE_SCALAR_DIM, TARGET_TO_IDX,
)
from vgc.config import FORMAT_ID, REPO_ROOT
from vgc.rl.agents import DirectAgent, PHASE2_PREVIEW_ORDER, make_direct_agent
from vgc.rl.encoding import CandidateFeatures, HISTORY_SCALAR_DIM
from vgc.rl.env import SimWorker
from vgc.rl.match import play_battle
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.player import PpoVgcPlayer
from vgc.rl.ppo import PpoConfig, RolloutBuffer, RolloutStep, _tensor_batch, ppo_update


def forward_batch(model, batch):
    return model(
        batch["state_indices"], batch["state_scalars"], batch["history_scalars"],
        batch["move_indices"], batch["target_indices"], batch["switch_species_indices"],
        batch["flags"], batch["candidate_mask"],
        **{k: batch[k] for k in (
            "meta_scalars", "information_indices", "information_scalars",
            "mechanics_tokens", "mechanics_mask",
        ) if k in batch},
        tactical_features=batch["tactical_features"] if model.use_tactical_features else None,
    )


def bandit(seed: int, config: PpoConfig) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    candidates = CandidateFeatures(
        move_indices=np.asarray([[MOVE_TO_IDX[m], MOVE_TO_IDX["protect"]]
                                 for m in ("protect", "dragonclaw", "heatwave")]),
        target_indices=np.full((3, 2), TARGET_TO_IDX["self_or_field"], dtype=np.int64),
        switch_species_indices=np.zeros((3, 2), dtype=np.int64),
        flags=np.zeros((3, 2, 4), dtype=np.float32),
    )
    template = RolloutStep(
        state_indices=np.zeros(INDEX_DIM, dtype=np.int64),
        state_scalars=np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
        history_scalars=np.zeros(HISTORY_SCALAR_DIM, dtype=np.float32),
        candidates=candidates, action_index=0, old_log_prob=0.0, old_value=0.0,
    )
    batch = _tensor_batch([template], "cpu")

    def probability():
        model.eval()
        with torch.no_grad():
            return float(forward_batch(model, batch)[0].softmax(-1)[0, 0])

    initial = probability()
    curve = []
    for iteration in range(40):
        model.eval()
        with torch.no_grad():
            logits, values = forward_batch(model, batch)
            dist = Categorical(logits=logits[0])
            actions = dist.sample((64,))
            log_probs = dist.log_prob(actions)
        buffer = RolloutBuffer()
        for action, log_prob in zip(actions.tolist(), log_probs.tolist(), strict=True):
            buffer.add(RolloutStep(
                state_indices=template.state_indices, state_scalars=template.state_scalars,
                history_scalars=template.history_scalars, candidates=candidates,
                action_index=action, old_log_prob=log_prob, old_value=float(values.item()),
            ))
            reward = 1.0 if action == 0 else -1.0
            buffer.finish_episode(reward, config)
            if buffer.steps[-1].return_value != reward:
                raise AssertionError("one-step terminal return is incorrect")
        metrics = ppo_update(model, optimizer, buffer, config, rng=rng)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise AssertionError("nonfinite optimizer metric")
        curve.append({"iteration": iteration + 1, "optimal_action_probability": probability()})
    return {"seed": seed, "episodes": 40 * 64, "initial": initial,
            "final": curve[-1]["optimal_action_probability"], "curve": curve,
            "passed": curve[-1]["optimal_action_probability"] >= 0.95}


def real_rollouts(seed: int, config: PpoConfig, games: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    # Stress the old dropout defect as well as variable-length mechanics inputs.
    model = CandidatePolicyValueNet(
        use_information_features=True, use_mechanics_features=True,
        use_tactical_features=True, head_dropout=0.1,
    )
    buffer = RolloutBuffer()
    team_path = REPO_ROOT / "teams/phase2_mirror.packed.txt"
    team = team_path.read_text().strip()
    outcomes, fallbacks = [], 0
    with SimWorker() as worker:
        for index in range(games):
            side = "p1" if index % 2 == 0 else "p2"
            other = "p2" if side == "p1" else "p1"
            player = PpoVgcPlayer(
                model=model, rollout_buffer=buffer, ppo_config=config,
                team=team, battle_format=FORMAT_ID, start_listening=False,
                policy_seed=seed + index,
            )
            learner = DirectAgent(player, name="learner", preview_order=PHASE2_PREVIEW_ORDER)
            opponent = make_direct_agent("maxpower", team)
            opponent.name = "opponent"
            opponent.preview_order = PHASE2_PREVIEW_ORDER
            outcome = play_battle(worker, f"commission-{index}",
                                  {side: learner, other: opponent}, {side: team, other: team},
                                  seed=[rng.randrange(1, 2**31) for _ in range(4)])
            outcomes.append({"side": side, "result": outcome.result_for("learner"),
                             "turns": outcome.turns})
            fallbacks += player.fallback_count
    max_error = 0.0
    model.eval()
    with torch.no_grad():
        for size in (1, 7, 32):
            for start in range(0, len(buffer.steps), size):
                batch = _tensor_batch(buffer.steps[start:start + size], "cpu")
                logits, _ = forward_batch(model, batch)
                actual = Categorical(logits=logits).log_prob(batch["actions"])
                max_error = max(max_error, float((actual - batch["old_log_probs"]).abs().max()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    metrics = ppo_update(model, optimizer, buffer, config, rng=np.random.default_rng(seed))
    return {"games": outcomes, "decisions": len(buffer.steps), "fallbacks": fallbacks,
            "team_sha256": hashlib.sha256(team.encode()).hexdigest(),
            "stress_head_dropout": 0.1, "max_log_probability_error": max_error,
            "zero_lr_update": metrics,
            "passed": (fallbacks == 0 and max_error <= 1e-5
                       and metrics["clip_fraction"] == 0.0
                       and abs(metrics["approx_kl"]) <= 1e-5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4)
    args = parser.parse_args()
    if args.games < 2:
        parser.error("at least two games are required to exercise both seats")
    torch.set_num_threads(1)
    config = PpoConfig(gamma=1.0, teacher_anchor_weight=0.0, reward_shaping_coef=0.0,
                       epochs=2, minibatch_size=32)
    started = time.monotonic()
    report = {"schema": "rl-commissioning-v1", "format": FORMAT_ID,
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "working_tree_diff_sha256": hashlib.sha256(subprocess.check_output(
                  ["git", "diff", "HEAD"])).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": asdict(config), "real_seed": 20261028,
              "torch_version": torch.__version__, "scope": "diagnostic; no strength claim"}
    identity_paths = (
        "src/vgc/rl/ppo.py", "src/vgc/rl/player.py", "src/vgc/rl/model.py",
        "src/vgc/rl/encoding.py", "src/vgc/rl/env.py", "tools/sim_worker.mjs",
        "data/champions/mechanics_catalog.json", "data/usage/set_priors.json",
        "data/usage/spreads.json",
    )
    report["source_and_data_sha256"] = {
        p: hashlib.sha256((REPO_ROOT / p).read_bytes()).hexdigest() for p in identity_paths
    }
    report["real_rollouts"] = real_rollouts(20261028, config, args.games)
    print("real rollouts:", report["real_rollouts"]["passed"], flush=True)
    report["bandit"] = [bandit(seed, config) for seed in (11, 22, 33)]
    report["elapsed_seconds"] = time.monotonic() - started
    report["passed"] = report["real_rollouts"]["passed"] and all(
        r["passed"] for r in report["bandit"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "seconds": report["elapsed_seconds"],
                      "bandit": [(r["seed"], r["initial"], r["final"])
                                 for r in report["bandit"]]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
