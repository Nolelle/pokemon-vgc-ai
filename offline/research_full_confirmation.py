"""Paired full-battle confirmation using the fixed-mirror trainer's actual evaluator.

Use PYTHONPATH to select the immutable, readiness-certified source snapshot.
The only wrapper records per-game results; model decisions and RNG schedules
come directly from the existing evaluator. No model is updated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from vgc.config import REPO_ROOT
from vgc.model_vocabulary import require_current_vocabulary
from vgc.rl.env import SimWorker
from vgc.rl.model import CandidatePolicyValueNet
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION

sys.path.insert(0, str(REPO_ROOT))
from selfplay import train_fixed_mirror as trainer  # noqa: E402


def load_model(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["architecture"] != RL_ARCHITECTURE_VERSION:
        raise ValueError("confirmation checkpoint architecture mismatch")
    require_current_vocabulary(saved, path)
    model = CandidatePolicyValueNet(
        use_meta_features=bool(saved.get("use_meta_features", False)),
        use_information_features=bool(saved["use_information_features"]),
        use_mechanics_features=bool(saved["use_mechanics_features"]),
        use_tactical_features=bool(saved["use_tactical_features"]),
        head_dropout=float(saved.get("head_dropout", 0.0)),
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.games < 2:
        parser.error("use a fresh output path and at least two games")
    from vgc.mechanics_gate import assert_mechanics_ready
    from vgc.battle_state_gate import assert_battle_state_ready
    from vgc.action_gate import assert_action_ready
    assert_mechanics_ready("research confirmation")
    assert_battle_state_ready("research confirmation")
    assert_action_ready("research confirmation")
    torch.set_num_threads(1)
    started = time.monotonic()
    team_path = REPO_ROOT / "teams/phase2_mirror.packed.txt"
    report = {"status": "diagnostic_unapproved", "complete": False,
              "source_snapshot": str(REPO_ROOT),
              "source_commit": subprocess.check_output(
                  ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
              "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "team_sha256": hashlib.sha256(team_path.read_bytes()).hexdigest(),
              "seed": args.seed, "games_per_arm": args.games, "opponent": "maxpower", "arms": {}}
    original = trainer.play_battle
    records = []
    arm = ""

    def record_game(*positional, **kwargs):
        result = original(*positional, **kwargs)
        records.append({"game": len(records), "side": result.agent_sides[trainer.LEARNER_NAME],
                        "outcome": result.result_for(trainer.LEARNER_NAME),
                        "simulator_seed": list(kwargs["seed"]), "turns": result.turns})
        if len(records) % 32 == 0:
            print(arm, len(records), "games", flush=True)
        return result

    trainer.play_battle = record_game
    try:
        with SimWorker() as worker:
            for arm, path in (("initial", args.initial), ("endpoint", args.endpoint)):
                records = []
                summary = trainer._run_eval(
                    worker, model=load_model(path), team=team_path.read_text().strip(),
                    opponent_name="maxpower", games=args.games,
                    ppo_config=trainer.build_ppo_config(), device="cpu", seed=args.seed, iteration=0)
                assert len(records) == args.games
                report["arms"][arm] = {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "summary": summary, "games": records}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        trainer.play_battle = original
    first = report["arms"]["initial"]["games"]
    last = report["arms"]["endpoint"]["games"]
    assert all(a["side"] == b["side"] and a["simulator_seed"] == b["simulator_seed"]
               for a, b in zip(first, last, strict=True))
    differences = np.asarray([(b["outcome"] - a["outcome"]) / 2
                              for a, b in zip(first, last, strict=True)])
    rng = np.random.default_rng(args.seed + 1000000)
    bootstrap = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(1)
    report.update({"complete": True, "paired_match_score_change": float(differences.mean()),
                   "paired_bootstrap_95_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                   "improved_pairs": int((differences > 0).sum()),
                   "worsened_pairs": int((differences < 0).sum()),
                   "elapsed_seconds": time.monotonic() - started})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("complete", report["paired_match_score_change"], report["paired_bootstrap_95_interval"], flush=True)


if __name__ == "__main__":
    main()
