"""Train a fresh Champions policy and value network by playing its own league."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vgc.actions import enumerate_joint_orders
from vgc.action_gate import enforce_action_gate_for_cli
from vgc.battle_state_gate import enforce_battle_state_gate_for_cli
from vgc.config import FORMAT_ID
from vgc.mechanics_gate import enforce_mechanics_gate_for_cli
from vgc.rl.env import SimWorker, choice_string
from vgc.rl.match import play_battle
from vgc.rl.replication_policy import (
    EncodedDecision,
    ReplicationPolicy,
    encode_decision,
    preview_order,
    stack_decisions,
)

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT / "data/selfplay/mc_sheet_pool_v2/train_manifest.json"


@dataclass(frozen=True)
class Decision:
    encoded: EncodedDecision
    index: int
    log_prob: float
    value: float


class LeagueAgent:
    def __init__(self, name, model, seed, *, record):
        self.name = name
        self.model = model
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.steps: list[Decision] = []
        self.record = record

    @torch.no_grad()
    def choose(self, battle):
        preview = battle.teampreview
        orders = None if preview else enumerate_joint_orders(battle)
        if orders == []:
            return "default"
        encoded = encode_decision(battle, orders, preview=preview)
        logits, value = self.model(*stack_decisions([encoded], self.model.device), preview=preview)
        log_probs = logits[0].log_softmax(dim=0)
        index = int(torch.multinomial(log_probs.exp().cpu(), 1, generator=self.rng).item())
        if self.record:
            self.steps.append(
                Decision(encoded, index, float(log_probs[index]), float(value.item()))
            )
        return choice_string(preview_order(encoded.candidates[index]) if preview else orders[index])

    def observe(self, battle_tag, lines):
        pass

    def finish(self, battle):
        pass


@dataclass
class Snapshot:
    game_count: int
    model: ReplicationPolicy
    games: int = 0
    learner_points: float = 0.0

    @property
    def priority(self):
        # A Beta(1, 1) starting point avoids discarding new opponents before play.
        win_rate = (1 + self.learner_points) / (2 + self.games)
        return (1 - win_rate) ** 2


def update_policy(model, optimizer, samples, rng, *, epochs, batch_size):
    """PPO with clipped likelihood ratios and undiscounted final battle outcomes."""
    if not samples:
        raise ValueError("cannot train without recorded decisions")
    device = model.device
    advantages = np.asarray([reward - step.value for step, reward in samples], dtype=np.float32)
    advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
    groups = [
        [idx for idx, (step, _) in enumerate(samples) if step.encoded.preview == preview]
        for preview in (False, True)
    ]
    loss_sum = 0.0
    updates = 0
    model.train()
    for _ in range(epochs):
        batches = []
        for group in groups:
            rng.shuffle(group)
            batches.extend(
                group[start : start + batch_size] for start in range(0, len(group), batch_size)
            )
        rng.shuffle(batches)
        for rows in batches:
            steps, rewards = zip(*(samples[i] for i in rows), strict=True)
            states = [step.encoded for step in steps]
            logits, value = model(*stack_decisions(states, device), preview=states[0].preview)
            distribution = torch.distributions.Categorical(logits=logits)
            indices = torch.tensor([step.index for step in steps], dtype=torch.long, device=device)
            old = torch.tensor(
                [step.log_prob for step in steps], dtype=torch.float32, device=device
            )
            returns = torch.tensor(rewards, dtype=torch.float32, device=device)
            advantage = torch.from_numpy(advantages[rows]).to(device)
            ratio = (distribution.log_prob(indices) - old).exp()
            policy = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            loss = policy + 0.5 * F.mse_loss(value, returns) - 0.01 * distribution.entropy().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("training loss was not finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            loss_sum += float(loss.detach().item())
    model.eval()
    return loss_sum / updates


def load_teams(manifest):
    rows = json.loads(manifest.read_text())
    teams = list(dict.fromkeys((manifest.parent / row["file"]).read_text().strip() for row in rows))
    if len(teams) < 2 or any(not team for team in teams):
        raise ValueError(f"need at least two different nonempty packed teams in {manifest}")
    return teams


def save_snapshot(out, model, games):
    path = out / f"policy_{games:07d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(model.checkpoint(), temporary)
    temporary.replace(path)
    return path


def train(args):
    enforce_mechanics_gate_for_cli("model training")
    enforce_battle_state_gate_for_cli("model training")
    enforce_action_gate_for_cli("model training")
    teams = load_teams(args.team_manifest)
    if args.engine == "pikapro":
        from vgc.rl.replica_engine import ReplicaRustWorker

        worker_factory = ReplicaRustWorker
    else:
        worker_factory = SimWorker
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = ReplicationPolicy(args.width, args.layers).to(args.device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    league = [Snapshot(0, copy.deepcopy(model))]
    save_snapshot(args.out, model, 0)
    args.out.joinpath("settings.json").write_text(
        json.dumps(
            {
                **vars(args),
                "format": FORMAT_ID,
                "pikapro_version": version("pikapro") if args.engine == "pikapro" else None,
                "team_count": len(teams),
            },
            default=str,
            indent=2,
        )
        + "\n"
    )
    started = time.monotonic()
    completed = 0
    snapshot_index = 0
    with worker_factory() as worker, args.out.joinpath("metrics.jsonl").open("w") as metrics:
        while completed < args.games:
            samples = []
            opponents_played: dict[int, int] = {}
            current_games = 0
            historical_score = 0.0
            draws = 0
            mean_turns = 0
            games = min(args.games - completed, args.games_per_update)
            for game in range(completed, completed + games):
                snapshot = None
                if game % 2 == 1:
                    snapshot = rng.choices(league, weights=[s.priority for s in league], k=1)[0]
                learner = LeagueAgent("learner", model, rng.getrandbits(32), record=True)
                opponent = LeagueAgent(
                    "opponent",
                    model if snapshot is None else snapshot.model,
                    rng.getrandbits(32),
                    record=snapshot is None,
                )
                if snapshot is None:
                    current_games += 1
                # Flip seats within both opponent modes, independently of team order.
                pair = (learner, opponent) if (game // 2) % 2 == 0 else (opponent, learner)
                team_pair = rng.sample(teams, k=2)
                result = play_battle(
                    worker,
                    f"replication-{game}",
                    dict(zip(("p1", "p2"), pair, strict=True)),
                    dict(zip(("p1", "p2"), team_pair, strict=True)),
                    seed=[rng.randrange(2**16) for _ in range(4)],
                )
                reward = result.result_for("learner")
                draws += reward == 0
                mean_turns += result.turns
                samples.extend((step, reward) for step in learner.steps)
                samples.extend((step, -reward) for step in opponent.steps)
                if snapshot is not None:
                    points = (reward + 1) / 2
                    historical_score += points
                    snapshot.learner_points += points
                    snapshot.games += 1
                    opponents_played[snapshot.game_count] = (
                        opponents_played.get(snapshot.game_count, 0) + 1
                    )
            loss = update_policy(
                model,
                optimizer,
                samples,
                rng,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            completed += games
            if completed >= snapshot_index + args.snapshot_games or completed == args.games:
                save_snapshot(args.out, model, completed)
                league.append(Snapshot(completed, copy.deepcopy(model)))
                league = league[-args.league_size :]
                snapshot_index = completed
            historical_games = games - current_games
            row = {
                "games": completed,
                "batch_games": games,
                "vs_current": current_games,
                "vs_snapshots": opponents_played,
                "vs_snapshots_score": historical_score / historical_games
                if historical_games
                else None,
                "draws": draws,
                "mean_turns": mean_turns / games,
                "decisions": len(samples),
                "preview_decisions": sum(step.encoded.preview for step, _ in samples),
                "training_loss": loss,
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            metrics.write(line + "\n")
            metrics.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--engine", choices=("pikapro", "showdown"), default="pikapro")
    parser.add_argument("--team-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--games-per-update", type=int, default=32)
    parser.add_argument("--snapshot-games", type=int, default=128)
    parser.add_argument("--league-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=37101)
    args = parser.parse_args()
    positive = (
        args.games,
        args.games_per_update,
        args.snapshot_games,
        args.league_size,
        args.batch_size,
        args.epochs,
        args.width,
        args.layers,
        args.threads,
        args.learning_rate,
    )
    if min(positive) <= 0 or args.width % 4:
        parser.error(
            "counts and learning rate must be positive, and width must be divisible by four"
        )
    train(args)


if __name__ == "__main__":
    main()
