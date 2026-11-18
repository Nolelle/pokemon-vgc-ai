"""Play a Reg M-C match with a saved replication model on local Showdown."""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
from pathlib import Path

import numpy as np
import torch
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from offline.replica_search_probe import search_public_decision
from vgc.actions import enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.baselines import BASELINES, make_player
from vgc.config import FORMAT_ID
from vgc.rl.env import choice_string
from vgc.rl.replication_policy import load_replication_policy


class LocalReplicaPlayer(VgcPlayer):
    # Never ask the server for a default order on a retry in this check.
    DEFAULT_CHOICE_CHANCE = 0

    def __init__(self, policy, team, name, *, search, seed):
        self.policy = policy
        self.packed_team = team
        self.search = search
        self.rng = np.random.default_rng(seed)
        self.decisions = []
        super().__init__(
            team=team,
            battle_format=FORMAT_ID,
            accept_open_team_sheet=False,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(name, None),
            record_decision_replays=True,
        )

    def decide_teampreview(self, battle):
        order = self.policy.choose_preview(battle)
        prefix, slots = order.split()
        request = battle.last_request
        team_size = len(request["side"]["pokemon"])
        if (
            prefix != "/team"
            or len(slots) != request["maxChosenTeamSize"]
            or len(set(slots)) != len(slots)
            or any(not 1 <= int(slot) <= team_size for slot in slots)
        ):
            raise ValueError(f"model selected an illegal preview: {order}")
        self.decisions.append({"turn": battle.turn, "preview": order})
        return order

    def decide(self, battle):
        legal = enumerate_joint_orders(battle)
        if not legal:
            raise ValueError("Showdown request has no enumerated legal orders")
        worlds = 0
        if self.search:
            result, audit = search_public_decision(
                battle,
                self.packed_team,
                self.policy,
                worlds=2,
                actions=2,
                seeds=2,
                iterations=200,
                seed=int(self.rng.integers(0, 2**31)),
            )
            chosen = result.actions[self.rng.choice(len(result.actions), p=result.probabilities)]
            worlds = audit["searched"]
        else:
            logits, _ = self.policy.score(battle, legal)
            if len(logits) != len(legal) or not np.isfinite(logits).all():
                raise ValueError("model returned invalid order scores")
            chosen = legal[int(np.argmax(logits))]
        wire = choice_string(chosen)
        if wire not in {choice_string(order) for order in legal}:
            raise ValueError(f"model selected an order absent from the current request: {wire}")
        self.decisions.append(
            {
                "turn": battle.turn,
                "forced_switch": any(battle.force_switch),
                "choice": wire,
                "legal_count": len(legal),
                "worlds": worlds,
            }
        )
        return chosen


async def play(args):
    team = args.team.read_text().strip()
    opponent_team = args.opponent_team.read_text().strip() if args.opponent_team else team
    policy = load_replication_policy(args.checkpoint)
    token = secrets.token_hex(3)
    player = LocalReplicaPlayer(policy, team, f"rep-{token}", search=args.search, seed=args.seed)
    opponent = make_player(
        args.opponent,
        opponent_team,
        FORMAT_ID,
        server_configuration=LocalhostServerConfiguration,
        accept_open_team_sheet=False,
        account_configuration=AccountConfiguration(f"opp-{token}", None),
    )
    try:
        await asyncio.wait_for(player.battle_against(opponent, n_battles=1), timeout=args.timeout)
    finally:
        await asyncio.wait_for(
            asyncio.gather(player.ps_client.stop_listening(), opponent.ps_client.stop_listening()),
            timeout=10,
        )
    battles = [
        {"tag": tag, "won": battle.won, "turns": battle.turn}
        for tag, battle in player.battles.items()
        if battle.finished
    ]
    server_errors = []
    for battle in player.battles.values():
        replay = player.decision_replay_bundle(battle)
        if replay is None:
            raise RuntimeError("Showdown battle observation was not recorded")
        server_errors.extend(
            "|".join(line[2:])
            for line in replay["messages"]
            if len(line) > 2 and line[1] in ("error", "bigerror")
        )
    result = {
        "format": FORMAT_ID,
        "server": LocalhostServerConfiguration.websocket_url,
        "checkpoint": str(args.checkpoint),
        "team": str(args.team),
        "opponent": args.opponent,
        "opponent_team": str(args.opponent_team or args.team),
        "search": args.search,
        "battles": battles,
        "fallbacks": player.fallback_count,
        "server_errors": server_errors,
        "decisions": player.decisions,
    }
    passed = (
        len(battles) == 1
        and player.fallback_count == 0
        and not server_errors
        and sum("preview" in row for row in player.decisions) == 1
        and any("choice" in row for row in player.decisions)
    )
    return result, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--team", type=Path, required=True, help="packed Showdown team file")
    parser.add_argument("--opponent", choices=sorted(BASELINES), default="maxpower")
    parser.add_argument("--opponent-team", type=Path, help="packed team (default: mirror match)")
    parser.add_argument("--search", action="store_true", help="use reduced sampled-world search")
    parser.add_argument("--seed", type=int, default=1, help="search sampling seed")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result, passed = asyncio.run(play(args))
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
