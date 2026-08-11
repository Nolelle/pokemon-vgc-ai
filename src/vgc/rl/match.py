"""Play battles and match series on the direct environment.

Step 4 of `docs/rl_roadmap.md`. `offline/run_matches.py`'s existing path drives two
networked players against a real Showdown server at ~3 games/sec, which makes the
mandated n>=500 / n>=1000 evaluation budget impractical and leaves training and
evaluation on two different execution paths. `run_series` is the replacement: same
result shape, same decision code (see `vgc.rl.agents`), no server.

## Sides are swapped, not randomized

`run_series` alternates which agent is p1. Doubles has real turn-order and slot
asymmetries, and a fixed seat assignment would fold any seat advantage into the measured
win rate. Alternating makes the comparison paired at the seat level, which is what
`docs/rl_roadmap.md`'s "swap battle sides" evaluation rule asks for.

## Seeding

`run_series`' `seed` pins the SIMULATOR seeds it hands each battle, so the same series
replays the same games. The policy half of `docs/rl_roadmap.md`'s `(seed_sim,
seed_policy)` pair belongs to the agents, not to the runner: `agent_factories` are
caller-supplied closures, so a caller that wants a reproducible stochastic policy passes
`policy_seed=` when it builds the agent (see `vgc.rl.player.PpoVgcPlayer`). Leaving
`seed` None gives every battle a fresh simulator seed, which is what you want when
measuring a win rate rather than reproducing one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Sequence

from vgc.config import FORMAT_ID
from vgc.evaluation import wilson_interval
from vgc.rl.agents import DirectAgent
from vgc.rl.env import DirectBattle, SimWorker

SIDES: tuple[str, str] = ("p1", "p2")

# A battle that has not ended after this many decision points is treated as a bug in the
# environment or a policy that cannot make progress, not as a long game -- real Reg M-B
# games resolve in well under 100 decisions and the sim enforces its own endless-battle
# clause well before this.
MAX_DECISIONS = 1000


@dataclass
class BattleOutcome:
    """One finished battle, from the perspective of the two agents that played it."""

    battle_id: str
    winner: str | None
    turns: int
    decisions: int
    agent_sides: dict[str, str]

    def result_for(self, agent_name: str) -> float:
        """`+1` win / `-1` loss / `0` draw for `agent_name`."""

        if self.winner is None:
            return 0.0
        return 1.0 if self.agent_sides[agent_name] == self.winner else -1.0


def play_battle(
    worker: SimWorker,
    battle_id: str,
    agents: dict[str, DirectAgent],
    teams: dict[str, str],
    *,
    battle_format: str = FORMAT_ID,
    seed: Sequence[int] | None = None,
) -> BattleOutcome:
    """Play one battle to completion, `agents` keyed by side (`"p1"` / `"p2"`).

    The two agents must have distinct `name`s, since the returned `BattleOutcome` maps
    name -> side. A mirror matchup (`vgc` vs `vgc`) therefore has to disambiguate them
    before getting here -- `run_series` does this from its factory keys.
    """

    if agents["p1"].name == agents["p2"].name:
        raise ValueError(
            f"both agents are named {agents['p1'].name!r}; give a mirror matchup two "
            "distinct names so results can be attributed"
        )
    battle = DirectBattle.start(
        worker,
        battle_id,
        teams["p1"],
        teams["p2"],
        battle_format=battle_format,
        seed=seed,
    )
    decisions = 0
    try:
        # The opening burst (team preview) has to reach BattleMemory too, so observe
        # `start`'s lines before the first decision as well as after every step.
        for side in SIDES:
            agents[side].observe(battle_id, battle.last_lines[side])
        while not battle.ended:
            to_move = battle.sides_to_move()
            choices = {side: agents[side].choose(battle.battles[side]) for side in to_move}
            decisions += len(choices)
            result = battle.step(choices)
            for side in SIDES:
                agents[side].observe(battle_id, result.lines[side])
            if decisions > MAX_DECISIONS:
                raise RuntimeError(f"battle {battle_id} exceeded {MAX_DECISIONS} decisions")
        for side in SIDES:
            agents[side].finish(battle.battles[side])
        return BattleOutcome(
            battle_id=battle_id,
            winner=battle.winner,
            turns=int(getattr(battle.battles["p1"], "turn", 0) or 0),
            decisions=decisions,
            agent_sides={agents[side].name: side for side in SIDES},
        )
    finally:
        battle.close()


def run_series(
    worker: SimWorker,
    agent_factories: dict[str, Callable[[], DirectAgent]],
    teams: dict[str, str],
    games: int,
    *,
    battle_format: str = FORMAT_ID,
    seed: int | None = None,
    swap_sides: bool = True,
) -> list[BattleOutcome]:
    """Play `games` battles between two named agents, alternating seats.

    `agent_factories` maps agent name -> zero-argument factory. A FRESH agent is built
    per battle: `VgcPlayer` accumulates `BattleMemory` and decision traces per battle
    tag, and `PpoVgcPlayer` accumulates rollout steps, so reusing one across a long
    series would grow unboundedly and leak one battle's habit-tracking into the next.
    """

    names = list(agent_factories)
    if len(names) != 2:
        raise ValueError(f"run_series needs exactly 2 agents, got {names}")
    rng = random.Random(seed)
    outcomes: list[BattleOutcome] = []
    for index in range(games):
        first, second = names
        if swap_sides and index % 2 == 1:
            first, second = second, first
        agents = {"p1": agent_factories[first](), "p2": agent_factories[second]()}
        # The factory key is the identity results are attributed to, so it wins over
        # whatever name the factory gave the agent -- otherwise a mirror matchup builds
        # two agents with the same name and BattleOutcome cannot tell them apart.
        agents["p1"].name, agents["p2"].name = first, second
        battle_seed = [rng.randrange(1, 2**31) for _ in range(4)]
        outcomes.append(
            play_battle(
                worker,
                f"b{index}",
                agents,
                {"p1": teams[first], "p2": teams[second]},
                battle_format=battle_format,
                seed=battle_seed,
            )
        )
    return outcomes


def summarize(
    outcomes: Sequence[BattleOutcome],
    p1_name: str,
    p2_name: str,
    *,
    battle_format: str = FORMAT_ID,
) -> dict[str, object]:
    """Aggregate into `offline/run_matches.py`'s result shape.

    Deliberately the same keys as the websocket path's output so gate reports, the
    experiment log, and any existing analysis of `runs/eval/*.json` keep working across
    the migration.
    """

    games = len(outcomes)
    p1_wins = sum(1 for outcome in outcomes if outcome.result_for(p1_name) > 0)
    p2_wins = sum(1 for outcome in outcomes if outcome.result_for(p2_name) > 0)
    p1_low, p1_high = wilson_interval(p1_wins, games)
    p2_low, p2_high = wilson_interval(p2_wins, games)
    return {
        "p1": p1_name,
        "p2": p2_name,
        "format": battle_format,
        "environment": "direct",
        "open_team_sheets": "n/a (direct env reveals nothing beyond the sim's own fog)",
        "games": games,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": games - p1_wins - p2_wins,
        "p1_win_rate": p1_wins / games if games else 0.0,
        "p2_win_rate": p2_wins / games if games else 0.0,
        "p1_wilson": [p1_low, p1_high],
        "p2_wilson": [p2_low, p2_high],
        "mean_turns": sum(o.turns for o in outcomes) / games if games else 0.0,
        "timestamp": datetime.now(UTC).isoformat(),
    }
