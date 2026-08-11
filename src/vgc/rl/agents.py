"""Drive existing poke-env players from `vgc.rl.env`, without their transport.

Step 3b of `docs/rl_roadmap.md`. `vgc.rl.env` produces `DoubleBattle`s; every opponent
we care about (`vgc.baselines`, `vgc.rl.player.PpoVgcPlayer`) is a `poke_env` `Player`
whose decision methods take a battle and return an order. This module is the thin seam
between them.

## Why this reuses `Player` instead of reimplementing the policies

`Player.__init__(start_listening=False)` builds a fully working player object and never
opens a websocket, so `player.choose_move(battle)` / `player.teampreview(battle)` can be
called directly on a battle the direct env parsed. That means the direct evaluation path
runs the SAME decision code as the poke-env path -- byte for byte, including
`VgcPlayer`'s exception-safe wrappers and its decision-trace recording -- rather than a
reimplementation that could quietly disagree. A migration whose whole purpose is to stop
maintaining two execution paths should not begin by creating a third.

## BattleMemory

`VgcPlayer` normally accumulates `BattleMemory` inside `_handle_battle_message`, which is
exactly the transport hook the direct env replaces. `DirectAgent.observe` feeds it from
the protocol lines `vgc.rl.env.StepResult` already carries, in the same
observe-then-decide order as the live path, so `vgc.rl.encoding.encode_battle_history`
and the evaluator's habit-tracking see what they would normally see.
"""

from __future__ import annotations

from typing import Sequence

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.player import Player

from vgc.baselines import make_player
from vgc.config import FORMAT_ID
from vgc.rl.env import choice_string


class DirectAgent:
    """One side's decision-maker, wrapping a non-listening `poke_env` `Player`."""

    def __init__(self, player: Player, *, name: str | None = None) -> None:
        self.player = player
        self.name = name or type(player).__name__

    def observe(self, battle_tag: str, lines: Sequence[str]) -> None:
        """Feed this step's protocol lines to the player's `BattleMemory`, if it has one.

        A no-op for players without one (poke-env's own baselines), so callers can treat
        every agent the same way.
        """

        memory_for_tag = getattr(self.player, "_memory_for_tag", None)
        if memory_for_tag is None or not lines:
            return
        memory_for_tag(battle_tag).observe_protocol([line.split("|") for line in lines])

    def choose(self, battle: AbstractBattle) -> str:
        """The choice string for `battle`'s current request, ready for `>p1 ...`.

        Routes team preview to `teampreview` and everything else to `choose_move`, the
        same split `Player._handle_battle_request` makes.
        """

        if battle.teampreview:
            return choice_string(self.player.teampreview(battle))
        return choice_string(self.player.choose_move(battle))

    def finish(self, battle: AbstractBattle) -> None:
        """Signal battle end, so trajectory-collecting players close their episode.

        `PpoVgcPlayer._battle_finished_callback` is what turns recorded steps into a
        finished episode with its terminal reward; without this call an RL rollout would
        silently collect steps that never get returns.
        """

        callback = getattr(self.player, "_battle_finished_callback", None)
        if callback is not None:
            callback(battle)


def make_direct_agent(
    name: str,
    team: str,
    *,
    battle_format: str = FORMAT_ID,
    **kwargs,
) -> DirectAgent:
    """Build a `DirectAgent` for a registered `vgc.baselines` name.

    Same names and semantics as `vgc.baselines.make_player` -- `start_listening=False`
    is forced, since the whole point is a player that never touches the network.
    """

    kwargs["start_listening"] = False
    return DirectAgent(make_player(name, team, battle_format, **kwargs), name=name)
