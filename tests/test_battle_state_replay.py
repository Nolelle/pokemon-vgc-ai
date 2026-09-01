from __future__ import annotations

import asyncio
from types import SimpleNamespace

from vgc.agent import VgcPlayer
from vgc.battle_state_replay import _legal_actions


def test_team_preview_records_every_legal_bring_four_order() -> None:
    battle = SimpleNamespace(
        team={f"pokemon-{index}": object() for index in range(6)},
        # poke-env uses this for the displayed roster size, not the bring count.
        max_team_size=6,
    )

    actions = _legal_actions(
        battle,
        team_preview=True,
        request_message=["", "request", '{"teamPreview":true,"maxChosenTeamSize":4}'],
    )

    assert len(actions) == 360
    assert len(set(actions)) == 360
    assert "/team 1234" in actions
    assert "/team 6543" in actions


def test_multiline_burst_cannot_leak_lines_after_request_into_memory() -> None:
    observed_at_request: list[dict] = []

    class PrefixPlayer(VgcPlayer):
        async def _handle_battle_message_line(self, split_messages) -> None:
            message = split_messages[1]
            if len(message) > 1 and message[1] == "request":
                memory = self._memory_for_tag("battle-prefix")
                observed_at_request.append(
                    {
                        species: dict(moves)
                        for species, moves in memory.opponent_moves.items()
                    }
                )

    player = PrefixPlayer.__new__(PrefixPlayer)
    player._battle_memories = {}
    player._decision_replay_recorder = None
    player._pending_ots_rejections = set()
    player._resolved_ots_rejections = set()
    player_role_memory = player._memory_for_tag("battle-prefix")
    player_role_memory.our_role = "p1"

    asyncio.run(
        player._handle_battle_message(
            [
                [">battle-prefix"],
                ["", "move", "p2a: Garchomp", "Earthquake", "p1a: Incineroar"],
                ["", "request", "{}"],
                ["", "move", "p2a: Garchomp", "Protect", "p2a: Garchomp"],
            ]
        )
    )

    assert observed_at_request == [{"garchomp": {"earthquake": 1}}]
    assert player_role_memory.opponent_moves["garchomp"]["protect"] == 1
