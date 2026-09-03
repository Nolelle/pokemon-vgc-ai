from __future__ import annotations

import asyncio
from types import SimpleNamespace

from vgc.actions import choice_wire_message
from vgc.agent import VgcPlayer
from vgc.battle_state_replay import (
    DecisionReplayRecorder,
    _legal_actions,
    legal_wire_messages,
)


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


def test_choice_wire_message_passes_preview_strings_through() -> None:
    assert choice_wire_message("/team 1234") == "/team 1234"


def test_choice_wire_message_returns_sendable_order_message() -> None:
    from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

    order = DoubleBattleOrder(
        SingleBattleOrder("/choose move heatwave 1"),
        SingleBattleOrder("/choose move protect"),
    )
    wire = choice_wire_message(order)
    assert wire.startswith("/choose ")
    assert "heatwave" in wire


def test_preview_wire_list_matches_legal_actions() -> None:
    battle = SimpleNamespace(
        team={f"pokemon-{index}": object() for index in range(6)},
        max_team_size=6,
    )
    request = ["", "request", '{"teamPreview":true,"maxChosenTeamSize":4}']
    assert legal_wire_messages(
        battle, team_preview=True, request_message=request
    ) == _legal_actions(battle, team_preview=True, request_message=request)


def test_record_choice_stores_display_and_wire() -> None:
    from pathlib import Path

    from vgc.models import PolicyConfig

    recorder = DecisionReplayRecorder(
        own_packed_team="team",
        config=PolicyConfig(),
        repo_root=Path("."),
        showdown_repo=Path("."),
    )
    recorder._stream("tag").decisions.append({"chosen_order": None, "chosen_order_wire": None})
    recorder.record_choice("tag", 0, "/team 1234")
    bundle_decision = recorder._stream("tag").decisions[0]
    assert bundle_decision["chosen_order"] == "/team 1234"
    assert bundle_decision["chosen_order_wire"] == "/team 1234"

    recorder.record_choice("tag", 0, "heatwave / protect", wire="/choose move heatwave")
    bundle_decision = recorder._stream("tag").decisions[0]
    assert bundle_decision["chosen_order"] == "heatwave / protect"
    assert bundle_decision["chosen_order_wire"] == "/choose move heatwave"
