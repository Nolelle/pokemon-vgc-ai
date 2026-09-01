"""Versioned, player-view decision replay bundles for Part B.

An HTML replay is a spectator log.  It does not contain every player-private Showdown
``request`` or the player's registered team, so it cannot prove what the agent knew at a
decision.  This module records the actual ordered message stream received by one player
and fingerprints the canonical state built at every request.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from poke_env.battle.double_battle import DoubleBattle

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.battle_memory import BattleMemory
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.opponent_belief import information_boundary_summary

DECISION_REPLAY_SCHEMA = "vgc-decision-replay-v1"


def canonical_json(value: object) -> str:
    """Stable JSON used for equality checks and SHA-256 fingerprints."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _git_state(path: Path) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=path, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {"commit": commit, "dirty": dirty}


def _phase(battle: Any, *, team_preview: bool) -> str:
    if team_preview:
        return "team_preview"
    force_switch = getattr(battle, "force_switch", False)
    if any(force_switch) if isinstance(force_switch, (list, tuple)) else bool(force_switch):
        return "forced_switch"
    return "move"


def _legal_actions(
    battle: Any,
    *,
    team_preview: bool,
    request_message: list[str] | None = None,
) -> list[str]:
    if team_preview:
        # Showdown accepts an ordered selection of ``maxChosenTeamSize`` distinct
        # preview slots. Keep the complete wire-level action set in the same digest as
        # move and forced-switch requests; 6-pick-4 produces only 360 entries.
        team_size = len(getattr(battle, "team", {}) or {})
        pick_count = 0
        if request_message and len(request_message) > 2 and request_message[1] == "request":
            try:
                request = json.loads(request_message[2])
                pick_count = int(request.get("maxChosenTeamSize", 0) or 0)
            except (json.JSONDecodeError, TypeError, ValueError):
                pick_count = 0
        try:
            fallback_count = int(getattr(battle, "max_team_size", 0) or 0)
        except (AttributeError, KeyError, TypeError, ValueError):
            fallback_count = 0
        pick_count = min(team_size, pick_count or fallback_count or team_size)
        return [
            "/team " + "".join(str(slot) for slot in ordering)
            for ordering in itertools.permutations(range(1, team_size + 1), pick_count)
        ]
    if not isinstance(battle, DoubleBattle):
        return []
    return sorted(describe_order(order) for order in enumerate_joint_orders(battle))


def decision_state_payload(battle: Any) -> dict[str, object]:
    """Canonical public mechanics state at one player decision."""

    return asdict(snapshot_battle(battle))


@dataclass
class BattleReplayStream:
    battle_tag: str
    messages: list[list[str]] = field(default_factory=list)
    decisions: list[dict[str, object]] = field(default_factory=list)
    next_observation_sequence: int = 0


class DecisionReplayRecorder:
    """Accumulate exact player-side observations and decision checkpoints."""

    def __init__(
        self,
        *,
        own_packed_team: str | None,
        config: PolicyConfig,
        repo_root: Path,
        showdown_repo: Path,
    ) -> None:
        self.own_packed_team = own_packed_team
        self.config = config
        self.repo_root = repo_root
        self.showdown_repo = showdown_repo
        self._streams: dict[str, BattleReplayStream] = {}

    def _stream(self, battle_tag: str) -> BattleReplayStream:
        if battle_tag not in self._streams:
            self._streams[battle_tag] = BattleReplayStream(battle_tag)
        return self._streams[battle_tag]

    def observe(self, battle_tag: str, message: list[object]) -> None:
        """Record one message before it is parsed by poke-env."""

        stream = self._stream(battle_tag)
        stream.messages.append([str(part) for part in message])
        stream.next_observation_sequence += 1

    def record_decision(
        self,
        battle: Any,
        memory: BattleMemory,
        *,
        team_preview: bool,
    ) -> int:
        stream = self._stream(battle.battle_tag)
        state = decision_state_payload(battle)
        beliefs = information_boundary_summary(battle, memory, self.config)
        legal_actions = _legal_actions(
            battle,
            team_preview=team_preview,
            request_message=stream.messages[-1] if stream.messages else None,
        )
        sequence = len(stream.decisions)
        stream.decisions.append(
            {
                "decision_sequence": sequence,
                "turn": int(getattr(battle, "turn", 0) or 0),
                "phase": _phase(battle, team_preview=team_preview),
                "request_sequence": max(0, stream.next_observation_sequence - 1),
                "observation_cutoff": stream.next_observation_sequence,
                "state": state,
                "state_sha256": payload_digest(state),
                "belief": beliefs,
                "belief_sha256": payload_digest(beliefs),
                "legal_actions": legal_actions,
                "legal_actions_sha256": payload_digest(legal_actions),
                "chosen_order": None,
            }
        )
        return sequence

    def record_choice(self, battle_tag: str, decision_sequence: int, choice: str) -> None:
        self._stream(battle_tag).decisions[decision_sequence]["chosen_order"] = choice

    def bundle(
        self,
        battle_tag: str,
        *,
        player_side: str | None,
        player_username: str | None,
    ) -> dict[str, object]:
        stream = self._stream(battle_tag)
        team = self.own_packed_team or ""
        return {
            "schema": DECISION_REPLAY_SCHEMA,
            "created_at": datetime.now(UTC).isoformat(),
            "battle_tag": battle_tag,
            "format": self.config.format_id,
            "player_side": player_side,
            "player_username": player_username,
            "open_team_sheets": (
                "accept" if self.config.accept_open_team_sheet else "reject"
            ),
            "policy_config": asdict(self.config),
            "own_packed_team": team,
            "own_team_sha256": hashlib.sha256(team.encode()).hexdigest(),
            "repository": _git_state(self.repo_root),
            "showdown": _git_state(self.showdown_repo),
            "messages": stream.messages,
            "decisions": stream.decisions,
        }

    def has_battle(self, battle_tag: str) -> bool:
        return battle_tag in self._streams


@dataclass(frozen=True)
class ReplayVerification:
    ready: bool
    decisions_expected: int
    decisions_rebuilt: int
    mismatches: tuple[str, ...]


async def verify_decision_replay_bundle(bundle: dict[str, object]) -> ReplayVerification:
    """Rebuild every decision through VgcPlayer and compare its exact inputs.

    The replay player deliberately makes cheap arbitrary choices: saved server messages
    already contain the resulting battle, so the submitted replay choice cannot affect
    parsing.  State capture happens in the final ``choose_move``/``teampreview`` wrappers
    before those choices are produced, exactly as it does for the live player.
    """

    from poke_env.ps_client.account_configuration import AccountConfiguration
    from poke_env.teambuilder.teambuilder import Teambuilder

    from vgc.agent import VgcPlayer
    from vgc.own_team import apply_own_spreads

    if bundle.get("schema") != DECISION_REPLAY_SCHEMA:
        return ReplayVerification(False, 0, 0, ("unsupported replay schema",))
    team = bundle.get("own_packed_team")
    messages = bundle.get("messages")
    expected = bundle.get("decisions")
    if not isinstance(team, str) or not team:
        return ReplayVerification(False, 0, 0, ("missing own packed team",))
    if not isinstance(messages, list) or not isinstance(expected, list):
        return ReplayVerification(False, 0, 0, ("missing messages or decisions",))

    raw_config = bundle.get("policy_config") or {}
    allowed = {entry.name for entry in fields(PolicyConfig)}
    config_values = {
        key: value
        for key, value in raw_config.items()
        if key in allowed
    } if isinstance(raw_config, dict) else {}
    config_values["accept_open_team_sheet"] = bundle.get("open_team_sheets") == "accept"

    class ReplayStatePlayer(VgcPlayer):
        def _record_decision(self, battle, *, team_preview):
            if not getattr(battle, "teambuilder_team", None):
                battle._teambuilder_team = Teambuilder.parse_packed_team(team)
                apply_own_spreads(battle)
            return super()._record_decision(battle, team_preview=team_preview)

        def decide(self, battle):
            return self.choose_random_move(battle)

        def decide_teampreview(self, battle):
            recorder = self._decision_replay_recorder
            index = len(recorder._stream(battle.battle_tag).decisions) - 1
            choice = str(expected[index].get("chosen_order") or "/team 1234")
            digits = [int(char) for char in choice if char.isdigit()]
            members = list(battle.team.values())
            for member in members:
                member._selected_in_teampreview = False
            for selected in digits:
                if 1 <= selected <= len(members):
                    members[selected - 1]._selected_in_teampreview = True
            return choice

    username = str(bundle.get("player_username") or "replaystate")
    player = ReplayStatePlayer(
        config=PolicyConfig(**config_values),
        team=team,
        start_listening=False,
        account_configuration=AccountConfiguration(username, None),
        record_decision_replays=True,
    )

    async def _noop(*_args, **_kwargs):
        return None

    player.ps_client.send_message = _noop
    battle_tag = str(bundle.get("battle_tag") or "")
    room = [f">{battle_tag}"]
    for raw in messages:
        if not isinstance(raw, list):
            return ReplayVerification(False, len(expected), 0, ("malformed message",))
        await player._handle_battle_message([room, [str(part) for part in raw]])

    battle = player._battles.get(battle_tag)
    if battle is None:
        return ReplayVerification(False, len(expected), 0, ("battle was not rebuilt",))
    rebuilt_bundle = player.decision_replay_bundle(battle)
    rebuilt = list((rebuilt_bundle or {}).get("decisions") or [])
    mismatches: list[str] = []
    if len(rebuilt) != len(expected):
        mismatches.append(
            f"decision count expected {len(expected)} rebuilt {len(rebuilt)}"
        )
    checked_fields = (
        "turn",
        "phase",
        "request_sequence",
        "observation_cutoff",
        "state_sha256",
        "belief_sha256",
        "legal_actions_sha256",
    )
    for index, (wanted, actual) in enumerate(zip(expected, rebuilt, strict=False)):
        for key in checked_fields:
            if wanted.get(key) != actual.get(key):
                mismatches.append(
                    f"decision {index} {key}: expected {wanted.get(key)!r}, "
                    f"rebuilt {actual.get(key)!r}"
                )
    return ReplayVerification(
        ready=not mismatches,
        decisions_expected=len(expected),
        decisions_rebuilt=len(rebuilt),
        mismatches=tuple(mismatches),
    )
