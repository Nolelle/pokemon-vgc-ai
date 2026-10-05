"""Saved decision positions: record them from offline games, rebuild them later.

Purpose: cheaply test whether an advisor's proposed moves beat the engine's own pick on a
fixed set of positions, with the exact Showdown engine as the judge, before spending real
games on the question (see ``offline/record_positions.py`` and ``offline/grade_positions.py``).

Why a new format.  ``DistillationSample`` and ``data/bc/decisions*`` do not store a
rebuildable root.  The existing player-view replay bundle (``vgc.battle_state_replay``,
schema ``vgc-decision-replay-v1``) does: it is the ordered message stream one player
received, and feeding it back through a ``VgcPlayer`` reconstructs the same public
``DoubleBattle`` the player had at the decision.  A position is such a bundle CUT at one
decision (messages up to that decision's observation cutoff, nothing later) plus the
engine's ranking at that decision.

Public-information boundary.  The stored bundle contains only what our side received.
Nothing about the opponent's hidden sets is saved; the grader treats them as beliefs
(``LiveExactMirror``), exactly like live play.  The engine's own answer for the final
decision is blanked inside the stored bundle and lives only in the position row, so a
reader of the bundle cannot see it by accident.

The engine here is the shipped decision path (``vgc.search.search_joint_orders`` through
``VgcPlayer.decide``), not exact search.  Its ranking is recorded at game time so grading
can ask "is the engine's 2nd/3rd/4th choice better than its 1st under exact mechanics?".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from poke_env.battle.double_battle import DoubleBattle
from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.actions import choice_wire_message, describe_order
from vgc.agent import VgcPlayer
from vgc.battle_state_replay import (
    DECISION_REPLAY_SCHEMA,
    DecisionReplayRecorder,
    _feed_messages,
    _replay_player,
    verify_decision_prefix,
)
from vgc.config import FORMAT_ID
from vgc.evaluator import score_joint_orders
from vgc.models import PolicyConfig
from vgc.own_team import apply_own_spreads
from vgc.rl.agents import DirectAgent
from vgc.search import search_joint_orders

POSITION_SCHEMA = "vgc-position-v1"
DEFAULT_TOP_K = 6
MYOPIC_TOP_N = 15  # how deep into the myopic ranking each position records


class RecordingPlayer(VgcPlayer):
    """The shipped engine, plus a record of its ranking at every move decision.

    ``decide`` reproduces ``VgcPlayer.decide``'s default path (two-ply search, falling
    back to the myopic evaluator, argmax) so the chosen order is exactly the shipped one,
    and keeps the top-K of the scored list keyed by decision sequence.
    """

    top_k: int = DEFAULT_TOP_K

    def __init__(self, *args: Any, top_k: int = DEFAULT_TOP_K, **kwargs: Any) -> None:
        self.top_k = top_k
        self.engine_rankings: dict[tuple[str, int], list[dict[str, Any]]] = {}
        # Per decision: our orders the live search actually scored ("searched set") and the
        # myopic ranking's top MYOPIC_TOP_N, for the proposer screen (offline/propose_positions).
        self.engine_candidates: dict[tuple[str, int], dict[str, Any]] = {}
        super().__init__(*args, **kwargs)

    def decide(self, battle):
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        memory = self._memory_for(battle)
        if self.config.use_two_ply_search:
            scored = search_joint_orders(battle, self.config)
        else:
            scored = score_joint_orders(battle, self.config)
        if not scored:
            return self.choose_random_move(battle)
        recorder = self._decision_replay_recorder
        if recorder is not None:
            sequence = len(recorder._stream(battle.battle_tag).decisions) - 1
            self.engine_rankings[(battle.battle_tag, sequence)] = [
                {
                    "order": describe_order(entry.order),
                    "wire": choice_wire_message(entry.order),
                    "score": round(float(entry.score), 4),
                }
                for entry in scored[: self.top_k]
            ]
            self.engine_candidates[(battle.battle_tag, sequence)] = candidate_record(scored)
        memory.record_choice(int(getattr(battle, "turn", 0) or 0), describe_order(scored[0].order))
        return scored[0].order


def candidate_record(scored) -> dict[str, Any]:
    """The searched set and the myopic top-N from a finished ``search_joint_orders`` list.

    Every entry's ``breakdown`` carries ``searched`` and ``myopic_score``; the stable sort by
    ``myopic_score`` recovers the ranking the shortlist was cut from.
    """

    by_myopic = sorted(
        scored, key=lambda e: float(e.breakdown.get("myopic_score", e.score)), reverse=True
    )
    return {
        "searched": [
            describe_order(e.order) for e in scored if e.breakdown.get("searched") is True
        ],
        "myopic_top": [
            {
                "rank": i + 1,
                "order": describe_order(e.order),
                "wire": choice_wire_message(e.order),
                "myopic_score": round(float(e.breakdown.get("myopic_score", e.score)), 4),
            }
            for i, e in enumerate(by_myopic[:MYOPIC_TOP_N])
        ],
    }


class RecordingAgent(DirectAgent):
    """``DirectAgent`` that also feeds the player-view message stream to the recorder.

    The direct env hands each side the protocol lines (including ``|request|``) it would
    receive over the wire; ``line.split("|")`` is exactly the message shape
    ``VgcPlayer._handle_battle_message`` records on the ladder.
    """

    def __init__(self, player: RecordingPlayer, team: str, *, name: str) -> None:
        super().__init__(player, name=name)
        self.team = team
        self._started: set[str] = set()

    def observe(self, battle_tag: str, lines) -> None:
        recorder = self.player._decision_replay_recorder
        if battle_tag not in self._started:
            # The wire stream opens with these two lines, which the direct env never
            # emits; poke-env only creates the battle object on ``|init|battle``.
            self._started.add(battle_tag)
            recorder.observe(battle_tag, ["", "init", "battle"])
            recorder.observe(battle_tag, ["", "title", "ours vs. opponent"])
        for line in lines or []:
            recorder.observe(battle_tag, line.split("|"))
        super().observe(battle_tag, lines)

    def choose(self, battle) -> str:
        # The live path fills our own Stat Points/nature from the supplied team; give the
        # direct battle the same teambuilder team so both agree (and the replay, which
        # does the same, reproduces the recorded state fingerprint).
        if not getattr(battle, "teambuilder_team", None):
            battle._teambuilder_team = Teambuilder.parse_packed_team(self.team)
            apply_own_spreads(battle)
        return super().choose(battle)


def make_recording_agent(
    team: str, *, config: PolicyConfig | None = None, top_k: int = DEFAULT_TOP_K, name: str = "ours"
) -> RecordingAgent:
    player = RecordingPlayer(
        config=config or PolicyConfig(format_id=FORMAT_ID),
        team=team,
        battle_format=FORMAT_ID,
        start_listening=False,
        record_decision_replays=True,
        top_k=top_k,
    )
    return RecordingAgent(player, team, name=name)


@dataclass
class RecordedGame:
    game_id: str
    seat: str
    winner: str | None
    bundle: dict[str, Any]
    rankings: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    candidates: dict[int, dict[str, Any]] = field(default_factory=dict)


def play_recorded_game(
    worker,
    game_id: str,
    *,
    our_team: str,
    opp_team: str,
    opp_agent,
    seed: list[int],
    seat: str = "p1",
    config: PolicyConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> RecordedGame:
    """Play one offline game and return our side's decision-replay bundle + rankings.

    ``game_id`` must look like a Showdown room tag (``battle-<format>-<n>``) because the
    replay parser derives the battle from it.
    """

    from vgc.rl.match import play_battle

    ours = make_recording_agent(our_team, config=config, top_k=top_k, name="ours")
    opp_agent.name = "opp"
    other = "p2" if seat == "p1" else "p1"
    outcome = play_battle(
        worker,
        game_id,
        {seat: ours, other: opp_agent},
        {seat: our_team, other: opp_team},
        seed=seed,
    )
    recorder = ours.player._decision_replay_recorder
    bundle = recorder.bundle(game_id, player_side=seat, player_username=seat)
    winner = outcome.winner
    rankings = {
        seq: rows for (tag, seq), rows in ours.player.engine_rankings.items() if tag == game_id
    }
    candidates = {
        seq: rec for (tag, seq), rec in ours.player.engine_candidates.items() if tag == game_id
    }
    return RecordedGame(game_id, seat, winner, bundle, rankings, candidates)


def sampleable_decisions(game: RecordedGame) -> list[int]:
    """Decision indices worth grading: real move turns with a choice and a ranking.

    Skips team preview, forced switches, and single-option turns.
    """

    out = []
    for index, decision in enumerate(game.bundle["decisions"]):
        if decision.get("phase") != "move":
            continue
        if len(decision.get("legal_actions") or []) <= 1:
            continue
        if not decision.get("chosen_order") or index not in game.rankings:
            continue
        out.append(index)
    return out


def trimmed_bundle(bundle: dict[str, Any], decision_index: int) -> dict[str, Any]:
    """The bundle cut at one decision: no later messages, no later decisions, and the
    engine's own answer for that decision blanked (it lives in the position row)."""

    decisions = [dict(d) for d in bundle["decisions"][: decision_index + 1]]
    cutoff = int(decisions[decision_index]["observation_cutoff"])
    decisions[decision_index]["chosen_order"] = None
    decisions[decision_index]["chosen_order_wire"] = None
    cut = dict(bundle)
    cut["messages"] = [list(m) for m in bundle["messages"][:cutoff]]
    cut["decisions"] = decisions
    return cut


async def _rebuild(bundle: dict[str, Any], decision_index: int):
    """Replay the saved stream through the live parser; capture the decision's state.

    Same technique as ``offline/review_lost_decisions.py``: override ``decide`` on the
    replay player so it fires at each decision with the live battle.  Earlier decisions
    feed their recorded choice into ``BattleMemory`` (the live ``decide`` does that),
    so habit tracking matches what the engine had; the target decision's own choice is
    never fed.
    """

    player, tag = _replay_player(bundle)
    decisions = bundle["decisions"]
    captured: dict[str, Any] = {}

    def decide(battle):
        recorder = player._decision_replay_recorder
        index = len(recorder._stream(battle.battle_tag).decisions) - 1
        if index == decision_index:
            captured["battle"] = battle
            captured["memory"] = getattr(battle, "_vgc_battle_memory", None)
        else:
            chosen = decisions[index].get("chosen_order")
            memory = getattr(battle, "_vgc_battle_memory", None)
            if chosen and memory is not None:
                memory.record_choice(int(getattr(battle, "turn", 0) or 0), str(chosen))
        return player.choose_random_move(battle)

    player.decide = decide
    await _feed_messages(player, tag, [[str(p) for p in m] for m in bundle["messages"]])
    if "battle" not in captured:
        raise ValueError(f"decision {decision_index} was not reached in the replay")
    return captured["battle"], captured["memory"]


def rebuild_position(bundle: dict[str, Any], decision_index: int):
    """Return ``(public DoubleBattle, BattleMemory)`` at the saved decision."""

    return asyncio.run(_rebuild(bundle, decision_index))


def verify_position(bundle: dict[str, Any], decision_index: int):
    """Compare the rebuilt decision fingerprint (state, legal actions) to the saved one."""

    return asyncio.run(verify_decision_prefix(bundle, decision_index))


def load_bundle(position: dict[str, Any], root: Path) -> dict[str, Any]:
    if position.get("bundle"):
        return position["bundle"]
    return json.loads((root / position["bundle_path"]).read_text())


def team_species_key(packed_team: str) -> tuple[str, ...]:
    """Sorted six-species set of a packed team (normalised ids).

    The same grouping ``tools/split_team_pool.py`` uses, so near-copies (same six
    species, different moves/items/file names) share one key.
    """

    species = []
    for member in packed_team.strip().split("]"):
        if not member:
            continue
        fields = member.split("|")
        name = fields[1] if len(fields) > 1 and fields[1] else fields[0]
        species.append(re.sub(r"[^a-z0-9]", "", name.lower()))
    return tuple(sorted(species))


def split_membership(file_name: str, split_file: Path | None) -> str | None:
    """``"tune"``/``"test"`` from ``split.json``, or ``None`` if it has no verdict."""

    if split_file is None or not split_file.exists():
        return None
    payload = json.loads(split_file.read_text())
    if file_name in set(payload.get("holdout_files") or []):
        return "test"
    if file_name in set(payload.get("train_files") or []):
        return "tune"
    return None


def team_split(
    file_name: str,
    split_file: Path | None,
    holdout_fraction: float = 0.3,
    *,
    packed_team: str | None = None,
) -> str:
    """``"tune"`` or ``"test"`` for an opponent team; stable across runs.

    ``split.json`` (``train_files`` / ``holdout_files`` as written by
    ``tools/split_team_pool.py``) wins when it names the file.  Otherwise the fallback
    hashes the team's CONTENT identity -- its sorted six-species set -- not its file
    name, so identical or near-copy teams saved under different names always land on
    the same side and tuning never sees a test team.  The unit is the OPPONENT TEAM.
    Without ``packed_team`` the fallback can only use the file name (not leak-safe;
    callers that care should pass the team or require a split file).
    """

    known = split_membership(file_name, split_file)
    if known is not None:
        return known
    if packed_team is not None:
        identity = ",".join(team_species_key(packed_team))
    else:
        identity = file_name
    digest = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if digest < holdout_fraction else "tune"


__all__ = [
    "DECISION_REPLAY_SCHEMA",
    "DecisionReplayRecorder",
    "POSITION_SCHEMA",
    "RecordedGame",
    "RecordingAgent",
    "RecordingPlayer",
    "candidate_record",
    "load_bundle",
    "make_recording_agent",
    "play_recorded_game",
    "rebuild_position",
    "sampleable_decisions",
    "team_species_key",
    "team_split",
    "split_membership",
    "trimmed_bundle",
    "verify_position",
]
