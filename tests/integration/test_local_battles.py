"""Integration tests that start a real local Showdown server and run real battles.

Excluded from the default `pytest` run (see [tool.pytest.ini_options] addopts in
pyproject.toml -- `-m "not integration"`). Run explicitly with:

    .venv/bin/python -m pytest -m integration
"""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import ForfeitBattleOrder, SingleBattleOrder

from vgc.actions import (
    choice_wire_message,
    describe_order,
    enumerate_joint_orders,
)
from vgc.damage import to_id
from vgc.agent import VgcPlayer
from vgc.battle_state_replay import (
    DECISION_INPUT_FIELDS,
    _legal_actions,
    decision_records_contain_outcome_labels,
    legal_wire_messages,
    messages_through_decision_cutoff,
    replay_battle_at_cutoff,
    verify_decision_prefix,
    verify_decision_replay_bundle,
)
from vgc.baselines import make_player
from vgc.config import FORMAT_ID, SHOWDOWN_REPO, TEAMS_DIR
from vgc.models import PolicyConfig
from vgc.node import find_node, node_environment
from ladder.run_ladder import run_local_smoke

pytestmark = pytest.mark.integration

SERVER_READY_TIMEOUT_SECONDS = 30

REPLAY_RECORD_CONFIG = PolicyConfig(
    format_id=FORMAT_ID,
    accept_open_team_sheet=False,
    use_heuristic_evaluator=False,
    use_two_ply_search=False,
)

OUTCOME_LABEL_KEYS = frozenset({"won", "lost", "winner"})
CANONICAL_DECISION_INPUT_KEYS = frozenset({"state", "belief", "legal_actions"})
CORPUS_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "replay_corpus"
WEATHER_KINDS = ("SunnyDay", "RainDance", "Sandstorm", "Snow")
TERRAIN_KINDS = ("Electric Terrain", "Grassy Terrain", "Psychic Terrain", "Misty Terrain")
WEATHER_STATE_IDS = frozenset({"sunnyday", "raindance", "sandstorm", "snow"})
TERRAIN_STATE_IDS = frozenset(
    {"electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"}
)
IN_BATTLE_REVEAL_TAGS = frozenset({"move", "-ability", "-item", "-enditem"})


class ScriptedPlayer(VgcPlayer):
    """Play a per-slot move script, then the first legal action, optionally forfeit."""

    def __init__(self, *args, **kwargs) -> None:
        self._slot_scripts = tuple(kwargs.pop("slot_scripts", ()))
        self._team_order = str(kwargs.pop("team_order", "/team 1234"))
        self._forfeit_after_moves = kwargs.pop("forfeit_after_moves", None)
        self._move_counts: dict[str, int] = {}
        super().__init__(*args, **kwargs)

    def decide_teampreview(self, battle) -> str:
        members = list(battle.team.values())
        digits = [int(char) for char in self._team_order if char.isdigit()]
        for member in members:
            member._selected_in_teampreview = False
        for selected in digits:
            if 1 <= selected <= len(members):
                members[selected - 1]._selected_in_teampreview = True
        return self._team_order

    def decide(self, battle):
        tag = battle.battle_tag
        force_switch = getattr(battle, "force_switch", False)
        forced_flags = (
            list(force_switch)
            if isinstance(force_switch, (list, tuple)) and force_switch
            else [bool(force_switch)]
        )
        all_forced = bool(forced_flags) and all(forced_flags)
        any_forced = any(forced_flags)
        # A partial replacement request is not a scripted turn; counting it ate a
        # later Protect and let the other side forfeit before terrain persisted.
        if not any_forced:
            count = self._move_counts.get(tag, 0)
            if self._forfeit_after_moves is not None and count >= self._forfeit_after_moves:
                return ForfeitBattleOrder()
            self._move_counts[tag] = count + 1
            script_index = count
        else:
            script_index = None

        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)
        orders = enumerate_joint_orders(battle)
        if not orders:
            return self.choose_random_move(battle)
        if all_forced or script_index is None:
            return orders[0]
        wanted = [
            script[script_index] if script_index < len(script) else None
            for script in self._slot_scripts
        ]
        while len(wanted) < 2:
            wanted.append(None)
        for order in orders:
            if _slot_matches(order.first_order, wanted[0]) and _slot_matches(
                order.second_order, wanted[1]
            ):
                return order
        stay = [
            order
            for order in orders
            if not isinstance(order.first_order.order, Pokemon)
            and not isinstance(order.second_order.order, Pokemon)
        ]
        return (stay or orders)[0]


def _slot_matches(order: SingleBattleOrder, wanted: str | None) -> bool:
    if wanted is None:
        return True
    want_mega = wanted.endswith("-mega")
    move_id = wanted[:-5] if want_mega else wanted
    target = order.order
    if not isinstance(target, Move) or to_id(target.id) != to_id(move_id):
        return False
    return bool(order.mega) if want_mega else True


def _message_lines(bundle: dict[str, object]) -> list[str]:
    return ["|".join(str(part) for part in message) for message in bundle["messages"]]


def _in_battle_decisions(bundle: dict[str, object]) -> list[dict[str, object]]:
    return [
        decision
        for decision in bundle["decisions"]
        if decision.get("phase") in {"move", "forced_switch"}
    ]


def _decision_states(bundle: dict[str, object]) -> list[dict[str, object]]:
    return [decision["state"] for decision in _in_battle_decisions(bundle)]


def _records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _effect_ids(effects: object) -> set[str]:
    return {str(effect.get("id")) for effect in _records(effects)}


def _side_pokemon(state: dict[str, object], side: str) -> list[dict[str, object]]:
    side_state = state.get(side)
    if not isinstance(side_state, dict):
        return []
    return _records(side_state.get("pokemon"))


def _all_pokemon(state: dict[str, object]) -> list[dict[str, object]]:
    return _side_pokemon(state, "our_side") + _side_pokemon(state, "opponent_side")


def _protocol_digest(bundle: dict[str, object]) -> list[str]:
    interesting = (
        "turn",
        "move",
        "switch",
        "drag",
        "faint",
        "-weather",
        "-fieldstart",
        "-fieldend",
        "-mega",
        "-miss",
        "-immune",
        "-boost",
        "-unboost",
        "-fail",
        "-start",
        "-end",
        "win",
    )
    digest: list[str] = []
    for line in _message_lines(bundle):
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag in interesting:
            digest.append(line)
    return digest


def _decision_field_digest(bundle: dict[str, object]) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for decision in _in_battle_decisions(bundle):
        state = decision["state"]
        rows.append(
            (
                decision.get("turn"),
                decision.get("phase"),
                sorted(_effect_ids(state.get("fields"))),
                sorted(_effect_ids(state.get("weather"))),
            )
        )
    return rows


def _assert_effect_persists_two_turns(
    bundle: dict[str, object],
    predicate: Callable[[dict[str, object]], bool],
) -> None:
    matching = [
        decision for decision in _in_battle_decisions(bundle) if predicate(decision["state"])
    ]
    assert matching, (_protocol_digest(bundle), bundle["decisions"])
    first_turn = int(matching[0].get("turn") or 0)
    assert any(int(decision.get("turn") or 0) >= first_turn + 2 for decision in matching), (
        first_turn,
        [(decision.get("turn"), decision.get("phase")) for decision in matching],
        _decision_field_digest(bundle),
        _protocol_digest(bundle),
    )


def _assert_weather_set(bundle: dict[str, object]) -> None:
    lines = _message_lines(bundle)
    assert any(
        "-weather" in line and any(kind in line for kind in WEATHER_KINDS) for line in lines
    ), lines

    def has_weather(state: dict[str, object]) -> bool:
        return bool(_effect_ids(state.get("weather")) & WEATHER_STATE_IDS)

    assert any(has_weather(state) for state in _decision_states(bundle))
    _assert_effect_persists_two_turns(bundle, has_weather)


def _assert_terrain_set(bundle: dict[str, object]) -> None:
    lines = _message_lines(bundle)
    assert any(
        "-fieldstart" in line and any(kind in line for kind in TERRAIN_KINDS) for line in lines
    ), lines

    def has_terrain(state: dict[str, object]) -> bool:
        return bool(_effect_ids(state.get("fields")) & TERRAIN_STATE_IDS)

    assert any(has_terrain(state) for state in _decision_states(bundle))
    _assert_effect_persists_two_turns(bundle, has_terrain)


def _assert_trick_room_active(bundle: dict[str, object]) -> None:
    lines = _message_lines(bundle)
    assert any("-fieldstart" in line and "Trick Room" in line for line in lines), lines

    def has_trick_room(state: dict[str, object]) -> bool:
        return "trickroom" in _effect_ids(state.get("fields"))

    assert any(has_trick_room(state) for state in _decision_states(bundle))
    _assert_effect_persists_two_turns(bundle, has_trick_room)


def _assert_mega_evolution(bundle: dict[str, object]) -> None:
    assert any(
        (len(message) > 1 and message[1] == "-mega")
        or (
            len(message) > 2
            and message[1] == "detailschange"
            and any("-Mega" in part for part in message)
        )
        for message in bundle["messages"]
    ), _message_lines(bundle)

    def has_mega(state: dict[str, object]) -> bool:
        return any(
            mon.get("mega_evolved") or "mega" in str(mon.get("species_id") or "")
            for mon in _all_pokemon(state)
        )

    assert any(has_mega(state) for state in _decision_states(bundle))
    _assert_effect_persists_two_turns(bundle, has_mega)


def _assert_double_faint(bundle: dict[str, object]) -> None:
    faint_run = 0
    found = False
    for line in _message_lines(bundle):
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if tag == "turn":
            faint_run = 0
        elif tag == "faint":
            faint_run += 1
            if faint_run >= 2:
                found = True
                break
    assert found, _message_lines(bundle)

    def has_two_fainted(state: dict[str, object]) -> bool:
        return any(
            sum(1 for mon in _side_pokemon(state, side) if mon.get("fainted")) >= 2
            for side in ("our_side", "opponent_side")
        )

    assert any(has_two_fainted(state) for state in _decision_states(bundle))
    _assert_effect_persists_two_turns(bundle, has_two_fainted)


def _assert_forced_switch_decision(*bundles: dict[str, object]) -> None:
    decisions = [decision for bundle in bundles for decision in bundle["decisions"]]
    assert any(decision.get("phase") == "forced_switch" for decision in decisions)


def _appeared_opponent_species(bundle: dict[str, object]) -> set[str]:
    role = str(bundle.get("player_side") or "")
    opponent = "p2" if role == "p1" else "p1"
    seen: set[str] = set()
    for message in bundle["messages"]:
        if len(message) < 3 or message[1] not in {"switch", "drag"}:
            continue
        ident = message[2]
        if not ident.startswith(opponent):
            continue
        details = message[3] if len(message) > 3 else ident.split(":", 1)[-1]
        seen.add(str(details).split(",")[0].strip())
    return seen


def _assert_fewer_than_four_revealed(bundle: dict[str, object]) -> None:
    appeared = _appeared_opponent_species(bundle)
    assert len(appeared) <= 3, appeared
    in_battle = _in_battle_decisions(bundle)
    assert in_battle
    final_state = in_battle[-1]["state"]
    revealed = [mon for mon in _side_pokemon(final_state, "opponent_side") if mon.get("revealed")]
    assert len(revealed) <= 3, revealed


def _ability_revealed_in_prefix(bundle: dict[str, object], cutoff: int) -> bool:
    for message in bundle["messages"][:cutoff]:
        joined = "|".join(str(part) for part in message)
        if len(message) > 1 and message[1] == "-ability":
            return True
        if "-weather" in joined and "[from] ability:" in joined:
            return True
    return False


def _cutoff_is_before_in_battle_reveals(bundle: dict[str, object], cutoff: int) -> bool:
    return not any(
        len(message) > 1 and message[1] in IN_BATTLE_REVEAL_TAGS
        for message in bundle["messages"][:cutoff]
    )


def _opponent_set_is_known(mon: dict[str, object]) -> bool:
    return bool(mon.get("item_known") and mon.get("ability_known") and mon.get("moves"))


def _assert_ots_accepted(bundle: dict[str, object]) -> None:
    assert bundle["open_team_sheets"] == "accept"
    assert any(message[1:2] == ["showteam"] for message in bundle["messages"])
    for decision in bundle["decisions"]:
        if decision.get("phase") not in {"team_preview", "move"}:
            continue
        cutoff = int(decision.get("observation_cutoff") or 0)
        if not _cutoff_is_before_in_battle_reveals(bundle, cutoff):
            continue
        known = [
            mon
            for mon in _side_pokemon(decision["state"], "opponent_side")
            if _opponent_set_is_known(mon)
        ]
        if known:
            return
    raise AssertionError("no pre-reveal decision has opponent moves/item/ability known")


def _assert_ots_rejected(bundle: dict[str, object]) -> None:
    assert bundle["open_team_sheets"] == "reject"
    assert not any(message[1:2] == ["showteam"] for message in bundle["messages"])
    first_move = next(
        decision for decision in bundle["decisions"] if decision.get("phase") == "move"
    )
    cutoff = int(first_move.get("observation_cutoff") or 0)
    ability_revealed = _ability_revealed_in_prefix(bundle, cutoff)
    opponent = _side_pokemon(first_move["state"], "opponent_side")
    assert opponent
    for mon in opponent:
        assert not mon.get("item_known"), mon
        assert not mon.get("moves"), mon
        if not ability_revealed:
            assert not mon.get("ability_known"), mon


PROTOCOL_CHECKS: dict[str, Callable[..., None]] = {
    "weather": _assert_weather_set,
    "terrain": _assert_terrain_set,
    "trick_room": _assert_trick_room_active,
    "mega": _assert_mega_evolution,
    "double_faint": _assert_double_faint,
    "forced_switch": _assert_forced_switch_decision,
    "few_revealed": _assert_fewer_than_four_revealed,
    "ots_accept": _assert_ots_accepted,
    "ots_reject": _assert_ots_rejected,
}


def _packed_team(name: str) -> str:
    if name in {"dev", "meta1"}:
        return (TEAMS_DIR / f"{name}.packed.txt").read_text().strip()
    return (CORPUS_DIR / f"{name}.packed.txt").read_text().strip()


def _assert_bundle_rebuilds(bundle: dict[str, object]) -> None:
    verification = asyncio.run(verify_decision_replay_bundle(bundle))
    assert verification.ready, verification.mismatches
    assert verification.mismatches == ()
    decisions = bundle["decisions"]
    assert isinstance(decisions, list) and decisions
    for index in range(len(decisions)):
        prefix = asyncio.run(verify_decision_prefix(bundle, index))
        assert prefix.ready, (index, prefix.mismatches)


async def record_scripted_bundles(
    *,
    our_team: str,
    their_team: str,
    our_scripts: tuple[tuple[str, ...], ...],
    their_scripts: tuple[tuple[str, ...], ...],
    our_team_order: str,
    their_team_order: str,
    accept_ots: bool,
    our_forfeit_after_moves: int | None = None,
    their_forfeit_after_moves: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    config = PolicyConfig(
        format_id=FORMAT_ID,
        accept_open_team_sheet=accept_ots,
        use_heuristic_evaluator=False,
        use_two_ply_search=False,
    )
    ours = ScriptedPlayer(
        config=config,
        team=our_team,
        record_decision_replays=True,
        slot_scripts=our_scripts,
        team_order=our_team_order,
        forfeit_after_moves=our_forfeit_after_moves,
    )
    theirs = ScriptedPlayer(
        config=config,
        team=their_team,
        record_decision_replays=True,
        slot_scripts=their_scripts,
        team_order=their_team_order,
        forfeit_after_moves=their_forfeit_after_moves,
    )
    try:
        await asyncio.wait_for(ours.battle_against(theirs, n_battles=1), timeout=45)
    finally:
        await ours.ps_client.stop_listening()
        await theirs.ps_client.stop_listening()
    battle1 = next(iter(ours.battles.values()))
    battle2 = next(iter(theirs.battles.values()))
    bundle1 = ours.decision_replay_bundle(battle1)
    bundle2 = theirs.decision_replay_bundle(battle2)
    assert bundle1 is not None and bundle2 is not None
    return bundle1, bundle2


async def record_replay_bundle(
    dev_team: str,
    *,
    opponent: str = "random",
) -> dict[str, object]:
    ours = VgcPlayer(
        config=REPLAY_RECORD_CONFIG,
        team=dev_team,
        record_decision_replays=True,
    )
    theirs = make_player(opponent, dev_team, FORMAT_ID)
    try:
        await asyncio.wait_for(ours.battle_against(theirs, n_battles=1), timeout=45)
    finally:
        await ours.ps_client.stop_listening()
        await theirs.ps_client.stop_listening()
    battle = next(iter(ours.battles.values()))
    bundle = ours.decision_replay_bundle(battle)
    assert bundle is not None
    return bundle


def _saved_digests(decision: dict[str, object]) -> tuple[object, ...]:
    return tuple(decision.get(key) for key in DECISION_INPUT_FIELDS if key.endswith("_sha256"))


def _assert_no_outcome_labels_under_decisions(decisions: object) -> None:
    assert decision_records_contain_outcome_labels(decisions) == []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key not in CANONICAL_DECISION_INPUT_KEYS:
                    assert key not in OUTCOME_LABEL_KEYS
                if key not in CANONICAL_DECISION_INPUT_KEYS:
                    walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(decisions)


@pytest.fixture(scope="module")
def local_server():
    """Start `pokemon-showdown start --no-security` for the duration of this module,
    then kill it. Waits for the "listening on" log line before yielding.
    """
    node = find_node()
    process = subprocess.Popen(
        [node, "pokemon-showdown", "start", "--no-security"],
        cwd=str(SHOWDOWN_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=node_environment(node),
    )
    try:
        deadline = time.monotonic() + SERVER_READY_TIMEOUT_SECONDS
        ready = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(f"showdown server exited early:\n{output}")
            line = process.stdout.readline() if process.stdout else ""
            if "listening on" in line.lower():
                ready = True
                break
        if not ready:
            process.kill()
            raise TimeoutError("showdown server did not report ready in time")
        # Give the websocket endpoint a brief moment past the log line before connecting.
        time.sleep(1.0)
        yield process
    finally:
        process.kill()
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def dev_team() -> str:
    return (TEAMS_DIR / "dev.packed.txt").read_text().strip()


def test_random_vs_random_battles_complete(local_server, dev_team) -> None:
    async def _run() -> tuple[int, int, int]:
        p1 = make_player("random", dev_team, FORMAT_ID)
        p2 = make_player("random", dev_team, FORMAT_ID)
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=5), timeout=30)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        return p1.n_finished_battles, p1.n_won_battles, p2.n_won_battles

    finished, p1_wins, p2_wins = asyncio.run(_run())

    assert finished == 5
    assert p1_wins + p2_wins <= finished  # draws allowed, but no double-counting
    assert p1_wins + p2_wins >= 0


def test_joint_order_enumeration_during_real_battle(local_server, dev_team) -> None:
    """Drives vgc.actions.enumerate_joint_orders/describe_order against a live battle
    (rather than a hand-built fixture) so any mismatch with poke-env's actual
    DoubleBattle.valid_orders shape is caught end-to-end.
    """
    observed_orders: list[str] = []

    class RecordingPlayer(VgcPlayer):
        def decide(self, battle):
            joint_orders = enumerate_joint_orders(battle)
            assert isinstance(joint_orders, list)
            if joint_orders:
                for order in joint_orders[:5]:
                    description = describe_order(order)
                    assert isinstance(description, str) and description
                    observed_orders.append(description)
                return joint_orders[0]
            return self.choose_random_move(battle)

    async def _run() -> int:
        p1 = RecordingPlayer(
            config=PolicyConfig(format_id=FORMAT_ID),
            team=dev_team,
            battle_format=FORMAT_ID,
        )
        p2 = make_player("random", dev_team, FORMAT_ID)
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=2), timeout=30)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        return p1.n_finished_battles

    finished = asyncio.run(_run())

    assert finished == 2
    assert observed_orders, "expected enumerate_joint_orders to yield orders across turns"


def test_chosen_wire_is_in_legal_wire_set_and_saved_faithfully(
    local_server, dev_team
) -> None:
    """Problem C gate 3 (live half): the sent command is always legal, and the saved
    wire matches what was sent.

    Captures the legal wire set beside every live decision, then aligns those
    captures with the saved replay bundle by decision sequence.
    """

    captured: list[dict[str, object]] = []

    class WireAuditPlayer(VgcPlayer):
        def decide_teampreview(self, battle) -> str:
            choice = "/team 1234"
            captured.append({"phase": "team_preview", "wire": choice})
            return choice

        def decide(self, battle):
            orders = enumerate_joint_orders(battle)
            assert isinstance(orders, list)
            wires = sorted(choice_wire_message(order) for order in orders)
            assert len(wires) == len(orders)
            assert len(set(wires)) == len(wires), "wire messages must be distinct"
            if isinstance(battle, DoubleBattle):
                for slot in (0, 1):
                    available = {
                        to_id(move.id) for move in battle.available_moves[slot]
                    }
                    for order in orders:
                        single = (
                            order.first_order if slot == 0 else order.second_order
                        )
                        if isinstance(single.order, Move):
                            assert to_id(single.order.id) in available, (
                                single.order.id,
                                available,
                            )
                    if battle.trapped[slot]:
                        for order in orders:
                            single = (
                                order.first_order if slot == 0 else order.second_order
                            )
                            assert not isinstance(single.order, Pokemon), (
                                "trapped slot offered a switch",
                                describe_order(order),
                            )
            if not orders:
                captured.append({"phase": "empty", "legal": [], "wire": None})
                return self.choose_random_move(battle)
            choice = orders[0]
            wire = choice_wire_message(choice)
            assert wire in wires
            force_switch = getattr(battle, "force_switch", False)
            forced = (
                any(force_switch)
                if isinstance(force_switch, (list, tuple))
                else bool(force_switch)
            )
            captured.append(
                {
                    "phase": "forced_switch" if forced else "move",
                    "legal": wires,
                    "wire": wire,
                }
            )
            return choice

    async def _run():
        p1 = WireAuditPlayer(
            config=PolicyConfig(format_id=FORMAT_ID),
            team=dev_team,
            battle_format=FORMAT_ID,
            record_decision_replays=True,
        )
        p2 = make_player("random", dev_team, FORMAT_ID)
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=45)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        battle = next(iter(p1.battles.values()))
        return p1.decision_replay_bundle(battle), p1.fallback_count

    bundle, fallbacks = asyncio.run(_run())
    assert bundle is not None
    decisions = bundle["decisions"]
    assert len(captured) == len(decisions), (len(captured), len(decisions))
    assert fallbacks == 0
    for entry, decision in zip(captured, decisions):
        assert entry["phase"] == decision.get("phase"), (entry, decision)
        assert decision.get("chosen_order_wire") == entry["wire"]
        if entry["phase"] == "team_preview":
            legal = decision.get("legal_actions")
            assert isinstance(legal, list) and len(legal) == 360, len(legal or [])
            assert entry["wire"] in legal
            assert decision.get("chosen_order") in legal
        elif entry["phase"] == "empty":
            assert decision.get("chosen_order_wire") is None
        else:
            assert entry["wire"] in entry["legal"]


def test_forced_switch_legal_lists_hold_only_switches(local_server) -> None:
    """Problem C gate 4 (live half): forced-switch lists never contain moves."""
    ours, theirs = asyncio.run(
        record_scripted_bundles(
            our_team=_packed_team("dev"),
            their_team=_packed_team("frail_leads"),
            our_scripts=(
                ("eruption", "protect", "protect", "protect"),
                ("protect-mega", "protect", "protect", "protect"),
            ),
            their_scripts=(
                ("electricterrain", "protect", "protect", "protect"),
                ("charge", "protect", "protect", "protect"),
            ),
            our_team_order="/team 3124",
            their_team_order="/team 1234",
            accept_ots=False,
            our_forfeit_after_moves=4,
            their_forfeit_after_moves=4,
        )
    )
    forced = [
        decision
        for bundle in (ours, theirs)
        for decision in bundle["decisions"]
        if decision.get("phase") == "forced_switch"
    ]
    assert forced, "expected the double-faint script to force a replacement"
    for decision in forced:
        for action in decision["legal_actions"]:
            lowered = str(action).lower()
            assert "switch" in lowered or "pass" in lowered, action
            assert "move " not in lowered
            assert "@" not in lowered
        wire = decision.get("chosen_order_wire")
        assert isinstance(wire, str) and wire.startswith("/choose "), wire


def test_target_variants_cover_offered_targets(local_server, dev_team) -> None:
    """Problem C gate 4 (live half): one move with several targets yields several orders."""
    from poke_env.battle.move import Move

    seen_multi_target: list[tuple[str, set[int | None]]] = []

    class TargetAuditPlayer(VgcPlayer):
        def decide(self, battle):
            orders = enumerate_joint_orders(battle)
            if orders:
                grouped: dict[tuple[str, str], set[int | None]] = {}
                for order in orders:
                    for single in (order.first_order, order.second_order):
                        target = single.order
                        if isinstance(target, Move):
                            key = (
                                "first" if single is order.first_order else "second",
                                str(target.id),
                            )
                            grouped.setdefault(key, set()).add(single.move_target)
                for key, targets in grouped.items():
                    if len(targets) >= 2:
                        seen_multi_target.append((key[1], targets))
                return orders[0]
            return self.choose_random_move(battle)

    async def _run() -> int:
        p1 = TargetAuditPlayer(
            config=PolicyConfig(format_id=FORMAT_ID),
            team=dev_team,
            battle_format=FORMAT_ID,
        )
        p2 = make_player("random", dev_team, FORMAT_ID)
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=2), timeout=45)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        return p1.n_finished_battles

    finished = asyncio.run(_run())
    assert finished == 2
    assert seen_multi_target, "expected one move with 2+ targets across two games"


def test_saved_wire_replays_against_rebuilt_request(local_server, dev_team) -> None:
    """Problem C gate 3: the saved wire is legal at the rebuilt cutoff.

    Replays each decision prefix through the live parser (same path Problem B
    verifies) and requires the saved wire message to be a member of the rebuilt
    request's legal wire set -- the exact set Showdown accepts there -- for all
    three phases.
    """
    bundle = asyncio.run(record_replay_bundle(dev_team))
    decisions = bundle["decisions"]
    assert len(decisions) >= 2
    for index, decision in enumerate(decisions):
        phase = decision.get("phase")
        wire = decision.get("chosen_order_wire")
        assert isinstance(wire, str) and wire, (index, decision)
        battle = asyncio.run(replay_battle_at_cutoff(bundle, index))
        if phase == "team_preview":
            prefix = messages_through_decision_cutoff(bundle, index)
            legal = _legal_actions(
                battle, team_preview=True, request_message=prefix[-1] if prefix else None
            )
        else:
            legal = legal_wire_messages(battle, team_preview=False)
        if legal:
            assert wire in legal, (index, phase, wire)
        else:
            assert "default" in wire, (index, phase, wire)


def test_encored_moves_are_absent_from_live_enumeration(local_server) -> None:
    """Problem C exclusion gate: a disabled move never appears in the enumeration.

    Whimsicott's Encore targets foe slot 1 (Clefable), which opens with Follow
    Me (+3, ahead of Encore's Prankster +1), so on later turns Clefable's request
    marks every other move disabled and poke-env drops them from `valid_orders`.
    The joint list must follow. (The opener must be a +2-or-faster move: Encore
    fails against a target that has not moved yet, and Protect would block the
    Encore itself. The Encore target is pinned explicitly because the default
    order aims at our own ally.)
    """

    class FoeEncorePlayer(ScriptedPlayer):
        def decide(self, battle):
            if int(getattr(battle, "turn", 0) or 0) == 1 and isinstance(
                battle, DoubleBattle
            ):
                encore = None
                guard = None
                for order in enumerate_joint_orders(battle):
                    first, second = order.first_order, order.second_order
                    if (
                        isinstance(first.order, Move)
                        and to_id(first.order.id) == "encore"
                        and first.move_target == 1
                    ):
                        encore = encore or first
                    if (
                        isinstance(second.order, Move)
                        and to_id(second.order.id) == "protect"
                    ):
                        guard = guard or second
                if encore is not None and guard is not None:
                    from poke_env.player.battle_order import DoubleBattleOrder

                    return DoubleBattleOrder(encore, guard)
            return super().decide(battle)

    async def _run():
        config = PolicyConfig(
            format_id=FORMAT_ID,
            accept_open_team_sheet=False,
            use_heuristic_evaluator=False,
            use_two_ply_search=False,
        )
        ours = FoeEncorePlayer(
            config=config,
            team=_packed_team("dev"),
            record_decision_replays=True,
            slot_scripts=(("protect", "protect", "protect"),) * 2,
            team_order="/team 4123",
            forfeit_after_moves=4,
        )
        theirs = ScriptedPlayer(
            config=config,
            team=_packed_team("frail_leads"),
            record_decision_replays=True,
            slot_scripts=(
                ("followme", "protect", "protect", "protect"),
                ("protect", "protect", "protect", "protect"),
            ),
            team_order="/team 3412",
            forfeit_after_moves=4,
        )
        try:
            await asyncio.wait_for(ours.battle_against(theirs, n_battles=1), timeout=45)
        finally:
            await ours.ps_client.stop_listening()
            await theirs.ps_client.stop_listening()
        battle1 = next(iter(ours.battles.values()))
        battle2 = next(iter(theirs.battles.values()))
        return ours.decision_replay_bundle(battle1), theirs.decision_replay_bundle(
            battle2
        )

    ours, theirs = asyncio.run(_run())
    assert ours is not None and theirs is not None
    used = [
        line
        for line in _message_lines(theirs)
        if line.split("|")[1:2] == ["move"] and "Encore" in line
    ]
    assert used, "expected Whimsicott to actually use Encore on turn 1"
    locked = [
        decision
        for decision in theirs["decisions"]
        if decision.get("phase") == "move" and int(decision.get("turn") or 0) >= 2
    ]
    assert locked, "expected post-Encore move decisions"
    found_locked_slot = False
    for decision in locked:
        halves = [str(action).split(" / ") for action in decision["legal_actions"]]
        for slot in (0, 1):
            slot_moves = {
                half[slot].split("@")[0] for half in halves if len(half) == 2
            }
            non_switch = {move for move in slot_moves if not move.startswith("switch")}
            if non_switch and non_switch <= {"followme", "pass"}:
                found_locked_slot = True
    assert found_locked_slot, [
        decision["legal_actions"] for decision in locked
    ]


def test_trapped_slot_offers_no_switch_live(local_server) -> None:
    """Problem C exclusion gate: a trapped slot cannot switch.

    Toxapex's Infestation traps foe slot 1 (Jolteon), which must attack on turn
    1 (Protect would block the trap). On later turns Jolteon's request marks it
    trapped and poke-env drops that slot's switches from `valid_orders`, while
    the untrapped slot keeps its switches -- the control that proves the
    absence is trapping, not an empty bench.
    """

    class FoeInfestationPlayer(ScriptedPlayer):
        def decide(self, battle):
            if int(getattr(battle, "turn", 0) or 0) == 1 and isinstance(
                battle, DoubleBattle
            ):
                trap = None
                guard = None
                for order in enumerate_joint_orders(battle):
                    first, second = order.first_order, order.second_order
                    if (
                        isinstance(first.order, Move)
                        and to_id(first.order.id) == "infestation"
                        and first.move_target == 1
                    ):
                        trap = first
                    if (
                        isinstance(second.order, Move)
                        and to_id(second.order.id) == "protect"
                    ):
                        guard = guard or second
                if trap is not None and guard is not None:
                    from poke_env.player.battle_order import DoubleBattleOrder

                    return DoubleBattleOrder(trap, guard)
            return super().decide(battle)

    async def _run():
        config = PolicyConfig(
            format_id=FORMAT_ID,
            accept_open_team_sheet=False,
            use_heuristic_evaluator=False,
            use_two_ply_search=False,
        )
        ours = FoeInfestationPlayer(
            config=config,
            team=_packed_team("trapper"),
            record_decision_replays=True,
            slot_scripts=(("protect", "protect", "protect"),) * 2,
            team_order="/team 6123",
            forfeit_after_moves=4,
        )
        theirs = ScriptedPlayer(
            config=config,
            team=_packed_team("frail_leads"),
            record_decision_replays=True,
            slot_scripts=(
                ("thunderbolt", "protect", "protect", "protect"),
                ("protect", "protect", "protect", "protect"),
            ),
            team_order="/team 1234",
            forfeit_after_moves=4,
        )
        try:
            await asyncio.wait_for(ours.battle_against(theirs, n_battles=1), timeout=45)
        finally:
            await ours.ps_client.stop_listening()
            await theirs.ps_client.stop_listening()
        battle1 = next(iter(ours.battles.values()))
        battle2 = next(iter(theirs.battles.values()))
        return ours.decision_replay_bundle(battle1), theirs.decision_replay_bundle(
            battle2
        )

    ours, theirs = asyncio.run(_run())
    assert ours is not None and theirs is not None
    trap_lines = [
        line for line in _message_lines(theirs) if "partiallytrapped" in line
    ]
    assert trap_lines, "expected Infestation to trap foe slot 1 on turn 1"
    trapped = [
        decision
        for decision in theirs["decisions"]
        if decision.get("phase") == "move" and int(decision.get("turn") or 0) >= 2
    ]
    assert trapped, "expected post-trap move decisions"
    for decision in trapped:
        halves = [str(action).split(" / ") for action in decision["legal_actions"]]
        assert all(len(half) == 2 for half in halves)
        assert not any(
            half[0].startswith("switch") for half in halves
        ), decision["legal_actions"]
        assert any(
            half[1].startswith("switch") for half in halves
        ), decision["legal_actions"]


def test_mega_and_switch_pair_properties_hold_live(local_server) -> None:
    """Problem C pair gates: no double-Mega ever; a double switch is offered;
    after evolving, no Mega variant is offered again."""
    ours, _theirs = asyncio.run(
        record_scripted_bundles(
            our_team=_packed_team("dev"),
            their_team=_packed_team("frail_leads"),
            our_scripts=(
                ("eruption", "protect", "protect", "protect"),
                ("protect-mega", "protect", "protect", "protect"),
            ),
            their_scripts=(
                ("electricterrain", "protect", "protect", "protect"),
                ("charge", "protect", "protect", "protect"),
            ),
            our_team_order="/team 3124",
            their_team_order="/team 1234",
            accept_ots=False,
            our_forfeit_after_moves=4,
            their_forfeit_after_moves=4,
        )
    )
    move_decisions = [
        decision for decision in ours["decisions"] if decision.get("phase") == "move"
    ]
    assert move_decisions
    assert any(
        "-mega" in str(action)
        for decision in move_decisions
        for action in decision["legal_actions"]
    ), "expected the Mega variant to be offered"
    assert not any(
        sum(1 for half in str(action).split(" / ") if half.endswith("-mega")) >= 2
        for decision in move_decisions
        for action in decision["legal_actions"]
    ), "a double-Mega pair must never be enumerated"
    assert any(
        all(half.startswith("switch") for half in str(action).split(" / "))
        for decision in move_decisions
        for action in decision["legal_actions"]
    ), "expected a double-switch joint order while the bench is live"
    mega_turns = [
        int(decision.get("turn") or 0)
        for decision in move_decisions
        if any(str(action).split(" / ")[1].endswith("-mega") for action in decision["legal_actions"])
    ]
    later = [
        decision
        for decision in move_decisions
        if mega_turns and int(decision.get("turn") or 0) > min(mega_turns)
    ]
    if later:
        assert not any(
            "-mega" in str(action)
            for decision in later
            for action in decision["legal_actions"]
        ), "after evolving, no Mega variant may be offered again"


def test_vgc_vs_random_battles_complete(local_server, dev_team) -> None:
    """End-to-end smoke test for the Phase 2b evaluator/team-preview wiring
    (`vgc.evaluator.score_joint_orders` + `vgc.team_preview.build_team_order`, both
    invoked through `VgcPlayer`'s real `decide()`/`decide_teampreview()` -- see
    vgc/agent.py) against a real local server: no exceptions escaping `choose_move`/
    `teampreview` (which would show up as `fallback_used` on the decision trace -- see
    vgc/decision_trace.py) across full games, not just individual decisions in isolation.
    """

    async def _run() -> tuple[int, int, int]:
        p1 = make_player("vgc", dev_team, FORMAT_ID)
        p2 = make_player("random", dev_team, FORMAT_ID)
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=3), timeout=45)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        return p1.n_finished_battles, p1.n_won_battles, p2.n_won_battles

    finished, p1_wins, p2_wins = asyncio.run(_run())

    assert finished == 3
    assert p1_wins + p2_wins <= finished


def test_ots_accept_reject_race_completes(local_server, dev_team) -> None:
    """An accepting VgcPlayer must not hang when a stock opponent rejects OTS first."""

    async def _run() -> int:
        accepting = make_player("vgc", dev_team, FORMAT_ID, accept_open_team_sheet=True)
        rejecting = make_player("random", dev_team, FORMAT_ID, accept_open_team_sheet=False)
        assert accepting.accept_open_team_sheet is True
        assert rejecting.accept_open_team_sheet is False
        try:
            await asyncio.wait_for(accepting.battle_against(rejecting, n_battles=1), timeout=20)
        finally:
            await accepting.ps_client.stop_listening()
            await rejecting.ps_client.stop_listening()
        return accepting.n_finished_battles

    assert asyncio.run(_run()) == 1


def test_decision_replay_rebuilds_with_open_team_sheets(local_server, dev_team) -> None:
    async def _run() -> tuple[dict, dict]:
        config = PolicyConfig(
            format_id=FORMAT_ID,
            accept_open_team_sheet=True,
            use_heuristic_evaluator=False,
            use_two_ply_search=False,
        )
        p1 = VgcPlayer(config=config, team=dev_team, record_decision_replays=True)
        p2 = VgcPlayer(config=config, team=dev_team, record_decision_replays=True)
        try:
            await asyncio.wait_for(p1.battle_against(p2, n_battles=1), timeout=30)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        battle1 = next(iter(p1.battles.values()))
        battle2 = next(iter(p2.battles.values()))
        return p1.decision_replay_bundle(battle1), p2.decision_replay_bundle(battle2)

    bundles = asyncio.run(_run())
    for bundle in bundles:
        assert bundle is not None
        assert bundle["open_team_sheets"] == "accept"
        assert any(message[1:2] == ["showteam"] for message in bundle["messages"])
        verification = asyncio.run(verify_decision_replay_bundle(bundle))
        assert verification.ready, verification.mismatches


def test_ladder_artifact_pipeline_local_smoke(local_server, dev_team, tmp_path) -> None:
    artifacts = tmp_path / "ladder"
    log_path = tmp_path / "ladder.jsonl"

    records = asyncio.run(
        run_local_smoke(
            n_games=2,
            team=dev_team,
            opponent="random",
            artifacts_dir=artifacts,
            log_path=log_path,
            timeout_seconds=30,
        )
    )

    assert len(records) == 2
    assert len(log_path.read_text().splitlines()) == 2
    assert len(list((artifacts / "replays").glob("*.html"))) == 2
    trace_files = list((artifacts / "traces").glob("*.json"))
    assert len(trace_files) == 2
    assert all(json.loads(path.read_text()) for path in trace_files)
    state_replay_files = list((artifacts / "state-replays").glob("*.json"))
    assert len(state_replay_files) == 2
    for path in state_replay_files:
        verification = asyncio.run(verify_decision_replay_bundle(json.loads(path.read_text())))
        assert verification.ready, verification.mismatches


def test_prefix_replay_matches_each_decision_cutoff(local_server, dev_team) -> None:
    bundle = asyncio.run(record_replay_bundle(dev_team))
    assert len(bundle["decisions"]) >= 2
    for index in range(len(bundle["decisions"])):
        verification = asyncio.run(verify_decision_prefix(bundle, index))
        assert verification.ready, verification.mismatches


def test_truncation_independence_survives_suffix_and_detects_prefix_breaks(
    local_server,
    dev_team,
) -> None:
    bundle = asyncio.run(record_replay_bundle(dev_team))
    decisions = bundle["decisions"]
    assert len(decisions) >= 2

    suffix_mutated = copy.deepcopy(bundle)
    suffix_mutated["messages"].append(["", "turn", "999"])
    for index in range(len(decisions)):
        verification = asyncio.run(verify_decision_prefix(suffix_mutated, index))
        assert verification.ready, verification.mismatches

    probe_index = min(2, len(decisions) - 1)
    cutoff = int(decisions[probe_index]["observation_cutoff"])
    if cutoff < 2:
        probe_index = 1
        cutoff = int(decisions[probe_index]["observation_cutoff"])
    prefix_mutated = copy.deepcopy(bundle)
    prefix_mutated["messages"][cutoff - 2] = ["", "turn", "mutated"]
    broken = asyncio.run(verify_decision_prefix(prefix_mutated, probe_index))
    assert not broken.ready


def test_future_label_mutations_leave_decision_inputs_unchanged(
    local_server,
    dev_team,
) -> None:
    bundle = asyncio.run(record_replay_bundle(dev_team))
    decisions = bundle["decisions"]
    assert decisions
    _assert_no_outcome_labels_under_decisions(decisions)
    original_digests = [_saved_digests(decision) for decision in decisions]

    final_cutoff = int(decisions[-1]["observation_cutoff"])
    mutated = copy.deepcopy(bundle)
    mutated["messages"] = list(bundle["messages"])[:final_cutoff]
    mutated["messages"].append(["", "win", "Opponent"])
    for decision in mutated["decisions"]:
        decision["teacher_action"] = "mutated"
        decision["teacher_score"] = 0.0

    for index in range(len(decisions)):
        verification = asyncio.run(verify_decision_prefix(mutated, index))
        assert verification.ready, verification.mismatches
        assert _saved_digests(mutated["decisions"][index]) == original_digests[index]


def test_forced_switch_records_distinct_decision_identity(local_server, dev_team) -> None:
    bundle = None
    for _attempt in range(5):
        candidate = asyncio.run(record_replay_bundle(dev_team, opponent="maxpower"))
        if any(decision.get("phase") == "forced_switch" for decision in candidate["decisions"]):
            bundle = candidate
            break
    assert bundle is not None, "expected a mid-turn forced switch within five games"

    forced = next(
        decision for decision in bundle["decisions"] if decision.get("phase") == "forced_switch"
    )
    prior = bundle["decisions"][int(forced["decision_sequence"]) - 1]
    assert prior["phase"] == "move"
    assert forced["turn"] == prior["turn"]
    assert forced["request_sequence"] != prior["request_sequence"]
    assert forced["decision_sequence"] == prior["decision_sequence"] + 1
    for action in forced["legal_actions"]:
        lowered = str(action).lower()
        assert "@" not in lowered
        assert "move " not in lowered
        assert "switch" in lowered or "pass" in lowered


def test_own_stat_points_are_known_even_when_open_team_sheets_never_fire(
    local_server, dev_team
) -> None:
    """The ladder case: no OTS, so poke-env never sends us our own spread.

    `Player._handle_battle_message`'s `showteam` branch is the ONLY thing that calls
    `apply_teambuilder_team`, and on the public ladder that message arrives in ~0.2% of
    games. Before `vgc.own_team`, that left `Pokemon.evs is None` for our OWN team and
    `vgc.evaluator._our_pokemon_state` fell back to `default_opponent_spread` -- the
    guess intended for unknown opponents, off by up to 35.6% on this team and
    underestimating Speed on every Pokemon.
    """

    async def _run() -> list[dict]:
        ours = make_player("vgc", dev_team, FORMAT_ID, accept_open_team_sheet=False)
        opponent = make_player("random", dev_team, FORMAT_ID)
        assert ours.accept_open_team_sheet is False
        try:
            await asyncio.wait_for(ours.battle_against(opponent, n_battles=1), timeout=30)
        finally:
            await ours.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()
        return [
            {
                "species": pokemon.species,
                "evs": pokemon.evs,
                "nature": pokemon.nature,
                "opponent_evs": [p.evs for p in battle.opponent_team.values()],
            }
            for battle in ours.battles.values()
            for pokemon in battle.team.values()
        ]

    entries = asyncio.run(_run())
    assert entries, "no battle state captured"
    assert all(entry["evs"] is not None for entry in entries), (
        "our own Stat Points are unknown without OTS -- vgc.own_team did not fire"
    )
    assert all(entry["nature"] is not None for entry in entries)
    # Symmetrically: this must NOT leak the opponent's spread, which we genuinely do not
    # know without a showteam.
    assert all(evs is None for entry in entries for evs in entry["opponent_evs"])


def _run_protocol_checks(
    checks: tuple[str, ...],
    ours: dict[str, object],
    theirs: dict[str, object],
) -> None:
    for check in checks:
        assertion = PROTOCOL_CHECKS[check]
        if check == "forced_switch":
            assertion(ours, theirs)
            continue
        assertion(ours)
        assertion(theirs)


@pytest.mark.parametrize(
    (
        "our_team",
        "their_team",
        "our_order",
        "their_order",
        "our_scripts",
        "their_scripts",
        "accept_ots",
        "our_forfeit_after",
        "checks",
    ),
    [
        pytest.param(
            "dev",
            "frail_leads",
            "/team 3124",
            "/team 1234",
            # Drought is a switch-in ability. Jolteon (Spe 200) outspeeds every
            # Pokemon on `dev` (max Whimsicott 184), so Electric Terrain (accuracy
            # `true`) always lands before Torkoal's Eruption (Spe 36, accuracy 100).
            # Eruption OHKOs both leads at the minimum damage roll; Charge is a
            # self-move so it cannot drop Torkoal's SpA or hit Lightning Rod.
            (
                ("eruption", "protect", "protect", "protect"),
                ("protect-mega", "protect", "protect", "protect"),
            ),
            (
                ("electricterrain", "protect", "protect", "protect"),
                ("charge", "protect", "protect", "protect"),
            ),
            False,
            4,
            (
                "weather",
                "terrain",
                "mega",
                "double_faint",
                "forced_switch",
                "ots_reject",
            ),
            id="weather_terrain_mega_double_faint_forced_ots_reject",
        ),
        pytest.param(
            "meta1",
            "dev",
            "/team 2134",
            "/team 1234",
            # Trick Room has accuracy `true` and priority -7; both sides only
            # choose 100% Protect afterward, so nothing can faint Farigiraf first.
            (
                ("trickroom", "protect", "protect", "protect"),
                ("protect", "protect", "protect", "protect"),
            ),
            (
                ("protect", "protect", "protect", "protect"),
                ("protect", "protect", "protect", "protect"),
            ),
            True,
            4,
            ("trick_room", "few_revealed", "ots_accept"),
            id="trick_room_few_revealed_ots_accept",
        ),
    ],
)
def test_varied_corpus_rebuilds_every_decision(
    local_server,
    our_team: str,
    their_team: str,
    our_order: str,
    their_order: str,
    our_scripts: tuple[tuple[str, ...], ...],
    their_scripts: tuple[tuple[str, ...], ...],
    accept_ots: bool,
    our_forfeit_after: int,
    checks: tuple[str, ...],
) -> None:
    ours, theirs = asyncio.run(
        record_scripted_bundles(
            our_team=_packed_team(our_team),
            their_team=_packed_team(their_team),
            our_scripts=our_scripts,
            their_scripts=their_scripts,
            our_team_order=our_order,
            their_team_order=their_order,
            accept_ots=accept_ots,
            our_forfeit_after_moves=our_forfeit_after,
            their_forfeit_after_moves=our_forfeit_after,
        )
    )
    _run_protocol_checks(checks, ours, theirs)
    _assert_bundle_rebuilds(ours)
    _assert_bundle_rebuilds(theirs)
