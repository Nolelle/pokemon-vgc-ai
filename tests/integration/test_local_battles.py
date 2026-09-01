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

import pytest

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.battle_state_replay import (
    DECISION_INPUT_FIELDS,
    decision_records_contain_outcome_labels,
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
        accepting = make_player(
            "vgc", dev_team, FORMAT_ID, accept_open_team_sheet=True
        )
        rejecting = make_player(
            "random", dev_team, FORMAT_ID, accept_open_team_sheet=False
        )
        assert accepting.accept_open_team_sheet is True
        assert rejecting.accept_open_team_sheet is False
        try:
            await asyncio.wait_for(
                accepting.battle_against(rejecting, n_battles=1), timeout=20
            )
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
        verification = asyncio.run(
            verify_decision_replay_bundle(json.loads(path.read_text()))
        )
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
