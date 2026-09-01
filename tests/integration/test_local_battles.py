"""Integration tests that start a real local Showdown server and run real battles.

Excluded from the default `pytest` run (see [tool.pytest.ini_options] addopts in
pyproject.toml -- `-m "not integration"`). Run explicitly with:

    .venv/bin/python -m pytest -m integration
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time

import pytest

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.baselines import make_player
from vgc.config import FORMAT_ID, SHOWDOWN_REPO, TEAMS_DIR
from vgc.models import PolicyConfig
from vgc.node import find_node, node_environment
from ladder.run_ladder import run_local_smoke

pytestmark = pytest.mark.integration

SERVER_READY_TIMEOUT_SECONDS = 30

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
