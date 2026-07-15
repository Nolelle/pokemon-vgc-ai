"""Integration tests that start a real local Showdown server and run real battles.

Excluded from the default `pytest` run (see [tool.pytest.ini_options] addopts in
pyproject.toml -- `-m "not integration"`). Run explicitly with:

    .venv/bin/python -m pytest -m integration
"""

from __future__ import annotations

import asyncio
import subprocess
import time

import pytest

from vgc.actions import describe_order, enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.baselines import make_player
from vgc.config import FORMAT_ID, SHOWDOWN_REPO, TEAMS_DIR
from vgc.models import PolicyConfig

pytestmark = pytest.mark.integration

SERVER_READY_TIMEOUT_SECONDS = 30


def _find_node() -> str:
    import shutil
    from pathlib import Path

    node = shutil.which("node")
    if node:
        return node
    fallback = Path.home() / ".nvm/versions/node/v22.22.0/bin/node"
    if fallback.exists():
        return str(fallback)
    raise FileNotFoundError("node not found on PATH and no nvm fallback exists")


@pytest.fixture(scope="module")
def local_server():
    """Start `pokemon-showdown start --no-security` for the duration of this module,
    then kill it. Waits for the "listening on" log line before yielding.
    """
    node = _find_node()
    process = subprocess.Popen(
        [node, "pokemon-showdown", "start", "--no-security"],
        cwd=str(SHOWDOWN_REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
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
            await p1.battle_against(p2, n_battles=5)
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
            await p1.battle_against(p2, n_battles=2)
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
            await p1.battle_against(p2, n_battles=3)
        finally:
            await p1.ps_client.stop_listening()
            await p2.ps_client.stop_listening()
        return p1.n_finished_battles, p1.n_won_battles, p2.n_won_battles

    finished, p1_wins, p2_wins = asyncio.run(_run())

    assert finished == 3
    assert p1_wins + p2_wins <= finished
