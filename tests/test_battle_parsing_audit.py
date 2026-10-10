"""The differential parsing audit, run small, as a regression gate (needs node + built Showdown).

`offline/audit_battle_parsing.py` compares what the bot believes after every step with
Showdown's own state. This plays a few dozen seeded battles in-process and requires that no
mismatch outside the documented limitations/legitimately-hidden lists appears, and that the
same battles WITHOUT the repairs do show defects (so the gate has teeth).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

import audit_battle_parsing as audit  # noqa: E402

from vgc import poke_env_compat  # noqa: E402

pytestmark = pytest.mark.integration

JOB = {
    "indices": list(range(40)),
    "seed": 20261009,
    "agents": ["random", "random", "maxpower", "heuristic"],
    "pool_teams": [],
    "pool_fraction": 0.0,
}


def test_audit_finds_no_unexplained_parsing_mismatch() -> None:
    acc = audit.worker_main({**JOB, "disable": []})
    try:
        assert acc["errors"] == []
        assert acc["compares"] > 500
        assert audit.unexplained_groups(acc) == []
    finally:
        poke_env_compat.set_disabled_fixes([])


def test_audit_has_teeth_without_the_repairs() -> None:
    try:
        acc = audit.worker_main({**JOB, "disable": list(poke_env_compat.ALL_FIXES)})
        groups = audit.unexplained_groups(acc)
    finally:
        poke_env_compat.set_disabled_fixes([])
    assert len(groups) >= 5
    assert any(g.startswith("volatile/stale/") for g in groups)


def test_unbrought_pokemon_are_not_left_active_after_team_preview() -> None:
    from vgc.rl.agents import make_direct_agent
    from vgc.rl.env import DirectBattle, SimWorker

    builder = audit.TeamBuilder()
    import random

    rng = random.Random(3)
    team = builder.pack(rng)
    with SimWorker() as worker:
        battle = DirectBattle.start(worker, "unbrought", team, team, seed=[1, 2, 3, 4])
        agents = {s: make_direct_agent("random", team) for s in ("p1", "p2")}
        try:
            choices = {s: agents[s].choose(battle.battles[s]) for s in battle.sides_to_move()}
            battle.step(choices)
            for side in ("p1", "p2"):
                active = [m for m in battle.battles[side].team.values() if m.active]
                assert len(active) == 2, (side, [m.name for m in active])
        finally:
            battle.close()
