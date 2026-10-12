"""The @pkmn/client shadow reader (offline trial only; the bot never uses it).

One integration test: the Node reader starts, passes its M-C data self-check (the Dex is built
from OUR pinned Showdown's `champions` mod, not pkmn's published M-B data), and grades a short
seeded battle through the same comparator as poke-env. Needs `npm ci --ignore-scripts` in
tools/pkmn_reader and the built local Showdown checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

import audit_battle_parsing as audit  # noqa: E402
import pkmn_shadow  # noqa: E402

from vgc import poke_env_compat  # noqa: E402


@pytest.mark.integration
def test_pkmn_reader_passes_data_selfcheck_and_grades_a_seeded_battle() -> None:
    try:
        with pkmn_shadow.PkmnReader() as reader:
            assert reader.selfcheck["all_ok"], reader.selfcheck
            # (the check itself would have raised on a mismatch; this pins the M-C values)
            assert reader.selfcheck["meteorassault_bp"]["got"] == 170
            job = {
                "indices": list(range(6)),
                "seed": 99,
                "agents": ["random", "heuristic"],
                "pool_teams": [],
                "pool_fraction": 0.0,
                "disable": [],
                "readers": ("pokeenv", "pkmn"),
            }
            # worker_main owns its reader; drive audit_battle directly with the open one
            import random

            from vgc.rl.env import SimWorker

            builder = audit.TeamBuilder()
            acc = audit._new_acc()
            with SimWorker() as worker:
                for index in job["indices"]:
                    rng = random.Random(job["seed"] * 1_000_003 + index)
                    teams = {s: builder.pack(rng) for s in audit.SIDES}
                    names = {s: rng.choice(job["agents"]) for s in audit.SIDES}
                    seed = [rng.randrange(1, 2**31) for _ in range(4)]
                    audit.audit_battle(
                        worker, f"pkmn_t{index}", names, teams, seed, acc,
                        readers=job["readers"], pkmn_reader=reader,
                    )  # fmt: skip
            pk = acc["by_reader"]["pkmn"]
            assert pk["compares"] > 50 and pk["compares"] == acc["compares"]
            assert reader.stats.process_failures == 0
            assert reader.stats.parse_errors == 0
            assert reader.stats.snapshots == pk["compares"]
            # protocol-level facts both readers must get right on every battle
            fields = {audit._field_of(g) for g in pk["episodes"]}
            assert not fields & {"turn", "fainted", "weather", "active_slots"}, fields
    finally:
        poke_env_compat.set_disabled_fixes([])
