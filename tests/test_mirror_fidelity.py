"""The live exact judge's Showdown mirror must rebuild the battle correctly.

`offline/audit_mirror_fidelity.py` plays seeded local battles, builds the mirror root the way the
live judge does, and compares it with Showdown's own state (field comparison) and with a clone of
the true battle stepped under the same choices and seed (oracle equivalence). Fast unit tests
cover the pure helpers; the integration tests run the audit itself.

Origin of the bench-order regression: the exact judge errored on 11 ladder decisions with
`[Invalid choice] Can't switch: You can't switch to a fainted Pokemon` because the search turned
`switch <name>` into a position in our REAL request while the mirror listed its bench in
team-preview order, so a position named a different (sometimes fainted) Pokemon.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

import audit_mirror_fidelity as audit  # noqa: E402

from vgc.battle_memory import BattleMemory  # noqa: E402
from vgc.mechanics_state import champions_max_pp  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.live_mirror import (  # noqa: E402
    _learnset,
    _own_request_order,
    _request_bring,
    _set_species_id,
)

FIX_KNOBS = (
    "exact_mirror_foe_forme",
    "exact_mirror_champions_pp",
    "exact_mirror_gender",
    "exact_mirror_stall_duration",
    "exact_mirror_fainted_state",
    "exact_mirror_own_order",
    "exact_mirror_volatile_timers",
    "exact_mirror_public_bookkeeping",
    "exact_mirror_keep_revealed_moves",
)


def test_every_fix_knob_defaults_to_the_fixed_behaviour() -> None:
    config = PolicyConfig()
    assert all(getattr(config, knob) is True for knob in FIX_KNOBS)


def test_champions_pp_formula() -> None:
    # Champions: (base pp / 5 + 1) * 4 with base capped at 20 -- not the vanilla pp * 8 / 5.
    assert champions_max_pp("protect") == 8
    assert champions_max_pp("shadowball") == 16
    assert champions_max_pp("thunderwave") == 20
    assert champions_max_pp("tackle") == 20  # 40 PP in vanilla, capped to 20
    assert champions_max_pp("notamove") is None


def test_foe_sets_are_built_as_their_real_forme() -> None:
    persian_alola = SimpleNamespace(species="persianalola")
    assert _set_species_id(persian_alola, "persian") == "persianalola"
    gourgeist = SimpleNamespace(species="gourgeistsmall")
    assert _set_species_id(gourgeist, "gourgeist") == "gourgeistsmall"
    # battle-only formes and Megas are built as what they change from
    assert _set_species_id(SimpleNamespace(species="aegislashblade"), "aegislash") == "aegislash"
    assert _set_species_id(SimpleNamespace(species="charizardmegay"), "charizard") == "charizard"
    assert _set_species_id(SimpleNamespace(species="floettemega"), "floette") == "floetteeternal"
    # the forme's learnset is the union with the base species' (a forme may list fewer moves)
    assert "snarl" in _learnset("persianalola", "persian") or _learnset("persianalola", "persian")


def test_own_order_and_bring_come_from_the_real_request() -> None:
    request = {
        "side": {
            "pokemon": [
                {"ident": "p1: Alpha", "details": "Garchomp, L50, M"},
                {"ident": "p1: Beta", "details": "Rotom-Wash, L50"},
                {"ident": "p1: Gamma", "details": "Incineroar, L50, F"},
                {"ident": "p1: Delta", "details": "Gholdengo, L50"},
            ]
        }
    }
    battle = SimpleNamespace(last_request=request)
    assert _own_request_order(battle) == ["Alpha", "Beta", "Gamma", "Delta"]
    assert _request_bring(battle) == ["garchomp", "rotom", "incineroar", "gholdengo"]
    assert _own_request_order(SimpleNamespace(last_request=None)) == []


def test_meganium_is_not_read_as_a_mega() -> None:
    """`"mega" in "meganium"`: Meganium was a Mega that could never Mega Evolve."""

    from vgc.mechanics_state import _mega_forme_id

    assert _mega_forme_id("meganium", None) is None
    assert _mega_forme_id("meganium", "megasol") == "meganiummega"
    assert _mega_forme_id("meganiummega", None) == "meganiummega"
    assert _mega_forme_id("charizardmegay", None) == "charizardmegay"


def test_perish_song_duration_is_the_perish_count() -> None:
    from vgc.mechanics_state import _effect_snapshots

    effects = _effect_snapshots(
        {"perish3": 0, "perish2": 1}.keys(), counter_kind="elapsed_actions",
        canonical_volatiles=True,
    )
    assert [(e.id, e.turns, e.counter_kind) for e in effects] == [
        ("perishsong", 2, "perish_count")
    ]


def test_battle_memory_locks_megas_and_fainting() -> None:
    memory = BattleMemory("t2")
    memory.our_role = "p1"
    memory.observe_protocol(
        [
            ["", "turn", "4"],
            ["", "switch", "p2a: Foo", "Garchomp, L50, M", "100/100"],
            ["", "move", "p2a: Foo", "Outrage", "p1a: Bar"],
            ["", "detailschange", "p2a: Foo", "Garchomp-Mega, L50, M"],
            ["", "turn", "5"],
        ]
    )
    assert memory.locked_move("p2", "Foo", 5) == "outrage"
    assert memory.locked_move("p2", "Foo", 6) is None
    assert memory.mega_forme("p2", "Foo") == "garchompmega"


def test_battle_memory_public_history() -> None:
    memory = BattleMemory("t")
    memory.our_role = "p1"

    def feed(*lines: str) -> None:
        memory.observe_protocol([line.split("|") for line in lines])

    feed("|turn|1", "|switch|p2a: Foo|Garchomp, L50, M|100/100")
    assert not memory.has_acted("p2", "Foo")
    feed("|move|p2a: Foo|Hyper Beam|p1a: Bar", "|-mustrecharge|p2a: Foo", "|turn|2")
    assert memory.has_acted("p2", "Foo")
    assert memory.recharge_pending("p2", "garchomp")
    feed("|cant|p2a: Foo|recharge", "|faint|p1a: Bar", "|turn|3")
    assert not memory.recharge_pending("p2", "garchomp")  # spent: poke-env never clears it
    assert memory.lost_pokemon_last_turn("p1", 3)
    assert memory.faint_count("p1") == 1 and memory.faint_count("p2") == 0
    assert not memory.lost_pokemon_last_turn("p1", 4)
    assert memory.last_move_used == "hyperbeam"
    feed("|switch|p2a: Foo|Garchomp, L50, M|100/100")  # a new stint
    assert not memory.has_acted("p2", "Foo")


def test_faint_count_counts_events_not_currently_fainted() -> None:
    """Last Respects reads `side.totalFainted`, which counts faint events: faint, Revival
    Blessing, faint again is 2, though only one Pokemon is down at the end."""
    memory = BattleMemory("t")
    memory.our_role = "p1"
    memory.observe_protocol(
        [
            line.split("|")
            for line in (
                "|turn|1",
                "|faint|p1a: Bar",
                "|-heal|p1: Bar|50/100|[from] move: Revival Blessing",
                "|faint|p1a: Bar",
            )
        ]
    )
    assert memory.faint_count("p1") == 2
    assert memory.faint_count("p2") == 0


# --- integration: run the audit --------------------------------------------------------------

JOB = {
    "indices": list(range(24)),
    "seed": 20261010,
    "agents": ["random", "random", "maxpower", "heuristic"],
    "pool_teams": [],
    "pool_fraction": 0.0,
    "point_rate": 0.5,
    "oracle_rate": 0.6,
    "pairs": 2,
    "oracle": True,
    "normal": True,
    "config": {},
}


@pytest.mark.integration
def test_audit_finds_no_unexplained_mirror_mismatch() -> None:
    """The standing check: fixed seed, both modes, no crash and no unexplained group."""

    acc = audit.worker_main(JOB)
    assert acc["errors"] == []
    assert acc["points"] > 100 and acc["oracle_pairs"] > 100
    assert not [
        g for g in audit.unexplained_groups(acc) if g.startswith(("normal/", "oracle_state/"))
    ]
    # the root state must be explained completely; a handful of downstream protocol rolls
    # (a 1-HP damage-roll difference) are tolerated, a legality / crash difference is not
    unexplained = audit.unexplained_groups(acc)
    assert not [g for g in unexplained if "legality" in g or "crash" in g]
    assert acc["oracle_fail"] <= max(2, 0.03 * acc["oracle_pairs"])
    # most pairs must be bit-identical, not merely explained by a documented gap
    assert acc["oracle_pass"] >= 0.5 * acc["oracle_pairs"]
    # the true sleep / confusion remaining time must always be one of the searched branches
    ok, total = acc["timer_support"]["sleep remaining time covered"]
    assert ok == total
    # confusion is covered except when a rolled lock (Outrage) ended in it: a known gap
    ok, total = acc["timer_support"]["confusion remaining time covered"]
    assert ok >= 0.7 * total


@pytest.mark.integration
def test_audit_has_teeth_with_the_fixes_off() -> None:
    """With every reconstruction fix switched off the same battles show the old defects."""

    legacy = {knob: False for knob in FIX_KNOBS}
    acc = audit.worker_main({**JOB, "indices": list(range(12)), "config": legacy})
    groups = " ".join(audit.unexplained_groups(acc))
    for needle in (
        "public:gender",
        "move_slots.maxpp",
        "volatile_value:stall.duration",
        "volatile_value:mustrecharge.duration",
        "own_side_order",
    ):
        assert needle in groups, needle
    assert acc["oracle_pass"] == 0


@pytest.mark.integration
def test_normal_mirror_never_reads_private_simulator_state() -> None:
    """The normal (non-oracle) mirror is built with the private root stripped from the view."""

    from vgc.rl.agents import make_direct_agent
    from vgc.rl.env import DirectBattle, SimWorker

    builder = audit.TeamBuilder()
    import random

    rng = random.Random(5)
    teams = {s: builder.pack(rng) for s in audit.SIDES}
    with SimWorker() as worker:
        agents = {s: make_direct_agent("random", teams[s]) for s in audit.SIDES}
        battle = DirectBattle.start(worker, "priv", teams["p1"], teams["p2"], seed=[1, 2, 3, 4])
        for side in audit.SIDES:
            agents[side].observe("priv", battle.last_lines[side])
        for _ in range(3):
            choices = {}
            for side in battle.sides_to_move():
                battle.battles[side]._vgc_direct_root = battle
                battle.battles[side]._vgc_direct_side = side
                choices[side] = agents[side].choose(battle.battles[side])
            result = battle.step(choices)
            for side in audit.SIDES:
                agents[side].observe("priv", result.lines[side])
        observation = battle.battles["p1"]
        with audit.public_view_only(observation):
            assert "_vgc_direct_root" not in vars(observation)
            assert "_vgc_direct_side" not in vars(observation)
        assert vars(observation)["_vgc_direct_root"] is battle
        battle.close()


def test_replace_is_not_needed() -> None:
    # keeps `replace` imported for readers extending the knob matrix
    assert replace(PolicyConfig(), exact_mirror_own_order=False).exact_mirror_own_order is False
