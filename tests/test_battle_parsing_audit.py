"""The differential parsing audit, run small, as a standing regression gate.

`offline/audit_battle_parsing.py` compares what the bot believes after every step with
Showdown's own state. The integration tests play a fixed-seed subset plus the targeted coverage
scenarios (a slice of every category) in-process and require no unexplained mismatch and no
crash, and that the same battles WITHOUT the repairs do show defects (so the gate has teeth).
The unit tests check that each exemption predicate holds for its documented case only.

Full command (about 2 minutes on 10 cores):

    .venv/bin/python offline/audit_battle_parsing.py --battles 2400 --workers 10 --seed 99 \
        --scenarios --coverage-report --fail-on-unexplained
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))

import audit_battle_parsing as audit  # noqa: E402
import audit_exemptions as ex  # noqa: E402
import audit_protocol as proto  # noqa: E402
import audit_scenarios as scenarios  # noqa: E402

from vgc import poke_env_compat  # noqa: E402

JOB = {
    "indices": list(range(40)),
    "seed": 20261009,
    "agents": ["random", "random", "maxpower", "heuristic"],
    "pool_teams": [],
    "pool_fraction": 0.0,
}


# --- exemption predicates: the documented case is exempt, near-misses are not ----------------


def _finding(field: str, kind: str, detail: str = "", **kw) -> dict:
    return {"field": field, "kind": kind, "detail": detail, "mon": "foe:X", "truth": kw.pop("truth", 0),
            "belief": kw.pop("belief", 0), "note": "", "extra": kw.pop("extra", {})}  # fmt: skip


def test_unknown_findings_are_unexplained() -> None:
    assert ex.classify(_finding("hp", "foe_percent"), {"label": "foe"}) is None
    assert ex.classify(_finding("boosts", "mismatch", "atk"), {"tm": {}, "label": "foe"}) is None


def test_collision_exemption_needs_the_collision() -> None:
    f = _finding("hp", "foe_percent")
    assert ex.classify(f, {"tm": {"collision": True}}) == "illusion_name_collision"
    cx = {"tm": {}, "name": "A", "role": "p2", "collided": {"p2": {"A"}}}
    assert ex.classify(f, cx) == "illusion_name_collision"  # residue on a collided name
    assert ex.classify(f, {**cx, "name": "B"}) is None  # another Pokemon
    assert ex.classify(_finding("species", "mismatch", "foe"), {**cx, "name": "B"}) is None


def test_hidden_extender_exemption_is_limited_to_a_foe_setter_and_plus_three() -> None:
    mon = {"item": "damprock", "last_item": "", "set_item": "damprock"}
    truth = {"sides": {"p2": {"mons": {"Pelipper": mon}}}}

    class Snap:
        class opponent_side:  # noqa: N801
            pokemon: list = []

    base = {"truth": truth, "snapshot": Snap}
    extra = {"setter": "p2", "foe_role": "p2", "condition": "weather:raindance"}
    f = _finding("duration", "mismatch", "weather:raindance", truth=7, belief=4, extra=extra)
    assert ex.classify(f, base) == "hidden_foe_extender"
    assert ex.classify({**f, "truth": 6}, base) is None  # not a 3-turn gap
    assert ex.classify({**f, "extra": {**extra, "setter": "p1"}}, base) is None  # our own setter
    wrong = {**f, "extra": {**extra, "condition": "weather:sunnyday"}}
    assert ex.classify(wrong, base) is None  # Damp Rock does not extend sun


def test_ally_skill_swap_exemption_needs_the_hidden_swap_event() -> None:
    know = proto.ProtocolKnowledge("p1")
    know.feed(["|-activate|p2a: A|move: Skill Swap|||[of] p2b: B"])
    f = _finding("ability", "foe_wrong", "x->y")
    cx = {"label": "foe", "role": "p2", "name": "A", "know": know}
    assert ex.classify(f, cx) == "hidden_ally_skill_swap"
    assert ex.classify(f, {**cx, "name": "C"}) is None
    assert ex.classify(_finding("hp", "foe_percent"), cx) is None  # only ability findings


def test_yawn_exemption_needs_a_sleeping_pokemon() -> None:
    f = _finding("volatile", "missing", "yawn")
    assert ex.classify(f, {"tm": {"status": "slp"}}) == "yawn_kept_through_rest"
    assert ex.classify(f, {"tm": {"status": ""}}) is None
    assert ex.classify(_finding("volatile", "missing", "taunt"), {"tm": {"status": "slp"}}) is None


def test_protocol_tracker_attributes_from_the_line_not_from_the_repairs() -> None:
    know = proto.ProtocolKnowledge("p1")
    know.feed(
        [
            "|-block|p2a: Alcremie|ability: Aroma Veil|[of] p2a: Alcremie",
            "|move|p2b: Torkoal|Struggle|p1a: X",
            "|-activate|p2b: Torkoal|item: Quick Claw",
        ]
    )
    mons = {
        ("p2", "Alcremie"): _truth("aromaveil", ""),
        ("p2", "Torkoal"): _truth("drought", "quickclaw"),
    }
    assert know.resolve(mons, {}) == []
    assert know.revealed_abilities("p2", "Alcremie") == {"aromaveil"}
    assert know.mons[("p2", "Torkoal")].item_shown == "quickclaw"
    assert know.revealed_moves("p2", "Torkoal") == set()  # Struggle is nobody's move


def _truth(ability: str, item: str) -> dict:
    return {"ability": ability, "base_ability": ability, "item": item, "last_item": "",
            "base_moves": [], "moves": [], "transformed": False}  # fmt: skip


# --- the audit itself ---------------------------------------------------------------------------


@pytest.mark.integration
def test_audit_finds_no_unexplained_parsing_mismatch() -> None:
    acc = audit.worker_main({**JOB, "disable": []})
    try:
        assert acc["errors"] == []
        assert acc["compares"] > 500
        assert audit.unexplained_groups(acc) == []
    finally:
        poke_env_compat.set_disabled_fixes([])


@pytest.mark.integration
def test_targeted_coverage_scenarios_are_clean() -> None:
    every_kind = scenarios.build_scenarios()
    picked = [s for kind in ("moves", "abilities", "items") for s in every_kind if kind in s["name"]][
        ::12
    ]
    acc = scenarios.run_job({"scenarios": picked, "disable": []})
    try:
        assert acc["errors"] == []
        assert acc["compares"] > 200
        assert audit.unexplained_groups(acc) == []
        assert acc["cov"]["move_used"]
    finally:
        poke_env_compat.set_disabled_fixes([])


@pytest.mark.integration
def test_audit_has_teeth_without_the_repairs() -> None:
    try:
        acc = audit.worker_main({**JOB, "disable": list(poke_env_compat.ALL_FIXES)})
        groups = audit.unexplained_groups(acc)
    finally:
        poke_env_compat.set_disabled_fixes([])
    assert len(groups) >= 5
    assert any(g.startswith("volatile/stale/") for g in groups)


@pytest.mark.integration
def test_unbrought_pokemon_are_not_left_active_after_team_preview() -> None:
    import random

    from vgc.rl.agents import make_direct_agent
    from vgc.rl.env import DirectBattle, SimWorker

    builder = audit.TeamBuilder()
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


def _called_protect_marked(lines: list[str]) -> bool:
    know = proto.ProtocolKnowledge("p1")
    know.feed(["|switch|p2a: Foo|Garchomp, L50|100/100"])
    know.feed(lines)
    return know.has_event("p2", "Foo", "called_protect")


def test_called_protect_needs_a_protect_family_call() -> None:
    # Instruct repeating Tackle is not a called Protect.
    assert not _called_protect_marked(
        [
            "|move|p2a: Foo|Tackle|p1a: X",
            "|-singleturn|p2a: Foo|move: Instruct|[of] p2b: Bar",
            "|move|p2a: Foo|Tackle|p1a: X|[from] move: Instruct",
        ]
    )
    # Instruct repeating Protect is.
    assert _called_protect_marked(
        [
            "|move|p2a: Foo|Protect|p2a: Foo",
            "|move|p2a: Foo|Protect|p2a: Foo|[from] move: Instruct",
        ]
    )


def test_called_protect_ends_on_switch_out_and_next_own_move() -> None:
    called = "|move|p2a: Foo|Protect|p2a: Foo|[from] move: Sleep Talk"
    assert _called_protect_marked([called])
    assert not _called_protect_marked([called, "|switch|p2a: Baz|Torkoal, L50|100/100"])
    assert not _called_protect_marked([called, "|move|p2a: Foo|Tackle|p1a: X"])
    assert not _called_protect_marked([called, "|faint|p2a: Foo"])


def test_protect_counter_exemption_only_after_a_genuine_called_protect() -> None:
    f = _finding("protect_counter", "mismatch")

    def classify(lines: list[str]):
        know = proto.ProtocolKnowledge("p1")
        know.feed(["|switch|p2a: Foo|Garchomp, L50|100/100"] + lines)
        return ex.classify(f, {"know": know, "role": "p2", "name": "Foo"})

    repro = [
        "|-singleturn|p2a: Foo|move: Instruct|[of] p2b: Bar",
        "|move|p2a: Foo|Tackle|p1a: X|[from] move: Instruct",
        "|switch|p2a: Baz|Torkoal, L50|100/100",
    ]
    assert classify(repro) is None
    genuine = ["|move|p2a: Foo|Protect|p2a: Foo|[from] move: Instruct"]
    assert classify(genuine) == "called_protect_counter"
