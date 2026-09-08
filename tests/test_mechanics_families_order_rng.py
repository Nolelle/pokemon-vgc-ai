"""Showdown-executed contracts for turn-order and randomness mechanics families.

Each test builds an isolating `DirectBattle` root, runs it through
`evaluate_exact_branches` with explicit `future_seeds`, and asserts the engine
property on the fogged branch state / protocol. Custom Game does not validate
legality, so species/move/ability combinations here isolate one mechanic.
"""

from __future__ import annotations

import pytest

from tests.mechanics_family_helpers import (
    DOUBLES,
    _IDS,
    _branches,
    _choice,
    _digest,
    _foe_partner,
    _has,
    _move_slots,
    _own,
    _partner,
    _root,
    _seeds,
    _slot_before,
    _status_of,
    _volatile_time,
)
from tests.sim_harness import pokeset
from vgc.damage import FieldState, PokemonState, damage_range
from vgc.mechanics_state import snapshot_battle
from vgc.rl.env import DirectBattle, SimWorker

pytestmark = pytest.mark.integration


# --- turn_priority_and_speed -----------------------------------------------------------


def test_faster_pokemon_acts_before_a_slower_one(worker) -> None:
    p1 = [
        pokeset(
            "Garchomp",
            ["tackle", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Inner Focus",
        ),
        _partner(),
    ]
    p2 = [
        pokeset(
            "Snorlax",
            ["tackle", "protect"],
            nature="Brave",
            sp={"hp": 32, "atk": 32, "def": 2},
            ability="Inner Focus",
        ),
        _foe_partner(),
    ]
    with _root(worker, p1, p2) as root:
        [branch] = _branches(
            root, _choice("tackle"), _choice("tackle"), [[11, 12, 13, 14]], "speed"
        )
    assert _slot_before(branch, "p1a", "p2a")


def test_plus_one_priority_from_a_slower_pokemon_beats_a_faster_normal_move(worker) -> None:
    p1 = [
        pokeset(
            "Snorlax",
            ["quickattack", "protect"],
            nature="Brave",
            sp={"hp": 32, "atk": 32, "def": 2},
            ability="Inner Focus",
        ),
        _partner(),
    ]
    p2 = [
        pokeset(
            "Garchomp",
            ["tackle", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Inner Focus",
        ),
        _foe_partner(),
    ]
    with _root(worker, p1, p2) as root:
        [branch] = _branches(
            root, _choice("quickattack"), _choice("tackle"), [[21, 22, 23, 24]], "prio"
        )
    assert _slot_before(branch, "p1a", "p2a")


def test_trick_room_reverses_speed_order(worker) -> None:
    p1 = [
        pokeset(
            "Garchomp",
            ["tackle", "trickroom"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Inner Focus",
        ),
        _partner(),
    ]
    p2 = [
        pokeset(
            "Snorlax",
            ["tackle", "protect"],
            nature="Brave",
            sp={"hp": 32, "atk": 32, "def": 2},
            ability="Inner Focus",
        ),
        _foe_partner(),
    ]
    with _root(worker, p1, p2, seed=(5, 6, 7, 8)) as root:
        root.step({"p1": _choice("trickroom"), "p2": _choice("protect")})
        assert any(effect.id == "trickroom" for effect in snapshot_battle(root.battles["p1"]).fields)
        [branch] = _branches(
            root, _choice("tackle"), _choice("tackle"), [[31, 32, 33, 34]], "tr"
        )
    assert _slot_before(branch, "p2a", "p1a")


# --- speed_ties ------------------------------------------------------------------------


def test_speed_ties_realize_both_orderings(worker) -> None:
    # n=64 independent 50/50 Speed ties. P(only one ordering) = 2 * 2^{-64} ≈ 3.6e-20.
    identical = pokeset(
        "Garchomp",
        ["tackle", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
    )
    p1 = [identical, _partner()]
    p2 = [
        pokeset(
            "Garchomp",
            ["tackle", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Inner Focus",
        ),
        _foe_partner(),
    ]
    with _root(worker, p1, p2) as root:
        branches = _branches(
            root, _choice("tackle"), _choice("tackle"), _seeds(64, origin=100), "tie"
        )
    p1_first = sum(_slot_before(branch, "p1a", "p2a") for branch in branches)
    p2_first = sum(_slot_before(branch, "p2a", "p1a") for branch in branches)
    assert p1_first + p2_first == 64
    assert p1_first > 0 and p2_first > 0


# --- random_branch_order ---------------------------------------------------------------


def test_identical_future_seeds_are_byte_identical_and_different_seeds_can_differ(worker) -> None:
    identical = pokeset(
        "Garchomp",
        ["tackle", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
    )
    p1 = [identical, _partner()]
    p2 = [
        pokeset(
            "Garchomp",
            ["tackle", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Inner Focus",
        ),
        _foe_partner(),
    ]
    same = [40, 41, 42, 43]
    with _root(worker, p1, p2, seed=(9, 9, 9, 9)) as root:
        first = _branches(root, _choice("tackle"), _choice("tackle"), [same], "det-a")
        second = _branches(root, _choice("tackle"), _choice("tackle"), [same], "det-b")
        varied = _branches(
            root, _choice("tackle"), _choice("tackle"), _seeds(16, origin=400), "det-var"
        )
    assert _digest(first[0]) == _digest(second[0])
    # Speed-tie 50/50 over 16 seeds: P(all identical orderings) = 2 * 2^{-16} ≈ 3.1e-5,
    # and damage rolls make collisions rarer still.
    assert len({_digest(branch) for branch in varied}) > 1


# --- damage_roll_distribution ----------------------------------------------------------


def test_damage_rolls_span_the_calculator_range_and_never_leave_it(worker) -> None:
    # n=64. 16 legal rolls; P(only one distinct value) is far below 1e-6.
    # Battle Armor keeps crits from escaping damage_range's non-crit min..max.
    attacker = pokeset(
        "Garchomp",
        ["tackle", "protect"],
        nature="Adamant",
        sp={"hp": 30, "atk": 32, "spe": 4},
        ability="Inner Focus",
    )
    defender = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Battle Armor",
    )
    expected = damage_range(
        PokemonState(
            "garchomp",
            sp_spread={"hp": 30, "atk": 32, "spe": 4},
            nature="adamant",
            ability="innerfocus",
        ),
        PokemonState(
            "snorlax",
            sp_spread={"hp": 32, "def": 32, "spd": 2},
            nature="bold",
            ability="battlearmor",
        ),
        "tackle",
        FieldState(is_doubles=True, num_targets=1),
    )
    with _root(worker, [defender, _partner()], [attacker, _foe_partner()]) as root:
        before = snapshot_battle(root.battles["p1"])
        start_hp = next(
            mon.current_hp for mon in before.our_side.pokemon if mon.species_id == "snorlax"
        )
        assert start_hp is not None
        branches = _branches(
            root, _choice("splash"), _choice("tackle"), _seeds(64, origin=500), "roll"
        )
    damages = []
    for branch in branches:
        assert not _has(branch, "|-crit|")
        hp = _own(branch, "p1", "snorlax").current_hp
        assert hp is not None
        dealt = start_hp - hp
        assert expected.min_damage <= dealt <= expected.max_damage, dealt
        damages.append(dealt)
    assert len(set(damages)) >= 2
    assert expected.min_damage < expected.max_damage


# --- accuracy_and_evasion --------------------------------------------------------------


def test_full_accuracy_never_misses_low_accuracy_does_both_always_hit_ignores_evasion(
    worker,
) -> None:
    attacker = pokeset(
        "Garchomp",
        ["tackle", "zapcannon", "aerialace", "protect"],
        nature="Modest",
        sp={"hp": 32, "spa": 32, "spe": 2},
        ability="Inner Focus",
    )
    defender = pokeset(
        "Snorlax",
        ["splash", "protect", "minimize"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Battle Armor",
    )
    p1 = [attacker, _partner()]
    p2 = [defender, _foe_partner()]
    with _root(worker, p1, p2) as root:
        full = _branches(
            root, _choice("tackle"), _choice("splash"), _seeds(32, origin=600), "acc100"
        )
        low = _branches(
            root, _choice("zapcannon"), _choice("splash"), _seeds(64, origin=700), "acc50"
        )
        # Minimize is +2 evasion; three uses reach +6.
        for _ in range(3):
            root.step({"p1": _choice("protect"), "p2": _choice("minimize")})
        always = _branches(
            root, _choice("aerialace"), _choice("splash"), _seeds(32, origin=800), "acceva"
        )
    assert all("p1a" in _move_slots(branch) for branch in full)
    assert all(not _has(branch, "|-miss|") for branch in full)
    # Zap Cannon 50%, n=64: P(all hit or all miss) = 2 * 2^{-64} ≈ 3.6e-20.
    hits = sum(not _has(branch, "|-miss|") for branch in low)
    misses = 64 - hits
    assert hits > 0 and misses > 0
    assert all("p1a" in _move_slots(branch) for branch in always)
    assert all(not _has(branch, "|-miss|") for branch in always)


# --- critical_hits ---------------------------------------------------------------------


def test_normal_high_crit_ratio_sometimes_crits_above_non_crit_max(worker) -> None:
    # Night Slash is high-crit-ratio (1/8). n=128: P(zero crits)=(7/8)^128 ≈ 5.7e-8.
    attacker = pokeset(
        "Garchomp",
        ["nightslash", "protect"],
        nature="Adamant",
        sp={"hp": 30, "atk": 32, "spe": 4},
        ability="Inner Focus",
    )
    defender = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Inner Focus",
    )
    expected = damage_range(
        PokemonState(
            "garchomp",
            sp_spread={"hp": 30, "atk": 32, "spe": 4},
            nature="adamant",
            ability="innerfocus",
        ),
        PokemonState(
            "snorlax",
            sp_spread={"hp": 32, "def": 32, "spd": 2},
            nature="bold",
            ability="innerfocus",
        ),
        "nightslash",
        FieldState(is_doubles=True, num_targets=1),
    )
    with _root(worker, [defender, _partner()], [attacker, _foe_partner()]) as root:
        before = snapshot_battle(root.battles["p1"])
        start_hp = next(
            mon.current_hp for mon in before.our_side.pokemon if mon.species_id == "snorlax"
        )
        assert start_hp is not None
        branches = _branches(
            root, _choice("splash"), _choice("nightslash"), _seeds(128, origin=900), "crit"
        )
    crits = [branch for branch in branches if _has(branch, "|-crit|")]
    non_crits = [branch for branch in branches if not _has(branch, "|-crit|")]
    assert crits and non_crits
    crit_damage = max(start_hp - (_own(branch, "p1", "snorlax").current_hp or 0) for branch in crits)
    assert crit_damage > expected.max_damage


def test_will_crit_move_always_crits_and_battle_armor_never_does(worker) -> None:
    thrower = pokeset(
        "Garchomp",
        ["stormthrow", "protect"],
        nature="Adamant",
        sp={"hp": 30, "atk": 32, "spe": 4},
        ability="Inner Focus",
    )
    normal = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Inner Focus",
    )
    armored = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Battle Armor",
    )
    with _root(worker, [normal, _partner()], [thrower, _foe_partner()]) as root:
        always = _branches(
            root, _choice("splash"), _choice("stormthrow"), _seeds(16, origin=1100), "willcrit"
        )
    with _root(worker, [armored, _partner()], [thrower, _foe_partner()]) as root:
        blocked = _branches(
            root, _choice("splash"), _choice("stormthrow"), _seeds(16, origin=1200), "armor"
        )
    assert all(_has(branch, "|-crit|") for branch in always)
    assert all(not _has(branch, "|-crit|") for branch in blocked)


# --- paralysis -------------------------------------------------------------------------


def _garchomp_incineroar_teams() -> tuple[list[dict], list[dict]]:
    garchomp = pokeset(
        "Garchomp",
        ["tackle", "protect", "splash"],
        nature="Serious",
        sp={"hp": 32, "atk": 32, "def": 2},
        ability="Inner Focus",
    )
    incineroar = pokeset(
        "Incineroar",
        ["tackle", "protect"],
        nature="Serious",
        sp={"hp": 32, "atk": 32, "def": 2},
        ability="Inner Focus",
    )
    klefki = pokeset(
        "Klefki",
        ["glare", "protect"],
        nature="Bold",
        sp={"hp": 32, "spe": 32, "def": 2},
        ability="Prankster",
    )
    return [garchomp, _partner()], [incineroar, klefki]


def test_paralysis_halves_speed_in_the_ordering(worker) -> None:
    # Garchomp Spe 122 outspeeds Incineroar Spe 80; after the 1/2 para drop, 61 does not.
    p1, p2 = _garchomp_incineroar_teams()
    with _root(worker, p1, p2) as root:
        [healthy] = _branches(
            root, _choice("tackle"), _choice("tackle"), [[1, 2, 3, 4]], "par-h"
        )
        root.step({"p1": _choice("splash"), "p2": _choice("protect", "glare")})
        assert _status_of(root, "p1", "garchomp") == "par"
        paralyzed = _branches(
            root, _choice("tackle"), _choice("tackle"), _seeds(32, origin=1300), "par-s"
        )
    assert _slot_before(healthy, "p1a", "p2a")
    acted = [branch for branch in paralyzed if not _has(branch, "|cant|p1a: Garchomp|par")]
    assert acted
    assert all(_slot_before(branch, "p2a", "p1a") for branch in acted)


def test_champions_full_paralysis_is_one_in_eight_not_one_in_four(worker) -> None:
    # n=512. Champions p=1/8: E[k]=64, sd=sqrt(512*1/8*7/8)≈7.48. Vanilla p=1/4: E[k]=128,
    # sd≈9.80. Window [30, 98] is ±4.5 sd under p=1/8 (two-sided false-failure ≈ 7e-6)
    # and its upper edge sits 3.1 sd below the vanilla mean, so vanilla behaviour would
    # slip through less than 0.1% of the time. Both hypotheses are discriminated.
    p1, p2 = _garchomp_incineroar_teams()
    with _root(worker, p1, p2) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("protect", "glare")})
        assert _status_of(root, "p1", "garchomp") == "par"
        branches = _branches(
            root, _choice("tackle"), _choice("protect"), _seeds(512, origin=1400), "par8"
        )
    denied = sum(_has(branch, "|cant|p1a: Garchomp|par") for branch in branches)
    assert 30 <= denied <= 98, denied


# --- freeze ----------------------------------------------------------------------------


def _flame_garchomp() -> dict:
    return pokeset(
        "Garchomp",
        ["flamethrower", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "spa": 2},
        ability="Inner Focus",
    )


def _ice_beam_torkoal() -> dict:
    # Slower than Snorlax so Ice Beam freezes AFTER Snorlax has already acted.
    return pokeset(
        "Torkoal",
        ["icebeam", "protect"],
        nature="Quiet",
        sp={"hp": 32, "spa": 32, "spd": 2},
        ability="White Smoke",
    )


def _scald_snorlax() -> dict:
    return pokeset(
        "Snorlax",
        ["scald", "tackle", "protect"],
        nature="Brave",
        sp={"hp": 32, "atk": 32, "spd": 2},
        ability="Inner Focus",
    )


def _start_frozen(worker: SimWorker) -> DirectBattle:
    """Ice Beam is 10% freeze; 80 tries fail with 0.9^80 ≈ 2.2e-4 if the engine works."""
    for offset in range(80):
        battle_id = f"fam-{next(_IDS)}"
        battle = DirectBattle.start(
            worker,
            battle_id,
            [_flame_garchomp(), _ice_beam_torkoal()],
            [_scald_snorlax(), _foe_partner()],
            battle_format=DOUBLES,
            seed=[10 + offset, 20 + offset, 30 + offset, 40 + offset],
        )
        battle.step({"p1": "team 12", "p2": "team 12"})
        battle.step({"p1": _choice("protect", "icebeam"), "p2": _choice("tackle")})
        if _status_of(battle, "p2", "snorlax") == "frz":
            return battle
        battle.close()
    raise AssertionError("Ice Beam never froze Snorlax in 80 tries")


def test_champions_freeze_thaws_by_the_three_step_limit_and_sometimes_earlier(worker) -> None:
    # Champions `frz` (data/mods/champions/conditions.ts): startTime=3, each action
    # decrements then thaws on time<=0 or randomChance(1,4). n=64 first-action samples:
    # P(no early thaw)=(3/4)^64 ≈ 1.3e-8. After two failed actions, the next always thaws.
    root = _start_frozen(worker)
    held: DirectBattle | None = None
    try:
        early = _branches(
            root, _choice("protect"), _choice("tackle"), _seeds(64, origin=1700), "frz3"
        )
        thawed_early = sum(_has(branch, "|-curestatus|p2a: Snorlax|frz") for branch in early)
        stayed = sum(_has(branch, "|cant|p2a: Snorlax|frz") for branch in early)
        assert thawed_early > 0 and stayed > 0

        for seed in _seeds(24, origin=3000):
            clone = root.clone(f"frz-hold-{next(_IDS)}", seed=seed)
            clone.step({"p1": _choice("protect"), "p2": _choice("tackle")})
            if _status_of(clone, "p2", "snorlax") != "frz":
                clone.close()
                continue
            clone.step({"p1": _choice("protect"), "p2": _choice("tackle")})
            if _status_of(clone, "p2", "snorlax") == "frz":
                held = clone
                break
            clone.close()
        assert held is not None, "never stayed frozen for two action opportunities"
        last = _branches(
            held, _choice("protect"), _choice("tackle"), _seeds(16, origin=1800), "frz1"
        )
        assert all(_has(branch, "|-curestatus|p2a: Snorlax|frz") for branch in last)
        assert all(not _has(branch, "|cant|p2a: Snorlax|frz") for branch in last)
        assert all(_own(branch, "p2", "snorlax").status is None for branch in last)
    finally:
        if held is not None:
            held.close()
        root.close()


def test_fire_hit_and_defrost_flag_move_thaw_a_frozen_pokemon(worker) -> None:
    root = _start_frozen(worker)
    try:
        fire = _branches(
            root, _choice("flamethrower"), _choice("tackle"), _seeds(8, origin=1900), "frzfire"
        )
        assert all(_has(branch, "|-curestatus|p2a: Snorlax|frz") for branch in fire)
        assert all(_own(branch, "p2", "snorlax").status is None for branch in fire)

        defrost = _branches(
            root, _choice("protect"), _choice("scald"), _seeds(8, origin=2000), "frzdefrost"
        )
        assert all(_has(branch, "|-curestatus|p2a: Snorlax|frz") for branch in defrost)
        assert all(not _has(branch, "|cant|p2a: Snorlax|frz") for branch in defrost)
        assert all("p2a" in _move_slots(branch) for branch in defrost)
    finally:
        root.close()


# --- confusion -------------------------------------------------------------------------


def test_confused_pokemon_sometimes_hits_itself_and_sometimes_acts(worker) -> None:
    # Confuse Ray rolls `random(2, 6)` (duration 2-5). The first confused action never
    # snaps out, then 33/100 self-hit. n=64: P(no self-hit)=(0.67)^64 ≈ 1.4e-12.
    garchomp = pokeset(
        "Garchomp",
        ["tackle", "protect", "splash"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
    )
    ray = pokeset(
        "Klefki",
        ["confuseray", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Inner Focus",
    )
    durations: list[int] = []
    for offset in range(24):
        with _root(
            worker,
            [garchomp, _partner()],
            [ray, _foe_partner()],
            seed=(60 + offset, 70 + offset, 80 + offset, 90 + offset),
        ) as probe:
            probe.step({"p1": _choice("splash"), "p2": _choice("confuseray")})
            remaining = _volatile_time(probe, 0, "confusion")
            assert remaining is not None
            durations.append(remaining)
    assert set(durations) <= {2, 3, 4, 5}
    assert len(set(durations)) >= 2

    with _root(worker, [garchomp, _partner()], [ray, _foe_partner()]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("confuseray")})
        assert _volatile_time(root, 0, "confusion") is not None
        branches = _branches(
            root, _choice("tackle"), _choice("protect"), _seeds(64, origin=2100), "conf"
        )
    self_hits = [branch for branch in branches if _has(branch, "|-activate|p1a: Garchomp|confusion")]
    acted = [branch for branch in branches if "p1a" in _move_slots(branch)]
    assert self_hits and acted
    assert any(_has(branch, "|-damage|p1a: Garchomp") for branch in self_hits)


def test_own_tempo_prevents_confusion(worker) -> None:
    attacker = pokeset(
        "Garchomp",
        ["confuseray", "protect"],
        nature="Timid",
        sp={"hp": 32, "spe": 32, "spa": 2},
        ability="Inner Focus",
    )
    tempo = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Own Tempo",
    )
    with _root(worker, [attacker, _partner()], [tempo, _foe_partner()]) as root:
        branches = _branches(
            root, _choice("confuseray"), _choice("splash"), _seeds(8, origin=2300), "tempo"
        )
    assert all(not _has(branch, "|-start|p2a: Snorlax|confusion") for branch in branches)
    assert all(
        all(effect.id != "confusion" for effect in _own(branch, "p2", "snorlax").effects)
        for branch in branches
    )
