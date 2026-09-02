"""Showdown-executed contracts for move-execution mechanics families.

Each test builds an isolating `DirectBattle` root, runs it through
`evaluate_exact_branches` with explicit `future_seeds`, and asserts the engine
property on the fogged branch state / protocol. Custom Game does not validate
legality, so species/move/ability combinations here isolate one mechanic.
"""

from __future__ import annotations

import pytest

from tests.mechanics_family_helpers import (
    _available,
    _branches,
    _choice,
    _foe_partner,
    _has,
    _hitcount,
    _hp,
    _lines,
    _max_hp,
    _move_target,
    _own,
    _partner,
    _root,
    _root_hp,
    _seeds,
    _targeted,
    _volatile_time,
)
from tests.sim_harness import pokeset
from vgc.damage import FieldState, PokemonState, damage_range
from vgc.data import load_items
from vgc.mechanics_state import snapshot_battle
from vgc.rl.env import InvalidChoice

pytestmark = pytest.mark.integration


def _garchomp(
    *,
    moves: list[str],
    nature: str = "Jolly",
    ability: str = "Inner Focus",
    item: str = "",
    **sp: int,
) -> dict:
    spread = {"hp": 32, "spe": 32, "atk": 2, **sp}
    return pokeset("Garchomp", moves, nature=nature, sp=spread, ability=ability, item=item)


def _snorlax(
    *,
    moves: list[str],
    nature: str = "Brave",
    ability: str = "Inner Focus",
    item: str = "",
    defense: int | None = None,
    **sp: int,
) -> dict:
    spread = {"hp": 32, "atk": 32, "def": 2, **sp}
    if defense is not None:
        spread["def"] = defense
    return pokeset("Snorlax", moves, nature=nature, sp=spread, ability=ability, item=item)


def _fainted(branch, side: str, species_id: str) -> bool:
    return _own(branch, side, species_id).fainted


# --- targeting_and_retargeting ---------------------------------------------------------


def test_single_target_retargets_after_faint_and_hits_the_incoming_switch(worker) -> None:
    # Faint: Extreme Speed KOs slot 1, then Tackle (originally aimed at slot 1) hits slot 2.
    # Switch: a start-of-turn switch occupies the same slot, so Tackle still hits slot 1
    # (the replacement) rather than the remaining adjacent foe. That is the engine's
    # actual retarget rule — fainted identity is gone; a filled slot is still valid.
    ko_user = _garchomp(moves=["extremespeed", "protect"], nature="Adamant", atk=32, spe=2)
    tackler = _snorlax(moves=["tackle", "protect"], nature="Brave", atk=32, spe=0, defense=2)
    frail = pokeset(
        "Diglett",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 0, "def": 0, "spd": 0},
        ability="Inner Focus",
    )
    bulky = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, atk=2)
    with _root(worker, [ko_user, tackler], [frail, bulky]) as root:
        [faint] = _branches(
            root,
            f"{_targeted('extremespeed', 1)}, {_targeted('tackle', 1)}",
            _choice("splash"),
            [[1, 2, 3, 4]],
            "retarget-faint",
        )
    assert _fainted(faint, "p2", "diglett")
    assert _move_target(faint, "p1b", "Tackle") == "p2b"
    assert not _fainted(faint, "p2", "snorlax")

    replacement = pokeset(
        "Blissey",
        ["splash", "protect"],
        nature="Calm",
        sp={"hp": 32, "spd": 32, "def": 2},
        ability="Natural Cure",
    )
    with _root(
        worker,
        [_garchomp(moves=["tackle", "protect"]), _partner()],
        [frail, bulky, replacement],
        p2_order="123",
    ) as root:
        [switched] = _branches(
            root,
            _choice("tackle"),
            "switch 3, move protect",
            [[5, 6, 7, 8]],
            "retarget-switch",
        )
    assert _move_target(switched, "p1a", "Tackle") == "p2a"
    assert _hp(switched, "p2", "blissey") < _max_hp(switched, "p2", "blissey")
    assert _hp(switched, "p2", "snorlax") == _max_hp(switched, "p2", "snorlax")


def test_follow_me_and_rage_powder_redirect_except_grass_and_overcoat(worker) -> None:
    # Safety Goggles is not in data/champions/items.json (not a legal Champions item),
    # so powder immunity is covered by Grass-type and Overcoat instead.
    attacker = _garchomp(moves=["tackle", "protect"])
    follow = pokeset(
        "Clefable",
        ["followme", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Magic Guard",
    )
    rage = pokeset(
        "Amoonguss",
        ["ragepowder", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Effect Spore",
    )
    target = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, atk=2)
    with _root(worker, [attacker, _partner()], [target, follow]) as root:
        [followed] = _branches(
            root, _choice("tackle"), _choice("splash", "followme"), [[9, 10, 11, 12]], "follow"
        )
    assert _move_target(followed, "p1a", "Tackle") == "p2b"

    with _root(worker, [attacker, _partner()], [target, rage]) as root:
        [powder] = _branches(
            root, _choice("tackle"), _choice("splash", "ragepowder"), [[13, 14, 15, 16]], "rage"
        )
    assert _move_target(powder, "p1a", "Tackle") == "p2b"

    grass = pokeset(
        "Venusaur",
        ["tackle", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Overgrow",
    )
    with _root(worker, [grass, _partner()], [target, rage]) as root:
        [immune] = _branches(
            root, _choice("tackle"), _choice("splash", "ragepowder"), [[17, 18, 19, 20]], "grass"
        )
    assert _move_target(immune, "p1a", "Tackle") == "p2a"

    overcoat = _garchomp(moves=["tackle", "protect"], ability="Overcoat")
    with _root(worker, [overcoat, _partner()], [target, rage]) as root:
        [coated] = _branches(
            root, _choice("tackle"), _choice("splash", "ragepowder"), [[21, 22, 23, 24]], "coat"
        )
    assert _move_target(coated, "p1a", "Tackle") == "p2a"


def test_spread_move_damage_is_three_quarters_when_two_targets_are_hit(worker) -> None:
    attacker = pokeset(
        "Charizard",
        ["heatwave", "protect"],
        nature="Modest",
        sp={"hp": 32, "spa": 32, "spe": 2},
        ability="No Guard",
    )
    partner = _garchomp(
        moves=["extremespeed", "splash", "protect"],
        nature="Jolly",
        atk=32,
    )
    fodder = pokeset(
        "Diglett",
        ["splash", "protect"],
        nature="Brave",
        sp={"hp": 0, "def": 0, "atk": 2},
        ability="Sand Veil",
    )
    torkoal = pokeset(
        "Torkoal",
        ["splash", "protect"],
        nature="Calm",
        sp={"hp": 32, "spd": 32, "def": 2},
        ability="Battle Armor",
    )
    one = damage_range(
        PokemonState(
            "charizard",
            sp_spread={"hp": 32, "spa": 32, "spe": 2},
            nature="modest",
            ability="noguard",
        ),
        PokemonState(
            "torkoal",
            sp_spread={"hp": 32, "spd": 32, "def": 2},
            nature="calm",
            ability="battlearmor",
        ),
        "heatwave",
        FieldState(is_doubles=True, num_targets=1),
    )
    two = damage_range(
        PokemonState(
            "charizard",
            sp_spread={"hp": 32, "spa": 32, "spe": 2},
            nature="modest",
            ability="noguard",
        ),
        PokemonState(
            "torkoal",
            sp_spread={"hp": 32, "spd": 32, "def": 2},
            nature="calm",
            ability="battlearmor",
        ),
        "heatwave",
        FieldState(is_doubles=True, num_targets=2),
    )
    assert two.max_damage < one.max_damage
    # Protecting one of two adjacent foes does not restore 1.0x: Showdown
    # still applies the spread modifier from the two targeted slots. Isolate
    # num_targets=1 by fainting the extra slot before Heat Wave resolves.
    with _root(worker, [fodder, torkoal], [attacker, partner]) as root:
        start = _root_hp(root, "p1", "torkoal")
        single = _branches(
            root,
            _choice("splash", "splash"),
            _choice("heatwave", "move extremespeed 1"),
            _seeds(16, origin=300),
            "spread1",
        )
        double = _branches(
            root,
            _choice("splash", "splash"),
            _choice("heatwave", "splash"),
            _seeds(16, origin=320),
            "spread2",
        )
    for branch in single:
        dealt = start - _hp(branch, "p1", "torkoal")
        assert one.min_damage <= dealt <= one.max_damage, dealt
        assert _fainted(branch, "p1", "diglett")
    for branch in double:
        dealt = start - _hp(branch, "p1", "torkoal")
        # Showdown can land one past damage_range's two-target max (0.75x
        # rounded after the roll vs floored in the calculator). Still strictly
        # below the one-target range.
        assert two.min_damage <= dealt <= two.max_damage + 1, dealt
        assert dealt < one.min_damage, dealt


# --- protect_family --------------------------------------------------------------------


def test_protect_blocks_damage_and_consecutive_protect_is_one_in_three(worker) -> None:
    attacker = _garchomp(moves=["tackle", "protect"], nature="Adamant", atk=32, spe=4)
    defender = pokeset(
        "Blissey",
        ["protect", "splash"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Natural Cure",
    )
    with _root(worker, [defender, _snorlax(moves=["splash", "protect"])], [attacker, _foe_partner()]) as root:
        [blocked] = _branches(
            root, _choice("protect", "splash"), _choice("tackle"), [[1, 2, 3, 4]], "prot1"
        )
        assert _has(blocked, "|-singleturn|p1a: Blissey|Protect")
        assert _hp(blocked, "p1", "blissey") == _max_hp(blocked, "p1", "blissey")
        root.step({"p1": _choice("protect", "splash"), "p2": _choice("protect")})
        # Consecutive Protect is 1/3. n=64: P(all succeed)=(2/3)^64 ≈ 2.6e-12;
        # P(all fail)=(1/3)^64 is smaller still. Partner splashes so stall is isolated
        # onto p1a; Protect is +4 so the opponent's Tackle is still queued (willAct).
        second = _branches(
            root,
            _choice("protect", "splash"),
            _choice("tackle"),
            _seeds(64, origin=400),
            "prot2",
        )
    successes = sum(_has(branch, "|-singleturn|p1a: Blissey|Protect") for branch in second)
    fails = sum(
        any("|move|p1a: Blissey|Protect||[still]" in line for line in _lines(branch))
        for branch in second
    )
    assert successes > 0 and fails > 0
    assert successes + fails == 64


def test_wide_guard_quick_guard_and_feint(worker) -> None:
    guarder = pokeset(
        "Hitmontop",
        ["wideguard", "quickguard", "protect", "splash"],
        nature="Impish",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Inner Focus",
    )
    spreader = pokeset(
        "Charizard",
        ["heatwave", "quickattack", "feint", "protect"],
        nature="Timid",
        sp={"hp": 32, "spa": 32, "spe": 2, "atk": 4},
        ability="Inner Focus",
    )
    single = _garchomp(
        moves=["tackle", "quickattack", "feint", "protect"], nature="Adamant", atk=32
    )
    bulky = _snorlax(
        moves=["splash", "protect"],
        nature="Bold",
        defense=32,
        hp=32,
        atk=2,
        ability="Magic Guard",
    )
    # Bulky in slot 1 so the default single-target is Snorlax; guarder is slot 2.
    with _root(worker, [bulky, guarder], [spreader, single]) as root:
        start = _root_hp(root, "p1", "snorlax")
        [wide_spread] = _branches(
            root, _choice("splash", "wideguard"), _choice("heatwave"), [[1, 2, 3, 4]], "wide-s"
        )
        [wide_single] = _branches(
            root,
            _choice("splash", "wideguard"),
            _choice("protect", "tackle"),
            [[5, 6, 7, 8]],
            "wide-1",
        )
        [quick_prio] = _branches(
            root,
            _choice("splash", "quickguard"),
            _choice("protect", "quickattack"),
            [[9, 10, 11, 12]],
            "qg-p",
        )
        [quick_normal] = _branches(
            root,
            _choice("splash", "quickguard"),
            _choice("protect", "tackle"),
            [[13, 14, 15, 16]],
            "qg-n",
        )
    assert _hp(wide_spread, "p1", "snorlax") == start
    assert _hp(wide_spread, "p1", "hitmontop") == _max_hp(wide_spread, "p1", "hitmontop")
    assert _hp(wide_single, "p1", "snorlax") < start
    assert _hp(quick_prio, "p1", "snorlax") == start
    assert _hp(quick_normal, "p1", "snorlax") < start

    with _root(worker, [bulky, _partner()], [single, _foe_partner()]) as root:
        [feint] = _branches(
            root, _choice("protect"), _choice("feint"), [[17, 18, 19, 20]], "feint"
        )
    assert _hp(feint, "p1", "snorlax") < _max_hp(feint, "p1", "snorlax")
    assert not _has(feint, "|-activate|p1a: Snorlax|move: Protect") or _has(feint, "Feint")


# --- one_hit_knockout_moves ------------------------------------------------------------


def test_ohko_hits_ko_and_immunities(worker) -> None:
    # Equal level-50 accuracy is 30%. n=64: P(all miss)=(0.7)^64 ≈ 1.3e-10;
    # P(all hit)=(0.3)^64 is smaller. Hits always KO. Sturdy / Flying / Ice are exact
    # immunities, not rolls.
    user = pokeset(
        "Garchomp",
        ["fissure", "sheercold", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
    )
    normal = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, atk=2)
    sturdy = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, atk=2, ability="Sturdy")
    flying = pokeset(
        "Crobat",
        ["splash", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "def": 2},
        ability="Inner Focus",
    )
    ice = pokeset(
        "Lapras",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Water Absorb",
    )
    with _root(worker, [user, _partner()], [normal, _foe_partner()]) as root:
        fissure = _branches(
            root, _choice("fissure"), _choice("splash"), _seeds(64, origin=500), "fissure"
        )
        sheer = _branches(
            root, _choice("sheercold"), _choice("splash"), _seeds(64, origin=600), "sheer"
        )
    hits = [b for b in fissure if _has(b, "|-ohko|") or _fainted(b, "p2", "snorlax")]
    misses = [b for b in fissure if _has(b, "|-miss|") or not _fainted(b, "p2", "snorlax")]
    assert hits and misses
    assert all(_fainted(b, "p2", "snorlax") for b in hits)
    sheer_hits = [b for b in sheer if _fainted(b, "p2", "snorlax")]
    assert sheer_hits
    assert all(_fainted(b, "p2", "snorlax") for b in sheer_hits)

    with _root(worker, [user, _partner()], [sturdy, _foe_partner()]) as root:
        blocked = _branches(
            root, _choice("fissure"), _choice("splash"), _seeds(16, origin=700), "sturdy"
        )
    assert all(not _fainted(b, "p2", "snorlax") for b in blocked)
    assert all(_has(b, "|-immune|") or _has(b, "|-fail|") for b in blocked)

    with _root(worker, [user, _partner()], [flying, _foe_partner()]) as root:
        fly = _branches(
            root, _choice("fissure"), _choice("splash"), _seeds(8, origin=720), "flyohko"
        )
    assert all(not _fainted(b, "p2", "crobat") for b in fly)
    assert all(_has(b, "|-immune|") for b in fly)

    with _root(worker, [user, _partner()], [ice, _foe_partner()]) as root:
        iced = _branches(
            root, _choice("sheercold"), _choice("splash"), _seeds(8, origin=740), "iceohko"
        )
    assert all(not _fainted(b, "p2", "lapras") for b in iced)
    assert all(_has(b, "|-immune|") for b in iced)


# --- multi_hit_moves -------------------------------------------------------------------


def test_multi_hit_counts_skill_link_population_bomb_double_kick(worker) -> None:
    # Rock Blast/Bullet Seed (Gen 5+): 2/3/4/5 at 35/35/15/15. n=64, No Guard so accuracy
    # does not drop hits. P(never 2)=(0.65)^64 ≈ 3.6e-13; P(never 5)=(0.85)^64 ≈ 3.5e-5.
    blaster = pokeset(
        "Garchomp",
        ["rockblast", "bulletseed", "doublekick", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "spe": 2},
        ability="No Guard",
    )
    linked = pokeset(
        "Cloyster",
        ["rockblast", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "spe": 2},
        ability="Skill Link",
    )
    bomber = pokeset(
        "Maushold",
        ["populationbomb", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Technician",
    )
    wall = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, spd=32, atk=2, ability="Battle Armor")
    with _root(worker, [blaster, _partner()], [wall, _foe_partner()]) as root:
        rock = _branches(
            root, _choice("rockblast"), _choice("splash"), _seeds(64, origin=800), "rockblast"
        )
        seed = _branches(
            root, _choice("bulletseed"), _choice("splash"), _seeds(64, origin=900), "bulletseed"
        )
        kicks = _branches(
            root, _choice("doublekick"), _choice("splash"), _seeds(8, origin=1000), "dkick"
        )
    rock_counts = { _hitcount(b) for b in rock }
    seed_counts = { _hitcount(b) for b in seed }
    assert None not in rock_counts and rock_counts <= {2, 3, 4, 5}
    assert 2 in rock_counts and 5 in rock_counts
    assert None not in seed_counts and seed_counts <= {2, 3, 4, 5}
    assert 2 in seed_counts and 5 in seed_counts
    assert all(_hitcount(b) == 2 for b in kicks)

    with _root(worker, [linked, _partner()], [wall, _foe_partner()]) as root:
        skill = _branches(
            root, _choice("rockblast"), _choice("splash"), _seeds(8, origin=1100), "skilllink"
        )
    assert all(_hitcount(b) == 5 for b in skill)

    # Population Bomb: 90% per-hit accuracy, up to 10. n=32: P(only one distinct count)
    # is dominated by always-10, (0.9^10)^32 ≈ 4e-15.
    with _root(worker, [bomber, _partner()], [wall, _foe_partner()]) as root:
        bombs = _branches(
            root, _choice("populationbomb"), _choice("splash"), _seeds(32, origin=1200), "popbomb"
        )
    bomb_counts = {_hitcount(b) for b in bombs if _hitcount(b) is not None}
    assert bomb_counts <= set(range(1, 11))
    assert len(bomb_counts) >= 2


# --- secondary_effects -----------------------------------------------------------------


def test_thunderbolt_paralysis_serene_grace_sheer_force_shield_dust(worker) -> None:
    modest = dict(nature="Modest", sp={"hp": 32, "spa": 32, "spe": 2})
    bolt = pokeset("Garchomp", ["thunderbolt", "protect"], ability="Inner Focus", **modest)
    grace = pokeset("Garchomp", ["thunderbolt", "protect"], ability="Serene Grace", **modest)
    force = pokeset("Garchomp", ["thunderbolt", "protect"], ability="Sheer Force", **modest)
    target = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, spd=32, atk=2, ability="Inner Focus")
    dusted = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, spd=32, atk=2, ability="Shield Dust")

    with _root(worker, [bolt, _partner()], [target, _foe_partner()]) as root:
        # 10% secondary. n=64: P(all fire or none)=0.1^64 + 0.9^64 ≈ 1.2e-3 for none;
        # raise to n=96 so P(zero)=(0.9)^96 ≈ 4.2e-5.
        base = _branches(
            root, _choice("thunderbolt"), _choice("splash"), _seeds(96, origin=1300), "tb10"
        )
    base_para = sum(_own(b, "p2", "snorlax").status == "par" or _has(b, "|-status|p2a: Snorlax|par") for b in base)
    assert 0 < base_para < 96

    with _root(worker, [grace, _partner()], [target, _foe_partner()]) as root:
        # Serene Grace doubles to 20%. n=1024: E[k]=204.8, sd≈12.8. Window [150, 260] is
        # about −4.3/+4.3 sd under p=0.2 (two-sided false-failure ≈ 2e-5). The lower edge
        # is 4.7 sd above the p=0.10 mean (102.4, sd≈9.6), so an un-doubled secondary
        # would slip through with probability ≈ 1e-6.
        sg = _branches(
            root, _choice("thunderbolt"), _choice("splash"), _seeds(1024, origin=1400), "sg20"
        )
    sg_para = sum(_own(b, "p2", "snorlax").status == "par" or _has(b, "|-status|p2a: Snorlax|par") for b in sg)
    assert 150 <= sg_para <= 260, sg_para

    with _root(worker, [force, _partner()], [target, _foe_partner()]) as root:
        # n=64. If the 10% secondary survived, P(zero para)=(0.9)^64 ≈ 1.2e-3.
        forced = _branches(
            root, _choice("thunderbolt"), _choice("splash"), _seeds(64, origin=1700), "sforce"
        )
    assert all(_own(b, "p2", "snorlax").status is None for b in forced)
    assert all(not _has(b, "|-status|p2a: Snorlax|par") for b in forced)

    with _root(worker, [bolt, _partner()], [dusted, _foe_partner()]) as root:
        dust = _branches(
            root, _choice("thunderbolt"), _choice("splash"), _seeds(64, origin=1800), "sdust"
        )
    assert all(_own(b, "p2", "snorlax").status is None for b in dust)


# --- recoil_drain_and_crash ------------------------------------------------------------


def test_recoil_drain_crash_and_life_orb(worker) -> None:
    bird = pokeset(
        "Staraptor",
        ["bravebird", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "spe": 2},
        ability="Inner Focus",
    )
    drainer = pokeset(
        "Venusaur",
        ["gigadrain", "splash", "protect"],
        nature="Modest",
        sp={"hp": 32, "spa": 32, "spe": 2},
        ability="Inner Focus",
    )
    kicker = pokeset(
        "Hitmonlee",
        ["highjumpkick", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "spe": 2},
        ability="Limber",
    )
    orb_user = _garchomp(moves=["tackle", "protect"], nature="Adamant", atk=32, spe=4, item="Life Orb")
    wall = _snorlax(moves=["splash", "protect", "tackle"], nature="Bold", defense=32, hp=32, atk=2, ability="Battle Armor")
    ooze = _snorlax(moves=["splash", "protect"], nature="Bold", defense=32, hp=32, atk=2, ability="Liquid Ooze")

    with _root(worker, [bird, _partner()], [wall, _foe_partner()]) as root:
        start_foe = _root_hp(root, "p2", "snorlax")
        start_user = _root_hp(root, "p1", "staraptor")
        [recoil] = _branches(
            root, _choice("bravebird"), _choice("splash"), [[1, 2, 3, 4]], "recoil"
        )
    dealt = start_foe - _hp(recoil, "p2", "snorlax")
    taken = start_user - _hp(recoil, "p1", "staraptor")
    expected_recoil = max(round(dealt / 3), 1)
    assert taken == expected_recoil, (dealt, taken, expected_recoil)

    with _root(worker, [drainer, _partner()], [wall, _foe_partner()]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("tackle")})
        damaged = _root_hp(root, "p1", "venusaur")
        foe_hp = _root_hp(root, "p2", "snorlax")
        max_hp = next(
            p.max_hp
            for p in snapshot_battle(root.battles["p1"]).our_side.pokemon
            if p.species_id == "venusaur"
        )
        assert max_hp is not None and damaged < max_hp
        [drain] = _branches(
            root, _choice("gigadrain"), _choice("splash"), [[5, 6, 7, 8]], "drain"
        )
    drained = foe_hp - _hp(drain, "p2", "snorlax")
    healed = _hp(drain, "p1", "venusaur") - damaged
    expected_heal = min(round(drained / 2), max_hp - damaged)
    assert healed == expected_heal, (drained, healed, expected_heal)

    with _root(worker, [drainer, _partner()], [ooze, _foe_partner()]) as root:
        start_ooze_user = _root_hp(root, "p1", "venusaur")
        [oozed] = _branches(
            root, _choice("gigadrain"), _choice("splash"), [[9, 10, 11, 12]], "ooze"
        )
    assert _hp(oozed, "p1", "venusaur") < start_ooze_user
    assert _has(oozed, "Liquid Ooze") or _has(oozed, "|-damage|p1a: Venusaur")

    with _root(worker, [kicker, _partner()], [wall, _foe_partner()]) as root:
        start_kicker = _root_hp(root, "p1", "hitmonlee")
        max_kicker = next(
            p.max_hp
            for p in snapshot_battle(root.battles["p1"]).our_side.pokemon
            if p.species_id == "hitmonlee"
        )
        assert max_kicker is not None
        [crash] = _branches(
            root, _choice("highjumpkick"), _choice("protect"), [[13, 14, 15, 16]], "crash"
        )
    crash_taken = start_kicker - _hp(crash, "p1", "hitmonlee")
    assert crash_taken == max_kicker // 2, (crash_taken, max_kicker)

    with _root(worker, [orb_user, _partner()], [wall, _foe_partner()]) as root:
        start_orb = _root_hp(root, "p1", "garchomp")
        max_orb = next(
            p.max_hp
            for p in snapshot_battle(root.battles["p1"]).our_side.pokemon
            if p.species_id == "garchomp"
        )
        assert max_orb is not None
        [orb] = _branches(
            root, _choice("tackle"), _choice("splash"), [[17, 18, 19, 20]], "lifeorb"
        )
    assert start_orb - _hp(orb, "p1", "garchomp") == max_orb // 10


# --- survival_effects ------------------------------------------------------------------


def test_focus_sash_sturdy_endure_and_disguise(worker) -> None:
    killer = _garchomp(moves=["earthquake", "protect"], nature="Adamant", atk=32, spe=4)
    chipper = pokeset(
        "Blissey",
        ["pound", "protect"],
        nature="Modest",
        sp={"hp": 32, "spa": 4, "atk": 0},
        ability="Inner Focus",
    )
    sashed = pokeset(
        "Diglett",
        ["splash", "protect"],
        nature="Jolly",
        sp={"hp": 0, "spe": 32, "def": 0},
        ability="Inner Focus",
        item="Focus Sash",
    )
    sturdy = pokeset(
        "Diglett",
        ["splash", "protect"],
        nature="Jolly",
        sp={"hp": 0, "spe": 32, "def": 0},
        ability="Sturdy",
    )
    endurer = pokeset(
        "Diglett",
        ["endure", "splash"],
        nature="Jolly",
        sp={"hp": 0, "spe": 32, "def": 0},
        ability="Inner Focus",
    )
    mimic = pokeset(
        "Mimikyu",
        ["splash", "protect"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "def": 2},
        ability="Disguise",
    )

    with _root(worker, [sashed, _partner()], [killer, _foe_partner()]) as root:
        [sash] = _branches(
            root, _choice("splash"), _choice("earthquake"), [[1, 2, 3, 4]], "sash-full"
        )
    assert not _fainted(sash, "p1", "diglett")
    assert _hp(sash, "p1", "diglett") == 1

    with _root(worker, [sashed, _partner()], [chipper, killer], p2_order="12") as root:
        root.step({"p1": _choice("splash"), "p2": _choice("pound")})
        chipped = _root_hp(root, "p1", "diglett")
        assert 1 < chipped
        [broken] = _branches(
            root, _choice("splash"), _choice("protect", "earthquake"), [[5, 6, 7, 8]], "sash-chip"
        )
    assert _fainted(broken, "p1", "diglett")

    with _root(worker, [sturdy, _partner()], [killer, _foe_partner()]) as root:
        [sturdy_full] = _branches(
            root, _choice("splash"), _choice("earthquake"), [[9, 10, 11, 12]], "sturdy-full"
        )
    assert not _fainted(sturdy_full, "p1", "diglett")
    assert _hp(sturdy_full, "p1", "diglett") == 1

    with _root(worker, [sturdy, _partner()], [chipper, killer]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("pound")})
        assert _root_hp(root, "p1", "diglett") < next(
            p.max_hp
            for p in snapshot_battle(root.battles["p1"]).our_side.pokemon
            if p.species_id == "diglett"
        )
        [sturdy_chip] = _branches(
            root, _choice("splash"), _choice("protect", "earthquake"), [[13, 14, 15, 16]], "sturdy-chip"
        )
    assert _fainted(sturdy_chip, "p1", "diglett")

    with _root(worker, [endurer, _partner()], [killer, _foe_partner()]) as root:
        [endured] = _branches(
            root, _choice("endure"), _choice("earthquake"), [[17, 18, 19, 20]], "endure"
        )
    assert not _fainted(endured, "p1", "diglett")
    assert _hp(endured, "p1", "diglett") == 1

    with _root(worker, [mimic, _partner()], [killer, _foe_partner()]) as root:
        start_mimic = _root_hp(root, "p1", "mimikyu")
        max_mimic = next(
            p.max_hp
            for p in snapshot_battle(root.battles["p1"]).our_side.pokemon
            if p.species_id == "mimikyu"
        )
        assert max_mimic is not None
        [mask] = _branches(
            root, _choice("splash"), _choice("earthquake"), [[21, 22, 23, 24]], "disguise"
        )
    assert _has(mask, "Disguise")
    busted = next(
        mon
        for mon in mask.state_for("p1").our_side.pokemon
        if mon.species_id and "mimikyu" in mon.species_id
    )
    assert not busted.fainted
    assert busted.current_hp is not None
    assert start_mimic - busted.current_hp == max_mimic // 8


# --- flinch_and_fake_out ---------------------------------------------------------------


def test_fake_out_inner_focus_and_iron_head_flinch(worker) -> None:
    faker = pokeset(
        "Kangaskhan",
        ["fakeout", "ironhead", "protect", "splash"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
    )
    focused = _snorlax(
        moves=["tackle", "protect"], nature="Brave", atk=32, defense=2, ability="Inner Focus"
    )
    unfocused = pokeset(
        "Snorlax",
        ["tackle", "protect"],
        nature="Brave",
        sp={"hp": 32, "atk": 32, "def": 2},
        ability="Thick Fat",
    )

    with _root(worker, [faker, _partner()], [unfocused, _foe_partner()]) as root:
        [first] = _branches(
            root, _choice("fakeout"), _choice("tackle"), [[1, 2, 3, 4]], "fake1"
        )
        assert _has(first, "|cant|p2a: Snorlax|flinch")
        root.step({"p1": _choice("splash"), "p2": _choice("protect")})
        assert "fakeout" not in _available(root, "p1", 0)

    with _root(worker, [faker, _partner()], [focused, _foe_partner()]) as root:
        [immune] = _branches(
            root, _choice("fakeout"), _choice("tackle"), [[9, 10, 11, 12]], "innerfocus"
        )
    assert not _has(immune, "|cant|p2a: Snorlax|flinch")
    assert _has(immune, "|move|p2a: Snorlax|Tackle")

    with _root(worker, [faker, _partner()], [unfocused, _foe_partner()]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("protect")})
        # Iron Head 30% flinch; user is faster. n=64: P(no flinch)=(0.7)^64 ≈ 1.3e-10;
        # P(all flinch)=(0.3)^64 is smaller.
        heads = _branches(
            root, _choice("ironhead"), _choice("tackle"), _seeds(64, origin=1900), "ironhead"
        )
    flinches = sum(_has(b, "|cant|p2a: Snorlax|flinch") for b in heads)
    acted = sum(_has(b, "|move|p2a: Snorlax|Tackle") for b in heads)
    assert flinches > 0 and acted > 0


# --- move_restrictions -----------------------------------------------------------------


def test_taunt_encore_disable_choice_scarf_imprison_and_zero_pp(worker) -> None:
    assert "choicescarf" in load_items()

    taunter = pokeset(
        "Klefki",
        ["taunt", "protect"],
        nature="Bold",
        sp={"hp": 32, "spe": 32, "def": 2},
        ability="Prankster",
    )
    # No Prankster: Encore/Disable must resolve after the target has already moved.
    locker = pokeset(
        "Clefable",
        ["encore", "disable", "protect"],
        nature="Sassy",
        sp={"hp": 32, "spd": 32, "def": 2},
        ability="Magic Guard",
    )
    locked = _garchomp(moves=["tackle", "protect", "splash", "swordsdance"], nature="Jolly")
    scarfed = _garchomp(
        moves=["tackle", "protect", "splash"],
        nature="Jolly",
        item="Choice Scarf",
    )
    jailer = _garchomp(moves=["imprison", "tackle", "protect"], nature="Jolly")
    inmate = _snorlax(moves=["tackle", "protect", "splash"], nature="Brave", atk=32, defense=2)
    trickster = pokeset(
        "Klefki",
        ["trickroom", "protect"],
        nature="Bold",
        sp={"hp": 32, "spe": 32, "def": 2},
        ability="Prankster",
    )

    with _root(worker, [locked, _partner()], [taunter, _foe_partner()]) as root:
        [taunted] = _branches(
            root, _choice("splash"), _choice("taunt"), [[1, 2, 3, 4]], "taunt"
        )
    assert _has(taunted, "Taunt")
    assert _has(taunted, "|cant|") or _has(taunted, "[still]")

    with _root(worker, [locked, _partner()], [locker, _foe_partner()]) as root:
        root.step({"p1": _choice("tackle"), "p2": _choice("encore")})
        legal = _available(root, "p1", 0)
    assert "tackle" in legal
    assert "protect" not in legal
    assert "splash" not in legal

    with _root(worker, [locked, _partner()], [locker, _foe_partner()]) as root:
        root.step({"p1": _choice("tackle"), "p2": _choice("disable")})
        legal = _available(root, "p1", 0)
    assert "tackle" not in legal
    assert "protect" in legal

    with _root(worker, [scarfed, _partner()], [inmate, _foe_partner()]) as root:
        root.step({"p1": _choice("tackle"), "p2": _choice("protect")})
        scarf_moves = _available(root, "p1", 0)
    assert "tackle" in scarf_moves
    assert "protect" not in scarf_moves
    assert "splash" not in scarf_moves

    with _root(worker, [jailer, _partner()], [inmate, _foe_partner()]) as root:
        root.step({"p1": _choice("imprison"), "p2": _choice("protect")})
        with pytest.raises(InvalidChoice, match="Tackle is disabled"):
            _branches(
                root, _choice("protect"), _choice("tackle"), [[17, 18, 19, 20]], "imprison"
            )

    with _root(worker, [trickster, _partner()], [inmate, _foe_partner()]) as root:
        for _ in range(8):
            root.step({"p1": _choice("trickroom"), "p2": _choice("protect")})
        remaining = _available(root, "p1", 0)
    assert "trickroom" not in remaining
    assert "protect" in remaining


# --- charge_recharge_and_locks ---------------------------------------------------------


def test_solar_beam_hyper_beam_outrage_and_fly(worker) -> None:
    charger = pokeset(
        "Venusaur",
        ["solarbeam", "protect"],
        nature="Modest",
        sp={"hp": 32, "spa": 32, "spe": 2},
        ability="Overgrow",
    )
    sun_partner = pokeset(
        "Torkoal",
        ["protect", "splash"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Drought",
    )
    beamer = pokeset(
        "Registeel",
        ["hyperbeam", "protect"],
        nature="Quiet",
        sp={"hp": 32, "spa": 32, "spd": 2},
        ability="No Guard",
    )
    locker = pokeset(
        "Goomy",
        ["outrage", "protect"],
        nature="Modest",
        sp={"hp": 32, "spe": 32},
        ability="Sap Sipper",
    )
    flyer = _garchomp(moves=["fly", "protect"], nature="Jolly")
    wall = _snorlax(moves=["splash", "protect", "tackle", "thunder"], nature="Brave", atk=32, defense=2, spa=32)
    gust = pokeset(
        "Pelipper",
        ["gust", "protect"],
        nature="Modest",
        sp={"hp": 32, "spa": 32, "spe": 2},
        ability="Keen Eye",
    )

    with _root(worker, [charger, _partner()], [wall, _foe_partner()]) as root:
        [charging] = _branches(
            root, _choice("solarbeam"), _choice("splash"), [[1, 2, 3, 4]], "solar-charge"
        )
    assert _own(charging, "p1", "venusaur").preparing or _has(charging, "|-prepare|")
    assert _hp(charging, "p2", "snorlax") == _max_hp(charging, "p2", "snorlax")

    with _root(worker, [charger, sun_partner], [wall, _foe_partner()]) as root:
        [instant] = _branches(
            root, _choice("solarbeam"), _choice("splash"), [[5, 6, 7, 8]], "solar-sun"
        )
    # In sun the move fires the same turn (damage lands). Showdown still emits
    # `-prepare` + `-anim` in that log, and poke-env may leave `preparing` set;
    # the observable contract is the HP drop, not the flag.
    assert _hp(instant, "p2", "snorlax") < _max_hp(instant, "p2", "snorlax")
    assert _has(instant, "|-damage|p2a: Snorlax")

    with _root(worker, [beamer, _partner()], [wall, _foe_partner()]) as root:
        root.step({"p1": _choice("hyperbeam"), "p2": _choice("splash")})
        assert "recharge" in _available(root)
        [recharge] = _branches(
            root, _choice("recharge"), _choice("protect"), [[9, 10, 11, 12]], "recharge"
        )
    assert _has(recharge, "|cant|") and _has(recharge, "recharge")

    def _confused(battle) -> bool:
        if _volatile_time(battle, 0, "confusion") is not None:
            return True
        return any("confusion" in line for line in battle.last_lines["p1"])

    def _locked_outrage(battle) -> str:
        dumped = battle.worker.request({"cmd": "dump", "id": battle.battle_id})
        loc = dumped["state"]["sides"][0]["pokemon"][0].get("lastMoveTargetLoc") or 1
        return f"move outrage {int(loc)}, move protect"

    def _foe_splash(battle) -> str:
        n = sum(1 for slot in snapshot_battle(battle.battles["p2"]).available_moves if slot)
        if n <= 1:
            return "move splash"
        return _choice("splash", "splash")

    # Outrage locks 2-3 turns then confuses. n=24: P(all 2-turn or all 3-turn)
    # = 2 * 2^{-24} ≈ 1.2e-7. First use is randomNormal (no target). While
    # locked the request requires the previous target loc. Immune/protected
    # hits do not apply lockedmove, so both foes Splash and are Dragon-hittable.
    two_turn = 0
    three_turn = 0
    punchbag = pokeset(
        "Chansey",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Natural Cure",
    )
    for seed in _seeds(24, origin=2200):
        with _root(
            worker,
            [locker, _partner()],
            [
                pokeset(
                    "Blissey",
                    ["splash", "protect"],
                    nature="Bold",
                    sp={"hp": 32, "def": 32, "spd": 2},
                    ability="Natural Cure",
                ),
                punchbag,
            ],
            seed=seed,
        ) as battle:
            battle.step({"p1": _choice("outrage"), "p2": _foe_splash(battle)})
            locked = _locked_outrage(battle)
            battle.step({"p1": locked, "p2": _foe_splash(battle)})
            if _confused(battle):
                two_turn += 1
                continue
            battle.step({"p1": locked, "p2": _foe_splash(battle)})
            assert _confused(battle)
            three_turn += 1
    assert two_turn > 0 and three_turn > 0, (two_turn, three_turn)

    with _root(worker, [flyer, _partner()], [wall, _foe_partner()]) as root:
        [air] = _branches(
            root, _choice("fly"), _choice("tackle"), [[21, 22, 23, 24]], "fly-miss"
        )
    assert _own(air, "p1", "garchomp").preparing or _has(air, "|-prepare|")
    assert _has(air, "|-miss|") or _hp(air, "p1", "garchomp") == _max_hp(air, "p1", "garchomp")

    with _root(worker, [flyer, _partner()], [gust, _foe_partner()]) as root:
        [hit] = _branches(
            root, _choice("fly"), _choice("gust"), [[25, 26, 27, 28]], "fly-gust"
        )
    assert _hp(hit, "p1", "garchomp") < _max_hp(hit, "p1", "garchomp")
    assert not _has(hit, "|-miss|p2a")
