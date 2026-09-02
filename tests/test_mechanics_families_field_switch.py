"""Showdown-executed contracts for field, switching, and residual families.

Each test builds an isolating `DirectBattle` root, applies setup via real
moves/switches, then drives `evaluate_exact_branches` (or asserts on the root
when the property is the setup itself). Custom Game does not validate legality.
"""

from __future__ import annotations

import json

import pytest

from tests.mechanics_family_helpers import (
    _branches,
    _choice,
    _has,
    _lines,
    _own,
    _root,
    _seeds,
    _slot_before,
    _status_of,
)
from tests.sim_harness import pokeset
from vgc.damage import FieldState, PokemonState, damage_range
from vgc.mechanics_state import snapshot_battle
from vgc.rl.env import DirectBattle
from vgc.rl.mechanics_oracle import ExactMechanicsBranch

pytestmark = pytest.mark.integration


def _named(species: str, moves: list[str], *, name: str, **kwargs) -> dict:
    row = pokeset(species, moves, **kwargs)
    row["name"] = name
    return row


def _support() -> dict:
    return pokeset(
        "Blissey", ["splash", "protect"], nature="Calm", sp={"hp": 32, "spd": 32, "def": 2}
    )


def _foe_support() -> dict:
    return pokeset(
        "Chansey", ["splash", "protect"], nature="Calm", sp={"hp": 32, "spd": 32, "def": 2}
    )


def _boost(root: DirectBattle, side: str, species_id: str, stat: str) -> int:
    for pokemon in snapshot_battle(root.battles[side]).our_side.pokemon:
        if pokemon.species_id == species_id:
            return dict(pokemon.boosts)[stat]
    raise AssertionError(f"{species_id} missing on {side}")


def _mon(root: DirectBattle, side: str, species_id: str):
    for pokemon in snapshot_battle(root.battles[side]).our_side.pokemon:
        if pokemon.species_id == species_id:
            return pokemon
    raise AssertionError(f"{species_id} missing on {side}")


def _request(source: ExactMechanicsBranch | DirectBattle, side: str = "p1") -> dict:
    if isinstance(source, ExactMechanicsBranch):
        return json.loads(source.state_for(side).last_request_json or "{}")
    return source.battles[side].last_request or {}


def _from_tags(lines: tuple[str, ...] | list[str], slot: str) -> list[str]:
    tags: list[str] = []
    for line in lines:
        if f"|{slot}:" not in line and f"|{slot}|" not in line:
            continue
        if "[from]" not in line:
            continue
        if "|-damage|" not in line and "|-heal|" not in line:
            continue
        tags.append(line.split("[from]", 1)[1].split("|")[0].strip())
    return tags


def _first_index(lines: tuple[str, ...] | list[str], token: str) -> int:
    for index, line in enumerate(lines):
        if token in line:
            return index
    raise AssertionError(f"{token!r} missing in {lines[:12]}")


# --- stat_stage_changes ----------------------------------------------------------------


def test_stat_stages_boost_damage_cap_and_ability_reactions(worker) -> None:
    attacker = pokeset(
        "Garchomp",
        ["swordsdance", "tackle", "protect"],
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
    unboosted = damage_range(
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
    boosted = damage_range(
        PokemonState(
            "garchomp",
            sp_spread={"hp": 30, "atk": 32, "spe": 4},
            nature="adamant",
            ability="innerfocus",
            boosts={"atk": 2},
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
    p1 = [attacker, _support()]
    p2 = [defender, _foe_support()]
    with _root(worker, p1, p2) as root:
        start_hp = _mon(root, "p2", "snorlax").current_hp
        assert start_hp is not None
        root.step({"p1": _choice("swordsdance", "splash"), "p2": _choice("splash")})
        assert _boost(root, "p1", "garchomp", "atk") == 2
        [branch] = _branches(
            root, _choice("tackle"), _choice("splash"), [[11, 12, 13, 14]], "sd-dmg"
        )
        dealt = start_hp - (_own(branch, "p2", "snorlax").current_hp or 0)
        assert boosted.min_damage <= dealt <= boosted.max_damage, dealt
        assert dealt > unboosted.max_damage

        for _ in range(2):
            root.step({"p1": _choice("swordsdance", "splash"), "p2": _choice("splash")})
        assert _boost(root, "p1", "garchomp", "atk") == 6
        [capped] = _branches(
            root, _choice("swordsdance", "splash"), _choice("splash"), [[21, 22, 23, 24]], "sd-cap"
        )
        assert _has(capped, "|-boost|p1a: Garchomp|atk|0")
        assert dict(_own(capped, "p1", "garchomp").boosts)["atk"] == 6

    incin = pokeset(
        "Incineroar",
        ["splash", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "spe": 2},
        ability="Intimidate",
    )
    with _root(
        worker,
        [
            pokeset(
                "Malamar",
                ["splash", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "def": 2},
                ability="Contrary",
            ),
            _support(),
        ],
        [incin, _foe_support()],
    ) as root:
        assert _boost(root, "p1", "malamar", "atk") == 1

    with _root(
        worker,
        [
            pokeset(
                "Metagross",
                ["splash", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "def": 2},
                ability="Clear Body",
            ),
            _support(),
        ],
        [incin, _foe_support()],
    ) as root:
        assert _boost(root, "p1", "metagross", "atk") == 0

    with _root(
        worker,
        [
            pokeset(
                "Bisharp",
                ["splash", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "def": 2},
                ability="Defiant",
            ),
            _support(),
        ],
        [incin, _foe_support()],
    ) as root:
        # Intimidate -1 then Defiant +2 nets +1. The +2 reaction is on the switch-in log.
        assert _boost(root, "p1", "bisharp", "atk") == 1
        assert any(
            "|-boost|" in line and "|atk|2" in line for line in root.last_lines["p1"]
        )

    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["haze", "protect", "splash"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _support(),
        ],
        [incin, _foe_support()],
    ) as root:
        assert _boost(root, "p1", "garchomp", "atk") == -1
        [hazed] = _branches(
            root, _choice("move haze", "splash"), _choice("splash"), [[31, 22, 23, 24]], "haze"
        )
        assert dict(_own(hazed, "p1", "garchomp").boosts)["atk"] == 0
        assert dict(_own(hazed, "p2", "incineroar").boosts)["atk"] == 0


# --- rooms_and_field_conditions ---------------------------------------------------------


def test_trick_room_gravity_magic_room_and_wonder_room(worker) -> None:
    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["trickroom", "protect", "splash"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _support(),
        ],
        [
            pokeset(
                "Snorlax",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
            _foe_support(),
        ],
    ) as root:
        root.step({"p1": _choice("trickroom", "splash"), "p2": _choice("splash")})
        ended = False
        for _ in range(4):
            result = root.step({"p1": _choice("splash"), "p2": _choice("splash")})
            ended = any("|-fieldend|move: Trick Room" in line for line in result.lines["p1"])
        assert ended
        [after] = _branches(
            root, _choice("splash"), _choice("splash"), [[51, 52, 53, 54]], "tr6"
        )
        assert not _has(after, "|-fieldend|move: Trick Room")

    p1 = [
        pokeset(
            "Torkoal",
            ["earthquake", "gravity", "protect", "splash"],
            nature="Quiet",
            sp={"hp": 32, "atk": 4, "spa": 30},
            ability="White Smoke",
        ),
        _support(),
    ]
    p2 = [
        pokeset(
            "Talonflame",
            ["splash", "protect", "fly", "ember"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Gale Wings",
        ),
        _foe_support(),
    ]
    with _root(worker, p1, p2) as root:
        [immune] = _branches(
            root, _choice("earthquake", "splash"), _choice("splash"), [[61, 42, 43, 44]], "eq-fly"
        )
        assert _has(immune, "|-immune|p2a: Talonflame")
        root.step({"p1": _choice("move gravity", "splash"), "p2": _choice("splash")})
        fly_moves = _request(root, "p2").get("active", [{}])[0].get("moves", [])
        fly = next(move for move in fly_moves if move.get("id") == "fly")
        assert fly["disabled"] is True
        [grounded] = _branches(
            root,
            _choice("earthquake", "protect"),
            _choice("ember", "protect"),
            [[71, 42, 43, 44]],
            "eq-grav",
        )
        assert _has(grounded, "|-damage|p2a: Talonflame")
        assert not _has(grounded, "|-immune|p2a: Talonflame")

    scarf = pokeset(
        "Snorlax",
        ["tackle", "protect", "splash"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
        item="Choice Scarf",
    )
    garchomp = pokeset(
        "Garchomp",
        ["tackle", "magicroom", "protect", "splash"],
        nature="Serious",
        sp={"hp": 32, "atk": 32, "def": 2},
        ability="Inner Focus",
    )
    with _root(worker, [scarf, _support()], [garchomp, _foe_support()]) as root:
        [scarfed] = _branches(
            root, _choice("tackle"), _choice("tackle"), [[81, 42, 43, 44]], "scarf"
        )
        assert _slot_before(scarfed, "p1a", "p2a")
        root.step({"p1": _choice("protect"), "p2": _choice("move magicroom", "splash")})
        [suppressed] = _branches(
            root, _choice("tackle"), _choice("tackle"), [[91, 42, 43, 44]], "scarf-mr"
        )
        assert _slot_before(suppressed, "p2a", "p1a")

    sitrus = pokeset(
        "Torkoal",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="White Smoke",
        item="Sitrus Berry",
    )
    fanger = pokeset(
        "Garchomp",
        ["superfang", "magicroom", "protect", "splash"],
        nature="Jolly",
        sp={"hp": 32, "spe": 32, "atk": 2},
        ability="Inner Focus",
    )
    with _root(worker, [sitrus, _support()], [fanger, _foe_support()]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("superfang")})
        [eaten] = _branches(
            root, _choice("splash"), _choice("superfang"), [[101, 42, 43, 44]], "sitrus"
        )
        assert _has(eaten, "item: Sitrus Berry")
    with _root(worker, [sitrus, _support()], [fanger, _foe_support()]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move magicroom", "splash")})
        root.step({"p1": _choice("splash"), "p2": _choice("superfang")})
        [blocked] = _branches(
            root, _choice("splash"), _choice("superfang"), [[111, 42, 43, 44]], "sitrus-mr"
        )
        assert not _has(blocked, "item: Sitrus Berry")

    blissey = pokeset(
        "Blissey",
        ["splash", "protect"],
        nature="Calm",
        sp={"hp": 32, "spd": 32, "def": 2},
        ability="Battle Armor",
    )
    gengar = pokeset(
        "Gengar",
        ["psychic", "wonderroom", "protect", "splash"],
        nature="Timid",
        sp={"hp": 32, "spe": 32, "spa": 2},
        ability="Inner Focus",
    )
    seed = [121, 42, 43, 44]
    with _root(worker, [blissey, _foe_support()], [gengar, _support()]) as root:
        start = _mon(root, "p1", "blissey").current_hp
        [plain] = _branches(
            root, _choice("splash"), _choice("psychic"), [seed], "wr-off"
        )
        plain_hp = _own(plain, "p1", "blissey").current_hp
    with _root(worker, [blissey, _foe_support()], [gengar, _support()]) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move wonderroom", "splash")})
        [swapped] = _branches(
            root, _choice("splash"), _choice("psychic"), [seed], "wr-on"
        )
        swapped_hp = _own(swapped, "p1", "blissey").current_hp
    assert start is not None and plain_hp is not None and swapped_hp is not None
    assert swapped_hp < plain_hp


# --- entry_hazards ---------------------------------------------------------------------


def test_entry_hazards_damage_layers_and_removal(worker) -> None:
    leads = [
        pokeset(
            "Blissey",
            ["splash", "protect", "rapidspin", "defog"],
            nature="Calm",
            sp={"hp": 32, "spd": 32, "def": 2},
        ),
        _foe_support(),
        pokeset(
            "Snorlax",
            ["splash", "protect"],
            nature="Bold",
            sp={"hp": 32, "def": 32, "spd": 2},
            ability="Inner Focus",
        ),
        pokeset(
            "Torkoal",
            ["splash", "protect"],
            nature="Bold",
            sp={"hp": 32, "def": 32, "spd": 2},
            ability="White Smoke",
        ),
        pokeset(
            "Machamp",
            ["splash", "protect"],
            nature="Adamant",
            sp={"hp": 32, "atk": 32, "def": 2},
        ),
        pokeset(
            "Talonflame",
            ["splash", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Gale Wings",
        ),
        pokeset(
            "Muk",
            ["splash", "protect"],
            nature="Adamant",
            sp={"hp": 32, "atk": 32, "def": 2},
        ),
    ]
    setter = [
        pokeset(
            "Garchomp",
            ["stealthrock", "spikes", "toxicspikes", "stickyweb", "splash"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
        ),
        _foe_support(),
    ]

    def _switch_hp(species: str, bench: str, hazard: str, prefix: str) -> tuple[int, int]:
        team = [leads[0], leads[1], next(mon for mon in leads if mon["species"] == species)]
        with _root(worker, team, setter, p1_order="123", p2_order="12") as root:
            root.step({"p1": _choice("splash"), "p2": _choice(f"move {hazard}", "splash")})
            [branch] = _branches(
                root,
                "switch 3, move splash",
                _choice("splash"),
                [[131, 42, 43, 44]],
                prefix,
            )
            mon = _own(branch, "p1", bench)
            assert mon.current_hp is not None and mon.max_hp is not None
            return mon.current_hp, mon.max_hp

    hp, max_hp = _switch_hp("Snorlax", "snorlax", "stealthrock", "sr-n")
    assert max_hp - hp == max_hp // 8
    hp, max_hp = _switch_hp("Torkoal", "torkoal", "stealthrock", "sr-2")
    assert max_hp - hp == max_hp // 4
    hp, max_hp = _switch_hp("Machamp", "machamp", "stealthrock", "sr-r")
    assert max_hp - hp == max_hp // 16

    team = [leads[0], leads[1], leads[2]]
    with _root(worker, team, setter, p1_order="123", p2_order="12") as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move spikes", "splash")})
        [one] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[141, 42, 43, 44]], "spk1"
        )
        lost = (_own(one, "p1", "snorlax").max_hp or 0) - (_own(one, "p1", "snorlax").current_hp or 0)
        assert lost == (_own(one, "p1", "snorlax").max_hp or 0) // 8
    with _root(worker, team, setter, p1_order="123", p2_order="12") as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move spikes", "splash")})
        root.step({"p1": _choice("splash"), "p2": _choice("move spikes", "splash")})
        [two] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[151, 42, 43, 44]], "spk2"
        )
        lost = (_own(two, "p1", "snorlax").max_hp or 0) - (_own(two, "p1", "snorlax").current_hp or 0)
        assert lost == (_own(two, "p1", "snorlax").max_hp or 0) // 6
    with _root(worker, team, setter, p1_order="123", p2_order="12") as root:
        for _ in range(3):
            root.step({"p1": _choice("splash"), "p2": _choice("move spikes", "splash")})
        [three] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[161, 42, 43, 44]], "spk3"
        )
        lost = (_own(three, "p1", "snorlax").max_hp or 0) - (
            _own(three, "p1", "snorlax").current_hp or 0
        )
        assert lost == (_own(three, "p1", "snorlax").max_hp or 0) // 4

    with _root(
        worker,
        [leads[0], leads[1], leads[2]],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move stickyweb", "splash")})
        [web] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[171, 42, 43, 44]], "web-g"
        )
        assert dict(_own(web, "p1", "snorlax").boosts)["spe"] == -1
    with _root(
        worker,
        [leads[0], leads[1], leads[5]],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move stickyweb", "splash")})
        [fly] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[181, 42, 43, 44]], "web-f"
        )
        assert dict(_own(fly, "p1", "talonflame").boosts)["spe"] == 0
    with _root(
        worker,
        [
            leads[0],
            leads[1],
            pokeset(
                "Bronzong",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
                ability="Levitate",
            ),
        ],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move stickyweb", "splash")})
        [lev] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[191, 42, 43, 44]], "web-l"
        )
        assert dict(_own(lev, "p1", "bronzong").boosts)["spe"] == 0

    with _root(
        worker,
        [leads[0], leads[1], leads[2]],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move toxicspikes", "splash")})
        assert any("Toxic Spikes" in line for line in root.last_lines["p1"])
        [tspikes] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[201, 42, 43, 44]], "tsp"
        )
        assert _own(tspikes, "p1", "snorlax").status == "psn"
    with _root(
        worker,
        [leads[0], leads[1], leads[6]],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move toxicspikes", "splash")})
        [absorb] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[211, 42, 43, 44]], "tsp-abs"
        )
        assert _own(absorb, "p1", "muk").status is None
        assert not any(
            effect.id == "toxicspikes" for effect in absorb.state_for("p1").our_side.side_conditions
        )

    with _root(
        worker,
        [leads[0], leads[1], leads[2]],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move stealthrock", "splash")})
        root.step({"p1": _choice("rapidspin"), "p2": _choice("splash")})
        [cleared] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[221, 42, 43, 44]], "spin"
        )
        snorlax = _own(cleared, "p1", "snorlax")
        assert snorlax.current_hp == snorlax.max_hp
    with _root(
        worker,
        [leads[0], leads[1], leads[2]],
        setter,
        p1_order="123",
        p2_order="12",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("move stealthrock", "splash")})
        root.step({"p1": _choice("defog"), "p2": _choice("splash")})
        [defogged] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[231, 42, 43, 44]], "defog"
        )
        snorlax = _own(defogged, "p1", "snorlax")
        assert snorlax.current_hp == snorlax.max_hp


# --- switching_trapping_and_phazing ---------------------------------------------------


def test_switching_happens_before_moves_trapping_pivots_and_phazing(worker) -> None:
    bench = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
    )
    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _support(),
            bench,
        ],
        [
            pokeset(
                "Incineroar",
                ["tackle", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
        p1_order="123",
        p2_order="12",
    ) as root:
        [branch] = _branches(
            root,
            "switch 3, move splash",
            _choice("tackle"),
            [[241, 42, 43, 44]],
            "sw-pre",
        )
        assert _first_index(_lines(branch), "|switch|") < _first_index(_lines(branch), "|move|")

    trapped_foe = [
        pokeset(
            "Wobbuffet",
            ["splash", "protect"],
            nature="Calm",
            sp={"hp": 32, "spd": 32, "def": 2},
            ability="Shadow Tag",
        ),
        _foe_support(),
        pokeset("Klefki", ["protect"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}),
    ]
    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _support(),
            bench,
        ],
        trapped_foe,
        p1_order="123",
        p2_order="123",
    ) as root:
        assert snapshot_battle(root.battles["p1"]).our_side.trapped[0] is True
        assert _request(root)["active"][0].get("trapped") is True
        assert snapshot_battle(root.battles["p1"]).available_switches[0] == ()
    with _root(
        worker,
        [
            pokeset(
                "Gengar",
                ["splash", "protect"],
                nature="Timid",
                sp={"hp": 32, "spe": 32, "spa": 2},
                ability="Levitate",
            ),
            _support(),
            bench,
        ],
        trapped_foe,
        p1_order="123",
        p2_order="123",
    ) as root:
        assert snapshot_battle(root.battles["p1"]).our_side.trapped[0] is not True
        assert "snorlax" in snapshot_battle(root.battles["p1"]).available_switches[0]

    arena = [
        pokeset(
            "Dugtrio",
            ["splash", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
            ability="Arena Trap",
        ),
        _foe_support(),
        pokeset("Klefki", ["protect"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2}),
    ]
    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _support(),
            bench,
        ],
        arena,
        p1_order="123",
        p2_order="123",
    ) as root:
        assert _request(root)["active"][0].get("trapped") is True
    with _root(
        worker,
        [
            pokeset(
                "Talonflame",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Gale Wings",
            ),
            _support(),
            bench,
        ],
        arena,
        p1_order="123",
        p2_order="123",
    ) as root:
        assert _request(root)["active"][0].get("trapped") is not True

    with _root(
        worker,
        [_support(), _foe_support(), bench],
        [
            pokeset(
                "Garchomp",
                ["uturn", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
            pokeset(
                "Snorlax",
                ["splash"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
        ],
        p1_order="123",
        p2_order="123",
    ) as root:
        [pivot] = _branches(
            root, _choice("splash"), _choice("uturn"), [[251, 42, 43, 44]], "uturn"
        )
        assert _has(pivot, "|-damage|")
        assert pivot.request_state == "switch"
        assert _request(pivot, "p2").get("forceSwitch") == [True, False]
        assert _first_index(_lines(pivot), "|-damage|") < _first_index(_lines(pivot), "|request|")

    roar_team = [
        pokeset(
            "Garchomp",
            ["splash", "protect"],
            nature="Jolly",
            sp={"hp": 32, "spe": 32, "atk": 2},
        ),
        _support(),
        pokeset(
            "Snorlax",
            ["splash", "protect"],
            nature="Bold",
            sp={"hp": 32, "def": 32, "spd": 2},
        ),
        pokeset(
            "Torkoal",
            ["splash", "protect"],
            nature="Bold",
            sp={"hp": 32, "def": 32, "spd": 2},
            ability="White Smoke",
        ),
    ]
    roar_user = [
        pokeset(
            "Incineroar",
            ["roar", "protect"],
            nature="Adamant",
            sp={"hp": 32, "atk": 32, "spe": 2},
        ),
        _foe_support(),
    ]
    with _root(worker, roar_team, roar_user, p1_order="1234") as root:
        # n=16, two bench options. P(all Roar drags identical) = 2 * 2^{-16} ≈ 3.1e-5.
        branches = _branches(
            root, _choice("splash"), _choice("roar"), _seeds(16, origin=300), "roar"
        )
        dragged = set()
        for branch in branches:
            assert _has(branch, "|drag|p1a:")
            for line in _lines(branch):
                if line.startswith("|drag|p1a:"):
                    dragged.add(line.split("|")[2].split(":")[1].strip())
        assert len(dragged) >= 2, dragged

    with _root(
        worker,
        [
            pokeset(
                "Snorlax",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
                ability="Battle Armor",
                item="Eject Button",
            ),
            _support(),
            bench,
        ],
        [
            pokeset(
                "Garchomp",
                ["tackle", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
        p1_order="123",
    ) as root:
        [eject] = _branches(
            root, _choice("splash"), _choice("tackle"), [[261, 42, 43, 44]], "eject"
        )
        assert _has(eject, "|-enditem|p1a: Snorlax|Eject Button")
        assert _request(eject).get("forceSwitch") == [True, False]

    with _root(
        worker,
        [
            pokeset(
                "Snorlax",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
                ability="Battle Armor",
                item="Red Card",
            ),
            _support(),
        ],
        [
            pokeset(
                "Garchomp",
                ["tackle", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
            pokeset(
                "Torkoal",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
            pokeset(
                "Machamp",
                ["splash", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "def": 2},
            ),
        ],
        p1_order="12",
        p2_order="1234",
    ) as root:
        [card] = _branches(
            root, _choice("splash"), _choice("tackle"), [[271, 42, 43, 44]], "redcard"
        )
        assert _has(card, "|-enditem|p1a: Snorlax|Red Card")
        assert _has(card, "|drag|p2a:")


# --- form_transform_and_illusion ------------------------------------------------------


def test_illusion_transform_mega_and_disguise(worker) -> None:
    with _root(
        worker,
        [
            pokeset(
                "Zoroark",
                ["splash", "protect"],
                nature="Timid",
                sp={"hp": 32, "spe": 32, "spa": 2},
                ability="Illusion",
            ),
            _support(),
            pokeset(
                "Snorlax",
                ["splash"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
            pokeset(
                "Chansey",
                ["splash", "protect"],
                nature="Calm",
                sp={"hp": 32, "spd": 32, "def": 2},
            ),
        ],
        [
            pokeset(
                "Garchomp",
                ["tackle", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
        p1_order="1234",
    ) as root:
        seen = [mon.species for mon in root.battles["p2"].opponent_active_pokemon if mon]
        assert seen[0] == "chansey"
        [broke] = _branches(
            root, _choice("splash"), _choice("tackle"), [[281, 42, 43, 44]], "illu"
        )
        assert _has(broke, "|replace|", "p2") or _has(broke, "|replace|")
        assert _own(broke, "p1", "zoroark").species_id == "zoroark"

    with _root(
        worker,
        [
            pokeset(
                "Ditto",
                ["transform", "protect"],
                nature="Serious",
                sp={"hp": 32, "def": 32, "spe": 2},
            ),
            _support(),
        ],
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
    ) as root:
        [copied] = _branches(
            root, _choice("transform"), _choice("splash"), [[291, 42, 43, 44]], "tf"
        )
        assert _has(copied, "|-transform|p1a: Ditto|p2a: Garchomp")
        copied_mon = next(
            mon
            for mon in copied.state_for("p1").our_side.pokemon
            if mon.transformed or mon.species_id in {"ditto", "garchomp"}
        )
        assert copied_mon.transformed is True
        assert {move.id for move in copied_mon.moves} >= {"splash", "protect"}

    with _root(
        worker,
        [
            pokeset(
                "Charizard",
                ["flamethrower", "protect"],
                nature="Modest",
                sp={"hp": 32, "spa": 32, "spe": 2},
                ability="Blaze",
                item="Charizardite Y",
            ),
            _support(),
            pokeset(
                "Snorlax",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
        ],
        [
            _foe_support(),
            _support(),
            pokeset(
                "Machamp",
                ["splash", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "def": 2},
            ),
        ],
        p1_order="123",
        p2_order="123",
    ) as root:
        [mega] = _branches(
            root,
            "move flamethrower mega 1, move splash",
            _choice("splash"),
            [[301, 42, 43, 44]],
            "mega",
        )
        assert _first_index(_lines(mega), "|-mega|") < _first_index(_lines(mega), "|move|p1a:")
        root.step({"p1": "move flamethrower mega 1, move splash", "p2": _choice("splash")})
        root.step({"p1": "switch 3, move splash", "p2": _choice("splash")})
        [back] = _branches(
            root, "switch 3, move splash", _choice("splash"), [[311, 42, 43, 44]], "mega-in"
        )
        assert _has(back, "Charizard-Mega-Y")

    with _root(
        worker,
        [
            pokeset(
                "Mimikyu",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Disguise",
            ),
            _support(),
        ],
        [
            pokeset(
                "Garchomp",
                ["crunch", "protect", "splash"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
    ) as root:
        [bust] = _branches(
            root, _choice("splash"), _choice("crunch"), [[321, 42, 43, 44]], "disguise"
        )
        assert _has(bust, "ability: Disguise")
        busted = _own(bust, "p1", "mimikyubusted")
        assert busted.species_id == "mimikyubusted"


# --- move_flags_and_interactions -------------------------------------------------------------


def test_sound_bullet_powder_substitute_bounce_and_prankster(worker) -> None:
    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Soundproof",
            ),
            _support(),
        ],
        [
            pokeset(
                "Exploud",
                ["hypervoice", "protect"],
                nature="Modest",
                sp={"hp": 32, "spa": 32, "spe": 2},
            ),
            _foe_support(),
        ],
    ) as root:
        [blocked] = _branches(
            root, _choice("splash"), _choice("move hypervoice", "splash"), [[331, 2, 3, 4]], "sound"
        )
        assert _has(blocked, "|-immune|p1a: Garchomp|[from] ability: Soundproof")

    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Bulletproof",
            ),
            _support(),
        ],
        [
            pokeset(
                "Venusaur",
                ["energyball", "protect"],
                nature="Modest",
                sp={"hp": 32, "spa": 32, "spe": 2},
            ),
            _foe_support(),
        ],
    ) as root:
        [bullet] = _branches(
            root, _choice("splash"), _choice("energyball"), [[341, 2, 3, 4]], "bullet"
        )
        assert _has(bullet, "|-immune|p1a: Garchomp|[from] ability: Bulletproof")

    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Overcoat",
            ),
            _support(),
        ],
        [
            pokeset(
                "Breloom",
                ["spore", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _foe_support(),
        ],
    ) as root:
        [coat] = _branches(
            root, _choice("splash"), _choice("spore"), [[351, 2, 3, 4]], "coat"
        )
        assert _has(coat, "|-immune|p1a: Garchomp|[from] ability: Overcoat")
        assert _status_of(root, "p1", "garchomp") is None

    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                item="Safety Goggles",
            ),
            _support(),
        ],
        [
            pokeset(
                "Breloom",
                ["spore", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
            ),
            _foe_support(),
        ],
    ) as root:
        [goggles] = _branches(
            root, _choice("splash"), _choice("spore"), [[361, 2, 3, 4]], "goggles"
        )
        assert _has(goggles, "item: Safety Goggles")
        assert _own(goggles, "p1", "garchomp").status is None

    sub_user = pokeset(
        "Blissey",
        ["substitute", "splash", "protect"],
        nature="Calm",
        sp={"hp": 32, "spd": 32, "def": 2},
    )
    with _root(
        worker,
        [sub_user, _foe_support()],
        [
            pokeset(
                "Garchomp",
                ["tackle", "protect", "splash"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            pokeset(
                "Exploud",
                ["hypervoice", "protect", "splash"],
                nature="Modest",
                sp={"hp": 32, "spa": 32, "spe": 2},
            ),
        ],
    ) as root:
        root.step({"p1": _choice("move substitute", "splash"), "p2": _choice("splash")})
        start = _mon(root, "p1", "blissey").current_hp
        [blocked_hit] = _branches(
            root, _choice("splash"), _choice("tackle"), [[371, 2, 3, 4]], "sub-t"
        )
        assert _has(blocked_hit, "|-activate|p1a: Blissey|Substitute") or _has(
            blocked_hit, "Substitute"
        )
        assert _own(blocked_hit, "p1", "blissey").current_hp == start
    with _root(
        worker,
        [sub_user, _foe_support()],
        [
            pokeset(
                "Garchomp",
                ["tackle", "protect", "splash"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            pokeset(
                "Exploud",
                ["hypervoice", "protect", "splash"],
                nature="Modest",
                sp={"hp": 32, "spa": 32, "spe": 2},
            ),
        ],
    ) as root:
        root.step({"p1": _choice("move substitute", "splash"), "p2": _choice("splash")})
        start = _mon(root, "p1", "blissey").current_hp
        [sound] = _branches(
            root,
            _choice("splash"),
            "move splash, move hypervoice",
            [[381, 2, 3, 4]],
            "sub-s",
        )
        assert (_own(sound, "p1", "blissey").current_hp or 0) < (start or 0)

    with _root(
        worker,
        [
            pokeset(
                "Espeon",
                ["splash", "protect"],
                nature="Timid",
                sp={"hp": 32, "spe": 32, "spa": 2},
                ability="Magic Bounce",
            ),
            _support(),
        ],
        [
            pokeset(
                "Klefki",
                ["thunderwave", "protect"],
                nature="Bold",
                sp={"hp": 32, "spe": 32, "def": 2},
                ability="Prankster",
            ),
            _foe_support(),
        ],
    ) as root:
        [bounce] = _branches(
            root, _choice("splash"), _choice("thunderwave"), [[391, 2, 3, 4]], "bounce"
        )
        assert _has(bounce, "[from] ability: Magic Bounce")
        assert _own(bounce, "p1", "espeon").status is None
        assert _own(bounce, "p2", "klefki").status == "par"

    with _root(
        worker,
        [
            pokeset(
                "Umbreon",
                ["splash", "protect"],
                nature="Calm",
                sp={"hp": 32, "spd": 32, "def": 2},
            ),
            _support(),
        ],
        [
            pokeset(
                "Klefki",
                ["thunderwave", "protect"],
                nature="Bold",
                sp={"hp": 32, "spe": 32, "def": 2},
                ability="Prankster",
            ),
            _foe_support(),
        ],
    ) as root:
        [dark] = _branches(
            root, _choice("splash"), _choice("thunderwave"), [[401, 2, 3, 4]], "prank"
        )
        assert _has(dark, "|-immune|p1a: Umbreon")
        assert _own(dark, "p1", "umbreon").status is None


# --- residual_and_end_of_turn_order -----------------------------------------------------


def test_residual_order_speed_and_weather_duration(worker) -> None:
    # Engine order, not the catalog's "weather then Leftovers before poison" guess:
    # Sandstorm damage, Leftovers, Leech Seed, then burn. Poison shares burn's slot.
    with _root(
        worker,
        [
            pokeset(
                "Snorlax",
                ["splash", "protect"],
                nature="Bold",
                sp={"hp": 32, "def": 32, "spd": 2},
                ability="Inner Focus",
                item="Leftovers",
            ),
            _support(),
        ],
        [
            pokeset(
                "Hippowdon",
                ["willowisp", "leechseed", "protect"],
                nature="Impish",
                sp={"hp": 32, "def": 32, "spd": 2},
                ability="Sand Stream",
            ),
            _foe_support(),
        ],
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("willowisp", "splash")})
        assert _status_of(root, "p1", "snorlax") == "brn"
        [branch] = _branches(
            root, _choice("splash"), _choice("leechseed"), [[411, 2, 3, 4]], "resid"
        )
        assert _from_tags(_lines(branch), "p1a") == [
            "Sandstorm",
            "item: Leftovers",
            "Leech Seed",
            "brn",
        ]

    with _root(
        worker,
        [
            pokeset(
                "Garchomp",
                ["splash", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Inner Focus",
                item="Leftovers",
            ),
            pokeset(
                "Snorlax",
                ["splash", "protect"],
                nature="Brave",
                sp={"hp": 32, "atk": 32, "def": 2},
                ability="Inner Focus",
                item="Leftovers",
            ),
        ],
        [
            pokeset(
                "Torkoal",
                ["earthquake", "protect"],
                nature="Quiet",
                sp={"hp": 32, "atk": 32, "spa": 2},
                ability="White Smoke",
            ),
            pokeset(
                "Talonflame",
                ["protect", "splash"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Gale Wings",
            ),
        ],
    ) as root:
        [heals] = _branches(
            root, _choice("splash", "splash"), _choice("earthquake", "protect"), [[421, 2, 3, 4]], "spd"
        )
        leftover_slots = [
            line.split("|")[2].split(":")[0]
            for line in _lines(heals)
            if "|-heal|" in line and "Leftovers" in line
        ]
        assert leftover_slots == ["p1a", "p1b"]

    sand = pokeset(
        "Tyranitar",
        ["splash", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "def": 2},
        ability="Sand Stream",
    )
    with _root(worker, [sand, _support()], [_foe_support(), _support()]) as root:
        for _ in range(4):
            root.step({"p1": _choice("splash"), "p2": _choice("splash")})
        [fifth] = _branches(
            root, _choice("splash"), _choice("splash"), [[431, 2, 3, 4]], "sand5"
        )
        assert _has(fifth, "|-weather|none")

    rocky = pokeset(
        "Tyranitar",
        ["splash", "protect"],
        nature="Adamant",
        sp={"hp": 32, "atk": 32, "def": 2},
        ability="Sand Stream",
        item="Smooth Rock",
    )
    with _root(worker, [rocky, _support()], [_foe_support(), _support()]) as root:
        for _ in range(4):
            root.step({"p1": _choice("splash"), "p2": _choice("splash")})
        [still] = _branches(
            root, _choice("splash"), _choice("splash"), [[441, 2, 3, 4]], "sand5r"
        )
        assert _has(still, "|-weather|Sandstorm|[upkeep]")
        for _ in range(3):
            root.step({"p1": _choice("splash"), "p2": _choice("splash")})
        [eighth] = _branches(
            root, _choice("splash"), _choice("splash"), [[451, 2, 3, 4]], "sand8"
        )
        assert _has(eighth, "|-weather|none")


# --- faint_and_replacement_order ---------------------------------------------------------


def test_faint_replacement_queued_move_destiny_bond_and_both_sides(worker) -> None:
    fish_a = _named("Magikarp", ["splash", "destinybond"], name="FishA", nature="Hardy")
    fish_b = _named("Feebas", ["splash", "protect"], name="FishB", nature="Hardy")
    with _root(
        worker,
        [fish_a, fish_b, _support(), _foe_support()],
        [
            pokeset(
                "Garchomp",
                ["earthquake", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            pokeset(
                "Talonflame",
                ["protect", "splash"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Gale Wings",
            ),
        ],
        p1_order="1234",
    ) as root:
        [both] = _branches(
            root, _choice("splash", "splash"), _choice("earthquake", "protect"), [[461, 2, 3, 4]], "dfaint"
        )
        assert _has(both, "|faint|p1a:")
        assert _has(both, "|faint|p1b:")
        assert _request(both).get("forceSwitch") == [True, True]
        assert _request(both, "p2").get("wait") is True

    with _root(
        worker,
        [fish_a, fish_b, _support()],
        [
            pokeset(
                "Garchomp",
                ["dragonclaw", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
        p1_order="123",
    ) as root:
        [queued] = _branches(
            root, _choice("splash", "splash"), _choice("dragonclaw"), [[471, 2, 3, 4]], "qmove"
        )
        faint_at = _first_index(_lines(queued), "|faint|p1a:")
        assert any(
            i > faint_at and "|move|p1b:" in line
            for i, line in enumerate(_lines(queued))
        )

    shedinja = pokeset(
        "Shedinja",
        ["splash", "protect"],
        nature="Hardy",
        item="Toxic Orb",
        ability="Wonder Guard",
    )
    with _root(
        worker,
        [shedinja, _support(), _foe_support()],
        [_foe_support(), _support()],
        p1_order="123",
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("splash")})
        assert _status_of(root, "p1", "shedinja") in {"tox", "psn"}
        [resid] = _branches(
            root, _choice("splash"), _choice("splash"), [[481, 2, 3, 4]], "psn-fnt"
        )
        assert _has(resid, "|faint|p1a:")
        assert resid.request_state == "switch"
        assert _request(resid).get("forceSwitch") == [True, False]

    with _root(
        worker,
        [
            _named("Magikarp", ["destinybond", "splash"], name="FishA", nature="Jolly", sp={"spe": 32}),
            _support(),
            _foe_support(),
        ],
        [
            pokeset(
                "Snorlax",
                ["doubleedge", "protect", "splash"],
                nature="Brave",
                sp={"hp": 32, "atk": 32, "def": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
        p1_order="123",
    ) as root:
        [bond] = _branches(
            root,
            _choice("move destinybond", "splash"),
            _choice("doubleedge"),
            [[491, 2, 3, 4]],
            "dbond",
        )
        assert _has(bond, "|faint|p1a:")
        assert _has(bond, "|faint|p2a:")

    with _root(
        worker,
        [
            fish_a,
            pokeset(
                "Garchomp",
                ["crunch", "protect"],
                nature="Jolly",
                sp={"hp": 32, "spe": 32, "atk": 2},
                ability="Inner Focus",
            ),
            _support(),
        ],
        [
            fish_b,
            pokeset(
                "Incineroar",
                ["darkestlariat", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "spe": 2},
                ability="Inner Focus",
            ),
            _foe_support(),
        ],
        p1_order="123",
        p2_order="123",
    ) as root:
        [both_sides] = _branches(
            root,
            "move splash, move crunch 1",
            "move splash, move darkestlariat 1",
            [[501, 2, 3, 4]],
            "bothfs",
        )
        assert _request(both_sides, "p1").get("forceSwitch") == [True, False]
        assert _request(both_sides, "p2").get("forceSwitch") == [True, False]
        assert both_sides.request_state == "switch"
        assert _request(both_sides, "p1").get("wait") is None
        assert _request(both_sides, "p2").get("wait") is None


# --- poison_and_toxic ------------------------------------------------------------------


def test_poison_toxic_immunities_heal_and_corrosion(worker) -> None:
    snorlax = pokeset(
        "Snorlax",
        ["splash", "protect"],
        nature="Bold",
        sp={"hp": 32, "def": 32, "spd": 2},
        ability="Inner Focus",
    )
    gengar = pokeset(
        "Gengar",
        ["toxic", "poisonpowder", "protect", "splash"],
        nature="Timid",
        sp={"hp": 32, "spe": 32, "spa": 2},
        ability="Inner Focus",
    )
    with _root(worker, [snorlax, _support()], [gengar, _foe_support()]) as root:
        max_hp = _mon(root, "p1", "snorlax").max_hp
        assert max_hp is not None
        root.step({"p1": _choice("splash"), "p2": _choice("poisonpowder")})
        assert _status_of(root, "p1", "snorlax") == "psn"
        after_apply = _mon(root, "p1", "snorlax").current_hp
        assert after_apply == max_hp - max_hp // 8
        [tick] = _branches(
            root, _choice("splash"), _choice("splash"), [[511, 2, 3, 4]], "psn"
        )
        assert (_own(tick, "p1", "snorlax").current_hp or 0) == after_apply - max_hp // 8

    with _root(worker, [snorlax, _support()], [gengar, _foe_support()]) as root:
        max_hp = _mon(root, "p1", "snorlax").max_hp
        assert max_hp is not None
        root.step({"p1": _choice("splash"), "p2": _choice("toxic")})
        assert _status_of(root, "p1", "snorlax") == "tox"
        hp = _mon(root, "p1", "snorlax").current_hp
        assert hp == max_hp - max_hp // 16
        for stage in (2, 3):
            result = root.step({"p1": _choice("splash"), "p2": _choice("splash")})
            lost = _from_tags(result.lines["p1"], "p1a")
            assert "psn" in lost
            now = _mon(root, "p1", "snorlax").current_hp
            assert hp is not None and now is not None
            assert hp - now == (max_hp // 16) * stage
            hp = now

    with _root(
        worker,
        [snorlax, _support(), _foe_support()],
        [gengar, _foe_support()],
        p1_order="123",
    ) as root:
        max_hp = _mon(root, "p1", "snorlax").max_hp
        assert max_hp is not None
        root.step({"p1": _choice("splash"), "p2": _choice("toxic")})
        root.step({"p1": _choice("splash"), "p2": _choice("splash")})
        root.step({"p1": "switch 3, move splash", "p2": _choice("splash")})
        benched = None
        for index, mon in enumerate(_request(root)["side"]["pokemon"], start=1):
            if "Snorlax" in mon["ident"] and not mon["active"]:
                benched = index
                break
        assert benched is not None
        [reset] = _branches(
            root,
            f"switch {benched}, move splash",
            _choice("splash"),
            [[521, 2, 3, 4]],
            "tox-rst",
        )
        snorlax_mon = _own(reset, "p1", "snorlax")
        assert snorlax_mon.status == "tox"
        switched_hp = None
        for line in _lines(reset):
            if "|switch|p1a: Snorlax|" in line:
                switched_hp = int(line.split("|")[4].split("/")[0])
                break
        assert switched_hp is not None
        assert switched_hp - (snorlax_mon.current_hp or 0) == (snorlax_mon.max_hp or 0) // 16

    with _root(
        worker,
        [
            pokeset(
                "Skarmory",
                ["splash", "protect"],
                nature="Impish",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
            _support(),
        ],
        [gengar, _foe_support()],
    ) as root:
        [steel] = _branches(
            root, _choice("splash"), _choice("toxic"), [[531, 2, 3, 4]], "steel"
        )
        assert _own(steel, "p1", "skarmory").status is None
        assert _has(steel, "|-immune|p1a: Skarmory") or _has(steel, "|-fail|")

    with _root(
        worker,
        [
            pokeset(
                "Muk",
                ["splash", "protect"],
                nature="Adamant",
                sp={"hp": 32, "atk": 32, "def": 2},
            ),
            _support(),
        ],
        [gengar, _foe_support()],
    ) as root:
        [poison_type] = _branches(
            root, _choice("splash"), _choice("toxic"), [[541, 2, 3, 4]], "ptype"
        )
        assert _own(poison_type, "p1", "muk").status is None

    with _root(
        worker,
        [
            pokeset(
                "Gliscor",
                ["splash", "protect"],
                nature="Impish",
                sp={"hp": 32, "def": 32, "spd": 2},
                ability="Poison Heal",
            ),
            _support(),
        ],
        [gengar, _foe_support()],
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("toxic")})
        before = _mon(root, "p1", "gliscor").current_hp
        max_hp = _mon(root, "p1", "gliscor").max_hp
        [heal] = _branches(
            root, _choice("splash"), _choice("splash"), [[551, 2, 3, 4]], "pheal"
        )
        assert _has(heal, "ability: Poison Heal") or (
            (_own(heal, "p1", "gliscor").current_hp or 0) >= (before or 0)
        )
        assert (_own(heal, "p1", "gliscor").current_hp or 0) == min(
            max_hp or 0, (before or 0) + (max_hp or 0) // 8
        ) or _has(heal, "|-heal|")

    with _root(
        worker,
        [
            pokeset(
                "Clefable",
                ["splash", "protect"],
                nature="Calm",
                sp={"hp": 32, "spd": 32, "def": 2},
                ability="Magic Guard",
            ),
            _support(),
        ],
        [gengar, _foe_support()],
    ) as root:
        root.step({"p1": _choice("splash"), "p2": _choice("toxic")})
        before = _mon(root, "p1", "clefable").current_hp
        [guard] = _branches(
            root, _choice("splash"), _choice("splash"), [[561, 2, 3, 4]], "mguard"
        )
        assert _own(guard, "p1", "clefable").current_hp == before
        assert "psn" not in _from_tags(_lines(guard), "p1a")

    with _root(
        worker,
        [
            pokeset(
                "Skarmory",
                ["splash", "protect"],
                nature="Impish",
                sp={"hp": 32, "def": 32, "spd": 2},
            ),
            _support(),
        ],
        [
            pokeset(
                "Salazzle",
                ["toxic", "protect"],
                nature="Timid",
                sp={"hp": 32, "spe": 32, "spa": 2},
                ability="Corrosion",
            ),
            _foe_support(),
        ],
    ) as root:
        [corroded] = _branches(
            root, _choice("splash"), _choice("toxic"), [[571, 2, 3, 4]], "corr"
        )
        assert _own(corroded, "p1", "skarmory").status == "tox"
