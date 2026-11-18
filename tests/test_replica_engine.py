"""Check the training bridge against both real player views from pinned Showdown."""

from pathlib import Path
import random

import pytest

from vgc.actions import enumerate_joint_orders
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SIDES, DirectBattle, SimWorker, choice_string
from vgc.rl.replica_engine import ReplicaRustWorker


@pytest.fixture(autouse=True)
def require_local_engines():
    pytest.importorskip("pikapro")
    if not (DEFAULT_SHOWDOWN_REPO / "dist/sim/index.js").exists():
        pytest.skip("requires the built local Showdown engine")


@pytest.fixture
def engine_teams():
    folder = Path(__file__).resolve().parents[1] / "teams"
    return [(folder / f"mc_ladder_0{i}.packed.txt").read_text().strip() for i in (1, 2)]


def test_waiting_player_receives_own_team_snapshot(engine_teams):
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as official, ReplicaRustWorker() as rust:
        for worker in (official, rust):
            game = DirectBattle.start(worker, "replica-wait", *engine_teams, seed=[83, 17, 2, 4])
            game.step({"p1": "team 1234", "p2": "team 3412"})
            game.step(
                {"p1": "move protect, move protect", "p2": "move partingshot -2, move fakeout 1"}
            )

            assert game.request_state == "switch"
            assert game.sides_to_move() == ["p2"]
            request = game.battles["p1"].last_request
            assert request["wait"] is True
            assert request["side"]["id"] == "p1"
            assert all(mon["ident"].startswith("p1: ") for mon in request["side"]["pokemon"])
            assert not game.battles["p1"].available_moves[0]
            assert not game.battles["p1"].available_moves[1]


def test_accepted_choice_is_not_requested_again_after_opponent_rejection(engine_teams):
    first, _ = engine_teams
    gengar = (
        "Gengar||Gengarite|CursedBody|Protect,ShadowBall,SludgeBomb,PerishSong"
        "|Modest|2,,,32,,32||||50|"
    )
    p1_team = "]".join([gengar, *first.split("]")[1:]])
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as official, ReplicaRustWorker() as rust:
        for worker in (official, rust):
            game = DirectBattle.start(worker, "replica-retry", p1_team, first, seed=[1, 2, 3, 4])
            game.step({"p1": "team 1234", "p2": "team 1234"})
            game.step({"p1": "move 1 mega, move 4", "p2": "move 4, move 4"})
            result = game.step({"p1": "move 1, move 4", "p2": "move 2 1, switch 3"})

            assert game.sides_to_move() == ["p2"]
            assert not any(line.startswith("|request|") for line in result.lines["p1"])
            assert game.battles["p2"].trapped == [True, True]


def test_tyranitar_survives_high_horsepower_after_two_uturn_switches():
    # The engines require six team entries even though only the first four are selected.
    p1 = (
        "Lucario||LucarioniteZ|Steadfast|AuraSphere,FlashCannon,CalmMind,Protect"
        "|Timid|2,32,,,,32|M|||50|]"
        "Rillaboom||MiracleSeed|GrassySurge|GrassyGlide,WoodHammer,FakeOut,Uturn"
        "|Adamant|32,32,,,2,|F|||50|]"
        "Incineroar||SitrusBerry|Intimidate|ThroatChop,FlareBlitz,FakeOut,PartingShot"
        "|Careful|32,,14,,20,|F|||50|]"
        "Sneasler||GrassySeed|Unburden|SwordsDance,Protect,CloseCombat,DireClaw"
        "|Adamant|2,32,,,,32|M|||50|]"
        "Aerodactyl||FocusSash|Unnerve|RockSlide,Tailwind,DualWingbeat,WideGuard"
        "|Jolly|2,32,,,,32|M|||50|]"
        "Floette-Eternal||Floettite|FlowerVeil|DazzlingGleam,CalmMind,Moonblast,Protect"
        "|Timid|2,,,32,,32|F|||50|"
    )
    p2 = (
        "Rillaboom||MiracleSeed|GrassySurge|FakeOut,GrassyGlide,WoodHammer,Uturn"
        "|Adamant|32,32,,,2,|F|||50|]"
        "Excadrill||FocusSash|SandRush|IronHead,HighHorsepower,RockSlide,Protect"
        "|Adamant|2,32,,,,32|M|||50|]"
        "Sneasler||GrassySeed|Unburden|CloseCombat,DireClaw,FakeOut,Protect"
        "|Jolly|2,32,,,,32|F|||50|]"
        "Tyranitar||Tyranitarite|SandStream|RockSlide,KnockOff,LowKick,Protect"
        "|Adamant|2,32,,,,32|F|||50|]"
        "Salamence||Salamencite|Intimidate|HyperVoice,DracoMeteor,Tailwind,Protect"
        "|Timid|9,,1,32,1,23|F|||50|]"
        "Milotic||SitrusBerry|Competitive|Scald,IceBeam,IcyWind,Protect"
        "|Modest|32,,29,,5,|F|||50|"
    )
    choices = [
        {"p1": "team 1234", "p2": "team 1234"},
        {"p1": "move flashcannon 1, move uturn 2", "p2": "move uturn 1, move highhorsepower -1"},
        {"p1": "pass, switch Sneasler"},
        {"p2": "switch Tyranitar, pass"},
        {
            "p1": "move protect mega, move direclaw 1",
            "p2": "move lowkick mega 1, move highhorsepower -1",
        },
    ]
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as official, ReplicaRustWorker() as rust:
        for label, worker in (("Showdown", official), ("Rust", rust)):
            game = DirectBattle.start(
                worker, "replica-speed-tie", p1, p2, seed=[0xB317, 0x4902, 0xC366, 0x058F]
            )
            for turn_choices in choices:
                game.step(turn_choices)
            tyranitar = game.battles["p2"].team["p2: Tyranitar"]
            assert tyranitar.current_hp == 23, label
            assert game.request_state == "move", label


def _visible_info(game, side):
    view = game.battles[side]
    return [
        (mon.species, mon.current_hp, mon.max_hp, mon.item, tuple(sorted(mon.moves)))
        for team in (view.team, view.opponent_team)
        for mon in team.values()
    ]


def test_rust_and_showdown_player_views_through_finish(engine_teams):
    rng = random.Random(3107)
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as official, ReplicaRustWorker() as rust:
        first, second = [
            DirectBattle.start(worker, "replica-check", *engine_teams, seed=[83, 17, 2, 4])
            for worker in (official, rust)
        ]
        for _ in range(1000):
            assert first.ended == second.ended
            assert first.sides_to_move() == second.sides_to_move()
            for side in SIDES:
                assert _visible_info(first, side) == _visible_info(second, side)
            if first.ended:
                assert first.winner == second.winner
                return
            choices = {
                side: "team 1234"
                if first.request_state == "teampreview"
                else choice_string(rng.choice(enumerate_joint_orders(first.battles[side])))
                for side in first.sides_to_move()
            }
            first.step(choices)
            second.step(choices)
    pytest.fail("identical seeded battles did not finish")
