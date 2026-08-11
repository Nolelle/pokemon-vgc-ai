"""Direct `BattleStream` environment (`vgc.rl.env`).

`choice_string` is pure and tested unmodified. Everything else drives the real Node
worker against the local Showdown checkout, so it is `integration`-marked -- no server
is needed (that is the entire point of the module), but `node` and a built `dist/sim`
are. Run with `.venv/bin/python -m pytest -m integration`.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder

from vgc.actions import enumerate_joint_orders
from vgc.rl.env import (
    DEFAULT_SHOWDOWN_REPO,
    DirectBattle,
    InvalidChoice,
    SimWorker,
    choice_string,
)

TEAM_PATH = Path(__file__).resolve().parents[1] / "teams" / "meta1.packed.txt"


# --- choice_string (pure) --------------------------------------------------------------


def test_choice_string_strips_the_client_side_choose_prefix() -> None:
    order = DoubleBattleOrder(
        SingleBattleOrder("/choose move heatwave 1"),
        SingleBattleOrder("/choose move protect"),
    )
    # DoubleBattleOrder.message already trims the second half's prefix; we trim the
    # first, because `>p1 ...` wants the bare choice and not the client command.
    assert choice_string(order) == "move heatwave 1, move protect"


def test_choice_string_accepts_raw_strings_and_leading_slash_commands() -> None:
    assert choice_string("team 1234") == "team 1234"
    assert choice_string("/team 1234") == "team 1234"
    assert choice_string("/choose default") == "default"


# --- the environment itself ------------------------------------------------------------


@pytest.fixture(scope="module")
def team() -> str:
    return TEAM_PATH.read_text().strip()


@pytest.fixture(scope="module")
def worker():
    if not (DEFAULT_SHOWDOWN_REPO / "dist" / "sim" / "index.js").exists():
        pytest.skip(f"no built showdown sim at {DEFAULT_SHOWDOWN_REPO}")
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as running:
        yield running


def _random_choice(battle, rng: random.Random) -> str:
    if battle.teampreview:
        return "team 1234"
    orders = enumerate_joint_orders(battle)
    return choice_string(rng.choice(orders)) if orders else "default"


def _play(worker, team: str, battle_id: str, *, seed: int, policy_seed: int) -> DirectBattle:
    rng = random.Random(policy_seed)
    battle = DirectBattle.start(worker, battle_id, team, team, seed=[seed, seed, seed, seed])
    steps = 0
    while not battle.ended:
        battle.step(
            {side: _random_choice(battle.battles[side], rng) for side in battle.sides_to_move()}
        )
        steps += 1
        assert steps < 500, "battle did not terminate"
    return battle


@pytest.mark.integration
def test_random_battle_runs_to_a_decided_result(worker, team: str) -> None:
    battle = _play(worker, team, "t-complete", seed=3, policy_seed=1)
    assert battle.ended
    assert battle.winner in {"p1", "p2"}
    # poke-env's own view of the outcome must agree with the omniscient stream's.
    won = {side: battle.battles[side].won for side in ("p1", "p2")}
    assert won[battle.winner] is True
    assert battle.outcome(battle.winner) == 1.0
    assert battle.outcome("p1") == -battle.outcome("p2")
    battle.close()


@pytest.mark.integration
def test_each_side_sees_its_own_team_exactly_and_the_opponent_only_fogged(
    worker, team: str
) -> None:
    battle = DirectBattle.start(worker, "t-fog", team, team, seed=[7, 7, 7, 7])
    battle.step({side: "team 1234" for side in battle.sides_to_move()})
    p1 = battle.battles["p1"]

    ours = p1.active_pokemon[0]
    theirs = p1.opponent_active_pokemon[0]
    # Our own side: real HP values and the full moveset from the request.
    assert ours.current_hp == ours.max_hp > 100
    assert len(ours.moves) == 4
    # The opponent, same species, same lead: percentage HP and nothing revealed yet.
    assert theirs.species == ours.species
    assert (theirs.current_hp, theirs.max_hp) == (100, 100)
    assert theirs.moves == {}
    assert theirs.evs is None
    # Bench that has not been sent out is not visible either.
    assert len(p1.opponent_team) < 4

    # The mirror side sees the exact same Pokemon the other way around, which is what
    # makes this fogging and not just missing data.
    p2_own = battle.battles["p2"].active_pokemon[0]
    assert p2_own.current_hp == p2_own.max_hp > 100
    assert len(p2_own.moves) == 4
    battle.close()


@pytest.mark.integration
def test_our_own_stat_points_and_nature_come_from_the_packed_team(worker, team: str) -> None:
    # Regression: poke-env only learns our spread from an Open Team Sheets |showteam|,
    # which never fires here, so vgc.rl.env fills it in itself. Without this the
    # evaluator silently falls back to default_opponent_spread for our OWN team.
    battle = DirectBattle.start(worker, "t-spread", team, team, seed=[5, 5, 5, 5])
    battle.step({side: "team 1234" for side in battle.sides_to_move()})
    charizard = battle.battles["p1"].team["p1: Charizard"]
    assert charizard.evs == [10, 0, 0, 32, 0, 24]
    assert charizard.nature == "modest"
    # The mod-correct stats the sim sent must survive: poke-env's own
    # apply_teambuilder_team would have recomputed these with the VANILLA EV formula.
    assert charizard.stats["hp"] == 163
    assert charizard.stats["spa"] == 177
    battle.close()


@pytest.mark.integration
def test_same_sim_and_policy_seeds_reproduce_the_battle_exactly(worker, team: str) -> None:
    first = _play(worker, team, "t-seed-a", seed=11, policy_seed=5)
    second = _play(worker, team, "t-seed-b", seed=11, policy_seed=5)
    third = _play(worker, team, "t-seed-c", seed=11, policy_seed=6)

    def trace(battle: DirectBattle) -> list[list[str]]:
        return battle.battles["p1"]._replay_data

    assert first.winner == second.winner
    assert trace(first) == trace(second)
    # Both seeds are load-bearing: holding the simulator seed fixed but changing the
    # policy seed must still change the battle, or we are not actually sampling.
    assert trace(first) != trace(third)
    for battle in (first, second, third):
        battle.close()


@pytest.mark.integration
def test_an_illegal_choice_raises_instead_of_silently_retrying(worker, team: str) -> None:
    battle = DirectBattle.start(worker, "t-illegal", team, team, seed=[2, 2, 2, 2])
    with pytest.raises(InvalidChoice):
        battle.step({side: "move 1 1, move 1 2" for side in battle.sides_to_move()})
    battle.close()


@pytest.mark.integration
def test_step_rejects_a_choice_set_that_does_not_match_who_owes_one(worker, team: str) -> None:
    battle = DirectBattle.start(worker, "t-sides", team, team, seed=[4, 4, 4, 4])
    assert battle.sides_to_move() == ["p1", "p2"]
    with pytest.raises(ValueError, match="expects choices from"):
        battle.step({"p1": "team 1234"})
    battle.close()
