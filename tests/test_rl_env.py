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
    start_many,
    step_many,
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
    # which never fires here. Exercise DirectBattle's explicit transport-level
    # compatibility flag; normal direct policy evaluation applies this per agent from
    # PolicyConfig in DirectAgent.choose().
    battle = DirectBattle.start(
        worker, "t-spread", team, team, seed=[5, 5, 5, 5], own_team_spreads=True
    )
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


def _battle_snapshot(battle) -> dict:
    """Every observable property of a parsed battle, for the equivalence test."""

    def mon(pokemon) -> tuple:
        return (
            pokemon.species,
            pokemon.current_hp,
            pokemon.max_hp,
            str(pokemon.status),
            sorted(pokemon.moves),
            pokemon.item,
            pokemon.ability,
            tuple(sorted((pokemon.boosts or {}).items())),
            pokemon.fainted,
            pokemon.active,
            pokemon.level,
        )

    return {
        "turn": battle.turn,
        "won": battle.won,
        "lost": battle.lost,
        "finished": battle.finished,
        "player_role": battle.player_role,
        "teampreview": battle.teampreview,
        "force_switch": tuple(battle.force_switch),
        "maybe_trapped": tuple(battle.maybe_trapped),
        "team": {key: mon(value) for key, value in sorted(battle.team.items())},
        "opponent_team": {
            key: mon(value) for key, value in sorted(battle.opponent_team.items())
        },
        "side_conditions": dict(battle.side_conditions),
        "opponent_side_conditions": dict(battle.opponent_side_conditions),
        "weather": dict(battle.weather),
        "fields": dict(battle.fields),
        "active": tuple(m.species if m else None for m in battle.active_pokemon),
        "opponent_active": tuple(
            m.species if m else None for m in battle.opponent_active_pokemon
        ),
        "teampreview_opponent_team": tuple(
            m.species for m in battle.teampreview_opponent_team
        ),
    }


def _replay_through_poke_env(battle_tag: str, bursts: list[list[str]], team: str):
    """Feed the same protocol lines through poke-env's own message pump.

    This is the reference implementation `vgc.rl.env._ingest` has to match. Both sides
    of the comparison are driven by the SAME simulator output, so any difference is
    ours: which tags we route to `parse_message` vs `parse_request`, and which we skip.

    Two bits of poke-env plumbing have to be neutralised, neither of which touches
    parsing: its team-preview handling tries to SEND a reply (no websocket here), and
    `_create_battle` puts onto a bounded queue that the terminal `|win|` later drains.
    """

    import asyncio

    from poke_env.ps_client.account_configuration import AccountConfiguration

    from vgc.baselines import make_player

    player = make_player(
        "random",
        team,
        start_listening=False,
        account_configuration=AccountConfiguration("alpha", None),
    )

    async def _noop(*_args, **_kwargs):
        return None

    player.ps_client.send_message = _noop

    async def feed() -> None:
        for index, lines in enumerate(bursts):
            messages: list[list[str]] = [[f">{battle_tag}"]]
            if index == 0:
                messages.append(["", "init", "battle"])
            messages.extend(line.split("|") for line in lines)
            await player._handle_battle_message(messages)

    asyncio.run(asyncio.wait_for(feed(), timeout=60))
    return player._battles[battle_tag]


@pytest.mark.integration
def test_direct_env_parses_a_battle_identically_to_poke_envs_own_pump(worker, team: str) -> None:
    """The protocol-equivalence criterion in `docs/rl_roadmap.md`'s Phase 1."""

    tag = "battle-gen9championsvgc2026regmb-1"
    rng = random.Random(0)
    battle = DirectBattle.start(
        worker, tag, team, team, seed=[4, 4, 4, 4], usernames={"p1": "alpha", "p2": "beta"}
    )
    bursts = [list(battle.last_lines["p1"])]
    while not battle.ended:
        choices = {
            side: _random_choice(battle.battles[side], rng) for side in battle.sides_to_move()
        }
        bursts.append(list(battle.step(choices).lines["p1"]))
    battle.close()

    ours = battle.battles["p1"]
    theirs = _replay_through_poke_env(tag, bursts, team)

    assert ours.finished and ours.turn > 1, "battle too trivial to be a real comparison"
    assert _battle_snapshot(ours) == _battle_snapshot(theirs)

    # DirectBattle is the raw transport parser, so it matches poke-env's no-OTS state.
    # Policy-owned enrichment happens in DirectAgent.choose(), where each side's
    # PolicyConfig can independently select true or fallback spreads.
    assert all(pokemon.evs is None for pokemon in theirs.team.values())
    assert all(pokemon.evs is None for pokemon in ours.team.values())


@pytest.mark.integration
def test_an_illegal_choice_raises_instead_of_silently_retrying(worker, team: str) -> None:
    battle = DirectBattle.start(worker, "t-illegal", team, team, seed=[2, 2, 2, 2])
    with pytest.raises(InvalidChoice):
        battle.step({side: "move 1 1, move 1 2" for side in battle.sides_to_move()})
    battle.close()


@pytest.mark.integration
def test_batched_battles_produce_the_same_games_as_stepping_them_one_at_a_time(
    worker, team: str
) -> None:
    """Batching is a throughput change and must not be a behaviour change.

    Same seeds and same scripted choices, run both ways: the per-battle protocol streams
    have to come out identical, or the concurrent settles in the worker's `handleBatch`
    are letting battles observe each other.
    """

    seeds = [[i + 1, 2, 3, 4] for i in range(4)]

    def scripted(battle, step_index: int) -> str:
        # Deterministic without an RNG, so the two runs are comparable by construction.
        if battle.teampreview:
            return "team 1234"
        orders = enumerate_joint_orders(battle)
        return choice_string(orders[step_index % len(orders)]) if orders else "default"

    sequential = []
    for index, seed in enumerate(seeds):
        battle = DirectBattle.start(worker, f"seq-{index}", team, team, seed=seed)
        steps = 0
        while not battle.ended:
            battle.step(
                {side: scripted(battle.battles[side], steps) for side in battle.sides_to_move()}
            )
            steps += 1
        sequential.append((battle.winner, battle.battles["p1"]._replay_data))
        battle.close()

    live = start_many(
        worker, [(f"bat-{i}", team, team) for i in range(len(seeds))], seeds=seeds
    )
    tracked = {battle.battle_id: battle for battle in live}
    steps = 0
    while live:
        step_many(
            worker,
            [
                (b, {side: scripted(b.battles[side], steps) for side in b.sides_to_move()})
                for b in live
            ],
        )
        steps += 1
        live = [b for b in live if not b.ended]
    batched = [
        (tracked[f"bat-{i}"].winner, tracked[f"bat-{i}"].battles["p1"]._replay_data)
        for i in range(len(seeds))
    ]
    for battle in tracked.values():
        battle.close()

    assert [w for w, _ in batched] == [w for w, _ in sequential]
    assert [t for _, t in batched] == [t for _, t in sequential]


@pytest.mark.integration
def test_a_batch_reports_one_failure_without_failing_its_siblings(worker, team: str) -> None:
    good = DirectBattle.start(worker, "batch-ok", team, team, seed=[1, 1, 1, 1])
    results = worker.batch(
        [
            good.step_payload({side: "team 1234" for side in good.sides_to_move()}),
            {"cmd": "choose", "id": "no-such-battle", "p1": "team 1234"},
        ]
    )
    assert "error" not in results[0]
    assert "unknown battle id" in results[1]["error"]
    good.close()


@pytest.mark.integration
def test_step_rejects_a_choice_set_that_does_not_match_who_owes_one(worker, team: str) -> None:
    battle = DirectBattle.start(worker, "t-sides", team, team, seed=[4, 4, 4, 4])
    assert battle.sides_to_move() == ["p1", "p2"]
    with pytest.raises(ValueError, match="expects choices from"):
        battle.step({"p1": "team 1234"})
    battle.close()
