"""Direct-environment agents and match running (`vgc.rl.agents`, `vgc.rl.match`).

Aggregation and guard rails are pure unit tests. Anything that needs a real battle is
`integration`-marked: it drives the Node worker against the local Showdown checkout, but
still needs no server. Run with `.venv/bin/python -m pytest -m integration`.
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from vgc.rl.agents import PHASE2_PREVIEW_ORDER, DirectAgent, make_direct_agent
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.match import BattleOutcome, play_battle, run_series, summarize

TEAM_PATH = Path(__file__).resolve().parents[1] / "teams" / "meta1.packed.txt"


# --- aggregation and guards (pure) ------------------------------------------------------


def _outcome(winner: str | None, sides: dict[str, str]) -> BattleOutcome:
    return BattleOutcome(
        battle_id="b", winner=winner, turns=7, decisions=14, agent_sides=sides
    )


def test_result_for_reads_the_winner_through_the_seat_assignment() -> None:
    swapped = _outcome("p2", {"alpha": "p2", "beta": "p1"})
    assert swapped.result_for("alpha") == 1.0
    assert swapped.result_for("beta") == -1.0
    drawn = _outcome(None, {"alpha": "p1", "beta": "p2"})
    assert drawn.result_for("alpha") == drawn.result_for("beta") == 0.0


def test_summarize_counts_wins_by_agent_not_by_seat() -> None:
    # alpha wins both, but from a different seat each time -- a seat-based count would
    # report 1-1 here, which is the bug alternating seats exists to expose.
    outcomes = [
        _outcome("p1", {"alpha": "p1", "beta": "p2"}),
        _outcome("p2", {"alpha": "p2", "beta": "p1"}),
    ]
    result = summarize(outcomes, "alpha", "beta")
    assert (result["p1_wins"], result["p2_wins"], result["draws"]) == (2, 0, 0)
    assert result["p1_win_rate"] == 1.0
    assert result["environment"] == "direct"
    # Same keys as the websocket path, so gate reports keep working across the migration.
    assert {"games", "p1_wilson", "p2_wilson", "timestamp"} <= set(result)


def test_play_battle_refuses_two_agents_with_the_same_name() -> None:
    agents = {side: DirectAgent(SimpleNamespace(), name="vgc") for side in ("p1", "p2")}
    with pytest.raises(ValueError, match="both agents are named"):
        play_battle(None, "b", agents, {"p1": "", "p2": ""})


def test_observe_is_a_no_op_for_players_without_battle_memory() -> None:
    # poke-env's own baselines have no _memory_for_tag; callers should not have to care.
    agent = DirectAgent(SimpleNamespace(), name="random")
    agent.observe("tag", ["|move|p2a: Incineroar|Fake Out|p1a: Charizard"])


# --- against the real simulator ---------------------------------------------------------


@pytest.fixture(scope="module")
def team() -> str:
    return TEAM_PATH.read_text().strip()


@pytest.fixture(scope="module")
def worker():
    if not (DEFAULT_SHOWDOWN_REPO / "dist" / "sim" / "index.js").exists():
        pytest.skip(f"no built showdown sim at {DEFAULT_SHOWDOWN_REPO}")
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as running:
        yield running


@pytest.mark.integration
@pytest.mark.parametrize("name", ["random", "maxpower", "heuristic", "vgc"])
def test_every_registered_baseline_can_play_in_the_direct_env(worker, team: str, name: str) -> None:
    # The whole point of vgc.rl.agents: these are networked poke-env Players, and they
    # must run unmodified against a battle the direct env parsed.
    opponent = make_direct_agent("random", team)
    opponent.name = "opponent"  # distinct from `name`, which may itself be "random"
    outcome = play_battle(
        worker,
        f"base-{name}",
        {"p1": make_direct_agent(name, team), "p2": opponent},
        {"p1": team, "p2": team},
        seed=[3, 3, 3, 3],
    )
    assert outcome.winner in {"p1", "p2"}
    assert outcome.decisions > 0
    assert outcome.turns > 0


@pytest.mark.integration
def test_run_series_gives_each_agent_the_p1_seat_exactly_half_the_time(
    worker, team: str
) -> None:
    from functools import partial

    games = 8
    outcomes = run_series(
        worker,
        {
            "random-a": partial(make_direct_agent, "random", team),
            "random-b": partial(make_direct_agent, "random", team),
        },
        {"random-a": team, "random-b": team},
        games,
        seed=2,
    )
    assert len(outcomes) == games
    assert sum(1 for o in outcomes if o.agent_sides["random-a"] == "p1") == games // 2


@pytest.mark.integration
def test_the_phase2_mirror_always_brings_the_same_four_in_the_same_slots(worker) -> None:
    """Phase 2's fixed mirror (`docs/rl_roadmap.md`).

    The format requires SIX Pokemon on the roster -- `validate-team` rejects a four-mon
    team outright -- so the pick is pinned with a hardcoded preview order instead, over a
    roster ordered so that order string brings the intended four. Letting the heuristic
    preview choose would hand the agent a different four-of-six per matchup, which is the
    non-stationarity Phase 2 exists to remove.
    """

    mirror_team = (TEAM_PATH.parent / "phase2_mirror.packed.txt").read_text().strip()
    leads: set[tuple] = set()
    rosters: set[tuple] = set()
    for index in range(4):
        agents = {}
        for side, label in (("p1", "a"), ("p2", "b")):
            agent = make_direct_agent(
                "random", mirror_team, preview_order=PHASE2_PREVIEW_ORDER
            )
            agent.name = label
            agents[side] = agent
        battle = DirectBattle.start(
            worker, f"mirror-{index}", mirror_team, mirror_team, seed=[index + 1] * 4
        )
        battle.step({side: agents[side].choose(battle.battles[side]) for side in ("p1", "p2")})
        for side in ("p1", "p2"):
            parsed = battle.battles[side]
            leads.add(tuple(m.species for m in parsed.active_pokemon))
            rosters.add(tuple(sorted(p.species for p in parsed.team.values())))
        battle.close()

    assert len(leads) == 1, f"lead pair varied across battles/sides: {leads}"
    assert leads == {("charizard", "venusaur")}
    assert len(rosters) == 1, f"picked roster varied: {rosters}"


def test_preview_order_overrides_the_players_own_team_preview() -> None:
    class _Player:
        def teampreview(self, _battle):
            raise AssertionError("the player's own preview must not be consulted")

    agent = DirectAgent(_Player(), name="fixed", preview_order=PHASE2_PREVIEW_ORDER)
    assert agent.choose(SimpleNamespace(teampreview=True)) == "team 1234"


def test_direct_agent_applies_own_spreads_only_when_its_policy_enables_them(
    monkeypatch,
) -> None:
    calls = []

    class _Player:
        def __init__(self, enabled: bool) -> None:
            self.config = SimpleNamespace(use_own_team_spreads=enabled)

        def teampreview(self, _battle):
            return "team 1234"

    monkeypatch.setattr("vgc.rl.agents.apply_own_spreads", calls.append)
    battle = SimpleNamespace(teampreview=True)

    DirectAgent(_Player(True)).choose(battle)
    DirectAgent(_Player(False)).choose(battle)

    assert calls == [battle]


@pytest.mark.integration
def test_battle_memory_accumulates_from_the_protocol_lines(worker, team: str) -> None:
    # BattleMemory is normally filled by VgcPlayer._handle_battle_message, the transport
    # hook the direct env replaces; DirectAgent.observe is what keeps it fed.
    agent = make_direct_agent("vgc", team)
    play_battle(
        worker,
        "mem",
        {"p1": agent, "p2": make_direct_agent("random", team)},
        {"p1": team, "p2": team},
        seed=[8, 8, 8, 8],
    )
    memory = agent.player._memory_for_tag("mem")
    assert memory.opponent_moves, "no opponent moves recorded"
    assert memory.our_orders, "no own orders recorded"


@pytest.mark.integration
def test_a_full_episode_replays_exactly_under_sim_policy_and_opponent_seeds(
    worker, team: str
) -> None:
    """`docs/rl_roadmap.md`'s deterministic-replay criterion, with a real network.

    THREE seeds are load-bearing, not two: the simulator's, our policy's sampling
    generator, and Python's global `random` -- poke-env's `RandomPlayer` draws from the
    module-level RNG, so an unseeded opponent desynchronises the replay even when both
    of ours are pinned.
    """

    torch = pytest.importorskip("torch")
    from vgc.rl.model import CandidatePolicyValueNet
    from vgc.rl.player import PpoVgcPlayer
    from vgc.rl.ppo import PpoConfig, RolloutBuffer

    torch.manual_seed(0)
    model = CandidatePolicyValueNet()

    def episode(tag: str, *, policy_seed: int, opponent_seed: int):
        random.seed(opponent_seed)
        buffer = RolloutBuffer()
        learner = PpoVgcPlayer(
            model=model,
            rollout_buffer=buffer,
            ppo_config=PpoConfig(),
            policy_seed=policy_seed,
            start_listening=False,
        )
        outcome = play_battle(
            worker,
            tag,
            {"p1": DirectAgent(learner, name="ppo"), "p2": make_direct_agent("random", team)},
            {"p1": team, "p2": team},
            seed=[9, 9, 9, 9],
        )
        return outcome.winner, tuple(step.action_index for step in buffer.steps)

    baseline = episode("rep-a", policy_seed=42, opponent_seed=7)
    assert episode("rep-b", policy_seed=42, opponent_seed=7) == baseline
    assert len(baseline[1]) > 1, "episode too short to be a meaningful replay check"
    # Each seed independently changes the episode, so none of them is dead weight.
    assert episode("rep-c", policy_seed=43, opponent_seed=7) != baseline
    assert episode("rep-d", policy_seed=42, opponent_seed=8) != baseline


@pytest.mark.integration
def test_rollout_steps_get_returns_when_the_battle_ends(worker, team: str) -> None:
    # DirectAgent.finish is what closes the episode; without it steps would be collected
    # and never turned into returns.
    torch = pytest.importorskip("torch")
    from vgc.rl.model import CandidatePolicyValueNet
    from vgc.rl.player import PpoVgcPlayer
    from vgc.rl.ppo import PpoConfig, RolloutBuffer

    torch.manual_seed(0)
    buffer = RolloutBuffer()
    learner = PpoVgcPlayer(
        model=CandidatePolicyValueNet(),
        rollout_buffer=buffer,
        ppo_config=PpoConfig(),
        policy_seed=1,
        start_listening=False,
    )
    outcome = play_battle(
        worker,
        "rollout",
        {"p1": DirectAgent(learner, name="ppo"), "p2": make_direct_agent("random", team)},
        {"p1": team, "p2": team},
        seed=[6, 6, 6, 6],
    )
    assert buffer.steps
    assert buffer.steps[-1].done is True
    expected = 1.0 if outcome.winner == "p1" else -1.0
    assert buffer.steps[-1].reward == pytest.approx(expected)
