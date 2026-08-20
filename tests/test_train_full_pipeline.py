from __future__ import annotations

import pytest

pytest.importorskip("torch")

from selfplay.train_full_pipeline import (
    TeamEntry,
    evaluate_policy,
    parse_mix,
    sample_mix,
    split_teams_by_archetype,
)


def test_evaluation_schedule_covers_team_opponent_cross_product(monkeypatch) -> None:
    seen: list[tuple[str, str]] = []

    class FakePlayer:
        fallback_count = 0

    class FakeAgent:
        def __init__(self, name: str) -> None:
            self.name = name
            self.player = FakePlayer()

    def fake_policy_agent(_model, team, **_kwargs):
        return FakeAgent(f"policy:{team}")

    def fake_opponent(name, team):
        seen.append((team, name))
        return FakeAgent(name)

    class Outcome:
        def result_for(self, _name):
            return 1.0

    monkeypatch.setattr("selfplay.train_full_pipeline._policy_agent", fake_policy_agent)
    monkeypatch.setattr("selfplay.train_full_pipeline.make_direct_agent", fake_opponent)
    monkeypatch.setattr("selfplay.train_full_pipeline.play_battle", lambda *_a, **_k: Outcome())
    teams = [
        TeamEntry(label="a", archetype="x", packed="team-a"),
        TeamEntry(label="b", archetype="x", packed="team-b"),
    ]

    result = evaluate_policy(
        object(),
        model=type("Model", (), {"eval": lambda self: None})(),
        teams=teams,
        opponents=["random", "heuristic"],
        games=8,
        seed=1,
        device="cpu",
    )

    assert set(seen) == {
        ("team-a", "random"),
        ("team-a", "heuristic"),
        ("team-b", "random"),
        ("team-b", "heuristic"),
    }
    assert result["fallback_count"] == 0


def test_parse_mix_normalizes_and_sampling_is_seeded() -> None:
    import random

    mix = parse_mix("random=1,heuristic=3")
    assert mix == [("random", 0.25), ("heuristic", 0.75)]
    first = random.Random(4)
    second = random.Random(4)
    assert [sample_mix(mix, first) for _ in range(20)] == [
        sample_mix(mix, second) for _ in range(20)
    ]
    with pytest.raises(ValueError, match="name=weight"):
        parse_mix("random")


def test_archetype_split_holds_out_whole_teams_from_every_group() -> None:
    teams = [
        TeamEntry(label=f"{archetype}/{index}", archetype=archetype, packed="team")
        for archetype in ("sun", "rain", "balance")
        for index in range(5)
    ]
    train, holdout = split_teams_by_archetype(
        teams, holdout_fraction=0.2, seed=7
    )
    assert {team.label for team in train}.isdisjoint(team.label for team in holdout)
    assert {team.archetype for team in train} == {"sun", "rain", "balance"}
    assert {team.archetype for team in holdout} == {"sun", "rain", "balance"}
    assert len(holdout) == 3


def test_frozen_evaluation_pins_python_opponent_randomness(monkeypatch) -> None:
    import random

    class FakePlayer:
        fallback_count = 0

    class FakeAgent:
        def __init__(self) -> None:
            self.player = FakePlayer()

    class Outcome:
        def __init__(self, result: float) -> None:
            self.result = result

        def result_for(self, _name):
            return self.result

    monkeypatch.setattr(
        "selfplay.train_full_pipeline._policy_agent", lambda *_a, **_k: FakeAgent()
    )
    monkeypatch.setattr(
        "selfplay.train_full_pipeline.make_direct_agent", lambda *_a, **_k: FakeAgent()
    )
    monkeypatch.setattr(
        "selfplay.train_full_pipeline.play_battle",
        lambda *_a, **_k: Outcome(1.0 if random.random() < 0.5 else -1.0),
    )
    model = type("Model", (), {"eval": lambda self: None})()
    teams = [TeamEntry(label="a", archetype="x", packed="team-a")]
    kwargs = {
        "model": model,
        "teams": teams,
        "opponents": ["random"],
        "games": 20,
        "seed": 44,
        "device": "cpu",
    }

    first = evaluate_policy(object(), **kwargs)
    for _ in range(100):
        random.random()
    second = evaluate_policy(object(), **kwargs)

    assert first == second
