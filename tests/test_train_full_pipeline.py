from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from selfplay.train_full_pipeline import (  # noqa: E402
    TeamEntry,
    crossed_milestone,
    evaluate_policy,
    load_resume_state,
    parse_args,
    parse_mix,
    sample_mix,
    save_training_checkpoint,
    split_teams_by_archetype,
)
from vgc.rl.model import CandidatePolicyValueNet  # noqa: E402
from vgc.rl.opponents import RL_ARCHITECTURE_VERSION  # noqa: E402
from vgc.rl.ppo import PpoConfig  # noqa: E402


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
    train, holdout = split_teams_by_archetype(teams, holdout_fraction=0.2, seed=7)
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

    monkeypatch.setattr("selfplay.train_full_pipeline._policy_agent", lambda *_a, **_k: FakeAgent())
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


# --- scaling-study support (split/eval seed decoupling, milestones, resume, extras) ---


def test_split_and_eval_seeds_default_to_the_frozen_promotion_run_seed() -> None:
    args = parse_args([])
    assert args.split_seed == 20260815
    assert args.eval_seed == 20260815
    custom = parse_args(["--seed", "999", "--split-seed", "123", "--eval-seed", "456"])
    assert (custom.seed, custom.split_seed, custom.eval_seed) == (999, 123, 456)


def test_team_split_is_identical_regardless_of_training_seed_value() -> None:
    teams = [
        TeamEntry(label=f"{archetype}/{index}", archetype=archetype, packed="team")
        for archetype in ("sun", "rain")
        for index in range(6)
    ]
    train_a, hold_a = split_teams_by_archetype(teams, holdout_fraction=0.2, seed=20260815)
    train_b, hold_b = split_teams_by_archetype(teams, holdout_fraction=0.2, seed=20260815)
    assert [team.label for team in train_a] == [team.label for team in train_b]
    assert [team.label for team in hold_a] == [team.label for team in hold_b]


@pytest.mark.parametrize(
    "prev_games,games,every,expected",
    [
        (7936, 8192, 8192, True),
        (8192, 8448, 8192, False),
        (7936, 16384, 8192, True),
        (0, 256, 8192, False),
        (0, 0, 0, False),
        (100, 356, 0, False),
    ],
)
def test_crossed_milestone_fires_once_per_interval_crossing(
    prev_games: int, games: int, every: int, expected: bool
) -> None:
    assert crossed_milestone(prev_games, games, every) is expected


def _tiny_checkpoint(path, *, with_optimizer: bool = True) -> CandidatePolicyValueNet:
    torch.manual_seed(7)
    model = CandidatePolicyValueNet(use_information_features=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    save_training_checkpoint(
        path,
        model,
        optimizer,
        iteration=3,
        games_seen=8192,
        ppo_config=PpoConfig(),
        include_optimizer=with_optimizer,
    )
    return model


def test_milestone_save_omits_optimizer_state(tmp_path) -> None:
    path = tmp_path / "games_008192.pt"
    _tiny_checkpoint(path, with_optimizer=False)
    payload = torch.load(path, weights_only=False)
    assert "optimizer_state_dict" not in payload
    assert payload["games_seen"] == 8192
    assert payload["architecture"] == RL_ARCHITECTURE_VERSION


def test_load_resume_state_restores_model_optimizer_and_counters(tmp_path) -> None:
    path = tmp_path / "latest.pt"
    original = _tiny_checkpoint(path)

    loaded_model, loaded_optimizer, iteration, games_seen = load_resume_state(path)
    assert (iteration, games_seen) == (3, 8192)
    for source, restored in zip(original.parameters(), loaded_model.parameters()):
        assert torch.equal(source, restored)
    assert loaded_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_load_resume_state_rejects_foreign_architecture(tmp_path) -> None:
    path = tmp_path / "latest.pt"
    _tiny_checkpoint(path)
    payload = torch.load(path, weights_only=False)
    payload["architecture"] = "some-other-version"
    torch.save(payload, path)
    with pytest.raises(SystemExit, match="architecture"):
        load_resume_state(path)


def test_load_resume_state_rejects_checkpoint_without_optimizer(tmp_path) -> None:
    path = tmp_path / "milestone_only.pt"
    _tiny_checkpoint(path, with_optimizer=False)
    with pytest.raises(SystemExit, match="no optimizer state"):
        load_resume_state(path)


def test_evaluate_policy_supports_checkpoint_opponents_and_game_records(
    monkeypatch, tmp_path
) -> None:
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
    monkeypatch.setattr(
        "selfplay.train_full_pipeline.load_snapshot",
        lambda _path, device="cpu": object(),
    )
    records = tmp_path / "games.jsonl"

    evaluate_policy(
        object(),
        model=type("Model", (), {"eval": lambda self: None})(),
        teams=[TeamEntry(label="a", archetype="x", packed="team-a")],
        opponents=["random"],
        games=6,
        seed=5,
        device="cpu",
        extra_opponents={"rl10k": Path("old.pt")},
        record_games_path=records,
    )

    # Baseline opponents go through make_direct_agent; the checkpoint opponent does
    # not (it is built from the loaded model), so its participation shows up in the
    # per-game records instead.
    assert set(seen) == {("team-a", "random")}
    rows = [json.loads(line) for line in records.read_text().splitlines()]
    assert len(rows) == 6
    assert {row["game_index"] for row in rows} == set(range(6))
    assert {row["opponent"] for row in rows} == {"random", "rl10k"}
    assert all(row["win"] == 1.0 for row in rows)
    assert all(set(row) == {"game_index", "team", "opponent", "side", "win"} for row in rows)
