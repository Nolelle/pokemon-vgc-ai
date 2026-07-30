"""Candidate encoding, legal masking, GAE, and PPO update smoke tests."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from poke_env.battle.field import Field  # noqa: E402
from poke_env.battle.move import Move  # noqa: E402
from poke_env.battle.pokemon import Pokemon  # noqa: E402
from poke_env.battle.side_condition import SideCondition  # noqa: E402

import selfplay.train_ppo as train_ppo  # noqa: E402
from selfplay.train_ppo import (  # noqa: E402
    OpponentTeamChoice,
    allocate_games,
    allocate_pairing_games,
    build_opponent_team_schedule,
    build_opponent_team_schedule_groups,
    evaluate_frozen_policy,
    load_diverse_opponent_teams,
    load_training_checkpoint,
    save_checkpoint,
    split_holdout_teams,
    teacher_anchor_weight_for_iteration,
)
from vgc.archetypes import ARCHETYPES  # noqa: E402
from vgc.bc.encoding import (  # noqa: E402
    MOVE_TO_IDX,
    INDEX_DIM,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    STATE_SCALAR_DIM,
    TARGET_TO_IDX,
)
from vgc.bc.model import HIDDEN_DIM  # noqa: E402
from vgc.battle_memory import BattleMemory, TurnMemory  # noqa: E402
from vgc.rl.encoding import (  # noqa: E402
    CandidateFeatures,
    HISTORY_SCALAR_DIM,
    META_ARCHETYPE_SLOTS,
    META_SCALAR_DIM,
    encode_battle_history,
    encode_candidates,
    encode_meta_context,
    pad_candidate_features,
)
from vgc.rl.distill import (  # noqa: E402
    DistillationConfig,
    DistillationSample,
    distill_policy,
    evaluate_agreement,
    split_samples_by_battle,
    teacher_action_index,
)
from vgc.rl.model import CandidatePolicyValueNet, HISTORY_HIDDEN_DIM, META_HIDDEN_DIM  # noqa: E402
from vgc.rl.opponents import (  # noqa: E402
    choose_opponent,
    discover_snapshots,
    load_snapshot,
    save_snapshot,
)
from vgc.rl.ppo import (  # noqa: E402
    PpoConfig,
    RolloutBuffer,
    RolloutStep,
    ppo_update,
    select_action,
)
from vgc.rl.rewards import board_potential  # noqa: E402


def _single(target, *, move_target=0, mega=False):
    return SimpleNamespace(
        order=target,
        move_target=move_target,
        mega=mega,
        z_move=False,
        dynamax=False,
        terastallize=False,
    )


def _joint(first, second):
    return SimpleNamespace(first_order=first, second_order=second)


def _candidate_features(count: int) -> CandidateFeatures:
    return CandidateFeatures(
        move_indices=np.full((count, 2), MOVE_TO_IDX["protect"], dtype=np.int64),
        target_indices=np.full((count, 2), TARGET_TO_IDX["self_or_field"], dtype=np.int64),
        switch_species_indices=np.full((count, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64),
        flags=np.zeros((count, 2, 4), dtype=np.float32),
    )


def _distinct_candidates() -> CandidateFeatures:
    moves = ("protect", "dragonclaw", "heatwave")
    return CandidateFeatures(
        move_indices=np.asarray(
            [[MOVE_TO_IDX[move], MOVE_TO_IDX["protect"]] for move in moves],
            dtype=np.int64,
        ),
        target_indices=np.asarray(
            [
                [TARGET_TO_IDX["self_or_field"], TARGET_TO_IDX["self_or_field"]],
                [TARGET_TO_IDX["opp0"], TARGET_TO_IDX["self_or_field"]],
                [TARGET_TO_IDX["spread"], TARGET_TO_IDX["self_or_field"]],
            ],
            dtype=np.int64,
        ),
        switch_species_indices=np.full((3, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64),
        flags=np.zeros((3, 2, 4), dtype=np.float32),
    )


def _state() -> tuple[np.ndarray, np.ndarray]:
    # The exact semantic values do not matter for the tensor/optimization smoke; valid
    # vocab indices and the real encoder widths do.
    return (
        np.zeros(INDEX_DIM, dtype=np.int64),
        np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
    )


def _history() -> np.ndarray:
    return np.zeros(HISTORY_SCALAR_DIM, dtype=np.float32)


def test_encode_candidates_preserves_joint_move_targets_switch_and_mega() -> None:
    attack = Move("dragonclaw", gen=9)
    switch = Pokemon(gen=9, species="incineroar")
    protect = Move("protect", gen=9)
    orders = [
        _joint(_single(attack, move_target=2, mega=True), _single(switch)),
        _joint(_single(protect), _single(None)),
    ]

    encoded = encode_candidates(orders)

    assert len(encoded) == 2
    assert encoded.move_indices[0].tolist() == [
        MOVE_TO_IDX["dragonclaw"],
        MOVE_TO_IDX["<switch>"],
    ]
    assert encoded.target_indices[0, 0] == TARGET_TO_IDX["opp1"]
    assert encoded.switch_species_indices[0, 1] == SPECIES_TO_IDX["incineroar"]
    assert encoded.flags[0, 0, 0] == 1.0
    assert encoded.move_indices[1, 1] == MOVE_TO_IDX["<pass>"]


def test_pad_candidate_features_marks_only_real_candidates_legal() -> None:
    one = _candidate_features(1)
    three = _candidate_features(3)
    _moves, _targets, _species, _flags, mask = pad_candidate_features([one, three])
    assert mask.tolist() == [[True, False, False], [True, True, True]]


def test_masked_policy_never_samples_padded_candidate() -> None:
    model = CandidatePolicyValueNet()
    state_indices, state_scalars = _state()
    candidates = _candidate_features(2)
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    # Add a third padded candidate and ensure it remains impossible even over many draws.
    moves = np.pad(moves, ((0, 0), (0, 1), (0, 0)))
    targets = np.pad(targets, ((0, 0), (0, 1), (0, 0)))
    species = np.pad(species, ((0, 0), (0, 1), (0, 0)))
    flags = np.pad(flags, ((0, 0), (0, 1), (0, 0), (0, 0)))
    mask = np.pad(mask, ((0, 0), (0, 1)), constant_values=False)

    sampled = set()
    for _ in range(50):
        action, _log_prob, _value = select_action(
            model,
            torch.as_tensor(state_indices[None, :]),
            torch.as_tensor(state_scalars[None, :]),
            torch.as_tensor(_history()[None, :]),
            torch.as_tensor(moves),
            torch.as_tensor(targets),
            torch.as_tensor(species),
            torch.as_tensor(flags),
            torch.as_tensor(mask),
        )
        sampled.add(int(action.item()))
    assert sampled <= {0, 1}


def test_model_rejects_state_with_no_legal_candidate() -> None:
    model = CandidatePolicyValueNet()
    state_indices, state_scalars = _state()
    candidates = _candidate_features(1)
    moves, targets, species, flags, _mask = pad_candidate_features([candidates])
    with pytest.raises(ValueError, match="at least one legal"):
        model(
            torch.as_tensor(state_indices[None, :]),
            torch.as_tensor(state_scalars[None, :]),
            torch.as_tensor(_history()[None, :]),
            torch.as_tensor(moves),
            torch.as_tensor(targets),
            torch.as_tensor(species),
            torch.as_tensor(flags),
            torch.zeros((1, 1), dtype=torch.bool),
        )


def test_finish_episode_propagates_terminal_win_backward() -> None:
    buffer = RolloutBuffer()
    indices, scalars = _state()
    for _ in range(3):
        buffer.add(
            RolloutStep(
                state_indices=indices,
                state_scalars=scalars,
                history_scalars=_history(),
                candidates=_candidate_features(2),
                action_index=0,
                old_log_prob=-0.69,
                old_value=0.0,
            )
        )
    buffer.finish_episode(1.0, PpoConfig(gamma=1.0, gae_lambda=1.0))
    assert [step.reward for step in buffer.steps] == [0.0, 0.0, 1.0]
    assert [step.return_value for step in buffer.steps] == pytest.approx([1.0, 1.0, 1.0])


def test_ppo_update_changes_parameters_and_returns_finite_metrics() -> None:
    torch.manual_seed(0)
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = RolloutBuffer()
    indices, scalars = _state()
    candidates = _candidate_features(3)
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    with torch.no_grad():
        action, log_prob, value = select_action(
            model,
            torch.as_tensor(indices[None, :]),
            torch.as_tensor(scalars[None, :]),
            torch.as_tensor(_history()[None, :]),
            torch.as_tensor(moves),
            torch.as_tensor(targets),
            torch.as_tensor(species),
            torch.as_tensor(flags),
            torch.as_tensor(mask),
        )
    for outcome in (1.0, -1.0, 1.0, 1.0):
        buffer.add(
            RolloutStep(
                state_indices=indices.copy(),
                state_scalars=scalars.copy(),
                history_scalars=_history(),
                candidates=candidates,
                action_index=int(action.item()),
                old_log_prob=float(log_prob.item()),
                old_value=float(value.item()),
            )
        )
        buffer.finish_episode(outcome, PpoConfig())

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = ppo_update(
        model,
        optimizer,
        buffer,
        PpoConfig(epochs=2, minibatch_size=2),
    )

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["steps"] == 4.0
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )


def test_teacher_anchor_increases_probability_of_search_action() -> None:
    torch.manual_seed(4)
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    buffer = RolloutBuffer()
    indices, scalars = _state()
    candidates = _distinct_candidates()

    def teacher_probability() -> float:
        moves, targets, species, flags, mask = pad_candidate_features([candidates])
        with torch.no_grad():
            logits, _value = model(
                torch.as_tensor(indices[None, :]),
                torch.as_tensor(scalars[None, :]),
                torch.as_tensor(_history()[None, :]),
                torch.as_tensor(moves),
                torch.as_tensor(targets),
                torch.as_tensor(species),
                torch.as_tensor(flags),
                torch.as_tensor(mask),
            )
        return float(logits.softmax(dim=-1)[0, 0])

    for _ in range(16):
        buffer.add(
            RolloutStep(
                state_indices=indices.copy(),
                state_scalars=scalars.copy(),
                history_scalars=_history(),
                candidates=candidates,
                action_index=1,
                old_log_prob=-1.0,
                old_value=0.0,
                teacher_action_index=0,
            )
        )
        buffer.finish_episode(0.0, PpoConfig())

    before = teacher_probability()
    metrics = ppo_update(
        model,
        optimizer,
        buffer,
        PpoConfig(
            epochs=8,
            minibatch_size=16,
            value_loss_weight=0.0,
            entropy_weight=0.0,
            teacher_anchor_weight=1.0,
        ),
    )
    after = teacher_probability()

    assert after > before + 0.1
    assert metrics["teacher_coverage"] == 1.0
    assert metrics["teacher_anchor_loss"] > 0.0


def test_teacher_action_index_matches_search_choice_to_network_candidates(monkeypatch) -> None:
    protect = Move("protect", gen=9)
    attack = Move("dragonclaw", gen=9)
    orders = [
        _joint(_single(protect), _single(protect)),
        _joint(_single(attack, move_target=1), _single(protect)),
    ]
    monkeypatch.setattr(
        "vgc.rl.distill.search_joint_orders",
        lambda _battle, _config: [SimpleNamespace(order=orders[1])],
    )
    config = SimpleNamespace(use_two_ply_search=True, use_heuristic_evaluator=True)

    assert teacher_action_index(SimpleNamespace(), config, orders) == 1


def test_encode_battle_history_captures_longitudinal_signals() -> None:
    memory = BattleMemory(battle_tag="battle-test")
    memory.opponent_moves["amoonguss"]["protect"] = 1
    memory.opponent_moves["amoonguss"]["spore"] = 2
    memory.opponent_protects["amoonguss"] = 1
    memory.opponent_switches["incineroar"] = 2
    memory.opponent_targets["charizard"] = 3
    memory.opponent_targets["farigiraf"] = 1
    memory.our_orders = [(1, "protect / heatwave"), (2, "protect / heatwave")]
    memory.turns = [
        TurnMemory(
            turn=1,
            our_active=("charizard", "farigiraf"),
            opponent_active=("amoonguss", "incineroar"),
            our_hp={"charizard": 100.0, "farigiraf": 100.0},
            opponent_hp={"amoonguss": 100.0, "incineroar": 100.0},
            weather=("sun",),
        ),
        TurnMemory(
            turn=2,
            our_active=("charizard", "venusaur"),
            opponent_active=("amoonguss", "torkoal"),
            our_hp={"charizard": 50.0, "venusaur": 100.0},
            opponent_hp={"amoonguss": 25.0, "torkoal": 100.0},
            weather=("rain",),
        ),
    ]

    encoded = encode_battle_history(memory)

    assert encoded.shape == (HISTORY_SCALAR_DIM,)
    assert encoded[3] == pytest.approx(1 / 3)  # observed Protect rate
    assert encoded[5] == pytest.approx(0.75)  # target concentration
    assert encoded[7] == 1.0  # repeated our last order
    assert encoded[9] == pytest.approx(2 / 3)  # repeated opponent move concentration
    assert encoded[10] > 0.0  # opponent switch diversity
    assert encoded[11] == pytest.approx(0.5)  # two distinct observed targets
    assert encoded[12] > 0.0  # weather changed
    assert encoded[13] > 0.0 and encoded[14] > 0.0  # both sides changed an active


def test_rollout_buffer_discards_only_unfinished_tail_and_merges_finished_workers() -> None:
    indices, scalars = _state()

    def step() -> RolloutStep:
        return RolloutStep(
            state_indices=indices.copy(),
            state_scalars=scalars.copy(),
            history_scalars=_history(),
            candidates=_candidate_features(2),
            action_index=0,
            old_log_prob=-0.69,
            old_value=0.0,
        )

    worker = RolloutBuffer()
    worker.add(step())
    worker.finish_episode(1.0, PpoConfig())
    worker.add(step())
    worker.add(step())
    assert worker.discard_unfinished_episode() == 2
    assert len(worker) == 1

    combined = RolloutBuffer()
    combined.extend_finished(worker)
    assert len(combined) == 1


def test_allocate_games_is_balanced_exact_and_never_creates_empty_workers() -> None:
    assert allocate_games(10, 4) == [3, 3, 2, 2]
    assert allocate_games(2, 8) == [1, 1]
    assert sum(allocate_games(17, 3)) == 17
    with pytest.raises(ValueError):
        allocate_games(0, 1)


def test_opponent_team_schedule_is_exact_deterministic_and_can_pair_sides() -> None:
    varied = [
        OpponentTeamChoice(label=f"team-{index}", packed=f"packed-{index}", group="diverse")
        for index in range(5)
    ]
    schedule = build_opponent_team_schedule(
        4,
        learner_team="learner",
        diverse_teams=varied,
        mirror_fraction=0.25,
        seed=7,
    )
    repeated = build_opponent_team_schedule(
        4,
        learner_team="learner",
        diverse_teams=varied,
        mirror_fraction=0.25,
        seed=7,
    )
    paired = build_opponent_team_schedule(
        4,
        learner_team="learner",
        diverse_teams=varied,
        mirror_fraction=0.50,
        seed=7,
        pair_groups=True,
    )

    assert schedule == repeated
    assert [choice.group for choice in schedule].count("mirror") == 1
    assert [choice.group for choice in schedule].count("diverse") == 3
    assert [choice.group for choice in paired] == ["mirror", "mirror", "diverse", "diverse"]


@pytest.mark.parametrize(
    "games,jobs,pairings_per_worker",
    [
        (50, 8, 4),  # the real smoke-test shape
        (10, 4, 1),
        (17, 3, 5),
        (2, 8, 4),  # games < jobs
        (6, 3, 10),  # games-per-worker < pairings_per_worker
        (1, 1, 4),
    ],
)
def test_allocate_pairing_games_sums_to_the_requested_budget(
    games: int, jobs: int, pairings_per_worker: int
) -> None:
    per_worker = allocate_pairing_games(games, jobs, pairings_per_worker)
    assert sum(sum(pairing_games) for pairing_games in per_worker) == games
    for pairing_games in per_worker:
        # Every pairing must get at least 1 game -- allocate_games never creates an
        # empty/zero-game chunk.
        assert all(count > 0 for count in pairing_games)
        assert len(pairing_games) <= pairings_per_worker


def test_allocate_pairing_games_at_one_pairing_reproduces_allocate_games() -> None:
    assert allocate_pairing_games(10, 4, 1) == [[count] for count in allocate_games(10, 4)]
    assert allocate_pairing_games(2, 8, 1) == [[count] for count in allocate_games(2, 8)]


def test_allocate_pairing_games_rejects_nonpositive_pairings_per_worker() -> None:
    with pytest.raises(ValueError):
        allocate_pairing_games(10, 4, 0)


def test_opponent_team_schedule_groups_at_one_pairing_reproduces_the_flat_schedule() -> None:
    varied = [
        OpponentTeamChoice(label=f"team-{index}", packed=f"packed-{index}", group="diverse")
        for index in range(5)
    ]
    flat = build_opponent_team_schedule(
        4,
        learner_team="learner",
        diverse_teams=varied,
        mirror_fraction=0.25,
        seed=7,
        pair_groups=True,
    )
    grouped = build_opponent_team_schedule_groups(
        4,
        learner_team="learner",
        diverse_teams=varied,
        mirror_fraction=0.25,
        seed=7,
        pairings_per_worker=1,
        pair_groups=True,
    )
    assert grouped == [[choice] for choice in flat]


def test_opponent_team_schedule_groups_sample_many_distinct_pairings() -> None:
    varied = [
        OpponentTeamChoice(label=f"team-{index}", packed=f"packed-{index}", group="diverse")
        for index in range(20)
    ]
    groups = build_opponent_team_schedule_groups(
        8,
        learner_team="learner",
        diverse_teams=varied,
        mirror_fraction=0.25,
        seed=7,
        pairings_per_worker=4,
        pair_groups=True,
    )
    assert len(groups) == 8
    for group in groups:
        assert len(group) == 4
    distinct_labels = {choice.label for group in groups for choice in group}
    # 8 workers x 4 pairings = 32 draws over 20 distinct teams -- substantially more
    # than the 8 distinct matchups the pre-fix single-pairing-per-worker design gave.
    assert len(distinct_labels) > 8


async def _fake_evaluate_worker(
    _model,
    *,
    worker_id: int,
    pairings: list[tuple[str, OpponentTeamChoice]],
    pairing_games: list[int],
    device: str,
    learner_challenges: bool,
) -> dict[str, object]:
    """Deterministic stand-in for ``_evaluate_worker`` (no poke-env players, no
    server): every pairing "wins" all but its own index-mod-3 games, so
    ``by_team_group``'s per-pairing totals are checkable by direct arithmetic. Used to
    unit-test the caller's aggregation-over-pairings logic in isolation.
    """

    pairing_rows: list[dict[str, object]] = []
    total_games = total_wins = total_losses = 0
    for index, ((_learner_team, opponent_team), games) in enumerate(zip(pairings, pairing_games)):
        losses = index % 3
        losses = min(losses, games)
        wins = games - losses
        total_games += games
        total_wins += wins
        total_losses += losses
        pairing_rows.append(
            {
                "requested": games,
                "games": games,
                "wins": wins,
                "losses": losses,
                "opponent_team": opponent_team.label,
                "team_group": opponent_team.group,
                "error": None,
            }
        )
    return {
        "worker": worker_id,
        "requested": sum(pairing_games),
        "games": total_games,
        "wins": total_wins,
        "losses": total_losses,
        "side": "challenger" if learner_challenges else "receiver",
        "error": None,
        "pairings": pairing_rows,
    }


def test_evaluate_frozen_policy_aggregates_by_team_group_over_pairings(monkeypatch) -> None:
    monkeypatch.setattr(train_ppo, "_evaluate_worker", _fake_evaluate_worker)
    varied = [
        OpponentTeamChoice(label=f"team-{index}", packed=f"packed-{index}", group="diverse")
        for index in range(6)
    ]

    evaluation = asyncio.run(
        evaluate_frozen_policy(
            model=None,
            games=50,
            jobs=8,
            learner_team="learner",
            diverse_teams=varied,
            mirror_fraction=0.5,
            seed=3,
            device="cpu",
            pairings_per_worker=4,
        )
    )

    assert evaluation["games"] == 50
    assert evaluation["worker_errors"] == 0
    # by_team_group must be computed over every PAIRING (a worker can hold pairings in
    # both groups once pairings_per_worker > 1), so its totals must equal the sum of
    # every pairing row across every worker, not the sum of one group_row per worker.
    expected_group_games: dict[str, int] = {"mirror": 0, "diverse": 0}
    expected_group_wins: dict[str, int] = {"mirror": 0, "diverse": 0}
    total_pairings = 0
    for row in evaluation["workers"]:
        for pairing_row in row["pairings"]:
            total_pairings += 1
            expected_group_games[pairing_row["team_group"]] += pairing_row["games"]
            expected_group_wins[pairing_row["team_group"]] += pairing_row["wins"]
    # With pairings_per_worker=4 the arm should sample well over 8 distinct pairings.
    assert total_pairings > 8
    for group in ("mirror", "diverse"):
        assert evaluation["by_team_group"][group]["games"] == expected_group_games[group]
        assert evaluation["by_team_group"][group]["wins"] == expected_group_wins[group]
    assert (
        evaluation["by_team_group"]["mirror"]["games"]
        + evaluation["by_team_group"]["diverse"]["games"]
        == evaluation["games"]
    )


def test_load_diverse_opponent_teams_reads_dev_and_sorted_pool(tmp_path) -> None:
    dev = tmp_path / "dev.packed.txt"
    pool = tmp_path / "pool"
    pool.mkdir()
    dev.write_text("dev-team\n")
    (pool / "team_01.packed.txt").write_text("team-one\n")
    (pool / "team_00.packed.txt").write_text("team-zero\n")

    choices = load_diverse_opponent_teams(pool, dev)

    assert [choice.label for choice in choices] == [
        "dev.packed",
        "team_00.packed",
        "team_01.packed",
    ]
    assert [choice.packed for choice in choices] == ["dev-team", "team-zero", "team-one"]


def _teams(count: int) -> list[OpponentTeamChoice]:
    return [
        OpponentTeamChoice(label=f"team-{index}", packed=f"packed-{index}", group="diverse")
        for index in range(count)
    ]


def test_split_holdout_teams_disabled_returns_all_train_no_holdout() -> None:
    teams = _teams(6)

    train_teams, holdout_teams = split_holdout_teams(teams, holdout_fraction=0.0, seed=1)

    assert train_teams == teams
    assert holdout_teams == []


def test_split_holdout_teams_partitions_deterministically_and_disjointly() -> None:
    teams = _teams(10)

    train_teams, holdout_teams = split_holdout_teams(teams, holdout_fraction=0.3, seed=5)
    repeated_train, repeated_holdout = split_holdout_teams(teams, holdout_fraction=0.3, seed=5)

    assert train_teams == repeated_train
    assert holdout_teams == repeated_holdout
    assert set(train_teams).isdisjoint(holdout_teams)
    assert sorted(train_teams + holdout_teams, key=lambda choice: choice.label) == sorted(
        teams, key=lambda choice: choice.label
    )
    assert len(holdout_teams) == round(0.3 * 10)


def test_split_holdout_teams_clamps_to_at_least_one_each_way() -> None:
    teams = _teams(20)

    # A fraction that rounds to 0 (0.5/20) still yields at least one held-out team.
    train_teams, holdout_teams = split_holdout_teams(teams, holdout_fraction=0.02, seed=3)
    assert len(holdout_teams) == 1
    assert len(train_teams) == 19

    # A tiny pool of 2 splits 1 train / 1 held-out even at a large fraction.
    pair = _teams(2)
    train_pair, holdout_pair = split_holdout_teams(pair, holdout_fraction=0.9, seed=3)
    assert len(train_pair) == 1
    assert len(holdout_pair) == 1


def test_split_holdout_teams_rejects_undersized_pool_and_bad_fractions() -> None:
    with pytest.raises(ValueError):
        split_holdout_teams(_teams(1), holdout_fraction=0.5, seed=0)
    with pytest.raises(ValueError):
        split_holdout_teams(_teams(5), holdout_fraction=1.0, seed=0)
    with pytest.raises(ValueError):
        split_holdout_teams(_teams(5), holdout_fraction=-0.1, seed=0)


def test_snapshot_pool_is_bounded_and_round_trips_model(tmp_path) -> None:
    torch.manual_seed(7)
    model = CandidatePolicyValueNet()
    pool = tmp_path / "pool"

    first = save_snapshot(pool, model, generation=0, max_snapshots=2)
    save_snapshot(pool, model, generation=1, max_snapshots=2)
    overwritten = save_snapshot(pool, model, generation=2, max_snapshots=2)

    assert first == overwritten
    assert len(discover_snapshots(pool)) == 2
    loaded = load_snapshot(overwritten)
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_opponent_pool_sampling_keeps_heuristic_anchor_and_snapshot_option(tmp_path) -> None:
    import random

    snapshot = tmp_path / "snapshot_slot_00.pt"
    snapshots = [snapshot]
    heuristic = choose_opponent(snapshots, heuristic_fraction=1.0, rng=random.Random(0))
    historical = choose_opponent(snapshots, heuristic_fraction=0.0, rng=random.Random(0))

    assert heuristic.kind == "heuristic"
    assert historical.kind == "snapshot"
    assert historical.checkpoint_path == snapshot


def test_training_checkpoint_restores_model_optimizer_and_progress(tmp_path) -> None:
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
    path = tmp_path / "latest.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        iteration=3,
        games_seen=48,
        ppo_config=PpoConfig(),
    )
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(10.0)

    iteration, games_seen, restored_config = load_training_checkpoint(
        path, model, optimizer, device="cpu"
    )

    assert (iteration, games_seen) == (3, 48)
    assert restored_config == PpoConfig()
    for key, value in expected.items():
        assert torch.equal(value, model.state_dict()[key])


def test_distillation_split_is_game_disjoint() -> None:
    indices, scalars = _state()
    samples = [
        DistillationSample(
            battle_id=f"battle-{battle}",
            state_indices=indices,
            state_scalars=scalars,
            history_scalars=_history(),
            candidates=_distinct_candidates(),
            teacher_action_index=0,
        )
        for battle in range(5)
        for _turn in range(3)
    ]

    train, val = split_samples_by_battle(samples, val_fraction=0.4, seed=3)

    train_games = {sample.battle_id for sample in train}
    val_games = {sample.battle_id for sample in val}
    assert train_games.isdisjoint(val_games)
    assert len(train_games) == 3
    assert len(val_games) == 2


def test_distillation_improves_teacher_agreement_on_held_out_games() -> None:
    torch.manual_seed(2)
    indices, scalars = _state()
    samples = [
        DistillationSample(
            battle_id=f"battle-{battle}",
            state_indices=indices.copy(),
            state_scalars=scalars.copy(),
            history_scalars=_history(),
            candidates=_distinct_candidates(),
            teacher_action_index=0,
        )
        for battle in range(6)
        for _turn in range(4)
    ]
    train, val = split_samples_by_battle(samples, val_fraction=0.33, seed=0)
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    before = evaluate_agreement(model, val, batch_size=32, device="cpu")

    metrics = distill_policy(
        model,
        optimizer,
        train,
        DistillationConfig(epochs=10, batch_size=32, seed=0),
        device="cpu",
    )
    after = evaluate_agreement(model, val, batch_size=32, device="cpu")

    assert metrics["loss"] >= 0.0
    assert after["accuracy"] > before["accuracy"]
    assert after["accuracy"] == 1.0


# --- Meta-aware state features (vgc.rl.encoding.encode_meta_context) -----------------


def _stub_pokemon(species: str, *, moves=None, **extra) -> SimpleNamespace:
    return SimpleNamespace(species=species, moves=moves or {}, fainted=False, **extra)


def _stub_battle(
    opponent_species: list[str],
    our_species: list[str],
    *,
    active_opponents=None,
    teampreview: bool = True,
) -> SimpleNamespace:
    opponent_pokemon = [_stub_pokemon(species) for species in opponent_species]
    return SimpleNamespace(
        teampreview_opponent_team=opponent_pokemon if teampreview else [],
        opponent_team={pokemon.species: pokemon for pokemon in opponent_pokemon},
        team={f"our{i}": _stub_pokemon(species) for i, species in enumerate(our_species)},
        opponent_active_pokemon=active_opponents or [None, None],
    )


def test_encode_meta_context_returns_well_formed_vector_on_minimal_battle() -> None:
    battle = _stub_battle(["pikachu", "eevee"], ["snorlax", "gyarados"])
    vector = encode_meta_context(battle)
    assert vector.shape == (META_SCALAR_DIM,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))


def test_encode_meta_context_returns_unknown_vector_on_empty_battle() -> None:
    vector = encode_meta_context(SimpleNamespace())
    assert vector.shape == (META_SCALAR_DIM,)
    assert vector.dtype == np.float32
    # Both archetype blocks fall back to their trailing "unknown" slot.
    assert vector[META_ARCHETYPE_SLOTS - 1] == 1.0
    assert vector[2 * META_ARCHETYPE_SLOTS - 1] == 1.0
    assert np.all(vector[2 * META_ARCHETYPE_SLOTS :] == 0.0)


def test_encode_meta_context_never_raises_on_malformed_battle() -> None:
    # Missing/wrong-shaped attributes must fall back to the zero/unknown vector, not
    # propagate an exception into a live battle.
    battle = SimpleNamespace(team=None, opponent_team=None, teampreview_opponent_team=None)
    vector = encode_meta_context(battle)
    assert vector.shape == (META_SCALAR_DIM,)


def test_encode_meta_context_flags_opponent_archetype_from_previewed_core_species() -> None:
    archetype = ARCHETYPES[0]
    battle = _stub_battle(list(archetype.core_species), ["pikachu"])
    vector = encode_meta_context(battle)
    expected_index = [entry.label for entry in ARCHETYPES].index(archetype.label)
    assert vector[expected_index] == 1.0
    assert vector[META_ARCHETYPE_SLOTS - 1] == 0.0


def test_encode_meta_context_ignores_ground_truth_hidden_information() -> None:
    """Two battles identical in revealed/preview species but differing only in a
    hidden field (e.g. a true Stat Point spread a real ladder opponent would never
    reveal) must produce IDENTICAL meta vectors -- this is the deployment-available
    guarantee the whole feature depends on.
    """

    opponent_species = ["charizard", "garchomp"]
    our_species = ["pelipper", "archaludon"]
    battle_a = _stub_battle(opponent_species, our_species)
    battle_b = _stub_battle(opponent_species, our_species)
    # Attach a hidden ground-truth field encode_meta_context must never read.
    battle_a.hidden_sp_spread = {"hp": 31, "atk": 0, "spe": 32}
    battle_b.hidden_sp_spread = {"hp": 0, "atk": 32, "spe": 4}

    vector_a = encode_meta_context(battle_a)
    vector_b = encode_meta_context(battle_b)
    np.testing.assert_array_equal(vector_a, vector_b)


# --- Meta-features model plumbing ----------------------------------------------------


def test_meta_features_off_model_architecture_is_unchanged() -> None:
    model = CandidatePolicyValueNet()
    assert model.use_meta_features is False
    assert not hasattr(model, "meta_encoder")
    first_linear = model.context_encoder[0]
    assert first_linear.in_features == HIDDEN_DIM + HISTORY_HIDDEN_DIM

    indices, scalars = _state()
    candidates = _candidate_features(2)
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    logits, values = model(
        torch.as_tensor(indices[None, :]),
        torch.as_tensor(scalars[None, :]),
        torch.as_tensor(_history()[None, :]),
        torch.as_tensor(moves),
        torch.as_tensor(targets),
        torch.as_tensor(species),
        torch.as_tensor(flags),
        torch.as_tensor(mask),
    )
    assert logits.shape == (1, 2)
    assert values.shape == (1,)


def test_meta_features_on_model_requires_meta_scalars_and_widens_context_encoder() -> None:
    torch.manual_seed(0)
    model = CandidatePolicyValueNet(use_meta_features=True)
    assert hasattr(model, "meta_encoder")
    first_linear = model.context_encoder[0]
    assert first_linear.in_features == HIDDEN_DIM + HISTORY_HIDDEN_DIM + META_HIDDEN_DIM

    indices, scalars = _state()
    candidates = _candidate_features(2)
    moves, targets, species, flags, mask = pad_candidate_features([candidates])
    common = dict(
        state_indices=torch.as_tensor(indices[None, :]),
        state_scalars=torch.as_tensor(scalars[None, :]),
        history_scalars=torch.as_tensor(_history()[None, :]),
        move_indices=torch.as_tensor(moves),
        target_indices=torch.as_tensor(targets),
        switch_species_indices=torch.as_tensor(species),
        action_flags=torch.as_tensor(flags),
        candidate_mask=torch.as_tensor(mask),
    )
    with pytest.raises(ValueError, match="meta_scalars"):
        model(**common)

    meta_scalars = torch.zeros((1, META_SCALAR_DIM), dtype=torch.float32)
    logits, values = model(**common, meta_scalars=meta_scalars)
    assert logits.shape == (1, 2)
    assert values.shape == (1,)


def test_warm_start_state_encoder_loads_same_tensor_count_regardless_of_meta_features(
    tmp_path,
) -> None:
    source = CandidatePolicyValueNet()
    checkpoint_path = tmp_path / "bc.pt"
    torch.save({"model_state_dict": source.state_encoder.state_dict()}, checkpoint_path)

    off_result = CandidatePolicyValueNet(use_meta_features=False).warm_start_state_encoder(
        checkpoint_path
    )
    on_result = CandidatePolicyValueNet(use_meta_features=True).warm_start_state_encoder(
        checkpoint_path
    )
    assert off_result == on_result
    assert off_result["loaded"] == off_result["available"] > 0


def test_snapshot_round_trip_preserves_meta_features_flag(tmp_path) -> None:
    on_model = CandidatePolicyValueNet(use_meta_features=True)
    on_path = save_snapshot(tmp_path / "on", on_model, generation=0, max_snapshots=2)
    assert load_snapshot(on_path).use_meta_features is True

    off_model = CandidatePolicyValueNet(use_meta_features=False)
    off_path = save_snapshot(tmp_path / "off", off_model, generation=0, max_snapshots=2)
    assert load_snapshot(off_path).use_meta_features is False


def test_training_checkpoint_rejects_meta_features_mismatch(tmp_path) -> None:
    model_on = CandidatePolicyValueNet(use_meta_features=True)
    optimizer_on = torch.optim.Adam(model_on.parameters(), lr=1e-3)
    path = tmp_path / "meta_on.pt"
    save_checkpoint(path, model_on, optimizer_on, iteration=1, games_seen=4, ppo_config=PpoConfig())

    mismatched = CandidatePolicyValueNet(use_meta_features=False)
    mismatched_optimizer = torch.optim.Adam(mismatched.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="use_meta_features"):
        load_training_checkpoint(path, mismatched, mismatched_optimizer, device="cpu")

    matching = CandidatePolicyValueNet(use_meta_features=True)
    matching_optimizer = torch.optim.Adam(matching.parameters(), lr=1e-3)
    iteration, games_seen, _ = load_training_checkpoint(
        path, matching, matching_optimizer, device="cpu"
    )
    assert (iteration, games_seen) == (1, 4)


# --- Dense potential-based reward shaping (vgc.rl.rewards.board_potential) -----------


def test_board_potential_is_near_zero_when_both_sides_are_full_hp() -> None:
    battle = _stub_battle(["snorlax", "gyarados"], ["pikachu", "eevee"])
    assert board_potential(battle) == pytest.approx(0.0)


def _fainted_pokemon(species: str) -> SimpleNamespace:
    return SimpleNamespace(species=species, moves={}, fainted=True)


def test_board_potential_is_positive_when_we_are_ahead() -> None:
    opponent = [_fainted_pokemon("snorlax"), _stub_pokemon("gyarados")]
    battle = SimpleNamespace(
        teampreview_opponent_team=opponent,
        opponent_team={pokemon.species: pokemon for pokemon in opponent},
        team={"our0": _stub_pokemon("pikachu"), "our1": _stub_pokemon("eevee")},
        opponent_active_pokemon=[None, None],
    )
    assert board_potential(battle) > 0.0


def test_board_potential_is_negative_when_we_are_losing() -> None:
    ours = [_fainted_pokemon("pikachu"), _stub_pokemon("eevee")]
    battle = SimpleNamespace(
        teampreview_opponent_team=[_stub_pokemon("snorlax"), _stub_pokemon("gyarados")],
        opponent_team={
            "snorlax": _stub_pokemon("snorlax"),
            "gyarados": _stub_pokemon("gyarados"),
        },
        team={pokemon.species: pokemon for pokemon in ours},
        opponent_active_pokemon=[None, None],
    )
    assert board_potential(battle) < 0.0


def test_board_potential_treats_unrevealed_opponents_as_full_hp() -> None:
    # Team preview saw 3 opponents; only 1 has been sent out and damaged so far. The
    # other 2 must count as full HP, not be omitted or treated as 0 HP.
    revealed = _stub_pokemon("snorlax", current_hp_fraction=0.5)
    battle = SimpleNamespace(
        teampreview_opponent_team=[
            revealed,
            _stub_pokemon("gyarados"),
            _stub_pokemon("dragonite"),
        ],
        opponent_team={"snorlax": revealed},
        team={"our0": _stub_pokemon("pikachu")},
        opponent_active_pokemon=[None, None],
    )
    # our=1.0, opp mean = (0.5 + 1.0 + 1.0) / 3 = 0.8333..., scaled by the HP term's
    # weight in the weighted-sum potential (see vgc.rl.rewards._HP_WEIGHT) -- no
    # strategic state is present here (no side conditions/fields/boosts on the stub),
    # so the HP term is the only nonzero component.
    from vgc.rl.rewards import _HP_WEIGHT

    assert board_potential(battle) == pytest.approx(_HP_WEIGHT * (1.0 - (0.5 + 1.0 + 1.0) / 3))


def test_board_potential_never_raises_on_empty_or_malformed_teams() -> None:
    assert board_potential(SimpleNamespace()) == 0.0
    assert board_potential(SimpleNamespace(team=None, opponent_team=None)) == 0.0
    assert board_potential(SimpleNamespace(team="not-a-dict", opponent_team=123)) == 0.0
    assert board_potential(None) == 0.0


def test_board_potential_is_bounded() -> None:
    ours = {"our0": _fainted_pokemon("pikachu")}
    opponent = [_stub_pokemon("snorlax")]
    battle = SimpleNamespace(
        teampreview_opponent_team=opponent,
        opponent_team={pokemon.species: pokemon for pokemon in opponent},
        team=ours,
        opponent_active_pokemon=[None, None],
    )
    potential = board_potential(battle)
    assert -1.0 <= potential <= 1.0


# --- Strategic potential terms (vgc.rl.rewards._potential_components) ---------------


def _active_stub(
    species: str,
    *,
    boosts: dict[str, int] | None = None,
    evs: list[int] | None = None,
    nature: str | None = None,
    current_hp_fraction: float = 1.0,
) -> SimpleNamespace:
    """A Pokemon-shaped stub with everything `_our_pokemon_state`/`opponent_state`
    (used by the boost and Trick Room components) need to read: species, boosts,
    status/item/ability, and (for our own side) evs/nature."""

    return SimpleNamespace(
        species=species,
        moves={},
        fainted=False,
        boosts=boosts or {},
        status=None,
        item=None,
        ability=None,
        evs=evs,
        nature=nature,
        current_hp=100,
        current_hp_fraction=current_hp_fraction,
    )


def _flat_board(*, side_conditions=None, opponent_side_conditions=None, fields=None) -> SimpleNamespace:
    """An even-HP board (no HP-term contribution) with strategic state layered in, so
    each strategic component's test isolates just that term."""

    return SimpleNamespace(
        teampreview_opponent_team=[_stub_pokemon("gyarados")],
        opponent_team={"gyarados": _stub_pokemon("gyarados")},
        team={"our0": _stub_pokemon("snorlax")},
        active_pokemon=[_active_stub("snorlax")],
        opponent_active_pokemon=[_active_stub("gyarados")],
        side_conditions=side_conditions or {},
        opponent_side_conditions=opponent_side_conditions or {},
        fields=fields or {},
    )


def test_tailwind_on_our_side_raises_potential_and_on_theirs_lowers_it() -> None:
    neutral = board_potential(_flat_board())
    ours = board_potential(_flat_board(side_conditions={SideCondition.TAILWIND: 1}))
    theirs = board_potential(_flat_board(opponent_side_conditions={SideCondition.TAILWIND: 1}))
    both = board_potential(
        _flat_board(
            side_conditions={SideCondition.TAILWIND: 1},
            opponent_side_conditions={SideCondition.TAILWIND: 1},
        )
    )
    assert ours > neutral
    assert theirs < neutral
    assert both == pytest.approx(neutral)


def test_screens_on_our_side_raise_potential_and_on_theirs_lower_it() -> None:
    neutral = board_potential(_flat_board())
    ours = board_potential(_flat_board(side_conditions={SideCondition.REFLECT: 1}))
    theirs = board_potential(_flat_board(opponent_side_conditions={SideCondition.LIGHT_SCREEN: 1}))
    assert ours > neutral
    assert theirs < neutral


def test_our_active_boost_raises_potential_and_opponents_lowers_it() -> None:
    neutral = board_potential(_flat_board())

    ours_boosted = _flat_board()
    ours_boosted.active_pokemon = [_active_stub("snorlax", boosts={"atk": 2})]
    assert board_potential(ours_boosted) > neutral

    theirs_boosted = _flat_board()
    theirs_boosted.opponent_active_pokemon = [_active_stub("gyarados", boosts={"atk": 2})]
    assert board_potential(theirs_boosted) < neutral


def test_trick_room_sign_depends_on_who_it_favors() -> None:
    # Snorlax (base Speed 30) is much slower than Pikachu (base Speed 90), so with
    # Trick Room up, "we" being Snorlax vs their Pikachu is a favorable Trick Room
    # (we're slower); "we" being Pikachu vs their Snorlax is unfavorable.
    slower_side_battle = SimpleNamespace(
        teampreview_opponent_team=[_stub_pokemon("pikachu")],
        opponent_team={"pikachu": _stub_pokemon("pikachu")},
        team={"our0": _stub_pokemon("snorlax")},
        active_pokemon=[_active_stub("snorlax")],
        opponent_active_pokemon=[_active_stub("pikachu")],
        side_conditions={},
        opponent_side_conditions={},
        fields={Field.TRICK_ROOM: 1},
    )
    faster_side_battle = SimpleNamespace(
        teampreview_opponent_team=[_stub_pokemon("snorlax")],
        opponent_team={"snorlax": _stub_pokemon("snorlax")},
        team={"our0": _stub_pokemon("pikachu")},
        active_pokemon=[_active_stub("pikachu")],
        opponent_active_pokemon=[_active_stub("snorlax")],
        side_conditions={},
        opponent_side_conditions={},
        fields={Field.TRICK_ROOM: 1},
    )
    no_trick_room = SimpleNamespace(
        teampreview_opponent_team=[_stub_pokemon("pikachu")],
        opponent_team={"pikachu": _stub_pokemon("pikachu")},
        team={"our0": _stub_pokemon("snorlax")},
        active_pokemon=[_active_stub("snorlax")],
        opponent_active_pokemon=[_active_stub("pikachu")],
        side_conditions={},
        opponent_side_conditions={},
        fields={},
    )

    from vgc.rl.rewards import _potential_components

    slower_components = _potential_components(slower_side_battle)
    faster_components = _potential_components(faster_side_battle)
    off_components = _potential_components(no_trick_room)

    assert slower_components["trick_room"] > 0.0
    assert faster_components["trick_room"] < 0.0
    assert off_components["trick_room"] == 0.0


def test_trick_room_term_is_zero_when_speed_cannot_be_computed() -> None:
    from vgc.rl.rewards import _potential_components

    battle = SimpleNamespace(
        teampreview_opponent_team=[],
        opponent_team={},
        team={},
        active_pokemon=[None],
        opponent_active_pokemon=[None],
        side_conditions={},
        opponent_side_conditions={},
        fields={Field.TRICK_ROOM: 1},
    )
    assert _potential_components(battle)["trick_room"] == 0.0


def test_board_potential_clamps_when_every_strategic_term_is_favorable() -> None:
    # Boost only atk/def/spa/spd (not spe) so the boost term doesn't flip our Speed
    # past the opponent's and invert the Trick Room term's sign -- this board is
    # constructed so every one of the five components pushes positive.
    battle = SimpleNamespace(
        teampreview_opponent_team=[_fainted_pokemon("pikachu")],
        opponent_team={"pikachu": _fainted_pokemon("pikachu")},
        team={"our0": _stub_pokemon("snorlax")},
        active_pokemon=[_active_stub("snorlax", boosts={"atk": 6, "def": 6, "spa": 6, "spd": 6})],
        opponent_active_pokemon=[_active_stub("pikachu")],
        side_conditions={
            SideCondition.TAILWIND: 1,
            SideCondition.REFLECT: 1,
            SideCondition.LIGHT_SCREEN: 1,
            SideCondition.AURORA_VEIL: 1,
        },
        opponent_side_conditions={},
        fields={Field.TRICK_ROOM: 1},
    )
    potential = board_potential(battle)
    assert potential > 0.9
    assert potential <= 1.0


def test_board_potential_clamps_when_every_strategic_term_is_unfavorable() -> None:
    battle = SimpleNamespace(
        teampreview_opponent_team=[_stub_pokemon("pikachu")],
        opponent_team={"pikachu": _stub_pokemon("pikachu")},
        team={"our0": _fainted_pokemon("snorlax")},
        active_pokemon=[_active_stub("pikachu")],
        opponent_active_pokemon=[
            _active_stub("snorlax", boosts={"atk": 6, "def": 6, "spa": 6, "spd": 6})
        ],
        side_conditions={},
        opponent_side_conditions={
            SideCondition.TAILWIND: 1,
            SideCondition.REFLECT: 1,
            SideCondition.LIGHT_SCREEN: 1,
            SideCondition.AURORA_VEIL: 1,
        },
        fields={Field.TRICK_ROOM: 1},
    )
    potential = board_potential(battle)
    assert potential < -0.9
    assert potential >= -1.0


def _shaped_step(potential: float) -> RolloutStep:
    indices, scalars = _state()
    return RolloutStep(
        state_indices=indices,
        state_scalars=scalars,
        history_scalars=_history(),
        candidates=_candidate_features(2),
        action_index=0,
        old_log_prob=-0.69,
        old_value=0.0,
        state_potential=potential,
    )


def test_finish_episode_with_zero_coef_is_unchanged_from_sparse_only_behavior() -> None:
    buffer = RolloutBuffer()
    for potential in (0.1, 0.4, -0.2):
        buffer.add(_shaped_step(potential))
    # reward_shaping_coef defaults to 0.0 -- the shaping loop must never run, and
    # rewards/returns must match the pre-shaping sparse-only behavior exactly.
    buffer.finish_episode(1.0, PpoConfig(gamma=1.0, gae_lambda=1.0), terminal_potential=0.9)
    assert [step.reward for step in buffer.steps] == [0.0, 0.0, 1.0]
    assert [step.return_value for step in buffer.steps] == pytest.approx([1.0, 1.0, 1.0])


def test_finish_episode_shapes_rewards_and_still_applies_terminal_outcome() -> None:
    buffer = RolloutBuffer()
    potentials = [0.0, 0.5, 0.8]
    for potential in potentials:
        buffer.add(_shaped_step(potential))
    coef = 0.2
    gamma = 1.0
    terminal_potential = 1.0
    buffer.finish_episode(
        1.0,
        PpoConfig(gamma=gamma, gae_lambda=1.0, reward_shaping_coef=coef),
        terminal_potential=terminal_potential,
    )
    expected_next = [potentials[1], potentials[2], terminal_potential]
    expected_rewards = [
        coef * (gamma * expected_next[i] - potentials[i]) for i in range(3)
    ]
    expected_rewards[-1] += 1.0
    assert [step.reward for step in buffer.steps] == pytest.approx(expected_rewards)
    assert buffer.steps[-1].done is True


def test_finish_episode_shaping_telescopes_to_near_zero_net_when_potential_is_flat() -> None:
    # A flat potential across the whole episode (no board change) must add ~0 net
    # shaping beyond the terminal step's own (gamma*terminal - flat) term.
    buffer = RolloutBuffer()
    flat = 0.3
    for _ in range(3):
        buffer.add(_shaped_step(flat))
    buffer.finish_episode(
        0.0,
        PpoConfig(gamma=1.0, gae_lambda=1.0, reward_shaping_coef=0.5),
        terminal_potential=flat,
    )
    # Every non-terminal step's shaping term is coef*(gamma*flat - flat) == 0 when
    # gamma == 1.0; the terminal step's is also 0 since terminal_potential == flat too.
    assert [step.reward for step in buffer.steps] == pytest.approx([0.0, 0.0, 0.0])


def test_ppo_config_round_trips_reward_shaping_coef_through_asdict() -> None:
    from dataclasses import asdict

    config = PpoConfig(reward_shaping_coef=0.35)
    rebuilt = PpoConfig(**asdict(config))
    assert rebuilt.reward_shaping_coef == pytest.approx(0.35)
    assert rebuilt == config


# --- Teacher-anchor annealing (selfplay.train_ppo.teacher_anchor_weight_for_iteration)


def test_teacher_anchor_anneal_hits_start_at_first_and_final_at_last_iteration() -> None:
    weights = [
        teacher_anchor_weight_for_iteration(
            iteration, total_iterations=5, start=0.2, final=0.0
        )
        for iteration in range(1, 6)
    ]
    assert weights[0] == pytest.approx(0.2)
    assert weights[-1] == pytest.approx(0.0)
    assert weights == sorted(weights, reverse=True)


def test_teacher_anchor_anneal_is_constant_when_final_is_none() -> None:
    for iteration in range(1, 6):
        assert teacher_anchor_weight_for_iteration(
            iteration, total_iterations=5, start=0.2, final=None
        ) == pytest.approx(0.2)


def test_teacher_anchor_anneal_handles_single_iteration_run() -> None:
    assert teacher_anchor_weight_for_iteration(
        1, total_iterations=1, start=0.2, final=0.0
    ) == pytest.approx(0.2)


def test_teacher_anchor_anneal_handles_equal_start_and_final() -> None:
    for iteration in range(1, 4):
        assert teacher_anchor_weight_for_iteration(
            iteration, total_iterations=3, start=0.1, final=0.1
        ) == pytest.approx(0.1)


def test_teacher_anchor_anneal_supports_increasing_schedule() -> None:
    weights = [
        teacher_anchor_weight_for_iteration(
            iteration, total_iterations=4, start=0.0, final=0.3
        )
        for iteration in range(1, 5)
    ]
    assert weights == sorted(weights)
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] == pytest.approx(0.3)


def test_main_rejects_negative_teacher_anchor_final_weight(monkeypatch) -> None:
    import selfplay.train_ppo as train_ppo_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ppo.py",
            "--iterations",
            "1",
            "--eval-games",
            "0",
            "--teacher-anchor-final-weight",
            "-0.1",
        ],
    )
    with pytest.raises(SystemExit):
        train_ppo_module.main()


def test_main_rejects_negative_reward_shaping_coef(monkeypatch) -> None:
    import selfplay.train_ppo as train_ppo_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ppo.py",
            "--iterations",
            "1",
            "--eval-games",
            "0",
            "--reward-shaping-coef",
            "-0.1",
        ],
    )
    with pytest.raises(SystemExit):
        train_ppo_module.main()


def test_main_rejects_negative_bootstrap_early_stopping_patience(monkeypatch) -> None:
    import selfplay.train_ppo as train_ppo_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ppo.py",
            "--iterations",
            "1",
            "--eval-games",
            "0",
            "--bootstrap-early-stopping-patience",
            "-1",
        ],
    )
    with pytest.raises(SystemExit):
        train_ppo_module.main()


@pytest.mark.parametrize("bad_value", ["-0.1", "1.0", "1.5"])
def test_main_rejects_out_of_range_head_dropout(monkeypatch, bad_value: str) -> None:
    import selfplay.train_ppo as train_ppo_module

    monkeypatch.setattr(
        "sys.argv",
        [
            "train_ppo.py",
            "--iterations",
            "1",
            "--eval-games",
            "0",
            "--head-dropout",
            bad_value,
        ],
    )
    with pytest.raises(SystemExit):
        train_ppo_module.main()


# --- head_dropout (action-scoring path regularization, see CandidatePolicyValueNet) --


def _forward_once(model: "CandidatePolicyValueNet") -> tuple[torch.Tensor, torch.Tensor]:
    state_indices, state_scalars = _state()
    moves, targets, species, flags, mask = pad_candidate_features([_distinct_candidates()])
    return model(
        torch.as_tensor(state_indices[None, :]),
        torch.as_tensor(state_scalars[None, :]),
        torch.as_tensor(_history()[None, :]),
        torch.as_tensor(moves),
        torch.as_tensor(targets),
        torch.as_tensor(species),
        torch.as_tensor(flags),
        torch.as_tensor(mask),
    )


def test_head_dropout_does_not_change_state_dict_keys() -> None:
    base = CandidatePolicyValueNet()
    with_head_dropout = CandidatePolicyValueNet(head_dropout=0.5)
    assert set(base.state_dict().keys()) == set(with_head_dropout.state_dict().keys())


def test_head_dropout_is_inactive_in_eval_mode() -> None:
    torch.manual_seed(11)
    model = CandidatePolicyValueNet(head_dropout=0.0)
    model.eval()
    logits_zero, values_zero = _forward_once(model)

    # Same weights, only head_dropout_p changes -- eval mode must make this a no-op.
    model.head_dropout_p = 0.5
    logits_half, values_half = _forward_once(model)

    assert torch.equal(logits_zero, logits_half)
    assert torch.equal(values_zero, values_half)


def test_head_dropout_is_active_in_train_mode_but_zero_stays_deterministic() -> None:
    torch.manual_seed(12)
    model = CandidatePolicyValueNet(head_dropout=0.5)
    model.train()
    torch.manual_seed(100)
    logits_a, _values_a = _forward_once(model)
    torch.manual_seed(200)
    logits_b, _values_b = _forward_once(model)
    assert not torch.equal(logits_a, logits_b)

    model.head_dropout_p = 0.0
    torch.manual_seed(100)
    logits_c, _values_c = _forward_once(model)
    torch.manual_seed(200)
    logits_d, _values_d = _forward_once(model)
    assert torch.equal(logits_c, logits_d)


# --- distill_policy early stopping / best-epoch selection ---------------------------


def _val_samples_for_early_stopping() -> list[DistillationSample]:
    indices, scalars = _state()
    return [
        DistillationSample(
            battle_id=f"battle-{battle}",
            state_indices=indices.copy(),
            state_scalars=scalars.copy(),
            history_scalars=_history(),
            candidates=_distinct_candidates(),
            teacher_action_index=0,
        )
        for battle in range(4)
        for _turn in range(2)
    ]


def test_distill_policy_restores_best_epoch_weights_and_reports_val_history(monkeypatch) -> None:
    import vgc.rl.distill as distill_module

    torch.manual_seed(5)
    samples = _val_samples_for_early_stopping()
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scripted_accuracies = [0.4, 0.9, 0.5, 0.5]
    snapshots: list[dict[str, torch.Tensor]] = []

    def fake_evaluate_agreement(m, _samples, *, batch_size, device):
        snapshots.append({key: value.detach().clone() for key, value in m.state_dict().items()})
        accuracy = scripted_accuracies[len(snapshots) - 1]
        return {
            "accuracy": accuracy,
            "teacher_probability": 0.5,
            "loss": 1.0,
            "top3_accuracy": 1.0,
            "entropy": 0.1,
            "samples": float(len(samples)),
        }

    monkeypatch.setattr(distill_module, "evaluate_agreement", fake_evaluate_agreement)

    config = DistillationConfig(epochs=4, batch_size=8, seed=0)
    metrics = distill_module.distill_policy(
        model, optimizer, samples, config, device="cpu", val_samples=samples
    )

    assert metrics["best_epoch"] == 2
    assert metrics["best_val_accuracy"] == pytest.approx(0.9)
    assert metrics["epochs_run"] == 4
    assert len(metrics["val_history"]) == 4
    assert [entry["epoch"] for entry in metrics["val_history"]] == [1.0, 2.0, 3.0, 4.0]

    best_snapshot = snapshots[1]  # epoch 2, 0-indexed
    for key, value in model.state_dict().items():
        assert torch.equal(value, best_snapshot[key])


def test_distill_policy_early_stopping_patience_stops_before_all_epochs(monkeypatch) -> None:
    import vgc.rl.distill as distill_module

    torch.manual_seed(6)
    samples = _val_samples_for_early_stopping()
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scripted_accuracies = [0.6, 0.9, 0.85, 0.8, 0.5]
    call_count = {"n": 0}

    def fake_evaluate_agreement(m, _samples, *, batch_size, device):
        accuracy = scripted_accuracies[call_count["n"]]
        call_count["n"] += 1
        return {
            "accuracy": accuracy,
            "teacher_probability": 0.5,
            "loss": 1.0,
            "top3_accuracy": 1.0,
            "entropy": 0.1,
            "samples": float(len(samples)),
        }

    monkeypatch.setattr(distill_module, "evaluate_agreement", fake_evaluate_agreement)

    config = DistillationConfig(epochs=5, batch_size=8, seed=0, early_stopping_patience=2)
    metrics = distill_module.distill_policy(
        model, optimizer, samples, config, device="cpu", val_samples=samples
    )

    assert metrics["best_epoch"] == 2
    assert metrics["best_val_accuracy"] == pytest.approx(0.9)
    assert metrics["epochs_run"] == 4
    assert metrics["epochs_run"] < config.epochs
    assert len(metrics["val_history"]) == 4


def test_distill_policy_without_val_samples_runs_full_epochs_and_keeps_existing_keys() -> None:
    torch.manual_seed(7)
    samples = _val_samples_for_early_stopping()
    train, val = split_samples_by_battle(samples, val_fraction=0.5, seed=0)
    model = CandidatePolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    config = DistillationConfig(epochs=3, batch_size=8, seed=0)
    metrics = distill_policy(model, optimizer, train, config, device="cpu")

    assert metrics["epochs"] == 3.0
    assert "samples" in metrics and "loss" in metrics and "grad_norm" in metrics
    assert "best_epoch" not in metrics
    assert "val_history" not in metrics
    # val is unused here (val_samples not passed) -- keep the split alive only so
    # this test documents the "legacy, no honest gating" call shape it's guarding.
    assert val
