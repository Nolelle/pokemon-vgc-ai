from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from offline.evaluate_counterfactual_value import (  # noqa: E402
    DEFAULT_MANIFEST,
    DEFAULT_Q_TEAM_SPLIT,
    _select_q_development_teams,
    main as counterfactual_main,
)
from selfplay.train_full_pipeline import load_team_pool  # noqa: E402
from vgc.bc.encoding import (  # noqa: E402
    INDEX_DIM,
    MOVE_TO_IDX,
    SLOT_FEATURE_DIM,
    SPECIES_TO_IDX,
    STATE_SCALAR_DIM,
    TARGET_TO_IDX,
)
from vgc.rl.counterfactual_q import (  # noqa: E402
    ActionResponseValueNet,
    CounterfactualQInput,
    CounterfactualQSample,
    QTrainingConfig,
    evaluate_q_model,
    load_q_dataset,
    save_q_dataset,
    split_q_samples_team_disjoint,
    train_q_model,
)
from vgc.rl.encoding import (  # noqa: E402
    HISTORY_SCALAR_DIM,
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    META_SCALAR_DIM,
    TACTICAL_FEATURE_DIM,
    CandidateFeatures,
    InformationFeatures,
)
from vgc.rl.model import CandidatePolicyValueNet  # noqa: E402
from vgc.rl.mechanics_encoding import MechanicsFeatures  # noqa: E402


def _action(move: str) -> CandidateFeatures:
    return CandidateFeatures(
        move_indices=np.asarray([[MOVE_TO_IDX[move], MOVE_TO_IDX["protect"]]], dtype=np.int64),
        target_indices=np.asarray(
            [[TARGET_TO_IDX["opp0"], TARGET_TO_IDX["self_or_field"]]],
            dtype=np.int64,
        ),
        switch_species_indices=np.full((1, 2), SPECIES_TO_IDX["<pad>"], dtype=np.int64),
        flags=np.zeros((1, 2, 4), dtype=np.float32),
        tactical=np.zeros((1, TACTICAL_FEATURE_DIM), dtype=np.float32),
    )


def _sample(
    root: str,
    team: str,
    opponent_team: str,
    own_action: str,
    response: str,
    outcome: float,
) -> CounterfactualQSample:
    encoded = CounterfactualQInput(
        state_indices=np.zeros(INDEX_DIM, dtype=np.int64),
        state_scalars=np.zeros(STATE_SCALAR_DIM + SLOT_FEATURE_DIM, dtype=np.float32),
        history_scalars=np.zeros(HISTORY_SCALAR_DIM, dtype=np.float32),
        own_action=_action(own_action),
        opponent_action=_action(response),
        meta_scalars=np.zeros(META_SCALAR_DIM, dtype=np.float32),
        information=InformationFeatures(
            indices=np.zeros(INFORMATION_INDEX_DIM, dtype=np.int64),
            scalars=np.zeros(INFORMATION_SCALAR_DIM, dtype=np.float32),
        ),
        mechanics=MechanicsFeatures(
            tokens=np.frombuffer(b"{}", dtype=np.uint8).astype(np.int64) + 1
        ),
    )
    return CounterfactualQSample(
        root_id=root,
        battle_id=f"battle-{root}",
        team_id=team,
        opponent_team_id=opponent_team,
        own_action=own_action,
        opponent_action=response,
        encoded=encoded,
        mean_outcome=outcome,
        outcome_variance=0.25,
        rollout_count=4,
        search_score=1.0,
        opponent_search_score=1.0,
        opponent_weight=0.5,
    )


def _root_samples(root: str, team: str, opponent_team: str):
    rows = []
    for response in ("heatwave", "protect"):
        rows.extend(
            (
                _sample(root, team, opponent_team, "dragonclaw", response, 0.75),
                _sample(root, team, opponent_team, "protect", response, -0.25),
            )
        )
    return rows


def test_q_dataset_round_trip_keeps_action_response_targets(tmp_path) -> None:
    samples = _root_samples("r1", "sun/a", "rain/a")
    path = tmp_path / "q.pt"
    save_q_dataset(path, samples, metadata={"team_partition": "development"})

    loaded, metadata = load_q_dataset(path)

    assert len(loaded) == 4
    assert loaded[0].rollout_count == 4
    assert loaded[0].encoded.opponent_action.move_indices.shape == (1, 2)
    assert metadata["team_partition"] == "development"


def test_q_split_keeps_both_sides_teams_disjoint() -> None:
    samples = [
        *_root_samples("r1", "sun/a", "rain/a"),
        *_root_samples("r2", "sun/b", "rain/b"),
        *_root_samples("r3", "sun/c", "rain/c"),
    ]

    train, validation = split_q_samples_team_disjoint(samples, validation_fraction=1 / 3, seed=7)

    train_teams = {team for sample in train for team in (sample.team_id, sample.opponent_team_id)}
    validation_teams = {
        team for sample in validation for team in (sample.team_id, sample.opponent_team_id)
    }
    assert train_teams.isdisjoint(validation_teams)
    assert {sample.root_id for sample in train}.isdisjoint(
        {sample.root_id for sample in validation}
    )


def test_action_response_model_is_bounded_and_trains_only_q_head_by_default() -> None:
    train = [
        *_root_samples("r1", "sun/a", "rain/a"),
        *_root_samples("r2", "sun/b", "rain/b"),
    ]
    validation = _root_samples("r3", "sun/c", "rain/c")
    torch.manual_seed(3)
    model = ActionResponseValueNet(
        CandidatePolicyValueNet(
            use_meta_features=True,
            use_information_features=True,
            use_tactical_features=True,
        )
    )
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    result = train_q_model(
        model,
        train,
        validation,
        QTrainingConfig(epochs=2, patience=2, seed=3),
        device="cpu",
    )
    metrics = evaluate_q_model(model, validation, batch_size=8, device="cpu")

    assert result["best_epoch"] >= 1
    assert -1.0 <= metrics["prediction_min"] <= metrics["prediction_max"] <= 1.0
    assert metrics["opponent_response_conditioned"]["pairwise_comparisons"] == 2
    assert metrics["search_baseline"]["action_aggregated"]["groups"] == 1
    assert set(metrics["improvement_over_search"]) == {
        "pairwise_accuracy",
        "top_action_accuracy",
        "regret_reduction",
    }
    assert any(
        not torch.equal(value, before[key])
        for key, value in model.state_dict().items()
        if key.startswith("q_head.")
    )
    assert all(
        torch.equal(value, before[key])
        for key, value in model.state_dict().items()
        if key.startswith("backbone.")
    )


def test_q_export_refuses_the_confirmation_partition(tmp_path) -> None:
    with pytest.raises(SystemExit, match="held-out teams are confirmation-only"):
        counterfactual_main(["--q-dataset", str(tmp_path / "unsafe.pt")])


def test_frozen_q_split_declares_129_development_and_31_confirmation_teams() -> None:
    split = json.loads(DEFAULT_Q_TEAM_SPLIT.read_text())
    all_labels = {team.label for team in load_team_pool(DEFAULT_MANIFEST)}
    development_excluded = set(split["development_excluded"])
    confirmation_manifest_path = (
        DEFAULT_Q_TEAM_SPLIT.parents[2] / split["confirmation_source_manifest"]
    )
    confirmation_manifest = json.loads(confirmation_manifest_path.read_text())
    rows_by_file = {row["file"]: row for row in confirmation_manifest}
    confirmation_files = split["confirmation_files"]

    def packed_hash(path) -> str:
        return hashlib.sha256(path.read_text().strip().encode()).hexdigest()

    old_root = DEFAULT_MANIFEST.parent
    old_rows = json.loads(DEFAULT_MANIFEST.read_text())
    old_hashes = {packed_hash(old_root / row["file"]) for row in old_rows}
    confirmation_root = confirmation_manifest_path.parent
    confirmation_hashes = {
        packed_hash(confirmation_root / relative) for relative in confirmation_files
    }

    assert len(all_labels) == 160
    assert len(all_labels - development_excluded) == split["development_count"] == 129
    validation_labels = set(split["development_validation_files"])
    assert len(validation_labels) == split["development_validation_count"] == 24
    assert split["development_training_count"] == 105
    assert validation_labels <= all_labels - development_excluded
    assert {label.split("/")[0] for label in validation_labels} == {
        "anti_meta_lycanroc_tech",
        "charizard_sun_offense",
        "gardevoir_maushold",
        "rain_offense",
        "sand_offense",
        "triple_setup_balance",
    }
    assert development_excluded <= all_labels
    assert development_excluded.isdisjoint(split["prior_confirmation_teams_retired_after_smoke"])
    assert len(confirmation_files) == split["confirmation_count"] == 31
    assert len(confirmation_hashes) == 31
    assert all(rows_by_file[relative]["source"] == "variant" for relative in confirmation_files)
    assert confirmation_hashes.isdisjoint(old_hashes)


def test_q_development_subpartitions_are_disjoint_before_matchmaking() -> None:
    split = json.loads(DEFAULT_Q_TEAM_SPLIT.read_text())
    all_teams = load_team_pool(DEFAULT_MANIFEST)

    training = _select_q_development_teams(all_teams, split, "training")
    validation = _select_q_development_teams(all_teams, split, "validation")

    training_labels = {team.label for team in training}
    validation_labels = {team.label for team in validation}
    assert len(training_labels) == 105
    assert len(validation_labels) == 24
    assert training_labels.isdisjoint(validation_labels)


def _tiny_q_setup():
    """Model + train/validation split just big enough to run a few training steps."""

    model = ActionResponseValueNet(
        CandidatePolicyValueNet(
            use_meta_features=True,
            use_information_features=True,
            use_tactical_features=True,
        )
    )
    train = [
        *_root_samples("r1", "sun/a", "rain/a"),
        *_root_samples("r2", "sun/b", "rain/b"),
    ]
    return model, train, _root_samples("r3", "sun/c", "rain/c")


def test_backbone_learning_rate_requires_train_backbone():
    model, train, validation = _tiny_q_setup()
    config = QTrainingConfig(train_backbone=False, backbone_learning_rate=1e-5)
    with pytest.raises(ValueError, match="requires train_backbone"):
        train_q_model(model, train, validation, config, device="cpu")


def test_negative_backbone_learning_rate_is_rejected():
    model, train, validation = _tiny_q_setup()
    config = QTrainingConfig(train_backbone=True, backbone_learning_rate=-1e-5)
    with pytest.raises(ValueError, match="nonnegative"):
        train_q_model(model, train, validation, config, device="cpu")


def test_frozen_backbone_leaves_encoder_weights_untouched():
    torch.manual_seed(3)
    model, train, validation = _tiny_q_setup()
    before = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
    report = train_q_model(
        model,
        train,
        validation,
        QTrainingConfig(epochs=2, patience=2, seed=3, train_backbone=False),
        device="cpu",
    )
    after = model.backbone.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)
    assert report["backbone_learning_rate"] == 0.0
    assert report["head_parameters"] > 0 and report["backbone_parameters"] > 0


def test_unfrozen_backbone_moves_the_encoder_and_records_its_rate():
    torch.manual_seed(3)
    model, train, validation = _tiny_q_setup()
    before = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
    report = train_q_model(
        model,
        train,
        validation,
        QTrainingConfig(
            epochs=3, patience=3, seed=3, train_backbone=True, backbone_learning_rate=1e-3
        ),
        device="cpu",
    )
    after = model.backbone.state_dict()
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert report["backbone_learning_rate"] == pytest.approx(1e-3)


def test_a_slower_backbone_rate_moves_the_encoder_less():
    """The point of the split rate: the representation drifts, it does not lurch."""

    def drift(backbone_lr):
        torch.manual_seed(3)
        model, train, validation = _tiny_q_setup()
        before = {k: v.detach().clone() for k, v in model.backbone.state_dict().items()}
        train_q_model(
            model,
            train,
            validation,
            QTrainingConfig(
                epochs=3,
                patience=3,
                seed=3,
                train_backbone=True,
                backbone_learning_rate=backbone_lr,
            ),
            device="cpu",
        )
        after = model.backbone.state_dict()
        return sum(float((after[k].float() - before[k].float()).abs().sum()) for k in before)

    assert drift(1e-5) < drift(1e-3)


def test_omitting_the_backbone_rate_reproduces_the_old_single_rate_behaviour():
    torch.manual_seed(3)
    model, train, validation = _tiny_q_setup()
    report = train_q_model(
        model,
        train,
        validation,
        QTrainingConfig(epochs=2, patience=2, seed=3, train_backbone=True),
        device="cpu",
    )
    assert report["backbone_learning_rate"] == pytest.approx(1e-3)
