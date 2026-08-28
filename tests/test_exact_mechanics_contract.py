"""What the mechanics gate promises, checked rather than asserted in prose.

The promise is narrow and testable: in every gated decision path, the outcome of a
predicted turn is produced by the official Pokemon Showdown engine, not by Python
re-implementations of accuracy, critical hits, sleep, secondaries, or ordering.

The strategic weights that rank those exactly-simulated outcomes are policy, not
mechanics, and are tracked separately in `data/champions/mechanics_coverage.json` under
`policy_approximations`.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from vgc.config import REPO_ROOT
from vgc.mechanics_gate import COVERAGE_PATH
from vgc.models import PolicyConfig
from vgc.rl import distill, exact_search
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.match import play_battle

APPROXIMATE_TRANSITION_NAMES = ("resolve_exchange", "search_joint_orders")


@pytest.mark.parametrize("module", (exact_search, distill))
def test_exact_modules_cannot_reach_the_python_exchange_simulator(module) -> None:
    for name in APPROXIMATE_TRANSITION_NAMES:
        assert not hasattr(module, name), (
            f"{module.__name__} can still call {name}; an approximate transition would "
            "be one import away from minting a training label"
        )


def test_every_training_and_ladder_entry_point_demands_mechanics_features() -> None:
    """A run cannot opt out of the complete snapshot and the exact teacher."""

    entry_points = (
        "selfplay/train_ppo.py",
        "selfplay/train_fixed_mirror.py",
        "selfplay/train_imitation.py",
        "selfplay/train_full_pipeline.py",
        "selfplay/train_counterfactual_q.py",
        "ladder/run_ladder.py",
    )
    for relative in entry_points:
        source = (REPO_ROOT / relative).read_text()
        assert "use_mechanics_features" in source, relative


def test_coverage_file_names_its_remaining_policy_approximations() -> None:
    """Exactness of transitions must not be quietly read as exactness of judgement."""

    coverage = json.loads(COVERAGE_PATH.read_text())
    approximations = coverage.get("policy_approximations")
    assert approximations, "the gate must keep listing what is still judgement, not rule"
    for entry in approximations:
        assert entry["id"] and entry["why_not_a_mechanics_gap"] and entry["where"]


@pytest.mark.integration
def test_a_recorded_teacher_label_is_backed_by_real_showdown_branches() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    from vgc.rl.agents import DirectAgent, make_direct_agent
    from vgc.rl.distill import TeacherRecordingPlayer

    team = (REPO_ROOT / "teams" / "meta1.packed.txt").read_text().strip()
    config = replace(
        PolicyConfig(),
        accept_open_team_sheet=False,
        use_rolling_horizon=False,
        search_our_candidates=2,
        search_opp_candidates=2,
        exact_search_future_samples=1,
    )
    teacher = TeacherRecordingPlayer(
        config=config, team=team, battle_format="gen9championsvgc2026regmb",
        start_listening=False,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        play_battle(
            worker,
            "exact-contract",
            {"p1": DirectAgent(teacher, name="teacher"), "p2": make_direct_agent("vgc", team)},
            {"p1": team, "p2": team},
            seed=[5, 6, 7, 8],
        )

    samples = teacher.distillation_samples
    assert samples, "the exact teacher recorded no labels at all"
    assert {sample.source_id for sample in samples} == {"exact_showdown_teacher_v1"}
    assert all(sample.mechanics is not None for sample in samples)
