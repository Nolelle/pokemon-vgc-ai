"""Unit tests for offline/run_scaling_curve.py's pure helpers."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from offline.run_scaling_curve import fit_power_law, materialize_nested_subsets  # noqa: E402


def test_fit_power_law_recovers_the_generating_exponent():
    # miss = N^-0.35 exactly -> slope -0.35, perfect fit.
    points = [(n, n**-0.35) for n in (1110, 2893, 5750, 11550)]
    fit = fit_power_law(points)
    assert fit["slope_alpha"] == pytest.approx(-0.35, abs=1e-6)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_fit_power_law_needs_three_positive_points():
    assert fit_power_law([(100, 0.1), (200, 0.05)])["slope_alpha"] is None
    assert fit_power_law([(100, 0.0), (200, 0.05), (400, 0.03)])["slope_alpha"] is None


def test_nested_subsets_are_prefixes_of_one_shuffle(tmp_path):
    from vgc.rl.distill import DistillationSample
    from vgc.rl.encoding import CandidateFeatures, InformationFeatures

    def sample(battle: str):
        return DistillationSample(
            battle_id=battle,
            state_indices=np.zeros(4, dtype=np.int64),
            state_scalars=np.zeros(4, dtype=np.float32),
            history_scalars=np.zeros(4, dtype=np.float32),
            candidates=CandidateFeatures(
                move_indices=np.zeros((1, 2), dtype=np.int64),
                target_indices=np.zeros((1, 2), dtype=np.int64),
                switch_species_indices=np.zeros((1, 2), dtype=np.int64),
                flags=np.zeros((1, 2, 4), dtype=np.float32),
                tactical=np.zeros((1, 42), dtype=np.float32),
            ),
            teacher_action_index=0,
            information=InformationFeatures(
                indices=np.zeros(84, dtype=np.int64), scalars=np.zeros(384, dtype=np.float32)
            ),
        )

    dataset = tmp_path / "all.pt"
    from vgc.rl.demonstrations import load_demonstrations, save_demonstrations

    samples = [sample(f"battle-{i:03d}") for i in range(40)]
    save_demonstrations(dataset, samples)

    paths = materialize_nested_subsets(
        dataset, [0.25, 0.5, 1.0], seed=7, out_dir=tmp_path / "curve"
    )

    battle_sets = {
        fraction: {s.battle_id for s in load_demonstrations(path)}
        for fraction, path in paths.items()
    }
    assert len(battle_sets[0.25]) == 10
    assert len(battle_sets[0.5]) == 20
    assert len(battle_sets[1.0]) == 40
    assert battle_sets[0.25] <= battle_sets[0.5] <= battle_sets[1.0]
