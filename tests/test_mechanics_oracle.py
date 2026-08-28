from __future__ import annotations

import pytest

from vgc.actions import enumerate_joint_orders
from vgc.config import REPO_ROOT
from vgc.rl.env import DirectBattle, SimWorker, choice_string
from vgc.rl.mechanics_oracle import evaluate_exact_branches

pytestmark = pytest.mark.integration


def test_exact_oracle_branches_showdown_without_mutating_root() -> None:
    team = (REPO_ROOT / "teams" / "phase2_mirror.packed.txt").read_text().strip()
    with SimWorker() as worker:
        root = DirectBattle.start(
            worker,
            "oracle-root",
            team,
            team,
            seed=[1, 2, 3, 4],
        )
        try:
            root.step({"p1": "team 1234", "p2": "team 1234"})
            root_before = root.inspect()
            choices = {}
            for side in ("p1", "p2"):
                orders = enumerate_joint_orders(root.battles[side])
                assert orders
                choices[side] = choice_string(orders[0])

            branches = evaluate_exact_branches(
                root,
                [choices],
                future_seeds=([11, 12, 13, 14], [21, 22, 23, 24]),
                branch_prefix="oracle-test",
            )

            assert len(branches) == 2
            assert all(branch.state_for("p1").turn >= 1 for branch in branches)
            assert all(dict(branch.public_lines)["p1"] for branch in branches)
            # Exact counterfactuals share the past but never advance the source battle.
            root_after = root.inspect()
            assert root_after["stateHash"] == root_before["stateHash"]
            assert root_after["prngSeed"] == root_before["prngSeed"]
        finally:
            root.close()


def test_exact_oracle_rejects_a_missing_side_choice() -> None:
    team = (REPO_ROOT / "teams" / "phase2_mirror.packed.txt").read_text().strip()
    with SimWorker() as worker:
        root = DirectBattle.start(worker, "oracle-invalid", team, team, seed=[1, 1, 1, 1])
        try:
            with pytest.raises(ValueError, match="expects choices"):
                evaluate_exact_branches(root, [{"p1": "team 1234"}])
        finally:
            root.close()
