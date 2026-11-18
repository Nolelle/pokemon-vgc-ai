"""Critical checks for the from-scratch league's information and learning boundaries."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from vgc.actions import enumerate_joint_orders
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker

torch = pytest.importorskip("torch")

from selfplay.train_replication_league import LeagueAgent, update_policy  # noqa: E402
from vgc.rl.replication_policy import (  # noqa: E402
    ReplicationPolicy,
    encode_decision,
    load_replication_policy,
)

TEAM = Path(__file__).resolve().parents[1] / "teams" / "meta1.packed.txt"


@pytest.fixture
def local_battle():
    if not (DEFAULT_SHOWDOWN_REPO / "dist/sim/index.js").exists():
        pytest.skip(f"no built Showdown simulator at {DEFAULT_SHOWDOWN_REPO}")
    with SimWorker() as worker:
        yield worker, TEAM.read_text().strip()


@pytest.mark.integration
def test_opponent_hidden_request_cannot_change_policy_input(local_battle):
    worker, team = local_battle
    # Change the opponent's held items in the simulator without changing its public species.
    changed = []
    for slot in team.split("]"):
        fields = slot.split("|")
        fields[2] = ""
        changed.append("|".join(fields))
    hidden_variant = "]".join(changed)
    assert hidden_variant != team
    seed = [1, 2, 3, 4]
    first = DirectBattle.start(worker, "regular", team, team, seed=seed)
    second = DirectBattle.start(worker, "without-items", team, hidden_variant, seed=seed)
    try:
        for preview in (True, False):
            encodings = []
            for match in (first, second):
                battle = match.battles["p1"]
                # The offline runner may attach these. The encoder may only read battle.
                battle._vgc_direct_root = match
                battle._vgc_direct_side = "p1"
                orders = None if preview else enumerate_joint_orders(battle)
                encodings.append(encode_decision(battle, orders, preview=preview))
            for name in ("pokemon", "scalars", "field", "candidates"):
                np.testing.assert_array_equal(
                    getattr(encodings[0], name),
                    getattr(encodings[1], name),
                )
            if preview:
                # Unrevealed opponent sets add six species with no item, move, or ability.
                assert np.all(encodings[0].pokemon[6:, 0] > 0)
                assert np.all(encodings[0].pokemon[6:, 1:] == 0)
                choices = encodings[0].candidates
                assert len(choices) == 360
                assert np.all((choices >= 1) & (choices <= 6))
                assert all(len(set(choice)) == 4 for choice in choices)
                view = first.battles["p1"]
                original_team = dict(view.team)
                try:
                    view.team = dict(reversed(list(original_team.items())))
                    reordered = encode_decision(view, preview=True)
                    np.testing.assert_array_equal(encodings[0].pokemon, reordered.pokemon)
                finally:
                    view.team = original_team
                # Leads with items that do not trigger on entry.
                for match in (first, second):
                    match.step({"p1": "team 1256", "p2": "team 1256"})
    finally:
        first.close()
        second.close()


@pytest.mark.integration
def test_terminal_updates_train_preview_and_round_trip_for_search(local_battle, tmp_path):
    worker, team = local_battle
    model = ReplicationPolicy(width=16, layers=1).eval()
    sampler = LeagueAgent("learner", model, seed=5, record=True)
    battle = DirectBattle.start(worker, "learning", team, team, seed=[1, 2, 3, 4])
    try:
        view = battle.battles["p1"]
        preview = [sampler.choose(view) for _ in range(2)]
        battle.step({"p1": preview[0], "p2": "team 1234"})
        for _ in range(2):
            sampler.choose(view)
        before_preview = model.preview_policy[-1].weight.detach().clone()
        before_turn = model.turn_policy[-1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
        loss = update_policy(
            model,
            optimizer,
            list(zip(sampler.steps, [1, -1, -1, 1], strict=True)),
            random.Random(5),
            epochs=1,
            batch_size=4,
        )
        assert np.isfinite(loss)
        assert not torch.equal(before_preview, model.preview_policy[-1].weight)
        assert not torch.equal(before_turn, model.turn_policy[-1].weight)
        path = tmp_path / "replication.pt"
        torch.save(model.checkpoint(), path)
        loaded = load_replication_policy(path)
        orders = enumerate_joint_orders(view)
        logits, value = loaded.score(view, orders)
        assert logits.shape == (len(orders),)
        assert np.all(np.isfinite(logits))
        assert -1 <= value <= 1
        np.testing.assert_array_equal(logits, model.score(view, orders)[0])
        assert loaded.value(view) == value
    finally:
        battle.close()
