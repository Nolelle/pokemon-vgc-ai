"""Critical-behavior tests for the native Rust rich encoder, patch_public, and Rust search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pikapro = pytest.importorskip("pikapro")
torch = pytest.importorskip("torch")

import pikapro.rl as rl  # noqa: E402
from vgc.rl.env import DirectBattle  # noqa: E402
from vgc.rl.replica_engine import ReplicaRustWorker  # noqa: E402
from vgc.rl.rust_policy import (  # noqa: E402
    RICH_OBS_SIZE,
    RustFactoredPolicy,
    load_rust_policy,
    save_rust_policy,
)
from vgc.rl.rust_search import RustExactMirror, search_public_rust  # noqa: E402

TEAM_PATH = Path(__file__).resolve().parents[1] / "teams" / "mc_ladder_01.packed.txt"


def _strip_items(packed: str) -> str:
    out = []
    for slot in packed.split("]"):
        fields = slot.split("|")
        fields[2] = ""
        out.append("|".join(fields))
    return "]".join(out)


def test_rust_rich_encoder_fog_of_war_and_channel_session_parity():
    team = TEAM_PATH.read_text().strip()
    no_items = _strip_items(team)
    b1 = pikapro.Battle("championsvgc", [team, team], seed=[1, 2, 3, 4], players=["p1", "p2"])
    b2 = pikapro.Battle("championsvgc", [team, no_items], seed=[1, 2, 3, 4], players=["p1", "p2"])

    # At team preview and turn 1 (with non-triggering leads), hidden opponent items must not leak.
    np.testing.assert_array_equal(b1.encode_rich(0), b2.encode_rich(0))

    b1.submit(0, "team 4123")
    b1.submit(1, "team 4123")
    b2.submit(0, "team 4123")
    b2.submit(1, "team 4123")
    obs1 = b1.encode_rich(0)
    obs2 = b2.encode_rich(0)
    assert obs1.shape == (RICH_OBS_SIZE,)
    np.testing.assert_array_equal(obs1, obs2)

    # ChannelSession driven by Showdown log lines must match Battle.encode_rich bit-for-bit.
    sess = rl.ChannelSession("championsvgc", observation="rich", team=team, username="p1")
    sess.feed_many(b1.log("p1") + ["|request|" + b1.request_json(0)])
    np.testing.assert_array_equal(obs1, sess.obs)

    # encode_state_rich must match encode_rich on own mons, sides, and field.
    state_obs = b1.encode_state_rich(0)
    np.testing.assert_array_equal(obs1[:480], state_obs[:480])
    np.testing.assert_array_equal(obs1[960:], state_obs[960:])


def test_battle_patch_public_and_rust_search_roundtrip(tmp_path):
    team = TEAM_PATH.read_text().strip()
    b = pikapro.Battle("championsvgc", [team, team], seed=[1, 2, 3, 4], players=["p1", "p2"])
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    b.patch_public(
        {
            "turn": 5,
            "weather": "sand",
            "weather_turns": 3,
            "terrain": "psychic",
            "terrain_turns": 4,
            "trick_room_turns": 2,
            "sides": [
                {
                    "mega_used": True,
                    "active_species": ["Salamence-Mega", "Excadrill"],
                    "force_switch": [False, False],
                    "side_conditions": [("tailwind", 3)],
                    "mons": [
                        {
                            "species": "Salamence",
                            "forme": "Salamence-Mega",
                            "hp_exact": 110,
                            "protect_counter": 1,
                            "active_turns": 2,
                            "active_move_actions": 1,
                        },
                        {"species": "Excadrill", "hp_exact": 150, "boosts": [1, 0, 0, 0, 0, 0, 0]},
                    ],
                },
                {
                    "mega_used": False,
                    "active_species": ["Tyranitar", "Milotic"],
                    "force_switch": [False, False],
                    "side_conditions": [],
                    "mons": [
                        {"species": "Tyranitar", "hp_pct": 60, "status": "brn"},
                        {"species": "Milotic", "hp_pct": 85},
                    ],
                },
            ],
        }
    )
    assert b.turn == 5
    l0 = b.legal_choices(0)
    l1 = b.legal_choices(1)
    assert len(l0) > 0 and len(l1) > 0
    forked = b.fork(seed=[7, 8, 9, 10])
    forked.submit(0, l0[0])
    forked.submit(1, l1[0])
    assert forked.turn >= 5

    # Save/load RustFactoredPolicy and run multi-world search_public_rust.
    policy = RustFactoredPolicy(width=32, layers=1).eval()
    ckpt = save_rust_policy(tmp_path / "rust.pt", policy)
    loaded = load_rust_policy(ckpt, device="cpu")
    with ReplicaRustWorker() as worker:
        live = DirectBattle.start(worker, "rust-search-test", team, team, seed=[1, 2, 3, 4])
        try:
            live.step({"p1": "team 1234", "p2": "team 1234"})
            mirror = RustExactMirror(team)
            res, audit = search_public_rust(
                live.battles["p1"],
                team,
                loaded,
                worlds=6,
                actions=4,
                seeds=2,
                iterations=500,
                mirror=mirror,
            )
            assert audit["searched"] >= 1
            assert len(res.actions) >= 1
            assert sum(res.probabilities) == pytest.approx(1.0, abs=1e-5)
            assert -1.0 <= res.value <= 1.0
        finally:
            live.close()


def test_rust_mirror_preserves_own_items_mask_parity_and_order_matching_both_seats() -> None:
    """After non-identity preview and switches, RustExactMirror must match ChannelSession
    bit-for-bit on own mons, globals, and factored action masks on both p1 and p2, preserve
    opponent HP%, and resolve every search/session choice back to a legal poke-env order.
    """
    from vgc.actions import enumerate_joint_orders
    from vgc.rl.env import DirectBattle, choice_string
    from vgc.rl.replica_engine import ReplicaRustWorker
    from vgc.rl.rust_policy import RustFactoredPolicy
    from vgc.rl.rust_search import (
        RustExactMirror,
        _factored_mask_for_battle,
        match_pikapro_choice_to_order,
        search_public_rust,
    )

    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()
    space = pikapro.ActionSpace("championsvgc")
    policy = RustFactoredPolicy(width=32, layers=1)

    with ReplicaRustWorker() as worker:
        live = DirectBattle.start(worker, "rust-seat-parity", t1, t2, seed=[1, 2, 3, 4])
        try:
            sess1 = rl.ChannelSession(
                "championsvgc", observation="rich", masks="factored", team=t1, username="p1"
            )
            sess2 = rl.ChannelSession(
                "championsvgc", observation="rich", masks="factored", team=t2, username="p2"
            )
            sess1.feed_many(live.last_lines["p1"])
            sess2.feed_many(live.last_lines["p2"])

            res = live.step({"p1": "team 1234", "p2": "team 2143"})
            sess1.feed_many(res.lines["p1"])
            sess2.feed_many(res.lines["p2"])

            c1 = next(
                choice_string(o)
                for o in enumerate_joint_orders(live.battles["p1"])
                if choice_string(o).startswith("switch ")
            )
            c2 = next(
                choice_string(o)
                for o in enumerate_joint_orders(live.battles["p2"])
                if choice_string(o).startswith("switch ")
            )
            res = live.step({"p1": c1, "p2": c2})
            sess1.feed_many(res.lines["p1"])
            sess2.feed_many(res.lines["p2"])

            for side, team, sess in (("p1", t1, sess1), ("p2", t2, sess2)):
                battle = live.battles[side]
                legal = enumerate_joint_orders(battle)
                root = RustExactMirror(team).build(battle)
                obs_m = root.battle.encode_state_rich(root.our_side_idx)
                fmask_m, min_sw_m, _ = _factored_mask_for_battle(
                    root.battle, root.our_side_idx, space, obs_m
                )
                assert np.allclose(obs_m[:480], sess.obs[:480], atol=1e-6)
                assert np.allclose(obs_m[960:], sess.obs[960:], atol=1e-6)
                assert np.array_equal(fmask_m, sess.mask)
                assert min_sw_m == sess.min_switches

                sr, audit = search_public_rust(
                    battle, team, policy, worlds=2, actions=4, seeds=1, seed=7
                )
                for act in sr.actions:
                    matched = match_pikapro_choice_to_order(
                        act,
                        battle,
                        legal,
                        slot_moves=audit["root_slot_moves"],
                        team_species=audit["root_team_species"],
                    )
                    assert matched is not None, f"unmatched search order on {side}: {act!r}"

                pairs, _, _ = policy.top_joint_actions(
                    sess.obs, sess.mask, sess.min_switches, space, limit=4
                )
                wire = str(sess.choice([pairs[0][0], pairs[0][1]]))
                assert match_pikapro_choice_to_order(wire, battle, legal) is not None
        finally:
            live.close()
