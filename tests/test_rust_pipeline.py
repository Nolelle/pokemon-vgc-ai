"""Critical-behavior tests for the native Rust rich encoder, patch_public, and Rust search."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

TEAM_PATH = Path(__file__).resolve().parents[1] / "teams" / "mc_ladder_01.packed.txt"


@pytest.fixture(autouse=True)
def _require_rust_stack():
    """Skip inside test execution (not module collection) so `pytest -m integration`
    never records a module-level skip when the optional `pikapro` extra is absent.
    """
    global pikapro, rl, torch, DirectBattle, ReplicaRustWorker, RICH_MON_WIDTH, RICH_OBS_SIZE
    global RustFactoredPolicy, load_rust_policy, save_rust_policy
    global RustExactMirror, search_public_rust
    pikapro = pytest.importorskip("pikapro")
    torch = pytest.importorskip("torch")
    import pikapro.rl as _rl

    from vgc.rl.env import DirectBattle as _DirectBattle
    from vgc.rl.replica_engine import ReplicaRustWorker as _ReplicaRustWorker
    from vgc.rl.rust_policy import (
        RICH_MON_WIDTH as _RICH_MON_WIDTH,
        RICH_OBS_SIZE as _RICH_OBS_SIZE,
        RustFactoredPolicy as _RustFactoredPolicy,
        load_rust_policy as _load_rust_policy,
        save_rust_policy as _save_rust_policy,
    )
    from vgc.rl.rust_search import (
        RustExactMirror as _RustExactMirror,
        search_public_rust as _search_public_rust,
    )

    rl = _rl
    DirectBattle = _DirectBattle
    ReplicaRustWorker = _ReplicaRustWorker
    RICH_MON_WIDTH = _RICH_MON_WIDTH
    RICH_OBS_SIZE = _RICH_OBS_SIZE
    RustFactoredPolicy = _RustFactoredPolicy
    load_rust_policy = _load_rust_policy
    save_rust_policy = _save_rust_policy
    RustExactMirror = _RustExactMirror
    search_public_rust = _search_public_rust


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
    np.testing.assert_array_equal(obs1[: 6 * RICH_MON_WIDTH], state_obs[: 6 * RICH_MON_WIDTH])
    np.testing.assert_array_equal(obs1[12 * RICH_MON_WIDTH :], state_obs[12 * RICH_MON_WIDTH :])


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
                assert np.allclose(
                    obs_m[: 6 * RICH_MON_WIDTH], sess.obs[: 6 * RICH_MON_WIDTH], atol=1e-6
                )
                assert np.allclose(
                    obs_m[12 * RICH_MON_WIDTH :], sess.obs[12 * RICH_MON_WIDTH :], atol=1e-6
                )
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


def test_rust_policy_is_equivariant_to_move_order_and_team_slot_order() -> None:
    """Swapping two moves on our active Pokémon or two bench Pokémon must leave value
    unchanged and swap the corresponding move/switch logits identically (generalizes to
    unseen teams regardless of move or team-slot ordering).
    """
    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()
    b = pikapro.Battle("championsvgc", [t1, t2], seed=[1, 2, 3, 4], log=True)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    obs = b.encode_rich(0)

    # Also verify encode_state_rich populates all 6 opponent preview rows (fog-safe).
    obs_state = b.encode_state_rich(0)
    foe_rows = obs_state[6 * RICH_MON_WIDTH : 12 * RICH_MON_WIDTH].reshape(6, RICH_MON_WIDTH)
    assert np.all(foe_rows[:, 8] == 1.0), "all 6 foe preview rows must be present"
    assert np.all(foe_rows[:, 0] > 0.0), "all 6 foe preview species must be populated"
    assert np.all(foe_rows[4:, 1] == 0.0), "unbrought foe items must stay hidden"
    assert np.all(foe_rows[4:, 2] == 0.0), "unbrought foe abilities must stay hidden"
    assert np.all(foe_rows[4:, 4:8] == 0.0), "unbrought foe moves must stay hidden"
    assert np.all(foe_rows[4:, 48:54] == 0.0), "unbrought foe exact stats must stay hidden"

    torch.manual_seed(17)
    policy = RustFactoredPolicy(width=64, layers=2).eval()

    # 1. Swap Move 0 and Move 3 on active slot 0 (own row 0)
    obs_mv = obs.copy()
    for a, c in ((4, 7), (66, 69), (70, 73)):
        obs_mv[a], obs_mv[c] = obs_mv[c], obs_mv[a]

    with torch.no_grad():
        enc_orig = policy.encode_board(torch.from_numpy(obs).unsqueeze(0))
        enc_mv = policy.encode_board(torch.from_numpy(obs_mv).unsqueeze(0))
        v_orig, _ = policy.value_from_context(enc_orig.context)
        v_mv, _ = policy.value_from_context(enc_mv.context)
        assert float((v_orig - v_mv).abs().max()) < 1e-5

        l0_orig = policy.slot0_logits(enc_orig)[0]
        l0_mv = policy.slot0_logits(enc_mv)[0]
        assert float((l0_orig[0:10] - l0_mv[30:40]).abs().max()) < 1e-5
        assert float((l0_orig[30:40] - l0_mv[0:10]).abs().max()) < 1e-5
        assert float((l0_orig[10:30] - l0_mv[10:30]).abs().max()) < 1e-5
        assert float((l0_orig[40:] - l0_mv[40:]).abs().max()) < 1e-5

        # Semantic Slot-1 conditioning must also follow the swapped move!
        l1_orig = policy.slot1_logits(enc_orig, torch.tensor([2]))[0]
        l1_mv = policy.slot1_logits(enc_mv, torch.tensor([32]))[0]
        assert float((l1_orig - l1_mv).abs().max()) < 1e-5

    # 2. Swap bench Pokémon rows 2 and 3 on our own side
    obs_sw = obs.copy()
    r2 = obs_sw[2 * RICH_MON_WIDTH : 3 * RICH_MON_WIDTH].copy()
    r3 = obs_sw[3 * RICH_MON_WIDTH : 4 * RICH_MON_WIDTH].copy()
    obs_sw[2 * RICH_MON_WIDTH : 3 * RICH_MON_WIDTH] = r3
    obs_sw[3 * RICH_MON_WIDTH : 4 * RICH_MON_WIDTH] = r2

    with torch.no_grad():
        enc_sw = policy.encode_board(torch.from_numpy(obs_sw).unsqueeze(0))
        v_sw, _ = policy.value_from_context(enc_sw.context)
        assert float((v_orig - v_sw).abs().max()) < 1e-5

        l0_sw = policy.slot0_logits(enc_sw)[0]
        assert float((l0_orig[42] - l0_sw[43]).abs()) < 1e-5
        assert float((l0_orig[43] - l0_sw[42]).abs()) < 1e-5
        assert float((l0_orig[:40] - l0_sw[:40]).abs().max()) < 1e-5

        l1_sw_orig = policy.slot1_logits(enc_orig, torch.tensor([42]))[0]
        l1_sw_swap = policy.slot1_logits(enc_sw, torch.tensor([43]))[0]
        assert float((l1_sw_orig[:40] - l1_sw_swap[:40]).abs().max()) < 1e-5
        assert float((l1_sw_orig[42] - l1_sw_swap[43]).abs()) < 1e-5
        assert float((l1_sw_orig[43] - l1_sw_swap[42]).abs()) < 1e-5


def test_zero_padded_checkpoint_loading_is_bit_exact(tmp_path: Path) -> None:
    """Loading an older 80-float-per-mon checkpoint into the 88-float counter-aware model
    must zero-pad only the new scalar columns so logits and values stay bit-exact even
    when the new counter features are non-zero.
    """
    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()
    b = pikapro.Battle("championsvgc", [t1, t2], seed=[1, 2, 3, 4], log=True)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    obs = b.encode_rich(0)

    torch.manual_seed(99)
    full = RustFactoredPolicy(width=64, layers=2).eval()
    # Zero out the 8 new tail weights manually and save a legacy 64-col checkpoint.
    with torch.no_grad():
        full.mon_scalars.weight[:, 64:].zero_()
    ckpt = full.checkpoint()
    ckpt["model"]["mon_scalars.weight"] = ckpt["model"]["mon_scalars.weight"][:, :64].clone()
    legacy_path = tmp_path / "legacy_80col.pt"
    torch.save(ckpt, legacy_path)

    loaded = load_rust_policy(legacy_path, device="cpu")
    obs_with_counters = obs.copy()
    # Activate new columns 80..88 on own0 (perish_turns, yawn, leech_seed, ...)
    obs_with_counters[80:88] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    with torch.no_grad():
        e_base = loaded.encode_board(torch.from_numpy(obs).unsqueeze(0))
        e_cnt = loaded.encode_board(torch.from_numpy(obs_with_counters).unsqueeze(0))
        v_base, _ = loaded.value_from_context(e_base.context)
        v_cnt, _ = loaded.value_from_context(e_cnt.context)
        l0_base = loaded.slot0_logits(e_base)
        l0_cnt = loaded.slot0_logits(e_cnt)
        assert float((v_base - v_cnt).abs().max()) == 0.0
        assert float((l0_base - l0_cnt).abs().max()) == 0.0


def test_patch_public_enforces_perish_yawn_choice_lock_protect_and_extenders() -> None:
    """Critical VGC gameplay counters must actually execute in patched Rust search roots."""
    from vgc.rl.rust_search import _turns_left

    # 1. Extender items: condition_clock reports negative elapsed (-3 for 8-turn Damp Rock / Light Clay).
    assert _turns_left(start_turn=4, current_turn=1, default_duration=5) == 8
    assert _turns_left(start_turn=4, current_turn=6, default_duration=5) == 3

    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()

    # 2. Perish Song (perish_turns=1): staying in must faint the Pokémon at end of turn!
    b = pikapro.Battle("championsvgc", [t1, t2], seed=[1, 2, 3, 4], log=True)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    b.patch_public(
        {
            "turn": 4,
            "sides": [
                {
                    "active_species": ["salamence", "excadrill"],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [
                        {"species": "salamence", "hp_pct": 100, "perish_turns": 1, "active_turns": 3},
                        {"species": "excadrill", "hp_pct": 100, "yawn_turns": 1, "active_turns": 2},
                    ],
                },
                {
                    "active_species": ["incineroar", "rillaboom"],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [],
                },
            ],
        }
    )
    obs_before = b.encode_state_rich(0)
    # Own0 (salamence) should show perish_turns == 1/3; Own1 (excadrill) should show vol_yawn == 1.0.
    assert abs(float(obs_before[80]) - (1.0 / 3.0)) < 1e-5
    assert float(obs_before[RICH_MON_WIDTH + 81]) == 1.0

    # Both Protect -> at end of turn Salamence faints to Perish 0 and Excadrill falls asleep to Yawn!
    b.submit(0, "move 4, move 4")
    b.submit(1, b.legal_choices(1)[0])
    # Resolve any forced switch on side 0 after Salamence faints to Perish Song!
    assert b.request_kind(0) == "force_switch", "Salamence at perish_turns=1 must faint at end of turn"
    b.submit(0, b.legal_choices(0)[0])
    obs_after = b.encode_state_rich(0)
    # Find Excadrill row (row 1): status_slp is column 21 (`16..22` = brn,frz,par,psn,tox,slp)
    assert float(obs_after[RICH_MON_WIDTH + 21]) == 1.0, "Excadrill with yawn_turns=1 must fall asleep"

    # 3. Choice Lock restricts legal moves to only the locked move (plus switches).
    b2 = pikapro.Battle("championsvgc", [t1, t2], seed=[1, 2, 3, 4], log=True)
    b2.submit(0, "team 1234")
    b2.submit(1, "team 1234")
    b2.patch_public(
        {
            "turn": 2,
            "sides": [
                {
                    "active_species": ["salamence", "excadrill"],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [
                        {
                            "species": "salamence",
                            "item": "choicescarf",
                            "choice_lock": "hypervoice",
                            "active_turns": 1,
                            "active_move_actions": 1,
                        },
                    ],
                },
                {
                    "active_species": [],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [],
                },
            ],
        }
    )
    slot0_moves = {str(c).split(",")[0].strip() for c in b2.legal_choices(0) if str(c).startswith("move")}
    assert slot0_moves == {"move 1"}, f"Choice-locked Salamence must only offer move 1 (Hyper Voice), got {slot0_moves}"

    # 4. Protect stall_turns=1 expires cleanly after 1 non-Protect attack turn!
    b3 = pikapro.Battle("championsvgc", [t1, t2], seed=[1, 2, 3, 4], log=True)
    b3.submit(0, "team 1234")
    b3.submit(1, "team 1234")
    b3.patch_public(
        {
            "turn": 2,
            "sides": [
                {
                    "active_species": ["salamence", "excadrill"],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [
                        {"species": "salamence", "protect_counter": 1, "active_turns": 1, "active_move_actions": 1},
                    ],
                },
                {
                    "active_species": [],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [],
                },
            ],
        }
    )
    # Turn 2: Salamence attacks instead of Protecting -> stall must expire by Turn 3!
    b3.submit(0, "move 1, move 4")
    b3.submit(1, b3.legal_choices(1)[0])
    while b3.request_kind(0) == "force_switch" or b3.request_kind(1) == "force_switch":
        for s_i in (0, 1):
            if b3.request_kind(s_i) == "force_switch":
                b3.submit(s_i, b3.legal_choices(s_i)[0])
    obs_t3 = b3.encode_state_rich(0)
    assert float(obs_t3[60]) == 0.0, "protect_chain must reset to 0 after skipping Protect for 1 turn"


def test_turn2_nash_solver_and_preview_dedup() -> None:
    """Verify exact 2x2 zero-sum Nash values and symmetric preview deduplication."""
    from vgc.rl.rust_search import _dedup_preview_groups, _solve_2x2_zero_sum

    # Pure saddle point: row 0 guarantees +0.4 against both opp responses.
    assert abs(_solve_2x2_zero_sum([[0.4, 0.8], [-0.2, 0.9]]) - 0.4) < 1e-9
    # Symmetric matching pennies (Protect vs Attack mindgame): exact mixed Nash value is 0.0.
    assert abs(_solve_2x2_zero_sum([[0.5, -0.5], [-0.5, 0.5]]) - 0.0) < 1e-9
    # Asymmetric mixed game: [[3, 0], [1, 2]] -> v = (6 - 0) / (3 - 0 - 1 + 2) = 1.5.
    assert abs(_solve_2x2_zero_sum([[3.0, 0.0], [1.0, 2.0]]) - 1.5) < 1e-9

    # Back-order duplicates (0,1,2,3) and (0,1,3,2) + lead swap (1,0,2,3) merge and sum probs.
    orders = np.array([[0, 1, 2, 3], [0, 1, 3, 2], [1, 0, 2, 3], [0, 2, 1, 3]], dtype=np.int64)
    grouped = _dedup_preview_groups([0, 1, 2, 3], [0.30, 0.25, 0.15, 0.20], orders)
    assert len(grouped) == 2
    assert grouped[0][0] == 0 and abs(grouped[0][2] - 0.70) < 1e-9
    assert grouped[1][0] == 3 and abs(grouped[1][2] - 0.20) < 1e-9


def test_player_view_extender_timers_stay_nonzero_on_turns_6_to_8() -> None:
    """When Damp Rock / Terrain Extender / Light Clay keeps a condition active past turn 5,
    PlayerView (`encode_rich`) must infer the 8-turn duration (0.6, 0.4, 0.2) instead of 0.0.
    """
    t1 = (
        "Politoed||DampRock|Drizzle|Protect,Scald,Yawn,HelpingHand|Calm|32,,16,,18,||||50|]"
        "Rillaboom||TerrainExtender|GrassySurge|Protect,FakeOut,LeechSeed,Uturn|Adamant|32,32,2,,,||||50|]"
        "Snorlax||SitrusBerry|ThickFat|Protect,Rest,BodySlam,Crunch|Careful|32,,16,,18,||||50|]"
        "Umbreon||Leftovers|InnerFocus|Protect,Moonlight,Snarl,FoulPlay|Calm|32,,16,,18,||||50|]"
        "Cresselia||RockyHelmet|Levitate|Protect,LunarBlessing,Psychic,IcyWind|Bold|32,,32,,2,||||50|]"
        "Blissey||CovertCloak|NaturalCure|Protect,SoftBoiled,SeismicToss,HelpingHand|Bold|32,,32,,2,||||50|"
    )
    t2 = (
        "Snorlax||SitrusBerry|ThickFat|Protect,Rest,BodySlam,Crunch|Careful|32,,16,,18,||||50|]"
        "Umbreon||Leftovers|InnerFocus|Protect,Moonlight,Snarl,FoulPlay|Calm|32,,16,,18,||||50|]"
        "Cresselia||RockyHelmet|Levitate|Protect,LunarBlessing,Psychic,IcyWind|Bold|32,,32,,2,||||50|]"
        "Blissey||CovertCloak|NaturalCure|Protect,SoftBoiled,SeismicToss,HelpingHand|Bold|32,,32,,2,||||50|]"
        "Incineroar||SafetyGoggles|Intimidate|Protect,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|]"
        "Milotic||FlameOrb|MarvelScale|Protect,Scald,Recover,IcyWind|Bold|32,,32,,2,||||50|"
    )
    b = pikapro.Battle("championsvgc", [t1, t2], seed=(1, 2, 3, 4), log=True)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    rain_by_turn: list[float] = []
    ter_by_turn: list[float] = []
    for _ in range(8):
        b.submit(0, "move 4 -2, move 1")
        b.submit(1, "move 2, move 2")
        o = b.encode_rich(0)
        rain_by_turn.append(round(float(o[12 * RICH_MON_WIDTH + 47]), 2))
        ter_by_turn.append(round(float(o[12 * RICH_MON_WIDTH + 52]), 2))
    # Turns 1..4 count down 5-turn base (0.8, 0.6, 0.4, 0.2), then Turns 5..7 infer 8-turn extender (0.6, 0.4, 0.2), Turn 8 expires (0.0).
    assert rain_by_turn == [0.8, 0.6, 0.4, 0.2, 0.6, 0.4, 0.2, 0.0], rain_by_turn
    assert ter_by_turn == [0.8, 0.6, 0.4, 0.2, 0.6, 0.4, 0.2, 0.0], ter_by_turn
