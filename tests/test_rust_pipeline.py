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


def test_encode_state_rich_parity_moved_revealed_and_extenders() -> None:
    """Track A fixes: `moved_this_turn` resets each turn, undamaged switched-out foes
    stay revealed via `foe_revealed`, and 8-turn extenders match `encode_rich` on turns 1..3.
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
    # Turn 1: Politoed uses Helping Hand, Rillaboom uses Protect; foe switches Snorlax out undamaged.
    b.submit(0, "move 4 -2, move 1")
    b.submit(1, "switch 3, move 1")

    # At start of Turn 2, active mons have NOT moved yet this turn (`o[65] == 0.0`).
    o_view = b.encode_rich(0)
    o_state = b.encode_state_rich(0, foe_revealed=[True, True, True, False, False, False])
    assert float(o_view[65]) == 0.0
    assert float(o_state[65]) == 0.0, "moved_this_turn must reset to 0.0 at start of Turn 2"
    assert float(o_state[RICH_MON_WIDTH + 65]) == 0.0

    # 8-turn Damp Rock / Terrain Extender on Turn 2 (`rem=7`, `elapsed=1`) must both read 0.8.
    assert abs(float(o_view[12 * RICH_MON_WIDTH + 47]) - 0.8) < 1e-6
    assert abs(float(o_state[12 * RICH_MON_WIDTH + 47]) - 0.8) < 1e-6
    assert abs(float(o_state[12 * RICH_MON_WIDTH + 52]) - 0.8) < 1e-6

    # Undamaged switched-out foe Snorlax (foe row 0 -> row index 6) must stay revealed & brought.
    foe_row0 = 6 * RICH_MON_WIDTH
    assert float(o_view[foe_row0 + 12]) == 1.0 and float(o_view[foe_row0 + 14]) == 1.0
    assert float(o_state[foe_row0 + 12]) == 1.0 and float(o_state[foe_row0 + 14]) == 1.0


def test_rust_policy_v3_equivariance_preview_symmetry_and_roundtrip(tmp_path: Path) -> None:
    """Track B: v3 policy (static move meta + GPU speed cues + symmetric preview head)
    must remain move/slot equivariant, give identical preview logits for swapped back/lead
    pairs, and round-trip cleanly alongside legacy v2 checkpoints.
    """
    import torch

    from vgc.rl.rust_policy import (
        RUST_POLICY_VERSION,
        RustFactoredPolicy,
        load_rust_policy,
        save_rust_policy,
    )

    assert RUST_POLICY_VERSION == 3
    torch.manual_seed(42)
    model = RustFactoredPolicy(width=64, layers=2, policy_version=3).eval()

    t1 = (
        "Tyranitar||ChoiceScarf|SandStream|RockSlide,Crunch,LowKick,Protect|Adamant|32,32,,,,2||||50|]"
        "Sneasler||FocusSash|Unburden|FakeOut,CloseCombat,DireClaw,Protect|Jolly|,32,,,,32||||50|]"
        "Gholdengo||LifeOrb|GoodAsGold|MakeItRain,ShadowBall,NastyPlot,Protect|Modest|,,2,32,,32||||50|]"
        "Incineroar||SafetyGoggles|Intimidate|FakeOut,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|]"
        "Rillaboom||MiracleSeed|GrassySurge|FakeOut,GrassyGlide,WoodHammer,Uturn|Adamant|32,32,2,,,||||50|]"
        "Pelipper||DampRock|Drizzle|WeatherBall,Hurricane,WideGuard,Protect|Modest|2,,,32,,32||||50|"
    )
    b = pikapro.Battle("championsvgc", [t1, t1], seed=(3, 4, 5, 6), log=True)
    obs_pv = b.encode_rich(0)
    with torch.no_grad():
        enc = model.encode_board(torch.from_numpy(obs_pv).unsqueeze(0))
        pv_logits = model.preview_logits(enc)[0].numpy()

    # Check exact symmetry for swapped back pair (0,1,2,3) vs (0,1,3,2) and swapped lead pair (1,0,2,3).
    orders = model.preview_orders.numpy()
    idx_0123 = int(np.where((orders == [0, 1, 2, 3]).all(axis=1))[0][0])
    idx_0132 = int(np.where((orders == [0, 1, 3, 2]).all(axis=1))[0][0])
    idx_1023 = int(np.where((orders == [1, 0, 2, 3]).all(axis=1))[0][0])
    assert abs(float(pv_logits[idx_0123]) - float(pv_logits[idx_0132])) < 1e-5
    assert abs(float(pv_logits[idx_0123]) - float(pv_logits[idx_1023])) < 1e-5

    # Fainted active-slot Pokémon (`active_slot > 0` AND `is_fainted == 1.0`, as PlayerView
    # keeps in 2v1/1v1 endgames) must be excluded from speed-rank comparisons!
    obs_faint = obs_pv.copy()
    # Set own0 (speed 100), own1 (speed 200), foe0 (speed 150, FAINTED), foe1 (speed 50) active.
    for row_i, slot_v, faint_v, spe_v in ((0, 1.0, 0.0, 100.0), (1, 2.0, 0.0, 200.0), (6, 1.0, 1.0, 150.0), (7, 2.0, 0.0, 50.0)):
        base_i = row_i * RICH_MON_WIDTH
        obs_faint[base_i + 3] = slot_v
        obs_faint[base_i + 8] = 1.0
        obs_faint[base_i + 11] = faint_v
        if row_i < 6:
            obs_faint[base_i + 53] = spe_v / 255.0
        else:
            obs_faint[base_i + 59] = (spe_v - 36.0) / 255.0
    x_f = torch.from_numpy(obs_faint).unsqueeze(0)
    m_f = x_f[:, : 12 * RICH_MON_WIDTH].view(1, 12, RICH_MON_WIDTH)
    g_f = x_f[:, 12 * RICH_MON_WIDTH :]
    ids_f = m_f[:, :, :8].long()
    sf = model._compute_speed_features(m_f, g_f, ids_f[:, :, 1], ids_f[:, :, 2], ids_f[:, :, 3])
    # Among 3 living actives (own1=200, own0=100, foe1=50), ranks must be 1.0, 0.5, 0.0; dead foe0 must be 0.0!
    assert abs(float(sf[0, 1, 2]) - 1.0) < 1e-5
    assert abs(float(sf[0, 0, 2]) - 0.5) < 1e-5
    assert abs(float(sf[0, 7, 2]) - 0.0) < 1e-5
    assert abs(float(sf[0, 6, 2]) - 0.0) < 1e-5

    # Advance into battle and test move-order equivariance + save/load roundtrip.
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    obs_turn = b.encode_rich(0)
    obs_swapped_mv = obs_turn.copy()
    # Swap move slots 0 and 1 on own row 0 (IDs 4/5, PP 66/67, disabled 70/71).
    obs_swapped_mv[4], obs_swapped_mv[5] = obs_swapped_mv[5], obs_swapped_mv[4]
    obs_swapped_mv[66], obs_swapped_mv[67] = obs_swapped_mv[67], obs_swapped_mv[66]
    obs_swapped_mv[70], obs_swapped_mv[71] = obs_swapped_mv[71], obs_swapped_mv[70]
    with torch.no_grad():
        enc1 = model.encode_board(torch.from_numpy(obs_turn).unsqueeze(0))
        enc2 = model.encode_board(torch.from_numpy(obs_swapped_mv).unsqueeze(0))
        v1, _ = model.value_from_context(enc1.context)
        v2, _ = model.value_from_context(enc2.context)
        l1 = model.slot0_logits(enc1)[0]
        l2 = model.slot0_logits(enc2)[0]
    assert abs(float(v1.item()) - float(v2.item())) < 1e-5
    assert torch.allclose(l1[0:10], l2[10:20], atol=1e-5)
    assert torch.allclose(l1[10:20], l2[0:10], atol=1e-5)

    # Checkpoint round-trip for v3 AND legacy v2.
    p_v3 = save_rust_policy(tmp_path / "v3.pt", model)
    loaded_v3 = load_rust_policy(p_v3, device="cpu")
    assert loaded_v3.policy_version == 3
    with torch.no_grad():
        v3_check, _ = loaded_v3.value_from_context(
            loaded_v3.encode_board(torch.from_numpy(obs_turn).unsqueeze(0)).context
        )
    assert abs(float(v1.item()) - float(v3_check.item())) < 1e-6

    model_v2 = RustFactoredPolicy(width=64, layers=2, policy_version=2).eval()
    p_v2 = save_rust_policy(tmp_path / "v2.pt", model_v2)
    loaded_v2 = load_rust_policy(p_v2, device="cpu")
    assert loaded_v2.policy_version == 2
    assert loaded_v2.ffn_mult == 2


def test_train_rust_vec_optimizer_decay_groups_and_ema_pfsp() -> None:
    """Track B: embeddings/LayerNorm/biases get 0 weight decay; PFSP EMA stays fresh."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "selfplay"))
    import train_rust_vec as trv

    from vgc.rl.rust_policy import RustFactoredPolicy

    model = RustFactoredPolicy(width=64, layers=2, policy_version=3)
    opt = trv.build_optimizer(model, lr=1e-4, weight_decay=1e-3)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 1e-3
    assert opt.param_groups[1]["weight_decay"] == 0.0
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert id(model.move_emb.weight) in no_decay_ids
    assert id(model.species_emb.weight) in no_decay_ids
    assert id(model.value_head[0].weight) in no_decay_ids  # LayerNorm

    snap = trv.Snapshot(0, Path("dummy.pt"))
    for _ in range(300):
        snap.record_game(0.0)
    assert snap.games <= 200.1
    # After learner starts beating it 100%, EMA priority must drop well below 0.25 within 200 games.
    for _ in range(200):
        snap.record_game(1.0)
    assert snap.priority < 0.25


def test_bug_hunt_p0_p1_locked_moves_consumed_items_mega_ability_and_unrevealed_gholdengo() -> None:
    """Regression suite for P0/P1 bug-hunt findings:
    - `_extract_rust_request_meta` + `match_pikapro_choice_to_order` on Locked/Recharge/Struggle
    - Outrage blocked by Protect clears `BattleMemory.locked_move`
    - Consumed items are not resurrected; Mega ability beats pre-Mega ability
    - Unrevealed single-ability bench mons (Gholdengo) do NOT leak `is_revealed=1.0`
    """
    from types import SimpleNamespace

    from vgc.battle_memory import BattleMemory
    from vgc.rl.rust_search import (
        _extract_rust_request_meta,
        _our_public_knowledge,
        match_pikapro_choice_to_order,
    )

    # 1. Locked rampage / two-turn / Recharge / Struggle request metadata extraction + matching.
    fake_req = SimpleNamespace(
        pokemon=[SimpleNamespace(species="Dragonite"), SimpleNamespace(species="Archaludon")],
        kind=SimpleNamespace(
            active=[
                SimpleNamespace(moves=SimpleNamespace(value="Outrage")),
                SimpleNamespace(moves="Recharge"),
            ]
        ),
    )
    slot_moves, team_sp = _extract_rust_request_meta(fake_req)
    assert slot_moves == [["outrage"], ["recharge"]]
    assert team_sp == ["dragonite", "archaludon"]

    fake_legal = [
        SimpleNamespace(
            first_order=SimpleNamespace(
                order=SimpleNamespace(id="outrage"), move_target=1, mega=False
            ),
            second_order=SimpleNamespace(
                order=SimpleNamespace(id="recharge"), move_target=0, mega=False
            ),
        )
    ]
    matched = match_pikapro_choice_to_order(
        "move 1, move 1",
        SimpleNamespace(last_request={}),
        fake_legal,
        slot_moves=slot_moves,
        team_species=team_sp,
    )
    assert matched is fake_legal[0]

    # 2. Outrage blocked by Protect must NOT leave `locked_move` active on the next turn.
    mem = BattleMemory("lock-test", our_role="p1")
    mem.observe_protocol(
        [
            ["", "turn", "2"],
            ["", "switch", "p1a: Dragonite", "Dragonite, L50, M", "100/100"],
            ["", "move", "p1a: Dragonite", "Outrage", "p2a: Incineroar"],
            ["", "-activate", "p2a: Incineroar", "move: Protect"],
        ]
    )
    assert mem.locked_move("p1", "Dragonite", 3) is None

    # 3. Consumed item is NOT resurrected & Mega Salamence gets Aerilate (not pre-Mega Intimidate).
    packed = (
        "Salamence||Salamencite|Intimidate|DoubleEdge,DracoMeteor,Tailwind,Protect|Jolly|,32,,,,32||||50|]"
        "Sneasler||FocusSash|Unburden|FakeOut,CloseCombat,DireClaw,Protect|Jolly|,32,,,,32||||50|]"
        "Gholdengo||LifeOrb|GoodAsGold|MakeItRain,ShadowBall,NastyPlot,Protect|Modest|,,2,32,,32||||50|]"
        "Incineroar||SafetyGoggles|Intimidate|FakeOut,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|]"
        "Tyranitar||SitrusBerry|SandStream|RockSlide,Crunch,LowKick,Protect|Adamant|32,32,,,,2||||50|]"
        "Rillaboom||MiracleSeed|GrassySurge|FakeOut,GrassyGlide,WoodHammer,Uturn|Adamant|32,32,2,,,||||50|"
    )
    mem2 = BattleMemory("mega-item-test", our_role="p1")
    mem2.observe_protocol(
        [
            ["", "switch", "p1a: Salamence", "Salamence, L50, M", "100/100"],
            ["", "-ability", "p1a: Salamence", "Intimidate", "boost"],
            ["", "-mega", "p1a: Salamence", "Salamence", "Salamencite"],
            ["", "switch", "p1b: Sneasler", "Sneasler, L50, M", "100/100"],
            ["", "-enditem", "p1b: Sneasler", "Focus Sash"],
        ]
    )
    fake_battle = SimpleNamespace(
        team={
            "p1: Salamence": SimpleNamespace(
                species="salamencemega", item="salamencite", active=True, fainted=False, current_hp_fraction=1.0, status=None, moves={}
            ),
            "p1: Sneasler": SimpleNamespace(
                species="sneasler", item=None, active=True, fainted=False, current_hp_fraction=0.01, status=None, moves={}
            ),
        }
    )
    our_it, our_ab, _, _ = _our_public_knowledge(fake_battle, packed, memory=mem2)
    assert our_ab[0] == "aerilate", f"expected aerilate after Mega, got {our_ab[0]}"
    assert our_it[1] is None, f"consumed Focus Sash must stay None, got {our_it[1]}"

    # 4. Unrevealed brought Gholdengo (single-ability Good as Gold) must NOT leak `is_revealed=1.0`.
    b = pikapro.Battle("championsvgc", [packed, packed], seed=[1, 2, 3, 4], log=True)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    os_ = b.encode_state_rich(
        0,
        foe_known_items=[None] * 6,
        foe_known_abilities=[None, None, "goodasgold", None, None, None],
        foe_known_moves=[[]] * 6,
        foe_revealed=[True, True, False, False, False, False],
    )
    foe_gholdengo = (6 + 2) * RICH_MON_WIDTH
    assert float(os_[foe_gholdengo + 2]) > 0.0  # Good as Gold known
    assert float(os_[foe_gholdengo + 12]) == 0.0, "unrevealed Gholdengo must NOT leak is_revealed"
    assert float(os_[foe_gholdengo + 14]) == 0.0, "unrevealed Gholdengo must NOT leak is_brought"


def test_patch_public_pivot_force_switch_never_switches_into_self() -> None:
    """Round 2 P0 regression: an alive pivoting mon (Parting Shot / U-turn / Flip Turn)
    with `force_switch=[False, True]` must remain in `active_slots` so pikapro only offers
    genuine bench switches (3/4), never switching the active mon back into itself (`switch 2`).
    """
    from vgc.rl.rust_search import _solve_2x2_zero_sum

    packed = (
        "Gardevoir||Gardevoirite|Trace|Moonblast,Psychic,DazzlingGleam,Protect|Timid|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FakeOut,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|]"
        "Basculegion||ChoiceScarf|Adaptability|WaveCrash,LastRespects,AquaJet,FlipTurn|Adamant|,32,,,,32||||50|]"
        "Indeedee-F||SafetyGoggles|PsychicSurge|FollowMe,HelpingHand,Psychic,TrickRoom|Bold|32,,32,,2,||||50|]"
        "Tyranitar||Leftovers|SandStream|RockSlide,Crunch,LowKick,Protect|Adamant|32,32,,,,2||||50|]"
        "Rillaboom||MiracleSeed|GrassySurge|FakeOut,GrassyGlide,WoodHammer,Uturn|Adamant|32,32,2,,,||||50|"
    )
    b = pikapro.Battle("championsvgc", [packed, packed], seed=[1, 2, 3, 4], log=False)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    b.patch_public(
        {
            "turn": 3,
            "sides": [
                {
                    "active_species": ["gardevoir", "incineroar"],
                    "force_switch": [False, True],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [
                        {"species": "gardevoir", "hp_pct": 80, "active_turns": 2},
                        {"species": "incineroar", "hp_pct": 90, "active_turns": 2},
                    ],
                },
                {
                    "active_species": ["gardevoir", "incineroar"],
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
    legal = [str(c) for c in b.legal_choices(0)]
    assert legal, "must have legal bench switches"
    assert all("switch 2" not in c and "switch 1" not in c for c in legal), f"illegal self-switch offered: {legal}"

    # 3x3 zero-sum Rock-Paper-Scissors value must be ~0.0 (not truncated to 2x2!).
    rps = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
    assert abs(_solve_2x2_zero_sum(rps)) < 0.05


def test_patch_public_recomputes_foe_ability_trapping() -> None:
    """`patch_public` must run `TrapPokemon` so Shadow Tag traps non-Ghosts but not Ghosts."""
    pikapro = pytest.importorskip("pikapro")

    t_gengar = (
        "Gengar||Gengarite|CursedBody|Disable,PerishSong,Protect,ShadowBall|Timid|2,,,32,,32||||50|]"
        "Archaludon||Leftovers|Stamina|FlashCannon,Protect,ElectroShot,DragonPulse|Modest|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FakeOut,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|]"
        "Rillaboom||MiracleSeed|GrassySurge|FakeOut,GrassyGlide,WoodHammer,Uturn|Adamant|32,32,2,,,||||50|]"
        "Tyranitar||ChoiceScarf|SandStream|RockSlide,Crunch,LowKick,Protect|Adamant|32,32,,,,2||||50|]"
        "Pelipper||DampRock|Drizzle|WeatherBall,Hurricane,WideGuard,Protect|Modest|2,,,32,,32||||50|"
    )
    t_target = (
        "Aegislash||FocusSash|StanceChange|Poltergeist,CloseCombat,WideGuard,ShadowSneak|Adamant|32,32,,,,2||||50|]"
        "Rillaboom||LifeOrb|GrassySurge|WoodHammer,GrassyGlide,HighHorsepower,FakeOut|Adamant|32,32,2,,,||||50|]"
        "Garchomp||GarchompiteZ|RoughSkin|DracoMeteor,PowerGem,EarthPower,Protect|Timid|2,,,32,,32||||50|]"
        "Volcarona||GrassySeed|FlameBody|HeatWave,GigaDrain,QuiverDance,Protect|Timid|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FakeOut,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|]"
        "Tyranitar||Leftovers|SandStream|RockSlide,Crunch,LowKick,Protect|Adamant|32,32,,,,2||||50|"
    )
    b = pikapro.Battle("championsvgc", [t_gengar, t_target], seed=[1, 2, 3, 4], log=False)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    b.patch_public(
        {
            "turn": 2,
            "sides": [
                {
                    "active_species": ["gengarmega", "archaludon"],
                    "force_switch": [False, False],
                    "mega_used": True,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [
                        {"species": "gengar", "is_mega": True, "ability": "shadowtag", "hp_pct": 100, "active_turns": 1},
                        {"species": "archaludon", "hp_pct": 100, "active_turns": 1},
                    ],
                },
                {
                    "active_species": ["aegislash", "rillaboom"],
                    "force_switch": [False, False],
                    "mega_used": False,
                    "total_fainted": 0,
                    "fainted_last_turn": False,
                    "side_conditions": [],
                    "mons": [
                        {"species": "aegislash", "hp_pct": 100, "active_turns": 1},
                        {"species": "rillaboom", "hp_pct": 100, "active_turns": 1},
                    ],
                },
            ],
        }
    )
    slot_masks = b.slot_mask(1)
    space = pikapro.ActionSpace("championsvgc")
    sw0 = [i for i in range(6) if slot_masks[0, space.switch_base + i]]
    sw1 = [i for i in range(6) if slot_masks[1, space.switch_base + i]]
    # Ghost-type Aegislash (slot 0) can still switch out; non-Ghost Rillaboom (slot 1) is trapped.
    assert len(sw0) == 2, f"Ghost-type Aegislash should not be trapped by Shadow Tag: {sw0}"
    assert sw1 == [], f"Non-Ghost Rillaboom must be trapped by Mega Gengar Shadow Tag: {sw1}"


def test_upgrade_v2_to_v3_is_bit_exact_and_flows_gradients() -> None:
    """Zero-initialized v2->v3 upgrade must preserve in-battle logits and values
    bit-for-bit, fit preview_head without mutating trunk/turn weights, and pass
    non-zero gradients into both `move_meta_proj` and `speed_proj` on Step 1.
    """
    from vgc.rl.rust_policy import fit_v3_preview_head_from_teacher, upgrade_v2_to_v3

    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()
    b = pikapro.Battle("championsvgc", [t1, t2], seed=[1, 2, 3, 4], log=True)
    b.submit(0, "team 1234")
    b.submit(1, "team 1234")
    obs = torch.from_numpy(b.encode_rich(0)).unsqueeze(0)

    torch.manual_seed(101)
    v2 = RustFactoredPolicy(width=48, layers=2, policy_version=2, ffn_mult=2).eval()
    v3 = upgrade_v2_to_v3(v2)
    assert v3.policy_version == 3
    assert v3.ffn_mult == 2

    with torch.no_grad():
        e2 = v2.encode_board(obs)
        e3 = v3.encode_board(obs)
        v2_val, _ = v2.value_from_context(e2.context)
        v3_val, _ = v3.value_from_context(e3.context)
        l0_2 = v2.slot0_logits(e2)
        l0_3 = v3.slot0_logits(e3)
        l1_2 = v2.slot1_logits(e2, torch.tensor([0]))
        l1_3 = v3.slot1_logits(e3, torch.tensor([0]))
        assert float((v2_val - v3_val).abs().max()) == 0.0
        assert float((l0_2 - l0_3).abs().max()) == 0.0
        assert float((l1_2 - l1_3).abs().max()) == 0.0

    # Fitting preview_head must keep trunk & turn logits bit-exact identical.
    stats = fit_v3_preview_head_from_teacher(
        v3, v2, [t1, t2], boards=128, steps=15, batch_size=64, lr=3e-3, seed=7, threads=2
    )
    assert stats["preview_kl"] >= 0.0
    with torch.no_grad():
        e3_after = v3.encode_board(obs)
        assert float((v2.slot0_logits(e2) - v3.slot0_logits(e3_after)).abs().max()) == 0.0

    # Gradients must immediately flow into both zero-initialized v3 projections.
    v3.train()
    enc = v3.encode_board(obs)
    val, _ = v3.value_from_context(enc.context)
    loss = val.sum() + v3.slot0_logits(enc).sum()
    loss.backward()
    assert v3.move_meta_proj is not None and v3.move_meta_proj.weight.grad is not None
    assert v3.speed_proj is not None and v3.speed_proj.weight.grad is not None
    assert float(v3.move_meta_proj.weight.grad.abs().sum()) > 0.0
    assert float(v3.speed_proj.weight.grad.abs().sum()) > 0.0


def test_train_rust_vec_minibatch_kl_guardrail_and_width_scaled_lr() -> None:
    """Trainer must break mid-epoch when minibatch KL breaches `1.5 * target_kl`,
    and auto-scale peak LR inversely with width/FFN when `--learning-rate` is omitted.
    """
    import random
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "selfplay"))
    import train_rust_vec as trv

    assert trv.resolve_learning_rate(None, width=192, ffn_mult=2, is_warm_start=False) == pytest.approx(1e-4)
    assert trv.resolve_learning_rate(None, width=256, ffn_mult=3, is_warm_start=False) == pytest.approx(5e-5)
    assert trv.resolve_learning_rate(None, width=192, ffn_mult=2, is_warm_start=True) == pytest.approx(3e-5)
    assert trv.resolve_learning_rate(None, width=256, ffn_mult=3, is_warm_start=True) < 2.5e-5
    assert trv.resolve_learning_rate(4.2e-5, width=256, ffn_mult=3, is_warm_start=False) == pytest.approx(4.2e-5)

    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()
    pool = rl.TeamPool([t1, t2], "championsvgc")
    env = rl.VecEnv(
        "championsvgc",
        n=16,
        teams=pool,
        threads=2,
        seed=11,
        opponent="simple_heuristics",
        ots=0.0,
        preview="learned",
        masks="factored",
        observation="rich",
        autoreset="next_step",
    )
    model = RustFactoredPolicy(width=32, layers=1, policy_version=3).eval()
    opt = trv.build_optimizer(model, lr=5e-3, weight_decay=1e-2)
    step = env.reset()
    trajs: list[list[trv.StepRecord]] = [[] for _ in range(16)]
    records: list[tuple[trv.StepRecord, float, float]] = []
    done_g = 0
    while done_g < 24:
        idx = np.flatnonzero(step.needs_action[:, 0])
        act = np.full((16, 2), -1, dtype=np.int64)
        if len(idx) > 0:
            a0, a1, lp, val, is_pv = trv._select_batch_actions(
                model, env.space, step.obs[idx, 0], step.masks[idx, 0], step.min_switches[idx, 0]
            )
            act[idx, 0] = a0
            act[idx, 1] = a1
            for k, ei in enumerate(idx):
                trajs[ei].append(
                    trv.StepRecord(
                        step.obs[ei, 0].copy(),
                        step.masks[ei, 0].copy(),
                        int(step.min_switches[ei, 0]),
                        bool(is_pv[k]),
                        int(a0[k]),
                        int(a1[k]),
                        float(lp[k]),
                        float(val[k]),
                    )
                )
        step = env.step(act)
        for ei in np.flatnonzero(step.done):
            trv._finish_trajectory(trajs[ei], float(step.reward[ei, 0]), gamma=1.0, gae_lambda=0.95, out_records=records)
            trajs[ei].clear()
            done_g += 1

    batch = trv._collate_samples(records)
    # Tight target_kl + large LR must trip early stopping before completing epoch 2.
    out = trv.ppo_update(
        model, opt, env.space, batch, random.Random(3), epochs=3, batch_size=32, target_kl=1e-4
    )
    assert out["kl_early_stopped"] == 1.0
    assert out["epochs_run"] == 1.0


def test_distill_rust_policy_wave_smoke() -> None:
    """One wave of policy+value distillation from a v2 teacher to a wider v3 student
    must run cleanly with finite KL, BCE, and value MAE.
    """
    import random

    from selfplay.distill_rust_policy import _collect_teacher_wave, distill_wave
    from selfplay.train_rust_vec import build_optimizer

    t1 = TEAM_PATH.read_text().strip()
    t2 = (TEAM_PATH.parent / "mc_ladder_02.packed.txt").read_text().strip()
    pool = rl.TeamPool([t1, t2], "championsvgc")
    env = rl.VecEnv(
        "championsvgc",
        n=16,
        teams=pool,
        threads=2,
        seed=29,
        opponent="external",
        ots=0.0,
        preview="learned",
        masks="factored",
        observation="rich",
        autoreset="next_step",
    )
    step = env.reset()
    teacher = RustFactoredPolicy(width=32, layers=1, policy_version=2, ffn_mult=2).eval()
    student = RustFactoredPolicy(width=48, layers=1, policy_version=3, ffn_mult=3).eval()
    opt = build_optimizer(student, lr=1e-3, weight_decay=1e-2)

    wave, _, games = _collect_teacher_wave(teacher, env, step, target_games=16, n_self=16)
    assert games >= 16 and len(wave["obs"]) > 0
    m1 = distill_wave(student, teacher, opt, env.space, wave, random.Random(5), epochs=1, batch_size=64, value_weight=1.0)
    m2 = distill_wave(student, teacher, opt, env.space, wave, random.Random(6), epochs=1, batch_size=64, value_weight=1.0)
    assert np.isfinite(m1["loss"]) and np.isfinite(m2["loss"])
    assert m2["loss"] < m1["loss"]
