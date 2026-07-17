"""Unit tests for `vgc.bc.policy` -- the BC v2 checkpoint deployed as a candidate
re-ranker. Split into three groups:

- `battle_state_record`: torch-free, exercised against hand-built fake poke-env objects
  (mirrors tests/test_team_preview.py's `_FakeMon`/`_FakeBattle` style) -- including a
  direct cross-check that this adapter's output runs through the SAME `encode_state` path
  as a hand-built schema-2 fixture record and produces IDENTICAL arrays (field-name drift
  between this adapter and `vgc.replay_parse`'s schema is the #1 silent-failure risk here).
- Token-mapping (`_single_tokens`/`_order_tokens`/`_target_token`): also torch-free, using
  a hand-built `BcPolicy` (its `model` field is unused by these tests, so no torch/model
  weights needed at all).
- `load_bc_policy`/`score_orders`: needs a real forward pass, so these use
  `pytest.importorskip("torch")` locally (this module's own import must stay torch-free
  regardless -- see vgc.bc.policy's own module docstring for why).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

from vgc.bc.encoding import (
    ABILITY_VOCAB,
    ENCODER_LAYOUT_VERSION,
    ITEM_VOCAB,
    MOVE_VOCAB,
    SPECIES_VOCAB,
    TARGET_VOCAB,
    encode_state,
)
from vgc.bc.policy import (
    BcPolicy,
    _order_tokens,
    _single_tokens,
    _target_token,
    battle_state_record,
    load_bc_policy,
    score_orders,
)
from vgc.data import load_moves
from vgc.evaluator import ScoredOrder
from vgc.models import PolicyConfig

# --- fake poke-env stand-ins (mirrors tests/test_team_preview.py's _FakeMon/_FakeBattle,
# --- extended with `fainted` since battle_state_record needs it) --------------------


@dataclass
class _FakeMon:
    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict = field(default_factory=dict)
    status: Any = None
    current_hp_fraction: float = 1.0
    fainted: bool = False
    moves: dict = field(default_factory=dict)


class _FakeBattle:
    def __init__(
        self,
        *,
        our_active,
        opp_active,
        our_bench=None,
        opp_bench=None,
        side_conditions=None,
        opponent_side_conditions=None,
        fields=None,
        weather=None,
        terrain_fields=None,
        turn=1,
    ):
        self.active_pokemon = our_active
        self.opponent_active_pokemon = opp_active
        # Real poke-env's team/opponent_team dicts hold only ACTUAL Pokemon (never a
        # `None` placeholder the way the fixed-size active_pokemon/opponent_active_pokemon
        # lists do for an empty slot) -- filter here to match that contract.
        our_members = [mon for mon in our_active if mon is not None] + (our_bench or [])
        opp_members = [mon for mon in opp_active if mon is not None] + (opp_bench or [])
        self.team = {f"our{i}": mon for i, mon in enumerate(our_members)}
        self.opponent_team = {f"opp{i}": mon for i, mon in enumerate(opp_members)}
        self.side_conditions = side_conditions or []
        self.opponent_side_conditions = opponent_side_conditions or []
        self.fields = fields or []
        self.weather = weather or []
        self.turn = turn


def _fake_single(move_id: str, move_target: int = 0, mega: bool = False):
    from types import SimpleNamespace

    move = Move(move_id, gen=9)
    return SimpleNamespace(
        order=move,
        mega=mega,
        move_target=move_target,
        z_move=False,
        dynamax=False,
        terastallize=False,
    )


def _fake_switch(species: str):
    from types import SimpleNamespace

    target = Pokemon(gen=9, species=species)
    return SimpleNamespace(
        order=target,
        mega=False,
        move_target=0,
        z_move=False,
        dynamax=False,
        terastallize=False,
    )


def _fake_order(first, second):
    from types import SimpleNamespace

    return SimpleNamespace(first_order=first, second_order=second)


def _fake_bc_policy(move_vocab=None, target_vocab=None) -> BcPolicy:
    """A `BcPolicy` with NO real model (only its vocabs matter for the token-mapping
    tests below) -- `model=None` is fine since none of these tests call `.model(...)`.
    """
    return BcPolicy(
        model=None,
        species_vocab=list(SPECIES_VOCAB),
        move_vocab=list(move_vocab if move_vocab is not None else MOVE_VOCAB),
        item_vocab=list(ITEM_VOCAB),
        ability_vocab=list(ABILITY_VOCAB),
        target_vocab=list(target_vocab if target_vocab is not None else TARGET_VOCAB),
        encoder_layout_version=ENCODER_LAYOUT_VERSION,
    )


# --- battle_state_record: field-by-field correctness --------------------------------


def _minimal_battle() -> _FakeBattle:
    our0 = _FakeMon(
        species="garchomp",
        item="lifeorb",
        ability="roughskin",
        boosts={"atk": 1},
        current_hp_fraction=0.8,
        moves={"earthquake": None, "dragonclaw": None},
    )
    our1 = _FakeMon(species="klefki", ability="prankster", moves={"protect": None})
    opp0 = _FakeMon(species="gholdengo", status=Status.BRN, moves={"shadowball": None})
    return _FakeBattle(our_active=[our0, our1], opp_active=[opp0, None])


def test_battle_state_record_our_side_is_exactly_known() -> None:
    record = battle_state_record(_minimal_battle(), None)
    our0 = record["state"]["our"]["active"][0]
    assert our0["species"] == "garchomp"
    assert our0["item"] == "lifeorb"
    assert our0["ability"] == "roughskin"
    assert our0["boosts"] == {"atk": 1}
    assert our0["hp_fraction"] == 0.8
    assert our0["mega"] is False
    assert our0["revealed_moves"] == ["dragonclaw", "earthquake"]


def test_battle_state_record_opp_side_is_revealed_only() -> None:
    record = battle_state_record(_minimal_battle(), None)
    opp0 = record["state"]["opp"]["active"][0]
    assert opp0["species"] == "gholdengo"
    assert opp0["status"] == "brn"
    assert opp0["item"] is None  # never revealed
    assert opp0["ability"] is None  # never revealed
    assert opp0["revealed_moves"] == ["shadowball"]
    assert record["state"]["opp"]["active"][1] is None  # empty slot


def test_battle_state_record_opp_revealed_item_and_ability_pass_through() -> None:
    opp0 = _FakeMon(species="gholdengo", item="airballoon", ability="goodasgold")
    battle = _FakeBattle(our_active=[_FakeMon(species="garchomp"), None], opp_active=[opp0, None])
    record = battle_state_record(battle, None)
    assert record["state"]["opp"]["active"][0]["item"] == "airballoon"
    assert record["state"]["opp"]["active"][0]["ability"] == "goodasgold"


def test_battle_state_record_unknown_item_sentinel_normalizes_to_none() -> None:
    opp0 = _FakeMon(species="gholdengo", item="unknown_item")
    battle = _FakeBattle(our_active=[_FakeMon(species="garchomp"), None], opp_active=[opp0, None])
    record = battle_state_record(battle, None)
    assert record["state"]["opp"]["active"][0]["item"] is None


def test_battle_state_record_bench_excludes_active_and_fainted() -> None:
    active = _FakeMon(species="garchomp")
    alive_bench = _FakeMon(species="incineroar", current_hp_fraction=0.5)
    fainted_bench = _FakeMon(species="sylveon", fainted=True)
    battle = _FakeBattle(
        our_active=[active, None],
        opp_active=[None, None],
        our_bench=[alive_bench, fainted_bench],
    )
    bench = battle_state_record(battle, None)["state"]["our"]["bench"]
    assert bench == [{"species_id": "incineroar", "hp_fraction": 0.5, "status": None}]


def test_battle_state_record_mega_detected_and_resolved_to_base_species() -> None:
    mega_mon = _FakeMon(species="charizardmegax", ability="toughclaws")
    battle = _FakeBattle(our_active=[mega_mon, None], opp_active=[None, None])
    our0 = battle_state_record(battle, None)["state"]["our"]["active"][0]
    assert our0["species"] == "charizard"
    assert our0["mega"] is True


def test_battle_state_record_side_conditions_include_tailwind_and_screens() -> None:
    battle = _FakeBattle(
        our_active=[_FakeMon(species="garchomp"), None],
        opp_active=[None, None],
        side_conditions=[SideCondition.TAILWIND, SideCondition.REFLECT],
    )
    conditions = battle_state_record(battle, None)["state"]["our"]["side_conditions"]
    assert conditions == ["reflect", "tailwind"]


def test_battle_state_record_field_dict() -> None:
    battle = _FakeBattle(
        our_active=[_FakeMon(species="garchomp"), None],
        opp_active=[None, None],
        weather=[Weather.SUNNYDAY],
        turn=7,
    )
    field_dict = battle_state_record(battle, None)["state"]["field"]
    assert field_dict == {"weather": "sun", "terrain": None, "trick_room": False, "turn": 7}


# --- adapter output must encode IDENTICALLY to a hand-built schema-2 fixture record -


def test_battle_state_record_encodes_identically_to_schema2_fixture() -> None:
    battle = _FakeBattle(
        our_active=[
            _FakeMon(
                species="garchomp",
                item="lifeorb",
                ability="roughskin",
                boosts={"atk": 1},
                current_hp_fraction=0.8,
                moves={"earthquake": None, "dragonclaw": None},
            ),
            _FakeMon(species="klefki", ability="prankster", moves={"protect": None}),
        ],
        opp_active=[
            _FakeMon(species="gholdengo", status=Status.BRN, moves={"shadowball": None}),
            None,
        ],
        our_bench=[_FakeMon(species="incineroar", current_hp_fraction=1.0)],
        side_conditions=[SideCondition.REFLECT],
        weather=[Weather.SUNNYDAY],
        turn=5,
    )

    fixture_record = {
        "state": {
            "our": {
                "active": [
                    {
                        "species": "garchomp",
                        "hp_fraction": 0.8,
                        "status": None,
                        "boosts": {"atk": 1},
                        "item": "lifeorb",
                        "ability": "roughskin",
                        "mega": False,
                        "revealed_moves": ["dragonclaw", "earthquake"],
                    },
                    {
                        "species": "klefki",
                        "hp_fraction": 1.0,
                        "status": None,
                        "boosts": {},
                        "item": None,
                        "ability": "prankster",
                        "mega": False,
                        "revealed_moves": ["protect"],
                    },
                ],
                "bench": [{"species_id": "incineroar", "hp_fraction": 1.0, "status": None}],
                "side_conditions": ["reflect"],
            },
            "opp": {
                "active": [
                    {
                        "species": "gholdengo",
                        "hp_fraction": 1.0,
                        "status": "brn",
                        "boosts": {},
                        "item": None,
                        "ability": None,
                        "mega": False,
                        "revealed_moves": ["shadowball"],
                    },
                    None,
                ],
                "bench": [],
                "side_conditions": [],
            },
            "field": {"weather": "sun", "terrain": None, "trick_room": False, "turn": 5},
        }
    }

    live_encoded = encode_state(battle_state_record(battle, None))
    fixture_encoded = encode_state(fixture_record)

    assert live_encoded.keys() == fixture_encoded.keys()
    for key in live_encoded:
        np.testing.assert_array_equal(live_encoded[key], fixture_encoded[key], err_msg=key)


# --- token mapping: move w/ target, spread, switch, pass, mega, out-of-vocab --------


def test_target_token_single_target_move_maps_to_opp_slot() -> None:
    move_data = load_moves()["dragonclaw"]
    assert _target_token(move_data, 1) == "opp0"
    assert _target_token(move_data, 2) == "opp1"


def test_target_token_single_target_move_at_ally_maps_to_ally() -> None:
    move_data = load_moves()["dragonclaw"]
    assert _target_token(move_data, -1) == "ally"
    assert _target_token(move_data, -2) == "ally"


def test_target_token_spread_move_maps_to_spread() -> None:
    assert _target_token(load_moves()["earthquake"], 0) == "spread"
    assert _target_token(load_moves()["icywind"], 0) == "spread"


def test_target_token_self_targeting_status_move_maps_to_self_or_field() -> None:
    assert _target_token(load_moves()["protect"], 0) == "self_or_field"


def test_single_tokens_move_with_target() -> None:
    policy = _fake_bc_policy()
    single = _fake_single("dragonclaw", move_target=1)
    move_idx, target_idx, skip, reason = _single_tokens(single, policy)
    assert skip is False
    assert reason is None
    assert policy.move_vocab[move_idx] == "dragonclaw"
    assert policy.target_vocab[target_idx] == "opp0"


def test_single_tokens_switch() -> None:
    policy = _fake_bc_policy()
    single = _fake_switch("klefki")
    move_idx, target_idx, skip, reason = _single_tokens(single, policy)
    assert skip is False
    assert policy.move_vocab[move_idx] == "<switch>"
    assert policy.target_vocab[target_idx] == "<none>"


def test_single_tokens_pass() -> None:
    policy = _fake_bc_policy()
    move_idx, target_idx, skip, reason = _single_tokens(None, policy)
    assert skip is False
    assert policy.move_vocab[move_idx] == "<pass>"
    assert policy.target_vocab[target_idx] == "<none>"


def test_single_tokens_mega_flag_does_not_change_tokens() -> None:
    policy = _fake_bc_policy()
    plain = _single_tokens(_fake_single("earthquake"), policy)
    mega = _single_tokens(_fake_single("earthquake", mega=True), policy)
    assert plain[:2] == mega[:2]


def test_single_tokens_out_of_vocab_move_is_skipped() -> None:
    # A restricted vocab that deliberately omits "earthquake" -- exercises the
    # skip-BC-adjustment path without needing an exotic real out-of-vocab move id.
    restricted_vocab = ["<pad>", "<switch>", "<pass>", "<unk>", "protect"]
    policy = _fake_bc_policy(move_vocab=restricted_vocab)
    single = _fake_single("earthquake")
    move_idx, target_idx, skip, reason = _single_tokens(single, policy)
    assert skip is True
    assert reason is not None


def test_order_tokens_skip_propagates_from_either_slot() -> None:
    restricted_vocab = ["<pad>", "<switch>", "<pass>", "<unk>", "protect"]
    policy = _fake_bc_policy(move_vocab=restricted_vocab)
    order = _fake_order(_fake_single("protect"), _fake_single("earthquake"))
    _move_tokens, _target_tokens, skip, reason = _order_tokens(order, policy)
    assert skip is True
    assert reason is not None


# --- load_bc_policy: never raises, refuses gracefully --------------------------------


def test_load_bc_policy_missing_checkpoint_returns_none(tmp_path) -> None:
    result = load_bc_policy(tmp_path / "does-not-exist.pt")
    assert result is None


def test_load_bc_policy_torch_unavailable_returns_none(tmp_path, monkeypatch) -> None:
    import vgc.bc.policy as policy_module

    monkeypatch.setattr(policy_module, "_TORCH_AVAILABLE", False)
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"")

    result = load_bc_policy(checkpoint_path)

    assert result is None


def test_load_bc_policy_layout_version_mismatch_refuses(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    import vgc.bc.policy as policy_module

    checkpoint_path = tmp_path / "stale.pt"
    checkpoint_path.write_bytes(b"placeholder")

    def fake_load(path, map_location=None, weights_only=None):  # noqa: ARG001
        return {"encoder_layout_version": "bc-encoding-v1"}

    monkeypatch.setattr(policy_module.torch, "load", fake_load)

    result = load_bc_policy(checkpoint_path)

    assert result is None


def test_load_bc_policy_corrupt_checkpoint_returns_none_not_raise(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    import vgc.bc.policy as policy_module

    checkpoint_path = tmp_path / "corrupt.pt"
    checkpoint_path.write_bytes(b"not a real checkpoint")

    def raising_load(path, map_location=None, weights_only=None):  # noqa: ARG001
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(policy_module.torch, "load", raising_load)

    result = load_bc_policy(checkpoint_path)

    assert result is None


def test_load_bc_policy_caches_per_path(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    import vgc.bc.policy as policy_module
    from vgc.bc.model import BcPolicyNet

    checkpoint_path = tmp_path / "tiny.pt"
    model = BcPolicyNet()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "species_vocab": SPECIES_VOCAB,
            "move_vocab": MOVE_VOCAB,
            "item_vocab": ITEM_VOCAB,
            "ability_vocab": ABILITY_VOCAB,
            "target_vocab": TARGET_VOCAB,
            "encoder_layout_version": ENCODER_LAYOUT_VERSION,
        },
        checkpoint_path,
    )
    policy_module._POLICY_CACHE.clear()

    first = load_bc_policy(checkpoint_path)
    second = load_bc_policy(checkpoint_path)

    assert first is not None
    assert first is second  # same object -- cached, not reloaded


# --- score_orders: blend math + top-K-only reranking invariant ----------------------


def _tiny_checkpoint_policy(tmp_path) -> BcPolicy:
    """A real (randomly initialized, untrained) BcPolicyNet wrapped in a BcPolicy --
    enough to exercise score_orders' actual forward pass and blend arithmetic without
    needing the real (slow-ish to load, and its exact predictions are irrelevant here)
    trained checkpoint.
    """
    torch = pytest.importorskip("torch")
    from vgc.bc.model import BcPolicyNet

    torch.manual_seed(0)
    model = BcPolicyNet()
    model.eval()
    return BcPolicy(
        model=model,
        species_vocab=list(SPECIES_VOCAB),
        move_vocab=list(MOVE_VOCAB),
        item_vocab=list(ITEM_VOCAB),
        ability_vocab=list(ABILITY_VOCAB),
        target_vocab=list(TARGET_VOCAB),
        encoder_layout_version=ENCODER_LAYOUT_VERSION,
    )


def _score_battle() -> _FakeBattle:
    return _FakeBattle(
        our_active=[
            _FakeMon(species="garchomp", moves={"earthquake": None, "dragonclaw": None}),
            _FakeMon(species="klefki", moves={"protect": None}),
        ],
        opp_active=[
            _FakeMon(species="gholdengo", moves={"shadowball": None}),
            _FakeMon(species="incineroar", moves={"flareblitz": None}),
        ],
    )


def test_score_orders_returns_input_unchanged_when_policy_is_none() -> None:
    scored = [ScoredOrder(order=_fake_order(_fake_single("earthquake"), None), score=10.0)]
    result = score_orders(None, _score_battle(), scored, PolicyConfig())
    assert result is scored


def test_score_orders_returns_input_unchanged_when_scored_orders_empty(tmp_path) -> None:
    policy = _tiny_checkpoint_policy(tmp_path)
    result = score_orders(policy, _score_battle(), [], PolicyConfig())
    assert result == []


def test_score_orders_blends_heuristic_and_bc_logprob(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    policy = _tiny_checkpoint_policy(tmp_path)
    battle = _score_battle()
    config = PolicyConfig(bc_blend_weight=10.0, bc_rerank_top_k=10)

    order = _fake_order(_fake_single("earthquake", move_target=0), _fake_single("protect"))
    scored = [ScoredOrder(order=order, score=100.0)]

    result = score_orders(policy, battle, scored, config)

    assert len(result) == 1
    # Independently recompute the same forward pass and check the blend arithmetic.
    from vgc.bc.encoding import SLOT_FEATURE_DIM, flatten_state
    from vgc.bc.policy import battle_state_record, _order_tokens

    record = battle_state_record(battle, config)
    index_array, scalar_array = flatten_state(encode_state(record))
    index_batch = np.tile(index_array, (2, 1))
    slot_onehots = np.eye(SLOT_FEATURE_DIM, dtype=np.float32)
    scalar_batch = np.concatenate([np.tile(scalar_array, (2, 1)), slot_onehots], axis=1)
    with torch.no_grad():
        move_logits, target_logits = policy.model(
            torch.from_numpy(index_batch), torch.from_numpy(scalar_batch)
        )
        move_logp = torch.log_softmax(move_logits, dim=-1)
        target_logp = torch.log_softmax(target_logits, dim=-1)
    move_tokens, target_tokens, skip, _reason = _order_tokens(order, policy)
    assert skip is False
    expected_logprob = 0.0
    for slot in (0, 1):
        expected_logprob += float(move_logp[slot, move_tokens[slot]])
        expected_logprob += float(target_logp[slot, target_tokens[slot]])
    expected_score = 100.0 + config.bc_blend_weight * expected_logprob

    assert result[0].score == pytest.approx(expected_score)
    assert result[0].breakdown["bc_logprob"] == pytest.approx(expected_logprob, abs=1e-3)


def test_score_orders_reranks_only_top_k_and_tail_never_overtakes(tmp_path) -> None:
    policy = _tiny_checkpoint_policy(tmp_path)
    battle = _score_battle()
    config = PolicyConfig(bc_blend_weight=1000.0, bc_rerank_top_k=2)

    head_a = _fake_order(_fake_single("earthquake", move_target=0), _fake_single("protect"))
    head_b = _fake_order(_fake_single("dragonclaw", move_target=1), _fake_single("protect"))
    # Deliberately given a much HIGHER raw score than the head block, but placed outside
    # bc_rerank_top_k=2 -- it must still end up ranked below the (rescored) head block.
    tail_order = _fake_order(_fake_single("dragonclaw", move_target=2), _fake_single("protect"))

    scored = [
        ScoredOrder(order=head_a, score=10.0),
        ScoredOrder(order=head_b, score=5.0),
        ScoredOrder(order=tail_order, score=10_000.0),
    ]

    result = score_orders(policy, battle, scored, config)

    assert len(result) == 3
    assert result[-1].order is tail_order
    reranked_scores = [entry.score for entry in result[:2]]
    assert result[-1].score < min(reranked_scores)
    # Tail keeps its position/identity, just demoted in score -- not dropped.
    result_ids = {id(entry.order) for entry in result}
    assert result_ids == {id(head_a), id(head_b), id(tail_order)}


def test_score_orders_out_of_vocab_candidate_keeps_heuristic_score() -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.model import BcPolicyNet

    torch.manual_seed(0)
    model = BcPolicyNet()
    model.eval()
    restricted_vocab = ["<pad>", "<switch>", "<pass>", "<unk>", "protect"]
    policy = BcPolicy(
        model=model,
        species_vocab=list(SPECIES_VOCAB),
        move_vocab=restricted_vocab,
        item_vocab=list(ITEM_VOCAB),
        ability_vocab=list(ABILITY_VOCAB),
        target_vocab=list(TARGET_VOCAB),
        encoder_layout_version=ENCODER_LAYOUT_VERSION,
    )
    battle = _score_battle()
    config = PolicyConfig(bc_rerank_top_k=10)
    order = _fake_order(_fake_single("earthquake"), _fake_single("protect"))
    scored = [ScoredOrder(order=order, score=42.0)]

    result = score_orders(policy, battle, scored, config)

    assert result[0].score == 42.0
    assert result[0].breakdown["bc_skipped"] is True
