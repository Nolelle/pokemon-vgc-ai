from __future__ import annotations

from dataclasses import asdict, fields, replace
import json

import numpy as np
import pytest

from vgc.mechanics_state import (
    BattleMechanicsState,
    EffectSnapshot,
    MoveSnapshot,
    PokemonMechanicsState,
    SideMechanicsState,
)
from vgc.rl.mechanics_encoding import (
    MECHANICS_ENCODING_VERSION,
    encode_mechanics_state,
    pad_mechanics_features,
)


def _pokemon(species: str) -> PokemonMechanicsState:
    return PokemonMechanicsState(
        species_id=species,
        base_species_id=species,
        name="nickname",
        level=50,
        gender="f",
        types=("fire", "flying"),
        base_types=("fire", "flying"),
        current_hp=101,
        max_hp=153,
        fainted=False,
        active=True,
        revealed=True,
        selected_in_preview=True,
        stats=(("atk", 90), ("def", 98), ("spa", 180), ("spd", 105), ("spe", 152)),
        stat_points=(("hp", 2), ("atk", 0), ("def", 0), ("spa", 32), ("spd", 0), ("spe", 32)),
        individual_values=(("hp", 31), ("atk", 31), ("def", 31), ("spa", 31), ("spd", 31), ("spe", 31)),
        nature="timid",
        boosts=(("atk", 0), ("def", 0), ("spa", 1), ("spd", 0), ("spe", -1), ("accuracy", 0), ("evasion", 0)),
        status="slp",
        status_counter=2,
        effects=(EffectSnapshot("confusion", turns=2, raw_value=2),),
        item_id="charizarditey",
        item_state="known",
        item_known=True,
        ability_id="drought",
        base_ability_id="blaze",
        temporary_ability_id=None,
        forme_change_ability_id="drought",
        ability_known=True,
        base_move_ids=("heatwave", "protect"),
        moves=(
            MoveSnapshot("heatwave", 14, 16, False, None, True),
            MoveSnapshot("protect", 8, 8, True, "taunt", False),
        ),
        last_move_id="heatwave",
        mimic_move_id=None,
        preparing_move_id=None,
        preparing_target=None,
        preparing=False,
        must_recharge=False,
        protect_counter=1,
        first_turn=False,
        transformed=False,
        mega_evolved=True,
        terastallized=False,
        tera_type=None,
        weight=100.5,
    )


def _side(species: str) -> SideMechanicsState:
    return SideMechanicsState(
        pokemon=(_pokemon(species),),
        active_species=(species, None),
        side_conditions=(EffectSnapshot("tailwind", turns=2, raw_value=2),),
        force_switch=(False, True),
        trapped=(False, False),
        maybe_trapped=(False, True),
        can_mega_evolve=(False, False),
        used_mega_evolution=True,
        can_dynamax=(False, False),
        used_dynamax=False,
        can_tera=(False, False),
        used_tera=False,
        can_z_move=(False, False),
        used_z_move=False,
    )


def _state() -> BattleMechanicsState:
    return BattleMechanicsState(
        format_id="gen9championsvgc2026regmb",
        generation=9,
        game_type="doubles",
        turn=4,
        max_team_size=6,
        team_size=4,
        team_preview=False,
        commanding=False,
        reviving=False,
        waiting=False,
        finished=False,
        won=False,
        lost=False,
        fields=(EffectSnapshot("trickroom", turns=3, raw_value=3),),
        weather=(EffectSnapshot("sunnyday", turns=4, raw_value=4),),
        available_moves=(("heatwave", "protect"), ()),
        available_switches=(("incineroar",), ()),
        valid_order_count=12,
        last_request_json='{"active":[]}',
        our_preview_species=("charizard", "farigiraf"),
        opponent_preview_species=("garchomp", "incineroar"),
        our_side=_side("charizardmegay"),
        opponent_side=_side("garchomp"),
    )


def test_mechanics_encoding_is_lossless_and_versioned() -> None:
    state = _state()
    encoded = encode_mechanics_state(state)
    assert encoded.schema_version == MECHANICS_ENCODING_VERSION
    assert encoded.decoded() == json.loads(json.dumps(asdict(state)))
    assert encoded.digest == encode_mechanics_state(state).digest


def test_every_top_level_snapshot_field_is_present_in_neural_input() -> None:
    decoded = encode_mechanics_state(_state()).decoded()
    assert set(decoded) == {field.name for field in fields(BattleMechanicsState)}
    assert set(decoded["our_side"]) == {field.name for field in fields(SideMechanicsState)}
    assert set(decoded["our_side"]["pokemon"][0]) == {
        field.name for field in fields(PokemonMechanicsState)
    }
    assert set(decoded["our_side"]["pokemon"][0]["moves"][0]) == {
        field.name for field in fields(MoveSnapshot)
    }
    assert set(decoded["fields"][0]) == {field.name for field in fields(EffectSnapshot)}


def test_different_mechanics_state_cannot_collapse_to_same_tokens() -> None:
    state = _state()
    changed = replace(state, turn=state.turn + 1)
    assert not np.array_equal(
        encode_mechanics_state(state).tokens,
        encode_mechanics_state(changed).tokens,
    )


def test_padding_never_truncates_and_marks_only_real_tokens() -> None:
    first = encode_mechanics_state(_state())
    second = encode_mechanics_state(replace(_state(), last_request_json="x" * 1000))
    tokens, mask = pad_mechanics_features([first, second])
    assert tokens.shape == mask.shape == (2, len(second.tokens))
    assert mask[0].sum() == len(first.tokens)
    assert mask[1].all()
    assert np.array_equal(tokens[0, : len(first.tokens)], first.tokens)
    assert np.all(tokens[0, len(first.tokens) :] == 0)


def test_mechanics_enabled_model_requires_and_reads_complete_snapshot() -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.encoding import INDEX_DIM, SLOT_FEATURE_DIM, STATE_SCALAR_DIM
    from vgc.rl.encoding import HISTORY_SCALAR_DIM
    from vgc.rl.model import CandidatePolicyValueNet

    model = CandidatePolicyValueNet(use_mechanics_features=True)
    args = (
        torch.zeros((1, INDEX_DIM), dtype=torch.long),
        torch.zeros((1, STATE_SCALAR_DIM + SLOT_FEATURE_DIM)),
        torch.zeros((1, HISTORY_SCALAR_DIM)),
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.zeros((1, 2, 2), dtype=torch.long),
        torch.zeros((1, 2, 2, 4)),
        torch.ones((1, 2), dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="mechanics"):
        model(*args)
    features = encode_mechanics_state(_state())
    tokens, mask = pad_mechanics_features([features])
    logits, values = model(
        *args,
        mechanics_tokens=torch.as_tensor(tokens),
        mechanics_mask=torch.as_tensor(mask),
    )
    assert logits.shape == (1, 2)
    assert values.shape == (1,)
