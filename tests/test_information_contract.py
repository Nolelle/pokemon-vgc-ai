from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.battle_memory import BattleMemory, DamageObservation, SpeedObservation
from vgc.config import REPO_ROOT
from vgc.damage import FieldState, PokemonState, damage_range
from vgc.opponent_belief import build_opponent_beliefs
from vgc.rl.encoding import (
    INFORMATION_INDEX_DIM,
    INFORMATION_SCALAR_DIM,
    encode_information_context,
    TACTICAL_FEATURE_DIM,
)


def _mon(species: str, *, moves=(), item=None, ability=None, hp=1.0):
    return SimpleNamespace(
        species=species,
        moves={move: None for move in moves},
        item=item,
        ability=ability,
        current_hp_fraction=hp,
        current_hp=100,
        fainted=False,
        boosts={},
        status=None,
        evs=None,
        nature=None,
    )


def _battle_with_complete_own_team():
    packed = (REPO_ROOT / "teams" / "phase2_mirror.packed.txt").read_text().strip()
    builders = Teambuilder.parse_packed_team(packed)
    mons = []
    for entry in builders:
        species = entry.species or entry.nickname
        mons.append(
            _mon(
                species,
                moves=entry.moves,
                item=entry.item,
                ability=entry.ability,
            )
        )
    opponent_preview = [_mon("Garchomp"), _mon("Incineroar")]
    return SimpleNamespace(
        team={f"p1: {entry.nickname or entry.species}": mon for entry, mon in zip(builders, mons)},
        teambuilder_team=builders,
        active_pokemon=mons[:2],
        opponent_team={},
        opponent_active_pokemon=[None, None],
        teampreview_opponent_team=opponent_preview,
    )


def test_information_contract_contains_all_six_own_sets_and_fixed_shapes() -> None:
    battle = _battle_with_complete_own_team()
    encoded = encode_information_context(battle, BattleMemory("test"))
    assert encoded.indices.shape == (INFORMATION_INDEX_DIM,)
    assert encoded.scalars.shape == (INFORMATION_SCALAR_DIM,)
    # Six own slots each begin with a presence marker.
    own = encoded.scalars[: 6 * 28].reshape(6, 28)
    assert np.all(own[:, 0] == 1.0)
    assert np.all(own[:2, 1] == 1.0)
    assert np.all(own[2:, 1] == 0.0)


def test_own_bench_move_and_spread_change_the_learned_input() -> None:
    first = _battle_with_complete_own_team()
    second = deepcopy(first)
    second.teambuilder_team[5].moves[0] = "Protect"
    second.teambuilder_team[5].evs = [32, 32, 0, 0, 0, 2]

    first_encoded = encode_information_context(first, BattleMemory("first"))
    second_encoded = encode_information_context(second, BattleMemory("second"))
    assert not np.array_equal(first_encoded.indices, second_encoded.indices)
    assert not np.array_equal(first_encoded.scalars, second_encoded.scalars)


def test_private_opponent_truth_cannot_change_the_encoded_input() -> None:
    first = _battle_with_complete_own_team()
    second = deepcopy(first)
    # A simulator-only attribute must be ignored.  Neither object has appeared or
    # revealed a move/item/ability in the player's public battle view.
    first.private_opponent_team = {"garchomp": {"moves": ["Earthquake"], "spe": 200}}
    second.private_opponent_team = {"garchomp": {"moves": ["Protect"], "spe": 1}}

    first_encoded = encode_information_context(first, BattleMemory("first"))
    second_encoded = encode_information_context(second, BattleMemory("second"))
    assert np.array_equal(first_encoded.indices, second_encoded.indices)
    assert np.array_equal(first_encoded.scalars, second_encoded.scalars)


def test_revealed_opponent_move_changes_the_input() -> None:
    battle = _battle_with_complete_own_team()
    before = encode_information_context(battle, BattleMemory("before"))
    appeared = _mon("Garchomp", moves=("Earthquake",))
    battle.opponent_team = {"p2: Garchomp": appeared}
    battle.opponent_active_pokemon = [appeared, None]
    after = encode_information_context(battle, BattleMemory("after"))
    assert not np.array_equal(before.indices, after.indices)
    assert not np.array_equal(before.scalars, after.scalars)


def test_move_order_reweights_speed_hypotheses() -> None:
    battle = _battle_with_complete_own_team()
    memory = BattleMemory("speed")
    memory.speed_observations.append(
        SpeedObservation("garchomp", threshold=145.0, relation="at_least")
    )
    usage = {
        "garchomp": [
            {"sp": {"spe": 0}, "nature": "brave", "weight": 0.5},
            {"sp": {"spe": 32}, "nature": "jolly", "weight": 0.5},
        ]
    }
    belief = build_opponent_beliefs(
        battle, memory, usage=usage, set_priors={}
    )[0]
    by_nature = {hypothesis.nature: hypothesis.probability for hypothesis in belief.hypotheses}
    assert by_nature["jolly"] > by_nature["brave"]


def test_observed_damage_reweights_attack_hypotheses() -> None:
    target = PokemonState("incineroar", sp_spread={"hp": 32}, nature="careful")
    high = PokemonState("garchomp", sp_spread={"atk": 32}, nature="adamant")
    observed = damage_range(high, target, "earthquake", FieldState(is_doubles=True))
    memory = BattleMemory("damage")
    memory.damage_observations.append(
        DamageObservation(
            opponent_species="garchomp",
            move_id="earthquake",
            target_species="incineroar",
            damage_fraction=observed.max_percent / 100.0,
            target_state=target,
        )
    )
    battle = _battle_with_complete_own_team()
    usage = {
        "garchomp": [
            {"sp": {"atk": 0}, "nature": "modest", "weight": 0.5},
            {"sp": {"atk": 32}, "nature": "adamant", "weight": 0.5},
        ]
    }
    belief = build_opponent_beliefs(
        battle, memory, usage=usage, set_priors={}
    )[0]
    by_nature = {hypothesis.nature: hypothesis.probability for hypothesis in belief.hypotheses}
    assert by_nature["adamant"] > by_nature["modest"]


def test_information_enabled_model_requires_and_accepts_contract_tensors() -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.encoding import INDEX_DIM, SLOT_FEATURE_DIM, STATE_SCALAR_DIM
    from vgc.rl.encoding import HISTORY_SCALAR_DIM
    from vgc.rl.model import CandidatePolicyValueNet

    model = CandidatePolicyValueNet(use_information_features=True)
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
    with pytest.raises(ValueError, match="information"):
        model(*args)
    logits, values = model(
        *args,
        information_indices=torch.zeros((1, INFORMATION_INDEX_DIM), dtype=torch.long),
        information_scalars=torch.zeros((1, INFORMATION_SCALAR_DIM)),
    )
    assert logits.shape == (1, 2)
    assert values.shape == (1,)


def test_tactical_enabled_model_requires_and_accepts_candidate_facts() -> None:
    torch = pytest.importorskip("torch")
    from vgc.bc.encoding import INDEX_DIM, SLOT_FEATURE_DIM, STATE_SCALAR_DIM
    from vgc.rl.encoding import HISTORY_SCALAR_DIM
    from vgc.rl.model import CandidatePolicyValueNet

    model = CandidatePolicyValueNet(use_tactical_features=True)
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
    with pytest.raises(ValueError, match="tactical"):
        model(*args)
    logits, values = model(
        *args, tactical_features=torch.zeros((1, 2, TACTICAL_FEATURE_DIM))
    )
    assert logits.shape == (1, 2)
    assert values.shape == (1,)
