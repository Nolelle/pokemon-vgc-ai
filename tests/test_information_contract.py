from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.battle_memory import BattleMemory, DamageObservation, SpeedObservation
from vgc.config import REPO_ROOT
from vgc.damage import FieldState, PokemonState, damage_range
from vgc.opponent_belief import build_opponent_beliefs, information_boundary_summary
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


def test_information_ledger_labels_own_facts_and_opponent_uncertainty() -> None:
    battle = _battle_with_complete_own_team()
    ledger = information_boundary_summary(battle, BattleMemory("ledger"))

    assert ledger["opponent_sheet_policy"] == "reject"
    assert ledger["own_team"]["status"] == "known"
    assert ledger["own_team"]["set_count"] == 6
    assert ledger["own_team"]["fields"] == [
        "species",
        "moves",
        "item",
        "ability",
        "nature",
        "stat_points",
    ]
    assert len(ledger["own_team"]["sets"]) == 6
    charizard = ledger["own_team"]["sets"][0]
    assert charizard["species"] == "charizard"
    assert charizard["item"] == "charizarditey"
    assert charizard["ability"] == "blaze"
    assert charizard["nature"] == "modest"
    assert charizard["stat_points"] == {
        "hp": 10,
        "atk": 0,
        "def": 0,
        "spa": 32,
        "spd": 0,
        "spe": 24,
    }
    assert len(ledger["opponent"]) == 2
    for opponent in ledger["opponent"]:
        assert opponent["species"]["status"] == "known"
        assert opponent["brought"]["status"] == "unknown"
        assert opponent["moves"]["known"] == []
        assert opponent["moves"]["unknown_slots"] == 4
        assert opponent["spread_and_nature"]["status"] == "estimated"
        assert opponent["current_state"]["status"] == "unknown"
    assert ledger["opponent_next_action"] == {"status": "unknown"}
    assert ledger["future_random_outcomes"]["status"] == "unknown"


def test_information_ledger_ignores_private_truth_and_promotes_public_reveals() -> None:
    first = _battle_with_complete_own_team()
    second = deepcopy(first)
    first.private_opponent_team = {"garchomp": {"item": "Choice Scarf"}}
    second.private_opponent_team = {"garchomp": {"item": "Life Orb"}}
    memory = BattleMemory("ledger")

    assert information_boundary_summary(first, memory) == information_boundary_summary(
        second, memory
    )

    appeared = _mon(
        "Garchomp", moves=("Earthquake",), item="Life Orb", ability="Rough Skin"
    )
    first.opponent_team = {"p2: Garchomp": appeared}
    first.opponent_active_pokemon = [appeared, None]
    revealed = information_boundary_summary(first, memory)
    garchomp = next(
        entry
        for entry in revealed["opponent"]
        if entry["species"]["value"] == "garchomp"
    )
    assert garchomp["brought"] == {"status": "known", "value": True}
    assert garchomp["moves"]["known"] == ["earthquake"]
    assert garchomp["item"] == {"status": "known", "value": "lifeorb"}
    assert garchomp["ability"] == {"status": "known", "value": "roughskin"}


def test_information_ledger_treats_unavailable_public_properties_as_unknown() -> None:
    class IncompleteBattle:
        teambuilder_team = []
        teampreview_opponent_team = []
        opponent_team = {}

        @property
        def opponent_active_pokemon(self):
            raise ValueError("opponent role is not available yet")

    ledger = information_boundary_summary(IncompleteBattle(), BattleMemory("incomplete"))
    assert ledger["own_team"]["set_count"] == 0
    assert ledger["opponent"] == []
    assert ledger["opponent_next_action"] == {"status": "unknown"}


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


def test_joint_set_prior_keeps_observed_item_ability_and_moves_together() -> None:
    battle = _battle_with_complete_own_team()
    priors = {
        "species": {
            "garchomp": {
                "appearances": 100,
                "moves": {"earthquake": 80, "protect": 70, "dragonclaw": 40},
                "items": {"lifeorb": 70, "choicescarf": 30},
                "abilities": {"roughskin": 70, "sandveil": 30},
                "configurations": [
                    {
                        "moves": ["earthquake", "protect"],
                        "item": "lifeorb",
                        "ability": "roughskin",
                        "count": 70,
                    },
                    {
                        "moves": ["dragonclaw", "protect"],
                        "item": "choicescarf",
                        "ability": "sandveil",
                        "count": 30,
                    },
                ],
            }
        }
    }

    belief = build_opponent_beliefs(
        battle, BattleMemory("sets"), usage={}, set_priors=priors
    )[0]

    assert len(belief.set_hypotheses) == 2
    assert belief.set_hypotheses[0].item == "lifeorb"
    assert belief.set_hypotheses[0].ability == "roughskin"
    assert "earthquake" in belief.set_hypotheses[0].moves
    assert belief.set_hypotheses[0].probability == pytest.approx(0.7)
    assert belief.set_hypotheses[1].item == "choicescarf"
    assert belief.set_hypotheses[1].ability == "sandveil"
    assert belief.set_hypotheses[1].probability == pytest.approx(0.3)


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
