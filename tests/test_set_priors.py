"""Unit tests for the Phase 3 set-priors feature: `tools/build_set_priors.py`'s
aggregation over parsed replay records, and `vgc.sets.opponent_move_ids`/
`load_set_priors` (the consumer side). One evaluator-level test confirms
`build_context` actually fills `opp_protect_prob` from priors when Open Team Sheets
hasn't revealed a slot's moves.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from poke_env.battle.pokemon import Pokemon

import vgc.evaluator as evaluator_module
from vgc.evaluator import build_context
from vgc.models import PolicyConfig
from vgc.sets import load_set_priors, opponent_move_ids
from tools.build_set_priors import _aggregate_replay

# --- tools/build_set_priors.py: aggregation over synthetic parsed records ------------


def _record(
    player: str,
    decision_kind: str,
    active: list[dict | None],
    action: dict | None = None,
    bench: list[dict] | None = None,
) -> dict:
    return {
        "player": player,
        "decision_kind": decision_kind,
        "state": {"our": {"active": active, "bench": bench or []}},
        "action": action or {},
    }


def _new_aggregators() -> tuple[dict, dict, dict, Counter]:
    return defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter()


def test_aggregate_replay_counts_moves_items_abilities_and_appearances() -> None:
    move_counts, item_counts, ability_counts, appearance_counts = _new_aggregators()
    active = [
        {"species": "garchomp", "item": "lifeorb", "ability": "roughskin"},
        {"species": "klefki", "item": None, "ability": "prankster"},
    ]
    records = [
        _record(
            "p1",
            "turn",
            active=active,
            action={
                "slot0": {"kind": "move", "move_id": "earthquake"},
                "slot1": {"kind": "move", "move_id": "protect"},
            },
        )
    ]

    _aggregate_replay(records, move_counts, item_counts, ability_counts, appearance_counts)

    assert appearance_counts["garchomp"] == 1
    assert appearance_counts["klefki"] == 1
    assert move_counts["garchomp"]["earthquake"] == 1
    assert move_counts["klefki"]["protect"] == 1
    assert item_counts["garchomp"]["lifeorb"] == 1
    assert ability_counts["klefki"]["prankster"] == 1


def test_aggregate_replay_dedups_repeated_move_within_one_game() -> None:
    move_counts, item_counts, ability_counts, appearance_counts = _new_aggregators()
    active = [{"species": "garchomp", "item": None, "ability": None}, None]
    records = [
        _record(
            "p1", "turn", active=active, action={"slot0": {"kind": "move", "move_id": "earthquake"}}
        ),
        _record(
            "p1", "turn", active=active, action={"slot0": {"kind": "move", "move_id": "earthquake"}}
        ),
        _record(
            "p1", "turn", active=active, action={"slot0": {"kind": "move", "move_id": "earthquake"}}
        ),
    ]

    _aggregate_replay(records, move_counts, item_counts, ability_counts, appearance_counts)

    # Used the SAME move 3 times in one game -- counted once, not three times, and the
    # species only appeared in one (replay, player), so appearances is 1 too.
    assert move_counts["garchomp"]["earthquake"] == 1
    assert appearance_counts["garchomp"] == 1


def test_aggregate_replay_bench_species_counts_as_appearance_without_moves() -> None:
    move_counts, item_counts, ability_counts, appearance_counts = _new_aggregators()
    records = [
        _record(
            "p1",
            "turn",
            active=[{"species": "garchomp", "item": None, "ability": None}, None],
            bench=[{"species": "klefki", "hp_fraction": 1.0}],
            action={"slot0": {"kind": "move", "move_id": "earthquake"}, "slot1": {"kind": "pass"}},
        )
    ]

    _aggregate_replay(records, move_counts, item_counts, ability_counts, appearance_counts)

    assert appearance_counts["klefki"] == 1
    assert "klefki" not in move_counts


def test_aggregate_replay_item_takes_last_revealed_value_per_game() -> None:
    move_counts, item_counts, ability_counts, appearance_counts = _new_aggregators()
    records = [
        _record(
            "p1",
            "turn",
            active=[{"species": "garchomp", "item": "lifeorb", "ability": None}, None],
            action={"slot0": {"kind": "move", "move_id": "dragonclaw"}},
        ),
        _record(
            "p1",
            "turn",
            active=[{"species": "garchomp", "item": "whiteherb", "ability": None}, None],
            action={"slot0": {"kind": "pass"}},
        ),
    ]

    _aggregate_replay(records, move_counts, item_counts, ability_counts, appearance_counts)

    assert item_counts["garchomp"]["whiteherb"] == 1
    assert "lifeorb" not in item_counts["garchomp"]


def test_aggregate_replay_tracks_players_independently() -> None:
    move_counts, item_counts, ability_counts, appearance_counts = _new_aggregators()
    records = [
        _record(
            "p1",
            "turn",
            active=[{"species": "garchomp", "item": None, "ability": None}, None],
            action={"slot0": {"kind": "move", "move_id": "earthquake"}},
        ),
        _record(
            "p2",
            "turn",
            active=[{"species": "garchomp", "item": None, "ability": None}, None],
            action={"slot0": {"kind": "move", "move_id": "dragonclaw"}},
        ),
    ]

    _aggregate_replay(records, move_counts, item_counts, ability_counts, appearance_counts)

    assert appearance_counts["garchomp"] == 2  # both p1 and p2 ran it this "replay"
    assert move_counts["garchomp"]["earthquake"] == 1
    assert move_counts["garchomp"]["dragonclaw"] == 1


# --- vgc.sets.load_set_priors / opponent_move_ids ------------------------------------


@dataclass
class _FakeMovesPokemon:
    species: str
    moves: dict[str, Any] = field(default_factory=dict)


def test_load_set_priors_missing_file_returns_empty_dict() -> None:
    load_set_priors.cache_clear()
    assert load_set_priors("/nonexistent/path/set_priors.json") == {}


def test_opponent_move_ids_no_fill_when_four_revealed() -> None:
    fake = _FakeMovesPokemon(
        species="garchomp",
        moves={"earthquake": None, "dragonclaw": None, "ironhead": None, "protect": None},
    )
    priors = {"species": {"garchomp": {"appearances": 100, "moves": {"swordsdance": 90}}}}

    result = opponent_move_ids(fake, priors=priors, config=PolicyConfig())

    assert set(result) == {"earthquake", "dragonclaw", "ironhead", "protect"}
    assert "swordsdance" not in result


def test_opponent_move_ids_fill_ordered_by_frequency() -> None:
    fake = _FakeMovesPokemon(species="garchomp", moves={"earthquake": None})
    priors = {
        "species": {
            "garchomp": {
                "appearances": 100,
                "moves": {"protect": 90, "ironhead": 80, "dragonclaw": 70, "swordsdance": 5},
            }
        }
    }

    result = opponent_move_ids(fake, priors=priors, config=PolicyConfig())

    assert result == ["earthquake", "protect", "ironhead", "dragonclaw"]


def test_opponent_move_ids_dedups_revealed_moves_from_fill() -> None:
    fake = _FakeMovesPokemon(species="garchomp", moves={"earthquake": None, "protect": None})
    priors = {
        "species": {
            "garchomp": {
                "appearances": 100,
                "moves": {"protect": 90, "ironhead": 80, "dragonclaw": 70},
            }
        }
    }

    result = opponent_move_ids(fake, priors=priors, config=PolicyConfig())

    assert result == ["earthquake", "protect", "ironhead", "dragonclaw"]
    assert result.count("protect") == 1


def test_opponent_move_ids_min_games_threshold_respected() -> None:
    fake = _FakeMovesPokemon(species="garchomp", moves={"earthquake": None})
    priors = {"species": {"garchomp": {"appearances": 2, "moves": {"protect": 2}}}}
    config = PolicyConfig(set_prior_min_games=5)

    result = opponent_move_ids(fake, priors=priors, config=config)

    assert result == ["earthquake"]


def test_opponent_move_ids_missing_species_is_revealed_only() -> None:
    fake = _FakeMovesPokemon(species="mewtwo", moves={"psychic": None})
    priors = {"species": {"garchomp": {"appearances": 100, "moves": {"protect": 90}}}}

    result = opponent_move_ids(fake, priors=priors, config=PolicyConfig())

    assert result == ["psychic"]


def test_opponent_move_ids_empty_priors_is_revealed_only() -> None:
    fake = _FakeMovesPokemon(species="garchomp", moves={"earthquake": None})

    result = opponent_move_ids(fake, priors={}, config=PolicyConfig())

    assert result == ["earthquake"]


def test_opponent_move_ids_disabled_via_config() -> None:
    fake = _FakeMovesPokemon(species="garchomp", moves={"earthquake": None})
    priors = {"species": {"garchomp": {"appearances": 100, "moves": {"protect": 90}}}}
    config = PolicyConfig(use_set_priors=False)

    result = opponent_move_ids(fake, priors=priors, config=config)

    assert result == ["earthquake"]


# --- evaluator-level: build_context fills opp_protect_prob from priors --------------


def test_build_context_fills_protect_probability_from_priors_when_unrevealed(monkeypatch) -> None:
    our_mon = Pokemon(gen=9, species="garchomp")
    opp_mon = Pokemon(gen=9, species="klefki")
    assert opp_mon.moves == {}  # NO revealed moves -- exactly the scenario under test

    fake_battle = SimpleNamespace(
        teampreview_opponent_team=None,
        opponent_team={},
        fields={},
        weather={},
        side_conditions=[],
        opponent_side_conditions=[],
        active_pokemon=[our_mon, None],
        opponent_active_pokemon=[opp_mon, None],
    )
    fake_priors = {
        "species": {"klefki": {"appearances": 50, "moves": {"protect": 45, "spikyshield": 2}}}
    }
    monkeypatch.setattr(evaluator_module, "load_set_priors", lambda: fake_priors)

    ctx = build_context(fake_battle, PolicyConfig())

    assert ctx.opp_protect_prob[0] > 0.0
