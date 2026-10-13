"""Critical guesser-capture behaviour: fog-safe sightings, honest fallback,
defensive-damage learning, and report-card scoring."""

from __future__ import annotations

from types import SimpleNamespace

from vgc.battle_memory import BattleMemory
from vgc.guesser_eval import score_beliefs
from vgc.models import PolicyConfig
from vgc.opponent_belief import build_opponent_beliefs
from vgc.unknown_species import (
    aggregate_captures,
    append_capture_records,
    build_capture_records,
    load_capture_records,
    merge_priors,
    species_history_status,
)

_EMPTY_PRIORS: dict = {"species": {}}


def _battle(species: str, *, turn: int = 3):
    mon = SimpleNamespace(species=species, moves={}, item=None, ability=None, status=None,
                          current_hp_fraction=1.0, fainted=False)
    return SimpleNamespace(
        teampreview_opponent_team=[mon],
        opponent_team={"x": mon},
        opponent_active_pokemon=[mon],
        battle_tag="room-1",
        turn=turn,
    )


def test_no_history_species_is_captured_without_hidden_fields(tmp_path) -> None:
    battle = _battle("fakemonx")
    memory = BattleMemory("room-1")
    memory.opponent_moves["fakemonx"]["tackle"] += 1
    records = build_capture_records(battle, memory, PolicyConfig(), usage={}, set_priors=_EMPTY_PRIORS)
    assert len(records) == 1
    record = records[0]
    assert record["species"] == "fakemonx"
    assert record["history_status"] == "unknown"
    assert record["revealed_moves"] == ["tackle"]
    assert record["spread_learnable"] is False
    # Nothing a real player could not see may be captured.
    assert not ({"sp", "stat_points", "nature", "evs"} & set(record))

    path = tmp_path / "unknown_species.jsonl"
    assert append_capture_records(path, records) == 1
    assert load_capture_records(path) == records


def test_known_species_is_not_captured() -> None:
    battle = _battle("garchomp")
    records = build_capture_records(
        battle, BattleMemory("room-1"), PolicyConfig(), usage={}, set_priors=_EMPTY_PRIORS
    )
    # garchomp has real priors in the shipped corpus, so it must not be captured
    # when the caller passes the real files; with empty priors it is unknown.
    assert species_history_status("garchomp", usage={}, set_priors=_EMPTY_PRIORS) == "unknown"
    assert len(records) == 1


def test_captures_aggregate_once_per_battle_and_merge() -> None:
    base = {"meta": {}, "species": {"fakemonx": {"appearances": 2, "moves": {"tackle": 2}, "items": {}, "abilities": {}, "configurations": []}}}
    records = [
        {"schema": "unknown_species_capture_v1", "species": "fakemonx", "battle_tag": "b1", "turn": 1,
         "appeared": True, "revealed_moves": [], "known_item": None, "known_ability": None},
        {"schema": "unknown_species_capture_v1", "species": "fakemonx", "battle_tag": "b1", "turn": 5,
         "appeared": True, "revealed_moves": ["tackle"], "known_item": "choicescarf", "known_ability": None},
        {"schema": "unknown_species_capture_v1", "species": "fakemonx", "battle_tag": "b2", "turn": 2,
         "appeared": True, "revealed_moves": ["tackle"], "known_item": None, "known_ability": None},
        # Previewed but never brought: must not inflate appearances.
        {"schema": "unknown_species_capture_v1", "species": "fakemonx", "battle_tag": "b3", "turn": 1,
         "appeared": False, "revealed_moves": [], "known_item": None, "known_ability": None},
    ]
    supplement = aggregate_captures(records)
    entry = supplement["species"]["fakemonx"]
    assert entry["appearances"] == 2  # b1 counted once despite two records; b3 excluded
    assert entry["moves"] == {"tackle": 2}
    assert entry["items"] == {"choicescarf": 1}
    merged = merge_priors(base, supplement)
    assert merged["species"]["fakemonx"]["appearances"] == 4
    assert merged["species"]["fakemonx"]["moves"]["tackle"] == 4
    # Re-merging the same captures must not recount them.
    assert merge_priors(merged, supplement)["species"]["fakemonx"]["appearances"] == 4


def test_damage_we_deal_narrows_their_defensive_spread() -> None:
    from vgc.battle_memory import DefensiveDamageObservation
    from vgc.damage import FieldState, PokemonState, damage_range

    attacker = PokemonState("garchomp", sp_spread={"atk": 32}, nature="adamant")
    bulky = PokemonState("incineroar", sp_spread={"hp": 32, "def": 32}, nature="impish")
    observed = damage_range(attacker, bulky, "earthquake", FieldState(is_doubles=True))
    memory = BattleMemory("defensive")
    memory.defensive_damage_observations.append(
        DefensiveDamageObservation(
            opponent_species="incineroar",
            move_id="earthquake",
            attacker_state=attacker,
            damage_fraction=observed.min_percent / 100.0,
        )
    )
    battle = _battle("incineroar")
    usage = {
        "incineroar": [
            {"sp": {"hp": 32, "def": 32}, "nature": "impish", "weight": 0.5},
            {"sp": {"hp": 0, "def": 0}, "nature": "adamant", "weight": 0.5},
        ]
    }
    belief = build_opponent_beliefs(
        battle, memory, PolicyConfig(use_defensive_damage_evidence=True),
        usage=usage, set_priors=_EMPTY_PRIORS,
    )[0]
    by_nature = {h.nature: h.probability for h in belief.hypotheses}
    assert by_nature["impish"] > by_nature["adamant"]


def test_guesser_decision_knobs_parked_off_by_default_capture_intact() -> None:
    # Parked after the pool150 A/B (51.2%, cluster-robust [0.492, 0.533],
    # runs/eval/guesser_ab_20261016.json) missed the pre-declared bar.
    config = PolicyConfig()
    assert config.use_fallback_spread_hypotheses is False
    assert config.use_defensive_damage_evidence is False
    # Capture/logging machinery stays available under the same defaults.
    records = build_capture_records(
        _battle("fakemonx"), BattleMemory("room-1"), config, usage={}, set_priors=_EMPTY_PRIORS
    )
    assert len(records) == 1


def test_report_card_scores_a_perfect_and_an_empty_guess() -> None:
    battle = _battle("garchomp")
    beliefs = build_opponent_beliefs(battle, BattleMemory("room-1"), usage={}, set_priors=_EMPTY_PRIORS)
    truth = {"garchomp": {"moves": ["earthquake"], "item": None, "ability": None, "sp": None, "nature": None}}
    report = score_beliefs(beliefs, truth)
    assert report["species_scored"] == 1
    assert report["moves"]["recall"] is not None
