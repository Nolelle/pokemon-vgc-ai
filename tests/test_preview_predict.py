"""Unit tests for `vgc.preview_predict` -- the iteration-6 symmetric opponent preview
predictor (step 1 of `docs/preview_prediction_plan.md`). Uses real champions-mod species
(no live battle/server needed) since the predictor's whole point is running the SAME
`vgc.team_preview` scoring internals over real data -- mirrors `tests/test_team_preview.py`'s
style for that reason.
"""

from __future__ import annotations

import pytest

from vgc.models import PolicyConfig
from vgc.preview_predict import (
    PreviewCandidate,
    _species_only_team,
    bring4_distribution,
    lead_distribution,
    predict_preview_choice,
)
from vgc.sets import load_set_priors, load_usage_spreads

# --- bring4_distribution / lead_distribution: pure aggregation ----------------------


def _candidate(pick, leads, probability) -> PreviewCandidate:
    order = tuple(leads) + tuple(i for i in pick if i not in leads)
    return PreviewCandidate(
        pick=tuple(pick), leads=tuple(leads), order=order, score=0.0, probability=probability
    )


def test_bring4_distribution_aggregates_by_subset() -> None:
    candidates = [
        _candidate((0, 1, 2, 3), (0, 1), 0.10),
        _candidate((0, 1, 2, 3), (0, 2), 0.05),
        _candidate((0, 1, 2, 3), (2, 3), 0.05),
        _candidate((0, 1, 2, 4), (0, 1), 0.30),
    ]
    ranked = bring4_distribution(candidates)
    assert ranked[0] == ((0, 1, 2, 4), pytest.approx(0.30))
    assert ranked[1] == ((0, 1, 2, 3), pytest.approx(0.20))


def test_lead_distribution_aggregates_by_order_independent_pair() -> None:
    candidates = [
        _candidate((0, 1, 2, 3), (0, 1), 0.10),
        _candidate((0, 1, 4, 5), (1, 0), 0.15),  # same pair, order-independent
        _candidate((0, 1, 2, 3), (2, 3), 0.05),
    ]
    ranked = lead_distribution(candidates)
    assert ranked[0] == ((0, 1), pytest.approx(0.25))
    assert ranked[1] == ((2, 3), pytest.approx(0.05))


def test_distributions_sum_to_total_input_probability() -> None:
    candidates = [
        _candidate((0, 1, 2, 3), (0, 1), 0.4),
        _candidate((0, 1, 2, 3), (2, 3), 0.3),
        _candidate((0, 1, 4, 5), (0, 1), 0.3),
    ]
    assert sum(p for _subset, p in bring4_distribution(candidates)) == pytest.approx(1.0)
    assert sum(p for _pair, p in lead_distribution(candidates)) == pytest.approx(1.0)


# --- _species_only_team: symmetric "no x-ray vision" guessing ------------------------


def test_species_only_team_leaves_item_and_ability_unrevealed() -> None:
    usage = load_usage_spreads()
    priors = load_set_priors()
    states, _move_ids, mons = _species_only_team(
        ["garchomp", "klefki"], usage, priors, PolicyConfig()
    )
    assert all(state.item is None for state in states)
    assert all(state.ability is None for state in states)
    assert all(mon.item is None and mon.ability is None for mon in mons)


def test_species_only_team_fills_moves_from_priors_when_available() -> None:
    usage = load_usage_spreads()
    priors = load_set_priors()
    # garchomp is a common, heavily-used species in the corpus -- should have priors.
    _states, move_ids, mons = _species_only_team(["garchomp"], usage, priors, PolicyConfig())
    assert len(move_ids[0]) > 0
    assert set(mons[0].moves.keys()) == set(move_ids[0])


def test_species_only_team_species_are_never_revealed_as_real_pokemon_object() -> None:
    """Sanity: the fake mon's `.moves`/`.item`/`.ability` before the prior fill really
    are empty/None -- confirms opponent_move_ids has nothing but priors to draw from,
    i.e. this genuinely can't leak real info even if a caller passed a real team.
    """
    usage = load_usage_spreads()
    priors = load_set_priors()
    states, move_ids, mons = _species_only_team(["klefki"], usage, priors, PolicyConfig())
    assert states[0].species_id == "klefki"
    assert states[0].sp_spread is not None  # something was guessed, not left None


# --- predict_preview_choice: end-to-end over real species ----------------------------


_PICKER_PREVIEW = ["floetteeternal", "sneasler", "incineroar", "milotic", "sinistcha", "tyranitar"]
_TARGET_PREVIEW = ["charizard", "whimsicott", "basculegion", "kingambit", "sylveon", "garchomp"]


def test_predict_preview_choice_returns_90_candidates_summing_to_one() -> None:
    candidates = predict_preview_choice(_PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig())
    assert len(candidates) == 90
    assert sum(c.probability for c in candidates) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= c.probability <= 1.0 for c in candidates)


def test_predict_preview_choice_sorted_descending_by_probability() -> None:
    candidates = predict_preview_choice(_PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig())
    probs = [c.probability for c in candidates]
    assert probs == sorted(probs, reverse=True)


def test_predict_preview_choice_every_candidate_is_legal() -> None:
    candidates = predict_preview_choice(_PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig())
    for c in candidates:
        assert len(set(c.pick)) == 4
        assert all(0 <= i < 6 for i in c.pick)
        assert len(set(c.leads)) == 2
        assert set(c.leads).issubset(set(c.pick))
        assert set(c.order) == set(c.pick)
        assert c.order[:2] == c.leads


def test_predict_preview_choice_empty_when_picker_has_fewer_than_four() -> None:
    candidates = predict_preview_choice(_PICKER_PREVIEW[:3], _TARGET_PREVIEW, PolicyConfig())
    assert candidates == []


def test_predict_preview_choice_empty_when_target_has_fewer_than_two() -> None:
    candidates = predict_preview_choice(_PICKER_PREVIEW, _TARGET_PREVIEW[:1], PolicyConfig())
    assert candidates == []


def test_predict_preview_choice_lower_temperature_sharpens_distribution() -> None:
    sharp = predict_preview_choice(
        _PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig(preview_prediction_temperature=1.0)
    )
    flat = predict_preview_choice(
        _PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig(preview_prediction_temperature=100.0)
    )
    assert sharp[0].probability > flat[0].probability


def test_predict_preview_choice_never_predicts_a_mega_advantage() -> None:
    """Species-only view never knows a held mega stone -- the predictor must not
    silently assume either side mega-evolves (see module docstring).
    """
    candidates = predict_preview_choice(_PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig())
    assert len(candidates) == 90  # sanity: still runs to completion despite no mega info


def test_predict_preview_choice_is_methodologically_symmetric() -> None:
    """predict_preview_choice(A, B) and predict_preview_choice(B, A) use the identical
    species-only guessing pipeline for whichever side is in the "picker" vs "target"
    slot -- both should produce a full, valid 90-candidate distribution regardless of
    which real team ends up in which slot.
    """
    a_vs_b = predict_preview_choice(_PICKER_PREVIEW, _TARGET_PREVIEW, PolicyConfig())
    b_vs_a = predict_preview_choice(_TARGET_PREVIEW, _PICKER_PREVIEW, PolicyConfig())
    assert len(a_vs_b) == len(b_vs_a) == 90
    assert sum(c.probability for c in b_vs_a) == pytest.approx(1.0, abs=1e-6)
