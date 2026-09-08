"""Unit tests for the shared guided-selection core and its offline replay."""

from __future__ import annotations

import numpy as np
import pytest
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

from vgc.rl.distill import DistillationSample, build_guidance_metadata, simulate_guided_hits
from vgc.rl.encoding import CandidateFeatures
from vgc.rl.guided_selection import select_guided_candidate_indices


def _tag_row(*tags: str) -> frozenset[str]:
    return frozenset(tags)


def _sample(
    candidates: int,
    *,
    ranks: np.ndarray | None,
    tag_rows: np.ndarray | None,
    teacher: int = 0,
) -> DistillationSample:
    return DistillationSample(
        battle_id="b",
        state_indices=np.zeros(4, dtype=np.int64),
        state_scalars=np.zeros(4, dtype=np.float32),
        history_scalars=np.zeros(4, dtype=np.float32),
        candidates=CandidateFeatures(
            move_indices=np.zeros((candidates, 2), dtype=np.int64),
            target_indices=np.zeros((candidates, 2), dtype=np.int64),
            switch_species_indices=np.zeros((candidates, 2), dtype=np.int64),
            flags=np.zeros((candidates, 2, 4), dtype=np.float32),
        ),
        teacher_action_index=teacher,
        team_id="pool/a",
        turn=1,
        legal_action_count=candidates,
        candidate_myopic_ranks=ranks,
        candidate_tags=tag_rows,
    )


def _identity_rands(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


# --- select_guided_candidate_indices: same expectations as the live-selector tests ------

_TAGS = {
    0: _tag_row("double_attack", "non_protect"),  # heatwave / earthquake
    1: _tag_row("switch", "protect"),  # switch->incineroar / protect
    2: _tag_row("protect"),  # protect / protect
    3: _tag_row("speed_control", "non_protect"),  # tailwind / protect
    4: _tag_row("double_attack", "non_protect"),
}
for _index in range(5, 12):
    _TAGS[_index] = _tag_row("non_protect")


def test_core_keeps_exact_k_and_reserves_safety_inside_budget():
    selected, safety = select_guided_candidate_indices(
        12,
        ranked_indices=list(range(11, -1, -1)),  # network prefers the tail, like _ranking
        myopic_position={index: index for index in range(12)},
        tags_by_index=_TAGS,
        cutoff=10,
        safety_slots=4,
    )

    assert len(selected) == 10
    assert [record["reason"] for record in safety] == ["heuristic_top", "switch", "control"]
    assert selected[0] == 0  # myopic leader is always the first safety pick
    assert set(selected) == {0, 1, 3, 11, 10, 9, 8, 7, 6, 5}


def test_core_caps_safety_at_half_the_budget():
    selected, safety = select_guided_candidate_indices(
        12,
        ranked_indices=list(range(11, -1, -1)),
        myopic_position={index: index for index in range(12)},
        tags_by_index=_TAGS,
        cutoff=5,
        safety_slots=4,
    )

    assert len(selected) == 5
    assert len(safety) == 2


def test_core_never_shrinks_the_budget_for_an_undersized_ranking():
    selected, _safety = select_guided_candidate_indices(
        12,
        ranked_indices=[],
        myopic_position={index: index for index in range(12)},
        tags_by_index=_TAGS,
        cutoff=10,
        safety_slots=4,
    )
    assert len(selected) == 10


# --- build_guidance_metadata -------------------------------------------------------------


def _move_order(first: str, second: str):
    single_first = type(
        "_Single",
        (),
        {
            "order": Move(first, gen=9),
            "move_target": 1,
            "mega": False,
            "z_move": False,
            "dynamax": False,
            "terastallize": False,
        },
    )()
    single_second = type(
        "_Single",
        (),
        {
            "order": Move(second, gen=9),
            "move_target": 0,
            "mega": False,
            "z_move": False,
            "dynamax": False,
            "terastallize": False,
        },
    )()
    return type("_Joint", (), {"first_order": single_first, "second_order": single_second})()


def _switch_protect_order():
    single_switch = type(
        "_Single",
        (),
        {
            "order": Pokemon(gen=9, species="incineroar"),
            "move_target": 0,
            "mega": False,
            "z_move": False,
            "dynamax": False,
            "terastallize": False,
        },
    )()
    single_protect = type(
        "_Single",
        (),
        {
            "order": Move("protect", gen=9),
            "move_target": 0,
            "mega": False,
            "z_move": False,
            "dynamax": False,
            "terastallize": False,
        },
    )()
    return type("_Joint", (), {"first_order": single_switch, "second_order": single_protect})()


def test_build_guidance_metadata_records_positions_and_tags():
    attack = _move_order("heatwave", "dragonclaw")
    switch_protect = _switch_protect_order()
    ranks, tags = build_guidance_metadata([attack, switch_protect], [switch_protect, attack])

    assert ranks.tolist() == [1, 0]
    assert tags[0].tolist() == [0, 0, 0, 1, 1]  # double_attack + non_protect
    assert tags[1].tolist() == [1, 1, 0, 0, 0]  # switch + protect


def test_build_guidance_metadata_returns_none_without_a_myopic_order():
    assert build_guidance_metadata([], []) is None
    order = _move_order("protect", "protect")
    assert build_guidance_metadata([order], None) is None


# --- simulate_guided_hits ----------------------------------------------------------------


def test_safety_slots_rescue_a_buried_myopic_leader():
    sample = _sample(40, ranks=_identity_rands(40), tag_rows=np.zeros((40, 5), dtype=np.int8))
    ranked = list(range(1, 40)) + [0]  # teacher action ranked LAST by the network
    hits = simulate_guided_hits([sample], [ranked], ks=(10,))
    assert hits[10] == [True]


def test_buried_teacher_without_safety_coverage_stays_missed():
    sample = _sample(
        40,
        ranks=_identity_rands(40),
        tag_rows=np.zeros((40, 5), dtype=np.int8),
        teacher=5,
    )
    ranks = sample.candidate_myopic_ranks.copy()
    ranks[5] = 39  # teacher is also last in the myopic order
    ranked = [index for index in range(40) if index != 5] + [5]
    hits = simulate_guided_hits([sample], [ranked], ks=(10,))
    assert hits[10] == [False]


def test_trivial_k_and_missing_metadata_are_distinguished():
    trivial = _sample(8, ranks=None, tag_rows=None)
    unknown = _sample(30, ranks=None, tag_rows=None)
    hits = simulate_guided_hits([trivial, unknown], [list(range(8)), list(range(30))], ks=(10,))
    # k >= legal count retains automatically even without metadata; otherwise unknown.
    assert hits[10] == [True, None]


def test_simulate_rejects_misaligned_rankings():
    sample = _sample(4, ranks=_identity_rands(4), tag_rows=np.zeros((4, 5), dtype=np.int8))
    with pytest.raises(ValueError):
        simulate_guided_hits([sample], [], ks=(1,))
