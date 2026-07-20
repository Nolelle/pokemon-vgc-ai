"""Symmetric opponent preview prediction (iteration 6 plan, step 1 --
`docs/preview_prediction_plan.md`): enumerate the OPPONENT's own 90 pick-4 x lead-pair
options over their previewed 6, scored with the SAME `vgc.team_preview` scoring
internals (`_build_matchup_matrix`/`_score_choice`) the bot uses for its OWN pick, but
run from THEIR side -- against a SPECIES-ONLY view of OUR team, since they can't see our
held items/exact moves/nature any more than we can see theirs without Open Team Sheets
(which essentially never triggers on the ladder -- see `vgc.replay_parse`'s module
docstring). "Symmetric" is the operative word: the predictor has no x-ray vision into
EITHER team's hidden info, so BOTH sides are built the identical way -- species known,
everything else (Stat Point spread/nature from `vgc.sets.load_usage_spreads`, moveset
from `vgc.sets.opponent_move_ids`'s corpus-prior fill, item/ability left unrevealed)
guessed from the SAME corpus priors `vgc.sets`/`vgc.team_preview` already use for a real
(non-OTS) opponent.

Produces a softmax DISTRIBUTION over the 90 (pick, leads) candidates, not a single best
guess -- `PolicyConfig.preview_prediction_temperature` controls how peaked it is. This
module deliberately does NOT enumerate `mega_evolved_state`/mega candidates: a
species-only view never knows a held mega stone, so `mega_species_id` correctly comes
back `None` for every mon on both sides and mega has no effect on either side's
predicted preference -- not a bug, exactly the information a real (non-OTS) opponent
prediction should reflect.

Step 2 (`tools/backtest_preview_prediction.py`) validates this against real teampreview
ground truth BEFORE any integration into `vgc.team_preview.build_team_order` (step 3,
gated on the backtest's numbers, a separate follow-up -- NOT implemented here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from statistics import mean

from vgc.damage import PokemonState
from vgc.evaluator import effective_speed
from vgc.gameplan import build_gameplan
from vgc.models import PolicyConfig
from vgc.principles import TeamSignals, detect_team_signals
from vgc.sets import load_set_priors, load_usage_spreads, opponent_move_ids, opponent_state
from vgc.team_preview import (
    _LEADS_COUNT,
    _PICK_COUNT,
    _PREVIEW_WEATHERS,
    _build_matchup_matrix,
    _score_choice,
)


@dataclass(frozen=True)
class _SpeciesOnlyMon:
    """Satisfies `vgc.sets.ObservedPokemon`'s Protocol with only a species id known --
    everything else (item/ability/boosts/status) is exactly what an observer with no
    Open Team Sheet and no in-battle reveals yet has: nothing. `moves` starts empty (so
    `opponent_move_ids` fills it entirely from priors) and is replaced afterward with
    the guessed set so `vgc.principles.detect_team_signals` sees a populated moveset --
    see `_species_only_team`.
    """

    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict = field(default_factory=dict)
    status: object = None
    current_hp_fraction: float = 1.0
    moves: dict = field(default_factory=dict)


def _species_only_team(
    species_ids: list[str],
    usage: dict,
    priors: dict,
    config: PolicyConfig,
) -> tuple[list[PokemonState], list[list[str]], list[_SpeciesOnlyMon]]:
    """`(states, move_id_lists, mons)` for a species-only guess of a team -- Stat Point
    spread/nature from `usage` (`load_usage_spreads`), moveset from `priors`
    (`opponent_move_ids`'s corpus-frequency fill), item/ability left `None` (unrevealed).
    """
    states: list[PokemonState] = []
    move_id_lists: list[list[str]] = []
    mons: list[_SpeciesOnlyMon] = []
    for species_id in species_ids:
        bare = _SpeciesOnlyMon(species=species_id)
        state = opponent_state(bare, usage=usage)
        move_ids = opponent_move_ids(bare, priors=priors, config=config)
        mon = _SpeciesOnlyMon(species=species_id, moves=dict.fromkeys(move_ids))
        states.append(state)
        move_id_lists.append(move_ids)
        mons.append(mon)
    return states, move_id_lists, mons


@dataclass(frozen=True)
class PreviewCandidate:
    """One (pick-4, lead-pair) candidate for the side being predicted.

    :param pick: the 4 chosen indices into that side's previewed-6 list (unordered set,
        but stored as the sorted tuple `_score_choice`'s `picked` arg expects).
    :param leads: the 2 lead indices, in LEAD ORDER (first two of `order`) -- a subset
        of `pick`.
    :param order: `leads + bench`, matching `vgc.team_preview.build_team_order`'s own
        `order` convention (what would become the `/team XXXX` digits).
    :param score: the raw `_score_choice` total for this candidate.
    :param probability: `softmax(score / temperature)` over all 90 candidates.
    """

    pick: tuple[int, ...]
    leads: tuple[int, ...]
    order: tuple[int, ...]
    score: float
    probability: float


def predict_preview_choice(
    picker_species: list[str],
    target_species: list[str],
    config: PolicyConfig | None = None,
) -> list[PreviewCandidate]:
    """Predict `picker_species`'s (their previewed 6, species ids) pick-4 + lead choice
    against `target_species` (the other side's previewed 6, species ids) -- both built
    as species-only guesses (see module docstring). Returns all 90 legal
    `PreviewCandidate`s, softmax-weighted, sorted by descending probability.

    Symmetric by construction: `predict_preview_choice(A, B)` scores A's choice against
    B exactly the way `predict_preview_choice(B, A)` would score B's choice against A --
    same internals, same guessing methodology, just the two species lists swapped.
    """
    config = config or PolicyConfig()
    if len(picker_species) < _PICK_COUNT or len(target_species) < _LEADS_COUNT:
        return []

    usage = load_usage_spreads()
    priors = load_set_priors()

    picker_states, picker_move_ids, picker_mons = _species_only_team(
        picker_species, usage, priors, config
    )
    target_states, target_move_ids, target_mons = _species_only_team(
        target_species, usage, priors, config
    )

    picker_signals: TeamSignals = detect_team_signals(picker_mons)
    target_signals: TeamSignals = detect_team_signals(target_mons)

    # A species-only view never knows a held mega stone -- mega_species_id needs an
    # item id, which is always None here, so this is correctly always empty (see module
    # docstring's mega paragraph), not a gap.
    mega_indices: frozenset[int] = frozenset()
    default_mega_idx: int | None = None

    # No Open Team Sheets means no gameplan-worthy hidden-set knowledge either; the
    # win-con/plan-breaker framework needs real move lists, which is exactly what the
    # priors-filled move_id lists here already are, so this reuses build_gameplan the
    # same way vgc.team_preview.build_team_order does for the real (non-OTS) opponent.
    gameplan = build_gameplan(
        picker_states, target_states, picker_move_ids, target_move_ids, config
    )
    closer_idx = gameplan.primary_win_con_idx
    opponent_closer_idx = gameplan.primary_threat_idx

    target_avg_speed = (
        mean(effective_speed(state) for state in target_states) if target_states else 0.0
    )

    matchup_by_weather = {
        weather: _build_matchup_matrix(
            picker_states, picker_move_ids, target_states, target_move_ids, weather=weather
        )
        for weather in _PREVIEW_WEATHERS
    }

    candidates: list[PreviewCandidate] = []
    for pick in combinations(range(len(picker_species)), _PICK_COUNT):
        for leads in combinations(pick, _LEADS_COUNT):
            bench = tuple(i for i in pick if i not in leads)
            order = leads + bench
            score, _breakdown = _score_choice(
                order=order,
                picked=pick,
                our_states=picker_states,
                our_move_id_lists=picker_move_ids,
                matchup_by_weather=matchup_by_weather,
                opp_avg_speed=target_avg_speed,
                n_opp=len(target_states),
                our_signals=picker_signals,
                opp_signals=target_signals,
                closer_idx=closer_idx,
                opponent_closer_idx=opponent_closer_idx,
                mega_indices=mega_indices,
                default_mega_idx=default_mega_idx,
                config=config,
            )
            candidates.append(
                PreviewCandidate(pick=pick, leads=leads, order=order, score=score, probability=0.0)
            )

    temperature = config.preview_prediction_temperature or 1e-6
    scaled = [c.score / temperature for c in candidates]
    max_scaled = max(scaled) if scaled else 0.0
    raw_weights = [math.exp(s - max_scaled) for s in scaled]
    total = sum(raw_weights)
    probabilities = (
        [w / total for w in raw_weights] if total > 0 else [1.0 / len(candidates)] * len(candidates)
    )

    weighted = [
        PreviewCandidate(pick=c.pick, leads=c.leads, order=c.order, score=c.score, probability=p)
        for c, p in zip(candidates, probabilities, strict=True)
    ]
    weighted.sort(key=lambda c: -c.probability)
    return weighted


def bring4_distribution(candidates: list[PreviewCandidate]) -> list[tuple[tuple[int, ...], float]]:
    """Aggregate `candidates`' probability by 4-subset (summing over the up-to-6
    lead-pair orderings sharing that subset) -- ranked descending. The bring-4 decision
    and the lead decision are reported separately (see `lead_distribution`) since a
    real opponent's choice of WHICH 4 and WHICH 2 of those 4 lead are different
    questions with different confidence.
    """
    totals: dict[tuple[int, ...], float] = {}
    for candidate in candidates:
        totals[candidate.pick] = totals.get(candidate.pick, 0.0) + candidate.probability
    return sorted(totals.items(), key=lambda item: -item[1])


def lead_distribution(candidates: list[PreviewCandidate]) -> list[tuple[tuple[int, ...], float]]:
    """Aggregate `candidates`' probability by lead PAIR (order-independent, sorted
    tuple), summed over every bring-4 subset that shares that lead pair -- ranked
    descending.
    """
    totals: dict[tuple[int, ...], float] = {}
    for candidate in candidates:
        key = tuple(sorted(candidate.leads))
        totals[key] = totals.get(key, 0.0) + candidate.probability
    return sorted(totals.items(), key=lambda item: -item[1])
