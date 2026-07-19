"""Win-condition game-plan layer (v1): a pairwise (our team x their team) matchup matrix
and the derived strategic signals VGC players reason about explicitly -- "who is our win
condition", "who beats it", "what's our answer to their scariest threat" -- distinct from
`vgc.evaluator`'s per-TURN damage/KO scoring, which has no notion of a game-spanning plan.

Pure over already-built `PokemonState`s (`build_gameplan` takes no battle/poke-env
objects at all) so it's directly unit-testable against hand-built states, mirroring
`tests/test_evaluator.py`'s style -- see `tests/test_gameplan.py`.

## The matchup matrix

For every (our mon i, their mon j) pair, a NEUTRAL-FIELD (no weather/terrain/screens --
this is a team-building-time, not turn-time, question) expected-damage-%-per-turn in
both directions, via a local `_best_move_percent` (same idea as
`vgc.evaluator._best_attacking_move`, trimmed to just the percent). Deliberately NOT
imported from there: `vgc.evaluator.build_context` calls INTO this module (to build one
`GamePlan` per turn, stored on `_Context.gameplan`), so this module must not import back
from `vgc.evaluator` -- that would be a circular import. `vgc.gameplan` is a foundation
module alongside `vgc.damage`/`vgc.sets`/`vgc.stats`, not a Phase-2b consumer the way
`vgc.team_preview` is (which DOES import from `vgc.evaluator`, safely, since nothing
imports back from `vgc.team_preview`).

## "Who wins the 1v1"

Reduced to a turns-to-KO race (`ceil(100 / pct)`, `inf` if `pct <= 0`): fewer turns wins;
a tie in turns is broken by who's faster (gets the last hit in within the tied turn); a
tie in both (including "neither can KO the other") is not a win for either side -- see
`_beats`. This is a simple, well-understood proxy for "who wins the 1v1" real VGC players
reason about, not a full turn-by-turn simulation (no Protect, no items triggering
mid-fight, no chip damage from hazards/status) -- v1 scope, same spirit as
`vgc.evaluator`'s own myopic 1-turn scoring.

## Derived fields (see `GamePlan`)

- `win_con_scores[i]` / `primary_win_con_idx`: the (uniformly-weighted -- this pure
  function has no usage-frequency data available to weight by) fraction of their known
  team our mon `i` wins the 1v1 against; the argmax is our primary win condition.
- `their_threat_scores[j]` / `primary_threat_idx`: the symmetric figure from their side.
- `answers`: for each of their mons, our best answer by RAW trade advantage (our_pct -
  their_pct, not the win/lose boolean -- a mon that "wins" narrowly on turn-count can
  still be a worse practical answer than one with a bigger damage cushion).
- `plan_breakers`: their mons that beat OUR primary win con specifically (the
  Chandelure-vs-Torkoal case: Torkoal is the team's primary win con, but Chandelure's
  Shadow Ball beats it outright, so removing Chandelure re-enables the plan).

`vgc.evaluator.build_context` builds one `GamePlan` per turn and stores it on
`_Context.gameplan` -- see that module for how `win_con_preservation_weight`,
`plan_breaker_target_bonus`, and `collapsed_matchup_switch_bonus` consume it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vgc.damage import FieldState, PokemonState, damage_range, to_id
from vgc.data import load_moves
from vgc.models import PolicyConfig

# Team-building-time matchup question, not a turn-time one -- no weather/terrain/screens,
# and a plain 1v1 (num_targets=1) rather than doubles spread dilution, since the matrix
# represents "how does this pair trade in the abstract", not "what does the current
# board look like" (that's `vgc.evaluator`'s job, every turn, via the live `_Context`).
_NEUTRAL_FIELD = FieldState(
    weather=None,
    terrain=None,
    screens=frozenset(),
    trick_room=False,
    is_doubles=True,
    num_targets=1,
)


def _best_move_percent(
    attacker: PokemonState, move_ids: list[str], defender: PokemonState
) -> float:
    """Best expected damage % `attacker` deals `defender` on a neutral field, over
    `move_ids` -- trimmed-down sibling of `vgc.evaluator._best_attacking_move` (percent
    only, no move_id/priority needed here; this module can't import that function, see
    module docstring).
    """
    moves_data = load_moves()
    best_pct = 0.0
    for raw_move_id in move_ids:
        move_id = to_id(raw_move_id)
        data = moves_data.get(move_id) if move_id else None
        if data is None or data["category"] == "Status":
            continue
        result = damage_range(attacker, defender, move_id, _NEUTRAL_FIELD)
        if not result.breakdown["move_supported"] or result.breakdown["immune"]:
            continue
        best_pct = max(best_pct, result.expected_percent)
    return best_pct


def _effective_speed(state: PokemonState) -> float:
    """Neutral-field effective Speed (Choice Scarf/paralysis only, no weather-ability
    boost -- matches `vgc.evaluator.effective_speed`'s own neutral scope). Duplicated
    rather than imported for the same reason `_best_move_percent` is -- see module
    docstring.
    """
    from vgc.damage import _apply_stage  # local: mirrors evaluator.effective_speed's own

    speed = float(_apply_stage(state.stats()["spe"], state.boost_stage("spe")))
    if state.item == "choicescarf":
        speed *= 1.5
    if state.status == "par":
        speed *= 0.5
    return speed


def _turns_to_ko(pct: float) -> float:
    if pct <= 0.0:
        return math.inf
    return math.ceil(100.0 / pct)


def _beats(pct_a: float, pct_b: float, a_faster: bool) -> bool:
    """True if the side dealing `pct_a` (per turn) beats the side dealing `pct_b` back,
    per the turns-to-KO race described in the module docstring's "Who wins the 1v1".
    """
    turns_a = _turns_to_ko(pct_a)
    turns_b = _turns_to_ko(pct_b)
    if turns_a < turns_b:
        return True
    if turns_a > turns_b:
        return False
    if math.isinf(turns_a):
        return False  # neither can KO the other -- not a real "win" for either side
    return a_faster


def _argmax(values: tuple[float, ...]) -> int | None:
    if not values:
        return None
    best_idx = 0
    for idx in range(1, len(values)):
        if values[idx] > values[best_idx]:
            best_idx = idx
    return best_idx


@dataclass(frozen=True)
class GamePlan:
    our_species: tuple[str, ...]
    opp_species: tuple[str, ...]
    # Fraction (0.0-1.0) of the OTHER side's team each mon wins the 1v1 against --
    # aligned index-for-index with our_species/opp_species respectively.
    win_con_scores: tuple[float, ...]
    their_threat_scores: tuple[float, ...]
    primary_win_con_idx: int | None
    primary_threat_idx: int | None
    # opp_species_id -> our best-answer species_id (argmax trade advantage).
    answers: dict[str, str]
    # opp_species_ids that beat our primary win con specifically.
    plan_breakers: frozenset[str]
    # (our_species_id, opp_species_id) -> our neutral-field expected % onto them.
    matrix: dict[tuple[str, str], float]

    @property
    def primary_win_con_species(self) -> str | None:
        if self.primary_win_con_idx is None:
            return None
        return self.our_species[self.primary_win_con_idx]

    @property
    def primary_threat_species(self) -> str | None:
        if self.primary_threat_idx is None:
            return None
        return self.opp_species[self.primary_threat_idx]

    def table_percent(self, our_species_id: str, opp_species_id: str) -> float:
        """Neutral-field expected % `our_species_id` deals `opp_species_id`, or 0.0 if
        that pair was never built into the matrix (e.g. an opponent mon we have no
        state for at all).
        """
        return self.matrix.get((our_species_id, opp_species_id), 0.0)

    def summary(self) -> dict:
        """Trace-friendly dict for `vgc.decision_trace.record_note`."""
        return {
            "primary_win_con": self.primary_win_con_species,
            "win_con_scores": {
                species: round(score, 3)
                for species, score in zip(self.our_species, self.win_con_scores, strict=True)
            },
            "primary_threat": self.primary_threat_species,
            "their_threat_scores": {
                species: round(score, 3)
                for species, score in zip(self.opp_species, self.their_threat_scores, strict=True)
            },
            "answers": dict(self.answers),
            "plan_breakers": sorted(self.plan_breakers),
        }


def build_gameplan(
    our_states: list[PokemonState],
    opp_states: list[PokemonState],
    our_move_ids: list[list[str]],
    opp_move_ids: list[list[str]],
    config: PolicyConfig | None = None,
    preferred_win_con_species: str | None = None,
) -> GamePlan:
    """Pure derivation of a `GamePlan` from already-built states + movesets -- see module
    docstring for the full algorithm. Callers pass only REAL (non-None, non-fainted)
    states, pre-filtered; `our_move_ids[i]`/`opp_move_ids[j]` must align index-for-index
    with `our_states`/`opp_states`.
    """
    del config  # accepted for calling-convention consistency with the rest of this
    # codebase's battle-context builders; nothing here is currently a tunable judgment
    # call (the matchup math is mechanical, and PolicyConfig weights that CONSUME a
    # GamePlan live on the evaluator side -- see vgc.evaluator's win_con_preservation_
    # weight/plan_breaker_target_bonus/collapsed_matchup_switch_bonus).
    our_species = tuple(state.species_id for state in our_states)
    opp_species = tuple(state.species_id for state in opp_states)
    our_speeds = tuple(_effective_speed(state) for state in our_states)
    opp_speeds = tuple(_effective_speed(state) for state in opp_states)

    # our_pct[i][j] / their_pct[i][j]: neutral-field % our mon i deals their mon j, and
    # vice versa.
    our_pct = [
        [
            _best_move_percent(our_states[i], our_move_ids[i], opp_states[j])
            for j in range(len(opp_states))
        ]
        for i in range(len(our_states))
    ]
    their_pct = [
        [
            _best_move_percent(opp_states[j], opp_move_ids[j], our_states[i])
            for j in range(len(opp_states))
        ]
        for i in range(len(our_states))
    ]

    matrix = {
        (our_species[i], opp_species[j]): our_pct[i][j]
        for i in range(len(our_states))
        for j in range(len(opp_states))
    }

    win_con_scores = tuple(
        (
            sum(
                1
                for j in range(len(opp_states))
                if _beats(our_pct[i][j], their_pct[i][j], our_speeds[i] > opp_speeds[j])
            )
            / len(opp_states)
            if opp_states
            else 0.0
        )
        for i in range(len(our_states))
    )
    their_threat_scores = tuple(
        (
            sum(
                1
                for i in range(len(our_states))
                if _beats(their_pct[i][j], our_pct[i][j], opp_speeds[j] > our_speeds[i])
            )
            / len(our_states)
            if our_states
            else 0.0
        )
        for j in range(len(opp_states))
    )
    primary_win_con_idx = _argmax(win_con_scores)
    if preferred_win_con_species is not None:
        preferred_id = to_id(preferred_win_con_species)
        preferred_idx = next(
            (idx for idx, species_id in enumerate(our_species) if species_id == preferred_id),
            None,
        )
        if preferred_idx is not None:
            primary_win_con_idx = preferred_idx
    primary_threat_idx = _argmax(their_threat_scores)

    answers: dict[str, str] = {}
    for j in range(len(opp_states)):
        if not our_states:
            continue
        best_i = max(range(len(our_states)), key=lambda i: our_pct[i][j] - their_pct[i][j])
        answers[opp_species[j]] = our_species[best_i]

    plan_breakers: set[str] = set()
    if primary_win_con_idx is not None:
        i = primary_win_con_idx
        for j in range(len(opp_states)):
            if _beats(their_pct[i][j], our_pct[i][j], opp_speeds[j] > our_speeds[i]):
                plan_breakers.add(opp_species[j])

    return GamePlan(
        our_species=our_species,
        opp_species=opp_species,
        win_con_scores=win_con_scores,
        their_threat_scores=their_threat_scores,
        primary_win_con_idx=primary_win_con_idx,
        primary_threat_idx=primary_threat_idx,
        answers=answers,
        plan_breakers=frozenset(plan_breakers),
        matrix=matrix,
    )
