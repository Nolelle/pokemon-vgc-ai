"""Explicit belief branches over battle facts that no player is ever told.

Champions sleep lasts either two or three action opportunities -- ``sample([2, 3, 3])``
in ``data/mods/champions/conditions.ts`` -- and confusion lasts two to five,
``this.random(2, 6)`` in ``data/conditions.ts``. Both durations are rolled privately, so
a live Showdown mirror that patches in one concrete number is asserting a fact the
player does not have. Picking the modal duration is not "exact"; it is a confident guess.

This module converts each observed sleeping or confused Pokemon into the set of
remaining durations still consistent with what has actually been seen, weighted by the
real sampling prior. :mod:`vgc.rl.exact_search` then averages its Showdown branches over
those hypotheses. Every branch remains a legal Showdown state; only the belief weights
live here.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from vgc.mechanics_state import BattleMechanicsState, PokemonMechanicsState, SideMechanicsState
from vgc.models import PolicyConfig

# `sample([2, 3, 3])`: two action opportunities one time in three, otherwise three.
SLEEP_DURATION_PRIOR: tuple[tuple[int, float], ...] = ((2, 1.0 / 3.0), (3, 2.0 / 3.0))
# `random(2, 6)`: an inclusive 2..5 uniform draw.
CONFUSION_DURATION_PRIOR: tuple[tuple[int, float], ...] = tuple(
    (total, 0.25) for total in (2, 3, 4, 5)
)
# Early Bird burns two sleep ticks per action opportunity.
_DOUBLE_SLEEP_TICK_ABILITY = "earlybird"


@dataclass(frozen=True)
class HiddenStateHypothesis:
    """One legal assignment of the hidden timers, with its prior probability."""

    weight: float
    payload: dict[str, dict[str, dict[str, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"hypothesis weight must be in (0, 1], got {self.weight!r}")


def _remaining_options(
    prior: tuple[tuple[int, float], ...], elapsed: int, ticks_per_action: int
) -> list[tuple[int, float]]:
    """Durations still possible after ``elapsed`` observed action opportunities."""

    consumed = max(0, elapsed) * ticks_per_action
    options = [
        (total - consumed, probability)
        for total, probability in prior
        if total - consumed >= 1
    ]
    if not options:
        # The effect should already have ended. Fall back to "one opportunity left"
        # rather than inventing a longer timer than any legal roll allows.
        return [(1, 1.0)]
    total_probability = sum(probability for _remaining, probability in options)
    return [
        (remaining, probability / total_probability) for remaining, probability in options
    ]


def _confusion_elapsed(mon: PokemonMechanicsState) -> int | None:
    for effect in mon.effects:
        if effect.id == "confusion":
            return int(effect.turns or 0)
    return None


def _pokemon_timer_options(
    mon: PokemonMechanicsState,
) -> list[tuple[str, list[tuple[int, float]]]]:
    """`(payload key, [(value, probability)])` for each hidden timer on ``mon``."""

    if mon.fainted:
        return []
    options: list[tuple[str, list[tuple[int, float]]]] = []
    if mon.status == "slp":
        ticks = 2 if mon.ability_id == _DOUBLE_SLEEP_TICK_ABILITY else 1
        options.append(
            ("sleepTime", _remaining_options(SLEEP_DURATION_PRIOR, mon.status_counter, ticks))
        )
    elapsed = _confusion_elapsed(mon)
    if elapsed is not None:
        options.append(
            ("confusionTime", _remaining_options(CONFUSION_DURATION_PRIOR, elapsed, 1))
        )
    # Certain-but-non-default timers are emitted too: the worker's own fallback assumes
    # the modal three-opportunity sleep, which is simply wrong for an Early Bird sleeper.
    return options


def _side_axes(
    side: SideMechanicsState, side_key: str
) -> list[tuple[str, str, str, list[tuple[int, float]]]]:
    axes: list[tuple[str, str, str, list[tuple[int, float]]]] = []
    for mon in side.pokemon:
        for key, values in _pokemon_timer_options(mon):
            axes.append((side_key, mon.species_id, key, values))
    return axes


def enumerate_hidden_state_hypotheses(
    state: BattleMechanicsState, config: PolicyConfig | None = None
) -> list[HiddenStateHypothesis]:
    """Legal hidden-timer assignments for ``state``, most likely first.

    Returns a single empty hypothesis when nothing is hidden, so callers always have at
    least one branch to run. The list is truncated to
    ``PolicyConfig.exact_search_state_hypotheses`` and its weights are renormalised, so
    the returned weights always sum to one over the branches actually evaluated.
    """

    config = config or PolicyConfig()
    limit = int(config.exact_search_state_hypotheses)
    if limit < 1:
        raise ValueError("exact_search_state_hypotheses must be at least 1")
    axes = [
        *_side_axes(state.our_side, "our"),
        *_side_axes(state.opponent_side, "opponent"),
    ]
    if not axes:
        return [HiddenStateHypothesis(weight=1.0)]

    combinations: list[HiddenStateHypothesis] = []
    for choice in itertools.product(*[values for *_meta, values in axes]):
        weight = 1.0
        payload: dict[str, dict[str, dict[str, int]]] = {}
        for (side_key, species_id, key, _values), (value, probability) in zip(
            axes, choice, strict=True
        ):
            weight *= probability
            payload.setdefault(side_key, {}).setdefault(species_id, {})[key] = int(value)
        if weight <= 0.0:
            continue
        combinations.append(HiddenStateHypothesis(weight=weight, payload=payload))

    combinations.sort(key=lambda hypothesis: hypothesis.weight, reverse=True)
    kept = combinations[:limit]
    total = sum(hypothesis.weight for hypothesis in kept)
    return [
        HiddenStateHypothesis(weight=hypothesis.weight / total, payload=hypothesis.payload)
        for hypothesis in kept
    ]
