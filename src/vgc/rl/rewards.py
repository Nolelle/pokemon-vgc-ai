"""Potential-based reward shaping for PPO (dense, optimal-policy-preserving signal).

`board_potential` is the potential function `Phi(s)` used by `vgc.rl.ppo.finish_episode`
when `PpoConfig.reward_shaping_coef > 0`: the per-step shaped reward added on top of the
sparse terminal +-1 is `coef * (gamma * Phi(s_{t+1}) - Phi(s_t))`, which telescopes to
zero over a full episode and therefore provably does not change the optimal policy --
see Ng, Harada & Russell (1999), "Policy invariance under reward transformations". It
only densifies the training signal.
"""

from __future__ import annotations

from typing import Any


def _hp_fraction(pokemon: Any) -> float:
    """A single Pokemon's remaining HP fraction, deployment-available and clamped.

    Fainted -> 0.0. Missing/unreadable `current_hp_fraction` (a stub in tests, a
    Pokemon poke-env hasn't populated yet) -> 1.0 (treated as full HP -- the same
    "unrevealed/unknown means full HP" convention used for opponent Pokemon that
    haven't been sent out yet).
    """

    if pokemon is None:
        return 1.0
    if bool(getattr(pokemon, "fainted", False)):
        return 0.0
    fraction = getattr(pokemon, "current_hp_fraction", None)
    if fraction is None:
        return 1.0
    try:
        fraction = float(fraction)
    except (TypeError, ValueError):
        return 1.0
    if fraction != fraction:  # NaN guard
        return 1.0
    return max(0.0, min(1.0, fraction))


def _mean_hp_fraction(pokemon_list: list[Any]) -> float:
    if not pokemon_list:
        # No roster information at all (e.g. before teams are populated) -- neutral
        # "full HP" default so an uninformative state contributes no bias.
        return 1.0
    return sum(_hp_fraction(pokemon) for pokemon in pokemon_list) / len(pokemon_list)


def _our_mean_hp_fraction(battle: Any) -> float:
    """Our own roster's mean remaining HP fraction -- always fully known (it's our team)."""

    team = getattr(battle, "team", None) or {}
    values = team.values() if hasattr(team, "values") else team
    return _mean_hp_fraction(list(values))


def _opponent_mean_hp_fraction(battle: Any) -> float:
    """Opponent roster's mean remaining HP fraction, deployment-available only.

    Prefers the full previewed 6 (`teampreview_opponent_team`, populated once during
    team preview and never cleared -- same source `vgc.rl.encoding` uses for the
    opponent archetype one-hot) so an opponent Pokemon that hasn't been sent out yet is
    still counted, at full HP (a real opponent could always still have it alive and
    healthy). Falls back to whatever has actually been revealed
    (`battle.opponent_team`) when no preview is available (e.g. a minimal battle stub,
    or before team preview has been parsed).
    """

    preview = list(getattr(battle, "teampreview_opponent_team", None) or [])
    revealed_container = getattr(battle, "opponent_team", None) or {}
    revealed_list = list(
        revealed_container.values() if hasattr(revealed_container, "values") else revealed_container
    )
    if not preview:
        return _mean_hp_fraction(revealed_list)

    revealed_by_species: dict[str, Any] = {}
    for pokemon in revealed_list:
        species = getattr(pokemon, "species", None)
        if species:
            revealed_by_species[species] = pokemon

    fractions = []
    for previewed in preview:
        species = getattr(previewed, "species", None)
        pokemon = revealed_by_species.get(species) if species else None
        # Not yet revealed this battle -> treat as full HP (deployment-available
        # convention: we cannot know it has taken damage until we see it).
        fractions.append(_hp_fraction(pokemon) if pokemon is not None else 1.0)
    return sum(fractions) / len(fractions) if fractions else 1.0


def board_potential(battle: Any) -> float:
    """Deployment-available board-state potential `Phi(s)`, bounded to `[-1, 1]`.

    Formula: `(mean HP fraction across our roster) - (mean HP fraction across the
    opponent's roster, treating any previewed-but-unrevealed opponent Pokemon as full
    HP)`. Positive means we are ahead on aggregate remaining HP, negative means we are
    behind, ~0 means the sides are (or appear, given what a real opponent would show
    us) even.

    Every input mirrors `vgc.rl.encoding.encode_meta_context`'s deployment-available
    rule: our own `Pokemon.fainted`/`current_hp_fraction` (always fully known -- it's
    our team), the opponent's REVEALED `Pokemon.fainted`/`current_hp_fraction`, and
    team-preview species membership (to know the opponent's roster without peeking at
    hidden state). Never reads a hidden ground-truth field the sim knows but a real
    opponent wouldn't reveal.

    Guarded end-to-end like `encode_meta_context`: any missing/malformed battle data
    (a stub battle in tests, a battle before team preview has populated anything, an
    empty team) falls back to `0.0` rather than raising -- this must never throw during
    a live battle.
    """

    try:
        potential = _our_mean_hp_fraction(battle) - _opponent_mean_hp_fraction(battle)
    except Exception:
        return 0.0
    return float(max(-1.0, min(1.0, potential)))
