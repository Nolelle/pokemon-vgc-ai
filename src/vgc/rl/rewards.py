"""Potential-based reward shaping for PPO (dense, optimal-policy-preserving signal).

`board_potential` is the potential function `Phi(s)` used by `vgc.rl.ppo.finish_episode`
when `PpoConfig.reward_shaping_coef > 0`: the per-step shaped reward added on top of the
sparse terminal +-1 is `coef * (gamma * Phi(s_{t+1}) - Phi(s_t))`, which telescopes to
zero over a full episode and therefore provably does not change the optimal policy --
see Ng, Harada & Russell (1999), "Policy invariance under reward transformations". It
only densifies the training signal.

`Phi(s)` is a weighted sum of an HP-differential term (the original, still dominant,
component) plus four strategic terms -- Tailwind, screens, stat boosts, and Trick Room
-- motivated by the fact that of the 58 archetype-pool teams (see
`data/selfplay/archetype_pool/`), 34 run Tailwind, 16 screens, 16 Nasty Plot, 13 Swords
Dance, 9 Shell Smash, and 5 Trick Room: the HP-only potential was blind to all of this,
so setup/speed-control moves registered as "nothing happened" and got no immediate
credit. Any function of state is a valid potential (Ng et al.'s policy-invariance result
holds for arbitrary `Phi`), so enriching it costs nothing in theory -- it only needs to
stay deployment-available (no hidden ground-truth fields) and cheap (called every
decision).

Each strategic term's weight is a named module-level constant, individually commented
below, and their sum is kept comfortably smaller than the HP weight so they act as
tie-breakers/credit signals rather than the main objective. `_potential_components`
exposes the unclamped per-term contributions (already weighted) so tests can assert on
individual terms; `board_potential` sums them and clamps to `[-1, 1]`.
"""

from __future__ import annotations

from typing import Any

# --- component weights (every strategic term is gated behind a named, commented
# constant -- no bare magic numbers standing in for a strategic judgment call) ---------

# HP differential: the original component and still the dominant signal -- winning the
# HP race is the closest proxy to actually winning the game. Chosen so it alone can
# still reach +-1 the way the original HP-only potential did before clamping, and so it
# comfortably outweighs the sum of every strategic term below (0.07+0.05+0.08+0.05 =
# 0.25 max magnitude vs 0.75 here).
_HP_WEIGHT = 0.75

# Tailwind: doubles Speed for the side that has it, which is a huge tempo swing in a
# doubles format (34/58 archetype-pool teams run it). Kept as a simple binary
# presence term rather than duration-decayed -- see `_tailwind_component`'s docstring
# for why decaying it robustly isn't as simple as it looks.
_TAILWIND_WEIGHT = 0.07

# Screens (Light Screen / Reflect / Aurora Veil): meaningfully reduces incoming
# damage but doesn't swing turn order the way Tailwind or Trick Room do, so it gets a
# smaller weight than either.
_SCREENS_WEIGHT = 0.05

# Stat boosts: credits setup (Nasty Plot/Swords Dance/Shell Smash are 16/13/9 of the
# 58 archetype-pool teams) the turn it happens, instead of waiting for the payoff hit
# to show up as an HP swing one or more turns later.
_BOOST_WEIGHT = 0.08

# Trick Room: sign flips based on who it favors (see `_trick_room_component`) -- this
# is the most situational term (only 5/58 archetype-pool teams run it) and the
# riskiest to get wrong, so it gets the smallest weight; a bad speed estimate should
# not be able to dominate the shaping signal.
_TRICK_ROOM_WEIGHT = 0.05

# Stats summed for the boost term: the five core combat stages, deliberately excluding
# accuracy/evasion (those track evasion-stacking degenerate play, not "setup", and
# shouldn't get credit from a shaping term meant to reward standard Nasty
# Plot/Swords Dance/Shell Smash/Calm Mind lines).
_BOOST_STATS = ("atk", "def", "spa", "spd", "spe")

# Normalizer for the boost term: up to 2 active Pokemon per side (doubles) x 5 stats x
# a max stage of 6 = 60 -- the largest the raw (ours - theirs) sum could be, used to
# keep the term in [-1, 1] before weighting.
_BOOST_STATS_MAX_MAGNITUDE = len(_BOOST_STATS) * 6 * 2

# Normalizer for the screens term: Light Screen, Reflect, and Aurora Veil, so the raw
# (ours - theirs) count is in [-3, 3].
_MAX_SCREEN_COUNT = 3


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


def _hp_component(battle: Any) -> float:
    """The original HP-differential term, weighted by `_HP_WEIGHT`.

    Formula unchanged from the HP-only potential: `(mean HP fraction across our
    roster) - (mean HP fraction across the opponent's roster, treating any
    previewed-but-unrevealed opponent Pokemon as full HP)`, in `[-1, 1]` before
    weighting.
    """

    raw = _our_mean_hp_fraction(battle) - _opponent_mean_hp_fraction(battle)
    return _HP_WEIGHT * max(-1.0, min(1.0, raw))


def _side_condition_present(conditions: Any, condition: Any) -> bool:
    try:
        return condition in conditions
    except TypeError:
        return False


def _tailwind_component(battle: Any) -> float:
    """Tailwind presence term, weighted by `_TAILWIND_WEIGHT`.

    Binary presence (not duration-decayed), even though poke-env's
    `side_conditions`/`opponent_side_conditions` dicts technically store the turn
    Tailwind was set up (so a remaining-duration estimate is computable in principle:
    Tailwind always lasts exactly 4 turns in this mod, unmodified by any item). We keep
    it binary anyway for consistency with the screens term below, where a duration
    estimate is NOT reliably computable (Light Clay extends screens from 5 to 8 turns,
    but Light Clay is carried by whichever Pokemon set the screen up -- which may since
    have switched out or fainted, at which point deployment-available state no longer
    lets us recover whether the item was in play). Using a matching binary convention
    for both keeps the two terms comparable instead of one being finer-grained than the
    other for an arbitrary implementation reason.
    """

    from poke_env.battle.side_condition import SideCondition

    our_conditions = getattr(battle, "side_conditions", None) or {}
    opp_conditions = getattr(battle, "opponent_side_conditions", None) or {}
    ours = _side_condition_present(our_conditions, SideCondition.TAILWIND)
    theirs = _side_condition_present(opp_conditions, SideCondition.TAILWIND)
    return _TAILWIND_WEIGHT * (int(ours) - int(theirs))


def _screens_component(battle: Any) -> float:
    """Screens (Light Screen/Reflect/Aurora Veil) term, weighted by `_SCREENS_WEIGHT`."""

    from poke_env.battle.side_condition import SideCondition

    screen_conditions = (
        SideCondition.LIGHT_SCREEN,
        SideCondition.REFLECT,
        SideCondition.AURORA_VEIL,
    )
    our_conditions = getattr(battle, "side_conditions", None) or {}
    opp_conditions = getattr(battle, "opponent_side_conditions", None) or {}
    our_count = sum(_side_condition_present(our_conditions, c) for c in screen_conditions)
    opp_count = sum(_side_condition_present(opp_conditions, c) for c in screen_conditions)
    raw = (our_count - opp_count) / _MAX_SCREEN_COUNT
    return _SCREENS_WEIGHT * raw


def _active_boost_sum(active_pokemon: list[Any]) -> float:
    total = 0.0
    for pokemon in active_pokemon or []:
        if pokemon is None or bool(getattr(pokemon, "fainted", False)):
            continue
        boosts = getattr(pokemon, "boosts", None) or {}
        for stat in _BOOST_STATS:
            try:
                total += float(boosts.get(stat, 0) or 0)
            except (TypeError, ValueError, AttributeError):
                continue
    return total


def _boost_component(battle: Any) -> float:
    """Net active-Pokemon stat-boost term, weighted by `_BOOST_WEIGHT`.

    Sums `_BOOST_STATS` boost stages (-6..+6 each) across our ACTIVE Pokemon minus the
    same sum for the opponent's ACTIVE Pokemon (not the whole roster -- a boosted
    benched Pokemon isn't contributing to the board right now), normalized by
    `_BOOST_STATS_MAX_MAGNITUDE` so the raw term is in `[-1, 1]` before weighting.
    """

    our_active = getattr(battle, "active_pokemon", None) or []
    opp_active = getattr(battle, "opponent_active_pokemon", None) or []
    raw = (_active_boost_sum(list(our_active)) - _active_boost_sum(list(opp_active))) / (
        _BOOST_STATS_MAX_MAGNITUDE
    )
    return _BOOST_WEIGHT * max(-1.0, min(1.0, raw))


def _mean_active_speed(active_pokemon: list[Any], *, ours: bool) -> float | None:
    """Mean deployment-available effective Speed across non-fainted active Pokemon.

    Reuses `vgc.evaluator.effective_speed` (boost stage, Choice Scarf, paralysis) over
    a `PokemonState` built the same way the rest of the RL/evaluator code builds one:
    `vgc.evaluator._our_pokemon_state` for our own side (Stat Points/nature are
    genuinely known from the Teambuilder team) and `vgc.sets.opponent_state` for the
    opponent's side (estimated from usage-stats priors -- never the hidden true
    spread). Returns `None` if no active Pokemon's state could be built (missing data,
    a test stub without the needed poke-env attributes, etc.) rather than guessing.
    """

    from vgc.evaluator import _our_pokemon_state, effective_speed
    from vgc.sets import opponent_state

    speeds = []
    for pokemon in active_pokemon or []:
        if pokemon is None or bool(getattr(pokemon, "fainted", False)):
            continue
        try:
            state = _our_pokemon_state(pokemon) if ours else opponent_state(pokemon)
            speeds.append(effective_speed(state))
        except Exception:
            continue
    if not speeds:
        return None
    return sum(speeds) / len(speeds)


def _trick_room_component(battle: Any) -> float:
    """Trick Room term, weighted by `_TRICK_ROOM_WEIGHT`.

    Trick Room reverses speed priority for BOTH sides, so its sign must depend on who
    it actually favors, not just whether it's up: 0 when Trick Room isn't active; when
    active, positive if OUR active Pokemon are (on average) slower than the
    opponent's -- Trick Room turns that disadvantage into a turn-order advantage -- and
    negative if we're faster (Trick Room is then working against us). If either side's
    speed can't be estimated from deployment-available state, the term is 0 rather than
    guessing a sign.
    """

    from poke_env.battle.field import Field

    fields = getattr(battle, "fields", None) or {}
    if not _side_condition_present(fields, Field.TRICK_ROOM):
        return 0.0

    our_speed = _mean_active_speed(list(getattr(battle, "active_pokemon", None) or []), ours=True)
    opp_speed = _mean_active_speed(
        list(getattr(battle, "opponent_active_pokemon", None) or []), ours=False
    )
    if our_speed is None or opp_speed is None:
        return 0.0
    if our_speed == opp_speed:
        return 0.0
    # We're slower (our_speed < opp_speed) -> Trick Room favors us -> positive.
    sign = 1.0 if our_speed < opp_speed else -1.0
    return _TRICK_ROOM_WEIGHT * sign


def _potential_components(battle: Any) -> dict[str, float]:
    """Per-term (already-weighted) contributions to `Phi(s)`, individually guarded.

    Exposed (private, but importable by tests) so tests can assert on individual
    strategic terms without reimplementing the weighting/normalization logic, and so a
    failure computing one term (e.g. missing poke-env attributes on a stub) can never
    zero out the others -- each term independently falls back to `0.0`.
    """

    components: dict[str, float] = {}
    for name, fn in (
        ("hp", _hp_component),
        ("tailwind", _tailwind_component),
        ("screens", _screens_component),
        ("boosts", _boost_component),
        ("trick_room", _trick_room_component),
    ):
        try:
            components[name] = float(fn(battle))
        except Exception:
            components[name] = 0.0
    return components


def board_potential(battle: Any) -> float:
    """Deployment-available board-state potential `Phi(s)`, bounded to `[-1, 1]`.

    A weighted sum of an HP-differential term (see `_hp_component`, still the
    dominant component) plus four strategic terms -- Tailwind (`_tailwind_component`),
    screens (`_screens_component`), active stat boosts (`_boost_component`), and Trick
    Room (`_trick_room_component`, whose sign depends on who it favors). See the module
    docstring for the motivation and each weight constant above for the reasoning
    behind its magnitude.

    Every input mirrors `vgc.rl.encoding.encode_meta_context`'s deployment-available
    rule: our own state is always fully known (it's our team), the opponent's is only
    ever read from what a real opponent would reveal (REVEALED Pokemon/side
    conditions/fields, plus team-preview species membership for roster HP). Never reads
    a hidden ground-truth field the sim knows but a real opponent wouldn't reveal.

    Guarded end-to-end like `encode_meta_context`: any missing/malformed battle data
    (a stub battle in tests, a battle before team preview has populated anything, an
    empty team) falls back to `0.0` rather than raising -- this must never throw during
    a live battle. Each component in `_potential_components` is additionally
    individually guarded, so one broken term can't take the others down with it.
    """

    try:
        total = sum(_potential_components(battle).values())
    except Exception:
        return 0.0
    return float(max(-1.0, min(1.0, total)))
