"""Stat-stage changes made by legal Reg M-C setup moves, for the fast search's simulation.

`data/champions/moves.json` does not export a move's ``boosts`` (see the note at
`vgc.evaluator`'s move table), so the stage changes live in this small explicit table.
It was derived on 2026-10-04 from the Showdown checkout's ``champions`` mod
(``Dex.mod("champions").moves``, pinned commit 9fb3a5b99, which is what `data/champions`
is exported from), restricted to moves that are legal in Reg M-C (``isNonstandard`` null
and in at least one learnset) AND that raise the user's, or its ally's, own stages.
`tests/test_search_setup_boosts.py::test_boost_table_matches_champions_move_data` checks
every entry against `data/champions/moves.json` (legal, Status, matching target) so a
data re-export that drops or retargets a move fails loudly.

Deliberately NOT in the table (each one is a modelling gap, not an oversight):

- Accuracy/evasion moves (Coil's accuracy, Double Team, Minimize): the damage calculator
  has no accuracy/evasion stage.
- No Retreat (also traps the user), Stockpile (stacking counter), Acupressure (random
  stat), Curse (Ghost/non-Ghost split), Magnetic Flux (ability-gated), Dragon Cheer
  (critical-hit stage, not a stat stage): the downside or the stat is not modelled.
- Attacks with self/secondary boosts (Close Combat's drops, Flame Charge's Speed): their
  stat changes already come with a damaging move, which the search simulates as damage.
- Stat DROPS on the opponent (Charm, Screech, ...): this table is for setup only.

Used only behind `PolicyConfig.search_apply_setup_boosts`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SetupBoost:
    """Stage changes one setup move applies when it succeeds.

    :param self_stages: stage deltas for the user (stat ids as in `PokemonState.boosts`).
    :param ally_stages: stage deltas for the user's partner (Coaching, Aromatic Mist;
        Howl gives both the user and the partner their own delta).
    :param hp_cost: fraction of the user's max HP paid to use the move (Belly Drum 1/2,
        Clangorous Soul 1/3). The move fails if current HP is not above that cost.
    :param doubled_in_sun: Growth's stage changes double in harsh sun.
    """

    self_stages: dict[str, int] = field(default_factory=dict)
    ally_stages: dict[str, int] = field(default_factory=dict)
    hp_cost: float = 0.0
    doubled_in_sun: bool = False


# Ids are Showdown move ids; stat ids match `vgc.damage.PokemonState.boosts` keys.
SETUP_BOOSTS: dict[str, SetupBoost] = {
    "acidarmor": SetupBoost({"def": 2}),
    "agility": SetupBoost({"spe": 2}),
    "amnesia": SetupBoost({"spd": 2}),
    "aromaticmist": SetupBoost(ally_stages={"spd": 1}),
    # Belly Drum sets Attack to +6 (Showdown applies +12, clamped) for half the user's HP.
    "bellydrum": SetupBoost({"atk": 12}, hp_cost=0.5),
    "bulkup": SetupBoost({"atk": 1, "def": 1}),
    "calmmind": SetupBoost({"spa": 1, "spd": 1}),
    "charge": SetupBoost({"spd": 1}),
    "clangoroussoul": SetupBoost(
        {"atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1}, hp_cost=0.33
    ),
    "coaching": SetupBoost(ally_stages={"atk": 1, "def": 1}),
    "coil": SetupBoost({"atk": 1, "def": 1}),
    "cosmicpower": SetupBoost({"def": 1, "spd": 1}),
    "cottonguard": SetupBoost({"def": 3}),
    "dragondance": SetupBoost({"atk": 1, "spe": 1}),
    "growth": SetupBoost({"atk": 1, "spa": 1}, doubled_in_sun=True),
    # Target "allies": the user and its partner each get +1 Attack.
    "howl": SetupBoost({"atk": 1}, {"atk": 1}),
    "irondefense": SetupBoost({"def": 2}),
    "nastyplot": SetupBoost({"spa": 2}),
    "quiverdance": SetupBoost({"spa": 1, "spd": 1, "spe": 1}),
    "rockpolish": SetupBoost({"spe": 2}),
    "shellsmash": SetupBoost({"atk": 2, "spa": 2, "spe": 2, "def": -1, "spd": -1}),
    "shelter": SetupBoost({"def": 2}),
    "shiftgear": SetupBoost({"atk": 1, "spe": 2}),
    "swordsdance": SetupBoost({"atk": 2}),
    # Tidy Up's hazard/Substitute clearing is not modelled; only its +1 Atk/+1 Spe is.
    "tidyup": SetupBoost({"atk": 1, "spe": 1}),
}

# Abilities that rewrite the stages a move applies to their holder.
_INVERTING_ABILITIES = frozenset({"contrary"})
_DOUBLING_ABILITIES = frozenset({"simple"})
_MAX_STAGE = 6


def _stage_delta(delta: int, ability: str | None, sun: bool, doubled_in_sun: bool) -> int:
    if doubled_in_sun and sun:
        delta *= 2
    if ability in _INVERTING_ABILITIES:
        delta = -delta
    elif ability in _DOUBLING_ABILITIES:
        delta *= 2
    return delta


def apply_stages(
    boosts: dict[str, int],
    stages: dict[str, int],
    *,
    ability: str | None = None,
    sun: bool = False,
    doubled_in_sun: bool = False,
) -> None:
    """Add ``stages`` to ``boosts`` in place, clamped to +/-6 (Contrary and Simple honoured)."""

    for stat, delta in stages.items():
        change = _stage_delta(delta, ability, sun, doubled_in_sun)
        boosts[stat] = max(-_MAX_STAGE, min(_MAX_STAGE, boosts.get(stat, 0) + change))
