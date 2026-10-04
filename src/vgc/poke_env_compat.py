"""Protocol rewrites for poke-env 0.15 parsing bugs, applied just before `parse_message`.

Callers that record or replay the raw stream (`BattleMemory`, decision-replay recorders)
keep the original line; only the copy handed to poke-env is rewritten.
"""

from __future__ import annotations

from poke_env.battle.pokemon import Pokemon

_ROUND_CHAIN_TAGS = ("[from] move: Round", "[from]move: Round")


def normalize_for_poke_env(split: list[str]) -> list[str]:
    """Return `split` (a `|`-split protocol line) with known poke-env traps removed.

    Round chain: when allies both use Round, Showdown moves the second one up and tags
    its line `|move|<mon>|Round|<target>|[from] move: Round`. poke-env treats that tag
    as a borrowed move (like Copycat), skips the reveal, then indexes `mon.moves["round"]`
    -- a `KeyError` for an opponent that has not revealed Round yet. The Pokemon did use
    its own Round, so dropping the tag gives poke-env the correct reading.
    """
    if (
        len(split) > 3
        and split[1] == "move"
        and split[3] == "Round"
        and any(part in _ROUND_CHAIN_TAGS for part in split[4:])
    ):
        return [part for part in split if part not in _ROUND_CHAIN_TAGS]
    return split


# --- Mega Evolution keeps the exact forme -------------------------------------------------
#
# Showdown sends `|detailschange|<mon>|Lucario-Mega-Z, ...` and THEN `|-mega|<mon>|Lucario|...`.
# poke-env's `detailschange` handler loads the exact Mega forme, but its `-mega` handler
# calls `Pokemon.mega_evolve`, which re-derives "<species>mega" from the base species and
# overwrites the stats/types/ability with the plain Mega whenever that id exists -- so Mega
# Lucario Z, Garchomp Z and Absol Z became their plain Megas (~9% of Mega Evolutions in the
# M-C corpus). `mega_evolve` is redundant once the forme change has been applied.

_original_mega_evolve = Pokemon.mega_evolve


def _mega_evolve_keeping_exact_forme(self: Pokemon, stone: str) -> None:
    if self.forme_change_ability is not None:
        self.temporary_ability = None  # the one side effect `mega_evolve` always has
        return
    _original_mega_evolve(self, stone)


if not getattr(Pokemon.mega_evolve, "_vgc_keeps_exact_forme", False):
    _mega_evolve_keeping_exact_forme._vgc_keeps_exact_forme = True  # type: ignore[attr-defined]
    Pokemon.mega_evolve = _mega_evolve_keeping_exact_forme  # type: ignore[method-assign]
