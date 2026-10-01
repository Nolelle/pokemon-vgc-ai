"""Protocol rewrites for poke-env 0.15 parsing bugs, applied just before `parse_message`.

Callers that record or replay the raw stream (`BattleMemory`, decision-replay recorders)
keep the original line; only the copy handed to poke-env is rewritten.
"""

from __future__ import annotations

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
