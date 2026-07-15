"""Opponent `PokemonState` assembly from a poke-env observed Pokemon.

This format's ladder always runs Open Team Sheets (see `PolicyConfig.
accept_open_team_sheet`, `vgc.agent.VgcPlayer`) -- every opponent Pokemon's
species/moves/item/ability is revealed before team preview, for the whole battle. Stat
Points and nature are NOT part of Open Team Sheets and stay hidden the entire battle;
`opponent_state` fills that gap with real usage-stats data when available
(`load_usage_spreads`) or `vgc.stats.default_opponent_spread`/`default_opponent_nature`
otherwise.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from vgc.config import DATA_DIR
from vgc.damage import PokemonState
from vgc.stats import SPSpread, default_opponent_nature, default_opponent_spread

# Not committed yet (see CLAUDE.md's data/champions/ note -- this directory is a Phase
# 2a stub, not exported data): a future usage-stats pipeline would populate
# data/usage/spreads.json. load_usage_spreads gracefully no-ops until that exists.
USAGE_DIR = DATA_DIR.parent / "usage"
DEFAULT_USAGE_SPREADS_PATH = USAGE_DIR / "spreads.json"

# poke_env.battle.status.Status member names, lowercased, that are damage-relevant here
# (only "brn" actually changes anything in vgc.damage today, but the others are passed
# through for forward compatibility / other future evaluator logic. "fnt" is dropped --
# a fainted Pokemon isn't a meaningful damage_range() input).
_RECOGNIZED_STATUS_IDS = frozenset({"brn", "par", "psn", "tox", "slp", "frz"})


class ObservedPokemon(Protocol):
    """Structural type for what `opponent_state` needs -- satisfied by poke-env's
    `Pokemon` (e.g. `DoubleBattle.opponent_active_pokemon_1/_2`, or any value from
    `DoubleBattle.opponent_team`), but written as a Protocol so tests/callers can pass a
    lightweight stand-in without depending on poke-env's constructor.
    """

    species: str
    item: str | None
    ability: str | None
    boosts: dict[str, int]
    status: Any  # poke_env.battle.status.Status | None
    current_hp_fraction: float


@lru_cache(maxsize=8)
def load_usage_spreads(path: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Load a JSON file of common Stat Point spreads per species:

        {"garchomp": [{"sp": {"hp": 30, "atk": 4, "spe": 32}, "nature": "adamant",
                        "weight": 0.42}, ...], ...}

    `weight` is any relative popularity score (higher = more common); `opponent_state`
    picks the highest-weight entry per species.

    Graceful fallback: if `path` is None, defaults to `data/usage/spreads.json`
    (`DEFAULT_USAGE_SPREADS_PATH`); if that file doesn't exist yet -- there is no usage
    export pipeline as of Phase 2a, this is a stub for a later one -- returns an empty
    dict rather than raising, and every caller falls back to
    `vgc.stats.default_opponent_spread` instead.
    """
    resolved = Path(path) if path is not None else DEFAULT_USAGE_SPREADS_PATH
    if not resolved.exists():
        return {}
    with resolved.open() as file:
        return json.load(file)


def _usage_spread_for_species(
    species_id: str, usage: dict[str, list[dict[str, Any]]]
) -> tuple[SPSpread, str] | None:
    entries = usage.get(species_id)
    if not entries:
        return None
    best = max(entries, key=lambda entry: entry.get("weight", 0))
    return best["sp"], best["nature"]


def normalize_item(item: str | None) -> str | None:
    # poke-env uses the sentinel string "unknown_item" (GenData.UNKNOWN_ITEM) before an
    # item is revealed; under Open Team Sheets this shouldn't linger past team preview,
    # but handle it defensively anyway. Public (not just for opponent_state) since
    # vgc.evaluator reuses it for our own Pokemon's item field too.
    if not item or item == "unknown_item":
        return None
    return item


def normalize_status(status: Any) -> str | None:
    # Public for the same reason as normalize_item -- vgc.evaluator builds PokemonStates
    # for our own side (poke-env Status enum) with the same normalization rules as
    # opponent_state uses for theirs.
    if status is None:
        return None
    status_id = getattr(status, "name", str(status)).lower()
    return status_id if status_id in _RECOGNIZED_STATUS_IDS else None


# Backwards-compatible private aliases (this module's own code below used the
# underscore-prefixed names before they were made public for vgc.evaluator's reuse).
_normalize_item = normalize_item
_normalize_status = normalize_status


def opponent_state(
    pokemon: ObservedPokemon,
    usage: dict[str, list[dict[str, Any]]] | None = None,
    nature_override: str | None = None,
) -> PokemonState:
    """Build a `PokemonState` for an opponent Pokemon observed through poke-env.

    :param pokemon: e.g. `battle.opponent_active_pokemon_1`/`_2`
        (`poke_env.battle.double_battle.DoubleBattle`) or any value from
        `battle.opponent_team`.
    :param usage: pre-loaded `load_usage_spreads()` result to reuse across many calls
        (avoids re-reading the file per Pokemon); `None` loads the default path (cheap
        after the first call -- `load_usage_spreads` is `lru_cache`d).
    :param nature_override: a higher-confidence hidden nature from an exactly recognized
        curated team. The live Open Team Sheet still supplies item/ability/moves; this
        only fills information the sheet does not reveal.
    """
    if usage is None:
        usage = load_usage_spreads()

    species_id = pokemon.species
    spread_and_nature = _usage_spread_for_species(species_id, usage)
    if spread_and_nature is not None:
        sp_spread, nature = spread_and_nature
    else:
        sp_spread = default_opponent_spread(species_id)
        nature = default_opponent_nature(species_id)
    if nature_override is not None:
        nature = nature_override

    boosts = {
        stat: value
        for stat, value in (pokemon.boosts or {}).items()
        if stat in ("atk", "def", "spa", "spd", "spe") and value
    }

    state = PokemonState(
        species_id=species_id,
        sp_spread=sp_spread,
        nature=nature,
        boosts=boosts,
        status=_normalize_status(pokemon.status),
        item=_normalize_item(pokemon.item),
        ability=pokemon.ability or None,
    )
    # poke-env reports current_hp/max_hp for OPPONENT Pokemon on a 0-100 or pixel scale,
    # not real Champions HP (see poke_env.battle.pokemon.Pokemon.current_hp's own
    # docstring) -- current_hp_fraction is the scale-independent number, so rescale
    # against our own computed max_hp instead of trusting pokemon.current_hp directly.
    fraction = getattr(pokemon, "current_hp_fraction", 1.0) or 0.0
    state.current_hp = round(state.max_hp() * fraction)
    return state
