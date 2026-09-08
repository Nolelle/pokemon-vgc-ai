"""Assembles a pool of legal, corpus-informed opponent teams for self-play data
generation (`selfplay/run_selfplay.py`).

## Design

Real per-species set frequencies (`data/usage/set_priors.json`, `vgc.sets.
load_set_priors`) plus Stat Point/nature usage (`data/usage/spreads.json`,
`vgc.sets.load_usage_spreads`) already carry exactly the corpus-derived information
needed to assemble a plausible legal set for a given species -- the top `TOP_N_SPECIES`
by tracked corpus appearances form a species pool, `build_priority_species` picks each
one's most-common (legal) ability/moves plus a RANKED list of its most-common (legal)
items, plus its most-common Stat Point spread (falling back to `vgc.stats.
default_opponent_spread`/`default_opponent_nature` when a species has no spread entry),
and `sample_teams` draws `n_teams` distinct-within-team 6-species combinations from that
pool (Species Clause: no repeats WITHIN one team; different teams MAY reuse a species,
which is fine -- a real corpus-derived pool of ~20 teams from ~30 popular species will
naturally overlap some).

`sample_team_with_fixed_core` is a variant of `sample_teams` for callers that need to
pin a few species (e.g. `tools/build_archetype_pool.py`'s archetype-defining core) while
still drawing the remaining slots at random from a pool.

`resolve_team_items` is a separate, TEAM-level step (not per-species): this format's
Item Clause caps every item at 1 per team, but two popular species sampled onto the same
team often share the same single most-popular item (Choice Scarf is extremely common
across many species in this corpus) -- greedily walks each species' ranked item list and
falls back to itemless (legal, if unusual) only if every candidate is already taken by an
earlier slot on the same team.

## Legality is NOT fully guaranteed by this module alone

This module does its best to only pick moves/abilities/items that are actually legal for
that species (checked against `data/champions/{learnsets,species,items}.json`), but the
authoritative legality check is still the real Showdown `validate-team` CLI (CLAUDE.md's
mandated validation step for any hand-or-generated team) -- `tools/build_selfplay_pool.py`
is the CLI that shells out to it and drops any generated team that fails, so a
best-effort assembly here plus real validation there is the actual legality guarantee.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from vgc.damage import to_id
from vgc.data import load_items, load_learnsets, load_moves, load_species
from vgc.stats import SPSpread, STAT_IDS, default_opponent_nature, default_opponent_spread

TOP_N_SPECIES = 30
TEAM_SIZE = 6
DEFAULT_N_TEAMS = 20
MOVES_PER_SET = 4
# How many ranked item candidates each GeneratedSet keeps -- enough headroom for
# resolve_team_items' greedy Item-Clause fallback to usually find a still-unused one.
ITEM_CANDIDATES_PER_SET = 6


@dataclass(frozen=True)
class GeneratedSet:
    """One species' assembled set. `item_candidates` is ranked highest-corpus-frequency
    first; `team_to_packed` never reads it directly -- callers must resolve one concrete
    item per species first (see `resolve_team_items`, this format's Item Clause is a
    TEAM-level constraint, not a per-species one).
    """

    species_id: str
    item_candidates: tuple[str, ...]
    ability: str
    moves: tuple[str, ...]
    nature: str
    sp: SPSpread


def _pick_ability(prior_abilities: dict[str, int], species_data: dict[str, Any]) -> str:
    """Highest-corpus-frequency ability that's actually one of this species' legal
    abilities; falls back to the species' own first listed ability (always legal) if the
    prior is empty/all-invalid (e.g. a species with too few tracked appearances for
    `tools/build_set_priors.py` to have ever recorded an `-ability|` reveal).
    """
    valid_ids = {to_id(name) for name in (species_data.get("abilities") or {}).values() if name}
    ranked = sorted(prior_abilities.items(), key=lambda kv: -kv[1])
    for ability_id, _count in ranked:
        if ability_id in valid_ids:
            return ability_id
    for name in (species_data.get("abilities") or {}).values():
        normalized = to_id(name)
        if normalized:
            return normalized
    return ""


def _pick_items(prior_items: dict[str, int], legal_item_ids: set[str]) -> tuple[str, ...]:
    """Ranked (highest corpus frequency first) legal item ids, up to
    `ITEM_CANDIDATES_PER_SET` -- see `resolve_team_items` for why this is a ranked list
    rather than a single choice.
    """
    ranked = sorted(prior_items.items(), key=lambda kv: -kv[1])
    picked = [item_id for item_id, _count in ranked if item_id in legal_item_ids]
    return tuple(picked[:ITEM_CANDIDATES_PER_SET])


def _pick_moves(
    prior_moves: dict[str, int], learnset_ids: set[str], legal_move_ids: set[str]
) -> list[str]:
    """Up to `MOVES_PER_SET` moves, ranked by corpus frequency, restricted to moves this
    species can actually learn AND that exist in `data/champions/moves.json`. Padded with
    other legal learnset moves (sorted for determinism) when the corpus prior is thin --
    real ladder sets aren't always fully captured by the (finite) replay corpus.
    """
    ranked = sorted(prior_moves.items(), key=lambda kv: -kv[1])
    picked: list[str] = []
    for move_id, _count in ranked:
        if len(picked) >= MOVES_PER_SET:
            break
        if move_id in learnset_ids and move_id in legal_move_ids and move_id not in picked:
            picked.append(move_id)
    if len(picked) < MOVES_PER_SET:
        for move_id in sorted(learnset_ids):
            if len(picked) >= MOVES_PER_SET:
                break
            if move_id in legal_move_ids and move_id not in picked:
                picked.append(move_id)
    return picked


def build_priority_species(
    set_priors: dict[str, Any],
    spreads: dict[str, list[dict[str, Any]]],
    *,
    top_n: int = TOP_N_SPECIES,
) -> list[GeneratedSet]:
    """`GeneratedSet`s for the top `top_n` species by tracked corpus appearances
    (`set_priors["species"][id]["appearances"]`). Skips a species entirely if it isn't a
    recognized champions-mod species (defensive -- `set_priors.json` is itself built from
    already-`to_id`-normalized replay data, so this shouldn't normally trigger) or if
    fewer than 2 legal moves could be assembled for it (too little learnset/corpus
    overlap to build a usable set).
    """
    species_data_all = load_species()
    legal_item_ids = set(load_items().keys())
    legal_move_ids = set(load_moves().keys())
    learnsets_all = load_learnsets()

    entries: dict[str, Any] = (set_priors.get("species") or {}) if set_priors else {}
    ranked_species = sorted(entries.items(), key=lambda kv: -(kv[1].get("appearances", 0) or 0))[
        :top_n
    ]

    result: list[GeneratedSet] = []
    for species_id, entry in ranked_species:
        species_data = species_data_all.get(species_id)
        if not species_data:
            continue
        learnset_ids = set(learnsets_all.get(species_id) or {})
        moves = _pick_moves(entry.get("moves") or {}, learnset_ids, legal_move_ids)
        if len(moves) < 2:
            continue
        ability = _pick_ability(entry.get("abilities") or {}, species_data)
        item_candidates = _pick_items(entry.get("items") or {}, legal_item_ids)

        spread_entries = (spreads or {}).get(species_id)
        if spread_entries:
            best = max(spread_entries, key=lambda e: e.get("weight", 0))
            sp, nature = dict(best["sp"]), best["nature"]
        else:
            sp, nature = default_opponent_spread(species_id), default_opponent_nature(species_id)

        result.append(
            GeneratedSet(
                species_id=species_id,
                item_candidates=item_candidates,
                ability=ability,
                moves=tuple(moves),
                nature=nature,
                sp=sp,
            )
        )
    return result


def sample_teams(
    pool: list[GeneratedSet], *, n_teams: int = DEFAULT_N_TEAMS, seed: int = 0
) -> list[list[GeneratedSet]]:
    """`n_teams` distinct-within-team 6-species samples from `pool`, deterministic for a
    given `seed`. Returns `[]` if `pool` has fewer than `TEAM_SIZE` entries (can't build
    even one legal 6-mon team).
    """
    if len(pool) < TEAM_SIZE:
        return []
    rng = random.Random(seed)
    return [rng.sample(pool, TEAM_SIZE) for _ in range(n_teams)]


def sample_team_with_fixed_core(
    core: list[GeneratedSet], flex_pool: list[GeneratedSet], *, seed: int = 0
) -> list[GeneratedSet] | None:
    """A `TEAM_SIZE`-species team that keeps every `core` entry exactly as given and
    fills the remaining slots with a deterministic random sample from `flex_pool`.

    Built for `tools/build_archetype_pool.py`'s within-archetype variant generation:
    it needs to fix a small defining-core species set (e.g. the weather setter plus its
    abuser) while still varying the rest of the team from the corpus-priority pool --
    plain `sample_teams` (which samples ALL `TEAM_SIZE` slots uniformly from one pool)
    can't express that constraint. `flex_pool` entries whose `species_id` is already in
    `core` are excluded first (Species Clause: no repeats within one team) so callers
    don't have to pre-filter their pool themselves.

    Returns `None` if, after that exclusion, `flex_pool` has fewer entries than the
    number of flex slots still needed -- can't legally fill out the team.
    """
    if len(core) > TEAM_SIZE:
        raise ValueError(f"core has {len(core)} entries, more than TEAM_SIZE={TEAM_SIZE}")
    flex_needed = TEAM_SIZE - len(core)
    core_species = {gen_set.species_id for gen_set in core}
    candidates = [gen_set for gen_set in flex_pool if gen_set.species_id not in core_species]
    if len(candidates) < flex_needed:
        return None
    rng = random.Random(seed)
    flex = rng.sample(candidates, flex_needed) if flex_needed else []
    return list(core) + flex


def resolve_team_items(team: list[GeneratedSet]) -> list[str | None]:
    """One concrete item per species in `team` (same order), respecting this format's
    Item Clause (max 1 of each item across the whole team). Walks each species' ranked
    `item_candidates` in order and takes the first one not already claimed by an earlier
    slot on this same team; a species whose every candidate is already taken goes
    itemless (`None` -- legal, if unusual) rather than force a duplicate.
    """
    used: set[str] = set()
    resolved: list[str | None] = []
    for gen_set in team:
        chosen: str | None = None
        for candidate in gen_set.item_candidates:
            if candidate not in used:
                chosen = candidate
                break
        if chosen is not None:
            used.add(chosen)
        resolved.append(chosen)
    return resolved


def _format_sp(sp: SPSpread) -> str:
    return ",".join(str(sp.get(stat, 0) or "") for stat in STAT_IDS)


def team_to_packed(team: list[GeneratedSet]) -> str:
    """Packed-team format string matching `teams/*.packed.txt`'s exact layout:
    `Species||Item|Ability|Move1,Move2,...|Nature|SP_HP,SP_Atk,SP_Def,SP_SpA,SP_SpD,
    SP_Spe||||50|`, `]`-joined across the team's 6 slots. Items are resolved via
    `resolve_team_items` first (a per-species `item_candidates` list is not itself
    packable -- Item Clause is a team-level constraint). Every field is written from the
    already-`to_id`-normalized identifiers this module works with throughout (species/
    item/ability/move ids) rather than display names -- Showdown's own team parser
    normalizes any input the same way internally (`toID()`), so the lowercase id form
    round-trips identically; `tools/build_selfplay_pool.py`'s real `validate-team` CLI
    call is what actually confirms this, not an assumption made here.
    """
    items = resolve_team_items(team)
    parts = []
    for gen_set, item in zip(team, items, strict=True):
        item_field = item or ""
        moves = ",".join(gen_set.moves)
        sp = _format_sp(gen_set.sp)
        parts.append(
            f"{gen_set.species_id}||{item_field}|{gen_set.ability}|{moves}|{gen_set.nature}|{sp}||||50|"
        )
    return "]".join(parts)
