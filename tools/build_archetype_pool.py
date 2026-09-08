#!/usr/bin/env python
"""Build an ARCHETYPE-TAGGED pool of legal, corpus-informed teams for archetype-
structured RL training -- a few archetypes (`vgc.archetypes.ARCHETYPES`), MANY validated
teams under each, so a later training step can test whether an agent learns the
archetype rather than memorizing specific teams.

For each archetype this writes:
  - one team per anchor (the real named team from `data/meta/popular_teams_
    H8v7TEZcbXo.json`, converted to a packed team with its stated item/ability/nature/
    moves/mega form preserved exactly -- only the Stat Point spread, which that source
    doesn't provide, is filled in from `data/usage/spreads.json` or
    `vgc.stats.default_opponent_spread`).
  - up to `--variants-per-archetype` additional teams that keep the archetype's defining
    core species fixed (using one of the anchors' own known-good sets for each core
    species, see `vgc.archetypes.core_set_options`) while sampling the remaining flex
    slots from the corpus-priority species pool (`vgc.selfplay_pool.
    build_priority_species`, the same one `tools/build_selfplay_pool.py` uses) and
    varying Stat Point spreads/natures/items across variants.

Every emitted team is validated against the real Showdown `validate-team` CLI (CLAUDE.md's
mandated legality check -- best-effort assembly is never itself a legality guarantee); a
failing variant is re-rolled with a new seed offset up to `--max-attempts` times before
being dropped, and a failing ANCHOR is reported loudly (anchors are real named teams --
silently dropping one would hide a real problem) with one minimal spread-only retry
before being skipped.

Output: `data/selfplay/archetype_pool/<archetype>/team_NN.packed.txt` (one packed team
per file, matching the existing pool's layout) plus `data/selfplay/archetype_pool/
manifest.json` (the training loop's consumption contract -- see `ManifestRecord` below).
Gitignored like `data/selfplay/pool/` -- rerun this tool to regenerate.

Usage:
    .venv/bin/python tools/build_archetype_pool.py
    .venv/bin/python tools/build_archetype_pool.py --variants-per-archetype 8 --seed 0 \
        --out data/selfplay/archetype_pool
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.archetypes import (  # noqa: E402
    ARCHETYPES,
    WEATHER_ABILITY_MAP,
    ArchetypeDef,
    active_weathers,
    anchors_for,
    core_set_options,
    weather_mega_stone_map,
)
from vgc.config import FORMAT_ID, SHOWDOWN_REPO  # noqa: E402
from vgc.data import load_items  # noqa: E402
from vgc.node import find_node, node_environment  # noqa: E402
from vgc.selfplay_pool import (  # noqa: E402
    ITEM_CANDIDATES_PER_SET,
    TEAM_SIZE,
    TOP_N_SPECIES,
    GeneratedSet,
    build_priority_species,
    resolve_team_items,
    sample_team_with_fixed_core,
    team_to_packed,
)
from vgc.sets import load_set_priors, load_usage_spreads  # noqa: E402
from vgc.stats import default_opponent_spread  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "selfplay" / "archetype_pool"
DEFAULT_VARIANTS_PER_ARCHETYPE = 10
DEFAULT_MAX_ATTEMPTS = 60
# How many alternative Stat-Point-only fixes to try on an anchor that fails validation
# before giving up on it -- deliberately small and NEVER touches item/ability/nature/
# moves (those are the anchor's stated, authoritative data): (1) the usage-derived
# spread per species (the normal path), (2) an all-default `default_opponent_spread`
# for every species (in case a usage spread is itself somehow the problem), (3) an
# all-zero spread (last resort -- always legal, just a bad spread).
ANCHOR_SPREAD_FIX_ATTEMPTS = 3

# Max distinct active weather effects (see `vgc.archetypes.active_weathers`) a VARIANT
# may end up with. 1 means "at most the archetype's own weather identity, nothing
# else" -- a weather=None archetype's core produces 0, so 1 still leaves room for a
# single stray setter to slip past the flex-exclusion filter below without silently
# corrupting the archetype into looking like a different one. Anchors are exempt (see
# `build_anchor_team`) -- e.g. the double_weather anchor legitimately runs both sun and
# rain, and that's real curated data, not a generation bug.
MAX_ACTIVE_WEATHERS_PER_VARIANT = 1


def validate_team(packed_team: str, node: str) -> str | None:
    """Runs the real `validate-team` CLI against `packed_team`. Returns `None` if legal,
    else the CLI's error output. Never raises: a subprocess failure (missing node/repo,
    timeout) is reported as a validation failure string rather than crashing the run.
    Identical to `tools/build_selfplay_pool.py`'s own `validate_team` -- kept as a
    separate copy (rather than a shared import) so each tool stays a self-contained,
    directly-runnable script, matching this repo's existing `tools/` convention.
    """
    try:
        result = subprocess.run(
            [node, "pokemon-showdown", "validate-team", FORMAT_ID],
            cwd=str(SHOWDOWN_REPO),
            input=packed_team,
            capture_output=True,
            text=True,
            env=node_environment(node),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"validate-team invocation failed: {exc}"
    output = (result.stdout + result.stderr).strip()
    return output or None


# --- Anchor team conversion ----------------------------------------------------------


def _spread_for_species(species_id: str, spreads: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    entries = spreads.get(species_id)
    if entries:
        best = max(entries, key=lambda entry: entry.get("weight", 0))
        return dict(best["sp"])
    return default_opponent_spread(species_id)


def anchor_to_generated_sets(
    anchor: dict[str, Any], spreads: dict[str, list[dict[str, Any]]]
) -> list[GeneratedSet]:
    """Convert one curated anchor team's `sets` into `GeneratedSet`s, preserving its
    stated item/ability/nature/moves/mega form EXACTLY (that source data is real and
    authoritative -- see `data/meta/popular_teams_H8v7TEZcbXo.json`'s own `"scope"`
    field) and filling only the Stat Point spread it doesn't provide, from
    `load_usage_spreads()` (falling back to `vgc.stats.default_opponent_spread` for a
    species with no usage entry).
    """
    generated: list[GeneratedSet] = []
    for entry in anchor["sets"]:
        species_id = entry["species_id"]
        generated.append(
            GeneratedSet(
                species_id=species_id,
                item_candidates=(entry["item_id"],),
                ability=entry["ability_id"],
                moves=tuple(entry["move_ids"]),
                nature=entry["nature_id"],
                sp=_spread_for_species(species_id, spreads),
            )
        )
    return generated


def _anchor_spread_fix(
    anchor: dict[str, Any], spreads: dict[str, list[dict[str, Any]]], fix_attempt: int
) -> list[GeneratedSet]:
    """Rebuild an anchor's `GeneratedSet`s using a progressively more conservative
    Stat-Point-only fallback -- see `ANCHOR_SPREAD_FIX_ATTEMPTS`'s docstring for the
    three strategies tried. `fix_attempt` is 1-indexed (1 = normal usage-derived path,
    already tried by the caller before this is ever invoked; this function is only
    called for attempt 2+).
    """
    generated: list[GeneratedSet] = []
    for entry in anchor["sets"]:
        species_id = entry["species_id"]
        if fix_attempt >= 3:
            sp = dict.fromkeys(("hp", "atk", "def", "spa", "spd", "spe"), 0)
        else:
            sp = default_opponent_spread(species_id)
        generated.append(
            GeneratedSet(
                species_id=species_id,
                item_candidates=(entry["item_id"],),
                ability=entry["ability_id"],
                moves=tuple(entry["move_ids"]),
                nature=entry["nature_id"],
                sp=sp,
            )
        )
    return generated


def build_anchor_team(
    anchor: dict[str, Any],
    spreads: dict[str, list[dict[str, Any]]],
    node: str,
) -> tuple[str | None, list[str]]:
    """Returns `(packed_team_or_None, notes)`. `notes` records what was tried; a loud
    `ERROR` note is included (and the anchor is skipped, i.e. returns `None`) only if
    every spread-only fix in `ANCHOR_SPREAD_FIX_ATTEMPTS` still fails validation --
    anchors are real named teams, so a failure here is never silently swallowed.
    """
    notes: list[str] = []
    for fix_attempt in range(1, ANCHOR_SPREAD_FIX_ATTEMPTS + 1):
        generated = (
            anchor_to_generated_sets(anchor, spreads)
            if fix_attempt == 1
            else _anchor_spread_fix(anchor, spreads, fix_attempt)
        )
        packed = team_to_packed(generated)
        error = validate_team(packed, node)
        if error is None:
            if fix_attempt > 1:
                notes.append(f"anchor {anchor['id']}: passed after spread fix #{fix_attempt}")
            return packed, notes
        notes.append(f"anchor {anchor['id']}: fix attempt {fix_attempt} failed: {error}")
    notes.append(
        f"ERROR: anchor {anchor['id']} ({anchor['name']}) failed validate-team after "
        f"{ANCHOR_SPREAD_FIX_ATTEMPTS} spread-only fix attempts -- skipping this anchor. "
        "Item/ability/nature/moves were never altered, so this points at a real problem "
        "with the anchor's stated data (see notes above for the CLI's error output)."
    )
    return None, notes


# --- Within-archetype variant generation ----------------------------------------------

_STAT_IDS = ("hp", "atk", "def", "spa", "spd", "spe")


def _sample_spread_and_nature(
    species_id: str,
    spreads: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    fallback_nature: str,
) -> tuple[dict[str, int], str]:
    """A Stat Point spread + nature for `species_id`, randomly drawn (weighted by
    corpus popularity) from `load_usage_spreads()`'s tracked entries when available, so
    each variant gets real spread diversity instead of always the single top spread.
    Falls back to `default_opponent_spread`/`fallback_nature` (the archetype's own
    known-good nature for this set) when the species has no tracked spread at all.
    """
    entries = spreads.get(species_id)
    if entries:
        weights = [max(entry.get("weight", 0) or 0, 0) for entry in entries]
        chosen = (
            rng.choices(entries, weights=weights, k=1)[0]
            if sum(weights) > 0
            else rng.choice(entries)
        )
        return dict(chosen["sp"]), chosen["nature"]
    return default_opponent_spread(species_id), fallback_nature


def _core_item_candidates(
    species_id: str,
    anchor_item: str,
    is_mega: bool,
    set_priors: dict[str, Any],
    legal_item_ids: set[str],
) -> tuple[str, ...]:
    """Item candidates for one core species in a variant. A Mega set's item IS its Mega
    Stone (`entry["battle_form_id"]` was set) -- swapping it would silently stop the
    Pokemon from Mega Evolving at all, defeating the archetype, so it's the only
    candidate. A non-Mega core species may vary its item across variants; the anchor's
    own stated item is kept as the top-ranked candidate (it's known-good), followed by
    the corpus's other ranked legal items for that species (same source
    `vgc.selfplay_pool._pick_items` uses) so `resolve_team_items`'s normal Item Clause
    resolution has real alternatives to fall back to.
    """
    if is_mega:
        return (anchor_item,)
    entry = (set_priors.get("species") or {}).get(species_id) or {}
    ranked = sorted((entry.get("items") or {}).items(), key=lambda kv: -kv[1])
    picked = [anchor_item] + [
        item_id for item_id, _count in ranked if item_id in legal_item_ids and item_id != anchor_item
    ]
    return tuple(picked[:ITEM_CANDIDATES_PER_SET])


def _filter_flex_pool_for_weather(
    flex_pool: list[GeneratedSet], archetype_weather: str | None, weather_stones: dict[str, str]
) -> list[GeneratedSet]:
    """Remove any flex candidate that would introduce an OFF-archetype active weather
    effect (see `vgc.archetypes.WEATHER_ABILITY_MAP`/`weather_mega_stone_map`) -- this
    is the actual fix for the bug where a flex Charizard's `charizarditey` mega-
    evolves it to Drought sun regardless of its own declared (non-weather) ability.

    Two mechanisms, handled differently because of what each one can and can't do
    about it:
      1. A GENERATED SET'S OWN ABILITY is a weather setter (e.g. a hypothetical flex
         Pelipper picked with Drizzle, or base Tyranitar's Sand Stream). Each pool
         entry has exactly one ability, so if it's an off-archetype weather setter the
         whole candidate is dropped -- there's no other ability to fall back to.
      2. A GENERATED SET'S ITEM CANDIDATE is a weather Mega Stone whose weather isn't
         the archetype's own. Only that specific item is stripped from the ranked
         `item_candidates` tuple -- the species itself stays usable as flex (e.g. flex
         Charizard without `charizarditey` is just a normal, non-weather Fire
         attacker, which is perfectly fine off-archetype).

    For a `weather=None` archetype (`archetype_weather` is `None`), every weather
    ability/stone counts as "off-archetype" -- that archetype has no weather identity
    to exempt, so no weather setter belongs in its flex pool at all.
    """
    filtered: list[GeneratedSet] = []
    for gen_set in flex_pool:
        set_weather = WEATHER_ABILITY_MAP.get(gen_set.ability)
        if set_weather is not None and set_weather != archetype_weather:
            continue  # mechanism 1: this species' own ability sets the wrong weather

        kept_items = tuple(
            item_id
            for item_id in gen_set.item_candidates
            if weather_stones.get(item_id, archetype_weather) == archetype_weather
        )
        if kept_items != gen_set.item_candidates:
            gen_set = replace(gen_set, item_candidates=kept_items)  # mechanism 2
        filtered.append(gen_set)
    return filtered


def build_core_generated_sets(
    archetype: ArchetypeDef,
    options: dict[str, list[dict[str, Any]]],
    spreads: dict[str, list[dict[str, Any]]],
    set_priors: dict[str, Any],
    legal_item_ids: set[str],
    rng: random.Random,
) -> list[GeneratedSet]:
    """The archetype's fixed-core `GeneratedSet`s for one variant: for each core
    species, randomly pick one of its known-good anchor sets (`options`, see
    `vgc.archetypes.core_set_options`) for item/ability/nature/moves, then layer a
    randomly-sampled spread/nature and (for non-Mega species) item variety on top.
    """
    core: list[GeneratedSet] = []
    for species_id in archetype.core_species:
        entry = rng.choice(options[species_id])
        is_mega = entry.get("battle_form_id") is not None
        sp, nature = _sample_spread_and_nature(species_id, spreads, rng, entry["nature_id"])
        item_candidates = _core_item_candidates(
            species_id, entry["item_id"], is_mega, set_priors, legal_item_ids
        )
        core.append(
            GeneratedSet(
                species_id=species_id,
                item_candidates=item_candidates,
                ability=entry["ability_id"],
                moves=tuple(entry["move_ids"]),
                nature=nature,
                sp=sp,
            )
        )
    return core


def _team_signature(team: list[GeneratedSet], items: list[str | None]) -> tuple[Any, ...]:
    """A team's dedup identity: the 6-species multiset plus the resolved item set. Two
    variants with the same species but different items (or vice versa) are NOT
    duplicates -- only an identical combination of both counts as one (see the module
    docstring / task spec: "no two emitted teams... may have identical 6-species
    multiset AND identical items").
    """
    species_multiset = tuple(sorted(gen_set.species_id for gen_set in team))
    item_set = tuple(sorted(item for item in items if item))
    return species_multiset, item_set


class ArchetypeResult(TypedDict):
    label: str
    anchor_packed: list[tuple[str, str]]  # (anchor_id, packed_team)
    anchor_failures: list[str]
    variant_packed: list[str]
    variant_failures: list[str]
    duplicates_skipped: int
    weather_conflicts_skipped: int


def build_archetype(
    archetype: ArchetypeDef,
    *,
    variants_requested: int,
    top_n_species: int,
    max_attempts: int,
    seed: int,
    set_priors: dict[str, Any],
    spreads: dict[str, list[dict[str, Any]]],
    flex_pool_all: list[GeneratedSet],
    legal_item_ids: set[str],
    weather_stones: dict[str, str],
    node: str,
) -> ArchetypeResult:
    result: ArchetypeResult = {
        "label": archetype.label,
        "anchor_packed": [],
        "anchor_failures": [],
        "variant_packed": [],
        "variant_failures": [],
        "duplicates_skipped": 0,
        "weather_conflicts_skipped": 0,
    }

    # --- Anchors: convert exactly, spread-fill, validate loudly ----------------------
    for anchor in anchors_for(archetype.label):
        packed, notes = build_anchor_team(anchor, spreads, node)
        for note in notes:
            print(f"  {note}")
        if packed is not None:
            result["anchor_packed"].append((anchor["id"], packed))
        else:
            result["anchor_failures"].append(anchor["id"])

    # --- Variants: fixed core + sampled flex, re-rolled on validation failure --------
    options = core_set_options(archetype)
    flex_pool = [
        gen_set for gen_set in flex_pool_all if gen_set.species_id not in archetype.core_species
    ]
    # Off-archetype weather safety (see module docstring / _filter_flex_pool_for_weather):
    # a flex slot must never introduce a DIFFERENT active weather than the archetype's
    # own (or ANY weather, for a weather=None archetype).
    flex_pool = _filter_flex_pool_for_weather(flex_pool, archetype.weather, weather_stones)
    flex_needed = TEAM_SIZE - len(archetype.core_species)
    if len(flex_pool) < flex_needed:
        result["variant_failures"].append(
            f"only {len(flex_pool)} flex-eligible species in the top-{top_n_species} pool "
            f"(excluding core), need {flex_needed} -- cannot build any variant"
        )
        return result

    seen_signatures: set[tuple[Any, ...]] = set()
    # Deterministic per-archetype seed space: offsetting by a large stride per archetype
    # index keeps every archetype's attempt sequence reproducible for a given --seed
    # while never overlapping another archetype's sequence (matters only for RNG-stream
    # hygiene, not correctness -- teams from different archetypes are independent).
    archetype_index = ARCHETYPES.index(archetype)
    seed_base = seed * 1_000_000 + archetype_index * 10_000

    attempt = 0
    while len(result["variant_packed"]) < variants_requested and attempt < max_attempts:
        variant_seed = seed_base + attempt
        rng = random.Random(variant_seed)
        core = build_core_generated_sets(archetype, options, spreads, set_priors, legal_item_ids, rng)
        team = sample_team_with_fixed_core(core, flex_pool, seed=variant_seed)
        attempt += 1
        if team is None:
            result["variant_failures"].append(f"attempt {attempt}: flex pool exhausted")
            continue

        items = resolve_team_items(team)

        # Post-generation guard (the backstop -- see MAX_ACTIVE_WEATHERS_PER_VARIANT):
        # holds even if `_filter_flex_pool_for_weather` somehow missed a path, since it
        # re-derives weather from the ACTUAL resolved (ability, item) pairs rather than
        # trusting the filter was exhaustive.
        pairs = [(gen_set.ability, item) for gen_set, item in zip(team, items, strict=True)]
        if len(active_weathers(pairs)) > MAX_ACTIVE_WEATHERS_PER_VARIANT:
            result["weather_conflicts_skipped"] += 1
            continue

        signature = _team_signature(team, items)
        if signature in seen_signatures:
            result["duplicates_skipped"] += 1
            continue

        packed = team_to_packed(team)
        error = validate_team(packed, node)
        if error is None:
            seen_signatures.add(signature)
            result["variant_packed"].append(packed)
        else:
            result["variant_failures"].append(f"attempt {attempt}: {error}")

    return result


def build_pool(
    *,
    variants_per_archetype: int,
    top_n_species: int,
    max_attempts: int,
    seed: int,
    node: str,
) -> list[ArchetypeResult]:
    set_priors = load_set_priors()
    spreads = load_usage_spreads()
    flex_pool_all = build_priority_species(set_priors, spreads, top_n=top_n_species)
    # Same legality source `vgc.selfplay_pool._pick_items` uses -- the full Champions-
    # mod legal item list (`data/champions/items.json`), not just corpus-observed ones,
    # so a rare-but-legal item a species' corpus prior happens to include is trusted the
    # same way `build_priority_species` already trusts it for flex species.
    legal_item_ids = set(load_items().keys())
    # Computed once from data (see `weather_mega_stone_map`'s docstring), not per
    # archetype -- it doesn't depend on which archetype is being built.
    weather_stones = weather_mega_stone_map()

    results: list[ArchetypeResult] = []
    for archetype in ARCHETYPES:
        print(f"building archetype: {archetype.label}")
        results.append(
            build_archetype(
                archetype,
                variants_requested=variants_per_archetype,
                top_n_species=top_n_species,
                max_attempts=max_attempts,
                seed=seed,
                set_priors=set_priors,
                spreads=spreads,
                flex_pool_all=flex_pool_all,
                legal_item_ids=legal_item_ids,
                weather_stones=weather_stones,
                node=node,
            )
        )
    return results


def write_pool(results: list[ArchetypeResult], out_dir: Path) -> list[dict[str, Any]]:
    """Writes every validated team to `<out_dir>/<archetype>/team_NN.packed.txt` and
    returns the manifest records (`{"file", "archetype", "source", "species"}`) --
    `main` writes those to `manifest.json`. Clears each archetype's directory first so
    stale teams from a previous run (e.g. a different `--variants-per-archetype`) never
    linger alongside the new ones.
    """
    manifest: list[dict[str, Any]] = []
    for result in results:
        archetype_dir = out_dir / result["label"]
        archetype_dir.mkdir(parents=True, exist_ok=True)
        for stale in archetype_dir.glob("team_*.packed.txt"):
            stale.unlink()

        index = 0
        for anchor_id, packed in result["anchor_packed"]:
            file_path = archetype_dir / f"team_{index:02d}.packed.txt"
            file_path.write_text(packed + "\n")
            manifest.append(
                {
                    "file": str(file_path.relative_to(out_dir)),
                    "archetype": result["label"],
                    "source": "anchor",
                    "species": [slot.split("||", 1)[0] for slot in packed.split("]")],
                }
            )
            index += 1
        for packed in result["variant_packed"]:
            file_path = archetype_dir / f"team_{index:02d}.packed.txt"
            file_path.write_text(packed + "\n")
            manifest.append(
                {
                    "file": str(file_path.relative_to(out_dir)),
                    "archetype": result["label"],
                    "source": "variant",
                    "species": [slot.split("||", 1)[0] for slot in packed.split("]")],
                }
            )
            index += 1
    return manifest


def print_summary(results: list[ArchetypeResult]) -> None:
    header = (
        f"{'archetype':<28}{'anchors':>12}{'variants':>14}{'failures':>10}"
        f"{'dupes':>8}{'weather':>9}"
    )
    print(header)
    print("-" * len(header))
    total_teams = 0
    total_failures = 0
    for result in results:
        anchors_ok = len(result["anchor_packed"])
        anchors_total = anchors_ok + len(result["anchor_failures"])
        variants_ok = len(result["variant_packed"])
        n_failures = len(result["variant_failures"])
        print(
            f"{result['label']:<28}{f'{anchors_ok}/{anchors_total}':>12}"
            f"{variants_ok:>14}{n_failures:>10}{result['duplicates_skipped']:>8}"
            f"{result['weather_conflicts_skipped']:>9}"
        )
        total_teams += anchors_ok + variants_ok
        total_failures += n_failures + len(result["anchor_failures"])
    print("-" * len(header))
    print(f"total teams written: {total_teams}")
    print(f"total failures (anchor + variant): {total_failures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants-per-archetype", type=int, default=DEFAULT_VARIANTS_PER_ARCHETYPE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n-species", type=int, default=TOP_N_SPECIES)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    node = find_node()
    results = build_pool(
        variants_per_archetype=args.variants_per_archetype,
        top_n_species=args.top_n_species,
        max_attempts=args.max_attempts,
        seed=args.seed,
        node=node,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = write_pool(results, args.out)
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print()
    print_summary(results)
    print(f"manifest: {manifest_path}")
    print(f"output: {args.out}")

    any_anchor_failed = any(result["anchor_failures"] for result in results)
    any_archetype_short = any(
        len(result["variant_packed"]) < args.variants_per_archetype for result in results
    )
    if any_anchor_failed or any_archetype_short:
        print(
            "WARNING: at least one anchor failed validation or one archetype fell short "
            "of --variants-per-archetype -- see the notes/summary above",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
