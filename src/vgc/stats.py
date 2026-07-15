"""Level-50 stat calculation under the Champions mod's **Stat Points** system.

The champions mod (backing `gen9championsvgc2026regmb`, see `vgc.config.FORMAT_ID`)
replaces vanilla gen9's 0-252-per-stat / 510-total EV system with **Stat Points**: max 32
per stat, 66 total across all six stats (see CLAUDE.md). This is NOT the same number
space as EVs -- a Stat Point is not "1/4 of an EV", it is a direct +1 to the post-base
stat, derived below.

## Formula derivation (read this before touching the constants)

Source: the local Showdown checkout's `sim/team-validator.ts` and
`data/mods/champions/scripts.ts`, both confirmed against `config/formats.ts` for the
specific rules this format applies, and empirically verified against the built sim (see
"Ground truth" below).

1. `sim/team-validator.ts::validateStats` (`useStatPoints = dex.currentMod.startsWith
   ('champions')`, ~line 1144):
   - IVs must all be exactly 31 ("this format requires all IVs to be 31" problem string)
     when `useStatPoints` is set and any IV isn't 31 -- i.e. every legal Champions set has
     **fixed IVs of 31** in every stat. There is no way to legally field a non-31 IV in
     this mod.
   - Each `set.evs[stat]` (the field is still called `evs` in the PokemonSet type, but
     is documented in `sim/teams.ts` as "Stat Points for Champions") is capped at
     `> 32` -> illegal. That's `MAX_SP_PER_STAT` below.
   - `sim/dex-formats.ts` (`valueRules.get('evlimit') === 'Auto'` branch, ~line 342):
     `if (format.mod.startsWith('champions')) this.evLimit = 66;` -- the 66-total cap.
     (`gen9championsvgc2026regmb`'s ruleset doesn't override `evlimit`, so this Auto
     value applies.)

2. `data/mods/champions/scripts.ts::statModify` -- the actual stat formula. It has two
   branches gated on the `levelclausemod` ruleset (which recomputes stats from `set.level`
   instead of assuming level 50). `gen9championsvgc2026regmb`'s ruleset is
   `['Flat Rules', 'VGC Timer', 'Open Team Sheets']`; `Flat Rules` (`rulesets.ts`) does
   NOT include `Level Clause Mod`, and this format has no adjustable level (`Adjust Level
   = 50` in Flat Rules forces every Pokemon to level 50 for battle purposes). So the
   *non*-`levelclausemod` branch is the one that applies here, and it is level-independent
   (level 50 is baked into the constants, there is no `level` term at all):

   ```js
   // statModify(baseStats, set, statName), useStatPoints implied by evs already being SP:
   if (statName === 'hp') {
       return stat + evs + 75;                    // evs here means the Stat Point value
   }
   stat = stat + evs + 20;
   // then nature, 16-bit-truncated:
   if (nature.plus === statName)  stat = trunc(trunc(stat * 110, 16) / 100);
   if (nature.minus === statName) stat = trunc(trunc(stat * 90, 16) / 100);
   ```

   i.e. **every Stat Point is worth exactly +1 to the raw (pre-nature) stat** -- there's no
   `floor(EV/4)` step like vanilla EVs. This is confirmed consistent with the standard
   level-50 EV formula's *endpoints*: at SP=0 the standard formula
   (`floor((2*base+31+floor(EV/4))*50/100)+5`, IV=31, EV=0) reduces to `base+20`, matching
   `base+0+20`; at SP=32 (~= 252 EVs, `floor(252/4)=63`) it reduces to `base+52`, matching
   `base+32+20`. HP works out the same way (`base+75` at SP=0, `base+107` at SP=32). The
   Champions system is a deliberately simplified *linear* replacement for the same
   level-50 stat range vanilla EVs cover, at a much smaller (66 vs 510) budget.

3. Nature multipliers use the same `natures.json` `plus`/`minus` stat ids as vanilla
   (`atk`/`def`/`spa`/`spd`/`spe`; HP is never affected by nature). The 16-bit truncation
   in `statModify` only matters for absurd (Pure Hackmons-scale) stat values and is a
   no-op for any legal Champions stat, so plain integer floor division is used here.

## Ground truth

Verified against the built sim directly: `tools/sim_probe.mjs` drives a
`[Gen 9 Champions] Custom Game` battle (which has `debug: true`, enabling `>eval`) with a
Garchomp at a known SP spread/nature and dumps `pokemon.maxhp` / `pokemon.storedStats` via
`>eval`; see `tests/test_stats_ground_truth.py` (integration) for the automated version of
this check. `offline`/dev runs of this during Phase 2a confirmed exact agreement with the
formula above for several spreads (0 SP, 32-in-one-stat, and a full 66-point spread).
"""

from __future__ import annotations

from typing import Literal

from vgc.data import load_natures, load_species

StatId = Literal["hp", "atk", "def", "spa", "spd", "spe"]
STAT_IDS: tuple[StatId, ...] = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTABLE_STAT_IDS: tuple[StatId, ...] = ("atk", "def", "spa", "spd", "spe")

# Champions Stat Points system (see module docstring for derivation/sources).
MAX_SP_PER_STAT = 32
MAX_SP_TOTAL = 66
# `useStatPoints` requires all IVs == 31 (sim/team-validator.ts); there is no legal way
# to field anything else in this mod, so this is a fixed assumption, not a parameter.
FIXED_IV = 31
# The format is hardcoded to level 50 ('Adjust Level = 50' in Flat Rules); the formula
# below has level baked into its constants (see module docstring) and doesn't generalize.
FORMAT_LEVEL = 50

SPSpread = dict[str, int]


def empty_spread() -> SPSpread:
    """An all-zero Stat Point spread (every stat at its SP=0 floor)."""
    return dict.fromkeys(STAT_IDS, 0)


def validate_spread(spread: SPSpread) -> None:
    """Raise ValueError if `spread` isn't a legal Champions Stat Point allocation.

    Mirrors `sim/team-validator.ts::validateStats`'s Stat Point checks: 0-32 per stat,
    <=66 total. Missing stat keys are treated as 0 (same as the validator's
    `TeamValidator.fillStats(null, 0)` default for a format with an evLimit).
    """
    total = 0
    for stat in STAT_IDS:
        value = spread.get(stat, 0)
        if value < 0 or value > MAX_SP_PER_STAT:
            raise ValueError(
                f"{stat} Stat Points must be between 0 and {MAX_SP_PER_STAT}, got {value}"
            )
        total += value
    if total > MAX_SP_TOTAL:
        raise ValueError(f"total Stat Points must be <= {MAX_SP_TOTAL}, got {total}")
    unknown = set(spread) - set(STAT_IDS)
    if unknown:
        raise ValueError(f"unknown stat id(s) in spread: {sorted(unknown)}")


def calculate_stats(
    species_id: str,
    sp: SPSpread | None = None,
    nature: str = "serious",
    level: int = FORMAT_LEVEL,
) -> dict[StatId, int]:
    """Final level-50 stats for `species_id` under the given Stat Point spread/nature.

    :param species_id: key into `vgc.data.load_species()` -- for mega formes, pass the
        mega's own id (e.g. `"garchompmega"`), not the base forme; base stats, types, and
        ability all come from the mega's own species entry (see CLAUDE.md's mega note).
    :param sp: Stat Point spread, e.g. `{"hp": 32, "atk": 32, "spe": 2}`. Missing stats
        default to 0. `None` means an all-zero spread (SP=0 everywhere).
    :param nature: nature id (`vgc.data.load_natures()` key), e.g. `"adamant"`.
    :param level: must be 50 -- this format has no adjustable level (see module
        docstring); anything else raises since the formula doesn't generalize to it.
    """
    if level != FORMAT_LEVEL:
        raise ValueError(
            f"calculate_stats only implements the level-{FORMAT_LEVEL} Champions formula "
            f"(gen9championsvgc2026regmb has no adjustable level); got level={level}"
        )
    species = load_species().get(species_id)
    if species is None:
        raise KeyError(f"unknown species id {species_id!r} (check vgc.data.load_species())")
    spread = sp if sp is not None else empty_spread()
    validate_spread(spread)
    nature_data = load_natures().get(nature)
    if nature_data is None:
        raise KeyError(f"unknown nature id {nature!r} (check vgc.data.load_natures())")

    base_stats = species["baseStats"]
    stats: dict[StatId, int] = {}
    for stat in STAT_IDS:
        base = base_stats[stat]
        points = spread.get(stat, 0)
        if stat == "hp":
            stats[stat] = base + points + 75
            continue
        raw = base + points + 20
        if nature_data["plus"] == stat:
            raw = (raw * 110) // 100
        elif nature_data["minus"] == stat:
            raw = (raw * 90) // 100
        stats[stat] = raw
    return stats


def default_opponent_spread(species_id: str) -> SPSpread:
    """A reasonable default Stat Point spread for an opponent whose real spread is
    hidden (Open Team Sheets reveal species/item/ability/moves but not Stat Points).

    Heuristic: max SP (32) into HP and whichever of Atk/SpA is the higher base stat (a
    reasonable stand-in for "this Pokemon attacks with its better offensive stat"), and
    the 2 leftover SP (66 - 32 - 32) into Speed, which is the next-most-common
    tiebreaker investment in real VGC spreads. This is intentionally simple -- real
    per-species usage data belongs in `vgc.sets.load_usage_spreads`, which should be
    preferred over this fallback when available.
    """
    species = load_species().get(species_id)
    if species is None:
        raise KeyError(f"unknown species id {species_id!r} (check vgc.data.load_species())")
    base_stats = species["baseStats"]
    offensive_stat: StatId = "atk" if base_stats["atk"] >= base_stats["spa"] else "spa"

    spread = empty_spread()
    spread["hp"] = MAX_SP_PER_STAT
    spread[offensive_stat] = MAX_SP_PER_STAT
    spread["spe"] += MAX_SP_TOTAL - 2 * MAX_SP_PER_STAT
    return spread


def default_opponent_nature(species_id: str) -> str:
    """A nature id matching `default_opponent_spread`'s offense choice: Adamant
    (+Atk/-SpA) if the higher offensive stat is Atk, Modest (+SpA/-Atk) if it's SpA.
    """
    species = load_species().get(species_id)
    if species is None:
        raise KeyError(f"unknown species id {species_id!r} (check vgc.data.load_species())")
    base_stats = species["baseStats"]
    return "adamant" if base_stats["atk"] >= base_stats["spa"] else "modest"
