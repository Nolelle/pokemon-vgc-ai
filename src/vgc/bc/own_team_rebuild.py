"""Rebuild the acting player's own-team item/ability/moves in replay decision records.

Live (`vgc.bc.policy.battle_state_record` -> `vgc.bc.encoding.encode_state`): our own
active Pokemon's item, ability, and all 4 moves are exactly known. In replay records
(schema 8) the acting player's own side only carries what was PUBLIC so far (~18%
items known), because replay logs are spectator views. This module fills the acting
player's own ACTIVE-slot `item`, `ability`, and `revealed_moves` -- the only own fields
`encode_state` reads that are affected (bench entries carry no item/ability/moves; stat
points do not exist in records and are not read by the BC trunk). The opponent side is
never touched.

## Sources, in priority order

Layer 1 -- same game, raw protocol facts the player knew from the start:

- Item: initial held item, evidenced by `-mega` (the mega stone), `-enditem`
  (consumed/popped/knocked off), `-item` without a transfer `[from]`, `[from] item: X`
  effects attributed to that Pokemon (`[of]` wins over the position token, so a Rocky
  Helmet's damage line credits the holder), and Frisk reveals. If ANY transfer-in for
  that Pokemon (Trick, Switcheroo, Thief, Covet, Bestow, Symbiosis, Pickpocket,
  Magician, Pickup, Harvest, Recycle) precedes the first such evidence -> no item fact.
  Fill window: that Pokemon's snapshots (record `log_line`) before its first
  item-change event (removal or transfer-in); after that the parser's value stands
  (matches live: a consumed item reads `None` -> `<unk>`).
- Ability: base-forme ability from `[from] ability: X` attributed to that Pokemon or
  plain `-ability` reveals, excluding copies/changes (a `-ability` line whose `[from]`
  is an ability credits the `[from]` ability -- Trace/Receiver/Mummy-style -- never the
  copied value; `[from] move:` change sources credit nothing). Data facts: a species
  with exactly one legal ability in data/champions/species.json -> fact; post-mega
  snapshots get the mega forme's ability (matches live: poke-env's `pokemon.ability`
  returns `forme_change_ability` -- the mega dex entry's `abilities["0"]` -- once
  `mega_evolve` runs, see tools' notes).
- Moves: distinct moves that Pokemon used via `|move|` lines WITHOUT a calling
  `[from]` (excludes Metronome/Copycat/Sleep Talk/Assist/Me First/Mirror Move called
  moves, Dancer copies, Magic Bounce/Magic Coat reflections; `[from]move: Instruct` IS
  the mon's own move being repeated, so it counts; `struggle` is excluded). More than
  4 distinct -> anomaly: no move fill for that Pokemon in that game (counted). Fill
  window: whole game -- the player always knew all 4.

Layer 2 -- same player, same team in other games. Group key = (normalized account,
sorted six-species preview lineup after `canonical_species_id`). Donor facts = each
donor game's Layer-1 start-of-game facts, plus team sheets (`|showteam|` lines).
Donor eligibility by time (prevents later-period information leaking into training
inputs): boundaries DEV_START / TEST_START come from the v3 selection manifest
(earliest validation/test game start times). A train target may only use donors
starting before DEV_START; validation before TEST_START; test may use any donor.
Set-aside games (excluded/reserved) are allowed as donors under the same rule. The
target game is never its own donor. Acceptance: (1) prefer the nearest-in-time donor
SHEET whose item/ability/moves are consistent with the target's Layer-1 facts; take
its values for unknown fields. (2) Otherwise per-field: union of non-sheet donor
facts iff all donors agree (single item / single ability / move union <=4) and are
consistent with the target's facts (union with target moves <=4).

Layer 3 -- common sets, priors computed ONLY from games starting before DEV_START
(Layer-1 facts + sheets, per base species). Item = most common item not already
held/filled by a teammate (the format's Flat Rules include Item Clause -- confirmed
against the pinned Showdown's rule table, `itemclause`). Ability = most common among
the species' legal abilities. Moves = greedy fill of remaining slots: candidate m
(must be in data/champions/learnsets.json, not already present) scores the number of
complete 4-move prior sets containing m AND all currently known moves; if no complete
set contains all known moves, back off to m's marginal prior frequency; ties by move
id. Species with no prior facts -> leave unknown.

## Bookkeeping

`own_info` (per record, NOT a model input): per own active slot `item_source` /
`ability_source` in {observed, same_game, data, same_player_sheet, same_player_games,
common_set, unknown} plus move counts by source. Record-level `own_fully_confirmed`
is true iff no own active field resolved to common_set/unknown. `encode_state` never
reads `own_info` -- a test proves its output is invariant to it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from vgc.damage import to_id
from vgc.data import canonical_species_id, load_learnsets, load_species
from vgc.replay_parse import (
    _parse_position,
    _resolve_species,
    _species_from_details,
    parse_showteam_line,
)

# --- source labels (the exact vocabulary of `own_info`) ----------------------------------

OBSERVED = "observed"
SAME_GAME = "same_game"
DATA = "data"
SAME_PLAYER_SHEET = "same_player_sheet"
SAME_PLAYER_GAMES = "same_player_games"
COMMON_SET = "common_set"
UNKNOWN = "unknown"
SOURCES = (
    OBSERVED,
    SAME_GAME,
    DATA,
    SAME_PLAYER_SHEET,
    SAME_PLAYER_GAMES,
    COMMON_SET,
    UNKNOWN,
)
# Anything better than a guess/absence -- `own_fully_confirmed` counts a field as
# confirmed exactly when its source is not in this set.
_UNCONFIRMED = (COMMON_SET, UNKNOWN)

# Transfer-in `[from]` sources: a `-item` line citing one of these means the item
# ARRIVED mid-game, so it says nothing about the initial held item (Layer-1 spec).
_TRANSFER_MOVES = {"trick", "switcheroo", "thief", "covet", "bestow", "recycle"}
_TRANSFER_ABILITIES = {"symbiosis", "pickpocket", "magician", "pickup", "harvest"}

# `|move|` lines citing a calling `[from]` are not the mon's own moves; the one
# exception (Layer-1 spec) is an Instruct repeat, which IS its own move.
_CALLING_MOVE_EXEMPTION = "instruct"


def _parts(line: str) -> list[str]:
    return line.split("|")


def _tagged(parts: list[str], prefix: str) -> list[str]:
    """Values of `[prefix]...` extra parts, e.g. `[from] move: Trick` -> 'move: Trick'."""
    return [part[len(prefix) :].strip() for part in parts if part.startswith(prefix)]


def _from_kinds(parts: list[str]) -> list[tuple[str, str]]:
    """`(kind, id)` pairs from `[from]` extras: `[from] move: Trick` -> ('move','trick'),
    `[from] ability: X` -> ('ability','x'), `[from] stealeat` -> ('stealeat','')."""
    out = []
    for value in _tagged(parts, "[from]"):
        if ":" in value:
            kind, name = value.split(":", 1)
            out.append((kind.strip(), to_id(name)))
        else:
            out.append(("", to_id(value)))
    return out


def _of_position(parts: list[str]) -> tuple[str, int | None] | None:
    """`(side, slot)` an `[of] p2a: Name` extra points at, or None."""
    for value in _tagged(parts, "[of]"):
        side, slot, _name = _parse_position(value)
        if side in ("p1", "p2") and slot is not None:
            return side, slot
    return None


# --- Layer 1: per-game fact extraction -----------------------------------------------------


@dataclass
class MonFacts:
    """Layer-1 start-of-game facts for one (side, species) in one game."""

    species: str
    item: str | None = None  # first-evidence item (None: none seen or transfer-blocked)
    item_evidence_count: int = 0
    item_conflict: bool = False  # disagreeing evidence (first wins, flagged)
    item_blocked: bool = False  # a transfer-in preceded the first evidence -> no fact
    first_item_change: int | None = None  # raw idx of first removal/transfer-in (fill window end)
    ability: str | None = None  # first-evidence base ability
    ability_conflict: bool = False
    ability_from_data: bool = False  # filled by _finish() from single-legal-ability species
    first_seen: int | None = None  # raw idx of the mon's first switch/drag/replace in
    mega: bool = False
    mega_ability: str | None = None
    moves: list[str] = field(default_factory=list)  # distinct, first-use order
    move_anomaly: bool = False  # >4 distinct -> no move fill

    def note_item_evidence(self, idx: int, item: str) -> None:
        if not item:
            return
        self.item_evidence_count += 1
        if self.item is None:
            self.item = item
        elif self.item != item:
            self.item_conflict = True

    def note_item_change(self, idx: int) -> None:
        if self.first_item_change is None or idx < self.first_item_change:
            self.first_item_change = idx

    def note_ability_evidence(self, idx: int, ability: str) -> None:
        del idx  # ability facts are start-of-game facts; order only matters for conflicts
        if not ability:
            return
        if self.ability is None:
            self.ability = ability
        elif self.ability != ability:
            self.ability_conflict = True

    def note_move(self, move_id: str) -> None:
        if move_id and move_id not in self.moves:
            self.moves.append(move_id)
            if len(self.moves) > 4:
                self.move_anomaly = True


@dataclass
class GameFacts:
    replay_id: str
    players: dict[str, str] = field(default_factory=dict)  # p1/p2 -> display name
    preview: dict[str, list[str]] = field(default_factory=lambda: {"p1": [], "p2": []})
    mons: dict[tuple[str, str], MonFacts] = field(default_factory=dict)  # (side, base species)
    sheets: dict[str, dict[str, dict]] = field(default_factory=dict)  # side -> species -> set
    fainted: dict[str, set[str]] = field(
        default_factory=lambda: {"p1": set(), "p2": set()}
    )  # side -> base species that fainted (bench completion must not add them)


def _mega_ability_for(base_species: str, stone_id: str | None) -> str | None:
    """The mega forme's ability for `base_species` (item pins the right mega for
    two-mega species like Charizard; without a stone take the unique mega if there is
    exactly one). Mirrors what live poke-env reports post-mega (forme_change_ability).
    """
    species = load_species()

    def parent(entry: dict) -> str:
        origin = entry.get("battleOnly") or entry.get("baseSpecies")
        if isinstance(origin, list):
            origin = origin[0] if origin else None
        return canonical_species_id(to_id(origin))

    candidates = [
        entry
        for entry in species.values()
        if entry.get("isMega") and parent(entry) == base_species
    ]
    if stone_id:
        for entry in candidates:
            if to_id(entry.get("requiredItem")) == stone_id:
                abilities = (entry.get("abilities") or {}).values()
                return to_id(next(iter(abilities))) if abilities else None
        return None
    if len(candidates) == 1:
        abilities = (candidates[0].get("abilities") or {}).values()
        return to_id(next(iter(abilities))) if abilities else None
    return None


def _single_legal_ability(species_id: str) -> str | None:
    entry = load_species().get(species_id)
    if not entry:
        return None
    abilities = [to_id(name) for name in (entry.get("abilities") or {}).values() if name]
    return abilities[0] if len(abilities) == 1 else None


def analyze_log(replay_id: str, log: str) -> GameFacts:
    """Layer-1 facts + team sheets from one raw replay log. Species keyed by canonical
    base id; position->species follows the same switch/drag/replace/faint tracking as
    `vgc.replay_parse` (so `[of]` attributions land on the right team slot).
    """
    facts = GameFacts(replay_id=replay_id)
    slots: dict[str, list[str | None]] = {"p1": [None, None], "p2": [None, None]}

    def mon_at(side: str, slot: int | None) -> MonFacts | None:
        if side not in ("p1", "p2") or slot is None:
            return None
        species_id = slots[side][slot]
        if species_id is None:
            return None
        key = (side, species_id)
        if key not in facts.mons:
            facts.mons[key] = MonFacts(species_id)
        return facts.mons[key]

    for idx, line in enumerate(log.splitlines()):
        if not line or line == "|":
            continue
        parts = _parts(line)
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == "player":
            if len(parts) >= 4 and parts[2] in ("p1", "p2") and parts[3]:
                facts.players[parts[2]] = parts[3]
            continue

        if tag == "poke":
            if len(parts) >= 4 and parts[2] in ("p1", "p2"):
                species_id, _ = _resolve_species(_species_from_details(parts[3]))
                if species_id and species_id not in facts.preview[parts[2]]:
                    facts.preview[parts[2]].append(species_id)
            continue

        if tag == "showteam":
            result = parse_showteam_line(parts)
            if result is not None:
                side, sets = result
                facts.sheets.setdefault(side, {}).update(sets)
            continue

        if tag in ("switch", "drag", "replace"):
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id, is_mega = _resolve_species(_species_from_details(parts[3]))
            slots[side][slot] = species_id
            mon = mon_at(side, slot)
            if mon is not None:
                if mon.first_seen is None:
                    mon.first_seen = idx
                if is_mega:
                    # Switched back in post-mega: Showdown shows the MEGA forme
                    # name in details, so snapshots here are post-mega too.
                    mon.mega = True
                    mon.mega_ability = (
                        _mega_ability_for(mon.species, None) or mon.mega_ability
                    )
            continue

        if tag == "faint":
            side, slot, _name = _parse_position(parts[2])
            if side in ("p1", "p2") and slot is not None:
                species = slots[side][slot]
                if species:
                    facts.fainted[side].add(species)
                slots[side][slot] = None
            continue

        # `|swap|p1a: 1`: Ally Switch swaps the two ACTIVE slots -- the
        # position->species map must follow the occupants or later facts land on
        # the wrong species (replay_parse applies the same |swap| swap to its
        # `active` map so parser and facts agree).
        if tag == "swap" and len(parts) > 2:
            side, _slot, _name = _parse_position(parts[2])
            if side in ("p1", "p2"):
                slots[side][0], slots[side][1] = slots[side][1], slots[side][0]
            continue

        if tag == "move":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            froms = _from_kinds(parts[5:])
            if froms and not any(
                kind == "move" and name == _CALLING_MOVE_EXEMPTION for kind, name in froms
            ):
                continue  # called/reflected/copied move -- not the mon's own
            mon = mon_at(side, slot)
            move_id = to_id(parts[3])
            if mon is not None and move_id != "struggle":
                mon.note_move(move_id)
            continue

        if tag == "-mega":
            if len(parts) < 5:
                continue
            side, slot, _name = _parse_position(parts[2])
            mon = mon_at(side, slot)
            if mon is None:
                continue
            stone = to_id(parts[4])
            mon.note_item_evidence(idx, stone)
            mon.mega = True
            mon.mega_ability = _mega_ability_for(mon.species, stone) or mon.mega_ability
            continue

        # `|-activate|USER|move: trick/switcheroo/bestow/...`: the USER is about to
        # receive an item via the swap -- an incoming transfer that arrives before
        # any `-item`/`[from]` line (a received White Herb can even `-enditem`
        # before its `-item` reveal logs). Blocking here prevents crediting the
        # received item as the initial one.
        if tag == "-activate" and len(parts) > 3 and parts[3].startswith("move: "):
            if to_id(parts[3][len("move: ") :]) in _TRANSFER_MOVES:
                side, slot, _name = _parse_position(parts[2])
                mon = mon_at(side, slot)
                if mon is not None and not mon.item_evidence_count:
                    mon.item_blocked = True
            continue

        if tag == "-item":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            mon = mon_at(side, slot)
            if mon is None:
                continue
            froms = _from_kinds(parts[4:])
            is_transfer = any(
                (kind == "move" and name in _TRANSFER_MOVES)
                or (kind == "ability" and name in _TRANSFER_ABILITIES)
                or (kind == "" and name in _TRANSFER_MOVES | _TRANSFER_ABILITIES)
                for kind, name in froms
            )
            if is_transfer:
                if not mon.item_evidence_count:
                    mon.item_blocked = True
                mon.note_item_change(idx)
            else:
                mon.note_item_evidence(idx, to_id(parts[3]))
            continue

        if tag == "-enditem":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            mon = mon_at(side, slot)
            if mon is None:
                continue
            # parts[3] is the item that ended -- it was held until now. `[of]` marks
            # the thief/eater (Bug Bite/Pluck log `-enditem` ON the victim), never the
            # owner, so the position mon is always the right attribution here.
            mon.note_item_evidence(idx, to_id(parts[3]))
            mon.note_item_change(idx)
            continue

        if tag == "-ability":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            position_mon = mon_at(side, slot)
            ability_id = to_id(parts[3])
            all_froms = _from_kinds(parts[4:])
            ability_froms = [name for kind, name in all_froms if kind == "ability" and name]
            if all_froms and not ability_froms:
                # `[from] move: Entrainment/Skill Swap/...`, `[from] item:`, etc.:
                # an ability CHANGE, not a base reveal -- never a fact. (Do not
                # recover the old ability from the spare field either.)
                continue
            if not all_froms:
                if position_mon is None:
                    continue
                if position_mon.mega:
                    # Post-mega reveals are the MEGA forme's ability (Drought on
                    # Mega Charizard Y, Fairy Aura on Mega Floette, ...), never the
                    # base's -- crediting them as base was the dominant L1 miss.
                    if position_mon.mega_ability is None:
                        position_mon.mega_ability = ability_id
                    continue
                position_mon.note_ability_evidence(idx, ability_id)
                continue
            for source_ability in ability_froms:
                if ability_id == source_ability:
                    # `|-ability|A|X|[from] ability: X|[of] B`: the contact-spread
                    # ability (Mummy/Wandering Spirit/Lingering Aroma) -- the fact
                    # belongs to the `[of]` mon, not the receiver.
                    of_pos = _of_position(parts[4:])
                    target = mon_at(*of_pos) if of_pos else position_mon
                    if target is not None and not target.mega:
                        target.note_ability_evidence(idx, source_ability)
                else:
                    # `|-ability|A|<copied>|[from] ability: Trace` (or Receiver /
                    # Power of Alchemy): the `[from]` IS the position mon's own
                    # ability; the copied value is never a base fact.
                    if position_mon is not None and not position_mon.mega:
                        position_mon.note_ability_evidence(idx, source_ability)
            continue

        # Generic `[from] item: X` / `[from] ability: X` evidence on every other line:
        # the effect's owner is the `[of]` mon when present (Rough Skin/Rocky Helmet
        # chip lines credit the holder, not the damaged attacker), else the position
        # mon. `-ability`/`-item`/`-enditem` are handled above.
        if len(parts) < 3:
            continue
        side, slot, _name = _parse_position(parts[2])
        of_pos = _of_position(parts[3:])
        target_pos = of_pos or (side, slot)
        target = mon_at(*target_pos)
        if target is None:
            continue
        for kind, name in _from_kinds(parts[3:]):
            if kind == "item" and name:
                target.note_item_evidence(idx, name)
            elif kind == "ability" and name and not target.mega:
                target.note_ability_evidence(idx, name)
        # `|cant|HOLDER|ability: X|MOVE|[of] BLOCKED`: the POSITION mon owns X
        # (Showdown: `add('cant', armorTailHolder, 'ability: Armor Tail', move,
        # '[of] ' + target)`); `[of]` is the blocked attacker, NOT the owner.
        if tag == "cant" and len(parts) > 3 and parts[3].startswith("ability: "):
            holder = mon_at(side, slot)
            if holder is not None and not holder.mega:
                holder.note_ability_evidence(idx, to_id(parts[3][len("ability: ") :]))

    # Data facts: a species with exactly one legal ability needs no reveal.
    for mon in facts.mons.values():
        if mon.ability is None:
            single = _single_legal_ability(mon.species)
            if single:
                mon.ability = single
                mon.ability_from_data = True
    return facts


# --- Layer 2/3 context over the whole corpus ----------------------------------------------


def normalize_account(name: str) -> str:
    return to_id(name)


def group_key(account: str, lineup: list[str]) -> tuple[str, tuple[str, ...]]:
    return normalize_account(account), tuple(
        sorted(canonical_species_id(to_id(species)) for species in lineup)
    )


@dataclass
class DonorFacts:
    """One game's start-of-game facts for one species, usable as a Layer-2 donor."""

    game_id: str
    time: int
    item: str | None = None
    ability: str | None = None
    moves: tuple[str, ...] = ()
    sheet: bool = False


@dataclass
class SpeciesPrior:
    item_counts: Counter = field(default_factory=Counter)
    ability_counts: Counter = field(default_factory=Counter)
    move_counts: Counter = field(default_factory=Counter)
    complete_sets: list[tuple[str, ...]] = field(default_factory=list)
    # v2 (default): HIDDEN-conditional priors. v1 counters are dominated by what
    # announces itself (Sitrus, Focus Sash, mega stones, Intimidate), which is not
    # what a still-hidden field looks like. `_hidden` counts only sheet sets whose
    # same-game Layer-1 had NO fact for that field; `sheet_sets`/`l1_sets` split the
    # complete 4-move sets by provenance for the co-occurrence move guess.
    item_counts_hidden: Counter = field(default_factory=Counter)
    ability_counts_hidden: Counter = field(default_factory=Counter)
    hidden_item_examples: int = 0
    hidden_ability_examples: int = 0
    sheet_sets: list[tuple[str, ...]] = field(default_factory=list)
    l1_sets: list[tuple[str, ...]] = field(default_factory=list)


def game_prior_contribution(facts: GameFacts) -> dict[str, dict]:
    """`{species: {...}}` -- exactly what one game adds to a species' Layer-3 prior
    (both sides, sheet-over-L1 preference, same shape as the `_build` accumulation).
    Used for leave-one-game-out answer-key priors.
    """
    contrib: dict[str, dict] = {}
    for side in ("p1", "p2"):
        sheets = facts.sheets.get(side) or {}
        species_set = {
            species for (fact_side, species) in facts.mons if fact_side == side
        } | set(sheets)
        for species in species_set:
            mon = facts.mons.get((side, species))
            sheet = sheets.get(species)
            item = sheet.get("item") if sheet else (None if not mon or mon.item_blocked else mon.item)
            ability = sheet.get("ability") if sheet else (mon.ability if mon else None)
            moves = list(sheet.get("moves") or []) if sheet else (list(mon.moves) if mon and not mon.move_anomaly else [])
            contrib[species] = {
                "item": item, "ability": ability, "moves": tuple(sorted(moves)),
                "from_sheet": bool(sheet),
                "l1_item_known": bool(mon and mon.item and not mon.item_blocked),
                "l1_ability_evidence": bool(mon and mon.ability and not mon.ability_from_data),
            }
    return contrib


def subtract_prior(prior: SpeciesPrior, contribution: dict) -> SpeciesPrior:
    """A copy of `prior` with one game's contribution removed (leave-one-game-out
    for answer-key evaluation -- the unit's own truth must not sit inside the prior
    that guesses it)."""
    out = SpeciesPrior(
        item_counts=Counter(prior.item_counts),
        ability_counts=Counter(prior.ability_counts),
        move_counts=Counter(prior.move_counts),
        complete_sets=list(prior.complete_sets),
        item_counts_hidden=Counter(prior.item_counts_hidden),
        ability_counts_hidden=Counter(prior.ability_counts_hidden),
        hidden_item_examples=prior.hidden_item_examples,
        hidden_ability_examples=prior.hidden_ability_examples,
        sheet_sets=list(prior.sheet_sets),
        l1_sets=list(prior.l1_sets),
    )
    item, ability = contribution.get("item"), contribution.get("ability")
    moves = list(contribution.get("moves") or [])
    if item:
        out.item_counts[item] -= 1
    if ability:
        out.ability_counts[ability] -= 1
    for move_id in moves:
        out.move_counts[move_id] -= 1
    if len(moves) == 4:
        complete = tuple(sorted(moves))
        if complete in out.complete_sets:
            out.complete_sets.remove(complete)
        provenance = out.sheet_sets if contribution.get("from_sheet") else out.l1_sets
        if complete in provenance:
            provenance.remove(complete)
    if contribution.get("from_sheet"):
        if not contribution.get("l1_item_known") and item:
            out.item_counts_hidden[item] -= 1
            out.hidden_item_examples -= 1
        if not contribution.get("l1_ability_evidence") and ability:
            out.ability_counts_hidden[ability] -= 1
            out.hidden_ability_examples -= 1
    return out


# Games whose replay flags mark position/species-tracking ambiguity (Illusion
# `|replace|`, Transform, Baton Pass) can misattribute Layer-1 facts to the wrong
# species. They still rebuild their own records, but they never contribute facts
# to cross-game donors or priors, and they are never answer-key units.
EXCLUDED_PROTOCOL_FLAGS = ("illusion_or_replace", "transform", "baton_pass")


def game_flagged(game: dict) -> bool:
    flags = game.get("protocol_flags") or {}
    return any(flags.get(name) for name in EXCLUDED_PROTOCOL_FLAGS)


def illusion_possible(preview_species: list[str]) -> bool:
    """Any previewed species that COULD carry Illusion among its legal abilities
    (derived from data/champions/species.json). A never-broken Illusion emits no
    `|replace|` line, so the manifest flag alone misses these games -- see
    gen9championsvgc2026regmc-2696148503 (Focus Blast credited to 'Archaludon').
    """
    if not hasattr(illusion_possible, "_illusion_species"):
        illusion_possible._illusion_species = {
            species_id
            for species_id, entry in load_species().items()
            if "illusion" in {
                to_id(name) for name in (entry.get("abilities") or {}).values()
            }
        }
    return any(
        canonical_species_id(species) in illusion_possible._illusion_species
        for species in (preview_species or [])
    )


def illusion_affected(facts: GameFacts) -> bool:
    return any(illusion_possible(facts.preview.get(side) or []) for side in ("p1", "p2"))


class RebuildContext:
    """Corpus-wide donor index + Layer-3 priors for `rebuild_record`.

    `games` maps replay id -> {"split", "time", "accounts": {side: name},
    "rosters": {side: [6 species]}} -- the v3 selection manifest's shape. `game_facts`
    maps replay id -> GameFacts for every frozen-corpus game (donors need facts too,
    not just selected games).
    """

    def __init__(self, games: dict, game_facts: dict[str, GameFacts]):
        self.games = games
        self.game_facts = game_facts
        train_times = [g["time"] for g in games.values() if g["split"] == "train" and g.get("time")]
        dev_times = [g["time"] for g in games.values() if g["split"] == "validation" and g.get("time")]
        test_times = [g["time"] for g in games.values() if g["split"] == "test" and g.get("time")]
        if not (train_times and dev_times and test_times):
            raise ValueError("manifest is missing train/validation/test start times")
        self.dev_start = min(dev_times)
        self.test_start = min(test_times)
        # The v3 selection is time-contiguous: train < DEV_START <= validation <
        # TEST_START <= test, so the time-based donor rule below reproduces the
        # split-based one for the three held splits (asserted, not assumed).
        assert max(train_times) < self.dev_start, "train games spill past DEV_START"
        assert max(dev_times) < self.test_start, "validation games spill past TEST_START"
        assert min(test_times) >= self.test_start, "test games start before TEST_START"
        self.donors: dict[tuple[str, tuple[str, ...]], dict[tuple[str, str], list[DonorFacts]]] = {}
        self.priors: dict[str, SpeciesPrior] = {}
        self._build()

    def _build(self) -> None:
        for game_id, game in self.games.items():
            facts = self.game_facts.get(game_id)
            if (
                facts is None
                or not game.get("time")
                or game_flagged(game)
                or illusion_affected(facts)
            ):
                continue
            for side in ("p1", "p2"):
                account = (game.get("accounts") or {}).get(side)
                lineup = (game.get("rosters") or {}).get(side) or facts.preview.get(side) or []
                if not account or not lineup:
                    continue
                key = group_key(account, lineup)
                group = self.donors.setdefault(key, {})
                sheets = facts.sheets.get(side) or {}
                species_set = {
                    species
                    for (fact_side, species) in facts.mons
                    if fact_side == side
                } | set(sheets)
                for species in species_set:
                    mon = facts.mons.get((side, species))
                    sheet = sheets.get(species)
                    if sheet:
                        group.setdefault(species, []).append(
                            DonorFacts(
                                game_id=game_id,
                                time=game["time"],
                                item=sheet.get("item"),
                                ability=sheet.get("ability"),
                                moves=tuple(m for m in (sheet.get("moves") or []) if m),
                                sheet=True,
                            )
                        )
                    if mon is not None:
                        group.setdefault(species, []).append(
                            DonorFacts(
                                game_id=game_id,
                                time=game["time"],
                                item=None if mon.item_blocked else mon.item,
                                ability=mon.ability,
                                moves=() if mon.move_anomaly else tuple(mon.moves),
                                sheet=False,
                            )
                        )

        # Layer-3 priors: games starting before DEV_START only.
        for game_id, game in self.games.items():
            if not game.get("time") or game["time"] >= self.dev_start or game_flagged(game):
                continue
            facts = self.game_facts.get(game_id)
            if facts is None or illusion_affected(facts):
                continue
            for side in ("p1", "p2"):
                sheets = facts.sheets.get(side) or {}
                species_set = {
                    species for (fact_side, species) in facts.mons if fact_side == side
                } | set(sheets)
                for species in species_set:
                    prior = self.priors.setdefault(species, SpeciesPrior())
                    mon = facts.mons.get((side, species))
                    sheet = sheets.get(species)
                    item = sheet.get("item") if sheet else (None if not mon or mon.item_blocked else mon.item)
                    ability = sheet.get("ability") if sheet else (mon.ability if mon else None)
                    moves = list(sheet.get("moves") or []) if sheet else (list(mon.moves) if mon and not mon.move_anomaly else [])
                    if item:
                        prior.item_counts[item] += 1
                    if ability:
                        prior.ability_counts[ability] += 1
                    if len(moves) == 4:
                        complete = tuple(sorted(moves))
                        prior.complete_sets.append(complete)
                        (prior.sheet_sets if sheet else prior.l1_sets).append(complete)
                    for move_id in moves:
                        prior.move_counts[move_id] += 1
                    # v2 hidden-conditional: sheet values for fields the same game's
                    # Layer-1 never learned (ability data facts don't count as
                    # evidence -- they guess, not observe).
                    if sheet:
                        l1_item_known = bool(mon and mon.item and not mon.item_blocked)
                        if item and not l1_item_known:
                            prior.item_counts_hidden[item] += 1
                            prior.hidden_item_examples += 1
                        l1_ability_seen = bool(mon and mon.ability and not mon.ability_from_data)
                        if ability and not l1_ability_seen:
                            prior.ability_counts_hidden[ability] += 1
                            prior.hidden_ability_examples += 1

    def donor_boundary(self, target_time: float) -> float:
        """Latest donor start time (exclusive) usable for a target that STARTED at
        `target_time`: pre-dev period -> donors before DEV_START; dev period ->
        donors before TEST_START; test period (or missing time) -> any donor. This
        matches the split-based rule for the three held splits (see the contiguity
        assertion in __init__) and generalizes it to set-aside games by their own
        start time rather than letting them draw on everything.
        """
        if target_time and target_time < self.dev_start:
            return self.dev_start
        if target_time and target_time < self.test_start:
            return self.test_start
        return float("inf")

    def eligible_donors(
        self, account: str, lineup: list[str], species: str, target: tuple[str, float, str]
    ) -> list[DonorFacts]:
        """Donor facts for `species` in the (account, lineup) group, newer-first-free:
        eligible under `target`'s (split, time, game_id) time rule, never the target."""
        _target_split, target_time, target_id = target
        boundary = self.donor_boundary(target_time)
        entries = [
            donor
            for donor in self.donors.get(group_key(account, lineup), {}).get(species, [])
            if donor.time < boundary and donor.game_id != target_id
        ]
        return sorted(entries, key=lambda donor: abs(donor.time - target_time))


# --- resolution ----------------------------------------------------------------------------


@dataclass
class ResolvedFields:
    item: str | None = None
    item_source: str = UNKNOWN
    ability: str | None = None
    ability_source: str = UNKNOWN
    moves: list[str] = field(default_factory=list)  # resolved superset of observed
    move_sources: dict[str, int] = field(default_factory=dict)
    move_source_ids: dict[str, str] = field(default_factory=dict)  # move_id -> source
    move_anomaly: bool = False  # >4 distinct |move| lines -> no move fill (counted upstream)
    item_prior_share: float = 0.0  # chosen common-set value's share of its prior (measurement)
    ability_prior_share: float = 0.0


def resolve_fields(
    mon: MonFacts | None,
    own_sheet: dict | None,
    donors: list[DonorFacts],
    prior: SpeciesPrior | None,
    observed_moves: list[str],
    teammate_items: set[str],
    prior_version: str = "v2",
) -> ResolvedFields:
    """Resolve one own-active slot's item/ability/moves through Layers 1-3.

    `own_sheet` is the TARGET game's own `|showteam|` set for this species (public
    truth the acting player always knew; None in answer-key mode or when the side
    never opened its sheet). `donors` are pre-filtered eligible Layer-2 entries.
    Layer-1 evidence is `mon`'s; `observed_moves` are what the record already shows.
    """
    out = ResolvedFields()
    known_moves = list(dict.fromkeys(m for m in observed_moves if m))
    out.move_sources[OBSERVED] = len(set(known_moves))
    for move_id in known_moves:
        out.move_source_ids[move_id] = OBSERVED

    l1_item = None if mon is None or mon.item_blocked else mon.item
    l1_ability = mon.ability if mon else None
    l1_moves = [] if mon is None or mon.move_anomaly else list(mon.moves)

    # --- Layer 1 ---
    if l1_item:
        out.item, out.item_source = l1_item, SAME_GAME
    if l1_ability:
        out.ability = l1_ability
        out.ability_source = DATA if mon and mon.ability_from_data else SAME_GAME

    # --- the target game's OWN team sheet (public when Open Team Sheets fired):
    # same-game truth, applied ahead of other-game donors. Answer-key mode passes
    # own_sheet=None to rebuild as-if-it-never-existed.
    own_sheet_donor = None
    if own_sheet:
        if out.item is None and own_sheet.get("item"):
            out.item, out.item_source = own_sheet["item"], SAME_PLAYER_SHEET
        if out.ability is None and own_sheet.get("ability"):
            out.ability, out.ability_source = own_sheet["ability"], SAME_PLAYER_SHEET
        own_sheet_donor = DonorFacts(
            game_id="", time=0,
            item=own_sheet.get("item"), ability=own_sheet.get("ability"),
            moves=tuple(m for m in (own_sheet.get("moves") or []) if m), sheet=True,
        )

    # --- Layer 2: nearest consistent sheet, else agreeing non-sheet donors ---
    sheet_donor = next(
        (
            donor
            for donor in donors
            if donor.sheet and _donor_consistent(donor, mon)
        ),
        None,
    )
    non_sheet = [donor for donor in donors if not donor.sheet]
    if sheet_donor is not None:
        if out.item is None and sheet_donor.item:
            out.item, out.item_source = sheet_donor.item, SAME_PLAYER_SHEET
        if out.ability is None and sheet_donor.ability:
            out.ability, out.ability_source = sheet_donor.ability, SAME_PLAYER_SHEET
    else:
        # All donors must agree on one value, and it must not contradict the
        # target's own Layer-1 facts (Layer-2 rule 2's "consistent" clause).
        donor_items = {donor.item for donor in non_sheet if donor.item}
        if out.item is None and len(donor_items) == 1:
            item = donor_items.pop()
            if not l1_item or item == l1_item:
                out.item, out.item_source = item, SAME_PLAYER_GAMES
        donor_abilities = {donor.ability for donor in non_sheet if donor.ability}
        if out.ability is None and len(donor_abilities) == 1:
            ability = donor_abilities.pop()
            if not l1_ability or ability == l1_ability:
                out.ability, out.ability_source = ability, SAME_PLAYER_GAMES

    # --- Layer 3: common-set priors ---
    # v2 (default): hidden-conditional counters (fields the same game's Layer-1
    # never learned), backing off to v1 when the species has <5 hidden examples.
    item_counts = prior.item_counts if prior else Counter()
    ability_counts = prior.ability_counts if prior else Counter()
    if prior is not None and prior_version == "v2":
        if prior.hidden_item_examples >= 5:
            item_counts = prior.item_counts_hidden
        if prior.hidden_ability_examples >= 5:
            ability_counts = prior.ability_counts_hidden
    if out.item is None and item_counts:
        total = sum(item_counts.values()) or 1
        for item, count in sorted(item_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if item not in teammate_items:  # Item Clause (Flat Rules)
                out.item, out.item_source = item, COMMON_SET
                out.item_prior_share = count / total
                break
    if out.ability is None and ability_counts:
        legal = {
            to_id(name)
            for name in ((load_species().get(mon.species) or {}).get("abilities") or {}).values()
        } if mon else set()
        total = sum(ability_counts.values()) or 1
        for ability, count in sorted(ability_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if not legal or ability in legal:
                out.ability, out.ability_source = ability, COMMON_SET
                out.ability_prior_share = count / total
                break

    # --- moves: union observed -> same_game -> sheet -> donors -> common_set ---
    resolved = list(dict.fromkeys(known_moves))

    def add(move_id: str, source: str) -> None:
        if move_id and move_id not in resolved and len(resolved) < 4:
            resolved.append(move_id)
            out.move_sources[source] = out.move_sources.get(source, 0) + 1
            out.move_source_ids[move_id] = source

    # >4 distinct |move| lines -> anomaly: no move fill at all for this Pokemon in
    # this game (Layer-1 spec); observed entries stay.
    if mon is not None and mon.move_anomaly:
        out.move_anomaly = True
        out.moves = sorted(set(known_moves))
        return out
    for move_id in l1_moves:
        add(move_id, SAME_GAME)
    if own_sheet_donor is not None:
        for move_id in own_sheet_donor.moves:
            add(move_id, SAME_PLAYER_SHEET)
    if sheet_donor is not None:
        for move_id in sheet_donor.moves:
            add(move_id, SAME_PLAYER_SHEET)
    else:
        donor_union: set[str] = set()
        for donor in non_sheet:
            donor_union |= set(donor.moves)
        if donor_union and len(donor_union | set(known_moves) | set(l1_moves)) <= 4:
            for move_id in sorted(donor_union):
                add(move_id, SAME_PLAYER_GAMES)
    if prior is not None and len(resolved) < 4:
        learnset = set(load_learnsets().get(mon.species if mon else "", {}) or {})
        while len(resolved) < 4:
            known = set(resolved)
            if prior_version == "v2":
                # v2: co-occur against complete SHEET sets first (real teammate
                # sets), then complete Layer-1 sets, then marginal frequency.
                containing = [s for s in prior.sheet_sets if known <= set(s)] or [
                    s for s in prior.l1_sets if known <= set(s)
                ]
            else:
                containing = [s for s in prior.complete_sets if known <= set(s)]
            if containing:
                scores = Counter()
                for complete in containing:
                    scores.update(set(complete) - known)
            else:
                scores = Counter({m: n for m, n in prior.move_counts.items() if m not in known})
            candidates = [
                (move_id, scores.get(move_id, 0))
                for move_id in learnset - known
                if scores.get(move_id, 0)
            ]
            if not candidates:
                break
            best = sorted(candidates, key=lambda kv: (-kv[1], kv[0]))[0][0]
            add(best, COMMON_SET)

    out.moves = sorted(resolved)
    return out


def _donor_consistent(donor: DonorFacts, mon: MonFacts | None) -> bool:
    """Sheet-donor consistency against the target's Layer-1 facts (Layer-2 rule 1)."""
    if mon is None:
        return True
    if mon.item and not mon.item_blocked and donor.item != mon.item:
        return False
    if mon.ability and donor.ability != mon.ability:
        return False
    if mon.moves and not set(mon.moves) <= set(donor.moves):
        return False
    return True


# --- applying the rebuild to records --------------------------------------------------------


def _verify_rebuild_invariants(
    before: dict, after: dict, allowed_bench_additions: set[str] = frozenset()
) -> None:
    """The only permitted own-side changes: previously-null `item`/`ability` filled
    and `revealed_moves` extended (never shrunk/reordered-removed). The opponent side
    must be byte-identical. Raises AssertionError on any violation -- this is the
    "exported inputs equal original snapshots" verification, narrowed to exactly the
    fields this module may fill.
    """
    if before["state"]["opp"] != after["state"]["opp"]:
        raise AssertionError("rebuild changed the opponent side")
    for side_name in ("our",):
        before_side = before["state"][side_name]
        after_side = after["state"][side_name]
        for key in before_side:
            if key in ("active", "bench"):
                continue
            if before_side[key] != after_side[key]:
                raise AssertionError(f"rebuild changed own side key {key}")
        # Bench: entries may only be APPENDED. Existing entries keep their
        # recorded hp/status; added ones must be brought, not active, not fainted
        # (the caller's `allowed_bench_additions`), at full HP and no status.
        old_bench = before_side.get("bench") or []
        new_bench = after_side.get("bench") or []
        if len(new_bench) < len(old_bench) or new_bench[: len(old_bench)] != old_bench:
            raise AssertionError("rebuild changed existing bench entries")
        for entry in new_bench[len(old_bench) :]:
            if (
                entry.get("species_id") not in allowed_bench_additions
                or entry.get("hp_fraction") != 1.0
                or entry.get("status") is not None
            ):
                raise AssertionError(f"rebuild added an invalid bench entry: {entry}")
        for idx, (old_mon, new_mon) in enumerate(
            zip(before_side["active"], after_side["active"])
        ):
            if old_mon is None or new_mon is None:
                if old_mon != new_mon:
                    raise AssertionError(f"rebuild changed empty active slot {idx}")
                continue
            for key in old_mon:
                if key in ("item", "ability", "revealed_moves"):
                    continue
                if old_mon[key] != new_mon.get(key):
                    raise AssertionError(f"rebuild changed own mon key {key}")
            if set(new_mon.keys()) != set(old_mon.keys()):
                raise AssertionError("rebuild added/removed own mon keys")
            for field_name in ("item", "ability"):
                if old_mon.get(field_name) is not None and old_mon[field_name] != new_mon.get(field_name):
                    raise AssertionError(f"rebuild overwrote a known {field_name}")
            if not set(old_mon.get("revealed_moves") or []) <= set(
                new_mon.get("revealed_moves") or []
            ):
                raise AssertionError("rebuild removed a revealed move")


def rebuild_record(
    record: dict,
    game_facts: GameFacts,
    ctx: RebuildContext | None,
    *,
    exclude_own_sheet: bool = False,
    prior_version: str = "v2",
) -> dict:
    """Fill own-active `item`/`ability`/`revealed_moves` in one schema-8 record.
    Mutates `record` in place and returns it (callers that need the original must
    deep-copy first). Adds `own_info` + `own_fully_confirmed`; `teampreview` records
    have no active slots and get neither. `ctx=None` runs Layer-1-only rebuild
    (no donors/priors); `exclude_own_sheet` is answer-key mode -- pretend the acting
    side's own `|showteam|` never existed.
    """
    if record.get("decision_kind") not in ("turn", "forced_switch"):
        return record
    before = json.loads(json.dumps(record))  # invariant check baseline
    side = record["player"]
    sheets = {} if exclude_own_sheet else (game_facts.sheets.get(side) or {})
    game = (ctx.games.get(record["replay_id"]) if ctx else None) or {}
    account = (game.get("accounts") or {}).get(side) or game_facts.players.get(side, "")
    lineup = (game.get("rosters") or {}).get(side) or game_facts.preview.get(side) or []
    target = (game.get("split") or "", game.get("time") or 0, record["replay_id"])

    slot_infos: list[dict | None] = []
    active = record["state"]["our"]["active"]
    for mon in active:
        if not mon:
            slot_infos.append(None)
            continue
        species = canonical_species_id(mon.get("species") or "")
        facts = game_facts.mons.get((side, species))
        # Fill-window guard for item: only fill while the parser's `None` still
        # means "unrevealed" -- once an item-change event passed, `None` means the
        # item is gone (live poke-env reports nothing either).
        item_fill_ok = (
            facts is None
            or facts.first_item_change is None
            or record.get("log_line", 0) < facts.first_item_change
        )
        donors = (
            ctx.eligible_donors(account, lineup, species, target) if ctx else []
        )
        prior = ctx.priors.get(species) if ctx else None
        teammate_items = {
            other["item"]
            for other in active
            if other and other is not mon and other.get("item")
        }
        sheet = sheets.get(species)
        resolved = resolve_fields(
            facts, sheet, donors, prior,
            observed_moves=list(mon.get("revealed_moves") or []),
            teammate_items=teammate_items,
            prior_version=prior_version,
        )
        info: dict[str, object] = {"move_sources": dict(resolved.move_sources)}
        if mon.get("item"):
            info["item_source"] = OBSERVED
        elif item_fill_ok and resolved.item:
            mon["item"] = resolved.item
            info["item_source"] = resolved.item_source
        else:
            info["item_source"] = OBSERVED if mon.get("item") else UNKNOWN
        if mon.get("ability"):
            info["ability_source"] = OBSERVED
        else:
            ability = None
            ability_source = UNKNOWN
            if mon.get("mega"):
                # Post-mega snapshots report the mega forme's ability (live poke-env
                # parity: forme_change_ability once mega_evolve runs).
                ability = facts.mega_ability if facts else None
                ability_source = DATA if ability else UNKNOWN
            if ability is None and resolved.ability:
                ability, ability_source = resolved.ability, resolved.ability_source
            if ability:
                mon["ability"] = ability
                info["ability_source"] = ability_source
            else:
                info["ability_source"] = UNKNOWN
        if not (facts and facts.move_anomaly):
            merged = sorted(set(mon.get("revealed_moves") or []) | set(resolved.moves))
            mon["revealed_moves"] = merged
        slot_infos.append(info)

    # Own bench completion (live parity: poke-env's request lists the whole
    # brought four minus active/fainted; replay records only list mons that
    # already appeared). `brought` = the same appearance evidence the parser's
    # preview tracking uses; an unappeared mon cannot be damaged -> 1.0/None.
    brought = {sp for (s, sp) in game_facts.mons if s == side}
    # Additions are exactly the brought mons whose FIRST appearance is after this
    # record's raw-log line -- a mon that already appeared is covered by the
    # parser's own bench/active/faint tracking; one that faints LATER still counts
    # as a live bench mon now (the old fainted-set subtraction dropped those).
    log_line = record.get("log_line")
    additions = {
        sp for (s, sp), mon in game_facts.mons.items()
        if s == side and mon.first_seen is not None
        and log_line is not None and mon.first_seen >= log_line
    }
    bench = record["state"]["our"]["bench"]
    for species in sorted(additions):
        bench.append({"species_id": species, "hp_fraction": 1.0, "status": None})
    bench_complete = len(brought) >= 4
    fully = bench_complete and all(
        info is None
        or (
            info.get("item_source") not in _UNCONFIRMED
            and info.get("ability_source") not in _UNCONFIRMED
            and not any(src in _UNCONFIRMED for src in (info.get("move_sources") or {}))
            and sum((info.get("move_sources") or {}).values()) >= 4
        )
        for info in slot_infos
    )
    # A slot with <4 resolved move slots leaves the rest unknown.
    record["own_info"] = {"active": slot_infos, "bench_complete": bench_complete}
    record["own_fully_confirmed"] = bool(fully)
    _verify_rebuild_invariants(before, record, additions)
    return record


def load_manifest(manifest_path: Path) -> dict:
    """replay id -> manifest row for the frozen corpus."""
    games = {}
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            games[row["id"]] = row
    return games


def build_corpus_facts(manifest: dict, raw_dir: Path, cache_path: Path | None = None) -> dict[str, GameFacts]:
    """Layer-1 `analyze_log` over every frozen-corpus game; caches the serialized
    facts at `cache_path` (JSONL, one GameFacts per line) since this is the bulk of
    the corpus pre-pass.
    """
    if cache_path is not None and cache_path.exists():
        facts: dict[str, GameFacts] = {}
        import gzip

        opener = gzip.open if cache_path.suffix == ".gz" else open
        with opener(cache_path, "rt") as handle:
            for line in handle:
                facts[replay_id_from_facts_line(line)] = game_facts_from_json(json.loads(line))
        return facts
    facts = {}
    for game_id, game in sorted(manifest.items()):
        path = raw_dir / f"{game_id}.json"
        payload = json.loads(path.read_text())
        log = payload.get("log") or ""
        facts[game_id] = analyze_log(game_id, log)
    if cache_path is not None:
        import gzip

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if cache_path.suffix == ".gz" else open
        with opener(cache_path, "wt") as handle:
            for game_id in sorted(facts):
                handle.write(game_facts_to_json(facts[game_id]) + "\n")
    return facts


def replay_id_from_facts_line(line: str) -> str:
    return json.loads(line)["replay_id"]


def game_facts_to_json(facts: GameFacts) -> str:
    return json.dumps(
        {
            "replay_id": facts.replay_id,
            "players": facts.players,
            "preview": facts.preview,
            "mons": {
                f"{side}|{species}": {
                    "item": mon.item,
                    "item_blocked": mon.item_blocked,
                    "first_item_change": mon.first_item_change,
                    "ability": mon.ability,
                    "ability_from_data": mon.ability_from_data,
                    "mega": mon.mega,
                    "mega_ability": mon.mega_ability,
                    "moves": mon.moves,
                    "move_anomaly": mon.move_anomaly,
                    "first_seen": mon.first_seen,
                }
                for (side, species), mon in facts.mons.items()
            },
            "sheets": facts.sheets,
            "fainted": {side: sorted(species) for side, species in facts.fainted.items()},
        },
        sort_keys=True,
    )


def game_facts_from_json(row: dict) -> GameFacts:
    facts = GameFacts(
        replay_id=row["replay_id"], players=row["players"], preview=row["preview"],
        fainted={side: set(species) for side, species in (row.get("fainted") or {}).items()},
    )
    for key, raw in row["mons"].items():
        side, species = key.split("|", 1)
        mon = MonFacts(species)
        mon.item = raw["item"]
        mon.item_blocked = raw["item_blocked"]
        mon.first_item_change = raw["first_item_change"]
        mon.ability = raw["ability"]
        mon.ability_from_data = raw["ability_from_data"]
        mon.mega = raw["mega"]
        mon.mega_ability = raw["mega_ability"]
        mon.moves = raw["moves"]
        mon.move_anomaly = raw["move_anomaly"]
        mon.first_seen = raw.get("first_seen")
        facts.mons[(side, species)] = mon
    facts.sheets = row["sheets"]
    return facts
