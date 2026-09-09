"""Phase 3 behavior-cloning prep: parse downloaded Showdown replay logs
(`tools/download_replays.py`'s output, `data/replays/<format>/<id>.json`) into
per-decision JSONL training records.

## A critical finding that shapes this whole module

The task that specified this module assumed replay logs would reliably contain
`|showteam|` lines (from Open Team Sheets) revealing both players' full sets. Verified
against the downloaded corpus: the M-B tree has ~2940 replays and the M-C tree is still
small (~44). Only about **0.2%** of the M-B public logs contain a `|showteam|` line at
all. This is NOT a parsing bug -- it's a real property of this ladder format.
`config/formats.ts` (in the Showdown checkout) shows both
`gen9championsvgc2026regmb` and `gen9championsvgc2026regmc` use the ruleset
`"Open Team Sheets"` (opt-in), not `"Force Open Team Sheets"` (the Bo3 variant's
ruleset) -- and `server/chat-commands/core.ts`'s
`acceptopenteamsheets` handler only ever sends `>show-openteamsheets` to the sim when
`battle.players.every(curPlayer => curPlayer.wantsOpenTeamSheets)`, i.e. BOTH players
must explicitly run `/acceptopenteamsheets` before Team Preview ends. Our own bot always
accepts (`PolicyConfig.accept_open_team_sheet`), but most human ladder opponents never
opt in, so OTS essentially never actually triggers in this public corpus. `CLAUDE.md`'s
"OTS reliability" notes are about OUR bot's own behavior, not a claim about how often
real ladder games reveal sheets -- there's no contradiction, just a fact worth knowing
before training on this data: full held-item/ability/moveset ground truth is available
for well under 1% of (replay, player) pairs. Every other Pokemon's item/ability/moves
must instead be inferred incrementally from in-battle reveal events (`|-ability|`,
`|-item|`, `|-mega|`, moves actually used, etc.) -- which this module already does
regardless of `sets` availability, since that's necessary for the vast majority case.

## Design

Each replay's `log` field is walked as one flat sequence of protocol lines, segmented by
`|turn|N|` markers: segment 0 (everything before the first `|turn|1|`) is the team
preview / lead-selection phase; segment N (between `|turn|N|` and `|turn|N+1|`, or the
end of the log) is turn N's action-resolution window.

Within a normal turn segment, the first `|move|`/`|switch|` seen for each
(side, slot) is that slot's OBSERVED action for the turn (`decision_kind="turn"`,
`state` = a snapshot of PUBLIC state as of the START of the segment, i.e. before any of
its lines are applied). A `|switch|`/`|drag|` for a (side, slot) that already acted, or
whose occupant just fainted this same segment, is instead a **forced switch** (Parting
Shot/U-turn/Volt Switch self-switch, a fainted-mon replacement, Eject Button/Pack, etc.)
-- `decision_kind="forced_switch"`, its own record with a fresh mid-segment snapshot,
since the player is making a genuine (if reactive) choice about which bench mon to send.
`|cant|` records the ATTEMPTED move when the line names one (still a real choice that
got blocked -- flinch, paralysis, etc. -- see the real example this module was built
against: `|cant|p2a: Farigiraf|ability: Armor Tail|Fake Out|...`), else it's `"unknown"`.
Observed events do not certify submitted choices: called/locked moves and redirection
still require a separate reconstruction audit.

State is per-decision-player-relative (`"our"`/`"opp"`, matching `vgc.evaluator`'s own
naming), computed once per segment as a neutral `p1`/`p2` snapshot and relabeled per
player. "Bench" only ever lists species that have ACTUALLY appeared in the battle so far
(switched in at least once) -- the other 2 previewed-but-unseen species are genuinely
unknown-whether-brought until they appear, so listing them as "bench" would be a guess,
not a tracked fact.

## Schema (`SCHEMA_VERSION`)

Schema 5 is current. Missing actions in occupied turn-start slots are `unknown`;
empty slots are `no_action_required`. `action_status` records the same distinction
from observed actions. Mega Evolution is preserved even when a move is unknown.
Every record has `outcome`: win/loss/draw/unresolved. The compatibility field `won`
is True/False for wins/losses and None otherwise. Older schema descriptions below
are historical; their collapsed pass and outcome labels are not authoritative.

Every emitted record carries a `"schema": SCHEMA_VERSION` int so datasets built from
different parser versions stay distinguishable -- additive changes bump this rather than
silently changing meaning underneath an unversioned key. Schema 2 (current) added, on
top of schema 1's fields (all still present, unchanged):

- Each ACTIVE mon dict gains `"revealed_moves"`: every move id that mon has been SEEN
  using so far this game (including a blocked `|cant|`-attempted move -- the player still
  chose it, it just didn't connect), sorted for deterministic JSONL output. This is
  USAGE-based reveal, distinct from `"sets"` (the rare Open Team Sheets full moveset) --
  the two are never merged, since one is "confirmed used" and the other is "confirmed
  by the sheet," and conflating them would lose that distinction.
- Each BENCH entry changes from an HP-only summary to an identity-bearing dict:
  `{"species_id": ..., "hp_fraction": ..., "status": ...}` (previously just
  `{"species": ..., "hp_fraction": ...}` under schema 1 -- note the bench key is
  `"species_id"`, while active-mon dicts still use `"species"`; this asymmetry is
  intentional, matching exactly what was specified when schema 2 was built, not an
  oversight).

Schema 3 (current) adds, on top of schema 2's fields (all still present, unchanged):

- Every record gains `"won": bool` -- whether THIS record's `"player"` ultimately won
  the game (resolved from the log's `|win|NAME|` line via `_parse_player_names`'
  `|player|p1|NAME|`/`|player|p2|NAME|` mapping). `False` for BOTH players on a real
  `|tie|` or an unresolvable winner name, not `None` -- outcome-value training wants a
  plain binary label, and a tie is at least "not a win" for either side even though it
  isn't a "loss" either (ties are rare enough in practice that this simplification
  wasn't worth a tri-state). `ParsedReplay.winner` (`"p1"`/`"p2"`/`None`) is the
  game-level fact this is derived from, for callers that want it directly.

Schema 4 (current) adds, on top of schema 3's fields (all still present, unchanged) --
motivated by the value network being blind to anything that hasn't appeared on the
field yet, when the previewed-but-unseen roster (both sides' full 6, and which of them
have shown up so far) is exactly the per-battle context a human uses to judge a
position (e.g. "their back two are probably the rain core they haven't brought in
yet"):

- Each side dict (`"our"`/`"opp"` in a `"turn"`/`"forced_switch"` record's `"state"`)
  gains `"preview_species"`: the six species ids from that side's Team Preview `|poke|`
  lines, mega formes resolved to their base species via `_resolve_species` (defensive
  only -- a `|poke|` line never actually shows a mega forme in practice, since Mega
  Evolution can't happen before turn 1, but resolving keeps this list keyed the same way
  `known{}`/bench identity already are, and costs nothing when it's a no-op).
- Each side dict also gains `"unseen_count"`: how many of that side's `preview_species`
  have NOT appeared (switched in) as of this snapshot, i.e.
  `len(preview_species) - |preview_species ∩ appeared_order|`, always >= 0. This
  format's bring-6-pick-4 rule guarantees at least 2 of the 6 previewed species can
  NEVER appear (not brought), so `unseen_count` doesn't trend to 0 even in a long game
  -- it's "how much of this side's full previewed roster remains genuinely unknown to
  us," not a proxy for "how close to the end of the roster we are."

Never raises on a malformed replay: `parse_replay` catches any exception during its own
walk and reports it as a failed parse with a reason string; individual malformed/
ambiguous EVENTS within an otherwise-fine replay are skipped (counted, not silently
dropped) rather than aborting the whole replay. See `tools/parse_replays.py` for the CLI
that aggregates these into a corpus-wide summary.

## Known best-effort limitations (documented, not oversights)

- `|replace|` (Illusion breaking, e.g. Zoroark) retroactively corrects the ACTIVE
  slot's tracked species going forward, carrying over HP/status/boosts -- but any
  decision records ALREADY emitted while the illusion was still up keep showing the
  illusioned identity (there is no way to retcon already-written JSONL lines, and
  redoing the whole replay backward isn't worth it for an intentionally rare mechanic).
- `|-formechange|` (Morpeko Hunger Switch, etc.) is a no-op here -- it doesn't change
  Species-Clause identity (this format enforces "Species Clause: Limit one of each
  Pokemon", confirmed via the `|rule|` line every replay carries, which is why this
  module can safely key all per-mon tracking by species id alone, no nickname
  resolution needed) and isn't part of this schema's state beyond the `mega` flag.
- `|-transform|` (Ditto/Imposter) is not unwound -- the transformed mon's tracked
  identity/moves are not corrected to mirror what it copied. Out of scope for v1.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.damage import STATUS_IDS, to_id
from vgc.data import load_species

# --- protocol constants -----------------------------------------------------------------

# Bumped whenever a decision record's shape changes (additive so far: schema 2 added
# `revealed_moves` to active-mon entries and turned bench entries from HP-only summaries
# into identity-bearing {species_id, hp_fraction, status} dicts; schema 3 added a
# per-record `"won"` bool; schema 4 added per-side `"preview_species"`/`"unseen_count"`
# -- see module docstring). Every emitted record carries `"schema": SCHEMA_VERSION` so
# old datasets stay distinguishable from newer ones instead of silently being read as if
# compatible.
SCHEMA_VERSION = 5

_HP_RE = re.compile(r"(\d+)/(\d+)")

_WEATHER_ID_TO_LABEL = {
    "sunnyday": "sun",
    "desolateland": "sun",
    "raindance": "rain",
    "primordialsea": "rain",
    "sandstorm": "sand",
    "hail": "snow",
    "snowscape": "snow",
    "snow": "snow",
}
_TERRAIN_ID_TO_LABEL = {
    "electricterrain": "electric",
    "grassyterrain": "grassy",
    "psychicterrain": "psychic",
    "mistyterrain": "misty",
}
_TRICK_ROOM_ID = "trickroom"

# Side/field condition name prefixes stripped before `to_id`-normalizing (e.g.
# "move: Light Screen" -> "lightscreen", "ability: Slow Start" would strip similarly).
_CONDITION_PREFIXES = ("move: ", "ability: ", "item: ")


def _condition_id(raw: str) -> str:
    text = raw
    for prefix in _CONDITION_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return to_id(text) or text


def _species_from_details(details: str) -> str:
    """`"Floette-Eternal, L50, F"` -> `"floetteeternal"` -- species is everything
    before the first comma (level/gender/shiny are the remaining comma-separated
    fields, irrelevant to this schema).
    """
    return to_id(details.split(",", 1)[0])


def _resolve_species(species_id: str) -> tuple[str, bool]:
    """`(base_species_id, is_mega)`.

    Mega Evolution persists across switches (unlike Dynamax) -- a Pokemon that
    mega-evolved earlier and later switches out and back in shows its MEGA forme name
    in the `|switch|`/`|drag|` details field, not its base species name. Resolving that
    back to the base species id (via `vgc.data.load_species()`'s `baseSpecies` field,
    the same table `vgc.evaluator.mega_species_id` uses in the other direction) keeps
    `known{}`/`appeared_order`/bench tracking keyed by the ONE persistent team-slot
    identity (Species Clause guarantees uniqueness) instead of splitting one team
    member's pre-mega and post-mega appearances into two different tracked "species".
    """
    species = load_species().get(species_id)
    if species and species.get("isMega"):
        base = to_id(species.get("baseSpecies"))
        if base:
            return base, True
    return species_id, False


def _parse_position(token: str) -> tuple[str, int | None, str]:
    """`("p1"|"p2"|"", slot_index|None, name)` from a position token like
    `"p1a: Gholdengo"` (active slot) or `"p2: username"` (side-wide, no slot letter --
    used by `-sidestart`/`-sideend`). Returns `side=""` for anything not starting with
    `p1`/`p2` (defensive -- callers must check before using).
    """
    token = token.strip()
    if ": " in token:
        head, name = token.split(": ", 1)
    else:
        head, name = token, ""
    if head[:2] not in ("p1", "p2"):
        return "", None, name
    side = head[:2]
    slot = None
    if len(head) > 2 and head[2] in ("a", "b"):
        slot = 0 if head[2] == "a" else 1
    return side, slot, name


def _parse_hp(hp_field: str) -> tuple[float, str | None, bool] | None:
    """`(hp_fraction, status_id_or_None, fainted)` from an HP field like `"77/100"`,
    `"40/100 brn"`, or `"0 fnt"`. Returns `None` if the field doesn't match any
    recognized shape (caller should skip updating HP/status for that event rather than
    guess -- see `_STATUS_IDS`/module docstring's robustness contract).
    """
    text = hp_field.strip()
    if text.startswith("0 fnt"):
        return 0.0, None, True
    match = _HP_RE.search(text)
    if match is None:
        return None
    current, maximum = int(match.group(1)), int(match.group(2))
    if maximum <= 0:
        return None
    fraction = current / maximum
    rest = text[match.end() :].lower()
    status = next((status_id for status_id in STATUS_IDS if status_id in rest), None)
    return fraction, status, current <= 0


# --- mutable per-battle tracking state ---------------------------------------------------


@dataclass
class MonKnowledge:
    species_id: str
    hp_fraction: float = 1.0
    status: str | None = None
    boosts: dict[str, int] = field(default_factory=dict)
    item: str | None = None
    ability: str | None = None
    mega: bool = False
    tera_type: str | None = None
    fainted: bool = False
    # Every move id this mon has been SEEN using so far this game (accumulates across
    # turns; a `|cant|`-attempted move counts too -- see the "move"/"cant" tag handlers
    # below). Schema 2+ only -- see module docstring and `SCHEMA_VERSION`.
    revealed_moves: set[str] = field(default_factory=set)


@dataclass
class SideState:
    preview_species: list[str] = field(default_factory=list)
    known: dict[str, MonKnowledge] = field(default_factory=dict)
    active: list[str | None] = field(default_factory=lambda: [None, None])
    side_conditions: set[str] = field(default_factory=set)
    appeared_order: list[str] = field(default_factory=list)
    lead_species: list[str] = field(default_factory=list)
    sets_by_species: dict[str, dict] = field(default_factory=dict)

    def mon(self, species_id: str) -> MonKnowledge:
        known = self.known.get(species_id)
        if known is None:
            known = MonKnowledge(species_id)
            self.known[species_id] = known
        return known

    def note_appearance(self, species_id: str) -> None:
        if species_id not in self.appeared_order:
            self.appeared_order.append(species_id)


@dataclass
class BattleState:
    sides: dict[str, SideState] = field(
        default_factory=lambda: {"p1": SideState(), "p2": SideState()}
    )
    weather: str | None = None
    terrain: str | None = None
    trick_room: bool = False
    turn: int = 0


# --- state snapshotting ------------------------------------------------------------------


def _mon_dict(known: MonKnowledge | None) -> dict | None:
    if known is None:
        return None
    result: dict[str, object] = {
        "species": known.species_id,
        "hp_fraction": round(known.hp_fraction, 4),
        "status": known.status,
        "boosts": dict(known.boosts),
        "item": known.item,
        "ability": known.ability,
        "mega": known.mega,
        # Schema 2+: every move id observed for this mon so far this game, sorted for
        # deterministic JSONL output -- see MonKnowledge.revealed_moves' docstring.
        "revealed_moves": sorted(known.revealed_moves),
    }
    if known.tera_type:
        result["tera_type"] = known.tera_type
    return result


def _side_dict(side: SideState) -> dict:
    active_species = set(species_id for species_id in side.active if species_id)
    bench = [
        {
            "species_id": species_id,
            "hp_fraction": round(side.known[species_id].hp_fraction, 4),
            "status": side.known[species_id].status,
        }
        for species_id in side.appeared_order
        if species_id not in active_species and not side.known[species_id].fainted
    ]
    appeared = set(side.appeared_order)
    # Schema 4: how much of this side's full previewed roster is still genuinely
    # unknown to us -- see module docstring's schema-4 note for why this doesn't trend
    # to 0 (bring-6-pick-4 guarantees at least 2 previewed species can never appear).
    unseen_count = sum(1 for species_id in side.preview_species if species_id not in appeared)
    return {
        "active": [
            _mon_dict(side.known.get(species_id)) if species_id else None
            for species_id in side.active
        ],
        "bench": bench,
        "side_conditions": sorted(side.side_conditions),
        "preview_species": list(side.preview_species),
        "unseen_count": unseen_count,
    }


def _field_dict(state: BattleState) -> dict:
    return {
        "weather": state.weather,
        "terrain": state.terrain,
        "trick_room": state.trick_room,
        "turn": state.turn,
    }


def _snapshot(state: BattleState) -> dict:
    """Neutral (`p1`/`p2`) snapshot of public state, relabeled per-player by
    `_for_player` when a decision record is actually emitted.
    """
    return {
        "p1": _side_dict(state.sides["p1"]),
        "p2": _side_dict(state.sides["p2"]),
        "field": _field_dict(state),
    }


def _for_player(snapshot: dict, player: str) -> dict:
    other = "p2" if player == "p1" else "p1"
    return {"our": snapshot[player], "opp": snapshot[other], "field": snapshot["field"]}


# --- showteam (packed team) parsing -------------------------------------------------------


def parse_showteam_line(parts: list[str]) -> tuple[str, dict[str, dict]] | None:
    """`(side, {species_id: set_dict})` from a split `|showteam|SIDE|PACKED` line's
    parts (i.e. `line.split("|")`). Reuses poke-env's own packed-team parser (the exact
    format `Teams.pack()`/this project's `teams/*.packed.txt` files already use) rather
    than reimplementing it. Returns `None` if the line doesn't have the expected shape.
    """
    if len(parts) < 4 or parts[1] != "showteam":
        return None
    side = parts[2]
    if side not in ("p1", "p2"):
        return None
    packed = "|".join(parts[3:])
    try:
        mons = Teambuilder.parse_packed_team(packed)
    except Exception:  # noqa: BLE001 -- a malformed showteam line must not crash the parse
        return None
    sets: dict[str, dict] = {}
    for mon in mons:
        species_id = to_id(mon.species or mon.nickname)
        if not species_id:
            continue
        sets[species_id] = {
            "species": species_id,
            "item": to_id(mon.item) or None,
            "ability": to_id(mon.ability) or None,
            "moves": [to_id(move) for move in (mon.moves or [])],
            "nature": to_id(mon.nature) or None,
            "tera_type": to_id(mon.tera_type) or None if mon.tera_type else None,
        }
    return side, sets


# --- decision records ---------------------------------------------------------------------


@dataclass
class ParsedReplay:
    replay_id: str
    ok: bool
    fail_reason: str | None
    records: list[dict]
    skipped: Counter  # reason -> count, for records skipped within an otherwise-ok parse
    showteam_players: set[str]
    # "p1" | "p2" | None -- None covers both a real `|tie|` (Showdown protocol has no
    # winner in that case) and a `|win|NAME|` whose NAME didn't match either side's
    # `|player|` line (shouldn't happen on a real replay; defensive for hand-built test
    # logs and any truly malformed input). See `_parse_player_names`/schema 3's `"won"`.
    winner: str | None = None


def _target_slot_label(
    mover_side: str, mover_slot: int, target_token: str, is_spread: bool
) -> str | None:
    if is_spread:
        return "spread"
    if not target_token:
        return "self"
    target_side, target_slot, _name = _parse_position(target_token)
    if target_side not in ("p1", "p2"):
        return None  # unparseable target -- caller decides whether to skip
    if target_side == mover_side:
        if target_slot is None or target_slot == mover_slot:
            return "self"
        return "ally"
    return f"opp{target_slot}" if target_slot is not None else "opp0"


def _apply_switch(
    state: BattleState, side: str, slot: int, species_id: str, is_mega: bool, hp_field: str
) -> None:
    """Applies a switch-in to `state`. `species_id`/`is_mega` must already be resolved
    via `_resolve_species` by the caller (single resolution point, so the action record
    built alongside this call and the tracked state always agree on the same base
    species id -- see `_resolve_species`'s docstring for why that resolution matters).
    """
    side_state = state.sides[side]
    side_state.active[slot] = species_id
    side_state.note_appearance(species_id)
    known = side_state.mon(species_id)
    known.fainted = False
    if is_mega:
        known.mega = True
    parsed_hp = _parse_hp(hp_field)
    if parsed_hp is not None:
        fraction, status, fainted = parsed_hp
        known.hp_fraction = fraction
        known.status = status
        known.fainted = fainted


def _process_segment(
    state: BattleState,
    turn_num: int,
    lines: list[str],
    records: list[dict],
    skipped: Counter,
    showteam_players: set[str],
) -> None:
    state.turn = turn_num
    if turn_num == 0:
        _process_teampreview_segment(state, lines, showteam_players)
        return

    snapshot = _snapshot(state)
    acted: dict[tuple[str, int], dict] = {}
    pending_mega: set[tuple[str, int]] = set()
    fainted_this_segment: set[tuple[str, int]] = set()

    for line in lines:
        if not line or line == "|":
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == "showteam":
            result = parse_showteam_line(parts)
            if result is not None:
                side, sets = result
                state.sides[side].sets_by_species.update(sets)
                showteam_players.add(side)
            continue

        if tag in ("switch", "drag"):
            if len(parts) < 5:
                skipped["malformed_switch"] += 1
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                skipped["malformed_switch"] += 1
                continue
            species_id, is_mega = _resolve_species(_species_from_details(parts[3]))
            already_acted = (side, slot) in acted
            just_fainted = (side, slot) in fainted_this_segment
            if already_acted or just_fainted:
                # Snapshot BEFORE applying the switch: the decision point is "my mon
                # just fainted/pivoted, what do I send in", so the slot should still
                # read as empty/pre-switch in the recorded state.
                forced_snapshot = _for_player(_snapshot(state), side)
                records.append(
                    {
                        "replay_id": None,  # filled in by parse_replay
                        "rating": None,
                        "schema": SCHEMA_VERSION,
                        "player": side,
                        "turn": turn_num,
                        "decision_kind": "forced_switch",
                        "state": forced_snapshot,
                        "action": {"slot": slot, "switch_species": species_id},
                    }
                )
            else:
                acted[(side, slot)] = {"kind": "switch", "switch_species": species_id}
            _apply_switch(state, side, slot, species_id, is_mega, parts[4])
            continue

        if tag == "-mega":
            if len(parts) < 5:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is None:
                continue
            known = state.sides[side].mon(species_id)
            known.mega = True
            known.item = to_id(parts[4]) or known.item
            if (side, slot) in acted and acted[(side, slot)].get("kind") == "move":
                acted[(side, slot)]["mega"] = True
            else:
                pending_mega.add((side, slot))
            continue

        if tag == "-terastallize":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).tera_type = to_id(parts[3])
            continue

        if tag == "move":
            if len(parts) < 4:
                skipped["malformed_move"] += 1
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                skipped["malformed_move"] += 1
                continue
            if (side, slot) in acted:
                continue  # a second move report for an already-acted slot -- ignore it
            move_id = to_id(parts[3])
            target_token = parts[4] if len(parts) > 4 else ""
            extra = parts[5:]
            is_spread = any(part.startswith("[spread]") for part in extra)
            target_slot = _target_slot_label(side, slot, target_token, is_spread)
            mega = (side, slot) in pending_mega
            pending_mega.discard((side, slot))
            acted[(side, slot)] = {
                "kind": "move",
                "move_id": move_id,
                "target_slot": target_slot,
                "mega": mega,
            }
            acting_species = state.sides[side].active[slot]
            if acting_species is not None and move_id:
                state.sides[side].mon(acting_species).revealed_moves.add(move_id)
            continue

        if tag == "cant":
            if len(parts) < 3:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            if (side, slot) in acted:
                continue
            attempted_move = parts[4] if len(parts) > 4 and parts[4] else None
            if attempted_move:
                attempted_move_id = to_id(attempted_move)
                acted[(side, slot)] = {
                    "kind": "move",
                    "move_id": attempted_move_id,
                    "target_slot": None,
                    "mega": (side, slot) in pending_mega,
                }
                cant_species = state.sides[side].active[slot]
                if cant_species is not None and attempted_move_id:
                    state.sides[side].mon(cant_species).revealed_moves.add(attempted_move_id)
            else:
                # The protocol says the mon could not act, but does not reveal the
                # submitted choice. It must not become a teachable pass label.
                acted[(side, slot)] = {
                    "kind": "unknown", "mega": (side, slot) in pending_mega
                }
            continue

        if tag == "faint":
            if len(parts) < 3:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                known = state.sides[side].mon(species_id)
                known.fainted = True
                known.hp_fraction = 0.0
            state.sides[side].active[slot] = None
            fainted_this_segment.add((side, slot))
            continue

        if tag in ("-damage", "-heal"):
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is None:
                continue
            parsed_hp = _parse_hp(parts[3])
            if parsed_hp is None:
                continue
            fraction, status, fainted = parsed_hp
            known = state.sides[side].mon(species_id)
            known.hp_fraction = fraction
            known.status = status
            if fainted:
                known.fainted = True
            continue

        if tag in ("-boost", "-unboost"):
            if len(parts) < 5:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is None:
                continue
            stat = parts[3]
            try:
                amount = int(parts[4])
            except ValueError:
                continue
            known = state.sides[side].mon(species_id)
            delta = amount if tag == "-boost" else -amount
            known.boosts[stat] = max(-6, min(6, known.boosts.get(stat, 0) + delta))
            continue

        if tag == "-setboost":
            if len(parts) < 5:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is None:
                continue
            try:
                value = int(parts[4])
            except ValueError:
                continue
            state.sides[side].mon(species_id).boosts[parts[3]] = max(-6, min(6, value))
            continue

        if tag == "-clearboost":
            if len(parts) < 3:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).boosts = {}
            continue

        if tag == "-status":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).status = (
                    parts[3] if parts[3] in STATUS_IDS else None
                )
            continue

        if tag == "-curestatus":
            if len(parts) < 3:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).status = None
            continue

        if tag == "-ability":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).ability = to_id(parts[3])
            continue

        if tag == "-item":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).item = to_id(parts[3])
            continue

        if tag == "-enditem":
            if len(parts) < 3:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id = state.sides[side].active[slot]
            if species_id is not None:
                state.sides[side].mon(species_id).item = None
            continue

        if tag == "-sidestart":
            if len(parts) < 4:
                continue
            side, _slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2"):
                continue
            state.sides[side].side_conditions.add(_condition_id(parts[3]))
            continue

        if tag == "-sideend":
            if len(parts) < 4:
                continue
            side, _slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2"):
                continue
            state.sides[side].side_conditions.discard(_condition_id(parts[3]))
            continue

        if tag == "-fieldstart":
            if len(parts) < 3:
                continue
            condition = _condition_id(parts[2])
            if condition == _TRICK_ROOM_ID:
                state.trick_room = True
            elif condition in _TERRAIN_ID_TO_LABEL:
                state.terrain = _TERRAIN_ID_TO_LABEL[condition]
            continue

        if tag == "-fieldend":
            if len(parts) < 3:
                continue
            condition = _condition_id(parts[2])
            if condition == _TRICK_ROOM_ID:
                state.trick_room = False
            elif _TERRAIN_ID_TO_LABEL.get(condition) == state.terrain:
                state.terrain = None
            continue

        if tag == "-weather":
            if len(parts) < 3:
                continue
            if any(part.startswith("[upkeep]") for part in parts[3:]):
                continue  # a reminder tick, not a new weather -- no state change
            weather_id = to_id(parts[2])
            state.weather = (
                None
                if weather_id in (None, "", "none")
                else _WEATHER_ID_TO_LABEL.get(weather_id, weather_id)
            )
            continue

        if tag == "replace":
            if len(parts) < 4:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            new_species, is_mega = _resolve_species(_species_from_details(parts[3]))
            old_species = state.sides[side].active[slot]
            new_known = state.sides[side].mon(new_species)
            if is_mega:
                new_known.mega = True
            if old_species is not None and old_species in state.sides[side].known:
                old_known = state.sides[side].known[old_species]
                new_known.hp_fraction = old_known.hp_fraction
                new_known.status = old_known.status
                new_known.boosts = dict(old_known.boosts)
            state.sides[side].active[slot] = new_species
            state.sides[side].note_appearance(new_species)
            continue

        # Everything else (-formechange, -transform, -start, -end, -fail, -immune,
        # -activate, -resisted, -supereffective, -crit, -miss, -hint, -message, raw,
        # inactive, j/l join-leave, c chat, etc.) is intentionally a no-op -- see module
        # docstring's "known best-effort limitations".

    for player in ("p1", "p2"):
        side_state = state.sides[player]
        if not any(side_state.active) and not side_state.appeared_order:
            skipped["no_alive_mons_for_player"] += 1
            continue
        action: dict[str, dict] = {}
        action_status: dict[str, str] = {}
        for slot in (0, 1):
            key = f"slot{slot}"
            observed = acted.get((player, slot))
            if observed is not None:
                action[key] = observed
                action_status[key] = "unknown" if observed["kind"] == "unknown" else "observed"
            elif snapshot[player]["active"][slot] is None:
                action[key] = {"kind": "no_action_required"}
                action_status[key] = "no_action_required"
            else:
                action[key] = {"kind": "unknown", "mega": (player, slot) in pending_mega}
                action_status[key] = "unknown"
        records.append(
            {
                "replay_id": None,
                "rating": None,
                "schema": SCHEMA_VERSION,
                "player": player,
                "turn": turn_num,
                "decision_kind": "turn",
                "state": _for_player(snapshot, player),
                "action": action,
                "action_status": action_status,
            }
        )


def _process_teampreview_segment(
    state: BattleState, lines: list[str], showteam_players: set[str]
) -> None:
    lead_seen: dict[str, int] = {"p1": 0, "p2": 0}
    for line in lines:
        if not line or line == "|":
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]

        if tag == "poke":
            if len(parts) < 4:
                continue
            side = parts[2]
            if side not in ("p1", "p2"):
                continue
            # Mega resolution here is defensive-only (see schema 4's docstring note) --
            # a `|poke|` line never actually names a mega forme in practice, since Mega
            # Evolution can't happen before turn 1.
            species_id, _is_mega = _resolve_species(_species_from_details(parts[3]))
            if species_id and species_id not in state.sides[side].preview_species:
                state.sides[side].preview_species.append(species_id)
            continue

        if tag == "showteam":
            result = parse_showteam_line(parts)
            if result is not None:
                side, sets = result
                state.sides[side].sets_by_species.update(sets)
                showteam_players.add(side)
            continue

        if tag in ("switch", "drag"):
            if len(parts) < 5:
                continue
            side, slot, _name = _parse_position(parts[2])
            if side not in ("p1", "p2") or slot is None:
                continue
            species_id, is_mega = _resolve_species(_species_from_details(parts[3]))
            _apply_switch(state, side, slot, species_id, is_mega, parts[4])
            if lead_seen[side] < 2:
                state.sides[side].lead_species.append(species_id)
                lead_seen[side] += 1
            continue

        # -ability (e.g. an Intimidate trigger on the leads switching in) and anything
        # else pre-turn-1 is a no-op here -- state effects before turn 1 barely matter
        # for the teampreview record itself (picked/lead_order only).


def _teampreview_record(state: BattleState, showteam_players: set[str]) -> list[dict]:
    records = []
    preview_snapshot = {
        "p1": {"preview": list(state.sides["p1"].preview_species)},
        "p2": {"preview": list(state.sides["p2"].preview_species)},
        "field": {"turn": 0},
    }
    for player in ("p1", "p2"):
        side_state = state.sides[player]
        records.append(
            {
                "replay_id": None,
                "rating": None,
                "schema": SCHEMA_VERSION,
                "player": player,
                "turn": 0,
                "decision_kind": "teampreview",
                "state": _for_player(preview_snapshot, player),
                "action": {
                    "picked": list(side_state.appeared_order[:4]),
                    "lead_order": list(side_state.lead_species[:2]),
                    "derived": True,
                },
            }
        )
    return records


def _parse_player_names(log: str) -> dict[str, str]:
    """`{"p1": display_name, "p2": display_name}` from every `|player|p1|NAME|...`/
    `|player|p2|NAME|...` line in the log (last-write-wins if a side's line appears more
    than once, e.g. a mid-battle name/avatar update -- rare, but the protocol allows
    it). Needed to resolve `|win|NAME|`'s display name back to a side -- see `winner`.
    A hand-built test log with no `|player|` lines at all just yields `{}`, so `winner`
    stays `None` (matches this function's own "no info -> unresolved" contract).
    """
    names: dict[str, str] = {}
    for line in log.splitlines():
        if not line.startswith("|player|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        side = parts[2]
        name = parts[3]
        if side in ("p1", "p2") and name:
            names[side] = name
    return names


def parse_replay(replay_id: str, rating: int | None, log: str) -> ParsedReplay:
    """Parse one replay's protocol `log` into per-decision JSONL-ready records.

    Never raises -- any exception during the walk is caught and reported via
    `ParsedReplay.ok=False`/`fail_reason` instead (see module docstring's robustness
    contract). Individual malformed/ambiguous EVENTS within an otherwise-parseable
    replay are skipped and counted in `ParsedReplay.skipped`, not treated as a failure.
    """
    try:
        state = BattleState()
        records: list[dict] = []
        skipped: Counter = Counter()
        showteam_players: set[str] = set()
        player_names = _parse_player_names(log)
        winner: str | None = None

        segment_lines: list[str] = []
        pending_turn = 0
        ended = False
        outcome_kind = "unresolved"
        for line in log.splitlines():
            if line.startswith("|turn|"):
                _process_segment(
                    state, pending_turn, segment_lines, records, skipped, showteam_players
                )
                segment_lines = []
                parts = line.split("|")
                try:
                    pending_turn = int(parts[2])
                except (IndexError, ValueError):
                    pending_turn += 1
                continue
            if line.startswith("|win|") or line.startswith("|tie|") or line == "|tie":
                _process_segment(
                    state, pending_turn, segment_lines, records, skipped, showteam_players
                )
                segment_lines = []
                ended = True
                if line.startswith("|win|"):
                    parts = line.split("|")
                    winner_name = parts[2] if len(parts) > 2 else ""
                    for side, name in player_names.items():
                        if name == winner_name:
                            winner = side
                            outcome_kind = "win"
                            break
                    else:
                        skipped["unresolved_winner_name"] += 1
                else:
                    outcome_kind = "draw"
                break
            segment_lines.append(line)
        if not ended:
            _process_segment(state, pending_turn, segment_lines, records, skipped, showteam_players)

        records = _teampreview_record(state, showteam_players) + records

        # Attach each player's showteam parse (if any) to the FIRST record we emit for
        # that (replay, player) -- see module docstring's "sets" schema note. `"won"` is
        # attached to EVERY record (a game-level fact, not decision-specific) -- see
        # schema 3's note in the module docstring.
        attached: set[str] = set()
        for record in records:
            player = record["player"]
            record["replay_id"] = replay_id
            record["rating"] = rating
            if outcome_kind == "win":
                record["outcome"] = "win" if player == winner else "loss"
            else:
                record["outcome"] = outcome_kind
            # Compatibility for schema 3/4 readers; schema 5 readers use `outcome`.
            if record["outcome"] == "win":
                record["won"] = True
            elif record["outcome"] == "loss":
                record["won"] = False
            else:
                record["won"] = None
            if player in showteam_players and player not in attached:
                record["sets"] = state.sides[player].sets_by_species
                attached.add(player)

        return ParsedReplay(
            replay_id=replay_id,
            ok=True,
            fail_reason=None,
            records=records,
            skipped=skipped,
            showteam_players=showteam_players,
            winner=winner,
        )
    except Exception as exc:  # noqa: BLE001 -- a single bad replay must never crash a corpus run
        reason = f"{type(exc).__name__}: {exc}"
        return ParsedReplay(
            replay_id=replay_id,
            ok=False,
            fail_reason=reason,
            records=[],
            skipped=Counter(),
            showteam_players=set(),
        )
