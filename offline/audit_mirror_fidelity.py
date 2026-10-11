"""Audit: does the live exact judge's Showdown mirror rebuild the battle correctly?

The live judge (`vgc.exact_judge` -> `vgc.rl.public_search` -> `vgc.rl.live_mirror.LiveExactMirror`
-> `vgc.mechanics_state` snapshot -> `tools/sim_worker.mjs` `patchPublic`) rebuilds the CURRENT
battle inside a local Showdown from the bot's PUBLIC view plus guesses for hidden information,
then simulates candidate moves from that rebuilt root. Nothing else checks the rebuild, and a
rebuild defect silently corrupts every judged decision (2026-10-09 code reading found foes with
half their real bulk, blanked items, impossible opponent Megas, ...).

This script plays seeded local battles (`vgc.rl.env.DirectBattle`, varied teams, simple players)
and at many decision points, from BOTH seats, builds the mirror root EXACTLY the way the live
judge does (judge config, posterior belief selection, every hidden-timer branch), then compares
it with the true battle's Showdown state (`dump`).

Two modes:

1. FIELD COMPARISON (normal mirror: public information only, hidden facts guessed).
   * public fields must match exactly (species/forme, HP, status, boosts, volatiles + their
     public counters, field/side conditions + turns, Choice lock, Mega state, our PP, ...);
     a foe's HP must be consistent with the percent it displays;
   * hidden fields (a foe's Stat Points / nature / unrevealed item, ability, moves, the bring)
     are not defects: they are reported as GUESS ACCURACY statistics.
2. ORACLE EQUIVALENCE. The mirror is built through the SAME rebuild code path, but with the
   TRUE hidden information injected (true foe sets/spreads/items/bring, true sleep/confusion
   timers). From that root and from a clone of the true battle the same joint choices are
   applied with the same PRNG seed; the protocol output (ignoring timestamps and names) and the
   next-state dumps must be identical. Any difference is a pure reconstruction bug.

Mismatches are grouped by field / mechanic with counts and one example each. Groups that are
documented limitations or legitimately hidden are classified; anything else is UNEXPLAINED
(a defect). `--fail-on-unexplained` exits 1 on those.

    .venv/bin/python offline/audit_mirror_fidelity.py --battles 60 --workers 8 --seed 20261010
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import multiprocessing
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from vgc.actions import enumerate_joint_orders
from vgc.battle_memory import BattleMemory
from vgc.config import REPO_ROOT, RUNS_DIR
from vgc.damage import to_id
from vgc.exact_judge import judge_config, with_exact_timers
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl import live_mirror as _live_mirror
from vgc.rl.agents import make_direct_agent
from vgc.rl.env import DirectBattle, InvalidChoice, SimWorker, choice_string
from vgc.rl.hidden_state import HiddenStateHypothesis
from vgc.rl.live_mirror import LiveExactMirror, MirrorHypothesis

SIDES = ("p1", "p2")
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "mirror_fidelity_audit.json"
DEFAULT_MANIFESTS = (
    REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json",
    REPO_ROOT / "data" / "selfplay" / "mc_sheet_pool_v2" / "manifest.json",
)
MAX_STEPS = 300
MAX_EXAMPLES = 2

# --- synthetic teams (mechanics-heavy, Item Clause respected like the real format) -----------

INTERESTING_MOVES = (
    "psychup skillswap simplebeam worryseed entrainment roleplay trick switcheroo knockoff "
    "thief covet bugbite pluck haze clearsmog spectralthief topsyturvy heartswap guardswap "
    "powerswap bellydrum swordsdance nastyplot dragondance calmmind bulkup curse substitute "
    "leechseed taunt encore disable torment yawn toxic willowisp spore trickroom tailwind "
    "reflect lightscreen auroraveil raindance sunnyday sandstorm snowscape stealthrock spikes "
    "toxicspikes stickyweb destinybond perishsong imprison transform gastroacid coreenforcer "
    "soak magicpowder reflecttype wideguard quickguard helpinghand followme ragepowder "
    "allyswitch partingshot uturn voltswitch flipturn batonpass teleport recycle fling "
    "electricterrain grassyterrain mistyterrain psychicterrain gravity magicroom wonderroom "
    "trickortreat forestscurse electrify instruct afteryou quash dragontail circlethrow roar "
    "whirlwind mist safeguard lifedew acupressure shellsmash tidyup victorydance stuffcheeks "
    "cottonguard rest sleeptalk snore copycat mimic metronome assist memento hurricane "
    "thunderwave glare nuzzle confuseray swagger flatter fakeout finalgambit explosion "
    "healingwish lunardance protect detect spikyshield banefulbunker kingsshield endure "
    "focusenergy dragoncheer acidspray screech charm feint incinerate "
    "burnup ingrain aquaring wish rapidspin defog courtchange mortalspin leechlife "
    "payback bodypress stoneaxe ceaselessedge scald chillingwater rockslide earthquake "
    "closecombat flareblitz dragonclaw tripleaxel icywind electroweb snarl bulldoze "
    "outrage thrash petaldance uproar rollout iceball furycutter solarbeam skullbash "
    "fly dig dive bounce phantomforce shadowforce skyattack futuresight doomdesire "
    "hyperbeam gigaimpact bind wrap firespin whirlpool sandtomb infestation magmastorm "
    "wish healbell aromatherapy mudsport sleeptalk bulletpunch suckerpunch aquajet "
    "iceshard extremespeed lastrespects ragefist terablast meanlook block spiderweb "
    "guillotine fissure rockpolish autotomize agility ironthorns icebeam thunderbolt "
    "flamethrower surf shadowball psychic moonblast dazzlinggleam knockoff"
).split()
GOOD_ITEMS = (
    "sitrusberry focussash leftovers lifeorb rockyhelmet shellbell whiteherb mentalherb "
    "lumberry leppaberry electricseed grassyseed mistyseed psychicseed ejectbutton redcard "
    "airballoon choicescarf lightclay damprock heatrock smoothrock icyrock terrainextender "
    "focusband expertbelt muscleband wiseglasses quickclaw widelens zoomlens scopelens "
    "oranberry persimberry chestoberry cheriberry rawstberry pechaberry aspearberry "
    "chartiberry babiriberry occaberry yacheberry kasibberry shucaberry passhoberry "
    "bindingband shedshell ironball kingsrock metronome normalgem powerherb ejectpack "
    "throatspray clearamulet covertcloak loadeddice mirrorherb punchingglove "
    "blunderpolicy roomservice weaknesspolicy utilityumbrella boosterenergy"
).split()


class TeamBuilder:
    """Random legal-ish teams biased toward mechanics that move state around."""

    def __init__(self) -> None:
        data = REPO_ROOT / "data" / "champions"
        self.species = json.loads((data / "species.json").read_text())
        self.learnsets = json.loads((data / "learnsets.json").read_text())
        self.items = json.loads((data / "items.json").read_text())
        self.moves = json.loads((data / "moves.json").read_text())
        self.interesting = set(INTERESTING_MOVES)
        self.stones: dict[str, str] = {}
        for item_id, item in self.items.items():
            if item.get("megaStone"):
                for user in item.get("itemUser") or ([item["megaEvolves"]] if item.get("megaEvolves") else []):
                    self.stones.setdefault(to_id(user), item["name"])
        self.legal_items = {to_id(i) for i in self.items}
        self.good_items = [i for i in GOOD_ITEMS if i in self.legal_items]
        self.base_species = sorted(
            sid
            for sid, s in self.species.items()
            if s.get("isNonstandard") is None
            and not s.get("isMega")
            and not s.get("battleOnly")
            and s.get("forme") not in ("Gmax",)
        )
        self.megaable = sorted(sid for sid in self.base_species if sid in self.stones)

    def _learnable(self, sid: str) -> list[str]:
        for key in (sid, to_id(self.species[sid].get("baseSpecies"))):
            ls = self.learnsets.get(key)
            if ls:
                return [m for m in ls if m in self.moves]
        return []

    def pack(self, rng: random.Random, *, must_include: tuple[str, ...] = ()) -> str:
        chosen: list[str] = [s for s in must_include if s in self.species]
        used_bases: set[str] = {to_id(self.species[s].get("baseSpecies")) for s in chosen}
        mega_bias = rng.random() < 0.6
        while len(chosen) < 6:
            pool = self.megaable if (mega_bias and len(chosen) < 3) else self.base_species
            sid = rng.choice(pool)
            base = to_id(self.species[sid].get("baseSpecies"))
            if base not in used_bases:
                used_bases.add(base)
                chosen.append(sid)
        members = []
        stoned = 0
        used_items: set[str] = set()
        for sid in chosen:
            spec = self.species[sid]
            learn = self._learnable(sid)
            hot = [m for m in learn if m in self.interesting]
            rest = [m for m in learn if m not in self.interesting]
            picks: list[str] = []
            pool_hot = hot[:]
            rng.shuffle(pool_hot)
            picks += pool_hot[: rng.choice((2, 3, 3, 4))]
            pool_rest = rest[:]
            rng.shuffle(pool_rest)
            for move in pool_rest:
                if len(picks) >= 4:
                    break
                if move not in picks:
                    picks.append(move)
            if "protect" not in picks and rng.random() < 0.5 and len(picks) >= 1:
                picks[-1] = "protect"
            if not picks:
                picks = ["protect"]
            item = None
            stone_name = self.stones.get(sid)
            if stone_name and stoned < 2 and rng.random() < 0.7:
                item = stone_name
                stoned += 1
            else:
                options = [i for i in self.good_items if i not in used_items]
                item = rng.choice(options)
            used_items.add(to_id(item))
            abilities = [to_id(a) for a in spec["abilities"].values()]
            ability = rng.choice(abilities)
            name = spec["name"]
            evs = ["", "", "", "", "", ""]
            for stat in rng.sample(range(6), 2):
                evs[stat] = "32"
            evs[rng.randrange(6)] = evs[rng.randrange(6)] or "2"
            members.append(
                f"{name}||{item}|{ability}|{','.join(picks)}|Serious|{','.join(evs)}||||50|"
            )
        return "]".join(members)


def load_pool_teams(manifests: list[Path], limit: int) -> list[str]:
    teams: list[str] = []
    for manifest in manifests:
        if not manifest.exists():
            continue
        for entry in json.loads(manifest.read_text()):
            path = manifest.parent / entry["file"]
            if path.exists():
                teams.append(path.read_text().strip())
    random.Random(7).shuffle(teams)
    # Species Clause: a pool team naming one species twice cannot occur in the real format and
    # breaks every by-species lookup (the live bot and this audit alike).
    legal = [t for t in teams if len(_species_names(t)) == len(set(_species_names(t)))]
    return legal[:limit]


def _species_names(packed: str) -> list[str]:
    names = []
    for member in packed.split("]"):
        fields = member.split("|")
        name = fields[1] or fields[0] if len(fields) > 1 else fields[0]
        names.append(to_id(name))
    return names


# --- reading Showdown's state into comparable views ---------------------------------------

_REF = re.compile(r"^\[(\w+):(.*)\]$")


def _ref_id(value: Any) -> str:
    """``"[Species:vivillon]"`` -> ``"vivillon"``; ids of any other value via to_id."""

    if isinstance(value, str):
        match = _REF.match(value)
        if match:
            return to_id(match.group(2)) if match.group(1) != "Pokemon" else match.group(2)
        return to_id(value)
    if isinstance(value, dict) and "move" in value:
        return _ref_id(value["move"])
    return ""


def _clean(value: Any, drop: frozenset[str] = frozenset()) -> Any:
    if isinstance(value, str):
        match = _REF.match(value)
        return f"{match.group(1)}:{match.group(2)}" if match else value
    if isinstance(value, list):
        return [_clean(v, drop) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v, drop) for k, v in value.items() if k not in drop}
    return value


# keys of an effect state that are bookkeeping rather than battle state
_EFFECT_DROP = frozenset(
    {"effectOrder", "id", "name", "target", "source", "sourceEffect", "isSlotCondition", "m",
     "publicValue"}
)


def _effect_view(state: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = _clean(state or {}, _EFFECT_DROP)
    return {k: v for k, v in cleaned.items() if v is not None}


def mon_view(p: dict[str, Any]) -> dict[str, Any]:
    set_ = p.get("set") or {}
    status_state = p.get("statusState") or {}
    last_move = p.get("lastMove")
    hurt = p.get("hurtThisTurn")
    return {
        "name": set_.get("name", ""),
        "set_species": to_id(set_.get("species", "")),
        "species": _ref_id(p.get("species")),
        "base_species": _ref_id(p.get("baseSpecies")),
        "gender": p.get("gender") or "",
        "types": [to_id(t) for t in (p.get("types") or [])],
        "added_type": to_id(p.get("addedType")),
        "hp": int(p.get("hp", 0)),
        "maxhp": int(p.get("maxhp", 0)),
        "status": p.get("status") or "",
        "status_state": {
            k: status_state[k] for k in ("time", "startTime", "stage") if k in status_state
        },
        "boosts": {k: int(v) for k, v in (p.get("boosts") or {}).items()},
        "ability": to_id(p.get("ability")),
        "base_ability": to_id(p.get("baseAbility")),
        "item": to_id(p.get("item")),
        "last_item": to_id(p.get("lastItem")),
        "item_knocked_off": bool(p.get("itemKnockedOff")),
        "ate_berry": bool(p.get("ateBerry")),
        "used_item_this_turn": bool(p.get("usedItemThisTurn")),
        "stored_stats": dict(p.get("storedStats") or {}),
        "base_stored_stats": dict(p.get("baseStoredStats") or {}),
        "weighthg": p.get("weighthg"),
        "move_slots": [
            {
                "id": to_id(slot.get("id")),
                "pp": slot.get("pp"),
                "maxpp": slot.get("maxpp"),
                "disabled": bool(slot.get("disabled")),
            }
            for slot in (p.get("moveSlots") or [])
        ],
        "volatiles": {
            to_id(vid): _effect_view(state) for vid, state in (p.get("volatiles") or {}).items()
        },
        "last_move": _ref_id(last_move) if last_move else "",
        "last_move_used": _ref_id(p.get("lastMoveUsed")) if p.get("lastMoveUsed") else "",
        "move_this_turn": to_id(p.get("moveThisTurn")),
        "hurt_this_turn": bool(hurt),
        "last_damage": int(p.get("lastDamage") or 0),
        "times_attacked": int(p.get("timesAttacked") or 0),
        "move_last_turn_failed": p.get("moveLastTurnResult") is False,
        "is_active": bool(p.get("isActive")),
        "position": p.get("position"),
        "fainted": bool(p.get("fainted")),
        "switch_flag": bool(p.get("switchFlag")),
        "force_switch_flag": bool(p.get("forceSwitchFlag")),
        "newly_switched": bool(p.get("newlySwitched")),
        "previously_switched_in": int(p.get("previouslySwitchedIn") or 0),
        "active_turns": min(1, int(p.get("activeTurns") or 0)),  # only zero vs non-zero is read
        "active_move_actions": min(1, int(p.get("activeMoveActions") or 0)),
        "truant_turn": bool(p.get("truantTurn")),
        "trapped": p.get("trapped"),
        "maybe_trapped": bool(p.get("maybeTrapped")),
        "can_mega_evo": bool(p.get("canMegaEvo")),
        "transformed": bool(p.get("transformed")),
        "terastallized": bool(p.get("terastallized")),
        "illusion": bool(p.get("illusion")),
        "set": {
            "nature": to_id(set_.get("nature")),
            "evs": dict(set_.get("evs") or {}),
            "item": to_id(set_.get("item")),
            "ability": to_id(set_.get("ability")),
            "moves": [to_id(mv) for mv in (set_.get("moves") or [])],
            "species": to_id(set_.get("species")),
        },
    }


def side_view(side: dict[str, Any]) -> dict[str, Any]:
    mons = [mon_view(p) for p in side["pokemon"]]
    active_names = []
    for ref in side.get("active") or []:
        slot = _ref_id(ref)
        active_names.append(slot)
    return {
        "id": side["id"],
        "mons": mons,
        "active_slots": active_names,
        "pokemon_left": side.get("pokemonLeft"),
        "fainted_last_turn": bool(side.get("faintedLastTurn")),
        "fainted_this_turn": bool(side.get("faintedThisTurn")),
        "total_fainted": side.get("totalFainted"),
        "conditions": {
            to_id(cid): _effect_view(cs) for cid, cs in (side.get("sideConditions") or {}).items()
        },
        "slot_conditions": [
            {to_id(cid): _effect_view(cs) for cid, cs in (slot or {}).items()}
            for slot in (side.get("slotConditions") or [])
        ],
        "side_last_move": _ref_id(side.get("lastMove")) if side.get("lastMove") else "",
    }


def state_view(state: dict[str, Any]) -> dict[str, Any]:
    field = state["field"]
    return {
        "turn": state["turn"],
        "request_state": state.get("requestState"),
        "weather": to_id(field.get("weather")),
        "weather_state": _effect_view(field.get("weatherState")),
        "terrain": to_id(field.get("terrain")),
        "terrain_state": _effect_view(field.get("terrainState")),
        "pseudo": {
            to_id(k): _effect_view(v) for k, v in (field.get("pseudoWeather") or {}).items()
        },
        "last_move": _ref_id(state.get("lastMove")) if state.get("lastMove") else "",
        "sides": {side["id"]: side_view(side) for side in state["sides"]},
    }


def _hp_percent(hp: int, maxhp: int) -> int:
    if hp <= 0:
        return 0
    return max(1, hp * 100 // maxhp)  # Showdown (Champions): floor(100 * hp / maxhp) || 1


# --- comparing views -----------------------------------------------------------------------

_MISSING = "<missing>"


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict) and value:
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), inner, out)
    elif isinstance(value, list) and value and any(isinstance(v, (dict, list)) for v in value):
        for index, inner in enumerate(value):
            _flatten(f"{prefix}[{index}]", inner, out)
    else:
        out[prefix] = value


def leaf_diffs(a: Any, b: Any, prefix: str = "") -> list[tuple[str, Any, Any]]:
    fa: dict[str, Any] = {}
    fb: dict[str, Any] = {}
    _flatten(prefix, a, fa)
    _flatten(prefix, b, fb)
    out = []
    for key in sorted(set(fa) | set(fb)):
        left, right = fa.get(key, _MISSING), fb.get(key, _MISSING)
        if left != right and not (left in ({}, _MISSING) and right in ({}, _MISSING)):
            out.append((key, left, right))
    return out


class Findings:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(
        self, category: str, path: str, mon: str, truth: Any, mirror: Any, **extra: Any
    ) -> None:
        self.items.append(
            {
                "category": category,
                "path": path,
                "mon": mon,
                "truth": truth,
                "mirror": mirror,
                **extra,
            }
        )


STRICT_BOTH = (
    "species", "base_species", "gender", "types", "added_type", "status", "boosts", "fainted",
    "is_active", "switch_flag", "force_switch_flag", "transformed", "terastallized",
)  # fmt: skip
STRICT_OWN = (
    "hp", "maxhp", "stored_stats", "base_stored_stats", "weighthg", "item", "ability",
    "base_ability", "last_item", "move_slots", "can_mega_evo",
)  # fmt: skip
# State Showdown reads on later turns that is not (or not fully) public. `move_this_turn`,
# `hurt_this_turn`, `last_damage`, `newly_switched` and `previously_switched_in` are reset
# every turn or read by no Champions move/ability/item, so they are not compared at all.
BOOKKEEPING = (
    "last_move", "last_move_used", "times_attacked", "move_last_turn_failed", "active_turns",
    "active_move_actions", "truant_turn", "item_knocked_off", "ate_berry", "trapped",
    "maybe_trapped",
)  # fmt: skip


def _species_base(species_id: str, species_data: dict[str, Any]) -> str:
    return to_id((species_data.get(species_id) or {}).get("baseSpecies")) or species_id


def pair_mons(
    truth_side: dict[str, Any], mirror_side_view: dict[str, Any], species_data: dict[str, Any]
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """(matched pairs, unmatched truth mons, unmatched mirror mons)."""

    remaining = list(mirror_side_view["mons"])
    pairs: list[tuple[dict, dict]] = []
    leftover: list[dict] = []
    for t in truth_side["mons"]:
        key = to_id(t["name"])
        match = next((m for m in remaining if to_id(m["name"]) == key), None)
        if match is None:
            base = _species_base(t["set_species"], species_data)
            match = next(
                (m for m in remaining if _species_base(m["set_species"], species_data) == base),
                None,
            )
        if match is None:
            leftover.append(t)
            continue
        remaining.remove(match)
        pairs.append((t, match))
    return pairs, leftover, remaining


def compare_mon(
    out: Findings, label: str, t: dict[str, Any], m: dict[str, Any], snap_mon: Any
) -> None:
    """Compare one matched Pokemon. ``label`` is ``own`` or ``foe``; ``snap_mon`` is the
    bot's `PokemonMechanicsState` for it (what is publicly known)."""

    key = f"{label}:{t['name']}"
    both_fainted = t["fainted"] and m["fainted"]
    for f in STRICT_BOTH:
        if both_fainted and f in ("status", "boosts", "types", "added_type", "gender"):
            continue  # a fainted Pokemon's leftovers are not state
        if f == "gender" and label == "foe" and not getattr(snap_mon, "revealed", False):
            continue  # an unrevealed foe's gender is hidden
        if t[f] != m[f]:
            for path, a, b in leaf_diffs(t[f], m[f], f):
                out.add("public", path, key, a, b)
    if t["fainted"] and m["fainted"]:
        return
    if label == "own":
        for f in STRICT_OWN:
            if f == "move_slots":
                tm, mm = t[f], m[f]
                tids = [s["id"] for s in tm]
                mids = [s["id"] for s in mm]
                if sorted(tids) != sorted(mids):
                    out.add("public", "move_slots.ids", key, tids, mids)
                else:
                    mby = {s["id"]: s for s in mm}
                    for s in tm:
                        o = mby[s["id"]]
                        # `disabled` is compared through the request instead: the snapshot marks
                        # every move of a recharging/locked mon unavailable, Showdown does not.
                        for attr in ("pp", "maxpp"):
                            if s[attr] != o[attr]:
                                out.add("public", f"move_slots.{attr}", key, s, o)
            elif t[f] != m[f]:
                for path, a, b in leaf_diffs(t[f], m[f], f):
                    out.add("public", path, key, a, b)
    else:
        # A foe's HP is public only as a percent.
        tp = _hp_percent(t["hp"], t["maxhp"])
        mp = _hp_percent(m["hp"], m["maxhp"])
        hp_hidden = (not t["is_active"]) and t["ability"] == "regenerator"
        if tp != mp and not hp_hidden:
            out.add(
                "public", "foe_hp_percent", key, tp, mp,
                truth_abs=[t["hp"], t["maxhp"]], mirror_abs=[m["hp"], m["maxhp"]],
            )  # fmt: skip
        # item knowledge
        item_state = getattr(snap_mon, "item_state", "unknown") if snap_mon else "unknown"
        if item_state == "known" and t["item"] != m["item"]:
            out.add("public", "foe_item_known", key, t["item"], m["item"])
        elif item_state == "consumed" and m["item"]:
            out.add("public", "foe_item_consumed", key, t["item"], m["item"])
        revealed = {mv.id for mv in (getattr(snap_mon, "moves", ()) or ())}
        mirror_moves = {s["id"] for s in m["move_slots"]}
        if not t["transformed"] and revealed - mirror_moves - {"struggle", "recharge"}:
            out.add(
                "public", "foe_revealed_move_missing", key, sorted(revealed), sorted(mirror_moves)
            )
        if (
            snap_mon is not None
            and getattr(snap_mon, "ability_known", False)
            and getattr(snap_mon, "ability_id", None)
            and snap_mon.ability_id != m["ability"]
        ):
            out.add(
                "public", "foe_ability_known", key, snap_mon.ability_id, m["ability"],
                truth_ability=t["ability"],
            )  # fmt: skip
    for f in BOOKKEEPING:
        if f in ("active_turns", "active_move_actions") and not t["is_active"]:
            continue  # Showdown resets them on switch-IN; a benched value is stale and unread
        if t[f] != m[f]:
            out.add("bookkeeping", f, key, t[f], m[f])
    # status counters
    if t["status"] and t["status"] == m["status"]:
        for path, a, b in leaf_diffs(t["status_state"], m["status_state"], "status_state"):
            out.add(
                "timer" if t["status"] in ("slp", "frz") else "public",
                f"{t['status']}.{path}", key, a, b,
            )  # fmt: skip
    # volatiles
    tv, mv = t["volatiles"], m["volatiles"]
    for vid in sorted(set(tv) | set(mv)):
        if vid in tv and vid not in mv:
            out.add("volatile_missing", vid, key, tv[vid], None)
        elif vid in mv and vid not in tv:
            out.add("volatile_extra", vid, key, None, mv[vid])
        else:
            for path, a, b in leaf_diffs(tv[vid], mv[vid], vid):
                out.add("volatile_value", path, key, a, b)


def compare_views(
    out: Findings,
    tview: dict[str, Any],
    mview: dict[str, Any],
    seat: str,
    snapshot: Any,
    species_data: dict[str, Any],
    *,
    skip_foe_detail: bool = False,
) -> dict[str, Any]:
    """Compare two `state_view`s. Returns pairing info for the guess-accuracy statistics."""

    foe = "p2" if seat == "p1" else "p1"
    if tview["turn"] != mview["turn"]:
        out.add("public", "turn", "-", tview["turn"], mview["turn"])
    if tview["request_state"] != mview["request_state"]:
        out.add("public", "request_state", "-", tview["request_state"], mview["request_state"])
    for f in ("weather", "terrain"):
        if tview[f] != mview[f]:
            out.add("public", f, "field", tview[f], mview[f])
    for f in ("weather_state", "terrain_state"):
        for path, a, b in leaf_diffs(tview[f], mview[f], f):
            out.add(
                "field_value", path, "field", a, b,
                setter=str(tview[f].get("sourceSlot", ""))[:2],
            )  # fmt: skip
    for pid in sorted(set(tview["pseudo"]) | set(mview["pseudo"])):
        if pid not in tview["pseudo"]:
            out.add("field_extra", pid, "field", None, mview["pseudo"][pid])
        elif pid not in mview["pseudo"]:
            out.add("field_missing", pid, "field", tview["pseudo"][pid], None)
        else:
            for path, a, b in leaf_diffs(tview["pseudo"][pid], mview["pseudo"][pid], pid):
                out.add(
                    "field_value", path, "field", a, b,
                    setter=str(tview["pseudo"][pid].get("sourceSlot", ""))[:2],
                )  # fmt: skip
    if tview["last_move"] != mview["last_move"]:
        out.add("bookkeeping", "battle.last_move", "-", tview["last_move"], mview["last_move"])
    pairing: dict[str, Any] = {}
    for side_id in SIDES:
        ts, ms = tview["sides"][side_id], mview["sides"][side_id]
        label = "foe" if side_id == foe else "own"
        for cid in sorted(set(ts["conditions"]) | set(ms["conditions"])):
            if cid not in ms["conditions"]:
                out.add("side_missing", cid, label, ts["conditions"][cid], None)
            elif cid not in ts["conditions"]:
                out.add("side_extra", cid, label, None, ms["conditions"][cid])
            else:
                for path, a, b in leaf_diffs(ts["conditions"][cid], ms["conditions"][cid], cid):
                    out.add(
                        "side_value", path, label, a, b,
                        setter=str(ts["conditions"][cid].get("sourceSlot", ""))[:2],
                    )  # fmt: skip
        if ts["slot_conditions"] != ms["slot_conditions"]:
            out.add("side_value", "slot_conditions", label, ts["slot_conditions"],
                    ms["slot_conditions"])  # fmt: skip
        for f in ("pokemon_left", "fainted_last_turn", "fainted_this_turn", "total_fainted",
                  "side_last_move"):  # fmt: skip
            if ts[f] != ms[f]:
                out.add("bookkeeping", f"side.{f}", label, ts[f], ms[f])
        pairs, lost_truth, extra_mirror = pair_mons(ts, ms, species_data)
        pairing[label] = {"pairs": pairs, "lost_truth": lost_truth, "extra_mirror": extra_mirror}
        if label == "foe" and (lost_truth or extra_mirror):
            out.add(
                "bring", "unmatched", label, [m["name"] for m in lost_truth],
                [m["name"] for m in extra_mirror],
            )  # fmt: skip
        if label == "foe" and skip_foe_detail:
            continue
        snap_side = snapshot.opponent_side if label == "foe" else snapshot.our_side
        snap_by_name = {to_id(sm.name): sm for sm in snap_side.pokemon}
        # active slot membership by position
        t_active = {m["position"]: to_id(m["name"]) for m in ts["mons"] if m["is_active"]}
        m_active = {m["position"]: to_id(m["name"]) for m in ms["mons"] if m["is_active"]}
        if sorted(t_active) != sorted(m_active):
            out.add("public", "active_positions", label, sorted(t_active), sorted(m_active))
        for t, m in pairs:
            if t["illusion"] or m["illusion"]:
                continue
            compare_mon(out, label, t, m, snap_by_name.get(to_id(t["name"])))
        if label == "own":
            t_order = [to_id(x["name"]) for x in ts["mons"]]
            m_order = [to_id(x["name"]) for x in ms["mons"]]
            if t_order != m_order:
                out.add("bookkeeping", "own_side_order", label, t_order, m_order)
    return pairing


# --- requests and protocol lines ------------------------------------------------------------

_IDENT = re.compile(r"(p[12][abc]?): ([^|\]]+)")


def _canon_ident_name(name: str, name_map: dict[str, str]) -> str:
    return name_map.get(name.strip(), name.strip())


def canonical_line(line: str, name_map: dict[str, str]) -> str:
    return _IDENT.sub(lambda m: f"{m.group(1)}: {_canon_ident_name(m.group(2), name_map)}", line)


def name_map_for(state: dict[str, Any], species_data: dict[str, Any]) -> dict[str, str]:
    """Nickname -> base species id (one per team: Species Clause)."""

    mapping: dict[str, str] = {}
    for side in state["sides"]:
        for p in side["pokemon"]:
            sp = to_id((p.get("set") or {}).get("species", ""))
            mapping[(p.get("set") or {}).get("name", "")] = _species_base(sp, species_data)
    return mapping


def request_from_lines(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        split = line.split("|")
        if len(split) > 2 and split[1] == "request":
            payload = "|".join(split[2:])
            if payload:
                return json.loads(payload)
    return None


def normalize_request(
    request: dict[str, Any] | None,
    name_map: dict[str, str],
    species_data: dict[str, Any],
) -> dict[str, Any]:
    if not request:
        return {}
    req = copy.deepcopy(request)
    req.pop("rqid", None)
    side = req.get("side") or {}
    pokes = side.get("pokemon") or []
    canon: list[dict[str, Any]] = []
    for p in pokes:
        ident = str(p.get("ident", ""))
        prefix, _, name = ident.partition(": ")
        species = to_id(str(p.get("details", "")).split(",")[0])
        base = _species_base(species, species_data)
        key = _canon_ident_name(name, name_map)
        p = dict(p)
        p["ident"] = f"{prefix}: {key}"
        p["_key"] = key or base
        canon.append(p)
    side["pokemon"] = sorted(canon, key=lambda q: q["_key"])
    req["side"] = side
    return req


# --- the audit context ----------------------------------------------------------------------


@contextlib.contextmanager
def _shared_worker(worker: SimWorker):
    original = _live_mirror.SimWorker
    _live_mirror.SimWorker = lambda *a, **k: worker  # type: ignore[assignment]
    try:
        yield
    finally:
        _live_mirror.SimWorker = original  # type: ignore[assignment]


TRACE = ""  # regex on group names: print matching findings as they happen (--trace)


def _zero_pair() -> list[int]:
    return [0, 0]


def _new_acc() -> dict[str, Any]:
    return {
        "battles": 0,
        "steps": 0,
        "points": 0,
        "oracle_points": 0,
        "oracle_pairs": 0,
        "oracle_pass": 0,
        "oracle_fail": 0,
        "oracle_gap": 0,
        "observations": Counter(),
        "episodes": Counter(),
        "examples": defaultdict(list),
        "coverage": Counter(),
        "guess": defaultdict(_zero_pair),
        "timer_support": defaultdict(_zero_pair),
        "errors": [],
        "winners": Counter(),
        "agents": Counter(),
        "seconds": Counter(),
    }


def merge_acc(into: dict[str, Any], part: dict[str, Any]) -> None:
    for key in ("battles", "steps", "points", "oracle_points", "oracle_pairs", "oracle_pass",
                "oracle_fail", "oracle_gap"):  # fmt: skip
        into[key] += part[key]
    for key in ("observations", "episodes", "coverage", "winners", "agents", "seconds"):
        into[key].update(part[key])
    for key in ("guess", "timer_support"):
        for name, (ok, total) in part[key].items():
            into[key][name][0] += ok
            into[key][name][1] += total
    for group, examples in part["examples"].items():
        room = MAX_EXAMPLES - len(into["examples"][group])
        into["examples"][group].extend(examples[:room])
    into["errors"].extend(part["errors"])


class Ctx:
    def __init__(self, worker: SimWorker, config: PolicyConfig, job: dict[str, Any]) -> None:
        self.worker = worker
        self.config = config
        self.job = job
        self.species = json.loads((REPO_ROOT / "data" / "champions" / "species.json").read_text())
        self.mirrors: dict[tuple[str, str], LiveExactMirror] = {}
        self.clone_counter = 0

    def mirror(self, battle_id: str, seat: str, team: str, config: PolicyConfig) -> LiveExactMirror:
        key = (battle_id, seat)
        mirror = self.mirrors.get(key)
        if mirror is None:
            with _shared_worker(self.worker):
                mirror = LiveExactMirror(team, config)
            # unique root ids on the shared worker
            mirror._counter = iter(
                f"{battle_id}-{seat}-{i}" for i in range(10**9)
            )  # type: ignore[assignment]
            self.mirrors[key] = mirror
        mirror.config = config
        return mirror

    def clone_id(self, prefix: str) -> str:
        self.clone_counter += 1
        return f"{prefix}{self.clone_counter}"

    def dump(self, worker: SimWorker, battle_id: str) -> dict[str, Any]:
        return worker.request({"cmd": "dump", "id": battle_id})["state"]


def _group(mode: str, f: dict[str, Any]) -> str:
    return f"{mode}/{f['category']}:{re.sub(r'\[\d+\]', '', f['path'])}"


def record(
    acc: dict[str, Any], mode: str, findings: list[dict[str, Any]], meta: dict[str, Any],
    episodes: set[tuple],
) -> list[str]:
    groups = []
    for f in findings:
        group = _group(mode, f)
        groups.append(group)
        if TRACE and re.search(TRACE, group):
            print(
                f"[trace] {meta['battle']} turn {meta.get('turn')} seat {meta['seat']} {group} "
                f"path={f['path']} "
                f"mon={f['mon']} truth={json.dumps(f['truth'], default=str)[:200]} "
                f"mirror={json.dumps(f['mirror'], default=str)[:200]} "
                f"choices={meta.get('choices')}"
            )
            all_lines = meta.get("lines") or []
            name = str(f["mon"]).split(":", 1)[-1].strip()
            touching = [ln for ln in all_lines if name and name in ln] if name != "-" else []
            for line in (touching[-40:] if touching else all_lines[-8:]):
                print("      |", line[:140])
        acc["observations"][group] += 1
        ep = (meta["battle"], meta["seat"], f["mon"], group)
        if ep in episodes:
            continue
        episodes.add(ep)
        acc["episodes"][group] += 1
        if len(acc["examples"][group]) < MAX_EXAMPLES:
            acc["examples"][group].append(
                {
                    **{k: v for k, v in meta.items() if k != "lines"},
                    "mon": f["mon"],
                    "truth": f["truth"],
                    "mirror": f["mirror"],
                    "extra": {k: v for k, v in f.items()
                              if k not in ("category", "path", "mon", "truth", "mirror")},
                    "lines": [
                        ln for ln in (meta.get("lines") or [])
                        if str(f["mon"]).split(":", 1)[-1].strip() in ln
                    ][-8:] or (meta.get("lines") or [])[-6:],
                }
            )  # fmt: skip
    return groups


# --- statistics gathered at every audited point -------------------------------------------


def note_coverage(acc: dict[str, Any], tview: dict[str, Any]) -> None:
    cov = acc["coverage"]
    cov[f"weather:{tview['weather'] or 'none'}"] += 1
    cov[f"terrain:{tview['terrain'] or 'none'}"] += 1
    for pid in tview["pseudo"]:
        cov[f"field:{pid}"] += 1
    for side in tview["sides"].values():
        for cid, state in side["conditions"].items():
            layers = state.get("layers")
            cov[f"side:{cid}" + (f"x{layers}" if layers else "")] += 1
        for m in side["mons"]:
            if m["fainted"]:
                cov["fainted"] += 1
                continue
            if m["status"]:
                cov[f"status:{m['status']}"] += 1
            for vid in m["volatiles"]:
                cov[f"volatile:{vid}"] += 1
            for stat, value in m["boosts"].items():
                if value:
                    cov[f"boost:{stat}{'+' if value > 0 else '-'}"] += 1
            if m["item"]:
                cov[f"item:{m['item']}"] += 1
            elif m["set"]["item"]:
                cov["item:consumed_or_lost"] += 1
            if m["ability"]:
                cov[f"ability:{m['ability']}"] += 1
            if "mega" in m["species"]:
                cov["mega_evolved"] += 1
                cov[f"mega:{m['species']}"] += 1
            if m["can_mega_evo"]:
                cov["can_mega_evo"] += 1
            if m["transformed"]:
                cov["transformed"] += 1
            if m["illusion"]:
                cov["illusion_active"] += 1
            if m["is_active"]:
                cov["active_mon"] += 1
            if m["base_ability"] != m["ability"]:
                cov["ability_changed"] += 1
            if m["status_state"].get("stage"):
                cov["toxic_stage>0"] += 1


def _guess(acc: dict[str, Any], name: str, ok: bool) -> None:
    acc["guess"][name][0] += int(ok)
    acc["guess"][name][1] += 1


def note_guess_accuracy(
    acc: dict[str, Any], pairing: dict[str, Any], snapshot: Any
) -> None:
    """How good are the mirror's guesses of what the bot cannot see?"""

    foe = pairing["foe"]
    snap_by_name = {to_id(sm.name): sm for sm in snapshot.opponent_side.pokemon}
    for t in foe["lost_truth"]:
        _guess(acc, "bring: truth foe mon present in mirror", False)
    for t, m in foe["pairs"]:
        if t["illusion"] or m["illusion"]:
            continue
        sm = snap_by_name.get(to_id(t["name"]))
        revealed = bool(sm and sm.revealed)
        if not revealed:
            _guess(acc, "bring: truth foe mon present in mirror", True)
            _guess(acc, "bring: UNREVEALED truth foe mon present in mirror", True)
        else:
            _guess(acc, "bring: truth foe mon present in mirror", True)
        if sm is not None and sm.item_state == "unknown" and t["set"]["item"]:
            _guess(acc, "foe item (hidden) exactly right", t["item"] == m["item"])
        if sm is not None and not sm.ability_known:
            _guess(acc, "foe ability (hidden) exactly right", t["ability"] == m["ability"])
        known_moves = {mv.id for mv in (sm.moves if sm else ())}
        hidden_truth = [mv for mv in t["set"]["moves"] if mv not in known_moves]
        mirror_moves = {s["id"] for s in m["move_slots"]}
        for mv in hidden_truth:
            _guess(acc, "foe unrevealed move present in mirror set", mv in mirror_moves)
        _guess(acc, "foe nature exactly right", t["set"]["nature"] == m["set"]["nature"])
        for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
            tv = int(t["set"]["evs"].get(stat, 0) or 0)
            mv = int(m["set"]["evs"].get(stat, 0) or 0)
            _guess(acc, f"foe Stat Points {stat} exactly right", tv == mv)
        ts = t["stored_stats"].get("spe")
        ms = m["stored_stats"].get("spe")
        if ts and ms:
            _guess(acc, "foe Speed within 5% of truth", abs(ts - ms) <= 0.05 * ts)


def timer_support(
    acc: dict[str, Any], tview: dict[str, Any], seat: str, snapshot: Any,
    payloads: list[dict[str, Any]],
) -> None:
    """Is the TRUE sleep / confusion remaining time among the branches the judge searches?"""

    for key, side_id in (("our", seat), ("opponent", "p2" if seat == "p1" else "p1")):
        side = tview["sides"][side_id]
        by_name = {to_id(m["name"]): m for m in side["mons"]}
        snap_side = snapshot.our_side if key == "our" else snapshot.opponent_side
        for sm in snap_side.pokemon:
            tm = by_name.get(to_id(sm.name))
            if tm is None or tm["fainted"] or tm["illusion"]:
                continue
            if tm["status"] == "slp" and sm.status == "slp":
                truth_time = tm["status_state"].get("time")
                options = {
                    p.get(key, {}).get(sm.species_id, {}).get("sleepTime") for p in payloads
                }
                acc["timer_support"]["sleep remaining time covered"][0] += int(truth_time in options)
                acc["timer_support"]["sleep remaining time covered"][1] += 1
            conf = tm["volatiles"].get("confusion")
            if conf is not None and any(e.id == "confusion" for e in sm.effects):
                truth_time = conf.get("time")
                options = {
                    p.get(key, {}).get(sm.species_id, {}).get("confusionTime") for p in payloads
                }
                acc["timer_support"]["confusion remaining time covered"][0] += int(
                    truth_time in options
                )
                acc["timer_support"]["confusion remaining time covered"][1] += 1


# --- audit points ------------------------------------------------------------------------


@contextlib.contextmanager
def public_view_only(bt: Any):
    """Strip the private simulator root DirectBattle attaches to the observation, as a live
    websocket observation never carries it: the NORMAL mirror must not be able to read it."""

    saved = {}
    for name in ("_vgc_direct_root", "_vgc_direct_side"):
        if name in vars(bt):
            saved[name] = vars(bt).pop(name)
    try:
        yield
    finally:
        vars(bt).update(saved)


TIMER_PATHS = ("confusion.time", "slp.", "frz.")


def _reclassify_timers(items: list[dict[str, Any]]) -> None:
    for f in items:
        if f["category"] == "volatile_value" and f["path"] == "confusion.time":
            f["category"] = "timer"


def own_foe_names(tview: dict[str, Any], seat: str) -> dict[str, str]:
    return {}


def _illusion_up(tview: dict[str, Any]) -> bool:
    """An intact Illusion shows the wrong identity BY DESIGN (a documented limitation)."""

    return any(m["illusion"] for side in tview["sides"].values() for m in side["mons"])


def audit_normal_point(
    ctx: Ctx, d: DirectBattle, seat: str, memory: BattleMemory, team: str,
    acc: dict[str, Any], meta: dict[str, Any], episodes: set[tuple],
) -> None:
    """Build the judge's mirror root(s) for ``seat`` and compare with the true battle."""

    worker = ctx.worker
    bt = d.battles[seat]
    truth_state = ctx.dump(worker, d.battle_id)
    tview = state_view(truth_state)
    note_coverage(acc, tview)
    if _illusion_up(tview):
        acc["coverage"]["points_skipped_illusion_up"] += 1
        return
    memory.observe_battle(bt)
    setattr(bt, "_vgc_battle_memory", memory)
    cfg = ctx.config
    jcfg = with_exact_timers(judge_config(cfg, 6), bt, cfg)
    mirror = ctx.mirror(d.battle_id, seat, team, jcfg)
    started = time.perf_counter()
    with public_view_only(bt):
        beliefs = mirror.hypotheses(bt, memory)
        snapshot = snapshot_battle(bt, opponent_mega_unknown=cfg.exact_mirror_opponent_mega)
        root = None
        first = True
        try:
            for index, belief in enumerate(beliefs):
                root = (
                    mirror.rebase(root, bt, belief)
                    if root is not None
                    else mirror.build(bt, belief)
                )
                mview = state_view(ctx.dump(worker, root.battle_id))
                findings = Findings()
                pairing = compare_views(findings, tview, mview, seat, snapshot, ctx.species)
                # our own request as the mirror regenerates it vs the real one
                name_map = name_map_for(truth_state, ctx.species)
                real_req = normalize_request(bt.last_request, name_map, ctx.species)
                mirror_req = normalize_request(
                    request_from_lines(root.last_lines[seat]), name_map, ctx.species
                )
                for path, a, b in leaf_diffs(real_req, mirror_req, "request"):
                    findings.add("request", path, "own", a, b)
                _reclassify_timers(findings.items)
                record(acc, "normal", findings.items, {**meta, "branch": index}, episodes)
                if first:
                    note_guess_accuracy(acc, pairing, snapshot)
                    first = False
            timer_support(acc, tview, seat, snapshot, [b.payload for b in beliefs])
        finally:
            if root is not None:
                root.close()
    acc["seconds"]["normal"] += time.perf_counter() - started
    acc["points"] += 1


def true_timer_payload(truth_state: dict[str, Any], seat: str, snapshot: Any) -> dict[str, Any]:
    tview = state_view(truth_state)
    payload: dict[str, dict[str, dict[str, int]]] = {}
    for key, side_id in (("our", seat), ("opponent", "p2" if seat == "p1" else "p1")):
        by_name = {to_id(m["name"]): m for m in tview["sides"][side_id]["mons"]}
        snap_side = snapshot.our_side if key == "our" else snapshot.opponent_side
        for sm in snap_side.pokemon:
            tm = by_name.get(to_id(sm.name))
            if tm is None or tm["fainted"]:
                continue
            entry: dict[str, int] = {}
            if key == "opponent":
                entry["hp"] = int(tm["hp"])  # oracle only: the foe's true absolute HP
            if tm["status"] == "slp" and tm["status_state"].get("time") is not None:
                entry["sleepTime"] = int(tm["status_state"]["time"])
            conf = tm["volatiles"].get("confusion")
            if conf is not None and conf.get("time") is not None:
                entry["confusionTime"] = int(conf["time"])
            if entry:
                payload.setdefault(key, {})[sm.species_id] = entry
    return payload


def oracle_hypothesis(
    truth_state: dict[str, Any], seat: str, snapshot: Any, species_data: dict[str, Any]
) -> MirrorHypothesis:
    """The mirror hypothesis that carries the TRUE hidden facts about the foe."""

    from vgc.stats import STAT_IDS

    foe = "p2" if seat == "p1" else "p1"
    side = next(s for s in truth_state["sides"] if s["id"] == foe)
    spreads: dict[str, tuple[dict[str, int], str]] = {}
    sets: dict[str, dict[str, object]] = {}
    brought: list[str] = []
    for p in side["pokemon"]:
        s = p["set"]
        base = _species_base(to_id(s["species"]), species_data)
        evs = s.get("evs") or {}
        spreads[base] = (
            {stat: int(evs.get(stat, 0) or 0) for stat in STAT_IDS},
            to_id(s.get("nature")) or "serious",
        )
        sets[base] = {
            "moves": [to_id(mv) for mv in s.get("moves") or []],
            "item": to_id(s.get("item")) or None,
            "ability": to_id(s.get("ability")) or None,
            "gender": p.get("gender") or "N",
            "source": "oracle",
        }
        brought.append(base)
    timers = HiddenStateHypothesis(
        weight=1.0, payload=true_timer_payload(truth_state, seat, snapshot)
    )
    return MirrorHypothesis(
        weight=1.0, spreads=spreads, timers=timers, sets=sets, brought=tuple(brought)
    )


def _slot_kind(order: Any) -> list[str]:
    from vgc.rl.exact_search import _order_plan

    return [kind for kind, _mid, _target in _order_plan(order)]


def pick_order(orders: list[Any], want: str, rng: random.Random) -> Any:
    from vgc.rl.exact_search import _PROTECT_FAMILY, _order_plan

    def matches(order: Any) -> bool:
        plan = _order_plan(order)
        if want == "protect":
            return all(kind == "move" and mid in _PROTECT_FAMILY for kind, mid, _t in plan)
        if want == "switch":
            return any(kind == "switch" for kind, _mid, _t in plan)
        if want == "mega":
            return bool(
                getattr(order.first_order, "mega", False)
                or getattr(order.second_order, "mega", False)
            )
        if want == "attack":
            return all(kind == "move" and mid not in _PROTECT_FAMILY for kind, mid, _t in plan)
        return True

    pool = [o for o in orders if matches(o)] or orders
    return rng.choice(pool)


CHOICE_PLAN = ("attack", "protect", "switch", "mega", "attack", "any")


def line_tag(line: str) -> str:
    split = line.split("|")
    tag = split[1] if len(split) > 1 else "?"
    return tag


def first_difference(
    a: list[str], b: list[str]
) -> tuple[int, str | None, str | None] | None:
    for index in range(max(len(a), len(b))):
        left = a[index] if index < len(a) else None
        right = b[index] if index < len(b) else None
        if left != right:
            return index, left, right
    return None


def _clean_lines(lines: list[str], name_map: dict[str, str]) -> list[str]:
    out = []
    for line in lines:
        split = line.split("|")
        tag = split[1] if len(split) > 1 else ""
        if tag in ("t:", "request", "") or line.startswith("|t:|"):
            continue
        out.append(canonical_line(line, name_map))
    return out


def audit_oracle_point(
    ctx: Ctx, d: DirectBattle, seat: str, memory: BattleMemory, team: str,
    acc: dict[str, Any], meta: dict[str, Any], episodes: set[tuple], rng: random.Random,
) -> None:
    """Oracle equivalence: true hidden info injected, same choices + seed on both roots."""

    worker = ctx.worker
    bt = d.battles[seat]
    foe_seat = "p2" if seat == "p1" else "p1"
    truth_state = ctx.dump(worker, d.battle_id)
    tview = state_view(truth_state)
    if _illusion_up(tview):
        return
    memory.observe_battle(bt)
    setattr(bt, "_vgc_battle_memory", memory)
    cfg = ctx.config
    jcfg = with_exact_timers(judge_config(cfg, 6), bt, cfg)
    mirror = ctx.mirror(d.battle_id, seat, team, jcfg)
    started = time.perf_counter()
    name_map = name_map_for(truth_state, ctx.species)
    with public_view_only(bt):
        snapshot = snapshot_battle(bt, opponent_mega_unknown=cfg.exact_mirror_opponent_mega)
        hypothesis = oracle_hypothesis(truth_state, seat, snapshot, ctx.species)
        root = mirror.build(bt, hypothesis)
    findings = Findings()
    try:
        mview = state_view(ctx.dump(worker, root.battle_id))
        compare_views(findings, tview, mview, seat, snapshot, ctx.species)
        # strict: with the truth injected, a FOE's hidden fields must match too
        pairing_pairs = [
            pair
            for pair in pair_mons(tview["sides"][foe_seat], mview["sides"][foe_seat], ctx.species)[0]
        ]
        for t, m in pairing_pairs:
            if t["illusion"] or m["illusion"]:
                continue
            for f in ("hp", "maxhp", "stored_stats", "base_stored_stats", "ability", "item",
                      "base_ability", "weighthg"):  # fmt: skip
                if t[f] != m[f]:
                    for path, a, b in leaf_diffs(t[f], m[f], f):
                        findings.add("oracle_hidden", path, f"foe:{t['name']}", a, b)
            tmv = sorted(s["id"] for s in t["move_slots"])
            mmv = sorted(s["id"] for s in m["move_slots"])
            if tmv != mmv:
                findings.add("oracle_hidden", "move_slots.ids", f"foe:{t['name']}", tmv, mmv)
        for req_seat in SIDES:
            real = d.battles[req_seat].last_request
            mirror_req = request_from_lines(root.last_lines[req_seat])
            for path, a, b in leaf_diffs(
                normalize_request(real, name_map, ctx.species),
                normalize_request(mirror_req, name_map, ctx.species),
                "request",
            ):
                findings.add("request", path, f"{req_seat}", a, b)
        _reclassify_timers(findings.items)
        record(acc, "oracle_state", findings.items, meta, episodes)
        meta = {
            **meta,
            "root_gap": any(
                classify_group(_group("oracle_state", f))[0] == "gap" for f in findings.items
            ),
        }

        # --- same choices, same seed ---------------------------------------------------
        orders = {s: enumerate_joint_orders(d.battles[s]) for s in d.sides_to_move()}
        if set(orders) != set(SIDES) or not all(orders.values()):
            return
        for pair_index in range(ctx.job["pairs"]):
            want = CHOICE_PLAN[(pair_index + rng.randrange(len(CHOICE_PLAN))) % len(CHOICE_PLAN)]
            choices = {
                seat: pick_order(orders[seat], want, rng),
                foe_seat: pick_order(orders[foe_seat], "any", rng),
            }
            messages = {s: choice_string(o.message) for s, o in choices.items()}
            seed = [rng.randrange(1, 2**31) for _ in range(4)]
            acc["oracle_pairs"] += 1
            pair_meta = {**meta, "choices": messages}
            outcome = {}
            for name, source in (("truth", d), ("mirror", root)):
                clone = source.clone(ctx.clone_id(name[0]), seed=seed)
                try:
                    result = clone.step(
                        {s: m for s, m in messages.items() if s in clone.sides_to_move()}
                    )
                    outcome[name] = ("ok", clone, result)
                except InvalidChoice as exc:
                    outcome[name] = ("invalid", clone, str(exc)[:200])
                except Exception as exc:  # noqa: BLE001 - recorded
                    outcome[name] = ("crash", clone, f"{type(exc).__name__}: {str(exc)[:200]}")
            try:
                ok, unexplained_in_step = _compare_outcomes(
                    ctx, acc, outcome, pair_meta, episodes, seat, snapshot, name_map
                )
            finally:
                for _kind, clone, _payload in outcome.values():
                    clone.close()
            if ok:
                acc["oracle_pass"] += 1
            elif meta.get("root_gap") and not unexplained_in_step:
                acc["oracle_gap"] += 1
            else:
                acc["oracle_fail"] += 1
    finally:
        root.close()
        acc["seconds"]["oracle"] += time.perf_counter() - started
        acc["oracle_points"] += 1


def _compare_outcomes(
    ctx: Ctx, acc: dict[str, Any], outcome: dict[str, tuple], meta: dict[str, Any],
    episodes: set[tuple], seat: str, snapshot: Any, name_map: dict[str, str],
) -> tuple[bool, bool]:
    """(identical?, unexplained difference?)"""

    t_kind, t_clone, t_res = outcome["truth"]
    m_kind, m_clone, m_res = outcome["mirror"]
    findings = Findings()
    if t_kind != "ok":
        # the true battle rejected its own legal order: not a mirror defect
        findings.add("harness", f"truth_{t_kind}", "-", t_res, None)
        record(acc, "oracle_step", findings.items, meta, episodes)
        return True, False
    if m_kind != "ok":
        text = str(m_res)
        kind = (
            "struggle" if "matching struggle" in text
            else "no_such_move" if "doesn't have a move matching" in text
            else "needs_target" if "needs a target" in text
            else "other"
        )  # fmt: skip
        findings.add("legality", f"mirror_{m_kind}:{kind}", "-", "ok", m_res)
        record(acc, "oracle_step", findings.items, meta, episodes)
        return False, True
    ok = True
    t_name_map = name_map
    for s in SIDES:
        t_lines = _clean_lines(t_res.lines[s], t_name_map)
        m_lines = _clean_lines(m_res.lines[s], t_name_map)
        diff = first_difference(t_lines, m_lines)
        if diff is not None:
            ok = False
            index, left, right = diff
            ref = left or right or ""
            tag = line_tag(ref)
            move = ""
            for line in reversed(t_lines[: index + 1] if left else m_lines[: index + 1]):
                if line.startswith("|move|"):
                    move = to_id(line.split("|")[3])
                    break
            findings.add(
                "protocol", f"{tag}" + (f"@{move}" if move else ""), s, left, right,
                index=index, context=(t_lines[max(0, index - 3): index + 1]),
            )  # fmt: skip
        t_req = normalize_request(request_from_lines(t_res.lines[s]), name_map, ctx.species)
        m_req = normalize_request(request_from_lines(m_res.lines[s]), name_map, ctx.species)
        if diff is None:
            for path, a, b in leaf_diffs(t_req, m_req, "request"):
                ok = False
                findings.add("request_after", path, s, a, b)
    if (t_res.ended, t_res.winner is None) != (m_res.ended, m_res.winner is None):
        ok = False
        findings.add("protocol", "ended", "-", t_res.ended, m_res.ended)
    if t_res.request_state != m_res.request_state:
        ok = False
        findings.add("protocol", "request_state", "-", t_res.request_state, m_res.request_state)
    if ok:
        # identical protocol: the full next states must also be identical
        tview = state_view(ctx.dump(ctx.worker, t_clone.battle_id))
        mview = state_view(ctx.dump(ctx.worker, m_clone.battle_id))
        post = Findings()
        compare_views(post, tview, mview, seat, snapshot, ctx.species)
        for s in ("p1", "p2"):  # hidden fields of both sides are true on the oracle root
            for t, m in pair_mons(tview["sides"][s], mview["sides"][s], ctx.species)[0]:
                for f in ("hp", "maxhp", "stored_stats", "ability", "item"):
                    if t[f] != m[f] and not (t["illusion"] or m["illusion"]):
                        for path, a, b in leaf_diffs(t[f], m[f], f):
                            post.add("oracle_hidden", path, f"{s}:{t['name']}", a, b)
        _reclassify_timers(post.items)
        post_items = [{**f, "category": "state_after_" + f["category"]} for f in post.items]
        findings.items.extend(post_items)
        # next-state differences that are documented gaps / noise do not fail the pair
        if any(classify_group(_group("oracle_step", f))[0] == "unexplained" for f in post_items):
            ok = False
    mode = "oracle_step_gap" if meta.get("root_gap") else "oracle_step"
    unexplained = any(
        classify_group(_group(mode, f))[0] == "unexplained" for f in findings.items
    )
    record(acc, mode, findings.items, meta, episodes)
    return ok, unexplained


# --- the audited battle -------------------------------------------------------------------


def audit_battle(
    ctx: Ctx, battle_id: str, names: dict[str, str], teams: dict[str, str], seed: list[int],
    acc: dict[str, Any], rng: random.Random, *, trace: bool = False,
) -> None:
    job = ctx.job
    worker = ctx.worker
    agents = {s: make_direct_agent(names[s], teams[s]) for s in SIDES}
    d = DirectBattle.start(worker, battle_id, teams["p1"], teams["p2"], seed=seed)
    memories = {s: BattleMemory(battle_tag=battle_id) for s in SIDES}
    episodes: set[tuple] = set()
    acc["battles"] += 1
    for s in SIDES:
        acc["agents"][names[s]] += 1
    recent: dict[str, list[str]] = {s: [] for s in SIDES}
    # Ally Switch (`|swap|`): poke-env then loses which Pokemon holds which slot, which the
    # parsing audit lists as a limitation; the rest of such a battle is not audited.
    swapped = {"seen": False}

    def feed(lines: dict[str, list[str]]) -> None:
        for s in SIDES:
            agents[s].observe(battle_id, lines[s])
            memories[s].observe_protocol([ln.split("|") for ln in lines[s]])
            recent[s].extend(ln for ln in lines[s] if not ln.startswith("|request"))
            recent[s][:] = recent[s][-400:]
            for ln in lines[s]:
                split = ln.split("|")
                if len(split) > 1 and split[1] == "swap":
                    swapped["seen"] = True
                if len(split) > 3 and split[1] == "move" and split[2].startswith(
                    "p1" if s == "p1" else "p2"
                ):
                    acc["coverage"][f"move:{to_id(split[3])}"] += 1

    feed(d.last_lines)
    step = 0
    # Illusion (Zoroark) shows a wrong identity by design: the whole battle is outside the audit.
    if any(
        mon["base_ability"] == "illusion"
        for side in state_view(ctx.dump(worker, battle_id))["sides"].values()
        for mon in side["mons"]
    ):
        acc["coverage"]["battles_skipped_illusion"] += 1
        swapped["seen"] = True
    try:
        while not d.ended and step < MAX_STEPS:
            to_move = d.sides_to_move()
            bt_ok = {s: (not d.battles[s].teampreview) for s in to_move}
            audited = [
                s
                for s in to_move
                if bt_ok[s]
                and not any(d.battles[s].force_switch or [False])
                and d.request_state == "move"
            ]
            if swapped["seen"]:
                acc["coverage"]["battles_skipped_after_ally_switch"] += 0
            elif audited and d.battles["p1"].turn >= 1 and rng.random() < job["point_rate"]:
                for seat in audited:
                    meta = {
                        "battle": battle_id, "seed": seed, "seat": seat, "step": step,
                        "turn": d.battles[seat].turn, "lines": list(recent[seat]),
                    }  # fmt: skip
                    for mode in ("normal", "oracle"):
                        if mode == "oracle" and (
                            not job["oracle"] or rng.random() >= job["oracle_rate"]
                        ):
                            continue
                        if mode == "normal" and not job["normal"]:
                            continue
                        try:
                            if mode == "normal":
                                audit_normal_point(
                                    ctx, d, seat, memories[seat], teams[seat], acc, meta, episodes
                                )
                            else:
                                audit_oracle_point(
                                    ctx, d, seat, memories[seat], teams[seat], acc, meta,
                                    episodes, rng,
                                )
                        except Exception as exc:  # noqa: BLE001 - the audit reports crashes
                            f = Findings()
                            f.add("crash", type(exc).__name__, seat, None, str(exc)[:300])
                            record(acc, mode, f.items, meta, episodes)
                            if len(acc["errors"]) < 12:
                                acc["errors"].append(
                                    {
                                        "battle": battle_id, "mode": mode, "turn": meta["turn"],
                                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                                        "trace": traceback.format_exc()[-1800:],
                                    }
                                )  # fmt: skip
            for side in to_move:
                d.battles[side]._vgc_direct_root = d
                d.battles[side]._vgc_direct_side = side
            choices = {side: agents[side].choose(d.battles[side]) for side in to_move}
            result = d.step(choices)
            step += 1
            acc["steps"] += 1
            feed(result.lines)
        acc["winners"][d.winner or "none"] += 1
    finally:
        d.close()


def make_config(job: dict[str, Any]) -> PolicyConfig:
    overrides = dict(job.get("config") or {})
    return replace(PolicyConfig(exact_judge_live=True), **overrides)


def worker_main(job: dict[str, Any]) -> dict[str, Any]:
    global TRACE
    TRACE = job.get("trace_pattern") or ""
    acc = _new_acc()
    builder = TeamBuilder()
    pool = job["pool_teams"]
    config = make_config(job)
    with SimWorker() as worker:
        ctx = Ctx(worker, config, job)
        for index in job["indices"]:
            rng = random.Random(job["seed"] * 1_000_003 + index)
            random.seed(job["seed"] * 7_919 + index)
            try:
                import numpy

                numpy.random.seed((job["seed"] * 7_919 + index) % 2**32)
            except ImportError:
                pass
            if pool and rng.random() < job["pool_fraction"]:
                teams = {s: rng.choice(pool) for s in SIDES}
            else:
                teams = {s: builder.pack(rng) for s in SIDES}
            names = {s: rng.choice(job["agents"]) for s in SIDES}
            seed = [rng.randrange(1, 2**31) for _ in range(4)]
            battle_id = f"mf{index}"
            try:
                audit_battle(ctx, battle_id, names, teams, seed, acc, rng, trace=job.get("trace", False))
            except Exception as exc:  # noqa: BLE001 - keep auditing
                acc["errors"].append(
                    {
                        "battle": battle_id, "mode": "driver",
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "trace": traceback.format_exc()[-1800:],
                    }
                )  # fmt: skip
                with contextlib.suppress(Exception):
                    worker.request({"cmd": "close", "id": battle_id})
    return acc


# --- classification -----------------------------------------------------------------------
#
# (regex on the group name, kind, reason). kind: "hidden" = information no player has (a
# guess, not a defect); "gap" = a real, documented divergence with its reason; "noise" = a
# field with no effect on any decision. A group matching none of these is UNEXPLAINED.

CLASSIFICATION: list[tuple[str, str, str]] = [
    # --- hidden information: guesses, not defects -------------------------------------------
    (r"^normal/bring:unmatched", "hidden", "which Pokemon the foe brought is a guess"),
    (r"^normal/timer:", "hidden", "sleep/confusion timers are branched (support is measured)"),
    (r"^normal/volatile_missing:choicelock$", "hidden",
     "a foe's Choice lock is invisible until its Choice item is revealed"),
    (r"^normal/volatile_missing:unburden$", "hidden",
     "a foe's Unburden is invisible until its item loss is seen"),
    (r"^normal/bookkeeping:trapped$", "hidden", "trapping is only learned by trying to switch"),
    (r"^normal/volatile_missing:(trapper|trapped)$", "hidden",
     "Mean Look / Block / Shadow Tag trapping is hidden"),
    # --- no decision effect --------------------------------------------------------------
    (r"sourceSlot$", "noise",
     "who applied a volatile/condition; read only by Leech Seed and the trap family, which "
     "the patch fills explicitly"),
    (r"side_value:stealthrock\.layers", "noise", "Stealth Rock has no layers"),
    (r"bookkeeping:last_move(_used)?$", "noise", "per-Pokemon last move; the patch sets it"),
    (r"oracle_hidden:(ability|base_ability|item|stored_stats|base_stored_stats|move_slots)", "gap",
     "Skill Swap / Entrainment / Trick results on a foe are not visible to the snapshot"),
    # --- documented gaps (public information the rebuild does not carry) -------------------
    (r"bookkeeping:times_attacked", "gap", "Rage Fist power (Annihilape/Primeape only)"),
    (r"bookkeeping:move_last_turn_failed", "gap",
     "Stomping Tantrum / Temper Flare doubling after a failed move"),
    (r"bookkeeping:ate_berry|public:last_item", "gap", "Belch / Recycle / Pickup / Cud Chew"),
    (r"volatile_missing:metronome", "gap", "Metronome item's consecutive-use counter"),
    (r"volatile_value:substitute\.hp", "gap", "Substitute HP is rebuilt at 1/4 max (known)"),
    (r"side_value:slot_conditions", "gap", "pending Future Sight / Doom Desire / Healing Wish"),
    (r"volatile_value:(taunt|encore|disable|yawn|partiallytrapped)\.duration", "gap",
     "remaining turns are off by one when the target had already moved (not in the protocol)"),
    (r"volatile_value:\w+\.targetLoc|volatile_missing:(skyattack|solarbeam|electroshot|"
     r"lockedmove|allyswitch)", "gap",
     "target slot of a charging / locked move is only known to its user"),
    (r"volatile_value:dragoncheer\.hasDragonType", "gap", "Dragon Cheer crit stage type flag"),
    (r"can_mega_evo|request:request\.active\.canMegaEvo", "gap",
     "a recharging/locked Pokemon is offered no Mega in the request; the mirror follows the request"),
    (r"request:request\.active\.(trapped|moves\.disabled)", "gap",
     "request flags for hidden/locked states (see above)"),
    (r"request:request\.side\.pokemon\.(ability|baseAbility|item|stats|details|condition|"
     r"ident|_key|active|moves)", "gap",
     "Skill Swap/Trace/Illusion/Ally Switch residue the parse audit lists as limitations"),
    (r"^(normal|oracle_state)/(field_value|side_value):\w+\.duration", "gap",
     "a screen/weather duration shifted by an unrevealed extender (Light Clay, Damp Rock)"),
    (r"volatile_missing:(mustrecharge|trapped|trapper|choicelock)|bookkeeping:trapped", "gap",
     "recharge/trap/lock volatiles poke-env drops or that only the holder's player can see"),
    (r"request:request\.active\.moves\.(id|move|pp|maxpp|target)", "gap",
     "a locked/recharging slot offers one move (recharge / the locked move); follows the above"),
    (r"foe_ability_known", "gap", "ability swapped (Skill Swap / Trace) after the reveal"),
    (r"^normal/public:(species|base_species|stored_stats|move_slots\.ids)", "gap",
     "Transform / Imposter copy a forme, stats and moves poke-env keeps under the old species"),
    (r"public:move_slots\.pp", "gap", "PP off by one (Pressure / Spite / Sleep Talk bookkeeping)"),
    (r"bookkeeping:active_move_actions", "gap",
     "move-action count after an interrupted/self-switch turn (only zero vs non-zero is read)"),
    (r"request:request\.(active\.maybe(Disabled|Locked)|update)", "gap",
     "request metadata for hidden-disable flows; no effect on legality"),
    (r"timer:frz\.", "hidden", "the freeze thaw timer is rolled privately (not branched yet)"),
    (r"oracle_hidden:weighthg", "gap", "weight after Transform / Autotomize is not tracked"),
    (r"volatile_(value|extra):lockedmove|volatile_value:lockedmove\.trueDuration", "hidden",
     "how many turns an Outrage-family lock has left is rolled privately"),
    (r"timer:slp\.status_state\.startTime", "noise", "display-only sleep roll"),
    (r"side_value:stickyweb\.layers", "noise", "Sticky Web has one layer"),
    (r"volatile_missing:(meteorbeam|solarblade|skullbash|razorwind|freezeshock|iceburn|geomancy)",
     "gap", "charge-turn marker of a two-turn move (the lock itself is restored)"),
    (r"public:tox\.status_state\.stage", "gap", "Toxic stage is one behind after a switch-in turn"),
    (r"volatile_value:(disable\.move|healblock\.duration|stockpile\.\w+)", "gap",
     "Disable target / Heal Block turns / Stockpile defence tracking"),
    (r"bookkeeping:maybe_trapped|request:request\.active\.maybeTrapped", "hidden",
     "hidden-trap flags only exist for a foe that might have a trapping ability"),
    (r"^oracle_state/public:(species|types|stored_stats|move_slots\.ids)", "gap",
     "Transform copies / Forecast & Flower Gift formes are not carried"),
    (r"legality:mirror_invalid:struggle", "gap",
     "PP drifts by one in either direction (Pressure, Spite, Sleep Talk), so Struggle differs"),
    (r"legality:mirror_invalid:(no_such_move|needs_target)", "gap",
     "a Transformed/Mimicking Pokemon's copied moves are not in the rebuilt set"),
    (r"normal/volatile_extra:choicelock|normal/public:weighthg", "gap",
     "Choice lock kept after the item changed hands / weight after Transform"),
    (r"^oracle_step/protocol:-ability", "gap", "Trace copies a foe ability the rebuild guesses"),
    (r"^oracle_state/oracle_hidden:hp", "gap", "foe absolute HP after Substitute / Transform"),
    (r"^oracle_step_gap/", "gap", "downstream of a documented gap in the root state"),
]


def classify_group(group: str) -> tuple[str, str]:
    group = group.replace(":state_after_", ":").replace("/state_after_", "/")
    for pattern, kind, reason in CLASSIFICATION:
        if re.search(pattern, group):
            return kind, reason
    return "unexplained", ""


def unexplained_groups(acc: dict[str, Any]) -> list[str]:
    return sorted(g for g in acc["episodes"] if classify_group(g)[0] == "unexplained")


def report_text(acc: dict[str, Any], elapsed: float) -> str:
    lines = [
        f"battles={acc['battles']} steps={acc['steps']} points={acc['points']} "
        f"oracle_points={acc['oracle_points']} oracle_pairs={acc['oracle_pairs']} "
        f"pass={acc['oracle_pass']} gap-explained={acc['oracle_gap']} fail={acc['oracle_fail']} errors={len(acc['errors'])} "
        f"elapsed={elapsed:.0f}s",
        f"agents={dict(acc['agents'])}",
    ]
    for kind, title in (
        ("unexplained", "DEFECTS (unexplained)"),
        ("gap", "DOCUMENTED GAPS"),
        ("hidden", "HIDDEN / GUESS"),
        ("noise", "NO DECISION EFFECT"),
    ):
        rows = [
            (g, n)
            for g, n in sorted(acc["episodes"].items(), key=lambda kv: -kv[1])
            if classify_group(g)[0] == kind
        ]
        lines += ["", f"== {title}: {len(rows)} groups, {sum(n for _, n in rows)} episodes"]
        for g, n in rows:
            lines.append(f"  {g:70s} ep={n:5d} obs={acc['observations'][g]:6d}")
    lines += ["", "== GUESS ACCURACY (hidden facts, not defects)"]
    for name, (ok, total) in sorted(acc["guess"].items()):
        lines.append(f"  {name:55s} {ok:6d}/{total:6d} = {ok / max(1, total):.3f}")
    lines += ["", "== HIDDEN-TIMER BRANCH SUPPORT"]
    for name, (ok, total) in sorted(acc["timer_support"].items()):
        lines.append(f"  {name:55s} {ok:6d}/{total:6d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--battles", type=int, default=60)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--agents", default="random,random,maxpower,heuristic,vgc_myopic")
    parser.add_argument("--pool-fraction", type=float, default=0.3)
    parser.add_argument("--pool-limit", type=int, default=120)
    parser.add_argument("--point-rate", type=float, default=0.4)
    parser.add_argument("--oracle-rate", type=float, default=0.6)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--no-normal", action="store_true")
    parser.add_argument("--config", default="", help="comma list FIELD=VALUE PolicyConfig overrides")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fail-on-unexplained", action="store_true")
    parser.add_argument("--examples", type=int, default=0, help="print N examples per defect group")
    parser.add_argument("--only", type=int, default=None, help="run only this battle index")
    parser.add_argument("--trace", default="", help="regex on group names to print as found")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    for part in [p for p in args.config.split(",") if p]:
        key, _, value = part.partition("=")
        overrides[key] = json.loads(value)
    pool_teams = load_pool_teams(list(DEFAULT_MANIFESTS), args.pool_limit)
    base = {
        "seed": args.seed,
        "agents": args.agents.split(","),
        "pool_teams": pool_teams,
        "pool_fraction": args.pool_fraction if pool_teams else 0.0,
        "point_rate": args.point_rate,
        "oracle_rate": args.oracle_rate,
        "pairs": args.pairs,
        "oracle": not args.no_oracle,
        "normal": not args.no_normal,
        "config": overrides,
        "trace_pattern": args.trace,
    }
    if args.only is not None:
        jobs = [{**base, "indices": [args.only]}]
        workers = 1
    else:
        workers = max(1, min(args.workers, args.battles))
        jobs = [
            {**base, "indices": list(range(w, args.battles, workers))} for w in range(workers)
        ]
    started = time.time()
    total = _new_acc()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for part in pool.map(worker_main, jobs):
            merge_acc(total, part)
    elapsed = time.time() - started
    print(report_text(total, elapsed))
    for err in total["errors"][:5]:
        print("\nERROR", err["battle"], err["mode"], err["error"], "\n", err["trace"])
    if args.examples:
        for group in unexplained_groups(total):
            print(f"\n--- {group}")
            for ex in total["examples"][group][: args.examples]:
                print(json.dumps(ex, default=str)[:1500])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        "battles": total["battles"],
        "points": total["points"],
        "oracle_points": total["oracle_points"],
        "oracle_pairs": total["oracle_pairs"],
        "oracle_pass": total["oracle_pass"],
        "oracle_fail": total["oracle_fail"],
        "oracle_gap": total["oracle_gap"],
        "elapsed_seconds": round(elapsed, 1),
        "seed": args.seed,
        "episodes": dict(total["episodes"]),
        "observations": dict(total["observations"]),
        "classification": {g: classify_group(g) for g in total["episodes"]},
        "unexplained": unexplained_groups(total),
        "guess": {k: v for k, v in total["guess"].items()},
        "timer_support": {k: v for k, v in total["timer_support"].items()},
        "examples": dict(total["examples"]),
        "coverage": dict(total["coverage"].most_common()),
        "errors": total["errors"],
        "seconds": dict(total["seconds"]),
    }
    args.output.write_text(json.dumps(serial, indent=1, default=str))
    print(f"\nwrote {args.output}")
    if total["errors"]:
        return 1
    if args.fail_on_unexplained and unexplained_groups(total):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

