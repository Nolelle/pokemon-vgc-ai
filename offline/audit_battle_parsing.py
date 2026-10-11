"""Differential audit: what the bot BELIEVES about a battle vs what Showdown KNOWS.

The bot reads Showdown through poke-env 0.15 (plus `vgc.poke_env_compat` repairs and our own
`BattleMemory` / `vgc.condition_clock` layers). poke-env has real parsing defects -- e.g.
`-copyboost` was applied backwards, so the bot believed a +evasion Slowbro was unboosted.
Nothing in the test suite compares the parsed state with the simulator's truth, so a defect
like that can sit unnoticed for as long as the message is rare.

This script plays many local battles through `vgc.rl.env.DirectBattle` and, after EVERY
simulator step, compares each side's bot-visible state (`vgc.mechanics_state.snapshot_battle`
on BOTH perspectives' poke-env battle objects, with a `BattleMemory` fed the same protocol,
exactly like a live decision) with the worker's `dump` of Showdown's own state, restricted to
what is PUBLICLY observable:

    active pieces BY EXACT SLOT / species forme / types / HP (exact own, percent foe) /
    fainted / status + sleep & toxic counters / all 7 boosts / item (known, consumed, lost)
    / ability / volatiles / protect counter / weather / terrain / pseudo-weather
    / side conditions / remaining durations / revealed foe moves

Mismatches are grouped by (field, kind, detail), counted two ways -- observations (turns it
was wrong) and episodes (distinct battle x perspective x Pokemon x thing that went wrong) --
and attributed to the protocol messages that touched the Pokemon on the step the mismatch
first appeared, so each group points at the message that broke it.

How the answer key is built (and what is exempt):

* The truth is Showdown's own dump. A live Illusion is compared as the other side's client
  sees it: the disguise's name/species/types over the holder's real HP, status, boosts and
  volatiles; only the deliberately deceptive identity (true name, item, ability, moves) is
  exempt, and only while the disguise holds. After it breaks the true Pokemon is compared
  normally.
* "Revealed" abilities / items / moves come from `audit_protocol.ProtocolKnowledge`, which
  shares no code with `vgc.poke_env_compat` or `vgc.battle_memory`.
* There are no group-level exemptions. Each documented cause is a predicate in
  `audit_exemptions.py` evaluated against the specific finding (move / ability / message plus
  the Pokemon's history); a mismatch that does not satisfy one counts as UNEXPLAINED.

    .venv/bin/python offline/audit_battle_parsing.py --battles 200 --workers 8
    .venv/bin/python offline/audit_battle_parsing.py --scenarios-only --coverage-report
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import audit_exemptions as EX
from audit_protocol import ProtocolKnowledge
from vgc.battle_memory import BattleMemory
from vgc.config import REPO_ROOT, RUNS_DIR
from vgc.damage import to_id
from vgc.mechanics_state import snapshot_battle
from vgc.rl.agents import make_direct_agent
from vgc.rl.env import DirectBattle, SimWorker

SIDES = ("p1", "p2")
DEFAULT_OUTPUT = RUNS_DIR / "eval" / "battle_parsing_audit.json"
DEFAULT_MANIFESTS = (
    REPO_ROOT / "data" / "selfplay" / "archetype_pool_150" / "manifest.json",
    REPO_ROOT / "data" / "selfplay" / "mc_sheet_pool_v2" / "manifest.json",
)
MAX_STEPS = 400
MAX_EXAMPLES = 3

# Moves whose protocol messages move boosts, items, abilities, volatiles or the field around.
# Synthetic audit teams are biased toward these; they are all legal in the Champions mod
# (the teams are not validated, but we only pick moves from each species' own learnset).
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
    "focusenergy dragoncheer acidspray screech charm feint partingshot incinerate "
    "burnup ingrain aquaring wish rapidspin defog courtchange mortalspin leechlife "
    "payback bodypress stoneaxe ceaselessedge scald chillingwater rockslide earthquake "
    "closecombat flareblitz dragonclaw tripleaxel icywind electroweb snarl bulldoze"
).split()
GOOD_ITEMS = (
    "sitrusberry focussash leftovers lifeorb rockyhelmet shellbell whiteherb mentalherb "
    "lumberry leppaberry electricseed grassyseed mistyseed psychicseed ejectbutton redcard "
    "airballoon choicescarf lightclay damprock heatrock smoothrock icyrock terrainextender "
    "focusband expertbelt muscleband wiseglasses quickclaw widelens zoomlens scopelens "
    "oranberry persimberry chestoberry cheriberry rawstberry pechaberry aspearberry "
    "chartiberry babiriberry occaberry yacheberry kasibberry shucaberry passhoberry "
    "bindingband shedshell ironball kingsrock metronome normalgem"
).split()


# --- synthetic teams --------------------------------------------------------------------


class TeamBuilder:
    def __init__(self) -> None:
        data = REPO_ROOT / "data" / "champions"
        self.species = json.loads((data / "species.json").read_text())
        self.learnsets = json.loads((data / "learnsets.json").read_text())
        self.items = json.loads((data / "items.json").read_text())
        self.moves = json.loads((data / "moves.json").read_text())
        self.interesting = set(INTERESTING_MOVES)
        self.stones: dict[str, str] = {}
        for item_id, item in self.items.items():
            if item.get("megaStone") and item.get("megaEvolves"):
                self.stones.setdefault(to_id(item["megaEvolves"]), item["name"])
        self.base_species = sorted(
            sid
            for sid, s in self.species.items()
            if s.get("isNonstandard") is None
            and not s.get("isMega")
            and not s.get("battleOnly")
            and s.get("forme") not in ("Gmax",)
        )

    def _learnable(self, sid: str) -> list[str]:
        for key in (sid, to_id(self.species[sid].get("baseSpecies"))):
            ls = self.learnsets.get(key)
            if ls:
                return [m for m in ls if m in self.moves]
        return []

    def pack(self, rng: random.Random) -> str:
        chosen: list[str] = []
        used_bases: set[str] = set()
        while len(chosen) < 6:
            sid = rng.choice(self.base_species)
            base = to_id(self.species[sid].get("baseSpecies"))
            if base not in used_bases:  # Species Clause, and nicknames would collide
                used_bases.add(base)
                chosen.append(sid)
        members = []
        stoned = 0
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
            item = rng.choice(GOOD_ITEMS)
            stone_name = self.stones.get(sid)
            if stone_name and stoned < 2 and rng.random() < 0.6:
                item = stone_name
                stoned += 1
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


# --- extracting Showdown's truth ---------------------------------------------------------


def _hp_percent(hp: int, maxhp: int) -> int:
    if hp <= 0:
        return 0
    return max(1, hp * 100 // maxhp)  # Showdown (Champions): floor(100 * hp / maxhp) || 1


_ILLUSION_REF = re.compile(r"\[Pokemon:p[1-4]([a-f])\]")


def _truth_mon(p: dict[str, Any]) -> dict[str, Any]:
    species = to_id(str(p.get("species", "")).replace("[Species:", "").rstrip("]")) or to_id(
        p.get("details", "").split(",")[0]
    )
    return {
        "name": p["set"]["name"],
        "species": species,
        "hp": int(p["hp"]),
        "maxhp": int(p["maxhp"]),
        "status": p.get("status") or "",
        "status_state": p.get("statusState") or {},
        "boosts": dict(p.get("boosts") or {}),
        "volatiles": {to_id(k): v for k, v in (p.get("volatiles") or {}).items()},
        "item": to_id(p.get("item")),
        "last_item": to_id(p.get("lastItem")),
        "ability": to_id(p.get("ability")),
        "base_ability": to_id(p.get("baseAbility")),
        "types": [to_id(t) for t in (p.get("types") or [])],
        "base_types": [to_id(t) for t in (p.get("baseTypes") or [])],
        "added_type": to_id(p.get("addedType")),
        "active": bool(p.get("isActive")),
        "position": p.get("position"),
        "fainted": bool(p.get("fainted")),
        "had_item": bool((p.get("set") or {}).get("item")),
        "set_item": to_id((p.get("set") or {}).get("item")),
        "transformed": bool(p.get("transformed")),
        "illusion": bool(p.get("illusion")),
        "illusion_ref": str(p.get("illusion") or ""),
        "illusion_name": None,  # filled in by truth_view once the whole side is known
        "moves": [to_id(m.get("id") or m.get("move")) for m in (p.get("moveSlots") or [])],
        "base_moves": [to_id(m) for m in (p.get("set", {}).get("moves") or [])],
        "disguised_as": False,
        "true_name": None,
    }


def truth_view(state: dict[str, Any]) -> dict[str, Any]:
    sides = {}
    for side in state["sides"]:
        raw = side["pokemon"]
        mons = {m["name"]: m for m in (_truth_mon(p) for p in raw)}
        names = [p["set"]["name"] for p in raw]
        for tm in mons.values():
            ref = _ILLUSION_REF.match(tm["illusion_ref"])
            # (Showdown leaves `illusion` set on a benched Zoroark; it only disguises while active)
            if ref and tm["active"] and not tm["fainted"]:
                index = ord(ref.group(1)) - ord("a")
                if 0 <= index < len(names):
                    tm["illusion_name"] = names[index]
        # What the OTHER side's client is shown. A live Illusion shows the disguise's identity
        # (name, species, types) over the holder's HP, status, boosts and volatiles; the true
        # holder is not visible by name, and the disguise Pokemon's own state is replaced.
        visible = dict(mons)
        for name, tm in mons.items():
            if tm["illusion_name"] and tm["illusion_name"] in mons:
                disguise = mons[tm["illusion_name"]]
                view = dict(tm)
                view.update(
                    name=disguise["name"],
                    species=disguise["species"],
                    types=disguise["base_types"] or disguise["types"],
                    added_type="",
                    illusion=False,
                    disguised_as=True,
                    true_name=name,
                    transformed=False,
                )
                if disguise["active"] and not disguise["fainted"]:
                    # The disguise copies an ACTIVE ally (every benched Pokemon had fainted):
                    # two on-field Pokemon now share one protocol ident, which poke-env keys
                    # its team by, so the one entry cannot hold both.
                    view["collision"] = True
                    view["collides_with"] = disguise["name"]
                visible[disguise["name"]] = view
                visible[name] = {**tm, "hidden_identity": True}
        slots_true: list[str | None] = [None, None]
        for tm in mons.values():
            if tm["active"] and not tm["fainted"] and tm["position"] in (0, 1):
                slots_true[tm["position"]] = tm["name"]
        slots: list[str | None] = [None, None]
        for tm in mons.values():
            if tm["active"] and not tm["fainted"] and tm["position"] in (0, 1):
                disguised_as = tm["illusion_name"] if tm["illusion_name"] in mons else None
                slots[tm["position"]] = disguised_as or tm["name"]
        conditions = {}
        for cid, cs in (side.get("sideConditions") or {}).items():
            conditions[to_id(cid)] = {
                "duration": cs.get("duration"),
                "layers": cs.get("layers"),
                "setter": (cs.get("sourceSlot") or "")[:2],
            }
        sides[side["id"]] = {
            "mons": mons,
            "visible": visible,
            "slots": slots,
            "slots_true": slots_true,
            "conditions": conditions,
        }
    field = state["field"]
    return {
        "turn": state["turn"],
        # A forced-switch request arrives after the residual phase but before `|turn|`, and
        # poke-env only clears single-turn effects on `|turn|`: that window is not a decision
        # point for anything but the replacement switch, so volatile checks skip it.
        "mid_turn": state.get("requestState") != "move",
        "sides": sides,
        "weather": to_id(field.get("weather")),
        "weather_duration": (field.get("weatherState") or {}).get("duration"),
        "weather_setter": ((field.get("weatherState") or {}).get("sourceSlot") or "")[:2],
        "terrain_setter": ((field.get("terrainState") or {}).get("sourceSlot") or "")[:2],
        "terrain": to_id(field.get("terrain")),
        "terrain_duration": (field.get("terrainState") or {}).get("duration"),
        "pseudo": {
            to_id(k): v.get("duration") for k, v in (field.get("pseudoWeather") or {}).items()
        },
    }


def illusion_alias(*truths: dict[str, Any] | None) -> dict[tuple[str, str], tuple[str, str]]:
    """(role, disguise name) -> (role, true holder name), over every given truth snapshot."""
    alias: dict[tuple[str, str], tuple[str, str]] = {}
    for truth in truths:
        if not truth:
            continue
        for role, side in truth["sides"].items():
            for tm in side["mons"].values():
                if tm["illusion_name"]:
                    alias[(role, tm["illusion_name"])] = (role, tm["name"])
    return alias


# --- comparing ----------------------------------------------------------------------------

_BASE_DURATION = {
    "sunnyday": 5, "raindance": 5, "sandstorm": 5, "snowscape": 5, "snow": 5, "hail": 5,
    "electricterrain": 5, "grassyterrain": 5, "psychicterrain": 5, "mistyterrain": 5,
    "trickroom": 5, "tailwind": 4, "reflect": 5, "lightscreen": 5, "auroraveil": 5,
    "safeguard": 5, "mist": 5, "luckychant": 5, "gravity": 5, "magicroom": 5,
    "wonderroom": 5, "mudsport": 5, "watersport": 5,
}  # fmt: skip

# Truth volatiles with no public protocol message by design (cannot be a parse defect).
HIDDEN_VOLATILES = frozenset(
    {
        # Chosen or inferred by humans from the move history; Showdown sends nothing.
        "choicelock",
        "lockedmove",
        "lockedmovestate",
        "metronome",
        "allyswitch",
        "stall",
        # Silent in the protocol: the player only learns of them by trying to switch.
        "trapped",
        "trapper",
        # The sim worker rebuilds Unburden from "item lost this stint" (restoreState).
        "unburden",
        # Compared as `preparing` instead (see below).
        "twoturnmove",
        # Doubles / Dynamax internals.
        "dynamax",
        "commanded",
        "commanding",
        "mustrecharge",
        "fakeout",
    }
)

_SPECIES: dict[str, Any] | None = None


def legal_abilities(species_id: str) -> set[str]:
    """Ability ids a species can have in the exported data (empty if unknown)."""
    global _SPECIES
    if _SPECIES is None:
        _SPECIES = json.loads((REPO_ROOT / "data" / "champions" / "species.json").read_text())
    entry = _SPECIES.get(species_id) or {}
    return {to_id(a) for a in (entry.get("abilities") or {}).values()}


class Findings:
    """Collects mismatches and classifies each one at creation against the exemption registry
    (`audit_exemptions`): an exemption only applies when its predicate holds for THIS finding,
    so a documented cause can never excuse an unrelated mismatch of the same field."""

    def __init__(self, base: dict[str, Any]) -> None:
        self.items: list[dict[str, Any]] = []
        self.base = base
        self.bound: dict[str, Any] = {}

    def bind(self, **context: Any) -> None:
        self.bound = context

    def add(
        self,
        field: str,
        kind: str,
        detail: str,
        mon: str,
        *,
        truth: Any,
        belief: Any,
        note: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        finding = {
            "field": field,
            "kind": kind,
            "detail": detail,
            "mon": mon,
            "truth": truth,
            "belief": belief,
            "note": note,
            "extra": extra or {},
        }
        context = {**self.base, **self.bound, "finding": finding}
        finding["exempt"] = EX.classify(finding, context) or ""
        self.items.append(finding)


def _belief_remaining(snapshot_turn: int, effect: Any) -> int | None:
    base = _BASE_DURATION.get(effect.id)
    if base is None or effect.turns is None:
        return None
    return base - (snapshot_turn - effect.turns)


def _duration_finding(
    out: "Findings", detail: str, label: str, truth_dur: int, rem: int, setter: str, foe_role: str
) -> None:
    out.add(
        "duration", "mismatch", detail, label,
        truth=truth_dur, belief=rem,
        extra={"setter": setter, "foe_role": foe_role, "condition": detail},
    )  # fmt: skip


def compare_perspective(
    me: str,
    snapshot: Any,
    truth: dict[str, Any],
    memory: BattleMemory,
    know: ProtocolKnowledge,
    battle: Any,
    step_lines: list[str] | None = None,
    prev_truth: dict[str, Any] | None = None,
    collided: dict[str, set[str]] | None = None,
    deceived: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    foe_role = "p2" if me == "p1" else "p1"
    out = Findings(
        {
            "me": me,
            "foe_role": foe_role,
            "truth": truth,
            "prev_truth": prev_truth,
            "collided": collided or {},
            "deceived": deceived or {},
            "know": know,
            "memory": memory,
            "lines": step_lines or [],
            "snapshot": snapshot,
        }
    )
    if snapshot.turn != truth["turn"]:
        out.add("turn", "mismatch", "", "-", truth=truth["turn"], belief=snapshot.turn)

    for label, role, side_state, active_list in (
        ("own", me, snapshot.our_side, battle.active_pokemon),
        ("foe", foe_role, snapshot.opponent_side, battle.opponent_active_pokemon),
    ):
        side_truth = truth["sides"][role]
        # What this perspective's client is shown: its own team truthfully, the other side
        # through any live Illusion.
        tmons = side_truth["visible"] if label == "foe" else side_truth["mons"]
        out.bind(label=label, role=role)
        for bmon in side_state.pokemon:
            name = bmon.name
            tm = tmons.get(name)
            if tm is None:
                if label == "own" and not bmon.active:
                    continue  # the 2 of 6 left home: poke-env keeps them, Showdown drops them
                if bmon.revealed or label == "own":
                    out.add(
                        "roster", "unknown_mon", label, name,
                        truth=sorted(tmons), belief=name,
                    )  # fmt: skip
                continue
            if tm.get("hidden_identity"):
                # A live Illusion holder: its true name is deliberately not shown to this
                # perspective, so whatever the client holds under that name is stale by design.
                continue
            if label == "foe" and not bmon.revealed:
                continue  # never shown to this perspective
            _compare_mon(out, label, name, tm, bmon, memory, know, mid_turn=truth["mid_turn"])

        # active slots, compared by exact position (a left/right swap is a defect)
        truth_slots = side_truth["slots"] if label == "foe" else side_truth["slots_true"]
        belief_slots = [m.name if m is not None else None for m in active_list]
        out.bind(label=label, role=role)
        if truth_slots != belief_slots:
            out.add(
                "active_slots", "mismatch", label, "-",
                truth=truth_slots, belief=belief_slots,
            )  # fmt: skip

        # side conditions
        t_cond = {
            k: v
            for k, v in side_truth["conditions"].items()
            if k not in ("wideguard", "quickguard", "matblock", "craftyguard")  # -singleturn
        }
        b_cond = {e.id: e for e in side_state.side_conditions}
        for cid in sorted(set(t_cond) | set(b_cond)):
            if cid in t_cond and cid not in b_cond:
                out.add(
                    "side_condition", "missing", cid, label, truth=t_cond[cid], belief=None
                )  # fmt: skip
            elif cid in b_cond and cid not in t_cond:
                out.add(
                    "side_condition", "stale", cid, label, truth=None,
                    belief=_eff(b_cond[cid]),
                )  # fmt: skip
            else:
                tc, be = t_cond[cid], b_cond[cid]
                if cid in ("spikes", "toxicspikes") and tc["layers"] is not None:
                    layers = be.turns if be.counter_kind == "layers" else be.raw_value
                    if layers != tc["layers"]:
                        out.add(
                            "side_condition", "layers", cid, label,
                            truth=tc["layers"], belief=layers,
                        )  # fmt: skip
                rem = _belief_remaining(snapshot.turn, be)
                if rem is not None and tc["duration"] is not None and rem != tc["duration"]:
                    _duration_finding(
                        out, f"side:{cid}", label, tc["duration"], rem, tc["setter"], foe_role
                    )

    # field: weather / terrain / pseudo-weather
    out.bind(label="field", role="")
    t_weather = truth["weather"]
    b_weather = {e.id: e for e in snapshot.weather}
    b_weather_id = next(iter(b_weather), "")
    if t_weather != b_weather_id:
        out.add(
            "weather", "mismatch", t_weather or "-", "field", truth=t_weather, belief=b_weather_id
        )
    elif t_weather:
        rem = _belief_remaining(snapshot.turn, b_weather[t_weather])
        if (
            rem is not None
            and truth["weather_duration"] is not None
            and rem != truth["weather_duration"]
        ):
            _duration_finding(
                out,
                f"weather:{t_weather}",
                "field",
                truth["weather_duration"],
                rem,
                truth["weather_setter"],
                foe_role,
            )
    b_fields = {e.id: e for e in snapshot.fields}
    t_fields = dict(truth["pseudo"])
    if truth["terrain"]:
        t_fields[truth["terrain"]] = truth["terrain_duration"]
    for fid in sorted(set(t_fields) | set(b_fields)):
        if fid in t_fields and fid not in b_fields:
            out.add("field", "missing", fid, "field", truth=t_fields[fid], belief=None)
        elif fid in b_fields and fid not in t_fields:
            out.add("field", "stale", fid, "field", truth=None, belief=_eff(b_fields[fid]))
        else:
            rem = _belief_remaining(snapshot.turn, b_fields[fid])
            if rem is not None and t_fields[fid] is not None and rem != t_fields[fid]:
                setter = truth["terrain_setter"] if fid == truth["terrain"] else ""
                _duration_finding(
                    out, f"field:{fid}", "field", t_fields[fid], rem, setter, foe_role
                )
    return out.items


def _eff(e: Any) -> dict[str, Any]:
    return {"id": e.id, "turns": e.turns, "kind": e.counter_kind, "raw": e.raw_value}


def _compare_mon(
    out: Findings,
    label: str,
    name: str,
    tm: dict[str, Any],
    bm: Any,
    memory: BattleMemory,
    know: ProtocolKnowledge,
    *,
    mid_turn: bool,
) -> None:
    key = f"{label}:{name}"
    role = out.bound["role"]
    disguised = bool(tm.get("disguised_as"))
    out.bind(label=label, role=role, name=name, tm=tm, bm=bm, disguised=disguised)
    # fainted / bench
    if tm["fainted"] != bm.fainted:
        out.add("fainted", "mismatch", label, key, truth=tm["fainted"], belief=bm.fainted)
    if tm["fainted"]:
        return
    if tm["active"] != bm.active:
        out.add("active_flag", "mismatch", label, key, truth=tm["active"], belief=bm.active)
    # forme
    if not tm["transformed"] and tm["species"] != bm.species_id:
        out.add(
            "species", "mismatch", label, key,
            truth=tm["species"], belief=bm.species_id,
        )  # fmt: skip
    if tm["transformed"] and not bm.transformed:
        out.add("transform", "missing", label, key, truth=True, belief=False)
    t_types = set(tm["types"]) - {"???", ""}
    if tm["added_type"]:
        t_types.add(tm["added_type"])
    b_types = set(bm.types) - {"none", "", "threequestionmarks"}
    roosting = "roost" in tm["volatiles"]
    if t_types != b_types and not (roosting and (t_types - {"flying"}) == (b_types - {"flying"})):
        out.add(
            "types", "mismatch", label, key,
            truth=sorted(t_types), belief=sorted(b_types),
        )  # fmt: skip
    # HP (exact own, percent foe)
    if label == "own":
        if (tm["hp"], tm["maxhp"]) != (bm.current_hp, bm.max_hp):
            out.add(
                "hp", "own_exact", "", key,
                truth=[tm["hp"], tm["maxhp"]], belief=[bm.current_hp, bm.max_hp],
            )  # fmt: skip
    else:
        pct = _hp_percent(tm["hp"], tm["maxhp"])
        believed = bm.current_hp
        scale = bm.max_hp
        if scale and scale != 100:
            believed = round(100 * (believed or 0) / scale)
        if believed is None or pct != believed:
            out.add(
                "hp", "foe_percent", "", key, truth=pct, belief=believed,
            )  # fmt: skip
    # status
    t_status = tm["status"] or None
    if t_status != bm.status:
        out.add(
            "status", "mismatch", f"{t_status}->{bm.status}", key, truth=t_status, belief=bm.status
        )
    elif t_status == "slp" or (t_status == "tox" and tm["active"] and not mid_turn):
        # (a benched toxic Pokemon keeps a stale stage: Showdown resets it on switch-IN)
        ss = tm["status_state"]
        if t_status == "slp":
            t_count = (ss.get("startTime") or 0) - (ss.get("time") or 0)
        else:
            t_count = ss.get("stage")
        if t_count is not None and t_count != bm.status_counter:
            out.add(
                "status_counter", "mismatch", t_status, key,
                truth=t_count, belief=bm.status_counter,
            )  # fmt: skip
    # boosts
    b_boosts = dict(bm.boosts)
    for stat, val in tm["boosts"].items():
        if b_boosts.get(stat, 0) != val:
            out.add(
                "boosts", "mismatch", stat, key, truth=val, belief=b_boosts.get(stat, 0),
            )  # fmt: skip
    # item and ability: what a disguised Pokemon "has" is exactly what the Illusion hides
    if label == "own":
        t_item = tm["item"] or None
        if t_item != bm.item_id:
            out.add(
                "item", "own", f"{t_item}->{bm.item_id}", key, truth=t_item, belief=bm.item_id,
            )  # fmt: skip
        if tm["ability"] != (bm.ability_id or ""):
            out.add(
                "ability", "own", f"{tm['ability']}->{bm.ability_id}", key,
                truth=tm["ability"], belief=bm.ability_id,
            )  # fmt: skip
    elif not disguised:
        _compare_foe_item(out, tm, bm, key, know, role, name)
        _compare_foe_ability(out, tm, bm, key, know, role, name, memory)
    # volatiles
    t_vol = {
        vid
        for vid, state in tm["volatiles"].items()
        if vid not in HIDDEN_VOLATILES and "twoturnmove" not in str(state.get("sourceEffect"))
    }
    b_vol = {re.sub(r"^(stockpile|fallen)\d$", r"\1", e.id) for e in bm.effects}
    if not mid_turn:
        for vid in sorted(t_vol - b_vol):
            out.add("volatile", "missing", vid, key, truth=tm["volatiles"][vid], belief=None)
        for vid in sorted(b_vol - t_vol):
            if vid in tm["volatiles"]:
                continue
            out.add("volatile", "stale", vid, key, truth=None, belief=vid)
    # two-turn moves are tracked as `preparing`, not as an effect
    two_turn = tm["volatiles"].get("twoturnmove")
    t_prep = to_id(two_turn.get("move")) if two_turn else None
    if not mid_turn and (t_prep or None) != (bm.preparing_move_id if bm.preparing else None):
        out.add(
            "preparing", "mismatch", f"{t_prep}->{bm.preparing_move_id}", key,
            truth=t_prep, belief=bm.preparing_move_id if bm.preparing else None,
        )  # fmt: skip
    # protect counter
    stall = tm["volatiles"].get("stall")
    t_protect = 0
    if stall and stall.get("counter"):
        c, t_protect = int(stall["counter"]), 0
        while c > 1:
            c //= 3
            t_protect += 1
    if not mid_turn and t_protect != min(bm.protect_counter, 6):  # stall caps at 3**6
        out.add(
            "protect_counter", "mismatch", "", key, truth=t_protect, belief=bm.protect_counter
        )  # fmt: skip
    # foe moves (a disguise's moves are the deceived client's own business)
    if label == "foe" and not disguised and not tm["transformed"]:
        known = {m.id for m in bm.moves}
        shown = know.revealed_moves(role, name)
        legit = set(tm["moves"]) | set(tm["base_moves"]) | {"struggle", "recharge"}
        extra = sorted(known - legit)
        if extra:
            out.add(
                "foe_moves", "not_in_truth", ",".join(extra), key,
                truth=sorted(legit), belief=sorted(known),
            )  # fmt: skip
        missing = sorted(shown - known)
        if missing:
            out.add(
                "foe_moves", "shown_but_unknown", ",".join(missing), key,
                truth=sorted(shown), belief=sorted(known),
            )  # fmt: skip
        unrevealed = sorted((known & legit) - shown - {"struggle", "recharge"})
        if unrevealed:
            out.add(
                "foe_moves", "known_but_unrevealed", ",".join(unrevealed), key,
                truth=sorted(shown), belief=sorted(known),
            )  # fmt: skip


def _compare_foe_ability(
    out: Findings,
    tm: dict[str, Any],
    bm: Any,
    key: str,
    know: ProtocolKnowledge,
    role: str,
    name: str,
    memory: BattleMemory,
) -> None:
    believed = bm.ability_id or memory.opponent_abilities.get(bm.species_id)
    revealed = know.current_ability(role, name)
    if believed and believed != tm["ability"]:
        out.add(
            "ability", "foe_wrong", f"{tm['ability']}->{believed}", key,
            truth=tm["ability"], belief=believed,
        )  # fmt: skip
    elif not believed and revealed and revealed == tm["ability"]:
        out.add(
            "ability", "foe_revealed_but_unknown", revealed, key,
            truth=tm["ability"], belief=None,
        )  # fmt: skip
    elif (
        believed
        and not know.revealed_abilities(role, name)
        and len(legal_abilities(tm["species"])) != 1
    ):
        # the bot holds an ability the protocol never showed for a species with a choice
        out.add(
            "ability", "foe_known_but_unrevealed", believed, key,
            truth=tm["ability"], belief=believed,
        )  # fmt: skip


def _compare_foe_item(
    out: Findings,
    tm: dict[str, Any],
    bm: Any,
    key: str,
    know: ProtocolKnowledge,
    role: str,
    name: str,
) -> None:
    t_item = tm["item"]
    mk = know.mons.get((role, name))
    shown = mk.item_shown if mk else None
    if t_item:
        # Item still held: believing a different known item, or "consumed", is wrong;
        # "unknown" is legitimately hidden unless the protocol showed it.
        if bm.item_state == "known" and bm.item_id != t_item:
            out.add(
                "item", "foe_wrong_known", f"{t_item}->{bm.item_id}", key,
                truth=t_item, belief=bm.item_id,
            )  # fmt: skip
        elif bm.item_state == "consumed":
            out.add("item", "foe_marked_consumed", t_item, key, truth=t_item, belief="consumed")
        elif bm.item_state != "known" and shown == t_item and not (mk and mk.item_gone):
            out.add(
                "item", "foe_shown_but_unknown", t_item, key, truth=t_item, belief=bm.item_state
            )  # fmt: skip
        elif bm.item_state == "known" and shown != t_item:
            out.add(
                "item", "foe_known_but_unrevealed", t_item, key, truth=t_item, belief=bm.item_id
            )  # fmt: skip
    elif not tm["had_item"]:
        return  # never held one: "no item" is not public information
    else:
        # Item gone (consumed / knocked off / stolen / tricked away): public in the protocol.
        if bm.item_state == "known":
            out.add(
                "item", "foe_gone_but_known", str(bm.item_id), key, truth=None, belief=bm.item_id
            )
        elif bm.item_state == "unknown":
            out.add("item", "foe_gone_but_unknown", "", key, truth=None, belief="unknown")


# --- protocol bookkeeping (for attribution) ----------------------------------------------

_FROM = re.compile(r"\[from\] (?:ability|item|move): ([^|]+)")


def line_tag(split: list[str]) -> str:
    tag = split[1] if len(split) > 1 else "?"
    detail = ""
    if tag in ("move", "cant", "-activate", "-start", "-end", "-singleturn", "-singlemove",
               "-sidestart", "-sideend", "-fieldstart", "-fieldend", "-weather", "-status",
               "-curestatus", "-boost", "-unboost", "-setboost", "-item", "-enditem",
               "-ability", "-endability", "-immune", "-fail", "-block", "-miss", "-prepare",
               "-formechange", "detailschange", "replace", "-transform"):  # fmt: skip
        if tag in ("-boost", "-unboost", "-setboost"):
            detail = ""
        elif tag == "move":
            detail = split[3] if len(split) > 3 else ""
        elif tag in ("-sidestart", "-sideend"):
            detail = split[3] if len(split) > 3 else ""
        elif tag in ("-weather", "-fieldstart", "-fieldend"):
            detail = split[2] if len(split) > 2 else ""
        else:
            detail = split[3] if len(split) > 3 and not split[3].startswith("[") else ""
    sources = [m.group(0) for p in split[4:] for m in [_FROM.search(p)] if m]
    suffix = f" {sources[0]}" if sources else ""
    return f"{tag}:{detail}{suffix}" if detail else f"{tag}{suffix}"


def _group(f: dict[str, Any]) -> str:
    base = (
        f"{f['field']}/{f['kind']}/{f['detail']}" if f["detail"] else f"{f['field']}/{f['kind']}"
    )
    return f"[{f['exempt']}] {base}" if f.get("exempt") else base


# --- the audited battle -------------------------------------------------------------------


def _new_acc() -> dict[str, Any]:
    return {
        "battles": 0,
        "steps": 0,
        "compares": 0,
        "observations": Counter(),
        "episodes": Counter(),
        "tags": defaultdict(Counter),
        "examples": defaultdict(list),
        "coverage": Counter(),
        "cov": defaultdict(Counter),
        "errors": [],
        "agents": Counter(),
        "winners": Counter(),
        "tracker_notes": Counter(),
    }


def _record_coverage(acc: dict[str, Any], lines: list[str], truth_state: dict[str, Any]) -> None:
    """What this step exercised: protocol effects from the raw lines, and the Pokemon / items /
    abilities / conditions present on the board in the dump (presence is the weaker tier)."""
    cov = acc["cov"]
    for ln in lines:
        split = ln.split("|")
        if len(split) < 3 or split[1] == "request":
            continue
        cov["tags"][split[1]] += 1
        for part in split[2:]:
            if part.startswith("[from]"):
                body = part[len("[from]"):].strip()
                kind, _, name = body.partition(":")
                kind = kind.strip()
                if kind in ("ability", "item", "move"):
                    cov[f"{kind}_trigger"][to_id(name)] += 1
            elif part.startswith("ability:"):
                cov["ability_trigger"][to_id(part.split(":", 1)[1])] += 1
            elif part.startswith("item:"):
                cov["item_trigger"][to_id(part.split(":", 1)[1])] += 1
            elif part.startswith("move:") and split[1] in ("-activate", "-start", "-end", "-singleturn"):
                cov["move_trigger"][to_id(part.split(":", 1)[1])] += 1
        if split[1] == "move" and len(split) > 3:
            cov["move_used"][to_id(split[3])] += 1
        elif split[1] == "-ability" and len(split) > 3:
            cov["ability_trigger"][to_id(split[3])] += 1
        elif split[1] in ("-item", "-enditem") and len(split) > 3:
            cov["item_trigger"][to_id(split[3])] += 1
        elif split[1] == "-mega" and len(split) > 4:
            cov["item_trigger"][to_id(split[4])] += 1
        elif split[1] == "-weather":
            cov["field_condition"][to_id(split[2])] += 1
        elif split[1] in ("-fieldstart", "-sidestart") and len(split) > 3:
            cov["field_condition" if split[1] == "-fieldstart" else "side_condition"][
                to_id(split[3].split(":")[-1])
            ] += 1
        elif split[1] == "-start" and len(split) > 3:
            cov["volatile"][to_id(split[3].split(":")[-1])] += 1
    for side in truth_state["sides"]:
        for p in side["pokemon"]:
            if p.get("isActive"):
                cov["ability_present"][to_id(p.get("ability"))] += 1
                if p.get("item"):
                    cov["item_present"][to_id(p.get("item"))] += 1
                for vid in (p.get("volatiles") or {}):
                    cov["volatile"][to_id(vid)] += 1
                if p.get("status"):
                    cov["status"][p["status"]] += 1
                cov["species_present"][to_id(str(p.get("species", "")).replace("[Species:", "").rstrip("]"))] += 1
        for cid in side.get("sideConditions") or {}:
            cov["side_condition"][to_id(cid)] += 1
        for slot in side.get("slotConditions") or []:
            for cid in slot or {}:
                cov["slot_condition"][to_id(cid)] += 1
    field = truth_state["field"]
    if field.get("weather"):
        cov["field_condition"][to_id(field["weather"])] += 1
    if field.get("terrain"):
        cov["field_condition"][to_id(field["terrain"])] += 1
    for cid in field.get("pseudoWeather") or {}:
        cov["field_condition"][to_id(cid)] += 1


def audit_battle(
    worker: SimWorker,
    battle_id: str,
    names: dict[str, str],
    teams: dict[str, str],
    seed: list[int],
    acc: dict[str, Any],
    trace: bool = False,
    agents: dict[str, Any] | None = None,
    max_steps: int = MAX_STEPS,
) -> None:
    """Play one battle, comparing the bot's view with Showdown's after every step.

    `agents` replaces the baseline players (the targeted scenarios pass scripted ones)."""
    agents = agents or {s: make_direct_agent(names[s], teams[s]) for s in SIDES}
    battle = DirectBattle.start(worker, battle_id, teams["p1"], teams["p2"], seed=seed)
    memories = {s: BattleMemory(battle_tag=battle_id) for s in SIDES}
    knowledge = {s: ProtocolKnowledge(s) for s in SIDES}
    prev_truth: dict[str, Any] | None = None
    collided: dict[str, set[str]] = {s: set() for s in SIDES}
    deceived: dict[str, set[str]] = {s: set() for s in SIDES}
    episodes: set[tuple] = set()
    acc["battles"] += 1
    for s in SIDES:
        acc["agents"][names[s]] += 1

    def feed(lines: dict[str, list[str]]) -> None:
        for s in SIDES:
            agents[s].observe(battle_id, lines[s])
            memories[s].observe_protocol([ln.split("|") for ln in lines[s]])
            knowledge[s].feed(lines[s])
            for ln in lines[s]:
                acc["coverage"][line_tag(ln.split("|"))] += 1

    def record(f: dict[str, Any], s: str, truth: dict[str, Any], lines: list[str], step: int) -> None:
        g = _group(f)
        if trace:
            print(
                f"  !! {s} turn {truth['turn']} {g} {f['mon']} truth={f['truth']} "
                f"belief={f['belief']}"
            )
        acc["observations"][g] += 1
        ep = (s, f["mon"], g)
        if ep in episodes:
            return
        episodes.add(ep)
        acc["episodes"][g] += 1
        mon_name = f["mon"].split(":", 1)[-1]
        touching = [ln for ln in lines if mon_name and mon_name in ln] or lines
        for ln in touching:
            acc["tags"][g][line_tag(ln.split("|"))] += 1
        if len(acc["examples"][g]) < MAX_EXAMPLES:
            acc["examples"][g].append(
                {
                    "battle": battle_id,
                    "seed": seed,
                    "perspective": s,
                    "turn": truth["turn"],
                    "step": step,
                    "mon": f["mon"],
                    "truth": f["truth"],
                    "belief": f["belief"],
                    "lines": touching[-8:],
                }
            )

    def compare(lines: dict[str, list[str]], step: int) -> None:
        nonlocal prev_truth
        state = worker.request({"cmd": "dump", "id": battle_id})["state"]
        truth = truth_view(state)
        _record_coverage(acc, [ln for s in SIDES for ln in lines[s]], state)
        for snapshot_truth in (truth, prev_truth):
            if snapshot_truth is None:
                continue
            for role, side in snapshot_truth["sides"].items():
                for tm in side["visible"].values():
                    if tm.get("collision"):
                        collided[role] |= {tm["name"], tm["true_name"]}
        # (a real Pokemon switching in under a live disguise's ident collides mid-step even when
        # the break leaves no collision at either end of it)
        for (role, disguise), (_, holder) in illusion_alias(prev_truth).items():
            marker = f": {disguise}|"
            if any(
                ln.startswith(("|switch|", "|drag|")) and ln.split("|")[2].startswith(role)
                and marker in ln
                for ln in lines["p1"]
            ):
                collided[role] |= {disguise, holder}
        # A disguise episode that ENDS without the Illusion breaking (the Zoroark just switched
        # out, or the step carried no break line) leaves its moves / item / ability events filed
        # under the disguise's name for good: nothing ever tells a client they were not its own.
        now_disguised = illusion_alias(truth)
        for (role, disguise), (_, holder) in illusion_alias(prev_truth).items():
            if (role, disguise) in now_disguised:
                continue
            broke = any(f"illusion_break:{role}" in k.step_flags for k in knowledge.values())
            if not broke:
                deceived[role].add(disguise)
        truth_mons = {
            (role, name): tm
            for role, side in truth["sides"].items()
            for name, tm in side["mons"].items()
        }
        for s in SIDES:
            bt = battle.battles[s]
            if bt.teampreview:
                continue
            memories[s].observe_battle(bt)
            setattr(bt, "_vgc_battle_memory", memories[s])
            snap = snapshot_battle(bt)
            acc["compares"] += 1
            know = knowledge[s]
            unresolved = know.resolve(truth_mons, illusion_alias(prev_truth), illusion_alias(truth))
            findings = compare_perspective(
                s, snap, truth, memories[s], know, bt, lines[s], prev_truth, collided, deceived
            )
            for cand in unresolved:
                # the tracker's own rule could not attribute this line; informational (the
                # bot-vs-truth comparisons above are what fail the audit)
                acc["tracker_notes"][f"{cand.kind}:{cand.tag}"] += 1
            for f in findings:
                record(f, s, truth, lines[s], step)
        prev_truth = truth

    feed(battle.last_lines)
    step = 0
    try:
        while not battle.ended and step < max_steps:
            to_move = battle.sides_to_move()
            for side in to_move:
                battle.battles[side]._vgc_direct_root = battle
                battle.battles[side]._vgc_direct_side = side
            choices = {side: agents[side].choose(battle.battles[side]) for side in to_move}
            result = battle.step(choices)
            step += 1
            acc["steps"] += 1
            if trace:
                print(f"--- step {step} choices={choices}")
                for side in SIDES:
                    for ln in result.lines[side] if side == "p1" else []:
                        if not ln.startswith("|request"):
                            print("   ", ln[:170])
            feed(result.lines)
            if trace:
                for side in SIDES:
                    bt = battle.battles[side]
                    shown = {
                        k: {b: v for b, v in m.boosts.items() if v}
                        for k, m in list(bt.team.items()) + list(bt.opponent_team.items())
                        if any(m.boosts.values())
                    }
                    print(f"   [{side} believes boosts] {shown}")
            if not battle.ended:
                compare(result.lines, step)
        acc["winners"][battle.winner or "none"] += 1
    finally:
        battle.close()


def worker_main(job: dict[str, Any]) -> dict[str, Any]:
    from vgc import poke_env_compat

    poke_env_compat.set_disabled_fixes(job.get("disable") or [])
    trace = bool(job.get("trace"))
    acc = _new_acc()
    builder = TeamBuilder()
    pool = job["pool_teams"]
    with SimWorker() as worker:
        for index in job["indices"]:
            rng = random.Random(job["seed"] * 1_000_003 + index)
            random.seed(job["seed"] * 7_919 + index)  # the baseline players draw from here
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
            try:
                audit_battle(worker, f"audit{index}", names, teams, seed, acc, trace)
            except Exception as exc:  # keep auditing; report at the end
                acc["errors"].append(
                    {
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc()[-1500:],
                    }
                )
                try:
                    worker.request({"cmd": "close", "id": f"audit{index}"})
                except Exception:
                    pass
    return acc


def merge(into: dict[str, Any], part: dict[str, Any]) -> None:
    for key in ("battles", "steps", "compares"):
        into[key] += part[key]
    for key in ("observations", "episodes", "coverage", "agents", "winners", "tracker_notes"):
        into[key].update(part[key])
    for g, counter in part["tags"].items():
        into["tags"][g].update(counter)
    for g, counter in part["cov"].items():
        into["cov"][g].update(counter)
    for g, examples in part["examples"].items():
        room = MAX_EXAMPLES - len(into["examples"][g])
        into["examples"][g].extend(examples[:room])
    into["errors"].extend(part["errors"])


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
    return teams[:limit]


def classify_group(group: str) -> tuple[str, str]:
    """("exempt" | "unexplained", exemption id). Exemptions were decided per finding by
    `audit_exemptions.classify`; the group key carries the id as a `[id]` prefix."""
    match = re.match(r"^\[([a-z0-9_]+)\] ", group)
    return ("exempt", match.group(1)) if match else ("unexplained", "")


def unexplained_groups(acc: dict[str, Any]) -> list[str]:
    return sorted(g for g in acc["episodes"] if classify_group(g)[0] == "unexplained")


def report_text(acc: dict[str, Any], elapsed: float) -> str:
    lines = [
        f"battles={acc['battles']} steps={acc['steps']} perspective-compares={acc['compares']} "
        f"errors={len(acc['errors'])} elapsed={elapsed:.0f}s",
        f"agents={dict(acc['agents'])}",
    ]
    for kind, title in (
        ("unexplained", "DEFECTS (unexplained mismatches)"),
        ("exempt", "EXEMPT (documented predicate holds for the specific finding)"),
    ):
        rows = [
            (g, n)
            for g, n in sorted(acc["episodes"].items(), key=lambda kv: -kv[1])
            if classify_group(g)[0] == kind
        ]
        total = sum(n for _, n in rows)
        lines += ["", f"== {title}: {len(rows)} groups, {total} episodes"]
        if rows:
            lines.append(f"{'group':64s} {'episodes':>8s} {'obs':>8s}  top triggering messages")
        for g, n in rows:
            top = ", ".join(f"{t} x{c}" for t, c in acc["tags"][g].most_common(3))
            lines.append(f"{g:64s} {n:8d} {acc['observations'][g]:8d}  {top}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--battles", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--agents", default="random,random,maxpower,heuristic,vgc_myopic")
    parser.add_argument("--pool-fraction", type=float, default=0.3)
    parser.add_argument("--pool-limit", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-on-unexplained",
        action="store_true",
        help="exit 1 when a mismatch group matches no documented exemption",
    )
    parser.add_argument(
        "--disable-fixes",
        default="",
        help="comma list of vgc.poke_env_compat.ALL_FIXES names, or 'all' "
        "(the 'before' arm of a before/after comparison)",
    )
    parser.add_argument(
        "--trace-index",
        type=int,
        default=None,
        help="play only this battle index and print every step",
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="also play the targeted coverage scenarios (offline/audit_scenarios.py)",
    )
    parser.add_argument(
        "--scenarios-only",
        action="store_true",
        help="play only the targeted coverage scenarios (no seeded random battles)",
    )
    parser.add_argument(
        "--scenario-filter", default="", help="substring: play only matching scenario names"
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="print which legal moves / abilities / items / messages / conditions were exercised",
    )
    args = parser.parse_args(argv)
    if args.scenarios_only:
        args.scenarios = True
        args.battles = 0

    pool_teams = load_pool_teams(list(DEFAULT_MANIFESTS), args.pool_limit)
    from vgc import poke_env_compat

    disable = (
        list(poke_env_compat.ALL_FIXES)
        if args.disable_fixes == "all"
        else [n for n in args.disable_fixes.split(",") if n]
    )
    if args.trace_index is not None:
        job = {
            "disable": disable,
            "indices": [args.trace_index], "seed": args.seed, "agents": args.agents.split(","),
            "pool_teams": pool_teams,
            "pool_fraction": args.pool_fraction if pool_teams else 0.0, "trace": True,
        }  # fmt: skip
        acc = worker_main(job)
        print(report_text(acc, 0.0))
        for err in acc["errors"]:
            print(err["trace"])
        return 0
    jobs = []
    workers = max(1, min(args.workers, max(args.battles, 1)))
    for w in range(workers if args.battles else 0):
        jobs.append(
            {
                "indices": list(range(w, args.battles, workers)),
                "seed": args.seed,
                "agents": args.agents.split(","),
                "pool_teams": pool_teams,
                "pool_fraction": args.pool_fraction if pool_teams else 0.0,
                "disable": disable,
            }
        )
    scenario_jobs = []
    if args.scenarios:
        import audit_scenarios

        scenario_jobs = audit_scenarios.make_jobs(
            max(1, args.workers), disable=disable, name_filter=args.scenario_filter
        )
    started = time.time()
    total = _new_acc()
    ctx = multiprocessing.get_context("spawn")
    n_pool = max(1, len(jobs) + len(scenario_jobs))
    with ProcessPoolExecutor(max_workers=min(args.workers, n_pool), mp_context=ctx) as pool:
        futures = [pool.submit(worker_main, job) for job in jobs]
        if scenario_jobs:
            import audit_scenarios

            futures += [pool.submit(audit_scenarios.run_job, job) for job in scenario_jobs]
        for future in futures:
            merge(total, future.result())
    elapsed = time.time() - started
    print(report_text(total, elapsed))
    for err in total["errors"][:5]:
        print("\nERROR", err["index"], err["error"], "\n", err["trace"])
    if args.coverage_report:
        import audit_coverage

        print()
        print(audit_coverage.report_text(total["cov"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        "battles": total["battles"],
        "steps": total["steps"],
        "compares": total["compares"],
        "elapsed_seconds": round(elapsed, 1),
        "episodes": dict(total["episodes"]),
        "classification": {g: classify_group(g)[0] for g in total["episodes"]},
        "unexplained": unexplained_groups(total),
        "disabled_fixes": disable,
        "observations": dict(total["observations"]),
        "tags": {g: dict(c.most_common(6)) for g, c in total["tags"].items()},
        "examples": dict(total["examples"]),
        "coverage": dict(total["coverage"].most_common()),
        "cov": {k: dict(v) for k, v in total["cov"].items()},
        "errors": total["errors"],
        "agents": dict(total["agents"]),
        "seed": args.seed,
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
