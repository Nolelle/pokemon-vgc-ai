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

    active slots / species forme / types / HP (exact own, percent foe) / fainted / status
    + sleep & toxic counters / all 7 boosts / item (known, consumed, lost) / ability
    / volatiles / protect counter / weather / terrain / pseudo-weather (Trick Room, ...)
    / side conditions / remaining durations / revealed foe moves

Mismatches are grouped by (field, kind, detail), counted two ways -- observations (turns it
was wrong) and episodes (distinct battle x perspective x Pokemon x thing that went wrong) --
and attributed to the protocol messages that touched the Pokemon on the step the mismatch
first appeared, so each group points at the message that broke it.

Truth that is legitimately hidden (an unrevealed foe item, an unrevealed ability, Illusion
while it holds, hidden timers) is NOT a mismatch; `_check_*` documents each exemption.

    .venv/bin/python offline/audit_battle_parsing.py --battles 200 --workers 8
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
        "added_type": to_id(p.get("addedType")),
        "active": bool(p.get("isActive")),
        "position": p.get("position"),
        "fainted": bool(p.get("fainted")),
        "had_item": bool((p.get("set") or {}).get("item")),
        "transformed": bool(p.get("transformed")),
        "illusion": bool(p.get("illusion")),
        "moves": [to_id(m.get("id") or m.get("move")) for m in (p.get("moveSlots") or [])],
        "base_moves": [to_id(m) for m in (p.get("set", {}).get("moves") or [])],
    }


def truth_view(state: dict[str, Any]) -> dict[str, Any]:
    sides = {}
    for side in state["sides"]:
        mons = {m["name"]: m for m in (_truth_mon(p) for p in side["pokemon"])}
        conditions = {}
        for cid, cs in (side.get("sideConditions") or {}).items():
            conditions[to_id(cid)] = {
                "duration": cs.get("duration"),
                "layers": cs.get("layers"),
                "setter": (cs.get("sourceSlot") or "")[:2],
            }
        sides[side["id"]] = {"mons": mons, "conditions": conditions}
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
# poke-env effects that are bookkeeping, not battle state (never in Showdown's volatiles).
POKEENV_ONLY_EFFECTS = frozenset({"flashfire", "protean"})  # filled from audit review


class Findings:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

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
        legit: bool = False,
    ) -> None:
        self.items.append(
            {
                "legit": legit,
                "field": field,
                "kind": kind,
                "detail": detail,
                "mon": mon,
                "truth": truth,
                "belief": belief,
                "note": note,
            }
        )


def _belief_remaining(snapshot_turn: int, effect: Any) -> int | None:
    base = _BASE_DURATION.get(effect.id)
    if base is None or effect.turns is None:
        return None
    return base - (snapshot_turn - effect.turns)


def _duration_finding(
    out: "Findings", detail: str, label: str, truth_dur: int, rem: int, setter: str, foe_role: str
) -> None:
    """A remaining-duration disagreement. A foe's Damp Rock / Light Clay / Terrain Extender is
    hidden until it is revealed, so truth == belief + 3 on a foe-set condition is legitimate."""
    hidden = truth_dur - rem == 3 and setter == foe_role
    out.add(
        "duration",
        "hidden_foe_extender" if hidden else "mismatch",
        detail,
        label,
        truth=truth_dur,
        belief=rem,
        legit=hidden,
    )


def compare_perspective(
    me: str,
    snapshot: Any,
    truth: dict[str, Any],
    memory: BattleMemory,
    revealed_abilities: dict[str, set[str]],
    revealed_foe_moves: dict[str, set[str]],
) -> list[dict[str, Any]]:
    out = Findings()
    foe_role = "p2" if me == "p1" else "p1"
    if snapshot.turn != truth["turn"]:
        out.add("turn", "mismatch", "", "-", truth=truth["turn"], belief=snapshot.turn)

    illusion_names = {
        name
        for role in SIDES
        for name, mon in truth["sides"][role]["mons"].items()
        if mon["illusion"]
    }

    illusion_roles = {
        role
        for role in SIDES
        if any(m["base_ability"] == "illusion" for m in truth["sides"][role]["mons"].values())
    }

    for label, role, side_state in (
        ("own", me, snapshot.our_side),
        ("foe", foe_role, snapshot.opponent_side),
    ):
        tmons = truth["sides"][role]["mons"]
        seen_names = set()
        for bmon in side_state.pokemon:
            name = bmon.name
            seen_names.add(name)
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
            if tm["illusion"] or name in illusion_names:
                continue  # Illusion is deliberately deceptive; only post-break state counts
            if label == "foe" and (not bmon.revealed or role in illusion_roles):
                continue  # Illusion: the protocol deliberately shows the wrong identities
            _compare_mon(
                out, label, name, tm, bmon, memory, revealed_abilities, revealed_foe_moves,
                me, mid_turn=truth["mid_turn"],
            )  # fmt: skip

        # active slots
        truth_active = {m["position"]: m["name"] for m in tmons.values() if m["active"]}
        for pos, bmon in enumerate(side_state.active_species):
            pass
        belief_active = {}
        for bmon in side_state.pokemon:
            if bmon.active:
                belief_active[bmon.name] = True
        t_active_names = {n for n in truth_active.values() if not tmons[n]["fainted"]}
        b_active_names = {n for n in belief_active if n in tmons and not _fainted(side_state, n)}
        if label == "foe":
            t_active_names = {n for n in t_active_names if n not in illusion_names}
            b_active_names = {n for n in b_active_names if n not in illusion_names}
        if t_active_names != b_active_names and not (label == "foe" and role in illusion_roles):
            out.add(
                "active_set", "mismatch", label, "-",
                truth=sorted(t_active_names), belief=sorted(b_active_names),
            )  # fmt: skip

        # side conditions
        t_cond = {
            k: v
            for k, v in truth["sides"][role]["conditions"].items()
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


def _fainted(side_state: Any, name: str) -> bool:
    return any(m.name == name and m.fainted for m in side_state.pokemon)


def _compare_mon(
    out: Findings,
    label: str,
    name: str,
    tm: dict[str, Any],
    bm: Any,
    memory: BattleMemory,
    revealed_abilities: dict[str, set[str]],
    revealed_foe_moves: dict[str, set[str]],
    me: str,
    *,
    mid_turn: bool,
) -> None:
    key = f"{label}:{name}"
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
    # HP. A benched foe with Regenerator heals silently on the way out, so its percent is
    # legitimately hidden until it returns.
    hp_hidden = label == "foe" and not tm["active"] and tm["ability"] == "regenerator"
    if hp_hidden:
        pass
    elif label == "own":
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
    # item
    if label == "own":
        t_item = tm["item"] or None
        if t_item != bm.item_id:
            out.add(
                "item", "own", f"{t_item}->{bm.item_id}", key, truth=t_item, belief=bm.item_id,
            )  # fmt: skip
    else:
        _compare_foe_item(out, tm, bm, key)
    # ability
    if label == "own":
        if tm["ability"] != (bm.ability_id or ""):
            out.add(
                "ability", "own", f"{tm['ability']}->{bm.ability_id}", key,
                truth=tm["ability"], belief=bm.ability_id,
            )  # fmt: skip
    else:
        believed_ability = bm.ability_id or memory.opponent_abilities.get(bm.species_id)
        revealed = revealed_abilities.get(name, set())
        if believed_ability and believed_ability != tm["ability"]:
            out.add(
                "ability", "foe_wrong", f"{tm['ability']}->{believed_ability}", key,
                truth=tm["ability"], belief=believed_ability,
            )  # fmt: skip
        elif not believed_ability and revealed:
            out.add(
                "ability", "foe_revealed_but_unknown", ",".join(sorted(revealed)), key,
                truth=tm["ability"], belief=None,
            )  # fmt: skip
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
    # foe moves
    if label == "foe" and not tm["transformed"] and "mimic" not in tm["moves"]:
        known = {m.id for m in bm.moves}
        legit = set(tm["moves"]) | set(tm["base_moves"]) | {"struggle", "recharge"}
        extra = sorted(known - legit)
        if extra:
            out.add(
                "foe_moves",
                "not_in_truth",
                ",".join(extra),
                key,
                truth=sorted(legit),
                belief=sorted(known),
            )


def _compare_foe_item(out: Findings, tm: dict[str, Any], bm: Any, key: str) -> None:
    t_item = tm["item"]
    if t_item:
        # Item still held: believing a different known item, or "consumed", is wrong;
        # "unknown" is legitimately hidden.
        if bm.item_state == "known" and bm.item_id != t_item:
            out.add(
                "item",
                "foe_wrong_known",
                f"{t_item}->{bm.item_id}",
                key,
                truth=t_item,
                belief=bm.item_id,
            )
        elif bm.item_state == "consumed":
            out.add("item", "foe_marked_consumed", t_item, key, truth=t_item, belief="consumed")
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


# --- protocol bookkeeping (for attribution and for "revealed" ground truth) ---------------

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


_IDENT = re.compile(r"p[12][ab]?: ([^|,]+)")


def update_revealed(
    lines: list[str], my_role: str, abilities: dict[str, set[str]], moves: dict[str, set[str]]
) -> None:
    """Record, from the public protocol, which foe abilities/moves have been revealed.

    Uses `vgc.poke_env_compat._revealed_ability_holder` for the holder rule, so a wrong rule
    shows up as `foe_wrong` (the believed ability disagrees with Showdown) rather than here.
    """
    from vgc.poke_env_compat import _revealed_ability_holder

    foe = "p2" if my_role == "p1" else "p1"
    for line in lines:
        split = line.split("|")
        if len(split) < 3:
            continue
        tag = split[1]
        found = _revealed_ability_holder(split) if len(split) > 3 else None
        if tag == "-ability" and len(split) > 3:
            found = (split[2], split[3])
            if any("[from] move:" in part for part in split[4:]):
                found = (
                    (split[2], split[4])
                    if len(split) > 4 and not split[4].startswith("[")
                    else None
                )
        if found and found[0].startswith(foe):
            mm = _IDENT.match(found[0])
            if mm:
                abilities[mm.group(1).strip()].add(to_id(found[1]))
        if tag == "move" and len(split) > 3 and split[2].startswith(foe) and "[from]" not in line:
            mm = _IDENT.match(split[2])
            if mm:
                moves[mm.group(1).strip()].add(to_id(split[3]))


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
        "errors": [],
        "agents": Counter(),
        "winners": Counter(),
    }


def _group(f: dict[str, Any]) -> str:
    return f"{f['field']}/{f['kind']}/{f['detail']}" if f["detail"] else f"{f['field']}/{f['kind']}"


def audit_battle(
    worker: SimWorker,
    battle_id: str,
    names: dict[str, str],
    teams: dict[str, str],
    seed: list[int],
    acc: dict[str, Any],
    trace: bool = False,
) -> None:
    agents = {s: make_direct_agent(names[s], teams[s]) for s in SIDES}
    battle = DirectBattle.start(worker, battle_id, teams["p1"], teams["p2"], seed=seed)
    memories = {s: BattleMemory(battle_tag=battle_id) for s in SIDES}
    revealed_ab = {s: defaultdict(set) for s in SIDES}
    revealed_mv = {s: defaultdict(set) for s in SIDES}
    episodes: set[tuple] = set()
    acc["battles"] += 1
    for s in SIDES:
        acc["agents"][names[s]] += 1

    def feed(lines: dict[str, list[str]]) -> None:
        for s in SIDES:
            agents[s].observe(battle_id, lines[s])
            memories[s].observe_protocol([ln.split("|") for ln in lines[s]])
            update_revealed(lines[s], s, revealed_ab[s], revealed_mv[s])
            for ln in lines[s]:
                acc["coverage"][line_tag(ln.split("|"))] += 1

    def compare(lines: dict[str, list[str]], step: int) -> None:
        state = worker.request({"cmd": "dump", "id": battle_id})["state"]
        truth = truth_view(state)
        for s in SIDES:
            bt = battle.battles[s]
            if bt.teampreview:
                continue
            memories[s].observe_battle(bt)
            setattr(bt, "_vgc_battle_memory", memories[s])
            snap = snapshot_battle(bt)
            acc["compares"] += 1
            for f in compare_perspective(
                s, snap, truth, memories[s], revealed_ab[s], revealed_mv[s]
            ):
                g = _group(f)
                if trace:
                    print(
                        f"  !! {s} turn {truth['turn']} {g} {f['mon']} truth={f['truth']} "
                        f"belief={f['belief']}"
                    )
                acc["observations"][g] += 1
                ep = (s, f["mon"], g)
                if ep in episodes:
                    continue
                episodes.add(ep)
                acc["episodes"][g] += 1
                mon_name = f["mon"].split(":", 1)[-1]
                touching = [ln for ln in lines[s] if mon_name and mon_name in ln] or lines[s]
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

    feed(battle.last_lines)
    step = 0
    try:
        while not battle.ended and step < MAX_STEPS:
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
    for key in ("observations", "episodes", "coverage", "agents", "winners"):
        into[key].update(part[key])
    for g, counter in part["tags"].items():
        into["tags"][g].update(counter)
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


# Groups that are NOT parsing defects: Showdown does not show the information.
LEGITIMATELY_HIDDEN = (
    (
        "duration/hidden_foe_extender",
        "a foe's Damp Rock / Heat Rock / Light Clay / Terrain Extender is not shown until revealed",
    ),
)
# Real divergences left in the bot's view, each with the reason it is not repaired. All are
# rarer than ~1 in 100 random-moveset games. An entry may require a triggering protocol message
# (substring match on the tags attributed to the group) so it cannot hide an unrelated defect
# in the same field; a group that matches no entry is reported as UNEXPLAINED.
KNOWN_LIMITATIONS = (
    ("types/mismatch", None, "Trick-or-Treat / Forest's Curse add a THIRD type; poke-env has no slot for it"),
    ("ability/foe_wrong", None, "abilities swapped or changed by Skill Swap / Entrainment / Mummy / "
     "Imposter with the abilities hidden, then lost again on switch-out"),
    ("ability/foe_revealed_but_unknown", None, "same family: the holder's ability changed after the reveal"),
    ("hp/foe_percent", None, "a benched foe's silent Regenerator heal when the ability was acquired mid-battle"),
    ("foe_moves/not_in_truth/transform", None, "Imposter: poke-env lists Transform as a revealed move"),
    ("volatile/missing/yawn", None, "Rest while yawned: poke-env ends Yawn, Showdown keeps the volatile"),
    ("field/missing/fairylock", None, "Fairy Lock arrives as -fieldactivate, which poke-env ignores"),
    ("status_counter/mismatch/slp", None, "sleep timer edge cases (Rest then a Red Card drag)"),
    ("boosts/mismatch", ("Illusion", "replace"), "Illusion broke through a path the repairs do not cover"),
    ("volatile/stale", ("Illusion", "replace"), "Illusion: effects earned under a disguise"),
    ("volatile/stale/focusband", None, "Focus Band activation marker survived one request"),
    ("volatile/stale/gastroacid", None, "Gastro Acid on a disguised Zoroark, found through a request"),
    ("volatile/missing/gastroacid", None, "same: the effect stays on the disguise object"),
    ("duration/mismatch", None, "screens / weather set through Court Change or a swapped Light Clay"),
)  # fmt: skip


def classify_group(group: str, tags: dict[str, int] | None = None) -> tuple[str, str]:
    """("hidden" | "limitation" | "unexplained", reason)."""
    for prefix, reason in LEGITIMATELY_HIDDEN:
        if group.startswith(prefix):
            return "hidden", reason
    for prefix, needs, reason in KNOWN_LIMITATIONS:
        if not group.startswith(prefix):
            continue
        if needs is None or tags is None or any(n in t for n in needs for t in tags):
            return "limitation", reason
    return "unexplained", ""


def unexplained_groups(acc: dict[str, Any]) -> list[str]:
    return sorted(
        g for g in acc["episodes"] if classify_group(g, acc["tags"].get(g))[0] == "unexplained"
    )


def report_text(acc: dict[str, Any], elapsed: float) -> str:
    lines = [
        f"battles={acc['battles']} steps={acc['steps']} perspective-compares={acc['compares']} "
        f"errors={len(acc['errors'])} elapsed={elapsed:.0f}s",
        f"agents={dict(acc['agents'])}",
    ]
    for kind, title in (
        ("unexplained", "DEFECTS (unexplained mismatches)"),
        ("limitation", "KNOWN LIMITATIONS (not repaired)"),
        ("hidden", "LEGITIMATELY HIDDEN (not defects)"),
    ):
        rows = [
            (g, n)
            for g, n in sorted(acc["episodes"].items(), key=lambda kv: -kv[1])
            if classify_group(g, acc["tags"].get(g))[0] == kind
        ]
        total = sum(n for _, n in rows)
        lines += ["", f"== {title}: {len(rows)} groups, {total} episodes"]
        if rows:
            lines.append(f"{'group':52s} {'episodes':>8s} {'obs':>8s}  top triggering messages")
        for g, n in rows:
            top = ", ".join(f"{t} x{c}" for t, c in acc["tags"][g].most_common(3))
            lines.append(f"{g:52s} {n:8d} {acc['observations'][g]:8d}  {top}")
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
        help="exit 1 when a mismatch group matches no documented limitation",
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
    args = parser.parse_args(argv)

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
    workers = max(1, min(args.workers, args.battles))
    for w in range(workers):
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
    started = time.time()
    total = _new_acc()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for part in pool.map(worker_main, jobs):
            merge(total, part)
    elapsed = time.time() - started
    print(report_text(total, elapsed))
    for err in total["errors"][:5]:
        print("\nERROR", err["index"], err["error"], "\n", err["trace"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serial = {
        "battles": total["battles"],
        "steps": total["steps"],
        "compares": total["compares"],
        "elapsed_seconds": round(elapsed, 1),
        "episodes": dict(total["episodes"]),
        "classification": {
            g: classify_group(g, total["tags"].get(g))[0] for g in total["episodes"]
        },
        "unexplained": unexplained_groups(total),
        "disabled_fixes": disable,
        "observations": dict(total["observations"]),
        "tags": {g: dict(c.most_common(6)) for g, c in total["tags"].items()},
        "examples": dict(total["examples"]),
        "coverage": dict(total["coverage"].most_common()),
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
