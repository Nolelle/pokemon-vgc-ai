"""Native Rust `pikapro`-powered multi-world lookahead search.

Replaces Node.js `SimWorker` rollouts (~2.5s/turn for 6w x 6a x 2s) with in-process
Rust `pikapro.Battle.patch_public` + `Battle.fork()` + `Battle.encode_state_rich()`
(~30ms/turn for 6w x 6a x 2s, or <0.25s/turn for full 16w x 8a x 4s).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

try:
    import pikapro
except ImportError:  # pragma: no cover - optional replica extra
    pikapro = None  # type: ignore[assignment]

from vgc.actions import locked_move_ids
from vgc.damage import to_id
from vgc.data import load_items, load_species
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.live_mirror import (
    MirrorHypothesis,
    _disabled_move_payload,
    _opponent_sets,
    _team_preview_order,
    compute_mirror_hypotheses,
    mirror_side,
)
from vgc.rl.replica_search import ReplicaSearchResult, SampledWorld, search_sampled_worlds
from vgc.rl.rust_policy import RICH_MON_WIDTH, RustFactoredPolicy
from vgc.sets import normalize_item
from vgc.stats import STAT_IDS

_BOOST_ORDER = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_WEATHER_MAP = {
    "sunnyday": "sun",
    "raindance": "rain",
    "sandstorm": "sand",
    "snowscape": "snow",
    "hail": "snow",
    "desolateland": "harshsun",
    "primordialsea": "heavyrain",
    "deltastream": "strongwinds",
}
_TERRAIN_MAP = {
    "electricterrain": "electric",
    "grassyterrain": "grassy",
    "mistyterrain": "misty",
    "psychicterrain": "psychic",
}
_STATUS_NAMES = {
    "brn": "brn",
    "frz": "frz",
    "par": "par",
    "psn": "psn",
    "tox": "tox",
    "slp": "slp",
}


def _pack_sets(sets: list[dict[str, Any]]) -> str:
    """Serialize `_opponent_sets` dicts into Showdown packed format for `pikapro.Battle`."""
    parts: list[str] = []
    for s in sets:
        name = str(s.get("name") or s.get("species") or "")
        species = str(s.get("species") or "")
        item = str(s.get("item") or "")
        ability = str(s.get("ability") or "")
        moves = ",".join(str(m) for m in (s.get("moves") or []))
        nature = str(s.get("nature") or "serious")
        evs_dict = s.get("evs") or {}
        evs = ",".join(str(int(evs_dict.get(st, 0) or 0) or "") for st in STAT_IDS)
        gender = str(s.get("gender") or "")
        level = str(int(s.get("level") or 50))
        # name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness
        parts.append(f"{name}|{species}|{item}|{ability}|{moves}|{nature}|{evs}|{gender}|||{level}|")
    return "]".join(parts)


def _turns_left(start_turn: int | None, current_turn: int, default_duration: int) -> int:
    if start_turn is None:
        return default_duration
    # `elapsed` may be negative when `vgc.condition_clock` records an 8-turn extender item
    # (Damp/Heat/Smooth/Icy Rock, Terrain Extender, Light Clay) net of its +3-turn bonus.
    elapsed = int(current_turn) - int(start_turn)
    return max(1, min(8, default_duration - elapsed))


def _effect_turns_map(effects_raw: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in effects_raw or []:
        if isinstance(e, dict):
            eid = to_id(e.get("id"))
            if eid:
                out[eid] = e
        else:
            eid = to_id(e)
            if eid:
                out[eid] = {"id": eid}
    return out


def _remaining_volatile(eff: dict[str, Any] | None, default_duration: int) -> int:
    if not eff:
        return 0
    turns = eff.get("turns")
    if isinstance(turns, int):
        return max(1, default_duration - max(0, turns))
    return default_duration


def _is_choice_item(item_id: str | None) -> bool:
    if not item_id:
        return False
    entry = load_items().get(item_id)
    if entry is not None:
        return bool(entry.get("isChoice"))
    return item_id in ("choicescarf", "choiceband", "choicespecs")


def _side_patch_from_snapshot(
    side_snap: dict[str, Any],
    *,
    is_own: bool,
    current_turn: int,
    hidden_timers: dict[str, Any],
    disabled_by_species: dict[str, dict[str, str]] | None = None,
    own_locked_by_species: dict[str, str] | None = None,
    memory: Any = None,
    role: str | None = None,
    keep_hidden_items: bool = False,
) -> dict[str, Any]:
    active_species_raw = [
        (to_id(sp) if sp else None) for sp in (side_snap.get("active_species") or [None, None])[:2]
    ]
    trapped_flags = list(side_snap.get("trapped") or [False, False])
    trapped_by_species: dict[str, bool] = {}
    for slot_i, sp in enumerate(active_species_raw):
        if sp and slot_i < len(trapped_flags) and trapped_flags[slot_i]:
            trapped_by_species[sp] = True

    mons: list[dict[str, Any]] = []
    for m in side_snap.get("pokemon") or []:
        sp_id = str(m.get("species_id") or "")
        if not sp_id:
            continue
        base_id = str(m.get("base_species_id") or sp_id)
        mon_name = str(m.get("name") or sp_id)
        forme = sp_id if sp_id != base_id else None
        status_raw = str(m.get("status") or "").lower()
        status = _STATUS_NAMES.get(status_raw)
        timer_info = hidden_timers.get(sp_id) or hidden_timers.get(base_id) or {}
        elapsed_status = max(0, int(m.get("status_counter") or 0))
        if status == "slp":
            if timer_info.get("sleepTime") is not None:
                status_turns = max(1, int(timer_info["sleepTime"]))
            else:
                status_turns = max(1, 3 - elapsed_status)
        elif status == "tox":
            status_turns = min(15, elapsed_status)
        elif status == "frz":
            status_turns = max(1, 3 - elapsed_status)
        else:
            status_turns = 1

        boosts_raw = m.get("boosts") or {}
        boosts_map = boosts_raw if isinstance(boosts_raw, dict) else dict(boosts_raw)
        boosts = [int(boosts_map.get(st, 0) or 0) for st in _BOOST_ORDER]

        eff_map = _effect_turns_map(m.get("effects"))
        effects = set(eff_map)
        is_trapped = bool(
            m.get("trapped", False)
            or trapped_by_species.get(to_id(sp_id), False)
            or trapped_by_species.get(to_id(base_id), False)
        )
        is_active = bool(m.get("active", False))
        if memory is not None and role is not None and hasattr(memory, "has_acted"):
            acted = bool(memory.has_acted(role, mon_name))
        else:
            acted = not bool(m.get("first_turn", False))
        if m.get("active_turns"):
            active_turns = int(m["active_turns"])
        elif is_active and memory is not None and role is not None and hasattr(memory, "active_turns"):
            active_turns = int(memory.active_turns(role, mon_name, current_turn))
        else:
            active_turns = 1 if is_active else 0
        active_move_actions = 1 if acted else 0

        if "confusion" in effects:
            if timer_info.get("confusionTime") is not None:
                confusion_turns = max(1, int(timer_info["confusionTime"]))
            else:
                confusion_turns = _remaining_volatile(eff_map.get("confusion"), 3)
        else:
            confusion_turns = 0

        taunt_turns = _remaining_volatile(eff_map.get("taunt"), 3) if "taunt" in effects else 0
        last_move = to_id(m.get("last_move_id")) or None

        entry: dict[str, Any] = {
            "species": base_id,
            "fainted": bool(m.get("fainted", False)),
            "boosts": boosts,
            "protect_counter": int(m.get("protect_counter") or 0),
            "active_turns": active_turns,
            "active_move_actions": active_move_actions,
            "trapped": is_trapped,
            "taunt_turns": taunt_turns,
            "confusion_turns": confusion_turns,
            "substitute": "substitute" in effects,
            "must_recharge": bool(m.get("must_recharge", False)) or "mustrecharge" in effects,
            "unburden": bool(m.get("item_lost_this_stint")) or "unburden" in effects,
        }
        if last_move:
            entry["last_move"] = last_move
        if forme:
            entry["forme"] = forme
        if status:
            entry["status"] = status
            entry["status_turns"] = status_turns

        cur_hp = m.get("current_hp")
        max_hp = m.get("max_hp")
        if is_own and cur_hp is not None:
            entry["hp_exact"] = int(cur_hp)
        elif cur_hp is not None and max_hp:
            entry["hp_pct"] = max(0, min(100, int(round(100.0 * float(cur_hp) / float(max_hp)))))
        elif m.get("hp_fraction") is not None:
            entry["hp_pct"] = max(0, min(100, int(round(float(m["hp_fraction"]) * 100.0))))

        # `snapshot_battle` uses `item_id` + `item_state` ("known", "none", "unknown").
        raw_item = normalize_item(m.get("item_id") if "item_id" in m else m.get("item"))
        item_state = str(m.get("item_state") or "")
        effective_item: str | None = None
        if is_own:
            if item_state == "none":
                entry["item"] = None
            elif raw_item:
                entry["item"] = raw_item
                effective_item = raw_item
        else:
            if item_state == "known" and raw_item:
                entry["item"] = raw_item
                effective_item = raw_item
            elif item_state in ("none", "consumed"):
                entry["item"] = None
            elif not keep_hidden_items and raw_item:
                entry["item"] = raw_item
                effective_item = raw_item

        # `snapshot_battle` uses `ability_id` + `ability_known`.
        raw_ab = to_id(m.get("ability_id") if "ability_id" in m else m.get("ability"))
        if raw_ab and (is_own or m.get("ability_known", True)):
            entry["ability"] = raw_ab

        move_pp: list[tuple[str, int]] = []
        disabled_moves: list[str] = []
        enabled_moves: list[str] = []
        for mv in m.get("moves") or []:
            if isinstance(mv, dict) and mv.get("id"):
                mid = str(mv["id"])
                if mv.get("current_pp") is not None:
                    move_pp.append((mid, int(mv["current_pp"])))
                if mv.get("disabled"):
                    disabled_moves.append(mid)
                else:
                    enabled_moves.append(mid)
        if move_pp:
            entry["move_pp"] = move_pp
        if disabled_moves:
            entry["disabled_moves"] = disabled_moves

        # Choice lock (revealed Choice item + acted + last_move, or own request hard-lock)
        req_lock = (own_locked_by_species or {}).get(sp_id) or (own_locked_by_species or {}).get(base_id)
        if _is_choice_item(effective_item) and (acted or req_lock):
            lock_mv = req_lock or last_move
            if lock_mv and lock_mv != "struggle":
                entry["choice_lock"] = lock_mv

        # Encore (move + remaining turns)
        if "encore" in effects:
            any_mv = next((str(mv["id"]) for mv in (m.get("moves") or []) if isinstance(mv, dict) and mv.get("id")), None)
            encore_mv = (
                enabled_moves[0]
                if len(enabled_moves) == 1
                else (
                    req_lock
                    or last_move
                    or (enabled_moves[0] if enabled_moves else any_mv)
                )
            )
            if encore_mv:
                entry["encore"] = (encore_mv, _remaining_volatile(eff_map.get("encore"), 3))

        # Disable (move from protocol memory + remaining turns)
        if "disable" in effects:
            dis_info = (disabled_by_species or {}).get(sp_id) or (disabled_by_species or {}).get(base_id) or {}
            dis_mv = to_id(dis_info.get("disabledMove"))
            if dis_mv:
                entry["disable"] = (dis_mv, _remaining_volatile(eff_map.get("disable"), 4))

        # Extended VGC countdowns, DoT volatiles, charging & rampage locks
        if "perishsong" in effects:
            peff = eff_map["perishsong"]
            pcount = peff.get("turns") if isinstance(peff.get("turns"), int) else peff.get("raw_value")
            if isinstance(pcount, int) and pcount > 0:
                entry["perish_turns"] = min(3, pcount)
            else:
                entry["perish_turns"] = 1
        if "yawn" in effects:
            entry["yawn_turns"] = 1
        if "leechseed" in effects:
            entry["leech_seed"] = 0
        if "saltcure" in effects:
            entry["salt_cure"] = True
        if "curse" in effects:
            entry["curse"] = True
        if "syrupbomb" in effects:
            entry["syrup_bomb_turns"] = _remaining_volatile(eff_map.get("syrupbomb"), 3)
        if "healblock" in effects:
            entry["heal_block_turns"] = _remaining_volatile(eff_map.get("healblock"), 2)
        if "throatchop" in effects:
            entry["throat_chop_turns"] = _remaining_volatile(eff_map.get("throatchop"), 2)
        if "partiallytrapped" in effects:
            entry["partially_trapped"] = (
                "whirlpool",
                _remaining_volatile(eff_map.get("partiallytrapped"), 4),
            )

        if m.get("preparing") and m.get("preparing_move_id"):
            prep_target = m.get("preparing_target")
            t_loc = int(prep_target) if isinstance(prep_target, int) else 1
            entry["two_turn_move"] = (to_id(m["preparing_move_id"]), t_loc)

        if (
            not entry["fainted"]
            and "confusion" not in effects
            and memory is not None
            and role is not None
            and hasattr(memory, "locked_move")
        ):
            rampage = memory.locked_move(role, mon_name, current_turn)
            if rampage:
                # For our own side, confirm the live Showdown request actually locked us
                # (prevents forcing a lock if Outrage missed, hit Protect, or hit a Fairy).
                if is_own:
                    req_lock = (
                        (own_locked_by_species or {}).get(sp_id)
                        or (own_locked_by_species or {}).get(base_id)
                    )
                    enabled_mvs = [
                        mv for mv in (m.get("moves") or []) if not mv.get("disabled")
                    ]
                    if req_lock == rampage or (is_trapped and len(enabled_mvs) == 1):
                        entry["locked_move"] = (rampage, 1)
                else:
                    entry["locked_move"] = (rampage, 1)

        mons.append(entry)

    side_conds: list[tuple[str, int]] = []
    for cond in side_snap.get("side_conditions") or []:
        cid = to_id(cond.get("id") if isinstance(cond, dict) else cond)
        if not cid:
            continue
        start = (cond.get("turns") if isinstance(cond, dict) else None)
        layers = (cond.get("raw_value") if isinstance(cond, dict) else None)
        if cid == "tailwind":
            side_conds.append((cid, _turns_left(start, current_turn, 4)))
        elif cid in ("reflect", "lightscreen", "auroraveil", "safeguard", "mist"):
            side_conds.append((cid, _turns_left(start, current_turn, 5)))
        elif cid in ("spikes", "toxicspikes"):
            side_conds.append((cid, max(1, int(layers or 1))))
        else:
            side_conds.append((cid, 1))

    active_species = [
        (str(sp) if sp else None) for sp in (side_snap.get("active_species") or [None, None])[:2]
    ]
    while len(active_species) < 2:
        active_species.append(None)
    force_switch = [bool(x) for x in (side_snap.get("force_switch") or [False, False])[:2]]
    while len(force_switch) < 2:
        force_switch.append(False)

    fainted_now = sum(1 for m in mons if m["fainted"])
    if memory is not None and role is not None:
        total_fainted = max(
            fainted_now,
            int(memory.faint_count(role)) if hasattr(memory, "faint_count") else fainted_now,
        )
        fainted_last_turn = (
            bool(memory.lost_pokemon_last_turn(role, current_turn))
            if hasattr(memory, "lost_pokemon_last_turn")
            else False
        )
    else:
        total_fainted = fainted_now
        fainted_last_turn = False

    return {
        "mega_used": bool(side_snap.get("used_mega_evolution", False)),
        "total_fainted": total_fainted,
        "fainted_last_turn": fainted_last_turn,
        "active_species": active_species,
        "force_switch": force_switch,
        "side_conditions": side_conds,
        "mons": mons,
    }


def _build_rust_patch_dict(
    battle,
    snapshot: dict[str, Any],
    perspective: str,
    hypothesis: MirrorHypothesis | None,
    *,
    memory: Any = None,
) -> dict[str, Any]:
    turn = int(snapshot.get("turn") or getattr(battle, "turn", 1) or 1)
    patch: dict[str, Any] = {"turn": max(1, turn)}

    # Weather
    for w in snapshot.get("weather") or []:
        wid = to_id(w.get("id") if isinstance(w, dict) else w)
        if wid in _WEATHER_MAP:
            patch["weather"] = _WEATHER_MAP[wid]
            start = w.get("turns") if isinstance(w, dict) else None
            patch["weather_turns"] = _turns_left(start, turn, 5)
            break

    # Field effects (terrain, trick room, gravity, rooms)
    for f in snapshot.get("fields") or []:
        fid = to_id(f.get("id") if isinstance(f, dict) else f)
        start = f.get("turns") if isinstance(f, dict) else None
        if fid in _TERRAIN_MAP:
            patch["terrain"] = _TERRAIN_MAP[fid]
            patch["terrain_turns"] = _turns_left(start, turn, 5)
        elif fid == "trickroom":
            patch["trick_room_turns"] = _turns_left(start, turn, 5)
        elif fid == "gravity":
            patch["gravity_turns"] = _turns_left(start, turn, 5)
        elif fid == "magicroom":
            patch["magic_room_turns"] = _turns_left(start, turn, 5)
        elif fid == "wonderroom":
            patch["wonder_room_turns"] = _turns_left(start, turn, 5)
        elif fid == "fairylock":
            patch["fairy_lock_turns"] = 1

    timers_payload = getattr(hypothesis, "payload", None) or {}
    hidden_our = timers_payload.get("our") or {}
    hidden_opp = timers_payload.get("opponent") or {}

    mem = memory if memory is not None else getattr(battle, "_vgc_battle_memory", None)
    our_role = perspective
    opp_role = "p2" if perspective == "p1" else "p1"
    disabled_map = _disabled_move_payload(battle)

    own_locked: dict[str, str] = {}
    actives = list(getattr(battle, "active_pokemon", None) or ())
    for slot_i, mv_id in locked_move_ids(battle).items():
        if slot_i < len(actives) and actives[slot_i] is not None and mv_id != "struggle":
            sp = to_id(getattr(actives[slot_i], "species", None))
            if sp:
                own_locked[sp] = mv_id

    our_patch = _side_patch_from_snapshot(
        snapshot["our_side"],
        is_own=True,
        current_turn=turn,
        hidden_timers=hidden_our,
        disabled_by_species=disabled_map.get("our"),
        own_locked_by_species=own_locked,
        memory=mem,
        role=our_role,
    )
    opp_patch = _side_patch_from_snapshot(
        snapshot["opponent_side"],
        is_own=False,
        current_turn=turn,
        hidden_timers=hidden_opp,
        disabled_by_species=disabled_map.get("opponent"),
        memory=mem,
        role=opp_role,
        keep_hidden_items=True,
    )
    patch["sides"] = [our_patch, opp_patch] if perspective == "p1" else [opp_patch, our_patch]
    return patch


def _base_species_id(species_id: str | None) -> str:
    sp = to_id(species_id)
    if not sp:
        return ""
    entry = load_species().get(sp)
    if entry:
        base = to_id(entry.get("baseSpecies"))
        if base:
            return base
    return sp


def _single_legal_ability(species_id: str | None) -> str | None:
    sp = to_id(species_id)
    if not sp:
        return None
    entry = load_species().get(sp)
    if not entry:
        return None
    abilities = entry.get("abilities") or {}
    vals = {to_id(v) for v in abilities.values() if to_id(v)}
    return next(iter(vals)) if len(vals) == 1 else None


def _foe_public_knowledge(
    battle, opp_sets: list[dict[str, Any]], *, memory: Any = None
) -> tuple[list[str | None], list[str | None], list[list[str]], list[bool]]:
    """Return per-foe-row publicly revealed items, abilities, moves, and revealed flags (fog-safe)."""
    known_by_id: dict[str, Any] = {}
    known_by_base: dict[str, Any] = {}
    for mon in (getattr(battle, "opponent_team", None) or {}).values():
        sp = to_id(getattr(mon, "species", None))
        if sp:
            known_by_id[sp] = mon
            known_by_base.setdefault(_base_species_id(sp), mon)

    mem_items = getattr(memory, "opponent_items", None) or {}
    mem_abilities = getattr(memory, "opponent_abilities", None) or {}
    mem_moves = getattr(memory, "opponent_moves", None) or {}
    mem_switches = getattr(memory, "opponent_switches", None) or {}

    items: list[str | None] = []
    abilities: list[str | None] = []
    moves: list[list[str]] = []
    revealed: list[bool] = []
    opp_role = getattr(memory, "opponent_role", None)
    for s in opp_sets[:6]:
        sp = to_id(s.get("species"))
        base = _base_species_id(sp)
        mon = known_by_id.get(sp) or known_by_base.get(base)
        mon_sp = to_id(getattr(mon, "species", None)) if mon is not None else sp
        if memory is not None and opp_role and hasattr(memory, "mega_forme") and mon is not None:
            m_name = getattr(mon, "name", None) or getattr(mon, "species", None)
            mega_hint = memory.mega_forme(opp_role, m_name)
            if mega_hint:
                mon_sp = to_id(mega_hint)

        raw_mon_item = getattr(mon, "item", None) if mon is not None else "unknown_item"
        # If poke-env explicitly cleared `mon.item` to None/empty (Knock Off, consumed Berry/Sash),
        # do NOT resurrect the consumed item from `BattleMemory`.
        if mon is not None and (raw_mon_item is None or raw_mon_item == ""):
            it = None
        else:
            it = normalize_item(raw_mon_item) if mon is not None else None
            if not it:
                mem_it = mem_items.get(mon_sp) or mem_items.get(sp) or mem_items.get(base)
                it = normalize_item(mem_it)
            if not it and mon_sp:
                sp_entry = load_species().get(mon_sp) or {}
                if sp_entry.get("isMega") and sp_entry.get("requiredItem"):
                    it = to_id(sp_entry.get("requiredItem"))

        # Check current forme's ability & single-ability Mega forme BEFORE falling back to
        # pre-Mega base species ability in `BattleMemory`.
        ab = to_id(getattr(mon, "ability", None)) if mon is not None else None
        if not ab:
            ab = to_id(mem_abilities.get(mon_sp))
        if not ab and mon_sp:
            ab = _single_legal_ability(mon_sp)
        if not ab:
            ab = to_id(mem_abilities.get(sp) or mem_abilities.get(base))

        mvs_set: list[str] = []
        if mon is not None:
            for mid in getattr(mon, "moves", None) or {}:
                m_id = to_id(mid)
                if m_id and m_id not in mvs_set:
                    mvs_set.append(m_id)
        for key in (mon_sp, sp, base):
            for mid in mem_moves.get(key) or ():
                m_id = to_id(mid)
                if m_id and m_id not in mvs_set:
                    mvs_set.append(m_id)

        is_rev = bool(
            mon is not None
            or mem_switches.get(mon_sp, 0) > 0
            or mem_switches.get(sp, 0) > 0
            or mem_switches.get(base, 0) > 0
            or mvs_set
        )
        items.append(it or None)
        abilities.append(ab or None)
        moves.append(mvs_set)
        revealed.append(is_rev)
    return items, abilities, moves, revealed


def _our_public_knowledge(
    battle, own_packed_team: str, *, memory: Any = None
) -> tuple[list[str | None], list[str | None], list[list[str]], list[bool]]:
    """Return what the opponent has publicly observed of OUR team (strictly fog-safe).

    Matches rows ``0..5`` of ``own_packed_team`` so simulated opponent observations in
    ``encode_state_rich(opp_idx, ...)`` see our revealed items, abilities, moves, and
    switched-in Pokémon — without leaking unrevealed private items/abilities/moves.
    """
    packed_species: list[str] = []
    for chunk in str(own_packed_team or "").split("]"):
        chunk = chunk.strip()
        if not chunk:
            continue
        fields = chunk.split("|")
        sp_field = fields[1] if len(fields) > 1 and fields[1] else fields[0]
        packed_species.append(to_id(sp_field))

    own_by_id: dict[str, Any] = {}
    own_by_base: dict[str, Any] = {}
    for mon in (getattr(battle, "team", None) or {}).values():
        sp = to_id(getattr(mon, "species", None))
        if sp:
            own_by_id[sp] = mon
            own_by_base.setdefault(_base_species_id(sp), mon)

    mem_items = getattr(memory, "our_revealed_items", None) or {}
    mem_abilities = getattr(memory, "our_revealed_abilities", None) or {}
    mem_moves = getattr(memory, "our_revealed_moves", None) or {}
    mem_appeared = getattr(memory, "our_appeared", None) or set()

    items: list[str | None] = []
    abilities: list[str | None] = []
    moves: list[list[str]] = []
    revealed: list[bool] = []
    for sp in packed_species[:6]:
        base = _base_species_id(sp)
        mon = own_by_id.get(sp) or own_by_base.get(base)
        mon_sp = to_id(getattr(mon, "species", None)) if mon is not None else sp

        # Public moves: those recorded in protocol memory OR whose PP has been spent.
        mvs_set: list[str] = []
        for key in (mon_sp, sp, base):
            for mid in mem_moves.get(key) or ():
                m_id = to_id(mid)
                if m_id and m_id not in mvs_set:
                    mvs_set.append(m_id)
        if mon is not None:
            for mid, mv_obj in (getattr(mon, "moves", None) or {}).items():
                cur_pp = getattr(mv_obj, "current_pp", None)
                max_pp = getattr(mv_obj, "max_pp", None)
                if isinstance(cur_pp, int) and isinstance(max_pp, int) and max_pp > 0 and cur_pp < max_pp:
                    m_id = to_id(mid)
                    if m_id and m_id not in mvs_set:
                        mvs_set.append(m_id)

        # Public revealed status: active, fainted, damaged, statused, used a move, or in memory.
        is_rev = bool(
            mon_sp in mem_appeared
            or sp in mem_appeared
            or base in mem_appeared
            or bool(mvs_set)
        )
        if mon is not None and not is_rev:
            hp_frac = getattr(mon, "current_hp_fraction", 1.0)
            if (
                getattr(mon, "active", False)
                or getattr(mon, "fainted", False)
                or (isinstance(hp_frac, (int, float)) and hp_frac < 1.0)
                or getattr(mon, "status", None) is not None
            ):
                is_rev = True

        # Public item: only if still held AND revealed in protocol memory (or Mega-evolved stone).
        if mon is not None and normalize_item(getattr(mon, "item", None)) is None:
            it = None
        else:
            it = normalize_item(mem_items.get(mon_sp) or mem_items.get(sp) or mem_items.get(base))
            if not it and mon_sp:
                sp_entry = load_species().get(mon_sp) or {}
                if sp_entry.get("isMega") and sp_entry.get("requiredItem"):
                    it = to_id(sp_entry.get("requiredItem"))

        # Public ability: check current forme first (e.g. Salamence-Mega's Aerilate) before
        # falling back to pre-Mega base species ability in protocol memory.
        ab = to_id(mem_abilities.get(mon_sp))
        if not ab and mon_sp:
            ab = _single_legal_ability(mon_sp)
        if not ab:
            ab = to_id(mem_abilities.get(sp) or mem_abilities.get(base))

        items.append(it or None)
        abilities.append(ab or None)
        moves.append(mvs_set)
        revealed.append(is_rev)
    return items, abilities, moves, revealed


@dataclass
class RustWorldRoot:
    battle: pikapro.Battle
    our_side_idx: int
    opp_side_idx: int
    foe_items: list[str | None]
    foe_abilities: list[str | None]
    foe_moves: list[list[str]]
    foe_revealed: list[bool] = field(default_factory=list)
    our_public_items: list[str | None] = field(default_factory=list)
    our_public_abilities: list[str | None] = field(default_factory=list)
    our_public_moves: list[list[str]] = field(default_factory=list)
    our_revealed: list[bool] = field(default_factory=list)

    def encode_obs(self, clone: pikapro.Battle, side_idx: int) -> np.ndarray:
        """Encode ``clone`` from ``side_idx``'s fog-safe perspective."""
        if side_idx == self.our_side_idx:
            return clone.encode_state_rich(
                side_idx,
                foe_known_items=self.foe_items,
                foe_known_abilities=self.foe_abilities,
                foe_known_moves=self.foe_moves,
                foe_revealed=self.foe_revealed or None,
            )
        return clone.encode_state_rich(
            side_idx,
            foe_known_items=self.our_public_items or None,
            foe_known_abilities=self.our_public_abilities or None,
            foe_known_moves=self.our_public_moves or None,
            foe_revealed=self.our_revealed or None,
        )


class RustExactMirror:
    """Zero-Node.js public-state mirror backed directly by Rust `pikapro.Battle.patch_public`."""

    def __init__(self, own_packed_team: str, config: PolicyConfig | None = None) -> None:
        self.own_packed_team = own_packed_team
        self.config = config or PolicyConfig()
        self.last_hypothesis_audit: dict[str, Any] = {}

    def hypotheses(
        self, battle, memory=None, *, worlds: int | None = None
    ) -> list[MirrorHypothesis]:
        target_worlds = int(worlds) if worlds is not None else int(self.config.exact_search_total_hypotheses)
        # Scale per-axis hypothesis caps so larger world budgets (e.g. worlds=8..10) can
        # actually populate diverse spread/set/bring worlds instead of capping at 2x2x2=8
        # (which collapses to <=4 once 4 opponent species are revealed).
        spread_cap = max(self.config.exact_search_spread_hypotheses, 3 if target_worlds >= 6 else 2)
        set_cap = max(self.config.exact_search_set_hypotheses, 3 if target_worlds >= 6 else 2)
        bring_cap = max(self.config.exact_search_bring_hypotheses, 3 if target_worlds >= 8 else 2)
        cfg = dataclasses.replace(
            self.config,
            exact_search_total_hypotheses=target_worlds,
            exact_search_spread_hypotheses=spread_cap,
            exact_search_set_hypotheses=set_cap,
            exact_search_bring_hypotheses=bring_cap,
        )
        hyps, audit, _ = compute_mirror_hypotheses(battle, cfg, memory=memory)
        self.last_hypothesis_audit = dict(audit)
        return hyps

    def build(
        self, battle, hypothesis: MirrorHypothesis | None = None, *, memory: Any = None
    ) -> RustWorldRoot:
        side = mirror_side(battle)
        our_idx = 0 if side == "p1" else 1
        opp_idx = 1 - our_idx
        opp_sets = _opponent_sets(
            battle,
            self.config,
            getattr(hypothesis, "spreads", None),
            getattr(hypothesis, "sets", None),
            getattr(hypothesis, "brought", None),
        )
        opp_packed = _pack_sets(opp_sets)
        teams = [self.own_packed_team, opp_packed] if our_idx == 0 else [opp_packed, self.own_packed_team]
        rust_battle = pikapro.Battle("championsvgc", teams, seed=[1, 2, 3, 4], log=False)
        own_preview = _team_preview_order(self.own_packed_team, battle, self.config)
        rust_battle.submit(our_idx, own_preview)
        rust_battle.submit(opp_idx, "team 1234")
        snap = dataclasses.asdict(
            snapshot_battle(
                battle,
                opponent_mega_unknown=self.config.exact_mirror_opponent_mega,
                champions_pp=self.config.exact_mirror_champions_pp,
            )
        )
        patch_dict = _build_rust_patch_dict(battle, snap, side, hypothesis, memory=memory)
        rust_battle.patch_public(patch_dict)
        foe_items, foe_abilities, foe_moves, foe_revealed = _foe_public_knowledge(
            battle, opp_sets, memory=memory
        )
        our_items, our_abilities, our_moves, our_revealed = _our_public_knowledge(
            battle, self.own_packed_team, memory=memory
        )
        return RustWorldRoot(
            battle=rust_battle,
            our_side_idx=our_idx,
            opp_side_idx=opp_idx,
            foe_items=foe_items,
            foe_abilities=foe_abilities,
            foe_moves=foe_moves,
            foe_revealed=foe_revealed,
            our_public_items=our_items,
            our_public_abilities=our_abilities,
            our_public_moves=our_moves,
            our_revealed=our_revealed,
        )

    def close(self) -> None:
        pass


def _extract_rust_request_meta(req: Any) -> tuple[list[list[str]], list[str]]:
    """Extract JSON-serializable `(slot_moves, team_species)` from a `pikapro.RequestData`."""
    team_species = [to_id(getattr(p, "species", "")) for p in (getattr(req, "pokemon", None) or [])]
    slot_moves: list[list[str]] = []
    kind = getattr(req, "kind", None)
    active_list = getattr(kind, "active", None)
    if active_list:
        for act in active_list:
            mvs = getattr(act, "moves", None)
            val = getattr(mvs, "value", None) if mvs is not None else None
            if isinstance(val, str):
                # Locked rampage / two-turn move: `Locked(value="Outrage")` / `Locked(value="Fly")`
                slot_moves.append([to_id(val)])
            elif val is not None:
                slot_moves.append([to_id(getattr(m, "id", "")) for m in val])
            else:
                # Unit variants such as `"Recharge"` or `"Struggle"`
                mvs_str = to_id(str(mvs or ""))
                if mvs_str in ("recharge", "struggle"):
                    slot_moves.append([mvs_str])
                else:
                    slot_moves.append([])
    return slot_moves, team_species


def _extract_battle_request_meta(battle: Any) -> tuple[list[list[str]], list[str]]:
    """Extract `(slot_moves, team_species)` from poke-env's `battle.last_request`."""
    req = getattr(battle, "last_request", None) or {}
    side_mons = (req.get("side") or {}).get("pokemon") or []
    team_species = [
        to_id(str(m.get("details") or "").split(",", 1)[0]) for m in side_mons
    ]
    slot_moves: list[list[str]] = []
    for act in req.get("active") or []:
        if isinstance(act, dict):
            slot_moves.append([to_id(mv.get("id") or mv.get("move") or "") for mv in (act.get("moves") or [])])
        else:
            slot_moves.append([])
    return slot_moves, team_species


def _parse_pikapro_slot(
    part: str,
    slot_idx: int,
    slot_moves: list[list[str]],
    team_species: list[str],
) -> tuple[Any, ...]:
    tokens = part.strip().lower().split()
    if not tokens or tokens[0] in ("pass", "default"):
        return ("pass",)
    if tokens[0] == "switch" and len(tokens) >= 2:
        if tokens[1].isdigit():
            idx = int(tokens[1]) - 1
            if 0 <= idx < len(team_species):
                return ("switch", team_species[idx])
        return ("switch", to_id(" ".join(tokens[1:])))
    if tokens[0] == "move" and len(tokens) >= 2:
        move_tok = tokens[1]
        move_id = move_tok
        if move_tok.isdigit():
            m_idx = int(move_tok) - 1
            if slot_idx < len(slot_moves) and 0 <= m_idx < len(slot_moves[slot_idx]):
                move_id = slot_moves[slot_idx][m_idx]
        target = 0
        mega = False
        for extra in tokens[2:]:
            if extra == "mega":
                mega = True
            else:
                try:
                    target = int(extra)
                except ValueError:
                    pass
        return ("move", to_id(move_id), target, mega)
    return ("raw", part.strip().lower())


def _canonical_single_order(single: Any) -> tuple[Any, ...]:
    if single is None:
        return ("pass",)
    order = getattr(single, "order", None)
    if order is None:
        return ("pass",)
    if hasattr(order, "species"):
        return ("switch", to_id(getattr(order, "species", "")))
    if hasattr(order, "id"):
        target = int(getattr(single, "move_target", 0) or 0)
        mega = bool(getattr(single, "mega", False))
        return ("move", to_id(getattr(order, "id", "")), target, mega)
    return ("pass",)


def match_pikapro_choice_to_order(
    choice_str: str,
    battle: Any,
    legal_orders: list[Any],
    *,
    slot_moves: list[list[str]] | None = None,
    team_species: list[str] | None = None,
) -> Any | None:
    """Resolve a `pikapro` choice string (e.g. ``'move 1 1 mega, switch 3'``) to its
    exact matching `poke-env` `DoubleBattleOrder` from ``legal_orders``.
    """
    if not choice_str or not legal_orders:
        return None
    if slot_moves is None or team_species is None:
        b_moves, b_species = _extract_battle_request_meta(battle)
        if slot_moves is None:
            slot_moves = b_moves
        if team_species is None:
            team_species = b_species

    parts = [p.strip() for p in choice_str.split(",")]
    while len(parts) < 2:
        parts.append("pass")
    sig = (
        _parse_pikapro_slot(parts[0], 0, slot_moves, team_species),
        _parse_pikapro_slot(parts[1], 1, slot_moves, team_species),
    )

    for order in legal_orders:
        o_sig = (
            _canonical_single_order(getattr(order, "first_order", None)),
            _canonical_single_order(getattr(order, "second_order", None)),
        )
        if o_sig == sig:
            return order

    # Secondary check: match base species for form-changed switches (e.g. Mega / regional forms)
    # and match targetless locked moves (Recharge / Struggle / Outrage / Fly / Solar Beam /
    # Electro Shot) where pikapro emits `move 1` (target=0) while poke-env attaches a target.
    def _loose_slot(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
        if a == b:
            return True
        if a[0] == "switch" and b[0] == "switch":
            return _base_species_id(a[1]) == _base_species_id(b[1])
        if (
            a[0] == "move"
            and b[0] == "move"
            and a[1] == b[1]
            and a[3] == b[3]
            and (b[2] == 0 or a[2] == 0)
        ):
            return True
        return False

    for order in legal_orders:
        o_sig = (
            _canonical_single_order(getattr(order, "first_order", None)),
            _canonical_single_order(getattr(order, "second_order", None)),
        )
        if _loose_slot(o_sig[0], sig[0]) and _loose_slot(o_sig[1], sig[1]):
            return order
    return None


def _switch_row_maps(
    rust_battle: pikapro.Battle, side_idx: int, obs: np.ndarray
) -> tuple[dict[int, int], dict[int, int]]:
    """Build `(req_pos -> obs_row, obs_row -> req_pos)` for switch indexing.

    `Battle.slot_mask` indexes switches by Showdown request position (`req.pokemon`),
    whereas `VecEnv` and `ChannelSession` index switches by the permanent 0..5 row
    index in the `rich` observation (`own0..own5`). Remapping keeps the neural policy's
    switch logits 100% aligned across arbitrary team-preview orders and mid-game switches.
    """
    req = rust_battle.request(side_idx)
    sp_names = pikapro.names("species", "championsvgc")
    obs_rows_by_sp: dict[str, list[int]] = {}
    for row in range(6):
        sp_id = int(obs[row * RICH_MON_WIDTH])
        if 1 <= sp_id <= len(sp_names):
            key = to_id(sp_names[sp_id - 1])
            obs_rows_by_sp.setdefault(key, []).append(row)

    req_to_obs: dict[int, int] = {}
    obs_to_req: dict[int, int] = {}
    used_rows: set[int] = set()
    for req_pos, mon in enumerate(getattr(req, "pokemon", None) or []):
        key = to_id(getattr(mon, "species", ""))
        matched_row: int | None = None
        for r in obs_rows_by_sp.get(key, ()):
            if r not in used_rows:
                matched_row = r
                break
        if matched_row is None:
            # Base-species match (e.g. Mega forme vs base species; never raw prefix match)
            req_base = _base_species_id(key)
            for sp_k, rows in obs_rows_by_sp.items():
                if _base_species_id(sp_k) == req_base:
                    for r in rows:
                        if r not in used_rows:
                            matched_row = r
                            break
                    if matched_row is not None:
                        break
        if matched_row is not None:
            used_rows.add(matched_row)
            req_to_obs[req_pos] = matched_row
            obs_to_req[matched_row] = req_pos
    return req_to_obs, obs_to_req


def _factored_mask_for_battle(
    rust_battle: pikapro.Battle,
    side_idx: int,
    space: pikapro.ActionSpace,
    obs: np.ndarray,
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Build `(factored_mask, min_switches, obs_to_req)` matching `VecEnv(masks="factored")`."""
    slot_masks = rust_battle.slot_mask(side_idx)  # (2, slot_size)
    req_to_obs, obs_to_req = _switch_row_maps(rust_battle, side_idx, obs)
    s = space.slot_size
    sw_base = space.switch_base
    pass_idx = space.pass_index
    width = 2 * s + space.preview_size
    out = np.zeros(width, dtype=bool)
    for slot in range(2):
        dst = slot * s
        # Copy move bits and pass bit directly; remap switch bits req_pos -> obs_row.
        out[dst : dst + sw_base] = slot_masks[slot, :sw_base]
        out[dst + pass_idx : dst + s] = slot_masks[slot, pass_idx:s]
        for req_pos in range(pass_idx - sw_base):
            if slot_masks[slot, sw_base + req_pos]:
                obs_row = req_to_obs.get(req_pos, req_pos)
                if 0 <= obs_row < (pass_idx - sw_base):
                    out[dst + sw_base + obs_row] = True

    req_kind = rust_battle.request_kind(side_idx)
    min_sw = 0
    if req_kind == "force_switch":
        req = rust_battle.request(side_idx)
        kind_obj = getattr(req, "kind", None)
        fs = (
            getattr(kind_obj, "slots", None)
            or getattr(kind_obj, "force_switch", None)
            or getattr(req, "force_switch", None)
            or []
        )
        min_sw = sum(1 for x in fs if x)
        s0_sw = int(out[sw_base:pass_idx].sum())
        s1_sw = int(out[s + sw_base : s + pass_idx].sum())
        min_sw = min(min_sw, max(s0_sw, s1_sw))
    return out, min_sw, obs_to_req


def _choice_str_from_slots(
    a0: int, a1: int, space: pikapro.ActionSpace, obs_to_req: dict[int, int]
) -> str:
    sw_base = space.switch_base
    pass_idx = space.pass_index

    def _remap(a: int) -> int:
        if sw_base <= a < pass_idx:
            obs_row = a - sw_base
            req_pos = obs_to_req.get(obs_row, obs_row)
            return sw_base + req_pos
        return a

    s0 = space.decode_slot(_remap(a0))
    s1 = space.decode_slot(_remap(a1))
    choice = pikapro.Choice.from_actions([s0, s1])
    return str(choice)


def _batch_resolve_forced_switches(
    clones: list[pikapro.Battle],
    start_turns: list[int],
    root: RustWorldRoot | Sequence[RustWorldRoot],
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
) -> None:
    """Resolve all mid-turn forced switches (KOs, Parting Shot, U-turn, etc.) via batched GPU policy."""
    import torch

    if not clones:
        return
    roots: Sequence[RustWorldRoot] = (
        [root] * len(clones) if isinstance(root, RustWorldRoot) else root
    )
    s = policy.slot_size
    dev = policy.device

    for _ in range(8):
        jobs: list[tuple[int, int, dict[int, int]]] = []
        obs_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        ms_list: list[int] = []
        for ci, (clone, st_turn) in enumerate(zip(clones, start_turns, strict=True)):
            if clone.over or clone.turn > st_turn:
                continue
            r = roots[ci]
            for s_idx in (r.our_side_idx, r.opp_side_idx):
                if clone.request_kind(s_idx) != "force_switch":
                    continue
                choices = clone.legal_choices(s_idx)
                if len(choices) <= 1:
                    if choices:
                        clone.submit(s_idx, choices[0])
                    continue
                obs = r.encode_obs(clone, s_idx)
                fmask, min_sw, obs_to_req = _factored_mask_for_battle(clone, s_idx, space, obs)
                jobs.append((ci, s_idx, obs_to_req))
                obs_list.append(obs)
                mask_list.append(fmask)
                ms_list.append(min_sw)
        if not jobs:
            # Check if single-choice submits triggered another wave of forced switches.
            still_pending = any(
                (
                    not c.over
                    and c.turn <= st
                    and (
                        c.request_kind(roots[idx].our_side_idx) == "force_switch"
                        or c.request_kind(roots[idx].opp_side_idx) == "force_switch"
                    )
                )
                for idx, (c, st) in enumerate(zip(clones, start_turns, strict=True))
            )
            if not still_pending:
                break
            continue

        obs_np = np.stack(obs_list, axis=0)
        masks_np = np.stack(mask_list, axis=0)
        ms_np = np.asarray(ms_list, dtype=np.uint8)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(dev)
            enc = policy.encode_board(x)
            m0_t = torch.from_numpy(np.ascontiguousarray(masks_np[:, :s], dtype=bool)).to(dev)
            l0 = policy.slot0_logits(enc).masked_fill(~m0_t, -1e9)
            a0_t = l0.argmax(dim=-1)
            a0_np = a0_t.cpu().numpy()

            m1_np = space.pair_mask(masks_np, ms_np, a0_np)
            m1_t = torch.from_numpy(np.ascontiguousarray(m1_np, dtype=bool)).to(dev)
            l1 = policy.slot1_logits(enc, a0_t).masked_fill(~m1_t, -1e9)
            a1_np = l1.argmax(dim=-1).cpu().numpy()

        for (ci, s_idx, obs_to_req), a0, a1 in zip(jobs, a0_np.tolist(), a1_np.tolist(), strict=True):
            clone = clones[ci]
            if clone.over or clone.request_kind(s_idx) != "force_switch":
                continue
            cstr = _choice_str_from_slots(int(a0), int(a1), space, obs_to_req)
            try:
                clone.submit(s_idx, cstr)
            except ValueError:
                fallback = clone.legal_choices(s_idx)
                if fallback:
                    clone.submit(s_idx, fallback[0])


def _batch_step_policy_rollout(
    clones: list[pikapro.Battle],
    root: RustWorldRoot | Sequence[RustWorldRoot],
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
) -> None:
    """Advance a batch of non-terminal battles by 1 full turn using batched GPU policy."""
    import torch

    if not clones:
        return
    roots: Sequence[RustWorldRoot] = (
        [root] * len(clones) if isinstance(root, RustWorldRoot) else root
    )
    s = policy.slot_size
    dev = policy.device

    # Collect (clone_idx, side_idx, obs, fmask, min_sw, obs_to_req) for every acting side.
    jobs: list[tuple[int, int, dict[int, int]]] = []
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    ms_list: list[int] = []
    for ci, clone in enumerate(clones):
        if clone.over:
            continue
        r = roots[ci]
        for s_idx in (r.our_side_idx, r.opp_side_idx):
            rk = clone.request_kind(s_idx)
            if rk == "wait":
                continue
            obs = r.encode_obs(clone, s_idx)
            fmask, min_sw, obs_to_req = _factored_mask_for_battle(clone, s_idx, space, obs)
            jobs.append((ci, s_idx, obs_to_req))
            obs_list.append(obs)
            mask_list.append(fmask)
            ms_list.append(min_sw)

    if not jobs:
        return

    obs_np = np.stack(obs_list, axis=0)
    masks_np = np.stack(mask_list, axis=0)
    ms_np = np.asarray(ms_list, dtype=np.uint8)

    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(dev)
        enc = policy.encode_board(x)
        m0_t = torch.from_numpy(np.ascontiguousarray(masks_np[:, :s], dtype=bool)).to(dev)
        l0 = policy.slot0_logits(enc).masked_fill(~m0_t, -1e9)
        a0_t = l0.argmax(dim=-1)
        a0_np = a0_t.cpu().numpy()

        m1_np = space.pair_mask(masks_np, ms_np, a0_np)
        m1_t = torch.from_numpy(np.ascontiguousarray(m1_np, dtype=bool)).to(dev)
        l1 = policy.slot1_logits(enc, a0_t).masked_fill(~m1_t, -1e9)
        a1_np = l1.argmax(dim=-1).cpu().numpy()

    # Group choices per clone and submit 1 full turn.
    per_clone: dict[int, dict[int, str]] = {}
    for (ci, s_idx, obs_to_req), a0, a1 in zip(jobs, a0_np.tolist(), a1_np.tolist(), strict=True):
        cstr = _choice_str_from_slots(int(a0), int(a1), space, obs_to_req)
        per_clone.setdefault(ci, {})[s_idx] = cstr

    stepped_clones: list[pikapro.Battle] = []
    stepped_turns: list[int] = []
    stepped_roots: list[RustWorldRoot] = []
    for ci, side_choices in per_clone.items():
        clone = clones[ci]
        if clone.over:
            continue
        r = roots[ci]
        start_turn = clone.turn
        for s_idx in (r.our_side_idx, r.opp_side_idx):
            if clone.over or clone.request_kind(s_idx) == "wait":
                continue
            cstr = side_choices.get(s_idx)
            if cstr:
                try:
                    clone.submit(s_idx, cstr)
                except ValueError:
                    fallback = clone.legal_choices(s_idx)
                    if fallback:
                        clone.submit(s_idx, fallback[0])
        stepped_clones.append(clone)
        stepped_turns.append(start_turn)
        stepped_roots.append(r)
    _batch_resolve_forced_switches(stepped_clones, stepped_turns, stepped_roots, policy, space)


def select_search_action_index(
    result: ReplicaSearchResult[Any],
    rng: np.random.Generator,
    *,
    min_mix_prob: float = 0.10,
    deterministic_threshold: float = 0.85,
) -> int:
    """Choose a root action index from regret-matching probabilities without tail noise.

    Regret matching over finite iterations can leave 2-6% residual probability on a
    strictly dominated move. Pruning sub-`min_mix_prob` tails and taking the argmax when
    one line dominates (`>= deterministic_threshold`) preserves genuine mixed bluffs while
    eliminating random tail blunders.
    """
    probs = np.asarray(result.probabilities, dtype=np.float64)
    if len(probs) <= 1:
        return 0
    best = int(np.argmax(probs))
    if float(probs[best]) >= deterministic_threshold:
        return best
    kept = np.where(probs >= min_mix_prob, probs, 0.0)
    total = float(kept.sum())
    if total <= 0.0 or int(np.count_nonzero(kept)) <= 1:
        return best
    kept /= total
    return int(rng.choice(len(kept), p=kept))


def _batch_top_joint_actions(
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
    obs_np: np.ndarray,
    masks_np: np.ndarray,
    ms_np: np.ndarray,
    *,
    limit: int = 2,
) -> list[tuple[list[tuple[int, int]], np.ndarray]]:
    """Batched GPU exact top-K joint `(a0, a1)` ranking across N observations at once."""
    import torch

    n = len(obs_np)
    if n == 0:
        return []
    s = policy.slot_size
    dev = policy.device
    k0 = max(limit, 3)

    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(dev)
        enc = policy.encode_board(x)
        m0_t = torch.from_numpy(np.ascontiguousarray(masks_np[:, :s], dtype=bool)).to(dev)
        l0_t = policy.slot0_logits(enc).masked_fill(~m0_t, -1e9)
        lp0_t = torch.log_softmax(l0_t, dim=-1)
        lp0_np = lp0_t.cpu().numpy()

        # Collect top-k0 legal a0 candidates across all N rows into one batched slot1 call.
        flat_row: list[int] = []
        flat_a0: list[int] = []
        flat_lp0: list[float] = []
        for i in range(n):
            legal_a0 = np.flatnonzero(masks_np[i, :s])
            if len(legal_a0) == 0:
                continue
            top_a0 = legal_a0[np.argsort(-lp0_np[i, legal_a0], kind="stable")][:k0]
            for a0_val in top_a0.tolist():
                flat_row.append(i)
                flat_a0.append(int(a0_val))
                flat_lp0.append(float(lp0_np[i, a0_val]))

        if not flat_row:
            return [([], np.empty(0, dtype=np.float64)) for _ in range(n)]

        row_arr = np.asarray(flat_row, dtype=np.int64)
        a0_arr = np.asarray(flat_a0, dtype=np.int64)
        m1_arr = space.pair_mask(masks_np[row_arr], ms_np[row_arr], a0_arr)

        row_t = torch.from_numpy(row_arr).to(dev)
        a0_t = torch.from_numpy(a0_arr).to(dev)
        m1_t = torch.from_numpy(np.ascontiguousarray(m1_arr, dtype=bool)).to(dev)
        l1_t = policy.slot1_logits(enc.index(row_t), a0_t).masked_fill(~m1_t, -1e9)
        lp1_np = torch.log_softmax(l1_t, dim=-1).cpu().numpy()

    per_row: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]
    for flat_i, (ri, a0_val, lp0_val) in enumerate(zip(flat_row, flat_a0, flat_lp0, strict=True)):
        legal_a1 = np.flatnonzero(m1_arr[flat_i])
        if len(legal_a1) == 0:
            continue
        top_a1 = legal_a1[np.argsort(-lp1_np[flat_i, legal_a1], kind="stable")][:limit]
        for a1_val in top_a1.tolist():
            per_row[ri].append((lp0_val + float(lp1_np[flat_i, a1_val]), a0_val, int(a1_val)))

    out: list[tuple[list[tuple[int, int]], np.ndarray]] = []
    for scored in per_row:
        scored.sort(key=lambda t: -t[0])
        top = scored[:limit]
        if not top:
            out.append(([], np.empty(0, dtype=np.float64)))
            continue
        lps = np.asarray([t[0] for t in top], dtype=np.float64)
        lps -= float(np.max(lps))
        probs = np.exp(lps)
        probs /= float(np.sum(probs))
        out.append(([(t[1], t[2]) for t in top], probs))
    return out


def _solve_2x2_zero_sum(mat: list[list[float]]) -> float:
    """Zero-sum Nash equilibrium value of a small simultaneous payoff matrix."""
    m = len(mat)
    if m == 0 or not mat[0]:
        return 0.0
    n = len(mat[0])
    if m == 1:
        return min(mat[0])
    if n == 1:
        return max(row[0] for row in mat)
    if m > 2 or n > 2:
        from vgc.rl.replica_search import _solve

        arr = np.asarray(mat, dtype=float)
        _, _, val, _ = _solve([arr], np.array([1.0], dtype=float), iterations=200)
        return float(val)
    a, b = mat[0][0], mat[0][1]
    c, d = mat[1][0], mat[1][1]
    maximin = max(min(a, b), min(c, d))
    minimax = min(max(a, c), max(b, d))
    if maximin >= minimax - 1e-9:
        return 0.5 * (maximin + minimax)
    denom = a - b - c + d
    if abs(denom) < 1e-12:
        return 0.5 * (maximin + minimax)
    v = (a * d - b * c) / denom
    return max(maximin, min(minimax, v))


def _eval_turn2_minigame(
    turn1_clones: list[pikapro.Battle],
    root: RustWorldRoot,
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
    *,
    branch: int = 2,
    future_seed: tuple[int, int, int, int] = (7, 13, 29, 53),
) -> list[float]:
    """Evaluate Turn-1 post-exchange clones via a batched `branch x branch` Turn-2 Nash game."""
    if not turn1_clones:
        return []
    our_idx = root.our_side_idx
    opp_idx = root.opp_side_idx

    # 1. Batch-collect every acting side across all Turn-1 clones into ONE GPU forward pass!
    jobs: list[tuple[int, int, dict[int, int]]] = []
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    ms_list: list[int] = []
    for ci, clone in enumerate(turn1_clones):
        if clone.over:
            continue
        for s_idx in (our_idx, opp_idx):
            if clone.request_kind(s_idx) == "wait":
                continue
            obs = root.encode_obs(clone, s_idx)
            fmask, min_sw, obs_to_req = _factored_mask_for_battle(clone, s_idx, space, obs)
            jobs.append((ci, s_idx, obs_to_req))
            obs_list.append(obs)
            mask_list.append(fmask)
            ms_list.append(min_sw)

    batched_top = (
        _batch_top_joint_actions(
            policy,
            space,
            np.stack(obs_list, axis=0),
            np.stack(mask_list, axis=0),
            np.asarray(ms_list, dtype=np.uint8),
            limit=branch,
        )
        if jobs
        else []
    )
    ranked_by_side: dict[tuple[int, int], list[str]] = {}
    for (ci, s_idx, obs_to_req), (pairs, _probs) in zip(jobs, batched_top, strict=True):
        clone = turn1_clones[ci]
        legal_strs = {str(c) for c in clone.legal_choices(s_idx)}
        chosen: list[str] = []
        for a0, a1 in pairs:
            cstr = _choice_str_from_slots(a0, a1, space, obs_to_req)
            if cstr in legal_strs and cstr not in chosen:
                chosen.append(cstr)
        if not chosen:
            chosen = [str(c) for c in clone.legal_choices(s_idx)[:branch]] or ["<wait>"]
        ranked_by_side[(ci, s_idx)] = chosen

    leaf_actions: list[tuple[list[str], list[str]]] = []
    forked_clones: list[pikapro.Battle] = []
    forked_start_turns: list[int] = []
    forked_meta: list[tuple[int, int, int]] = []  # (clone_idx, a_idx, b_idx)

    for ci, clone in enumerate(turn1_clones):
        if clone.over:
            leaf_actions.append(([], []))
            continue
        our_cands = ranked_by_side.get((ci, our_idx), ["<wait>"])
        opp_cands = ranked_by_side.get((ci, opp_idx), ["<wait>"])
        leaf_actions.append((our_cands, opp_cands))
        for ai, oc in enumerate(our_cands):
            for bi, ec in enumerate(opp_cands):
                sub = clone.fork(seed=future_seed)
                st_turn = sub.turn
                if oc != "<wait>" and sub.request_kind(our_idx) != "wait":
                    try:
                        sub.submit(our_idx, oc)
                    except ValueError:
                        fb = sub.legal_choices(our_idx)
                        if fb:
                            sub.submit(our_idx, fb[0])
                if not sub.over and ec != "<wait>" and sub.request_kind(opp_idx) != "wait":
                    try:
                        sub.submit(opp_idx, ec)
                    except ValueError:
                        fb = sub.legal_choices(opp_idx)
                        if fb:
                            sub.submit(opp_idx, fb[0])
                forked_clones.append(sub)
                forked_start_turns.append(st_turn)
                forked_meta.append((ci, ai, bi))

    _batch_resolve_forced_switches(forked_clones, forked_start_turns, root, policy, space)

    # Batch-score all non-terminal Turn-2 leaves in one GPU pass.
    vals_by_clone: dict[int, dict[tuple[int, int], float]] = {}
    eval_obs: list[np.ndarray] = []
    eval_keys: list[tuple[int, int, int]] = []
    for (ci, ai, bi), sub in zip(forked_meta, forked_clones, strict=True):
        if sub.over:
            v = 0.0 if sub.winner is None else (1.0 if sub.winner == our_idx else -1.0)
            vals_by_clone.setdefault(ci, {})[(ai, bi)] = v
        else:
            eval_obs.append(root.encode_obs(sub, our_idx))
            eval_keys.append((ci, ai, bi))

    if eval_obs:
        pred = policy.values_from_obs(np.stack(eval_obs, axis=0))
        for (ci, ai, bi), v in zip(eval_keys, pred.tolist(), strict=True):
            vals_by_clone.setdefault(ci, {})[(ai, bi)] = float(v)

    out_vals: list[float] = []
    for ci, clone in enumerate(turn1_clones):
        if clone.over:
            out_vals.append(0.0 if clone.winner is None else (1.0 if clone.winner == our_idx else -1.0))
            continue
        our_cands, opp_cands = leaf_actions[ci]
        grid = vals_by_clone.get(ci) or {}
        mat = [
            [grid.get((ai, bi), 0.0) for bi in range(len(opp_cands))]
            for ai in range(len(our_cands))
        ]
        out_vals.append(_solve_2x2_zero_sum(mat))
    return out_vals


def search_public_rust(
    battle,
    packed_team: str,
    policy: RustFactoredPolicy,
    *,
    worlds: int = 8,
    actions: int = 8,
    seeds: int = 4,
    iterations: int = 2000,
    seed: int = 1,
    mirror: RustExactMirror | None = None,
    memory=None,
    worst_case_weight: float = 0.5,
    min_prior_prob: float = 0.01,
    prior_bonus: float = 0.025,
    rollout_turns: int = 1,
    second_turn_top_k: int = 3,
    second_turn_branch: int = 1,
) -> tuple[ReplicaSearchResult[str], dict[str, Any]]:
    """Run fast Rust-backed multi-world lookahead search (1-turn or 1+rollout_turns).

    Parameters
    ----------
    min_prior_prob:
        Minimum policy probability for admitting our own candidate actions beyond the
        top 2 (blocks 0.01% junk Protect/switch moves from exploiting 1-turn value errors).
    prior_bonus:
        Small KL-regularization bonus ``prior_bonus * ln(max(p_i, 1e-4))`` added to our
        row payoffs so search only overturns a strong neural preference when the rollout
        gain clearly outweighs it.
    rollout_turns:
        Extra on-policy batched-GPU turns to simulate past the 1-turn simultaneous matrix
        (``0`` = pure 1-turn lookahead; ``1`` = 2-turn lookahead).

    Returns ``(result, audit)`` where ``result.actions`` are Showdown choice strings
    (e.g. ``"move 1, move 2 1"``).
    """
    if not 1 <= worlds <= 16 or not 1 <= actions <= 8 or seeds < 1:
        raise ValueError("worlds/actions/seeds must be within 1..16/1..8/positive")
    if rollout_turns < 0:
        raise ValueError("rollout_turns must be >= 0")
    config = dataclasses.replace(PolicyConfig(), exact_search_total_hypotheses=worlds)
    space = pikapro.ActionSpace("championsvgc")
    rng = np.random.default_rng(seed)
    future_seeds = [tuple(map(int, rng.integers(0, 2**16, size=4))) for _ in range(seeds)]

    private = {
        key: vars(battle).pop(key)
        for key in ("_vgc_direct_root", "_vgc_direct_side")
        if key in vars(battle)
    }
    if mirror is None:
        mirror = RustExactMirror(packed_team, config)

    try:
        hypotheses = mirror.hypotheses(battle, memory=memory, worlds=worlds)
        world_roots: list[RustWorldRoot] = []
        sampled: list[SampledWorld[RustWorldRoot, tuple[RustWorldRoot, int] | None]] = []
        for hyp in hypotheses:
            root = mirror.build(battle, hyp, memory=memory)
            world_roots.append(root)
            opp_obs = (
                (root, root.opp_side_idx)
                if root.battle.request_kind(root.opp_side_idx) != "wait"
                else None
            )
            sampled.append(SampledWorld(root, opp_obs, hyp.weight))

        scored_priors: dict[int, dict[str, float]] = {}

        def _rank_for_side(root: RustWorldRoot, side_idx: int) -> tuple[list[str], dict[str, float]]:
            if root.battle.request_kind(side_idx) == "wait":
                return ["<wait>"], {"<wait>": 1.0}
            obs = root.encode_obs(root.battle, side_idx)
            fmask, min_sw, obs_to_req = _factored_mask_for_battle(
                root.battle, side_idx, space, obs
            )
            pairs, probs, _ = policy.top_joint_actions(obs, fmask, min_sw, space, limit=actions)
            legal_strs = {str(c) for c in root.battle.legal_choices(side_idx)}
            is_our_root = side_idx == root.our_side_idx
            ranked: list[str] = []
            prior_map: dict[str, float] = {}
            for (a0, a1), prob in zip(pairs, probs, strict=True):
                cstr = _choice_str_from_slots(a0, a1, space, obs_to_req)
                if cstr in legal_strs and cstr not in prior_map:
                    # Prune ultra-low-probability tail moves for our side (keep at least top 2).
                    if is_our_root and min_prior_prob > 0.0 and len(ranked) >= 2 and float(prob) < min_prior_prob:
                        continue
                    ranked.append(cstr)
                    prior_map[cstr] = float(prob)
            if not ranked:
                fallback = [str(c) for c in root.battle.legal_choices(side_idx)[:actions]]
                if not fallback:
                    return ["<wait>"], {"<wait>": 1.0}
                u = 1.0 / len(fallback)
                return fallback, {c: u for c in fallback}
            return ranked, prior_map

        root_obs = (world_roots[0], world_roots[0].our_side_idx)
        our_ranked_list, our_prior_map = _rank_for_side(world_roots[0], world_roots[0].our_side_idx)
        our_orders_tuple = tuple(our_ranked_list[:actions])
        scored_priors[id(root_obs)] = our_prior_map

        cached_orders: dict[int, list[str]] = {id(root_obs): list(our_orders_tuple)}
        opp_orders_by_world: list[tuple[str, ...]] = []
        for sw in sampled:
            obs_handle = sw.opponent_observation
            if obs_handle is None:
                opp_orders_by_world.append(("<wait>",))
                continue
            r_w, s_w = obs_handle
            r_ranked, r_pmap = _rank_for_side(r_w, s_w)
            r_trimmed = r_ranked[:actions]
            cached_orders[id(obs_handle)] = r_trimmed
            scored_priors[id(obs_handle)] = r_pmap
            opp_orders_by_world.append(tuple(r_trimmed))

        # Pre-evaluate ALL worlds in ONE unified cross-world GPU batch (Turn-1 forced
        # switches, Turn-2 rollouts, and leaf values) instead of W separate small GPU calls.
        cells_by_world: list[list[list[list[float]]]] = [
            [[[] for _ in opp_orders_by_world[wi]] for _ in our_orders_tuple]
            for wi in range(len(world_roots))
        ]
        all_meta: list[tuple[int, int, int]] = []  # (world_idx, i, j)
        all_clones: list[pikapro.Battle] = []
        all_start_turns: list[int] = []
        all_roots: list[RustWorldRoot] = []

        for wi, root in enumerate(world_roots):
            our_idx = root.our_side_idx
            opp_idx = root.opp_side_idx
            for i, our_c in enumerate(our_orders_tuple):
                for j, opp_c in enumerate(opp_orders_by_world[wi]):
                    for fseed in future_seeds:
                        clone = root.battle.fork(seed=fseed)
                        start_turn = clone.turn
                        if our_c != "<wait>" and clone.request_kind(our_idx) != "wait":
                            clone.submit(our_idx, our_c)
                        if not clone.over and opp_c != "<wait>" and clone.request_kind(opp_idx) != "wait":
                            clone.submit(opp_idx, opp_c)
                        all_meta.append((wi, i, j))
                        all_clones.append(clone)
                        all_start_turns.append(start_turn)
                        all_roots.append(root)

        _batch_resolve_forced_switches(all_clones, all_start_turns, all_roots, policy, space)

        saved_turn1_by_world: list[dict[tuple[int, int], list[pikapro.Battle]]] = [
            {} for _ in world_roots
        ]
        need_t2_check = rollout_turns == 1 and second_turn_branch >= 2 and len(our_orders_tuple) >= 2
        pending_meta: list[tuple[int, int, int]] = []
        pending_clones: list[pikapro.Battle] = []
        pending_roots: list[RustWorldRoot] = []

        for (wi, i, j), clone, root in zip(all_meta, all_clones, all_roots, strict=True):
            if clone.over:
                if clone.winner is None:
                    cells_by_world[wi][i][j].append(0.0)
                else:
                    cells_by_world[wi][i][j].append(1.0 if clone.winner == root.our_side_idx else -1.0)
            else:
                if need_t2_check and not saved_turn1_by_world[wi].get((i, j)):
                    saved_turn1_by_world[wi][(i, j)] = [clone.fork(seed=future_seeds[0])]
                pending_meta.append((wi, i, j))
                pending_clones.append(clone)
                pending_roots.append(root)

        # Multi-turn batched GPU rollout across ALL worlds simultaneously.
        for _ in range(rollout_turns):
            if not pending_clones:
                break
            _batch_step_policy_rollout(pending_clones, pending_roots, policy, space)
            next_meta: list[tuple[int, int, int]] = []
            next_clones: list[pikapro.Battle] = []
            next_roots: list[RustWorldRoot] = []
            for (wi, i, j), clone, root in zip(pending_meta, pending_clones, pending_roots, strict=True):
                if clone.over:
                    if clone.winner is None:
                        cells_by_world[wi][i][j].append(0.0)
                    else:
                        cells_by_world[wi][i][j].append(
                            1.0 if clone.winner == root.our_side_idx else -1.0
                        )
                else:
                    next_meta.append((wi, i, j))
                    next_clones.append(clone)
                    next_roots.append(root)
            pending_meta = next_meta
            pending_clones = next_clones
            pending_roots = next_roots

        if pending_clones:
            pending_obs = [
                root.encode_obs(clone, root.our_side_idx)
                for clone, root in zip(pending_clones, pending_roots, strict=True)
            ]
            vals = policy.values_from_obs(np.stack(pending_obs, axis=0))
            for (wi, i, j), v in zip(pending_meta, vals.tolist(), strict=True):
                cells_by_world[wi][i][j].append(float(v))

        # Assemble per-world payoff matrices (with optional Turn-2 Nash refinement & prior bonus).
        precomputed_matrices: dict[int, np.ndarray] = {}
        for wi, root in enumerate(world_roots):
            cells = cells_by_world[wi]
            opp_orders = opp_orders_by_world[wi]
            if need_t2_check and saved_turn1_by_world[wi]:
                refine_clones: list[pikapro.Battle] = []
                refine_keys: list[tuple[int, int]] = []
                for (i, j), c_list in saved_turn1_by_world[wi].items():
                    for c1 in c_list:
                        refine_clones.append(c1)
                        refine_keys.append((i, j))
                if refine_clones:
                    t2_vals = _eval_turn2_minigame(
                        refine_clones,
                        root,
                        policy,
                        space,
                        branch=second_turn_branch,
                        future_seed=future_seeds[0],
                    )
                    by_cell: dict[tuple[int, int], list[float]] = {}
                    for key, val in zip(refine_keys, t2_vals, strict=True):
                        by_cell.setdefault(key, []).append(val)
                    for (i, j), vals_list in by_cell.items():
                        base_mean = float(np.mean(cells[i][j])) if cells[i][j] else 0.0
                        cells[i][j] = [0.5 * base_mean + 0.5 * float(np.mean(vals_list))]

            mat = np.empty((len(our_orders_tuple), len(opp_orders)), dtype=float)
            for i, our_c in enumerate(our_orders_tuple):
                bonus = (
                    prior_bonus * float(np.log(max(our_prior_map.get(our_c, 1e-4), 1e-4)))
                    if prior_bonus > 0.0
                    else 0.0
                )
                for j in range(len(opp_orders)):
                    mat[i, j] = (float(np.mean(cells[i][j])) if cells[i][j] else 0.0) + bonus
            precomputed_matrices[id(root)] = mat

        def select(obs_handle: tuple[RustWorldRoot, int] | None) -> list[str]:
            if obs_handle is None:
                return ["<wait>"]
            cached = cached_orders.get(id(obs_handle))
            if cached is not None:
                return cached
            root, side_idx = obs_handle
            ranked, prior_map = _rank_for_side(root, side_idx)
            scored_priors[id(obs_handle)] = prior_map
            return ranked

        def opp_priors(obs_handle: tuple[RustWorldRoot, int] | None, opp_orders: tuple[str, ...]) -> list[float]:
            if obs_handle is None or len(opp_orders) <= 1:
                return [1.0] * len(opp_orders)
            pmap = scored_priors.get(id(obs_handle)) or {}
            return [max(pmap.get(c, 0.0), 1e-9) for c in opp_orders]

        def eval_matrix(
            root: RustWorldRoot, our_orders: tuple[str, ...], opp_orders: tuple[str, ...]
        ) -> np.ndarray:
            return precomputed_matrices[id(root)]

        result = search_sampled_worlds(
            root_obs,
            sampled,
            select,
            eval_matrix,
            action_key=str,
            max_actions=actions,
            max_worlds=worlds,
            iterations=iterations,
            worst_case_weight=worst_case_weight,
            opponent_priors=opp_priors,
        )
        root_slot_moves, root_team_species = _extract_rust_request_meta(
            world_roots[0].battle.request(world_roots[0].our_side_idx)
        )
        audit = dict(mirror.last_hypothesis_audit)
        audit["root_slot_moves"] = root_slot_moves
        audit["root_team_species"] = root_team_species
        return result, audit
    finally:
        vars(battle).update(private)


_WEATHER_SETTER_ABILITIES: dict[str, str] = {
    "drizzle": "rain",
    "drought": "sun",
    "orichalcumpulse": "sun",
    "sandstream": "sand",
    "snowwarning": "snow",
}
_WEATHER_ABUSER_ABILITIES: dict[str, str] = {
    "swiftswim": "rain",
    "chlorophyll": "sun",
    "sandrush": "sand",
    "slushrush": "snow",
}
_TERRAIN_SETTER_ABILITIES: frozenset[str] = frozenset(
    {"psychicsurge", "grassysurge", "electricsurge", "mistysurge", "hadronengine"}
)


def _detect_team_control_roles(packed_team: str) -> tuple[dict[int, str], dict[int, str]]:
    """Return ``(setter_roles_by_idx, weather_abusers_by_idx)`` purely from dex & team data."""
    from poke_env.teambuilder.teambuilder import Teambuilder
    from vgc.data import load_species

    entries = Teambuilder.parse_packed_team(packed_team)
    species_db = load_species()
    setters: dict[int, str] = {}
    abusers: dict[int, str] = {}
    for idx, entry in enumerate(entries[:6]):
        sp_id = to_id(entry.species or entry.nickname)
        ab_id = to_id(entry.ability)
        item_id = to_id(entry.item)
        moves = {to_id(mv) for mv in (entry.moves or ())}

        # Check both base ability and Mega forme ability if holding its Mega stone.
        abilities_to_check = [ab_id]
        sp_data = species_db.get(sp_id) or {}
        for forme_name in sp_data.get("otherFormes") or ():
            fid = to_id(forme_name)
            fdata = species_db.get(fid) or {}
            if fdata.get("isMega") and to_id(fdata.get("requiredItem")) == item_id:
                mega_ab = to_id((fdata.get("abilities") or {}).get("0"))
                if mega_ab:
                    abilities_to_check.append(mega_ab)

        for ab in abilities_to_check:
            if ab in _WEATHER_SETTER_ABILITIES:
                setters[idx] = f"weather:{_WEATHER_SETTER_ABILITIES[ab]}"
            elif ab in _TERRAIN_SETTER_ABILITIES:
                setters.setdefault(idx, f"terrain:{ab}")
            if ab in _WEATHER_ABUSER_ABILITIES:
                abusers[idx] = _WEATHER_ABUSER_ABILITIES[ab]
        if "trickroom" in moves:
            setters.setdefault(idx, "speed:trickroom")
        elif "tailwind" in moves:
            setters.setdefault(idx, "speed:tailwind")
    return setters, abusers


def _opponent_has_field_engine(opp_species: list[str], config: PolicyConfig) -> bool:
    """True when the opponent's previewed 6 show any weather, terrain, TR, or Tailwind engine."""
    from vgc.data import load_species
    from vgc.sets import set_priors_for

    species_db = load_species()
    priors = (set_priors_for(config) or {}).get("species") or {}
    for sp in opp_species:
        sp_data = species_db.get(sp) or {}
        abs_map = sp_data.get("abilities") or {}
        all_abs = {to_id(a) for a in abs_map.values() if a}
        for forme_name in sp_data.get("otherFormes") or ():
            fdata = species_db.get(to_id(forme_name)) or {}
            if fdata.get("isMega"):
                all_abs.add(to_id((fdata.get("abilities") or {}).get("0")))
        if any(a in _WEATHER_SETTER_ABILITIES or a in _TERRAIN_SETTER_ABILITIES for a in all_abs):
            return True
        p_entry = priors.get(sp) or {}
        top_moves = sorted(
            (p_entry.get("moves") or {}).items(), key=lambda row: (-row[1], row[0])
        )[:6]
        if any(to_id(mv) in ("trickroom", "tailwind", "raindance", "sunnyday", "sandstorm", "snowscape") for mv, _ in top_moves):
            return True
    return False


def _dedup_preview_groups(
    all_indices: list[int],
    all_probs: list[float],
    orders_table: np.ndarray,
) -> list[tuple[int, tuple[int, int, int, int], float]]:
    """Collapse symmetric `(lead2, back2)` permutations and sum their neural probabilities."""
    groups: dict[
        tuple[frozenset[int], frozenset[int]],
        tuple[int, tuple[int, int, int, int], float],
    ] = {}
    for pv_idx, prob in zip(all_indices, all_probs, strict=True):
        row = orders_table[pv_idx]
        order = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        key = (frozenset((order[0], order[1])), frozenset((order[2], order[3])))
        p = float(prob)
        prev = groups.get(key)
        if prev is None:
            groups[key] = (int(pv_idx), order, p)
        else:
            groups[key] = (prev[0], prev[1], prev[2] + p)
    ranked = sorted(groups.values(), key=lambda r: -r[2])
    return ranked


def _select_preview_candidates(
    packed_team: str,
    opp_species: list[str],
    policy: RustFactoredPolicy,
    preview_obs: np.ndarray,
    preview_mask: np.ndarray,
    config: PolicyConfig,
    *,
    candidates: int = 8,
    min_prior_prob: float = 0.04,
) -> list[tuple[int, tuple[int, int, int, int], float, bool]]:
    """Shortlist distinct `(pv_idx, order_0based, agg_prob, injected)` brings with setter guard."""
    all_indices, all_probs = policy.top_preview_indices(preview_obs, preview_mask, limit=360)
    if not all_indices:
        return []
    orders_table = policy.preview_orders.cpu().numpy()
    grouped = _dedup_preview_groups(all_indices, all_probs, orders_table)
    if not grouped:
        return []

    base_k = max(1, candidates - 2)
    selected: list[tuple[int, tuple[int, int, int, int], float, bool]] = []
    seen_pv: set[int] = set()
    # Always keep neural top-1; admit additional distinct brings only if >= min_prior_prob.
    for idx_in_list, (pv_idx, order, agg_prob) in enumerate(grouped):
        if len(selected) >= base_k:
            break
        if idx_in_list > 0 and agg_prob < min_prior_prob:
            break
        selected.append((pv_idx, order, agg_prob, False))
        seen_pv.add(pv_idx)

    # Diversity guard: ensure omitted weather/terrain/speed control setters are tested!
    setters, abusers = _detect_team_control_roles(packed_team)
    opp_engine = _opponent_has_field_engine(opp_species, config)
    needed_setters: list[int] = []
    for s_idx, role in setters.items():
        if all(s_idx not in order for _, order, _, _ in selected):
            if opp_engine and role.startswith(("weather:", "terrain:", "speed:")):
                needed_setters.append(s_idx)
            elif role.startswith("weather:"):
                w_kind = role.split(":", 1)[1]
                # If our top brings include a weather abuser without its setter, test the setter!
                if any(
                    a_idx in order
                    for _, order, _, _ in selected
                    for a_idx, aw in abusers.items()
                    if aw == w_kind
                ):
                    needed_setters.append(s_idx)

    for s_idx in needed_setters:
        if len(selected) >= candidates:
            break
        for pv_idx, order, agg_prob in grouped:
            if pv_idx in seen_pv:
                continue
            if s_idx in order:
                selected.append((pv_idx, order, agg_prob, True))
                seen_pv.add(pv_idx)
                break

    return selected


def _sample_preview_worlds(
    battle,
    packed_team: str,
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
    our_idx: int,
    config: PolicyConfig,
    *,
    worlds: int = 8,
) -> list[tuple[str, str, float]]:
    """Build fog-safe weighted ``(opp_packed, opp_team_cmd, weight)`` preview worlds.

    Combines our neural preview head (evaluated fog-safely on prior-estimated opponent sets)
    with the usage/matchup hybrid predictor so we cover both strong and usage-driven foes.
    """
    from poke_env.teambuilder.teambuilder import Teambuilder
    from vgc.battle_memory import BattleMemory
    from vgc.opponent_belief import build_opponent_beliefs
    from vgc.preview_predict import predict_preview_hybrid

    opp_preview = list(getattr(battle, "teampreview_opponent_team", None) or ())
    opp_species = [to_id(getattr(m, "species", None)) for m in opp_preview if getattr(m, "species", None)]
    our_entries = Teambuilder.parse_packed_team(packed_team)
    our_species = [to_id(e.species or e.nickname) for e in our_entries[:6]]
    if len(opp_species) < 4:
        return []

    # 1. Build 2 coherent fog-safe set/spread variants for opponent team from set_priors.
    beliefs = build_opponent_beliefs(battle, BattleMemory("preview"), config)
    set_variants: list[tuple[str, float]] = []
    primary_sets: dict[str, dict[str, Any]] = {}
    alt_sets: dict[str, dict[str, Any]] = {}
    has_alt = False
    for b in beliefs:
        if b.set_hypotheses:
            s0 = b.set_hypotheses[0]
            primary_sets[b.species_id] = {"moves": s0.moves, "item": s0.item, "ability": s0.ability}
            if len(b.set_hypotheses) >= 2:
                s1 = b.set_hypotheses[1]
                alt_sets[b.species_id] = {"moves": s1.moves, "item": s1.item, "ability": s1.ability}
                has_alt = True
            else:
                alt_sets[b.species_id] = primary_sets[b.species_id]

    opp_packed_primary = _pack_sets(_opponent_sets(battle, config, set_assignments=primary_sets))
    if has_alt:
        opp_packed_alt = _pack_sets(_opponent_sets(battle, config, set_assignments=alt_sets))
        if opp_packed_alt != opp_packed_primary:
            set_variants.append((opp_packed_primary, 0.7))
            set_variants.append((opp_packed_alt, 0.3))
        else:
            set_variants.append((opp_packed_primary, 1.0))
    else:
        set_variants.append((opp_packed_primary, 1.0))

    # 2. Predict opponent distinct (lead2, back2) brings via fog-safe neural preview + hybrid.
    opp_idx = 1 - our_idx
    order_weights: dict[tuple[frozenset[int], frozenset[int]], tuple[str, float]] = {}

    # 2a. Neural opponent preview on prior-estimated opponent sets (75% weight).
    try:
        teams_p = [packed_team, opp_packed_primary] if our_idx == 0 else [opp_packed_primary, packed_team]
        probe = pikapro.Battle("championsvgc", teams_p, seed=(1, 2, 3, 4), log=True)
        opp_obs = probe.encode_rich(opp_idx)
        pv_mask = np.ones(space.preview_size, dtype=bool)
        o_idx, o_probs = policy.top_preview_indices(opp_obs, pv_mask, limit=360)
        orders_table = policy.preview_orders.cpu().numpy()
        o_grouped = _dedup_preview_groups(o_idx, o_probs, orders_table)[:4]
        n_sum = sum(p for _, _, p in o_grouped) or 1.0
        for _, order, p in o_grouped:
            key = (frozenset((order[0], order[1])), frozenset((order[2], order[3])))
            cmd = "team " + "".join(str(i + 1) for i in order)
            order_weights[key] = (cmd, 0.75 * (p / n_sum))
    except Exception:
        pass

    # 2b. Hybrid usage+matchup predictor (25% weight, or 100% if neural probe failed).
    pred_orders = predict_preview_hybrid(opp_species, our_species, config)
    h_weight = 0.25 if order_weights else 1.0
    h_kept: list[tuple[tuple[frozenset[int], frozenset[int]], str, float]] = []
    seen_h: set[tuple[frozenset[int], frozenset[int]]] = set()
    for c in pred_orders or ():
        o = tuple(int(i) for i in c.order[:4])
        if len(o) < 4:
            continue
        key = (frozenset((o[0], o[1])), frozenset((o[2], o[3])))
        if key in seen_h:
            continue
        seen_h.add(key)
        h_kept.append((key, "team " + "".join(str(i + 1) for i in o), max(float(c.probability), 1e-6)))
        if len(h_kept) >= 2:
            break
    if h_kept:
        h_sum = sum(p for _, _, p in h_kept) or 1.0
        for key, cmd, p in h_kept:
            prev = order_weights.get(key)
            add_w = h_weight * (p / h_sum)
            order_weights[key] = (prev[0], prev[1] + add_w) if prev else (cmd, add_w)

    top_cmds = sorted(order_weights.values(), key=lambda r: -r[1])
    if not top_cmds:
        top_cmds = [("team 1234", 1.0)]

    # 3. Combine top opponent preview orders with set variants up to `worlds`.
    combined: list[tuple[str, str, float]] = []
    for opp_cmd, o_prob in top_cmds:
        for opp_packed, s_prob in set_variants:
            combined.append((opp_packed, opp_cmd, o_prob * s_prob))
    combined.sort(key=lambda row: -row[2])
    kept = combined[:worlds]
    total_w = sum(w for _, _, w in kept) or 1.0
    return [(p, cmd, w / total_w) for p, cmd, w in kept]


def _batch_step_log_rollout(
    clones: list[pikapro.Battle],
    our_idx: int,
    opp_idx: int,
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
) -> None:
    """Advance a batch of `log=True` Rust battles by 1 full turn using fog-safe `encode_rich`."""
    import torch

    if not clones:
        return
    s = policy.slot_size
    dev = policy.device

    jobs: list[tuple[int, int, dict[int, int]]] = []
    obs_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    ms_list: list[int] = []
    for ci, clone in enumerate(clones):
        if clone.over:
            continue
        for s_idx in (our_idx, opp_idx):
            if clone.request_kind(s_idx) == "wait":
                continue
            obs = clone.encode_rich(s_idx)
            fmask, min_sw, obs_to_req = _factored_mask_for_battle(clone, s_idx, space, obs)
            jobs.append((ci, s_idx, obs_to_req))
            obs_list.append(obs)
            mask_list.append(fmask)
            ms_list.append(min_sw)
    if not jobs:
        return

    obs_np = np.stack(obs_list, axis=0)
    masks_np = np.stack(mask_list, axis=0)
    ms_np = np.asarray(ms_list, dtype=np.uint8)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(dev)
        enc = policy.encode_board(x)
        m0_t = torch.from_numpy(np.ascontiguousarray(masks_np[:, :s], dtype=bool)).to(dev)
        l0 = policy.slot0_logits(enc).masked_fill(~m0_t, -1e9)
        a0_t = l0.argmax(dim=-1)
        a0_np = a0_t.cpu().numpy()

        m1_np = space.pair_mask(masks_np, ms_np, a0_np)
        m1_t = torch.from_numpy(np.ascontiguousarray(m1_np, dtype=bool)).to(dev)
        l1 = policy.slot1_logits(enc, a0_t).masked_fill(~m1_t, -1e9)
        a1_np = l1.argmax(dim=-1).cpu().numpy()

    per_clone: dict[int, dict[int, str]] = {}
    for (ci, s_idx, obs_to_req), a0, a1 in zip(jobs, a0_np.tolist(), a1_np.tolist(), strict=True):
        per_clone.setdefault(ci, {})[s_idx] = _choice_str_from_slots(int(a0), int(a1), space, obs_to_req)

    stepped: list[pikapro.Battle] = []
    start_turns: list[int] = []
    for ci, side_choices in per_clone.items():
        clone = clones[ci]
        if clone.over:
            continue
        st = clone.turn
        for s_idx in (our_idx, opp_idx):
            if clone.over or clone.request_kind(s_idx) == "wait":
                continue
            cstr = side_choices.get(s_idx)
            if cstr:
                try:
                    clone.submit(s_idx, cstr)
                except ValueError:
                    fb = clone.legal_choices(s_idx)
                    if fb:
                        clone.submit(s_idx, fb[0])
        stepped.append(clone)
        start_turns.append(st)

    # Resolve mid-turn forced switches with batched GPU policy (using fog-safe `encode_rich`).
    for _ in range(8):
        fs_jobs: list[tuple[int, int, dict[int, int]]] = []
        fs_obs: list[np.ndarray] = []
        fs_masks: list[np.ndarray] = []
        fs_ms: list[int] = []
        for ci, (clone, st) in enumerate(zip(stepped, start_turns, strict=True)):
            if clone.over or clone.turn > st:
                continue
            for s_idx in (our_idx, opp_idx):
                if clone.request_kind(s_idx) != "force_switch":
                    continue
                choices = clone.legal_choices(s_idx)
                if len(choices) <= 1:
                    if choices:
                        clone.submit(s_idx, choices[0])
                    continue
                obs = clone.encode_rich(s_idx)
                fmask, min_sw, obs_to_req = _factored_mask_for_battle(clone, s_idx, space, obs)
                fs_jobs.append((ci, s_idx, obs_to_req))
                fs_obs.append(obs)
                fs_masks.append(fmask)
                fs_ms.append(min_sw)
        if not fs_jobs:
            still = any(
                (not c.over and c.turn <= st and (c.request_kind(our_idx) == "force_switch" or c.request_kind(opp_idx) == "force_switch"))
                for c, st in zip(stepped, start_turns, strict=True)
            )
            if not still:
                break
            continue
        obs_np = np.stack(fs_obs, axis=0)
        masks_np = np.stack(fs_masks, axis=0)
        ms_np = np.asarray(fs_ms, dtype=np.uint8)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(dev)
            enc = policy.encode_board(x)
            m0_t = torch.from_numpy(np.ascontiguousarray(masks_np[:, :s], dtype=bool)).to(dev)
            l0 = policy.slot0_logits(enc).masked_fill(~m0_t, -1e9)
            a0_t = l0.argmax(dim=-1)
            a0_np = a0_t.cpu().numpy()
            m1_np = space.pair_mask(masks_np, ms_np, a0_np)
            m1_t = torch.from_numpy(np.ascontiguousarray(m1_np, dtype=bool)).to(dev)
            l1 = policy.slot1_logits(enc, a0_t).masked_fill(~m1_t, -1e9)
            a1_np = l1.argmax(dim=-1).cpu().numpy()
        for (ci, s_idx, obs_to_req), a0, a1 in zip(fs_jobs, a0_np.tolist(), a1_np.tolist(), strict=True):
            clone = stepped[ci]
            if clone.over or clone.request_kind(s_idx) != "force_switch":
                continue
            cstr = _choice_str_from_slots(int(a0), int(a1), space, obs_to_req)
            try:
                clone.submit(s_idx, cstr)
            except ValueError:
                fb = clone.legal_choices(s_idx)
                if fb:
                    clone.submit(s_idx, fb[0])


def search_teampreview_rust(
    battle,
    packed_team: str,
    policy: RustFactoredPolicy,
    preview_obs: np.ndarray,
    preview_mask: np.ndarray,
    *,
    candidates: int = 8,
    worlds: int = 8,
    seeds: int = 3,
    rollout_turns: int = 3,
    worst_case_weight: float = 0.15,
    prior_bonus: float = 0.035,
    min_prior_prob: float = 0.04,
    seed: int = 1,
) -> tuple[str, dict[str, Any]]:
    """Run batched multi-turn Rust lookahead search at Team Preview (fog-safe, regulation-indifferent).

    Returns ``("/team XXXX", audit_dict)``.
    """
    config = PolicyConfig()
    space = pikapro.ActionSpace("championsvgc")
    rng = np.random.default_rng(seed)
    opp_preview = list(getattr(battle, "teampreview_opponent_team", None) or ())
    opp_species = [to_id(getattr(m, "species", None)) for m in opp_preview if getattr(m, "species", None)]

    our_cands = _select_preview_candidates(
        packed_team,
        opp_species,
        policy,
        preview_obs,
        preview_mask,
        config,
        candidates=candidates,
        min_prior_prob=min_prior_prob,
    )
    if not our_cands:
        return "/team 1234", {"fallback": "no_candidates"}

    top_digits = "".join(str(i + 1) for i in our_cands[0][1])
    if len(our_cands) == 1:
        return f"/team {top_digits}", {
            "chosen": top_digits,
            "neural_top1": top_digits,
            "overturned": False,
            "candidates_tested": 1,
            "opp_worlds": 0,
            "top_candidates": [{"order": top_digits, "prior": round(our_cands[0][2], 4)}],
        }

    side = mirror_side(battle)
    our_idx = 0 if side == "p1" else 1
    opp_idx = 1 - our_idx
    opp_worlds = _sample_preview_worlds(
        battle, packed_team, policy, space, our_idx, config, worlds=worlds
    )
    if not opp_worlds:
        return f"/team {top_digits}", {"fallback": "no_opp_worlds"}

    future_seeds = [tuple(map(int, rng.integers(0, 2**16, size=4))) for _ in range(max(1, seeds))]

    clones: list[pikapro.Battle] = []
    meta: list[tuple[int, int]] = []  # (our_cand_i, opp_world_j)
    for ci, (_pv_idx, our_order, _prior, _inj) in enumerate(our_cands):
        our_cmd = "team " + "".join(str(k + 1) for k in our_order)
        for wj, (opp_packed, opp_cmd, _w) in enumerate(opp_worlds):
            teams = [packed_team, opp_packed] if our_idx == 0 else [opp_packed, packed_team]
            for fseed in future_seeds:
                b = pikapro.Battle("championsvgc", teams, seed=fseed, log=True)
                b.submit(our_idx, our_cmd)
                b.submit(opp_idx, opp_cmd)
                clones.append(b)
                meta.append((ci, wj))

    for _ in range(max(1, rollout_turns)):
        active = [b for b in clones if not b.over]
        if not active:
            break
        _batch_step_log_rollout(active, our_idx, opp_idx, policy, space)

    # Batch-evaluate all leaves using fog-safe `encode_rich(our_idx)`.
    cell_vals: dict[tuple[int, int], list[float]] = {}
    eval_obs: list[np.ndarray] = []
    eval_keys: list[tuple[int, int]] = []
    for (ci, wj), b in zip(meta, clones, strict=True):
        if b.over:
            v = 0.0 if b.winner is None else (1.0 if b.winner == our_idx else -1.0)
            cell_vals.setdefault((ci, wj), []).append(v)
        else:
            eval_obs.append(b.encode_rich(our_idx))
            eval_keys.append((ci, wj))

    if eval_obs:
        preds = policy.values_from_obs(np.stack(eval_obs, axis=0))
        for key, v in zip(eval_keys, preds.tolist(), strict=True):
            cell_vals.setdefault(key, []).append(float(v))

    scored_rows: list[dict[str, Any]] = []
    best_i = 0
    best_score = -1e9
    for ci, (pv_idx, our_order, prior, injected) in enumerate(our_cands):
        world_means = [
            float(np.mean(cell_vals.get((ci, wj), [0.0]))) for wj in range(len(opp_worlds))
        ]
        expected = float(sum(opp_worlds[wj][2] * world_means[wj] for wj in range(len(opp_worlds))))
        # Use mean of worst-2 worlds (less crit noise than single raw min).
        sorted_w = sorted(world_means)
        tail = float(np.mean(sorted_w[: min(2, len(sorted_w))]))
        # Diversity-injected field setters get a gentle prior floor so genuine blind spots can win.
        eff_prior = max(prior, 0.06) if injected else max(prior, 1e-4)
        bonus = prior_bonus * float(np.log(eff_prior)) if prior_bonus > 0.0 else 0.0
        total = (1.0 - worst_case_weight) * expected + worst_case_weight * tail + bonus
        digits = "".join(str(k + 1) for k in our_order)
        scored_rows.append(
            {
                "order": digits,
                "pv_idx": pv_idx,
                "prior": round(prior, 4),
                "injected": injected,
                "expected": round(expected, 4),
                "tail": round(tail, 4),
                "score": round(total, 4),
            }
        )
        if total > best_score:
            best_score = total
            best_i = ci

    chosen_digits = "".join(str(k + 1) for k in our_cands[best_i][1])
    scored_rows.sort(key=lambda r: -r["score"])
    audit = {
        "chosen": chosen_digits,
        "neural_top1": top_digits,
        "overturned": chosen_digits != top_digits,
        "candidates_tested": len(our_cands),
        "opp_worlds": len(opp_worlds),
        "top_candidates": scored_rows[:5],
    }
    return f"/team {chosen_digits}", audit
