"""Native Rust `pikapro`-powered multi-world lookahead search.

Replaces Node.js `SimWorker` rollouts (~2.5s/turn for 6w x 6a x 2s) with in-process
Rust `pikapro.Battle.patch_public` + `Battle.fork()` + `Battle.encode_state_rich()`
(~30ms/turn for 6w x 6a x 2s, or <0.25s/turn for full 16w x 8a x 4s).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pikapro

from vgc.damage import to_id
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.live_mirror import (
    LiveExactMirror,
    MirrorHypothesis,
    _opponent_sets,
    _team_preview_order,
    mirror_side,
)
from vgc.rl.replica_search import ReplicaSearchResult, SampledWorld, search_sampled_worlds
from vgc.rl.rust_policy import RustFactoredPolicy
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
    elapsed = max(0, int(current_turn) - int(start_turn))
    return max(1, default_duration - elapsed)


def _side_patch_from_snapshot(
    side_snap: dict[str, Any],
    *,
    is_own: bool,
    current_turn: int,
    hidden_timers: dict[str, Any],
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
        forme = (
            sp_id
            if (m.get("mega_evolved") and sp_id != base_id)
            or (sp_id != base_id and "mega" in sp_id)
            else None
        )
        status_raw = str(m.get("status") or "").lower()
        status = _STATUS_NAMES.get(status_raw)
        timer_info = hidden_timers.get(sp_id) or hidden_timers.get(base_id) or {}
        status_turns = int(timer_info.get("sleep_turns") or m.get("status_counter") or 1)
        boosts_raw = m.get("boosts") or {}
        boosts_map = boosts_raw if isinstance(boosts_raw, dict) else dict(boosts_raw)
        boosts = [int(boosts_map.get(st, 0) or 0) for st in _BOOST_ORDER]

        effects = {to_id(e.get("id") if isinstance(e, dict) else e) for e in (m.get("effects") or [])}
        is_trapped = bool(
            m.get("trapped", False)
            or trapped_by_species.get(to_id(sp_id), False)
            or trapped_by_species.get(to_id(base_id), False)
        )
        entry: dict[str, Any] = {
            "species": base_id,
            "fainted": bool(m.get("fainted", False)),
            "boosts": boosts,
            "protect_counter": int(m.get("protect_counter") or 0),
            "active_turns": int(m.get("active_turns") or (1 if m.get("active") else 0)),
            "active_move_actions": int(m.get("active_move_actions") or (0 if m.get("first_turn") else 1)),
            "trapped": is_trapped,
            "taunt_turns": 2 if "taunt" in effects else 0,
            "confusion_turns": int(timer_info.get("confusion_turns") or (2 if "confusion" in effects else 0)),
            "substitute": "substitute" in effects,
            "must_recharge": bool(m.get("must_recharge", False)) or "mustrecharge" in effects,
            "unburden": bool(m.get("item_lost_this_stint")) or "unburden" in effects,
        }
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
        if is_own:
            if item_state == "none":
                entry["item"] = None
            elif raw_item:
                entry["item"] = raw_item
        else:
            if item_state == "known" and raw_item:
                entry["item"] = raw_item
            elif item_state == "none":
                entry["item"] = None
            elif not keep_hidden_items and raw_item:
                entry["item"] = raw_item

        # `snapshot_battle` uses `ability_id` + `ability_known`.
        raw_ab = to_id(m.get("ability_id") if "ability_id" in m else m.get("ability"))
        if raw_ab and (is_own or m.get("ability_known", True)):
            entry["ability"] = raw_ab

        move_pp: list[tuple[str, int]] = []
        disabled_moves: list[str] = []
        for mv in m.get("moves") or []:
            if isinstance(mv, dict) and mv.get("id"):
                mid = str(mv["id"])
                if mv.get("current_pp") is not None:
                    move_pp.append((mid, int(mv["current_pp"])))
                if mv.get("disabled"):
                    disabled_moves.append(mid)
        if move_pp:
            entry["move_pp"] = move_pp
        if disabled_moves:
            entry["disabled_moves"] = disabled_moves
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

    return {
        "mega_used": bool(side_snap.get("used_mega_evolution", False)),
        "total_fainted": sum(1 for m in mons if m["fainted"]),
        "fainted_last_turn": False,
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

    timers_hyp = getattr(hypothesis, "timers", None)
    hidden_our = getattr(timers_hyp, "our", None) or {}
    hidden_opp = getattr(timers_hyp, "opponent", None) or {}

    our_patch = _side_patch_from_snapshot(
        snapshot["our_side"], is_own=True, current_turn=turn, hidden_timers=hidden_our
    )
    opp_patch = _side_patch_from_snapshot(
        snapshot["opponent_side"],
        is_own=False,
        current_turn=turn,
        hidden_timers=hidden_opp,
        keep_hidden_items=True,
    )
    patch["sides"] = [our_patch, opp_patch] if perspective == "p1" else [opp_patch, our_patch]
    return patch


def _foe_public_knowledge(
    battle, opp_sets: list[dict[str, Any]]
) -> tuple[list[str | None], list[str | None], list[list[str]]]:
    """Return per-foe-row publicly revealed items, abilities, and moves (fog-safe)."""
    known_by_base: dict[str, Any] = {}
    for mon in (getattr(battle, "opponent_team", None) or {}).values():
        sp = to_id(getattr(mon, "species", None))
        if sp:
            known_by_base[sp] = mon
    items: list[str | None] = []
    abilities: list[str | None] = []
    moves: list[list[str]] = []
    for s in opp_sets[:6]:
        sp = to_id(s.get("species"))
        mon = known_by_base.get(sp)
        if mon is None:
            items.append(None)
            abilities.append(None)
            moves.append([])
            continue
        it = normalize_item(getattr(mon, "item", None))
        ab = to_id(getattr(mon, "ability", None))
        mvs = [to_id(mid) for mid in (getattr(mon, "moves", None) or {}) if to_id(mid)]
        items.append(it or None)
        abilities.append(ab or None)
        moves.append(mvs)
    return items, abilities, moves


@dataclass
class RustWorldRoot:
    battle: pikapro.Battle
    our_side_idx: int
    opp_side_idx: int
    foe_items: list[str | None]
    foe_abilities: list[str | None]
    foe_moves: list[list[str]]


class RustExactMirror:
    """Zero-Node.js public-state mirror backed directly by Rust `pikapro.Battle.patch_public`."""

    def __init__(self, own_packed_team: str, config: PolicyConfig | None = None) -> None:
        self.own_packed_team = own_packed_team
        self.config = config or PolicyConfig()
        # Reuse LiveExactMirror purely for hypothesis generation (never touches .worker!).
        self._hyp_mirror = LiveExactMirror(own_packed_team, self.config)
        self.last_hypothesis_audit: dict[str, Any] = {}

    def hypotheses(
        self, battle, memory=None, *, worlds: int | None = None
    ) -> list[MirrorHypothesis]:
        if worlds is not None and worlds != self._hyp_mirror.config.exact_search_total_hypotheses:
            self._hyp_mirror.config = dataclasses.replace(
                self._hyp_mirror.config, exact_search_total_hypotheses=worlds
            )
        hyps = self._hyp_mirror.hypotheses(battle, memory=memory)
        self.last_hypothesis_audit = dict(self._hyp_mirror.last_hypothesis_audit)
        return hyps

    def build(self, battle, hypothesis: MirrorHypothesis | None = None) -> RustWorldRoot:
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
        patch_dict = _build_rust_patch_dict(battle, snap, side, hypothesis)
        rust_battle.patch_public(patch_dict)
        foe_items, foe_abilities, foe_moves = _foe_public_knowledge(battle, opp_sets)
        return RustWorldRoot(
            battle=rust_battle,
            our_side_idx=our_idx,
            opp_side_idx=opp_idx,
            foe_items=foe_items,
            foe_abilities=foe_abilities,
            foe_moves=foe_moves,
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
            if val is not None:
                slot_moves.append([to_id(getattr(m, "id", "")) for m in val])
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

    # Secondary check: match base species for form-changed switches (e.g. Mega / regional forms).
    def _loose_slot(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
        if a == b:
            return True
        if a[0] == "switch" and b[0] == "switch":
            return a[1].startswith(b[1]) or b[1].startswith(a[1])
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
        sp_id = int(obs[row * 80])
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
            # Base-species prefix match (e.g. Mega forme vs base species)
            for sp_k, rows in obs_rows_by_sp.items():
                if sp_k.startswith(key) or key.startswith(sp_k):
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


def _resolve_forced_switches(clone: pikapro.Battle, our_idx: int, opp_idx: int, start_turn: int) -> None:
    """Advance any mid-turn forced switches until the turn finishes or the game ends."""
    for _ in range(8):
        if clone.over or clone.turn > start_turn:
            break
        progressed = False
        for s_idx in (our_idx, opp_idx):
            if clone.request_kind(s_idx) == "force_switch":
                choices = clone.legal_choices(s_idx)
                if choices:
                    clone.submit(s_idx, choices[0])
                    progressed = True
        if not progressed:
            break


def _batch_step_policy_rollout(
    clones: list[pikapro.Battle],
    root: RustWorldRoot,
    policy: RustFactoredPolicy,
    space: pikapro.ActionSpace,
) -> None:
    """Advance a batch of non-terminal battles by 1 full turn using batched GPU policy."""
    import torch

    if not clones:
        return
    our_idx = root.our_side_idx
    opp_idx = root.opp_side_idx
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
        for s_idx in (our_idx, opp_idx):
            rk = clone.request_kind(s_idx)
            if rk == "wait":
                continue
            obs = clone.encode_state_rich(
                s_idx,
                foe_known_items=root.foe_items if s_idx == our_idx else None,
                foe_known_abilities=root.foe_abilities if s_idx == our_idx else None,
                foe_known_moves=root.foe_moves if s_idx == our_idx else None,
            )
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

    for ci, side_choices in per_clone.items():
        clone = clones[ci]
        if clone.over:
            continue
        start_turn = clone.turn
        for s_idx in (our_idx, opp_idx):
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
        _resolve_forced_switches(clone, our_idx, opp_idx, start_turn)


def search_public_rust(
    battle,
    packed_team: str,
    policy: RustFactoredPolicy,
    *,
    worlds: int = 16,
    actions: int = 8,
    seeds: int = 4,
    iterations: int = 20000,
    seed: int = 1,
    mirror: RustExactMirror | None = None,
    memory=None,
    worst_case_weight: float = 0.5,
    min_prior_prob: float = 0.01,
    prior_bonus: float = 0.025,
    rollout_turns: int = 1,
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
            root = mirror.build(battle, hyp)
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
            obs = root.battle.encode_state_rich(
                side_idx,
                foe_known_items=root.foe_items if side_idx == root.our_side_idx else None,
                foe_known_abilities=root.foe_abilities if side_idx == root.our_side_idx else None,
                foe_known_moves=root.foe_moves if side_idx == root.our_side_idx else None,
            )
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

        def select(obs_handle: tuple[RustWorldRoot, int] | None) -> list[str]:
            if obs_handle is None:
                return ["<wait>"]
            root, side_idx = obs_handle
            ranked, prior_map = _rank_for_side(root, side_idx)
            scored_priors[id(obs_handle)] = prior_map
            return ranked

        def opp_priors(obs_handle: tuple[RustWorldRoot, int] | None, opp_orders: tuple[str, ...]) -> list[float]:
            if obs_handle is None or len(opp_orders) <= 1:
                return [1.0] * len(opp_orders)
            pmap = scored_priors.get(id(obs_handle)) or {}
            return [max(pmap.get(c, 0.0), 1e-9) for c in opp_orders]

        root_obs = (world_roots[0], world_roots[0].our_side_idx)

        def eval_matrix(
            root: RustWorldRoot, our_orders: tuple[str, ...], opp_orders: tuple[str, ...]
        ) -> np.ndarray:
            cells: list[list[list[float]]] = [[[] for _ in opp_orders] for _ in our_orders]
            pending_pos: list[tuple[int, int]] = []
            pending_clones: list[pikapro.Battle] = []
            our_idx = root.our_side_idx
            opp_idx = root.opp_side_idx

            for i, our_c in enumerate(our_orders):
                for j, opp_c in enumerate(opp_orders):
                    for fseed in future_seeds:
                        clone = root.battle.fork(seed=fseed)
                        start_turn = clone.turn
                        if our_c != "<wait>" and clone.request_kind(our_idx) != "wait":
                            clone.submit(our_idx, our_c)
                        if not clone.over and opp_c != "<wait>" and clone.request_kind(opp_idx) != "wait":
                            clone.submit(opp_idx, opp_c)
                        _resolve_forced_switches(clone, our_idx, opp_idx, start_turn)
                        if clone.over:
                            if clone.winner is None:
                                cells[i][j].append(0.0)
                            else:
                                cells[i][j].append(1.0 if clone.winner == our_idx else -1.0)
                        else:
                            pending_pos.append((i, j))
                            pending_clones.append(clone)

            # Optional multi-turn batched GPU rollout past the 1-turn horizon.
            for _ in range(rollout_turns):
                if not pending_clones:
                    break
                _batch_step_policy_rollout(pending_clones, root, policy, space)
                next_pos: list[tuple[int, int]] = []
                next_clones: list[pikapro.Battle] = []
                for (i, j), clone in zip(pending_pos, pending_clones, strict=True):
                    if clone.over:
                        if clone.winner is None:
                            cells[i][j].append(0.0)
                        else:
                            cells[i][j].append(1.0 if clone.winner == our_idx else -1.0)
                    else:
                        next_pos.append((i, j))
                        next_clones.append(clone)
                pending_pos = next_pos
                pending_clones = next_clones

            if pending_clones:
                pending_obs = [
                    clone.encode_state_rich(
                        our_idx,
                        foe_known_items=root.foe_items,
                        foe_known_abilities=root.foe_abilities,
                        foe_known_moves=root.foe_moves,
                    )
                    for clone in pending_clones
                ]
                vals = policy.values_from_obs(np.stack(pending_obs, axis=0))
                for (i, j), v in zip(pending_pos, vals.tolist(), strict=True):
                    cells[i][j].append(float(v))

            our_priors = scored_priors.get(id(root_obs)) or {}
            mat = np.empty((len(our_orders), len(opp_orders)), dtype=float)
            for i, our_c in enumerate(our_orders):
                bonus = (
                    prior_bonus * float(np.log(max(our_priors.get(our_c, 1e-4), 1e-4)))
                    if prior_bonus > 0.0
                    else 0.0
                )
                for j in range(len(opp_orders)):
                    mat[i, j] = (float(np.mean(cells[i][j])) if cells[i][j] else 0.0) + bonus
            return mat

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
