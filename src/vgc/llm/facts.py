"""Public `DoubleBattle` -> the packet the LLM reads.

`build_packet(battle, scored, ...)` is the only entry point the player code needs. It turns
OUR side's view of a battle (poke-env's fogged `DoubleBattle`, never a simulator root) and
the engine's ranked orders into a `ContextPacket` plus the numbered options.

Two parts, kept apart for OpenAI prompt caching:

* FIXED text: rules digest, the owner's reasoning checklist, our team, our written team
  plan and the opponent's six species with GUESSED sets. Built once per battle and stored
  on the battle object, so it is byte-identical on every turn of the game.
* PER-TURN text: the board, what the opponent has actually shown (facts) next to what we
  only guess (labelled), engine-computed facts (who moves first, damage and KO chances),
  and the numbered options each with a one-line engine note.

The model never has to do arithmetic: speed order and damage come from `vgc.evaluator`
and `vgc.damage`. Everything here is best effort: a missing piece of data skips that fact
instead of raising, and `build_packet` falls back to a bare-bones packet if even that
fails (it never raises on a real battle object).

Information boundary (CLAUDE.md "Public information only"): only fields poke-env fills
from our own request and from what Showdown has publicly shown are read. Opponent hidden
information appears only as set-prior GUESSES, always labelled as guesses.
"""

from __future__ import annotations

import dataclasses
import hashlib
from functools import lru_cache
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from vgc.damage import FieldState, damage_range, to_id
from vgc.data import load_items, load_moves, load_species
from vgc.llm.config import LLMConfig
from vgc.llm.packet import build_options
from vgc.llm.types import ContextPacket, Option

REPO_ROOT = Path(__file__).resolve().parents[3]
PLANS_DIR = REPO_ROOT / "teams" / "owner" / "plans"

# Attribute on the battle object that holds the per-game fixed text.
_FIXED_ATTR = "_vgc_llm_fixed_text"

RULES_DIGEST = """\
You advise a Pokemon Showdown bot playing gen9championsvgc2026regmc (Champions VGC, Reg M-C): \
doubles, bring 6 pick 4, level 50, best of one.
- Stats use Stat Points (max 32 per stat, 66 total), not EVs. Items and moves differ from \
normal VGC: only the items and moves shown in this packet are real here.
- Each player may Mega Evolve ONE Pokemon per game (holding its Mega Stone).
- Doubles targets: "left"/"right" foe means the opposing slot; spread moves hit both foes.
- You propose; the engine checks and decides. Answer ONLY with option IDs from the OPTIONS \
list of the current turn. Never invent options.
- Never do damage or speed arithmetic. Use the ENGINE FACTS lines; they are computed exactly \
by code. Foe sets, items and speeds marked GUESS are estimates, not facts.
- Reply with the JSON schema given: "plan" = at most 2 short sentences, "proposals" = up to 3 \
objects {id, why} best first, "why" = one short sentence."""

REASONING_CHECKLIST = """\
How to think each turn (the owner's method):
1. Rule out bad options first: moves that do nothing (immune, blocked by terrain/ability), \
that get blocked by a likely Protect, that leave us open to a KO, or that undo our own plan.
2. Read their plan from the board and what they have shown: their likely leads, speed control, \
setup, Protect habits, and the biggest threat to our team.
3. Pick ours: choose the option that advances OUR game plan (below) and keeps our win \
conditions alive; prefer lines that are good against both their likely plays (attack and \
Protect/switch).
4. Adapt: if the engine's top option is already clearly best, say so and propose it first; \
only overrule the engine when a fact above supports it."""

# Prior moves/items seen in fewer than this share of a species' games are noise.
_MIN_PRIOR_SHARE = 0.05

_SPREAD_TARGET_TEXT = {
    "allAdjacentFoes": "hits both foes",
    "allAdjacent": "hits everyone else",
    "allySide": "our side",
    "foeSide": "their side",
    "adjacentAlly": "ally only",
    "self": "self",
}

_PROTECT_IDS = {
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
    "burningbulwark", "obstruct", "wideguard", "quickguard", "maxguard",
}  # fmt: skip


# ---------------------------------------------------------------------------------------
# Team plan files
# ---------------------------------------------------------------------------------------


def load_team_plan(team_path_or_name: str | Path) -> str:
    """Return the plan text for an owner team, or "" if there is none.

    Accepts a team name ("psyspam_sand"), or a path such as
    "teams/owner/psyspam_sand.packed.txt" / "psyspam_sand.txt" / the plan file itself.
    """
    try:
        name = Path(str(team_path_or_name)).name
        for suffix in (".md", ".txt", ".packed"):
            name = name.removesuffix(suffix)
        path = PLANS_DIR / f"{name}.md"
        return path.read_text().strip() if path.is_file() else ""
    except OSError:
        return ""


# ---------------------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------------------


def _table_name(table: dict[str, Any], key: str | None) -> str:
    if not key:
        return "?"
    entry = table.get(to_id(key) or key)
    if isinstance(entry, dict) and entry.get("name"):
        return str(entry["name"])
    return str(key)


def _species(sid: str | None) -> str:
    return _table_name(load_species(), sid)


def _mon_name(mon: Any) -> str:
    """Display name. poke-env can report our Mega-holder under its Mega forme before it has
    evolved, so a Mega forme is shown by its base species (the stone says it can Mega)."""
    sid = to_id(str(getattr(mon, "species", "") or ""))
    data = load_species().get(sid) or {}
    if data.get("isMega") and getattr(mon, "base_species", None):
        sid = to_id(str(mon.base_species))
    return _species(sid)


def _move(mid: str | None) -> str:
    return _table_name(load_moves(), mid)


@lru_cache(maxsize=1)
def _ability_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for entry in load_species().values():
        for name in (entry.get("abilities") or {}).values():
            names[to_id(str(name)) or str(name)] = str(name)
    return names


def _ability(aid: str | None) -> str:
    if not aid:
        return "?"
    return _ability_names().get(to_id(str(aid)) or "", str(aid))


def _item(iid: str | None) -> str:
    return _table_name(load_items(), iid)


def _known_item(mon: Any) -> str | None:
    """Item id if one is actually known (poke-env uses 'unknown_item' / None / '')."""
    value = getattr(mon, "item", None)
    if not value or str(value).lower() in ("unknown_item", "none"):
        return None
    return to_id(str(value))


def _enum_name(value: Any) -> str:
    return str(getattr(value, "name", value)).replace("_", " ").title()


def _hp_pct(mon: Any) -> int:
    try:
        return round(float(mon.current_hp_fraction) * 100)
    except Exception:
        return 100


def _status(mon: Any) -> str:
    status = getattr(mon, "status", None)
    return str(getattr(status, "name", status)).upper() if status else ""


def _boosts(mon: Any) -> str:
    out = []
    for stat in ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion"):
        value = (getattr(mon, "boosts", None) or {}).get(stat, 0)
        if value:
            out.append(f"{value:+d} {stat}")
    return ", ".join(out)


def _pad(items: Sequence[Any], n: int = 2) -> list[Any]:
    out = list(items or [])
    while len(out) < n:
        out.append(None)
    return out[:n]


def _names(mons: Iterable[Any]) -> str:
    return ", ".join(_mon_name(m) for m in mons) or "none"


def _prior_entry(priors: dict[str, Any], mon: Any) -> dict[str, Any] | None:
    entry = (priors.get("species") or {}).get(to_id(getattr(mon, "species", "") or ""))
    return entry if entry and entry.get("appearances", 0) > 0 else None


def _prior_moves(priors: dict[str, Any], mon: Any, skip: Iterable[str], limit: int) -> list[str]:
    """Most commonly seen moves for this species (id order = frequency), minus `skip`."""
    entry = _prior_entry(priors, mon)
    if not entry:
        return []
    total = float(entry["appearances"])
    skipped = set(skip)
    ranked = sorted((entry.get("moves") or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for move_id, count in ranked:
        if move_id in skipped:
            continue
        if count / total < _MIN_PRIOR_SHARE:
            break
        out.append(f"{_move(move_id)} {round(100 * count / total)}%")
        if len(out) >= limit:
            break
    return out


def _prior_item(priors: dict[str, Any], mon: Any) -> str:
    entry = _prior_entry(priors, mon)
    items = (entry or {}).get("items") or {}
    if not items:
        return ""
    item_id, count = max(items.items(), key=lambda kv: (kv[1], kv[0]))
    share = count / float(entry["appearances"])
    return f"{_item(item_id)} {round(100 * share)}%" if share >= _MIN_PRIOR_SHARE else ""


def _prior_ability(priors: dict[str, Any], mon: Any) -> str:
    entry = _prior_entry(priors, mon)
    abilities = (entry or {}).get("abilities") or {}
    if not abilities:
        return ""
    ability_id, count = max(abilities.items(), key=lambda kv: (kv[1], kv[0]))
    share = count / float(entry["appearances"])
    return f"{_ability(ability_id)} {round(100 * share)}%" if share >= _MIN_PRIOR_SHARE else ""


# ---------------------------------------------------------------------------------------
# Fixed part
# ---------------------------------------------------------------------------------------


def _is_mega_form(mon: Any) -> bool:
    return bool((load_species().get(to_id(str(getattr(mon, "species", "") or ""))) or {}).get("isMega"))


def _our_team_lines(battle: Any) -> list[str]:
    from vgc.evaluator import _our_pokemon_state

    lines = []
    for mon in list((getattr(battle, "team", None) or {}).values()):
        spe = ""
        try:
            state = _our_pokemon_state(mon)
            if _is_mega_form(mon) and getattr(mon, "base_species", None):
                mega_speed = state.stats()["spe"]
                base = dataclasses.replace(state, species_id=to_id(str(mon.base_species)))
                spe = f", Speed {base.stats()['spe']} ({mega_speed} after Mega Evolving)"
            else:
                spe = f", Speed {state.stats()['spe']}"
        except Exception:
            pass
        item = _known_item(mon)
        ability = getattr(mon, "base_ability", None) or getattr(mon, "ability", None)
        moves = ", ".join(_move(m) for m in (getattr(mon, "moves", None) or {}))
        nature = f", {str(mon.nature).title()}" if getattr(mon, "nature", None) else ""
        lines.append(
            f"- {_mon_name(mon)} @ {_item(item) if item else 'no item'}, "
            f"{_ability(ability)}{nature}{spe}: {moves}"
        )
    return lines


def _move_reference_lines(battle: Any, priors: dict[str, Any]) -> list[str]:
    """One line per move on either team: type, category, power, priority, who it hits."""
    ids: list[str] = []
    for mon in (getattr(battle, "team", None) or {}).values():
        ids += [to_id(m) for m in (getattr(mon, "moves", None) or {})]
    for mon in _opponent_team(battle):
        entry = _prior_entry(priors, mon)
        if entry:
            ranked = sorted((entry.get("moves") or {}).items(), key=lambda kv: (-kv[1], kv[0]))
            ids += [m for m, c in ranked[:4] if c / entry["appearances"] >= _MIN_PRIOR_SHARE]
    moves = load_moves()
    seen: set[str] = set()
    lines = []
    for mid in ids:
        if mid in seen or mid not in moves:
            continue
        seen.add(mid)
        d = moves[mid]
        power = f" {d['basePower']}" if d.get("basePower") else ""
        extras = []
        if d.get("priority"):
            extras.append(f"priority {d['priority']:+d}")
        target = str(d.get("target", ""))
        if target in _SPREAD_TARGET_TEXT:
            extras.append(_SPREAD_TARGET_TEXT[target])
        tail = f", {', '.join(extras)}" if extras else ""
        lines.append(f"{d['name']}: {d.get('type', '?')} {d.get('category', '?')}{power}{tail}")
    return lines


def _opponent_team(battle: Any) -> list[Any]:
    preview = list(getattr(battle, "teampreview_opponent_team", None) or [])
    if len(preview) == 6:
        return preview
    return list((getattr(battle, "opponent_team", None) or {}).values())


def _opponent_guess_lines(battle: Any, priors: dict[str, Any]) -> list[str]:
    lines = []
    for mon in _opponent_team(battle):
        moves = _prior_moves(priors, mon, (), 4)
        item = _prior_item(priors, mon)
        parts = []
        if moves:
            parts.append("moves " + ", ".join(moves))
        if item:
            parts.append(f"item {item}")
        lines.append(f"- {_mon_name(mon)}: " + ("; ".join(parts) if parts else "no data"))
    return lines


def _fixed_text(battle: Any, team_plan: str, priors: dict[str, Any]) -> str:
    parts = [RULES_DIGEST, REASONING_CHECKLIST]
    team = _safe(lambda: _our_team_lines(battle), [])
    if team:
        parts.append("## OUR TEAM (known exactly)\n" + "\n".join(team))
    if team_plan and team_plan.strip():
        parts.append("## OUR GAME PLAN\n" + team_plan.strip())
    foes = _safe(lambda: _opponent_guess_lines(battle, priors), [])
    if foes:
        parts.append(
            "## THEIR TEAM (six species are known; sets below are GUESSES from ladder "
            "replay frequencies, % = share of games where seen)\n" + "\n".join(foes)
        )
    ref = _safe(lambda: _move_reference_lines(battle, priors), [])
    if ref:
        parts.append("## MOVE REFERENCE (type, category, base power; engine data)\n" + "\n".join(ref))
    return "\n\n".join(parts)


def _safe(fn: Any, default: Any) -> Any:
    try:
        return fn()
    except Exception:
        return default


def _fixed_for_battle(battle: Any, team_plan: str, priors: dict[str, Any]) -> str:
    """Build once per battle and reuse, so later turns cannot change a byte of it (item
    consumption, a Mega forme, newly revealed opponent Pokemon...)."""
    digest = hashlib.sha256((team_plan or "").encode("utf-8")).hexdigest()
    cached = getattr(battle, _FIXED_ATTR, None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == digest:
        return str(cached[1])
    text = _fixed_text(battle, team_plan, priors)
    try:
        setattr(battle, _FIXED_ATTR, (digest, text))
    except Exception:
        pass
    return text


# ---------------------------------------------------------------------------------------
# Per-turn part
# ---------------------------------------------------------------------------------------


def _since(turn_started: Any, now: int) -> str:
    try:
        started = int(turn_started)
    except (TypeError, ValueError):
        return ""
    return f" (since turn {started})" if started and started <= now else ""


_LAYERED = {"SPIKES", "TOXIC_SPIKES", "STEALTH_ROCK", "STICKY_WEB"}


def _condition(cond: Any, value: Any, now: int) -> str:
    """Side condition text: layers for hazards, the start turn for timed effects."""
    name = _enum_name(cond)
    if str(getattr(cond, "name", cond)).upper() in _LAYERED:
        return f"{name} x{value}" if isinstance(value, int) and value > 1 else name
    return name + _since(value, now)


def _field_lines(battle: Any, now: int) -> list[str]:
    lines = []
    weather = [f"{_enum_name(w)}{_since(t, now)}" for w, t in (battle.weather or {}).items()]
    fields = list((battle.fields or {}).items())
    terrain = [f"{_enum_name(f)}{_since(t, now)}" for f, t in fields if "TERRAIN" in str(f).upper()]
    other = [f"{_enum_name(f)}{_since(t, now)}" for f, t in fields if "TERRAIN" not in str(f).upper()]
    lines.append("Weather: " + (", ".join(weather) or "none"))
    lines.append("Terrain: " + (", ".join(terrain) or "none"))
    lines.append("Trick Room / other field: " + (", ".join(other) or "none"))
    ours = [_condition(c, v, now) for c, v in (battle.side_conditions or {}).items()]
    theirs = [_condition(c, v, now) for c, v in (battle.opponent_side_conditions or {}).items()]
    lines.append("Our side: " + (", ".join(ours) or "none"))
    lines.append("Their side: " + (", ".join(theirs) or "none"))
    return lines


def _our_mon_line(mon: Any, tag: str, mega_ok: bool) -> str:
    hp = f"{_hp_pct(mon)}% ({mon.current_hp}/{mon.max_hp})"
    bits = [hp]
    if _status(mon):
        bits.append(_status(mon))
    if _boosts(mon):
        bits.append(_boosts(mon))
    item = _known_item(mon)
    bits.append(f"item {_item(item)}" if item else "no item")
    bits.append(f"ability {_ability(getattr(mon, 'ability', None))}")
    moves = ", ".join(_move(m) for m in (getattr(mon, "moves", None) or {}))
    mega = "; CAN MEGA EVOLVE this turn" if mega_ok else ""
    return f"- {tag} {_mon_name(mon)}: " + ", ".join(bits) + f"; moves: {moves}{mega}"


def _board_lines(battle: Any) -> list[str]:
    lines = []
    can_mega = _pad(list(getattr(battle, "can_mega_evolve", None) or [False, False]))
    our_active = _pad(battle.active_pokemon)
    for slot, mon in enumerate(our_active):
        if mon is not None and not mon.fainted:
            lines.append(_our_mon_line(mon, f"OUR {'left' if slot == 0 else 'right'}",
                                       bool(can_mega[slot])))
    active_ids = {id(m) for m in our_active if m is not None}
    bench = [m for m in (battle.team or {}).values() if id(m) not in active_ids]
    fainted = [m for m in bench if m.fainted]
    alive = [m for m in bench if not m.fainted and getattr(m, "selected_in_teampreview", True)]
    for mon in alive:
        lines.append(_our_mon_line(mon, "OUR bench", False))
    if fainted:
        lines.append("OUR fainted: " + _names(fainted))
    return lines


def _foe_lines(battle: Any, priors: dict[str, Any]) -> list[str]:
    lines = []
    active = _pad(battle.opponent_active_pokemon)
    for slot, mon in enumerate(active):
        if mon is None or mon.fainted:
            continue
        revealed = list((getattr(mon, "moves", None) or {}))
        item = _known_item(mon)
        ability = getattr(mon, "ability", None)
        facts = [f"{_hp_pct(mon)}% HP"]
        if _status(mon):
            facts.append(_status(mon))
        if _boosts(mon):
            facts.append(_boosts(mon))
        facts.append(f"item {_item(item)} (seen)" if item else "item not seen")
        facts.append(f"ability {_ability(ability)} (seen)" if ability else "ability not seen")
        facts.append("moves seen: " + (", ".join(_move(m) for m in revealed) or "none"))
        guess = []
        if len(revealed) < 4:
            more = _prior_moves(priors, mon, revealed, 4 - len(revealed))
            if more:
                guess.append("other moves " + ", ".join(more))
        if not item:
            g = _prior_item(priors, mon)
            if g:
                guess.append("item " + g)
        if not ability:
            g = _prior_ability(priors, mon)
            if g:
                guess.append("ability " + g)
        tail = f"; GUESS: {'; '.join(guess)}" if guess else ""
        lines.append(f"- THEIR {'left' if slot == 0 else 'right'} {_mon_name(mon)}: "
                     + ", ".join(facts) + tail)
    team = _opponent_team(battle)
    active_ids = {id(m) for m in active if m is not None}
    seen_back = [m for m in (getattr(battle, "opponent_team", None) or {}).values()
                 if id(m) not in active_ids and not m.fainted and m.revealed]
    if seen_back:
        lines.append("THEIR seen on bench: " + ", ".join(
            f"{_mon_name(m)} {_hp_pct(m)}%" for m in seen_back))
    fainted = [m for m in team if m.fainted]
    if fainted:
        lines.append("THEIR fainted: " + _names(fainted))
    return lines


def _engine_states(battle: Any, config: Any) -> dict[str, Any]:
    """States + field shared by the engine facts. Raises if the battle is unusable."""
    from vgc.evaluator import (
        _our_pokemon_state,
        _screens_from,
        _terrain_str,
        _weather_str,
    )
    from vgc.sets import opponent_state, usage_spreads_for
    from poke_env.battle.field import Field
    from poke_env.battle.side_condition import SideCondition

    usage = usage_spreads_for(config)
    ours = _pad(battle.active_pokemon)
    foes = _pad(battle.opponent_active_pokemon)
    our_states = [_our_pokemon_state(m) if m is not None and not m.fainted else None for m in ours]
    foe_states = [
        opponent_state(m, usage=usage) if m is not None and not m.fainted else None for m in foes
    ]
    return {
        "usage": usage,
        "ours": ours,
        "foes": foes,
        "our_states": our_states,
        "foe_states": foe_states,
        "weather": _weather_str(battle),
        "terrain": _terrain_str(battle),
        "trick_room": Field.TRICK_ROOM in battle.fields,
        "our_tailwind": SideCondition.TAILWIND in battle.side_conditions,
        "foe_tailwind": SideCondition.TAILWIND in battle.opponent_side_conditions,
        "our_screens": _screens_from(battle.side_conditions),
        "foe_screens": _screens_from(battle.opponent_side_conditions),
    }


def _speed_lines(st: dict[str, Any]) -> list[str]:
    from vgc.evaluator import field_effective_speed
    from vgc.sets import opponent_spread_hypotheses

    entries: list[tuple[str, float, float, bool]] = []  # name, lo, hi, ours
    for mon, state in zip(st["ours"], st["our_states"], strict=True):
        if state is None:
            continue
        speed = field_effective_speed(state, weather=st["weather"], tailwind=st["our_tailwind"])
        entries.append((_mon_name(mon), speed, speed, True))
    for mon, state in zip(st["foes"], st["foe_states"], strict=True):
        if state is None:
            continue
        speeds = []
        for sp, nature, _w in opponent_spread_hypotheses(state.species_id, st["usage"], limit=3):
            variant = dataclasses.replace(state, sp_spread=sp, nature=nature)
            speeds.append(field_effective_speed(variant, weather=st["weather"],
                                                tailwind=st["foe_tailwind"]))
        speeds = speeds or [field_effective_speed(state, weather=st["weather"],
                                                  tailwind=st["foe_tailwind"])]
        entries.append((_mon_name(mon), min(speeds), max(speeds), False))
    if not entries:
        return []
    tr = st["trick_room"]
    order = sorted(entries, key=lambda e: (e[1] + e[2]) / 2, reverse=not tr)
    lines = []
    for name, lo, hi, ours in entries:
        # A foe's spread, item and sometimes ability are hidden: its Speed is ALWAYS an
        # estimate, even when every sampled spread (or the fallback spread) agrees.
        if ours:
            spd = f"{lo:.0f}"
        else:
            spd = (f"~{lo:.0f}" if round(lo) == round(hi) else f"~{lo:.0f}-{hi:.0f}") + " GUESS"
        lines.append(f"{'ours' if ours else 'theirs'} {name} Speed {spd}")
    any_foe = any(not e[3] for e in entries)
    seq = " > ".join(f"{e[0]} ({'ours' if e[3] else 'theirs'})" for e in order)
    note = "Trick Room: slower moves first" if tr else "faster moves first"
    # An order is only reliable when every foe's range is clear of every one of ours.
    unclear = any(
        (e1[3] != e2[3]) and e1[1] <= e2[2] and e2[1] <= e1[2]
        for e1 in entries
        for e2 in entries
        if e1 is not e2
    )
    flag = " Some matchups are close." if unclear else ""
    head = "Likely move order (foe Speeds are GUESSES)" if any_foe else "Move order"
    return [
        "Speed (same-priority moves; priority moves go first regardless): "
        + "; ".join(lines),
        f"{head}, {note}: {seq}.{flag}",
    ]


def _hit_count(target: str, defenders: int, attacker_ally_alive: bool) -> int:
    """How many Pokemon a move with this `target` hits this turn (spread reduction needs 2+).

    Mirrors `vgc.evaluator._resolve_targets`: `allAdjacentFoes` hits every live foe of the
    attacker; `allAdjacent` also hits the attacker's own live partner.
    """
    if target == "allAdjacentFoes":
        return max(1, defenders)
    if target == "allAdjacent":
        return max(1, defenders + (1 if attacker_ally_alive else 0))
    return 1


def _field_state(st: dict[str, Any], defender_ours: bool, num_targets: int) -> FieldState:
    return FieldState(
        weather=st["weather"],
        terrain=st["terrain"],
        screens=st["our_screens"] if defender_ours else st["foe_screens"],
        trick_room=st["trick_room"],
        is_doubles=True,
        num_targets=num_targets,
    )


def _damage_lines(st: dict[str, Any]) -> list[str]:
    from vgc.evaluator import guaranteed_ko, likely_ko

    moves_data = load_moves()
    lines = []
    n_foes = sum(1 for s in st["foe_states"] if s is not None)
    n_ours = sum(1 for s in st["our_states"] if s is not None)
    for mon, atk in zip(st["ours"], st["our_states"], strict=True):
        if atk is None:
            continue
        for foe, dfn in zip(st["foes"], st["foe_states"], strict=True):
            if dfn is None:
                continue
            hp = dfn.hp_or_max()
            rows = []
            for move_id in list(getattr(mon, "moves", None) or {}):
                data = moves_data.get(to_id(move_id))
                if data is None or data.get("category") == "Status":
                    continue
                hits = _hit_count(str(data.get("target", "")), n_foes, n_ours >= 2)
                fs = _field_state(st, False, hits)
                res = damage_range(atk, dfn, to_id(move_id), fs)
                if not res.breakdown.get("move_supported") or res.breakdown.get("immune"):
                    continue
                tag = " KO" if guaranteed_ko(res, hp) else " likely KO" if likely_ko(res, hp) else ""
                rows.append((res.expected_percent,
                             f"{_move(move_id)} {res.min_percent:.0f}-{res.max_percent:.0f}%{tag}"))
            rows.sort(key=lambda r: -r[0])
            if rows:
                lines.append(f"{_mon_name(mon)} -> {_mon_name(foe)} "
                             f"({_hp_pct(foe)}% left): " + "; ".join(r[1] for r in rows[:3]))
    if lines:
        lines.insert(0, "Our damage as % of the target's FULL HP; KO = kills it at its current HP "
            "(foe spread/item are GUESSES):")
    return lines


def _threat_lines(st: dict[str, Any], priors: dict[str, Any], config: Any) -> list[str]:
    from vgc.sets import opponent_move_ids

    moves_data = load_moves()
    lines = []
    n_ours = sum(1 for s in st["our_states"] if s is not None)
    n_foes = sum(1 for s in st["foe_states"] if s is not None)
    for foe, fst in zip(st["foes"], st["foe_states"], strict=True):
        if fst is None:
            continue
        revealed = {to_id(m) for m in (getattr(foe, "moves", None) or {})}
        candidates = _safe(lambda: opponent_move_ids(foe, priors=priors, config=config),
                           list(revealed))
        for mon, ost in zip(st["ours"], st["our_states"], strict=True):
            if ost is None:
                continue
            best: tuple[float, str, float, bool] | None = None
            for move_id in candidates:
                data = moves_data.get(to_id(move_id))
                if data is None or data.get("category") == "Status":
                    continue
                hits = _hit_count(str(data.get("target", "")), n_ours, n_foes >= 2)
                res = damage_range(fst, ost, to_id(move_id), _field_state(st, True, hits))
                if not res.breakdown.get("move_supported") or res.breakdown.get("immune"):
                    continue
                if best is None or res.expected_percent > best[0]:
                    best = (res.expected_percent, move_id, res.max_percent,
                            to_id(move_id) in revealed)
            if best is not None:
                kind = "seen" if best[3] else "GUESS"
                lines.append(f"{_mon_name(foe)} -> {_mon_name(mon)}: best "
                             f"{_move(best[1])} ({kind}) about {best[0]:.0f}% "
                             f"(max {best[2]:.0f}%) of its full HP")
    if lines:
        lines.insert(0, "Their best hit on us (all moves seen + guessed; their spread, item and "
                     "ability are GUESSES, so these are estimates):")
    return lines


def _engine_fact_lines(battle: Any, priors: dict[str, Any], config: Any) -> list[str]:
    st = _engine_states(battle, config)
    lines: list[str] = []
    for part in (_speed_lines, _damage_lines):
        lines += _safe(lambda p=part: p(st), [])
    lines += _safe(lambda: _threat_lines(st, priors, config), [])
    return lines


# ---------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------


def _single_text(battle: Any, single: Any, actor: Any) -> str:
    name = _mon_name(actor) if actor is not None else "slot"
    target = getattr(single, "order", None)
    if target is None or isinstance(target, str):
        return f"{name}: {target or 'pass'}"
    if hasattr(target, "base_power") or hasattr(target, "category"):  # a Move
        text = f"{name}: {_move(getattr(target, 'id', None))}"
        if getattr(single, "mega", False):
            text += " (Mega Evolve first)"
        pos = getattr(single, "move_target", 0) or 0
        foes = _pad(battle.opponent_active_pokemon)
        ours = _pad(battle.active_pokemon)
        if pos in (1, 2) and foes[pos - 1] is not None:
            side = "left" if pos == 1 else "right"
            text += f" -> foe {side} ({_mon_name(foes[pos - 1])})"
        elif pos in (-1, -2) and ours[-pos - 1] is not None:
            partner = ours[-pos - 1]
            text += " -> self" if partner is actor else f" -> ally ({_mon_name(partner)})"
        return text
    return f"{name}: switch to {_mon_name(target)}"


def describe_order_text(battle: Any, order: Any) -> str:
    """Readable one-line version of a joint order; raw text if anything is unusual."""
    try:
        first, second = _pad(battle.active_pokemon)
        parts = []
        for single, actor in ((order.first_order, first), (order.second_order, second)):
            if actor is None or str(getattr(single, "message", "")).strip() == "/choose pass":
                continue  # empty or fainted slot: nothing to say
            parts.append(_single_text(battle, single, actor))
        return " | ".join(parts) or str(getattr(order, "message", order))
    except Exception:
        return str(getattr(order, "message", order))


def _option_lines(battle: Any, scored: Sequence[Any], options: Sequence[Option]) -> list[str]:
    by_text: dict[str, tuple[int, Any]] = {}
    for rank, item in enumerate(scored, start=1):
        text = str(getattr(item.order, "message", None) or item.order)
        by_text.setdefault(text, (rank, item))
    top = float(scored[0].score) if scored else 0.0
    lines = []
    for opt in options:
        found = by_text.get(opt.order)
        if found is None:
            lines.append(f"{opt.id}: {opt.order}  [{opt.note}]")
            continue
        rank, item = found
        gap = float(item.score) - top
        engine = f"engine #{rank}, score {float(item.score):.1f}" + (
            " (top)" if rank == 1 else f" ({gap:+.1f})")
        lines.append(f"{opt.id}: {describe_order_text(battle, item.order)}  [{engine}; {opt.kind}]")
    return lines


def _history_lines(memory: Any, turn: int) -> list[str]:
    if memory is None:
        return []
    lines = []
    orders = [(t, o) for t, o in getattr(memory, "our_orders", []) if t < turn][-3:]
    for t, o in orders:
        lines.append(f"turn {t}: we played {o}")
    protects = getattr(memory, "opponent_protects", None)
    if protects:
        lines.append("Their Protect uses so far: " + ", ".join(
            f"{_species(k)} x{v}" for k, v in dict(protects).items()))
    if lines:
        lines.insert(0, "Recent turns:")
    return lines


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------


def build_packet(
    battle: Any,
    scored: Sequence[Any],
    *,
    llm_config: LLMConfig | None = None,
    team_plan: str = "",
    memory: Any = None,
    request_id: str,
    policy_config: Any = None,
) -> tuple[ContextPacket, list[Option]]:
    """Build the packet for OUR current decision. Never raises on a real battle object.

    `scored` is the engine ranking (`vgc.evaluator.ScoredOrder`, best first); option IDs
    come from `vgc.llm.packet.build_options` (P01 = engine top). `team_plan` is the
    owner's written plan for our team (`load_team_plan`). `memory` is the optional
    `BattleMemory` for recent-turn lines.
    """
    from vgc.models import PolicyConfig
    from vgc.sets import set_priors_for

    cfg = llm_config or LLMConfig()
    policy = policy_config or PolicyConfig()
    options = build_options(list(scored), cfg.max_options)
    turn = _safe(lambda: int(getattr(battle, "turn", 0) or 0), 0)
    priors = _safe(lambda: set_priors_for(policy), {})

    fixed = _safe(lambda: _fixed_for_battle(battle, team_plan, priors), "")
    if not fixed:
        fixed = "\n\n".join([RULES_DIGEST, REASONING_CHECKLIST])

    sections = [f"## TURN {turn}"]
    for title, fn in (
        ("FIELD", lambda: _field_lines(battle, turn)),
        ("BOARD (ours: exact; theirs: public info only)",
         lambda: _board_lines(battle) + _foe_lines(battle, priors)),
        ("ENGINE FACTS (computed by code from the engine's damage/speed math; do not recompute. "
         "Anything about a foe's hidden spread/item/ability is an ESTIMATE)",
         lambda: _engine_fact_lines(battle, priors, policy)),
        ("HISTORY", lambda: _history_lines(memory, turn)),
    ):
        lines = _safe(fn, [])
        if lines:
            sections.append(f"## {title}\n" + "\n".join(lines))
    option_lines = _safe(lambda: _option_lines(battle, scored, options), None) or [
        f"{o.id}: {o.order}  [{o.note}]" for o in options
    ]
    sections.append("## OPTIONS (answer with these IDs only)\n" + "\n".join(option_lines))
    packet = ContextPacket(
        fixed_text=fixed,
        turn_text="\n\n".join(sections),
        option_ids=tuple(o.id for o in options),
        request_id=request_id,
        turn=turn,
    )
    return packet, options
