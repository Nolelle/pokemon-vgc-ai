"""How many end-of-turn ticks each field/side condition has really had.

poke-env records when a weather, terrain, Trick Room or side condition began as the
battle's turn number at the moment the line arrived. Two things make that wrong:

* **Weather is restamped every turn.** Showdown re-announces an ongoing weather each
  residual with ``|-weather|SunnyDay|[upkeep]``; poke-env treats every such line as a
  fresh start (``abstract_battle.py``: ``self._weather = {W: self.turn}``), so a sun on
  its last turn looked brand new.
* **Switch-in setters lose a turn.** Drought, Psychic Surge, etc. fire on switch-in --
  at turn 0 or after a faint at the end of a turn, i.e. AFTER that turn's residual has
  already ticked durations down. Showdown does not charge them that turn; "current turn
  minus start turn" did.

Measured on 16 direct games (2026-10-05): the public mirror rebuilt weather with the
wrong remaining duration 56/72 times and terrain 108/134 times (Trick Room 48/48 and
Tailwind 12/12 were right). Every exact branch and continuation inherited those errors.

This module counts Showdown's own ``|upkeep|`` lines (emitted once per turn, after all
residual effects) since each condition's start line, which is exactly how many times
Showdown has decremented its duration. `vgc.mechanics_state.snapshot_battle` reports
the condition's start turn as ``turn - ticks`` so every consumer's existing
"turn - start" arithmetic yields the true elapsed count.

Duration extensions (Heat/Damp/Smooth/Icy Rock and Terrain Extender: 5 -> 8 turns) are
known from the start when the setter (the line's ``[of]`` Pokemon, else the last move's
user) visibly holds the item -- always true for our own Pokemon, and for an opponent once
the item is revealed. Otherwise they become known when a condition outlives its base
duration. Either way the elapsed count is reported net of the 3-turn extension, so
"base - elapsed" is the true remaining time. An unrevealed opponent extender before that
point is still read as 5 turns.
"""

from __future__ import annotations

from typing import Any

from vgc.damage import to_id
from vgc.poke_env_compat import fix_enabled

CLOCK_ATTRIBUTE = "_vgc_condition_clock"

# Base durations (Showdown data/conditions.ts + moves.ts; the champions mod changes none).
_BASE_DURATION = {
    "sunnyday": 5,
    "raindance": 5,
    "sandstorm": 5,
    "snowscape": 5,
    "snow": 5,
    "hail": 5,
    "electricterrain": 5,
    "grassyterrain": 5,
    "psychicterrain": 5,
    "mistyterrain": 5,
    "trickroom": 5,
    "tailwind": 4,
    "reflect": 5,
    "lightscreen": 5,
    "auroraveil": 5,
}
# Light Clay stretches the screens 5 -> 8 turns when their setter holds it.
_SCREENS = frozenset(("reflect", "lightscreen", "auroraveil"))
# Item extensions that only become visible when a condition outlives its base duration.
_EXTENSION = {
    **{weather: 3 for weather in ("sunnyday", "raindance", "sandstorm", "snowscape", "snow")},
    **{terrain: 3 for terrain in ("electricterrain", "grassyterrain", "psychicterrain")},
    "mistyterrain": 3,
    **{screen: 3 for screen in _SCREENS},
}
_TERRAINS = frozenset(("electricterrain", "grassyterrain", "psychicterrain", "mistyterrain"))


def _condition_id(raw: str) -> str:
    text = raw.split(":", 1)[1] if raw.lower().startswith("move:") else raw
    return to_id(text)


def _clock(battle: Any) -> dict[tuple[str, ...], Any]:
    clock = vars(battle).get(CLOCK_ATTRIBUTE)
    if clock is None:
        clock = {}
        setattr(battle, CLOCK_ATTRIBUTE, clock)
    return clock


_TIMED = ("weather", "field", "side")


def _setter(split: list[str], clock: dict) -> str | None:
    """Who set the condition: the line's ``[of]`` Pokemon, else the last move's user."""

    for part in split[3:]:
        if part.startswith("[of] "):
            return part[len("[of] ") :]
    return clock.get(("last_move_user",))


def _start(battle: Any, clock: dict, key: tuple[str, ...], setter: str | None) -> None:
    clock[key] = 0
    clock[("setter", *key)] = setter
    # Showdown fixes the duration when the condition starts; losing the rock later
    # (Knock Off, Trick) does not shorten it, so record what the setter held THEN.
    clock[("extended", *key)] = _setter_holds_extender(battle, key[-1], setter)


def _end(clock: dict, key: tuple[str, ...]) -> None:
    clock.pop(key, None)
    clock.pop(("setter", *key), None)
    clock.pop(("extended", *key), None)


def observe_condition_line(battle: Any, split: list[str]) -> None:
    """Update ``battle``'s condition clock from one ``|``-split protocol line."""

    if battle is None or len(split) < 2:
        return
    tag = split[1]
    if tag not in (
        "upkeep", "move", "-weather", "-fieldstart", "-fieldend", "-sidestart", "-sideend"
    ):
        return
    clock = _clock(battle)
    if tag == "upkeep":
        for key in clock:
            if key[0] in _TIMED:
                clock[key] += 1
    elif tag == "move" and len(split) > 2:
        clock[("last_move_user",)] = split[2]
    elif tag == "-weather" and len(split) > 2:
        starting = split[2] != "none" and "[upkeep]" not in split[3:]
        if split[2] == "none" or starting:
            for key in [key for key in clock if key[0] == "weather"]:
                _end(clock, key)
        if starting:
            _start(battle, clock, ("weather", to_id(split[2])), _setter(split, clock))
    elif tag in ("-fieldstart", "-fieldend") and len(split) > 2:
        effect_id = _condition_id(split[2])
        if tag == "-fieldend":
            _end(clock, ("field", effect_id))
            return
        if effect_id in _TERRAINS:
            for key in [key for key in clock if key[0] == "field" and key[1] in _TERRAINS]:
                _end(clock, key)
        _start(battle, clock, ("field", effect_id), _setter(split, clock))
    elif tag in ("-sidestart", "-sideend") and len(split) > 3:
        key = ("side", split[2][:2], _condition_id(split[3]))
        if tag == "-sideend":
            _end(clock, key)
        else:
            _start(battle, clock, key, _setter(split, clock))


# Item that extends each condition 5 -> 8 turns when its SETTER holds it.
_EXTENDER = {
    "sunnyday": "heatrock",
    "raindance": "damprock",
    "sandstorm": "smoothrock",
    "snowscape": "icyrock",
    "snow": "icyrock",
    **{terrain: "terrainextender" for terrain in _TERRAINS},
    **{screen: "lightclay" for screen in _SCREENS},
}


def _setter_holds_extender(battle: Any, effect_id: str, setter: str | None) -> bool:
    item = _EXTENDER.get(effect_id)
    if not item or not setter or ":" not in setter:
        return False
    if effect_id in _SCREENS and not fix_enabled("screen_clock"):
        return False
    role = getattr(battle, "player_role", None)
    team = getattr(battle, "team", None) if setter[:2] == role else getattr(
        battle, "opponent_team", None
    )
    # Team keys drop the slot letter: "p1a: Torkoal" -> "p1: Torkoal".
    mon = (team or {}).get(setter[:2] + setter[3:])
    return mon is not None and to_id(str(getattr(mon, "item", "") or "")) == item


def elapsed_ticks(battle: Any, kind: str, effect_id: str, side: str | None = None) -> int | None:
    """Duration ticks Showdown has charged ``effect_id`` so far, or None if unseen.

    ``kind`` is ``weather``/``field``/``side``; ``side`` is ``p1``/``p2`` for side
    conditions. A condition that outlived its base duration was item-extended; its count
    is returned net of the extension so ``base - elapsed`` is the true remaining time.
    """

    clock = vars(battle).get(CLOCK_ATTRIBUTE) or {}
    key = (kind, side, effect_id) if kind == "side" else (kind, effect_id)
    ticks = clock.get(key)
    if ticks is None:
        return None
    base = _BASE_DURATION.get(effect_id)
    if effect_id in _SCREENS and not fix_enabled("screen_clock"):
        return ticks
    extended = bool(clock.get(("extended", *key)))
    setter = clock.get(("setter", *key))
    own_setter = (
        fix_enabled("screen_clock")
        and bool(setter)
        and setter[:2] == getattr(battle, "player_role", None)
    )
    # (Our own item is known from the start; a later Trick / Thief must not stretch a screen.)
    if not extended and not own_setter and _setter_holds_extender(battle, effect_id, setter):
        # Revealed after the start (e.g. an opponent's rock shown later): it was there then.
        extended = clock[("extended", *key)] = True
    if extended or (base is not None and ticks >= base):
        ticks -= _EXTENSION.get(effect_id, 0)
    # May be negative for a known-extended condition: base - elapsed must reach 8 - ticks.
    return ticks


def remaining_turns(
    battle: Any, kind: str, effect_id: str, side: str | None = None
) -> int | None:
    """Turns ``effect_id`` still lasts INCLUDING the one being decided, or None if unknown.

    ``base_duration - elapsed_ticks`` (net of any item extension, see `elapsed_ticks`).
    None means the tracker has no entry or the effect has no timer (Desolate Land and
    friends) -- callers should treat that as "lasts through the horizon". Never below 1,
    since a condition still on the board is active for the turn being decided.
    """

    ticks = elapsed_ticks(battle, kind, effect_id, side)
    base = _BASE_DURATION.get(effect_id)
    if ticks is None or base is None:
        return None
    return max(1, base - ticks)


def base_duration(effect_id: str) -> int | None:
    """Showdown's base duration in turns for a timed condition (None if untimed)."""

    return _BASE_DURATION.get(effect_id)
