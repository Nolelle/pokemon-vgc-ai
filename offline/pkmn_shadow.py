"""Shadow reader: @pkmn/client battle state, adapted to the audit's `BattleMechanicsState`.

`tools/pkmn_reader/reader.mjs` hosts one @pkmn/client `Battle` per (battle, perspective) and
is fed the SAME raw protocol lines a live poke-env client would get (unfiltered: cosmetic
lines included, `|request|` JSON included). `PkmnShadow.snapshot()` returns the client's
state as the same `BattleMechanicsState` dataclasses `vgc.mechanics_state.snapshot_battle`
builds from poke-env, so `audit_battle_parsing.compare_perspective` grades both readers with
IDENTICAL rules. Nothing here is used by the bot.

Adapter mappings (every place where the client's vocabulary is translated; an adapter bug must
never be counted against pkmn, so each is listed):

* species        client `speciesForme` (formechange volatile applied) -> `to_id`.
* types          client `types` getter + `addedType` (the comparator adds Showdown's added
                 type to the truth set, so the belief gets the client's added type too).
* status counter sleep -> `statusState.sleepTurns`; tox -> `statusState.toxicTurns`; else 0.
* boosts         client sparse table -> the 7 ids, missing = 0.
* item           own side: `item` ('' -> none). Foe: `item` if set -> known; else any
                 `lastItem` / 'knocked off' -> consumed; else unknown. (The client cannot tell
                 "never had one" from "unknown": `item` is '' for both.)
* ability        `ability` ('' -> None); `baseAbility`.
* revealed       own: always. Foe: a Pokemon has an `originalIdent` only once it was sent out
                 (`|poke|` team-preview entries have ident '' and hp 0; the client fills
                 their `name` with the species, so the name cannot be used).
* volatiles      client `volatiles` keys, minus client-only bookkeeping that is not a Showdown
                 volatile (`CLIENT_ONLY_VOLATILES`), with the same id translation the poke-env
                 snapshot applies (`vgc.mechanics_state._EFFECT_ALIASES`: the client files a
                 partial trap under the move's name -- `wrap`, `firespin`, `bind` -- where
                 Showdown says `partiallytrapped`; `stockpileN`/`perishN` likewise) and the same
                 drop list of effects Showdown does not keep as volatiles
                 (`_NON_VOLATILE_EFFECTS`: Supreme Overlord's `fallenN`, `futuresight`, ...).
* side conditions client `sideConditions` minus one-turn guards (`remove: true`, the audit
                 never compares them) and Future Sight / Doom Desire (Showdown slot
                 conditions, which the audit does not compare). Layers = client `level`.
* durations      the client keeps [minDuration, maxDuration] and decrements both on `|upkeep|`;
                 `minDuration` is "turns left assuming no hidden extender", the same
                 assumption poke-env's `_BASE_DURATION` arithmetic makes. It is converted to
                 the `turns` start-turn field the comparator's `_belief_remaining` expects.
                 A condition whose client duration is 0 (unknown) reports `turns=None`.
* weather/terrain client names (Sun, Rain, Sand, Hail, Snow, ...) -> Showdown ids.
* NOT TRACKED    `preparing` (two-turn moves; the client ignores `|-prepare|`) and the
                 Protect/stall counter. They report 0/False and the exemption registry's
                 `pkmn_not_tracked` entry (pkmn-only) labels the resulting findings, so they
                 are reported rather than silently passing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vgc.config import REPO_ROOT, SHOWDOWN_REPO
from vgc.damage import to_id
from vgc.mechanics_state import (
    _EFFECT_ALIASES,
    _NON_VOLATILE_EFFECTS,
    BOOST_IDS,
    BattleMechanicsState,
    EffectSnapshot,
    MoveSnapshot,
    PokemonMechanicsState,
    SideMechanicsState,
)
from vgc.node import find_node

READER_DIR = REPO_ROOT / "tools" / "pkmn_reader"
READER_SCRIPT = READER_DIR / "reader.mjs"

# Client-side bookkeeping entries in `Pokemon.volatiles` that are NOT Showdown volatiles:
# the client stores type changes, forme changes and "item was removed" as pseudo-volatiles,
# while Showdown keeps them in `types` / `species` / item fields.
CLIENT_ONLY_VOLATILES = frozenset(
    {"formechange", "typechange", "typeadd", "itemremoved", "airballoon", "transform"}
)  # (`transform`: Showdown keeps it as the Pokemon's `transformed` flag, reported separately)
# Side conditions the audit never compares with the truth (see module docstring).
SKIPPED_SIDE_CONDITIONS = frozenset({"futuresight", "doomdesire"})
LAYERED = frozenset({"spikes", "toxicspikes", "stealthrock", "stickyweb"})
WEATHER_IDS = {
    "Sand": "sandstorm", "Sun": "sunnyday", "Rain": "raindance", "Hail": "hail",
    "Snow": "snowscape", "Harsh Sunshine": "desolateland", "Heavy Rain": "primordialsea",
    "Strong Winds": "deltastream",
}  # fmt: skip
TERRAIN_IDS = {
    "Electric": "electricterrain", "Grassy": "grassyterrain",
    "Psychic": "psychicterrain", "Misty": "mistyterrain",
}  # fmt: skip


class PkmnReaderError(RuntimeError):
    """The Node reader process failed (died, bad self-check, protocol error)."""


@dataclass
class ReaderStats:
    """Per-process counters merged into the audit report."""

    feeds: int = 0
    snapshots: int = 0
    parse_errors: int = 0
    process_failures: int = 0
    snapshot_ms: list[float] = field(default_factory=list)  # node-side snapshot time
    feed_ms: list[float] = field(default_factory=list)  # node-side feed time per step
    roundtrip_ms: list[float] = field(default_factory=list)  # feed+snapshot as Python sees it
    error_samples: list[dict[str, Any]] = field(default_factory=list)
    error_kinds: dict[str, int] = field(default_factory=dict)


class PkmnReader:
    """The long-lived `reader.mjs` process (JSON lines, like `vgc.rl.env.SimWorker`)."""

    def __init__(self, showdown_repo: str | Path = SHOWDOWN_REPO) -> None:
        if not (READER_DIR / "node_modules" / "@pkmn" / "client").exists():
            raise PkmnReaderError(
                "tools/pkmn_reader/node_modules is missing: run `npm ci --ignore-scripts` there "
                "(node 22: ~/.nvm/versions/node/v22.*/bin)"
            )
        self._next_rid = 0
        self.stats = ReaderStats()
        self._process = subprocess.Popen(
            [find_node(), str(READER_SCRIPT), str(showdown_repo)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(READER_DIR),
        )
        assert self._process.stdout is not None
        first = self._process.stdout.readline()
        if not first:
            raise PkmnReaderError(f"reader exited at start: {self._read_stderr()}")
        hello = json.loads(first)
        if hello.get("fatal"):
            raise PkmnReaderError(f"reader failed to start: {hello['fatal']}")
        self.selfcheck = hello["selfcheck"]
        if not self.selfcheck.get("all_ok"):
            bad = {k: v for k, v in self.selfcheck.items() if isinstance(v, dict) and not v["ok"]}
            self.close()
            raise PkmnReaderError(f"M-C data self-check failed: {bad}")

    def _read_stderr(self) -> str:
        try:
            return self._process.stderr.read() if self._process.stderr else ""
        except Exception:
            return ""

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._process.poll() is not None:
            raise PkmnReaderError(f"reader exited with code {self._process.returncode}")
        self._next_rid += 1
        payload = {**payload, "rid": self._next_rid}
        assert self._process.stdin is not None and self._process.stdout is not None
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise PkmnReaderError(f"reader pipe broke: {exc}") from exc
        if not line:
            raise PkmnReaderError(f"reader closed stdout; stderr:\n{self._read_stderr()}")
        response = json.loads(line)
        if response.get("rid") != payload["rid"]:
            raise PkmnReaderError(f"rid mismatch {response.get('rid')} != {payload['rid']}")
        return response

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                assert self._process.stdin is not None
                self._process.stdin.close()
                self._process.wait(timeout=5)
            except (subprocess.TimeoutExpired, BrokenPipeError, AssertionError):
                self._process.kill()
                self._process.wait(timeout=5)

    def __enter__(self) -> PkmnReader:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class PkmnShadow:
    """One battle: a client per perspective, fed raw lines, snapshotted on demand."""

    def __init__(self, reader: PkmnReader, battle_id: str, perspectives: tuple[str, ...] = ("p1", "p2")):
        self.reader = reader
        self.battle_id = battle_id
        self.perspectives = perspectives
        self.dead = False  # a process failure ends this battle's pkmn grading
        self._step_ms = 0.0
        for side in perspectives:
            reader.request({"cmd": "open", "id": f"{battle_id}:{side}", "perspective": side})

    def feed(self, side: str, lines: list[str]) -> None:
        if self.dead:
            return
        stats = self.reader.stats
        if dump_dir := os.environ.get("VGC_PKMN_LINES_OUT"):
            # debugging aid: the exact lines this client receives, one JSON list per feed
            with open(Path(dump_dir) / f"{self.battle_id}_{side}.jsonl", "a") as fh:
                fh.write(json.dumps(lines) + "\n")
        t0 = time.perf_counter()
        try:
            res = self.reader.request({"cmd": "feed", "id": f"{self.battle_id}:{side}", "lines": lines})
        except PkmnReaderError as exc:
            self._fail(exc)
            return
        self._step_ms += (time.perf_counter() - t0) * 1000
        stats.feeds += 1
        stats.feed_ms.append(res.get("ms", 0.0))
        for err in res.get("errors") or []:
            stats.parse_errors += 1
            kind = err["error"].split("|")[0][:80]
            stats.error_kinds[kind] = stats.error_kinds.get(kind, 0) + 1
            if len(stats.error_samples) < 40:
                stats.error_samples.append({"battle": self.battle_id, "side": side, **err})

    def snapshot(self, side: str) -> dict[str, Any] | None:
        if self.dead:
            return None
        stats = self.reader.stats
        t0 = time.perf_counter()
        try:
            res = self.reader.request({"cmd": "snapshot", "id": f"{self.battle_id}:{side}"})
        except PkmnReaderError as exc:
            self._fail(exc)
            return None
        self._step_ms += (time.perf_counter() - t0) * 1000
        if not res.get("ok"):
            self._fail(PkmnReaderError(str(res.get("error"))))
            return None
        stats.snapshots += 1
        stats.snapshot_ms.append(res["ms"])
        return res["snapshot"]

    def end_step(self) -> None:
        """Close the books on one decision point: feed+snapshot round trips for BOTH sides."""
        if not self.dead:
            self.reader.stats.roundtrip_ms.append(self._step_ms)
        self._step_ms = 0.0

    def _fail(self, exc: Exception) -> None:
        self.dead = True
        self.reader.stats.process_failures += 1
        if len(self.reader.stats.error_samples) < 40:
            self.reader.stats.error_samples.append(
                {"battle": self.battle_id, "line": "", "error": f"PROCESS: {exc}"}
            )

    def close(self) -> None:
        for side in self.perspectives:
            try:
                self.reader.request({"cmd": "close", "id": f"{self.battle_id}:{side}"})
            except PkmnReaderError:
                pass


# --- adapter ----------------------------------------------------------------------------


class NullMemory:
    """Stands in for `BattleMemory` where the comparator reads poke-env-side bookkeeping.

    The comparator's foe-ability check falls back to `memory.opponent_abilities`, which is
    bot-side bookkeeping on top of poke-env; pkmn gets no such help (it must stand alone)."""

    opponent_abilities: dict[str, str] = {}


def _remaining_to_turns(base: int | None, remaining: int, turn: int) -> int | None:
    """Start-turn value `t` with `base - (turn - t) == remaining` (see module docstring)."""
    if base is None or remaining <= 0:
        return None
    return turn - base + remaining


def _pokemon_state(p: dict[str, Any], *, opponent: bool) -> PokemonMechanicsState:
    boosts = p["boosts"]
    species_id = to_id(p["speciesForme"])
    types = tuple(to_id(t) for t in p["types"])
    if p["addedType"]:
        types = types + (to_id(p["addedType"]),)
    status = p["status"] or None
    if status == "slp":
        counter = p["sleepTurns"]
    elif status == "tox":
        counter = p["toxicTurns"]
    else:
        counter = 0
    if opponent:
        if p["item"]:
            item_state, item_id = "known", p["item"]
        elif p["lastItem"] or p["itemEffect"] == "knocked off" or p["lastItemEffect"]:
            item_state, item_id = "consumed", None
        else:
            item_state, item_id = "unknown", None
    else:
        item_state, item_id = ("known", p["item"]) if p["item"] else ("none", None)
    effect_ids = {
        _EFFECT_ALIASES.get(vid, vid)
        for vid in p["volatiles"]
        if vid not in CLIENT_ONLY_VOLATILES and vid not in _NON_VOLATILE_EFFECTS
    }
    effects = tuple(
        EffectSnapshot(id=vid, turns=None, raw_value=None, counter_kind="elapsed_actions")
        for vid in sorted(effect_ids)
    )
    moves = tuple(
        sorted(
            (
                MoveSnapshot(
                    id=m["id"], current_pp=None, max_pp=None, disabled=False,
                    disabled_reason=None, last_used=False,
                )
                for m in p["moves"]
            ),
            key=lambda m: m.id,
        )
    )
    ability = p["ability"] or None
    return PokemonMechanicsState(
        species_id=species_id,
        base_species_id=to_id(p["baseSpeciesForme"]) or None,
        name=p["name"],
        level=p["level"],
        gender=p["gender"],
        types=types,
        base_types=(),
        current_hp=p["hp"],
        max_hp=p["maxhp"],
        fainted=p["fainted"],
        active=p["active"],
        revealed=bool(p["ident"]) if opponent else True,
        selected_in_preview=False,
        stats=(),
        stat_points=None,
        individual_values=None,
        nature=None,
        boosts=tuple((b, int(boosts.get(b, 0))) for b in BOOST_IDS),
        status=status,
        status_counter=int(counter or 0),
        effects=effects,
        item_id=item_id,
        item_state=item_state,
        item_known=item_state != "unknown",
        ability_id=ability,
        base_ability_id=p["baseAbility"] or None,
        temporary_ability_id=None,
        forme_change_ability_id=None,
        ability_known=not opponent or ability is not None,
        base_move_ids=(),
        moves=moves,
        last_move_id=p["lastMove"] or None,
        mimic_move_id=None,
        preparing_move_id=None,  # NOT TRACKED by pkmn
        preparing_target=None,
        preparing=False,
        must_recharge=False,
        protect_counter=0,  # NOT TRACKED by pkmn
        first_turn=False,
        transformed=p["transformed"],
        mega_evolved=False,
        terastallized=bool(p["teraType"]),
        tera_type=to_id(p["teraType"]) or None,
        weight=None,
    )


def _side_state(side: dict[str, Any], turn: int, *, opponent: bool) -> SideMechanicsState:
    from audit_battle_parsing import _BASE_DURATION

    pokemon = tuple(_pokemon_state(p, opponent=opponent) for p in side["team"])
    conditions: list[EffectSnapshot] = []
    for cid, c in side["conditions"].items():
        if c["remove"] or cid in SKIPPED_SIDE_CONDITIONS:
            continue
        if cid in LAYERED:
            conditions.append(EffectSnapshot(cid, c["level"], c["level"], "layers"))
        else:
            turns = _remaining_to_turns(_BASE_DURATION.get(cid), c["minDuration"], turn)
            conditions.append(EffectSnapshot(cid, turns, turns, "side_start_turn"))
    active = list(side["active"]) + [None] * (2 - len(side["active"]))
    return SideMechanicsState(
        pokemon=pokemon,
        active_species=tuple(None for _ in active),
        side_conditions=tuple(sorted(conditions, key=lambda e: e.id)),
        force_switch=(None, None),
        trapped=(None, None),
        maybe_trapped=(None, None),
        can_mega_evolve=(None, None),
        used_mega_evolution=False,
        can_dynamax=(None, None),
        used_dynamax=False,
        can_tera=(None, None),
        used_tera=False,
        can_z_move=(None, None),
        used_z_move=False,
    )


def adapt(snap: dict[str, Any], me: str) -> tuple[BattleMechanicsState, SimpleNamespace]:
    """pkmn snapshot JSON (from `me`'s client) -> (`BattleMechanicsState`, battle-like view).

    The view carries the two things `compare_perspective` reads off the poke-env battle
    object itself: `active_pokemon` / `opponent_active_pokemon` (objects with `.name`)."""
    from audit_battle_parsing import _BASE_DURATION

    foe = "p2" if me == "p1" else "p1"
    turn = int(snap["turn"] or 0)
    weather: tuple[EffectSnapshot, ...] = ()
    if snap["weather"]:
        w = snap["weather"]
        wid = WEATHER_IDS.get(w["name"], to_id(w["name"]))
        turns = _remaining_to_turns(_BASE_DURATION.get(wid), w["minDuration"], turn)
        weather = (EffectSnapshot(wid, turns, turns, "start_turn"),)
    fields: list[EffectSnapshot] = []
    if snap["terrain"]:
        t = snap["terrain"]
        tid = TERRAIN_IDS.get(t["name"], to_id(t["name"]) + "terrain")
        turns = _remaining_to_turns(_BASE_DURATION.get(tid), t["minDuration"], turn)
        fields.append(EffectSnapshot(tid, turns, turns, "start_turn"))
    for pid, s in snap["pseudoWeather"].items():
        turns = _remaining_to_turns(_BASE_DURATION.get(pid), s["minDuration"], turn)
        fields.append(EffectSnapshot(pid, turns, turns, "start_turn"))
    state = BattleMechanicsState(
        format_id=to_id(snap["tier"]), generation=9, game_type="doubles", turn=turn,
        max_team_size=None, team_size=None, team_preview=False, commanding=False,
        reviving=False, waiting=False, finished=False, won=False, lost=False,
        fields=tuple(sorted(fields, key=lambda e: e.id)), weather=weather,
        available_moves=(), available_switches=(), valid_order_count=None,
        last_request_json=None, our_preview_species=(), opponent_preview_species=(),
        our_side=_side_state(snap[me], turn, opponent=False),
        opponent_side=_side_state(snap[foe], turn, opponent=True),
    )  # fmt: skip

    def actives(side: dict[str, Any]) -> list[Any]:
        names = list(side["active"]) + [None] * (2 - len(side["active"]))
        return [SimpleNamespace(name=n) if n is not None else None for n in names]

    view = SimpleNamespace(
        active_pokemon=actives(snap[me]), opponent_active_pokemon=actives(snap[foe])
    )
    return state, view
