"""Systematic check of the FAST calculators against the real Showdown engine.

The fast search (`vgc.search`) and the quick scorer (`vgc.evaluator`) rely on our own maths:
`vgc.damage.damage_range`, `vgc.stats`, `vgc.accuracy.hit_probability`, effective Speed and
priority ordering, `vgc.priority_rules`. If any of it is wrong a good move can be cut before the
exact judge ever sees it. This module builds controlled one-turn engine scenarios
(`tools/calc_probe.mjs`) for every legal move, ability and item, runs our calculators on the same
situation, and compares them:

* ``damage``   -- our 16 damage rolls vs the engine's 16 rolls (exact vectors, crits off), including
                  whether the engine delivers any damage at all (immunities, absorbs, redirection,
                  Psychic Terrain / Dazzling priority blocks);
* ``order``    -- who moves first (Speed, Tailwind, Trick Room, paralysis, items, abilities, priority
                  brackets and priority-changing abilities) against the order the engine plays;
* ``accuracy`` -- our hit probability against the engine's final integer accuracy.

Every (category, entry) is classified:

``EXACT``              every exercised scenario matches the engine bit for bit.
``APPROXIMATE``        mismatches exist but are small (<= ``APPROX_BOUND`` relative error on mean
                       damage / accuracy) and never change who acts first or whether a hit lands.
``UNSUPPORTED-SAFE``   our code reports it cannot compute the entry (``move_supported`` is False) so
                       consumers score no damage rather than a wrong number.
``WRONG``              a silent disagreement: the consumers would act on a wrong number.

The CLI is `offline/sweep_calculator_vs_engine.py`; the committed result is
`data/champions/calculator_coverage.json`; `tests/test_calculator_sweep.py` re-runs a fixed subset
and fails if an entry recorded as EXACT stops matching.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from vgc.accuracy import hit_probability
from vgc.config import DATA_DIR, REPO_ROOT, SHOWDOWN_REPO
from vgc.damage import FieldState, PokemonState, damage_range, to_id
from vgc.data import load_moves, load_species, load_typechart
from vgc.evaluator import field_effective_speed, resolves_before
from vgc.node import find_node
from vgc.priority_rules import psychic_terrain_blocks

PROBE_SCRIPT = REPO_ROOT / "tools" / "calc_probe.mjs"
COVERAGE_PATH = DATA_DIR / "calculator_coverage.json"

# A mismatch no larger than this (relative error of the mean damage / of the hit chance) is
# APPROXIMATE rather than WRONG. 5% is below the spread between damage rolls (15%), so a
# decision based on it only changes at knife-edge KO thresholds.
APPROX_BOUND = 0.05

SLOTS = ("p1a", "p1b", "p2a", "p2b")
WEATHER_ENGINE = {"sun": "sunnyday", "rain": "raindance", "sand": "sandstorm", "snow": "snowscape"}
TERRAIN_ENGINE = {
    "electric": "electricterrain",
    "grassy": "grassyterrain",
    "psychic": "psychicterrain",
    "misty": "mistyterrain",
}
WEATHERS = tuple(WEATHER_ENGINE)
TERRAINS = tuple(TERRAIN_ENGINE)
SCREENS = ("reflect", "lightscreen", "auroraveil")

SP_ATTACKER = {"hp": 2, "atk": 32, "spa": 32}
SP_DEFENDER = {"hp": 32, "def": 17, "spd": 17}
SP_DUMMY = {"hp": 32, "def": 17, "spd": 17}

STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")

# Entries the scenario family cannot represent; reported as NOT_EXERCISED with the reason.
UNTESTABLE: dict[str, str] = {
    "imposter": "transforms into the opposing Pokemon on switch-in (stats, moves, ability copied)",
}


# --- legal entities, usage weights ------------------------------------------------------------


@lru_cache(maxsize=1)
def _catalog() -> dict:
    with (DATA_DIR / "mechanics_catalog.json").open() as file:
        return json.load(file)


@lru_cache(maxsize=1)
def legal_moves() -> list[str]:
    return sorted(_catalog()["legal_entities"]["moves"])


@lru_cache(maxsize=1)
def legal_abilities() -> list[str]:
    return sorted(_catalog()["legal_entities"]["abilities"])


@lru_cache(maxsize=1)
def legal_items() -> list[str]:
    return sorted(_catalog()["legal_entities"]["items"])


@lru_cache(maxsize=1)
def legal_species() -> list[str]:
    return sorted(_catalog()["legal_entities"]["species"])


@lru_cache(maxsize=1)
def usage_weights() -> dict[str, dict[str, float]]:
    """Real-play frequency per entry, normalised to sum to 1 inside each kind.

    Moves come from the M-C replay corpus set priors (`data/usage/set_priors.json`, how often
    each move was seen on a Pokemon). The corpus only records an ability or item when a replay
    revealed it, which is biased toward the few that announce themselves, so abilities and items
    use Smogon's M-C usage statistics (`data/usage/gen9championsvgc2026regmc-1760.json`, the same
    file `data/usage/spreads.json` is built from) and fall back to the corpus when it is absent.
    """

    out: dict[str, dict[str, float]] = {"move": {}, "ability": {}, "item": {}}
    priors_path = DATA_DIR.parent / "usage" / "set_priors.json"
    if priors_path.exists():
        with priors_path.open() as file:
            species = json.load(file)["species"]
        for data in species.values():
            for key, kind in (("moves", "move"), ("abilities", "ability"), ("items", "item")):
                for entry, count in (data.get(key) or {}).items():
                    out[kind][entry] = out[kind].get(entry, 0.0) + float(count)
    chaos_path = DATA_DIR.parent / "usage" / "gen9championsvgc2026regmc-1760.json"
    if chaos_path.exists():
        with chaos_path.open() as file:
            chaos = json.load(file)["data"]
        abilities: dict[str, float] = {}
        items: dict[str, float] = {}
        for data in chaos.values():
            for entry, weight in (data.get("Abilities") or {}).items():
                abilities[entry] = abilities.get(entry, 0.0) + float(weight)
            for entry, weight in (data.get("Items") or {}).items():
                items[entry] = items.get(entry, 0.0) + float(weight)
        if abilities:
            out["ability"] = abilities
        if items:
            out["item"] = items
    for kind, table in out.items():
        total = sum(table.values()) or 1.0
        out[kind] = {entry: weight / total for entry, weight in table.items()}
    return out


# --- type helpers ---------------------------------------------------------------------------


def type_multiplier(move_type: str, defender_types: Iterable[str]) -> float:
    chart = load_typechart()
    mult = 1.0
    for defender_type in defender_types:
        code = chart[defender_type.lower()]["damageTaken"].get(move_type, 0)
        mult *= {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}[code]
    return mult


def _plain_species(sid: str) -> bool:
    data = load_species()[sid]
    return not data.get("isMega") and not data.get("forme") and not data.get("battleOnly")


@lru_cache(maxsize=None)
def defender_for(move_type: str, mult: float) -> str | None:
    """A bulky plain species whose typing gives ``mult`` against ``move_type``."""

    species = load_species()
    best: tuple[int, str] | None = None
    for sid in legal_species():
        if not _plain_species(sid):
            continue
        if type_multiplier(move_type, species[sid]["types"]) != mult:
            continue
        stats = species[sid]["baseStats"]
        score = stats["hp"] + stats["def"] + stats["spd"]
        if best is None or (score, sid) > best:
            best = (score, sid)
    return best[1] if best else None


@lru_cache(maxsize=None)
def stab_attacker_for(move_type: str) -> str:
    species = load_species()
    best: tuple[int, int, str] | None = None
    for sid in legal_species():
        if not _plain_species(sid) or move_type not in species[sid]["types"]:
            continue
        stats = species[sid]["baseStats"]
        single = 1 if len(species[sid]["types"]) == 1 else 0
        key = (single, stats["atk"] + stats["spa"], sid)
        if best is None or key > best:
            best = key
    return best[2]


NEUTRAL_ATTACKERS = ("garchomp", "incineroar", "gardevoir", "corviknight", "gengar")


def neutral_attacker_for(move_type: str) -> str:
    species = load_species()
    for sid in NEUTRAL_ATTACKERS:
        if sid in species and move_type not in species[sid]["types"]:
            return sid
    return "snorlax"


# --- scenario model -----------------------------------------------------------------------


@dataclass
class Mon:
    species: str
    moves: tuple[str, ...] = ("splash",)
    sp: dict[str, int] = field(default_factory=dict)
    nature: str = "serious"
    ability: str = ""
    item: str = ""
    boosts: dict[str, int] = field(default_factory=dict)
    status: str | None = None
    hp_frac: float | None = None
    no_item: bool = False
    faint: bool = False

    def pokeset(self) -> dict:
        species = load_species()[self.species]
        evs = {stat: self.sp.get(stat, 0) for stat in ("hp", "atk", "def", "spa", "spd", "spe")}
        return {
            "species": species["name"],
            "name": species["name"],
            "item": self.item,
            "ability": self.ability,
            "moves": list(self.moves),
            "nature": self.nature.capitalize(),
            "evs": evs,
            "ivs": {stat: 31 for stat in ("hp", "atk", "def", "spa", "spd", "spe")},
            "level": 50,
            "gender": "M",
        }

    def state_payload(self) -> dict | None:
        payload: dict[str, Any] = {}
        if self.boosts:
            payload["boosts"] = self.boosts
        if self.status:
            payload["status"] = self.status
        if self.hp_frac is not None:
            payload["hpFrac"] = self.hp_frac
        if self.no_item:
            payload["noItem"] = True
        if self.faint:
            payload["faint"] = True
        return payload or None

    def state(self) -> PokemonState:
        state = PokemonState(
            self.species,
            sp_spread=dict(self.sp),
            nature=self.nature,
            boosts=dict(self.boosts),
            status=self.status,
            item=None if self.no_item else (self.item or None),
            ability=self.ability or None,
        )
        if self.hp_frac is not None:
            state.current_hp = max(1, int(state.max_hp() * self.hp_frac))
        state.item_lost = bool(self.no_item and self.item)
        return state


@dataclass
class Field:
    weather: str | None = None
    terrain: str | None = None
    trick_room: bool = False
    gravity: bool = False
    p1side: tuple[str, ...] = ()
    p2side: tuple[str, ...] = ()

    def payload(self) -> dict:
        out: dict[str, Any] = {}
        if self.weather:
            out["weather"] = WEATHER_ENGINE[self.weather]
        if self.terrain:
            out["terrain"] = TERRAIN_ENGINE[self.terrain]
        if self.trick_room:
            out["trickroom"] = True
        if self.gravity:
            out["gravity"] = True
        if self.p1side:
            out["p1side"] = list(self.p1side)
        if self.p2side:
            out["p2side"] = list(self.p2side)
        return out


@dataclass
class Case:
    kind: str  # damage | order | accuracy
    entry_kind: str  # move | ability | item | field | species
    entry: str
    variant: str
    mons: dict[str, Mon]
    moves: dict[str, str]
    fld: Field = field(default_factory=Field)
    targets: dict[str, int] = field(default_factory=dict)
    seeds: tuple[tuple[int, int, int, int], ...] = ((1, 2, 3, 4),)
    turns: int = 3
    mega: dict[str, bool] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    control: "Case | None" = None

    @property
    def cid(self) -> str:
        return f"{self.kind}|{self.entry_kind}|{self.entry}|{self.variant}"

    def request(self) -> dict:
        state = {slot: mon.state_payload() for slot, mon in self.mons.items()}
        return {
            "kind": "turn",
            "p1": [self.mons["p1a"].pokeset(), self.mons["p1b"].pokeset()],
            "p2": [self.mons["p2a"].pokeset(), self.mons["p2b"].pokeset()],
            "moves": self.moves,
            "targets": self.targets,
            "state": {slot: payload for slot, payload in state.items() if payload},
            "field": self.fld.payload(),
            "seeds": [list(seed) for seed in self.seeds],
            "turns": self.turns,
            "mega": self.mega,
        }


def dummy_mon(species: str = "snorlax", **kwargs: Any) -> Mon:
    return Mon(species, sp=dict(SP_DUMMY), **kwargs)


# --- probe workers ------------------------------------------------------------------------------


class ProbeWorker:
    """One long-lived `tools/calc_probe.mjs` process (JSON lines over stdin/stdout)."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [find_node(), str(PROBE_SCRIPT), str(SHOWDOWN_REPO)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lock = threading.Lock()
        self.rid = 0

    def request(self, payload: dict) -> dict:
        with self.lock:
            self.rid += 1
            payload = {**payload, "rid": self.rid}
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write(json.dumps(payload) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("calc_probe.mjs exited unexpectedly")
            return json.loads(line)

    def close(self) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def run_cases(cases: list[Case], workers: int = 8) -> list[dict]:
    """Engine results for ``cases`` (same order); each item is the probe's per-seed result list."""

    pool = [ProbeWorker() for _ in range(max(1, workers))]
    local = threading.local()
    counter = {"next": 0}
    counter_lock = threading.Lock()

    def worker_for_thread() -> ProbeWorker:
        if not hasattr(local, "worker"):
            with counter_lock:
                local.worker = pool[counter["next"] % len(pool)]
                counter["next"] += 1
        return local.worker

    def one(case: Case) -> dict:
        try:
            return worker_for_thread().request(case.request())
        except Exception as exc:  # noqa: BLE001
            return {"results": [], "error": f"{type(exc).__name__}: {exc}"}

    try:
        with ThreadPoolExecutor(max_workers=len(pool)) as executor:
            return list(executor.map(one, cases, chunksize=64))
    finally:
        for worker in pool:
            worker.close()


# --- shared builders ---------------------------------------------------------------------------


def _move(move_id: str) -> dict:
    return load_moves()[move_id]


def attacker_nature(category: str) -> str:
    return "adamant" if category == "Physical" else "modest"


def base_mons(
    move_id: str,
    *,
    attacker_species: str | None = None,
    defender_species: str | None = None,
    attacker_extra: dict | None = None,
    defender_extra: dict | None = None,
) -> dict[str, Mon]:
    md = _move(move_id)
    move_type = md["type"]
    attacker_species = attacker_species or neutral_attacker_for(move_type)
    defender_species = defender_species or defender_for(move_type, 1.0) or "snorlax"
    attacker = Mon(
        attacker_species,
        moves=(move_id, "splash"),
        sp=dict(SP_ATTACKER),
        nature=attacker_nature(md["category"]),
        **(attacker_extra or {}),
    )
    defender = Mon(
        defender_species,
        sp=dict(SP_DEFENDER),
        nature="relaxed" if md["category"] == "Physical" else "calm",
        **(defender_extra or {}),
    )
    return {"p1a": attacker, "p1b": dummy_mon(), "p2a": defender, "p2b": dummy_mon()}


def spread_target_count(move_id: str, mons: dict[str, Mon]) -> int:
    """Targets the engine will pick for ``move_id`` (it counts fainted-ness at selection time)."""

    target = _move(move_id)["target"]
    foes = sum(1 for slot in ("p2a", "p2b") if not mons[slot].faint)
    ally = 1 if not mons["p1b"].faint else 0
    if target == "allAdjacentFoes":
        return foes
    if target == "allAdjacent":
        return foes + ally
    return 1


NEEDS_TARGET = {"normal", "any", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe"}


# --- damage cases --------------------------------------------------------------------------------

# Moves whose precondition the neutral scenario cannot satisfy by itself; the entry is flagged
# per variant so the report says "not exercised" rather than blaming the calculator.
FOE_ATTACKS = {"suckerpunch", "thunderclap", "upperhand"}


def damage_case(
    entry_kind: str,
    entry: str,
    variant: str,
    move_id: str,
    mons: dict[str, Mon],
    fld: Field | None = None,
    *,
    mega: bool = False,
    target_slot: str = "p2a",
    moves_override: dict[str, str] | None = None,
) -> Case:
    fld = fld or Field()
    moves = {"p1a": move_id, "p1b": "splash", "p2a": "splash", "p2b": "splash"}
    if move_id in FOE_ATTACKS:
        moves["p2a"] = "tackle"
        mons["p2a"].moves = ("tackle", "splash")
    if moves_override:
        moves.update(moves_override)
        for slot, mv in moves_override.items():
            if mv not in mons[slot].moves:
                mons[slot].moves = (mv, *mons[slot].moves)
    return Case(
        "damage",
        entry_kind,
        entry,
        variant,
        mons,
        moves,
        fld,
        targets={"p1a": 1 if target_slot == "p2a" else 2},
        mega={"p1a": mega} if mega else {},
        meta={"target_slot": target_slot},
    )


def move_variants(move_id: str) -> list[Case]:
    """The per-move damage scenario family (field, type, boosts, status, HP, items...)."""

    md = _move(move_id)
    move_type = md["type"]
    category = md["category"]
    cases: list[Case] = []

    def add(variant: str, fld: Field | None = None, **kwargs: Any) -> Case:
        extra_att = kwargs.pop("att", None)
        extra_def = kwargs.pop("dfn", None)
        mons = base_mons(
            move_id,
            attacker_species=kwargs.pop("attacker_species", None),
            defender_species=kwargs.pop("defender_species", None),
            attacker_extra=extra_att,
            defender_extra=extra_def,
        )
        for slot, overrides in (kwargs.pop("mon_overrides", None) or {}).items():
            for key, value in overrides.items():
                setattr(mons[slot], key, value)
        case = damage_case("move", move_id, variant, move_id, mons, fld, **kwargs)
        cases.append(case)
        return case

    add("base")
    # Type effectiveness classes against a bulky species of each class.
    for mult in (0.0, 0.25, 0.5, 2.0, 4.0):
        species = defender_for(move_type, mult)
        if species:
            add(f"type_x{mult}", defender_species=species)
    # STAB / no STAB, and Adaptability-free baselines.
    add("stab", attacker_species=stab_attacker_for(move_type))
    # Weather / terrain / screens.
    for weather in WEATHERS:
        add(f"weather_{weather}", Field(weather=weather))
    for terrain in TERRAINS:
        add(f"terrain_{terrain}", Field(terrain=terrain))
    for screen in SCREENS:
        add(f"screen_{screen}", Field(p2side=(screen,)))
    # Spread vs single target (the second foe and our ally fainted).
    if md["target"] in ("allAdjacent", "allAdjacentFoes"):
        add(
            "single_target",
            mon_overrides={"p2b": {"faint": True}, "p1b": {"faint": True}},
        )
    offensive = "atk" if category == "Physical" else "spa"
    defensive = "def" if category == "Physical" else "spd"
    add("atk_plus2", att={"boosts": {offensive: 2}})
    add("atk_minus2", att={"boosts": {offensive: -2}})
    add("def_plus2", dfn={"boosts": {defensive: 2}})
    add("def_minus1", dfn={"boosts": {defensive: -1}})
    add("other_boosts", att={"boosts": {"def": 2, "spe": 1}}, dfn={"boosts": {"atk": 1, "spa": 1}})
    for status in ("brn", "par", "psn", "tox", "slp", "frz"):
        add(f"att_{status}", att={"status": status})
        add(f"def_{status}", dfn={"status": status})
    for frac in (0.3, 0.6):
        add(f"att_hp_{frac}", att={"hp_frac": frac})
        add(f"def_hp_{frac}", dfn={"hp_frac": frac})
    add("def_item", dfn={"item": "Leftovers"})
    add("def_item_noitem_att", att={"item": "Leftovers"})
    # Weight / speed dependent moves: a heavy and a light/fast defender.
    add("def_heavy", defender_species="snorlax")
    add("def_light", defender_species="pikachu")
    add("def_fast", defender_species="regieleki")
    add("def_slow", defender_species="shuckle")
    add("att_slow", attacker_species="shuckle")
    add("att_heavy", attacker_species="snorlax")
    # Psychic Terrain priority blocks use the same family as terrains above; add a flyer.
    add("terrain_psychic_flyer", Field(terrain="psychic"), defender_species="corviknight")
    add("terrain_electric_flyer", Field(terrain="electric"), defender_species="corviknight")
    add("hp_both_low", att={"hp_frac": 0.2}, dfn={"hp_frac": 0.2})
    return cases


def ours_damage(case: Case, *, default_config: bool = True) -> dict[str, Any]:
    """Our prediction for a damage case: rolls (ascending), delivery flag, flags."""

    move_id = case.moves["p1a"]
    md = _move(move_id)
    slot = case.meta.get("target_slot", "p2a")
    attacker = case.mons["p1a"].state()
    fld = case.fld
    if case.mega.get("p1a"):
        from vgc.evaluator import mega_evolved_state
        from vgc.weather_abilities import SETTER_ABILITY_WEATHER

        attacker = mega_evolved_state(attacker)
        # The Mega's weather ability (Drought...) resolves before the move, as in the evaluator.
        new_weather = SETTER_ABILITY_WEATHER.get(attacker.ability or "")
        if new_weather:
            fld = Field(
                weather=new_weather,
                terrain=fld.terrain,
                trick_room=fld.trick_room,
                p1side=fld.p1side,
                p2side=fld.p2side,
            )
    defender = case.mons[slot].state()
    num_targets = spread_target_count(move_id, case.mons)
    state = FieldState(
        weather=fld.weather,
        terrain=fld.terrain,
        screens=frozenset(c for c in fld.p2side if c in SCREENS),
        trick_room=fld.trick_room,
        is_doubles=True,
        num_targets=num_targets,
    )
    result = damage_range(attacker, defender, move_id, state)
    breakdown = result.breakdown
    supported = bool(breakdown["move_supported"])
    blocked_terrain = bool(
        default_config and psychic_terrain_blocks(md, attacker, defender, fld.terrain)
    )
    delivers = supported and result.max_damage > 0 and not blocked_terrain
    return {
        "rolls": list(breakdown["rolls"]),
        "delivers": delivers,
        "supported": supported,
        "immune": bool(breakdown["immune"]),
        "terrain_block": blocked_terrain,
        "move_type": breakdown["move_type"],
        "mean": (sum(breakdown["rolls"]) / 16.0) if breakdown["rolls"] else 0.0,
        "ability_handled": breakdown["ability_handled"],
        "item_handled": breakdown["item_handled"],
    }


def _engine_call(case: Case, probe: dict) -> tuple[dict | None, str]:
    """The engine's damage call from p1a onto the case's target, plus a short reason string."""

    results = probe.get("results") or []
    if not results:
        return None, f"probe_error:{probe.get('error')}"
    result = results[0]
    if result.get("err"):
        return None, f"engine_error:{result['err'][:120]}"
    move_id = case.moves["p1a"]
    slot = case.meta.get("target_slot", "p2a")
    log = result.get("log") or []
    for call in result.get("calls", []):
        if call["source"] == "p1a" and call["move"] == move_id and call["target"] == slot:
            if call["rolls"] is not None:
                return call, "hit"
    used = any(line.startswith("|move|p1a:") for line in log)
    if not used:
        return None, "not_used"
    if any(line.startswith("|-fail|") or line.startswith("|-notarget") for line in log):
        return None, "failed"
    return None, "blocked"


def state_not_applied(case: Case, probe: dict) -> str | None:
    """Reason the requested pre-turn state did not exist in the engine (an ability that prevents
    the status, say), so the scenario cannot test anything; ``None`` when it was applied."""

    results = probe.get("results") or []
    if not results or "applied" not in results[0]:
        return None
    applied = results[0]["applied"]
    for slot, mon in case.mons.items():
        if mon.status and applied[slot]["status"] != mon.status:
            return f"status_prevented:{slot}:{mon.status}"
    return None


def engine_outcome(case: Case, probe: dict) -> dict[str, Any]:
    """What the engine did for a damage case: hit (ascending rolls), blocked, or unexercised."""

    not_applied = state_not_applied(case, probe)
    if not_applied:
        return {"state": "unexercised", "reason": not_applied}

    call, reason = _engine_call(case, probe)
    if call is None and reason in ("not_used", "failed") or reason.startswith(("probe_error", "engine_error")):
        return {"state": "unexercised", "reason": reason}
    if call is None:
        return {"state": "blocked"}
    return {
        "state": "hit",
        "rolls": [call["rolls"][15 - i] for i in range(16)],  # ascending 85%..100%
        "move_type": call.get("moveType"),
        "crit": bool(call.get("crit")),
    }


def roll_gap(ours_rolls: list[int], engine_rolls: list[int]) -> tuple[int, float]:
    """(largest absolute difference over the 16 rolls, that difference relative to the engine max)."""

    gap = max(abs(a - b) for a, b in zip(ours_rolls, engine_rolls, strict=True))
    return gap, gap / max(max(engine_rolls), 1)


def compare_damage(
    case: Case, probe: dict, control: Case | None = None, control_probe: dict | None = None
) -> dict[str, Any]:
    """Compare one damage case; with a ``control`` (same scenario minus the ability/item under
    test) the entry is judged on its own EFFECT, so baseline inexactness shared by both cannot
    be blamed on it."""

    engine = engine_outcome(case, probe)
    ours = ours_damage(case)
    out: dict[str, Any] = {"cid": case.cid, "variant": case.variant}
    if engine["state"] == "unexercised":
        out.update(status="unexercised", reason=engine["reason"])
        return out
    out["ours_supported"] = ours["supported"]
    if engine["state"] == "blocked":
        if not ours["delivers"]:
            out["status"] = "match"
            return out
        out.update(status="mismatch", reason="ours_credits_blocked_damage", abs_diff=999, rel_err=1.0)
        return out
    engine_rolls = engine["rolls"]
    if not ours["supported"]:
        if control is not None and not ours_damage(control)["supported"]:
            # The move itself is unsupported with or without the entry: not the entry's doing.
            out.update(status="unexercised", reason="move_unsupported_baseline")
            return out
        out.update(status="mismatch", reason="ours_unsupported", abs_diff=999, rel_err=1.0)
        return out
    if not ours["delivers"]:
        out.update(status="mismatch", reason="ours_zero_engine_damages", abs_diff=999, rel_err=1.0)
        return out
    gap, rel = roll_gap(ours["rolls"], engine_rolls)
    if gap == 0 and engine["move_type"] == ours["move_type"]:
        out["status"] = "match"
        return out
    detail = {
        "ours_range": [ours["rolls"][0], ours["rolls"][-1]],
        "eng_range": [engine_rolls[0], engine_rolls[-1]],
    }
    if control is not None and control_probe is not None:
        control_engine = engine_outcome(control, control_probe)
        control_ours = ours_damage(control)
        if (
            control_engine["state"] == "hit"
            and control_ours["supported"]
            and control_ours["delivers"]
        ):
            eng_mean = sum(engine_rolls) / 16.0
            eng_c_mean = sum(control_engine["rolls"]) / 16.0
            ours_c_mean = control_ours["mean"]
            if eng_c_mean > 0 and ours_c_mean > 0:
                eng_ratio = eng_mean / eng_c_mean
                ours_ratio = ours["mean"] / ours_c_mean
                tolerance = 0.02 + 1.5 / eng_c_mean
                if abs(ours_ratio - eng_ratio) / max(eng_ratio, 1e-9) <= tolerance:
                    out["status"] = "match"
                    out["note"] = "effect_matches_baseline_off"
                    return out
                rel = abs(ours_ratio - eng_ratio) / max(eng_ratio, 1e-9)
                out.update(status="mismatch", reason="damage_differs", abs_diff=gap, rel_err=rel, **detail)
                return out
    out.update(status="mismatch", reason="damage_differs", abs_diff=gap, rel_err=rel, **detail)
    out["crit"] = engine.get("crit", False)
    return out


# --- classification & aggregation -----------------------------------------------------------------


@dataclass
class EntryResult:
    category: str
    entry_kind: str
    entry: str
    n: int = 0
    n_match: int = 0
    n_unexercised: int = 0
    n_mismatch: int = 0
    n_unsupported: int = 0
    n_approx: int = 0
    n_severe: int = 0
    mismatches: list[dict] = field(default_factory=list)

    def add(self, row: dict) -> None:
        self.n += 1
        if row["status"] == "match":
            self.n_match += 1
            return
        if row["status"] == "unexercised":
            self.n_unexercised += 1
            return
        self.n_mismatch += 1
        if row.get("reason") == "ours_unsupported":
            self.n_unsupported += 1
        elif row.get("reason") == "order_random":
            self.n_approx += 1
        elif row.get("reason") in ("damage_differs", "accuracy_differs") and (
            row.get("abs_diff", 999) <= 1 or row.get("rel_err", 1.0) <= APPROX_BOUND
        ):
            self.n_approx += 1
        else:
            self.n_severe += 1
        if len(self.mismatches) < 60 or (row.get("reason") != "damage_differs" and len(self.mismatches) < 200):
            self.mismatches.append(row)

    @property
    def status(self) -> str:
        exercised = self.n - self.n_unexercised
        if exercised == 0:
            return "NOT_EXERCISED"
        if not self.n_mismatch:
            return "EXACT"
        if self.n_severe:
            return "WRONG"
        if self.n_unsupported:
            return "UNSUPPORTED-SAFE"
        return "APPROXIMATE"


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def aggregate(
    cases: list[Case], probes: list[dict], compare: Callable[[Case, dict], dict]
) -> dict[tuple[str, str, str], EntryResult]:
    table: dict[tuple[str, str, str], EntryResult] = {}
    for case, probe in zip(cases, probes, strict=True):
        row = compare(case, probe)
        key = (case.kind, case.entry_kind, case.entry)
        if key not in table:
            table[key] = EntryResult(case.kind, case.entry_kind, case.entry)
        table[key].add(row)
    return table


def usage_coverage(
    results: Iterable[EntryResult],
    universe: dict[str, list[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Counts and usage-weighted share per status, per (category, entry_kind)."""

    weights = usage_weights()
    grouped: dict[tuple[str, str], dict[str, EntryResult]] = defaultdict(dict)
    for result in results:
        grouped[(result.category, result.entry_kind)][result.entry] = result
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for (category, entry_kind), by_entry in sorted(grouped.items()):
        counts: dict[str, int] = defaultdict(int)
        weighted: dict[str, float] = defaultdict(float)
        total_weight = 0.0
        table = weights.get(entry_kind, {})
        for entry, result in by_entry.items():
            status = result.status
            counts[status] += 1
            weight = table.get(entry, 0.0)
            weighted[status] += weight
            total_weight += weight
        report.setdefault(category, {})[entry_kind] = {
            "entries": len(by_entry),
            "counts": dict(counts),
            "usage_weighted": {
                status: (weight / total_weight if total_weight else 0.0)
                for status, weight in weighted.items()
            },
        }
    return report
