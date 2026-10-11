"""Scenario families and comparators for `vgc.calc_sweep` (abilities, items, order, accuracy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from vgc.accuracy import hit_probability
from vgc.calc_sweep import (
    SP_ATTACKER,
    STATUSES,
    TERRAINS,
    WEATHERS,
    Case,
    Field,
    Mon,
    _move,
    base_mons,
    damage_case,
    defender_for,
    dummy_mon,
    legal_moves,
    state_not_applied,
)
from vgc.damage import PokemonState, to_id
from vgc.data import load_items, load_moves, load_species
from vgc.evaluator import field_effective_speed, mega_evolved_state, resolves_before
from vgc.stats import calculate_stats

# --- ability / item damage batteries ---------------------------------------------------------------

# One representative move per mechanic an ability or item might touch: contact, punch/bite/
# slicing/sound/pulse/bullet/wind flags, secondary effects, recoil, multi-hit, spread, every
# damaging type, variable-power moves, fixed damage, always-crit, priority, status-boosted power.
BATTERY: tuple[str, ...] = (
    "bodyslam", "earthquake", "rockslide", "closecombat", "machpunch", "icepunch", "crunch",
    "leafblade", "poisonjab", "ironhead", "knockoff", "facade", "acrobatics", "bravebird",
    "doubleedge", "bulletseed", "dragonclaw", "suckerpunch", "bodypress", "foulplay",
    "stompingtantrum", "psychocut", "flareblitz", "playrough", "tripleaxel", "stormthrow",
    "hypervoice", "dazzlinggleam", "flamethrower", "surf", "thunderbolt", "energyball", "icebeam",
    "shadowball", "darkpulse", "psychic", "moonblast", "flashcannon", "bugbuzz", "sludgebomb",
    "hex", "venoshock", "heatwave", "muddywater", "airslash", "weatherball", "terrainpulse",
    "eruption", "electroball", "gyroball", "grassknot", "lowkick", "seismictoss", "superfang",
    "scald", "frostbreath", "expandingforce", "aurasphere", "boomburst", "quickattack",
)
QUICK_BATTERY: tuple[str, ...] = (
    "bodyslam", "earthquake", "rockslide", "closecombat", "machpunch", "crunch", "leafblade",
    "ironhead", "knockoff", "facade", "bravebird", "suckerpunch", "hypervoice", "dazzlinggleam",
    "flamethrower", "surf", "thunderbolt", "energyball", "icebeam", "shadowball", "darkpulse",
    "psychic", "hex", "heatwave", "weatherball", "scald", "quickattack",
)


PARTNER_CONDS = (
    "base", "weather_sun", "weather_rain", "weather_sand", "weather_snow",
    "terrain_electric", "terrain_grassy", "terrain_psychic", "terrain_misty",
)


@dataclass
class Cond:
    name: str
    holder: dict = field(default_factory=dict)  # Mon attribute overrides on the holder
    other: dict = field(default_factory=dict)  # ... on the opposing mon of the pair
    fld: Field = field(default_factory=Field)


def _boost(stat_kind: str, stage: int, category: str) -> dict:
    if stat_kind == "off":
        return {"boosts": {"atk" if category == "Physical" else "spa": stage}}
    return {"boosts": {"def" if category == "Physical" else "spd": stage}}


def condition_list(category: str, profile: str) -> list[Cond]:
    """Holder/other conditions for one battery move. ``category`` picks which stats to boost."""

    conds = [Cond("base")]
    statuses = STATUSES if profile == "full" else ("brn", "par", "psn", "slp")
    for status in statuses:
        conds.append(Cond(f"H_{status}", holder={"status": status}))
    conds.append(Cond("H_hp_0.3", holder={"hp_frac": 0.3}))
    conds.append(Cond("H_hp_0.6", holder={"hp_frac": 0.6}))
    conds.append(Cond("H_off+1", holder=_boost("off", 1, category)))
    conds.append(Cond("H_def+1", holder=_boost("def", 1, category)))
    conds.append(Cond("H_noitem", holder={"no_item": True}))
    conds.append(Cond("O_brn", other={"status": "brn"}))
    conds.append(Cond("O_psn", other={"status": "psn"}))
    conds.append(Cond("O_hp_0.5", other={"hp_frac": 0.5}))
    conds.append(Cond("O_def+1", other=_boost("def", 1, category)))
    conds.append(Cond("O_off+1", other=_boost("off", 1, category)))
    conds.append(Cond("O_item", other={"item": "leftovers"}))
    for weather in WEATHERS:
        conds.append(Cond(f"weather_{weather}", fld=Field(weather=weather)))
    for terrain in TERRAINS:
        conds.append(Cond(f"terrain_{terrain}", fld=Field(terrain=terrain)))
    if profile == "full":
        conds.append(Cond("screens", fld=Field(p2side=("reflect", "lightscreen"))))
        conds.append(Cond("H_off-1", holder=_boost("off", -1, category)))
        conds.append(Cond("O_def-1", other=_boost("def", -1, category)))
    return conds


def item_species(item_id: str) -> str | None:
    """A species the item is built for (Light Ball: Pikachu), when it has a restriction."""

    users = (load_items().get(item_id) or {}).get("itemUser") or []
    for user in users:
        sid = to_id(user)
        if sid in load_species():
            return sid
    return None


def _copy(value: Any) -> Any:
    return dict(value) if isinstance(value, dict) else value


# Moves whose every condition is explored for an ability/item (the others run the base
# condition only): both categories, contact and spread, and the types weather/terrain touch.
COND_MOVES = (
    "bodyslam", "earthquake", "icepunch", "flamethrower", "surf", "thunderbolt", "energyball",
    "shadowball", "hex", "facade", "hypervoice", "quickattack", "knockoff", "dazzlinggleam",
    "bravebird", "psychic",
)
QUICK_COND_MOVES = ("bodyslam", "earthquake", "flamethrower", "hypervoice", "hex", "quickattack")


def holder_cases(entry_kind: str, entry: str, *, profile: str = "full") -> list[Case]:
    """Damage cases for one ability or item in each role: attacker, defender, the defender's
    partner (Dazzling, redirection...) and the attacker's partner (Battery, Power Spot...).

    Every case carries a ``control`` twin without the entry, so the entry is judged on its own
    effect (see `vgc.calc_sweep.compare_damage`)."""

    battery = BATTERY if profile == "full" else QUICK_BATTERY
    cond_moves = set(COND_MOVES if profile == "full" else QUICK_COND_MOVES)
    legal = set(legal_moves())
    cases: list[Case] = []
    forced_species = item_species(entry) if entry_kind == "item" else None
    for move_id in battery:
        if move_id not in legal:
            continue
        md = _move(move_id)
        conds = condition_list(md["category"], profile) if move_id in cond_moves else [Cond("base")]
        for role in ("att", "def", "pfoe", "pally"):
            for cond in conds:
                partner = role in ("pfoe", "pally")
                if partner and (cond.name not in PARTNER_CONDS or cond.holder or cond.other):
                    continue
                pair = []
                for with_entry in (True, False):
                    mons = base_mons(move_id)
                    holder_slot = {"att": "p1a", "def": "p2a", "pfoe": "p2b", "pally": "p1b"}[role]
                    other_slot = "p2a" if role == "att" else "p1a"
                    if with_entry:
                        setattr(mons[holder_slot], entry_kind, entry)
                    elif entry_kind == "item":
                        # An inert item keeps "target holds an item" true (Knock Off, Poltergeist,
                        # Acrobatics) so only the item under test differs between the twins.
                        setattr(mons[holder_slot], "item", "sitrusberry" if entry == "leftovers" else "leftovers")
                    if forced_species and holder_slot in ("p1a", "p2a"):
                        mons[holder_slot].species = forced_species
                    for attr, value in cond.holder.items():
                        setattr(mons[holder_slot], attr, _copy(value))
                    for attr, value in cond.other.items():
                        setattr(mons[other_slot], attr, _copy(value))
                    variant = f"{role}|{move_id}|{cond.name}"
                    pair.append(damage_case(entry_kind, entry, variant, move_id, mons, cond.fld))
                case, control = pair
                case.control = control
                cases.append(case)
    return cases


def mega_cases(stone: str) -> list[Case]:
    """The stone's holder Mega Evolves and attacks in the same turn (engine order: Mega first)."""

    item = load_items()[stone]
    base = None
    for user in item.get("itemUser") or []:
        sid = to_id(user)
        if sid in load_species() and not load_species()[sid].get("isMega"):
            base = sid
            break
    if base is None:
        return []
    cases = []
    for move_id in ("bodyslam", "hypervoice", "earthquake", "flamethrower"):
        for cond_name, fld in (("base", Field()), ("sun", Field(weather="sun")), ("rain", Field(weather="rain"))):
            mons = base_mons(move_id, attacker_species=base)
            mons["p1a"].item = stone
            case = damage_case("item", stone, f"mega|{move_id}|{cond_name}", move_id, mons, fld, mega=True)
            case.meta["mega_stone"] = True
            cases.append(case)
    return cases


# --- speed order & priority ---------------------------------------------------------------------------

SPEED_POOL = ("shuckle", "snorlax", "incineroar", "garchomp", "dragapult", "regieleki")
A_SPECIES = "garchomp"
A_SPEED_SP = 16


@lru_cache(maxsize=1)
def speed_options() -> dict[int, tuple[str, int, str]]:
    options: dict[int, tuple[str, int, str]] = {}
    for sid in SPEED_POOL:
        for nature in ("serious", "jolly", "brave"):
            for points in range(33):
                spe = calculate_stats(sid, sp={"spe": points}, nature=nature)["spe"]
                options.setdefault(spe, (sid, points, nature))
    return options


def speed_partner(target: int) -> tuple[str, int, str, int]:
    options = speed_options()
    best = min(options, key=lambda stat: (abs(stat - target), stat))
    sid, points, nature = options[best]
    return sid, points, nature, best


def base_speed(species: str = A_SPECIES) -> int:
    return calculate_stats(species, sp={"spe": A_SPEED_SP}, nature="serious")["spe"]


@dataclass
class SpeedCond:
    name: str
    a: dict = field(default_factory=dict)
    b: dict = field(default_factory=dict)
    fld: Field = field(default_factory=Field)


def speed_conditions() -> list[SpeedCond]:
    conds = [SpeedCond("none")]
    for weather in WEATHERS:
        conds.append(SpeedCond(f"weather_{weather}", fld=Field(weather=weather)))
    for terrain in TERRAINS:
        conds.append(SpeedCond(f"terrain_{terrain}", fld=Field(terrain=terrain)))
    conds.append(SpeedCond("tailwind_own", fld=Field(p1side=("tailwind",))))
    conds.append(SpeedCond("tailwind_foe", fld=Field(p2side=("tailwind",))))
    conds.append(SpeedCond("trick_room", fld=Field(trick_room=True)))
    conds.append(SpeedCond("A_par", a={"status": "par"}))
    conds.append(SpeedCond("A_brn", a={"status": "brn"}))
    conds.append(SpeedCond("A_noitem", a={"no_item": True}))
    conds.append(SpeedCond("A_slp", a={"status": "slp"}))
    conds.append(SpeedCond("A_hp_0.3", a={"hp_frac": 0.3}))
    return conds


# stat ratios B/A at which a multiplier on A's side should flip the order, plus off-by-one probes
SPEED_RATIOS = ((0.5, "0.5"), (2 / 3, "0.667"), (1.0, "1"), (1.5, "1.5"), (2.0, "2"))
RANDOM_ORDER_ENTRIES = {"quickclaw", "quickdraw"}
TIE_SEEDS = tuple((n, n + 1, n + 2, n + 3) for n in range(1, 41, 4))


def order_case(
    entry_kind: str,
    entry: str,
    variant: str,
    a: Mon,
    b: Mon,
    a_move: str,
    b_move: str,
    fld: Field,
    *,
    seeds: tuple = ((1, 2, 3, 4),),
) -> Case:
    for mon, move in ((a, a_move), (b, b_move)):
        if move not in mon.moves:
            mon.moves = (move, *mon.moves)
    slow = Mon("shuckle", sp={"hp": 32, "def": 17, "spd": 17}, nature="brave")
    slow2 = Mon("shuckle", sp={"hp": 32, "def": 17, "spd": 17}, nature="brave")
    mons = {"p1a": a, "p1b": slow, "p2a": b, "p2b": slow2}
    targets = {}
    if a_move != "splash" and _move(a_move)["target"] in ("normal", "any", "adjacentFoe"):
        targets["p1a"] = 1
    if b_move != "splash" and _move(b_move)["target"] in ("normal", "any", "adjacentFoe"):
        targets["p2a"] = 1
    return Case(
        "order",
        entry_kind,
        entry,
        variant,
        mons,
        {"p1a": a_move, "p1b": "splash", "p2a": b_move, "p2b": "splash"},
        fld,
        targets=targets,
        seeds=seeds,
        turns=1,
    )


def speed_cases(entry_kind: str, entry: str) -> list[Case]:
    """Speed-ratio grid for an ability/item on A, under every field/status condition."""

    cases: list[Case] = []
    forced = item_species(entry) if entry_kind == "item" else None
    base = base_speed(forced or A_SPECIES)
    for cond in speed_conditions():
        for ratio, ratio_name in SPEED_RATIOS:
            for delta in (-1, 0, 1):
                a = Mon(forced or A_SPECIES, sp={"spe": A_SPEED_SP}, nature="serious")
                if forced:
                    a.sp = {"spe": A_SPEED_SP}
                setattr(a, entry_kind, entry)
                for attr, value in cond.a.items():
                    setattr(a, attr, _copy(value))
                target = int(base * ratio) + delta
                sid, points, nature, _ = speed_partner(target)
                b = Mon(sid, sp={"spe": points}, nature=nature)
                for attr, value in cond.b.items():
                    setattr(b, attr, _copy(value))
                seeds = TIE_SEEDS if (delta == 0 or entry in RANDOM_ORDER_ENTRIES) else ((1, 2, 3, 4),)
                variant = f"speed|{cond.name}|r{ratio_name}|d{delta:+d}"
                cases.append(order_case(entry_kind, entry, variant, a, b, "splash", "splash", cond.fld, seeds=seeds))
    return cases


PRIORITY_BATTERY = (
    "thunderwave", "swordsdance", "bravebird", "recover", "drainpunch", "bodyslam", "tailwind",
    "protect", "gigadrain", "aerialace", "airslash", "roost",
)


def priority_cases(entry_kind: str, entry: str) -> list[Case]:
    """A slower/faster holder uses a priority-sensitive move against a splash: who goes first."""

    cases: list[Case] = []
    legal = set(legal_moves())
    forced = item_species(entry) if entry_kind == "item" else None
    base = base_speed(forced or A_SPECIES)
    for move_id in PRIORITY_BATTERY:
        if move_id not in legal:
            continue
        for relation, ratio in (("A_slow", 2.0), ("A_fast", 0.5)):
            for cond_name, a_extra, fld in (
                ("none", {}, Field()),
                ("hp_0.9", {"hp_frac": 0.9}, Field()),
                ("grassy", {}, Field(terrain="grassy")),
                ("trick_room", {}, Field(trick_room=True)),
            ):
                a = Mon(forced or A_SPECIES, sp={"spe": A_SPEED_SP}, nature="serious")
                setattr(a, entry_kind, entry)
                for attr, value in a_extra.items():
                    setattr(a, attr, value)
                sid, points, nature, _ = speed_partner(int(base * ratio))
                b = Mon(sid, sp={"spe": points}, nature=nature)
                variant = f"priority|{move_id}|{relation}|{cond_name}"
                seeds = TIE_SEEDS if entry in RANDOM_ORDER_ENTRIES else ((1, 2, 3, 4),)
                cases.append(order_case(entry_kind, entry, variant, a, b, move_id, "splash", fld, seeds=seeds))
    return cases


def move_priority_cases(move_id: str) -> list[Case]:
    """Every legal move: user slower/faster than a splash-user, plain / Trick Room / terrain."""

    cases: list[Case] = []
    base = base_speed()
    for relation, ratio in (("A_slow", 2.0), ("A_fast", 0.5)):
        for cond_name, fld in (
            ("none", Field()),
            ("trick_room", Field(trick_room=True)),
            ("grassy", Field(terrain="grassy")),
            ("psychic", Field(terrain="psychic")),
        ):
            a = Mon(A_SPECIES, sp={"spe": A_SPEED_SP}, nature="serious")
            sid, points, nature, _ = speed_partner(int(base * ratio))
            b = Mon(sid, sp={"spe": points}, nature=nature)
            variant = f"priority|{relation}|{cond_name}"
            cases.append(order_case("move", move_id, variant, a, b, move_id, "splash", fld))
    return cases


def engine_pair(results: list[dict], first: str = "p1a", second: str = "p2a") -> str | None:
    first_wins = second_wins = 0
    for result in results:
        order = result.get("order") or []
        if first in order and second in order:
            if order.index(first) < order.index(second):
                first_wins += 1
            else:
                second_wins += 1
    if first_wins and second_wins:
        return "tie"
    if first_wins:
        return "A"
    if second_wins:
        return "B"
    return None


def _priority_of(case: Case, slot: str) -> int:
    from vgc.models import PolicyConfig

    mon = case.mons[slot]
    md = _move(case.moves[slot])
    try:
        from vgc.priority_rules import action_priority
    except ImportError:  # the pre-sweep calculators (baseline measurement)
        return int(md.get("priority", 0))
    effective = getattr(PolicyConfig(), "effective_move_priority", False)
    state = mon.state()
    at_full = state.hp_or_max() >= state.max_hp()
    return action_priority(md, state.ability, at_full, case.fld.terrain, effective=effective)


def ours_order(case: Case) -> str:
    a_state = case.mons["p1a"].state()
    b_state = case.mons["p2a"].state()
    a_tail = "tailwind" in case.fld.p1side
    b_tail = "tailwind" in case.fld.p2side
    a_speed = field_effective_speed(a_state, weather=case.fld.weather, tailwind=a_tail)
    b_speed = field_effective_speed(b_state, weather=case.fld.weather, tailwind=b_tail)
    pa = _priority_of(case, "p1a")
    pb = _priority_of(case, "p2a")
    a_first = resolves_before(pa, a_speed, pb, b_speed, case.fld.trick_room)
    b_first = resolves_before(pb, b_speed, pa, a_speed, case.fld.trick_room)
    if a_first and b_first:
        return "tie"
    return "A" if a_first else "B"


def compare_order(case: Case, probe: dict) -> dict[str, Any]:
    results = probe.get("results") or []
    out: dict[str, Any] = {"cid": case.cid, "variant": case.variant}
    if not results or any(r.get("err") for r in results):
        out.update(status="unexercised", reason="probe_error")
        return out
    not_applied = state_not_applied(case, probe)
    if not_applied:
        out.update(status="unexercised", reason=not_applied)
        return out
    engine = engine_pair(results)
    if engine is None:
        out.update(status="unexercised", reason="no_order")
        return out
    ours = ours_order(case)
    if ours == engine:
        out["status"] = "match"
        return out
    if engine == "tie" and case.entry in RANDOM_ORDER_ENTRIES:
        # The engine's order is a die roll (Quick Claw / Quick Draw); a deterministic guess is
        # the documented approximation.
        out.update(status="mismatch", reason="order_random", ours=ours, engine=engine, rel_err=0.2, abs_diff=0)
        return out
    out.update(status="mismatch", reason="order_differs", ours=ours, engine=engine, rel_err=1.0)
    return out


# --- accuracy ---------------------------------------------------------------------------------------

ACCURACY_BATTERY = (
    "thunder", "hurricane", "blizzard", "focusblast", "stoneedge", "thunderwave", "earthquake",
    "swordsdance", "dazzlinggleam", "willowisp", "hypnosis", "rockslide",
)


def accuracy_case(
    entry_kind: str,
    entry: str,
    variant: str,
    move_id: str,
    a: Mon,
    b: Mon,
    fld: Field,
    *,
    target_moves_first: bool | None = None,
) -> Case:
    if move_id not in a.moves:
        a.moves = (move_id, *a.moves)
    mons = {"p1a": a, "p1b": dummy_mon(), "p2a": b, "p2b": dummy_mon()}
    md = _move(move_id)
    targets = {"p1a": 1} if md["target"] in ("normal", "any", "adjacentFoe") else {}
    case = Case(
        "accuracy",
        entry_kind,
        entry,
        variant,
        mons,
        {"p1a": move_id, "p1b": "splash", "p2a": "splash", "p2b": "splash"},
        fld,
        targets=targets,
        turns=1,
    )
    case.meta["target_moves_first"] = target_moves_first
    return case


def _acc_mons(move_id: str) -> tuple[Mon, Mon]:
    md = _move(move_id)
    a = Mon("garchomp", moves=(move_id, "splash"), sp=dict(SP_ATTACKER), nature="adamant")
    # A defender the move's type can hit, so the type immunity (checked before accuracy) cannot
    # hide the accuracy roll.
    b = Mon(
        defender_for(md["type"], 1.0) or "snorlax",
        sp={"hp": 32, "def": 17, "spd": 17},
        nature="relaxed",
    )
    return a, b


def move_accuracy_cases(move_id: str) -> list[Case]:
    cases = []

    def add(variant: str, a_extra: dict | None = None, b_extra: dict | None = None, fld: Field | None = None) -> None:
        a, b = _acc_mons(move_id)
        for attr, value in (a_extra or {}).items():
            setattr(a, attr, _copy(value))
        for attr, value in (b_extra or {}).items():
            setattr(b, attr, _copy(value))
        cases.append(accuracy_case("move", move_id, variant, move_id, a, b, fld or Field(), target_moves_first=False))

    add("base")
    for stage in (1, 3, 6, -1, -3, -6):
        add(f"acc{stage:+d}", a_extra={"boosts": {"accuracy": stage}})
        add(f"eva{stage:+d}", b_extra={"boosts": {"evasion": stage}})
    add("acc+2_eva+1", a_extra={"boosts": {"accuracy": 2}}, b_extra={"boosts": {"evasion": 1}})
    add("acc-1_eva-2", a_extra={"boosts": {"accuracy": -1}}, b_extra={"boosts": {"evasion": -2}})
    for weather in WEATHERS:
        add(f"weather_{weather}", fld=Field(weather=weather))
    return cases


def holder_accuracy_cases(entry_kind: str, entry: str) -> list[Case]:
    cases = []
    legal = set(legal_moves())
    forced = item_species(entry) if entry_kind == "item" else None
    for move_id in ACCURACY_BATTERY:
        if move_id not in legal:
            continue
        for role in ("att", "def"):
            for cond in ("base", "acc+1", "eva+1", "acc-2", "eva-2", "sun", "rain", "sand", "snow", "foe_first"):
                a, b = _acc_mons(move_id)
                holder = a if role == "att" else b
                setattr(holder, entry_kind, entry)
                if forced:
                    holder.species = forced
                fld = Field()
                first = False
                if cond == "acc+1":
                    a.boosts = {"accuracy": 1}
                elif cond == "acc-2":
                    a.boosts = {"accuracy": -2}
                elif cond == "eva+1":
                    b.boosts = {"evasion": 1}
                elif cond == "eva-2":
                    b.boosts = {"evasion": -2}
                elif cond in ("sun", "rain", "sand", "snow"):
                    fld = Field(weather=cond)
                elif cond == "foe_first":
                    b.boosts = {**b.boosts, "spe": 6}
                    first = True
                cases.append(
                    accuracy_case(entry_kind, entry, f"{role}|{move_id}|{cond}", move_id, a, b, fld, target_moves_first=first)
                )
    return cases


def ours_accuracy(case: Case) -> float:
    move_id = case.moves["p1a"]
    md = load_moves()[move_id]
    attacker = case.mons["p1a"].state()
    defender = case.mons["p2a"].state()
    return hit_probability(
        md, attacker, defender, case.fld.weather, target_moves_first=case.meta.get("target_moves_first")
    )


def compare_accuracy(case: Case, probe: dict) -> dict[str, Any]:
    results = probe.get("results") or []
    out: dict[str, Any] = {"cid": case.cid, "variant": case.variant}
    if not results or results[0].get("err"):
        out.update(status="unexercised", reason="probe_error")
        return out
    not_applied = state_not_applied(case, probe)
    if not_applied:
        out.update(status="unexercised", reason=not_applied)
        return out
    log = results[0].get("log") or []
    if not any(line.startswith("|move|p1a:") for line in log):
        out.update(status="unexercised", reason="not_used")
        return out
    numbers = [entry["n"] for entry in (results[0].get("accuracy") or []) if entry["t"] == "p2a"]
    if not numbers and any(line.startswith(("|-immune", "|-fail", "|-block", "|-activate", "|-ability", "|-miss")) for line in log):
        out.update(status="unexercised", reason="blocked_before_accuracy")
        return out
    engine = 1.0 if not numbers else min(1.0, max(0.0, numbers[0] / 100.0))
    ours = ours_accuracy(case)
    if abs(ours - engine) < 1e-9:
        out["status"] = "match"
        return out
    out.update(
        status="mismatch",
        reason="accuracy_differs",
        ours=ours,
        engine=engine,
        rel_err=abs(ours - engine) / max(engine, 0.01),
    )
    return out


# --- raw stats ---------------------------------------------------------------------------------------


def stat_cases(species_id: str) -> list[Case]:
    """One case per species: four mons of that species with four spreads/natures."""

    specs = (
        ({"hp": 2, "atk": 32, "spe": 32}, "adamant"),
        ({"hp": 32, "spa": 32, "spe": 2}, "timid"),
        ({"hp": 32, "def": 17, "spd": 17}, "bold"),
        ({}, "serious"),
    )
    mons = {
        slot: Mon(species_id, sp=sp, nature=nature)
        for slot, (sp, nature) in zip(("p1a", "p1b", "p2a", "p2b"), specs, strict=True)
    }
    return [
        Case(
            "stats",
            "species",
            species_id,
            "stats",
            mons,
            {slot: "splash" for slot in mons},
            turns=0,
        )
    ]


def compare_stats(case: Case, probe: dict) -> dict[str, Any]:
    results = probe.get("results") or []
    out: dict[str, Any] = {"cid": case.cid, "variant": case.variant}
    if not results or results[0].get("err"):
        out.update(status="unexercised", reason="probe_error")
        return out
    engine = results[0]["stats"]
    for slot, mon in case.mons.items():
        ours = mon.state().stats()
        theirs = engine[slot]
        for stat in ("hp", "atk", "def", "spa", "spd", "spe"):
            if ours[stat] != theirs[stat]:
                out.update(
                    status="mismatch",
                    reason="accuracy_differs",
                    detail=f"{slot} {stat} ours {ours[stat]} engine {theirs[stat]}",
                    rel_err=1.0,
                )
                return out
    out["status"] = "match"
    return out


# Ours-side Mega handling for the Mega cases: the evaluator evolves the base state first.
def ours_mega_state(state: PokemonState) -> PokemonState:
    return mega_evolved_state(state)
