"""Targeted scenario battles: put every legal move / ability / item on the field.

The seeded random battles only reach what random movesets happen to do. These scenarios are
generated from `data/champions` so that EVERY legal move is selected, EVERY legal ability is on
an active Pokemon and EVERY legal item is held by an active Pokemon at least once, and the
audit compares the bot's view with Showdown's after each step exactly as for random battles.

The format's worker does not validate sets (see tests/sim_harness.py), so any move / ability /
item goes on any species. Each scenario is one seeded battle:

* `moves`: two "testers" with four moves under test each, preferring a move they have not used
  yet every turn (so all eight get selected), against bulky foes with a spread of status and
  support moves and a Protect.
* `abilities`: two holders per battle vs. foes with Intimidate-free but varied setups.
* `items`: two holders per battle (a Mega stone goes on the species it belongs to).

`make_jobs` splits scenarios over workers; `run_job` plays them. Which entries ran is reported by
`audit_coverage` (a trigger needs protocol evidence, mere presence is the weaker tier).
"""

from __future__ import annotations

import random
import sys
import zlib
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_battle_parsing as A  # noqa: E402
from vgc.config import REPO_ROOT  # noqa: E402
from vgc.damage import to_id  # noqa: E402
from vgc.rl.agents import make_direct_agent  # noqa: E402

import json  # noqa: E402

DATA = REPO_ROOT / "data" / "champions"
FILLER_SPECIES = [
    "Snorlax", "Blissey", "Dragonite", "Garchomp", "Incineroar", "Gengar", "Hippowdon",
    "Corviknight", "Gyarados", "Metagross", "Tyranitar", "Milotic", "Amoonguss", "Kommo-o",
]  # fmt: skip
SUPPORT_MOVES = [
    "protect", "willowisp", "thunderwave", "toxic", "swordsdance", "tailwind", "helpinghand",
    "knockoff", "fakeout", "sunnyday", "raindance", "trickroom", "reflect", "taunt", "spore",
]  # fmt: skip
SCENARIO_TURNS = 14
MOVES = json.loads((DATA / "moves.json").read_text())
NEEDS_TARGET = {"normal", "any", "adjacentAlly", "adjacentAllyOrSelf", "adjacentFoe"}


def _catalog() -> dict[str, list[str]]:
    moves = json.loads((DATA / "moves.json").read_text())
    items = json.loads((DATA / "items.json").read_text())
    catalog = json.loads((DATA / "mechanics_catalog.json").read_text())
    return {
        "moves": sorted(k for k, v in moves.items() if v.get("isNonstandard") is None),
        "items": sorted(items),
        "abilities": sorted(catalog["legal_entities"]["abilities"]),
    }


def _mon(species: str, item: str, ability: str, moves: list[str]) -> str:
    return f"{species}||{item}|{ability}|{','.join(moves)}|Serious|32,,,,,||||50|"


def _team(members: list[str]) -> str:
    return "]".join(members)


def _fillers(rng: random.Random, n: int, avoid: tuple[str, ...] = ()) -> list[str]:
    out = []
    pool = [sp for sp in FILLER_SPECIES if sp not in avoid]
    for species in rng.sample(pool, n):
        out.append(_mon(species, "leftovers", "", rng.sample(SUPPORT_MOVES, 4)))
    return out


def _passive(rng: random.Random) -> list[str]:
    """Bulky foes that mostly stall, so the testers survive long enough to use every move."""
    return [
        _mon(sp, "leftovers", "", ["protect", "tailwind", "sunnyday", "raindance"])
        for sp in rng.sample(FILLER_SPECIES, 6)
    ]


class ScriptedAgent:
    """Chooses legal joint orders, preferring moves from `wanted` that were not used yet."""

    def __init__(self, base: Any, wanted: set[str], rng: random.Random) -> None:
        self.base, self.wanted, self.rng = base, set(wanted), rng

    def observe(self, tag: str, lines: Any) -> None:
        self.base.observe(tag, lines)

    def choose(self, battle: Any) -> str:
        from vgc.actions import enumerate_joint_orders
        from vgc.rl.env import choice_string

        if battle.teampreview:
            return "team 1234"
        try:
            orders = enumerate_joint_orders(battle)
        except Exception:
            orders = []
        if not orders:
            return self.base.choose(battle)

        def score(order: Any) -> float:
            parts = order.message.removeprefix("/choose ").split(", ")
            value = self.rng.random() * 0.5
            for part in parts:
                bits = part.split()
                if len(bits) > 1 and bits[0] == "move":
                    need = MOVES.get(to_id(bits[1]), {}).get("target")
                    if need in NEEDS_TARGET and not any(b.lstrip("-").isdigit() for b in bits[2:]):
                        value -= 100  # the simulator rejects a targeted move sent without one
                if len(bits) > 1 and bits[0] == "move" and to_id(bits[1]) in self.wanted:
                    value += 10
            return value

        best = max(orders, key=score)
        for part in best.message.removeprefix("/choose ").split(", "):
            bits = part.split()
            if len(bits) > 1 and bits[0] == "move":
                self.wanted.discard(to_id(bits[1]))
        return choice_string(best)


def build_scenarios(name_filter: str = "") -> list[dict[str, Any]]:
    cat = _catalog()
    items_data = json.loads((DATA / "items.json").read_text())
    scenarios: list[dict[str, Any]] = []
    rng = random.Random(20261010)
    # moves: 8 per battle
    moves = cat["moves"]
    for i in range(0, len(moves), 8):
        chunk = moves[i : i + 8]
        a, b = chunk[:4], chunk[4:]
        members = [
            _mon("Snorlax", "leftovers", "", a or ["protect"]),
            _mon("Blissey", "leftovers", "", b or ["protect"]),
        ] + _fillers(rng, 4, ("Snorlax", "Blissey"))
        scenarios.append(
            {"name": f"moves_{i // 8:03d}", "p1": _team(members), "wanted": chunk,
             "p2": _team(_passive(rng))}  # fmt: skip
        )
    # abilities: 2 per battle
    abilities = cat["abilities"]
    for i in range(0, len(abilities), 2):
        chunk = abilities[i : i + 2]
        members = [
            _mon(FILLER_SPECIES[j], "leftovers", ab, rng.sample(SUPPORT_MOVES, 4))
            for j, ab in enumerate(chunk)
        ] + _fillers(rng, 4, tuple(FILLER_SPECIES[:2]))
        scenarios.append(
            {"name": f"abilities_{i // 2:03d}", "p1": _team(members), "wanted": [],
             "p2": _team(_fillers(rng, 6))}  # fmt: skip
        )
    # items: 2 per battle (mega stones on their own species)
    items = cat["items"]
    for i in range(0, len(items), 2):
        chunk = items[i : i + 2]
        members = []
        for j, item in enumerate(chunk):
            users = items_data[item].get("itemUser") or []
            species = users[0] if users else FILLER_SPECIES[j]
            if any(m.startswith(species + "||") for m in members):
                taken = {m.split("||")[0] for m in members}
                species = next(sp for sp in FILLER_SPECIES[2:] if sp not in taken)
            members.append(_mon(species, item, "", rng.sample(SUPPORT_MOVES, 4)))
        held = {m.split("||")[0] for m in members}
        members += _fillers(rng, 4, tuple(held))
        scenarios.append(
            {"name": f"items_{i // 2:03d}", "p1": _team(members), "wanted": [],
             "p2": _team(_fillers(rng, 6))}  # fmt: skip
        )
    return [s for s in scenarios if name_filter in s["name"]]


def make_jobs(workers: int, *, disable: list[str], name_filter: str = "") -> list[dict[str, Any]]:
    scenarios = build_scenarios(name_filter)
    jobs = [{"scenarios": scenarios[w::workers], "disable": disable} for w in range(workers)]
    return [job for job in jobs if job["scenarios"]]


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    from vgc import poke_env_compat
    from vgc.rl.env import SimWorker

    poke_env_compat.set_disabled_fixes(job.get("disable") or [])
    acc = A._new_acc()
    with SimWorker() as worker:
        for index, sc in enumerate(job["scenarios"]):
            rng = random.Random(zlib.crc32(sc["name"].encode()))
            random.seed(index)
            seed = [(index * 7 + k + 1) for k in range(4)]
            teams = {"p1": sc["p1"], "p2": sc["p2"]}
            agents = {
                "p1": ScriptedAgent(make_direct_agent("random", sc["p1"]), set(sc["wanted"]), rng),
                "p2": ScriptedAgent(make_direct_agent("random", sc["p2"]), set(), rng),
            }
            try:
                A.audit_battle(
                    worker, f"scn_{sc['name']}", {"p1": "scripted", "p2": "scripted"}, teams,
                    seed, acc, agents=agents, max_steps=SCENARIO_TURNS * 3,
                )  # fmt: skip
            except Exception as exc:
                acc["errors"].append(
                    {"index": sc["name"], "error": f"{type(exc).__name__}: {exc}",
                     "trace": traceback.format_exc()[-1500:]}  # fmt: skip
                )
                try:
                    worker.request({"cmd": "close", "id": f"scn_{sc['name']}"})
                except Exception:
                    pass
    return acc
