"""Shared helpers for Showdown-executed mechanics-family tests.

Custom Game does not validate legality, so species/move/ability combinations isolate
one mechanic. Helpers here are imported by the per-batch test modules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict
from itertools import count

import pytest

from tests.sim_harness import pokeset
from vgc.mechanics_state import snapshot_battle
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.mechanics_oracle import ExactMechanicsBranch, evaluate_exact_branches

DOUBLES = "gen9championsdoublescustomgame"
_IDS = count(1)

# Status/self/side/spread moves that Showdown rejects if a slot target is appended.
_NO_TARGET_MOVES = frozenset(
    {
        "protect",
        "splash",
        "trickroom",
        "minimize",
        "followme",
        "ragepowder",
        "wideguard",
        "quickguard",
        "endure",
        "sunnyday",
        "imprison",
        "heatwave",
        "earthquake",
        "outrage",
        "recover",
        "swordsdance",
        "bellydrum",
    }
)


def _seeds(n: int, *, origin: int) -> list[list[int]]:
    return [[origin + i, origin + 1000 + i, origin + 2000 + i, origin + 3000 + i] for i in range(n)]


def _partner() -> dict:
    return pokeset("Blissey", ["protect", "splash"], nature="Calm", sp={"hp": 32, "spd": 32, "def": 2})


def _foe_partner() -> dict:
    return pokeset("Klefki", ["protect", "splash"], nature="Bold", sp={"hp": 32, "def": 32, "spd": 2})


@pytest.fixture(scope="module")
def worker():
    if not (DEFAULT_SHOWDOWN_REPO / "dist" / "sim" / "index.js").exists():
        pytest.skip(f"no built showdown sim at {DEFAULT_SHOWDOWN_REPO}")
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as running:
        yield running


@contextmanager
def _root(
    worker: SimWorker,
    p1: list[dict],
    p2: list[dict],
    *,
    seed: Sequence[int] = (1, 2, 3, 4),
    p1_order: str = "12",
    p2_order: str = "12",
) -> Iterator[DirectBattle]:
    battle_id = f"fam-{next(_IDS)}"
    battle = DirectBattle.start(
        worker,
        battle_id,
        p1,
        p2,
        battle_format=DOUBLES,
        seed=list(seed),
    )
    try:
        battle.step({"p1": f"team {p1_order}", "p2": f"team {p2_order}"})
        yield battle
    finally:
        battle.close()


def _choice(first: str, second: str = "protect") -> str:
    def _part(move: str) -> str:
        if move.startswith("move ") or move.startswith("switch "):
            return move
        if move in _NO_TARGET_MOVES:
            return f"move {move}"
        return f"move {move} 1"

    return f"{_part(first)}, {_part(second)}"


def _targeted(move: str, slot: int) -> str:
    return f"move {move} {slot}"


def _branches(
    root: DirectBattle,
    p1_choice: str,
    p2_choice: str,
    seeds: Sequence[Sequence[int]],
    prefix: str,
) -> list[ExactMechanicsBranch]:
    return evaluate_exact_branches(
        root,
        [{"p1": p1_choice, "p2": p2_choice}],
        future_seeds=seeds,
        branch_prefix=prefix,
    )


def _lines(branch: ExactMechanicsBranch, side: str = "p1") -> tuple[str, ...]:
    return dict(branch.public_lines)[side]


def _has(branch: ExactMechanicsBranch, token: str, side: str = "p1") -> bool:
    return any(token in line for line in _lines(branch, side))


def _move_slots(branch: ExactMechanicsBranch) -> list[str]:
    slots: list[str] = []
    for line in _lines(branch):
        parts = line.split("|")
        if len(parts) > 2 and parts[1] == "move":
            slots.append(parts[2].split(":")[0])
    return slots


def _slot_before(branch: ExactMechanicsBranch, first: str, second: str) -> bool:
    order = _move_slots(branch)
    assert first in order and second in order, order
    return order.index(first) < order.index(second)


def _move_target(branch: ExactMechanicsBranch, actor: str, move_name: str) -> str | None:
    needle = move_name.lower()
    for line in _lines(branch):
        parts = line.split("|")
        if len(parts) < 5 or parts[1] != "move":
            continue
        if not parts[2].startswith(actor):
            continue
        if parts[3].replace(" ", "").lower() != needle:
            continue
        target = parts[4].split(":")[0]
        return target or None
    return None


def _own(branch: ExactMechanicsBranch, side: str, species_id: str):
    state = branch.state_for(side)
    for pokemon in state.our_side.pokemon:
        if pokemon.species_id == species_id:
            return pokemon
    raise AssertionError(f"{species_id} missing on {side}")


def _hp(branch: ExactMechanicsBranch, side: str, species_id: str) -> int:
    hp = _own(branch, side, species_id).current_hp
    assert hp is not None
    return hp


def _max_hp(branch: ExactMechanicsBranch, side: str, species_id: str) -> int:
    hp = _own(branch, side, species_id).max_hp
    assert hp is not None
    return hp


def _status_of(root: DirectBattle, side: str, species_id: str) -> str | None:
    for pokemon in snapshot_battle(root.battles[side]).our_side.pokemon:
        if pokemon.species_id == species_id:
            return pokemon.status
    raise AssertionError(f"{species_id} missing on {side}")


def _volatile_time(root: DirectBattle, side_index: int, volatile: str) -> int | None:
    dumped = root.worker.request({"cmd": "dump", "id": root.battle_id})["state"]
    for pokemon in dumped["sides"][side_index]["pokemon"]:
        volatiles = pokemon.get("volatiles") or {}
        if volatile in volatiles:
            return int(volatiles[volatile]["time"])
    return None


def _available(
    source: ExactMechanicsBranch | DirectBattle, side: str = "p1", slot: int = 0
) -> tuple[str, ...]:
    if isinstance(source, ExactMechanicsBranch):
        state = source.state_for(side)
    else:
        state = snapshot_battle(source.battles[side])
    moves = state.available_moves
    if slot >= len(moves):
        return ()
    return moves[slot]


def _hitcount(branch: ExactMechanicsBranch) -> int | None:
    for line in _lines(branch):
        if "|-hitcount|" not in line:
            continue
        return int(line.split("|")[-1])
    return None


def _root_hp(root: DirectBattle, side: str, species_id: str) -> int:
    for pokemon in snapshot_battle(root.battles[side]).our_side.pokemon:
        if pokemon.species_id == species_id:
            assert pokemon.current_hp is not None
            return pokemon.current_hp
    raise AssertionError(f"{species_id} missing on {side}")


def _digest(branch: ExactMechanicsBranch) -> str:
    def _norm(lines: tuple[str, ...]) -> tuple[str, ...]:
        return tuple("|t:|" if line.startswith("|t:|") else line for line in lines)

    payload = {
        "p1_lines": _norm(_lines(branch, "p1")),
        "p2_lines": _norm(_lines(branch, "p2")),
        "p1": asdict(branch.state_for("p1")),
        "p2": asdict(branch.state_for("p2")),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
