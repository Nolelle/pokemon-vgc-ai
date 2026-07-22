"""Persistent, per-battle strategic memory.

The evaluator's normal ``_Context`` is intentionally a snapshot of one turn.  This
module supplies the missing longitudinal layer: what the preview plan was, how that
plan changed as Pokemon fainted, what the opponent has actually shown, and which lines
we chose on earlier turns.  It stores facts and summaries only; strategic weights stay
in :class:`vgc.models.PolicyConfig` and position scoring stays in ``vgc.search``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from vgc.damage import to_id


def _side(token: object) -> str | None:
    text = str(token or "")
    return text[:2] if len(text) >= 2 and text[0] == "p" and text[1].isdigit() else None


def _display_name(token: object) -> str:
    text = str(token or "")
    return text.split(":", 1)[-1].strip()


@dataclass
class TurnMemory:
    turn: int
    our_active: tuple[str, ...]
    opponent_active: tuple[str, ...]
    our_hp: dict[str, float]
    opponent_hp: dict[str, float]
    weather: tuple[str, ...]
    chosen_order: str | None = None


@dataclass
class BattleMemory:
    """Facts accumulated over one Showdown room, owned by one ``VgcPlayer``."""

    battle_tag: str
    our_role: str | None = None
    current_turn: int = 0
    turns: list[TurnMemory] = field(default_factory=list)
    opponent_moves: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    opponent_targets: Counter[str] = field(default_factory=Counter)
    opponent_switches: Counter[str] = field(default_factory=Counter)
    opponent_protects: Counter[str] = field(default_factory=Counter)
    our_orders: list[tuple[int, str]] = field(default_factory=list)
    initial_win_con: str | None = None
    current_win_con: str | None = None
    initial_primary_threat: str | None = None
    current_primary_threat: str | None = None
    plan_breakers: frozenset[str] = frozenset()
    plan_changes: list[dict[str, object]] = field(default_factory=list)

    @property
    def opponent_role(self) -> str | None:
        if self.our_role == "p1":
            return "p2"
        if self.our_role == "p2":
            return "p1"
        return None

    def observe_protocol(self, split_messages: Iterable[Iterable[object]]) -> None:
        """Consume poke-env's already-split Showdown protocol messages.

        This is deliberately tolerant: malformed or unfamiliar protocol lines are
        ignored because memory must never become a battle-failing dependency.
        """

        for raw in split_messages:
            message = list(raw)
            if len(message) < 2:
                continue
            kind = str(message[1])
            if kind == "turn" and len(message) > 2:
                try:
                    self.current_turn = int(message[2])
                except (TypeError, ValueError):
                    pass
                continue
            if kind == "move" and len(message) > 3:
                actor_role = _side(message[2])
                if self.opponent_role is None or actor_role != self.opponent_role:
                    continue
                actor = to_id(_display_name(message[2])) or "unknown"
                move_id = to_id(message[3]) or "unknown"
                self.opponent_moves[actor][move_id] += 1
                if move_id in {"protect", "detect", "kingsshield", "spikyshield"}:
                    self.opponent_protects[actor] += 1
                if len(message) > 4 and _side(message[4]) == self.our_role:
                    target = to_id(_display_name(message[4])) or "unknown"
                    self.opponent_targets[target] += 1
                continue
            if kind in {"switch", "drag", "replace"} and len(message) > 2:
                if _side(message[2]) == self.opponent_role:
                    species_text = message[3] if len(message) > 3 else _display_name(message[2])
                    species = to_id(str(species_text).split(",", 1)[0]) or "unknown"
                    self.opponent_switches[species] += 1

    def observe_battle(self, battle) -> None:
        """Save one deduplicated live board snapshot for the current decision turn."""

        self.our_role = self.our_role or getattr(battle, "player_role", None)
        turn = int(getattr(battle, "turn", 0) or self.current_turn or 0)
        self.current_turn = max(self.current_turn, turn)

        def active(side: str) -> list:
            try:
                value = getattr(battle, side, None) or []
            except (AttributeError, TypeError, ValueError):
                value = []
            return [mon for mon in value if mon is not None and not getattr(mon, "fainted", False)]

        ours = active("active_pokemon")
        theirs = active("opponent_active_pokemon")

        def names(mons: list) -> tuple[str, ...]:
            return tuple(to_id(getattr(mon, "species", None)) for mon in mons)

        def hp(mons: list) -> dict[str, float]:
            return {
                to_id(getattr(mon, "species", None)): round(
                    100.0 * float(getattr(mon, "current_hp_fraction", 0.0) or 0.0), 2
                )
                for mon in mons
            }

        snapshot = TurnMemory(
            turn=turn,
            our_active=names(ours),
            opponent_active=names(theirs),
            our_hp=hp(ours),
            opponent_hp=hp(theirs),
            weather=tuple(sorted(str(weather) for weather in (getattr(battle, "weather", {}) or {}))),
        )
        if self.turns and self.turns[-1].turn == turn:
            previous_choice = self.turns[-1].chosen_order
            snapshot.chosen_order = previous_choice
            self.turns[-1] = snapshot
        else:
            self.turns.append(snapshot)

    def update_strategy(self, gameplan) -> None:
        """Carry the original plan forward and record only material plan changes."""

        if gameplan is None:
            return
        win_con = gameplan.primary_win_con_species
        threat = gameplan.primary_threat_species
        if self.initial_win_con is None:
            self.initial_win_con = win_con
            self.initial_primary_threat = threat
        if self.current_win_con is not None and win_con != self.current_win_con:
            self.plan_changes.append(
                {
                    "turn": self.current_turn,
                    "from": self.current_win_con,
                    "to": win_con,
                    "reason": "previous plan no longer best or available",
                }
            )
        self.current_win_con = win_con
        self.current_primary_threat = threat
        self.plan_breakers = gameplan.plan_breakers

    def record_choice(self, turn: int, order: str) -> None:
        if not self.our_orders or self.our_orders[-1] != (turn, order):
            self.our_orders.append((turn, order))
        if self.turns and self.turns[-1].turn == turn:
            self.turns[-1].chosen_order = order

    def target_bias(self, species_id: str) -> float:
        """Observed share of opponent single-target moves aimed at ``species_id``."""

        total = sum(self.opponent_targets.values())
        return self.opponent_targets[to_id(species_id)] / total if total else 0.0

    def summary(self) -> dict[str, object]:
        return {
            "turns_observed": len(self.turns),
            "initial_win_con": self.initial_win_con,
            "current_win_con": self.current_win_con,
            "initial_primary_threat": self.initial_primary_threat,
            "current_primary_threat": self.current_primary_threat,
            "plan_breakers": sorted(self.plan_breakers),
            "plan_changes": list(self.plan_changes),
            "opponent_moves": {
                actor: dict(counts) for actor, counts in sorted(self.opponent_moves.items())
            },
            "opponent_targets": dict(self.opponent_targets),
            "opponent_switches": dict(self.opponent_switches),
            "opponent_protects": dict(self.opponent_protects),
            "our_orders": [{"turn": turn, "order": order} for turn, order in self.our_orders],
        }
