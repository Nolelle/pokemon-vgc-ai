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
from typing import Any, Iterable

from vgc.damage import to_id


@dataclass(frozen=True)
class SpeedObservation:
    """A fog-safe relative Speed fact inferred from one same-priority move pair."""

    opponent_species: str
    threshold: float
    relation: str  # "at_least" or "at_most" effective Speed
    weather: str | None = None
    tailwind: bool = False
    trick_room: bool = False
    status: str | None = None
    item: str | None = None
    ability: str | None = None


@dataclass(frozen=True)
class DamageObservation:
    """Direct move damage that can narrow an opponent's possible stat spread."""

    opponent_species: str
    move_id: str
    target_species: str
    damage_fraction: float
    target_state: Any
    weather: str | None = None
    terrain: str | None = None
    screens: frozenset[str] = frozenset()
    item: str | None = None
    ability: str | None = None
    num_targets: int = 1


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
    trick_room: bool = False
    opponent_tailwind: bool = False
    our_effective_speed: dict[str, float] = field(default_factory=dict)
    our_states: dict[str, Any] = field(default_factory=dict)
    opponent_status: dict[str, str | None] = field(default_factory=dict)
    opponent_item: dict[str, str | None] = field(default_factory=dict)
    opponent_ability: dict[str, str | None] = field(default_factory=dict)
    terrain: str | None = None
    our_screens: frozenset[str] = frozenset()


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
    speed_observations: list[SpeedObservation] = field(default_factory=list)
    damage_observations: list[DamageObservation] = field(default_factory=list)
    opponent_items: dict[str, str] = field(default_factory=dict)
    opponent_abilities: dict[str, str] = field(default_factory=dict)
    _ident_species: dict[str, str] = field(default_factory=dict, repr=False)
    _turn_actions: list[tuple[str, str, int]] = field(default_factory=list, repr=False)
    _last_move: tuple[str, str, str] | None = field(default=None, repr=False)
    _protocol_hp: dict[tuple[str, str], float] = field(default_factory=dict, repr=False)

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
                self._turn_actions.clear()
                self._last_move = None
                continue
            if kind == "move" and len(message) > 3:
                actor_role = _side(message[2])
                actor = self._species_for_ident(message[2])
                move_id = to_id(message[3]) or "unknown"
                self._last_move = (actor_role or "", actor, move_id)
                self._observe_action_order(actor_role, actor, move_id)
                if self.opponent_role is not None and actor_role == self.opponent_role:
                    self.opponent_moves[actor][move_id] += 1
                    if move_id in {"protect", "detect", "kingsshield", "spikyshield"}:
                        self.opponent_protects[actor] += 1
                    if len(message) > 4 and _side(message[4]) == self.our_role:
                        target = self._species_for_ident(message[4])
                        self.opponent_targets[target] += 1
                continue
            if kind in {"-ability", "ability"} and len(message) > 3:
                if _side(message[2]) == self.opponent_role:
                    self.opponent_abilities[self._species_for_ident(message[2])] = (
                        to_id(message[3]) or "unknown"
                    )
                continue
            if kind in {"-item", "item"} and len(message) > 3:
                if _side(message[2]) == self.opponent_role:
                    self.opponent_items[self._species_for_ident(message[2])] = (
                        to_id(message[3]) or "unknown"
                    )
                continue
            if kind == "-enditem" and len(message) > 3:
                if _side(message[2]) == self.opponent_role:
                    # Keep the revealed original item as knowledge even after consumption.
                    self.opponent_items.setdefault(
                        self._species_for_ident(message[2]), to_id(message[3]) or "unknown"
                    )
                continue
            if kind == "-damage" and len(message) > 3:
                self._observe_damage(message)
                continue
            if kind in {"switch", "drag", "replace"} and len(message) > 2:
                if _side(message[2]) == self.opponent_role:
                    species_text = message[3] if len(message) > 3 else _display_name(message[2])
                    species = to_id(str(species_text).split(",", 1)[0]) or "unknown"
                    self._ident_species[str(message[2])] = species
                    self.opponent_switches[species] += 1

    def _species_for_ident(self, token: object) -> str:
        raw = str(token or "")
        return self._ident_species.get(raw, to_id(_display_name(raw)) or "unknown")

    @staticmethod
    def _hp_fraction(token: object) -> float | None:
        hp = str(token or "").split(" ", 1)[0]
        if hp == "0":
            return 0.0
        if "/" not in hp:
            return None
        current, maximum = hp.split("/", 1)
        try:
            maximum_value = float(maximum)
            return float(current) / maximum_value if maximum_value > 0 else None
        except ValueError:
            return None

    def _observe_action_order(self, actor_role: str | None, actor: str, move_id: str) -> None:
        """Record a conservative relative-Speed observation from visible action order."""

        if actor_role not in {self.our_role, self.opponent_role}:
            return
        try:
            from vgc.data import load_moves

            move = load_moves().get(move_id) or {}
            # Status moves can have hidden dynamic priority (for example Prankster), so
            # they are not safe evidence from the public log alone.
            if move.get("category") == "Status":
                return
            priority = int(move.get("priority", 0))
        except (TypeError, ValueError):
            return

        previous = next(
            (
                entry
                for entry in reversed(self._turn_actions)
                if entry[0] != actor_role and entry[2] == priority
            ),
            None,
        )
        self._turn_actions.append((actor_role, actor, priority))
        if previous is None or not self.turns:
            return
        snapshot = self.turns[-1]
        if snapshot.turn != self.current_turn:
            return
        if actor_role == self.opponent_role:
            opponent_species = actor
            our_species = previous[1]
            opponent_was_first = False
        else:
            opponent_species = previous[1]
            our_species = actor
            opponent_was_first = True
        threshold = snapshot.our_effective_speed.get(our_species)
        if threshold is None:
            return
        # Within one priority bracket, faster acts first normally and slower acts first
        # under Trick Room. Inclusive bounds keep random Speed ties possible.
        at_least = opponent_was_first != snapshot.trick_room
        self.speed_observations.append(
            SpeedObservation(
                opponent_species=opponent_species,
                threshold=threshold,
                relation="at_least" if at_least else "at_most",
                weather=snapshot.weather[0] if snapshot.weather else None,
                tailwind=snapshot.opponent_tailwind,
                trick_room=snapshot.trick_room,
                status=snapshot.opponent_status.get(opponent_species),
                item=snapshot.opponent_item.get(opponent_species),
                ability=snapshot.opponent_ability.get(opponent_species),
            )
        )

    def _observe_damage(self, message: list[object]) -> None:
        if self._last_move is None or any("[from]" in str(part) for part in message[4:]):
            return
        target_role = _side(message[2])
        attacker_role, attacker_species, move_id = self._last_move
        if attacker_role != self.opponent_role or target_role != self.our_role or not self.turns:
            return
        target_species = self._species_for_ident(message[2])
        new_fraction = self._hp_fraction(message[3])
        if new_fraction is None:
            return
        key = (target_role or "", target_species)
        old_fraction = self._protocol_hp.get(key)
        if old_fraction is None:
            old_fraction = self.turns[-1].our_hp.get(target_species, 0.0) / 100.0
        self._protocol_hp[key] = new_fraction
        lost = old_fraction - new_fraction
        target_state = self.turns[-1].our_states.get(target_species)
        if lost <= 0.0 or target_state is None:
            return
        snapshot = self.turns[-1]
        self.damage_observations.append(
            DamageObservation(
                opponent_species=attacker_species,
                move_id=move_id,
                target_species=target_species,
                damage_fraction=lost,
                target_state=target_state,
                weather=snapshot.weather[0] if snapshot.weather else None,
                terrain=snapshot.terrain,
                screens=snapshot.our_screens,
                item=snapshot.opponent_item.get(attacker_species),
                ability=snapshot.opponent_ability.get(attacker_species),
            )
        )

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

        our_states: dict[str, Any] = {}
        our_effective_speed: dict[str, float] = {}
        opponent_status: dict[str, str | None] = {}
        opponent_item: dict[str, str | None] = {}
        opponent_ability: dict[str, str | None] = {}
        weather_id: str | None = None
        terrain_id: str | None = None
        trick_room = False
        opponent_tailwind = False
        our_screens: frozenset[str] = frozenset()
        try:
            from poke_env.battle.field import Field
            from poke_env.battle.side_condition import SideCondition

            from vgc.evaluator import (
                _our_pokemon_state,
                _screens_from,
                _terrain_str,
                _weather_str,
                field_effective_speed,
            )
            from vgc.sets import normalize_item, normalize_status

            weather_id = _weather_str(battle)
            terrain_id = _terrain_str(battle)
            trick_room = Field.TRICK_ROOM in (getattr(battle, "fields", {}) or {})
            our_tailwind = SideCondition.TAILWIND in (
                getattr(battle, "side_conditions", {}) or {}
            )
            opponent_tailwind = SideCondition.TAILWIND in (
                getattr(battle, "opponent_side_conditions", {}) or {}
            )
            our_screens = _screens_from(getattr(battle, "side_conditions", {}) or {})
            for mon in ours:
                species = to_id(getattr(mon, "species", None))
                state = _our_pokemon_state(mon)
                our_states[species] = state
                our_effective_speed[species] = field_effective_speed(
                    state, weather=weather_id, tailwind=our_tailwind
                )
            for mon in theirs:
                species = to_id(getattr(mon, "species", None))
                opponent_status[species] = normalize_status(getattr(mon, "status", None))
                opponent_item[species] = self.opponent_items.get(
                    species, normalize_item(getattr(mon, "item", None))
                )
                opponent_ability[species] = self.opponent_abilities.get(
                    species, to_id(getattr(mon, "ability", None)) or None
                )
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

        for side_name, team in (
            (self.our_role, getattr(battle, "team", None) or {}),
            (self.opponent_role, getattr(battle, "opponent_team", None) or {}),
        ):
            for ident, mon in team.items():
                species = to_id(getattr(mon, "species", None)) or "unknown"
                self._ident_species[str(ident)] = species
                self._protocol_hp[(side_name or "", species)] = float(
                    getattr(mon, "current_hp_fraction", 0.0) or 0.0
                )

        snapshot = TurnMemory(
            turn=turn,
            our_active=names(ours),
            opponent_active=names(theirs),
            our_hp=hp(ours),
            opponent_hp=hp(theirs),
            weather=(weather_id,) if weather_id else (),
            trick_room=trick_room,
            opponent_tailwind=opponent_tailwind,
            our_effective_speed=our_effective_speed,
            our_states=our_states,
            opponent_status=opponent_status,
            opponent_item=opponent_item,
            opponent_ability=opponent_ability,
            terrain=terrain_id,
            our_screens=our_screens,
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

    def move_frequency(self, move_id: str) -> float:
        """Share of all observed opponent moves matching ``move_id``."""

        total = sum(sum(counts.values()) for counts in self.opponent_moves.values())
        count = sum(counts[to_id(move_id)] for counts in self.opponent_moves.values())
        return count / total if total else 0.0

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
            "speed_observations": [
                {
                    "species": entry.opponent_species,
                    "relation": entry.relation,
                    "threshold": entry.threshold,
                }
                for entry in self.speed_observations
            ],
            "damage_observations": [
                {
                    "species": entry.opponent_species,
                    "move": entry.move_id,
                    "damage_fraction": entry.damage_fraction,
                }
                for entry in self.damage_observations
            ],
        }
