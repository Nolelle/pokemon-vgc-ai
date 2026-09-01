"""Fog-safe probability estimates for the opponent's still-hidden set details.

The simulator knows both teams, but a real player does not.  This module deliberately
accepts only the player's parsed battle view, public replay-derived priors, and facts
recorded in :mod:`vgc.battle_memory`.  It never reads the simulator's private side.

Each possible Stat Point spread/nature starts with its corpus usage weight.  Visible
move order and damage then reweight those possibilities.  A mismatch is kept at a
small probability instead of being deleted because speed ties, critical hits, and
mechanics not covered by the local calculator can make one observation noisy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from vgc.battle_memory import BattleMemory
from vgc.damage import FieldState, PokemonState, damage_range, to_id
from vgc.evaluator import field_effective_speed
from vgc.models import PolicyConfig
from vgc.sets import load_set_priors, load_usage_spreads, normalize_item, normalize_status
from vgc.stats import (
    STAT_IDS,
    calculate_stats,
    default_opponent_nature,
    default_opponent_spread,
)

MAX_SPREAD_HYPOTHESES = 3
_EVIDENCE_MISMATCH_WEIGHT = 0.05
_DAMAGE_TOLERANCE_PERCENT = 2.0


@dataclass(frozen=True)
class SpreadHypothesis:
    sp: dict[str, int]
    nature: str
    probability: float

    def stats(self, species_id: str) -> dict[str, int]:
        return calculate_stats(species_id, self.sp, self.nature)


@dataclass(frozen=True)
class PokemonBelief:
    species_id: str
    hypotheses: tuple[SpreadHypothesis, ...]
    move_probabilities: dict[str, float] = field(default_factory=dict)
    item_probabilities: dict[str, float] = field(default_factory=dict)
    ability_probabilities: dict[str, float] = field(default_factory=dict)
    revealed_moves: tuple[str, ...] = ()
    known_item: str | None = None
    known_ability: str | None = None
    appeared: bool = False
    active: bool = False
    hp_fraction: float = 1.0
    fainted: bool = False

    @property
    def speed_min(self) -> float:
        return float(min(h.stats(self.species_id)["spe"] for h in self.hypotheses))

    @property
    def speed_max(self) -> float:
        return float(max(h.stats(self.species_id)["spe"] for h in self.hypotheses))

    @property
    def speed_mean(self) -> float:
        return float(
            sum(
                h.probability * h.stats(self.species_id)["spe"] for h in self.hypotheses
            )
        )

    @property
    def spread_entropy(self) -> float:
        if len(self.hypotheses) <= 1:
            return 0.0
        entropy = -sum(
            h.probability * math.log(max(h.probability, 1e-12)) for h in self.hypotheses
        )
        return float(entropy / math.log(len(self.hypotheses)))

    def move_probability(self, move_id: str) -> float:
        return float(self.move_probabilities.get(to_id(move_id), 0.0))


def _estimated_field(probabilities: dict[str, float]) -> dict[str, object]:
    """Trace-safe label for one hidden categorical field."""

    ranked = sorted(probabilities.items(), key=lambda entry: (-entry[1], entry[0]))
    if not ranked:
        return {"status": "unknown", "possibilities": []}
    return {
        "status": "estimated",
        "possibilities": [
            {"value": value, "probability": round(float(probability), 6)}
            for value, probability in ranked[:4]
        ],
    }


def _public_attr(battle, name: str, default=None):
    """Read a public battle property without letting an incomplete request break tracing."""

    try:
        return getattr(battle, name, default)
    except (AttributeError, ValueError):
        return default


def information_boundary_summary(
    battle,
    memory: BattleMemory,
    config: PolicyConfig | None = None,
) -> dict[str, object]:
    """Human-auditable known/estimated/unknown labels for one public observation.

    This deliberately reads through :func:`build_opponent_beliefs`, the same fog-safe
    path used by the learned input and live mirror. It never inspects a simulator root or
    any private opponent-team attribute.
    """

    config = config or PolicyConfig()
    own_entries = list(_public_attr(battle, "teambuilder_team") or [])[:6]
    own_sets = [
        {
            "species": to_id(getattr(entry, "species", None) or entry.nickname),
            "moves": [to_id(move) for move in (getattr(entry, "moves", None) or [])],
            "item": to_id(getattr(entry, "item", None)),
            "ability": to_id(getattr(entry, "ability", None)),
            "nature": str(getattr(entry, "nature", None) or "serious").lower(),
            "stat_points": dict(
                zip(STAT_IDS, getattr(entry, "evs", None) or [0] * 6, strict=True)
            ),
        }
        for entry in own_entries
    ]
    opponents = []
    for belief in build_opponent_beliefs(battle, memory, config):
        revealed_moves = set(belief.revealed_moves)
        estimated_moves = {
            move_id: probability
            for move_id, probability in belief.move_probabilities.items()
            if move_id not in revealed_moves
        }
        item = (
            {"status": "known", "value": to_id(belief.known_item)}
            if belief.known_item
            else _estimated_field(belief.item_probabilities)
        )
        ability = (
            {"status": "known", "value": to_id(belief.known_ability)}
            if belief.known_ability
            else _estimated_field(belief.ability_probabilities)
        )
        opponents.append(
            {
                "species": {"status": "known", "value": belief.species_id},
                "brought": {
                    "status": "known" if belief.appeared else "unknown",
                    "value": True if belief.appeared else None,
                },
                "moves": {
                    "known": sorted(revealed_moves),
                    "estimated": _estimated_field(estimated_moves),
                    "unknown_slots": max(0, 4 - len(revealed_moves)),
                },
                "item": item,
                "ability": ability,
                "spread_and_nature": {
                    "status": "estimated",
                    "possibilities": [
                        {
                            "stat_points": hypothesis.sp,
                            "nature": hypothesis.nature,
                            "probability": round(float(hypothesis.probability), 6),
                        }
                        for hypothesis in belief.hypotheses
                    ],
                },
                "current_state": {
                    "status": "known" if belief.appeared else "unknown",
                },
            }
        )
    return {
        "contract": "own_exact_opponent_public_beliefs_v1",
        "opponent_sheet_policy": (
            "accept" if config.accept_open_team_sheet else "reject"
        ),
        "own_team": {
            "status": "known",
            "set_count": len(own_entries),
            "sets": own_sets,
            "fields": [
                "species",
                "moves",
                "item",
                "ability",
                "nature",
                "stat_points",
            ],
        },
        "opponent": opponents,
        "opponent_next_action": {"status": "unknown"},
        "opponent_selected_four": {"status": "unknown_until_revealed"},
        "future_random_outcomes": {"status": "unknown", "handling": "showdown_branches"},
        "private_effect_durations": {"status": "estimated", "handling": "weighted_branches"},
    }


def _normalize_hypotheses(
    entries: list[tuple[dict[str, int], str, float]],
) -> list[SpreadHypothesis]:
    total = sum(max(0.0, weight) for _sp, _nature, weight in entries)
    if total <= 0.0:
        total = float(len(entries))
        entries = [(sp, nature, 1.0) for sp, nature, _weight in entries]
    return [
        SpreadHypothesis(sp=dict(sp), nature=str(nature).lower(), probability=weight / total)
        for sp, nature, weight in entries
    ]


def _initial_hypotheses(species_id: str, usage: dict[str, list[dict[str, Any]]]) -> list:
    rows = sorted(
        usage.get(species_id) or [],
        key=lambda entry: float(entry.get("weight", 0.0)),
        reverse=True,
    )[:MAX_SPREAD_HYPOTHESES]
    if rows:
        entries = [
            (dict(row.get("sp") or {}), str(row.get("nature") or "serious"), float(row.get("weight", 0.0)))
            for row in rows
        ]
    else:
        entries = [
            (default_opponent_spread(species_id), default_opponent_nature(species_id), 1.0)
        ]
    return _normalize_hypotheses(entries)


def _reweight(
    species_id: str,
    hypotheses: list[SpreadHypothesis],
    memory: BattleMemory,
    *,
    status: str | None,
    item: str | None,
    ability: str | None,
) -> tuple[SpreadHypothesis, ...]:
    weighted: list[tuple[dict[str, int], str, float]] = []
    speed_evidence = [
        observation
        for observation in memory.speed_observations
        if observation.opponent_species == species_id
    ]
    damage_evidence = [
        observation
        for observation in memory.damage_observations
        if observation.opponent_species == species_id
    ]
    for hypothesis in hypotheses:
        evidence_weight = 1.0
        for observation in speed_evidence:
            state = PokemonState(
                species_id=species_id,
                sp_spread=hypothesis.sp,
                nature=hypothesis.nature,
                status=observation.status or status,
                item=observation.item or item,
                ability=observation.ability or ability,
            )
            speed = field_effective_speed(
                state,
                weather=observation.weather,
                tailwind=observation.tailwind,
            )
            matches = (
                speed >= observation.threshold
                if observation.relation == "at_least"
                else speed <= observation.threshold
            )
            if not matches:
                evidence_weight *= _EVIDENCE_MISMATCH_WEIGHT

        for observation in damage_evidence:
            attacker = PokemonState(
                species_id=species_id,
                sp_spread=hypothesis.sp,
                nature=hypothesis.nature,
                item=observation.item or item,
                ability=observation.ability or ability,
            )
            result = damage_range(
                attacker,
                observation.target_state,
                observation.move_id,
                FieldState(
                    weather=observation.weather,
                    terrain=observation.terrain,
                    screens=observation.screens,
                    is_doubles=True,
                    num_targets=observation.num_targets,
                ),
            )
            if not result.breakdown.get("move_supported", False):
                continue
            observed_percent = observation.damage_fraction * 100.0
            matches = (
                result.min_percent - _DAMAGE_TOLERANCE_PERCENT
                <= observed_percent
                <= result.max_percent + _DAMAGE_TOLERANCE_PERCENT
            )
            if not matches:
                evidence_weight *= _EVIDENCE_MISMATCH_WEIGHT
        weighted.append(
            (hypothesis.sp, hypothesis.nature, hypothesis.probability * evidence_weight)
        )
    return tuple(_normalize_hypotheses(weighted))


def _frequency_probabilities(
    entry: dict[str, Any] | None, key: str, *, known: str | None = None
) -> dict[str, float]:
    if known:
        return {known: 1.0}
    if not entry:
        return {}
    appearances = max(1, int(entry.get("appearances", 0) or 0))
    counts = entry.get(key) or {}
    return {
        to_id(value): min(1.0, max(0.0, float(count) / appearances))
        for value, count in counts.items()
        if to_id(value)
    }


def _preview_species(battle) -> list[str]:
    preview = list(_public_attr(battle, "teampreview_opponent_team") or [])
    species = [to_id(getattr(mon, "species", None)) for mon in preview]
    appeared = [
        to_id(getattr(mon, "species", None))
        for mon in (_public_attr(battle, "opponent_team") or {}).values()
    ]
    for species_id in appeared:
        if species_id and species_id not in species:
            species.append(species_id)
    return [species_id for species_id in species if species_id][:6]


def build_opponent_beliefs(
    battle,
    memory: BattleMemory,
    config: PolicyConfig | None = None,
    *,
    usage: dict[str, list[dict[str, Any]]] | None = None,
    set_priors: dict[str, Any] | None = None,
) -> list[PokemonBelief]:
    """Return one probability estimate per visible opponent preview species."""

    config = config or PolicyConfig()
    usage = load_usage_spreads() if usage is None else usage
    set_priors = load_set_priors() if set_priors is None else set_priors
    observed_by_species = {
        to_id(getattr(mon, "species", None)): mon
        for mon in (_public_attr(battle, "opponent_team") or {}).values()
    }
    active_species = {
        to_id(getattr(mon, "species", None))
        for mon in (_public_attr(battle, "opponent_active_pokemon") or [])
        if mon is not None and not getattr(mon, "fainted", False)
    }
    beliefs: list[PokemonBelief] = []
    for species_id in _preview_species(battle):
        pokemon = observed_by_species.get(species_id)
        revealed_moves = set(memory.opponent_moves.get(species_id, {}))
        if pokemon is not None:
            revealed_moves.update(
                move_id
                for move_id in (to_id(raw) for raw in (getattr(pokemon, "moves", {}) or {}))
                if move_id
            )
        known_item = memory.opponent_items.get(species_id)
        known_ability = memory.opponent_abilities.get(species_id)
        status = None
        if pokemon is not None:
            known_item = known_item or normalize_item(getattr(pokemon, "item", None))
            known_ability = known_ability or to_id(getattr(pokemon, "ability", None)) or None
            status = normalize_status(getattr(pokemon, "status", None))

        prior_entry = (set_priors.get("species") or {}).get(species_id) if set_priors else None
        move_probabilities = _frequency_probabilities(prior_entry, "moves")
        move_probabilities.update({move_id: 1.0 for move_id in revealed_moves})
        hypotheses = _reweight(
            species_id,
            _initial_hypotheses(species_id, usage),
            memory,
            status=status,
            item=known_item,
            ability=known_ability,
        )
        beliefs.append(
            PokemonBelief(
                species_id=species_id,
                hypotheses=hypotheses,
                move_probabilities=move_probabilities,
                item_probabilities=_frequency_probabilities(
                    prior_entry, "items", known=known_item
                ),
                ability_probabilities=_frequency_probabilities(
                    prior_entry, "abilities", known=known_ability
                ),
                revealed_moves=tuple(sorted(revealed_moves))[:4],
                known_item=known_item,
                known_ability=known_ability,
                appeared=pokemon is not None,
                active=species_id in active_species,
                hp_fraction=float(
                    getattr(pokemon, "current_hp_fraction", 1.0) if pokemon is not None else 1.0
                ),
                fainted=bool(getattr(pokemon, "fainted", False)) if pokemon is not None else False,
            )
        )
    return beliefs
