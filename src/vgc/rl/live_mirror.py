"""Build a mechanics-exact Showdown branch root from one live public observation.

Public battles do not expose the server's private ``Battle`` object.  This module starts
a fresh local Showdown battle from our known team and a fog-safe opponent hypothesis,
then patches every currently observable mechanics field before any branch is evaluated.
Unknown opponent Stat Points, nature, unrevealed bring choices, or rejected-sheet sets
remain explicit estimates; they are information uncertainty, not hand-written battle
rules.  All transitions after the patch are executed by Showdown itself.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from poke_env.teambuilder.teambuilder import Teambuilder

from vgc.battle_memory import BattleMemory
from vgc.damage import to_id
from vgc.data import load_items, load_learnsets, load_moves, load_species
from vgc.mechanics_state import snapshot_battle
from vgc.models import PolicyConfig
from vgc.opponent_belief import build_opponent_beliefs
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.exact_search import combine_belief_rankings
from vgc.rl.hidden_state import HiddenStateHypothesis, enumerate_hidden_state_hypotheses
from vgc.sets import (
    load_set_priors,
    normalize_item,
    opponent_move_ids,
    opponent_spread_hypotheses,
    opponent_state,
)
from vgc.stats import STAT_IDS


def _base_species_id(species_id: str) -> str:
    data = load_species().get(species_id) or {}
    return to_id(data.get("baseSpecies")) or species_id


def _ordered_unique(values):
    return list(dict.fromkeys(value for value in values if value))


def _team_preview_order(packed_team: str, battle) -> str:
    entries = Teambuilder.parse_packed_team(packed_team)
    species = [to_id(entry.species or entry.nickname) for entry in entries]
    active = [
        _base_species_id(to_id(getattr(mon, "species", None)))
        for mon in (getattr(battle, "active_pokemon", None) or ())
        if mon is not None
    ]
    switches = [
        _base_species_id(to_id(getattr(mon, "species", None)))
        for slot in (getattr(battle, "available_switches", None) or ())
        for mon in slot
    ]
    fainted = [
        _base_species_id(to_id(getattr(mon, "species", None)))
        for mon in (getattr(battle, "team", None) or {}).values()
        if getattr(mon, "fainted", False)
    ]
    wanted = _ordered_unique([*active, *switches, *fainted])
    ordered_indices: list[int] = []
    for wanted_id in wanted:
        match = next(
            (
                index
                for index, species_id in enumerate(species)
                if index not in ordered_indices and _base_species_id(species_id) == wanted_id
            ),
            None,
        )
        if match is not None:
            ordered_indices.append(match)
    ordered_indices.extend(index for index in range(len(entries)) if index not in ordered_indices)
    # Champions VGC is bring-four. The first two digits become the current template
    # leads and the next two are a concrete bench hypothesis.
    return "team " + "".join(str(index + 1) for index in ordered_indices[:4])


def _known_opponent_by_species(battle) -> dict[str, object]:
    known: dict[str, object] = {}
    for mon in (getattr(battle, "opponent_team", None) or {}).values():
        species_id = _base_species_id(to_id(getattr(mon, "species", None)))
        known[species_id] = mon
    return known


def _fallback_moves(species_id: str) -> list[str]:
    learnset = list((load_learnsets().get(species_id) or {}).keys())
    moves = load_moves()
    ranked = sorted(
        learnset,
        key=lambda move_id: (
            move_id != "protect",
            -int((moves.get(move_id) or {}).get("basePower") or 0),
            move_id,
        ),
    )
    return ranked[:4]


def _opponent_sets(
    battle,
    config: PolicyConfig,
    spreads: dict[str, tuple[dict[str, int], str]] | None = None,
    set_assignments: dict[str, dict[str, object]] | None = None,
    brought: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    """Six opponent sets rebuilt from public information only.

    ``spreads`` overrides the hidden Stat Points/nature for the species it names, so a
    caller can build one root per belief in the distribution instead of hardening
    `opponent_state`'s single most-popular guess into a fact. Species it omits keep
    that point estimate.
    """

    preview = list(getattr(battle, "teampreview_opponent_team", None) or ())
    known = _known_opponent_by_species(battle)
    active_ids = [
        _base_species_id(to_id(getattr(mon, "species", None)))
        for mon in (getattr(battle, "opponent_active_pokemon", None) or ())
        if mon is not None
    ]
    preview_by_id = {
        _base_species_id(to_id(getattr(mon, "species", None))): mon for mon in preview
    }
    # Once a Pokemon has appeared, it is publicly known to be one of the brought four.
    # Keep all such species ahead of preview-only possibilities so a later-turn mirror
    # cannot accidentally leave a fainted former active outside its concrete bring.
    ordered_ids = _ordered_unique([*active_ids, *known, *(brought or ()), *preview_by_id])
    priors = load_set_priors()
    species_data = load_species()
    legal_items = load_items()
    used_items: set[str] = set()
    result: list[dict[str, object]] = []
    for species_id in ordered_ids:
        mon = known.get(species_id) or preview_by_id[species_id]
        state = opponent_state(mon)
        set_assignment = (set_assignments or {}).get(species_id) or {}
        assigned_moves = [to_id(move) for move in set_assignment.get("moves", [])]
        moves = [
            move_id
            for move_id in (
                assigned_moves
                or opponent_move_ids(mon, priors=priors, config=config)
            )
            if move_id in (load_learnsets().get(species_id) or {})
        ][:4]
        if not moves:
            moves = _fallback_moves(species_id)
        item = normalize_item(getattr(mon, "item", None)) or normalize_item(
            set_assignment.get("item")
        )
        prior = ((priors.get("species") or {}).get(species_id) or {})
        if not item:
            ranked_items = sorted(
                (prior.get("items") or {}).items(), key=lambda row: (-row[1], row[0])
            )
            item = next(
                (
                    item_id
                    for item_id, _count in ranked_items
                    if item_id in legal_items and item_id not in used_items
                ),
                None,
            )
        if item in used_items:
            item = None
        if item:
            used_items.add(item)
        ability = to_id(getattr(mon, "ability", None)) or to_id(
            set_assignment.get("ability")
        )
        if not ability:
            abilities = (species_data.get(species_id) or {}).get("abilities") or {}
            ability = to_id(next(iter(abilities.values()), ""))
        believed = (spreads or {}).get(species_id)
        spread = (believed[0] if believed else state.sp_spread) or dict.fromkeys(STAT_IDS, 0)
        nature = (believed[1] if believed else state.nature) or "serious"
        result.append(
            {
                "name": species_id,
                "species": species_id,
                "item": item or "",
                "ability": ability,
                "moves": moves,
                "nature": nature,
                "evs": {stat: int(spread.get(stat, 0) or 0) for stat in STAT_IDS},
                "ivs": dict.fromkeys(STAT_IDS, 31),
                "level": 50,
            }
        )
    if len(result) != 6:
        raise ValueError(
            "an exact live mirror needs all six opponent preview species; "
            f"the public observation currently has {len(result)}"
        )
    return result


def _active_opponent_species(battle) -> list[str]:
    return _ordered_unique(
        _base_species_id(to_id(getattr(mon, "species", None)))
        for mon in (getattr(battle, "opponent_active_pokemon", None) or ())
        if mon is not None
    )


def _species_spread_beliefs(
    battle, config: PolicyConfig, memory, limit: int
) -> dict[str, list[tuple[dict, str, float]]]:
    """Per-species spread beliefs, POSTERIOR where the battle has told us something.

    `vgc.opponent_belief.build_opponent_beliefs` starts from the same corpus weights that
    `vgc.sets.opponent_spread_hypotheses` returns, then reweights them by what this battle
    has actually shown -- who moved first at what speed, and how hard a move hit. Using
    the static prior when that evidence exists would throw away the only opponent
    information a real game ever hands us.

    Falls back to the flat corpus prior when no memory is available (offline callers and
    unit tests), which is the same distribution before any evidence arrives.
    """

    species_ids = _active_opponent_species(battle)
    if memory is None:
        return {
            species_id: opponent_spread_hypotheses(species_id, limit=limit)
            for species_id in species_ids
        }
    posterior = {
        belief.species_id: belief
        for belief in build_opponent_beliefs(battle, memory, config)
    }
    result: dict[str, list[tuple[dict, str, float]]] = {}
    for species_id in species_ids:
        belief = posterior.get(species_id)
        if belief is None or not belief.hypotheses:
            result[species_id] = opponent_spread_hypotheses(species_id, limit=limit)
            continue
        kept = sorted(
            belief.hypotheses, key=lambda entry: -entry.probability
        )[:limit]
        total = sum(entry.probability for entry in kept)
        if total <= 0.0:
            result[species_id] = opponent_spread_hypotheses(species_id, limit=limit)
            continue
        result[species_id] = [
            (entry.sp, entry.nature, entry.probability / total) for entry in kept
        ]
    return result


def _spread_beliefs(battle, config: PolicyConfig, memory=None) -> list[tuple[float, dict]]:
    """Weighted Stat Point/nature beliefs for the opponent's ACTIVE Pokemon.

    Only the active pair varies. Their spreads decide this turn's damage, speed order,
    and survival, which is what the search is actually asking about; branching the bench
    too would multiply Showdown roots for hypotheses that cannot change the current
    ranking. Benched species keep `opponent_state`'s point estimate.

    Returns ``(probability, {species_id: (spread, nature)})`` pairs, most likely first
    and renormalized after the cap. A cap of 1 -- the shipped default -- returns exactly
    one belief holding each species' most likely spread, i.e. today's behavior.
    """

    limit = config.exact_search_spread_hypotheses
    if limit < 1:
        raise ValueError(
            f"exact_search_spread_hypotheses must be at least 1, got {limit!r}"
        )
    by_species = _species_spread_beliefs(battle, config, memory, limit)
    per_species = [
        (species_id, by_species[species_id])
        for species_id in _active_opponent_species(battle)
    ]
    if not per_species:
        return [(1.0, {})]
    combined: list[tuple[float, dict]] = []
    for choice in itertools.product(*(hypotheses for _species, hypotheses in per_species)):
        weight = 1.0
        assignment: dict[str, tuple[dict[str, int], str]] = {}
        for (species_id, _hypotheses), (spread, nature, probability) in zip(
            per_species, choice, strict=True
        ):
            weight *= probability
            assignment[species_id] = (spread, nature)
        combined.append((weight, assignment))
    combined.sort(key=lambda row: row[0], reverse=True)
    combined = combined[:limit]
    total = sum(weight for weight, _assignment in combined)
    return [(weight / total, assignment) for weight, assignment in combined]


def _set_beliefs(battle, config: PolicyConfig, memory=None) -> list[tuple[float, dict]]:
    """Joint move/item/ability possibilities for the active opponent Pokemon."""

    limit = int(config.exact_search_set_hypotheses)
    if limit < 1:
        raise ValueError("exact_search_set_hypotheses must be at least 1")
    beliefs = {
        belief.species_id: belief
        for belief in build_opponent_beliefs(
            battle,
            memory
            or getattr(battle, "_vgc_battle_memory", None)
            or BattleMemory("live-mirror"),
            config,
        )
    }
    axes: list[tuple[str, list]] = []
    for species_id in _active_opponent_species(battle):
        belief = beliefs.get(species_id)
        entries = list(belief.set_hypotheses if belief is not None else ())
        if not entries:
            continue
        axes.append((species_id, entries[:limit]))
    if not axes:
        return [(1.0, {})]
    combined: list[tuple[float, dict]] = []
    for choice in itertools.product(*(entries for _species, entries in axes)):
        weight = 1.0
        assignment: dict[str, dict[str, object]] = {}
        for (species_id, _entries), entry in zip(axes, choice, strict=True):
            weight *= entry.probability
            assignment[species_id] = {
                "moves": entry.moves,
                "item": entry.item,
                "ability": entry.ability,
                "source": entry.source,
            }
        combined.append((weight, assignment))
    combined.sort(key=lambda row: (-row[0], str(row[1])))
    kept = combined[:limit]
    total = sum(weight for weight, _assignment in kept)
    return [(weight / total, assignment) for weight, assignment in kept]


def _bring_beliefs(battle, config: PolicyConfig) -> list[tuple[float, tuple[str, ...]]]:
    """Plausible brought-four sets consistent with every public appearance."""

    limit = int(config.exact_search_bring_hypotheses)
    if limit < 1:
        raise ValueError("exact_search_bring_hypotheses must be at least 1")
    preview = _ordered_unique(
        _base_species_id(to_id(getattr(mon, "species", None)))
        for mon in (getattr(battle, "teampreview_opponent_team", None) or ())
    )
    appeared = set(_known_opponent_by_species(battle))
    if len(preview) < 4:
        return [(1.0, tuple(preview))]
    candidates: list[tuple[float, tuple[str, ...]]] = []
    try:
        from vgc.preview_predict import bring4_distribution, predict_preview_hybrid

        ours = [
            _base_species_id(to_id(getattr(mon, "species", None)))
            for mon in (getattr(battle, "teampreview_team", None) or ())
        ]
        predicted = bring4_distribution(predict_preview_hybrid(preview, ours, config))
        for indices, probability in predicted:
            brought = tuple(preview[index] for index in indices)
            if appeared.issubset(brought):
                candidates.append((probability, brought))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        candidates = []
    if not candidates:
        candidates = [
            (1.0, tuple(combo))
            for combo in itertools.combinations(preview, 4)
            if appeared.issubset(combo)
        ]
    candidates.sort(key=lambda row: (-row[0], row[1]))
    kept = candidates[:limit]
    total = sum(weight for weight, _brought in kept) or float(len(kept))
    return [(weight / total, brought) for weight, brought in kept]


@dataclass(frozen=True)
class MirrorHypothesis:
    """One complete belief about everything the opponent has not shown us.

    Two independent kinds of hidden information are combined here: Stat Points/nature,
    which are baked into the team a mirror root is STARTED from, and sleep/confusion
    timers, which are patched onto an existing root. `LiveExactMirror.rebase` relies on
    that split -- changing timers re-patches, changing spreads has to rebuild.
    """

    weight: float
    spreads: dict
    timers: HiddenStateHypothesis | None = None
    sets: dict | None = None
    brought: tuple[str, ...] = ()

    @property
    def spread_key(self) -> tuple:
        spread_key = tuple(
            (species_id, tuple(sorted(spread.items())), nature)
            for species_id, (spread, nature) in sorted(self.spreads.items())
        )
        set_key = tuple(
            (
                species_id,
                tuple(values.get("moves", ())),
                values.get("item"),
                values.get("ability"),
            )
            for species_id, values in sorted((self.sets or {}).items())
        )
        return spread_key, set_key, tuple(self.brought)

    @property
    def payload(self) -> dict:
        return self.timers.payload if self.timers is not None else {}


def _hypothesis_distance(left: MirrorHypothesis, right: MirrorHypothesis) -> int:
    """Count hidden-information families that differ between two legal roots."""

    left_spreads, left_sets, left_brought = left.spread_key
    right_spreads, right_sets, right_brought = right.spread_key
    return sum(
        (
            left_spreads != right_spreads,
            left_sets != right_sets,
            left_brought != right_brought,
            left.payload != right.payload,
        )
    )


def _compress_hypotheses(
    hypotheses: list[MirrorHypothesis], limit: int
) -> tuple[list[MirrorHypothesis], float, list[MirrorHypothesis]]:
    """Choose diverse legal representatives and assign all belief mass to them."""

    if len(hypotheses) <= limit:
        return hypotheses, sum(entry.weight for entry in hypotheses), []
    ranked = sorted(hypotheses, key=lambda entry: -entry.weight)
    selected = [ranked[0]]
    while len(selected) < limit:
        remaining = [entry for entry in ranked if entry not in selected]
        chosen = max(
            remaining,
            key=lambda entry: (
                min(_hypothesis_distance(entry, kept) for kept in selected),
                entry.weight,
            ),
        )
        selected.append(chosen)
    selected_ids = {id(entry) for entry in selected}
    excluded = [entry for entry in ranked if id(entry) not in selected_ids]
    direct_mass = sum(entry.weight for entry in selected)
    assigned = [0.0] * len(selected)
    for entry in ranked:
        best_index = min(
            range(len(selected)),
            key=lambda index: (
                _hypothesis_distance(entry, selected[index]),
                -selected[index].weight,
                index,
            ),
        )
        assigned[best_index] += entry.weight
    compressed = [
        MirrorHypothesis(
            weight=assigned[index],
            spreads=entry.spreads,
            timers=entry.timers,
            sets=entry.sets,
            brought=entry.brought,
        )
        for index, entry in enumerate(selected)
    ]
    return compressed, direct_mass, excluded


@dataclass(frozen=True)
class BranchSensitivity:
    """Whether excluded belief branches would have changed the searched top action."""

    excluded_branches_checked: int
    flipping_branches: int
    flipping_mass: float


def _top_order_message(ranking: Sequence) -> str | None:
    if not ranking:
        return None
    return ranking[0].order.message


def branch_sensitivity(
    rank_fn: Callable[[MirrorHypothesis], Sequence],
    representatives: Sequence[MirrorHypothesis],
    excluded: Sequence[MirrorHypothesis],
) -> BranchSensitivity:
    """Compare each excluded hypothesis's top action to the representatives' combined top.

    ``rank_fn`` maps one hypothesis to a ranked list of orders. Live play does not call
    this: searching the dropped branches is an audit cost, not a decision cost.
    """

    if not representatives:
        raise ValueError("branch sensitivity needs at least one representative")
    combined = combine_belief_rankings(
        [(entry.weight, list(rank_fn(entry))) for entry in representatives]
    )
    representative_top = _top_order_message(combined)
    flipping_branches = 0
    flipping_mass = 0.0
    for hypothesis in excluded:
        if _top_order_message(rank_fn(hypothesis)) != representative_top:
            flipping_branches += 1
            flipping_mass += hypothesis.weight
    return BranchSensitivity(
        excluded_branches_checked=len(excluded),
        flipping_branches=flipping_branches,
        flipping_mass=flipping_mass,
    )


@dataclass
class LiveExactMirror:
    """Own one local Showdown worker and rebuild a current-state root per decision."""

    own_packed_team: str
    config: PolicyConfig
    showdown_repo: object = DEFAULT_SHOWDOWN_REPO

    def __post_init__(self) -> None:
        self.worker = SimWorker(self.showdown_repo)
        self._counter = itertools.count()
        self._root_spread_key: tuple | None = None
        self.last_hypothesis_audit: dict[str, object] = {}
        self.last_excluded_hypotheses: list[MirrorHypothesis] = []
        self._last_representatives: list[MirrorHypothesis] = []

    def hypotheses(self, battle, memory=None) -> list[MirrorHypothesis]:
        """Every hidden-information belief for this observation, most likely first.

        The cross product of the opponent's possible Stat Point spreads
        (`exact_search_spread_hypotheses`) and their possible sleep/confusion timers
        (`exact_search_state_hypotheses`). At the shipped defaults this is exactly the
        timer branches, carrying the single most likely spread -- unchanged behavior.

        Passing ``memory`` (a `vgc.battle_memory.BattleMemory`) uses the POSTERIOR spread
        distribution -- the corpus prior reweighted by the speed and damage this battle
        has actually shown -- instead of the flat prior. Omitting it is the pre-battle
        distribution, which is all an offline caller with no history has.

        Ordered spread-major, most likely first within each spread, so :meth:`rebase`
        pays one team-preview start per spread rather than one per branch.
        """

        timers = enumerate_hidden_state_hypotheses(snapshot_battle(battle), self.config)
        spreads = _spread_beliefs(battle, self.config, memory)
        sets = _set_beliefs(battle, self.config, memory)
        brought = _bring_beliefs(battle, self.config)
        combined = [
            MirrorHypothesis(
                weight=spread_weight * set_weight * bring_weight * timer.weight,
                spreads=spread_assignment,
                timers=timer,
                sets=set_assignment,
                brought=bring_assignment,
            )
            for spread_weight, spread_assignment in spreads
            for set_weight, set_assignment in sets
            for bring_weight, bring_assignment in brought
            for timer in timers
        ]
        combined.sort(key=lambda entry: -entry.weight)
        total_count = len(combined)
        limit = int(self.config.exact_search_total_hypotheses)
        if limit < 1:
            raise ValueError("exact_search_total_hypotheses must be at least 1")
        kept, direct_mass, excluded = _compress_hypotheses(combined, limit)
        retained_mass = sum(entry.weight for entry in kept)
        combined = [
            MirrorHypothesis(
                weight=entry.weight / retained_mass,
                spreads=entry.spreads,
                timers=entry.timers,
                sets=entry.sets,
                brought=entry.brought,
            )
            for entry in kept
        ]
        self.last_excluded_hypotheses = list(excluded)
        self.last_hypothesis_audit = {
            "total_before_cap": total_count,
            "searched": len(combined),
            "retained_probability_mass": retained_mass,
            "direct_representative_mass": direct_mass,
            "compressed": total_count > len(combined),
            "spread_branches": len(spreads),
            "set_branches": len(sets),
            "bring_branches": len(brought),
            "timer_branches": len(timers),
        }
        # Spread-major, then most likely first within a spread: a caller stepping through
        # in order rebuilds the Showdown root once per spread rather than once per branch.
        combined.sort(key=lambda entry: (entry.spread_key, -entry.weight))
        self._last_representatives = list(combined)
        return combined

    def record_branch_sensitivity(
        self,
        rank_fn: Callable[[MirrorHypothesis], Sequence],
        representatives: Sequence[MirrorHypothesis] | None = None,
    ) -> BranchSensitivity:
        """Audit excluded branches against searched representatives. Not used in live play."""

        kept = list(self._last_representatives if representatives is None else representatives)
        result = branch_sensitivity(rank_fn, kept, self.last_excluded_hypotheses)
        self.last_hypothesis_audit = {
            **self.last_hypothesis_audit,
            "excluded_branches_checked": result.excluded_branches_checked,
            "flipping_branches": result.flipping_branches,
            "flipping_mass": result.flipping_mass,
        }
        return result

    def build(
        self, battle, hypothesis: MirrorHypothesis | HiddenStateHypothesis | None = None
    ) -> DirectBattle:
        """Rebuild the current public state under one belief about hidden information.

        Omitting ``hypothesis`` leaves the worker on its own conservative default. Use
        :meth:`hypotheses` and pass each belief in turn when the caller wants the hidden
        spreads and privately rolled durations averaged rather than assumed.
        """

        battle_id = f"live-mirror-{next(self._counter)}"
        spreads = getattr(hypothesis, "spreads", None)
        self._root_spread_key = getattr(hypothesis, "spread_key", None)
        opponent_sets = _opponent_sets(
            battle,
            self.config,
            spreads,
            getattr(hypothesis, "sets", None),
            getattr(hypothesis, "brought", None),
        )
        root = DirectBattle.start(
            self.worker,
            battle_id,
            self.own_packed_team,
            opponent_sets,
            seed=[1, 2, 3, 4],
        )
        try:
            root.step(
                {
                    "p1": _team_preview_order(self.own_packed_team, battle),
                    # Opponent sets were ordered with the current active pair first.
                    "p2": "team 1234",
                }
            )
            root.patch_public_state(
                snapshot_battle(battle),
                perspective="p1",
                observation_battle=battle,
                hidden_hypothesis=hypothesis.payload if hypothesis is not None else None,
            )
            return root
        except Exception:
            root.close()
            raise

    def rebase(
        self,
        root: DirectBattle,
        battle,
        hypothesis: MirrorHypothesis | HiddenStateHypothesis,
    ) -> DirectBattle:
        """Move an existing mirror root onto a different belief.

        ``patchPublic`` rewrites the whole battle from the supplied snapshot, so one
        Showdown battle can serve every hidden-TIMER branch instead of paying a fresh
        team-preview start per hypothesis. A different hidden SPREAD cannot be patched
        the same way -- Stat Points are baked into the team the battle started from -- so
        that case closes this root and starts a new one.
        """

        wanted = getattr(hypothesis, "spread_key", None)
        if wanted != self._root_spread_key:
            root.close()
            return self.build(battle, hypothesis)
        root.patch_public_state(
            snapshot_battle(battle),
            perspective="p1",
            observation_battle=battle,
            hidden_hypothesis=hypothesis.payload,
        )
        return root

    def close(self) -> None:
        self.worker.close()
