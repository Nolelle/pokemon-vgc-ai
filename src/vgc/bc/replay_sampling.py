"""Training-only, game-controlled replay sampling; no model training or legal choices."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

TASKS = ("move", "target", "switch_destination", "preview_brought", "preview_leads", "outcome")
POLICY_TASKS = tuple(t for t in TASKS if t != "outcome")
METHOD = "equal_game_then_perspective_then_unit_both_role_inverse_sqrt_roster_frequency"
Group = tuple[str, str, str]


@dataclass(frozen=True)
class SamplingPlan:
    phase: str
    method: str
    per_unit_weights: dict[Group, float]
    group_counts: dict[Group, int]
    diagnostics: dict[str, dict]
    normalizers: dict[str, float]
    roster_frequencies: dict[tuple[str, ...], int]


def _known(rating) -> bool:
    return isinstance(rating, (int, float)) and not isinstance(rating, bool) and math.isfinite(rating)


def _eligible(
    game: dict, side: str, task: str, phase: str,
    broad_min_pre_rating: float | None = None,
    broad_outcome_min_pre_rating: float | None = None,
) -> bool:
    """Per-(game, side, task) eligibility.

    Expert: known rating >= 1600, all tasks. Broad policy tasks: no open-sheet
    games (those players chose knowing the opponent's sets, which the ladder bot
    never has) plus the optional `broad_min_pre_rating` floor. Broad outcome:
    `broad_outcome_min_pre_rating` floor only -- pass ``-inf`` for "any known
    finite rating" (all rated games including <1200); None keeps today's
    no-requirement meaning.
    """
    rating = game.get("ratings", {}).get(side)
    if phase == "expert":
        return _known(rating) and rating >= 1600
    if task == "outcome":
        if broad_outcome_min_pre_rating is None:
            return True
        return _known(rating) and rating >= broad_outcome_min_pre_rating
    if game.get("showteam"):
        return False
    if broad_min_pre_rating is None:
        return True
    return _known(rating) and rating >= broad_min_pre_rating


def _rosters(game: dict) -> set[tuple[str, ...]]:
    rosters = game.get("rosters", {})
    result = set()
    for side in ("p1", "p2"):
        roster = rosters.get(side)
        if not isinstance(roster, (list, tuple)) or not roster or not all(
            isinstance(species, str) and species for species in roster
        ):
            raise ValueError("training games require both recorded rosters")
        result.add(tuple(sorted(roster)))
    return result


def build_sampling_plan(
    manifest: Mapping[str, dict], unit_counts: Mapping[Group, int], *, phase: str = "broad",
    broad_min_pre_rating: float | None = None,
    broad_outcome_min_pre_rating: float | None = None,
) -> SamplingPlan:
    """Assign probabilities to eligible units using only fixed training assignments.

    A group's probability is its per-unit weight times its eligible unit count.
    Roster frequencies count each lineup once per game, including mirror matches.
    Eligibility is per (game, side, task) -- see `_eligible`; a game stays in the
    pool if ANY side+task is eligible, so an outcome-only <floor game still
    contributes outcome units. Held-out group counts are ignored entirely.
    """
    if phase not in ("broad", "expert"):
        raise ValueError("phase must be broad or expert")
    games = {
        rid: game for rid, game in manifest.items()
        if game.get("split") == "train"
        and any(
            _eligible(game, side, task, phase, broad_min_pre_rating,
                      broad_outcome_min_pre_rating)
            for side in ("p1", "p2") for task in TASKS
        )
    }
    roster_sets = {rid: _rosters(game) for rid, game in games.items()}
    frequencies = Counter(roster for rosters in roster_sets.values() for roster in rosters)
    counts = {}
    by_task = {task: defaultdict(list) for task in TASKS}
    for key, count in sorted(unit_counts.items()):
        rid, side, task = key
        if rid not in games:
            continue
        if side not in ("p1", "p2") or task not in TASKS:
            raise ValueError(f"invalid training group: {key}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("eligible unit counts must be nonnegative integers")
        if count and _eligible(games[rid], side, task, phase, broad_min_pre_rating,
                               broad_outcome_min_pre_rating):
            counts[key] = count
            by_task[task][rid].append(key)
    weights, diagnostics, normalizers = {}, {}, {}
    for task in TASKS:
        groups = by_task[task]
        game_mass = {
            rid: 1 / math.sqrt(max(frequencies[r] for r in roster_sets[rid]))
            for rid in groups
        }
        normalizer = sum(game_mass.values())
        normalizers[task] = normalizer
        probabilities = {rid: mass / normalizer for rid, mass in game_mass.items()}
        for rid, keys in groups.items():
            for key in keys:
                weights[key] = probabilities[rid] / len(keys) / counts[key]
        before, after, accounts = Counter(), Counter(), Counter()
        for rid, p_game in probabilities.items():
            for roster in roster_sets[rid]:
                before[roster] += 1 / len(groups)
                after[roster] += p_game
            for account in set(games[rid].get("accounts", {}).values()) - {None, ""}:
                accounts[account] += p_game
        diagnostics[task] = {
            "eligible_games": len(groups),
            "eligible_perspectives": sum(map(len, groups.values())),
            "eligible_units": sum(counts[key] for keys in groups.values() for key in keys),
            "max_team_participation_before": max(before.values(), default=0.0),
            "max_team_participation_after": max(after.values(), default=0.0),
            "max_account_share": max(accounts.values(), default=0.0),
            "effective_games": 1 / sum(p * p for p in probabilities.values()) if groups else 0.0,
            "max_game_weight": max(probabilities.values(), default=0.0),
        }
    return SamplingPlan(phase, METHOD, weights, counts, diagnostics, normalizers, dict(frequencies))


def sample_task_units(
    plan: SamplingPlan, task: str, n: int, seed: int,
) -> list[tuple[str, str, str, int]]:
    """Draw with replacement; ordinal indexes the caller's eligible units in a group."""
    if task not in TASKS or not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("valid task and nonnegative integer n required")
    if n == 0:
        return []
    keys = sorted(key for key in plan.group_counts if key[2] == task)
    if not keys:
        raise ValueError(f"no eligible training units for {task}")
    rng = random.Random(seed)
    chosen = rng.choices(keys, weights=[
        plan.per_unit_weights[key] * plan.group_counts[key] for key in keys
    ], k=n)
    return [(*key, rng.randrange(plan.group_counts[key])) for key in chosen]


def pool_task_within_game(plan: SamplingPlan, task: str) -> SamplingPlan:
    """Same per-game mass for `task`, but every eligible unit in a game gets the same weight
    (game mass / total eligible units in that game), so both sides' rows are drawn equally often."""
    game_mass: dict[str, float] = {}
    game_units: dict[str, int] = {}
    for key, count in plan.group_counts.items():
        rid, _side, key_task = key
        if key_task != task:
            continue
        game_mass[rid] = game_mass.get(rid, 0.0) + plan.per_unit_weights[key] * count
        game_units[rid] = game_units.get(rid, 0) + count
    weights = dict(plan.per_unit_weights)
    for key in plan.group_counts:
        rid, _side, key_task = key
        if key_task == task:
            weights[key] = game_mass[rid] / game_units[rid]
    return replace(plan, per_unit_weights=weights)
