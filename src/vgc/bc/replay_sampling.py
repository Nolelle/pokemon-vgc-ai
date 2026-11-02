"""Training-only, game-controlled replay sampling; no model training or legal choices."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping

TASKS = ("move", "target", "switch_destination", "preview_brought", "preview_leads", "outcome")
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


def _eligible(game: dict, side: str, phase: str) -> bool:
    rating = game.get("ratings", {}).get(side)
    return phase == "broad" or (
        isinstance(rating, (int, float)) and not isinstance(rating, bool)
        and math.isfinite(rating) and rating >= 1600
    )


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
) -> SamplingPlan:
    """Assign probabilities to eligible units using only fixed training assignments.

    A group's probability is its per-unit weight times its eligible unit count.
    Roster frequencies count each lineup once per game, including mirror matches.
    Unknown expert ratings are excluded. Held-out group counts are ignored entirely.
    """
    if phase not in ("broad", "expert"):
        raise ValueError("phase must be broad or expert")
    games = {
        rid: game for rid, game in manifest.items()
        if game.get("split") == "train"
        and any(_eligible(game, side, phase) for side in ("p1", "p2"))
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
        if count and _eligible(games[rid], side, phase):
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
