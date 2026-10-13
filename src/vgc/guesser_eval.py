"""Grade opponent guesses against known true sets (the guesser report card).

Pure scoring over ``PokemonBelief`` objects: the caller supplies beliefs built
through the normal fog-safe path plus a truth dict it obtained elsewhere
(exact team files, a simulator's own team in a self-play harness). This module
never reads truth itself, so grading can never leak into guessing.
"""

from __future__ import annotations

import math
from typing import Any

_EPS = 1e-9


def _categorical_metrics(probabilities: dict[str, float], true_value: str | None) -> dict[str, float]:
    if not true_value:
        return {}
    ranked = sorted(probabilities.items(), key=lambda e: (-e[1], e[0]))
    p_true = float(probabilities.get(true_value, 0.0))
    return {
        "top1": 1.0 if ranked and ranked[0][0] == true_value else 0.0,
        "top3": 1.0 if true_value in {k for k, _ in ranked[:3]} else 0.0,
        "logloss": -math.log(max(p_true, _EPS)),
        "brier": sum(
            (prob - (1.0 if value == true_value else 0.0)) ** 2
            for value, prob in probabilities.items()
        )
        + (0.0 if true_value in probabilities else 1.0),
    }


def score_beliefs(beliefs, truth: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics per hidden field over ``beliefs`` vs ``truth``.

    ``truth`` maps species id -> {"moves": iterable, "item": str|None,
    "ability": str|None, "sp": dict|None, "nature": str|None}.
    """

    fields: dict[str, list[dict[str, float]]] = {"item": [], "ability": [], "moves": []}
    spread_exact = []
    n = 0
    for belief in beliefs:
        true = truth.get(belief.species_id)
        if true is None:
            continue
        n += 1
        for field_name, probs in (
            ("item", belief.item_probabilities),
            ("ability", belief.ability_probabilities),
        ):
            metrics = _categorical_metrics(probs, true.get(field_name))
            if metrics:
                fields[field_name].append(metrics)
        true_moves = {str(m) for m in true.get("moves") or []}
        if true_moves:
            hits = sum(1 for m in true_moves if belief.move_probability(m) > 0.0)
            top_moves = {
                m
                for m, _ in sorted(
                    belief.move_probabilities.items(), key=lambda e: (-e[1], e[0])
                )[:4]
            }
            fields["moves"].append(
                {
                    "recall": hits / len(true_moves),
                    "top4_recall": len(true_moves & top_moves) / len(true_moves),
                }
            )
        if true.get("sp") is not None:
            spread_exact.append(
                1.0
                if any(
                    dict(h.sp) == dict(true["sp"]) and h.nature == true.get("nature")
                    for h in belief.hypotheses
                )
                else 0.0
            )

    def mean(rows: list[dict[str, float]], key: str) -> float | None:
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else None

    return {
        "species_scored": n,
        "item": {k: mean(fields["item"], k) for k in ("top1", "top3", "logloss", "brier")},
        "ability": {k: mean(fields["ability"], k) for k in ("top1", "top3", "logloss", "brier")},
        "moves": {k: mean(fields["moves"], k) for k in ("recall", "top4_recall")},
        "spread_in_hypotheses": (sum(spread_exact) / len(spread_exact)) if spread_exact else None,
    }
