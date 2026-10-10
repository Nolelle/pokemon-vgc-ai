"""The audit's documented exemptions, each a PREDICATE on one specific finding.

An exemption never excuses a field or a category. It excuses one mismatch when a named,
demonstrated cause holds for THAT Pokemon at THAT time (a specific move, ability or protocol
message in its history, a specific state of the truth). Anything that satisfies no predicate is
reported as unexplained, so a new defect in an already-excused field still fails the audit.

Every exemption has an id (used as the `[id]` prefix of the group in the report), a one-line
reason, a regex over the finding's `field/kind/detail` key, and the predicate. Tests in
`tests/test_battle_parsing_audit.py` build synthetic findings and assert both that the
documented case is exempt and that near-misses are not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Predicate = Callable[[dict[str, Any], dict[str, Any]], bool]


@dataclass(frozen=True)
class Exemption:
    id: str
    key: str  # regex matched against "field/kind/detail" (fullmatch)
    reason: str
    kind: str  # "hidden" (Showdown does not show it) | "limitation" (shown, not modelled)
    predicate: Predicate


def finding_key(finding: dict[str, Any]) -> str:
    return "/".join(p for p in (finding["field"], finding["kind"], finding["detail"]) if p)


REGISTRY: list[Exemption] = []


def register(*, id: str, key: str, reason: str, kind: str) -> Callable[[Predicate], Predicate]:
    def deco(fn: Predicate) -> Predicate:
        REGISTRY.append(Exemption(id, key, reason, kind, fn))
        return fn

    return deco


def classify(finding: dict[str, Any], cx: dict[str, Any]) -> str | None:
    """The id of the first exemption whose key matches AND whose predicate holds, else None."""
    key = finding_key(finding)
    for ex in REGISTRY:
        if re.fullmatch(ex.key, key) and ex.predicate(finding, cx):
            return ex.id
    return None
