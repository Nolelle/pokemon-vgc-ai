"""Whose team is deciding right now, for per-team measured caches.

`vgc.plan_value` and `vgc.speed_payoff` hold measurements for OUR team, looked up by
species + moves from inside `vgc.field_control`. A process-wide table keyed only by species
and moves let two players in one process (every offline A/B: both arms share a worker)
overwrite each other's entries -- 160 colliding identities across the 320 test teams
(Codex review, 2026-10-06) -- so one player could read the other's private set.

Each registered team now gets its own table, keyed by a hash of its packed text, and a
player binds its own team for the duration of each decision (`VgcPlayer._guarded`). Lookups
read only the bound team's table. With nothing bound, a lookup succeeds only when exactly
one team is registered in the process (a single ladder bot, a unit test); otherwise it
finds nothing and the caller falls back to its estimate -- never another team's numbers.
"""

from __future__ import annotations

import contextvars
import hashlib

_CURRENT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vgc_own_team_key", default=None
)


def team_key(packed_team: str) -> str:
    return hashlib.sha1(packed_team.strip().encode()).hexdigest()


def bind_own_team(packed_team: str | None) -> None:
    """Bind ``packed_team`` as the deciding player's team in the current context."""

    _CURRENT.set(team_key(packed_team) if packed_team else None)


def current_team_key() -> str | None:
    return _CURRENT.get()


def resolve_table(tables: dict[str, dict]) -> dict | None:
    """The bound team's table; the only one if exactly one is registered; else None."""

    key = _CURRENT.get()
    if key is not None:
        return tables.get(key)
    if len(tables) == 1:
        return next(iter(tables.values()))
    return None
