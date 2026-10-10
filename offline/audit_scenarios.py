"""Targeted scenario battles (placeholder, filled in by the coverage work)."""

from __future__ import annotations

from typing import Any


def make_jobs(workers: int, *, disable: list[str], name_filter: str = "") -> list[dict[str, Any]]:
    return []


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError
