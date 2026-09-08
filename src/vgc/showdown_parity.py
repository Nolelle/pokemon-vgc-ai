"""Fail-closed check that the local Showdown checkout still matches public master.

Exact mechanics are exact relative to the pinned local checkout, not to whatever
play.pokemonshowdown.com currently deploys. This module fetches the public
smogon/pokemon-showdown remote and reports Champions-relevant commits that HEAD
does not contain.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vgc.config import DATA_DIR, SHOWDOWN_REPO

DEFAULT_PATHS: tuple[str, ...] = (
    "data/mods/champions",
    "config/formats.ts",
    "sim",
    "data/moves.ts",
    "data/abilities.ts",
    "data/items.ts",
    "data/conditions.ts",
    "data/pokedex.ts",
    "data/learnsets.ts",
    "data/typechart.ts",
    "data/natures.ts",
    "data/scripts.ts",
)
CATALOG_PATH = DATA_DIR / "mechanics_catalog.json"
DEFAULT_UPSTREAM_BRANCH = "master"
FETCH_TIMEOUT_SECONDS = 30
_SMOGON_REMOTE_MARKER = "smogon/pokemon-showdown"
_LOG_FORMAT = "%h %s"


@dataclass(frozen=True)
class ParityReport:
    local_head: str
    local_dirty: bool
    pinned_commit: str
    pinned_matches_head: bool
    upstream_ref: str
    fetched: bool
    fetch_error: str | None
    missing_upstream_commits: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.pinned_matches_head
            and not self.local_dirty
            and self.fetch_error is None
            and not self.missing_upstream_commits
        )


def load_pinned_commit(catalog_path: Path | None = None) -> str:
    catalog = json.loads((catalog_path or CATALOG_PATH).read_text())
    return str(catalog["generated_from"]["showdown_commit"])


def format_parity_report(report: ParityReport) -> str:
    if report.missing_upstream_commits:
        missing = "\n".join(f"  {line}" for line in report.missing_upstream_commits)
    else:
        missing = "  (none)"
    return "\n".join(
        [
            f"local_head: {report.local_head}",
            f"local_dirty: {report.local_dirty}",
            f"pinned_commit: {report.pinned_commit}",
            f"pinned_matches_head: {report.pinned_matches_head}",
            f"upstream_ref: {report.upstream_ref}",
            f"fetched: {report.fetched}",
            f"fetch_error: {report.fetch_error}",
            "missing_upstream_commits:",
            missing,
        ]
    )


def check_showdown_parity(
    showdown_repo: Path,
    pinned_commit: str,
    *,
    fetch: bool = True,
    paths: tuple[str, ...] = DEFAULT_PATHS,
) -> ParityReport:
    if not showdown_repo.is_dir():
        return ParityReport(
            local_head="",
            local_dirty=True,
            pinned_commit=pinned_commit,
            pinned_matches_head=False,
            upstream_ref="",
            fetched=False,
            fetch_error=f"showdown repo not found: {showdown_repo}",
            missing_upstream_commits=(),
        )

    local_head = _rev_parse(showdown_repo, "HEAD") or ""
    local_dirty = _working_tree_dirty(showdown_repo)
    pinned_resolved = _rev_parse(showdown_repo, pinned_commit)
    pinned_matches_head = bool(local_head) and pinned_resolved == local_head

    remote_branch = _upstream_remote_and_branch(showdown_repo)
    if remote_branch is None:
        return ParityReport(
            local_head=local_head,
            local_dirty=local_dirty,
            pinned_commit=pinned_commit,
            pinned_matches_head=pinned_matches_head,
            upstream_ref="",
            fetched=False,
            fetch_error="no git remote found",
            missing_upstream_commits=(),
        )
    remote, branch = remote_branch
    upstream_ref = f"{remote}/{branch}"

    fetched = False
    fetch_error: str | None = None
    if fetch:
        fetch_error = _fetch(showdown_repo, remote, branch)
        fetched = fetch_error is None
        if fetch_error is not None:
            return ParityReport(
                local_head=local_head,
                local_dirty=local_dirty,
                pinned_commit=pinned_commit,
                pinned_matches_head=pinned_matches_head,
                upstream_ref=upstream_ref,
                fetched=False,
                fetch_error=fetch_error,
                missing_upstream_commits=(),
            )

    missing, log_error = _missing_upstream_commits(showdown_repo, upstream_ref, paths)
    if log_error is not None:
        fetch_error = log_error
        missing = ()
    return ParityReport(
        local_head=local_head,
        local_dirty=local_dirty,
        pinned_commit=pinned_commit,
        pinned_matches_head=pinned_matches_head,
        upstream_ref=upstream_ref,
        fetched=fetched,
        fetch_error=fetch_error,
        missing_upstream_commits=missing,
    )


def enforce_showdown_parity_for_cli(action: str = "public ladder play") -> None:
    report = check_showdown_parity(SHOWDOWN_REPO, load_pinned_commit(), fetch=True)
    if report.ready:
        return
    reasons: list[str] = []
    if not report.pinned_matches_head:
        reasons.append(
            f"pinned commit {report.pinned_commit} does not match HEAD {report.local_head}"
        )
    if report.local_dirty:
        reasons.append("local Showdown checkout has uncommitted changes")
    if report.fetch_error:
        reasons.append(f"could not fetch {report.upstream_ref}: {report.fetch_error}")
    if report.missing_upstream_commits:
        reasons.append(
            "public smogon/pokemon-showdown is ahead: "
            + "; ".join(report.missing_upstream_commits)
        )
    raise SystemExit(
        f"Blocked {action}: local Showdown is not in parity with the public server. "
        + "; ".join(reasons)
        + ". Run `.venv/bin/python offline/check_showdown_parity.py`."
    )


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _run_git(
    repo: Path,
    args: tuple[str, ...],
    *,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=_git_env(),
    )


def _command_error(result: subprocess.CompletedProcess[str], fallback: str) -> str:
    return (result.stderr or result.stdout or fallback).strip() or fallback


def _rev_parse(repo: Path, rev: str) -> str | None:
    try:
        result = _run_git(repo, ("rev-parse", "--verify", rev))
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _working_tree_dirty(repo: Path) -> bool:
    try:
        result = _run_git(repo, ("status", "--porcelain"))
    except OSError:
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _upstream_remote_and_branch(repo: Path) -> tuple[str, str] | None:
    try:
        result = _run_git(repo, ("remote", "-v"))
    except OSError:
        return None
    if result.returncode != 0:
        return None
    remotes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]
        remotes.setdefault(name, url)
    if not remotes:
        return None
    preferred = next(
        (name for name, url in remotes.items() if _SMOGON_REMOTE_MARKER in url),
        None,
    )
    if preferred is not None:
        return preferred, DEFAULT_UPSTREAM_BRANCH
    if "origin" in remotes:
        return "origin", DEFAULT_UPSTREAM_BRANCH
    return next(iter(remotes)), DEFAULT_UPSTREAM_BRANCH


def _fetch(repo: Path, remote: str, branch: str) -> str | None:
    try:
        result = _run_git(
            repo,
            ("fetch", remote, branch),
            timeout=FETCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"git fetch timed out after {FETCH_TIMEOUT_SECONDS}s"
    except OSError as exc:
        return str(exc)
    if result.returncode != 0:
        return _command_error(result, "git fetch failed")
    return None


def _missing_upstream_commits(
    repo: Path,
    upstream_ref: str,
    paths: tuple[str, ...],
) -> tuple[tuple[str, ...], str | None]:
    try:
        result = _run_git(
            repo,
            (
                "log",
                upstream_ref,
                "--not",
                "HEAD",
                f"--format={_LOG_FORMAT}",
                "--",
                *paths,
            ),
        )
    except OSError as exc:
        return (), str(exc)
    if result.returncode != 0:
        return (), _command_error(result, "git log failed")
    commits = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    return commits, None
