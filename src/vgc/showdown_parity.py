"""Fail-closed check that the local Showdown checkout still matches public master.

Exact mechanics are exact relative to the pinned local checkout, not to whatever
play.pokemonshowdown.com currently deploys. This module fetches the public
smogon/pokemon-showdown remote and reports Champions-relevant commits that HEAD
does not contain.

A missing commit blocks only if `vgc.showdown_relevance` cannot prove it irrelevant to
our format (e.g. a `config/formats.ts` edit that touches only other formats). Cleared
commits are still reported, as "behind but not relevant"; anything unclassifiable blocks.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from vgc.config import DATA_DIR, FORMAT_ID, SHOWDOWN_REPO
from vgc.showdown_relevance import (
    classify_commit,
    git_file_reader,
    load_format_context,
    node_parser,
)

DEFAULT_PATHS: tuple[str, ...] = (
    "data/mods/champions",
    "config/formats.ts",
    # Merged into the format list when present (sim/dex-formats.ts).
    "config/custom-formats.ts",
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
    # Base clauses the format's ruleset pulls in (VGC Timer, Species/Item Clause, Open
    # Team Sheets); the champions mod overrides only some rules in its own rulesets.ts.
    "data/rulesets.ts",
    # Flat Rules bans "Mythical" and "Restricted Legendary"; those categories live here.
    "data/tags.ts",
    # Name lookups resolve through aliases before the canonical tables.
    "data/aliases.ts",
    # Base species metadata the Champions mod does not override.
    "data/formats-data.ts",
    # Shared helpers used by the battle engine and team validator.
    "lib",
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
    # Upstream commits on watched paths that were proven irrelevant to our format.
    irrelevant_upstream_commits: tuple[str, ...] = ()
    # (commit line, reason) for every classified commit, blocking or not.
    commit_reasons: tuple[tuple[str, str], ...] = ()

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
    reasons: dict[str, list[str]] = {}
    for line, reason in report.commit_reasons:
        reasons.setdefault(line, []).append(reason)

    def listing(lines: tuple[str, ...]) -> str:
        if not lines:
            return "  (none)"
        out: list[str] = []
        for line in lines:
            out.append(f"  {line}")
            out.extend(f"      - {reason}" for reason in reasons.get(line, ()))
        return "\n".join(out)

    return "\n".join(
        [
            f"local_head: {report.local_head}",
            f"local_dirty: {report.local_dirty}",
            f"pinned_commit: {report.pinned_commit}",
            f"pinned_matches_head: {report.pinned_matches_head}",
            f"upstream_ref: {report.upstream_ref}",
            f"fetched: {report.fetched}",
            f"fetch_error: {report.fetch_error}",
            "missing_upstream_commits (block -- affect Reg M-C or unproven):",
            listing(report.missing_upstream_commits),
            "irrelevant_upstream_commits (behind, but cannot affect Reg M-C):",
            listing(report.irrelevant_upstream_commits),
        ]
    )


def check_showdown_parity(
    showdown_repo: Path,
    pinned_commit: str,
    *,
    fetch: bool = True,
    paths: tuple[str, ...] = DEFAULT_PATHS,
    data_dir: Path = DATA_DIR,
    format_id: str = FORMAT_ID,
    parser_repo: Path | None = None,
) -> ParityReport:
    """`parser_repo` is the checkout whose TypeScript parses changed files (default:
    `showdown_repo` itself; tests point it at a real checkout)."""
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

    commits, log_error = _missing_upstream_commits(showdown_repo, upstream_ref, paths)
    missing: tuple[str, ...] = ()
    irrelevant: tuple[str, ...] = ()
    commit_reasons: tuple[tuple[str, str], ...] = ()
    if log_error is not None:
        fetch_error = log_error
    else:
        missing, irrelevant, commit_reasons = _classify_commits(
            showdown_repo, commits, paths, data_dir, format_id, parser_repo or showdown_repo
        )
    return ParityReport(
        local_head=local_head,
        local_dirty=local_dirty,
        pinned_commit=pinned_commit,
        pinned_matches_head=pinned_matches_head,
        upstream_ref=upstream_ref,
        fetched=fetched,
        fetch_error=fetch_error,
        missing_upstream_commits=missing,
        irrelevant_upstream_commits=irrelevant,
        commit_reasons=commit_reasons,
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


def _classify_commits(
    repo: Path,
    commits: tuple[tuple[str, str], ...],
    paths: tuple[str, ...],
    data_dir: Path,
    format_id: str,
    parser_repo: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Split (sha, line) commits into blocking and provably irrelevant. Fails closed."""
    missing: list[str] = []
    irrelevant: list[str] = []
    reasons: list[tuple[str, str]] = []
    try:
        context = load_format_context(data_dir, format_id)
    except Exception as exc:
        for _sha, line in commits:
            missing.append(line)
            reasons.append((line, f"could not load {data_dir} legality data ({exc})"))
        return tuple(missing), (), tuple(reasons)
    read_file = git_file_reader(repo)
    parse = node_parser(parser_repo)
    for sha, line in commits:
        try:
            result = _run_git(repo, ("diff", "--name-only", f"{sha}^", sha, "--", *paths))
            if result.returncode != 0:
                raise RuntimeError(_command_error(result, "git diff failed"))
            files = [name for name in result.stdout.splitlines() if name.strip()]
            verdict = classify_commit(sha, files, read_file, context, parse)
            relevant, why = verdict.relevant, verdict.reasons
        except Exception as exc:
            relevant, why = True, (f"could not classify ({exc})",)
        (missing if relevant else irrelevant).append(line)
        reasons.extend((line, reason) for reason in why)
    return tuple(missing), tuple(irrelevant), tuple(reasons)


def _missing_upstream_commits(
    repo: Path,
    upstream_ref: str,
    paths: tuple[str, ...],
) -> tuple[tuple[tuple[str, str], ...], str | None]:
    try:
        result = _run_git(
            repo,
            (
                "log",
                upstream_ref,
                "--not",
                "HEAD",
                f"--format=%H%x09{_LOG_FORMAT}",
                "--",
                *paths,
            ),
        )
    except OSError as exc:
        return (), str(exc)
    if result.returncode != 0:
        return (), _command_error(result, "git log failed")
    commits = tuple(
        tuple(line.strip().split("\t", 1))  # (full sha, "short-sha subject")
        for line in result.stdout.splitlines()
        if line.strip()
    )
    return commits, None  # type: ignore[return-value]
