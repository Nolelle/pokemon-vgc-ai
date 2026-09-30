"""Bring the local Showdown checkout and the Champions rulebook up to date before a run.

`vgc.showdown_parity` only *detects* drift and blocks. This module performs the manual
update chain that used to follow a block:

1. fast-forward the Showdown checkout to public master and force-rebuild ``dist/sim``
   (an unforced ``node build`` can leave a stale ``dist/sim``);
2. re-export ``data/champions/*.json`` and the mechanics catalogue;
3. re-pin ``mechanics_coverage.json``'s ``catalog_sha256``;
4. commit the refreshed data (the readiness gates refuse to certify a dirty tree);
5. run the mechanics, battle-state and action readiness gates.

It stops (``SyncBlocked``) instead of guessing whenever a person has to decide:

- either checkout has uncommitted changes, or Showdown is not on ``master``;
- the fetch fails (no network means parity cannot be established);
- the mechanics catalogue changed in anything but its recorded commit. Per
  ``docs/champions_mechanics_catalog.md`` a catalogue change invalidates the coverage gate
  until the new callbacks/fields/entities are reviewed. The exported files are reverted
  and the changed catalogue paths are listed; rerun with ``accept_catalog_changes=True``
  after reviewing them;
- any readiness gate fails afterwards.

Parity still means "matches public smogon/pokemon-showdown master", not "matches what
play.pokemonshowdown.com has deployed this minute" (see `vgc.showdown_parity`).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vgc.action_gate import action_readiness
from vgc.battle_state_gate import battle_state_readiness
from vgc.config import DATA_DIR, REPO_ROOT, SHOWDOWN_REPO
from vgc.mechanics_gate import mechanics_readiness
from vgc.node import find_node
from vgc.showdown_parity import (
    CATALOG_PATH,
    DEFAULT_UPSTREAM_BRANCH,
    check_showdown_parity,
    load_pinned_commit,
)

COVERAGE_PATH = DATA_DIR / "mechanics_coverage.json"
READINESS_SCRIPTS: tuple[str, ...] = (
    "offline/check_mechanics_readiness.py",
    "offline/check_battle_state_readiness.py",
    "offline/check_action_readiness.py",
)
BUILD_TIMEOUT_SECONDS = 900
GATE_TIMEOUT_SECONDS = 1800
MAX_LISTED_CHANGES = 25
# Catalogue fields allowed to change without review: where and from which commit it was
# exported. Neither is a mechanic.
_VOLATILE_CATALOG_PATHS = frozenset(
    {("generated_from", "showdown_commit"), ("generated_from", "showdown_repo")}
)
_CATALOG_SHA_RE = re.compile(r'("catalog_sha256":\s*")[^"]*(")')


class SyncBlocked(RuntimeError):
    """The update needs a person; nothing was left half-applied in this repo."""


@dataclass(frozen=True)
class SyncResult:
    status: str  # "already_current" or "synced"
    showdown_commit: str
    repo_commit: str | None = None
    upstream_commits: tuple[str, ...] = ()


def catalog_changes(old: object, new: object, path: tuple[str, ...] = ()) -> list[str]:
    """Paths where two catalogues differ, ignoring the recorded Showdown commit."""

    if path in _VOLATILE_CATALOG_PATHS:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[str] = []
        for key in sorted(set(old) | set(new), key=str):
            if key not in old:
                changes.append("added " + "/".join((*path, str(key))))
            elif key not in new:
                changes.append("removed " + "/".join((*path, str(key))))
            else:
                changes.extend(catalog_changes(old[key], new[key], (*path, str(key))))
        return changes
    if isinstance(old, list) and isinstance(new, list) and old != new:
        label = "/".join(path)
        added = [item for item in new if item not in old]
        removed = [item for item in old if item not in new]
        if not added and not removed:
            return [f"reordered {label}"]
        return [f"added {label}: {item!r}" for item in added] + [
            f"removed {label}: {item!r}" for item in removed
        ]
    if old != new:
        return [f"changed {'/'.join(path)}: {old!r} -> {new!r}"]
    return []


def repin_catalog_hash(coverage_path: Path, catalog_path: Path) -> str:
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    text = coverage_path.read_text()
    updated, count = _CATALOG_SHA_RE.subn(rf"\g<1>{digest}\g<2>", text, count=1)
    if count != 1:
        raise SyncBlocked(f"no catalog_sha256 field found in {coverage_path}")
    coverage_path.write_text(updated)
    return digest


def sync_showdown(
    *,
    accept_catalog_changes: bool = False,
    run_gates: bool = True,
    showdown_repo: Path = SHOWDOWN_REPO,
    repo_root: Path = REPO_ROOT,
    log: Callable[[str], None] = print,
) -> SyncResult:
    data_dir = repo_root / DATA_DIR.relative_to(REPO_ROOT)
    catalog_path = repo_root / CATALOG_PATH.relative_to(REPO_ROOT)
    coverage_path = repo_root / COVERAGE_PATH.relative_to(REPO_ROOT)

    report = check_showdown_parity(showdown_repo, load_pinned_commit(catalog_path))
    if report.ready:
        log(f"showdown: already in parity at {report.local_head[:9]}")
        # A previous sync may have committed the refresh and then failed a gate.
        if run_gates and not _gates_current():
            _run_gates(repo_root, log)
        return SyncResult("already_current", report.local_head)
    if report.fetch_error:
        raise SyncBlocked(f"cannot check public Showdown: {report.fetch_error}")
    if report.local_dirty:
        raise SyncBlocked(f"{showdown_repo} has uncommitted changes; not touching it")
    if _git(repo_root, "status", "--porcelain"):
        raise SyncBlocked(
            f"{repo_root} has uncommitted changes; commit them first so the data refresh "
            "lands as its own commit and the gates can certify a clean tree"
        )
    branch = _git(showdown_repo, "branch", "--show-current")
    if branch != DEFAULT_UPSTREAM_BRANCH:
        raise SyncBlocked(f"{showdown_repo} is on {branch or 'a detached HEAD'}, not master")

    for line in report.missing_upstream_commits:
        log(f"showdown: upstream has {line}")
    # Fast-forward to exactly the ref the parity check fetched and compared against, so a
    # fork configured as `origin` cannot slip in commits that were never checked.
    log(f"showdown: fast-forwarding to {report.upstream_ref}")
    _run(["git", "merge", "--ff-only", report.upstream_ref], showdown_repo)
    node = find_node()
    log("showdown: force-rebuilding dist/sim")
    _run([node, "build", "--force"], showdown_repo, timeout=BUILD_TIMEOUT_SECONDS)
    head = _git(showdown_repo, "rev-parse", "HEAD")

    old_catalog = json.loads(catalog_path.read_text())
    try:
        repo_commit = _refresh_data(
            repo_root, data_dir, catalog_path, coverage_path, showdown_repo, node, head,
            old_catalog, report.missing_upstream_commits, accept_catalog_changes, log,
        )
    except BaseException:
        # Never leave data/champions half-exported: restore the committed files.
        _run(["git", "checkout", "--", str(data_dir)], repo_root)
        raise

    if run_gates:
        _run_gates(repo_root, log)

    final = check_showdown_parity(showdown_repo, load_pinned_commit(catalog_path))
    if not final.ready:
        raise SyncBlocked(f"still not in parity after the update: {final}")
    log(f"showdown: in parity at {head[:9]}")
    return SyncResult("synced", head, repo_commit, report.missing_upstream_commits)


def _refresh_data(
    repo_root: Path,
    data_dir: Path,
    catalog_path: Path,
    coverage_path: Path,
    showdown_repo: Path,
    node: str,
    head: str,
    old_catalog: dict,
    upstream_commits: tuple[str, ...],
    accept_catalog_changes: bool,
    log: Callable[[str], None],
) -> str | None:
    log("data: re-exporting data/champions")
    _run(
        [node, str(repo_root / "tools/export_champions_data.mjs"), str(showdown_repo),
         str(data_dir)],
        repo_root,
    )
    _run(
        [node, str(repo_root / "tools/export_mechanics_catalog.mjs"), str(showdown_repo),
         str(catalog_path)],
        repo_root,
    )
    changes = catalog_changes(old_catalog, json.loads(catalog_path.read_text()))
    if changes and not accept_catalog_changes:
        listed = "\n".join(f"  {c}" for c in changes[:MAX_LISTED_CHANGES])
        more = len(changes) - MAX_LISTED_CHANGES
        if more > 0:
            listed += f"\n  ... and {more} more"
        raise SyncBlocked(
            f"Showdown is now at {head[:9]}, but the mechanics catalogue changed shape:\n"
            f"{listed}\n"
            "Review these against data/champions/mechanics_coverage.json, then rerun with "
            "--accept-catalog-changes. Exported data was reverted; ladder play stays blocked."
        )
    for change in changes:
        log(f"data: accepted catalogue change: {change}")
    repin_catalog_hash(coverage_path, catalog_path)

    if not _git(repo_root, "status", "--porcelain", "--", str(data_dir)):
        return None
    _git(repo_root, "add", "--", str(data_dir))
    _git(
        repo_root, "commit", "-q", "-m",
        f"Pin Showdown to {head[:9]} and refresh Champions data (automatic sync)\n\n"
        + "\n".join(upstream_commits),
    )
    repo_commit = _git(repo_root, "rev-parse", "HEAD")
    log(f"data: committed refresh as {repo_commit[:9]}")
    return repo_commit


def _gates_current() -> bool:
    try:
        return (
            mechanics_readiness().ready
            and battle_state_readiness().ready
            and action_readiness().ready
        )
    except Exception:
        return False


def _run_gates(repo_root: Path, log: Callable[[str], None]) -> None:
    for script in READINESS_SCRIPTS:
        log(f"gates: running {script}")
        result = subprocess.run(
            [sys.executable, script], cwd=repo_root, capture_output=True, text=True,
            timeout=GATE_TIMEOUT_SECONDS, check=False,
        )
        if result.returncode != 0:
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-15:])
            raise SyncBlocked(f"{script} did not pass after the update:\n{tail}")


def _git(repo: Path, *args: str) -> str:
    return _run(["git", *args], repo).strip()


def _run(cmd: list[str], cwd: Path, *, timeout: float | None = None) -> str:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        raise SyncBlocked(f"`{' '.join(cmd)}` failed in {cwd}:\n{output}")
    return result.stdout
