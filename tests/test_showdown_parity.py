from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from vgc.config import DATA_DIR
from vgc.showdown_parity import (
    DEFAULT_PATHS,
    ParityReport,
    check_showdown_parity,
    load_pinned_commit,
)

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "parity-test",
    "GIT_AUTHOR_EMAIL": "parity@example.test",
    "GIT_COMMITTER_NAME": "parity-test",
    "GIT_COMMITTER_EMAIL": "parity@example.test",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=_GIT_ENV,
    )
    if result.returncode != 0:
        raise AssertionError(f"git {args} failed: {result.stderr or result.stdout}")
    return result


def _init_seeded_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-b", "master")
    champions = path / "data" / "mods" / "champions"
    champions.mkdir(parents=True)
    (champions / "base.ts").write_text("export {}\n")
    (path / "config").mkdir()
    (path / "config" / "formats.ts").write_text("export {}\n")
    (path / "sim").mkdir()
    (path / "sim" / "index.ts").write_text("export {}\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-m", "initial champions checkout")


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _push_origin_commit(tmp_path: Path, origin: Path, relative: str, message: str) -> None:
    work = tmp_path / "upstream_work"
    if work.exists():
        raise AssertionError("upstream worktree already exists")
    _git(tmp_path, "clone", str(origin), str(work))
    target = work / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("upstream change\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", message)
    _git(work, "push", "origin", "HEAD:master")


@pytest.fixture
def parity_repos(tmp_path: Path) -> tuple[Path, Path]:
    local = tmp_path / "local"
    origin = tmp_path / "origin.git"
    _init_seeded_repo(local)
    _git(tmp_path, "clone", "--bare", str(local), str(origin))
    _git(local, "remote", "add", "origin", str(origin))
    _git(local, "fetch", "origin")
    return local, origin


def test_load_pinned_commit_matches_catalog() -> None:
    catalog = json.loads((DATA_DIR / "mechanics_catalog.json").read_text())
    assert load_pinned_commit() == catalog["generated_from"]["showdown_commit"]


def test_default_paths_cover_champions_and_inherited_dex() -> None:
    assert DEFAULT_PATHS[0] == "data/mods/champions"
    assert "config/formats.ts" in DEFAULT_PATHS
    assert "sim" in DEFAULT_PATHS
    assert "data/moves.ts" in DEFAULT_PATHS


def test_parity_ok_when_head_matches_pin_and_upstream(parity_repos: tuple[Path, Path]) -> None:
    local, _origin = parity_repos
    head = _head(local)

    report = check_showdown_parity(local, head, fetch=True)

    assert report.ready
    assert report.fetched
    assert report.fetch_error is None
    assert report.pinned_matches_head
    assert not report.local_dirty
    assert report.local_head == head
    assert report.pinned_commit == head
    assert report.upstream_ref == "origin/master"
    assert report.missing_upstream_commits == ()


def test_upstream_champions_commit_is_missing(
    tmp_path: Path, parity_repos: tuple[Path, Path]
) -> None:
    local, origin = parity_repos
    head = _head(local)
    _push_origin_commit(tmp_path, origin, "data/mods/champions/x.ts", "buff incineroar")

    report = check_showdown_parity(local, head, fetch=True)

    assert not report.ready
    assert report.fetched
    assert report.fetch_error is None
    assert report.missing_upstream_commits
    assert any("buff incineroar" in line for line in report.missing_upstream_commits)


def test_unrelated_upstream_commit_is_ignored(
    tmp_path: Path, parity_repos: tuple[Path, Path]
) -> None:
    local, origin = parity_repos
    head = _head(local)
    _push_origin_commit(tmp_path, origin, "docs/readme.txt", "docs only")

    report = check_showdown_parity(local, head, fetch=True)

    assert report.ready
    assert report.missing_upstream_commits == ()


def test_local_dirty_is_not_ready(parity_repos: tuple[Path, Path]) -> None:
    local, _origin = parity_repos
    head = _head(local)
    (local / "config" / "formats.ts").write_text("dirty\n")

    report = check_showdown_parity(local, head, fetch=True)

    assert report.local_dirty
    assert not report.ready


def test_pinned_commit_mismatch_is_not_ready(parity_repos: tuple[Path, Path]) -> None:
    local, _origin = parity_repos
    pinned = _head(local)
    (local / "extra.ts").write_text("local only\n")
    _git(local, "add", "extra.ts")
    _git(local, "commit", "-m", "local ahead of pin")

    report = check_showdown_parity(local, pinned, fetch=True)

    assert not report.pinned_matches_head
    assert report.local_head != pinned
    assert not report.ready


def test_fetch_failure_sets_error(tmp_path: Path, parity_repos: tuple[Path, Path]) -> None:
    local, _origin = parity_repos
    head = _head(local)
    _git(local, "remote", "set-url", "origin", str(tmp_path / "does-not-exist"))

    report = check_showdown_parity(local, head, fetch=True)

    assert report.fetch_error
    assert not report.fetched
    assert not report.ready


def test_ready_requires_clean_pin_fetch_and_no_missing() -> None:
    ok = ParityReport(
        local_head="abc",
        local_dirty=False,
        pinned_commit="abc",
        pinned_matches_head=True,
        upstream_ref="origin/master",
        fetched=True,
        fetch_error=None,
        missing_upstream_commits=(),
    )
    assert ok.ready
    assert not replace(ok, local_dirty=True).ready
    assert not replace(ok, pinned_matches_head=False).ready
    assert not replace(ok, fetch_error="timeout").ready
    assert not replace(ok, missing_upstream_commits=("abc subject",)).ready
