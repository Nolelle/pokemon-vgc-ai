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


# --- relevance filter: clear only provably-unrelated commits, fail closed otherwise ---
# These parse with the real Showdown checkout's TypeScript, so they run with the engine.

_OUR_ENTRY = (
    "\t{\n\t\tname: \"[Gen 9 Champions] VGC 2026 Reg M-C\",\n\t\tmod: 'champions',\n"
    "\t\truleset: ['Flat Rules', 'VGC Timer'],\n\t},\n"
)
_RULESETS = (
    "export const Rulesets = {\n\tflatrules: {\n\t\tname: 'Flat Rules',\n\t},\n"
    "\tvgctimer: {\n\t\tname: 'VGC Timer',\n\t},\n\tspeciesclause: {\n\t\tname: 'x',\n\t},\n};\n"
)


# A neighbour on a non-base mod, so a commit can delete the mod under an unchanged entry.
_MOD_ENTRY = "\t{\n\t\tname: \"[Gen 8] Kept\",\n\t\tmod: 'gen8',\n\t},\n"


def _formats(other_ruleset: str = "'Standard'", other_name: str = "[Gen 9] Other") -> str:
    return (
        "export const Formats = [\n"
        + _OUR_ENTRY
        + f'\t{{\n\t\tname: "{other_name}",\n\t\truleset: [{other_ruleset}],\n\t}},\n'
        + _MOD_ENTRY
        + "];\n"
    )


def _aliases(*lines: str) -> str:
    # Showdown's alias loader also iterates CompoundWordNames; without it nothing loads.
    return (
        "export const Aliases = {\n"
        + "".join(f"\t{line}\n" for line in lines)
        + "};\nexport const CompoundWordNames: string[] = [];\n"
    )


_BASE_ALIASES = ('randbats: "[Gen 9] Random Battle",', '/* protect: "No Such Move", */')


def _commit_upstream(
    tmp_path: Path, origin: Path, files: dict[str, str | None], message: str
) -> None:
    """Write each file, or delete it when its text is None."""
    work = tmp_path / "upstream_work"
    _git(tmp_path, "clone", str(origin), str(work))
    for relative, text in files.items():
        target = work / relative
        if text is None:
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    _git(work, "add", "-A")
    _git(work, "commit", "-m", message)
    _git(work, "push", "origin", "HEAD:master")


@pytest.fixture
def format_repos(tmp_path: Path) -> tuple[Path, Path]:
    from vgc.config import SHOWDOWN_REPO

    if not (SHOWDOWN_REPO / "node_modules" / "typescript").is_dir():
        pytest.skip(f"no Showdown checkout with TypeScript at {SHOWDOWN_REPO}")
    if not (SHOWDOWN_REPO / "dist" / "server" / "rooms.js").is_file():
        pytest.skip(f"Showdown checkout at {SHOWDOWN_REPO} is not built")
    local = tmp_path / "local"
    seed = {
        "config/formats.ts": _formats(),
        "data/aliases.ts": _aliases(*_BASE_ALIASES),
        "data/rulesets.ts": _RULESETS,
        "data/mods/champions/rulesets.ts": "export const Rulesets = {\n\tvgctimer: {},\n};\n",
        "data/mods/gen8/scripts.ts": "export const Scripts = {};\n",
        "data/moves.ts": "export const Moves = {\n\tprotect: {\n\t\tpriority: 4,\n\t},\n};\n",
    }
    for relative, text in seed.items():
        (local / relative).parent.mkdir(parents=True, exist_ok=True)
        (local / relative).write_text(text)
    _git(local, "init", "-b", "master")
    _git(local, "add", "-A")
    _git(local, "commit", "-m", "seed")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "--bare", str(local), str(origin))
    _git(local, "remote", "add", "origin", str(origin))
    _git(local, "fetch", "origin")
    return local, origin


def _check(local: Path) -> ParityReport:
    from vgc.config import SHOWDOWN_REPO

    return check_showdown_parity(local, _head(local), fetch=True, parser_repo=SHOWDOWN_REPO)


_HARMLESS_CHANGES = {
    "october rotation": {
        "config/formats.ts": _formats("'Standard', 'Dynamax Clause'"),
        "data/aliases.ts": _aliases(*_BASE_ALIASES, 'omotm: "[Gen 9] Bad n\' Boosted",'),
    },
    # Mod existence is judged at the commit, not against the pinned build.
    "new mod with its format": {
        "data/mods/reviewnewmod/scripts.ts": "export const Scripts = {};\n",
        "config/formats.ts": _formats().replace(
            "];", '\t{\n\t\tname: "[Gen 9] New Mod",\n\t\tmod: "reviewnewmod",\n\t},\n];'
        ),
    },
    # The server never builds a hidden format's rule table, so it cannot break the list.
    "hidden neighbour with a bad rule": {
        "config/formats.ts": _formats("'No Such Review Rule'").replace(
            '\t\tname: "[Gen 9] Other",',
            '\t\tname: "[Gen 9] Other",\n'
            "\t\tchallengeShow: false,\n\t\tsearchShow: false,\n\t\ttournamentShow: false,",
        )
    },
}


@pytest.mark.integration
@pytest.mark.parametrize("message", sorted(_HARMLESS_CHANGES))
def test_other_format_and_alias_edits_do_not_block(
    tmp_path: Path, format_repos: tuple[Path, Path], message: str
) -> None:
    local, origin = format_repos
    _commit_upstream(tmp_path, origin, _HARMLESS_CHANGES[message], message)

    report = _check(local)

    assert report.ready, report.commit_reasons
    assert any(message in line for line in report.irrelevant_upstream_commits)


_FAKE_BOUNDARY = _formats().replace(
    "export const Formats = [\n",
    "const flat = 'Flat Rules';\nexport const Formats = [\n",
)

_BLOCKING_CHANGES = {
    "our entry": {"config/formats.ts": _formats().replace("'VGC Timer'", "'X'")},
    "new format named like a rule": {"config/formats.ts": _formats(other_name="Species Clause")},
    "duplicate of our format": {
        "config/formats.ts": _formats(other_name="Gen 9 Champions VGC 2026 Reg MC")
    },
    "code outside the format list": {"config/formats.ts": _FAKE_BOUNDARY},
    "unparseable formats": {"config/formats.ts": _formats().replace("];", "")},
    "alias onto a legal move": {
        "data/aliases.ts": _aliases(*_BASE_ALIASES, 'pp: "Protect",'),
    },
    "escaped legal alias key": {
        "data/aliases.ts": _aliases(*_BASE_ALIASES, '"\\x70rotect": "No Such Move",'),
    },
    "alias onto a rule name": {
        "data/aliases.ts": _aliases(*_BASE_ALIASES, 'flatrules: "Team Preview",'),
    },
    "comment edit activating an alias": {
        "data/aliases.ts": _aliases(_BASE_ALIASES[0], 'protect: "No Such Move",'),
    },
    "spread hiding a rule name": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",',
            '\t\tname: "[Gen 9] Other",\n\t\t...{name: "Species Clause"},',
        )
    },
    "load-time code in another format": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",', '\t\tname: "[Gen 9] Other",\n\t\tdesc: (() => "x")(),'
        )
    },
    "other format with a missing mod": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",', '\t\tname: "[Gen 9] Other",\n\t\tmod: "nonexistent",'
        )
    },
    "duplicate unrelated names": {
        "config/formats.ts": _formats(other_name="[Gen 9] Other").replace(
            "];", '\t{\n\t\tname: "[Gen 9] Other",\n\t},\n];'
        )
    },
    # Parse cleanly but Showdown's loader throws on them, which breaks our format too.
    "neighbour name with no alphanumerics": {"config/formats.ts": _formats(other_name="!!!")},
    "neighbour with mod null": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",', '\t\tname: "[Gen 9] Other",\n\t\tmod: null,'
        )
    },
    "neighbour with deprecated maxLevel": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",', '\t\tname: "[Gen 9] Other",\n\t\tmaxLevel: 100,'
        )
    },
    "neighbour with an undefined identifier": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",', '\t\tname: "[Gen 9] Other",\n\t\tdesc: notDefined,'
        )
    },
    "neighbour rule table rejected": {
        "config/formats.ts": _formats("'Standard', 'Max Team Size = 30'")
    },
    # The server's format-list text cannot stringify a null-prototype object.
    "neighbour section that is not text": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",',
            '\t\tname: "[Gen 9] Other",\n\t\tsection: {__proto__: null},',
        )
    },
    # The commit deletes a mod that an UNCHANGED neighbour still uses.
    "neighbour whose mod the commit deletes": {
        "data/mods/gen8/scripts.ts": None,
        "config/formats.ts": _formats(other_name="[Gen 9] Other Renamed"),
    },
    # Resolves against the real (pinned) aliases, but not the commit's own.
    "neighbour rule through a retargeted alias": {
        "config/formats.ts": _formats("'randbats'").replace(
            "];", '\t{\n\t\tname: "[Gen 9] Random Battle",\n\t},\n];'
        ),
        "data/aliases.ts": _aliases('randbats: "[Gen 9] Gone",', _BASE_ALIASES[1]),
    },
    # Showdown's esbuild build (useDefineForClassFields: false) keeps A's value; a
    # plain TypeScript transpile would let B's field declaration erase it to "gen9".
    "neighbour compiled as Showdown builds it": {
        "config/formats.ts": _formats().replace(
            '\t\tname: "[Gen 9] Other",',
            '\t\tname: "[Gen 9] Other",\n\t\tmod: {toString() {\n'
            "\t\t\tclass A { value = 'notamod'; }\n"
            "\t\t\tclass B extends A { value: string; }\n"
            "\t\t\treturn new B().value || 'gen9';\n\t\t}} as any,",
        )
    },
    "shadowed alias initializer": {
        "data/aliases.ts": _aliases(*_BASE_ALIASES, "x: String(1),", 'x: "y",'),
    },
    "dex entry": {
        "data/moves.ts": "export const Moves = {\n\tprotect: {\n\t\tpriority: 3,\n\t},\n};\n"
    },
}


@pytest.mark.integration
@pytest.mark.parametrize("message", sorted(_BLOCKING_CHANGES))
def test_changes_that_may_reach_our_format_block(
    tmp_path: Path, format_repos: tuple[Path, Path], message: str
) -> None:
    local, origin = format_repos
    _commit_upstream(tmp_path, origin, _BLOCKING_CHANGES[message], message)

    report = _check(local)

    assert not report.ready
    assert any(message in line for line in report.missing_upstream_commits), report.commit_reasons
    if message.startswith("neighbour"):
        assert "Showdown cannot load the format list" in str(report.commit_reasons)
