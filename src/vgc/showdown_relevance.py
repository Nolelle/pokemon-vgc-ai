"""Decide whether an upstream Showdown commit can affect our format.

`vgc.showdown_parity` lists every upstream commit that touches a watched path. Public
Showdown changes those shared files almost daily (`config/formats.ts` for every format
rotation, `data/aliases.ts` for format nicknames), so a whole-file rule blocks on changes
that cannot reach a Reg M-C battle. This module clears a commit only when every watched
file it touches is PROVABLY irrelevant, and fails closed on anything else.

`config/formats.ts` and `data/aliases.ts` are read with the TypeScript parser from the
Showdown checkout (`tools/parse_showdown_entries.mjs`), not line-matched, so comments,
template strings and string escapes cannot disguise a change:

- `config/formats.ts`: cleared when the file outside the `Formats` array is identical,
  our format appears exactly once with an identical entry, and every changed entry has a
  plain string name that is neither our format's id nor any rule's id (formats and rules
  share one lookup namespace). Other formats' entries may change freely, but only if
  Showdown can still load the whole list: the new file is then handed to the pinned
  checkout's BUILT code (`Dex.formats.all()`, then the server's own format-list getter,
  which builds every visible format's rule table), with the commit's aliases, and any
  error blocks. A broken neighbour breaks our format too, and Showdown's own code is the
  only complete list of what it rejects.
- `data/aliases.ts`: cleared when the file outside the `Aliases` object is identical and
  every alias whose effective value changed has a string target, and neither key nor old
  or new target is a Reg M-C species/move/item/ability, our format, or any rule
  (rule and format names resolve through aliases).
- Every other watched path blocks. For the dex tables (`data/moves.ts` etc.) the reason
  names the changed entries and which of them are in our exported data, so a block says
  WHY -- but entry-level clearing is deliberately not automatic: today's legal lists
  cannot see a new entry or an entry that becomes legal upstream.

Anything that fails to read, parse or diff blocks with "could not classify".
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

FORMATS_FILE = "config/formats.ts"
ALIASES_FILE = "data/aliases.ts"
BASE_MOD = "gen9"  # sim/dex.ts: the base data directory, not a data/mods/ folder
RULESET_FILES = ("data/rulesets.ts", "data/mods/champions/rulesets.ts")
PARSER_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "parse_showdown_entries.mjs"
PARSER_TIMEOUT_SECONDS = 120

# Dex tables whose top-level entries can be named in a block reason, and which exported
# file decides whether a named entry is in our format.
_DEX_TABLES: dict[str, str] = {
    "data/moves.ts": "moves",
    "data/abilities.ts": "abilities",
    "data/items.ts": "items",
    "data/pokedex.ts": "species",
    "data/learnsets.ts": "species",
    "data/formats-data.ts": "species",
}
_MAX_NAMED_ENTRIES = 8

_DEX_ENTRY_START = re.compile(r"^\t(?:\"([^\"]+)\"|([A-Za-z0-9_]+)): \{$")
_ENTRY_END = "\t},"

# (jobs) -> results; see tools/parse_showdown_entries.mjs for the JSON contract.
Parser = Callable[[list[dict]], list[dict]]
FileReader = Callable[[str, str], "str | None"]


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass(frozen=True)
class CommitVerdict:
    relevant: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FormatContext:
    """What 'affects us' means: our format id and our exported legal ids."""

    format_id: str
    legal: dict[str, frozenset[str]]

    @property
    def all_legal_ids(self) -> frozenset[str]:
        return frozenset().union(*self.legal.values())


def load_format_context(data_dir: Path, format_id: str) -> FormatContext:
    # Deliberately uncached: a re-export in the same process must be seen immediately.
    species = json.loads((data_dir / "species.json").read_text())
    moves = json.loads((data_dir / "moves.json").read_text())
    items = json.loads((data_dir / "items.json").read_text())
    legal_species = {k: v for k, v in species.items() if v.get("isNonstandard") is None}
    abilities = {
        to_id(name) for v in legal_species.values() for name in v.get("abilities", {}).values()
    }
    return FormatContext(
        format_id=format_id,
        legal={
            "species": frozenset(legal_species),
            "moves": frozenset(k for k, v in moves.items() if v.get("isNonstandard") is None),
            "items": frozenset(items),
            "abilities": frozenset(abilities),
        },
    )


def node_parser(showdown_repo: Path, node: str | None = None) -> Parser:
    """Run tools/parse_showdown_entries.mjs with the checkout's own TypeScript."""

    def parse(jobs: list[dict]) -> list[dict]:
        if node is None:
            from vgc.node import find_node

            executable = find_node()
        else:
            executable = node
        # A loadformats job evaluates upstream files nobody has reviewed yet: give it no
        # environment (no inherited credentials) and only read access to the checkout
        # and the script. Damage limitation, not a security boundary -- see the script.
        sandbox: list[str] = []
        env: dict[str, str] | None = None
        if any(job["kind"] == "loadformats" for job in jobs):
            sandbox = [
                "--permission",
                f"--allow-fs-read={Path(showdown_repo).resolve()}",
                f"--allow-fs-read={PARSER_SCRIPT.parent}",
            ]
            env = {}
        result = subprocess.run(
            [executable, *sandbox, str(PARSER_SCRIPT), str(showdown_repo)],
            input=json.dumps(jobs),
            capture_output=True,
            text=True,
            check=False,
            timeout=PARSER_TIMEOUT_SECONDS,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"parser failed: {(result.stderr or result.stdout).strip()}")
        results = json.loads(result.stdout)
        if len(results) != len(jobs):
            raise RuntimeError("parser returned the wrong number of results")
        return results

    return parse


def classify_commit(
    sha: str,
    changed_files: Iterable[str],
    read_file: FileReader,
    context: FormatContext,
    parse: Parser,
) -> CommitVerdict:
    """Classify one commit from the watched files it changed.

    `read_file(rev, path)` returns the file at a revision, or None if it is absent there.
    """
    files = tuple(changed_files)
    if not files:
        return CommitVerdict(True, ("could not classify: no changed watched files listed",))
    parsed: dict[tuple[str, str], dict] = {}
    rule_ids: frozenset[str] = frozenset()
    if FORMATS_FILE in files or ALIASES_FILE in files:
        try:
            parsed = _parse_revisions(sha, files, read_file, parse)
            rule_ids = _rule_ids(parsed)
        except Exception as exc:
            return CommitVerdict(True, (f"could not classify ({exc})",))
    reasons: list[str] = []
    relevant = False
    for path in files:
        try:
            if path == FORMATS_FILE:
                file_relevant, reason = _classify_formats(
                    parsed[("before", path)],
                    parsed[("after", path)],
                    context,
                    rule_ids,
                    lambda mod: mod == BASE_MOD or read_file(sha, f"data/mods/{mod}") is not None,
                )
                if not file_relevant:
                    file_relevant, reason = _check_formats_load(sha, read_file, parse, reason)
            elif path == ALIASES_FILE:
                file_relevant, reason = _classify_aliases(
                    parsed[("before", path)], parsed[("after", path)], context, rule_ids
                )
            elif path in _DEX_TABLES:
                before = read_file(f"{sha}^", path)
                after = read_file(sha, path)
                file_relevant, reason = True, _describe_dex_change(path, before, after, context)
            else:
                file_relevant = True
                reason = f"{path}: always relevant (engine, Champions mod, or shared rules)"
        except Exception as exc:  # fail closed on anything unexpected
            file_relevant, reason = True, f"{path}: could not classify ({exc})"
        relevant = relevant or file_relevant
        reasons.append(reason)
    return CommitVerdict(relevant, tuple(reasons))


def _parse_revisions(
    sha: str, files: tuple[str, ...], read_file: FileReader, parse: Parser
) -> dict[tuple[str, str], dict]:
    """Parse rulesets (always) plus formats/aliases (if touched) at both revisions."""
    wanted: list[tuple[str, str, str]] = []  # (side, path, kind)
    for side in ("before", "after"):
        for path in RULESET_FILES:
            wanted.append((side, path, "rulesets"))
        if FORMATS_FILE in files:
            wanted.append((side, FORMATS_FILE, "formats"))
        if ALIASES_FILE in files:
            wanted.append((side, ALIASES_FILE, "aliases"))
    jobs: list[dict] = []
    for side, path, kind in wanted:
        text = read_file(f"{sha}^" if side == "before" else sha, path)
        if text is None:
            raise RuntimeError(f"{path} missing on the {side} side")
        jobs.append({"kind": kind, "text": text})
    out: dict[tuple[str, str], dict] = {}
    for (side, path, _kind), result in zip(wanted, parse(jobs), strict=True):
        if not result.get("ok"):
            raise RuntimeError(f"{path} ({side}): {result.get('error')}")
        out[(side, path)] = result
    return out


def _rule_ids(parsed: dict[tuple[str, str], dict]) -> frozenset[str]:
    ids: set[str] = set()
    for side in ("before", "after"):
        for path in RULESET_FILES:
            ids.update(to_id(key) for key in parsed[(side, path)]["keys"])
    if not ids:
        raise RuntimeError("no rules parsed")
    return frozenset(ids)


# --- config/formats.ts -------------------------------------------------------------


def _classify_formats(
    old: dict,
    new: dict,
    context: FormatContext,
    rule_ids: frozenset[str],
    mod_exists: Callable[[str], bool],
) -> tuple[bool, str]:
    if old["skeleton"] != new["skeleton"]:
        return True, f"{FORMATS_FILE}: code outside the format list changed"
    ours_old = [e for e in old["elements"] if _entry_id(e) == context.format_id]
    ours_new = [e for e in new["elements"] if _entry_id(e) == context.format_id]
    if len(ours_old) != 1 or len(ours_new) != 1:
        return True, f"{FORMATS_FILE}: our format entry not found exactly once"
    if ours_old[0]["text"] != ours_new[0]["text"]:
        return True, f"{FORMATS_FILE}: our format entry changed"
    old_texts = {e["text"] for e in old["elements"]}
    new_texts = {e["text"] for e in new["elements"]}
    changed = [
        e
        for e in old["elements"] + new["elements"]
        if (e["text"] in old_texts) != (e["text"] in new_texts)
    ]
    # Showdown loads every format together, so a broken neighbour (duplicate id, unknown
    # mod) stops ours loading too.
    new_ids = [_entry_id(e) for e in new["elements"] if e["nameStatus"] == "ok"]
    if len(new_ids) != len(set(new_ids)):
        return True, f"{FORMATS_FILE}: duplicate format names"
    changed_ids: set[str] = set()
    for entry in changed:
        if entry["loadTimeCode"]:
            return True, f"{FORMATS_FILE}: changed entry runs code when the file loads"
        if entry["mod"] is not None and entry in new["elements"] and not mod_exists(entry["mod"]):
            return True, f"{FORMATS_FILE}: changed entry uses missing mod {entry['mod']!r}"
        if entry["section"] and entry["nameStatus"] == "missing":
            continue  # a section heading: display only
        if entry["nameStatus"] != "ok":
            return True, f"{FORMATS_FILE}: changed entry without a plain string name"
        changed_ids.add(_entry_id(entry))
    protected = rule_ids | {to_id(rule.lstrip("!+-*")) for rule in ours_new[0]["ruleset"] or ()}
    clash = sorted(changed_ids & (protected | {context.format_id}))
    if clash:
        return True, f"{FORMATS_FILE}: changed entry shares an id with rule(s) {clash}"
    return False, f"{FORMATS_FILE}: other formats only ({len(changed_ids)} entries changed)"


def _check_formats_load(
    sha: str, read_file: FileReader, parse: Parser, cleared_reason: str
) -> tuple[bool, str]:
    """Block unless Showdown's own loader accepts the commit's whole format list.

    Rules and formats resolve through aliases, so the commit's aliases go in too.
    Compiling (esbuild spawns a binary) and loading (sandboxed) run as separate processes.
    """
    compile_jobs = []
    for path in (FORMATS_FILE, ALIASES_FILE):
        text = read_file(sha, path)
        if text is None:
            raise RuntimeError(f"{path} missing at {sha}")
        compile_jobs.append({"kind": "compile", "text": text, "file": path})
    compiled = parse(compile_jobs)
    for path, result in zip((FORMATS_FILE, ALIASES_FILE), compiled, strict=True):
        if not result.get("ok"):
            raise RuntimeError(f"could not compile {path}: {result.get('error')}")
    formats_js, aliases_js = (result["js"] for result in compiled)
    job = {"kind": "loadformats", "formats": formats_js, "aliases": aliases_js}
    (result,) = parse([{**job, "mods": _mod_dirs(sha, read_file)}])
    if not result.get("ok"):
        raise RuntimeError(f"load check failed to run: {result.get('error')}")
    if result.get("loadError") is not None:
        error = result["loadError"]
        return True, f"{FORMATS_FILE}: Showdown cannot load the format list ({error})"
    if "loadError" not in result:
        raise RuntimeError("load check returned no verdict")
    return False, cleared_reason


def _mod_dirs(sha: str, read_file: FileReader) -> list[str]:
    """The data/mods folders at `sha` (`git show rev:dir` lists a tree, dirs end in /)."""
    listing = read_file(sha, "data/mods")
    if listing is None or not listing.startswith("tree "):
        raise RuntimeError(f"could not list data/mods at {sha}")
    mods = [line[:-1] for line in listing.splitlines()[1:] if line.endswith("/")]
    if not mods:
        raise RuntimeError(f"no mods listed at {sha}")
    return mods


def _entry_id(entry: dict) -> str:
    return to_id(entry["name"]) if entry["nameStatus"] == "ok" else ""


# --- data/aliases.ts ---------------------------------------------------------------


def _classify_aliases(
    old: dict, new: dict, context: FormatContext, rule_ids: frozenset[str]
) -> tuple[bool, str]:
    if old["skeleton"] != new["skeleton"]:
        return True, f"{ALIASES_FILE}: code outside the alias table changed"
    # Checked before collapsing duplicates: a shadowed initializer still runs.
    if old["loadTimeCode"] or new["loadTimeCode"]:
        return True, f"{ALIASES_FILE}: alias table runs code when the file loads"
    if any(value is None for _key, value in old["entries"] + new["entries"]):
        return True, f"{ALIASES_FILE}: alias table has a non-string target"
    # dict() keeps the LAST value for a repeated key, exactly as a JS object literal does.
    old_map, new_map = dict(old["entries"]), dict(new["entries"])
    protected = context.all_legal_ids | rule_ids | {context.format_id}
    touched: list[str] = []
    for key in sorted(old_map.keys() | new_map.keys()):
        before, after = old_map.get(key), new_map.get(key)
        if before == after and (key in old_map) == (key in new_map):
            continue
        if (key in old_map and before is None) or (key in new_map and after is None):
            return True, f"{ALIASES_FILE}: alias {key!r} has a non-string target"
        hit = {to_id(key), to_id(before or ""), to_id(after or "")} & protected
        if hit:
            return True, f"{ALIASES_FILE}: alias touches Reg M-C or rule id(s) {sorted(hit)}"
        touched.append(to_id(key))
    return False, f"{ALIASES_FILE}: unrelated aliases only ({', '.join(touched) or 'none'})"


# --- dex tables (explanation only; always relevant) --------------------------------


def _describe_dex_change(
    path: str, before: str | None, after: str | None, context: FormatContext
) -> str:
    old = _split_entries(before, _DEX_ENTRY_START)
    new = _split_entries(after, _DEX_ENTRY_START)
    if old is None or new is None:
        return f"{path}: changed (could not map to entries)"
    old_skeleton, old_entries = old
    new_skeleton, new_entries = new
    old_map, new_map = dict(old_entries), dict(new_entries)
    changed = sorted(k for k in old_map.keys() | new_map.keys() if old_map.get(k) != new_map.get(k))
    legal = context.legal[_DEX_TABLES[path]]
    named = [f"{key} (in Reg M-C)" if key in legal else key for key in changed]
    named.sort(key=lambda text: "(in Reg M-C)" not in text)
    shown = ", ".join(named[:_MAX_NAMED_ENTRIES])
    if len(named) > _MAX_NAMED_ENTRIES:
        shown += f", +{len(named) - _MAX_NAMED_ENTRIES} more"
    extra = "; shared code outside entries changed" if old_skeleton != new_skeleton else ""
    return f"{path}: {shown or 'no entry changed'}{extra}"


# --- shared parsing ----------------------------------------------------------------


def _split_entries(
    text: str | None, start: re.Pattern[str]
) -> tuple[str, list[tuple[str, str]]] | None:
    """Split a Showdown data file into (text outside entries, [(key, entry text)]).

    Showdown's lint enforces tab indentation, so a top-level entry opens on a line
    matching `start` at one tab and closes on the next `\\t},` line. Returns None for a
    missing file or a layout this cannot follow (an entry opening inside another, or an
    unclosed entry) so the caller blocks instead of guessing.
    """
    if text is None:
        return None
    skeleton: list[str] = []
    entries: list[tuple[str, str]] = []
    current: list[str] | None = None
    key = ""
    for line in text.splitlines():
        match = start.match(line)
        if current is None:
            if match:
                current = [line]
                groups = [g for g in match.groups() if g] if match.groups() else []
                key = to_id(groups[0]) if groups else ""
            else:
                skeleton.append(line)
            continue
        if match:
            return None
        current.append(line)
        if line == _ENTRY_END:
            entries.append((key, "\n".join(current)))
            current = None
    if current is not None:
        return None
    return "\n".join(skeleton), entries


def git_file_reader(repo: Path) -> Callable[[str, str], str | None]:
    def read(rev: str, path: str) -> str | None:
        result = subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout
        # Absent at that revision is a real answer; any other failure must not be.
        exists = subprocess.run(
            ["git", "cat-file", "-e", rev],
            cwd=repo,
            capture_output=True,
            check=False,
        )
        if exists.returncode != 0:
            raise RuntimeError(f"unknown revision {rev}")
        return None

    return read
