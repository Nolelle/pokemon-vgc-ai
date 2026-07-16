#!/usr/bin/env python
"""Download public rated replays of a Showdown format for later behavior-cloning
(Phase 3 RL/BC prep) from Pokemon Showdown's replay API.

Two endpoints, verified manually:
  - `https://replay.pokemonshowdown.com/search.json?format=<id>` -- up to 50 entries
    (fields: uploadtime, id, format, players, rating, private, password), newest first.
    Paginates with `&before=<uploadtime of the oldest entry on this page>`. REQUIRES a
    User-Agent header (some requests are rejected without one).
  - `https://replay.pokemonshowdown.com/<id>.json` -- the full replay: the same
    id/rating/uploadtime/players fields plus the battle `log` text (and `views`/
    `formatid`/`format`/`private`/`password`).

As of writing, the `gen9championsvgc2026regmb` corpus has 3000+ replays, ~950 rated
>=1200 -- `--min-rating` (default 1100) keeps this from downloading low-quality/ladder-
noise games nobody would want to clone from.

One JSON file per replay (`<out>/<id>.json`, written atomically via tmp+rename) plus an
`<out>/index.jsonl` (one line per on-disk replay: id/rating/uploadtime/players),
REGENERATED FROM THE ON-DISK FILES at the end of every run so it's always consistent
even after a partial/aborted run. Already-downloaded ids are skipped without a network
call, so re-running this incrementally is cheap and safe.

Usage:
    .venv/bin/python tools/download_replays.py
    .venv/bin/python tools/download_replays.py --min-rating 1300 --max-pages 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import FORMAT_ID  # noqa: E402

# Identifies us to Showdown's replay server -- some requests 403 without any
# User-Agent at all; a descriptive one is also just good etiquette for a scraper.
USER_AGENT = "vgc-research/0.1 (github.com/edmundyu personal research)"

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
REPLAY_URL_TEMPLATE = "https://replay.pokemonshowdown.com/{id}.json"
# search.json returns at most this many entries per page (a short page means we've
# reached the end of the corpus -- see `next_before_cursor`).
PAGE_SIZE = 50
# How long to wait before the single retry on a transient HTTP/JSON failure.
RETRY_SLEEP_SECONDS = 5.0

DEFAULT_MIN_RATING = 1100
DEFAULT_MAX_PAGES = 100
DEFAULT_SLEEP_SECONDS = 0.3
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "replays" / FORMAT_ID

# Showdown replay ids look like "gen9championsvgc2026regmb-2649967888" -- alphanumeric
# plus hyphen only. Rejecting anything else before it touches the filesystem closes off
# path-traversal via a maliciously/corruptly-shaped "id" field (e.g. "../../evil").
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")


# --- pure decision logic (unit tested directly, no network) -------------------------


def _meets_rating(entry: dict, min_rating: int) -> bool:
    """True if `entry["rating"]` is present and >= `min_rating`. A missing/null rating
    (unrated ladder games, or a game that never got its rating attached) never passes --
    unrated is treated as "unknown quality", not as "elite enough to skip the check".
    """
    rating = entry.get("rating")
    return rating is not None and rating >= min_rating


def should_download(entry: dict, min_rating: int, existing_ids: set[str]) -> bool:
    """True if `entry` (one row from search.json, or a full replay detail payload --
    both carry the same id/rating fields) is worth fetching: it has an id, that id isn't
    already in `existing_ids` (an incremental re-run's on-disk replay ids -- see
    `_existing_replay_ids`), and its rating clears `min_rating`.
    """
    entry_id = entry.get("id")
    if not entry_id:
        return False
    if entry_id in existing_ids:
        return False
    return _meets_rating(entry, min_rating)


def replay_filename(entry_id: str) -> str:
    """`<entry_id>.json`, after validating `entry_id` is alphanumeric+hyphen only.
    Raises `ValueError` for anything else (path separators, `..`, empty string, etc.) --
    defense in depth against a malformed/malicious id ever being used to build a
    filesystem path, even though the real API is not expected to ever send one.
    """
    if not entry_id or not _SAFE_ID_RE.match(entry_id):
        raise ValueError(f"unsafe replay id: {entry_id!r}")
    return f"{entry_id}.json"


def next_before_cursor(entries: list[dict], page_size: int = PAGE_SIZE) -> int | None:
    """The `before=` cursor for the NEXT search.json page, or `None` if `entries` was
    the last page. search.json returns up to `page_size` entries newest-first; a full
    page means there may be more (retry with `before=<oldest uploadtime on this page>`,
    i.e. the minimum -- entries are sorted descending so that's the last one), a SHORT
    page (< page_size, including empty) means the corpus is exhausted.
    """
    if not entries or len(entries) < page_size:
        return None
    return min(entry["uploadtime"] for entry in entries)


# --- networking (thin wrappers -- everything above is what's actually unit tested) --


def _fetch_json(url: str, timeout: float = 30.0) -> object:
    """GET `url` as JSON with the required User-Agent header. One retry after
    `RETRY_SLEEP_SECONDS` on a transient failure (connection error, timeout, or a
    non-JSON/malformed response body); the second attempt's exception propagates
    uncaught so the caller decides whether that's fatal (a search page) or
    loggable-and-continue (a single replay) -- see module docstring.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(
            f"  request failed ({exc}); retrying once in {RETRY_SLEEP_SECONDS}s: {url}",
            file=sys.stderr,
        )
        time.sleep(RETRY_SLEEP_SECONDS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())


def _search_page_url(format_id: str, before: int | None) -> str:
    url = f"{SEARCH_URL}?format={format_id}"
    if before is not None:
        url += f"&before={before}"
    return url


def _write_json_atomic(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload))
    tmp_path.replace(path)  # atomic on the same filesystem (POSIX rename)


def _existing_replay_ids(out_dir: Path) -> set[str]:
    return {path.stem for path in out_dir.glob("*.json")}


def rebuild_index(out_dir: Path) -> int:
    """Regenerate `<out_dir>/index.jsonl` from the on-disk `*.json` replay files (NOT
    from anything held in memory during the run), so it's always an accurate reflection
    of what's actually saved -- correct even after an aborted run, a manually deleted
    replay, or replays added by a separate process. Returns the row count written.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(out_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue  # a stray/corrupt file -- skip it rather than abort the whole index
        rows.append(
            {
                "id": payload.get("id", path.stem),
                "rating": payload.get("rating"),
                "uploadtime": payload.get("uploadtime"),
                "players": payload.get("players"),
            }
        )
    rows.sort(key=lambda row: row.get("uploadtime") or 0, reverse=True)

    index_path = out_dir / "index.jsonl"
    with index_path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


# --- CLI ------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=DEFAULT_MIN_RATING,
        help="skip replays rated below this, and null/unrated replays (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="search.json pages to walk at most, 50 replays/page (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="directory to write <id>.json replays + index.jsonl into (default: %(default)s)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="seconds to sleep between HTTP requests (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        default=FORMAT_ID,
        help="Showdown format id to search (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    existing_ids = _existing_replay_ids(args.out)

    pages_walked = 0
    replays_seen = 0
    skipped_by_rating = 0
    already_present = 0
    newly_downloaded = 0
    before: int | None = None

    for _page in range(args.max_pages):
        url = _search_page_url(args.format, before)
        try:
            entries = _fetch_json(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"search page request failed after retry ({exc}); stopping here", file=sys.stderr)
            break
        pages_walked += 1
        time.sleep(args.sleep)
        if not entries:
            break

        for entry in entries:
            replays_seen += 1
            if not should_download(entry, args.min_rating, existing_ids):
                entry_id = entry.get("id")
                if entry_id and entry_id in existing_ids:
                    already_present += 1
                else:
                    skipped_by_rating += 1
                continue

            entry_id = entry["id"]
            try:
                replay = _fetch_json(REPLAY_URL_TEMPLATE.format(id=entry_id))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"  replay {entry_id} failed after retry ({exc}); skipping", file=sys.stderr)
                time.sleep(args.sleep)
                continue
            time.sleep(args.sleep)

            _write_json_atomic(args.out / replay_filename(entry_id), replay)
            existing_ids.add(entry_id)
            newly_downloaded += 1

        before = next_before_cursor(entries)
        if before is None:
            break

    index_rows = rebuild_index(args.out)

    print("download_replays summary:")
    print(f"  pages walked:      {pages_walked}")
    print(f"  replays seen:      {replays_seen}")
    print(f"  skipped by rating: {skipped_by_rating}")
    print(f"  already present:   {already_present}")
    print(f"  newly downloaded:  {newly_downloaded}")
    print(f"  index rows:        {index_rows} ({args.out / 'index.jsonl'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
