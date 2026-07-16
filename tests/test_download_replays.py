"""Unit tests for tools/download_replays.py's pure decision logic. NO network calls --
`_fetch_json`/`main` (the only functions that touch the network) are intentionally not
exercised here; see the module docstring for how those were manually verified instead.
"""

from __future__ import annotations

import pytest

from tools.download_replays import (
    PAGE_SIZE,
    next_before_cursor,
    replay_filename,
    should_download,
)


def _entry(
    entry_id: str = "gen9championsvgc2026regmb-1", rating: int | None = 1200, uploadtime: int = 100
) -> dict:
    return {"id": entry_id, "rating": rating, "uploadtime": uploadtime}


# --- should_download: rating filter (incl. null rating) + existing-id skip ----------


def test_should_download_true_for_new_sufficiently_rated_entry() -> None:
    assert should_download(_entry(rating=1200), min_rating=1100, existing_ids=set()) is True


def test_should_download_false_below_min_rating() -> None:
    assert should_download(_entry(rating=1050), min_rating=1100, existing_ids=set()) is False


def test_should_download_false_at_exactly_min_rating_is_true() -> None:
    # min_rating is an inclusive floor ("skip lower-rated", not "skip at-or-below").
    assert should_download(_entry(rating=1100), min_rating=1100, existing_ids=set()) is True


def test_should_download_false_for_null_rating() -> None:
    assert should_download(_entry(rating=None), min_rating=1100, existing_ids=set()) is False


def test_should_download_false_for_missing_rating_key() -> None:
    entry = {"id": "gen9championsvgc2026regmb-1", "uploadtime": 100}
    assert should_download(entry, min_rating=1100, existing_ids=set()) is False


def test_should_download_false_for_already_existing_id() -> None:
    entry = _entry(entry_id="gen9championsvgc2026regmb-42", rating=1500)
    existing = {"gen9championsvgc2026regmb-42"}
    assert should_download(entry, min_rating=1100, existing_ids=existing) is False


def test_should_download_false_for_missing_id() -> None:
    entry = {"rating": 1500, "uploadtime": 100}
    assert should_download(entry, min_rating=1100, existing_ids=set()) is False


# --- replay_filename: sanitize ids, reject traversal -----------------------------------


def test_replay_filename_happy_path() -> None:
    assert (
        replay_filename("gen9championsvgc2026regmb-2649967888")
        == "gen9championsvgc2026regmb-2649967888.json"
    )


def test_replay_filename_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="unsafe replay id"):
        replay_filename("../evil")


def test_replay_filename_rejects_path_separator() -> None:
    with pytest.raises(ValueError):
        replay_filename("some/path")


def test_replay_filename_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        replay_filename("")


def test_replay_filename_rejects_dot_dot() -> None:
    with pytest.raises(ValueError):
        replay_filename("..")


# --- next_before_cursor: pagination-cursor extraction ----------------------------------


def test_next_before_cursor_full_page_returns_oldest_uploadtime() -> None:
    entries = [_entry(uploadtime=100 - i) for i in range(PAGE_SIZE)]  # descending, newest first
    assert next_before_cursor(entries) == entries[-1]["uploadtime"]


def test_next_before_cursor_short_page_returns_none() -> None:
    entries = [_entry(uploadtime=100), _entry(uploadtime=99)]
    assert next_before_cursor(entries) is None


def test_next_before_cursor_empty_page_returns_none() -> None:
    assert next_before_cursor([]) is None


def test_next_before_cursor_custom_page_size() -> None:
    entries = [_entry(uploadtime=100), _entry(uploadtime=99), _entry(uploadtime=98)]
    assert next_before_cursor(entries, page_size=3) == 98
    assert next_before_cursor(entries, page_size=4) is None
