import json
from pathlib import Path

import pytest

from ladder.run_ladder import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR,
    DEFAULT_LOCAL_SMOKE_LOG_PATH,
    DEFAULT_LOG_PATH,
    PASSWORD_ENV,
    USERNAME_ENV,
    Credentials,
    load_credentials,
    resolve_output_paths,
)


def test_credentials_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(USERNAME_ENV, "example-user")
    monkeypatch.setenv(PASSWORD_ENV, "example-password")

    assert load_credentials() == Credentials("example-user", "example-password")


def test_credentials_load_from_gitignored_file(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(USERNAME_ENV, raising=False)
    monkeypatch.delenv(PASSWORD_ENV, raising=False)
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"username": "file-user", "password": "file-password"}))

    assert load_credentials(path) == Credentials("file-user", "file-password")


def test_partial_environment_credentials_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv(USERNAME_ENV, "example-user")
    monkeypatch.delenv(PASSWORD_ENV, raising=False)

    with pytest.raises(ValueError, match="set both"):
        load_credentials()


# --- resolve_output_paths: --local-smoke must never default onto the public ladder's
# --- log/artifacts paths, but an explicit override always wins in either mode --------


def test_resolve_output_paths_live_mode_defaults() -> None:
    artifacts_dir, log_path = resolve_output_paths(False, None, None)
    assert artifacts_dir == DEFAULT_ARTIFACTS_DIR
    assert log_path == DEFAULT_LOG_PATH


def test_resolve_output_paths_local_smoke_defaults_to_smoke_paths() -> None:
    artifacts_dir, log_path = resolve_output_paths(True, None, None)
    assert artifacts_dir == DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR
    assert log_path == DEFAULT_LOCAL_SMOKE_LOG_PATH
    # The whole point of the fix: smoke mode must never fall back to the public paths.
    assert artifacts_dir != DEFAULT_ARTIFACTS_DIR
    assert log_path != DEFAULT_LOG_PATH


def test_resolve_output_paths_explicit_log_wins_in_local_smoke_mode() -> None:
    override = Path("/tmp/custom.jsonl")
    artifacts_dir, log_path = resolve_output_paths(True, override, None)
    assert log_path == override
    assert artifacts_dir == DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR


def test_resolve_output_paths_explicit_artifacts_dir_wins_in_live_mode() -> None:
    override = Path("/tmp/custom-artifacts")
    artifacts_dir, log_path = resolve_output_paths(False, None, override)
    assert artifacts_dir == override
    assert log_path == DEFAULT_LOG_PATH


def test_resolve_output_paths_both_explicit_in_either_mode() -> None:
    log_override = Path("/tmp/explicit.jsonl")
    artifacts_override = Path("/tmp/explicit-artifacts")
    for local_smoke in (True, False):
        artifacts_dir, log_path = resolve_output_paths(
            local_smoke, log_override, artifacts_override
        )
        assert artifacts_dir == artifacts_override
        assert log_path == log_override
