import json

import pytest

from ladder.run_ladder import (
    PASSWORD_ENV,
    USERNAME_ENV,
    Credentials,
    load_credentials,
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
