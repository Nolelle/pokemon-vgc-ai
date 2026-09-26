"""Minimal HTTP client for TypeSafe System One (Jev) batch evaluation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

TYPESAFE_SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_TIMEOUT_SECONDS = 2.0


class SystemOneClientError(Exception):
    """Non-fatal client failure; callers should record and continue."""


def evaluate_system_one(
    state: object,
    questions: dict[str, dict[str, object]],
    *,
    api_key: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST one batched evaluation; returns the parsed JSON response body."""

    payload = {
        "state": state,
        "model": TYPESAFE_MODEL,
        "questions": questions,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        TYPESAFE_SYSTEMONE_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise SystemOneClientError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise SystemOneClientError(f"connection failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SystemOneClientError("request timed out") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemOneClientError("invalid JSON response") from exc
    if not isinstance(parsed, dict):
        raise SystemOneClientError("response is not a JSON object")
    return parsed


def read_api_key_from_env() -> str | None:
    value = os.environ.get(TYPESAFE_API_KEY_ENV)
    if value is None or not str(value).strip():
        return None
    return str(value).strip()
