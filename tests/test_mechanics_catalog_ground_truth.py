"""The checked-in catalogue must exactly match the current Showdown checkout."""

from __future__ import annotations

import json
import subprocess

import pytest

from vgc.config import REPO_ROOT, SHOWDOWN_REPO
from vgc.node import find_node

pytestmark = pytest.mark.integration


def test_mechanics_catalog_matches_fully_merged_champions_dex() -> None:
    result = subprocess.run(
        [
            find_node(),
            str(REPO_ROOT / "tools" / "export_mechanics_catalog.mjs"),
            str(SHOWDOWN_REPO),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    generated = json.loads(result.stdout)
    checked_in = json.loads(
        (REPO_ROOT / "data" / "champions" / "mechanics_catalog.json").read_text()
    )

    assert generated == checked_in
