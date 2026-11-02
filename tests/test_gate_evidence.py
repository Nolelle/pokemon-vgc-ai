"""A successful pytest exit alone must not certify skipped required evidence."""
from pathlib import Path

from vgc.gate_evidence import run_gate_pytest


def test_required_gate_cases_must_execute(tmp_path: Path) -> None:
    probe = tmp_path / "test_required_probe.py"
    probe.write_text(
        "import pytest\n"
        "def test_pass():\n    assert 2 + 2 == 4\n"
        "def test_missing_fixture():\n    pytest.skip('required fixture is missing')\n"
    )
    passing = run_gate_pytest([f"{probe}::test_pass"])
    assert passing.returncode == 0, passing.stdout + passing.stderr
    incomplete = run_gate_pytest([str(probe)])
    assert incomplete.returncode != 0
    assert "required gate tests were skipped" in incomplete.stderr
    assert "test_missing_fixture" in incomplete.stderr
