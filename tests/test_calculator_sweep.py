"""Standing check: our fast calculators against the real Showdown engine.

`offline/sweep_calculator_vs_engine.py` compares `vgc.damage` / `vgc.accuracy` / the Speed and
priority ordering with the engine for every legal move, ability and item and writes
`data/champions/calculator_coverage.json`. This module re-runs a fixed ~1/25 thinning of that
sweep (well under three minutes) and fails if an entry recorded as EXACT stops matching, plus a
full run of the entries the sweep found WRONG and that were fixed (each one is a ground-truth
test: the engine is the oracle). Full command, ~15-20 minutes:

    .venv/bin/python offline/sweep_calculator_vs_engine.py --profile full \
        --output data/champions/calculator_coverage.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vgc import calc_sweep as cs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "offline"))
import sweep_calculator_vs_engine as sweep  # noqa: E402

pytestmark = pytest.mark.integration

SUBSET_STEP = 25

# Entries the sweep found WRONG that now match the engine exactly, with the family that proved it.
FIXED_ENTRIES = {
    "moves": [
        "electroshot", "expandingforce", "terrainpulse", "risingvoltage", "knockoff", "acrobatics",
        "stormthrow", "frostbreath", "flowertrick", "freezedry", "solarbeam", "brickbreak",
        "hex", "facade", "seismictoss", "superfang", "earthquake", "flyingpress", "shellsidearm",
        "heatwave", "rockslide", "dazzlinggleam",
    ],
    "abilities": [
        "toughclaws", "hugepower", "sheerforce", "fairyaura", "ironfist", "sharpness",
        "megalauncher", "technician", "armortail", "queenlymajesty", "soundproof", "bulletproof",
        "sapsipper", "protean", "libero", "unaware", "punkrock", "friendguard", "thickfat",
        "megasol", "cloudnine", "eelevate", "lightningrod", "prankster", "galewings", "quickfeet",
        "stall", "surgesurfer",
    ],
    "items": ["blackbelt", "spelltag", "grassyseed", "psychicseed", "cheriberry", "rawstberry", "ironball"],
}


def _coverage() -> dict:
    path = cs.COVERAGE_PATH
    if not path.exists():
        pytest.skip(f"{path} not generated yet (run offline/sweep_calculator_vs_engine.py)")
    with path.open() as file:
        return json.load(file)


def test_exact_entries_do_not_regress() -> None:
    coverage = _coverage()
    cases = sweep.build_cases(
        set(sweep.FAMILIES), coverage["meta"]["profile"], SUBSET_STEP, None
    )
    table, _totals = sweep.evaluate(cases, workers=8)
    recorded = coverage["entries"]
    regressions = []
    for (category, entry_kind, entry), result in table.items():
        was = recorded.get(category, {}).get(entry_kind, {}).get(entry)
        if was is not None and was["status"] == "EXACT" and result.status != "EXACT":
            regressions.append(
                f"{category}/{entry_kind}/{entry}: {result.status} "
                f"({result.n_mismatch} mismatches; e.g. {result.mismatches[:1]})"
            )
    assert not regressions, "entries that were EXACT now disagree with the engine:\n" + "\n".join(
        regressions[:30]
    )


@pytest.mark.parametrize("family", sorted(FIXED_ENTRIES))
def test_fixed_entries_match_engine_exactly(family: str) -> None:
    entries = set(FIXED_ENTRIES[family])
    cases = sweep.build_cases({family, "order"} if family != "moves" else {family}, "quick", 1, entries)
    table, _totals = sweep.evaluate(cases, workers=8)
    bad = [
        f"{kind}/{entry_kind}/{entry}: {result.status} e.g. {result.mismatches[:1]}"
        for (kind, entry_kind, entry), result in table.items()
        if entry in entries and result.status not in ("EXACT", "NOT_EXERCISED")
    ]
    assert not bad, "\n".join(bad)
