#!/usr/bin/env python
"""Build a pool of legal, corpus-informed opponent teams for self-play data generation
(`selfplay/run_selfplay.py`) -- assembles teams via `vgc.selfplay_pool` from `data/usage/
set_priors.json` + `data/usage/spreads.json`'s real corpus frequencies, then validates
EVERY one against the real Showdown `validate-team` CLI (CLAUDE.md's mandated legality
check -- `vgc.selfplay_pool`'s own best-effort assembly is not itself a legality
guarantee) and only writes the ones that pass. A team that fails validation is
re-sampled (different seed offset) up to `--max-attempts` times before being dropped, so
a single bad combination doesn't silently shrink the pool below `--n-teams`.

Output: `data/selfplay/pool/team_NN.packed.txt`, one packed team per file (gitignored --
see `data/selfplay/` in `.gitignore`).

Usage:
    .venv/bin/python tools/build_selfplay_pool.py
    .venv/bin/python tools/build_selfplay_pool.py --n-teams 20 --seed 0 \
        --out data/selfplay/pool
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vgc.config import FORMAT_ID, SHOWDOWN_REPO  # noqa: E402
from vgc.node import find_node, node_environment  # noqa: E402
from vgc.selfplay_pool import (  # noqa: E402
    DEFAULT_N_TEAMS,
    TOP_N_SPECIES,
    build_priority_species,
    sample_teams,
    team_to_packed,
)
from vgc.sets import load_set_priors, load_usage_spreads  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "data" / "selfplay" / "pool"
DEFAULT_MAX_ATTEMPTS = 60


def validate_team(packed_team: str, node: str) -> str | None:
    """Runs the real `validate-team` CLI against `packed_team`. Returns `None` if legal,
    else the CLI's error output. Never raises: a subprocess failure (missing node/repo,
    timeout) is reported as a validation failure string rather than crashing the run.
    """
    try:
        result = subprocess.run(
            [node, "pokemon-showdown", "validate-team", FORMAT_ID],
            cwd=str(SHOWDOWN_REPO),
            input=packed_team,
            capture_output=True,
            text=True,
            env=node_environment(node),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"validate-team invocation failed: {exc}"
    output = (result.stdout + result.stderr).strip()
    return output or None


def build_pool(
    *, n_teams: int, seed: int, top_n: int, max_attempts: int, node: str
) -> tuple[list[str], list[str]]:
    """Returns `(valid_packed_teams, failure_reasons)`. Draws candidate teams with
    increasing seed offsets from the same corpus-derived species pool until `n_teams`
    pass validation or `max_attempts` candidates have been tried.
    """
    set_priors = load_set_priors()
    spreads = load_usage_spreads()
    pool = build_priority_species(set_priors, spreads, top_n=top_n)
    if len(pool) < 6:
        raise RuntimeError(
            f"only {len(pool)} usable species assembled from set_priors.json -- "
            "need at least 6 to build any team (is data/usage/set_priors.json populated?)"
        )

    valid: list[str] = []
    failures: list[str] = []
    attempt = 0
    while len(valid) < n_teams and attempt < max_attempts:
        # One fresh candidate team per attempt -- sample_teams(n_teams=1, seed=...) with
        # a strictly increasing seed offset keeps every attempt deterministic overall
        # (same --seed always tries the same sequence of candidates) while still trying
        # a different combination each time.
        candidate = sample_teams(pool, n_teams=1, seed=seed + attempt)[0]
        packed = team_to_packed(candidate)
        error = validate_team(packed, node)
        if error is None:
            valid.append(packed)
        else:
            failures.append(f"attempt {attempt}: {error}")
        attempt += 1
    return valid, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-teams", type=int, default=DEFAULT_N_TEAMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n-species", type=int, default=TOP_N_SPECIES)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    node = find_node()
    valid, failures = build_pool(
        n_teams=args.n_teams,
        seed=args.seed,
        top_n=args.top_n_species,
        max_attempts=args.max_attempts,
        node=node,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    # Clear any stale team files from a previous run before writing the new pool, so the
    # directory never silently accumulates teams from an older species/corpus snapshot.
    for stale in args.out.glob("team_*.packed.txt"):
        stale.unlink()
    for index, packed in enumerate(valid):
        (args.out / f"team_{index:02d}.packed.txt").write_text(packed + "\n")

    print(f"build_selfplay_pool summary: {len(valid)}/{args.n_teams} requested teams built")
    print(f"  attempts: {len(valid) + len(failures)} (failures: {len(failures)})")
    if failures:
        print("  failure reasons (first 5):")
        for reason in failures[:5]:
            print(f"    {reason}")
    print(f"  output: {args.out}")

    if len(valid) < args.n_teams:
        print(
            f"WARNING: only produced {len(valid)}/{args.n_teams} teams within "
            f"{args.max_attempts} attempts",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
