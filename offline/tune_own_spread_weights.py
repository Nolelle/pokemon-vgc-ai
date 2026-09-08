"""Screen Protect/switch weights after enabling accurate own-team spreads.

Every candidate knows its real Stat Points. By default, the incumbent is the legacy
fake-spread policy with Protect weight 0.8, preserving the policy the retune must beat.
``--equal-weights`` instead copies each candidate's Protect/switch overrides to the
fake-spread incumbent for a strict one-variable causal mirror. Both agents use Phase 2's
fixed team-preview order, alternate battle sides, and receive the same simulator seed
stream for every candidate.

This is a screening tool, not a promotion gate. Select the strongest simple candidate on
one seed, then confirm it with a larger run and a different seed before changing defaults.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vgc.config import FORMAT_ID, RUNS_DIR, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.rl.agents import PHASE2_PREVIEW_ORDER, make_direct_agent  # noqa: E402
from vgc.rl.env import SimWorker  # noqa: E402
from vgc.rl.match import run_series, summarize  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    name: str
    overrides: dict[str, float]


CANDIDATES = (
    Candidate("accurate_shipped", {}),
    Candidate("accurate_untuned", {"protect_threat_weight": 0.8}),
    Candidate("protect_070", {"protect_threat_weight": 0.7}),
    Candidate("protect_060", {"protect_threat_weight": 0.6}),
    Candidate("protect_050", {"protect_threat_weight": 0.5}),
    Candidate("switch_tempo_10", {"switch_tempo_cost": 10.0}),
    Candidate(
        "protect_060_switch_tempo_10",
        {"protect_threat_weight": 0.6, "switch_tempo_cost": 10.0},
    ),
    Candidate(
        "protect_050_switch_tempo_10",
        {"protect_threat_weight": 0.5, "switch_tempo_cost": 10.0},
    ),
    Candidate(
        "protect_060_switch_tempo_05",
        {"protect_threat_weight": 0.6, "switch_tempo_cost": 5.0},
    ),
    Candidate(
        "protect_060_switch_matchup_125",
        {"protect_threat_weight": 0.6, "switch_matchup_weight": 1.25},
    ),
)


def evaluate(
    candidate: Candidate, *, games: int, seed: int, team: str, equal_weights: bool = False
) -> dict[str, object]:
    candidate_config = PolicyConfig(
        format_id=FORMAT_ID,
        use_own_team_spreads=True,
        **candidate.overrides,
    )
    incumbent_overrides = candidate.overrides if equal_weights else {"protect_threat_weight": 0.8}
    incumbent_config = PolicyConfig(
        format_id=FORMAT_ID,
        use_own_team_spreads=False,
        **incumbent_overrides,
    )
    candidate_name = candidate.name
    incumbent_name = "fake_spread_default"
    factories = {
        candidate_name: partial(
            make_direct_agent,
            "vgc",
            team,
            config=candidate_config,
            preview_order=PHASE2_PREVIEW_ORDER,
        ),
        incumbent_name: partial(
            make_direct_agent,
            "vgc",
            team,
            config=incumbent_config,
            preview_order=PHASE2_PREVIEW_ORDER,
        ),
    }
    with SimWorker() as worker:
        outcomes = run_series(
            worker,
            factories,
            {candidate_name: team, incumbent_name: team},
            games,
            seed=seed,
        )
    result = summarize(outcomes, candidate_name, incumbent_name)
    result["candidate_overrides"] = candidate.overrides
    result["incumbent_overrides"] = incumbent_overrides
    result["fixed_preview_order"] = PHASE2_PREVIEW_ORDER
    result["seed"] = seed
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--team", type=Path, default=TEAMS_DIR / "phase2_mirror.packed.txt")
    parser.add_argument(
        "--candidates",
        nargs="*",
        choices=[candidate.name for candidate in CANDIDATES],
        help="candidate names to run (default: all)",
    )
    parser.add_argument(
        "--equal-weights",
        action="store_true",
        help="copy each candidate's weight overrides to the fake-spread incumbent",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "eval" / "own_spread_weight_screen.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [
        candidate
        for candidate in CANDIDATES
        if not args.candidates or candidate.name in args.candidates
    ]
    team = args.team.read_text().strip()
    results = []
    for candidate in selected:
        print(f"running {candidate.name}: n={args.games}, seed={args.seed}", flush=True)
        result = evaluate(
            candidate,
            games=args.games,
            seed=args.seed,
            team=team,
            equal_weights=args.equal_weights,
        )
        results.append(result)
        print(
            f"  {result['p1_wins']}/{result['games']} = {result['p1_win_rate']:.3f}, "
            f"95% CI [{result['p1_wilson'][0]:.3f}, {result['p1_wilson'][1]:.3f}]",
            flush=True,
        )

    payload = {
        "purpose": "retune Protect/switch weights after accurate own-team spreads",
        "team": str(args.team),
        "incumbent": {
            "policy": (
                "candidate weights" if args.equal_weights else "legacy PolicyConfig at Protect 0.8"
            ),
            "use_own_team_spreads": False,
        },
        "equal_weights": args.equal_weights,
        "games_per_candidate": args.games,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
