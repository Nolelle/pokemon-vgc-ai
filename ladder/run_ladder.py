#!/usr/bin/env python
"""Run a small, observable Champions VGC ladder session.

Use ``--local-smoke`` against a running local Showdown server before using the public
ladder. Public credentials come from ``VGC_SHOWDOWN_USERNAME`` /
``VGC_SHOWDOWN_PASSWORD`` or the gitignored ``.showdown-credentials.json`` file.

Smoke sessions default their replay/trace artifacts and outcome log to the
``-local-smoke`` paths (``runs/ladder-local-smoke/`` and
``runs/ladder-local-smoke.jsonl``) instead of the public-ladder defaults, so running
``--local-smoke`` never silently appends to (or overwrites replays under) the real
``runs/ladder.jsonl``/``runs/ladder/`` record -- pass ``--log``/``--artifacts-dir``
explicitly to override either mode's default.

``--search`` opts this session into the Phase 2c 2-ply search (``PolicyConfig.
use_two_ply_search``, False by default -- see its comment in ``vgc/models.py``) instead
of the plain myopic evaluator, for an A/B against real ladder opponents. Every
``runs/ladder.jsonl`` record carries a ``"policy"`` field (``"search"`` | ``"myopic"``)
so sessions stay attributable after the fact; the running mode is also printed at
startup and in the session summary (see ``session_config``/``policy_label``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Ladder sessions always retain the reasoning needed for post-loss review. The trace
# module reads this dynamically, so this remains effective even when imported in tests.
os.environ["VGC_TRACE"] = "1"

from poke_env.battle.abstract_battle import AbstractBattle  # noqa: E402
from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import (  # noqa: E402
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from vgc.agent import VgcPlayer  # noqa: E402
from vgc.baselines import BASELINES, make_player  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402

USERNAME_ENV = "VGC_SHOWDOWN_USERNAME"
PASSWORD_ENV = "VGC_SHOWDOWN_PASSWORD"
DEFAULT_CREDENTIALS_FILE = REPO_ROOT / ".showdown-credentials.json"
DEFAULT_ARTIFACTS_DIR = RUNS_DIR / "ladder"
DEFAULT_LOG_PATH = RUNS_DIR / "ladder.jsonl"
# --local-smoke defaults -- kept separate from the public-ladder defaults above so a
# smoke run against the local server can never silently append to (or write replays
# alongside) the real public-ladder record just because the caller forgot to override
# --log/--artifacts-dir.
DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR = RUNS_DIR / "ladder-local-smoke"
DEFAULT_LOCAL_SMOKE_LOG_PATH = RUNS_DIR / "ladder-local-smoke.jsonl"


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def load_credentials(path: Path | None = None) -> Credentials:
    """Load credentials without ever logging or returning them as a plain dictionary."""

    username = os.environ.get(USERNAME_ENV)
    password = os.environ.get(PASSWORD_ENV)
    if username is not None or password is not None:
        if not username or not password:
            raise ValueError(f"set both {USERNAME_ENV} and {PASSWORD_ENV}, or neither")
        return Credentials(username=username, password=password)

    resolved = path or DEFAULT_CREDENTIALS_FILE
    if not resolved.exists():
        raise FileNotFoundError(
            f"Showdown credentials not found. Set {USERNAME_ENV}/{PASSWORD_ENV} or create "
            f"{resolved} with username/password fields."
        )
    payload = json.loads(resolved.read_text())
    username = payload.get("username")
    password = payload.get("password")
    if (
        not isinstance(username, str)
        or not username
        or not isinstance(password, str)
        or not password
    ):
        raise ValueError(f"{resolved} must contain non-empty username and password strings")
    return Credentials(username=username, password=password)


class LadderPlayer(VgcPlayer):
    """VgcPlayer that writes replay-adjacent outcome and reasoning artifacts."""

    def __init__(
        self,
        *,
        artifacts_dir: Path,
        log_path: Path,
        session_id: str,
        **player_kwargs,
    ) -> None:
        self.artifacts_dir = artifacts_dir
        self.replay_dir = artifacts_dir / "replays"
        self.trace_dir = artifacts_dir / "traces"
        self.log_path = log_path
        self.session_id = session_id
        self.completed_records: list[dict[str, object]] = []
        self._pending_finished_battles: dict[str, AbstractBattle] = {}
        self.replay_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        player_kwargs.setdefault("save_replays", str(self.replay_dir))
        player_kwargs.setdefault("start_timer_on_battle_start", True)
        super().__init__(**player_kwargs)

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        # Showdown can send rating updates immediately after the win/tie line that
        # triggers this synchronous callback. Defer the JSONL write until the session
        # coroutine has briefly yielded, so battle.rating has time to populate.
        self._pending_finished_battles[battle.battle_tag] = battle

    async def flush_finished_battles(self, delay_seconds: float = 0.0) -> None:
        if delay_seconds > 0 and self._pending_finished_battles:
            await asyncio.sleep(delay_seconds)
        pending = list(self._pending_finished_battles.values())
        self._pending_finished_battles.clear()
        for battle in pending:
            self._write_battle_record(battle)

    def _write_battle_record(self, battle: AbstractBattle) -> None:
        traces = [
            trace
            for trace in self.decision_trace_history
            if trace.get("battle_tag") == battle.battle_tag
        ]
        trace_path = self.trace_dir / f"{battle.battle_tag}.json"
        trace_path.write_text(json.dumps(traces, indent=2, sort_keys=True))

        replay_matches = list(self.replay_dir.glob(f"*{battle.battle_tag}.html"))
        replay_path = replay_matches[0] if replay_matches else None
        record: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "battle_tag": battle.battle_tag,
            "format": battle.format,
            "opponent": battle.opponent_username,
            "won": battle.won,
            "lost": battle.lost,
            "turns": battle.turn,
            "rating": battle.rating,
            "opponent_rating": battle.opponent_rating,
            "fallback_count": sum(bool(trace.get("fallback_used")) for trace in traces),
            # Machine-readable A/B tag: "search" (2-ply search, --search) vs "myopic"
            # (score_joint_orders directly, the default) -- see PolicyConfig.
            # use_two_ply_search's comment for why the default flipped to False.
            "policy": "search" if self.config.use_two_ply_search else "myopic",
            "trace_path": str(trace_path.resolve()),
            "replay_path": str(replay_path.resolve()) if replay_path else None,
        }
        with self.log_path.open("a") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.completed_records.append(record)


def _session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_output_paths(
    local_smoke: bool, log: Path | None, artifacts_dir: Path | None
) -> tuple[Path, Path]:
    """Resolve the effective ``(artifacts_dir, log_path)`` pair from parsed CLI args.

    An explicit ``--log``/``--artifacts-dir`` value always wins, in either mode. When the
    caller didn't override them, the default depends on ``local_smoke``:
    ``--local-smoke`` sessions fall back to the ``-local-smoke`` paths (never the public
    ladder's), while live sessions fall back to the original public-ladder defaults. Pure
    and argparse-free so it's directly unit-testable (see tests/test_ladder.py).
    """
    if local_smoke:
        default_artifacts_dir = DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR
        default_log = DEFAULT_LOCAL_SMOKE_LOG_PATH
    else:
        default_artifacts_dir = DEFAULT_ARTIFACTS_DIR
        default_log = DEFAULT_LOG_PATH
    resolved_artifacts_dir = artifacts_dir if artifacts_dir is not None else default_artifacts_dir
    resolved_log = log if log is not None else default_log
    return resolved_artifacts_dir, resolved_log


def session_config(search: bool) -> PolicyConfig:
    """The `PolicyConfig` for one ladder session (smoke or live): the default config
    (myopic evaluator only), or the same config with the Phase 2c 2-ply search
    explicitly enabled via `--search`. `PolicyConfig.use_two_ply_search` defaults to
    False -- the offline gate proxy (SimpleHeuristicsPlayer) systematically punishes
    opponent-response modeling it doesn't itself exhibit, so the search's real value can
    only be measured against actual ladder opponents, opted into per session here rather
    than by flipping the global default. Pure and argparse-free so it's directly
    unit-testable (see tests/test_ladder.py).
    """
    config = PolicyConfig(log_decisions=True)
    if search:
        config = replace(config, use_two_ply_search=True)
    return config


def policy_label(config: PolicyConfig) -> str:
    """Human-readable mode name for startup/summary output, so a ladder session's
    ``runs/ladder.jsonl`` records (which also carry a machine-readable ``"policy"``
    field -- see `LadderPlayer._write_battle_record`) are attributable at a glance.
    """
    return "2-ply search" if config.use_two_ply_search else "myopic evaluator"


async def run_local_smoke(
    *,
    n_games: int,
    team: str,
    opponent: str,
    artifacts_dir: Path,
    log_path: Path,
    config: PolicyConfig,
    timeout_seconds: float = 60.0,
) -> list[dict[str, object]]:
    """Exercise the ladder artifact pipeline using a local direct challenge."""

    session_id = f"local-{_session_id()}"
    player = LadderPlayer(
        artifacts_dir=artifacts_dir,
        log_path=log_path,
        session_id=session_id,
        config=config,
        team=team,
        battle_format=FORMAT_ID,
        accept_open_team_sheet=True,
        server_configuration=LocalhostServerConfiguration,
    )
    anchor = make_player(
        opponent,
        team,
        FORMAT_ID,
        accept_open_team_sheet=True,
        server_configuration=LocalhostServerConfiguration,
    )
    try:
        await asyncio.wait_for(
            player.battle_against(anchor, n_battles=n_games), timeout=timeout_seconds
        )
        await player.flush_finished_battles(delay_seconds=0.1)
    finally:
        await player.ps_client.stop_listening()
        await anchor.ps_client.stop_listening()
    return player.completed_records


async def run_live_session(
    *,
    n_games: int,
    team: str,
    credentials: Credentials,
    artifacts_dir: Path,
    log_path: Path,
    config: PolicyConfig,
    game_timeout_seconds: float,
    max_retries: int,
) -> list[dict[str, object]]:
    """Play one ladder game at a time, recreating the client after connection failures."""

    session_id = _session_id()
    records: list[dict[str, object]] = []
    player: LadderPlayer | None = None
    consecutive_failures = 0
    try:
        while len(records) < n_games:
            if player is None:
                player = LadderPlayer(
                    artifacts_dir=artifacts_dir,
                    log_path=log_path,
                    session_id=session_id,
                    account_configuration=AccountConfiguration(
                        credentials.username, credentials.password
                    ),
                    config=config,
                    team=team,
                    battle_format=FORMAT_ID,
                    accept_open_team_sheet=True,
                    server_configuration=ShowdownServerConfiguration,
                )
            previous_finished = player.n_finished_battles
            previous_records = len(player.completed_records)
            try:
                await asyncio.wait_for(player.ladder(1), timeout=game_timeout_seconds)
                await player.flush_finished_battles(delay_seconds=1.0)
                if player.n_finished_battles <= previous_finished:
                    raise RuntimeError("ladder call returned without a completed battle")
                records.extend(player.completed_records[previous_records:])
                consecutive_failures = 0
            except Exception:  # noqa: BLE001 - reconnect boundary for network failures
                await player.flush_finished_battles()
                consecutive_failures += 1
                player.logger.exception(
                    "Ladder game failed; reconnecting (%d/%d)",
                    consecutive_failures,
                    max_retries,
                )
                await player.ps_client.stop_listening()
                player = None
                if consecutive_failures > max_retries:
                    raise
    finally:
        if player is not None:
            await player.ps_client.stop_listening()
    return records[:n_games]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1, help="number of games")
    parser.add_argument("--team", type=Path, default=TEAMS_DIR / "meta1.packed.txt")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help=(
            "directory for replays/traces (default: "
            f"{DEFAULT_ARTIFACTS_DIR} normally, {DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR} "
            "under --local-smoke)"
        ),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            "append-only outcome JSONL path (default: "
            f"{DEFAULT_LOG_PATH} normally, {DEFAULT_LOCAL_SMOKE_LOG_PATH} "
            "under --local-smoke)"
        ),
    )
    parser.add_argument("--credentials-file", type=Path, default=DEFAULT_CREDENTIALS_FILE)
    parser.add_argument("--game-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--local-smoke",
        action="store_true",
        help="challenge a local baseline instead of joining the public ladder",
    )
    parser.add_argument("--opponent", choices=sorted(BASELINES), default="heuristic")
    parser.add_argument(
        "--search",
        action="store_true",
        help=(
            "enable the Phase 2c 2-ply search (PolicyConfig.use_two_ply_search) for this "
            "session instead of the default myopic evaluator -- an explicit per-session "
            "A/B opt-in against real opponents (see PolicyConfig's comment for why)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n must be at least 1")
    artifacts_dir, log_path = resolve_output_paths(args.local_smoke, args.log, args.artifacts_dir)
    config = session_config(args.search)
    print(f"policy: {policy_label(config)}")
    team = args.team.read_text().strip()
    if args.local_smoke:
        records = asyncio.run(
            run_local_smoke(
                n_games=args.n,
                team=team,
                opponent=args.opponent,
                artifacts_dir=artifacts_dir,
                log_path=log_path,
                config=config,
                timeout_seconds=args.game_timeout,
            )
        )
    else:
        credentials = load_credentials(args.credentials_file)
        records = asyncio.run(
            run_live_session(
                n_games=args.n,
                team=team,
                credentials=credentials,
                artifacts_dir=artifacts_dir,
                log_path=log_path,
                config=config,
                game_timeout_seconds=args.game_timeout,
                max_retries=args.max_retries,
            )
        )
    wins = sum(record.get("won") is True for record in records)
    print(f"completed {len(records)} games: {wins} wins, {len(records) - wins} non-wins")
    print(f"policy: {policy_label(config)}")
    print(f"artifacts: {artifacts_dir}")
    print(f"session log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
