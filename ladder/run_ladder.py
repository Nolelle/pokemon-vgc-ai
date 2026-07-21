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

The first-principles robust-response search (``PolicyConfig.use_two_ply_search``) is the
default. ``--myopic`` opts a diagnostic session back into the old one-turn evaluator;
``--search`` remains as a backwards-compatible explicit spelling of the default. ``--bc``
independently opts into the BC v2 candidate re-ranker (``PolicyConfig.use_bc_policy``,
also False by default -- see its comment in ``vgc/models.py``) on top of whichever of
those two an evaluator path (search or myopic) is running. ``--value`` independently
opts into the Phase 3 outcome value head (``PolicyConfig.use_value_head``, also False by
default) -- only actually consumed inside the 2-ply search, so it's a no-op without
``--search``. Every ``runs/ladder.jsonl`` record carries a ``"policy"`` field
(``"myopic"`` | any ``"+"``-joined combo of ``"search"``/``"bc"``/``"value"``, e.g.
``"search+bc"`` or ``"search+bc+value"``) so sessions stay attributable after the fact;
the running mode is also printed at startup and in the session summary (see
``session_config``/``policy_label``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Ladder sessions always retain the reasoning needed for post-loss review. The trace
# module reads this dynamically, so this remains effective even when imported in tests.
os.environ["VGC_TRACE"] = "1"

from poke_env.battle.abstract_battle import AbstractBattle  # noqa: E402
from poke_env.player.player import Player  # noqa: E402
from poke_env.ps_client.account_configuration import AccountConfiguration  # noqa: E402
from poke_env.ps_client.server_configuration import (  # noqa: E402
    LocalhostServerConfiguration,
    ShowdownServerConfiguration,
)

from vgc.agent import VgcPlayer  # noqa: E402
from vgc.baselines import BASELINES, make_player  # noqa: E402
from vgc.config import FORMAT_ID, RUNS_DIR, TEAMS_DIR  # noqa: E402
from vgc.models import PolicyConfig  # noqa: E402
from vgc.postmortem import classify_loss  # noqa: E402

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
POKE_ENV_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LADDER_PROGRESS_POLL_SECONDS = 1.0


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
        _deduplicate_poke_env_stream_handlers(self.logger)

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
            # Machine-readable A/B tag for the evaluator plus optional BC/value layers.
            "policy": _policy_tag(self.config),
            "trace_path": str(trace_path.resolve()),
            "replay_path": str(replay_path.resolve()) if replay_path else None,
        }
        if battle.lost:
            record["loss_classification"] = classify_loss(traces)
        with self.log_path.open("a") as log_file:
            log_file.write(json.dumps(record, sort_keys=True) + "\n")
        self.completed_records.append(record)


def _session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


async def _safe_stop_listening(player: Player, timeout_seconds: float = 15.0) -> None:
    """Best-effort websocket teardown. A connection that already died server-side (e.g.
    a keepalive ping timeout) can make `stop_listening()` itself hang or raise
    `TimeoutError` ("timed out while closing connection") -- during reconnect/cleanup we
    don't care, we're discarding this client regardless, so swallow (and log) any
    failure here rather than let it propagate and abort session-level retry logic.
    """
    try:
        await asyncio.wait_for(player.ps_client.stop_listening(), timeout=timeout_seconds)
    except Exception:  # noqa: BLE001 - teardown of a dead client must never propagate
        player.logger.debug(
            "stop_listening during teardown failed; discarding client anyway", exc_info=True
        )


def _deduplicate_poke_env_stream_handlers(logger: logging.Logger) -> None:
    """Keep one copy of poke-env's console handler for a reused account logger.

    ``PSClient`` adds a new ``StreamHandler`` every time a client is constructed, while
    Python returns the same logger object for every reconnect under one username. Without
    cleanup, the same network error is printed once per client created during the session.
    Leave custom handlers alone; only collapse handlers with poke-env's exact formatter.
    """

    matching_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and handler.formatter is not None
        and handler.formatter._fmt == POKE_ENV_LOG_FORMAT
    ]
    for handler in matching_handlers[1:]:
        logger.removeHandler(handler)
        handler.close()


def _listener_stopped(player: Player) -> bool:
    """Return whether poke-env's websocket listener has exited."""

    listener = getattr(player.ps_client, "_listening_coroutine", None)
    return listener is not None and listener.done()


async def _await_one_ladder_game(
    player: LadderPlayer,
    *,
    poll_seconds: float = LADDER_PROGRESS_POLL_SECONDS,
) -> None:
    """Wait through matchmaking and battle while the websocket listener is healthy.

    Neither public matchmaking nor an active timer-controlled Showdown battle has a safe
    client-side wall-clock limit. Applying ``asyncio.wait_for`` to either can orphan a
    search or disconnect a battle that the server still considers active. The websocket
    listener exiting is the reconnect signal; Showdown's battle timer bounds actual play.
    """

    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    ladder_task = asyncio.create_task(player.ladder(1))
    try:
        while True:
            if ladder_task.done():
                await ladder_task
                return

            if _listener_stopped(player):
                raise ConnectionError("Showdown websocket listener stopped during ladder search")

            await asyncio.wait({ladder_task}, timeout=poll_seconds)
    finally:
        if not ladder_task.done():
            ladder_task.cancel()
            with suppress(asyncio.CancelledError):
                await ladder_task


def _append_completed_records(records: list[dict[str, object]], player: LadderPlayer) -> int:
    """Append each flushed battle record once and return the number added."""

    seen_battle_tags = {record.get("battle_tag") for record in records}
    added = 0
    for record in player.completed_records:
        battle_tag = record.get("battle_tag")
        if battle_tag in seen_battle_tags:
            continue
        records.append(record)
        seen_battle_tags.add(battle_tag)
        added += 1
    return added


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


def session_config(search: bool = True, bc: bool = False, value: bool = False) -> PolicyConfig:
    """The `PolicyConfig` for one ladder session (smoke or live): the default config
    (robust-response search), optionally changed to the diagnostic myopic path
    (`--myopic`) and composably extended with the BC v2 candidate re-ranker (`--bc`)
    and/or the outcome value head
    (`--value`). Search is now required by the first-principles policy and defaults on;
    BC/value remain opt-in. `--value` is only actually CONSUMED inside
    `vgc.search.search_joint_orders` (see `PolicyConfig.use_value_head`'s comment), so
    it composes freely but is a no-op without `--search` -- same "orthogonal flags,
    some combinations degrade to a no-op" contract `--bc` already has (e.g. a value-only
    checkpoint at `bc_checkpoint_path` would make `--bc` itself a no-op the same way).
    Pure and argparse-free so it's directly unit-testable (see tests/test_ladder.py).
    """
    config = replace(PolicyConfig(log_decisions=True), use_two_ply_search=search)
    if bc:
        config = replace(config, use_bc_policy=True)
    if value:
        config = replace(config, use_value_head=True)
    return config


def _policy_tag(config: PolicyConfig) -> str:
    """Machine-readable ``"policy"`` field for ``runs/ladder.jsonl`` records: ``"myopic"``
    when no opt-in knob is engaged, else ``"+"``-joined tags for whichever of
    ``search``/``bc``/``value`` are (in that order, e.g. ``"search+bc"``,
    ``"search+value"``, ``"bc+value"``, ``"search+bc+value"``). ``"myopic"`` alone
    (no flags) matches the pre-``--bc``/``--value`` tag exactly, so old records stay
    comparable.
    """
    parts = []
    if config.use_two_ply_search:
        parts.append("search")
    if config.use_bc_policy:
        parts.append("bc")
    if config.use_value_head:
        parts.append("value")
    return "+".join(parts) if parts else "myopic"


def policy_label(config: PolicyConfig) -> str:
    """Human-readable mode name for startup/summary output, so a ladder session's
    ``runs/ladder.jsonl`` records (which also carry a machine-readable ``"policy"``
    field -- see `_policy_tag`/`LadderPlayer._write_battle_record`) are attributable at a
    glance.
    """
    base = "2-ply search" if config.use_two_ply_search else "myopic evaluator"
    extras = []
    if config.use_bc_policy:
        extras.append("BC re-rank")
    if config.use_value_head:
        extras.append("value head")
    if extras:
        return f"{base} + {' + '.join(extras)}"
    return base


async def run_local_smoke(
    *,
    n_games: int,
    team: str,
    opponent: str,
    artifacts_dir: Path,
    log_path: Path,
    config: PolicyConfig | None = None,
    timeout_seconds: float = 60.0,
) -> list[dict[str, object]]:
    """Exercise the ladder artifact pipeline using a local direct challenge."""

    config = config or PolicyConfig(log_decisions=True)
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
        # Same dead-connection teardown hazard as run_live_session: guard both closes so a
        # hung/raising stop_listening() can't mask the real result (or a real exception).
        await _safe_stop_listening(player)
        await _safe_stop_listening(anchor)
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
    """Play one ladder game at a time, recreating the client after connection failures.

    ``game_timeout_seconds`` is retained for API/CLI compatibility but intentionally does
    not bound a public search or battle. A quiet queue and a timer-controlled active game
    are both healthy; a stopped websocket listener is the reconnect signal while waiting.
    """

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
            try:
                await _await_one_ladder_game(player)
                await player.flush_finished_battles(delay_seconds=1.0)
                if player.n_finished_battles <= previous_finished:
                    raise RuntimeError("ladder call returned without a completed battle")
                _append_completed_records(records, player)
                consecutive_failures = 0
            except Exception:  # noqa: BLE001 - reconnect boundary for network failures
                try:
                    await player.flush_finished_battles()
                except Exception:  # noqa: BLE001 - the connection that failed above is
                    # often the same one flush_finished_battles would need -- never let
                    # a flush-on-the-way-out mask the real failure or abort the retry.
                    player.logger.debug(
                        "flush_finished_battles during teardown failed", exc_info=True
                    )
                recovered_records = _append_completed_records(records, player)
                if recovered_records:
                    consecutive_failures = 0
                    player.logger.exception(
                        "Ladder call failed after %d completed battle(s) were recovered; "
                        "reconnecting",
                        recovered_records,
                    )
                else:
                    consecutive_failures += 1
                    player.logger.exception(
                        "Ladder game failed; reconnecting (%d/%d)",
                        consecutive_failures,
                        max_retries,
                    )
                dead_player = player
                player = None  # discard BEFORE teardown -- a hung/raising
                # stop_listening() on an already-dead connection must never leave
                # `player` looking "still alive" to the retry loop above.
                await _safe_stop_listening(dead_player)
                if not recovered_records and consecutive_failures > max_retries:
                    raise
    finally:
        if player is not None:
            await _safe_stop_listening(player)
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
    parser.add_argument(
        "--game-timeout",
        type=float,
        default=300.0,
        help=(
            "local-smoke timeout in seconds; public ladder searches and timer-controlled "
            "battles wait while their websocket is healthy"
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help=(
            "consecutive reconnects allowed before giving up (resets to 0 after any "
            "completed game, so this bounds a losing STREAK of connection failures, "
            "not a lifetime/cumulative count across the whole session)"
        ),
    )
    parser.add_argument(
        "--local-smoke",
        action="store_true",
        help="challenge a local baseline instead of joining the public ladder",
    )
    parser.add_argument("--opponent", choices=sorted(BASELINES), default="heuristic")
    parser.add_argument(
        "--search",
        action="store_true",
        default=True,
        help=(
            "use the first-principles robust-response search (now the default; retained "
            "as a backwards-compatible explicit flag)"
        ),
    )
    parser.add_argument(
        "--myopic",
        action="store_false",
        dest="search",
        help="diagnostic opt-out: use the old one-turn evaluator without response search",
    )
    parser.add_argument(
        "--bc",
        action="store_true",
        help=(
            "enable the BC v2 candidate re-ranker (PolicyConfig.use_bc_policy) for this "
            "session, composable with --search -- an explicit per-session A/B opt-in "
            "against real opponents (see PolicyConfig's comment for why)"
        ),
    )
    parser.add_argument(
        "--value",
        action="store_true",
        help=(
            "enable the outcome value head (PolicyConfig.use_value_head) for this "
            "session, composable with --search/--bc -- only actually consumed inside "
            "the 2-ply search, so it's a no-op without --search (see PolicyConfig's "
            "comment for why)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n < 1:
        raise ValueError("--n must be at least 1")
    artifacts_dir, log_path = resolve_output_paths(args.local_smoke, args.log, args.artifacts_dir)
    config = session_config(args.search, args.bc, args.value)
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
