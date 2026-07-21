import asyncio
import concurrent.futures
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import ladder.run_ladder as run_ladder_module
from ladder.run_ladder import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_LOCAL_SMOKE_ARTIFACTS_DIR,
    DEFAULT_LOCAL_SMOKE_LOG_PATH,
    DEFAULT_LOG_PATH,
    PASSWORD_ENV,
    USERNAME_ENV,
    Credentials,
    load_credentials,
    policy_label,
    resolve_output_paths,
    session_config,
)
from ladder.run_ladder import _safe_stop_listening


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


# --- session_config: --search/--bc opt-in map to PolicyConfig.use_two_ply_search/ ----
# --- use_bc_policy, composably ---------------------------------------------------------


def test_session_config_without_search_keeps_default_two_ply_search_off() -> None:
    config = session_config(search=False)
    assert config.use_two_ply_search is False
    assert config.use_bc_policy is False
    assert config.log_decisions is True


def test_session_config_with_search_enables_two_ply_search() -> None:
    config = session_config(search=True)
    assert config.use_two_ply_search is True
    assert config.use_bc_policy is False
    assert config.log_decisions is True


def test_session_config_with_bc_enables_bc_policy() -> None:
    config = session_config(search=False, bc=True)
    assert config.use_two_ply_search is False
    assert config.use_bc_policy is True
    assert config.log_decisions is True


def test_session_config_search_and_bc_are_composable() -> None:
    config = session_config(search=True, bc=True)
    assert config.use_two_ply_search is True
    assert config.use_bc_policy is True


def test_session_config_search_flag_is_the_only_difference() -> None:
    without_search = session_config(search=False)
    with_search = session_config(search=True)
    # dataclasses.replace should only ever touch use_two_ply_search here -- every other
    # field stays at PolicyConfig's own default.
    from dataclasses import replace

    assert with_search == replace(without_search, use_two_ply_search=True)


def test_session_config_bc_flag_is_the_only_difference() -> None:
    without_bc = session_config(search=False)
    with_bc = session_config(search=False, bc=True)
    from dataclasses import replace

    assert with_bc == replace(without_bc, use_bc_policy=True)


def test_policy_label_matches_use_two_ply_search_and_use_bc_policy() -> None:
    assert policy_label(session_config(search=False)) == "myopic evaluator"
    assert policy_label(session_config(search=True)) == "2-ply search"
    assert policy_label(session_config(search=False, bc=True)) == "myopic evaluator + BC re-rank"
    assert policy_label(session_config(search=True, bc=True)) == "2-ply search + BC re-rank"


# --- session_config: --value opt-in maps to PolicyConfig.use_value_head, composably --


def test_session_config_with_value_enables_use_value_head() -> None:
    config = session_config(search=False, value=True)
    assert config.use_value_head is True
    assert config.use_two_ply_search is False
    assert config.use_bc_policy is False


def test_session_config_value_flag_is_the_only_difference() -> None:
    without_value = session_config(search=False)
    with_value = session_config(search=False, value=True)
    from dataclasses import replace

    assert with_value == replace(without_value, use_value_head=True)


def test_session_config_search_bc_value_are_all_composable() -> None:
    config = session_config(search=True, bc=True, value=True)
    assert config.use_two_ply_search is True
    assert config.use_bc_policy is True
    assert config.use_value_head is True


def test_policy_label_includes_value_head() -> None:
    assert policy_label(session_config(search=True, value=True)) == "2-ply search + value head"
    assert (
        policy_label(session_config(search=True, bc=True, value=True))
        == "2-ply search + BC re-rank + value head"
    )
    assert policy_label(session_config(search=False, value=True)) == "myopic evaluator + value head"


def test_policy_tag_covers_all_four_combos() -> None:
    policy_tag = run_ladder_module._policy_tag
    assert policy_tag(session_config(search=False)) == "myopic"
    assert policy_tag(session_config(search=True)) == "search"
    assert policy_tag(session_config(search=False, bc=True)) == "bc"
    assert policy_tag(session_config(search=True, bc=True)) == "search+bc"


def test_policy_tag_covers_value_head_combos() -> None:
    policy_tag = run_ladder_module._policy_tag
    assert policy_tag(session_config(search=False, value=True)) == "value"
    assert policy_tag(session_config(search=True, value=True)) == "search+value"
    assert policy_tag(session_config(search=False, bc=True, value=True)) == "bc+value"
    assert policy_tag(session_config(search=True, bc=True, value=True)) == "search+bc+value"


# --- _safe_stop_listening: teardown of a dead connection must never raise -----------


class _FakePsClient:
    def __init__(self, stop_listening):
        self.stop_listening = stop_listening


class _FakePlayer:
    def __init__(self, stop_listening):
        self.ps_client = _FakePsClient(stop_listening)
        self.logger = logging.getLogger("test-fake-ladder-player")


def test_safe_stop_listening_swallows_timeout_error() -> None:
    async def _raising_stop_listening():
        raise TimeoutError("timed out while closing connection")

    player = _FakePlayer(_raising_stop_listening)

    async def _run():
        await _safe_stop_listening(player)

    asyncio.run(_run())  # must not raise


def test_safe_stop_listening_passes_through_on_clean_close() -> None:
    calls = []

    async def _clean_stop_listening():
        calls.append("called")

    player = _FakePlayer(_clean_stop_listening)

    async def _run():
        await _safe_stop_listening(player)

    asyncio.run(_run())
    assert calls == ["called"]


# --- ladder wait/reconnect: quiet matchmaking is not a game timeout ----------------


class _WaitTestPlayer:
    def __init__(self, ladder_impl) -> None:
        self.battles = {}
        self._ladder_impl = ladder_impl
        self.ps_client = SimpleNamespace(_listening_coroutine=concurrent.futures.Future())

    async def ladder(self, n_games: int) -> None:
        await self._ladder_impl(self, n_games)


def test_await_one_ladder_game_does_not_time_out_during_matchmaking() -> None:
    async def _ladder(player, _n_games: int) -> None:
        # Stay in a healthy queue longer than the active-battle timeout. The clock must
        # not start until the battle appears.
        await asyncio.sleep(0.03)
        player.battles["battle-test-1"] = SimpleNamespace(finished=False)
        await asyncio.sleep(0.005)

    player = _WaitTestPlayer(_ladder)

    async def _run() -> None:
        await run_ladder_module._await_one_ladder_game(
            player, battle_timeout_seconds=0.01, poll_seconds=0.001
        )

    asyncio.run(_run())


def test_await_one_ladder_game_times_out_after_battle_starts() -> None:
    cancelled = []

    async def _ladder(player, _n_games: int) -> None:
        player.battles["battle-test-2"] = SimpleNamespace(finished=False)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    player = _WaitTestPlayer(_ladder)

    async def _run() -> None:
        with pytest.raises(TimeoutError, match="battle exceeded"):
            await run_ladder_module._await_one_ladder_game(
                player, battle_timeout_seconds=0.01, poll_seconds=0.001
            )

    asyncio.run(_run())
    assert cancelled == [True]


def test_await_one_ladder_game_detects_stopped_listener_while_searching() -> None:
    async def _ladder(_player, _n_games: int) -> None:
        await asyncio.Event().wait()

    player = _WaitTestPlayer(_ladder)
    player.ps_client._listening_coroutine.set_result(None)

    async def _run() -> None:
        with pytest.raises(ConnectionError, match="listener stopped"):
            await run_ladder_module._await_one_ladder_game(
                player, battle_timeout_seconds=1.0, poll_seconds=0.001
            )

    asyncio.run(_run())


def test_deduplicate_poke_env_stream_handlers_preserves_custom_handlers() -> None:
    logger = logging.getLogger("test-poke-env-handler-deduplication")
    original_handlers = list(logger.handlers)
    logger.handlers.clear()
    try:
        poke_handlers = [logging.StreamHandler(), logging.StreamHandler()]
        for handler in poke_handlers:
            handler.setFormatter(logging.Formatter(run_ladder_module.POKE_ENV_LOG_FORMAT))
            logger.addHandler(handler)
        custom_handler = logging.StreamHandler()
        custom_handler.setFormatter(logging.Formatter("custom: %(message)s"))
        logger.addHandler(custom_handler)

        run_ladder_module._deduplicate_poke_env_stream_handlers(logger)

        assert logger.handlers == [poke_handlers[0], custom_handler]
    finally:
        logger.handlers.clear()
        for handler in original_handlers:
            logger.addHandler(handler)


# --- run_live_session: a teardown that raises must not abort the retry loop ---------


class _FakeLadderPlayer:
    """Stands in for `ladder.run_ladder.LadderPlayer`: the first `ladder()` call
    simulates a dead websocket (raises, like the real crash log), the second succeeds.
    `attempt_count` is a CLASS attribute (reset per test) because `run_live_session`
    constructs a brand-new instance on reconnect -- the "first call fails" behavior has
    to survive across that reconstruction, same as the real bug did.
    """

    attempt_count = 0

    def __init__(self, **kwargs) -> None:
        self.config = kwargs.get("config")
        self.completed_records: list[dict[str, object]] = []
        self.n_finished_battles = 0
        self.battles = {}
        self.logger = logging.getLogger("test-fake-ladder-player")
        self.logger.setLevel(logging.CRITICAL)  # keep the simulated failure quiet
        self.ps_client = SimpleNamespace(
            stop_listening=self._stop_listening,
            _listening_coroutine=concurrent.futures.Future(),
        )

    async def ladder(self, _n: int) -> None:
        type(self).attempt_count += 1
        if type(self).attempt_count == 1:
            raise TimeoutError("simulated dead websocket (keepalive ping timeout)")
        self.n_finished_battles += 1
        self.completed_records.append(
            {"won": True, "battle_tag": f"fake-{type(self).attempt_count}"}
        )

    async def flush_finished_battles(self, delay_seconds: float = 0.0) -> None:
        return None

    async def _stop_listening(self) -> None:
        # Mirrors the confirmed crash: tearing down an already-dead connection raises.
        raise TimeoutError("timed out while closing connection")


def test_run_live_session_survives_a_teardown_that_raises(monkeypatch) -> None:
    _FakeLadderPlayer.attempt_count = 0
    monkeypatch.setattr(run_ladder_module, "LadderPlayer", _FakeLadderPlayer)
    monkeypatch.setattr(
        run_ladder_module,
        "AccountConfiguration",
        lambda username, password: SimpleNamespace(username=username, password=password),
    )

    async def _run():
        return await run_ladder_module.run_live_session(
            n_games=1,
            team="fake-team",
            credentials=Credentials(username="u", password="p"),
            artifacts_dir=Path("/tmp/fake-artifacts"),
            log_path=Path("/tmp/fake-log.jsonl"),
            config=None,
            game_timeout_seconds=5.0,
            max_retries=2,
        )

    # Before the fix, the dead-connection stop_listening() inside the except block (or
    # the finally block on the way out) would raise and this call would never return.
    records = asyncio.run(_run())

    assert len(records) == 1
    assert records[0]["won"] is True
    assert _FakeLadderPlayer.attempt_count == 2  # one simulated failure, one recovery


class _FakeRecoveredRecordPlayer(_FakeLadderPlayer):
    """Simulate the real race: a record is flushed before ``ladder()`` raises."""

    attempt_count = 0

    async def ladder(self, _n: int) -> None:
        type(self).attempt_count += 1
        self.n_finished_battles += 1
        self.completed_records.append({"won": False, "battle_tag": "recovered-1"})
        raise TimeoutError("simulated late ladder failure")


def test_run_live_session_counts_record_recovered_during_failure(monkeypatch) -> None:
    _FakeRecoveredRecordPlayer.attempt_count = 0
    monkeypatch.setattr(run_ladder_module, "LadderPlayer", _FakeRecoveredRecordPlayer)
    monkeypatch.setattr(
        run_ladder_module,
        "AccountConfiguration",
        lambda username, password: SimpleNamespace(username=username, password=password),
    )

    async def _run():
        return await run_ladder_module.run_live_session(
            n_games=1,
            team="fake-team",
            credentials=Credentials(username="u", password="p"),
            artifacts_dir=Path("/tmp/fake-artifacts"),
            log_path=Path("/tmp/fake-log.jsonl"),
            config=None,
            game_timeout_seconds=5.0,
            max_retries=0,
        )

    records = asyncio.run(_run())

    assert records == [{"won": False, "battle_tag": "recovered-1"}]
    assert _FakeRecoveredRecordPlayer.attempt_count == 1
