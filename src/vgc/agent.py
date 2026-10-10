"""Base poke-env Player for this project.

`VgcPlayer` wires `PolicyConfig` into poke-env's `Player` constructor and gives
subclasses a single overridable `decide()`/`decide_teampreview()` hook instead of the
raw `choose_move`/`teampreview` API -- `choose_move`/`teampreview` themselves stay final
in spirit (not literally, poke-env doesn't support that) and exist only to wrap the hook
in a try/except: a bug in `decide()` must never crash or forfeit a battle, it should just
fall back to a legal random move and log the exception (mirrors pokemon-tcg-ai's
`src/agent_runtime.decide()` exception-safe fallback contract).
"""

from __future__ import annotations

import asyncio
import contextvars
import random
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from poke_env.player.player import Player

from vgc.actions import (
    choice_wire_message,
    describe_order,
    index_locked_choice,
    index_switch_choice,
)
from vgc.battle_state_replay import DecisionReplayRecorder
from vgc.battle_memory import BattleMemory
from vgc.bc.policy import load_bc_policy, score_orders
from vgc.clock import (
    Budget,
    ClockTracker,
    WorkerSlot,
    bind_cancel,
    bind_deadline,
    budget_seconds,
    cancelled,
    run_with_deadline,
)
from vgc.decision_trace import (
    current_trace,
    finish_trace,
    record_fallback,
    record_note,
    start_trace,
    trace_enabled,
)
from vgc.evaluator import breakdown_for_order, score_joint_orders
from vgc.condition_clock import observe_condition_line
from vgc.team_scope import bind_own_team
from vgc.poke_env_compat import normalize_for_poke_env
from vgc.config import REPO_ROOT, SHOWDOWN_REPO
from vgc.models import PolicyConfig
from vgc.opponent_belief import information_boundary_summary
from vgc.own_team import apply_own_spreads
from vgc.search import search_joint_orders
from vgc.team_preview import build_team_order


def _move_kind(battle: AbstractBattle) -> str:
    """Clock-guard decision kind: forced switch, endgame ("critical"), or normal."""

    forced = getattr(battle, "force_switch", False)
    if any(forced) if isinstance(forced, list) else forced:
        return "forced_switch"
    try:
        ours = sum(not mon.fainted for mon in battle.team.values())
        # The opponent's team is only partly revealed; count what is known to be alive
        # against the full bring of four.
        theirs = 4 - sum(mon.fainted for mon in battle.opponent_team.values())
    except Exception:  # noqa: BLE001 - classification must never break a decision
        return "normal"
    return "critical" if min(ours, theirs) <= 2 else "normal"


class VgcPlayer(Player):
    """poke-env Player driven by a `PolicyConfig`.

    Phase 2b: `decide()`'s default implementation is the real heuristic evaluator
    (`vgc.evaluator.score_joint_orders`, gated behind
    `PolicyConfig.use_heuristic_evaluator` so it can still be A/B'd against plain random
    play from offline/run_gates.py without swapping baselines); `decide_teampreview()`'s
    default is `vgc.team_preview.build_team_order`. Subclasses override `decide()` /
    `decide_teampreview()` to add different strategy; they never need to touch
    `choose_move` or `teampreview` directly.
    """

    def __init__(self, config: PolicyConfig | None = None, **player_kwargs) -> None:
        self.config = config or PolicyConfig()
        record_decision_replays = bool(player_kwargs.pop("record_decision_replays", False))
        supplied_team = player_kwargs.get("team")
        # The exact packed team string, for the live exact judge's public mirror
        # (`PolicyConfig.exact_judge_live`); None when the team was not supplied as text.
        self._own_packed_team = supplied_team if isinstance(supplied_team, str) else None
        self._decision_replay_recorder = (
            DecisionReplayRecorder(
                own_packed_team=supplied_team if isinstance(supplied_team, str) else None,
                config=self.config,
                repo_root=REPO_ROOT,
                showdown_repo=SHOWDOWN_REPO,
            )
            if record_decision_replays
            else None
        )
        # poke-env 0.15 can receive an opponent's OTS rejection before its team-preview
        # request. Remember that room until the request arrives so the battle does not
        # wait forever for a `showteam` message that Showdown will never send.
        self._pending_ots_rejections: set[str] = set()
        self._resolved_ots_rejections: set[str] = set()
        self._battle_memories: dict[str, BattleMemory] = {}
        self.decision_trace_history: list[dict[str, object]] = []
        # Always-on safety counter for automated gates. Detailed traces remain opt-in,
        # but a gate must still detect that inference crashed and random fallback play
        # was used when VGC_TRACE is unset.
        self.fallback_count = 0
        # Clock guard (vgc.clock): per-battle timer tracking, plus an always-on log of
        # one record per decision (elapsed_ms, budget_s, bank_left_s, decision_kind,
        # fallback_reason). A player that asks Showdown to start the timer itself knows a
        # fresh bank is ticking even before the first "Time left" line arrives.
        self._clock_trackers: dict[str, ClockTracker] = {}
        self._assume_timer_on = bool(player_kwargs.get("start_timer_on_battle_start", False))
        self.clock_log: list[dict[str, object]] = []
        player_kwargs.setdefault("battle_format", self.config.format_id)
        player_kwargs.setdefault("accept_open_team_sheet", self.config.accept_open_team_sheet)
        super().__init__(**player_kwargs)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ) -> None:
        """Choose outside poke-env's socket loop so exact search cannot kill heartbeats.

        poke-env normally calls the synchronous ``choose_move`` method directly inside
        its shared asyncio loop. A mechanics-exact search can take tens of seconds,
        preventing every client on that loop from answering its WebSocket heartbeat.
        ``asyncio.to_thread`` keeps the battle request serialized while allowing network
        I/O and connection-failure detection to continue normally.
        """

        if battle._wait:
            self._waiting.set()
            return
        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
        elif battle.teampreview:
            preview = await asyncio.to_thread(self.teampreview, battle)
            if isinstance(preview, Awaitable):
                preview = await preview
            message = preview
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = await asyncio.to_thread(self.choose_move, battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message
        if message:
            message = index_switch_choice(battle, index_locked_choice(battle, message))
            await self.ps_client.send_message(message, battle.battle_tag)

    async def _handle_battle_message(self, split_messages) -> None:
        """Consume one protocol line at a time so a decision sees only its prefix.

        poke-env accepts a multiline websocket burst and invokes ``choose_move`` as soon
        as it reaches a ``request`` line.  BattleMemory used to consume the whole burst
        before poke-env parsed any of it, allowing later lines in that same burst to
        enter an earlier decision.  Serial processing makes the observation cutoff
        explicit and also supplies the exact stream recorded by Part B replay bundles.
        """

        if not split_messages:
            return
        room = split_messages[0]
        room_marker = room[0] if room else ""
        battle_tag = room_marker[1:] if room_marker.startswith(">") else room_marker
        for message in split_messages[1:]:
            recorder = getattr(self, "_decision_replay_recorder", None)
            if recorder is not None and battle_tag:
                recorder.observe(battle_tag, message)
            if battle_tag:
                self._memory_for_tag(battle_tag).observe_protocol([message])
                if len(message) > 1 and message[1] in ("inactive", "inactiveoff"):
                    self._clock_for_tag(battle_tag).observe(message)
                elif len(message) > 2 and message[1] == "request" and message[2]:
                    payload = "|".join(message[2:])
                    # "wait" requests never start our clock; "update" requests are
                    # resends of the same server request and do not restart it.
                    if '"wait":true' not in payload:
                        self._clock_for_tag(battle_tag).note_request(
                            update='"update":true' in payload
                        )
            await self._handle_battle_message_line([room, normalize_for_poke_env(message)])
            battles = getattr(self, "_battles", None)
            if battle_tag and battles:
                observe_condition_line(battles.get(battle_tag), message)

    async def _handle_battle_message_line(self, split_messages) -> None:
        """Work around poke-env 0.15's Open Team Sheets accept/reject race.

        Upstream defers team-preview choice while an OTS-accepting player waits for the
        opponent's ``showteam`` message. If the rejection text arrives before the
        ``request`` message has set ``battle.teampreview``, upstream misses the rejection
        and later waits forever. Showdown may alternatively return an ``error`` saying
        the opponent already rejected. In either order, remember the rejection and send
        our team-preview choice once the request has actually been parsed.
        """

        room_marker = split_messages[0][0] if split_messages and split_messages[0] else ""
        battle_tag = room_marker[1:] if room_marker.startswith(">") else room_marker
        rejection_was_pending = battle_tag in self._pending_ots_rejections
        rejection_positions: list[int] = []
        request_positions: list[int] = []
        saw_rejection_error = False

        for index, message in enumerate(split_messages[1:], start=1):
            joined = "|".join(message).lower()
            if "rejected open team sheets." in joined:
                rejection_positions.append(index)
                if len(message) > 1 and message[1] == "error":
                    saw_rejection_error = True
            if len(message) > 1 and message[1] == "request":
                request_positions.append(index)

        if rejection_positions and battle_tag not in self._resolved_ots_rejections:
            self._pending_ots_rejections.add(battle_tag)

        await super()._handle_battle_message(split_messages)

        # Fill in our OWN Stat Points/nature from the team we supplied. poke-env only
        # ever learns them from an Open Team Sheets `showteam`, which on the public
        # ladder essentially never arrives (~0.2% of replays), leaving the evaluator to
        # guess our own spread with `default_opponent_spread` -- off by up to 35.6% and
        # underestimating Speed on every Pokemon. See `vgc.own_team`. Must run BEFORE
        # the OTS early-return below, since the no-OTS case is exactly the one that
        # needs it. A no-op when a showteam did arrive and poke-env already filled them.
        #
        # Accurate self-knowledge exposed an old Protect miscalibration; that weight has
        # since been retuned and confirmed above 50% against the legacy fake-spread
        # policy. See PolicyConfig for the screen and held-out confirmation numbers.
        if battle_tag and self.config.use_own_team_spreads:
            own_battle = self._battles.get(battle_tag)
            if own_battle is not None:
                apply_own_spreads(own_battle)

        if not self.accept_open_team_sheet or battle_tag in self._resolved_ots_rejections:
            return

        battle = self._battles.get(battle_tag)
        saw_plain_rejection = bool(rejection_positions) and not saw_rejection_error
        if (
            saw_plain_rejection
            and battle is not None
            and battle.teampreview
            and not request_positions
        ):
            # Upstream's plain-text branch handled this ordering itself.
            self._pending_ots_rejections.discard(battle_tag)
            self._resolved_ots_rejections.add(battle_tag)
            return
        if battle_tag not in self._pending_ots_rejections:
            return

        # If rejection followed a request in this same batch, upstream already resumed
        # team preview from its plain-text rejection branch. Recover only when the
        # rejection was known earlier, preceded the request, or arrived as an error (an
        # error is only logged by upstream and never resumes the request).
        rejection_preceded_request = bool(
            request_positions
            and rejection_positions
            and min(rejection_positions) < max(request_positions)
        )
        should_recover = (
            saw_rejection_error
            or (bool(request_positions) and rejection_was_pending)
            or rejection_preceded_request
        )
        if should_recover and battle is not None and battle.teampreview:
            self._pending_ots_rejections.discard(battle_tag)
            self._resolved_ots_rejections.add(battle_tag)
            await self._handle_battle_request(battle)

    # --- overridable hooks --------------------------------------------------------

    def _memory_for_tag(self, battle_tag: str) -> BattleMemory:
        # Lazy initialization keeps tests that intentionally construct ``VgcPlayer``
        # through ``__new__`` (without running ``__init__``) working.
        memories = getattr(self, "_battle_memories", None)
        if memories is None:
            memories = {}
            self._battle_memories = memories
        if battle_tag not in memories:
            memories[battle_tag] = BattleMemory(battle_tag=battle_tag)
        return memories[battle_tag]

    def _clock_for_tag(self, battle_tag: str) -> ClockTracker:
        # Lazy for the same reason as `_memory_for_tag` (tests build players via __new__).
        trackers = getattr(self, "_clock_trackers", None)
        if trackers is None:
            trackers = {}
            self._clock_trackers = trackers
        if battle_tag not in trackers:
            trackers[battle_tag] = ClockTracker(
                assume_full_bank=bool(getattr(self, "_assume_timer_on", False))
            )
        return trackers[battle_tag]

    def _worker_slot(self, battle: AbstractBattle | None = None) -> WorkerSlot:
        """The player's single decision-worker slot.

        Given the battle about to decide, a slot still held by a worker of a DIFFERENT battle
        that is already over (for example a slow LLM team-preview call whose opponent left)
        is abandoned: that worker is cancelled -- it can no longer write memory or samples,
        see `vgc.clock.cancelled` -- and a fresh slot replaces it, so it cannot make the next
        battle's preview or first turns fall back with ``previous-worker-busy``. A worker of a
        live battle (including the same battle's earlier turn) still blocks, as before."""

        slot = getattr(self, "_decide_slot", None)
        if slot is None:
            slot = WorkerSlot()
            self._decide_slot = slot
        if battle is not None and slot.busy():
            owner = slot.owner
            if (
                owner is not None
                and owner != battle.battle_tag
                and self._battle_is_over(owner)
            ):
                if slot.cancel is not None:
                    slot.cancel.set()
                slot = WorkerSlot()
                self._decide_slot = slot
        return slot

    def _battle_is_over(self, battle_tag: str) -> bool:
        tracked = getattr(self, "_battles", None)
        if not isinstance(tracked, dict):
            return False
        battle = tracked.get(battle_tag)
        return battle is None or bool(getattr(battle, "finished", False))

    def _memory_for(self, battle: AbstractBattle) -> BattleMemory:
        memory = self._memory_for_tag(battle.battle_tag)
        memory.observe_battle(battle)
        try:
            setattr(battle, "_vgc_battle_memory", memory)
        except (AttributeError, TypeError):
            pass
        return memory

    def _record_decision(self, battle: AbstractBattle, *, team_preview: bool) -> int | None:
        recorder = getattr(self, "_decision_replay_recorder", None)
        if recorder is None:
            return None
        # The decision snapshot must include our known nature and Stat Points even on
        # the very first team-preview request.  The transport-level enrichment after
        # ``super()._handle_battle_message`` is too late for a recorder invoked inside
        # poke-env's request callback.
        if self.config.use_own_team_spreads:
            apply_own_spreads(battle)
        return recorder.record_decision(
            battle,
            self._memory_for(battle),
            team_preview=team_preview,
        )

    def decision_replay_bundle(self, battle: AbstractBattle) -> dict[str, object] | None:
        recorder = getattr(self, "_decision_replay_recorder", None)
        if recorder is None or not recorder.has_battle(battle.battle_tag):
            return None
        return recorder.bundle(
            battle.battle_tag,
            player_side=getattr(battle, "player_role", None),
            player_username=getattr(battle, "player_username", None),
        )

    def decide(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move: the argmax of `vgc.search.search_joint_orders` (opponent
        response search plus the gated rolling position forecast) when
        `PolicyConfig.use_two_ply_search` is set
        (True by default; `ladder/run_ladder.py --myopic` is the diagnostic opt-out),
        falling back to the plain myopic
        `vgc.evaluator.score_joint_orders` when the search is disabled but the heuristic
        evaluator (`PolicyConfig.use_heuristic_evaluator`, the actual default) is still
        on, or a random legal move if both are disabled or there's nothing to score
        (e.g. `enumerate_joint_orders` came back empty -- see its own docstring for when
        that happens). Only wired up for `DoubleBattle` (this project's format is always
        doubles -- see vgc/config.py's FORMAT_ID); any other battle type falls back to
        random rather than guessing.

        When `PolicyConfig.use_bc_policy` is set, the scored list from whichever path
        above ran (search or myopic) is then passed through `vgc.bc.policy.score_orders`,
        which blends the trained BC v2 checkpoint's learned move/target log-probability
        into the top-ranked candidates before the argmax is taken (see that module's
        docstring). `load_bc_policy` is cached and itself never raises -- a missing
        checkpoint/torch install just leaves the scored list unchanged, same as this
        whole method's outer exception-safe wrapper (`choose_move`) already guarantees
        for any other failure here.
        """
        memory = self._memory_for(battle)
        if trace_enabled():
            record_note(
                "information_boundary",
                information_boundary_summary(battle, memory, self.config),
            )
        scored: list = []
        if self.config.use_two_ply_search and isinstance(battle, DoubleBattle):
            scored = self._search(battle, memory)
        elif self.config.use_heuristic_evaluator and isinstance(battle, DoubleBattle):
            scored = score_joint_orders(battle, self.config)
        if scored:
            if self.config.use_bc_policy:
                policy = load_bc_policy(self.config.bc_checkpoint_path)
                scored = score_orders(policy, battle, scored, self.config)
            if self.config.log_decisions:
                record_note("chosen_order_score", round(scored[0].score, 3))
            if trace_enabled():
                self._record_final_choice(battle, scored[0])
            if isinstance(scored[0].order, DoubleBattleOrder) and not cancelled():
                memory.record_choice(
                    int(getattr(battle, "turn", 0) or 0), describe_order(scored[0].order)
                )
            if self.config.llm_proposer_enabled:
                record_note("llm_chosen", bool(scored[0].breakdown.get("llm_proposed")))
            record_note("battle_memory", memory.summary())
            return scored[0].order
        record_note("battle_memory", memory.summary())
        return self.choose_random_move(battle)

    def _record_final_choice(self, battle: AbstractBattle, final) -> None:
        """Make the trace describe the order that is actually sent.

        The evaluator records `chosen_breakdown` for ITS top pick every time it ranks orders,
        including the shortlist the search builds and each exact-mirror root, so the note held
        whichever ranking ran last: ladder game 2695880700 turn 8 played a double Protect
        while the trace explained Trick. After search, judge and BC blending this re-scores
        the final order with the evaluator and overwrites both notes (the myopic ranking stays
        in `top_candidates`, labelled as such).
        """

        order = final.order
        if not isinstance(order, DoubleBattleOrder):
            return
        record_note(
            "final_choice",
            {"order": describe_order(order), "score": round(float(final.score), 3)},
        )
        try:
            breakdown = breakdown_for_order(battle, order, self.config)
        except Exception:  # noqa: BLE001 - diagnostics must never cost a decision
            breakdown = None
        if breakdown is not None:
            record_note("chosen_breakdown", breakdown)
            record_note("chosen_breakdown_order", describe_order(order))

    def _search(self, battle: DoubleBattle, memory: BattleMemory) -> list:
        """The fast search, then (opt-in) the exact judge over its best candidates.

        With `PolicyConfig.exact_judge_live` off this is exactly `_fast_search`. With it
        on, `vgc.exact_judge` re-ranks the fast search's top candidates through the public
        exact mirror and returns the list with the exact-best order first; any failure or
        timeout returns the fast search's own list.
        """
        scored = self._fast_search(battle, memory)
        if self.config.exact_judge_live and scored:
            scored = self._exact_judge().rerank(battle, memory, scored)
        return scored

    def _exact_judge(self):
        judge = getattr(self, "_exact_judge_obj", None)
        if judge is None:
            from vgc.exact_judge import ExactJudge  # lazy: nothing loads when the knob is off

            judge = ExactJudge(self.config, getattr(self, "_own_packed_team", None))
            self._exact_judge_obj = judge
        return judge

    @property
    def exact_judge_log(self) -> list[dict[str, object]]:
        """One record per judged decision (empty when the judge is off or never ran)."""
        judge = getattr(self, "_exact_judge_obj", None)
        return judge.log if judge is not None else []

    def _fast_search(self, battle: DoubleBattle, memory: BattleMemory) -> list:
        """`search_joint_orders`, plus the LLM proposer (or the equal-time control's extra
        candidates) when configured. With both off this is exactly the plain call."""
        if self.config.llm_proposer_enabled:
            from vgc.llm.proposer import make_order_proposer

            return search_joint_orders(
                battle,
                self.config,
                order_proposer=make_order_proposer(battle, self.config, memory),
            )
        if self.config.llm_control_extra_candidates > 0:
            return search_joint_orders(
                battle,
                self.config,
                extra_candidates=self.config.llm_control_extra_candidates,
            )
        return search_joint_orders(battle, self.config)

    def decide_teampreview(self, battle: AbstractBattle) -> str:
        """Choose a teampreview order: `vgc.team_preview.build_team_order`, or poke-env's
        random (format-aware, e.g. truncates to a bring-4 pick for vgc formats) when the
        evaluator is disabled.
        """
        if self.config.use_heuristic_evaluator:
            order = build_team_order(battle, self.config)
            if self.config.llm_preview_enabled:
                # Off by default. The heuristic order above is the fallback for every
                # LLM failure (see vgc.llm.preview.choose_preview, which never raises).
                from vgc.llm.preview import choose_preview

                order = choose_preview(battle, self.config, order)
            return order
        return self.random_teampreview(battle)

    # --- clock guard ----------------------------------------------------------------

    def _cheap_order(self, battle: AbstractBattle) -> BattleOrder:
        """Instant legal fallback: the myopic evaluator's top order, else random."""

        if self.config.use_heuristic_evaluator and isinstance(battle, DoubleBattle):
            scored = score_joint_orders(battle, self.config)
            if scored:
                return scored[0].order
        return self.choose_random_move(battle)

    def _cheap_teampreview(self, battle: AbstractBattle) -> str:
        """Preview fallback: the heuristic pick (~25 ms measured), else random bring-4."""

        if self.config.use_heuristic_evaluator:
            return build_team_order(battle, self.config)
        return self.random_teampreview(battle)

    def _guarded(
        self,
        battle: AbstractBattle,
        kind: str,
        decide: Callable[[], object],
        fallback: Callable[[], object],
    ) -> object:
        """Run ``decide`` under the clock budget; see `vgc.clock.run_with_deadline`.

        With an unknown clock (offline direct env, or no timer announced) this is just
        ``decide()`` -- same call, same exceptions, same trace. With a known clock the
        decision runs on a worker thread against an isolated trace (merged back only if
        it finishes in time, so a late worker can never write into a finished trace) and
        the cheap fallback is sent if the deadline passes. Records one `clock_log` entry
        and a ``clock`` trace note either way.
        """

        own_team = getattr(self, "_own_packed_team", None)

        def scoped_decide() -> object:
            # Per-team measured caches read only this player's own team (vgc.team_scope).
            bind_own_team(own_team)
            return decide()

        tracker = self._clock_for_tag(battle.battle_tag)
        idx, state = tracker.begin_decision()
        budget = budget_seconds(state, kind, self.config) if state else Budget(None, kind=kind)
        parent_trace = current_trace()
        cancel = threading.Event()
        start = time.monotonic()
        result = None
        try:
            slot = self._worker_slot(battle)
            if budget.seconds is None and not slot.busy():
                value = contextvars.copy_context().run(scoped_decide)
                reason = "none"
            else:
                deadline = None if budget.seconds is None else start + budget.seconds
                result = run_with_deadline(
                    lambda: self._run_isolated(scoped_decide, cancel, deadline),
                    lambda: self._run_isolated(fallback, None)[0],
                    budget,
                    slot=slot,
                    owner=battle.battle_tag,
                    cancel=cancel,
                )
                reason = result.reason
                if reason == "none":
                    value, sub_trace = result.value
                    if parent_trace is not None and sub_trace is not None:
                        parent_trace.notes.update(sub_trace.notes)
                        if sub_trace.fallback_used:
                            record_fallback(sub_trace.fallback_reason or "decide() fallback")
                else:
                    value = result.value
        finally:
            elapsed = time.monotonic() - start
            tracker.end_decision(idx, elapsed)
            if result is None or result.reason != "none":
                cancel.set()  # a still-running worker must not act on its late result
        if reason == "exception":
            self.fallback_count += 1
            record_fallback(f"decide() raised {result.error!r}")
            self.logger.warning("decide() raised under clock guard: %r", result.error)
        if reason in ("deadline", "fallback-only", "previous-worker-busy"):
            self.fallback_count += 1
            record_fallback(f"clock guard: {reason} (budget {budget.seconds}s)")
        if reason != "none" and isinstance(value, DoubleBattleOrder):
            self._memory_for(battle).record_choice(
                int(getattr(battle, "turn", 0) or 0), describe_order(value)
            )
        entry = {
            "battle_tag": battle.battle_tag,
            "turn": getattr(battle, "turn", None),
            "decision_kind": kind,
            "elapsed_ms": round(elapsed * 1000, 1),
            "budget_s": budget.seconds,
            "bank_left_s": None if budget.bank_left_s is None else round(budget.bank_left_s, 1),
            "fallback_reason": reason,
        }
        self.clock_log.append(entry)
        record_note("clock", entry)
        return value

    @staticmethod
    def _run_isolated(
        fn: Callable[[], object],
        cancel: threading.Event | None,
        deadline: float | None = None,
    ) -> tuple[object, object]:
        """Run ``fn`` in a copied context with its own throwaway `DecisionTrace`."""

        def inner() -> tuple[object, object]:
            if cancel is not None:
                bind_cancel(cancel)
            bind_deadline(deadline)
            start_trace()
            return fn(), current_trace()

        return contextvars.copy_context().run(inner)

    # --- exception-safe wrappers (never override these) ---------------------------

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        trace_token = start_trace()
        chosen_order: BattleOrder | None = None
        replay_sequence = self._record_decision(battle, team_preview=False)
        try:
            chosen_order = self._guarded(
                battle,
                _move_kind(battle),
                lambda: self.decide(battle),
                lambda: self._cheap_order(battle),
            )
            return chosen_order
        except Exception as exc:  # noqa: BLE001 - must never crash a battle
            self.fallback_count += 1
            record_fallback(f"decide() raised {exc!r}")
            if self.config.log_decisions:
                self.logger.exception(
                    "decide() raised; falling back to random move (turn=%s)",
                    getattr(battle, "turn", None),
                )
            chosen_order = self.choose_random_move(battle)
            return chosen_order
        finally:
            recorder = getattr(self, "_decision_replay_recorder", None)
            if recorder is not None and replay_sequence is not None and chosen_order is not None:
                recorder.record_choice(
                    battle.battle_tag,
                    replay_sequence,
                    (
                        describe_order(chosen_order)
                        if isinstance(chosen_order, DoubleBattleOrder)
                        else str(chosen_order)
                    ),
                    wire=(
                        choice_wire_message(chosen_order)
                        if isinstance(chosen_order, BattleOrder)
                        else str(chosen_order)
                    ),
                )
            trace = current_trace()
            if trace is not None:
                trace.turn = getattr(battle, "turn", None)
                if isinstance(chosen_order, DoubleBattleOrder):
                    trace.chosen_order = describe_order(chosen_order)
                elif chosen_order is not None:
                    trace.chosen_order = str(chosen_order)
            finished_trace = finish_trace(trace_token)
            if finished_trace is not None:
                self.decision_trace_history.append(
                    {"battle_tag": battle.battle_tag, **asdict(finished_trace)}
                )

    def teampreview(self, battle: AbstractBattle) -> str:
        trace_token = start_trace()
        chosen_order: str | None = None
        replay_sequence = self._record_decision(battle, team_preview=True)
        try:
            chosen_order = self._guarded(
                battle,
                "preview",
                lambda: self.decide_teampreview(battle),
                lambda: self._cheap_teampreview(battle),
            )
            return chosen_order
        except Exception as exc:  # noqa: BLE001 - must never crash a battle
            self.fallback_count += 1
            record_fallback(f"decide_teampreview() raised {exc!r}")
            if self.config.log_decisions:
                self.logger.exception(
                    "decide_teampreview() raised; falling back to /team 1234: %r", exc
                )
            chosen_order = "/team 1234"
            return chosen_order
        finally:
            recorder = getattr(self, "_decision_replay_recorder", None)
            if recorder is not None and replay_sequence is not None and chosen_order is not None:
                recorder.record_choice(battle.battle_tag, replay_sequence, chosen_order)
            trace = current_trace()
            if trace is not None:
                trace.turn = 0
                trace.chosen_order = chosen_order
            finished_trace = finish_trace(trace_token)
            if finished_trace is not None:
                self.decision_trace_history.append(
                    {"battle_tag": battle.battle_tag, **asdict(finished_trace)}
                )
            elif trace_enabled():
                # Defensive ladder instrumentation: team preview can be requested from
                # poke-env's OTS recovery path before a ContextVar trace survives the
                # callback boundary. Preserve the chosen order even in that case.
                self.decision_trace_history.append(
                    {
                        "battle_tag": battle.battle_tag,
                        "turn": 0,
                        "chosen_order": chosen_order,
                        "fallback_used": False,
                        "fallback_reason": None,
                        "notes": {},
                    }
                )
