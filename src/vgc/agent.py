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

from dataclasses import asdict

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from poke_env.player.player import Player

from vgc.actions import describe_order
from vgc.battle_memory import BattleMemory
from vgc.bc.policy import load_bc_policy, score_orders
from vgc.decision_trace import (
    current_trace,
    finish_trace,
    record_fallback,
    record_note,
    start_trace,
    trace_enabled,
)
from vgc.evaluator import score_joint_orders
from vgc.models import PolicyConfig
from vgc.own_team import apply_own_spreads
from vgc.search import search_joint_orders
from vgc.team_preview import build_team_order


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
        # poke-env 0.15 can receive an opponent's OTS rejection before its team-preview
        # request. Remember that room until the request arrives so the battle does not
        # wait forever for a `showteam` message that Showdown will never send.
        self._pending_ots_rejections: set[str] = set()
        self._resolved_ots_rejections: set[str] = set()
        self._battle_memories: dict[str, BattleMemory] = {}
        self.decision_trace_history: list[dict[str, object]] = []
        player_kwargs.setdefault("battle_format", self.config.format_id)
        player_kwargs.setdefault("accept_open_team_sheet", self.config.accept_open_team_sheet)
        super().__init__(**player_kwargs)

    async def _handle_battle_message(self, split_messages) -> None:
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
        if battle_tag:
            self._memory_for_tag(battle_tag).observe_protocol(split_messages)
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
        # Gated OFF by default: a n=500 mirror A/B says this makes the heuristic bot
        # WORSE (40.0%, CI [0.358, 0.444]) because it corrects only our half of a
        # comparison whose other half is still estimated. See PolicyConfig.
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

    def _memory_for(self, battle: AbstractBattle) -> BattleMemory:
        memory = self._memory_for_tag(battle.battle_tag)
        memory.observe_battle(battle)
        try:
            setattr(battle, "_vgc_battle_memory", memory)
        except (AttributeError, TypeError):
            pass
        return memory

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
        scored: list = []
        if self.config.use_two_ply_search and isinstance(battle, DoubleBattle):
            scored = search_joint_orders(battle, self.config)
        elif self.config.use_heuristic_evaluator and isinstance(battle, DoubleBattle):
            scored = score_joint_orders(battle, self.config)
        if scored:
            if self.config.use_bc_policy:
                policy = load_bc_policy(self.config.bc_checkpoint_path)
                scored = score_orders(policy, battle, scored, self.config)
            if self.config.log_decisions:
                record_note("chosen_order_score", round(scored[0].score, 3))
            if isinstance(scored[0].order, DoubleBattleOrder):
                memory.record_choice(
                    int(getattr(battle, "turn", 0) or 0), describe_order(scored[0].order)
                )
            record_note("battle_memory", memory.summary())
            return scored[0].order
        record_note("battle_memory", memory.summary())
        return self.choose_random_move(battle)

    def decide_teampreview(self, battle: AbstractBattle) -> str:
        """Choose a teampreview order: `vgc.team_preview.build_team_order`, or poke-env's
        random (format-aware, e.g. truncates to a bring-4 pick for vgc formats) when the
        evaluator is disabled.
        """
        if self.config.use_heuristic_evaluator:
            return build_team_order(battle, self.config)
        return self.random_teampreview(battle)

    # --- exception-safe wrappers (never override these) ---------------------------

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        trace_token = start_trace()
        chosen_order: BattleOrder | None = None
        try:
            chosen_order = self.decide(battle)
            return chosen_order
        except Exception as exc:  # noqa: BLE001 - must never crash a battle
            record_fallback(f"decide() raised {exc!r}")
            if self.config.log_decisions:
                self.logger.exception(
                    "decide() raised; falling back to random move (turn=%s)",
                    getattr(battle, "turn", None),
                )
            chosen_order = self.choose_random_move(battle)
            return chosen_order
        finally:
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
        try:
            chosen_order = self.decide_teampreview(battle)
            return chosen_order
        except Exception as exc:  # noqa: BLE001 - must never crash a battle
            record_fallback(f"decide_teampreview() raised {exc!r}")
            if self.config.log_decisions:
                self.logger.exception(
                    "decide_teampreview() raised; falling back to /team 1234: %r", exc
                )
            chosen_order = "/team 1234"
            return chosen_order
        finally:
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
