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

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player

from vgc.decision_trace import finish_trace, record_fallback, record_note, start_trace
from vgc.evaluator import score_joint_orders
from vgc.models import PolicyConfig
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
        player_kwargs.setdefault("battle_format", self.config.format_id)
        player_kwargs.setdefault(
            "accept_open_team_sheet", self.config.accept_open_team_sheet
        )
        super().__init__(**player_kwargs)

    # --- overridable hooks --------------------------------------------------------

    def decide(self, battle: AbstractBattle) -> BattleOrder:
        """Choose a move: the argmax of `vgc.evaluator.score_joint_orders`, or a random
        legal move if the evaluator is disabled (`PolicyConfig.use_heuristic_evaluator`)
        or has nothing to score (e.g. `enumerate_joint_orders` came back empty -- see its
        own docstring for when that happens). Only wired up for `DoubleBattle` (this
        project's format is always doubles -- see vgc/config.py's FORMAT_ID); any other
        battle type falls back to random rather than guessing.
        """
        if self.config.use_heuristic_evaluator and isinstance(battle, DoubleBattle):
            scored = score_joint_orders(battle, self.config)
            if scored:
                if self.config.log_decisions:
                    record_note("chosen_order_score", round(scored[0].score, 3))
                return scored[0].order
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
        try:
            return self.decide(battle)
        except Exception as exc:  # noqa: BLE001 - must never crash a battle
            record_fallback(f"decide() raised {exc!r}")
            if self.config.log_decisions:
                self.logger.exception(
                    "decide() raised; falling back to random move (turn=%s)",
                    getattr(battle, "turn", None),
                )
            return self.choose_random_move(battle)
        finally:
            finish_trace(trace_token)

    def teampreview(self, battle: AbstractBattle) -> str:
        try:
            return self.decide_teampreview(battle)
        except Exception as exc:  # noqa: BLE001 - must never crash a battle
            if self.config.log_decisions:
                self.logger.exception(
                    "decide_teampreview() raised; falling back to /team 1234: %r", exc
                )
            return "/team 1234"
