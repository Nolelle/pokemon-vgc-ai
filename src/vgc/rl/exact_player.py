"""A player whose every move is picked by the public-information exact search.

This exists so the multi-team A/B harness can measure changes to the exact judge
(`vgc.rl.exact_search._position_value` and its aggregation). The shipped `vgc` player
decides with the fast Python search, so an exact-only `PolicyConfig` knob has no effect
on it at all -- testing such a knob through `vgc` is a null by construction.

The decision path is the same one the teacher and hybrid mode use:
`vgc.rl.public_search.public_information_exact_search` over a `LiveExactMirror`, i.e.
only what a real player can see. A decision the exact search cannot make (it raised, or
its ranking is illegal on the real battle) falls back to the shipped fast search and is
COUNTED, so a report can show how much of a run was really the exact judge playing.
"""

from __future__ import annotations

from poke_env.battle.double_battle import DoubleBattle

from vgc.actions import enumerate_joint_orders
from vgc.agent import VgcPlayer
from vgc.rl.live_mirror import LiveExactMirror
from vgc.rl.public_search import public_information_exact_search


class ExactSearchPlayer(VgcPlayer):
    """Play the exact Showdown search's top legal order every turn."""

    def __init__(self, **player_kwargs) -> None:
        supplied_team = player_kwargs.get("team")
        self._exact_own_packed_team = supplied_team if isinstance(supplied_team, str) else None
        self._public_exact_mirror: LiveExactMirror | None = None
        self.exact_decisions = 0
        self.exact_fallbacks = 0
        self.exact_fallback_reasons: list[str] = []
        super().__init__(**player_kwargs)

    def close_public_mirror(self) -> None:
        if self._public_exact_mirror is not None:
            self._public_exact_mirror.close()
            self._public_exact_mirror = None

    def _battle_finished_callback(self, battle) -> None:
        self.close_public_mirror()
        super()._battle_finished_callback(battle)

    def _fallback(self, battle, reason: str):
        self.exact_fallbacks += 1
        self.exact_fallback_reasons.append(
            f"{battle.battle_tag} turn {int(getattr(battle, 'turn', 0) or 0)}: {reason}"
        )
        return super().decide(battle)

    def decide(self, battle):
        if not isinstance(battle, DoubleBattle) or not self.config.use_two_ply_search:
            return super().decide(battle)
        self.exact_decisions += 1
        if not self._exact_own_packed_team:
            return self._fallback(battle, "packed own team is unavailable")
        try:
            if self._public_exact_mirror is None:
                self._public_exact_mirror = LiveExactMirror(
                    self._exact_own_packed_team, self.config
                )
            scored = public_information_exact_search(
                battle,
                self.config,
                self._exact_own_packed_team,
                memory=self._memory_for(battle),
                mirror=self._public_exact_mirror,
            )
        except Exception as exc:  # noqa: BLE001 -- counted and reported, never hidden
            return self._fallback(battle, f"exact search raised {exc!r}")
        legal_here = {order.message for order in enumerate_joint_orders(battle)}
        for entry in scored:
            if entry.order.message in legal_here:
                return entry.order
        return self._fallback(battle, "no ranked order is legal on the real battle")
