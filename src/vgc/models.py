"""Typed internal models for the VGC battle policy.

Mirrors the ~/code/projects/pokemon-tcg-ai `PolicyConfig` pattern: a single frozen
dataclass of individually-commented knobs that every decision path threads through, so
behavior changes are one flag away from being A/B-able in offline/run_gates.py instead of
buried in code.
"""

from __future__ import annotations

from dataclasses import dataclass

from vgc.config import FORMAT_ID


@dataclass(frozen=True)
class PolicyConfig:
    """Knobs for `vgc.agent.VgcPlayer`. Phase 1 scaffold only -- no real evaluator yet
    (that's Phase 2), so most of this is placeholder plumbing that Phase 2 will read.
    """

    # Which Showdown format this policy plays. Always the Champions-mod Reg M-B doubles
    # format unless a caller is deliberately testing against something else.
    format_id: str = FORMAT_ID
    # We WANT to see the opponent's Open Team Sheet (revealed sets/items/abilities/Tera
    # before team preview) -- poke-env auto-rejects OTS unless this is True, and rejecting
    # it would throw away free information the ladder format hands us.
    accept_open_team_sheet: bool = True
    # Emit one log line per `decide()` exception (see vgc.agent.VgcPlayer) so a battle that
    # silently fell back to random play is visible instead of just... quietly losing.
    log_decisions: bool = False

    # --- Phase 2b evaluator (vgc.evaluator.score_joint_orders) -------------------------
    # Master switch for the real heuristic evaluator; False keeps VgcPlayer.decide()
    # falling through to random move selection (useful as a quick A/B anchor in
    # offline/run_gates.py without swapping the whole baseline name).
    use_heuristic_evaluator: bool = True
    # Placeholder weight for a future speed-control (Tailwind/Trick Room/priority)
    # scoring term; superseded by the explicit trick_room_* weights below but kept for
    # any external caller that still reads it.
    speed_control_weight: float = 1.0

    # -- Offense: damage/KO scoring ------------------------------------------------------
    # Points per 1% of the target's max HP dealt in expected damage. This is the
    # baseline currency every other offense bonus below is denominated against.
    damage_percent_weight: float = 1.0
    # Flat bonus when the move's MINIMUM roll already KOs the target (guaranteed KO --
    # no roll-dependent risk), on top of the damage_percent_weight score for the (100%)
    # damage itself. Weighted well above a mere 100-damage_percent_weight-points hit so
    # removing a threat from the field decisively outweighs chip damage optimization --
    # doubles VGC is generally a game about who runs out of Pokemon first, not who's
    # ahead on aggregate % dealt (tuned up from 60 -> 80 after offline/run_gates.py
    # showed the evaluator under-prioritizing kills against SimpleHeuristicsPlayer).
    guaranteed_ko_bonus: float = 80.0
    # Smaller flat bonus when only the EXPECTED (not minimum) roll KOs -- real but
    # roll-dependent, so weighted well below guaranteed_ko_bonus.
    likely_ko_bonus: float = 35.0
    # Multiplier (>1) applied to damage a spread move (Earthquake et al, moves whose
    # data/champions/moves.json target is "allAdjacent") deals to OUR OWN ally, then
    # subtracted from that order's score -- hurting your own team is worse than simply
    # forgoing that damage, not merely "free" the way it is against a fainted/immune ally.
    ally_damage_penalty_weight: float = 1.5

    # -- Speed/turn order -----------------------------------------------------------------
    # When a target is faster than the attacker (post Trick Room inversion) and its best
    # known move can KO the attacker first, expected offensive output for the attacker's
    # chosen move this turn is multiplied by this factor (<1) -- it may never get to act.
    threatened_output_discount: float = 0.5
    # Bonus for outspeeding a target AND KOing it (guaranteed or likely) before it can
    # act -- rewards proactive "kill the threat on the switch-in" plays over reactive ones.
    outspeed_ko_bonus: float = 25.0

    # -- Defense/utility: Protect --------------------------------------------------------
    # Points per 1% of estimated incoming damage (from the opponent's best revealed move
    # onto this slot) that choosing Protect this turn avoids.
    protect_threat_weight: float = 0.8
    # Flat penalty applied when this Pokemon's poke-env `protect_counter` is already >= 1
    # (Protect/a protect-family move was used last turn too) -- mirrors the real success-
    # rate falloff for chaining protection moves without pretending to model the exact
    # (1/3)^n formula.
    protect_repeat_penalty: float = 35.0
    # Flat penalty for choosing Protect when the estimated incoming threat on this slot is
    # below this many HP percent -- keeps the evaluator from reflexively protecting both
    # slots when neither is actually in danger this turn.
    protect_low_threat_penalty: float = 12.0
    # Threat percent (of this Pokemon's max HP) below which protect_low_threat_penalty
    # applies.
    protect_low_threat_floor: float = 15.0

    # -- Status/utility: Trick Room -------------------------------------------------------
    # Points per (opponent_avg_speed - our_avg_speed) when Trick Room is NOT currently up
    # -- positive when we're the slower side (Trick Room helps us), negative (a bad idea)
    # when we're already the faster side.
    trick_room_setup_weight: float = 0.15
    # Flat penalty applied instead of trick_room_setup_weight's formula when Trick Room is
    # ALREADY up -- using the move again cancels it early (data/moves.ts's toggle
    # behavior), which is almost always wrong if it's currently helping our slower side.
    trick_room_teardown_penalty: float = 50.0

    # -- Status/utility: Fake Out ----------------------------------------------------------
    # Base value for a legal (first-turn-out) Fake Out on a target without flinch immunity
    # (Inner Focus et al), scaled by that target's own estimated offensive threat (a flinch
    # on a scary attacker is worth more than one on a harmless support mon).
    fake_out_weight: float = 20.0

    # -- Status/utility: Sleep Powder -------------------------------------------------------
    # Base value for landing Sleep Powder on the fastest eligible (non-Grass, non-Overcoat,
    # not-already-statused) opposing target.
    sleep_powder_weight: float = 35.0

    # -- Status/utility: Helping Hand --------------------------------------------------------
    # Fraction of the partner slot's own expected-damage score counted as Helping Hand's
    # marginal value (approximates the 1.5x damage multiplier's EV delta without re-running
    # the full damage calc through a modeled volatile).
    helping_hand_weight: float = 0.5

    # -- Screens / generic field utility (Reflect/Light Screen/Aurora Veil/Tailwind) --------
    # Flat value for putting up a not-yet-active screen/Tailwind side condition; deliberately
    # simple (no lookahead into how many hits it blocks) for a myopic v1 evaluator.
    screen_setup_weight: float = 18.0

    # -- Switches ------------------------------------------------------------------------
    # Weight on the matchup-improvement term (our best expected damage % onto the field
    # minus the field's best expected damage % onto us, incoming bench mon vs current
    # opponent actives) when scoring a switch.
    switch_matchup_weight: float = 1.0
    # Flat tempo-loss cost subtracted from every switch (mirrors the free turn of
    # opponent action a reactive switch effectively concedes).
    switch_tempo_cost: float = 15.0
    # Bonus per opposing active without an Intimidate-immune ability (Clear Body/Own
    # Tempo/Inner Focus/Oblivious/Scrappy/Full Metal Body/Hyper Cutter/White Smoke/mist,
    # simplified to a fixed exemption list below) when switching in an Intimidate user.
    intimidate_switch_bonus: float = 10.0

    # -- Mega Evolution --------------------------------------------------------------------
    # Default "mega as soon as legal" knob: True always scores an available mega order at
    # least as high as the non-mega version of the same move (mega stats strictly help
    # here since this evaluator has no lookahead cost to "saving" the mega turn).
    mega_evolve_asap: bool = True

    # -- Tracing -----------------------------------------------------------------------
    # How many top-scoring candidate orders decision_trace.py records per turn when
    # VGC_TRACE is set.
    trace_top_k: int = 3

    # --- Team preview (vgc.team_preview.build_team_order) -------------------------------
    # Weight on the pairwise expected-damage-exchange ratio term (our estimated output
    # onto their previewed 6 vs theirs onto us) when scoring a 4-of-6 pick + lead order.
    team_preview_damage_weight: float = 1.0
    # Weight on the average-speed-advantage term for a candidate lead pair.
    team_preview_speed_weight: float = 0.3
    # Bonus for bringing a known Trick Room setter (a revealed "trickroom" move in that
    # mon's kit) alongside slow attackers -- weights up bringing genuinely slow mons once
    # the setter is also being brought, since TR mode wants a coherent slow core, not just
    # the setter alone.
    team_preview_tr_coherence_weight: float = 0.4
