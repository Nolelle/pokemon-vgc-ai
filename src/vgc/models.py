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
    # Joint-order penalty when both slots choose the same single target even though one
    # move's minimum damage already removes it. Without this, the second move can become
    # wasted or auto-retarget into an immunity after the first knockout resolves.
    redundant_ko_target_penalty: float = 45.0
    # Joint-order bonus when neither attack KOs alone but their combined expected damage
    # removes the shared target. This preserves deliberate focus fire while separating
    # it from the redundant-overkill case above.
    focus_fire_ko_bonus: float = 35.0
    # Per-percent weight on the redundant stacked attack's expected damage into a
    # Protect-capable shared target (see `_cross_slot_adjustments`): if that slot
    # Protects, both single-target attacks blank, so the smaller attack's expected value
    # -- which could have gone at the OTHER opposing slot instead -- is forfeited with
    # probability `opp_protect_prob`. 1.0 = treat it as a full EV forfeit at that
    # probability, not a partial one.
    protect_stack_penalty_weight: float = 1.0

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

    # -- Defense/utility: opponent Protect anticipation -----------------------------------
    # Baseline chance an opponent slot Protects this turn when it has an unused
    # self-protecting move in its known kit (Open Team Sheets make possession a fact, not
    # a guess) -- even an unpressured Protect happens some of the time as a scouting/tempo
    # play, so this floor is never zero.
    opp_protect_base_prob: float = 0.15
    # Additional Protect probability layered on top of the base rate as our best single
    # attack's expected damage onto that slot approaches 100% of its max HP -- scaled
    # linearly by pressure_pct/100, so a lethal-looking attack makes Protect much more
    # likely than a chip-damage one.
    opp_protect_pressure_scale: float = 0.3
    # Hard cap on the combined Protect probability -- Protect should never be treated as
    # more likely than a coin flip against us; higher would make the evaluator overreact
    # and abandon otherwise-correct attacks out of excessive caution.
    opp_protect_prob_cap: float = 0.5
    # Multiplier that collapses the Protect probability when the opponent's
    # `protect_counter` shows they already Protected last turn -- consecutive Protects
    # have a sharply reduced real success chance in the actual engine, so back-to-back
    # Protect is rare and shouldn't be modeled as equally likely.
    opp_protect_repeat_factor: float = 0.15
    # Flat probability an opponent slot switches out instead of attacking when it is under
    # heavy incoming pressure but its own offensive output is weak (see the floor/ceiling
    # pair below) -- a cornered, low-value attacker pivoting out is a common ladder play.
    opp_switch_prob: float = 0.2
    # Minimum "our best expected % onto that slot" required before switch incentive
    # applies -- below this the slot isn't actually threatened enough to justify bailing.
    opp_switch_pressure_floor: float = 60.0
    # That opponent slot's own best expected % onto either of our actives must be BELOW
    # this for switch incentive to apply -- a slot still hitting hard has a reason to stay
    # in even while under pressure.
    opp_switch_output_ceiling: float = 30.0

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

    # --- Phase 2c: shallow 2-ply search (vgc.search.search_joint_orders) ----------------
    # Master switch: True routes VgcPlayer.decide() through the 2-ply search instead of
    # the plain myopic evaluator, so the bot anticipates the opponent's best responses
    # (Protect, their best attack, a defensive pivot) instead of assuming they stand
    # still. Defaults to False: the offline gate proxy (poke-env's
    # SimpleHeuristicsPlayer) never Protects and doesn't target-optimize, so it
    # systematically punishes opponent-response modeling it doesn't itself exhibit
    # (2026-07 gate: myopic-only 81.7% vs search 73.3% vs SimpleHeuristicsPlayer, n=300
    # each) -- the search specifically models human ladder behaviors (Protect timing,
    # coordinated targeting) that a non-Protecting, non-coordinating bot opponent can
    # never demonstrate a benefit against. False must leave decide() byte-for-byte
    # identical to the pre-Phase-2c myopic-only behavior (score_joint_orders directly).
    # Enabled explicitly per ladder session (`ladder/run_ladder.py --search`) as an A/B
    # against real opponents until that data settles what the default should actually be.
    use_two_ply_search: bool = False
    # How many of the myopic evaluator's top-ranked orders get the (expensive) exchange
    # search at all -- orders outside this cutoff are almost never the real best move, so
    # spending damage_range calls on them is wasted; they keep their myopic score (scaled
    # by search_myopic_weight) so the returned list stays complete and comparable.
    search_our_candidates: int = 10
    # Per opponent slot, how many of their known damaging moves (ranked by expected
    # damage against whichever of our actives it hits hardest) are kept as response
    # candidates -- a Protect candidate (see search_opp_candidates below) is ALWAYS added
    # on top of this cap, not counted against it, since it's usually the single most
    # important response to get right.
    search_opp_moves_per_slot: int = 4
    # Cap on the joint (both opponent slots) response candidates kept after the per-slot
    # cross product, ranked by a cheap enumeration-time score (summed expected damage,
    # with Protect valued via protect_threat_weight) -- keeps resolve_exchange's
    # damage_range call count bounded regardless of how many moves either slot knows.
    search_opp_candidates: int = 8
    # Points per 1% of a Pokemon's max HP lost during a simulated exchange -- the same
    # currency as damage_percent_weight, so exchange-derived and myopic-derived HP
    # percentages are directly comparable once blended together.
    search_hp_weight: float = 1.0
    # Points per Pokemon fainted during a simulated exchange. Set slightly ABOVE
    # guaranteed_ko_bonus's scale (not equal) because a faint that survives a full
    # opponent-response simulation is a higher-confidence removal than the myopic
    # evaluator's single-ply guaranteed_ko_bonus, which only knows the target didn't
    # dodge -- it hasn't checked whether the opponent's own best response also changes
    # the picture (e.g. a revenge-kill that would have made the KO moot anyway).
    search_faint_weight: float = 90.0
    # Weight on the WORST-case opponent response when aggregating a candidate order's
    # simulated exchange values into one number: `w * min(values) + (1-w) *
    # response_likelihood_weighted_expectation(values)` (see search_response_temperature
    # below for the expectation term's weights). Dropped from 0.7 -> 0.25 after v1 gate
    # testing (2026-07) showed a HIGH worst-case weight is systematically pessimistic: a
    # pure minimax treats "they Protect the focused slot" as certain every turn (every
    # response set has a Protect candidate), which made the search drift passive --
    # preferring its own Protect over a 44-point-better attack, and dodging imaginary
    # Protects by picking worse attack targets. An opponent response's influence on the
    # score should track how LIKELY that response actually is (the expectation term now
    # does this, weighted by enumeration score and each Protect's real opp_protect_prob),
    # not treat every enumerated worst case as equally certain -- the residual worst-case
    # weight here is a tail-risk hedge on top of that, not the dominant term.
    search_worst_case_weight: float = 0.25
    # Points of enumeration score (see _enumerate_opp_responses' cheap per-response
    # score) per e-fold of response likelihood in the softmax that turns those scores
    # into weights for the expectation term above (`p_i ∝ exp(enum_score_i /
    # search_response_temperature)`). Lower = sharper/more adversarial (the opponent's
    # best-looking responses dominate the expectation almost as much as a pure worst
    # case would); higher = flatter/closer to a plain average across all enumerated
    # responses.
    search_response_temperature: float = 30.0
    # Weight on the plain myopic score in the final per-order blend (see
    # search_position_weight below) -- kept at 1.0 by default so the search is additive
    # on top of the existing (already-tuned) myopic weights, not a wholesale replacement
    # of their calibration.
    search_myopic_weight: float = 1.0
    # Weight on the aggregated simulated-exchange value in the final per-order blend:
    # `search_myopic_weight * myopic_score + search_position_weight * aggregated_value`.
    # Setting this to 0.0 disables the search's actual influence on ranking while still
    # paying its compute cost -- useful as an isolation test, not a recommended setting.
    search_position_weight: float = 1.0

    # --- Phase 3: replay-corpus set priors (vgc.sets.opponent_move_ids) -----------------
    # Master switch for filling UNREVEALED opponent moves from data/usage/set_priors.json
    # (see tools/build_set_priors.py) -- Open Team Sheets essentially never triggers on
    # the real public ladder (vgc.replay_parse's module docstring: ~0.2% of downloaded
    # replays reveal a full sheet, since this format's "Open Team Sheets" ruleset needs
    # BOTH players to opt in and almost no human ladder opponent does), so most opposing
    # movesets the threat/Protect model sees in a real game are otherwise 0-4 known moves
    # out of the real 4, starving `_opp_protect_probability`/`_best_attacking_move` of the
    # information they need. Gate-neutral by construction: offline gates run mutual OTS
    # accept (every move is already revealed there), so filling has nothing left to fill
    # and this knob is a no-op in that setting -- its real effect only shows up against
    # real ladder opponents.
    use_set_priors: bool = True
    # Minimum tracked appearances (set_priors.json's per-species "appearances" count)
    # before that species' move-frequency prior is trusted enough to fill unrevealed
    # moves -- below this the sample is too thin to distinguish a real trend from noise.
    set_prior_min_games: int = 5
    # Total moves per Pokemon after filling. Revealed moves are ALWAYS kept even if
    # somehow more than this many are already revealed (this only caps how many PRIOR
    # moves get layered on top, never truncates real information from Open Team Sheets).
    set_prior_max_moves: int = 4

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

    # --- Final integration: BC v2 candidate re-ranker (vgc.bc.policy.score_orders) ------
    # Master switch: True blends the trained BC v2 checkpoint's learned move/target
    # log-probabilities into the top-ranked heuristic candidates' scores instead of
    # taking the heuristic's (search or myopic) argmax directly. Defaults to False for
    # the same reason use_two_ply_search does: this needs a real ladder A/B before it's
    # trusted as a default, not just an offline gate proxy -- opted in per session via
    # `ladder/run_ladder.py --bc`.
    use_bc_policy: bool = False
    # Converts nats (the BC model's log-probability units) into heuristic score points
    # for the additive blend `heuristic_score + bc_blend_weight * bc_logprob`. The
    # heuristic's own top candidates typically sit within ~40 points of each other, so
    # ~1.3 nats of model preference (30.0 * 1.3 ~= 40) is enough to flip a genuinely close
    # call but not enough to override a decisive KO (which scores far outside that
    # ~40-point band via guaranteed_ko_bonus etc).
    bc_blend_weight: float = 30.0
    # How many of the heuristic's top-ranked candidates get the (batched, but still not
    # free) BC forward pass -- same enumeration-cost reasoning as
    # search_our_candidates: candidates outside this cutoff are almost never the real
    # best move, so re-scoring them is wasted; they keep their heuristic score, pinned
    # strictly below the reranked block (see vgc.bc.policy.score_orders's module
    # docstring for why, and search.py's earlier Fix-1 regression for the lesson this
    # repeats).
    bc_rerank_top_k: int = 10
    # Checkpoint path `vgc.bc.policy.load_bc_policy` reads. Relative to the repo root
    # when run from `ladder/run_ladder.py` (matches every other data/-prefixed default
    # path in this codebase, e.g. TEAMS_DIR).
    bc_checkpoint_path: str = "data/models/bc_policy_v2.pt"
