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
    # Fill in our OWN Stat Points/nature from the team file when no Open Team Sheets
    # `showteam` arrives (see `vgc.own_team`). poke-env otherwise leaves `Pokemon.evs`
    # None for our own team on ~99.8% of ladder games, so the evaluator guesses our own
    # spread with `default_opponent_spread` -- off by up to 35.6% on meta1, and
    # underestimating Speed on all six.
    #
    # DEFAULT OFF, against intuition, because a same-session mirror A/B says turning it
    # on makes the heuristic bot WORSE: identical `vgc` bots, one knowing its own spread
    # and one guessing, alternating seats, n=500 -> the knowing bot won 200/500 = 40.0%
    # (95% CI [0.358, 0.444]; 50% excluded, so this is a real effect, not noise).
    #
    # The likely mechanism is consistency, not accuracy. The evaluator compares OUR speed
    # against the opponent's ESTIMATED speed, and that estimate uses the same
    # usage/default spreads. Guessing both sides keeps the comparison like-for-like;
    # correcting only our side makes us believe we outspeed opponents we do not, and the
    # bot plays too aggressively as a result. Fixing our half of a two-sided comparison
    # made the comparison worse.
    #
    # Do not flip this without rerunning that A/B plus the follow-up it implies: give the
    # bot the opponent's TRUE spread as well (the OTS configuration) and see whether
    # accurate-both beats guessed-both. If it does, the work belongs in the opponent
    # estimator, and this knob comes along for free.
    use_own_team_spreads: bool = False
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
    # Multiplier on a `charge`-flag move's (Solar Beam, Solar Blade, Sky Attack, Dig,
    # Fly, ...) entire per-order contribution when it does NOT skip its charge turn
    # under current conditions (sun skips Solar Beam/Solar Blade's charge; Power Herb
    # also skips it once -- NOT modeled here, see `_score_attack_order`'s comment for
    # the gap). You telegraph the move and spend two turns to land roughly one turn's
    # worth of damage, MINUS the free hit the opponent gets on your charge turn -- 0.35
    # approximates "about a third of a normal attack's value" rather than a naive 0.5
    # (charge turns are worse than merely "slow", since Protect/switch punishes them for
    # free and the opponent has already seen the tell).
    charge_move_discount: float = 0.35
    # Multiplier on a `recharge`-flag move's (Hyper Beam, Giga Impact, ...) entire
    # per-order contribution -- unlike charge moves, the recharge turn happens AFTER the
    # hit lands (so this turn's damage is real, not merely telegraphed), but the
    # following free turn for the opponent is still a real cost baked into "is this move
    # worth clicking" -- 0.6 is a milder discount than charge_move_discount's 0.35 for
    # exactly that reason (the damage already happened; only the follow-up cost is
    # hypothetical).
    recharge_move_discount: float = 0.6

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
    # SUPERSEDED by `protect_success_decay` (below): `_score_protect` used to subtract
    # this flat penalty once `protect_counter >= 1` instead of actually modeling Gen 9's
    # real geometric success-rate falloff. Kept (unread) rather than deleted so any
    # external caller/experiment log still referencing this field name doesn't break.
    protect_repeat_penalty: float = 35.0
    # Real Gen 9 mechanics: each consecutive protect-family use divides success odds by
    # ~3 (counter 0 = 100%, 1 = ~33%, 2 = ~11%, ...). `_score_protect` now multiplies the
    # threat-avoidance term by `protect_success_decay ** protect_counter` instead of
    # `protect_repeat_penalty`'s flat subtraction, so the SCORE decays geometrically the
    # same way the move's real success chance does (see `_opp_protect_probability`'s
    # `opp_protect_repeat_factor` for the same 1/3 idea modeled from the OPPONENT's side
    # -- that one stays a flat multiplier rather than switching to this exact formula
    # since it only ever applies once, not compounding across a whole streak).
    protect_success_decay: float = 1.0 / 3.0
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
    # Protect is rare and shouldn't be modeled as equally likely. Approximates the same
    # ~1/3-per-streak falloff `protect_success_decay` models exactly for OUR OWN Protect
    # scoring (see that field's comment) -- kept as a flat one-shot multiplier here
    # rather than switched to the same `** protect_counter` formula since
    # `_opp_protect_probability` only ever needs "did they Protect last turn" (one
    # lookback), not a compounding streak counter.
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
    # Points per 1% of threat avoided by FLIPPING a slot's speed-race outcome against its
    # biggest known threat (see `_speed_control_flip_value`) -- on top of
    # trick_room_setup_weight's plain average-speed-gap term, which alone is far too
    # small to ever win the argmax (confirmed against a reconstructed real postmortem
    # trace, gen9championsvgc2026regmb-2651715825: a 228-point average speed deficit
    # scored only ~34 points under trick_room_setup_weight=0.15 alone, while a single
    # attacking move scored 260+ -- Trick Room could never be chosen even while outsped
    # on every slot). 1.0 was tuned so the reconstructed scenario's Trick Room order
    # (both our slots flip from losing to winning their speed race) becomes competitive
    # with/beats passive Protect+chip-damage orders, while staying well below what a
    # guaranteed-KO order scores (guaranteed_ko_bonus=80 + ~100 points of raw damage +
    # possibly outspeed_ko_bonus=25 on top, easily 200+ from ONE target alone) --
    # see tests/test_evaluator.py's speed-control-flip tests for both directions.
    # Applies identically to Tailwind (`_score_screen`'s tailwind branch) since it flips
    # the same speed-race outcome via a different mechanism (2x our speed, not an
    # inverted comparison) -- see that function's comment.
    speed_control_flip_weight: float = 1.0

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
    mega_evolve_asap: bool = False
    # Preserve the once-per-battle Mega resource when evolving does not materially alter
    # this turn's damage, Speed order, ability, weather, or survival. Applied to a Mega
    # order whose estimated improvement is below mega_material_gain_floor.
    mega_unnecessary_penalty: float = 18.0
    # Score-point improvement required before Mega is considered to change the current
    # turn enough to spend the resource without the delay penalty.
    mega_material_gain_floor: float = 10.0
    # Small tie-shaping bonus for the matchup-selected default Mega from team preview;
    # alternate Megas remain legal but need a real current-turn gain to overcome it.
    default_mega_bonus: float = 6.0
    alternate_mega_penalty: float = 8.0

    # -- Tracing -----------------------------------------------------------------------
    # How many top-scoring candidate orders decision_trace.py records per turn when
    # VGC_TRACE is set.
    trace_top_k: int = 3

    # --- Phase 2c: shallow 2-ply search (vgc.search.search_joint_orders) ----------------
    # Master switch: True routes VgcPlayer.decide() through the 2-ply search instead of
    # the plain myopic evaluator, so the bot anticipates the opponent's best responses
    # (Protect, their best attack, a defensive pivot) instead of assuming they stand
    # still. Defaults to True because robust coverage of reasonable opposing actions is a
    # required policy principle. The old offline gate proxy (poke-env's
    # SimpleHeuristicsPlayer) never Protects and doesn't target-optimize, so it
    # systematically punishes opponent-response modeling it doesn't itself exhibit
    # (2026-07 gate: myopic-only 81.7% vs search 73.3% vs SimpleHeuristicsPlayer, n=300
    # each) -- the search specifically models human ladder behaviors (Protect timing,
    # coordinated targeting) that a non-Protecting, non-coordinating bot opponent can
    # never demonstrate a benefit against. False must leave decide() byte-for-byte
    # identical to the pre-Phase-2c myopic-only behavior (score_joint_orders directly).
    # can still be selected explicitly for diagnostic A/Bs with
    # `ladder/run_ladder.py --myopic`; it is no longer the shipped decision process.
    use_two_ply_search: bool = True
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
    search_opp_candidates: int = 12
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
    # Raised 0.25 -> 0.4 in campaign iteration 4: ladder block 3's postmortem made
    # unnecessary_prediction co-dominant (6/14 losses = "the selected line had a severely
    # losing reasonable opponent response" -- 1200+-rated opponents actually find those
    # punishes), and the same-session A/B agreed: 0.4 won 96/100 vs 0.25's 92/100 (with
    # a heavier expected_death_cost_weight adding nothing on top). 0.4 keeps the
    # likelihood-weighted expectation dominant while hedging real punishes harder.
    search_worst_case_weight: float = 0.4
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
    # Make exchange search use the real geometric success odds for OUR repeated
    # Protect-family moves, matching `_score_protect`. Before campaign iteration 8 the
    # myopic score decayed correctly but `resolve_exchange` still treated every repeat
    # as guaranteed, which drove consecutive Protect attempts in 10/13 Block-7 losses.
    # False exists only as the exact legacy-control path for same-session A/Bs.
    search_respect_our_protect_odds: bool = True
    # Number of strategically distinct non-damaging opponent actions retained per slot
    # (setup, denial, speed control, redirection, screens). These sit alongside attacks
    # and Protect so the robust-play check cannot ignore a free Trick Room/Tailwind/setup.
    search_opp_utility_per_slot: int = 2
    # Number of previewed bench candidates retained as plausible defensive switches for
    # each opposing slot. Search resolves these before moves, like the real engine.
    search_opp_switches_per_slot: int = 2
    # Exchange-value cost when the opponent successfully establishes an important
    # non-damaging effect. Individual utility actions scale this shared currency.
    search_opp_utility_weight: float = 25.0

    # --- Phase 3b: persistent context + short rolling horizon --------------------------
    # Master switch for the persistent-context future-position score. Promoted after
    # 2026-07-22 local gates: 98/100 vs heuristic (Wilson low .930, threshold .65),
    # 99/100 vs random (Wilson low .946, threshold .90), and a side-swapped 200-game
    # mirror against the old shallow search was neutral (95/200, CI contains .5). False
    # remains the exact shallow-search control path for future ladder A/Bs.
    use_rolling_horizon: bool = True
    # Additional projected turns AFTER the normal searched exchange. Two is long enough
    # to recognize setup/payoff and looming traps without pretending our compact damage
    # model is a full Showdown simulator.
    rolling_horizon_turns: int = 2
    # Blend weight on the projected joint-position value. Kept below the immediate
    # exchange's 1.0 weight because uncertainty grows each projected turn.
    rolling_horizon_weight: float = 0.45
    # A bench Pokemon is a safe pivot when the two opposing actives' combined best
    # expected damage stays below this percent of its maximum HP.
    rolling_safe_switch_damage_ceiling: float = 55.0
    # Position-value bonus per safe pivot we retain relative to the opponent.
    rolling_safe_switch_bonus: float = 12.0
    # Penalty for each active slot projected to die with no safe pivot and no projected
    # opposing removal -- the concrete "Protect will soon be the only good move" state.
    rolling_trap_penalty: float = 45.0
    # Reward/penalty for advancing or losing the battle's persistent win-condition plan
    # (removing a plan-breaker or losing the planned closer).
    rolling_plan_progress_weight: float = 25.0
    # How strongly observed opponent move repetition biases otherwise-similar response
    # likelihoods. 0 disables history's ranking effect while retaining its trace.
    battle_history_response_weight: float = 0.25
    # Search a strategically diverse top-K (best switch/control/non-Protect lines as well
    # as raw myopic leaders) so a setup line cannot be pruned before horizon evaluation.
    search_diverse_candidates: bool = True

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
    # Explicit first-principles preview terms. The raw damage/speed proxy remains useful,
    # but these ensure engine denial, a closer, functional leads, and repair-capable backs
    # can change the selected four rather than existing only in a trace.
    team_preview_engine_answer_bonus: float = 22.0
    team_preview_closer_pick_bonus: float = 28.0
    team_preview_closer_back_bonus: float = 10.0
    team_preview_lead_function_bonus: float = 12.0
    team_preview_passive_lead_penalty: float = 30.0
    team_preview_backline_safety_weight: float = 0.2
    team_preview_role_coverage_bonus: float = 7.0
    team_preview_default_mega_bonus: float = 10.0
    team_preview_extra_mega_penalty: float = 6.0
    team_preview_lead_engine_denial_bonus: float = 15.0
    team_preview_second_speed_mode_bonus: float = 10.0
    team_preview_balanced_structure_bonus: float = 8.0

    # --- Iteration 6: symmetric opponent preview prediction (vgc.preview_predict) ------
    # Softmax temperature for `vgc.preview_predict.predict_preview_choice`'s distribution
    # over the 90 (pick, leads) candidates: `p_i ∝ exp(score_i / temperature)`. Lower =
    # sharper/more confident the predicted opponent always takes their single
    # highest-scoring option; higher = flatter, closer to uniform over all 90. 20.0 is a
    # first-pass default (candidate score gaps in this scoring scale typically run
    # tens of points -- see `_score_choice`'s per-term weights -- so a handful of nats
    # of separation between the top few candidates is plausible without the whole
    # distribution collapsing onto one option); also used by
    # `predict_preview_hybrid`'s lead-pair softmax WITHIN a fixed bring-4 (see that
    # function's docstring). Not directly re-tuned by the backtest (which only compares
    # ranked orderings, insensitive to this knob's exact value), only the two GATED
    # knobs below were.
    preview_prediction_temperature: float = 20.0
    # Master switch: True scores our 90 team-preview candidates against
    # `vgc.preview_predict.predict_preview_hybrid`'s predicted opponent bring/lead
    # distribution (likelihood-weighted + a worst-case hedge, see
    # `team_preview_opponent_worst_case_weight` below) instead of the flat
    # assume-all-6-equally-likely baseline `_score_choice` used before this feature.
    # Defaults to True (unlike `use_two_ply_search`/`use_bc_policy`/`use_value_head`,
    # which stay opt-in pending a real ladder A/B) because this one has a real backtest
    # gate behind it, over 4,871 real teampreview records
    # (`tools/backtest_preview_prediction.py`, iteration-6 plan): the hybrid predictor's
    # bring-4 top-1/top-3 accuracy (27.4%/41.2%) matches or slightly beats the strongest
    # baseline (top-4-by-corpus-usage: 27.4%/40.9%, essentially by construction -- the
    # hybrid's bring-4 IS that same usage signal), and its LEAD accuracy clearly beats
    # that same baseline (10.8%/21.1% vs 5.3%/15.3%, roughly 2x on top-1) -- the gate the
    # plan specified before integrating. The pure matchup-only predictor (no longer used
    # for bring-4) failed this gate badly on its own (7.7%/21.1% bring-4 top-1/top-3,
    # barely above the 6.7%/20.0% uniform-random floor) -- see
    # `vgc.preview_predict.predict_preview_choice`'s module docstring for why bring-4
    # needed the usage signal instead.
    use_preview_prediction: bool = True
    # Blend weight for `_score_choice`'s opponent-matchup term when
    # `use_preview_prediction` is on: `(1 - w) * likelihood_weighted_exchange_score + w
    # * worst_case_exchange_score`, where the likelihood-weighted term averages over ALL
    # 6 previewed opponent mons weighted by each one's marginal predicted bring
    # probability, and the worst-case term restricts that same average to ONLY the
    # single highest-probability predicted bring-4 (a sharper, less-diluted "what we
    # actually expect to face" signal) -- mirrors `search_worst_case_weight`'s blend
    # philosophy (a real opponent isn't a perfect predictable optimizer, but hedging
    # entirely against "any of their 6" when 2 are very unlikely to be brought is overly
    # conservative). 0.3 is a first-pass default in the same spirit as
    # `search_worst_case_weight`'s own tuning history (dropped from an initially-higher
    # value after over-weighting worst cases made search too passive) -- kept modest
    # rather than re-deriving from this backtest, which doesn't measure this specific
    # blend's in-battle effect.
    team_preview_opponent_worst_case_weight: float = 0.3

    # --- Final integration: BC v2 candidate re-ranker (vgc.bc.policy.score_orders) ------
    # Master switch: True blends the trained BC v2 checkpoint's learned move/target
    # log-probabilities into the top-ranked heuristic candidates' scores instead of
    # taking the heuristic's (search or myopic) argmax directly. Defaults to False for
    # the same reason use_two_ply_search does: this needs a real ladder A/B before it's
    # trusted as a default, not just an offline gate proxy -- opted in per session via
    # `ladder/run_ladder.py --bc`.
    use_bc_policy: bool = False
    # Converts nats (the BC model's log-probability units) into heuristic score points
    # for the additive blend `heuristic_score + bc_blend_weight * bc_logprob`. Tuned
    # empirically via same-session gate A/Bs (2026-07-17): at 30.0 the blend cost ~13
    # win-rate points vs heuristic-only (candidate log-prob gaps of 2-4 nats swung
    # +/-60-120 points, letting a 44%-top-1 imitation of ~average ladder players
    # override decisive damage-math calls); at 10.0 it was exactly proxy-neutral
    # (78/100 vs 78/100 same-session). 10.0 keeps the model advisory: ~1 nat of
    # preference (~10 points) flips genuinely close calls only.
    bc_blend_weight: float = 10.0
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
    # path in this codebase, e.g. TEAMS_DIR). Points at the value-head checkpoint
    # trained on corpus + self-play data (`bc_policy_v3sp.pt` -- 63%+ value accuracy /
    # 0.675+ AUC on the held-out corpus val split, vs the prior corpus-only
    # `bc_policy_v3.pt`'s 59.93%/0.6496 -- see runs/experiments.jsonl's self-play
    # entries). This checkpoint uses the PRE-v4 `bc-encoding-v2` layout (trained before
    # the preview-context encoder existed, which self-play showed doesn't pay off yet --
    # see that experiment log); `vgc.bc.policy.load_bc_policy`/`vgc.bc.model.BcPolicyNet`
    # serve it via their `legacy_v2_layout` compatibility path rather than refusing it.
    bc_checkpoint_path: str = "data/models/bc_policy_v3sp.pt"

    # --- Game-plan layer (vgc.gameplan.build_gameplan, via _Context.gameplan) ----------
    # `build_context` builds one `GamePlan` per turn (see vgc/gameplan.py) capturing the
    # win-condition framework VGC players reason about explicitly -- "who is our win
    # con", "who beats it", "what's our answer to their scariest threat" -- as a
    # complement to the myopic per-turn damage/KO scoring above, which has no notion of
    # a game-spanning plan. These three weights are how that plan actually influences
    # scoring; see `_score_attack_order`/`_score_switch` for exactly where each is read.
    #
    # If the ACTOR slot IS our primary win condition and it's facing a >=100%-HP threat
    # this turn with no KO of its own that resolves first, staying in and attacking is
    # penalized by `weight * threat.percent` -- protects the win con from being traded
    # away on a turn where retreating (switch/Protect, which never pass through
    # `_score_attack_order` and so never see this penalty) was available. Named
    # "preservation" rather than e.g. "retreat_bonus" since the mechanism is a penalty on
    # the risky choice, not a bonus on the safe one -- symmetric either way.
    win_con_preservation_weight: float = 0.4
    # Expected-death cost for ANY of our mons (not just the win con) staying in under a
    # lethal threat: when the bigger of the single-threat percent and the (weighted)
    # combined double-target threat on the ACTOR slot is >= 100 and this order doesn't
    # itself remove the threat before it lands, the order pays
    # `weight * min(200, that threat percent)` -- pricing the mon's likely faint into the
    # myopic argmax the way the search's exchange delta already prices it one layer up.
    # Added after ladder block 2's postmortem put lost_positioning first (9/16 losses,
    # "stayed exposed to a combined KO threat"): stay-in chip attacks previously paid
    # NOTHING for the actor dying, so a doomed Psychic always outbid a defensive switch
    # (switch orders never pass through `_score_attack_order`, so they escape this cost
    # exactly like they escape win_con_preservation_weight's). 0.5 puts a full-lethal
    # (100%) exposure at ~50-100 points: a doomed chip attack (~60-90 pts) now loses to a
    # decent switch, while a guaranteed KO elsewhere (~205 pts) still trades on purpose
    # -- deliberate sacrifices for a KO remain legal, mindless exposure does not. Stacks
    # with win_con_preservation_weight on the win con by design (the win con should be
    # MORE protected than the average team member, not equally).
    expected_death_cost_weight: float = 0.5
    # Flat bonus added to a target's contribution in `_score_attack_order` when that
    # target is one of `GamePlan.plan_breakers` -- KOing the piece that specifically
    # invalidates our win con re-enables the whole plan (the Chandelure-vs-Torkoal case:
    # Chandelure is worth killing even off-plan-value, because its mere presence is what
    # was stopping Torkoal from winning games). Comparable in scale to
    # `likely_ko_bonus` (35.0) since "restores a game plan" is roughly as valuable as a
    # real (if not guaranteed) KO.
    plan_breaker_target_bonus: float = 25.0
    # Weight on `_score_switch`'s collapsed-matchup term: for the OUTGOING (currently
    # active) mon in that slot, compares its CURRENT best expected damage % onto the
    # field's actives (real weather/field, via the same `_best_attacking_move` math
    # `_score_switch` already runs) against `GamePlan.table_percent`'s NEUTRAL-field
    # expectation for that same matchup. When current has collapsed to below
    # `collapsed_matchup_floor` of the table value, every order that switches this mon
    # out gets `weight * (table_pct - current_pct)` added. This is the Charizard-in-rain
    # fix: rain gutting a Fire-type's damage makes current << table, generating real
    # switch pressure that pure current-field scoring (which already just sees "this mon
    # is doing less damage than usual", not "this mon's WHOLE role has collapsed") was
    # underweighting relative to the tempo cost of switching.
    collapsed_matchup_switch_bonus: float = 0.5
    # Fraction of the gameplan-table (neutral-field) expected % below which the current
    # (real-field) expected % counts as "collapsed" for collapsed_matchup_switch_bonus
    # above -- 0.5 means "doing under half of what this matchup should do" triggers real
    # switch pressure; small weather-driven fluctuations above that floor don't.
    collapsed_matchup_floor: float = 0.5

    # --- First-principles turn layer ---------------------------------------------------
    # Target-selection bonuses for the opponent's whole-team primary threat and explicit
    # engine enablers (weather/TR/Tailwind/screens/redirection/pivot cycle).
    primary_threat_target_bonus: float = 18.0
    engine_enabler_target_bonus: float = 15.0
    # Action-economy values for utility/control moves that the old generic status fallback
    # scored as zero. Accuracy is folded into the move-specific scorer when available.
    redirection_base_value: float = 20.0
    taunt_base_value: float = 28.0
    encore_base_value: float = 20.0
    yawn_base_value: float = 22.0
    generic_sleep_value: float = 32.0
    burn_base_value: float = 24.0
    setup_base_value: float = 24.0
    unsafe_setup_penalty: float = 32.0
    wide_defense_base_value: float = 22.0
    recovery_base_value: float = 18.0
    # Speed-lowering attacks and Tailwind/TR are worth more when they flip the partner's
    # order into an immediate attack/KO, not merely because speed control exists.
    speed_drop_target_value: float = 10.0
    speed_control_immediate_ko_bonus: float = 25.0
    # Expected action denial from flinch-bearing moves other than Fake Out.
    generic_flinch_weight: float = 20.0
    # Explicit two-slot pressure synergies.
    dual_target_pressure_bonus: float = 12.0
    fake_out_setup_bonus: float = 18.0
    redirection_setup_bonus: float = 22.0
    protect_partner_cleanup_bonus: float = 20.0
    # Information/position components of Protect and switching.
    protect_information_per_unknown: float = 3.0
    protect_field_stall_per_turn: float = 5.0
    protect_reposition_bonus: float = 10.0
    switch_activation_bonus: float = 10.0
    switch_safe_both_bonus: float = 15.0
    switch_endgame_preservation_bonus: float = 15.0
    # Hard double-target threats are built from both opposing slots rather than only the
    # single strongest move; this weight makes that realistic danger affect preservation.
    double_target_threat_weight: float = 0.35

    # --- Phase 3: outcome value head (vgc.bc.policy.position_value, in vgc.search) -----
    # Master switch: True adds the value head's judgment of each simulated exchange's
    # resulting position to that exchange's value in vgc.search's 2-ply search, ON TOP
    # of the existing HP/faint-based `_exchange_value` -- the move/target heads only
    # ever learn to imitate an average-rated ladder player's CLICK (an imitation
    # ceiling); a value head trained on game OUTCOMES has no such ceiling, since "who's
    # actually winning this position" doesn't care that the players it learned from were
    # average. Defaults to False for the same reason use_bc_policy/use_two_ply_search do
    # -- needs a real ladder A/B, not just an offline proxy, before it's trusted as a
    # default. Requires BOTH this flag AND the checkpoint at `bc_checkpoint_path`
    # actually carrying a value head (see `vgc.bc.policy.BcPolicy.has_value_head`) --
    # silently a no-op (falls back to the pre-value-head exchange value) if either is
    # false, same graceful-degradation contract as `use_bc_policy`. Opted into per
    # ladder session via `ladder/run_ladder.py --value`.
    use_value_head: bool = False
    # Converts a win-probability delta (the value head's sigmoid output is in [0, 1])
    # into the search's percent-of-HP currency: `value_head_weight * 100 * (v_after -
    # v_before)`. 100x puts a full 0->1 probability swing (e.g. "this exchange looks
    # like a guaranteed loss" -> "a guaranteed win") on the same scale as losing/gaining
    # 100% of a Pokemon's HP -- deliberately the SAME scale `search_hp_weight`/
    # `search_faint_weight` already use, so a 10% win-probability swing (10 points) is
    # roughly comparable to a 10%-HP swing, not dominating or negligible by construction.
    value_head_weight: float = 1.0
