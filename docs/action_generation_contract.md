# Problem C: action-generation contract

This document defines what counts as a legal choice, where legality comes from,
and how we prove that every logged action can be turned back into a command
Pokemon Showdown accepts. The current checkout passes this contract; the evidence
is recorded below.

## Takeaway

The unit we care about is **one legal choice at one decision**, not a strategy
score. For every team-preview request, joint move request, and forced-switch
request, the system must list every legal option, never invent an illegal one,
and save its pick in a form that converts back into a Showdown command.

Legality is delegated, not hand-listed. The Showdown request message parsed by
poke-env is the authority for what is legal. `vgc.actions` only combines the
two per-slot lists and filters the pairs Showdown forbids. This is why a new
move, item, or Mega — including Regulation M-C additions — never needs a new
action table; it arrives inside the request.

A human-readable label such as `earthquake@1 / protect` is not a command. The
command is the order object's wire message (`/choose move ...`, `/team ...`).
Both are saved, but only the wire form proves the round trip.

## 1. Terms and authority

- **Decision:** one request for a team-preview order, joint moves, or forced
  switches. Same identity as Problem B: battle id, side, decision sequence,
  turn, phase, request sequence, observation cutoff.
- **Joint order:** what both active Pokemon do together on one move turn, for
  example attack-plus-Protect or attack-plus-switch.
- **Single order:** one slot's half: a move with its target, a switch to a
  bench species, a Mega-evolved move variant, or a pass.
- **Team-preview order:** a `/team XXXX` string. The first two digits are the
  leads, matching poke-env's own `random_teampreview` wire format.
- **Forced switch:** a replacement request after a faint. Only switches and
  passes are legal; moves are not.
- **Fallback:** the exception-safe random legal choice used when strategy code
  crashes or returns nothing. It must still be legal.
- **Wire message:** the exact string sent to Showdown (`order.message` for
  in-battle orders, `/team ...` for preview). This is what the round-trip test
  replays.
- **Display label:** the human summary from `describe_order`. Useful for traces
  and debugging, never sent to Showdown.

Pokemon Showdown's request message, as parsed into poke-env's
`DoubleBattle.valid_orders`, is the authority for legality. The local pinned
Showdown engine remains the authority for whether a replayed wire message is
accepted.

## 2. Two labels for every candidate

Every candidate at decision time is either:

- `legal`: present in the combined enumeration for this request, sendable
  without a Showdown rejection; or
- `illegal`: anything else, including a move that is disabled, has zero Power
  Points, targets an invalid slot, Mega evolves twice, or reuses a fainted
  slot.

There is no third state. An empty legal list is not an illegal action; it is
the defined trigger for the fallback path.

Every legal list records the request it came from. Every fallback records the
reason it fired. A fallback must never be mistaken for a strategy choice in
traces or training data.

## 3. Required enumeration by phase

### Team preview

- All ordered bring-four selections are known: 6-pick-4 produces 360
  `/team ...` wire strings (`itertools.permutations`, see
  `vgc.battle_state_replay._legal_actions`).
- `vgc.team_preview.build_team_order` scores all `C(6,4)` picks x `C(4,2)`
  lead pairs (90 candidates) and returns one `/team XXXX` string with leads
  first.
- Short or malformed teams never crash: fewer than four present returns
  `/team ...` over the slots that exist; no opponent preview returns
  `/team 1234`.
- Crash fallback is `/team 1234`, always legal for a 6-pick-4 request.

### Move (both slots to act)

- The full joint list comes from `vgc.actions.enumerate_joint_orders`:
  `battle.valid_orders` per slot, combined by
  `DoubleBattleOrder.join_orders`, which already filters Showdown's
  mutual-exclusion rules (both slots Mega evolve, both pass, and similar).
- Per-slot variants are preserved: move-plus-target, Mega variant, switch to
  a bench species, Protect-like status moves, setup moves, and pass when the
  request offers it.
- Disabled moves, zero-Power-Point moves, and invalid targets are excluded
  because the request excludes them, not because of a local ban list.
- An empty joint list triggers the fallback to `choose_random_move` /
  `choose_default_move`, the same path poke-env's own random doubles player
  uses.

### Forced switch (one or both slots must replace)

- The phase is `forced_switch` (see `vgc.battle_state_replay._phase`, which
  reads poke-env's `force_switch` flag, not the turn number).
- The legal list for this phase must contain only switches and passes. A move
  string in a forced-switch list is a contract violation.
- Mid-turn double faints produce a second decision with its own sequence
  number, request sequence, and cutoff. Turn number alone does not identify
  it.

## 4. Wire format and logging

Each saved decision keeps both forms:

- `legal_actions`: the sorted display labels for the full enumeration, plus
  its SHA-256 digest. Team preview keeps all 360 wire strings; in-battle
  phases keep the sorted `describe_order` labels.
- `chosen_order`: the display label of the pick, recorded by
  `DecisionReplayRecorder.record_choice` from `VgcPlayer.choose_move` /
  `teampreview` wrappers.
- The sendable form is the order object's `.message` (in battle) or the
  `/team ...` string (preview). `vgc.rl.env.choice_string` strips it to the
  bare sim input (`move ...`, `team ...`).

Display labels are for humans. The round-trip proof must use the wire form:
take the saved wire message and require it to be a member of the rebuilt
request's legal wire set -- the exact set Showdown accepts there. Live games
already sent these wires successfully; the rebuild check proves the log, not
just the live send, round-trips.

## 5. Round-trip rebuild algorithm

For each saved bundle and each decision in it:

1. Rebuild the player-side battle through the decision cutoff, exactly as
   Problem B does (same messages, same order, same request).
2. Recompute the legal list through the same functions used in live play
   (`_legal_actions` / `enumerate_joint_orders` / preview permutations).
3. Compare the recomputed list and its digest with the saved record. Fail on
   any mismatch.
4. Convert the saved `chosen_order_wire` to the rebuilt request's legal wire
   set (`legal_wire_messages` on the replayed battle) and require membership.
   The legal set is derived from Showdown's own parsed request, so membership
   is the request-state acceptance criterion -- no string guessing.
5. Require presence: fail if the wire form is missing or outside the rebuilt
   set. Fail on any mismatch; do not skip the decision.
6. Cover all three phases: team preview, move, and forced switch.

A display-label-only log that cannot be converted back does not satisfy this
gate, even if the live battle completed.

## 6. Hard cases that must be tested

- Normal doubles turn with moves, switches, Protect, and targeting variants.
- Double switch (both slots switch in the same joint order).
- Mega Evolution offered on one slot: the Mega variant is present, and the
  joint list forbids both slots Mega evolving together.
- Disabled move / zero Power Points: the move is absent from that slot's
  options on that turn.
- Trapped slot: the switch is absent for that slot.
- Fainted active with a live bench: forced-switch list has switches plus
  pass, no moves.
- Both slots fainted with an empty bench side: joint list is empty and the
  fallback path fires without an exception.
- Target coverage: every target Showdown offers for a chosen move appears as
  a distinct joint order (spread, single-foe, ally, self).
- Team preview: all 360 wire orders recorded; the chosen order is a member
  of that set; short-team and no-opponent fallbacks return a legal string.
- Fallback accounting: every fallback increments `VgcPlayer.fallback_count`
  and records a trace reason, so gates can detect silent random play.

## 7. No-illegal-action proof

The automated tests must include:

- **Exhaustion test:** on a live battle, every returned joint order's wire
  message is accepted by the request state that produced it.
- **Exclusion test:** a disabled or zero-Power-Point move, a trapped-slot
  switch, and a double-Mega pair never appear in the enumeration on a turn
  where the request forbids them.
- **Phase-purity test:** a forced-switch legal list contains `switch`/`pass`
  only; the strings `move ` and `@` ordering from move turns do not leak in.
- **Preview-membership test:** the chosen `/team ...` string is always a
  member of the saved 360-entry legal list.
- **Fallback test:** an empty joint list returns a legal random/default move
  instead of raising or sending a hand-built string.
- **Round-trip test:** every saved wire message is a member of the rebuilt
  request's legal wire set, for all three phases.

## 8. Completion gates

Problem C is complete only when all gates pass in the same clean checkout:

1. **Coverage:** every request type (preview, move, forced switch) lists its
   full legal set from the Showdown request, with Mega, target, switch,
   pass, disabled, and trapped variants preserved.
2. **Legality:** no illegal action is ever produced; mutual-exclusion pairs
   are absent; fallbacks are always legal and always counted.
3. **Round trip:** every saved choice converts to a wire message Showdown
   accepts at the rebuilt cutoff, for all three phases.
4. **Hard cases:** fainted partners, forced switches, target changes, preview
   membership, fallback accounting, and request fidelity (enumerated moves are
   a subset of the request's available moves; a trapped slot offers no switch)
   are all asserted live on real battles. The empty-list fallback path is
   pinned by unit test: an empty enumeration only arises when a side has
   nothing left to send, which ends the battle rather than producing a
   decision, so no live decision can exercise it.

## 9. Current checkout audit (2026-09-03, commit `a96ec07`)

### Confirmed implementation

- `vgc.actions.enumerate_joint_orders` combines `battle.valid_orders`
  through `DoubleBattleOrder.join_orders` (`src/vgc/actions.py:32`).
- `vgc.actions.describe_order` provides the display label
  (`src/vgc/actions.py:64`); `vgc.actions.choice_wire_message` provides the
  sendable wire message (`src/vgc/actions.py:18`).
- `VgcPlayer.decide` returns the top searched/scored joint order or a legal
  random move (`src/vgc/agent.py:274`); `decide_teampreview` returns
  `build_team_order` or poke-env random preview (`src/vgc/agent.py:323`).
- Exception-safe wrappers guarantee a legal fallback and count it
  (`choose_move` -> `choose_random_move`, `teampreview` -> `/team 1234`,
  `src/vgc/agent.py:334,381`).
- `vgc.team_preview.build_team_order` scores 90 candidates and returns
  `/team XXXX` with leads first (`src/vgc/team_preview.py:121`).
- `vgc.battle_state_replay._legal_actions` saves 360 preview wire strings
  and the sorted in-battle display list plus digests
  (`src/vgc/battle_state_replay.py:65`); `_phase` separates move from forced
  switch (`src/vgc/battle_state_replay.py:56`).
- Every decision saves both `chosen_order` (display) and
  `chosen_order_wire` (sendable), via `DecisionReplayRecorder.record_choice`.
- `vgc.battle_state_replay.legal_wire_messages` lists the sendable commands
  for the same request the display list covers.
- `vgc.rl.env.choice_string` converts an order to bare sim input
  (`src/vgc/rl/env.py:230`).
- `vgc.action_gate` pins the action coverage; `offline/check_action_readiness.py`
  prints the gate; ladder and all six training entry points refuse to run
  unless it says PASS.

### Verification evidence

- Coverage: 7 exact families, 0 partial, 0 missing; all four scopes ready.
- Gate tests: 19 passed (9 live against the local server, 10 unit).
- Round trip: every saved wire is a member of the rebuilt request's legal set
  at its own cutoff, for team preview, move, and forced-switch decisions
  (`test_saved_wire_replays_against_rebuilt_request`).
- Live legality: the sent wire is always a member of the live legal wire set
  and the saved wire matches the sent wire on every decision of a full game,
  with zero fallbacks; every enumerated move belongs to the request's
  available moves; wire messages are distinct per joint order.
- Exclusion: an Encore-disabled move disappears from later enumerations;
  an Infestation-trapped slot offers no switches while the untrapped slot
  keeps its own (`test_trapped_slot_offers_no_switch_live`, on a validated
  `trapper` team fixture); no double-Mega, double-pass, or same-switch pair
  is ever enumerated; no Mega variant is offered after evolving; a
  double-switch order is offered while the bench is live.
- Forced switch: replacement lists hold only switches/passes, with
  switch-bearing wires.
- Targets: one move with 2+ targets observed yielding distinct orders across
  two games; preview choice is a member of the saved 360-entry list.
- Unit: double-Mega, double-pass, and same-switch pairs excluded; empty
  enumeration returns `[]`; wire helper returns sendable messages;
  exception fallback is counted.

Run the fail-closed gate with:

```bash
.venv/bin/python offline/check_action_readiness.py
```

Problem C is **complete for action generation**. This does not claim the bot
chooses well, only that every choice it can produce is legal and every saved
choice replays as an accepted Showdown command.

## 10. Implementation order (completed)

1. ~~Freeze this contract (this file). No behavior changes.~~ Done `04ceacd`.
2. ~~Add the round-trip path: one function from saved choice to wire message,
   reused by live play logging and the verifier.~~ Done `bded541`.
3. ~~Add hard-case live tests: exclusion, phase purity, target coverage,
   empty-list fallback, preview membership.~~ Done `9ffffab`.
4. ~~Add a fail-closed action gate alongside the mechanics and battle-state
   gates, and require it in training and ladder entry points.~~ Done `a91c1c0`.
