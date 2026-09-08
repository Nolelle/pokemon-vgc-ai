# Problem B: battle-state understanding contract

This document defines what the agent must know before each decision, what must stay
uncertain, and how we prove that no future or private opponent information entered the
decision. The current checkout passes this contract; the evidence and remaining product
boundary are recorded below.

## Takeaway

The unit we care about is **one player decision**, not merely one numbered turn. A turn
can contain a normal two-Pokemon choice and then a later forced replacement choice. For
every such decision we need to reproduce the exact public facts the player had received,
the private facts that player legitimately owned, and a probability-weighted set of
possibilities for everything still hidden.

An ordinary saved Showdown HTML replay is not sufficient for this. It normally lacks the
player-private request message that lists currently legal moves and switches, and it
usually lacks full sets because Open Team Sheets are opt-in. The required saved artifact
is therefore a **decision replay bundle**: the visible message history, that player's
request messages, the player's own registered team, and a saved state check at every
decision.

## 1. Terms and authority

- **Observation:** one fact delivered to the player by Pokemon Showdown before a
  decision, such as damage, a switch, an ability activation, or a request for moves.
- **Decision:** one request for a team-preview order, joint moves, or forced switches.
- **Public state:** facts visible to that player at that decision.
- **Owned private state:** facts the player is allowed to know about its own registered
  team, including nature and Stat Points.
- **Belief state:** several coherent possibilities for hidden opponent facts, each with a
  probability. “Coherent” means each possibility is a legal set/team and its fields fit
  together; it is not an arbitrary mix of independently popular moves, item, and ability.
- **Decision cutoff:** the final received message allowed to influence one decision.
- **Replay:** feeding saved messages back through the same state builder in the same
  order. It does not mean reading the finished battle backward.
- **State digest:** a SHA-256 fingerprint of the canonical saved state. Equal digests
  mean the serialized states are byte-for-byte equal.

Pokemon Showdown's player-facing protocol and request message are the authority for what
was observable and legal. The local pinned Showdown engine remains the authority for how
each belief possibility evolves after an action.

## 2. Three labels for every field

Every decision-time field must have one of these labels and a source:

- `known`: supplied by our registered team or already revealed in a player-visible
  Showdown message;
- `estimated`: one or more probability-weighted possibilities derived only from data and
  observations available before the decision cutoff; or
- `unknown`: not safely known and not represented by a trustworthy estimate.

`None`, an empty string, and zero must not ambiguously mean all three things. A hidden
opponent item, an item that was consumed, and a Pokemon holding no item are different
states.

Every known field records the observation sequence that revealed it. Every estimated
field records the prior-data version and the observations used to update it. This makes a
future-information leak auditable instead of relying on code review alone.

## 3. Stable decision identity and ordering

Each saved decision has this identity:

- battle id;
- player side (`p1` or `p2`);
- monotonically increasing decision sequence number;
- Showdown turn number;
- phase: `team_preview`, `move`, or `forced_switch`;
- request message sequence number;
- observation cutoff sequence number.

The decision sequence number is the primary ordering key. Turn number alone is
insufficient because a faint or pivot move can create another decision during the same
turn.

Each Pokemon also needs a stable team-slot identity. Species is a useful display value,
but the slot identity must survive Mega Evolution, form changes, Transform, Illusion
breaking, switching, and nickname changes.

## 4. Required known-state snapshot

The canonical state before every decision must contain the following.

### Battle and request

- format id, generation, doubles mode, turn, phase, waiting/finished state;
- raw player-visible request JSON;
- every legal joint action, including targets, switches, Mega availability, forced
  passes, and disabled choices;
- whether the request is a normal move request or a replacement request;
- the previous public action and result needed to interpret the current request.

### Each side's roster and resources

- all six preview species;
- our selected four and exact bench;
- opponent Pokemon confirmed to have appeared;
- opponent preview Pokemon not yet revealed as brought, explicitly marked unknown;
- active slot occupancy and stable team-slot identity;
- benched, fainted, and still-available Pokemon;
- remaining Mega resource and whether Mega Evolution has already been used;
- remaining move Power Points when legitimately known;
- consumed, removed, changed, or still-held item state when legitimately known.

The opponent's unseen selected four must not be inferred as known. A predicted bring-four
is a belief, not a roster fact.

### Each Pokemon

- current and base species/form, Mega state, types, level, gender, and weight when
  observable;
- current health, maximum health when known, fainted state, and public health display;
- major status and publicly knowable counter information;
- all seven stat stages: Attack, Defense, Special Attack, Special Defense, Speed,
  accuracy, and evasion;
- temporary effects and counters, including Protect chain, recharge, preparation,
  trapping, Substitute, confusion, and similar effects;
- known moves, current Power Points, disabled state, last move, and move locks;
- known current/base/temporary ability;
- known current/original item and whether it was consumed or removed;
- exact nature, Stat Points, Individual Values, and calculated stats for our side;
- explicit unknown markers for hidden opponent spread and exact stats.

### Field, sides, and slots

- weather and its public duration information;
- terrain and its public duration information;
- Trick Room, Gravity, Magic Room, Wonder Room, and other whole-field effects;
- Tailwind, screens, safeguards, hazards, Wide Guard-style effects, and other side
  conditions for both sides;
- slot conditions, future attacks, delayed healing, and their public counters;
- speed-order-relevant state, including paralysis, item/ability reveals, and active
  speed-control effects.

### Knowledge available through Open Team Sheets

Species, moves, item, ability, and Mega-stone information become `known` only if the
player actually received the Open Team Sheets message before the decision cutoff. Our
own registered set is always known to us. Nature and Stat Points remain hidden for the
opponent even when sheets are open.

If sheets were rejected or the opponent did not also accept, the system must learn those
fields only from later public reveals. The saved artifact records accept/reject status
and the exact message that made a sheet available.

## 5. Belief-state contract

One opponent possibility contains a complete, mutually compatible assignment for the
hidden facts that can affect the decision:

- which four preview Pokemon were brought and which two are on the unseen bench;
- moves, item, ability, nature, and Stat Point spread for relevant Pokemon;
- exact health values consistent with any rounded public health display when that can
  change damage outcomes;
- privately rolled sleep/confusion duration still consistent with observed turns;
- unrevealed form or ability possibilities when legal and relevant.

Each possibility contains:

- a normalized probability greater than zero;
- the source/version of its pre-battle frequency estimate;
- the observations that increased or decreased its probability;
- a legal-team validation result;
- a stable fingerprint so the same possibility can be recognized across decisions.

The probability update rules are:

1. Start from frozen public-data frequencies or an explicitly documented fallback.
2. Remove only configurations made impossible by a public reveal or format legality.
3. Reduce, but do not normally erase, a configuration after noisy evidence such as one
   damage roll or a possible Speed tie.
4. Renormalize probabilities after filtering and after any compute-budget cap.
5. Never use the simulator's opponent team, private random timers, later replay lines, or
   final result.

The belief must preserve correlations. For example, a move, item, and ability observed
together on real sets should remain together; combining the most common value of each
independently can create a configuration that no player used or that violates Item
Clause.

## 6. Which possibilities search must receive

The search system must receive more than one possibility whenever all of the following
are true:

- at least two legal possibilities retain material probability;
- they can change legal actions, move order, damage, survival, targeting, or the value of
  a candidate action; and
- the uncertainty has not been resolved by a public reveal.

A configurable branch budget is allowed, but a budget of one must not silently turn an
uncertain fact into a known fact. If compute is limited, use a deterministic selection
that covers probability mass and decision diversity, then report:

- total possibilities before the cap;
- possibilities searched;
- probability mass retained;
- which hidden fields differ;
- whether excluded possibilities could change the top action in a small audit search.

At minimum, the powered completion test must cover uncertainty in active Pokemon spreads,
move/item/ability sets, hidden sleep/confusion timers, and an unresolved opponent bench.

## 7. Decision replay bundle

Each battle bundle is versioned and contains:

- repository commit and dirty-state flag;
- Pokemon Showdown commit and dirty-state flag;
- format and policy configuration;
- our packed team or an exact recoverable reference plus its SHA-256 fingerprint;
- player side and Open Team Sheets policy;
- the ordered player-visible protocol messages;
- every player-private request message in its original position;
- our submitted choices;
- one record per decision containing the cutoff, canonical state JSON, state digest,
  belief JSON, belief digest, and legal-action list;
- final outcome as a label stored outside all decision inputs.

Secrets such as ladder credentials are never stored. Opponent private simulator state is
never stored in a promoted public-information dataset.

## 8. Replay rebuild algorithm

For each saved bundle:

1. Start from an empty player-side battle object and the player's own registered team.
2. Feed only messages up to the first decision cutoff, in original order.
3. Build the canonical state and belief through the same functions used in live play.
4. Compare state digest, belief support/weights, and legal actions with the saved record.
5. Feed the next observed messages and repeat for every normal and forced-switch
   decision.
6. Fail on the first mismatch; do not skip the decision or silently substitute a simpler
   state.

An HTML replay parser can remain useful for public human data, but it is a lower-detail
source and must not be claimed to satisfy this exact rebuild gate.

## 9. No-future-information proof

The automated leakage tests must include:

- **Same-prefix test:** two battles with identical player-visible messages before a
  decision but different opponent private teams, hidden timers, or future outcomes must
  produce identical known state, belief input, model input, and candidate ranking.
- **Truncation test:** parsing a replay only through decision N must produce the same
  decision-N state as parsing the saved full battle while enforcing the same cutoff.
- **Reveal timing test:** an item, ability, move, Mega, selected-four member, or Open Team
  Sheet field changes from unknown/estimated to known only at its reveal message.
- **Label separation test:** winner, future turns, teacher action, and teacher score may
  be training answers, but changing them must not change any decision input.
- **Private-root mutation test:** changing simulator-only opponent fields while holding
  the public observation fixed must not change state, beliefs, model features, or search
  roots.
- **Split audit:** historical private-root teacher data remains rejected and cannot be
  merged into a promoted dataset.

## 10. Completion gates

Problem B is complete only when all gates pass in the same clean checkout:

1. **Coverage:** every relevant poke-env battle, Pokemon, move, side, field, and request
   property is captured, disabled by this format, or explicitly derived.
2. **Replay equality:** a varied local corpus rebuilds every decision state, belief, and
   legal-action list exactly, including mid-turn forced switches.
3. **No leakage:** all same-prefix, truncation, reveal-timing, label-separation, and
   private-root tests pass.
4. **Belief breadth:** search receives several coherent configurations when material
   uncertainty remains; retained probability mass and branch sensitivity are reported.
5. **Belief updating:** public Speed, damage, move, item, ability, form, status, and bring
   reveals update later decisions without changing earlier saved decisions.
6. **Fail closed:** collection, training, and public play stop if a required decision
   snapshot cannot be built, a replay mismatch occurs, belief probability is invalid, or
   a newly added observable property has no classification.
7. **Showdown integration:** end-to-end local battles agree between the live state and
   replayed state under both accepted and rejected Open Team Sheets.

Recommended varied replay cases include weather, terrain, Tailwind, Trick Room, Mega
Evolution, status and hidden timers, redirection, item consumption, ability reveal,
pivoting, double faint, forced replacement, Illusion, Transform, form change, and a game
where fewer than four opponent Pokemon ever appear.

## 11. Completed checkout audit (2026-09-01)

### Confirmed implementation

- `vgc.mechanics_state.snapshot_battle` is a rich canonical snapshot of the current
  poke-env observation.
- The observation-contract test fails when a new poke-env property is not classified.
- `vgc.battle_memory.BattleMemory` records earlier public actions plus conservative Speed
  and damage evidence.
- `vgc.opponent_belief` keeps up to three spread/nature possibilities and reweights them
  from earlier public evidence without deleting noisy mismatches.
- `vgc.rl.hidden_state` branches over legal hidden sleep/confusion timers.
- Training labels now use `public_mirror_exact_showdown_teacher_v2`; promoted data rejects
  the historical private-root teacher.
- The information ledger labels own facts, opponent reveals, estimates, and future random
  outcomes separately.
- `vgc.battle_state_replay` saves the exact ordered player-visible protocol, private
  requests, submitted choices, canonical state, belief state, legal actions, and their
  digests at every decision. The verifier feeds that prefix through the live parser and
  fails on any mismatch.
- Protocol bursts are consumed one line at a time. A request cannot see a later line that
  happened to arrive in the same WebSocket message.
- `data/usage/set_priors.json` now retains move/item/ability combinations observed
  together in public replays. Reveals filter those combinations without inventing an
  independently mixed set.
- The live mirror branches jointly over hidden spread/nature, coherent set, brought four,
  and private timer possibilities. The default examines two decision-diverse
  representatives whenever uncertainty remains; the trace reports the original branch
  count, direct representative mass, and the full probability mass assigned to those
  representatives.
- `vgc.battle_state_gate` pins the replay schema and joint-prior fingerprint. Public
  ladder and every training entry point stop if a family becomes partial/missing, a
  scope is disabled, the prior changes without review, or defaults collapse search to
  one hidden configuration.
- Exact search can take longer than a WebSocket heartbeat. Move selection now runs on a
  worker thread, leaving poke-env's connection loop free to answer heartbeats and detect
  genuinely dead connections at its normal timeout.

### Verification evidence

- Readiness catalogue: 16 exact families, 0 partial, 0 missing; all six scopes ready.
- Regular tests: 922 passed.
- Showdown-backed integration tests: 84 passed, including accepted and rejected Open
  Team Sheets plus normal and forced decisions.
- Clean mechanics-complete hybrid smoke: one completed local win, 13 recorded decisions,
  198 visible messages, all 360 legal team-preview orders, 0 random fallbacks, and 0
  replay mismatches.
- Its 11 in-battle searches each received two representative hypotheses, with as many as
  8 joint spread/set/bring configurations before compression; all assigned probability
  mass remained represented.
- Reduced-width latency audit: 0.480 seconds for the old point guess and 0.780 seconds
  for the multi-belief path, about 1.63 times as long. This measures state handling, not
  playing strength.

Run the fail-closed gate with:

```bash
.venv/bin/python offline/check_battle_state_readiness.py
```

Problem B is **complete for battle-state understanding**. This does not promote the
available mechanics smoke checkpoint as a strong public-ladder model. A production
mechanics-complete checkpoint still needs its separate strength gates; state correctness
and playing strength are deliberately different claims.
