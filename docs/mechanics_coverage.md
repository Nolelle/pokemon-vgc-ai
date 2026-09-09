# Champions VGC mechanics coverage audit

Audit date: 2026-08-27 (Reg M-B snapshot). This file classifies the **Python**
search forecast (`vgc.search`), not the exact Showdown teacher path. M-C legality
(Rocky Helmet, Octolock/Grapploct, expanded species) flows through
`data/champions/*.json`; re-audit the ladder-team gap table before treating those
rows as current.

## Bottom line

The bot does **not** misunderstand the rules used to run the battle. Pokemon Showdown
still enforces the real Champions rules. The weakness is in the bot's internal forecast:
it predicts what a turn will look like before choosing an action, and that forecast is
not yet a complete copy of the Showdown engine.

The earlier first-principles checklist answered, "Does this idea affect the score?" It
did not answer, "Is this mechanic simulated exactly?" A score bonus for Fake Out or
Helping Hand is not the same as actually cancelling the target's move or multiplying the
partner's damage. This audit uses the stricter question.

The hybrid training result remains valid for what it measured: the neural shortlist did
not make the existing full search weaker. It did **not** prove that the shared search
teacher understood every Showdown mechanic. Both sides of that comparison used the same
forecast, so a blind spot shared by both sides was invisible to the test.

## Meaning of the labels

- **Exact:** the forecast changes the battle state in the relevant way and is checked
  against Showdown or a direct mechanics test.
- **Approximate:** the idea affects the choice, but some outcomes, probabilities, or
  follow-on effects are simplified.
- **Score only:** the bot assigns a strategic bonus or penalty without performing the
  mechanic in its turn forecast.
- **Missing:** the forecast does not account for it in a useful way.
- **Not applicable:** the Champions format disables the mechanic.

## High-level map

| Area | Status | What is true today |
|---|---|---|
| Format legality and Champions data | Exact | Species, moves, items, Mega forms, Stat Points, and learnsets come from the local Champions mod; teams can be checked by Showdown's validator. |
| Core stats and ordinary damage | Exact for covered inputs | Stat Point conversion and the ordinary damage formula have direct Showdown comparison tests. |
| Legal action generation | Exact at decision time | `poke-env` supplies the moves, switches, targets, and Mega options Showdown currently allows. |
| Priority and speed order | Approximate | Priority, Trick Room, Tailwind, weather Speed abilities, paralysis Speed, and Choice Scarf are included. Speed ties are deterministic instead of a 50/50 branch. |
| Damage-move accuracy | Missing | A 70%, 80%, or 90% accurate damaging move is currently forecast as if it always hits. This affects current-team moves such as Heat Wave, Leaf Storm, and Rock Slide. |
| Damage rolls | Exact in expectation | The 16 legal damage rolls are averaged for search. That is an expected value, not 16 separate future branches. |
| Protect | Approximate | Ordinary Protect success and repeat-use odds are included. Side-wide guards such as Wide Guard are not mechanically applied. |
| Switching | Approximate | Switching, entry weather, and some entry abilities affect scoring. Many entry effects and the opponent's exact brought four remain uncertain. |
| Mega Evolution | Approximate | Our selected Mega's stats, type, ability, and weather are used. Opponent Mega Evolution on the current turn is not searched. |
| Terastallization | Not applicable | The Champions mod disables it. |
| Later-turn forecast | Approximate | It is a compact damage race, not a complete multi-turn Showdown simulation. |

## Status and action-denial mechanics

| Mechanic | Status | Forecast behavior |
|---|---|---|
| Sleep | Approximate after this fix | Target choice, sleep-move accuracy, Grass/powder immunity, Overcoat, Safety Goggles, Soundproof, sleep-blocking abilities, Sweet Veil, Leaf Guard in sun, Safeguard, Electric/Misty Terrain, Chesto/Lum Berry (including Unnerve), Protect, Early Bird, and the Champions 2-or-3-action duration are included. The longer forecast uses an expected-value simplification rather than branching every possible wake turn; Substitute, Uproar, Magic Bounce reflection, and unusual groundedness changes remain gaps. |
| Burn | Approximate | The physical-damage reduction is included. End-of-turn burn damage is not. |
| Paralysis | Approximate | The Speed reduction is included. The Champions mod's 1-in-8 chance to lose the action is not. |
| Freeze | Missing | The Champions duration and thaw chances are not forecast. |
| Poison and toxic poison | Missing | End-of-turn damage and toxic's increasing counter are not forecast. |
| Confusion | Missing | Self-hit and recovery chances are not forecast. |
| Flinch and Fake Out | Score only | They receive strategic value, but the exchange forecast does not cancel the target's action. |
| Yawn | Score only | The delayed sleep is valued but not placed onto the future battle state. |
| Taunt, Encore, Disable | Score only | Denial is valued, but the affected move list is not restricted in the forecast. |

## Common move effects

| Mechanic | Status | Important limitation |
|---|---|---|
| Tailwind, Trick Room, screens | Approximate to exact | Their main field effects are carried into search and the short forecast; turn expiry and every interaction are not fully reproduced. |
| Redirection | Approximate | Follow Me/Rage Powder can reroute single-target moves. All immunity and order edge cases are not proven complete. |
| Helping Hand | Score only | Partner support is valued, but the exchange does not apply the exact damage multiplier. |
| Stat boosts and drops | Score only | Moves such as Nasty Plot, Swords Dance, Icy Wind, and Parting Shot receive strategic values, but most stage changes are not applied to later damage and Speed calculations. |
| Intimidate | Score only | Switching an Intimidate user receives value, but the opponents' Attack stages are not changed in the forecast. |
| Wide Guard, Quick Guard, Mat Block, Crafty Shield | Score only | These moves receive defensive value but do not block the matching actions in the exchange. |
| Recoil, drain, and crash damage | Missing | The resulting health changes are generally not applied. |
| Multi-hit moves | Missing or simplified | Hit-count distributions and repeated interactions with items/abilities are not reproduced. |
| Secondary effects | Mostly missing | Poison, burns, stat drops, and other secondary effects from damaging moves are generally ignored; a few receive score bonuses. |
| Charge and recharge moves | Approximate | Solar Beam and Hyper Beam are deliberately optimistic inside the exchange; charge/recharge turns are not fully represented. |
| Critical hits | Missing | Critical-hit branches are not part of the forecast. |

## Damage formulas, items, and abilities

The exported Champions catalog lists 509 legal moves (M-C, 2026-09-09). Of those, 42 use
a special damage formula or fixed-damage rule in Showdown. The calculator explicitly
handles the ordinary formula plus these special families:

- Weather Ball;
- Water Spout and Eruption;
- Electro Ball and Gyro Ball;
- Grass Knot, Low Kick, Heavy Slam, and Heat Crash.

Other special families—including fixed damage, Counter-style retaliation, Endeavor,
Flail/Reversal, Stored Power, Super Fang, and several multi-hit formulas—are unsupported
or simplified. A supported/unsupported flag exists for several of these, but coverage is
not complete enough to treat every returned number as exact without checking the move.

Items and abilities use explicit allowlists for direct damage and Speed effects. That is
safer than guessing, but it means an unlisted effect is normally ignored. We must audit
each team we put on the ladder instead of assuming that all legal items and abilities are
covered.

## Current ladder team's important gaps

| Pokemon | Covered well enough | Material blind spots |
|---|---|---|
| Charizard | Mega stats, Drought, sun, Weather Ball | Heat Wave accuracy; Solar Beam charging; several secondary effects |
| Farigiraf | Trick Room, ordinary attacks | Armor Tail priority blocking; Helping Hand's exact multiplier; Colbur Berry |
| Venusaur | Chlorophyll, ordinary damage, sleep after this fix | Focus Sash survival; Leaf Storm accuracy and Special Attack drop; Sludge Bomb poison |
| Garchomp | Ordinary damage, Life Orb | Rock Slide accuracy/flinch; Rough Skin; recoil/contact interactions |
| Incineroar | Ordinary attacks and a switch-in Intimidate bonus | Fake Out action denial; actual Intimidate drops; Parting Shot drops/pivot; Sitrus Berry; Flare Blitz recoil |
| Sylveon | Ordinary attacks and Fairy Feather | **Pixilate type conversion and power boost are not in the damage allowlist**, so Hyper Voice, Hyper Beam, and Quick Attack can be valued with the wrong type and power; Hyper Beam recharge |

Pixilate is the most direct current-team damage error. Damaging-move accuracy, Focus
Sash, and true action denial can also reverse a choice rather than merely adjust its
score. These should be fixed before treating another public ladder sample as a clean
measurement of the hybrid approach.

## Why training did not expose this

1. The neural model learned to rank choices made by the full-search teacher. If the
   teacher treated Sleep Powder as a flat utility score, the learner inherited that
   worldview.
2. The hybrid safety test compared guided search with full search. Both paths called the
   same mechanics forecast, so shared errors cancelled out in the comparison.
3. Offline battles are run by a simplified direct environment for scale. They are useful
   for fair A/B comparisons, but they are not a completeness test against the real
   Showdown engine.
4. The old checklist used **Yes** to mean "a named signal exists." That wording was too
   strong and allowed score-only approximations to look mechanically complete.

This is an evaluation-design failure, not evidence that more training data is needed.
The right next step is a mechanics gate: test the prediction layer against Showdown one
mechanic at a time, beginning with mechanics used by the ladder team.

## Release rule from this audit

Before the next meaningful ladder run:

1. Fix and directly test Pixilate for the current team.
2. Apply damaging-move accuracy in the exchange forecast.
3. Mechanically apply the action denial we depend on most: Fake Out/flinch, paralysis,
   and freeze.
4. Apply Focus Sash/Sturdy survival.
5. Replace current-team score-only effects—Helping Hand, Intimidate, and Parting Shot—
   with state changes, or explicitly exclude decisions that rely on them.
6. Add small Showdown comparison cases for every completed mechanic.

Do these as separate, reviewable changes. Re-run the existing hybrid-vs-full-search
gate afterward: the search teacher itself will have changed, so the old strength result
cannot automatically certify the new decision policy.
