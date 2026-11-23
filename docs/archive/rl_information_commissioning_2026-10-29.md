# Section 6: using a previously revealed move

## Hypothesis and task

The existing public structured memory is enough to learn a decision that depends
on an earlier reveal. No recurrence or private-state features are added.

Actual legal Healing Wish / Explosion prefixes leave Primarina against Swampert.
Swampert previously attempted Counter or Mirror Coat while Primarina protected.
All current HP, species, stats, items and boosts are the same between variants.
Moonblast safely defeats Counter; Liquidation safely defeats Mirror Coat.
Only the public move reveal distinguishes the variants for the actor.

The native damage oracle gives Liquidation 78–140 and Moonblast 113–200 damage
across all normal/critical rolls, against 175 HP. Either wrong nonlethal attack is
reflected for lethal damage against Primarina's 155 HP. A critical Moonblast can
occasionally KO before reflection: the blind ceiling is not asserted to be exactly
50%. Both correct policies won 128/128 engine trials. Wrong Liquidation won 0/128;
wrong Moonblast won 5/128. Random play won 58/128 and 72/128.

The research-only blind adapter removes the two move reveals from public protocol
memory and parsed known moves before calling the unchanged production actor. It
does not change simulator state. Every network input tensor at the paired roots
is exactly equal under the blind adapter. The normal actor's inputs differ when
the reveal is present, and are exactly equal when the private move changes before
either move has been revealed. This tests a specific private-information boundary,
not all possible hidden sets or histories.

## Frozen comparison (before main training)

- Initialize from each section 5 endpoint, seeds 77/88/99. Both arms begin with
  identical weights per seed. Start a fresh Adam optimizer.
- Compare the ordinary actor with the reveal-blind adapter. Same policy sampling,
  task order, simulator seeds, PPO configuration and budget in both arms.
- 1,024 training games per arm and seed: 32 updates of 32 games, evenly balanced
  between variants. Learning rate 0.0003, four epochs, minibatch 64, gamma 1,
  lambda .95, entropy .01; terminal outcomes only, no teacher or shaping.
- Development: 64 games per variant at updates 0/8/16/24/32, future base 21,500,000.
  Primary model is the fixed endpoint, with no best-checkpoint selection.
- Confirmation: 256 games per variant, future base 22,000,000, both arms and all
  three seeds. Require each aware variant's point win rate >=95%, and an aware
  advantage >=25 percentage points in each seed. Report paired uncertainty and
  all seeds; these are bounded commissioning criteria, not ladder-strength claims.
- Run section 4 retention checks on aware endpoints and report any regressions;
  this specialized experiment does not automatically replace the full-battle model.

The 64-game pilot took 7.1 seconds including two 32-per-variant evaluations. It
changed the 32-game development wins from 15/17 to 12/20; no improvement claim.
No pilot weights enter the main comparison. A failing main comparison stops at
the frozen cap and receives a diagnosis, not an unreported extension.

Core source: certified snapshot `85ed1294eb113eb9632b99b84d18dd50de0f6f0b`.
The external research harness is separately SHA256-bound in every report.
Teams are validated against the pinned local M-C engine on every invocation.
All models remain diagnostic and unapproved for deployment.
