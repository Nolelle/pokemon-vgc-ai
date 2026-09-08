# RL roadmap (frozen 2026-08-09, amended 2026-08-11)

**Current direction (2026-09-07) supersedes the historical objective below:** build
trustworthy data and training, then demonstrate 1700+ Elo on the public Champions
Reg M-B ladder. Elo is the ladder's playing-strength rating. A specific learning method
is a means to that goal, not the goal itself. Follow the September audit implementation
plan first. Coaching app work stays deferred until the battle goal is demonstrated.

The scope-narrowing plan that replaces the ad-hoc PPO experiments in `runs/ppo/`.
Read this before proposing any change to `vgc.rl` or `selfplay/train_ppo.py`.

The objective is narrow and it is not "ship a bot":

> Build a Reg M-B battle policy whose strength **measurably improves with additional RL
> experience**, and build an experimental setup trustworthy enough to prove it.

Coaching, counterfactual analysis, and explanations are downstream consumers of that
policy. Team building and learned team preview are deferred (Phase 6).

## Why the reset

The July PPO work expanded scope before it had a readable signal. Evidence, from the
runs' own metrics files:

| Run | Bootstrap battles | Wall clock | Eval (n=60) vs `vgc-shallow-search` |
|---|---|---|---|
| `honest-500` | 500 | 241 s | 33.3% |
| `honest-2000` | 2,000 | 716 s | 28.3% |
| `honest-8000` | 8,000 | 2,500 s | 30.0% |

16x the data, no movement -- and at n=60 the 95% CI is about +/-12 points, so those are
one number, not three. Meanwhile the distillation's own
`train_vs_fresh_val_accuracy_gap` (teacher agreement on training states vs freshly
collected states) went **0.586 -> 0.668 -> 0.684**: more data fit the collected
distribution better and transferred worse. That is covariate shift from training across
6 archetypes x ~10 teams x randomized learner teams.

PPO hyperparameters are NOT the current problem. Throughput, state-distribution
stability, evaluation resolution, and environment trustworthiness are. Do not tune PPO
until Phase 3.

## What is already built (do not redesign these)

These exist and are correct; the roadmap consumes them rather than proposing them:

- **Candidate-aware joint-action policy/value net.** `CandidatePolicyValueNet`
  (`src/vgc/rl/model.py:31`) scores a padded set of complete legal doubles orders and
  masks illegal/padded candidates before normalization (`model.py:147`), with a value
  head on the shared trunk. Both slots' decisions are scored as ONE candidate, which is
  required -- `move 1 1` + `Helping Hand -> slot 1` is not two independent choices. No
  policy-network redesign is currently justified.
- **Constant entropy bonus.** `entropy_weight = 0.01` (`src/vgc/rl/ppo.py:33`), never
  annealed. VGC turns are simultaneous, so the game has mixed-strategy equilibria:
  `Protect 50% / attack 35% / switch 15%` can be strictly stronger than `Protect 100%`,
  and a deterministic policy is exploitable by construction. Keep it nonzero unless a
  population-level experiment says otherwise.
- **PPO + GAE, self-play snapshots, archetype pools, learner-team randomization,
  holdout evaluation, BC warm-start, potential-shaping infrastructure.** All present
  (commits `a0a270c`, `9aa97a6`, `0616e5e`, `afcee0c`). Phases 4-6 turn this code back
  ON; they do not build it.
- **Persistent direct BattleStream worker.** `tools/sim_worker.mjs` -- fogged `.p1`/`.p2`
  streams, JSON-lines protocol, complete battles, ~20x the old throughput.

## Encoding: schema-v4 stays

**The proposal to shrink the Phase 2 vocabulary to ~4 species / ~16 moves is rejected.**
`vgc/rl/encoding.py` deliberately reuses the schema-v4 BC vocab so
`CandidatePolicyValueNet.warm_start_state_encoder` (`model.py:151`) can load BC trunk
weights. Shrinking it would (a) throw away BC warm-start, (b) force a re-encode of the BC
dataset, and (c) guarantee a SECOND vocab migration at Phases 4-5 when teams diversify.
The claimed benefit is near zero: unused embedding rows receive no gradient, so a large
vocab costs parameters we never touch, not sample complexity. The extra parameter and
optimizer-state cost is accepted deliberately.

Preserve the full BC-compatible schema for the whole project unless an architecture
change shows a substantial MEASURED benefit.

## Phases

Each phase has an exit criterion. Do not start phase N+1 until phase N's criterion is
met and recorded in `runs/experiments.jsonl`.

### Phase 1 -- trustworthy high-throughput environment

Replace `poke-env transport -> websocket -> Showdown server` with a direct
`BattleStream` driver. Phase 1 is NOT complete merely because battles run fast; its
purpose is to prove the laboratory is trustworthy. A bug here makes every later "RL
didn't learn" result meaningless.

**Keep poke-env's parser, drop only its transport.** `DoubleBattle` is what
`vgc.actions.enumerate_joint_orders`, `vgc.evaluator`, `vgc.sets`, and
`vgc.rl.encoding` all consume; reimplementing the protocol parser is not in scope and
never will be. `AbstractBattle.parse_message` / `parse_request` are drivable directly,
so protocol lines from a local `BattleStream` feed a `DoubleBattle` we construct
ourselves -- no `PSClient`, no asyncio, no accounts, no OTS race shim, no
client-recreation-on-timeout.

Use `getPlayerStreams()`'s per-player `.p1`/`.p2` streams, not `.omniscient`: those are
already the fogged views the real server sends each player, so Phase 2's information
masking is correct by construction rather than reimplemented. `.omniscient` is read only
for the terminal `|win|`/`|tie|`.

#### Concurrency: per-battle serialization, not one global queue

The first direct worker measured ~63 games/sec, ~908 decisions/sec, 14.4 decisions/game
including Python-side JSON round trips. The remaining per-step cost is dominated by
`settle()` (`tools/sim_worker.mjs:129`), which burns event-loop ticks per step and
requires `quiet >= 2` before responding -- not by JSON size. So batching is not "combine
several commands into one message"; it is:

```
Battle A step --+
Battle B step --+
Battle C step --+-- Promise.all(...)   (settles overlap)
Battle N step --+
```

Today a single global `queue` (`tools/sim_worker.mjs:258`) serializes every command in
the process. That comment's reasoning ("throughput comes from one worker per core, not
concurrency inside one process") is right about CPU parallelism and wrong about latency
amortization. The correct model is **serialization per battle id, concurrency across
independent battle ids** -- per-battle ordering is what keeps `drainBuffers` correct, and
that guarantee survives running distinct ids concurrently.

#### Evaluation runs on the direct environment too

Training on the direct env while `offline/run_gates.py` / `offline/run_matches.py` still
go through poke-env + a real server is not an acceptable Phase 1 exit state. It gives us
two different execution paths for the same battles, and at ~3 games/sec an n=1000 gate
takes hours -- so the evaluation budget below (which is non-negotiable) becomes the new
bottleneck the moment Phase 3 starts. **Migrating the eval harness onto the direct env is
in Phase 1's scope.** Training and evaluation should differ in policy/opponent
configuration, not in simulator transport.

#### Throughput: three numbers, not one

A single `games/sec/core` figure measured with a scripted policy does not predict
experiment cost.

1. **`TPS_sim`** -- cheap scripted/random policy. Measures BattleStream + Node event loop
   + IPC + parser + env stepping. This is the ceiling.
   Target guideline: **>= 200 games/sec/core** after concurrent batching. A target, not
   an absolute requirement.

   **Measured: the per-core target is NOT met, the aggregate one is.** On the fixed
   mirror, one worker: 51 games/sec sequential, 67 batched at width 32 (batching
   plateaus there). Profiling the sequential loop: 77% of wall clock inside the worker
   round trip, 12% `enumerate_joint_orders`, 11% our parsing. Since batching only
   recovers ~1.3x, most of that 77% is real simulator CPU work, not idle settle-waiting.

   So the amendment claiming "per-decision IPC overhead remains the major bottleneck"
   was wrong, and `tools/sim_worker.mjs`'s original comment -- throughput comes from one
   worker PROCESS per core, not concurrency inside one process -- was substantially
   right. Across processes (12-core machine): ~45 games/sec at 1 worker, ~38 each at 4,
   ~30 each at 8, i.e. **~240 games/sec aggregate at 8 workers**. Batching is still
   worth having (it is free and composes with multiprocessing), it is just not where the
   order of magnitude lives.
2. **`TPS_train`** -- end to end with the real encoder, real candidate enumeration,
   `CandidatePolicyValueNet` forward pass, policy sampling, opponent policy, reward, and
   trajectory storage. This is the number that sets the cost of a 1e5-1e6 battle
   experiment.

   **Measured, and the answer is not the network.** Single worker, one thread, PPO net
   vs `random` on the fixed mirror:

   | configuration | games/sec | rollout steps/sec |
   |---|---|---|
   | `PpoConfig()` default | **1.6** | 20 |
   | `teacher_anchor_weight=0.0` | **26.3** | 351 |
   | anchor off + `reward_shaping_coef=0.3` | 28.4 | 350 |

   `teacher_anchor_weight` DEFAULTS TO 0.05 (`vgc/rl/ppo.py:36`), and any value above
   zero makes `PpoVgcPlayer.decide` call `teacher_action_index` on every decision --
   which runs the full `vgc` search engine. **The teacher anchor costs 16x throughput.**
   The network forward pass and the trajectory storage are both negligible beside it,
   and potential-based shaping is free.

   Decide the anchor's weight on its training value, knowing it costs 16x, rather than
   leaving it on by default.

   Resist reading July's failure into this number. `honest-8000` recorded 2,500 s of
   wall clock, which is FASTER than 8,000 battles at 1.6 games/sec would allow, so those
   runs were not paying this cost at this rate -- bootstrap distillation is a different
   loop from PPO collection. The measurement here says the anchor is expensive going
   forward; it does not diagnose the old runs, and nothing in this repo currently can.
3. **`TPS_eval(opponent)`** -- benchmarked separately per ladder rung. `vgc-shallow-search`
   and `vgc` do real search; they can dominate evaluation wall clock no matter how fast
   the simulator is. Requirement is not a number: 500-1000 game evaluations must be
   operationally practical.

   **Measured, and the warning was right.** On the direct env: `maxpower` vs `random`
   runs at ~75 games/sec, while anything involving `vgc` runs at ~2.9-3.2 games/sec --
   i.e. roughly what the OLD websocket path managed. For search-heavy opponents the
   simulator was never the bottleneck, so the direct env buys the RL loop ~25x and buys
   a `vgc` gate almost nothing. Making n=1000 gates against `vgc` practical is a
   separate problem (parallel workers, or a cheaper rung), not something Phase 1's
   transport change solves.

#### Deterministic replay needs THREE seeds

`sim_worker.mjs` accepts a sim `seed` (damage rolls, accuracy, crits, speed ties). That
alone does not reproduce an episode, because the policy SAMPLES from `pi(a|o)`. So the
roadmap originally called for `(seed_sim, seed_policy)`. Building it turned up a third:

1. **`seed_sim`** -- `DirectBattle.start(..., seed=[...])`.
2. **`seed_policy`** -- `PpoVgcPlayer(policy_seed=...)`, a private `torch.Generator` so
   it cannot be perturbed by dropout or shuffling elsewhere in the process.
3. **Python's global `random`** -- poke-env's `RandomPlayer` (and anything else built on
   `Player.choose_random_move`) draws from the MODULE-level RNG. An unseeded opponent
   desynchronises the replay even when both of ours are pinned; observed directly, the
   trajectory matched for 6 decisions and then diverged.

`test_a_full_episode_replays_exactly_under_sim_policy_and_opponent_seeds` pins all three
and asserts each one independently changes the episode, so none is dead weight.

**Phase 1 certification checklist:**

```
[x] direct BattleStream battles terminate correctly
[x] p1/p2 observations contain no hidden-information leakage
[x] legal-action generation produces only simulator-valid actions
    (many hundreds of battles off enumerate_joint_orders across the benchmarks
     and test suite drew no |error|, which vgc.rl.env raises on)
[x] direct env matches the legacy poke-env path on a seeded scripted battle
    (identical DoubleBattle state -- the protocol-equivalence test)
[x] fixed team / leads / ordering never vary (see Phase 2)
[x] sim seed reproduces simulator randomness
[x] policy seed reproduces sampled actions
[x] combined seeds reproduce an exact complete episode
    (with a real CandidatePolicyValueNet; needs all THREE seeds -- see above)
[x] potential-shaping terminal state uses Phi = 0 (see "Reward" below)
[x] training CAN use the direct environment
    (vgc.rl.match.play_battle collects rollouts end to end; selfplay/train_ppo.py
     itself has NOT been switched over -- that is Phase 2's first task)
[x] evaluation uses the direct environment
    (offline/run_matches.py --direct; still opt-in, see "OTS" below)
[x] TPS_sim / TPS_train / TPS_eval(rung) all measured and recorded
```

#### One reason --direct is still opt-in

The equivalence test proves the two paths PARSE identically. It does not make their
results interchangeable, because the information model differs: the offline gates run
both players with Open Team Sheets ACCEPTED, so each side sees the other's full team,
while the direct env has no OTS at all (it is a room-level client command the raw
`BattleStream` never sees) and opponents are known only through in-battle reveals plus
`data/usage/set_priors.json`.

That makes the direct env a BETTER match for deployment -- CLAUDE.md records OTS firing
in ~0.2% of real ladder games -- but it means direct-env win rates are a different
measurement from the recorded gate numbers (`99/100` vs random, `225/300` vs heuristic).
Switching the default would silently redefine what the gates measure. The right move is
to re-establish reference numbers ON the direct env and treat those as the baseline
going forward, not to declare the old ones transferable.

#### The agent adapter

`vgc.rl.agents.DirectAgent` wraps a `poke_env` `Player` built with
`start_listening=False`, which turned out to construct fine and never touch the network.
So the direct path runs the SAME decision code as the websocket path -- byte for byte,
including `VgcPlayer`'s exception-safe wrappers and decision traces -- instead of a
reimplementation that could quietly disagree. A migration whose purpose is to stop
maintaining two execution paths should not start by creating a third.

`BattleMemory` was the one thing that needed wiring: `VgcPlayer` normally accumulates it
in `_handle_battle_message`, the transport hook the direct env replaces, so
`DirectAgent.observe` feeds it from the protocol lines the env already carries, in the
same observe-then-decide order as the live path.

### Phase 2 -- fixed four-Pokemon mirror, fogged observations

Both sides run the SAME fixed 4-Pokemon team.

The point is not "remove team preview" -- `PpoVgcPlayer` already never controlled it
(it overrides only `decide()`; preview is `vgc.team_preview.build_team_order`). The
point is that the heuristic preview hands the agent a DIFFERENT 4-of-6 per matchup,
which makes the agent's state distribution non-stationary. Pinning the team removes
that confound and the archetype/learner-team randomization at the same time.

**Supplying a 4-mon team is not sufficient.** Preview can still pick a different lead
pair or ordering, which reintroduces exactly the non-stationarity this phase exists to
remove. Phase 2 must pin all of it, identically on both sides:

```
Pokemon 1 = lead slot 1
Pokemon 2 = lead slot 2
Pokemon 3 = back slot 1
Pokemon 4 = back slot 2
```

Observations are fogged from day one: the agent sees its own item/moves/stats and only
REVEALED opponent information. Open Team Sheets fire in ~0.26% of real ladder games
(6 of 2,338 replays -- see `vgc.replay_parse`'s module docstring), so training on
perfect information would optimize a different problem than the one we deploy into.

`Flat Rules` uses `Picked Team Size = Auto` (4 for doubles) with max 6, so a 4-mon team
should validate. Confirm before building on it, then confirm what preview actually does
with it:

```bash
cat teams/fixed4.packed.txt | ./pokemon-showdown validate-team gen9championsvgc2026regmb
```

If it rejects 4, fall back to a 6-mon team with a hardcoded `/team 1234` on both sides.
Either way the order string is hardcoded, not chosen.

**Exit criterion:** fixed-mirror battles run end-to-end in the Phase 1 env with team,
leads, and ordering provably constant; teacher agreement is measured on FRESHLY
COLLECTED states, not training states.

### Phase 3 -- RL fundamentals against an opponent ladder

Ladder the opponent instead of aiming one binary at the strongest engine:

```
random -> maxpower -> heuristic -> vgc-myopic -> vgc-shallow-search -> vgc
```

A profile like `99 / 91 / 73 / 58 / 39 / 25` localizes exactly where strategic capability
breaks. A single "25% vs vgc" does not. `vgc` is a benchmark, not a starting opponent --
it already contains substantial search and evaluator logic. Snapshot self-play
(`vgc.rl.opponents`) runs alongside, even in the mirror phase.

**Track three quantities, not one:**

1. Win rate vs each FIXED ladder rung (diagnostic -- where capability breaks)
2. Win rate vs the historical snapshot pool (detects cycling and forgetting)
3. Elo across all of the above (the summary number)

A single win rate against one deterministic opponent measures exploitation of that
opponent, not strength. Keep the entropy bonus meaningful, do not anneal it to zero, keep
the snapshot pool on, and remember that `--eval-games`' deterministic mode takes the
argmax of a policy that is SUPPOSED to be stochastic -- it can legitimately score worse
than the sampled policy.

**Evaluation budget is non-negotiable.** CLAUDE.md already documents +/-4-6 points of
cross-session variance at n=100-300; n=60 cannot resolve anything.

| Purpose | Minimum n |
|---|---|
| Development check | 500 |
| Important comparison / gate | 1000 |

Same-session A/B only for deltas, per CLAUDE.md's gate methodology. Prefer side swapping
and paired seeds.

### Phase 4 -- opponent team diversity

Re-enable `vgc.selfplay_pool` / `tools/build_archetype_pool.py`. The code already
exists (commits `a0a270c`, `9aa97a6`, `afcee0c`) -- this phase turns it back ON, it does
not build it. Do not delete that code during Phases 1-3; gate it behind flags. The
learner team stays fixed; the objective becomes expected win rate over opponent teams.

**Exit criterion:** hold Phase 3's strength while opponent teams vary.

### Phase 5 -- own-team diversity

Re-enable learner-team randomization + within-archetype holdout. Pokemon identity
becomes part of the state rather than baked into one composition. Schema-v4 encoding is
retained (see "Encoding" above) -- this is the phase that would have paid for a vocab
migration, which is precisely why we don't do one in Phase 2.

**Exit criterion:** generalization gap < 5 points at n >= 1000.

### Phase 6 -- learned team preview

`(our 6, their 6) -> (bring 4, lead 2)`, initially a separate model from the battle
policy. `vgc.preview_predict` and `vgc.team_preview` are the incumbents to beat.

**Exit criterion:** beats the heuristic preview head-to-head at a fixed battle policy.

## Reward

**Default for Phases 2-3: sparse terminal only.** 0 per non-terminal step, +1 win,
-1 loss, 0 draw. Shaping stays available behind `--reward-shaping-coef` but is OFF by
default, so if learning fails we know it failed under the objective we actually care
about. Introducing shaping is then an explicit experimental variable with a shaped vs
unshaped comparison, not an always-on default.

With +/-1 terminal rewards, gamma = 1, no shaping, and no draws, the value head has an
exact reading: `V(o) = P(win) - P(loss)`, so `P(win) = (V(o) + 1) / 2`. Discounting or
shaping destroys that interpretation -- another reason to keep the baseline clean.

### Known bug: terminal potential breaks policy invariance

`vgc/rl/rewards.py` is theoretically correct -- potential-based shaping
(Ng, Harada & Russell 1999) adds `coef * (gamma * Phi(s') - Phi(s))`, which telescopes
over an episode to `coef * (gamma * Phi(s_T) - Phi(s_0))` and provably cannot change the
optimal policy. Note that sum is NOT zero; it is harmless because it is
policy-INDEPENDENT, which requires `Phi(s_T)` to be the same constant (conventionally 0)
for every terminal state. `src/vgc/rl/player.py:121-125` instead passed
`terminal_potential=board_potential(battle)` -- the actual potential of the finished
board. `Phi(s_T)` then varied with HOW we won -- more HP and more surviving Pokemon meant
more shaped return -- so the episode total was policy-dependent and shaping was quietly
optimizing "win cleanly" alongside "win". Measured on the regression test: at
`coef = 0.5` a blowout win collected `0.325` of shaping against `0.075` for a narrow win
from the same opening board, a `0.25` spread on a `+/-1` terminal signal.

**FIXED** (`src/vgc/rl/player.py`, this branch): the callback passes
`terminal_potential=0.0`. Guarded by
`test_battle_finished_callback_shaping_is_independent_of_how_cleanly_we_won`, which
asserts the property that matters -- two wins from the same start state get identical
shaping regardless of the final board -- rather than a specific number.

## Discount factor

Current default `gamma = 0.995` (`src/vgc/rl/ppo.py:28`). Over a 20-decision game that is
`0.995^20 ~= 0.90` -- a real, unintended preference for fast wins. The objective is
winning, not winning quickly.

Phase 2 compares `gamma = 1.0` (preferred baseline) against `0.995` empirically. Treat
this as a modeling decision, not a PPO convention inherited from Atari.

## Standing rules

- No PPO hyperparameter tuning before Phase 3.
- No decision from an n < 500 evaluation.
- Phase 4/5 code stays in the tree, flagged off, during Phases 1-3. Same for reward
  shaping, the teacher anchor, and meta-features.
- Reward shaping off by default until sparse-reward learning is characterized.
- Every phase exit gets a `runs/experiments.jsonl` entry via `offline/log_experiment.py`.

## Immediate engineering order

Do not tune PPO. Execute in this order:

```
 1. fix terminal_potential = 0.0                          [done]
 2. add the shaping-invariance regression test            [done]
 3. Python DoubleBattle direct driver on sim_worker.mjs   [done] vgc/rl/env.py
 3b. agent adapter + BattleMemory feeding                 [done] vgc/rl/agents.py
 4. move the evaluation harness onto the direct env       [done] vgc/rl/match.py,
     offline/run_matches.py --direct (opt-in; see "OTS" above)
 4b. protocol-equivalence test                            [done]
 5. policy-sampling RNG seeding                           [done]
 6. deterministic exact-episode replay test               [done]
 7. validate the fixed 4-mon team                         [done] rejected -- six
     required, so the PICK is pinned instead
 8. hardcode leads/ordering on both sides                 [done] PHASE2_PREVIEW_ORDER
 9. per-battle-id concurrent stepping in the worker       [done] ~1.3x, see above
10. benchmark TPS_sim                                     [done] 67 batched /
     ~240 aggregate at 8 workers
11. benchmark TPS_train (real policy in the loop)         [done] 26.3 games/sec
     with the teacher anchor off, 1.6 with it on
12. benchmark TPS_eval per ladder rung                    [done] see "Throughput"
13. run the Phase 1 certification checklist               [done] all items pass

--- Phase 1 complete; Phase 2 starts here ---

14. re-establish reference gate numbers on the direct env
15. point training at vgc.rl.match (fixed-mirror script; leave the
    websocket train_ppo.py paths flagged off rather than rewriting them)
16. teacher_anchor_weight = 0.0 for the first experiment (16x cost, and we
    need to know whether RL works without an imitation crutch)
17. gamma = 1.0 for the first experiment (winning, not winning quickly)
18. begin the first controlled RL experiment: one fixed-team PPO agent,
    capability ladder starting at random, reproducible learning curve

Heuristic PolicyConfig weights are frozen (2026-08-12). Do not retune Protect
or other evaluator knobs as part of this phase.
```

## First controlled RL experiment

Only after Phase 1 certification:

```
Format:        Champions Reg M-B
Team:          fixed four, both sides identical
Leads/order:   fixed and hardcoded
Observation:   fogged (.p1/.p2)
Reward:        terminal +/-1 only
Gamma:         1.0 baseline
Policy:        CandidatePolicyValueNet (schema-v4 encoding)
Algorithm:     PPO, entropy 0.01 constant
Opponents:     random -> maxpower -> heuristic   (NOT vgc-full)
```

The success criterion is not falling loss, rising training reward, or one win rate
moving. It is:

> more controlled RL experience -> measurably stronger policy,

across statistically meaningful samples, multiple opponents, historical checkpoints, and
repeated experiments. Only once that relationship is visible should the state
distribution widen.
