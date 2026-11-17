# A first-principles research plan for the VGC learner

Date: 2026-10-28. Starting checkout: `d853c64`.

**Goal:** demonstrate a reliable agent that reaches 1700+ Elo on the public Champions Reg M-C ladder. The proposed research direction is an RL policy that learns from complete match outcomes, using accurate player-visible observations and the official simulator. Search is a measured baseline and a possible later addition.

This is a proposed project protocol informed by published RL research practice, not an internal OpenAI procedure. Creating this plan does not execute experiments, certify today's environment, change playing defaults, or approve a checkpoint. It defines a new hypothesis; the old failed 100k recipe remains closed.

**New hypothesis:** after validating the observation/action/reward path and correcting PPO probability consistency, an outcome-trained policy can improve without ongoing pressure to imitate our search teacher. We will test that on a small problem before expanding to full VGC.

The ordering matters: **correct experience → correct updates → demonstrated learning → generalization → scale → deployment.** Every stage below identifies a question, minimal experiment, evidence needed to advance, and what to do if it fails. Engineering checks establish correctness; learning checks establish improvement; neither alone establishes ladder strength.

## Starting evidence and its limits

| Finding | Consequence for this plan |
| --- | --- |
| The August 100k runs used real Showdown battles but preceded major observation and calculator repairs | They measure that historical setup, not the repaired learner |
| All three saved 100k checkpoints record gamma 1.0 and reward shaping 0.0 | Do not blame those runs on discounting or the earlier shaping bug |
| They also record teacher-anchor weight 0.02 and head dropout 0.1 | They were outcome-trained policies with continuing imitation pressure |
| Historical collection used evaluation mode; PPO updates used training mode | Dropout changed action probabilities before any parameter update; fix and test this |
| A frozen historical-model diagnostic produced about 15% clipped probability ratios on synthetic inputs; consistent dropout-off produced zero | Confirms a numerical confound, not its causal contribution to lost games |
| The public-mirror no-op defect began after the documented August scaling run | Do not use that later defect to explain the earlier run |
| Later students included imitation-only training and suspect teacher labels | Copying failures are not clean tests of learning from outcomes |
| October repairs materially improved state parsing, reconstruction, and calculators | Reuse that work; verify the actual RL path instead of rebuilding everything |

Historical run source identities are incomplete. Distinguish confirmed checkpoint settings and code behavior from unmeasured historical effects. The current parser and mirror still have documented limitations; a previous PASS is not a promise of universal correctness.

## 1. Define exactly what is being learned

**Question:** What experience, objective, and deployment behavior constitute success?

Write a small experiment specification before training:

- Format: `vgc.config.FORMAT_ID`; Champions data and Showdown revision pinned. All species, sets, moves, and mechanics come from the project exports. Validate legal teams with the pinned engine.
- Environment: two-player, simultaneous-choice, partially observed game. The engine knows the real hidden sets. The actor sees its own exact information, public history, legal requests, and explicitly marked estimates of unknown facts.
- Action: one complete legal doubles order, including targets, switches and Mega decisions. Preview becomes a separate learned decision later. Candidate lists initially include all request-legal joint orders, with no heuristic strategic pruning.
- Reward: zero during the battle; +1 win, -1 loss, 0 genuine draw; gamma 1.0. This maximizes expected match score, equivalent to wins plus half draws, not strictly win probability when draws occur. Report wins/losses/draws separately.
- First baseline: corrected PPO, fresh compatible initialization, teacher anchor off, reward shaping off, dropout off in the policy, frozen opponents, and fixed preview. Start with existing structured inputs; validate their sufficiency before adding architecture complexity.
- Runtime failures, abandoned games, safety fallbacks and artificial time limits are recorded separately. They must never silently become wins, draws, or valid on-policy actions.
- Research budget: measure actual collection and update throughput first; freeze a compute cap and evaluation schedule before each comparison.

Freeze the current fast-search and exact-judge configurations as separate references. Record whether each is enabled; do not describe every current bot as exact search. Preserve existing artifacts as historical controls.

**Advance when:** a versioned specification and run-identity format exist, and a reader can tell what the policy is optimizing and what it can observe. This stage does not require a large new framework.

**If unclear:** resolve the contract before interpreting a learning curve. Search authority remains the live release rule until a later explicit policy-release review; it is not imposed on the offline experimental actor.

## 2. Verify the laboratory the learner actually uses

**Question:** Does each recorded observation/action/outcome correspond to the real battle and the player's information at that moment?

Use the existing direct Showdown environment. Standard RL experience should come from continuing actual games, without rebuilding a guessed search world for every move. A public-mirror audit is additionally required for experiments that use mirror-generated teachers, search, or reconstructed starts.

Reuse the readiness gates, parsing audit, legal-choice checks, and real battle fixtures. Inspect the encoded inputs consumed by the RL player as well as the intermediate board object. Verify:

1. Own stats, HP, sets and brought Pokémon are correct; public foe HP, reveals, boosts, forms, conditions and observed timers match the player's protocol. Unknown opponent facts remain unknown.
2. Training and live adapters produce the same encoded observations and action meanings from the same player-visible history, from both seats.
3. Changing unrevealed opponent data while holding visible history and request fixed leaves actor inputs and action probabilities unchanged. The future game may differ legitimately.
4. Chosen actions map to the intended executable joint order. Forced switches, empty slots, move locks, fainted reserves, Mega flags and target slots are covered.
5. Expected hidden-information rejections follow the server protocol without manufacturing a normal completed turn or wrong reward. Unexpected substitutions invalidate the affected training trajectory and are counted; repeated failure blocks collection.
6. Every terminal outcome is attached once to the correct player's episode. Reset clears the prior game's memory and recurrent state. Parallel games cannot share trajectories or hidden state.
7. Identical simulator, actor, initialization and opponent seeds reproduce a short run on the recorded hardware/settings; save/reload reproduces policy outputs. Separate RNG streams and seed minibatch shuffling too.

**Advance when:** the actual RL path passes the relevant readiness checks, representative complete games pass the differential audit, and no unexplained mismatch remains in the exercised training scenarios. Add only focused regressions for defects that can invalidate learning or play. Maintain explicit coverage and narrowly justified exceptions.

A rare out-of-scope mechanic need not block a small experiment that provably cannot encounter it. Expand audit coverage before expanding that experiment. An unexplained mismatch in a reachable mechanic does block it. Do not claim complete mechanics coverage from a small smoke.

**If it fails:** repair the observation, action, reward or reset defect. Do not compensate by changing rewards, increasing the model, or collecting more bad experience.

## 3. Prove the learning update is internally consistent

**Question:** Is PPO updating the policy we actually used to collect the actions?

Use a small batch of real stored decisions plus tiny analytically solvable examples. Before any optimizer step:

- Recompute selected-action log probabilities with the same weights, observations, candidate identities, masks, preprocessing and model mode. Require agreement to a declared numerical tolerance (initial CPU float32 target: absolute log-probability error at most 1e-5; investigate any adjustment).
- Probability ratios should be approximately one; exact categorical KL should be approximately zero. The same checks must hold across supported batch sizes and padding. This directly catches the discovered dropout mismatch.
- Disabling dropout during PPO does not mean disabling gradients. Do not inadvertently put the update inside `no_grad` or inference mode.
- Candidate-list permutation must only permute the corresponding probabilities; padded actions must have zero probability and no gradient contribution. Preserve target/slot meanings.
- Compute returns and advantages by hand for short win/loss/draw episodes, with terminal bootstrap zero. Check no leakage across episodes or player seats. On a one-step terminal example, the value target must equal the terminal reward.
- A zero-learning-rate update must not change policy outputs through mutable preprocessing or hidden state. A small controlled update on a known positive-advantage action should increase its probability when other objectives are disabled.
- Collect with one frozen policy version per synchronous batch. No silent use of stale asynchronously collected actions. Record the sampling distribution, including any temperature or filtering; the first baseline uses neither extra transformation.
- Complete episodes initially. If truncation is later introduced, distinguish genuine terminal states from time limits and define bootstrap semantics explicitly.

Add a tiny masked bandit or short chain with known action values using the same distribution/loss/update code. Its purpose is optimizer diagnosis, not a Pokémon benchmark. Compare numerical terms against a small independent calculation, not a duplicate of the production implementation.

**Advance when:** unchanged-policy checks pass on real batches, analytic returns agree, and the toy learner consistently improves across several seeds. Track policy/value loss, entropy, KL, clipping fraction and gradient norms; finite loss alone is insufficient.

**If it fails:** fix the update or its inputs. Do not interpret the failure as insufficient Pokémon data. This is where the dropout correction belongs, before any new large run.

## 4. Demonstrate learning on tiny, solvable Pokémon tasks

**Question:** Can the model represent a useful decision and acquire it from rewards?

Build a small set of real-Showdown endgames that finish within a few decisions. Start with known-information cases, then add controlled hidden information. Use validated team sets and real protocol prefixes to reach states; avoid relying on an unverified public reconstruction to create the test.

Include a few exact, checkable families: target selection that secures a win; a legal replacement switch; a short setup-then-payoff sequence; and a sacrifice that wins the remaining endgame. Only use a family if the proposed advantage is established independently. Do not create strategic labels from an LLM or the existing hand-scored search.

For very small tasks, enumerate the complete short game with the engine against a fixed, declared opponent policy. Account for randomness or use deterministic cases first. This gives an oracle for that task and opponent, not an oracle for general VGC. In hidden cases the reference must obey the same information boundary.

Use two distinct probes:

- **Representability:** deliberately overfit a tiny set of oracle-labeled examples. If the model cannot express these distinctions, investigate encoding, target association and optimization. This is supervised debugging, not evidence of RL success.
- **Learning:** reset to fresh weights and train from terminal outcomes, without feeding those labels to the actor. Evaluate on unseen initial conditions or random seeds from the same task family.

**Advance when:** three independently initialized runs learn the deterministic tasks and approach the known task ceiling on the prescribed stochastic cases. Proposed commissioning targets are at least 95% success on deterministically winnable tasks and within five match-score percentage points of the task oracle on stochastic cases, with uncertainty shown. Freeze attainable per-family targets before fitting. These are project debugging targets, not scientific constants or promotion gates.

**If it fails:** supervised failure points to representation/update problems; supervised success with RL failure points to exploration, credit assignment or training targets. A task lacking a reliable oracle cannot certify this stage; simplify it.

## 5. Learn complete battles in a controlled setting

**Question:** Can reliable local learning extend to a full battle with delayed rewards?

Use one legal six-Pokémon roster and pin the brought four, lead slots and bench order. The learner controls all in-battle decisions. Begin against one fixed opponent policy; repeat against a second policy as a separate experiment before introducing a mixture. Keep opponent settings unchanged during each run.

Start with random/max-power behavior to find execution and credit-assignment failures cheaply, then test a competent fixed baseline. Full games use fogged observations from the outset. A fixed opponent team can be memorized, so success here says nothing about inference over unknown sets.

Evaluate the untrained model at step zero, then frozen checkpoints on a separate, fixed development schedule with new game seeds and both seats. Compare final performance and the learning curve with a frozen, no-update copy of the initialization. Record the deployed action-selection mode; sampled policy is the primary initial evaluation because that is what was trained. Greedy results are a separate diagnostic.

**Advance when:** the learner reproducibly improves full-match score over its own initialization on development games across at least three training seeds, with uncertainty and all runs shown, and retains the earlier task competencies. It need not beat full search to pass this learning milestone. If the initialization is already at ceiling, choose a harder opponent instead of demanding an impossible gain.

**If it fails:** inspect representative trajectories, exploration, value targets, gradient flow and optimization diagnostics. A lower training loss or better teacher agreement is not a substitute for higher match score.

## 6. Show the agent can use history and handle hidden information

**Question:** Does the policy adapt to what has been revealed, rather than memorize one opponent set?

Hold our team and the opponent policy fixed. Vary opponent legal sets in a controlled way while keeping species previews similar. Test paired histories where the current public board looks similar but a previously revealed move, item or speed relation changes the correct choice.

The actor receives public observations and its own known team; simulator truth is reserved for diagnostics. Use real prefixes to supply the history. No future reveal may enter an earlier observation.

First determine whether the current structured memory captures the needed facts. If it does, do not add recurrence solely because the game is partially observed. If it does not, add the smallest useful history representation and compare it with the existing one under matched budgets. A recurrent model requires episode resets, sequence batches, hidden-state handling and probability-recomputation checks; randomly shuffled isolated turns are not sufficient.

Explicit belief probabilities are optional. If provided, evaluate their calibration on data generated with known hidden sets, and preserve known-versus-guessed flags. Improved prediction accuracy is only a diagnostic; decisions must improve too.

**Advance when:** the policy responds appropriately to informative reveals in the controlled tasks, beats an otherwise matched history-blind control where history is necessary, and remains invariant to unrevealed private changes. Freeze the paired test design before comparison.

**If it fails:** determine whether relevant history is missing, ignored or misinterpreted. Do not increase hidden-state samples or network size without that diagnosis.

## 7. Expand teams and opponents one axis at a time

**Question:** Does improvement survive changes in matchup and opposing behavior?

Expand in this order: fixed own team against varied opponent teams; then varied opponent policies; then multiple own teams. Keep preview fixed by an explicit schedule initially. Team variation and policy variation are separate axes and both need evaluation.

Partition both sides' team identities before generating training games. Group near-identical teams by family/core as well as packed content. Maintain development pools for repeated decisions and a fresh final test. Previously reused holdouts are development evidence. Pin usage priors and audit warm-start exposure before claiming an untouched test; species overlap itself is expected.

Use a frozen opponent panel containing different behaviors: random/max-power controls, existing heuristic/search baselines, and saved policies. Report each matchup instead of letting easy opponents dominate an average. Stronger external opponents are useful if available, but are not a dependency for starting this stage.

Only then introduce self-play as one training component. Keep historical snapshots and frozen external-to-the-learner baselines. A current policy's self-play win rate is not a progress metric: two equally strong or equally weak copies can both average 50%. Evaluate each new policy against the fixed panel and older snapshots; look for cycles where A beats B, B beats C, and C beats A.

**Advance when:** learning gains persist across training seeds on development teams and multiple opponent behaviors, without a material collapse in a predeclared important subgroup. Use team-aware uncertainty and multiplicity correction for subgroup claims. If results are inconclusive, estimate whether more teams or more games can resolve them before spending compute.

**If it fails:** a training-only gain suggests overfitting; success against one policy suggests exploitation; rotating wins against snapshots suggest cycling. Address the observed failure rather than blindly enlarging the population.

## 8. Learn preview as part of the whole agent

**Question:** Can selecting the four and their leads improve results with the battle policy we actually have?

Keep the in-battle policy frozen first. Train a preview policy using species-only opponent information plus allowed priors and our complete team. It receives the eventual battle outcome, so its choices are optimized for this battle policy rather than a human team-ranking proxy.

Compare learned preview with fixed and current heuristic preview using the same in-battle policy, same team matchups, and matched evaluation schedules. Respect the actual ordering semantics of preview: do not collapse lead or bench permutations unless equivalence is established for the relevant mechanics and action mapping.

Once preview helps, consider alternating preview and battle-policy training, retaining the frozen-control comparison. Do not initially update both and then guess which caused the change.

**Advance when:** preview improves development match score under a frozen battle policy across multiple teams/seeds and inference remains fog-safe. Otherwise keep the established preview baseline while continuing battle-policy research; this stage does not require a learned preview to deploy a strong learned battler.

**If it fails:** determine whether the battle policy cannot exploit the selected plan, the preview input is weak, or the training signal is too noisy. Do not use historical human bring choices as unquestioned optimal labels.

## 9. Decide what deserves more compute

**Question:** Which change or resource increase has evidence that it will help?

Increase complexity only in response to a demonstrated bottleneck. Examples:

| Evidence | Next controlled comparison |
| --- | --- |
| Correct tiny-task learner, slow full-game learning | Fresh start versus newly verified imitation warm start; keep RL teacher anchor off initially |
| Warm start helps early but improvement stalls | Anchored versus unanchored learning from the same starting checkpoint |
| Training improves, new teams do not | Broader training situations and regularization before more updates on the same games |
| Missing history causes controlled-task failures | Minimal history representation versus existing features |
| Poorly represented ownership/target relationships | Structured relational encoder versus existing encoder at comparable size/budget |
| Values and actions fluctuate under otherwise correct updates | Investigate return variance and optimization stability; compare one bounded change |
| Stable, improving learning curves at the current cap | A larger compute budget with the same recipe and evaluation schedule |

Do not run the full cross-product of all these choices. Choose one explanatory hypothesis, hold the rest fixed, and include a control. PPO is the first debugging baseline because it already exists; its failure under a verified setup would justify a specific algorithm comparison, not prove all RL impossible.

Record both environment decisions/games and wall-clock cost. A sample-efficient method may cost more per sample. Training throughput must include encoding, opponent decisions, updates and evaluation, not just simulator speed.

**Advance when:** a frozen recipe has reproducible development gains and an explicit compute cap justified by its measured curve. More games, a bigger network, a more complex algorithm, and a different opponent schedule are separate experiments.

**If it fails:** keep the simpler reference and write down what was falsified. “Inconclusive” is not “zero effect”; it is also not permission to scale indefinitely.

## 10. Run an independent final comparison

**Question:** Does the frozen learned agent improve the actual deployment objective beyond the incumbent?

Choose the checkpoint and settings using development results only. Bind the final reports to exact model bytes, relevant source, Showdown/data revisions, observation/action schemas, priors, training exposure and policy settings. Run on fresh opponents/teams with no relevant training overlap, including a harder family-shift component.

Compare the learned policy directly with both frozen search references and the diverse panel. A pure policy must be allowed to choose from its legal action set. It does not need to reproduce the teacher or retain its top action in a shortlist. The existing 98% guided-recall requirement applies to that guided-search design, not to a pure-policy strength claim.

A later hybrid comparison uses the same learned checkpoint with and without search. Report equal wall-clock decision budgets and an equal-search-work comparison where applicable. Search must demonstrate additional strength or a useful strength/latency tradeoff. Any hybrid path must also pass mirror fidelity and public-information checks.

Preserve existing release safeguards. The current public workflow is primarily hybrid-oriented: a pure-policy release needs an explicit policy-specific approval/report path before deployment, not a bypass disguised as hybrid mode. Define the final minimum useful effect, no-collapse limits and sample size before opening the final set; retain existing applicable strength requirements unless explicitly revised.

**Advance when:** the candidate clears the frozen strength criterion with team-aware uncertainty, has acceptable failures and latency, and passes the appropriate release checks. Report all final runs, not just the best seed. If the test informs another revision, it becomes development data and a new final set is needed.

**If it fails:** leave the checkpoint experimental. Diagnose the failed population or runtime condition. A rejection protects the ladder; it does not close every possible RL design.

## 11. Validate live execution and public ladder strength

**Question:** Does the approved candidate work under real requests and opponents, and does it reach the target?

Replay recorded player-visible histories through the live actor and verify action-distribution parity. Run local websocket battles with full logging, both seats, replacement switches, and the actual clock/fallback behavior. Measure median and tail decision latency, not only inference time. Confirm the exact saved checkpoint reloads and reproduces outputs.

Then use controlled public ladder blocks with fixed team/model/settings and no mid-block tuning. Record the protocol, replays, model identity, opponent ratings, results, failures and start/end/peak rating. Keep real timer losses and failures in the deployment record; do not erase them from strength claims. Do not train on final-evaluation games while claiming that block was untouched.

Review errors immediately and strength at predefined block endpoints. Distinguish a single peak at 1700 from repeated performance at that level. Report both honestly; offline wins cannot certify either. If the regulation changes before completion, document the change, recheck compatibility, and revise the deployment claim instead of carrying M-C approval into a different game.

**Advance when:** public results establish the declared target and the operating behavior remains reliable. Preserve the exact release and a working rollback reference.

**If it fails:** distinguish deployment defects, team-policy mismatch, opponent distribution shift and weak decisions before returning to the relevant earlier stage.

## Measurement rules used throughout

**Different uncertainties need different treatment.** Hundreds of games from one trained model do not measure variability between training runs. Show every seed's curve and a pooled result that preserves seed and matchup structure. Start confirmations with at least three training seeds; add more when seed variation makes the answer unstable. Three seeds are a minimum practical starting point, not a guarantee of reliable inference.

For matchup tests, pair team schedules and seats and use common random seeds where meaningful; different action paths can consume randomness differently, so paired seeds do not make games identical. Cluster by the independent sampled unit (team or team-pair as appropriate), and account for shared teams when both sides vary. Keep fixed-team results clearly labeled. Check statistical power from development variance before a large gate; more games on the same teams cannot eliminate between-team uncertainty.

When the harness changes, run an A/A comparison. Its interval should contain the null and be narrow enough to rule out a predeclared practically important bias. Merely obtaining an extremely wide interval containing 50% is a weak smoke check. Use deterministic seat/protocol checks alongside it.

Keep debugging, model selection, and final inference separate. Repeated development checks are for diagnosis; a successful checkpoint selected after repeated checks needs fresh confirmation. Freeze final endpoints or use a properly specified sequential design. Do not stop at the first significant win.

Every run records: hypothesis; one changed variable; source/data/model identities; all relevant RNG seeds; teams/opponents/splits; initial checkpoint; reward and discount; teacher/shaping/dropout settings; action sampling; game and decision counts; failures; compute; evaluation endpoints; and a conclusion of supported, rejected, or unresolved. Save sufficient checkpoint/optimizer/RNG state for an honest resume; a resume under changed settings is a new experiment.

Use focused tests for the critical contracts above and short complete runs for the rest. Reuse existing infrastructure. The plan is not a mandate to write a test for every helper or finish every rare mechanic before a bounded experiment.

## The first executable milestone

The first deliverable is **a verified learner that solves a small real Pokémon task from outcomes**, with reproducible evidence. It is not a large self-play job.

1. Record Stage 1's spec and frozen reference identities.
2. Run the affected readiness/observation checks through the real RL adapter; classify any reachable failures.
3. Fix and regress PPO collection/update probability consistency, including the dropout case on real stored decisions.
4. Check analytic returns, candidate masks, batch invariance and save/reload; run the tiny optimizer task.
5. Learn a few verified short Pokémon endgames from fresh weights, across three seeds, including a delayed-payoff case.
6. Produce one short result report: trustworthy parts, remaining defects, actual throughput, whether full-game training is justified, and its proposed cap.

Target short debug loops where possible. Run a throughput pilot before assigning hours or training-game counts. A provisional full-game development schedule is evaluation at initialization and after 1k, 4k and 16k games per seed, adjusted and frozen from the pilot before comparison. These are diagnostic checkpoints, not automatic scale milestones or statistical guarantees. A run with broken inputs or updates stops immediately; a valid but flat run ends at its predeclared cap and receives a diagnosis.

## Existing implementation map

| Need | Existing starting point |
| --- | --- |
| Rules/state/action readiness | `offline/check_mechanics_readiness.py`, `offline/check_battle_state_readiness.py`, `offline/check_action_readiness.py` |
| Observation comparison | `offline/audit_battle_parsing.py`, `src/vgc/poke_env_compat.py`, `src/vgc/mechanics_state.py` |
| Optional search reconstruction | `offline/audit_mirror_fidelity.py`, `src/vgc/rl/live_mirror.py`, `src/vgc/rl/public_search.py` |
| Actual games and player adapters | `src/vgc/rl/env.py`, `src/vgc/rl/agents.py`, `src/vgc/rl/match.py` |
| Policy, encoding and PPO | `src/vgc/rl/player.py`, `src/vgc/rl/model.py`, `src/vgc/rl/encoding.py`, `src/vgc/rl/ppo.py` |
| Fixed-team experiment | `selfplay/train_fixed_mirror.py`, `tests/test_train_fixed_mirror.py` |
| Later diverse training | `selfplay/train_full_pipeline.py`, `src/vgc/rl/opponents.py` |
| Evaluation and release evidence | `src/vgc/evaluation.py`, `src/vgc/model_release.py`, `data/models/registry.json` |
| Deployment | `ladder/run_ladder.py`, `src/vgc/agent.py` |

These are starting points, not preapproved commands. Historical scripts may contain old defaults or hybrid-specific assumptions. Stage 1–3 verification precedes their use for new conclusions. Behavior settings remain centralized in `PolicyConfig`; optimizer settings belong in the explicit training configuration.

## Sources and provenance

The experiment ordering, stage targets and architecture choices above are proposals for this project. Public sources support the general research method; they do not certify a particular budget or algorithm for VGC.

- [OpenAI Spinning Up: Spinning Up as a Deep RL Researcher](https://spinningup.openai.com/en/latest/spinningup/spinningup.html): small debugging experiments, instrumentation, fair baselines, multiple seeds and separate confirmation.
- [OpenAI Spinning Up: Introduction to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html): policy-gradient objectives and learning from trajectory returns.
- [Engstrom et al., Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729): implementation choices can materially change the behavior attributed to PPO.
- [Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264): uncertainty from limited training runs and evaluation aggregates.
- Project evidence: `CLAUDE.md`; `docs/project_audit_2026-09-07.md`; `docs/replay_state_refit_2026-10-26.md`; saved 100k checkpoint metadata; historical code at `5be3ff2`; and the read-only diagnostic performed in the 2026-10-28 discussion. The diagnostic's approximately 15% figure was from synthetic inputs and is not a measured battle effect.

## Execution status

Stages 1–5 have been exercised: see [initial commissioning](rl_commissioning_2026-10-28.md), [execution results](rl_sections_1_8_progress_2026-10-28.md), [tactical commissioning](rl_tactical_commissioning_2026-10-28.md), and [full-battle commissioning](rl_full_battle_commissioning_2026-10-28.md). Required websocket tests and all clean-snapshot gates pass. Three selected policies meet the four-family tactical target; bounded full-game training meets its point criteria while retaining those skills. Two full-game gains are clear and the third is statistically uncertain. Optimization drift and earlier failed interventions remain documented. Section 6 is next; sections 6–11 remain unexecuted. This plan authorizes no checkpoint promotion by itself.
