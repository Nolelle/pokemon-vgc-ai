# Tactical RL commissioning

This expands section 4 beyond the original one-versus-one endgames. No full-battle
learning or public promotion is implied.

## Task construction and oracle

`data/research/tactical_v1.json` records validated six-Pokémon teams, the fixed
bring-four order, initial simulator seed, and every real choice leading to each
root. Legal Healing Wish, Memento and Explosion actions produce the small states;
HP, fainted flags and statuses are never patched. Teams are validated with the
actual `gen9championsvgc2026regmc` team validator. These are deliberately controlled
commissioning situations, not a representative distribution of competitive play.

| Task | Verified winning witness |
| --- | --- |
| Setup | Three Swords Dances, then Earthquake against the fixed recovering Slowbro |
| Replacement | Submit a legal replacement, then solve the setup task |
| Target | KO Starmie before its lethal Ice Beam; then beat Clefable's healing |
| Sacrifice | Metagross uses Explosion while Garchomp uses first-use Protect |

Each witness won **128/128** real engine continuations. Independently, the native
engine's damage function enumerated all 16 damage rolls with and without critical
hits on exact serialized roots, including the real roots after Swords Dance.
The checks verify KO margins, targeting-task speed order, accuracy/evasion,
relevant status conditions and first-use Protect. Two Poison Jabs beat Clefable's
quarter-HP Life Dew even at minimum damage. No oracle output enters actor features.

The witness reaches the maximum possible outcome of +1 for these deterministic
task/opponent combinations. It is an optimal winning witness, not a claim that
every alternative has been enumerated or that the witness is uniquely optimal.
For example, a safe delay followed by the same plan may also win. The replacement
task exercises legal request handling; it does not establish a strategic preference
between otherwise equivalent active slots.

The fixed opponent policies use their own request-legal moves. Slowbro heals
while Slack Off is available, then attacks. Starmie attacks the opposing Garchomp;
Clefable uses Helping Hand with a partner and Life Dew alone, with legal fallbacks
when PP is exhausted. The learner gets ordinary fogged observations, complete
own-team metadata and public history, and may select every legal joint action.

The random actor won 57/128 setup, 58/128 replacement, 33/128 targeting and 4/128
sacrifice trials. Those numbers characterize this task distribution; they are not
the oracle or competitive strength measurements.

Raw proof and controls: `runs/rl_sections_1_8/tactical_oracles_v3.json`.

## Representability

Fresh models with seeds 77, 88 and 99 overfit the four winning root choices using
the production actor's stored inputs and legal-candidate mapping. After 200 Adam
steps, every prescribed action had probability above 99%. This establishes that
these input/action distinctions can be represented and optimized. It is supervised
debugging; those weights are discarded before RL.

## Frozen throughput pilot

Before assigning the main budget, run seed **707**, four updates of 32 complete
episodes (128 training games), learning rate 0.001, four PPO epochs, sparse terminal
rewards, gamma 1, default lambda 0.95, entropy 0.01, no teacher, no shaping or dropout.
Uniformly sample the four tasks. Evaluate 16 futures per task at initialization
and after update 4. This pilot measures execution, wall time and trajectory length;
it is not a passing three-seed learning experiment. Its weights are not reused.

The pilot completed in **23.1 seconds**, with no fallback, invalid trajectory or
nonfinite update. Development wins moved from 21/64 to 24/64; that small change
is not a strength conclusion. The measured cost supports a bounded main run.

## Frozen main comparison

- Fresh RL initializations **77, 88, 99**; no supervised weights are loaded.
- 64 updates × 32 complete episodes = **2,048 training episodes per seed**.
- Same model, inputs, optimizer and reward settings as the pilot. Uniform tasks.
- Development: 64 futures per task at 0/16/32/48/64 updates, future base 8,100,000.
- Select highest minimum task wins, earliest tie. Retain the endpoint separately.
- Evaluate **all three** selected checkpoints on 128 fresh futures per task,
  future base **9,100,000**, including any development failure.
- Target: at least 95% wins on every task for every training seed. Show per-task
  counts and uncertainty. No aggregate override and no extra training on the
  confirmation outcomes. A failure triggers diagnosis rather than automatic scaling.

The roots remain fixed. Fresh future randomness does not constitute held-out
initial-state, team or hidden-history generalization; those belong to later stages.

## Results

| Seed | Selected update | Setup | Replacement | Target | Sacrifice |
| --- | ---: | ---: | ---: | ---: | ---: |
| 77 | 48 | 128/128 | 128/128 | 126/128 | 126/128 |
| 88 | 48 | 128/128 | 128/128 | 128/128 | 128/128 |
| 99 | 16 | 128/128 | 128/128 | 128/128 | 128/128 |

All three selected policies met the predeclared fresh-future point target on
every task. A 95% Wilson interval is approximately 94.5–99.6% for 126/128 and
97.1–100% for 128/128. These are per-task intervals, not simultaneous guarantees
and not a claim that the true success probability is proven above 95% in every case.

The runs took 253.3, 251.6 and 246.4 seconds respectively while running concurrently.
No oracle label or supervised weight was used by RL. Raw reports and both selected
and endpoint checkpoints are under `runs/rl_sections_1_8/tactical_seed{77,88,99}/`;
fresh outcomes are `tactical_confirm_{77,88,99}.json`.

The endpoint is not uniformly reliable: seed 77 finished at 58/64 on replacement,
and its update-32 minimum had fallen to 39/64. The selected checkpoint rule was
fixed before these runs; fresh confirmation validates that selection procedure
for this commissioning suite. It does not solve the observed optimization drift.
The earlier eight-task retention failures remain failures in the follow-up report.

**Decision:** the four-family, oracle-backed section-4 commissioning milestone
passes at the stated scope. A bounded section-5 full-battle pilot is justified,
with tactical retention explicitly checked. Broad task generalization, hidden-set
adaptation and competitive strength are not established.
