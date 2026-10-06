# pokemon-vgc-ai

Pokemon Showdown VGC bot for `gen9championsvgc2026regmc` -- "[Gen 9 Champions] VGC 2026
Reg M-C": doubles, bring-6-pick-4, level 50, Megas allowed. The format offers mutual-
consent Open Team Sheets, but this bot rejects them and assumes no opponent sheet on the
best-of-one ladder. Backed by the local Showdown checkout's `champions` mod (see
`config/formats.ts` in that repo), not vanilla gen9.

## Regulation M-C (2026-09-09)

Live format id is `vgc.config.FORMAT_ID` (`gen9championsvgc2026regmc`). Champions data
was re-exported on 2026-09-09 from Showdown `efe494857`: **390** legal species/formes,
**166** items, **509** moves, **222** abilities. Readiness gates passed after that
export. M-C ranked is 2026-09-09 through 2026-12-02.

M-C replay corpus (2026-10-01): the public listing held ~65k M-C replays (~3k/day).
`data/replays/gen9championsvgc2026regmc/` has 20,215: every public game rated >=1200
since launch, plus the 468 launch-day files (some 1100-1199). Downloaded with
`--min-rating 1200 --max-pages 1400 --exclude-player laplacestheorems` (our own account).
Ladder bots are NOT excluded by name: the per-game rating filter is the quality gate.
Strong bots stay (`Scorecard-Pokemon`, `SC-SME`, `SC-Control`, median game rating
1264-1301); the weak one (`pcrlbot12d159c39a`, median 1079) is mostly filtered out.
A full backfill needs ~1300 listing pages; the default `--max-pages 100` reaches back only
~1.5 days. Parsed to `data/bc/decisions_regmc.jsonl` (389,762 decision records, all
`format_id`-stamped). Keep the M-B tree (`data/replays/gen9championsvgc2026regmb/`, 2939
replays) as history. `data/usage/set_priors.json` is M-C (`corpus_size: 20215`, built at
`--min-rating 1200` from 19,818 games). Real-team pool: `data/selfplay/mc_sheet_pool_v2/`
(314 teams, M-C spreads), split by `tools/split_team_pool.py` into
`train_manifest.json` (226) and `holdout_manifest.json` (88), grouped by six-species set
so near-copies never straddle the split. The M-C priors A/B numbers below were measured on
the earlier 14,445-replay notes and the M-B-spread `mc_sheet_pool`. `data/usage/spreads.json` is M-C since 2026-10-01: built
from Smogon's 2026-09 chaos stats (`gen9championsvgc2026regmc-1760.json`, 1.63M battles);
`spreads_regmb.json` is the M-B control (`PolicyConfig.usage_spreads_file`). Mix formats in training only with an explicit
`format_id` on every dataset. `tools/parse_replays.py` stamps `format_id` on every record
(replay's `formatid`, cross-checked against the directory name) and aborts without writing
if either is missing/disagrees or a tree spans two formats; `vgc.bc.dataset.check_format_mix`
refuses to combine any unlabelled dataset with another. Decision files parsed before
2026-09-29 are unlabelled -- re-parse them before mixing.

July–August gate numbers, the hybrid guided-search PASS, and the pure-RL 100k closeout
are **M-B-era** measurements. Re-run before treating them as current M-C strength.

Phase 1 scaffold: project skeleton, data export, poke-env baselines, eval harness.

Phase 2a added the damage/mechanics engine (`vgc.stats`, `vgc.damage`, `vgc.sets`) that
Phase 2b's evaluator builds on -- see "Damage/mechanics engine (Phase 2a)" below.

Phase 2b added the real heuristic decision layer: `vgc.evaluator.score_joint_orders`
(in-battle move/switch scoring) and `vgc.team_preview.build_team_order` (4-of-6 pick +
lead order), both wired into `VgcPlayer.decide()`/`decide_teampreview()` and registered
as the "vgc" baseline -- see "Heuristic evaluator / team preview (Phase 2b)" below.

## Champions-mod quirks (read this before writing any game-data code)

This is the single most important thing to know about this repo. The champions mod
diverges from vanilla gen9 in ways that will silently produce illegal or wrong output if
you assume vanilla data:

- **Stat Points, not EVs.** Max 32/stat, 66 total across all six stats -- not the
  vanilla 252/510 EV system. Team files in `teams/` already reflect this.
- **Only 166 legal items** (see `data/champions/items.json`, already filtered to
  `isNonstandard == null`; M-C export 2026-09-09). No Assault Vest, no Heavy-Duty Boots,
  no Eviolite. Choice Band and Choice Specs are explicitly banned -- **but Choice Scarf
  is legal**, and **Rocky Helmet is legal in M-C**. Don't assume any vanilla-VGC item
  list, and don't assume "no Choice items" as a blanket rule.
- **Custom mega roster** with mod-specific stats/abilities (`data/champions/species.json`,
  `isMega` + `requiredItem`). Always resolve megas from the exported data, never from
  memory of vanilla mega mechanics.
- **Learnsets differ from vanilla gen9** -- e.g. Chilling Water replaces Scald for at
  least some species (see `tests/test_data_export.py`). Check
  `data/champions/learnsets.json`, don't assume a move is learnable because it is in
  vanilla gen9.
- **Many real-VGC staples are `Illegal` in this mod.** Always validate a team with the
  showdown repo's `validate-team` CLI (below) before trusting it's legal.
- **ALL game data must come from `data/champions/*.json`** (regenerated by
  `tools/export_champions_data.py` from the showdown repo's `champions` mod), never from
  vanilla-gen9 training-data assumptions.

## Damage/mechanics engine (Phase 2a)

`src/vgc/stats.py`, `src/vgc/damage.py`, and `src/vgc/sets.py` implement Gen-9-mechanics
damage calculation on top of the Champions-mod data. Read `vgc/stats.py`'s and
`vgc/damage.py`'s module docstrings for the full derivation/pipeline -- summary:

- **Stat Points, not vanilla EVs** (see the quirks list above): max 32/stat, 66 total,
  **IVs fixed at 31** (`useStatPoints` requires all-31 IVs -- `sim/team-validator.ts`),
  level fixed at 50 (`Adjust Level = 50`, no `Level Clause Mod` in this format's
  ruleset). The stat formula (`data/mods/champions/scripts.ts::statModify`, confirmed
  empirically against the built sim -- see `tests/test_stats_ground_truth.py`) is
  **linear**, not the vanilla `floor(EV/4)` step function:
  `HP = base + SP + 75`, other stats = `nature(base + SP + 20)`. Every Stat Point is
  worth exactly +1 to the raw stat; this is deliberately designed so SP=32 lands on the
  same number as 252 EVs at level 50 (SP=0 and SP=32 both match the vanilla endpoints
  exactly), just linear in between instead of stepped. `vgc.stats.calculate_stats` is
  the only place this formula should live -- don't reimplement it.
- `vgc.damage.damage_range(attacker, defender, move_id, field)` is the main entry
  point; `PokemonState`/`FieldState` are the small mutable dataclasses it takes,
  `DamageResult` (with a `breakdown` trace dict) is what it returns. No crit roll and no
  Terastallization/Stellar-type mechanics in v1 (out of scope for Phase 2a -- see
  `vgc/damage.py`'s module docstring).
- `vgc.sets.opponent_state(pokemon)` turns a poke-env opponent `Pokemon` (species/item/
  ability/moves known via Open Team Sheets, Stat Points/nature NOT known) into a
  `PokemonState`, using `vgc.sets.load_usage_spreads` (reads `data/usage/spreads.json`,
  generated by `tools/build_usage_spreads.py` -- see "Heuristic evaluator / team preview
  (Phase 2b)" below; gracefully returns `{}` if that file is ever missing/regenerated
  away) or `vgc.stats.default_opponent_spread`/`default_opponent_nature` as the fallback
  for the hidden spread.
- **Ground truth**: `tools/sim_probe.mjs` drives the local Showdown checkout's built
  `dist/sim` `BattleStream` directly (no websocket server) to run scripted battles and
  report actual `|-damage|`/stat values. `tests/test_damage_ground_truth.py` and
  `tests/test_stats_ground_truth.py` (both `integration`-marked) assert our calculator
  against it -- run with `.venv/bin/python -m pytest -m integration` (needs `node`, no
  local server). `tests/sim_harness.py` has the shared driver code and documents a
  useful fact discovered while building this: `[Gen 9 Champions] (Doubles) Custom
  Game`'s ruleset doesn't validate move/item/ability legality at all, so ground-truth
  cases freely mix whatever isolates one mechanic onto a convenient species.

## Heuristic evaluator / team preview (Phase 2b)

`src/vgc/evaluator.py` and `src/vgc/team_preview.py` are the real decision layer built on
Phase 2a's damage engine. Read `vgc/evaluator.py`'s module docstring for the full design
(offense/speed/Protect/status-move/switch/mega scoring) -- summary:

- `vgc.evaluator.score_joint_orders(battle, config) -> list[ScoredOrder]` scores every
  legal joint order from `vgc.actions.enumerate_joint_orders` for the CURRENT turn (a
  myopic 1-turn evaluator -- no multi-ply search). Every weight lives on `PolicyConfig`
  (`vgc/models.py`), individually commented -- there should never be a bare numeric
  literal standing in for a strategic judgment call in `evaluator.py`/`team_preview.py`.
  Pure, unit-tested building blocks (`effective_speed`, `resolves_before`,
  `guaranteed_ko`, `likely_ko`, `mega_species_id`) live at module level for direct
  testing against hand-built `PokemonState`s (see `tests/test_evaluator.py`).
- `vgc.team_preview.build_team_order(battle, config) -> str` scores all `C(6,4)` picks x
  `C(4,2)` lead pairs (90 candidates) against the opponent's previewed 6 and returns the
  `"/team XXXX"` order string (leads first, matching poke-env's own
  `Player.random_teampreview` wire format).
- **Our own team's Stat Points/nature are read directly from poke-env**
  (`Pokemon.evs`/`Pokemon.nature`, populated from the Teambuilder team we supplied) --
  NOT estimated the way `vgc.sets.opponent_state` estimates the opponent's hidden spread.
  Only fall back to `vgc.stats.default_opponent_spread`/`default_opponent_nature` if
  poke-env didn't populate them (shouldn't happen for our own team; see
  `vgc.evaluator._our_pokemon_state`).
- `vgc.sets.normalize_item`/`normalize_status` were promoted from module-private to
  public (small, backward-compatible change -- old `_`-prefixed names still work as
  aliases) so `vgc.evaluator` can reuse the same normalization for our own side instead
  of duplicating it; this was the only `sets.py` contract change Phase 2b needed
  (`load_usage_spreads`'s schema was already exactly what `tools/build_usage_spreads.py`
  produces -- no change needed there).
- `data/usage/spreads.json` (consumed by `vgc.sets.load_usage_spreads`) is generated by
  `tools/build_usage_spreads.py` from a downloaded Smogon chaos-stats file (not
  auto-scraped). Since 2026-10-01 it is the **M-C** dump
  (`data/usage/gen9championsvgc2026regmc-1760.json`, Smogon 2026-09); the M-B build is
  `spreads_regmb.json`. Per id-normalized species, keeps the top 3 Stat Point spreads by
  usage share.
- **`VGC_TRACE=1`** (see `vgc/decision_trace.py`) makes `score_joint_orders` and
  `build_team_order` record their top-K candidates/scores (`PolicyConfig.trace_top_k`)
  and the team-preview choice's score breakdown onto the current `DecisionTrace`,
  readable via `vgc.decision_trace.get_last_trace()` after a `choose_move`/`teampreview`
  call -- the fastest way to see WHY the bot picked what it picked for a given turn.
- Known v1 scope gaps (see individual docstrings for the reasoning, not oversights):
  no multi-ply search (each turn scored independently); opponent's response move for the
  speed/threat discount is their single best KNOWN move, not a full opponent-side
  search; Parting Shot's stat-drop value is a flat proxy (its `-1 Atk/-1 SpA` isn't in
  the exported move data to compute exactly). `vgc.damage._variable_base_power` now
  computes real base power for Weather Ball (type + BP by weather), Water Spout/
  Eruption (BP by attacker HP fraction), Electro Ball/Gyro Ball (BP by a Speed ratio),
  and Grass Knot/Low Kick/Heavy Slam/Heat Crash (BP by species weightkg) -- Charizard-Y's
  Weather Ball is no longer a blind spot (confirmed: it now outranks Heat Wave in sun
  against a neutral target, both single-target and doubles-spread). Everything else with
  a `basePowerCallback` (Flail/Reversal, Return/Frustration, Punishment, Stored Power,
  Natural Gift, Present, etc.) still isn't computable from `PokemonState`/`FieldState`
  alone and still short-circuits with `breakdown["move_supported"] = False`.

## OTS reliability, final Phase 2b gates, and ladder runner

- `VgcPlayer._handle_battle_message` contains a narrow poke-env 0.15 compatibility shim
  for the Open Team Sheets accept/reject race: if rejection arrives before the team-
  preview request, the bot remembers it and resumes exactly once when the request is
  parsed. `tests/integration/test_local_battles.py::test_ots_accept_reject_race_completes`
  is the regression test.
- Offline evaluation makes both players' OTS choice explicit (`--open-team-sheets` by
  default, or `--no-open-team-sheets` for both to reject) and uses unique guest accounts
  so concurrent/stale local sessions do not collide.
- Team preview evaluates revealed Mega stones as their actual Mega forms and models
  lead-set weather plus Chlorophyll/Swift Swim/Sand Rush/Slush Rush. This fixed the
  meta1 preview's earlier failure to bring Charizard-Y; it now selects the intended sun
  mode rather than always excluding the team's main attacker.
- **M-B-era** clean gates on 2026-07-15, meta1 mirror and both players accepting OTS:
  `runs/eval/final_vgc_vs_random_gate.json` = 99/100, Wilson low 0.946 (>0.90);
  `runs/eval/final_vgc_vs_heuristic_gate.json` = 225/300, 75.0%, Wilson low 0.698
  (>0.65). Historical for the previous regulation; `runs/` is gitignored, so rerun on
  M-C before claiming these are current.
- `ladder/run_ladder.py` is the Phase 3 session runner. It saves HTML replays, per-battle
  decision traces, and append-only `runs/ladder.jsonl` outcomes; runs one public ladder
  game at a time; and recreates the client after a timeout/network failure. Credentials
  come from `VGC_SHOWDOWN_USERNAME` + `VGC_SHOWDOWN_PASSWORD` or the gitignored
  `.showdown-credentials.json`. Always run `--local-smoke` first.
- `vgc.node.find_node` selects Node 22 even when an older Node is first on PATH. Override
  discovery with `VGC_NODE` if necessary.
- **On the public ladder, OTS essentially never triggers** (opponents must explicitly
  `/acceptopenteamsheets`; ~0.2% of the 2939 **M-B** public replays contain a showteam).
  Opponent info in real games comes from in-battle reveals plus corpus-derived set
  priors (`data/usage/set_priors.json`, `vgc.sets.opponent_move_ids`). That priors file
  was rebuilt from the M-C corpus on 2026-10-01 (19,818 games rated >=1200). Offline gates run mutual-OTS-accept, so
  the priors fill is a no-op there (verified 74/100 vs 74/100 same-session) -- don't
  expect gate results to reflect priors quality.
- **Gate methodology: cross-session variance is +/-4-6 win-rate points** on the n=100-300
  heuristic gate (same code measured 74-82% across sessions on 2026-07-16). Win-rate
  DELTAS are only meaningful from a same-session A/B (two configs against the same
  freshly-started server, interleaved or back-to-back); a single cross-session run only
  supports pass/fail against the absolute Wilson-low threshold.

## Multi-team gates: cluster by team, and check your power first

`offline/evaluate_own_spread_pool.py` plays a change against MANY teams
(`data/selfplay/archetype_pool*/manifest.json`, built by `tools/build_archetype_pool.py`).
Four rules learned the hard way on 2026-08-11/12, all now enforced in code:

- **Never tune on one team and confirm on the same team.** A held-out SEED is not a
  held-out TEAM. `protect_threat_weight=0.6` read 61.0% on `teams/phase2_mirror` (n=500,
  fresh seed) and 50.7% on the 58-team pool -- a 10-point overfit. The screen that picked
  it also had a 7.7-point SE on differences between candidates that spanned 7.5 points,
  i.e. it was choosing between four indistinguishable options.
- **Pool win rates need a cluster-robust interval, not Wilson.** Games are clustered in
  teams and teams genuinely differ, which inflated the variance 1.76x on the real gate
  and made the reported `[0.476, 0.537]` really `[0.466, 0.547]`. Use
  `vgc.evaluation.clustered_interval` / `variance_components`; `wilson_interval` is only
  correct for a single fixed matchup.
- **Check the pool's power floor BEFORE running.** Between-team variance divides by the
  number of TEAMS, so a pool of K teams has an irreducible SE floor of `sqrt(tau^2 / K)`
  that more games per team cannot lower. The 58-team pool could never certify an edge
  below ~+2.8 points at any game count. The 160-team pool
  (`data/selfplay/archetype_pool_150/`, 25 variants/archetype, 0 validation failures)
  ran the same A/B at 1426/2880 = 49.5%, cluster-robust CI [0.470, 0.520], floor +1.8pts
  -- still a wash, now tight enough that a real +3pt edge would have cleared. Use this
  pool for anything that needs to resolve below ~+3 points. The gate prints both numbers.
- **Subgroup checks need a family-wise correction.** Six uncorrected per-archetype 95%
  checks trip on noise ~14% of runs. `gardevoir_maushold` was flagged at 40.7%, a policy
  change was built to chase it, and it measured 54.9% on the next seed. The guardrail is
  now a one-sided cluster-robust test per archetype, Holm-corrected.

**Run `--null-test` (A/A: both arms identical) whenever the harness changes.** It must
return 50%. This is what retired the phantom "accurate own spreads cost 10 points"
result: that 200/500 = 40.0% run predates commit `e8417bd`, before which `DirectBattle`
enriched both sides globally and `PolicyConfig.use_own_team_spreads` had no per-agent
effect in the direct env at all, so the two seats had identical self-knowledge. The
post-fix rerun of the same comparison gave 51.8%, and the A/A null test gives 537/1044 =
51.4%, CI [0.483, 0.546]. The A/A run also has team-effect SD 0.029 (consistent with
zero) against the A/B gate's 0.106 (above the 99th percentile of the null), which is how
we know the +/-10-point team-to-team spread under A/B is a real property of the policy
change rather than noise. The 160-team confirmation measured the same spread (tau 0.114)
around a 49.5% mean, so the heterogeneity is real and the overall edge is not. A 160-team
A/A (seed 20260903, n=2880) came back 1434/2880 = 49.8%, cluster-robust CI [0.481, 0.515],
team-effect SD 0.000 -- 50% inside, no seat bias.

## Counterfactual Q / search integration: closed, do not resume without a new hypothesis

Learned Q(o, a, b) was evaluated against the engineered search evaluator on a powered,
team-separated test set (768 roots, 192 teams, 424 matchups --
`data/selfplay/counterfactual_q_powered_validation_pool`, declared in
`data/meta/counterfactual_q_powered_validation_split.json`). **No variant showed a
reproducible advantage.** Nine configurations -- frozen backbone, and trainable backbone
at lr 1e-5 and 1e-4, three seeds each -- gave `search - model` differences scattered
around zero (+0.0085, -0.0046, +0.0068, -0.0098, -0.0055, +0.0042, +0.0072, +0.0049,
-0.0150); none significant, none surviving Holm. Unfreezing the shared encoder tested and
**rejected** the last live hypothesis (that features fitted for policy/value lacked what Q
needs): it widened the train-validation gap from 0.032 to 0.053-0.056 and did not
generalize better. Opponent-response weighting was closed separately -- noise-corrected
headroom +0.0011 +/- 0.0013 over the collected top-two responses, with 92.3% of roots
having exactly zero at any weight.

Consequences, in order of how much time they save:

- **Do not add learned Q, an opponent-response model, or learned-value blending to the
  playing agent** on current evidence. Engineered search stays the reasoning engine. This
  is a measured dead end, not an unexplored one -- reopening it needs a materially
  different hypothesis, not another seed or another learning rate.
- **This says nothing about the RL policy work**, which measured a large reproducible
  win and is unaffected. "Can the RL policy approach or beat engineered search?" is still
  the open benchmark; "can learned Q improve search?" is not.
- **The earlier "best run" was an artifact of an underpowered test set**, and this is the
  reusable lesson: 120 roots drawn from only 24 teams had a cluster-robust SE floor of
  0.035 against a ~0.015 effect, so it ranked seeds by noise. Cluster by team and check
  the power floor BEFORE running -- same rule as the multi-team gates above.
  `vgc.evaluation.clustered_mean` is the continuous-value counterpart of
  `clustered_interval`/`variance_components` for per-position quantities like regret;
  `offline/evaluate_q_vs_search_powered.py` is the worked example.

## Exact Showdown mechanics: the gate, and what it does not cover

Every predicted turn in a gated decision path is now executed by the official local
Showdown engine, not by Python re-implementations. `offline/check_mechanics_readiness.py`
prints the gate; it must say `verdict: PASS` or training and ladder play refuse to run
(`vgc.mechanics_gate`, fail-closed, catalogue-hash pinned). Full write-up:
`docs/champions_mechanics_catalog.md`.

- `vgc.rl.exact_search.search_joint_orders_exact` ranks orders by cloning the real battle
  and playing each candidate/response through Showdown. `vgc.rl.mechanics_oracle` does
  the cloning; `tools/sim_worker.mjs` gained `dump` and `patchPublic` for it.
- `vgc.rl.live_mirror.LiveExactMirror` rebuilds a *public* observation into a real local
  Showdown battle so live play searches exactly too. `vgc.rl.hidden_state` turns the
  privately rolled Champions sleep (`sample([2, 3, 3])`) and confusion (`random(2, 6)`)
  durations into weighted legal branches, combined by
  `vgc.rl.exact_search.combine_belief_rankings` -- never assumed.
- `vgc.rl.mechanics_encoding` feeds the model the whole `vgc.mechanics_state` snapshot as
  byte tokens: no hashing, no fixed vocabulary, no truncation, so a newly exposed field
  reaches the network automatically. Every training entry point hardcodes
  `use_mechanics_features=True`.
- **The approximate Python and private-information teachers are gone, not merely
  unused.** `vgc.rl.distill` now builds every teaching root through `LiveExactMirror`,
  using the same public-information boundary as real play, and will not mint a label
  without an exact Showdown branch. `vgc.rl.demonstrations` accepts only
  `public_mirror_exact_showdown_teacher_v2`; its v3 dataset format intentionally blocks
  older private-root files from new training.
  `tests/test_exact_mechanics_contract.py` asserts `resolve_exchange`/`search_joint_orders`
  are unreachable from the exact modules.
- **Hidden opponent spreads are a distribution, and it is not a confident one.**
  `vgc.sets.opponent_spread_hypotheses` returns the weighted Stat Point/nature beliefs
  behind `opponent_state`'s single point estimate (median confidence in the most popular
  spread across the 275-species **M-B usage** corpus is 52.4%; no species reaches
  certainty). That figure is M-B; spreads were rebuilt for M-C on 2026-10-01
  (M-C median top-build share 0.25).
  `PolicyConfig.exact_search_spread_hypotheses` makes `vgc.rl.live_mirror` build real
  Showdown roots for multiple beliefs -- verified reaching the engine as genuinely different stats
  (`tests/test_exact_search.py::test_hidden_spread_beliefs_reach_showdown_as_different_opponent_stats`
  shows three near-equally-likely Charizards at Speed 144/152/167). Part B now requires
  two spread, coherent set, and brought-four inputs at the source, then searches two
  decision-diverse representatives of their joint distribution. Audit metadata preserves
  the original branch counts and assigns the full probability mass to those
  representatives. Spreads rebuild the root (Stat Points are baked into the starting
  team); timers re-patch it, which is why `LiveExactMirror.hypotheses` is ordered
  spread-major.
- **Those beliefs are a posterior, not a fixed prior.** `vgc.opponent_belief.
  build_opponent_beliefs` reweights the corpus spreads by what the battle has shown --
  `BattleMemory.speed_observations` (who moved first, at what effective Speed, weather/
  Tailwind/Trick-Room aware) and `damage_observations` (observed % vs `damage_range`) --
  keeping a contradicted hypothesis at 5% rather than deleting it, because speed ties and
  crits make one observation noisy. `LiveExactMirror.hypotheses(battle, memory)` consumes
  it; omitting `memory` gives the pre-battle prior. One Speed observation typically moves
  Charizard from a three-way 34/34/33 split to 91/4/4. **Note the shipped ladder search
  (`vgc.search`) and `vgc.evaluator` still use `vgc.sets.opponent_state`'s point estimate
  -- `build_opponent_beliefs` reaches only the neural network's input features
  (`vgc.rl.encoding`) and now the mirror.**
- **The teacher's opponent-information leak was measured and then removed.** On
  2026-08-28, `offline/measure_opponent_information_leak.py` compared 34 real
  checkpoints across `data/selfplay/archetype_pool_150` with identical reduced search
  width on both roots. The search's top pick agreed only 28/34 = 82.4% of the time between the
  true root and the reconstruction (Wilson low 66.5%) -- roughly 1 in 6 disagreed, and the
  disagreements are substantive (Protect+switch vs attack, a Weather Ball/Hurricane
  speed-order flip), not tie-break noise. This is frequent enough that the leak is a real
  problem, not an academic one -- the old teacher advised moves the deployed agent could
  not justify from what it actually saw roughly one turn in six. As of 2026-08-30,
  `TeacherRecordingPlayer` and PPO's optional teacher anchor both use the public mirror;
  `docs/data_requirements.md` defines the saved contract and
  `offline/audit_training_data.py` enforces it.
  **The live_mirror path also costs ~9.6x the search time** at the same reduced width
  (0.10s peek vs 0.99s guess median); at full production search width a single probe
  showed 2.0s vs 28.0s. Rerouting training through `live_mirror` is now justified by the
  disagreement rate, but the ~10x per-decision cost still needs budgeting before a large
  collection.
  Results: `runs/eval/opponent_information_leak.json`. A handful of checkpoints with
  transient `patch_public_state` failures were skipped rather than counted in that old
  diagnostic. The fail-closed collector exposed three direct-offline mirror defects: a
  private simulator root containing a non-copyable thread lock, poke-env's `null` active
  slot after a faint, and a rebuilt bring-four omitting an already revealed Pokemon.
  `DirectBattle.patch_public_state`, `tools/sim_worker.mjs`, and `vgc.rl.live_mirror` now
  handle those cases. The real-Showdown forced-switch/exhausted-bench integration contract
  and a two-game/14-decision smoke collection both pass; high-volume reliability is still
  unclaimed.
- **The gate promises exact transitions, not good judgement.** The final rank still blends
  `vgc.evaluator`'s myopic heuristic score (`search_myopic_weight`, deliberately left at
  1.0 -- zeroing it changes frozen gate-tuned weights and needs a same-session A/B), the
  shortlist is a compute budget, and `_position_value` is a hand-weighted value function.
  All three are listed in `data/champions/mechanics_coverage.json` under
  `policy_approximations`; hidden opponent spreads/nature/bring are under
  `information_uncertainty`. Do not let "mechanics are exact" drift into "the search is
  optimal" -- they are different claims and the file keeps them apart.
- **The public-mirror exact search was a no-op from 33d3a0e (2026-08-28) to a30bec3
  (2026-09-01).** `handlePatchPublic` replaced Showdown's `BattleQueue` with `[]`; the
  next `go()` threw inside the stream, the worker drains swallowed it, and every `choose`
  on a `LiveExactMirror` root returned no lines and an unchanged state. `exchange_value`
  was therefore identical for every candidate and the ranking was the myopic evaluator
  plus a constant -- on 16/16 diagnostic and 4/4 production-width decisions. Nothing
  caught it because every test checked stats and structure, not that a branch MOVED.
  Treat any live_mirror result from that window (ladder hybrid sessions, the leak
  measurement's "guess" side, post-08-30 teacher labels) as myopic-evaluator output.
  `tests/test_live_mirror_branches.py` (in both gates, family
  `live_mirror_branch_execution`) now asserts branches emit protocol, diverge, and give
  non-constant exact values; `DirectBattle._apply` raises on a silent no-op step. Two
  further mirror defects fell out of the same run and are fixed in 7590c4f: a stale
  poke-env `preparing` flag after a charge-skipped Solar Beam was materialised as a real
  two-turn lock (snapshot now trusts the request, not the flag), and the non-perspective
  clone base was a blank turn-1 parser (no Trick Room/weather/HP), which both KeyError'd
  on `-fieldend` and fed opponent-response scoring a wrong board. Open: 2/140 recall-gate
  decisions still skip with `Can't switch: trapped` on a mirror root. Likely cause (not
  re-measured): Showdown's hidden-trap flow. A side told only `maybeTrapped` (e.g. vs
  Mega Gengar's Shadow Tag) may try a switch; Showdown answers `[Unavailable choice]`
  plus an updated `trapped` request. Since 2026-10-01 `DirectBattle` retries that in a
  live game (as poke-env does on the ladder), while `evaluate_exact_branches` still
  raises `InvalidChoice` rather than score an unresolved turn. `[Invalid choice]`, and a
  switch tried while already known trapped, stay fatal. Clones copy `_waiting` from the
  source in both modes. Test: `test_rl_env.py::test_hidden_trap_rejection_*`.

## Search opponent model: bench, self-sleep, no-prior species (2026-09-29)

Found from ladder game 2678505187 turn 5 (the bot put its own Incineroar to sleep while a
full-HP Rillaboom was about to KO its 1-HP Venusaur). `vgc.search` changes, each with a
`PolicyConfig` legacy control:

- `search_public_bench_filter` (ON): opponent switch-ins come from `_opp_switch_pool`,
  not raw `teampreview_opponent_team`. Preview copies never get `fainted=True`, and
  preview shows six of a bring-four, so the old list offered switches into fainted and
  unbrought Pokemon. poke-env keeps the base species name after Mega Evolution
  (`store_species=False`), so set-prior lookups survive a Mega.
- Self-sleep: the search credited a sleep move on OUR ally as a gain (+25 x accuracy,
  the same kind of sign error as Rung 3a). Fixed on main by 701a66c
  (`vgc.principles.harms_ally_target`, `ally_harmful_status_penalty`), which scores it
  as a cost. This branch's narrower `search_sleep_credit_foes_only` was dropped on merge.
- `set_prior_learnset_fallback` (**OFF**): best legal STAB attacks from the learnset for
  species with no usable prior. The old M-B `set_priors.json` left ~22% of M-C preview
  slots with no prior moves (Rillaboom, Salamence, Indeedee-F, Golisopod), and they read
  as harmless until they revealed moves. The M-C rebuild (2026-09-30) covers 99.9% of
  preview slots, so this fallback now matters only for rare species. A "best move of every type" version was
  measured and is dangerous: foes got perfect four-type coverage and the bot Protected
  ~4.7x as often (5.5% on mc_ladder_04). STAB-only was 50.6% overall but -29..+19 points
  per team. Six teams cannot settle it.

Same-session A/Bs (`runs/eval/opp_model_fixes_*`): search fixes on the 160-team pool
1424/2880 = 49.4%, cluster-robust [0.474, 0.515] (a wash; the pool has zero no-prior
species, so the fallback cannot be tested there). On the six `teams/mc_ladder_*` teams
(`data/selfplay/mc_ladder_pool/manifest.json`, gitignored), search-only was 50.9%, and
the A/A was 50.2%. These are correctness fixes, not strength claims. Testing the fallback
needs a pool with many teams that use no-prior species.

Status-utility scoring (2026-09-30): `vgc.evaluator`'s Taunt/Encore, Will-O-Wisp and
Wide Guard values read the foe's moves revealed-only, so they score ~0 early on the
ladder. `PolicyConfig.status_utility_uses_set_priors` routes them through
`opponent_move_ids` instead. **OFF**: 160-team pool A/B 2862/5760 = 49.7%, cluster-robust
[0.484, 0.509]. Only 57/160 pool teams carry those moves (no Taunt), so the pool is weak
for this; the post-hoc 57-team subgroup (51.6%) was within the noise of the 103
unaffected teams (48.6%). Result: `runs/eval/status_utility_priors_pool160.json`.

M-C set priors (2026-10-01): `PolicyConfig.set_priors_file` picks the data/usage/ file
every bot-side prior load reads (`vgc.sets.set_priors_for`). The default is the M-C
build `set_priors.json`; `set_priors_regmb.json` is the M-B legacy control. The M-C
build covers 99.9% of M-C preview slots vs 77.5% for M-B. Same-session A/Bs (new vs
old):

- 160-team archetype pool: 2794/5760 = 48.5%, cluster-robust [0.455, 0.515], tau
  0.177. That pool has no species missing from the M-B file, so it tests move
  frequencies only.
- 298 real M-C team sheets (`data/selfplay/mc_sheet_pool`, built by
  `tools/build_ladder_team_pool.py`): 2782/5364 = 51.9%, [0.490, 0.547], tau 0.229;
  A/A 48.9% [0.473, 0.504]. Post hoc by species missing from M-B notes: 0 -> 50.1%
  (40 teams), 1 -> 51.8% (142), 2+ -> 52.6% (116), all overlapping.

Neither pool shows a significant edge; per-team effects are large both ways. Kept on
M-C as a current-format data refresh, not a strength claim.

Real-team pool: `tools/build_ladder_team_pool.py` takes the latest `|showteam|` sheet
per player from the replay corpus (298 at >=1200 on 2026-09-30, all passing
`validate-team`). Sheets hide Stat Points, so spreads come from the M-B `spreads.json`
(nature-matched where possible); some sets carry an M-B spread that does not fit their
nature. Unlike the archetype pools, it includes Mega Gengar (13 teams) and Round
users, which exposed two `DirectBattle` crashes (hidden trap, above; Round chain,
`vgc.poke_env_compat`). poke-env 0.15 KeyErrors on `|move|X|Round|Y|[from] move: Round`
for an opponent that has not shown Round (33/14,445 M-C replays);
`normalize_for_poke_env` drops the tag in `DirectBattle` and `VgcPlayer`.

## Rung 2 (belief-aware shortlist): built, gated, not enabled

`vgc.belief_scoring` scores joint orders as a probability-weighted mixture over the
opponent's posterior Stat Point spreads (`joint_spread_hypotheses` +
`score_joint_orders_under_beliefs`), and `belief_ordered_candidates` re-sorts the myopic
list by that mixture before top-K selection in both `vgc.search` and
`vgc.rl.exact_search` (same objects, scores untouched, opponent-response enumeration
untouched). `PolicyConfig.shortlist_belief_hypotheses` controls it and **ships at 1**
(identity). Evidence, 2026-09-01:

- `offline/evaluate_belief_shortlist_recall.py` (derived gate: does the winner of a
  wide exact search on the public mirror survive into K=10?) -- 138 decisions / 46
  teams: point estimate 0.971, mixture 0.978, paired diff +0.007 [-0.007, +0.022],
  `verdict: PASS` (pre-registered non-inferiority, margin 0.02). Winner was myopic rank
  1 in only 82/138, so exact search does overturn the myopic pick ~40% of the time; the
  misses sit at myopic rank 18-48 under BOTH selectors.
- `offline/evaluate_own_spread_pool.py --candidate shortlist_belief_hypotheses=3`
  (160 teams, 2880 games, `vgc.search` path): 1427/2880 = 49.5%, cluster-robust
  [0.476, 0.515]; A/A null 50.8% OK. A wash, not a drop.
- Conclusion: the shortlist is not the bottleneck at K=10 (97% recall). The remaining
  recall loss is orders the myopic evaluator ranks very low, which is a search/value
  question (Rung 3), not a belief question. Do not raise the default without a new
  hypothesis; the knob exists so Rung 3 can revisit it once the judge changes.

## Rung 3a (signed effect term): the exact search's value function had a sign error

`vgc.rl.exact_search._side_position` scored volatiles and side conditions with
`len(...)`, so it could not tell a benefit from an injury. Every active effect was worth
+`exact_search_effect_weight` (12.0) whatever it did:

    our own Leech Seed  +12      our own Substitute  +12
    Stealth Rock, our side  +12  Tailwind, our side  +12

Because `_position_value` is `our_side - opponent_side`, that ran backwards in BOTH
directions at once -- the search read walking into a Leech Seed as good for us, and read
landing a Taunt on the opponent as bad for us -- at roughly 12% of a Pokemon's HP per
effect, which is larger than the gap between many candidate moves. It also paid full
weight for one-shot ability-activation markers (`aftermath`, `dancer`, `ironbarbs`,
`quickdraw`) that are not position advantages at all; those dominate poke-env's
224-member `Effect` vocabulary.

**This was invisible until 2026-09-01.** The public-mirror exact search was a no-op from
33d3a0e to a30bec3 (see the previous section), so `_position_value` deltas were a constant
and no wrong sign inside it could move a decision. Fixing the queue bug is what armed this
one. `vgc/search.py`'s Phase 2c search is NOT affected -- it names Tailwind and the other
side conditions explicitly rather than counting them.

`vgc.position_effects` fixes the sign and nothing else. `exact_search_effect_weight` keeps
its frozen value; this is not a calibration change and no new weight was added.

- **The sign is derived, not hand-listed.** A volatile or side condition applied by a
  FOE-targeting move hurts its holder; one applied by a SELF/ALLY-targeting move helps.
  `tools/export_champions_data.mjs` now exports `volatileStatus`, `selfVolatileStatus`,
  `secondaryVolatileStatuses`, `sideCondition` and `slotCondition` alongside `target` so
  the map regenerates with the data instead of going stale.
- **The rule cross-checks against something written independently.** Derived
  "harmful to the side holding it" reproduces `mechanics_state._LAYERED_SIDE_CONDITIONS`
  (spikes/toxicspikes/stealthrock/stickyweb) exactly, from move targets alone. That
  assertion is a gate test, so a bad derivation cannot land quietly.
- **Restricting to LEGAL moves is what makes the derivation complete.** Octolock,
  Telekinesis, Embargo, Nightmare, Tar Shot, Glaive Rush, Obstruct, Burning Bulwark, Silk
  Trap, Mist, Lucky Chant, Crafty Shield and Mat Block are all `isNonstandard: "Past"` in
  this mod -- a vanilla-gen9 derivation would sign a dozen volatiles that can never occur.
- **Unsignable effects score 0, not +12.** This is deliberate and is itself part of the
  fix. Two small tables cover what the rule cannot see: `_SUPPLEMENT` for effects no legal
  move applies (perish counters, `trapped`, `slowstart`, the Protosynthesis/Quark Drive
  families) and `_OVERRIDES` for the handful the rule mis-signs because they are engine
  bookkeeping (`sparklingaria` marks targets for burn-curing) or genuinely two-sided
  (`lockedmove`, `uproar`, `roost`). Every entry is +1/-1/0 -- there are no magnitudes in
  that module, so it cannot become a tuning surface.
- `PolicyConfig.exact_search_signed_effects` ships **True**. False is the exact
  pre-3a behavior, kept only as the legacy control for same-session A/Bs -- the same
  pattern as `search_respect_our_protect_odds`.
- Gate family `exact_branch_effect_polarity` (`tests/test_position_effects.py`, 32 tests)
  is in the mechanics gate. `policy_approximations.exact_branch_position_value` still
  stands: the weights remain hand-chosen and uncalibrated, which is later work. Only the
  sign is fixed.
- **An effect already on the board cannot change the ranking, and this is general.**
  `search_joint_orders_exact` scores every branch as `_position_value(after) - before`
  with the SAME `before` for every candidate, so any term that is identical across
  branches cancels out of the comparison entirely. A pre-existing Leech Seed is invisible
  to the ranking; only an effect GAINED OR LOST inside the searched turn moves it. This
  is not specific to the effect term -- it is how the whole exact value function behaves,
  and it is the reason a diagnostic that counts effects at the ROOT measures the one case
  that provably cannot matter. The first version of the script did exactly that and
  returned a meaningless 0/110.
- `offline/measure_effect_polarity_impact.py` therefore keys off
  `exchange_values_differ`: did signing the term move any SEARCHED candidate's exchange
  value at all? Root effect counts are still reported, but only as context. Rates are
  clustered by our own team file.
- **Measured impact, 2026-09-02** (`runs/eval/effect_polarity_impact.json`, 110
  decisions / 43 teams, `archetype_pool_150`, diagnostic width): signing moved a searched
  candidate's exchange value on **7/110** decisions, with a largest shift of **12.00**
  points -- exactly `exact_search_effect_weight`, which is the confirmation that the
  mechanism is live and that one effect flip costs exactly one weight. The top pick
  changed on **0/110**.
- **Do not spend a pool A/B on this.** With 0/110 decision changes a win-rate gate is a
  null by construction and would only buy a wide confidence interval around zero. 3a is a
  correctness fix held by a gate test, not a strength claim, and that is the whole of its
  claim. Two real caveats before anyone reads 0/110 as "the effect term does not matter":
  the pool is offence-heavy (only `triple_setup_balance` sets much up), and the diagnostic
  budget is far narrower than production (`search_our_candidates=4`,
  `exact_search_future_samples=1`, one spread hypothesis), so it explores far fewer
  branches in which an effect could appear or disappear. A targeted board that actually
  creates hazards/screens/Leech Seed would measure this properly; the pool cannot.

## Exact judge: seat bug, bookkeeping fixes, and a harness that can measure it (2026-10-05)

- **The public mirror searched a stale board whenever we were p2.** `LiveExactMirror`
  always seated us as mirror p1, so a p2 observation could not become the decision view and
  the search read a template parser the public patch never updated: ~30% of p2 decisions
  had the wrong active Pokemon and ~24% the wrong weather. Exact-vs-exact A/A went
  906/1440 = 62.9% for p1. `vgc.rl.live_mirror.mirror_side` now seats us in the
  observation's own seat; every caller searches that side. After the fix the same A/A gave
  p1 48.8% vs p2 48.6% (2880 games). **Treat p2-seat exact output before this fix as
  suspect**: about half of the first M-C student's teacher labels (`runs/mcv2`), p2 hybrid
  ladder decisions, and p2 graded/reviewed positions. Test: `tests/test_live_mirror_seat.py`.
- **Account names.** Mirror battles used simulator names `p1`/`p2` while branch parsers
  copy the observation's real account name, so on any named account a won branch read as
  lost (-10,000). The mirror now starts Showdown under the observation's names. The
  offline harness (names `p1`/`p2`) never hit this; ladder/hybrid and recorded ladder
  positions did.
- **Scorecard bookkeeping** (`PolicyConfig.exact_search_consistent_accounting`, ships
  True; False = legacy control): unseen opponent reserves count toward the public bring
  size so a first reveal is not a -100 swing; fainted Pokemon score 0 (poke-env keeps
  their boosts/volatiles); draws score 0; `combine_belief_rankings` averages
  `exchange_value`/`myopic_score` instead of copying the modal belief's.
  Test: `tests/test_exact_judge_accounting.py`.
- **Measuring the exact judge.** `evaluate_own_spread_pool.py` built only the fast-search
  `vgc` player, so exact-only knobs were a null there by construction. `--player vgc_exact`
  (`vgc.rl.exact_player.ExactSearchPlayer`) makes every move with the public exact search;
  `--both FIELD=VALUE` sets a shared width; the report counts per-arm fallbacks to the fast
  search and warns above 5%. Narrow width (`search_our_candidates=4`,
  `exact_search_future_samples=1`, `search_opp_candidates=4`) plays 2880 games in ~25 min
  on 10 workers (~0.3 s/decision). Results at narrow width are not production-width claims.
- **Follow-up results (2026-10-05, branch claude/exact-judge-field-horizon; narrow width,
  `--player vgc_exact`, cluster-robust CIs).**
  - KO term (`exact_search_alive_weight` 90 vs 0), 160-team pool: 52.5% [0.501, 0.550]. Ships at 90.
  - **Drop the myopic blend** (`search_myopic_weight` 0 vs 1): M-C train 226 teams 53.5%
    [0.501, 0.569]; archetype 160 pool 53.7% [0.508, 0.567]; M-C holdout 88 teams 51.6%
    [0.467, 0.566] (underpowered, floor +4.1). The myopic score double-counts damage
    (uncapped at remaining HP) on top of the exact value. Shipped as the exact-only knob
    `exact_search_myopic_weight = 0.0` (1.0 = legacy); the fast search keeps
    `search_myopic_weight`. The live `vgc.exact_judge` already ranks by pure
    `exchange_value`.
  - Field-control leaf (`vgc.field_control`, off): 50.2% (pool160), 49.3% x3 weights,
    48.8% x3 on M-C train, 50.2% x3 with myopic 0. No effect. Diagnostic
    (`offline/diagnose_setup_ranking.py`): setup moves are searched when legal (51/54) but
    lose by a median ~200 pts (myopic ~86, one-turn exchange ~74); the field term's median
    contribution is 0. Weather start turns are overwritten by poke-env every upkeep, so
    weather duration is a guess.
  - Multi-turn continuation: N=1 greedy 49.4% (pool160), N=2 greedy 47.8% (M-C train);
    with myopic weight 0 on both arms, N=1 greedy 47.4%, N=1 fast-search continuation
    49.9% [0.468, 0.530]. Four nulls: a fixed continuation policy does not help; do not
    raise N without a smarter continuation (e.g. a small search per continuation turn). Latency (narrow): N=1 p99 ~4 s, N=2 ~7 s, N=3 ~8 s myopic; production width
    N=1 p99 ~42 s (does not fit the 12 s clock cap).
- **Condition durations were wrong in every exact branch** (fixed 2026-10-05, owner's
  hunch, confirmed independently by Codex). poke-env restamps weather on every
  `[upkeep]` line, and switch-in setters (Drought, Surge abilities) were charged a turn
  Showdown never charges, so the mirror rebuilt weather wrong 56/72 and terrain 108/134
  times (Trick Room/Tailwind were right). `vgc.condition_clock` counts Showdown's
  `|upkeep|` ticks since each start line (both ingest paths, copied into mirror views);
  known extenders on the setter (always for our side) count 8 turns from the start.
  Re-verified 266/266 vs the live simulator; hidden opponent extenders still read 5 until
  outlived (Smogon M-C: Pelipper Damp Rock 7%, other setters ~0%). Re-run with correct
  timers on M-C train: field leaf x1 50.1%, x3 49.7% (still null); N=1 fast-search
  continuation **46.3% [0.431, 0.495] -- worse**. Suspect: one random sample per branch
  makes an extra simulated turn mostly dice noise; N=1 with 4 samples is the test.
- **Still open** (see the 2026-10-05 review): no KO/faint term in `_position_value`
  (violates docs/search_contract.md section 4), Trick Room/weather/terrain score 0, flat
  boost/status weights, one-turn horizon (the fast search's 2-turn forecast was worth
  +7.6 pts), myopic blend double-counts damage.

## Neural shortlist distillation (Phase 4): M-B-era guided gate, current models unapproved

The student policy that ranks legal joint orders for `vgc.rl.search_guidance` is trained
by `selfplay/train_imitation.py` (BC from the search teacher) and evaluated by
`offline/evaluate_shortlist_recall.py`. August 2026 standing below is **M-B-era** and
does not approve a current checkpoint. See `docs/learned_model_contract.md` and
`data/models/registry.json`. **No registered checkpoint loads on M-C**: the A3/B1
models have a 150-row item table vs the M-C export's 168, so they are marked
`incompatible_vocabulary` (2026-09-29), and the BC files (`bc_policy_*.pt`) differ in
species/item/ability. `vgc.model_vocabulary` is the check: RL checkpoints now save the
ordered species/item/ability/move token lists (`data_vocabulary`), and
`load_snapshot`, PPO/fixed-mirror resume, `--init-from` and release validation refuse
a file whose lists differ from `data/champions/*.json` or that records none (row counts
alone miss reordering). `load_bc_policy` disables a stale BC file; warm-start skips
stale embedding tables and says so. `tests/test_model_registry.py` requires a
`compatible` entry's `loader.vocabulary_sha256` to match today's vocabulary. Any
learned policy on M-C needs retraining first.

- **The shortlist is NOT the network's raw top-K** -- up to half its budget is
  heuristic safety slots (`vgc.rl.guided_selection`, shared by live play and offline
  replay). Screens must report BOTH flavors; a guided verdict on a metadata-covered
  subset prints INDETERMINATE, never PASS.
- **Schema v2.x**: every collected `DistillationSample` also carries per-candidate
  myopic ranks, safety-tag columns, the teacher's full search-score vector, and a
  searched-mask. Old datasets load unchanged but cannot replay guided selection.
- **M-B-era standing (2026-08-25):** retrained on 76,801 v2.x decisions. On the clean
  150-team expanded holdout (18,209 decisions): pure R@10 97.7% (LCB 0.974),
  **guided@10 LCB 0.984 = PASS**, live shadow **guided@10 99.1% / LCB 0.985**. Powered
  re-gate after `guided_upset_margin=10`: hybrid +0.4pts vs full search. Historical
  public smoke 6-4. Path:
  `ladder/run_ladder.py --policy-checkpoint <ckpt> --policy-mode hybrid`. Public ladder
  still requires a named compatible checkpoint and current release approval; a file
  existing is not approval.
- Expanded holdout hygiene: exclude byte-identical training-pool teams by packed
  content, not filename.
- **First M-C student (2026-10-02), not approved.** Teacher collection at 20308ce on
  `mc_sheet_pool_v2` (train 226 / holdout 88 teams, team-disjoint): 5 train shards (34,002
  decisions; shard 1 lost to a full disk) + holdout shard (4,529 decisions, 600 games),
  skip rate 0.3-0.6%. `runs/mcv2/train_v1/best.pt` (warm start `runs/mcv2/bc_regmc`, M-B
  recipe, best epoch 4/7). Holdout recall: pure R@10 91.8% (LCB 0.907), **guided@10 LCB
  0.920 = FAIL** (bar 0.98). The heuristic's own top-10 contains the teacher's pick 90.9%
  of the time, so the student roughly equals the shortlist it imitates: the K=10 teacher
  caps it (see the 2026-10-01 architecture review). Next: A/B search width (K=10/20/40)
  before re-collecting with a wider teacher. `runs/` is gitignored.

## Pure-RL 100k scaling closeout: do not promote or scale this recipe further

The frozen scaling study completed on 2026-08-26 (M-B). Three seeds each trained for
100,096 games from the same imitation checkpoint. Verdict:
`MIXED_OR_INSUFFICIENT_EVIDENCE`. All three exact-100k promotion gates failed vs full
search on unseen teams (42.6%, 39.5%, 36.4%). Do not promote those checkpoints or
launch a larger run with the same recipe. Search remains the decision authority.
Evidence: `runs/full_pipeline/rl_scale/scaling_verdict.json`.

## Commands

All Python invocations use `.venv/bin/python` -- there is no `python` on PATH in fresh
shells on this machine, and `node` may also need an absolute path
(`~/.nvm/versions/node/v22.22.0/bin/node`) if it isn't on PATH.

```bash
# Mechanics gate -- must PASS before any training or ladder command runs
.venv/bin/python offline/check_mechanics_readiness.py

# Battle-state gate -- must also PASS before any training or ladder command runs
.venv/bin/python offline/check_battle_state_readiness.py

# Action-generation gate (Problem C) -- must also PASS before any training or
# ladder command runs. Full write-up: `docs/action_generation_contract.md`.
.venv/bin/python offline/check_action_readiness.py

# Showdown parity -- both gates and the public ladder run this; it fetches origin/master
# and BLOCKS if the local checkout is dirty, differs from the catalog's pinned commit, or
# is missing upstream commits on watched paths (mod, sim, lib, dex tables, rulesets, tags,
# aliases, formats) that `vgc.showdown_relevance` cannot prove irrelevant to Reg M-C. Only
# other-format edits to config/formats.ts and unrelated data/aliases.ts nicknames are
# cleared (parsed with Showdown's own TypeScript via tools/parse_showdown_entries.mjs),
# and a formats edit only if the pinned BUILT Showdown can still load and serve the
# commit's whole format list (its own loader + server format-list code, with the commit's
# aliases; needs `dist/`) -- a broken neighbour format breaks ours too. Cleared commits
# print as "irrelevant_upstream_commits". Every other change blocks, and the report
# names which Reg M-C moves/abilities/etc. each blocking commit touched. When a sync is not
# needed, `tools/sync_showdown.py` reports "already in parity" and does not pull. To update,
# run tools/sync_showdown.py (below): it fast-forwards + `node build --force`es Showdown,
# re-exports data/champions and the mechanics catalog, re-pins `catalog_sha256`, commits, and runs the three gates. It
# STOPS for review if the catalog changed beyond its recorded commit/path (rerun with
# --accept-catalog-changes once reviewed). The public ladder runner calls it automatically
# (`--no-sync-showdown` to opt out). Training only checks the saved gate certificates, so
# sync before a training run too. Last done 2026-10-01 -> 1ce9b34f9 (M-C).
.venv/bin/python offline/check_showdown_parity.py
.venv/bin/python tools/sync_showdown.py

# Start the local server (from the showdown repo, port 8000, no auth)
cd /Users/edmundyu/code/projects/pokemon-showdown && node pokemon-showdown start --no-security

# Validate a team against the target format (from the showdown repo)
cat teams/dev.packed.txt | ./pokemon-showdown validate-team gen9championsvgc2026regmc

# Download rated public replays for the live format (incremental; M-C corpus is small)
.venv/bin/python tools/download_replays.py --min-rating 1100

# Regenerate data/champions/*.json from the showdown repo's champions mod
.venv/bin/python tools/export_champions_data.py

# Regenerate data/usage/spreads.json from a downloaded Smogon chaos-stats file (see
# "Heuristic evaluator / team preview (Phase 2b)" above)
.venv/bin/python tools/build_usage_spreads.py

# Run matches between two registered baselines against the local server
.venv/bin/python offline/run_matches.py --p1 random --p2 maxpower --n 50 \
    --team teams/dev.packed.txt

# Acceptance gate (Wilson-CI lower bound over a threshold)
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent random --n 100 \
    --threshold 0.55 --team teams/dev.packed.txt

# Varied-team gate (see "Multi-team gates" above). Build the pool once, then A/A the
# harness, then run the A/B. --null-test MUST come back at 50% or the A/B means nothing.
.venv/bin/python tools/build_archetype_pool.py --variants-per-archetype 25 \
    --seed 20260901 --out data/selfplay/archetype_pool_150
.venv/bin/python offline/evaluate_own_spread_pool.py --null-test \
    --manifest data/selfplay/archetype_pool_150/manifest.json \
    --output runs/eval/pool_null_test.json
.venv/bin/python offline/evaluate_own_spread_pool.py \
    --manifest data/selfplay/archetype_pool_150/manifest.json --workers 10 \
    --output runs/eval/own_spread_pool160_gate.json

# First controlled RL experiment (fixed team, fixed leads, fogged, terminal ±1,
# gamma=1.0). Heuristic weights stay frozen; this is the learning-curve run.
.venv/bin/python selfplay/train_fixed_mirror.py --opponent random \
    --iterations 20 --games-per-iteration 256 --eval-games 500 \
    --eval-every-iterations 4 --out-dir runs/ppo/fixed_mirror_vs_random
.venv/bin/python selfplay/train_fixed_mirror.py --opponent maxpower \
    --init-from runs/ppo/fixed_mirror_vs_random/latest.pt \
    --iterations 20 --games-per-iteration 256 --eval-games 500 \
    --eval-every-iterations 4 --out-dir runs/ppo/fixed_mirror_vs_maxpower

# The two Phase 2b acceptance gates (meta1 team mirror on both sides -- see "gate
# results" in the Phase 2b experiment log, runs/experiments.jsonl, for the latest run):
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent random --n 100 \
    --threshold 0.90 --team teams/meta1.packed.txt \
    --output runs/eval/vgc_vs_random_gate.json
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent heuristic --n 100 \
    --threshold 0.65 --team teams/meta1.packed.txt \
    --output runs/eval/vgc_vs_heuristic_gate.json

# See a turn's evaluator reasoning (top-K scored candidates) or a team-preview pick's
# breakdown -- set before any script that drives a real/local battle:
VGC_TRACE=1 .venv/bin/python offline/run_matches.py --p1 vgc --p2 heuristic --n 1 \
    --team teams/meta1.packed.txt

# Exercise replay/trace/outcome logging against the local server before public ladder:
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2

# Public ladder (credentials are read from env or .showdown-credentials.json):
.venv/bin/python ladder/run_ladder.py --n 1

# Rung 3a diagnostic: how often does signing the exact search's effect term change its
# top pick? Reports an overall rate AND a rate restricted to decisions that actually had
# a signed effect on the board -- the second is the honest denominator.
.venv/bin/python offline/measure_effect_polarity_impact.py --pairs 60

# Log an experiment note
.venv/bin/python offline/log_experiment.py --name "..." --summary "..."

# Tests (unit only by default; integration starts a real local server for
# tests/integration/, and separately drives the sim directly -- no server needed -- for
# tests/test_*_ground_truth.py)
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m integration

# Run tools/sim_probe.mjs directly (mainly useful for debugging a ground-truth case) --
# scenario JSON shape is documented in the script's header comment.
node tools/sim_probe.mjs /Users/edmundyu/code/projects/pokemon-showdown scenario.json
```

## CI (GitHub Actions, 2026-10-01)

- `.github/workflows/ci.yml`, on every PR and push to main: **unit** (ruff + default
  pytest) and **engine** (clones public Showdown at the catalog's pinned commit, builds
  it, rebuilds `archetype_pool_150` with the seeded command below -- byte-identical to
  local -- runs the three readiness gates `--static-only`, then `pytest -m integration`).
  The engine job fails if any integration test skips: a skip there means a test could
  not find the engine or pool, i.e. silently lost coverage.
- It does NOT check whether the pin is current, so upstream commits cannot turn
  unrelated PRs red. `.github/workflows/showdown-watch.yml` does that weekly and keeps
  one "Showdown drift" issue open/updated/closed.
- `VGC_SHOWDOWN_REPO` overrides `vgc.config.SHOWDOWN_REPO` (the only place the path
  lives; `vgc.rl.env.DEFAULT_SHOWDOWN_REPO` reads it). CI installs torch from the CPU
  index, constrained to `uv.lock` (`.github/scripts/install_python_deps.sh`, Linux only).
- Not in CI on purpose: win-rate gates, training, ladder play (credentials stay local),
  formatting checks.

## Testing and iteration preference (2026-09-07)

Prioritize getting the battle bot running and iterating on data and training. Do not
write unit tests for everything. Add tests only for critical behavior where a failure
would invalidate a run, silently corrupt its evidence, or stop the bot from playing.
Examples include public/private information boundaries, training/evaluation separation,
legal battle choices, correct model loading, and essential training/battle execution.

Prefer existing checks and small end-to-end trial runs over expanding the test suite.
For low-impact helpers, formatting, routine plumbing, and reversible changes, use a
quick manual check and debug problems when they occur. Do not add tests that merely
repeat implementation details or delay a useful experiment to chase exhaustive coverage.
Run the checks affected by a change; broaden testing when a failure or material risk
justifies it. Preserve critical readiness checks and honest strength measurements.
This preference does not require deleting existing tests or relaxing release criteria.

## Conventions

- `uv` for the environment (`uv sync --extra dev`); `.venv/bin/python`, never a bare
  `python`.
- `src/vgc/` is a proper src-layout package, importable as `vgc` once
  `uv sync` has run (editable install) -- do not import it as `src.vgc`.
- `runs/` is gitignored (match results, gate reports, experiment log). `data/champions/`
  IS committed -- it's the format's ground truth and regenerating it is one command.
- `ruff` line-length 100.
- `PolicyConfig` (`vgc/models.py`) is the single frozen-dataclass gate for behavior
  changes -- new strategic knobs go there, individually commented, not as bare
  literals in the decision code. Mirrors `~/code/projects/pokemon-tcg-ai`'s pattern.
  Heuristic weights themselves are frozen as of 2026-08-12 (the Protect retune did
  not generalize). Treat the shipped heuristic as a benchmark, not something to
  keep optimizing.
- `VgcPlayer.decide()` / `decide_teampreview()` (`vgc/agent.py`) are the only methods
  subclasses should override; `choose_move`/`teampreview` themselves exist only to wrap
  those hooks in an exception-safe fallback (random move / `/team 1234`) so a bug in
  strategy code can never crash or forfeit a battle.
- `torch` is a `train` extra (`uv sync --extra train`). Training code may import it
  after that extra is installed. Do not add a hard runtime dependency on torch in the
  default (non-train) install.
- `vgc/stats.py` (Stat Point -> final stat), `vgc/damage.py` (the damage calculator),
  and `vgc/sets.py` (opponent `PokemonState` assembly) are Phase 2a's API surface for
  Phase 2b's evaluator to build on -- see "Damage/mechanics engine (Phase 2a)" above.
  New damage-relevant item/ability support belongs in `vgc/damage.py`'s
  `ITEM_*`/`ABILITY_WHITELIST` tables, not scattered literals. Treat `vgc/stats.py` and
  `vgc/damage.py` as effectively frozen -- their ground-truth tests
  (`tests/test_*_ground_truth.py`) are the bar for touching them; a Phase 2b-era change
  there should be a genuine bugfix backed by a new/updated test, not a scoring tweak
  (scoring tweaks belong in `vgc/evaluator.py`/`vgc/team_preview.py`'s `PolicyConfig`
  weights instead).
- `vgc/evaluator.py` (in-battle move/switch scoring) and `vgc/team_preview.py` (4-of-6 +
  lead pick) are Phase 2b's API surface -- see "Heuristic evaluator / team preview
  (Phase 2b)" above. New strategic behavior belongs there (and its weight in
  `PolicyConfig`), not as a new special case bolted onto `vgc/agent.py`.

## TypeSafe/Jev usage

Use TypeSafe/Jev for fuzzy or semantic judgment that would otherwise need brittle
heuristics, regex/string matching for meaning, classification, scoring/ranking subjective
properties, probabilistic yes/no decisions, or an LLM call for a small judgment.

Do NOT use Jev for deterministic calculations, schema validation, exact comparisons,
normal business logic, or open-ended generation / complex reasoning.

When Jev is appropriate:

1. Load the TypeSafe skill.
2. Decompose the problem into atomic judgments.
3. Batch independent judgments where possible.
4. Keep questions and thresholds centralized.
5. Validate important judgments with test cases.

**Status: parked (2026-09-29).** No Jev code is in the bot. The first planned use,
tagging why ladder games were lost, needs a VGC-competent person to hand-label an answer
key first (`tools/build_loss_review.py` builds the sheet); labels written by another AI
only measure AI-to-AI agreement. Until such a reviewer exists, study losses with code
that needs no judgement: `tools/loss_patterns.py` (checkable facts, losses vs wins) and
the engine re-check of lost decisions (`offline/review_lost_decisions.py`). Reopen Jev only with a real answer key or a new,
concrete fuzzy-judgement need.

Project rules for this bot (reviewed 2026-09-29; see the closed PR #8 for why):

- **Never call Jev inside the per-turn move-choice loop** (`score_joint_orders`,
  `search_joint_orders`, exact search, rollouts, or any reusable scoring function).
  Those run many times per decision -- once per side, per hidden-state hypothesis, and
  again during teacher collection -- so a hook there multiplies network calls and adds
  blocking latency before search. Call Jev explicitly, once per thing being judged
  (e.g. once per game at team preview, or offline over replays and traces).
- **Public information only.** Build Jev state through the same fog-safe boundary as
  live play; never from a private simulator root or the opponent's side of a direct
  battle. Normalize sentinels (`vgc.sets.normalize_item` for `unknown_item`) and keep
  observed facts separate from estimates in the state you send.
- **Jev is not a Pokemon strategist.** Exact outcomes come from Showdown, and move
  judgement from search. Good fits are fuzzy, text-shaped, time-insensitive labels:
  opponent team archetype at preview (baseline: `vgc.principles.detect_team_signals`),
  loss-reason tagging next to `vgc.postmortem.classify_loss`, replay-corpus labelling.
  Never use Jev to invent replay action labels or training targets.
- **Measure before it touches play.** Pin the evaluated model version (not
  `jev-latest`), check accuracy on a hand-labelled set, put any behavior knob and
  threshold on `PolicyConfig`, and require a same-session A/B before it changes moves.
- **"Faster" needs a named saving.** A Jev call is a network round trip, slower than
  the local myopic evaluator or neural shortlist. Only claim a speedup if it removes
  measured work (e.g. fewer searched candidates) worth more than the call costs,
  including slow responses.

## Pattern source

`~/code/projects/pokemon-tcg-ai` is the sibling project this one's conventions are
copied from (`PolicyConfig`, exception-safe `decide()`, Wilson-CI
`offline/run_matches.py` + `offline/run_gates.py`, JSONL `offline/log_experiment.py`,
`ContextVar`-based `decision_trace.py`). When in doubt about a pattern not covered here,
check how that repo does it first.
