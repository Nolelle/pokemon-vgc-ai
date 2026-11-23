# Our replays support partial learning, with important gaps in complete choices

Prepared November 2, 2026. Frozen replay coverage through November 1.

**Current assessment**

**Executive summary.** We have 65,011 saved Regulation M-C battles and 1,281,380 decision records. After conservative selection, 58,961 battles supply 1,155,762 prepared records. The data is useful for initial learning of observed moves, team selection, and match outcomes. It cannot yet teach the complete original two-Pokémon choice.

The training pool contains 604,210 enabled move observations, 75,597 brought-four sets, and 915,867 outcome records. Its 1600+ subset contains 41,223 move observations. Missing answers are explicitly disabled. The repeated-lineup correction reduces the largest lineup's expected outcome-sample participation from 34.0% to 2.1%.

The main risks are outcome-dependent collection near the rating cutoff, absent private team details, correlated examples from repeated players and games, and differences between older training games and newer testing games. No model was trained or evaluated for playing strength as part of these checks.

**What each number means**

**A game is the independent collection unit.** One battle supplies two player perspectives, many turn-start positions, and sometimes two move observations per position. A label is an answer used to teach the model. A mask tells the learner whether an answer is known. These counts overlap and must not be added together as independent samples.

| Unit | Count | Meaning |
|---|---|---|
| Games | 65,011 | Distinct replay IDs |
| Player-game perspectives | 130,022 | Two views of each game |
| Normal turn records | 924,518 | One player view at turn start |
| Replacement records | 226,840 | Mid-turn or replacement state |
| Preview records | 130,022 | One per player per game |

Copies of the same outcome on many turns are correlated. Confidence intervals or later significance tests should group by battle; team/account dependence may require additional grouping. A million records do not mean a million independent experiences.

**Collection scope and time coverage**

**This is a saved public-replay sample, not the whole ladder.** The latest refresh added 10,141 battles to the 54,870-game baseline. Saved uploads run from September 9 through November 1, 2026; the latest battle begins November 1 at 16:32 Edmonton time in the full collection. The report was prepared November 2. No November 2 games are implied.

Collection used a shared post-game replay rating threshold of 1200, a 1,200-page limit, 0.3-second request spacing, and exclusion of the known project account. The incremental listing audit checked 479 pages, repaired two missed IDs, retained all baseline IDs, and found all audited eligible IDs present. Timestamp pagination and public replay availability prevent a claim of complete archive coverage.

Daily bars show **upload counts in UTC**, not ladder activity or battle-start volume. The first and last days are partial; the November 1 endpoint is a collection cutoff. Changes in saved volume cannot establish changes in player activity.

**Most recorded players were below 1600**

**The median pre-game rating is 1,349; the middle half lies between 1,289 and 1,446.** The 90th percentile is 1,589. Rating is a ladder estimate, not a certificate that every move is strong.

There are 12,101 player-game perspectives at 1600+, or 9.3% of all player perspectives. In 4,589 games both players qualify; in 7,512 at least one qualifies. Those qualifying perspectives contain 6,151 wins and 5,950 losses. Both outcomes are retained.

Ratings come from each player's rating-change message before the battle result is applied. The shared replay rating and player-join rating are not interchangeable with this value. Every player perspective has a recovered pre-game rating in this frozen snapshot.

**The download threshold creates a visible selection problem**

**Games near the collection threshold have a very different result mix.** Among 2,682 games with unequal ratings and a lower-rated player below 1200, the higher-rated player wins only 5.4%. When both players start at 1234 or above, the higher-rated player wins 53.2% across 56,015 games. Equal-rated games are omitted from this comparison.

The download filter uses a post-game rating, so the result can influence whether a battle enters the sample. This pattern is consistent with that mechanism; it is not evidence that lower-rated players normally beat stronger players 94.6% of the time. Filtering by pre-game rating later cannot recover uncollected games. The 1234 boundary is a documented sensitivity cut, not proof of zero bias above it.

This particularly affects outcome-learning claims and comparisons of skill groups. A future collection without an outcome-dependent threshold would provide a cleaner reference.

**Game endings and length**

**The typical saved game lasts seven turns; 90% end by turn 11.** The longest has 68 turns. There are 11,517 games shorter than five turns (17.7%); short games can reflect a decisive opening or a concession and are not automatically discarded.

41,832 games are logged as played out, 22,392 end in forfeits, and 787 end through timer/offline losses. Outcome teaching omits timer/offline endings and retains ordinary forfeits. Cause classifications follow the recorded messages; an unannounced disconnect cannot be identified reliably.

Every resolved battle supplies one winner and one loser, so a 50% game-side win rate is automatic and says nothing about model strength. The unequal win/loss record totals arise because outcomes are repeated across different numbers of positions.

**Teams and Pokémon are highly unevenly represented**

**The collection contains 15,040 normalized account names and 28,313 six-species lineup signatures.** Accounts are not verified people; aliases and automated players may exist. No additional account is classified as a bot by this report.

The most common lineup—Archaludon, Charizard, Farigiraf, Golisopod, Grimmsnarl, and Politoed—appears in 17,587 games, 27.1% of the collection. Mirrored lineups count once per battle. The largest account participates in 6,020 games, 9.3% of the collection.

There are 296 distinct raw preview names. Of these, 292 match the 390-species/form catalogue (74.9%); four names fall outside that exact catalogue: Alcremie-Salted-Cream, Florges-White, Florges-Yellow, and Furfrou-Kabuki. This is an identifier/form-normalization question to review, not proof of illegal battles. The catalogue includes battle forms that may not occur by name at preview, so this fraction is not a clean measure of strategic coverage.

The chart counts appearances in player previews (denominator 130,022), not bring-four selections or wins. Popularity is not measured strength.

**The recorded inputs omit much of the player’s private knowledge**

**Public observations are incomplete even on the player's own side.** The chart covers 1,751,349 active-Pokémon snapshots from normal-turn records. An item is recorded in about 18.4%, an ability in 18.0%, and at least one move in 70.0%. These are repeated snapshots, not distinct Pokémon. Both seats are included symmetrically, which makes the aggregate own/opponent rates equal.

An absent field means “not observed in the public log,” not “the Pokémon has no item or ability.” A real player knows their own private team information; the spectator record often does not. Future item or move reveals remain outside earlier inputs. Exact Stat Points, unrevealed moves, and complete legal choice requests generally cannot be recovered.

Across 1,849,036 active-slot opportunities in normal-turn records, 254,963 actions are unknown (13.8%); 97,675 slots require no action (5.3%). These are different states, and neither is taught as an intentional pass. A logged target is also not proof of the original submitted target after redirection.

**Conservative selection preserves 58,961 games**

**4,683 distinct battles are set aside for information-setting or reconstruction review.** Another 1,367 older games are reserved away from training/development to protect the unfamiliar-lineup test. Original files are preserved.

| Review reason | Games |
|---|---|
| open_team_sheet | 2625 |
| state_review_baton_pass | 1376 |
| state_review_illusion_or_replace | 610 |
| state_review_transform | 122 |

The review reasons overlap and must not be summed. Disclosed opponent sheets do not match the intended closed-sheet setting. Baton Pass, Transform, and Illusion introduce known state or identity concerns. Basic checks found no malformed format/schema records, invalid health/stat ranges, or preview-appearance mismatches in the inventory. Those checks do not certify every battle mechanic.

**Testing moves forward in time and keeps entire games together**

**The split begins with approximately 80/10/10 of eligible games by battle start time, then removes reserved-lineup appearances from earlier groups.** All turns, both players, all tasks, and both broad/1600+ stages keep the same game assignments.

| Group | Games | Records | 1600+ perspectives | 1600+ share |
|---|---|---|---|---|
| train | 47096 | 925046 | 6081 | 6.5% |
| validation | 5832 | 114118 | 1564 | 13.4% |
| test | 6033 | 116598 | 2054 | 17.0% |

Training spans Sep 9–Oct 19; development spans Oct 19–24; final testing spans Oct 24–Nov 1 (Edmonton dates). Development is the practice exam used for model decisions. Final testing is the sealed exam used after selection. The newer test has substantially more 1600+ perspectives than training, so its difficulty and population differ. A lower test score could reflect this shift as well as poor learning.

The final test includes 445 battles involving 176 reserved six-species lineups. Those lineups occur on neither side in retained training/development. This subgroup overlaps the 6,033-game final test and must not be added to it. The other participant may have a familiar lineup. The ordinary chronological test allows overlap: 873 lineup signatures and 2,054 accounts also appear in training.

No final-test IDs occur in the old 54,870-game baseline used by documented replay fits. Older model files with unknown data history still prevent a universal non-exposure claim. Fresh initialization or exact model provenance is required.

**Enabled answers support partial learning**

**The prepared files explicitly separate observed facts from missing submitted choices.** Unknown answers are null with their teaching mask disabled. Raw move events are checked; games with detected called/locked/charge/recharge mechanisms conservatively lose move teaching. Only early switches before resolution events qualify. Bring-four sets and ordered leads must match independent appearance evidence. Bench order is not taught.

| Task | Training | 1600+ training | Development | Test |
|---|---|---|---|---|
| move | 604210 | 41223 | 73500 | 80613 |
| switch_destination | 100853 | 6975 | 12340 | 13515 |
| preview_brought | 75597 | 5106 | 9370 | 9784 |
| preview_leads | 93835 | 6074 | 11644 | 12024 |
| outcome | 915867 | 60628 | 112702 | 115350 |
| target | 0 | 0 | 0 | 0 |

Target-choice and complete two-Pokémon choice teaching are disabled. These numbers are enabled observations, not certified legal submitted actions. Conservative filtering reduces the initial 973,633 training move candidates to 604,210 enabled observations. Preview answers and outcomes overlap those same battles; adding task counts would double-count evidence.

The 1600+ training group contains 6,081 player-game perspectives before task-specific masking. After masking, 4,199 perspectives provide a move answer, 5,106 provide a bring-four set, and 6,048 provide accepted outcome positions.

**Balancing reduces repeated-lineup influence, with a tradeoff**

**The working sampler first controls each game's influence, then reduces emphasis on common lineups on either side.** Within each learning task, a game's weight is divided across eligible player perspectives and known answers. Long games and extra known slots do not automatically gain more influence. A game is downweighted by the square root of its more frequent lineup's training count; frequencies are fitted separately for broad and 1600+ training, never from development/test.

For outcome examples, the largest lineup's expected participation falls from 34.0% to 2.1%; for move examples it falls from 24.5% to 1.3%. These are changes to sampling, not evidence of stronger play. Actual 100,000-draw checks per phase found no held-out games, with the broad maximum lineup share 2.058% versus 2.095% expected.

Unequal weighting increases emphasis on rare lineups. Broad outcome weights have an effective size of about 14,678 equally weighted games out of 46,528 eligible games; the analogous move count is about 11,121 out of 31,174. Effective size describes concentration of influence, not a count of deleted games. In 1600+ sampling, the largest account still contributes roughly 9–10% depending on task. The recipe's effect on learning remains untested.

**What has been verified—and what has not**

**File integrity and data handling have strong direct checks; strategic strength has no new evidence here.** The fixed compressed source fingerprint matches its saved selection. All 65,011 raw joins and 130,022 player ratings reconciled in the independent inventory audit. The preparation audit compared all 1,155,762 exported inputs with their original snapshots and independently reconciled each task's sampling probabilities.

The reviewed split is pinned by fingerprint, so swapping training and test games fails even if totals remain identical. Unknown answers cannot be consumed as enabled labels. Final-test access requires explicit final-evaluation mode, and its records never receive training probabilities. Fifteen focused preparation tests passed; the preceding inventory/parser checks had also passed. These are code/data checks, not battle-performance tests.

Not established: full correctness of every reconstructed mechanic; the original complete action or target; exact private sets; representation of the entire ladder; independence of repeated accounts/teams; absence of all historical model exposure; simulator readiness; or any rating achieved by a learned model.

**What the dataset can support next**

**Use the prepared contract for partial-observation learning and outcome prediction, with evaluation kept separate.** It can train extra prediction outputs that improve shared board-reading representations. The old replay warm-start copies that shared part of the network; it does not train the complete-action scoring layers.

The current complete-action network combines both Pokémon's choices nonlinearly. Filling a missing partner choice with zero or pass creates an invented action. Training that scorer requires an explicitly compatible objective and suitable evidence, such as actually recorded legal requests and submitted choices. This report does not resolve that architectural/data gap.

Before a substantive experiment: implement a trainer that consumes these masks and saved sampling probabilities correctly; compare balanced and natural sampling on development data; report results by rating, turn length, team familiarity, and rare mechanics; verify model initialization provenance; and keep final-test results out of design decisions. For population-level claims, improve collection around the rating cutoff and resolve the four preview aliases. Practice through the simulator additionally requires the separate environment checks.

**Methods, definitions, and evidence trail**

**This report describes the frozen current dataset, with an evidence cutoff of November 1 and a report date of November 2, 2026.** All analyses are descriptive counts of saved files. No model was fitted, no simulator changed, and no ladder games played for the report. No uncertainty interval is attached to census counts; generalizing to uncollected battles remains limited by selection bias and dependence.

Account names are normalized platform IDs, not distinct people. Lineup signatures are sorted six-species previews, not complete hidden sets or strategically distinct teams. Species appearances count preview slots. Move/target observations count active slots. Outcome records repeat one final result across many positions. Stronger-player selection always uses the acting player's pre-game rating.

Sources include the verified refresh report, schema-7 records, immutable selection manifest, prepared dataset summary, independent verification outputs, the Champions catalogue, and fresh descriptive profiles. Source panels expose the exact inputs and reviewed rows for each section. The preparation and report scripts are retained in the repository's runs folders for reproduction. Historical reports using 54,870 games are not substituted for this 65,011-game snapshot.

The original post-game threshold and opt-in/public replay availability constrain coverage. The 1600 threshold and balancing exponent are design choices, not proven optimal settings. The unfamiliar-lineup test protects exact six-species signatures only; similar teams and accounts may still overlap.
