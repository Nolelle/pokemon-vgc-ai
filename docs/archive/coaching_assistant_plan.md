# Replay coaching assistant: build plan

Status: PLAN v1 (2026-10-05). Reviewed by Opus 5.5 and Codex/GPT-6.1 Sol (independent,
read-only); Sol's corrections are folded in. Owner vision: last section.

## What we are building

A player gives us a Showdown replay of their Reg M-C game plus their team paste. We
return a post-game review of the few decisions that mattered: what they chose, what the
engine would have tried, what each line actually does in Showdown, and *why* -- explained
from first principles (speed, damage, threats, positioning) and as a short plan for the
next turns, not generic tips. Later: many games per player, to find their habits.

**First promise (narrow on purpose):** "Understand a few important decisions, compare the
alternatives, see the evidence." Calling a move a *mistake* is a stronger claim we earn
only after Part 6 passes. A fluent explanation of a weak finding is the fastest way to lose
a player's trust.

Pipeline:

    replay link + team paste (+ player name)
      -> [1 reference games]  answer keys for every later check
      -> [2 intake]           fetch replay, validate paste, pick the player's seat
      -> [3 rebuild]          what the player could see at each decision
      -> [4 choices]          what the player actually chose at each decision
      -> [5 grading]          Showdown-exact comparison of their choice vs alternatives
      -> [6 judge checks]     when is a grading trustworthy enough to coach on?
      -> [7 evidence report]  key decisions + every fact with an ID, reproducible
      -> [8 coach]            LLM writes the review, citing evidence only
      -> [9 combine]          one command, then a web app
      -> [10 profile]         many games -> the player's habits, tracked game by game

Part 0 (judge strength / ladder climb) is a separate track the coaching depends on.

## Rules for every part

- **Each part has a "done when" check and must pass it before a later part relies on it.**
  Parts hand over saved files, so each can be tested alone.
- **Public information only, cut at the decision.** Each decision is rebuilt from the
  replay up to that moment. Later moves, revealed items or the winner must never leak into
  an earlier judgement. The opponent is beliefs, exactly as in live play.
- **Unknown is a valid answer.** If a decision cannot be rebuilt or a choice cannot be
  recovered, mark it and skip it. Never guess and then coach on the guess. Reports show
  how many decisions were analysed vs skipped.
- **The LLM explains; it never decides.** Every number and claim points at a replay event
  or an engine result.
- **Replay chat is never shown to the LLM** (a player could type instructions into chat).

## Part 0 -- judge strength (separate track)

Coaching can only be as right as the engine's judgement. Verified gaps in
`vgc.rl.exact_search._position_value` on main: no KO term (PR #29 has one, not merged),
no value for Trick Room/weather/terrain beyond their immediate effect, flat boost/status
weights, one-turn horizon, myopic score added on top (damage counted twice). Sol's
example of the danger: with one Pokemon left, Protect avoids the -10,000 "lost" score for
one more turn, so the judge can rate a pointless Protect hugely -- a terrible lesson.

- Merge PR #29; work toward multi-turn lookahead (needed for "think turns ahead").
- Ladder-test the exact judge (no public ladder games yet; best M-C rating ~1150).
- Define the marketing claim precisely before using it: account, rank, date, game count,
  sustained (not a one-off peak).

## Part 1 -- reference games (answer keys; build first)

Two sources:

- The bot's 46 recorded M-C ladder games (`runs/ladder/state-replays/*.json`, main
  checkout): full player view including Showdown's private `|request|` messages and the
  submitted choice.
- New local games from `offline/record_positions.py` / `vgc.positions.play_recorded_game`
  (`RecordingAgent`, `DecisionReplayRecorder`), recorded **from both seats**, across the
  pool teams. Deliberately include forced switches, Choice items, Encore/Taunt, Follow
  Me/Rage Powder, Shadow Tag trapping, spread moves, Mega, fainting before moving.

For each game also save the spectator view: what a public replay shows (no `|request|`,
both sides' HP as rounded percentages, no submitted choices).

Done when: 200+ reference games (both seats) with player view, spectator view, exact
teams and submitted choices, and a coverage table of the hard mechanics above.

## Part 2 -- intake

Fetch `https://replay.pokemonshowdown.com/<id>.json`; parse the paste (text or
pokepast.es) into a packed team; run `validate-team`; the player names their account,
which maps to p1/p2 (matching species alone fails when both teams are similar). The paste
must match the preview's six. Which four were brought is only partly visible (a brought
Pokemon may never switch in); carry that as explicit uncertainty or ask.

Done when: reference games and 10+ real Open-Team-Sheet replays ingest correctly; a
mismatched paste, wrong format or unknown player is rejected with a clear message.

## Part 3 -- rebuild the player's view (the main feasibility question)

New adapter: spectator log + packed team -> a `vgc-decision-replay-v1` bundle
(`vgc.battle_state_replay`) with synthetic `|request|` messages (actives, HP, moves, PP,
Mega availability, forced switches, visible trapping/disabling) and a decision cut point
before each turn and each forced switch. Then reuse `_replay_player`, `LiveExactMirror`
and `vgc.positions.rebuild_position` unchanged.

Own HP from a public replay is a rounded percentage. Exact max HP comes from the paste,
but current HP can be ambiguous by a point or two. Near a survival threshold keep the
compatible range rather than silently picking one value.

Done when, on the reference games:
- every accepted decision matches the real one on actives, visible effects, HP within
  the rounding range, and the legal action set (compared via
  `vgc.mechanics_state.snapshot_battle` / `decision_state_payload`);
- unsupported decisions are marked incomplete, never wrong;
- **no-leak test:** appending later events to the log never changes an earlier decision;
- coverage (share of decisions accepted) is reported per mechanic. This number decides
  how much coaching replay-plus-paste can honestly support.

## Part 4 -- recover the player's choices

Replay lines are outcomes, not submissions. Map each turn's events to a joint order from
`vgc.actions.enumerate_joint_orders` (`describe_order`, `choice_wire_message`). Hard
cases: redirection (Follow Me shows the redirected target), called/forced moves, a
Pokemon that fainted or flinched before moving, `|cant|` with no move name, spread moves,
and a forced replacement or Roar/Whirlwind drag, which is not a chosen switch.

Done when: on the reference games, **every** choice declared recovered equals the
recorded submission (a wrong recovery is worse than unknown), and recovery coverage is
reported separately.

## Part 5 -- grading

For each recovered decision run `offline/grade_positions.py`'s `grade_position` with a
forced candidate set: the player's choice plus the engine's top choices
(`vgc.rl.public_search.public_information_exact_search`). Every compared order is
actually searched, under every opponent belief, with the same random futures (a fair
comparison). Keep what explanations need, which `search_joint_orders_exact` currently
throws away: per-branch Showdown lines and state changes (damage, KOs, move order),
per-opponent-response values, per-belief values. Unsearched orders get placeholder low
scores in the existing ranking; those must never reach a report.

Done when: branches really advance and the original state is untouched (existing
`tests/test_live_mirror_branches.py` pattern); results on the bot's lost games agree with
`runs/eval/lost_decision_review.json`; whole-replay time, slow cases and memory are
measured (old full-width runs: ~30 s/decision, ~4 min/game on 6 workers).

## Part 6 -- judge checks: when is a grading trustworthy?

No VGC expert is available, so each check proves something narrower:

| Check | Proves |
|---|---|
| **Continuation test:** from a position, play the recommended order and the player's order, then finish the game with several fixed bots, seats swapped, many seeds | The recommendation actually wins more (main evidence) |
| **Small solved endgames**, incl. the Protect-delay trap | The judge does not reward stalling a lost position |
| **Stability:** more samples, wider search, other beliefs | Advice does not hang on one fragile assumption |
| **High-rated player agreement** on decisions we can recover, by rating band, humans and bots separated | Plausibility only; the corpus has just 776 games rated >=1500 and 82 >=1600, and most lack exact sets |

Regret vs eventual game result is NOT used as evidence: losing positions naturally score
badly, so the correlation would look good even for a broken judge.

Keep evaluation teams/players separate from anything used to build beliefs or tune the
judge (team-disjoint split, clustered intervals, as in the multi-team gates).

Done when: the continuation test shows recommended orders win more than played orders on
held-out teams by a pre-declared margin, and the Protect-delay endgames are not flagged.
Until then, reports say "the engine preferred X" with evidence, never "mistake".

## Part 7 -- evidence report

Pick 3-5 decisions worth discussing: large, stable gaps that pass Part 6's filters (not
"already lost anyway"). For each, a structured record: board, speed order, the player's
choice and its Showdown lines, the engine's line and its lines, the next 1-2 turns, what
was known vs estimated vs unknown about the opponent, every fact with an ID.

Done when: the same saved evidence regenerates the identical report without re-searching;
every fact traces to a replay event or an engine branch; the picked decisions are stable
across seeds.

## Part 8 -- LLM coach

Reuse `vgc.llm` (client, `SpendMeter`, fake client, deadlines). New coaching prompt and
response schema: `claims[{text, evidence_ids}]`, `principle`, `next_turns_plan`.
Citations alone are not enough (a real fact can be misread), so start conservative: hard
claims (numbers, KOs, speed) come from code-written sentences built from the evidence;
the LLM adds the explanation around them. Banned: "you would have won", hidden items
stated as known, causes not in the evidence. If validation fails, fall back to the plain
factual report.

Done when: fake-client tests reject invented causes, wrong numbers, hidden info stated
as fact, and unsupported win claims; on 20 real reviews nothing ungrounded gets through;
the owner reads 5 reports and judges them clear (readability, not correctness).

## Part 9 -- combine

`offline/coach_replay.py --replay <url> --team <paste> --player <name>` -> report. Then a
thin web app with background jobs (a review takes minutes), the Showdown engine hosted
server-side, and reports that show analysed vs skipped decisions, assumptions and engine
version.

Done when: held-out end-to-end games pass all earlier checks.

## Part 10 -- player profile: habits across many games

Owner idea (2026-10-05): grade a batch of the player's past replays first, so feedback is
about *their* recurring habits ("you attack into Protect on turn 1 in 6 of 10 games"),
then track each new game against that baseline.

- Fetch a player's public replays by username (replay search API); one paste per team.
  Ladder games are public only if someone saved them, so also accept uploads.
- Habit tags are counted by code from graded decisions (situation x choice type: turn-1
  leads, speed control, Protect, switching, targeting), never labelled by an LLM.
- Compare with players one rating band up (from Part 6's graded corpus): "you do this
  more than 1500-rated players" is backable; "you do this a lot" is not.
- Minimum games plus a clustered interval per habit; below that, "too few games to tell".

Done when: on a few real players with 20+ public games, the top habits are the same in
both halves of their games.

## Build order

| Milestone | Parts | Proves |
|---|---|---|
| M1 | 1, 2, 3, 4 | We can see what the player saw and what they chose -- the feasibility answer |
| M2 | 5, 7 | The engine can compare choices and produce traceable evidence |
| M3 | 6 | The comparison is trustworthy enough to call something a mistake |
| M4 | 8, 9 (CLI) | End-to-end grounded review on real games |
| M5 | 10 | Personal habits across games |
| M6 | 9 (web) | Product, gated on Part 0's ladder claim |

Part 1 goes first: it is cheap and every later check needs it. Parts 2, 3 and 4 can then
be built in parallel. Part 0 runs throughout.

## Owner vision

Marketing basis: "we built a bot that climbed the Showdown ladder to the top; it reviews
your replays with you to help you do the same." Coaching teaches the mindset from first
principles and how to look turns ahead, personalised from the player's own body of games
-- not generic advice.
