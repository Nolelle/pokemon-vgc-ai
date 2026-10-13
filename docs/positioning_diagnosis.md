# Positioning diagnosis (2026-10-12, branch claude/positioning-diagnosis)

Question: losses are tagged `lost_positioning` more than anything else. What is the
bot actually doing wrong, and which fix would have fired on real losses? No weights
were changed; this is measurement only.

## Evidence

1. **Loss tags, all 201 public M-C ladder games** (`runs/ladder.jsonl`): 102 wins,
   99 losses. Of the 99 losses, **69 (70%) are primarily `lost_positioning`** — the
   bot stayed in against a combined KO threat without Protecting or switching
   (`vgc/postmortem.py`). Next: unnecessary_prediction 12, poor_speed_control 11,
   sacrificed_closer 6.

2. **Checkable spectator facts** (`tools/loss_patterns.py`, same 201 games):
   - One of ours KO'd before it could act: **84/99 losses vs 42/102 wins** (p≈0.000).
   - We scored the first KO: 39/99 losses vs 71/102 wins (p≈0.000).
   - They set Tailwind: **29/99 losses vs 9/102 wins** (p≈0.000).
   - Luck also shows up (our misses 33 vs 16, p=0.005; crits on us 35 vs 18,
     p=0.006) — real, but smaller than the speed/positioning gaps.
   Facts are correlations, not proven causes.

3. **Engine re-check of the 6 most recent `lost_positioning` losses**
   (`offline/review_lost_decisions.py`, Oct-12 session, current code;
   `runs/eval/positioning_diagnosis_review.json`): 44 decisions reviewed,
   live reproduction 59% (the tool rebuilds the fast pick, not the judge's —
   known limit). Regret > 10 on 20/44 (45%). Among those 20:
   - Deep best was **Protect in 11**, a different attack/target in 8, a switch in 1.
   - The bot had attacked in 14, Protected in 4, switched in 2.
   - With two of our Pokémon active (35 decisions, 16 at regret > 10): deep best
     Protect 9, attack 6, switch 1; bot switches 2 vs deep-best 1.
   - The largest regrets (3000+) are late-game turns (T5–T8) scored as terminal
     values; per the tool's own caveat, with one Pokémon left Protect wins any
     one-turn comparison, so those mostly measure one turn of delay.

## What this corrects

The Oct-10 deep review found the deep best switched far more than the bot
(16 vs 6 at regret > 10). **That does not replicate on current code** in this
sample: switches are not what the wider search asks for. Two things changed
since: the repeat-Protect rule and the mechanics-correctness pass. The
remaining same-scoring misses are (a) attacking into a lethal threat instead
of Protecting, and (b) target choice (e.g. Knock Off @1 vs @2; Tailwind vs
Brave Bird on T5).

The re-check **cannot** judge the valuation hypothesis (price placement in
`_position_value`): both arms share the same hand-weighted scoring, so a
systematically wrong valuation is invisible here by construction.

## Ranked next levers

1. **Protect-vs-attack under lethal threat** (fires on real losses, measurable
   by replay): the fast scorer's expected-death cost only switches on at
   exactly 100% threat. Price expected health loss continuously below 100%
   and re-run this review; success = fewer regret>10 attack-into-Protect
   misses on held-out ladder losses.
2. **Target selection** (8/20 misses): attack targeting, not move choice.
   Diagnose per-target regret before any change.
3. **Speed control as position** (Tailwind gap 29 vs 9): still unpriced in
   `_position_value`; needs the pre-registered field/plan confirmation, not
   tuning on these games.
4. **Placement value in `_position_value`** (the earlier hypothesis): untested,
   not refuted — this tool cannot see it. It needs its own A/B with a
   PolicyConfig knob; do not claim this diagnosis as evidence for or against it.

Sample limits: 6 games / 44 decisions for (3); one team (psyspam_sand session);
regret is measured under hidden-set beliefs. Widen (3) before acting on the
exact counts.
