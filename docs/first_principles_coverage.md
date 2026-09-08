# Reg M-B strategic-signal coverage

> **Important correction (2026-08-27):** this checklist records whether a strategic
> idea changes the bot's score. It is **not** proof that the underlying game mechanic is
> simulated exactly. For example, Fake Out can receive a useful score bonus while the
> forecast still fails to cancel the target's action. See
> [the mechanics coverage audit](mechanics_coverage.md) for the stricter exact,
> approximate, score-only, and missing classification.

In this checklist, **Yes** means the principle has a named input or derived signal and
changes preview, action, or search scoring. The old heading called this
"first-principles policy coverage," which was too broad: it made score-only signals look
like verified mechanics. These rows should therefore be read as design coverage, not as
ladder-readiness evidence.

| Principle | Verdict | Implemented decision path |
|---|---|---|
| Bring four for the matchup | Yes | Preview scores all 90 choices using matchup, engine answers, closer inclusion, role coverage, and Mega plan. |
| Choose a functional lead | Yes | Leads are scored for immediate pressure/KO, speed control, setup denial, partner protection, forced switches, safe information, and denial of the opposing engine; passive leads are penalized. |
| Choose repair-capable backs | Yes | Backline safety, a second speed mode, defensive pivot coverage, and closer-in-back value all affect preview scoring. |
| Action economy | Yes | KOs, Fake Out, sleep/Yawn, flinch, Taunt, Encore, Protect, redirection, burn/Intimidate, and setup denial have explicit value. |
| Speed control | Yes | Tailwind, Trick Room, Icy Wind/Electroweb/Bulldoze, priority, Choice Scarf, paralysis, and weather abilities affect order or score; immediate partner KOs receive an extra synergy bonus. |
| Positioning and switching | Yes | Switches compare both likely incoming attacks, outgoing/incoming pressure, tempo, safe dual resistance, weather/Intimidate/Hospitality activation, collapsed roles, and closer preservation/entry. |
| Information | Yes | Revealed sets override priors; missing moves/item/ability/spread are counted as uncertainty; Protect receives bounded information value and future turns consume newly revealed battle state. |
| Endgame resources | Yes | Preview commits to a closer; per-turn game plans identify the closer, primary opposing threat, answers, and plan breakers; closer preservation and endgame entry affect actions. |
| Opponent engine | Yes | Rain, sun, sand, snow, Trick Room, Tailwind, screens, redirection/setup, spread offense, priority offense, action denial, and pivot cycles are detected. |
| Opponent closer | Yes | Whole-team matchup races identify the primary opposing threat; preview requires an answer and turn scoring prioritizes it. |
| What they can KO or establish | Yes | Context builds single- and double-target damage threats plus control/setup threats; default robust search includes joint attacks, Protect, utility/control, redirection, and defensive switches. |
| What we can KO and target value | Yes | Damage ranges, KO certainty, order, engine enablers, primary threats, plan breakers, low HP, Focus Sash/Sturdy, and useful focus fire affect target choice. |
| All four action classes | Yes | Every legal Attack, Protect, Switch, and Utility/Control pair is enumerated and scored. |
| Worst reasonable outcome | Yes | Robust search is the default and blends response likelihood with an explicit worst-case term across attacks, Protect, utility/control, redirection, and switches. |
| Pressure pairing | Yes | Spread+cleanup, spread+ally Protect, redirection+setup, speed control+immediate attack, Fake Out+setup, useful focus fire, and different-target pressure have explicit cross-slot terms. |
| Protect from first principles | Yes | Threat commitment, double-target danger, partner cleanup/disable, field/status stalling, information, repeated-use odds, low-threat cost, and next-turn repositioning affect Protect. Opponent free setup is represented in robust responses. |
| Mega Evolution strategy | Yes | Preview selects a default Mega; current-turn damage, Speed, weather/ability, and searched survival effects determine timing; unnecessary or off-plan Mega use is penalized. |
| Balanced-offense learning plan | Yes | Preview scores speed control, redirection/Fake Out, two attackers, a defensive pivot, a closer, engine denial, and a second speed mode. |
| Practical per-turn checklist | Yes | The trace records engine/closer plan, speed, double-target danger, uncertainty, control threats, candidates, response search, and chosen breakdown every turn. |
| Post-loss classification | Yes | Every ladder loss is automatically ranked into wrong four, wrong lead, poor speed control, lost positioning, sacrificed closer, missed calculation/fallback, unnecessary prediction, and unknown information, with trace evidence. |

## Verification commands

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m integration
```

The behavior gates are written to `runs/eval/first_principles_vs_random.json` and
`runs/eval/first_principles_vs_heuristic.json` (the `runs/` tree is intentionally
gitignored).
