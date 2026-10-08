# pokemon-vgc-ai

Pokemon Showdown VGC bot for `gen9championsvgc2026regmc` -- "[Gen 9 Champions] VGC 2026
Reg M-C": doubles, bring-6-pick-4, level 50, Megas allowed. The format offers mutual-
consent Open Team Sheets, but the bot rejects them and assumes no opponent sheet on the
Bo1 ladder. Backed by the local Showdown checkout's `champions` mod, not vanilla gen9.

**Read [CLAUDE.md](CLAUDE.md) for the full handbook** (Champions quirks, gates, exact
mechanics, closed experiments, commands). When this file and CLAUDE.md disagree,
CLAUDE.md wins. Human overview: [README.md](README.md).

The live format id is `vgc.config.FORMAT_ID`. Champions data was re-exported for M-C on
2026-09-09 (390 species/formes, 166 items, 509 moves, 222 abilities). Public M-C replays
are still scarce; keep the M-B replay tree and fold new rated M-C games in incrementally.
Do not rebuild usage priors from the small M-C snapshot. Goal: 1700+ Elo on the public
M-C ladder; offline checks do not establish that.

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
  `python`. Training code needs `uv sync --extra train` (installs `torch`).
- `src/vgc/` is a proper src-layout package, importable as `vgc` -- not `src.vgc`.
- **ALL game data must come from `data/champions/*.json`.** Stat Points, not EVs.
  Validate teams with `validate-team gen9championsvgc2026regmc`.
- `PolicyConfig` (`vgc/models.py`) is the single frozen-dataclass gate for behavior
  changes. Heuristic weights are frozen as of 2026-08-12.
- `VgcPlayer.decide()` / `decide_teampreview()` are the only methods subclasses should
  override.
- Learned Q, opponent-response models, and the pure-RL 100k recipe are **closed**.
  Current learned checkpoints are experimental; search remains the decision authority.

## Cursor Cloud specific instructions

- Dependencies: `.github/scripts/install_python_deps.sh` (locked `dev` extra plus CPU
  torch). Run `.venv/bin/python`. Node 22 is the nvm install under `~/.nvm`;
  `vgc.node.find_node` selects it.
- Pokemon Showdown is checked out at `~/pokemon-showdown` and symlinked to the
  default `SHOWDOWN_REPO` path (`/Users/edmundyu/code/projects/pokemon-showdown`).
  The pin is `data/champions/mechanics_catalog.json`. Set `VGC_SHOWDOWN_REPO` to
  point somewhere else. Refresh with
  `.github/scripts/checkout_pinned_showdown.sh ~/pokemon-showdown --build`.
- Boot starts `node pokemon-showdown start --no-security` on port 8000.
  `offline/run_matches.py` and `ladder/run_ladder.py --local-smoke` use that
  server. `pytest -m integration` starts its own server on port 8000, so stop
  the boot server before that suite.
- `data/selfplay/archetype_pool_150` is not in git. Tests that need it skip
  until `tools/build_archetype_pool.py --variants-per-archetype 25 --seed
  20260901 --out data/selfplay/archetype_pool_150`.
- Public ladder credentials are not required for local tests or smoke battles.

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

Re-reviewed 2026-09-30 (Opus and Codex/Sol independently): team-preview archetype
tagging, loss labelling and replay labelling all stay parked. A tag that can be checked
is cheaper and exact as plain code; one that cannot has no answer key. That review found
the real preview gap was code, not judgement -- see "Hidden opponent sets at preview".

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
