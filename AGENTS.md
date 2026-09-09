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
