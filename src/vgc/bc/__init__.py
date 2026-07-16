"""Phase 3 behavior-cloning (BC) scaffold over `data/bc/decisions.jsonl`
(`vgc.replay_parse`'s output).

SCOPE: this package validates a PIPELINE (encoder -> dataset -> training -> accuracy),
it does NOT produce a playable agent. `runs/bc/bc_policy.pt` predicts a human-plausible
next move id from a state snapshot -- it has no notion of legal targeting, switching
constraints, or how to turn its output into a `DoubleBattleOrder` `vgc.agent.VgcPlayer`
could execute. Do not wire this checkpoint into `VgcPlayer.decide()` and call it a bot.

Only `vgc.bc.encoding` is imported here -- it has NO torch dependency (pure numpy/stdlib)
so `import vgc.bc` and `import vgc` never require the `train` extra. `vgc.bc.dataset`/
`vgc.bc.model`/`vgc.bc.train` all need torch (`uv sync --extra train`) and are only
imported explicitly by `tools/train_bc.py` and `tests/test_bc.py` (which skips cleanly
via `pytest.importorskip` when torch is absent) -- never import them from here.
"""

from __future__ import annotations
