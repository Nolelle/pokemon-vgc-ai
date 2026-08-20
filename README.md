# pokemon-vgc-ai

A rules-first Pokémon Showdown bot for the Champions VGC 2026 Reg M-B doubles ladder.
It uses the Champions mod's exported data, a simulator-checked damage engine, and an
explicit one-turn evaluator before any learned components are introduced.

## Full learning pipeline

The learned bot now has one end-to-end, default-off path:

1. exact Champions rules and our complete six-Pokemon team;
2. public high-level replays for an initial human-like state representation;
3. full joint-action imitation from the simulator-backed search teacher;
4. reinforcement learning against mixed opponents, old model snapshots, and varied teams;
5. a held-out-team promotion gate; and
6. explicit checkpoint deployment to local smoke or public ladder games.

Opponent information stays fogged. Revealed moves, items, abilities, move order, and
damage update a probability distribution over plausible hidden sets; the model never
reads the simulator's private opponent state. Every legal doubles order also receives
raw damage, knockout, Speed, threat, Protect, switching, targeting, and coordination
facts from the rules engine.

The shipped heuristic remains the default. A learned checkpoint is usable only through
an explicit path and is not promoted merely because training completed. See
[`docs/full_learning_pipeline.md`](docs/full_learning_pipeline.md) for the commands,
data boundaries, measured smoke results, and promotion rules.

## Local verification

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m integration
.venv/bin/python -m ruff check .
```

The project automatically selects an installed Node 22 executable. Set `VGC_NODE` if
Node 22 lives somewhere unusual.

## Offline evaluation

Start the sibling Showdown server, then run the two acceptance gates:

```bash
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent random --n 100 \
  --threshold 0.90 --team teams/meta1.packed.txt
.venv/bin/python offline/run_gates.py --candidate vgc --incumbent heuristic --n 300 \
  --threshold 0.65 --team teams/meta1.packed.txt
```

Both players explicitly accept Open Team Sheets by default. Use
`--no-open-team-sheets` to make both reject for a controlled comparison.

## Curated metagame teams

`data/meta/popular_teams_H8v7TEZcbXo.json` contains the ten teams shown in JoeUX9's
"The Most Popular Teams In Pokemon Champions Explained": 60 complete sets plus the
stated roles and common leads. At team preview, `vgc.meta` recognizes only an exact
six-species match. The evaluator then uses the video's hidden nature for its Speed and
damage estimates and records the archetype in `VGC_TRACE` output.

Live Open Team Sheet information always wins for moves, items, and abilities. The video
did not provide Stat Point spreads, so those still come from `data/usage/spreads.json`.
Validate the curated ids against the Champions export with:

```bash
.venv/bin/python -c 'from vgc.meta import validate_meta_teams; assert not validate_meta_teams()'
```

## Ladder sessions

First exercise the replay, trace, and JSONL logging pipeline locally:

```bash
.venv/bin/python ladder/run_ladder.py --local-smoke --n 2
```

For the public ladder, provide credentials through environment variables:

```bash
export VGC_SHOWDOWN_USERNAME='your-account'
export VGC_SHOWDOWN_PASSWORD='your-password'
.venv/bin/python ladder/run_ladder.py --n 1
```

Alternatively create the gitignored `.showdown-credentials.json`:

```json
{"username": "your-account", "password": "your-password"}
```

Use owner-only permissions (`chmod 600 .showdown-credentials.json`). Replays and
decision traces go under `runs/ladder/`; one outcome record per game is appended to
`runs/ladder.jsonl`. Start with small, respectful sessions.
