# pokemon-vgc-ai

A rules-first Pokémon Showdown bot for the Champions VGC 2026 Reg M-B doubles ladder.
It uses the Champions mod's exported data, a simulator-checked damage engine, and an
explicit one-turn evaluator before any learned components are introduced.

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
