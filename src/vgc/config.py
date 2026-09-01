"""Repo-wide constants: the Showdown repo we drive, the format we target, and where
exported mod data lives. Everything else (tools/export_champions_data.py,
offline/run_matches.py, tests) should import these instead of hardcoding paths/ids.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Local Pokemon Showdown checkout this project drives (server + validator + data source).
# Not part of this repo -- cloned/built separately. See CLAUDE.md for setup.
SHOWDOWN_REPO = Path("/Users/edmundyu/code/projects/pokemon-showdown")

# "[Gen 9 Champions] VGC 2026 Reg M-B" -- doubles, bring-6-pick-4, level 50, Megas allowed.
# The format offers mutual-consent Open Team Sheets, but the bot rejects them and assumes
# no opponent sheet on the best-of-one ladder. Backed by the Showdown repo's "champions"
# mod (see config/formats.ts there), NOT vanilla gen9 -- Stat Points, item legality,
# abilities, and learnsets all diverge from vanilla gen9 VGC. Never assume vanilla data.
FORMAT_ID = "gen9championsvgc2026regmb"

# Local Showdown server poke-env connects to (`node pokemon-showdown start --no-security`).
LOCAL_SERVER_HOST = "localhost:8000"
LOCAL_SERVER_WS_URL = f"ws://{LOCAL_SERVER_HOST}/showdown/websocket"

# Exported champions-mod data (see tools/export_champions_data.py).
DATA_DIR = REPO_ROOT / "data" / "champions"

TEAMS_DIR = REPO_ROOT / "teams"
RUNS_DIR = REPO_ROOT / "runs"
