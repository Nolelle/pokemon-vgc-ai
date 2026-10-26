"""Data-health report for the Reg M-C training data (raw replay corpus + parsed decisions).

Answers the questions to settle before trusting a dataset: collection gaps and
duplicates, sampling bias (who is in it), label quality (are win/loss outcomes real and
consistent with the raw logs), how many per-decision action labels were actually
observed, how much of the state is hidden, and species coverage / meta drift.

Writes PNG figures, ``index.html`` (figures + pass/warn checklist) and ``summary.json``
under ``runs/viz/data_health/`` (gitignored). Scans are cached under ``cache/``.

    .venv/bin/python tools/data_health_report.py             # reuse cache if present
    .venv/bin/python tools/data_health_report.py --refresh   # re-scan replays + decisions
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import html
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MC = "gen9championsvgc2026regmc"
REPLAY_DIR = REPO_ROOT / "data" / "replays" / MC
OUT_DIR = REPO_ROOT / "runs" / "viz" / "data_health"
CACHE_DIR = OUT_DIR / "cache"
EXCLUDED_PLAYERS = {"laplacestheorems"}
MAX_TURN_BUCKET = 20

_RE = {
    "player": re.compile(r"^\|player\|(p[12])\|([^\n]*)$", re.M),
    "poke": re.compile(r"^\|poke\|(p[12])\|([^,|\n]+)", re.M),
    "turn": re.compile(r"^\|turn\|(\d+)", re.M),
    "win": re.compile(r"^\|win\|([^\n]*)$", re.M),
    "tie": re.compile(r"^\|tie\b", re.M),
    "forfeit": re.compile(r"^\|-message\|[^\n]*forfeited\.", re.M),
    "inactive": re.compile(r"^\|-message\|[^\n]*(lost due to inactivity|lost by being offline)", re.M),
    "showteam": re.compile(r"^\|showteam\|", re.M),
    "rating_change": re.compile(r"^\|raw\|(.+?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>", re.M),
}
def toid(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", html.unescape(s).lower())


def _pre_turn1_state(seg0: str) -> tuple[bool, bool]:
    """(weather/terrain active, any nonzero stat stage) at the end of the pre-turn-1 segment."""
    weather = None
    terrains: set[str] = set()
    stages: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in seg0.split("\n"):
        p = line.split("|")
        if len(p) < 3:
            continue
        tag, pos = p[1], p[2][:3]
        if tag == "-weather":
            weather = None if p[2] == "none" else p[2]
        elif tag == "-fieldstart":
            terrains.add(p[2])
        elif tag == "-fieldend":
            terrains.discard(p[2])
        elif tag in ("switch", "drag"):
            stages.pop(pos, None)
        elif tag in ("-boost", "-unboost") and len(p) > 4 and p[4].lstrip("-").isdigit():
            stages[pos][p[3]] += int(p[4]) * (1 if tag == "-boost" else -1)
        elif tag == "-setboost" and len(p) > 4 and p[4].lstrip("-").isdigit():
            stages[pos][p[3]] = int(p[4])
        elif tag == "-clearnegativeboost":
            stages[pos] = collections.Counter({k: v for k, v in stages[pos].items() if v > 0})
        elif tag == "-clearboost":
            stages.pop(pos, None)
        elif tag == "-clearallboost":
            stages.clear()
    return bool(weather or terrains), any(v for c in stages.values() for v in c.values())


# --------------------------------------------------------------------------- scans
def scan_replay(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    log = d.get("log") or ""
    names: dict[str, str] = {}
    ratings: dict[str, int] = {}
    for side, rest in _RE["player"].findall(log):
        parts = rest.split("|")
        if parts[0] and side not in names:
            names[side] = parts[0]
        if len(parts) > 2 and parts[2].isdigit() and side not in ratings:
            ratings[side] = int(parts[2])
    pre: dict[str, int] = {}
    post: dict[str, int] = {}
    for name, before, after in _RE["rating_change"].findall(log):
        side = next((s for s, nm in names.items() if toid(nm) == toid(name)), None)
        if side:
            pre[side], post[side] = int(before), int(after)
    pre_field, pre_boost = _pre_turn1_state(log.split("\n|turn|1\n")[0])
    preview: dict[str, list[str]] = {"p1": [], "p2": []}
    for side, sp in _RE["poke"].findall(log):
        preview[side].append(toid(sp))
    wins = _RE["win"].findall(log)
    winner = None
    if wins:
        w = toid(wins[-1])
        winner = next((s for s, n in names.items() if toid(n) == w), "?")
    if _RE["tie"].search(log):
        end = "tie"
    elif _RE["forfeit"].search(log):
        end = "forfeit"
    elif _RE["inactive"].search(log):
        end = "timer/offline"
    elif wins:
        end = "played out"
    else:
        end = "no result"
    return {
        "id": d.get("id"),
        "formatid": d.get("formatid"),
        "uploadtime": d.get("uploadtime"),
        "rating": d.get("rating"),
        "p1": toid(names.get("p1", "")),
        "p2": toid(names.get("p2", "")),
        "r1": ratings.get("p1"),
        "r2": ratings.get("p2"),
        "pre1": pre.get("p1"),
        "pre2": pre.get("p2"),
        "post1": post.get("p1"),
        "post2": post.get("p2"),
        "winner": winner,
        "end": end,
        "turns": max(map(int, _RE["turn"].findall(log)), default=0),
        "preview1": preview["p1"],
        "preview2": preview["p2"],
        "showteam": bool(_RE["showteam"].search(log)),
        "pre_t1_field": pre_field,
        "pre_t1_boost": pre_boost,
        "ots_rejected": "rejected open team sheets" in log,
        "log_md5": hashlib.md5(log.encode()).hexdigest(),
    }


def scan_replays() -> list[dict]:
    files = sorted(str(p) for p in REPLAY_DIR.glob("*.json"))
    with cf.ProcessPoolExecutor() as ex:
        return list(ex.map(scan_replay, files, chunksize=400))


def scan_decisions(path: Path) -> dict:
    kinds: collections.Counter = collections.Counter()
    schemas: collections.Counter = collections.Counter()
    per_replay: collections.Counter = collections.Counter()
    outcomes: collections.Counter = collections.Counter()
    replay_outcome: dict[str, dict[str, str]] = collections.defaultdict(dict)
    status_by_turn = collections.defaultdict(collections.Counter)
    action_kind: collections.Counter = collections.Counter()
    moves: collections.Counter = collections.Counter()
    targets: collections.Counter = collections.Counter()
    mega_observed = 0
    fog = {side: collections.defaultdict(lambda: [0, 0, 0, 0, 0]) for side in ("opp", "our")}
    on_team: collections.Counter = collections.Counter()
    brought: collections.Counter = collections.Counter()
    led: collections.Counter = collections.Counter()
    picked_len: collections.Counter = collections.Counter()
    bad = collections.Counter()
    bad_species: collections.Counter = collections.Counter()
    t1_field: set[str] = set()
    t1_boost: set[str] = set()
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        kind, rid = r["decision_kind"], r["replay_id"]
        kinds[kind] += 1
        schemas[str(r.get("schema"))] += 1
        per_replay[rid] += 1
        outcome = r.get("outcome")
        expected_won = {"win": True, "loss": False}.get(outcome)
        if r.get("won") != expected_won:
            bad["won field disagrees with outcome"] += 1
        state = r["state"]
        if kind == "teampreview":
            outcomes[outcome] += 1
            replay_outcome[rid][r["player"]] = outcome
            prev = state["our"]["preview"]
            picked = r["action"].get("picked") or []
            picked_len[len(picked)] += 1
            if not set(picked) <= set(prev):
                bad["picked species not on own preview"] += 1
            on_team.update(prev)
            brought.update(picked)
            led.update(r["action"].get("lead_order") or [])
            continue
        for side in ("our", "opp"):
            s = state[side]
            prev = set(s.get("preview_species") or [])
            if len(s["active"]) > 2:
                bad["more than 2 active Pokemon"] += 1
            for m in s["active"]:
                if m is None:
                    continue
                if not 0.0 <= m["hp_fraction"] <= 1.0:
                    bad["hp fraction outside 0-1"] += 1
                if any(abs(v) > 6 for v in m["boosts"].values()):
                    bad["stat stage outside -6..+6"] += 1
                if prev and m["species"] not in prev:
                    bad["active species not on team preview"] += 1
                    bad_species[m["species"]] += 1
            for b in s["bench"]:
                if prev and b["species_id"] not in prev:
                    bad["bench species not on team preview"] += 1
                    bad_species[b["species_id"]] += 1
        if kind != "turn":
            continue
        t = min(r["turn"], MAX_TURN_BUCKET)
        if r["turn"] == 1 and r["player"] == "p1":
            fld = state["field"]
            if fld.get("weather") or fld.get("terrain"):
                t1_field.add(rid)
            if any(m and m["boosts"] for side in ("our", "opp") for m in state[side]["active"]):
                t1_boost.add(rid)
        for slot in ("slot0", "slot1"):
            st = r["action_status"].get(slot, "missing")
            status_by_turn[t][st] += 1
            if st != "observed":
                continue
            a = r["action"][slot]
            action_kind[a["kind"]] += 1
            mega_observed += bool(a.get("mega"))
            if a["kind"] == "move":
                moves[a.get("move_id")] += 1
                targets[str(a.get("target_slot"))] += 1
        for side in ("opp", "our"):
            for m in state[side]["active"]:
                if m is None:
                    continue
                f = fog[side][t]
                f[0] += 1
                f[1] += m.get("item") is not None
                f[2] += m.get("ability") is not None
                f[3] += bool(m.get("revealed_moves"))
                f[4] += len(m.get("revealed_moves") or [])
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "kinds": kinds, "schemas": schemas, "per_replay": per_replay, "outcomes": outcomes,
        "replay_outcome": replay_outcome,
        "status_by_turn": {str(k): v for k, v in status_by_turn.items()},
        "action_kind": action_kind, "moves": moves, "targets": targets,
        "mega_observed": mega_observed,
        "fog": {side: {str(k): v for k, v in d.items()} for side, d in fog.items()},
        "on_team": on_team, "brought": brought, "led": led,
        "picked_len": {str(k): v for k, v in picked_len.items()},
        "bad": bad, "bad_species": bad_species,
        "t1_field": sorted(t1_field), "t1_boost": sorted(t1_boost),
    }


def latest_decisions_file() -> Path:
    files = sorted((REPO_ROOT / "data" / "bc").glob("decisions_regmc*.jsonl.gz"),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit("no data/bc/decisions_regmc*.jsonl.gz")
    return files[-1]


def load(refresh: bool, decisions_path: Path) -> tuple[list[dict], dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rep_cache = CACHE_DIR / "replays.jsonl"
    dec_cache = CACHE_DIR / f"decisions_{decisions_path.name.removesuffix('.jsonl.gz')}.json"
    if refresh or not rep_cache.exists():
        print(f"scanning {REPLAY_DIR.relative_to(REPO_ROOT)} ...", flush=True)
        rows = scan_replays()
        rep_cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rows = [json.loads(line) for line in rep_cache.open()]
    if refresh or not dec_cache.exists():
        print(f"scanning {decisions_path.relative_to(REPO_ROOT)} ...", flush=True)
        dec_cache.write_text(json.dumps(scan_decisions(decisions_path)))
    return rows, json.loads(dec_cache.read_text())


# --------------------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return math.nan, math.nan, math.nan
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, c - h, c + h


def day(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()


def pct(x: float) -> str:
    return f"{x:.1%}"


# --------------------------------------------------------------------------- report
def build(rows: list[dict], dec: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.titlesize": 10})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, str, str, bool | None]] = []  # name, value, rule, ok
    sections: list[tuple[str, str, list[str]]] = []  # title, png, notes
    summary: dict = {"decisions_source": dec["source"], "replays": len(rows)}
    n = len(rows)

    def save(fig, name: str, title: str, notes: list[str]) -> None:
        fig.tight_layout()
        fig.savefig(OUT_DIR / name, dpi=110)
        plt.close(fig)
        sections.append((title, name, notes))

    # ---- 1. collection ---------------------------------------------------------------
    days = collections.Counter(day(r["uploadtime"]) for r in rows if r["uploadtime"])
    d0, d1 = min(days), max(days)
    all_days = [d0 + dt.timedelta(i) for i in range((d1 - d0).days + 1)]
    zero_days = [d for d in all_days if days[d] == 0]
    prior7 = [days[d] for d in all_days[-8:-1]] or [days[d1]]
    md5_dups = n - len({r["log_md5"] for r in rows})
    id_dups = n - len({r["id"] for r in rows})
    wrong_fmt = sum(r["formatid"] != MC for r in rows)
    excluded = sum(r["p1"] in EXCLUDED_PLAYERS or r["p2"] in EXCLUDED_PLAYERS for r in rows)
    ratings = np.array([r["rating"] for r in rows if r["rating"] is not None])

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("1. Collection: how much data, when, and at what skill level", fontsize=13)
    ax[0, 0].bar(all_days, [days[d] for d in all_days], width=1.0, color="tab:blue")
    for d in zero_days:
        ax[0, 0].axvspan(d, d + dt.timedelta(1), color="red", alpha=0.3)
    ax[0, 0].set_title(f"Replays per day (UTC), {d0} to {d1}; red = days with zero replays ({len(zero_days)})")
    ax[0, 0].tick_params(axis="x", rotation=30)
    cum = np.cumsum([days[d] for d in all_days])
    ax[0, 1].plot(all_days, cum, color="tab:blue")
    ax[0, 1].set_title(f"Cumulative corpus size (total {n:,} replays)")
    ax[0, 1].tick_params(axis="x", rotation=30)
    ax[1, 0].hist(ratings, bins=np.arange(1000, ratings.max() + 25, 20), color="tab:purple")
    for thr in (1100, 1200):
        ax[1, 0].axvline(thr, color="k", ls="--", lw=1)
        ax[1, 0].text(thr, ax[1, 0].get_ylim()[1] * 0.9, f" {thr}", fontsize=8)
    ax[1, 0].set_title(f"Replay rating (download filter field): median {np.median(ratings):.0f}, "
                       f">=1300: {pct((ratings >= 1300).mean())}, >=1500: {pct((ratings >= 1500).mean())}")
    ax[1, 0].set_xlabel("replay rating")
    by_day = collections.defaultdict(list)
    for r in rows:
        if r["uploadtime"] and r["rating"] is not None:
            by_day[day(r["uploadtime"])].append(r["rating"])
    ds = [d for d in all_days if len(by_day[d]) >= 20]
    q = np.array([np.percentile(by_day[d], [25, 50, 75, 90]) for d in ds])
    ax[1, 1].fill_between(ds, q[:, 0], q[:, 2], alpha=0.3, label="25th-75th percentile")
    ax[1, 1].plot(ds, q[:, 1], label="median")
    ax[1, 1].plot(ds, q[:, 3], ls=":", label="90th percentile")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("Replay rating by day (days with >=20 replays) -- skill mix drift")
    ax[1, 1].tick_params(axis="x", rotation=30)
    save(fig, "01_collection.png", "1. Collection", [
        f"{n:,} replays from {d0} to {d1}; {len(zero_days)} day(s) with no replays"
        + (f" ({', '.join(map(str, zero_days[:8]))}{'...' if len(zero_days) > 8 else ''})" if zero_days else "") + ".",
        (f"The last day ({d1}, {days[d1]:,} replays vs a median of {int(np.median(prior7))} over the 7 days before) "
         "looks partial: the download probably stopped partway through it. "
          if days[d1] < 0.5 * np.median(prior7) else "") +
        f"Exact duplicate logs: {md5_dups}; duplicate ids: {id_dups}; wrong format id: {wrong_fmt}; games with our own bot account: {excluded}.",
        "Most games sit just above the download cutoff. The data mostly shows mid-ladder play, "
        f"not the 1700+ level the project is aiming for (only {pct((ratings >= 1500).mean())} of games are rated 1500+).",
    ])
    checks += [
        ("Days with zero replays in collection window", str(len(zero_days)), "0", len(zero_days) == 0),
        ("Exact duplicate replay logs", str(md5_dups), "0", md5_dups == 0),
        ("Replays with the wrong format id", str(wrong_fmt), "0", wrong_fmt == 0),
        ("Replays involving our own bot account", str(excluded), "0 (self-play must not leak in)", excluded == 0),
        ("Share of games rated 1500+", pct((ratings >= 1500).mean()), "info: target skill is 1700+", None),
    ]
    summary["collection"] = {"first_day": str(d0), "last_day": str(d1), "zero_days": len(zero_days),
                             "md5_dups": md5_dups, "rating_median": float(np.median(ratings))}

    # ---- 2. sampling bias ----------------------------------------------------------------
    appear = collections.Counter()
    first_seen: dict[str, dt.date] = {}
    for r in sorted(rows, key=lambda r: r["uploadtime"] or 0):
        for p in (r["p1"], r["p2"]):
            if p:
                appear[p] += 1
                if r["uploadtime"]:
                    first_seen.setdefault(p, day(r["uploadtime"]))
    counts = np.sort(np.array(list(appear.values())))
    total = counts.sum()
    lor_y = np.concatenate([[0], np.cumsum(counts) / total])
    lor_x = np.linspace(0, 1, len(lor_y))
    gini = 1 - 2 * np.trapezoid(lor_y, lor_x)
    top1 = counts[::-1][: max(1, len(counts) // 100)].sum() / total
    top10 = counts[::-1][: max(1, len(counts) // 10)].sum() / total
    top_player_share = counts[-1] / n

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("2. Who is in the data -- is it dominated by a few players?", fontsize=13)
    ax[0, 0].plot(lor_x, lor_y, label=f"actual (Gini {gini:.2f})")
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=1, label="every player equal")
    ax[0, 0].set_xlabel("share of players (fewest games -> most)")
    ax[0, 0].set_ylabel("share of player-games")
    ax[0, 0].legend()
    ax[0, 0].set_title(f"Concentration (Lorenz curve): top 1% of players = {pct(top1)} of player-games, top 10% = {pct(top10)}")
    p_teams = collections.defaultdict(collections.Counter)
    p_wins = collections.Counter()
    for r in rows:
        for side, p, prev in (("p1", r["p1"], r["preview1"]), ("p2", r["p2"], r["preview2"])):
            p_teams[p][tuple(sorted(prev))] += 1
            p_wins[p] += r["winner"] == side
    one_team = {p for p, c in appear.items()
                if c >= 200 and p_teams[p].most_common(1)[0][1] / c >= 0.9}
    one_team_games = sum(r["p1"] in one_team or r["p2"] in one_team for r in rows)
    top = appear.most_common(25)
    ax[0, 1].barh([f"{p} (win {p_wins[p] / c:.0%})" for p, c in top][::-1], [c / n for _, c in top][::-1],
                  color=["tab:red" if p in one_team else "tab:orange" for p, _ in top][::-1])
    ax[0, 1].set_title(f"Top-25 players: share of all {n:,} games they appear in\n"
                       "red = 200+ games with the same 6 species in >=90% of them; win = raw rate in kept games")
    ax[0, 1].tick_params(axis="y", labelsize=7)
    ax[1, 0].hist(counts, bins=np.logspace(0, math.log10(counts.max() + 1), 40))
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("games per player")
    ax[1, 0].set_title(f"Games per player ({len(appear):,} players; median {np.median(counts):.0f}, "
                       f"1-game players {pct((counts == 1).mean())})")
    distinct_by_day = collections.defaultdict(set)
    for r in rows:
        if r["uploadtime"]:
            distinct_by_day[day(r["uploadtime"])].update([r["p1"], r["p2"]])
    new_by_day = collections.Counter(first_seen.values())
    ax[1, 1].plot(all_days, [len(distinct_by_day[d]) for d in all_days], label="distinct players that day")
    ax[1, 1].plot(all_days, [new_by_day[d] for d in all_days], label="first-time players that day")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].set_title("Player turnover per day")
    ax[1, 1].tick_params(axis="x", rotation=30)
    save(fig, "02_players.png", "2. Sampling bias: players", [
        f"{len(appear):,} distinct players. The single most frequent player is in {pct(top_player_share)} of games; "
        f"the top 1% of players account for {pct(top1)} of player-games.",
        f"{len(one_team)} accounts ({', '.join(sorted(one_team, key=lambda p: -appear[p])[:6])}) play 200+ games, almost "
        f"always with the same 6 species (moves and items may still vary). Together they appear in "
        f"{pct(one_team_games / n)} of all games, so a model that copies replays mostly copies them. The win rates shown "
        "are raw rates among the kept games: they are not adjusted for opponent strength or the download-filter bias "
        "(figure 3). These accounts also inflate their species in the usage counts and 'meta drift' lines of figure 6.",
        "Why it matters: a model trained on this copies the habits of whoever plays most. When splitting into training "
        "and test sets, keep each player's games on one side. Otherwise the test score is inflated by recognising "
        "familiar players.",
    ])
    checks += [
        ("Games involving a 200+ game account that always brings the same 6 species", f"{one_team_games:,} ({one_team_games / n:.1%})",
         "<= 10% (consider down-weighting)", one_team_games <= 0.10 * n),
        ("Top 1% of players' share of player-games", pct(top1), "<= 15% (rule of thumb)", top1 <= 0.15),
        ("Most frequent single player's share of games", pct(top_player_share), "<= 2%", top_player_share <= 0.02),
    ]
    summary["players"] = {"distinct": len(appear), "gini": float(gini), "top1_share": float(top1),
                          "one_team_accounts": {p: {"games": appear[p], "win_rate": p_wins[p] / appear[p]}
                                                for p in sorted(one_team, key=lambda p: -appear[p])},
                          "one_team_game_share": one_team_games / n,
                          "top10_share": float(top10), "top_player_game_share": float(top_player_share)}

    # ---- 3. outcome labels ------------------------------------------------------------------
    ends = collections.Counter(r["end"] for r in rows)
    decided = [r for r in rows if r["winner"] in ("p1", "p2")]
    p1w = sum(r["winner"] == "p1" for r in decided)
    p1_rate, p1_lo, p1_hi = wilson(p1w, len(decided))
    # Pre-game ratings come from the end-of-game "X's rating: A -> B" messages (the `|player|` line
    # rating can be stale), so pre and post are one consistent source.
    rated = [{**r, "r1": r["pre1"], "r2": r["pre2"]} for r in decided
             if r["pre1"] is not None and r["pre2"] is not None and r["pre1"] != r["pre2"]]
    hi_wins = sum((r["r1"] > r["r2"]) == (r["winner"] == "p1") for r in rated)
    hi_rate, hi_lo, hi_hi = wilson(hi_wins, len(rated))
    early_ff = sum(r["end"] in ("forfeit", "timer/offline") and r["turns"] <= 2 for r in rows)
    # The download filter keys on the replay `rating` field. If that is a POST-game rating, inclusion
    # depends on who won near the cutoff (selection bias). Games where both pre-game ratings exceed
    # the cutoff by more than the largest single-game loss observed are far less exposed to it.
    with_post = [r for r in rated if r["rating"] is not None]
    field_is_min_post = sum(r["rating"] == min(r["post1"], r["post2"]) for r in with_post) / max(1, len(with_post))
    max_drop = max((max(r["r1"] - r["post1"], r["r2"] - r["post2"]) for r in with_post), default=0)
    safe_floor = 1200 + max_drop
    safe = [r for r in with_post if min(r["r1"], r["r2"]) >= safe_floor]
    near = [r for r in with_post if min(r["r1"], r["r2"]) < 1200]
    safe_rate, safe_lo, safe_hi = wilson(sum((r["r1"] > r["r2"]) == (r["winner"] == "p1") for r in safe), len(safe))
    near_rate = wilson(sum((r["r1"] > r["r2"]) == (r["winner"] == "p1") for r in near), len(near))[0]
    # parsed label vs raw log winner
    ro = dec["replay_outcome"]
    confusion = collections.Counter()
    for r in rows:
        raw = {"p1": "p1 won", "p2": "p2 won"}.get(r["winner"], "tie" if r["end"] == "tie" else "no winner")
        o = ro.get(r["id"])
        parsed = "not parsed" if o is None else {
            ("win", "loss"): "p1 won", ("loss", "win"): "p2 won", ("draw", "draw"): "tie",
        }.get((o.get("p1"), o.get("p2")), f"{o.get('p1')}/{o.get('p2')}")
        confusion[(raw, parsed)] += 1
    agree = sum(v for (a, b), v in confusion.items() if a == b)
    parsed_ids = set(dec["per_replay"])
    corpus_ids = {r["id"] for r in rows}
    unparsed = len(corpus_ids - parsed_ids)
    orphan = len(parsed_ids - corpus_ids)

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("3. Outcome labels -- are the win/loss labels real, balanced and consistent?", fontsize=13)
    order = ["played out", "forfeit", "timer/offline", "tie", "no result"]
    vals = [ends.get(k, 0) for k in order]
    ax[0, 0].bar(order, vals, color=["tab:green", "tab:orange", "tab:red", "tab:gray", "black"])
    for i, v in enumerate(vals):
        ax[0, 0].text(i, v, f"{v:,}\n{v / n:.1%}", ha="center", va="bottom", fontsize=8)
    ax[0, 0].set_title("How games ended")
    bins = np.arange(0, 31)
    stack = [[min(r["turns"], 30) for r in rows if r["end"] == k] for k in order[:3]]
    ax[0, 1].hist(stack, bins=bins, stacked=True, label=order[:3],
                  color=["tab:green", "tab:orange", "tab:red"])
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set_xlabel("turns (30 = 30+)")
    ax[0, 1].set_title(f"Game length by ending; forfeit/timeout by turn 2: {early_ff:,} ({early_ff / n:.1%})")
    edges = np.arange(-300, 325, 50)
    for subset, label, color, off in ((rated, f"all games (n={len(rated):,})", "tab:red", -6),
                                      (safe, f"both pre-game >= {safe_floor} (n={len(safe):,})", "tab:blue", 6)):
        diffs = np.array([np.clip(r["r1"] - r["r2"], -299, 299) for r in subset])
        p1won = np.array([r["winner"] == "p1" for r in subset])
        xs, ys, lo, hi = [], [], [], []
        for a, b in zip(edges, edges[1:]):
            m = (diffs >= a) & (diffs < b)
            if m.sum() >= 30:
                p, low, high = wilson(int(p1won[m].sum()), int(m.sum()))
                xs.append((a + b) / 2 + off)
                ys.append(p)
                lo.append(p - low)
                hi.append(high - p)
        ax[0, 2].errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=color, label=label, capsize=2)
    grid = np.linspace(-300, 300, 100)
    ax[0, 2].plot(grid, 1 / (1 + 10 ** (-grid / 400)), "k--", lw=1, label="Elo expectation")
    ax[0, 2].set_xlabel("p1 rating minus p2 rating (pre-game, clipped +-300)")
    ax[0, 2].set_ylabel("p1 win rate (95% CI)")
    ax[0, 2].set_ylim(0, 1)
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].set_title(f"Higher-rated player wins: {pct(hi_rate)} overall, {pct(safe_rate)} away from the cutoff,\n"
                       f"{pct(near_rate)} when the weaker player started below 1200 (download-filter bias)")
    by_end = []
    for k in order[:3]:
        g = [r for r in decided if r["end"] == k]
        by_end.append((k, *wilson(sum(r["winner"] == "p1" for r in g), len(g)), len(g)))
    by_end.append(("all decided", p1_rate, p1_lo, p1_hi, len(decided)))
    labels_ = [f"{k}\n(n={m:,})" for k, *_, m in by_end]
    ax[1, 0].bar(labels_, [b[1] for b in by_end],
                 yerr=[[b[1] - b[2] for b in by_end], [b[3] - b[1] for b in by_end]], capsize=4)
    ax[1, 0].axhline(0.5, color="k", ls=":")
    ax[1, 0].set_ylim(0.4, 0.6)
    ax[1, 0].set_title("Player-1 win rate by ending (should be ~50%: no side bias)")
    raws = ["p1 won", "p2 won", "tie", "no winner"]
    parseds = sorted({b for _, b in confusion}, key=lambda s: (s not in raws, s))
    mat = np.array([[confusion[(a, b)] for b in parseds] for a in raws])
    ax[1, 1].imshow(np.log1p(mat), cmap="Blues")
    for i in range(len(raws)):
        for j in range(len(parseds)):
            ax[1, 1].text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=8)
    ax[1, 1].set_xticks(range(len(parseds)), parseds, rotation=30, ha="right", fontsize=8)
    ax[1, 1].set_yticks(range(len(raws)), raws)
    ax[1, 1].set_xlabel("label in parsed decisions file")
    ax[1, 1].set_ylabel("winner in raw replay log")
    ax[1, 1].grid(False)
    ax[1, 1].set_title(f"Label cross-check: parsed vs raw agree on {agree / n:.2%}")
    oc = dec["outcomes"]
    ax[1, 2].bar(list(oc), list(oc.values()), color="tab:cyan")
    for i, v in enumerate(oc.values()):
        ax[1, 2].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax[1, 2].set_title("Outcome label per (game, player) in parsed decisions")
    save(fig, "03_labels.png", "3. Outcome labels", [
        f"{pct(ends['played out'] / n)} of games end with a winner and no forfeit/timer/offline message ('played out'); "
        f"{pct(ends['forfeit'] / n)} ended by forfeit and {pct(ends['timer/offline'] / n)} by the inactivity timer or a "
        f"disconnect. {early_ff:,} games ended by forfeit/timer by turn 2. Those carry a win/loss label that says almost "
        "nothing about play quality, so consider dropping them from value fits.",
        f"Selection bias from the download filter: the replay 'rating' field that the downloader filters on equals the "
        f"LOWER of the two players' ratings AFTER the game in {pct(field_is_min_post)} of games. Near the cutoff, whether a "
        "game was kept depends on who won. When the weaker player started below 1200, the favourite won just "
        f"{pct(near_rate)} of the kept games. Overall the higher-rated player wins only {pct(hi_rate)}, "
        "below the 50% you would expect.",
        f"Games where both players started at {safe_floor}+ (the 1200 cutoff plus {max_drop}, the largest single-game "
        f"rating loss seen in the corpus) are much less exposed to that filter. In them the favourite wins "
        f"{pct(safe_rate)} (95% CI {pct(safe_lo)}-{pct(safe_hi)}). Prefer that subset (n={len(safe):,}) for anything that "
        "learns from win/loss, and filter future downloads on PRE-game ratings.",
        f"Player 1 wins {pct(p1_rate)} (95% CI {pct(p1_lo)}-{pct(p1_hi)}), so there is no meaningful side bias.",
        f"Parsed outcome labels match the raw log winner for {agree:,}/{n:,} games. Corpus replays missing from the parsed "
        f"file: {unparsed}; parsed replays missing from the corpus: {orphan}.",
    ])
    checks += [
        ("Parsed-vs-raw outcome label agreement", f"{agree / n:.3%}", ">= 99.9%", agree / n >= 0.999),
        ("Corpus replays missing from parsed decisions", str(unparsed), "0 (re-parse if stale)", unparsed == 0),
        ("Games with no result / unresolved", str(ends.get("no result", 0)), "<= 0.5% of games",
         ends.get("no result", 0) <= 0.005 * n),
        ("Player-1 win rate (side bias)", f"{pct(p1_rate)} [{pct(p1_lo)}, {pct(p1_hi)}]", "CI overlaps 48-52%",
         p1_lo <= 0.52 and p1_hi >= 0.48),
        ("Higher-rated player win rate, all games", f"{pct(hi_rate)} [{pct(hi_lo)}, {pct(hi_hi)}]", "> 50%", hi_lo > 0.5),
        ("Download-filter field = min post-game rating", pct(field_is_min_post),
         "0% (filter should not depend on the result)", field_is_min_post < 0.01),
        (f"Higher-rated win rate, both pre-game >= {safe_floor}", f"{pct(safe_rate)} [{pct(safe_lo)}, {pct(safe_hi)}]",
         "> 50% (subset far from the cutoff)", safe_lo > 0.5),
        (f"Games far from the filter cutoff (both pre-game >= {safe_floor})", f"{len(safe):,} ({len(safe) / n:.1%})",
         "info", None),
        ("Forfeit/timeout by turn 2 (junk labels)", f"{early_ff:,} ({early_ff / n:.1%})", "info: filter for value fits", None),
    ]
    summary["labels"] = {"ends": dict(ends), "p1_win_rate": p1_rate, "higher_rated_win_rate": hi_rate,
                         "filter_field_is_min_post_rating": field_is_min_post, "safe_floor": safe_floor,
                         "safe_games": len(safe), "safe_higher_rated_win_rate": safe_rate,
                         "near_cutoff_higher_rated_win_rate": near_rate,
                         "label_agreement": agree / n, "unparsed": unparsed, "early_forfeits": early_ff}

    # ---- 4. action labels in parsed decisions --------------------------------------------
    kinds = dec["kinds"]
    rec_per = np.array(list(dec["per_replay"].values()))
    sbt = {int(k): v for k, v in dec["status_by_turn"].items()}
    turns_ = sorted(sbt)
    statuses = ["observed", "unknown", "no_action_required"]
    tot_status = collections.Counter()
    for v in sbt.values():
        tot_status.update(v)
    occupied = tot_status["observed"] + tot_status["unknown"]
    unknown_share = tot_status["unknown"] / max(1, occupied)
    moves = collections.Counter(dec["moves"])
    move_total = sum(moves.values())
    cum_share = np.cumsum(sorted(moves.values(), reverse=True)) / move_total
    k50, k90, k99 = (int(np.searchsorted(cum_share, q)) + 1 for q in (0.5, 0.9, 0.99))
    ak = collections.Counter(dec["action_kind"])
    observed = sum(ak.values())

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    fig.suptitle("4. Per-decision action labels -- how many are usable for imitation?", fontsize=13)
    ax[0, 0].bar(list(kinds), list(kinds.values()), color="tab:blue")
    for i, v in enumerate(kinds.values()):
        ax[0, 0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax[0, 0].set_title(f"Records by kind (total {sum(kinds.values()):,}, schema {dict(dec['schemas'])})")
    ax[0, 1].hist(rec_per, bins=range(0, int(np.percentile(rec_per, 99.5)) + 2))
    ax[0, 1].set_title(f"Records per game (median {np.median(rec_per):.0f}, min {rec_per.min()}, max {rec_per.max()})")
    bottom = np.zeros(len(turns_))
    colors = {"observed": "tab:green", "unknown": "tab:red", "no_action_required": "lightgray"}
    for st in statuses:
        share = np.array([sbt[t].get(st, 0) / max(1, sum(sbt[t].values())) for t in turns_])
        ax[0, 2].bar(turns_, share, bottom=bottom, label=st, color=colors[st])
        bottom += share
    ax[0, 2].set_xlabel(f"turn ({MAX_TURN_BUCKET} = {MAX_TURN_BUCKET}+)")
    ax[0, 2].legend(fontsize=8, loc="lower left")
    ax[0, 2].set_title(f"Slot action status by turn; unknown = {pct(unknown_share)} of occupied slots")
    ax2 = ax[0, 2].twinx()
    ax2.plot(turns_, [sum(sbt[t].values()) / 2 for t in turns_], "k.-", lw=1)
    ax2.set_ylabel("turn records (line)")
    ax2.grid(False)
    comp = {f"{k} ({v / observed:.1%})": v for k, v in ak.most_common()}
    comp[f"with Mega ({dec['mega_observed'] / observed:.1%})"] = dec["mega_observed"]
    ax[1, 0].barh(list(comp)[::-1], list(comp.values())[::-1], color="tab:olive")
    ax[1, 0].set_title(f"Observed slot actions by type (n={observed:,})")
    top = moves.most_common(25)
    ax[1, 1].barh([m for m, _ in top][::-1], [c / move_total for _, c in top][::-1], color="tab:brown")
    ax[1, 1].tick_params(axis="y", labelsize=7)
    ax[1, 1].set_title(f"Top-25 move labels (share of {move_total:,} observed moves)")
    ax[1, 2].plot(np.arange(1, len(cum_share) + 1), cum_share)
    for k, q in ((k50, 0.5), (k90, 0.9), (k99, 0.99)):
        ax[1, 2].axhline(q, color="gray", ls=":", lw=1)
        ax[1, 2].annotate(f"{k} moves -> {q:.0%}", (k, q), fontsize=8, xytext=(5, -12), textcoords="offset points")
    ax[1, 2].set_xscale("log")
    ax[1, 2].set_xlabel("number of distinct moves (most common first)")
    ax[1, 2].set_title(f"Label long tail: {len(moves)} distinct moves")
    save(fig, "04_action_labels.png", "4. Action labels", [
        f"{pct(unknown_share)} of occupied Pokemon slots on a turn have no action recovered from the log (for example "
        "the Pokemon fainted or was stopped before it could move). These must be masked out of imitation training (behaviour "
        "cloning: training a model to copy recorded human choices), not treated as a 'do nothing' choice.",
        f"Move labels are very uneven (class imbalance): {k50} moves cover half of all labels and {k90} cover 90%. "
        f"The other {len(moves) - k90} moves have few examples each.",
    ])
    checks += [
        ("Unknown action share of occupied slots", pct(unknown_share), "info: mask these in training", None),
    ]
    summary["actions"] = {"kinds": kinds, "unknown_share": unknown_share, "moves_distinct": len(moves),
                          "moves_for_50_90_99": [k50, k90, k99]}

    # ---- 5. hidden information ------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle("5. Hidden information -- how much of each Pokemon is known at decision time?", fontsize=13)
    f = {int(k): v for k, v in dec["fog"]["opp"].items()}
    ts = sorted(f)
    for idx, name, color in ((1, "item known", "tab:blue"), (2, "ability known", "tab:orange"),
                             (3, ">=1 move seen", "tab:green")):
        ax[0].plot(ts, [f[t][idx] / f[t][0] for t in ts], marker=".", color=color, label=name)
    ax[1].plot(ts, [f[t][4] / f[t][0] for t in ts], marker=".")
    fog = {t: {"item": f[t][1] / f[t][0], "ability": f[t][2] / f[t][0], "moves": f[t][4] / f[t][0]}
           for t in (1, 5, 10) if t in f}
    own = {int(k): v for k, v in dec["fog"]["our"].items()}
    own_item_t5 = own[5][1] / own[5][0] if 5 in own else math.nan
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel(f"turn ({MAX_TURN_BUCKET} = {MAX_TURN_BUCKET}+)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Opponent's active Pokemon: share with each fact revealed (start of turn)")
    ax[1].axhline(4, color="k", ls=":", lw=1)
    ax[1].set_ylim(0, 4.2)
    ax[1].set_xlabel(f"turn ({MAX_TURN_BUCKET} = {MAX_TURN_BUCKET}+)")
    ax[1].set_title("Opponent's active Pokemon: mean revealed moves (max 4)")
    st = sum(r["showteam"] for r in rows)
    rej = sum(r["ots_rejected"] for r in rows)
    ax[2].bar(["full team sheet\nshown", "team sheet\nrejected", "neither logged"],
              [st / n, rej / n, 1 - (st + rej) / n], color=["tab:green", "tab:red", "lightgray"])
    ax[2].set_ylim(0, 1)
    ax[2].set_title(f"Open Team Sheets in replays (sheet shown in {pct(st / n)} of games)")
    save(fig, "05_hidden_info.png", "5. Hidden information", [
        f"At the start of turn 5 an opponent's item is known {pct(fog[5]['item'])} of the time and its ability "
        f"{pct(fog[5]['ability'])}. By turn 10 only ~{fog[10]['moves']:.1f} of 4 moves have been seen.",
        f"The acting player's OWN items are just as hidden in these records (item known at turn 5: {pct(own_item_t5)}), "
        "because a replay shows only public information. Live, the bot knows its own full team. A model trained on these "
        "states sees different inputs from the ones it gets in real games (a train/serve mismatch); fill in own-side "
        "sets, or train with that information masked on purpose.",
        f"Full team sheets appear in only {pct(st / n)} of games, so true sets cannot be read straight from the corpus.",
    ])
    summary["hidden_info"] = {"opp_by_turn": fog, "own_item_known_t5": own_item_t5, "showteam_share": st / n}
    t1f, t1b = set(dec["t1_field"]), set(dec["t1_boost"])
    raw_f = {r["id"] for r in rows if r["pre_t1_field"]}
    raw_b = {r["id"] for r in rows if r["pre_t1_boost"]}
    miss_f, miss_b = len(raw_f - t1f), len(raw_b - t1b)
    checks += [
        ("Lead weather/terrain set before turn 1 but absent from the turn-1 state",
         f"{miss_f:,} of {len(raw_f):,} games ({miss_f / max(1, len(raw_f)):.0%})",
         "0", miss_f == 0),
        ("Lead stat changes (e.g. Intimidate) before turn 1 but absent from the turn-1 state",
         f"{miss_b:,} of {len(raw_b):,} games ({miss_b / max(1, len(raw_b)):.0%})",
         "0", miss_b == 0),
    ]
    summary["pre_turn1"] = {"field_raw": len(raw_f), "field_missing": miss_f,
                            "boost_raw": len(raw_b), "boost_missing": miss_b}

    # ---- 6. species coverage & drift -------------------------------------------------------
    team_slots = [(r, r["preview1"]) for r in rows] + [(r, r["preview2"]) for r in rows]
    usage = collections.Counter()
    for _, p in team_slots:
        usage.update(set(p))
    n_slots = len(team_slots)
    teams = collections.Counter(tuple(sorted(p)) for _, p in team_slots if p)
    repeat5 = sum(c for c in teams.values() if c >= 5) / n_slots
    rare = sum(1 for c in usage.values() if c < 100)
    on_team = collections.Counter(dec["on_team"])
    brought = collections.Counter(dec["brought"])
    bad_preview_len = sum(len(p) != 6 for _, p in team_slots)

    fig, ax = plt.subplots(2, 2, figsize=(17, 12))
    fig.suptitle("6. Species coverage and meta drift", fontsize=13)
    ranked = usage.most_common()
    ax[0, 0].plot(range(1, len(ranked) + 1), [c / n_slots for _, c in ranked], marker=".")
    ax[0, 0].set_yscale("log")
    ax[0, 0].axhline(100 / n_slots, color="r", ls="--", lw=1, label="100 teams")
    for i, (s, c) in enumerate(ranked[:12]):
        ax[0, 0].annotate(s, (i + 1, c / n_slots), fontsize=7, rotation=30)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_xlabel("species rank")
    ax[0, 0].set_ylabel("share of teams carrying it")
    ax[0, 0].set_title(f"Species usage (rank-frequency): {len(usage)} species seen, {rare} on fewer than 100 teams")
    week = {}
    for r in rows:
        if r["uploadtime"]:
            d = day(r["uploadtime"])
            week[r["id"]] = d - dt.timedelta(d.weekday())
    weeks = sorted(set(week.values()))
    w_slots = collections.Counter()
    w_use = collections.Counter()
    for r, p in team_slots:
        w = week.get(r["id"])
        if w is None:
            continue
        w_slots[w] += 1
        for s in set(p):
            w_use[(w, s)] += 1
    weeks = [w for w in weeks if w_slots[w] >= 500]
    for s, _ in ranked[:10]:
        ax[0, 1].plot(weeks, [w_use[(w, s)] / w_slots[w] for w in weeks], marker="o", label=s)
    ax[0, 1].legend(fontsize=7, ncol=2)
    ax[0, 1].set_title("Weekly usage of the top-10 species (weeks with >=500 teams)")
    ax[0, 1].tick_params(axis="x", rotation=30)
    xs = [s for s, c in on_team.items() if c >= 300]
    x = np.array([on_team[s] / sum(dec["picked_len"].values()) for s in xs])
    y = np.array([brought[s] / on_team[s] for s in xs])
    seen_rate_all = sum(brought.values()) / sum(on_team.values())
    ax[1, 0].scatter(x, y, s=12, alpha=0.6)
    for i in np.argsort(-x)[:20]:
        ax[1, 0].annotate(xs[i], (x[i], y[i]), fontsize=7)
    ax[1, 0].axhline(seen_rate_all, color="k", ls=":", lw=1, label=f"all species ({seen_rate_all:.0%})")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("share of teams carrying it")
    ax[1, 0].set_ylabel("seen in battle when on team")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("Team preview: how often each species is SEEN in battle when on the team (>=300 teams)\n"
                       "undercounts picks: a chosen Pokemon that never switched in is not recorded")
    tc = sorted(teams.values(), reverse=True)
    ax[1, 1].loglog(range(1, len(tc) + 1), tc)
    ax[1, 1].set_xlabel("team rank (exact same 6 species)")
    ax[1, 1].set_ylabel("times seen")
    ax[1, 1].set_title(f"Team repetition: {len(teams):,} distinct teams; {pct(repeat5)} of teams are a 6-species "
                       "set seen 5+ times")
    save(fig, "06_species.png", "6. Species coverage and drift", [
        f"{len(usage)} species appear; {rare} of them are on fewer than 100 teams. That is thin coverage: expect "
        "unreliable set guesses and play for those species (not measured here).",
        "Weekly lines show whether the metagame (the mix of popular Pokemon) is still shifting. If they slope, give "
        "old data less weight or split train/test by date rather than at random, so the test set reflects the current meta.",
        f"{pct(repeat5)} of teams reuse a 6-species set seen 5+ times, and many different players share the same set. "
        "A random split puts the same roster in both training and test data (leakage, so test scores look better than "
        "they are). To test on unseen teams, group the split by the 6-species set. To test on future games, split by date.",
    ])
    picked_bad = sum(v for k, v in dec["picked_len"].items() if k != "4")
    checks += [
        ("Team previews without exactly 6 species", str(bad_preview_len), "0", bad_preview_len == 0),
        ("Team-preview picks with != 4 species", f"{picked_bad:,} ({picked_bad / max(1, sum(dec['picked_len'].values())):.1%})",
         "0 (picks are derived from switch-ins, so unused bench mons go missing)", picked_bad == 0),
        ("Species on < 100 teams (thin coverage)", f"{rare} of {len(usage)}", "info", None),
    ]
    culprits = ", ".join(f"{s} x{c:,}" for s, c in collections.Counter(dec.get("bad_species", {})).most_common(4))
    for name, v in sorted(dec["bad"].items()):
        val = f"{v:,}" + (f" ({culprits})" if "not on team preview" in name and culprits else "")
        checks.append((f"Parsed record invariant: {name}", val, "0", v == 0))
    for name in ("hp fraction outside 0-1", "stat stage outside -6..+6", "active species not on team preview",
                 "bench species not on team preview", "more than 2 active Pokemon",
                 "won field disagrees with outcome", "picked species not on own preview"):
        if name not in dec["bad"]:
            checks.append((f"Parsed record invariant: {name}", "0", "0", True))
    summary["species"] = {"distinct": len(usage), "under_100_teams": rare, "distinct_teams": len(teams),
                          "repeat5_share": repeat5}
    summary["invariant_violations"] = dec["bad"]

    write_html(checks, sections, summary)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nreport: {OUT_DIR / 'index.html'}")


def write_html(checks, sections, summary) -> None:
    status = {True: ("PASS", "#1a7f37"), False: ("WARN", "#bf8700"), None: ("INFO", "#57606a")}
    rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(val)}</td><td>{html.escape(rule)}</td>"
        f"<td style='color:{status[ok][1]};font-weight:600'>{status[ok][0]}</td></tr>"
        for name, val, rule, ok in checks
    )
    body = "".join(
        f"<h2>{html.escape(title)}</h2><ul>{''.join(f'<li>{html.escape(n)}</li>' for n in notes)}</ul>"
        f"<img src='{png}' alt='{html.escape(title)}'>"
        for title, png, notes in sections
    )
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    (OUT_DIR / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Reg M-C data health</title><style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;color:#1f2328}}
table{{border-collapse:collapse;margin:12px 0 24px}}td,th{{border:1px solid #d0d7de;padding:5px 10px;font-size:14px;text-align:left}}
th{{background:#f6f8fa}}img{{max-width:100%;border:1px solid #d0d7de;margin-bottom:28px}}li{{margin:4px 0}}
</style></head><body>
<h1>Reg M-C training data: health report</h1>
<p>Raw corpus: <code>data/replays/{MC}/</code> ({summary['replays']:,} replays) &middot;
Parsed decisions: <code>{html.escape(summary['decisions_source'])}</code> &middot; generated {generated}</p>
<h2>Checklist</h2><table><tr><th>Check</th><th>Value</th><th>Expected</th><th>Status</th></tr>{rows}</table>
{body}</body></html>""")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-scan replays and decisions")
    ap.add_argument("--decisions", type=Path, default=None,
                    help="parsed decisions .jsonl.gz (default: newest data/bc/decisions_regmc*.jsonl.gz)")
    args = ap.parse_args()
    path = (args.decisions or latest_decisions_file()).resolve()
    rows, dec = load(args.refresh, path)
    build(rows, dec)


if __name__ == "__main__":
    main()
