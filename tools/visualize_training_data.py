"""Holistic visual summary of every training-data source in this repo.

Produces PNG figures under ``runs/viz/`` (gitignored) covering:

1. corpus overview -- replay counts/ratings/dates per format, teacher-decision counts
   per shard, BC replay-decision counts;
2. teacher-demonstration structure -- turns, legal-action counts, decisions per battle,
   request kinds, mechanics snapshot sizes;
3. teacher behaviour -- teacher vs myopic rank, searched-set size, top-1/top-2 search
   score margin, action composition (Protect/switch/mega/spread), top moves;
4. team coverage -- own-team x opponent-archetype matrix and per-team decision counts;
5. species coverage -- species seen in the M-C teacher data vs M-C public replays vs the
   (M-B) set-prior file;
6. ladder outcomes -- rating trajectory per format/policy and win rate by opponent rating.

The demonstration ``.pt`` shards are ~850 MB each, so ``extract`` loads ONE shard per
subprocess and writes a slim JSONL cache (one row per decision, big arrays dropped).
``plot`` reads only the cache.

    .venv/bin/python tools/visualize_training_data.py extract   # slow, once per shard
    .venv/bin/python tools/visualize_training_data.py plot
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = REPO_ROOT / "runs"
CACHE_DIR = RUNS / "viz" / "cache"
OUT_DIR = RUNS / "viz"
SHARD_GLOB = "collect_mc_s*/demonstrations.pt"
MB = "gen9championsvgc2026regmb"
MC = "gen9championsvgc2026regmc"


# --------------------------------------------------------------------------- extract
def _extract_shard(pt_path: Path, out_path: Path) -> None:
    import numpy as np
    import torch

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    shard = pt_path.parent.name
    with out_path.open("w") as fh:
        for s in data["samples"]:
            scores = np.asarray(s.search_scores, dtype=np.float64)
            searched = np.asarray(s.searched_mask, dtype=bool)
            ranks = np.asarray(s.candidate_myopic_ranks)
            ti = int(s.teacher_action_index)
            searched_scores = np.sort(scores[searched])[::-1] if searched.any() else np.array([])
            margin = (
                float(searched_scores[0] - searched_scores[1]) if len(searched_scores) > 1 else None
            )
            desc = str(s.teacher_action_description)
            row = {
                "shard": shard,
                "battle_id": s.battle_id,
                "turn": int(s.turn),
                "decision_index": int(s.decision_index),
                "request_kind": s.request_kind,
                "legal": int(s.legal_action_count),
                "n_searched": int(searched.sum()),
                "teacher_index": ti,
                "teacher_myopic_rank": int(ranks[ti]),
                "teacher_searched": bool(searched[ti]),
                "top_margin": margin,
                "team_id": s.team_id,
                "opp_team_id": s.opponent_team_id,
                "mech_tokens": int(len(s.mechanics.tokens)) if s.mechanics is not None else None,
                "teacher_desc": desc,
                "has_protect": "protect" in desc,
                "has_switch": "switch->" in desc,
                "has_mega": "-mega" in desc,
                "has_pass": "pass" in desc,
            }
            fh.write(json.dumps(row) + "\n")


def cmd_extract(args: argparse.Namespace) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(RUNS.glob(SHARD_GLOB))
    if not shards:
        sys.exit(f"no shards matched runs/{SHARD_GLOB}")
    for pt in shards:
        out = CACHE_DIR / f"{pt.parent.name}.jsonl"
        if out.exists() and not args.force:
            print(f"cached  {out.relative_to(REPO_ROOT)}")
            continue
        print(f"extract {pt.relative_to(REPO_ROOT)} ...", flush=True)
        subprocess.run(
            [sys.executable, __file__, "_extract_one", str(pt), str(out)], check=True
        )


# --------------------------------------------------------------------------- loaders
def load_decisions() -> list[dict]:
    rows: list[dict] = []
    for f in sorted(CACHE_DIR.glob("collect_mc_s*.jsonl")):
        rows.extend(json.loads(line) for line in f.open())
    if not rows:
        sys.exit("no cache -- run `extract` first")
    return rows


def load_replays(format_id: str) -> list[dict]:
    out = []
    for f in glob.glob(str(REPO_ROOT / "data" / "replays" / format_id / "*.json")):
        d = json.load(open(f))
        species = re.findall(r"\|poke\|p[12]\|([^,|]+)", d.get("log", ""))
        out.append(
            {
                "rating": d.get("rating"),
                "uploadtime": d.get("uploadtime"),
                "species": [_species_id(x) for x in species],
            }
        )
    return out


def _species_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_ladder() -> list[dict]:
    p = RUNS / "ladder.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.open()]
    return [r for r in rows if "lost" in r and r.get("rating")]


def archetype(team_id: str) -> str:
    return team_id.split("/")[0]


# --------------------------------------------------------------------------- plots
def cmd_plot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dec = load_decisions()
    rep_mb, rep_mc = load_replays(MB), load_replays(MC)
    ladder = load_ladder()
    priors = json.load(open(REPO_ROOT / "data" / "usage" / "set_priors.json"))
    species_legal = json.load(open(REPO_ROOT / "data" / "champions" / "species.json"))
    bc_rows = sum(1 for _ in open(REPO_ROOT / "data" / "bc" / "decisions.jsonl"))
    summary: dict = {}

    # ---- 1. corpus overview -------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("1. Corpus overview -- what data exists, and for which regulation")
    ratings_mb = [r["rating"] for r in rep_mb if r["rating"]]
    ratings_mc = [r["rating"] for r in rep_mc if r["rating"]]
    ax[0, 0].hist([ratings_mb, ratings_mc], bins=25, label=[f"M-B n={len(rep_mb)}", f"M-C n={len(rep_mc)}"])
    ax[0, 0].set_title("Public replay ratings")
    ax[0, 0].set_xlabel("player rating")
    ax[0, 0].legend()
    day_counts: collections.Counter = collections.Counter()
    for rows, label in ((rep_mb, "M-B"), (rep_mc, "M-C")):
        for r in rows:
            if r["uploadtime"]:
                day_counts[(label, dt.datetime.fromtimestamp(r["uploadtime"]).date())] += 1
    keys = sorted(day_counts, key=lambda k: k[1])
    ax[0, 1].bar([f"{d}\n{lab}" for lab, d in keys], [day_counts[k] for k in keys],
                 color=["tab:blue" if k[0] == "M-B" else "tab:orange" for k in keys])
    ax[0, 1].set_title("Replays by upload date -- each corpus is a one-off snapshot, not a stream")
    ax[0, 1].tick_params(axis="x", labelsize=8)
    shard_counts = collections.Counter(r["shard"] for r in dec)
    ax[1, 0].bar(sorted(shard_counts), [shard_counts[k] for k in sorted(shard_counts)])
    ax[1, 0].set_title(f"M-C teacher decisions per shard (total {len(dec):,})")
    ax[1, 0].tick_params(axis="x", rotation=30)
    sources = {
        "M-B replays": len(rep_mb), "M-C replays": len(rep_mc),
        "BC replay decisions (M-B)": bc_rows, "M-C teacher decisions": len(dec),
        "M-C teacher battles": len({(r["shard"], r["battle_id"]) for r in dec}),
    }
    ax[1, 1].barh(list(sources), list(sources.values()), log=True)
    for i, v in enumerate(sources.values()):
        ax[1, 1].text(v, i, f" {v:,}", va="center")
    ax[1, 1].set_title("All sources (log scale)")
    ax[1, 1].set_xlabel("count")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_corpus_overview.png", dpi=120)
    plt.close(fig)
    summary["sources"] = sources

    # ---- 2. demonstration structure -------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("2. Teacher-demonstration structure (M-C, all shards)")
    turns = [r["turn"] for r in dec]
    ax[0, 0].hist(turns, bins=range(1, max(turns) + 2), align="left")
    ax[0, 0].set_title("Decisions by turn")
    ax[0, 0].set_xlabel("turn")
    legal = np.array([r["legal"] for r in dec])
    ax[0, 1].hist(legal, bins=60)
    ax[0, 1].set_yscale("log")
    ax[0, 1].axvline(10, color="r", ls="--", label="K=10 shortlist")
    ax[0, 1].set_title(f"Legal joint actions per decision (median {np.median(legal):.0f}, ≤10: {(legal<=10).mean():.0%})")
    ax[0, 1].legend()
    per_battle = collections.Counter((r["shard"], r["battle_id"]) for r in dec)
    ax[0, 2].hist(list(per_battle.values()), bins=range(1, 25))
    ax[0, 2].set_title(f"Decisions per battle (mean {np.mean(list(per_battle.values())):.1f})")
    kinds = collections.Counter(r["request_kind"] for r in dec)
    ax[1, 0].pie(kinds.values(), labels=[f"{k} ({v:,})" for k, v in kinds.items()], autopct="%1.0f%%")
    ax[1, 0].set_title("Request kind")
    mech = [r["mech_tokens"] for r in dec if r["mech_tokens"]]
    ax[1, 1].hist(mech, bins=50)
    ax[1, 1].set_title(f"Mechanics snapshot size (bytes) median {np.median(mech):,.0f}")
    by_turn = collections.defaultdict(list)
    for r in dec:
        by_turn[r["turn"]].append(r["legal"])
    xs = sorted(by_turn)
    ax[1, 2].plot(xs, [np.median(by_turn[t]) for t in xs], marker="o")
    ax[1, 2].set_title("Median legal actions by turn")
    ax[1, 2].set_xlabel("turn")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_demo_structure.png", dpi=120)
    plt.close(fig)

    # ---- 3. teacher behaviour ---------------------------------------------------------
    moves_only = [r for r in dec if r["request_kind"] == "move"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("3. What the teacher (exact Showdown search) actually chose")
    tr = np.array([r["teacher_myopic_rank"] for r in moves_only])
    ax[0, 0].hist(np.clip(tr, 0, 30), bins=range(0, 32), align="left")
    ax[0, 0].set_title(f"Teacher action's MYOPIC rank\nrank0 = {(tr==0).mean():.0%}, in top-10 = {(tr<10).mean():.0%}")
    ax[0, 0].set_xlabel("myopic rank (30 = 30+)")
    ns = [r["n_searched"] for r in moves_only]
    ax[0, 1].hist(ns, bins=range(0, max(ns) + 2), align="left")
    ax[0, 1].set_title(f"Candidates actually searched per decision\nteacher pick was searched: {np.mean([r['teacher_searched'] for r in moves_only]):.1%}")
    margins = np.array([r["top_margin"] for r in moves_only if r["top_margin"] is not None])
    ax[0, 2].hist(np.clip(margins, 0, 60), bins=60)
    ax[0, 2].set_title(f"Search-score gap: best vs 2nd searched\nmedian {np.median(margins):.1f}; <2pts: {(margins<2).mean():.0%}; 60+: {(margins>=60).mean():.0%}")
    ax[0, 2].set_xlabel("score points (60 = 60+)")
    comp = {
        "contains Protect": np.mean([r["has_protect"] for r in moves_only]),
        "contains switch": np.mean([r["has_switch"] for r in moves_only]),
        "mega evolves": np.mean([r["has_mega"] for r in moves_only]),
        "contains pass": np.mean([r["has_pass"] for r in moves_only]),
    }
    ax[1, 0].barh(list(comp), list(comp.values()))
    for i, v in enumerate(comp.values()):
        ax[1, 0].text(v, i, f" {v:.1%}", va="center")
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_title("Teacher action composition (move requests)")
    move_counter: collections.Counter = collections.Counter()
    for r in moves_only:
        for part in r["teacher_desc"].split(" / "):
            tok = part.split("@")[0].replace("-mega", "").strip()
            if tok and not tok.startswith("switch") and tok != "/choose pass":
                move_counter[tok] += 1
    top = move_counter.most_common(20)
    ax[1, 1].barh([t for t, _ in top][::-1], [c for _, c in top][::-1])
    ax[1, 1].set_title("Top-20 moves in teacher actions")
    by_turn_rank = collections.defaultdict(list)
    for r in moves_only:
        by_turn_rank[r["turn"]].append(r["teacher_myopic_rank"] == 0)
    xs = sorted(by_turn_rank)
    ax[1, 2].plot(xs, [np.mean(by_turn_rank[t]) for t in xs], marker="o")
    ax[1, 2].set_ylim(0, 1)
    ax[1, 2].set_title("Share of turns where search agreed with the myopic #1")
    ax[1, 2].set_xlabel("turn")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_teacher_behaviour.png", dpi=120)
    plt.close(fig)
    summary["teacher"] = {
        "myopic_rank0_share": float((tr == 0).mean()),
        "myopic_top10_share": float((tr < 10).mean()),
        "median_margin": float(np.median(margins)),
        "composition": {k: float(v) for k, v in comp.items()},
    }

    # ---- 4. team coverage -------------------------------------------------------------
    own = sorted({r["team_id"] for r in dec})
    own_arch = sorted({archetype(r["team_id"]) for r in dec})
    opp_arch = sorted({archetype(r["opp_team_id"]) for r in dec})
    mat = np.zeros((len(own_arch), len(opp_arch)))
    for r in dec:
        mat[own_arch.index(archetype(r["team_id"])), opp_arch.index(archetype(r["opp_team_id"]))] += 1
    own_counts = collections.Counter(r["team_id"] for r in dec)
    arch_team_counts = collections.Counter(archetype(t) for t in own)
    fig, ax = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    fig.suptitle(f"4. Team coverage in the M-C teacher data ({len(own)} own teams, {len(own_arch)} archetypes)")
    ax[0].imshow(mat, cmap="viridis")
    ax[0].set_xticks(range(len(opp_arch)))
    ax[0].set_xticklabels(opp_arch, rotation=45, ha="right")
    ax[0].set_yticks(range(len(own_arch)))
    ax[0].set_yticklabels(own_arch)
    for i in range(len(own_arch)):
        for j in range(len(opp_arch)):
            ax[0].text(j, i, f"{int(mat[i, j]):,}", ha="center", va="center", color="w", fontsize=8)
    ax[0].set_title("decisions: our archetype (rows) x opponent archetype (cols)")
    ax[0].set_xlabel("opponent")
    ax[0].set_ylabel("us")
    ax[1].barh(list(arch_team_counts), list(arch_team_counts.values()))
    ax[1].set_title("Distinct own teams per archetype\n(all but mc_ladder are generated variants of one core)")
    ax[2].hist(list(own_counts.values()), bins=25)
    ax[2].set_title(f"Decisions per own team (min {min(own_counts.values())}, max {max(own_counts.values())})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_team_coverage.png", dpi=120)
    plt.close(fig)
    summary["teams"] = {"own_teams": len(own), "opp_archetypes": len(opp_arch),
                        "opp_teams": len({r["opp_team_id"] for r in dec})}

    # ---- 5. species coverage ------------------------------------------------------------
    legal_ids = {sid for sid, row in species_legal.items() if row.get("isNonstandard") is None}
    prior_ids = set(priors["species"])
    mc_replay_species: collections.Counter = collections.Counter()
    for r in rep_mc:
        mc_replay_species.update(set(r["species"]))
    # teacher species come from the packed team files referenced by team ids
    teacher_species: collections.Counter = collections.Counter()
    for team_id, n in own_counts.items():
        for sid in _team_species(team_id):
            teacher_species[sid] += n
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("5. Species coverage: M-C ladder usage vs our teacher data vs the (M-B) opponent priors")
    top_mc = mc_replay_species.most_common(40)
    names = [s for s, _ in top_mc][::-1]
    ax[0].barh(names, [mc_replay_species[s] / max(1, len(rep_mc)) for s in names],
               color=["tab:green" if s in prior_ids else "tab:red" for s in names])
    ax[0].set_title("Top-40 species in M-C public replays (share of games)\nred = NO entry in set_priors.json (opponent moves unknown to the bot)")
    covered = [s for s in names if s in teacher_species]
    ax[1].barh(names, [teacher_species.get(s, 0) for s in names],
               color=["tab:blue" if s in teacher_species else "lightgray" for s in names])
    ax[1].set_title(f"Same species: decisions where OUR team carried it in teacher data\ngray = never on our side ({len(names)-len(covered)}/40 missing)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_species_coverage.png", dpi=120)
    plt.close(fig)
    top40 = [s for s, _ in top_mc]
    summary["species"] = {
        "mc_replay_distinct": len(mc_replay_species),
        "top40_without_prior": [s for s in top40 if s not in prior_ids],
        "top40_not_in_teacher_own_teams": [s for s in top40 if s not in teacher_species],
        "legal_species_with_prior": len(legal_ids & prior_ids), "legal_species": len(legal_ids),
    }

    # ---- 6. ladder ------------------------------------------------------------------------
    if ladder:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("6. Public ladder outcomes (runs/ladder.jsonl)")
        for fmt, mark in ((MB, "M-B"), (MC, "M-C")):
            rows = [r for r in ladder if r.get("format") == fmt]
            if rows:
                ax[0].plot(range(len(rows)), [r["rating"] for r in rows], label=f"{mark} n={len(rows)} win={np.mean([not r['lost'] for r in rows]):.0%}")
        ax[0].axhline(1700, color="r", ls="--", label="goal 1700")
        ax[0].legend()
        ax[0].set_title("Our rating after each game")
        ax[0].set_xlabel("game # within format")
        bins = [0, 1050, 1100, 1150, 1200, 1300, 3000]
        for fmt, mark in ((MB, "M-B"), (MC, "M-C")):
            rows = [r for r in ladder if r.get("format") == fmt and r.get("opponent_rating")]
            ys, labels = [], []
            for lo, hi in zip(bins, bins[1:]):
                grp = [not r["lost"] for r in rows if lo <= r["opponent_rating"] < hi]
                ys.append(np.mean(grp) if grp else np.nan)
                labels.append(f"{lo}-{hi}\n(n={len(grp)})")
            ax[1].plot(range(len(ys)), ys, marker="o", label=mark)
        ax[1].set_xticks(range(len(labels)))
        ax[1].set_xticklabels(labels, fontsize=8)
        ax[1].axhline(0.5, color="gray", ls=":")
        ax[1].set_ylim(0, 1)
        ax[1].legend()
        ax[1].set_title("Win rate by opponent rating band")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "06_ladder.png", dpi=120)
        plt.close(fig)

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures: {OUT_DIR}")


_TEAM_CACHE: dict[str, list[str]] = {}


def _team_species(team_id: str) -> list[str]:
    """Resolve 'mc_ladder/mc_ladder_06.packed' or 'rain_offense/team_20.packed' to species ids."""
    if not _TEAM_CACHE:
        # The collection manifests are the authoritative team_id -> file mapping; the
        # same 'archetype/team_NN.packed' name exists in several pools with different bytes.
        for manifest in (REPO_ROOT / "data" / "selfplay" / "collect_mc").glob("shard_*.json"):
            for entry in json.load(open(manifest)):
                name = Path(entry["file"]).name.removesuffix(".txt")
                tid = f"{entry['archetype']}/{name}"
                species = entry.get("species")
                if species is None:
                    path = (manifest.parent / entry["file"]).resolve()
                    species = [
                        _species_id((mon.split("|") + [""])[1] or mon.split("|")[0])
                        for mon in path.read_text().strip().split("]") if mon.strip()
                    ]
                _TEAM_CACHE.setdefault(tid, [_species_id(s) for s in species])
    return _TEAM_CACHE.get(team_id, [])


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--force", action="store_true")
    sub.add_parser("plot")
    one = sub.add_parser("_extract_one")
    one.add_argument("pt")
    one.add_argument("out")
    args = p.parse_args()
    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "plot":
        cmd_plot(args)
    else:
        _extract_shard(Path(args.pt), Path(args.out))


if __name__ == "__main__":
    main()
