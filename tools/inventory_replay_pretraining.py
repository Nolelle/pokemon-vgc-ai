"""Read-only inventory of replay records and a proposed game-level split.

This does not produce training examples or certify recovered choices.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import html
import json
import re
import sys
from pathlib import Path

try:
    from tools.audit_replay_state import raw_identities
except ModuleNotFoundError:
    from audit_replay_state import raw_identities

FORMAT = "gen9championsvgc2026regmc"
SEATS = ("p1", "p2")
RATING_RE = re.compile(r"^\|raw\|(.+?)'s rating: (\d+) &rarr; <strong>(\d+)</strong>", re.M)


def toid(value):
    return re.sub(r"[^a-z0-9]", "", html.unescape(str(value)).lower())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def raw_metadata(path, species=None):
    raw = path.read_bytes()
    d = json.loads(raw)
    log = d.get("log") or ""
    names = {}
    roster = {s: [] for s in SEATS}
    timestamps = []
    for line in log.splitlines():
        p = line.split("|")
        if len(p) > 3 and p[1] == "player" and p[2] in SEATS and toid(p[3]):
            names.setdefault(p[2], toid(p[3]))
        if len(p) > 3 and p[1] == "poke" and p[2] in SEATS:
            roster[p[2]].append(toid(p[3].split(",")[0]))
        if len(p) > 2 and p[1] == "t:":
            try:
                t = int(p[2])
                if 0 < t < 4102444800:
                    timestamps.append(t)
            except ValueError:
                pass
    ratings = {s: set() for s in SEATS}
    for name, before, _ in RATING_RE.findall(log):
        matches = [s for s in SEATS if names.get(s) and names[s] == toid(name)]
        if len(matches) == 1:
            ratings[matches[0]].add(int(before))
    wins = re.findall(r"^\|win\|([^\n]*)", log, re.M)
    winner_name = toid(wins[-1]) if wins else None
    matches = [s for s in SEATS if names.get(s) and names[s] == winner_name]
    winner = matches[0] if len(matches) == 1 else None
    if re.search(r"^\|-message\|[^\n]*(lost due to inactivity|lost by being offline)", log, re.M):
        ending = "timer_offline"
    elif re.search(r"^\|-message\|[^\n]*forfeited\.", log, re.M):
        ending = "forfeit"
    elif re.search(r"^\|tie(?:\||$)", log, re.M):
        ending = "draw"
    elif winner:
        ending = "played_out"
    else:
        ending = "unresolved"
    upload = d.get("uploadtime")
    upload = upload if isinstance(upload, int) and 0 < upload < 4102444800 else None
    anomalies = []
    if not timestamps:
        anomalies.append("missing_valid_battle_time")
    if timestamps and upload and timestamps[0] > upload:
        anomalies.append("battle_time_after_upload")
    if any(a > b for a, b in zip(timestamps, timestamps[1:])):
        anomalies.append("decreasing_battle_timestamps")
    chosen_time = timestamps[0] if timestamps else upload
    time_source = "battle" if timestamps else "upload"
    if timestamps and upload and timestamps[0] > upload:
        chosen_time, time_source = upload, "upload_after_impossible_battle_time"
    if any(len(v) > 1 for v in ratings.values()):
        anomalies.append("conflicting_pre_ratings")
    return {"id": d.get("id"), "format": d.get("formatid"), "file_sha256": hashlib.sha256(raw).hexdigest(),
            "log_sha256": hashlib.sha256(log.encode()).hexdigest(), "accounts": names,
            "ratings": {s: next(iter(v)) if len(v) == 1 else None for s, v in ratings.items()},
            "rating_status": {s: "missing" if not v else "conflicting" if len(v) > 1 else "known"
                              for s, v in ratings.items()},
            "rosters": {s: sorted(v) for s, v in roster.items()}, "winner": winner,
            "ending": ending, "rated": bool(re.search(r"^\|rated(?:\||$)", log, re.M)),
            "showteam": bool(re.search(r"^\|showteam\|", log, re.M)),
            "time": chosen_time, "time_source": time_source,
            "uploadtime": upload, "anomalies": anomalies,
            "protocol_flags": {"called_or_locked": bool(re.search(
                r"\|(?:move|cant)\|[^\n]*(?:\[from\]|lockedmove|recharge)", log, re.I)),
                "drag": "|drag|" in log, "transform": "|-transform|" in log,
                "illusion_or_replace": "|replace|" in log or bool(re.search(r"\|ability\|[^\n]*illusion", log, re.I)),
                "baton_pass": "|Baton Pass|" in log},
            "expected_preview": raw_identities(log, species)[1] if species is not None else None}


def preview_complete(row):
    a = row.get("action") or {}
    picked, leads = a.get("picked") or [], a.get("lead_order") or []
    preview = (row.get("state") or {}).get("our", {}).get("preview") or []
    return (len(picked) == len(set(picked)) == 4 and len(leads) == len(set(leads)) == 2
            and all(isinstance(x, str) and x for x in picked + leads)
            and set(picked) <= set(preview) and set(leads) <= set(picked))


def row_checks(row):
    errors = []
    state = row.get("state") or {}
    if row.get("decision_kind") not in ("turn", "forced_switch", "teampreview"):
        errors.append("unknown_decision_kind")
    if any(not isinstance(state.get(side), dict) for side in ("our", "opp", "field")):
        errors.append("missing_state_fields")
    if row.get("decision_kind") != "teampreview":
        for side in ("our", "opp"):
            s = state.get(side) or {}
            active, bench = s.get("active") or [], s.get("bench") or []
            preview = set(s.get("preview_species") or [])
            if len(active) > 2:
                errors.append("too_many_active")
            for mon in active:
                if mon is None:
                    continue
                hp = mon.get("hp_fraction")
                if not isinstance(hp, (int, float)) or not 0 <= hp <= 1:
                    errors.append("invalid_hp")
                if any(not isinstance(v, (int, float)) or not -6 <= v <= 6
                       for v in (mon.get("boosts") or {}).values()):
                    errors.append("invalid_boost")
                if not preview or mon.get("species") not in preview:
                    errors.append("active_not_on_preview")
            for mon in bench:
                if not preview or mon.get("species_id") not in preview:
                    errors.append("bench_not_on_preview")
    return sorted(set(errors))


def summarize_row(row):
    c = collections.Counter()
    kind = row.get("decision_kind", "missing")
    c["records"] += 1
    c["kind_" + kind] += 1
    if row_checks(row):
        c["bad_state_rows"] += 1
    if kind == "teampreview":
        c["preview_complete"] += int(preview_complete(row) and not row_checks(row))
    if kind == "turn":
        for slot in ("slot0", "slot1"):
            status = (row.get("action_status") or {}).get(slot, "missing")
            c["slot_" + status] += 1
            action = (row.get("action") or {}).get(slot) or {}
            if status != "observed" or row_checks(row):
                continue
            if action.get("kind") == "move" and action.get("move_id"):
                c["observed_move"] += 1
                if action.get("target_slot") is not None:
                    c["observed_known_target"] += 1
            elif action.get("kind") == "switch":
                c["observed_switch"] += 1
    return c


def outcome_matches(g, seat):
    expected = "win" if g["winner"] == seat else "loss" if g["winner"] else None
    return bool(expected and g["outcomes"][seat] == {(expected, expected == "win")})


def outcome_candidate(g, group):
    allowed = {"open_team_sheet"} if group == "all_games" else set()
    return (g["ending"] in ("forfeit", "played_out")
            and not (set(g["exclusions"]) - allowed))


def assign_splits(games):
    """Place same-time and exact-log duplicates together; exclude duplicate games."""
    by_hash = collections.defaultdict(list)
    for g in games:
        by_hash[g["log_sha256"]].append(g)
    for group in by_hash.values():
        if len(group) > 1:
            for g in group:
                g["exclusions"].append("exact_duplicate")
    eligible = sorted((g for g in games if not g["exclusions"] and g["time"] is not None),
                      key=lambda x: (x["time"], x["id"]))
    n = len(eligible)
    i = 0
    while i < n:
        j = i + 1
        while j < n and eligible[j]["time"] == eligible[i]["time"]:
            j += 1
        midpoint = (i + j) / 2
        split = "train" if midpoint < n * .8 else "validation" if midpoint < n * .9 else "test"
        for g in eligible[i:j]:
            g["base_split"] = split
        i = j
    for g in games:
        g.setdefault("base_split", "excluded")
        g["split"] = g["base_split"]


def reserve_teams(games):
    test_rosters = {tuple(r) for g in games if g["split"] == "test" for r in g["rosters"].values()}
    # Rank hashes, selecting the nearest 5% (at least one for a nonempty test).
    count = max(1, round(len(test_rosters) * .05)) if test_rosters else 0
    reserved = set(sorted(test_rosters, key=lambda r: hashlib.sha256(
        ("20261102:" + ",".join(r)).encode()).hexdigest())[:count])
    for g in games:
        match = any(tuple(r) in reserved for r in g["rosters"].values())
        g["team_challenge"] = g["split"] == "test" and match
        if g["split"] in ("train", "validation") and match:
            g["split"] = "reserved_team_excluded"
    return reserved


def baseline_ids(path):
    if path is None:
        return set()
    ids = set()
    with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, str):
                ids.add(obj)
            elif isinstance(obj, dict):
                value = obj.get("replay_id") or obj.get("id")
                if value:
                    ids.add(value)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-index", type=Path, default=Path("runs/replay_refresh_20261101/index_before.jsonl"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = baseline_ids(args.baseline_index)
    species_path = Path(__file__).resolve().parents[1] / "data/champions/species.json"
    species = json.loads(species_path.read_text())
    games = {}
    rows_total = 0
    with gzip.open(args.data, "rt") as f:
        for line in f:
            row = json.loads(line)
            rid = row["replay_id"]
            rows_total += 1
            g = games.setdefault(rid, {"id": rid, "counts": {s: collections.Counter() for s in SEATS},
                                      "row_errors": collections.Counter(), "exclusions": [],
                                      "outcomes": {s: set() for s in SEATS}, "formats": set(), "schemas": set(),
                                      "preview_actions": {s: [] for s in SEATS}})
            seat = row.get("player")
            if seat not in SEATS:
                g["row_errors"]["invalid_seat"] += 1
                continue
            if row.get("decision_kind") == "teampreview":
                g["preview_actions"][seat].append({"action": row.get("action"),
                                                   "complete": preview_complete(row)})
            g["counts"][seat].update(summarize_row(row))
            g["row_errors"].update(row_checks(row))
            outcome, won = row.get("outcome"), row.get("won")
            g["outcomes"][seat].add((outcome, won))
            g["formats"].add(row.get("format_id"))
            g["schemas"].add(row.get("schema"))
            if rows_total % 100000 == 0:
                print(f"Read {rows_total:,} rows", file=sys.stderr, flush=True)
    raw_extra = 0
    for i, path in enumerate(sorted(args.raw_dir.glob("*.json")), 1):
        if path.stem not in games:
            raw_extra += 1
            continue
        g = games[path.stem]
        metadata = raw_metadata(path, species)
        g.update({k: v for k, v in metadata.items() if k != "id"})
        if metadata["id"] != g["id"]:
            g["exclusions"].append("raw_id_mismatch")
        if metadata["format"] != FORMAT or g["formats"] != {FORMAT}:
            g["exclusions"].append("format_mismatch")
        if g["schemas"] != {7}:
            g["exclusions"].append("schema_mismatch")
        if any(len(r) != 6 or len(set(r)) != 6 or not all(r) for r in g["rosters"].values()):
            g["exclusions"].append("incomplete_preview")
        if any(len(g["preview_actions"][seat]) != 1 for seat in SEATS):
            g["exclusions"].append("preview_record_count")
        for seat in SEATS:
            for record in g["preview_actions"][seat]:
                matched = record["action"] == g["expected_preview"][seat]
                g["counts"][seat]["preview_raw_matched" if matched else "preview_raw_mismatch"] += 1
                if matched and record["complete"]:
                    g["counts"][seat]["preview_complete_raw_matched"] += 1
        if any(g["counts"][seat]["preview_raw_mismatch"] for seat in SEATS):
            g["exclusions"].append("preview_raw_mismatch")
        if not g["rated"]:
            g["exclusions"].append("unrated")
        if "laplacestheorems" in g["accounts"].values():
            g["exclusions"].append("own_account")
        # These can invalidate reconstructed state or apparent preview identities.
        # Keep the entire game aside in this first conservative proposal. This is
        # not an exhaustive detector of unsupported mechanics or action ambiguity.
        for flag in ("transform", "illusion_or_replace", "baton_pass"):
            if g["protocol_flags"][flag]:
                g["exclusions"].append("state_review_" + flag)
        if g["row_errors"]:
            g["exclusions"].append("bad_rows")
        for seat in SEATS:
            if not outcome_matches(g, seat):
                g["exclusions"].append("outcome_mismatch_or_unresolved")
                break
        if not g["time"]:
            g["exclusions"].append("missing_time")
        if i % 10000 == 0:
            print(f"Scanned {i:,} raw files", file=sys.stderr, flush=True)
    for g in games.values():
        if "log_sha256" not in g:
            g.update(log_sha256="missing:" + g["id"], time=None, rosters={}, ratings={},
                     accounts={}, ending="missing", showteam=False, rated=False, winner=None)
            g["exclusions"].append("missing_raw")
        if g["showteam"]:
            g["exclusions"].append("open_team_sheet")
        if any(len(g["preview_actions"][seat]) != 1 for seat in SEATS):
            g["exclusions"].append("preview_record_count")
        g["baseline"] = g["id"] in baseline
        g["exclusions"] = sorted(set(g["exclusions"]))
    assign_splits(list(games.values()))
    reserved = reserve_teams(list(games.values()))
    groups = collections.defaultdict(lambda: {"games": 0, "baseline_games": 0, "counts": collections.Counter(),
                                             "counts_pre1600": collections.Counter(), "times": [],
                                             "accounts": set(), "rosters": set(), "endings": collections.Counter(),
                                             "counts_by_seat": {s: collections.Counter() for s in SEATS},
                                             "rating_bands": collections.Counter(),
                                             "rating_status": collections.Counter(), "account_frequency": collections.Counter(),
                                             "roster_frequency": collections.Counter()})
    for g in games.values():
        for name in ("all_games", g["split"], "team_challenge" if g.get("team_challenge") else None):
            if name is None:
                continue
            group = groups[name]
            group["games"] += 1
            group["baseline_games"] += g["baseline"]
            group["endings"][g["ending"]] += 1
            if g["time"]:
                group["times"].append(g["time"])
            group["accounts"].update(g["accounts"].values())
            group["account_frequency"].update(set(g["accounts"].values()))
            group["roster_frequency"].update({tuple(r) for r in g["rosters"].values()})
            group["rosters"].update(tuple(r) for r in g["rosters"].values())
            for seat in SEATS:
                group["counts"]["player_game_examples"] += 1
                rating = g["ratings"].get(seat)
                group["rating_status"][g.get("rating_status", {}).get(seat, "missing")] += 1
                band = "missing_or_conflicting" if rating is None else (
                    "under1200" if rating < 1200 else "1200_1399" if rating < 1400 else
                    "1400_1599" if rating < 1600 else "1600_1799" if rating < 1800 else "1800_plus")
                group["rating_bands"][band] += 1
                if rating is not None and rating >= 1600:
                    group["counts"]["player_games_pre1600"] += 1
                    if g["winner"] in SEATS:
                        group["counts"]["pre1600_wins" if g["winner"] == seat else "pre1600_losses"] += 1
                group["counts"].update(g["counts"][seat])
                group["counts_by_seat"][seat].update(g["counts"][seat])
                if (g["ratings"].get(seat) or 0) >= 1600:
                    group["counts_pre1600"].update(g["counts"][seat])
                if outcome_candidate(g, name):
                    group["counts"]["outcome_candidates"] += g["counts"][seat]["records"]
                    if (g["ratings"].get(seat) or 0) >= 1600:
                        group["counts_pre1600"]["outcome_candidates"] += g["counts"][seat]["records"]
    overlaps = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlaps[a + "_" + b] = {k: len(groups[a][k] & groups[b][k]) for k in ("accounts", "rosters")}
    summary_groups = {}
    for name, group in groups.items():
        times = group.pop("times")
        group["date_range_utc"] = [dt.datetime.fromtimestamp(t, dt.UTC).isoformat() for t in
                                   (min(times), max(times))] if times else None
        for key in ("account_frequency", "roster_frequency"):
            frequency = group.pop(key)
            group["top_" + key + "_share"] = max(frequency.values(), default=0) / max(1, group["games"])
        group["accounts"] = len(group["accounts"])
        group["rosters"] = len(group["rosters"])
        group["counts"]["certified_complete_joint_actions"] = 0
        summary_groups[name] = group
    exclusions = collections.Counter(reason for g in games.values() for reason in g["exclusions"])
    assert sum(g["games"] for name, g in summary_groups.items()
               if name not in ("all_games", "team_challenge")) == len(games)
    assert sum(g["counts"]["records"] for name, g in summary_groups.items()
               if name not in ("all_games", "team_challenge")) == rows_total
    assert not any(tuple(r) in reserved for g in games.values() if g["split"] in ("train", "validation")
                   for r in g["rosters"].values())
    retained_hashes = collections.defaultdict(set)
    for g in games.values():
        if g["split"] in ("train", "validation", "test"):
            retained_hashes[g["log_sha256"]].add(g["split"])
    assert all(len(splits) == 1 for splits in retained_hashes.values())
    assert all(all(len(g["preview_actions"][s]) == 1 for s in SEATS)
               or "preview_record_count" in g["exclusions"] for g in games.values())
    summary = {"input_path": str(args.data), "raw_dir": str(args.raw_dir),
               "script_sha256": digest(Path(__file__)), "species_sha256": digest(species_path),
               "baseline_path": str(args.baseline_index) if args.baseline_index else None,
               "input_sha256": digest(args.data), "baseline_sha256": digest(args.baseline_index) if args.baseline_index else None,
               "baseline_supplied": args.baseline_index is not None, "records": rows_total, "games": len(games),
               "raw_extra_files": raw_extra, "exclusions": exclusions, "groups": summary_groups,
               "overlaps": overlaps, "reserved_rosters": len(reserved),
               "base_split_games": dict(collections.Counter(g["base_split"] for g in games.values())),
               "protocol_flags": dict(collections.Counter(k for g in games.values()
                   for k, v in g.get("protocol_flags", {}).items() if v)),
               "timestamp_anomalies": dict(collections.Counter(a for g in games.values()
                   for a in g.get("anomalies", []))),
               "reserved_roster_overlap_train_validation": sum(
                   tuple(r) in reserved for g in games.values() if g["split"] in ("train", "validation")
                   for r in g["rosters"].values())}
    with (args.out / "manifest.jsonl").open("w") as f:
        for g in sorted(games.values(), key=lambda x: x["id"]):
            g["formats"] = sorted(str(v) for v in g["formats"])
            g["schemas"] = sorted(str(v) for v in g["schemas"])
            g["outcomes"] = {s: sorted([list(v) for v in values], key=str) for s, values in g["outcomes"].items()}
            f.write(json.dumps(g, sort_keys=True) + "\n")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    table = "\n".join(
        f"| {name} | {group['games']:,} | {group['counts']['records']:,} | "
        f"{group['counts']['player_games_pre1600']:,} | {group['baseline_games']:,} | "
        f"{group['date_range_utc']} |" for name, group in summary_groups.items())
    (args.out / "report.md").write_text(
        f"# Replay inventory\n\n{rows_total:,} records from {len(games):,} games; {raw_extra:,} extra raw files.\n\n"
        "| Group | Games | Records | Player games at 1600+ before play | Baseline games | UTC date range |\n"
        "|---|---:|---:|---:|---:|---|\n" + table + "\n\n"
        + f"Base split counts: {summary['base_split_games']}.\n\n"
        + f"Exclusions (may overlap): {dict(exclusions)}. Split overlaps: {overlaps}.\n\n"
        "These splits are proposed, not approved for training. The oldest 80%, next 10%, and newest 10% are grouped by battle timestamp. Exact log "
        "duplicates are excluded. The challenge reserves approximately 5% of unique test rosters, "
        "ranked by SHA-256 of `20261102:` plus the comma-separated sorted roster; earlier games "
        "containing either roster are removed. The challenge is a subset of the full test.\n\n"
        "Move, logged target and switch counts are observed candidate labels, not proof of submitted "
        "legal choices. No complete joint action is certified: legal requests are absent. "
        "Preview candidates are structural only, not independently verified submitted choices; they need four distinct valid picks and two distinct leads among them. "
        "Outcome candidates need matching binary outcomes and a played-out game or forfeit; timer "
        "and offline endings are excluded. Counts for 1600+ use the acting player's pre-game rating, "
        "including wins and losses; unknown ratings are excluded. State checks do not certify all mechanics.\n\n"
        "Games involving Transform, Illusion replacement, or Baton Pass are conservatively set aside "
        "for state/identity review. Called/locked-move and drag flags are diagnostics, not proof that "
        "all remaining recorded moves or switches were submitted choices.\n\n"
        "The data collection filter can favor particular outcomes near its rating cutoff. Existing model "
        "provenance is unknown, so the test cannot be called universally untouched; a fresh model and "
        "independent evaluation are needed. This inventory does not establish readiness or strength.\n")


if __name__ == "__main__":
    main()
