"""Prepare and consume explicitly partial replay supervision with fixed splits.

This is a data preparation path, not a candidate-policy trainer. It never invents
joint orders, legal requests, private sets, or loss targets for unknown fields.
"""
from __future__ import annotations

import collections
import contextlib
import gzip
import hashlib
import json
import math
from pathlib import Path

from vgc.bc.replay_sampling import TASKS, SamplingPlan, build_sampling_plan, sample_task_units
from vgc.bc.replay_supervision import audit_raw_log, build_supervision

FORMAT = "gen9championsvgc2026regmc"
SCHEMA = "partial-replay-pretraining-v1"
SPLITS = ("train", "validation", "test")


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def task_units(supervision: dict, task: str):
    """Yield enabled components only; reject missing/contradictory mask contracts."""
    if task not in TASKS or supervision.get("joint_choice_certified") is not False:
        raise ValueError("explicit partial-only supervision is required")
    masks, targets = supervision["masks"][task], supervision["targets"][task]
    if task in TASKS[:3]:
        if not isinstance(masks, list) or not isinstance(targets, list) or len(masks) != 2 or len(targets) != 2:
            raise ValueError("slot tasks require two explicit masks and targets")
        pairs = list(enumerate(zip(masks, targets)))
    else:
        pairs = [(None, (masks, targets))]
    for component, (mask, target) in pairs:
        if type(mask) is not bool or mask != (target is not None):
            raise ValueError("disabled targets must be null; enabled targets must be present")
        if mask:
            yield component, target


def _load_selection(manifest: Path, summary_path: Path, data: Path, expected_selection_sha256: str):
    if not expected_selection_sha256 or file_hash(manifest) != expected_selection_sha256:
        raise ValueError("selection manifest does not match the pinned fingerprint")
    summary = json.loads(summary_path.read_text())
    if file_hash(data) != summary["input_sha256"]:
        raise ValueError("source dataset does not match the selected fingerprint")
    games = {}
    for line in manifest.open():
        game = json.loads(line)
        rid = game["id"]
        if rid in games:
            raise ValueError("duplicate replay in selection manifest")
        if game["split"] not in (*SPLITS, "excluded", "reserved_team_excluded"):
            raise ValueError("unrecognized split")
        if game["split"] in SPLITS and game["exclusions"]:
            raise ValueError("excluded replay assigned to a retained split")
        games[rid] = game
    if len(games) != summary["games"]:
        raise ValueError("selection game count mismatch")
    actual = collections.Counter(g["split"] for g in games.values())
    for split, count in actual.items():
        if count != summary["groups"][split]["games"]:
            raise ValueError("selection split count mismatch")
    return games, summary


def prepare(data: Path, manifest: Path, selection_summary: Path, raw_dir: Path, out: Path,
            *, expected_selection_sha256: str):
    """Write immutable prepared shards and training-only sampling plans, no fitting."""
    games, selection = _load_selection(manifest, selection_summary, data, expected_selection_sha256)
    manifest_fingerprint = file_hash(manifest)
    summary_fingerprint = file_hash(selection_summary)
    out.mkdir(parents=True, exist_ok=False)
    counts = collections.Counter()
    split_counts = {s: collections.Counter() for s in SPLITS}
    unit_counts = collections.Counter()
    source_per_game = collections.Counter()
    expert_counts = {s: collections.Counter() for s in SPLITS}
    reasons = {s: collections.Counter() for s in SPLITS}
    seen = set()
    current_id, audit = None, None
    input_state_digest = hashlib.sha256()
    output_paths = {s: out / f"{s}.jsonl.gz" for s in SPLITS}
    with contextlib.ExitStack() as stack:
        handles = {s: stack.enter_context(gzip.open(path, "xt", compresslevel=3)) for s, path in output_paths.items()}
        source = stack.enter_context(gzip.open(data, "rt"))
        for line_number, line in enumerate(source, 1):
            record = json.loads(line)
            rid, side = record["replay_id"], record["player"]
            if rid not in games or side not in ("p1", "p2"):
                raise ValueError("source replay or player not in the fixed selection")
            if record.get("schema") != 7 or record.get("format_id") != FORMAT:
                raise ValueError("source schema or format mismatch")
            source_per_game[rid] += 1
            seen.add(rid)
            counts["source_records"] += 1
            game = games[rid]
            split = game["split"]
            if split not in SPLITS:
                counts["excluded_records"] += 1
                continue
            if current_id != rid:
                raw_path = raw_dir / f"{rid}.json"
                if file_hash(raw_path) != game["file_sha256"]:
                    raise ValueError(f"raw replay changed: {rid}")
                raw = json.loads(raw_path.read_text())
                if raw.get("id") != rid or raw.get("formatid") != FORMAT:
                    raise ValueError("raw replay identity/format mismatch")
                audit = audit_raw_log(raw["log"])
                current_id = rid
            supervision = build_supervision(record, game, audit)
            # Only the pre-decision snapshot is a model input. Outcomes, selected
            # actions, future preview appearances, names, and ratings are metadata.
            example = {
                "schema": SCHEMA, "replay_id": rid, "player": side, "split": split,
                "source_line": line_number, "decision_kind": record["decision_kind"],
                "turn": record["turn"], "input": {"state": record["state"]},
                "pre_rating": game["ratings"][side], "supervision": supervision,
            }
            state_bytes = json.dumps(example["input"], sort_keys=True, separators=(",", ":")).encode()
            input_state_digest.update(str(line_number).encode() + b"\0" + state_bytes + b"\n")
            expert = isinstance(example["pre_rating"], (int, float)) and example["pre_rating"] >= 1600
            for task in TASKS:
                n = len(list(task_units(supervision, task)))
                unit_counts[rid, side, task] += n
                split_counts[split][task] += n
                if expert:
                    expert_counts[split][task] += n
                reason = supervision["reasons"][task]
                reasons[split].update(f"{task}:{r}" for r in (reason if isinstance(reason, list) else [reason]))
            handles[split].write(json.dumps(example, separators=(",", ":")) + "\n")
            split_counts[split]["records"] += 1
            if counts["source_records"] % 100000 == 0:
                print(f"Prepared through {counts['source_records']:,} source records", flush=True)
    if counts["source_records"] != selection["records"] or seen != set(games):
        raise ValueError("source records/games failed reconciliation")
    for rid, game in games.items():
        if source_per_game[rid] != sum(c["records"] for c in game["counts"].values()):
            raise ValueError("per-game source record count mismatch")
    sampling = {}
    for phase in ("broad", "expert"):
        plan = build_sampling_plan(games, unit_counts, phase=phase)
        with (out / f"sampling_{phase}.jsonl").open("x") as handle:
            for key in sorted(plan.group_counts):
                rid, side, task = key
                handle.write(json.dumps({"replay_id": rid, "player": side, "task": task,
                                         "units": plan.group_counts[key],
                                         "probability_per_unit": plan.per_unit_weights[key]}) + "\n")
        smoke = {}
        for task in TASKS:
            mass = math.fsum(plan.per_unit_weights[k] * n for k, n in plan.group_counts.items() if k[2] == task)
            expected = 1.0 if plan.diagnostics[task]["eligible_units"] else 0.0
            if not math.isclose(mass, expected, abs_tol=1e-10):
                raise ValueError("sampling task mass failed reconciliation")
            if expected:
                refs = sample_task_units(plan, task, 10000, 20261102)
                if any(games[rid]["split"] != "train" for rid, _, _, _ in refs):
                    raise ValueError("held-out replay sampled for training")
                smoke[task] = {"draws": len(refs), "unique_games": len({x[0] for x in refs}),
                               "heldout_games": 0}
        sampling[phase] = {"method": plan.method, "diagnostics": plan.diagnostics, "smoke": smoke}
    files = list(output_paths.values()) + [out / f"sampling_{phase}.jsonl" for phase in ("broad", "expert")]
    report = {
        "status": "complete", "schema": SCHEMA, "source_records": counts["source_records"],
        "excluded_records": counts["excluded_records"], "source_games": len(games),
        "splits": split_counts, "expert_splits": expert_counts, "mask_reasons": reasons,
        "sampling": sampling, "inputs": {"dataset": str(data.resolve()),
        "dataset_sha256": selection["input_sha256"], "selection": str(manifest.resolve()),
        "selection_sha256": manifest_fingerprint, "selection_summary_sha256": summary_fingerprint},
        "input_state_digest": input_state_digest.hexdigest(),
        "files_sha256": {p.name: file_hash(p) for p in files},
        "joint_choice_certified": False, "model_training_performed": False,
        "limits": ["Observed partial supervision, not submitted-choice certification.",
                   "All target-choice and complete-joint-choice teaching disabled.",
                   "No candidate scorer is trained by this exporter or its sampler.",
                   "Training weights are per task; development/test frequencies are unchanged.",
                   "Hidden own sets and complete legal alternatives remain unavailable."],
    }
    if file_hash(manifest) != manifest_fingerprint or file_hash(data) != selection["input_sha256"]:
        raise ValueError("inputs changed while preparing")
    (out / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def load_sampling_plan(directory: Path, *, phase="broad") -> SamplingPlan:
    """Load the saved train-only plan for sample_task_units, checking its identity."""
    if phase not in ("broad", "expert"):
        raise ValueError("phase must be broad or expert")
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("schema") != SCHEMA:
        raise ValueError("incomplete or incompatible prepared dataset")
    selection = Path(summary["inputs"]["selection"])
    if file_hash(selection) != summary["inputs"]["selection_sha256"]:
        raise ValueError("saved selection fingerprint mismatch")
    games = {g["id"]: g for g in map(json.loads, selection.open())}
    path = directory / f"sampling_{phase}.jsonl"
    if file_hash(path) != summary["files_sha256"][path.name]:
        raise ValueError("sampling plan fingerprint mismatch")
    counts, weights = {}, {}
    mass = collections.Counter()
    for line in path.open():
        group = json.loads(line)
        rid, side, task = group["replay_id"], group["player"], group["task"]
        key = rid, side, task
        game = games.get(rid, {})
        if (key in counts or game.get("split") != "train" or game.get("exclusions")
                or side not in ("p1", "p2") or task not in TASKS):
            raise ValueError("invalid or held-out sampling group")
        rating = game["ratings"].get(side)
        if phase == "expert" and (not isinstance(rating, (int, float)) or rating < 1600):
            raise ValueError("non-expert player in expert sampling plan")
        count, weight = group["units"], group["probability_per_unit"]
        if type(count) is not int or count <= 0 or not math.isfinite(weight) or weight <= 0:
            raise ValueError("invalid sampling probability/count")
        counts[key], weights[key] = count, weight
        mass[task] += count * weight
    info = summary["sampling"][phase]
    for task in TASKS:
        expected = 1 if info["diagnostics"][task]["eligible_units"] else 0
        if not math.isclose(mass[task], expected, abs_tol=1e-10):
            raise ValueError("sampling probabilities do not sum to one")
    return SamplingPlan(phase, info["method"], weights, counts, info["diagnostics"], {}, {})


def iter_task_examples(directory: Path, task: str, *, split="train", phase="broad", final_evaluation=False):
    """Consume only enabled labels with per-unit sampling probability attached.

    This iterator is exhaustive, not itself a random sampler. Use the saved group
    probabilities to sample; do not also apply them as loss weights after sampling.
    Ordinals match sample_task_units' within-group references. Final-test access
    requires an explicit evaluation mode; it never supplies training probabilities.
    """
    if task not in TASKS or phase not in ("broad", "expert") or split not in SPLITS:
        raise ValueError("invalid task, phase, or split")
    if split == "test" and not final_evaluation:
        raise ValueError("final test is reserved for explicit final evaluation")
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("schema") != SCHEMA:
        raise ValueError("incomplete or incompatible prepared dataset")
    shard = directory / f"{split}.jsonl.gz"
    if file_hash(shard) != summary["files_sha256"][shard.name]:
        raise ValueError("prepared shard fingerprint mismatch")
    groups = {}
    if split == "train":
        plan_path = directory / f"sampling_{phase}.jsonl"
        if file_hash(plan_path) != summary["files_sha256"][plan_path.name]:
            raise ValueError("sampling plan fingerprint mismatch")
        for line in plan_path.open():
            g = json.loads(line)
            if g["task"] == task:
                groups[g["replay_id"], g["player"]] = g
    ordinals = collections.Counter()
    with gzip.open(shard, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != split or row.get("schema") != SCHEMA:
                raise ValueError("prepared row split/schema mismatch")
            group_key = row["replay_id"], row["player"]
            if split == "train" and group_key not in groups:
                continue
            rating = row.get("pre_rating")
            if phase == "expert" and (not isinstance(rating, (int, float)) or rating < 1600):
                continue
            for component, target in task_units(row["supervision"], task):
                ordinal = ordinals[group_key]
                ordinals[group_key] += 1
                yield {"input": row["input"], "target": target, "component": component,
                       "reference": (*group_key, task, ordinal), "source_line": row["source_line"],
                       "sampling_probability": groups[group_key]["probability_per_unit"] if split == "train" else None}
    if split == "train" and any(ordinals[k] != g["units"] for k, g in groups.items()):
        raise ValueError("prepared label counts no longer match the sampler")
