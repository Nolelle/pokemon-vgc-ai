import copy
import gzip
import json

import pytest

from vgc.bc.replay_pretraining import file_hash, iter_task_examples, load_sampling_plan, prepare, task_units
from vgc.bc.replay_sampling import sample_task_units
from vgc.bc.replay_supervision import audit_raw_log, build_supervision


def fixture(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    roster = ["pikachu", "charizard", "incineroar", "salamence", "kingambit", "sneasler"]
    games, records = [], []
    for i, split in enumerate(("train", "validation", "test", "excluded")):
        rid = f"replay-{i}"
        log = ("|player|p1|Alice\n|player|p2|Bob\n"
               "|switch|p1a: Pikachu|Pikachu, L50|100/100\n"
               "|switch|p2a: Charizard|Charizard, L50|100/100\n"
               "|turn|1\n|move|p1a: Pikachu|Thunderbolt|p2a: Charizard\n"
               "|move|p2a: Charizard|Protect|p2a: Charizard\n|win|Alice\n")
        path = raw_dir / f"{rid}.json"
        path.write_text(json.dumps({"id": rid, "formatid": "gen9championsvgc2026regmc", "log": log}))
        games.append({"id": rid, "split": split, "exclusions": ["review"] if split == "excluded" else [],
                      "winner": "p1", "ending": "played_out", "file_sha256": file_hash(path),
                      "ratings": {"p1": 1700, "p2": 1300}, "rosters": {"p1": roster, "p2": roster},
                      "accounts": {"p1": "alice", "p2": "bob"},
                      "expected_preview": {}, "counts": {"p1": {"records": 1}, "p2": {"records": 1}}})
        for side, move in (("p1", "thunderbolt"), ("p2", "protect")):
            records.append({"replay_id": rid, "player": side, "schema": 7,
                            "format_id": "gen9championsvgc2026regmc", "decision_kind": "turn", "turn": 1,
                            "state": {"our": {"preview_species": roster}, "opp": {}, "field": {"turn": 1}},
                            "action": {"slot0": {"kind": "move", "move_id": move}, "slot1": {"kind": "unknown"}},
                            "action_status": {"slot0": "observed", "slot1": "unknown"},
                            "outcome": "win" if side == "p1" else "loss", "won": side == "p1"})
    data, manifest, summary = tmp_path / "source.gz", tmp_path / "manifest.jsonl", tmp_path / "selection.json"
    with gzip.open(data, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    manifest.write_text("".join(json.dumps(g) + "\n" for g in games))
    summary.write_text(json.dumps({"input_sha256": file_hash(data), "games": 4, "records": 8,
                                  "groups": {g["split"]: {"games": 1} for g in games}}))
    return data, manifest, summary, raw_dir, records


def test_prepare_and_consume_preserves_input_and_fixed_split(tmp_path):
    data, manifest, summary, raw, original = fixture(tmp_path)
    out = tmp_path / "prepared"
    hashes = [file_hash(p) for p in (data, manifest)]
    report = prepare(data, manifest, summary, raw, out, expected_selection_sha256=hashes[1])
    assert report["source_records"] == 8 and report["excluded_records"] == 2
    assert [file_hash(p) for p in (data, manifest)] == hashes
    examples = list(iter_task_examples(out, "move", phase="expert"))
    assert len(examples) == 1
    assert examples[0]["reference"][:2] == ("replay-0", "p1")
    assert examples[0]["input"] == {"state": original[0]["state"]}
    assert examples[0]["sampling_probability"] == 1
    plan = load_sampling_plan(out, phase="expert")
    assert sample_task_units(plan, "move", 3, 7) == [("replay-0", "p1", "move", 0)] * 3
    assert not list(iter_task_examples(out, "target"))
    dev = list(iter_task_examples(out, "outcome", split="validation"))
    assert len(dev) == 2 and all(x["sampling_probability"] is None for x in dev)
    with pytest.raises(ValueError, match="final test"):
        list(iter_task_examples(out, "outcome", split="test"))
    assert len(list(iter_task_examples(out, "outcome", split="test", final_evaluation=True))) == 2
    with pytest.raises(FileExistsError):
        prepare(data, manifest, summary, raw, out, expected_selection_sha256=hashes[1])


def test_fingerprint_changes_fail_before_use(tmp_path):
    data, manifest, summary, raw, _ = fixture(tmp_path)
    out = tmp_path / "prepared"
    prepare(data, manifest, summary, raw, out, expected_selection_sha256=file_hash(manifest))
    with (out / "train.jsonl.gz").open("ab") as f:
        f.write(b"modified")
    with pytest.raises(ValueError, match="fingerprint"):
        list(iter_task_examples(out, "move"))
    data.write_bytes(b"different input")
    with pytest.raises(ValueError, match="fingerprint"):
        prepare(data, manifest, summary, raw, tmp_path / "other", expected_selection_sha256=file_hash(manifest))


def test_swapping_train_and_test_is_rejected_even_with_same_counts(tmp_path):
    data, manifest, summary, raw, _ = fixture(tmp_path)
    pinned = file_hash(manifest)
    games = [json.loads(line) for line in manifest.open()]
    games[0]["split"], games[2]["split"] = games[2]["split"], games[0]["split"]
    manifest.write_text("".join(json.dumps(g) + "\n" for g in games))
    with pytest.raises(ValueError, match="pinned fingerprint"):
        prepare(data, manifest, summary, raw, tmp_path / "bad", expected_selection_sha256=pinned)
    assert not (tmp_path / "bad").exists()


def test_unknown_targets_cannot_be_consumed_and_future_input_is_not_added():
    row = {"decision_kind": "turn", "player": "p1", "turn": 1,
           "state": {"our": {}, "opp": {"item": None}, "field": {}},
           "action_status": {"slot0": "unknown", "slot1": "unknown"},
           "action": {"slot0": {"kind": "unknown", "move_id": "fake"}},
           "outcome": "win", "won": True}
    before = copy.deepcopy(row)
    log = "|player|p1|A\n|player|p2|B\n|turn|1\n|turn|2\n|-item|p2a: Foe|Leftovers\n|win|A\n"
    sup = build_supervision(row, {"winner": "p1", "ending": "played_out"}, audit_raw_log(log))
    assert row == before
    assert not list(task_units(sup, "move"))
    assert list(task_units(sup, "outcome")) == [(None, "win")]
    sup["targets"]["move"][0] = "fake"
    with pytest.raises(ValueError, match="disabled targets"):
        list(task_units(sup, "move"))
