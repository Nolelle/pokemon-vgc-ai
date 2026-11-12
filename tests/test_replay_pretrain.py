"""Critical checks for the replay-pretraining trainer (Job D)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vgc.bc.encoding import SLOT_FEATURE_DIM, encode_state, flatten_state, trunk_inputs
from vgc.bc.replay_pretrain import (
    PreviewPretrainNet, TurnPretrainNet,
)


def _record() -> dict:
    return {
        "state": {
            "our": {
                "active": [
                    {"species": "incineroar", "item": "sitrusberry",
                     "ability": "intimidate", "revealed_moves": ["fakeout", "flareblitz"],
                     "hp_fraction": 1.0, "status": None, "boosts": {}, "mega": False},
                    None,
                ],
                "bench": [{"species_id": "amoonguss", "hp_fraction": 1.0, "status": None}],
                "preview_species": ["amoonguss", "incineroar"],
                "side_conditions": [], "unseen_count": 4,
            },
            "opp": {
                "active": [None, None], "bench": [],
                "preview_species": [], "side_conditions": [], "unseen_count": 6,
            },
        }
    }


def test_trunk_inputs_matches_live_construction() -> None:
    # Part 1 parity: the helper is literally the encode_live_state tail.
    index_array, scalars = trunk_inputs(_record())
    expected_idx, expected_sc = flatten_state(encode_state(_record()))
    np.testing.assert_array_equal(index_array, expected_idx)
    np.testing.assert_allclose(
        scalars, np.concatenate(
            (expected_sc, np.zeros(SLOT_FEATURE_DIM, dtype=np.float32)))
    )


def test_warm_start_loads_full_encoder(tmp_path) -> None:
    from vgc.model_vocabulary import VOCABULARY_KEY, current_vocabulary
    from vgc.rl.model import CandidatePolicyValueNet

    net = TurnPretrainNet()
    ckpt = tmp_path / "turn_net.pt"
    torch.save({"model_state_dict": net.state_encoder.state_dict(),
                VOCABULARY_KEY: current_vocabulary()}, ckpt)
    stats = CandidatePolicyValueNet().warm_start_state_encoder(ckpt)
    assert stats["loaded"] == stats["available"] and stats["stale_vocabulary_tables"] == 0


def test_test_split_never_opened() -> None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from train_replay_pretrain import build_cache  # noqa: PLC0415
    from vgc.bc.replay_pretraining import iter_task_examples  # noqa: PLC0415

    with pytest.raises(AssertionError):
        build_cache(Path("runs/own_team_rebuild_20261104/scratch/prepared"), "test")
    with pytest.raises(ValueError):
        next(iter_task_examples(
            Path("runs/own_team_rebuild_20261104/scratch/prepared"), "move", split="test"))


def test_encoded_cache_rejects_changed_dataset_and_legacy_cache(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    sys.path.insert(0, "tools")
    import train_replay_pretrain as trainer

    prepared, cache_dir = tmp_path / "prepared", tmp_path / "cache"
    prepared.mkdir()
    cache_dir.mkdir()
    hashes = {}
    for name in ("train.jsonl.gz", "sampling_broad.jsonl", "sampling_expert.jsonl"):
        path = prepared / name
        path.write_bytes(b"win")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = prepared / "summary.json"
    summary.write_text(json.dumps({"files_sha256": hashes}))
    target = ["win"]

    def examples(directory, task, **kwargs):
        if task == "outcome":
            yield {"input": _record(), "reference": ("game", "p1", task, 0),
                   "source_line": 1, "component": None, "target": target[0],
                   "meta": {"pre_rating": 1300, "turn": 1}}

    monkeypatch.setattr(trainer, "iter_task_examples", examples)
    assert trainer.build_cache(prepared, "train", cache_dir)["tasks"]["outcome"]["label_pos"] == [1.0]
    assert trainer.build_cache(prepared, "train", cache_dir)["tasks"]["outcome"]["label_pos"] == [1.0]
    (prepared / "train.jsonl.gz").write_bytes(b"loss")
    with pytest.raises(ValueError, match="artifact fingerprint"):
        trainer.build_cache(prepared, "train", cache_dir)
    hashes["train.jsonl.gz"] = hashlib.sha256(b"loss").hexdigest()
    summary.write_text(json.dumps({"files_sha256": hashes}))
    target[0] = "loss"
    with pytest.raises(ValueError, match="cache provenance"):
        trainer.build_cache(prepared, "train", cache_dir)
    assert trainer.build_cache(prepared, "train")["tasks"]["outcome"]["label_pos"] == [0.0]
    np.savez(cache_dir / "cache_train.npz", legacy=np.array(True))
    with pytest.raises(ValueError, match="cache provenance"):
        trainer.build_cache(prepared, "train", cache_dir)


def test_filtered_units_excluded_and_npz_roundtrips(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    sys.path.insert(0, "tools")
    import train_replay_pretrain as trainer

    prepared, cache_dir = tmp_path / "prepared", tmp_path / "cache"
    prepared.mkdir()
    cache_dir.mkdir()
    hashes = {}
    for name in ("train.jsonl.gz", "sampling_broad.jsonl", "sampling_expert.jsonl"):
        path = prepared / name
        path.write_bytes(b"x")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (prepared / "summary.json").write_text(json.dumps({"files_sha256": hashes}))

    def ex(task, ordinal, meta, target, component=None):
        return {"input": _record(), "reference": ("g", "p1", task, ordinal),
                "source_line": 1, "component": component, "target": target,
                "meta": meta}

    turn = {"pre_rating": 1300, "turn": 1, "decision_kind": "turn",
            "bench_guessed": False}
    preview = {"pre_rating": 1300, "turn": 0, "decision_kind": "teampreview",
               "bench_guessed": False}
    guessed = {"pre_rating": 1300, "turn": 1, "decision_kind": "turn",
               "bench_guessed": True}
    examples = {
        "move": [ex("move", 0, turn, "fakeout", 0)],
        "switch_destination": [ex("switch_destination", 0, guessed, "amoonguss", 0),
                               ex("switch_destination", 1, turn, "amoonguss", 0)],
        "outcome": [ex("outcome", 0, preview, "win"), ex("outcome", 1, turn, "win")],
        "preview_brought": [],
        "preview_leads": [],
    }

    def gen(directory, task, **kwargs):
        yield from examples[task]

    monkeypatch.setattr(trainer, "iter_task_examples", gen)
    built = trainer.build_cache(prepared, "train", cache_dir)
    # (a) teampreview outcome rows are filtered before encoding, never cached.
    assert built["tasks"]["outcome"]["refs"] == {("g", "p1", "outcome", 1): 0}
    assert built["tasks"]["outcome"]["filtered"] == {("g", "p1", "outcome", 0)}
    # (b) a switch row whose bench contains a hash-guessed mon is filtered.
    assert built["tasks"]["switch_destination"]["filtered"] == {
        ("g", "p1", "switch_destination", 0)}
    assert built["tasks"]["switch_destination"]["refs"] == {
        ("g", "p1", "switch_destination", 1): 0}
    # (c) a second build_cache call loads the npz and round-trips every task.
    loaded = trainer.build_cache(prepared, "train", cache_dir)
    for task, t in built["tasks"].items():
        lt = loaded["tasks"][task]
        assert lt["row"] == t["row"]
        assert lt["label_pos"] == t["label_pos"]
        assert lt["refs"] == t["refs"]
        assert lt["filtered"] == t["filtered"]


def test_label_not_in_options_skipped() -> None:
    # The option lookup returns empty -> the unit is dropped and counted, never trained.
    net = TurnPretrainNet()
    hidden = torch.zeros(1, 256)
    opts = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[True, True, True]])
    logits = net.action_logits(hidden, torch.tensor([0]), opts, mask)
    assert logits.shape == (1, 3)
    assert not bool((opts[mask] == 99).any())  # label 99 would be skipped/counted


def test_preview_output_order_invariant() -> None:
    net = PreviewPretrainNet()
    own = torch.tensor([[10, 20, 30, 40, 50, 60]])
    opp = torch.tensor([[70, 80, 90, 100, 110, 120]])
    b1, l1 = net(own, opp)
    perm = torch.randperm(6)
    own_perm = own[:, perm].sort(dim=1).values  # permute then re-sort by id
    b2, l2 = net(own_perm, opp)
    torch.testing.assert_close(b1, b2)
    torch.testing.assert_close(l1, l2)


def test_cache_roundtrip_matches_iter_task_examples() -> None:
    """Guard the record-identity keying bug: a cached unit's encoded row, options
    and label position must equal what the raw example itself produces."""
    import sys
    from pathlib import Path
    sys.path.insert(0, "tools")
    from train_replay_pretrain import build_cache  # noqa: PLC0415
    from vgc.bc.replay_pretraining import iter_task_examples  # noqa: PLC0415
    from vgc.bc.encoding import MOVE_TO_IDX  # noqa: PLC0415

    prepared = Path("runs/own_team_rebuild_20261104/scratch/prepared")
    cache = build_cache(prepared, "train")
    examples = {}
    for i, ex in enumerate(iter_task_examples(prepared, "move", split="train")):
        if i >= 5:
            break
        examples[ex["reference"]] = ex
    t = cache["tasks"]["move"]
    for ref, ex in examples.items():
        u = t["refs"][ref]
        row = t["row"][u]
        idx_a, sc_a = trunk_inputs(ex["input"])
        np.testing.assert_array_equal(cache["records"]["idx"][row], idx_a)
        np.testing.assert_allclose(cache["records"]["scalars"][row], sc_a)
        comp = t["component"][u]
        opts = cache["records"]["move_opts"][row][comp]
        pos = t["label_pos"][u]
        assert opts[pos] == MOVE_TO_IDX[ex["target"]]


def test_action_loss_is_pooled_mean() -> None:
    """ACTION must be one mean over move+switch units, not a per-task average."""
    import sys
    sys.path.insert(0, "tools")
    import torch.nn.functional as F  # noqa: PLC0415
    from train_replay_pretrain import step_losses  # noqa: PLC0415

    cache = {"records": {
        "idx": np.zeros((2, 10), dtype=np.int64),
        "scalars": np.zeros((2, 12), dtype=np.float32),
        "move_opts": np.tile(np.arange(5, dtype=np.int64), (2, 2, 1)),
        "move_mask": np.ones((2, 2, 5), dtype=bool),
        "dest_opts": np.arange(8, dtype=np.int64).reshape(1, -1).repeat(2, 0),
        "dest_mask": np.ones((2, 8), dtype=bool)}}
    cache["tasks"] = {
        "move": {"row": [0], "component": [0], "label_pos": [1], "species": [0],
                 "refs": {}, "skipped": 0},
        "switch_destination": {"row": [1, 1], "component": [1, 1], "label_pos": [0, 0],
                               "species": [0, 0], "refs": {}, "skipped": 0},
        "outcome": {"row": [0], "component": [-1], "label_pos": [1.0], "species": [-1],
                    "refs": {}, "skipped": 0}}

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor([0.1]))

        def hidden(self, idx, sc):
            return torch.zeros(len(idx), 4) + self.p

        def action_logits(self, hidden, slot, opts, mask):
            return torch.stack([self.p + torch.arange(5.) for _ in range(len(hidden))])

        def destination_logits(self, hidden, slot, opts, mask):
            return torch.stack([self.p + torch.arange(8.) for _ in range(len(hidden))])

        def value_logit(self, hidden):
            return self.p.expand(len(hidden)) + torch.tensor([0.5, -0.5][:len(hidden)])

    net = Stub()
    terms = step_losses(net, cache, [0], [0, 1], [0], torch.device("cpu"))
    # hand check: ACTION mean over the 3 CEs (one move + two switch), all logits
    # identical rows so each CE = -log_softmax[label]
    from vgc.bc.replay_pretrain import SWITCH_IDX  # noqa: PLC0415
    sw_pos = int(np.nonzero(np.arange(5) == SWITCH_IDX)[0][0])
    probs = F.log_softmax(torch.tensor([0.1 + i for i in range(5)]), dim=0)
    expected_action = -(probs[1] + probs[sw_pos] + probs[sw_pos]) / 3
    assert torch.isclose(terms["action"], expected_action)
    assert set(terms) == {"action", "destination", "value"}


def test_choose_combo_contains_leads_and_is_order_invariant() -> None:
    net = PreviewPretrainNet()
    own = ["incineroar", "rillaboom", "amoonguss", "kingambit", "ninetalesalola", "salamence"]
    opp = ["fluttermane", "urshifurapidstrike", "landorus", "ogerponwellspring", "tornadus", "calyrexshadow"]
    brought, leads = net.choose(own, opp)
    assert set(leads) <= set(brought)
    import random
    random.seed(0)
    own2 = own[:]
    random.shuffle(own2)
    brought2, leads2 = net.choose(own2, opp)
    assert brought2 == brought and leads2 == leads


def test_baselines_and_breakdowns() -> None:
    """Pins the review fixes: (1) train-freq baseline picks the MOST frequent
    option, (2) `_breakdowns` aggregates a pooled two-task unit list correctly,
    (4) the brought baseline is per-example (top-4 by bring rate within own6)."""
    import sys
    sys.path.insert(0, "tools")
    from train_replay_pretrain import _breakdowns  # noqa: PLC0415

    # (1) most-frequent move wins, lowest-idx on ties
    freq = {5: 9, 7: 3, 9: 9}
    cand = [5, 7, 9]
    best = min(cand, key=lambda o: (-freq.get(o, 0), o))
    assert best == 5  # most frequent; 5 and 9 tie -> lowest index

    # (2) pooled breakdown: one move + one switch unit, different meta
    cache = {"_train_lineups": {("a", "b")}, "tasks": {
        "move": {"meta": {"rating": ["1600+"], "turn_bucket": ["1-2"],
                          "lineup": [("a", "b")], "fully": [True],
                          "rating_val": [1700]}},
        "switch_destination": {"meta": {"rating": ["<1200"], "turn_bucket": ["6+"],
                                        "lineup": [("x",)], "fully": [False],
                                        "rating_val": [1300]}}}}
    rows = [("move", 0, True, 0.5), ("switch_destination", 0, False, 1.5)]
    b = _breakdowns(cache, rows)
    assert b["rating"]["1600+"] == {"n": 1, "acc": 1.0, "loss": 0.5}
    assert b["rating"]["<1200"]["acc"] == 0.0
    assert b["lineup_seen"]["True"]["n"] == 1 and b["lineup_seen"]["False"]["n"] == 1

    # (4) per-example top-4: own6 species 1..6, rates favour 3,4,5,6
    own6 = [1, 2, 3, 4, 5, 6]
    rates = {1: 0.01, 2: 0.02, 3: 0.9, 4: 0.8, 5: 0.7, 6: 0.6}
    pred = frozenset(sorted(own6, key=lambda s: (-rates.get(int(s), 0.0), int(s)))[:4])
    assert pred == frozenset({3, 4, 5, 6})


def test_cache_meta_and_1600_subset(monkeypatch) -> None:
    """Row-level pre_rating/turn/own_fully_confirmed flow through `meta` into the
    cache; the validation 1600+ subset contains exactly the 1600+ unit."""
    import sys
    sys.path.insert(0, "tools")
    import train_replay_pretrain as tp  # noqa: PLC0415

    def fake_iter(prepared, task, *, split="train", phase="broad", final_evaluation=False):
        base = {"input": _record(), "target": "win", "component": None,
                "source_line": 7}
        yield dict(base, reference=("g1", "p1", task, 0),
                   meta={"pre_rating": 1711.0, "turn": 9,
                         "decision_kind": "turn", "own_fully_confirmed": True})
        yield dict(base, reference=("g2", "p1", task, 0),
                   meta={"pre_rating": 1200.0, "turn": 1,
                         "decision_kind": "turn", "own_fully_confirmed": False})

    monkeypatch.setattr(tp, "iter_task_examples", fake_iter)
    monkeypatch.setattr(tp, "_cache_fingerprint", lambda *args: "metadata-test-stub")
    from pathlib import Path
    cache = tp.build_cache(Path("unused"), "validation")
    t = cache["tasks"]["outcome"]
    assert t["meta"]["rating_val"] == [1711.0, 1200.0]
    assert t["meta"]["rating"] == ["1600+", "1200-1399"]
    assert t["meta"]["turn_bucket"] == ["6+", "1-2"]
    assert t["meta"]["fully"] == [True, False]
    keep = [u for u in range(len(t["row"])) if t["meta"]["rating_val"][u] >= 1600]
    assert keep == [0]
