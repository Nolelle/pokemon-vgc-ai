"""Pure-Python / PyTorch unit tests for the Rust search & trainer logic.

These tests intentionally avoid importing or instantiating `pikapro` so they
always run in CI's `unit` job even when the optional patched Rust engine is
not installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vgc.battle_memory import BattleMemory
from vgc.rl.rust_search import (
    _dedup_preview_groups,
    _our_public_knowledge,
    _solve_2x2_zero_sum,
)


def test_turn2_nash_solver_and_preview_dedup() -> None:
    """Verify exact 2x2 zero-sum Nash values and symmetric preview deduplication."""
    # Pure saddle point: row 0 guarantees +0.4 against both opp responses.
    assert abs(_solve_2x2_zero_sum([[0.4, 0.8], [-0.2, 0.9]]) - 0.4) < 1e-9
    # Symmetric matching pennies (Protect vs Attack mindgame): exact mixed Nash value is 0.0.
    assert abs(_solve_2x2_zero_sum([[0.5, -0.5], [-0.5, 0.5]]) - 0.0) < 1e-9
    # Asymmetric mixed game: [[3, 0], [1, 2]] -> v = (6 - 0) / (3 - 0 - 1 + 2) = 1.5.
    assert abs(_solve_2x2_zero_sum([[3.0, 0.0], [1.0, 2.0]]) - 1.5) < 1e-9
    # 3x3 zero-sum Rock-Paper-Scissors value must be ~0.0 (not truncated to 2x2!).
    rps = [[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]
    assert abs(_solve_2x2_zero_sum(rps)) < 0.05

    # Back-order duplicates (0,1,2,3) and (0,1,3,2) + lead swap (1,0,2,3) merge and sum probs.
    orders = np.array([[0, 1, 2, 3], [0, 1, 3, 2], [1, 0, 2, 3], [0, 2, 1, 3]], dtype=np.int64)
    grouped = _dedup_preview_groups([0, 1, 2, 3], [0.30, 0.25, 0.15, 0.20], orders)
    assert len(grouped) == 2
    assert grouped[0][0] == 0 and abs(grouped[0][2] - 0.70) < 1e-9
    assert grouped[1][0] == 3 and abs(grouped[1][2] - 0.20) < 1e-9


def test_our_public_knowledge_reveals_only_public_protocol_facts() -> None:
    """Simulated opponent sees our publicly revealed items/abilities/moves, never hidden ones."""
    packed = (
        "Tyranitar||AssaultVest|SandStream|RockSlide,Crunch,LowKick,Protect|Adamant|32,32,,,,2||||50|]"
        "Sneasler||FocusSash|Unburden|FakeOut,CloseCombat,DireClaw,Protect|Jolly|,32,,,,32||||50|]"
        "Gholdengo||LifeOrb|GoodAsGold|MakeItRain,ShadowBall,NastyPlot,Protect|Modest|,,2,32,,32||||50|]"
        "Incineroar||SafetyGoggles|Intimidate|FakeOut,FlareBlitz,KnockOff,PartingShot|Careful|32,,16,,18,||||50|"
    )
    mem = BattleMemory("test", our_role="p1")
    mem.observe_protocol(
        [
            ["", "switch", "p1a: Tyranitar", "Tyranitar, L50, M", "100/100"],
            ["", "switch", "p1b: Sneasler", "Sneasler, L50, M", "100/100"],
            ["", "-weather", "Sandstorm", "[from] ability: Sand Stream", "[of] p1a: Tyranitar"],
            ["", "move", "p1b: Sneasler", "Fake Out", "p2a: Pelipper"],
        ]
    )
    battle = SimpleNamespace(team={})
    items, abilities, moves, revealed = _our_public_knowledge(battle, packed, memory=mem)
    # Tyranitar: revealed + Sand Stream known from weather line, item still hidden.
    assert revealed[0] is True
    assert abilities[0] == "sandstream"
    assert items[0] is None
    # Sneasler: revealed + Fake Out known, Focus Sash & Unburden still hidden (never leaked!).
    assert revealed[1] is True
    assert moves[1] == ["fakeout"]
    assert items[1] is None
    assert abilities[1] is None
    # Gholdengo: unrevealed on bench, single-ability Good as Gold known, Life Orb hidden.
    assert revealed[2] is False
    assert abilities[2] == "goodasgold"
    assert items[2] is None
    # Incineroar: unrevealed on bench, Safety Goggles & Intimidate hidden.
    assert revealed[3] is False
    assert items[3] is None
    assert abilities[3] is None


def test_train_rust_vec_lr_scaling_decay_groups_and_ema_pfsp() -> None:
    """Pure-PyTorch trainer invariants: width-scaled LR, decay groups, and PFSP EMA."""
    pytest.importorskip("torch")
    from torch import nn

    selfplay_dir = str(Path(__file__).resolve().parents[1] / "selfplay")
    if selfplay_dir not in sys.path:
        sys.path.insert(0, selfplay_dir)
    import train_rust_vec as trv

    # Inverse width/FFN LR scaling + explicit override.
    assert trv.resolve_learning_rate(None, width=192, ffn_mult=2, is_warm_start=False) == pytest.approx(1e-4)
    assert trv.resolve_learning_rate(None, width=256, ffn_mult=3, is_warm_start=False) == pytest.approx(5e-5)
    assert trv.resolve_learning_rate(None, width=192, ffn_mult=2, is_warm_start=True) == pytest.approx(3e-5)
    assert trv.resolve_learning_rate(None, width=256, ffn_mult=3, is_warm_start=True) < 2.5e-5
    assert trv.resolve_learning_rate(4.2e-5, width=256, ffn_mult=3, is_warm_start=False) == pytest.approx(4.2e-5)

    # Embeddings, LayerNorm, and biases must get 0 weight decay; Linear weights keep decay.
    class _ToyNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(8, 16)
            self.norm = nn.LayerNorm(16)
            self.proj = nn.Linear(16, 4)

    net = _ToyNet()
    opt = trv.build_optimizer(net, lr=1e-4, weight_decay=1e-2)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 1e-2
    assert opt.param_groups[1]["weight_decay"] == 0.0
    decay_ids = {id(p) for p in opt.param_groups[0]["params"]}
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    assert id(net.proj.weight) in decay_ids
    assert id(net.proj.bias) in no_decay_ids
    assert id(net.emb.weight) in no_decay_ids
    assert id(net.norm.weight) in no_decay_ids

    # PFSP EMA window caps effective history and responds to improvement.
    snap = trv.Snapshot(0, Path("dummy.pt"))
    for _ in range(300):
        snap.record_game(0.0)
    assert snap.games <= 200.1
    for _ in range(200):
        snap.record_game(1.0)
    assert snap.priority < 0.25
