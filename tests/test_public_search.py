"""Shared public-search boundary and deterministic-resource contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from vgc.actions import describe_order
from vgc.config import REPO_ROOT
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.public_search import (
    PUBLIC_SEARCH_CONTRACT_VERSION,
    public_information_exact_search,
    public_search_randomness_key,
)


def _empty_observation(tag: str, private_root: object) -> SimpleNamespace:
    return SimpleNamespace(
        battle_tag=tag,
        _vgc_direct_root=private_root,
        _vgc_direct_side="p2",
        team={},
        opponent_team={},
        active_pokemon=(),
        opponent_active_pokemon=(),
        turn=3,
        gen=9,
        format="gen9championsvgc2026regmb",
    )


def test_randomness_key_ignores_battle_name_and_attached_private_root() -> None:
    first = _empty_observation("evaluation-a", object())
    second = _empty_observation("evaluation-b", object())

    assert public_search_randomness_key(first, "exact-own-team") == (
        public_search_randomness_key(second, "exact-own-team")
    )


def test_public_reveal_changes_randomness_key() -> None:
    hidden = _empty_observation("same", object())
    revealed = _empty_observation("same", object())
    revealed.opponent_team = {
        "p2: foe": SimpleNamespace(species="charizard", item="lifeorb", moves={})
    }

    assert public_search_randomness_key(hidden, "exact-own-team") != (
        public_search_randomness_key(revealed, "exact-own-team")
    )


def test_public_search_never_uses_private_root_and_closes_owned_resources(monkeypatch) -> None:
    private_root = object()
    observation = _empty_observation("offline", private_root)
    belief = SimpleNamespace(weight=1.0)
    built_root = SimpleNamespace(close_calls=0)
    built_root.close = lambda: setattr(built_root, "close_calls", built_root.close_calls + 1)
    mirrors = []
    exact_calls = []

    class FakeMirror:
        def __init__(self, own_team, config):
            self.own_team = own_team
            self.config = config
            self.close_calls = 0
            self.last_hypothesis_audit = {"searched": 1}
            mirrors.append(self)

        def hypotheses(self, battle, memory):
            assert battle is observation
            return [belief]

        def build(self, battle, hypothesis):
            assert battle is observation and hypothesis is belief
            return built_root

        def close(self):
            self.close_calls += 1

    def fake_exact(root, side, config, *, candidate_selector, randomness_key):
        exact_calls.append((root, side, candidate_selector, randomness_key))
        return [SimpleNamespace(order=SimpleNamespace(message="public-action"), score=7.0, breakdown={})]

    monkeypatch.setattr("vgc.rl.public_search.LiveExactMirror", FakeMirror)
    monkeypatch.setattr("vgc.rl.public_search.search_joint_orders_exact", fake_exact)
    audit = {}

    scored = public_information_exact_search(
        observation,
        PolicyConfig(),
        "exact-own-team",
        candidate_selector=None,
        audit=audit,
    )

    assert scored[0].order.message == "public-action"
    assert exact_calls[0][0] is built_root
    assert exact_calls[0][0] is not private_root
    assert exact_calls[0][1] == "p1"
    assert built_root.close_calls == 1
    assert mirrors[0].close_calls == 1
    assert audit["information_contract"] == PUBLIC_SEARCH_CONTRACT_VERSION
    assert audit["searched"] == 1


def test_public_search_closes_resources_when_exact_search_raises(monkeypatch) -> None:
    observation = _empty_observation("failure", object())
    built_root = SimpleNamespace(close_calls=0)
    built_root.close = lambda: setattr(built_root, "close_calls", built_root.close_calls + 1)
    mirror = SimpleNamespace(
        last_hypothesis_audit={},
        hypotheses=lambda _battle, _memory: [SimpleNamespace(weight=1.0)],
        build=lambda _battle, _belief: built_root,
        close_calls=0,
    )
    mirror.close = lambda: setattr(mirror, "close_calls", mirror.close_calls + 1)

    def fail(*_args, **_kwargs):
        raise RuntimeError("branch failed")

    monkeypatch.setattr("vgc.rl.public_search.search_joint_orders_exact", fail)
    with pytest.raises(RuntimeError, match="branch failed"):
        public_information_exact_search(
            observation, PolicyConfig(), "exact-own-team", mirror=mirror
        )

    assert built_root.close_calls == 1
    assert mirror.close_calls == 0  # caller-owned worker remains reusable


def _changed_private_team(team: str) -> str:
    """Same public roster, different unrevealed move, item, spread, and nature."""

    changed = team.replace("CharizarditeY", "CharizarditeX", 1)
    changed = changed.replace("HeatWave,WeatherBall", "Flamethrower,WeatherBall", 1)
    changed = changed.replace("|Modest|10,,,32,,24|", "|Timid|32,,,,,32|", 1)
    assert changed != team
    return changed


@pytest.mark.integration
@pytest.mark.parametrize("seat", ["p1", "p2"])
def test_public_search_is_invariant_to_opponent_private_set_for_both_seats(seat: str) -> None:
    """The same visible turn must give the same hypotheses, scores, and action."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = (Path(REPO_ROOT) / "teams" / "meta1.packed.txt").read_text().strip()
    changed = _changed_private_team(team)
    config = replace(
        PolicyConfig(log_decisions=False),
        search_our_candidates=2,
        search_opp_candidates=1,
        exact_search_future_samples=1,
        exact_search_state_hypotheses=1,
        exact_search_spread_hypotheses=1,
        exact_search_set_hypotheses=1,
        exact_search_bring_hypotheses=1,
        exact_search_total_hypotheses=1,
    )

    def evaluate(source_worker, battle_id: str, changed_side: str):
        teams = {"p1": team, "p2": team}
        teams[changed_side] = changed
        source = DirectBattle.start(
            source_worker,
            battle_id,
            teams["p1"],
            teams["p2"],
            seed=[81, 82, 83, 84],
        )
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            observation = source.battles[seat]
            # This is the exact shortcut that existed in offline evaluation.  Its
            # presence must have no effect on the shared public path.
            observation._vgc_direct_root = source
            observation._vgc_direct_side = seat
            audit = {}
            scored = public_information_exact_search(
                observation, config, team, audit=audit
            )
            return (
                audit,
                [(describe_order(entry.order), entry.score) for entry in scored],
            )
        finally:
            source.close()

    opponent_side = "p2" if seat == "p1" else "p1"
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        baseline = evaluate(source_worker, f"public-{seat}-baseline", "none")
        private_changed = evaluate(source_worker, f"public-{seat}-changed", opponent_side)

    assert baseline == private_changed
    assert baseline[1]

