"""The live exact mirror must let the OPPONENT Mega Evolve in simulated branches.

poke-env has no `opponent_can_mega_evolve`, so the public snapshot told the worker the foe
could never Mega, and the exact judge searched every opponent reply as a non-Megaing
Pokemon. These tests build mirrors from a fogged view of a real direct battle and assert
what Showdown itself offers the foe: Mega orders while its Mega is unspent, none once it
has been used, the right forme and stone for a foe that already evolved, and no Mega at
all under the legacy `exact_mirror_opponent_mega=False` control.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vgc.actions import enumerate_joint_orders
from vgc.config import REPO_ROOT
from vgc.evaluator import score_joint_orders
from vgc.mechanics_state import _side_snapshot, snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.live_mirror import LiveExactMirror

META1 = REPO_ROOT / "teams" / "meta1.packed.txt"

# Single hypothesis everywhere: these tests are about what the root offers, not belief mixing.
CONFIG = replace(
    PolicyConfig(),
    exact_search_state_hypotheses=1,
    exact_search_spread_hypotheses=1,
    exact_search_set_hypotheses=1,
    exact_search_bring_hypotheses=1,
    exact_search_total_hypotheses=1,
)
LEGACY_CONFIG = replace(CONFIG, exact_mirror_opponent_mega=False)
_MEGA = re.compile(r"\bmega\b")


def _mega_orders(battle) -> list:
    return [order for order in enumerate_joint_orders(battle) if _MEGA.search(order.message)]


def _worker_side(root: DirectBattle, side: str) -> dict:
    index = 0 if side == "p1" else 1
    state = root.worker.request({"cmd": "dump", "id": root.battle_id})["state"]
    return state["sides"][index]


def _opponent_team(stone: str) -> str:
    if not META1.is_file():
        pytest.skip("meta1 packed team is unavailable")
    return META1.read_text().strip().replace("CharizarditeY", stone)


@contextmanager
def _fogged_source(
    *, opponent_stone: str = "CharizarditeY", opponent_mega_on_turn_1: bool = False
) -> Iterator[tuple[DirectBattle, str]]:
    """A turn-2 (or turn-1) direct battle seen from p1; p2 holds the stone given."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    own_team = _opponent_team("CharizarditeY")
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source = DirectBattle.start(
            worker, "opp-mega-source", own_team, _opponent_team(opponent_stone),
            seed=[1, 2, 3, 4],
        )
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            if opponent_mega_on_turn_1:
                mega = _mega_orders(source.battles["p2"])
                assert mega, "the source foe must be able to Mega on turn 1"
                source.step(
                    {
                        "p1": choice_string(enumerate_joint_orders(source.battles["p1"])[0]),
                        "p2": choice_string(mega[0]),
                    }
                )
                assert source.battles["p1"].opponent_used_mega_evolve
            yield source, own_team
        finally:
            source.close()


@contextmanager
def _mirror_root(source: DirectBattle, own_team: str, config: PolicyConfig):
    mirror = LiveExactMirror(own_team, config)
    root = None
    try:
        root = mirror.build(source.battles["p1"])
        yield root
    finally:
        if root is not None:
            root.close()
        mirror.close()


@pytest.mark.integration
def test_unrevealed_opponent_can_mega_and_a_branch_executes_it() -> None:
    with _fogged_source() as (source, own_team):
        assert not source.battles["p1"].opponent_used_mega_evolve
        with _mirror_root(source, own_team, CONFIG) as root:
            mega = _mega_orders(root.battles["p2"])
            assert mega, "the foe's prior stone should make a Mega order legal"
            ours = choice_string(enumerate_joint_orders(root.battles["p1"])[0])
            clone = root.clone("opp-mega-branch")
            try:
                result = clone.step({"p1": ours, "p2": choice_string(mega[0])})
                text = "\n".join(result.lines["p1"])
                assert re.search(r"\|-mega\|p2[ab]: Charizard", text), text
                after = snapshot_battle(clone.battles["p1"])
                assert after.opponent_side.used_mega_evolution
                assert any(
                    mon.species_id == "charizardmegay" for mon in after.opponent_side.pokemon
                )
            finally:
                clone.close()
            # The foe offered exactly one Mega-capable active (Charizard); the bench stone
            # (Venusaur) is a switch-in away, so it must also be live in the engine.
            side = _worker_side(root, "p2")
            capable = [mon["item"] for mon in side["pokemon"] if mon.get("canMegaEvo")]
            assert "charizarditey" in capable


@pytest.mark.integration
def test_no_opponent_mega_anywhere_once_its_mega_is_used() -> None:
    with _fogged_source(opponent_mega_on_turn_1=True) as (source, own_team):
        with _mirror_root(source, own_team, CONFIG) as root:
            assert not _mega_orders(root.battles["p2"])
            side = _worker_side(root, "p2")
            assert side["megaEvoUsed"] is True
            # Active and bench alike: the one Mega per side is spent.
            assert not [mon for mon in side["pokemon"] if mon.get("canMegaEvo")]
            for order in enumerate_joint_orders(root.battles["p2"]):
                assert not _MEGA.search(order.message)


@pytest.mark.integration
def test_already_megaed_opponent_keeps_its_forme_and_its_real_stone() -> None:
    # The corpus prior for Charizard is overwhelmingly the Y stone; this foe holds X.
    with _fogged_source(
        opponent_stone="CharizarditeX", opponent_mega_on_turn_1=True
    ) as (source, own_team):
        observed = snapshot_battle(source.battles["p1"], opponent_mega_unknown=True)
        assert any(mon.species_id == "charizardmegax" for mon in observed.opponent_side.pokemon)
        with _mirror_root(source, own_team, CONFIG) as root:
            side = _worker_side(root, "p2")
            charizard = next(mon for mon in side["pokemon"] if mon["item"].startswith("charizard"))
            assert charizard["item"] == "charizarditex"
            # Showdown's own state: the Mega-X forme with its X ability, not Mega-Y.
            assert charizard["species"] == "[Species:charizardmegax]"
            assert charizard["ability"] == "toughclaws"


@pytest.mark.integration
def test_legacy_knob_false_never_lets_the_opponent_mega() -> None:
    with _fogged_source() as (source, own_team):
        with _mirror_root(source, own_team, LEGACY_CONFIG) as root:
            assert not _mega_orders(root.battles["p2"])
            assert _mega_orders(root.battles["p1"]), "our own side must still be able to Mega"


@pytest.mark.integration
def test_opponent_reply_shortlist_holds_distinct_plans() -> None:
    """The judge's N opp replies are N distinct plans, not Mega/non-Mega twins."""

    from vgc.rl.exact_search import _opponent_replies

    config = replace(CONFIG, search_opp_candidates=CONFIG.exact_judge_opp_candidates)
    with _fogged_source() as (source, own_team):
        with _mirror_root(source, own_team, CONFIG) as root:
            ranked = score_joint_orders(root.battles["p2"], config)
            legacy = _opponent_replies(
                ranked, replace(config, exact_search_dedupe_mega_replies=False)
            )
            deduped = _opponent_replies(ranked, config)
            plans = {_MEGA.sub("", entry.order.message) for entry in deduped}
            assert len(deduped) == config.search_opp_candidates
            assert len(plans) == len(deduped)
            assert len(legacy) == len(deduped)


def test_snapshot_reports_opponent_mega_as_unknown_until_used() -> None:
    battle = SimpleNamespace(
        opponent_team={},
        opponent_active_pokemon=[],
        opponent_used_mega_evolve=False,
        opponent_can_mega_evolve=[True, False],
    )
    legacy = _side_snapshot(battle, opponent=True)
    assert legacy.can_mega_evolve == (True, False)
    unknown = _side_snapshot(battle, opponent=True, opponent_mega_unknown=True)
    assert unknown.can_mega_evolve == (None, None)
    assert unknown.used_mega_evolution is False
    battle.opponent_used_mega_evolve = True
    used = _side_snapshot(battle, opponent=True, opponent_mega_unknown=True)
    assert used.can_mega_evolve == (False, False)
    assert used.used_mega_evolution is True
