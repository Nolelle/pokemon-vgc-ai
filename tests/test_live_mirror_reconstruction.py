"""The live mirror must rebuild what the public patch used to drop.

Each test plays a real direct battle, builds a `LiveExactMirror` root from p1's fogged
view, and compares what Showdown holds in the mirror against what the real battle holds:
foe HP scale, hidden items, Mega-forme stats, Fake Out reuse, the Choice lock, the Toxic
stage, Disable's target move and Unburden. The legacy knob (False) must reproduce the old
reconstruction so same-session A/Bs stay possible.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vgc.actions import enumerate_joint_orders
from vgc.battle_memory import BattleMemory
from vgc.config import REPO_ROOT
from vgc.exact_judge import judge_config, with_exact_timers
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker
from vgc.rl.exact_search import _opponent_replies
from vgc.rl.live_mirror import LiveExactMirror
from vgc.stats import calculate_stats

META1 = REPO_ROOT / "teams" / "meta1.packed.txt"

CONFIG = replace(
    PolicyConfig(),
    exact_search_state_hypotheses=1,
    exact_search_spread_hypotheses=1,
    exact_search_set_hypotheses=1,
    exact_search_bring_hypotheses=1,
    exact_search_total_hypotheses=1,
)
LEGACY = replace(
    CONFIG,
    exact_mirror_hp_scale=False,
    exact_mirror_keep_hidden_items=False,
    exact_mirror_mega_stats=False,
    exact_mirror_restore_state=False,
)


def _mons() -> dict[str, str]:
    if not META1.is_file():
        pytest.skip("meta1 packed team is unavailable")
    return {entry.split("|")[0]: entry for entry in META1.read_text().strip().split("]")}


def _team(*entries: str) -> str:
    return "]".join(entries)


def _sets() -> tuple[str, str]:
    """p1 leads Farigiraf (Toxic, Disable) + Sylveon; p2 leads Incineroar + Scarf Garchomp."""

    by = _mons()
    farigiraf = by["Farigiraf"].replace(
        "Psychic,Thunderbolt,HelpingHand,TrickRoom", "Toxic,Disable,HelpingHand,Protect"
    )
    p1 = _team(
        farigiraf, by["Sylveon"], by["Venusaur"], by["Garchomp"], by["Incineroar"], by["Charizard"]
    )
    p2 = _team(
        by["Incineroar"],
        by["Garchomp"].replace("LifeOrb", "ChoiceScarf"),
        by["Charizard"], by["Farigiraf"], by["Venusaur"], by["Sylveon"],
    )
    return p1, p2


@contextmanager
def _battle(
    p1: str, p2: str, turns: list[tuple[str, str]], *, seed: tuple[int, ...] = (5, 6, 7, 8)
) -> Iterator[tuple[DirectBattle, BattleMemory]]:
    """A direct battle after team preview and ``turns``, plus a memory fed p1's protocol."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as worker:
        source = DirectBattle.start(worker, "reconstruction-source", p1, p2, seed=list(seed))
        try:
            memory = BattleMemory(battle_tag="reconstruction", our_role="p1")
            result = source.step({"p1": "team 1234", "p2": "team 1234"})
            memory.observe_protocol([line.split("|") for line in result.lines["p1"]])
            for p1_choice, p2_choice in turns:
                result = source.step({"p1": p1_choice, "p2": p2_choice})
                assert result.error is None, result.error
                memory.observe_protocol([line.split("|") for line in result.lines["p1"]])
            setattr(source.battles["p1"], "_vgc_battle_memory", memory)
            yield source, memory
        finally:
            source.close()


@contextmanager
def _root(source: DirectBattle, own_team: str, config: PolicyConfig = CONFIG):
    mirror = LiveExactMirror(own_team, config)
    root = None
    try:
        root = mirror.build(source.battles["p1"])
        yield root
    finally:
        if root is not None:
            root.close()
        mirror.close()


def _side(root: DirectBattle, side: str) -> list[dict]:
    state = root.worker.request({"cmd": "dump", "id": root.battle_id})["state"]
    return state["sides"][0 if side == "p1" else 1]["pokemon"]


def _mon(root: DirectBattle, side: str, species: str) -> dict:
    return next(
        mon for mon in _side(root, side) if mon["species"] == f"[Species:{species}]"
    )


def _messages(root: DirectBattle, side: str) -> set[str]:
    return {order.message for order in enumerate_joint_orders(root.battles[side])}


# --- 1. HP scale ---------------------------------------------------------------------


@pytest.mark.integration
def test_foe_hp_percent_is_converted_to_the_mirror_sets_max_hp() -> None:
    p1, p2 = _sets()
    by_turn = [("move protect, move hypervoice", "move darkestlariat 2, move protect")]
    with _battle(p1, p2, by_turn) as (source, _memory):
        view = source.battles["p1"]
        public = {
            mon.species: round(mon.current_hp_fraction * 100)
            for mon in view.opponent_active_pokemon
        }
        assert any(percent < 100 for percent in public.values()), "need a damaged foe"
        with _root(source, p1) as root:
            for species, percent in public.items():
                mon = _mon(root, "p2", species)
                assert mon["maxhp"] > 100, "foe must keep the set's calculated max HP"
                assert int(100 * mon["hp"] / mon["maxhp"]) == percent or percent == 100
                if percent == 100:
                    assert mon["hp"] == mon["maxhp"]
            # Our own side shows exact HP and is untouched.
            truth = {
                mon["species"]: (mon["hp"], mon["maxhp"]) for mon in _side(source, "p1")
            }
            for mon in _side(root, "p1"):
                if mon["species"] in truth and mon["hp"]:
                    assert (mon["hp"], mon["maxhp"]) == truth[mon["species"]]
        with _root(source, p1, LEGACY) as legacy:
            assert _mon(legacy, "p2", "incineroar")["maxhp"] == 100


# --- 2. hidden items -----------------------------------------------------------------


@pytest.mark.integration
def test_unrevealed_foe_item_guess_survives_and_a_consumed_item_does_not() -> None:
    p1, p2 = _sets()
    with _battle(p1, p2, []) as (source, memory):
        view = source.battles["p1"]
        with _root(source, p1) as root:
            kept = [_mon(root, "p2", mon.species)["item"] for mon in view.opponent_active_pokemon]
            assert all(kept), f"active foes should keep their guessed items, got {kept}"
        with _root(source, p1, LEGACY) as legacy:
            # (A Mega stone is kept by the separate Mega fix; every other guess is blanked.)
            assert _mon(legacy, "p2", "incineroar")["item"] == ""
        # Publicly consumed: the item is gone and must not be resurrected from the guess.
        incineroar = next(m for m in view.opponent_active_pokemon if m.species == "incineroar")
        incineroar.item = None
        memory.opponent_items["incineroar"] = "sitrusberry"
        with _root(source, p1) as root:
            assert _mon(root, "p2", "incineroar")["item"] == ""
        # Publicly known to be something else: that item wins.
        incineroar.item = "leftovers"
        with _root(source, p1) as root:
            assert _mon(root, "p2", "incineroar")["item"] == "leftovers"


# --- 3. post-Mega stats --------------------------------------------------------------


@pytest.mark.integration
def test_already_megaed_foe_has_mega_forme_stats() -> None:
    by = _mons()
    p1, _ = _sets()
    p2 = _team(
        by["Charizard"].replace("CharizarditeY", "CharizarditeX"),
        by["Farigiraf"], by["Venusaur"], by["Garchomp"], by["Incineroar"], by["Sylveon"],
    )
    with _battle(p1, p2, [("move protect, move protect", "move heatwave mega, move psychic 1")]) as (
        source, _memory
    ):
        assert source.battles["p1"].opponent_used_mega_evolve
        with _root(source, p1) as root:
            mon = _mon(root, "p2", "charizardmegax")
            expected = calculate_stats("charizardmegax", mon["set"]["evs"], mon["set"]["nature"].lower())
            for stat in ("atk", "def", "spa", "spd", "spe"):
                assert mon["storedStats"][stat] == expected[stat], stat
        with _root(source, p1, LEGACY) as legacy:
            stale = _mon(legacy, "p2", "charizardmegax")["storedStats"]
            assert stale["atk"] != expected["atk"], "legacy keeps the base forme's stats"


# --- 4. Fake Out reuse / 5. Choice lock ----------------------------------------------


@pytest.mark.integration
def test_fake_out_is_not_offered_again_and_a_known_choice_lock_is_recreated() -> None:
    p1, p2 = _sets()
    turn = [("move toxic 1, move protect", "move fakeout 1, move rockslide")]
    with _battle(p1, p2, turn[:0]) as (source, _memory):
        with _root(source, p1) as root:
            assert any("fakeout" in m for m in _messages(root, "p2")), "turn 1 may Fake Out"
    with _battle(p1, p2, turn) as (source, _memory):
        view = source.battles["p1"]
        for mon in view.opponent_active_pokemon:
            if mon.species == "garchomp":
                mon.item = "choicescarf"
        with _root(source, p1) as root:
            messages = _messages(root, "p2")
            assert not any("fakeout" in m for m in messages), "Fake Out only works first turn"
            garchomp = _mon(root, "p2", "garchomp")
            assert garchomp["volatiles"]["choicelock"]["move"] == "rockslide"
            # The locked Garchomp (slot b) has exactly one move left.
            slot_b_moves = {
                action.split()[1]
                for message in messages
                for action in [message[len("/choose "):].split(", ")[1]]
                if action.startswith("move")
            }
            assert slot_b_moves == {"rockslide"}
        with _root(source, p1, LEGACY) as legacy:
            assert any("fakeout" in m for m in _messages(legacy, "p2"))
            assert "choicelock" not in _mon(legacy, "p2", "garchomp")["volatiles"]


# --- 8. Toxic stage / 9. Disable / 10. Unburden --------------------------------------


@pytest.mark.integration
def test_toxic_stage_is_restored() -> None:
    p1, p2 = _sets()
    for seed in range(1, 12):
        turns = [
            ("move toxic 1, move protect", "move darkestlariat 2, move rockslide"),
            ("move protect, move protect", "move darkestlariat 2, move rockslide"),
        ]
        with _battle(p1, p2, turns, seed=(seed, 2, 3, 4)) as (source, _memory):
            view = source.battles["p1"]
            target = next(m for m in view.opponent_active_pokemon if m.species == "incineroar")
            if target.status is None or target.status.name != "TOX":
                continue
            truth = _mon(source, "p2", "incineroar")["statusState"]["stage"]
            assert truth >= 1
            with _root(source, p1) as root:
                assert _mon(root, "p2", "incineroar")["statusState"].get("stage") == truth
            with _root(source, p1, LEGACY) as legacy:
                assert _mon(legacy, "p2", "incineroar")["statusState"].get("stage") is None
            return
    pytest.skip("Toxic never landed in 11 seeds")


@pytest.mark.integration
def test_disable_target_move_is_recovered_from_the_protocol() -> None:
    p1, p2 = _sets()
    for seed in range(1, 12):
        turns = [
            ("move protect, move protect", "move darkestlariat 2, move rockslide"),
            ("move disable 1, move protect", "move darkestlariat 2, move rockslide"),
        ]
        with _battle(p1, p2, turns, seed=(seed, 2, 3, 4)) as (source, memory):
            truth = _mon(source, "p2", "incineroar")["volatiles"].get("disable")
            if not truth:
                continue
            assert memory.disabled_moves[("p2", "incineroar")] == truth["move"]
            with _root(source, p1) as root:
                restored = _mon(root, "p2", "incineroar")["volatiles"]["disable"]
                assert restored["move"] == truth["move"]
                assert not any(f"move {truth['move']}" in m for m in _messages(root, "p2"))
            return
    pytest.skip("Disable never landed in 11 seeds")


@pytest.mark.integration
def test_unburden_survives_after_the_foes_item_is_knocked_off() -> None:
    by = _mons()
    p1 = _team(
        by["Garchomp"].replace("RockSlide,DragonClaw,Earthquake,Protect", "KnockOff,Protect,Earthquake,DragonClaw"),
        by["Sylveon"], by["Venusaur"], by["Farigiraf"], by["Incineroar"], by["Charizard"],
    )
    hitmonlee = "Hitmonlee||SitrusBerry|Unburden|CloseCombat,KnockOff,Protect,FakeOut|Jolly|,32,,,2,32||||50|"
    p2 = _team(hitmonlee, by["Farigiraf"], by["Venusaur"], by["Garchomp"], by["Incineroar"], by["Sylveon"])
    with _battle(p1, p2, [("move knockoff 1, move protect", "move closecombat 1, move psychic 1")]) as (
        source, memory
    ):
        truth = _mon(source, "p2", "hitmonlee")
        if "unburden" not in truth["volatiles"]:
            pytest.skip("Knock Off did not land")
        # The ability is private; a foe that revealed it (here: set on the view) keeps the
        # doubled Speed. A guessed non-Unburden ability must not conjure one.
        next(
            m for m in source.battles["p1"].opponent_team.values() if m.species == "hitmonlee"
        ).ability = "unburden"
        with _root(source, p1) as root:
            assert "unburden" in _mon(root, "p2", "hitmonlee")["volatiles"]
        with _root(source, p1, LEGACY) as legacy:
            assert "unburden" not in _mon(legacy, "p2", "hitmonlee")["volatiles"]


# --- 6. sleep timers / 11. reply dedupe ------------------------------------------------


@pytest.mark.integration
def test_judge_searches_every_remaining_sleep_branch() -> None:
    p1, p2 = _sets()
    p1 = p1.replace("Toxic,Disable,HelpingHand,Protect", "SleepPowder,Disable,HelpingHand,Protect")
    for seed in range(1, 12):
        with _battle(
            p1, p2, [("move sleeppowder 1, move protect", "move darkestlariat 2, move rockslide")],
            seed=(seed, 2, 3, 4),
        ) as (source, _memory):
            view = source.battles["p1"]
            if not any(m.status is not None and m.status.name == "SLP" for m in view.opponent_active_pokemon):
                continue
            base = PolicyConfig()
            judged = judge_config(base, 4)
            assert judged.exact_search_state_hypotheses == 1
            widened = with_exact_timers(judged, view, base)
            assert widened.exact_search_total_hypotheses >= 2
            assert widened.exact_search_state_hypotheses == base.exact_search_state_hypotheses
            legacy = with_exact_timers(judged, view, replace(base, exact_judge_exact_timers=False))
            assert legacy == judged
            return
    pytest.skip("Sleep Powder never landed")


def _scored(message: str) -> SimpleNamespace:
    return SimpleNamespace(order=SimpleNamespace(message=message))


def test_mega_twins_collapse_to_the_higher_scored_plan() -> None:
    replies = [
        _scored("/choose move heatwave mega, move psychic 1"),
        _scored("/choose move heatwave, move psychic 1"),
        _scored("/choose move heatwave mega, move trickroom"),
        _scored("/choose move megahorn 1, move trickroom"),
        _scored("/choose move heatwave, move trickroom"),
    ]
    config = replace(PolicyConfig(), search_opp_candidates=3)
    kept = [entry.order.message for entry in _opponent_replies(replies, config)]
    assert kept == [
        "/choose move heatwave mega, move psychic 1",
        "/choose move heatwave mega, move trickroom",
        "/choose move megahorn 1, move trickroom",
    ]
    legacy = replace(config, exact_search_dedupe_mega_replies=False)
    assert len(_opponent_replies(replies, legacy)) == 3
    assert _opponent_replies(replies, legacy)[1].order.message.endswith("psychic 1")
