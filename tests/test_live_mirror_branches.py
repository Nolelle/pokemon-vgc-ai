"""LiveExactMirror roots must actually execute Showdown branches after patchPublic.

A patched battle whose BattleQueue was replaced with a plain array used to throw inside
the next `go()`, which the worker drains swallowed. Every clone.step then returned no
protocol, no request, and an unchanged public state -- so exact search scored every
candidate with the same exchange_value. These tests fail closed on that signature.
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace

import pytest

from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition

from vgc.actions import enumerate_joint_orders
from vgc.config import REPO_ROOT
from vgc.mechanics_state import BattleMechanicsState, snapshot_battle
from vgc.models import PolicyConfig
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, DirectBattle, SimWorker, choice_string
from vgc.rl.exact_search import search_joint_orders_exact
from vgc.rl.live_mirror import LiveExactMirror
from vgc.rl.mechanics_oracle import evaluate_exact_branches

pytestmark = pytest.mark.integration

POOL = REPO_ROOT / "data" / "selfplay" / "archetype_pool_150"
OWN_TEAM_PATH = POOL / "rain_offense" / "team_11.packed.txt"
OPP_TEAM_PATH = POOL / "triple_setup_balance" / "team_02.packed.txt"

COMPACT_CONFIG = replace(
    PolicyConfig(),
    search_our_candidates=3,
    search_opp_candidates=2,
    search_opp_moves_per_slot=1,
    exact_search_future_samples=1,
    exact_search_state_hypotheses=1,
    exact_search_spread_hypotheses=2,
    exact_search_set_hypotheses=1,
    exact_search_bring_hypotheses=1,
    exact_search_total_hypotheses=2,
    use_rolling_horizon=False,
    use_value_head=False,
)

_CLONE_IDS = itertools.count()


def _require_showdown_and_pool() -> tuple[str, str]:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    if not OWN_TEAM_PATH.is_file() or not OPP_TEAM_PATH.is_file():
        pytest.skip("archetype_pool_150 teams are unavailable")
    return OWN_TEAM_PATH.read_text().strip(), OPP_TEAM_PATH.read_text().strip()


def _progress_signature(state: BattleMechanicsState) -> tuple:
    hp = tuple(
        (mon.species_id, mon.current_hp, mon.fainted)
        for mon in (*state.our_side.pokemon, *state.opponent_side.pokemon)
    )
    return state.turn, hp


def _joint_choices(root: DirectBattle, count: int) -> list[dict[str, str]]:
    expected = set(root.sides_to_move())
    assert expected == {"p1", "p2"}, (
        f"need a both-sides-to-move turn, not a forced switch; waiting on {sorted(expected)}"
    )
    orders = {side: enumerate_joint_orders(root.battles[side]) for side in expected}
    assert all(orders[side] for side in expected)
    p2_choice = choice_string(orders["p2"][0])
    choices: list[dict[str, str]] = []
    seen: set[str] = set()
    for order in orders["p1"]:
        encoded = choice_string(order)
        if encoded in seen:
            continue
        seen.add(encoded)
        choices.append({"p1": encoded, "p2": p2_choice})
        if len(choices) >= count:
            break
    if len(choices) < count and len(orders["p2"]) > 1:
        choices.append(
            {"p1": choice_string(orders["p1"][0]), "p2": choice_string(orders["p2"][1])}
        )
    assert len(choices) >= count, "not enough distinct joint choices on this turn"
    return choices[:count]


def _assert_clone_step_advances(
    root: DirectBattle, choices: dict[str, str], *, label: str
) -> None:
    before = snapshot_battle(root.battles["p1"])
    clone = root.clone(f"{root.battle_id}-{label}-{next(_CLONE_IDS)}")
    try:
        result = clone.step(dict(choices))
        assert result.lines["p1"], f"{label}: clone.step returned no p1 protocol lines"
        assert result.request_state, f"{label}: clone.step returned an empty request_state"
        after = snapshot_battle(clone.battles["p1"])
        assert _progress_signature(after) != _progress_signature(before), (
            f"{label}: clone snapshot did not advance (turn or HP unchanged)"
        )
    finally:
        clone.close()


@contextmanager
def _mirror_turn1(
    config: PolicyConfig = COMPACT_CONFIG,
) -> Iterator[tuple[DirectBattle, LiveExactMirror, DirectBattle]]:
    own_team, opp_team = _require_showdown_and_pool()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-branch-source",
            own_team,
            opp_team,
            seed=[7, 8, 9, 10],
        )
        mirror: LiveExactMirror | None = None
        root: DirectBattle | None = None
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            assert set(source.sides_to_move()) == {"p1", "p2"}
            mirror = LiveExactMirror(own_team, config)
            root = mirror.build(source.battles["p1"])
            assert set(root.sides_to_move()) == {"p1", "p2"}
            yield source, mirror, root
        finally:
            if root is not None:
                root.close()
            if mirror is not None:
                mirror.close()
            source.close()


def test_mirror_clone_step_emits_protocol_and_advances_public_state() -> None:
    with _mirror_turn1() as (source, mirror, root):
        choices = _joint_choices(root, 1)[0]
        _assert_clone_step_advances(root, choices, label="built")
        hypotheses = mirror.hypotheses(source.battles["p1"])
        if len(hypotheses) <= 1:
            # Only one hidden-information hypothesis; rebase sub-assert skipped.
            return
        rebased = mirror.rebase(root, source.battles["p1"], hypotheses[1])
        try:
            rebase_choices = _joint_choices(rebased, 1)[0]
            _assert_clone_step_advances(rebased, rebase_choices, label="rebased")
        finally:
            # `_mirror_turn1` closes the original root; rebase may have replaced it.
            if rebased is not root:
                rebased.close()


def test_mirror_exact_branches_diverge_from_each_other_and_the_root() -> None:
    with _mirror_turn1() as (_source, _mirror, root):
        pair = _joint_choices(root, 2)
        branches = evaluate_exact_branches(root, pair)
        assert len(branches) == 2
        root_state = snapshot_battle(root.battles["p1"])
        first = branches[0].state_for("p1")
        second = branches[1].state_for("p1")
        assert first != root_state
        assert second != root_state
        assert first != second


def test_mirror_exact_search_exchange_values_are_not_constant() -> None:
    with _mirror_turn1() as (_source, _mirror, root):
        scored = search_joint_orders_exact(root, "p1", COMPACT_CONFIG)
        searched = [entry for entry in scored if entry.breakdown.get("searched")]
        assert len(searched) >= 3
        exchanges = [float(entry.breakdown["exchange_value"]) for entry in searched]
        assert statistics.pstdev(exchanges) > 0
        assert len({round(value, 9) for value in exchanges}) >= 2
        offsets = [
            float(entry.score) - float(entry.breakdown["myopic_score"]) for entry in searched
        ]
        assert len({round(value, 9) for value in offsets}) >= 2, (
            "exact search collapsed to a constant offset of the myopic score"
        )


def test_patch_public_leaves_a_playable_request_and_null_stream_error() -> None:
    with _mirror_turn1() as (_source, _mirror, root):
        inspected = root.inspect()
        assert inspected["requestState"] in {"move", "switch"}
        assert "error" in inspected
        assert inspected["error"] is None
        choices = _joint_choices(root, 1)[0]
        clone = root.clone(f"{root.battle_id}-worker-contract-{next(_CLONE_IDS)}")
        try:
            result = clone.step(dict(choices))
            assert result.error is None
            assert result.lines["p1"]
            assert result.lines["p2"]
        finally:
            clone.close()


META1 = REPO_ROOT / "teams" / "meta1.packed.txt"
FARIGIRAF_CHARIZARD = "team 2134"


def _meta1_team() -> str:
    if not META1.is_file():
        pytest.skip("meta1 packed team is unavailable")
    return META1.read_text().strip()


def _hp_fraction(mon) -> float | None:
    if mon is None:
        return None
    return round(float(mon.current_hp_fraction), 2)


def _enum_ids(mapping) -> set[str]:
    return {getattr(key, "name", str(key)) for key in (mapping or {})}


@contextmanager
def _meta1_trick_room_mirror() -> Iterator[tuple[DirectBattle, DirectBattle]]:
    """Source game with Farigiraf Trick Room up, then a LiveExactMirror of that view."""

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = _meta1_team()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-trick-room-source",
            team,
            team,
            seed=[11, 12, 13, 14],
        )
        mirror: LiveExactMirror | None = None
        root: DirectBattle | None = None
        try:
            source.step({"p1": FARIGIRAF_CHARIZARD, "p2": FARIGIRAF_CHARIZARD})
            source.step(
                {
                    "p1": "move psychic 1, move heatwave",
                    "p2": "move trickroom, move protect",
                }
            )
            observation = source.battles["p1"]
            assert Field.TRICK_ROOM in observation.fields
            observation.side_conditions[SideCondition.TAILWIND] = observation.turn
            mirror = LiveExactMirror(team, COMPACT_CONFIG)
            root = mirror.build(observation)
            yield source, root
        finally:
            if root is not None:
                root.close()
            if mirror is not None:
                mirror.close()
            source.close()


def test_mirror_p2_base_matches_p1_public_observation() -> None:
    with _meta1_trick_room_mirror() as (source, root):
        observation = source.battles["p1"]
        p2 = root.battles["p2"]
        assert Field.TRICK_ROOM in p2.fields
        assert _enum_ids(p2.fields) == _enum_ids(observation.fields)
        assert _enum_ids(p2.weather) == _enum_ids(observation.weather)
        assert p2.turn == observation.turn
        assert SideCondition.TAILWIND in p2.opponent_side_conditions
        assert _enum_ids(p2.side_conditions) == _enum_ids(observation.opponent_side_conditions)
        assert _enum_ids(p2.opponent_side_conditions) == _enum_ids(observation.side_conditions)
        assert [_hp_fraction(mon) for mon in p2.active_pokemon] == [
            _hp_fraction(mon) for mon in observation.opponent_active_pokemon
        ]
        assert [_hp_fraction(mon) for mon in p2.opponent_active_pokemon] == [
            _hp_fraction(mon) for mon in observation.active_pokemon
        ]


def test_mirror_trick_room_survives_fieldend_on_p2_parser() -> None:
    with _meta1_trick_room_mirror() as (_source, root):
        assert Field.TRICK_ROOM in root.battles["p2"].fields
        ended = False
        for _ in range(6):
            choices = _joint_choices(root, 1)[0]
            branches = evaluate_exact_branches(root, [choices])
            lines = dict(branches[0].public_lines)["p1"]
            ended = any("-fieldend" in line and "Trick Room" in line for line in lines)
            if ended:
                break
            root.step(dict(choices))
        assert ended, "Trick Room never ended on the mirrored root"


def test_mirror_stale_preparing_does_not_lock_search_to_solarbeam() -> None:
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = _meta1_team()
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-stale-solarbeam-source",
            team,
            team,
            seed=[21, 22, 23, 24],
        )
        mirror: LiveExactMirror | None = None
        root: DirectBattle | None = None
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            observation = source.battles["p1"]
            charizard = next(
                mon
                for mon in observation.active_pokemon
                if mon is not None and "charizard" in (mon.species or "")
            )
            slot = 0 if observation.active_pokemon[0] is charizard else 1
            charizard.prepare("solarbeam", True)
            assert charizard.preparing
            assert len(observation.available_moves[slot]) >= 2
            mirror = LiveExactMirror(team, COMPACT_CONFIG)
            root = mirror.build(observation)
            search_config = replace(COMPACT_CONFIG, search_our_candidates=6)
            scored = search_joint_orders_exact(root, "p1", search_config)
            searched = [entry for entry in scored if entry.breakdown.get("searched")]
            assert searched
            unlocked = [
                entry
                for entry in searched
                if "solarbeam"
                not in choice_string(entry.order).split(",")[slot].lower()
            ]
            assert unlocked, (
                "stale Solar Beam lock trapped search; expected a non-solarbeam "
                f"order in { [choice_string(entry.order) for entry in searched] }"
            )
        finally:
            if root is not None:
                root.close()
            if mirror is not None:
                mirror.close()
            source.close()


def test_public_teacher_labels_from_p2_seat_without_keyerror() -> None:
    """Regression: teacher collection failed closed for every p2-seat game.

    `patch_public_state` reused the teacher's battle object as the perspective
    side's parser base. From the p2 seat that put p2-keyed team dicts under p1
    request lines, so the first clone step raised `KeyError: 'p1: ...'` and the
    whole game recorded nothing. Multi-hypothesis defaults are kept on: the
    single-hypothesis path never broke.
    """
    from vgc.rl.distill import public_information_exact_search

    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    team = _meta1_team()
    search_config = replace(
        COMPACT_CONFIG,
        search_our_candidates=4,
        exact_search_spread_hypotheses=2,
        exact_search_set_hypotheses=2,
        exact_search_bring_hypotheses=2,
        exact_search_total_hypotheses=2,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-p2-seat-source",
            team,
            team,
            seed=[31, 32, 33, 34],
        )
        try:
            source.step({"p1": "team 1234", "p2": "team 1234"})
            assert set(source.sides_to_move()) == {"p1", "p2"}
            scored = public_information_exact_search(
                source.battles["p2"], search_config, team
            )
            assert scored, "p2-seat teacher labeling returned no orders"
            assert all(entry.order.message for entry in scored)
        finally:
            source.close()


def test_mirror_reproduces_choice_lock_from_snapshot() -> None:
    """Regression: the mirror offered moves a Choice-locked mon cannot use.

    The snapshot never marked request-withheld moves disabled, so the patched
    root offered the full moveset, exact search ranked a forbidden move, and
    the teacher submitted it into an InvalidChoice crash. Basculegion fires
    Aqua Jet turn 1 (into a Protect, so nothing faints and no RNG matters);
    turn 2 the mirror must offer only Aqua Jet for that slot, matching live.
    """
    from poke_env.battle.move import Move

    from vgc.damage import to_id

    pool_team = POOL / "gardevoir_maushold" / "team_17.packed.txt"
    if not DEFAULT_SHOWDOWN_REPO.exists():
        pytest.skip("local Pokemon Showdown checkout is unavailable")
    if not pool_team.is_file():
        pytest.skip("archetype_pool_150 teams are unavailable")
    team = pool_team.read_text().strip()
    search_config = replace(
        COMPACT_CONFIG,
        search_our_candidates=4,
        exact_search_spread_hypotheses=1,
        exact_search_set_hypotheses=1,
        exact_search_bring_hypotheses=1,
        exact_search_total_hypotheses=1,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-choicelock-source",
            team,
            team,
            seed=[41, 42, 43, 44],
        )
        mirror: LiveExactMirror | None = None
        try:
            source.step({"p1": "team 3124", "p2": "team 1234"})
            live = source.battles["p1"]
            locked_slot = next(
                index
                for index, mon in enumerate(live.active_pokemon or ())
                if mon is not None and to_id(mon.species) == "basculegion"
            )
            live_locked = {
                to_id(move.id) for move in live.available_moves[locked_slot]
            }
            assert live_locked, "expected Basculegion to move on turn 1"
            first_move = sorted(live_locked)[0]
            source.step(
                {
                    "p1": (
                        f"move {first_move} 1, move protect"
                        if locked_slot == 0
                        else f"move protect, move {first_move} 1"
                    ),
                    "p2": "move protect, move protect",
                }
            )
            live = source.battles["p1"]
            live_locked = {to_id(move.id) for move in live.available_moves[locked_slot]}
            assert len(live_locked) == 1, (
                f"expected a Choice lock after turn 1, offered {sorted(live_locked)}"
            )
            mirror = LiveExactMirror(team, search_config)
            root = mirror.build(live, mirror.hypotheses(live)[0])
            try:
                mirror_locked = {
                    to_id(move.id)
                    for move in root.battles["p1"].available_moves[locked_slot]
                }
                assert mirror_locked == live_locked, (
                    f"mirror offers {sorted(mirror_locked)} but live offers "
                    f"{sorted(live_locked)}"
                )
                offered = {
                    to_id(single.order.id)
                    for order in enumerate_joint_orders(root.battles["p1"])
                    for single in (
                        order.first_order if locked_slot == 0 else order.second_order,
                    )
                    if isinstance(single.order, Move)
                }
                assert offered <= live_locked, (
                    f"mirror enumerates {sorted(offered - live_locked)} "
                    "that live forbids"
                )
            finally:
                root.close()
        finally:
            if mirror is not None:
                mirror.close()
            source.close()


def test_mirror_branches_survive_a_patched_encore_volatile() -> None:
    """Regression: branching a mirror with a patched Encore crashed the sim.

    The patch restored the `encore` volatile without the engine's locked-move
    field, so the first branch step crashed reading flags off undefined
    (`failencore`) and the whole decision recorded nothing. Whimsicott Encores
    a Follow-Me Clefable turn 1; turn 2 the mirrored exact search must run to
    a ranking instead of raising.
    """
    from vgc.rl.distill import public_information_exact_search

    dev = (REPO_ROOT / "teams" / "dev.packed.txt").read_text().strip()
    frail = (REPO_ROOT / "tests" / "fixtures" / "replay_corpus" / "frail_leads.packed.txt").read_text().strip()
    if not dev or not frail:
        pytest.skip("dev/frail_leads teams are unavailable")
    search_config = replace(
        COMPACT_CONFIG,
        search_our_candidates=4,
        exact_search_spread_hypotheses=1,
        exact_search_set_hypotheses=1,
        exact_search_bring_hypotheses=1,
        exact_search_total_hypotheses=1,
    )
    with SimWorker(DEFAULT_SHOWDOWN_REPO) as source_worker:
        source = DirectBattle.start(
            source_worker,
            "live-mirror-encore-branch-source",
            dev,
            frail,
            seed=[51, 52, 53, 54],
        )
        try:
            source.step({"p1": "team 4123", "p2": "team 3412"})
            source.step(
                {
                    "p1": "move encore 1, move protect",
                    "p2": "move followme, move protect",
                }
            )
            assert set(source.sides_to_move()) == {"p1", "p2"}
            scored = public_information_exact_search(
                source.battles["p1"], search_config, dev
            )
            assert scored, "exact search over an encored position returned no orders"
        finally:
            source.close()
