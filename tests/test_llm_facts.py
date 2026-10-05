"""`vgc.llm.facts.build_packet` on real battles from one short direct game.

Critical behaviour only: option IDs match `build_options`, the cacheable fixed text never
changes within a game, opponent information is only ever a fact (seen) or a labelled
GUESS, and a broken battle never raises.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from vgc.config import FORMAT_ID
from vgc.evaluator import score_joint_orders
from vgc.llm.config import LLMConfig
from vgc.llm.facts import build_packet, load_team_plan
from vgc.llm.packet import build_options
from vgc.models import PolicyConfig
from vgc.rl.agents import DirectAgent, make_direct_agent
from vgc.rl.env import DEFAULT_SHOWDOWN_REPO, SimWorker
from vgc.rl.match import play_battle

ROOT = Path(__file__).resolve().parents[1]
OURS = ROOT / "teams" / "owner" / "psyspam_sand.packed.txt"
THEIRS = ROOT / "teams" / "owner" / "salamence_tw.packed.txt"


def _recorder_class():
    from vgc.agent import VgcPlayer

    class Recorder(VgcPlayer):
        def __init__(self, *args, sink, plan, **kwargs):
            super().__init__(*args, **kwargs)
            self.sink, self.plan = sink, plan

        def decide(self, battle):
            scored = score_joint_orders(battle, self.config)
            if len(scored) > 1:
                packet, options = build_packet(
                    battle, scored, llm_config=LLMConfig(), team_plan=self.plan,
                    memory=self._memory_for(battle), request_id=f"t{battle.turn}",
                )
                self.sink.append((battle, scored, packet, options, battle.turn))
            return scored[0].order if scored else self.choose_random_move(battle)

    return Recorder


@pytest.fixture(scope="module")
def game():
    if not (DEFAULT_SHOWDOWN_REPO / "dist" / "sim" / "index.js").exists():
        pytest.skip(f"no built showdown sim at {DEFAULT_SHOWDOWN_REPO}")
    sink: list = []
    snapshots: list = []
    ours, theirs = OURS.read_text().strip(), THEIRS.read_text().strip()
    config = PolicyConfig(format_id=FORMAT_ID, use_two_ply_search=False)
    player = _recorder_class()(
        config=config, team=ours, battle_format=FORMAT_ID, start_listening=False,
        sink=sink, plan=load_team_plan(OURS),
    )
    agents = {"p1": DirectAgent(player, name="ours"), "p2": make_direct_agent("vgc", theirs)}
    agents["p2"].name = "opp"
    with SimWorker() as worker:
        play_battle(worker, "facts-test", agents, {"p1": ours, "p2": theirs}, seed=[5, 6, 7, 8])
    assert len(sink) >= 3
    # The battle object keeps changing; capture what it showed at the end for comparison.
    snapshots.append(sink[-1][0])
    return sink, snapshots[0]


def test_option_ids_match_build_options(game):
    sink, _ = game
    for _battle, scored, packet, options, _turn in sink:
        expected = build_options(scored, LLMConfig().max_options)
        assert [o.id for o in options] == [o.id for o in expected]
        assert [o.order for o in options] == [o.order for o in expected]
        assert packet.option_ids == tuple(o.id for o in expected)
        assert options[0].id == "P01"
        for opt in options:
            assert f"{opt.id}:" in packet.turn_text
        assert packet.turn_text.count("\nP") >= len(options) - 1  # one line per option


def test_fixed_text_identical_across_turns(game):
    sink, _ = game
    fixed = {packet.fixed_text for _b, _s, packet, _o, _t in sink}
    assert len(fixed) == 1
    assert len({turn for *_rest, turn in sink}) >= 3
    turn_texts = {packet.turn_text for _b, _s, packet, _o, _t in sink}
    assert len(turn_texts) == len(sink)  # the per-turn part does change
    text = next(iter(fixed))
    assert "Psyspam Sand" in text and "MOVE REFERENCE" in text and "GUESSES" in text


def test_opponent_info_is_seen_or_labelled_guess(game):
    sink, _ = game
    assert any("- THEIR " in x[2].turn_text for x in sink)
    for battle, _scored, packet, _o, _t in sink:
        foe_lines = [ln for ln in packet.turn_text.splitlines() if ln.startswith("- THEIR ")]
        # The stored battle keeps changing, so only turns that show a foe are checked.
        for line in foe_lines:
            facts, _sep, guess = line.partition("GUESS:")
            # An item/ability is a fact only when marked seen; otherwise "not seen".
            assert "item not seen" in facts or "item " in facts and "(seen)" in facts
            assert "ability not seen" in facts or "(seen)" in facts
            # Anything guessed sits after the label, so the fact part holds only seen moves.
        seen = {
            m
            for mon in (battle.opponent_team or {}).values()
            for m in (mon.moves or {})
        }
        # No move name on a "moves seen:" list that poke-env has not actually seen.
        from vgc.llm.facts import _move

        for line in foe_lines:
            part = line.partition("moves seen:")[2].partition("GUESS:")[0].strip(" ;")
            if part and part != "none":
                for name in (n.strip() for n in part.split(",")):
                    assert name in {_move(m) for m in seen}


def test_never_raises_on_broken_battle():
    scored = [SimpleNamespace(order=SimpleNamespace(message="/choose move a, move b"), score=1.0)]
    packet, options = build_packet(
        SimpleNamespace(), scored, llm_config=LLMConfig(), team_plan="", request_id="x"
    )
    assert packet.option_ids == ("P01",) and [o.id for o in options] == ["P01"]
    assert "P01:" in packet.turn_text and packet.fixed_text


def test_load_team_plan():
    assert "Plan" in load_team_plan("psyspam_sand")
    assert load_team_plan(OURS) == load_team_plan("psyspam_sand")
    assert load_team_plan("no_such_team") == ""


# ---- hand-built boards: no engine needed ------------------------------------------------


def _mon(species: str, moves=(), hp: float = 1.0):
    return SimpleNamespace(
        species=species, base_species=species, moves={m: object() for m in moves},
        current_hp_fraction=hp, fainted=False,
    )


def _st(ours, foes, our_states=None, foe_states=None):
    from vgc.damage import PokemonState

    return {
        "usage": {}, "ours": ours, "foes": foes,
        "our_states": our_states or [PokemonState(m.species) for m in ours],
        "foe_states": foe_states or [PokemonState(m.species) for m in foes],
        "weather": None, "terrain": None, "trick_room": False,
        "our_tailwind": False, "foe_tailwind": False, "our_screens": frozenset(),
        "foe_screens": frozenset(),
    }


def test_hidden_foe_speed_is_always_a_guess_even_with_one_hypothesis():
    from vgc.llm import facts

    # usage={} -> a single fallback hypothesis, so every "sample" agrees on one number.
    st = _st([_mon("garchomp"), _mon("incineroar")], [_mon("rillaboom"), _mon("fluttermane")])
    text = "\n".join(facts._speed_lines(st))
    ours_part, _, theirs_part = text.partition("theirs")
    assert "ours Garchomp Speed" in ours_part and "GUESS" not in ours_part.split(";")[0]
    for line in text.split(";"):
        if "theirs" in line:
            assert "GUESS" in line and "~" in line
    assert "Likely move order (foe Speeds are GUESSES)" in text


def test_threat_damage_uses_spread_target_count(monkeypatch):
    from vgc.llm import facts

    seen: list[tuple[str, int]] = []
    real = facts.damage_range

    def spy(attacker, defender, move_id, field):
        seen.append((move_id, field.num_targets))
        return real(attacker, defender, move_id, field)

    monkeypatch.setattr(facts, "damage_range", spy)
    ours = [_mon("incineroar"), _mon("rillaboom", ("earthquake",))]
    foes = [_mon("garchomp", ("earthquake", "closecombat")), _mon("fluttermane")]
    st = _st(ours, foes)
    facts._threat_lines(st, {}, PolicyConfig())
    by_move = {m: n for m, n in seen}
    # Earthquake (allAdjacent) from a foe hits both our actives AND its own partner.
    assert by_move["earthquake"] == 3
    facts._damage_lines(st)
    ours_eq = [n for m, n in seen[-4:] if m == "earthquake"]
    assert ours_eq and all(n == 3 for n in ours_eq)  # both foes + our partner
    assert facts._hit_count("normal", 2, True) == 1
    assert facts._hit_count("allAdjacentFoes", 2, True) == 2
    assert facts._hit_count("allAdjacentFoes", 1, True) == 1


def test_blind_options_hide_engine_rank_and_keep_the_same_choices(game):
    sink, _ = game
    battle, scored, _packet, options, turn = sink[0]
    blind_cfg = LLMConfig(blind_options=True)
    packet, blind = build_packet(
        battle, scored, llm_config=blind_cfg, team_plan="", request_id=f"blind-{turn}",
    )
    option_block = packet.turn_text.split("## OPTIONS", 1)[1]
    assert "engine #" not in option_block and "score" not in option_block
    assert sorted(o.order for o in blind) == sorted(o.order for o in options)
    assert "engine's top option" not in packet.fixed_text
