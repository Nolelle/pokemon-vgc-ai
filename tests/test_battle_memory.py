from types import SimpleNamespace

from vgc.battle_memory import BattleMemory, TurnMemory
from vgc.damage import PokemonState


def _mon(species: str, hp: float = 1.0):
    return SimpleNamespace(species=species, current_hp_fraction=hp, fainted=False)


def test_memory_keeps_turn_history_protocol_actions_and_choices() -> None:
    memory = BattleMemory("battle-test", our_role="p1")
    memory.observe_protocol(
        [
            ["", "turn", "2"],
            ["", "move", "p2a: Incineroar", "Fake Out", "p1a: Flutter Mane"],
            ["", "switch", "p2b: Rillaboom", "Rillaboom, L50", "100/100"],
        ]
    )
    battle = SimpleNamespace(
        player_role="p1",
        turn=2,
        active_pokemon=[_mon("Flutter Mane", 0.5), _mon("Urshifu")],
        opponent_active_pokemon=[_mon("Incineroar"), _mon("Rillaboom")],
        weather={},
    )
    memory.observe_battle(battle)
    memory.record_choice(2, "protect / closecombat@opp0")

    assert memory.opponent_moves["incineroar"]["fakeout"] == 1
    assert memory.opponent_targets["fluttermane"] == 1
    assert memory.opponent_switches["rillaboom"] == 1
    assert memory.turns[0].our_hp["fluttermane"] == 50.0
    assert memory.turns[0].chosen_order == "protect / closecombat@opp0"


def test_memory_records_plan_changes_instead_of_forgetting_original_plan() -> None:
    memory = BattleMemory("battle-test")
    first = SimpleNamespace(
        primary_win_con_species="charizard",
        primary_threat_species="garchomp",
        plan_breakers=frozenset({"garchomp"}),
    )
    second = SimpleNamespace(
        primary_win_con_species="rillaboom",
        primary_threat_species="garchomp",
        plan_breakers=frozenset(),
    )

    memory.update_strategy(first)
    memory.current_turn = 4
    memory.update_strategy(second)

    assert memory.initial_win_con == "charizard"
    assert memory.current_win_con == "rillaboom"
    assert memory.plan_changes == [
        {
            "turn": 4,
            "from": "charizard",
            "to": "rillaboom",
            "reason": "previous plan no longer best or available",
        }
    ]


def test_memory_infers_relative_speed_and_direct_damage_from_public_protocol() -> None:
    target_state = PokemonState(
        "incineroar", sp_spread={"hp": 32}, nature="careful", current_hp=202
    )
    memory = BattleMemory("battle-test", our_role="p1", current_turn=1)
    memory.turns.append(
        TurnMemory(
            turn=1,
            our_active=("incineroar",),
            opponent_active=("garchomp",),
            our_hp={"incineroar": 100.0},
            opponent_hp={"garchomp": 100.0},
            weather=(),
            our_effective_speed={"incineroar": 100.0},
            our_states={"incineroar": target_state},
        )
    )
    memory._ident_species.update(
        {"p1a: Cat": "incineroar", "p2a: Chomp": "garchomp"}
    )
    memory._protocol_hp[("p1", "incineroar")] = 1.0
    memory.observe_protocol(
        [
            ["", "move", "p2a: Chomp", "Earthquake", "p1a: Cat"],
            ["", "-damage", "p1a: Cat", "150/202"],
            ["", "move", "p1a: Cat", "Flare Blitz", "p2a: Chomp"],
        ]
    )

    assert memory.speed_observations[-1].opponent_species == "garchomp"
    assert memory.speed_observations[-1].relation == "at_least"
    assert memory.speed_observations[-1].threshold == 100.0
    assert memory.damage_observations[-1].move_id == "earthquake"
    assert memory.damage_observations[-1].damage_fraction == (202 - 150) / 202
