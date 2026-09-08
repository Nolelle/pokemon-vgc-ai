from types import SimpleNamespace

from vgc.battle_memory import BattleMemory, DamageObservation, TurnMemory
from vgc.config import REPO_ROOT
from vgc.damage import FieldState, PokemonState, damage_range
from vgc.opponent_belief import build_opponent_beliefs
from poke_env.teambuilder.teambuilder import Teambuilder


def _mon(species: str, hp: float = 1.0):
    return SimpleNamespace(species=species, current_hp_fraction=hp, fainted=False)


def _setup_mon(species: str, *, moves=(), item=None, ability=None, hp=1.0):
    return SimpleNamespace(
        species=species,
        moves={move: None for move in moves},
        item=item,
        ability=ability,
        current_hp_fraction=hp,
        current_hp=100,
        fainted=False,
        boosts={},
        status=None,
        evs=None,
        nature=None,
    )


def _battle_with_complete_own_team():
    packed = (REPO_ROOT / "teams" / "phase2_mirror.packed.txt").read_text().strip()
    builders = Teambuilder.parse_packed_team(packed)
    mons = []
    for entry in builders:
        species = entry.species or entry.nickname
        mons.append(
            _setup_mon(
                species,
                moves=entry.moves,
                item=entry.item,
                ability=entry.ability,
            )
        )
    opponent_preview = [_setup_mon("Garchomp"), _setup_mon("Incineroar")]
    return SimpleNamespace(
        team={f"p1: {entry.nickname or entry.species}": mon for entry, mon in zip(builders, mons)},
        teambuilder_team=builders,
        active_pokemon=mons[:2],
        opponent_team={},
        opponent_active_pokemon=[None, None],
        teampreview_opponent_team=opponent_preview,
    )


def _battle_with_opponent(species: str):
    battle = _battle_with_complete_own_team()
    appeared = _setup_mon(species)
    battle.opponent_team = {f"p2: {species}": appeared}
    battle.opponent_active_pokemon = [appeared, None]
    battle.teampreview_opponent_team = [appeared]
    return battle


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
    assert memory.damage_observations[-1].num_targets == 1


def test_spread_move_damage_records_shared_target_count() -> None:
    incineroar = PokemonState("incineroar", sp_spread={"hp": 32}, nature="careful", current_hp=202)
    flutter = PokemonState("fluttermane", sp_spread={"hp": 0}, nature="timid", current_hp=135)
    memory = BattleMemory("spread", our_role="p1", current_turn=1)
    memory.turns.append(
        TurnMemory(
            turn=1,
            our_active=("incineroar", "fluttermane"),
            opponent_active=("charizard",),
            our_hp={"incineroar": 100.0, "fluttermane": 100.0},
            opponent_hp={"charizard": 100.0},
            weather=(),
            our_states={"incineroar": incineroar, "fluttermane": flutter},
        )
    )
    memory._ident_species.update(
        {
            "p1a: Cat": "incineroar",
            "p1b: Ghost": "fluttermane",
            "p2a: Sun": "charizard",
        }
    )
    memory._protocol_hp[("p1", "incineroar")] = 1.0
    memory._protocol_hp[("p1", "fluttermane")] = 1.0
    memory.observe_protocol(
        [
            ["", "move", "p2a: Sun", "Heat Wave"],
            ["", "-damage", "p1a: Cat", "150/202"],
            ["", "-damage", "p1b: Ghost", "90/135"],
            ["", "move", "p1a: Cat", "Fake Out", "p2a: Sun"],
        ]
    )

    assert len(memory.damage_observations) == 2
    assert all(observation.num_targets == 2 for observation in memory.damage_observations)
    assert memory.damage_observations[0].move_id == "heatwave"


def test_spread_damage_reweights_matching_attack_spread() -> None:
    incineroar = PokemonState("incineroar", sp_spread={"hp": 32}, nature="careful", current_hp=202)
    flutter = PokemonState("fluttermane", sp_spread={"hp": 0}, nature="timid", current_hp=135)
    high = PokemonState("charizard", sp_spread={"spa": 32}, nature="modest")
    observed_incin = damage_range(
        high,
        incineroar,
        "heatwave",
        FieldState(is_doubles=True, num_targets=2),
    )
    observed_flutter = damage_range(
        high,
        flutter,
        "heatwave",
        FieldState(is_doubles=True, num_targets=2),
    )
    memory = BattleMemory("spread-belief")
    memory.damage_observations.extend(
        [
            DamageObservation(
                opponent_species="charizard",
                move_id="heatwave",
                target_species="incineroar",
                damage_fraction=observed_incin.max_percent / 100.0,
                target_state=incineroar,
                num_targets=2,
            ),
            DamageObservation(
                opponent_species="charizard",
                move_id="heatwave",
                target_species="fluttermane",
                damage_fraction=observed_flutter.max_percent / 100.0,
                target_state=flutter,
                num_targets=2,
            ),
        ]
    )
    battle = _battle_with_opponent("charizard")
    usage = {
        "charizard": [
            {"sp": {"spa": 0}, "nature": "timid", "weight": 0.5},
            {"sp": {"spa": 32}, "nature": "modest", "weight": 0.5},
        ]
    }
    belief = build_opponent_beliefs(battle, memory, usage=usage, set_priors={})[0]
    by_nature = {hypothesis.nature: hypothesis.probability for hypothesis in belief.hypotheses}
    assert by_nature["modest"] > by_nature["timid"]


def test_crit_damage_observation_is_ignored_for_spread_reweighting() -> None:
    target = PokemonState("incineroar", sp_spread={"hp": 32}, nature="careful", current_hp=202)
    high = PokemonState("garchomp", sp_spread={"atk": 32}, nature="adamant")
    observed = damage_range(high, target, "earthquake", FieldState(is_doubles=True))
    crit_fraction = min(1.0, observed.max_percent * 1.5 / 100.0)

    memory = BattleMemory("crit")
    memory.damage_observations.append(
        DamageObservation(
            opponent_species="garchomp",
            move_id="earthquake",
            target_species="incineroar",
            damage_fraction=crit_fraction,
            target_state=target,
            crit=True,
        )
    )
    battle = _battle_with_opponent("garchomp")
    usage = {
        "garchomp": [
            {"sp": {"atk": 0}, "nature": "modest", "weight": 0.5},
            {"sp": {"atk": 32}, "nature": "adamant", "weight": 0.5},
        ]
    }
    belief = build_opponent_beliefs(battle, memory, usage=usage, set_priors={})[0]
    by_nature = {hypothesis.nature: hypothesis.probability for hypothesis in belief.hypotheses}
    assert by_nature["adamant"] == by_nature["modest"]


def test_crit_line_marks_damage_observation_before_damage_protocol() -> None:
    target_state = PokemonState("incineroar", sp_spread={"hp": 32}, nature="careful", current_hp=202)
    memory = BattleMemory("crit-protocol", our_role="p1", current_turn=1)
    memory.turns.append(
        TurnMemory(
            turn=1,
            our_active=("incineroar",),
            opponent_active=("garchomp",),
            our_hp={"incineroar": 100.0},
            opponent_hp={"garchomp": 100.0},
            weather=(),
            our_states={"incineroar": target_state},
        )
    )
    memory._ident_species.update({"p1a: Cat": "incineroar", "p2a: Chomp": "garchomp"})
    memory._protocol_hp[("p1", "incineroar")] = 1.0
    memory.observe_protocol(
        [
            ["", "move", "p2a: Chomp", "Earthquake", "p1a: Cat"],
            ["", "-crit", "p1a: Cat"],
            ["", "-damage", "p1a: Cat", "120/202"],
            ["", "move", "p1a: Cat", "Protect"],
        ]
    )

    assert memory.damage_observations[-1].crit is True


def test_first_damage_observation_uses_full_hp_percent_baseline() -> None:
    target_state = PokemonState("incineroar", sp_spread={"hp": 32}, nature="careful", current_hp=202)
    memory = BattleMemory("first-hit", our_role="p1", current_turn=1)
    memory.turns.append(
        TurnMemory(
            turn=1,
            our_active=("incineroar", "rillaboom"),
            opponent_active=("garchomp",),
            our_hp={"rillaboom": 100.0},
            opponent_hp={"garchomp": 100.0},
            weather=(),
            our_states={"incineroar": target_state},
        )
    )
    memory._ident_species.update({"p1a: Cat": "incineroar", "p2a: Chomp": "garchomp"})
    memory.observe_protocol(
        [
            ["", "move", "p2a: Chomp", "Earthquake", "p1a: Cat"],
            ["", "-damage", "p1a: Cat", "150/202"],
            ["", "move", "p1a: Cat", "Protect"],
        ]
    )

    assert memory.damage_observations[-1].damage_fraction == (202 - 150) / 202
