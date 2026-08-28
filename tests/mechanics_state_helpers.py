"""Minimal explicit builders for `vgc.mechanics_state` snapshots in unit tests.

The snapshot dataclasses are deliberately field-complete and default-free so that a new
observable property cannot be forgotten. These helpers fill every field once so tests
can state only the handful of facts they actually care about.
"""

from __future__ import annotations

from vgc.mechanics_state import (
    BOOST_IDS,
    STAT_POINT_IDS,
    BattleMechanicsState,
    EffectSnapshot,
    PokemonMechanicsState,
    SideMechanicsState,
)


def make_pokemon_state(
    species_id: str,
    *,
    status: str | None = None,
    status_counter: int = 0,
    effects: tuple[EffectSnapshot, ...] = (),
    ability_id: str | None = None,
    fainted: bool = False,
    active: bool = True,
) -> PokemonMechanicsState:
    return PokemonMechanicsState(
        species_id=species_id,
        base_species_id=species_id,
        name=species_id,
        level=50,
        gender=None,
        types=("Normal",),
        base_types=("Normal",),
        current_hp=0 if fainted else 100,
        max_hp=100,
        fainted=fainted,
        active=active,
        revealed=True,
        selected_in_preview=True,
        stats=tuple((stat, 100) for stat in STAT_POINT_IDS),
        stat_points=None,
        individual_values=None,
        nature=None,
        boosts=tuple((boost, 0) for boost in BOOST_IDS),
        status=status,
        status_counter=status_counter,
        effects=effects,
        item_id=None,
        item_known=False,
        ability_id=ability_id,
        base_ability_id=ability_id,
        temporary_ability_id=None,
        forme_change_ability_id=None,
        ability_known=ability_id is not None,
        base_move_ids=(),
        moves=(),
        last_move_id=None,
        mimic_move_id=None,
        preparing_move_id=None,
        preparing_target=None,
        preparing=False,
        must_recharge=False,
        protect_counter=0,
        first_turn=False,
        transformed=False,
        mega_evolved=False,
        terastallized=False,
        tera_type=None,
        weight=10.0,
    )


def make_side_state(
    pokemon: list[PokemonMechanicsState] | None = None,
) -> SideMechanicsState:
    members = tuple(pokemon or ())
    return SideMechanicsState(
        pokemon=members,
        active_species=tuple(mon.species_id for mon in members if mon.active),
        side_conditions=(),
        force_switch=(False, False),
        trapped=(False, False),
        maybe_trapped=(False, False),
        can_mega_evolve=(False, False),
        used_mega_evolution=False,
        can_dynamax=(False, False),
        used_dynamax=False,
        can_tera=(False, False),
        used_tera=False,
        can_z_move=(False, False),
        used_z_move=False,
    )


def make_battle_state(
    *,
    our_pokemon: list[PokemonMechanicsState] | None = None,
    opponent_pokemon: list[PokemonMechanicsState] | None = None,
    turn: int = 1,
) -> BattleMechanicsState:
    return BattleMechanicsState(
        format_id="gen9championsvgc2026regmb",
        generation=9,
        game_type="doubles",
        turn=turn,
        max_team_size=4,
        team_size=6,
        team_preview=False,
        commanding=False,
        reviving=False,
        waiting=False,
        finished=False,
        won=False,
        lost=False,
        fields=(),
        weather=(),
        available_moves=(),
        available_switches=(),
        valid_order_count=None,
        last_request_json=None,
        our_preview_species=(),
        opponent_preview_species=(),
        our_side=make_side_state(our_pokemon),
        opponent_side=make_side_state(opponent_pokemon),
    )
