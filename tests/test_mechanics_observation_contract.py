"""Fail when a poke-env state property is added without an explicit disposition."""

from __future__ import annotations

from poke_env.battle import DoubleBattle, Move, Pokemon


def _properties(cls) -> set[str]:
    return {
        name
        for parent in cls.mro()
        for name, value in vars(parent).items()
        if isinstance(value, property)
    }


BATTLE_CAPTURED = {
    "active_pokemon", "available_moves", "available_switches", "can_mega_evolve",
    "commanding", "fields", "finished", "force_switch", "format", "gen",
    "last_request", "lost", "max_team_size", "maybe_trapped", "opponent_active_pokemon",
    "opponent_side_conditions", "opponent_team", "opponent_used_mega_evolve", "reviving",
    "side_conditions", "team", "team_size", "teampreview", "teampreview_opponent_team",
    "teampreview_team", "trapped", "turn", "used_mega_evolve", "valid_orders", "wait",
    "weather", "won",
}
BATTLE_DISABLED_BY_FORMAT = {
    "can_dynamax", "can_tera", "can_z_move", "dynamax_turns_left",
    "opponent_dynamax_turns_left", "opponent_used_dynamax", "opponent_used_tera",
    "opponent_used_z_move", "used_dynamax", "used_tera", "used_z_move",
}
BATTLE_DERIVED_OR_NON_MECHANICAL = {
    "all_active_pokemons", "battle_tag", "grounded", "opponent_rating", "opponent_role",
    "opponent_username", "player_role", "player_username", "players", "rating",
    "teambuilder_team",
}

POKEMON_CAPTURED = {
    "ability", "active", "base_ability", "base_moves", "base_species", "boosts",
    "current_hp", "effects", "evs", "fainted", "first_turn", "forme_change_ability",
    "gender", "item", "ivs", "last_move", "level", "max_hp", "mimic_move", "moves",
    "must_recharge", "name", "nature", "preparing", "preparing_move", "preparing_target",
    "protect_counter", "revealed", "selected_in_teampreview", "species", "stats", "status",
    "status_counter", "temporary_ability", "tera_type", "transformed", "types", "weight",
}
POKEMON_DISABLED_BY_FORMAT = {
    "available_z_moves", "is_dynamaxed", "is_terastallized",
}
POKEMON_DERIVED_OR_NON_MECHANICAL = {
    "base_stats", "base_types", "current_hp_fraction", "gen", "height", "hp_status", "pokeball",
    "possible_abilities", "shiny", "stab_multiplier", "type_1", "type_2",
}

MOVE_DYNAMIC_CAPTURED = {"current_pp", "entry", "is_last_used", "max_pp"}
MOVE_STATIC_CATALOG = {
    "accuracy", "base_power", "boosts", "breaks_protect", "can_z_move", "category",
    "crit_ratio", "damage", "deduced_target", "defensive_category", "drain", "dynamaxed",
    "expected_hits", "flags", "force_switch", "gen", "heal", "id", "ignore_ability",
    "ignore_defensive", "ignore_evasion", "ignore_immunity", "is_protect_counter",
    "is_protect_move", "is_side_protect_move", "is_z", "n_hit", "no_pp_boosts",
    "non_ghost_target", "priority", "pseudo_weather", "recoil", "request_target",
    "secondary", "self_boost", "self_destruct", "self_switch", "side_condition",
    "sleep_usable", "slot_condition", "stalling_move", "status", "steals_boosts", "target",
    "terrain", "thaws_target", "type", "use_target_offensive", "volatile_status", "weather",
    "z_move_boost", "z_move_effect", "z_move_power",
}


def _assert_complete(actual: set[str], *groups: set[str]) -> None:
    classified = set().union(*groups)
    assert classified == actual, {
        "unclassified": sorted(actual - classified),
        "stale": sorted(classified - actual),
    }
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert not group & other, f"property classified twice: {sorted(group & other)}"


def test_every_double_battle_property_is_explicitly_classified() -> None:
    _assert_complete(
        _properties(DoubleBattle),
        BATTLE_CAPTURED,
        BATTLE_DISABLED_BY_FORMAT,
        BATTLE_DERIVED_OR_NON_MECHANICAL,
    )


def test_every_pokemon_property_is_explicitly_classified() -> None:
    _assert_complete(
        _properties(Pokemon),
        POKEMON_CAPTURED,
        POKEMON_DISABLED_BY_FORMAT,
        POKEMON_DERIVED_OR_NON_MECHANICAL,
    )


def test_every_move_property_is_dynamic_or_in_the_static_catalog() -> None:
    _assert_complete(_properties(Move), MOVE_DYNAMIC_CAPTURED, MOVE_STATIC_CATALOG)
