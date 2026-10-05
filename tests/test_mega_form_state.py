"""A Pokemon that has Mega Evolved must be modelled as its Mega forme.

poke-env keeps `Pokemon.species` as the BASE species after `|-mega|`, so live states built
from `species` carried base stats/types with the Mega ability. These tests pin the fix on
real poke-env objects (the `mega_evolve` path and the full protocol path) plus the opt-in
`search_model_opponent_mega` response model.
"""

from __future__ import annotations

import logging

from poke_env.battle import DoubleBattle, Pokemon

import vgc.poke_env_compat  # noqa: F401  (installs the exact-forme `mega_evolve` guard)
from vgc.damage import PokemonState
from vgc.data import load_species
from vgc.evaluator import _our_pokemon_state
from vgc.models import PolicyConfig
from vgc.search import _enumerate_opp_responses, _opp_slot_candidates, resolve_exchange
from vgc.sets import evolved_mega_id, opponent_state
from tests.test_search import _build_ctx, _fake_order, _fake_single, _mon


def test_opponent_state_is_the_mega_forme_after_mega_evolve() -> None:
    mon = Pokemon(9, species="charizard")
    before = opponent_state(mon)
    mon.mega_evolve("Charizardite Y")

    after = opponent_state(mon)

    assert mon.species == "charizard"  # the poke-env behaviour this guards against
    assert (before.species_id, before.ability) == ("charizard", None)
    assert (after.species_id, after.ability) == ("charizardmegay", "drought")
    assert after.item == "charizarditey"  # Mega Evolution reveals the stone
    assert after.stats()["spa"] > before.stats()["spa"]
    assert load_species()[after.species_id]["types"] == ["Fire", "Flying"]
    # Legacy control (PolicyConfig.mega_state_uses_evolved_form=False): base stats again.
    legacy = opponent_state(mon, evolved_form=False)
    assert legacy.species_id == "charizard" and legacy.stats() == before.stats()


def test_our_state_is_the_mega_forme_even_before_the_next_request() -> None:
    mon = Pokemon(9, species="salamence")
    mon.item = "salamencite"
    mon.ability = "intimidate"
    mon.forme_change("Salamence-Mega, L50, M")  # `|detailschange|`; species stays base

    state = _our_pokemon_state(mon)

    assert mon.species == "salamence"
    assert (state.species_id, state.ability) == ("salamencemega", "aerilate")
    assert _our_pokemon_state(mon, evolved_form=False).species_id == "salamence"


def test_protocol_path_keeps_the_exact_mega_forme() -> None:
    """Lucario-Mega-Z has a plain Lucario-Mega sibling: poke-env's `-mega` handler used to
    overwrite the exact forme from `|detailschange|` with the plain one."""
    battle = DoubleBattle("battle-test", "user", logging.getLogger("test"), gen=9)
    battle.player_role = "p1"
    for line in (
        "|switch|p2a: Lucario|Lucario, L50, M|100/100",
        "|switch|p2b: Gengar|Gengar, L50, M|100/100",
        "|detailschange|p2a: Lucario|Lucario-Mega-Z, L50, M",
        "|-mega|p2a: Lucario|Lucario|Lucarionite Z",
        "|detailschange|p2b: Gengar|Gengar-Mega, L50, M",
        "|-mega|p2b: Gengar|Gengar|Gengarite",
    ):
        battle.parse_message(line.split("|"))
    lucario = battle.get_pokemon("p2a: Lucario")
    gengar = battle.get_pokemon("p2b: Gengar")

    assert evolved_mega_id(lucario) == "lucariomegaz"
    assert opponent_state(lucario).species_id == "lucariomegaz"
    assert (
        opponent_state(lucario).stats()["spa"]
        > opponent_state(lucario, evolved_form=False).stats()["spa"]
    )
    assert opponent_state(gengar).species_id == "gengarmega"
    assert battle.opponent_used_mega_evolve


def test_unevolved_pokemon_are_untouched() -> None:
    for species in ("garchomp", "incineroar", "floetteeternal"):
        assert evolved_mega_id(Pokemon(9, species=species)) is None
    assert opponent_state(Pokemon(9, species="garchomp")).species_id == "garchomp"


# --- search_model_opponent_mega ---------------------------------------------------------


def _mega_holder_ctx(item: str | None, priors: dict | None = None):
    opp = PokemonState(
        "garchomp", sp_spread={"hp": 2, "atk": 32, "spe": 32}, nature="jolly", item=item
    )
    opp_mon = _mon(moves={"earthquake": None, "protect": None}, species="garchomp")
    our = PokemonState("incineroar", sp_spread={"hp": 32, "def": 32}, nature="careful")
    return _build_ctx(
        our_states=[our, None],
        opp_states=[opp, None],
        our_pokemon=[_mon(species="incineroar"), None],
        opp_pokemon=[opp_mon, None],
        priors=priors,
    )


def _megas(responses):
    return [r for r in responses if r.slot0.mega_state is not None]


def test_opponent_mega_is_off_by_default() -> None:
    ctx = _mega_holder_ctx("garchompite")
    assert _megas(_enumerate_opp_responses(ctx, PolicyConfig())) == []


def test_opponent_mega_twins_use_mega_stats_for_a_known_stone() -> None:
    ctx = _mega_holder_ctx("garchompite")
    config = PolicyConfig(search_model_opponent_mega=True)

    candidates = _opp_slot_candidates(0, ctx, config)
    plain = next(c for c in candidates if c.kind == "move" and c.mega_state is None)
    twin = next(c for c in candidates if c.kind == "move" and c.mega_state is not None)
    assert twin.mega_state.species_id == "garchompmega"
    assert twin.value > plain.value  # Mega Garchomp hits harder

    # In the exchange, the Mega state replaces the slot's state before it acts.
    response = next(
        r for r in _megas(_enumerate_opp_responses(ctx, config)) if r.slot0.kind == "move"
    )
    order = _fake_order(_fake_single("flamethrower", move_target=1), None)
    result = resolve_exchange(order, response, ctx, config)
    assert result.opp_states[0].species_id == "garchompmega"


def test_hidden_item_needs_a_stone_heavy_prior_and_only_one_mega_per_battle() -> None:
    config = PolicyConfig(search_model_opponent_mega=True)
    heavy = {"species": {"garchomp": {"appearances": 100, "items": {"garchompite": 70}}}}
    light = {"species": {"garchomp": {"appearances": 100, "items": {"garchompite": 20}}}}

    assert _megas(_enumerate_opp_responses(_mega_holder_ctx(None, heavy), config))
    assert not _megas(_enumerate_opp_responses(_mega_holder_ctx(None, light), config))
    assert not _megas(_enumerate_opp_responses(_mega_holder_ctx(None, None), config))

    used = _mega_holder_ctx("garchompite")
    used.battle.opponent_used_mega_evolve = True
    assert not _megas(_enumerate_opp_responses(used, config))

    two_holders = _mega_holder_ctx("garchompite")
    two_holders.opp_states[1] = PokemonState(
        "garchomp", sp_spread={"hp": 2, "atk": 32, "spe": 32}, nature="jolly", item="garchompite"
    )
    two_holders.opp_pokemon[1] = _mon(moves={"earthquake": None}, species="garchomp")
    for response in _enumerate_opp_responses(two_holders, config):
        assert not (response.slot0.mega_state and response.slot1.mega_state)
