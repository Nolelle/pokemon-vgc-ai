"""Unit tests for `vgc.team_preview` -- lightweight stand-ins for poke-env's `Pokemon`/
`AbstractBattle` (same style as tests/test_sets.py's `_FakePokemon`) so these don't need
a real battle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vgc.damage import PokemonState
from vgc.models import PolicyConfig
from vgc.team_preview import _preview_speed, build_team_order


@dataclass
class _FakeMon:
    species: str
    item: str | None = None
    ability: str | None = None
    boosts: dict = field(default_factory=dict)
    status: Any = None
    current_hp_fraction: float = 1.0
    current_hp: int | None = None
    evs: list[int] | None = None
    nature: str | None = None
    moves: dict = field(default_factory=dict)
    _selected_in_teampreview: bool = False


class _FakeBattle:
    def __init__(self, our_team, opp_team):
        self.team = {f"p1: mon{i}": mon for i, mon in enumerate(our_team)}
        self.teampreview_opponent_team = opp_team


def _mon(species, moves, **kwargs):
    return _FakeMon(species=species, moves=dict.fromkeys(moves), **kwargs)


def _our_team():
    return [
        _mon("charizard", ["heatwave", "protect"], item="charizarditey", ability="blaze"),
        _mon("farigiraf", ["psychic", "trickroom", "helpinghand"], ability="armortail"),
        _mon("venusaur", ["leafstorm", "sleeppowder", "protect"], ability="chlorophyll"),
        _mon(
            "garchomp", ["earthquake", "dragonclaw", "protect"], item="lifeorb", ability="roughskin"
        ),
        _mon("incineroar", ["flareblitz", "fakeout", "partingshot"], ability="intimidate"),
        _mon("sylveon", ["hypervoice", "protect"], ability="pixilate"),
    ]


def _opp_team():
    return [
        _mon("klefki", ["playrough", "protect"], ability="prankster"),
        _mon("garchomp", ["earthquake", "protect"], ability="roughskin"),
        _mon("incineroar", ["flareblitz", "fakeout"], ability="intimidate"),
        _mon("sylveon", ["hypervoice", "protect"], ability="pixilate"),
        _mon("venusaur", ["leafstorm", "sleeppowder"], ability="chlorophyll"),
        _mon("charizard", ["heatwave", "protect"], ability="blaze"),
    ]


def test_returns_a_valid_team_order_string() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    order = build_team_order(battle, PolicyConfig())

    assert order.startswith("/team ")
    digits = order.removeprefix("/team ")
    assert len(digits) == 4
    picks = [int(d) for d in digits]
    assert all(1 <= pick <= 6 for pick in picks)
    assert len(set(picks)) == 4  # no repeats


def test_leads_are_a_subset_of_the_picked_four() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    order = build_team_order(battle, PolicyConfig())
    digits = [int(d) for d in order.removeprefix("/team ")]
    leads, bench = digits[:2], digits[2:]
    assert set(leads).isdisjoint(bench)
    assert len(leads) == 2
    assert len(bench) == 2


def test_marks_picked_pokemon_as_selected_in_teampreview() -> None:
    our_team = _our_team()
    battle = _FakeBattle(our_team, _opp_team())
    order = build_team_order(battle, PolicyConfig())
    digits = [int(d) for d in order.removeprefix("/team ")]

    for i, mon in enumerate(our_team, start=1):
        assert mon._selected_in_teampreview == (i in digits)


def test_falls_back_gracefully_with_no_opponent_preview() -> None:
    battle = _FakeBattle(_our_team(), [])
    order = build_team_order(battle, PolicyConfig())
    assert order == "/team 1234"


def test_sun_mega_is_recognized_as_a_lead_mode() -> None:
    battle = _FakeBattle(_our_team(), _our_team())

    order = build_team_order(battle, PolicyConfig())

    leads = {int(digit) for digit in order.removeprefix("/team ")[:2]}
    assert 1 in leads  # Charizard-Y's Mega stats and Drought are no longer ignored.


def test_chlorophyll_speed_is_doubled_in_preview_sun() -> None:
    venusaur = PokemonState(species_id="venusaur", ability="chlorophyll")

    assert _preview_speed(venusaur, "sun") == 2 * _preview_speed(venusaur, None)


def test_preview_carries_engine_closer_lead_backline_and_mega_plan_into_battle() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    order = build_team_order(battle, PolicyConfig())
    plan = battle._vgc_preview_plan

    picked = {to_id for to_id in plan.picked_species}
    assert plan.our_closer_species in picked
    assert plan.default_mega_species == "charizard"
    assert len(plan.lead_functions) >= 2
    assert len(plan.picked_species) == 4
    assert len(plan.lead_species) == 2
    assert plan.lead_covers_engine
    assert plan.balanced_structure["two_attackers"]
    assert plan.balanced_structure["closer"]
    assert plan.speed_modes
    assert order.startswith("/team ")


# --- iteration 6 integration: opponent preview prediction shifts our own pick --------


def _teambuilder_team(packed_text: str):
    from poke_env.battle.pokemon import Pokemon
    from poke_env.teambuilder.teambuilder import Teambuilder

    team = []
    for tb_mon in Teambuilder.parse_packed_team(packed_text):
        mon = Pokemon(gen=9, teambuilder=tb_mon)
        mon._active = False
        mon._current_hp = mon.max_hp
        team.append(mon)
    return team


def _bare_opponent(species: str):
    from poke_env.battle.pokemon import Pokemon

    mon = Pokemon(gen=9, species=species)
    mon._active = False
    mon._current_hp = 100
    mon._max_hp = 100
    return mon


def test_predicted_rain_engine_opponent_shifts_our_own_bring_four(monkeypatch) -> None:
    """Integration-level test for iteration 6's gate-passed integration
    (docs/preview_prediction_plan.md step 3): an opponent previewing a Drizzle setter
    (Pelipper) + a Swift Swim sweeper (Barraskewda) -- a textbook rain engine -- should,
    once corpus usage strongly favors that pair being brought together (monkeypatched
    set_priors appearances, isolating the effect from real-corpus noise), change which
    4 of OUR OWN 6 we select: Torkoal (Fire, weak to Water, a bad answer to a rain
    engine) should be DROPPED in favor of Sylveon (Fairy, a clean matchup against
    Water/Flying) once `use_preview_prediction` correctly weights their likely bring
    over the flat assume-all-6-equally-likely baseline.
    """
    import vgc.preview_predict as preview_predict_module

    real_load_set_priors = preview_predict_module.load_set_priors

    def boosted_priors(path=None):
        real = real_load_set_priors(path)
        species = dict(real.get("species") or {})
        species["pelipper"] = {**species.get("pelipper", {}), "appearances": 100_000}
        species["barraskewda"] = {**species.get("barraskewda", {}), "appearances": 100_000}
        return {"meta": real.get("meta", {}), "species": species}

    monkeypatch.setattr(preview_predict_module, "load_set_priors", boosted_priors)

    our_team_text = (
        "Raichu||Leftovers|Static|Thunderbolt,Protect,FocusBlast,VoltSwitch|Timid|2,,,32,,32||||50|]"
        "Torkoal||WhiteHerb|Drought|Overheat,Protect,SolarBeam,RapidSpin|Modest|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FlareBlitz,Protect,DarkestLariat,PartingShot|Careful"
        "|32,,14,,20,||||50|]"
        "Charizard||CharizarditeY|Blaze|HeatWave,Protect,SolarBeam,WeatherBall|Modest|10,,,32,,24||||50|]"
        "Camerupt||Leftovers|SolidRock|EarthPower,Protect,Overheat,RockSlide|Modest|2,,,32,,32||||50|]"
        "Sylveon||FairyFeather|Pixilate|HyperVoice,Protect,HyperBeam,QuickAttack|Modest"
        "|9,,22,20,,15||||50|"
    )
    our_team = _teambuilder_team(our_team_text)
    opp_species = ["pelipper", "barraskewda", "tyranitar", "incineroar", "milotic", "excadrill"]

    def make_battle():
        opp_team = [_bare_opponent(s) for s in opp_species]
        return _FakeBattle(our_team, opp_team)

    config_on = PolicyConfig(
        use_preview_prediction=True, team_preview_opponent_worst_case_weight=0.9
    )
    config_off = PolicyConfig(use_preview_prediction=False)

    order_on = build_team_order(make_battle(), config_on)
    order_off = build_team_order(make_battle(), config_off)

    picked_on = {our_team[int(d) - 1].species for d in order_on.removeprefix("/team ")}
    picked_off = {our_team[int(d) - 1].species for d in order_off.removeprefix("/team ")}

    assert picked_on != picked_off
    assert "torkoal" not in picked_on
    assert "torkoal" in picked_off
    assert "sylveon" in picked_on


def test_predicted_opponent_leads_seed_the_preview_plan() -> None:
    our_team = _teambuilder_team(
        "Raichu||Leftovers|Static|Thunderbolt,Protect,FocusBlast,VoltSwitch|Timid|2,,,32,,32||||50|]"
        "Torkoal||WhiteHerb|Drought|Overheat,Protect,SolarBeam,RapidSpin|Modest|2,,,32,,32||||50|]"
        "Incineroar||SitrusBerry|Intimidate|FlareBlitz,Protect,DarkestLariat,PartingShot|Careful"
        "|32,,14,,20,||||50|]"
        "Charizard||CharizarditeY|Blaze|HeatWave,Protect,SolarBeam,WeatherBall|Modest|10,,,32,,24||||50|]"
        "Camerupt||Leftovers|SolidRock|EarthPower,Protect,Overheat,RockSlide|Modest|2,,,32,,32||||50|]"
        "Sylveon||FairyFeather|Pixilate|HyperVoice,Protect,HyperBeam,QuickAttack|Modest"
        "|9,,22,20,,15||||50|"
    )
    opp_species = ["pelipper", "barraskewda", "tyranitar", "incineroar", "milotic", "excadrill"]
    opp_team = [_bare_opponent(s) for s in opp_species]
    battle = _FakeBattle(our_team, opp_team)

    build_team_order(battle, PolicyConfig(use_preview_prediction=True))
    plan = battle._vgc_preview_plan

    assert plan.predicted_opponent_leads is not None
    assert len(plan.predicted_opponent_leads) == 2
    assert set(plan.predicted_opponent_leads).issubset(set(opp_species))
    assert "predicted_opponent_leads" in plan.summary()


def test_use_preview_prediction_off_leaves_predicted_opponent_leads_none() -> None:
    battle = _FakeBattle(_our_team(), _opp_team())
    build_team_order(battle, PolicyConfig(use_preview_prediction=False))
    plan = battle._vgc_preview_plan
    assert plan.predicted_opponent_leads is None
