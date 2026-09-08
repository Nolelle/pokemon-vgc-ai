"""Tests for the archetype-tagged team pool's domain layer (`vgc.archetypes`) and the
pure (non-CLI) parts of `tools/build_archetype_pool.py`'s anchor conversion.

CLI legality validation (`validate_team`) is deliberately NOT exercised here -- it's
slow and needs a local Showdown checkout/node -- see `tools/build_archetype_pool.py`
itself (run manually / at pool-build time) for that check.
"""

from __future__ import annotations

from vgc.archetypes import (
    ANCHOR_ID_TO_LABEL,
    ARCHETYPES,
    WEATHER_ABILITY_MAP,
    active_weathers,
    anchors_for,
    classify_team,
    core_set_options,
    weather_mega_stone_map,
)
from vgc.meta import load_meta_teams

# --- Assignment table integrity: every anchor id is real and used exactly once -------


def test_every_archetype_anchor_id_is_a_real_anchor_team() -> None:
    real_ids = {team["id"] for team in load_meta_teams()}
    for archetype in ARCHETYPES:
        for anchor_id in archetype.anchor_ids:
            assert anchor_id in real_ids, f"{archetype.label}: unknown anchor id {anchor_id!r}"


def test_every_anchor_team_is_assigned_to_exactly_one_archetype() -> None:
    real_ids = {team["id"] for team in load_meta_teams()}
    assigned_ids = [anchor_id for archetype in ARCHETYPES for anchor_id in archetype.anchor_ids]
    # No duplicates across archetypes (each anchor belongs to exactly one archetype)...
    assert len(assigned_ids) == len(set(assigned_ids))
    # ...and every real anchor team is covered, none left unassigned.
    assert set(assigned_ids) == real_ids
    assert ANCHOR_ID_TO_LABEL.keys() == real_ids


def test_archetype_count_is_in_the_target_range() -> None:
    # The task's target: "roughly 5-8 archetypes" grouping the 10 anchor teams, not 10
    # forced singletons.
    assert 5 <= len(ARCHETYPES) <= 8


def test_core_species_is_small_and_nonempty() -> None:
    for archetype in ARCHETYPES:
        assert 2 <= len(archetype.core_species) <= 4, archetype.label


# --- Core species actually appear in every one of their archetype's anchors ---------


def test_core_species_present_in_every_anchor_of_its_archetype() -> None:
    for archetype in ARCHETYPES:
        for anchor in anchors_for(archetype.label):
            anchor_species = {entry["species_id"] for entry in anchor["sets"]}
            missing = set(archetype.core_species) - anchor_species
            assert not missing, (
                f"{archetype.label}'s core {archetype.core_species} missing "
                f"{missing} from anchor {anchor['id']}"
            )


def test_core_set_options_only_returns_entries_for_core_species() -> None:
    for archetype in ARCHETYPES:
        options = core_set_options(archetype)
        assert set(options) == set(archetype.core_species)
        for species_id, entries in options.items():
            assert entries, f"{archetype.label}/{species_id}: no known-good sets found"
            for entry in entries:
                assert entry["species_id"] == species_id


# --- Weather identity: each archetype's `weather` field is consistent with what its
# own core actually produces (the ground truth the flex-exclusion filter in
# tools/build_archetype_pool.py relies on to keep flex slots from smuggling in a
# different active weather source) ---------------------------------------------------


def test_weather_field_is_a_recognized_label_or_none() -> None:
    valid_labels = set(WEATHER_ABILITY_MAP.values()) | {None}
    for archetype in ARCHETYPES:
        assert archetype.weather in valid_labels, archetype.label


def test_archetype_weather_matches_what_its_own_core_actually_produces() -> None:
    for archetype in ARCHETYPES:
        produced: set[str] = set()
        for anchor in anchors_for(archetype.label):
            core_entries = [e for e in anchor["sets"] if e["species_id"] in archetype.core_species]
            pairs = [(entry["ability_id"], entry["item_id"]) for entry in core_entries]
            produced |= active_weathers(pairs)
        if archetype.weather is None:
            # A weather=None archetype's core must produce NO weather at all -- that's
            # exactly why any weather setter in its flex pool is unconditionally
            # off-archetype (see _filter_flex_pool_for_weather).
            assert produced == set(), (archetype.label, produced)
        else:
            assert archetype.weather in produced, (archetype.label, archetype.weather, produced)


def test_weather_mega_stone_map_detects_charizard_y_and_tyranitar_mega() -> None:
    # These are exactly the two real leaks the coordinator's ability-level scan found
    # (a flex Charizard's Charizardite Y smuggling sun into rain_offense/sand_offense
    # variants) -- confirms the general, data-driven detection actually catches them
    # without any Charizard/Tyranitar-specific code.
    stones = weather_mega_stone_map()
    assert stones["charizarditey"] == "sun"
    assert stones["tyranitarite"] == "sand"


def test_active_weathers_detects_the_ability_mechanism() -> None:
    assert active_weathers([("drought", None), ("roughskin", "lifeorb")]) == {"sun"}


def test_active_weathers_detects_the_mega_stone_mechanism() -> None:
    # Charizard's OWN declared (non-Mega) ability is Blaze, not Drought -- but holding
    # Charizardite Y still counts as an active sun source, since Mega Evolution
    # auto-swaps the ability regardless of what the set declares.
    assert active_weathers([("blaze", "charizarditey")]) == {"sun"}


def test_double_weather_anchor_legitimately_produces_two_weathers() -> None:
    # The anchor exemption in tools/build_archetype_pool.py's guard exists because this
    # is REAL curated data, not a generation bug -- confirm active_weathers agrees.
    double_weather = next(team for team in load_meta_teams() if team["id"] == "double_weather")
    pairs = [(entry["ability_id"], entry["item_id"]) for entry in double_weather["sets"]]
    assert active_weathers(pairs) == {"sun", "rain"}


# --- classify_team: every anchor's own species list round-trips to its archetype ----


def test_classify_team_recognizes_every_anchor_as_its_assigned_archetype() -> None:
    for team in load_meta_teams():
        species = [entry["species_id"] for entry in team["sets"]]
        expected_label = ANCHOR_ID_TO_LABEL[team["id"]]
        assert classify_team(species) == expected_label, team["id"]


def test_classify_team_is_order_independent() -> None:
    aerozard = next(team for team in load_meta_teams() if team["id"] == "aerozard")
    species = [entry["species_id"] for entry in aerozard["sets"]]
    assert classify_team(reversed(species)) == classify_team(species)


def test_classify_team_returns_none_for_an_obviously_unrelated_species_set() -> None:
    # None of these six ever co-occur as a defining core in any archetype above.
    assert classify_team(["pikachu", "eevee", "snorlax", "magikarp", "ditto", "gyarados"]) is None


def test_classify_team_returns_none_for_a_partial_core() -> None:
    # Only Charizard, not Charizard + Garchomp -- shouldn't false-positive the sun
    # archetype off a single shared species.
    assert classify_team(["charizard", "sylveon", "kingambit", "toxapex"]) is None


def test_classify_team_accepts_extra_flex_species_beyond_the_core() -> None:
    # A generated variant has 6 species total, only 2-3 of which are the fixed core --
    # containment (not exact match) is what should drive classification.
    assert (
        classify_team(["charizard", "garchomp", "pikachu", "eevee", "snorlax", "ditto"])
        == "charizard_sun_offense"
    )


# --- Packed-team conversion for one anchor (pure logic, no CLI) ---------------------


def test_anchor_to_generated_sets_preserves_stated_fields_and_fills_a_spread() -> None:
    from tools.build_archetype_pool import anchor_to_generated_sets

    aerozard = next(team for team in load_meta_teams() if team["id"] == "aerozard")
    charizard_entry = next(e for e in aerozard["sets"] if e["species_id"] == "charizard")

    # No usage spread available for "charizard" in this fake table -- exercises the
    # default_opponent_spread fallback path.
    generated = anchor_to_generated_sets(aerozard, spreads={})
    charizard_set = next(g for g in generated if g.species_id == "charizard")

    assert charizard_set.item_candidates == (charizard_entry["item_id"],)
    assert charizard_set.ability == charizard_entry["ability_id"]
    assert charizard_set.moves == tuple(charizard_entry["move_ids"])
    assert charizard_set.nature == charizard_entry["nature_id"]
    # A spread was filled in (not empty/all-zero) even though none was supplied.
    assert sum(charizard_set.sp.values()) > 0


def test_anchor_to_generated_sets_uses_usage_spread_when_available() -> None:
    from tools.build_archetype_pool import anchor_to_generated_sets

    aerozard = next(team for team in load_meta_teams() if team["id"] == "aerozard")
    spreads = {
        "charizard": [
            {"sp": {"hp": 4, "spa": 32, "spe": 30}, "nature": "timid", "weight": 0.9},
        ]
    }
    generated = anchor_to_generated_sets(aerozard, spreads)
    charizard_set = next(g for g in generated if g.species_id == "charizard")

    assert charizard_set.sp == {"hp": 4, "spa": 32, "spe": 30}
    # Nature stays the anchor's OWN stated nature (real, authoritative data) even though
    # the usage table's entry has a different one -- only the Stat Point spread is
    # "missing" data this function is meant to fill.
    assert charizard_set.nature == charizard_entry_nature(aerozard)


def charizard_entry_nature(aerozard: dict) -> str:
    return next(e for e in aerozard["sets"] if e["species_id"] == "charizard")["nature_id"]


def test_anchor_to_generated_sets_produces_six_sets_for_a_complete_anchor() -> None:
    from tools.build_archetype_pool import anchor_to_generated_sets

    for team in load_meta_teams():
        generated = anchor_to_generated_sets(team, spreads={})
        assert len(generated) == 6
        assert {g.species_id for g in generated} == {e["species_id"] for e in team["sets"]}


# --- Flex-pool weather exclusion (the actual bug fix, pure logic no CLI) ------------


def _flex_gen_set(species_id: str, ability: str, item_candidates: tuple[str, ...]):
    from vgc.selfplay_pool import GeneratedSet

    return GeneratedSet(
        species_id=species_id,
        item_candidates=item_candidates,
        ability=ability,
        moves=("protect", "tackle"),
        nature="jolly",
        sp={"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
    )


def test_filter_flex_pool_drops_a_species_whose_own_ability_sets_the_wrong_weather() -> None:
    from tools.build_archetype_pool import _filter_flex_pool_for_weather

    pool = [_flex_gen_set("tyranitar", "sandstream", ("tyranitarite",))]
    filtered = _filter_flex_pool_for_weather(pool, "rain", weather_mega_stone_map())
    assert filtered == []


def test_filter_flex_pool_strips_only_the_offending_mega_stone_keeps_species() -> None:
    from tools.build_archetype_pool import _filter_flex_pool_for_weather

    # Charizard's OWN ability here (blaze) is not a weather ability, so the species
    # stays flex-eligible -- but charizarditey (-> sun) must be stripped out of a rain
    # archetype's item candidates, exactly the coordinator's reported leak.
    pool = [_flex_gen_set("charizard", "blaze", ("charizarditey", "lifeorb"))]
    filtered = _filter_flex_pool_for_weather(pool, "rain", weather_mega_stone_map())
    assert len(filtered) == 1
    assert filtered[0].item_candidates == ("lifeorb",)


def test_filter_flex_pool_keeps_the_archetypes_own_weather_mega_stone() -> None:
    from tools.build_archetype_pool import _filter_flex_pool_for_weather

    pool = [_flex_gen_set("charizard", "blaze", ("charizarditey", "lifeorb"))]
    filtered = _filter_flex_pool_for_weather(pool, "sun", weather_mega_stone_map())
    assert filtered[0].item_candidates == ("charizarditey", "lifeorb")


def test_filter_flex_pool_excludes_all_weather_for_a_none_weather_archetype() -> None:
    from tools.build_archetype_pool import _filter_flex_pool_for_weather

    pool = [
        _flex_gen_set("tyranitar", "sandstream", ("tyranitarite",)),
        _flex_gen_set("charizard", "blaze", ("charizarditey", "lifeorb")),
        _flex_gen_set("garchomp", "roughskin", ("lifeorb",)),
    ]
    filtered = _filter_flex_pool_for_weather(pool, None, weather_mega_stone_map())
    assert [g.species_id for g in filtered] == ["charizard", "garchomp"]
    assert filtered[0].item_candidates == ("lifeorb",)
