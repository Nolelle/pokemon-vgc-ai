"""Sanity checks on the exported champions-mod data (data/champions/*.json).

These guard the #1 failure mode for this project: silently drifting back to vanilla
gen9 assumptions instead of the champions mod's actual (smaller, differently-legal)
data. See CLAUDE.md for the full list of mod quirks these encode.
"""

from __future__ import annotations

from vgc.data import load_items, load_learnsets, load_moves, load_species


def test_species_has_megas_with_stone_mapping() -> None:
    species = load_species()
    megas = [s for s in species.values() if s["isMega"]]
    assert megas, "expected at least one mega forme in species.json"

    garchomp_mega = species["garchompmega"]
    assert garchomp_mega["isMega"] is True
    assert garchomp_mega["baseSpecies"] == "Garchomp"
    assert garchomp_mega["requiredItem"] == "Garchompite"

    # Every LEGAL mega forme must carry the stone/item that triggers it -- a mega with no
    # requiredItem would be unusable by src/vgc/actions.py's future mega-selection logic.
    # (Rayquaza-Mega is the one real exception to "megas need a stone" -- it mega evolves
    # via the move Dragon Ascent instead -- but this mod also bans it outright
    # (isNonstandard: "Past", tier: "Illegal"), so it's excluded here along with any
    # other banned mega.)
    legal_megas = [m for m in megas if m["isNonstandard"] is None]
    assert legal_megas, "expected at least one legal mega forme in species.json"
    for mega in legal_megas:
        assert mega["requiredItem"], f"{mega['id']} is a legal mega forme with no requiredItem"


def test_known_illegal_item_absent() -> None:
    items = load_items()
    # Rocky Helmet is a real, common VGC item that this mod bans (isNonstandard: "Past"
    # in data/mods/champions/items.ts) -- items.json only keeps legal (isNonstandard is
    # None) items, so it must not appear here.
    assert "rockyhelmet" not in items
    # Spot-check a couple more of the vanilla-VGC staples this mod bans (see
    # data/mods/champions/items.ts). NOTE: Choice Band and Choice Specs are explicitly
    # banned (isNonstandard: "Past"), but Choice Scarf is NOT banned in this mod (no
    # override entry -> inherits vanilla gen9's standard/legal status) -- a correction to
    # the Phase 0 "no Choice items" assumption, confirmed here so it isn't silently wrong.
    for banned in ("choicespecs", "choiceband", "assaultvest", "heavydutyboots", "eviolite"):
        assert banned not in items, f"{banned} should not be a legal item in this mod"
    assert "choicescarf" in items, "Choice Scarf is legal in this mod (unlike Band/Specs)"


def test_chilling_water_replaces_scald_in_learnset() -> None:
    learnsets = load_learnsets()
    moves = load_moves()

    # Chilling Water is the champions-mod move that replaces Scald for at least some
    # learnsets (Toxapex is the validated regmb test-team case -- see teams/dev.paste.txt).
    assert "chillingwater" in moves
    toxapex_learnset = learnsets.get("toxapex")
    assert toxapex_learnset is not None
    assert "chillingwater" in toxapex_learnset


def test_legal_items_count_matches_expected_order_of_magnitude() -> None:
    # Phase 0 found ~148 legal items in this mod (vs. ~700+ in vanilla gen9). A count
    # wildly outside this range means the isNonstandard filter in
    # tools/export_champions_data.mjs broke.
    items = load_items()
    assert 100 <= len(items) <= 200
