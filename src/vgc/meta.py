"""Curated, team-level metagame knowledge.

Open Team Sheets reveal an opponent's item, ability, and moves in the actual battle, so
curated sets must never replace those live facts.  Their useful extra information is
team identity, the hidden nature, and human-described roles/lead patterns.  This module
recognizes an exact six-species lineup and exposes the matching set without coupling the
rest of the decision code to the JSON schema.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from vgc.config import REPO_ROOT
from vgc.damage import to_id
from vgc.data import load_items, load_learnsets, load_moves, load_natures, load_species

DEFAULT_META_TEAMS_PATH = REPO_ROOT / "data" / "meta" / "popular_teams_H8v7TEZcbXo.json"


@lru_cache(maxsize=4)
def load_meta_teams(path: str | None = None) -> tuple[dict[str, Any], ...]:
    """Load the curated teams. Missing optional data degrades to no recognition."""

    resolved = Path(path) if path is not None else DEFAULT_META_TEAMS_PATH
    if not resolved.exists():
        return ()
    with resolved.open() as file:
        payload = json.load(file)
    return tuple(payload.get("teams", ()))


def _species_id(value: Any) -> str:
    raw = value if isinstance(value, str) else getattr(value, "species", "")
    return to_id(raw) or ""


def _team_signature(values: Iterable[Any]) -> Counter[str]:
    return Counter(_species_id(value) for value in values if _species_id(value))


def recognize_meta_team(
    pokemon: Iterable[Any], teams: Iterable[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """Return a curated team only for an exact six-species match.

    Requiring all six species avoids false confidence when two archetypes share a core
    such as Charizard/Garchomp/Kingambit.
    """

    observed = list(pokemon)
    if len(observed) != 6:
        return None
    signature = _team_signature(observed)
    for team in teams if teams is not None else load_meta_teams():
        expected = Counter(entry["species_id"] for entry in team["sets"])
        if signature == expected:
            return team
    return None


def known_set_for_pokemon(team: dict[str, Any] | None, pokemon: Any) -> dict[str, Any] | None:
    """Find this Pokemon's set within a recognized curated team.

    Active Pokemon may already be Mega Evolved, so both the preview species id and the
    optional battle-form id are accepted. The held item breaks a tie defensively.
    """

    if team is None:
        return None
    species_id = _species_id(pokemon)
    candidates = [
        entry
        for entry in team["sets"]
        if species_id in {entry["species_id"], entry.get("battle_form_id")}
    ]
    if len(candidates) <= 1:
        return candidates[0] if candidates else None
    item_id = to_id(getattr(pokemon, "item", None))
    return next((entry for entry in candidates if entry["item_id"] == item_id), candidates[0])


def known_nature(team: dict[str, Any] | None, pokemon: Any) -> str | None:
    """Return the video-stated hidden nature for a Pokemon on a recognized team."""

    entry = known_set_for_pokemon(team, pokemon)
    return entry["nature_id"] if entry is not None else None


def validate_meta_teams(teams: Iterable[dict[str, Any]] | None = None) -> list[str]:
    """Validate curated ids against the exported Champions-mod ground truth."""

    species = load_species()
    items = load_items()
    moves = load_moves()
    natures = load_natures()
    learnsets = load_learnsets()
    errors: list[str] = []
    seen_ids: set[str] = set()
    seen_signatures: set[tuple[str, ...]] = set()

    for team in teams if teams is not None else load_meta_teams():
        team_id = team.get("id", "<missing-id>")
        if team_id in seen_ids:
            errors.append(f"duplicate team id: {team_id}")
        seen_ids.add(team_id)
        entries = team.get("sets", [])
        if len(entries) != 6:
            errors.append(f"{team_id}: expected 6 sets, got {len(entries)}")
        signature = tuple(sorted(entry.get("species_id", "") for entry in entries))
        if signature in seen_signatures:
            errors.append(f"{team_id}: duplicate species signature")
        seen_signatures.add(signature)

        for entry in entries:
            label = f"{team_id}/{entry.get('species_id', '<missing-species>')}"
            species_id = entry.get("species_id")
            form_id = entry.get("battle_form_id")
            if species_id not in species:
                errors.append(f"{label}: unknown species")
                continue
            if form_id is not None and form_id not in species:
                errors.append(f"{label}: unknown battle form {form_id}")
            if entry.get("item_id") not in items:
                errors.append(f"{label}: unknown or illegal item {entry.get('item_id')}")
            if entry.get("nature_id") not in natures:
                errors.append(f"{label}: unknown nature {entry.get('nature_id')}")

            legal_abilities = {
                to_id(ability)
                for candidate_id in (species_id, form_id)
                if candidate_id in species
                for ability in species[candidate_id].get("abilities", {}).values()
            }
            if entry.get("ability_id") not in legal_abilities:
                errors.append(f"{label}: ability {entry.get('ability_id')} not on base/form")

            legal_moves = learnsets.get(species_id, {})
            move_ids = entry.get("move_ids", [])
            if len(move_ids) != 4:
                errors.append(f"{label}: expected 4 moves, got {len(move_ids)}")
            for move_id in move_ids:
                if move_id not in moves:
                    errors.append(f"{label}: unknown move {move_id}")
                elif move_id not in legal_moves:
                    errors.append(f"{label}: {move_id} absent from Champions learnset")
    return errors
