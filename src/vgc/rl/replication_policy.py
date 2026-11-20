"""Small from-scratch policy and value network for the league replication.

The encoder reads the supplied side's battle view and the Champions dex. Never pass
a private simulator root or use usage data, inferred sets, or computed damage here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.data import to_id_str
from torch import nn

from vgc.data import canonical_species_id, load_items, load_moves, load_species, load_typechart

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = (*STATS[1:], "accuracy", "evasion")
PREVIEW_COUNT = 6
CONDITION_BINS = 64
CHECKPOINT_VERSION = 1


def _species_id(mon):
    return canonical_species_id(to_id_str(mon.species))


def _vocabulary(keys):
    return {key: i for i, key in enumerate(sorted(key for key in keys if key), 1)}


@lru_cache(maxsize=1)
def _dex():
    species, moves = load_species(), load_moves()
    species_ids = _vocabulary(species)
    move_ids = _vocabulary(moves)
    type_ids = _vocabulary(load_typechart())
    abilities = {
        to_id_str(ability) for entry in species.values() for ability in entry["abilities"].values()
    }
    vocabs = (species_ids, move_ids, _vocabulary(load_items()), _vocabulary(abilities))
    species_rows = np.zeros((len(species_ids) + 1, 6 + len(type_ids)), dtype=np.float32)
    move_rows = np.zeros((len(move_ids) + 1, 7 + len(type_ids)), dtype=np.float32)
    for key, i in species_ids.items():
        entry = species[key]
        species_rows[i, :6] = [entry["baseStats"][name] / 255 for name in STATS]
        for type_name in entry["types"]:
            type_id = type_ids.get(to_id_str(type_name))
            if type_id is not None:
                species_rows[i, 5 + type_id] = 1
    for key, i in move_ids.items():
        entry = moves[key]
        move_rows[i, :4] = [
            entry["basePower"] / 200,
            (1.0 if entry["accuracy"] is True else entry["accuracy"] / 100),
            entry["pp"] / 64,
            entry["priority"] / 7,
        ]
        category = ("Status", "Physical", "Special").index(entry["category"])
        move_rows[i, 4 + category] = 1
        move_rows[i, 6 + type_ids[to_id_str(entry["type"])]] = 1
    return vocabs, species_rows, move_rows


def _conditions(entries):
    vector = np.zeros(CONDITION_BINS, dtype=np.float32)
    for name in entries:
        digest = hashlib.blake2s(str(name).encode(), digest_size=2).digest()
        vector[int.from_bytes(digest, "little") % CONDITION_BINS] = 1
    return vector


def _move_ids(mon):
    return sorted(_dex()[0][1].get(move_id, 0) for move_id in mon.moves)[:4]


def _pokemon_rows(battle):
    our_actives = list(battle.active_pokemon)
    their_actives = list(battle.opponent_active_pokemon)
    request = getattr(battle, "last_request", None) or {}
    requested = {entry["ident"]: entry for entry in (request.get("side") or {}).get("pokemon", [])}
    # Preview wire numbers are positions in Showdown's request, independently of
    # the parser's current dictionary insertion order.
    if battle.teampreview and requested:
        own_idents = requested
    else:
        own_idents = battle.team
    our_rows = [
        (battle.team[ident], requested.get(ident), True)
        for ident in own_idents
        if battle.teampreview
        or not requested
        or ident in requested
        or battle.team[ident] in our_actives
    ]
    seen_enemies = list(battle.opponent_team.values())
    their_rows = [(mon, None, True) for mon in seen_enemies]
    unseen_enemies = list(battle.teampreview_opponent_team)
    for seen in seen_enemies:
        species = _species_id(seen)
        dex_entry = load_species().get(species, {})
        base = to_id_str(dex_entry.get("baseSpecies") or "")
        for candidate in (species, base):
            matching = next((mon for mon in unseen_enemies if _species_id(mon) == candidate), None)
            if matching is not None:
                unseen_enemies.remove(matching)
                break
    their_rows.extend((mon, None, False) for mon in unseen_enemies)
    padding = (None, None, False)
    rows = (
        our_rows[:PREVIEW_COUNT]
        + [padding] * max(0, PREVIEW_COUNT - len(our_rows))
        + their_rows[:PREVIEW_COUNT]
        + [padding] * max(0, PREVIEW_COUNT - len(their_rows))
    )
    active_at = {
        id(mon): slot
        for actives in (our_actives, their_actives)
        for slot, mon in enumerate(actives, start=1)
        if mon is not None
    }
    return rows, active_at


@dataclass(frozen=True)
class EncodedDecision:
    pokemon: np.ndarray
    scalars: np.ndarray
    field: np.ndarray
    candidates: np.ndarray | None = None
    preview: bool = False


def encode_decision(battle, orders=None, *, preview=False):
    species, moves, items, abilities = _dex()[0]
    rows, active_at = _pokemon_rows(battle)
    pokemon = np.zeros((12, 8), dtype=np.int64)
    scalars = np.zeros((12, 16 + CONDITION_BINS), dtype=np.float32)
    positions = {}
    active_rows = {}
    for i, (mon, request_entry, revealed) in enumerate(rows):
        if mon is None:
            continue
        positions[id(mon)] = i + 1
        active = active_at.get(id(mon), 0)
        if active:
            active_rows[(i < 6, active)] = i + 1
        pokemon[i, :4] = [
            species.get(_species_id(mon), 0),
            items.get(to_id_str(mon.item or ""), 0) if revealed else 0,
            abilities.get(to_id_str(mon.ability or ""), 0) if revealed else 0,
            active,
        ]
        known_moves = _move_ids(mon) if revealed else []
        pokemon[i, 4 : 4 + len(known_moves)] = known_moves
        status = mon.status
        scalars[i, :4] = [
            float(revealed),
            mon.current_hp_fraction if revealed else 0,
            (status.value / 10) if revealed and status is not None else 0,
            float(mon.fainted and revealed),
        ]
        if revealed:
            scalars[i, 4:11] = [mon.boosts.get(name, 0) / 6 for name in BOOSTS]
            scalars[i, 11:16] = [
                ((request_entry or {}).get("stats") or {}).get(name, 0) / 255 for name in STATS[1:]
            ]
            scalars[i, 16:] = _conditions(mon.effects)
    field = np.concatenate(
        (
            np.asarray([battle.turn / 100, float(battle.teampreview)], dtype=np.float32),
            _conditions(battle.weather),
            _conditions(battle.fields),
            _conditions(battle.side_conditions),
            _conditions(battle.opponent_side_conditions),
        )
    )
    candidates = None
    if preview:
        own_count = sum(mon is not None for mon, _, _ in rows[:PREVIEW_COUNT])
        choices = tuple(permutations(range(1, own_count + 1), 4))
        candidates = np.asarray(choices, dtype=np.int64).reshape(-1, 4)
    elif orders is not None:
        candidates = np.zeros((len(orders), 2, 5), dtype=np.int64)
        for i, order in enumerate(orders):
            for slot, single in enumerate((order.first_order, order.second_order)):
                actor = active_rows.get((True, slot + 1), 0)
                payload = single.order
                if isinstance(payload, Move):
                    target = single.move_target or 0
                    target_row = active_rows.get((target < 0, abs(target)), 0)
                    candidates[i, slot] = (
                        1,
                        moves.get(payload.id, 0),
                        actor,
                        target_row,
                        int(single.mega),
                    )
                elif isinstance(payload, Pokemon):
                    candidates[i, slot] = (2, 0, actor, positions.get(id(payload), 0), 0)
                else:
                    candidates[i, slot] = (0, 0, actor, 0, 0)
    return EncodedDecision(pokemon, scalars, field, candidates, preview)


def stack_decisions(observations, device):
    pokemon = torch.from_numpy(np.stack([state.pokemon for state in observations])).to(device)
    scalars = torch.from_numpy(np.stack([state.scalars for state in observations])).to(device)
    field = torch.from_numpy(np.stack([state.field for state in observations])).to(device)
    if observations[0].candidates is None:
        return pokemon, scalars, field, None, None
    count = max(len(state.candidates) for state in observations)
    candidates = np.zeros(
        (len(observations), count, *observations[0].candidates.shape[1:]), dtype=np.int64
    )
    mask = np.zeros((len(observations), count), dtype=bool)
    for i, state in enumerate(observations):
        candidates[i, : len(state.candidates)] = state.candidates
        mask[i, : len(state.candidates)] = True
    return (
        pokemon,
        scalars,
        field,
        torch.from_numpy(candidates).to(device),
        torch.from_numpy(mask).to(device),
    )


class ReplicationPolicy(nn.Module):
    def __init__(self, width=48, layers=2):
        super().__init__()
        self.width = width
        self.layers = layers
        (species, moves, items, abilities), species_rows, move_rows = _dex()
        self.register_buffer("species_dex", torch.from_numpy(species_rows))
        self.register_buffer("move_dex", torch.from_numpy(move_rows))
        self.species = nn.Embedding(len(species) + 1, width, padding_idx=0)
        self.moves = nn.Embedding(len(moves) + 1, width, padding_idx=0)
        self.items = nn.Embedding(len(items) + 1, width, padding_idx=0)
        self.abilities = nn.Embedding(len(abilities) + 1, width, padding_idx=0)
        self.positions = nn.Embedding(12, width)
        self.active = nn.Embedding(3, width)
        self.move_features = nn.Linear(move_rows.shape[1], width, bias=False)
        self.species_features = nn.Linear(species_rows.shape[1], width, bias=False)
        self.pokemon_features = nn.Linear(16 + CONDITION_BINS, width)
        self.field_features = nn.Linear(2 + 4 * CONDITION_BINS, width)
        block = nn.TransformerEncoderLayer(
            width,
            4,
            2 * width,
            dropout=0,
            batch_first=True,
            norm_first=True,
        )
        self.board = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.value_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1), nn.Tanh())
        self.action_kind = nn.Embedding(3, width)
        self.mega = nn.Embedding(2, width)
        self.action = nn.Sequential(nn.Linear(4 * width, width), nn.GELU())
        self.turn_policy = nn.Sequential(
            nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.preview_policy = nn.Sequential(
            nn.Linear(5 * width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def _move_repr(self, ids):
        return self.moves(ids) + self.move_features(self.move_dex[ids])

    def forward(self, pokemon, scalars, field, candidates=None, mask=None, *, preview=False):
        species, items, abilities, active = pokemon[..., :4].unbind(-1)
        tokens = (
            self.species(species)
            + self.species_features(self.species_dex[species])
            + self.items(items)
            + self.abilities(abilities)
            + self.active(active)
            + self.positions.weight
            + self._move_repr(pokemon[..., 4:]).sum(dim=-2)
            + self.pokemon_features(scalars)
        )
        encoded = self.board(torch.cat((self.field_features(field)[:, None], tokens), dim=1))
        context = encoded[:, 0]
        value = self.value_head(context).squeeze(-1)
        if candidates is None:
            return None, value
        entities = torch.cat((torch.zeros_like(encoded[:, :1]), encoded[:, 1:]), dim=1)

        def gather(indices):
            batch = torch.arange(len(indices), device=indices.device)
            batch = batch.reshape((len(indices),) + (1,) * (indices.ndim - 1))
            return entities[batch, indices]

        global_context = context[:, None, :].expand(-1, candidates.shape[1], -1)
        if preview:
            choices = gather(candidates).flatten(start_dim=-2)
            logits = self.preview_policy(torch.cat((global_context, choices), dim=-1))
        else:
            kind, move, actor, target, mega = candidates.unbind(-1)
            parts = self.action(
                torch.cat(
                    (
                        self.action_kind(kind) + self.mega(mega),
                        self._move_repr(move),
                        gather(actor),
                        gather(target),
                    ),
                    dim=-1,
                )
            ).flatten(start_dim=-2)
            logits = self.turn_policy(torch.cat((global_context, parts), dim=-1))
        return logits.squeeze(-1).masked_fill(~mask, -torch.inf), value

    @property
    def device(self):
        return self.species.weight.device

    @torch.no_grad()
    def score(self, battle, orders):
        inputs = stack_decisions([encode_decision(battle, orders)], self.device)
        logits, value = self(*inputs)
        return logits[0].cpu().numpy(), float(value.item())

    @torch.no_grad()
    def value(self, battle):
        _, value = self(*stack_decisions([encode_decision(battle)], self.device))
        return float(value.item())

    @torch.no_grad()
    def values(self, battles):
        if not battles:
            return []
        _, values = self(*stack_decisions([encode_decision(b) for b in battles], self.device))
        return [float(v) for v in values.cpu().tolist()]

    @torch.no_grad()
    def choose_preview(self, battle):
        state = encode_decision(battle, preview=True)
        logits, _ = self(*stack_decisions([state], self.device), preview=True)
        return preview_order(state.candidates[logits[0].argmax().item()])

    def checkpoint(self):
        return {
            "replication_version": CHECKPOINT_VERSION,
            "width": self.width,
            "layers": self.layers,
            "model": self.state_dict(),
            "vocabularies": _dex()[0],
        }


def preview_order(choice):
    return "/team " + "".join(str(slot) for slot in choice)


def load_replication_policy(path: str | Path, device: str = "cpu") -> ReplicationPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("replication_version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported replication policy checkpoint: {path}")
    if checkpoint.get("vocabularies") != _dex()[0]:
        raise ValueError(f"Champions vocabulary differs from the checkpoint: {path}")
    model = ReplicationPolicy(width=checkpoint["width"], layers=checkpoint["layers"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()
