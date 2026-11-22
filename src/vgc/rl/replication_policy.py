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
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.data import to_id_str
from torch import nn

from vgc.data import canonical_species_id, load_items, load_moves, load_species, load_typechart

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOSTS = (*STATS[1:], "accuracy", "evasion")
PREVIEW_COUNT = 6
CONDITION_BINS = 64
CHECKPOINT_VERSION = 2
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2)
_V1_POKEMON_SCALARS = 16
_V2_POKEMON_SCALARS = 22
_V2_FIELD_TIMERS = 16
_TERRAIN_FIELDS = (
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
)
_SCREEN_CONDITIONS = (
    SideCondition.REFLECT,
    SideCondition.LIGHT_SCREEN,
    SideCondition.AURORA_VEIL,
)


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


def _remaining_fraction(start_turn: int | None, current_turn: int, duration: int) -> float:
    if start_turn is None:
        return 0.0
    elapsed = max(0, current_turn - int(start_turn))
    return max(0.0, float(duration - elapsed) / float(duration))


def _field_timer_features(battle) -> np.ndarray:
    """Explicit zero-collision active flags and remaining-turn fractions for core VGC timers."""
    turn = int(getattr(battle, "turn", 0) or 0)
    weather = getattr(battle, "weather", None) or {}
    fields = getattr(battle, "fields", None) or {}
    ours = getattr(battle, "side_conditions", None) or {}
    theirs = getattr(battle, "opponent_side_conditions", None) or {}

    weather_active = float(bool(weather))
    weather_rem = (
        max(_remaining_fraction(start, turn, 5) for start in weather.values()) if weather else 0.0
    )

    tr_start = fields.get(Field.TRICK_ROOM)
    tr_active = float(tr_start is not None)
    tr_rem = _remaining_fraction(tr_start, turn, 5)

    terrain_starts = [fields[t] for t in _TERRAIN_FIELDS if t in fields]
    terrain_active = float(bool(terrain_starts))
    terrain_rem = (
        max(_remaining_fraction(start, turn, 5) for start in terrain_starts)
        if terrain_starts
        else 0.0
    )

    our_tw_start = ours.get(SideCondition.TAILWIND)
    our_tw_active = float(our_tw_start is not None)
    our_tw_rem = _remaining_fraction(our_tw_start, turn, 4)

    opp_tw_start = theirs.get(SideCondition.TAILWIND)
    opp_tw_active = float(opp_tw_start is not None)
    opp_tw_rem = _remaining_fraction(opp_tw_start, turn, 4)

    our_screens = [ours[s] for s in _SCREEN_CONDITIONS if s in ours]
    our_screen_active = float(bool(our_screens))
    our_screen_rem = (
        max(_remaining_fraction(start, turn, 5) for start in our_screens) if our_screens else 0.0
    )

    opp_screens = [theirs[s] for s in _SCREEN_CONDITIONS if s in theirs]
    opp_screen_active = float(bool(opp_screens))
    opp_screen_rem = (
        max(_remaining_fraction(start, turn, 5) for start in opp_screens) if opp_screens else 0.0
    )

    our_safeguard = float(SideCondition.SAFEGUARD in ours)
    opp_safeguard = float(SideCondition.SAFEGUARD in theirs)

    return np.asarray(
        [
            weather_active,
            weather_rem,
            tr_active,
            tr_rem,
            terrain_active,
            terrain_rem,
            our_tw_active,
            our_tw_rem,
            opp_tw_active,
            opp_tw_rem,
            our_screen_active,
            our_screen_rem,
            opp_screen_active,
            opp_screen_rem,
            our_safeguard,
            opp_safeguard,
        ],
        dtype=np.float32,
    )


def encode_decision(battle, orders=None, *, preview=False, version: int = CHECKPOINT_VERSION):
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(f"unsupported replication encoding version: {version}")
    species, moves, items, abilities = _dex()[0]
    rows, active_at = _pokemon_rows(battle)
    pokemon = np.zeros((12, 8), dtype=np.int64)
    scalar_base = _V2_POKEMON_SCALARS if version >= 2 else _V1_POKEMON_SCALARS
    scalars = np.zeros((12, scalar_base + CONDITION_BINS), dtype=np.float32)
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
        if version >= 2:
            scalars[i, 21] = float(i < 6)
        if revealed:
            scalars[i, 4:11] = [mon.boosts.get(name, 0) / 6 for name in BOOSTS]
            scalars[i, 11:16] = [
                ((request_entry or {}).get("stats") or {}).get(name, 0) / 255 for name in STATS[1:]
            ]
            if version >= 2:
                scalars[i, 16:21] = [
                    min(int(getattr(mon, "protect_counter", 0) or 0), 4) / 4.0,
                    float(bool(getattr(mon, "first_turn", False))),
                    float(bool(getattr(mon, "must_recharge", False))),
                    float(bool(getattr(mon, "preparing", False))),
                    min(int(getattr(mon, "status_counter", 0) or 0), 6) / 6.0,
                ]
            scalars[i, scalar_base:] = _conditions(mon.effects)
    field_parts = [
        np.asarray([battle.turn / 100, float(battle.teampreview)], dtype=np.float32),
    ]
    if version >= 2:
        field_parts.append(_field_timer_features(battle))
    field_parts.extend(
        (
            _conditions(battle.weather),
            _conditions(battle.fields),
            _conditions(battle.side_conditions),
            _conditions(battle.opponent_side_conditions),
        )
    )
    field = np.concatenate(field_parts)
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
    def __init__(self, width=48, layers=2, *, version: int = CHECKPOINT_VERSION):
        super().__init__()
        if version not in SUPPORTED_CHECKPOINT_VERSIONS:
            raise ValueError(f"unsupported replication policy version: {version}")
        self.width = width
        self.layers = layers
        self.version = version
        (species, moves, items, abilities), species_rows, move_rows = _dex()
        self.register_buffer("species_dex", torch.from_numpy(species_rows))
        self.register_buffer("move_dex", torch.from_numpy(move_rows))
        self.species = nn.Embedding(len(species) + 1, width, padding_idx=0)
        self.moves = nn.Embedding(len(moves) + 1, width, padding_idx=0)
        self.items = nn.Embedding(len(items) + 1, width, padding_idx=0)
        self.abilities = nn.Embedding(len(abilities) + 1, width, padding_idx=0)
        if version >= 2:
            self.side_embedding = nn.Embedding(2, width)
            self.own_slot_embedding = nn.Embedding(PREVIEW_COUNT + 1, width, padding_idx=0)
            self.register_buffer(
                "side_ids",
                torch.tensor([0] * PREVIEW_COUNT + [1] * PREVIEW_COUNT, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "own_slot_ids",
                torch.tensor(
                    list(range(1, PREVIEW_COUNT + 1)) + [0] * PREVIEW_COUNT, dtype=torch.long
                ),
                persistent=False,
            )
            pokemon_scalar_dim = _V2_POKEMON_SCALARS + CONDITION_BINS
            field_dim = 2 + _V2_FIELD_TIMERS + 4 * CONDITION_BINS
        else:
            self.positions = nn.Embedding(12, width)
            pokemon_scalar_dim = _V1_POKEMON_SCALARS + CONDITION_BINS
            field_dim = 2 + 4 * CONDITION_BINS
        self.active = nn.Embedding(3, width)
        self.move_features = nn.Linear(move_rows.shape[1], width, bias=False)
        self.species_features = nn.Linear(species_rows.shape[1], width, bias=False)
        self.pokemon_features = nn.Linear(pokemon_scalar_dim, width)
        self.field_features = nn.Linear(field_dim, width)
        block = nn.TransformerEncoderLayer(
            width,
            4,
            2 * width,
            dropout=0,
            batch_first=True,
            norm_first=True,
        )
        self.board = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        if version >= 2:
            self.value_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        else:
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

    def _slot_embeddings(self):
        if self.version >= 2:
            return self.side_embedding(self.side_ids) + self.own_slot_embedding(self.own_slot_ids)
        return self.positions.weight

    def forward(
        self,
        pokemon,
        scalars,
        field,
        candidates=None,
        mask=None,
        *,
        preview=False,
        return_value_logit=False,
    ):
        species, items, abilities, active = pokemon[..., :4].unbind(-1)
        tokens = (
            self.species(species)
            + self.species_features(self.species_dex[species])
            + self.items(items)
            + self.abilities(abilities)
            + self.active(active)
            + self._slot_embeddings()
            + self._move_repr(pokemon[..., 4:]).sum(dim=-2)
            + self.pokemon_features(scalars)
        )
        encoded = self.board(torch.cat((self.field_features(field)[:, None], tokens), dim=1))
        context = encoded[:, 0]
        raw_value = self.value_head(context).squeeze(-1)
        if self.version >= 2:
            value_logit = raw_value
            value = 2.0 * torch.sigmoid(value_logit) - 1.0
        else:
            value_logit = None
            value = raw_value
        if candidates is None:
            if return_value_logit:
                return None, value, value_logit
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
        masked_logits = logits.squeeze(-1).masked_fill(~mask, -torch.inf)
        if return_value_logit:
            return masked_logits, value, value_logit
        return masked_logits, value

    @property
    def device(self):
        return self.species.weight.device

    @torch.no_grad()
    def score(self, battle, orders):
        inputs = stack_decisions(
            [encode_decision(battle, orders, version=self.version)], self.device
        )
        logits, value = self(*inputs)
        return logits[0].cpu().numpy(), float(value.item())

    @torch.no_grad()
    def value(self, battle):
        _, value = self(
            *stack_decisions([encode_decision(battle, version=self.version)], self.device)
        )
        return float(value.item())

    @torch.no_grad()
    def values(self, battles):
        if not battles:
            return []
        _, values = self(
            *stack_decisions(
                [encode_decision(b, version=self.version) for b in battles], self.device
            )
        )
        return [float(v) for v in values.cpu().tolist()]

    @torch.no_grad()
    def choose_preview(self, battle):
        state = encode_decision(battle, preview=True, version=self.version)
        logits, _ = self(*stack_decisions([state], self.device), preview=True)
        return preview_order(state.candidates[logits[0].argmax().item()])

    def checkpoint(self):
        return {
            "replication_version": self.version,
            "width": self.width,
            "layers": self.layers,
            "model": self.state_dict(),
            "vocabularies": _dex()[0],
        }


def preview_order(choice):
    return "/team " + "".join(str(slot) for slot in choice)


def load_replication_policy(path: str | Path, device: str = "cpu") -> ReplicationPolicy:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    version = checkpoint.get("replication_version")
    if version not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise ValueError(f"unsupported replication policy checkpoint: {path}")
    if checkpoint.get("vocabularies") != _dex()[0]:
        raise ValueError(f"Champions vocabulary differs from the checkpoint: {path}")
    model = ReplicationPolicy(
        width=checkpoint["width"], layers=checkpoint["layers"], version=version
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()
