"""Build the numbered option list and the context packet the model sees.

Everything here takes already-computed PUBLIC facts as plain strings/dicts. Nothing reads a
battle object, so nothing private can leak in by accident.

TODO(adapter): a public `DoubleBattle` -> facts adapter (our active/bench HP and moves, the
opponent's revealed info and set priors, field/weather/Trick Room, turn number) plugs in
where callers currently pass `fixed_sections` / `turn_sections`. It must go through the
same fog-safe boundary as live play (see CLAUDE.md "Public information only").
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from vgc.llm.types import ContextPacket, Option

INSTRUCTIONS = (
    "You advise a Pokemon VGC bot. Propose up to 3 of OUR joint moves, best first, by the "
    "option IDs listed under OPTIONS. Use only IDs from the current list and only "
    "Pokemon, moves and items that appear in this packet. Do no damage arithmetic; the "
    "engine computes outcomes and makes the final choice. Reply with the JSON schema given."
)

_PROTECT = {
    "protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "silktrap",
    "burningbulwark", "obstruct", "wideguard", "quickguard", "maxguard",
}  # fmt: skip


def new_request_id(turn: int | None = None) -> str:
    prefix = f"t{turn:03d}" if turn is not None else "t---"
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _order_text(order: Any) -> str:
    text = getattr(order, "message", None)
    return str(text if text else order)


def classify_order(order_text: str) -> str:
    tokens = order_text.lower().replace(",", " ").split()
    if any(t in _PROTECT for t in tokens):
        return "protect"
    if "switch" in tokens:
        return "switch"
    return "attack"


def build_options(
    scored_orders: Sequence[Any],
    max_options: int = 30,
    *,
    blind: bool = False,
    seed: str | None = None,
) -> list[Option]:
    """Number the engine's best orders P01.. in engine rank order.

    `scored_orders` is best-first and each item needs `.order` and `.score` (a
    `vgc.evaluator.ScoredOrder`). The top (max_options - 4) are kept, then topped up with
    the two best switch lines and two best Protect lines if not already in, then filled by
    rank. IDs are stable for a given ranking: P01 is always the engine's top pick.

    With ``blind`` the same options are numbered in a shuffled order (deterministic in
    ``seed``) and their notes carry only the move kind, never the engine's score or rank.
    """
    if not scored_orders or max_options <= 0:
        return []
    texts = [_order_text(s.order) for s in scored_orders]
    kinds = [classify_order(t) for t in texts]
    chosen: list[int] = list(range(min(len(texts), max(0, max_options - 4))))
    for kind in ("switch", "protect"):
        extra = [i for i in range(len(texts)) if kinds[i] == kind]
        have = [i for i in chosen if kinds[i] == kind]
        for i in extra:
            if len(have) >= 2 or len(chosen) >= max_options:
                break
            if i not in chosen:
                chosen.append(i)
                have.append(i)
    for i in range(len(texts)):
        if len(chosen) >= max_options:
            break
        if i not in chosen:
            chosen.append(i)
    chosen.sort()
    if blind:
        import random

        random.Random(seed or "").shuffle(chosen)
    top = float(scored_orders[0].score)
    width = max(2, len(str(len(chosen))))
    return [
        Option(
            id=f"P{n:0{width}d}",
            order=texts[i],
            note=kinds[i] if blind else f"engine score {float(scored_orders[i].score):.1f} "
            f"({float(scored_orders[i].score) - top:+.1f} vs top); {kinds[i]}",
            kind=kinds[i],
        )
        for n, i in enumerate(chosen, start=1)
    ]


def render_section(title: str, body: Any) -> str:
    """Render a facts section: a string as-is, a mapping as 'key: value' lines, an
    iterable as bullet lines."""
    if isinstance(body, str):
        text = body.strip()
    elif isinstance(body, Mapping):
        text = "\n".join(f"{k}: {v}" for k, v in body.items())
    elif isinstance(body, Iterable):
        text = "\n".join(f"- {line}" for line in body)
    else:
        text = str(body)
    return f"## {title}\n{text}"


def build_packet(
    fixed_sections: Mapping[str, Any],
    turn_sections: Mapping[str, Any],
    options: Sequence[Option],
    *,
    request_id: str | None = None,
    turn: int | None = None,
    instructions: str = INSTRUCTIONS,
) -> ContextPacket:
    """Fixed sections (team sheets, rules) form the cacheable prefix; per-turn sections
    and the option list come after it."""
    fixed = [instructions.strip()] + [render_section(k, v) for k, v in fixed_sections.items()]
    option_lines = "\n".join(f"{o.id}: {o.order}  [{o.note}]" for o in options)
    turn_parts = [render_section(k, v) for k, v in turn_sections.items()]
    turn_parts.append(f"## OPTIONS\n{option_lines}")
    return ContextPacket(
        fixed_text="\n\n".join(fixed),
        turn_text="\n\n".join(turn_parts),
        option_ids=tuple(o.id for o in options),
        request_id=request_id or new_request_id(turn),
        turn=turn,
    )
