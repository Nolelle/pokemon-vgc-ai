"""Local spend meter: price table, cumulative cost, hard cap, worst-case pre-flight.

Spend is persisted to a small JSON file so the cap survives restarts. A corrupt file fails
closed (spent is set to the cap) rather than silently resetting to zero. One process is
assumed to own the file; the in-process lock does not protect against two bot processes.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    input: float
    cached_input: float
    cache_write: float
    output: float  # thinking tokens are billed as output


PRICE_TABLE: dict[str, Prices] = {
    "gpt-6-luna": Prices(input=0.10, cached_input=0.01, cache_write=0.125, output=0.50),
}


def cost_usd(
    model: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
) -> float:
    """Exact cost of a finished call. `input_tokens` includes cached and cache-write."""
    p = PRICE_TABLE[model]
    fresh = max(0, input_tokens - cached_tokens - cache_write_tokens)
    return (
        fresh * p.input
        + cached_tokens * p.cached_input
        + cache_write_tokens * p.cache_write
        + output_tokens * p.output
    ) / 1_000_000


def max_input_tokens(*texts: str) -> int:
    """Upper bound on input tokens: one token per UTF-8 byte of everything sent.

    Tokenizers never emit more than about one token per byte, so unlike a chars-per-token
    guess this cannot undercount. Pass every string sent: instructions, input and the
    JSON schema text.
    """
    return sum(len(t.encode("utf-8")) for t in texts)


@dataclass(frozen=True)
class Reservation:
    id: int
    amount_usd: float


class SpendMeter:
    def __init__(self, path: str | Path | None = None, cap_usd: float = 20.0,
                 model: str = "gpt-6-luna") -> None:
        self.path = Path(path) if path else None
        self.cap_usd = float(cap_usd)
        self.model = model
        self._lock = threading.Lock()
        self._next_id = 1
        self._reserved: dict[int, float] = {}
        self.spent_usd = 0.0
        self.calls = 0
        self._load()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            self.spent_usd = float(data["spent_usd"])
            self.calls = int(data.get("calls", 0))
            # Requests sent but never settled (crash mid-call) may have been billed:
            # count each at its reserved worst case.
            outstanding = data.get("reserved", {})
            self.spent_usd += sum(float(v) for v in outstanding.values())
            if outstanding:
                self._save()
        except (OSError, ValueError, KeyError, TypeError):
            self.spent_usd = self.cap_usd  # fail closed

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(
                    {
                        "spent_usd": self.spent_usd,
                        "calls": self.calls,
                        "reserved": {str(k): v for k, v in self._reserved.items()},
                    }
                ))
            os.replace(tmp, self.path)
        except OSError:
            pass  # in-memory meter still enforces the cap for this process

    def worst_case_usd(self, input_tokens_bound: int, max_output_tokens: int) -> float:
        p = PRICE_TABLE[self.model]
        # Input is billed at the dearest of fresh / cache-write, output at the full cap.
        return (
            input_tokens_bound * max(p.input, p.cache_write) + max_output_tokens * p.output
        ) / 1_000_000

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + sum(self._reserved.values())

    def reserve(self, input_tokens_bound: int, max_output_tokens: int) -> Reservation | None:
        """Hold the worst-case cost of a request, or return None if the cap would break."""
        worst = self.worst_case_usd(input_tokens_bound, max_output_tokens)
        with self._lock:
            if self.committed_usd + worst > self.cap_usd:
                return None
            res = Reservation(self._next_id, worst)
            self._next_id += 1
            self._reserved[res.id] = worst
            self._save()  # persisted BEFORE the request is sent
            return res

    def settle(self, res: Reservation, actual_usd: float | None = None) -> None:
        """Release the hold and book the actual cost (default: the worst case)."""
        with self._lock:
            held = self._reserved.pop(res.id, None)
            if held is None:
                return
            self.spent_usd += held if actual_usd is None else actual_usd
            self.calls += 1
            self._save()

    def cancel(self, res: Reservation) -> None:
        """Release the hold with no charge (request never reached the provider)."""
        with self._lock:
            if self._reserved.pop(res.id, None) is not None:
                self._save()
