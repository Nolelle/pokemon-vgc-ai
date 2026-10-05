"""Local spend meter: price table, cumulative cost, hard cap, worst-case pre-flight.

Spend is persisted to a small JSON file so the cap survives restarts. A corrupt file fails
closed (spent is set to the cap) rather than silently resetting to zero. One process is
assumed to own the file; a file lock plus re-read-before-write keeps concurrent meters and
processes from overwriting each other's tallies.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows: thread lock only
    fcntl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Prices:
    """USD per 1M tokens."""

    input: float
    cached_input: float
    cache_write: float
    output: float  # thinking tokens are billed as output


PRICE_TABLE: dict[str, Prices] = {
    "gpt-6-luna": Prices(input=0.10, cached_input=0.01, cache_write=0.125, output=0.50),
    # OpenAI pricing page, standard tier, short context (checked 2026-10-03).
    "gpt-6.1-sol": Prices(input=2.00, cached_input=0.10, cache_write=2.50, output=10.00),
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
    """Cap-enforcing spend tally, safe across threads, meters and processes.

    The JSON file is the source of truth. Every reserve/settle/cancel takes the in-process
    lock plus an exclusive file lock, re-reads the file, applies its change and writes it
    back, so two meters (or two bot processes) on one file merge instead of overwriting.
    """

    def __init__(self, path: str | Path | None = None, cap_usd: float = 20.0,
                 model: str = "gpt-6-luna") -> None:
        self.path = Path(path) if path else None
        self.cap_usd = float(cap_usd)
        self.model = model
        self._lock = threading.Lock()
        self._uid = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._next_id = 1
        self._reserved: dict[int, float] = {}  # this meter's own holds
        self._foreign: dict[str, float] = {}  # holds other live meters/processes persisted
        self.spent_usd = 0.0
        self.calls = 0
        self._load()

    def _key(self, res_id: int) -> str:
        return f"{self._uid}:{res_id}"

    def _read_disk(self) -> dict | None:
        """The file's contents, `{}` if absent, or None if unreadable (fail closed)."""
        if self.path is None or not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            float(data["spent_usd"])
            return data
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _refresh(self) -> None:
        """Merge the on-disk state into memory. Call with the locks held."""
        if self.path is None:
            return
        data = self._read_disk()
        if data is None:
            self.spent_usd = self.cap_usd  # fail closed
            return
        if not data:
            return
        self.spent_usd = float(data["spent_usd"])
        self.calls = int(data.get("calls", 0))
        own = {self._key(i) for i in self._reserved}
        try:
            self._foreign = {
                str(k): float(v) for k, v in data.get("reserved", {}).items() if k not in own
            }
        except (ValueError, TypeError, AttributeError):
            self.spent_usd = self.cap_usd

    @contextmanager
    def _txn(self):
        """Exclusive section: thread lock + cross-process file lock + fresh disk state."""
        with self._lock:
            lock_file = None
            if self.path is not None and fcntl is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    lock_file = open(self.path.with_suffix(self.path.suffix + ".lock"), "a")
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                except OSError:
                    lock_file = None
            try:
                self._refresh()
                yield
            finally:
                if lock_file is not None:
                    try:
                        fcntl.flock(lock_file, fcntl.LOCK_UN)
                    finally:
                        lock_file.close()

    def _load(self) -> None:
        with self._txn():
            if self.path is None or not self.path.exists() or not self._foreign:
                return
            # Holds left by an earlier run (crash mid-call) may have been billed: count
            # each at its reserved worst case.
            self.spent_usd += sum(self._foreign.values())
            self._foreign = {}
            self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + f".{self._uid}.tmp")
            reserved = dict(self._foreign)
            reserved.update({self._key(k): v for k, v in self._reserved.items()})
            tmp.write_text(json.dumps(
                {"spent_usd": self.spent_usd, "calls": self.calls, "reserved": reserved}
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
        return self.spent_usd + sum(self._reserved.values()) + sum(self._foreign.values())

    def reserve(self, input_tokens_bound: int, max_output_tokens: int) -> Reservation | None:
        """Hold the worst-case cost of a request, or return None if the cap would break."""
        worst = self.worst_case_usd(input_tokens_bound, max_output_tokens)
        with self._txn():
            if self.committed_usd + worst > self.cap_usd:
                return None
            res = Reservation(self._next_id, worst)
            self._next_id += 1
            self._reserved[res.id] = worst
            self._save()  # persisted BEFORE the request is sent
            return res

    def settle(self, res: Reservation, actual_usd: float | None = None) -> None:
        """Release the hold and book the actual cost (default: the worst case)."""
        with self._txn():
            held = self._reserved.pop(res.id, None)
            if held is None:
                return
            self.spent_usd += held if actual_usd is None else actual_usd
            self.calls += 1
            self._save()

    def cancel(self, res: Reservation) -> None:
        """Release the hold with no charge (request never reached the provider)."""
        with self._txn():
            if self._reserved.pop(res.id, None) is not None:
                self._save()


_shared_lock = threading.Lock()
_shared: dict[str, SpendMeter] = {}


def shared_meter(path: str | Path | None, cap_usd: float, model: str = "gpt-6-luna") -> SpendMeter:
    """One process-wide meter per spend file, enforcing the SMALLEST cap ever requested.

    Configs that share a spend file but differ in cap must not keep separate tallies (each
    would see only its own spend and together overshoot). `path=None` (in-memory) gets a
    private meter.
    """
    if path is None:
        return SpendMeter(None, cap_usd, model)
    key = str(Path(path).expanduser().resolve())
    with _shared_lock:
        meter = _shared.get(key)
        if meter is None:
            meter = _shared[key] = SpendMeter(key, cap_usd, model)
        else:
            meter.cap_usd = min(meter.cap_usd, float(cap_usd))
        return meter


def reset_shared_meters() -> None:
    with _shared_lock:
        _shared.clear()
