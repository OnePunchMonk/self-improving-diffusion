"""Exact, request-scoped checkpoint retention for resumable trajectories."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import RLock

from .executor import Checkpoint


class CheckpointCapacityError(RuntimeError):
    """Raised instead of silently evicting a live resumable trajectory."""


class ExactCheckpointStore:
    """An in-memory v0 store with explicit TTL and request isolation.

    A checkpoint can only be recovered with the original request ID, sample ID,
    and denoising step. There is intentionally no approximate or cross-request
    lookup API. A production implementation can replace the backing storage
    while preserving this safety contract.
    """

    def __init__(
        self,
        max_entries: int = 1_024,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._entries: dict[tuple[str, str, int], tuple[Checkpoint, datetime]] = {}
        self._lock = RLock()

    def put(self, request_id: str, checkpoint: Checkpoint, ttl: timedelta) -> None:
        if not request_id:
            raise ValueError("request_id must be non-empty")
        if ttl <= timedelta(0):
            raise ValueError("checkpoint TTL must be positive")
        key = (request_id, checkpoint.sample_id, checkpoint.step)
        with self._lock:
            self._purge_expired_locked()
            if key not in self._entries and len(self._entries) >= self._max_entries:
                raise CheckpointCapacityError("checkpoint store capacity is exhausted")
            self._entries[key] = (checkpoint, self._clock() + ttl)

    def get(self, request_id: str, sample_id: str, step: int) -> Checkpoint | None:
        key = (request_id, sample_id, step)
        with self._lock:
            self._purge_expired_locked()
            entry = self._entries.get(key)
            return entry[0] if entry else None

    def count(self) -> int:
        with self._lock:
            self._purge_expired_locked()
            return len(self._entries)

    def _purge_expired_locked(self) -> None:
        now = self._clock()
        expired = [key for key, (_, expires_at) in self._entries.items() if expires_at <= now]
        for key in expired:
            del self._entries[key]
