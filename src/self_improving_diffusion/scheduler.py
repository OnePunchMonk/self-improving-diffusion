"""Compatibility-key batching for dependency-ready program nodes.

This is intentionally only an admission scheduler. The program executor (or a
future DAG runtime) decides when a node is ready; this queue only batches work
whose immutable denoiser configuration is identical.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from .program import GenerationProgram, NodeKind


@dataclass(frozen=True)
class ReadyNode:
    request_id: str
    program: GenerationProgram
    node_index: int

    @property
    def node_kind(self) -> NodeKind:
        return self.program.nodes[self.node_index].kind

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        """Only work with the same backend shape and operation can co-batch."""

        return (self.node_kind.value, *self.program.spec.compatibility_key())


class StepBatchScheduler:
    """FIFO across compatibility classes and FIFO within each class."""

    def __init__(self) -> None:
        self._queues: dict[tuple[object, ...], Deque[ReadyNode]] = {}
        self._arrival_order: Deque[tuple[object, ...]] = deque()
        self._queued_request_nodes: set[tuple[str, int]] = set()

    def submit(self, ready: ReadyNode) -> None:
        if not ready.request_id:
            raise ValueError("request_id must be non-empty")
        if not 0 <= ready.node_index < len(ready.program.nodes):
            raise ValueError("node_index is outside the program")
        identity = (ready.request_id, ready.node_index)
        if identity in self._queued_request_nodes:
            raise ValueError("a ready request node may be queued only once")

        key = ready.compatibility_key
        queue = self._queues.setdefault(key, deque())
        if not queue:
            self._arrival_order.append(key)
        queue.append(ready)
        self._queued_request_nodes.add(identity)

    def next_batch(self, max_items: int) -> tuple[ReadyNode, ...]:
        if max_items < 1:
            raise ValueError("max_items must be positive")
        if not self._arrival_order:
            return ()

        key = self._arrival_order.popleft()
        queue = self._queues[key]
        batch = tuple(queue.popleft() for _ in range(min(max_items, len(queue))))
        for ready in batch:
            self._queued_request_nodes.remove((ready.request_id, ready.node_index))
        if queue:
            self._arrival_order.append(key)
        else:
            del self._queues[key]
        return batch

    @property
    def pending(self) -> int:
        return len(self._queued_request_nodes)
