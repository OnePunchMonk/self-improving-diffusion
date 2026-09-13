"""Checkpoint ownership: logical/physical handles, refcounts, copy-on-write.

Borrows the discipline LLM serving engines use for KV-cache/session state --
logical handle vs. physical storage, explicit reference counts, a
reclaimable/live distinction, copy-on-write forking -- and applies it to
`TrajectoryState` (JX-01's real, content-addressed checkpoint data) instead
of building a custom paged allocator up front. A diffusion latent is
typically replaced wholesale each step rather than growing like an
autoregressive KV cache, so PagedAttention-style paging is not a drop-in
design here; this registry is the ownership layer that would sit underneath
paging if profiling ever justified adding it (JX-03/JX-05 haven't shown
that yet for a model this small).

The one rule this module exists to enforce: **a physical checkpoint is
never reclaimed while any branch still holds a reference to it.**
"""

from __future__ import annotations

from dataclasses import dataclass

from .state import TrajectoryState


class OwnershipError(RuntimeError):
    """Raised instead of silently evicting live state or double-releasing a handle."""


@dataclass
class _PhysicalEntry:
    state: TrajectoryState
    refcount: int


class CheckpointOwnershipRegistry:
    """Content-addressed checkpoint storage with explicit branch ownership.

    Physical identity is `TrajectoryState.state_digest()` -- two branches
    whose latents happen to be bit-identical share one physical entry
    automatically, for free, because the address *is* the content hash.
    Logical identity is a caller-chosen `branch_id` (e.g. "req1:sample0" or
    "req1:sample0:refine-a"); this registry never inspects it beyond using
    it as a dictionary key.
    """

    def __init__(self) -> None:
        self._physical: dict[str, _PhysicalEntry] = {}
        self._branch_to_physical: dict[str, str] = {}

    def put(self, branch_id: str, state: TrajectoryState) -> str:
        """Register a new logical branch, owning (a reference to) `state`."""

        if branch_id in self._branch_to_physical:
            raise OwnershipError(f"branch_id {branch_id!r} already owns a checkpoint; use diverge() to replace it")

        physical_id = state.state_digest()
        entry = self._physical.get(physical_id)
        if entry is None:
            self._physical[physical_id] = _PhysicalEntry(state=state, refcount=1)
        else:
            entry.refcount += 1
        self._branch_to_physical[branch_id] = physical_id
        return physical_id

    def fork(self, parent_branch_id: str, child_branch_id: str) -> str:
        """Copy-on-write: child shares the parent's physical checkpoint, no copy yet."""

        if parent_branch_id not in self._branch_to_physical:
            raise OwnershipError(f"unknown parent branch_id {parent_branch_id!r}")
        if child_branch_id in self._branch_to_physical:
            raise OwnershipError(f"branch_id {child_branch_id!r} already owns a checkpoint")

        physical_id = self._branch_to_physical[parent_branch_id]
        self._physical[physical_id].refcount += 1
        self._branch_to_physical[child_branch_id] = physical_id
        return physical_id

    def diverge(self, branch_id: str, new_state: TrajectoryState) -> str:
        """The actual copy-on-write moment: this branch now owns different content.

        Releases the branch's reference to whatever it previously pointed at
        (which may still be shared by other branches -- their refcounts are
        untouched) and registers the new state as this branch's checkpoint.
        """

        if branch_id not in self._branch_to_physical:
            raise OwnershipError(f"unknown branch_id {branch_id!r}; use put() to register it first")

        old_physical_id = self._branch_to_physical[branch_id]
        self._decref(old_physical_id)

        new_physical_id = new_state.state_digest()
        entry = self._physical.get(new_physical_id)
        if entry is None:
            self._physical[new_physical_id] = _PhysicalEntry(state=new_state, refcount=1)
        else:
            entry.refcount += 1
        self._branch_to_physical[branch_id] = new_physical_id
        return new_physical_id

    def release(self, branch_id: str) -> None:
        """Drop a branch's ownership. Does not reclaim storage by itself."""

        if branch_id not in self._branch_to_physical:
            raise OwnershipError(f"unknown branch_id {branch_id!r}, or it was already released")
        physical_id = self._branch_to_physical.pop(branch_id)
        self._decref(physical_id)

    def _decref(self, physical_id: str) -> None:
        entry = self._physical[physical_id]
        entry.refcount -= 1
        if entry.refcount < 0:  # pragma: no cover - defends against a registry bug, not reachable via the public API
            raise OwnershipError(f"refcount underflow for physical checkpoint {physical_id!r}")

    def refcount(self, physical_id: str) -> int:
        return self._physical[physical_id].refcount

    def is_reclaimable(self, physical_id: str) -> bool:
        return physical_id in self._physical and self._physical[physical_id].refcount == 0

    def reclaim(self, physical_id: str) -> TrajectoryState:
        """Free a physical checkpoint's storage. Raises if any branch still holds it live."""

        if physical_id not in self._physical:
            raise OwnershipError(f"unknown physical_id {physical_id!r}, or it was already reclaimed")
        entry = self._physical[physical_id]
        if entry.refcount > 0:
            raise OwnershipError(
                f"refusing to reclaim physical checkpoint {physical_id!r}: "
                f"{entry.refcount} branch(es) still reference it"
            )
        del self._physical[physical_id]
        return entry.state

    def get(self, branch_id: str) -> TrajectoryState:
        physical_id = self._branch_to_physical[branch_id]
        return self._physical[physical_id].state

    @property
    def num_live_physical_checkpoints(self) -> int:
        return len(self._physical)
