"""Checkpoint ownership: refcounting, copy-on-write forking, and no silent eviction."""

import numpy as np
import pytest

from self_improving_diffusion.jax_backend.ownership import CheckpointOwnershipRegistry, OwnershipError
from self_improving_diffusion.jax_backend.state import TrajectoryState


def _state(sample_id: str, step: int, fill: float) -> TrajectoryState:
    return TrajectoryState(
        sample_id=sample_id,
        step=step,
        base_seed=0,
        model_revision="tiny-eps-mlp@v0",
        latents=np.full((1, 4), fill, dtype=np.float32),
    )


def test_put_registers_a_new_branch_with_refcount_one():
    registry = CheckpointOwnershipRegistry()
    physical_id = registry.put("root", _state("s1", 12, 1.0))
    assert registry.refcount(physical_id) == 1
    assert registry.num_live_physical_checkpoints == 1


def test_put_twice_for_the_same_branch_is_rejected():
    registry = CheckpointOwnershipRegistry()
    registry.put("root", _state("s1", 12, 1.0))
    with pytest.raises(OwnershipError):
        registry.put("root", _state("s1", 12, 1.0))


def test_fork_shares_physical_state_without_copying():
    registry = CheckpointOwnershipRegistry()
    physical_id = registry.put("root", _state("s1", 12, 1.0))
    child_physical_id = registry.fork("root", "child1")
    assert child_physical_id == physical_id
    assert registry.refcount(physical_id) == 2
    assert registry.get("root") is registry.get("child1")


def test_full_branch_tree_scenario_refcounts_and_reclaim():
    registry = CheckpointOwnershipRegistry()
    state_a = _state("s1", 12, 1.0)
    physical_a = registry.put("root", state_a)

    registry.fork("root", "child1")
    registry.fork("root", "child2")
    assert registry.refcount(physical_a) == 3

    registry.release("root")
    assert registry.refcount(physical_a) == 2
    with pytest.raises(OwnershipError):
        registry.reclaim(physical_a)

    state_b = _state("s1", 16, 2.0)
    physical_b = registry.diverge("child1", state_b)
    assert physical_b != physical_a
    assert registry.refcount(physical_a) == 1
    assert registry.refcount(physical_b) == 1

    registry.release("child2")
    assert registry.refcount(physical_a) == 0
    assert registry.is_reclaimable(physical_a)
    reclaimed = registry.reclaim(physical_a)
    assert np.array_equal(reclaimed.latents, state_a.latents)
    assert registry.num_live_physical_checkpoints == 1  # only physical_b remains

    registry.release("child1")
    assert registry.is_reclaimable(physical_b)
    registry.reclaim(physical_b)
    assert registry.num_live_physical_checkpoints == 0


def test_reclaim_unknown_physical_id_raises():
    registry = CheckpointOwnershipRegistry()
    with pytest.raises(OwnershipError):
        registry.reclaim("not-a-real-id")


def test_release_unknown_branch_raises():
    registry = CheckpointOwnershipRegistry()
    with pytest.raises(OwnershipError):
        registry.release("ghost-branch")


def test_double_release_raises():
    registry = CheckpointOwnershipRegistry()
    registry.put("root", _state("s1", 12, 1.0))
    registry.release("root")
    with pytest.raises(OwnershipError):
        registry.release("root")


def test_identical_content_from_independent_branches_deduplicates():
    registry = CheckpointOwnershipRegistry()
    physical_1 = registry.put("branch1", _state("s1", 12, 5.0))
    physical_2 = registry.put("branch2", _state("s1", 12, 5.0))
    assert physical_1 == physical_2
    assert registry.refcount(physical_1) == 2
    assert registry.num_live_physical_checkpoints == 1


def test_fork_from_unknown_parent_raises():
    registry = CheckpointOwnershipRegistry()
    with pytest.raises(OwnershipError):
        registry.fork("no-such-parent", "child")


def test_diverge_on_unknown_branch_raises():
    registry = CheckpointOwnershipRegistry()
    with pytest.raises(OwnershipError):
        registry.diverge("no-such-branch", _state("s1", 12, 1.0))
