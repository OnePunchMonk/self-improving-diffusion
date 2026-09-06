from datetime import datetime, timedelta, timezone

import pytest

from self_improving_diffusion import CheckpointCapacityError, ExactCheckpointStore
from self_improving_diffusion.executor import Checkpoint


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 7, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now


def checkpoint(sample_id: str = "sample_a", step: int = 12) -> Checkpoint:
    return Checkpoint(sample_id=sample_id, step=step, state_digest="state_digest")


def test_checkpoint_is_retrievable_only_by_its_original_request() -> None:
    store = ExactCheckpointStore()
    store.put("request_a", checkpoint(), timedelta(minutes=5))

    assert store.get("request_a", "sample_a", 12) == checkpoint()
    assert store.get("request_b", "sample_a", 12) is None


def test_checkpoint_expires_before_capacity_is_checked() -> None:
    clock = Clock()
    store = ExactCheckpointStore(max_entries=1, clock=clock)
    store.put("request_a", checkpoint(), timedelta(seconds=1))
    clock.now += timedelta(seconds=1)

    store.put("request_b", checkpoint("sample_b"), timedelta(minutes=1))
    assert store.count() == 1
    assert store.get("request_a", "sample_a", 12) is None


def test_checkpoint_store_never_silently_evicts_live_state() -> None:
    store = ExactCheckpointStore(max_entries=1)
    store.put("request_a", checkpoint(), timedelta(minutes=5))

    with pytest.raises(CheckpointCapacityError, match="capacity"):
        store.put("request_b", checkpoint("sample_b"), timedelta(minutes=5))
