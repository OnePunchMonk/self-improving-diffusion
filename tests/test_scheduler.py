import pytest

from self_improving_diffusion import GenerationSpec, ModelRef, ProgramBuilder
from self_improving_diffusion.scheduler import ReadyNode, StepBatchScheduler


def program(width: int = 1024):
    return ProgramBuilder(
        GenerationSpec("a red cube", ModelRef("base", "r17"), seed=1, width=width)
    ).sample().emit_trace().build()


def test_scheduler_batches_only_compatible_ready_nodes() -> None:
    scheduler = StepBatchScheduler()
    same_shape_a = ReadyNode("request_a", program(), 0)
    same_shape_b = ReadyNode("request_b", program(), 0)
    different_shape = ReadyNode("request_c", program(width=512), 0)
    scheduler.submit(same_shape_a)
    scheduler.submit(same_shape_b)
    scheduler.submit(different_shape)

    first = scheduler.next_batch(max_items=8)
    second = scheduler.next_batch(max_items=8)

    assert first == (same_shape_a, same_shape_b)
    assert second == (different_shape,)


def test_scheduler_rotates_between_compatibility_classes() -> None:
    scheduler = StepBatchScheduler()
    first_class_a = ReadyNode("request_a", program(), 0)
    first_class_b = ReadyNode("request_b", program(), 0)
    second_class = ReadyNode("request_c", program(width=512), 0)
    scheduler.submit(first_class_a)
    scheduler.submit(first_class_b)
    scheduler.submit(second_class)

    assert scheduler.next_batch(max_items=1) == (first_class_a,)
    assert scheduler.next_batch(max_items=1) == (second_class,)
    assert scheduler.next_batch(max_items=1) == (first_class_b,)


def test_scheduler_rejects_duplicate_ready_node() -> None:
    scheduler = StepBatchScheduler()
    ready = ReadyNode("request_a", program(), 0)
    scheduler.submit(ready)

    with pytest.raises(ValueError, match="only once"):
        scheduler.submit(ready)
