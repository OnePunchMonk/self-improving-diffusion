"""JX-02: the frozen quality/compute protocol must be deterministic and complete."""

from self_improving_diffusion.eval.quality_compute import (
    FROZEN_SEEDS,
    FROZEN_STEP_BUDGETS,
    QualityComputeProtocol,
)


def test_run_covers_full_seed_by_step_grid():
    protocol = QualityComputeProtocol(seeds=(1, 2), step_budgets=(4, 8))
    records = protocol.run()
    assert len(records) == 4
    seen = {(r.seed, r.steps) for r in records}
    assert seen == {(1, 4), (1, 8), (2, 4), (2, 8)}


def test_same_seed_and_steps_reproduce_same_quality():
    protocol = QualityComputeProtocol(seeds=(3,), step_budgets=(6,))
    first = protocol.run()[0]
    second = protocol.run()[0]
    assert first.quality == second.quality
    assert first.estimated_denoising_steps == second.estimated_denoising_steps


def test_frontier_summarizes_each_step_budget():
    protocol = QualityComputeProtocol(seeds=(1, 2, 3), step_budgets=(4, 8))
    records = protocol.run()
    frontier = protocol.frontier(records)
    assert [point["steps"] for point in frontier] == [4, 8]
    for point in frontier:
        assert point["n"] == 3


def test_default_protocol_constants_are_declared_and_non_empty():
    assert len(FROZEN_SEEDS) > 0
    assert len(FROZEN_STEP_BUDGETS) > 0
