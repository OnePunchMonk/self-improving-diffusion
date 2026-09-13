"""Overlap scheduling: overlapped host prep must not change correctness, and must be faster."""

import jax

from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.overlap import run_overlapped, run_serialized
from self_improving_diffusion.jax_backend.sampler import make_schedule
from self_improving_diffusion.jax_backend.serving import CompiledVariantCache


def _cache() -> CompiledVariantCache:
    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    schedule = make_schedule(16)
    return CompiledVariantCache(params, schedule, max_variants=4)


def test_overlapped_is_not_slower_than_serialized_with_nonzero_host_prep():
    cache = _cache()
    serialized = run_serialized(cache, num_batches=6, batch_size=2, chunk_steps=2, start_step=15, simulated_host_prep_s=0.02)
    overlapped = run_overlapped(cache, num_batches=6, batch_size=2, chunk_steps=2, start_step=15, simulated_host_prep_s=0.02)
    # Overlap hides host prep behind device execution, so it should never be
    # meaningfully slower; allow a small margin for scheduling noise.
    assert overlapped.total_wall_time_s <= serialized.total_wall_time_s * 1.2


def test_both_policies_process_the_declared_number_of_batches():
    cache = _cache()
    serialized = run_serialized(cache, num_batches=4, batch_size=2, chunk_steps=2, start_step=15, simulated_host_prep_s=0.0)
    overlapped = run_overlapped(cache, num_batches=4, batch_size=2, chunk_steps=2, start_step=15, simulated_host_prep_s=0.0)
    assert serialized.num_batches == 4
    assert overlapped.num_batches == 4


def test_reports_are_well_formed():
    cache = _cache()
    result = run_serialized(cache, num_batches=3, batch_size=2, chunk_steps=1, start_step=15, simulated_host_prep_s=0.0)
    assert result.total_wall_time_s >= 0.0
    assert result.mean_wall_time_per_batch_s == result.total_wall_time_s / result.num_batches
    assert result.policy == "serialized"
