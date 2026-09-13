"""JX-05: compiled-variant cache bounds, admission/queueing, and scheduling comparison."""

import jax
import pytest

from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import make_schedule
from self_improving_diffusion.jax_backend.serving import (
    CompiledVariantCache,
    CompiledVariantCacheError,
    generate_poisson_arrivals,
    simulate_chunk_scheduling,
    simulate_static_batching,
)


def _cache(max_variants: int = 4) -> CompiledVariantCache:
    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    schedule = make_schedule(16)
    return CompiledVariantCache(params, schedule, max_variants=max_variants)


def test_cache_hit_on_repeated_shape():
    cache = _cache()
    fn_a = cache.get_or_compile(batch_size=4, chunk_steps=2)
    fn_b = cache.get_or_compile(batch_size=4, chunk_steps=2)
    assert fn_a is fn_b
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_raises_clearly_instead_of_unbounded_growth():
    cache = _cache(max_variants=2)
    cache.get_or_compile(batch_size=2, chunk_steps=1)
    cache.get_or_compile(batch_size=4, chunk_steps=1)
    with pytest.raises(CompiledVariantCacheError):
        cache.get_or_compile(batch_size=8, chunk_steps=1)


def test_poisson_arrivals_are_reproducible_for_a_fixed_seed():
    a = generate_poisson_arrivals(rate_per_s=10.0, num_requests=20, steps=8, seed=1)
    b = generate_poisson_arrivals(rate_per_s=10.0, num_requests=20, steps=8, seed=1)
    assert [r.arrival_time_s for r in a] == [r.arrival_time_s for r in b]
    assert all(a[i].arrival_time_s < a[i + 1].arrival_time_s for i in range(len(a) - 1))


def test_static_batching_preserves_every_request_identity():
    cache = _cache()
    requests = generate_poisson_arrivals(rate_per_s=20.0, num_requests=10, steps=8, seed=2)
    report = simulate_static_batching(cache, requests, batch_size=4, total_steps=8)
    completed_ids = {o.request_id for o in report.outcomes}
    assert completed_ids == {r.request_id for r in requests}


def test_chunk_scheduling_preserves_every_request_identity():
    cache = _cache()
    requests = generate_poisson_arrivals(rate_per_s=20.0, num_requests=10, steps=8, seed=3)
    report = simulate_chunk_scheduling(cache, requests, max_active_slots=4, chunk_steps=2, total_steps=8)
    completed_ids = {o.request_id for o in report.outcomes}
    assert completed_ids == {r.request_id for r in requests}


def test_chunk_scheduling_never_completes_a_request_before_it_arrives():
    cache = _cache()
    requests = generate_poisson_arrivals(rate_per_s=15.0, num_requests=8, steps=8, seed=4)
    report = simulate_chunk_scheduling(cache, requests, max_active_slots=3, chunk_steps=2, total_steps=8)
    for outcome in report.outcomes:
        assert outcome.admitted_time_s >= outcome.arrival_time_s
        assert outcome.completed_time_s >= outcome.admitted_time_s


def test_report_percentiles_and_goodput_are_well_formed():
    cache = _cache()
    requests = generate_poisson_arrivals(rate_per_s=20.0, num_requests=10, steps=8, seed=5)
    report = simulate_static_batching(cache, requests, batch_size=4, total_steps=8)
    percentiles = report.percentiles()
    assert percentiles["p50_s"] <= percentiles["p95_s"] <= percentiles["p99_s"]
    assert 0.0 <= report.goodput_fraction(sla_s=10.0) <= 1.0
    assert report.throughput_rps() > 0
