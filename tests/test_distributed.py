"""JX-04: distributed correctness -- device count/placement must not change results.

These tests need >= 2 simulated CPU devices, which JAX can only be told about
before it initializes its backend. Run with:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest tests/test_distributed.py

pytest.ini's addopts don't set this, so these tests skip themselves (rather
than fail) when only one device is visible, e.g. under the default `pytest`
invocation used for the rest of the suite.
"""

import jax
import numpy as np
import pytest

from self_improving_diffusion.jax_backend.distributed import (
    DeviceCountError,
    measure_collective_latency,
    run_data_parallel_samples,
)
from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import make_schedule

requires_multi_device = pytest.mark.skipif(
    len(jax.local_devices()) < 2,
    reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=N (N>=2) set before jax import",
)


def _params_and_schedule(steps: int = 8):
    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    return params, make_schedule(steps)


@requires_multi_device
def test_distributed_result_matches_sequential_ground_truth():
    params, schedule = _params_and_schedule()
    outputs, report = run_data_parallel_samples(params, schedule, seed=0, request_id="req-1", num_samples=6)
    assert report.matches_single_device
    assert outputs.shape == (6, 1, TinyEpsilonModel.latent_dim)


@requires_multi_device
def test_sample_count_not_a_multiple_of_device_count_still_matches():
    params, schedule = _params_and_schedule()
    num_devices = len(jax.local_devices())
    outputs, report = run_data_parallel_samples(
        params, schedule, seed=1, request_id="req-2", num_samples=num_devices + 1
    )
    assert report.matches_single_device
    assert report.num_samples == num_devices + 1


@requires_multi_device
def test_no_duplicate_rng_streams_across_samples():
    params, schedule = _params_and_schedule()
    outputs, _ = run_data_parallel_samples(params, schedule, seed=2, request_id="req-3", num_samples=4)
    flattened = outputs.reshape(4, -1)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(flattened[i], flattened[j])


@requires_multi_device
def test_same_request_id_and_seed_is_reproducible_regardless_of_rerun():
    params, schedule = _params_and_schedule()
    first, _ = run_data_parallel_samples(params, schedule, seed=3, request_id="req-4", num_samples=3)
    second, _ = run_data_parallel_samples(params, schedule, seed=3, request_id="req-4", num_samples=3)
    assert np.array_equal(first, second)


@requires_multi_device
def test_different_request_id_changes_stream_even_with_same_seed():
    params, schedule = _params_and_schedule()
    a, _ = run_data_parallel_samples(params, schedule, seed=4, request_id="req-a", num_samples=2)
    b, _ = run_data_parallel_samples(params, schedule, seed=4, request_id="req-b", num_samples=2)
    assert not np.array_equal(a, b)


def test_zero_or_negative_samples_rejected_clearly():
    params, schedule = _params_and_schedule()
    with pytest.raises(DeviceCountError):
        run_data_parallel_samples(params, schedule, seed=0, request_id="req", num_samples=0)


@requires_multi_device
def test_collective_latency_swept_by_message_size():
    results = measure_collective_latency(message_sizes=(1_024, 8_192))
    assert len(results) == 2
    assert results[0]["message_bytes_per_device"] == 4_096
    for r in results:
        assert r["elapsed_s"] >= 0.0
        assert r["num_devices"] == len(jax.local_devices())


def test_collective_latency_requires_multiple_devices_when_run_alone():
    if len(jax.local_devices()) >= 2:
        pytest.skip("this process has multiple devices; the single-device error path is untestable here")
    with pytest.raises(DeviceCountError):
        measure_collective_latency()
