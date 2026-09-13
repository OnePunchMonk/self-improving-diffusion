"""JX-04: distributed correctness before hybrid optimization.

Starts where the charter says to start: request/data parallelism only. Each
device independently runs one full sample; there is no intra-request
sharding here because the tiny JX-01 model is far too small to justify it
(the charter explicitly gates sequence/tensor sharding behind "model
size/latency warrants it").

The correctness property this module exists to prove: **device count and
device assignment must never change the result.** A sample's RNG stream is
derived from ``fold_in(PRNGKey(seed), (request_id, sample_index))``, which
depends only on logical identity, never on which physical device happened to
run it or how many devices were available. Running the same batch of samples
on 1 device or N devices must produce bit-identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from .model import ModelParams, TinyEpsilonModel
from .sampler import NoiseSchedule, ddpm_step


class DeviceCountError(ValueError):
    """Raised instead of silently under-utilizing or oversubscribing devices."""


def _global_sample_key(seed: int, request_id: str, sample_index: int) -> jax.Array:
    """A key derived only from logical identity -- never from rank or placement."""

    base = jax.random.fold_in(jax.random.PRNGKey(seed), hash(request_id) & 0xFFFFFFFF)
    return jax.random.fold_in(base, sample_index)


def _run_one_sample(params: ModelParams, schedule: NoiseSchedule, key: jax.Array) -> jnp.ndarray:
    init_latents = jax.random.normal(key, (1, TinyEpsilonModel.latent_dim))

    def body(carry, step):
        return ddpm_step(params, schedule, carry, step, key), None

    steps = jnp.arange(schedule.betas.shape[0] - 1, -1, -1)
    final, _ = jax.lax.scan(body, init_latents, steps)
    return final


@dataclass(frozen=True)
class DistributedRunReport:
    num_devices_available: int
    num_devices_used: int
    num_samples: int
    wall_time_s: float
    sample_ids: tuple[str, ...]
    matches_single_device: bool


def run_data_parallel_samples(
    params: ModelParams,
    schedule: NoiseSchedule,
    seed: int,
    request_id: str,
    num_samples: int,
) -> tuple[np.ndarray, DistributedRunReport]:
    """Run ``num_samples`` independent samples, one per available device (round-robin).

    Uses ``jax.pmap`` over as many devices as are available, batching samples
    into round-robin groups when ``num_samples`` exceeds the device count.
    Every sample's RNG key depends only on ``(seed, request_id, sample_index)``
    -- identical regardless of how many devices exist or which one executes it.
    """

    if num_samples < 1:
        raise DeviceCountError("num_samples must be positive")

    devices = jax.local_devices()
    num_devices = len(devices)
    keys = jnp.stack([_global_sample_key(seed, request_id, i) for i in range(num_samples)])

    mapped = jax.pmap(lambda k: _run_one_sample(params, schedule, k), in_axes=0)

    start = time.perf_counter()
    outputs = []
    for batch_start in range(0, num_samples, num_devices):
        batch_keys = keys[batch_start : batch_start + num_devices]
        pad = num_devices - batch_keys.shape[0]
        if pad:
            batch_keys = jnp.concatenate([batch_keys, jnp.tile(batch_keys[:1], (pad, 1))], axis=0)
        batch_out = mapped(batch_keys)
        jax.block_until_ready(batch_out)
        outputs.append(np.asarray(batch_out)[: num_devices - pad if pad else num_devices])
    wall_time_s = time.perf_counter() - start

    all_outputs = np.concatenate(outputs, axis=0)[:num_samples]

    # The independent, sequential ground truth: same keys, one device, no pmap.
    sequential = np.concatenate(
        [np.asarray(_run_one_sample(params, schedule, keys[i])) for i in range(num_samples)],
        axis=0,
    )
    matches = bool(np.array_equal(all_outputs.reshape(num_samples, -1), sequential))

    sample_ids = tuple(f"sample_{request_id}_{i}" for i in range(num_samples))
    report = DistributedRunReport(
        num_devices_available=num_devices,
        num_devices_used=min(num_devices, num_samples),
        num_samples=num_samples,
        wall_time_s=wall_time_s,
        sample_ids=sample_ids,
        matches_single_device=matches,
    )
    return all_outputs, report


def measure_collective_latency(message_sizes: tuple[int, ...] = (1_024, 65_536, 1_048_576)) -> list[dict]:
    """Time an all-reduce (psum) over the local device mesh, by message size.

    On a single-host CPU mesh this measures the CPU collective-emulation path,
    not real network fabric -- it exists to establish the *methodology*
    (systematic sweep over message size, synchronized before timing) that
    carries over unchanged when run on a real multi-host GPU/TPU mesh.
    """

    devices = jax.local_devices()
    num_devices = len(devices)
    if num_devices < 2:
        raise DeviceCountError(
            "collective latency measurement requires >= 2 devices; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=N before importing jax"
        )

    results = []
    for size in message_sizes:
        data = jnp.ones((num_devices, size), dtype=jnp.float32)
        psum = jax.pmap(lambda x: jax.lax.psum(x, axis_name="devices"), axis_name="devices")
        jax.block_until_ready(psum(data))  # warm

        reps = 10
        start = time.perf_counter()
        for _ in range(reps):
            out = psum(data)
        jax.block_until_ready(out)
        elapsed_s = (time.perf_counter() - start) / reps

        bytes_per_device = size * 4
        results.append(
            {
                "message_bytes_per_device": bytes_per_device,
                "num_devices": num_devices,
                "elapsed_s": elapsed_s,
                "achieved_gbps": (bytes_per_device * num_devices) / elapsed_s / 1e9,
            }
        )
    return results
