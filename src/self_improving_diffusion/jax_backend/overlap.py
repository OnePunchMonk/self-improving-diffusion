"""Overlap scheduling: prepare the next batch's inputs while the current one executes.

From the serving-ideas backlog (issue #5), build-order priority 4: borrow
SGLang's overlap-scheduler principle (prepare metadata/inputs for the next
batch while the current batch runs on-device) without pretending this tiny
model's compute is where a real win would show up.

**Honesty about what's synthetic here**: `TinyEpsilonModel`'s actual device
compute is sub-millisecond (JX-02's own measurement), so there is no real
CPU-side preprocessing step (a text encoder, tokenization, image
preprocessing) in this repo yet whose cost this can measure. `_simulate_host_prep`
is an explicit, labeled `time.sleep` stand-in for that future work -- this
module demonstrates the *mechanism* (JAX's asynchronous dispatch lets host
prep genuinely overlap with device execution) and measures it correctly
(synchronizing only where the comparison requires it), not a claim about
this specific model's real preprocessing cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import jax

from .serving import CompiledVariantCache


def _simulate_host_prep(seconds: float) -> None:
    """Stand-in for real host-side preprocessing (e.g. a text encoder call)."""

    time.sleep(seconds)


def _prepare_batch(batch_size: int, latent_dim: int, batch_index: int):
    latents = jax.random.normal(jax.random.PRNGKey(batch_index), (batch_size, latent_dim))
    active_mask = jax.numpy.ones((batch_size,), dtype=bool)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(batch_index + 1), i))(
        jax.numpy.arange(batch_size)
    )
    return latents, active_mask, keys


@dataclass
class OverlapComparison:
    policy: str
    num_batches: int
    simulated_host_prep_s: float
    total_wall_time_s: float
    mean_wall_time_per_batch_s: float


def run_serialized(
    cache: CompiledVariantCache,
    num_batches: int,
    batch_size: int,
    chunk_steps: int,
    start_step: int,
    simulated_host_prep_s: float,
) -> OverlapComparison:
    """Prepare, dispatch, and block on every batch before starting the next one."""

    fn = cache.get_or_compile(batch_size, chunk_steps)
    latent_dim = 64  # matches TinyEpsilonModel.latent_dim; avoided importing to keep this module standalone

    start = time.perf_counter()
    for i in range(num_batches):
        latents, active_mask, keys = _prepare_batch(batch_size, latent_dim, i)
        _simulate_host_prep(simulated_host_prep_s)
        out = fn(latents, active_mask, start_step, keys)
        jax.block_until_ready(out)
    total_s = time.perf_counter() - start

    return OverlapComparison("serialized", num_batches, simulated_host_prep_s, total_s, total_s / num_batches)


def run_overlapped(
    cache: CompiledVariantCache,
    num_batches: int,
    batch_size: int,
    chunk_steps: int,
    start_step: int,
    simulated_host_prep_s: float,
) -> OverlapComparison:
    """Dispatch batch i, then prepare batch i+1's inputs while it runs on-device.

    JAX's dispatch is asynchronous: `fn(...)` returns as soon as the
    computation is enqueued, before it has actually run. Not calling
    `block_until_ready` immediately is what lets the following
    `_prepare_batch`/`_simulate_host_prep` genuinely execute concurrently
    with device work, instead of the host sitting idle waiting for it.
    """

    fn = cache.get_or_compile(batch_size, chunk_steps)
    latent_dim = 64  # matches TinyEpsilonModel.latent_dim; avoided importing to keep this module standalone

    start = time.perf_counter()
    latents, active_mask, keys = _prepare_batch(batch_size, latent_dim, 0)
    pending = fn(latents, active_mask, start_step, keys)  # dispatched, not blocked

    for i in range(1, num_batches):
        latents, active_mask, keys = _prepare_batch(batch_size, latent_dim, i)
        _simulate_host_prep(simulated_host_prep_s)  # overlaps with `pending`'s device execution
        jax.block_until_ready(pending)  # only now do we need last batch's result
        pending = fn(latents, active_mask, start_step, keys)

    jax.block_until_ready(pending)
    total_s = time.perf_counter() - start

    return OverlapComparison("overlapped", num_batches, simulated_host_prep_s, total_s, total_s / num_batches)
