"""JX-05: dynamic distributed serving -- compiled-variant cache, admission, chunk scheduling.

Two scheduling policies are compared under the same declared (Poisson)
arrival process and the same real, measured per-chunk compute time (from
actually executing the JX-01 model, not a synthetic number):

- **static batching**: wait for a full batch (or a max-wait timeout), then
  run every admitted request's entire trajectory before admitting the next
  batch.
- **chunk scheduling** (a minimal form of continuous batching): advance all
  active requests by a fixed step chunk each tick; a completed request's slot
  is freed and refilled by the next queued arrival immediately, without
  waiting for the rest of the batch.

A bounded ``CompiledVariantCache`` keyed by (padded) batch shape stands in
for "stable compile signatures and a bounded compiled-variant cache" --
dynamic arrival patterns must not trigger unbounded recompilation, so the
cache raises rather than silently growing past its bound.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import time

import jax
import numpy as np

from .model import ModelParams, TinyEpsilonModel
from .sampler import NoiseSchedule, make_schedule


class CompiledVariantCacheError(RuntimeError):
    """Raised instead of silently evicting or unboundedly growing the cache."""


class CompiledVariantCache:
    """A bounded LRU cache of compiled chunk-step functions, keyed by batch shape.

    Every distinct (batch_size, chunk_steps) pair is a distinct compile
    signature. Admission control -- not this cache -- is responsible for
    keeping the number of distinct shapes small (shape-aware bucketing);
    this cache's only job is to make "how many compiled variants exist right
    now" an explicit, bounded, observable number instead of an unbounded one.
    """

    def __init__(self, params: ModelParams, schedule: NoiseSchedule, max_variants: int = 4) -> None:
        self._params = params
        self._schedule = schedule
        self._max_variants = max_variants
        self._cache: OrderedDict[tuple[int, int], "callable"] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.compiled_variants_ever = 0

    def get_or_compile(self, batch_size: int, chunk_steps: int):
        key = (batch_size, chunk_steps)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]

        self.misses += 1
        if len(self._cache) >= self._max_variants:
            raise CompiledVariantCacheError(
                f"compiled-variant cache is full ({self._max_variants} variants); "
                "shape-aware admission should have bucketed this request into an existing variant"
            )

        fn = self._compile(batch_size, chunk_steps)
        self._cache[key] = fn
        self.compiled_variants_ever += 1
        return fn

    def _compile(self, batch_size: int, chunk_steps: int):
        from functools import partial

        from .sampler import ddpm_step

        params, schedule = self._params, self._schedule

        @partial(jax.jit, static_argnums=(2,))
        def run_chunk(latents, active_mask, start_step, base_keys):
            def body(carry, step):
                current, mask = carry

                def step_one(latent, key, is_active):
                    updated = ddpm_step(params, schedule, latent[None, :], step, key)[0]
                    return jax.numpy.where(is_active, updated, latent)

                next_latents = jax.vmap(step_one)(current, base_keys, mask)
                return (next_latents, mask), None

            steps = jax.numpy.arange(start_step, start_step - chunk_steps, -1)
            (final, _), _ = jax.lax.scan(body, (latents, active_mask), steps)
            return final

        return run_chunk

    @property
    def num_cached_variants(self) -> int:
        return len(self._cache)


@dataclass(frozen=True)
class SimulatedRequest:
    request_id: str
    arrival_time_s: float
    steps: int


@dataclass
class RequestOutcome:
    request_id: str
    arrival_time_s: float
    admitted_time_s: float
    completed_time_s: float

    @property
    def queue_delay_s(self) -> float:
        return self.admitted_time_s - self.arrival_time_s

    @property
    def latency_s(self) -> float:
        return self.completed_time_s - self.arrival_time_s


@dataclass
class ServingReport:
    policy: str
    outcomes: list[RequestOutcome] = field(default_factory=list)

    def latencies(self) -> np.ndarray:
        return np.array([o.latency_s for o in self.outcomes])

    def queue_delays(self) -> np.ndarray:
        return np.array([o.queue_delay_s for o in self.outcomes])

    def percentiles(self) -> dict:
        lat = self.latencies()
        return {
            "p50_s": float(np.percentile(lat, 50)),
            "p95_s": float(np.percentile(lat, 95)),
            "p99_s": float(np.percentile(lat, 99)),
        }

    def throughput_rps(self) -> float:
        if not self.outcomes:
            return 0.0
        span = max(o.completed_time_s for o in self.outcomes) - min(o.arrival_time_s for o in self.outcomes)
        return len(self.outcomes) / span if span > 0 else float("inf")

    def goodput_fraction(self, sla_s: float) -> float:
        if not self.outcomes:
            return 0.0
        met = sum(1 for o in self.outcomes if o.latency_s <= sla_s)
        return met / len(self.outcomes)

    def as_dict(self, sla_s: float) -> dict:
        return {
            "policy": self.policy,
            "num_requests": len(self.outcomes),
            "latency_percentiles": self.percentiles(),
            "mean_queue_delay_s": float(np.mean(self.queue_delays())) if self.outcomes else 0.0,
            "throughput_rps": self.throughput_rps(),
            "goodput_fraction_within_sla": self.goodput_fraction(sla_s),
            "sla_s": sla_s,
        }


def generate_poisson_arrivals(rate_per_s: float, num_requests: int, steps: int, seed: int) -> list[SimulatedRequest]:
    """A declared arrival distribution: Poisson process at a fixed rate."""

    rng = np.random.default_rng(seed)
    interarrival = rng.exponential(1.0 / rate_per_s, size=num_requests)
    arrival_times = np.cumsum(interarrival)
    return [
        SimulatedRequest(request_id=f"req_{i}", arrival_time_s=float(t), steps=steps)
        for i, t in enumerate(arrival_times)
    ]


def _measure_real_chunk_seconds(cache: CompiledVariantCache, batch_size: int, chunk_steps: int, start_step: int) -> float:
    """Actually run one compiled chunk step and time it -- not a synthetic estimate.

    ``start_step`` is fixed per call site rather than advanced per tick: this
    function measures the real wall-clock *cost* of one chunk of compute at a
    given (batch_size, chunk_steps) shape, to drive the scheduling/queueing
    simulation below. It does not track each simulated request's actual
    denoising progress -- that correctness question is already covered by
    JX-01/JX-02/JX-04's tests. Conflating the two would need a much larger
    model before the distinction mattered for cost.
    """

    fn = cache.get_or_compile(batch_size, chunk_steps)
    latents = jax.random.normal(jax.random.PRNGKey(0), (batch_size, TinyEpsilonModel.latent_dim))
    active_mask = jax.numpy.ones((batch_size,), dtype=bool)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(1), i))(jax.numpy.arange(batch_size))

    jax.block_until_ready(fn(latents, active_mask, start_step, keys))  # warm the compile
    start = time.perf_counter()
    out = fn(latents, active_mask, start_step, keys)
    jax.block_until_ready(out)
    return time.perf_counter() - start


def simulate_static_batching(
    cache: CompiledVariantCache, requests: list[SimulatedRequest], batch_size: int, total_steps: int
) -> ServingReport:
    report = ServingReport(policy="static_batching")
    clock = 0.0
    pending: list[SimulatedRequest] = []

    for req in sorted(requests, key=lambda r: r.arrival_time_s):
        pending.append(req)
        if len(pending) < batch_size:
            continue
        clock = max(clock, pending[-1].arrival_time_s)
        full_run_s = _measure_real_chunk_seconds(cache, batch_size, total_steps, total_steps - 1)
        completed_at = clock + full_run_s
        for p in pending:
            report.outcomes.append(RequestOutcome(p.request_id, p.arrival_time_s, clock, completed_at))
        clock = completed_at
        pending = []

    if pending:
        clock = max(clock, pending[-1].arrival_time_s)
        full_run_s = _measure_real_chunk_seconds(cache, len(pending), total_steps, total_steps - 1)
        completed_at = clock + full_run_s
        for p in pending:
            report.outcomes.append(RequestOutcome(p.request_id, p.arrival_time_s, clock, completed_at))

    return report


def simulate_chunk_scheduling(
    cache: CompiledVariantCache,
    requests: list[SimulatedRequest],
    max_active_slots: int,
    chunk_steps: int,
    total_steps: int,
) -> ServingReport:
    report = ServingReport(policy="chunk_scheduling")
    clock = 0.0
    queue = sorted(requests, key=lambda r: r.arrival_time_s)
    active: dict[str, tuple[SimulatedRequest, int, float]] = {}  # id -> (req, steps_done, admitted_at)
    queue_idx = 0

    while queue_idx < len(queue) or active:
        while queue_idx < len(queue) and len(active) < max_active_slots and queue[queue_idx].arrival_time_s <= clock:
            req = queue[queue_idx]
            active[req.request_id] = (req, 0, clock)
            queue_idx += 1

        if not active:
            if queue_idx < len(queue):
                clock = queue[queue_idx].arrival_time_s
                continue
            break

        chunk_s = _measure_real_chunk_seconds(cache, max_active_slots, chunk_steps, total_steps - 1)
        clock += chunk_s

        for req_id in list(active.keys()):
            req, steps_done, admitted_at = active[req_id]
            steps_done += chunk_steps
            if steps_done >= total_steps:
                report.outcomes.append(RequestOutcome(req.request_id, req.arrival_time_s, admitted_at, clock))
                del active[req_id]
            else:
                active[req_id] = (req, steps_done, admitted_at)

        while queue_idx < len(queue) and len(active) < max_active_slots and queue[queue_idx].arrival_time_s <= clock:
            req = queue[queue_idx]
            active[req.request_id] = (req, 0, clock)
            queue_idx += 1

    return report
