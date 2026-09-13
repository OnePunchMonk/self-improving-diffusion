"""JX-02: a trustworthy single-device baseline.

Separates compile time, dispatch time, device execution time, host transfer
time, verifier time, and decode time -- and never reports a duration without
first synchronizing (JAX dispatch is asynchronous; timing a call that hasn't
finished on-device measures dispatch latency, not execution latency). Also
reports the config needed to make a number reproducible: dtype, resolution,
batch size, step count, and whether the call was cold (first trace, includes
compilation) or warm (cached compiled executable).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import resource
import sys
import time

import jax
import numpy as np

from .backend import JaxVerifierBackend
from .model import ModelParams, TinyEpsilonModel
from .sampler import NoiseSchedule, ddpm_step, make_schedule
from ..executor import SampleArtifact


@dataclass
class PhaseTimings:
    compile_s: float
    dispatch_s: float
    execute_s: float
    transfer_s: float
    decode_s: float
    verify_s: float

    def total_s(self) -> float:
        return self.compile_s + self.dispatch_s + self.execute_s + self.transfer_s + self.decode_s + self.verify_s


@dataclass
class BaselineConfig:
    dtype: str
    resolution: int
    batch_size: int
    steps: int
    cache_state: str  # "cold" | "warm"


@dataclass
class BaselineReport:
    config: BaselineConfig
    timings: PhaseTimings
    peak_rss_bytes: int
    device: str
    numerical_match_vs_numpy: bool | None = None
    max_abs_diff_vs_numpy: float | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "timings_seconds": asdict(self.timings),
            "total_seconds": self.timings.total_s(),
            "peak_rss_bytes": self.peak_rss_bytes,
            "device": self.device,
            "numerical_match_vs_numpy": self.numerical_match_vs_numpy,
            "max_abs_diff_vs_numpy": self.max_abs_diff_vs_numpy,
            **self.extra,
        }


def _decode(latents: np.ndarray) -> np.ndarray:
    """A stand-in decode step: normalize to [0, 255] uint8, like the CLI does."""

    grid = latents.reshape(-1)
    normalized = (grid - grid.min()) / (np.ptp(grid) + 1e-8)
    return (normalized * 255).astype(np.uint8)


def run_single_device_baseline(
    params: ModelParams,
    schedule: NoiseSchedule,
    seed: int,
    steps: int,
    cache_state: str,
    jit_cache: dict | None = None,
) -> BaselineReport:
    """Run one trajectory and attribute wall-clock time to each real phase.

    ``jit_cache`` lets a caller reuse a compiled executable across calls to
    produce a genuine cold-vs-warm comparison (a fresh dict per process gives
    a cold call; passing the same dict on a second call gives a warm one).
    """

    base_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    init_latents = jax.random.normal(base_key, (1, TinyEpsilonModel.latent_dim))

    def full_trajectory(latents):
        def body(carry, step):
            return ddpm_step(params, schedule, carry, step, base_key), None

        steps_arr = jax.numpy.arange(schedule.betas.shape[0] - 1, -1, -1)
        final, _ = jax.lax.scan(body, latents, steps_arr)
        return final

    cache = jit_cache if jit_cache is not None else {}
    compile_s = 0.0
    if "compiled" not in cache:
        compile_start = time.perf_counter()
        lowered = jax.jit(full_trajectory).lower(init_latents)
        cache["compiled"] = lowered.compile()
        compile_s = time.perf_counter() - compile_start

    dispatch_start = time.perf_counter()
    device_result = cache["compiled"](init_latents)
    dispatch_s = time.perf_counter() - dispatch_start

    execute_start = time.perf_counter()
    jax.block_until_ready(device_result)
    execute_s = time.perf_counter() - execute_start

    transfer_start = time.perf_counter()
    host_result = np.asarray(device_result)
    transfer_s = time.perf_counter() - transfer_start

    decode_start = time.perf_counter()
    _decode(host_result)
    decode_s = time.perf_counter() - decode_start

    verify_start = time.perf_counter()
    sample = SampleArtifact(sample_id="baseline", sample_index=0, model_revision="tiny-eps-mlp@v0", steps=steps)
    verifier = JaxVerifierBackend(_ConstLatentDenoiser(host_result))
    verifier.score(sample, "tiny-heuristic-verifier@v0")
    verify_s = time.perf_counter() - verify_start

    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_rss_bytes *= 1024  # ru_maxrss is KB on Linux, bytes on macOS

    config = BaselineConfig(
        dtype=str(host_result.dtype),
        resolution=int(host_result.size**0.5),
        batch_size=host_result.shape[0],
        steps=steps,
        cache_state=cache_state,
    )
    timings = PhaseTimings(
        compile_s=compile_s,
        dispatch_s=dispatch_s,
        execute_s=execute_s,
        transfer_s=transfer_s,
        decode_s=decode_s,
        verify_s=verify_s,
    )
    return BaselineReport(
        config=config,
        timings=timings,
        peak_rss_bytes=peak_rss_bytes,
        device=str(jax.devices()[0]),
    )


def compare_against_numpy_reference(
    params: ModelParams,
    schedule: NoiseSchedule,
    seed: int,
    tolerance: float = 1e-4,
) -> tuple[bool, float]:
    """Run the JAX and eager-NumPy paths on the same weights/schedule/noise.

    Returns ``(within_tolerance, max_abs_diff)``. This is the "relevant
    reference implementation at matched outputs/configuration" JX-02 calls
    for: same math, same RNG draws, no compiler.
    """

    from .numpy_reference import run_trajectory_numpy
    from .sampler import run_trajectory

    steps = schedule.betas.shape[0]
    base_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    init_latents = jax.random.normal(base_key, (1, TinyEpsilonModel.latent_dim))

    jax_result = np.asarray(run_trajectory(params, schedule, init_latents, base_key, steps - 1, -1))

    step_noises = {
        step: np.asarray(jax.random.normal(jax.random.fold_in(base_key, step), init_latents.shape))
        for step in range(steps)
    }
    numpy_result = run_trajectory_numpy(
        params, schedule, np.asarray(init_latents), step_noises, steps - 1, -1
    )

    max_abs_diff = float(np.max(np.abs(jax_result - numpy_result)))
    return max_abs_diff <= tolerance, max_abs_diff


class _ConstLatentDenoiser:
    """Adapter so JaxVerifierBackend can score an already-computed array."""

    def __init__(self, latents: np.ndarray) -> None:
        self._latents = latents

    def final_latents(self, sample_id: str) -> np.ndarray:
        return self._latents
