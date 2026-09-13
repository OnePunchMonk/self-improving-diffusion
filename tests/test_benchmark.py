"""JX-02: phase-separated timing must be non-negative, synchronized, and cold != warm."""

import jax

from self_improving_diffusion.jax_backend.benchmark import run_single_device_baseline
from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import make_schedule


def _params_and_schedule(steps: int = 8):
    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    return params, make_schedule(steps)


def test_all_phase_timings_are_non_negative():
    params, schedule = _params_and_schedule()
    report = run_single_device_baseline(params, schedule, seed=0, steps=8, cache_state="cold")
    timings = report.timings
    for value in (
        timings.compile_s,
        timings.dispatch_s,
        timings.execute_s,
        timings.transfer_s,
        timings.decode_s,
        timings.verify_s,
    ):
        assert value >= 0.0


def test_cold_call_compiles_and_warm_call_reuses_cache():
    params, schedule = _params_and_schedule()
    jit_cache: dict = {}
    cold = run_single_device_baseline(params, schedule, seed=0, steps=8, cache_state="cold", jit_cache=jit_cache)
    warm = run_single_device_baseline(params, schedule, seed=0, steps=8, cache_state="warm", jit_cache=jit_cache)

    assert cold.timings.compile_s > 0.0
    assert warm.timings.compile_s == 0.0
    assert cold.config.cache_state == "cold"
    assert warm.config.cache_state == "warm"


def test_report_records_reproducibility_config():
    params, schedule = _params_and_schedule(steps=12)
    report = run_single_device_baseline(params, schedule, seed=0, steps=12, cache_state="cold")
    assert report.config.steps == 12
    assert report.config.batch_size == 1
    assert report.peak_rss_bytes > 0
    assert report.device
