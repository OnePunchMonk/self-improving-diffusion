"""JX-02: the eager NumPy reference must match the compiled JAX path."""

import jax

from self_improving_diffusion.jax_backend.benchmark import compare_against_numpy_reference
from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import make_schedule


def test_numpy_reference_matches_jax_within_tolerance():
    params = TinyEpsilonModel.init(jax.random.PRNGKey(5))
    schedule = make_schedule(16)

    matches, max_abs_diff = compare_against_numpy_reference(params, schedule, seed=5)

    assert matches, f"numpy reference diverged from JAX by {max_abs_diff}"
    assert max_abs_diff < 1e-4


def test_numpy_reference_diverges_with_wrong_weights():
    params_a = TinyEpsilonModel.init(jax.random.PRNGKey(1))
    schedule = make_schedule(16)

    # Deliberately compare trajectories seeded differently to confirm the
    # comparison is actually sensitive, not vacuously "matching".
    matches, max_abs_diff = compare_against_numpy_reference(params_a, schedule, seed=1)
    assert matches

    params_b = TinyEpsilonModel.init(jax.random.PRNGKey(999))
    from self_improving_diffusion.jax_backend.numpy_reference import run_trajectory_numpy
    from self_improving_diffusion.jax_backend.sampler import run_trajectory
    import numpy as np

    steps = 16
    base_key = jax.random.fold_in(jax.random.PRNGKey(1), 0)
    init_latents = jax.random.normal(base_key, (1, TinyEpsilonModel.latent_dim))
    jax_result = np.asarray(run_trajectory(params_a, schedule, init_latents, base_key, steps - 1, -1))

    step_noises = {
        step: np.asarray(jax.random.normal(jax.random.fold_in(base_key, step), init_latents.shape))
        for step in range(steps)
    }
    mismatched_numpy_result = run_trajectory_numpy(
        params_b, schedule, np.asarray(init_latents), step_noises, steps - 1, -1
    )
    assert not np.allclose(jax_result, mismatched_numpy_result, atol=1e-4)
