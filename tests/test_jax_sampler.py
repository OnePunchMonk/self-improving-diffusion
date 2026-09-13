"""JX-01: fixed-seed determinism for the real JAX sampler."""

import jax
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import make_schedule, run_trajectory


def _run(seed: int, steps: int = 8) -> np.ndarray:
    params = TinyEpsilonModel.init(jax.random.PRNGKey(seed))
    schedule = make_schedule(steps)
    base_key = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    init_latents = jax.random.normal(base_key, (1, TinyEpsilonModel.latent_dim))
    final = run_trajectory(params, schedule, init_latents, base_key, steps - 1, -1)
    return np.asarray(final)


def test_same_seed_is_bit_identical():
    first = _run(seed=7)
    second = _run(seed=7)
    assert np.array_equal(first, second)


def test_different_seed_diverges():
    first = _run(seed=1)
    second = _run(seed=2)
    assert not np.array_equal(first, second)


def test_output_shape_matches_latent_dim():
    result = _run(seed=0)
    assert result.shape == (1, TinyEpsilonModel.latent_dim)
    assert np.all(np.isfinite(result))
