"""JX-01: exact resume must reproduce an uninterrupted run, bit-for-bit."""

from pathlib import Path

import jax
import numpy as np

from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import make_schedule, run_trajectory
from self_improving_diffusion.jax_backend.state import TrajectoryState, load_checkpoint, save_checkpoint

STEPS = 12


def _setup():
    params = TinyEpsilonModel.init(jax.random.PRNGKey(3))
    schedule = make_schedule(STEPS)
    base_key = jax.random.fold_in(jax.random.PRNGKey(3), 0)
    init_latents = jax.random.normal(base_key, (1, TinyEpsilonModel.latent_dim))
    return params, schedule, base_key, init_latents


def test_resume_from_checkpoint_matches_uninterrupted_run():
    params, schedule, base_key, init_latents = _setup()
    resume_step = STEPS // 2

    full_run = run_trajectory(params, schedule, init_latents, base_key, STEPS - 1, -1)

    midpoint = run_trajectory(params, schedule, init_latents, base_key, STEPS - 1, resume_step)
    resumed = run_trajectory(params, schedule, midpoint, base_key, resume_step, -1)

    assert np.array_equal(np.asarray(full_run), np.asarray(resumed))


def test_serialized_checkpoint_round_trips_and_resumes(tmp_path: Path):
    params, schedule, base_key, init_latents = _setup()
    resume_step = STEPS // 2

    midpoint = run_trajectory(params, schedule, init_latents, base_key, STEPS - 1, resume_step)
    state = TrajectoryState(
        sample_id="sample_test",
        step=resume_step,
        base_seed=0,
        model_revision="tiny-eps-mlp@v0",
        latents=np.asarray(midpoint),
    )
    checkpoint_path = tmp_path / "checkpoint"
    save_checkpoint(checkpoint_path, state)

    loaded = load_checkpoint(checkpoint_path)
    assert np.array_equal(loaded.latents, np.asarray(midpoint))

    resumed_from_disk = run_trajectory(
        params, schedule, jax.numpy.asarray(loaded.latents), base_key, resume_step, -1
    )
    full_run = run_trajectory(params, schedule, init_latents, base_key, STEPS - 1, -1)
    assert np.array_equal(np.asarray(resumed_from_disk), np.asarray(full_run))


def test_tampered_checkpoint_fails_integrity_check(tmp_path: Path):
    state = TrajectoryState(
        sample_id="sample_test",
        step=0,
        base_seed=0,
        model_revision="tiny-eps-mlp@v0",
        latents=np.zeros((1, TinyEpsilonModel.latent_dim), dtype=np.float32),
    )
    checkpoint_path = tmp_path / "checkpoint"
    save_checkpoint(checkpoint_path, state)

    import json

    meta_path = checkpoint_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["state_digest"] = "tampered"
    meta_path.write_text(json.dumps(meta))

    try:
        load_checkpoint(checkpoint_path)
        assert False, "expected integrity failure"
    except ValueError:
        pass
