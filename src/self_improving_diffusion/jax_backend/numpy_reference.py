"""A pure-NumPy, eager reimplementation of the sampler -- the JX-02 baseline.

JX-02 asks to "compare ordinary JAX execution with a relevant reference
implementation at matched outputs/configuration." The most honest reference
for a hand-written JAX model is the same math with no compiler and no
acceleration: plain NumPy, run eagerly, one step at a time. It shares
``ModelParams`` and the linear-beta schedule with the JAX path so a
comparison is apples-to-apples -- same weights, same schedule, same RNG
draws, same conditioning embedding -- and only the execution substrate
differs.
"""

from __future__ import annotations

import numpy as np

from .model import COND_DIM, ModelParams
from .sampler import NoiseSchedule


def _gelu(x: np.ndarray) -> np.ndarray:
    # Matches jax.nn.gelu's default (tanh) approximation so the two
    # implementations are numerically comparable, not just structurally similar.
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def apply_numpy(params: ModelParams, x: np.ndarray, t: np.ndarray, cond: np.ndarray | None = None) -> np.ndarray:
    w1, b1, w2, b2, w3, b3 = (np.asarray(p) for p in params)
    cond_batch = np.zeros((x.shape[0], COND_DIM), dtype=np.float32) if cond is None else np.broadcast_to(
        cond, (x.shape[0], COND_DIM)
    )
    t_embed = (t.astype(np.float32) / 1000.0).reshape(x.shape[0], 1)
    h = np.concatenate([x, t_embed, cond_batch], axis=-1)
    h = _gelu(h @ w1 + b1)
    h = _gelu(h @ w2 + b2)
    return h @ w3 + b3


def ddpm_step_numpy(
    params: ModelParams,
    schedule: NoiseSchedule,
    latents: np.ndarray,
    step: int,
    step_noise: np.ndarray,
    cond: np.ndarray | None = None,
) -> np.ndarray:
    """One reverse-process update, given externally-drawn noise for this step.

    Noise is passed in (rather than drawn here) so callers can reuse the exact
    ``jax.random.fold_in`` draws from the JAX path -- comparing sampler math,
    not RNG implementations.
    """

    beta_t = float(np.asarray(schedule.betas)[step])
    alpha_t = float(np.asarray(schedule.alphas)[step])
    alpha_bar_t = float(np.asarray(schedule.alphas_cumprod)[step])

    t_batch = np.full((latents.shape[0],), step)
    eps_pred = apply_numpy(params, latents, t_batch, cond)

    mean = (latents - (beta_t / np.sqrt(1.0 - alpha_bar_t)) * eps_pred) / np.sqrt(alpha_t)
    noise = step_noise if step > 0 else np.zeros_like(latents)
    return mean + np.sqrt(beta_t) * noise


def run_trajectory_numpy(
    params: ModelParams,
    schedule: NoiseSchedule,
    latents: np.ndarray,
    step_noises: dict[int, np.ndarray],
    start_step: int,
    end_step: int = -1,
    cond: np.ndarray | None = None,
) -> np.ndarray:
    """Eager, unrolled reverse process -- one Python-level step per iteration."""

    current = latents
    for step in range(start_step, end_step, -1):
        current = ddpm_step_numpy(params, schedule, current, step, step_noises[step], cond)
    return current
