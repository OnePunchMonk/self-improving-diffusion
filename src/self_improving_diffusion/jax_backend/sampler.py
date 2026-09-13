"""A linear-beta DDPM sampler with per-step RNG derived from a fixed base key.

Determinism and resumability both depend on one rule: the RNG used at step
``t`` is ``jax.random.fold_in(base_key, t)``, never a mutated running key. That
makes the trajectory a pure function of ``(base_key, step)``, so restarting
from a stored ``(step, latents)`` pair and continuing forward reproduces the
same noise draws -- and therefore bit-identical output -- as an uninterrupted
run. Branching (a deliberately different continuation) must fold in a
different base key or step offset, never reuse this one silently.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .model import ModelParams, TinyEpsilonModel


class NoiseSchedule(NamedTuple):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray


def make_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> NoiseSchedule:
    betas = jnp.linspace(beta_start, beta_end, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)
    return NoiseSchedule(betas=betas, alphas=alphas, alphas_cumprod=alphas_cumprod)


def ddpm_step(
    params: ModelParams,
    schedule: NoiseSchedule,
    latents: jnp.ndarray,
    step: int,
    base_key: jax.Array,
    cond: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Reverse-process update from ``step`` to ``step - 1`` (step 0 is clean).

    ``cond`` is the text-conditioning embedding for this trajectory (see
    ``text_encoder.py``); omitting it uses the null/unconditioned embedding,
    matching every call site that predates real text conditioning.
    """

    beta_t = schedule.betas[step]
    alpha_t = schedule.alphas[step]
    alpha_bar_t = schedule.alphas_cumprod[step]

    t_batch = jnp.full((latents.shape[0],), step)
    eps_pred = TinyEpsilonModel.apply(params, latents, t_batch, cond)

    mean = (latents - (beta_t / jnp.sqrt(1.0 - alpha_bar_t)) * eps_pred) / jnp.sqrt(alpha_t)

    step_key = jax.random.fold_in(base_key, step)
    noise = jnp.where(
        step > 0,
        jax.random.normal(step_key, latents.shape),
        jnp.zeros_like(latents),
    )
    return mean + jnp.sqrt(beta_t) * noise


def run_trajectory(
    params: ModelParams,
    schedule: NoiseSchedule,
    latents: jnp.ndarray,
    base_key: jax.Array,
    start_step: int,
    end_step: int = -1,
    cond: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Run the reverse process from ``start_step`` down to ``end_step`` (exclusive).

    Steps count down: ``start_step`` is the noisiest state being consumed,
    ``end_step`` (default ``-1``, i.e. fully denoised) is where iteration stops.
    Resuming from a checkpoint means calling this with the checkpointed
    ``latents`` and ``start_step`` equal to the checkpointed step, and the
    same ``cond`` the original trajectory used -- conditioning is part of
    the trajectory's identity, exactly like ``base_key``.
    """

    def body(carry, step):
        return ddpm_step(params, schedule, carry, step, base_key, cond), None

    steps = jnp.arange(start_step, end_step, -1)
    final_latents, _ = jax.lax.scan(body, latents, steps)
    return final_latents
