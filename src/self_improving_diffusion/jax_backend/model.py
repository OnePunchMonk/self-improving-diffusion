"""A tiny epsilon-prediction MLP: enough real math to exercise JAX, not SOTA.

The model operates on a flattened ``LATENT_DIM``-pixel grayscale image. It is
conditioned on the scalar diffusion timestep only (no text conditioning yet --
JX-01 asks for one small real path first, not prompt fidelity). Parameters are
a plain pytree of ``jnp.ndarray`` so the whole thing stays inspectable in
JAXPR/HLO dumps.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

LATENT_DIM = 64  # 8x8 flattened grayscale latent
HIDDEN_DIM = 128


class ModelParams(NamedTuple):
    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray
    w3: jnp.ndarray
    b3: jnp.ndarray


class TinyEpsilonModel:
    """Stateless functional model: init(key) -> params, apply(params, x, t) -> eps."""

    latent_dim = LATENT_DIM

    @staticmethod
    def init(key: jax.Array) -> ModelParams:
        k1, k2, k3 = jax.random.split(key, 3)
        in_dim = LATENT_DIM + 1  # +1 for the timestep embedding scalar
        scale1 = jnp.sqrt(2.0 / in_dim)
        scale2 = jnp.sqrt(2.0 / HIDDEN_DIM)
        return ModelParams(
            w1=jax.random.normal(k1, (in_dim, HIDDEN_DIM)) * scale1,
            b1=jnp.zeros((HIDDEN_DIM,)),
            w2=jax.random.normal(k2, (HIDDEN_DIM, HIDDEN_DIM)) * scale2,
            b2=jnp.zeros((HIDDEN_DIM,)),
            w3=jax.random.normal(k3, (HIDDEN_DIM, LATENT_DIM)) * scale2,
            b3=jnp.zeros((LATENT_DIM,)),
        )

    @staticmethod
    def apply(params: ModelParams, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Predict the noise added to ``x`` at (possibly batched) timestep ``t``."""

        t_embed = (t.astype(jnp.float32) / 1000.0).reshape(x.shape[0], 1)
        h = jnp.concatenate([x, t_embed], axis=-1)
        h = jax.nn.gelu(h @ params.w1 + params.b1)
        h = jax.nn.gelu(h @ params.w2 + params.b2)
        return h @ params.w3 + params.b3
