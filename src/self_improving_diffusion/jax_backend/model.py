"""A tiny epsilon-prediction MLP: enough real math to exercise JAX, not SOTA.

The model operates on a flattened ``LATENT_DIM``-pixel grayscale image,
conditioned on the scalar diffusion timestep and a ``COND_DIM`` text
embedding from ``text_encoder.py``. Parameters are a plain pytree of
``jnp.ndarray`` so the whole thing stays inspectable in JAXPR/HLO dumps.

``cond`` defaults to a zero vector when omitted -- classifier-free
guidance's "null" embedding convention, not a special case bolted on. Every
call site that predates real text conditioning keeps working unchanged and
unconditioned; only ``JaxDenoiserBackend`` (and anything that explicitly
wants conditioning) needs to pass a real one.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

LATENT_DIM = 64  # 8x8 flattened grayscale latent
HIDDEN_DIM = 128
COND_DIM = 16  # must match text_encoder.EMBED_DIM


class ModelParams(NamedTuple):
    w1: jnp.ndarray
    b1: jnp.ndarray
    w2: jnp.ndarray
    b2: jnp.ndarray
    w3: jnp.ndarray
    b3: jnp.ndarray


class TinyEpsilonModel:
    """Stateless functional model: init(key) -> params, apply(params, x, t, cond) -> eps."""

    latent_dim = LATENT_DIM
    cond_dim = COND_DIM

    @staticmethod
    def init(key: jax.Array) -> ModelParams:
        k1, k2, k3 = jax.random.split(key, 3)
        in_dim = LATENT_DIM + 1 + COND_DIM  # +1 timestep embedding, +COND_DIM text conditioning
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
    def apply(
        params: ModelParams, x: jnp.ndarray, t: jnp.ndarray, cond: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Predict the noise added to ``x`` at (possibly batched) timestep ``t``.

        ``cond`` is a single ``(COND_DIM,)`` embedding broadcast across the
        batch (one prompt per denoising trajectory, per JX-01's one-sample
        contract) or an already-batched ``(batch, COND_DIM)`` array.
        Defaults to the null (zero) embedding when omitted.
        """

        if cond is None:
            cond_batch = jnp.zeros((x.shape[0], COND_DIM))
        elif cond.ndim == 1:
            cond_batch = jnp.broadcast_to(cond, (x.shape[0], COND_DIM))
        else:
            cond_batch = cond

        t_embed = (t.astype(jnp.float32) / 1000.0).reshape(x.shape[0], 1)
        h = jnp.concatenate([x, t_embed, cond_batch], axis=-1)
        h = jax.nn.gelu(h @ params.w1 + params.b1)
        h = jax.nn.gelu(h @ params.w2 + params.b2)
        return h @ params.w3 + params.b3
