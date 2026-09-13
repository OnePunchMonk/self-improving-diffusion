"""A tiny, real (if architecturally minimal) text conditioning path.

Every prior milestone's `GenerationSpec.prompt` was accepted and hashed into
sample/program digests but never actually influenced denoising -- the model
only ever saw the timestep. This closes that gap with the smallest thing
that is genuinely a text encoder: whitespace tokenization, deterministic
hashing into a fixed vocabulary (sha256, not Python's salted `hash()`, so
token ids are stable across processes and runs), an embedding table lookup,
and mean pooling.

This is not CLIP. It has no training signal and no notion of semantic
similarity beyond "these two prompts happen to share hashed tokens" -- but
it is a real embedding lookup and a real pooling reduction, not a stand-in
digest. `encode_prompt` is a frozen, module-level singleton (never
retrained, seeded once) so that changing which *sampling* seed a request
uses never changes how its prompt is conditioned on -- conditioning and
sampling are independent knobs, matching how a real frozen text encoder
would sit in front of a diffusion model whose own seed varies per request.
"""

from __future__ import annotations

from hashlib import sha256
from typing import NamedTuple

import jax
import jax.numpy as jnp

VOCAB_SIZE = 4_096
EMBED_DIM = 16  # == COND_DIM in model.py; kept as a separate name for encoder-side clarity

FROZEN_TEXT_ENCODER_SEED = 4_242


class TextEncoderParams(NamedTuple):
    embedding_table: jnp.ndarray  # (VOCAB_SIZE, EMBED_DIM)


class TinyTextEncoder:
    """Stateless functional encoder: init(key) -> params, encode(params, prompt) -> (EMBED_DIM,)."""

    embed_dim = EMBED_DIM

    @staticmethod
    def init(key: jax.Array) -> TextEncoderParams:
        scale = 1.0 / jnp.sqrt(EMBED_DIM)
        table = jax.random.normal(key, (VOCAB_SIZE, EMBED_DIM)) * scale
        return TextEncoderParams(embedding_table=table)

    @staticmethod
    def tokenize(prompt: str) -> list[int]:
        """Deterministic hash tokenization -- stable across processes, unlike hash()."""

        tokens = prompt.strip().lower().split()
        if not tokens:
            return [0]
        return [int(sha256(token.encode()).hexdigest(), 16) % VOCAB_SIZE for token in tokens]

    @staticmethod
    def encode(params: TextEncoderParams, prompt: str) -> jnp.ndarray:
        """Mean-pooled embedding lookup over the prompt's hashed tokens."""

        token_ids = jnp.asarray(TinyTextEncoder.tokenize(prompt))
        embeddings = params.embedding_table[token_ids]
        return jnp.mean(embeddings, axis=0)


_FROZEN_PARAMS = TinyTextEncoder.init(jax.random.PRNGKey(FROZEN_TEXT_ENCODER_SEED))


def encode_prompt(prompt: str) -> jnp.ndarray:
    """The frozen, shared text encoder every backend/CLI should use.

    Never re-seeded per request or per model instance -- a real frozen text
    encoder doesn't change because the diffusion model's own sampling seed
    does.
    """

    return TinyTextEncoder.encode(_FROZEN_PARAMS, prompt)
