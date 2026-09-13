"""Real JAX implementations of the executor's ``DenoiserBackend``/``VerifierBackend``.

These satisfy the same protocols as ``executor.DeterministicBackend`` so a
``ProgramExecutor`` can run either one unmodified, but every artifact here is
derived from actual denoising math instead of a digest over the request.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

from ..executor import Checkpoint, SampleArtifact, ScoreReport
from ..program import GenerationProgram
from .model import ModelParams, TinyEpsilonModel
from .sampler import ddpm_step, make_schedule
from .state import TrajectoryState, save_checkpoint
from .text_encoder import encode_prompt


class JaxDenoiserBackend:
    """One tiny model instance driving real ``sample``/``checkpoint`` calls.

    ``sample()`` runs the full reverse trajectory once and caches every
    intermediate latents array in memory, keyed by sample id, so a later
    ``checkpoint(sample, step)`` call can persist the *real* state at that
    step rather than a synthetic digest.
    """

    def __init__(self, seed: int = 0, checkpoint_dir: Path | None = None) -> None:
        self._params: ModelParams = TinyEpsilonModel.init(jax.random.PRNGKey(seed))
        self._checkpoint_dir = checkpoint_dir
        self._trajectories: Dict[str, Dict[int, np.ndarray]] = {}

    def sample(self, program: GenerationProgram, sample_index: int) -> SampleArtifact:
        spec = program.spec
        schedule = make_schedule(spec.steps)
        base_key = jax.random.fold_in(jax.random.PRNGKey(spec.seed), sample_index)
        init_latents = jax.random.normal(base_key, (1, TinyEpsilonModel.latent_dim))
        cond = encode_prompt(spec.prompt)

        history: Dict[int, np.ndarray] = {spec.steps - 1: np.asarray(init_latents)}
        latents = init_latents
        for step in range(spec.steps - 1, -1, -1):
            latents = ddpm_step(self._params, schedule, latents, step, base_key, cond)
            history[step - 1] = np.asarray(latents)

        final_latents = history[-1]
        digest_source = final_latents.astype(np.float32).tobytes()
        sample_id = f"sample_{sha256(digest_source).hexdigest()[:20]}"
        self._trajectories[sample_id] = history
        return SampleArtifact(
            sample_id=sample_id,
            sample_index=sample_index,
            model_revision=spec.model.identifier,
            steps=spec.steps,
        )

    def checkpoint(self, sample: SampleArtifact, step: int) -> Checkpoint:
        history = self._trajectories[sample.sample_id]
        latents = history[step]
        state = TrajectoryState(
            sample_id=sample.sample_id,
            step=step,
            base_seed=sample.sample_index,
            model_revision=sample.model_revision,
            latents=latents,
        )
        if self._checkpoint_dir is not None:
            save_checkpoint(self._checkpoint_dir / f"{sample.sample_id}_step{step}", state)
        return Checkpoint(sample_id=sample.sample_id, step=step, state_digest=state.state_digest())

    def final_latents(self, sample_id: str) -> np.ndarray:
        return self._trajectories[sample_id][-1]

    def params(self) -> ModelParams:
        return self._params


class JaxVerifierBackend:
    """Score a sample by real statistics of its decoded latent, not a hash."""

    def __init__(self, denoiser: JaxDenoiserBackend) -> None:
        self._denoiser = denoiser

    def score(self, sample: SampleArtifact, verifier_revision: str) -> ScoreReport:
        latents = jnp.asarray(self._denoiser.final_latents(sample.sample_id))
        # Tiny, real proxy metrics: lower spatial variance reads as smoother
        # (higher "quality") and closeness to zero mean reads as "alignment".
        # These do not yet compare the decoded latent against the real text
        # conditioning embedding now available (backend.py encodes spec.prompt),
        # so "alignment" here is still a placeholder for an actual verifier
        # model, not digest-derived fixtures.
        quality = float(1.0 / (1.0 + jnp.var(latents)))
        alignment = float(1.0 / (1.0 + jnp.abs(jnp.mean(latents))))
        uncertainty = float(jnp.std(latents))
        return ScoreReport(
            sample_id=sample.sample_id,
            verifier_revision=verifier_revision,
            quality=quality,
            alignment=alignment,
            uncertainty=uncertainty,
        )
