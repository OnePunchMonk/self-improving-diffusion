"""JX-01: a tiny, real JAX denoising path implementing the v0 executor contracts.

Everything here operates on actual arrays and actual model math -- unlike the
``DeterministicBackend`` in :mod:`self_improving_diffusion.executor`, which only
produces digest-derived placeholders. This package is intentionally small: one
MLP epsilon-predictor over a flattened tiny image, a linear-beta DDPM sampler,
and exact checkpoint/resume of real trajectory state.
"""

from .backend import JaxDenoiserBackend, JaxVerifierBackend
from .model import TinyEpsilonModel
from .state import TrajectoryState, load_checkpoint, save_checkpoint

__all__ = [
    "JaxDenoiserBackend",
    "JaxVerifierBackend",
    "TinyEpsilonModel",
    "TrajectoryState",
    "load_checkpoint",
    "save_checkpoint",
]
