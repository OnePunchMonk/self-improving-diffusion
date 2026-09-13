# JX-01: a small, real JAX generation path

This is the first real (non-fixture) vertical slice, sitting under the
`ProgramExecutor`/`GenerationProgram` contracts built in #1/#2. It replaces
`DeterministicBackend`'s digest-derived placeholders with an actual, if tiny,
JAX model and DDPM sampler for one `DenoiserBackend`/`VerifierBackend` pair.

## What's real here

- **Model** (`jax_backend/model.py`): a 3-layer MLP epsilon-predictor over a
  flattened 8x8 grayscale latent (64-dim), conditioned on the scalar
  timestep. Parameters are a plain `NamedTuple` of `jnp.ndarray`, initialized
  from a `jax.random.PRNGKey` — inspectable in JAXPR/HLO dumps, no framework
  dependency (no Flax/Optax yet).
- **Sampler** (`jax_backend/sampler.py`): linear-beta-schedule DDPM. The
  reverse process is a single `jax.lax.scan` from `run_trajectory`, so it
  compiles to one XLA computation per contiguous step range.
- **RNG discipline**: every step's noise draw is `jax.random.fold_in(base_key,
  step)`, never a mutated running key. This is what makes exact resume
  possible — the trajectory is a pure function of `(base_key, step)`, not of
  how many times you've called into the sampler.
- **Checkpoint/resume** (`jax_backend/state.py`): `TrajectoryState` serializes
  real latents (`.npz`) plus metadata (`.json`) with a genuine sha256 digest
  over the array bytes, not a digest over the request. `state_digest()` is
  checked on load; a tampered checkpoint is rejected.
- **Backend adapter** (`jax_backend/backend.py`): `JaxDenoiserBackend` and
  `JaxVerifierBackend` implement the same `DenoiserBackend`/`VerifierBackend`
  protocols `executor.ProgramExecutor` already expects, so `#1`/`#2`'s
  scheduler, checkpoint store, and program DSL work unmodified against real
  generation.

## What's deliberately not real yet

- No text conditioning — the model only sees the timestep. Prompt fidelity is
  out of scope for "one small real path"; GenEval-style checks come later
  (JX-08 reading list).
- No Pallas/vendor kernels — reference JAX ops only, per JX-01/JX-02 ordering
  (measure before you optimize).
- `JaxVerifierBackend`'s quality/alignment scores are statistics of the raw
  latent (variance, mean), not a trained verifier. They exist to exercise the
  `score`/`choose` program nodes, not to make a quality claim.
- Only one accelerator family (CPU here; the same code runs on a single GPU
  device with no changes, but that hasn't been measured yet — that's JX-02).

## Evidence

`python -m self_improving_diffusion.jax_backend.cli` is the one command asked
for in JX-01: it runs a program end to end, decodes+saves an image, writes a
`manifest.json` with the full execution trace, timings split into
first-call (compile + run) versus warm-call, and an explicit
`exact_resume_equivalence` boolean computed by running the same trajectory
uninterrupted and via checkpoint-then-resume and comparing bit-for-bit.

## Migrating off the Python 3.9 fixture implementation

The digest-only fixture code (`executor.py`, `program.py`, etc. from #1/#2)
declared `requires-python = ">=3.9"` because it only used the standard
library. `jax_backend` needs `>=3.11` (current JAX wheels), so `pyproject.toml`
now requires `>=3.11` project-wide and gates JAX itself behind the `jax`
extra (`pip install -e ".[dev,jax]"`) so the fixture-only tests still run
without a JAX install.
