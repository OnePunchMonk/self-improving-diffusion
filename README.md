# self-improving-diffusion

An inference framework for **image** diffusion models, structurally analogous to SGLang for LLMs — but where recursive self-improvement (RSI) is a scheduler-level concern built into the serving path, not a separate offline training pipeline.

Video and general vision tasks are explicitly out of scope for the core engine (see `docs/ARCHITECTURE.md`).

- `docs/DESIGN.md` — literature review: test-time scaling, self-play preference optimization (SPIN-Diffusion, Diffusion-DPO, VideoDPO), reward modeling, known failure modes.
- `docs/ARCHITECTURE.md` — the SGLang-shaped framework design: step-cache (RadixAttention analog), step-level continuous batching, a generation-program DSL, and RSI wired into the request lifecycle.
- `docs/DESIGN_V2.md` — recommended next iteration: a quality-gated serving and learning fabric with exact trajectory checkpoints, consent-aware data capture, and shadow/canary promotion gates.
- `docs/RESEARCH.md` — full raw reading list behind both docs above, with a suggested reading order.
- `docs/STUDY_PLAN.md` — broader curriculum across five tracks: inference frameworks, RSI, diffusion, harness/eval/post-training, and GPU internals.

## Layout
- `src/engine/` — scheduler, step-level continuous batching, paged latent memory, step-cache
- `src/runtime/` — denoiser backends (model forward passes)
- `src/api/` — generation-program DSL (denoise → critic → refine → self-play-emit as one program)
- `src/verifiers/` — critic / reward models (discriminator + region localization)
- `src/rsi/` — self-improvement subsystem, called from the engine during serving
  - `search/` — localized best-of-N / iterative refinement
  - `data/` — self-play preference pair construction from served traffic
  - `training/` — async DPO/self-play fine-tune + distillation, threshold-triggered
- `src/eval/` — held-out reward tracking, critic-human agreement, cache/batching efficiency
- `experiments/` — run configs and logs
- `notebooks/` — exploratory analysis

## First executable slice

The initial implementation deliberately starts with the control-plane contract,
not a model dependency. It provides a typed and bounded generation program:

```python
from self_improving_diffusion import GenerationSpec, ModelRef, ProgramBuilder

program = (
    ProgramBuilder(GenerationSpec("a red cube", ModelRef("base", "r17"), seed=42))
    .sample(checkpoint_at=(12,))
    .score(ModelRef("critic", "r6"))
    .choose()
    .emit_trace()
    .build()
)
```

Run its contract tests with `python -m pytest`. The local deterministic backend
then provides a replayable `sample -> checkpoint -> score -> choose -> trace`
execution path without downloading model weights; it is a control-plane test
double, not an image generator.

Execution traces can be converted into an append-only local learning event only
when `emit_trace(TrainingConsent.OPT_IN)` is declared. The default event holds
provenance, aggregate verifier scores, and a prompt digest—not the raw prompt,
pixels, or latent state.

The v0 checkpoint store is explicitly request-scoped and TTL-bounded. It can
resume an exact trajectory for a retry or later repair, but intentionally has
no cross-request similarity lookup or silent live-state eviction.

`StepBatchScheduler` is the current step-level batching primitive. It accepts
only dependency-ready nodes, batches identical immutable denoiser shapes and
operation kinds, and rotates between compatibility classes so one workload
cannot starve another.

## JX-01: a real JAX generation path

`src/self_improving_diffusion/jax_backend/` implements the same
`DenoiserBackend`/`VerifierBackend` protocols above with actual JAX math
instead of digest-derived fixtures: a tiny MLP epsilon-predictor, a
linear-beta DDPM sampler, and exact checkpoint/resume of real trajectory
state. See `docs/architecture/jax-runtime.md` for what's real, what's
deliberately still a stand-in, and why `pyproject.toml` now requires
Python >= 3.11.

Install it with `pip install -e ".[dev,jax]"`, then run:

```
python -m self_improving_diffusion.jax_backend.cli --prompt "a red square" --seed 0 --steps 24 --out-dir runs/jx01
```

This writes `runs/jx01/image.png` and `runs/jx01/manifest.json` (execution
trace, compile-vs-warm timings, and an explicit bit-for-bit
checkpoint/resume equivalence check).

## JX-02: a trustworthy single-device baseline

```
python -m self_improving_diffusion.jax_backend.benchmark_cli --out reports/jx02_baseline.json
python -m self_improving_diffusion.eval.cli --out reports/jx02_quality_compute.json
```

The first command separates compile, dispatch, execute, transfer, decode,
and verify time (each synchronized before it's recorded), reports a
cold-vs-warm comparison and peak host RSS, and cross-checks the JAX result
against `jax_backend/numpy_reference.py` -- an eager, unrolled, pure-NumPy
reimplementation sharing the same weights/schedule/RNG draws. The second
runs the frozen quality/compute protocol (`eval/quality_compute.py`, fixed
seeds and step budgets, declared once so results stay comparable across
code changes) and reports the mean-quality-per-step-budget frontier.
Findings from the current CPU baseline: `docs/learning/jx-02-baseline.md`.

## JX-03: GPU internals through measured kernels

`benchmarks/modal/jx03_gpu_internals.py` runs the JX-02 baseline plus two
kernel exercises (elementwise fusion, tiled GEMM with a NumPy correctness
oracle) on a cheap Modal T4:

```
pip install modal && modal setup   # once, to authenticate
modal run benchmarks/modal/jx03_gpu_internals.py
```

Raw output: `reports/jx03_gpu_internals.json`. Findings, including a kept
negative result (the tiny JX-01 model is ~11x *slower* on this GPU than on
CPU -- launch overhead dominates at this model size) and a real
hardware-capability boundary (T4 has no bf16 tensor cores, so bf16 GEMM
doesn't beat fp32 here): `docs/learning/jx-03-gpu-internals.md`. Nsight
Compute kernel-level profiling was not available in this container; that
gap is called out rather than papered over.

## JX-04: distributed correctness before hybrid optimization

Request/data parallelism only (no intra-request sharding yet -- the tiny
model doesn't warrant it). Needs >= 2 devices, simulated on CPU via:

```
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    python -m self_improving_diffusion.jax_backend.distributed_cli --out reports/jx04_distributed.json
```

Proves device count/placement never changes the result (every sample's RNG
key is derived from `(seed, request_id, sample_index)` only, never from rank
or device assignment) and sweeps collective (`psum`) latency by message
size. Findings, including what's not yet covered (real multi-host, rank
failure): `docs/learning/jx-04-distributed-correctness.md`.
