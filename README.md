# self-improving-diffusion

An inference framework for **image** diffusion models, structurally analogous to SGLang for LLMs — but where recursive self-improvement (RSI) is a scheduler-level concern built into the serving path, not a separate offline training pipeline.

Video and general vision tasks are explicitly out of scope for the core engine (see `docs/ARCHITECTURE.md`).

- `docs/DESIGN.md` — literature review: test-time scaling, self-play preference optimization (SPIN-Diffusion, Diffusion-DPO, VideoDPO), reward modeling, known failure modes.
- `docs/ARCHITECTURE.md` — the SGLang-shaped framework design: step-cache (RadixAttention analog), step-level continuous batching, a generation-program DSL, and RSI wired into the request lifecycle.
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
