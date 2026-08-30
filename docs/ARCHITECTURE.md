# Architecture: an inference framework, not a training pipeline

Reframing from `DESIGN.md`: this isn't a training loop with a serving afterthought — it's an **inference framework for image diffusion models**, structurally analogous to SGLang for LLMs, where recursive self-improvement (RSI) is a first-class scheduler concern rather than a separate offline job.

## Why "framework," and why the SGLang analogy

SGLang's core bet: serving efficiency comes from exploiting *structure* in how requests relate to each other (shared prefixes → RadixAttention; overlapping lifetimes → continuous batching; multi-call programs → a DSL for structured generation). The framework, not the model, owns that structure.

Image diffusion serving has an analogous structure nobody exploits well yet:
- Many requests share a **denoising trajectory shape** even with different prompts/seeds — same number of steps, same scheduler, often similar guidance scale.
- Consecutive denoising steps produce **similar intermediate activations** (this is what TeaCache/MagCache exploit for single requests) — but across a *batch* of concurrent requests this becomes a cache-sharing problem, not just a per-request approximation.
- Verifier/critic calls, localized re-sampling, and self-play data logging are **repeated sub-programs** across almost every request — exactly the shape SGLang's DSL was built to make cheap and composable for LLM programs.

So the mapping:

| SGLang (LLM serving) | This framework (image diffusion serving) |
|---|---|
| RadixAttention (KV cache reuse by token prefix) | **Step-cache reuse by trajectory similarity** — a radix-like structure keyed on (scheduler config, step index, activation fingerprint) instead of token prefix; TeaCache/MagCache-style skip decisions promoted to a batch-level cache, not per-request heuristics |
| Continuous batching / overlapped scheduler | **Step-level continuous batching** — requests at different denoising steps and different image sizes share GPU batches; a request that finishes a step doesn't block others, and a slot frees the instant an image completes rather than waiting for the whole batch |
| Paged KV cache | **Paged latent/activation memory** — same problem (fragmentation from variable-length lifetimes), applied to per-step activation tensors and intermediate latents instead of token KV |
| Structured generation DSL | **A DSL for generation programs**, not just a single denoise call: "generate → critic-score → if region flagged, localized-refine that region → emit self-play pair" is one program, not four separate API calls glued together by client code |
| (nothing — LLM serving is generally not self-improving at the framework level) | **RSI as a scheduler hook, not a cron job.** Every served request that passes through the critic can optionally emit a training signal; the scheduler batches these into self-play preference pairs and triggers async fine-tune/distill rounds without an operator loop |

The last row is the actual novelty claim: most "self-improving" systems (see `DESIGN.md` survey — SPIN-Diffusion, VideoDPO, etc.) are training scripts that happen to consume inference-time samples. Here, self-improvement is wired into the serving path itself — the framework's request lifecycle *is* the data-collection and update-triggering mechanism. Operating the system in production is the RSI loop; there is no separate "now let's go collect data and retrain" phase.

## Scope discipline: image only

Deliberately **not** doing video or general vision tasks in the core engine. Reasons:
1. SGLang won by being excellent at one structural pattern (LLM token generation) before generalizing. Chasing image + video + general vision at once means building none of the three cache/batching primitives well.
2. Video's cost structure (frame count × step count × critic calls) is a superset of image's — solving image-level step-caching and localized TTS correctly is a prerequisite, not a parallel track. `DESIGN.md` Section 4 already noted "video by frame-chunking, don't start with video" — the framework reframing makes that a hard scope boundary, not just a build-order preference.
3. Vision-understanding tasks (classification, detection) don't have the iterative denoising-trajectory structure the whole caching/batching design leans on — they're a different framework.

If video support is wanted later, it's a second engine that reuses `verifiers/` and the RSI subsystem, built after the image engine's cache/batch primitives are proven — not a flag on this one.

## Component layout (mapped to SGLang's decomposition)

```
src/
  engine/      request scheduler, step-level continuous batching, paged latent memory,
               step-cache (radix-like structure keyed on trajectory similarity)
  runtime/     denoiser backends (the actual model forward passes: DiT/UNet step execution,
               swappable — this is the "model worker" layer, deliberately thin)
  api/         the generation-program DSL: compose denoise + critic + localized-refine +
               self-play-emit into one served program (SGLang's structured-gen analog)
  verifiers/   critic/reward models — real/fake discriminator + region localization
               (unchanged from DESIGN.md Section 4)
  rsi/         the self-improvement subsystem, triggered by engine/ request lifecycle hooks:
    search/    localized best-of-N / iterative refinement (invoked via api/ DSL, not standalone)
    data/      self-play preference pair construction from served-traffic samples
    training/  async DPO/self-play fine-tune + distillation jobs, triggered by rsi/ thresholds
               (e.g. N pairs collected, or held-out reward plateau) rather than a human-run script
  eval/        held-out reward tracking, critic-human agreement, cache-hit-rate /
               batching-efficiency metrics (the framework's own serving metrics, SGLang-style)
```

`rsi/search`, `rsi/data`, `rsi/training` are nested under `rsi/` (not siblings of `engine/`) specifically to signal: these are subsystems the engine calls into during normal serving, not an external pipeline that reads logs after the fact.

## Open engineering questions (framework-specific, beyond the RSI risks in DESIGN.md §5)
- What's the right cache key for step-cache reuse across *different* prompts/seeds? Token-prefix identity is exact for SGLang; activation similarity for diffusion is fuzzy — need a similarity threshold + eviction policy analogous to RadixAttention's LRU, but tolerant of approximate matches (correctness risk: caching wrong = visibly wrong pixels, not just a wrong token).
- Step-level continuous batching requires that different requests' models be batchable despite different resolutions/step counts — likely needs bucketing (SGLang doesn't have to bucket by "sequence shape" the way variable image resolution forces here).
- Where does the RSI trigger threshold live — in `engine/` (scheduler decides "enough signal, kick off training") or `rsi/` (pull-based, polls the data store)? Push-based keeps the "serving is the loop" story honest; pull-based is operationally simpler. Default to push-based to match the framing, revisit if it complicates the scheduler.
