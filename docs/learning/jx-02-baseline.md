# JX-02: trustworthy single-device baseline -- learning artifact

Measured on: CPU (`cpu:0`), JAX 0.11.1, the tiny JX-01 model
(8x8 latent, 24 steps), via `python -m
self_improving_diffusion.jax_backend.benchmark_cli`.

## Bottleneck classification, from measured evidence

| Phase | Cold (s) | Warm (s) | Classification |
|---|---|---|---|
| compile | 0.0652 | 0.0 (cached) | one-time XLA lowering + compilation cost |
| dispatch | 0.00016 | 0.000008 | host-side enqueue of the compiled executable |
| execute | 0.00020 | 0.00014 | device-side compute, synchronized via `block_until_ready` |
| transfer | 0.00001 | 0.000005 | device -> host array copy |
| decode | 0.00007 | 0.00003 | host-side normalize-to-uint8 |
| verify | 0.0679 | 0.0001 | first call to `jnp.var`/`jnp.mean` also pays one-time XLA compilation |

**Cold-call bottleneck: compilation, twice over.** The obvious `compile_s`
line is one hit, but the "verifier" phase's first call is *also* dominated by
JIT overhead the harness hadn't isolated -- `jnp.var`/`jnp.mean` inside
`JaxVerifierBackend.score` are each their own XLA computations that get
compiled on first use. Total cold-call time is ~130ms; ~100ms of that is
compilation happening in two different places, not model execution.

**Warm-call bottleneck: execute (device compute), not dispatch or
transfer.** Once both compiled executables are cached, `execute_s` (0.14ms)
dominates `dispatch_s` (0.008ms) and `transfer_s` (0.005ms) by roughly an
order of magnitude. For this tiny model, that means the harness itself
(Python dispatch overhead, host<->device transfer) is not the limiting
factor -- the actual `lax.scan` compute is. This will very likely invert once
a realistic-sized DiT block replaces the 3-layer MLP; JX-03's kernel work
should re-measure this before assuming compute dominance carries over.

**Compile-overhead ratio (cold total / warm total): ~465x** for this model
size. That ratio is a property of *how small the model is*, not evidence that
compilation is expensive in absolute terms -- 65ms of compilation is fixed
cost that amortizes away entirely under any serving workload issuing more
than a handful of requests per compiled shape. It matters here only because
so little of the total time is device compute.

## Numerical cross-check against a reference implementation

`compare_against_numpy_reference` runs the identical model weights, noise
schedule, and per-step RNG draws (`jax.random.fold_in`) through both the
compiled JAX path and an eager, unrolled, pure-NumPy reimplementation
(`jax_backend/numpy_reference.py`). Max absolute difference: `3.16e-06`,
well under the `1e-4` tolerance -- consistent with float32 accumulation
differences between XLA's fused GELU/matmul lowering and NumPy's, not a
correctness bug in either path.

This is the "relevant reference implementation at matched outputs/
configuration" JX-02 asks for. It is deliberately *not* a performance
comparison (eager NumPy on 64-dim vectors will trivially lose or win
depending on call overhead at this scale) -- it exists to give the compiled
path an independent numerical witness before any GPU-side optimization work
begins on top of it.

## What this baseline does not yet cover

- Single GPU path and CPU-vs-GPU comparison (needs GPU access; JX-02 in the
  charter lists this before JX-03's kernel-level work).
- A realistic-sized model. Every timing above will shift once compute scales
  past what a Python-level `for` loop over reverse-process steps in
  `JaxDenoiserBackend.sample` can build history for cheaply -- that loop is
  fine for an 8x8 latent's checkpoint bookkeeping but is not the shape of a
  production sampling loop.
- Device memory (`peak_rss_bytes` here is host process RSS, not JAX device
  allocator stats -- `jax.devices()[0].memory_stats()` returns `None` on the
  CPU backend used for this measurement).
