# JX-03: GPU internals through measured kernels -- learning artifact

Measured on a Modal-provisioned Tesla T4 (15360 MiB, driver 580.95.05),
JAX 0.10.2+cuda12, via `benchmarks/modal/jx03_gpu_internals.py`. Raw output:
`reports/jx03_gpu_internals.json`.

**Nsight Compute was not available.** The Modal container doesn't grant the
privileged perf-counter access `ncu` needs, so this uses wall-clock timing
and derived roofline arithmetic (bytes/FLOPs, arithmetic intensity) instead
of kernel-level profiler traces. That is a real gap against JX-03's ask for
"profiler evidence" for each exercise -- noted here rather than silently
worked around.

## Finding 1 (negative result, kept per the charter): the tiny model is slower on GPU

| | CPU (JX-02 baseline) | GPU (T4) |
|---|---|---|
| cold total | 0.134s | 0.981s |
| warm total | 0.000287s | 0.00329s |

The warm-path GPU run is **~11x slower** than CPU for this exact model and
config. This is not a bug -- it's the expected outcome for a model this
small: an 8x8-latent, 3-layer MLP has essentially no arithmetic for a GPU to
parallelize against, so every phase is dominated by fixed per-launch
overhead (kernel launch, host<->device synchronization, PCIe round trip)
that a CPU simply doesn't pay. Cold-call compilation is also ~9x slower on
GPU (0.59s vs 0.065s) -- CUDA backend JIT lowering has more fixed cost than
the CPU backend's.

**Implication for JX-02/JX-04 going forward:** GPU work only starts paying
off once the model is large enough that compute time exceeds this launch
overhead floor. The charter's own ordering (tiny model first, "one realistic
open-weight image model" second) exists for exactly this reason; this
result is the measured evidence for why, not an assumption.

## Finding 2: XLA already fuses the naive elementwise chain

Hand-fusing `tanh(2x+1)*exp(-x^2)` into one expression gave a 1.002x
"speedup" over writing it as six separate ops -- i.e., no measurable
difference. XLA's compiler fuses the unfused chain into the same kernel
under `jax.jit` regardless of how the Python is structured. Both versions
hit ~242 GB/s achieved bandwidth on a T4 (spec peak ~320 GB/s memory
bandwidth), consistent with this being a genuinely memory-bound op reaching
a reasonable fraction of peak.

**This directly informs future kernel work**: writing a custom Pallas
fusion for elementwise chains like this would not beat what XLA already
does. A real elementwise Pallas exercise needs an op XLA's fusion heuristics
don't already handle well (e.g., an op with a data-dependent branch, or one
spanning a shape XLA tiles poorly) -- not a plain chain of pointwise ops.

## Finding 3: GEMM achieved throughput and a real hardware-capability boundary

| Size | dtype | Achieved TFLOPs | Max rel err vs NumPy fp32 |
|---|---|---|---|
| 1024 | fp32 | 2.52 | 5.5e-07 |
| 4096 | fp32 | 4.88 | 2.8e-06 |
| 4096 | bf16 | 4.89 | 2.9e-03 |

Two things worth flagging, both against the naive expectation:

- **fp32 and bf16 achieved the same TFLOPs at 4096.** A T4 is a Turing-generation
  GPU: it has fp16 tensor cores but *not* bf16 tensor cores (those arrived
  with Ampere). So `bfloat16` matmul on this hardware doesn't get a tensor-core
  path and falls back to a similar throughput as fp32 -- this is a hardware
  capability boundary, not a bug in the benchmark. Any future claim like
  "bf16 gives a free speedup" needs to state the GPU generation it was
  measured on.
- **~4.9 TFLOPs is well under this T4's rated ~8.1 TFLOPs fp32 peak** (and far
  under its ~65 TFLOPs fp16-tensor-core peak). The gap is expected: this
  measurement includes Python-loop dispatch overhead per call (10 reps timed
  including re-dispatch, not a single steady-state device-side loop), so it
  is a lower bound on achievable throughput, not a tuned peak measurement.
  A tighter measurement would use `jax.block_until_ready` only once per
  batch of chained calls or an XLA-level benchmarking loop.
- bf16's larger relative error (2.9e-03 vs ~1e-6 for fp32) against the fp32
  NumPy oracle is expected precision loss, not a correctness bug -- consistent
  with bf16's ~3 decimal digits of mantissa precision.

## What JX-03 still owes

- Real Nsight Compute kernel traces (occupancy, memory hierarchy, tensor
  core utilization) -- blocked on container privilege, not attempted here.
- The reduction/normalization, attention, and async-overlap exercises from
  the roadmap's 5-exercise list -- only elementwise fusion and tiled GEMM
  are covered so far.
- A steady-state (non-Python-loop-dominated) GEMM measurement to get a
  tighter achieved-vs-peak comparison.
- Re-running once JX-01's tiny model is replaced with a realistic-sized DiT
  block, where GPU parallelism should actually pay off (Finding 1 predicts
  it won't for the current model).
