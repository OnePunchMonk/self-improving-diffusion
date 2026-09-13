# Overlap scheduling: prepare the next batch while the current one executes

From the serving-ideas backlog (issue #5), build-order priority 4: borrow
SGLang's overlap-scheduler principle. Implemented in `jax_backend/overlap.py`,
measured via `overlap_cli.py`. Raw output: `reports/overlap_scheduling.json`.

## What's real and what's labeled synthetic

The overlap *mechanism* is real: `run_overlapped` dispatches a batch via a
compiled JAX function and does **not** call `block_until_ready` before
starting host-side work for the next batch, relying on JAX's genuinely
asynchronous dispatch (a `jit`-compiled call returns as soon as the
computation is enqueued, not once it's finished). What's synthetic and
explicitly labeled as such: `simulated_host_prep_ms`, a `time.sleep()`
stand-in for real preprocessing (a text encoder call, tokenization) that
doesn't exist yet in this repo -- `TinyEpsilonModel` has no conditioning
input to encode.

## Results (20 batches, batch_size=4, chunk_steps=2)

| Simulated host prep | Serialized total | Overlapped total | Speedup |
|---|---|---|---|
| 0.0 ms | 0.1865s | 0.0752s | **2.48x** |
| 1.0 ms | 0.1018s | 0.1017s | 1.00x |
| 5.0 ms | 0.2037s | 0.2152s | 0.95x |
| 20.0 ms | 0.5936s | 0.5974s | 0.99x |

## Why the speedup disappears as simulated prep grows -- and why that's expected

This is the opposite of the naive prediction (more host work to hide should
mean *more* benefit from overlap), and it's a direct consequence of JX-02's
own earlier finding: this model's real device compute per chunk is
sub-millisecond. Overlap only saves wall-clock time by hiding host work
*behind* device work that's already running -- if the device finishes in
0.1-0.2ms and the host then sleeps for 1-20ms anyway, there's nothing left
for the "overlap" to hide the sleep behind; the host sleep dominates total
time regardless of whether it happens before, during, or after dispatch.

**The 2.48x speedup at zero simulated prep is real, but it isn't overlap
hiding preprocessing** -- it's `run_overlapped` calling `block_until_ready`
19 times instead of 20 (once per pair of consecutive batches, rather than
once per batch) and never fully draining the async dispatch queue between
iterations, i.e. it measures the cost of *synchronization points themselves*
at this tiny scale, not the cost of hidden host computation.

**The honest conclusion, consistent with the JX-03 GPU-too-small pattern**:
overlap scheduling as implemented here has no measurable benefit for a
model whose device compute is this cheap relative to any realistic
preprocessing cost. The mechanism is correctly implemented and the
measurement methodology is sound (both are tested in `tests/test_overlap.py`
and reused correctly in the CLI), but this specific model is the wrong
scale to demonstrate the technique's real-world value. Re-running this
exact script once a real text encoder (with genuine multi-millisecond
preprocessing cost) and a larger denoiser exist is the correct next test --
not assuming the mechanism doesn't work.

## What this still owes

- A real preprocessing step to replace `_simulate_host_prep`.
- A model whose per-chunk device compute is large enough for overlap to
  plausibly hide meaningful host work behind it.
- Profiler timeline evidence (the original idea asked for profiler
  timelines showing idle gaps) -- this uses wall-clock timing only, per the
  same Nsight Compute access gap noted in JX-03.
