# JX-05: dynamic distributed serving -- static batching vs chunk scheduling

Measured on CPU (single device), 30 requests, Poisson arrivals at 15 req/s,
8-step tiny model, via `python -m self_improving_diffusion.jax_backend.serving_cli`.
Raw output: `reports/jx05_serving.json`.

## What's compared, and how the timing is real

Both policies process the *same* arrival sequence and the *same* measured,
real per-chunk compute time -- `_measure_real_chunk_seconds` actually runs a
compiled, vmapped `ddpm_step` chunk on JAX and times it with
`block_until_ready`, rather than assuming a synthetic per-step cost. What's
simulated is the *scheduling/queueing logic* around that real cost (arrival
times, admission, batch composition), not the compute itself.

| Policy | p50 | p95 | p99 | mean queue delay | throughput | goodput (SLA=1s) |
|---|---|---|---|---|---|---|
| static batching | 0.0631s | 0.2124s | 0.5060s | 0.0908s | 12.91 req/s | 1.00 |
| chunk scheduling | 0.00016s | 0.00018s | 0.00020s | 0.0000010s | 12.91 req/s | 1.00 |

## Interpretation

**Throughput is identical between the two policies** -- both process the
same 30 requests at the same arrival rate, so neither is compute-bound here;
this comparison is entirely about *latency distribution*, not capacity.

**Static batching's queue delay is the batch-fill wait.** A request sits in
`pending` until 3 more requests arrive (batch size 4) or the trickle of
arrivals happens to line up; at 15 req/s that's tens to hundreds of
milliseconds of pure waiting before any compute starts, which shows up
directly as p95/p99 latency inflation. This is the charter's own prediction
("Measure the tradeoff between long scans... and short chunks") made
concrete: static batching trades latency for simpler scheduling.

**Chunk scheduling admits into a free slot immediately** -- with
`max_active_slots=4` and requests arriving well below the compute rate,
there's almost always a free slot, so `mean_queue_delay_s` is effectively
zero and total latency is dominated by actual chunk compute (a handful of
tiny model steps), not waiting.

**Both hit 100% goodput at a 1s SLA** -- this arrival rate (15 req/s) isn't
close to saturating a model this cheap. The gap between the two policies
would need a much higher arrival rate, a slower model, or a tighter SLA to
show up as a goodput difference rather than only a latency-percentile one;
this is a case to re-run once JX-01's model is replaced with something
compute-heavier.

## Compiled-variant cache: bounded, and mostly hit

- Static batching: 2 compiled variants (the steady batch-size-4 shape, plus
  one smaller shape for the final partial batch), 6 hits / 2 misses.
- Chunk scheduling: 1 compiled variant (slots are always padded to
  `max_active_slots`, so shape never changes regardless of how many are
  actually active), 118 hits / 1 miss.

This is the intended effect of shape-aware admission: chunk scheduling's
fixed-size padded slots mean *one* compile signature serves the entire run
regardless of arrival pattern, while static batching's variable final-batch
size is the one place a second compile signature is unavoidable without
further padding logic. `CompiledVariantCacheError` (tested in
`tests/test_serving.py`) fires instead of silently growing past
`max_variants` if a workload produced more distinct shapes than that.

## What JX-05 still owes

- A workload where the two policies' *goodput* actually diverges (this run's
  arrival rate is too low relative to the tiny model's speed).
- Real GPU/multi-host serving -- this is single-device CPU only.
- Cancellation and OOM handling are not yet exercised; `max_active_slots`
  here is an arbitrary constant, not derived from an actual memory budget.
- A closed-loop admission policy that adapts chunk size or batch size to
  observed load, rather than the two fixed policies compared here.
