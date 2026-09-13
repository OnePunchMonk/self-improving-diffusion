# JX-04: distributed correctness before hybrid optimization

Measured on 4 simulated CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`
before importing JAX -- device count can't change after backend init). Raw
output: `reports/jx04_distributed.json`.

## What's implemented: request/data parallelism only

Per the charter's explicit ordering ("begin with request/data parallelism...
add intra-request sharding only where model size/latency warrants it"), this
milestone does exactly one thing: run independent samples across devices via
`jax.pmap`, with **no** intra-request tensor/sequence sharding. The tiny
JX-01 model doesn't warrant it yet -- adding sharding here would be
premature complexity with nothing to validate it against.

## Correctness property, measured

`run_data_parallel_samples` runs N samples across however many devices are
available (round-robin batching when N exceeds the device count) and
compares the result against a sequential, single-device ground truth using
**identical RNG keys**. Result: `matches_single_device: true` for all tested
configurations, including sample counts that don't divide evenly into the
device count (padding a partial batch with repeated keys, then discarding
the padding on collection).

This is the load-bearing design decision: every sample's key is
`fold_in(fold_in(PRNGKey(seed), hash(request_id)), sample_index)` --
a pure function of **logical identity**, never of which physical device
happened to execute it or how many devices existed at the time. That's what
makes "device count and placement don't change the result" provable rather
than just observed to hold by luck on one run.

Also verified: different `request_id` values with the same `seed` produce
different streams (no accidental key collision across requests sharing a
seed), and no two samples within the same request share a stream.

## Collective latency by message size

| Message size (bytes/device) | Elapsed (s) | Achieved GB/s |
|---|---|---|
| 4,096 | 0.000098 | 0.17 |
| 262,144 | 0.000136 | 7.69 |
| 4,194,304 | 0.000659 | 25.4 |

Achieved bandwidth scales up with message size, as expected: small messages
are latency-bound (fixed per-collective overhead dominates), larger ones
approach the CPU emulation path's bandwidth ceiling. **This measures the
single-host CPU collective-emulation path, not real interconnect** (NVLink,
InfiniBand, etc.) -- the absolute numbers are not meaningful for a GPU/TPU
mesh. What carries over is the methodology: sweep by message size, always
synchronize (`block_until_ready`) before recording, report devices used
alongside the numbers. Re-running this exact script against a real
multi-GPU or multi-host mesh gives numbers worth trusting; these do not.

## What JX-04 still owes

- A real multi-GPU or multi-host run (needs >= 2 physical devices; not
  available in this environment). Everything above validates correctness
  logic and methodology, not real network topology.
- Rank failure / cancellation / ordering-error exercises: the charter asks
  to "fail affected mesh work clearly rather than silently continuing with
  incomplete collectives." `DeviceCountError` covers the
  under-provisioned-device case (raised clearly rather than silently
  degrading to fewer devices); a genuine mid-collective rank failure isn't
  reproducible in JAX's CPU device simulation and needs real multi-host
  infrastructure to exercise honestly.
- Intra-request sharding (`shard_map`) is deliberately not implemented yet
  -- it has no justified use case until a model larger than JX-01's tiny MLP
  exists.
- Strong-vs-weak scaling curves: meaningful only once there's a model with
  enough compute per sample that per-sample wall time is measurable above
  noise (JX-02's own finding was that this tiny model's execute time is a
  fraction of a millisecond).
