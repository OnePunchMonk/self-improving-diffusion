# Checkpoint ownership: logical/physical handles before a paged allocator

From the serving-ideas backlog (issue #5 comment, priority 3 in the build
order: "checkpoint ownership before a custom paged allocator"). Implemented
in `jax_backend/ownership.py`, tested in `tests/test_ownership.py`.

## Why this, and why not PagedAttention directly

LLM serving engines (vLLM's PagedAttention, SGLang's radix cache) manage KV
cache with logical-vs-physical handles, reference counts, and copy-on-write
sharing because the KV cache *grows* token by token and branches (beam
search, speculative decoding) share long common prefixes. A diffusion
latent doesn't grow -- it's replaced wholesale at each denoising step -- so
naively porting PagedAttention's block-table design has no clear target
here. What *does* transfer directly is the ownership discipline underneath
it: explicit reference counts, a reclaimable/live distinction, and
copy-on-write forking. This module is that layer, without committing to a
paged allocator until profiling (JX-03/JX-05, so far) actually shows a
fragmentation or retention bottleneck for a model this size.

## Design

`CheckpointOwnershipRegistry` is content-addressed: physical identity is
`TrajectoryState.state_digest()` (a real sha256 over the latents array plus
metadata, from JX-01), not an allocated slot number. Two consequences fall
out of that for free, without any extra bookkeeping:

- **Copy-on-write forking is exact sharing, not a lazy-copy flag.** `fork()`
  just increments a refcount and points a new logical `branch_id` at the
  same physical entry; no latents are copied until `diverge()` is called
  with genuinely different content.
- **Identical content from unrelated branches deduplicates automatically.**
  If two branches happen to reach bit-identical latents independently,
  `put()` for the second one finds the same physical id already present and
  increments its refcount instead of storing a duplicate --
  `test_identical_content_from_independent_branches_deduplicates` checks
  this directly.

The one rule this whole module exists to enforce, mirroring the charter's
"no silent eviction of live state" language from both the checkpoint-store
and improvement-controller sections: **`reclaim()` raises `OwnershipError`
if any branch still holds a reference**, rather than freeing storage a
still-live branch depends on.

## What this does not yet do

- No actual memory reclamation -- `reclaim()` returns the `TrajectoryState`
  Python object; nothing here frees device/host memory itself (JAX arrays
  are garbage-collected by Python's own refcounting once nothing holds them,
  which this registry's refcount is a *logical* mirror of, not a substitute
  for).
- No integration yet with `jax_backend/serving.py`'s chunk scheduler --
  branches created by JX-06's `best_of_n`/`adaptive` policies don't route
  through this registry today. Wiring that up is the natural next step once
  a workload actually creates a branch tree worth managing (JX-06's policies
  currently draw independent samples, not forked continuations).
- No tie-in to `jax.experimental.buffer_donation` yet, mentioned in the
  original idea as the mechanism a continuation must respect (a continuation
  must not donate a buffer another branch still references) -- this
  registry's refcounts are the bookkeeping that donation logic would need to
  consult, but nothing here calls into JAX's donation API.
