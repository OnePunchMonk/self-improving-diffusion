# Real text conditioning

Every milestone through JX-07 accepted `GenerationSpec.prompt` and folded it
into program/sample digests, but the model itself never saw it -- denoising
was conditioned on the timestep only. This closes that gap.

## What's real, and what isn't

`text_encoder.py`'s `TinyTextEncoder` is a genuine (if architecturally
minimal) text encoder: whitespace tokenization, deterministic hashing of
each token into a fixed `VOCAB_SIZE=4096` vocabulary via sha256 (not
Python's salted `hash()`, which differs across processes), an embedding
table lookup, and mean pooling into a `COND_DIM=16` vector. It is **not**
CLIP -- there is no training signal, so "a red square" and "a crimson
square" have no reason to encode similarly beyond accidentally sharing a
hashed token. What's real is the computation: an actual embedding lookup
and a real pooling reduction feed into the model, not a stand-in digest.

`encode_prompt` is a **frozen, module-level singleton**, seeded once
(`FROZEN_TEXT_ENCODER_SEED = 4242`) and never re-initialized per backend
instance or per sampling seed. `test_frozen_encoder_params_are_not_affected_by_sampling_seed`
checks this directly -- conditioning and sampling are independent knobs,
matching how a real frozen text encoder sits in front of a diffusion model
whose own seed varies per request.

## Design: additive, not a breaking rewrite

`TinyEpsilonModel.apply`, `ddpm_step`, and `run_trajectory` all gained an
optional `cond` parameter defaulting to `None` (a zero/null embedding --
classifier-free guidance's own convention for "unconditioned", not a
special case invented for this repo). Every pre-existing call site
(`distributed.py`, `serving.py`, `benchmark.py`, `overlap.py`) keeps working
completely unchanged, still exercising the unconditioned path -- appropriate
for those modules, whose job is measuring scheduling/distribution
mechanics, not conditioning fidelity. Only `JaxDenoiserBackend.sample()`
(and the CLI's independent `_verify_exact_resume` check) were changed to
call `encode_prompt(spec.prompt)` and thread a real embedding through.

## What changed as a side effect, and why it's fine

`TinyEpsilonModel.init`'s input dimension grew from `LATENT_DIM + 1` to
`LATENT_DIM + 1 + COND_DIM`, so every existing model's random weight
initialization now draws different numbers from the same PRNG key than it
did before this change -- every downstream milestone's actual measured
quality/cost numbers shift slightly (compare JX-06's best_of_n mean quality:
0.4284 before this change, 0.4443 after, on the same held-out seeds). No
test anywhere asserts a fixed golden numeric value; every test compares
determinism, equivalence, or ordering, so nothing broke -- but anyone
reading old `reports/*.json` files should know the underlying model weights
are no longer bit-identical to what produced them.

## What's verified

- `encode_prompt` is deterministic (same prompt -> same embedding, always).
- Different prompts encode differently, and -- the actual point -- produce
  different generations at the same seed
  (`test_different_prompts_produce_different_generations_at_the_same_seed`).
- Exact resume still holds bit-for-bit with real conditioning threaded
  through (verified by the JX-01 CLI's own resume check, now conditioned).
- The full JX-02 NumPy cross-check, JX-04 distributed correctness, JX-05
  serving, JX-06 adaptive inference, and JX-07 self-improvement suites all
  still pass unmodified against the new model shape.

## What this still owes

- `JaxVerifierBackend`'s "alignment" score still doesn't compare the
  decoded latent against the conditioning embedding it's supposedly aligned
  to -- it's untouched by this change and remains a placeholder.
- No real semantic structure (a trained CLIP-style encoder, or even a
  learned projection) -- two semantically similar prompts have no reason to
  encode similarly here.
- `jax_backend/distributed.py` and `serving.py`'s simulated workloads still
  don't carry per-request prompts; conditioning exists in the model now but
  isn't exercised by those modules' synthetic benchmarks.
