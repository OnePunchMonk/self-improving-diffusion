# JX-06: budgeted adaptive inference

Four policies, same 24-step total compute budget, evaluated on
`HELD_OUT_SEEDS` (101-110) -- disjoint from `FROZEN_SEEDS` (1-5) already
used to build JX-02's step-budget frontier, per the "disjoint
tuning/selection/final evaluation" principle. Raw output:
`reports/jx06_adaptive_inference.json`.

| Policy | Mean quality | Mean cost (steps) | Mean samples drawn |
|---|---|---|---|
| best_of_n (N=3, 8 steps each) | 0.4284 | 24.0 | 3.00 |
| single_sample (8 steps) | 0.3716 | 8.0 | 1.00 |
| adaptive (8-step probe, threshold 0.3) | 0.3716 | 8.0 | 1.00 |
| more_steps (24 steps, 1 sample) | 0.1583 | 24.0 | 1.00 |

## Acceptance check: does the frontier beat the simple baseline?

**Yes, for best_of_n against more_steps at equal cost.** Both spend the full
24-step budget; `best_of_n` (split 3x8) reaches 0.4284 mean quality,
`more_steps` (1x24) reaches 0.1583. At matched compute, splitting the budget
across independent samples and picking the verifier's favorite clearly beats
spending it all on one longer trajectory, on this model.

**But `best_of_n` costs 3x what `single_sample` costs, and `single_sample`
already asymptotically matches it (0.37 vs 0.43) for a third of the
compute.** The honest framing per JX-02's own frontier (`docs/learning/jx-02-baseline.md`
and its underlying data) is: quality-per-step is not monotonically
increasing here at all -- `more_steps`'s 24-step single sample is *worse*
than an 8-step single sample. That's a property of the model, not the
policy: `TinyEpsilonModel`'s weights are freshly initialized, never trained,
so its noise predictions are not calibrated to actually reduce variance over
more reverse-process steps -- the heuristic "quality" score (inverse latent
variance) tends to degrade with more untrained denoising, not improve. This
frontier will look qualitatively different once JX-01's model is replaced
with something trained; re-running this exact comparison then is the right
regression test.

## Rejection condition, published rather than hidden: `adaptive` never diverges from `single_sample` here

The `adaptive` policy is defined to probe at 8 steps, accept if quality
`>= 0.3`, otherwise draw one more sample with the remaining 16 steps and
keep the best. Across all 10 held-out seeds, every probe's quality already
cleared 0.3 (range: 0.306-0.486) -- so the refinement branch was **never
exercised**, and `adaptive` is byte-for-byte identical to `single_sample` in
this run (same seeds, same threshold never triggers a second sample).

This is not a bug -- `0.3` was chosen ahead of time, not fit to this run
(per JX-07's ban on tuning acceptance criteria to manufacture progress) --
but it does mean **this run provides no evidence either way about whether
the adaptive policy's refinement branch is useful.** A meaningful test needs
either a threshold recalibrated against the *tuning* seeds (never the
held-out set used here) or a harder held-out distribution where probes
sometimes score below threshold. Both are open work, not silently assumed
to be fine.

## What JX-06 still owes

- A threshold picked by looking at `FROZEN_SEEDS` (tuning) data, then
  evaluated fresh on `HELD_OUT_SEEDS` -- this run's threshold was a
  reasonable guess, not derived from tuning data, which is why it didn't
  end up exercising the refine branch.
- A trained model, so "quality" reflects something other than an untrained
  network's incidental variance statistics.
- Verifier calibration: the charter is explicit that "a high intermediate
  score does not establish final quality" -- this heuristic verifier has no
  calibration story at all yet.
- Intermediate particle selection / tree branching are deliberately not
  attempted, per the charter's "only then" ordering after simple
  accept/another-sample/one-bounded-refinement is understood.
