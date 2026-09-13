# JX-07: bounded self-improvement (inference-policy scope)

This deliberately does **not** attempt the full JX-07 charter. It freezes
two of the three changing objects the charter names -- model parameters
(`TinyEpsilonModel`'s weights never change) and the execution policy
(scheduling from JX-05 is untouched) -- and updates only the smallest of the
three: the **inference policy**, specifically JX-06's best-of-N split
`n_samples`. A real training loop over model parameters is a materially
larger project than everything built through JX-06 and is not attempted
here; scoping down explicitly rather than quietly under-delivering.

Raw output: `reports/jx07_self_improvement.json`.

## The three disjoint seed ranges

| Range | Seeds | Used for | Reused across cycles? |
|---|---|---|---|
| `BEHAVIOR_LOG_SEEDS` | 201-210 | offline logged feedback, uniform-random behavior policy | yes, but never the source of a promotion decision by itself |
| `SELECTION_SEEDS` | 301-305 | on-policy promote/rollback comparison | yes, by design -- this is the repeated-decision set |
| `FINAL_SEEDS` | 401-410 | the one "did this sustain improvement" claim | no -- evaluated once, after all cycles |

All three ranges are disjoint from each other and from `FROZEN_SEEDS`
(JX-02's tuning set) and `HELD_OUT_SEEDS` (JX-06's held-out set) --
verified in `tests/test_rsi_controller.py`.

## Off-policy candidate proposal: why IPS, not a direct average

`collect_offline_logs` runs a declared **behavior policy** (uniform-random
choice of `n` in `{1, 2, 3, 4}`, propensity `0.25` recorded on every log
line) rather than the candidate policy itself. `ips_estimate` then applies
the standard inverse-propensity-scoring estimator:

```
V_hat(n) = (1/N) * sum_i [ 1{a_i == n} / propensity_i * quality_i ]
```

The point of doing this instead of just averaging the logged quality for
each action: it makes "this action was never tried" visible as a **zero
estimate**, not silently backfilled with the best observed action's value.
`test_ips_estimate_is_zero_for_an_action_never_drawn` checks this directly.
With only 10 behavior-log seeds and 4 candidate actions, some cycles do see
an action drawn zero times -- exactly the case IPS is supposed to handle
honestly rather than paper over.

## What actually happened, this run (2 cycles)

| Cycle | Candidate n | Baseline n (active) | Candidate quality (selection set) | Baseline quality | Promoted? |
|---|---|---|---|---|---|
| 0 | 3 | 3 | 0.4311 | 0.4311 | No (candidate == active policy) |
| 1 | 4 | 3 | 0.4647 | 0.4311 | **Yes** |

- **Final evaluation** (seeds never touched during search): promoted policy
  (`n=4`) scores **0.4816** mean quality on `FINAL_SEEDS`, versus **0.4426**
  for the original frozen baseline (`n=3`) evaluated on the same seeds.
  `sustained_improvement: true`.
- **Negative transfer check**: did any promoted candidate score *worse* than
  what it replaced, on the very selection set used to justify promoting it?
  No (`negative_transfer_detected: false`) -- promotion only fired when the
  selection-stage comparison already favored the candidate.

This is one honest positive result, not a general claim: `n=4` beating
`n=3` on this untrained tiny model's heuristic verifier says nothing about
whether `n=4` is the right choice once a trained model and a calibrated
verifier exist. Re-running this exact pipeline after JX-01's model is
replaced is the correct regression test, not treating this result as final.

## Cost accounting and the break-even question

Total offline pipeline cost across 2 cycles: **960 denoising steps**
(logging rollouts + selection-stage on-policy evaluation of both candidate
and baseline, every cycle). Final-evaluation cost is reported separately
since it's a one-time cost, not repeated per cycle.

**There is no compute break-even to report**, and that's stated explicitly
rather than obscured: every candidate `n` spends the *same* `TOTAL_STEP_BUDGET`
per request (best-of-N divides a fixed budget evenly across samples), so
promoting `n=4` over `n=3` does not change per-request serving cost at all.
The offline pipeline's 960-step cost must be justified purely by the
quality gain, amortized over however many requests the promoted policy ends
up serving -- not by a compute-savings argument. A future controller that
also varies total budget (not just its split) would make a real break-even
calculation possible; this one doesn't.

## What JX-07 still owes

- Model-parameter updates (the actual "training" object) -- not attempted.
  This milestone only ever touched which fixed-budget split to use.
- Independent task/quality checks, reward-hacking tests, diversity and
  protected-slice evaluation -- the heuristic verifier has none of these;
  IPS correctness is verified, but the *verifier itself* is not calibrated
  to anything external.
- More than 2 update cycles -- 2 is the minimum the charter asks for
  ("at least multiple"), not evidence of a stable long-run trend.
- A scenario where offline search cost and per-request serving cost trade
  off against each other, so a genuine break-even calculation becomes
  possible.
