"""JX-06: budgeted adaptive inference.

Compares four policies at the *same* total denoising-step budget, on a
held-out seed set disjoint from the seeds `QualityComputeProtocol` already
used (JX-07's "disjoint tuning/selection/final evaluation" principle,
applied a milestone early since nothing about it requires waiting):

- ``single_sample``: one sample using a small fraction of the budget.
- ``more_steps``: one sample using the *entire* budget -- isolates whether
  more denoising steps alone beats the single-sample baseline.
- ``best_of_n``: split the budget across N independent samples, verifier
  picks the best. Uses the same `ProgramBuilder`/`ProgramExecutor` contract
  as JX-01, so `choose(by="quality")` is exercised for real.
- ``adaptive``: a bounded accept/refine policy -- draw one probe sample; if
  its verifier score clears a threshold, accept it; otherwise draw exactly
  one more sample with the remaining budget and keep whichever scores
  higher. At most two samples are ever drawn (a bounded action set, not
  open-ended search) -- JX-06 explicitly asks to start here before any
  particle/tree search.

Cost accounting charges every sample actually executed, including ones the
policy discards -- a rejected sample in `best_of_n` or the probe sample in
`adaptive` is not free just because it wasn't selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax

from ..executor import ProgramExecutor
from ..jax_backend.backend import JaxDenoiserBackend, JaxVerifierBackend
from ..program import ProgramBuilder
from ..types import Budget, GenerationSpec, ModelRef
from .quality_compute import FROZEN_MODEL, FROZEN_VERIFIER

# Disjoint from FROZEN_SEEDS in quality_compute.py -- those are the "tuning"
# seeds already used to build the step-budget frontier; these are reserved
# for judging the policies below and must never be reused for tuning.
HELD_OUT_SEEDS: tuple[int, ...] = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110)

TOTAL_STEP_BUDGET = 24
PROBE_STEPS = 8
ADAPTIVE_ACCEPT_THRESHOLD = 0.3  # chosen ahead of the held-out run, not fit to it


@dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    seed: int
    quality: float
    total_cost_steps: int
    samples_drawn: int
    verifier_calls: int
    decode_calls: int


def _execute(seed: int, steps: int, num_samples: int) -> tuple[float, list]:
    """Run a `sample x num_samples -> score -> choose` program; return (quality, scores)."""

    spec = GenerationSpec(
        prompt="JX-06 budgeted adaptive inference probe",
        model=FROZEN_MODEL,
        seed=seed,
        width=64,
        height=64,
        steps=steps,
    )
    budget = Budget(max_samples=num_samples, max_total_steps=steps * num_samples)
    builder = ProgramBuilder(spec, budget)
    for _ in range(num_samples):
        builder = builder.sample()
    program = builder.score(FROZEN_VERIFIER).choose(by="quality").emit_trace().build()

    backend = JaxDenoiserBackend(seed=seed)
    verifier = JaxVerifierBackend(backend)
    trace = ProgramExecutor(backend, verifier).execute(program)
    jax.block_until_ready(backend.final_latents(trace.selected_sample_id))

    selected_score = next(s for s in trace.scores if s.sample_id == trace.selected_sample_id)
    return selected_score.quality, trace.scores


def run_single_sample(seed: int) -> PolicyOutcome:
    quality, scores = _execute(seed, steps=PROBE_STEPS, num_samples=1)
    return PolicyOutcome("single_sample", seed, quality, PROBE_STEPS, 1, len(scores), 1)


def run_more_steps(seed: int) -> PolicyOutcome:
    quality, scores = _execute(seed, steps=TOTAL_STEP_BUDGET, num_samples=1)
    return PolicyOutcome("more_steps", seed, quality, TOTAL_STEP_BUDGET, 1, len(scores), 1)


def run_best_of_n(seed: int, n: int = 3) -> PolicyOutcome:
    steps_per_sample = TOTAL_STEP_BUDGET // n
    quality, scores = _execute(seed, steps=steps_per_sample, num_samples=n)
    return PolicyOutcome("best_of_n", seed, quality, steps_per_sample * n, n, len(scores), 1)


def run_adaptive(seed: int) -> PolicyOutcome:
    """Bounded accept/refine: at most one probe, at most one refinement."""

    probe_quality, probe_scores = _execute(seed, steps=PROBE_STEPS, num_samples=1)
    if probe_quality >= ADAPTIVE_ACCEPT_THRESHOLD:
        return PolicyOutcome("adaptive", seed, probe_quality, PROBE_STEPS, 1, len(probe_scores), 1)

    remaining_steps = TOTAL_STEP_BUDGET - PROBE_STEPS
    refine_seed = seed + 1_000_000  # a distinct, deterministic refinement stream, never reused as a probe seed
    refine_quality, refine_scores = _execute(refine_seed, steps=remaining_steps, num_samples=1)

    best_quality = max(probe_quality, refine_quality)
    return PolicyOutcome(
        "adaptive",
        seed,
        best_quality,
        PROBE_STEPS + remaining_steps,
        2,
        len(probe_scores) + len(refine_scores),
        1,
    )


POLICIES = {
    "single_sample": run_single_sample,
    "more_steps": run_more_steps,
    "best_of_n": run_best_of_n,
    "adaptive": run_adaptive,
}


def run_all_policies(seeds: tuple[int, ...] = HELD_OUT_SEEDS) -> list[PolicyOutcome]:
    outcomes = []
    for policy_fn in POLICIES.values():
        for seed in seeds:
            outcomes.append(policy_fn(seed))
    return outcomes


def summarize(outcomes: list[PolicyOutcome]) -> list[dict]:
    by_policy: dict[str, list[PolicyOutcome]] = {}
    for outcome in outcomes:
        by_policy.setdefault(outcome.policy, []).append(outcome)

    summary = []
    for policy, group in by_policy.items():
        summary.append(
            {
                "policy": policy,
                "n": len(group),
                "mean_quality": sum(o.quality for o in group) / len(group),
                "mean_total_cost_steps": sum(o.total_cost_steps for o in group) / len(group),
                "mean_samples_drawn": sum(o.samples_drawn for o in group) / len(group),
            }
        )
    return summary
