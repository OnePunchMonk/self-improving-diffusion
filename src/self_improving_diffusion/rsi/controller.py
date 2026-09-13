"""JX-07: bounded self-improvement, scoped honestly to what's real here.

The charter separates three changing objects -- execution policy, inference
policy, model parameters -- and asks to freeze two while measuring the
third. This module freezes the model (`TinyEpsilonModel`'s weights never
change) and the execution policy (scheduling is untouched), and updates only
the **inference policy**: which best-of-N split (`n_samples`) to use within
JX-06's fixed step budget. That is deliberately the smallest of the three
knobs -- a real training loop over model parameters is a materially larger
scope than everything built so far and is explicitly not attempted here.

Three disjoint seed ranges, each used for exactly one purpose (JX-07's own
"disjoint tuning/selection/final evaluation" requirement, applied to the
inference-policy controller itself rather than deferred to a future model):

- ``BEHAVIOR_LOG_SEEDS`` (201-210): offline logged feedback, collected under
  an explicit uniform-random *behavior* policy with recorded propensities --
  never the candidate policy itself, so off-policy estimates aren't
  contaminated by the thing they're trying to evaluate.
- ``SELECTION_SEEDS`` (301-305): on-policy comparison of a proposed
  candidate against the frozen baseline, run fresh every cycle to decide
  promote/rollback. Reused across cycles by design -- it exists precisely
  for repeated promotion decisions.
- ``FINAL_SEEDS`` (401-410): evaluated exactly once, after every cycle has
  run, for the final "did this sustain improvement" claim. Not touched
  during candidate search, so repeated search can't quietly overfit to it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..eval.adaptive_inference import TOTAL_STEP_BUDGET, run_best_of_n

CANDIDATE_N_VALUES: tuple[int, ...] = (1, 2, 3, 4)
FROZEN_BASELINE_N = 3  # JX-06's best_of_n default -- the policy in production before any cycle runs

BEHAVIOR_LOG_SEEDS: tuple[int, ...] = tuple(range(201, 211))
SELECTION_SEEDS: tuple[int, ...] = tuple(range(301, 306))
FINAL_SEEDS: tuple[int, ...] = tuple(range(401, 411))

PROMOTION_TOLERANCE = 0.0  # candidate must be >= baseline on the selection set, no benefit of the doubt


@dataclass(frozen=True)
class LoggedRecord:
    seed: int
    action_n: int
    propensity: float
    quality: float
    cost_steps: int


def collect_offline_logs(cycle_index: int) -> list[LoggedRecord]:
    """Behavior policy: uniform-random choice of n, propensity recorded honestly.

    ``cycle_index`` seeds the behavior policy's own randomness (which action
    it happens to draw per logging seed) so successive cycles see different
    logged draws -- not different *seeds* (those stay BEHAVIOR_LOG_SEEDS,
    fixed) but different realized actions, mimicking production traffic
    varying run to run under a fixed exploration policy.
    """

    rng = np.random.default_rng(1000 + cycle_index)
    propensity = 1.0 / len(CANDIDATE_N_VALUES)
    logs = []
    for seed in BEHAVIOR_LOG_SEEDS:
        action_n = int(rng.choice(CANDIDATE_N_VALUES))
        outcome = run_best_of_n(seed, n=action_n)
        logs.append(
            LoggedRecord(
                seed=seed,
                action_n=action_n,
                propensity=propensity,
                quality=outcome.quality,
                cost_steps=outcome.total_cost_steps,
            )
        )
    return logs


def ips_estimate(logs: list[LoggedRecord], target_n: int) -> float:
    """Inverse-propensity-scored value of the deterministic policy 'always pick target_n'.

    Standard IPS: V_hat = (1/N) * sum_i [1{a_i == target_n} / propensity_i * quality_i].
    Untried actions (target_n never drawn by the behavior policy in this log)
    correctly contribute zero evidence rather than being silently assumed
    equal to the best observed action.
    """

    if not logs:
        raise ValueError("cannot estimate from an empty log")
    total = sum((record.quality / record.propensity) for record in logs if record.action_n == target_n)
    return total / len(logs)


def select_candidate(logs: list[LoggedRecord]) -> tuple[int, dict[int, float]]:
    estimates = {n: ips_estimate(logs, n) for n in CANDIDATE_N_VALUES}
    best_n = max(estimates, key=estimates.get)
    return best_n, estimates


def evaluate_on_policy(n: int, seeds: tuple[int, ...]) -> tuple[float, float]:
    outcomes = [run_best_of_n(seed, n=n) for seed in seeds]
    mean_quality = sum(o.quality for o in outcomes) / len(outcomes)
    mean_cost = sum(o.total_cost_steps for o in outcomes) / len(outcomes)
    return mean_quality, mean_cost


@dataclass
class CycleRecord:
    cycle_index: int
    logged_action_counts: dict[int, int]
    ips_estimates: dict[int, float]
    candidate_n: int
    baseline_n: int
    candidate_selection_quality: float
    baseline_selection_quality: float
    promoted: bool
    active_n_after_cycle: int
    offline_cost_steps: int


class PolicyRegistry:
    """Tracks the currently active inference policy and every promotion decision."""

    def __init__(self, initial_n: int = FROZEN_BASELINE_N) -> None:
        self._active_n = initial_n
        self.history: list[CycleRecord] = []

    @property
    def active_n(self) -> int:
        return self._active_n

    def apply(self, record: CycleRecord) -> None:
        self.history.append(record)
        self._active_n = record.active_n_after_cycle

    def rollback(self) -> int:
        """Revert to the policy active immediately before the last cycle."""

        if len(self.history) < 1:
            raise ValueError("no cycle to roll back")
        previous = self.history[-2].active_n_after_cycle if len(self.history) >= 2 else FROZEN_BASELINE_N
        self._active_n = previous
        return previous


def run_cycle(registry: PolicyRegistry, cycle_index: int) -> CycleRecord:
    logs = collect_offline_logs(cycle_index)
    offline_cost_steps = sum(record.cost_steps for record in logs)
    logged_action_counts = {n: sum(1 for r in logs if r.action_n == n) for n in CANDIDATE_N_VALUES}

    candidate_n, ips_estimates = select_candidate(logs)
    baseline_n = registry.active_n

    candidate_quality, candidate_cost = evaluate_on_policy(candidate_n, SELECTION_SEEDS)
    baseline_quality, baseline_cost = evaluate_on_policy(baseline_n, SELECTION_SEEDS)
    offline_cost_steps += int(candidate_cost * len(SELECTION_SEEDS) + baseline_cost * len(SELECTION_SEEDS))

    promoted = candidate_quality >= baseline_quality - PROMOTION_TOLERANCE and candidate_n != baseline_n
    active_after = candidate_n if promoted else baseline_n

    record = CycleRecord(
        cycle_index=cycle_index,
        logged_action_counts=logged_action_counts,
        ips_estimates=ips_estimates,
        candidate_n=candidate_n,
        baseline_n=baseline_n,
        candidate_selection_quality=candidate_quality,
        baseline_selection_quality=baseline_quality,
        promoted=promoted,
        active_n_after_cycle=active_after,
        offline_cost_steps=offline_cost_steps,
    )
    registry.apply(record)
    return record


@dataclass
class SelfImprovementReport:
    cycles: list[CycleRecord]
    final_active_n: int
    final_active_quality: float
    final_frozen_baseline_quality: float
    total_offline_cost_steps: int
    final_evaluation_cost_steps: float
    negative_transfer_detected: bool

    def as_dict(self) -> dict:
        return {
            "cycles": [asdict(c) for c in self.cycles],
            "final_active_n": self.final_active_n,
            "final_active_quality": self.final_active_quality,
            "final_frozen_baseline_quality": self.final_frozen_baseline_quality,
            "sustained_improvement": self.final_active_quality > self.final_frozen_baseline_quality,
            "total_offline_cost_steps": self.total_offline_cost_steps,
            "final_evaluation_cost_steps": self.final_evaluation_cost_steps,
            "negative_transfer_detected": self.negative_transfer_detected,
            "cost_note": (
                "every candidate n spends the same TOTAL_STEP_BUDGET per request "
                "(best_of_n divides it evenly), so there is no per-request compute "
                "cost difference between the promoted policy and the original "
                "frozen baseline -- the offline pipeline cost above must be "
                "justified by quality alone, amortized over however many "
                "requests the promoted policy serves, not by a compute break-even."
            ),
        }


def run_multiple_cycles(num_cycles: int = 2) -> SelfImprovementReport:
    registry = PolicyRegistry(initial_n=FROZEN_BASELINE_N)
    cycles = [run_cycle(registry, i) for i in range(num_cycles)]

    final_active_quality, final_active_cost = evaluate_on_policy(registry.active_n, FINAL_SEEDS)
    final_baseline_quality, final_baseline_cost = evaluate_on_policy(FROZEN_BASELINE_N, FINAL_SEEDS)

    # Negative transfer: any cycle's own selection-stage comparison regressed
    # after promotion, i.e. we promoted something that then measured worse
    # than what it replaced on the same selection set used to justify it.
    negative_transfer = any(
        c.promoted and c.candidate_selection_quality < c.baseline_selection_quality for c in cycles
    )

    total_offline_cost = sum(c.offline_cost_steps for c in cycles)
    final_eval_cost = (final_active_cost + final_baseline_cost) * len(FINAL_SEEDS)

    return SelfImprovementReport(
        cycles=cycles,
        final_active_n=registry.active_n,
        final_active_quality=final_active_quality,
        final_frozen_baseline_quality=final_baseline_quality,
        total_offline_cost_steps=total_offline_cost,
        final_evaluation_cost_steps=final_eval_cost,
        negative_transfer_detected=negative_transfer,
    )
