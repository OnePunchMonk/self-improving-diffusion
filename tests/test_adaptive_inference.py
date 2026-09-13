"""JX-06: budgeted policies must stay within their declared cost, and results must be reproducible."""

from self_improving_diffusion.eval.adaptive_inference import (
    HELD_OUT_SEEDS,
    TOTAL_STEP_BUDGET,
    run_adaptive,
    run_best_of_n,
    run_more_steps,
    run_single_sample,
    run_all_policies,
    summarize,
)
from self_improving_diffusion.eval.quality_compute import FROZEN_SEEDS


def test_held_out_seeds_are_disjoint_from_tuning_seeds():
    assert set(HELD_OUT_SEEDS).isdisjoint(set(FROZEN_SEEDS))


def test_more_steps_uses_the_full_budget_and_single_sample_uses_less():
    single = run_single_sample(seed=101)
    more = run_more_steps(seed=101)
    assert more.total_cost_steps == TOTAL_STEP_BUDGET
    assert single.total_cost_steps < more.total_cost_steps


def test_best_of_n_charges_for_every_sample_including_rejected_ones():
    outcome = run_best_of_n(seed=102, n=3)
    assert outcome.samples_drawn == 3
    assert outcome.verifier_calls == 3
    assert outcome.total_cost_steps <= TOTAL_STEP_BUDGET


def test_adaptive_draws_at_most_two_samples():
    for seed in HELD_OUT_SEEDS:
        outcome = run_adaptive(seed)
        assert outcome.samples_drawn in (1, 2)
        assert outcome.total_cost_steps <= TOTAL_STEP_BUDGET


def test_adaptive_is_deterministic_for_a_fixed_seed():
    first = run_adaptive(seed=103)
    second = run_adaptive(seed=103)
    assert first.quality == second.quality
    assert first.samples_drawn == second.samples_drawn


def test_run_all_policies_covers_every_policy_and_seed():
    outcomes = run_all_policies(seeds=(101, 102))
    seen = {(o.policy, o.seed) for o in outcomes}
    assert seen == {
        (policy, seed)
        for policy in ("single_sample", "more_steps", "best_of_n", "adaptive")
        for seed in (101, 102)
    }


def test_summarize_produces_one_row_per_policy():
    outcomes = run_all_policies(seeds=(101, 102, 103))
    summary = summarize(outcomes)
    policies = {row["policy"] for row in summary}
    assert policies == {"single_sample", "more_steps", "best_of_n", "adaptive"}
    for row in summary:
        assert row["n"] == 3
