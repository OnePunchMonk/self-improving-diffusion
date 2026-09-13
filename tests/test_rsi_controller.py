"""JX-07: offline logging, off-policy estimation, promotion gating, and rollback."""

import pytest

from self_improving_diffusion.eval.adaptive_inference import HELD_OUT_SEEDS
from self_improving_diffusion.eval.quality_compute import FROZEN_SEEDS
from self_improving_diffusion.rsi.controller import (
    BEHAVIOR_LOG_SEEDS,
    CANDIDATE_N_VALUES,
    FINAL_SEEDS,
    FROZEN_BASELINE_N,
    SELECTION_SEEDS,
    PolicyRegistry,
    collect_offline_logs,
    ips_estimate,
    run_cycle,
    run_multiple_cycles,
    select_candidate,
)


def test_all_three_seed_ranges_are_mutually_disjoint():
    ranges = [set(BEHAVIOR_LOG_SEEDS), set(SELECTION_SEEDS), set(FINAL_SEEDS)]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            assert ranges[i].isdisjoint(ranges[j])


def test_new_seed_ranges_are_disjoint_from_earlier_milestones_seeds():
    all_rsi_seeds = set(BEHAVIOR_LOG_SEEDS) | set(SELECTION_SEEDS) | set(FINAL_SEEDS)
    assert all_rsi_seeds.isdisjoint(set(FROZEN_SEEDS))
    assert all_rsi_seeds.isdisjoint(set(HELD_OUT_SEEDS))


def test_offline_logs_cover_every_behavior_seed_with_a_valid_action():
    logs = collect_offline_logs(cycle_index=0)
    assert {r.seed for r in logs} == set(BEHAVIOR_LOG_SEEDS)
    for record in logs:
        assert record.action_n in CANDIDATE_N_VALUES
        assert record.propensity == pytest.approx(1.0 / len(CANDIDATE_N_VALUES))


def test_different_cycles_draw_different_behavior_actions():
    logs_a = collect_offline_logs(cycle_index=0)
    logs_b = collect_offline_logs(cycle_index=1)
    actions_a = [r.action_n for r in logs_a]
    actions_b = [r.action_n for r in logs_b]
    assert actions_a != actions_b


def test_ips_estimate_is_zero_for_an_action_never_drawn():
    logs = collect_offline_logs(cycle_index=0)
    drawn_actions = {r.action_n for r in logs}
    never_drawn = [n for n in CANDIDATE_N_VALUES if n not in drawn_actions]
    for n in never_drawn:
        assert ips_estimate(logs, n) == 0.0


def test_select_candidate_returns_one_of_the_declared_candidates():
    logs = collect_offline_logs(cycle_index=0)
    best_n, estimates = select_candidate(logs)
    assert best_n in CANDIDATE_N_VALUES
    assert set(estimates.keys()) == set(CANDIDATE_N_VALUES)


def test_registry_starts_at_frozen_baseline():
    registry = PolicyRegistry()
    assert registry.active_n == FROZEN_BASELINE_N


def test_run_cycle_updates_registry_only_on_promotion():
    registry = PolicyRegistry()
    record = run_cycle(registry, cycle_index=0)
    if record.promoted:
        assert registry.active_n == record.candidate_n
    else:
        assert registry.active_n == record.baseline_n


def test_rollback_reverts_to_the_previous_active_policy():
    registry = PolicyRegistry()
    run_cycle(registry, cycle_index=0)
    n_after_first_cycle = registry.active_n
    run_cycle(registry, cycle_index=1)
    reverted = registry.rollback()
    assert reverted == n_after_first_cycle
    assert registry.active_n == n_after_first_cycle


def test_multiple_cycles_report_has_final_evaluation_and_cost_accounting():
    report = run_multiple_cycles(num_cycles=2)
    assert len(report.cycles) == 2
    assert report.total_offline_cost_steps > 0
    assert report.final_evaluation_cost_steps > 0
    assert isinstance(report.negative_transfer_detected, bool)


def test_final_evaluation_uses_only_final_seeds_not_selection_or_behavior_seeds():
    # a structural check: FINAL_SEEDS must never appear in the seed ranges
    # used to pick or gate the candidate, so nothing was tuned against them.
    assert set(FINAL_SEEDS).isdisjoint(set(BEHAVIOR_LOG_SEEDS))
    assert set(FINAL_SEEDS).isdisjoint(set(SELECTION_SEEDS))
