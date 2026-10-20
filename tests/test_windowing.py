"""Unit tests for time-windowing and pairwise drift diagnostics."""

import math
import pytest
from quantum_drift_scheduler.analysis.windowing import (
    analyze_pairwise_drift,
    compute_windowed_stats,
    compute_wilson_interval,
)
from quantum_drift_scheduler.data.models import CircuitExperimentRecord


def test_wilson_interval_properties():
    # 50 successes out of 100 trials
    lower, upper = compute_wilson_interval(50, 100, confidence=0.95)
    assert 0.0 < lower < 0.5
    assert 0.5 < upper < 1.0
    assert abs((lower + upper) / 2.0 - 0.5) < 0.01

    # Boundary case: 0 successes
    low_0, high_0 = compute_wilson_interval(0, 100, confidence=0.95)
    assert low_0 == 0.0
    assert 0.0 < high_0 < 0.05

    # Boundary case: 100% successes
    low_1, high_1 = compute_wilson_interval(100, 100, confidence=0.95)
    assert 0.95 < low_1 < 1.0
    assert high_1 == 1.0

    # Empty trials
    low_nan, high_nan = compute_wilson_interval(0, 0)
    assert math.isnan(low_nan) and math.isnan(high_nan)


def test_exact_window_boundary_assignment():
    # Times exactly on boundaries: [0.0, 10.0, 20.0, 30.0] with window_size = 10.0
    # Rule [start, end):
    # 0.0 -> w=0
    # 9.9 -> w=0
    # 10.0 -> w=1
    # 19.9 -> w=1
    # 20.0 -> w=2
    times = [0.0, 5.0, 9.999, 10.0, 15.0, 20.0]
    outcomes = [1, 0, 1, 1, 1, 0]
    rec = CircuitExperimentRecord(
        circuit_label="C_bound",
        times=times,
        outcomes=outcomes,
    )

    stats = compute_windowed_stats(rec, window_size_seconds=10.0, start_time=0.0)
    # Windows: [0, 10), [10, 20), [20, 30)
    assert stats.total_windows == 3
    assert stats.observation_counts[0] == 3  # 0.0, 5.0, 9.999
    assert stats.observation_counts[1] == 2  # 10.0, 15.0
    assert stats.observation_counts[2] == 1  # 20.0

    # Check proportions
    assert stats.one_counts[0] == 2
    assert stats.proportions[0] == pytest.approx(2.0 / 3.0)
    assert stats.one_counts[1] == 2
    assert stats.proportions[1] == pytest.approx(1.0)
    assert stats.one_counts[2] == 0
    assert stats.proportions[2] == pytest.approx(0.0)


def test_empty_and_incomplete_windows():
    # Window 0 has observations, window 1 is empty, window 2 has observations
    times = [1.0, 2.0, 25.0]
    outcomes = [1, 0, 1]
    rec = CircuitExperimentRecord(
        circuit_label="C_sparse",
        times=times,
        outcomes=outcomes,
    )
    stats = compute_windowed_stats(rec, window_size_seconds=10.0, start_time=0.0)
    assert stats.total_windows == 3
    assert stats.observation_counts[0] == 2
    assert stats.observation_counts[1] == 0
    assert stats.observation_counts[2] == 1

    # Check that empty window has NaN proportions and standard errors
    assert math.isnan(stats.proportions[1])
    assert math.isnan(stats.standard_errors[1])
    assert math.isnan(stats.wilson_lowers[1])
    assert math.isnan(stats.wilson_uppers[1])


def test_pairwise_drift_equal_changes():
    # Both circuits undergo the exact same change between window 0 and 1
    # Circuit A: p_0 = 0.2 -> p_1 = 0.5 (delta = +0.3)
    # Circuit B: p_0 = 0.6 -> p_1 = 0.9 (delta = +0.3)
    # Disparity should be exactly 0.0
    times = [1.0] * 10 + [15.0] * 10
    outcomes_a = [1] * 2 + [0] * 8 + [1] * 5 + [0] * 5  # 2/10 -> 5/10
    outcomes_b = [1] * 6 + [0] * 4 + [1] * 9 + [0] * 1  # 6/10 -> 9/10

    rec_a = CircuitExperimentRecord("A", times=times, outcomes=outcomes_a)
    rec_b = CircuitExperimentRecord("B", times=times, outcomes=outcomes_b)

    stats_a = compute_windowed_stats(rec_a, window_size_seconds=10.0)
    stats_b = compute_windowed_stats(rec_b, window_size_seconds=10.0)

    diag = analyze_pairwise_drift(stats_a, stats_b, min_observations_per_window=5)
    assert diag.valid_transitions == 1
    assert diag.unusable_transitions == 0
    assert diag.delta_p_a[0] == pytest.approx(0.3)
    assert diag.delta_p_b[0] == pytest.approx(0.3)
    assert diag.disparities[0] == pytest.approx(0.0)
    assert diag.z_scores[0] == pytest.approx(0.0)
    assert diag.mean_absolute_disparity == pytest.approx(0.0)


def test_pairwise_drift_differing_changes():
    # Circuit A: delta = +0.4
    # Circuit B: delta = -0.2
    # Disparity = +0.4 - (-0.2) = +0.6
    times = [1.0] * 10 + [15.0] * 10
    outcomes_a = [1] * 2 + [0] * 8 + [1] * 6 + [0] * 4  # 0.2 -> 0.6 (delta = +0.4)
    outcomes_b = [1] * 5 + [0] * 5 + [1] * 3 + [0] * 7  # 0.5 -> 0.3 (delta = -0.2)

    rec_a = CircuitExperimentRecord("A", times=times, outcomes=outcomes_a)
    rec_b = CircuitExperimentRecord("B", times=times, outcomes=outcomes_b)

    stats_a = compute_windowed_stats(rec_a, window_size_seconds=10.0)
    stats_b = compute_windowed_stats(rec_b, window_size_seconds=10.0)

    diag = analyze_pairwise_drift(stats_a, stats_b, min_observations_per_window=5)
    assert diag.valid_transitions == 1
    assert diag.disparities[0] == pytest.approx(0.6)
    assert diag.z_scores[0] > 0.0


def test_pairwise_symmetry_under_swap():
    # D_{AB}(w) = - D_{BA}(w)
    # |Z_{AB}(w)| = |Z_{BA}(w)|
    times = [1.0] * 10 + [15.0] * 10
    outcomes_a = [1] * 2 + [0] * 8 + [1] * 6 + [0] * 4
    outcomes_b = [1] * 5 + [0] * 5 + [1] * 3 + [0] * 7

    rec_a = CircuitExperimentRecord("A", times=times, outcomes=outcomes_a)
    rec_b = CircuitExperimentRecord("B", times=times, outcomes=outcomes_b)

    stats_a = compute_windowed_stats(rec_a, window_size_seconds=10.0)
    stats_b = compute_windowed_stats(rec_b, window_size_seconds=10.0)

    diag_ab = analyze_pairwise_drift(stats_a, stats_b, min_observations_per_window=5)
    diag_ba = analyze_pairwise_drift(stats_b, stats_a, min_observations_per_window=5)

    assert diag_ab.disparities[0] == pytest.approx(-diag_ba.disparities[0])
    assert diag_ab.z_scores[0] == pytest.approx(-diag_ba.z_scores[0])
    assert diag_ab.standard_errors[0] == pytest.approx(diag_ba.standard_errors[0])
    assert diag_ab.mean_absolute_disparity == pytest.approx(diag_ba.mean_absolute_disparity)
    assert diag_ab.mean_absolute_z == pytest.approx(diag_ba.mean_absolute_z)


def test_insufficient_observations_handling():
    # Window 0 has only 2 observations (< min_obs = 5)
    times = [1.0] * 2 + [15.0] * 10
    outcomes_a = [1, 0] + [1] * 5 + [0] * 5
    outcomes_b = [0, 0] + [1] * 6 + [0] * 4

    rec_a = CircuitExperimentRecord("A", times=times, outcomes=outcomes_a)
    rec_b = CircuitExperimentRecord("B", times=times, outcomes=outcomes_b)

    stats_a = compute_windowed_stats(rec_a, window_size_seconds=10.0)
    stats_b = compute_windowed_stats(rec_b, window_size_seconds=10.0)

    diag = analyze_pairwise_drift(stats_a, stats_b, min_observations_per_window=5)
    assert diag.valid_transitions == 0
    assert diag.unusable_transitions == 1
    assert math.isnan(diag.disparities[0])
    assert math.isnan(diag.z_scores[0])
    assert math.isnan(diag.mean_absolute_disparity)


def test_deterministic_output():
    times = [1.0, 5.0, 12.0, 18.0]
    outcomes = [1, 0, 1, 0]
    rec = CircuitExperimentRecord("Det", times=times, outcomes=outcomes)

    stats1 = compute_windowed_stats(rec, window_size_seconds=10.0)
    stats2 = compute_windowed_stats(rec, window_size_seconds=10.0)

    assert stats1.observation_counts == stats2.observation_counts
    assert stats1.proportions == stats2.proportions
    assert stats1.wilson_lowers == stats2.wilson_lowers

