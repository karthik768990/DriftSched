"""Tests for reference differences, comparison errors, MAE and reliability."""

import math

import numpy as np
import pytest

from quantum_drift_scheduler.modeling.comparison_metrics import (
    ComparisonMetricError,
    classify_reliability,
    compute_comparison_errors,
    mean_absolute_error,
    reference_time,
    window_index,
)
from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel, RamseyParameters
from _qcs_helpers import true_model


def test_window_index_and_reference_time_use_fixed_clock_windows():
    t = np.array([0.0, 599.9, 600.0, 1250.0])
    assert window_index(t, 0.0, 600.0).tolist() == [0, 0, 1, 2]
    assert reference_time(t, 0.0, 600.0).tolist() == [300.0, 300.0, 900.0, 1500.0]
    assert reference_time(np.array([10.5]), 0.5, 600.0).tolist() == [300.5]
    with pytest.raises(ComparisonMetricError):
        window_index(t, 0.0, 0.0)
    with pytest.raises(ComparisonMetricError):
        window_index(t, 0.0, float("nan"))


def test_both_circuits_share_one_reference_timestamp():
    model = true_model(1000, gammas=(0.1, 0.3))
    la, lb = np.array([64.0]), np.array([1024.0])
    ta, tb = np.array([1310.0]), np.array([1312.0])
    errs = compute_comparison_errors(model, la, ta, lb, tb, 0.0, 600.0)
    t_ref = 1500.0  # midpoint of the window [1200, 1800) containing the earlier execution
    expected_ref = model.circuit_probability(64, np.array([t_ref])) - model.circuit_probability(1024, np.array([t_ref]))
    assert errs.reference_time[0] == t_ref and errs.window_index[0] == 2
    assert errs.reference_delta[0] == pytest.approx(expected_ref[0])
    expected_meas = model.circuit_probability(64, ta)[0] - model.circuit_probability(1024, tb)[0]
    assert errs.measured_delta[0] == pytest.approx(expected_meas)
    assert errs.error[0] == pytest.approx(abs(expected_meas - expected_ref[0]))
    # the reference window follows the *earlier* execution regardless of which circuit ran first
    flipped = compute_comparison_errors(model, la, tb, lb, ta, 0.0, 600.0)
    assert flipped.reference_time[0] == errs.reference_time[0]
    straddle = compute_comparison_errors(model, la, np.array([1795.0]), lb, np.array([1805.0]), 0.0, 600.0)
    assert straddle.reference_time[0] == 1500.0  # window of the earlier execution (t = 1795)


def test_no_drift_means_zero_error_for_any_schedule():
    model = RamseyModel(RamseyParameters(0.5, 0.5, 1e-4, (0.2,)), true_model(100).basis)  # constant Omega
    rng = np.random.default_rng(0)
    la, lb = rng.integers(1, 4000, 20), rng.integers(1, 4000, 20)
    ta = rng.uniform(0, 5000, 20)
    errs = compute_comparison_errors(model, la, ta, lb, ta + rng.uniform(0, 400, 20), 0.0, 600.0)
    assert np.allclose(errs.error, 0.0, atol=1e-12) and np.allclose(errs.synchronous_error, 0.0, atol=1e-12)


def test_synchronous_error_vanishes_for_simultaneous_execution_and_grows_with_separation():
    model = true_model(1000, gammas=(0.0, 0.5))
    la, lb = np.full(3, 2048.0), np.full(3, 512.0)
    t0 = np.array([100.0, 2000.0, 3500.0])
    same = compute_comparison_errors(model, la, t0, lb, t0, 0.0, 600.0)
    assert np.allclose(same.synchronous_error, 0.0)
    near = compute_comparison_errors(model, la, t0, lb, t0 + 1.0, 0.0, 600.0)
    far = compute_comparison_errors(model, la, t0, lb, t0 + 30.0, 0.0, 600.0)
    assert np.all(far.synchronous_error > near.synchronous_error)


def test_error_is_symmetric_under_swapping_circuit_roles():
    model = true_model(1000, gammas=(0.1, 0.4))
    la, lb = np.array([256.0, 8.0]), np.array([2048.0, 512.0])
    ta, tb = np.array([900.0, 4000.0]), np.array([903.0, 4002.0])
    ab = compute_comparison_errors(model, la, ta, lb, tb, 0.0, 600.0)
    ba = compute_comparison_errors(model, lb, tb, la, ta, 0.0, 600.0)
    assert np.allclose(ab.error, ba.error) and np.allclose(ab.measured_delta, -ba.measured_delta)


def test_empty_input_is_allowed_and_bad_input_rejected():
    model = true_model(100)
    empty = np.array([])
    assert len(compute_comparison_errors(model, empty, empty, empty, empty, 0.0, 600.0)) == 0
    with pytest.raises(ComparisonMetricError, match="identical shapes"):
        compute_comparison_errors(model, np.ones(2), np.ones(2), np.ones(3), np.ones(3), 0.0, 600.0)
    with pytest.raises(ComparisonMetricError, match="finite"):
        compute_comparison_errors(model, np.ones(1), np.array([np.nan]), np.ones(1), np.ones(1), 0.0, 600.0)
    with pytest.raises(ComparisonMetricError, match="positive"):
        compute_comparison_errors(model, np.zeros(1), np.ones(1), np.ones(1), np.ones(1), 0.0, 600.0)


def test_mae_ignores_incomplete_comparisons_and_rejects_empty():
    assert mean_absolute_error(np.array([0.1, 0.3, np.nan])) == pytest.approx(0.2)
    assert mean_absolute_error(np.array([0.0])) == 0.0
    with pytest.raises(ComparisonMetricError, match="MAE is undefined"):
        mean_absolute_error(np.array([np.nan, np.nan]))
    with pytest.raises(ComparisonMetricError):
        mean_absolute_error(np.array([]))


def test_reliability_classification():
    errors = np.array([0.0, 0.05, 0.0500001, 0.2, np.nan])
    assert classify_reliability(errors, 0.05).tolist() == [True, True, False, False, False]
    assert classify_reliability(errors, 1.0).tolist() == [True, True, True, True, False]  # NaN never reliable
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ComparisonMetricError, match="threshold"):
            classify_reliability(errors, bad)
