"""Tests for maximum-likelihood fitting of the shared-detuning Ramsey model."""

import math

import numpy as np
import pytest
from scipy.optimize import check_grad

import quantum_drift_scheduler.modeling.drift_fitting as df
from quantum_drift_scheduler.modeling.drift_fitting import (
    OptimizationError,
    fit_nested_sequence,
    fit_ramsey_model,
)
from quantum_drift_scheduler.modeling.model_validation import log_loss
from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel, RamseyParameters
from _qcs_helpers import synthetic_observations, true_model

KS = (1, 16, 128, 1024, 4096)
N_ROUNDS = 1200


@pytest.fixture(scope="module")
def drifting():
    model = true_model(N_ROUNDS, gammas=(0.1, 0.25), decay_rate=2e-5)
    return model, synthetic_observations(model, KS, N_ROUNDS, seed=3)


def test_gradient_matches_finite_differences(drifting):
    model, obs = drifting
    objective = df._Objective(obs, model.basis.design(obs.times, 3), 400e-6)
    theta = np.array([0.45, 0.8, math.log(1e-4), 0.12, 0.2, -0.05, 0.03])
    f = lambda x: objective(x)[0]
    g = lambda x: objective(x)[1]
    assert check_grad(f, g, theta) < 1e-6 * max(1.0, np.linalg.norm(g(theta)))
    theta2 = theta.copy()
    theta2[0] = 0.6  # exercises the A > 1/2 branch of B = c * min(A, 1 - A)
    assert check_grad(f, g, theta2) < 1e-6 * max(1.0, np.linalg.norm(g(theta2)))


def test_recovers_constant_detuning():
    model = true_model(N_ROUNDS, gammas=(0.15,), decay_rate=2e-5)
    obs = synthetic_observations(model, KS, N_ROUNDS, seed=5)
    fit = fit_ramsey_model(obs, model.basis, 0, n_starts=5)
    assert fit.converged and fit.n_components == 0
    assert fit.model.parameters.gammas[0] == pytest.approx(0.15, abs=0.02)
    assert fit.model.parameters.offset == pytest.approx(0.5, abs=0.03)


def test_recovers_time_varying_detuning(drifting):
    model, obs = drifting
    fit = fit_ramsey_model(obs, model.basis, 2, n_starts=6, seed=1)
    assert fit.converged and fit.n_parameters == 6
    g = fit.model.parameters.gammas
    assert g[0] == pytest.approx(0.1, abs=0.03) and g[1] == pytest.approx(0.25, abs=0.04)
    # fitted trajectory is close to the truth
    t = np.linspace(obs.times[0], obs.times[-1], 50)
    assert np.max(np.abs(fit.model.detuning(t) - model.detuning(t))) < 0.08


def test_fit_beats_a_constant_detuning_model(drifting):
    model, obs = drifting
    constant = fit_ramsey_model(obs, model.basis, 0, n_starts=5)
    drift = fit_ramsey_model(obs, model.basis, 2, n_starts=5, warm_start=constant.model.parameters)
    assert drift.mean_nll < constant.mean_nll - 0.005
    assert drift.bic < constant.bic


def test_parameter_constraints_always_hold(drifting):
    model, obs = drifting
    for k in (0, 1, 4):
        p = fit_ramsey_model(obs, model.basis, k, n_starts=3).model.parameters
        assert 0.0 < p.contrast <= min(p.offset, 1 - p.offset) + 1e-12
        assert p.decay_rate > 0 and all(abs(g) <= 3.0 for g in p.gammas)
        assert 0.0 <= model.probability(obs.wait_lengths[:50], obs.times[:50]).min()


def test_multiple_initializations_pick_the_best(drifting):
    model, obs = drifting
    fit = fit_ramsey_model(obs, model.basis, 1, n_starts=6, seed=2)
    assert fit.starts_tried == 6 and len(fit.start_nlls) == 6
    assert fit.mean_nll == pytest.approx(min(fit.start_nlls))
    # a single bad start cannot beat the multi-start result
    single = fit_ramsey_model(obs, model.basis, 1, n_starts=1, seed=2)
    assert fit.mean_nll <= single.mean_nll + 1e-9


def test_fit_is_deterministic_for_a_seed(drifting):
    model, obs = drifting
    a = fit_ramsey_model(obs, model.basis, 2, n_starts=4, seed=7)
    b = fit_ramsey_model(obs, model.basis, 2, n_starts=4, seed=7)
    assert a.model.parameters == b.model.parameters and a.mean_nll == b.mean_nll


def test_information_criteria_use_the_parameter_count(drifting):
    model, obs = drifting
    fit = fit_ramsey_model(obs, model.basis, 3, n_starts=3)
    n, k = len(obs), 7
    assert fit.n_parameters == k
    assert fit.bic == pytest.approx(k * math.log(n) + 2 * fit.total_nll)
    assert fit.aic == pytest.approx(2 * k + 2 * fit.total_nll)
    assert fit.total_nll == pytest.approx(fit.mean_nll * n)
    p = fit.model.probability(obs.wait_lengths, obs.times)
    assert fit.mean_nll == pytest.approx(log_loss(obs.outcomes, p), rel=1e-6)


def test_nested_sequence_is_monotone_in_likelihood(drifting):
    model, obs = drifting
    fits = fit_nested_sequence(obs, model.basis, (2, 0, 1, 2), n_starts=4)
    assert [f.n_components for f in fits] == [0, 1, 2]  # sorted, de-duplicated
    nll = [f.mean_nll for f in fits]
    assert nll[0] >= nll[1] - 1e-6 and nll[1] >= nll[2] - 1e-6  # warm starts never get worse
    with pytest.raises(ValueError):
        fit_nested_sequence(obs, model.basis, ())


def test_fixed_shape_holds_amplitude_decay_and_estimates_only_detuning(drifting):
    model, obs = drifting
    shape = RamseyParameters(0.49, 0.45, 3e-5, (0.0,))
    fit = fit_ramsey_model(obs, model.basis, 2, n_starts=4, fixed_shape=shape)
    p = fit.model.parameters
    assert p.offset == pytest.approx(0.49) and p.contrast == pytest.approx(0.45)
    assert p.decay_rate == pytest.approx(3e-5, rel=1e-6)
    assert p.gammas[1] != 0.0


def test_invalid_fit_arguments(drifting):
    model, obs = drifting
    with pytest.raises(ValueError, match="n_components"):
        fit_ramsey_model(obs, model.basis, -1)
    with pytest.raises(ValueError, match="n_starts"):
        fit_ramsey_model(obs, model.basis, 0, n_starts=0)
    with pytest.raises(ValueError, match="Insufficient data"):
        fit_ramsey_model(obs.subset(np.arange(len(obs)) < 5), model.basis, 2)


def test_optimization_failure_is_reported_not_hidden(drifting, monkeypatch):
    model, obs = drifting

    class Bad:
        fun = float("nan")
        x = np.zeros(4)
        success = False
        message = "synthetic failure"

    monkeypatch.setattr(df, "minimize", lambda *a, **k: Bad())
    with pytest.raises(OptimizationError, match="every initialisation"):
        fit_ramsey_model(obs, model.basis, 0, n_starts=2)

    def boom(*a, **k):
        raise ValueError("solver exploded")

    monkeypatch.setattr(df, "minimize", boom)
    with pytest.raises(OptimizationError):
        fit_ramsey_model(obs, model.basis, 0, n_starts=2)


def test_near_zero_contrast_data_gives_valid_parameters_at_chance_likelihood():
    # Outcomes carry (almost) no Ramsey signal: the fit must stay valid and cannot beat chance by much.
    model = true_model(600, gammas=(0.0,), decay_rate=1e-3)
    flat = RamseyModel(RamseyParameters(0.5, 0.02, 1e-3, (0.0,)), model.basis)
    obs = synthetic_observations(flat, (1, 64, 512), 600, seed=2)
    fit = fit_ramsey_model(obs, flat.basis, 0, n_starts=3)
    p = fit.model.parameters
    assert 0.0 < p.offset < 1.0 and 0.0 < p.contrast <= min(p.offset, 1 - p.offset) + 1e-12
    assert math.log(2) - 0.02 < fit.mean_nll <= math.log(2) + 1e-6
