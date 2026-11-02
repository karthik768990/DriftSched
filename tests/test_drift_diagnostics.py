"""Tests for the observational drift-model diagnostics (synthetic data only)."""

import math

import numpy as np
import pytest

from quantum_drift_scheduler.data.parser import DatasetParser
from quantum_drift_scheduler.modeling.diagnostics import (
    PersistenceModel,
    baseline_probabilities,
    cluster_optima,
    horizon_log_loss,
    independent_log_loss,
    independent_probability,
    lag1_autocorrelation,
    multistart_study,
    numerical_hessian,
    parameter_uncertainty,
    read_raw_dataset,
    reflection_gap,
    window_table,
)
from quantum_drift_scheduler.modeling.drift_fitting import fit_ramsey_model
from quantum_drift_scheduler.modeling.model_validation import constant_baseline, log_loss
from quantum_drift_scheduler.modeling.ramsey_model import DctBasis, RamseyModel, RamseyParameters
from _qcs_helpers import synthetic_observations, true_model

KS = (1, 16, 128, 1024, 4096)


@pytest.fixture(scope="module")
def case():
    model = true_model(1200, gammas=(0.1, 0.25, -0.15), a=0.5, b=0.45, decay_rate=2e-5)
    obs = synthetic_observations(model, KS, 1200, seed=3)
    return model, obs


# ---------------------------------------------------------------- independent re-implementations


def test_independent_probability_matches_the_production_model(case):
    model, obs = case
    mine = independent_probability(obs.wait_lengths, obs.times, model.parameters, model.basis)
    assert np.max(np.abs(mine - model.probability(obs.wait_lengths, obs.times))) < 1e-12
    far = np.array([20000.0, 29000.0])  # also outside the fitted span
    assert np.allclose(independent_probability(np.array([512, 512]), far, model.parameters, model.basis),
                       model.probability(np.array([512, 512]), far), atol=1e-12)


def test_independent_log_loss_matches_production_and_known_values(case):
    model, obs = case
    p = model.probability(obs.wait_lengths, obs.times)
    assert independent_log_loss(obs.outcomes, p) == pytest.approx(log_loss(obs.outcomes, p), rel=1e-12)
    y = np.array([1, 0, 1, 0])
    assert independent_log_loss(y, np.full(4, 0.5)) == pytest.approx(math.log(2))
    assert independent_log_loss(y, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(-(2 * math.log(0.9) + 2 * math.log(0.8)) / 4)
    assert math.isfinite(independent_log_loss(np.array([1, 0]), np.array([0.0, 1.0])))  # clipped, not infinite


def test_baseline_probabilities_reproduce_the_production_baseline(case):
    _, obs = case
    cut = obs.times[0] + 0.75 * (obs.times[-1] - obs.times[0])
    train, hold = obs.subset(obs.times < cut), obs.subset(obs.times >= cut)
    p = baseline_probabilities(train, hold)
    assert independent_log_loss(hold.outcomes, p) == pytest.approx(constant_baseline(train, hold)[0], rel=1e-12)
    for i, circuit in enumerate(train.circuits):  # training means only
        assert p[hold.circuit_index == i][0] == pytest.approx(train.outcomes[train.circuit_index == i].mean())


def test_read_raw_dataset_agrees_with_the_project_parser(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("## Outcomes = 1, 0\nGx(Gw)Gy\ntimes: 0.5 5.5 10.5\noutcomes: 1 0 1\nrepetitions: 1 1 1\n"
                    "Gx(Gw)^2Gy\ntimes: 1.5 6.5 11.5\noutcomes: 0 0 1\nrepetitions: 1 1 1\n", encoding="utf-8")
    raw = read_raw_dataset(path)
    records, errors = DatasetParser(validate=True).parse_file(path)
    assert not errors and list(raw) == [r.circuit_label for r in records]
    for r in records:
        assert np.array_equal(raw[r.circuit_label][0], np.array(r.times)) and np.array_equal(raw[r.circuit_label][1], np.array(r.outcomes))
    path.write_text("Gx(Gw)Gy\ntimes: 1 2 3\noutcomes: 1 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        read_raw_dataset(path)


# ---------------------------------------------------------------- the basis outside the fitted span


def test_dct_basis_extrapolates_by_reflection_about_the_end_of_the_fitted_span():
    basis = DctBasis(t0=100.0, t_step=5.0, n_steps=400)
    taus = np.array([5.0, 50.0, 500.0, 1500.0])
    t_mirror, gaps = reflection_gap(basis, (0.2, 0.3, -0.1, 0.05), taus)
    assert t_mirror == pytest.approx(100.0 + 399.5 * 5.0)  # half a raster after the last raster
    assert np.max(np.abs(gaps)) < 1e-12  # Omega(T + tau) == Omega(T - tau) exactly
    model = RamseyModel(RamseyParameters(0.5, 0.5, 0.0, (0.2, 0.3, -0.1, 0.05)), basis)
    slope = (model.detuning(np.array([t_mirror + 1e-3]))[0] - model.detuning(np.array([t_mirror - 1e-3]))[0]) / 2e-3
    assert abs(slope) < 1e-9  # zero slope at the boundary: the modelled drift cannot continue a trend
    inside = model.detuning(np.array([t_mirror - 400.0, t_mirror - 200.0]))
    assert abs(inside[0] - inside[1]) > 1e-3  # while it does vary strictly inside the span


def test_persistence_model_holds_the_detuning_after_the_hold_time(case):
    model, _ = case
    persist = PersistenceModel(model, hold_time=3000.0)
    l = np.full(3, 1024.0)
    t = np.array([1000.0, 3000.0, 5000.0])
    p = persist.probability(l, t)
    assert p[0] == pytest.approx(model.probability(l[:1], t[:1])[0])
    assert p[1] == pytest.approx(p[2]) == pytest.approx(model.probability(l[:1], np.array([3000.0]))[0])


# ---------------------------------------------------------------- tables


def test_window_table_accounts_for_every_observation_and_labels_periods(case):
    model, obs = case
    boundary = obs.times[0] + 0.75 * (obs.times[-1] - obs.times[0])
    p = model.probability(obs.wait_lengths, obs.times)
    base = np.full(len(obs), 0.5)
    rows = window_table(obs, p, base, float(obs.times[0]), 600.0, boundary)
    assert sum(r["n"] for r in rows) == len(obs)
    for i, circuit in enumerate(obs.circuits):
        mine = [r for r in rows if r["circuit"] == circuit]
        assert sum(r["n"] for r in mine) == (obs.circuit_index == i).sum()
        assert sum(r["n"] * r["observed_frequency"] for r in mine) == pytest.approx(obs.outcomes[obs.circuit_index == i].sum())
    assert all(abs(r["residual"] - (r["observed_frequency"] - r["predicted_mean"])) < 1e-12 for r in rows)
    assert {r["period"] for r in rows} == {"train", "holdout"}
    assert all((r["window_start"] >= boundary) == (r["period"] == "holdout") for r in rows)
    constant = [r for r in rows if r["n"] > 0]
    assert all(r["baseline_log_loss"] >= 0 for r in constant)


def test_horizon_log_loss_partitions_the_holdout(case):
    model, obs = case
    boundary = obs.times[0] + 0.75 * (obs.times[-1] - obs.times[0])
    hold = obs.subset(obs.times >= boundary)
    p = model.probability(hold.wait_lengths, hold.times)
    subsets = {"all": np.ones(len(hold), bool), "high": hold.wait_lengths >= 1024}
    rows = horizon_log_loss(hold, {"model": p, "half": np.full(len(hold), 0.5)}, boundary, 600.0, subsets)
    assert sum(r["n"] for r in rows if r["subset"] == "all") == len(hold)
    assert sum(r["n"] for r in rows if r["subset"] == "high") == int((hold.wait_lengths >= 1024).sum())
    assert all(abs(r["log_loss_half"] - math.log(2)) < 1e-12 for r in rows)
    assert min(r["horizon_bin"] for r in rows) == 0


def test_lag1_autocorrelation_known_values():
    assert lag1_autocorrelation([1, -1, 1, -1, 1, -1]) < -0.7
    assert lag1_autocorrelation(np.arange(20.0)) > 0.8
    assert lag1_autocorrelation([1.0, 1.0, 1.0]) is None and lag1_autocorrelation([1.0, 2.0]) is None


# ---------------------------------------------------------------- identifiability and local optima


def test_numerical_hessian_recovers_a_quadratic():
    q = np.array([[3.0, 0.5, 0.0], [0.5, 2.0, 0.3], [0.0, 0.3, 1.0]])
    h = numerical_hessian(lambda x: q @ x, np.array([0.3, -1.0, 2.0]), [0, 1, 2])
    assert np.allclose(h, q, atol=1e-7)
    sub = numerical_hessian(lambda x: q @ x, np.array([0.3, -1.0, 2.0]), [0, 2])
    assert np.allclose(sub, q[np.ix_([0, 2], [0, 2])], atol=1e-7)


def test_parameter_uncertainty_at_the_fitted_maximum_is_calibrated(case):
    model, obs = case
    fit = fit_ramsey_model(obs, model.basis, 2, n_starts=4, seed=1)
    info = parameter_uncertainty(obs, model.basis, fit.model.parameters, fit.model.parameters.t_w)
    assert info["positive_definite"] and info["observations"] == len(obs) and info["condition_number"] > 1.0
    assert not any(info["at_bound"][:2])  # A and c are interior for this case
    se, truth = info["standard_errors"], model.parameters.gammas
    for j, name in enumerate(("gamma_0", "gamma_1", "gamma_2")):
        assert 0 < se[name] < 0.05
        assert abs(fit.model.parameters.gammas[j] - truth[j]) < 4 * se[name]  # the SE honestly sizes the error


def test_the_contrast_kink_at_a_equals_one_half_is_reported_not_hidden(case):
    model, obs = case  # A = 0.5 exactly: the finite-difference Hessian straddles the kink of min(A, 1 - A)
    info = parameter_uncertainty(obs, model.basis, model.parameters, model.parameters.t_w)
    assert info["positive_definite"] is False and info["standard_errors"] == {}  # no spurious standard errors


def test_parameter_uncertainty_flags_parameters_on_a_bound():
    boundary_case = true_model(600, gammas=(0.15,), a=0.5, b=0.5, decay_rate=2e-5)  # c = 1: on its upper bound
    obs = synthetic_observations(boundary_case, KS, 600, seed=1)
    info = parameter_uncertainty(obs, boundary_case.basis, boundary_case.parameters, boundary_case.parameters.t_w)
    assert info["at_bound"][1] and "c (B = c*min(A,1-A))" not in info["standard_errors"]


def test_multistart_study_finds_the_best_optimum_and_groups_the_rest(case):
    model, obs = case
    rows = multistart_study(obs, model.basis, 1, n_starts=6, seed=4, t_w=model.parameters.t_w)
    assert len(rows) == 6 and all(set(r) >= {"nll", "converged", "iterations", "theta"} for r in rows)
    assert rows == multistart_study(obs, model.basis, 1, n_starts=6, seed=4, t_w=model.parameters.t_w)  # seeded
    clusters = cluster_optima(rows)
    assert sum(c["count"] for c in clusters) == sum(math.isfinite(r["nll"]) for r in rows)
    assert [c["nll"] for c in clusters] == sorted(c["nll"] for c in clusters)
    assert min(r["nll"] for r in rows) < math.log(2)  # beats chance on drifting data
    assert cluster_optima([{"nll": 1.0}, {"nll": 1.00001}, {"nll": 2.0}, {"nll": math.inf}]) == [
        {"nll": 1.0, "count": 2}, {"nll": 2.0, "count": 1}]
