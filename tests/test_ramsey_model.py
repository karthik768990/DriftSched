"""Tests for the Ramsey probability model, parameters, DCT basis and observations."""

import dataclasses
import math

import numpy as np
import pytest

from quantum_drift_scheduler.modeling.observations import (
    ObservationError,
    data_report,
    observations_from_shots,
    validate_arrays,
)
from quantum_drift_scheduler.modeling.ramsey_model import (
    DctBasis,
    ParameterError,
    RamseyModel,
    RamseyParameters,
    clip_probability,
    ramsey_probability,
)
from quantum_drift_scheduler.scheduling import ShotArrival
from _qcs_helpers import raster_times, true_model
from _sched_helpers import label, make_trace

TW = 400e-6


# ---------------------------------------------------------------- equation convention


def test_ramsey_probability_is_a_sine_with_no_extra_phase():
    # Omega = 0  ->  sin(0) = 0 -> p = A for every l (a cosine would give A + B e^{-l/l0}).
    p = ramsey_probability(np.array([1, 64, 4096]), np.zeros(3), 0.4, 0.3, 1e-4, TW)
    assert np.allclose(p, 0.4)
    # Quarter-period detuning: 2*pi*l*t_w*Omega = pi/2  ->  p = A + B exp(-l/l0)
    l = 1000
    omega = 0.25 / (l * TW)
    p = ramsey_probability(np.array([l]), np.array([omega]), 0.5, 0.5, 1e-4, TW)
    assert p[0] == pytest.approx(0.5 + 0.5 * math.exp(-1000 * 1e-4))
    # Sign of detuning flips the sine
    q = ramsey_probability(np.array([l]), np.array([-omega]), 0.5, 0.5, 1e-4, TW)
    assert q[0] == pytest.approx(0.5 - 0.5 * math.exp(-1000 * 1e-4))


def test_decay_is_exp_minus_l_over_l0():
    omega_quarter = lambda l: 0.25 / (l * TW)
    for l, l0 in ((10, 100.0), (512, 2000.0)):
        p = ramsey_probability(np.array([l]), np.array([omega_quarter(l)]), 0.5, 0.5, 1.0 / l0, TW)
        assert p[0] == pytest.approx(0.5 + 0.5 * math.exp(-l / l0))


def test_probabilities_stay_in_unit_interval_for_valid_parameters():
    rng = np.random.default_rng(1)
    for a in (0.2, 0.5, 0.8):
        b = min(a, 1 - a)
        p = ramsey_probability(rng.integers(1, 8193, 500), rng.normal(0, 1.5, 500), a, b, 1e-5, TW)
        assert p.min() >= -1e-12 and p.max() <= 1 + 1e-12


# ---------------------------------------------------------------- parameters


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(offset=0.0, contrast=0.1, decay_rate=0.0, gammas=(0.0,)), "offset A"),
        (dict(offset=1.0, contrast=0.1, decay_rate=0.0, gammas=(0.0,)), "offset A"),
        (dict(offset=0.5, contrast=0.0, decay_rate=0.0, gammas=(0.0,)), "contrast B must be positive"),
        (dict(offset=0.5, contrast=-0.1, decay_rate=0.0, gammas=(0.0,)), "contrast B must be positive"),
        (dict(offset=0.3, contrast=0.4, decay_rate=0.0, gammas=(0.0,)), "exceeds min"),
        (dict(offset=0.5, contrast=0.5, decay_rate=-1.0, gammas=(0.0,)), "decay_rate"),
        (dict(offset=0.5, contrast=0.5, decay_rate=float("nan"), gammas=(0.0,)), "decay_rate"),
        (dict(offset=0.5, contrast=0.5, decay_rate=0.0, gammas=()), "gamma_0"),
        (dict(offset=0.5, contrast=0.5, decay_rate=0.0, gammas=(float("inf"),)), "finite"),
        (dict(offset=0.5, contrast=0.5, decay_rate=0.0, gammas=(0.0,), t_w=0.0), "t_w"),
    ],
)
def test_invalid_parameters_rejected(kwargs, message):
    with pytest.raises(ParameterError, match=message):
        RamseyParameters(**kwargs)


def test_parameter_properties():
    p = RamseyParameters(0.5, 0.5, 1e-4, (0.1, 0.2, 0.3))
    assert p.n_components == 2 and p.l0 == pytest.approx(1e4)
    assert RamseyParameters(0.5, 0.5, 0.0, (0.1,)).l0 == math.inf


# ---------------------------------------------------------------- DCT-like basis


def test_basis_matches_the_paper_formula():
    basis = DctBasis(t0=10.0, t_step=5.0, n_steps=100)
    t = np.array([10.0, 12.5, 100.0, 509.0])
    x = basis.design(t, 3)
    assert x.shape == (4, 4) and np.all(x[:, 0] == 1.0)
    for w in (1, 2, 3):
        expected = np.cos(w * math.pi / 100 * ((t - 10.0) / 5.0 + 0.5))
        assert np.allclose(x[:, w], expected)


def test_basis_from_raster_times_and_frequency_scaling():
    times = raster_times(1000)
    basis = DctBasis.from_raster_times(times)
    assert basis.n_steps == 1000 and basis.t0 == times[0]
    assert basis.t_step == pytest.approx((times[-1] - times[0]) / 999)
    # component w has frequency w / (2 N t_step): 1/(2*1000*5) = 1e-4 Hz
    assert basis.frequency_hz(1) == pytest.approx(1.0 / (2 * 1000 * basis.t_step))
    assert basis.frequency_hz(3) == pytest.approx(3 * basis.frequency_hz(1))


def test_basis_components_are_orthogonal_on_the_raster_grid():
    n = 400
    times = raster_times(n)
    basis = DctBasis.from_raster_times(times)
    x = basis.design(times, 6)[:, 1:]
    gram = x.T @ x
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() < 1e-8 * np.diag(gram).max()  # DCT-II orthogonality
    assert np.allclose(np.diag(gram), n / 2, rtol=1e-6)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DctBasis(0.0, 1.0, 1),
        lambda: DctBasis(0.0, 0.0, 10),
        lambda: DctBasis(0.0, -1.0, 10),
        lambda: DctBasis(float("nan"), 1.0, 10),
        lambda: DctBasis.from_raster_times([1.0]),
        lambda: DctBasis.from_raster_times([1.0, 1.0]),
        lambda: DctBasis.from_raster_times([1.0, float("nan")]),
    ],
)
def test_invalid_basis_rejected(factory):
    with pytest.raises(ParameterError):
        factory()


def test_negative_components_rejected():
    with pytest.raises(ParameterError):
        DctBasis(0.0, 1.0, 10).design(np.array([0.0]), -1)


def test_model_detuning_is_gamma0_plus_weighted_basis():
    model = true_model(500, gammas=(0.2, 0.3, -0.1))
    t = np.array([100.0, 900.0, 2000.0])
    x = model.basis.design(t, 2)
    assert np.allclose(model.detuning(t), 0.2 + 0.3 * x[:, 1] - 0.1 * x[:, 2])
    constant = RamseyModel(RamseyParameters(0.5, 0.5, 0.0, (0.7,)), model.basis)
    assert np.allclose(constant.detuning(t), 0.7)  # constant drift
    p = model.circuit_probability(256, t)
    assert np.allclose(p, model.probability(np.full(3, 256), t))


def test_clip_probability_bounds():
    assert clip_probability(np.array([0.0, 1.0, 0.5])).min() > 0.0
    assert clip_probability(np.array([0.0, 1.0])).max() < 1.0


# ---------------------------------------------------------------- observations


def test_observations_round_trip_and_ordering():
    trace = make_trace([(0.0, 8, 1), (1.0, 1, 0), (2.0, 8, 0), (3.0, 1, 1)])
    obs = observations_from_shots(trace)
    assert len(obs) == 4 and obs.circuits == (label(1), label(8))  # ordered by wait length
    assert obs.wait_lengths.tolist() == [8, 1, 8, 1]
    assert obs.counts_per_circuit() == {label(1): 2, label(8): 2}
    sub = obs.subset(obs.times >= 2.0)
    assert len(sub) == 2 and sub.shot_ids == (trace[2].shot_id, trace[3].shot_id)
    report = data_report(obs)
    assert report["observations_excluded"] == 0 and report["time_unit"] == "seconds"


def test_observation_validation_errors():
    with pytest.raises(ObservationError, match="No observations"):
        observations_from_shots(())
    base = make_trace([(0.0, 1, 1), (1.0, 2, 0)])
    with pytest.raises(ObservationError, match="Non-binary"):
        observations_from_shots((dataclasses.replace(base[0], outcome=2), base[1]))
    with pytest.raises(ObservationError, match="not in chronological order"):
        observations_from_shots((base[1], dataclasses.replace(base[0], arrival_time=0.5)))
    with pytest.raises(ObservationError, match="Missing or non-finite"):
        observations_from_shots((dataclasses.replace(base[0], arrival_time=float("nan")), base[1]))
    with pytest.raises(ObservationError, match="Missing or non-finite"):
        observations_from_shots((dataclasses.replace(base[0], arrival_time=None), base[1]))
    with pytest.raises(ObservationError, match="Negative timestamps"):
        observations_from_shots((dataclasses.replace(base[0], arrival_time=-1.0), base[1]))
    with pytest.raises(ObservationError, match="Duplicate shot ids"):
        observations_from_shots((base[0], dataclasses.replace(base[1], shot_id=base[0].shot_id)))
    with pytest.raises(ObservationError, match="wait length"):
        observations_from_shots((ShotArrival("x", 0.0, "NotRamsey", 1, 1, 0, "t"),))
    with pytest.raises(ObservationError, match="same length"):
        validate_arrays(np.zeros(2), np.ones(3, dtype=int), np.zeros(2, dtype=int))
    with pytest.raises(ObservationError, match="Wait lengths"):
        validate_arrays(np.zeros(1), np.zeros(1, dtype=int), np.zeros(1, dtype=int))


def test_duplicate_timestamps_are_reported_not_dropped():
    trace = make_trace([(1.0, 1, 1), (1.0, 1, 0), (2.0, 2, 1)])
    obs = observations_from_shots(trace)
    assert len(obs) == 3 and data_report(obs)["duplicate_timestamps_within_circuit"] == 1
