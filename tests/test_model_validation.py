"""Tests for scoring rules, calibration and chronological component selection."""

import math

import numpy as np
import pytest

from quantum_drift_scheduler.modeling.model_validation import (
    ValidationError,
    brier_score,
    calibration_table,
    chronological_blocks,
    constant_baseline,
    evaluate_model,
    expected_calibration_error,
    log_loss,
    select_components,
)
from _qcs_helpers import synthetic_observations, true_model

KS = (1, 16, 128, 1024, 4096)


# ---------------------------------------------------------------- scoring rules


def test_log_loss_known_values():
    y = np.array([1, 0, 1, 0])
    assert log_loss(y, np.full(4, 0.5)) == pytest.approx(math.log(2))
    assert log_loss(y, np.array([0.9, 0.1, 0.8, 0.2])) == pytest.approx(
        -(math.log(0.9) * 2 + math.log(0.8) * 2) / 4
    )
    # perfectly wrong, certain predictions are finite thanks to clipping
    assert math.isfinite(log_loss(np.array([1, 0]), np.array([0.0, 1.0])))


def test_brier_known_values():
    y = np.array([1, 0, 1, 0])
    assert brier_score(y, np.full(4, 0.5)) == pytest.approx(0.25)
    assert brier_score(y, np.array([1.0, 0.0, 1.0, 0.0])) == 0.0
    assert brier_score(y, np.array([0.0, 1.0, 0.0, 1.0])) == 1.0


@pytest.mark.parametrize(
    "y, p, message",
    [
        (np.array([0, 2]), np.array([0.5, 0.5]), "binary"),
        (np.array([0, 1]), np.array([0.5, 1.5]), "within"),
        (np.array([0, 1]), np.array([0.5, float("nan")]), "finite"),
        (np.array([0, 1]), np.array([0.5]), "equal length"),
        (np.array([]), np.array([]), "empty"),
        (np.array([[0, 1]]), np.array([[0.5, 0.5]]), "1-D"),
    ],
)
def test_scoring_rejects_invalid_inputs(y, p, message):
    for fn in (log_loss, brier_score):
        with pytest.raises(ValidationError, match=message):
            fn(y, p)


def test_calibration_table_and_ece():
    p = np.array([0.05, 0.05, 0.55, 0.55, 0.95, 0.95])
    y = np.array([0, 0, 1, 0, 1, 1])
    table = calibration_table(y, p, n_bins=10)
    assert sum(b.count for b in table) == 6
    nonempty = [b for b in table if b.count]
    assert [b.observed_frequency for b in nonempty] == [0.0, 0.5, 1.0]
    assert expected_calibration_error(table) == pytest.approx((2 * 0.05 + 2 * 0.05 + 2 * 0.05) / 6)
    assert table[9].count == 2  # probability 1.0 would still land in the last bin
    perfect = calibration_table(np.array([0, 1]), np.array([0.0, 1.0]), 5)
    assert expected_calibration_error(perfect) == 0.0
    with pytest.raises(ValidationError):
        calibration_table(y, p, 0)
    with pytest.raises(ValidationError):
        expected_calibration_error(())


# ---------------------------------------------------------------- chronological split integrity


def test_chronological_blocks_cover_the_timeline():
    times = np.linspace(0.0, 100.0, 50)
    edges = chronological_blocks(times, 4)
    assert edges.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]
    with pytest.raises(ValidationError, match="at least 2 blocks"):
        chronological_blocks(times, 1)
    with pytest.raises(ValidationError, match="Insufficient data"):
        chronological_blocks(times[:5], 4)


@pytest.fixture(scope="module")
def selection_case():
    model = true_model(1500, gammas=(0.1, 0.25), decay_rate=2e-5)
    obs = synthetic_observations(model, KS, 1500, seed=11)
    sel = select_components(obs, model.basis, candidates=(0, 1, 2, 4), n_blocks=4, n_starts=3, seed=1)
    return model, obs, sel


def test_folds_are_expanding_window_and_strictly_chronological(selection_case):
    _, obs, sel = selection_case
    for cand in sel.candidates:
        assert len(cand.folds) == 3  # blocks - 1
        for a, b in zip(cand.folds, cand.folds[1:]):
            assert a.validation_end == b.validation_start  # contiguous validation blocks
            assert b.n_train > a.n_train  # training window expands
        for f in cand.folds:
            assert f.train_end == f.validation_start  # validation strictly follows training
            assert f.validation_start < f.validation_end <= obs.times[-1]
            assert f.n_train > 0 and f.n_validation > 0
    assert sum(f.n_validation for f in sel.candidates[0].folds) <= len(obs)


def test_selection_rule_is_applied(selection_case):
    _, _, sel = selection_case
    best = min(c.mean_validation_log_loss for c in sel.candidates)
    near = [c for c in sel.candidates if c.mean_validation_log_loss <= best + sel.tolerance]
    expected = min(near, key=lambda c: (c.bic, c.n_components))
    assert sel.selected_components == expected.n_components
    assert sel.frozen_fit is expected.training_fit
    assert sel.score_for(sel.selected_components) is expected
    assert str(sel.selected_components) in sel.rule
    for c in sel.candidates:
        assert c.mean_validation_log_loss == pytest.approx(np.mean([f.log_loss for f in c.folds]))
        assert c.bic == c.training_fit.bic


def test_selection_is_deterministic():
    model = true_model(600, gammas=(0.2,), decay_rate=2e-5)
    obs = synthetic_observations(model, KS, 600, seed=4)
    a = select_components(obs, model.basis, (0, 1), n_blocks=3, n_starts=2, seed=3)
    b = select_components(obs, model.basis, (0, 1), n_blocks=3, n_starts=2, seed=3)
    assert a.selected_components == b.selected_components
    assert [c.mean_validation_log_loss for c in a.candidates] == [c.mean_validation_log_loss for c in b.candidates]


def test_constant_drift_selects_no_components():
    model = true_model(1500, gammas=(0.15,), decay_rate=2e-5)  # constant detuning
    obs = synthetic_observations(model, KS, 1500, seed=21)
    sel = select_components(obs, model.basis, candidates=(0, 2, 4), n_blocks=4, n_starts=3, seed=2)
    assert sel.selected_components == 0
    assert sel.frozen_fit.model.parameters.gammas[0] == pytest.approx(0.15, abs=0.03)


def test_invalid_selection_arguments(selection_case):
    model, obs, _ = selection_case
    with pytest.raises(ValidationError, match="non-negative"):
        select_components(obs, model.basis, candidates=(), n_blocks=3)
    with pytest.raises(ValidationError, match="non-negative"):
        select_components(obs, model.basis, candidates=(-1, 0), n_blocks=3)
    with pytest.raises(ValidationError, match="tolerance"):
        select_components(obs, model.basis, candidates=(0,), n_blocks=3, tolerance=-1.0)
    with pytest.raises(ValidationError, match="at least 2 blocks"):
        select_components(obs, model.basis, candidates=(0,), n_blocks=1)


def test_evaluate_model_and_constant_baseline(selection_case):
    model, obs, sel = selection_case
    ev = evaluate_model(sel.frozen_fit.model, obs)
    assert ev.n_observations == len(obs)
    assert 0.0 < ev.log_loss < math.log(2) + 0.05 and 0.0 < ev.brier < 0.3
    assert len(ev.per_circuit_log_loss) == len(KS) and len(ev.calibration) == 10
    assert 0.0 <= ev.expected_calibration_error < 0.2
    # truth-model scores are at least as good as the baseline on the data it generated
    truth = evaluate_model(model, obs)
    base_ll, base_brier = constant_baseline(obs, obs)
    assert truth.log_loss < base_ll and truth.brier < base_brier
