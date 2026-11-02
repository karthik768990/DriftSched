"""Tests for the deployment-faithful chronological validation (synthetic data only; no holdout)."""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_drift_scheduler.modeling.deployment_validation import (
    A_DCT,
    APPROACHES,
    B_HOLD,
    C_CONST,
    D_HALF,
    HoldoutLeakageError,
    ValidationDesignError,
    ValidationFeatures,
    bootstrap_window_differences,
    decision_rule,
    estimate_window_detuning,
    fold_basis,
    fold_values,
    forecast_fold,
    make_folds,
    run_design,
    select_then_confirm,
)
from quantum_drift_scheduler.modeling.model_validation import chronological_blocks
from quantum_drift_scheduler.modeling.ramsey_model import DctBasis, ramsey_probability
from quantum_drift_scheduler.modeling.diagnostics import per_circuit_constants
from _qcs_helpers import synthetic_observations, true_model

KS = (1, 16, 128, 1024, 4096)
COMMON = dict(candidates=(0, 1), primary_k=1, n_starts=2, seed=1)


@pytest.fixture(scope="module")
def synth():
    model = true_model(900, gammas=(0.1, 0.25, -0.1), decay_rate=2e-5)
    obs = synthetic_observations(model, KS, 900, seed=3)
    boundary = float(obs.times[-1]) + 1.0
    return model, obs, boundary, run_design(obs, boundary, n_blocks=4, **COMMON)


def split_fold(obs, index, n_blocks=4):
    fold = make_folds(obs, n_blocks)[index]
    return fold, obs.subset(fold.train_mask), obs.subset(fold.val_mask)


# ---------------------------------------------------------------- chronology and basis span


def test_every_validation_interval_is_strictly_later_than_its_training_interval(synth):
    _, obs, _, res = synth
    folds = make_folds(obs, 4)
    assert len(folds) == 3 and [f.index for f in folds] == [1, 2, 3]  # blocks - 1, in order, never random
    edges = chronological_blocks(obs.times, 4)
    for f in folds:
        assert obs.times[f.val_mask].min() > obs.times[f.train_mask].max()
        assert f.val_start == pytest.approx(edges[f.index]) and f.val_end == pytest.approx(edges[f.index + 1])
        assert not np.any(f.train_mask & f.val_mask)
    for a, b in zip(folds, folds[1:]):  # validation blocks are contiguous; training windows expand
        assert a.val_end == b.val_start and b.train_mask.sum() > a.train_mask.sum()
    assert all(i["validation_strictly_after_training"] for i in res.folds)
    with pytest.raises(ValidationDesignError, match="strictly after"):
        _, train, val = split_fold(obs, 1)
        forecast_fold(train, ValidationFeatures.from_observations(train), 1, (0,), n_starts=1)  # overlapping interval


def test_each_fold_basis_span_comes_only_from_its_own_training_interval(synth):
    _, obs, _, res = synth
    sizes = []
    for i, info in enumerate(res.folds):
        _, train, val = split_fold(obs, i)
        basis = fold_basis(train)
        ref = train.times[train.circuit_index == 0]
        assert (basis.t0, basis.n_steps) == (ref[0], ref.size) == (info["basis_t0"], info["basis_N"])
        assert basis.t_step == pytest.approx((ref[-1] - ref[0]) / (ref.size - 1))
        mirror = basis.t0 + (basis.n_steps - 0.5) * basis.t_step
        assert mirror == pytest.approx(ref[-1] + 0.5 * basis.t_step)  # the span ends with the fold's own training
        assert val.times.min() > ref[-1]  # validation is beyond the fitted span
        assert info["validation_shots_before_mirror"] <= 2 * len(obs.circuits)  # at most half a raster of overlap
        sizes.append(basis.n_steps)
        other = obs.subset(np.ones(len(obs), bool))  # change everything AFTER this fold's training interval
        later = obs.times >= info["validation_start"]
        scrambled = dataclasses.replace(other, outcomes=np.where(later, 1 - other.outcomes, other.outcomes).astype(other.outcomes.dtype))
        _, train2, _ = split_fold(scrambled, i)
        assert fold_basis(train2) == basis
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)  # each fold's span is larger than the last


# ---------------------------------------------------------------- no leakage


def test_validation_outcomes_never_enter_fitting_or_forecasting(synth):
    _, obs, _, _ = synth
    _, train, val = split_fold(obs, 1)
    feats = ValidationFeatures.from_observations(val)
    assert {f.name for f in dataclasses.fields(ValidationFeatures)} == {"times", "wait_lengths", "circuit_index"}  # no outcomes
    flipped = dataclasses.replace(val, outcomes=(1 - val.outcomes).astype(val.outcomes.dtype))
    feats2 = ValidationFeatures.from_observations(flipped)
    for name in ("times", "wait_lengths", "circuit_index"):
        assert np.array_equal(getattr(feats, name), getattr(feats2, name))
    a = forecast_fold(train, feats, 2, (0, 1), n_starts=2, seed=1)
    b = forecast_fold(train, feats2, 2, (0, 1), n_starts=2, seed=1)
    for k in (0, 1):
        assert a.fits[k].model.parameters == b.fits[k].model.parameters
        assert np.array_equal(a.p_dct[k], b.p_dct[k]) and np.array_equal(a.p_hold[k], b.p_hold[k])
    assert np.array_equal(a.p_constant, b.p_constant)
    from quantum_drift_scheduler.modeling.model_validation import log_loss
    assert log_loss(val.outcomes, a.p_dct[1]) != log_loss(flipped.outcomes, a.p_dct[1])  # outcomes only matter for scoring


def test_no_observation_from_the_original_holdout_can_enter(synth):
    _, obs, boundary, _ = synth
    with pytest.raises(HoldoutLeakageError, match="holdout boundary"):
        run_design(obs, float(obs.times[-1]), n_blocks=4, **COMMON)  # the last observation is at the boundary
    with pytest.raises(HoldoutLeakageError):
        run_design(obs, float(obs.times[-1]) - 100.0, n_blocks=4, **COMMON)
    assert all(info["original_holdout_observations"] == 0 for info in synth[3].folds)


# ---------------------------------------------------------------- the approaches


def test_last_value_hold_uses_only_the_final_training_estimate(synth):
    _, obs, _, _ = synth
    _, train, val = split_fold(obs, 1)
    feats = ValidationFeatures.from_observations(val)
    fc = forecast_fold(train, feats, 2, (0, 1), n_starts=2, seed=1)
    assert fc.last_train_time == float(train.times.max())
    fit = fc.fits[1]
    prm = fit.model.parameters
    assert fc.omega_hold[1] == pytest.approx(float(fit.model.detuning(np.array([fc.last_train_time]))[0]))
    expected = ramsey_probability(feats.wait_lengths, np.full(feats.times.shape, fc.omega_hold[1]), prm.offset, prm.contrast, prm.decay_rate, prm.t_w)
    assert np.allclose(fc.p_hold[1], expected)
    for ci in range(len(obs.circuits)):  # constant in time for each circuit
        assert np.unique(fc.p_hold[1][feats.circuit_index == ci]).size == 1
    shifted = ValidationFeatures(feats.times + 5000.0, feats.wait_lengths, feats.circuit_index)  # the validation times are irrelevant
    fc2 = forecast_fold(train, shifted, 2, (0, 1), n_starts=2, seed=1)
    assert np.allclose(fc2.p_hold[1], fc.p_hold[1]) and not np.allclose(fc2.p_dct[1], fc.p_dct[1])  # but they move approach A


def test_constant_baseline_uses_only_fold_training_outcomes(synth):
    _, obs, _, _ = synth
    _, train, val = split_fold(obs, 2)
    feats = ValidationFeatures.from_observations(val)
    fc = forecast_fold(train, feats, 3, (0,), n_starts=1, seed=1)
    means = np.array([train.outcomes[train.circuit_index == i].mean() for i in range(len(obs.circuits))])
    assert np.allclose(fc.p_constant, means[feats.circuit_index]) and np.allclose(means, per_circuit_constants(train))
    flipped_train = dataclasses.replace(train, outcomes=(1 - train.outcomes).astype(train.outcomes.dtype))
    assert not np.allclose(forecast_fold(flipped_train, feats, 3, (0,), n_starts=1, seed=1).p_constant, fc.p_constant)
    flipped_val = dataclasses.replace(val, outcomes=(1 - val.outcomes).astype(val.outcomes.dtype))
    assert np.array_equal(forecast_fold(train, ValidationFeatures.from_observations(flipped_val), 3, (0,), n_starts=1, seed=1).p_constant, fc.p_constant)


def test_half_baseline_always_predicts_one_half(synth):
    _, obs, _, res = synth
    _, train, val = split_fold(obs, 0)
    fc = forecast_fold(train, ValidationFeatures.from_observations(val), 1, (0,), n_starts=1, seed=1)
    assert np.all(fc.p_half == 0.5) and fc.p_half.shape == val.times.shape
    for r in res.fold_rows:
        if r["approach"] == D_HALF:
            assert r["log_loss"] == pytest.approx(math.log(2)) and r["brier"] == pytest.approx(0.25)
    assert all(np.all(res.pooled[s][D_HALF] == 0.5) for s in res.pooled)


def test_all_approaches_are_scored_on_identical_observations(synth):
    _, obs, _, res = synth
    for info in res.folds:
        for subset in ("all", "low_l", "high_l"):
            ns = {r["n"] for r in res.fold_rows if r["fold"] == info["fold"] and r["subset"] == subset}
            assert len(ns) == 1  # same n for A, B (every K), C and D
        assert {r["n"] for r in res.fold_rows if r["fold"] == info["fold"] and r["subset"] == "all"} == {info["n_validation"]}
    for subset, d in res.pooled.items():  # pooled arrays line up one-to-one
        lengths = {len(v) for v in d.values()}
        assert len(lengths) == 1
    assert set(res.pooled["all"]) == {"y", *APPROACHES}


def test_low_and_high_l_subsets_partition_the_circuits(synth):
    _, obs, _, res = synth
    for info in res.folds:
        n = {s: next(r["n"] for r in res.fold_rows if r["fold"] == info["fold"] and r["subset"] == s and r["approach"] == D_HALF)
             for s in ("all", "low_l", "high_l")}
        assert n["low_l"] + n["high_l"] == n["all"]  # KS has no l between 257 and 511


def test_the_experiment_is_deterministic(synth):
    _, obs, boundary, res = synth
    again = run_design(obs, boundary, n_blocks=4, **COMMON)
    assert again.fold_rows == res.fold_rows and again.circuit_rows == res.circuit_rows
    assert again.detuning_rows == res.detuning_rows and again.folds == res.folds
    assert all(np.array_equal(again.pooled["all"][a], res.pooled["all"][a]) for a in APPROACHES)
    assert np.array_equal(again.window_table["ll"], res.window_table["ll"])


# ---------------------------------------------------------------- retrospective detuning (diagnostic only)


def test_retrospective_window_estimate_recovers_a_known_detuning():
    rng = np.random.default_rng(0)
    ls = np.repeat(np.array([1, 16, 128, 1024, 4096]), 150)
    truth, a, b, lam, tw = 0.21, 0.5, 0.45, 2e-5, 400e-6
    y = (rng.random(ls.size) < ramsey_probability(ls, np.full(ls.size, truth), a, b, lam, tw)).astype(int)
    est = estimate_window_detuning(ls, y, a, b, lam, tw)
    assert abs(est["omega"] - truth) < 0.03 and est["low"] <= truth <= est["high"] and not est["ambiguous"]
    only_low = estimate_window_detuning(ls[ls <= 16], y[ls <= 16], a, b, lam, tw)  # no phase information: wide interval
    assert only_low["high"] - only_low["low"] > 1.0


def test_detuning_diagnostic_rows_are_labelled_and_complete(synth):
    _, _, _, res = synth
    assert res.detuning_rows
    for r in res.detuning_rows:
        assert set(r) >= {"omega_forecast_A_dct_reflecting", "omega_forecast_B_last_value", "omega_retrospective_diagnostic"}
        assert r["error_A"] == pytest.approx(r["omega_forecast_A_dct_reflecting"] - r["omega_retrospective_diagnostic"])
        assert r["error_B"] == pytest.approx(r["omega_forecast_B_last_value"] - r["omega_retrospective_diagnostic"])
    by_fold = {r["fold"]: r["omega_forecast_B_last_value"] for r in res.detuning_rows}
    assert all(len({r["omega_forecast_B_last_value"] for r in res.detuning_rows if r["fold"] == f}) == 1 for f in by_fold)


# ---------------------------------------------------------------- bootstrap, decision rule, confirmation


def crafted(rows, n_folds=4):
    return SimpleNamespace(fold_rows=rows, folds=[{"fold": i + 1} for i in range(n_folds)])


def rows_for(a_vals, c_vals, d_vals, k=2, approach=A_DCT):
    rows = []
    for i, (a, c, d) in enumerate(zip(a_vals, c_vals, d_vals), start=1):
        rows += [{"fold": i, "approach": B_HOLD, "K": k, "subset": s, "log_loss": 0.9} for s in ("all", "low_l", "high_l")]
        rows += [{"fold": i, "approach": approach, "K": k, "subset": s, "log_loss": a} for s in ("all", "low_l", "high_l")]
        rows += [{"fold": i, "approach": C_CONST, "K": None, "subset": s, "log_loss": c} for s in ("all", "low_l", "high_l")]
        rows += [{"fold": i, "approach": D_HALF, "K": None, "subset": s, "log_loss": d} for s in ("all", "low_l", "high_l")]
    return rows


def cis(lower, upper, subset="all"):
    ci = SimpleNamespace(lower=lower, upper=upper)
    return {f"{s}|{a}-{b}": ci for s in ("all", "low_l", "high_l") for a in (A_DCT, B_HOLD) for b in (C_CONST, D_HALF)}


def test_decision_rule_requires_both_baselines_consistency_and_a_clear_interval():
    base = dict(c_vals=[0.70] * 4, d_vals=[0.693] * 4)
    win = decision_rule(crafted(rows_for([0.60] * 4, **base)), cis(-0.1, -0.05), 2)["A_dct_reflecting|all"]
    assert win["shows_forecasting_value"] and win["folds_better_than_C"] == 4 and win["folds_needed"] == 3
    beats_only_one = decision_rule(crafted(rows_for([0.695] * 4, **base)), cis(-0.1, -0.05), 2)["A_dct_reflecting|all"]
    assert beats_only_one["lower_mean_than_C"] and not beats_only_one["lower_mean_than_D"] and not beats_only_one["shows_forecasting_value"]
    two_of_four = decision_rule(crafted(rows_for([0.60, 0.60, 0.80, 0.80], **base)), cis(-0.1, -0.05), 2)["A_dct_reflecting|all"]
    assert not two_of_four["consistent_vs_C"] and not two_of_four["shows_forecasting_value"]
    crosses_zero = decision_rule(crafted(rows_for([0.60] * 4, **base)), cis(-0.1, 0.02), 2)["A_dct_reflecting|all"]
    assert not crosses_zero["ci_below_zero_vs_C"] and not crosses_zero["shows_forecasting_value"]


def test_select_then_confirm_chooses_on_early_folds_and_judges_on_later_ones():
    rows = []
    for k, vals in ((0, [0.70, 0.70, 0.70, 0.70]), (1, [0.60, 0.60, 0.90, 0.90])):  # K=1 wins early, loses late
        rows += rows_for(vals, [0.70] * 4, [0.693] * 4, k=k)
    out = select_then_confirm(crafted(rows), (0, 1))["A_dct_reflecting|all"]
    assert out["chosen_K"] == 1 and out["chosen_on_folds"] == [1, 2] and out["confirmed_on_folds"] == [3, 4]
    assert out["confirm_mean"] == pytest.approx(0.90) and not out["beats_both_baselines_on_confirmation_folds"]


def test_window_bootstrap_detects_a_consistent_improvement_deterministically():
    rng = np.random.default_rng(1)
    w = 40
    n = np.full((w, 3), 100.0)
    ll = np.zeros((w, 3, 4))
    ll[:, :, 0] = 100 * (0.60 + rng.normal(0, 0.01, (w, 3)))  # A
    ll[:, :, 1] = 100 * 0.72  # B
    ll[:, :, 2] = 100 * 0.70  # C
    ll[:, :, 3] = 100 * 0.693  # D
    res = SimpleNamespace(window_table={"n": n, "ll": ll})
    out = bootstrap_window_differences(res, n_replicates=200, seed=3)
    assert out["all|A_dct_reflecting-C_circuit_constant"].upper < 0 and out["all|B_last_value_hold-D_half"].lower > 0
    assert out["all|A_dct_reflecting-D_half"].estimate == pytest.approx(0.60 - 0.693, abs=0.01)
    again = bootstrap_window_differences(res, n_replicates=200, seed=3)
    assert out["all|A_dct_reflecting-C_circuit_constant"].lower == again["all|A_dct_reflecting-C_circuit_constant"].lower
    assert len(fold_values(synth_result_stub(), A_DCT, 2, "all")) == 4


def synth_result_stub():
    return crafted(rows_for([0.6] * 4, [0.7] * 4, [0.693] * 4))
