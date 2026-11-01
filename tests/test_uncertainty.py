"""Tests for the temporal block bootstrap and confidence intervals."""

import numpy as np
import pytest

from quantum_drift_scheduler.modeling.uncertainty import (
    MIN_BLOCKS,
    InsufficientBlocksError,
    UncertaintyError,
    block_bootstrap_indices,
    bootstrap_statistics,
    choose_block_length,
    integrated_autocorrelation_time,
)


def ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def test_autocorrelation_time_white_noise_vs_correlated():
    white = np.random.default_rng(1).normal(size=4000)
    assert integrated_autocorrelation_time(white) < 2.0
    assert integrated_autocorrelation_time(ar1(4000, 0.9)) > 8.0  # theory: (1+phi)/(1-phi) = 19
    assert integrated_autocorrelation_time(np.ones(10)) == 1.0
    with pytest.raises(UncertaintyError):
        integrated_autocorrelation_time([1.0, 2.0])
    with pytest.raises(UncertaintyError):
        integrated_autocorrelation_time([1.0, 2.0, np.nan, 4.0, 5.0])


def test_block_length_is_a_whole_number_of_workload_cycles_and_capped():
    series = ar1(5000, 0.8)
    length = choose_block_length(series, cycle_length=7)
    assert length % 7 == 0 and length >= 2
    capped = choose_block_length(series, cycle_length=7, n_items=210)
    assert capped <= 21 and capped % 7 == 0  # at least MIN_BLOCKS blocks of 21 fit into 210
    assert choose_block_length(np.random.default_rng(0).normal(size=2000), cycle_length=1) >= 2
    with pytest.raises(InsufficientBlocksError):
        choose_block_length(series, cycle_length=100, n_items=500)
    with pytest.raises(UncertaintyError):
        choose_block_length(series, cycle_length=0)


def test_bootstrap_indices_form_contiguous_circular_blocks():
    rng = np.random.default_rng(3)
    idx = block_bootstrap_indices(1000, 25, rng)
    assert idx.shape == (1000,) and idx.min() >= 0 and idx.max() < 1000
    for start in range(0, 1000, 25):
        block = idx[start : start + 25]
        assert np.all((np.diff(block) % 1000) == 1)  # consecutive (mod n) within a block
    a = block_bootstrap_indices(1000, 25, np.random.default_rng(9))
    b = block_bootstrap_indices(1000, 25, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_bootstrap_requires_enough_blocks():
    with pytest.raises(InsufficientBlocksError):
        block_bootstrap_indices(100, 50, np.random.default_rng(0))  # only 2 blocks
    assert MIN_BLOCKS == 10
    with pytest.raises(UncertaintyError, match="block_length"):
        block_bootstrap_indices(100, 0, np.random.default_rng(0))
    with pytest.raises(InsufficientBlocksError):
        bootstrap_statistics(100, lambda i: {"m": 0.0}, block_length=50, n_replicates=100)


def test_confidence_interval_brackets_the_estimate_and_is_reproducible():
    data = ar1(3000, 0.5, seed=5) + 2.0
    stat = lambda idx: {"mean": float(data[idx].mean()), "spread": float(data[idx].std())}
    ci = bootstrap_statistics(data.size, stat, block_length=30, n_replicates=500, seed=4)
    assert set(ci) == {"mean", "spread"}
    m = ci["mean"]
    assert m.estimate == pytest.approx(data.mean()) and m.lower < m.estimate < m.upper
    assert m.confidence == 0.95 and m.replicates == 500 and m.block_length == 30 and m.seed == 4
    assert m.standard_error > 0
    again = bootstrap_statistics(data.size, stat, block_length=30, n_replicates=500, seed=4)
    assert again["mean"].lower == m.lower and again["mean"].upper == m.upper
    other_seed = bootstrap_statistics(data.size, stat, block_length=30, n_replicates=500, seed=5)
    assert other_seed["mean"].lower != m.lower


def test_block_bootstrap_widens_intervals_for_dependent_data():
    data = ar1(4000, 0.9, seed=2)
    stat = lambda idx: {"mean": float(data[idx].mean())}
    iid = bootstrap_statistics(data.size, stat, block_length=1, n_replicates=400, seed=1)["mean"]
    blocked = bootstrap_statistics(data.size, stat, block_length=100, n_replicates=400, seed=1)["mean"]
    assert blocked.standard_error > 1.5 * iid.standard_error  # iid resampling understates uncertainty


def test_interval_coverage_on_iid_data_is_reasonable():
    rng = np.random.default_rng(0)
    covered = 0
    for trial in range(40):
        data = rng.normal(1.0, 1.0, 600)
        ci = bootstrap_statistics(600, lambda i: {"m": float(data[i].mean())}, 20, n_replicates=200, seed=trial)["m"]
        covered += ci.lower <= 1.0 <= ci.upper
    assert covered >= 32  # nominal 95%


def test_difference_statistics_share_resamples_and_zero_exclusion():
    x = np.random.default_rng(1).normal(size=2000)
    y = x + 0.5  # a constant offset: the paired difference has zero variance
    stat = lambda i: {"d": float(y[i].mean() - x[i].mean())}
    d = bootstrap_statistics(2000, stat, 20, n_replicates=200, seed=0)["d"]
    assert d.lower == pytest.approx(0.5) and d.upper == pytest.approx(0.5) and d.excludes_zero
    noise = bootstrap_statistics(2000, lambda i: {"d": float(x[i].mean())}, 20, n_replicates=200, seed=0)["d"]
    assert not noise.excludes_zero


def test_invalid_bootstrap_arguments():
    f = lambda i: {"m": 0.0}
    with pytest.raises(UncertaintyError, match="n_replicates"):
        bootstrap_statistics(1000, f, 10, n_replicates=10)
    with pytest.raises(UncertaintyError, match="confidence"):
        bootstrap_statistics(1000, f, 10, n_replicates=200, confidence=1.0)
    with pytest.raises(UncertaintyError, match="undefined"):
        bootstrap_statistics(1000, lambda i: {"m": float("nan")}, 10, n_replicates=200)
