"""Tests for model-based circuit grouping and holdout stability."""

import numpy as np
import pytest

from quantum_drift_scheduler.modeling.circuit_grouping import (
    GroupingError,
    adjusted_rand_index,
    cluster_labels,
    correlation_diagnostic,
    empirical_probabilities,
    group_circuits,
    model_probability_matrix,
    probability_changes,
    rms_distance_matrix,
    silhouette_score,
    stability_report,
    window_centers,
)
from _qcs_helpers import synthetic_observations, true_model


def two_cluster_changes(seed=0, windows=30):
    """Six circuits: three with tiny changes, three with large, differently shaped changes."""
    rng = np.random.default_rng(seed)
    small = rng.normal(0, 0.002, (3, windows))
    shape = np.sin(np.linspace(0, 6, windows))
    large = np.vstack([0.2 * shape + rng.normal(0, 0.005, windows) for _ in range(3)])
    return np.vstack([small, large])


def test_window_centers_lie_inside_the_period():
    c = window_centers(0.0, 600.0, 100.0, 3000.0)
    assert c[0] >= 600.0 and c[-1] <= 3000.0 and np.allclose(np.diff(c), 600.0)
    with pytest.raises(GroupingError, match="complete window"):
        window_centers(0.0, 600.0, 10.0, 500.0)
    with pytest.raises(GroupingError):
        window_centers(0.0, -1.0, 0.0, 100.0)


def test_distance_matrix_properties_and_errors():
    changes = two_cluster_changes()
    d = rms_distance_matrix(changes)
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0.0)
    assert d[0, 1] < d[0, 4] and d[3, 4] < d[0, 4]
    with pytest.raises(GroupingError, match="at least 3 circuits"):
        rms_distance_matrix(changes[:2])
    sparse = changes.copy()
    sparse[0, :] = np.nan
    with pytest.raises(GroupingError, match="valid windows"):
        rms_distance_matrix(sparse)
    partial = changes.copy()
    partial[0, ::2] = np.nan  # still enough common windows -> fine
    assert np.isfinite(rms_distance_matrix(partial)).all()


def test_grouping_recovers_two_clusters_and_is_magnitude_aware():
    d = rms_distance_matrix(two_cluster_changes())
    labels = [f"c{i}" for i in range(6)]
    res = group_circuits(d, labels)
    assert res.n_groups == 2
    assert len({res.assignments[i] for i in range(3)}) == 1 and len({res.assignments[i] for i in range(3, 6)}) == 1
    assert res.assignments[0] != res.assignments[3]
    assert res.mean_within_group_distance < res.mean_between_group_distance
    assert dict(res.silhouette_by_groups)[2] == max(s for _, s in res.silhouette_by_groups)
    assert sorted(res.groups()) == [0, 1] and sum(len(v) for v in res.groups().values()) == 6


def test_perfectly_correlated_but_differently_scaled_circuits_are_not_grouped_together():
    base = np.sin(np.linspace(0, 6, 40))
    changes = np.vstack([0.001 * base, 0.0012 * base, 0.0011 * base, 0.3 * base, 0.32 * base, 0.31 * base])
    corr = correlation_diagnostic(changes)
    assert corr["min"] > 0.999  # correlation alone cannot tell them apart ...
    res = group_circuits(rms_distance_matrix(changes), list("abcdef"))
    assert res.assignments[0] != res.assignments[3]  # ... the magnitude-aware distance can


def test_grouping_input_validation():
    d = rms_distance_matrix(two_cluster_changes())
    with pytest.raises(GroupingError, match="match"):
        group_circuits(d, ["a", "b"])
    with pytest.raises(GroupingError, match="at least 3"):
        group_circuits(np.zeros((2, 2)), ["a", "b"])
    with pytest.raises(GroupingError, match="group range"):
        group_circuits(d, list("abcdef"), min_groups=6, max_groups=6)


def test_silhouette_and_ari_known_values():
    d = rms_distance_matrix(two_cluster_changes())
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert silhouette_score(d, labels) > 0.8
    assert silhouette_score(d, np.zeros(6, dtype=int)) == 0.0
    assert adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    assert adjusted_rand_index([0, 0, 0, 0], [0, 0, 0, 0]) == 1.0
    with pytest.raises(GroupingError):
        adjusted_rand_index([0], [0])
    assert cluster_labels(d, 2).tolist() == [0, 0, 0, 1, 1, 1]


def test_stability_report_flags_stable_and_unstable_groups():
    changes = two_cluster_changes(seed=1)
    train = group_circuits(rms_distance_matrix(changes), list("abcdef"))
    holdout_model = rms_distance_matrix(two_cluster_changes(seed=2))
    holdout_empirical = rms_distance_matrix(two_cluster_changes(seed=3))
    good = stability_report(train, holdout_model, holdout_empirical)
    assert good.stable and good.model_holdout_ari == pytest.approx(1.0)
    assert good.empirical_within_between_ratio < 1.0
    assert good.training_vs_empirical_holdout_spearman > 0.5

    shuffled = rms_distance_matrix(two_cluster_changes(seed=4)[[0, 3, 1, 4, 2, 5]])
    bad = stability_report(train, shuffled, shuffled)
    assert not bad.stable


def test_model_probability_changes_reflect_circuit_sensitivity():
    model = true_model(1000, gammas=(0.1, 0.4), decay_rate=1e-5)
    centers = window_centers(0.0, 600.0, 0.0, 4500.0)
    p = model_probability_matrix(model, [1, 64, 4096], centers)
    assert p.shape == (3, centers.size)
    ch = probability_changes(p)
    assert np.abs(ch[0]).max() < np.abs(ch[1]).max() < np.abs(ch[2]).max()
    with pytest.raises(GroupingError, match="at least 3 windows"):
        probability_changes(p[:, :2])


def test_empirical_probabilities_respect_minimum_counts():
    model = true_model(600, gammas=(0.2,))
    obs = synthetic_observations(model, (1, 64, 512), 600, seed=2)
    centers = window_centers(0.0, 600.0, 0.0, 3000.0)
    emp = empirical_probabilities(obs, 0.0, 600.0, centers, min_observations=20)
    assert emp.shape == (3, centers.size) and np.isfinite(emp).all()
    sparse = empirical_probabilities(obs, 0.0, 600.0, centers, min_observations=10_000)
    assert np.isnan(sparse).all()


def test_correlation_diagnostic_reports_no_variation_instead_of_failing():
    out = correlation_diagnostic(np.zeros((3, 10)))
    assert out["pairs"] == 0.0 and np.isnan(out["mean"])


def test_constant_detuning_gives_a_single_group_and_a_homogeneous_stability_report():
    zero = np.zeros((4, 12))
    d = rms_distance_matrix(zero)
    res = group_circuits(d, list("abcd"))
    assert res.n_groups == 1 and res.assignments == (0, 0, 0, 0) and res.silhouette_by_groups == ()
    report = stability_report(res, d, d)
    assert report.stable and report.model_holdout_ari == 1.0 and np.isnan(report.empirical_within_between_ratio)
    drifting = rms_distance_matrix(two_cluster_changes())
    assert not stability_report(res, drifting, drifting).stable  # holdout became heterogeneous
