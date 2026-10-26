"""Tests for the deterministic chronological train/holdout split."""

import math

import pytest

from quantum_drift_scheduler.analysis.holdout import SplitError, split_trace
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.scheduling import load_recorded_trace
from _sched_helpers import label, make_trace


def linear_trace(n=101, circuits=(1, 2)):
    """Shots at t = 0, 1, ..., n-1, cycling through ``circuits``."""
    return make_trace([(float(i), circuits[i % len(circuits)], i % 2) for i in range(n)])


def small_split(trace, fraction=0.75):
    return split_trace(trace, fraction, min_train_shots=1, min_holdout_shots=1)


def test_exact_boundary_behaviour():
    split = small_split(linear_trace())
    assert split.boundary == 75.0 and split.min_timestamp == 0.0 and split.max_timestamp == 100.0
    assert max(s.arrival_time for s in split.train) == 74.0  # t < boundary
    assert min(s.arrival_time for s in split.holdout) == 75.0  # t >= boundary: the boundary shot is holdout
    assert len(split.train) == 75 and len(split.holdout) == 26


def test_boundary_formula_with_offset_origin():
    trace = make_trace([(10.0 + i, 1 + i % 2, 0) for i in range(41)])  # 10 .. 50
    split = small_split(trace, 0.5)
    assert split.boundary == 10.0 + 0.5 * 40.0
    assert split.train_range == (10.0, 29.0) and split.holdout_range == (30.0, 50.0)


def test_partitions_are_disjoint_complete_and_order_preserving():
    trace = linear_trace()
    split = small_split(trace)
    train_ids = [s.shot_id for s in split.train]
    holdout_ids = [s.shot_id for s in split.holdout]
    assert set(train_ids).isdisjoint(holdout_ids)
    assert sorted(train_ids + holdout_ids) == sorted(s.shot_id for s in trace)
    assert train_ids == [s.shot_id for s in trace if s.shot_id in set(train_ids)]  # original order kept
    assert [s.source_index for s in split.holdout] == sorted(s.source_index for s in split.holdout)


def test_split_preserves_input_order_even_if_unsorted():
    trace = tuple(reversed(linear_trace(21)))
    split = small_split(trace, 0.5)
    assert [s.arrival_time for s in split.train] == sorted((s.arrival_time for s in split.train), reverse=True)


def test_split_is_chronological_not_per_circuit_index():
    # Circuit 2 has far fewer shots; a per-circuit 75% index split would differ from a time split.
    rows = [(float(i), 1, 0) for i in range(100)] + [(float(i), 2, 1) for i in (3, 10, 90, 99)]
    trace = make_trace(sorted(rows))
    split = small_split(trace)
    assert {s.circuit for s in split.holdout if s.arrival_time < split.boundary} == set()
    counts = {c: (t, h) for c, t, h in split.circuit_counts()}
    assert counts[label(2)] == (2, 2)  # shots at 3 and 10 train; 90 and 99 holdout
    assert counts[label(1)] == (75, 25)


def test_split_is_deterministic():
    trace = linear_trace()
    assert small_split(trace) == small_split(trace)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.2, 1.5, float("nan"), float("inf"), None, "0.5"])
def test_invalid_train_fraction_rejected(fraction):
    with pytest.raises(SplitError, match="train_fraction"):
        split_trace(linear_trace(), fraction, min_train_shots=1, min_holdout_shots=1)


def test_empty_and_degenerate_traces_rejected():
    with pytest.raises(SplitError, match="empty"):
        split_trace(())
    with pytest.raises(SplitError, match="zero time span"):
        split_trace(make_trace([(5.0, 1, 0), (5.0, 2, 1)]))
    with pytest.raises(SplitError, match="duplicate shot ids"):
        trace = linear_trace(10)
        split_trace(trace + trace[:1])


def test_insufficient_data_per_circuit_rejected():
    rows = [(float(i), 1, 0) for i in range(100)] + [(1.0, 2, 1), (2.0, 2, 1)]  # circuit 2: no holdout
    trace = make_trace(sorted(rows, key=lambda r: r[0]))
    with pytest.raises(SplitError, match=r"Gx\(Gw\)\^2Gy.*holdout"):
        split_trace(trace, 0.75, min_train_shots=1, min_holdout_shots=1)
    with pytest.raises(SplitError, match="training shots"):
        split_trace(linear_trace(20), 0.75, min_train_shots=50, min_holdout_shots=1)


def test_required_circuits_must_be_present():
    with pytest.raises(SplitError, match=r"Gx\(Gw\)\^4Gy.*0 training shots"):
        split_trace(linear_trace(), 0.75, required_circuits=[label(1), label(2), label(4)],
                    min_train_shots=1, min_holdout_shots=1)


def test_default_dataset_split_matches_75_25():
    trace = load_recorded_trace()
    registry = load_compatibility_registry()
    split = split_trace(trace, 0.75, required_circuits=registry.circuits)
    assert split.boundary == pytest.approx(0.120257 + 0.75 * (29227.1 - 0.120257))
    assert len(split.train) == 63041 and len(split.holdout) == 20959
    assert len(split.train) + len(split.holdout) == 84000
    assert split.train_range == (0.120257, 21919.8) and split.holdout_range == (21921.7, 29227.1)
    assert math.isclose(len(split.train) / 84000, 0.75, abs_tol=0.001)
    assert {c for c, _, _ in split.circuit_counts()} == set(registry.circuits)
    assert all(t >= 4500 and h >= 1497 for _, t, h in split.circuit_counts())
