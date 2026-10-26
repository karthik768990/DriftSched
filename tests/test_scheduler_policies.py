"""Tests for pairing policies."""

import pytest

from quantum_drift_scheduler.data.compatibility import CompatibilityStatus
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ShotArrival,
    MinimumRecordedGapPolicy,
    PendingQueue,
    make_policy,
    run_simulation,
)
from _sched_helpers import make_registry, make_trace


def queue_of(trace):
    queue = PendingQueue()
    for shot in trace:
        queue.add(shot)
    return queue


def test_make_policy_names():
    assert make_policy("arrival-order").name == "arrival-order"
    assert make_policy("minimum-recorded-gap").name == "minimum-recorded-gap"
    with pytest.raises(ValueError, match="Unknown policy"):
        make_policy("drift-aware")


def test_arrival_order_pairs_oldest_with_oldest_compatible():
    trace = make_trace([(0.0, "A", 1), (1.0, "A", 0), (2.0, "B", 1), (3.0, "C", 0)])
    a, b = ArrivalOrderPolicy().select_pair(queue_of(trace), make_registry(), 3.0)
    assert (a.shot_id, b.shot_id) == ("s0000", "s0002")


def test_arrival_order_skips_oldest_without_counterpart():
    # A is oldest but only compatible partner is none (A-B incompatible, A-C incompatible).
    registry = make_registry(
        overrides={
            ("A", "B"): CompatibilityStatus.INCOMPATIBLE,
            ("A", "C"): CompatibilityStatus.CONDITIONALLY_COMPATIBLE,
        }
    )
    trace = make_trace([(0.0, "A", 1), (1.0, "B", 0), (2.0, "C", 1)])
    a, b = ArrivalOrderPolicy().select_pair(queue_of(trace), registry, 2.0)
    assert (a.circuit, b.circuit) == ("B", "C")


def test_policies_return_none_for_same_circuit_only_queue():
    trace = make_trace([(0.0, "A", 1), (1.0, "A", 0), (2.0, "A", 1)])
    queue = queue_of(trace)
    assert ArrivalOrderPolicy().select_pair(queue, make_registry(), 2.0) is None
    assert MinimumRecordedGapPolicy().select_pair(queue, make_registry(), 2.0) is None


def test_policies_return_none_on_empty_queue():
    assert ArrivalOrderPolicy().select_pair(PendingQueue(), make_registry(), 0.0) is None
    assert MinimumRecordedGapPolicy().select_pair(PendingQueue(), make_registry(), 0.0) is None


def test_minimum_gap_picks_smallest_separation():
    trace = make_trace([(0.0, "A", 1), (10.0, "B", 0), (10.5, "C", 1)])
    a, b = MinimumRecordedGapPolicy().select_pair(queue_of(trace), make_registry(), 10.5)
    assert (a.circuit, b.circuit) == ("B", "C")


def test_minimum_gap_ignores_incompatible_closer_pair():
    registry = make_registry(overrides={("B", "C"): CompatibilityStatus.INCOMPATIBLE})
    trace = make_trace([(0.0, "A", 1), (10.0, "B", 0), (10.5, "C", 1)])
    a, b = MinimumRecordedGapPolicy().select_pair(queue_of(trace), registry, 10.5)
    assert (a.circuit, b.circuit) == ("A", "B")


def test_minimum_gap_ties_prefer_earlier_max_arrival():
    # Two disjoint pairs with equal gap 1.0: (A@0, B@1) and (C@5, D@6).
    registry = make_registry(("A", "B", "C", "D"))
    trace = make_trace([(0.0, "A", 1), (1.0, "B", 0), (5.0, "C", 0), (6.0, "D", 1)])
    a, b = MinimumRecordedGapPolicy().select_pair(queue_of(trace), registry, 6.0)
    assert (a.circuit, b.circuit) == ("A", "B")


def test_minimum_gap_ties_break_by_source_index():
    # All three pairs have gap 0; the lowest source indices win.
    trace = make_trace([(7.0, "A", 1), (7.0, "B", 0), (7.0, "C", 1)])
    a, b = MinimumRecordedGapPolicy().select_pair(queue_of(trace), make_registry(), 7.0)
    assert (a.shot_id, b.shot_id) == ("s0000", "s0001")


def test_minimum_gap_ties_break_by_shot_id_when_source_index_equal():
    registry = make_registry(("A", "B", "C"))
    shots = [
        ShotArrival(shot_id=sid, arrival_time=3.0, circuit=c, outcome=1, repetitions=1,
                    source_index=0, input_source="synthetic")
        for sid, c in (("x1", "A"), ("x2", "B"), ("x3", "C"))
    ]
    a, b = MinimumRecordedGapPolicy().select_pair(queue_of(shots), registry, 3.0)
    assert (a.shot_id, b.shot_id) == ("x1", "x2")


def test_minimum_gap_considers_all_shots_of_a_circuit():
    # The closest A shot to B@10 is A@9, not the oldest A@0.
    trace = make_trace([(0.0, "A", 1), (9.0, "A", 0), (10.0, "B", 1)])
    a, b = MinimumRecordedGapPolicy().select_pair(queue_of(trace), make_registry(), 10.0)
    assert (a.shot_id, b.shot_id) == ("s0001", "s0002")


def test_policies_are_deterministic():
    trace = make_trace(
        [(0.0, "A", 1), (0.0, "B", 0), (1.0, "C", 1), (1.0, "A", 0), (2.0, "B", 1), (2.0, "C", 0)]
    )
    registry = make_registry()
    for name in ("arrival-order", "minimum-recorded-gap"):
        first = run_simulation(trace, registry, make_policy(name))
        second = run_simulation(trace, registry, make_policy(name))
        assert first == second
