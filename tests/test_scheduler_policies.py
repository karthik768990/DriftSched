"""Tests for the pair-request workload, slot schedule and policy ordering keys."""

import pytest

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityStatus,
    load_compatibility_registry,
    wait_length,
)
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    ShotArrival,
    SimulationError,
    build_pair_requests,
    build_slot_schedule,
    make_baseline_policy,
    request_order_key,
    round_robin_matchings,
    shot_round_indices,
)
from quantum_drift_scheduler.scheduling.workload import make_request
from _sched_helpers import label, make_registry, make_trace, ramsey_registry


def test_wait_length_parsing():
    assert wait_length("Gx(Gw)Gy") == 1
    assert wait_length("Gx(Gw)^8192Gy") == 8192
    for bad in ("A", "Gx(Gw)^0Gy", "Gx(Gw)^-2Gy", "Gx(Gw)^xGy", ""):
        with pytest.raises(ValueError, match="wait length"):
            wait_length(bad)


def test_make_baseline_policy():
    assert isinstance(make_baseline_policy("arrival-order"), ArrivalOrderPolicy)
    assert isinstance(make_baseline_policy("consecutive-pair"), ConsecutivePairPolicy)
    assert ArrivalOrderPolicy.dispatch_unit == "shot"
    assert ConsecutivePairPolicy.dispatch_unit == "pair"
    for stale in ("minimum-recorded-gap", "drift-aware", "nope"):
        with pytest.raises(ValueError, match="Unknown policy"):
            make_baseline_policy(stale)


# ---------------------------------------------------------------- round-robin workload


def test_round_robin_matchings_partition_all_91_pairs():
    registry = load_compatibility_registry()
    matchings = round_robin_matchings(registry.circuits)
    assert len(matchings) == 13
    assert all(len(m) == 7 for m in matchings)
    for m in matchings:  # perfect matching: every circuit exactly once
        assert sorted(c for pair in m for c in pair) == sorted(registry.circuits)
    pairs = [frozenset(p) for m in matchings for p in m]
    assert len(pairs) == len(set(pairs)) == 91
    assert set(pairs) == {frozenset((p.circuit_a, p.circuit_b)) for p in registry.pairs}


def test_round_robin_matchings_odd_and_tiny():
    assert round_robin_matchings([label(1)]) == ()
    three = round_robin_matchings([label(k) for k in (1, 2, 4)])
    assert len(three) == 3 and all(len(m) == 1 for m in three)
    assert len({frozenset(m[0]) for m in three}) == 3


def test_shot_round_indices_count_per_circuit():
    trace = make_trace([(0, 1, 0), (1, 2, 1), (2, 1, 0), (3, 2, 1), (4, 1, 1)])
    assert shot_round_indices(trace) == {"s0000": 0, "s0001": 0, "s0002": 1, "s0003": 1, "s0004": 2}


def test_build_pair_requests_each_shot_in_at_most_one_request():
    rows = [(float(r * 10 + i), k, (r + i) % 2) for r in range(4) for i, k in enumerate((1, 2, 4, 8))]
    trace = make_trace(rows)
    requests = build_pair_requests(trace, ramsey_registry())
    assert len(requests) == 8  # 2 pairs per round x 4 rounds
    ids = [sid for r in requests for sid in r.shot_ids]
    assert len(ids) == len(set(ids)) == 16
    assert [r.request_id for r in requests] == sorted(r.request_id for r in requests)
    for r in requests:
        assert r.wait_length_first < r.wait_length_second
        assert r.first.circuit != r.second.circuit
        assert r.ready_time == max(r.first.arrival_time, r.second.arrival_time)


def test_build_pair_requests_is_deterministic_and_policy_independent():
    trace = make_trace([(float(i), (1, 2, 4, 8)[i % 4], i % 2) for i in range(16)])
    registry = ramsey_registry()
    assert build_pair_requests(trace, registry) == build_pair_requests(trace, registry)


def test_build_pair_requests_skips_non_compatible_pairs():
    registry = make_registry((1, 2, 4, 8), {(1, 8): CompatibilityStatus.INCOMPATIBLE, (2, 4): CompatibilityStatus.CONDITIONALLY_COMPATIBLE})
    trace = make_trace([(float(i), (1, 2, 4, 8)[i], 0) for i in range(4)])
    requests = build_pair_requests(trace, registry)
    assert requests == ()  # round 0 matches exactly (1,8) and (2,4)


def test_build_pair_requests_uses_supplied_rounds_for_a_subtrace():
    full = make_trace([(float(r * 10 + i), k, 0) for r in range(3) for i, k in enumerate((1, 2))])
    rounds = shot_round_indices(full)
    tail = full[2:]  # drops round 0 entirely
    requests = build_pair_requests(tail, ramsey_registry((1, 2)), rounds)
    assert [r.request_id for r in requests] == ["req-r000001-00", "req-r000002-00"]
    # Without supplied rounds the subtrace is re-indexed from 0.
    assert build_pair_requests(tail, ramsey_registry((1, 2)))[0].request_id == "req-r000000-00"
    with pytest.raises(SimulationError, match="No round index"):
        build_pair_requests(tail, ramsey_registry((1, 2)), {"s0000": 0})


def test_requests_put_smaller_wait_length_first():
    trace = make_trace([(0.0, 8, 1), (1.0, 2, 0)])
    request = make_request("x", trace[0], trace[1])
    assert (request.first.circuit, request.second.circuit) == (label(2), label(8))
    assert request.pair_key == (label(2), label(8))
    with pytest.raises(SimulationError, match="wait length"):
        make_request("y", *make_trace([(0.0, "A", 0), (1.0, "B", 0)]))


# ---------------------------------------------------------------- ordering keys


def _request(rid, t1, k1, i1, t2, k2, i2):
    first = ShotArrival(f"{rid}-a", t1, label(k1), 0, 1, i1, "x")
    second = ShotArrival(f"{rid}-b", t2, label(k2), 1, 1, i2, "x")
    return make_request(rid, first, second)


def test_request_order_key_tie_breakers_in_order():
    base = _request("b", 5.0, 2, 10, 6.0, 4, 11)
    earlier_ready = _request("z", 4.0, 8, 90, 5.5, 16, 91)
    assert request_order_key(earlier_ready) < request_order_key(base)  # 1. earlier arrival time

    lower_k = _request("z", 5.0, 1, 90, 6.0, 4, 91)
    assert request_order_key(lower_k) < request_order_key(base)  # 2. lower (min k, max k)

    lower_src = _request("z", 5.0, 2, 1, 6.0, 4, 2)
    assert request_order_key(lower_src) < request_order_key(base)  # 3. earlier source indices

    same = _request("a", 5.0, 2, 10, 6.0, 4, 11)
    assert request_order_key(same) < request_order_key(base)  # 4. request id


def test_consecutive_pair_key_is_request_order_key():
    request = _request("r", 1.0, 1, 0, 2.0, 2, 1)
    assert ConsecutivePairPolicy().request_key(request) == request_order_key(request)
    assert ArrivalOrderPolicy().shot_key(request.first) == request.first.sort_key


# ---------------------------------------------------------------- slot schedule


def test_build_slot_schedule_spans_trace():
    trace = make_trace([(10.0, 1, 0), (20.0, 2, 0), (30.0, 1, 0), (40.0, 2, 0)])
    schedule = build_slot_schedule(trace)
    assert schedule.total_slots == 4
    assert schedule.start_time == 10.0
    assert schedule.slot_time(3) == pytest.approx(40.0)
    doubled = build_slot_schedule(trace, 2.0)
    assert doubled.total_slots == 8 and doubled.slot_time(7) == pytest.approx(40.0)


def test_build_slot_schedule_edge_cases():
    assert build_slot_schedule(()).total_slots == 0
    single = build_slot_schedule(make_trace([(5.0, 1, 0)]))
    assert single.total_slots == 1 and single.slot_seconds == 1.0
    simultaneous = build_slot_schedule(make_trace([(5.0, 1, 0), (5.0, 2, 0)]))
    assert simultaneous.total_slots == 2 and simultaneous.slot_seconds > 0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(SimulationError, match="slots_per_shot"):
            build_slot_schedule(make_trace([(0.0, 1, 0)]), bad)
