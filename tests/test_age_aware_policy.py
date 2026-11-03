"""Tests for the experimental ``drift-aware-age`` policy, its calibration and its starvation metrics."""

import dataclasses
import math

import numpy as np
import pytest

from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import label, make_registry, ramsey_registry
from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.analysis.age_calibration import CalibrationError, calibrate
from quantum_drift_scheduler.analysis.holdout import split_trace
from quantum_drift_scheduler.analysis.starvation_metrics import overtaking_counts, starvation_summary
from quantum_drift_scheduler.scheduling import (
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    ShotArrival,
    SlotSchedule,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.age_aware_policy import (
    DRIFT_AWARE_AGE,
    AgeConfig,
    AgeConfigError,
    DriftAgePolicy,
    priority_range,
    waiting_time,
)
from quantum_drift_scheduler.scheduling.policies import request_order_key
from quantum_drift_scheduler.scheduling.workload import make_request

KS = (1, 2, 4, 8)
REGISTRY = ramsey_registry(KS)
LOW, HIGH = (1, 2), (4, 8)  # a low-priority and a high-priority circuit pair


class Table:
    """Frozen, read-only priorities keyed by the unordered circuit pair."""

    def __init__(self, scores):
        self._scores = {frozenset((label(a), label(b))): s for (a, b), s in scores.items()}
        self.calls = 0

    def score(self, a, b):
        self.calls += 1
        return self._scores[frozenset((a, b))]


SCORES = Table({LOW: 1.0, HIGH: 2.0, (1, 4): 1.5, (1, 8): 1.2, (2, 4): 1.7, (2, 8): 1.4})


def shot(i, t, k, source=None):
    return ShotArrival(f"s{i:05d}", t, label(k), i % 2, 1, i if source is None else source, "synthetic")


def request(idx, release, pair, outcome_seed=0):
    a, b = shot(2 * idx, release, pair[0]), shot(2 * idx + 1, release, pair[1])
    return make_request(f"req-r{idx:06d}-00", a, b)


def manual_workload(releases_and_pairs, slots, slot_seconds=1.0):
    reqs = tuple(request(i, t, p) for i, (t, p) in enumerate(releases_and_pairs))
    shots = tuple(sorted((s for q in reqs for s in (q.first, q.second)), key=lambda s: s.sort_key))
    return sd.Workload("manual", shots, reqs, SlotSchedule(0.0, slot_seconds, slots))


def saturated_stream(n=40, slot=1.0):
    """One low-priority request at t=0 and a fresh high-priority request every two slots: the device is saturated
    and a strict-priority policy never reaches the low-priority request."""
    items = [(0.0, LOW)] + [(2.0 * i * slot, HIGH) for i in range(n)]
    return manual_workload(items, slots=2 * n + 10, slot_seconds=slot)


# ---------------------------------------------------------------- age function


def test_age_bonus_is_monotone_zero_at_zero_and_rejects_negative_waiting():
    cfg = AgeConfig(horizon_seconds=20.0, priority_range=1.1)
    assert cfg.age_bonus(0.0) == 0.0
    ws = [0.0, 0.1, 1.0, 5.0, 19.9, 20.0, 80.0]
    bonuses = [cfg.age_bonus(w) for w in ws]
    assert all(b2 > b1 for b1, b2 in zip(bonuses, bonuses[1:]))
    assert cfg.age_bonus(20.0) == pytest.approx(1.1)  # the bonus equals the whole priority range at the horizon H
    assert cfg.coefficient_per_second == pytest.approx(1.1 / 20.0)
    for bad in (-1e-9, -5.0, math.nan, math.inf):
        with pytest.raises(AgeConfigError):
            cfg.age_bonus(bad)
    assert waiting_time(10.0, 7.0) == 0.0 and waiting_time(10.0, 12.5) == 2.5  # never negative
    with pytest.raises(AgeConfigError):
        waiting_time(math.nan, 1.0)
    for kw in ({"horizon_seconds": 0.0, "priority_range": 1.0}, {"horizon_seconds": 1.0, "priority_range": -1.0}, {"horizon_seconds": math.inf, "priority_range": 1.0}):
        with pytest.raises(AgeConfigError):
            AgeConfig(**kw)


def test_priority_range_is_the_spread_of_the_frozen_table():
    assert priority_range([0.92, 1.4, 2.05]) == pytest.approx(1.13)
    with pytest.raises(AgeConfigError):
        priority_range([1.0, 1.0])
    with pytest.raises(AgeConfigError):
        priority_range([1.0])


# ---------------------------------------------------------------- ordering semantics


def test_priority_direction_is_preserved_and_matches_the_production_policy():
    pol = DriftAgePolicy(SCORES, AgeConfig(20.0, 1.0))
    lo, hi = request(0, 0.0, LOW), request(1, 0.0, HIGH)
    assert pol.effective_priority(hi, 5.0) > pol.effective_priority(lo, 5.0)  # equal waiting: higher drift score is more urgent
    assert pol.selection_key(hi, 5.0) < pol.selection_key(lo, 5.0)  # lowest key dispatches first
    prod = DriftAwarePairPolicy(SCORES)
    assert prod.request_key(hi) < prod.request_key(lo)  # same direction as the existing policy
    assert pol.selection_key(lo, 0.0)[1:] == request_order_key(lo)  # tie-breakers are the production ones


def test_equal_drift_priorities_favour_the_older_request():
    pol = DriftAgePolicy(SCORES, AgeConfig(20.0, 1.0))
    older, newer = request(0, 0.0, HIGH), request(1, 6.0, HIGH)
    assert pol.effective_priority(older, 10.0) > pol.effective_priority(newer, 10.0)
    assert pol.selection_key(older, 10.0) < pol.selection_key(newer, 10.0)
    wl = manual_workload([(0.0, HIGH), (6.0, HIGH)], slots=30)
    pick = pol.chooser()([0, 1], 10, wl, {})
    assert pick == 0


def test_the_frozen_drift_priority_is_never_mutated():
    pol = DriftAgePolicy(SCORES, AgeConfig(10.0, 1.0))
    q = request(0, 0.0, LOW)
    before = SCORES.score(label(1), label(2))
    for now in (0.0, 3.0, 30.0, 300.0):
        eff, bonus = pol.effective_priority(q, now), pol.age_bonus(q, now)
        assert pol.drift_priority(q) == before and eff - bonus == pytest.approx(before)
    wl = saturated_stream()
    sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, pol.chooser())
    assert SCORES.score(label(1), label(2)) == before and SCORES.score(label(4), label(8)) == 2.0
    with pytest.raises(Exception):
        DriftAgePolicy(Table({LOW: -1.0, HIGH: 2.0, (1, 4): 1, (1, 8): 1, (2, 4): 1, (2, 8): 1}), AgeConfig(1.0, 1.0)).drift_priority(request(0, 0.0, LOW))


def test_priority_is_recalculated_at_each_scheduling_decision():
    pol = DriftAgePolicy(SCORES, AgeConfig(10.0, 1.0))
    q = request(0, 0.0, LOW)
    values = [pol.effective_priority(q, t) for t in (0.0, 2.0, 4.0, 8.0)]
    assert values == sorted(values) and len(set(values)) == 4  # not frozen at request creation
    # a request released at t=0 with the lowest drift score overtakes a fresh high-score request once its age bonus exceeds the score gap
    wl = manual_workload([(0.0, LOW), (14.0, HIGH)], slots=40)
    choose = pol.chooser()
    assert choose([0, 1], 14, wl, {}) == 0  # age 14 s > H: the old low-priority request now outranks the fresh high-priority one
    early = [0, 1]
    wl2 = manual_workload([(0.0, LOW), (3.0, HIGH)], slots=40)
    assert choose(early, 3, wl2, {}) == 1  # at t=3 the age bonus (0.3) is below the score gap (1.0): drift priority still decides


# ---------------------------------------------------------------- starvation behaviour and the H bound


def test_strict_priority_starves_but_the_age_policy_serves_the_low_priority_request():
    wl = saturated_stream(n=60)
    strict = sd.simulate_pair_dispatch(wl, "drift-aware", sd.chooser_from_key(lambda q: (-SCORES.score(q.first.circuit, q.second.circuit),) + request_order_key(q)))
    low_id = wl.requests[0].request_id
    last_high = wl.requests[-1].request_id
    start = {c.request_id: c.execution_time_a for c in strict.comparisons}
    assert start[low_id] > start[last_high]  # strict priority serves it only after EVERY fresh high-priority request, i.e. when the stream ends
    H = 12.0
    aged = sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, DriftAgePolicy(SCORES, AgeConfig(H, 1.0)).chooser())
    c = next(c for c in aged.comparisons if c.request_id == low_id)
    assert H <= c.execution_time_a <= H + 4.0  # served roughly when its bonus equals the priority range
    assert c.execution_time_a < start[low_id] / 2  # far earlier than under strict priority
    assert all(x.slot_b == x.slot_a + 1 for x in aged.comparisons)


def test_a_request_is_only_overtaken_by_requests_released_at_most_H_later():
    rows_release, starts = [], []
    trace_model = true_model(120, gammas=(0.1, 0.25))
    trace = synthetic_shots(trace_model, KS, 120, seed=3)
    base = sd.build_workload("syn", trace, REGISTRY, trace)
    wl = sd.with_capacity(base, 0.8)  # overloaded so that a backlog (and overtaking) occurs
    for H in (2.0, 6.0, 20.0):
        res = sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, DriftAgePolicy(SCORES, AgeConfig(H, 1.0)).chooser())
        by_id = {c.request_id: c for c in res.comparisons}
        releases = [q.ready_time for q in wl.requests]
        starts = [min(by_id[q.request_id].execution_time_a, by_id[q.request_id].execution_time_b) if q.request_id in by_id else None for q in wl.requests]
        ov = overtaking_counts(releases, starts, gap_seconds=H)
        assert sum(ov["beyond_gap"]) == 0
        assert max(ov["counts"]) >= 0


def test_extreme_horizons_recover_the_two_limiting_policies():
    wl = saturated_stream(n=30)
    strict = sd.simulate_pair_dispatch(wl, "drift-aware", sd.chooser_from_key(lambda q: (-SCORES.score(q.first.circuit, q.second.circuit),) + request_order_key(q)))
    huge = sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, DriftAgePolicy(SCORES, AgeConfig(1e12, 1.0)).chooser())
    assert [(c.request_id, c.slot_a) for c in huge.comparisons] == [(c.request_id, c.slot_a) for c in strict.comparisons]  # H -> infinity: drift-aware
    tiny = sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, DriftAgePolicy(SCORES, AgeConfig(1e-9, 1.0)).chooser())
    rel = {q.request_id: q.ready_time for q in wl.requests}
    ordered = [rel[c.request_id] for c in sorted(tiny.comparisons, key=lambda c: c.slot_a)]
    assert ordered == sorted(ordered)  # H -> 0: oldest first (FIFO by release)


# ---------------------------------------------------------------- pair synchronisation and compatibility


def test_pair_atomicity_and_compatibility_are_preserved():
    model = true_model(100, gammas=(0.1, 0.25))
    trace = synthetic_shots(model, KS, 100, seed=5)
    reg = make_registry(KS, not_comparable=[(1, 8), (2, 4)])
    wl = sd.build_workload("syn", trace, reg, trace)
    res = sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, DriftAgePolicy(SCORES, AgeConfig(8.0, 1.0)).chooser())
    ids = {q.request_id for q in wl.requests}
    assert res.comparisons and all(c.request_id in ids for c in res.comparisons)
    assert all(reg.is_eligible(c.circuit_a, c.circuit_b) for c in res.comparisons)  # reviewer_decision only; nothing new is paired
    assert all(c.slot_separation == 1 and c.slot_b == c.slot_a + 1 for c in res.comparisons)
    assert all(c.time_separation == pytest.approx(wl.schedule.slot_seconds) for c in res.comparisons)
    used = [s for c in res.comparisons for s in (c.shot_id_a, c.shot_id_b)]
    assert len(used) == len(set(used)) and len(res.executed) == 2 * len(res.comparisons)
    assert res.workload_requests == len(wl.requests) and res.source_shots == len(wl.shots)


def test_original_policies_are_unchanged_and_the_offline_simulator_matches_production():
    model = true_model(100, gammas=(0.1, 0.25))
    trace = synthetic_shots(model, KS, 100, seed=6)
    wl = sd.build_workload("syn", trace, REGISTRY, trace)
    for name, prod_policy, key in (
        ("consecutive-pair", ConsecutivePairPolicy(), request_order_key),
        ("drift-aware", DriftAwarePairPolicy(SCORES), lambda q: (-SCORES.score(q.first.circuit, q.second.circuit),) + request_order_key(q)),
    ):
        prod = run_simulation(wl.shots, REGISTRY, prod_policy, schedule=wl.schedule, requests=wl.requests)
        rep = sd.simulate_pair_dispatch(wl, name, sd.chooser_from_key(key))
        assert [(c.request_id, c.slot_a, c.slot_b) for c in prod.comparisons] == [(c.request_id, c.slot_a, c.slot_b) for c in rep.comparisons]
    q = wl.requests[3]
    assert DriftAwarePairPolicy(SCORES).request_key(q) == (-SCORES.score(q.first.circuit, q.second.circuit),) + request_order_key(q)
    assert ConsecutivePairPolicy().request_key(q) == request_order_key(q)


# ---------------------------------------------------------------- calibration is training-only


def build_train_workload(trace, reg, boundary_fraction=0.75):
    split = split_trace(trace, boundary_fraction, required_circuits=reg.circuits)
    wl = sd.build_workload("train", split.train, reg, trace)
    return wl, split


def test_calibration_is_training_only_and_ignores_holdout_perturbations():
    model = true_model(200, gammas=(0.1, 0.25))
    trace = synthetic_shots(model, KS, 200, seed=7)
    wl, split = build_train_workload(trace, REGISTRY)
    cal = calibrate(wl, SCORES, split.boundary, candidate_rounds=(1, 2, 4))
    assert cal.selected_rounds in (1, 2, 4) and cal.criterion and cal.priority_range == pytest.approx(1.0)
    assert cal.config.horizon_seconds == pytest.approx(cal.selected_rounds * cal.tau_round_seconds)
    # perturb ONLY the holdout (outcomes and interior timestamps; the boundary is unchanged): the training input, hence the choice, is identical
    perturbed = []
    for s in trace:
        if s.arrival_time >= split.boundary and s is not trace[-1] and s is not trace[0]:
            s = dataclasses.replace(s, outcome=1 - s.outcome, arrival_time=min(s.arrival_time + 0.01, trace[-1].arrival_time - 1e-6))
        perturbed.append(s)
    perturbed.sort(key=lambda s: (s.arrival_time, s.source_index))
    wl2, split2 = build_train_workload(tuple(perturbed), REGISTRY)
    assert split2.boundary == split.boundary and wl2.requests == wl.requests
    cal2 = calibrate(wl2, SCORES, split2.boundary, candidate_rounds=(1, 2, 4))
    assert (cal2.selected_rounds, cal2.selected_horizon_seconds, cal2.candidates) == (cal.selected_rounds, cal.selected_horizon_seconds, cal.candidates)


def test_calibration_refuses_input_that_contains_holdout_requests():
    model = true_model(60, gammas=(0.1, 0.25))
    trace = synthetic_shots(model, KS, 60, seed=8)
    wl = sd.build_workload("all", trace, REGISTRY, trace)
    with pytest.raises(CalibrationError, match="holdout boundary"):
        calibrate(wl, SCORES, boundary=float(trace[-1].arrival_time))
    with pytest.raises(CalibrationError):
        calibrate(wl, SCORES, boundary=float(trace[len(trace) // 2].arrival_time))


# ---------------------------------------------------------------- fairness metrics


def rows_for(releases, starts):
    return [{"release_time": r, "completed": s is not None, "actual_execution_first": s, "actual_execution_second": None if s is None else s + 1.0}
            for r, s in zip(releases, starts)]


def test_starvation_metrics_include_the_observed_age_of_unfinished_requests():
    # cutoff 200 s, slot 1 s: request 3 released at 100 s never ran (age 100, not served in time); request 4 released at 199.5 s (age 0.5 s: truncated)
    rows = rows_for([0.0, 10.0, 20.0, 100.0, 199.5], [5.0, 45.0, 91.0, None, None])
    m = starvation_summary(rows, cutoff=200.0, slot_seconds=1.0)
    assert m["requests"] == 5 and m["completed_requests"] == 3 and m["unfinished_requests"] == 2
    assert m["wait_completed_max_s"] == pytest.approx(71.0) and m["wait_incl_unfinished_max_s"] == pytest.approx(100.0)
    assert m["unfinished_age_max_s"] == pytest.approx(100.0) and m["unfinished_age_mean_s"] == pytest.approx((100.0 + 0.5) / 2)
    assert m["unfinished_truncated_by_horizon"] == 1 and m["unfinished_not_served_in_time"] == 1
    assert (m["waiting_over_10s_incl_unfinished"], m["waiting_over_30s_incl_unfinished"], m["waiting_over_60s_incl_unfinished"]) == (3, 3, 2)
    assert (m["waiting_over_10s_completed"], m["waiting_over_30s_completed"], m["waiting_over_60s_completed"]) == (2, 2, 1)
    assert m["wait_incl_unfinished_p99_s"] > m["wait_completed_p99_s"]


def test_overtaking_counts_are_established_from_actual_start_times():
    # r0 (released 0) starts last; r1, r2 (released later) start first; r3 (released 2) never starts
    c = overtaking_counts([0.0, 1.0, 1.5, 2.0], [30.0, 10.0, 20.0, None])["counts"]
    assert c == [2, 0, 0, 0]  # r0 is overtaken by r1 and r2 (released later, started earlier); nobody started that was released after r3
    # an unfinished request is overtaken by every started request released after it
    assert overtaking_counts([0.5, 1.0, 1.5], [None, 10.0, 20.0])["counts"] == [2, 0, 0]
    # requests released together never overtake one another
    assert overtaking_counts([3.0, 3.0], [9.0, 5.0])["counts"] == [0, 0]
