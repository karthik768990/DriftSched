"""Tests for the scheduler bottleneck diagnostics (synthetic traces only; production scheduler untouched)."""

import numpy as np
import pytest

from _sched_helpers import make_trace, ramsey_registry
from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.scheduling import ConsecutivePairPolicy, DriftAwarePairPolicy, run_simulation

KS = (1, 2, 4, 8)
REGISTRY = ramsey_registry(KS)


class Table:
    """Distinct, deterministic priorities (higher = sooner), keyed by the unordered circuit pair."""

    def __init__(self, scores=None):
        self.scores = scores

    def score(self, a, b):
        if self.scores is None:
            return 1.0 + (sum(map(ord, a + b)) % 97) / 50.0
        return self.scores


def bursty_trace(rounds=60, period=2.2):
    """One shot per circuit per round, 0.1 s apart, a round every ``period`` seconds."""
    rows = []
    for r in range(rounds):
        for j, k in enumerate(KS):
            rows.append((r * period + 0.1 * j, k, (r + j) % 2))
    return make_trace(rows)


@pytest.fixture(scope="module")
def world():
    trace = bursty_trace()
    work = sd.build_workload("synthetic", trace, REGISTRY, trace)
    pri = Table()
    refs = sd.run_reference_policies(work, REGISTRY, pri)
    return work, pri, refs


def test_replica_simulator_reproduces_production_executions(world):
    work, pri, refs = world
    for name, policy in (("consecutive-pair", ConsecutivePairPolicy()), ("drift-aware", DriftAwarePairPolicy(pri))):
        rep = sd.simulate_pair_dispatch(work, name, sd.chooser_from_key(policy.request_key))
        prod = refs[name]
        assert [(c.request_id, c.slot_a, c.slot_b) for c in rep.comparisons] == [(c.request_id, c.slot_a, c.slot_b) for c in prod.comparisons]
        assert rep.idle_slots == prod.idle_slots and len(rep.executed) == len(prod.executed)


def test_waiting_reconstruction_matches_production_for_every_policy(world):
    work, _, refs = world
    for name, res in refs.items():
        w = sd.waiting_breakdown(work, res)
        assert w["reconstruction_matches_production"]
        parts = [w[f"mean_{k}_seconds"] for k in ("partner", "alignment", "capacity", "overtaking", "atomic")]
        assert all(p >= -1e-12 for p in parts) and sum(parts) == pytest.approx(w["mean_wait_seconds"])
    arrival = sd.waiting_breakdown(work, refs["arrival-order"])
    assert arrival["mean_partner_seconds"] == 0 and arrival["mean_atomic_seconds"] == 0
    pair = sd.waiting_breakdown(work, refs["consecutive-pair"])
    assert pair["mean_atomic_seconds"] == pytest.approx(work.schedule.slot_seconds / 2)  # half the shots wait one extra slot
    assert pair["mean_overtaking_seconds"] == 0  # FIFO by ready time never serves later work first


def test_utilisation_accounts_for_every_slot_and_idle_cause(world):
    work, _, refs = world
    for name, res in refs.items():
        u = sd.utilisation(work, res)
        assert u["busy_slots"] + u["idle_slots"] == u["total_slots"]
        assert u["idle_end_of_grid"] + u["idle_scheduler_induced_partner_wait"] + u["idle_unavoidable_nothing_pending"] == u["idle_slots"]
        assert u["incomplete_requests"] == u["workload_requests"] - u["completed_comparisons"]
        assert u["utilisation_pct"] == pytest.approx(100 * u["busy_slots"] / u["total_slots"])
        if name == "arrival-order":
            assert u["idle_scheduler_induced_partner_wait"] == 0 and u["idle_end_of_grid"] == 0


def test_ready_queue_statistics_are_consistent(world):
    work, _, refs = world
    a = sd.ready_queue_stats(work, refs["consecutive-pair"])
    b = sd.ready_queue_stats(work, refs["drift-aware"])
    assert sum(a["histogram_ready_at_decision"].values()) == a["decisions"] == len(refs["consecutive-pair"].comparisons)
    assert min(a["histogram_ready_at_decision"]) >= 1
    assert a["pct_decisions_one_ready"] + a["pct_decisions_multiple_ready"] == pytest.approx(100.0)
    assert a["mean"] == pytest.approx(b["mean"])  # same arrivals, same work: the queue-size distribution is order-independent
    s = sd.shot_queue_stats(work, refs["arrival-order"])
    assert s["max"] >= 1 and s["decisions"] == len(refs["arrival-order"].executed)


def test_priority_effectiveness_counts_are_consistent(world):
    work, pri, refs = world
    out, rows = sd.priority_effectiveness(work, refs["drift-aware"], refs["consecutive-pair"], pri)
    assert out["decisions_where_priority_choice_differs_from_fifo"] <= out["decisions_with_multiple_ready"] <= out["scheduling_decisions"]
    assert out["identical_sequence"] == (out["positions_where_execution_sequence_differs"] == 0)
    assert len(rows) == out["requests_compared"]
    assert out["mean_queue_wait_drift_aware"] == pytest.approx(out["mean_queue_wait_consecutive"])  # work conservation, equal service


def test_equal_priorities_make_drift_aware_identical_to_consecutive_pair():
    trace = bursty_trace()
    work = sd.build_workload("synthetic", trace, REGISTRY, trace)
    pri = Table(scores=1.0)
    refs = sd.run_reference_policies(work, REGISTRY, pri)
    out, _ = sd.priority_effectiveness(work, refs["drift-aware"], refs["consecutive-pair"], pri)
    assert out["identical_sequence"] and out["decisions_where_priority_choice_differs_from_fifo"] == 0
    assert out["execution_shift_abs_max_seconds"] == 0


def test_candidates_never_drop_requests_or_change_the_workload(world):
    work, pri, _ = world
    choosers = [sd.chooser_oldest_member_first(), sd.chooser_aging_priority(pri, 2), sd.chooser_batched(2, 2), sd.chooser_batched(3, 4)]
    all_ids = {q.request_id for q in work.requests}
    for ch in choosers:
        res = sd.simulate_pair_dispatch(work, "candidate", ch)
        ids = [c.request_id for c in res.comparisons]
        assert len(ids) == len(set(ids)) and set(ids) <= all_ids
        assert len(res.comparisons) + (len(all_ids) - len(ids)) == len(work.requests)
        assert all(REGISTRY.is_eligible(c.circuit_a, c.circuit_b) for c in res.comparisons)  # reviewer_decision only
        assert all(c.slot_separation == 1 for c in res.comparisons)  # atomic adjacent slots preserved
        executed_ids = [e.shot_id for e in res.executed]
        assert len(executed_ids) == len(set(executed_ids)) and len(res.executed) + res.idle_slots == work.schedule.total_slots
        assert res.schedule == work.schedule and res.workload_requests == len(work.requests)


def test_batching_can_idle_with_ready_work_but_fifo_candidate_matches_production(world):
    work, pri, refs = world
    batched = sd.simulate_pair_dispatch(work, "b", sd.chooser_batched(4, 10))
    assert batched.idle_slots > refs["consecutive-pair"].idle_slots
    edf = sd.simulate_pair_dispatch(work, "e", sd.chooser_oldest_member_first())
    assert len(edf.comparisons) == len(refs["consecutive-pair"].comparisons)


def test_censored_waits_count_requests_that_never_ran():
    trace = bursty_trace(rounds=40, period=1.0)  # offered load well above capacity: a backlog remains
    work = sd.with_capacity(sd.build_workload("overload", trace, REGISTRY, trace), 0.5)
    res = run_simulation(work.shots, REGISTRY, ConsecutivePairPolicy(), schedule=work.schedule, requests=work.requests)
    done = sd.request_queue_waits(work, res)
    allw = sd.request_waits_censored(work, res)
    assert allw.size == len(work.requests) > done.size and np.all(allw >= 0)
    assert allw.max() >= done.max()


def test_capacity_change_keeps_shots_and_requests_identical(world):
    work, _, _ = world
    fast = sd.with_capacity(work, 2.0)
    assert fast.shots == work.shots and fast.requests == work.requests
    assert fast.schedule.total_slots == 2 * work.schedule.total_slots and fast.schedule.start_time == work.schedule.start_time


def test_first_slot_at_or_after_matches_the_admission_test(world):
    work, _, _ = world
    sched = work.schedule
    for q in work.requests[:40]:
        s = sd.first_slot_at_or_after(sched, q.ready_time)
        assert sched.slot_time(s) >= q.ready_time and (s == 0 or sched.slot_time(s - 1) < q.ready_time)


def test_arrival_structure_is_deterministic_and_sensible(world):
    work, _, _ = world
    a, b = sd.arrival_structure(work), sd.arrival_structure(work)
    assert a == b and a["bursts"] == 60 and a["burst_size_mean"] == pytest.approx(4.0)
    assert a["offered_load_shots_x_slot_over_span"] == pytest.approx(1.0, abs=0.05)
