"""Focused tests of the ACTUAL priority wiring in ``run_holdout_evaluation``.

These use spies on the real functions (``learn_pair_priorities``, ``run_simulation``,
``PriorityTable.score``) and small synthetic traces, so they verify what the code does, not what its
docstrings say. Nothing here touches the real dataset or any result file.
"""

import builtins
import dataclasses

import numpy as np
import pytest

import quantum_drift_scheduler.analysis.priorities as priorities_module
import quantum_drift_scheduler.evaluation as evaluation_module
from quantum_drift_scheduler.analysis.holdout import split_trace
from quantum_drift_scheduler.analysis.priorities import (
    LeakageError,
    PriorityError,
    PriorityTable,
    learn_pair_priorities,
    priority_table_to_dict,
)
from quantum_drift_scheduler.evaluation import run_holdout_evaluation
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    SlotSchedule,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.workload import make_request
from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import label, make_registry, make_trace, ramsey_registry

KS = (1, 16, 256, 2048)
NOT_COMPARABLE = ((1, 2048),)


def inputs(seed=3, rounds=2500, not_comparable=NOT_COMPARABLE):
    model = true_model(rounds, gammas=(0.1, 0.3), decay_rate=2e-5)
    return synthetic_shots(model, KS, rounds, seed=seed), make_registry(KS, not_comparable=not_comparable)


def pair_key(a, b):
    return frozenset((label(a), label(b)))


@pytest.fixture(scope="module")
def traced():
    """One real run with spies on the actual calls (restored afterwards)."""
    trace, registry = inputs()
    log = {"learn": [], "scores": [], "policies": [], "current": None}
    real_learn = evaluation_module.learn_pair_priorities
    real_run = evaluation_module.run_simulation
    real_score = PriorityTable.score

    def learn_spy(train_shots, reg, **kwargs):
        log["learn"].append((tuple(train_shots), reg, kwargs))
        return real_learn(train_shots, reg, **kwargs)

    def run_spy(shots, reg, policy, **kwargs):
        log["policies"].append(policy)
        log["current"] = policy.name
        try:
            return real_run(shots, reg, policy, **kwargs)
        finally:
            log["current"] = None

    def score_spy(self, circuit_a, circuit_b):
        value = real_score(self, circuit_a, circuit_b)
        log["scores"].append((log["current"], frozenset((circuit_a, circuit_b)), value, id(self)))
        return value

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(evaluation_module, "learn_pair_priorities", learn_spy)
        mp.setattr(evaluation_module, "run_simulation", run_spy)
        mp.setattr(PriorityTable, "score", score_spy)
        ev = run_holdout_evaluation(trace, registry)
    return trace, registry, ev, log


# ---------------------------------------------------------------- what is passed to the priority calculation


def test_priorities_are_learned_from_exactly_the_complete_training_partition(traced):
    trace, registry, ev, log = traced
    split = split_trace(trace, 0.75, required_circuits=registry.eligible_circuits())
    assert len(log["learn"]) == 1  # recalculated once for this run
    train_shots, reg, kwargs = log["learn"][0]
    assert reg is registry  # the current registry's decisions
    assert [s.shot_id for s in train_shots] == [s.shot_id for s in split.train]  # the complete, ordered partition
    assert len(train_shots) == ev.priorities.metadata.train_shot_count == len(split.train)
    assert kwargs["boundary"] == split.boundary and kwargs["window_origin"] == split.min_timestamp
    assert max(s.arrival_time for s in train_shots) < kwargs["boundary"]  # nothing at or after the boundary
    assert {s.shot_id for s in train_shots}.isdisjoint(s.shot_id for s in split.holdout)  # no holdout shot
    # not a rolling subset: the first and the last training shots are both present
    assert train_shots[0].shot_id == split.train[0].shot_id and train_shots[-1].shot_id == split.train[-1].shot_id
    assert ev.priorities.metadata.train_start == split.train[0].arrival_time
    assert ev.priorities.metadata.train_end == split.train[-1].arrival_time


def test_the_table_is_fresh_for_this_run(traced):
    trace, registry, ev, _ = traced
    split = split_trace(trace, 0.75, required_circuits=registry.eligible_circuits())
    ev.priorities.assert_matches(split.train, registry)  # fingerprints: this run's training data and decisions
    assert ev.priorities.metadata.training_fingerprint == priorities_module.training_fingerprint(split.train)
    assert ev.priorities.metadata.eligible_pairs_fingerprint == priorities_module.eligible_pairs_fingerprint(registry)


# ---------------------------------------------------------------- who uses the mapping


def test_only_the_drift_aware_policy_holds_or_uses_the_priority_mapping(traced):
    _, _, ev, log = traced
    arrival, consecutive, drift = log["policies"]
    assert [p.name for p in log["policies"]] == ["arrival-order", "consecutive-pair", "drift-aware"]
    assert isinstance(arrival, ArrivalOrderPolicy) and isinstance(consecutive, ConsecutivePairPolicy)
    assert drift._priorities is ev.priorities  # the one table of this run
    for baseline in (arrival, consecutive):
        assert not hasattr(baseline, "_priorities") and not hasattr(baseline, "score")
    assert log["scores"], "the drift-aware policy should have read priorities"
    assert {caller for caller, _, _, _ in log["scores"]} == {"drift-aware"}  # no other policy ever read a score
    assert {table_id for _, _, _, table_id in log["scores"]} == {id(ev.priorities)}  # always the same table


def test_the_mapping_is_frozen_while_the_policies_execute(traced):
    _, _, ev, log = traced
    seen = {}
    for _, key, value, _ in log["scores"]:
        assert seen.setdefault(key, value) == value  # a pair never changed score during the run
    for key, value in seen.items():
        a, b = sorted(key)
        assert ev.priorities.score(a, b) == value  # and the final table still holds those values
    assert isinstance(ev.priorities.priorities, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.priorities.priorities = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.priorities.priorities[0].priority_score = 9.9
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.priorities.metadata.boundary = 0.0


def test_only_comparable_pairs_are_assigned_priorities_or_scheduled(traced):
    _, registry, ev, log = traced
    eligible = {frozenset((p.circuit_a, p.circuit_b)) for p in registry.eligible_pairs()}
    assert {p.key for p in ev.priorities.priorities} == eligible and len(eligible) == 5
    assert all(p.reviewer_decision == "Comparable" for p in ev.priorities.priorities)
    assert {key for _, key, _, _ in log["scores"]} <= eligible
    with pytest.raises(PriorityError, match="No learned priority"):
        ev.priorities.score(label(1), label(2048))  # the Not comparable pair has no priority
    assert all(frozenset(r.pair_key) in eligible for r in ev.requests)
    for run in ev.runs:
        assert all(frozenset(c.pair_key) in eligible for c in run.result.comparisons)


def test_all_policies_get_the_same_workload_arrivals_budget_and_capacity(traced):
    _, _, ev, _ = traced
    results = [r.result for r in ev.runs]
    source_ids = {s.shot_id for s in ev.source_shots}
    assert len({r.schedule for r in results}) == 1 and results[0].schedule == ev.schedule  # one device capacity/budget
    assert len({r.source_shots for r in results}) == 1 == len({r.workload_requests for r in results})
    for r in results:  # every policy saw exactly the same shots and the same arrival times
        assert {e.shot_id for e in r.executed} | {s.shot_id for s in r.unexecuted_shots} == source_ids
    arrival_times = [{e.shot_id: e.arrival_time for e in r.executed} for r in results]
    common = set(arrival_times[0]) & set(arrival_times[1]) & set(arrival_times[2])
    assert common and all(len({a[sid] for a in arrival_times}) == 1 for sid in common)


# ---------------------------------------------------------------- holdout cannot influence priorities


def test_holdout_outcomes_and_composition_cannot_change_the_priorities(traced):
    trace, registry, ev, _ = traced
    boundary = ev.split.boundary
    base = priority_table_to_dict(ev.priorities)
    rng = np.random.default_rng(0)
    redrawn = tuple(dataclasses.replace(s, outcome=int(rng.integers(0, 2))) if s.arrival_time >= boundary else s for s in trace)
    flipped = tuple(dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time >= boundary else s for s in trace)
    last = len(trace) - 1  # keep the final shot: it defines the time span, hence the boundary
    # drop every other acquisition round of the holdout (all circuits keep half of their holdout shots)
    thinned = tuple(s for i, s in enumerate(trace)
                    if s.arrival_time < boundary or (i // len(KS)) % 2 == 0 or i == last)
    assert len(thinned) < len(trace) and thinned[-1] == trace[-1]
    for variant in (redrawn, flipped, thinned):
        other = run_holdout_evaluation(variant, registry)
        assert other.split.boundary == boundary and other.split.train == ev.split.train
        assert priority_table_to_dict(other.priorities) == base
        assert other.holdout_fingerprint != ev.holdout_fingerprint  # the holdout really was different


def test_holdout_reaches_the_priorities_only_through_the_time_span_that_defines_the_boundary(traced):
    trace, registry, ev, _ = traced
    later = dataclasses.replace(trace[-1], arrival_time=trace[-1].arrival_time + 2000.0)  # a later final shot
    moved = run_holdout_evaluation(trace[:-1] + (later,), registry)
    assert moved.split.boundary > ev.split.boundary  # a longer experiment places the 75% boundary later
    assert moved.priorities.metadata.train_shot_count > ev.priorities.metadata.train_shot_count
    assert moved.priorities.metadata.training_fingerprint != ev.priorities.metadata.training_fingerprint


def test_a_holdout_shot_handed_to_the_priority_calculation_is_rejected(traced):
    trace, registry, ev, _ = traced
    split = ev.split
    with pytest.raises(LeakageError, match="only training data may be used"):
        learn_pair_priorities(split.train + split.holdout[:1], registry, boundary=split.boundary,
                              window_origin=split.min_timestamp, train_fraction=0.75, window_seconds=600.0)


def test_no_file_is_read_when_a_run_calculates_its_priorities(monkeypatch):
    trace, registry = inputs(rounds=1500)
    real_open = builtins.open

    def guarded(file, *args, **kwargs):
        if str(file).lower().endswith((".csv", ".json", ".txt", ".pkl")):
            raise AssertionError(f"a scheduler run must not read {file}")
        return real_open(file, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("a scheduler run must not load a saved priority table")

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(priorities_module, "load_priority_table", forbidden)
    ev = run_holdout_evaluation(trace, registry)
    assert len(ev.priorities.priorities) == 5


# ---------------------------------------------------------------- recalculation and staleness


def test_a_new_run_recalculates_and_a_previous_table_cannot_be_reused(monkeypatch, traced):
    trace, registry, first, _ = traced
    calls = []
    real = evaluation_module.learn_pair_priorities
    monkeypatch.setattr(evaluation_module, "learn_pair_priorities",
                        lambda train, reg, **kw: calls.append(len(train)) or real(train, reg, **kw))
    second = run_holdout_evaluation(trace, registry)  # identical inputs: recomputed, equal, but a new object
    assert calls == [len(first.split.train)] and second.priorities == first.priorities
    assert second.priorities is not first.priorities

    other_trace, _ = inputs(seed=11)  # a different run
    third = run_holdout_evaluation(other_trace, registry)
    assert len(calls) == 2 and third.priorities != first.priorities
    assert third.priorities.metadata.training_fingerprint != first.priorities.metadata.training_fingerprint
    with pytest.raises(PriorityError, match="different training data"):  # the old table is rejected for the new run
        first.priorities.assert_matches(third.split.train, registry)
    with pytest.raises(PriorityError, match="different training data"):
        third.priorities.assert_matches(first.split.train, registry)


def test_run_holdout_evaluation_refuses_a_stale_table_injected_at_the_call_site(monkeypatch, traced):
    _, registry, previous_run, _ = traced
    other_trace, _ = inputs(seed=11)  # a different run whose training data differs
    # If the calculation ever handed back an earlier run's table, the run must fail rather than use it.
    monkeypatch.setattr(evaluation_module, "learn_pair_priorities", lambda train, reg, **kw: previous_run.priorities)
    with pytest.raises(PriorityError, match="Stale priorities: the table was computed from different training data"):
        run_holdout_evaluation(other_trace, registry)
    # ... and a table built for different registry decisions is refused for the same reason.
    narrower = make_registry(KS, not_comparable=((1, 2048), (16, 256)))
    with pytest.raises(PriorityError, match="Comparable pairs"):
        run_holdout_evaluation(previous_run.split.train + previous_run.split.holdout, narrower)


def test_a_table_for_different_registry_decisions_is_rejected(traced):
    _, registry, ev, _ = traced
    swapped = make_registry(KS, not_comparable=((16, 256),))  # same number of pairs, different eligible set
    with pytest.raises(PriorityError, match="Comparable pairs"):
        ev.priorities.assert_matches(ev.split.train, swapped)
    assert (priorities_module.eligible_pairs_fingerprint(swapped)
            != priorities_module.eligible_pairs_fingerprint(registry))
    shrunk = run_holdout_evaluation(*(lambda t, r: (t, make_registry(KS, not_comparable=((1, 2048), (16, 256)))))(*inputs()))
    assert len(shrunk.priorities.priorities) == 4  # the new decisions give a new, smaller table
    assert shrunk.priorities.metadata.eligible_pairs_fingerprint != ev.priorities.metadata.eligible_pairs_fingerprint


# ---------------------------------------------------------------- the other policies ignore priority values


def test_arrival_order_and_consecutive_pair_results_do_not_depend_on_priority_values(traced, monkeypatch):
    trace, registry, base, _ = traced
    real = evaluation_module.learn_pair_priorities

    def zeroed(train_shots, reg, **kwargs):
        table = real(train_shots, reg, **kwargs)
        return dataclasses.replace(table, priorities=tuple(dataclasses.replace(p, priority_score=0.0) for p in table.priorities))

    monkeypatch.setattr(evaluation_module, "learn_pair_priorities", zeroed)
    altered = run_holdout_evaluation(trace, registry)
    assert altered.priorities.positive_count == 0 and base.priorities.positive_count == 5
    for index in (0, 1):  # arrival-order and consecutive-pair: byte-for-byte identical results
        assert altered.runs[index].result == base.runs[index].result


def test_priority_values_change_only_the_drift_aware_dispatch_order():
    class Table:
        def __init__(self, scores):
            self._scores = {frozenset((label(a), label(b))): v for (a, b), v in scores.items()}

        def score(self, circuit_a, circuit_b):
            return self._scores[frozenset((circuit_a, circuit_b))]

    shots = make_trace([(0.0, 1, 1), (0.0, 2, 0), (0.0, 4, 1), (0.0, 8, 0)])
    requests = tuple(make_request(f"r{i}", shots[a], shots[b]) for i, (a, b) in enumerate([(0, 1), (2, 3)]))
    grid = SlotSchedule(0.0, 1.0, 4)
    favour_first = Table({(1, 2): 9.0, (4, 8): 1.0})
    favour_second = Table({(1, 2): 1.0, (4, 8): 9.0})

    def order(policy):
        result = run_simulation(shots, ramsey_registry(), policy, grid, requests)
        return [c.request_id for c in result.comparisons], result

    assert order(DriftAwarePairPolicy(favour_first))[0] == ["r0", "r1"]
    assert order(DriftAwarePairPolicy(favour_second))[0] == ["r1", "r0"]  # priorities reorder the drift-aware policy
    for make_policy in (ArrivalOrderPolicy, ConsecutivePairPolicy):  # ... and nothing else
        assert order(make_policy())[1] == order(make_policy())[1]
