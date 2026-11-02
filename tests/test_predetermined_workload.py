"""Tests for the predetermined workload, its serialization, the fairness audit and the scoring of replays."""

import dataclasses
import json
from collections import Counter

import numpy as np
import pytest

from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import make_registry, make_trace, ramsey_registry
from quantum_drift_scheduler.data.compatibility import CompatibilityStatus
from quantum_drift_scheduler.modeling.predetermined_evaluation import drift_fingerprint, policy_summary, trace_rows
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    build_slot_schedule,
    prepare_scheduler_input,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.predetermined_workload import (
    Device,
    FairnessAuditError,
    PlannedRequest,
    SimulatorInput,
    WorkloadError,
    audit_fairness,
    build_predetermined_workload,
    load_workload,
    make_policy_input,
    save_workload,
    to_simulator_input,
)

KS = (1, 2, 4, 8)
REGISTRY = ramsey_registry(KS)
POLICIES = ("arrival-order", "consecutive-pair", "drift-aware")


class Table:
    def score(self, a, b):
        return 1.0 + (sum(map(ord, a + b)) % 97) / 50.0


@pytest.fixture(scope="module")
def model():
    return true_model(80, gammas=(0.1, 0.25))


@pytest.fixture(scope="module")
def parts(model):
    trace = synthetic_shots(model, KS, 80, seed=2)
    prepared = prepare_scheduler_input(trace, REGISTRY, round_reference=trace)
    schedule = build_slot_schedule(prepared.source_shots)
    return trace, prepared, schedule


@pytest.fixture(scope="module")
def workload(parts):
    _, prepared, schedule = parts
    return build_predetermined_workload(prepared.requests, schedule, REGISTRY, {"partition": "synthetic"})


def run_all(sim: SimulatorInput, registry=REGISTRY):
    pols = {"arrival-order": ArrivalOrderPolicy(), "consecutive-pair": ConsecutivePairPolicy(), "drift-aware": DriftAwarePairPolicy(Table())}
    return {n: run_simulation(sim.shots, registry, p, schedule=sim.schedule, requests=sim.requests) for n, p in pols.items()}


def make_inputs(workload, sim, registry=REGISTRY, drift="d" * 64):
    return {n: make_policy_input(n, sim, registry, drift, [r.planned_time for r in workload.requests]) for n in POLICIES}


def audit(workload, inputs, results, drift="d" * 64, refs=None):
    return audit_fairness(workload, inputs, results, drift, refs if refs is not None else [r.planned_time for r in workload.requests])


# ---------------------------------------------------------------- determinism, persistence, fingerprint


def test_generation_is_deterministic(parts, workload):
    _, prepared, schedule = parts
    again = build_predetermined_workload(prepared.requests, schedule, REGISTRY, {"partition": "synthetic"})
    assert again == workload and again.fingerprint == workload.fingerprint
    assert workload.config["random_seed"] is None and "no random" in workload.config["construction_method"]


def test_serialization_round_trip_and_tamper_detection(workload, tmp_path):
    path = tmp_path / "workload.json"
    fp = save_workload(workload, path)
    loaded = load_workload(path)
    assert loaded == workload and loaded.fingerprint == fp == workload.fingerprint
    data = json.loads(path.read_text())
    assert data["summary"] == {"requests": workload.request_count, "eligible_pairs": workload.eligible_pair_count, "total_shot_budget": workload.total_shot_budget}
    data["requests"][3]["planned_time"] += 1.0
    path.write_text(json.dumps(data))
    with pytest.raises(WorkloadError, match="fingerprint mismatch"):
        load_workload(path)


def test_fingerprint_is_stable_and_sensitive_to_every_policy_visible_field(workload):
    fp = workload.fingerprint
    assert len(fp) == 64 and workload.fingerprint == fp
    assert dataclasses.replace(workload, config={"different": "config"}).fingerprint == fp  # description is not content

    def with_request(i, **kw):
        reqs = list(workload.requests)
        reqs[i] = dataclasses.replace(reqs[i], **kw)
        return dataclasses.replace(workload, requests=tuple(reqs)).fingerprint

    r = workload.requests[5]
    for kw in ({"release_time": r.release_time + 0.01}, {"planned_time": r.planned_time + 0.01}, {"circuit_second": "Gx(Gw)^99Gy"},
               {"shots_first": 2}, {"outcome_first": 1 - r.outcome_first}):
        assert with_request(5, **kw) != fp
    assert dataclasses.replace(workload, device=Device(workload.device.start_time, workload.device.slot_seconds * 1.01, workload.device.total_slots)).fingerprint != fp
    assert dataclasses.replace(workload, requests=workload.requests[:-1]).fingerprint != fp


# ---------------------------------------------------------------- eligibility


def test_only_comparable_pairs_are_eligible_and_status_is_not_the_authority(model):
    not_comparable = [(1, 8), (2, 4)]
    reg = make_registry(KS, overrides={(1, 2): CompatibilityStatus.INCOMPATIBLE if hasattr(CompatibilityStatus, "INCOMPATIBLE") else CompatibilityStatus.COMPATIBLE}, not_comparable=not_comparable)
    trace = synthetic_shots(model, KS, 40, seed=1)
    prepared = prepare_scheduler_input(trace, reg, round_reference=trace)
    wl = build_predetermined_workload(prepared.requests, build_slot_schedule(prepared.source_shots), reg)
    assert {tuple(sorted(p)) for p in wl.eligible_pairs} == {tuple(sorted((a, b))) for a, b in
        [(x.circuit_a, x.circuit_b) for x in reg.eligible_pairs()]}
    excluded = {frozenset(("Gx(Gw)Gy", "Gx(Gw)^8Gy")), frozenset(("Gx(Gw)^2Gy", "Gx(Gw)^4Gy"))}
    assert all(frozenset(r.pair_key) not in excluded for r in wl.requests) and wl.request_count > 0
    ok = prepared.requests[0]
    bad = dataclasses.replace(ok, second=dataclasses.replace(ok.second, circuit="Gx(Gw)^8Gy"), first=dataclasses.replace(ok.first, circuit="Gx(Gw)Gy"))
    with pytest.raises(WorkloadError, match="Comparable"):
        build_predetermined_workload([bad], build_slot_schedule(prepared.source_shots), reg)


# ---------------------------------------------------------------- timestamps


def test_release_and_planned_time_semantics_and_immutability(parts, workload):
    _, prepared, schedule = parts
    by_round = {}
    for q, r in zip(prepared.requests, workload.requests):
        by_round.setdefault(r.round_id, []).append((q, r))
    for rnd, items in by_round.items():
        earliest = min(min(q.first.arrival_time, q.second.arrival_time) for q, _ in items)
        assert {r.release_time for _, r in items} == {earliest}  # one release time per round
        for q, r in items:
            assert r.planned_time == q.ready_time == max(q.first.arrival_time, q.second.arrival_time)
            assert schedule.start_time <= r.release_time <= r.planned_time
            assert (r.historical_time_first, r.historical_time_second) == (q.first.arrival_time, q.second.arrival_time)
    with pytest.raises(dataclasses.FrozenInstanceError):
        workload.requests[0].release_time = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        workload.requests[0].planned_time = 0.0
    assert "execution" not in " ".join(PlannedRequest.__dataclass_fields__).replace("execution_time", "")  # no executed times stored


def test_actual_execution_times_are_policy_produced_and_inputs_unchanged(workload):
    sim = to_simulator_input(workload)
    before = workload.fingerprint
    results = run_all(sim)
    assert workload.fingerprint == before
    release = {r.request_id: r.release_time for r in workload.requests}
    seps = {}
    for name, res in results.items():
        slot_s = sim.schedule.slot_seconds
        for e in res.executed:
            assert e.execution_time == sim.schedule.slot_time(e.slot)  # on the slot grid, set by the simulation
        for c in res.comparisons:
            assert c.arrival_time_a == c.arrival_time_b == release[c.request_id]
            assert min(c.execution_time_a, c.execution_time_b) >= release[c.request_id]
        seps[name] = np.mean([c.time_separation for c in res.comparisons])
        assert slot_s > 0
    assert seps["consecutive-pair"] == pytest.approx(sim.schedule.slot_seconds) and seps["arrival-order"] > seps["consecutive-pair"]
    first = {n: [c.execution_time_a for c in r.comparisons] for n, r in results.items()}
    assert first["arrival-order"] != first["consecutive-pair"]  # decisions differ, inputs do not


def test_equal_shot_budgets_across_policies(workload):
    sim = to_simulator_input(workload)
    assert workload.total_shot_budget == 2 * workload.request_count == len(sim.shots) == sum(s.repetitions for s in sim.shots)
    for res in run_all(sim).values():
        assert len(res.executed) + len(res.unexecuted_shots) == workload.total_shot_budget
        assert res.source_shots == len(sim.shots) and res.workload_requests == workload.request_count


# ---------------------------------------------------------------- identical input and the audit


def test_all_policies_receive_identical_input_and_the_audit_passes(workload):
    sim = to_simulator_input(workload)
    inputs = make_inputs(workload, sim)
    a = audit(workload, inputs, run_all(sim))
    assert a.passed and a.require() is a
    assert len({c.name for c in a.checks}) == 13 and len(a.checks) == 13 * 3
    assert all(inputs[n].simulator_input is sim for n in POLICIES)  # one frozen input, not one per policy
    assert a.workload_fingerprint == workload.fingerprint


def mutated(sim, **changes):
    return dataclasses.replace(sim, **changes)


@pytest.mark.parametrize("case,expected", [
    ("release", "release times identical"), ("planned", "planned measurement times identical"),
    ("drop_request", "request count identical"), ("swap_circuit", "pair identities and multiplicities identical"),
    ("shot_count", "shot requirements identical"), ("device", "device capacity and slot duration identical"),
    ("drift", "drift inputs identical"), ("reference", "comparison references identical"),
    ("registry", "eligible pair set identical and Comparable-only"),
])
def test_audit_detects_deliberately_modified_inputs(workload, case, expected):
    sim = to_simulator_input(workload)
    inputs = make_inputs(workload, sim)
    results = run_all(sim)
    bad = inputs["drift-aware"]
    q0 = sim.requests[7]
    if case == "release":
        shifted = dataclasses.replace(q0.second, arrival_time=q0.second.arrival_time + 0.5)
        reqs = list(sim.requests); reqs[7] = dataclasses.replace(q0, second=shifted)
        bad = dataclasses.replace(bad, simulator_input=mutated(sim, requests=tuple(reqs)))
    elif case == "planned":
        planned = list(sim.planned_times); planned[7] += 0.25
        bad = dataclasses.replace(bad, simulator_input=mutated(sim, planned_times=tuple(planned)))
    elif case == "drop_request":
        bad = dataclasses.replace(bad, simulator_input=mutated(sim, requests=sim.requests[:-1], planned_times=sim.planned_times[:-1]))
    elif case == "swap_circuit":
        reqs = list(sim.requests); reqs[7] = dataclasses.replace(q0, second=dataclasses.replace(q0.second, circuit="Gx(Gw)^16Gy"))
        bad = dataclasses.replace(bad, simulator_input=mutated(sim, requests=tuple(reqs)))
    elif case == "shot_count":
        reqs = list(sim.requests); reqs[7] = dataclasses.replace(q0, first=dataclasses.replace(q0.first, repetitions=2))
        bad = dataclasses.replace(bad, simulator_input=mutated(sim, requests=tuple(reqs)))
    elif case == "device":
        sch = dataclasses.replace(sim.schedule, slot_seconds=sim.schedule.slot_seconds * 1.1)
        bad = dataclasses.replace(bad, simulator_input=mutated(sim, schedule=sch))
    elif case == "drift":
        bad = dataclasses.replace(bad, drift_fingerprint="e" * 64)
    elif case == "reference":
        bad = dataclasses.replace(bad, reference_times=tuple(t + 1.0 for t in bad.reference_times))
    elif case == "registry":
        bad = dataclasses.replace(bad, registry=make_registry(KS, not_comparable=[(1, 8)]))
    inputs = {**inputs, "drift-aware": bad}
    a = audit(workload, inputs, results)
    assert not a.passed
    assert expected in {c.name for c in a.failures}
    assert all(c.policy == "drift-aware" for c in a.failures)  # the other two policies still pass
    with pytest.raises(FairnessAuditError):
        a.require()


def test_audit_detects_input_changed_after_handover_and_tampered_results(workload):
    sim = to_simulator_input(workload)
    inputs = make_inputs(workload, sim)
    inputs["consecutive-pair"] = dataclasses.replace(inputs["consecutive-pair"], fingerprint_before="0" * 64)
    a = audit(workload, inputs, run_all(sim))
    assert "input unchanged by the run" in {c.name for c in a.failures}
    results = run_all(sim)
    c = results["arrival-order"].comparisons[0]
    results["arrival-order"] = dataclasses.replace(
        results["arrival-order"], comparisons=(dataclasses.replace(c, arrival_time_a=c.arrival_time_a + 1.0),) + results["arrival-order"].comparisons[1:])
    a2 = audit(workload, make_inputs(workload, sim), results)
    assert "results reference only frozen requests and release times" in {c.name for c in a2.failures}


# ---------------------------------------------------------------- existing behaviour preserved


def test_existing_scheduler_behaviour_is_preserved_when_release_equals_arrival():
    rows = []
    for r in range(30):
        for k in KS:
            rows.append((r * 3.0, k, (r + k) % 2))  # every shot of a round arrives together: old arrival == release
    trace = make_trace(rows)
    prepared = prepare_scheduler_input(trace, REGISTRY, round_reference=trace)
    schedule = build_slot_schedule(prepared.source_shots)
    wl = build_predetermined_workload(prepared.requests, schedule, REGISTRY)
    sim = to_simulator_input(wl)
    pol = {"arrival-order": ArrivalOrderPolicy(), "consecutive-pair": ConsecutivePairPolicy(), "drift-aware": DriftAwarePairPolicy(Table())}
    for name, p in pol.items():
        old = run_simulation(prepared.source_shots, REGISTRY, p, schedule=schedule, requests=prepared.requests)
        new = run_simulation(sim.shots, REGISTRY, p, schedule=sim.schedule, requests=sim.requests)
        assert [(c.request_id, c.slot_a, c.slot_b) for c in old.comparisons] == [(c.request_id, c.slot_a, c.slot_b) for c in new.comparisons]
        assert old.idle_slots == new.idle_slots and len(old.executed) == len(new.executed)


def test_simulator_input_uses_the_unchanged_device_grid(parts, workload):
    _, _, schedule = parts
    assert to_simulator_input(workload).schedule == schedule


# ---------------------------------------------------------------- scoring


def test_trace_rows_keep_the_three_timestamps_separate(workload, model):
    sim = to_simulator_input(workload)
    results = run_all(sim)
    origin = sim.schedule.start_time
    for name, res in results.items():
        rows = trace_rows(workload, res, model, origin, 600.0, 0.05)
        assert len(rows) == workload.request_count and {r["request_id"] for r in rows} == {q.request_id for q in workload.requests}
        for row, q in zip(rows, workload.requests):
            assert row["release_time"] == q.release_time and row["planned_time"] == q.planned_time  # never overwritten
            if row["completed"]:
                assert row["wait_release_to_first_execution"] == pytest.approx(row["actual_execution_first"] - q.release_time)
                assert row["separation_seconds"] == pytest.approx(abs(row["actual_execution_second"] - row["actual_execution_first"]))
                assert row["actual_first_minus_planned"] == pytest.approx(row["actual_execution_first"] - q.planned_time)
                assert 0.0 <= row["error_E_AB_model"] <= 1.0
        s = policy_summary(workload, res, rows, 0.05)
        assert s["completed_comparisons"] == sum(r["completed"] for r in rows) and s["total_shot_budget"] == workload.total_shot_budget
        assert s["budget_device_hours"] == pytest.approx(workload.device.total_slots * workload.device.slot_seconds / 3600.0)


def test_drift_fingerprint_is_stable_and_parameter_sensitive(model):
    assert drift_fingerprint(model, model.basis) == drift_fingerprint(model, model.basis)
    other = true_model(80, gammas=(0.1, 0.26))
    assert drift_fingerprint(other, other.basis) != drift_fingerprint(model, model.basis)
