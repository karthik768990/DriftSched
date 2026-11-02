"""Tests for the explicit device-capacity scenarios (synthetic workloads; the frozen workload is never altered)."""

import dataclasses
import math

import numpy as np
import pytest

from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import ramsey_registry
from quantum_drift_scheduler.modeling.capacity_evaluation import scenario_policy_metrics
from quantum_drift_scheduler.modeling.predetermined_evaluation import trace_rows
from quantum_drift_scheduler.scheduling import (
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    build_slot_schedule,
    prepare_scheduler_input,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.capacity_scenarios import (
    RECORDED_CIRCUITS,
    RECORDED_CYCLE_SECONDS,
    STATED_VALUES,
    CapacityError,
    audit_scenario,
    check_slot_timeline,
    default_scenarios,
    device_timeline_stats,
    horizon_end,
    queue_length_stats,
    requests_fingerprint,
    scenario_device,
    scenario_workload,
)
from quantum_drift_scheduler.scheduling.predetermined_workload import (
    Device,
    audit_fairness,
    build_predetermined_workload,
    make_policy_input,
    to_simulator_input,
)

KS = (1, 2, 4, 8)
REGISTRY = ramsey_registry(KS)
POLICIES = ("arrival-order", "consecutive-pair", "drift-aware")
REAL_BASELINE = 0.7568363221726959


class Table:
    def score(self, a, b):
        return 1.0 + (sum(map(ord, a + b)) % 97) / 50.0


@pytest.fixture(scope="module")
def model():
    return true_model(150, gammas=(0.1, 0.25))


@pytest.fixture(scope="module")
def frozen(model):
    trace = synthetic_shots(model, KS, 150, seed=4)
    prepared = prepare_scheduler_input(trace, REGISTRY, round_reference=trace)
    return build_predetermined_workload(prepared.requests, build_slot_schedule(prepared.source_shots), REGISTRY)


@pytest.fixture(scope="module")
def scenarios(frozen):
    return default_scenarios(frozen.device.slot_seconds, check_expected=False)


def run_scenario(frozen, scen, sim_base=None, drift="d" * 64, model=None):
    sw = scenario_workload(frozen, scen)
    base = sim_base or to_simulator_input(frozen, input_source="frozen")
    sim = dataclasses.replace(base, schedule=sw.device.to_schedule())
    pols = {"arrival-order": ArrivalOrderPolicy(), "consecutive-pair": ConsecutivePairPolicy(), "drift-aware": DriftAwarePairPolicy(Table())}
    results = {n: run_simulation(sim.shots, REGISTRY, p, schedule=sim.schedule, requests=sim.requests) for n, p in pols.items()}
    inputs = {n: make_policy_input(n, sim, REGISTRY, drift, [r.planned_time for r in frozen.requests]) for n in POLICIES}
    return sw, sim, results, inputs


# ---------------------------------------------------------------- scenario definitions


def test_scenario_values_match_the_experiment_definition():
    s = {x.name: x for x in default_scenarios(REAL_BASELINE)}
    assert tuple(s) == tuple(STATED_VALUES)
    assert s["historical_baseline"].service_seconds == REAL_BASELINE
    assert s["slower_device"].service_seconds == pytest.approx(1.1 * REAL_BASELINE, rel=1e-15)
    assert s["faster_device"].service_seconds == pytest.approx(0.9 * REAL_BASELINE, rel=1e-15)
    assert s["recorded_cadence_reference"].service_seconds == pytest.approx(RECORDED_CYCLE_SECONDS / RECORDED_CIRCUITS)
    for name, stated in STATED_VALUES.items():
        tol = 5e-7 if name != "recorded_cadence_reference" else 3e-5
        assert abs(s[name].service_seconds - stated) < tol
        assert s[name].stated_seconds == stated and s[name].hardware_verified is False  # never described as hardware
    assert "not hardware capacity" in s["historical_baseline"].interpretation or "not a device specification" in s["slower_device"].interpretation
    with pytest.raises(CapacityError):
        default_scenarios(0.5)
    with pytest.raises(CapacityError):
        dataclasses.replace(s["slower_device"], service_seconds=0.0)


def test_duration_and_rate_conversion(scenarios):
    for s in scenarios:
        assert s.service_rate_shots_per_second == pytest.approx(1.0 / s.service_seconds)
        assert s.service_seconds * s.service_rate_shots_per_second == pytest.approx(1.0)
    by = {s.name: s for s in scenarios}
    assert by["slower_device"].speed_relative_to_baseline == pytest.approx(STATED_VALUES["historical_baseline"] / by["slower_device"].service_seconds)
    assert by["faster_device"].service_rate_shots_per_second > by["historical_baseline"].service_rate_shots_per_second > by["slower_device"].service_rate_shots_per_second


# ---------------------------------------------------------------- timeline and device time


def test_slot_timeline_uses_a_fixed_wall_clock_horizon(frozen, scenarios):
    end = horizon_end(frozen)
    for s in scenarios:
        dev = scenario_device(frozen, s)
        assert dev.start_time == frozen.device.start_time and dev.slot_seconds == s.service_seconds
        sched = dev.to_schedule()
        assert sched.slot_time(dev.total_slots - 1) <= end + 1e-6 < sched.slot_time(dev.total_slots)
        assert dev.total_slots == int(math.floor((end - dev.start_time) / s.service_seconds + 1e-9)) + 1
        ok, detail = check_slot_timeline(frozen, s)
        assert ok, detail
        st = device_timeline_stats(frozen, s)
        assert st["total_potential_service_seconds"] == pytest.approx(dev.total_slots * s.service_seconds)
        assert st["horizon_seconds"] < st["total_potential_service_seconds"] <= st["horizon_seconds"] + s.service_seconds + 1e-9  # within one slot of the horizon
        assert st["offered_load"] == pytest.approx(frozen.total_shot_budget * s.service_seconds / st["horizon_seconds"])
    by = {s.name: scenario_device(frozen, s) for s in scenarios}
    assert by["historical_baseline"] == frozen.device  # the baseline timeline is exactly the frozen one
    assert by["faster_device"].total_slots > by["historical_baseline"].total_slots > by["slower_device"].total_slots
    assert by["recorded_cadence_reference"].total_slots > by["faster_device"].total_slots


def test_a_fixed_slot_count_would_truncate_the_horizon(frozen, scenarios):
    """Why the wall-clock horizon is used: keeping total_slots would end before the last release for faster devices."""
    n = frozen.device.total_slots
    fast = next(s for s in scenarios if s.name == "faster_device")
    assert (n - 1) * fast.service_seconds + frozen.device.start_time < horizon_end(frozen)
    assert scenario_device(frozen, fast).total_slots > n


def test_timeline_check_detects_a_wrong_grid(frozen, scenarios):
    s = scenarios[1]
    bad = dataclasses.replace(frozen, device=Device(frozen.device.start_time, s.service_seconds, scenario_device(frozen, s).total_slots + 3))
    sim = dataclasses.replace(to_simulator_input(frozen), schedule=bad.device.to_schedule())
    inputs = {n: make_policy_input(n, sim, REGISTRY, "d" * 64, [r.planned_time for r in frozen.requests]) for n in POLICIES}
    checks = audit_scenario(frozen, frozen.fingerprint, s, inputs)
    assert "slot grid equals the scenario grid" in {c.name for c in checks if not c.passed}


# ---------------------------------------------------------------- workload identity and fairness


def test_workload_is_identical_across_scenarios_and_policies(frozen, scenarios):
    before = frozen.fingerprint
    snapshot = dataclasses.replace(frozen)  # equality reference
    base = to_simulator_input(frozen, input_source="frozen")
    for s in scenarios:
        sw, sim, results, inputs = run_scenario(frozen, s, base)
        assert frozen.fingerprint == before and frozen == snapshot  # never mutated
        assert requests_fingerprint(sw) == requests_fingerprint(frozen)
        assert sw.request_count == frozen.request_count and sw.total_shot_budget == frozen.total_shot_budget == 2 * frozen.request_count
        assert sim.shots is base.shots and sim.requests is base.requests  # the very same objects in every scenario
        audit = audit_fairness(sw, inputs, results, "d" * 64, [r.planned_time for r in frozen.requests])
        assert audit.passed, [c.detail for c in audit.failures]
        extra = audit_scenario(frozen, before, s, inputs)
        assert all(c.passed for c in extra), [c.detail for c in extra if not c.passed]
        for res in results.values():
            assert res.workload_requests == frozen.request_count and len(res.executed) + len(res.unexecuted_shots) == frozen.total_shot_budget
    assert scenario_workload(frozen, scenarios[0]).fingerprint == before  # baseline device == frozen device
    assert scenario_workload(frozen, scenarios[1]).fingerprint != before  # the full fingerprint covers the device


def test_audits_detect_mismatched_inputs(frozen, scenarios):
    s = scenarios[2]
    sw, sim, results, inputs = run_scenario(frozen, s)
    refs = [r.planned_time for r in frozen.requests]
    other = dataclasses.replace(sim, schedule=frozen.device.to_schedule())  # one policy on the baseline grid
    bad = {**inputs, "drift-aware": make_policy_input("drift-aware", other, REGISTRY, "d" * 64, refs)}
    assert not audit_fairness(sw, bad, results, "d" * 64, refs).passed
    assert "slot grid equals the scenario grid" in {c.name for c in audit_scenario(frozen, frozen.fingerprint, s, bad) if not c.passed}
    copy = dataclasses.replace(sim, shots=tuple(list(sim.shots)), requests=tuple(list(sim.requests)))  # equal content, different objects
    bad2 = {**inputs, "arrival-order": make_policy_input("arrival-order", copy, REGISTRY, "d" * 64, refs)}
    assert "same shot and request objects as the other policies" in {c.name for c in audit_scenario(frozen, frozen.fingerprint, s, bad2) if not c.passed}
    assert "frozen workload fingerprint unchanged" in {c.name for c in audit_scenario(frozen, "0" * 64, s, inputs) if not c.passed}
    reqs = list(sim.requests)
    q = reqs[3]
    reqs[3] = dataclasses.replace(q, second=dataclasses.replace(q.second, arrival_time=q.second.arrival_time + 1.0))
    bad3 = {**inputs, "consecutive-pair": make_policy_input("consecutive-pair", dataclasses.replace(sim, requests=tuple(reqs)), REGISTRY, "d" * 64, refs)}
    assert not audit_fairness(sw, bad3, results, "d" * 64, refs).passed
    assert "request content identical to the frozen workload (device excluded)" in {c.name for c in audit_scenario(frozen, frozen.fingerprint, s, bad3) if not c.passed}


# ---------------------------------------------------------------- utilisation, idle time, device-hour denominators


def test_utilisation_idle_time_and_denominators(frozen, scenarios, model):
    origin = frozen.device.start_time
    for s in scenarios:
        sw, sim, results, _ = run_scenario(frozen, s)
        for name, res in results.items():
            rows = trace_rows(sw, res, model, origin, 600.0, 0.05)
            m = scenario_policy_metrics(sw, s, res, rows, 0.05)
            d, total = sw.device.slot_seconds, sw.device.total_slots
            assert m["total_slots"] == total and m["busy_slots"] + m["idle_slots"] == total
            assert m["busy_device_seconds"] == pytest.approx(m["busy_slots"] * d)
            assert m["idle_device_seconds"] == pytest.approx(m["idle_slots"] * d)
            assert m["busy_device_seconds"] + m["idle_device_seconds"] == pytest.approx(total * d)
            assert m["device_utilisation_pct"] == pytest.approx(100.0 * m["busy_slots"] / total)
            assert m["budget_device_hours"] == pytest.approx(total * d / 3600.0)
            if m["busy_device_seconds"] > 0 and m["reliable_comparisons_model"]:
                assert m["reliable_per_busy_device_hour_model"] == pytest.approx(m["reliable_comparisons_model"] / (m["busy_device_seconds"] / 3600.0))
            assert m["reliable_per_device_hour_model"] == pytest.approx(m["reliable_comparisons_model"] / m["budget_device_hours"])
            assert m["completed_comparisons"] + m["unfinished_requests"] == frozen.request_count
            if name != "arrival-order":
                assert m["separation_mean_s"] == pytest.approx(d) and m["separation_mean_slots"] == pytest.approx(1.0)  # atomic: one slot
            assert m["separation_mean_slots"] == pytest.approx(m["separation_mean_s"] / d)


def test_queue_length_definition():
    q = queue_length_stats([0.0, 0.0, 1.5], [0.5, 2.0, None], [0.0, 1.0, 2.0, 3.0])
    # t=0: 2 released, none started; t=1: 2 released, 1 started before; t=2: 3 released, 1 started before (the one starting at 2.0 is
    # still counted); t=3: 3 released, 2 started before, the never-started request keeps waiting
    assert q["queue_max"] == 2 and q["queue_final"] == 1 and q["queue_mean"] == pytest.approx((2 + 1 + 2 + 1) / 4)


def test_capacity_changes_completion_in_the_expected_direction(frozen, scenarios):
    done = {}
    for s in scenarios:
        _, _, results, _ = run_scenario(frozen, s)
        done[s.name] = {n: len(r.comparisons) for n, r in results.items()}
    for n in POLICIES:
        assert done["faster_device"][n] >= done["historical_baseline"][n] >= done["slower_device"][n]
        assert done["recorded_cadence_reference"][n] >= done["faster_device"][n]
        assert done["recorded_cadence_reference"][n] <= frozen.request_count


def test_repeated_runs_are_identical(frozen, scenarios, model):
    origin = frozen.device.start_time
    def metrics():
        out = []
        for s in scenarios:
            sw, _, results, _ = run_scenario(frozen, s)
            for n, res in results.items():
                out.append(scenario_policy_metrics(sw, s, res, trace_rows(sw, res, model, origin, 600.0, 0.05), 0.05))
        return out
    assert metrics() == metrics()
