"""Tests for the SIMULATED five-hour workload, the controlled drift process and the fair policy evaluation."""

import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom

from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd
from quantum_drift_scheduler.analysis.age_calibration import CalibrationError, calibrate
from quantum_drift_scheduler.analysis.priorities import LeakageError, learn_pair_priorities
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry
from quantum_drift_scheduler.scheduling.age_aware_policy import DRIFT_AWARE_AGE, AgeConfig
from quantum_drift_scheduler.scheduling.models import SimulationError
from quantum_drift_scheduler.scheduling.policies import ARRIVAL_ORDER, CONSECUTIVE_PAIR, DEFAULT_POLICY, DRIFT_AWARE
from quantum_drift_scheduler.synthetic import (
    DriftConfig,
    SyntheticDrift,
    SyntheticWorkload,
    WorkloadConfig,
    capacity_equations,
    generate_training_shots,
    generate_workload,
    load_workload,
    save_workload,
)
from quantum_drift_scheduler.synthetic.evaluation import (
    ALL_POLICIES,
    audit_runs,
    common_uniforms,
    evaluate,
    learn_priorities,
    policy_metrics,
    run_policy,
    score_run,
    training_workload,
)
from quantum_drift_scheduler.synthetic.workload import _fingerprint

ROOT = Path(__file__).resolve().parent.parent
PAIR_A, PAIR_B = "Gx(Gw)Gy", "Gx(Gw)^2Gy"


@pytest.fixture(scope="module")
def registry():
    return load_compatibility_registry()


@pytest.fixture(scope="module")
def config():
    return WorkloadConfig()


@pytest.fixture(scope="module")
def workload(registry, config):
    return generate_workload(config, registry)


@pytest.fixture(scope="module")
def wl(workload):
    return workload.scheduler_workload()


@pytest.fixture(scope="module")
def drift():
    return SyntheticDrift(DriftConfig(seed=0, rate_factor=100.0))


@pytest.fixture(scope="module")
def priorities(registry, config, drift):
    return learn_priorities(drift, registry, config, seed=0)


@pytest.fixture(scope="module")
def runs(wl, registry, priorities, config):
    cal = calibrate(training_workload(config, registry), priorities, boundary=config.start_time)
    return {
        ARRIVAL_ORDER: run_policy(ARRIVAL_ORDER, wl, registry),
        CONSECUTIVE_PAIR: run_policy(CONSECUTIVE_PAIR, wl, registry),
        DRIFT_AWARE: run_policy(DRIFT_AWARE, wl, registry, priorities),
        DRIFT_AWARE_AGE: run_policy(DRIFT_AWARE_AGE, wl, registry, priorities, cal.config),
    }, cal


# --------------------------------------------------------------------------- five-hour horizon, capacity and slot accounting


def test_five_hour_simulated_horizon(config, wl):
    assert config.horizon_seconds == 18000.0
    assert config.total_slots * config.service_seconds == 18000.0
    eq = capacity_equations(config)
    assert eq["device_hours"] == pytest.approx(5.0)
    assert wl.schedule.total_slots == 18000
    assert wl.schedule.slot_time(wl.schedule.total_slots - 1) < config.start_time + config.horizon_seconds
    assert all(config.start_time <= q.ready_time < config.start_time + config.horizon_seconds for q in wl.requests)


def test_workload_size_follows_the_capacity_equations(config, workload):
    eq = capacity_equations(config)
    assert eq["capacity_requests_per_second"] == pytest.approx(1.0 / (2.0 * config.service_seconds))
    assert eq["expected_requests"] == pytest.approx(sum(l * config.block_seconds / (2 * config.service_seconds) for l in config.block_loads))
    n = len(workload.requests)
    assert abs(n - eq["expected_requests"]) < 6 * math.sqrt(eq["expected_requests"]) * 1.5  # Poisson with burst modulation
    # halving the service time doubles the capacity and (at the same offered load) the request count
    fast = capacity_equations(replace(config, service_seconds=0.5))
    assert fast["expected_requests"] == pytest.approx(2 * eq["expected_requests"])
    assert fast["total_slots"] == 36000


def test_queue_pressure_has_bursts_quiet_periods_and_overload(config, workload):
    release = np.array([r[3] for r in workload.requests])
    per_block = np.histogram(release, bins=config.start_time + config.block_seconds * np.arange(len(config.block_loads) + 1))[0]
    load = per_block * 2 * config.service_seconds / config.block_seconds
    assert load.min() < 0.7 and load.max() > 1.2          # quiet and overloaded periods
    per_minute = np.histogram(release, bins=np.arange(config.start_time, config.start_time + config.horizon_seconds + 60, 60.0))[0]
    assert per_minute.max() > 60 / (2 * config.service_seconds)  # the minute-level rate exceeds capacity at times
    assert per_minute.std() / per_minute.mean() > 0.3            # genuinely bursty, not a constant rate


def test_slot_accounting_for_every_policy(runs, wl):
    results, _cal = runs
    for name, res in results.items():
        assert len(res.executed) + res.idle_slots == wl.schedule.total_slots, name
        assert len({e.shot_id for e in res.executed}) == len(res.executed), name
        for e in res.executed[:500]:
            assert e.execution_time == pytest.approx(wl.schedule.start_time + e.slot * wl.schedule.slot_seconds)
            assert e.execution_time >= e.arrival_time - 1e-9


# --------------------------------------------------------------------------- workload generation and reproducibility


def test_generation_is_reproducible_and_seed_sensitive(registry, config, workload):
    again = generate_workload(config, registry)
    assert again.requests == workload.requests and again.fingerprint == workload.fingerprint
    other = generate_workload(replace(config, seed=config.seed + 1), registry)
    assert other.fingerprint != workload.fingerprint and other.requests != workload.requests


def test_save_load_roundtrip_and_tamper_detection(tmp_path, workload):
    path = tmp_path / "w.json"
    save_workload(workload, path)
    assert (tmp_path / "w_fingerprint.txt").read_text().strip() == workload.fingerprint
    loaded = load_workload(path)
    assert loaded.requests == workload.requests and loaded.fingerprint == workload.fingerprint
    data = json.loads(path.read_text())
    data["requests"][0][5] = 999999  # change a measurement requirement
    path.write_text(json.dumps(data))
    with pytest.raises(SimulationError, match="fingerprint"):
        load_workload(path)


def test_only_eligible_pairs_and_varied_requirements(registry, workload, wl):
    eligible = {(p.circuit_a, p.circuit_b) for p in registry.eligible_pairs()}
    used = {(q.first.circuit, q.second.circuit) for q in wl.requests}
    assert used <= eligible and len(used) == len(eligible) == 15          # every eligible pair appears, nothing else
    assert all(registry.is_eligible(a, b) for a, b in used)
    assert {r[5] for r in workload.requests} == {100, 400, 1600}           # different measurement requirements
    shots = [s.shot_id for s in wl.shots]
    assert len(shots) == len(set(shots)) == 2 * len(wl.requests)           # each shot belongs to exactly one request
    assert len({tuple(sorted(q.shot_ids)) for q in wl.requests}) == len(wl.requests)


def test_workload_is_synthetic_not_recorded(workload, tmp_path):
    save_workload(workload, tmp_path / "w.json")
    text = (tmp_path / "w.json").read_text()
    assert "SIMULATED" in text and "synthetic-five-hour" in workload.scheduler_workload().name
    assert all(s.input_source == "synthetic-five-hour" for s in workload.scheduler_workload().shots)


# --------------------------------------------------------------------------- identical workload, atomicity, fairness


def test_identical_workload_and_device_across_policies(runs, wl, workload, registry, priorities):
    results, cal = runs
    for name, res in results.items():
        assert res.schedule == wl.schedule and res.workload_requests == len(wl.requests) and res.source_shots == len(wl.shots), name
    audits = audit_runs(wl, registry, results, workload.fingerprint, workload.fingerprint, priorities, workload.config,
                        {CONSECUTIVE_PAIR: sd.simulate_pair_dispatch(wl, CONSECUTIVE_PAIR, sd.chooser_from_key(__import__("quantum_drift_scheduler.scheduling.policies", fromlist=["x"]).request_order_key))})
    failed = [a for a in audits if not a["passed"]]
    assert not failed, failed


def test_pair_atomicity_of_pair_policies_and_arrival_order_contrast(runs):
    results, _ = runs
    for name in (CONSECUTIVE_PAIR, DRIFT_AWARE, DRIFT_AWARE_AGE):
        assert all(c.slot_b - c.slot_a == 1 for c in results[name].comparisons), name
    assert max(c.slot_separation for c in results[ARRIVAL_ORDER].comparisons) > 1


def test_audit_detects_a_broken_run(runs, wl, workload, registry, priorities):
    results, _ = runs
    bad = dict(results)
    bad[CONSECUTIVE_PAIR] = results[ARRIVAL_ORDER]  # a non-atomic schedule presented as a pair policy
    audits = audit_runs(wl, registry, bad, workload.fingerprint, workload.fingerprint, priorities, workload.config, {})
    assert any(a["check"].startswith("pair atomicity") and not a["passed"] for a in audits)
    wrong_fp = audit_runs(wl, registry, results, workload.fingerprint, "0" * 64, priorities, workload.config, {})
    assert any(a["check"].startswith("workload fingerprint") and not a["passed"] for a in wrong_fp)


def test_default_policy_and_policy_set_are_preserved():
    assert DEFAULT_POLICY == CONSECUTIVE_PAIR
    assert set(ALL_POLICIES) == {ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE, DRIFT_AWARE_AGE}


def test_age_policy_limits_reduce_to_the_existing_policies(runs, wl, registry, priorities):
    """Integration check: huge H = strict drift-aware, tiny H = FIFO ordering (the formula is unchanged)."""
    results, cal = runs
    huge = run_policy(DRIFT_AWARE_AGE, wl, registry, priorities, AgeConfig(1e12, cal.priority_range))
    assert [(c.request_id, c.slot_a) for c in huge.comparisons] == [(c.request_id, c.slot_a) for c in results[DRIFT_AWARE].comparisons]


# --------------------------------------------------------------------------- outcomes generated at the actual execution times


def test_scored_values_are_computed_at_actual_execution_times(runs, wl, drift, workload):
    results, _ = runs
    reps = np.array([q.first.repetitions for q in wl.requests], dtype=float)
    u = common_uniforms(1, len(wl.requests))
    s_fifo = score_run(results[CONSECUTIVE_PAIR], wl, drift, u, reps)
    s_arr = score_run(results[ARRIVAL_ORDER], wl, drift, u, reps)
    i = int(np.flatnonzero(s_fifo["completed"])[100])
    q = wl.requests[i]
    pa = drift.probability(q.wait_length_first, np.array([s_fifo["ta"][i]]))[0]
    pb = drift.probability(q.wait_length_second, np.array([s_fifo["tb"][i]]))[0]
    assert s_fifo["delta_measured"][i] == pytest.approx(pa - pb, abs=1e-12)
    ref = drift.probability(q.wait_length_first, np.array([q.ready_time]))[0] - drift.probability(q.wait_length_second, np.array([q.ready_time]))[0]
    assert s_fifo["delta_reference"][i] == pytest.approx(ref, abs=1e-12)
    assert s_fifo["error"][i] == pytest.approx(abs(pa - pb - ref), abs=1e-12)
    xa = binom.ppf(u[i, 0], q.first.repetitions, pa)
    xb = binom.ppf(u[i, 1], q.second.repetitions, pb)
    assert s_fifo["outcome_error"][i] == pytest.approx(abs((xa - xb) / q.first.repetitions - ref), abs=1e-12)
    # reordering changes execution times, hence the measured quantities, for the same request and the same drift
    both = s_fifo["completed"] & s_arr["completed"] & ((s_fifo["ta"] != s_arr["ta"]) | (s_fifo["tb"] != s_arr["tb"]))
    assert both.sum() > 100
    assert np.any(s_fifo["delta_measured"][both] != s_arr["delta_measured"][both])
    same = s_fifo["completed"] & s_arr["completed"] & (s_fifo["ta"] == s_arr["ta"]) & (s_fifo["tb"] == s_arr["tb"])
    if same.any():  # same times, same drift, same uniforms -> identical numbers (common random numbers)
        assert np.array_equal(s_fifo["outcome_error"][same], s_arr["outcome_error"][same])


def test_outcomes_depend_on_the_time_at_which_the_job_runs():
    fast = SyntheticDrift(DriftConfig(seed=3, rate_factor=1000.0))
    t = np.linspace(100.0, 160.0, 600)
    p = fast.probability(32, t)
    assert p.max() - p.min() > 0.005                       # drift is visible over a minute at the fast rate
    u = np.full(t.size, 0.5)
    x = binom.ppf(u, 100000, p)
    assert x.max() != x.min()                              # same uniform, different time -> different count
    assert np.array_equal(binom.ppf(u, 100000, p), x)      # deterministic


def test_drift_is_deterministic_and_seed_dependent():
    a, b, c = (SyntheticDrift(DriftConfig(seed=s)) for s in (0, 0, 1))
    t = np.linspace(0, 25000, 50)
    assert np.array_equal(a.detuning_hz(t), b.detuning_hz(t)) and not np.array_equal(a.detuning_hz(t), c.detuning_hz(t))
    for k in (1, 2, 4, 8, 16, 32):
        p = a.probability(k, t)
        assert np.all((p >= 0) & (p <= 1))


# --------------------------------------------------------------------------- no future information


def test_drift_value_at_a_time_does_not_depend_on_later_times(drift):
    t = np.array([8000.0])
    alone = drift.detuning_hz(t)[0]
    with_future = drift.detuning_hz(np.array([8000.0, 20000.0, 24999.0]))[0]
    assert alone == with_future


def test_training_data_is_strictly_before_the_horizon(registry, config, drift, priorities, wl):
    shots = generate_training_shots(drift, registry, config.train_seconds, seed=0)
    assert max(s.arrival_time for s in shots) < config.start_time
    assert priorities.metadata.train_end < config.start_time
    assert min(q.ready_time for q in wl.requests) >= config.start_time
    late = shots[:-1] + (replace(shots[-1], arrival_time=config.start_time + 1.0),)
    with pytest.raises(LeakageError):
        learn_pair_priorities(late, registry, boundary=config.start_time, window_origin=0.0, train_fraction=0.3)


def test_priorities_do_not_depend_on_the_evaluation_workload_or_outcomes(registry, config, drift, priorities):
    again = learn_priorities(drift, registry, config, seed=0)
    assert [(p.circuit_a, p.circuit_b, p.priority_score) for p in again.priorities] == [(p.circuit_a, p.circuit_b, p.priority_score) for p in priorities.priorities]
    other_workload = generate_workload(replace(config, seed=config.seed + 5), registry)  # a different evaluation workload
    assert other_workload.fingerprint
    third = learn_priorities(drift, registry, config, seed=0)
    assert [p.priority_score for p in third.priorities] == [p.priority_score for p in priorities.priorities]


def test_calibration_uses_only_pre_horizon_requests(registry, config, priorities, workload):
    train = training_workload(config, registry)
    assert max(q.ready_time for q in train.requests) < config.start_time
    assert train.requests[0].first.arrival_time != workload.scheduler_workload().requests[0].first.arrival_time
    with pytest.raises(CalibrationError):
        calibrate(workload.scheduler_workload(), priorities, boundary=config.start_time)  # evaluation requests are rejected


def test_schedules_of_drift_blind_policies_do_not_depend_on_drift_or_outcomes(wl, registry):
    a = run_policy(CONSECUTIVE_PAIR, wl, registry)
    b = run_policy(CONSECUTIVE_PAIR, wl, registry)
    assert [(c.request_id, c.slot_a) for c in a.comparisons] == [(c.request_id, c.slot_a) for c in b.comparisons]
    assert all(s.outcome == 0 for s in wl.shots)  # the scheduler is handed placeholders, never outcomes


# --------------------------------------------------------------------------- waiting time and starvation calculations


def _tiny(registry, requests, horizon=20.0):
    cfg = WorkloadConfig(horizon_seconds=horizon, service_seconds=1.0, train_seconds=0.0, block_seconds=horizon, block_loads=(1.0,), start_time=0.0)
    return SyntheticWorkload(cfg, tuple(requests), registry.content_fingerprint, ((PAIR_A, PAIR_B),), "t").scheduler_workload("tiny")


def test_waiting_time_and_unfinished_accounting(registry):
    # slot starts 0..9 (10 slots). r0 ready 0 -> slots 0,1 (wait 0); r1 ready 0.5 -> slots 2,3 (wait 1.5);
    # r2 ready 8.5 -> slot 9 is the last slot, so a pair cannot start: unfinished, truncated by the horizon (age 0.5).
    reqs = [("req-r000000-00", PAIR_A, PAIR_B, 0.0, 0.0, 100), ("req-r000001-00", PAIR_A, PAIR_B, 0.5, 0.5, 100), ("req-r000002-00", PAIR_A, PAIR_B, 8.5, 8.5, 100)]
    wl = _tiny(registry, reqs, horizon=10.0)
    res = run_policy(CONSECUTIVE_PAIR, wl, registry)
    drift = SyntheticDrift(DriftConfig(seed=0))
    sc = score_run(res, wl, drift, common_uniforms(0, 3), np.array([100.0, 100.0, 100.0]))
    row, q = policy_metrics(CONSECUTIVE_PAIR, res, wl, sc)
    assert (row["completed_comparisons"], row["unfinished_requests"]) == (2, 1)
    assert row["req_wait_completed_mean_s"] == pytest.approx(0.75) and row["req_wait_completed_max_s"] == pytest.approx(1.5)
    assert row["req_unfinished_truncated_by_horizon"] == 1 and row["req_unfinished_not_served_in_time"] == 0
    assert row["req_unfinished_age_max_s"] == pytest.approx(0.5)
    assert row["req_waiting_over_10s_incl_unfinished"] == 0
    assert row["separation_s_mean"] == pytest.approx(1.0) and row["separation_slots_max"] == 1
    assert row["executed_shots"] == 4 and row["idle_slots"] == 6 and row["device_utilisation"] == pytest.approx(0.4)
    assert row["device_hours"] == pytest.approx(10.0 / 3600.0)
    assert q.max() <= 1


def test_strict_priority_starves_and_age_policy_serves_the_old_request(registry):
    class Scores:  # pair B scores higher than pair A
        def score(self, a, b):
            return 1.0 if (a, b) == (PAIR_A, "Gx(Gw)^4Gy") else 0.0

    # one old request of the low-priority pair at t=0, then a continuous stream of high-priority requests
    reqs = [("req-r000000-00", PAIR_A, PAIR_B, 0.0, 0.0, 100)]
    reqs += [(f"req-r{i:06d}-00", PAIR_A, "Gx(Gw)^4Gy", 2.0 * (i - 1), 2.0 * (i - 1), 100) for i in range(1, 11)]
    cfg = WorkloadConfig(horizon_seconds=20.0, service_seconds=1.0, train_seconds=0.0, block_seconds=20.0, block_loads=(1.0,), start_time=0.0)
    wl = SyntheticWorkload(cfg, tuple(reqs), registry.content_fingerprint, ((PAIR_A, PAIR_B), (PAIR_A, "Gx(Gw)^4Gy")), "t").scheduler_workload("starve")
    strict = run_policy(DRIFT_AWARE, wl, registry, Scores())
    aged = run_policy(DRIFT_AWARE_AGE, wl, registry, Scores(), AgeConfig(horizon_seconds=3.0, priority_range=1.0))
    drift, u, reps = SyntheticDrift(DriftConfig(seed=0)), common_uniforms(0, len(reqs)), np.full(len(reqs), 100.0)
    m_strict, _ = policy_metrics(DRIFT_AWARE, strict, wl, score_run(strict, wl, drift, u, reps))
    m_aged, _ = policy_metrics(DRIFT_AWARE_AGE, aged, wl, score_run(aged, wl, drift, u, reps), 3.0)
    assert "req-r000000-00" not in {c.request_id for c in strict.comparisons}
    assert m_strict["req_unfinished_not_served_in_time"] >= 1 and m_strict["req_overtaken_max"] >= 5
    assert "req-r000000-00" in {c.request_id for c in aged.comparisons}
    assert m_aged["req_overtaken_max"] < m_strict["req_overtaken_max"]
    assert m_aged["req_wait_incl_unfinished_max_s"] < m_strict["req_wait_incl_unfinished_max_s"]


def test_paired_bootstrap_and_queue_metric_basics():
    from quantum_drift_scheduler.synthetic.evaluation import paired_bootstrap

    mean, lo, hi = paired_bootstrap(np.full(500, -1.0) + np.random.default_rng(0).normal(0, 0.1, 500), reps=300)
    assert lo < mean < hi < 0
    assert math.isnan(paired_bootstrap(np.array([np.nan, np.nan]))[0])


# --------------------------------------------------------------------------- end-to-end scripts


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_generate_verify_and_evaluate(tmp_path, capsys):
    gen, ev = _load("generate_five_hour_workload"), _load("run_five_hour_evaluation")
    out = tmp_path / "w"
    assert gen.main(["--output-dir", str(out)]) == 0
    assert gen.main(["--output-dir", str(out)]) == 1                    # never silently overwritten
    assert gen.main(["--output-dir", str(out), "--verify"]) == 0
    workload = out / "five_hour_workload.json"
    res = tmp_path / "r"
    code = ev.main(["--workload", str(workload), "--output-dir", str(res), "--drift-rate-factors", "100", "--drift-seeds", "1", "--bootstrap-reps", "50"])
    assert code == 0
    for name in ("summary.json", "policy_metrics.csv", "paired_differences.csv", "fairness_audit.csv", "experiment_report.md", "per_request_first_drift_seed.csv"):
        assert (res / name).exists(), name
    summary = json.loads((res / "summary.json").read_text())
    assert summary["audits"]["failed"] == 0 and summary["default_policy"] == "consecutive-pair"
    assert summary["workload_fingerprint"] == (out / "five_hour_workload_fingerprint.txt").read_text().strip()
    assert ev.main(["--workload", str(workload), "--output-dir", str(res), "--drift-seeds", "1"]) == 1  # existing run protected
    only = tmp_path / "only"
    assert ev.main(["--workload", str(workload), "--output-dir", str(only), "--policies", "consecutive-pair", "--drift-rate-factors", "1", "--drift-seeds", "1"]) == 0
    assert json.loads((only / "summary.json").read_text())["policies"] == ["consecutive-pair"]
