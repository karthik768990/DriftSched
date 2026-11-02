"""Tests for the three-policy evaluation: metrics, fairness audit, references, outputs, safeguards.

Everything here uses small synthetic traces or hand-built arrays; nothing runs the real dataset.
"""

import csv
import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from quantum_drift_scheduler.analysis.holdout import SplitError
from quantum_drift_scheduler.modeling.comparison_metrics import classify_reliability
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD
from quantum_drift_scheduler.modeling.policy_evaluation import POLICY_ORDER, PolicyArrays
from quantum_drift_scheduler.modeling.three_policy import (
    OUTPUT_FILES,
    PROVENANCE_MODEL,
    PROVENANCE_REPLAY,
    ThreePolicyConfig,
    ThreePolicyError,
    TimeScaledModel,
    audit_fairness,
    build_results,
    check_output_dir,
    compare_with_priority_analysis,
    extreme_metrics,
    run_three_policy_evaluation,
    scalar_metrics,
    write_figures,
    write_outputs,
)
from _qcs_helpers import synthetic_shots, true_model
from _sched_helpers import label, make_registry

ROOT = Path(__file__).resolve().parent.parent
KS = (1, 16, 256, 2048)


def fast_config(tmp_path, **extra):
    base = dict(output_dir=tmp_path / "out", priority_analysis_csv=tmp_path / "none.csv", candidates=(0, 1),
                cv_blocks=3, n_starts=2, n_bootstrap=100, stress_drift_rate_factors=(5.0,), make_figures=False)
    base.update(extra)
    return ThreePolicyConfig(**base)


def synthetic_inputs(seed=3, rounds=2500, not_comparable=((1, 2048),)):
    model = true_model(rounds, gammas=(0.1, 0.3), decay_rate=2e-5)
    return synthetic_shots(model, KS, rounds, seed=seed), make_registry(KS, not_comparable=not_comparable)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("three")
    trace, registry = synthetic_inputs()
    return tmp, trace, registry, run_three_policy_evaluation(fast_config(tmp), trace=trace, registry=registry)


def hand_arrays(name="p", budget=7200.0, busy=3600.0):
    nan = np.nan
    err = np.array([0.01, 0.02, 0.07, nan])
    done = np.array([True, True, True, False])
    return PolicyArrays(
        name=name, request_ids=("r0", "r1", "r2", "r3"), completed=done, comparison_id=np.array([1, 2, 3, -1]),
        circuit_a=(label(1),) * 4, circuit_b=(label(16),) * 4,
        arrival_a=np.array([0.0, 9.0, 19.0, nan]), arrival_b=np.array([0.0, 10.0, 19.0, nan]),
        execution_a=np.array([1.0, 10.0, 20.0, nan]), execution_b=np.array([2.0, 13.0, 20.5, nan]),
        slot_a=np.array([0.0, 5.0, 9.0, nan]), slot_b=np.array([1.0, 6.0, 10.0, nan]),
        waiting_a=np.array([1.0, 1.0, 1.0, nan]), waiting_b=np.array([2.0, 3.0, 1.5, nan]),
        measured_delta=np.zeros(4), reference_delta=np.zeros(4), error=err,
        synchronous_error=np.array([0.001, 0.002, 0.003, nan]), window_index=np.zeros(4),
        reliable=classify_reliability(err, RELIABILITY_THRESHOLD), budget_seconds=budget, busy_seconds=busy,
        slot_seconds=1.0, executed_shots=6, unexecuted_shots=2, idle_slots=2,
    )


# ---------------------------------------------------------------- formulas on hand-built arrays


def test_metric_values_match_hand_calculation():
    a = hand_arrays()
    m = scalar_metrics(a, np.arange(4))
    assert m["error_mean"] == pytest.approx((0.01 + 0.02 + 0.07) / 3) and m["error_median"] == pytest.approx(0.02)
    assert m["sync_error_mean"] == pytest.approx(0.002) and m["sync_error_median"] == pytest.approx(0.002)
    assert m["separation_mean"] == pytest.approx((1.0 + 3.0 + 0.5) / 3) and m["separation_median"] == pytest.approx(1.0)
    waits = [1.0, 2.0, 1.0, 3.0, 1.0, 1.5]
    assert m["waiting_mean"] == pytest.approx(np.mean(waits)) and m["waiting_median"] == pytest.approx(np.median(waits))
    assert m["completed_comparisons"] == 3 and m["completion_rate"] == pytest.approx(0.75)
    ext = extreme_metrics(a)
    assert ext["error_max"] == pytest.approx(0.07) and ext["separation_max"] == pytest.approx(3.0)
    assert ext["waiting_max"] == pytest.approx(3.0)
    assert ext["error_p95"] == pytest.approx(np.percentile([0.01, 0.02, 0.07], 95))


def test_reliability_uses_the_fixed_inclusive_threshold():
    assert RELIABILITY_THRESHOLD == 0.05 and ThreePolicyConfig().reliability_threshold == 0.05
    a = hand_arrays()
    assert a.reliable.tolist() == [True, True, False, False]  # 0.07 too large; incomplete never reliable
    assert classify_reliability(np.array([0.05, 0.0500001]), 0.05).tolist() == [True, False]
    assert "reliability_threshold" not in {f.name for f in dataclasses.fields(ThreePolicyConfig)}  # not an option


def test_reliable_throughput_definition_and_device_time():
    a = hand_arrays(budget=7200.0, busy=3600.0)
    m = scalar_metrics(a, np.arange(4))
    assert m["reliable_comparisons"] == 2
    assert m["reliable_throughput_per_device_hour"] == pytest.approx(2 / 2.0)  # 2 reliable / 2 budgeted hours
    assert m["reliable_throughput_per_device_second"] == pytest.approx(2 / 7200.0)
    assert m["reliable_throughput_per_busy_hour"] == pytest.approx(2 / 1.0)
    # same device-time budget => throughput ranks by reliable count; a longer budget lowers it
    assert scalar_metrics(hand_arrays(budget=14400.0), np.arange(4))["reliable_throughput_per_device_hour"] == pytest.approx(0.5)


def test_resampling_rescales_counts_to_the_workload_size():
    a = hand_arrays()
    doubled = scalar_metrics(a, np.array([0, 1, 2, 3, 0, 1, 2, 3]))
    assert doubled["completed_comparisons"] == 3 and doubled["reliable_comparisons"] == 2
    only_incomplete = scalar_metrics(a, np.array([3, 3, 3, 3]))
    assert math.isnan(only_incomplete["error_mean"]) and only_incomplete["completed_comparisons"] == 0
    assert only_incomplete["reliable_throughput_per_device_hour"] == 0.0


# ---------------------------------------------------------------- controlled drift model


def test_time_scaled_model_semantics():
    base = true_model(500, gammas=(0.1, 0.4))
    t = np.array([2000.0, 2500.0, 3000.0])
    l = np.full(3, 256.0)
    assert np.allclose(TimeScaledModel(base, 1.0, 2000.0).probability(l, t), base.probability(l, t))
    fast = TimeScaledModel(base, 3.0, 2000.0).probability(l, t)
    assert np.allclose(fast, base.probability(l, 2000.0 + 3.0 * (t - 2000.0)))
    assert fast[0] == pytest.approx(base.probability(l, t)[0])  # unchanged at the scaling origin
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ThreePolicyError, match="rate_factor"):
            TimeScaledModel(base, bad, 0.0)


# ---------------------------------------------------------------- full synthetic experiment


def test_primary_scenario_and_labelled_stress_scenarios(run):
    _, _, _, res = run
    assert res.primary.kind == "primary" and res.primary.rate_factor == 1.0
    assert [s.kind for s in res.scenarios] == ["primary", "synthetic-stress"] and res.scenarios[1].rate_factor == 5.0
    for sc in res.scenarios:
        assert tuple(sc.metrics) == POLICY_ORDER and set(sc.differences) == {
            "consecutive-pair - arrival-order", "drift-aware - arrival-order", "drift-aware - consecutive-pair"}
    # queue behaviour is a property of the replay, identical across scenarios; errors respond to the drift
    for p in POLICY_ORDER:
        assert res.scenarios[1].metrics[p]["waiting_mean"].estimate == res.primary.metrics[p]["waiting_mean"].estimate
        assert res.scenarios[1].metrics[p]["error_mean"].estimate > res.primary.metrics[p]["error_mean"].estimate


def test_metrics_cover_everything_requested(run):
    _, _, _, res = run
    for p in POLICY_ORDER:
        m = res.primary.metrics[p]
        for name in ("error_mean", "error_median", "error_p95", "error_max", "separation_mean", "separation_median",
                     "separation_max", "waiting_mean", "waiting_median", "waiting_max", "completed_comparisons", "idle_slots",
                     "reliable_throughput_per_device_hour", "reliable_throughput_per_busy_hour"):
            assert m[name].estimate is not None, name
        for name in ("error_mean", "error_median", "separation_mean", "waiting_mean", "reliable_throughput_per_device_hour"):
            v = m[name]
            assert v.interval and v.lower <= v.estimate <= v.upper
        assert not m["error_max"].interval and not m["waiting_max"].interval  # extremes are point estimates
        assert m["error_max"].estimate >= m["error_p95"].estimate >= m["error_median"].estimate >= 0


def test_pair_aware_policies_keep_measurements_together(run):
    _, _, _, res = run
    arrival = res.primary.metrics["arrival-order"]
    for p in ("consecutive-pair", "drift-aware"):
        pair = res.primary.metrics[p]
        assert pair["separation_max"].estimate == pytest.approx(res.scheduler.schedule.slot_seconds)  # exactly one slot
        assert pair["separation_mean"].estimate < arrival["separation_mean"].estimate
        assert pair["sync_error_mean"].estimate <= arrival["sync_error_mean"].estimate


def test_idle_slots_and_budget_match_the_scheduler(run):
    _, _, _, res = run
    sched = res.scheduler
    for run_ in sched.runs:
        m = res.primary.metrics[run_.result.policy_name]
        assert m["idle_slots"].estimate == run_.result.idle_slots
        assert m["total_slots"].estimate == pytest.approx(sched.schedule.total_slots)
        assert m["budget_seconds"].estimate == pytest.approx(sched.schedule.total_slots * sched.schedule.slot_seconds)
    budgets = {res.primary.metrics[p]["budget_seconds"].estimate for p in POLICY_ORDER}
    assert len(budgets) == 1  # one device-time definition and budget for every policy


def test_fairness_audit_passes_on_a_genuine_run(run):
    _, _, _, res = run
    names = {c.name for c in res.fairness}
    assert names == {"same_source_shots_and_measurement_budget", "same_workload_and_eligible_pairs",
                     "same_device_capacity_and_time_budget", "same_arrival_pattern_and_initial_queue",
                     "same_reference_values_independent_of_scheduler", "same_simulated_drift_for_all_policies",
                     "only_comparable_pairs_scheduled", "priorities_fresh_training_only_and_frozen",
                     "no_holdout_in_model_or_priorities"}
    assert res.audits_passed, [c for c in res.fairness if not c.passed]


def test_fairness_audit_detects_tampering(run):
    _, _, registry, res = run
    arrays = res.primary.arrays
    model = res.training.model
    origin = res.scheduler.split.min_timestamp
    train_end = float(res.training.model.basis.t0)  # any pre-boundary value

    def audit(modified):
        return {c.name: c.passed for c in audit_fairness(res.scheduler, modified, model, registry, res.reference_times,
                                                         origin, 600.0, train_end)}

    assert all(audit(arrays).values())
    other = dict(arrays)
    other["drift-aware"] = dataclasses.replace(arrays["drift-aware"], request_ids=arrays["drift-aware"].request_ids[::-1])
    assert not audit(other)["same_workload_and_eligible_pairs"]
    other["drift-aware"] = dataclasses.replace(arrays["drift-aware"], budget_seconds=arrays["drift-aware"].budget_seconds * 2)
    assert not audit(other)["same_device_capacity_and_time_budget"]
    tampered = arrays["consecutive-pair"].reference_delta + 0.01
    other["consecutive-pair"] = dataclasses.replace(arrays["consecutive-pair"], reference_delta=tampered)
    assert not audit(other)["same_reference_values_independent_of_scheduler"]
    other["arrival-order"] = dataclasses.replace(arrays["arrival-order"], error=arrays["arrival-order"].error * 2)
    assert not audit(other)["same_simulated_drift_for_all_policies"]
    shifted = dataclasses.replace(arrays["drift-aware"], arrival_a=arrays["drift-aware"].arrival_a + 1.0)
    assert not audit({**arrays, "drift-aware": shifted})["same_arrival_pattern_and_initial_queue"]
    assert not audit_fairness(res.scheduler, arrays, model, registry, res.reference_times, origin, 600.0,
                              res.scheduler.split.boundary + 1.0)[-1].passed  # holdout-time model data is rejected


def test_references_are_identical_across_policies_and_independent_of_outcomes(run):
    tmp, trace, registry, res = run
    a, b, c = (res.primary.arrays[p] for p in POLICY_ORDER)
    for x, y in ((a, b), (a, c)):
        both = x.completed & y.completed
        assert both.sum() > 0 and np.array_equal(x.reference_delta[both], y.reference_delta[both])
    flipped = tuple(dataclasses.replace(s, outcome=1 - s.outcome) if s.arrival_time >= res.scheduler.split.boundary else s
                    for s in trace)
    other = run_three_policy_evaluation(fast_config(tmp, output_dir=tmp / "flipped"), trace=flipped, registry=registry)
    assert other.training.model.parameters == res.training.model.parameters
    assert other.scheduler.priorities == res.scheduler.priorities
    assert np.array_equal(other.reference_times, res.reference_times)
    for p in POLICY_ORDER:
        assert np.array_equal(other.primary.arrays[p].error, res.primary.arrays[p].error, equal_nan=True)
        assert np.array_equal(other.primary.arrays[p].reference_delta, res.primary.arrays[p].reference_delta, equal_nan=True)
    assert other.model_diagnostic.log_loss != res.model_diagnostic.log_loss  # only the post-freeze diagnostic moves


def test_priorities_are_shared_frozen_and_reproducible(run):
    tmp, trace, registry, res = run
    sched = res.scheduler
    assert len(sched.priorities.priorities) == len(sched.eligible_pairs) == 5
    with pytest.raises(dataclasses.FrozenInstanceError):
        sched.priorities.priorities[0].priority_score = 1.0
    again = run_three_policy_evaluation(fast_config(tmp, output_dir=tmp / "again"), trace=trace, registry=registry)
    assert again.scheduler.priorities == sched.priorities  # identical inputs -> identical priorities
    assert again.primary.metrics["drift-aware"]["error_mean"].estimate == res.primary.metrics["drift-aware"]["error_mean"].estimate


def test_only_comparable_pairs_are_ever_scheduled(run):
    _, _, registry, res = run
    banned = {label(1), label(2048)}
    for p in POLICY_ORDER:
        a = res.primary.arrays[p]
        assert not any({x, y} == banned for x, y in zip(a.circuit_a, a.circuit_b))
        assert all(registry.is_eligible(x, y) for x, y in zip(a.circuit_a, a.circuit_b))
    assert res.scheduler.excluded_pair_count == 1


# ---------------------------------------------------------------- priority-analysis cross-check


def test_priority_analysis_cross_check(run, tmp_path):
    _, _, _, res = run
    table = res.scheduler.priorities
    missing = compare_with_priority_analysis(table, tmp_path / "absent.csv")
    assert missing["found"] is False and missing["consistent"] is None

    def write(path, mutate=None):
        rows = [{"rank": str(p.rank), "circuit_a": p.circuit_a, "circuit_b": p.circuit_b,
                 "priority_score": repr(p.priority_score)} for p in table.priorities]
        if mutate:
            mutate(rows)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    good = tmp_path / "good.csv"
    write(good)
    assert compare_with_priority_analysis(table, good)["consistent"] is True

    bad = tmp_path / "bad.csv"
    write(bad, lambda rows: rows[0].update(priority_score="123.0"))
    result = compare_with_priority_analysis(table, bad)
    assert result["consistent"] is False and result["problems"]

    short = tmp_path / "short.csv"
    write(short, lambda rows: rows.pop())
    assert compare_with_priority_analysis(table, short)["consistent"] is False  # a pair is missing from the file


# ---------------------------------------------------------------- outputs and safeguards


def test_outputs_are_complete_strict_and_labelled(run, tmp_path):
    _, _, _, res = run
    out = tmp_path / "written"
    files = write_outputs(res, out)
    figs = write_figures(res, out)
    assert set(f.name for f in files.values()) == set(OUTPUT_FILES)
    assert len(figs) == 5 and all(f.stat().st_size > 4000 for f in figs)
    data = json.loads((out / "results.json").read_text(encoding="utf-8"),
                      parse_constant=lambda c: pytest.fail(f"non-finite constant {c}"))
    assert "NOT a real-QPU" in data["evaluation_kind"] and set(data["provenance_labels"]) == {
        "recorded_trace_timing_replay", "model_simulated", "synthetic_stress"}
    assert data["reliability_threshold"] == 0.05 and data["definitions"]["window_seconds"] == 600.0
    assert data["eligibility"]["authority"] == "reviewer_decision" and data["fairness_audit"]["all_passed"] is True
    assert set(data["primary"]["policies"]) == set(POLICY_ORDER)
    assert data["stress_scenarios"][0]["kind"] == "synthetic-stress"
    assert data["priorities_for_this_run"]["metadata"]["training_fingerprint"] and data["config"]["seed"] == 20260101
    assert set(data["csv_sha256"]) == {"policy_comparisons.csv", "comparison_errors.csv", "queue_metrics.csv", "reliability_metrics.csv"}
    n_requests = res.primary.arrays["arrival-order"].n_requests
    with open(out / "policy_comparisons.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * n_requests and {r["provenance"] for r in rows} == {PROVENANCE_REPLAY}
    assert {r["status"] for r in rows} <= {"completed", "incomplete"}
    with open(out / "comparison_errors.csv", encoding="utf-8", newline="") as f:
        errs = list(csv.DictReader(f))
    assert len(errs) == 3 * n_requests and {r["provenance"] for r in errs} == {PROVENANCE_MODEL}
    done = [r for r in errs if r["status"] == "completed"]
    assert all(abs(abs(float(r["measured_delta"]) - float(r["reference_delta"])) - float(r["absolute_error"])) < 1e-12 for r in done)
    assert all((float(r["absolute_error"]) <= 0.05) == (r["reliable"] == "True") for r in done)
    for name, want_queue in (("queue_metrics.csv", True), ("reliability_metrics.csv", False)):
        with open(out / name, encoding="utf-8", newline="") as f:
            metric_rows = list(csv.DictReader(f))
        metrics = {r["metric"] for r in metric_rows}
        assert (("waiting_mean" in metrics) and ("idle_slots" in metrics)) == want_queue
        assert (("reliable_throughput_per_device_hour" in metrics) and ("error_mean" in metrics)) == (not want_queue)
        assert {r["row_type"] for r in metric_rows} == {"policy", "paired_difference"}
        assert any(r["scenario"].startswith("synthetic-stress") for r in metric_rows)


def test_existing_results_are_never_overwritten_by_default(run, tmp_path):
    _, _, _, res = run
    out = tmp_path / "keep"
    write_outputs(res, out)
    before = (out / "results.json").read_bytes()
    with pytest.raises(ThreePolicyError, match="already contains results"):
        write_outputs(res, out)
    with pytest.raises(ThreePolicyError, match="already contains results"):
        check_output_dir(out, overwrite=False)
    assert (out / "results.json").read_bytes() == before  # untouched
    write_outputs(res, out, overwrite=True)  # explicit opt-in
    check_output_dir(tmp_path / "does_not_exist_yet", overwrite=False)


def test_invalid_configuration_fails_before_any_computation(tmp_path):
    missing = tmp_path / "no_such_dataset.txt"
    base = dict(dataset_path=missing, registry_path=tmp_path / "no_registry.csv", output_dir=tmp_path / "o")
    with pytest.raises(SplitError, match="train_fraction"):
        run_three_policy_evaluation(ThreePolicyConfig(train_fraction=1.5, **base))
    with pytest.raises(SplitError, match="window_seconds"):
        run_three_policy_evaluation(ThreePolicyConfig(window_seconds=0.0, **base))
    for bad in ((0.0,), (-2.0,), (1.0,), (float("nan"),)):
        with pytest.raises(ThreePolicyError, match="stress drift-rate factors"):
            run_three_policy_evaluation(ThreePolicyConfig(stress_drift_rate_factors=bad, **base))
    with pytest.raises(ThreePolicyError, match="n_bootstrap"):
        run_three_policy_evaluation(ThreePolicyConfig(n_bootstrap=10, **base))
    with pytest.raises(ThreePolicyError, match="slots_per_shot"):
        run_three_policy_evaluation(ThreePolicyConfig(slots_per_shot=0.0, **base))
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "results.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ThreePolicyError, match="already contains results"):
        run_three_policy_evaluation(ThreePolicyConfig(**{**base, "output_dir": tmp_path / "taken"}))


def test_no_comparable_pairs_is_an_explicit_error(tmp_path):
    trace, _ = synthetic_inputs(rounds=300)
    nothing = make_registry(KS, not_comparable=[(a, b) for i, a in enumerate(KS) for b in KS[i + 1:]])
    with pytest.raises(ThreePolicyError, match="no 'Comparable' pairs"):
        run_three_policy_evaluation(fast_config(tmp_path), trace=trace, registry=nothing)


def test_results_json_is_deterministic_apart_from_the_environment(run):
    tmp, trace, registry, res = run
    again = run_three_policy_evaluation(fast_config(tmp, output_dir=tmp / "det"), trace=trace, registry=registry)
    a, b = build_results(res), build_results(again)
    for key in a:
        if key not in ("environment", "config"):  # config holds the output directory
            assert a[key] == b[key], key


# ---------------------------------------------------------------- command line (fail-fast paths only)


def load_cli():
    spec = importlib.util.spec_from_file_location("run_three_policy_evaluation", ROOT / "scripts" / "run_three_policy_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_rejects_bad_arguments_without_running_the_experiment(tmp_path, capsys):
    cli = load_cli()
    out = str(tmp_path / "cli_out")
    for args, message in (
        (["--train-fraction", "2"], "train_fraction"),
        (["--drift-rate-factors", "0"], "stress drift-rate factors"),
        (["--drift-rate-factors", "1"], "stress drift-rate factors"),
        (["--n-bootstrap", "5"], "n_bootstrap"),
    ):
        assert cli.main(["--output-dir", out, *args]) == 1
        assert message in capsys.readouterr().err
    (tmp_path / "cli_taken").mkdir()
    (tmp_path / "cli_taken" / "results.json").write_text("{}", encoding="utf-8")
    assert cli.main(["--output-dir", str(tmp_path / "cli_taken")]) == 1
    assert "already contains results" in capsys.readouterr().err
    assert not (tmp_path / "cli_out").exists()  # nothing was created
    assert cli._factors("none") == () and cli._factors("10, 100") == (10.0, 100.0)
