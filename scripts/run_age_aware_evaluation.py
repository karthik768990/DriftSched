#!/usr/bin/env python3
"""Experimental ``drift-aware-age`` policy versus the three existing policies on the frozen workload.

    python scripts/run_age_aware_evaluation.py

1. Calibrates the age horizon H from the TRAINING partition only (pre-registered rule, see analysis/age_calibration.py).
2. Replays the frozen predetermined workload (fingerprint 16efca32...) through arrival-order, consecutive-pair and
   drift-aware (production simulator, unchanged) and drift-aware-age (offline replica simulator that accepts a
   time-dependent priority; checked equal to production for the static-key policies) under the four capacity
   scenarios, with the same device, workload, drift inputs, comparison reference and training-only priorities.
3. Audits every run (a failed audit invalidates the evaluation) and writes metrics, per-request diagnostics,
   comparison tables, plots and the calibration record.

The holdout is never used to choose H. Model-derived comparison errors are not measured reliability; no capacity
or workload definition is changed. Writes only to results/age_aware_scheduler_evaluation/; refuses to overwrite.
"""

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantum_drift_scheduler.analysis import scheduler_diagnostics as sd  # noqa: E402
from quantum_drift_scheduler.analysis.age_calibration import calibrate  # noqa: E402
from quantum_drift_scheduler.analysis.starvation_metrics import overtaking_counts, request_starts, starvation_summary  # noqa: E402
from quantum_drift_scheduler.config import DEFAULT_CONFIG  # noqa: E402
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry  # noqa: E402
from quantum_drift_scheduler.evaluation import run_holdout_evaluation  # noqa: E402
from quantum_drift_scheduler.modeling.capacity_evaluation import scenario_policy_metrics  # noqa: E402
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD, FullEvaluationConfig  # noqa: E402
from quantum_drift_scheduler.modeling.observations import observations_from_shots  # noqa: E402
from quantum_drift_scheduler.modeling.pipeline import fit_training_stage  # noqa: E402
from quantum_drift_scheduler.modeling.predetermined_evaluation import drift_fingerprint, trace_rows  # noqa: E402
from quantum_drift_scheduler.modeling.reporting import to_jsonable  # noqa: E402
from quantum_drift_scheduler.scheduling import (  # noqa: E402
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    load_recorded_trace,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.age_aware_policy import DRIFT_AWARE_AGE, DriftAgePolicy  # noqa: E402
from quantum_drift_scheduler.scheduling.capacity_scenarios import (  # noqa: E402
    audit_scenario,
    default_scenarios,
    requests_fingerprint,
    scenario_workload,
)
from quantum_drift_scheduler.scheduling.policies import request_order_key  # noqa: E402
from quantum_drift_scheduler.scheduling.predetermined_workload import (  # noqa: E402
    AuditCheck,
    audit_fairness,
    build_predetermined_workload,
    load_workload,
    make_policy_input,
    to_simulator_input,
)

OUT_DEFAULT = DEFAULT_CONFIG.base_dir / "results" / "age_aware_scheduler_evaluation"
FROZEN_PATH = DEFAULT_CONFIG.base_dir / "results" / "predetermined_workload_evaluation" / "frozen_workload.json"
RISK2_METRICS = DEFAULT_CONFIG.base_dir / "results" / "capacity_sensitivity_evaluation" / "policy_metrics_by_scenario.csv"
EXPECTED_FINGERPRINT = "16efca32e41087df7255ea9e7cfcd71802f2fc192763fee63cea5b5cefcc3c86"
EXISTING = ("arrival-order", "consecutive-pair", "drift-aware")
POLICIES = EXISTING + (DRIFT_AWARE_AGE,)
COLORS = {"arrival-order": "gray", "consecutive-pair": "C0", "drift-aware": "C3", DRIFT_AWARE_AGE: "C2"}
WINDOW_SECONDS = 600.0
COMPARE_METRICS = (
    "completed_comparisons", "completion_rate", "unfinished_requests", "shot_wait_from_release_mean_s", "shot_wait_from_release_median_s",
    "wait_completed_mean_s", "wait_completed_median_s", "wait_completed_p95_s", "wait_completed_p99_s", "wait_completed_max_s",
    "waiting_over_10s_incl_unfinished", "waiting_over_30s_incl_unfinished", "waiting_over_60s_incl_unfinished",
    "unfinished_age_max_s", "unfinished_not_served_in_time", "overtaken_mean", "overtaken_max", "separation_mean_s", "separation_median_s",
    "separation_mean_slots", "device_utilisation_pct", "idle_slots", "queue_mean", "model_error_mean",
    "reliable_per_device_hour_model", "reliable_per_busy_device_hour_model",
)
REPRODUCED = ("completed_comparisons", "shot_wait_from_release_mean_s", "shot_wait_from_release_max_s", "separation_mean_s", "idle_slots",
              "reliable_per_device_hour_model", "model_error_mean")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    cols = list(columns or list(dict.fromkeys(k for r in rows for k in r)))  # union of keys, first-seen order
    with open(path, "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None or (isinstance(r.get(c), float) and not math.isfinite(r[c])) else
                        (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.output_dir
    if ((out / "summary.json").exists() or (out / "INVALID_fairness_audit_failed.csv").exists()) and not args.overwrite:
        print(f"error: {out} already contains a run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    ev = run_holdout_evaluation(trace, registry)
    frozen = load_workload(FROZEN_PATH)
    if frozen.fingerprint != EXPECTED_FINGERPRINT:
        print(f"error: frozen workload fingerprint {frozen.fingerprint} != {EXPECTED_FINGERPRINT}", file=sys.stderr)
        return 3
    consistent = build_predetermined_workload(ev.requests, ev.schedule, registry, dict(frozen.config)).fingerprint == frozen.fingerprint

    # ---- 1. training-only calibration of H (the holdout workload is not an input)
    say("calibrating the age horizon on the training partition")
    tw = sd.build_workload("training", ev.split.train, registry, trace)
    train_pw = build_predetermined_workload(tw.requests, tw.schedule, registry)
    train_sim = to_simulator_input(train_pw)
    train_workload = sd.Workload("training", train_sim.shots, train_sim.requests, train_sim.schedule)
    cal = calibrate(train_workload, ev.priorities, ev.split.boundary)
    cfg = cal.config
    say(f"selected k={cal.selected_rounds} rounds: H = {cfg.horizon_seconds:.4f} s, R = {cfg.priority_range:.4f}, "
        f"coefficient {cfg.coefficient_per_second:.5f} per s; provisional = {cal.provisional}")
    write_csv(out / "calibration_candidates.csv", cal.candidates)
    params = {
        "policy": DRIFT_AWARE_AGE,
        "formula": "effective(r,t) = drift_priority(r) + R * max(0, t - release_r) / H; highest effective priority first; ties by request_order_key",
        "H_horizon_seconds": cfg.horizon_seconds, "R_priority_range": cfg.priority_range, "coefficient_per_second": cfg.coefficient_per_second,
        "tau_round_seconds": cal.tau_round_seconds, "selected_rounds_k": cal.selected_rounds, "candidate_rounds": [r["rounds_k"] for r in cal.candidates],
        "selection_criterion": cal.criterion, "provisional": cal.provisional, "reference_runs_on_training": cal.reference,
        "calibration_data": f"training partition only: {len(train_workload.requests)} requests released before the holdout boundary {ev.split.boundary!r}",
        "holdout_used_for_selection": False,
        "overtake_bound": "a request can only be overtaken by requests released at most H seconds after it (conditional bound; not a maximum-wait guarantee)",
    }
    (out / "age_parameters.json").write_text(json.dumps(to_jsonable(params), indent=1, allow_nan=False), encoding="utf-8")

    # ---- 2. model and shared inputs
    say("fitting the drift model on the training partition (identical in every scenario)")
    stage = fit_training_stage(observations_from_shots(ev.split.train), FullEvaluationConfig(), say)
    model, origin = stage.model, ev.split.min_timestamp
    drift_fp = drift_fingerprint(model, stage.basis)
    references = tuple(r.planned_time for r in frozen.requests)
    scenarios = default_scenarios(frozen.device.slot_seconds)
    base_sim = to_simulator_input(frozen, input_source=f"frozen-workload:{frozen.fingerprint[:16]}")
    pri = ev.priorities
    age_policy = DriftAgePolicy(pri, cfg)
    production = {"arrival-order": ArrivalOrderPolicy(), "consecutive-pair": ConsecutivePairPolicy(), "drift-aware": DriftAwarePairPolicy(pri)}
    risk2 = list(csv.DictReader(open(RISK2_METRICS, encoding="utf-8"))) if RISK2_METRICS.exists() else []

    metrics: List[Dict[str, Any]] = []
    per_request: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    scores = {q.request_id: pri.score(q.first.circuit, q.second.circuit) for q in base_sim.requests}
    for sc in scenarios:
        sw = scenario_workload(frozen, sc)
        sim = dataclasses.replace(base_sim, schedule=sw.device.to_schedule())
        wl = sd.Workload("frozen", sim.shots, sim.requests, sim.schedule)
        results = {n: run_simulation(sim.shots, registry, p, schedule=sim.schedule, requests=sim.requests) for n, p in production.items()}
        results[DRIFT_AWARE_AGE] = sd.simulate_pair_dispatch(wl, DRIFT_AWARE_AGE, age_policy.chooser())
        inputs = {n: make_policy_input(n, sim, registry, drift_fp, references) for n in POLICIES}
        checks: List[AuditCheck] = list(audit_scenario(frozen, EXPECTED_FINGERPRINT, sc, inputs)) + list(
            audit_fairness(sw, inputs, results, drift_fp, references).checks)
        # replica fidelity: the offline simulator reproduces production for the static-key policies
        for n, key in (("consecutive-pair", request_order_key),
                       ("drift-aware", lambda q: (-pri.score(q.first.circuit, q.second.circuit),) + request_order_key(q))):
            rep = sd.simulate_pair_dispatch(wl, n, sd.chooser_from_key(key))
            same = [(c.request_id, c.slot_a, c.slot_b) for c in rep.comparisons] == [(c.request_id, c.slot_a, c.slot_b) for c in results[n].comparisons] \
                and rep.idle_slots == results[n].idle_slots
            checks.append(AuditCheck("offline simulator reproduces production", n, same, "identical executions" if same else "executions differ"))
        age_res = results[DRIFT_AWARE_AGE]
        atomic = all(c.slot_b == c.slot_a + 1 for c in age_res.comparisons)
        checks.append(AuditCheck("pair atomicity preserved (consecutive slots)", DRIFT_AWARE_AGE, atomic, "all pairs in consecutive slots" if atomic else "non-adjacent pair"))
        eligible = all(registry.is_eligible(c.circuit_a, c.circuit_b) for c in age_res.comparisons)
        checks.append(AuditCheck("only Comparable pairs executed", DRIFT_AWARE_AGE, eligible, "ok" if eligible else "ineligible pair executed"))
        shots = [s for c in age_res.comparisons for s in (c.shot_id_a, c.shot_id_b)]
        checks.append(AuditCheck("no shot reused; budget respected", DRIFT_AWARE_AGE, len(shots) == len(set(shots)) and len(age_res.executed) <= frozen.total_shot_budget, "ok"))
        for c in checks:
            audit_rows.append({"scenario": sc.name, "policy": c.policy, "check": c.name, "passed": c.passed, "detail": c.detail})
        for n in POLICIES:
            rows = trace_rows(sw, results[n], model, origin, WINDOW_SECONDS, RELIABILITY_THRESHOLD, references)
            m = scenario_policy_metrics(sw, sc, results[n], rows, RELIABILITY_THRESHOLD)
            horizon = cfg.horizon_seconds if n == DRIFT_AWARE_AGE else None
            end = sw.device.start_time + (sw.device.total_slots - 1) * sw.device.slot_seconds
            m.update(starvation_summary(rows, end, sw.device.slot_seconds, horizon))
            m["workload_fingerprint_frozen"] = frozen.fingerprint
            metrics.append(m)
            starts = request_starts(rows)
            oc = overtaking_counts([r["release_time"] for r in rows], starts)["counts"]
            for r, s, c in zip(rows, starts, oc):
                age = None if s is not None else max(end - r["release_time"], 0.0)
                cls = "completed" if s is not None else ("unfinished_truncated_by_horizon" if age < 2 * sw.device.slot_seconds else "unfinished_not_served_in_time")
                per_request.append({"scenario": sc.name, "policy": n, "request_id": r["request_id"], "circuit_first": r["circuit_first"],
                                    "circuit_second": r["circuit_second"], "drift_priority": scores[r["request_id"]], "release_time": r["release_time"],
                                    "planned_time": r["planned_time"], "completed": s is not None, "start_time": s,
                                    "request_wait_s": None if s is None else s - r["release_time"], "observed_age_if_unfinished_s": age,
                                    "overtaken_count": c, "class": cls})
        say(f"{sc.name}: audit {sum(c.passed for c in checks)}/{len(checks)}")

    # baseline reproduction of the capacity-sensitivity results for the three existing policies
    for m in metrics:
        if m["policy"] in EXISTING and risk2:
            prev = next((r for r in risk2 if r["scenario"] == m["scenario"] and r["policy"] == m["policy"]), None)
            same = prev is not None and all(math.isclose(float(m[k]), float(prev[k]), rel_tol=1e-12, abs_tol=1e-12) for k in REPRODUCED)
            audit_rows.append({"scenario": m["scenario"], "policy": m["policy"], "check": "reproduces the capacity-sensitivity evaluation", "passed": bool(same),
                               "detail": "identical to results/capacity_sensitivity_evaluation" if same else "differs from the earlier run"})
    audit_rows.append({"scenario": "all", "policy": "all", "check": "frozen workload equals an independent rebuild from the production pipeline",
                       "passed": consistent, "detail": frozen.fingerprint[:16]})
    audit_rows.append({"scenario": "all", "policy": DRIFT_AWARE_AGE, "check": "age horizon selected from training data only",
                       "passed": not params["holdout_used_for_selection"], "detail": params["calibration_data"]})
    write_csv(out / "fairness_audit.csv", audit_rows)
    failed = [r for r in audit_rows if not r["passed"]]
    say(f"fairness audits: {len(audit_rows) - len(failed)}/{len(audit_rows)} passed")
    if failed:
        write_csv(out / "INVALID_fairness_audit_failed.csv", failed)
        print(f"EVALUATION INVALID: {len(failed)} failed checks, first: {failed[0]}", file=sys.stderr)
        return 2

    # ---- 3. tables
    write_csv(out / "policy_metrics_by_scenario.csv", metrics)
    write_csv(out / "per_request_diagnostics.csv", per_request)
    by = {(m["scenario"], m["policy"]): m for m in metrics}
    cmp_rows = []
    for sc in scenarios:
        for other in EXISTING:
            for k in COMPARE_METRICS:
                a, b = by[(sc.name, DRIFT_AWARE_AGE)].get(k), by[(sc.name, other)].get(k)
                cmp_rows.append({"scenario": sc.name, "service_seconds": sc.service_seconds, "policy_a": DRIFT_AWARE_AGE, "policy_b": other,
                                 "metric": k, "value_a": a, "value_b": b, "difference_a_minus_b": None if a is None or b is None else a - b})
    write_csv(out / "policy_comparison.csv", cmp_rows)
    cfg_out = {"parameters": params, "scenarios": [sc.to_dict() for sc in scenarios], "frozen_workload_fingerprint": frozen.fingerprint,
               "drift_input_fingerprint": drift_fp, "reliability_threshold": RELIABILITY_THRESHOLD,
               "limitations": ["no capacity value is verified hardware capacity", "recorded outcomes replayed; not a QPU experiment",
                               "model-derived comparison error is not measured reliability; the drift model has not shown deployment-faithful skill",
                               "one frozen workload, deterministic replay", "H selected from the training workload at one operating point"]}
    (out / "summary.json").write_text(json.dumps(to_jsonable({"config": cfg_out, "metrics": metrics, "audits": {"checks": len(audit_rows), "failed": 0}}), indent=1, allow_nan=False), encoding="utf-8")

    # ---- 4. plots
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)

    pr = {(r["scenario"], r["policy"]): [] for r in per_request}
    for r in per_request:
        v = r["request_wait_s"] if r["completed"] else r["observed_age_if_unfinished_s"]
        pr[(r["scenario"], r["policy"])].append(v)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    for a, sc in zip(ax, scenarios):
        for n in POLICIES:
            xs = np.sort(np.array(pr[(sc.name, n)]))
            a.plot(np.maximum(xs, 1e-2), 1 - np.arange(1, xs.size + 1) / xs.size, color=COLORS[n], label=n, lw=1.4)
        for t in (10, 30, 60):
            a.axvline(t, color="k", ls=":", lw=0.5)
        a.set(xscale="log", yscale="log", xlabel="request wait (s; unfinished at observed age)", title=sc.name)
        a.set_ylim(bottom=1e-4)
    ax[0].set_ylabel("fraction of requests waiting longer")
    ax[0].legend(fontsize=7)
    save(fig, "waiting_time_tails.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    x = np.arange(len(scenarios))
    for j, n in enumerate(POLICIES):
        for a, key, ttl in ((ax[0], "waiting_over_30s_incl_unfinished", "requests waiting > 30 s (incl. unfinished)"),
                            (ax[1], "waiting_over_60s_incl_unfinished", "requests waiting > 60 s (incl. unfinished)"),
                            (ax[2], "unfinished_not_served_in_time", "unfinished, not truncated by the horizon")):
            a.bar(x + (j - 1.5) * 0.2, [by[(sc.name, n)][key] for sc in scenarios], 0.2, color=COLORS[n], label=n)
            a.set_title(ttl, fontsize=9)
            a.set_xticks(x)
            a.set_xticklabels([sc.name.replace("_", "\n") for sc in scenarios], fontsize=6)
            a.set_yscale("symlog", linthresh=1)
    ax[0].legend(fontsize=7)
    save(fig, "starvation_indicators.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    for n in POLICIES:
        for a, key, ttl in ((ax[0], "wait_completed_p99_s", "p99 completed-request wait (s)"), (ax[1], "wait_completed_max_s", "maximum completed-request wait (s)"),
                            (ax[2], "separation_mean_slots", "mean pair separation (slots)")):
            a.plot(x, [by[(sc.name, n)][key] for sc in scenarios], "o-", color=COLORS[n], label=n)
            a.set_title(ttl, fontsize=9)
            a.set_xticks(x)
            a.set_xticklabels([sc.name.replace("_", "\n") for sc in scenarios], fontsize=6)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].legend(fontsize=7)
    save(fig, "tails_and_separation_by_scenario.png")

    say(f"wrote {out}")
    for sc in scenarios:
        for n in POLICIES:
            m = by[(sc.name, n)]
            print(f"  {sc.name:27s} {n:17s} done {m['completed_comparisons']:5d} unfin {m['unfinished_requests']:3d}(not-served {m['unfinished_not_served_in_time']:3d})  "
                  f"wait p99/max {m['wait_completed_p99_s']:8.2f}/{m['wait_completed_max_s']:8.2f}  >30s {m['waiting_over_30s_incl_unfinished']:4d} >60s {m['waiting_over_60s_incl_unfinished']:4d}  "
                  f"overtaken max {m['overtaken_max']:4d}  sep {m['separation_mean_slots']:.2f}sl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
