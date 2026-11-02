#!/usr/bin/env python3
"""Explicit device-capacity sensitivity of the three existing policies on the frozen workload.

    python scripts/run_capacity_sensitivity_evaluation.py

Loads the frozen predetermined workload of the previous experiment (read-only, fingerprint-verified), then
replays it through arrival-order, consecutive-pair and drift-aware under four explicit service-duration
scenarios. The workload (requests, release and planned times, shot budget, eligibility) is never changed; only
the device's slot grid does, over an unchanged wall-clock horizon (see ``scheduling/capacity_scenarios.py``).
Every scenario/policy run is audited; a failed audit invalidates the evaluation (non-zero exit, no metrics).

No service duration is verified hardware capacity. The experiment replays recorded outcomes with
drift-model-derived comparison errors; it is not a QPU experiment and not a multiple-workload study.
Writes only to results/capacity_sensitivity_evaluation/ and refuses to overwrite an earlier run.
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
from quantum_drift_scheduler.scheduling.capacity_scenarios import (  # noqa: E402
    audit_scenario,
    default_scenarios,
    device_timeline_stats,
    requests_fingerprint,
    scenario_workload,
)
from quantum_drift_scheduler.scheduling.predetermined_workload import (  # noqa: E402
    audit_fairness,
    build_predetermined_workload,
    load_workload,
    make_policy_input,
    to_simulator_input,
)

OUT_DEFAULT = DEFAULT_CONFIG.base_dir / "results" / "capacity_sensitivity_evaluation"
FROZEN_PATH = DEFAULT_CONFIG.base_dir / "results" / "predetermined_workload_evaluation" / "frozen_workload.json"
RISK1_SUMMARY = DEFAULT_CONFIG.base_dir / "results" / "predetermined_workload_evaluation" / "summary.json"
EXPECTED_FINGERPRINT = "16efca32e41087df7255ea9e7cfcd71802f2fc192763fee63cea5b5cefcc3c86"
POLICIES = ("arrival-order", "consecutive-pair", "drift-aware")
COLORS = {"arrival-order": "gray", "consecutive-pair": "C0", "drift-aware": "C3"}
WINDOW_SECONDS = 600.0
PAIRS = (("consecutive-pair", "arrival-order"), ("drift-aware", "arrival-order"), ("drift-aware", "consecutive-pair"))
DIFF_METRICS = (
    "completed_comparisons", "completion_rate", "unfinished_requests", "shot_wait_from_release_mean_s", "shot_wait_from_release_median_s",
    "shot_wait_from_release_p95_s", "shot_wait_from_release_max_s", "request_wait_mean_s", "separation_mean_s", "separation_median_s",
    "separation_mean_slots", "device_utilisation_pct", "idle_slots", "queue_mean", "queue_max", "model_error_mean",
    "reliable_comparisons_model", "reliable_per_device_hour_model", "reliable_per_busy_device_hour_model",
)
RANKED = (  # metric, higher_is_better
    ("completed_comparisons", True), ("reliable_per_device_hour_model", True), ("reliable_per_busy_device_hour_model", True),
    ("shot_wait_from_release_mean_s", False), ("shot_wait_from_release_p95_s", False), ("shot_wait_from_release_max_s", False),
    ("separation_mean_s", False), ("queue_mean", False), ("model_error_mean", False),
)
TIE_REL = 1e-9


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    cols = list(columns or (rows[0].keys() if rows else []))
    with open(path, "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow(["" if r.get(c) is None or (isinstance(r.get(c), float) and not math.isfinite(r[c])) else
                        (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])


def rank_policies(values: Dict[str, float], higher_is_better: bool) -> Dict[str, int]:
    """Dense-by-position rank (1 = best); policies within a relative tolerance of each other share a rank."""
    order = sorted(values, key=lambda p: -values[p] if higher_is_better else values[p])
    ranks, current = {}, 1
    for i, p in enumerate(order):
        if i and abs(values[p] - values[order[i - 1]]) > TIE_REL * max(abs(values[p]), abs(values[order[i - 1]]), 1e-300):
            current = i + 1
        ranks[p] = current
    return ranks


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    out = args.output_dir
    if ((out / "summary.json").exists() or (out / "INVALID_fairness_audit_failed.csv").exists()) and not args.overwrite:
        print(f"error: {out} already contains a run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    (out / "execution_traces").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    ev = run_holdout_evaluation(trace, registry)  # production pipeline: split, training-only priorities (frozen), requests
    frozen = load_workload(FROZEN_PATH)  # read-only; fingerprint-verified on load
    if frozen.fingerprint != EXPECTED_FINGERPRINT:
        print(f"error: frozen workload fingerprint {frozen.fingerprint} != expected {EXPECTED_FINGERPRINT}", file=sys.stderr)
        return 3
    rebuilt = build_predetermined_workload(ev.requests, ev.schedule, registry, dict(frozen.config))
    consistent = rebuilt.fingerprint == frozen.fingerprint
    say(f"frozen workload: {frozen.request_count} requests, {frozen.eligible_pair_count} pairs, {frozen.total_shot_budget} shots, "
        f"sha256 {frozen.fingerprint[:16]}; independently rebuilt from the production pipeline: {'identical' if consistent else 'DIFFERENT'}")
    scenarios = default_scenarios(frozen.device.slot_seconds)

    say("fitting the drift model on the training partition (identical in every scenario)")
    stage = fit_training_stage(observations_from_shots(ev.split.train), FullEvaluationConfig(), say)
    model, origin = stage.model, ev.split.min_timestamp
    drift_fp = drift_fingerprint(model, stage.basis)
    references = tuple(r.planned_time for r in frozen.requests)

    base_sim = to_simulator_input(frozen, input_source=f"frozen-workload:{frozen.fingerprint[:16]}")  # ONE object set for all 12 runs
    policy_objects = {"arrival-order": ArrivalOrderPolicy(), "consecutive-pair": ConsecutivePairPolicy(),
                      "drift-aware": DriftAwarePairPolicy(ev.priorities)}

    metrics: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    for sc in scenarios:
        sw = scenario_workload(frozen, sc)
        sim = dataclasses.replace(base_sim, schedule=sw.device.to_schedule())
        inputs = {n: make_policy_input(n, sim, registry, drift_fp, references) for n in POLICIES}
        results = {n: run_simulation(sim.shots, registry, p, schedule=sim.schedule, requests=sim.requests) for n, p in policy_objects.items()}
        checks = list(audit_scenario(frozen, EXPECTED_FINGERPRINT, sc, inputs)) + list(audit_fairness(sw, inputs, results, drift_fp, references).checks)
        for c in checks:
            audit_rows.append({"scenario": sc.name, "policy": c.policy, "check": c.name, "passed": c.passed, "detail": c.detail})
        timeline.append(device_timeline_stats(frozen, sc))
        for n in POLICIES:
            rows = trace_rows(sw, results[n], model, origin, WINDOW_SECONDS, RELIABILITY_THRESHOLD, references)
            m = scenario_policy_metrics(sw, sc, results[n], rows, RELIABILITY_THRESHOLD)
            # unfinished requests that were released too close to the horizon end for ANY device to serve them (end-of-horizon truncation)
            unfinished = [r for r in rows if not r["completed"]]
            tail = float(sc.service_seconds * 2)  # a pair needs two slots
            m["unfinished_released_less_than_two_slots_before_horizon_end"] = int(sum((timeline[-1]["horizon_end"] - r["release_time"]) < tail for r in unfinished))
            m["unfinished_min_seconds_release_to_horizon_end"] = float(min((timeline[-1]["horizon_end"] - r["release_time"]) for r in unfinished)) if unfinished else None
            m["unfinished_max_seconds_release_to_horizon_end"] = float(max((timeline[-1]["horizon_end"] - r["release_time"]) for r in unfinished)) if unfinished else None
            m["workload_fingerprint_frozen"] = frozen.fingerprint
            m["requests_fingerprint"] = requests_fingerprint(sw)
            metrics.append(m)
            write_csv(out / "execution_traces" / f"{sc.name}__{n.replace('-', '_')}.csv", rows)
        say(f"{sc.name}: {sw.device.total_slots} slots x {sc.service_seconds:.6f} s; audit {sum(c.passed for c in checks)}/{len(checks)}")

    # ---- baseline must reproduce the previous experiment
    if RISK1_SUMMARY.exists():
        prev = json.load(open(RISK1_SUMMARY, encoding="utf-8"))["policies"]
        for n in POLICIES:
            cur = next(m for m in metrics if m["scenario"] == "historical_baseline" and m["policy"] == n)
            same = all(math.isclose(cur[k], prev[n][k], rel_tol=1e-12, abs_tol=1e-12) for k in (
                "completed_comparisons", "separation_mean_s", "shot_wait_from_release_mean_s", "shot_wait_from_release_p95_s",
                "shot_wait_from_release_max_s", "idle_slots", "reliable_per_device_hour_model", "model_error_mean"))
            audit_rows.append({"scenario": "historical_baseline", "policy": n, "check": "baseline reproduces the previous predetermined-workload evaluation",
                               "passed": bool(same), "detail": "identical to results/predetermined_workload_evaluation/summary.json" if same else "differs from the previous run"})
    audit_rows.append({"scenario": "all", "policy": "all", "check": "frozen workload equals an independent rebuild from the production pipeline",
                       "passed": consistent, "detail": f"{rebuilt.fingerprint[:16]} vs {frozen.fingerprint[:16]}"})
    write_csv(out / "fairness_audit.csv", audit_rows)
    failed = [r for r in audit_rows if not r["passed"]]
    say(f"fairness audits: {len(audit_rows) - len(failed)}/{len(audit_rows)} passed")
    if failed:
        write_csv(out / "INVALID_fairness_audit_failed.csv", failed)
        print(f"EVALUATION INVALID: {len(failed)} failed checks, first: {failed[0]}", file=sys.stderr)
        return 2

    # ---- tables
    write_csv(out / "policy_metrics_by_scenario.csv", metrics)
    by = {(m["scenario"], m["policy"]): m for m in metrics}
    diffs = []
    for sc in scenarios:
        for a, b in PAIRS:
            for k in DIFF_METRICS:
                va, vb = by[(sc.name, a)][k], by[(sc.name, b)][k]
                diffs.append({"scenario": sc.name, "service_seconds": sc.service_seconds, "policy_a": a, "policy_b": b, "metric": k,
                              "value_a": va, "value_b": vb, "difference_a_minus_b": None if va is None or vb is None else va - vb})
    write_csv(out / "pairwise_policy_differences.csv", diffs)

    ranking = []
    for sc in scenarios:
        for k, hi in RANKED:
            vals = {n: by[(sc.name, n)][k] for n in POLICIES}
            if any(v is None for v in vals.values()):
                continue
            r = rank_policies(vals, hi)
            ranking.append({"scenario": sc.name, "metric": k, "better": "higher" if hi else "lower",
                            **{f"value_{n}": vals[n] for n in POLICIES}, **{f"rank_{n}": r[n] for n in POLICIES},
                            "best": "+".join(n for n in POLICIES if r[n] == 1), "all_tied": len(set(r.values())) == 1})
    write_csv(out / "policy_ranking_by_scenario.csv", ranking)
    changes = {}
    for k, _ in RANKED:
        sets = {tuple(sorted((n, r[f"rank_{n}"]) for n in POLICIES)) for r in ranking if r["metric"] == k for _ in [0]}
        changes[k] = len(sets) > 1
    summary_rows = []
    for t in timeline:
        row = {k: t[k] for k in ("scenario", "service_seconds", "service_rate_shots_per_second", "total_slots", "total_potential_service_hours",
                                  "horizon_seconds", "release_horizon_seconds", "offered_load", "capacity_over_demand")}
        for n in POLICIES:
            m = by[(t["scenario"], n)]
            for k in ("completed_comparisons", "unfinished_requests", "device_utilisation_pct", "shot_wait_from_release_mean_s", "shot_wait_from_release_max_s",
                      "separation_mean_s", "separation_mean_slots", "reliable_per_device_hour_model", "reliable_per_busy_device_hour_model"):
                row[f"{n}__{k}"] = m[k]
        summary_rows.append(row)
    write_csv(out / "capacity_summary.csv", summary_rows)

    cfg = {
        "purpose": "explicit device service duration as an experimental parameter; no duration is verified hardware capacity",
        "scenarios": [dict(sc.to_dict(), timeline=t) for sc, t in zip(scenarios, timeline)],
        "timeline_semantics": "fixed wall-clock horizon: slots = floor((H - start)/d) + 1 over [start, H]; H = start of the last historical slot; release times unchanged",
        "device_time_denominators": {"budget": "total_slots * d (production primary; differs between scenarios by < 1 slot)", "busy": "executed shots * d (actual device time used)"},
        "workload": {"fingerprint_sha256_with_historical_device": frozen.fingerprint, "requests": frozen.request_count,
                     "eligible_pairs": frozen.eligible_pair_count, "total_shot_budget": frozen.total_shot_budget,
                     "requests_fingerprint_device_excluded": requests_fingerprint(frozen), "independent_rebuild_identical": consistent},
        "drift_input_fingerprint": drift_fp, "reliability_threshold": RELIABILITY_THRESHOLD,
        "provenance": {"queue_and_timing": "production simulator replay of the frozen workload on each scenario grid",
                       "comparison_error": "drift-model-derived (model has not shown deployment-faithful forecasting skill); not measured reliability",
                       "outcomes": "recorded outcomes replayed; nothing newly measured"},
        "limitations": ["no service duration is verified hardware capacity", "0.756836322 s is workload-derived (mean inter-arrival gap)",
                        "0.3486 s is a recorded cadence reference", "faster/slower are +/-10% modelling assumptions",
                        "one frozen workload, deterministic replay: not a multiple-workload statistical study",
                        "recorded outcomes are replayed; this is not a QPU experiment"],
    }
    (out / "capacity_scenarios.json").write_text(json.dumps(to_jsonable(cfg), indent=1, allow_nan=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(to_jsonable({"metrics": metrics, "ranking_changes_across_scenarios": changes,
                                                              "audits": {"checks": len(audit_rows), "failed": 0}}), indent=1, allow_nan=False), encoding="utf-8")

    # ---- plots
    xs = [sc.service_seconds for sc in scenarios]
    order = np.argsort(xs)

    def line(ax, key, ylabel, log=False, title=None):
        for n in POLICIES:
            ys = [by[(sc.name, n)][key] for sc in scenarios]
            ax.plot(np.array(xs)[order], np.array(ys, dtype=float)[order], "o-", color=COLORS[n], label=n)
        for sc, x in zip(scenarios, xs):
            ax.annotate(sc.name.replace("_", "\n"), (x, ax.get_ylim()[0]), fontsize=5, ha="center", va="bottom")
        ax.set(xlabel="service duration per slot (s)", ylabel=ylabel, title=title or ylabel)
        if log:
            ax.set_yscale("log")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    line(ax[0], "completed_comparisons", "completed comparisons (of 4,824)")
    line(ax[1], "reliable_per_device_hour_model", "reliable per budget device-hour (model-derived)")
    line(ax[2], "reliable_per_busy_device_hour_model", "reliable per busy device-hour (model-derived)")
    ax[0].legend(fontsize=7)
    save(fig, "completion_and_throughput_vs_service_duration.png")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    line(ax[0], "shot_wait_from_release_mean_s", "mean shot wait from release (s)", log=True)
    line(ax[1], "shot_wait_from_release_p95_s", "p95 shot wait (s)", log=True)
    line(ax[2], "shot_wait_from_release_max_s", "longest shot wait (s)", log=True)
    ax[0].legend(fontsize=7)
    save(fig, "waiting_vs_service_duration.png")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    line(ax[0], "separation_mean_s", "mean paired separation (s)", log=True)
    line(ax[1], "separation_mean_slots", "mean paired separation (slots)")
    line(ax[2], "device_utilisation_pct", "device utilisation (%)")
    ax[0].legend(fontsize=7)
    save(fig, "separation_and_utilisation_vs_service_duration.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    line(ax[0], "queue_mean", "mean queue length (requests waiting)", log=True)
    line(ax[1], "queue_max", "maximum queue length", log=True)
    ax[0].legend(fontsize=7)
    save(fig, "queue_length_vs_service_duration.png")

    say(f"wrote {out}")
    for sc in scenarios:
        for n in POLICIES:
            m = by[(sc.name, n)]
            print(f"  {sc.name:27s} {n:17s} done {m['completed_comparisons']:5d}  util {m['device_utilisation_pct']:6.2f}%  wait mean {m['shot_wait_from_release_mean_s']:8.3f} "
                  f"max {m['shot_wait_from_release_max_s']:9.2f}  sep {m['separation_mean_s']:.4f}s={m['separation_mean_slots']:.2f}sl  reliable/h {m['reliable_per_device_hour_model']:8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
