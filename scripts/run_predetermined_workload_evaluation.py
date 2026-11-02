#!/usr/bin/env python3
"""Fair three-policy replay of ONE predetermined, frozen workload.

    python scripts/run_predetermined_workload_evaluation.py

1. builds the request workload once (release time, planned measurement time, shot requirements) from the
   unchanged production pipeline for the evaluated partition, freezes it to disk and reloads it;
2. hands the very same simulator objects to arrival-order, consecutive-pair and drift-aware (production
   policies, unchanged; priorities frozen from training only);
3. audits the contents of every policy's input and result; a failed audit invalidates the evaluation
   (non-zero exit, no metrics written);
4. scores the timelines (queue metrics from the simulator; drift-model-derived comparison errors against
   the planned measurement time) and compares them with the previous evaluation.

Writes only to results/predetermined_workload_evaluation/ and refuses to overwrite an earlier run.
No production scheduler, model, registry, priority, device-capacity or metric code is changed.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quantum_drift_scheduler.config import DEFAULT_CONFIG  # noqa: E402
from quantum_drift_scheduler.data.compatibility import load_compatibility_registry  # noqa: E402
from quantum_drift_scheduler.evaluation import run_holdout_evaluation  # noqa: E402
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD, FullEvaluationConfig  # noqa: E402
from quantum_drift_scheduler.modeling.observations import observations_from_shots  # noqa: E402
from quantum_drift_scheduler.modeling.pipeline import fit_training_stage  # noqa: E402
from quantum_drift_scheduler.modeling.predetermined_evaluation import drift_fingerprint, policy_summary, trace_rows  # noqa: E402
from quantum_drift_scheduler.modeling.reporting import to_jsonable  # noqa: E402
from quantum_drift_scheduler.scheduling import (  # noqa: E402
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    load_recorded_trace,
    run_simulation,
)
from quantum_drift_scheduler.scheduling.predetermined_workload import (  # noqa: E402
    FairnessAuditError,
    audit_fairness,
    build_predetermined_workload,
    load_workload,
    make_policy_input,
    save_workload,
    to_simulator_input,
)

OUT_DEFAULT = DEFAULT_CONFIG.base_dir / "results" / "predetermined_workload_evaluation"
STORED = DEFAULT_CONFIG.base_dir / "results" / "qcs_three_policy_evaluation_v2" / "results.json"
POLICIES = ("arrival-order", "consecutive-pair", "drift-aware")
WINDOW_SECONDS = 600.0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> None:
    cols = list(columns or (rows[0].keys() if rows else []))
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
    if ((out / "summary.json").exists() or (out / "INVALID_fairness_audit_failed.json").exists()) and not args.overwrite:
        print(f"error: {out} already contains a run; use a new --output-dir or --overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    say = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    registry = load_compatibility_registry()
    trace = load_recorded_trace()
    say("running the production pipeline once for the evaluated partition (split, training-only priorities, requests)")
    ev = run_holdout_evaluation(trace, registry)

    # ---- 1. build, freeze, reload: the artifact is the single source of the workload
    config = {
        "partition": "holdout (last 25% of the timeline, the partition the three-policy evaluation scores)",
        "dataset": str(DEFAULT_CONFIG.default_dataset_path.relative_to(DEFAULT_CONFIG.base_dir).as_posix()),
        "dataset_sha256": sha256_file(DEFAULT_CONFIG.default_dataset_path),
        "registry": str(DEFAULT_CONFIG.compatibility_registry_path.relative_to(DEFAULT_CONFIG.base_dir).as_posix()),
        "registry_sha256": sha256_file(DEFAULT_CONFIG.compatibility_registry_path),
        "eligibility_authority": "reviewer_decision == Comparable (compatibility_status is informational)",
        "train_fraction": ev.split.train_fraction, "split_boundary": ev.split.boundary,
        "slots_per_shot": ev.slots_per_shot, "device_note": "slot grid identical to the previous evaluation (capacity unchanged)",
        "request_pairing": "existing production round-robin matching over the Comparable pairs (unchanged)",
        "shots_per_request": "1 shot of each of the two circuits (2 per request), as in the previous evaluation",
        "outcomes": "recorded outcomes carried as metadata for replay; nothing is newly measured",
    }
    workload = build_predetermined_workload(ev.requests, ev.schedule, registry, config)
    path = out / "frozen_workload.json"
    fingerprint = save_workload(workload, path)
    workload = load_workload(path)  # every policy below runs from the reloaded, fingerprint-verified artifact
    assert workload.fingerprint == fingerprint
    (out / "workload_fingerprint.txt").write_text(f"{fingerprint}\n", encoding="utf-8")
    (out / "workload_generation_config.json").write_text(json.dumps(to_jsonable({
        **dict(workload.config), "request_count": workload.request_count, "eligible_pair_count": workload.eligible_pair_count,
        "total_shot_budget": workload.total_shot_budget, "fingerprint_sha256": fingerprint,
        "device": {"start_time": workload.device.start_time, "slot_seconds": workload.device.slot_seconds, "total_slots": workload.device.total_slots},
    }), indent=1, allow_nan=False), encoding="utf-8")
    say(f"froze {workload.request_count} requests, {workload.eligible_pair_count} eligible pairs, {workload.total_shot_budget} shots; sha256 {fingerprint[:16]}")

    # ---- 2. drift model (training only) and the common comparison reference
    say("fitting the drift model on the training partition (as in the previous evaluation)")
    train_obs = observations_from_shots(ev.split.train)
    stage = fit_training_stage(train_obs, FullEvaluationConfig(), say)
    model, origin = stage.model, ev.split.min_timestamp
    drift_fp = drift_fingerprint(model, stage.basis)
    references = tuple(r.planned_time for r in workload.requests)

    # ---- 3. one simulator input, the same objects for all three policies
    sim = to_simulator_input(workload)
    policy_objects = {"arrival-order": ArrivalOrderPolicy(), "consecutive-pair": ConsecutivePairPolicy(),
                      "drift-aware": DriftAwarePairPolicy(ev.priorities)}
    inputs = {n: make_policy_input(n, sim, registry, drift_fp, references) for n in POLICIES}
    results = {n: run_simulation(sim.shots, registry, p, schedule=sim.schedule, requests=sim.requests) for n, p in policy_objects.items()}

    # ---- 4. fairness audit: failure invalidates the evaluation
    audit = audit_fairness(workload, inputs, results, drift_fp, references)
    (out / "fairness_audit.json").write_text(json.dumps(audit.to_dict(), indent=1), encoding="utf-8")
    say(f"fairness audit: {sum(c.passed for c in audit.checks)}/{len(audit.checks)} checks passed")
    if not audit.passed:
        (out / "INVALID_fairness_audit_failed.json").write_text(json.dumps(audit.to_dict(), indent=1), encoding="utf-8")
        try:
            audit.require()
        except FairnessAuditError as exc:
            print(f"EVALUATION INVALID: {exc}", file=sys.stderr)
        return 2

    # ---- 5. scoring and outputs
    rows = {n: trace_rows(workload, results[n], model, origin, WINDOW_SECONDS, RELIABILITY_THRESHOLD, references) for n in POLICIES}
    summaries = {n: policy_summary(workload, results[n], rows[n], RELIABILITY_THRESHOLD) for n in POLICIES}
    for n in POLICIES:  # waiting over the shots of completed comparisons: the definition of the stored evaluation
        summaries[n]["shot_wait_paired_mean_s"] = float(np.mean([w for c in results[n].comparisons for w in (
            c.execution_time_a - c.arrival_time_a, c.execution_time_b - c.arrival_time_b)]))
        write_csv(out / f"execution_trace_{n.replace('-', '_')}.csv", rows[n])
    write_csv(out / "policy_metrics.csv", [summaries[n] for n in POLICIES])

    stored = json.load(open(STORED, encoding="utf-8"))["primary"]["policies"] if STORED.exists() else None
    comparison = []
    if stored:
        pairs = (("completed_comparisons", "completed_comparisons", "completed_comparisons"), ("separation_mean_s", "separation_mean", "separation_mean"),
                 ("shot_wait_paired_mean_s", "waiting_mean", "waiting_mean"), ("idle_slots", "idle_slots", "idle_slots"),
                 ("model_error_mean", "error_mean", "error_mean"), ("model_error_max", "error_max", "error_max"),
                 ("reliable_per_device_hour_model", "reliable_throughput_per_device_hour", "reliable_throughput_per_device_hour"))
        for n in POLICIES:
            for new_key, old_key, label in pairs:
                comparison.append({"policy": n, "metric": label, "previous_evaluation": stored[n][old_key]["estimate"] if isinstance(stored[n][old_key], dict) else stored[n][old_key],
                                   "predetermined_workload": summaries[n][new_key]})
        write_csv(out / "comparison_with_previous_evaluation.csv", comparison)

    shift = np.array([r.planned_time - r.release_time for r in workload.requests])
    summary = {
        "evaluation_type": "recorded-trace timing replay of one frozen predetermined workload; model-derived comparison errors; not a real-QPU experiment",
        "provenance": {"queue_and_timing": "production simulator replay of the frozen workload",
                       "comparison_error": "drift-model-derived (training-fitted model that failed forecasting validation) against the planned measurement time; no outcome is simulated or newly measured",
                       "outcomes": "recorded outcomes replayed as observed"},
        "workload": {"fingerprint_sha256": fingerprint, "requests": workload.request_count, "eligible_pairs": workload.eligible_pair_count,
                     "total_shot_budget": workload.total_shot_budget, "construction": workload.config["construction_method"],
                     "device": {"slot_seconds": workload.device.slot_seconds, "total_slots": workload.device.total_slots},
                     "planned_minus_release_seconds": {"mean": float(shift.mean()), "median": float(np.median(shift)), "max": float(shift.max()), "min": float(shift.min())}},
        "drift_input_fingerprint": drift_fp, "reliability_threshold": RELIABILITY_THRESHOLD,
        "fairness_audit": {"passed": audit.passed, "checks": len(audit.checks)},
        "policies": summaries, "comparison_with_previous_evaluation": comparison,
    }
    (out / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=1, allow_nan=False), encoding="utf-8")
    say(f"wrote {out}")
    for n in POLICIES:
        s = summaries[n]
        print(f"  {n:17s} completed {s['completed_comparisons']}  separation {s['separation_mean_s']:.4f} s  wait(release->exec) {s['shot_wait_from_release_mean_s']:.3f} s "
              f"(p95 {s['shot_wait_from_release_p95_s']:.3f}, max {s['shot_wait_from_release_max_s']:.2f})  idle {s['idle_slots']}  "
              f"E_AB mean {s['model_error_mean']:.5f} max {s['model_error_max']:.5f}  reliable/h {s['reliable_per_device_hour_model']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
