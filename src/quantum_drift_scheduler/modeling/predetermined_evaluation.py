"""Scoring of the three policies' execution timelines on the frozen predetermined workload.

Provenance (kept explicit in every output):

* queue and timing quantities (release, execution, waiting, separation, idle slots, completions) come
  from replaying the frozen workload through the unchanged production simulator;
* ``Delta_measured``, ``Delta_reference`` and ``E_AB`` are **model-derived** from the training-fitted drift
  model evaluated at the policy's actual execution times and at the workload's planned measurement time.
  Recorded outcomes are replayed as observed; no outcome is simulated, nothing is newly measured, and the
  drift model failed forecasting validation, so these errors are not evidence of real-device accuracy.

    Delta_measured  = p_A(t_A) - p_B(t_B)       t_A, t_B: the policy's actual execution times
    Delta_reference = p_A(t_plan) - p_B(t_plan) t_plan: the request's frozen planned measurement time
    E_AB            = |Delta_measured - Delta_reference|
    reliable        = completed and E_AB <= threshold (the existing fixed threshold)
"""

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.modeling.comparison_metrics import classify_reliability, compute_comparison_errors
from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel
from quantum_drift_scheduler.scheduling.metrics import compute_metrics, percentile
from quantum_drift_scheduler.scheduling.models import SimulationResult
from quantum_drift_scheduler.scheduling.predetermined_workload import PredeterminedWorkload


def drift_fingerprint(model: RamseyModel, basis: Optional[Any] = None) -> str:
    """SHA-256 of the fitted drift-model parameters (and basis span): the 'drift input' of an evaluation."""
    p = model.parameters
    content = {"offset": p.offset, "contrast": p.contrast, "l0": p.l0, "t_w": p.t_w,
               "gammas": [float(g) for g in p.gammas], "n_components": p.n_components}
    if basis is not None:
        content["basis"] = {"t0": basis.t0, "t_step": basis.t_step, "N": basis.n_steps}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def trace_rows(
    workload: PredeterminedWorkload,
    result: SimulationResult,
    model: RamseyModel,
    window_origin: float,
    window_seconds: float,
    threshold: float,
    reference_times: Optional[Sequence[float]] = None,
) -> List[Dict[str, Any]]:
    """One row per workload request: the three timestamps, waits, separation and the model-derived error."""
    refs = np.array([r.planned_time for r in workload.requests] if reference_times is None else reference_times, dtype=float)
    by_id = {c.request_id: c for c in result.comparisons}
    done = np.array([r.request_id in by_id for r in workload.requests])
    n = len(workload.requests)
    error = np.full(n, np.nan)
    measured, reference = np.full(n, np.nan), np.full(n, np.nan)
    if done.any():
        idx = np.where(done)[0]
        errs = compute_comparison_errors(
            model,
            np.array([workload.requests[i].wait_length_first for i in idx], dtype=float),
            np.array([by_id[workload.requests[i].request_id].execution_time_a for i in idx]),
            np.array([workload.requests[i].wait_length_second for i in idx], dtype=float),
            np.array([by_id[workload.requests[i].request_id].execution_time_b for i in idx]),
            window_origin, window_seconds, reference_times=refs[idx],
        )
        error[idx], measured[idx], reference[idx] = errs.error, errs.measured_delta, errs.reference_delta
    reliable = classify_reliability(error, threshold)
    end = workload.device.start_time + (workload.device.total_slots - 1) * workload.device.slot_seconds
    rows = []
    for i, r in enumerate(workload.requests):
        c = by_id.get(r.request_id)
        row = {
            "policy": result.policy_name, "request_id": r.request_id, "round_id": r.round_id,
            "circuit_first": r.circuit_first, "circuit_second": r.circuit_second,
            "release_time": r.release_time, "planned_time": r.planned_time, "completed": c is not None,
            "slot_first": c.slot_a if c else None, "slot_second": c.slot_b if c else None,
            "actual_execution_first": c.execution_time_a if c else None,
            "actual_execution_second": c.execution_time_b if c else None,
            "wait_release_to_first_execution": (c.execution_time_a - r.release_time) if c else None,
            "wait_release_to_second_execution": (c.execution_time_b - r.release_time) if c else None,
            "wait_so_far_if_unexecuted": None if c else max(end, r.release_time) - r.release_time,
            "actual_first_minus_planned": (c.execution_time_a - r.planned_time) if c else None,
            "separation_seconds": abs(c.execution_time_b - c.execution_time_a) if c else None,
            "delta_measured_model": None if not c else float(measured[i]),
            "delta_reference_model": None if not c else float(reference[i]),
            "error_E_AB_model": None if not c else float(error[i]),
            "reliable_model": bool(reliable[i]),
        }
        rows.append(row)
    return rows


def policy_summary(
    workload: PredeterminedWorkload, result: SimulationResult, rows: Sequence[Dict[str, Any]], threshold: float
) -> Dict[str, Any]:
    """Queue metrics (production definitions, waiting measured from the release time) and model-derived metrics."""
    m = compute_metrics(result)
    dev = workload.device
    budget_hours = dev.total_slots * dev.slot_seconds / 3600.0
    done = [r for r in rows if r["completed"]]
    err = np.array([r["error_E_AB_model"] for r in done]) if done else np.array([])
    qw = [r["wait_release_to_first_execution"] for r in done]
    allw = [r["wait_release_to_first_execution"] if r["completed"] else r["wait_so_far_if_unexecuted"] for r in rows]
    reliable = int(sum(r["reliable_model"] for r in rows))
    return {
        "policy": result.policy_name, "requests": len(rows), "total_shot_budget": workload.total_shot_budget,
        "completed_comparisons": m.completed_comparisons, "completion_rate": m.completion_rate,
        "incomplete_requests": len(rows) - m.completed_comparisons, "executed_shots": m.executed_shots,
        "unexecuted_shots": m.unexecuted_shots, "idle_slots": m.idle_slots, "total_slots": m.total_slots,
        "device_utilisation_pct": 100.0 * m.executed_shots / m.total_slots,
        "separation_mean_s": m.time_separation.mean, "separation_median_s": m.time_separation.median,
        "separation_p95_s": m.time_separation.p95, "separation_max_s": m.time_separation.maximum,
        "shot_wait_from_release_mean_s": m.waiting_time.mean, "shot_wait_from_release_median_s": m.waiting_time.median,
        "shot_wait_from_release_p95_s": m.waiting_time.p95, "shot_wait_from_release_max_s": m.waiting_time.maximum,
        "request_wait_mean_s": float(np.mean(qw)) if qw else None, "request_wait_p95_s": percentile(qw, 95.0) if qw else None,
        "request_wait_max_s": float(max(qw)) if qw else None,
        "request_wait_incl_unexecuted_max_s": float(max(allw)) if allw else None,
        "model_error_mean": float(err.mean()) if err.size else None,
        "model_error_p95": float(np.percentile(err, 95)) if err.size else None,
        "model_error_max": float(err.max()) if err.size else None,
        "reliable_comparisons_model": reliable, "reliability_threshold": threshold,
        "reliable_per_device_hour_model": reliable / budget_hours,
        "budget_device_hours": budget_hours,
    }
