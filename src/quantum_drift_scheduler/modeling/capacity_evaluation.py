"""Per-scenario policy metrics for the capacity-sensitivity evaluation.

Wraps ``predetermined_evaluation.policy_summary`` (all existing metric definitions are unchanged) and adds the
quantities that depend on the service duration: separation in slot units, device-time denominators, idle time
and queue length. Provenance is as in ``predetermined_evaluation``: queue and timing quantities come from the
simulator; ``E_AB`` is a drift-model-derived error (the model has not shown deployment-faithful forecasting
skill) and is not measured reliability.

Device-time denominators (both reported, never mixed):

* ``budget`` device time = ``total_slots * service_seconds`` (the production primary denominator; idle slots included);
  it differs between scenarios by less than one slot because the wall-clock horizon is fixed;
* ``busy`` device time = ``executed shots * service_seconds`` (device time actually used).
"""

from typing import Any, Dict, Sequence

import numpy as np

from quantum_drift_scheduler.modeling.predetermined_evaluation import policy_summary
from quantum_drift_scheduler.scheduling.capacity_scenarios import CapacityScenario, queue_length_stats
from quantum_drift_scheduler.scheduling.models import SimulationResult
from quantum_drift_scheduler.scheduling.predetermined_workload import PredeterminedWorkload


def scenario_policy_metrics(
    workload: PredeterminedWorkload,
    scenario: CapacityScenario,
    result: SimulationResult,
    rows: Sequence[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """Metrics of one policy on one scenario's grid; ``workload`` must carry that scenario's device."""
    s = policy_summary(workload, result, rows, threshold)
    dev = workload.device
    d = dev.slot_seconds
    done = [r for r in rows if r["completed"]]
    sep = np.array([r["separation_seconds"] for r in done]) if done else np.array([])
    starts = [min(r["actual_execution_first"], r["actual_execution_second"]) if r["completed"] else None for r in rows]
    slot_times = dev.start_time + np.arange(dev.total_slots) * d
    q = queue_length_stats([r.release_time for r in workload.requests], starts, slot_times)
    busy_slots = s["executed_shots"]
    busy_seconds = busy_slots * d
    reliable = s["reliable_comparisons_model"]
    out = {
        "scenario": scenario.name, "service_seconds": d, "service_rate_shots_per_second": 1.0 / d,
        "total_slots": dev.total_slots, "budget_device_seconds": dev.total_slots * d,
        "busy_slots": busy_slots, "busy_device_seconds": busy_seconds, "idle_device_seconds": s["idle_slots"] * d,
        "separation_mean_slots": float(sep.mean() / d) if sep.size else None,
        "separation_median_slots": float(np.median(sep) / d) if sep.size else None,
        "separation_p95_slots": float(np.percentile(sep, 95) / d) if sep.size else None,
        "reliable_per_busy_device_hour_model": reliable / (busy_seconds / 3600.0) if busy_seconds > 0 else None,
        "unfinished_requests": s["incomplete_requests"], **q,
    }
    return {**s, **out}
