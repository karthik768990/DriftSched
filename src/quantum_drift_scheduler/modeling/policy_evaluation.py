"""Matched evaluation of the three scheduling policies under a drift model.

The frozen scheduler (``scheduling``/``evaluation``) supplies the execution timeline of
``arrival-order``, ``consecutive-pair`` and ``drift-aware`` on identical holdout shots,
slot budget and prebuilt request workload. This module only *scores* those timelines:

* ``Delta_measured`` / ``Delta_reference`` / ``E_AB`` come from the supplied drift model
  and the actual execution timestamps (``comparison_metrics``); they are model-derived,
  not observed measurements, and **no outcomes are simulated**.
* Only pairs whose registry ``reviewer_decision`` is ``Comparable`` are scheduled and scored.
* A comparison is *reliable* iff it completed and ``E_AB <= reliability_threshold``; the
  threshold is a fixed configuration value (see ``config.RELIABILITY_THRESHOLD``).
* Primary outcome: reliable comparisons per device-hour of the (identical) slot budget.
* Intervals use a paired temporal block bootstrap over the time-ordered workload
  requests, so policy differences share every resample.

Evaluation type: recorded-timing replay scored by a model-based reference. It is not a
real-QPU experiment.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.evaluation import HoldoutEvaluation
from quantum_drift_scheduler.modeling.comparison_metrics import (
    classify_reliability,
    compute_comparison_errors,
)
from quantum_drift_scheduler.modeling.observations import ObservationError
from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel
from quantum_drift_scheduler.modeling.uncertainty import (
    ConfidenceInterval,
    bootstrap_statistics,
    choose_block_length,
)
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, wait_length
from quantum_drift_scheduler.scheduling import (
    ConsecutivePairPolicy,
    PairRequest,
    ShotArrival,
    prepare_scheduler_input,
    run_simulation,
)

POLICY_ORDER: Tuple[str, ...] = ("arrival-order", "consecutive-pair", "drift-aware")
METRICS: Tuple[str, ...] = (
    "mae",
    "synchronous_mae",
    "completed_comparisons",
    "completion_rate",
    "mean_waiting_time",
    "reliable_comparisons",
    "reliable_per_device_hour",
)


class PolicyEvaluationError(ValueError):
    """Raised for inconsistent policy results."""


@dataclass(frozen=True)
class PolicyArrays:
    """Per-request arrays for one policy; NaN/False/-1 mark incomplete comparisons."""

    name: str
    request_ids: Tuple[str, ...]
    completed: np.ndarray
    comparison_id: np.ndarray
    circuit_a: Tuple[str, ...]
    circuit_b: Tuple[str, ...]
    arrival_a: np.ndarray
    arrival_b: np.ndarray
    execution_a: np.ndarray
    execution_b: np.ndarray
    slot_a: np.ndarray
    slot_b: np.ndarray
    waiting_a: np.ndarray
    waiting_b: np.ndarray
    measured_delta: np.ndarray
    reference_delta: np.ndarray
    error: np.ndarray
    synchronous_error: np.ndarray
    window_index: np.ndarray
    reliable: np.ndarray
    budget_seconds: float
    busy_seconds: float
    slot_seconds: float
    executed_shots: int
    unexecuted_shots: int
    idle_slots: int

    @property
    def n_requests(self) -> int:
        return len(self.request_ids)

    @property
    def pair_waiting(self) -> np.ndarray:
        return (self.waiting_a + self.waiting_b) / 2.0


def build_policy_arrays(
    evaluation: HoldoutEvaluation,
    requests: Sequence[PairRequest],
    model: RamseyModel,
    window_origin: float,
    window_seconds: float,
    reliability_threshold: float,
    reference_times: Optional[np.ndarray] = None,
) -> Dict[str, PolicyArrays]:
    """Score every policy's holdout timeline against ``model`` on the common workload.

    ``reference_times`` (one per request, aligned with ``requests``) fixes each request's common
    reference timestamp independently of the policy; see ``compute_comparison_errors``.
    """
    ids = tuple(r.request_id for r in requests)
    if len(set(ids)) != len(ids):
        raise PolicyEvaluationError("Duplicate request ids in the workload")
    position = {rid: i for i, rid in enumerate(ids)}
    n = len(requests)
    if reference_times is not None and len(reference_times) != n:
        raise PolicyEvaluationError("reference_times must have one entry per request")
    k_a = np.array([r.wait_length_first for r in requests], dtype=float)
    k_b = np.array([r.wait_length_second for r in requests], dtype=float)

    out: Dict[str, PolicyArrays] = {}
    for run in evaluation.runs:
        result = run.result
        completed = np.zeros(n, dtype=bool)
        comparison_id = np.full(n, -1, dtype=np.int64)
        nan = lambda: np.full(n, np.nan)
        arrays = {k: nan() for k in ("arrival_a", "arrival_b", "execution_a", "execution_b",
                                      "slot_a", "slot_b")}
        for c in result.comparisons:
            i = position.get(c.request_id)
            if i is None:
                raise PolicyEvaluationError(f"Comparison references unknown request {c.request_id!r}")
            if completed[i]:
                raise PolicyEvaluationError(f"Request {c.request_id!r} completed twice")
            if (c.circuit_a, c.circuit_b) != (requests[i].first.circuit, requests[i].second.circuit):
                raise PolicyEvaluationError(f"Comparison {c.request_id!r} circuits do not match the workload")
            completed[i] = True
            comparison_id[i] = c.comparison_id
            arrays["arrival_a"][i], arrays["arrival_b"][i] = c.arrival_time_a, c.arrival_time_b
            arrays["execution_a"][i], arrays["execution_b"][i] = c.execution_time_a, c.execution_time_b
            arrays["slot_a"][i], arrays["slot_b"][i] = c.slot_a, c.slot_b

        measured, reference, error, sync, widx = nan(), nan(), nan(), nan(), nan()
        if completed.any():
            errs = compute_comparison_errors(
                model, k_a[completed], arrays["execution_a"][completed],
                k_b[completed], arrays["execution_b"][completed], window_origin, window_seconds,
                reference_times=None if reference_times is None else np.asarray(reference_times)[completed],
            )
            measured[completed], reference[completed] = errs.measured_delta, errs.reference_delta
            error[completed], sync[completed] = errs.error, errs.synchronous_error
            widx[completed] = errs.window_index
        schedule = result.schedule
        out[result.policy_name] = PolicyArrays(
            name=result.policy_name,
            request_ids=ids,
            completed=completed,
            comparison_id=comparison_id,
            circuit_a=tuple(r.first.circuit for r in requests),
            circuit_b=tuple(r.second.circuit for r in requests),
            arrival_a=arrays["arrival_a"], arrival_b=arrays["arrival_b"],
            execution_a=arrays["execution_a"], execution_b=arrays["execution_b"],
            slot_a=arrays["slot_a"], slot_b=arrays["slot_b"],
            waiting_a=arrays["execution_a"] - arrays["arrival_a"],
            waiting_b=arrays["execution_b"] - arrays["arrival_b"],
            measured_delta=measured, reference_delta=reference, error=error,
            synchronous_error=sync, window_index=widx,
            reliable=classify_reliability(error, reliability_threshold),
            budget_seconds=schedule.total_slots * schedule.slot_seconds,
            busy_seconds=len(result.executed) * schedule.slot_seconds,
            slot_seconds=schedule.slot_seconds,
            executed_shots=len(result.executed),
            unexecuted_shots=len(result.unexecuted_shots),
            idle_slots=result.idle_slots,
        )
    if set(out) != set(POLICY_ORDER):
        raise PolicyEvaluationError(f"Expected policies {POLICY_ORDER}, got {tuple(out)}")
    budgets = {round(a.budget_seconds, 9) for a in out.values()}
    if len(budgets) != 1:
        raise PolicyEvaluationError("Policies do not share an identical device-time budget")
    return out


def policy_metrics(arrays: PolicyArrays, indices: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Metrics of one policy over a (possibly resampled) set of workload positions."""
    idx = np.arange(arrays.n_requests) if indices is None else indices
    done = arrays.completed[idx]
    err = arrays.error[idx]
    sync = arrays.synchronous_error[idx]
    wait = arrays.pair_waiting[idx]
    n = idx.size
    reliable_fraction = float(arrays.reliable[idx].mean())
    scale = arrays.n_requests / n
    return {
        "mae": float(np.nanmean(err)) if done.any() else math.nan,
        "synchronous_mae": float(np.nanmean(sync)) if done.any() else math.nan,
        "completed_comparisons": float(done.sum() * scale),
        "completion_rate": float(done.mean()),
        "mean_waiting_time": float(np.nanmean(wait)) if done.any() else math.nan,
        "reliable_comparisons": float(arrays.reliable[idx].sum() * scale),
        "reliable_per_device_hour": reliable_fraction * arrays.n_requests / (arrays.budget_seconds / 3600.0),
    }


@dataclass(frozen=True)
class PolicyEvaluation:
    """Scored policies with point estimates and bootstrap intervals."""

    reference_label: str
    arrays: Dict[str, PolicyArrays]
    intervals: Dict[str, Dict[str, ConfidenceInterval]]
    differences: Dict[str, Dict[str, ConfidenceInterval]]
    block_length: int
    n_requests: int
    reliability_threshold: float
    window_seconds: float


DIFFERENCES: Tuple[Tuple[str, str], ...] = (
    ("consecutive-pair", "arrival-order"),
    ("drift-aware", "arrival-order"),
    ("drift-aware", "consecutive-pair"),
)


def evaluate_policies(
    arrays: Dict[str, PolicyArrays],
    block_length: int,
    reference_label: str,
    reliability_threshold: float,
    window_seconds: float,
    n_bootstrap: int,
    seed: int,
) -> PolicyEvaluation:
    """Paired temporal block bootstrap of every policy metric and policy difference."""
    n = next(iter(arrays.values())).n_requests

    def statistic(idx: np.ndarray) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        per_policy = {name: policy_metrics(arrays[name], idx) for name in POLICY_ORDER}
        for name, m in per_policy.items():
            for key, value in m.items():
                stats[f"{name}|{key}"] = value
        for a, b in DIFFERENCES:
            for key in METRICS:
                stats[f"{a}-{b}|{key}"] = per_policy[a][key] - per_policy[b][key]
        return stats

    cis = bootstrap_statistics(n, statistic, block_length, n_bootstrap, seed)
    intervals = {name: {k: cis[f"{name}|{k}"] for k in METRICS} for name in POLICY_ORDER}
    differences = {f"{a} - {b}": {k: cis[f"{a}-{b}|{k}"] for k in METRICS} for a, b in DIFFERENCES}
    return PolicyEvaluation(
        reference_label=reference_label,
        arrays=arrays,
        intervals=intervals,
        differences=differences,
        block_length=block_length,
        n_requests=n,
        reliability_threshold=reliability_threshold,
        window_seconds=window_seconds,
    )


def training_block_length(
    train_trace: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    model: RamseyModel,
    window_origin: float,
    window_seconds: float,
    n_holdout_requests: int,
) -> int:
    """Block length from the **training** period only.

    Replays the ``consecutive-pair`` policy on the training shots, scores it with the
    training-fitted ``model``, standardises each pair's errors, and applies
    ``choose_block_length`` (integrated autocorrelation time, whole workload cycles).
    """
    prepared = prepare_scheduler_input(train_trace, registry)
    requests = prepared.requests
    if not requests:
        raise ObservationError("Training trace produced no comparison requests")
    result = run_simulation(prepared.source_shots, registry, ConsecutivePairPolicy(), requests=requests)
    by_request = {c.request_id: c for c in result.comparisons}
    ordered = [by_request[r.request_id] for r in requests if r.request_id in by_request]
    if len(ordered) < 20:
        raise ObservationError("Too few completed training comparisons to choose a block length")
    errs = compute_comparison_errors(
        model,
        np.array([wait_length(c.circuit_a) for c in ordered], dtype=float),
        np.array([c.execution_time_a for c in ordered]),
        np.array([wait_length(c.circuit_b) for c in ordered], dtype=float),
        np.array([c.execution_time_b for c in ordered]),
        window_origin,
        window_seconds,
    ).error
    pairs = np.array([f"{c.circuit_a}|{c.circuit_b}" for c in ordered])
    series = np.empty_like(errs)
    for key in np.unique(pairs):
        m = pairs == key
        sd = errs[m].std()
        series[m] = (errs[m] - errs[m].mean()) / (sd if sd > 0 else 1.0)
    return choose_block_length(series, cycle_length=len(registry.eligible_pairs()), n_items=n_holdout_requests)
