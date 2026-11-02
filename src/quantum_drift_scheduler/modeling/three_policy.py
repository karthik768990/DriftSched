"""Three-policy QCS evaluation: arrival-order vs consecutive-pair vs drift-aware.

Policies (the frozen scheduler, reused unchanged):

1. ``arrival-order``     individual-shot FIFO, the baseline without QCS;
2. ``consecutive-pair``  the two shots of a comparison run back to back (isolates pair-aware execution);
3. ``drift-aware``       pair-aware plus the priority mapping recalculated for this run.

Fair conditions: all three run on the identical eligible workload, source shots, arrival
pattern, slot budget/capacity, split and (for drift-aware) one frozen priority mapping; they
are scored against the same simulated drift and the same reference values. Only the policy
differs. Eligibility comes only from the registry's ``reviewer_decision``.

Provenance (kept explicit in every output):

* **recorded-trace timing replay**: arrival pattern, queueing, waiting, measurement separation,
  idle slots and completions come from replaying the recorded holdout arrival times through the
  scheduler;
* **model-simulated**: ``p_A``, ``p_B``, ``Delta_measured``, ``Delta_reference`` and the comparison
  errors come from a drift model fitted on the *training* portion only and evaluated at those
  times. No outcomes are simulated and nothing here is a device observation;
* **synthetic stress** scenarios (drift-rate factors other than 1) are labelled as such and are
  never the primary result.

Definitions::

    Delta_measured  = p_A(t_A) - p_B(t_B)                 t_A, t_B: the policy's execution times
    Delta_reference = p_A(t_ref) - p_B(t_ref)             t_ref: midpoint of the fixed 600 s window
                                                          that contains the request's arrival
                                                          (identical for every policy)
    E_AB            = |Delta_measured - Delta_reference|
    reliable        = completed and E_AB <= 0.05          (threshold fixed in advance)
    throughput      = reliable completed comparisons / device time
                      primary denominator: the budgeted device time total_slots * slot_seconds, identical
                      for all policies (idle slots included); the busy-time variant is also reported.
"""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quantum_drift_scheduler.analysis.holdout import split_trace, validate_split_parameters
from quantum_drift_scheduler.analysis.priorities import PriorityError, PriorityTable, priority_table_to_dict
from quantum_drift_scheduler.config import DEFAULT_CONFIG
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, load_compatibility_registry, wait_length
from quantum_drift_scheduler.evaluation import HoldoutEvaluation, run_holdout_evaluation
from quantum_drift_scheduler.modeling.comparison_metrics import compute_comparison_errors, reference_time
from quantum_drift_scheduler.modeling.config import RELIABILITY_THRESHOLD, FullEvaluationConfig
from quantum_drift_scheduler.modeling.model_validation import ModelEvaluation, constant_baseline, evaluate_model
from quantum_drift_scheduler.modeling.observations import data_report, observations_from_shots
from quantum_drift_scheduler.modeling.pipeline import TrainingStage, fit_training_stage
from quantum_drift_scheduler.modeling.policy_evaluation import (
    POLICY_ORDER,
    PolicyArrays,
    build_policy_arrays,
    training_block_length,
)
from quantum_drift_scheduler.modeling.ramsey_model import RamseyModel
from quantum_drift_scheduler.modeling.reporting import environment_info, to_jsonable
from quantum_drift_scheduler.modeling.uncertainty import bootstrap_statistics
from quantum_drift_scheduler.scheduling import ShotArrival, load_recorded_trace

DEFAULT_OUTPUT_DIR = DEFAULT_CONFIG.base_dir / "results" / "qcs_three_policy_evaluation"
DEFAULT_PRIORITY_ANALYSIS_CSV = DEFAULT_CONFIG.base_dir / "results" / "priority_analysis" / "priority_results.csv"
DEFAULT_STRESS_FACTORS: Tuple[float, ...] = (10.0, 100.0)

OUTPUT_FILES: Tuple[str, ...] = (
    "results.json", "policy_comparisons.csv", "comparison_errors.csv", "queue_metrics.csv",
    "reliability_metrics.csv", "fig1_comparison_error.png", "fig2_measurement_separation.png",
    "fig3_waiting_time.png", "fig4_reliable_throughput.png", "fig5_policy_comparison.png",
)

PROVENANCE_REPLAY = "recorded-trace timing replay"
PROVENANCE_MODEL = "model-simulated (frozen training-fitted drift model; no outcomes simulated)"

RESAMPLED_METRICS: Tuple[str, ...] = (
    "error_mean", "error_median", "sync_error_mean", "sync_error_median", "separation_mean",
    "separation_median", "waiting_mean", "waiting_median", "completed_comparisons", "completion_rate",
    "reliable_comparisons", "reliable_throughput_per_device_hour", "reliable_throughput_per_device_second",
    "reliable_throughput_per_busy_hour",
)
EXTREME_METRICS: Tuple[str, ...] = (
    "error_p95", "error_max", "sync_error_p95", "sync_error_max", "separation_p95", "separation_max",
    "waiting_p95", "waiting_max",
)
DIFFERENCE_METRICS: Tuple[str, ...] = (
    "error_mean", "error_median", "sync_error_mean", "separation_mean", "waiting_mean",
    "completed_comparisons", "reliable_throughput_per_device_hour",
)
DIFFERENCES: Tuple[Tuple[str, str], ...] = (
    ("consecutive-pair", "arrival-order"),
    ("drift-aware", "arrival-order"),
    ("drift-aware", "consecutive-pair"),
)
UNITS: Dict[str, str] = {
    "error": "probability", "sync_error": "probability", "separation": "s", "waiting": "s",
    "completed_comparisons": "comparisons", "completion_rate": "fraction", "reliable_comparisons": "comparisons",
    "reliable_throughput_per_device_hour": "reliable comparisons / device-hour",
    "reliable_throughput_per_device_second": "reliable comparisons / device-second",
    "reliable_throughput_per_busy_hour": "reliable comparisons / busy device-hour",
}
QUEUE_PREFIXES = ("waiting_", "separation_", "completed_", "completion_", "idle_", "executed_", "unexecuted_",
                   "total_slots", "budget_", "busy_")


class ThreePolicyError(ValueError):
    """Raised for invalid configuration, an unsafe output directory or an inconsistent experiment."""


@dataclass(frozen=True)
class ThreePolicyConfig:
    """All settings of the three-policy evaluation. The reliability threshold is fixed, not an option."""

    dataset_path: Path = DEFAULT_CONFIG.default_dataset_path
    registry_path: Path = DEFAULT_CONFIG.compatibility_registry_path
    output_dir: Path = DEFAULT_OUTPUT_DIR
    priority_analysis_csv: Path = DEFAULT_PRIORITY_ANALYSIS_CSV
    train_fraction: float = 0.75
    window_seconds: float = 600.0
    slots_per_shot: float = 1.0
    seed: int = 20260101
    n_bootstrap: int = 2000
    confidence: float = 0.95
    n_starts: int = 6
    cv_blocks: int = 5
    candidates: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13)
    stress_drift_rate_factors: Tuple[float, ...] = DEFAULT_STRESS_FACTORS
    overwrite: bool = False
    make_figures: bool = True

    @property
    def reliability_threshold(self) -> float:
        return RELIABILITY_THRESHOLD

    def to_full_config(self) -> FullEvaluationConfig:
        return FullEvaluationConfig(
            train_fraction=self.train_fraction, window_seconds=self.window_seconds, candidates=self.candidates,
            cv_blocks=self.cv_blocks, n_starts=self.n_starts, seed=self.seed, slots_per_shot=self.slots_per_shot,
            n_bootstrap=self.n_bootstrap, confidence=self.confidence, cv_sensitivity_blocks=(),
            output_dir=self.output_dir, make_figures=self.make_figures,
        )


@dataclass(frozen=True)
class TimeScaledModel:
    """Controlled-drift stress model: the base model's drift runs ``rate_factor`` times faster.

    ``p(l, t) = base.p(l, origin + rate_factor * (t - origin))``. A factor of 1 is the base model. Used only for
    labelled synthetic stress scenarios.
    """

    base: RamseyModel
    rate_factor: float
    origin: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.rate_factor) and self.rate_factor > 0.0):
            raise ThreePolicyError(f"rate_factor must be positive and finite, got {self.rate_factor!r}")

    def probability(self, wait_lengths: np.ndarray, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        return self.base.probability(wait_lengths, self.origin + self.rate_factor * (t - self.origin))


@dataclass(frozen=True)
class MetricValue:
    """A metric estimate, with a 95% temporal-block-bootstrap interval when it is resampled."""

    estimate: Optional[float]
    lower: Optional[float]
    upper: Optional[float]
    unit: str
    interval: bool


@dataclass(frozen=True)
class ScenarioEvaluation:
    label: str
    rate_factor: float
    kind: str  # "primary" or "synthetic-stress"
    arrays: Dict[str, PolicyArrays]
    metrics: Dict[str, Dict[str, MetricValue]]
    differences: Dict[str, Dict[str, MetricValue]]


@dataclass(frozen=True)
class AuditCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ThreePolicyResult:
    config: ThreePolicyConfig
    data: Dict[str, Any]
    scheduler: HoldoutEvaluation
    training: TrainingStage
    model_diagnostic: ModelEvaluation
    model_baseline: Tuple[float, float]
    reference_times: np.ndarray
    block_length: int
    scenarios: Tuple[ScenarioEvaluation, ...]
    fairness: Tuple[AuditCheck, ...]
    priority_consistency: Dict[str, Any]
    hashes: Dict[str, str]

    @property
    def primary(self) -> ScenarioEvaluation:
        return self.scenarios[0]

    @property
    def audits_passed(self) -> bool:
        return all(c.passed for c in self.fairness)


# --------------------------------------------------------------------------- metrics


def _stat(values: np.ndarray, fn: Callable[[np.ndarray], float]) -> float:
    return float(fn(values)) if values.size else math.nan


def _unit(metric: str) -> str:
    if metric in UNITS:
        return UNITS[metric]
    for stem in ("sync_error", "error", "separation", "waiting"):
        if metric.startswith(stem):
            return UNITS.get(stem, "")
    return ""


def scalar_metrics(a: PolicyArrays, idx: np.ndarray) -> Dict[str, float]:
    """Resampleable metrics of one policy over workload positions ``idx`` (NaN if undefined)."""
    done = a.completed[idx]
    n = idx.size
    scale = a.n_requests / n
    err = a.error[idx][done]
    sync = a.synchronous_error[idx][done]
    sep = np.abs(a.execution_b[idx][done] - a.execution_a[idx][done])
    wait = np.concatenate((a.waiting_a[idx][done], a.waiting_b[idx][done]))
    reliable = float(a.reliable[idx].sum()) * scale
    return {
        "error_mean": _stat(err, np.mean), "error_median": _stat(err, np.median),
        "sync_error_mean": _stat(sync, np.mean), "sync_error_median": _stat(sync, np.median),
        "separation_mean": _stat(sep, np.mean), "separation_median": _stat(sep, np.median),
        "waiting_mean": _stat(wait, np.mean), "waiting_median": _stat(wait, np.median),
        "completed_comparisons": float(done.sum()) * scale,
        "completion_rate": float(done.mean()),
        "reliable_comparisons": reliable,
        "reliable_throughput_per_device_hour": reliable / (a.budget_seconds / 3600.0),
        "reliable_throughput_per_device_second": reliable / a.budget_seconds,
        "reliable_throughput_per_busy_hour": reliable / (a.busy_seconds / 3600.0) if a.busy_seconds > 0 else math.nan,
    }


def extreme_metrics(a: PolicyArrays) -> Dict[str, float]:
    """Distribution tail metrics over all completed comparisons (point estimates only)."""
    done = a.completed
    err, sync = a.error[done], a.synchronous_error[done]
    sep = np.abs(a.execution_b[done] - a.execution_a[done])
    wait = np.concatenate((a.waiting_a[done], a.waiting_b[done]))
    p95 = lambda x: np.percentile(x, 95)
    return {
        "error_p95": _stat(err, p95), "error_max": _stat(err, np.max),
        "sync_error_p95": _stat(sync, p95), "sync_error_max": _stat(sync, np.max),
        "separation_p95": _stat(sep, p95), "separation_max": _stat(sep, np.max),
        "waiting_p95": _stat(wait, p95), "waiting_max": _stat(wait, np.max),
    }


def device_metrics(a: PolicyArrays) -> Dict[str, float]:
    """Device-time and slot accounting (identical budget for every policy)."""
    return {
        "idle_slots": float(a.idle_slots), "executed_shots": float(a.executed_shots),
        "unexecuted_shots": float(a.unexecuted_shots), "total_slots": a.budget_seconds / a.slot_seconds,
        "budget_seconds": a.budget_seconds, "busy_seconds": a.busy_seconds, "slot_seconds": a.slot_seconds,
    }


def _value(estimate: float, unit_name: str, lower: Optional[float] = None, upper: Optional[float] = None,
           interval: bool = False) -> MetricValue:
    est = float(estimate) if estimate is not None and math.isfinite(estimate) else None
    return MetricValue(est, lower if interval else None, upper if interval else None, unit_name, interval)


def evaluate_scenario(
    label: str, rate_factor: float, kind: str, arrays: Dict[str, PolicyArrays], block_length: int,
    n_bootstrap: int, seed: int, confidence: float,
) -> ScenarioEvaluation:
    """Point estimates plus paired temporal-block-bootstrap intervals for every policy and difference."""
    n = next(iter(arrays.values())).n_requests

    def statistic(idx: np.ndarray) -> Dict[str, float]:
        per = {p: scalar_metrics(arrays[p], idx) for p in POLICY_ORDER}
        out = {f"{p}|{m}": v for p, ms in per.items() for m, v in ms.items()}
        for a, b in DIFFERENCES:
            for m in DIFFERENCE_METRICS:
                out[f"{a} - {b}|{m}"] = per[a][m] - per[b][m]
        return out

    cis = bootstrap_statistics(n, statistic, block_length, n_bootstrap, seed, confidence)
    metrics: Dict[str, Dict[str, MetricValue]] = {}
    for p in POLICY_ORDER:
        row = {m: _value(cis[f"{p}|{m}"].estimate, _unit(m), cis[f"{p}|{m}"].lower, cis[f"{p}|{m}"].upper, True)
               for m in RESAMPLED_METRICS}
        row.update({m: _value(v, _unit(m)) for m, v in extreme_metrics(arrays[p]).items()})
        row.update({m: _value(v, "s" if m.endswith("seconds") else "count") for m, v in device_metrics(arrays[p]).items()})
        metrics[p] = row
    differences = {
        f"{a} - {b}": {m: _value(cis[f"{a} - {b}|{m}"].estimate, _unit(m), cis[f"{a} - {b}|{m}"].lower,
                                 cis[f"{a} - {b}|{m}"].upper, True) for m in DIFFERENCE_METRICS}
        for a, b in DIFFERENCES
    }
    return ScenarioEvaluation(label, float(rate_factor), kind, arrays, metrics, differences)


# --------------------------------------------------------------------------- audits


def audit_fairness(
    scheduler: HoldoutEvaluation, arrays: Dict[str, PolicyArrays], model: Any, registry: CompatibilityRegistry,
    reference_times: np.ndarray, window_origin: float, window_seconds: float, training_end_time: float,
) -> Tuple[AuditCheck, ...]:
    """Independent checks that only the scheduling policy differs between the three runs."""
    checks: List[AuditCheck] = []
    runs = {r.result.policy_name: r.result for r in scheduler.runs}
    source_ids = {s.shot_id for s in scheduler.source_shots}
    same_shots = all(r.source_shots == len(source_ids)
                     and {e.shot_id for e in r.executed} | {s.shot_id for s in r.unexecuted_shots} == source_ids
                     for r in runs.values())
    checks.append(AuditCheck("same_source_shots_and_measurement_budget", same_shots,
                             f"every policy saw exactly the same {len(source_ids)} shots (one shot per slot)"))
    first = arrays[POLICY_ORDER[0]]
    same_workload = all(a.request_ids == first.request_ids and (a.circuit_a, a.circuit_b) == (first.circuit_a, first.circuit_b)
                        for a in arrays.values()) and len({r.workload_requests for r in runs.values()}) == 1
    checks.append(AuditCheck("same_workload_and_eligible_pairs", same_workload,
                             f"{first.n_requests} identical requests for all policies"))
    schedules = {r.schedule for r in runs.values()}
    budgets = {round(a.budget_seconds, 9) for a in arrays.values()}
    checks.append(AuditCheck("same_device_capacity_and_time_budget", len(schedules) == 1 and len(budgets) == 1,
                             f"one slot grid ({scheduler.schedule.total_slots} slots x {scheduler.schedule.slot_seconds!r} s)"))
    starts_equal = scheduler.schedule.start_time == min(s.arrival_time for s in scheduler.source_shots)
    arrivals_equal = True
    for p in POLICY_ORDER[1:]:
        both = first.completed & arrays[p].completed
        arrivals_equal &= bool(np.array_equal(first.arrival_a[both], arrays[p].arrival_a[both])
                               and np.array_equal(first.arrival_b[both], arrays[p].arrival_b[both]))
    checks.append(AuditCheck("same_arrival_pattern_and_initial_queue", bool(starts_equal and arrivals_equal),
                             "all policies start at the first arrival with an empty queue and see identical arrival times"))
    k_a = np.array([wait_length(c) for c in first.circuit_a], dtype=float)
    k_b = np.array([wait_length(c) for c in first.circuit_b], dtype=float)
    ref_independent = model.probability(k_a, reference_times) - model.probability(k_b, reference_times)
    refs_ok = True
    drift_ok = True
    for a in arrays.values():
        d = a.completed
        refs_ok &= bool(np.allclose(a.reference_delta[d], ref_independent[d], rtol=1e-12, atol=1e-15))
        recomputed = compute_comparison_errors(model, k_a[d], a.execution_a[d], k_b[d], a.execution_b[d],
                                               window_origin, window_seconds, reference_times=reference_times[d])
        drift_ok &= bool(np.allclose(a.error[d], recomputed.error, rtol=1e-12, atol=1e-15))
    checks.append(AuditCheck("same_reference_values_independent_of_scheduler", refs_ok,
                             "Delta_reference is recomputed from the model and the request's arrival window only; identical for every policy"))
    checks.append(AuditCheck("same_simulated_drift_for_all_policies", drift_ok,
                             "each policy's errors equal an independent recomputation with the one drift model"))
    eligible = {frozenset(p) for p in scheduler.eligible_pairs}
    only_eligible = all(frozenset(pair) in eligible
                        for a in arrays.values() for pair in zip(a.circuit_a, a.circuit_b))
    only_eligible &= not any(c.reviewer_decision.value != "Comparable" for c in (registry.get_pair(*p) for p in scheduler.eligible_pairs))
    checks.append(AuditCheck("only_comparable_pairs_scheduled", bool(only_eligible),
                             f"{len(eligible)} Comparable pairs; {scheduler.excluded_pair_count} Not comparable excluded"))
    try:
        scheduler.priorities.assert_matches(scheduler.split.train, registry)
        fresh = True
        detail = (f"priorities recalculated for this run from all {scheduler.priorities.metadata.train_shot_count} "
                  "training shots; one frozen mapping shared by the policies that use priorities")
    except PriorityError as exc:
        fresh, detail = False, str(exc)
    checks.append(AuditCheck("priorities_fresh_training_only_and_frozen", fresh, detail))
    train_end = max(s.arrival_time for s in scheduler.split.train)
    isolated = (train_end < scheduler.split.boundary and training_end_time < scheduler.split.boundary
                and scheduler.priorities.metadata.train_end < scheduler.split.boundary)
    checks.append(AuditCheck("no_holdout_in_model_or_priorities", bool(isolated),
                             f"model fitted on observations up to {training_end_time!r} s and priorities on shots up to "
                             f"{scheduler.priorities.metadata.train_end!r} s, both before the boundary {scheduler.split.boundary!r} s"))
    return tuple(checks)


def compare_with_priority_analysis(table: PriorityTable, csv_path: Path) -> Dict[str, Any]:
    """Cross-check this run's freshly calculated priorities against the standalone analysis CSV.

    Informational only: the file is never used to schedule anything.
    """
    path = Path(csv_path)
    if not path.is_file():
        return {"file": str(path), "found": False, "consistent": None,
                "note": "standalone priority analysis output not found; nothing to compare"}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    problems: List[str] = []
    seen = set()
    for row in rows:
        key = frozenset((row["circuit_a"], row["circuit_b"]))
        seen.add(key)
        try:
            fresh = table.get(row["circuit_a"], row["circuit_b"])
        except PriorityError:
            problems.append(f"{row['circuit_a']} | {row['circuit_b']} has a standalone priority but none in this run")
            continue
        if not math.isclose(fresh.priority_score, float(row["priority_score"]), rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"{row['circuit_a']} | {row['circuit_b']}: {fresh.priority_score!r} vs {row['priority_score']}")
        if str(fresh.rank) != row["rank"]:
            problems.append(f"{row['circuit_a']} | {row['circuit_b']}: rank {fresh.rank} vs {row['rank']}")
    missing = {p.key for p in table.priorities} - seen
    if missing:
        problems.append(f"{len(missing)} pair(s) of this run are missing from the standalone file")
    return {"file": str(path), "found": True, "consistent": not problems, "pairs_compared": len(rows),
            "problems": problems[:10],
            "note": "cross-check only; this run's priorities were calculated freshly and are not read from the file"}


# --------------------------------------------------------------------------- orchestration


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_output_dir(out_dir: Path, overwrite: bool) -> None:
    """Refuse to touch a directory that already holds results unless ``overwrite`` is set."""
    existing = [name for name in OUTPUT_FILES if (Path(out_dir) / name).exists()]
    if existing and not overwrite:
        raise ThreePolicyError(
            f"{out_dir} already contains results ({', '.join(existing[:3])}{'...' if len(existing) > 3 else ''}); "
            "choose a new --output-dir, or pass --overwrite to replace them deliberately"
        )


def _validate(config: ThreePolicyConfig) -> None:
    validate_split_parameters(config.train_fraction, config.window_seconds)
    if not (math.isfinite(config.slots_per_shot) and config.slots_per_shot > 0):
        raise ThreePolicyError(f"slots_per_shot must be positive and finite, got {config.slots_per_shot!r}")
    if config.n_bootstrap < 100:
        raise ThreePolicyError("n_bootstrap must be at least 100")
    if not 0.0 < config.confidence < 1.0:
        raise ThreePolicyError("confidence must be within (0, 1)")
    for f in config.stress_drift_rate_factors:
        if not (math.isfinite(f) and f > 0.0) or f == 1.0:
            raise ThreePolicyError(f"stress drift-rate factors must be positive, finite and not 1 (1 is the primary scenario); got {f!r}")
    check_output_dir(config.output_dir, config.overwrite)


def run_three_policy_evaluation(
    config: ThreePolicyConfig,
    trace: Optional[Sequence[ShotArrival]] = None,
    registry: Optional[CompatibilityRegistry] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ThreePolicyResult:
    """Run the full three-policy experiment (nothing is written; see ``write_outputs``)."""
    say = progress or (lambda _m: None)
    _validate(config)
    registry = registry or load_compatibility_registry(Path(config.registry_path))
    if not registry.eligible_pairs():
        raise ThreePolicyError("The registry has no 'Comparable' pairs; nothing can be evaluated")
    trace_from_file = trace is None
    trace = tuple(trace) if trace is not None else load_recorded_trace(Path(config.dataset_path))
    full_cfg = config.to_full_config()

    # 1. drift model: training portion only (the controlled ground-truth drift for the simulation).
    split = split_trace(trace, config.train_fraction, required_circuits=registry.circuits)
    train_obs = observations_from_shots(split.train)
    say(f"fitting the drift model on {len(split.train)} training shots")
    training = fit_training_stage(train_obs, full_cfg, say)
    model = training.model

    # 2. scheduler runs: priorities recalculated here, immediately before the policies run.
    say("recalculating priorities and running the three policies on identical holdout input")
    scheduler = run_holdout_evaluation(
        trace, registry, train_fraction=config.train_fraction, window_seconds=config.window_seconds,
        slots_per_shot=config.slots_per_shot,
    )
    if scheduler.split.boundary != split.boundary:
        raise ThreePolicyError("Model and scheduler disagree on the split boundary")
    origin = split.min_timestamp
    requests = scheduler.requests

    # 3. policy-independent reference time: window midpoint of each request's arrival.
    reference_times = reference_time(np.array([r.ready_time for r in requests]), origin, config.window_seconds)

    def arrays_for(m: Any) -> Dict[str, PolicyArrays]:
        return build_policy_arrays(scheduler, requests, m, origin, config.window_seconds,
                                   config.reliability_threshold, reference_times=reference_times)

    block_length = training_block_length(split.train, registry, model, origin, config.window_seconds, len(requests))
    say(f"bootstrap block length {block_length} requests (chosen from training data only)")

    primary_arrays = arrays_for(model)
    scenarios = [evaluate_scenario("primary: frozen training-fitted drift (x1)", 1.0, "primary", primary_arrays,
                                   block_length, config.n_bootstrap, config.seed, config.confidence)]
    for factor in config.stress_drift_rate_factors:
        say(f"synthetic stress scenario: drift {factor:g}x faster")
        stressed = arrays_for(TimeScaledModel(model, factor, split.boundary))
        scenarios.append(evaluate_scenario(f"synthetic stress: drift x{factor:g} faster", factor, "synthetic-stress",
                                           stressed, block_length, config.n_bootstrap, config.seed, config.confidence))

    # 4. diagnostics and audits (none of these feeds back into any decision).
    holdout_obs = observations_from_shots(split.holdout)
    diagnostic = evaluate_model(model, holdout_obs)
    baseline = constant_baseline(train_obs, holdout_obs)
    fairness = audit_fairness(scheduler, primary_arrays, model, registry, reference_times, origin,
                              config.window_seconds, float(train_obs.times[-1]))
    consistency = compare_with_priority_analysis(scheduler.priorities, Path(config.priority_analysis_csv))
    hashes: Dict[str, str] = {}
    if trace_from_file and Path(config.dataset_path).is_file():  # only when the trace really came from the file
        hashes["dataset_sha256"] = _sha256(Path(config.dataset_path))
    if Path(config.registry_path).is_file():
        hashes["registry_sha256"] = _sha256(Path(config.registry_path))
    return ThreePolicyResult(
        config=config, data=data_report(observations_from_shots(trace)), scheduler=scheduler, training=training,
        model_diagnostic=diagnostic, model_baseline=baseline, reference_times=reference_times,
        block_length=block_length, scenarios=tuple(scenarios), fairness=fairness,
        priority_consistency=consistency, hashes=hashes,
    )


# --------------------------------------------------------------------------- outputs


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else repr(float(value))
    return str(value)


def _write_rows(path: Path, columns: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(v) for v in row])


def _metric_rows(result: ThreePolicyResult, want_queue: bool) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for sc in result.scenarios:
        provenance_base = PROVENANCE_REPLAY if want_queue else PROVENANCE_MODEL
        scenario = f"{sc.kind} (drift x{sc.rate_factor:g})"
        for source, items in list(sc.metrics.items()) + list(sc.differences.items()):
            kind = "policy" if source in sc.metrics else "paired_difference"
            for metric, v in items.items():
                if metric.startswith(QUEUE_PREFIXES) != want_queue:
                    continue
                if metric in ("slot_seconds",):
                    continue
                prov = provenance_base if (want_queue or sc.kind == "primary") else "synthetic stress scenario; " + provenance_base
                rows.append([scenario, kind, source, metric, v.estimate, v.lower, v.upper, v.unit, prov])
    return rows


def build_results(result: ThreePolicyResult) -> Dict[str, Any]:
    cfg, sched, training = result.config, result.scheduler, result.training
    params = training.model.parameters
    def block(sc: ScenarioEvaluation) -> Dict[str, Any]:
        return {"label": sc.label, "kind": sc.kind, "drift_rate_factor": sc.rate_factor,
                "policies": sc.metrics, "paired_differences": sc.differences}
    return to_jsonable({
        "evaluation_kind": ("recorded-trace timing replay of the frozen scheduler scored by a model-simulated drift reference; "
                            "NOT a real-QPU experiment and no outcomes are simulated"),
        "provenance_labels": {
            "recorded_trace_timing_replay": "arrival pattern, queueing, waiting, measurement separation, idle slots and completions",
            "model_simulated": "p_A, p_B, Delta_measured, Delta_reference, comparison error and reliability, from a drift model fitted "
                               "on the training portion only and evaluated at those times (not device observations)",
            "synthetic_stress": "scenarios with drift-rate factor != 1 are synthetic stress tests, never the primary result",
        },
        "config": cfg, "reliability_threshold": cfg.reliability_threshold, "environment": environment_info(),
        "hashes": result.hashes, "data": result.data,
        "definitions": {
            "Delta_measured": "p_A(t_A) - p_B(t_B), t_A and t_B the policy's execution times",
            "Delta_reference": "p_A(t_ref) - p_B(t_ref), t_ref the midpoint of the fixed window containing the request's arrival (same for all policies)",
            "comparison_error": "E_AB = abs(Delta_measured - Delta_reference)",
            "synchronous_error": "diagnostic: reference at the comparison's own earlier execution time; isolates the drift between the two executions",
            "reliable": f"completed and E_AB <= {cfg.reliability_threshold} (threshold fixed in advance, inclusive)",
            "reliable_comparison_throughput": "reliable completed comparisons / device time; primary denominator = budgeted device time "
                                              "(total_slots * slot_seconds, idle slots included, identical for all policies); busy-time variant also reported",
            "time_separation": "abs(execution time of the second measurement - execution time of the first) of each completed comparison",
            "waiting_time": "execution time - arrival time per executed shot of a completed comparison",
            "window_seconds": cfg.window_seconds,
        },
        "split": {"train_fraction": cfg.train_fraction, "boundary_seconds": sched.split.boundary,
                  "train_shots": len(sched.split.train), "holdout_shots": len(sched.split.holdout),
                  "train_range": sched.split.train_range, "holdout_range": sched.split.holdout_range},
        "eligibility": {"authority": "reviewer_decision", "eligible_pairs": [list(p) for p in sched.eligible_pairs],
                        "excluded_pair_count": sched.excluded_pair_count,
                        "ineligible_circuit_shots_excluded": sched.ineligible_circuit_shots,
                        "unrequested_shots_excluded": sched.unrequested_shots,
                        "simulated_source_shots": len(sched.source_shots), "workload_requests": sched.workload_requests},
        "priorities_for_this_run": priority_table_to_dict(sched.priorities),
        "priority_analysis_cross_check": result.priority_consistency,
        "drift_model": {
            "equation": "p_l(t) = A + B exp(-l/l0) sin(2 pi l t_w Omega(t)); Omega(t) = gamma_0 + sum_w gamma_w f_w(t)",
            "selected_components": training.selection.selected_components, "selection_rule": training.selection.rule,
            "A": params.offset, "B": params.contrast, "l0": params.l0, "t_w_seconds": params.t_w,
            "gamma_0_hz": params.gammas[0], "gammas_hz": list(params.gammas[1:]),
            "fitted_on": "training portion only",
            "validity_diagnostic_after_freezing": {
                "note": "scored on holdout outcomes AFTER the model was frozen; influences nothing",
                "holdout_log_loss": result.model_diagnostic.log_loss,
                "holdout_constant_baseline_log_loss": result.model_baseline[0],
                "caveat": ("the frozen model does not forecast the holdout drift when its log loss exceeds the constant baseline; "
                           "results describe scheduling under this simulated drift, not the real device"),
            },
        },
        "bootstrap": {"method": "paired circular temporal block bootstrap over time-ordered requests",
                      "block_length_requests": result.block_length, "block_length_chosen_from": "training period only",
                      "replicates": cfg.n_bootstrap, "seed": cfg.seed, "confidence": cfg.confidence,
                      "note": "intervals reflect which time blocks were observed, not uncertainty in the drift-model parameters"},
        "primary": block(result.primary),
        "stress_scenarios": [block(s) for s in result.scenarios[1:]],
        "fairness_audit": {"all_passed": result.audits_passed,
                           "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in result.fairness]},
        "limitations": [
            "The drift is a model fitted on the training portion; it is not the real holdout drift.",
            "Errors scale with circuit wait length; the Comparable pairs are the low-wait-length circuits, so errors are small.",
            "Intervals exclude drift-model parameter uncertainty.",
            "Stress scenarios multiply the drift rate synthetically and say nothing about the real device.",
        ],
    })


def write_outputs(result: ThreePolicyResult, out_dir: Path, overwrite: bool = False) -> Dict[str, Path]:
    """Write ``results.json`` and the four CSV files (figures: ``write_figures``)."""
    out_dir = Path(out_dir)
    check_output_dir(out_dir, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = result.primary
    sched = result.scheduler
    files: Dict[str, Path] = {name: out_dir / name for name in OUTPUT_FILES}

    cols = ["policy", "request_id", "comparison_id", "status", "circuit_a", "circuit_b", "wait_length_a",
            "wait_length_b", "shots_a", "shots_b", "arrival_a", "arrival_b", "execution_a", "execution_b", "slot_a",
            "slot_b", "waiting_a", "waiting_b", "time_separation", "device_time_consumed", "reliable", "provenance"]
    rows = []
    for p in POLICY_ORDER:
        a = primary.arrays[p]
        for i, rid in enumerate(a.request_ids):
            d = bool(a.completed[i])
            v = lambda x: x if d else None
            rows.append([p, rid, int(a.comparison_id[i]) if d else None, "completed" if d else "incomplete",
                         a.circuit_a[i], a.circuit_b[i], wait_length(a.circuit_a[i]), wait_length(a.circuit_b[i]), 1, 1,
                         v(a.arrival_a[i]), v(a.arrival_b[i]), v(a.execution_a[i]), v(a.execution_b[i]), v(a.slot_a[i]),
                         v(a.slot_b[i]), v(a.waiting_a[i]), v(a.waiting_b[i]),
                         abs(a.execution_b[i] - a.execution_a[i]) if d else None,
                         2 * a.slot_seconds if d else None, bool(a.reliable[i]), PROVENANCE_REPLAY])
    _write_rows(files["policy_comparisons.csv"], cols, rows)

    cols = ["policy", "request_id", "status", "circuit_a", "circuit_b", "reference_time", "measured_delta",
            "reference_delta", "absolute_error", "synchronous_error", "reliable", "reliability_threshold", "provenance"]
    rows = []
    for p in POLICY_ORDER:
        a = primary.arrays[p]
        for i, rid in enumerate(a.request_ids):
            d = bool(a.completed[i])
            v = lambda x: x if d else None
            rows.append([p, rid, "completed" if d else "incomplete", a.circuit_a[i], a.circuit_b[i],
                         result.reference_times[i], v(a.measured_delta[i]), v(a.reference_delta[i]), v(a.error[i]),
                         v(a.synchronous_error[i]), bool(a.reliable[i]), result.config.reliability_threshold,
                         PROVENANCE_MODEL])
    _write_rows(files["comparison_errors.csv"], cols, rows)

    mcols = ["scenario", "row_type", "policy_or_difference", "metric", "estimate", "ci_lower", "ci_upper", "unit",
             "provenance"]
    _write_rows(files["queue_metrics.csv"], mcols, _metric_rows(result, True))
    _write_rows(files["reliability_metrics.csv"], mcols, _metric_rows(result, False))

    results = build_results(result)
    results["outputs"] = {k: str(v) for k, v in files.items()}
    results["csv_sha256"] = {name: _sha256(files[name]) for name in
                             ("policy_comparisons.csv", "comparison_errors.csv", "queue_metrics.csv", "reliability_metrics.csv")}
    with open(files["results.json"], "w", encoding="utf-8", newline="") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return files


# --------------------------------------------------------------------------- figures


def write_figures(result: ThreePolicyResult, out_dir: Path) -> List[Path]:
    """The five report figures (PNG, Agg backend)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"arrival-order": "C0", "consecutive-pair": "C1", "drift-aware": "C2"}
    primary = result.primary
    written: List[Path] = []

    def ecdf(ax, data, label, color):
        x = np.sort(data[np.isfinite(data)])
        if x.size:
            ax.step(x, np.arange(1, x.size + 1) / x.size, where="post", label=label, color=color)

    def save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    def fmt(m):
        return "n/a" if m.estimate is None else f"{m.estimate:.3g}"

    # fig1 comparison error distributions (ECDF) -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, attr, title in ((axes[0], "error", "E_AB (reference at request arrival window)"),
                            (axes[1], "synchronous_error", "pair-synchronous error (diagnostic)")):
        for p in POLICY_ORDER:
            a = primary.arrays[p]
            data = getattr(a, attr)[a.completed]
            key = "error_mean" if attr == "error" else "sync_error_mean"
            ecdf(ax, np.maximum(data, 1e-12), f"{p} (mean {fmt(primary.metrics[p][key])})", colors[p])
        ax.set(xscale="log", xlabel="absolute error (probability)", ylabel="cumulative fraction", title=title)
        ax.legend(fontsize=7)
    save(fig, "fig1_comparison_error.png")

    # fig2 measurement separation ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for p in POLICY_ORDER:
        a = primary.arrays[p]
        ecdf(axes[0], np.abs(a.execution_b - a.execution_a)[a.completed], p, colors[p])
    axes[0].set(xlabel="time between the two measurements (s)", ylabel="cumulative fraction", title="Separation distribution")
    axes[0].legend(fontsize=8)
    width = 0.26
    for j, stat in enumerate(("separation_mean", "separation_median", "separation_max")):
        axes[1].bar(np.arange(3) + (j - 1) * width, [primary.metrics[p][stat].estimate or 0.0 for p in POLICY_ORDER],
                    width, label=stat.split("_")[1])
    axes[1].set_xticks(range(3), POLICY_ORDER, fontsize=8)
    axes[1].set(ylabel="seconds", title="Mean / median / maximum separation")
    axes[1].legend(fontsize=8)
    save(fig, "fig2_measurement_separation.png")

    # fig3 waiting time ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for p in POLICY_ORDER:
        a = primary.arrays[p]
        ecdf(axes[0], np.concatenate((a.waiting_a[a.completed], a.waiting_b[a.completed])), p, colors[p])
    axes[0].set(xlabel="queue waiting time per shot (s)", ylabel="cumulative fraction", title="Waiting-time distribution")
    axes[0].legend(fontsize=8)
    for j, stat in enumerate(("waiting_mean", "waiting_median", "waiting_p95", "waiting_max")):
        axes[1].bar(np.arange(3) + (j - 1.5) * 0.2, [primary.metrics[p][stat].estimate or 0.0 for p in POLICY_ORDER],
                    0.2, label=stat.split("_")[1])
    axes[1].set_xticks(range(3), POLICY_ORDER, fontsize=8)
    axes[1].set(ylabel="seconds", title="Waiting time summary")
    axes[1].legend(fontsize=8)
    save(fig, "fig3_waiting_time.png")

    # fig4 reliable throughput ----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    key = "reliable_throughput_per_device_hour"
    est = [primary.metrics[p][key].estimate or 0.0 for p in POLICY_ORDER]
    lo = [est[i] - (primary.metrics[p][key].lower or est[i]) for i, p in enumerate(POLICY_ORDER)]
    hi = [(primary.metrics[p][key].upper or est[i]) - est[i] for i, p in enumerate(POLICY_ORDER)]
    axes[0].bar(range(3), est, yerr=[lo, hi], capsize=4, color=[colors[p] for p in POLICY_ORDER])
    axes[0].set_xticks(range(3), POLICY_ORDER, fontsize=8)
    axes[0].set(ylabel="reliable comparisons per device-hour",
                title=f"Primary: E_AB <= {result.config.reliability_threshold} (95% CI)")
    factors = [s.rate_factor for s in result.scenarios]
    for p in POLICY_ORDER:
        axes[1].plot(factors, [s.metrics[p][key].estimate or 0.0 for s in result.scenarios], "o-", label=p, color=colors[p])
    axes[1].set(xscale="log", xlabel="drift-rate factor (1 = fitted drift; others synthetic stress)",
                ylabel="reliable comparisons per device-hour", title="Controlled drift-rate scenarios")
    axes[1].legend(fontsize=8)
    save(fig, "fig4_reliable_throughput.png")

    # fig5 relative summary --------------------------------------------------------------------------
    names = [("error_mean", "mean error"), ("sync_error_mean", "mean sync. error"), ("separation_mean", "mean separation"),
             ("waiting_mean", "mean waiting"), (key, "reliable throughput")]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for j, p in enumerate(POLICY_ORDER):
        ratios = []
        for metric, _ in names:
            base = primary.metrics["arrival-order"][metric].estimate
            val = primary.metrics[p][metric].estimate
            ratios.append(val / base if (base and val is not None) else np.nan)
        ax.bar(np.arange(len(names)) + (j - 1) * 0.26, ratios, 0.26, label=p, color=colors[p])
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xticks(range(len(names)), [n for _, n in names], fontsize=8)
    ax.set(ylabel="ratio to arrival-order (baseline = 1)",
           title="Policy comparison relative to the no-QCS baseline (lower is better except throughput)")
    ax.legend(fontsize=8)
    save(fig, "fig5_policy_comparison.png")
    return written
