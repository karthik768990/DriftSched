"""Held-out replay evaluation: train priorities early, evaluate policies later.

Pipeline::

    full trace -> chronological split -> learn priorities (training only) -> freeze
              -> holdout shots only -> {arrival-order, consecutive-pair, drift-aware}

All three policies receive the identical holdout shots, slot schedule, prebuilt
pair-request workload, compatibility registry and frozen priority table. The results
are held-out replay scheduling metrics on recorded outcomes; they are not
counterfactual hardware predictions and the observed outcome difference is not a
physical comparison error.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

from quantum_drift_scheduler.analysis.holdout import (
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_WINDOW_SECONDS,
    SplitError,
    TraceSplit,
    split_trace,
    validate_split_parameters,
)
from quantum_drift_scheduler.analysis.priorities import (
    DEFAULT_MIN_VALID_TRANSITIONS,
    PriorityTable,
    learn_pair_priorities,
    priority_table_to_dict,
    training_fingerprint,
)
from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry
from quantum_drift_scheduler.scheduling import (
    DEFAULT_SLOTS_PER_SHOT,
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    ShotArrival,
    SimulationMetrics,
    SimulationResult,
    SlotSchedule,
    build_pair_requests,
    build_slot_schedule,
    compute_metrics,
    run_simulation,
    shot_round_indices,
)

LIMITATION = (
    "These are held-out replay scheduling metrics on recorded outcomes. They do not predict "
    "counterfactual hardware outcomes, and the observed outcome difference is not a physical "
    "comparison error. Counterfactual hardware-performance claims require a later drift/outcome model."
)


@dataclass(frozen=True)
class PolicyRun:
    """One policy's simulation result and metrics."""

    result: SimulationResult
    metrics: SimulationMetrics


@dataclass(frozen=True)
class HoldoutEvaluation:
    """Everything produced by a holdout evaluation run."""

    input_source: str
    split: TraceSplit
    priorities: PriorityTable
    schedule: SlotSchedule
    slots_per_shot: float
    workload_requests: int
    unrequested_shots: int
    runs: Tuple[PolicyRun, ...]
    holdout_fingerprint: str

    @property
    def leakage_check(self) -> str:
        return (
            "Priorities were learned from the "
            f"{self.priorities.metadata.train_shot_count} training shots only "
            f"(all with arrival_time < boundary {self.split.boundary!r}); "
            "no holdout shot was passed to priority learning."
        )


def run_holdout_evaluation(
    trace: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    slots_per_shot: float = DEFAULT_SLOTS_PER_SHOT,
    min_valid_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS,
    input_source: Optional[str] = None,
) -> HoldoutEvaluation:
    """Split, learn priorities from training only, and evaluate all three policies on holdout.

    Raises:
        SplitError: invalid split parameters or insufficient partition data.
        PriorityError / SimulationError: invalid priority learning or simulation input.
    """
    validate_split_parameters(train_fraction, window_seconds)
    split = split_trace(trace, train_fraction, required_circuits=registry.circuits)

    priorities = learn_pair_priorities(
        split.train,
        registry,
        boundary=split.boundary,
        window_origin=split.min_timestamp,
        train_fraction=train_fraction,
        window_seconds=window_seconds,
        min_valid_transitions=min_valid_transitions,
    )

    holdout = split.holdout
    schedule = build_slot_schedule(holdout, slots_per_shot)
    # Round alignment comes from per-circuit shot order in the full trace; it uses no outcomes,
    # priorities or training data, and requests only link two holdout shots.
    requests = build_pair_requests(holdout, registry, shot_round_indices(trace))
    requested = {sid for r in requests for sid in r.shot_ids}
    source = input_source if input_source is not None else (trace[0].input_source if trace else "")

    policies = (ArrivalOrderPolicy(), ConsecutivePairPolicy(), DriftAwarePairPolicy(priorities))
    runs = []
    for policy in policies:
        result = run_simulation(
            holdout, registry, policy, schedule=schedule, requests=requests, input_source=source
        )
        runs.append(PolicyRun(result=result, metrics=compute_metrics(result)))

    budgets = {(r.metrics.source_shots, r.metrics.total_slots, r.result.workload_requests) for r in runs}
    if len(budgets) != 1:  # defensive: fairness invariant
        raise SplitError(f"Policies did not share an identical budget and workload: {budgets}")

    return HoldoutEvaluation(
        input_source=source,
        split=split,
        priorities=priorities,
        schedule=schedule,
        slots_per_shot=float(slots_per_shot),
        workload_requests=len(requests),
        unrequested_shots=sum(1 for s in holdout if s.shot_id not in requested),
        runs=tuple(runs),
        holdout_fingerprint=training_fingerprint(holdout),
    )


def evaluation_to_dict(evaluation: HoldoutEvaluation) -> dict:
    """JSON-serializable form of a holdout evaluation."""
    split = evaluation.split
    return {
        "limitation": LIMITATION,
        "leakage_check": evaluation.leakage_check,
        "input_source": evaluation.input_source,
        "split": {
            "train_fraction": split.train_fraction,
            "boundary": split.boundary,
            "min_timestamp": split.min_timestamp,
            "max_timestamp": split.max_timestamp,
            "train_range": list(split.train_range),
            "holdout_range": list(split.holdout_range),
            "train_shots": len(split.train),
            "holdout_shots": len(split.holdout),
            "per_circuit": [
                {"circuit": c, "train_shots": t, "holdout_shots": h} for c, t, h in split.circuit_counts()
            ],
            "holdout_fingerprint": evaluation.holdout_fingerprint,
        },
        "schedule": {
            "start_time": evaluation.schedule.start_time,
            "slot_seconds": evaluation.schedule.slot_seconds,
            "total_slots": evaluation.schedule.total_slots,
            "slots_per_shot": evaluation.slots_per_shot,
            "workload_requests": evaluation.workload_requests,
            "unrequested_shots": evaluation.unrequested_shots,
        },
        "priorities": priority_table_to_dict(evaluation.priorities),
        "policies": {
            run.result.policy_name: {
                "metrics": asdict(run.metrics),
                "comparisons": [
                    {
                        **asdict(c),
                        "slot_separation": c.slot_separation,
                        "time_separation": c.time_separation,
                        "observed_difference": c.observed_difference,
                    }
                    for c in run.result.comparisons
                ],
            }
            for run in evaluation.runs
        },
    }
