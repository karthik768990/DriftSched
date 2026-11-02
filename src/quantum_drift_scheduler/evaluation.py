"""Held-out replay evaluation: train priorities early, evaluate policies later.

Pipeline (one call = one scheduler run)::

    full trace -> chronological split -> eligible pairs (registry ``reviewer_decision``)
              -> priorities recalculated NOW from the complete training portion -> freeze
              -> holdout shots of reviewer-approved comparisons only
              -> {arrival-order, consecutive-pair, drift-aware}

Eligibility comes only from ``reviewer_decision`` (``Comparable`` vs ``Not comparable``).
Priorities are never cached: every call recomputes them for its own training portion and
passes the immutable table straight to the drift-aware policy. All three policies receive
the identical holdout shots, slot schedule, prebuilt pair-request workload, registry and
frozen priority table. The results
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
    SimulationError,
    DEFAULT_SLOTS_PER_SHOT,
    ArrivalOrderPolicy,
    ConsecutivePairPolicy,
    DriftAwarePairPolicy,
    ShotArrival,
    SimulationMetrics,
    SimulationResult,
    SlotSchedule,
    PairRequest,
    build_slot_schedule,
    compute_metrics,
    prepare_scheduler_input,
    run_simulation,
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
    source_shots: Tuple[ShotArrival, ...] = ()
    requests: Tuple[PairRequest, ...] = ()
    eligible_pairs: Tuple[Tuple[str, str], ...] = ()
    excluded_pair_count: int = 0
    ineligible_circuit_shots: int = 0

    @property
    def leakage_check(self) -> str:
        return (
            "Priorities were learned from the "
            f"{self.priorities.metadata.train_shot_count} training shots only "
            f"(all with arrival_time < boundary {self.split.boundary!r}) for the "
            f"{len(self.eligible_pairs)} Comparable pairs only; "
            "no holdout shot was passed to priority learning."
        )


def calculate_run_priorities(
    train_shots: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    boundary: float,
    window_origin: float,
    train_fraction: float,
    window_seconds: float,
    min_valid_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS,
) -> PriorityTable:
    """Calculate the priorities for ONE scheduler run, immediately before it starts.

    Uses the run's complete training portion and only the pairs whose ``reviewer_decision``
    is ``Comparable``. The table is verified against exactly those inputs, so a stale table
    cannot slip through, and it is immutable for the rest of the run.
    """
    table = learn_pair_priorities(
        train_shots,
        registry,
        boundary=boundary,
        window_origin=window_origin,
        train_fraction=train_fraction,
        window_seconds=window_seconds,
        min_valid_transitions=min_valid_transitions,
    )
    table.assert_matches(train_shots, registry)
    return table


def run_holdout_evaluation(
    trace: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    slots_per_shot: float = DEFAULT_SLOTS_PER_SHOT,
    min_valid_transitions: int = DEFAULT_MIN_VALID_TRANSITIONS,
    input_source: Optional[str] = None,
) -> HoldoutEvaluation:
    """One scheduler run: split, recalculate priorities from training, evaluate all policies on holdout.

    Raises:
        SplitError: invalid split parameters or insufficient partition data.
        PriorityError / SimulationError: invalid priority learning, no eligible pairs, or
            invalid simulation input.
    """
    validate_split_parameters(train_fraction, window_seconds)
    eligible_circuits = registry.eligible_circuits()
    if not eligible_circuits:
        raise SimulationError("The registry has no 'Comparable' pairs; nothing can be scheduled")
    # The chronological split is defined on the full trace (the existing split rule); eligibility
    # only decides which circuits must have enough data and which shots are scheduled.
    split = split_trace(trace, train_fraction, required_circuits=eligible_circuits)

    # Priorities: recalculated here, once, from the complete training portion; never read from disk.
    priorities = calculate_run_priorities(
        split.train, registry, split.boundary, split.min_timestamp, train_fraction, window_seconds,
        min_valid_transitions,
    )

    # Round alignment comes from per-circuit shot order in the full trace; it uses no outcomes,
    # priorities or training data, and requests only link two holdout shots.
    prepared = prepare_scheduler_input(split.holdout, registry, round_reference=trace)
    source_shots, requests = prepared.source_shots, prepared.requests
    schedule = build_slot_schedule(source_shots, slots_per_shot)
    source = input_source if input_source is not None else (trace[0].input_source if trace else "")

    policies = (ArrivalOrderPolicy(), ConsecutivePairPolicy(), DriftAwarePairPolicy(priorities))
    runs = []
    for policy in policies:
        result = run_simulation(
            source_shots, registry, policy, schedule=schedule, requests=requests, input_source=source
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
        unrequested_shots=prepared.unrequested_shots,
        runs=tuple(runs),
        holdout_fingerprint=training_fingerprint(split.holdout),
        source_shots=source_shots,
        requests=requests,
        eligible_pairs=tuple((p.circuit_a, p.circuit_b) for p in registry.eligible_pairs()),
        excluded_pair_count=len(registry.excluded_pairs()),
        ineligible_circuit_shots=prepared.ineligible_circuit_shots,
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
        "eligibility": {
            "authority": "reviewer_decision",
            "eligible_pairs": [list(p) for p in evaluation.eligible_pairs],
            "excluded_pair_count": evaluation.excluded_pair_count,
            "ineligible_circuit_shots_excluded": evaluation.ineligible_circuit_shots,
            "unrequested_shots_excluded": evaluation.unrequested_shots,
            "simulated_source_shots": len(evaluation.source_shots),
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
