"""Queue-based compatible-shot pairing simulator over a recorded trace.

Lifecycle at each distinct arrival timestamp ``t``:

1. every shot arriving at ``t`` is added to the pending queue;
2. the policy is asked for a pair; each pair is completed at simulator time ``t`` and
   removed permanently from the queue;
3. step 2 repeats until the policy returns ``None``;
4. the simulator advances to the next arrival timestamp.

At the end of the trace, remaining pending shots are reported as unmatched. This is a
replay of observed historical shots; it does not model what hardware would have
measured under a different schedule.
"""

from typing import List, Sequence, Set

from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, CompatibilityStatus
from quantum_drift_scheduler.data.compatibility import UnknownPairError
from quantum_drift_scheduler.scheduling.models import (
    CompletedComparison,
    PendingQueue,
    ShotArrival,
    SimulationError,
    SimulationResult,
)
from quantum_drift_scheduler.scheduling.policies import PairingPolicy


def _validate_trace(trace: Sequence[ShotArrival], registry: CompatibilityRegistry) -> str:
    known = set(registry.circuits)
    seen: Set[str] = set()
    previous = None
    for shot in trace:
        if shot.circuit not in known:
            raise SimulationError(
                f"Shot {shot.shot_id!r} has circuit {shot.circuit!r} that is not in the compatibility registry"
            )
        if shot.outcome not in (0, 1):
            raise SimulationError(f"Shot {shot.shot_id!r} has non-binary outcome {shot.outcome!r}")
        if shot.shot_id in seen:
            raise SimulationError(f"Duplicate shot_id {shot.shot_id!r} in trace")
        seen.add(shot.shot_id)
        if previous is not None and shot.sort_key < previous.sort_key:
            raise SimulationError(
                f"Trace is not in non-decreasing (arrival_time, source_index) order at shot {shot.shot_id!r}"
            )
        previous = shot
    return trace[0].input_source if trace else ""


def run_simulation(
    trace: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    policy: PairingPolicy,
    input_source: str = "",
) -> SimulationResult:
    """Replay ``trace`` through ``policy`` and return completed comparisons and leftovers.

    Args:
        trace: Shots in non-decreasing ``(arrival_time, source_index)`` order.
        registry: Compatibility registry; only ``compatible`` pairs may complete.
        policy: Pairing policy.
        input_source: Optional label for the input; defaults to the first shot's source.

    Raises:
        SimulationError: invalid trace, or the policy returned an invalid pair.
    """
    derived_source = _validate_trace(trace, registry)
    pending = PendingQueue()
    comparisons: List[CompletedComparison] = []
    consumed: Set[str] = set()

    position = 0
    total = len(trace)
    while position < total:
        now = trace[position].arrival_time
        while position < total and trace[position].arrival_time == now:
            pending.add(trace[position])
            position += 1

        while True:
            choice = policy.select_pair(pending, registry, now)
            if choice is None:
                break
            first, second = choice
            _check_choice(first, second, pending, registry, consumed, policy)
            a, b = (first, second) if first.sort_key <= second.sort_key else (second, first)
            pending.remove(a)
            pending.remove(b)
            consumed.add(a.shot_id)
            consumed.add(b.shot_id)
            comparisons.append(
                CompletedComparison(
                    comparison_id=len(comparisons) + 1,
                    policy_name=policy.name,
                    shot_id_a=a.shot_id,
                    shot_id_b=b.shot_id,
                    circuit_a=a.circuit,
                    circuit_b=b.circuit,
                    outcome_a=a.outcome,
                    outcome_b=b.outcome,
                    arrival_time_a=a.arrival_time,
                    arrival_time_b=b.arrival_time,
                    timestamp_separation=abs(a.arrival_time - b.arrival_time),
                    completion_time=now,
                    waiting_time_a=now - a.arrival_time,
                    waiting_time_b=now - b.arrival_time,
                )
            )

    return SimulationResult(
        input_source=input_source or derived_source,
        policy_name=policy.name,
        input_shots=total,
        comparisons=tuple(comparisons),
        unmatched_shots=pending.all_shots(),
    )


def _check_choice(
    first: ShotArrival,
    second: ShotArrival,
    pending: PendingQueue,
    registry: CompatibilityRegistry,
    consumed: Set[str],
    policy: PairingPolicy,
) -> None:
    who = f"Policy {policy.name!r}"
    if first.shot_id == second.shot_id:
        raise SimulationError(f"{who} paired shot {first.shot_id!r} with itself")
    for shot in (first, second):
        if shot.shot_id in consumed:
            raise SimulationError(f"{who} reused already-consumed shot {shot.shot_id!r}")
        if shot not in pending:
            raise SimulationError(f"{who} selected shot {shot.shot_id!r} that is not pending")
    if first.circuit == second.circuit:
        raise SimulationError(f"{who} paired two shots of the same circuit {first.circuit!r}")
    try:
        status = registry.get_status(first.circuit, second.circuit)
    except UnknownPairError:
        raise SimulationError(
            f"{who} selected pair ({first.circuit!r}, {second.circuit!r}) that is not in the registry"
        ) from None
    if status is not CompatibilityStatus.COMPATIBLE:
        raise SimulationError(
            f"{who} selected pair ({first.circuit!r}, {second.circuit!r}) with status {status.value!r}; "
            "only 'compatible' pairs may complete"
        )
