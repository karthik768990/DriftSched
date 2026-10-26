"""Slot-based queue simulator over a recorded shot trace.

Model: one logical shot executes per service slot. A shot can execute in slot ``s``
only if it arrived by the slot's start time. Policies decide what runs next:

* shot policies (``arrival-order``) execute the lowest-``shot_key`` arrived shot each
  slot and never idle while a shot is pending; comparisons complete when both shots
  of a workload request have executed (see ``ArrivalOrderPolicy``);
* pair policies dispatch the lowest-``request_key`` ready request as an atomic block
  occupying slots ``s`` and ``s + 1`` and idle when nothing is ready. A block needs
  two remaining slots; with one slot left the slot idles.

Every policy receives the same shots, slot schedule and prebuilt request workload,
and every shot executes at most once and is consumed by at most one comparison. The
recorded outcomes are replayed as observed; this is not a model of what hardware
would have measured under a different schedule.
"""

import heapq
from typing import Dict, List, Optional, Sequence, Set, Tuple

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityRegistry,
    CompatibilityStatus,
    UnknownPairError,
    wait_length,
)
from quantum_drift_scheduler.scheduling.models import (
    CompletedComparison,
    ExecutedShot,
    PairRequest,
    ShotArrival,
    SimulationError,
    SimulationResult,
    SlotSchedule,
)
from quantum_drift_scheduler.scheduling.workload import (
    DEFAULT_SLOTS_PER_SHOT,
    build_pair_requests,
    build_slot_schedule,
)


def validate_trace(trace: Sequence[ShotArrival], registry: CompatibilityRegistry) -> None:
    """Reject traces with unknown circuits, bad outcomes, duplicate ids or disorder."""
    known = set(registry.circuits)
    seen: Set[str] = set()
    previous: Optional[ShotArrival] = None
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


def validate_requests(
    requests: Sequence[PairRequest], trace: Sequence[ShotArrival], registry: CompatibilityRegistry
) -> None:
    """Reject a workload that references foreign shots or non-``compatible`` pairs."""
    by_id = {shot.shot_id: shot for shot in trace}
    ids: Set[str] = set()
    for request in requests:
        if request.request_id in ids:
            raise SimulationError(f"Duplicate request_id {request.request_id!r}")
        ids.add(request.request_id)
        for shot in (request.first, request.second):
            if by_id.get(shot.shot_id) != shot:
                raise SimulationError(
                    f"Request {request.request_id!r} references shot {shot.shot_id!r} that is not in the trace"
                )
        a, b = request.first.circuit, request.second.circuit
        if a == b:
            raise SimulationError(f"Request {request.request_id!r} pairs two shots of circuit {a!r}")
        try:
            status = registry.get_status(a, b)
        except UnknownPairError:
            raise SimulationError(
                f"Request {request.request_id!r} pair ({a!r}, {b!r}) is not in the registry"
            ) from None
        if status is not CompatibilityStatus.COMPATIBLE:
            raise SimulationError(
                f"Request {request.request_id!r} pair ({a!r}, {b!r}) has status {status.value!r}; "
                "only 'compatible' pairs may complete"
            )
        if (wait_length(a), a) > (wait_length(b), b):
            raise SimulationError(
                f"Request {request.request_id!r} must list the smaller wait length first"
            )


def run_simulation(
    trace: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    policy,
    schedule: Optional[SlotSchedule] = None,
    requests: Optional[Sequence[PairRequest]] = None,
    input_source: str = "",
    slots_per_shot: float = DEFAULT_SLOTS_PER_SHOT,
) -> SimulationResult:
    """Run one policy over ``trace`` and return executions, comparisons and leftovers.

    Args:
        trace: Shots in non-decreasing ``(arrival_time, source_index)`` order.
        registry: Compatibility registry; only ``compatible`` pairs may complete.
        policy: A shot policy (``dispatch_unit == "shot"``) or pair policy (``"pair"``).
        schedule: Slot grid. Default: ``build_slot_schedule(trace, slots_per_shot)``.
        requests: Prebuilt pair-request workload. Default: ``build_pair_requests(trace)``.
            Pass the same objects to every policy being compared.
        input_source: Label for the input; defaults to the first shot's source.

    Raises:
        SimulationError: invalid trace, schedule, workload or policy.
    """
    validate_trace(trace, registry)
    if schedule is None:
        schedule = build_slot_schedule(trace, slots_per_shot)
    if requests is None:
        requests = build_pair_requests(trace, registry)
    validate_requests(requests, trace, registry)

    unit = getattr(policy, "dispatch_unit", None)
    if unit == "pair":
        executed, comparisons, idle = _run_pair(trace, requests, policy, schedule)
    elif unit == "shot":
        executed, comparisons, idle = _run_shots(trace, requests, policy, schedule)
    else:
        raise SimulationError(f"Policy {getattr(policy, 'name', policy)!r} has invalid dispatch_unit {unit!r}")

    done = {e.shot_id for e in executed}
    return SimulationResult(
        input_source=input_source or (trace[0].input_source if trace else ""),
        policy_name=policy.name,
        source_shots=len(trace),
        schedule=schedule,
        executed=tuple(executed),
        comparisons=tuple(comparisons),
        unexecuted_shots=tuple(s for s in trace if s.shot_id not in done),
        idle_slots=idle,
        workload_requests=len(requests),
    )


def _executed(shot: ShotArrival, slot: int, schedule: SlotSchedule) -> ExecutedShot:
    return ExecutedShot(
        shot_id=shot.shot_id,
        circuit=shot.circuit,
        arrival_time=shot.arrival_time,
        slot=slot,
        execution_time=schedule.slot_time(slot),
    )


def _comparison(
    number: int,
    policy_name: str,
    request: PairRequest,
    slot_first: int,
    slot_second: int,
    schedule: SlotSchedule,
) -> CompletedComparison:
    completion_slot = max(slot_first, slot_second)
    return CompletedComparison(
        comparison_id=number,
        policy_name=policy_name,
        request_id=request.request_id,
        shot_id_a=request.first.shot_id,
        shot_id_b=request.second.shot_id,
        circuit_a=request.first.circuit,
        circuit_b=request.second.circuit,
        outcome_a=request.first.outcome,
        outcome_b=request.second.outcome,
        arrival_time_a=request.first.arrival_time,
        arrival_time_b=request.second.arrival_time,
        slot_a=slot_first,
        slot_b=slot_second,
        execution_time_a=schedule.slot_time(slot_first),
        execution_time_b=schedule.slot_time(slot_second),
        completion_slot=completion_slot,
        completion_time=schedule.slot_time(completion_slot),
    )


def _run_pair(
    trace: Sequence[ShotArrival],
    requests: Sequence[PairRequest],
    policy,
    schedule: SlotSchedule,
) -> Tuple[List[ExecutedShot], List[CompletedComparison], int]:
    order = sorted(range(len(requests)), key=lambda i: (requests[i].ready_time, i))
    heap: List[Tuple[tuple, int]] = []
    done: Set[str] = set()
    executed: List[ExecutedShot] = []
    comparisons: List[CompletedComparison] = []
    idle = 0
    pointer = 0
    slot = 0
    total = schedule.total_slots

    while slot < total:
        now = schedule.slot_time(slot)
        while pointer < len(order) and requests[order[pointer]].ready_time <= now:
            index = order[pointer]
            heapq.heappush(heap, (policy.request_key(requests[index]), index))
            pointer += 1

        while heap:
            request = requests[heap[0][1]]
            if request.first.shot_id in done or request.second.shot_id in done:
                heapq.heappop(heap)  # a shot was already executed by another request
                continue
            break

        if not heap or slot + 1 >= total:
            idle += 1
            slot += 1
            continue

        request = requests[heapq.heappop(heap)[1]]
        executed.append(_executed(request.first, slot, schedule))
        executed.append(_executed(request.second, slot + 1, schedule))
        done.add(request.first.shot_id)
        done.add(request.second.shot_id)
        comparisons.append(
            _comparison(len(comparisons) + 1, policy.name, request, slot, slot + 1, schedule)
        )
        slot += 2

    return executed, comparisons, idle


def _run_shots(
    trace: Sequence[ShotArrival],
    requests: Sequence[PairRequest],
    policy,
    schedule: SlotSchedule,
) -> Tuple[List[ExecutedShot], List[CompletedComparison], int]:
    partners: Dict[str, List[PairRequest]] = {}
    for request in requests:
        partners.setdefault(request.first.shot_id, []).append(request)
        partners.setdefault(request.second.shot_id, []).append(request)

    heap: List[Tuple[tuple, int]] = []
    slot_of: Dict[str, int] = {}
    consumed: Set[str] = set()
    executed: List[ExecutedShot] = []
    comparisons: List[CompletedComparison] = []
    idle = 0
    pointer = 0

    for slot in range(schedule.total_slots):
        now = schedule.slot_time(slot)
        while pointer < len(trace) and trace[pointer].arrival_time <= now:
            heapq.heappush(heap, (policy.shot_key(trace[pointer]), pointer))
            pointer += 1
        if not heap:
            idle += 1
            continue

        shot = trace[heapq.heappop(heap)[1]]
        slot_of[shot.shot_id] = slot
        executed.append(_executed(shot, slot, schedule))

        best: Optional[Tuple[int, str, PairRequest]] = None
        for request in partners.get(shot.shot_id, ()):
            other = request.second if request.first.shot_id == shot.shot_id else request.first
            if other.shot_id in slot_of and other.shot_id not in consumed:
                candidate = (slot_of[other.shot_id], request.request_id, request)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is not None:
            request = best[2]
            consumed.add(request.first.shot_id)
            consumed.add(request.second.shot_id)
            comparisons.append(
                _comparison(
                    len(comparisons) + 1,
                    policy.name,
                    request,
                    slot_of[request.first.shot_id],
                    slot_of[request.second.shot_id],
                    schedule,
                )
            )

    return executed, comparisons, idle
