"""Prebuilt pair-request workload and slot-budget construction.

The workload is built once from the shot trace and the compatibility registry, with
no policy and no drift information involved, so every policy sees the identical set
of comparison requests.

Round-robin rule. Circuits present in the trace are ordered by wait length. The i-th shot of every circuit
(its *round* ``r = i``) belongs to acquisition round ``r``. Round ``r`` uses perfect
matching ``r mod (n - 1)`` of a round-robin tournament (circle method) over the ``n``
circuits; each matched circuit pair yields one request pairing the two circuits'
round-``r`` shots. For the 14 Ramsey circuits the 13 matchings partition all 91
unordered circuit pairs, so every compatible pair recurs once every 13 rounds, and
each shot belongs to exactly one request. A pair that the registry does not mark
``compatible`` produces no request.

Round indices may be supplied (for example computed on the full trace) so a
sub-trace such as a holdout partition keeps the same round alignment. Requests are
only created when both shots are present in the trace being scheduled.
"""

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityRegistry,
    UnknownPairError,
    wait_length,
)
from quantum_drift_scheduler.scheduling.models import (
    PairRequest,
    ShotArrival,
    SimulationError,
    SlotSchedule,
)

DEFAULT_SLOTS_PER_SHOT = 1.0


def circuits_compatible(registry: CompatibilityRegistry, circuit_a: str, circuit_b: str) -> bool:
    """True if two distinct circuits form a registry-``compatible`` pair.

    Same-circuit shots are never paired, and pairs absent from the registry are not
    compatible; this never defaults to ``True``.
    """
    if circuit_a == circuit_b:
        return False
    try:
        return registry.is_compatible(circuit_a, circuit_b)
    except UnknownPairError:
        return False


def make_request(request_id: str, shot_x: ShotArrival, shot_y: ShotArrival) -> PairRequest:
    """Build a request with the smaller-wait-length shot first."""
    try:
        kx, ky = wait_length(shot_x.circuit), wait_length(shot_y.circuit)
    except ValueError as exc:
        raise SimulationError(str(exc)) from None
    if (kx, shot_x.circuit) <= (ky, shot_y.circuit):
        return PairRequest(request_id, shot_x, shot_y, kx, ky)
    return PairRequest(request_id, shot_y, shot_x, ky, kx)


def shot_round_indices(trace: Sequence[ShotArrival]) -> Dict[str, int]:
    """Round index of every shot: its zero-based occurrence among its circuit's shots in ``trace``."""
    counters: Dict[str, int] = {}
    rounds: Dict[str, int] = {}
    for shot in trace:
        rounds[shot.shot_id] = counters.get(shot.circuit, 0)
        counters[shot.circuit] = rounds[shot.shot_id] + 1
    return rounds


def round_robin_matchings(circuits: Sequence[str]) -> Tuple[Tuple[Tuple[str, str], ...], ...]:
    """Round-robin (circle method) matchings over ``circuits`` ordered by wait length.

    ``n`` circuits (``n`` even) give ``n - 1`` perfect matchings whose pairs partition
    all ``n (n - 1) / 2`` unordered pairs. For odd ``n`` one circuit sits out each round.
    """
    nodes: List[Optional[str]] = sorted(circuits, key=lambda c: (wait_length(c), c))
    if len(nodes) < 2:
        return ()
    if len(nodes) % 2:
        nodes.append(None)
    n = len(nodes)
    matchings = []
    for _ in range(n - 1):
        pairs = []
        for i in range(n // 2):
            x, y = nodes[i], nodes[n - 1 - i]
            if x is not None and y is not None:
                pairs.append((x, y))
        matchings.append(tuple(pairs))
        nodes = [nodes[0], nodes[-1]] + nodes[1:-1]
    return tuple(matchings)


def build_pair_requests(
    trace: Sequence[ShotArrival],
    registry: CompatibilityRegistry,
    round_index: Optional[Mapping[str, int]] = None,
) -> Tuple[PairRequest, ...]:
    """Build the deterministic round-robin request workload for ``trace``.

    Args:
        trace: The shots to be scheduled.
        registry: Compatibility registry; non-``compatible`` pairs yield no request.
        round_index: ``shot_id -> round`` mapping (default: occurrence index within
            ``trace``). Must cover every shot in ``trace``.

    Matchings are taken over the circuits present in ``trace``.

    Request ids are ``req-r<round>-<position in matching>``; requests are ordered by
    round, then matching position.
    """
    rounds = dict(round_index) if round_index is not None else shot_round_indices(trace)
    shot_at: Dict[Tuple[str, int], ShotArrival] = {}
    for shot in trace:
        if shot.shot_id not in rounds:
            raise SimulationError(f"No round index for shot {shot.shot_id!r}")
        key = (shot.circuit, rounds[shot.shot_id])
        if key in shot_at:
            raise SimulationError(f"Two shots of circuit {shot.circuit!r} share round {key[1]}")
        shot_at[key] = shot

    matchings = round_robin_matchings(sorted({shot.circuit for shot in trace}))
    if not matchings:
        return ()
    requests = []
    for r in sorted({r for _, r in shot_at}):
        for position, (a, b) in enumerate(matchings[r % len(matchings)]):
            shot_a, shot_b = shot_at.get((a, r)), shot_at.get((b, r))
            if shot_a is None or shot_b is None or not circuits_compatible(registry, a, b):
                continue
            requests.append(make_request(f"req-r{r:06d}-{position:02d}", shot_a, shot_b))
    return tuple(requests)


def build_slot_schedule(
    trace: Sequence[ShotArrival], slots_per_shot: float = DEFAULT_SLOTS_PER_SHOT
) -> SlotSchedule:
    """Slot grid spanning the trace: ``ceil(slots_per_shot * N)`` slots.

    Slot 0 starts at the first arrival and the last slot starts at the last arrival,
    so the grid covers the trace's wall-clock span evenly. With the default of one
    slot per source shot, no policy can execute more shots than the trace contains.
    """
    if not (math.isfinite(slots_per_shot) and slots_per_shot > 0.0):
        raise SimulationError(f"slots_per_shot must be positive and finite, got {slots_per_shot}")
    n = len(trace)
    if n == 0:
        return SlotSchedule(start_time=0.0, slot_seconds=1.0, total_slots=0)
    total_slots = max(1, math.ceil(slots_per_shot * n))
    start = trace[0].arrival_time
    span = trace[-1].arrival_time - start
    slot_seconds = span / (total_slots - 1) if total_slots > 1 and span > 0.0 else 1.0
    return SlotSchedule(start_time=start, slot_seconds=slot_seconds, total_slots=total_slots)
