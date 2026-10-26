"""Scheduling policies for the slot-based simulator.

Two dispatch units exist, matching the required scheduler semantics:

* ``"shot"`` policies (``arrival-order``) execute individual shots, one per service
  slot, in the order given by ``shot_key`` (lowest first).
* ``"pair"`` policies (``consecutive-pair``, ``drift-aware``) dispatch a ready
  comparison request as a two-shot atomic block: the first circuit executes in one
  slot and the second in the immediately following slot, with nothing in between. If
  no request is ready, the slot idles (strict idling): a pair policy never executes a
  lone shot.

A pair policy defines ``request_key(request)``: the simulator always dispatches the
ready, still-unexecuted request with the lowest key. Keys must be deterministic,
immutable and totally ordered (end with ``request_id``). This is the extension point
for new pair policies; compatibility is enforced by the workload and the simulator,
never by a policy.
"""

from typing import Protocol, Tuple, runtime_checkable

from quantum_drift_scheduler.scheduling.models import PairRequest, ShotArrival

ARRIVAL_ORDER = "arrival-order"
CONSECUTIVE_PAIR = "consecutive-pair"
DRIFT_AWARE = "drift-aware"
BASELINE_POLICY_NAMES: Tuple[str, ...] = (ARRIVAL_ORDER, CONSECUTIVE_PAIR)
POLICY_NAMES: Tuple[str, ...] = (ARRIVAL_ORDER, CONSECUTIVE_PAIR, DRIFT_AWARE)


@runtime_checkable
class ShotPolicy(Protocol):
    """Individual-shot dispatch policy."""

    name: str
    dispatch_unit: str  # "shot"

    def shot_key(self, shot: ShotArrival) -> tuple:
        """Sort key among arrived, unexecuted shots; the lowest key executes next."""
        ...


@runtime_checkable
class PairPolicy(Protocol):
    """Atomic two-slot pair dispatch policy."""

    name: str
    dispatch_unit: str  # "pair"

    def request_key(self, request: PairRequest) -> tuple:
        """Sort key among ready requests; the lowest key dispatches next."""
        ...


def request_order_key(request: PairRequest) -> tuple:
    """Deterministic request order shared by the pair policies.

    1. earlier pair-request arrival time (when both shots have arrived);
    2. lower sorted wait-length tuple ``(min(k_a, k_b), max(k_a, k_b))``;
    3. earlier source indices ``(min, max)``;
    4. stable request id.
    """
    ks = sorted((request.wait_length_first, request.wait_length_second))
    sources = sorted((request.first.source_index, request.second.source_index))
    return (request.ready_time, (ks[0], ks[1]), (sources[0], sources[1]), request.request_id)


class ArrivalOrderPolicy:
    """Individual-shot FIFO baseline.

    Executes the oldest arrived, unexecuted shot in every slot (ties: source index,
    then shot id) and never idles while a shot is pending. A workload request completes
    as a comparison when both of its shots have executed, so the two shots of a
    comparison are generally not adjacent in execution order.
    """

    name = ARRIVAL_ORDER
    dispatch_unit = "shot"

    def shot_key(self, shot: ShotArrival) -> tuple:
        return shot.sort_key


class ConsecutivePairPolicy:
    """Pair-aware baseline without drift information.

    Dispatches the ready request with the earliest pair-request arrival time (see
    ``request_order_key``) as an atomic consecutive two-slot block, lower wait length
    first.
    """

    name = CONSECUTIVE_PAIR
    dispatch_unit = "pair"

    def request_key(self, request: PairRequest) -> tuple:
        return request_order_key(request)


def make_baseline_policy(name: str):
    """Build a baseline policy from its CLI name."""
    if name == ARRIVAL_ORDER:
        return ArrivalOrderPolicy()
    if name == CONSECUTIVE_PAIR:
        return ConsecutivePairPolicy()
    raise ValueError(f"Unknown policy {name!r}; expected one of: {', '.join(BASELINE_POLICY_NAMES)}")
