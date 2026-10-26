"""Pairing policies for the recorded-trace queue simulator.

A policy answers one question: given the pending queue, which two compatible shots
should be completed next (or none)? The simulator calls ``select_pair`` repeatedly
at each arrival timestamp until it returns ``None``.

Extension point for a future drift-aware policy: implement ``PairingPolicy`` (a
``name`` attribute and ``select_pair``) and pass an instance to ``run_simulation``.
Policies must be deterministic and must only return pending, distinct shots whose
circuit pair is ``compatible`` in the registry; the simulator verifies this and
raises ``SimulationError`` otherwise. Compatibility is a registry property; a
drift-aware policy may use empirical diagnostics for priority only, never to
override compatibility.
"""

from typing import Optional, Protocol, Tuple, runtime_checkable

from quantum_drift_scheduler.data.compatibility import CompatibilityRegistry, UnknownPairError
from quantum_drift_scheduler.scheduling.models import PendingQueue, ShotArrival

ShotPair = Tuple[ShotArrival, ShotArrival]

ARRIVAL_ORDER = "arrival-order"
MINIMUM_RECORDED_GAP = "minimum-recorded-gap"
POLICY_NAMES: Tuple[str, ...] = (ARRIVAL_ORDER, MINIMUM_RECORDED_GAP)


@runtime_checkable
class PairingPolicy(Protocol):
    """Interface every pairing policy implements."""

    name: str

    def select_pair(
        self, pending: PendingQueue, registry: CompatibilityRegistry, now: float
    ) -> Optional[ShotPair]:
        """Return the next two shots to pair, or ``None`` if no pair should complete now.

        Args:
            pending: Pending, unconsumed shots (read-only for the policy).
            registry: Compatibility registry.
            now: Current simulator time (the current arrival timestamp).
        """
        ...


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


def _ordered(first: ShotArrival, second: ShotArrival) -> ShotPair:
    return (first, second) if first.sort_key <= second.sort_key else (second, first)


class ArrivalOrderPolicy:
    """FIFO baseline.

    Takes the oldest pending shot that has any compatible pending counterpart and
    pairs it with the oldest compatible pending shot. Shots with no counterpart stay
    pending. Because compatibility depends only on circuits, this is evaluated on
    each circuit's oldest pending shot.
    """

    name = ARRIVAL_ORDER

    def select_pair(
        self, pending: PendingQueue, registry: CompatibilityRegistry, now: float
    ) -> Optional[ShotPair]:
        circuits = pending.circuits()
        best: Optional[ShotPair] = None
        for circuit in circuits:
            head = pending.head(circuit)
            partner: Optional[ShotArrival] = None
            for other in circuits:
                if not circuits_compatible(registry, circuit, other):
                    continue
                candidate = pending.head(other)
                if partner is None or candidate.sort_key < partner.sort_key:
                    partner = candidate
            if partner is None:
                continue
            if best is None or head.sort_key < best[0].sort_key:
                best = (head, partner)
        return best


class MinimumRecordedGapPolicy:
    """Close-together baseline on the recorded trace.

    Picks the compatible pending pair with the smallest ``abs`` difference of recorded
    arrival timestamps. Ties break by earlier maximum arrival time, earlier minimum
    arrival time, source indices, then shot ids. It uses recorded timestamps only to
    assess pairing quality on the recorded trace; it does not predict what hardware
    would have measured at a different time.
    """

    name = MINIMUM_RECORDED_GAP

    def select_pair(
        self, pending: PendingQueue, registry: CompatibilityRegistry, now: float
    ) -> Optional[ShotPair]:
        circuits = pending.circuits()
        best_key = None
        best_pair: Optional[ShotPair] = None
        for i, circuit_a in enumerate(circuits):
            for circuit_b in circuits[i + 1:]:
                if not circuits_compatible(registry, circuit_a, circuit_b):
                    continue
                small, large = sorted(
                    (circuit_a, circuit_b), key=lambda c: len(pending.shots_for(c))
                )
                for shot in pending.shots_for(small):
                    for other in pending.nearest(large, shot.arrival_time):
                        first, second = _ordered(shot, other)
                        key = (
                            abs(first.arrival_time - second.arrival_time),
                            max(first.arrival_time, second.arrival_time),
                            min(first.arrival_time, second.arrival_time),
                            first.source_index,
                            second.source_index,
                            first.shot_id,
                            second.shot_id,
                        )
                        if best_key is None or key < best_key:
                            best_key = key
                            best_pair = (first, second)
        return best_pair


def make_policy(name: str) -> PairingPolicy:
    """Build a built-in policy from its CLI name."""
    if name == ARRIVAL_ORDER:
        return ArrivalOrderPolicy()
    if name == MINIMUM_RECORDED_GAP:
        return MinimumRecordedGapPolicy()
    raise ValueError(f"Unknown policy {name!r}; expected one of: {', '.join(POLICY_NAMES)}")
