"""Shared synthetic-trace helpers for scheduling tests."""

from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityPair,
    CompatibilityRegistry,
    CompatibilityStatus,
    ReviewerDecision,
)
from quantum_drift_scheduler.scheduling import ShotArrival

CircuitRef = Union[int, str]


def label(circuit: CircuitRef) -> str:
    """Ramsey label for an integer wait length k; strings pass through unchanged."""
    if isinstance(circuit, str):
        return circuit
    return "Gx(Gw)Gy" if circuit == 1 else f"Gx(Gw)^{circuit}Gy"


def make_registry(
    circuits: Sequence[CircuitRef] = ("A", "B", "C"),
    overrides: Optional[Dict[Tuple[CircuitRef, CircuitRef], CompatibilityStatus]] = None,
    not_comparable: Iterable[Tuple[CircuitRef, CircuitRef]] = (),
) -> CompatibilityRegistry:
    """Registry over all unordered pairs of ``circuits``.

    Every pair is ``reviewer_decision = Comparable`` except those listed in ``not_comparable``.
    ``overrides`` set only the *informational* ``compatibility_status`` column, which must
    never change eligibility.
    """
    names = [label(c) for c in circuits]
    overrides = {(label(a), label(b)): v for (a, b), v in (overrides or {}).items()}
    excluded = {frozenset((label(a), label(b))) for a, b in not_comparable}
    pairs = []
    for i, (a, b) in enumerate(combinations(names, 2), start=1):
        status = overrides.get((a, b), overrides.get((b, a), CompatibilityStatus.COMPATIBLE))
        decision = ReviewerDecision.NOT_COMPARABLE if frozenset((a, b)) in excluded else ReviewerDecision.COMPARABLE
        pairs.append(
            CompatibilityPair(
                pair_id=i,
                circuit_a=a,
                circuit_b=b,
                status=status,
                evidence_found=True,
                evidence_summary="synthetic",
                missing_information="",
                rationale="synthetic",
                source_reference="tests",
                reviewer_decision=decision,
            )
        )
    return CompatibilityRegistry(pairs)


def ramsey_registry(ks: Sequence[int] = (1, 2, 4, 8)) -> CompatibilityRegistry:
    """All-compatible registry over Ramsey circuits with wait lengths ``ks``."""
    return make_registry(ks)


def make_trace(
    rows: Iterable[Tuple[float, CircuitRef, int]], source: str = "synthetic"
) -> Tuple[ShotArrival, ...]:
    """Build a trace from ``(arrival_time, circuit_or_k, outcome)`` rows in the given order."""
    return tuple(
        ShotArrival(
            shot_id=f"s{i:04d}",
            arrival_time=t,
            circuit=label(c),
            outcome=o,
            repetitions=1,
            source_index=i,
            input_source=source,
        )
        for i, (t, c, o) in enumerate(rows)
    )
