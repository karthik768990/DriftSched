"""Shared synthetic-trace helpers for scheduling tests."""

from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityPair,
    CompatibilityRegistry,
    CompatibilityStatus,
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
) -> CompatibilityRegistry:
    """Registry over all unordered pairs of ``circuits``; ``overrides`` set other statuses."""
    names = [label(c) for c in circuits]
    overrides = {(label(a), label(b)): v for (a, b), v in (overrides or {}).items()}
    pairs = []
    for i, (a, b) in enumerate(combinations(names, 2), start=1):
        status = overrides.get((a, b), overrides.get((b, a), CompatibilityStatus.COMPATIBLE))
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
                reviewer_decision="Comparable",
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
