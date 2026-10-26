"""Shared synthetic-trace helpers for scheduling tests."""

from itertools import combinations
from typing import Dict, Iterable, Optional, Sequence, Tuple

from quantum_drift_scheduler.data.compatibility import (
    CompatibilityPair,
    CompatibilityRegistry,
    CompatibilityStatus,
)
from quantum_drift_scheduler.scheduling import ShotArrival


def make_registry(
    circuits: Sequence[str] = ("A", "B", "C"),
    overrides: Optional[Dict[Tuple[str, str], CompatibilityStatus]] = None,
) -> CompatibilityRegistry:
    """Registry over all unordered pairs of ``circuits``; ``overrides`` set other statuses."""
    overrides = overrides or {}
    pairs = []
    for i, (a, b) in enumerate(combinations(circuits, 2), start=1):
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


def make_trace(rows: Iterable[Tuple[float, str, int]], source: str = "synthetic") -> Tuple[ShotArrival, ...]:
    """Build a trace from ``(arrival_time, circuit, outcome)`` rows in the given order."""
    return tuple(
        ShotArrival(
            shot_id=f"s{i:04d}",
            arrival_time=t,
            circuit=c,
            outcome=o,
            repetitions=1,
            source_index=i,
            input_source=source,
        )
        for i, (t, c, o) in enumerate(rows)
    )
