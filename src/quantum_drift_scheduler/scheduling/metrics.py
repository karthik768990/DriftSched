"""Summary metrics for recorded-trace simulation results.

These metrics describe pairing quality and queue behaviour on the recorded trace
only. They make no claim about comparison-error reduction, uncertainty reduction or
hardware performance.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

from quantum_drift_scheduler.scheduling.models import SimulationMetrics, SimulationResult


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Percentile with linear interpolation between order statistics (q in [0, 100]).

    Returns ``None`` for an empty sequence.
    """
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be within [0, 100], got {q}")
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _mean(values: Sequence[float]) -> Optional[float]:
    return math.fsum(values) / len(values) if values else None


def compute_metrics(result: SimulationResult) -> SimulationMetrics:
    """Compute summary metrics from a simulation result."""
    comparisons = result.comparisons
    completed = len(comparisons)
    max_possible = result.input_shots // 2

    separations = [c.timestamp_separation for c in comparisons]
    waits: List[float] = []
    for c in comparisons:
        waits.append(c.waiting_time_a)
        waits.append(c.waiting_time_b)
    diffs = [float(c.observed_difference) for c in comparisons]

    pair_counter: Counter = Counter(c.pair_key for c in comparisons)
    pair_counts = tuple((a, b, n) for (a, b), n in sorted(pair_counter.items()))
    unmatched_counter: Dict[str, int] = Counter(s.circuit for s in result.unmatched_shots)
    unmatched_by_circuit = tuple(sorted(unmatched_counter.items()))

    return SimulationMetrics(
        total_input_shots=result.input_shots,
        completed_comparisons=completed,
        consumed_shots=2 * completed,
        unmatched_shots=len(result.unmatched_shots),
        completion_rate=(completed / max_possible) if max_possible > 0 else None,
        separation_mean=_mean(separations),
        separation_median=percentile(separations, 50.0),
        separation_max=max(separations) if separations else None,
        separation_p95=percentile(separations, 95.0),
        waiting_mean=_mean(waits),
        waiting_median=percentile(waits, 50.0),
        waiting_max=max(waits) if waits else None,
        waiting_p95=percentile(waits, 95.0),
        mean_absolute_observed_difference=_mean([abs(d) for d in diffs]),
        mean_signed_observed_difference=_mean(diffs),
        pair_counts=pair_counts,
        unmatched_by_circuit=unmatched_by_circuit,
    )
