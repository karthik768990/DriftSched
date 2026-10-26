"""Held-out replay scheduling metrics.

These metrics describe slot usage, queue behaviour and the separation of the paired
shots on a recorded trace. They make no claim about counterfactual hardware outcomes,
physical comparison error, uncertainty reduction or hardware performance. The observed
outcome difference is a recorded historical value, not a physical comparison error.
"""

import math
from collections import Counter
from typing import List, Optional, Sequence

from quantum_drift_scheduler.scheduling.models import (
    SimulationMetrics,
    SimulationResult,
    SummaryStats,
)


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


def summarize(values: Sequence[float]) -> SummaryStats:
    """Mean, median, 95th percentile and maximum; all ``None`` when empty."""
    return SummaryStats(
        mean=_mean(values),
        median=percentile(values, 50.0),
        p95=percentile(values, 95.0),
        maximum=float(max(values)) if values else None,
    )


def compute_metrics(result: SimulationResult) -> SimulationMetrics:
    """Compute held-out replay scheduling metrics from a simulation result."""
    comparisons = result.comparisons
    completed = len(comparisons)
    possible = result.source_shots // 2
    diffs = [float(c.observed_difference) for c in comparisons]
    pair_counter = Counter(c.pair_key for c in comparisons)

    return SimulationMetrics(
        source_shots=result.source_shots,
        total_slots=result.schedule.total_slots,
        executed_shots=len(result.executed),
        unexecuted_shots=len(result.unexecuted_shots),
        idle_slots=result.idle_slots,
        completed_comparisons=completed,
        consumed_shots=2 * completed,
        executed_unpaired_shots=len(result.executed) - 2 * completed,
        completion_rate=(completed / possible) if possible > 0 else None,
        slot_separation=summarize([float(c.slot_separation) for c in comparisons]),
        time_separation=summarize([c.time_separation for c in comparisons]),
        waiting_time=summarize([e.waiting_time for e in result.executed]),
        mean_absolute_observed_difference=_mean([abs(d) for d in diffs]),
        mean_signed_observed_difference=_mean(diffs),
        pair_counts=tuple((a, b, n) for (a, b), n in sorted(pair_counter.items())),
    )


def format_number(value: Optional[float], digits: int = 4) -> str:
    """Fixed-point text, ``n/a`` for ``None``."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_summary(stats: SummaryStats, digits: int = 4) -> str:
    return (
        f"mean={format_number(stats.mean, digits)}  median={format_number(stats.median, digits)}  "
        f"p95={format_number(stats.p95, digits)}  max={format_number(stats.maximum, digits)}"
    )


def format_metrics(metrics: SimulationMetrics) -> List[str]:
    """Human-readable metric lines shared by the CLIs."""
    lines = [
        f"Source shots:            {metrics.source_shots}",
        f"Total logical slots:     {metrics.total_slots}",
        f"Executed shots:          {metrics.executed_shots}",
        f"Unexecuted shots:        {metrics.unexecuted_shots}",
        f"Idle slots:              {metrics.idle_slots}",
        f"Completed comparisons:   {metrics.completed_comparisons}",
        f"Executed, unpaired:      {metrics.executed_unpaired_shots}",
        f"Completion rate:         {format_number(metrics.completion_rate, 6)}",
        "Execution-slot separation of paired shots (slots):",
        f"  {format_summary(metrics.slot_separation)}",
        "Execution-time separation of paired shots (s):",
        f"  {format_summary(metrics.time_separation)}",
        "Per-shot queue waiting time (s):",
        f"  {format_summary(metrics.waiting_time)}",
        "Observed historical outcome difference (outcome_a - outcome_b, a = smaller wait length):",
        f"  mean absolute={format_number(metrics.mean_absolute_observed_difference, 6)}  "
        f"signed mean={format_number(metrics.mean_signed_observed_difference, 6)}",
        f"Completed comparisons by circuit pair ({len(metrics.pair_counts)} pairs):",
    ]
    lines += [f"  {a} | {b}: {n}" for a, b, n in metrics.pair_counts]
    return lines
