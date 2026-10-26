"""Statistical analysis package for quantum drift characterization."""

from quantum_drift_scheduler.analysis.holdout import SplitError, TraceSplit, split_trace
from quantum_drift_scheduler.analysis.priorities import (
    LeakageError,
    PairPriority,
    PriorityError,
    PriorityTable,
    learn_pair_priorities,
    load_priority_table,
    save_priority_table,
)
from quantum_drift_scheduler.analysis.windowing import (
    PairDriftDiagnostics,
    WindowedCircuitStats,
    analyze_pairwise_drift,
    compute_windowed_stats,
    compute_wilson_interval,
)

__all__ = [
    "SplitError",
    "TraceSplit",
    "split_trace",
    "LeakageError",
    "PairPriority",
    "PriorityError",
    "PriorityTable",
    "learn_pair_priorities",
    "load_priority_table",
    "save_priority_table",
    "WindowedCircuitStats",
    "PairDriftDiagnostics",
    "compute_wilson_interval",
    "compute_windowed_stats",
    "analyze_pairwise_drift",
]

